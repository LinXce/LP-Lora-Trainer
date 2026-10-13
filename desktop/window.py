import ctypes
import os
import threading
from pathlib import Path
from desktop.bridge import DesktopBridge


# A frameless pywebview window is a WinForms form with FormBorderStyle.None, and
# that also removes the sizing border Windows resizes from: `resizable=True` has
# no effect there. Adding WS_THICKFRAME back keeps our own chrome while restoring
# edge/corner drag-resize, Aero snap and DWM rounded corners.
GWL_STYLE = -16
WS_THICKFRAME = 0x00040000
WS_MAXIMIZEBOX = 0x00010000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
PAGE_BACKGROUND = 0x262626


def enable_window_resize(title, user32=None, dwmapi=None):
    """Give a frameless window its sizing border back. Returns True when applied.

    The DWM border and caption colours are set to the page colour (#262626).
    Windows builds without these attributes ignore them. ``user32``/``dwmapi``
    are injectable so the calls are testable without a window.
    A missing window is reported, never raised: it is a cosmetic detail.
    """
    injected = user32 is not None
    api = user32 if injected else ctypes.windll.user32
    dwm = dwmapi if dwmapi is not None else ctypes.windll.dwmapi
    if not injected:
        dwm.DwmSetWindowAttribute.restype = ctypes.c_long
        dwm.DwmSetWindowAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                              ctypes.c_void_p, ctypes.c_uint32]
    hwnd = api.FindWindowW(None, title)
    if not hwnd:
        return False
    style = api.GetWindowLongW(hwnd, GWL_STYLE)
    api.SetWindowLongW(hwnd, GWL_STYLE, style | WS_THICKFRAME | WS_MAXIMIZEBOX)
    colour = ctypes.c_uint32(PAGE_BACKGROUND)
    for attribute in (DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR):
        dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(colour), ctypes.sizeof(colour))
    api.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
    return True


# Shown the instant the window opens, while the local backend is still starting.
# Mirrors the boot screen in frontend/index.html so the hand-over is seamless.
SPLASH_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="color-scheme" content="dark"><style>
html, body { margin: 0; height: 100%; background: #262626; overflow: hidden; user-select: none; }
body { display: grid; place-items: center; color: #f2f2f2;
  font-family: 'Segoe UI Variable Text', 'Segoe UI', 'Microsoft YaHei UI', system-ui, sans-serif; }
.boot { display: flex; flex-direction: column; align-items: center; gap: 18px; animation: in .5s cubic-bezier(.2,.7,.2,1) both; }
.boot__logo { width: 64px; height: 64px; animation: breathe 2.4s ease-in-out infinite; }
.boot__name { font-size: 15px; font-weight: 600; letter-spacing: .02em; }
.boot__bar { position: relative; width: 180px; height: 3px; border-radius: 3px; background: rgba(255,255,255,.1); overflow: hidden; }
.boot__bar::after { content: ''; position: absolute; inset: 0; width: 40%; border-radius: 3px; background: #ececec;
  animation: slide 1.2s cubic-bezier(.65,0,.35,1) infinite; }
.boot__text { font-size: 12px; color: #959595; }
@keyframes in { from { opacity: 0; transform: translateY(8px); } }
@keyframes breathe { 50% { transform: scale(.94); opacity: .8; } }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(250%); } }
</style></head>
<body class="pywebview-drag-region"><div class="boot">
<svg class="boot__logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><circle cx="24" cy="24" r="24" fill="#555"/><path d="M10 14h4v16h8v4H10V14Zm16 0h7a6 6 0 0 1 0 12h-3v8h-4V14Zm4 4v4h3a2 2 0 0 0 0-4h-3Z" fill="#eee"/></svg>
<div class="boot__name">LP LoRA Trainer</div>
<div class="boot__bar"></div>
<div class="boot__text">正在启动本地服务…</div>
</div></body></html>"""


def _load_winforms_backend():
    """Import the exact module pywebview needs; it is what fails without .NET."""
    import webview.platforms.winforms  # noqa: F401


def desktop_bridge_error(checker=None):
    """Return None when the WebView2/.NET bridge can initialize, else the reason.

    Checked before the backend is started so a broken desktop runtime can never
    leave an unreachable backend/supervisor pair behind. ``checker`` is only for
    tests; production always exercises the real pywebview import.
    """
    if os.name != "nt":
        return None
    probe = checker or _load_winforms_backend
    try:
        probe()
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        return f"{type(exc).__name__}: {exc}"
    return None


def open_window(url, prepare=None, bounds_file=None, on_close=None):
    """Show the window immediately; navigate to ``url`` once ``prepare()`` returns.

    ``prepare`` (e.g. starting the backend) runs off the UI thread behind the splash.
    If it raises, the window closes and the error is re-raised to the caller.
    """
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError('桌面依赖缺失：便携版请重新生成 python_runtime；开发环境请用 .venv\\Scripts\\python.exe -m pip install -e ".[desktop]"') from exc
    bridge = DesktopBridge(bounds_file)
    options = dict(width=1440, height=960, min_size=(1000, 700),
                   js_api=bridge, frameless=True, easy_drag=False, resizable=True,
                   background_color="#262626")
    if prepare is None:
        window = webview.create_window("LP LoRA Trainer", url, **options)
    else:
        window = webview.create_window("LP LoRA Trainer", html=SPLASH_HTML, **options)
    bridge._attach(window)
    failure = []
    boot_thread = None
    closed = threading.Event()
    bootstrap_lock = threading.Lock()

    def closing():
        if failure or on_close is None:
            return True
        return on_close(window)

    window.events.closing += closing
    window.events.closed += closed.set
    resize_applied = threading.Event()

    def apply_resize():
        # `shown` can arrive more than once; the window style is set once.
        if resize_applied.is_set():
            return
        resize_applied.set()
        if os.name == "nt":
            enable_window_resize("LP LoRA Trainer")

    window.events.shown += apply_resize

    def boot():
        try:
            prepare()
            if not closed.is_set():
                window.load_url(url)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the launcher thread
            failure.append(exc)
            window.destroy()
            return

    if prepare is not None:
        # `shown` handlers already run on a worker thread; start once the splash is visible.
        def on_shown():
            nonlocal boot_thread
            # shown callbacks can arrive while webview.start is unwinding.
            # Starting and claiming the thread must be atomic with final cleanup.
            with bootstrap_lock:
                if closed.is_set() or boot_thread is not None:
                    return
                boot_thread = threading.Thread(
                    target=boot, name="lp-backend-bootstrap", daemon=True,
                )
                boot_thread.start()

        window.events.shown += on_shown
    icon = Path(__file__).resolve().parents[1] / "assets" / "logo.ico"
    try:
        webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False,
                      icon=str(icon) if icon.is_file() else None)
    finally:
        # Never abandon the bootstrap worker on close or a WebView failure.
        # The splash disappears immediately; bounded backend startup finishes
        # before launcher cleanup and before another instance acquires the lock.
        with bootstrap_lock:
            closed.set()
            thread = boot_thread
        if thread is not None:
            thread.join()
    if failure:
        raise failure[0]
