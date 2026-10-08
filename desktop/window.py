import threading
from pathlib import Path
from desktop.bridge import DesktopBridge


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
    import os
    if os.name != "nt":
        return None
    probe = checker or _load_winforms_backend
    try:
        probe()
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        return f"{type(exc).__name__}: {exc}"
    return None


def open_window(url, prepare=None, bounds_file=None):
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

    def boot():
        try:
            prepare()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the launcher thread
            failure.append(exc)
            window.destroy()
            return
        window.load_url(url)

    if prepare is not None:
        # `shown` handlers already run on a worker thread; start once the splash is visible.
        started = threading.Event()

        def on_shown():
            if not started.is_set():
                started.set()
                boot()

        window.events.shown += on_shown
    # A closed WebView does not own the backend or training supervisor lifetime.
    import os
    icon = Path(__file__).resolve().parents[1] / "assets" / "logo.ico"
    webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False,
                  icon=str(icon) if icon.is_file() else None)
    if failure:
        raise failure[0]
