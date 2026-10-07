from pathlib import Path
from desktop.bridge import DesktopBridge


def open_window(url):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError('桌面依赖缺失：便携版请重新生成 python_runtime；开发环境请用 .venv\\Scripts\\python.exe -m pip install -e ".[desktop]"') from exc
    bridge = DesktopBridge()
    window = webview.create_window(
        "LP LoRA Trainer", url, width=1440, height=960, min_size=(1000, 700),
        js_api=bridge, frameless=True, easy_drag=False, resizable=True,
        background_color="#262626",
    )
    bridge._attach(window)
    # A closed WebView does not own the backend or training supervisor lifetime.
    import os
    icon = Path(__file__).resolve().parents[1] / "assets" / "logo.ico"
    webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False,
                  icon=str(icon) if icon.is_file() else None)
