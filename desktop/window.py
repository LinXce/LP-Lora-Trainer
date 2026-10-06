from desktop.bridge import DesktopBridge


def open_window(url):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError('请先安装桌面依赖：python -m pip install -e ".[desktop]"') from exc
    bridge = DesktopBridge()
    window = webview.create_window("LP LoRA Trainer", url, width=1440, height=960, min_size=(1000, 700), js_api=bridge)
    bridge._attach(window)
    # A closed WebView does not own the backend or training supervisor lifetime.
    import os
    webview.start(gui="edgechromium" if os.name == "nt" else None, debug=False)
