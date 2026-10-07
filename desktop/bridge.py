"""Narrow native bridge. It never executes engine commands."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from desktop import window_chrome


class DesktopBridge:
    def __init__(self):
        self._window = None
        self._maximized = False
        self._page_loaded = False

    def _attach(self, window):
        self._window = window
        window.events.before_show += self._prepare_frame
        window.events.before_load += self._before_load
        window.events.loaded += self._loaded
        window.events.maximized += self._on_maximized
        window.events.restored += self._on_restored
        window.events.closed += self._on_closed

    def _prepare_frame(self):
        window_chrome.fit_work_area(self._window)

    def _before_load(self):
        self._page_loaded = False

    def _loaded(self):
        self._page_loaded = True
        self._publish_window_state()

    def _on_closed(self):
        self._page_loaded = False

    def _on_maximized(self):
        self._maximized = True
        self._publish_window_state()

    def _on_restored(self):
        self._maximized = False
        self._publish_window_state()

    def _require_window(self):
        if self._window is None:
            raise RuntimeError("Desktop window is not attached")
        return self._window

    def get_window_state(self):
        window = self._require_window()
        return {
            "maximized": window_chrome.is_maximized(window, self._maximized),
            "resizable": os.name == "nt",
        }

    def _publish_window_state(self):
        if not self._page_loaded:
            return
        payload = json.dumps(self.get_window_state())
        try:
            self._window.evaluate_js(
                "window.dispatchEvent(new CustomEvent('lp-window-state', {detail: " + payload + "}));"
            )
        except Exception:
            # The page can disappear between the event and evaluate_js during close/reload.
            logging.getLogger(__name__).debug("Window state notification skipped", exc_info=True)

    def window_action(self, action):
        if not isinstance(action, str) or action not in ("minimize", "toggle_maximize", "close"):
            raise ValueError("Unsupported window action")
        window = self._require_window()
        if action == "minimize":
            window.minimize()
        elif action == "toggle_maximize":
            if self.get_window_state()["maximized"]:
                window.restore()
            else:
                window_chrome.maximize(window)
        else:
            # Never stop the API, supervisor, or training processes here.
            window.destroy()
            return None
        return self.get_window_state()

    def begin_window_resize(self, edge):
        window_chrome.begin_resize(self._require_window(), edge)

    def pick_path(self, kind, title, file_types=None):
        if kind not in ("file", "directory") or not self._window: raise ValueError("Unsupported dialog")
        if not isinstance(title, str) or len(title) > 200: raise ValueError("Invalid title")
        import webview
        dialog = webview.FileDialog.FOLDER if kind == "directory" else webview.FileDialog.OPEN
        types = tuple(t for t in (file_types or []) if isinstance(t, str) and len(t) < 200)
        paths = self._window.create_file_dialog(dialog, allow_multiple=False, file_types=types or ("All files (*.*)",))
        return str(paths[0]) if paths else None

    def open_in_explorer(self, value):
        if not isinstance(value, str) or "\x00" in value or len(value) > 32767: raise ValueError("Invalid path")
        path = Path(value)
        if not path.is_absolute() or not path.exists(): raise ValueError("Path does not exist")
        path = path.resolve()
        if os.name == "nt":
            argv = ["explorer.exe", str(path)] if path.is_dir() else ["explorer.exe", "/select,", str(path)]
        elif sys.platform == "darwin": argv = ["open", "-R", str(path)]
        else: argv = ["xdg-open", str(path if path.is_dir() else path.parent)]
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
