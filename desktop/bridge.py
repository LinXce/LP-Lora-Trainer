"""Narrow native bridge. It never executes engine commands."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from desktop import window_chrome


class DesktopBridge:
    def __init__(self, bounds_file=None):
        self._window = None
        self._maximized = False
        self._page_loaded = False
        self._bounds_file = Path(bounds_file) if bounds_file else None

    def _attach(self, window):
        self._window = window
        window.events.before_show += self._prepare_frame
        window.events.before_load += self._before_load
        window.events.loaded += self._loaded
        window.events.maximized += self._on_maximized
        window.events.restored += self._on_restored
        window.events.closing += self._on_closing
        window.events.closed += self._on_closed

    def _prepare_frame(self):
        window_chrome.apply_bounds(self._window, self._load_bounds())
        window_chrome.fit_work_area(self._window)

    def _load_bounds(self):
        if self._bounds_file is None:
            return None
        try:
            bounds = json.loads(self._bounds_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return bounds if isinstance(bounds, dict) else None

    def _on_closing(self):
        # Runs synchronously on the UI thread while the native form still exists.
        if self._bounds_file is None:
            return None
        try:
            bounds = window_chrome.read_bounds(self._window)
            if bounds:
                self._bounds_file.parent.mkdir(parents=True, exist_ok=True)
                self._bounds_file.write_text(json.dumps(bounds), encoding="utf-8")
        except Exception:
            # Remembering the position must never block closing the window.
            logging.getLogger(__name__).debug("Window bounds not saved", exc_info=True)
        return None

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
            # The launcher closing hook owns graceful backend/supervisor shutdown.
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

    def save_text_file(self, suggested_name, text):
        """Save UTF-8 text only to a destination explicitly picked by the user."""
        if (not isinstance(suggested_name, str) or not suggested_name
                or len(suggested_name) > 180
                or any(c in suggested_name for c in '/\\:*?"<>|')
                or any(ord(c) < 32 for c in suggested_name)):
            raise ValueError("Invalid export filename")
        extension = Path(suggested_name).suffix.lower()
        if extension not in (".json", ".toml", ".yaml", ".yml", ".txt"):
            raise ValueError("Unsupported export format")
        if not isinstance(text, str) or len(text.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("Export is too large")
        import webview
        paths = self._require_window().create_file_dialog(
            webview.FileDialog.SAVE, save_filename=suggested_name,
            file_types=(f"Configuration (*{extension})", "All files (*.*)"),
        )
        if not paths:
            return None
        path = Path(paths[0])
        if not path.suffix:
            path = path.with_suffix(extension)
        try:
            path.write_text(text, encoding="utf-8")
        except PermissionError as exc:
            from desktop.permissions import MEDIUM_INTEGRITY, process_integrity_level
            try:
                level = process_integrity_level()
            except OSError:
                level = None
            if level is not None and level < MEDIUM_INTEGRITY:
                reason = "应用运行在受限权限下，请完整退出后通过更新后的 EXE 重新启动。"
            else:
                reason = "请检查目标文件是否只读、被占用，或该目录是否限制写入；也可重新选择保存位置。"
            raise PermissionError(f"配置未能保存到 {path}。{reason}") from exc
        return str(path)

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
