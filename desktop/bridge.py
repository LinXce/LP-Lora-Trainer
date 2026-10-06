"""Narrow native bridge. It never executes engine commands."""
import os
import subprocess
import sys
from pathlib import Path


class DesktopBridge:
    def __init__(self):
        self._window = None

    def _attach(self, window):
        self._window = window

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
