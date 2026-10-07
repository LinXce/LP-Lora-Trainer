"""Windows-only frame helpers; operate only on the attached application window."""
import ctypes
import os
from ctypes import wintypes


RESIZE_EDGES = {
    "w": 10, "e": 11, "n": 12, "nw": 13,
    "ne": 14, "s": 15, "sw": 16, "se": 17,
}


def _user32():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsZoomed.argtypes = [wintypes.HWND]
    user32.IsZoomed.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.ReleaseCapture.argtypes = []
    user32.ReleaseCapture.restype = wintypes.BOOL
    return user32


def is_maximized(window, fallback=False):
    if os.name == "nt" and window.native is not None:
        return bool(_user32().IsZoomed(window.native.Handle.ToInt64()))
    return fallback


def fit_work_area(window):
    """Called on the WinForms UI thread, including before the window is shown."""
    if os.name == "nt" and window.native is not None:
        from System.Windows.Forms import Screen
        window.native.MaximizedBounds = Screen.FromControl(window.native).WorkingArea


def maximize(window):
    if os.name == "nt" and window.native is not None:
        from System import Action
        from System.Windows.Forms import FormWindowState

        def apply():
            # Recompute for the current monitor, rather than covering its taskbar.
            fit_work_area(window)
            window.native.WindowState = FormWindowState.Maximized

        window.native.Invoke(Action(apply))
    else:
        window.maximize()


def begin_resize(window, edge):
    if not isinstance(edge, str) or edge not in RESIZE_EDGES:
        raise ValueError("Unsupported resize edge")
    if os.name != "nt" or window.native is None:
        raise RuntimeError("Native resize is unavailable")
    if is_maximized(window):
        return
    user32 = _user32()
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise ctypes.WinError(ctypes.get_last_error())
    position = (point.x & 0xFFFF) | ((point.y & 0xFFFF) << 16)
    user32.ReleaseCapture()
    # Let Windows own the resize loop. No mousemove polling or per-frame bridge calls.
    if not user32.PostMessageW(window.native.Handle.ToInt64(), 0x00A1, RESIZE_EDGES[edge], position):
        raise ctypes.WinError(ctypes.get_last_error())
