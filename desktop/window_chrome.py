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


def read_bounds(window):
    """Physical-pixel restore bounds of the native window, or None.

    Uses RestoreBounds while maximized/minimized so reopening returns to the
    normal-size frame the user last arranged.
    """
    if os.name != "nt" or window.native is None:
        return None
    from System.Windows.Forms import FormWindowState
    form = window.native
    state = form.WindowState
    rect = form.Bounds if state == FormWindowState.Normal else form.RestoreBounds
    return {
        "x": int(rect.X), "y": int(rect.Y), "width": int(rect.Width), "height": int(rect.Height),
        "maximized": state == FormWindowState.Maximized,
    }


def apply_bounds(window, bounds):
    """Place the window before it is shown: saved bounds if still on a monitor, else centred.

    Runs on the UI thread from ``before_show``. Positioning is always explicit so the
    frameless form never falls back to an arbitrary system-chosen location.
    """
    if os.name != "nt" or window.native is None:
        return
    from System.Drawing import Rectangle
    from System.Windows.Forms import FormStartPosition, FormWindowState, Screen
    form = window.native
    form.StartPosition = FormStartPosition.Manual
    target = None
    if bounds:
        try:
            rect = Rectangle(int(bounds["x"]), int(bounds["y"]), int(bounds["width"]), int(bounds["height"]))
        except (KeyError, TypeError, ValueError):
            rect = None
        # Require a usable part of the frame (incl. the title bar) to land on a current monitor.
        if rect is not None and rect.Width >= form.MinimumSize.Width and rect.Height >= form.MinimumSize.Height:
            grip = Rectangle(rect.X, rect.Y, rect.Width, 40)
            if any(screen.WorkingArea.IntersectsWith(grip) for screen in Screen.AllScreens):
                target = rect
    if target is None:
        area = Screen.PrimaryScreen.WorkingArea
        width, height = min(form.Width, area.Width), min(form.Height, area.Height)
        target = Rectangle(area.X + (area.Width - width) // 2, area.Y + (area.Height - height) // 2, width, height)
    form.Bounds = target
    if bounds and bounds.get("maximized"):
        fit_work_area(window)
        form.WindowState = FormWindowState.Maximized


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
