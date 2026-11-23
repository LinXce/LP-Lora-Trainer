"""Deliver Ctrl+C to a training stage in its own hidden Windows console."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path


def stage_process_options():
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    # Ctrl+C broadcasts within a console. Give this stage and its children a
    # dedicated console so a stop cannot reach the desktop or another engine.
    return {"creationflags": subprocess.CREATE_NEW_CONSOLE, "startupinfo": startup}


def send_interrupt(process):
    if process.poll() is not None:
        return True
    try:
        # Only the short-lived helper attaches. Switching the supervisor's own
        # console would affect its other threads, handlers and standard streams.
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), str(process.pid)],
            creationflags=subprocess.DETACHED_PROCESS, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return process.poll() is not None
    return result.returncode == 0 or process.poll() is not None


def _signal_console(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.AttachConsole(pid):
        return False
    try:
        # Null handler + TRUE makes this helper ignore its own broadcast.
        if not kernel.SetConsoleCtrlHandler(None, True):
            return False
        return bool(kernel.GenerateConsoleCtrlEvent(0, 0))
    finally:
        kernel.FreeConsole()


if __name__ == "__main__":
    if os.name != "nt" or len(sys.argv) != 2:
        raise SystemExit(2)
    raise SystemExit(0 if _signal_console(int(sys.argv[1])) else 1)
