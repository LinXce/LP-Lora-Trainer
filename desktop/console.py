"""Terminal + browser fallback used when the desktop WebView cannot start.

Enter requests an ownership-aware close. Losing the terminal leaves training alive.
"""
import argparse
import ctypes
import os
import signal
import sys
import time
import webbrowser
from pathlib import Path

from app.config import DEFAULT_API_PORT
from desktop.shutdown import close_surface

TITLE = "LP LoRA Trainer（终端 + 浏览器模式）"

# The fallback launcher cannot write a non-ASCII data root into its ASCII-only
# cmd script, so the root travels in the environment instead.
CONSOLE_DATA_ROOT_ENV = "LP_CONSOLE_DATA_ROOT"


def _open_console_streams():
    """Bind the new console's own devices and make sure its window is visible.

    The EXE host starts pythonw with a hidden window state, which can leave the
    console created here invisible; CONOUT$/CONIN$ also keep output working even
    when the inherited std handles are unusable.
    """
    kernel32, user32 = ctypes.windll.kernel32, ctypes.windll.user32
    handle = kernel32.GetConsoleWindow()
    if handle:
        user32.ShowWindow(handle, 5)          # SW_SHOW
        user32.SetForegroundWindow(handle)
        kernel32.SetConsoleTitleW(TITLE)
    for name, device, mode in (("stdout", "CONOUT$", "w"), ("stderr", "CONOUT$", "w"),
                               ("stdin", "CONIN$", "r")):
        try:
            stream = open(device, mode, encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            continue
        setattr(sys, name, stream)


def _configure_console():
    """UTF-8 output so the Chinese banner is readable in the console window."""
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _set_title(title=TITLE):
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except (AttributeError, OSError):
        pass


class ConsoleLog:
    """Always-on file log, so a failed fallback can never die silently."""

    def __init__(self, path):
        self.path = Path(path)
        self._failed = False

    def note(self, *parts):
        line = " ".join(str(part) for part in parts)
        if not self._failed:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as log:
                    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            except OSError:
                self._failed = True
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001 - a missing console must never be fatal
            pass

    def ask(self, prompt=""):
        """Real console input only; ``None`` means input is unavailable.

        EOF is deliberately *not* treated as "the user pressed Enter": the
        launcher gives this process a NUL stdin, and reading that as a request to
        exit would tear the backend down the moment it came up.
        """
        if sys.stdin is None:
            return None
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:  # noqa: BLE001 - no usable console
            return None


def console_attached():
    """True when this process really owns a console window."""
    try:
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except (AttributeError, OSError):
        return False


def finish(backend, url, note=print, ask=input):
    def confirm(reason, title, body, label):
        note(title, body)
        if reason == "notice":
            return False
        try:
            answer = ask(f"{label}？ 输入 y 确认，回车取消 [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return str(answer).strip().lower() in ("y", "yes")

    shown = None
    def report(message, _cancellable):
        nonlocal shown
        if message != shown:
            note(message)
            shown = message

    return close_surface(url, backend, confirm, report)


def run(port, data_root, reason, note=None, ask=None, open_browser=True):
    # Imported lazily: this module must never import the WebView/pywebview stack,
    # because that stack is exactly what failed on the way here.
    from desktop.launcher import ProcessLock, ensure_backend

    workspace = Path(__file__).resolve().parents[1]
    runtime = workspace / "runtime"
    runtime.mkdir(exist_ok=True)
    log = ConsoleLog(runtime / "console.log")
    note = note or log.note
    ask = ask or log.ask
    _configure_console()
    _open_console_streams()
    note("=" * 66)
    note(" 桌面窗口无法启动，已切换到「终端 + 浏览器」模式。")
    note(" 原因：" + reason)
    note(f" 日志：{log.path}")
    note(" 按回车查看退出选项。直接关闭终端会保留后台任务。")
    note("=" * 66)

    lock = ProcessLock(runtime / "desktop.lock")
    if not lock.acquire():
        note("LP LoRA Trainer 已在运行（桌面窗口或另一个终端窗口），本窗口将退出。")
        return 0
    url = f"http://127.0.0.1:{port}"
    backend = None
    handler = None
    try:
        # The backend comes first: the UI must work even if console output or the
        # browser launch misbehaves on this machine.
        backend = ensure_backend(url, Path(data_root).resolve() if data_root else workspace / "data", port)
        note(f" 界面地址：{url}" + ("（已尝试用默认浏览器打开）" if open_browser else ""))
        note(" 后端与监管进程已就绪。按回车查看退出选项。")
        if open_browser and not webbrowser.open(url):
            note("未能自动打开浏览器，请手动访问上面的地址。")
        note("")

        # Closing the console window is CTRL_CLOSE_EVENT, which does not reach
        # Python as a signal. OS console closure only detaches this surface.
        quit_flag = {"closing": False}

        def on_console_event(event):
            # Windows can terminate a closing console before a dialog or a network
            # request completes. Release only this surface on those OS events.
            quit_flag["closing"] = True
            return False

        handler = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(on_console_event)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
        while not quit_flag["closing"]:
            answer = ask("")
            if answer is None:
                note("控制台输入不可用。可使用浏览器界面继续管理任务，或直接关闭本窗口。")
                while not quit_flag["closing"] and (backend is None or backend.poll() is None):
                    time.sleep(1)
                break
            if not quit_flag["closing"] and finish(backend, url, note=note, ask=ask):
                break
        return 0
    except Exception as exc:  # noqa: BLE001 - keep the window open with the reason
        import traceback
        note("")
        note(f"终端模式启动失败：{type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines():
            note("  " + line)
        note("请把上面的内容或 runtime\\console.log 提供给维护者。")
        ask("按回车键关闭…")
        return 1
    finally:
        try:
            if handler is not None:
                ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, False)
        except (AttributeError, OSError):
            pass
        lock.close()


def main():
    parser = argparse.ArgumentParser(description="LP LoRA Trainer terminal + browser mode")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--data-root")
    parser.add_argument("--reason", default="未提供原因")
    parser.add_argument("--reason-file", help="从文件读取原因（避免命令行转义问题）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器（仅打印地址）")
    args = parser.parse_args()
    # --data-root wins; otherwise the launcher's ASCII-only fallback script passes
    # the root through the environment (it cannot encode a non-ASCII path there).
    data_root = args.data_root or os.environ.get(CONSOLE_DATA_ROOT_ENV) or None
    reason = args.reason
    if args.reason_file:
        try:
            reason = Path(args.reason_file).read_text(encoding="utf-8").strip() or reason
        except OSError:
            pass
    raise SystemExit(run(args.port, data_root, reason, open_browser=not args.no_browser))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
