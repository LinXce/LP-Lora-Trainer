"""Terminal + browser fallback used when the desktop WebView cannot start.

The console window owns the backend lifetime: closing this window (or pressing
Ctrl+C / Enter) stops the backend and its supervisor, so the terminal is the one
thing the user closes. A desktop window that *can* start never shows a console.
"""
import argparse
import ctypes
import http.cookiejar
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from app.config import DEFAULT_API_PORT

TITLE = "LP LoRA Trainer（终端 + 浏览器模式）"


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


def request_shutdown(url, timeout=5.0):
    """Ask the backend to exit gracefully.

    Returns "stopped" (accepted), "busy" (refused: active work must not be
    interrupted) or "unreachable" (no answer). The endpoint also signals the
    supervisor, so a graceful exit takes both processes with it.
    """
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
    try:
        client.open(urllib.request.Request(url + "/api/v1/session", data=b"", method="POST",
                                          headers={"Origin": url}), timeout=timeout)
        response = client.open(urllib.request.Request(url + "/api/v1/system/shutdown", data=b"", method="POST",
                                                     headers={"Origin": url}), timeout=timeout)
        return "stopped" if response.status == 202 else "unreachable"
    except urllib.error.HTTPError as exc:
        return "busy" if exc.code == 409 else "unreachable"
    except (OSError, ValueError):
        return "unreachable"


def kill_tree(pid):
    """Force-stop a process tree using a fixed command and a numeric pid."""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        pass


def finish(backend, url, note=print, ask=input):
    """Stop the backend this console owns, then return.

    A backend started here is always stopped, asking first only when training is
    still active. A backend owned by another instance is stopped only when it can
    exit gracefully, so this window can never kill someone else's training run.
    """
    outcome = request_shutdown(url)
    if outcome == "stopped":
        note("已请求后端与监管进程优雅退出。")
        if backend is not None:
            try:
                backend.wait(timeout=30)
            except (OSError, subprocess.SubprocessError):
                kill_tree(backend.pid)
        return
    if outcome == "busy":
        note("后端拒绝退出：仍有活动任务或需人工核实的任务。")
        if backend is None:
            note("该后端由其他实例启动，本窗口不会强制结束它。")
            return
        try:
            answer = ask("仍要强制结束后端与训练进程吗？输入 y 确认：[y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if str(answer).strip().lower() in ("y", "yes"):
            note("正在强制结束后端与监管进程…")
            kill_tree(backend.pid)
        else:
            note("已取消，后端继续运行（训练不受影响）。")
        return
    # Unreachable: nothing to signal, so only an owned process is cleaned up.
    note("后端未响应退出请求。")
    if backend is not None:
        kill_tree(backend.pid)


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
    note(" 关闭本窗口（或按 Ctrl+C / 回车）即结束后端与监管进程。")
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
        note(" 后端与监管进程已就绪。按回车键结束，或直接关闭本窗口。")
        if open_browser and not webbrowser.open(url):
            note("未能自动打开浏览器，请手动访问上面的地址。")
        note("")

        # Closing the console window is CTRL_CLOSE_EVENT, which does not reach
        # Python as a signal: a console control handler makes it a clean stop.
        quit_flag = {"closing": False}

        def on_console_event(event):
            if quit_flag["closing"]:
                return False
            quit_flag["closing"] = True
            try:
                finish(backend, url, note=note, ask=ask)
            finally:
                sys.stdout.flush()
            return False  # let Windows terminate this console

        handler = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(on_console_event)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
        answer = ask("")
        if answer is None:
            # No usable console input: stay alive as the backend owner rather than
            # shutting the backend down just because stdin was not interactive.
            note("控制台输入不可用：本窗口保持运行以持有后端与监管进程。")
            note("可用界面「设置 → 退出后端与监督」结束，或直接关闭本窗口。")
            while not quit_flag["closing"]:
                time.sleep(1)
        if not quit_flag["closing"]:
            quit_flag["closing"] = True
            finish(backend, url, note=note, ask=ask)
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
    reason = args.reason
    if args.reason_file:
        try:
            reason = Path(args.reason_file).read_text(encoding="utf-8").strip() or reason
        except OSError:
            pass
    raise SystemExit(run(args.port, args.data_root, reason, open_browser=not args.no_browser))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
