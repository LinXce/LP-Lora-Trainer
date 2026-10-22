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
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from app.config import DEFAULT_API_PORT

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


# One table for every close dialog. The page renders these strings when it is
# available; the native dialog and console mode fall back to yes/no with the same
# wording, so the two never drift apart. "yes"/"no" are the actions to run.
CLOSE_PROMPTS = {
    "install": dict(
        reason="install", title="环境安装仍在运行",
        body="环境安装可以续跑：停止安装后，下次安装会从中断处继续。训练任务不会被中断。",
        yes_label="停止安装并退出", yes_action="stop_install",
        no_label="返回", no_action="cancel",
    ),
    "training": dict(
        reason="training", title="训练仍在运行",
        body="正在训练。退出会终止训练进程，未完成的进度会丢失；建议先在「训练任务」中停止训练。",
        yes_label="终止训练并退出", yes_action="close",
        no_label="返回", no_action="cancel",
    ),
    "unverified": dict(
        reason="unverified", title="有任务需要先人工核实",
        body="有任务处于「连接丢失」状态。退出会结束该任务并保留已有结果；若想先确认原训练进程是否仍在运行，请返回后处理。",
        yes_label="结束任务并退出", yes_action="close",
        no_label="返回", no_action="cancel",
    ),
    "foreign": dict(
        reason="foreign", title="关闭本窗口",
        body="本窗口连接的是其他实例启动的后端；关闭本窗口不会影响正在运行的任务。",
        yes_label="关闭窗口", yes_action="close",
        no_label="取消", no_action="cancel",
    ),
    "terminate": dict(
        reason="terminate", title="确认终止所有进程？",
        body="继续将终止 LP 的后端、监管进程，以及正在运行的训练或安装。该操作不可恢复，未保存的训练进度会丢失。",
        yes_label="终止所有进程并退出", yes_action="close",
        no_label="取消", no_action="cancel",
    ),
}


def close_prompt(reason):
    """The dialog for one refusal reason, defaulting to the training wording."""
    return CLOSE_PROMPTS.get(str(reason or "").strip(), CLOSE_PROMPTS["training"])


def request_shutdown(url, timeout=5.0, force=False):
    """Ask the backend to exit gracefully.

    Returns "stopped" (accepted), "busy" (refused: active work must not be
    interrupted) or "unreachable" (no answer). The endpoint also signals the
    supervisor, so a graceful exit takes both processes with it. ``force`` asks
    it to cancel a running environment installation first; an active training
    run is still refused.
    """
    return request_shutdown_detail(url, timeout=timeout, force=force)[0]


def request_shutdown_detail(url, timeout=5.0, force=False):
    """``request_shutdown`` plus the refusal detail the backend sends.

    Returns ``(outcome, detail)`` where detail is ``{}`` unless the backend
    refused: it then carries ``reason`` ("training" | "install" | "unverified")
    and the blocking task names, so the caller can pick the right wording.
    """
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
    target = url + "/api/v1/system/shutdown" + ("?force=true" if force else "")
    try:
        client.open(urllib.request.Request(url + "/api/v1/session", data=b"", method="POST",
                                          headers={"Origin": url}), timeout=timeout)
        response = client.open(urllib.request.Request(target, data=b"", method="POST",
                                                     headers={"Origin": url}), timeout=timeout)
        return ("stopped" if response.status == 202 else "unreachable"), {}
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            return "unreachable", {}
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
        except (ValueError, OSError):
            body = {}
        detail = body.get("shutdown") if isinstance(body, dict) else None
        return "busy", detail if isinstance(detail, dict) else {}
    except (OSError, ValueError):
        return "unreachable", {}


def listening_pid(url):
    """PID listening on the port of ``url``, or None.

    Used only for a compatible backend of the same data root that no process of
    ours owns any more (a leftover from an earlier run): it would keep its tasks
    "running" forever, so a confirmed close must be able to end it.
    """
    try:
        port = str(urllib.parse.urlsplit(url).port or 80)
    except ValueError:
        return None
    try:
        result = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True,
                                timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(":" + port) \
                and parts[3].upper() == "LISTENING":
            try:
                return int(parts[4])
            except ValueError:
                return None
    return None


def kill_tree(pid):
    """Force-stop a process tree using a fixed command and a numeric pid."""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        pass


# Windows kills every member of a job object once its last handle closes. Putting
# the backend in such a job means the whole tree — backend, supervisor and any
# engine installer it started — dies with this process even when the window is
# killed from Task Manager or the WebView crashes, not just on a clean close.
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOBS = []  # Module-level: the handle must outlive the launcher, never be collected.


def create_kill_on_close_job(kernel32=None):
    """Create the job object, or return None when it is unavailable."""
    if os.name != "nt" and kernel32 is None:
        return None
    import ctypes
    from ctypes import wintypes

    api = kernel32 if kernel32 is not None else ctypes.WinDLL("kernel32", use_last_error=True)

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    handle = api.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = ExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not api.SetInformationJobObject(handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                       ctypes.byref(limits), ctypes.sizeof(limits)):
        api.CloseHandle(handle)
        return None
    _JOBS.append(handle)
    return handle


def assign_to_job(handle, process, kernel32=None):
    """Move an already running child into the job; False when that is impossible."""
    if not handle or process is None:
        return False
    import ctypes
    api = kernel32 if kernel32 is not None else ctypes.WinDLL("kernel32", use_last_error=True)
    raw = getattr(process, "_handle", None)
    if raw is None:
        return False
    try:
        return bool(api.AssignProcessToJobObject(handle, int(raw)))
    except (TypeError, ValueError):
        # Not a real OS handle (a stub process in tests, or a closed handle).
        return False


def finish(backend, url, note=print, ask=input):
    """Stop the backend this console owns, then return.

    A backend started here is always stopped, asking first only when training is
    still active. A backend owned by another instance is stopped only when it can
    exit gracefully, so this window can never kill someone else's training run.
    """
    outcome, detail = request_shutdown_detail(url)
    if outcome == "stopped":
        note("已请求后端与监管进程优雅退出。")
        if backend is not None:
            try:
                backend.wait(timeout=30)
            except (OSError, subprocess.SubprocessError):
                kill_tree(backend.pid)
        return
    if outcome == "busy":
        # Same table as the window: console mode has no window to minimise, so the
        # primary answer either stops a resumable install or keeps everything alive.
        prompt = close_prompt((detail or {}).get("reason"))
        note(prompt["body"])
        if backend is None:
            note("该后端由其他实例启动，本窗口不会强制结束它。")
            return
        if prompt["yes_action"] != "stop_install":
            note(f"已保留后端与训练进程（{prompt['no_label']}）；请继续使用浏览器界面。")
            return
        try:
            answer = ask(f"{prompt['yes_label']}？输入 y 确认，直接回车表示「{prompt['no_label']}」：[y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if str(answer).strip().lower() not in ("y", "yes"):
            note(f"已取消，后端继续运行（{prompt['no_label']}）。")
            return
        note("正在停止环境安装并请求退出…")
        if request_shutdown(url, force=True) != "stopped":
            note("后端仍拒绝退出：训练任务不会被强制中断，请先在「训练任务」中停止。")
            return
        try:
            backend.wait(timeout=15)
        except (OSError, subprocess.SubprocessError):
            kill_tree(backend.pid)
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
