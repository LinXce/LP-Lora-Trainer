"""Start/reconnect a local backend and then show one desktop WebView."""
import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from app import APP_VERSION, INSTALLATION_WORKFLOW_VERSION
from app.config import DEFAULT_API_PORT
from desktop.window import desktop_bridge_error, open_window
from desktop.console import CONSOLE_DATA_ROOT_ENV
from desktop.shutdown import close_surface
from supervisor.worker import ProcessLock


class IncompatibleBackend(RuntimeError):
    """The port is already served by an old or different-data-root LP backend.

    This is a configuration conflict the user resolves by stopping that backend,
    not a desktop-window failure, so it must never trigger the browser fallback.
    """


def backend_ready(url, data_root=None):
    # Bootstrap without putting any session secret in the URL or logs.
    try:
        import http.cookiejar
        jar = http.cookiejar.CookieJar()
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
        req = urllib.request.Request(url + "/api/v1/session", method="POST", headers={"Origin": url})
        with client.open(req, timeout=2) as res:
            if res.status != 204: return False
        with client.open(url + "/api/v1/system/status", timeout=2) as res:
            status = json.load(res)
        if (status.get("backend_version") != APP_VERSION
                or status.get("installation_workflow_version") != INSTALLATION_WORKFLOW_VERSION):
            # An authenticated but incompatible API is NOT an unused port.
            # Never spawn a second backend or silently reuse old installation code.
            raise IncompatibleBackend(
                "该端口正在运行旧版或不兼容的 LP 后端，不能复用其安装流程。"
                "请先停止安装/训练，确认进程已退出，再退出本项目旧 API 和 supervisor 后重新打开 EXE；"
                "旧版后台可能在关闭窗口后仍驻留，请勿在活动任务期间强制结束进程。"
            )
        if data_root is not None:
            with client.open(url + "/api/v1/settings", timeout=2) as res:
                settings = json.load(res)
            if Path(settings["data_root"]).resolve() != Path(data_root).resolve():
                raise IncompatibleBackend("该端口的后端使用其他数据目录，请选择其他端口或先退出原后端")
        return True
    except (OSError, ValueError, urllib.error.URLError): return False


def ensure_backend(url, data_root, port, workspace=None, runtime=None):
    """Reuse a compatible local backend or start one; return the process (or None).

    ``None`` means an already-running backend was reused, so this window does not
    own its lifetime.
    """
    workspace = Path(workspace or Path(__file__).resolve().parents[1])
    runtime = Path(runtime or workspace / "runtime")
    if backend_ready(url, data_root):
        return None
    argv = [sys.executable, "-m", "app.api.server", "--port", str(port)]
    if data_root is not None:
        argv.extend(["--data-root", str(Path(data_root).resolve())])
    options = dict(cwd=workspace, stdin=subprocess.DEVNULL)
    if os.name == "nt": options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else: options["start_new_session"] = True
    with (runtime / "backend.log").open("ab") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=log, **options)
    # The backend and its training must survive an unexpected window/console exit.
    deadline = time.monotonic() + 30
    while not backend_ready(url, data_root):
        if process.poll() is not None or time.monotonic() > deadline:
            # A probe may time out after the backend has already resumed a job.
            # Leave its lifetime alone; the fallback may reconnect once it is ready.
            raise RuntimeError(f"Local backend startup failed; see {runtime / 'backend.log'}")
        time.sleep(0.3)
    return process


class DesktopBackend:
    """Both window gestures delegate to one asynchronous, ownership-aware decision."""

    def __init__(self, url):
        self.url = url
        self.process = None
        self.ready = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._answer = None
        self._answered = threading.Event()
        self._cancel = threading.Event()
        self.closing_approved = False

    def prepare(self, start):
        try:
            self.process = start()
        finally:
            self.ready.set()

    def closing(self, window):
        # FormClosing must return before we query the API or call into WebView.
        with self._lock:
            if self.closing_approved:
                return True
            if self._thread is None:
                self._cancel.clear()
                self._thread = threading.Thread(target=self._decide, args=(window,),
                                                name="lp-close", daemon=True)
                self._thread.start()
        return False

    @staticmethod
    def _emit(window, event, payload):
        js = getattr(window, "evaluate_js", None)
        if not callable(js):
            return False
        try:
            data = json.dumps(payload, ensure_ascii=False)
            return js(f"!window.dispatchEvent(new CustomEvent('{event}', {{detail:{data},cancelable:true}}))") is True
        except Exception:
            return False

    def _decide(self, window):
        veil_since = None

        def ask(reason, title, body, label):
            nonlocal veil_since
            veil_since = None  # A confirmation replaces the progress view.
            with self._lock:
                self._answer = None
                self._answered.clear()
            prompt = dict(reason=reason, title=title, body=body, yes_label=label)
            if self._emit(window, "lp-close-request", prompt):
                self._answered.wait()
                return self._answer == "confirm"
            return window.create_confirmation_dialog(title, body + "\n" + label + " / 返回")

        def progress(message, cancellable):
            nonlocal veil_since
            shown = self._emit(window, "lp-close-progress", dict(message=message, cancellable=cancellable))
            if shown and message and veil_since is None:
                veil_since = time.monotonic()

        try:
            self.ready.wait()
            if close_surface(self.url, self.process, ask, progress, self._cancel.is_set):
                if veil_since is None:
                    progress("正在退出…", False)
                # A quick detach can otherwise destroy the WebView in the same
                # frame. Time already spent showing shutdown progress counts.
                if veil_since is not None:
                    remaining = 0.45 - (time.monotonic() - veil_since)
                    if remaining > 0:
                        time.sleep(remaining)
                with self._lock:
                    self.closing_approved = True
                # Re-enter FormClosing only after its initial event has returned.
                window.destroy()
        except Exception:
            logging.getLogger(__name__).exception("Window close failed")
            with self._lock:
                self.closing_approved = False
        finally:
            if not self.closing_approved:
                progress("", False)
            with self._lock:
                self._thread = None

    def resolve_close(self, window, action):
        with self._lock:
            if self._thread is None:
                return
            if action == "cancel_wait":
                self._cancel.set()
            elif action in ("confirm", "cancel") and not self._answered.is_set():
                self._answer = action
                self._answered.set()


# The fallback script lives in <workspace>\runtime, so it addresses everything it
# starts through %~dp0 (its own directory) instead of absolute paths. cmd expands
# that prefix before Python starts, which keeps the generated text pure ASCII even
# when the workspace path (e.g. a Chinese desktop folder) is not.
BROWSER_MODE_SCRIPT = """@echo off
chcp 65001 >nul
title LP LoRA Trainer - browser mode
type "{logo}"
echo.
"{python}" -m desktop.console --port {port} --reason-file "{reason_file}"
"""


def _runtime_relative(target, runtime):
    """Path for browser-mode.cmd, relative to the script's own directory.

    Everything the fallback needs lives in or next to the workspace, so ``%~dp0``
    addresses it without writing the workspace path into the file. A target on a
    different drive has no relative form; its absolute path is then used, which is
    still fine for ASCII-only install locations.
    """
    try:
        return "%~dp0" + os.path.relpath(Path(target), Path(runtime))
    except ValueError:
        return str(Path(target))


def spawn_console(port, data_root, reason, workspace=None):
    """Open the machine's own terminal for the no-window fallback.

    The terminal window is the system console (`cmd.exe`), not a Python-owned
    console: it prints the text logo from `assets/logo.txt` and then hands over to
    ``desktop.console``, which owns the backend and turns *closing this window*
    into a graceful stop of the backend and its supervisor.

    A child that dies immediately is reported instead of failing silently.
    """
    workspace = Path(workspace or Path(__file__).resolve().parents[1])
    runtime = Path(workspace) / "runtime"
    runtime.mkdir(exist_ok=True)
    interpreter = Path(sys.executable).with_name("python.exe")
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    # The reason can contain Windows paths and quotes, so it travels in a file
    # rather than on the cmd command line.
    reason_file = runtime / "console-reason.txt"
    reason_file.write_text(str(reason), encoding="utf-8")
    script = runtime / "browser-mode.cmd"
    # Only ASCII goes into the script: a non-ASCII workspace would otherwise raise
    # UnicodeEncodeError here and fail the fallback that was just entered.
    script.write_text(
        BROWSER_MODE_SCRIPT.format(
            logo=_runtime_relative(workspace / "assets" / "logo.txt", runtime),
            python=_runtime_relative(interpreter, runtime),
            port=port,
            reason_file=_runtime_relative(reason_file, runtime),
        ),
        encoding="ascii",
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # The data root can contain non-ASCII characters too, and console.run() prefers
    # its --data-root; passing it here keeps that path out of the .cmd file.
    if data_root is not None:
        env[CONSOLE_DATA_ROOT_ENV] = str(Path(data_root).resolve())
    else:
        env.pop(CONSOLE_DATA_ROOT_ENV, None)
    # No stdio redirection and no STARTUPINFO: both would make Python pass
    # STARTF_USESTDHANDLES with empty output handles, and cmd.exe dies at once
    # when its stdout/stderr are invalid. Inheriting lets the new console supply
    # its own handles; visibility is enforced by the console host itself.
    options = dict(cwd=workspace, env=env)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        options["start_new_session"] = True
    argv = ["cmd.exe", "/c", str(script)] if os.name == "nt" else [str(interpreter), "-m", "desktop.console"]
    process = subprocess.Popen(argv, **options)
    time.sleep(1.5)
    if process.poll() is not None:
        raise RuntimeError(
            f"终端窗口进程立即退出（退出码 {process.returncode}）；"
            f"详见 {runtime / 'console.log'}"
        )
    return process


def main():
    parser = argparse.ArgumentParser(description="LP LoRA Trainer desktop")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--data-root")
    parser.add_argument("--browser", action="store_true",
                        help="Force the terminal + browser mode even if a desktop window could start")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if not (workspace / "frontend" / "dist" / "index.html").is_file():
        raise RuntimeError("前端构建缺失：请携带 frontend/dist，开发时执行 npm --prefix frontend run build")
    runtime = workspace / "runtime"
    runtime.mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    data_root = Path(args.data_root).resolve() if args.data_root else workspace / "data"

    # Check the desktop runtime first: a window that cannot be created must not
    # leave an unreachable backend/supervisor pair behind.
    reason = "命令行指定了 --browser" if args.browser else desktop_bridge_error()
    if reason:
        # The console window takes the desktop lock and owns the backend.
        spawn_console(args.port, args.data_root, reason, workspace)
        return

    lock = ProcessLock(runtime / "desktop.lock")
    if not lock.acquire(): raise SystemExit("LP LoRA Trainer 桌面窗口已打开")
    window_error = None
    backend = DesktopBackend(url)
    try:
        # The window shows a splash right away; the backend starts behind it.
        open_window(url, prepare=lambda: backend.prepare(
            lambda: ensure_backend(url, data_root, args.port, workspace, runtime)),
            bounds_file=runtime / "window-bounds.json", on_close=backend.closing,
            on_close_action=backend.resolve_close)
    except IncompatibleBackend:
        # A conflicting backend is resolved by stopping it, not by opening a
        # browser surface that would fail against the same port.
        raise
    except (Exception, SystemExit) as exc:  # desktop/bootstrap failures use browser fallback
        window_error = exc
    finally:
        lock.close()

    if window_error is None:
        return
    # The desktop surface failed (WebView2/.NET, pywebview, or the backend
    # bootstrap behind the splash). Fall back to the terminal + browser surface
    # instead of leaving the user with only an error box; the lock is free by now
    # so the console can own the backend.
    fallback_reason = f"桌面窗口启动失败：{type(window_error).__name__}: {window_error}"
    try:
        spawn_console(args.port, args.data_root, fallback_reason, workspace)
    except Exception as exc:  # noqa: BLE001 - report both failures to the caller
        raise RuntimeError(f"{fallback_reason}；终端回退也失败：{exc}") from exc


def run():
    """pythonw has no stderr: always persist and surface startup errors."""
    try:
        main()
    except SystemExit as exc:
        # argparse --help and a second desktop invocation are not crashes.
        if exc.code in (None, 0) or "桌面窗口已打开" in str(exc):
            return
        _startup_error(exc)
    except Exception as exc:
        _startup_error(exc)


def _startup_error(exc):
    import traceback
    log_path = Path(__file__).resolve().parents[1] / "runtime" / "desktop.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=log)
    except OSError:
        pass
    message = f"LP LoRA Trainer 启动失败\n\n{exc}\n\n日志：{log_path}"
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "LP LoRA Trainer", 0x10)
    elif sys.stderr is not None:
        print(message, file=sys.stderr)


if __name__ == "__main__": run()
