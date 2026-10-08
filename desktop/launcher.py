"""Start/reconnect a local backend and then show one desktop WebView."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from app import INSTALLATION_WORKFLOW_VERSION
from app.config import DEFAULT_API_PORT
from desktop.window import desktop_bridge_error, open_window
from supervisor.worker import ProcessLock


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
        if (status.get("backend_version") != "0.1.0"
                or status.get("installation_workflow_version") != INSTALLATION_WORKFLOW_VERSION):
            # An authenticated but incompatible API is NOT an unused port.
            # Never spawn a second backend or silently reuse old installation code.
            raise RuntimeError(
                "该端口正在运行旧版或不兼容的 LP 后端，不能复用其安装流程。"
                "请先停止安装/训练，确认进程已退出，再退出本项目旧 API 和 supervisor 后重新打开 EXE；"
                "仅关闭桌面窗口不会退出后台，请勿在活动任务期间强制结束进程。"
            )
        if data_root is not None:
            with client.open(url + "/api/v1/settings", timeout=2) as res:
                settings = json.load(res)
            if Path(settings["data_root"]).resolve() != Path(data_root).resolve():
                raise RuntimeError("该端口的后端使用其他数据目录，请选择其他端口或先退出原后端")
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
    deadline = time.monotonic() + 30
    while not backend_ready(url, data_root):
        if process.poll() is not None or time.monotonic() > deadline:
            raise SystemExit(f"本地后端启动失败，请查看 {runtime / 'backend.log'}")
        time.sleep(0.3)
    return process


BROWSER_MODE_SCRIPT = """@echo off
chcp 65001 >nul
title LP LoRA Trainer - browser mode
type "{logo}"
echo.
"{python}" -m desktop.console --port {port}{data_root} --reason-file "{reason_file}"
"""


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
    data_root_arg = f' --data-root "{Path(data_root).resolve()}"' if data_root is not None else ""
    script = runtime / "browser-mode.cmd"
    script.write_text(
        BROWSER_MODE_SCRIPT.format(logo=workspace / "assets" / "logo.txt", python=interpreter,
                                   port=port, data_root=data_root_arg, reason_file=reason_file),
        encoding="ascii",
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
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
    try:
        ensure_backend(url, data_root, args.port, workspace, runtime)
        open_window(url)
    finally: lock.close()


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
