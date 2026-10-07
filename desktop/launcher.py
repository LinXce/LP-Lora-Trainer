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
from desktop.window import open_window
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


def main():
    parser = argparse.ArgumentParser(description="LP LoRA Trainer desktop")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if not (workspace / "frontend" / "dist" / "index.html").is_file():
        raise RuntimeError("前端构建缺失：请携带 frontend/dist，开发时执行 npm --prefix frontend run build")
    runtime = workspace / "runtime"
    runtime.mkdir(exist_ok=True)
    lock = ProcessLock(runtime / "desktop.lock")
    if not lock.acquire(): raise SystemExit("LP LoRA Trainer 桌面窗口已打开")
    url = f"http://127.0.0.1:{args.port}"
    data_root = Path(args.data_root).resolve() if args.data_root else workspace / "data"
    try:
        if not backend_ready(url, data_root):
            argv = [sys.executable, "-m", "app.api.server", "--port", str(args.port)]
            if args.data_root: argv.extend(["--data-root", str(Path(args.data_root).resolve())])
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
