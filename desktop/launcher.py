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
        if status.get("backend_version") != "0.1.0": return False
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
        raise SystemExit("前端未构建：请在 frontend 目录执行 npm run build")
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
            if os.name == "nt": options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
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


if __name__ == "__main__": main()
