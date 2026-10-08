"""Loopback-only API matching the existing Vue UI. No training imports."""
import argparse
import asyncio
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from app import INSTALLATION_WORKFLOW_VERSION
from app.api.models import SettingsInput, DatasetInput, CaptionInput, PythonInput, EngineTypeInput, TrainingInput, StopInput, PublishInput, AcknowledgeInput
from app.config import AppPaths, DEFAULT_API_PORT
from app.services.common import ServiceError, local_path, contained, now
from app.services.datasets import DatasetService
from app.services.engines import EngineService
from app.services.installation import InstallationService
from app.api.models import InstallEnvironmentInput
from app.services.training import TrainingService, ACTIVE
from app.storage.state import StateStore


def supervisor_status(root):
    try:
        data = json.loads((root / "supervisor.json").read_text(encoding="utf-8"))
        if time.time() - data["heartbeat"] < 5:
            return "running" if data["task_id"] else "idle"
    except (OSError, ValueError, KeyError): pass
    return "unreachable"


def ensure_supervisor(root, workspace):
    if supervisor_status(root) != "unreachable": return
    options = dict(cwd=workspace, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.name == "nt": options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else: options["start_new_session"] = True
    subprocess.Popen([sys.executable, "-m", "supervisor.worker", "--data-root", str(root)], **options)


class SystemMonitor:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.gpu, self.last_poll = None, 0
        self.lock = threading.Lock()

    def status(self):
        if not self.settings()["gpu_monitor"]:
            self.gpu = None
        elif time.monotonic() - self.last_poll > 15:
            with self.lock:
                if time.monotonic() - self.last_poll > 15:
                    self.last_poll = time.monotonic()
                    try:
                        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        row = p.stdout.splitlines()[0].split(",") if p.returncode == 0 else []
                        self.gpu = dict(name=row[0].strip(), memory_used_mb=int(row[1]), memory_total_mb=int(row[2]), utilization=int(row[3])) if len(row) == 4 else None
                    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired): self.gpu = None
        return dict(backend_version="0.1.0", installation_workflow_version=INSTALLATION_WORKFLOW_VERSION,
                    supervisor=supervisor_status(self.store.root), gpu=self.gpu)


def create_app(paths=None, token=None, start_worker=True, development=False):
    paths = paths or AppPaths.for_workspace(Path(__file__).resolve().parents[2])
    store = StateStore(paths.data_root)
    # Starting the backend begins a fresh lifetime: a shutdown request left over
    # from a previous backend/supervisor pair must neither stop the supervisor we
    # are about to start nor block new submissions.
    store.delete("meta", "shutdown")
    token = token or secrets.token_urlsafe(32)
    defaults = dict(engine_root=str(paths.engine_root), data_root=str(paths.data_root), comfyui_lora_dir=None, gpu_monitor=False, log_tail_lines=500)
    if store.get("meta", "settings") is None: store.put("meta", "settings", defaults)
    else: store.patch("meta", "settings", {"data_root": str(store.root)})
    settings = lambda: store.get("meta", "settings")
    engines = EngineService(store, settings, paths.project_root)
    installations = InstallationService(store, engines, paths.project_root)
    datasets = DatasetService(store)
    training = TrainingService(store, engines, datasets)
    monitor = SystemMonitor(store, settings)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker: await asyncio.to_thread(ensure_supervisor, store.root, paths.project_root)
        try:
            yield
        finally:
            await asyncio.to_thread(installations.shutdown)
        # The independent supervisor stays alive when the window/API exits.

    app = FastAPI(title="LP LoRA Trainer", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.engines, app.state.datasets, app.state.training = store, engines, datasets, training
    app.state.installations = installations
    app.state.session_token = token
    # Set by ``main()`` in real runs; ``None`` means this process cannot be asked to exit.
    app.state.server = None

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.middleware("http")
    async def local_security(request, call_next):
        host = request.url.hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            return JSONResponse({"detail": "Only local requests are allowed"}, status_code=403)
        origin = request.headers.get("origin")
        allowed = {f"{request.url.scheme}://{request.url.netloc}"}
        if development: allowed.update({"http://127.0.0.1:5173", "http://localhost:5173"})
        if origin and origin not in allowed:
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        if request.url.path.startswith("/api/"):
            provided = request.headers.get("X-LP-Session") or request.cookies.get("lp_session", "")
            bootstrap = request.url.path == "/api/v1/session" and request.method == "POST" and origin in allowed
            if not bootstrap and not hmac.compare_digest(provided, token):
                return JSONResponse({"detail": "本机会话失效，请重新连接应用"}, status_code=401)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        result = await call_next(request)
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "no-referrer"
        result.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"): result.headers["Cache-Control"] = "no-store"
        return result

    @app.post("/api/v1/session")
    def session():
        result = Response(status_code=204)
        result.set_cookie("lp_session", token, httponly=True, samesite="strict", path="/api/", secure=False)
        return result

    @app.get("/api/v1/system/status")
    def status(): return monitor.status()

    @app.post("/api/v1/system/shutdown", status_code=202)
    def shutdown():
        """Ask the backend and the independent supervisor to exit gracefully.

        Refused while training is active or an orphan is unverified, so this can
        never be a back door for killing a job.
        """
        states = {t["state"] for t in store.list("task")}
        if states & ACTIVE:
            raise ServiceError("仍有训练任务处于活动状态，已拒绝退出；请先等待任务结束或停止任务", 409)
        if "connection_lost" in states:
            raise ServiceError("存在连接丢失、需人工核实的任务，已拒绝退出", 409)
        # The supervisor polls this marker; the backend flips uvicorn's own flag.
        store.put("meta", "shutdown", dict(requested_at=now(), backend_pid=os.getpid()))
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True
        return dict(status="shutting_down")

    @app.get("/api/v1/settings")
    def get_settings(): return settings()

    @app.put("/api/v1/settings")
    def save_settings(body: SettingsInput):
        value = body.model_dump()
        engine_root = local_path(value["engine_root"])
        if not engine_root.is_dir(): raise ServiceError("Engine directory does not exist")
        value["engine_root"] = str(engine_root)
        data_root = local_path(value["data_root"])
        if data_root != store.root:
            raise ServiceError("The data directory cannot be changed while the service is running; restart with --data-root", 409)
        value["data_root"] = str(store.root)
        if value["comfyui_lora_dir"]:
            dest = local_path(value["comfyui_lora_dir"])
            if not dest.is_dir(): raise ServiceError("ComfyUI LoRA directory does not exist")
            value["comfyui_lora_dir"] = str(dest)
        return store.put("meta", "settings", value)

    @app.get("/api/v1/engines")
    def engine_list(): return engines.list()

    @app.post("/api/v1/engines/rescan")
    def engine_rescan(): return engines.rescan()

    @app.post("/api/v1/engines/{key}/default", status_code=204)
    def engine_default(key: str): engines.set_default(key)

    @app.post("/api/v1/engines/{key}/diagnose", status_code=204)
    def engine_diagnose(key: str): engines.diagnose(key)

    @app.put("/api/v1/engines/{key}/python", status_code=204)
    def engine_python(key: str, body: PythonInput): engines.bind(key, body.python_executable)

    @app.put("/api/v1/engines/{key}/engine-type", status_code=204)
    def engine_type(key: str, body: EngineTypeInput): engines.confirm(key, body.engine_id)

    @app.post("/api/v1/engines/{key}/install", status_code=202)
    def engine_install(key: str, body: InstallEnvironmentInput):
        return installations.start(key, body.python_executable, body.torch_source, body.mirror_url)

    @app.get("/api/v1/terminal/sessions")
    def terminal_sessions(): return installations.list()

    @app.get("/api/v1/terminal/sessions/{key}/log")
    def terminal_log(key: str, offset: int = Query(0, ge=0)):
        return installations.read_log(key, offset)

    @app.post("/api/v1/terminal/sessions/{key}/stop", status_code=204)
    def terminal_stop(key: str): installations.cancel(key)

    @app.get("/api/v1/engines/{key}/capabilities")
    def engine_capabilities(key: str): return engines.capabilities(key)

    @app.get("/api/v1/base-models")
    def base_models(): return engines.base_models()

    @app.get("/api/v1/datasets")
    def dataset_list(): return datasets.list()

    @app.post("/api/v1/datasets")
    def dataset_add(body: DatasetInput): return datasets.add(body.path, body.name)

    @app.post("/api/v1/datasets/{key}/scan", status_code=204)
    def dataset_scan(key: str): datasets.scan(key)

    @app.get("/api/v1/datasets/{key}/images")
    def dataset_images(key: str, offset: int = Query(0, ge=0), limit: int = Query(48, ge=1, le=200), issue: str | None = None):
        if issue and issue not in ("corrupt", "missing_caption", "odd_size", "duplicate"): raise ServiceError("Unknown image filter")
        return datasets.images(key, offset, limit, issue)

    @app.put("/api/v1/datasets/{key}/images/{image_id}/caption", status_code=204)
    def caption_save(key: str, image_id: str, body: CaptionInput): datasets.caption(key, image_id, body.caption)

    @app.get("/api/v1/datasets/{key}/images/{image_id}/thumbnail")
    def thumbnail(key: str, image_id: str): return FileResponse(datasets.thumbnail(key, image_id), media_type="image/jpeg")

    @app.post("/api/v1/training/validate")
    def validate(body: TrainingInput): return training.validate(body.model_dump())

    @app.post("/api/v1/training/submit")
    def submit(body: TrainingInput):
        # A graceful exit is already in flight; never accept work no supervisor will claim.
        if store.get("meta", "shutdown"):
            raise ServiceError("后端与监管进程正在退出，无法提交新任务；请重新启动应用", 409)
        if start_worker and supervisor_status(store.root) == "unreachable":
            ensure_supervisor(store.root, paths.project_root)
        if start_worker and any(t["state"] == "connection_lost" for t in store.list("task")):
            raise ServiceError("A task lost its supervisor connection; verify the training process before continuing", 409)
        return training.submit(body.model_dump())

    @app.get("/api/v1/tasks")
    def task_list(): return training.tasks()

    @app.get("/api/v1/tasks/{key}/metrics")
    def metrics(key: str): return training.metrics(key)

    @app.get("/api/v1/tasks/{key}/log")
    def logs(key: str, tail: int = Query(500, ge=1, le=10000)): return training.log(key, tail)

    @app.post("/api/v1/tasks/{key}/stop", status_code=204)
    def stop(key: str, body: StopInput): training.stop(key, body.force)

    @app.post("/api/v1/tasks/{key}/acknowledge-exit", status_code=204)
    def acknowledge_exit(key: str, body: AcknowledgeInput): training.acknowledge_exit(key)

    @app.get("/api/v1/artifacts")
    def artifacts(task_id: str | None = None): return training.artifacts(task_id)

    @app.post("/api/v1/artifacts/{key}/publish", status_code=204)
    def publish(key: str, body: PublishInput): training.publish(key, body.target_dir, body.file_name)

    @app.get("/api/v1/artifacts/{key}/preview")
    def preview(key: str):
        artifact = store.get("artifact", key)
        if not artifact or artifact["kind"] != "sample" or not artifact["complete"]: raise ServiceError("Preview is unavailable", 404)
        path = contained(artifact["path"], training.task(artifact["task_id"])["output_dir"])
        if not path.is_file(): raise ServiceError("Sample image no longer exists", 404)
        from PIL import Image, ImageOps
        import io
        with Image.open(path) as image:
            image.thumbnail((768, 768))
            im = ImageOps.exif_transpose(image).convert("RGB")
            data = io.BytesIO(); im.save(data, "JPEG", quality=85)
        return Response(content=data.getvalue(), media_type="image/jpeg")

    @app.get("/api/v1/events")
    async def events(request: Request):
        async def frames():
            previous_tasks, previous_engines = {}, {}
            cursors, metric_cursors = {}, {}
            old_status, ticks = None, 0
            while not await request.is_disconnected():
                tasks, engine_items = await asyncio.to_thread(lambda: (training.tasks(), engines.list()))
                for name, items, previous, id_key, body_key in (("task.updated", tasks, previous_tasks, "task_id", "task"), ("engine.updated", engine_items, previous_engines, "installation_id", "installation")):
                    for item in items:
                        key = item[id_key]
                        if previous.get(key) != item:
                            previous[key] = item
                            yield "data: " + json.dumps(dict(kind=name, **{body_key: item}), ensure_ascii=False) + "\n\n"
                for task in tasks:
                    key = task["task_id"]
                    for filename, offsets, kind, field in (("stdout.log", cursors, "task.log", "lines"), ("metrics.jsonl", metric_cursors, "task.metrics", "points")):
                        p = store.root / "jobs" / key / filename
                        if key not in offsets:
                            # Snapshot already loads history; only stream subsequent bytes.
                            offsets[key] = p.stat().st_size if p.is_file() else 0
                        if p.is_file() and p.stat().st_size > offsets[key]:
                            def read_chunk():
                                with p.open("rb") as f:
                                    f.seek(offsets[key]); chunk = f.read(256 * 1024)
                                    end = chunk.rfind(b"\n")
                                    if end < 0: return b"", offsets[key]
                                    return chunk[:end+1], offsets[key] + end + 1
                            data, offsets[key] = await asyncio.to_thread(read_chunk)
                            lines = data.decode("utf-8", "replace").splitlines()
                            if field == "points":
                                points = []
                                for line in lines:
                                    try: points.append(json.loads(line))
                                    except ValueError: pass
                                lines = points
                            if lines: yield "data: " + json.dumps(dict(kind=kind, task_id=key, **{field: lines}), ensure_ascii=False) + "\n\n"
                status = await asyncio.to_thread(monitor.status)
                if old_status != status:
                    old_status = status
                    yield "data: " + json.dumps(dict(kind="system.status", status=status), ensure_ascii=False) + "\n\n"
                ticks += 1
                if ticks % 15 == 0: yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(frames(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    if paths.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=paths.frontend_dist, html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend(): return JSONResponse({"detail": "前端未构建，请在 frontend 中执行 npm run build"}, status_code=503)
    return app


def main():
    parser = argparse.ArgumentParser(description="LP LoRA Trainer local backend")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--data-root")
    parser.add_argument("--dev", action="store_true", help="Allow Vite's local dev origin")
    args = parser.parse_args()
    paths = AppPaths.for_workspace(Path(__file__).resolve().parents[2])
    if args.data_root:
        from dataclasses import replace
        paths = replace(paths, data_root=local_path(args.data_root))
    from supervisor.worker import ProcessLock
    lock = ProcessLock(paths.data_root / "backend.lock")
    if not lock.acquire(): raise SystemExit("Another backend is already running for this data directory")
    import uvicorn
    app = create_app(paths, development=args.dev)
    # A real server object lets /system/shutdown unwind the ASGI stack (lifespan
    # included) instead of leaving the process to a hard exit.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, access_log=False))
    app.state.server = server
    try: server.run()
    finally: lock.close()


if __name__ == "__main__": main()
