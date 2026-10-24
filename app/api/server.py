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
from app import APP_VERSION, INSTALLATION_WORKFLOW_VERSION
from app.api.models import SettingsInput, DatasetInput, CaptionInput, PythonInput, EngineTypeInput, TrainingInput, TrainingExportInput, TrainingDraftInput, StopInput, PublishInput, AcknowledgeInput, SourceRepairInput, ResumeInput, TerminalInput
from app.config import AppPaths, DEFAULT_API_PORT
from app.services.common import ServiceError, local_path, contained, now, validate_proxy_url
from app.services.compat import load_manifest
from app.services import hardware
from app.api.models import PromptTemplateInput
from app.services.prompt_templates import PromptTemplateService
from app.services.datasets import DatasetService
from app.services.engines import EngineService
from app.services.installation import InstallationService
from app.api.models import InstallEnvironmentInput, EngineCloneInput
from app.services.training import TrainingService, ACTIVE
from app.services.artifacts import prune_missing_artifacts
from app.storage.state import StateStore


# Longest edge of the images shown in the viewer (thumbnails/previews stay small).
FULL_IMAGE_SIZE = 2048


def supervisor_status(root):
    try:
        data = json.loads((root / "supervisor.json").read_text(encoding="utf-8"))
        if time.time() - data["heartbeat"] < 5:
            return "running" if data["task_id"] else "idle"
    except (OSError, ValueError, KeyError, TypeError): pass
    return "unreachable"


def ensure_supervisor(root, workspace):
    if supervisor_status(root) != "unreachable": return
    options = dict(cwd=workspace, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.name == "nt": options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else: options["start_new_session"] = True
    return subprocess.Popen([sys.executable, "-m", "supervisor.worker", "--data-root", str(root)], **options)


def wait_supervisor_exit(root, timeout=10, processes=()):
    from supervisor.worker import ProcessLock
    deadline = time.monotonic() + timeout
    # A newly spawned worker may not have acquired its lock yet. Wait for owned
    # children first, otherwise an unlocked probe can falsely report completion.
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return
    probe = ProcessLock(Path(root) / "supervisor.lock")
    while time.monotonic() < deadline:
        if probe.acquire():
            probe.close()
            return
        time.sleep(0.1)


class SystemMonitor:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.gpu, self.last_poll = None, 0
        self.net, self.net_at = None, 0.0
        self.lock = threading.Lock()

    @staticmethod
    def _interface_bytes():
        """System-wide received/sent bytes, or None when they cannot be read.

        ``netstat -e`` prints one row whose two integer columns are the interface
        totals; the label itself is localised, so the row is recognised by shape
        rather than by text. psutil is not part of the application runtime.
        """
        try:
            p = subprocess.run(
                ['netstat', '-e'], capture_output=True, text=True, timeout=3,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) != 3 or ':' in line:
                continue
            try:
                return int(parts[1]), int(parts[2])
            except ValueError:
                continue
        return None

    def _network(self):
        """Rates in bytes per second between two status polls, plus the totals."""
        now = time.monotonic()
        current = self._interface_bytes()
        if current is None:
            self.net, self.net_at = None, 0.0
            return None
        previous, previous_at = self.net, self.net_at
        self.net, self.net_at = current, now
        if previous is None or previous_at <= 0 or now <= previous_at:
            return dict(rx_bps=None, tx_bps=None, rx_total=current[0], tx_total=current[1])
        elapsed = now - previous_at
        # A counter reset (interface down/up) would otherwise report a negative rate.
        rx_delta, tx_delta = current[0] - previous[0], current[1] - previous[1]
        return dict(
            rx_bps=max(0.0, rx_delta / elapsed),
            tx_bps=max(0.0, tx_delta / elapsed),
            rx_total=current[0], tx_total=current[1],
        )

    @staticmethod
    def _system_memory():
        """Host RAM in MiB via GlobalMemoryStatusEx (psutil is not bundled).

        Shown next to the GPU panel: most OOM reports here are host memory, not VRAM.
        """
        if os.name != 'nt':
            return None
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total', ctypes.c_ulonglong), ('available', ctypes.c_ulonglong),
                    ('total_page_file', ctypes.c_ulonglong),
                    ('available_page_file', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong),
                    ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended_virtual', ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
        except (AttributeError, OSError):
            return None
        total = status.total // (1024 * 1024)
        available = status.available // (1024 * 1024)
        used = max(0, total - available)
        return dict(used_mb=used, total_mb=total,
                    percent=round(used * 100 / total) if total else None)

    @staticmethod
    def _number(value, integer=False):
        value = str(value or '').strip()
        if not value or value.upper() in {'N/A', '[N/A]', 'NOT SUPPORTED'}:
            return None
        try:
            number = float(value)
            return int(number) if integer else number
        except ValueError:
            return None

    def status(self):
        if not self.settings()["gpu_monitor"]:
            self.gpu = None
        elif time.monotonic() - self.last_poll > 3:
            with self.lock:
                if time.monotonic() - self.last_poll > 3:
                    self.last_poll = time.monotonic()
                    try:
                        query = ','.join([
                            'name', 'memory.used', 'memory.total', 'utilization.gpu',
                            'temperature.gpu', 'utilization.memory', 'power.draw',
                            'power.limit', 'fan.speed',
                        ])
                        p = subprocess.run(
                            ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=3,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                        )
                        row = [item.strip() for item in p.stdout.splitlines()[0].split(',')] if p.returncode == 0 and p.stdout.splitlines() else []
                        if len(row) >= 9:
                            self.gpu = dict(
                                name=row[0],
                                memory_used_mb=self._number(row[1], integer=True),
                                memory_total_mb=self._number(row[2], integer=True),
                                utilization=self._number(row[3], integer=True),
                                temperature_c=self._number(row[4], integer=True),
                                memory_utilization=self._number(row[5], integer=True),
                                power_draw_w=self._number(row[6]),
                                power_limit_w=self._number(row[7]),
                                fan_speed_percent=self._number(row[8], integer=True),
                            )
                        else:
                            self.gpu = None
                    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired): self.gpu = None
        return dict(backend_version=APP_VERSION, installation_workflow_version=INSTALLATION_WORKFLOW_VERSION,
                    supervisor=supervisor_status(self.store.root), gpu=self.gpu,
                    memory=self._system_memory(), net=self._network())


def create_app(paths=None, token=None, start_worker=True, development=False):
    paths = paths or AppPaths.for_workspace(Path(__file__).resolve().parents[2])
    store = StateStore(paths.data_root)
    # Starting the backend begins a fresh lifetime: a shutdown request left over
    # from a previous backend/supervisor pair must neither stop the supervisor we
    # are about to start nor block new submissions.
    store.delete("meta", "shutdown")
    token = token or secrets.token_urlsafe(32)
    defaults = dict(engine_root=str(paths.engine_root), data_root=str(paths.data_root), comfyui_lora_dir=None, gpu_monitor=True, log_tail_lines=500, proxy_url=None, proxy_enabled=False)
    stored_settings = store.get("meta", "settings")
    if stored_settings is None:
        store.put("meta", "settings", defaults)
    else:
        patch = {"data_root": str(store.root)}
        # The old default was disabled. Upgrade that untouched legacy default once,
        # while preserving an explicit choice made after this marker exists.
        if store.get("meta", "gpu_monitor_initialized") is None:
            patch.update(gpu_monitor=True)
            store.put("meta", "gpu_monitor_initialized", dict(initialized_at=now()))
        store.patch("meta", "settings", patch)
    settings = lambda: store.get("meta", "settings")
    engines = EngineService(store, settings, paths.project_root)
    installations = InstallationService(store, engines, paths.project_root)
    datasets = DatasetService(store)
    prompt_templates = PromptTemplateService(store)
    training = TrainingService(store, engines, datasets)
    monitor = SystemMonitor(store, settings)

    owned_workers = []

    def start_supervisor():
        process = ensure_supervisor(store.root, paths.project_root)
        if process is not None:
            owned_workers.append(process)

    # Artifact rows are an index of what the output folders held; the files
    # themselves are user data. Sweep the index at startup and then periodically so
    # checkpoints deleted/moved by hand stop lingering in the database.
    ARTIFACT_PRUNE_SECONDS = 6 * 3600

    async def maintain_artifact_index():
        while True:
            try:
                await asyncio.to_thread(prune_missing_artifacts, store)
            except Exception:  # noqa: BLE001 - maintenance must never break the API
                pass
            await asyncio.sleep(ARTIFACT_PRUNE_SECONDS)

    @asynccontextmanager
    async def lifespan(app):
        training.reconcile_orphans()
        maintenance = None
        if start_worker:
            await asyncio.to_thread(start_supervisor)
            maintenance = asyncio.create_task(maintain_artifact_index())
        try:
            yield
        finally:
            await asyncio.to_thread(installations.shutdown)
            if maintenance is not None:
                maintenance.cancel()
                try:
                    await maintenance
                except BaseException:  # noqa: BLE001 - only the cancelled sweep
                    pass
        if start_worker and store.get("meta", "shutdown"):
            # Do not let a rapid relaunch clear the marker while the old worker
            # still owns its lock. This does not signal unrelated workers.
            await asyncio.to_thread(wait_supervisor_exit, store.root, processes=tuple(owned_workers))
        # An unexpected API exit still leaves active work under its supervisor.

    app = FastAPI(title="LP LoRA Trainer", version=APP_VERSION, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.engines, app.state.datasets, app.state.training = store, engines, datasets, training
    app.state.installations = installations
    app.state.session_token = token
    # Set by ``main()`` in real runs; ``None`` means this process cannot be asked to exit.
    app.state.server = None

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        # ``payload`` lets a refusal explain itself in machine-readable form (the
        # shutdown endpoint uses it) without changing the ``detail`` contract.
        body = {"detail": exc.message}
        if getattr(exc, "payload", None):
            body.update(exc.payload)
        return JSONResponse(body, status_code=exc.status)

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

    @app.get("/api/v1/system/hardware")
    def system_hardware(): return hardware.probe()

    @app.get("/api/v1/compat")
    def engine_compat_manifest(): return load_manifest()

    def uv_cache_root():
        """The project's uv download cache; nothing else is ever cleared."""
        root = (paths.project_root / "runtime" / "uv" / "cache").resolve()
        if root.name != "cache" or root.parent.name != "uv":
            raise ServiceError("uv 缓存目录异常，已拒绝操作")
        return root

    @app.get("/api/v1/system/uv-cache")
    def uv_cache():
        root = uv_cache_root()
        files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
        return dict(path=str(root), exists=root.is_dir(), count=len(files),
                    bytes=sum(p.stat().st_size for p in files))

    @app.delete("/api/v1/system/uv-cache")
    def uv_cache_clear():
        """Clear the uv download cache; only future installs are affected."""
        import shutil
        root = uv_cache_root()
        if not root.is_dir():
            return dict(removed=0, bytes=0)
        files = [p for p in root.rglob("*") if p.is_file()]
        total, removed = sum(p.stat().st_size for p in files), len(files)
        shutil.rmtree(root)
        return dict(removed=removed, bytes=total)

    @app.post("/api/v1/system/shutdown", status_code=202)
    def shutdown(force: bool = Query(False)):
        """Ask the backend and the independent supervisor to exit gracefully.

        Refused while training is active or an orphan is unverified, so this can
        never be a back door for killing a job. ``force`` additionally cancels a
        running environment installation — an install is resumable, training is
        not — which is what lets the window close while a stuck install runs.
        """
        states = {t["state"] for t in store.list("task")}
        # Whoever asks (the window's close hook) gets the reason, not just a refusal.
        blocking = [{"name": t.get("name"), "state": t.get("state")} for t in store.list("task")
                    if t.get("state") in ACTIVE or t.get("state") == "connection_lost"]
        if states & ACTIVE:
            raise ServiceError("仍有训练任务处于活动状态，已拒绝退出；请先等待任务结束或停止任务", 409,
                               payload={"shutdown": {"reason": "training", "tasks": blocking}})
        if "connection_lost" in states:
            raise ServiceError("存在连接丢失、需人工核实的任务，已拒绝退出", 409,
                               payload={"shutdown": {"reason": "unverified", "tasks": blocking}})
        from app.services.installation import RECOVERABLE
        installing = [i for i in store.list("installation") if i.get("state") in RECOVERABLE]
        if installing and not force:
            raise ServiceError("仍有环境安装任务，请先取消或等待安装完成后退出", 409,
                               payload={"shutdown": {"reason": "install", "tasks": [
                                   {"name": i.get("engine_id") or i.get("installation_id"), "state": i.get("state")}
                                   for i in installing]}})
        if installing:
            installations.shutdown()
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
        store.put("meta", "gpu_monitor_initialized", dict(initialized_at=now()))
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
        value["proxy_url"] = validate_proxy_url(value.get("proxy_url"))
        return store.put("meta", "settings", value)

    @app.get("/api/v1/adapters")
    def adapters_catalog():
        from adapters.registry import adapter_catalog
        return adapter_catalog()

    @app.get("/api/v1/engines")
    def engine_list(): return engines.list()

    @app.post("/api/v1/engines/rescan")
    def engine_rescan(): return engines.rescan()

    @app.post("/api/v1/engines/clone", status_code=202)
    def engine_clone(body: EngineCloneInput):
        # Refuse while training runs: a new engine folder must not race task bindings.
        if any(t.get("state") in ACTIVE for t in store.list("task")):
            raise ServiceError("有训练任务正在运行，暂不能拉取引擎", 409)
        return installations.clone(body.url, body.name, body.ref, body.use_proxy, body.commit)

    @app.post("/api/v1/engines/{key}/default", status_code=204)
    def engine_default(key: str): engines.set_default(key)

    @app.post("/api/v1/engines/{key}/repair", status_code=202)
    def engine_repair(key: str, body: SourceRepairInput):
        return installations.repair_source(key, body.use_proxy)

    @app.post("/api/v1/engines/{key}/update", status_code=202)
    def engine_update(key: str):
        """Bring the clone to its upstream reference: git fetch + merge --ff-only only."""
        return installations.start_update(key)

    @app.post("/api/v1/engines/{key}/diagnose", status_code=204)
    def engine_diagnose(key: str): engines.diagnose(key)

    @app.delete("/api/v1/engines/{key}", status_code=204)
    def engine_remove(key: str): engines.remove(key)

    @app.put("/api/v1/engines/{key}/python", status_code=204)
    def engine_python(key: str, body: PythonInput): engines.bind(key, body.python_executable)

    @app.put("/api/v1/engines/{key}/engine-type", status_code=204)
    def engine_type(key: str, body: EngineTypeInput): engines.confirm(key, body.engine_id)

    @app.post("/api/v1/engines/{key}/install", status_code=202)
    def engine_install(key: str, body: InstallEnvironmentInput):
        return installations.start(
            key,
            body.python_executable,
            body.torch_source,
            body.mirror_url,
            body.torch_mirror_url,
            body.python_mirror_url,
            body.use_proxy,
            body.extras,
        )

    @app.get("/api/v1/terminal/sessions")
    def terminal_sessions(): return installations.list()

    @app.get("/api/v1/terminal/sessions/{key}/log")
    def terminal_log(key: str, offset: int = Query(0, ge=0)):
        return installations.read_log(key, offset)

    @app.get("/api/v1/terminal/sessions/{key}/diagnosis")
    def terminal_diagnosis(key: str):
        return installations.diagnosis(key)

    @app.delete("/api/v1/terminal/sessions/{key}", status_code=204)
    def terminal_delete(key: str): installations.delete(key)

    @app.post("/api/v1/terminal/sessions/{key}/stop", status_code=204)
    def terminal_stop(key: str): installations.cancel(key)

    @app.post("/api/v1/engines/{key}/terminal", status_code=204)
    def engine_terminal(key: str, body: TerminalInput):
        """Open the interpreter console from browser mode.

        The desktop window uses the pywebview bridge; this lets the same button
        work when LP runs in the terminal + browser fallback.
        """
        record = engines.get(key)
        python = body.python.strip() or str(record.get("python_executable") or "")
        if not python:
            raise ServiceError("当前引擎没有绑定 Python 解释器", 409)
        from desktop.bridge import DesktopBridge
        try:
            DesktopBridge().open_terminal(python, body.arguments)
        except ValueError as exc:
            raise ServiceError(str(exc), 422) from exc
        return None

    @app.get("/api/v1/engines/{key}/capabilities")
    def engine_capabilities(key: str): return engines.capabilities(key)

    @app.get("/api/v1/engines/{key}/install-plan")
    def engine_install_plan(key: str, torch_source: str | None = Query(None)):
        return engines.install_plan(key, torch_source)

    @app.get("/api/v1/base-models")
    def base_models(): return engines.base_models()

    @app.get("/api/v1/prompt-templates")
    def prompt_template_list(): return prompt_templates.list()

    @app.get("/api/v1/prompt-templates/{key}")
    def prompt_template_get(key: str): return prompt_templates.get(key)

    @app.post("/api/v1/prompt-templates", status_code=201)
    def prompt_template_add(body: PromptTemplateInput): return prompt_templates.save(body.name, body.prompt)

    @app.put("/api/v1/prompt-templates/{key}")
    def prompt_template_update(key: str, body: PromptTemplateInput): return prompt_templates.save(body.name, body.prompt, key)

    @app.delete("/api/v1/prompt-templates/{key}", status_code=204)
    def prompt_template_delete(key: str): prompt_templates.delete(key)

    @app.get("/api/v1/datasets")
    def dataset_list(sort: str = "time"): return datasets.list(sort)

    @app.post("/api/v1/datasets")
    def dataset_add(body: DatasetInput): return datasets.add(body.path, body.name)

    @app.delete("/api/v1/datasets/{key}", status_code=204)
    def dataset_remove(key: str): datasets.remove(key)

    @app.post("/api/v1/datasets/{key}/scan", status_code=204)
    def dataset_scan(key: str): datasets.scan(key)

    @app.get("/api/v1/datasets/{key}/images")
    def dataset_images(key: str, offset: int = Query(0, ge=0), limit: int = Query(48, ge=1, le=200), issue: str | None = None):
        if issue and issue not in ("corrupt", "missing_caption", "odd_size", "duplicate"): raise ServiceError("Unknown image filter")
        return datasets.images(key, offset, limit, issue)

    @app.put("/api/v1/datasets/{key}/images/{image_id}/caption", status_code=204)
    def caption_save(key: str, image_id: str, body: CaptionInput): datasets.caption(key, image_id, body.caption, body.caption_nl)

    @app.get("/api/v1/datasets/{key}/images/{image_id}/thumbnail")
    def thumbnail(key: str, image_id: str): return FileResponse(datasets.thumbnail(key, image_id), media_type="image/jpeg")

    @app.get("/api/v1/datasets/{key}/images/{image_id}/full")
    def image_full(key: str, image_id: str):
        # A screen-sized JPEG for the viewer; the original file is never served or modified.
        return FileResponse(datasets.thumbnail(key, image_id, FULL_IMAGE_SIZE), media_type="image/jpeg")

    @app.post("/api/v1/training/export-parameters")
    def export_parameters(body: TrainingExportInput): return training.export_parameters(body.model_dump())

    @app.post("/api/v1/training/validate")
    def validate(body: TrainingInput): return training.validate(body.model_dump())

    @app.get("/api/v1/training/draft")
    def training_draft(): return training.draft()

    @app.put("/api/v1/training/draft")
    def save_training_draft(body: TrainingDraftInput): return training.save_draft(body.model_dump())

    @app.delete("/api/v1/training/draft", status_code=204)
    def clear_training_draft(): training.clear_draft()

    @app.post("/api/v1/training/submit")
    def submit(body: TrainingInput):
        # A graceful exit is already in flight; never accept work no supervisor will claim.
        if store.get("meta", "shutdown"):
            raise ServiceError("后端与监管进程正在退出，无法提交新任务；请重新启动应用", 409)
        if start_worker and supervisor_status(store.root) == "unreachable":
            start_supervisor()
        if start_worker and any(t["state"] == "connection_lost" for t in store.list("task")):
            raise ServiceError("A task lost its supervisor connection; verify the training process before continuing", 409)
        return training.submit(body.model_dump())

    @app.get("/api/v1/tasks")
    def task_list(): return training.tasks()

    @app.get("/api/v1/tasks/{key}/configuration")
    def task_configuration(key: str): return training.configuration(key)

    @app.delete("/api/v1/tasks/{key}", status_code=204)
    def task_delete(key: str): training.delete(key)

    @app.get("/api/v1/tasks/{key}/metrics")
    def metrics(key: str): return training.metrics(key)

    @app.get("/api/v1/tasks/{key}/log")
    def logs(key: str, tail: int = Query(500, ge=1, le=10000)): return training.log(key, tail)

    @app.post("/api/v1/tasks/{key}/stop", status_code=204)
    def stop(key: str, body: StopInput): training.stop(key, body.force)

    @app.post("/api/v1/tasks/{key}/acknowledge-exit", status_code=204)
    def acknowledge_exit(key: str, body: AcknowledgeInput): training.acknowledge_exit(key)

    @app.get("/api/v1/tasks/{key}/resume")
    def task_resume_info(key: str): return training.resume_info(key)

    @app.post("/api/v1/tasks/{key}/resume", status_code=201)
    def task_resume(key: str, body: ResumeInput): return training.resume(key, body.source)

    @app.get("/api/v1/artifacts/tasks")
    def artifact_task_history(): return training.result_history()

    @app.get("/api/v1/artifacts")
    def artifacts(task_id: str | None = None): return training.artifacts(task_id)

    @app.post("/api/v1/artifacts/{key}/publish", status_code=204)
    def publish(key: str, body: PublishInput): training.publish(key, body.target_dir, body.file_name)

    @app.get("/api/v1/artifacts/{key}/preview")
    def preview(key: str, full: bool = False):
        artifact = store.get("artifact", key)
        if not artifact or artifact["kind"] not in {"sample", "video"} or not artifact["complete"]:
            raise ServiceError("Preview is unavailable", 404)
        path = contained(artifact["path"], training.artifact_output_dir(artifact["task_id"]))
        if not path.is_file(): raise ServiceError("Sample image no longer exists", 404)
        if artifact["kind"] == "video":
            media = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
            return FileResponse(path, media_type=media)
        from PIL import Image, ImageOps
        import io
        with Image.open(path) as image:
            limit = FULL_IMAGE_SIZE if full else 768
            image.thumbnail((limit, limit))
            im = ImageOps.exif_transpose(image).convert("RGB")
            data = io.BytesIO(); im.save(data, "JPEG", quality=90 if full else 85)
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
                deleted = set(previous_tasks) - {t["task_id"] for t in tasks}
                for key in deleted:
                    previous_tasks.pop(key, None)
                    cursors.pop(key, None)
                    metric_cursors.pop(key, None)
                    yield "data: " + json.dumps(dict(kind="task.deleted", task_id=key)) + "\n\n"
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
        class _FrontendFiles(StaticFiles):
            """Serve the built UI; never let the entry document go stale.

            A hashed bundle plus a cached ``index.html`` is how "the file changed but
            the window still shows the old UI" happens: the entry keeps pointing at the
            previous chunk. Only the document is revalidated; assets keep their long
            cache because their names carry a content hash.
            """

            async def get_response(self, path, scope):
                response = await super().get_response(path, scope)
                if path in ("", ".", "index.html"):
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                return response

        app.mount("/", _FrontendFiles(directory=paths.frontend_dist, html=True), name="frontend")
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
