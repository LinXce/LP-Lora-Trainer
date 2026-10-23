import json
import hashlib
import math
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from adapters.sampling import PROMPT_TEXT_LIMIT
from app.services.common import ServiceError, local_path, now, atomic_text, identity, contained
from app.services.engines import revision
from app.services.artifacts import checkpoint_complete

ACTIVE = {"queued", "preparing", "running", "stopping"}
# A task can continue from a save point only once it definitely is not running:
# succeeded runs are finished, and connection_lost must be acknowledged first.
RESUMABLE = {"stopped", "failed", "interrupted"}
PROGRESS = dict(step=None, total_steps=None, epoch=None, loss=None, lr=None, it_per_sec=None, eta_seconds=None)
DRAFT_KEY = "training_draft"
# Bounds for a staged form. Drafts are convenience data, so they are never
# allowed to grow unbounded or to store values the engines could not accept.
DRAFT_LIMITS = {"name": 200, "output_name": 200, "installation_id": 500, "architecture": 200,
                "base_model_path": 4096, "dataset_id": 500, "output_dir": 4096}
DRAFT_TEXT_LIMIT = 2048
DRAFT_PARAM_LIMIT = 500
DRAFT_ARCHITECTURE_LIMIT = 64


def safe_output_folder_name(value):
    """Return a Windows-safe task folder name without changing the task label."""
    value = str(value or "").strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(" .")[:200] or "training"
    if value.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }:
        value = f"_{value}"
    return value


class TrainingService:
    def __init__(self, store, engines, datasets):
        self.store, self.engines, self.datasets = store, engines, datasets
        self.lock = threading.RLock()

    def draft(self):
        """The staged training form, shared by every client of this data root."""
        stored = self.store.get("meta", DRAFT_KEY)
        if not isinstance(stored, dict) or not isinstance(stored.get("draft"), dict):
            return dict(draft=None, params_by_architecture={}, updated_at=None)
        draft = stored.get("draft")
        if isinstance(draft, dict) and not str(draft.get("output_dir") or "").strip():
            # Default output lives in the project, not in an absolute path the UI would
            # have to guess: the data root's parent is the project directory.
            stored = dict(stored)
            stored["draft"] = dict(draft, output_dir=str(Path(self.store.root).parent / "output"))
        return stored

    @staticmethod
    def _draft_text(value, field):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ServiceError(f"暂存内容格式不正确：{field}")
        if len(value) > DRAFT_LIMITS[field]:
            raise ServiceError(f"暂存内容过长：{field}", 422)
        return value

    @staticmethod
    def _draft_params(value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ServiceError("暂存参数格式不正确")
        if len(value) > DRAFT_PARAM_LIMIT:
            raise ServiceError("暂存参数过多，请先清理后再保存", 422)
        out = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not key or len(key) > 200:
                continue
            if raw is None or isinstance(raw, bool):
                out[key] = raw
            elif isinstance(raw, str):
                if len(raw) > (PROMPT_TEXT_LIMIT if key == "sample_prompts" else DRAFT_TEXT_LIMIT):
                    raise ServiceError(f"暂存参数值过长：{key}", 422)
                out[key] = raw
            elif isinstance(raw, (int, float)) and math.isfinite(raw):
                out[key] = raw
            # Anything else (lists, objects, NaN/Inf) is dropped rather than stored.
        return out

    def _draft_architecture_params(self, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ServiceError("暂存参数格式不正确")
        if len(value) > DRAFT_ARCHITECTURE_LIMIT:
            raise ServiceError("暂存底模类型过多，请先清理后再保存", 422)
        out = {}
        for architecture, params in value.items():
            if not isinstance(architecture, str) or not architecture or len(architecture) > 64:
                continue
            out[architecture] = self._draft_params(params)
        return out

    def save_draft(self, payload):
        record = dict(
            draft=dict(
                name=self._draft_text(payload.get("name"), "name"),
                output_name=self._draft_text(payload.get("output_name"), "output_name"),
                installation_id=self._draft_text(payload.get("installation_id"), "installation_id"),
                architecture=self._draft_text(payload.get("architecture"), "architecture"),
                base_model_path=self._draft_text(payload.get("base_model_path"), "base_model_path"),
                dataset_id=self._draft_text(payload.get("dataset_id"), "dataset_id"),
                output_dir=self._draft_text(payload.get("output_dir"), "output_dir"),
                params=self._draft_params(payload.get("params")),
            ),
            params_by_architecture=self._draft_architecture_params(payload.get("params_by_architecture")),
            updated_at=now(),
        )
        return self.store.put("meta", DRAFT_KEY, record)

    def clear_draft(self):
        self.store.delete("meta", DRAFT_KEY)

    def tasks(self):
        return [{k:v for k,v in t.items() if k not in ("binding", "draft", "stop_requested", "pid")} for t in reversed(self.store.list("task"))]

    def reconcile_orphans(self):
        """End tasks a previous run left active.

        At backend startup no supervisor exists yet, so an active task can only be
        a leftover from a window that was closed (or killed) while it ran. Leaving
        it "running" made the next start reuse or orphan state that no process
        could ever finish.
        """
        recovered = []
        with self.store.connection() as db:
            keys = [row[0] for row in db.execute(
                "select id from records where kind = 'task'")]
        for key in keys:
            task = self.store.get("task", key)
            if not isinstance(task, dict) or task.get("state") not in ACTIVE:
                continue
            if self.store.patch("task", key,
                                dict(state="interrupted", finished_at=now(), pid=None),
                                expected={"state": task["state"]}):
                recovered.append(key)
        return recovered

    def task(self, key):
        t = self.store.get("task", key)
        if not t: raise ServiceError("任务不存在", 404)
        return t

    def configuration(self, key):
        """Return the submitted form, not the engine-native/rendered config."""
        t = self.task(key)
        saved = t.get("draft")
        if not isinstance(saved, dict):
            raise ServiceError("历史任务没有保存可复用的配置", 404)
        draft = {field: saved.get(field, "") for field in DRAFT_LIMITS}
        # Tasks created before output_name existed use the task name as the model prefix.
        draft["output_name"] = saved.get("output_name") or saved.get("name", "")
        draft["params"] = dict(saved.get("params") or {})
        # Earlier tasks store the resolved per-job output directory. A new
        # submission must use the user's output root, never nest in an old job.
        if draft["output_dir"] == t["output_dir"]:
            draft["output_dir"] = str(Path(t["output_dir"]).parent)
        return dict(format="lp-training-parameters", version=1,
                    engine_id=t["engine_id"], dataset_path=saved.get("dataset_path"), **draft)

    def delete(self, key):
        with self.lock:
            t = self.task(key)
            if t["state"] not in {"succeeded", "failed", "stopped"} or t.get("pid"):
                raise ServiceError("只能删除已结束的任务；连接丢失的任务请先确认进程已退出", 409)
            if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
                raise ServiceError("无效的任务 ID", 422)
            target = self.store.root / "jobs" / key
            job = contained(target, self.store.root)
            if job != target.absolute():
                raise ServiceError("任务日志目录不能通过链接指向其他目录", 403)
            if job.exists():
                shutil.rmtree(job)
            # Keep the output boundary for existing preview/publish records.
            # We never delete model weights, samples, input data or engine files.
            with self.store.connection() as db:
                db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                           ("task_output", key, json.dumps({k: v for k, v in t.items() if k not in ("binding", "draft", "stop_requested", "pid")}, ensure_ascii=False)))
                db.execute("DELETE FROM records WHERE kind='task' AND id=?", (key,))

    def artifact_output_dir(self, key):
        record = self.store.get("task", key) or self.store.get("task_output", key)
        if not record:
            raise ServiceError("任务输出记录不存在", 404)
        return record["output_dir"]

    def export_parameters(self, payload):
        """Render a portable native parameter profile, even without an installation.

        Export is read-only and never asserts that this profile can start a job.
        Actual submission still requires the full validation and environment.
        """
        adapter = ADAPTERS.get(payload.get("engine_id"))
        if not adapter:
            raise ServiceError("找不到对应引擎适配器，仍可导出 LP 完整备份", 422)
        config = {key: self._draft_text(payload.get(key), key) for key in DRAFT_LIMITS}
        config["params"] = self._draft_params(payload.get("params"))
        config["dataset_path"] = payload.get("dataset_path") or ""
        try:
            text, warnings = adapter.export_parameters(config)
            files = adapter.export_parameter_files(config)
        except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
            raise ServiceError(f"无法生成原生参数：{exc}", 422) from exc
        if adapter.native_format not in {"toml", "yaml", "json"}:
            raise ServiceError("适配器声明的导出格式不受支持", 422)
        return dict(engine_id=adapter.engine_id, engine_name=adapter.display_name or adapter.engine_id,
                    native_format=adapter.native_format, text=text, warnings=list(warnings),
                    files=files if isinstance(files, list) else [])

    def validate(self, draft):
        issues = []
        def issue(field, message, level="error"):
            issues.append(dict(field=field, message=message, level=level))
        try:
            r = self.engines.get(draft["installation_id"])
        except ServiceError:
            r = None
            issue("installation_id", "请选择已登记的引擎实例")
        if r and r["state"] != "ready": issue("installation_id", "引擎环境尚未就绪，请先诊断")
        if not draft["name"].strip(): issue("name", "请输入任务名称")
        if not r or not r["engine_id"]:
            issue("installation_id", "引擎类型未确认")
            adapter = None
        else:
            adapter = ADAPTERS[r["engine_id"]]
            for i in adapter.validate(domain(r), draft): issue(i.field, i.message, i.severity)
        dataset = None
        try:
            dataset = self.datasets.get(draft["dataset_id"])
            if not Path(dataset["path"]).is_dir(): issue("dataset_id", "数据集路径已不存在")
            elif dataset["image_count"] is None: issue("dataset_id", "请先扫描数据集")
            elif not dataset["image_count"]: issue("dataset_id", "数据集没有可用图片")
            for i in dataset["issues"]:
                if i["kind"] == "corrupt": issue("dataset_id", "数据集包含损坏图片，请先移除并重新扫描")
                elif i["kind"] == "missing_caption": issue("dataset_id", "部分图片没有 caption，将按无文本样本训练", "warning")
            images = self.store.get("dataset_images", draft["dataset_id"], [])
            if adapter:
                for i in adapter.validate_dataset(domain(r), dataset, images):
                    issue(i.field, i.message, i.severity)
        except ServiceError:
            issue("dataset_id", "请选择已登记的数据集")
        if dataset and self.datasets.changed_since_scan(draft["dataset_id"]):
            issue("dataset_id", "数据集在上次扫描后有改动，建议重新扫描后再提交", "warning")
            if adapter:
                params = draft.get("params", {})
                for field, message in adapter.caption_cache_notices():
                    # Only when the caption-dependent cache is actually switched on.
                    if params.get(field) not in (None, "", False):
                        issue(field, message, "warning")
        # MOSS-style base models train from a model directory, so the generic
        # base-model file is not part of that form and must not be required here.
        optional_base = set(getattr(adapter, "base_model_optional_architectures", ())) if adapter else set()
        for field in ("base_model_path", "output_dir"):
            if field == "base_model_path" and draft.get("architecture") in optional_base:
                continue
            try:
                p = local_path(draft[field])
                if field == "base_model_path" and not p.exists(): issue(field, "基础模型路径不存在")
                if field == "output_dir":
                    if p.exists() and not p.is_dir(): issue(field, "输出路径不是目录")
                    # Never write generated training data into copied engine or input dataset/model.
                    protected = [Path(e["source_path"]) for e in self.engines.list()]
                    protected.append(self.store.root)
                    try:
                        model = local_path(draft["base_model_path"])
                        protected.append(model if model.is_dir() else model.parent)
                    except ServiceError: pass
                    if dataset: protected.append(Path(dataset["path"]))
                    if any(p == q.resolve() or p.is_relative_to(q.resolve()) for q in protected): issue(field, "输出目录不能位于引擎源码、数据集、基础模型或应用数据目录内部")
            except ServiceError as exc: issue(field, exc.message)
        ok = not any(i["level"] == "error" for i in issues)
        native, argv, pipeline = None, None, None
        if ok and adapter and dataset:
            preview = self.store.root / "jobs" / "<task-id>"
            config = {**draft, "dataset_path": dataset["path"], "output_dir": str(Path(draft["output_dir"]) / safe_output_folder_name(draft.get("name")))}
            try:
                native = adapter.native_config_text(config, preview / "dataset.toml")
                commands = adapter.preview_commands(domain(r), config, preview / f"config.native.{adapter.native_format}")
                argv = list(commands[-1])
                pipeline = [list(command) for command in commands]
            except (NotImplementedError, KeyError, ValueError):
                native, argv, pipeline = None, None, None
        return dict(ok=ok, issues=issues, native_config=native,
                    native_format=adapter.native_format if native else None, argv=argv, pipeline=pipeline,
                    submittable=bool(adapter and adapter.submittable))

    def submit(self, draft):
        with self.lock, self.engines.lock:
            self.engines.require_idle()
            result = self.validate(draft)
            if not result["ok"]: raise ServiceError("；".join(i["message"] for i in result["issues"] if i["level"] == "error"), 422)
            r = self.engines.get(draft["installation_id"])
            current = revision(Path(r["source_path"]))
            if current != r["revision"]: raise ServiceError("引擎源码已变化，请重新扫描和诊断", 409)
            adapter = ADAPTERS[r["engine_id"]]
            if not adapter.submittable:
                raise ServiceError("该引擎当前仅支持配置预览，尚不能提交训练", 409)
            key = uuid.uuid4().hex
            job = self.store.root / "jobs" / key
            job.mkdir(parents=True)
            # An empty output directory falls back to the project's output/, the same
            # default the draft read returns; local_path() would otherwise reject it.
            output_root = str(draft.get("output_dir") or "").strip() or str(Path(self.store.root).parent / "output")
            config = {**draft, "dataset_path": self.datasets.get(draft["dataset_id"])["path"],
                      "output_dir": str(local_path(output_root) / safe_output_folder_name(draft.get("name")))}
            try:
                native = adapter.write_native_config(domain(r), config, job)
                launch = adapter.build_launch(domain(r), native)
                atomic_text(job / "job.json", json.dumps(config, ensure_ascii=False, indent=2))
                atomic_text(job / "command.json", json.dumps(dict(commands=[list(a) for a in launch.commands()], cwd=str(launch.cwd)), ensure_ascii=False, indent=2))
                manifest_id = r.get("environment_manifest_id")
                if not manifest_id: raise ServiceError("环境清单不存在，请重新诊断", 409)
                binding = dict(installation_id=r["installation_id"], revision=r["revision"], environment_id=r["environment_id"],
                               environment_manifest_id=manifest_id, adapter_version=adapter.adapter_version, schema_version="1", installation=r,
                               config_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in job.iterdir() if p.is_file()})
                summary = dict(task_id=key, name=draft["name"].strip(), state="queued", engine_id=r["engine_id"], installation_id=r["installation_id"],
                    installation_label=r["label"], architecture=draft["architecture"], dataset_name=self.datasets.get(draft["dataset_id"])["name"],
                    output_dir=config["output_dir"], created_at=now(), started_at=None, finished_at=None, progress=PROGRESS.copy(), error_summary=None,
                    recovery=self._recovery_record(adapter, domain(r), draft["architecture"]), binding=binding, draft=config, stop_requested=None, pid=None)
                atomic_text(job / "binding.json", json.dumps(binding, ensure_ascii=False, indent=2))
                self.store.put("task", key, summary)
                return next(t for t in self.tasks() if t["task_id"] == key)
            except Exception:
                # Delete only files created in this new job directory, never user inputs.
                for p in job.iterdir():
                    if p.is_file(): p.unlink()
                job.rmdir()
                raise

    def stop(self, key, force):
        t = self.task(key)
        if t["state"] not in ACTIVE: raise ServiceError("该任务已结束，不能再次停止", 409)
        if t["state"] == "queued":
            cancelled = self.store.patch("task", key, dict(state="stopped", finished_at=now(), stop_requested="stop"), expected={"state": "queued"})
            if cancelled: return
        current = self.task(key)
        if current["state"] not in ACTIVE: raise ServiceError("任务状态已变化，请刷新后重试", 409)
        self.store.patch("task", key, dict(stop_requested="force" if force else "stop"), expected={"state": current["state"]})

    @staticmethod
    def _recovery_record(adapter, installation, architecture):
        caps = adapter.recovery_capabilities(installation, architecture)
        return dict(graceful_stop=caps.graceful_stop, resume_from_weights=caps.resume_weights,
                    resume_full_state=caps.resume_full_state)

    def _resume_adapter(self, task):
        adapter = ADAPTERS.get(task.get("engine_id"))
        if adapter is None:
            raise ServiceError("任务引擎类型未知，无法断点续训", 409)
        try:
            record = self.engines.get(task["installation_id"])
        except ServiceError as exc:
            raise ServiceError("原任务的引擎安装实例已不存在，无法断点续训", 409) from exc
        return adapter, record

    def _resume_sources(self, adapter, task):
        job = self.store.root / "jobs" / task["task_id"]
        try:
            return tuple(adapter.resume_sources(job))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            # A damaged job directory only means "nothing to continue from".
            return ()

    @staticmethod
    def _same_path(left, right):
        try:
            return Path(left).resolve() == Path(right).resolve()
        except (OSError, ValueError, TypeError):
            return False

    def resume_info(self, key):
        """Whether a finished task can continue, and from which save points.

        Read-only: the page offers 继续 only when the engine declares a resume
        capability *and* the task folder still holds a complete save point.
        """
        task = self.task(key)
        empty = dict(graceful_stop=False, resume_from_weights=False, resume_full_state=False)
        try:
            adapter, record = self._resume_adapter(task)
        except ServiceError as exc:
            return dict(supported=False, capabilities=empty, sources=[], reason=exc.message)
        caps = adapter.recovery_capabilities(domain(record), task.get("architecture"))
        capabilities = dict(graceful_stop=caps.graceful_stop, resume_from_weights=caps.resume_weights,
                            resume_full_state=caps.resume_full_state)
        reason = None
        sources = ()
        if task["state"] not in RESUMABLE:
            reason = "只有已停止、失败或中断的任务可以继续"
        elif not (caps.resume_full_state or caps.resume_weights):
            reason = "该引擎没有声明断点续训能力"
        elif not self._resume_inputs_present(task):
            reason = "原任务的数据集已不在注册表中或路径已不存在，请先重新添加数据集再继续"
        else:
            sources = self._resume_sources(adapter, task)
            if not sources:
                reason = "没有找到可续训的保存点：请在训练参数中开启「保存训练状态」后重新训练"
        return dict(supported=bool(sources) and reason is None, capabilities=capabilities,
                    sources=[dict(kind=s.kind, path=str(s.path), label=s.label, step=s.step) for s in sources],
                    reason=reason)

    def _resume_inputs_present(self, task):
        """The parent run's dataset must still be registered and on disk."""
        dataset_id = (task.get("draft") or {}).get("dataset_id")
        if not dataset_id:
            return False
        try:
            dataset = self.datasets.get(dataset_id)
        except ServiceError:
            return False
        return Path(str(dataset.get("path") or "")).is_dir()

    def resume(self, key, source_path):
        """Create a new queued task that continues ``key`` from one save point.

        The task itself is never replayed: a fresh job directory and task record
        are created, the parent stays untouched, and the engine parameter that
        consumes the save point is injected by the adapter.
        """
        task = self.task(key)
        info = self.resume_info(key)
        if not info["supported"]:
            raise ServiceError(info["reason"] or "该任务当前不能继续", 409)
        adapter, _record = self._resume_adapter(task)
        target = next((s for s in self._resume_sources(adapter, task)
                       if self._same_path(s.path, source_path)), None)
        if target is None:
            raise ServiceError("所选保存点不属于该任务或已失效，请刷新后重试", 422)
        stored = task.get("draft") or {}
        if not stored.get("name") or not stored.get("dataset_id"):
            raise ServiceError("原任务没有保存可续训的配置", 409)
        params = dict(stored.get("params") or {})
        params.update(adapter.resume_parameters(target))
        draft = dict(name=stored["name"], output_name=stored.get("output_name") or "",
                     installation_id=task["installation_id"], architecture=task["architecture"],
                     base_model_path=stored.get("base_model_path") or "", dataset_id=stored["dataset_id"],
                     # submit() appends the sanitized task name again, so pass the parent
                     # to keep writing into the same run folder as the interrupted job.
                     output_dir=str(Path(task["output_dir"]).parent), params=params)
        created = self.submit(draft)
        self.store.patch("task", created["task_id"], dict(
            resumed_from=key,
            resume_source=dict(kind=target.kind, path=str(target.path), label=target.label, step=target.step),
        ))
        return next(t for t in self.tasks() if t["task_id"] == created["task_id"])

    def acknowledge_exit(self, key):
        t = self.task(key)
        if t["state"] != "connection_lost": raise ServiceError("仅能确认连接丢失的任务", 409)
        # Explicit user assertion, never kill/reuse a historical PID or replay the job.
        from supervisor.worker import register_artifacts
        register_artifacts(self.store, t)
        self.store.patch("task", key, dict(state="stopped", finished_at=now(), pid=None,
            error_summary="用户已人工确认原训练进程退出；未自动重跑任务"), expected={"state": "connection_lost"})

    def log(self, key, tail):
        self.task(key)
        p = self.store.root / "jobs" / key / "stdout.log"
        if not p.is_file(): return []
        # Read backwards in chunks, not the whole multi-GB training log.
        with p.open("rb") as f:
            f.seek(0, 2); pos = f.tell(); data = b""
            while pos and data.count(b"\n") <= tail:
                size = min(pos, 65536); pos -= size; f.seek(pos); data = f.read(size) + data
        return data.decode("utf-8", errors="replace").splitlines()[-tail:]

    def _fill_learning_rate(self, task, job_directory, points):
        """Attach explicit `lr` log values to old loss-only metric points.

        Metrics written before the learning-rate chart existed contain only loss.
        The raw stdout log still carries the values for those runs, so read them
        once per metrics request and fill the steps that already exist. No new
        points and no scheduler values are invented.
        """
        adapter = ADAPTERS.get(task.get("engine_id"))
        log = Path(job_directory) / "stdout.log"
        if adapter is None or not log.is_file():
            return
        try:
            if log.stat().st_size > 64 * 1024 * 1024:
                return
            lr_by_step = {}
            with log.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    for event in adapter.parse_log(line):
                        step = event.payload.get("step")
                        lr = event.payload.get("lr")
                        if isinstance(step, int) and isinstance(lr, (int, float)) and math.isfinite(lr):
                            lr_by_step[step] = lr
        except OSError:
            return
        for point in points:
            lr = lr_by_step.get(point.get("step"))
            if lr is not None:
                point["lr"] = lr

    def metrics(self, key):
        task = self.task(key)
        p = self.store.root / "jobs" / key / "metrics.jsonl"
        if not p.is_file(): return []
        result = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try: result.append(json.loads(line))
            except ValueError: pass
        # Bound chart payload; raw points remain on disk.
        stride = max(1, len(result) // 4000)
        points = result[::stride][-4000:]
        if points and all("lr" not in point for point in points):
            self._fill_learning_rate(task, p.parent, points)
        return points

    def result_history(self):
        # Only tasks whose files still exist: history follows the output folder too.
        artifact_tasks = {a["task_id"] for a in self.store.list("artifact") if Path(a["path"]).is_file()}
        return [{**t, "history_deleted": True} for t in reversed(self.store.list("task_output")) if t["task_id"] in artifact_tasks]

    def artifacts(self, task_id=None):
        """Results follow the output folder, not the records taken while running.

        The folder is re-collected on read (idempotent, keyed by path), and records
        whose file is gone are left out, so deleting or adding a checkpoint by hand
        is reflected immediately.
        """
        if not task_id:
            return list(self.store.list("artifact"))
        self.artifact_output_dir(task_id)
        # A deleted task keeps its records: there is nothing left to re-collect.
        task = self.store.get("task", task_id)
        job = self.store.root / "jobs" / task_id
        if task and job.is_dir():
            from supervisor.worker import register_artifacts
            try:
                register_artifacts(self.store, task)
            except Exception:
                pass  # a read must not fail because a folder became unreadable
        return [a for a in self.store.list("artifact")
                if a["task_id"] == task_id and Path(a["path"]).is_file()]

    def publish(self, key, target_dir, file_name):
        artifact = self.store.get("artifact", key)
        if not artifact: raise ServiceError("产物不存在", 404)
        if not artifact["complete"] or artifact["kind"] != "checkpoint": raise ServiceError("只能发布已完成的模型文件", 409)
        if Path(file_name).name != file_name or file_name in (".", "..") or any(c in file_name for c in '<>:\"/\\|?*'):
            raise ServiceError("请提供文件名，不允许路径分隔符或特殊字符")
        if not file_name.lower().endswith(".safetensors"): raise ServiceError("发布文件必须保留 .safetensors 扩展名")
        src = Path(artifact["path"])
        if not src.is_file(): raise ServiceError("源模型已不存在", 404)
        if not checkpoint_complete(src): raise ServiceError("源模型不完整或已被修改，无法发布", 409)
        target = local_path(target_dir)
        if not target.is_dir(): raise ServiceError("发布目录不存在")
        dst = contained(target / file_name, target)
        if dst.exists(): raise ServiceError("目标文件已存在，不会覆盖", 409)
        import tempfile
        fd, temp = tempfile.mkstemp(prefix=".lp-publish-", dir=target)
        os.close(fd)
        try:
            shutil.copyfile(src, temp)
            if not checkpoint_complete(Path(temp)):
                raise ServiceError("复制后的模型不完整，已取消发布，请检查源文件", 409)
            # Hard-link creates the complete destination atomically and fails if it already exists.
            try: os.link(temp, dst)
            except FileExistsError as exc: raise ServiceError("目标文件已存在，不会覆盖", 409) from exc
            except OSError as exc: raise ServiceError(f"目标文件系统不支持安全的原子发布：{exc}") from exc
        finally: Path(temp).unlink(missing_ok=True)
