import json
import hashlib
import inspect
import math
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from adapters.capabilities import form_value_errors
from adapters.registry import ADAPTERS, domain
from adapters.sampling import PROMPT_TEXT_LIMIT
from app.services.common import ServiceError, local_path, now, atomic_text, identity, contained, safe_file_stem
from app.services.engines import revision
from app.services import output_ownership, queue_control
from app.services.progress_history import ProgressHistory, chart_sample
from app.services.progress import PROGRESS

ACTIVE = {"queued", "preparing", "running", "stopping"}
# A task can continue from a save point only once it definitely is not running:
# succeeded runs are finished, and connection_lost must be acknowledged first.
RESUMABLE = {"stopped", "failed", "interrupted"}
DRAFT_KEY = "training_draft"
# Bounds for a staged form. Drafts are convenience data, so they are never
# allowed to grow unbounded or to store values the engines could not accept.
DRAFT_LIMITS = {"name": 200, "output_name": 200, "installation_id": 500, "architecture": 200,
                "base_model_path": 4096, "dataset_id": 500, "output_dir": 4096}
DRAFT_TEXT_LIMIT = 2048
DRAFT_PARAM_LIMIT = 500
DRAFT_ARCHITECTURE_LIMIT = 64
# Model buckets contain the installation ID, architecture and (possibly JSON-escaped) path.
DRAFT_BUCKET_KEY_LIMIT = DRAFT_LIMITS["installation_id"] + DRAFT_LIMITS["architecture"] + 2 * DRAFT_LIMITS["base_model_path"] + 32


def safe_output_folder_name(value):
    """Return a Windows-safe task folder name without changing the task label."""
    return safe_file_stem(value, "training")


class TrainingService:
    def __init__(self, store, engines, datasets):
        self.store, self.engines, self.datasets = store, engines, datasets
        self.lock = threading.RLock()
        self.progress_history = ProgressHistory(store.root)

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
                if len(raw) > (PROMPT_TEXT_LIMIT if key.lower() == "sample_prompts" else DRAFT_TEXT_LIMIT):
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
            if not isinstance(architecture, str) or not architecture:
                continue
            if len(architecture) > DRAFT_BUCKET_KEY_LIMIT:
                raise ServiceError("底模草稿标识过长", 422)
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
        private = {"binding", "draft", "stop_requested", "pid", "exit_savepoint_baseline"}
        return [{k: v for k, v in self.progress_history.project(t).items() if k not in private}
                for t in reversed(self.store.list("task"))]

    def task(self, key):
        t = self.store.get("task", key)
        if not t: raise ServiceError("任务不存在", 404)
        return self.progress_history.project(t)

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
            t = self.progress_history.archive(t)
            if job.exists():
                shutil.rmtree(job)
            # Keep the output boundary for retained artifact records and previews.
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
        installation = None
        accepts_installation = "installation" in inspect.signature(adapter.export_parameters).parameters
        if accepts_installation and config.get("installation_id"):
            try:
                installation = domain(self.engines.get(config["installation_id"]))
            except Exception:  # noqa: BLE001 - export still works without an installation
                installation = None
        kwargs = {}
        if installation is not None:
            kwargs["installation"] = installation
        try:
            text, warnings = adapter.export_parameters(config, **kwargs)
            files = adapter.export_parameter_files(config)
        except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
            raise ServiceError(f"无法生成原生参数：{exc}", 422) from exc
        if adapter.native_format not in {"toml", "yaml", "json"}:
            raise ServiceError("适配器声明的导出格式不受支持", 422)
        return dict(engine_id=adapter.engine_id, engine_name=adapter.display_name or adapter.engine_id,
                    native_format=adapter.native_format, text=text, warnings=list(warnings),
                    files=files if isinstance(files, list) else [])

    def validate(self, draft, *, task_id=None):
        issues = []
        # Render using a legal path component. Angle brackets are display-only.
        run_id = task_id or "LP-PREVIEW-TASK-ID"
        target = None
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
                    target = p / (safe_output_folder_name(draft.get("name")) + "--" + run_id)
            except ServiceError as exc: issue(field, exc.message)
        config = {**draft, "dataset_path": dataset["path"] if dataset else "",
                  "output_dir": str(target) if target else draft.get("output_dir", ""),
                  "output_layout": "task_id"}
        if adapter:
            installation = domain(r)
            architecture = config.get("architecture")
            if architecture not in {a.id for a in adapter.architectures(installation.source_path)}:
                issue("architecture", "当前引擎版本不支持所选底模类型")
            else:
                fields = adapter.capabilities_for_architecture(installation, architecture)["params"]
                malformed = list(form_value_errors(fields, config.get("params", {})))
                for field, message in malformed:
                    issue(field, message)
                if not malformed:
                    if target:
                        for field, message in output_ownership.write_path_issues(self.store, adapter, config, target):
                            issue(field, message)
                    for i in adapter.validate(installation, config):
                        issue(i.field, i.message, i.severity)
        ok = not any(i["level"] == "error" for i in issues)
        native, argv, pipeline = None, None, None
        if ok and adapter and dataset:
            preview_root = self.store.root / "jobs"
            preview_root.mkdir(parents=True, exist_ok=True)
            # "<task-id>" is not a valid Windows path; some adapters must create a
            # prompt/config file while rendering. Render in a disposable real folder
            # and put the display placeholder back afterwards.
            preview = Path(tempfile.mkdtemp(prefix="preview-", dir=preview_root))
            try:
                try:
                    native_kwargs = {}
                    if "installation" in inspect.signature(adapter.native_config_text).parameters:
                        native_kwargs["installation"] = domain(r)
                    native = adapter.native_config_text(config, preview / "dataset.toml", **native_kwargs)
                    commands = adapter.preview_commands(domain(r), config, preview / adapter.native_filename)
                    argv = list(commands[-1])
                    pipeline = [list(command) for command in commands]
                except (NotImplementedError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                    native, argv, pipeline = None, None, None
                    if adapter.submittable:
                        ok = False
                        issue(None, f"无法生成引擎训练配置：{exc}")
            finally:
                shutil.rmtree(preview, ignore_errors=True)
            if native is not None:
                marker = "<task-id>"
                def display(value):
                    value = str(value).replace(str(preview).replace("\\", "\\\\"), marker)
                    value = value.replace(str(preview), marker).replace(preview.as_posix(), marker)
                    return value if task_id else value.replace("--" + run_id, "--" + marker)
                native = display(native)
                argv = [display(item) for item in argv] if argv else argv
                pipeline = [
                    [display(item) for item in command]
                    for command in pipeline
                ] if pipeline else pipeline
        output_preview = str(target) if target else None
        if output_preview and not task_id:
            output_preview = str(target.parent / (safe_output_folder_name(draft.get("name")) + "--<task-id>"))
        return dict(ok=ok, issues=issues, native_config=native, output_dir_preview=output_preview,
                    native_format=adapter.native_format if native else None, argv=argv, pipeline=pipeline,
                    submittable=bool(adapter and adapter.submittable))

    def submit(self, draft, *, resumed_from=None, resume_source=None):
        with self.lock, self.engines.lock:
            queue_control.require_open(self.store)
            self.engines.require_training_idle()
            key = uuid.uuid4().hex
            result = self.validate(draft, task_id=key)
            if not result["ok"]: raise ServiceError("；".join(i["message"] for i in result["issues"] if i["level"] == "error"), 422)
            r = self.engines.get(draft["installation_id"])
            current = revision(Path(r["source_path"]))
            if current != r["revision"]: raise ServiceError("引擎源码已变化，请重新扫描和诊断", 409)
            adapter = ADAPTERS[r["engine_id"]]
            if not adapter.submittable:
                raise ServiceError("该引擎当前仅支持配置预览，尚不能提交训练", 409)
            target_output = Path(result["output_dir_preview"])
            job = self.store.root / "jobs" / key
            job.mkdir(parents=True)
            config = {**draft, "dataset_path": self.datasets.get(draft["dataset_id"])["path"],
                      "output_dir": str(target_output), "output_layout": "task_id"}
            marker = None
            try:
                marker = output_ownership.reserve(self.store, key, target_output)
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
                    recovery=self._recovery_record(adapter, domain(r), draft["architecture"]), binding=binding, draft=config,
                    output_verified=False, stop_requested=None, pid=None)
                if resumed_from:
                    summary.update(resumed_from=resumed_from, resume_source=resume_source)
                    binding.update(resumed_from=resumed_from, resume_source=resume_source)
                atomic_text(job / "binding.json", json.dumps(binding, ensure_ascii=False, indent=2))
                queue_control.enqueue(self.store, summary)
            except Exception:
                if marker is not None:
                    marker.unlink(missing_ok=True)
                shutil.rmtree(job)
                raise
            return next(t for t in self.tasks() if t["task_id"] == key)

    def stop(self, key, force):
        t = self.task(key)
        if t["state"] not in ACTIVE: raise ServiceError("该任务已结束，不能再次停止", 409)
        if t["state"] == "queued":
            cancelled = self.store.patch("task", key, dict(state="stopped", finished_at=now(), stop_requested="stop"), expected={"state": "queued"})
            if cancelled: return
        current = self.task(key)
        if current["state"] not in ACTIVE: raise ServiceError("任务状态已变化，请刷新后重试", 409)
        stopped = self.store.patch("task", key, dict(stop_requested="force" if force else "stop"), expected={"state": current["state"]})
        if stopped is None: raise ServiceError("任务状态已变化，请刷新后重试", 409)

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
            root = Path(task["output_dir"]).resolve()
            return tuple(source for source in adapter.resume_sources(job)
                         if source.path.resolve().is_relative_to(root))
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
        elif task.get("output_conflict"):
            reason = "任务启动时未取得输出目录归属，无法使用此目录中的保存点续训"
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
        with self.lock:
            return self._submit_resume(key, source_path)

    def _submit_resume(self, key, source_path):
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
                     output_dir=str(Path(task["output_dir"]).parent), params=params)
        source = dict(kind=target.kind, path=str(target.path), label=target.label, step=target.step)
        return self.submit(draft, resumed_from=key, resume_source=source)

    def acknowledge_exit(self, key):
        t = self.task(key)
        if t["state"] != "connection_lost": raise ServiceError("仅能确认连接丢失的任务", 409)
        # Explicit user assertion, never kill/reuse a historical PID or replay the job.
        from supervisor.worker import register_artifacts
        register_artifacts(self.store, t)
        changes = dict(state="stopped", finished_at=now(), pid=None,
                       error_summary="用户已人工确认原训练进程退出；未自动重跑任务")
        if t.get("exit_requested_at"):
            changes.update(exit_savepoint_baseline=None, stop_result=dict(
                status="unknown", sources=[], message="监管曾中断，原进程已人工确认退出，无法核实本次停止是否写入保存点。"))
        updated = self.store.patch("task", key, changes, expected={"state": "connection_lost"})
        if updated is None: raise ServiceError("任务状态已变化，请刷新后重试", 409)

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

    def metrics(self, key):
        task = self.task(key)
        p = self.store.root / "jobs" / key / "metrics.jsonl"
        result = []
        if p.is_file():
            with p.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        point = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(point, dict) and isinstance(point.get("step"), int):
                        result.append(point)
        return chart_sample(self.progress_history.corrected_points(task, result))

    def result_history(self):
        # Only tasks whose files still exist: history follows the output folder too.
        artifact_tasks = {a["task_id"] for a in self.store.list("artifact") if Path(a["path"]).is_file()}
        return [{**self.progress_history.project(t), "history_deleted": True}
                for t in reversed(self.store.list("task_output")) if t["task_id"] in artifact_tasks]

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
