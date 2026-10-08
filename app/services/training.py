import json
import hashlib
import math
import os
import shutil
import threading
import uuid
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from app.services.common import ServiceError, local_path, now, atomic_text, identity, contained
from app.services.engines import revision
from app.services.artifacts import checkpoint_complete

ACTIVE = {"queued", "preparing", "running", "stopping"}
PROGRESS = dict(step=None, total_steps=None, epoch=None, loss=None, it_per_sec=None, eta_seconds=None)
DRAFT_KEY = "training_draft"
# Bounds for a staged form. Drafts are convenience data, so they are never
# allowed to grow unbounded or to store values the engines could not accept.
DRAFT_LIMITS = {"name": 200, "installation_id": 500, "architecture": 200,
                "base_model_path": 4096, "dataset_id": 500, "output_dir": 4096}
DRAFT_TEXT_LIMIT = 2048
DRAFT_PARAM_LIMIT = 500
DRAFT_ARCHITECTURE_LIMIT = 64


class TrainingService:
    def __init__(self, store, engines, datasets):
        self.store, self.engines, self.datasets = store, engines, datasets
        self.lock = threading.RLock()

    def draft(self):
        """The staged training form, shared by every client of this data root."""
        stored = self.store.get("meta", DRAFT_KEY)
        if not isinstance(stored, dict) or not isinstance(stored.get("draft"), dict):
            return dict(draft=None, params_by_architecture={}, updated_at=None)
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
                if len(raw) > DRAFT_TEXT_LIMIT:
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

    def task(self, key):
        t = self.store.get("task", key)
        if not t: raise ServiceError("任务不存在", 404)
        return t

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
            # sd-scripts consumes direct children of image_dir; do not silently omit nested images.
            images = self.store.get("dataset_images", draft["dataset_id"], [])
            if any("/" in i["file_name"] for i in images):
                issue("dataset_id", "当前 Kohya 数据集适配要求图片位于数据集目录一级，请分别登记子目录")
        except ServiceError:
            issue("dataset_id", "请选择已登记的数据集")
        for field in ("base_model_path", "output_dir"):
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
            config = {**draft, "dataset_path": dataset["path"], "output_dir": str(Path(draft["output_dir"]) / "<task-id>")}
            try:
                native = adapter.render(config, preview / "dataset.toml")
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
            config = {**draft, "dataset_path": self.datasets.get(draft["dataset_id"])["path"], "output_dir": str(local_path(draft["output_dir"]) / key)}
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
                    recovery=dict(graceful_stop=False, resume_from_weights=False, resume_full_state=False), binding=binding, draft=config, stop_requested=None, pid=None)
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

    def metrics(self, key):
        self.task(key)
        p = self.store.root / "jobs" / key / "metrics.jsonl"
        if not p.is_file(): return []
        result = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try: result.append(json.loads(line))
            except ValueError: pass
        # Bound chart payload; raw points remain on disk.
        stride = max(1, len(result) // 4000)
        return result[::stride][-4000:]

    def artifacts(self, task_id=None):
        if task_id: self.task(task_id)
        return [a for a in self.store.list("artifact") if not task_id or a["task_id"] == task_id]

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
