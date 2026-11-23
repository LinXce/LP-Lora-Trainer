"""Reserve a run's canonical output path across LP data stores and restarts."""
import hashlib
import json
import os
from pathlib import Path

from app.services.common import ServiceError, local_path


RESERVATIONS = ".lp-reservations"


def _marker(target):
    key = hashlib.sha256(os.fsencode(os.path.normcase(target.name))).hexdigest()
    return target.parent / RESERVATIONS / (key + ".json")


def _owner(store, key, target):
    return dict(task_id=key, data_root=str(store.root), output_dir=str(target))


def check_boundary(store, target, key=None):
    if target.resolve() != target or RESERVATIONS in (os.path.normcase(part) for part in target.parts):
        raise ServiceError(f"输出目录路径已重定向或被 LP 保留：{target}", 409)
    for parent in target.parents:
        if parent.name and _marker(parent).exists():
            raise ServiceError(f"输出目录不能嵌套在其他任务目录中：{parent}", 409)
    # Includes deleted tasks whose output files and artifact records are retained.
    with store.connection() as db:
        rows = db.execute("SELECT id, json_extract(body,'$.output_dir') FROM records "
                          "WHERE kind IN ('task','task_output') AND id != ?", (key or "",))
        for other_key, value in rows:
            if value:
                other = Path(value).resolve()
                if target.is_relative_to(other) or other.is_relative_to(target):
                    raise ServiceError(f"输出目录与任务 {other_key} 的输出重叠：{other}", 409)


def write_path_issues(store, adapter, config, target, key=None):
    """Check the paths the adapter will actually use, including custom caches."""
    protected = [store.root]
    for kind, field in (("engine", "source_path"), ("dataset", "path")):
        protected.extend(Path(row[field]).resolve() for row in store.list(kind) if row.get(field))
    if config.get("dataset_path"):
        protected.append(Path(config["dataset_path"]).resolve())
    base_model = config.get("base_model_path")
    if base_model and Path(base_model).is_absolute():
        model = Path(base_model).resolve()
        protected.append(model if model.is_dir() else model.parent)
    # An auxiliary model file can sit alongside many independent output roots.
    # Protect the actual selected file/directory and its contents.
    protected.extend(Path(value).resolve() for value in adapter.model_input_paths(config).values()
                     if Path(value).is_absolute())

    paths = {"output_dir": str(target), **adapter.custom_write_dirs(config)}
    for field, value in paths.items():
        try:
            path = local_path(value)
            if any(path.is_relative_to(location) or location.is_relative_to(path) for location in protected):
                raise ServiceError("写入目录不能与引擎源码、数据集、模型或应用数据目录重叠", 409)
            if field != "output_dir":
                if (path == path.parent or _marker(path).exists() or (path / RESERVATIONS).exists()
                        or path.is_relative_to(target) or target.is_relative_to(path)):
                    raise ServiceError("自定义写入目录不能与任务输出目录重叠", 409)
            check_boundary(store, path, key)
        except (OSError, ValueError, RuntimeError, ServiceError) as exc:
            yield field, str(exc)


def reserve(store, key, target):
    """Exclusive sibling marker lets adapters create their own folder on first write."""
    check_boundary(store, target, key)
    if os.path.lexists(target):
        raise ServiceError(f"目标输出目录已经存在，已保留原文件：{target}", 409)
    marker = _marker(target)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.parent.resolve() != marker.parent:
            raise ServiceError(f"输出预留目录被重定向：{marker.parent}", 409)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(_owner(store, key, target), stream, ensure_ascii=False)
    except FileExistsError as exc:
        raise ServiceError(f"输出目录已被预留：{target}", 409) from exc
    except OSError as exc:
        raise ServiceError(f"无法预留输出目录 {target}：{exc}", 409) from exc
    return marker


def before_launch(store, task, job, adapter=None):
    target = Path(task["output_dir"])
    key = task["task_id"]
    config = json.loads((job / "job.json").read_text(encoding="utf-8"))
    if adapter is not None:
        for field, message in write_path_issues(store, adapter, config, target, key):
            raise ServiceError(f"{field}：{message}", 409)
    if Path(config["output_dir"]) != target or target.resolve() != target:
        raise ServiceError("任务输出目录与提交时绑定的配置不一致", 409)
    binding = task["binding"]
    if binding.get("resumed_from"):
        parent_id = binding["resumed_from"]
        parent = store.get("task", parent_id) or store.get("task_output", parent_id)
        source = Path(binding["resume_source"]["path"]).resolve()
        if not parent or not source.exists() or not source.is_relative_to(Path(parent["output_dir"]).resolve()):
            raise ServiceError("续训保存点已失效或位置已改变", 409)
    if config.get("output_layout") == "task_id":
        marker = _marker(target)
        if marker.resolve() != marker or not marker.is_file():
            raise ServiceError(f"输出目录预留记录缺失或被替换：{target}", 409)
        if json.loads(marker.read_text(encoding="utf-8")) != _owner(store, key, target):
            raise ServiceError(f"输出目录归属已变化：{target}", 409)
        check_boundary(store, target, key)
        if os.path.lexists(target):
            raise ServiceError(f"启动前发现目标目录已存在，已保留原文件：{target}", 409)
    else:
        # A queued pre-upgrade job may run only if its original target is
        # untouched and unclaimed. Its native configuration cannot be relocated.
        if adapter is not None:
            issue = adapter.legacy_output_issue(config)
            if issue:
                raise ServiceError(issue, 409)
        reserve(store, key, target)
