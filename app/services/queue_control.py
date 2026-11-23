"""Coordinate API writers and the independent worker using the same SQLite transaction."""
import json
import os
import time

from app.services.common import ServiceError, now


EXECUTING = {"preparing", "running", "stopping"}
SHUTDOWN_TIMEOUT = 300
EXIT_PROTOCOL = 1


def _get(db, kind, key):
    row = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
    return json.loads(row[0]) if row else None


def _put(db, kind, key, value):
    db.execute("INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",
               (kind, key, json.dumps(value, ensure_ascii=False)))


def _frozen(db):
    return db.execute("SELECT 1 FROM records WHERE kind='meta' AND id IN ('shutdown','exit_drain') LIMIT 1").fetchone()


def require_open(store):
    with store.connection() as db:
        if _frozen(db):
            raise ServiceError("后端正在退出，队列已冻结", 409)


def enqueue(store, task):
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if _frozen(db):
            raise ServiceError("后端正在退出，队列已冻结", 409)
        db.execute("INSERT INTO records VALUES ('task',?,?)",
                   (task["task_id"], json.dumps(task, ensure_ascii=False)))


def claim(store, key):
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if _frozen(db):
            return None
        task = _get(db, "task", key)
        if task is None or task["state"] != "queued" or task.get("stop_requested"):
            return None
        task["state"] = "preparing"
        task["output_verified"] = False
        _put(db, "task", key, task)
        return task


def _snapshot(db):
    from app.services.installation import RECOVERABLE

    tasks = []
    for row in db.execute("SELECT body FROM records WHERE kind='task' ORDER BY rowid"):
        task = json.loads(row[0])
        if task["state"] in EXECUTING | {"queued", "connection_lost"}:
            tasks.append({k: task.get(k) for k in ("task_id", "name", "state")})
    installs = []
    for row in db.execute("SELECT body FROM records WHERE kind='installation'"):
        session = json.loads(row[0])
        if session.get("state") in RECOVERABLE:
            installs.append({"name": session.get("engine_id") or session.get("installation_id"),
                             "state": session["state"]})
    active = [t for t in tasks if t["state"] in EXECUTING]
    unknown = [t for t in tasks if t["state"] == "connection_lost"]
    target = (unknown or active or [{}])[0].get("task_id")
    reason = ("unverified" if unknown or len(active) > 1 else "training" if active else
              "install" if installs else "queued" if tasks else "idle")
    result = dict(status="idle", reason=reason, tasks=tasks + installs, task_id=target, message="",
                  force_allowed=False, save_result=None, backend_pid=os.getpid(), installing=bool(installs))
    drain = _get(db, "meta", "exit_drain")
    if drain:
        key = drain["task_id"]
        task = _get(db, "task", key)
        result.update(task_id=key, save_result=(task or {}).get("stop_result"))
        if not task or unknown or len(active) > 1 or (active and active[0]["task_id"] != key):
            result.update(status="blocked", message="监管失联或任务归属未知，请先人工核实进程。")
        elif task["state"] in EXECUTING:
            timed_out = time.time() >= drain["deadline"]
            issue = task.get("graceful_exit_issue")
            forced = task.get("stop_requested") == "force"
            result.update(status="blocked" if (timed_out or issue) and not forced else "draining",
                          force_allowed=bool((timed_out or issue) and not forced and task.get("pid")),
                          message=("正在等待已确认的强制停止完成。" if forced else issue or
                                   ("等待引擎正常停止已超时。" if timed_out else "正在等待引擎停止并检查可续训保存点。")))
        elif task["state"] in {"stopped", "failed", "succeeded"} and not task.get("pid"):
            result.update(status="ready_to_exit", message=(result["save_result"] or {}).get("message", "任务已结束。"))
        else:
            result.update(status="blocked", message="无法确认训练进程已退出，请人工核实。")
    if _get(db, "meta", "shutdown"):
        result["status"] = "shutting_down"
    return result


def status(store):
    with store.connection() as db:
        db.execute("BEGIN")
        return _snapshot(db)


def _refuse(message, info):
    raise ServiceError(message, 409, payload={"shutdown": info})


def request(store, *, graceful, force=False, expected_reason=None, expected_task_id=None):
    """Freeze and request stopping as one commit, or finish a previously drained queue."""
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        info = _snapshot(db)
        if info["status"] == "shutting_down":
            return info
        if ((expected_reason is not None and expected_reason != info["reason"]) or
                (expected_task_id is not None and expected_task_id != info["task_id"])):
            _refuse("任务状态已变化，请重新确认。", info)
        if info["reason"] == "unverified" or info["installing"]:
            _refuse("有进程需要人工核实或仍有安装任务，请先处理。", info)
        if not graceful and (info["tasks"] or info["status"] != "idle"):
            # Preserve the existing shutdown refusal contract for legacy callers.
            if info["reason"] == "queued":
                info["reason"] = "training"
            _refuse("仍有训练任务处于活动状态，已拒绝退出。", info)
        if force:
            if not expected_task_id or not info["force_allowed"]:
                _refuse("强制停止需要对当前受阻任务单独确认。", info)
            task = _get(db, "task", expected_task_id)
            task["stop_requested"] = "force"
            _put(db, "task", expected_task_id, task)
        elif info["status"] == "idle" and info["reason"] == "training":
            key = info["task_id"]
            task = _get(db, "task", key)
            if task.get("stop_requested") != "force":
                task.update(stop_requested="graceful_exit", graceful_exit_issue=None)
            task["exit_requested_at"] = now()
            _put(db, "task", key, task)
            _put(db, "meta", "exit_drain", dict(task_id=key, deadline=time.time() + SHUTDOWN_TIMEOUT))
        elif info["status"] in {"idle", "ready_to_exit"}:
            _put(db, "meta", "shutdown", dict(requested_at=now(), backend_pid=os.getpid()))
        return _snapshot(db)


def cancel(store):
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if _get(db, "meta", "shutdown"):
            _refuse("退出已进入最后阶段。", _snapshot(db))
        db.execute("DELETE FROM records WHERE kind='meta' AND id='exit_drain'")
        return _snapshot(db)
