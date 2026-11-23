"""Read old progress as a projection of the retained engine log, without rewriting it."""
import json
import math
import re
import threading
from collections import OrderedDict
from pathlib import Path

from adapters.registry import ADAPTERS
from app.services.progress import LOG_PROGRESS_VERSION, PROGRESS, merge_progress


READ_BYTES = 256 * 1024
MAX_POINTS = 4000
_STAGES = re.compile(r"^\[supervisor\] 开始第 (\d+)/(\d+) 阶段")


def chart_sample(points):
    """Keep the first and last actual points across the whole run."""
    if len(points) <= MAX_POINTS:
        return points
    return [points[i * (len(points) - 1) // (MAX_POINTS - 1)] for i in range(MAX_POINTS)]


class _Replay:
    def __init__(self, path, task, with_points):
        self.progress = PROGRESS.copy()
        self.points = {} if with_points else None
        self.offset = 0
        self.stamp = None
        self.training = True
        self.pending = b""
        self.adapter = ADAPTERS.get(task.get("engine_id"))
        # A missing stage header is ambiguous if the job has multiple commands.
        try:
            command = path.with_name("command.json")
            if command.stat().st_size < 1024 * 1024:
                self.training = len(json.loads(command.read_text(encoding="utf-8"))["commands"]) == 1
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            pass

    def line(self, text):
        if text.startswith("[supervisor]"):
            if text.startswith("[supervisor] 包含 ") and "前置缓存阶段" in text:
                self.training = False
            elif match := _STAGES.match(text):
                self.training = match[1] == match[2]
            return
        if not self.training or self.adapter is None:
            return
        for event in self.adapter.parse_log(text):
            if not merge_progress(self.progress, event.payload, event.kind):
                continue
            step = self.progress.get("step")
            if self.points is None or not isinstance(step, int) or "step" not in event.payload:
                continue
            values = {k: event.payload[k] for k in ("loss", "lr") if isinstance(event.payload.get(k), (int, float))
                      and math.isfinite(event.payload[k])}
            if values:
                self.points.setdefault(step, {"step": step}).update(values)
                if len(self.points) > 2 * MAX_POINTS:
                    self.points = {k: v for i, (k, v) in enumerate(self.points.items()) if i % 2 == 0}

    def chunk(self, data, final=False):
        lines = (self.pending + data).splitlines(keepends=True)
        self.pending = b""
        if lines and not final and not lines[-1].endswith((b"\r", b"\n")):
            self.pending = lines.pop()[-65536:]
        for line in lines:
            self.line(line.decode("utf-8", errors="replace").strip("\r\n"))

    def update(self, path, stat, task):
        size = stat.st_size
        terminal = task.get("state") not in {"running", "stopping", "preparing", "queued"}
        with path.open("rb") as stream:
            stream.seek(self.offset)
            remaining = size - self.offset
            while remaining:
                data = stream.read(min(remaining, READ_BYTES))
                if not data:
                    raise OSError("训练日志在读取中被截短")
                self.chunk(data)
                remaining -= len(data)
            if terminal:
                self.chunk(b"", final=True)
        self.offset = size
        self.stamp = (stat.st_dev, stat.st_ino, size, stat.st_mtime_ns, terminal)


class ProgressHistory:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.task_locks = {}
        self.summaries = {}
        # Only open charts retain sample arrays. The per-task list cache stores
        # one small summary and cursor even when the original log was gigabytes.
        self.charts = OrderedDict()

    @staticmethod
    def legacy(task):
        old = task.get("progress")
        version = old.get("log_version") if isinstance(old, dict) else None
        return not isinstance(version, int) or version < LOG_PROGRESS_VERSION

    def _read(self, task, chart=False):
        key = task.get("task_id")
        if not isinstance(key, str) or Path(key).name != key or key in (".", ".."):
            return None
        cache = self.charts if chart else self.summaries
        path = Path(self.root) / "jobs" / key / "stdout.log"
        with self.lock:
            task_lock = self.task_locks.setdefault(key, threading.Lock())
        with task_lock:
            try:
                stat = path.stat()
                terminal = task.get("state") not in {"running", "stopping", "preparing", "queued"}
                stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, terminal)
                with self.lock:
                    replay = cache.get(key)
                if (replay is None or replay.stamp[:2] != stamp[:2] or stat.st_size < replay.offset
                        or (stat.st_size == replay.offset and replay.stamp[3] != stat.st_mtime_ns)):
                    replay = _Replay(path, task, chart)
                if replay.adapter is None:
                    return None
                if replay.stamp != stamp:
                    replay.update(path, stat, task)
                with self.lock:
                    cache[key] = replay
                    if chart:
                        cache.move_to_end(key)
                        if len(cache) > 8:
                            cache.popitem(last=False)
                if chart:
                    return replay.progress.copy(), [p.copy() for p in replay.points.values()]
                return {name: replay.progress.get(name) for name in PROGRESS}
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                with self.lock:
                    cache.pop(key, None)
                return None

    def project(self, task):
        if not self.legacy(task):
            return task
        evidence = self._read(task) if task.get("state") != "queued" else None
        if evidence is None:
            evidence = PROGRESS.copy()
            if task.get("state") != "queued":
                old = task.get("progress") if isinstance(task.get("progress"), dict) else {}
                for name in ("loss", "lr"):
                    if type(old.get(name)) in (float, int) and math.isfinite(old[name]):
                        evidence[name] = old[name]
        return {**task, "progress": evidence}

    def archive(self, task):
        # Explicit history deletion removes stdout.log. Preserve the observations
        # already available at that instant in the existing tombstone record.
        if self.legacy(task):
            verified = self._read(task)
            if verified is not None:
                return {**task, "progress": {**verified, "log_version": LOG_PROGRESS_VERSION}}
        return task

    def corrected_points(self, task, recorded):
        """Replace a demonstrably wrong axis only with observations from the log."""
        if not self.legacy(task):
            return recorded
        evidence = self._read(task, chart=True)
        if evidence is None:
            return recorded
        progress, points = evidence
        if progress.get("step") is None:
            return []
        steps = [p["step"] for p in recorded if isinstance(p.get("step"), int)]
        if not steps:
            return points
        peak = max(steps)
        total = progress.get("total_steps")
        if ((isinstance(total, int) and peak > total) or
                (points and any(a > b for a, b in zip(steps, steps[1:])))):
            return points
        rates = {p["step"]: p["lr"] for p in points if "lr" in p}
        for point in recorded:
            if "lr" not in point and point.get("step") in rates:
                point["lr"] = rates[point["step"]]
        # A missing tail is not evidence that the earlier dense samples were
        # assigned wrong steps. Extend only with actual later measurements.
        return recorded + [p for p in points if p["step"] > peak]
