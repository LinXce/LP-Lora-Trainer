"""Independent, single-GPU queue worker. Never imports torch or an engine."""
import argparse
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from app.services.adapters import ADAPTERS, domain
from app.services.jobs import JobFiles
from app.services.common import ServiceError, atomic_text, now, identity, contained
from app.services.engines import revision
from app.storage.state import StateStore
from app.services.artifacts import checkpoint_complete
from app.services import output_ownership, queue_control
from app.services.progress import LOG_PROGRESS_VERSION, PROGRESS, merge_progress


# tqdm writes one frame per step with a carriage return; persisting every frame
# turns a 1,500-step run into thousands of near-identical log lines. Keep the
# parsed progress live, but only persist a frame to the human log periodically.
PROGRESS_LOG_INTERVAL = 20.0
_TQDM_PROGRESS = re.compile(r"\d{1,3}%\|")


def is_tqdm_progress(line: str) -> bool:
    """A carriage-return progress frame, not a normal engine log line."""
    return bool(_TQDM_PROGRESS.search(line))


def metric_point(progress):
    """Build one metrics.jsonl row from the latest parsed values.

    Only fields the adapter actually printed are copied, so a run whose engine has
    no learning-rate line produces loss-only points (and vice versa) instead of
    invented values. Returns None until a step is known.
    """
    step = progress.get("step")
    if not isinstance(step, int):
        return None
    point = {"step": step}
    for name in ("loss", "lr"):
        value = progress.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            point[name] = value
    return point if len(point) > 1 else None


def estimate_eta(progress, first_step, first_step_at, now):
    """Fallback ETA when the engine's log carries no ``eta=`` field.

    Prefer the explicit it/s rate; otherwise use the observed progress since the
    first step. Returns None until at least one rate measurement exists.
    """
    step = progress.get("step")
    total = progress.get("total_steps")
    if not isinstance(step, int) or not isinstance(total, int):
        return None
    remaining = max(0, total - step)
    if remaining == 0:
        return 0
    rate = progress.get("it_per_sec")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        return remaining / rate
    if (first_step is not None and first_step_at is not None
            and step > first_step and now > first_step_at):
        elapsed = now - first_step_at
        return remaining * elapsed / (step - first_step)
    return None


class ProcessLock:
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def acquire(self):
        if self.file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = self.path.open("a+b")
            # Reading another owner's locked byte raises PermissionError on Windows.
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.close()
            return False

    def close(self):
        if self.file: self.file.close(); self.file = None


class UnverifiedChildren(RuntimeError):
    """The stage parent exited while another process still owned the stdout pipe."""


def savepoint_signature(path):
    """Observe replacements and in-place updates without reading tensor payloads."""
    digest = hashlib.sha256()
    def add(item):
        stat = item.stat()
        record = (str(item.relative_to(path)), stat.st_dev, stat.st_ino, stat.st_size,
                  stat.st_mtime_ns, stat.st_ctime_ns)
        digest.update(json.dumps(record).encode("utf-8"))
    add(path)
    if path.is_dir():
        def raise_walk_error(error):
            raise error
        for directory, children, files in os.walk(path, onerror=raise_walk_error):
            children[:] = sorted(p for p in children if not (Path(directory) / p).is_symlink())
            for name in sorted(files):
                child = Path(directory) / name
                if not child.is_symlink():
                    add(child)
    return digest.hexdigest()


def register_artifacts(store, task, noted=None):
    """Register every artifact the adapter can currently see.

    ``noted`` is an optional set of already-reported paths: while a job runs this
    is called repeatedly, and a file that is still being written must not repeat
    the same failure line every pass.
    """
    if task.get("output_verified") is False or task.get("state") == "queued":
        return
    job = store.root / "jobs" / task["task_id"]
    root = Path(task["output_dir"]).resolve()
    try:
        candidates = ADAPTERS[task["engine_id"]].collect_artifacts(job)
    except Exception as exc:
        if noted is None or "__collect__" not in noted:
            if noted is not None: noted.add("__collect__")
            with (job / "stdout.log").open("a", encoding="utf-8") as f: f.write(f"[supervisor] 产物收集失败：{exc}\n")
        return
    for candidate in candidates:
        try:
            path = contained(candidate.path, root)
            if candidate.path.is_symlink(): continue
            complete = checkpoint_complete(path) if candidate.kind == "checkpoint" else False
            if candidate.kind == "video":
                # Video previews are written atomically enough for the size check to be
                # meaningful; the browser can play them directly.
                complete = path.stat().st_size > 0
            elif candidate.kind == "sample":
                from PIL import Image
                with Image.open(path) as image: image.verify()
                complete = True
            step = candidate.step
            final = getattr(candidate, "final", None)
            if step is None and final is None:
                # Last-resort heuristic for engines that name steps explicitly.
                # Never guess from a bare numeric suffix: musubi's epoch files are
                # "{name}-{epoch:06d}", which is not a training step.
                match = re.search(r"step(\d{3,})(?:\D|$)", path.stem)
                step = int(match[1]) if match else None
            # A name without a trailing number is the engine's final save: it writes
            # "{name}" at the end and "{name}-{step:08d}" / "{name}-{epoch:06d}" while running.
            if final is None:
                final = step is None and not re.search(r"\d{3,}$", path.stem)
            key = identity(task["task_id"], str(path))
            store.put("artifact", key, dict(artifact_id=key, task_id=task["task_id"], task_name=task["name"], kind=candidate.kind,
                file_name=path.name, path=str(path), step=step, final=final, size_bytes=path.stat().st_size,
                complete=complete, created_at=now(), preview_url=f"/api/v1/artifacts/{key}/preview" if candidate.kind in {"sample", "video"} and complete else None,
                architecture=task["architecture"], base_model=task["draft"]["base_model_path"]))
        except Exception as exc:
            # A malformed or half-written file is not a usable artifact; preserve
            # the reason in the task log, once per file.
            if noted is not None:
                if str(candidate.path) in noted: continue
                noted.add(str(candidate.path))
            with (job / "stdout.log").open("a", encoding="utf-8") as f: f.write(f"[supervisor] 产物登记失败：{candidate.path}: {exc}\n")


class Worker:
    # Never let the launching shell's Python/pip/uv/conda state leak into the engine.
    ENV_STRIP = frozenset((
        "PYTHONHOME", "PYTHONPATH", "PYTHON", "PYTHONSTARTUP", "PYTHONUSERBASE",
        "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        "PYENV_ROOT", "PYENV_VERSION",
    ))
    ENV_STRIP_PREFIXES = ("PIP_", "UV_")
    # How often a running job has its output directory re-scanned so that
    # checkpoints and sample images appear in the UI before the job ends.
    ARTIFACT_POLL_SECONDS = 15.0

    def __init__(self, store):
        self.store = store
        self.process = None
        self.stage_job = None
        self.reader = None
        self.reader_error = None
        self.pipe_eof = False
        self.task_id = None
        self.last_heartbeat = 0
        self._log_lock = threading.Lock()
        self._log_stream = None
        self._artifacts_at = 0.0
        self._artifact_notes = set()

    def heartbeat(self):
        if time.monotonic() - self.last_heartbeat > 1:
            try:
                atomic_text(self.store.root / "supervisor.json", json.dumps(dict(pid=os.getpid(), heartbeat=time.time(),
                    task_id=self.task_id, exit_protocol=queue_control.EXIT_PROTOCOL)))
            except OSError:
                # Liveness is advisory: a reader can briefly hold the file open on
                # Windows. Never let that interrupt a training run; the next loop
                # iteration retries, and last_heartbeat stays stale until it lands.
                return
            self.last_heartbeat = time.monotonic()

    def _child_env(self, overrides):
        env = os.environ.copy()
        for name in list(env):
            if name in self.ENV_STRIP or name.startswith(self.ENV_STRIP_PREFIXES):
                env.pop(name, None)
        env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", PYTHONUTF8="1")
        # Engine training subprocesses download their own weights; the LP proxy
        # setting was previously used only for installs/clones, so a working proxy
        # in Settings did not help MOSS/Anima HuggingFace fetches.
        settings = self.store.get("meta", "settings") or {}
        if settings.get("proxy_enabled") and settings.get("proxy_url"):
            proxy = str(settings["proxy_url"])
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env.setdefault(name, proxy)
        env.update(overrides)
        return env

    def log(self, key, text):
        # Single writer per log file: prefer the reader thread's open handle so
        # supervisor messages cannot interleave mid-line with tqdm output.
        line = f"[supervisor] {text}\n"
        with self._log_lock:
            stream = self._log_stream
            if stream is not None:
                try:
                    stream.write(line)
                    return
                except Exception:
                    self._log_stream = None
            p = self.store.root / "jobs" / key / "stdout.log"
            with p.open("a", encoding="utf-8") as f: f.write(line)

    def read_output(self, task, process, training_stage=True):
        key = task["task_id"]
        job = self.store.root / "jobs" / key
        adapter = ADAPTERS[task["engine_id"]]
        last_point = None
        progress = {**PROGRESS, **task["progress"], "log_version": LOG_PROGRESS_VERSION}
        task["progress"] = progress.copy()
        first_step, first_step_at = None, None
        noted = False

        def note(exc):
            # A parse/write fault must never kill a healthy training process; degrade
            # to raw-log-only and keep draining stdout so the child never blocks.
            nonlocal noted
            self.reader_error = str(exc)
            if noted: return
            noted = True
            try: self.log(key, f"日志解析/写入异常，已降级为仅记录原始日志：{exc}")
            except Exception: pass

        try:
            # Read chunks rather than readline: tqdm uses carriage returns and large output bursts.
            pending = ""
            last_progress_log = 0.0
            import codecs
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            with (job / "stdout.log").open("a", encoding="utf-8", buffering=1) as raw, (job / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics:
                with self._log_lock:
                    self._log_stream = raw
                while True:
                    chunk = process.stdout.read1(8192)
                    self.pipe_eof = not chunk
                    pending += decoder.decode(chunk, final=self.pipe_eof)
                    parts = re.split(r"[\r\n]", pending)
                    pending = parts.pop()
                    if self.pipe_eof or len(pending) > 65536:
                        parts.append(pending)
                        pending = ""
                    for line in parts:
                        if not line: continue
                        try:
                            progress_frame = is_tqdm_progress(line)
                            complete = "100%|" in line
                            stamp = time.monotonic()
                            if (not progress_frame or complete
                                    or stamp - last_progress_log >= PROGRESS_LOG_INTERVAL):
                                with self._log_lock:
                                    raw.write(line + "\n")
                                if progress_frame:
                                    last_progress_log = stamp
                            if not training_stage:
                                continue
                            for event in adapter.parse_log(line):
                                previous_step = progress.get("step")
                                if not merge_progress(progress, event.payload, event.kind):
                                    continue
                                step = progress.get("step")
                                if isinstance(step, int) and step > 0 and first_step is None:
                                    first_step, first_step_at = step, time.monotonic()
                                if "eta_seconds" not in event.payload and step != previous_step:
                                    progress["eta_seconds"] = estimate_eta(progress, first_step, first_step_at, stamp)
                                if "loss" in event.payload or "lr" in event.payload:
                                    point = metric_point(progress)
                                    if point is not None and point != last_point:
                                        metrics.write(json.dumps(point) + "\n"); last_point = point
                                # Publish a complete snapshot. The supervision loop saves
                                # the latest one even when the engine stops printing.
                                task["progress"] = progress.copy()
                        except Exception as exc:
                            note(exc)
                    if self.pipe_eof:
                        break
        except Exception as exc:
            # The pipe itself failed; keep the raw log written so far and stop parsing.
            note(exc)
        finally:
            with self._log_lock:
                self._log_stream = None
            process.stdout.close()

    def poll_artifacts(self, task):
        """Best-effort artifact registration while a job is still running.

        Without this, the results/overview pages stay empty for the whole run
        because the authoritative pass only happens after the last stage exits.
        """
        if time.monotonic() - self._artifacts_at < self.ARTIFACT_POLL_SECONDS:
            return
        self._artifacts_at = time.monotonic()
        try:
            register_artifacts(self.store, task, self._artifact_notes)
        except Exception as exc:
            # Never let a live-registration fault disturb a healthy training run;
            # the final pass in ``execute`` remains authoritative.
            self.log(task["task_id"], f"运行中产物登记异常（不影响训练）：{exc}")

    def _request_release_pause(self, job_directory):
        """Ask an engine that supports it to save state and exit at the next step."""
        try:
            atomic_text(
                Path(job_directory) / "pause.request",
                json.dumps({"release_model": True}),
            )
            return True
        except OSError:
            return False

    def _start_stage(self, argv, options):
        if os.name == "nt":
            from supervisor.windows_interrupt import stage_process_options
            from supervisor.windows_stage_job import StageJob
            options.update(stage_process_options())
            self.stage_job = StageJob()
            return self.stage_job.start(argv, **options)
        return subprocess.Popen(argv, start_new_session=True, **options)

    def _verify_stage_tree(self):
        if self.stage_job is None:
            return
        try:
            empty = self.stage_job.empty(timeout=1)
        except OSError as exc:
            raise UnverifiedChildren(f"无法确认训练进程树已退出：{exc}") from exc
        if not empty:
            raise UnverifiedChildren("阶段父进程已退出，但其训练进程树仍有存活成员")

    def _run_stage(self, task, adapter, argv, cwd, env, *, training_stage=True):
        """Run one pipeline command to completion and return its exit code."""
        key = task["task_id"]
        options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        self.reader_error = None
        self.pipe_eof = False
        self.process = self._start_stage(argv, options)
        current = self.store.get("task", key)
        patch = dict(state="running", pid=self.process.pid)
        if not current.get("started_at"): patch["started_at"] = now()
        self.store.patch("task", key, patch)
        stage_task = task.copy()
        published = stage_task.setdefault("progress", {})
        next_publish = 0

        def publish_progress(final=False):
            nonlocal published, next_publish
            snapshot = stage_task["progress"]
            if snapshot == published or (not final and time.monotonic() < next_publish):
                return
            next_publish = time.monotonic() + 0.5
            try:
                self.store.patch("task", key, dict(progress=snapshot))
                published = snapshot
            except (OSError, sqlite3.Error) as exc:
                # A failed display update must not stop the engine. Retry the
                # latest snapshot on the next tick; the log remains on disk.
                self.reader_error = str(exc)

        self.reader = threading.Thread(target=self.read_output, args=(stage_task, self.process, training_stage), daemon=True)
        self.reader.start()
        release_pause = training_stage and bool(env.get("ANIMA_DAEMON_JOB_DIR"))
        pause_requested = False

        def request_save_and_stop():
            if pause_requested:
                return True
            if release_pause:
                return self._request_release_pause(env["ANIMA_DAEMON_JOB_DIR"])
            if training_stage and adapter is not None:
                return adapter.native_stop(dict(task, pid=self.process.pid), argv)
            return False

        requested, stop_at = None, None
        while self.process.poll() is None:
            self.heartbeat()
            publish_progress()
            self.poll_artifacts(task)
            command = self.store.get("task", key)["stop_requested"]
            if command and command != requested:
                requested = command; stop_at = time.monotonic()
                self.store.patch("task", key, dict(state="stopping"))
                if command == "graceful_exit":
                    # Closing a window authorizes only a normal engine stop. Never
                    # substitute a hard kill for a missing capability or failed signal.
                    try:
                        baseline = {r["path"]: r["signature"] for r in self._savepoints(task)}
                    except Exception:
                        baseline = None  # A damaged/inaccessible baseline cannot prove a new save.
                    self.store.patch("task", key, dict(exit_savepoint_baseline=baseline))
                    if not (task.get("recovery") or {}).get("graceful_stop"):
                        issue = "引擎没有声明正常停止能力。训练仍由监管进程持有。"
                    elif request_save_and_stop():
                        pause_requested = True
                        issue = None
                    elif release_pause:
                        issue = "引擎暂停请求写入失败。"
                    else:
                        try:
                            delivered = self.interrupt(False, graceful_only=True)
                            issue = None if delivered else "无法向训练进程投递正常中断。"
                        except (OSError, subprocess.SubprocessError) as exc:
                            issue = f"投递正常中断失败：{exc}"
                    self.store.patch("task", key, dict(graceful_exit_issue=issue))
                    self.log(key, issue or "已向引擎请求正常停止，等待退出后核对可续训保存点。")
                elif command != "force" and request_save_and_stop():
                    pause_requested = True
                    self.log(key, "已请求引擎在下一步保存可续训状态并退出。")
                else:
                    self.log(key, "停止请求已发送，不承诺权重或完整状态保存。")
                    self.interrupt(command == "force")
            elif stop_at and requested != "graceful_exit":
                elapsed = time.monotonic() - stop_at
                # Release-pause must be allowed to finish a full optimizer state save;
                # an engine that never acknowledges it is force-killed eventually.
                deadline = 300 if pause_requested else 15
                if elapsed > deadline:
                    self.interrupt(True); stop_at = time.monotonic(); pause_requested = False
            time.sleep(0.2)
        self.reader.join(timeout=15)
        publish_progress(final=True)
        if self.reader.is_alive(): raise UnverifiedChildren("训练已退出，但日志管道未关闭；需要人工检查子进程")
        if self.store.get("task", key).get("exit_requested_at") and not self.pipe_eof:
            raise UnverifiedChildren("训练父进程已退出，日志通道异常断开，需核实子进程。")
        self._verify_stage_tree()
        if self.stage_job:
            self.stage_job.close()
            self.stage_job = None
        return self.process.returncode

    def interrupt(self, force, *, graceful_only=False):
        p = self.process
        if not p or p.poll() is not None: return True
        if os.name == "nt":
            if not force:
                from supervisor.windows_interrupt import send_interrupt
                if send_interrupt(p): return True
                if graceful_only: return False
            # Fixed command and numeric child PID; never use shell=True.
            result = subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=10,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode and p.poll() is None:
                raise RuntimeError("无法结束训练进程树，请检查系统权限并人工确认进程是否退出")
        else:
            try: os.killpg(p.pid, signal.SIGKILL if force else signal.SIGINT)
            except ProcessLookupError: pass
        return True

    def _savepoints(self, task):
        if task.get("output_verified") is False:
            return []
        job = self.store.root / "jobs" / task["task_id"]
        adapter = ADAPTERS[task["engine_id"]]
        found = {}
        def observe(path, kind, label, step):
            path = contained(path, task["output_dir"])
            if path.exists() and str(path) not in found:
                found[str(path)] = dict(kind=kind, path=str(path), label=label, step=step,
                                        signature=savepoint_signature(path))
        for source in adapter.resume_sources(job):
            observe(source.path, source.kind, source.label, source.step)
        # An engine may support full-state resume only, yet leave valid LoRA
        # weights. The result can describe the weights without promising resume.
        for artifact in adapter.collect_artifacts(job):
            if artifact.kind != "checkpoint" or artifact.path.is_symlink():
                continue
            path = contained(artifact.path, task["output_dir"])
            if path.is_file() and checkpoint_complete(path):
                observe(path, "weights", path.name, artifact.step)
        return list(found.values())

    def _stop_result(self, task):
        """Compare the post-exit inventory with what the engine had before signalling."""
        try:
            baseline = task.get("exit_savepoint_baseline")
            sources = self._savepoints(task)
            for source in sources:
                signature = source.pop("signature")
                path = source["path"]
                source["change"] = ("unknown" if baseline is None else "new" if path not in baseline else
                                    "existing" if baseline[path] == signature else "updated")
            kind = ("full_state" if any(s["kind"] == "state" for s in sources) else
                    "weights" if sources else "no_savepoint")
            message = {"full_state": "发现可续训的完整状态，请核对保存步数。",
                       "weights": "发现可用权重，未发现完整训练状态。",
                       "no_savepoint": "训练已退出，未发现适配器认可的可续训保存点。"}[kind]
            if sources:
                if baseline is None:
                    message += "缺少停止前快照，无法确认是否在此次停止时写入。"
                else:
                    counts = [sum(s["change"] == change for s in sources) for change in ("new","updated","existing")]
                    message += "与正常停止前相比，新增 %d 个、更新 %d 个、原有 %d 个。" % tuple(counts)
            return dict(status=kind, sources=sources, message=message)
        except Exception as exc:
            return dict(status="unknown", sources=[], message=f"训练已退出，保存点检查失败：{exc}")

    def finish_task(self, key, values):
        # The terminal state and the save-point verdict must be visible together.
        if self.store.patch("task", key, values, expected={"exit_requested_at": None}) is not None:
            return
        task = self.store.get("task", key)
        values.update(stop_result=self._stop_result(task), exit_savepoint_baseline=None)
        self.store.patch("task", key, values)

    def execute(self, task):
        key = task["task_id"]
        self.task_id = key
        self._artifacts_at = 0.0
        self._artifact_notes = set()
        reader = None
        try:
            if task["stop_requested"]:
                self.store.patch("task", key, dict(state="stopped", finished_at=now()))
                return
            claimed = queue_control.claim(self.store, key)
            if claimed is None: return
            task = claimed
            binding = task["binding"]
            if binding["schema_version"] != "1": raise RuntimeError("任务契约版本不兼容")
            job = self.store.root / "jobs" / key
            for name, digest in binding["config_hashes"].items():
                if hashlib.sha256((job / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError("绑定的任务配置已变化，拒绝执行")
            installation = binding["installation"]
            adapter = ADAPTERS[task["engine_id"]]
            if adapter.adapter_version != binding["adapter_version"]: raise RuntimeError("适配器版本变化，拒绝使用新适配器执行旧任务")
            if revision(Path(installation["source_path"])) != binding["revision"]: raise RuntimeError("绑定源码已变化，拒绝运行；请重新诊断并提交")
            if not Path(installation["python_executable"]).is_file(): raise RuntimeError("绑定解释器已不存在")
            manifest = self.store.root / "manifests" / f'{binding["environment_manifest_id"]}.json'
            if not manifest.is_file(): raise RuntimeError("绑定环境清单已不存在")
            # Re-check the environment versions without importing the engine source.
            # The module list must be the adapter's: the diagnostic only knows which
            # modules to look for from these arguments, and the manifest was written
            # with exactly that list.
            from app.services.engines import DIAGNOSTIC_CODE
            check = subprocess.Popen([installation["python_executable"], "-I", "-c", DIAGNOSTIC_CODE,
                                      *getattr(adapter, "diagnostic_modules", ()),
                                      *getattr(adapter, "optional_modules", ())],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 90
            while True:
                self.heartbeat()
                if time.monotonic() > deadline or self.store.get("task", key)["stop_requested"]:
                    check.kill(); check.communicate(); raise RuntimeError("环境复核超时或任务已取消")
                try:
                    stdout, stderr = check.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired: pass
            lines = [l[8:] for l in stdout.decode("utf-8", "replace").splitlines() if l.startswith("LP_DIAG=")]
            if check.returncode or not lines: raise RuntimeError("环境复核失败：" + stderr.decode("utf-8", "replace")[-1500:])
            fresh = json.loads(lines[-1])
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            # Compare only what the manifest recorded. A diagnostic that learned new
            # modules must not invalidate tasks bound to an older manifest; an actual
            # change of a recorded value (python, torch, cuda, a module) still refuses.
            if {key: fresh.get(key) for key in recorded} != recorded:
                raise RuntimeError("绑定环境已变化，请重新诊断并提交")
            job = self.store.root / "jobs" / key
            launch = adapter.build_launch(domain(installation), job / "config.native.toml")
            recorded = JobFiles(job).read_command()
            commands = launch.commands()
            if [list(a) for a in commands] != recorded["commands"] or str(launch.cwd) != recorded["cwd"]:
                raise RuntimeError("启动配置已变化，拒绝执行")
            if self.store.get("task", key)["stop_requested"]: raise RuntimeError("任务已取消")
            try:
                output_ownership.before_launch(self.store, task, job, adapter)
            except (OSError, ValueError, RuntimeError, ServiceError):
                self.store.patch("task", key, dict(output_conflict=True))
                raise
            if not getattr(adapter, "creates_output_dir", False):
                # Other engines expect the folder to exist when they start.
                Path(task["output_dir"]).mkdir(parents=True, exist_ok=False)
            task = self.store.patch("task", key, dict(output_verified=True))
            if "graceful_stop" not in (task.get("recovery") or {}):
                # Older bound jobs predate the per-architecture capability snapshot.
                caps = adapter.recovery_capabilities(domain(installation), task.get("architecture"))
                task["recovery"] = {**(task.get("recovery") or {}), "graceful_stop": caps.graceful_stop}
            env = self._child_env(launch.environment_overrides)
            if len(commands) > 1:
                self.log(key, f"包含 {len(commands) - 1} 个前置缓存阶段。")
            self.log(key, "启动绑定的引擎；中断不保证保存 checkpoint。")
            codes = []
            for index, argv in enumerate(commands):
                if index:
                    self.log(key, f"开始第 {index + 1}/{len(commands)} 阶段。")
                code = self._run_stage(
                    task, adapter, argv, launch.cwd, env,
                    training_stage=(index == len(commands) - 1),
                )
                codes.append(code)
                if code != 0 or self.store.get("task", key)["stop_requested"]:
                    break
            stopped = bool(self.store.get("task", key)["stop_requested"])
            code = codes[-1] if codes else -1
            complete = len(codes) == len(commands)
            if not stopped and code == 0 and complete:
                # Adapter-owned post-run work (e.g. writing a ComfyUI-readable copy of the
                # trained LoRA). A failure here is reported, never turned into a failed
                # run: the trained result already exists.
                try:
                    for notice in adapter.finalize_artifacts(
                        job, str(installation.get("python_executable") or ""), Path(installation["source_path"])
                    ):
                        self.log(key, notice)
                except Exception as exc:
                    self.log(key, f"收尾步骤失败（训练结果不受影响）：{exc}")
            state = "stopped" if stopped else "succeeded" if code == 0 and complete else "failed"
            # A log-pipeline fault is a note, never the reason a task failed:
            # the engine's own exit code stays the conclusion.
            error = None if state != "failed" else f"引擎退出码 {code}，请查看原始日志"
            if state == "failed" and self.reader_error:
                error += f"；日志管线异常（原始日志仍完整）：{self.reader_error}"
            register_artifacts(self.store, task)
            self.finish_task(key, dict(state=state, finished_at=now(), error_summary=error, pid=None))
            self.log(key, f"任务结束：{state}，退出码 {code}")
        except Exception as exc:
            uncertain = isinstance(exc, UnverifiedChildren)
            cleanup_error = None
            current = self.store.get("task", key)
            if self.process and self.process.poll() is None and current.get("stop_requested") == "graceful_exit":
                uncertain = True
            elif self.process and self.process.poll() is None:
                try:
                    self.interrupt(True)
                    self.process.wait(timeout=15)
                except Exception as cleanup:
                    uncertain = True
                    cleanup_error = str(cleanup)
            if self.reader:
                self.reader.join(timeout=2)
                uncertain = uncertain or self.reader.is_alive() or bool(current.get("exit_requested_at") and not self.pipe_eof)
            try:
                self._verify_stage_tree()
            except UnverifiedChildren as cleanup:
                uncertain = True
                cleanup_error = str(cleanup)
            stopped = bool(self.store.get("task", key)["stop_requested"])
            if uncertain:
                # Never free the GPU queue or claim stopped while children may still exist.
                message = f"{exc}；无法确认训练进程退出，需要人工核实"
                if cleanup_error: message += f"；{cleanup_error}"
                self.store.patch("task", key, dict(state="connection_lost", error_summary=message))
            else:
                self.finish_task(key, dict(state="stopped" if stopped else "failed", finished_at=now(),
                    error_summary=None if stopped else str(exc), pid=None))
            self.log(key, str(exc))

        finally:
            if self.stage_job:
                self.stage_job.close()  # Passive accounting handle. Surviving members continue.
                self.stage_job = None
            if self.process and self.process.poll() is None:
                # Reap a child if it exits later, without claiming an orphan stopped.
                # Its task remains connection_lost until explicit user acknowledgement.
                process, reader = self.process, self.reader
                def reap():
                    process.wait()
                    if reader: reader.join()
                threading.Thread(target=reap, daemon=True).start()
            self.process = None; self.reader = None; self.task_id = None

    def shutdown_requested(self):
        """True when the backend asked both itself and this supervisor to stop."""
        return bool(self.store.get("meta", "shutdown"))

    def run(self):
        # Unknown children from a crashed supervisor are never auto-replayed or killed by a reused PID.
        for task in self.store.list("task"):
            if task["state"] in ("preparing", "running", "stopping"):
                self.store.patch("task", task["task_id"], dict(state="connection_lost", error_summary="监管进程曾中断；原训练进程状态未知，请人工核实，不会自动重跑"))
        while True:
            self.heartbeat()
            # An explicit, protected shutdown request unwinds this loop so the
            # process lock is released instead of being inherited by a new build.
            if self.shutdown_requested(): return
            if self.store.get("meta", "exit_drain"):
                time.sleep(0.5); continue
            tasks = self.store.list("task")
            # Do not launch more GPU jobs while an orphan may still be training.
            if any(t["state"] == "connection_lost" for t in tasks):
                time.sleep(1); continue
            task = next((t for t in tasks if t["state"] == "queued"), None)
            if task: self.execute(task)
            else: time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", required=True); args = parser.parse_args()
    store = StateStore(Path(args.data_root))
    # Only the backend clears stale shutdown markers before spawning us.
    # Clearing here could erase a real close-during-bootstrap request.
    lock = ProcessLock(store.root / "supervisor.lock")
    if not lock.acquire(): return
    try: Worker(store).run()
    finally: lock.close()


if __name__ == "__main__": main()
