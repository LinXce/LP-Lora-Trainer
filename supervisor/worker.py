"""Independent, single-GPU queue worker. Never imports torch or an engine."""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from app.services.common import atomic_text, now, identity, contained
from app.services.engines import revision
from app.storage.state import StateStore
from app.services.artifacts import checkpoint_complete


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


def register_artifacts(store, task, noted=None):
    """Register every artifact the adapter can currently see.

    ``noted`` is an optional set of already-reported paths: while a job runs this
    is called repeatedly, and a file that is still being written must not repeat
    the same failure line every pass.
    """
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
            if candidate.kind == "sample":
                from PIL import Image
                with Image.open(path) as image: image.verify()
                complete = True
            step = candidate.step
            if step is None:
                # Last-resort heuristic for engines that name steps explicitly.
                # Never guess from a bare numeric suffix: musubi's epoch files are
                # "{name}-{epoch:06d}", which is not a training step.
                match = re.search(r"step(\d{3,})(?:\D|$)", path.stem)
                step = int(match[1]) if match else None
            key = identity(task["task_id"], str(path))
            store.put("artifact", key, dict(artifact_id=key, task_id=task["task_id"], task_name=task["name"], kind=candidate.kind,
                file_name=path.name, path=str(path), step=step, size_bytes=path.stat().st_size,
                complete=complete, created_at=now(), preview_url=f"/api/v1/artifacts/{key}/preview" if candidate.kind == "sample" and complete else None,
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
        self.reader = None
        self.reader_error = None
        self.task_id = None
        self.last_heartbeat = 0
        self._log_lock = threading.Lock()
        self._log_stream = None
        self._artifacts_at = 0.0
        self._artifact_notes = set()

    def heartbeat(self):
        if time.monotonic() - self.last_heartbeat > 1:
            atomic_text(self.store.root / "supervisor.json", json.dumps(dict(pid=os.getpid(), heartbeat=time.time(), task_id=self.task_id)))
            self.last_heartbeat = time.monotonic()

    def _child_env(self, overrides):
        env = os.environ.copy()
        for name in list(env):
            if name in self.ENV_STRIP or name.startswith(self.ENV_STRIP_PREFIXES):
                env.pop(name, None)
        env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", PYTHONUTF8="1")
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

    def read_output(self, task, process):
        key = task["task_id"]
        job = self.store.root / "jobs" / key
        adapter = ADAPTERS[task["engine_id"]]
        last_point, last_update = None, 0
        progress = task["progress"].copy()
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
            import codecs
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            with (job / "stdout.log").open("a", encoding="utf-8", buffering=1) as raw, (job / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics:
                with self._log_lock:
                    self._log_stream = raw
                while True:
                    chunk = process.stdout.read1(8192)
                    if not chunk: break
                    pending += decoder.decode(chunk)
                    parts = re.split(r"[\r\n]", pending)
                    pending = parts.pop()
                    if len(pending) > 65536: parts.append(pending); pending = ""
                    for line in parts:
                        if not line: continue
                        try:
                            with self._log_lock:
                                raw.write(line + "\n")
                            for event in adapter.parse_log(line):
                                progress.update(event.payload)
                                if "loss" in event.payload:
                                    point = dict(step=progress["step"], loss=progress["loss"])
                                    if point != last_point:
                                        metrics.write(json.dumps(point) + "\n"); last_point = point
                                if time.monotonic() - last_update >= 0.5:
                                    self.store.patch("task", key, dict(progress=progress.copy()))
                                    last_update = time.monotonic()
                        except Exception as exc:
                            note(exc)
                pending += decoder.decode(b"", final=True)
                if pending:
                    try:
                        with self._log_lock:
                            raw.write(pending + "\n")
                    except Exception as exc: note(exc)
                self.store.patch("task", key, dict(progress=progress.copy()))
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

    def _run_stage(self, task, adapter, argv, cwd, env):
        """Run one pipeline command to completion and return its exit code."""
        key = task["task_id"]
        options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        if os.name == "nt": options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else: options["start_new_session"] = True
        self.reader_error = None
        self.process = subprocess.Popen(argv, **options)
        current = self.store.get("task", key)
        patch = dict(state="running", pid=self.process.pid)
        if not current.get("started_at"): patch["started_at"] = now()
        self.store.patch("task", key, patch)
        self.reader = threading.Thread(target=self.read_output, args=(task, self.process), daemon=True)
        self.reader.start()
        requested, stop_at = None, None
        while self.process.poll() is None:
            self.heartbeat()
            self.poll_artifacts(task)
            command = self.store.get("task", key)["stop_requested"]
            if command and command != requested:
                requested = command; stop_at = time.monotonic()
                self.store.patch("task", key, dict(state="stopping"))
                self.log(key, "停止请求已发送，不承诺权重或完整状态保存。")
                self.interrupt(command == "force")
            elif stop_at and time.monotonic() - stop_at > 15:
                self.interrupt(True); stop_at = time.monotonic()
            time.sleep(0.2)
        self.reader.join(timeout=15)
        if self.reader.is_alive(): raise RuntimeError("训练已退出，但日志管道未关闭；需要人工检查子进程")
        return self.process.returncode

    def interrupt(self, force):
        p = self.process
        if not p or p.poll() is not None: return
        if os.name == "nt":
            if not force:
                try: p.send_signal(signal.CTRL_BREAK_EVENT); return
                except OSError: pass
            # Fixed command and numeric child PID; never use shell=True.
            result = subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=10,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode and p.poll() is None:
                raise RuntimeError("无法结束训练进程树，请检查系统权限并人工确认进程是否退出")
        else:
            try: os.killpg(p.pid, signal.SIGKILL if force else signal.SIGINT)
            except ProcessLookupError: pass

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
            claimed = self.store.patch("task", key, dict(state="preparing"), expected={"state": "queued"})
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
            from app.services.engines import DIAGNOSTIC_CODE
            check = subprocess.Popen([installation["python_executable"], "-I", "-c", DIAGNOSTIC_CODE], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
            if json.loads(lines[-1]) != json.loads(manifest.read_text(encoding="utf-8")): raise RuntimeError("绑定环境已变化，请重新诊断并提交")
            job = self.store.root / "jobs" / key
            launch = adapter.build_launch(domain(installation), job / "config.native.toml")
            recorded = json.loads((job / "command.json").read_text(encoding="utf-8"))
            commands = launch.commands()
            if [list(a) for a in commands] != recorded["commands"] or str(launch.cwd) != recorded["cwd"]:
                raise RuntimeError("启动配置已变化，拒绝执行")
            if self.store.get("task", key)["stop_requested"]: raise RuntimeError("任务已取消")
            Path(task["output_dir"]).mkdir(parents=True, exist_ok=False)
            env = self._child_env(launch.environment_overrides)
            if len(commands) > 1:
                self.log(key, f"包含 {len(commands) - 1} 个前置缓存阶段。")
            self.log(key, "启动绑定的引擎；中断不保证保存 checkpoint。")
            codes = []
            for index, argv in enumerate(commands):
                if index:
                    self.log(key, f"开始第 {index + 1}/{len(commands)} 阶段。")
                code = self._run_stage(task, adapter, argv, launch.cwd, env)
                codes.append(code)
                if code != 0 or self.store.get("task", key)["stop_requested"]:
                    break
            stopped = bool(self.store.get("task", key)["stop_requested"])
            code = codes[-1] if codes else -1
            complete = len(codes) == len(commands)
            state = "stopped" if stopped else "succeeded" if code == 0 and complete else "failed"
            # A log-pipeline fault is a note, never the reason a task failed:
            # the engine's own exit code stays the conclusion.
            error = None if state != "failed" else f"引擎退出码 {code}，请查看原始日志"
            if state == "failed" and self.reader_error:
                error += f"；日志管线异常（原始日志仍完整）：{self.reader_error}"
            register_artifacts(self.store, task)
            self.store.patch("task", key, dict(state=state, finished_at=now(), error_summary=error, pid=None))
            self.log(key, f"任务结束：{state}，退出码 {code}")
        except Exception as exc:
            uncertain = False
            cleanup_error = None
            if self.process and self.process.poll() is None:
                try:
                    self.interrupt(True)
                    self.process.wait(timeout=15)
                except Exception as cleanup:
                    uncertain = True
                    cleanup_error = str(cleanup)
            if self.reader:
                self.reader.join(timeout=2)
                uncertain = uncertain or self.reader.is_alive()
            stopped = bool(self.store.get("task", key)["stop_requested"])
            if uncertain:
                # Never free the GPU queue or claim stopped while children may still exist.
                message = f"{exc}；无法确认训练进程退出，需要人工核实"
                if cleanup_error: message += f"；{cleanup_error}"
                self.store.patch("task", key, dict(state="connection_lost", error_summary=message))
            else:
                self.store.patch("task", key, dict(state="stopped" if stopped else "failed", finished_at=now(),
                    error_summary=None if stopped else str(exc), pid=None))
            self.log(key, str(exc))

        finally:
            if self.process and self.process.poll() is None:
                # Reap a child if it exits later, without claiming an orphan stopped.
                # Its task remains connection_lost until explicit user acknowledgement.
                process = self.process
                def reap():
                    process.wait()
                    if self.reader: self.reader.join()
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
    # A freshly started supervisor owns a fresh lifetime: a shutdown request left
    # behind by a previous backend/supervisor pair must not stop it immediately.
    store.delete("meta", "shutdown")
    lock = ProcessLock(store.root / "supervisor.lock")
    if not lock.acquire(): return
    try: Worker(store).run()
    finally: lock.close()


if __name__ == "__main__": main()
