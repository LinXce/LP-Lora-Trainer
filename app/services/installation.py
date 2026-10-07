"""Explicit engine installation with persistent, bounded terminal log reads.

Never installs into the application runtime, never uses shell=True, and never
executes a copied engine during discovery. Installation requires user consent.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid

from adapters.registry import ADAPTERS
from app.services.common import ServiceError, local_path, now, contained

RUNNING = {"queued", "running", "stopping"}
RECOVERABLE = RUNNING | {"verifying"}
PROBE = "import sys,json,importlib.util; print('LP_ENV='+json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,pip=importlib.util.find_spec('pip') is not None)))"


class InstallationService:
    def __init__(self, store, engines, project_root):
        self.store, self.engines = store, engines
        self.project_root = Path(project_root).resolve()
        self.lock = threading.RLock()
        self.workers = {}
        # Backend process ownership is exclusive. Old sessions cannot be resumed
        # safely and are not misrepresented as running after a restart.
        for session in self.store.list("installation"):
            if session["state"] in RECOVERABLE:
                self.store.patch("installation", session["session_id"], dict(
                    state="interrupted", finished_at=now(), error="后端已重启；请检查残留安装进程后再重新安装"))
                record = self.store.get("engine", session["installation_id"])
                if record and record["state"] == "preparing":
                    self.store.patch("engine", session["installation_id"], dict(
                        state="failed", verification="unverified", issues=["环境安装被中断，请检查后重试"]))

    def get(self, key):
        session = self.store.get("installation", key)
        if session is None:
            raise ServiceError("安装会话不存在", 404)
        return session

    def list(self):
        return list(reversed(self.store.list("installation")))[:100]

    def log_path(self, key):
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ServiceError("无效安装会话", 404)
        return contained(self.store.root / "installations" / key / "output.log", self.store.root / "installations")

    def read_log(self, key, offset):
        self.get(key)
        path = self.log_path(key)
        if not path.is_file():
            return dict(text="", offset=0)
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start = min(offset, size)
            stream.seek(start)
            chunk = stream.read(128 * 1024)
            # Advance by bytes actually consumed, even when the final UTF-8
            # codepoint is split across polls; replacement keeps the cursor
            # monotonic and prevents repeating already displayed bytes.
            text = chunk.decode("utf-8", "replace")
        return dict(text=text, offset=start + len(chunk))

    def check_python(self, python):
        p = local_path(python)
        if not p.is_file():
            raise ServiceError("请选择存在的独立引擎 Python 解释器")
        runtime = self.project_root / "python_runtime"
        if p.is_relative_to(runtime):
            raise ServiceError("禁止将训练依赖安装到应用 python_runtime；请选择引擎独立环境")
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith(("PIP_", "PYTHON")):
                environment.pop(name)
        environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1")
        try:
            proc = subprocess.run([str(p), "-I", "-c", PROBE], cwd=self.store.root,
                env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            lines = [line[7:] for line in proc.stdout.splitlines() if line.startswith("LP_ENV=")]
            if proc.returncode or not lines:
                raise ValueError((proc.stderr or proc.stdout)[-2000:] or "解释器未返回环境信息")
            info = json.loads(lines[-1])
            prefix = Path(info["prefix"]).resolve()
            if prefix.is_relative_to(runtime):
                raise ValueError("解释器指向应用运行时")
            portable = any(p.parent.glob("python*._pth")) and prefix == p.parent
            if info["prefix"] == info["base_prefix"] and not portable:
                raise ValueError("目标为系统/全局 Python；请使用独立 venv 或独立便携 Python")
            if not info["pip"]:
                raise ValueError("独立环境缺少 pip，请先在该环境中安装 pip")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ServiceError(f"无法安装到该解释器：{exc}") from exc
        return str(p), environment

    def start(self, key, python=None, torch_source="cu124"):
        with self.lock, self.engines.lock:
            if self.workers:
                raise ServiceError("已有引擎环境安装正在进行，请在终端查看", 409)
            if any(t["state"] in ("queued", "preparing", "running", "stopping") for t in self.store.list("task")):
                raise ServiceError("请先停止或完成训练队列，避免安装修改正在使用的环境", 409)
            record = self.engines.get(key)
            adapter = ADAPTERS.get(record["engine_id"])
            path = Path(record["source_path"])
            if adapter is None or not path.is_dir() or not adapter.detect(path).matched:
                raise ServiceError("请先确认引擎类型并补全源码")
            python = python or record["python_executable"]
            if not python:
                raise ServiceError("请先选择该引擎独立环境的 Python；不会使用系统 Python 或应用运行时")
            if torch_source not in ("cu124", "cu126", "cu128", "existing"):
                raise ServiceError("不支持的 PyTorch 安装来源")
            python, env = self.check_python(python)
            try:
                plan = adapter.installation_plan(path, python, torch_source)
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc
            session_id = uuid.uuid4().hex
            session = dict(session_id=session_id, installation_id=key, label=record["label"],
                engine_id=record["engine_id"], title=plan.title, cwd=str(path), python_executable=python,
                commands=[list(c) for c in plan.commands], command_index=None,
                state="queued", started_at=now(), finished_at=None, error=None)
            log = self.log_path(session_id)
            log.parent.mkdir(parents=True, exist_ok=False)
            log.write_text("", encoding="utf-8")
            self.store.put("installation", session_id, session)
            self.engines.installing.add(key)
            self.store.patch("engine", key, dict(state="preparing", verification="unverified",
                environment_id=None, issues=["正在调用引擎安装流程；可在终端查看输出"]))
            # Clear global Python/Pip destination overrides, prepend this env's
            # scripts so upstream installers cannot accidentally use system pip.
            env["PATH"] = os.pathsep.join([str(Path(python).parent), str(Path(python).parent / "Scripts"), env.get("PATH", "")])
            env.update(PYTHONUNBUFFERED="1", PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
            state = dict(cancel=threading.Event(), process=None, thread=None)
            thread = threading.Thread(target=self._run, args=(session, plan, env, state), daemon=True,
                                      name="engine-install-" + session_id[:8])
            state["thread"] = thread
            self.workers[session_id] = state
            try:
                thread.start()
            except RuntimeError:
                self.workers.pop(session_id)
                self.engines.installing.discard(key)
                self.store.patch("installation", session_id, dict(state="failed", finished_at=now(), error="无法启动安装线程"))
                self.store.patch("engine", key, dict(state="failed", issues=["无法启动安装线程"]))
                raise
            return session

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run([str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"),
                "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(process.pid, signal.SIGKILL)

    def cancel(self, key):
        with self.lock:
            session = self.get(key)
            state = self.workers.get(key)
            if state is None or session["state"] not in RUNNING:
                raise ServiceError("安装会话已结束", 409)
            state["cancel"].set()
            self.store.patch("installation", key, dict(state="stopping"))
            process = state["process"]
        try:
            self._terminate(process)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ServiceError(f"停止安装失败：{exc}", 500) from exc

    def _run(self, session, plan, env, state):
        sid, key = session["session_id"], session["installation_id"]
        final, error = "failed", None
        with self.log_path(sid).open("ab", buffering=0) as log:
            def emit(text):
                log.write(text.encode("utf-8", "replace"))
            try:
                with self.lock:
                    if state["cancel"].is_set():
                        raise InterruptedError("安装已停止")
                    self.store.patch("installation", sid, dict(state="running"))
                emit("[LP] " + plan.title + "\n[LP] 工作目录：" + session["cwd"] + "\n")
                for index, argv in enumerate(plan.commands):
                    with self.lock:
                        if state["cancel"].is_set():
                            raise InterruptedError("安装已停止")
                        self.store.patch("installation", sid, dict(command_index=index))
                        emit("\n> " + subprocess.list2cmdline(argv) + "\n")
                        options = dict(cwd=session["cwd"], env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        if os.name == "nt":
                            options["creationflags"] = subprocess.CREATE_NO_WINDOW
                        else:
                            options["start_new_session"] = True
                        process = subprocess.Popen(list(argv), **options)
                        state["process"] = process
                    import codecs
                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    with process.stdout:
                        while chunk := process.stdout.read1(8192):
                            emit(decoder.decode(chunk))
                        emit(decoder.decode(b"", final=True))
                    code = process.wait()
                    if state["cancel"].is_set():
                        raise InterruptedError("安装已停止；部分依赖可能已安装，请重新诊断")
                    if code:
                        raise RuntimeError(f"安装步骤 {index + 1} 返回退出码 {code}")
                with self.lock:
                    if state["cancel"].is_set():
                        raise InterruptedError("安装已停止")
                    self.store.patch("installation", sid, dict(state="verifying"))
                final = "succeeded"
                emit("\n[LP] 引擎安装命令全部完成，正在诊断环境。\n")
            except InterruptedError as exc:
                final, error = "cancelled", str(exc)
                emit("\n[LP] " + error + "\n")
            except Exception as exc:
                error = str(exc)
                emit("\n[LP] 安装失败：" + error + "\n")
                try:
                    self._terminate(state["process"])
                except (OSError, subprocess.TimeoutExpired):
                    pass
            finally:
                with self.engines.lock:
                    self.engines.installing.discard(key)
                    self.store.patch("engine", key, dict(python_executable=session["python_executable"],
                        state="discovered" if final == "succeeded" else "failed", verification="unverified",
                        environment_id=None, issues=["依赖安装完成，尚需诊断"] if final == "succeeded" else [error or "安装失败"]))
                    if final == "succeeded":
                        try:
                            self.engines.diagnose(key)
                            emit("[LP] 基础环境诊断通过；训练支持以适配器能力为准。\n")
                        except ServiceError as exc:
                            emit("[LP] 依赖安装完成，但环境诊断未通过：" + str(exc) + "\n")
                with self.lock:
                    self.store.patch("installation", sid, dict(state=final, error=error, finished_at=now()))
                    self.workers.pop(sid, None)

    def shutdown(self):
        with self.lock:
            keys = list(self.workers)
            threads = [s["thread"] for s in self.workers.values()]
        for key in keys:
            try:
                self.cancel(key)
            except ServiceError:
                pass
        for thread in threads:
            thread.join(timeout=5)
