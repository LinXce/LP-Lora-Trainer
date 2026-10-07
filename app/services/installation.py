"""Run each copied trainer's official environment installer.

The application runtime is a bootstrap only. Training dependencies are created
inside the selected engine directory and the resolved interpreter is persisted
before optional diagnosis starts.
"""
from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import signal
import subprocess
import threading
import uuid

from adapters.registry import ADAPTERS
from app.services.common import ServiceError, contained, local_path, now

RUNNING = {"queued", "running", "stopping"}
RECOVERABLE = RUNNING | {"verifying"}
PROBE = (
    "import sys,json,importlib.util; "
    "print('LP_ENV='+json.dumps(dict(prefix=sys.prefix,"
    "base_prefix=sys.base_prefix,pip=importlib.util.find_spec('pip') is not None)))"
)


class InstallationService:
    def __init__(self, store, engines, project_root):
        self.store = store
        self.engines = engines
        self.project_root = Path(project_root).resolve()
        self.lock = threading.RLock()
        self.workers: dict[str, dict] = {}

        # A process is not safely resumable after a backend restart. Mark the
        # durable records explicitly instead of leaving the UI in a running state.
        for session in self.store.list("installation"):
            if session.get("state") not in RECOVERABLE:
                continue
            sid = session["session_id"]
            self.store.patch(
                "installation",
                sid,
                {
                    "state": "interrupted",
                    "finished_at": now(),
                    "error": "Application restart interrupted installation",
                },
            )
            engine = self.store.get("engine", session["installation_id"])
            if engine and engine.get("state") == "preparing":
                self.store.patch(
                    "engine",
                    session["installation_id"],
                    {
                        "state": "failed",
                        "verification": "unverified",
                        "issues": ["Installation was interrupted; please run it again"],
                    },
                )

    def get(self, key: str) -> dict:
        session = self.store.get("installation", key)
        if session is None:
            raise ServiceError("Installation session not found", 404)
        return session

    def list(self) -> list[dict]:
        return list(reversed(self.store.list("installation")))[:100]

    def log_path(self, key: str) -> Path:
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ServiceError("Invalid installation session id", 404)
        return contained(
            self.store.root / "installations" / key / "output.log",
            self.store.root / "installations",
        )

    def read_log(self, key: str, offset: int) -> dict[str, object]:
        session = self.get(key)
        path = self.log_path(key)
        if not path.is_file():
            return {"text": "", "offset": 0}
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start = min(max(offset, 0), size)
            stream.seek(start)
            chunk = stream.read(128 * 1024)
        # Keep partial UTF-8 characters for the next poll instead of emitting
        # replacement characters at the byte-sized response boundary.
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        text = decoder.decode(chunk, final=session.get("state") not in RECOVERABLE and start + len(chunk) == size)
        pending, _ = decoder.getstate()
        return {"text": text, "offset": start + len(chunk) - len(pending)}

    def _runtime_python(self) -> Path:
        name = "python.exe" if os.name == "nt" else "python"
        runtime = self.project_root / "python_runtime" / name
        if not runtime.is_file():
            raise ServiceError("Bundled application Python is missing; check python_runtime",)
        return runtime.resolve()

    @staticmethod
    def _environment_python(source_path: Path, adapter) -> Path | None:
        if adapter is None:
            return None
        root = Path(source_path).resolve()
        for relative in adapter.environment_candidates:
            env_root = root / relative
            candidates = (
                env_root / "Scripts" / "python.exe",
                env_root / "bin" / "python",
            )
            for candidate in candidates:
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(root):
                        raise ServiceError("Engine Python must stay inside the engine directory")
                    return resolved
        return None

    @staticmethod
    def _environment_root(source_path: Path, adapter) -> Path:
        relative = adapter.environment_dir or (
            adapter.environment_candidates[0]
            if adapter.environment_candidates
            else ".venv"
        )
        root = Path(source_path).resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ServiceError("The engine environment must stay inside the engine source directory")
        return target

    def _install_environment(self, source_path, adapter, runtime, mirror_url=None):
        """Return preparation commands, installer Python, environment and uv.

        LP never guesses a generic requirements/pip command. Each adapter owns
        the actual installer; LP only bootstraps uv or an empty venv where the
        copied trainer's documented workflow explicitly requires it.
        """
        environment = os.environ.copy()
        for name in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHON",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "PYTHONEXECUTABLE",
            "PIP_CONFIG_FILE", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
            "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL",
            "UV_CONFIG_FILE", "UV_WORKING_DIRECTORY", "UV_PROJECT",
            "UV_NO_SYNC", "UV_NO_BUILD", "UV_NO_DEPS", "UV_OFFLINE",
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_TARGET", "PIP_PREFIX", "PIP_USER",
            "UV_PYTHON", "UV_MANAGED_PYTHON", "UV_NO_MANAGED_PYTHON",
            "UV_SYSTEM_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_NO_SYNC",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "PYENV_ROOT",
            "PYENV_VERSION",
        }:
            environment.pop(name, None)
        environment.update(
            PYTHONUNBUFFERED="1",
            PYTHONNOUSERSITE="1",
            PYTHONUTF8="1",
            PIP_NO_INPUT="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            PIP_CONFIG_FILE=os.devnull,
            UV_NO_PROGRESS="1",
            UV_PYTHON_PREFERENCE="only-managed",
            UV_PYTHON_DOWNLOADS="automatic",
            UV_CACHE_DIR=str(self.project_root / "runtime" / "uv" / "cache"),
            UV_PYTHON_INSTALL_DIR=str(self.project_root / "runtime" / "uv" / "python"),
        )
        if mirror_url:
            # The copied engine still owns the install command. These standard
            # variables are inherited by pip and uv without replacing a
            # trainer-specific PyTorch/CUDA index declared by its project.
            environment.update(
                PIP_INDEX_URL=mirror_url,
                UV_INDEX_URL=mirror_url,
                UV_DEFAULT_INDEX=mirror_url,
            )

        uv = self.project_root / "runtime" / "uv" / (
            "uv.exe" if os.name == "nt" else "uv"
        )
        bootstrap = self.project_root / "scripts" / "bootstrap_uv.py"
        preparation = []

        if adapter.environment_strategy not in {"native", "precreate_venv", "uv_project"}:
            raise ServiceError(
                "Unsupported engine environment strategy: " + adapter.environment_strategy
            )
        # Every official workflow gets the same project-local uv. This also
        # prevents AI Toolkit's native manager from falling back to a user or
        # system uv when the project is moved to another computer.
        if not uv.is_file():
            if not bootstrap.is_file():
                raise ServiceError("Missing scripts/bootstrap_uv.py")
            preparation.append((str(runtime), "-u", str(bootstrap), str(uv)))

        if adapter.environment_strategy == "native":
            # AI Toolkit's manager owns Python, .venv, torch and requirements.
            # LP only makes its uv deterministic and exposes it through PATH.
            environment["PATH"] = os.pathsep.join((str(uv.parent), environment.get("PATH", "")))
            return preparation, str(runtime), environment, str(uv)

        env_root = self._environment_root(source_path, adapter)
        python_name = "python.exe" if os.name == "nt" else "python"
        python_dir = "Scripts" if os.name == "nt" else "bin"
        expected_python = env_root / python_dir / python_name
        if adapter.environment_strategy == "precreate_venv":
            if not adapter.python_version:
                raise ServiceError(f"{adapter.engine_id} does not declare a Python version")
            helper = self.project_root / "scripts" / "ensure_engine_venv.py"
            if not helper.is_file():
                raise ServiceError("Missing scripts/ensure_engine_venv.py")
            preparation.append((str(runtime), "-u", str(helper),
                                "--engine-root", str(source_path),
                                "--environment", str(env_root),
                                "--python-version", adapter.python_version,
                                "--uv", str(uv)))

        # Add the venv executable directory even before preparation creates it:
        # official installers may spawn `python`/`pip` through PATH themselves.
        path_entries = [str(expected_python.parent), str(uv.parent)]
        if adapter.environment_strategy == "uv_project":
            environment["UV_PROJECT_ENVIRONMENT"] = str(env_root)
        else:
            environment["VIRTUAL_ENV"] = str(env_root)
        path_entries.append(environment.get("PATH", ""))
        environment["PATH"] = os.pathsep.join(path_entries)
        return preparation, str(expected_python), environment, str(uv)

    def check_python(self, python):
        """Legacy helper; installation itself does not use user Python."""
        p = local_path(python)
        if not p.is_file():
            raise ServiceError("Selected Python file does not exist")
        runtime = self.project_root / "python_runtime"
        if p.is_relative_to(runtime):
            raise ServiceError("The application python_runtime cannot be used as a training environment")
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith(("PIP_", "PYTHON")):
                environment.pop(name)
        environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1")
        try:
            proc = subprocess.run(
                [str(p), "-I", "-c", PROBE],
                cwd=self.store.root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            lines = [line[7:] for line in proc.stdout.splitlines() if line.startswith("LP_ENV=")]
            if proc.returncode or not lines:
                raise ValueError((proc.stderr or proc.stdout)[-2000:] or "Python did not return environment information")
            info = json.loads(lines[-1])
            prefix = Path(info["prefix"]).resolve()
            if prefix.is_relative_to(runtime):
                raise ValueError("Python cannot be inside the application runtime directory")
            portable = any(p.parent.glob("python*._pth")) and prefix == p.parent
            if info["prefix"] == info["base_prefix"] and not portable:
                raise ValueError("Select a virtual-environment Python, not system Python")
            if not info["pip"]:
                raise ValueError("Selected environment has no pip")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ServiceError(f"Python check failed: {exc}") from exc
        return str(p), environment

    @staticmethod
    def _validate_mirror_url(value):
        if value is None or not str(value).strip():
            return None
        value = str(value).strip().rstrip("/")
        # urlsplit rejects malformed IPv6 addresses, and .port rejects invalid
        # ports. Convert those failures to a client error, not an API 500.
        if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
            raise ServiceError("Mirror URL must not contain whitespace or control characters")
        try:
            parsed = urlsplit(value)
            port = parsed.port
            hostname = parsed.hostname
        except ValueError as exc:
            raise ServiceError("Mirror must be a complete HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not hostname or port == 0 or "\\" in parsed.netloc:
            raise ServiceError("Mirror must be a complete HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ServiceError("Mirror URL must not contain credentials")
        return value

    def start(self, key, python=None, torch_source="cu124", mirror_url=None):
        # Retain python for compatibility with old clients. The engine's own
        # official workflow owns the actual interpreter and dependencies.
        del python
        with self.lock, self.engines.lock:
            if self.workers or self.engines.installing:
                raise ServiceError("Another engine installation is already running", 409)
            if any(
                task.get("state") in {"queued", "preparing", "running", "stopping"}
                for task in self.store.list("task")
            ):
                raise ServiceError("A training task is running; engine installation is blocked", 409)

            record = self.engines.get(key)
            adapter = ADAPTERS.get(record.get("engine_id"))
            path = Path(record["source_path"]).resolve()
            if adapter is None or not path.is_dir() or not adapter.detect(path).matched:
                raise ServiceError("Engine type is not confirmed or source is incomplete")
            import re
            if torch_source != "existing" and not re.fullmatch(r"cu\d{3}", torch_source):
                raise ServiceError("Unsupported PyTorch source")
            mirror_url = self._validate_mirror_url(mirror_url)

            runtime = self._runtime_python()
            try:
                preparation, install_python, env, uv = self._install_environment(
                    path, adapter, runtime, mirror_url
                )
                plan = adapter.installation_plan(path, install_python, torch_source, uv=uv)
            except ServiceError:
                raise
            except (ValueError, OSError) as exc:
                raise ServiceError(str(exc)) from exc

            commands = tuple(preparation) + tuple(plan.commands)
            session_id = uuid.uuid4().hex
            session = {
                "session_id": session_id,
                "installation_id": key,
                "label": record["label"],
                "engine_id": record["engine_id"],
                "title": plan.title,
                "cwd": str(path),
                "python_executable": install_python,
                "commands": [list(command) for command in commands],
                "command_index": None,
                "state": "queued",
                "started_at": now(),
                "finished_at": None,
                "error": None,
                "diagnostic_error": None,
                "mirror_url": mirror_url,
            }
            log = self.log_path(session_id)
            stored = False
            marked_installing = False
            engine_preparing = False
            try:
                log.parent.mkdir(parents=True, exist_ok=False)
                log.write_text("", encoding="utf-8")
                self.store.put("installation", session_id, session)
                stored = True
                self.engines.installing.add(key)
                marked_installing = True
                self.store.patch(
                    "engine",
                    key,
                    {
                        "state": "preparing",
                        "verification": "unverified",
                        "environment_id": None,
                        "issues": ["Installing the environment using the engine official workflow"],
                    },
                )
                engine_preparing = True
                state = {"cancel": threading.Event(), "process": None, "thread": None}
                thread = threading.Thread(
                    target=self._run,
                    args=(session, plan, env, state),
                    daemon=True,
                    name="engine-install-" + session_id[:8],
                )
                state["thread"] = thread
                self.workers[session_id] = state
                thread.start()
                return session
            except Exception as exc:
                # Starting an installation is a transaction: if any part of
                # registration or thread creation fails, no stale lock or
                # preparing engine may be left behind.
                state = self.workers.pop(session_id, None)
                if state is not None:
                    state["cancel"].set()
                    try:
                        self._terminate(state.get("process"))
                    except Exception:
                        pass
                if marked_installing:
                    self.engines.installing.discard(key)
                if stored:
                    try:
                        self.store.patch(
                            "installation",
                            session_id,
                            {
                                "state": "failed",
                                "finished_at": now(),
                                "error": f"Installation could not be started: {exc}",
                            },
                        )
                    except Exception:
                        pass
                if engine_preparing:
                    try:
                        self.store.patch(
                            "engine",
                            key,
                            {
                                "state": "failed",
                                "verification": "unverified",
                                "issues": [f"Installation could not be started: {exc}"],
                            },
                        )
                    except Exception:
                        pass
                if isinstance(exc, ServiceError):
                    raise
                raise ServiceError("Unable to start installation thread") from exc

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                [
                    str(Path(os.environ.get("SystemRoot", r"C:\\Windows")) / "System32" / "taskkill.exe"),
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)

    def cancel(self, key):
        with self.lock:
            session = self.get(key)
            state = self.workers.get(key)
            if state is None or session.get("state") not in RUNNING:
                raise ServiceError("This installation session cannot be stopped", 409)
            state["cancel"].set()
            self.store.patch("installation", key, {"state": "stopping"})
            process = state.get("process")
        try:
            self._terminate(process)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ServiceError(f"Failed to stop installation process: {exc}", 500) from exc

    @staticmethod
    def _popen_options(cwd, env):
        options = {
            "cwd": cwd,
            "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        return options

    @staticmethod
    def _read_output(stream, size=8192):
        """Read from real and mocked subprocess pipes.

        BufferedReader exposes ``read1`` while simple test doubles and some
        platform wrappers expose only ``read``. The installer must support both
        without changing the byte-oriented log behaviour.
        """
        reader = getattr(stream, "read1", None) or stream.read
        return reader(size)

    def _run(self, session, plan, env, state):
        sid = session["session_id"]
        engine_key = session["installation_id"]
        final = "failed"
        error = None
        diagnostic_error = None
        resolved_python = None
        log = None

        def emit(text: str):
            if log is None:
                return
            try:
                log.write(text.encode("utf-8", "replace"))
                log.flush()
            except (OSError, ValueError):
                # A logging failure must not strand the installation lock. The
                # durable session state remains the source of truth.
                pass

        try:
            try:
                log = self.log_path(sid).open("ab", buffering=0)
            except OSError as exc:
                error = f"Unable to open installation log: {exc}"
                raise RuntimeError(error) from exc

            with self.lock:
                if state["cancel"].is_set():
                    raise InterruptedError("Installation cancelled")
                self.store.patch("installation", sid, {"state": "running"})
            emit(f"[LP] {plan.title}\n[LP] Engine directory: {session['cwd']}\n")
            emit("[LP] Package index: " + (session.get("mirror_url") or "official default") + "\n")

            for index, argv in enumerate(session["commands"]):
                with self.lock:
                    if state["cancel"].is_set():
                        raise InterruptedError("Installation cancelled")
                    self.store.patch(
                        "installation", sid, {"command_index": index, "state": "running"}
                    )
                    emit("\n> " + subprocess.list2cmdline(argv) + "\n")
                    try:
                        process = subprocess.Popen(
                            list(argv), **self._popen_options(session["cwd"], env)
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not start installer step {index + 1}: {exc}"
                        ) from exc
                    state["process"] = process

                code = None
                try:
                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    stream = process.stdout
                    if stream is not None:
                        with stream:
                            while True:
                                chunk = self._read_output(stream)
                                if not chunk:
                                    break
                                if isinstance(chunk, str):
                                    emit(chunk)
                                else:
                                    emit(decoder.decode(chunk))
                            emit(decoder.decode(b"", final=True))
                    code = process.wait()
                finally:
                    if code is None:
                        try:
                            self._terminate(process)
                        except Exception as exc:
                            emit("[LP] Failed to stop installer process: " + str(exc) + "\n")
                    with self.lock:
                        state["process"] = None
                if state["cancel"].is_set():
                    raise InterruptedError("Installation cancelled")
                if code:
                    raise RuntimeError(
                        f"Official installer failed (step {index + 1}, exit code {code})"
                    )

            adapter = ADAPTERS.get(session["engine_id"])
            source_path = Path(session["cwd"])
            source_issues = adapter.source_issues(source_path) if adapter else ()
            if source_issues:
                raise RuntimeError("; ".join(source_issues))
            resolved_python = self._environment_python(source_path, adapter) if adapter else None
            if not resolved_python:
                raise RuntimeError("Official installer finished, but no engine Python was found")

            with self.lock:
                if state["cancel"].is_set():
                    raise InterruptedError("Installation cancelled")
                self.store.patch("installation", sid, {"state": "verifying"})
            final = "succeeded"
            emit("\n[LP] Official installation finished; engine Python found\n")
        except InterruptedError as exc:
            final, error = "cancelled", str(exc)
            emit("\n[LP] " + error + "\n")
        except Exception as exc:
            final = "cancelled" if state["cancel"].is_set() else "failed"
            error = "Installation cancelled" if final == "cancelled" else str(exc)
            emit("\n[LP] Installation " + final + ": " + error + "\n")
            try:
                self._terminate(state.get("process"))
            except Exception as terminate_exc:
                emit("[LP] Failed to stop installer process: " + str(terminate_exc) + "\n")
        finally:
            # Environment discovery is best effort even after a non-zero exit:
            # an installer may have created a usable interpreter before failing.
            if resolved_python is None:
                try:
                    adapter = ADAPTERS.get(session["engine_id"])
                    resolved_python = self._environment_python(Path(session["cwd"]), adapter)
                except Exception:
                    resolved_python = None

            engine_update = {
                "state": "discovered" if final == "succeeded" else "failed",
                "verification": "unverified",
                "environment_id": None,
                "issues": (
                    ["Environment installed; running environment diagnosis"]
                    if final == "succeeded"
                    else [error or "Installation failed"]
                ),
            }
            if resolved_python is not None:
                engine_update["python_executable"] = str(resolved_python)
            try:
                self.store.patch("engine", engine_key, engine_update)
            except Exception as exc:
                diagnostic_error = f"Failed to save engine environment metadata: {exc}"
                emit("[LP] " + diagnostic_error + "\n")
                if final == "succeeded":
                    final = "failed"
                    error = diagnostic_error

            # Diagnosis is intentionally after the interpreter path is saved.
            # If it fails, retain that path so the user can repair/re-diagnose
            # instead of losing the environment that was just installed.
            if final == "succeeded" and diagnostic_error is None:
                try:
                    self.engines.diagnose(engine_key, installation=True)
                    emit("[LP] Environment diagnosis passed; installation complete\n")
                except Exception as exc:
                    final = "failed"
                    diagnostic_error = getattr(exc, "message", None) or str(exc)
                    error = "Environment diagnosis failed: " + diagnostic_error
                    emit("[LP] Environment diagnosis failed: " + diagnostic_error + "\n")
                    try:
                        self.store.patch(
                            "engine",
                            engine_key,
                            {
                                "state": "failed",
                                "verification": "unverified",
                                "issues": ["Environment diagnosis failed: " + diagnostic_error],
                            },
                        )
                    except Exception as metadata_exc:
                        emit("[LP] Failed to save diagnosis result: " + str(metadata_exc) + "\n")

            try:
                self.store.patch(
                    "installation",
                    sid,
                    {
                        "state": final,
                        "error": error,
                        "diagnostic_error": diagnostic_error,
                        "finished_at": now(),
                    },
                )
            except Exception as exc:
                # There is no caller to receive a worker exception. Emit the
                # failure, but continue with process/worker cleanup below.
                emit("[LP] Failed to persist final installation state: " + str(exc) + "\n")
            finally:
                # Release locks only after diagnosis and final state persistence;
                # an external bind/rescan must not race the installation result.
                try:
                    with self.engines.lock:
                        self.engines.installing.discard(engine_key)
                except Exception as exc:
                    emit("[LP] Failed to release installation lock: " + str(exc) + "\n")
                try:
                    with self.lock:
                        state["process"] = None
                except Exception:
                    pass
                try:
                    with self.lock:
                        self.workers.pop(sid, None)
                except Exception:
                    pass
                if log is not None:
                    try:
                        log.close()
                    except OSError:
                        pass

    def shutdown(self):
        with self.lock:
            keys = list(self.workers)
            threads = [state["thread"] for state in self.workers.values()]
        for key in keys:
            try:
                self.cancel(key)
            except ServiceError:
                pass
        for thread in threads:
            thread.join(timeout=5)
