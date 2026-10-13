"""Run each copied trainer's official environment installer.

The application runtime is a bootstrap only. Training dependencies are created
inside the selected engine directory and the resolved interpreter is persisted
before optional diagnosis starts.
"""
from __future__ import annotations

import codecs
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
import signal
import shutil
import subprocess
import threading
import uuid

from adapters.registry import ADAPTERS
from app.services import error_hints
from app.services.common import ServiceError, contained, local_path, now, validate_proxy_url

RUNNING = {"queued", "running", "stopping"}
# Only the end of the log is scanned: installers report their failure last.
DIAGNOSIS_TAIL_BYTES = 64 * 1024
RECOVERABLE = RUNNING | {"verifying"}
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
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
            engine = self.store.get("engine", session["installation_id"]) if session.get("installation_id") else None
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
        # A clone interrupted by a restart leaves only its hidden staging folder.
        try:
            root = Path(self.engines.settings()["engine_root"])
            for leftover in root.glob(".lp-clone-*") if root.is_dir() else ():
                if leftover.is_dir() and not leftover.is_symlink():
                    shutil.rmtree(leftover, ignore_errors=True)
        except Exception:
            pass

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

    def delete(self, key: str):
        with self.lock:
            session = self.get(key)
            if session.get("state") not in {"succeeded", "failed", "cancelled", "interrupted"} or key in self.workers:
                raise ServiceError("只能删除已结束的安装会话和日志", 409)
            self.log_path(key)  # Validate the session ID before any filesystem operation.
            target = self.store.root / "installations" / key
            directory = contained(target, self.store.root)
            if directory != target.absolute():
                raise ServiceError("安装日志目录不能通过链接指向其他目录", 403)
            if directory.exists():
                shutil.rmtree(directory)
            self.store.delete("installation", key)

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

    @staticmethod
    def _tail_text(path: Path, size: int = DIAGNOSIS_TAIL_BYTES) -> str:
        """Decode the end of a log file; a mid-character boundary is replaced."""
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            length = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, length - size))
            return stream.read().decode("utf-8", "replace")

    def diagnosis(self, key: str) -> dict[str, object]:
        """Explain a finished session from its error fields and log tail.

        Read-only and side-effect free: the terminal panel calls it once when a
        session is selected, never from the polling loop.
        """
        session = self.get(key)
        return {
            "session_id": key,
            "state": session.get("state"),
            "hints": error_hints.analyze(
                session.get("state"),
                session.get("error"),
                session.get("diagnostic_error"),
                self._tail_text(self.log_path(key)),
                session.get("kind"),
            ),
        }

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

    def _install_environment(self, source_path, adapter, runtime, mirror_url=None, torch_mirror_url=None, python_mirror_url=None, proxy_url=None):
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
            "UV_PYTHON_INSTALL_MIRROR",
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
            # pip hides its progress bar when stdout is a pipe, so a multi-GB
            # download shows nothing at all for minutes and looks like a hang.
            # Engines that re-emit pip lines then surface real progress.
            PIP_PROGRESS_BAR="on",
            # Engines install packages from their own tree (kohya: `-e
            # ./sd-scripts`). Creating a temporary build environment for those
            # can sit for many minutes inside pip, while the same build against
            # the venv's own setuptools/wheel finishes in about a second.
            # pip reads this as the value of the store_false option
            # `--no-build-isolation`, so only "0" disables isolation; "1" keeps it on.
            PIP_NO_BUILD_ISOLATION="0",
            UV_NO_PROGRESS="1",
            UV_PYTHON_PREFERENCE="only-managed",
            UV_PYTHON_DOWNLOADS="automatic",
            # A cache, not a dependency: safe to keep in the project.
            UV_CACHE_DIR=str(self.project_root / "runtime" / "uv" / "cache"),
        )
        # The engine keeps its own managed interpreter and its own uv binary, so
        # engine/<name>/ stays self-contained: deleting the LP project leaves a
        # venv whose base interpreter is still there.
        engine_home = Path(source_path).resolve() / ".lp"
        environment["UV_PYTHON_INSTALL_DIR"] = str(engine_home / "python")
        self._provide_engine_uv(engine_home)
        if mirror_url:
            # The copied engine still owns the install command. These standard
            # variables are inherited by pip and uv without replacing a
            # trainer-specific PyTorch/CUDA index declared by its project.
            environment.update(
                PIP_INDEX_URL=mirror_url,
                UV_INDEX_URL=mirror_url,
                UV_DEFAULT_INDEX=mirror_url,
            )
        # A PyTorch/CUDA wheel mirror is a *find-links* source for the engines
        # that declare CUDA-tagged pins (kohya) and a named uv index for musubi.
        # It is never added as an extra package index: an index URL of that shape
        # answers every lookup with a 404 and only slows resolution down.
        # uv fetches a managed interpreter from GitHub unless a mirror is given.
        # An unreachable GitHub otherwise surfaces as "Failed to download
        # .../python-build-standalone/..." and blocks engines whose installer
        # needs a Python version that is not installed yet.
        if python_mirror_url:
            environment["UV_PYTHON_INSTALL_MIRROR"] = python_mirror_url
        # An explicit proxy replaces whatever the host environment provides; when
        # the switch is off the ambient setting is inherited unchanged.
        environment.update(self._proxy_environment(proxy_url))
        # Engine installers run git themselves (kohya calls `git submodule
        # update`). Without a ceiling git walks up and finds the application's
        # own repository, where the command silently succeeds while doing
        # nothing for the engine. The ceiling keeps git inside the engine tree.
        environment["GIT_CEILING_DIRECTORIES"] = str(Path(source_path).resolve())

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

    def _proxy_for(self, requested):
        """Resolve the proxy for one download, or None when it stays disabled.

        ``requested`` is the per-operation switch; None follows the saved
        setting. The URL itself lives only in settings and is applied to the
        child environment — it is never written to a session or a log.
        """
        settings = self.engines.settings()
        enabled = bool(settings.get("proxy_enabled", False) if requested is None else requested)
        if not enabled:
            return None
        url = validate_proxy_url(settings.get("proxy_url"))
        if not url:
            raise ServiceError("已启用代理，请先在“设置”中填写代理地址")
        return url

    @staticmethod
    def _proxy_environment(proxy_url):
        """Proxy variables understood by pip, uv, git and urllib."""
        if not proxy_url:
            return {}
        return {name: proxy_url for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        )}

    def _provide_engine_uv(self, engine_home: Path) -> None:
        """Copy the bundled uv next to the engine so it can repair itself later.

        Best effort: uv is only needed to (re)install an engine, never to run
        one, so a failed copy must not fail the installation.
        """
        name = "uv.exe" if os.name == "nt" else "uv"
        source = self.project_root / "runtime" / "uv" / name
        target = engine_home / name
        try:
            if not source.is_file():
                return
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                return
            engine_home.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError:
            pass

    def _build_install(self, path, adapter, runtime, torch_source, mirror_url, torch_mirror_url, python_mirror_url, proxy_url=None):
        """Return ``(commands, installer python, environment, plan)`` for one attempt.

        Shared by the first attempt and the official-source fallback, which
        calls this with every mirror cleared so the retry really is official.
        """
        preparation, install_python, env, uv = self._install_environment(
            path, adapter, runtime, mirror_url, torch_mirror_url, python_mirror_url, proxy_url
        )
        plan = adapter.installation_plan(
            path,
            install_python,
            torch_source,
            uv=uv,
            torch_mirror_url=torch_mirror_url,
        )
        # Fetch the engine's own exact wheel pins straight from the selected
        # mirror. They are the only candidates pip cannot find on the PyPI
        # mirror, and searching several large indexes for them is what makes its
        # resolver spin for a long time without transferring anything. This step
        # is optional: if the mirror lacks them, the engine's installer still
        # runs exactly as before.
        # Adapters own the whole install command (kohya installs its declared
        # requirements itself); LP only bootstraps the environment it needs.
        return tuple(preparation) + tuple(plan.commands), install_python, env, plan

    def start(
        self,
        key,
        python=None,
        torch_source="cu124",
        mirror_url=None,
        torch_mirror_url=None,
        python_mirror_url=None,
        use_proxy=None,
    ):
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
            if torch_source != "existing" and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", torch_source):
                raise ServiceError("Unsupported PyTorch source")
            mirror_url = self._validate_mirror_url(mirror_url)
            torch_mirror_url = self._validate_mirror_url(torch_mirror_url)
            python_mirror_url = self._validate_mirror_url(python_mirror_url)
            proxy_url = self._proxy_for(use_proxy)

            runtime = self._runtime_python()
            try:
                commands, install_python, env, plan = self._build_install(
                    path, adapter, runtime, torch_source, mirror_url, torch_mirror_url, python_mirror_url, proxy_url
                )
            except ServiceError:
                raise
            except (ValueError, OSError) as exc:
                raise ServiceError(str(exc)) from exc
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
                "torch_source": torch_source,
                "torch_mirror_url": torch_mirror_url,
                "python_mirror_url": python_mirror_url,
                # Notices for the run itself, e.g. the official-source fallback.
                "warnings": [],
                # Only whether a proxy was used; the URL stays in settings.
                "proxy": bool(proxy_url),
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
                state = {"cancel": threading.Event(), "process": None, "thread": None, "proxy_url": proxy_url}
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
    def _failure_output(text: str) -> str:
        """Only the output of the step that failed.

        Header lines and earlier steps name the selected mirror URLs, and a
        mirror URL contains words like ``python-build-standalone``; scanning
        them would make an unrelated failure look like a download failure of
        that mirror.
        """
        _, _, last_step = text.rpartition("\n> ")
        return "\n".join(
            line for line in (last_step or text).splitlines() if not line.startswith("[LP] ")
        )

    def _official_fallback_reason(self, session, already_used, log_path):
        """Cause to retry on the official source, or None to leave the failure alone.

        Only mirror/network failures qualify: a broken engine script fails the
        same way on every source, and the fallback runs at most once.
        """
        if already_used:
            return None
        if not any(session.get(name) for name in ("mirror_url", "torch_mirror_url", "python_mirror_url")):
            return None
        return error_hints.retry_reason(self._failure_output(self._tail_text(log_path)))

    def _official_attempt(self, session, proxy_url=None):
        """Rebuild the install steps with every mirror cleared.

        The proxy is not a source and stays in effect: an unreachable mirror is
        exactly when a proxy is most likely to help.
        """
        adapter = ADAPTERS.get(session.get("engine_id"))
        if adapter is None:
            raise RuntimeError("Engine adapter is unavailable")
        return self._build_install(
            Path(session["cwd"]),
            adapter,
            self._runtime_python(),
            session.get("torch_source") or "existing",
            None,
            None,
            None,
            proxy_url,
        )

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

        # Declared before the try so the failure handler can report the outcome
        # of the official-source fallback even if it never got that far.
        fallback_reason = None

        def set_warning(notice):
            session["warnings"] = [notice]
            try:
                with self.lock:
                    self.store.patch("installation", sid, {"warnings": [notice]})
            except Exception:
                # The notice is cosmetic; never let it fail the install.
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

            def emit_sources():
                emit("[LP] Package index: " + (session.get("mirror_url") or "official default") + "\n")
                emit(
                    "[LP] PyTorch index ("
                    + str(session.get("torch_source") or "default")
                    + "): "
                    + (session.get("torch_mirror_url") or "project default")
                    + "\n"
                )
                emit(
                    "[LP] Python download source: "
                    + (session.get("python_mirror_url") or "GitHub default")
                    + "\n"
                )
                emit("[LP] Proxy: " + (self._redact(state.get("proxy_url")) if state.get("proxy_url") else "off") + "\n")

            def run_commands(commands, attempt_env):
                for index, argv in enumerate(commands):
                    with self.lock:
                        if state["cancel"].is_set():
                            raise InterruptedError("Installation cancelled")
                        self.store.patch(
                            "installation", sid, {"command_index": index, "state": "running"}
                        )
                        emit("\n> " + subprocess.list2cmdline(argv) + "\n")
                        try:
                            process = subprocess.Popen(
                                list(argv), **self._popen_options(session["cwd"], attempt_env)
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

            emit(f"[LP] {plan.title}\n[LP] Engine directory: {session['cwd']}\n")
            emit_sources()

            # A failed mirror download is retried once against the official
            # sources, with every mirror cleared, and reported as a warning.
            commands, attempt_env = session["commands"], env
            while True:
                try:
                    run_commands(commands, attempt_env)
                    break
                except RuntimeError:
                    reason = self._official_fallback_reason(
                        session, fallback_reason is not None, self.log_path(sid)
                    )
                    if reason is None:
                        raise
                    fallback_reason = reason
                    emit("\n[LP] 镜像源下载失败（" + reason + "），改用官方源重试\n")
                    set_warning(error_hints.mirror_fallback_notice(reason, "retrying"))
                    try:
                        commands, _, attempt_env, _ = self._official_attempt(session, state.get("proxy_url"))
                    except Exception as exc:
                        emit("[LP] 无法构建官方源安装步骤: " + str(exc) + "\n")
                        raise
                    session["commands"] = [list(command) for command in commands]
                    session["command_index"] = None
                    session["mirror_url"] = session["torch_mirror_url"] = session["python_mirror_url"] = None
                    with self.lock:
                        self.store.patch(
                            "installation",
                            sid,
                            {
                                "commands": session["commands"],
                                "command_index": None,
                                "mirror_url": None,
                                "torch_mirror_url": None,
                                "python_mirror_url": None,
                            },
                        )
                    emit_sources()
            if fallback_reason is not None:
                set_warning(error_hints.mirror_fallback_notice(fallback_reason, "succeeded"))

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
            if fallback_reason is not None and final == "failed":
                set_warning(error_hints.mirror_fallback_notice(fallback_reason, "failed"))
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

    # ---- git clone into engine/ ----

    CLONE_KIND = "clone"
    REPAIR_KIND = "repair"
    _NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    _URL_RE = re.compile(r"^(?:https?://|ssh://|file:///|git@)[^\s\"'<>|^`]+$")

    @staticmethod
    def _git_executable() -> str | None:
        return shutil.which("git")

    @staticmethod
    def _redact(url: str) -> str:
        """Drop user:password@ so credentials never reach logs or the store."""
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f":{parts.port}"
            return parts._replace(netloc=netloc).geturl()
        return url

    @classmethod
    def default_clone_name(cls, url: str) -> str:
        """Folder name a plain `git clone <url>` would pick (e.g. musubi-tuner)."""
        tail = url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        name = re.sub(r"[^A-Za-z0-9._-]", "-", tail).strip(".-")
        return name[:100] or "engine"

    def clone(self, url: str, name: str | None = None, ref: str | None = None, use_proxy=None, commit: str | None = None) -> dict:
        url = (url or "").strip()
        if not self._URL_RE.match(url):
            raise ServiceError("请填写 https://、ssh://、file:/// 或 git@ 开头的 Git 仓库地址")
        name = (name or "").strip() or self.default_clone_name(url)
        if not self._NAME_RE.match(name) or name in (".", "..") or name.upper().split(".")[0] in WINDOWS_RESERVED:
            raise ServiceError("目录名只能包含字母、数字、点、下划线和连字符，且不能以符号开头")
        ref = (ref or "").strip() or None
        if ref is not None and (ref.startswith("-") or not re.match(r"^[A-Za-z0-9._/-]{1,200}$", ref)):
            raise ServiceError("分支或标签名称不合法")
        commit = (commit or "").strip() or None
        if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            raise ServiceError("提交号只能包含十六进制字符（7–40 位）")
        git = self._git_executable()
        if not git:
            raise ServiceError("未找到 git，请先安装 Git for Windows 并确保 git 在 PATH 中", 409)

        root = Path(self.engines.settings()["engine_root"]).resolve()
        target = root / name
        # Clone into a hidden sibling: rescan skips dot-folders, so a half-finished
        # clone never shows up as an engine, and the final rename is atomic.
        staging = root / f".lp-clone-{uuid.uuid4().hex[:12]}"
        with self.lock, self.engines.lock:
            if self.workers or self.engines.installing:
                raise ServiceError("已有引擎安装或拉取任务在运行，请等待其结束", 409)
            root.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ServiceError(f"engine 目录下已存在 {name}，请换一个目录名", 409)
            safe_url = self._redact(url)
            proxy_url = self._proxy_for(use_proxy)
            argv = [git, "-c", "core.fsmonitor=false", "clone", "--progress"]
            if ref:
                argv += ["--branch", ref]
            argv += ["--", url, str(staging)]
            session_id = uuid.uuid4().hex
            session = {
                "session_id": session_id, "kind": self.CLONE_KIND,
                "installation_id": "", "label": name, "engine_id": "",
                "title": f"Git 拉取 {safe_url}" + f"（{ref or '默认分支'}" + (f" @ {commit[:8]}" if commit else "") + "）",
                "cwd": str(target), "python_executable": "",
                # The URL as typed may hold a token; persist and display only the redacted form.
                "commands": [[*argv[:-2], safe_url, str(target)]], "command_index": None,
                "state": "queued", "started_at": now(), "finished_at": None,
                "error": None, "diagnostic_error": None,
                "repository_url": safe_url, "ref": ref,
                "pinned_commit": commit,
                # Only whether a proxy was used; the URL stays in settings.
                "proxy": bool(proxy_url),
            }
            log = self.log_path(session_id)
            log.parent.mkdir(parents=True, exist_ok=False)
            log.write_text("", encoding="utf-8")
            self.store.put("installation", session_id, session)
            state = {"cancel": threading.Event(), "process": None, "thread": None}
            thread = threading.Thread(target=self._run_clone, args=(session, argv, staging, target, state, proxy_url, commit),
                                      daemon=True, name="engine-clone-" + session_id[:8])
            state["thread"] = thread
            self.workers[session_id] = state
            try:
                thread.start()
            except Exception as exc:
                self.workers.pop(session_id, None)
                self.store.patch("installation", session_id, {"state": "failed", "finished_at": now(), "error": str(exc)})
                raise ServiceError("无法启动拉取任务") from exc
            return session

    def repair_source(self, key, use_proxy=None) -> dict:
        """Complete a partially fetched engine source, e.g. a git submodule.

        The adapter decides what the step is; LP only runs it inside the engine
        directory, streams it to the terminal and refreshes detection.
        """
        with self.lock, self.engines.lock:
            if self.workers or self.engines.installing:
                raise ServiceError("已有引擎安装或拉取任务在运行，请等待其结束", 409)
            if any(task.get("state") in RUNNING for task in self.store.list("task")):
                raise ServiceError("有训练任务正在运行，暂不能补齐引擎源码", 409)
            record = self.engines.get(key)
            adapter = ADAPTERS.get(record.get("engine_id"))
            path = Path(record["source_path"]).resolve()
            if adapter is None or not path.is_dir():
                raise ServiceError("引擎类型未确认或源码目录不存在")
            command = adapter.source_repair(path)
            if not command:
                raise ServiceError("该引擎源码当前完整，或无法自动补齐：请重新完整 clone 后刷新发现")
            tool = shutil.which(command[0])
            if not tool:
                raise ServiceError(f"未找到 {command[0]}，请先安装并确保它在 PATH 中", 409)
            argv = [tool, *command[1:]]
            proxy_url = self._proxy_for(use_proxy)

            session_id = uuid.uuid4().hex
            session = {
                "session_id": session_id, "kind": self.REPAIR_KIND,
                "installation_id": key, "label": record["label"],
                "engine_id": record["engine_id"],
                "title": f"补齐引擎源码：{adapter.source_repair_label}",
                "cwd": str(path), "python_executable": "",
                "commands": [list(argv)], "command_index": None,
                "state": "queued", "started_at": now(), "finished_at": None,
                "error": None, "diagnostic_error": None,
                "proxy": bool(proxy_url),
            }
            log = self.log_path(session_id)
            log.parent.mkdir(parents=True, exist_ok=False)
            log.write_text("", encoding="utf-8")
            self.store.put("installation", session_id, session)
            state = {"cancel": threading.Event(), "process": None, "thread": None}
            thread = threading.Thread(target=self._run_repair, args=(session, argv, path, state, proxy_url),
                                      daemon=True, name="engine-repair-" + session_id[:8])
            state["thread"] = thread
            self.workers[session_id] = state
            try:
                thread.start()
            except Exception as exc:
                self.workers.pop(session_id, None)
                self.store.patch("installation", session_id, {"state": "failed", "finished_at": now(), "error": str(exc)})
                raise ServiceError("无法启动补齐任务") from exc
            return session

    def _run_git_session(self, session, argv, cwd, state, proxy_url, *, action,
                         cancel_message, directory_label, failure_message, finish=None, cleanup=None):
        """Run one git command as a tracked session, streaming it to the log.

        ``finish(emit)`` runs after a zero exit and may raise to fail the
        session; ``cleanup`` always runs last (e.g. dropping a staging folder).
        """
        sid = session["session_id"]
        final, error = "failed", None
        log = None
        process = None

        def emit(text: str):
            if log is None:
                return
            try:
                log.write(text.encode("utf-8", "replace"))
                log.flush()
            except (OSError, ValueError):
                pass

        env = os.environ.copy()
        # Never block on a credential/host-key prompt nobody can answer.
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_ASKPASS="echo",
                   SSH_ASKPASS="echo", GIT_SSH_COMMAND=env.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes"))
        env.update(self._proxy_environment(proxy_url))
        # Keep git inside the engine tree (see _install_environment).
        env["GIT_CEILING_DIRECTORIES"] = str(Path(cwd).resolve())
        try:
            log = self.log_path(sid).open("ab", buffering=0)
            with self.lock:
                if state["cancel"].is_set():
                    raise InterruptedError(cancel_message)
                self.store.patch("installation", sid, {"state": "running", "command_index": 0})
            emit(f"[LP] {session['title']}\n[LP] {directory_label}：{session['cwd']}\n")
            emit("[LP] Proxy: " + (self._redact(proxy_url) if proxy_url else "off") + "\n")
            # Echo the session's display command: a clone keeps its credentials out
            # of ``commands`` even though the real argv still carries them.
            display = session.get("commands") or []
            emit("\n> " + subprocess.list2cmdline(list(display[0] if display else argv)) + "\n")
            with self.lock:
                process = subprocess.Popen(list(argv), **self._popen_options(str(cwd), env))
                state["process"] = process
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            with process.stdout:
                while True:
                    chunk = self._read_output(process.stdout)
                    if not chunk:
                        break
                    emit(chunk if isinstance(chunk, str) else decoder.decode(chunk))
                emit(decoder.decode(b"", final=True))
            code = process.wait()
            if state["cancel"].is_set():
                raise InterruptedError(cancel_message)
            if code:
                raise RuntimeError(failure_message(code))
            if finish is not None:
                finish(emit)
            final = "succeeded"
        except InterruptedError as exc:
            final, error = "cancelled", str(exc)
            emit("\n[LP] " + error + "\n")
        except Exception as exc:
            final = "cancelled" if state["cancel"].is_set() else "failed"
            error = cancel_message if final == "cancelled" else (getattr(exc, "message", None) or str(exc))
            emit("\n[LP] " + (action + "已取消" if final == "cancelled" else action + "失败：" + error) + "\n")
        finally:
            try:
                self._terminate(process)
            except Exception:
                pass
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    pass
            try:
                self.store.patch("installation", sid, {"state": final, "error": error, "finished_at": now()})
            finally:
                with self.lock:
                    state["process"] = None
                    self.workers.pop(sid, None)
                if log is not None:
                    try:
                        log.close()
                    except OSError:
                        pass

    def _run_repair(self, session, argv, path, state, proxy_url=None):
        def finish(emit):
            emit(f"\n[LP] {session['title']}完成\n")
            try:
                self.engines.rescan()
                emit("[LP] 已刷新引擎列表；缺失源码的提示会随之更新\n")
            except Exception as exc:
                emit(f"[LP] 刷新引擎列表失败，可手动点击“刷新发现”：{getattr(exc, 'message', exc)}\n")

        self._run_git_session(
            session, argv, path, state, proxy_url,
            action="补齐", cancel_message="补齐已取消", directory_label="引擎目录",
            failure_message=lambda code: f"命令以退出码 {code} 结束，请检查网络或仓库访问权限",
            finish=finish,
        )

    def _run_clone(self, session, argv, staging, target, state, proxy_url=None, commit=None):
        def finish(emit):
            if commit:
                # Pin the fetched version: an engine is only supported at the
                # commit its adapter was written against, not wherever the
                # branch has moved on to.
                emit(f"[LP] 检出指定提交 {commit}\n")
                checkout = subprocess.run(
                    [argv[0], "-c", "core.fsmonitor=false", "-C", str(staging), "checkout", "--quiet", commit],
                    capture_output=True, text=True, timeout=120,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if checkout.returncode:
                    detail = (checkout.stderr or checkout.stdout or "").strip()
                    raise RuntimeError(f"检出提交 {commit} 失败：{detail}")
            with self.lock, self.engines.lock:
                if target.exists():
                    raise RuntimeError(f"拉取期间 engine 目录下出现了同名目录 {target.name}")
                os.replace(staging, target)
            emit(f"\n[LP] 已拉取到 {target}\n")
            try:
                self.engines.rescan()
                emit("[LP] 已刷新引擎列表；请在引擎管理中确认引擎类型并安装环境\n")
            except Exception as exc:
                emit(f"[LP] 刷新引擎列表失败，可手动点击“刷新发现”：{getattr(exc, 'message', exc)}\n")

        def cleanup():
            # Only ever the hidden folder this job created; the real target is untouched.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        self._run_git_session(
            session, argv, Path(target).parent, state, proxy_url,
            action="拉取", cancel_message="拉取已取消", directory_label="目标目录",
            failure_message=lambda code: f"git clone 失败（退出码 {code}），请检查地址、网络或分支名",
            finish=finish, cleanup=cleanup,
        )

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
