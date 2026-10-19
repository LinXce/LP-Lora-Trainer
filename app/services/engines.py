"""Copied source discovery is static. Only explicit diagnosis executes Python."""
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from app.services import hardware
from app.services import install_plan as install_recommendation
from app.services.common import ServiceError, identity, now, local_path, atomic_text

# Only torch must execute to verify CUDA. Check the other adapter requirements
# without eagerly importing unrelated packages or any engine source code.
DIAGNOSTIC_CODE = "import sys,json,importlib.util; import torch; print('LP_DIAG='+json.dumps(dict(python_version=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix, torch_version=torch.__version__, cuda_wheel=torch.version.cuda, cuda_available=torch.cuda.is_available(), **{name:importlib.util.find_spec(name) is not None for name in (sys.argv[1:] or ['accelerate','transformers','safetensors','diffusers','yaml','toml'])})))"


SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".cache", "outputs", "output", "models", "datasets", "logs", ".lp"}

TYPE_UNKNOWN = "\u65e0\u6cd5\u552f\u4e00\u8bc6\u522b\u5f15\u64ce\u7c7b\u578b\uff0c\u8bf7\u624b\u52a8\u786e\u8ba4"
TYPE_AUTO = "\u7c7b\u578b\u5df2\u81ea\u52a8\u8bc6\u522b\uff0c\u8bf7\u8bca\u65ad\u73af\u5883"
TYPE_CONFIRMED = "\u7c7b\u578b\u5df2\u786e\u8ba4\uff0c\u8bf7\u8bca\u65ad\u73af\u5883"
TYPE_STALE = "\u672c\u6b21\u626b\u63cf\u672a\u5b8c\u6574\u5339\u914d\u6e90\u7801\uff0c\u5df2\u4fdd\u7559\u5f53\u524d\u5f15\u64ce\u7c7b\u578b\uff1b\u8bf7\u68c0\u67e5\u5b50\u6a21\u5757\u6216\u6e90\u7801\u540e\u518d\u8bca\u65ad"
TYPE_MARKERS = {TYPE_UNKNOWN, TYPE_AUTO, TYPE_CONFIRMED, TYPE_STALE}
KNOWN_ENGINE_IDS = ADAPTERS.keys()

# Presentation order for the base-model-type catalog (common image models first).
BASE_MODEL_ORDER = (
    "sd1", "sd2", "sdxl", "sd3", "flux1", "flux2", "pixart", "pixart_sigma", "auraflow",
    "lumina", "anima", "vega", "ssd", "wan21", "wan", "hunyuan_image", "hunyuan_video",
    "hunyuan_video_15", "qwen_image", "z_image", "hidream", "hidream_o1", "ideogram4",
    "kandinsky5", "krea2", "flex1", "flex2", "framepack", "flux1_kontext", "minimax_h3",
)

def _clean_locator(value):
    """Strip surrounding quotes/spaces from a stored path.

    ``pyvenv.cfg`` values are bare, but an edited or older file can hand a quoted path
    to the launcher, which then reports ``No Python at '"H:..."'``. Normalising when
    the value is read keeps a quoted record usable without rewriting the file first.
    """
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def revision(path):
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".venv") and not (Path(root) / d).is_symlink())
        for name in sorted(files):
            p = Path(root) / name
            if p.is_symlink() or p.suffix.lower() not in (".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".cfg", ".ini"):
                continue
            digest.update(p.relative_to(path).as_posix().encode())
            try:
                with p.open("rb") as f:
                    while chunk := f.read(1024 * 1024): digest.update(chunk)
            except OSError as exc:
                raise ServiceError(f"无法读取引擎源码：{p}: {exc}") from exc
    result = dict(source=None, requested_ref=None, commit=None, fingerprint=digest.hexdigest())
    if (path / ".git").exists():
        for key, args in (("commit", ["rev-parse", "HEAD"]), ("requested_ref", ["symbolic-ref", "--short", "HEAD"]), ("source", ["config", "--get", "remote.origin.url"])):
            try:
                proc = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(path), *args], capture_output=True, text=True, timeout=5,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if proc.returncode == 0:
                    value = proc.stdout.strip() or None
                    if key == "source" and value:
                        from urllib.parse import urlsplit, urlunsplit
                        parsed = urlsplit(value)
                        if parsed.scheme and parsed.netloc:
                            value = urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
                    result[key] = value
            except (OSError, subprocess.TimeoutExpired): pass
    return result


class EngineService:
    def __init__(self, store, settings, project_root=None):
        self.store, self.settings = store, settings
        self.project_root = Path(project_root or store.root.parent).resolve()
        self.lock = threading.RLock()
        self.installing = set()
        # Detection is static (file presence). Cache it so the SSE-driven
        # ``list()`` does not re-scan every engine source on each tick.
        self._detect_cache = {}

    def _invalidate_detection(self):
        self._detect_cache.clear()

    def require_idle(self, key=None):
        if self.installing and (key is None or key in self.installing):
            raise ServiceError("引擎环境正在安装，请先等待或在终端停止安装", 409)

    def get(self, key):
        value = self.store.get("engine", key)
        if value is None: raise ServiceError("找不到引擎安装实例", 404)
        return value

    def _detect_candidates(self, path):
        """Detect a cloned engine from source files only; never import or run it.

        Detection is a source hint, not the durable identity of an installation.
        A clone can be temporarily incomplete (for example, an uninitialised
        git submodule), so callers must not clear a previously selected type
        just because this pass returns no candidates.
        """
        path = Path(path)
        try:
            cache_key = os.path.normcase(str(path.resolve()))
        except OSError:
            cache_key = os.path.normcase(str(path))
        cached = self._detect_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        if not path.is_dir():
            return []
        candidates = []
        for adapter in ADAPTERS.values():
            try:
                result = adapter.detect(path)
                # Adapters are the source of truth for detection, but never
                # let a malformed/legacy adapter result create an invalid type.
                engine_id = str(getattr(result, "engine_id", "")).strip()
                if result.matched and engine_id in KNOWN_ENGINE_IDS:
                    candidates.append(engine_id)
            except (OSError, UnicodeError, ValueError, TypeError):
                # One broken adapter must not make the whole engine directory
                # disappear from the scan result.
                continue
        detection = list(dict.fromkeys(candidates))
        self._detect_cache[cache_key] = detection
        return list(detection)

    @staticmethod
    def _python_candidates(source_path):
        """Return common, existing Python interpreters inside one engine clone.

        Engine source is copied into ``engine/<name>`` and its environment is
        intentionally user-managed. Discovery therefore only suggests files
        that already exist; it never creates an environment or falls back to
        the application runtime/system Python.
        """
        root = Path(source_path)
        if not root.is_dir():
            return []
        if os.name == "nt":
            preferred = (
                ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
                "env/Scripts/python.exe", "Scripts/python.exe", "python.exe",
            )
            patterns = ("*/Scripts/python.exe", "*/*/Scripts/python.exe",
                        "*/python.exe", "*/*/python.exe")
        else:
            preferred = (
                ".venv/bin/python", "venv/bin/python", "env/bin/python",
                "bin/python", "python",
            )
            patterns = ("*/bin/python", "*/*/bin/python",
                        "*/python", "*/*/python")

        paths = [root / relative for relative in preferred]
        for pattern in patterns:
            try:
                paths.extend(root.glob(pattern))
            except (OSError, RuntimeError):
                continue

        # The normal virtual-environment directories are deliberately not in
        # this exclusion list.  ``SKIP`` is used by source fingerprinting and
        # includes ``.venv``/``venv``/``env`` because those directories are not
        # source code; here they are exactly the directories we need to find.
        python_skip = SKIP - {".venv", "venv", "env"}
        root_resolved = root.resolve()
        candidates = []
        for candidate in paths:
            try:
                candidate = candidate.resolve()
                relative_parts = candidate.relative_to(root_resolved).parts
                if any(part in python_skip for part in relative_parts):
                    continue
                if candidate.is_file() and candidate not in candidates:
                    candidates.append(candidate)
            except (OSError, RuntimeError, ValueError):
                continue
        return [str(candidate) for candidate in candidates]

    @staticmethod
    def _engine_type(record, candidates):
        """Return the durable type without allowing a scan to erase it.

        ``engine_type_confirmed`` is written by the manual confirmation API.
        For old records, a non-null engine_id is also retained when detection is
        temporarily inconclusive, which provides a safe backwards-compatible
        migration from the pre-confirmation records.
        """
        old_type = record.get("engine_id")
        if not isinstance(old_type, str):
            old_type = None
        old_type = old_type.strip() if old_type else None
        candidates = [c for c in candidates if c in KNOWN_ENGINE_IDS]
        if record.get("engine_type_confirmed") and old_type in KNOWN_ENGINE_IDS:
            return old_type
        # A single adapter match is authoritative, including when an old
        # record was previously unclassified.  This is what makes a refresh
        # repair legacy records instead of putting them back in an unclassified state.
        if len(candidates) == 1:
            return candidates[0]
        # Detection can be temporarily incomplete while a git clone/submodule
        # is being updated.  Preserve an already selected valid type rather
        # than erasing it on one refresh.
        if old_type in KNOWN_ENGINE_IDS:
            return old_type
        return None

    def _type_issues(self, path, engine_id, candidates, confirmed=False, prefix=(), existing=()):
        """Update type-related notices while preserving environment diagnostics."""
        preserved = [issue for issue in existing if issue not in TYPE_MARKERS]
        adapter = ADAPTERS.get(engine_id)
        if adapter:
            preserved = [issue for issue in preserved if adapter.preserve_issue(issue)]
        if not engine_id or engine_id not in KNOWN_ENGINE_IDS:
            return list(dict.fromkeys(tuple(prefix) + tuple(preserved) + (TYPE_UNKNOWN,)))
        adapter = ADAPTERS[engine_id]
        issues = list(prefix) + preserved
        if confirmed:
            issues.append(TYPE_CONFIRMED)
        elif len(candidates) == 1 and candidates[0] == engine_id:
            issues.append(TYPE_AUTO)
        elif engine_id not in candidates:
            issues.append(TYPE_STALE)
        issues.extend(adapter.source_issues(path))
        if adapter.training_notice:
            issues.append(adapter.training_notice)
        return list(dict.fromkeys(issues))

    def _find_existing(self, key, path):
        """Find an existing installation even when path spelling changed.

        The installation id is derived from the normalized absolute path. A
        record created before relocation (or with a different drive-letter
        case, slash style, or relative spelling) can therefore have a
        different id even though it points at the same directory. Reusing
        that record prevents a rescan from creating a new unclassified
        installation and marking the old one as missing.
        """
        record = self.store.get("engine", key)
        if record:
            return record
        try:
            canonical = os.path.normcase(str(Path(path).resolve()))
        except OSError:
            return None
        for candidate in self.store.list("engine"):
            source = candidate.get("source_path")
            if not source:
                continue
            try:
                if os.path.normcase(str(Path(source).resolve())) == canonical:
                    return candidate
            except (OSError, RuntimeError, ValueError):
                continue
        return None

    def _source_repair(self, path, engine_id):
        """Label of the adapter's source-completion step, or None when not needed.

        Computed together with the type issues (never from ``list()``): it reads
        the engine directory, which the per-second poll must not do.
        """
        adapter = ADAPTERS.get(engine_id)
        if adapter is None or not adapter.source_repair_label:
            return None
        try:
            command = adapter.source_repair(path)
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        return adapter.source_repair_label if command else None

    def _retarget(self, record):
        """Follow the project: re-point a record whose absolute path no longer exists.

        Engine source belongs under ``<project>/engine``. After the project folder is
        copied or moved, every recorded absolute path is stale, so the same directory
        name is looked up under the current project root before the engine is treated
        as missing. A bound interpreter inside a moved clone no longer starts, so the
        record returns to "discovered" and needs a rebind and diagnose.
        """
        path = Path(record.get("source_path", ""))
        if path.is_dir():
            # The directory exists, but a clone replaced in place (renamed onto an old
            # name) keeps stale environment metadata. Only a broken or quoted
            # interpreter path makes that worth checking, so healthy records stay free
            # of side effects here.
            bound = _clean_locator(record.get("python_executable"))
            if not bound or Path(bound).is_file():
                return record
        else:
            path = Path(record.get("source_path", ""))
        candidate = self._engine_root() / path.name
        if not candidate.is_dir():
            candidate = self._match_renamed(record)
        if not candidate.is_dir():
            return record
        changes = {"source_path": str(candidate.resolve())}
        candidates = self._python_candidates(candidate)
        if candidates:
            changes["python_candidates"] = candidates
        # The environment still names the previous engine directory — a moved project
        # or a renamed clone — so rebase it from that directory onto this one instead
        # of demanding a reinstall.
        old_root = path
        new_root = candidate.resolve()
        adapter = ADAPTERS.get(record.get("engine_id"))
        for name in (adapter.environment_candidates if adapter else (".venv", "venv")):
            environment = candidate / name
            if not environment.is_dir():
                continue
            if self._environment_is_stale(environment, old_root):
                self._repair_engine_environment(environment, old_root, new_root)
                changes["verification"] = "unverified"
            break
        bound = record.get("python_executable")
        if bound and not Path(bound).is_file():
            rebased = self._rebase_inside(bound, old_root, new_root)
            if rebased and Path(rebased).is_file():
                changes["python_executable"] = rebased
                changes["verification"] = "unverified"
            else:
                changes["python_executable"] = next((c for c in candidates if Path(c).is_file()), None)
                changes["state"] = "discovered"
                changes["verification"] = "unverified"
        self._invalidate_detection()
        return self.store.patch("engine", record["installation_id"], changes)

    @staticmethod
    def _environment_issues(record, path):
        """Light check on a recorded interpreter: present, inside, and it starts.

        Refreshing an engine must say "this environment is broken" without a manual
        diagnose. Only the interpreter process is started here (``-I``, no imports
        beyond ``sys``); the module/CUDA check remains in :meth:`diagnose`.
        """
        python = _clean_locator(record.get("python_executable"))
        if not python:
            return ["尚未绑定 Python 解释器"]
        try:
            # Resolved on both sides: 8.3 short names and case differences must compare
            # equal, or a healthy engine is reported as "interpreter outside".
            interpreter = Path(python).resolve()
            if not interpreter.is_file():
                return ["解释器路径不存在，请重新绑定或修复环境"]
            if not interpreter.is_relative_to(path.resolve()):
                return ["解释器不在当前引擎目录内，请重新绑定"]
            run = subprocess.run(
                [str(interpreter), "-I", "-c", "import sys"], capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return ["解释器无法启动，请修复或重新安装环境"]
        if run.returncode != 0:
            lines = (run.stderr or run.stdout or "").strip().splitlines()
            detail = lines[-1][:160] if lines else "未知错误"
            return ["解释器无法启动：" + detail]
        return []

    def _match_renamed(self, record):
        """Find a renamed clone: the only directory under engine/ of this engine type.

        Renaming an engine folder changes the recorded path, and a name-based lookup
        then fails, which used to hide the record (and its environment binding) from
        the list entirely. The adapter signature identifies it instead, but only a
        unique match is accepted, so two checkouts of one engine are never swapped.
        """
        adapter = ADAPTERS.get(record.get("engine_id"))
        if adapter is None:
            return Path()
        matches = []
        for child in self._engine_root().iterdir():
            if not child.is_dir() or child.is_symlink() or child.name.startswith("."):
                continue
            try:
                if adapter.detect(child).matched:
                    matches.append(child.resolve())
            except (OSError, ValueError):
                continue
        return matches[0] if len(matches) == 1 else Path()

    def _rebase_inside(self, value, old_root, new_root):
        """Rebase a recorded path from the previous engine directory onto this one."""
        try:
            suffix = Path(value).relative_to(old_root)
        except (ValueError, OSError):
            return None
        return str(Path(new_root) / suffix)

    @staticmethod
    def _environment_is_stale(environment, old_root):
        """True while the environment metadata still names the previous project root."""
        cfg = Path(environment) / "pyvenv.cfg"
        try:
            return cfg.is_file() and str(old_root) in cfg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    def _repair_engine_environment(self, environment, old_root, new_root):
        """Rebase an engine environment in place; never downloads or removes anything."""
        script = self.project_root / "scripts" / "repair_engine_venv.py"
        if not script.is_file():
            return False
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(environment),
                 "--old-root", str(old_root), "--new-root", str(new_root)],
                cwd=str(self.project_root), capture_output=True, text=True, timeout=180,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _engine_root(self):
        """The engine directory of *this* project, repairing a stale stored root."""
        configured = (self.settings() or {}).get("engine_root")
        root = Path(configured) if configured else self.project_root / "engine"
        if root.is_dir():
            return root
        fallback = self.project_root / "engine"
        if fallback.is_dir():
            if configured and str(fallback) != str(configured):
                # The project moved: keep the setting pointing at this project.
                self.store.patch("meta", "settings", {"engine_root": str(fallback)})
            return fallback
        return root

    def remove(self, installation_id):
        """Forget one engine record. Files on disk are never touched.

        Refused while a task still references the installation, because task history
        displays it; otherwise the record disappears from the list for good.
        """
        record = self.store.get("engine", installation_id)
        if record is None:
            raise ServiceError("该引擎条目不存在")
        if any(t.get("installation_id") == installation_id for t in self.store.list("task")):
            raise ServiceError("该引擎实例已被任务引用，不能删除；请先处理相关任务", 409)
        with self.lock:
            self.store.delete("engine", installation_id)
        self._invalidate_detection()
        return record

    def _repair_detection(self, record):
        """Refresh detection metadata without erasing a selected engine type."""
        record = self._retarget(record)
        path = Path(record.get("source_path", ""))
        if not path.is_dir():
            return record
        candidates = self._detect_candidates(path)
        engine_id = self._engine_type(record, candidates)
        changes = {}
        # A single discovered interpreter is unambiguous, so it is bound here: after a
        # relocation or a fresh clone the user should not have to pick it manually.
        python_candidates = self._python_candidates(path)
        bound = _clean_locator(record.get("python_executable"))
        if python_candidates and (not bound or not Path(bound).is_file()) and len(python_candidates) == 1:
            changes["python_executable"] = python_candidates[0]
            changes["verification"] = "unverified"
        repair = self._source_repair(path, engine_id)
        if record.get("source_repair") != repair:
            changes["source_repair"] = repair
        if list(record.get("candidate_engines") or []) != candidates:
            changes["candidate_engines"] = candidates
        if record.get("engine_id") != engine_id:
            changes["engine_id"] = engine_id
            changes["state"] = "discovered"
            changes["verification"] = "unverified"
        if record.get("issues") != self._type_issues(
            path, engine_id, candidates,
            confirmed=bool(record.get("engine_type_confirmed")),
            existing=record.get("issues") or (),
        ):
            changes["issues"] = self._type_issues(
                path, engine_id, candidates,
                confirmed=bool(record.get("engine_type_confirmed")),
                existing=record.get("issues") or (),
            )
        if not changes:
            return record
        return self.store.patch("engine", record["installation_id"], changes)

    def list(self):
        """Read-only engine view; never scans source trees and never writes.

        The event stream calls this once per second per connected client, so it
        must stay side-effect free. Detection repair happens in :meth:`rescan`
        and on the diagnose/bind paths, never here.
        """
        with self.lock:
            default = self.store.get("meta", "default_engine")
            tasks = self.store.list("task")
            result = self.store.list("engine")
            repaired = []
            for r in result:
                if not Path(r.get("source_path", "")).is_dir():
                    # A removed directory is no longer shown after refresh; the
                    # record is kept so a restored clone reappears with its binding.
                    continue
                r["is_default"] = r["installation_id"] == default
                r["referenced_by"] = sum(t["installation_id"] == r["installation_id"] for t in tasks)
                python_candidates = list(r.get("python_candidates") or [])
                bound_python = r.get("python_executable")
                if bound_python:
                    try:
                        bound_path = Path(bound_python).resolve()
                        if bound_path.is_file() and str(bound_path) not in python_candidates:
                            # Prefer an interpreter discovered inside the engine
                            # clone for the install dialog.  Keep an existing
                            # external binding as a fallback so users do not
                            # lose a valid, intentionally shared environment.
                            python_candidates.append(str(bound_path))
                    except OSError:
                        pass
                r["python_candidates"] = python_candidates
                adapter = ADAPTERS.get(r.get("engine_id"))
                r["installation_sources"] = list(adapter.installation_sources(Path(r["source_path"]))) if adapter else []
                repaired.append(r)
            return repaired

    def rescan(self):
        """Rebuild every engine record from the current source tree.

        This is the only bulk repair path: it refreshes detection, repairs
        legacy/unclassified records and marks removed directories missing.
        """
        with self.lock:
            self.require_idle()
            self._invalidate_detection()
            root = Path(self.settings()["engine_root"])
            if not root.is_dir():
                raise ServiceError("\u5f15\u64ce\u76ee\u5f55\u4e0d\u5b58\u5728\uff0c\u8bf7\u5728\u8bbe\u7f6e\u4e2d\u914d\u7f6e\u73b0\u6709\u76ee\u5f55")
            seen = set()
            for path in sorted(root.iterdir()):
                if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                    continue
                path = path.resolve()
                key = identity(os.path.normcase(str(path)))
                candidates = self._detect_candidates(path)
                old = self._find_existing(key, path)
                installation_id = old.get("installation_id", key) if old else key
                seen.add(installation_id)
                rev = revision(path)
                changed = bool(old and old.get("revision") != rev)
                engine_id = self._engine_type(old or {}, candidates)

                if old:
                    record = dict(old)
                    record.setdefault("engine_type_confirmed", False)
                    # A directory that came back revives a record parked as missing.
                    revive = changed or old.get("state") == "missing"
                    record.update(
                        installation_id=installation_id,
                        engine_id=engine_id,
                        label=path.name,
                        source_path=str(path),
                        revision=rev,
                        state="discovered" if revive else old.get("state", "discovered"),
                        verification="unverified" if revive else old.get("verification", "unverified"),
                        candidate_engines=candidates,
                    )
                else:
                    record = dict(
                        installation_id=installation_id, engine_id=engine_id, label=path.name,
                        source_path=str(path), management_mode="user_managed",
                        revision=rev, state="discovered", verification="unverified",
                        environment_id=None, python_executable=None,
                        python_version=None, torch_version=None, cuda_wheel=None,
                        is_default=False, referenced_by=0, candidate_engines=candidates,
                        issues=[], source_repair=None, discovered_at=now(),
                        environment_manifest_id=None,
                        engine_type_confirmed=False,
                    )

                record["python_candidates"] = self._python_candidates(path)
                prefix = ("\u6e90\u7801\u5df2\u53d8\u5316\uff0c\u987b\u91cd\u65b0\u8bca\u65ad\u73af\u5883\uff1b\u5df2\u63d0\u4ea4\u4efb\u52a1\u4e0d\u4f1a\u81ea\u52a8\u4f7f\u7528\u65b0\u6e90\u7801",) if changed else ()
                record["issues"] = self._type_issues(
                    path, engine_id, candidates,
                    confirmed=bool(record.get("engine_type_confirmed")),
                    prefix=prefix,
                    existing=() if changed else (record.get("issues") or ()),
                )
                record["source_repair"] = self._source_repair(path, engine_id)
                # Refreshes must not stack the same notice: dedupe while preserving order.
                record["issues"] = list(dict.fromkeys(
                    list(record["issues"]) + self._environment_issues(record, path)))
                self.store.put("engine", installation_id, record)

            for old in self.store.list("engine"):
                if old["installation_id"] not in seen:
                    self.store.patch(
                        "engine", old["installation_id"],
                        dict(state="missing", issues=["\u5b89\u88c5\u76ee\u5f55\u5df2\u79fb\u9664\u6216\u4e0d\u5728\u5f53\u524d\u5f15\u64ce\u76ee\u5f55\u4e2d"]),
                    )
            return self.list()

    def confirm(self, key, engine_id):
        with self.lock:
            self.require_idle(key)
            r = self.get(key)
            self._invalidate_detection()
            adapter = ADAPTERS.get(engine_id)
            if adapter is None:
                raise ServiceError("\u672a\u77e5\u5f15\u64ce\u7c7b\u578b")
            path = Path(r["source_path"])
            if not path.is_dir():
                raise ServiceError("\u5f15\u64ce\u6e90\u7801\u76ee\u5f55\u4e0d\u5b58\u5728")
            # Manual confirmation still requires the adapter's static signature;
            # a directory name alone can never select an engine.  Incomplete
            # clones that the adapter can identify (such as Kohya with an
            # uninitialised sd-scripts submodule) are valid confirmations.
            try:
                detected = adapter.detect(path).matched
            except (OSError, UnicodeError, ValueError):
                detected = False
            if not detected:
                raise ServiceError("\u6240\u9009\u7c7b\u578b\u7f3a\u5c11\u5fc5\u8981\u7684\u5f15\u64ce\u6587\u4ef6\uff0c\u4e0d\u80fd\u4ec5\u9760\u540d\u79f0\u786e\u8ba4")
            candidates = self._detect_candidates(path)
            issues = self._type_issues(path, engine_id, candidates, confirmed=True)
            self.store.patch(
                "engine", key,
                dict(
                    engine_id=engine_id,
                    engine_type_confirmed=True,
                    candidate_engines=candidates,
                    state="discovered",
                    verification="unverified",
                    issues=issues,
                    source_repair=self._source_repair(path, engine_id),
                ),
            )

    def _engine_python(self, record, executable):
        """A training interpreter belongs to this engine, never the app/system."""
        python = local_path(_clean_locator(executable))
        source = Path(record["source_path"]).resolve()
        if python.is_relative_to(self.project_root / "python_runtime"):
            raise ServiceError("应用 python_runtime 仅用于管理界面，不能作为训练环境")
        if not python.is_relative_to(source) or python == source:
            raise ServiceError("请选择当前引擎目录内部的独立 Python 环境")
        if not python.is_file():
            raise ServiceError("引擎 Python 解释器不存在，请安装或修复引擎环境")
        return python

    def bind(self, key, executable):
        with self.lock:
            self.require_idle(key)
            # A record written by an older build can still carry a stale type;
            # binding is an explicit user action, so repairing it here is safe.
            record = self._repair_detection(self.get(key))
            python = self._engine_python(record, executable)
            self.store.patch("engine", key, dict(
                python_executable=str(python), state="discovered", verification="unverified", environment_id=None,
                python_candidates=self._python_candidates(record["source_path"])))
            self.diagnose(key)

    def diagnose(self, key, *, installation=False):
        with self.lock:
            if not installation:
                self.require_idle(key)
            self._invalidate_detection()
            r = self._repair_detection(self.get(key))
            if not r["engine_id"]: raise ServiceError("请先确认引擎类型")
            path = Path(r["source_path"])
            if not path.is_dir(): raise ServiceError("引擎源码目录不存在")
            adapter = ADAPTERS[r["engine_id"]]
            if not adapter.detect(path).matched: raise ServiceError("引擎文件不完整")
            source_issues = adapter.source_issues(path)
            if source_issues:
                self.store.patch("engine", key, dict(state="discovered", verification="unverified",
                                                     issues=list(source_issues),
                                                     source_repair=self._source_repair(path, r["engine_id"])))
                raise ServiceError("；".join(source_issues))
            python = _clean_locator(r["python_executable"])
            if not python: raise ServiceError("请先选择 Python 解释器；应用不会自动执行复制的环境")
            self.store.patch("engine", key, dict(state="preparing"))
            # -I prevents importing code from the copied engine or current directory during diagnosis.
            code = DIAGNOSTIC_CODE
            try:
                python = str(self._engine_python(r, python))
                # The module list is the adapter's: a fixed tuple here silently reported
                # every engine-specific module (torch aside) as missing.
                proc = subprocess.run([python, "-I", "-c", code, *adapter.diagnostic_modules],
                                      capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                lines = [l[8:] for l in proc.stdout.splitlines() if l.startswith("LP_DIAG=")]
                if proc.returncode or not lines:
                    raise ValueError((proc.stderr or proc.stdout)[-3000:] or "解释器未返回环境信息")
                info = json.loads(lines[-1])
                # The engine declares the interpreter LP installs for it; a different
                # minor can still be a valid user-managed environment, so this is a
                # notice on a passing diagnosis, never a failure.
                expected = str(getattr(adapter, "python_version", "") or "")
                actual = str(info.get("python_version") or "")
                python_notice = ("" if not expected or actual.startswith(expected) else
                                 f"环境 Python 为 {actual or '未知'}，适配器按 {expected} 安装；如遇异常请重建环境")
                prefix = Path(info["prefix"]).resolve()
                base_prefix = Path(info["base_prefix"]).resolve()
                if not prefix.is_relative_to(path.resolve()):
                    raise ValueError("解释器环境不在当前引擎目录内")
                if prefix == base_prefix:
                    portable = any(Path(python).parent.glob("python*._pth")) and prefix == Path(python).parent
                    if not portable:
                        raise ValueError("不允许系统 Python，请使用引擎独立环境")
                if not info["cuda_available"]: raise ValueError("未检测到可用 CUDA GPU；当前训练适配不支持 CPU 模式")
                missing = [name for name in adapter.diagnostic_modules if not info.get(name)]
                if missing:
                    raise ValueError("缺少 " + "、".join(missing) + "，请按当前引擎的官方流程修复独立环境")
                rev = revision(path)
                env_id = identity(str(Path(python).resolve()), json.dumps(info, sort_keys=True))
                manifest_id = identity(env_id, rev["fingerprint"])
                atomic_text(self.store.root / "manifests" / f"{manifest_id}.json", json.dumps(info, ensure_ascii=False, indent=2))
                self.store.patch("engine", key, dict(state="ready", verification="experimental", environment_id=env_id,
                    environment_manifest_id=manifest_id, revision=rev, python_version=info["python_version"], torch_version=info["torch_version"], cuda_wheel=info["cuda_wheel"],
                    issues=["环境基础诊断通过，尚未进行该版本训练兼容性认证"]
                    + ([python_notice] if python_notice else [])
                    + ([adapter.training_notice] if adapter.training_notice else [])))
            except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, ServiceError) as exc:
                self.store.patch("engine", key, dict(state="failed", verification="unverified", issues=[f"环境诊断失败：{exc}"]))
                raise ServiceError(f"环境诊断失败：{exc}") from exc

    def set_default(self, key):
        self.get(key)
        self.store.put("meta", "default_engine", key)

    def capabilities(self, key):
        r = self.get(key)
        if not r["engine_id"]: raise ServiceError("请先确认引擎类型")
        return ADAPTERS[r["engine_id"]].capabilities(domain(r))

    def install_plan(self, key, requested_source=None):
        """Read-only: recommend a CUDA source, mirrors and blockers for one install.

        Never raises for an unconfirmed/missing type — the result carries a
        ``blocked_reason`` so the UI can explain it.
        """
        record = self.get(key)
        adapter = ADAPTERS.get(record["engine_id"]) if record.get("engine_id") else None
        return install_recommendation.build(adapter, record, hardware.probe(), requested_source)

    def base_models(self):
        """Union of base-model types across registered engines, with engine support."""
        catalog = {}
        for r in self.list():
            if not r["engine_id"]: continue
            try:
                specs = ADAPTERS[r["engine_id"]].architectures(Path(r["source_path"]))
            except (OSError, ValueError):
                continue
            for spec in specs:
                entry = catalog.setdefault(spec.id, dict(id=spec.id, label=spec.label, engines=[]))
                if r["engine_id"] not in entry["engines"]:
                    entry["engines"].append(r["engine_id"])
        order = {name: index for index, name in enumerate(BASE_MODEL_ORDER)}
        return sorted(catalog.values(), key=lambda e: (order.get(e["id"], len(order)), e["label"]))

