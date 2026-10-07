"""Copied source discovery is static. Only explicit diagnosis executes Python."""
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from app.services.common import ServiceError, identity, now, local_path, atomic_text

# Only torch must execute to verify CUDA. Check the other adapter requirements
# without eagerly importing unrelated packages or any engine source code.
DIAGNOSTIC_CODE = "import sys,json,importlib.util; import torch; print('LP_DIAG='+json.dumps(dict(python_version=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix, torch_version=torch.__version__, cuda_wheel=torch.version.cuda, cuda_available=torch.cuda.is_available(), **{name:importlib.util.find_spec(name) is not None for name in ('accelerate','transformers','safetensors','diffusers','yaml','toml')})))"


SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".cache", "outputs", "output", "models", "datasets", "logs"}

TYPE_UNKNOWN = "\u65e0\u6cd5\u552f\u4e00\u8bc6\u522b\u5f15\u64ce\u7c7b\u578b\uff0c\u8bf7\u624b\u52a8\u786e\u8ba4"
TYPE_AUTO = "\u7c7b\u578b\u5df2\u81ea\u52a8\u8bc6\u522b\uff0c\u8bf7\u8bca\u65ad\u73af\u5883"
TYPE_CONFIRMED = "\u7c7b\u578b\u5df2\u786e\u8ba4\uff0c\u8bf7\u8bca\u65ad\u73af\u5883"
TYPE_STALE = "\u672c\u6b21\u626b\u63cf\u672a\u5b8c\u6574\u5339\u914d\u6e90\u7801\uff0c\u5df2\u4fdd\u7559\u5f53\u524d\u5f15\u64ce\u7c7b\u578b\uff1b\u8bf7\u68c0\u67e5\u5b50\u6a21\u5757\u6216\u6e90\u7801\u540e\u518d\u8bca\u65ad"
TYPE_MARKERS = {TYPE_UNKNOWN, TYPE_AUTO, TYPE_CONFIRMED, TYPE_STALE}
KNOWN_ENGINE_IDS = frozenset(ADAPTERS)
# Older builds wrote AI Toolkit notices with a mismatched console encoding.
# They are safe to remove because the adapter appends the canonical notice below;
# unrelated diagnostics (including real installer errors) remain untouched.
def _is_legacy_ai_toolkit_notice(issue):
    """Recognize old AI Toolkit notices damaged by the previous code page."""
    if not isinstance(issue, str):
        return False
    if issue == "?????????????":
        return True
    # Replacement characters and runs of question marks identify the old
    # damaged notices. A single question mark in a real diagnostic (offline?)
    # must not cause that diagnostic to disappear on refresh.
    return (issue.startswith(("AI Toolkit ", "\ufffd", "?"))
            and ("\ufffd" in issue or "???" in issue))



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
        return list(dict.fromkeys(candidates))

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
        # repair legacy records instead of putting them back in ???.
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
        if engine_id == "ai_toolkit":
            preserved = [issue for issue in preserved if not _is_legacy_ai_toolkit_notice(issue)]
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

    def _repair_detection(self, record):
        """Refresh detection metadata without erasing a selected engine type."""
        path = Path(record.get("source_path", ""))
        if not path.is_dir():
            return record
        candidates = self._detect_candidates(path)
        engine_id = self._engine_type(record, candidates)
        changes = {}
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
        with self.lock:
            default = self.store.get("meta", "default_engine")
            tasks = self.store.list("task")
            result = self.store.list("engine")
            repaired = []
            for r in result:
                if Path(r.get("source_path", "")).is_dir():
                    r = self._repair_detection(r)
                r["is_default"] = r["installation_id"] == default
                r["referenced_by"] = sum(t["installation_id"] == r["installation_id"] for t in tasks)
                python_candidates = self._python_candidates(r.get("source_path", ""))
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
                if not Path(r["source_path"]).is_dir():
                    r["state"] = "missing"
                repaired.append(r)
            return repaired

    def rescan(self):
        with self.lock:
            self.require_idle()
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
                    record.update(
                        installation_id=installation_id,
                        engine_id=engine_id,
                        label=path.name,
                        source_path=str(path),
                        revision=rev,
                        state="discovered" if changed else old.get("state", "discovered"),
                        verification="unverified" if changed else old.get("verification", "unverified"),
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
                        issues=[], discovered_at=now(),
                        environment_manifest_id=None,
                        engine_type_confirmed=False,
                    )

                prefix = ("\u6e90\u7801\u5df2\u53d8\u5316\uff0c\u987b\u91cd\u65b0\u8bca\u65ad\u73af\u5883\uff1b\u5df2\u63d0\u4ea4\u4efb\u52a1\u4e0d\u4f1a\u81ea\u52a8\u4f7f\u7528\u65b0\u6e90\u7801",) if changed else ()
                record["issues"] = self._type_issues(
                    path, engine_id, candidates,
                    confirmed=bool(record.get("engine_type_confirmed")),
                    prefix=prefix,
                    existing=() if changed else (record.get("issues") or ()),
                )
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
                ),
            )

    def _engine_python(self, record, executable):
        """A training interpreter belongs to this engine, never the app/system."""
        python = local_path(executable)
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
            record = self.get(key)
            python = self._engine_python(record, executable)
            self.store.patch("engine", key, dict(python_executable=str(python), state="discovered", verification="unverified", environment_id=None))
            self.diagnose(key)

    def diagnose(self, key, *, installation=False):
        with self.lock:
            if not installation:
                self.require_idle(key)
            r = self.get(key)
            if not r["engine_id"]: raise ServiceError("请先确认引擎类型")
            path = Path(r["source_path"])
            if not path.is_dir(): raise ServiceError("引擎源码目录不存在")
            adapter = ADAPTERS[r["engine_id"]]
            if not adapter.detect(path).matched: raise ServiceError("引擎文件不完整")
            source_issues = adapter.source_issues(path)
            if source_issues:
                self.store.patch("engine", key, dict(state="discovered", verification="unverified", issues=list(source_issues)))
                raise ServiceError("；".join(source_issues))
            python = r["python_executable"]
            if not python: raise ServiceError("请先选择 Python 解释器；应用不会自动执行复制的环境")
            self.store.patch("engine", key, dict(state="preparing"))
            # -I prevents importing code from the copied engine or current directory during diagnosis.
            code = DIAGNOSTIC_CODE
            try:
                python = str(self._engine_python(r, python))
                proc = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                lines = [l[8:] for l in proc.stdout.splitlines() if l.startswith("LP_DIAG=")]
                if proc.returncode or not lines:
                    raise ValueError((proc.stderr or proc.stdout)[-3000:] or "解释器未返回环境信息")
                info = json.loads(lines[-1])
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
                    issues=["环境基础诊断通过，尚未进行该版本训练兼容性认证"] + ([adapter.training_notice] if adapter.training_notice else [])))
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
