"""Copied source discovery is static. Only explicit diagnosis executes Python."""
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from adapters.registry import ADAPTERS, domain
from app.services.common import ServiceError, identity, now, local_path, atomic_text

DIAGNOSTIC_CODE = "import sys,json,importlib.util; import torch; import accelerate; import transformers; import safetensors; print('LP_DIAG='+json.dumps(dict(python_version=sys.version.split()[0], executable=sys.executable, torch_version=torch.__version__, cuda_wheel=torch.version.cuda, cuda_available=torch.cuda.is_available(), diffusers=importlib.util.find_spec('diffusers') is not None, yaml=importlib.util.find_spec('yaml') is not None, toml=importlib.util.find_spec('toml') is not None)))"

SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".cache", "outputs", "output", "models", "datasets", "logs"}


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
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.lock = threading.RLock()

    def get(self, key):
        value = self.store.get("engine", key)
        if value is None: raise ServiceError("找不到引擎安装实例", 404)
        return value

    def list(self):
        default = self.store.get("meta", "default_engine")
        tasks = self.store.list("task")
        result = self.store.list("engine")
        for r in result:
            r["is_default"] = r["installation_id"] == default
            r["referenced_by"] = sum(t["installation_id"] == r["installation_id"] for t in tasks)
            if not Path(r["source_path"]).is_dir(): r["state"] = "missing"
        return result

    def rescan(self):
        with self.lock:
            root = Path(self.settings()["engine_root"])
            if not root.is_dir(): raise ServiceError("引擎目录不存在，请在设置中配置现有目录")
            seen = set()
            for path in sorted(root.iterdir()):
                if not path.is_dir() or path.is_symlink() or path.name.startswith("."): continue
                path = path.resolve()
                key = identity(os.path.normcase(str(path)))
                seen.add(key)
                candidates = [a.engine_id for a in ADAPTERS.values() if a.detect(path).matched]
                old = self.store.get("engine", key)
                rev = revision(path)
                changed = old and old["revision"] != rev
                typ = old["engine_id"] if old and old["engine_id"] in candidates else candidates[0] if len(candidates) == 1 else None
                record = dict(installation_id=key, engine_id=typ, label=path.name, source_path=str(path),
                    management_mode="user_managed", revision=rev, state="discovered", verification="unverified",
                    environment_id=None, python_executable=None, python_version=None, torch_version=None, cuda_wheel=None,
                    is_default=False, referenced_by=0, candidate_engines=candidates, issues=[], discovered_at=now())
                if old:
                    for k in ("environment_id", "python_executable", "python_version", "torch_version", "cuda_wheel", "discovered_at", "environment_manifest_id"):
                        record[k] = old.get(k)
                    if not changed and old["state"] not in ("missing",):
                        record["state"], record["verification"], record["issues"] = old["state"], old["verification"], old["issues"]
                if changed:
                    record["issues"] = ["源码已变化，须重新诊断环境；已提交任务不会自动使用新源码"]
                if not typ: record["issues"] = ["无法唯一识别引擎类型，请手动确认"]
                if typ == "ai_toolkit": record["issues"] = list(dict.fromkeys(record["issues"] + ["已支持版本登记；训练配置适配尚未实现"]))
                self.store.put("engine", key, record)
            for old in self.store.list("engine"):
                if old["installation_id"] not in seen:
                    self.store.patch("engine", old["installation_id"], dict(state="missing", issues=["安装目录已移除或不在当前引擎目录中"]))
            return self.list()

    def confirm(self, key, engine_id):
        with self.lock:
            r = self.get(key)
            if not ADAPTERS[engine_id].detect(Path(r["source_path"])).matched:
                raise ServiceError("所选类型缺少必要的引擎文件，不能仅靠名称确认")
            self.store.patch("engine", key, dict(engine_id=engine_id, state="discovered", verification="unverified", issues=["类型已确认，请诊断环境"]))

    def bind(self, key, executable):
        p = local_path(executable)
        if not p.is_file(): raise ServiceError("Python 解释器不存在")
        with self.lock:
            self.get(key)
            self.store.patch("engine", key, dict(python_executable=str(p), state="discovered", verification="unverified", environment_id=None))
            self.diagnose(key)

    def diagnose(self, key):
        with self.lock:
            r = self.get(key)
            if not r["engine_id"]: raise ServiceError("请先确认引擎类型")
            path = Path(r["source_path"])
            if not path.is_dir(): raise ServiceError("引擎源码目录不存在")
            python = r["python_executable"]
            if not python: raise ServiceError("请先选择 Python 解释器；应用不会自动执行复制的环境")
            adapter = ADAPTERS[r["engine_id"]]
            if not adapter.detect(path).matched: raise ServiceError("引擎文件不完整")
            self.store.patch("engine", key, dict(state="preparing"))
            # -I prevents importing code from the copied engine or current directory during diagnosis.
            code = DIAGNOSTIC_CODE
            try:
                proc = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                lines = [l[8:] for l in proc.stdout.splitlines() if l.startswith("LP_DIAG=")]
                if proc.returncode or not lines:
                    raise ValueError((proc.stderr or proc.stdout)[-3000:] or "解释器未返回环境信息")
                info = json.loads(lines[-1])
                if not info["cuda_available"]: raise ValueError("未检测到可用 CUDA GPU；当前训练适配不支持 CPU 模式")
                if not info["diffusers"] or not info["toml"] or not info["yaml"]: raise ValueError("缺少 diffusers、toml 或 PyYAML，请在引擎独立环境中按其要求安装")
                rev = revision(path)
                env_id = identity(str(Path(python).resolve()), json.dumps(info, sort_keys=True))
                manifest_id = identity(env_id, rev["fingerprint"])
                atomic_text(self.store.root / "manifests" / f"{manifest_id}.json", json.dumps(info, ensure_ascii=False, indent=2))
                self.store.patch("engine", key, dict(state="ready", verification="experimental", environment_id=env_id,
                    environment_manifest_id=manifest_id, revision=rev, python_version=info["python_version"], torch_version=info["torch_version"], cuda_wheel=info["cuda_wheel"],
                    issues=["环境基础诊断通过，尚未进行该版本训练兼容性认证"] + (["AI Toolkit 训练配置适配尚未实现"] if r["engine_id"] == "ai_toolkit" else [])))
            except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
                self.store.patch("engine", key, dict(state="failed", verification="unverified", issues=[f"环境诊断失败：{exc}"]))
                raise ServiceError(f"环境诊断失败：{exc}") from exc

    def set_default(self, key):
        self.get(key)
        self.store.put("meta", "default_engine", key)

    def capabilities(self, key):
        r = self.get(key)
        if not r["engine_id"]: raise ServiceError("请先确认引擎类型")
        return ADAPTERS[r["engine_id"]].capabilities(domain(r))
