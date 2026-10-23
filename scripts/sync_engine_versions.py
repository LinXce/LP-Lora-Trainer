#!/usr/bin/env python
"""Synchronise the LP version manifest with the engine git baselines.

Typical release flow after pulling an engine and updating its adapter:

    python scripts/sync_engine_versions.py --bump patch

It bumps ``APP_VERSION`` (and pyproject/README badges), records the current git
ref/commit of every known engine under that version, and writes
``app/engine_compat.json``. Re-running with ``--bump none`` refreshes the current
version's entries without changing the version number.

``--check`` is read-only: it fails when an installed engine no longer matches the
baseline recorded for the current ``APP_VERSION``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "app" / "engine_compat.json"

# One installation directory per adapter (the first existing one wins). A version
# may carry several engine entries; this table only chooses what to record.
ENGINE_DIRS = {
    "anima_lora": ("anima_lora",),
    "kohaku_moss_trainer": ("KohakuMossTrainer",),
    "kohya": ("sd-scripts", "kohya_ss"),
    "musubi_tuner": ("musubi-tuner",),
    "ai_toolkit": ("ai-toolkit",),
}


def _run_git(path: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=10,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"版本号必须是 x.y.z：{value}")
    return tuple(int(part) for part in match.groups())


def bump_version(current: str, part: str) -> str:
    major, minor, patch = _version_tuple(current)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return current


def _replace(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"无法在 {path} 中找到版本号位置")
    path.write_text(updated, encoding="utf-8", newline="\n")


def write_app_version(version: str) -> None:
    _replace(ROOT / "app" / "__init__.py", r'^APP_VERSION = "[^"]+"', f'APP_VERSION = "{version}"')
    _replace(ROOT / "pyproject.toml", r'^version = "[^"]+"', f'version = "{version}"')
    for name in ("README.md", "README.zh-CN.md"):
        path = ROOT / name
        if path.is_file():
            _replace(path, r"badge/version-[0-9]+\.[0-9]+\.[0-9]+-", f"badge/version-{version}-")


def collect_engines(engine_root: Path) -> list[dict]:
    from adapters.registry import ADAPTERS

    entries = []
    for engine_id, candidates in ENGINE_DIRS.items():
        adapter = ADAPTERS.get(engine_id)
        if adapter is None:
            continue
        directory = next((engine_root / candidate for candidate in candidates if (engine_root / candidate).is_dir()), None)
        if directory is None:
            print(f"skip {engine_id}: no engine directory", file=sys.stderr)
            continue
        try:
            ref = _run_git(directory, "rev-parse", "--abbrev-ref", "HEAD") or "HEAD"
            commit = _run_git(directory, "rev-parse", "HEAD")
        except (RuntimeError, OSError) as exc:
            print(f"skip {engine_id}: {exc}", file=sys.stderr)
            continue
        entries.append({
            "engine_id": engine_id,
            "ref": ref,
            "commit": commit,
            "adapter_version": adapter.adapter_version,
        })
    return entries


def load_manifest() -> dict:
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema", 1)
    data.setdefault("versions", [])
    return data


def write_manifest(data: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".engine-compat-", suffix=".json", dir=MANIFEST.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, MANIFEST)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", default=str(ROOT / "engine"))
    parser.add_argument("--version", help="record exactly this version instead of the current APP_VERSION")
    parser.add_argument("--bump", choices=("none", "patch", "minor", "major"), default="none",
                        help="bump APP_VERSION/pyproject/README before recording (default: none)")
    parser.add_argument("--check", action="store_true", help="only verify the current baseline")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from app import APP_VERSION

    if args.check:
        from app.services.compat import engine_compat
        data = load_manifest()
        entry = next((e for e in data.get("versions", []) if e.get("version") == APP_VERSION), None)
        if entry is None:
            print(f"{APP_VERSION}: no engine baseline recorded", file=sys.stderr)
            return 1
        entries = {e["engine_id"]: e for e in entry.get("engines", [])}
        bad = 0
        for engine in collect_engines(Path(args.engine_root)):
            current = {"engine_id": engine["engine_id"], "revision": {"requested_ref": engine["ref"], "commit": engine["commit"]}}
            status = engine_compat(current, engine["adapter_version"], version=APP_VERSION)
            mark = "OK" if status["status"] == "supported" else "MISMATCH"
            print(f"{mark:8} {engine['engine_id']:22} {engine['ref']} {engine['commit'][:12]}")
            if status["status"] != "supported":
                bad += 1
        return 1 if bad else 0

    version = args.version or bump_version(APP_VERSION, args.bump)
    entries = collect_engines(Path(args.engine_root))
    if not entries:
        print("没有找到可记录的引擎 git 仓库", file=sys.stderr)
        return 1
    if args.bump != "none":
        write_app_version(version)

    data = load_manifest()
    record = {"version": version, "released_at": date.today().isoformat(), "engines": entries}
    versions = [v for v in data["versions"] if not (isinstance(v, dict) and v.get("version") == version)]
    versions.append(record)
    versions.sort(key=lambda item: _version_tuple(str(item.get("version") or "0.0.0")), reverse=True)
    data["versions"] = versions
    write_manifest(data)
    print(f"recorded {version} with {len(entries)} engine(s):")
    for engine in entries:
        print(f"  {engine['engine_id']:22} {engine['ref']} {engine['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
