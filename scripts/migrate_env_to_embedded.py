"""Migrate one engine venv to a relocatable embedded CPython.

The engine keeps its dependency set: the script freezes the existing venv with uv,
creates ``<engine>/.lp/python`` (a copied CPython with a relative ``._pth``), installs
the same requirements there, verifies the engine's core imports, and only then renames
the old environment to ``<env>.migrated``. It never deletes anything and never touches
the engine source.

Usage:
    python scripts/migrate_env_to_embedded.py --engine-root engine/<name> \
        --from <engine>/.venv --python-version 3.12 --uv runtime/uv/uv.exe
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ensure_engine_venv import create_embedded  # noqa: E402

FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(command: list[str], *, env: dict | None = None, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        input=stdin, env=env, check=False, creationflags=FLAGS,
    )


def _tail(result: subprocess.CompletedProcess) -> str:
    text = (result.stderr or result.stdout or "").strip().splitlines()
    return text[-1][:300] if text else ""


def migrate(engine_root: Path, source_env: Path, python_version: str, uv: Path,
            verify: tuple[str, ...]) -> int:
    engine_root = engine_root.resolve()
    source_env = source_env.resolve()
    uv = uv.resolve()
    if not engine_root.is_dir():
        print(f"Engine directory does not exist: {engine_root}", file=sys.stderr)
        return 2
    source_python = source_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not source_python.is_file():
        print(f"No interpreter in {source_env}", file=sys.stderr)
        return 2

    print(f"[migrate] freezing {source_env}", flush=True)
    # Freeze with the venv's own pip: uv refuses an environment whose pyvenv.cfg points
    # at a base interpreter that no longer exists, which is exactly the state a moved
    # project leaves behind.
    freeze = run([str(source_python.resolve()), "-I", "-m", "pip", "freeze"])
    if freeze.returncode != 0:
        freeze = run([str(uv), "pip", "freeze", "--python", str(source_python.resolve())])
    if freeze.returncode != 0:
        print(f"[migrate] freeze failed: {_tail(freeze)}", file=sys.stderr)
        return 1
    requirements = freeze.stdout.strip()
    if not requirements:
        print("[migrate] the source environment is empty; nothing to migrate", file=sys.stderr)
        return 1

    target = engine_root / ".lp" / "python"
    print(f"[migrate] creating embedded CPython at {target}", flush=True)
    create_embedded(engine_root, target, python_version, uv)
    target_python = (target / ("python.exe" if os.name == "nt" else "bin/python")).resolve()

    print(f"[migrate] installing {len(requirements.splitlines())} requirement(s)", flush=True)
    install = run([str(uv), "pip", "install", "--python", str(target_python), "-r", "-"],
                  stdin=requirements)
    if install.returncode != 0:
        print(f"[migrate] install failed: {_tail(install)}", file=sys.stderr)
        print("[migrate] the old environment was left untouched", file=sys.stderr)
        return 1

    for module in verify:
        check = run([str(target_python), "-I", "-c", f"import {module}"])
        if check.returncode != 0:
            print(f"[migrate] verification failed (import {module}): {_tail(check)}", file=sys.stderr)
            print("[migrate] the old environment was left untouched", file=sys.stderr)
            return 1
        print(f"[migrate] import {module}: ok", flush=True)

    kept = source_env.with_name(source_env.name + ".migrated")
    if kept.exists():
        print(f"[migrate] {kept.name} already exists; leaving both environments in place", file=sys.stderr)
        return 1
    source_env.rename(kept)
    print(f"[migrate] done. Old environment kept as {kept.name}", flush=True)
    print(f"[migrate] point the engine record at {target_python} and diagnose again", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate an engine venv to a relocatable embedded CPython")
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--from", dest="source_env", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--python-install-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runtime" / "uv" / "python",
                        help="where uv keeps its managed CPython builds")
    parser.add_argument("--verify", action="append", default=[],
                        help="module that must import in the new environment (repeatable)")
    args = parser.parse_args(argv)
    # Standalone runs must resolve the same managed interpreters LP uses; without this
    # uv only sees its own default location and reports "No interpreter found".
    os.environ["UV_PYTHON_INSTALL_DIR"] = str(args.python_install_dir.resolve())
    try:
        return migrate(args.engine_root, args.source_env, args.python_version, args.uv,
                       tuple(args.verify))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[migrate] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
