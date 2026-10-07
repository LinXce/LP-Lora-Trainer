"""Create an engine-local virtual environment with the bundled uv executable.

This helper is intentionally dependency-free.  It is called by the application
runtime only for adapters whose official installer expects an already-created
venv (currently Kohya).  uv owns Python discovery and installation; the host
machine's Python installation is never used.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _python_path(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def create_environment(engine_root: Path, environment: Path, python_version: str, uv: Path) -> None:
    engine_root = engine_root.resolve()
    environment = environment.resolve()
    uv = uv.resolve()

    if not engine_root.is_dir():
        raise ValueError(f"Engine directory does not exist: {engine_root}")
    if environment == engine_root or not _inside(environment, engine_root):
        raise ValueError("Engine environment must stay inside the engine directory")
    if not uv.is_file():
        raise FileNotFoundError(f"Bundled uv is missing: {uv}")
    if not python_version or any(char.isspace() for char in python_version):
        raise ValueError("Invalid Python version")

    expected = _python_path(environment)
    if expected.is_file():
        print(f"[LP] Engine Python already exists: {expected}", flush=True)
        return

    environment.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(uv),
        "venv",
        str(environment),
        "--python",
        python_version,
        "--managed-python",
        "--seed",
    ]
    child_env = os.environ.copy()
    # Do not let a copied user's active environment redirect uv to another
    # project, interpreter, or package index.
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHON",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_NO_MANAGED_PYTHON",
        "UV_SYSTEM_PYTHON",
    ):
        child_env.pop(name, None)
    child_env.update(
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        UV_NO_PROGRESS="1",
        UV_PYTHON_PREFERENCE="only-managed",
        UV_PYTHON_DOWNLOADS="automatic",
    )
    if os.name == "nt":
        child_env.setdefault("PIP_NO_INPUT", "1")

    print("[LP] Creating engine-local Python environment:", flush=True)
    print("[LP] " + " ".join(_display(value) for value in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(engine_root),
        env=child_env,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"uv venv failed with exit code {completed.returncode}")
    if not expected.is_file():
        raise RuntimeError(f"uv completed but did not create {expected}")
    print(f"[LP] Engine Python ready: {expected}", flush=True)


def _display(value: str) -> str:
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an engine-local venv using bundled uv")
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--uv", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        create_environment(args.engine_root, args.environment, args.python_version, args.uv)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[LP] Failed to create engine environment: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
