"""Install a uv project's dependency groups into a non-venv interpreter.

Embedded environments have no venv, so ``uv sync`` cannot be used. This helper exports
the clone's own dependency set (respecting its pyproject groups and `[tool.uv.sources]`)
and installs it straight into the given interpreter:

    python scripts/install_uv_project.py --uv runtime/uv/uv.exe \
        --project engine/anima_lora --python engine/anima_lora/.lp/python/python.exe \
        --group cuda-windows

Nothing is guessed: the groups and version pins come from the project itself.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(command: list[str], cwd: Path, env: dict, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=str(cwd), env=env, input=stdin, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, creationflags=FLAGS,
    )


def install(uv: Path, project: Path, python: Path, groups: list[str],
            no_default_groups: bool, find_links: str | None) -> int:
    if not uv.is_file():
        print(f"uv is missing: {uv}", file=sys.stderr)
        return 2
    if not project.is_dir():
        print(f"project is missing: {project}", file=sys.stderr)
        return 2
    if not python.is_file():
        print(f"interpreter is missing: {python}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)

    export = [str(uv), "export", "--format", "requirements.txt", "--no-hashes",
              # The project itself is installed editable below; exporting it as a
              # pinned wheel-less requirement would shadow the engine's source.
              "--no-emit-project"]
    if no_default_groups:
        export.append("--no-default-groups")
    for group in groups:
        export.extend(["--group", group])
    exported = _run(export, project, env)
    if exported.returncode != 0:
        tail = (exported.stderr or exported.stdout or "").strip().splitlines()[-1:]
        print(f"uv export failed: {' '.join(tail)}", file=sys.stderr)
        return 1
    requirements = exported.stdout
    if not requirements.strip():
        print("uv export produced no requirements", file=sys.stderr)
        return 1

    command = [str(uv), "pip", "install", "--python", str(python), "-r", "-"]
    if find_links:
        command.extend(["--find-links", find_links])
    installed = _run(command, project, env, stdin=requirements)
    if installed.returncode != 0:
        tail = (installed.stderr or installed.stdout or "").strip().splitlines()[-1:]
        print(f"uv pip install failed: {' '.join(tail)}", file=sys.stderr)
        return 1

    # The engine must keep running from its own source tree, exactly as `uv sync`
    # would leave it: install the project editable into the same interpreter.
    editable = [str(uv), "pip", "install", "--python", str(python), "--no-build-isolation", "-e", str(project)]
    linked = _run(editable, project, env)
    if linked.returncode != 0:
        tail = (linked.stderr or linked.stdout or "").strip().splitlines()[-1:]
        print(f"editable install failed: {' '.join(tail)}", file=sys.stderr)
        return 1
    print(f"Installed {len(requirements.splitlines())} requirement line(s) into {python}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--no-default-groups", action="store_true")
    parser.add_argument("--find-links")
    args = parser.parse_args(argv)
    try:
        return install(args.uv.resolve(), args.project.resolve(), args.python.resolve(),
                       list(args.group), args.no_default_groups, args.find_links)
    except OSError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
