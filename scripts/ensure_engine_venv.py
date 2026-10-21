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
import shutil
import subprocess
import sys
import time


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
    # project, interpreter, or package index. ``UV_PYTHON_PREFERENCE`` is
    # scrubbed as well: ``--managed-python`` below already selects it, and uv
    # rejects a command that passes both ("the argument `--managed-python`
    # cannot be used with `--python-preference`"). ``UV_PYTHON_INSTALL_MIRROR``
    # is deliberately kept: it is the only way to download a managed
    # interpreter when GitHub is unreachable.
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHON",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_NO_MANAGED_PYTHON",
        "UV_SYSTEM_PYTHON",
        "UV_PYTHON_PREFERENCE",
    ):
        child_env.pop(name, None)
    child_env.update(
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        UV_NO_PROGRESS="1",
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        tail = " / ".join(line.strip() for line in output.splitlines() if line.strip())[-400:]
        detail = f": {tail}" if tail else ""
        raise RuntimeError(f"uv venv failed with exit code {completed.returncode}{detail}")
    if not expected.is_file():
        raise RuntimeError(f"uv completed but did not create {expected}")
    print(f"[LP] Engine Python ready: {expected}", flush=True)


def _pth_name(version: str) -> str:
    """"python312._pth" for a "3.12.15" style version."""
    parts = version.split(".")
    return f"python{parts[0]}{parts[1] if len(parts) > 1 else '0'}._pth"


def _managed_python_dir(uv: Path, python_version: str, env: dict) -> Path:
    """Ask uv where its managed CPython for this version lives.

    Falls back to a direct scan of ``UV_PYTHON_INSTALL_DIR`` because ``uv python
    find`` only sees the managed installations under the install dir it resolves
    for *this* environment, which a caller may have pointed elsewhere.
    """
    completed = subprocess.run(
        [str(uv), "python", "find", "--managed-python", python_version],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    text = (completed.stdout or "").strip().splitlines()
    if completed.returncode == 0 and text:
        interpreter = Path(text[-1].strip())
        if interpreter.is_file():
            return interpreter.parent
    install_dir = env.get("UV_PYTHON_INSTALL_DIR")
    if install_dir:
        candidates = sorted(
            (p for p in Path(install_dir).glob(f"cpython-{python_version}.*-windows*") if p.is_dir()),
            reverse=True,
        )
        for candidate in candidates:
            if (candidate / "python.exe").is_file():
                return candidate
    raise RuntimeError(
        "uv could not locate a managed CPython "
        f"{python_version}: {(completed.stderr or completed.stdout or '').strip()[-300:]}"
    )


def create_embedded(engine_root: Path, environment: Path, python_version: str, uv: Path) -> None:
    """Copy a managed CPython into the engine and make it relocatable.

    This is the ComfyUI ``python_embeded`` shape: no ``pyvenv.cfg``, no ``Scripts``
    shims, no ``activate`` scripts. A ``pythonXY._pth`` with relative entries makes
    the interpreter find its own ``Lib`` / ``Lib/site-packages`` / ``DLLs`` after the
    whole project is copied to another drive, so nothing has to be rebased later.
    """
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

    # Embedded layout: the interpreter sits at the environment root, not in Scripts/.
    expected = environment / ("python.exe" if os.name == "nt" else "bin/python")
    if expected.is_file():
        # A copied CPython can be left behind half-written (interrupted copy, deleted
        # DLLs), and a bare existence check would then report "already exists" while
        # every later pip/uv call fails against a broken interpreter. Start it once.
        try:
            probe = subprocess.run(
                [str(expected), "-I", "-c", "import sys; print(sys.prefix)"],
                capture_output=True, text=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            probe = None
        healthy = (
            probe is not None
            and probe.returncode == 0
            and probe.stdout.strip() == str(environment)
        )
        if healthy:
            print(f"[LP] Embedded engine Python already exists: {expected}", flush=True)
            return
        broken = environment.with_name(f"{environment.name}.broken-{int(time.time() * 1e6)}")
        environment.rename(broken)
        print(f"[LP] Existing interpreter failed its probe; moved aside: {broken}", flush=True)

    child_env = os.environ.copy()
    # `--managed-python` cannot be combined with `--python-preference` or a pinned
    # `UV_PYTHON`, and uv rejects the command outright. The installation service sets
    # UV_PYTHON_PREFERENCE=only-managed for other flows, so it must be cleared here.
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHON", "VIRTUAL_ENV",
                 "UV_PROJECT_ENVIRONMENT", "UV_PYTHON", "UV_SYSTEM_PYTHON",
                 "UV_PYTHON_PREFERENCE", "UV_NO_MANAGED_PYTHON"):
        child_env.pop(name, None)
    child_env.update(PYTHONNOUSERSITE="1", UV_NO_PROGRESS="1", UV_PYTHON_DOWNLOADS="automatic")

    source = _managed_python_dir(uv, python_version, child_env)
    print(f"[LP] Embedding managed CPython from {source}", flush=True)
    environment.parent.mkdir(parents=True, exist_ok=True)
    if environment.exists():
        # Never delete a user's environment: a failed or partial copy is moved aside so
        # it can still be inspected (and it may hold packages worth diagnosing).
        import time as _time
        aside = environment.with_name(f"{environment.name}.broken-{int(_time.time())}")
        try:
            environment.rename(aside)
            print(f"[LP] Moved the incomplete environment aside: {aside}", flush=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot replace {environment}: {exc}") from exc
    shutil.copytree(source, environment, symlinks=False)

    # The venv-free interpreter needs an explicit, relative search path. No "import
    # site" line: user-site and stray .pth hooks must never execute (same rule as
    # scripts/build_python_runtime.py).
    (environment / _pth_name(python_version)).write_text(
        "# Paths are relative to this interpreter, not the current directory.\n"
        "# A ._pth replaces sys.path entirely: the engine root, its parent and a\n"
        "# conventional src/ must be listed at their real depths, or engine scripts\n"
        "# cannot import their own packages.\n"
        ".\nLib\nLib/site-packages\nDLLs\n../..\n../../..\n../../../..\n../../src\n",
        encoding="utf-8",
    )
    (environment / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)

    # uv's managed CPython ships PEP 668's EXTERNALLY-MANAGED marker, which makes pip
    # refuse every install. This copy is a private engine interpreter that LP owns, so
    # the marker is removed rather than worked around with --break-system-packages.
    # pip looks next to the stdlib as well as in site-packages, so both are cleared.
    for marker in (environment / "Lib" / "EXTERNALLY-MANAGED",
                   environment / "Lib" / "site-packages" / "EXTERNALLY-MANAGED"):
        try:
            marker.unlink()
        except OSError:
            pass

    # A copied CPython carries no pip, and ensurepip cannot work here: the ._pth
    # disables `site`, which ensurepip needs; uv refuses to touch a "managed"
    # interpreter. Copy the seeding packages straight out of the application runtime
    # instead: no network, no uv, no site hook.
    bundled = Path(sys.prefix) / "Lib" / "site-packages"
    targets = environment / "Lib" / "site-packages"
    if bundled.is_dir() and str(bundled.resolve()) != str(targets.resolve()):
        for pattern in ("pip", "pip-*", "setuptools", "setuptools-*", "wheel", "wheel-*", "pkg_resources"):
            for item in bundled.glob(pattern):
                if item.is_dir():
                    destination = targets / item.name
                    if destination.exists():
                        shutil.rmtree(destination, ignore_errors=True)
                    shutil.copytree(item, destination, symlinks=False,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if not (targets / "setuptools").is_dir():
        # The application runtime deliberately ships no build tooling, so the build
        # backend comes from the index the installation already uses.
        seed = subprocess.run(
            [str(expected), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
             "setuptools", "wheel"],
            env={**os.environ, "PIP_NO_INPUT": "1", "PYTHONNOUSERSITE": "1"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if seed.returncode != 0:
            print(f"[LP] setuptools seeding failed: {(seed.stderr or seed.stdout or '').strip()[-200:]}",
                  flush=True)

    probe = subprocess.run(
        [str(expected), "-I", "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if probe.returncode != 0 or not (probe.stdout or "").strip():
        raise RuntimeError(f"Embedded interpreter does not start: {(probe.stderr or '').strip()[-300:]}")
    if not _inside(Path(probe.stdout.strip()), environment):
        raise RuntimeError(f"Embedded interpreter reports a foreign prefix: {probe.stdout.strip()}")
    print(f"[LP] Embedded engine Python ready: {expected}", flush=True)


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
    parser.add_argument("--embedded", action="store_true",
                        help="copy a managed CPython instead of creating a venv (relocatable)")
    args = parser.parse_args(argv)
    try:
        if args.embedded:
            create_embedded(args.engine_root, args.environment, args.python_version, args.uv)
        else:
            create_environment(args.engine_root, args.environment, args.python_version, args.uv)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[LP] Failed to create engine environment: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
