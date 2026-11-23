"""Prepare or reuse the private, relocatable Python installation used by an engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
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


def _find_managed(uv: Path, version: str, env: dict) -> Path | None:
    result = subprocess.run(
        [str(uv), "python", "find", "--no-project", "--managed-python", "--no-python-downloads", version],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        return None
    interpreter = Path(result.stdout.strip())
    if not interpreter.is_file():
        return None
    return interpreter.parent if os.name == "nt" else interpreter.parent.parent


def _managed_python_dir(uv: Path, version: str, env: dict) -> Path:
    source = _find_managed(uv, version, env)
    if source is not None:
        return source
    # Earlier LP releases kept each version in a subdirectory. Reuse its copy
    # in place without moving a base interpreter another environment may reference.
    if env.get("UV_PYTHON_INSTALL_DIR"):
        legacy = Path(env["UV_PYTHON_INSTALL_DIR"]) / version
        if legacy.is_dir():
            source = _find_managed(uv, version, {**env, "UV_PYTHON_INSTALL_DIR": str(legacy)})
            if source is not None:
                return source
    print(f"[LP] Downloading managed CPython {version}", flush=True)
    command = [str(uv), "python", "install", "--no-bin", version]
    if os.name == "nt":
        command.append("--no-registry")
    # Inherit the session's output streams so uv's actual download failure and
    # mirror URL reach the installation log and official-source retry logic.
    result = subprocess.run(command, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"uv python install {version} failed ({result.returncode})")
    source = _find_managed(uv, version, env)
    if source is None:
        raise RuntimeError(f"uv installed CPython {version} but could not locate its interpreter")
    return source


def _embedded_python(environment: Path) -> Path:
    return environment / ("python.exe" if os.name == "nt" else "bin/python")


def _runtime_matches(environment: Path, version: str) -> bool:
    python = _embedded_python(environment)
    if not python.is_file():
        return False
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", "import json, ssl, sys; print(json.dumps([sys.prefix, sys.version_info[:3]]))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return False
        prefix, actual = json.loads(result.stdout)
        wanted = [int(part) for part in version.split(".")]
        return Path(prefix).resolve() == environment.resolve() and actual[:len(wanted)] == wanted
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return False


def _prepare_layout(environment: Path, version: str, uv: Path, env: dict) -> None:
    python = _embedded_python(environment)
    if os.name == "nt":
        stdlib = environment / "Lib"
        pth = environment / _pth_name(version)
        if not pth.exists():
            pth.write_text("# LP embedded Python (relative paths)\n.\nLib\nLib/site-packages\nDLLs\n"
                           "../..\n../../..\n../../../..\n../../src\n", encoding="utf-8")
    else:
        stdlib = environment / "lib" / ("python" + ".".join(version.split(".")[:2]))
    site = stdlib / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for marker in (stdlib / "EXTERNALLY-MANAGED", site / "EXTERNALLY-MANAGED"):
        marker.unlink(missing_ok=True)
    if not (site / "pip" / "__main__.py").is_file():
        # The application ships a pure-Python pip. Use it without any download
        # on a clean installation. Engine-specific build dependencies belong to
        # the package installation plan, which knows the required versions.
        bundled = Path(sys.prefix) / "Lib/site-packages"
        if (bundled / "pip" / "__main__.py").is_file():
            for item in [bundled / "pip", *bundled.glob("pip-*.dist-info")]:
                shutil.copytree(item, site / item.name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    probe = subprocess.run([str(python), "-I", "-m", "pip", "--version"], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if probe.returncode:
        # A partial pip installation should not cost an engine its installed
        # training packages. Prefer the wheel bundled with this CPython for repair.
        wheel = next((stdlib / "ensurepip/_bundled").glob("pip-*.whl"), None)
        command = [str(uv), "pip", "install", "--python", str(python), "--reinstall", "--no-deps", str(wheel or "pip")]
        if subprocess.run(command, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode:
            raise RuntimeError("Could not repair pip in the engine interpreter")
        subprocess.run([str(python), "-I", "-m", "pip", "--version"], env=env, check=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def create_embedded(engine_root: Path, environment: Path, python_version: str, uv: Path) -> None:
    """Prepare a private CPython, keeping the old tree until a replacement is usable."""
    engine_root = engine_root.resolve()
    environment = environment.resolve()
    uv = uv.resolve()
    if not engine_root.is_dir():
        raise ValueError(f"Engine directory does not exist: {engine_root}")
    if environment == engine_root or not _inside(environment, engine_root):
        raise ValueError("Engine environment must stay inside the engine directory")
    if not uv.is_file():
        raise FileNotFoundError(f"Bundled uv is missing: {uv}")
    if not re.fullmatch(r"3\.\d+(?:\.\d+)?", python_version):
        raise ValueError("Invalid Python version")

    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHON", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT",
                 "UV_PYTHON", "UV_SYSTEM_PYTHON", "UV_PYTHON_PREFERENCE", "UV_NO_MANAGED_PYTHON"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", UV_NO_PROGRESS="1")

    if (environment / (_pth_name(python_version) + ".lp-install")).exists():
        raise RuntimeError("The package installer has not yet restored this interpreter's path file")
    if _runtime_matches(environment, python_version):
        _prepare_layout(environment, python_version, uv, env)
        print(f"[LP] Reusing embedded CPython {python_version}: {environment}", flush=True)
        return

    source = _managed_python_dir(uv, python_version, env)
    print(f"[LP] Preparing embedded CPython from {source}", flush=True)
    environment.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lp-bootstrap-", dir=environment.parent) as work:
        candidate = Path(work) / "python"
        shutil.copytree(source, candidate, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _prepare_layout(candidate, python_version, uv, env)
        if not _runtime_matches(candidate, python_version):
            raise RuntimeError(f"Prepared interpreter does not match the requested Python {python_version}")
        backup = None
        if environment.exists():
            backup = environment.with_name(f"{environment.name}.broken-{time.time_ns()}")
            environment.rename(backup)
        try:
            candidate.rename(environment)
        except OSError:
            if backup is not None:
                backup.rename(environment)
            raise
        if backup is not None:
            print(f"[LP] Previous interpreter preserved at {backup}", flush=True)
    print(f"[LP] Embedded engine Python ready: {_embedded_python(environment)}", flush=True)


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
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[LP] Failed to create engine environment: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
