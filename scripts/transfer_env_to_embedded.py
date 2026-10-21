"""Move an existing venv's packages into a relocatable embedded environment offline.

Nothing is downloaded: the already-installed packages are copied into the copied
CPython, pip's absolute editable hooks are rewritten into relative roots, and the
console scripts (``kogine.exe`` and friends) are regenerated from the installed
distributions with CPython's own ``ScriptMaker``.

Usage:
    python scripts/transfer_env_to_embedded.py --engine-root engine/<name> \
        --from <engine>/.venv --python-version 3.11 --uv runtime/uv/uv.exe \
        [--verify torch]
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
SKIP_NAMES = {"__pycache__"}


def _site(environment: Path) -> Path:
    return environment / ("Lib/site-packages" if os.name == "nt" else "lib/python3/site-packages")


def _copy_packages(source_site: Path, target_site: Path) -> int:
    copied = 0
    for item in source_site.iterdir():
        if item.name in SKIP_NAMES or item.name == "EXTERNALLY-MANAGED":
            continue
        destination = target_site / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(item, destination, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, destination)
        copied += 1
    return copied


def _regenerate_scripts(python: Path, site: Path) -> int:
    """Recreate console scripts in the interpreter's own directory."""
    data = Path("" if os.name != "nt" else "").resolve()
    venv_scripts = site.parent.parent / ("Scripts" if os.name == "nt" else "bin")
    helper = f"""
import importlib.metadata, os, sys
import venv
from pip._vendor.distlib.scripts import ScriptMaker
site = r"{site}"
target = r"{venv_scripts}"
entries = [e for d in importlib.metadata.distributions(path=[site])
           for e in d.entry_points
           if e.group in {{"console_scripts", "gui_scripts"}}]
maker = ScriptMaker(None, target)
maker.executable = r"{python}"
maker.variants = {{""}}
maker.clobber = True
count = 0
for entry in entries:
    try:
        maker.make(f"{{entry.name}} = {{entry.value}}",
                   {{"gui": entry.group == "gui_scripts"}})
        count += 1
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print(f"  skip {{entry.name}}: {{exc}}")
print(f"generated {{count}} console script(s)")
"""
    result = subprocess.run([str(python), "-c", helper], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False,
                            creationflags=FLAGS)
    print((result.stdout or "").strip())
    if result.returncode != 0:
        print((result.stderr or "").strip()[-400:], file=sys.stderr)
    return 0 if result.returncode == 0 else 1


def transfer(engine_root: Path, source_env: Path, python_version: str, uv: Path,
             verify: tuple[str, ...]) -> int:
    engine_root = engine_root.resolve()
    source_env = source_env.resolve()
    source_site = _site(source_env)
    if not source_site.is_dir():
        print(f"source site-packages is missing: {source_site}", file=sys.stderr)
        return 2

    target = engine_root / ".lp" / "python"
    create_embedded(engine_root, target, python_version, uv)
    target_site = _site(target)
    target_site.mkdir(parents=True, exist_ok=True)

    print(f"[transfer] copying packages {source_site} -> {target_site}", flush=True)
    copied = _copy_packages(source_site, target_site)
    print(f"[transfer] {copied} item(s) copied (no download)", flush=True)

    normalize = Path(__file__).resolve().parent / "normalize_editable.py"
    result = subprocess.run([str(target / ("python.exe" if os.name == "nt" else "bin/python")),
                             "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"embedded interpreter does not start: {(result.stderr or '').strip()[-300:]}", file=sys.stderr)
        return 1
    print(f"[transfer] {normalize}", flush=True)
    subprocess.run([sys.executable, "-u", str(normalize), str(target)], check=False,
                   creationflags=FLAGS)

    python = target / ("python.exe" if os.name == "nt" else "bin/python")
    _regenerate_scripts(python, target_site)

    for module in verify:
        check = subprocess.run([str(python), "-I", "-c", f"import {module}"],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", check=False, creationflags=FLAGS)
        if check.returncode != 0:
            print(f"[transfer] import {module} FAILED: {(check.stderr or '').strip().splitlines()[-1:]}",
                  file=sys.stderr)
            return 1
        print(f"[transfer] import {module}: ok", flush=True)

    kept = source_env.with_name(source_env.name + ".migrated")
    if not kept.exists():
        try:
            source_env.rename(kept)
            print(f"[transfer] old environment kept as {kept.name}", flush=True)
        except OSError as exc:
            print(f"[transfer] could not rename the old environment: {exc}", file=sys.stderr)
    print(f"[transfer] done: {python}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transfer a venv into a relocatable embedded CPython offline")
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--from", dest="source_env", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--python-install-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runtime" / "uv" / "python")
    parser.add_argument("--verify", action="append", default=[])
    args = parser.parse_args(argv)
    os.environ["UV_PYTHON_INSTALL_DIR"] = str(args.python_install_dir.resolve())
    try:
        return transfer(args.engine_root, args.source_env, args.python_version, args.uv,
                        tuple(args.verify))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[transfer] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
