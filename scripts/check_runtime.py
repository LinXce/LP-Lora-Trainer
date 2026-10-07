"""Offline self-check. Execute with python_runtime/python.exe, not system Python."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import ssl
import subprocess
import sys
import tempfile


def check():
    runtime = Path(sys.executable).resolve().parent
    workspace = Path(__file__).resolve().parents[1]
    manifest_file = runtime / "runtime-manifest.json"
    if not manifest_file.is_file():
        raise RuntimeError("Run this check with python_runtime\\python.exe")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not sys.flags.isolated or not sys.flags.ignore_environment:
        raise RuntimeError("Python is not isolated from system environment settings")
    if Path(sys.prefix).resolve() != runtime or Path(sys.base_prefix).resolve() != runtime:
        raise RuntimeError("Interpreter still depends on a system Python or a venv")
    if "site" in sys.modules:
        raise RuntimeError("Site initialization unexpectedly enabled")
    if platform.python_version() != manifest["python_version"]:
        raise RuntimeError("Python version differs from runtime manifest")
    allowed = {runtime, runtime / "Lib", runtime / "DLLs",
               runtime / "Lib" / "site-packages", workspace}
    if any(Path(value).resolve() not in allowed for value in sys.path):
        raise RuntimeError(f"External import path detected: {sys.path}")
    for name, version in manifest["packages"].items():
        dist = metadata.distribution(name)
        if dist.version != version or not Path(dist.locate_file("")).resolve().is_relative_to(runtime):
            raise RuntimeError(f"External or mismatched dependency: {name}")
    modules = ("fastapi", "uvicorn", "PIL.Image", "pydantic_core", "webview", "pythonnet",
               "app.api.server", "desktop.launcher", "supervisor.worker", "adapters.registry")
    for name in modules:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(workspace):
            raise RuntimeError(f"External module detected: {name}")
    # clr is a dynamically generated managed module (__file__ == "unknown").
    # Its loader and Python.Runtime.dll are supplied by the bundled pythonnet.
    import clr  # noqa: F401; loads the Windows .NET bridge without opening a GUI
    with sqlite3.connect(":memory:") as connection:
        assert connection.execute("SELECT 42").fetchone()[0] == 42
    ssl.create_default_context()
    from io import BytesIO
    from PIL import Image
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
    image_bytes.seek(0)
    with Image.open(image_bytes) as image:
        image.load()
        assert image.size == (2, 2)
    env = os.environ.copy()
    env.update(PYTHONHOME="Z:\\missing-python", PYTHONPATH="Z:\\missing-packages",
               PYTHONUSERBASE="Z:\\missing-user-site")
    env["PATH"] = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
    # Same mechanism used for backend and supervisor spawning; no PATH lookup.
    result = subprocess.run(
        [sys.executable, "-c", "import sys,json,app,supervisor,ssl,sqlite3; "
         "print(json.dumps({'prefix':sys.base_prefix,'isolated':sys.flags.isolated,'paths':sys.path}))"],
        cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    child = json.loads(result.stdout)
    if Path(child["prefix"]).resolve() != runtime or not child["isolated"]:
        raise RuntimeError("Child process did not use the bundled interpreter")
    if {Path(value).resolve() for value in child["paths"]} != allowed:
        raise RuntimeError("Child process imported external paths")
    return {"ok": True, "python": platform.python_version(), "runtime": str(runtime),
            "packages": len(manifest["packages"]), "isolated": True,
            "frontend_built": (workspace / "frontend" / "dist" / "index.html").is_file(),
            "checks": ["imports", "native_extensions", "dotnet_bridge", "sqlite", "ssl",
                       "pillow", "child_process", "polluted_environment"]}


def main():
    try:
        print(json.dumps(check(), ensure_ascii=True, indent=2))
    except Exception as exc:
        print(f"Portable runtime check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
