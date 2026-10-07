"""Bootstrap the bundled uv executable without requiring system Python or pip.

This script intentionally uses only the application runtime standard library.
It downloads the latest official Windows uv archive into the project runtime
folder and atomically installs uv.exe at the requested path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import urllib.request
import zipfile


API = "https://api.github.com/repos/astral-sh/uv/releases/latest"


def _asset_name() -> str:
    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or platform.machine()
    ).lower()
    if "arm64" in arch or "aarch64" in arch:
        return "uv-aarch64-pc-windows-msvc.zip"
    if "64" in arch or "amd" in arch or "x86_64" in arch:
        return "uv-x86_64-pc-windows-msvc.zip"
    raise RuntimeError(f"不支持的 Windows 架构: {arch}")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LP-Lora-Trainer/1.0", "Accept": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def install(target: Path) -> None:
    target = target.expanduser().resolve()
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    asset = _asset_name()
    print(f"[LP] 获取官方 uv release 信息: {asset}", flush=True)
    request = urllib.request.Request(
        API,
        headers={"User-Agent": "LP-Lora-Trainer/1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    url = next((
        item.get("browser_download_url")
        for item in release.get("assets", [])
        if item.get("name") == asset
    ), None)
    if not url:
        raise RuntimeError(f"官方 uv release 未包含当前架构资源: {asset}")

    with tempfile.TemporaryDirectory(prefix="lp-uv-", dir=str(target.parent)) as temp:
        temp_dir = Path(temp)
        archive = temp_dir / asset
        print(f"[LP] 下载 uv: {url}", flush=True)
        _download(url, archive)
        extract = temp_dir / "extract"
        extract.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            member = next((
                name for name in bundle.namelist()
                if Path(name).name.lower() == "uv.exe"
            ), None)
            if member is None:
                raise RuntimeError("官方 uv 压缩包中没有 uv.exe")
            # Extract only the executable; never trust archive paths.
            with bundle.open(member) as source, (extract / "uv.exe").open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        # Keep the staging file inside the private temporary directory. Failed
        # downloads/replaces cannot leave a shared .new file or partial binary.
        os.replace(extract / "uv.exe", target)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("用法: bootstrap_uv.py <runtime/uv/uv.exe>")
    try:
        install(Path(sys.argv[1]))
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"[LP] uv 准备失败: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    print(f"[LP] uv 已准备: {Path(sys.argv[1]).resolve()}", flush=True)
