"""Best-effort local hardware probe: OS/arch and the NVIDIA driver/CUDA level.

Deliberately tolerant and cheap: any failure yields ``None`` fields instead of
raising, and results are cached briefly. Callers must not put this on the
per-second status path — it shells out to ``nvidia-smi``.
"""

import platform
import re
import subprocess
import threading
import time

_CACHE_TTL = 30.0
_cache: dict | None = None
_cache_at = 0.0
_lock = threading.Lock()

# nvidia-smi header wording varies: classic "Driver Version: 566.03 / CUDA Version: 12.6"
# vs newer "KMD Version: 616.64 / CUDA UMD Version: 13.4".
_DRIVER_LINE = re.compile(r"(?:Driver|KMD) Version:\s*([0-9]+(?:\.[0-9]+)?)")
_CUDA_LINE = re.compile(r"CUDA(?: UMD)? Version:\s*([0-9]+(?:\.[0-9]+)?)")
_NOT_SUPPORTED = {"", "n/a", "[n/a]", "not supported", "unknown"}


def _run(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _query(fields: str) -> list[str] | None:
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return None
    return [part.strip() for part in out.splitlines()[0].split(",")]


def reset_cache() -> None:
    """Drop the cached probe (tests and explicit refresh)."""
    global _cache, _cache_at
    with _lock:
        _cache, _cache_at = None, 0.0


def probe() -> dict:
    """Return {"os","arch","gpu_name","driver_version","cuda_version","compute_cap","cuda_usable"}.

    ``cuda_version`` is the driver's *maximum supported* CUDA runtime (from the
    nvidia-smi header), not the CUDA a torch wheel was built against.
    """
    global _cache, _cache_at
    with _lock:
        if _cache is not None and time.monotonic() - _cache_at < _CACHE_TTL:
            return dict(_cache)

    info: dict = {
        "os": platform.system(),
        "arch": platform.machine(),
        "gpu_name": None,
        "driver_version": None,
        "cuda_version": None,
        "compute_cap": None,
        "cuda_usable": False,
    }

    header = _run(["nvidia-smi"])
    if header:
        match = _CUDA_LINE.search(header)
        if match:
            info["cuda_version"] = match.group(1)
        match = _DRIVER_LINE.search(header)
        if match:
            info["driver_version"] = match.group(1)

    row = _query("name,driver_version,compute_cap")
    if row is None:
        # Older drivers do not support compute_cap in --query-gpu.
        row = _query("name,driver_version")
    if row and len(row) >= 2:
        info["gpu_name"] = row[0] or None
        info["driver_version"] = info["driver_version"] or (row[1] or None)
        if len(row) >= 3 and row[2].lower() not in _NOT_SUPPORTED:
            info["compute_cap"] = row[2]

    info["cuda_usable"] = bool(info["cuda_version"] and info["driver_version"])

    with _lock:
        _cache, _cache_at = dict(info), time.monotonic()
    return dict(info)


def cuda_rank(source: str | None) -> tuple[int, int] | None:
    """Rank a CUDA extra like ``cu124`` → ``(12, 4)``; None when not parseable."""
    if not isinstance(source, str) or not source.startswith("cu"):
        return None
    body = source[2:]
    if "." in body:
        major, _, minor = body.partition(".")
        return (int(major), int(minor)) if major.isdigit() and minor.isdigit() else None
    if body.isdigit() and len(body) >= 3:
        return (int(body[:-1]), int(body[-1]))
    return None


def driver_rank(cuda_version: str | None) -> tuple[int, int] | None:
    """Rank a driver CUDA version like ``12.6`` → ``(12, 6)``."""
    if not isinstance(cuda_version, str):
        return None
    parts = cuda_version.strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return (int(parts[0]), int(parts[1]))


def cuda_extra(rank: tuple[int, int] | None) -> str | None:
    """Inverse of :func:`cuda_rank` for a driver version: ``(12, 6)`` → ``cu126``."""
    return f"cu{rank[0]}{rank[1]}" if rank else None
