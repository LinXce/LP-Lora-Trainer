"""Checkpoint precision helpers for Musubi Tuner.

Musubi's Wan trainer asserts that ``--mixed_precision`` matches the DiT
checkpoint's own storage dtype (fp16 weights -> ``fp16``/``no``, bf16 weights ->
``bf16``/``no``), so a fixed default aborts every run against the other dtype.
The probe below reads only the safetensors header; every failure path returns
``None`` so rendering a config never depends on the model file being readable
(the pure parameter export must keep working for a missing file).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

_HEADER_LIMIT = 16 * 1024 * 1024
# Stored safetensors dtypes mapped to the precision families the engine knows.
_FLOAT_DTYPES = {
    "F64": "fp32",
    "F32": "fp32",
    "BF16": "bf16",
    "F16": "fp16",
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
}

_CACHE: dict[tuple[str, int, int], str | None] = {}
_CACHE_ORDER: list[tuple[str, int, int]] = []
_CACHE_LIMIT = 64


def _header(path: Path) -> dict | None:
    """Parsed safetensors header, or None when the file is not a usable checkpoint."""
    try:
        size = path.stat().st_size
        if size < 9:
            return None
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= min(size - 8, _HEADER_LIMIT):
                return None
            payload = handle.read(length)
        header = json.loads(payload)
    except (OSError, ValueError, struct.error, UnicodeError):
        return None
    return header if isinstance(header, dict) else None


def weight_dtype(path) -> str | None:
    """Read the same cross-attention key selected by upstream detect_wan_sd_dtype."""
    if not path or not isinstance(path, (str, Path)):
        return None
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    key = (str(target), stat.st_size, stat.st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    header = _header(target)
    result = None
    if header is not None:
        for name in ("model.diffusion_model.blocks.0.cross_attn.k.weight", "blocks.0.cross_attn.k.weight"):
            if name in header:
                entry = header[name]
                if isinstance(entry, dict):
                    result = _FLOAT_DTYPES.get(str(entry.get("dtype")))
                break
    _CACHE[key] = result
    _CACHE_ORDER.append(key)
    if len(_CACHE_ORDER) > _CACHE_LIMIT:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)
    return result
