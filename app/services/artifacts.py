"""Lightweight artifact validation shared by publishing and supervision."""
import json
import struct
from pathlib import Path


def checkpoint_complete(path):
    """Check safetensors header and payload size without loading tensors."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= length <= min(size - 8, 16 * 1024 * 1024): return False
            header = json.loads(f.read(length))
        tensors = [v for k,v in header.items() if k != "__metadata__"]
        if not tensors: return False
        widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                  "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}
        for tensor in tensors:
            width = widths.get(tensor["dtype"])
            if width is None: return False  # Unknown formats are never claimed usable.
            count = 1
            for dimension in tensor["shape"]:
                if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0: return False
                count *= dimension
            start, end = tensor["data_offsets"]
            if end - start != count * width: return False
        ranges = sorted(t["data_offsets"] for t in tensors)
        cursor = 0
        for start, end in ranges:
            if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end < start: return False
            cursor = end
        return size == 8 + length + cursor
    except (OSError, ValueError, KeyError, TypeError, struct.error, AttributeError): return False


def prune_missing_artifacts(store) -> int:
    """Delete artifact index records whose file is gone; return how many.

    Intended callers: the backend startup path and a periodic maintenance task.
    The record index is a cache of what the output folder contained, so a
    checkpoint deleted or moved by hand should stop appearing in lists.

    Boundary: only SQLite records are removed via ``store.delete("artifact", ...)``.
    Files on disk are never created, moved or deleted here -- pruning must not
    race a training job that is still writing its outputs. Other record kinds
    (``task``, ``task_output``, ``dataset``, ...) are left untouched, and a
    record without a usable ``artifact_id`` cannot be addressed, so it is kept.
    """
    removed = 0
    for record in store.list("artifact"):
        if not isinstance(record, dict):
            continue
        artifact_id = record.get("artifact_id")
        if not artifact_id:
            continue  # Unaddressable record: deleting by a guessed key could hit another one.
        path = record.get("path")
        try:
            present = bool(path) and Path(path).is_file()
        except (OSError, TypeError, ValueError):
            present = False  # Empty, non-path or malformed values count as missing.
        if present:
            continue
        store.delete("artifact", artifact_id)
        removed += 1
    return removed

