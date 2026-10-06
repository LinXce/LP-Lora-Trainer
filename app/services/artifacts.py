"""Lightweight artifact validation shared by publishing and supervision."""
import json
import struct


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

