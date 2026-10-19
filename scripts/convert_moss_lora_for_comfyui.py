"""CLI wrapper around the adapter's converter: one implementation, not two.

Usage:
    engine/anima_lora/.venv/Scripts/python.exe scripts/convert_moss_lora_for_comfyui.py <in> [out]

Without ``out`` the copy is written next to the input as ``<name>_comfyui.safetensors``.
Run it from the project root so ``adapters`` is importable. The input is never modified.
"""

import sys
from pathlib import Path

# Running from anywhere: the project root holds the ``adapters`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.anima_lora.adapter import AnimaLoraAdapter


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_comfyui.safetensors")
    count = AnimaLoraAdapter.convert_lora_for_comfyui(source, target)
    print(f"in : {source}")
    print(f"out: {target}")
    print(f"tensors: {count}")


if __name__ == "__main__":
    main()
