"""Convert an anima_lora / MOSS LoRA into ComfyUI's diffusion-model LoRA layout.

The engine saves ``adapters.layers_<i>_proj_<name>.<up|down>.weight``; ComfyUI's
patcher looks for ``diffusion_model.layers.<i>.proj.<name>.lora_<up|down>.weight``
and reads ``ss_network_dim`` / ``ss_network_alpha`` from the file metadata.

Usage:
    engine/anima_lora/.venv/Scripts/python.exe scripts/convert_moss_lora_for_comfyui.py <in> [out]

Without ``out`` the converted file is written next to the input as
``<name>_comfyui.safetensors``. The script never modifies the input.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_KEY = re.compile(r"^adapters\.layers_(\d+)_proj_([a-z_]+)\.(up|down)\.weight$")


def convert(source: Path, target: Path) -> None:
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    converted, unmapped = {}, []
    for key, tensor in tensors.items():
        match = SOURCE_KEY.match(key)
        if match is None:
            unmapped.append(key)
            continue
        index, name, side = match.groups()
        converted[f"diffusion_model.layers.{index}.proj.{name}.lora_{side}.weight"] = tensor
    if not converted:
        raise SystemExit("没有匹配到任何 adapters.layers_* 键，这不是 anima_lora/MOSS 的 LoRA 吗？")
    converted_metadata = {
        "ss_network_dim": str(metadata.get("rank", "")),
        "ss_network_alpha": str(metadata.get("alpha", "")),
        "ss_network_module": "networks.lora",
        "converted_from": "anima_lora",
    }
    converted_metadata = {k: v for k, v in converted_metadata.items() if v}
    save_file(converted, str(target), metadata=converted_metadata)
    print(f"in : {source}")
    print(f"out: {target}")
    print(f"tensors: {len(converted)}  (dropped {len(unmapped)} unmatched)")
    print(f"dim={converted_metadata.get('ss_network_dim')}  alpha={converted_metadata.get('ss_network_alpha')}")
    for key in sorted(converted)[:4]:
        print("  " + key)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_comfyui.safetensors")
    convert(source, target)


if __name__ == "__main__":
    main()
