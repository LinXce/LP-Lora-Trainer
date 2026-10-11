"""Adapter-owned launcher for Musubi Krea 2 text-encoder caching.

This keeps the cloned engine source untouched while optionally redirecting the
engine's official Qwen3-VL tokenizer lookup to a user-provided local directory.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

TOKENIZER_REPO = "Qwen/Qwen3-VL-4B-Instruct"


def parse_adapter_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine_source", required=True)
    parser.add_argument("--tokenizer_path", default="")
    return parser.parse_known_args(argv)


def patch_tokenizer(tokenizer_path: str) -> None:
    import transformers

    original_auto = transformers.AutoTokenizer.from_pretrained
    original_qwen2 = transformers.Qwen2TokenizerFast.from_pretrained

    def redirect(original):
        def wrapped(pretrained_model_name_or_path, *args, **kwargs):
            if pretrained_model_name_or_path == TOKENIZER_REPO:
                pretrained_model_name_or_path = tokenizer_path
            return original(pretrained_model_name_or_path, *args, **kwargs)

        return wrapped

    transformers.AutoTokenizer.from_pretrained = redirect(original_auto)
    transformers.Qwen2TokenizerFast.from_pretrained = redirect(original_qwen2)


def main() -> None:
    adapter_args, remaining = parse_adapter_args(sys.argv[1:])
    engine_source = Path(adapter_args.engine_source).expanduser().resolve()
    if not engine_source.is_dir():
        raise SystemExit(f"??????????{engine_source}")

    engine_src = engine_source / "src"
    if not engine_src.is_dir():
        raise SystemExit(f"??????????{engine_src}")
    sys.path.insert(0, str(engine_src))

    tokenizer_path = adapter_args.tokenizer_path.strip()
    if tokenizer_path:
        tokenizer = Path(tokenizer_path).expanduser().resolve()
        if not tokenizer.is_dir():
            raise SystemExit(f"Tokenizer ??????{tokenizer}")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if not (tokenizer / name).is_file():
                raise SystemExit(f"Tokenizer ?????{tokenizer / name}")
        patch_tokenizer(str(tokenizer))

    # Leave only the arguments understood by the unmodified engine script.
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path(
        str(engine_source / "src" / "musubi_tuner" / "krea2_cache_text_encoder_outputs.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
