"""Adapter-owned launcher for Musubi Krea 2 stages that load the Qwen3-VL tokenizer.

Musubi still fetches the tokenizer by Hugging Face repo id. Both the text-encoder
cache stage and the training stage (when it encodes sample prompts) load it, so
both run through this launcher, which redirects that lookup to a user-provided
local directory. The cloned engine source stays untouched.

The repo id is read from the engine source (``krea2/krea2_encoder.py``) instead of
being duplicated here: if upstream renames it, the redirect must follow, and a
silent miss would make the engine download from the Hub again mid-job.
"""
from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path

# Only used when the engine source cannot be inspected (older/newer layout).
TOKENIZER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_ENGINE_SCRIPT = "krea2_cache_text_encoder_outputs.py"
ENCODER_SOURCE = Path("krea2") / "krea2_encoder.py"
_REPO_ASSIGNMENT = re.compile(r'^\s*QWEN3_VL_4B_INSTRUCT_REPO_ID\s*=\s*"([^"]+)"', re.MULTILINE)


def parse_adapter_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine_source", required=True)
    parser.add_argument("--engine_script", default=DEFAULT_ENGINE_SCRIPT)
    parser.add_argument("--tokenizer_path", default="")
    return parser.parse_known_args(argv)


def tokenizer_repo(engine_source) -> str | None:
    """The tokenizer repository id the engine itself requests, or None.

    None means "could not be determined": the caller decides whether to fall back
    to the built-in default and warn, so a layout change degrades visibly instead
    of silently pointing the redirect at a name nobody asks for.
    """
    source = Path(engine_source) / "src" / "musubi_tuner" / ENCODER_SOURCE
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = _REPO_ASSIGNMENT.search(text)
    return match.group(1).strip() if match else None


def patch_tokenizer(tokenizer_path: str, repo_id: str = TOKENIZER_REPO) -> None:
    import transformers

    original_auto = transformers.AutoTokenizer.from_pretrained
    original_qwen2 = transformers.Qwen2TokenizerFast.from_pretrained

    def redirect(original):
        def wrapped(pretrained_model_name_or_path, *args, **kwargs):
            if pretrained_model_name_or_path == repo_id:
                pretrained_model_name_or_path = tokenizer_path
            return original(pretrained_model_name_or_path, *args, **kwargs)

        return wrapped

    transformers.AutoTokenizer.from_pretrained = redirect(original_auto)
    transformers.Qwen2TokenizerFast.from_pretrained = redirect(original_qwen2)


def main() -> None:
    adapter_args, remaining = parse_adapter_args(sys.argv[1:])
    engine_source = Path(adapter_args.engine_source).expanduser().resolve()
    if not engine_source.is_dir():
        raise SystemExit(f"Engine source is not a directory: {engine_source}")

    engine_src = engine_source / "src"
    if not engine_src.is_dir():
        raise SystemExit(f"Engine source is not a directory: {engine_src}")
    # Only a script that ships inside the engine package may be launched.
    name = adapter_args.engine_script
    if Path(name).name != name or not name.endswith(".py"):
        raise SystemExit(f"Invalid engine script: {name}")
    script = engine_src / "musubi_tuner" / name
    if not script.is_file():
        raise SystemExit(f"Engine script not found: {script}")
    sys.path.insert(0, str(engine_src))

    tokenizer_path = adapter_args.tokenizer_path.strip()
    if tokenizer_path:
        tokenizer = Path(tokenizer_path).expanduser().resolve()
        if not tokenizer.is_dir():
            raise SystemExit(f"Tokenizer is not a directory: {tokenizer}")
        for file_name in ("tokenizer.json", "tokenizer_config.json"):
            if not (tokenizer / file_name).is_file():
                raise SystemExit(f"Tokenizer is missing file: {tokenizer / file_name}")
        repo_id = tokenizer_repo(engine_source)
        if repo_id is None:
            # Keep working with the known id, but say loudly that the redirect may
            # miss, because the engine would then fetch the tokenizer from the Hub.
            print(
                f"[lp] 未能在引擎源码 {ENCODER_SOURCE.as_posix()} 中找到 Qwen3-VL 分词器 repo id；"
                f"回退到 {TOKENIZER_REPO}。若本地分词器未被使用，请检查引擎版本或先放好该 repo 的 HF 缓存。",
                file=sys.stderr,
            )
        patch_tokenizer(str(tokenizer), repo_id or TOKENIZER_REPO)

    # Leave only the arguments understood by the unmodified engine script.
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
