"""Read cache headers before reuse, or stage live captions alongside resized images.

Runs inside the engine interpreter with the engine checkout as working directory.
"""

from pathlib import Path
import shutil
import sys


def verify(original, staged, cache, vocab, recursive):
    from library.preprocess import walk_images
    from library.preprocess.text import _te_cache_path
    from library.anima.vocab_pack import load_vocab_pack
    from safetensors import safe_open

    cache = Path(cache)
    if not cache.is_dir():
        return
    original, staged = Path(original), Path(staged)
    digest = None
    for image in walk_images(staged, recursive=bool(int(recursive)), pattern="*"):
        path = _te_cache_path(original / image.relative_to(staged), cache, original)
        if not path.is_file():
            continue
        if digest is None:
            pack = load_vocab_pack(vocab)
            digest = pack.digest if pack else ""
        with safe_open(str(path), framework="pt") as handle:
            cached = (handle.metadata() or {}).get("vocab_pack_sha", "")
        if cached != digest:
            raise SystemExit(f"文本缓存与当前 CJK 词表包不匹配：{path}。请选择新的缓存目录或重建该文本缓存。")


def captions(original, staged, extension):
    from library.preprocess import walk_images

    original, staged = Path(original), Path(staged)
    for image in walk_images(staged, recursive=True, pattern="*"):
        dest = image.with_suffix(extension)
        source = original / dest.relative_to(staged)
        if source.is_file():
            shutil.copy2(source, dest)
        elif dest.is_file():
            dest.unlink()  # A previously staged copy whose source caption was removed.


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    if sys.argv[1] == "verify":
        verify(*sys.argv[2:])
    else:
        captions(*sys.argv[2:])
