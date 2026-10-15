import codecs
import hashlib
import os
import threading
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from app.services.common import ServiceError, local_path, contained, identity, now, atomic_text

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# Tagging tools disagree on caption encoding: UTF-8 (with or without BOM) is the
# norm, but Japanese captions are often Shift-JIS on Windows and Notepad writes
# UTF-16. Decoding tolerantly keeps such images out of "missing caption".
UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def read_caption(path):
    raw = path.read_bytes()
    if raw.startswith(UTF16_BOMS):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeError:
        pass
    # Shift-JIS accepts almost any byte pair, so it is the fallback, not the first try.
    return raw.decode("cp932", errors="replace")


class DatasetService:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()

    def get(self, key):
        r = self.store.get("dataset", key)
        if not r: raise ServiceError("数据集不存在", 404)
        return r

    def list(self):
        return self.store.list("dataset")

    def add(self, path, name):
        p = local_path(path)
        if not p.is_dir(): raise ServiceError("数据集目录不存在")
        key = identity(os.path.normcase(str(p)))
        existing = self.store.get("dataset", key)
        if existing: return existing
        return self.store.put("dataset", key, dict(dataset_id=key, name=name.strip() or p.name, path=str(p), image_count=None, caption_count=None, issues=[], scanned_at=None))

    def remove(self, key):
        """Drop LP's index for one dataset; the folder on disk is never touched."""
        dataset = self.get(key)
        with self.lock:
            self.store.delete("dataset_images", key)
            self.store.delete("dataset", key)
        return dataset

    def scan(self, key):
        with self.lock:
            dataset = self.get(key)
            root = Path(dataset["path"])
            if not root.is_dir(): raise ServiceError("数据集目录已不存在")
            images, seen, captions = [], {}, 0
            snapshot = 0.0
            for directory, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
                for name in sorted(files):
                    p = Path(directory) / name
                    suffix = p.suffix.lower()
                    # Snapshot images and captions, so a later submit can tell whether
                    # anything changed after this scan.
                    if suffix == ".txt" or suffix in EXTENSIONS:
                        try: snapshot = max(snapshot, p.stat().st_mtime)
                        except OSError: pass
                    if suffix not in EXTENSIONS or p.is_symlink(): continue
                    p = contained(p, root)
                    issues, width, height = [], 0, 0
                    try:
                        with Image.open(p) as im:
                            width, height = im.size
                            im.verify()
                        if min(width, height) < 64 or max(width, height) / max(1, min(width, height)) > 4:
                            issues.append("odd_size")
                    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
                        issues.append("corrupt")
                    caption_path = p.with_suffix(".txt")
                    caption = None
                    if caption_path.is_file():
                        try:
                            contained(caption_path, root)
                            if caption_path.stat().st_size <= 400000:
                                caption = read_caption(caption_path)
                        except (OSError, UnicodeError, ServiceError): pass
                    if caption is None or not caption.strip(): issues.append("missing_caption")
                    else: captions += 1
                    digest = hashlib.sha256()
                    with p.open("rb") as f:
                        while chunk := f.read(1024 * 1024): digest.update(chunk)
                    sha = digest.hexdigest()
                    if sha in seen:
                        issues.append("duplicate")
                        first = images[seen[sha]]
                        if "duplicate" not in first["issues"]:
                            first["issues"].append("duplicate")
                    else: seen[sha] = len(images)
                    image_id = identity(p.relative_to(root).as_posix())
                    images.append(dict(image_id=image_id, file_name=p.relative_to(root).as_posix(), path=str(p), width=width, height=height,
                        thumbnail_url=None if "corrupt" in issues else f"/api/v1/datasets/{key}/images/{image_id}/thumbnail",
                        caption=caption, issues=issues))
            self.store.put("dataset_images", key, images)
            # Summary counts come from the per-image issue lists, so an image that
            # is both flagged and later re-flagged is still counted exactly once.
            counts = Counter(issue for image in images for issue in image["issues"])
            self.store.patch("dataset", key, dict(image_count=len(images), caption_count=captions,
                issues=[dict(kind=k, count=v) for k,v in counts.items() if v], scanned_at=now(), source_mtime=snapshot))

    def changed_since_scan(self, key):
        """Whether images or captions were modified after the recorded scan.

        Cheap by design: only mtimes are compared, and the walk stops at the first
        file newer than the snapshot, so an unchanged dataset is the worst case.
        Deletions are not detected this way (nothing is left to compare).
        """
        dataset = self.get(key)
        recorded = dataset.get("source_mtime")
        if recorded is None:
            return False
        root = Path(dataset["path"])
        if not root.is_dir():
            return True
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
            for name in files:
                p = Path(directory) / name
                suffix = p.suffix.lower()
                if suffix != ".txt" and suffix not in EXTENSIONS:
                    continue
                try:
                    if p.stat().st_mtime > recorded:
                        return True
                except OSError:
                    continue
        return False

    def images(self, key, offset, limit, issue=None):
        self.get(key)
        items = self.store.get("dataset_images", key, [])
        if issue: items = [i for i in items if issue in i["issues"]]
        return dict(items=[{k:v for k,v in i.items() if k != "path"} for i in items[offset:offset+limit]], total=len(items), offset=offset, limit=limit)

    def image(self, key, image_id):
        dataset = self.get(key)
        images = self.store.get("dataset_images", key, [])
        record = next((i for i in images if i["image_id"] == image_id), None)
        if not record: raise ServiceError("图片不存在，请重新扫描", 404)
        p = contained(record["path"], dataset["path"])
        if not p.is_file(): raise ServiceError("图片已不存在", 404)
        return p, record

    def caption(self, key, image_id, text):
        with self.lock:
            p, _ = self.image(key, image_id)
            target = contained(p.with_suffix(".txt"), self.get(key)["path"])
            if p.with_suffix(".txt").is_symlink(): raise ServiceError("不允许写入符号链接 caption", 403)
            if target.is_file():
                backup = self.store.root / "backups" / "captions" / key / image_id
                import uuid
                atomic_text(backup / f"{uuid.uuid4().hex}.txt", read_caption(target))
            atomic_text(target, text)
            images = self.store.get("dataset_images", key, [])
            for i in images:
                if i["image_id"] == image_id:
                    i["caption"] = text
                    i["issues"] = [v for v in i["issues"] if v != "missing_caption"]
                    if not text.strip(): i["issues"].append("missing_caption")
            counts = Counter(v for i in images for v in i["issues"])
            self.store.put("dataset_images", key, images)
            self.store.patch("dataset", key, dict(caption_count=sum(bool(i["caption"] and i["caption"].strip()) for i in images),
                issues=[dict(kind=k, count=v) for k,v in counts.items() if v],
                source_mtime=max(self.get(key).get("source_mtime") or 0.0, target.stat().st_mtime)))

    def thumbnail(self, key, image_id, size=384):
        p, _ = self.image(key, image_id)
        cache = self.store.root / "cache" / "thumbnails" / f"{identity(key, image_id, p.stat().st_mtime_ns, p.stat().st_size, size)}.jpg"
        if not cache.is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(p) as image:
                    image.thumbnail((size, size))
                    im = ImageOps.exif_transpose(image).convert("RGB")
                    import tempfile
                    fd, temp = tempfile.mkstemp(dir=cache.parent, suffix=".jpg")
                    os.close(fd)
                    try:
                        im.save(temp, "JPEG", quality=80 if size <= 384 else 90)
                        os.replace(temp, cache)
                    finally: Path(temp).unlink(missing_ok=True)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise ServiceError(f"无法生成缩略图：{exc}", 422) from exc
        return cache
