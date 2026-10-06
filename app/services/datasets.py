import hashlib
import os
import threading
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from app.services.common import ServiceError, local_path, contained, identity, now, atomic_text

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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

    def scan(self, key):
        with self.lock:
            dataset = self.get(key)
            root = Path(dataset["path"])
            if not root.is_dir(): raise ServiceError("数据集目录已不存在")
            images, seen, counts, captions = [], {}, Counter(), 0
            for directory, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
                for name in sorted(files):
                    p = Path(directory) / name
                    if p.suffix.lower() not in EXTENSIONS or p.is_symlink(): continue
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
                                caption = caption_path.read_text(encoding="utf-8-sig")
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
                            first["issues"].append("duplicate"); counts["duplicate"] += 1
                    else: seen[sha] = len(images)
                    image_id = identity(p.relative_to(root).as_posix())
                    images.append(dict(image_id=image_id, file_name=p.relative_to(root).as_posix(), path=str(p), width=width, height=height,
                        thumbnail_url=None if "corrupt" in issues else f"/api/v1/datasets/{key}/images/{image_id}/thumbnail",
                        caption=caption, issues=issues))
                    counts.update(issues)
            self.store.put("dataset_images", key, images)
            self.store.patch("dataset", key, dict(image_count=len(images), caption_count=captions,
                issues=[dict(kind=k, count=v) for k,v in counts.items() if v], scanned_at=now()))

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
                atomic_text(backup / f"{uuid.uuid4().hex}.txt", target.read_text(encoding="utf-8-sig"))
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
                issues=[dict(kind=k, count=v) for k,v in counts.items() if v]))

    def thumbnail(self, key, image_id):
        p, _ = self.image(key, image_id)
        cache = self.store.root / "cache" / "thumbnails" / f"{identity(key, image_id, p.stat().st_mtime_ns, p.stat().st_size)}.jpg"
        if not cache.is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(p) as image:
                    image.thumbnail((384, 384))
                    im = ImageOps.exif_transpose(image).convert("RGB")
                    import tempfile
                    fd, temp = tempfile.mkstemp(dir=cache.parent, suffix=".jpg")
                    os.close(fd)
                    try:
                        im.save(temp, "JPEG", quality=80)
                        os.replace(temp, cache)
                    finally: Path(temp).unlink(missing_ok=True)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise ServiceError(f"无法生成缩略图：{exc}", 422) from exc
        return cache
