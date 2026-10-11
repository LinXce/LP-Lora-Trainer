import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path


class ServiceError(Exception):
    def __init__(self, message, status=400):
        self.message, self.status = message, status
        super().__init__(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(*values):
    return hashlib.sha256("\0".join(str(v) for v in values).encode()).hexdigest()[:24]


def local_path(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ServiceError("路径不能为空")
    p = Path(value).expanduser()
    if not p.is_absolute():
        raise ServiceError("请提供绝对路径")
    return p.resolve()


def contained(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    if not p.is_relative_to(r):
        raise ServiceError("路径超出已登记目录", 403)
    return p


def replace_file(source, target, attempts=20, delay=0.02):
    """``os.replace`` that tolerates Windows sharing violations.

    Windows refuses to replace a file that another handle (an API reader polling
    supervisor.json, antivirus, an indexer) has open at that instant, raising
    PermissionError. That lock is released within milliseconds, so retry briefly
    before giving up; any other error is raised immediately.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(delay * min(attempt + 1, 5))


def atomic_text(path, text):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".lp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        replace_file(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
