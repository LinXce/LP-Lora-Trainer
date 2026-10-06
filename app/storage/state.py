"""Small SQLite index. Model weights, images and full logs stay on disk."""
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager


class StateStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(kind,id))")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, kind, key, default=None):
        with self.connection() as db:
            row = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else default

    def list(self, kind):
        with self.connection() as db:
            rows = db.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put(self, kind, key, value):
        with self.connection() as db:
            db.execute("INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body", (kind, key, json.dumps(value, ensure_ascii=False)))
        return value

    def patch(self, kind, key, values, expected=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
            if not row:
                raise KeyError(key)
            body = json.loads(row[0])
            if expected and any(body.get(k) != v for k, v in expected.items()): return None
            body.update(values)
            db.execute("UPDATE records SET body=? WHERE kind=? AND id=?", (json.dumps(body, ensure_ascii=False), kind, key))
        return body

    def delete(self, kind, key):
        with self.connection() as db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, key))
