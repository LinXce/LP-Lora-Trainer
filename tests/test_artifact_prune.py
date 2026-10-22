"""Pruning artifact records whose files were deleted never touches the disk."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.services.artifacts import prune_missing_artifacts
from app.services.training import TrainingService
from app.storage.state import StateStore


class ArtifactPruneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = StateStore(self.root / "data")
        self.output = self.root / "output"
        self.output.mkdir()

    def put(self, artifact_id, path=None, **extra):
        record = dict(artifact_id=artifact_id, task_id="task-1", kind="checkpoint", **extra)
        if path is not None:
            record["path"] = path
        self.store.put("artifact", artifact_id, record)

    def artifact_ids(self):
        return {record["artifact_id"] for record in self.store.list("artifact")
                if isinstance(record, dict) and record.get("artifact_id")}

    def test_prunes_deleted_empty_and_malformed_paths_but_keeps_existing_file(self):
        kept_file = self.output / "lora.safetensors"
        kept_file.write_bytes(b"weights")
        self.put("kept", str(kept_file))
        self.put("deleted", str(self.output / "deleted.safetensors"))
        self.put("blank", "")
        self.put("missing-path")  # no path key at all
        directory = self.output / "a-directory"
        directory.mkdir()
        self.put("directory", str(directory))

        self.assertEqual(prune_missing_artifacts(self.store), 4)
        self.assertEqual(self.artifact_ids(), {"kept"})
        # Only the record is removed: existing files, referenced or not, stay put.
        self.assertTrue(kept_file.is_file())
        (self.output / "unreferenced.txt").write_text("keep me", encoding="utf-8")
        self.assertEqual(prune_missing_artifacts(self.store), 0)
        self.assertEqual(self.artifact_ids(), {"kept"})
        self.assertTrue((self.output / "unreferenced.txt").is_file())
        self.assertTrue(kept_file.is_file())

    def test_prune_does_not_touch_other_record_kinds_or_unaddressable_records(self):
        kept_file = self.output / "lora.safetensors"
        kept_file.write_bytes(b"weights")
        self.put("kept", str(kept_file))
        self.put("deleted", str(self.output / "gone.safetensors"))
        self.store.put("task", "task-1", dict(task_id="task-1", output_dir=str(self.output)))
        self.store.put("task_output", "task-1", dict(task_id="task-1", output_dir=str(self.output)))
        self.store.put("dataset", "dataset-1", dict(dataset_id="dataset-1", path=str(self.output / "gone.safetensors")))
        # Bodies are JSON, so tolerating odd values keeps one bad row from blocking the sweep.
        self.store.put("artifact", "garbage", "not-a-dict")
        self.store.put("artifact", "no-id", dict(path=str(self.output / "gone.safetensors")))

        self.assertEqual(prune_missing_artifacts(self.store), 1)
        self.assertIsNotNone(self.store.get("task", "task-1"))
        self.assertIsNotNone(self.store.get("task_output", "task-1"))
        self.assertIsNotNone(self.store.get("dataset", "dataset-1"))
        remaining = self.store.list("artifact")
        self.assertIn("not-a-dict", remaining)
        self.assertIn({"path": str(self.output / "gone.safetensors")}, remaining)
        self.assertEqual(self.artifact_ids(), {"kept"})

    def test_training_service_lists_after_prune_without_missing_records(self):
        kept_file = self.output / "lora.safetensors"
        kept_file.write_bytes(b"weights")
        self.store.put("task", "task-1", dict(task_id="task-1", output_dir=str(self.output)))
        self.put("kept", str(kept_file))
        self.put("deleted", str(self.output / "deleted.safetensors"))

        self.assertEqual(prune_missing_artifacts(self.store), 1)
        service = TrainingService(self.store, Mock(), Mock())
        self.assertEqual([artifact["artifact_id"] for artifact in service.artifacts("task-1")], ["kept"])

    def test_prune_opens_the_given_store_only(self):
        # An isolated temp store: no real data/state.sqlite is ever opened.
        self.assertEqual(self.store.path.parent, (self.root / "data").resolve())
        self.assertTrue(self.store.path.is_file())
        # Windows: the temp directory is only removable once the handle is gone.
        db = sqlite3.connect(self.store.path)
        try:
            count = db.execute("SELECT COUNT(*) FROM records WHERE kind='artifact'").fetchone()[0]
        finally:
            db.close()
        self.assertEqual(count, 0)

    def test_prune_survives_paths_that_raise_on_is_file(self):
        self.put("null-byte", "bad\x00path.safetensors")
        self.put("not-a-string", 42)
        self.assertEqual(prune_missing_artifacts(self.store), 2)
        self.assertEqual(self.store.list("artifact"), [])


if __name__ == "__main__":
    unittest.main()
