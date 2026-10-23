"""Proactive engine freshness: source/environment drift marks a ready engine stale."""

import tempfile
import unittest
from pathlib import Path

from app.services.engines import EngineService
from app.storage.state import StateStore


class EngineFreshnessTests(unittest.TestCase):
    def test_changed_source_or_environment_is_marked_before_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "engine"
            source.mkdir()
            (source / "train.py").write_text("print('v1')\n", encoding="utf-8")
            python = root / "python.exe"
            python.write_bytes(b"fake")
            store = StateStore(root / "data")
            key = "a" * 24
            store.put("engine", key, {
                "installation_id": key,
                "engine_id": None,
                "label": "probe",
                "source_path": str(source),
                "python_executable": str(python),
                "python_candidates": [str(python)],
                "state": "ready",
                "verification": "experimental",
                "revision": {"fingerprint": "old-source"},
                "environment_fingerprint": "old-environment",
                "environment_manifest_id": "old-manifest",
                "environment_id": "old-environment",
                "issues": [],
            })
            service = EngineService(store, lambda: {"engine_root": str(source.parent)}, root)
            item = service.list()[0]
            self.assertEqual(item["state"], "discovered")
            self.assertEqual(item["verification"], "unverified")
            self.assertIsNone(item["environment_manifest_id"])
            self.assertIn("变化", " ".join(item["issues"]))
            # The stored record, not only the returned copy, is stale.
            self.assertEqual(store.get("engine", key)["state"], "discovered")


if __name__ == "__main__":
    unittest.main()
