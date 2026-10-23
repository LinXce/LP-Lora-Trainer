"""Project version ↔ engine git baseline mapping."""

import tempfile
import unittest
from pathlib import Path

from adapters.registry import ADAPTERS
from app import APP_VERSION
from app.services.compat import engine_compat, load_manifest, version_entry
from app.services.engines import EngineService
from app.storage.state import StateStore


class CompatManifestTests(unittest.TestCase):
    def test_current_version_has_one_entry_per_adapter(self):
        entry = version_entry(APP_VERSION)
        self.assertIsNotNone(entry)
        engines = {item["engine_id"] for item in entry["engines"]}
        self.assertEqual(engines, set(ADAPTERS))
        for item in entry["engines"]:
            self.assertTrue(item["ref"])
            self.assertTrue(item["commit"])
            self.assertEqual(item["adapter_version"], ADAPTERS[item["engine_id"]].adapter_version)

    def test_supported_mismatch_and_unlisted(self):
        entry = version_entry(APP_VERSION)
        expected = entry["engines"][0]
        compatible = {
            "engine_id": expected["engine_id"],
            "revision": {"requested_ref": expected["ref"], "commit": expected["commit"]},
        }
        self.assertEqual(engine_compat(compatible, expected["adapter_version"])["status"], "supported")
        wrong = {
            "engine_id": expected["engine_id"],
            "revision": {"requested_ref": expected["ref"], "commit": "0" * 40},
        }
        self.assertEqual(engine_compat(wrong, expected["adapter_version"])["status"], "mismatch")
        self.assertEqual(engine_compat({"engine_id": "not-an-engine", "revision": {}}, None)["status"], "unlisted")

    def test_engine_list_exposes_compat_and_unmapped_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "engine"
            source.mkdir()
            (source / "train.py").write_text("", encoding="utf-8")
            python = root / "python.exe"
            python.write_bytes(b"fake")
            store = StateStore(root / "data")
            key = "b" * 24
            store.put("engine", key, {
                "installation_id": key,
                "engine_id": "anima_lora",
                "label": "probe",
                "source_path": str(source),
                "python_executable": str(python),
                "python_candidates": [str(python)],
                "state": "discovered",
                "verification": "unverified",
                "revision": {"requested_ref": "main", "commit": "e6ed99f2136569a6a01c09005ba058fea60cf1cd"},
                "issues": [],
            })
            item = EngineService(store, lambda: {"engine_root": str(source.parent)}, root).list()[0]
            self.assertEqual(item["compat"]["status"], "supported")
            self.assertEqual(engine_compat(item, "0.1.0", version="9.9.9")["status"], "unmapped")


if __name__ == "__main__":
    unittest.main()
