"""Video previews are indexed and served like samples, not discarded."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.storage.state import StateStore
from supervisor.worker import register_artifacts


class VideoArtifactTests(unittest.TestCase):
    def test_video_preview_is_registered_as_a_complete_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = StateStore(root / "data")
            work = root / "output"
            work.mkdir()
            job = store.root / "jobs" / ("a" * 24)
            job.mkdir(parents=True)
            video = work / "sample" / "0001-00000200-0.mp4"
            video.parent.mkdir()
            video.write_bytes(b"not-a-real-video-but-complete")
            (job / "job.json").write_text(json.dumps({
                "name": "probe", "output_dir": str(work), "output_name": "probe",
                "architecture": "wan", "dataset_path": str(root), "params": {},
            }), encoding="utf-8")
            task = {
                "task_id": "a" * 24, "name": "probe", "engine_id": "musubi_tuner",
                "architecture": "wan", "output_dir": str(work),
                "draft": {"base_model_path": ""},
            }
            register_artifacts(store, task)
            rows = [a for a in store.list("artifact") if a["file_name"] == video.name]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "video")
            self.assertTrue(rows[0]["complete"])
            self.assertTrue(rows[0]["preview_url"])


if __name__ == "__main__":
    unittest.main()
