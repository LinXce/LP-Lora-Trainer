"""The job-directory contract shared by the supervisor and every adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from app.services.jobs import JobFiles


class JobFilesTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.files = JobFiles(self.directory)

    def test_paths_are_fixed_names(self):
        self.assertEqual("job.json", self.files.job.name)
        self.assertEqual("stages.json", self.files.stages.name)
        self.assertEqual("config.native.toml", self.files.native_config("toml").name)

    def test_job_round_trip(self):
        config = {"architecture": "moss_image", "output_dir": "D:/out", "params": {"moss_epochs": 8}}
        self.files.job.write_text(json.dumps(config), encoding="utf-8")
        self.assertEqual(config, self.files.read_job())

    def test_stages_round_trip_keeps_stage_boundaries(self):
        stages = (("python", "-u", "train.py"), ("python", "-u", "cache.py", "--src", "a b"))
        self.files.write_stages(stages)
        self.assertEqual(stages, self.files.read_stages())

    def test_stages_are_written_as_lists_of_lists(self):
        self.files.write_stages((("a", "b"), ("c",)))
        self.assertEqual([["a", "b"], ["c"]], json.loads(self.files.stages.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
