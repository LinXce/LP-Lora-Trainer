"""Exercise the live training snapshot with a real CPU process and an isolated DB."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from adapters.registry import ADAPTERS
from app.services.progress import merge_progress
from app.services.training import PROGRESS
from app.storage.state import StateStore
from supervisor.worker import Worker


class WorkerTrainingProgressTests(unittest.TestCase):
    def test_live_snapshot_ignores_preparation_and_flushes_burst_and_final_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root)
            task = dict(task_id="cpu", engine_id="anima_lora", state="preparing",
                        stop_requested=None, progress=PROGRESS.copy())
            store.put("task", "cpu", task)
            job = root / "jobs" / "cpu"
            job.mkdir(parents=True)
            worker = Worker(store)
            adapter = ADAPTERS["anima_lora"]
            def stage(script, training=True):
                return worker._run_stage(task, adapter, [sys.executable, "-u", "-c", script],
                                         str(root), os.environ.copy(), training_stage=training)
            with patch.object(worker, "poll_artifacts"):
                self.assertEqual(stage("print('steps: 100%|##| 759/759 [00:00, 4000it/s]')", False), 0)
                self.assertIsNone(store.get("task", "cpu")["progress"]["step"])
                script = """import os, time
print('progress 48/1080 epoch 2/40 loss 0.3 eta 0:10:00')
print('step 48/1080 loss 0.25 lr 5e-5 peak 11.88 GB 1.44s/step')
deadline = time.monotonic() + 10
while not os.path.isfile('release') and time.monotonic() < deadline: time.sleep(.03)
print('progress 1080/1080 epoch 40/40 loss 0.31 eta 0:00:00', end='')
"""
                codes = []
                thread = threading.Thread(target=lambda: codes.append(stage(script)))
                thread.start()
                try:
                    end = time.monotonic() + 8
                    while time.monotonic() < end:
                        progress = store.get("task", "cpu")["progress"]
                        if progress.get("it_per_sec"):
                            break
                        time.sleep(.05)
                    self.assertEqual((progress["step"], progress["total_steps"]), (48, 1080))
                    self.assertEqual((progress["epoch"], progress["total_epochs"]), (2, 40))
                    self.assertAlmostEqual(progress["it_per_sec"], 1 / 1.44)
                finally:
                    (root / "release").touch()
                    thread.join(15)
            self.assertEqual(codes, [0])
            progress = store.get("task", "cpu")["progress"]
            self.assertEqual((progress["step"], progress["epoch"], progress["eta_seconds"]), (1080, 40, 0))
            self.assertIn("759/759", (job / "stdout.log").read_text(encoding="utf-8"))

    def test_relative_resume_bar_uses_logged_global_units(self):
        adapter = ADAPTERS["anima_lora"]
        first_epoch = {}
        for line in ("skipping 4 steps: 0 full epochs + 8 batches of epoch 1",
                     "steps:  50%|#####| 3/6 [00:04<00:04, 1.4s/it]"):
            for event in adapter.parse_log(line):
                merge_progress(first_epoch, event.payload, event.kind)
        self.assertEqual((first_epoch["step"], first_epoch["total_steps"], first_epoch["epoch"]), (7, 10, 1))
        for initial in ({"total_steps": 1000}, {"resume_step": 900}, {"resume_step": 900, "total_steps": 1000}):
            progress = dict(initial)
            for step in (0, 1, 100):
                self.assertTrue(merge_progress(progress, {"step": step, "total_steps": 100}, "remaining_steps"))
                self.assertEqual((progress["step"], progress["total_steps"]), (900 + step, 1000))
            self.assertFalse(merge_progress(progress, {"step": 10, "it_per_sec": 3.0}))
            self.assertEqual(progress["step"], 1000)
