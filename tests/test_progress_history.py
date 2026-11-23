"""Read-only legacy progress projection, with real adapter parsers and private logs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import progress_history
from app.services.training import TrainingService
from app.storage.state import StateStore


class LegacyProgressTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = StateStore(Path(tmp.name))
        self.training = TrainingService(self.store, None, None)
        self.job = self.store.root / "jobs" / "old"
        self.job.mkdir(parents=True)
        self.log = self.job / "stdout.log"

    def task(self, engine, state="succeeded", progress=None):
        row = dict(task_id="old", engine_id=engine, state=state,
                   progress=progress or dict(step=759, total_steps=759, it_per_sec=4783.6))
        self.store.put("task", "old", row)
        return row

    def test_summary_detail_and_history_agree_without_writing_back(self):
        raw = self.task("anima_lora")
        output = self.store.root / "weights"
        output.touch()
        self.store.put("artifact", "out", dict(task_id="old", path=str(output)))
        self.store.put("task_output", "old", raw)
        self.log.write_text(
            "[supervisor] 包含 1 个前置缓存阶段。\n"
            "progress 9000/9000 epoch 99/99 loss 0.1 eta 0:00:00\n"
            "[supervisor] 开始第 2/2 阶段。\n"
            "1080 steps (40 epochs x 27), warmup 54\n"
            "progress 1080/1080 epoch 40/40 loss 0.3134 eta 0:00:00\n"
            "step 1080/1080 loss 0.2514 |g| 0.086 lr 5.00e-05 peak 11.88 GB 1.44s/step\n",
            encoding="utf-8")
        before = self.log.read_bytes()
        with patch.object(progress_history._Replay, "update", autospec=True,
                          side_effect=progress_history._Replay.update) as update:
            current = self.training.tasks()[0]["progress"]
            self.assertEqual(current, self.training.task("old")["progress"])
            self.assertEqual(current, self.training.result_history()[0]["progress"])
            self.assertEqual(update.call_count, 1)
        self.assertEqual([current[k] for k in ("step", "total_steps", "epoch", "total_epochs")], [1080, 1080, 40, 40])
        self.assertAlmostEqual(current["it_per_sec"], 1/1.44)
        self.assertEqual(self.store.get("task", "old"), raw)
        self.assertEqual(self.log.read_bytes(), before)

    def test_real_resumed_bar_overrides_false_metric_axis_without_filling_gaps(self):
        self.task("kohaku_moss_trainer", progress=dict(step=1, total_steps=2000))
        self.log.write_text(
            "train: 76%|###| 1511/2000 [00:22<17:36, 2.16s/step, loss=0.3613, lr=1.55e-05]\n"
            "[02:15:29] [training.callbacks.kernels] [INFO] step 1 7.031 s/step, steps 2-10 median 2.162 s/step\n"
            "train: 100%|###| 2000/2000 [17:40<00:00, 2.12s/step, loss=0.319, lr=0.00e+00]\n",
            encoding="utf-8")
        metrics = self.job / "metrics.jsonl"
        metrics.write_text('{"step":1,"loss":0.3}\n{"step":3,"loss":0.32}\n{"step":1,"loss":0.319}\n')
        before = metrics.read_bytes()
        p = self.training.tasks()[0]["progress"]
        self.assertEqual((p["step"], p["total_steps"], p["epoch"]), (2000, 2000, None))
        self.assertAlmostEqual(p["it_per_sec"], 1/2.12)
        self.assertEqual([p["step"] for p in self.training.metrics("old")], [1511, 2000])
        self.assertEqual(metrics.read_bytes(), before)

    def test_only_loading_or_missing_log_clears_false_speed_steps_and_epoch(self):
        self.task("anima_lora", "stopped")
        (self.job / "command.json").write_text(json.dumps({"commands": [["cache"], ["train"]]}))
        self.log.write_text("Loading weights: 100%|###| 759/759 [00:00<00:00, 4783.60it/s]\n")
        self.assertTrue(all(v is None for v in self.training.tasks()[0]["progress"].values()))
        self.log.unlink()
        self.assertTrue(all(v is None for v in self.training.task("old")["progress"].values()))
        self.task("anima_lora", "queued")
        self.assertIsNone(self.training.task("old")["progress"]["it_per_sec"])

    def test_no_log_retains_independent_loss_and_lr_fields(self):
        self.task("anima_lora", "succeeded", dict(step=759, epoch=9, loss=0.21, lr=0.001))
        p = self.training.task("old")["progress"]
        self.assertEqual((p["loss"], p["lr"], p["step"], p["epoch"]), (0.21, 0.001, None, None))

    def test_user_initiated_deletion_keeps_verified_tombstone_after_log_removed(self):
        key = "b"*32
        row = self.task("anima_lora")
        self.store.put("task", key, dict(row, task_id=key))
        self.log.write_text("progress 24/24 epoch 3/3 loss 0.31 eta 0:00:00\n")
        self.job.rename(self.job.parent / key)
        artifact = self.store.root / "weight"
        artifact.touch()
        self.store.put("artifact", "a", dict(task_id=key, path=str(artifact)))
        self.training.delete(key)
        p = self.training.result_history()[0]["progress"]
        self.assertEqual((p["step"], p["epoch"], p["log_version"]), (24, 3, 2))

    def test_log_supplies_only_observed_chart_points_if_jsonl_missing(self):
        self.task("anima_lora")
        self.log.write_text("progress 8/20 epoch 1/3 loss 0.21 eta 0:00:19\n")
        self.assertEqual(self.training.metrics("old"), [{"step": 8, "loss": 0.21}])

    def test_active_log_appends_truncate_replacement_and_partial_line(self):
        self.task("anima_lora", "running")
        self.log.write_bytes(b"progress 2/10 epoch 1/2 loss 0.2 eta 0:00:08\nprogress 4/10")
        self.assertEqual(self.training.tasks()[0]["progress"]["step"], 2)
        with self.log.open("ab") as f:
            f.write(b" epoch 1/2 loss 0.3 eta 0:00:06\n")
        self.assertEqual(self.training.tasks()[0]["progress"]["step"], 4)
        replacement = self.job / "next"
        replacement.write_bytes(b"progress 1/20 epoch 1/3 loss 0.2 eta 0:00:19\n")
        replacement.replace(self.log)
        self.assertEqual(self.training.tasks()[0]["progress"]["step"], 1)
        self.log.write_bytes(b"loading\n")
        self.assertIsNone(self.training.tasks()[0]["progress"]["step"])

    def test_large_log_keeps_unique_stage_and_training_observation_in_middle(self):
        self.task("anima_lora")
        padding = (b"x"*8000 + b"\n") * 138
        self.log.write_bytes("[supervisor] 包含 1 个前置缓存阶段。\n".encode() + padding +
                             "[supervisor] 开始第 2/2 阶段。\n".encode() +
                             b"progress 1080/1080 epoch 40/40 loss 0.31 eta 0:00:00\n" + padding)
        with patch.object(progress_history._Replay, "chunk", autospec=True,
                          side_effect=progress_history._Replay.chunk) as chunk:
            self.assertEqual(self.training.tasks()[0]["progress"]["step"], 1080)
            count = chunk.call_count
            self.training.tasks()
            self.assertEqual(chunk.call_count, count)
            self.assertLessEqual(max(len(call.args[1]) for call in chunk.call_args_list), progress_history.READ_BYTES)

    def test_new_worker_snapshot_does_not_open_old_log(self):
        snapshot = dict(log_version=2, step=42, total_steps=80, epoch=2, total_epochs=4)
        self.task("anima_lora", "running", snapshot)
        with patch.object(progress_history._Replay, "update", side_effect=AssertionError):
            self.assertEqual(self.training.tasks()[0]["progress"], snapshot)

    def test_valid_dense_axis_and_old_lr_samples_are_kept(self):
        self.task("anima_lora")
        self.log.write_text("step 6/10 loss 0.3960 lr 3.78e-06 peak 12.2 GB 3.78s/step\n")
        data = [{"step": n, "loss": 0.4} for n in range(1, 9)]
        (self.job / "metrics.jsonl").write_text("".join(json.dumps(p)+"\n" for p in data))
        points = self.training.metrics("old")
        self.assertEqual([p["step"] for p in points], list(range(1, 9)))
        self.assertEqual(points[5]["lr"], 3.78e-06)
        self.assertNotIn("lr", points[4])
        with self.log.open("a") as stream:
            stream.write("step 10/10 loss 0.3 lr 2e-06 peak 12.2 GB 3s/step\n")
        self.assertEqual([p["step"] for p in self.training.metrics("old")], [*range(1, 9), 10])
