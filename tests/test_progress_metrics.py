"""Learning-rate parsing and metric-point assembly for the charts.

The engines print different log shapes; only explicit `lr` values may become
points, and a point without a step must never be written.
"""

import math
import tempfile
import unittest
from pathlib import Path

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.capabilities import parse_eta, parse_progress
from adapters.kohaku_moss_trainer.progress import parse_progress as parse_kohaku
from app.services.training import TrainingService
from supervisor.worker import configured_total_steps, estimate_eta, is_tqdm_progress, metric_point


class LearningRateParserTests(unittest.TestCase):
    def test_tqdm_line_carries_loss_and_lr(self):
        events = parse_progress("steps: 100%|xxx| 12/100 [00:01<00:05, 2.00it/s, avr_loss=0.25, lr=2e-4]")
        self.assertEqual(len(events), 1)
        payload = events[0].payload
        self.assertEqual(payload["step"], 12)
        self.assertEqual(payload["loss"], 0.25)
        self.assertEqual(payload["lr"], 2e-4)

    def test_kohaku_callback_line_carries_lr(self):
        events = parse_kohaku(
            "[23:15:21] [training.callbacks.monitor] [INFO] step 10 [loss=0.3703, grad_norm=0.06446, lr=1e-05]"
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["step"], 10)
        self.assertEqual(events[0].payload["loss"], 0.3703)
        self.assertEqual(events[0].payload["lr"], 1e-5)

    def test_anima_plain_and_jsonl_lines_carry_lr(self):
        adapter = AnimaLoraAdapter()
        plain = adapter.parse_log(
            "step 6/1770 loss 0.3960 |g| 0.052 sigma 0.837 lr 3.78e-06 peak 12.2 GB 3.78s/step"
        )
        self.assertEqual(plain[0].payload["step"], 6)
        self.assertEqual(plain[0].payload["lr"], 3.78e-06)
        jsonl = adapter.parse_log('{"step": 7, "loss": 0.3, "lr": 1e-5}')
        self.assertEqual(jsonl[0].payload["lr"], 1e-5)


class EtaTests(unittest.TestCase):
    def test_anima_progress_eta_is_captured(self):
        self.assertEqual(parse_eta("eta 0:24:32"), 24 * 60 + 32)
        events = AnimaLoraAdapter().parse_log(
            "progress 1240/2040 epoch 15/24 loss 0.2961 eta 0:24:32")
        self.assertEqual(events[0].payload["step"], 1240)
        self.assertEqual(events[0].payload["total_steps"], 2040)
        self.assertEqual(events[0].payload["eta_seconds"], 24 * 60 + 32)

    def test_kohaku_step_without_eta_gets_a_fallback_estimate(self):
        events = parse_kohaku(
            "[20:05:03] [training.callbacks.monitor] [INFO] step 740 "
            "[loss=0.3411, grad_norm=2.067, lr=5e-05]"
        )
        self.assertNotIn("eta_seconds", events[0].payload)
        # 100 remaining steps at 2 steps/second.
        self.assertEqual(estimate_eta({"step": 10, "total_steps": 110, "it_per_sec": 2}, None, None, 0), 50)
        # No it/s line yet: use the observed steps since the first announced step.
        self.assertEqual(estimate_eta({"step": 20, "total_steps": 100}, 10, 0, 10), 80)
        # Nothing to estimate before the total or a second step is known.
        self.assertIsNone(estimate_eta({"step": 10, "total_steps": None}, None, None, 0))
        self.assertIsNone(estimate_eta({"step": 10, "total_steps": 100}, 10, 0, 0))
        self.assertEqual(configured_total_steps({"draft": {"params": {"STEPS": 850}}}), 850)
        self.assertEqual(configured_total_steps({"draft": {"params": {"max_train_steps": 1600}}}), 1600)
        self.assertIsNone(configured_total_steps({"draft": {"params": {}}}))


class MetricPointTests(unittest.TestCase):
    def test_point_includes_only_known_metrics(self):
        self.assertEqual(metric_point({"step": 4, "loss": 0.2, "lr": 1e-5}),
                         {"step": 4, "loss": 0.2, "lr": 1e-5})
        self.assertEqual(metric_point({"step": 4, "loss": 0.2, "lr": None}), {"step": 4, "loss": 0.2})
        self.assertEqual(metric_point({"step": 4, "loss": None, "lr": 1e-5}), {"step": 4, "lr": 1e-5})

    def test_point_requires_a_step_and_finite_values(self):
        self.assertIsNone(metric_point({"step": None, "loss": 0.2}))
        self.assertIsNone(metric_point({"step": 3, "loss": None, "lr": None}))
        self.assertEqual(metric_point({"step": 3, "loss": math.inf, "lr": 1e-5}), {"step": 3, "lr": 1e-5})

    def test_only_carriage_return_progress_frames_are_throttled(self):
        self.assertTrue(is_tqdm_progress(
            "train:  17%|###       | 257/1500 [09:29<45:53, 2.22s/step, loss=0.3474]"))
        self.assertFalse(is_tqdm_progress(
            "[23:15:41] [training.callbacks.kernels] [INFO] step 1 18.371 s/step"))
        self.assertFalse(is_tqdm_progress(
            "[23:15:21] [training.callbacks.monitor] [INFO] step 1 [loss=0.2752, lr=1e-06]"))


class LearningRateBackfillTests(unittest.TestCase):
    def test_old_metrics_get_lr_from_the_raw_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "stdout.log").write_text(
                "step 6/1770 loss 0.3960 |g| 0.052 sigma 0.837 lr 3.78e-06 peak 12.2 GB 3.78s/step\n",
                encoding="utf-8",
            )
            points = [{"step": 6, "loss": 0.396}]
            TrainingService._fill_learning_rate(
                None, {"engine_id": "anima_lora"}, job, points
            )
            self.assertEqual(points, [{"step": 6, "loss": 0.396, "lr": 3.78e-06}])

    def test_missing_adapter_or_log_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            points = [{"step": 1, "loss": 0.2}]
            TrainingService._fill_learning_rate(None, {}, Path(tmp), points)
            self.assertEqual(points, [{"step": 1, "loss": 0.2}])


if __name__ == "__main__":
    unittest.main()
