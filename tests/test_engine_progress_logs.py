"""Regression fixtures from the installed engines' training and sampling logs."""

import unittest

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter
from adapters.sd_scripts.adapter import KohyaAdapter


class EngineProgressLogs(unittest.TestCase):
    def test_moss_step_epoch_and_rate_have_separate_sources(self):
        adapter = AnimaLoraAdapter()
        lines = (
            "1080 steps (40 epochs x 27), warmup 54",
            "  progress 1080/1080 epoch 40/40 loss 0.3134 eta 0:00:00",
            "  step 1080/1080 loss 0.2514 |g| 0.086 sigma 0.774 lr 5.00e-05 peak 11.88 GB 1.44s/step",
            "epoch 40/40: loss 0.3134 38s peak 11.88 GB",
        )
        result = {}
        for line in lines:
            result.update(adapter.parse_log(line)[0].payload)
        self.assertEqual(result, {
            "step": 1080, "total_steps": 1080, "epoch": 40, "total_epochs": 40,
            "loss": 0.2514, "lr": 5e-5, "it_per_sec": 1 / 1.44, "eta_seconds": 0,
        })

    def test_load_cache_and_sample_bars_do_not_count_as_training(self):
        for adapter in (KohyaAdapter(), MusubiTunerAdapter(), AnimaLoraAdapter(), KohakuMossTrainerAdapter()):
            for label in ("Loading weights: ", "cache latents: ", "sampling: ", "validation: ", ""):
                line = f"{label}100%|########| 759/759 [00:01<00:00, 4783.60it/s]"
                with self.subTest(adapter=adapter.engine_id, label=label):
                    self.assertEqual(adapter.parse_log(line), ())

    def test_native_step_bars_and_epoch_announcements(self):
        for adapter, kind in ((KohyaAdapter(), "remaining_steps"), (AnimaLoraAdapter(), "remaining_steps"),
                              (MusubiTunerAdapter(), "progress")):
            with self.subTest(adapter=adapter.engine_id):
                for line, expected in (
                    ("epoch 3/8", {"epoch": 3, "total_epochs": 8}),
                    ("  num epochs / epoch数: 8", {"total_epochs": 8}),
                    ("  total optimization steps / 学習ステップ数: 80", {"total_steps": 80}),
                ):
                    self.assertEqual(adapter.parse_log(line)[0].payload, expected)
                event, = adapter.parse_log("steps:  42%|####      | 34/80 [05:00<06:08, 8.00s/it, avr_loss=0.4]")
                self.assertEqual(event.kind, kind)
                self.assertEqual(event.payload, {
                    "step": 34, "total_steps": 80, "eta_seconds": 368, "it_per_sec": 0.125, "loss": 0.4,
                })
                event, = adapter.parse_log("steps:  42%|####      | 34/80 [00:04<00:05, 8.00it/s]")
                self.assertEqual(event.payload["it_per_sec"], 8)

    def test_anima_explicit_resume_and_structured_events(self):
        adapter = AnimaLoraAdapter()
        self.assertEqual(adapter.parse_log("INFO resuming at epoch 5, step 160")[0].payload,
                         {"epoch": 5, "step": 160, "resume_step": 160})
        self.assertEqual(adapter.parse_log('{"ev":"run_start","total_epochs":8,"total_steps":640}')[0].payload,
                         {"total_epochs": 8, "total_steps": 640})
        self.assertEqual(adapter.parse_log('{"ev":"step","epoch":3,"global_step":24,"loss/current":0.31}')[0].payload,
                         {"step": 24, "epoch": 3, "loss": 0.31})
        self.assertEqual(adapter.parse_log('{"ev":"val","epoch":3,"global_step":24,"loss":10}'), ())

    def test_kohaku_timing_summary_cannot_rewind_resumed_global_step(self):
        adapter = KohakuMossTrainerAdapter()
        span, = adapter.parse_log("[00:00:01] [training.entry] [INFO] training from step 1500 to 2000")
        self.assertEqual(span.payload, {"step": 1500, "total_steps": 2000})
        last, = adapter.parse_log("train: 100%|########| 2000/2000 [01:09:41<00:00, 2.12s/step, s/it=2.16]")
        self.assertEqual(last.payload, {"step": 2000, "total_steps": 2000, "eta_seconds": 0, "it_per_sec": 1 / 2.12})
        retrospective, = adapter.parse_log(
            "[23:15:41] [training.callbacks.kernels] [INFO] step 1 7.031 s/step, steps 2-10 median 2.162 s/step")
        self.assertEqual(retrospective.payload, {"it_per_sec": 1 / 2.162})
        self.assertEqual(adapter.parse_log("[23:15:41] [inference.steplog] [INFO] step 50/50: 2.134 s"), ())

    def test_kohaku_fast_unit_and_fallback_postfix(self):
        adapter = KohakuMossTrainerAdapter()
        for rate, expected in (("4.00step/s", 4), ("?step/s, s/it=2.50", 0.4), ("?step/s, s/it=0.00", None)):
            with self.subTest(rate=rate):
                event, = adapter.parse_log(f"train:  50%|#####     | 4/8 [00:10<?, {rate}]")
                self.assertEqual(event.payload.get("it_per_sec"), expected)
                self.assertNotIn("eta_seconds", event.payload)


if __name__ == "__main__":
    unittest.main()
