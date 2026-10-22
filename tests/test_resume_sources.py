"""Adapter-level resume scanning: only complete save points are offered."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter
from adapters.recovery import accelerate_resume_sources
from adapters.sd_scripts.adapter import KohyaAdapter


def write_job(job: Path, output: Path, architecture: str | None = None) -> Path:
    job.mkdir(parents=True, exist_ok=True)
    payload = {"output_dir": str(output)}
    if architecture is not None:
        payload["architecture"] = architecture
    (job / "job.json").write_text(json.dumps(payload), encoding="utf-8")
    return job


def write_state(directory: Path) -> None:
    """A complete accelerate.save_state directory (model + RNG + optimizer)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(b"")
    (directory / "random_states_0.pkl").write_bytes(b"rng")
    (directory / "optimizer.bin").write_bytes(b"opt")


class AccelerateResumeSourceTests(unittest.TestCase):
    def test_complete_state_dirs_are_listed_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "run-step00000010-state"
            newer = root / "run-step00000100-state"
            for path in (older, newer):
                write_state(path)
            # Explicit mtimes: the newest save point is listed first.
            os.utime(older, (1_700_000_000, 1_700_000_000))
            os.utime(newer, (1_700_000_100, 1_700_000_100))
            incomplete = root / "run-step00000200-state"
            incomplete.mkdir()
            (incomplete / "notes.txt").write_text("half written", encoding="utf-8")
            # weights were written but the RNG/step pickle never landed: not resumable.
            partial = root / "run-step00000300-state"
            partial.mkdir()
            (partial / "model.safetensors").write_bytes(b"")
            sources = accelerate_resume_sources(root)
            self.assertEqual([source.step for source in sources], [100, 10])
            self.assertEqual(sources[0].kind, "state")
            self.assertTrue(sources[0].label.startswith("第 100 步"))
            listed = [str(source.path) for source in sources]
            self.assertNotIn(str(incomplete), listed)
            self.assertNotIn(str(partial), listed)

    def test_epoch_and_final_state_dirs_are_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epoch = root / "run-000003-state"
            write_state(epoch)
            final = root / "run-state"
            write_state(final)
            labels = {source.path.name: source.label for source in accelerate_resume_sources(root)}
            self.assertEqual(labels["run-000003-state"], "第 3 轮的状态")
            self.assertEqual(labels["run-state"], "训练结束时的状态")


class AdapterResumeTests(unittest.TestCase):
    def test_musubi_and_sd_scripts_map_state_to_the_resume_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            state = output / "task-step00000100-state"
            write_state(state)
            job = write_job(Path(tmp) / "job", output)
            for adapter in (MusubiTunerAdapter(), KohyaAdapter()):
                with self.subTest(adapter=type(adapter).__name__):
                    sources = adapter.resume_sources(job)
                    self.assertEqual(len(sources), 1)
                    self.assertEqual(adapter.resume_parameters(sources[0]), {"resume": str(state)})
                    capabilities = adapter.recovery_capabilities(None)
                    self.assertTrue(capabilities.graceful_stop)
                    self.assertTrue(capabilities.resume_full_state)

    def test_kohaku_checkpoint_directories_become_state_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            step = output / "run" / "checkpoints" / "step_00000120"
            step.mkdir(parents=True)
            (step / "state.json").write_text(json.dumps({"step": 120}), encoding="utf-8")
            (step / "trainable.safetensors").write_bytes(b"")
            partial = output / "run" / "checkpoints" / "step_00000130.partial"
            partial.mkdir()
            (partial / "state.json").write_text("{}", encoding="utf-8")
            adapter = KohakuMossTrainerAdapter()
            sources = adapter.resume_sources(write_job(Path(tmp) / "job", output))
            self.assertEqual([source.step for source in sources], [120])
            self.assertEqual(adapter.resume_parameters(sources[0]), {"RESUME": str(step)})
            self.assertTrue(adapter.recovery_capabilities(None, "moss_image").resume_full_state)

    def test_anima_resumes_the_anima_line_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            state = output / "run" / "run-step00000040-state"
            write_state(state)
            adapter = AnimaLoraAdapter()
            sources = adapter.resume_sources(write_job(Path(tmp) / "anima-job", output, "anima"))
            self.assertEqual([source.step for source in sources], [40])
            self.assertEqual(adapter.resume_parameters(sources[0]), {"resume": str(state)})
            self.assertTrue(adapter.recovery_capabilities(None, "anima").resume_full_state)
            # Its MOSS line never saves state, so it must not offer a resume point.
            self.assertEqual(
                adapter.resume_sources(write_job(Path(tmp) / "moss-job", output, "moss_image")), ())
            self.assertFalse(adapter.recovery_capabilities(None, "moss_image").resume_full_state)

    def test_missing_job_directory_yields_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for adapter in (MusubiTunerAdapter(), KohyaAdapter(), KohakuMossTrainerAdapter()):
                with self.subTest(adapter=type(adapter).__name__):
                    self.assertEqual(adapter.resume_sources(Path(tmp) / "missing"), ())


if __name__ == "__main__":
    unittest.main()
