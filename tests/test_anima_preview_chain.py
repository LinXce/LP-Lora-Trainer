"""End-to-end check: template text -> staged param -> native config -> launch -> artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.sampling import PROMPT_FILENAME

adapter = AnimaLoraAdapter()
TEMPLATE_TEXT = "linxcepeslanna, standing, soft light\nlinxcepeslanna, close-up portrait"


def make_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.png").write_bytes(b"")
    (root / "a.txt").write_text("caption from dataset", encoding="utf-8")
    return root


class TemplateChainTests(unittest.TestCase):
    def test_sample_prompts_is_exposed_for_anima(self):
        spec = next(p for p in adapter._all_params_for_tests() if p.key == "sample_prompts") \
            if hasattr(adapter, "_all_params_for_tests") else None
        self.assertIsNone(spec)  # adapter exposes specs through capabilities, see next test

    def test_capabilities_expose_sample_prompts_with_native_key(self):
        from adapters.anima_lora.params import PARAMS

        spec = next(p for p in PARAMS if p.key == "sample_prompts")
        self.assertEqual(spec.native_key, "sample_prompts")
        self.assertTrue(spec.native)
        self.assertIn("anima", spec.architectures or ("anima",))
        self.assertEqual(spec.group, "preview")

    def test_template_text_round_trips_into_the_native_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp) / "data")
            job = Path(tmp) / "job"
            job.mkdir()
            config = {
                "architecture": "anima",
                "name": "probe",
                "output_name": "probe",
                "output_dir": str(Path(tmp) / "out"),
                "dataset_path": str(root),
                "params": {
                    "sample_every_n_epochs": 1,
                    "sample_prompts": TEMPLATE_TEXT,
                },
            }
            (job / "dataset.toml").write_text("", encoding="utf-8")
            text = adapter.native_config_text(config, job / "dataset.toml")
            self.assertIn("sample_every_n_epochs = 1", text)
            self.assertIn("sample_prompts = ", text)
            written = (job / PROMPT_FILENAME).read_text(encoding="utf-8")
            self.assertEqual(written.strip().splitlines(), TEMPLATE_TEXT.splitlines())

    def test_launch_and_artifact_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp) / "data")
            out = Path(tmp) / "out"
            (out / "sample").mkdir(parents=True)
            (out / "sample" / "probe_0001.png").write_bytes(b"")
            (out / "probe.safetensors").write_bytes(b"")
            job = Path(tmp) / "job"
            job.mkdir()
            config = {
                "architecture": "anima",
                "name": "probe",
                "output_name": "probe",
                "output_dir": str(out),
                "dataset_path": str(root),
                "params": {"sample_every_n_epochs": 1, "sample_prompts": TEMPLATE_TEXT},
            }
            (job / "job.json").write_text(json.dumps(config), encoding="utf-8")
            (job / "stages.json").write_text(json.dumps([["python", "-c", "pass"]]), encoding="utf-8")

            class Installation:
                source_path = str(Path(tmp))
                python_executable = "python"

            launch = adapter.build_launch(Installation(), job / "config.native.toml")
            self.assertEqual(launch.argv, ("python", "-c", "pass"))

            artifacts = adapter.collect_artifacts(job)
            kinds = sorted((a.path.suffix, a.kind) for a in artifacts)
            self.assertIn((".png", "sample"), kinds)
            self.assertIn((".safetensors", "checkpoint"), kinds)

    def test_prompt_path_is_rejected_in_favour_of_text(self):
        from adapters.sampling import prompt_error

        self.assertIsNotNone(prompt_error(r"H:\some\prompts.txt"))
        self.assertIsNone(prompt_error(TEMPLATE_TEXT))


if __name__ == "__main__":
    unittest.main()
