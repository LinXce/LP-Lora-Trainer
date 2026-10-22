"""anima_lora preview images: sampling params and the caption fallback."""

import tempfile
import unittest
from pathlib import Path

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.sampling import PROMPT_FILENAME

adapter = AnimaLoraAdapter()


def dataset_with_caption(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.png").write_bytes(b"")
    (root / "a.txt").write_text(text, encoding="utf-8")
    return root


class PreviewPromptFallbackTests(unittest.TestCase):
    def test_first_caption_is_used_when_sampling_is_on_without_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = dataset_with_caption(Path(tmp), "linxcepeslanna, solo, standing\nsecond line\n")
            text = adapter._preview_prompt_fallback(
                {"params": {"sample_every_n_epochs": 1}, "dataset_path": str(root)})
            self.assertEqual(text, "linxcepeslanna, solo, standing")

    def test_explicit_prompt_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = dataset_with_caption(Path(tmp), "from caption")
            text = adapter._preview_prompt_fallback(
                {"params": {"sample_every_n_steps": 5, "sample_prompts": "custom"},
                 "dataset_path": str(root)})
            self.assertEqual(text, "")

    def test_no_sampling_means_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = dataset_with_caption(Path(tmp), "from caption")
            self.assertEqual(
                adapter._preview_prompt_fallback({"params": {}, "dataset_path": str(root)}), "")

    def test_missing_caption_yields_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.png").write_bytes(b"")
            self.assertEqual(
                adapter._preview_prompt_fallback(
                    {"params": {"sample_at_first": True}, "dataset_path": tmp}), "")

    def test_fallback_file_is_written_into_the_job_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = dataset_with_caption(Path(tmp) / "data", "prompt from caption")
            job = Path(tmp) / "job"
            job.mkdir()
            config = {
                "architecture": "anima",
                "params": {"sample_every_n_epochs": 1},
                "dataset_path": str(root),
            }
            text = adapter.native_config_text(config, job / "dataset.toml")
            self.assertIn("sample_prompts", text)
            self.assertEqual((job / PROMPT_FILENAME).read_text(encoding="utf-8").strip(), "prompt from caption")


if __name__ == "__main__":
    unittest.main()
