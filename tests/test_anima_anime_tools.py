"""Anima's MOSS cache contract is only present in anime-tools >= 0.8.1."""

import tempfile
import unittest
from pathlib import Path

from adapters.anima_lora.adapter import AnimaLoraAdapter
from types import SimpleNamespace


def fake_source(root: Path, version: str | None):
    (root / "uv.lock").write_text("", encoding="utf-8")
    (root / "train.py").write_text("", encoding="utf-8")
    site = root / ".lp" / "python" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    if version:
        (site / f"anime_tools-{version}.dist-info").mkdir()


class AnimaAnimeToolsTests(unittest.TestCase):
    def test_tag_csv_is_found_in_project_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine" / "anima_lora"
            engine.mkdir(parents=True)
            (Path(tmp) / "models").mkdir()
            (Path(tmp) / "models" / "danbooru_tags_classified.csv").write_text("x", encoding="utf-8")
            self.assertTrue(AnimaLoraAdapter._tag_csv_present(
                SimpleNamespace(source_path=str(engine))))

    def test_old_anime_tools_is_reported_before_cache_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_source(root, "0.7.13")
            issues = AnimaLoraAdapter().source_issues(root)
            self.assertTrue(any("anime-tools=0.7.13" in issue for issue in issues))

    def test_current_anime_tools_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_source(root, "0.8.1")
            self.assertEqual(AnimaLoraAdapter().source_issues(root), ())

    def test_user_managed_venv_does_not_get_a_static_false_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_source(root, "0.7.13")
            (root / ".venv").mkdir()
            self.assertEqual(AnimaLoraAdapter().source_issues(root), ())


if __name__ == "__main__":
    unittest.main()
