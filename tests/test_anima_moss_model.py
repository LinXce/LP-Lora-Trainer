"""MOSS-Image model directory validation for the anima_lora adapter.

The MOSS-Image line trains from a diffusers folder (MOSS-Image-Base) instead of the
generic base-model file, so the adapter must refuse a missing / malformed directory
at submit time rather than letting cache.py or train.py fail on it.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.anima_lora.adapter import MOSS_MODEL_SUBDIRS, AnimaLoraAdapter
from adapters.anima_lora.params import PARAMS


class MossModelDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # A source checkout that contains the MOSS entry, otherwise validate() would
        # also report the architecture itself as unavailable.
        source = self.root / "engine"
        (source / "scripts" / "moss").mkdir(parents=True)
        (source / "scripts" / "moss" / "train.py").touch()
        self.installation = SimpleNamespace(source_path=source, installation_id="test")
        self.adapter = AnimaLoraAdapter()

    def issues(self, model_dir=None):
        params = {} if model_dir is None else {"moss_model_dir": model_dir}
        config = {"architecture": "moss_image", "params": params}
        return tuple(self.adapter.validate(self.installation, config))

    def moss_issues(self, model_dir=None):
        return tuple(issue for issue in self.issues(model_dir) if issue.field == "moss_model_dir")

    def test_empty_model_directory_is_an_error(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                issues = self.moss_issues(value)
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].severity, "error")
                self.assertIn("MOSS-Image-Base", issues[0].message)

    def test_a_missing_path_is_an_error(self):
        missing = self.root / "does-not-exist" / "MOSS-Image-Base"
        issues = self.moss_issues(str(missing))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("不存在", issues[0].message)

    def test_a_file_is_not_a_model_directory(self):
        target = self.root / "MOSS-Image-Base"
        target.write_text("not a directory", encoding="utf-8")
        issues = self.moss_issues(str(target))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("不是目录", issues[0].message)

    def test_each_missing_subdirectory_is_reported(self):
        model_dir = self.root / "MOSS-Image-Base"
        (model_dir / "transformer").mkdir(parents=True)
        issues = self.moss_issues(str(model_dir))
        self.assertEqual(len(issues), 3)
        self.assertTrue(all(issue.severity == "error" for issue in issues))
        for name in ("text_encoder", "vae", "tokenizer"):
            with self.subTest(subdirectory=name):
                self.assertTrue(any(name in issue.message for issue in issues))
        self.assertFalse(any("transformer" in issue.message for issue in issues))

    def test_a_complete_model_directory_passes(self):
        model_dir = self.root / "MOSS-Image-Base"
        for name in MOSS_MODEL_SUBDIRS:
            (model_dir / name).mkdir(parents=True, exist_ok=True)
        self.assertEqual(self.moss_issues(str(model_dir)), ())
        self.assertEqual(self.issues(str(model_dir)), ())

    def test_form_metadata_marks_the_model_directory_required(self):
        p = next(p for p in PARAMS if p.key == "moss_model_dir")
        self.assertEqual(p.required_for, ("moss_image",))
        specs = {p["key"]: p for p in self.adapter.capabilities(self.installation)["params"]}
        self.assertEqual(specs["moss_model_dir"]["required_for"], ["moss_image"])
        # The generic base-model file stays hidden for this architecture.
        self.assertEqual(
            self.adapter.capabilities(self.installation)["base_model_optional_architectures"],
            ["moss_image"],
        )


if __name__ == "__main__":
    unittest.main()
