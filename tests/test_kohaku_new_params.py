"""KohakuMossTrainer 0.5+ parameter surface: new save/export and parallel knobs."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter
from adapters.kohaku_moss_trainer.params import PARAMS

ENGINE = Path("engine/KohakuMossTrainer/scripts/train/moss.py")


def param(key):
    return next(spec for spec in PARAMS if spec.key == key)


def engine_globals():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8", errors="ignore"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class KohakuNewParamsTests(unittest.TestCase):
    def test_engine_globals_are_covered_or_composed(self):
        declared = {p.native_key or p.key for p in PARAMS if p.native}
        missing = engine_globals() - declared
        # These are composed by the adapter from dedicated UI fields.
        self.assertEqual(missing, {"CAPTION", "DATASETS", "OUTPUT_DIR"})

    def test_new_defaults_follow_the_engine(self):
        cases = {
            "TRANSFORMER": "",
            "ARCH_OVERRIDES": "{}",
            "SAVE_FORMAT": "comfy",
            "SAVE_MERGED": False,
            "SAVE_FUSED_QKV": "auto",
            "CHECKPOINT_EXPORT": True,
            "FINAL_CHECKPOINT": "auto",
            "FSDP_RESHARD": "forward",
            "METRICS_EVERY": None,
            "BASE_PRECISION": "auto",
            "EXPORT_FORMATS": None,
            "PINNED": None,
            "CHECKPOINT": None,
            "VAE_ENCODE_BATCH": 8,
        }
        for key, expected in cases.items():
            self.assertEqual(param(key).default, expected, key)
        self.assertIn("fp16", param("CACHE_DTYPE").enum_values)
        self.assertIn("fp32", param("CACHE_DTYPE").enum_values)

    def test_render_uses_new_save_and_model_knobs(self):
        adapter = KohakuMossTrainerAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "architecture": "moss_image",
                "output_dir": str(root),
                "name": "probe",
                "dataset_path": str(root),
                "params": {
                    "TRANSFORMER": "dit.safetensors",
                    "ARCH_OVERRIDES": '{"layers": 42}',
                    "SAVE_FORMAT": "hf",
                    "SAVE_MERGED": True,
                    "SAVE_FUSED_QKV": "true",
                    "CHECKPOINT_EXPORT": False,
                    "FINAL_CHECKPOINT": "false",
                    "FSDP_RESHARD": "backward",
                    "METRICS_EVERY": 5,
                },
            }
            text = adapter._render(config)
            self.assertIn("TRANSFORMER = 'dit.safetensors'", text)
            self.assertIn("ARCH_OVERRIDES = {'layers': 42}", text)
            self.assertIn("SAVE_FORMAT = 'hf'", text)
            self.assertIn("SAVE_MERGED = True", text)
            self.assertIn("SAVE_FUSED_QKV = True", text)
            self.assertIn("CHECKPOINT_EXPORT = False", text)
            self.assertIn("FINAL_CHECKPOINT = False", text)
            self.assertIn("FSDP_RESHARD = 'backward'", text)
            self.assertIn("METRICS_EVERY = 5", text)

    def test_cache_stages_get_their_own_encode_batch(self):
        adapter = KohakuMossTrainerAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "architecture": "moss_image",
                "output_dir": str(root),
                "name": "probe",
                "dataset_path": str(root),
                "params": {
                    "WEIGHTS": str(root),
                    "TEXT_ENCODER": "",
                    "VAE": "",
                    "ENCODE_BATCH": 64,
                    "VAE_ENCODE_BATCH": 4,
                },
            }
            installation = SimpleNamespace(source_path=str(root), python_executable="python")
            stages = adapter.preview_commands(installation, config, root / "native.py")
            self.assertEqual(len(stages), 3)
            self.assertIn("ENCODE_BATCH=64", stages[0])
            self.assertNotIn("ENCODE_BATCH=4", stages[0])
            self.assertIn("ENCODE_BATCH=4", stages[1])
            self.assertNotIn("ENCODE_BATCH=64", stages[1])


if __name__ == "__main__":
    unittest.main()
