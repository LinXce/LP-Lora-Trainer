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
    @unittest.skipUnless(ENGINE.is_file(), "requires the pinned Kohaku checkout")
    def test_every_literal_training_default_and_both_cache_batch_sizes(self):
        adapter = KohakuMossTrainerAdapter()
        install = SimpleNamespace(installation_id="test", source_path=ENGINE.parents[2])
        fields = {p["key"]: p for p in adapter.capabilities_for_architecture(install, "moss_image")["params"]}
        checked = 0
        for stmt in ast.parse(ENGINE.read_text(encoding="utf-8")).body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            target = stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id not in fields:
                continue
            try:
                native = ast.literal_eval(stmt.value)
            except ValueError:
                continue
            value = fields[target.id]["default"]
            if isinstance(native, (list, tuple, dict)):
                value = ast.literal_eval(value)
            elif isinstance(native, bool) and isinstance(value, str):
                value = {"true": True, "false": False}[value]
            self.assertEqual(value, native, target.id)
            checked += 1
        self.assertGreaterEqual(checked, 160)
        for key, stage in (("ENCODE_BATCH", "cache_text"), ("VAE_ENCODE_BATCH", "cache_latents")):
            script = install.source_path / "scripts/data" / (stage + ".py")
            count = next(ast.literal_eval(s.value) for s in ast.parse(script.read_text(encoding="utf-8")).body
                         if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "ENCODE_BATCH")
            self.assertEqual(fields[key]["default"], count)

    def test_engine_globals_are_covered_or_composed(self):
        declared = {p.native_key or p.key for p in PARAMS if p.native}
        missing = engine_globals() - declared
        # Three are composed from UI fields. DRIVER is an upstream no-op.
        self.assertEqual(missing, {"CAPTION", "DATASETS", "OUTPUT_DIR", "DRIVER"})

    def test_new_defaults_follow_the_engine(self):
        cases = {
            "TRANSFORMER": "",
            "ARCH_OVERRIDES": "{}",
            "SAVE_FORMAT": "comfy",
            "SAVE_MERGED": False,
            "SAVE_FUSED_QKV": "false",
            "CHECKPOINT_EXPORT": True,
            "FINAL_CHECKPOINT": "auto",
            "FSDP_RESHARD": "forward",
            "METRICS_EVERY": None,
            "BASE_PRECISION": None,
            "ENCODE_DEVICES": "()",
            "ENCODE_DEPTH": 32,
            "ENCODE_CHUNK": 8,
            "INT8_COMPUTE": None,
            "INT8_DGRAD": None,
            "EXPORT_FORMATS": None,
            "PINNED": None,
            "CHECKPOINT": None,
            "VAE_ENCODE_BATCH": 8,
            "DIST_TIMEOUT": 900,
            "DDP_BUCKET_MB": 25,
            "DATA_BALANCE": False,
            "STEP_SYNC": "step",
            "STREAM_LOSS": "{}",
            "CONCENTRATION_LOSS": "{}",
            "STREAM_LOSS_PROBE": 0,
            "HARD_WEIGHT": 1.0,
            "TEACHER": "",
            "DISTILL_WEIGHT": 0.0,
            "DISTILL_WEIGHTING": "",
            "TEACHER_PRECISION": "auto",
            "TEACHER_STREAM": "compute",
            "OUTLIER_DECAY": "{}",
            "WD_TARGET": "zero",
            "METRICS_LEVEL": "train",
            "SAMPLE_BATCH": 4,
            "SAMPLE_COLUMNS": 0,
            "SAMPLE_AT_START": False,
            "INIT_FROM": "",
        }
        for key, expected in cases.items():
            self.assertEqual(param(key).default, expected, key)
        self.assertIn("fp16", param("CACHE_DTYPE").enum_values)
        self.assertIn("fp32", param("CACHE_DTYPE").enum_values)
        self.assertIn("bf16", param("DDP_REDUCE_DTYPE").enum_values)

    def test_native_loop_keeps_the_requested_precision_and_gbc_image_source(self):
        adapter = KohakuMossTrainerAdapter()
        config = dict(architecture="moss_image", name="demo", output_dir="out", dataset_path="dataset",
                      params=dict(PRECISION="fp16", FP16_FIX="none", DRIVER="lightning", SOURCE="gbc",
                                  DATASET_GBC_IMAGE_INDEX="index", DATASET_GBC_IMAGE_ROOT="https://example.test/",
                                  STREAM_LOSS='{"rms": {"weight": 0.02}}'))
        tree = ast.parse(adapter._render(config))
        knobs = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)}
        self.assertIn("scripts/train/moss.py", ast.get_docstring(tree))
        self.assertEqual(knobs["FP16_FIX"], "none")
        self.assertNotIn("DRIVER", knobs)
        self.assertEqual(knobs["STREAM_LOSS"], {"rms": {"weight": 0.02}})
        self.assertEqual(knobs["DATASETS"][0]["images"], {"index": "index", "root": "https://example.test/"})

    def test_packed_sampling_accepts_mixed_sizes_and_shared_seed(self):
        values = dict(SAMPLE_PROMPTS="cat\ndog", SAMPLE_SEEDS="(123,)", SAMPLE_SIZE="[(512, 640), (640, 512)]")
        issues = KohakuMossTrainerAdapter._memory_and_sample_issues(values)
        self.assertFalse(any(issue.field in {"SAMPLE_SEEDS", "SAMPLE_SIZE"} for issue in issues))
        values["SAMPLE_SIZE"] = "[(512, 512)]"
        self.assertIn("SAMPLE_SIZE", [issue.field for issue in KohakuMossTrainerAdapter._memory_and_sample_issues(values)])

    def test_older_registered_checkout_omits_unknown_native_defaults(self):
        adapter = KohakuMossTrainerAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts/train/moss.py"
            script.parent.mkdir(parents=True)
            script.write_text("WEIGHTS = ''\nDATA_BALANCE = False\n", encoding="utf-8")
            installation = SimpleNamespace(source_path=root, installation_id="prior")
            config = dict(architecture="moss_image", name="demo", output_dir="out", dataset_path="images", params={})
            fields = {p["key"] for p in adapter.capabilities(installation)["params"]}
            self.assertIn("DATA_BALANCE", fields)
            self.assertNotIn("METRICS_LEVEL", fields)
            preview = adapter.native_config_text(config, root / "dataset.toml", installation)
            self.assertIn("DATA_BALANCE = False", preview)
            self.assertNotIn("METRICS_LEVEL =", preview)

    def test_step_zero_sheet_and_grid_keep_their_own_step(self):
        adapter = KohakuMossTrainerAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "run" / "samples"
            step_dir = samples / "step_00000000"
            step_dir.mkdir(parents=True)
            for file in (samples / "grid.png", step_dir / "sheet.png", step_dir / "000.png"):
                file.touch()
            (root / "job.json").write_text(json.dumps(dict(output_dir=str(root / "run"))), encoding="utf-8")
            results = {a.path.name: a.step for a in adapter.collect_artifacts(root)}
            self.assertEqual(results, {"grid.png": None, "sheet.png": 0, "000.png": 0})

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
