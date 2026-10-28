"""Anima 2.x / MOSS Toolkit 0.5 parameter surface: no stale options or flags."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.anima_lora.params import MOSS_PARAMS, PARAMS, default_values


def param(key):
    return next(spec for spec in PARAMS if spec.key == key)


class AnimaParameterSurfaceTests(unittest.TestCase):
    def test_removed_timestep_sampler_is_not_offered(self):
        spec = param("timestep_sampling")
        self.assertNotIn("logsnr", spec.enum_values)
        self.assertEqual(spec.default, "sigmoid")

    def test_new_weighting_schemes_are_offered(self):
        values = set(param("weighting_scheme").enum_values)
        self.assertTrue({"min_snr", "uniform"} <= values, values)

    def test_save_precision_matches_engine_choices(self):
        spec = param("save_precision")
        self.assertEqual(spec.type, "enum")
        self.assertEqual(set(spec.enum_values), {"None", "float", "fp16", "bf16"})

    def test_base_config_scheduler_and_cache_defaults_are_mirrored(self):
        self.assertEqual(param("lr_scheduler").default, "cosine")
        self.assertEqual(param("lr_warmup_steps").default, 0.05)
        self.assertTrue(param("use_vae_cache").default)
        self.assertTrue(param("use_text_cache").default)

    def test_release_pause_environment_is_only_for_the_anima_line(self):
        adapter = AnimaLoraAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def launch(stages):
                (root / "stages.json").write_text(json.dumps(stages), encoding="utf-8")
                (root / "config.native.toml").write_text("", encoding="utf-8")
                return adapter.build_launch(
                    SimpleNamespace(source_path=tmp, python_executable="python"),
                    root / "config.native.toml",
                )

            anima = launch([["python", "train.py"]])
            self.assertIn("ANIMA_DAEMON_JOB_DIR", anima.environment_overrides)
            moss = launch([
                ["python", "scripts/moss/cache.py"],
                ["python", "scripts/moss/train.py"],
            ])
            self.assertNotIn("ANIMA_DAEMON_JOB_DIR", moss.environment_overrides)


class MossToolkitParamsTests(unittest.TestCase):
    def test_new_train_and_cache_fields_are_declared(self):
        keys = {p.key for p in MOSS_PARAMS}
        expected = {
            "moss_grad_accum", "moss_svd_slice", "moss_svd_slice_count",
            "moss_repa_layer", "moss_repa_pe_root", "moss_fit_report",
            "moss_expandable_segments", "moss_cache_text_encoder",
            "moss_cache_vae_dtype", "moss_cache_max_text_tokens", "moss_cache_overwrite",
            "moss_cache_skip_text", "moss_cache_skip_latents", "moss_cache_save_crops",
            "moss_optimizer", "moss_ema_window", "moss_ema_extra",
            "moss_swap_h2d_only", "moss_svd_drop_loud", "moss_tone_weight",
            "moss_region_weight",
        }
        self.assertTrue(expected <= keys, expected - keys)

    def test_new_engine_defaults_are_mirrored(self):
        self.assertEqual(param("moss_down_init").default, "kaiming")
        self.assertEqual(param("moss_optimizer").default, "muon")
        self.assertEqual(param("moss_blocks_to_swap").default, 36)
        self.assertEqual(param("moss_activation_memory_budget").default, 0.5)
        self.assertTrue(param("moss_expandable_segments").default)
        self.assertEqual(param("moss_ema_window").default, 0.1)
        self.assertTrue(param("moss_swap_h2d_only").default)
        self.assertTrue(default_values("engine/anima_lora")["moss_swap_h2d_only"])

    def test_preview_commands_use_the_new_flags(self):
        adapter = AnimaLoraAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            installation = SimpleNamespace(source_path=tmp, python_executable="python")
            config = {
                "architecture": "moss_image",
                "dataset_path": str(Path(tmp) / "data"),
                "output_dir": str(Path(tmp) / "out"),
                "output_name": "probe",
                "params": {
                    "moss_grad_accum": 2,
                    "moss_svd_slice": 1,
                    "moss_repa_layer": 8,
                    "moss_fit_report": True,
                    "moss_expandable_segments": True,
                },
            }
            cache, train = adapter.preview_commands(installation, config, Path(tmp) / "native.txt")
            cache_text = " ".join(cache)
            train_text = " ".join(train)
            self.assertIn("--text_encoder int8", cache_text)
            self.assertIn("--vae_dtype fp32", cache_text)
            self.assertIn("--max_text_tokens 4096", cache_text)
            self.assertIn("--grad_accum 2", train_text)
            self.assertIn("--svd_slice 1", train_text)
            self.assertIn("--repa_layer 8", train_text)
            self.assertIn("--fit_report", train_text)
            self.assertIn("--expandable_segments", train_text)


class EngineDefaultTests(unittest.TestCase):
    def _engine(self, root: Path) -> None:
        (root / "configs" / "methods").mkdir(parents=True)
        (root / "train.py").write_text("", encoding="utf-8")
        (root / "configs" / "base.toml").write_text(
            'network_module = "networks.lora_anima"\n'
            'lr_scheduler = "cosine"\n'
            'use_vae_cache = true\n'
            'use_text_cache = true\n',
            encoding="utf-8",
        )
        (root / "configs" / "presets.toml").write_text(
            '[default]\nblocks_to_swap = 0\n', encoding="utf-8")
        (root / "configs" / "methods" / "lora.toml").write_text(
            'learning_rate = 2e-5\n'
            'network_dim = 32\n'
            'network_alpha = 128\n'
            'max_train_epochs = 8\n'
            'save_every_n_epochs = 8\n'
            'torch_compile = true\n'
            'blocks_to_swap = 2\n',
            encoding="utf-8",
        )

    def test_capabilities_follow_the_engine_config_chain(self):
        adapter = AnimaLoraAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._engine(root)
            caps = adapter.capabilities(SimpleNamespace(installation_id="x", source_path=str(root)))
            specs = {spec["key"]: spec["default"] for spec in caps["params"]}
            self.assertEqual(specs["learning_rate"], 2e-5)
            self.assertEqual(specs["network_dim"], 32)
            self.assertEqual(specs["network_alpha"], 128)
            self.assertEqual(specs["lr_scheduler"], "cosine")
            self.assertEqual(specs["lr_warmup_steps"], 0.05)  # static fallback; fake base omits it
            self.assertEqual(specs["max_train_epochs"], 8)
            self.assertEqual(specs["save_every_n_epochs"], 8)
            self.assertEqual(specs["blocks_to_swap"], 2)
            self.assertFalse(specs["torch_compile"])  # Windows-safe override

    def test_moss_defaults_come_from_request_dataclasses(self):
        adapter = AnimaLoraAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._engine(root)
            (root / "scripts" / "moss").mkdir(parents=True)
            (root / "scripts" / "moss" / "train.py").write_text("", encoding="utf-8")
            (root / "library" / "moss").mkdir(parents=True)
            (root / "library" / "moss" / "requests.py").write_text(
                'DEFAULT_TE = "int8"\n'
                'def _f(default, help): return default\n'
                'class CacheRequest:\n'
                '    text_encoder: str = _f(DEFAULT_TE, "x")\n'
                '    vae_dtype: str = _f("bf16", "x")\n'
                'class TrainRequest:\n'
                '    epochs: int = _f(9, "x")\n'
                '    compile: bool = _f(True, "x")\n',
                encoding="utf-8",
            )
            caps = adapter.capabilities(SimpleNamespace(installation_id="x", source_path=str(root)))
            specs = {spec["key"]: spec["default"] for spec in caps["params"]}
            self.assertEqual(specs["moss_cache_text_encoder"], "int8")
            self.assertEqual(specs["moss_cache_vae_dtype"], "bf16")
            self.assertEqual(specs["moss_epochs"], 9)
            self.assertFalse(specs["moss_compile"])  # adapter's Windows-safe override


if __name__ == "__main__":
    unittest.main()
