"""Regression tests for the audited Anima LoRA training-chain defects.

Every test targets one confirmed finding: the missing ``preprocess-config`` stage,
explicit ``False`` booleans being dropped, ``log_with`` without ``logging_dir``,
preview rendering into a ``"<task-id>"`` placeholder, the ``--config_file`` chain
bypass, unsupported schedulers, the MSVC environment for ``torch_compile``, the
unforwarded ``moss_ref_isolate`` switch, and state/cache folders leaking into the
artifact list.
"""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters.anima_lora import adapter as adapter_module
from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.anima_lora.params import PARAMS, engine_chain_extras

adapter = AnimaLoraAdapter()


def param(key):
    return next(spec for spec in PARAMS if spec.key == key)


def engine_source(root: Path) -> Path:
    """Minimal engine checkout with the real base → preset → method chain shape."""
    (root / "configs" / "methods").mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text("", encoding="utf-8")
    (root / "tasks.py").write_text("", encoding="utf-8")
    (root / "configs" / "base.toml").write_text(
        'output_dir = "output/ckpt"\n'
        'output_name = "anima_lora"\n'
        'logging_dir = "output/logs"\n'
        'log_with = "tensorboard"\n'
        'use_vae_cache = true\n'
        'use_text_cache = true\n'
        'skip_cache_check = true\n'
        'channel_scaling_alpha = 0.5\n',
        encoding="utf-8",
    )
    (root / "configs" / "presets.toml").write_text(
        "[default]\nblocks_to_swap = 0\n", encoding="utf-8")
    (root / "configs" / "methods" / "lora.toml").write_text(
        'learning_rate = 3e-5\n'
        'down_init = "weight_svd"\n'
        'use_repa = true\n'
        'repa_layer = 8\n'
        'repa_weight = 0.05\n',
        encoding="utf-8",
    )
    return root


def installation(root):
    return SimpleNamespace(source_path=str(root), python_executable="python")


def make_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.png").write_bytes(b"")
    (root / "a.txt").write_text("a caption", encoding="utf-8")
    return root


def job_config(tmp: Path, **params) -> dict:
    return {
        "architecture": "anima",
        "name": "probe",
        "output_name": "probe",
        "base_model_path": str(tmp / "base.safetensors"),
        "dataset_path": str(make_dataset(tmp / "data")),
        "output_dir": str(tmp / "out"),
        "params": params,
    }


class PreprocessStageTests(unittest.TestCase):
    """The engine refuses to train on an incomplete cache; it needs a cache pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = engine_source(Path(self.tmp.name) / "engine")
        self.install = installation(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _commands(self, **params):
        config = job_config(Path(self.tmp.name), **params)
        native = Path(self.tmp.name) / "job" / "config.native.toml"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_text("", encoding="utf-8")
        return adapter.preview_commands(self.install, config, native)

    def test_preprocess_config_runs_before_training(self):
        commands = self._commands(use_vae_cache=True, use_text_cache=True,
                                  vae=str(Path(self.tmp.name) / "vae.safetensors"),
                                  qwen3=str(Path(self.tmp.name) / "qwen3.safetensors"))
        self.assertEqual(len(commands), 2)
        cache, train = commands
        self.assertEqual(Path(cache[2]).name, "tasks.py")
        self.assertIn("preprocess-config", cache)
        self.assertIn("--dataset_config", cache)
        self.assertIn("--src", cache)
        dataset_config = cache[cache.index("--dataset_config") + 1]
        self.assertEqual(Path(dataset_config).name, "dataset.toml")
        # The originals are the read-only input, never the resize destination.
        self.assertEqual(cache[cache.index("--src") + 1],
                         job_config(Path(self.tmp.name))["dataset_path"])
        self.assertIn("--dit", cache)
        self.assertIn(str(Path(self.tmp.name) / "base.safetensors"), cache)
        self.assertIn("--vae", cache)
        self.assertIn("--qwen3", cache)
        self.assertEqual(Path(train[2]).name, "train.py")

    def test_one_cache_flag_is_enough_to_require_preprocess(self):
        for params in ({"use_vae_cache": True, "use_text_cache": False},
                       {"use_vae_cache": False, "use_text_cache": True}):
            with self.subTest(params=params):
                commands = self._commands(**params)
                self.assertEqual(len(commands), 2)
                self.assertEqual(Path(commands[0][2]).name, "tasks.py")

    def test_both_cache_flags_off_skips_preprocess(self):
        commands = self._commands(use_vae_cache=False, use_text_cache=False)
        self.assertEqual(len(commands), 1)
        self.assertEqual(Path(commands[0][2]).name, "train.py")

    def test_dataset_config_points_at_the_staged_tree_the_cache_pass_fills(self):
        config = job_config(Path(self.tmp.name))
        job = Path(self.tmp.name) / "job"
        job.mkdir()
        native = adapter.write_native_config(self.install, config, job)
        dataset = native.parent / "dataset.toml"
        self.assertTrue(dataset.is_file())
        values = tomllib.loads(dataset.read_text(encoding="utf-8"))
        subset = values["datasets"][0]["subsets"][0]
        self.assertEqual(subset["image_dir"], str(Path(self.tmp.name) / "out" / "resized"))
        self.assertEqual(subset["cache_dir"], str(Path(self.tmp.name) / "out" / "cache"))
        # The written dataset.toml is exactly the file the preprocess stage reads.
        stages = json.loads((job / "stages.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0][stages[0].index("--dataset_config") + 1], str(dataset))

    def test_live_encoding_reads_the_original_dataset(self):
        config = job_config(Path(self.tmp.name), use_vae_cache=False, use_text_cache=False)
        job = Path(self.tmp.name) / "job"
        job.mkdir()
        native = adapter.write_native_config(self.install, config, job)
        values = tomllib.loads((native.parent / "dataset.toml").read_text(encoding="utf-8"))
        self.assertEqual(values["datasets"][0]["subsets"][0]["image_dir"], config["dataset_path"])
        stages = json.loads((job / "stages.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stages), 1)


class ExplicitFalseBooleanTests(unittest.TestCase):
    """A false value on a default-true knob must reach the config file."""

    def test_default_true_knobs_keep_an_explicit_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / "job"
            job.mkdir()
            config = job_config(tmp, use_vae_cache=False, use_text_cache=False,
                                use_shuffled_caption_variants=False,
                                compile_dynamic_seq=False)
            text = adapter.native_config_text(config, job / "dataset.toml")
            self.assertIn("use_vae_cache = false", text)
            self.assertIn("use_text_cache = false", text)
            self.assertIn("use_shuffled_caption_variants = false", text)
            self.assertIn("compile_dynamic_seq = false", text)

    def test_default_false_knobs_stay_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / "job"
            job.mkdir()
            config = job_config(tmp, gradient_checkpointing=False, highvram=False)
            text = adapter.native_config_text(config, job / "dataset.toml")
            self.assertNotIn("gradient_checkpointing", text)
            self.assertNotIn("highvram", text)

    def test_engine_defaults_keep_false_for_a_chain_knob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = engine_source(Path(tmp) / "engine")
            job = Path(tmp) / "job"
            job.mkdir()
            config = job_config(Path(tmp), use_vae_cache=False)
            native = adapter.write_native_config(installation(root), config, job)
            values = tomllib.loads(native.read_text(encoding="utf-8"))
            self.assertIs(values["use_vae_cache"], False)
            self.assertIs(values["use_text_cache"], True)


class LoggingDirectoryTests(unittest.TestCase):
    def test_tensorboard_without_a_directory_gets_a_task_local_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / "job"
            job.mkdir()
            config = job_config(tmp, log_with="tensorboard")
            values = tomllib.loads(adapter.native_config_text(config, job / "dataset.toml"))
            self.assertEqual(values["log_with"], "tensorboard")
            self.assertEqual(values["logging_dir"], str(tmp / "out" / "logs"))

    def test_a_user_directory_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            job = tmp / "job"
            job.mkdir()
            chosen = tmp / "my-logs"
            config = job_config(tmp, log_with="tensorboard", logging_dir=str(chosen))
            values = tomllib.loads(adapter.native_config_text(config, job / "dataset.toml"))
            self.assertEqual(values["logging_dir"], str(chosen))

    def test_the_engine_relative_default_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = engine_source(Path(tmp) / "engine")
            caps = adapter.capabilities(
                SimpleNamespace(installation_id="x", source_path=str(root)))
            defaults = {spec["key"]: spec["default"] for spec in caps["params"]}
            # base.toml says "output/logs" (relative to the engine checkout).
            self.assertEqual(defaults["logging_dir"], "")


class PlaceholderPreviewTests(unittest.TestCase):
    """A preview can render with the "<task-id>" display marker; it must not write."""

    def test_rendering_a_placeholder_directory_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = job_config(tmp, sample_prompts="a prompt")
            placeholder = Path("<task-id>") / "dataset.toml"
            text = adapter.native_config_text(config, placeholder)
            self.assertIn("sample_prompts = ", text)
            self.assertFalse(Path("<task-id>").exists())

    def test_the_caption_fallback_is_skipped_for_a_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = job_config(tmp, sample_every_n_epochs=1)
            text = adapter.native_config_text(config, Path("<task-id>") / "dataset.toml")
            self.assertNotIn("sample_prompts", text)
            self.assertFalse(Path("<task-id>").exists())


class MethodChainTests(unittest.TestCase):
    """--config_file skips base → preset → method, so the chain must be pinned."""

    def test_training_argv_requests_the_lora_method_and_default_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = engine_source(tmp / "engine")
            config = job_config(tmp, use_vae_cache=False, use_text_cache=False)
            native = tmp / "job" / "config.native.toml"
            native.parent.mkdir()
            native.write_text("", encoding="utf-8")
            train = adapter.preview_argv(installation(root), config, native)
            self.assertEqual(Path(train[2]).name, "train.py")
            self.assertEqual(train[train.index("--method") + 1], "lora")
            self.assertEqual(train[train.index("--preset") + 1], "default")

    def test_unmodelled_chain_defaults_reach_the_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = engine_source(tmp / "engine")
            extras = engine_chain_extras(root)
            for key in ("down_init", "use_repa", "repa_layer", "repa_weight",
                        "channel_scaling_alpha"):
                self.assertIn(key, extras)
            # Job paths/identity are owned by LP and must never be inherited.
            for key in ("output_dir", "output_name", "logging_dir"):
                self.assertNotIn(key, extras)
            job = tmp / "job"
            job.mkdir()
            config = job_config(tmp, use_vae_cache=False, use_text_cache=False)
            native = adapter.write_native_config(installation(root), config, job)
            values = tomllib.loads(native.read_text(encoding="utf-8"))
            self.assertEqual(values["down_init"], "weight_svd")
            self.assertIs(values["use_repa"], True)
            self.assertEqual(values["repa_layer"], 8)
            # Modeled chain values are rendered too (from the same defaults overlay).
            self.assertIs(values["skip_cache_check"], True)
            self.assertEqual(values["output_dir"], str(tmp / "out"))


class SchedulerChoiceTests(unittest.TestCase):
    """The engine instantiates transformers SchedulerType; diffusers-only names fail."""

    ENGINE_VALUES = {
        "constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
        "cosine_with_min_lr", "linear", "polynomial", "inverse_sqrt",
        "piecewise_constant",
    }

    def test_unsupported_schedulers_are_gone(self):
        offered = set(param("lr_scheduler").enum_values)
        self.assertNotIn("adafactor", offered)
        self.assertNotIn("rex", offered)

    def test_only_engine_supported_schedulers_are_offered(self):
        offered = set(param("lr_scheduler").enum_values)
        self.assertTrue(offered <= self.ENGINE_VALUES, offered - self.ENGINE_VALUES)
        self.assertEqual(param("lr_scheduler").default, "cosine")


class CompileEnvironmentTests(unittest.TestCase):
    """torch.compile shells out to cl.exe, so the MSVC environment must be injected."""

    def _launch(self, params, native_text=""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "stages.json").write_text(
            json.dumps([["python", "-u", "train.py"]]), encoding="utf-8")
        (root / "config.native.toml").write_text(native_text, encoding="utf-8")
        if params is not None:
            (root / "job.json").write_text(json.dumps({"params": params}), encoding="utf-8")
        with patch.object(adapter_module, "_msvc_environment", return_value={"INCLUDE": "msvc"}):
            return adapter.build_launch(
                SimpleNamespace(source_path=str(root), python_executable="python"),
                root / "config.native.toml",
            )

    def test_anima_torch_compile_injects_the_msvc_environment(self):
        launch = self._launch({"torch_compile": True})
        self.assertEqual(launch.environment_overrides.get("INCLUDE"), "msvc")

    def test_no_compile_keeps_the_plain_environment(self):
        launch = self._launch({"torch_compile": False})
        self.assertNotIn("INCLUDE", launch.environment_overrides)

    def test_the_rendered_config_is_the_fallback_before_job_json_exists(self):
        launch = self._launch(None, "torch_compile = true\n")
        self.assertEqual(launch.environment_overrides.get("INCLUDE"), "msvc")

    def test_anima_torch_compile_without_msvc_is_reported_before_queueing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = engine_source(Path(tmp) / "engine")
            config = job_config(Path(tmp), torch_compile=True,
                                pretrained_model_name_or_path=str(Path(tmp) / "base.safetensors"),
                                vae=str(Path(tmp) / "vae.safetensors"),
                                qwen3=str(Path(tmp) / "qwen3.safetensors"))
            with patch.object(adapter_module, "_msvc_available", return_value=False):
                issues = adapter.validate(installation(root), config)
            fields = [issue.field for issue in issues if issue.severity == "error"]
            self.assertIn("torch_compile", fields)


class MossRefIsolateTests(unittest.TestCase):
    def test_ref_isolate_is_forwarded_to_the_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tmp / "MOSS-Image-Base"
            model.mkdir()
            config = {
                "architecture": "moss_image",
                "dataset_path": str(make_dataset(tmp / "data")),
                "output_dir": str(tmp / "out"),
                "output_name": "probe",
                "params": {"moss_model_dir": str(model), "moss_ref_isolate": True},
            }
            _, train = adapter.preview_commands(installation(tmp), config,
                                                tmp / "config.native.toml")
            self.assertIn("--ref_isolate", train)

    def test_ref_isolate_stays_off_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tmp / "MOSS-Image-Base"
            model.mkdir()
            config = {
                "architecture": "moss_image",
                "dataset_path": str(make_dataset(tmp / "data")),
                "output_dir": str(tmp / "out"),
                "output_name": "probe",
                "params": {"moss_model_dir": str(model), "moss_ref_isolate": False},
            }
            _, train = adapter.preview_commands(installation(tmp), config,
                                                tmp / "config.native.toml")
            self.assertNotIn("--ref_isolate", train)


class ArtifactCollectionTests(unittest.TestCase):
    def test_state_and_cache_folders_are_not_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "out"
            job = tmp / "job"
            job.mkdir()
            for relative in (
                "probe.safetensors",
                "sample/probe_0001.png",
                "cache/a_1024x1024_anima.npz",
                "resized/a.png",
                "probe-step00000100-state/model.safetensors",
                "probe-state/model.safetensors",
                "probe-checkpoint-state/model.safetensors",
                "probe-step00000100-state/random_states_0.pkl",
            ):
                path = out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
            (job / "job.json").write_text(
                json.dumps({"output_dir": str(out)}), encoding="utf-8")
            names = {artifact.path.name for artifact in adapter.collect_artifacts(job)}
            self.assertIn("probe.safetensors", names)
            self.assertIn("probe_0001.png", names)
            self.assertNotIn("model.safetensors", names)


class DynamicDefaultTests(unittest.TestCase):
    """Preview/export can be handed the engine chain, so the shown values agree."""

    def test_export_uses_the_engine_chain_when_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = engine_source(tmp / "engine")
            config = {
                "architecture": "anima",
                "params": {},
                "dataset_path": str(make_dataset(tmp / "data")),
            }
            without, _ = adapter.export_parameters(config)
            with_defaults, _ = adapter.export_parameters(
                config, adapter._defaults(installation(root)))
            self.assertIn("learning_rate = 2e-05", without)
            self.assertIn("learning_rate = 3e-05", with_defaults)


if __name__ == "__main__":
    unittest.main()
