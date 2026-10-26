"""Regression tests for the P2 Anima LoRA adapter items.

Covered findings (audit numbering):

* M1 — preview/export used the adapter's static defaults while the queued job used
  the engine chain, so the shown/exported config differed from the run config.
* M3 — ``vocab_pack`` silently disabled the engine's CJK pack with no explanation.
* M5 — parameters the engine declares but never reads (EMA, per-group LRs,
  metadata_usage_hint/thumbnail/merged_from/trigger_phrase, llm_adapter_path).
* M6 — the MOSS native view listed only the train-stage switches, so it disagreed
  with the argv ``preview_commands`` produces (missing --src/--out/--cache/
  --output/--model/--model_dir and the explicit --no- forms).
* output_name — the checkpoint file stem was the raw task name, so a name with a
  path separator or a Windows-reserved character could escape output_dir.
"""

import json
import re
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters.anima_lora import adapter as adapter_module
from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.anima_lora.params import PARAMS

adapter = AnimaLoraAdapter()

# Parameters the pinned engine declares but never reads (see params.py markers).
DEAD_KEYS = (
    "ema", "ema_decay", "ema_device", "ema_use_num_updates", "ema_use_feedback",
    "ema_param_multiplier", "ema_resume_path",
    "llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr",
    "metadata_usage_hint", "metadata_thumbnail", "metadata_merged_from",
    "metadata_trigger_phrase", "llm_adapter_path",
)


def param(key):
    return next(spec for spec in PARAMS if spec.key == key)


def engine_source(root: Path) -> Path:
    """Minimal checkout with the real base → preset → method chain shape.

    The chain carries values the form models (learning_rate) *and* values it does
    not (down_init, use_repa, channel_scaling_alpha), which is what M1 is about.
    """
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
        'discrete_flow_shift = 1.0\n'
        'channel_scaling_alpha = 0.5\n'
        'vocab_pack = "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"\n',
        encoding="utf-8",
    )
    (root / "configs" / "presets.toml").write_text(
        "[default]\nblocks_to_swap = 0\n", encoding="utf-8")
    (root / "configs" / "methods" / "lora.toml").write_text(
        'learning_rate = 3e-5\n'
        'down_init = "weight_svd"\n'
        'use_repa = true\n'
        'repa_layer = 8\n',
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


def job_config(tmp: Path, architecture="anima", **params) -> dict:
    config = {
        "architecture": architecture,
        "name": "probe",
        "output_name": "probe",
        "base_model_path": str(tmp / "base.safetensors"),
        "dataset_path": str(make_dataset(tmp / "data")),
        "output_dir": str(tmp / "out"),
        "params": params,
    }
    if architecture == "moss_image":
        model = tmp / "moss" / "MOSS-Image-Base"
        for name in ("transformer", "text_encoder", "vae", "tokenizer"):
            (model / name).mkdir(parents=True, exist_ok=True)
        config["params"]["moss_model_dir"] = str(model)
    return config


def split_flags(line: str):
    """Split a rendered flag line; quoted tokens may contain spaces."""
    tokens = []
    for raw in re.findall(r'"[^"]*"|\S+', line):
        tokens.append(raw[1:-1] if len(raw) > 1 and raw[0] == raw[-1] == '"' else raw)
    return tokens


class PreviewAndExportUseTheEngineChainTests(unittest.TestCase):
    """M1: preview/export render exactly what write_native_config will run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.install = installation(engine_source(self.root / "engine"))

    def tearDown(self):
        self.tmp.cleanup()

    def _job(self, **params):
        config = job_config(self.root, **params)
        job = self.root / "job"
        native = adapter.write_native_config(self.install, config, job)
        return config, job, native

    def test_preview_matches_the_queued_config_when_given_the_installation(self):
        config, job, native = self._job()
        shown = adapter.native_config_text(
            config, job / "dataset.toml", installation=self.install)
        self.assertEqual(shown, native.read_text(encoding="utf-8"))

    def test_export_matches_the_queued_config_when_given_the_installation(self):
        config, job, native = self._job()
        exported, _notices = adapter.export_parameters(config, installation=self.install)
        body = exported.splitlines()[1:]  # the export prepends one notice line
        self.assertEqual("\n".join(body) + "\n", native.read_text(encoding="utf-8"))

    def test_the_chain_values_the_form_does_not_model_are_pinned(self):
        _config, _job, native = self._job()
        values = tomllib.loads(native.read_text(encoding="utf-8"))
        self.assertEqual(values["down_init"], "weight_svd")
        self.assertTrue(values["use_repa"])
        self.assertEqual(values["learning_rate"], 3e-05)
        self.assertEqual(values["channel_scaling_alpha"], 0.5)
        self.assertEqual(values["discrete_flow_shift"], 1.0)

    def test_preview_without_the_installation_still_renders(self):
        # Backwards compatible fallback: the old two-argument call must keep working
        # (and is what the service falls back to until it passes the installation).
        config, job, _native = self._job()
        text = adapter.native_config_text(config, job / "dataset.toml")
        self.assertIn("learning_rate = 2e-05", text)  # adapter static default
        self.assertNotIn("down_init", text)

    def test_export_without_arguments_is_unchanged(self):
        config = job_config(self.root)
        text, notices = adapter.export_parameters(config)
        self.assertTrue(text.startswith("# "))
        self.assertEqual(len(notices), 1)


class VocabPackCompatibilityTests(unittest.TestCase):
    """M3: the engine default is stated explicitly instead of being silently lost."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.install = installation(engine_source(self.root / "engine"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_default_stays_off_but_says_why_and_where(self):
        caps = adapter.capabilities(
            SimpleNamespace(installation_id="x", source_path=self.install.source_path))
        spec = next(item for item in caps["params"] if item["key"] == "vocab_pack")
        self.assertEqual(spec["default"], "")
        # The form must carry the engine default and the exact relative prefix.
        self.assertIn("models/vocab_packs/anima_cjk_vocab_pack_jp_v1", spec["help"])
        self.assertIn("preprocess-config", spec["help"])
        self.assertIn("make preprocess-te", spec["help"])

    def test_enabling_it_with_caching_on_warns_about_the_cache_pass(self):
        config = {
            "architecture": "anima",
            "name": "probe",
            "base_model_path": str(self.root / "base.safetensors"),
            "dataset_path": str(make_dataset(self.root / "data")),
            "output_dir": str(self.root / "out"),
            "params": {
                "qwen3": "qwen3.safetensors",
                "vae": "vae.safetensors",
                "vocab_pack": "models/vocab_packs/anima_cjk_vocab_pack_jp_v1",
            },
        }
        issues = adapter.validate(self.install, config)
        warnings = [i for i in issues if i.field == "vocab_pack" and i.severity == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("preprocess-config", warnings[0].message)

    def test_live_encoding_needs_no_warning_because_no_cache_is_built(self):
        config = {
            "architecture": "anima",
            "name": "probe",
            "base_model_path": str(self.root / "base.safetensors"),
            "dataset_path": str(make_dataset(self.root / "data")),
            "output_dir": str(self.root / "out"),
            "params": {
                "qwen3": "qwen3.safetensors",
                "vae": "vae.safetensors",
                "use_vae_cache": False,
                "use_text_cache": False,
                "vocab_pack": "models/vocab_packs/anima_cjk_vocab_pack_jp_v1",
            },
        }
        issues = adapter.validate(self.install, config)
        self.assertEqual([i for i in issues if i.field == "vocab_pack"], [])

    def test_a_chosen_pack_still_reaches_the_config(self):
        config = job_config(self.root)
        config["params"]["vocab_pack"] = "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
        job = self.root / "job"
        native = adapter.write_native_config(self.install, config, job)
        values = tomllib.loads(native.read_text(encoding="utf-8"))
        self.assertEqual(values["vocab_pack"], "models/vocab_packs/anima_cjk_vocab_pack_jp_v1")


class UnsupportedParameterTests(unittest.TestCase):
    """M5: knobs the engine never consumes are marked, not silently dead."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.install = installation(engine_source(self.root / "engine"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dead_parameters_are_marked_unavailable_and_not_native(self):
        for key in DEAD_KEYS:
            spec = param(key)
            with self.subTest(key=key):
                self.assertTrue(spec.unsupported_reason, key)
                self.assertFalse(spec.native, key)

    def test_parameters_the_engine_does_read_stay_available(self):
        for key in ("metadata_title", "metadata_author", "metadata_description",
                    "metadata_license", "metadata_tags", "training_comment"):
            spec = param(key)
            with self.subTest(key=key):
                self.assertEqual(spec.unsupported_reason, "", key)
                self.assertTrue(spec.native, key)

    def test_dead_parameters_never_reach_the_generated_config(self):
        config = job_config(
            self.root,
            ema=True, ema_decay=0.999, ema_device="cpu", ema_use_feedback=True,
            self_attn_lr=1e-4, llm_adapter_lr=1e-5,
            metadata_usage_hint="hint", metadata_trigger_phrase="trigger",
            llm_adapter_path="adapter.safetensors",
        )
        job = self.root / "job"
        native = adapter.write_native_config(self.install, config, job)
        values = tomllib.loads(native.read_text(encoding="utf-8"))
        for key in DEAD_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, values)
        # The disabled field's default used to be written even when untouched.
        self.assertNotIn("ema_device", values)

    def test_capabilities_expose_the_reason_for_the_frontend(self):
        caps = adapter.capabilities(
            SimpleNamespace(installation_id="x", source_path=self.install.source_path))
        specs = {item["key"]: item for item in caps["params"]}
        for key in DEAD_KEYS:
            with self.subTest(key=key):
                self.assertFalse(specs[key]["native"], key)
                self.assertTrue(specs[key]["unsupported_reason"], key)


class MossNativeViewTests(unittest.TestCase):
    """M6: the native view is generated from the same flags the launcher uses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.install = installation(engine_source(self.root / "engine"))
        (self.root / "engine" / "scripts" / "moss").mkdir(parents=True)
        (self.root / "engine" / "scripts" / "moss" / "cache.py").write_text("", encoding="utf-8")
        (self.root / "engine" / "scripts" / "moss" / "train.py").write_text("", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _view_and_argv(self, **params):
        config = job_config(self.root, "moss_image", **params)
        view = adapter.native_config_text(config, None, installation=self.install)
        native = self.root / "job" / "config.native.toml"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_text(view, encoding="utf-8")
        commands = adapter.preview_commands(self.install, config, native)
        return view, commands

    @staticmethod
    def _view_flags(view):
        stages = []
        for line in view.splitlines():
            if not line or line.startswith("#"):
                continue
            stages.append(split_flags(line))
        return stages

    def test_the_view_lists_both_stages_and_their_path_flags(self):
        view, _commands = self._view_and_argv()
        self.assertIn("# 阶段 1/2：scripts/moss/cache.py", view)
        self.assertIn("# 阶段 2/2：scripts/moss/train.py", view)
        for flag in ("--src", "--out", "--cache", "--output", "--model base", "--model_dir"):
            with self.subTest(flag=flag):
                self.assertIn(flag, view)

    def test_the_view_flags_equal_the_launch_argv(self):
        view, commands = self._view_and_argv(moss_fit_report=True)
        view_stages = self._view_flags(view)
        # Drop the interpreter, "-u" and the script path from each command.
        argv_stages = [list(command[3:]) for command in commands]
        self.assertEqual(view_stages, argv_stages)

    def test_the_view_shows_the_explicit_no_forms(self):
        view, _commands = self._view_and_argv(
            moss_compile=False, moss_swap_h2d_only=False, moss_expandable_segments=False)
        flags = {token for stage in self._view_flags(view) for token in stage}
        self.assertIn("--no-compile", flags)
        self.assertIn("--no-swap_h2d_only", flags)
        self.assertIn("--no-expandable_segments", flags)
        self.assertNotIn("--compile", flags)
        self.assertNotIn("--swap_h2d_only", flags)

    def test_the_view_shows_the_on_forms(self):
        view, _commands = self._view_and_argv(
            moss_compile=True, moss_swap_h2d_only=True, moss_expandable_segments=True)
        flags = {token for stage in self._view_flags(view) for token in stage}
        self.assertIn("--compile", flags)
        self.assertIn("--swap_h2d_only", flags)
        self.assertIn("--expandable_segments", flags)

    def test_build_launch_reads_compile_from_the_staged_view_before_job_json(self):
        config = job_config(self.root, "moss_image", moss_compile=True)
        job = self.root / "job"
        native = adapter.write_native_config(self.install, config, job)
        self.assertFalse((job / "job.json").is_file())
        with patch.object(adapter_module, "_msvc_environment",
                          return_value={"INCLUDE": "patched"}) as mocked:
            launch = adapter.build_launch(self.install, native)
        self.assertEqual(launch.environment_overrides.get("INCLUDE"), "patched")
        # MOSS must never receive the Anima release-pause environment.
        self.assertNotIn("ANIMA_DAEMON_JOB_DIR", launch.environment_overrides)
        self.assertTrue(mocked.called)

    def test_build_launch_ignores_the_no_compile_form(self):
        config = job_config(self.root, "moss_image", moss_compile=False)
        job = self.root / "job"
        native = adapter.write_native_config(self.install, config, job)
        with patch.object(adapter_module, "_msvc_environment",
                          return_value={"INCLUDE": "patched"}) as mocked:
            launch = adapter.build_launch(self.install, native)
        self.assertNotIn("INCLUDE", launch.environment_overrides)
        self.assertFalse(mocked.called)


class OutputNameTests(unittest.TestCase):
    """The checkpoint stem is sanitized exactly like the output folder."""

    def test_reserved_characters_and_separators_are_replaced(self):
        self.assertEqual(adapter._output_name({"name": "a/b:c*?"}), "a_b_c__")
        self.assertEqual(adapter._output_name({"name": "..\\..\\escape"}), ".._.._escape")

    def test_quotes_are_left_verbatim_like_the_other_adapters(self):
        self.assertEqual(adapter._output_name({"output_name": '角色 "test"'}), '角色 "test"')

    def test_empty_names_fall_back(self):
        self.assertEqual(adapter._output_name({}), "lora")
        # Whitespace-only and trailing-dot names collapse to the fallback, not to a
        # stem Windows would reject.
        self.assertEqual(adapter._output_name({"name": "   "}), "lora")
        self.assertEqual(adapter._output_name({"name": "probe..."}), "probe")

    def test_the_generated_config_uses_the_sanitized_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            install = installation(engine_source(tmp / "engine"))
            config = job_config(tmp)
            config["name"] = "a/b:c*"
            config["output_name"] = ""
            job = tmp / "job"
            native = adapter.write_native_config(install, config, job)
            values = tomllib.loads(native.read_text(encoding="utf-8"))
            self.assertEqual(values["output_name"], "a_b_c_")
            self.assertNotIn("/", values["output_name"])


if __name__ == "__main__":
    unittest.main()
