"""Regression coverage for the Musubi Tuner audit fixes.

The engine is never imported: scripts are faked with empty files (the same way
``tests/test_parameter_capabilities.py`` does), so these tests run without the
engine environment and without a GPU.

Each test pins one defect that was reported by the audit:
* Z-Image must use ``networks.lora_zimage`` (networks.lora targets HunyuanVideo).
* Wan's ``--mixed_precision`` must follow the DiT checkpoint dtype, because the
  trainer asserts instead of converting.
* Parameter enums must stay inside what the current engine accepts
  (optimizers, dynamo backends, H3 sampling, vae_dtype per architecture).
* The cache stage must get the derived ``--i2v`` flag and the VAE dtype where the
  cache script accepts it.
* Checkpoint collection must not report Accelerate ``*-state`` weights as LoRAs,
  and must index the mp4/webm previews the video architectures write.
* P2 cleanup: timestep buckets need >=2, H3/HiDream get no generic VAE key, a
  custom cache directory cannot overlap the inputs/output/engine source, and the
  Krea 2 tokenizer redirect follows the repo id the engine source declares.
"""

import json
import struct
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters.musubi_tuner import krea2_cache_text_encoder as krea2_launcher
from adapters.musubi_tuner import precision
from adapters.musubi_tuner.adapter import MusubiTunerAdapter
from adapters.musubi_tuner.params import NETWORK_MODULES, PARAMS
from adapters.musubi_tuner.pipeline import PIPELINES

PARAMS_BY_KEY = {p.key: p for p in PARAMS}


def write_safetensors(path, dtype="F32"):
    """A structurally valid one-tensor safetensors file with the requested dtype."""
    header = json.dumps({"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    return Path(path)


class MusubiFixTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-musubi-fix-")
        self.root = Path(self.temp.name)
        self.adapter = MusubiTunerAdapter()
        self.model = write_safetensors(self.root / "model.safetensors")
        self.addCleanup(self.temp.cleanup)

    def installation(self, *scripts):
        source = self.root / "engine"
        package = source / "src" / "musubi_tuner"
        package.mkdir(parents=True, exist_ok=True)
        (source / "pyproject.toml").write_text('[project]\nname = "musubi-tuner"\n', encoding="utf-8")
        for name in scripts:
            (package / name).write_text("# entry\n", encoding="utf-8")
        return SimpleNamespace(source_path=source, python_executable=str(self.root / "python.exe"),
                               installation_id="test-musubi")

    def config(self, architecture, params, name="job", base_model=None):
        return dict(name=name, architecture=architecture, base_model_path=str(base_model or self.model),
                    output_dir=str(self.root / "output"), dataset_path=str(self.root / "images"),
                    params=params)

    def render(self, architecture, params):
        text = self.adapter._render_train(self.config(architecture, params))
        return tomllib.loads(text)

    def issues(self, installation, architecture, params):
        return list(self.adapter.validate(installation, self.config(architecture, params)))


class ParameterSurfaceTests(MusubiFixTestCase):
    def test_z_image_uses_its_own_network_module(self):
        # networks.lora targets HunyuanVideo blocks: a Z-Image run builds zero modules.
        self.assertEqual(NETWORK_MODULES["z_image"], "networks.lora_zimage")
        self.assertEqual(PIPELINES["z_image"].module, "networks.lora_zimage")
        values = self.render("z_image", {"vae": "vae.safetensors", "zimage_text_encoder": "te.safetensors"})
        self.assertEqual(values["network_module"], "networks.lora_zimage")

    def test_network_module_table_matches_every_pipeline(self):
        for architecture, pipeline in PIPELINES.items():
            with self.subTest(architecture=architecture):
                self.assertEqual(NETWORK_MODULES[architecture], pipeline.module)

    def test_optimizer_enum_only_lists_engine_supported_names(self):
        # trainer_base resolves only adamw / adamw8bit / adafactor itself; everything
        # else is imported by full path, which needs an extra package installed.
        self.assertEqual(PARAMS_BY_KEY["optimizer_type"].enum_values,
                         ["AdamW", "AdamW8bit", "Adafactor"])

    def test_dynamo_backend_enum_has_no_stale_values(self):
        values = PARAMS_BY_KEY["dynamo_backend"].enum_values
        self.assertNotIn("NVFUSER", values)  # absent from accelerate 1.15 DynamoBackend
        self.assertEqual(values, ["NO", "EAGER", "AOT_EAGER", "INDUCTOR", "ONNXRT",
                                  "TENSORRT", "TORCHXLA_TRACE_ONCE"])

    def test_h3_timestep_sampling_only_accepts_uniform(self):
        self.assertEqual(PARAMS_BY_KEY["h3_timestep_sampling"].enum_values, ["uniform"])

    def test_engine_rejected_sampling_values_are_not_rendered(self):
        # minimax_h3 raises on weighting_scheme != none and discrete_flow_shift != 1.0;
        # ideogram4 and hidream_o1 raise on weighting_scheme != none.
        for architecture, params in (
            ("minimax_h3", {"h3_video_vae": "vv", "h3_audio_vae": "av", "h3_text_encoder": "te"}),
            ("ideogram4", {"vae": "vae", "ideogram_text_encoder": "te"}),
            ("hidream_o1", {}),
        ):
            with self.subTest(architecture=architecture):
                values = self.render(architecture, {**params, "weighting_scheme": "logit_normal",
                                                    "discrete_flow_shift": 3.0})
                self.assertNotIn("weighting_scheme", values)
                if architecture == "minimax_h3":
                    self.assertNotIn("discrete_flow_shift", values)
                else:
                    self.assertEqual(values["discrete_flow_shift"], 3.0)
                # Each restricted architecture keeps its own engine-side default.
                self.assertEqual(values["timestep_sampling"], {
                    "minimax_h3": "uniform",
                    "ideogram4": "ideogram4_shift",
                    "hidream_o1": "sigma",
                }[architecture])

    def test_vae_dtype_is_scoped_to_architectures_that_honour_it(self):
        for architecture in ("wan", "flux2", "kandinsky5", "krea2"):
            with self.subTest(architecture=architecture):
                self.assertIn("vae_dtype", self.render(architecture, {"vae": "vae", "vae_dtype": "bfloat16"}))
        # FramePack asserts float16, Z-Image ignores the flag, H3 fixes its VAE dtypes.
        for architecture in ("framepack", "z_image", "minimax_h3", "hidream_o1"):
            with self.subTest(architecture=architecture):
                self.assertNotIn("vae_dtype", self.render(architecture, {"vae": "vae", "vae_dtype": "bfloat16"}))

    def test_lr_warmup_and_decay_accept_step_ratios(self):
        for key in ("lr_warmup_steps", "lr_decay_steps"):
            self.assertEqual(PARAMS_BY_KEY[key].type, "float")
        values = self.render("wan", {"vae": "vae", "wan_t5": "t5", "lr_warmup_steps": 0.05,
                                     "lr_decay_steps": 0.2})
        self.assertIsInstance(values["lr_warmup_steps"], float)
        self.assertAlmostEqual(values["lr_warmup_steps"], 0.05)
        self.assertAlmostEqual(values["lr_decay_steps"], 0.2)
        # An integral value must stay an int: the engine reads a float as "x train steps".
        values = self.render("wan", {"vae": "vae", "wan_t5": "t5", "lr_warmup_steps": 500.0})
        self.assertIsInstance(values["lr_warmup_steps"], int)
        self.assertEqual(values["lr_warmup_steps"], 500)

    def test_output_name_cannot_escape_the_output_directory(self):
        self.assertEqual(self.adapter._output_name({"name": "a/b:c*"}), "a_b_c_")
        self.assertEqual(self.adapter._output_name({"output_name": "角色 \"test\""}), "角色 \"test\"")
        self.assertEqual(self.adapter._output_name({}), "lora")
        config = self.config("wan", {"vae": "vae", "wan_t5": "t5"}, name="../../escape")
        self.assertEqual(tomllib.loads(self.adapter._render_train(config))["output_name"], ".._.._escape")

    def test_sample_prompt_help_no_longer_claims_a_count_option(self):
        help_text = PARAMS_BY_KEY["sample_prompts"].help
        self.assertIn("负面提示词", help_text)  # musubi --n is the negative prompt
        self.assertNotIn("--n、--w", help_text)


class ValidationTests(MusubiFixTestCase):
    def setUp(self):
        super().setUp()
        self.engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                        "wan_cache_text_encoder_outputs.py")

    def test_attention_choice_is_required_but_defaults_to_sdpa(self):
        self.assertEqual([i for i in self.issues(self.engine, "wan", {"vae": "vae", "wan_t5": "t5"})], [])
        fields = [i.field for i in self.issues(self.engine, "wan", {"vae": "vae", "wan_t5": "t5", "sdpa": False})]
        self.assertIn("sdpa", fields)
        # Any other implementation satisfies the engine.
        fields = [i.field for i in self.issues(
            self.engine, "wan", {"vae": "vae", "wan_t5": "t5", "sdpa": False, "flash_attn": True})]
        self.assertNotIn("sdpa", fields)

    def test_step_retention_requires_a_save_interval(self):
        fields = [i.field for i in self.issues(
            self.engine, "wan", {"vae": "vae", "wan_t5": "t5", "save_last_n_steps": 3,
                                 "save_every_n_steps": None})]
        self.assertIn("save_every_n_steps", fields)
        fields = [i.field for i in self.issues(
            self.engine, "wan", {"vae": "vae", "wan_t5": "t5", "save_last_n_steps_state": 3,
                                 "save_every_n_steps": None})]
        self.assertIn("save_every_n_steps", fields)

    def test_wan_i2v_requires_clip_but_wan22_does_not(self):
        fields = [i.field for i in self.issues(self.engine, "wan", {"vae": "vae", "wan_t5": "t5",
                                                                   "wan_task": "i2v-14B"})]
        self.assertIn("wan_clip", fields)
        fields = [i.field for i in self.issues(self.engine, "wan", {"vae": "vae", "wan_t5": "t5",
                                                                   "wan_task": "i2v-A14B"})]
        self.assertNotIn("wan_clip", fields)
        self.assertEqual([], [i for i in self.issues(
            self.engine, "wan", {"vae": "vae", "wan_t5": "t5", "wan_task": "i2v-14B",
                                 "wan_clip": "clip.safetensors"})])

    def test_hv15_i2v_requires_the_image_encoder(self):
        engine = self.installation("hv_1_5_train_network.py", "hv_1_5_cache_latents.py",
                                   "hv_1_5_cache_text_encoder_outputs.py")
        base = {"vae": "vae", "hv15_text_encoder": "te", "hv15_byt5": "byt5", "hv15_task": "i2v"}
        fields = [i.field for i in self.issues(engine, "hunyuan_video_15", base)]
        self.assertIn("hv15_image_encoder", fields)
        self.assertEqual([], [i for i in self.issues(
            engine, "hunyuan_video_15", {**base, "hv15_image_encoder": "siglip"})])


class WanPrecisionTests(MusubiFixTestCase):
    def test_fp16_checkpoint_forces_fp16_training_precision(self):
        fp16 = write_safetensors(self.root / "wan_fp16.safetensors", "F16")
        values = self.render("wan", {"vae": "vae", "wan_t5": "t5"})
        self.assertEqual(values["mixed_precision"], "bf16")
        text = self.adapter._render_train(self.config("wan", {"vae": "vae", "wan_t5": "t5"}, base_model=fp16))
        self.assertEqual(tomllib.loads(text)["mixed_precision"], "fp16")

    def test_explicit_no_is_kept_and_other_architectures_are_untouched(self):
        fp16 = write_safetensors(self.root / "wan_fp16.safetensors", "F16")
        values = tomllib.loads(self.adapter._render_train(self.config(
            "wan", {"vae": "vae", "wan_t5": "t5", "mixed_precision": "no"}, base_model=fp16)))
        self.assertEqual(values["mixed_precision"], "no")
        # An untouched/cleared enum falls back to the declared default instead of
        # omitting the key: the Wan trainer also rejects a missing precision.
        values = tomllib.loads(self.adapter._render_train(self.config(
            "wan", {"vae": "vae", "wan_t5": "t5", "mixed_precision": None}, base_model=fp16)))
        self.assertEqual(values["mixed_precision"], "fp16")
        values = tomllib.loads(self.adapter._render_train(self.config(
            "krea2", {"vae": "vae"}, base_model=fp16)))
        self.assertEqual(values["mixed_precision"], "bf16")

    def test_correction_is_reported_before_submission(self):
        engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                   "wan_cache_text_encoder_outputs.py")
        fp16 = write_safetensors(self.root / "wan_fp16.safetensors", "F16")
        issues = list(self.adapter.validate(engine, self.config(
            "wan", {"vae": "vae", "wan_t5": "t5"}, base_model=fp16)))
        warnings = [i for i in issues if i.field == "mixed_precision"]
        self.assertEqual([i.severity for i in warnings], ["warning"])
        self.assertIn("fp16", warnings[0].message)
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_probe_is_optional_and_reads_only_the_header(self):
        self.assertIsNone(precision.weight_dtype(self.root / "missing.safetensors"))
        broken = self.root / "broken.safetensors"
        broken.write_bytes(b"\x00" * 4)
        self.assertIsNone(precision.weight_dtype(broken))
        self.assertIsNone(precision.weight_dtype(None))
        fp8 = write_safetensors(self.root / "fp8.safetensors", "F8_E4M3")
        self.assertEqual(precision.weight_dtype(fp8), "fp8")
        # An unknown/other dtype never overrides the user's choice.
        self.assertEqual(precision.wan_mixed_precision("bf16", fp8), "bf16")
        self.assertEqual(precision.wan_mixed_precision(None, fp8), None)


class CacheStageTests(MusubiFixTestCase):
    def stages(self, engine, architecture, params):
        return self.adapter._stages(engine, self.config(architecture, params), self.root / "job")

    @staticmethod
    def flag(argv, name):
        return argv[argv.index(name) + 1] if name in argv else None

    def test_wan_i2v_flag_is_derived_from_the_task(self):
        engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                   "wan_cache_text_encoder_outputs.py")
        latents, _, _ = self.stages(engine, "wan", {"vae": "vae", "wan_t5": "t5",
                                                    "wan_task": "i2v-14B", "wan_clip": "clip"})
        self.assertIn("--i2v", latents)
        self.assertEqual(self.flag(latents, "--clip"), "clip")
        latents, _, _ = self.stages(engine, "wan", {"vae": "vae", "wan_t5": "t5"})
        self.assertNotIn("--i2v", latents)

    def test_hv15_i2v_flag_is_derived_from_the_task(self):
        engine = self.installation("hv_1_5_train_network.py", "hv_1_5_cache_latents.py",
                                   "hv_1_5_cache_text_encoder_outputs.py")
        latents, _, _ = self.stages(engine, "hunyuan_video_15", {
            "vae": "vae", "hv15_text_encoder": "te", "hv15_byt5": "byt5",
            "hv15_task": "i2v", "hv15_image_encoder": "siglip"})
        self.assertIn("--i2v", latents)
        self.assertEqual(self.flag(latents, "--image_encoder"), "siglip")

    def test_vae_dtype_reaches_cache_stages_that_accept_it(self):
        engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                   "wan_cache_text_encoder_outputs.py")
        latents, _, _ = self.stages(engine, "wan", {"vae": "vae", "wan_t5": "t5", "vae_dtype": "float32"})
        self.assertEqual(self.flag(latents, "--vae_dtype"), "float32")

    def test_vae_dtype_is_not_passed_to_cache_scripts_that_reject_it(self):
        engine = self.installation("krea2_train_network.py", "krea2_cache_latents.py",
                                   "krea2_cache_text_encoder_outputs.py")
        latents, text_encoder, _ = self.stages(engine, "krea2", {
            "vae": "vae", "krea2_text_encoder": "te", "krea2_tokenizer": "tokenizer",
            "vae_dtype": "float32"})
        self.assertNotIn("--vae_dtype", latents)
        self.assertNotIn("--vae_dtype", text_encoder)


class ArtifactTests(MusubiFixTestCase):
    def test_state_weights_are_not_reported_as_loras(self):
        out = self.root / "output"
        (out / "sample").mkdir(parents=True)
        write_safetensors(out / "job.safetensors")
        write_safetensors(out / "job-step00000200.safetensors")
        state = out / "job-step00000200-state"
        state.mkdir()
        write_safetensors(state / "model.safetensors")
        write_safetensors(out / "job-000002.safetensors")
        job = self.root / "job"
        job.mkdir()
        (job / "job.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")

        artifacts = {a.path.relative_to(out).as_posix(): a for a in self.adapter.collect_artifacts(job)}
        self.assertNotIn("job-step00000200-state/model.safetensors", artifacts)
        self.assertEqual(artifacts["job-step00000200.safetensors"].step, 200)
        self.assertIsNone(artifacts["job-000002.safetensors"].step)
        self.assertIsNone(artifacts["job.safetensors"].step)

    def test_video_previews_are_collected(self):
        out = self.root / "output"
        (out / "sample").mkdir(parents=True)
        (out / "sample" / "0001-00000200-0.mp4").write_bytes(b"")
        (out / "sample" / "0001-00000200-1.webm").write_bytes(b"")
        (out / "sample" / "0001-00000200-2.png").write_bytes(b"")
        job = self.root / "job"
        job.mkdir()
        (job / "job.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")

        kinds = {a.path.name: a.kind for a in self.adapter.collect_artifacts(job)}
        self.assertEqual(kinds["0001-00000200-0.mp4"], "video")
        self.assertEqual(kinds["0001-00000200-1.webm"], "video")
        self.assertEqual(kinds["0001-00000200-2.png"], "sample")


class TimestepBucketTests(MusubiFixTestCase):
    def test_timestep_buckets_require_at_least_two(self):
        # The engine only enables stratified sampling for values > 1, so 1 is a
        # silent no-op that the form must not offer.
        self.assertEqual(PARAMS_BY_KEY["num_timestep_buckets"].min, 2)
        engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                   "wan_cache_text_encoder_outputs.py")
        fields = [i.field for i in self.issues(engine, "wan", {"vae": "vae", "wan_t5": "t5",
                                                               "num_timestep_buckets": 1})]
        self.assertIn("num_timestep_buckets", fields)
        self.assertEqual([], [i for i in self.issues(engine, "wan", {"vae": "vae", "wan_t5": "t5",
                                                                     "num_timestep_buckets": 4})])


class UnusedVaeTests(MusubiFixTestCase):
    def test_generic_vae_is_not_offered_for_h3_or_hidream(self):
        vae = PARAMS_BY_KEY["vae"]
        self.assertFalse(vae.applies_to("minimax_h3"))
        self.assertFalse(vae.applies_to("hidream_o1"))
        for architecture in ("wan", "hunyuan_video", "krea2"):
            with self.subTest(architecture=architecture):
                self.assertTrue(vae.applies_to(architecture))

    def test_stale_vae_value_is_not_rendered_and_is_reported(self):
        values = self.render("minimax_h3", {"vae": "v.safetensors", "h3_video_vae": "vv",
                                            "h3_audio_vae": "av", "h3_text_encoder": "te"})
        self.assertNotIn("vae", values)
        self.assertEqual(values["video_vae"], "vv")
        engine = self.installation("minimax_h3_train_network.py", "minimax_h3_cache_latents.py",
                                   "minimax_h3_cache_text_encoder_outputs.py")
        issues = self.issues(engine, "minimax_h3", {"vae": "v.safetensors", "h3_video_vae": "vv",
                                                    "h3_audio_vae": "av", "h3_text_encoder": "te"})
        warnings = [i for i in issues if i.severity == "warning" and i.field == "vae"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual([i for i in issues if i.severity == "error"], [])


class CacheDirectoryTests(MusubiFixTestCase):
    def setUp(self):
        super().setUp()
        self.engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                        "wan_cache_text_encoder_outputs.py")
        # config() points dataset_path at <root>/images and output at <root>/output,
        # while the fake engine source is <root>/engine.
        (self.root / "images").mkdir(exist_ok=True)

    def cache_fields(self, cache_directory):
        return [i.field for i in self.issues(self.engine, "wan", {
            "vae": "vae", "wan_t5": "t5", "cache_directory": str(cache_directory)})]

    def test_cache_inside_dataset_output_or_engine_is_rejected(self):
        for cache in (self.root / "images" / "cache",       # musubi writes *.safetensors here
                      self.root / "output" / "cache",       # would be collected as checkpoints
                      self.root / "engine" / "cache"):      # would modify the cloned engine
            with self.subTest(cache=cache):
                self.assertIn("cache_directory", self.cache_fields(cache))

    def test_cache_that_contains_the_inputs_is_rejected(self):
        # A cache directory that is an ancestor of the dataset/output/engine would
        # intermix job outputs with cache files.
        self.assertIn("cache_directory", self.cache_fields(self.root))

    def test_drive_root_is_rejected(self):
        issues = self.issues(self.engine, "wan", {
            "vae": "vae", "wan_t5": "t5", "cache_directory": str(Path(self.root.anchor))})
        self.assertTrue(any("磁盘根目录" in i.message for i in issues))

    def test_a_sibling_cache_directory_is_allowed(self):
        self.assertNotIn("cache_directory", self.cache_fields(self.root / "cache"))

    def test_default_cache_directory_needs_no_validation(self):
        self.assertNotIn("cache_directory", self.cache_fields(""))


class Krea2TokenizerRepoTests(MusubiFixTestCase):
    def encoder_source(self):
        path = self.root / "engine" / "src" / "musubi_tuner" / "krea2" / "krea2_encoder.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_repo_id_is_read_from_the_engine_source(self):
        source = self.encoder_source()
        source.write_text('QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"\n', encoding="utf-8")
        self.assertEqual(krea2_launcher.tokenizer_repo(self.root / "engine"), "Qwen/Qwen3-VL-4B-Instruct")
        source.write_text('    QWEN3_VL_4B_INSTRUCT_REPO_ID = "Other/Qwen3-VL-9B"\n', encoding="utf-8")
        self.assertEqual(krea2_launcher.tokenizer_repo(self.root / "engine"), "Other/Qwen3-VL-9B")

    def test_unknown_source_returns_none_so_the_launcher_can_warn(self):
        self.assertIsNone(krea2_launcher.tokenizer_repo(self.root / "missing"))
        source = self.encoder_source()
        source.write_text("# renamed upstream\n", encoding="utf-8")
        self.assertIsNone(krea2_launcher.tokenizer_repo(self.root / "engine"))

    def test_redirect_only_covers_the_discovered_repo(self):
        calls = []

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, name, *args, **kwargs):
                calls.append((cls.__name__, name))
                return name

        class FakeAuto(FakeTokenizer):
            pass

        fake = SimpleNamespace(AutoTokenizer=FakeAuto, Qwen2TokenizerFast=FakeTokenizer)
        with patch.dict(sys.modules, {"transformers": fake}):
            krea2_launcher.patch_tokenizer("D:/local-tokenizer", "Repo/Discovered")
            FakeAuto.from_pretrained("Repo/Discovered")
            FakeTokenizer.from_pretrained("Repo/Discovered")
            FakeAuto.from_pretrained("Repo/Unrelated")

        self.assertEqual(calls, [("FakeAuto", "D:/local-tokenizer"),
                                 ("FakeTokenizer", "D:/local-tokenizer"),
                                 ("FakeAuto", "Repo/Unrelated")])


class EngineCrashGuardTests(MusubiFixTestCase):
    def setUp(self):
        super().setUp()
        self.engine = self.installation("wan_train_network.py", "wan_cache_latents.py",
                                        "wan_cache_text_encoder_outputs.py")
        self.base = {"vae": "vae", "wan_t5": "t5"}

    def fields(self, params):
        return [i.field for i in self.issues(self.engine, "wan", {**self.base, **params})]

    def test_list_arguments_must_be_key_value_pairs(self):
        for key in ("optimizer_args", "lr_scheduler_args", "network_args"):
            with self.subTest(key=key):
                self.assertIn(key, self.fields({key: "weight_decay"}))       # no "="
                self.assertIn(key, self.fields({key: "a=b=c"}))              # two "="
        self.assertEqual([], self.fields({"optimizer_args": "betas=0.9,0.99\nweight_decay=0.01"}))

    def test_dim_from_weights_requires_weights(self):
        self.assertIn("dim_from_weights", self.fields({"dim_from_weights": True}))
        self.assertEqual([], self.fields({"dim_from_weights": True, "network_weights": "w.safetensors"}))

    def test_inverted_timestep_range_is_rejected(self):
        self.assertIn("min_timestep", self.fields({"min_timestep": 800, "max_timestep": 200}))
        self.assertEqual([], self.fields({"min_timestep": 200, "max_timestep": 800}))

    def test_wan_high_noise_block_swap_and_boundary(self):
        high_noise = {"wan_dit_high_noise": "hi.safetensors", "blocks_to_swap": 8,
                      "wan_offload_inactive_dit": True}
        fields = self.fields(high_noise)
        self.assertIn("wan_offload_inactive_dit", fields)
        self.assertIn("wan_timestep_boundary", fields)   # non-A14B task has no default
        fields = self.fields({**high_noise, "wan_timestep_boundary": 900})
        self.assertIn("wan_offload_inactive_dit", fields)
        self.assertNotIn("wan_timestep_boundary", fields)
        fields = self.fields({**high_noise, "wan_task": "t2v-A14B"})
        self.assertNotIn("wan_timestep_boundary", fields)  # A14B configs define it

    def test_preview_sampling_without_prompts_warns(self):
        warnings = [i for i in self.issues(self.engine, "wan", {
            **self.base, "sample_every_n_steps": 100}) if i.severity == "warning"]
        self.assertIn("sample_prompts", [i.field for i in warnings])
        self.assertEqual([], [i for i in self.issues(self.engine, "wan", {
            **self.base, "sample_every_n_steps": 100, "sample_prompts": "a prompt"})])

    def test_i2v_cache_flag_conflicting_with_the_task_warns(self):
        warnings = [i for i in self.issues(self.engine, "wan", {**self.base, "wan_i2v": True})
                    if i.severity == "warning"]
        self.assertIn("wan_i2v", [i.field for i in warnings])
        hv15 = self.installation("hv_1_5_train_network.py", "hv_1_5_cache_latents.py",
                                 "hv_1_5_cache_text_encoder_outputs.py")
        issues = self.issues(hv15, "hunyuan_video_15", {
            "vae": "vae", "hv15_text_encoder": "te", "hv15_byt5": "byt5", "hv15_i2v": True})
        self.assertIn("hv15_i2v", [i.field for i in issues if i.severity == "warning"])


if __name__ == "__main__":
    unittest.main()
