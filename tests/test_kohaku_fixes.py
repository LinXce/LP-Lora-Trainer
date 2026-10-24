"""Regression tests for the KohakuMossTrainer adapter fixes.

Every case here reproduces a defect found in the engine/adaptor audit: a knob the
shared ``config.py`` must not carry, a stage list that followed the wrong cache
knobs, a weights layout the engine supports but the adapter refused, a combination
the engine rejects only after the cache stages, or an environment rule that used to
let a broken environment look "ready".
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.kohaku_moss_trainer.adapter import DEPENDENCY_RULES, KohakuMossTrainerAdapter
from adapters.kohaku_moss_trainer.params import PARAMS

ENGINE = Path("engine/KohakuMossTrainer")


def defaults():
    """Every parameter at its own default (the UI fills the form this way)."""
    return {spec.key: spec.default for spec in PARAMS}


def param(key):
    return next(spec for spec in PARAMS if spec.key == key)


class AdapterCase(unittest.TestCase):
    """Fixtures: the four weights layouts the engine reads, and a rendered config."""

    def setUp(self):
        self.adapter = KohakuMossTrainerAdapter()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # A minimal source tree: the architecture is declared by its entry script's
        # presence, so a valid case must not also report "no training entry".
        scripts = self.root / "engine" / "scripts"
        (scripts / "train").mkdir(parents=True)
        (scripts / "data").mkdir(parents=True)
        for name in ("train/moss.py", "data/cache_text.py", "data/cache_latents.py"):
            (scripts / name).write_text("", encoding="utf-8")
        self.installation = SimpleNamespace(
            source_path=str(self.root / "engine"),
            python_executable=str(self.root / "python.exe"),
        )

    def tearDown(self):
        self.temp.cleanup()

    # -- weights layouts (kohakumoss.weights.layout) --

    def comfy(self):
        root = self.root / "comfy"
        for sub, name in (("diffusion_models", "moss_image_base_int8_convrot"),
                          ("text_encoders", "qwen3.5_9b_int8_convrot"),
                          ("vae", "flux2-vae_small")):
            (root / sub).mkdir(parents=True, exist_ok=True)
            (root / sub / f"{name}.safetensors").write_bytes(b"x")
        return root

    def hf(self):
        root = self.root / "hf"
        for sub in ("transformer", "text_encoder", "vae"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        return root

    def variants(self):
        root = self.root / "variants"
        for sub in ("transformer", "text_encoder", "vae"):
            (root / "base" / sub).mkdir(parents=True, exist_ok=True)
        return root

    def single(self):
        path = self.root / "models" / "dit.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    # -- config rendering --

    def config(self, params=None, **extra):
        values = defaults()
        values.update(params or {})
        cfg = {"architecture": "moss_image", "output_dir": str(self.root / "out"),
               "name": "probe", "output_name": "probe",
               "dataset_path": str(self.root / "dataset")}
        cfg.update(extra)
        cfg["params"] = {key: value for key, value in values.items() if value is not None}
        return cfg

    def render(self, params=None, **extra):
        return self.adapter._render(self.config(params, **extra))

    def stages(self, params=None):
        return self.adapter.preview_commands(
            self.installation, self.config(params), self.root / "config.py")

    @staticmethod
    def sets(stage):
        """The ``--set KEY=VALUE`` pairs of one command as ``{key: value}``."""
        items = list(stage)
        out = {}
        for index, item in enumerate(items[:-1]):
            if item == "--set":
                key, _, value = items[index + 1].partition("=")
                out[key] = value
        return out

    def issues(self, config):
        return list(self.adapter.validate(self.installation, config))

    def errors(self, config):
        return [issue for issue in self.issues(config) if issue.severity == "error"]

    def fields(self, config, severity=None):
        return [issue.field for issue in self.issues(config)
                if severity is None or issue.severity == severity]

    def assignments(self, params=None, **extra):
        """The rendered config's top-level assignments (a duplicate keeps the last)."""
        out = {}
        for node in ast.parse(self.render(params, **extra)).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            out[target.id] = ast.literal_eval(node.value)
                        except ValueError:
                            out[target.id] = ast.unparse(node.value)
        return out


class ConfigRenderTests(AdapterCase):
    def test_component_paths_never_reach_the_shared_config(self):
        """TEXT_ENCODER / VAE mean something else in cache_text / cache_latents, and
        kogine applies a config to every stage by name."""
        for weights in (self.comfy(), self.hf(), self.variants(), self.single()):
            with self.subTest(layout=weights.name):
                text = self.render({"WEIGHTS": str(weights),
                                    "TEXT_ENCODER": str(self.root / "te"),
                                    "VAE": str(self.root / "vae")})
                self.assertNotIn("TEXT_ENCODER =", text)
                self.assertNotIn("\nVAE =", text)

    def test_train_stage_carries_the_component_paths(self):
        text_encoder = self.root / "custom_te"
        vae = self.root / "custom_vae"
        train = self.stages({"WEIGHTS": str(self.hf()),
                             "TEXT_ENCODER": str(text_encoder),
                             "VAE": str(vae)})[-1]
        sets = self.sets(train)
        self.assertEqual(sets.get("TEXT_ENCODER"), str(text_encoder))
        self.assertEqual(sets.get("VAE"), str(vae))

    def test_train_stage_omits_components_left_empty(self):
        train = self.stages({"WEIGHTS": str(self.hf())})[-1]
        self.assertNotIn("TEXT_ENCODER", self.sets(train))
        self.assertNotIn("VAE", self.sets(train))

    def test_comfy_file_names_are_passed_through_to_the_trainer(self):
        train = self.stages({"WEIGHTS": str(self.comfy()),
                             "TEXT_ENCODER": "qwen3.5_9b_bf16"})[-1]
        self.assertEqual(self.sets(train).get("TEXT_ENCODER"), "qwen3.5_9b_bf16")

    def test_caption_fields_fall_back_to_their_parameter_defaults(self):
        """An older/partial draft must keep the engine's `keep` (artist), not lose it."""
        config = self.config({"WEIGHTS": str(self.hf())})
        del config["params"]["CAPTION_KEEP"]
        caption = self.adapter._caption_config(config)
        self.assertEqual(caption["keep"], ["artist"])
        self.assertIs(caption["shuffle"], True)

    def test_numeric_step_release_renders_an_int(self):
        text = self.render({"WEIGHTS": str(self.hf()), "STEP_RELEASE": "30"})
        self.assertIn("STEP_RELEASE = 30", text)
        self.assertNotIn("'30'", text)

    def test_python_literal_betas_are_accepted(self):
        config = self.config({"WEIGHTS": str(self.hf()), "BETAS": "(0.9, 0.95)"})
        self.assertIn("BETAS = (0.9, 0.95)", self.adapter._render(config))
        self.assertNotIn("BETAS", self.fields(config))

    def test_malformed_container_knob_is_reported(self):
        config = self.config({"WEIGHTS": str(self.hf()), "KERNELS": "attention_varlen=triton"})
        self.assertIn("KERNELS", self.fields(config))

    def test_exported_parameters_carry_the_component_sets(self):
        """A hand-run exported config must still receive the paths it cannot hold."""
        text, notices = self.adapter.export_parameters(self.config({
            "WEIGHTS": str(self.hf()),
            "TEXT_ENCODER": str(self.root / "te"),
            "VAE": str(self.root / "vae")}))
        self.assertIn("--set TEXT_ENCODER=", text)
        self.assertIn("--set VAE=", text)
        self.assertNotIn("\nTEXT_ENCODER =", text)
        self.assertIn("--set", notices[0])

    def test_config_only_carries_engine_knobs(self):
        """Nothing the trainer would reject as an unknown knob reaches config.py."""
        knobs = ENGINE / "src" / "kohakumoss" / "training" / "knobs.py"
        if not knobs.is_file():
            self.skipTest("引擎源码不存在")
        known = set()
        for node in ast.parse(knobs.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                    getattr(target, "id", "") == "KNOWN" for target in node.targets):
                for piece in ast.walk(node.value):
                    if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                        known.update(piece.value.split())
        tree = ast.parse(self.render({"WEIGHTS": str(self.comfy())}))
        emitted = {target.id for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
        self.assertEqual(sorted(emitted - known), [])


class StageTests(AdapterCase):
    def test_cache_stages_follow_the_cache_knobs(self):
        cases = {
            ("offline", "offline"): 3,
            ("off", "off"): 1,
            ("online", "off"): 1,
            ("off", "online"): 1,
            ("online", "online"): 1,
        }
        for (text_cache, latent_cache), expected in cases.items():
            with self.subTest(text=text_cache, latent=latent_cache):
                stages = self.stages({"WEIGHTS": str(self.hf()),
                                      "TEXT_CACHE": text_cache,
                                      "LATENT_CACHE": latent_cache})
                self.assertEqual(len(stages), expected)
                scripts = [Path(part).name for part in stages[-1] if part.endswith(".py")]
                self.assertTrue(scripts and scripts[0] == "moss.py", stages[-1])

    def test_cache_stages_run_text_then_latents_then_training(self):
        stages = self.stages({"WEIGHTS": str(self.hf())})
        scripts = [next(part for part in stage if part.endswith(".py")) for stage in stages]
        self.assertEqual([Path(script).name for script in scripts],
                         ["cache_text.py", "cache_latents.py", "moss.py"])

    def test_all_stages_share_one_config(self):
        configs = {stage[list(stage).index("--config") + 1]
                   for stage in self.stages({"WEIGHTS": str(self.hf())})}
        self.assertEqual(len(configs), 1)

    def test_hf_layout_derives_the_component_paths(self):
        weights = self.hf()
        stages = self.stages({"WEIGHTS": str(weights)})
        text = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        latent = ast.literal_eval(self.sets(stages[1])["IMAGE_ENCODER_KWARGS"])
        self.assertEqual(Path(text["path"]), weights / "text_encoder")
        self.assertEqual(Path(latent["path"]), weights / "vae")

    def test_variant_root_derives_the_component_paths(self):
        weights = self.variants()
        stages = self.stages({"WEIGHTS": str(weights)})
        text = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        self.assertEqual(Path(text["path"]), weights / "base" / "text_encoder")

    def test_comfy_layout_resolves_file_names_and_defaults(self):
        weights = self.comfy()
        stages = self.stages({"WEIGHTS": str(weights),
                              "TEXT_ENCODER": "qwen3.5_9b_bf16",
                              "VAE": "flux2-vae"})
        text = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        latent = ast.literal_eval(self.sets(stages[1])["IMAGE_ENCODER_KWARGS"])
        self.assertEqual(Path(text["path"]),
                         weights / "text_encoders" / "qwen3.5_9b_bf16.safetensors")
        self.assertEqual(Path(latent["path"]), weights / "vae" / "flux2-vae.safetensors")

    def test_comfy_layout_falls_back_to_the_engine_defaults(self):
        weights = self.comfy()
        stages = self.stages({"WEIGHTS": str(weights), "VARIANT": "base"})
        text = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        self.assertEqual(Path(text["path"]),
                         weights / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors")

    def test_text_device_cpu_moves_the_cache_pass_to_the_host(self):
        stages = self.stages({"WEIGHTS": str(self.hf()), "TEXT_DEVICE": "cpu"})
        kwargs = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["plan"].get("memory"), "cpu")

    def test_text_device_defaults_to_cuda(self):
        stages = self.stages({"WEIGHTS": str(self.hf())})
        kwargs = ast.literal_eval(self.sets(stages[0])["TEXT_ENCODER_KWARGS"])
        self.assertEqual(kwargs["device"], "cuda")


class WeightsLayoutTests(AdapterCase):
    def test_supported_layouts_are_accepted(self):
        for layout in (self.comfy(), self.hf(), self.variants()):
            with self.subTest(layout=layout.name):
                self.assertNotIn("WEIGHTS", self.fields(self.config({"WEIGHTS": str(layout)})))

    def test_missing_path_is_rejected(self):
        self.assertIn("WEIGHTS", self.fields(self.config({"WEIGHTS": str(self.root / "nope")})))

    def test_unrecognised_directory_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        issues = self.issues(self.config({"WEIGHTS": str(empty)}))
        self.assertIn("WEIGHTS", [issue.field for issue in issues])
        self.assertIn("ComfyUI", " ".join(issue.message for issue in issues))

    def test_comfy_folder_missing_a_component_is_reported(self):
        weights = self.comfy()
        (weights / "vae" / "flux2-vae_small.safetensors").unlink()
        self.assertIn("VAE", self.fields(self.config({"WEIGHTS": str(weights)})))

    def test_comfy_folder_missing_the_dit_is_reported(self):
        weights = self.comfy()
        (weights / "diffusion_models" / "moss_image_base_int8_convrot.safetensors").unlink()
        self.assertIn("WEIGHTS", self.fields(self.config({"WEIGHTS": str(weights)})))

    def test_variant_root_missing_a_component_is_reported(self):
        weights = self.variants()
        (weights / "base" / "text_encoder").rmdir()
        self.assertIn("TEXT_ENCODER", self.fields(self.config({"WEIGHTS": str(weights)})))

    def test_single_file_needs_components_only_when_the_run_reads_them(self):
        weights = self.single()
        offline = self.config({"WEIGHTS": str(weights)})
        self.assertEqual(sorted({"TEXT_ENCODER", "VAE"} & set(self.fields(offline))),
                         ["TEXT_ENCODER", "VAE"])
        no_cache = self.config({"WEIGHTS": str(weights),
                                "TEXT_CACHE": "off", "LATENT_CACHE": "off",
                                "SAMPLE_EVERY": 0})
        self.assertEqual(sorted({"TEXT_ENCODER", "VAE"} & set(self.fields(no_cache))), [])
        sampling = self.config({"WEIGHTS": str(weights),
                                "TEXT_CACHE": "off", "LATENT_CACHE": "off",
                                "SAMPLE_EVERY": 10,
                                "SAMPLE_PROMPTS": "['a cat']"})
        self.assertIn("TEXT_ENCODER", self.fields(sampling))
        self.assertIn("VAE", self.fields(sampling))

    def test_single_file_component_must_exist(self):
        weights = self.single()
        missing = self.config({"WEIGHTS": str(weights),
                               "TEXT_ENCODER": str(self.root / "nope"),
                               "VAE": str(self.root / "nope2")})
        self.assertIn("TEXT_ENCODER", self.fields(missing))
        self.assertIn("VAE", self.fields(missing))


class PreflightTests(AdapterCase):
    def config_for(self, **values):
        params = {"WEIGHTS": str(self.hf())}
        params.update(values)
        return self.config(params)

    def test_offline_latents_reject_random_crop(self):
        self.assertIn("CROP", self.fields(self.config_for(CROP="random")))

    def test_offline_latents_reject_shifted_square(self):
        self.assertIn("CROP", self.fields(self.config_for(RESOLUTION_MODE="shifted_square")))

    def test_random_crop_is_fine_with_online_latents(self):
        config = self.config_for(CROP="random", LATENT_CACHE="online")
        self.assertNotIn("CROP", self.fields(config))

    def test_exact_token_budget_needs_the_offline_text_cache(self):
        self.assertIn("TOKEN_BUDGET", self.fields(self.config_for(
            TOKEN_BUDGET="exact", TEXT_CACHE="online")))

    def test_size_step_must_be_a_latent_multiple(self):
        self.assertIn("SIZE_STEP", self.fields(self.config_for(SIZE_STEP=12)))
        self.assertNotIn("SIZE_STEP", self.fields(self.config_for(SIZE_STEP=32)))

    def test_pipeline_requires_its_data_shape(self):
        fields = self.fields(self.config_for(PARALLEL="pipeline"))
        self.assertIn("MAX_SAMPLES", fields)
        filled = self.fields(self.config_for(PARALLEL="pipeline", MAX_SAMPLES=4,
                                             TOKEN_BUDGET="exact"))
        self.assertEqual(filled, [])

    def test_sampling_without_any_prompt_is_rejected(self):
        self.assertIn("SAMPLE_PROMPTS", self.fields(self.config_for(SAMPLE_EVERY=10)))

    def test_sampling_prompt_falls_back_to_the_dataset_caption(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "a.png").write_bytes(b"x")
        (dataset / "a.txt").write_text("1girl, solo", encoding="utf-8")
        config = self.config_for(SAMPLE_EVERY=10)
        config["dataset_path"] = str(dataset)
        self.assertNotIn("SAMPLE_PROMPTS", self.fields(config))
        self.assertIn("1girl, solo", self.adapter._render(config))

    def test_dora_rejects_input_dropout(self):
        self.assertIn("DROPOUT", self.fields(self.config_for(
            DORA=True, DROPOUT='{"input": 0.1, "rank": 0.0, "module": 0.0}')))

    def test_background_and_resample_are_checked(self):
        self.assertIn("BACKGROUND", self.fields(self.config_for(BACKGROUND="transparent")))
        self.assertIn("RESAMPLE", self.fields(self.config_for(RESAMPLE="nearest")))

    def test_step_release_values(self):
        self.assertIn("STEP_RELEASE", self.fields(self.config_for(STEP_RELEASE="every")))
        self.assertNotIn("STEP_RELEASE", self.fields(self.config_for(STEP_RELEASE="30")))

    def test_max_text_tokens_and_sample_grid(self):
        self.assertIn("MAX_TOKENS", self.fields(self.config_for(MAX_TOKENS=1)))
        self.assertIn("SAMPLE_SIZE", self.fields(self.config_for(SAMPLE_SIZE="(1000, 1000)")))


class DependencyTests(AdapterCase):
    def test_lycoris_below_the_engine_minimum_is_an_error(self):
        issues = KohakuMossTrainerAdapter.dependency_issues(
            {"lycoris-lora": "4.1.2", "torch": "2.14.1+cu130"})
        self.assertEqual([issue.field for issue in issues], ["lycoris-lora"])
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("4.1.3", issues[0].message)

    def test_versions_inside_the_rules_pass(self):
        self.assertEqual(KohakuMossTrainerAdapter.dependency_issues(
            {"lycoris-lora": "4.1.3", "torch": "2.14.1",
             "transformers": "5.20.1", "diffusers": "0.41.0"}), ())

    def test_torch_outside_the_pin_is_reported(self):
        self.assertEqual(
            [issue.field for issue in KohakuMossTrainerAdapter.dependency_issues(
                {"torch": "2.12.0"})], ["torch"])
        self.assertEqual(
            [issue.field for issue in KohakuMossTrainerAdapter.dependency_issues(
                {"torch": "2.15.0"})], ["torch"])

    def test_only_reported_versions_are_judged(self):
        """A partial diagnosis must not invent a missing distribution."""
        self.assertEqual(KohakuMossTrainerAdapter.dependency_issues({"torch": "2.14.1"}), ())

    def test_a_complete_set_reports_the_unreported_ones(self):
        fields = [issue.field for issue in KohakuMossTrainerAdapter.dependency_issues(
            {"torch": "2.14.1"}, require_all=True)]
        self.assertIn("lycoris-lora", fields)
        self.assertIn("transformers", fields)

    def test_unparsable_version_is_not_assumed_ok(self):
        issues = KohakuMossTrainerAdapter.dependency_issues({"lycoris-lora": "unknown"})
        self.assertEqual([issue.field for issue in issues], ["lycoris-lora"])

    def test_environment_issues_read_the_record(self):
        record = SimpleNamespace(torch_version="2.12.0")
        self.assertEqual([issue.field for issue in
                          self.adapter._environment_issues(record)], ["torch"])
        record = {"torch_version": "2.14.1", "dependency_versions": {"lycoris-lora": "4.1.2"}}
        self.assertEqual([issue.field for issue in
                          self.adapter._environment_issues(record)], ["lycoris-lora"])

    def test_validate_reports_a_bad_recorded_torch(self):
        installation = SimpleNamespace(source_path=str(self.root / "engine"),
                                       python_executable="python",
                                       torch_version="2.12.0")
        issues = self.adapter.validate(installation, self.config({"WEIGHTS": str(self.hf())}))
        self.assertIn("torch", [issue.field for issue in issues])

    def test_optional_features_warn_until_the_module_is_probed_installed(self):
        config = self.config({"WEIGHTS": str(self.hf()), "OPTIMIZER": "prodigy"})
        self.assertIn("OPTIMIZER", self.fields(config, "warning"))
        installed = SimpleNamespace(source_path=str(self.root / "engine"),
                                    python_executable="python",
                                    optional_modules={"prodigyopt": True})
        issues = self.adapter.validate(installed, config)
        self.assertNotIn("OPTIMIZER", [issue.field for issue in issues])

    def test_every_optional_feature_has_an_extra_and_a_probe(self):
        extras = dict(KohakuMossTrainerAdapter.optional_extras)
        for choices in KohakuMossTrainerAdapter.OPTIONAL_FEATURES.values():
            for module, extra, _ in choices.values():
                self.assertIn(module, KohakuMossTrainerAdapter.optional_modules)
                self.assertIn(extra, extras)

    def test_logger_offers_only_engine_backends(self):
        self.assertEqual(param("LOGGER").enum_values, ["console", "wandb"])

    # -- the embedded environment LP created for the engine --

    def fake_environment(self, versions):
        site = self.root / "engine" / ".lp" / "python" / "Lib" / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
        for name, found in versions.items():
            (site / f"{name.replace('-', '_')}-{found}.dist-info").mkdir()
        return site

    def test_embedded_environment_below_a_rule_is_a_source_issue(self):
        self.fake_environment({"lycoris-lora": "4.1.2", "torch": "2.14.1"})
        issues = self.adapter.source_issues(self.root / "engine")
        self.assertTrue(any("lycoris-lora=4.1.2" in issue for issue in issues), issues)
        self.assertFalse(any("torch=2.14.1" in issue for issue in issues), issues)

    def test_embedded_environment_inside_the_rules_is_clean(self):
        self.fake_environment({"lycoris-lora": "4.1.3", "torch": "2.14.1",
                               "transformers": "5.20.1", "diffusers": "0.41.0"})
        self.assertEqual(self.adapter.source_issues(self.root / "engine"), ())

    def test_a_venv_layout_makes_the_scan_ambiguous(self):
        self.fake_environment({"lycoris-lora": "4.1.2"})
        (self.root / "engine" / ".venv").mkdir()
        issues = self.adapter.source_issues(self.root / "engine")
        self.assertFalse(any("lycoris-lora" in issue for issue in issues), issues)

    def test_the_real_checkout_environment_matches_its_rules(self):
        """Whatever this checkout has installed, the report must agree with the rules."""
        site = self.adapter._site_packages(ENGINE / ".lp" / "python")
        if site is None:
            self.skipTest("引擎没有嵌入式环境")
        versions = self.adapter._distribution_versions(site)
        issues = " ".join(self.adapter.source_issues(ENGINE))
        for name, _specifier, _why in DEPENDENCY_RULES:
            found = versions.get(name)
            if not found:
                continue
            with self.subTest(distribution=name):
                expected = bool(KohakuMossTrainerAdapter.dependency_issues({name: found}))
                self.assertEqual(f"{name}={found}" in issues, expected)


class ExportTests(AdapterCase):
    def test_post_run_export_is_a_no_op(self):
        """The engine already writes per-checkpoint exports (CHECKPOINT_EXPORT)."""
        self.assertEqual(self.adapter.finalize_artifacts(
            self.root / "job", "python", str(self.root)), ())

    def test_export_ready_is_gone_but_old_drafts_still_validate(self):
        self.assertNotIn("EXPORT_READY", {spec.key for spec in PARAMS})
        config = self.config({"WEIGHTS": str(self.hf()), "EXPORT_READY": False})
        self.assertNotIn("EXPORT_READY", self.fields(config))


class StepReleaseTests(AdapterCase):
    def test_plain_names_stay_strings_and_numbers_become_ints(self):
        cases = {"none": "'none'", "empty_cache": "'empty_cache'", "20": "20", 20: "20"}
        for value, rendered in cases.items():
            with self.subTest(value=value):
                self.assertEqual(self.assignments({"WEIGHTS": str(self.hf()),
                                                   "STEP_RELEASE": value})["STEP_RELEASE"],
                                 ast.literal_eval(rendered))

    def test_unknown_value_is_rejected(self):
        self.assertIn("STEP_RELEASE", self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "STEP_RELEASE": "every"})))


class PlanKnobTests(AdapterCase):
    def manual(self, **values):
        params = {"WEIGHTS": str(self.hf()), "MEMORY_PROFILE": "manual"}
        params.update(values)
        return params

    def test_fractional_checkpoint_uses_the_engine_spelling(self):
        assignments = self.assignments(self.manual(CHECKPOINT_PARTIAL=0.5))
        self.assertEqual(assignments["CHECKPOINT"], "partial:0.5")
        self.assertEqual(self.fields(self.config(self.manual(CHECKPOINT_PARTIAL=0.5))), [])

    def test_fraction_keeps_its_value(self):
        self.assertEqual(self.assignments(self.manual(
            CHECKPOINT_PARTIAL=0.3333))["CHECKPOINT"], "partial:0.3333")

    def test_fraction_conflicts_with_a_named_strategy(self):
        config = self.config(self.manual(CHECKPOINT="selective", CHECKPOINT_PARTIAL=0.5))
        self.assertIn("CHECKPOINT_PARTIAL", self.fields(config))

    def test_fraction_is_dropped_under_a_named_profile(self):
        text = self.render({"WEIGHTS": str(self.hf()), "CHECKPOINT_PARTIAL": 0.5})
        self.assertNotIn("partial:", text)
        warnings = self.issues(self.config({"WEIGHTS": str(self.hf()),
                                            "CHECKPOINT_PARTIAL": 0.5}))
        self.assertTrue(any(issue.severity == "warning"
                            and "CHECKPOINT_PARTIAL" in issue.message
                            for issue in warnings), warnings)

    def test_fraction_outside_the_range_is_rejected(self):
        for value in (1.5, -0.2, "abc", True):
            with self.subTest(value=value):
                self.assertIn("CHECKPOINT_PARTIAL", self.fields(self.config(
                    self.manual(CHECKPOINT_PARTIAL=value))))

    def test_keep_last_zero_means_keep_everything(self):
        self.assertEqual(param("KEEP_LAST").min, 0)
        self.assertEqual(self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "KEEP_LAST": 0})), [])

    def test_catalog_profile_name_replaces_the_dropdown(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf()),
                                        "MEMORY_PROFILE_NAME": "ckpt-ring16-fused",
                                        "STREAM": "ring", "STREAM_RESIDENT": 16})
        self.assertEqual(assignments["MEMORY_PROFILE"], "ckpt-ring16-fused")
        # A named profile keeps the manual plan knobs out (the engine rejects both).
        self.assertNotIn("STREAM", assignments)

    def test_manual_as_the_catalog_name_keeps_the_manual_knobs(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf()),
                                        "MEMORY_PROFILE_NAME": "manual",
                                        "STREAM": "ring", "STREAM_RESIDENT": 4})
        self.assertEqual(assignments["MEMORY_PROFILE"], "manual")
        self.assertEqual(assignments["STREAM"], "ring")

    def test_profile_name_with_spaces_is_rejected(self):
        self.assertIn("MEMORY_PROFILE_NAME", self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "MEMORY_PROFILE_NAME": "ckpt ring16"})))

    def test_profile_name_left_empty_keeps_the_dropdown(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf()),
                                        "MEMORY_PROFILE": "low"})
        self.assertEqual(assignments["MEMORY_PROFILE"], "low")


class ProgressColourTests(AdapterCase):
    def launch(self, **values):
        job = self.root / "job"
        job.mkdir(exist_ok=True)
        params = {"WEIGHTS": str(self.hf())}
        params.update(values)
        target = self.adapter.write_native_config(
            self.installation, self.config(params), job)
        return self.adapter.build_launch(self.installation, target)

    def test_progress_bar_gets_a_colour_switch(self):
        """The job's stdout is a pipe, so tqdm needs FORCE_COLOR to appear at all."""
        self.assertEqual(self.launch().environment_overrides.get("FORCE_COLOR"), "1")

    def test_progress_off_keeps_the_environment_plain(self):
        self.assertNotIn("FORCE_COLOR", self.launch(PROGRESS=False).environment_overrides)

    def test_coloured_step_line_still_parses(self):
        line = ("[12:00:00] [training.callbacks.monitor] [INFO] step "
                "\x1b[93m12\x1b[0m [loss=\x1b[93m0.25\x1b[0m, grad_norm=\x1b[93m0.1\x1b[0m, "
                "lr=\x1b[93m0.0001\x1b[0m]")
        events = self.adapter.parse_log(line)
        self.assertEqual(events[0].payload["step"], 12)
        self.assertAlmostEqual(events[0].payload["loss"], 0.25)
        self.assertAlmostEqual(events[0].payload["lr"], 0.0001)

    def test_plain_line_still_parses(self):
        events = self.adapter.parse_log("step 3 [loss=1.5, grad_norm=0.2, lr=0.001]")
        self.assertEqual(events[0].payload["step"], 3)

    def test_launch_still_targets_the_training_stage(self):
        launch = self.launch()
        scripts = [Path(str(part).replace("\\", "/")).as_posix()
                   for part in launch.argv if str(part).endswith(".py")]
        self.assertTrue(any(script.endswith("scripts/train/moss.py")
                            for script in scripts), launch.argv)
        self.assertEqual(len(launch.pre_argv), 2)


class OutputLayoutTests(AdapterCase):
    def task_folder(self, name="probe"):
        return self.root / "outputs" / self.adapter._safe_folder_name(name)

    def test_the_task_folder_is_not_repeated(self):
        """output_dir already is `<root>/<task>`; RUN_NAME added the task name twice."""
        assignments = self.assignments({"WEIGHTS": str(self.hf())},
                                       output_dir=str(self.task_folder()), name="probe",
                                       output_name="")
        self.assertEqual(Path(assignments["OUTPUT_DIR"]), self.root / "outputs")
        self.assertEqual(assignments["RUN_NAME"], "probe")

    def test_an_explicit_model_name_still_nests(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf())},
                                       output_dir=str(self.task_folder()), name="probe",
                                       output_name="mylora")
        self.assertEqual(Path(assignments["OUTPUT_DIR"]), self.task_folder())
        self.assertEqual(assignments["RUN_NAME"], "mylora")

    def test_an_unrelated_output_dir_is_left_alone(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf())},
                                       output_dir=str(self.root / "outputs" / "other"),
                                       name="probe", output_name="")
        self.assertEqual(Path(assignments["OUTPUT_DIR"]), self.root / "outputs" / "other")
        self.assertEqual(assignments["RUN_NAME"], "probe")

    def test_a_sanitised_task_name_still_matches(self):
        assignments = self.assignments({"WEIGHTS": str(self.hf())},
                                       output_dir=str(self.root / "outputs" / "a_b"),
                                       name="a:b", output_name="")
        self.assertEqual(Path(assignments["OUTPUT_DIR"]), self.root / "outputs")
        self.assertEqual(assignments["RUN_NAME"], "a:b")


class DatasetOptionTests(AdapterCase):
    def entry(self, params):
        text = self.assignments({"WEIGHTS": str(self.hf()), **params})["DATASETS"]
        return text[0]

    def test_default_source_is_the_recursive_folder(self):
        self.assertEqual(param("SOURCE").default, "folder")
        entry = self.entry({})
        self.assertEqual(entry["source"], "folder")
        self.assertIs(entry["recursive"], True)
        self.assertEqual(sorted(entry), ["path", "recursive", "source", "txt_as", "weight"])

    def test_kohya_gets_only_its_own_arguments(self):
        entry = self.entry({"SOURCE": "kohya", "DATASET_INCLUDE_ROOT": True,
                            "DATASET_REPEAT_PREFIX": "product",
                            "DATASET_ARTIST_FROM_FOLDER": "name",
                            "DATASET_HASH_CONTENT": True})
        self.assertEqual(entry["include_root"], True)
        self.assertEqual(entry["repeat_prefix"], "product")
        self.assertEqual(entry["artist_from_folder"], "name")
        self.assertEqual(entry["hash_content"], True)
        self.assertNotIn("recursive", entry)

    def test_folder_gets_only_its_own_arguments(self):
        entry = self.entry({"SOURCE": "folder", "DATASET_RECURSIVE": False,
                            "DATASET_HASH_CONTENT": True})
        self.assertEqual(entry["recursive"], False)
        self.assertEqual(entry["hash_content"], True)
        self.assertNotIn("include_root", entry)

    def test_other_sources_get_no_constructor_arguments(self):
        """`source_kwargs` passes unknown keys to the constructor, which would raise."""
        entry = self.entry({"SOURCE": "moss_shard", "DATASET_INCLUDE_ROOT": True,
                            "DATASET_RECURSIVE": False, "DATASET_HASH_CONTENT": True})
        self.assertEqual(sorted(entry), ["path", "source", "txt_as", "weight"])

    def test_content_hashing_reaches_the_manifest_sources(self):
        for source in ("jsonl", "parquet"):
            with self.subTest(source=source):
                entry = self.entry({"SOURCE": source, "DATASET_HASH_CONTENT": True})
                self.assertEqual(entry["hash_content"], True)
                self.assertNotIn("include_root", entry)
                self.assertNotIn("recursive", entry)

    def test_the_artist_pattern_must_compile(self):
        self.assertIn("DATASET_ARTIST_FROM_FOLDER", self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "SOURCE": "kohya",
             "DATASET_ARTIST_FROM_FOLDER": "["})))

    def test_an_unknown_repeat_prefix_is_reported(self):
        self.assertIn("DATASET_REPEAT_PREFIX", self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "SOURCE": "kohya",
             "DATASET_REPEAT_PREFIX": "sum"})))

    def test_root_level_images_are_reported_for_the_kohya_source(self):
        dataset = self.root / "flat"
        dataset.mkdir()
        (dataset / "a.png").write_bytes(b"x")
        config = self.config({"WEIGHTS": str(self.hf()), "SOURCE": "kohya"},
                             dataset_path=str(dataset))
        self.assertIn("SOURCE", self.fields(config, "warning"))
        kept = self.config({"WEIGHTS": str(self.hf()), "SOURCE": "kohya",
                            "DATASET_INCLUDE_ROOT": True}, dataset_path=str(dataset))
        self.assertNotIn("SOURCE", self.fields(kept))
        folder = self.config({"WEIGHTS": str(self.hf()), "SOURCE": "folder"},
                             dataset_path=str(dataset))
        self.assertNotIn("SOURCE", self.fields(folder))

    def test_unknown_source_is_reported(self):
        self.assertIn("SOURCE", self.fields(self.config(
            {"WEIGHTS": str(self.hf()), "SOURCE": "bogus"})))


class ParameterSurfaceTests(unittest.TestCase):
    def test_weights_help_lists_every_supported_layout(self):
        help_text = param("WEIGHTS").help
        for marker in ("ComfyUI", "diffusion_models", "transformer/", "base", ".safetensors"):
            self.assertIn(marker, help_text)

    def test_component_knobs_accept_a_name_or_a_path(self):
        for key in ("TRANSFORMER", "TEXT_ENCODER", "VAE"):
            self.assertEqual(param(key).path_kind, "file_or_directory", key)


if __name__ == "__main__":
    unittest.main()
