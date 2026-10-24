"""Regression coverage for the kohya/sd-scripts fix pass.

Each test pins one defect that a real run would hit: list arguments rendered as
strings, a missing PyTorch install step, a forced SDXL flag, an invalid
``vae_batch_size``, enum values the engine does not accept, artifact naming that
decodes epochs as steps, and engine asserts that used to surface only after
model loading.
"""
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from adapters.sd_scripts.adapter import KohyaAdapter, list_values
from adapters.sd_scripts.params import PARAMS
from app.schemas.engine import EngineInstallation, EngineRevision, ManagementMode

ARCH_SCRIPTS = (
    "train_network.py",
    "sdxl_train_network.py",
    "sd3_train_network.py",
    "flux_train_network.py",
    "hunyuan_image_train_network.py",
    "lumina_train_network.py",
    "anima_train_network.py",
)

# The engine's argparse choices, quoted from ``train_network.py --help`` /
# ``library/args.py`` for the pinned commit (690ea7f.…). They are asserted here
# so an adapter option can never drift outside what the engine accepts again.
ENGINE_SAVE_MODEL_AS = {"ckpt", "pt", "safetensors"}
ENGINE_SAVE_PRECISION = {"float", "fp16", "bf16"}
ENGINE_SAMPLE_SAMPLERS = {
    "ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a",
    "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a",
    "k_dpm_2", "k_dpm_2_a",
}
ENGINE_MAX_TOKEN_LENGTHS = {None, 150, 225}


def make_source(root: Path, *, requirements: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "library").mkdir(exist_ok=True)
    (root / "library" / "train_util.py").write_text("# library\n", encoding="utf-8")
    for script in ARCH_SCRIPTS:
        (root / script).write_text("# entry\n", encoding="utf-8")
    if requirements:
        (root / "requirements.txt").write_text(
            "accelerate==1.15.0\nnumpy\n-e .\n", encoding="utf-8"
        )
    return root


def installation(root: Path) -> EngineInstallation:
    return EngineInstallation(
        installation_id="test-kohya",
        engine_id="kohya",
        source_path=str(root),
        management_mode=ManagementMode.USER_MANAGED,
        revision=EngineRevision(),
    )


def config(root: Path, architecture: str = "sd1", **params) -> dict:
    return dict(
        name="demo",
        output_name="demo",
        architecture=architecture,
        base_model_path=str(root / "base.safetensors"),
        dataset_path=str(root / "dataset"),
        output_dir=str(root / "output"),
        params=params,
    )


def param(key: str):
    return next(p for p in PARAMS if p.key == key)


class AdapterTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = make_source(self.root / "engine")
        self.adapter = KohyaAdapter()
        self.installation = installation(self.source)

    def tearDown(self):
        self._tmp.cleanup()

    def render(self, architecture: str = "sd1", **params) -> dict:
        text = self.adapter.native_config_text(
            config(self.root, architecture, **params), self.root / "job" / "dataset.toml"
        )
        return tomllib.loads(text)

    def render_kwargs(self, architecture: str = "sd1", **params) -> dict:
        return config(self.root, architecture, **params)


class ListArgumentTests(AdapterTestBase):
    def test_list_values_splits_on_whitespace_and_groups_quotes(self):
        self.assertEqual(list_values("a=1 b=2"), ["a=1", "b=2"])
        self.assertEqual(list_values("a=1\nb=2"), ["a=1", "b=2"])
        self.assertEqual(list_values('caption="a b" x=1'), ["caption=a b", "x=1"])
        self.assertEqual(list_values("  "), [])

    def test_list_arguments_render_as_toml_arrays(self):
        values = self.render(
            network_args="conv_dim=8 conv_alpha=8",
            optimizer_args="weight_decay=0.01 betas=0.9,0.99",
            lr_scheduler_args="T_max=100",
        )
        self.assertEqual(values["network_args"], ["conv_dim=8", "conv_alpha=8"])
        self.assertEqual(values["optimizer_args"], ["weight_decay=0.01", "betas=0.9,0.99"])
        self.assertEqual(values["lr_scheduler_args"], ["T_max=100"])

    def test_empty_list_arguments_are_omitted(self):
        values = self.render(network_args="", optimizer_args="", lr_scheduler_args="")
        for key in ("network_args", "optimizer_args", "lr_scheduler_args"):
            self.assertNotIn(key, values)

    def test_custom_parameters_targeting_a_list_are_normalised(self):
        values = self.render(extra_params='--network_args "conv_dim=8 conv_alpha=8"')
        self.assertEqual(values["network_args"], ["conv_dim=8", "conv_alpha=8"])


class EnumAlignmentTests(AdapterTestBase):
    def test_save_model_as_only_offers_values_lp_can_deliver(self):
        options = set(param("save_model_as").enum_values)
        self.assertTrue(options <= ENGINE_SAVE_MODEL_AS, options)
        # diffusers is not an engine choice at all: the extension is saved as a pickle.
        self.assertNotIn("diffusers", options)
        self.assertNotIn("diffusers_safetensors", options)

    def test_sample_sampler_matches_engine_choices(self):
        options = set(param("sample_sampler").enum_values)
        self.assertEqual(options, ENGINE_SAMPLE_SAMPLERS)
        self.assertIn("heun", options)
        self.assertIn("dpmsolver++", options)
        # Both used to fall through to DDIM in library/sampling.py.
        self.assertNotIn("dpm++_2m", options)
        self.assertNotIn("dpm++_2m_karras", options)

    def test_save_precision_matches_engine_choices(self):
        options = set(param("save_precision").enum_values)
        self.assertEqual(options, ENGINE_SAVE_PRECISION)
        self.assertNotIn("fp32", options)

    def test_max_token_length_rejects_values_outside_the_engine_choices(self):
        for value in (151, 300, 0):
            issues = self.adapter.validate(
                self.installation, config(self.root, "sd1", max_token_length=value)
            )
            self.assertIn("max_token_length", [i.field for i in issues if i.severity == "error"])
        for value in (150, 225):
            issues = self.adapter.validate(
                self.installation, config(self.root, "sd1", max_token_length=value)
            )
            self.assertNotIn("max_token_length", [i.field for i in issues if i.severity == "error"])

    def test_max_token_length_75_means_engine_default_and_is_omitted(self):
        values = self.render(max_token_length=75)
        self.assertNotIn("max_token_length", values)


class ClipSkipTests(AdapterTestBase):
    def test_clip_skip_only_applies_to_sd1_and_sd2(self):
        self.assertEqual(set(param("clip_skip").architectures), {"sd1", "sd2"})
        for architecture in ("sdxl", "sd3", "flux1", "hunyuan_image", "lumina", "anima"):
            self.assertNotIn("clip_skip", self.render(architecture))

    def test_clip_skip_is_written_only_when_it_changes_the_default(self):
        self.assertNotIn("clip_skip", self.render("sd1"))
        self.assertEqual(self.render("sd1", clip_skip=2)["clip_skip"], 2)


class VaeBatchSizeTests(AdapterTestBase):
    def test_minimum_is_one_everywhere(self):
        self.assertEqual(param("vae_batch_size").min, 1)
        for architecture in ("sd1", "sd2", "sdxl", "sd3", "flux1", "hunyuan_image", "lumina", "anima"):
            issues = self.adapter.validate(
                self.installation, config(self.root, architecture, vae_batch_size=0)
            )
            self.assertIn(
                "vae_batch_size",
                [i.field for i in issues if i.severity == "error"],
                architecture,
            )


class SdxlTrainingFlagTests(AdapterTestBase):
    def test_sdxl_does_not_force_unet_only_training(self):
        values = self.render("sdxl")
        self.assertFalse(values.get("network_train_unet_only"))

    def test_explicit_user_choice_is_written(self):
        self.assertTrue(self.render("sdxl", network_train_unet_only=True)["network_train_unet_only"])
        self.assertTrue(
            self.render("sdxl", network_train_text_encoder_only=True)["network_train_text_encoder_only"]
        )

    def test_launch_selects_the_architecture_entry_script(self):
        expected = {
            "sd1": "train_network.py",
            "sd2": "train_network.py",
            "sdxl": "sdxl_train_network.py",
            "sd3": "sd3_train_network.py",
            "flux1": "flux_train_network.py",
            "hunyuan_image": "hunyuan_image_train_network.py",
            "lumina": "lumina_train_network.py",
            "anima": "anima_train_network.py",
        }
        job = self.root / "job"
        for architecture, script in expected.items():
            self.adapter.write_native_config(
                self.installation, config(self.root, architecture), job
            )
            launch = self.adapter.build_launch(self.installation, job / "config.native.toml")
            self.assertEqual(Path(launch.argv[2]).name, script, architecture)

    def test_legacy_config_without_marker_still_selects_sdxl(self):
        legacy = self.root / "legacy.toml"
        legacy.write_text(
            'network_module = "networks.lora"\nnetwork_train_unet_only = true\n'
            'pretrained_model_name_or_path = "x"\ndataset_config = "d"\n'
            'output_dir = "o"\noutput_name = "n"\n',
            encoding="utf-8",
        )
        launch = self.adapter.build_launch(self.installation, legacy)
        self.assertEqual(Path(launch.argv[2]).name, "sdxl_train_network.py")


class InstallationPlanTests(AdapterTestBase):
    def test_adapter_declares_cuda_install_sources_and_installs_pytorch(self):
        self.assertEqual(self.adapter.installation_sources(self.source), ("cu124", "cu128"))
        plan = self.adapter.installation_plan(self.source, "py.exe", "cu124", uv="uv.exe")
        self.assertEqual(len(plan.commands), 1)
        install = plan.commands[0]
        self.assertIn("torch==2.6.0+cu124", install)
        self.assertIn("torchvision==0.21.0+cu124", install)
        self.assertIn("https://download.pytorch.org/whl/cu124", install)
        # The requirements file has no PyTorch entries; LP must add them itself.
        self.assertIn("accelerate==1.15.0", install)

    def test_rtx50_uses_the_documented_28_pin(self):
        install = self.adapter.installation_plan(self.source, "py.exe", "cu128", uv="uv.exe").commands[0]
        self.assertIn("torch==2.8.0+cu128", install)
        self.assertIn("torchvision==0.23.0+cu128", install)

    def test_existing_source_installs_no_pytorch(self):
        install = self.adapter.installation_plan(self.source, "py.exe", "existing", uv="uv.exe").commands[0]
        self.assertFalse(any(str(part).startswith(("torch==", "torchvision==")) for part in install))
        self.assertNotIn("download.pytorch.org", install)

    def test_wheel_mirror_replaces_the_official_index(self):
        mirror = "https://mirror.example/pytorch-wheels/cu128"
        install = self.adapter.installation_plan(
            self.source, "py.exe", "cu128", uv="uv.exe", torch_mirror_url=mirror
        ).commands[0]
        self.assertIn(f"{mirror}/torch/", install)
        self.assertIn(f"{mirror}/torchvision/", install)
        self.assertNotIn("download.pytorch.org", install)
        self.assertIn("torch==2.8.0+cu128", install)

    def test_unknown_cuda_tag_is_refused(self):
        with self.assertRaises(ValueError):
            self.adapter.installation_plan(self.source, "py.exe", "cu130", uv="uv.exe")

    def test_diagnosis_requires_torchvision(self):
        self.assertIn("torchvision", self.adapter.diagnostic_modules)

    def test_help_text_no_longer_claims_an_official_installer(self):
        self.assertNotIn("setup_windows.py", self.adapter.installation_help)
        self.assertNotIn("setup_windows.py", self.adapter.installation_source_help)
        self.assertNotIn("官方安装器", self.adapter.installation_help)


class EngineAssertionValidationTests(AdapterTestBase):
    def fields(self, architecture, **params):
        issues = self.adapter.validate(self.installation, config(self.root, architecture, **params))
        return [i.field for i in issues if i.severity == "error"]

    def test_bucket_steps_must_be_divisible_per_architecture(self):
        self.assertIn("bucket_reso_steps", self.fields("sdxl", bucket_reso_steps=24))
        self.assertIn("bucket_reso_steps", self.fields("sd1", bucket_reso_steps=32))
        self.assertNotIn("bucket_reso_steps", self.fields("sd1", bucket_reso_steps=64))
        self.assertNotIn("bucket_reso_steps", self.fields("anima", bucket_reso_steps=16))

    def test_latent_cache_and_augmentation_are_mutually_exclusive(self):
        self.assertIn("cache_latents", self.fields("sd1", cache_latents=True, color_aug=True))
        self.assertIn("cache_latents", self.fields("sd1", cache_latents=True, random_crop=True))
        # cache_latents defaults to True, so leaving it untouched must still be caught.
        self.assertIn("cache_latents", self.fields("sd1", color_aug=True))
        self.assertNotIn("cache_latents", self.fields("sd1", cache_latents=False, color_aug=True))

    def test_te_cache_with_caption_shuffling_is_rejected(self):
        self.assertIn(
            "cache_text_encoder_outputs",
            self.fields("sdxl", cache_text_encoder_outputs=True, shuffle_caption=True),
        )
        self.assertIn(
            "cache_text_encoder_outputs",
            self.fields("flux1", cache_text_encoder_outputs=True, caption_dropout_rate=0.1),
        )

    def test_fused_backward_pass_requires_adafactor_and_no_accumulation(self):
        self.assertIn("fused_backward_pass", self.fields("sd1", fused_backward_pass=True))
        self.assertIn(
            "fused_backward_pass",
            self.fields("sd1", fused_backward_pass=True, optimizer_type="Adafactor",
                        gradient_accumulation_steps=2),
        )
        self.assertNotIn(
            "fused_backward_pass",
            self.fields("sd1", fused_backward_pass=True, optimizer_type="Adafactor",
                        gradient_accumulation_steps=1),
        )

    def test_scale_v_pred_loss_requires_v_parameterization(self):
        self.assertIn(
            "scale_v_pred_loss_like_noise_pred",
            self.fields("sd1", scale_v_pred_loss_like_noise_pred=True),
        )

    def test_masked_loss_is_reported_as_ineffective(self):
        issues = self.adapter.validate(self.installation, config(self.root, "sd1", masked_loss=True))
        self.assertIn("masked_loss", [i.field for i in issues if i.severity == "warning"])


class ArtifactNamingTests(AdapterTestBase):
    def artifacts(self):
        out = self.root / "output"
        out.mkdir()
        for name in ("lora.safetensors", "lora-step00000200.safetensors", "lora-000002.safetensors"):
            (out / name).write_bytes(b"")
        (out / "sample.png").write_bytes(b"")
        state = out / "demo-000002-state"
        state.mkdir()
        (state / "model.safetensors").write_bytes(b"")
        job = self.root / "job-artifacts"
        job.mkdir()
        (job / "job.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")
        return {a.path.name: a for a in self.adapter.collect_artifacts(job)}

    def test_steps_come_from_the_step_file_name_only(self):
        artifacts = self.artifacts()
        self.assertEqual(artifacts["lora-step00000200.safetensors"].step, 200)
        self.assertIsNone(artifacts["lora-000002.safetensors"].step)
        self.assertIsNone(artifacts["lora.safetensors"].step)

    def test_accelerate_state_weights_are_not_artifacts(self):
        self.assertNotIn("model.safetensors", self.artifacts())

    def test_samples_are_still_collected(self):
        self.assertEqual(self.artifacts()["sample.png"].kind, "sample")


class ResumeFlagTests(unittest.TestCase):
    def test_resume_from_huggingface_is_a_boolean_flag(self):
        # library/args.py declares it action="store_true"; the engine parses the
        # resume value as the HF path, so this field is never text.
        self.assertEqual(param("resume_from_huggingface").type, "bool")
        self.assertIs(param("resume_from_huggingface").default, False)


class EngineDefaultAlignmentTests(AdapterTestBase):
    """Static defaults must equal the pinned engine's argparse defaults.

    Each value below is quoted with the engine location it comes from, so the
    adapter cannot silently keep an older sd-scripts default again.
    """

    ENGINE_DEFAULTS = {
        # library/args.py:287-292
        "max_token_length": None,
        # library/args.py:345-349
        "max_data_loader_n_workers": 8,
        # library/args.py:366-371
        "mixed_precision": "no",
        # train_network.py:1924-1927
        "network_alpha": 1,
        # library/args.py:1024-1028
        "max_bucket_reso": 1024,
    }

    def test_static_defaults_match_the_engine(self):
        for key, value in self.ENGINE_DEFAULTS.items():
            self.assertEqual(param(key).default, value, key)

    def test_untouched_form_renders_the_engine_defaults(self):
        values = self.render("sd1")
        self.assertEqual(values["max_bucket_reso"], 1024)
        self.assertEqual(values["max_data_loader_n_workers"], 8)
        self.assertEqual(values["mixed_precision"], "no")
        self.assertEqual(values["network_alpha"], 1)
        # None means "use the engine default", so the key is omitted entirely.
        self.assertNotIn("max_token_length", values)

    def test_network_alpha_accepts_fractional_values(self):
        # The engine declares it type=float; the old int-only field rejected 0.5.
        self.assertEqual(param("network_alpha").type, "float")
        self.assertEqual(self.render("sd1", network_alpha=0.5)["network_alpha"], 0.5)
        self.assertEqual(self.render("sd1", network_alpha=16)["network_alpha"], 16)


class PrecisionCombinationTests(AdapterTestBase):
    def errors(self, architecture="sd1", **params):
        issues = self.adapter.validate(self.installation, config(self.root, architecture, **params))
        return [i.field for i in issues if i.severity == "error"]

    def test_full_fp16_requires_fp16_mixed_precision(self):
        self.assertIn("full_fp16", self.errors(full_fp16=True, mixed_precision="no"))
        self.assertNotIn("full_fp16", self.errors(full_fp16=True, mixed_precision="fp16"))

    def test_full_bf16_requires_bf16_mixed_precision(self):
        self.assertIn("full_bf16", self.errors(full_bf16=True, mixed_precision="fp16"))
        self.assertNotIn("full_bf16", self.errors(full_bf16=True, mixed_precision="bf16"))

    def test_fp8_base_requires_a_mixed_precision(self):
        # The engine asserts fp8_base with mixed_precision='no'; that is now the
        # default, so this combination is easy to hit.
        self.assertIn("fp8_base", self.errors(fp8_base=True, mixed_precision="no"))
        self.assertNotIn("fp8_base", self.errors(fp8_base=True, mixed_precision="fp16"))

    def test_hunyuan_image_does_not_offer_fp8_base(self):
        self.assertNotIn("hunyuan_image", param("fp8_base").architectures)
        issues = self.adapter.validate(
            self.installation, config(self.root, "hunyuan_image", fp8_base=True))
        self.assertIn("fp8_base", [i.field for i in issues if i.severity == "warning"])


class HuggingFaceFieldTests(AdapterTestBase):
    def test_repo_type_accepts_free_text(self):
        # The engine takes an unconstrained string (library/args.py:187-193).
        self.assertEqual(param("huggingface_repo_type").type, "string")
        issues = self.adapter.validate(
            self.installation, config(self.root, "sd1", huggingface_repo_type="custom-repo-type"))
        self.assertNotIn("huggingface_repo_type", [i.field for i in issues if i.severity == "error"])

    def test_repo_visibility_accepts_free_text(self):
        self.assertEqual(param("huggingface_repo_visibility").type, "string")
        issues = self.adapter.validate(
            self.installation, config(self.root, "sd1", huggingface_repo_visibility="internal"))
        self.assertNotIn("huggingface_repo_visibility", [i.field for i in issues if i.severity == "error"])


class OptimizerChoiceTests(unittest.TestCase):
    def test_engine_documented_optimizers_are_selectable(self):
        options = set(param("optimizer_type").enum_values)
        # Only names optimizer.py actually implements (explicit branches), not every
        # string its --help comment lists: AdEMAMix8bit has no branch and would fall
        # through to torch.optim.AttributeError, so it is not offered.
        self.assertTrue({
            "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit", "PagedLion8bit",
            "SGDNesterov8bit", "DAdaptAdanIP",
        } <= options)
        self.assertNotIn("AdEMAMix8bit", options)
        self.assertNotIn("PagedAdEMAMix8bit", options)
        # The engine lowercases the name, so the historical spelling still works.
        self.assertIn("Adafactor", options)
        self.assertEqual(len(options), len(param("optimizer_type").enum_values))


class VaeChunkSizeScopeTests(AdapterTestBase):
    def test_shared_by_hunyuan_image_and_anima(self):
        self.assertEqual(set(param("vae_chunk_size").architectures),
                         {"hunyuan_image", "anima"})
        self.assertNotIn("vae_chunk_size", self.render("sd1"))
        self.assertEqual(self.render("hunyuan_image", vae_chunk_size=64)["vae_chunk_size"], 64)
        self.assertEqual(self.render("anima", vae_chunk_size=64)["vae_chunk_size"], 64)


class DiagnosticModuleTests(unittest.TestCase):
    def test_yaml_is_not_required(self):
        # sd-scripts imports no yaml module and requirements.txt does not install
        # pyyaml, so requiring it failed diagnosis on every real environment.
        self.assertNotIn("yaml", KohyaAdapter().diagnostic_modules)

    def test_stale_installer_claims_are_gone(self):
        adapter = KohyaAdapter()
        for text in (adapter.installation_help, adapter.installation_source_help):
            self.assertNotIn("setup_windows.py", text)
            self.assertNotIn("官方安装器", text)


if __name__ == "__main__":
    unittest.main()
