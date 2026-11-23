"""CPU acceptance against an installed sd-scripts 0.13 checkout and its own Python."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
ENGINE = PROJECT / "engine" / "sd-scripts"
ENGINE_PYTHON = ENGINE / ".lp" / "python" / ("python.exe" if os.name == "nt" else "bin/python")


def check_parsers(root):
    import tomllib
    from types import SimpleNamespace

    from adapters.sd_scripts.adapter import KohyaAdapter
    from adapters.sd_scripts.params import ARCHITECTURES, RELEASE_KEYS, parameters_for
    from adapters.sd_scripts.native_options import options_for
    from library import args as args_util, config_util
    from library.run_control import get_run_dir

    overrides = {
        "enable_multiline_caption": True, "multiline_caption_separator": "---next---",
        "gradient_checkpointing_skip_blocks": 1, "block_swap_h2d_only": True, "block_swap_ring_size": 3,
        "use_pinned_memory_for_block_swap": True, "timestep_sampling": "truncated_logit_normal",
        "timestep_cutoff": 0.82, "timestep_uniform_ratio": 0.15, "logit_mean": 0.12, "logit_std": 1.2,
        "discrete_flow_shift": 2.0, "sigmoid_scale": 1.2, "moss_max_token_length": 128,
        "text_encoder_blocks_to_swap": 2, "convrot_int8": True, "convrot_int8_text_encoder": True,
        "convrot_int8_bwd": "int8", "attn_mode": "flash", "split_attn": True,
        "sample_uncond_skip_layers": "19,21", "replica_devices": "1,2", "replica_warmup_steps": 3,
        "compile": True, "compile_backend": "eager", "compile_mode": "reduce-overhead",
        "compile_dynamic": "true", "compile_fullgraph": True, "compile_cache_size_limit": 16,
        "cuda_allow_tf32": True, "cuda_cudnn_benchmark": True,
    }
    checker = unittest.TestCase()
    adapter = KohyaAdapter()
    installation = SimpleNamespace(installation_id="test", source_path=ENGINE, python_executable=ENGINE_PYTHON)
    for arch in ARCHITECTURES:
        job = root / arch.id
        job.mkdir()
        inputs = job / "images"
        inputs.mkdir()
        supported = [p for p in parameters_for(arch.id) if p.key in RELEASE_KEYS]
        expected = {p.key: overrides[p.key] for p in supported}
        if arch.id == "lumina":
            expected["timestep_sampling"] = "nextdit_shift"
        draft = dict(name="unit", architecture=arch.id, base_model_path=str(job / "model.safetensors"),
                     dataset_path=str(inputs), output_dir=str(job / "output"),
                     params={**expected, "num_repeats": 3, "resolution": 512})
        native = adapter.write_native_config(installation, draft, job)
        launch = adapter.build_launch(installation, native)
        checker.assertEqual(launch.argv, adapter.preview_argv(installation, draft, native))
        checker.assertEqual(Path(launch.argv[2]), ENGINE / arch.script)
        checker.assertFalse(launch.pre_argv)
        parser = importlib.import_module(Path(arch.script).stem).setup_parser()
        # The upstream loader parses sys.argv a second time after reading the TOML.
        sys.argv = list(launch.argv[2:])
        opts = parser.parse_args()
        args_util.verify_command_line_training_args(opts)
        opts = args_util.read_config_from_file(opts, parser)
        native_values = tomllib.loads(native.read_text(encoding="utf-8"))
        actions = {a.dest: a for a in parser._actions}
        options = options_for(ENGINE, arch.script)
        checker.assertEqual(set(options), set(actions) - {"help"}, arch.id)
        for key, (_, choices, nargs) in options.items():
            checker.assertEqual(choices, list(actions[key].choices) if actions[key].choices is not None else None,
                                (arch.id, key))
            if actions[key].nargs != 0:
                checker.assertEqual(nargs, actions[key].nargs, (arch.id, key))
        fallbacks = dict(optimizer_type="AdamW", network_dim=4, xformers="none", clip_skip=1, attn_mode="torch")
        caps = adapter.capabilities_for_architecture(installation, arch.id)
        defaults = tomllib.loads(adapter.native_config_text({**draft, "params": {}}, job / "dataset.toml"))
        for field in caps["params"]:
            key = field["key"]
            if field["native"] or key in ("resolution", "enable_bucket", "caption_extension"):
                checker.assertEqual(field["default"], fallbacks.get(key, actions[key].default), (arch.id, key))
            if field["native"]:
                rendered = defaults.get(key, actions[key].default)
                if key in fallbacks and rendered == actions[key].default:
                    rendered = fallbacks[key]
                checker.assertEqual(rendered, field["default"], (arch.id, key))
        checker.assertFalse(native_values.keys() - actions.keys(), arch.id)
        checker.assertEqual(native_values.keys() & RELEASE_KEYS, expected.keys(), arch.id)
        for p in supported:
            key, value = p.key, expected[p.key]
            checker.assertEqual(native_values.get(key), value, (arch.id, key))
            checker.assertEqual(getattr(opts, key), value, (arch.id, key))
            if actions[key].choices is not None:
                checker.assertIn(value, actions[key].choices, (arch.id, key))
                checker.assertFalse({option for option, _ in p.options} - set(actions[key].choices), (arch.id, key))
        checker.assertTrue(opts.run_control)
        checker.assertEqual(Path(get_run_dir(opts)), job / "output" / "unit_run")
        dataset = config_util.load_user_config(opts.dataset_config)
        builder = config_util.BlueprintGenerator(config_util.ConfigSanitizer(True, True, opts.masked_loss, True))
        parsed = builder.generate(dataset, opts).dataset_group.datasets[0]
        checker.assertEqual(parsed.params.resolution, (512, 512))
        checker.assertFalse(parsed.params.enable_bucket)
        subset = parsed.subsets[0].params
        checker.assertEqual(Path(subset.image_dir), inputs)
        checker.assertEqual(subset.num_repeats, 3)
        checker.assertEqual(subset.caption_extension, ".caption")
        checker.assertTrue(subset.enable_multiline_caption)
        checker.assertEqual(subset.multiline_caption_separator, "---next---")

    base = dict(draft, architecture="sd1")
    def errors(arch="sd1", **params):
        return {v.field for v in adapter.validate(installation, {**base, "architecture": arch, "params": params})
                if v.severity == "error"}
    checker.assertEqual(errors(), {"resolution"})
    checker.assertEqual(errors("anima", resolution=512), {"vae", "qwen3"})
    checker.assertEqual(errors("flux1", resolution=512), {"clip_l", "t5xxl", "ae"})
    checker.assertEqual(errors("hunyuan_image", resolution=512), {"vae", "text_encoder", "byt5"})
    checker.assertFalse(errors("flux1", resolution=512, ae="ae", t5xxl="t5",
                               extra_params="--model_type chroma --apply_t5_attn_mask"))
    for extra in ("--fake_option", "--mixed_precision fp32", "--learning_rate 1e309", "--ddp_timeout 3.5",
                  "--log_tracker_name 12", "--text_encoder_lr NaN", "--network_module networks.lora_flux",
                  "--full_fp16", "--cache_latents_to_disk --random_crop", "--timestep_sampling sigmoid",
                  "--optimizer_args invalid", "--v_parameterization --v_pred_like_loss 0.5"):
        checker.assertTrue(errors(resolution=512, extra_params=extra), extra)
    checker.assertFalse(errors(resolution=512, extra_params="--v_parameterization --scale_v_pred_loss_like_noise_pred"))
    checker.assertFalse(errors(resolution=512, extra_params='--base_weights "path with spaces.safetensors" --base_weights_multiplier 0.5'))
    data = tomllib.loads(adapter.native_config_text({**base, "params": {
        "extra_params": '--base_weights "path with spaces.safetensors" --base_weights_multiplier 0.5 --text_encoder_lr "0.001 0.002"'}}, root / "dataset.toml"))
    checker.assertEqual(data["base_weights"], ["path with spaces.safetensors"])
    checker.assertEqual(data["base_weights_multiplier"], [0.5])
    checker.assertEqual(data["text_encoder_lr"], [0.001, 0.002])


def check_moss_cpu():
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from library import moss_image_models, moss_image_utils

    checker = unittest.TestCase()
    config = moss_image_utils.load_text_encoder_config()
    checker.assertIsInstance(config, Qwen3_5TextConfig)
    checker.assertFalse(config.use_cache)
    # Materialize only the model structure of the official encoder, without allocating 9B weights.
    with torch.device("meta"):
        encoder = Qwen3_5TextModel(config)
    checker.assertEqual(len(encoder.layers), config.num_hidden_layers)
    torch.manual_seed(23)
    torch.set_num_threads(1)
    model = moss_image_models.MossImage(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=24,
                                        freq_dim=16, latent_channels=8, rope_base=10000, attn_mode="torch")
    model.train()
    model.enable_gradient_checkpointing(skip_blocks=1)
    latents = torch.randn(2, 8, 3, 5, requires_grad=True)
    context = torch.randn(2, 7, 24)
    pred = model(latents, torch.tensor([750.0, 120.0]), context, torch.ones(2, 7, dtype=torch.int32))
    checker.assertEqual(pred.shape, latents.shape)
    pred.square().mean().backward()
    checker.assertTrue(torch.isfinite(latents.grad).all().item())


@unittest.skipUnless(ENGINE_PYTHON.is_file() and (ENGINE / "moss_image_train_network.py").is_file(),
                     "requires an installed sd-scripts 0.13 environment")
class SdScriptsReleaseTests(unittest.TestCase):
    def test_official_entrypoints_dataset_and_moss_on_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = subprocess.run([str(ENGINE_PYTHON), "-B", str(Path(__file__).resolve()), temp],
                                  cwd=temp, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
                                       "PYTHONIOENCODING": "utf-8"}, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    sys.path[:0] = [str(PROJECT), str(ENGINE)]
    check_parsers(Path(sys.argv[1]))
    check_moss_cpu()
