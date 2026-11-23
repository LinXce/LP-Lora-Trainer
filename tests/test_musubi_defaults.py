"""Compare each adapted training/caching stage with the installed upstream parser."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
ENGINE = PROJECT / "engine" / "musubi-tuner"
PYTHON = ENGINE / ".lp" / "python" / ("python.exe" if os.name == "nt" else "bin/python")


class Parsed(Exception):
    """Exit a real entry point immediately after obtaining its configured parser."""


def check_commands(jobs):
    import torch
    import toml
    from PIL import Image
    from adapters.musubi_tuner.adapter import MusubiTunerAdapter
    from adapters.musubi_tuner.krea2_cache_text_encoder import parse_adapter_args
    from adapters.musubi_tuner.pipeline import PIPELINES
    from adapters.musubi_tuner.extra_options import EXTRA_OPTIONS
    from musubi_tuner.training.parser_common import read_config_from_file
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
    from musubi_tuner.networks import lora
    from musubi_tuner.minimax_h3.cache_plan import validate_h3_image_dataset_task

    adapter = MusubiTunerAdapter()
    install = SimpleNamespace(installation_id="acceptance", source_path=ENGINE, python_executable=PYTHON)
    checker = unittest.TestCase()
    images = jobs / "images"
    images.mkdir()
    Image.new("RGB", (64, 64)).save(images / "one.png")
    controls = jobs / "controls"
    controls.mkdir()
    Image.new("RGB", (64, 64)).save(controls / "one.png")
    (images / "one.txt").write_text("one", encoding="utf-8")
    for arch, pipe in PIPELINES.items():
        job = jobs / arch
        job.mkdir()
        schema = adapter.capabilities_for_architecture(install, arch)
        defaults = {p["key"]: p["default"] for p in schema["params"]}
        config = dict(name="review", architecture=arch, base_model_path=str(job / "model.safetensors"),
                      output_dir=str(job / "out"), dataset_path=str(images), params={})
        checker.assertEqual(schema["architecture_by_network_module"][pipe.module], arch)
        network = importlib.import_module("musubi_tuner." + pipe.module)
        model = torch.nn.Module()
        model.hidream_o1_task = "t2i"
        with patch.object(lora, "LoRANetwork") as build:
            build.return_value.unet_loras = [object()]
            network.create_arch_network(1.0, None, None, None, [], model)
        checker.assertEqual(build.call_args.kwargs["lora_dim"], defaults["network_dim"], arch)
        untouched = toml.loads(adapter._render_train(config))
        for key in ("optimizer_type", "mixed_precision", "save_precision", "save_every_n_steps"):
            checker.assertNotIn(key, untouched, arch)
        checker.assertEqual(untouched["network_dim"], 4, arch)
        required = {key: str(job / (key + ".bin")) for key in pipe.required}
        if "kandinsky_task" in required:
            required["kandinsky_task"] = "k5-lite-t2i-hd"
        if "h3_task" in required:
            required.update(h3_task="t2va", h3_one_frame=True)
        if "control_directory" in required:
            required["control_directory"] = str(controls)
        if "fp_1f_target_index" in required:
            required.update(fp_1f_target_index=1, fpack_one_frame=True)
        cfg = {**config, "params": {**defaults, **required, "optimizer_type": "AdamW", "sdpa": True,
                                     "caption_extension": ".txt", "gradient_checkpointing": False,
                                     "max_data_loader_n_workers": 0}}
        issues = [i.message for i in adapter.validate(install, cfg) if i.severity == "error"]
        checker.assertFalse(issues, (arch, issues))
        native = adapter.write_native_config(install, cfg, job)
        commands = adapter.build_launch(install, native).commands()
        checker.assertEqual(commands, adapter.preview_commands(install, cfg, native))
        data = toml.load(job / "dataset.toml")
        checker.assertEqual(data["general"]["caption_extension"], ".txt")
        checker.assertEqual(data["general"]["resolution"], [960, 544])
        for stage, argv in enumerate(commands):
            script, flags = Path(argv[2]), list(argv[3:])
            if script.name == "krea2_cache_text_encoder.py":
                wrapper, flags = parse_adapter_args(flags)
                script = ENGINE / "src" / "musubi_tuner" / wrapper.engine_script
            original = argparse.ArgumentParser.parse_args
            def inspect(parser, *args, **kwargs):
                parsed = original(parser, *args, **kwargs)
                checker.assertEqual(Path(parsed.dataset_config), job / "dataset.toml")
                if stage == 0:
                    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(data, parsed).dataset_group
                    image_params = blueprint.datasets[0].params
                    datasource = ImageDirectoryDatasource(image_params.image_directory, image_params.caption_extension,
                                                          image_params.control_directory, control_count_per_image=1)
                    checker.assertEqual(datasource.get_caption(0)[1], "one")
                    if arch in ("framepack", "flux1_kontext"):
                        checker.assertEqual(len(datasource.get_image_data(0)[3]), 1)
                    if arch == "framepack":
                        checker.assertTrue(parsed.one_frame)
                        checker.assertEqual(image_params.fp_1f_target_index, 1)
                    if arch == "minimax_h3":
                        validate_h3_image_dataset_task(SimpleNamespace(fp_1f_clean_indices=None, has_control=False),
                                                       parsed.task, parsed.one_frame)
                if stage == 2:
                    actions = {action.dest: action for action in parser._actions}
                    shaped = toml.load(native)
                    checker.assertFalse(shaped.keys() - actions.keys(), arch)
                    for (scope, kind), names in EXTRA_OPTIONS.items():
                        if scope in ("common", arch):
                            checker.assertFalse(set(names) - actions.keys(), (arch, kind))
                    # The default representation may differ for path fields and choices left
                    # automatic. These keys are resolved by the LoRA factory or model entry.
                    resolved = {"network_dim", "dit_dtype", "vae_dtype", "qwen_model_version"}
                    for p in schema["params"]:
                        key = p.get("native_key") or p["key"]
                        if not p.get("native", True) or key not in actions or p["key"] in resolved:
                            continue
                        value = defaults[p["key"]]
                        expected = actions[key].default
                        if value in (None, "") and expected in (None, "", []):
                            continue
                        checker.assertEqual(value, expected, (arch, key))
                    with patch.object(argparse.ArgumentParser, "parse_args", original):
                        parsed = read_config_from_file(parsed, parser)
                    checker.assertEqual(parsed.network_dim, 4, arch)
                    checker.assertEqual(parsed.max_data_loader_n_workers, 0, arch)
                    checker.assertFalse(parsed.gradient_checkpointing)
                elif stage == 0 and arch == "kandinsky5":
                    checker.assertEqual(parsed.vae_dtype, "float16")
                    checker.assertEqual(untouched["vae_dtype"], "bfloat16")
                raise Parsed()
            sys.argv = [str(script), *flags]
            # Wan and FramePack call current_device() when defining the T5 class, before parsing.
            with patch.object(torch.cuda, "current_device", return_value=0), \
                    patch.object(argparse.ArgumentParser, "parse_args", inspect), checker.assertRaises(Parsed):
                runpy.run_path(str(script), run_name="__main__")


class MusubiDefaultTests(unittest.TestCase):
    @unittest.skipUnless(PYTHON.is_file(), "requires the installed Musubi environment")
    def test_all_39_official_stage_parsers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([str(PYTHON), "-B", str(Path(__file__).resolve()), tmp], cwd=tmp,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
                                         "PYTHONIOENCODING": "utf-8"}, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    sys.path[:0] = [str(PROJECT), str(ENGINE / "src")]
    check_commands(Path(sys.argv[1]))
