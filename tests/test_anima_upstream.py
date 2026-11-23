"""Check the adapter against the installed Anima CLI without loading training weights."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters.anima_lora.adapter import AnimaLoraAdapter
from adapters.anima_lora import validation
from adapters.anima_lora.params import VERSIONED_KEYS
from app.services.installation import InstallationService
from app.storage.state import StateStore
from supervisor.worker import register_artifacts


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine/anima_lora"
PYTHON = ENGINE / ".lp/python/python.exe"


class AnimaCheckoutTests(unittest.TestCase):
    def test_parameter_checks_run_without_opening_a_console(self):
        settings = '{"network_dim": 8}'
        with tempfile.TemporaryDirectory() as source:
            cases = [
                (validation._extra_schema, (0,), {"network_dim": ["int", None, "cli"]}, None),
                (validation._accelerators, (0,), [True, False], None),
                (validation._native_overrides, (settings, 0), "invalid network_args", settings),
            ]
            for probe, args, expected, stdin in cases:
                with self.subTest(probe=probe.__name__), patch.object(validation.subprocess, "run") as run:
                    run.return_value.stdout = "engine log\n" + json.dumps(expected)
                    self.assertEqual(probe(sys.executable, source, *args), expected)
                    options = run.call_args.kwargs
                    self.assertEqual(options["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self.assertEqual(options.get("input"), stdin)

    @unittest.skipUnless(PYTHON.is_file() and (ENGINE / "uv.lock").is_file(), "real Anima lockfile required")
    def test_fresh_install_and_repair_target_the_engine_python_and_export_lycoris(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "anima_lora"
            (engine / "library").mkdir(parents=True)
            (engine / "train.py").touch()
            for name in ("pyproject.toml", "uv.lock"):
                shutil.copyfile(ENGINE / name, engine / name)
            service = InstallationService(StateStore(root / "data"), None, ROOT)
            target = engine / ".lp/python/python.exe"
            for fresh in (True, False):
                with self.subTest(fresh=fresh):
                    if not fresh:
                        target.parent.mkdir(parents=True)
                        target.touch()
                    _, python, _, plan = service._build_install(
                        engine, AnimaLoraAdapter(), sys.executable, "existing", None, None, None)
                    install = plan.commands[0]
                    self.assertEqual(Path(python).resolve(), target.resolve())
                    self.assertEqual(Path(install[2]).name, "install_uv_project.py")
                    self.assertEqual(Path(install[install.index("--python") + 1]).resolve(), target.resolve())
            uv = install[install.index("--uv") + 1]
            exported = subprocess.run([uv, "export", "--frozen", "--offline", "--format", "requirements.txt",
                                       "--no-hashes", "--no-emit-project"], cwd=engine,
                                      capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
            self.assertIn("lycoris-lora==4.0.0", exported.stdout.splitlines())

    def test_incompatible_moss_distillation_settings_are_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "scripts/moss").mkdir(parents=True)
            (source / "scripts/moss/train.py").touch()
            instance = SimpleNamespace(source_path=source, installation_id="test")
            adapter = AnimaLoraAdapter()
            for expected, other in (
                ("moss_distill", {"moss_delta_rows": "text"}),
                ("moss_distill", {"moss_trigger_registers": 2}),
                ("moss_distill", {"moss_ref_adapters": "all"}),
                ("moss_distill_extra_sigma", {"moss_distill_extra_sigma": 0.2, "moss_repa_weight": 0.1}),
                ("moss_distill_balance", {"moss_distill_balance": True, "moss_distill_add": 0.1}),
                ("moss_distill_gate", {"moss_distill_gate": True, "moss_distill_mix": 0.1}),
                ("moss_distill_gate", {"moss_distill_gate": True, "moss_distill_sigma_ramp": 0.1}),
                ("moss_int8", {"moss_algo": "loha", "moss_int8": "stream"}),
            ):
                with self.subTest(options=other):
                    cfg = dict(architecture="moss_image", params={"moss_distill": "base", **other})
                    self.assertTrue(any(i.field == expected and "版本引擎不支持" not in i.message
                                        for i in adapter.validate(instance, cfg)))

    def test_index_preserves_moss_epoch_step_and_final_even_with_a_numbered_task_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = StateStore(root / "data")
            key, name = "a" * 24, "study-step123"
            output = root / "output"
            job = store.root / "jobs" / key
            job.mkdir(parents=True)
            task = dict(task_id=key, engine_id="anima_lora", architecture="moss_image", name=name,
                        output_dir=str(output), draft={"base_model_path": ""})
            (job / "job.json").write_text(json.dumps(task), encoding="utf-8")
            (output / name).mkdir(parents=True)
            header = json.dumps({"weight": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}}).encode()
            for tag in ("", "_e1", "_s25", "_raw", "_ema0.05"):
                dest = output / name if tag else output
                (dest / f"{name}{tag}.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0")
            register_artifacts(store, task)
            flags = {Path(row["file_name"]).stem.removeprefix(name): (row["step"], row["final"], row["complete"])
                     for row in store.list("artifact")}
            self.assertEqual(flags, {"": (None, True, True), "_e1": (None, False, True), "_s25": (25, False, True),
                                     "_raw": (None, True, True), "_ema0.05": (None, True, True)})

    def test_moss_step_snapshot_and_sample_text_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "job.json").write_text(json.dumps(dict(architecture="moss_image", name="demo",
                                                          output_dir=tmp)), encoding="utf-8")
            for relative in ("demo.safetensors", "demo/demo_s25.safetensors", "demo/samples/e1_1.png",
                             "_sample_te_cache/conditioning.safetensors"):
                path = root / relative
                path.parent.mkdir(exist_ok=True, parents=True)
                path.touch()
            files = {a.path.name: (a.kind, a.step) for a in AnimaLoraAdapter().collect_artifacts(root)}
            self.assertEqual(files, {"demo.safetensors": ("checkpoint", None),
                                     "demo_s25.safetensors": ("checkpoint", 25), "e1_1.png": ("sample", None)})

    def test_older_checkout_does_not_offer_or_emit_new_moss_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts/moss").mkdir(parents=True)
            (root / "scripts/moss/train.py").touch()
            instance = SimpleNamespace(source_path=root, installation_id="test", python_executable=PYTHON)
            adapter = AnimaLoraAdapter()
            caps = adapter.capabilities(instance)
            self.assertFalse({p["key"] for p in caps["params"]} & VERSIONED_KEYS)
            config = dict(architecture="moss_image", dataset_path=tmp, output_dir=tmp, params={})
            cache, train = adapter.preview_commands(instance, config, root / "config.native.toml")
            self.assertNotIn("--text_boxes", cache)
            self.assertNotIn("--ema_card", train)
            config["params"]["moss_distill"] = "base"
            self.assertIn("moss_distill", {i.field for i in adapter.validate(instance, config)})

    @unittest.skipUnless(PYTHON.is_file() and (ENGINE / "configs/moss/lora.toml").is_file(), "installed Anima required")
    def test_official_profiles_and_invalid_effective_overrides(self):
        adapter = AnimaLoraAdapter()
        install = SimpleNamespace(source_path=ENGINE, installation_id="probe", python_executable=PYTHON)
        ordinary = {x["key"]: x for x in adapter.capabilities_for_architecture(install, "anima")["params"]}
        moss = {x["key"]: x for x in adapter.capabilities_for_architecture(install, "moss_image")["params"]}
        self.assertNotIn("moss_rank", ordinary)
        self.assertNotIn("vocab_pack", moss)
        self.assertEqual([ordinary[key]["default"] for key in ("torch_compile", "learning_rate", "weighting_scheme", "target_res")],
                         [True, 0.00005, "uniform", "1024 896"])
        self.assertEqual([moss[key]["default"] for key in ("moss_compile", "moss_blocks_to_swap",
                                                           "moss_activation_memory_budget", "moss_clause_hw")],
                         [True, 20, 0.4, 1.0])
        for key in ("vae", "qwen3", "pretrained_model_name_or_path"):
            self.assertFalse(ordinary[key]["default"])
        self.assertFalse(moss["moss_model_dir"]["default"])
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "MOSS-Image-Base"
            for name in ("transformer", "text_encoder", "tokenizer", "vae"):
                (model / name).mkdir(parents=True)
            conf = dict(architecture="moss_image", dataset_path=tmp, output_dir=tmp, params={"moss_model_dir": str(model)})
            train = adapter.preview_commands(install, conf, model.parent / "config.native.toml")[-1]
            self.assertEqual(train[train.index("--model") + 1], "base")
            missing = str(Path(tmp) / "missing-model.safetensors")
            unresolved = adapter.validate(install, dict(architecture="anima", base_model_path=missing,
                                                       params={"qwen3": missing, "vae": missing}))
            self.assertTrue({"pretrained_model_name_or_path", "qwen3", "vae"} <=
                            {issue.field for issue in unresolved if missing in issue.message and issue.severity=="error"})

        def errors(architecture, params):
            config = {"architecture": architecture, "params": params}
            return {i.field for i in adapter.validate(install, config) if i.severity == "error"}

        self.assertTrue({"vae", "qwen3", "pretrained_model_name_or_path"} <= errors("anima", {}))
        for params, key in [
            ({"extra_params": "--loss_type imaginary"}, "extra_params"),
            ({"extra_params": "--down_init broken"}, "extra_params"),
            ({"network_args": "down_init=broken"}, "network_args"),
            ({"extra_params": "--timestep_sampling nope"}, "extra_params"),
            ({"extra_params": "--network_dim false"}, "extra_params"),
            ({"extra_params": "--unknown_for_anima true"}, "extra_params"),
            ({"extra_params": "--network_train_text_encoder_only"}, "network_train_text_encoder_only"),
            ({"extra_params": "--unsloth_offload_checkpointing --blocks_to_swap 4"}, "unsloth_offload_checkpointing"),
            ({"vocab_pack": "does-not-exist/non-default"}, "vocab_pack"),
            ({"target_res": "1000"}, "target_res"),
            ({"t5_max_token_length": 256}, "t5_max_token_length"),
            ({"clip_mode": "fixed"}, "clip_sigma_bands"),
            ({"compile_inductor_mode": "anything"}, "compile_inductor_mode"),
        ]:
            with self.subTest(architecture="anima", params=params):
                self.assertIn(key, errors("anima", params))
        for params, key in [
            ({"moss_distill": "ema"}, "moss_clause_hw"),
            ({"moss_ref_adapters": "kv", "moss_ref_isolate": False}, "moss_ref_adapters"),
            ({"moss_cache_skip_text": True}, "moss_cache_skip_text"),
            ({"moss_sample_at_first": True, "moss_sample_prompts": "missing-prompt.json"}, "moss_sample_prompts"),
            ({"moss_ema_extra": "1.5"}, "moss_ema_extra"),
        ]:
            with self.subTest(architecture="moss_image", params=params):
                self.assertIn(key, errors("moss_image", params))
        with tempfile.TemporaryDirectory() as tmp:
            native = Path(tmp) / "config.native.toml"
            params = {"extra_params": "--vocab_pack 'custom/prefix' --cache_latents false",
                      "torch_compile": False}
            config = dict(architecture="anima", name="probe", params=params,
                          dataset_path=tmp, output_dir=str(Path(tmp) / "output"))
            adapter.write_native_config(install, config, native.parent)
            commands = adapter.preview_commands(install, config, native)
            self.assertNotIn("cache_latents.py", [Path(c[2]).name for c in commands])
            te = next(c for c in commands if Path(c[2]).name == "cache_text_embeddings.py")
            self.assertEqual(te[te.index("--vocab_pack") + 1], "custom/prefix")
            import tomllib
            values = tomllib.loads(native.read_text(encoding="utf-8"))
            self.assertEqual((values["vocab_pack"], values["torch_compile"], values["use_vae_cache"]),
                             ("custom/prefix", False, False))

    @unittest.skipUnless(PYTHON.is_file(), "installed Anima required")
    def test_reused_text_cache_rejects_mismatched_pack_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = """
from pathlib import Path
import runpy, sys, torch
from safetensors.torch import save_file
from library.preprocess.text import _te_cache_path
root = Path(sys.argv[1])
staged,cache = root / 'resized', root / 'cache'
staged.mkdir()
cache.mkdir()
image = staged / 'probe.png'
image.touch()
target = _te_cache_path(image,cache,staged)
target.parent.mkdir(parents=True,exist_ok=True)
save_file({'value':torch.zeros(1)},str(target),metadata={'vocab_pack_sha':'old-pack'})
before = target.read_bytes()
check = runpy.run_path(sys.argv[2])['verify']
try:
    check(str(staged),str(staged),str(cache),'','1')
except SystemExit as e:
    assert str(target) in str(e)
else:
    raise AssertionError('must reject cached embeddings from another vocabulary')
assert target.read_bytes() == before
save_file({'value':torch.zeros(1)},str(target))
check(str(staged),str(staged),str(cache),'','1')
"""
            helper = Path(__file__).resolve().parents[1] / "adapters/anima_lora/cache_checks.py"
            result = subprocess.run([str(PYTHON), "-B", "-X", "utf8", "-c", script, str(root), str(helper)],
                                    cwd=ENGINE, encoding="utf-8", capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(PYTHON.is_file() and (ENGINE / "library/moss/caching/sources.py").is_file(),
                         "current Anima checkout with its own interpreter required")
    def test_installed_engine_resolves_both_native_pipelines(self):
        instance = SimpleNamespace(source_path=ENGINE, installation_id="test", python_executable=PYTHON)
        adapter = AnimaLoraAdapter()
        caps = {p["key"]: p for p in adapter.capabilities(instance)["params"]}
        self.assertTrue(VERSIONED_KEYS.issubset(caps))
        self.assertNotIn("ema_decay", caps)
        self.assertIn("tags", [v["value"] for v in caps["moss_caption_format"]["options"]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(architecture="anima", name="test", output_name="test", output_dir=str(root / "output"),
                          dataset_path=tmp, base_model_path=str(root / "model.safetensors"),
                          params=dict(vae=str(root / "vae"), qwen3=str(root / "qwen"), ema_card=False))
            native = adapter.write_native_config(instance, config, root)
            resize, captions, latent, text, train = adapter.preview_commands(instance, config, native)
            self.assertIn("preprocess-captions", captions)
            self.assertEqual(adapter.build_launch(instance, native).environment_overrides["CONFIG_FILE"], str(root / "preprocess.toml"))
            config.update(architecture="moss_image", params=dict(
                moss_model_dir=str(root / "MOSS-Image-SFT"), moss_mod_rank=12, moss_text_boxes=False,
                moss_caption_format="tags", moss_save_every_steps=301, moss_ema_card=False,
                moss_ref_scale=1.0,  # Imported from an older LP form
            ))
            cache, moss = adapter.preview_commands(instance, config, native)
            check = """
import json, sys
from unittest.mock import patch
from library.anima.trainer.cli import setup_parser, build_network_extras
from library.config import schema
from library.config.io import read_config_from_file
from library.moss.requests import CacheRequest, TrainRequest
from scripts.tasks import preprocess
import argparse, runpy
from dataclasses import asdict
from pathlib import Path
resize, latent, text, train, cache, moss = json.load(sys.stdin)
with patch.object(preprocess, '_execute') as resize_stage:
    preprocess.cmd_preprocess_resize(resize)
    req = resize_stage.call_args.args[1]
    assert str(req.src) == resize[resize.index('--src') + 1] and str(req.dst) == resize[resize.index('--dst') + 1]
import os
import scripts.tasks._common as common
saved = Path(train[train.index("--config_file") + 1]).parent / "preprocess.toml"
with patch.dict(os.environ, {"CONFIG_FILE": str(saved)}), patch.object(common, "_PATH_OVERRIDES_CACHE", None), \
     patch.object(preprocess, "_ensure_danbooru_tags"), patch.object(preprocess, "_execute") as caption_stage:
    preprocess.cmd_preprocess_captions([])
    for call in caption_stage.call_args_list:
        _, opts = call.args
        assert str(opts.src) == str(req.src) and str(opts.dst) == str(req.dst)
    assert any(c.args[0] == "correct" for c in caption_stage.call_args_list)
class Parsed(Exception):
    pass
read = argparse.ArgumentParser.parse_args
for stage in [latent, text]:
    def parse_only(parser, *args, **kwargs):
        values = read(parser, *args, **kwargs)
        if hasattr(values, 'vocab_pack'):
            assert values.vocab_pack == 'models/vocab_packs/anima_cjk_vocab_pack_jp_v1'
            assert values.caption_shuffle_variants == 4 and values.caption_tag_dropout_rate == .1 and values.dit
        raise Parsed
    sys.argv = stage[2:]
    try:
        with patch.object(argparse.ArgumentParser, 'parse_args', parse_only):
            runpy.run_path(stage[2], run_name='__main__')
    except Parsed:
        pass
    else:
        raise AssertionError('native parser was not reached')
parser = setup_parser()
schema.populate_schema(parser, extras=build_network_extras())
args = read_config_from_file(parser.parse_args(train), parser, argv=train)
assert args.ema_window == 0.1 and args.ema_card is False
assert (args.learning_rate, args.gradient_accumulation_steps, args.clip_mode, args.use_repa) == (5e-5, 2, 'median', False)
c, t = CacheRequest.from_argv(cache), TrainRequest.from_argv(moss)
assert c.caption_format == 'tags' and c.text_boxes is False and c.ref_scale == 1
assert (t.model, t.mod_rank, t.save_every_steps, t.ema_card) == ('sft', 12, 301, False)
assert (t.blocks_to_swap, t.activation_memory_budget, t.clause_hw, t.compile) == (20, 0.4, 1.0, True)
official = asdict(TrainRequest.from_argv(['--config', str(Path.cwd() / "configs/moss/lora.toml")]))
selected = asdict(t)
for k in official.keys() - {'cache', 'output', 'model_dir', 'model', 'mod_rank', 'save_every_steps', 'ema_card'}:
    assert official[k] == selected[k], (k, official[k], selected[k])
assert args.target_res == [1024,896] and req.target_res == (1024,896)
assert args.torch_compile and args.vocab_pack == 'models/vocab_packs/anima_cjk_vocab_pack_jp_v1'
with patch('library.config.io._write_config_snapshot'):
    opts = ['--method','lora','--preset','default']
    official_anima = read_config_from_file(parser.parse_args(opts), parser, argv=opts)
for k in vars(official_anima).keys() - {'pretrained_model_name_or_path', 'vae', 'qwen3', 'mask_dir',
    'output_dir', 'output_name', 'logging_dir', '_resolved_run_log_dir', 'config_file', 'dataset_config',
    'resized_image_dir', 'lora_cache_dir', 'ema_card'}:
    v,w = getattr(args,k,None),getattr(official_anima,k)
    assert v == w or {str(v), str(w)} == {'None',''}, (k,v,w)
"""
            result = subprocess.run([str(PYTHON), "-B", "-X", "utf8", "-c", check], cwd=ENGINE,
                                    input=json.dumps([resize[4:], latent, text, train[3:], cache[3:], moss[3:]]),
                                    text=True, encoding="utf-8", capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
