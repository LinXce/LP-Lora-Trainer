"""No GPU, upstream engines, GUI or persistent processes are needed by these tests."""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from PIL import Image
from tests.local_client import LocalClient
from tests.engine_fixtures import cleanup_directory, engine_python
from app import INSTALLATION_WORKFLOW_VERSION
from app.config import AppPaths
from app.api.models import TrainingInput
from app.api.server import create_app
from app.services.common import ServiceError, contained, identity
from app.services.datasets import DatasetService
from app.services.engines import EngineService, revision
from app.services.training import TrainingService
from app.storage.state import StateStore
from adapters.registry import ADAPTERS, domain
from adapters.musubi_tuner.params import PARAMS as MUSUBI_PARAMS
from desktop.bridge import DesktopBridge
from desktop.launcher import backend_ready
from supervisor.worker import Worker, ProcessLock, checkpoint_complete, register_artifacts

INFO = dict(python_version=sys.version.split()[0], executable=sys.executable, torch_version="test", cuda_wheel="test",
            cuda_available=True, accelerate=True, transformers=True, safetensors=True, diffusers=True, yaml=True, toml=True)
DIAG = "import sys,json; info=" + repr(INFO) + "; info.update(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix); print('LP_DIAG='+json.dumps(info))"
FAKE_ENGINE = r"""
import sys, json, struct, tomllib
from pathlib import Path
config = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))
print('steps: 100%|xxx| 2/2 [00:01<00:00, 2.00it/s, avr_loss=0.25]', flush=True)
out = Path(config['output_dir']); out.mkdir(exist_ok=True)
header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
(out / 'lora.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\x00' * 4)
"""


def make_engine(path, script=FAKE_ENGINE, nested=False):
    root = path / "sd-scripts" if nested else path
    (root / "library").mkdir(parents=True)
    (root / "library" / "train_util.py").write_text("# fake fixture\n", encoding="utf-8")
    (root / "train_network.py").write_text(script, encoding="utf-8")
    (root / "sdxl_train_network.py").write_text(script, encoding="utf-8")
    return root


# Cache stages have no --config_file; only the training stage writes an artifact.
MUSUBI_SCRIPT = r"""
import json, struct, sys
from pathlib import Path
Path(__file__ + '.ran').write_text('ok', encoding='utf-8')
print('steps: 100%|xxx| 2/2 [00:01<00:00, 2.00it/s, avr_loss=0.25]', flush=True)
if '--config_file' in sys.argv:
    import tomllib
    config = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))
    out = Path(config['output_dir']); out.mkdir(exist_ok=True)
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (out / 'lora.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\x00' * 4)
"""


MUSUBI_FAIL_TE = r"""
import sys
from pathlib import Path
Path(__file__ + '.ran').write_text('ok', encoding='utf-8')
sys.exit(7 if 'cache_text_encoder' in sys.argv[0] else 0)
"""


def make_musubi(path, script=MUSUBI_SCRIPT):
    (path / "src" / "musubi_tuner").mkdir(parents=True)
    (path / "pyproject.toml").write_text('[project]\nname = "musubi-tuner"\n', encoding="utf-8")
    (path / "wan_train_network.py").write_text("# entry\n", encoding="utf-8")
    for name in ("wan_train_network.py", "wan_cache_latents.py", "wan_cache_text_encoder_outputs.py"):
        (path / "src" / "musubi_tuner" / name).write_text(script, encoding="utf-8")
    return path


def write_safetensors(path):
    """A complete, structurally valid single-tensor safetensors file."""
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    return Path(path)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-tests-")
        self.root = Path(self.temp.name)
        self.engine_root = self.root / "engine"
        self.engine_root.mkdir()
        self.store = StateStore(self.root / "data")
        self.engines = EngineService(self.store, lambda: {"engine_root": str(self.engine_root)})
        self.datasets = DatasetService(self.store)
        self.training = TrainingService(self.store, self.engines, self.datasets)
        self.dataset_path = self.root / "images"
        self.dataset_path.mkdir()
        Image.new("RGB", (128, 128), "red").save(self.dataset_path / "one.png")
        (self.dataset_path / "one.txt").write_text("red", encoding="utf-8")
        self.dataset = self.datasets.add(str(self.dataset_path), "Test images")
        self.datasets.scan(self.dataset["dataset_id"])
        self.model = self.root / "models" / "base.safetensors"
        self.model.parent.mkdir()
        self.model.write_bytes(b"fixture")
        self.diag_patch = patch("app.services.engines.DIAGNOSTIC_CODE", DIAG)
        self.diag_patch.start()
        # Every engine child the supervisor spawns is tracked so tearDown can reap
        # it even when a test fails before its own cleanup. A surviving training
        # process would hold its process group and either data-root lock, and the
        # next real launch would then look like "the app will not open".
        self.spawned = []
        real_popen = subprocess.Popen

        def tracking_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            self.spawned.append(process)
            return process

        self.popen_patch = patch("supervisor.worker.subprocess.Popen", side_effect=tracking_popen)
        self.popen_patch.start()

    def tearDown(self):
        self.popen_patch.stop()
        self.reap_children()
        self.diag_patch.stop()
        self.cleanup_temp()

    def cleanup_temp(self):
        """Remove the fixture directory, retrying while Windows still holds files."""
        cleanup_directory(self.temp.name)

    def reap_children(self):
        """Force-stop every child still alive after a test, including its tree."""
        for process in self.spawned:
            if process.poll() is not None:
                continue
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                try: process.kill()
                except OSError: pass
        self.spawned = []

    def ready(self, name="copied source", script=FAKE_ENGINE, nested=False):
        make_engine(self.engine_root / name, script, nested)
        self.engines.rescan()
        record = next(r for r in self.engines.list() if r["label"] == name)
        self.engines.bind(record["installation_id"], str(engine_python(Path(record["source_path"]))))
        return self.engines.get(record["installation_id"])

    def draft(self, installation):
        return dict(name="Test task", installation_id=installation["installation_id"], architecture="sd1",
                    base_model_path=str(self.model), dataset_id=self.dataset["dataset_id"], output_dir=str(self.root / "outputs"), params={})

    def ready_musubi(self, name="musubi", script=MUSUBI_SCRIPT):
        make_musubi(self.engine_root / name, script)
        self.engines.rescan()
        record = next(r for r in self.engines.list() if r["label"] == name)
        self.engines.bind(record["installation_id"], str(engine_python(Path(record["source_path"]))))
        return self.engines.get(record["installation_id"])

    def musubi_draft(self, installation, params=None):
        return dict(name="Musubi task", installation_id=installation["installation_id"], architecture="wan",
                    base_model_path=str(self.model), dataset_id=self.dataset["dataset_id"],
                    output_dir=str(self.root / "outputs" / "musubi"),
                    params={"vae": str(self.root / "vae.safetensors"), "wan_t5": str(self.root / "t5.safetensors"),
                            **(params or {})})

    def musubi_arch_engine(self, name, scripts):
        """A musubi clone exposing exactly the given entry/cache scripts."""
        path = self.engine_root / name
        (path / "src" / "musubi_tuner").mkdir(parents=True)
        (path / "pyproject.toml").write_text('[project]\nname = "musubi-tuner"\n', encoding="utf-8")
        for script in scripts:
            (path / "src" / "musubi_tuner" / script).write_text("# entry\n", encoding="utf-8")
        # Detection requires a mirrored entry stub at the clone root, as upstream ships.
        entry = next(s for s in scripts if s.endswith("_train_network.py"))
        (path / entry).write_text("# entry\n", encoding="utf-8")
        self.engines.rescan()
        record = next(r for r in self.engines.list() if r["label"] == name)
        self.engines.bind(record["installation_id"], str(engine_python(Path(record["source_path"]))))
        return self.engines.get(record["installation_id"])

    def musubi_config(self, installation, architecture, params):
        return dict(name="Musubi task", installation_id=installation["installation_id"], architecture=architecture,
                    base_model_path=str(self.model), dataset_id=self.dataset["dataset_id"],
                    output_dir=str(self.root / "outputs" / architecture), dataset_path=str(self.dataset_path),
                    params=params)

    def render_musubi(self, installation, architecture, params, name="job"):
        job = self.root / f"{name}-{architecture}"
        job.mkdir()
        config = self.musubi_config(installation, architecture, params)
        native = ADAPTERS["musubi_tuner"].write_native_config(domain(installation), config, job)
        return job, native, tomllib.loads(native.read_text(encoding="utf-8"))

    def test_git_clone_layout_detection_and_legacy_recovery(self):
        # These fixtures intentionally resemble source-only git clones, including
        # Kohya's uninitialized sd-scripts submodule.
        kohya = self.engine_root / "kohya_ss"
        (kohya / "kohya_gui").mkdir(parents=True)
        (kohya / "kohya_gui.py").write_text("# wrapper\n", encoding="utf-8")
        (kohya / ".gitmodules").write_text(
            '[submodule "sd-scripts"]\npath = sd-scripts\nurl = https://github.com/kohya-ss/sd-scripts.git\n',
            encoding="utf-8",
        )
        (kohya / ".git").mkdir()

        musubi = self.engine_root / "musubi-tuner"
        (musubi / "src" / "musubi_tuner").mkdir(parents=True)
        (musubi / "pyproject.toml").write_text(
            '[project]\nname = "musubi-tuner"\nversion = "0.3.6"\n', encoding="utf-8"
        )
        (musubi / "flux_train_network.py").write_text("# entry\n", encoding="utf-8")
        (musubi / ".git").mkdir()

        toolkit = self.engine_root / "ai-toolkit"
        (toolkit / "toolkit").mkdir(parents=True)
        (toolkit / "jobs").mkdir()
        (toolkit / "run.py").write_text("# entry\n", encoding="utf-8")
        (toolkit / ".git").mkdir()

        records = self.engines.rescan()
        expected = {"kohya_ss": "kohya", "musubi-tuner": "musubi_tuner", "ai-toolkit": "ai_toolkit"}
        for label, engine_id in expected.items():
            record = next(r for r in records if r["label"] == label)
            self.assertEqual(record["engine_id"], engine_id)
            self.assertEqual(record["candidate_engines"], [engine_id])

        # Simulate records written by the old detector before clone-aware rules.
        for record in records:
            self.store.patch("engine", record["installation_id"], {
                "engine_id": None,
                "candidate_engines": [],
                "issues": ["无法唯一识别引擎类型，请手动确认"],
            })

        repaired = self.engines.rescan()
        for label, engine_id in expected.items():
            record = next(r for r in repaired if r["label"] == label)
            self.assertEqual(record["engine_id"], engine_id)
            self.assertEqual(record["candidate_engines"], [engine_id])
            self.assertEqual(record["state"], "discovered")
            self.assertNotIn("无法唯一识别引擎类型，请手动确认", record["issues"])
    def test_rescan_keeps_type_when_detection_is_temporarily_empty(self):
        make_engine(self.engine_root / "temporary detector failure")
        records = self.engines.rescan()
        record = next(r for r in records if r["label"] == "temporary detector failure")
        self.assertEqual(record["engine_id"], "kohya")

        # A failed/incomplete static pass must not erase the durable type.
        with patch.object(self.engines, "_detect_candidates", return_value=[]):
            refreshed = self.engines.rescan()
        record = next(r for r in refreshed if r["label"] == "temporary detector failure")
        self.assertEqual(record["engine_id"], "kohya")
        self.assertEqual(record["candidate_engines"], [])

    def test_rescan_reuses_record_when_path_spelling_changed(self):
        make_engine(self.engine_root / "path spelling")
        record = next(r for r in self.engines.rescan() if r["label"] == "path spelling")

        # Simulate a legacy record whose id was generated from another path
        # spelling, while its source still resolves to the same directory.
        legacy_id = "legacy-path-id"
        legacy = dict(record, installation_id=legacy_id,
                      source_path=str(Path(record["source_path"]) / "."),
                      candidate_engines=[], engine_id="kohya")
        self.store.delete("engine", record["installation_id"])
        self.store.put("engine", legacy_id, legacy)

        refreshed = self.engines.rescan()
        current = next(r for r in refreshed if r["label"] == "path spelling")
        self.assertEqual(current["installation_id"], legacy_id)
        self.assertEqual(current["engine_id"], "kohya")
        self.assertNotEqual(current["state"], "missing")
        self.assertEqual(len([r for r in refreshed if r["label"] == "path spelling"]), 1)

    def test_copy_detection_and_shared_adapter(self):
        a = self.ready()
        b = self.ready("older arbitrary folder", nested=True)
        self.assertNotEqual(a["installation_id"], b["installation_id"])
        self.assertEqual(a["engine_id"], b["engine_id"])
        self.assertEqual(a["management_mode"], "user_managed")
        self.assertEqual(a["verification"], "experimental")
        self.engines.set_default(b["installation_id"])
        self.assertTrue(next(r for r in self.engines.list() if r["installation_id"] == b["installation_id"])["is_default"])
        self.assertEqual(self.engines.capabilities(a["installation_id"])["architectures"], ["sd1", "sd2", "sdxl"])

    def test_source_change_invalidates_ready(self):
        a = self.ready()
        p = Path(a["source_path"]) / "train_network.py"
        p.write_text(p.read_text() + "\n# updated source", encoding="utf-8")
        self.assertNotEqual(revision(Path(a["source_path"])), a["revision"])
        self.engines.rescan()
        self.assertEqual(self.engines.get(a["installation_id"])["state"], "discovered")
        self.assertFalse(self.training.validate(self.draft(a))["ok"])

    def test_unknown_and_ai_toolkit_never_fake_training(self):
        (self.engine_root / "unknown").mkdir()
        root = self.engine_root / "toolkit-copy"
        (root / "toolkit").mkdir(parents=True)
        (root / "jobs").mkdir()
        (root / "run.py").write_text("raise RuntimeError('must not be run')", encoding="utf-8")
        records = self.engines.rescan()
        unknown = next(r for r in records if r["label"] == "unknown")
        self.assertIsNone(unknown["engine_id"])
        with self.assertRaises(ServiceError): self.engines.confirm(unknown["installation_id"], "kohya")
        ai = next(r for r in records if r["label"] == "toolkit-copy")
        self.assertEqual(ai["engine_id"], "ai_toolkit")
        # AI Toolkit now advertises its base-model types and parameters, but must
        # stay preview-only until its training submission is adapted.
        capabilities = self.engines.capabilities(ai["installation_id"])
        self.assertTrue(capabilities["architectures"])
        self.assertTrue(capabilities["params"])
        self.assertFalse(capabilities["submittable"])
        self.assertFalse(self.training.validate(self.draft(ai))["ok"])
        self.assertFalse(self.training.validate(self.draft(ai))["submittable"])

    def test_diagnosis_failure_is_persisted(self):
        a = self.ready()
        with patch("app.services.engines.DIAGNOSTIC_CODE", "raise RuntimeError('missing dependencies')"):
            with self.assertRaises(ServiceError): self.engines.diagnose(a["installation_id"])
        self.assertEqual(self.engines.get(a["installation_id"])["state"], "failed")

    def test_dataset_scan_caption_backup_thumbnail_and_filters(self):
        Image.new("RGB", (128, 128), "red").save(self.dataset_path / "duplicate.png")
        (self.dataset_path / "broken.png").write_bytes(b"not an image")
        key = self.dataset["dataset_id"]
        self.datasets.scan(key)
        summary = self.datasets.get(key)
        self.assertEqual(summary["image_count"], 3)
        self.assertEqual({i["kind"]: i["count"] for i in summary["issues"]}["duplicate"], 2)
        self.assertEqual(self.datasets.images(key, 0, 1, "corrupt")["total"], 1)
        one = next(i for i in self.datasets.images(key, 0, 48)["items"] if i["file_name"] == "one.png")
        self.datasets.caption(key, one["image_id"], "new caption")
        self.assertEqual((self.dataset_path / "one.txt").read_text(), "new caption")
        backups = list((self.store.root / "backups" / "captions").rglob("*.txt"))
        self.assertEqual(backups[0].read_text(), "red")
        with Image.open(self.datasets.thumbnail(key, one["image_id"])) as image:
            self.assertEqual(image.format, "JPEG")
        self.assertNotIn("path", one)

    def test_containment_rejects_escape(self):
        with self.assertRaises(ServiceError): contained(self.root / "outside.txt", self.dataset_path)
        with self.assertRaises(ServiceError): self.datasets.add("relative", "relative")

    def test_native_toml_and_type_validation(self):
        a = self.ready()
        d = self.draft(a)
        self.assertTrue(self.training.validate(d)["ok"])
        for params in ({"network_dim": True}, {"network_dim": 1.5}, {"learning_rate": "oops"}, {"unsupported": 1}, {"resolution": 513}):
            self.assertFalse(self.training.validate({**d, "params": params})["ok"], params)
        job = self.root / "configs"; job.mkdir()
        config = {**d, "architecture": "sdxl", "dataset_path": str(self.dataset_path)}
        native = ADAPTERS["kohya"].write_native_config(domain(a), config, job)
        values = tomllib.loads(native.read_text(encoding="utf-8"))
        self.assertTrue(values["network_train_unet_only"])
        self.assertEqual(tomllib.loads((job / "dataset.toml").read_text())["datasets"][0]["subsets"][0]["image_dir"], str(self.dataset_path))
        self.assertEqual(Path(ADAPTERS["kohya"].build_launch(domain(a), native).argv[2]).name, "sdxl_train_network.py")
        self.assertIs(TrainingInput(**{**d, "params": {"network_dim": True}}).params["network_dim"], True)

    def test_base_model_catalog_and_trainer_selection(self):
        a = self.ready()
        catalog = {m["id"]: m for m in self.engines.base_models()}
        self.assertEqual(set(catalog), {"sd1", "sd2", "sdxl"})
        self.assertIn("kohya", catalog["sdxl"]["engines"])
        preview = self.training.validate({**self.draft(a), "architecture": "sdxl"})
        self.assertTrue(preview["ok"])
        self.assertTrue(preview["submittable"])
        self.assertIn("network_train_unet_only", preview["native_config"])
        self.assertEqual(Path(preview["argv"][2]).name, "sdxl_train_network.py")

    def test_musubi_pipeline_preview_and_required_paths(self):
        a = self.ready_musubi()
        caps = self.engines.capabilities(a["installation_id"])
        self.assertEqual(caps["architectures"], ["wan"])
        self.assertTrue(caps["submittable"])
        # Missing required model paths must be reported instead of silently submitted.
        bad = self.training.validate({**self.musubi_draft(a), "params": {}})
        self.assertFalse(bad["ok"])
        self.assertIn("vae", [i["field"] for i in bad["issues"] if i["level"] == "error"])
        preview = self.training.validate(self.musubi_draft(a))
        self.assertTrue(preview["ok"])
        self.assertTrue(preview["submittable"])
        self.assertEqual(len(preview["pipeline"]), 3)
        self.assertEqual(Path(preview["pipeline"][0][2]).name, "wan_cache_latents.py")
        self.assertEqual(Path(preview["pipeline"][1][2]).name, "wan_cache_text_encoder_outputs.py")
        self.assertEqual(Path(preview["pipeline"][2][2]).name, "wan_train_network.py")
        self.assertEqual(preview["argv"], preview["pipeline"][-1])

    def test_musubi_native_configs_and_launch(self):
        a = self.ready_musubi()
        d = self.musubi_draft(a)
        job = self.root / "musubi-job"; job.mkdir()
        config = {**d, "dataset_path": str(self.dataset_path), "output_dir": str(self.root / "outputs" / "j1")}
        native = ADAPTERS["musubi_tuner"].write_native_config(domain(a), config, job)
        dataset = tomllib.loads((job / "dataset.toml").read_text(encoding="utf-8"))
        self.assertEqual(dataset["datasets"][0]["image_directory"], str(self.dataset_path))
        self.assertIn("cache_directory", dataset["datasets"][0])
        values = tomllib.loads(native.read_text(encoding="utf-8"))
        self.assertEqual(values["network_module"], "networks.lora_wan")
        self.assertEqual(values["output_dir"], config["output_dir"])
        self.assertEqual(values["dit"], str(self.model))
        stages = json.loads((job / "stages.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stages), 3)
        launch = ADAPTERS["musubi_tuner"].build_launch(domain(a), native)
        self.assertEqual(len(launch.pre_argv), 2)
        self.assertEqual([list(c) for c in launch.commands()], stages)
        self.assertEqual(list(ADAPTERS["musubi_tuner"].preview_argv(domain(a), config, native)), stages[-1])

    def test_worker_runs_musubi_cache_then_training(self):
        a = self.ready_musubi()
        t = self.training.submit(self.musubi_draft(a))
        recorded = json.loads((self.store.root / "jobs" / t["task_id"] / "command.json").read_text(encoding="utf-8"))
        self.assertEqual(len(recorded["commands"]), 3)
        Worker(self.store).execute(self.training.task(t["task_id"]))
        self.assertEqual(self.training.task(t["task_id"])["state"], "succeeded")
        package = Path(a["source_path"]) / "src" / "musubi_tuner"
        for name in ("wan_cache_latents.py", "wan_cache_text_encoder_outputs.py", "wan_train_network.py"):
            self.assertTrue((package / (name + ".ran")).is_file(), name)
        self.assertEqual(self.training.metrics(t["task_id"])[0]["loss"], 0.25)
        self.assertTrue(self.training.artifacts(t["task_id"]))

    def test_musubi_stage_failure_stops_pipeline(self):
        a = self.ready_musubi(name="musubi-fail", script=MUSUBI_FAIL_TE)
        t = self.training.submit(self.musubi_draft(a))
        Worker(self.store).execute(self.training.task(t["task_id"]))
        task = self.training.task(t["task_id"])
        self.assertEqual(task["state"], "failed")
        self.assertIn("7", task["error_summary"])
        package = Path(a["source_path"]) / "src" / "musubi_tuner"
        self.assertTrue((package / "wan_cache_text_encoder_outputs.py.ran").is_file())
        self.assertFalse((package / "wan_train_network.py.ran").is_file())

    def test_musubi_bucket_resolutions_and_memory_options(self):
        adapter = ADAPTERS["musubi_tuner"]
        a = self.musubi_arch_engine("musubi-krea2", (
            "krea2_train_network.py", "krea2_cache_latents.py", "krea2_cache_text_encoder_outputs.py"))
        params = {
            "vae": str(self.root / "vae.safetensors"),
            "krea2_text_encoder": str(self.root / "te.safetensors"),
            "enable_bucket": True, "bucket_no_upscale": True,
            # Musubi has no --min/max_bucket_reso: the bucket set *is* the resolution list.
            "bucket_resolutions": "960x544\n832x480",
            "gc_cpu_offload": True, "block_swap_h2d_only": True, "block_swap_ring_size": 2,
            "vae_dtype": "bfloat16", "flash3": True, "cuda_allow_tf32": True,
            "krea2_turbo_dit": str(self.root / "turbo.safetensors"), "krea2_turbo_dit_cache": True,
            "krea2_convrot_int8_bwd": "int8", "krea2_turbo_lora_multiplier": 1.2,
            "optimizer_args": "betas=0.9,0.99\nweight_decay=0.01",
        }
        job, native, values = self.render_musubi(a, "krea2", params, name="krea2")
        dataset = tomllib.loads((job / "dataset.toml").read_text(encoding="utf-8"))
        self.assertEqual(dataset["general"]["resolution"], [[960, 544], [832, 480]])
        self.assertTrue(dataset["general"]["enable_bucket"])
        self.assertTrue(dataset["general"]["bucket_no_upscale"])
        # The previously missing VRAM / block-swap / krea2 options now reach the trainer.
        for key, expected in (("gradient_checkpointing_cpu_offload", True), ("block_swap_h2d_only", True),
                              ("block_swap_ring_size", 2), ("vae_dtype", "bfloat16"), ("flash3", True),
                              ("cuda_allow_tf32", True), ("turbo_dit_cache", True),
                              ("convrot_int8_bwd", "int8"), ("turbo_lora_multiplier", 1.2)):
            self.assertEqual(values[key], expected, key)
        # --turbo_dit is a path upstream, not a flag.
        self.assertEqual(values["turbo_dit"], str(self.root / "turbo.safetensors"))
        # Multi-line list arguments become TOML arrays.
        self.assertEqual(values["optimizer_args"], ["betas=0.9,0.99", "weight_decay=0.01"])
        self.assertEqual([i.message for i in adapter.validate(domain(a), self.musubi_config(a, "krea2", params))
                          if i.severity == "error"], [])

    def test_musubi_rejects_malformed_bucket_resolutions(self):
        adapter = ADAPTERS["musubi_tuner"]
        a = self.musubi_arch_engine("musubi-buckets", (
            "wan_train_network.py", "wan_cache_latents.py", "wan_cache_text_encoder_outputs.py"))
        base = {"vae": str(self.root / "vae.safetensors"), "wan_t5": str(self.root / "t5.safetensors")}
        bad = adapter.validate(domain(a), self.musubi_config(a, "wan", {**base, "bucket_resolutions": "960x544\nnope"}))
        self.assertIn("bucket_resolutions", [i.field for i in bad if i.severity == "error"])
        # A valid list passes and falls back to single resolution when empty.
        job, _, values = self.render_musubi(a, "wan", {**base, "bucket_resolutions": "960x544, 832x480"},
                                           name="buckets-ok")
        dataset = tomllib.loads((job / "dataset.toml").read_text(encoding="utf-8"))
        self.assertEqual(dataset["general"]["resolution"], [[960, 544], [832, 480]])
        self.assertNotIn("resolution", values)

    def test_list_arguments_and_resolutions_are_multiline_inputs(self):
        specs = {p["key"]: p for p in (p.spec() for p in MUSUBI_PARAMS)}
        for key in ("network_args", "optimizer_args", "lr_scheduler_args"):
            self.assertTrue(specs[key].get("list_arg"), key)
            self.assertTrue(specs[key].get("multiline"), key)
        # Multi-resolution keeps its lines (each line is one bucket size).
        self.assertTrue(specs["bucket_resolutions"].get("multiline"))
        self.assertNotIn("list_arg", specs["bucket_resolutions"])
        self.assertNotIn("multiline", specs["network_dim"])

    def test_training_draft_is_bounded_and_shared(self):
        draft = dict(
            name="aki", installation_id="inst", architecture="wan",
            base_model_path="H:\\m.safetensors", dataset_id="ds", output_dir="H:\\out",
            params={"network_dim": 32, "beta": 0.5, "flag": True, "none": None, "text": "a" * 10,
                    "list": [1, 2], "obj": {"a": 1}, "nan": float("nan"), "inf": float("inf")},
            params_by_architecture={"wan": {"network_dim": 16}, "krea2": {"dropped": [1]}},
        )
        stored = self.training.save_draft(draft)
        self.assertEqual(stored["draft"]["name"], "aki")
        # Values an engine could not accept are dropped, never stored.
        self.assertEqual(stored["draft"]["params"],
                         {"network_dim": 32, "beta": 0.5, "flag": True, "none": None, "text": "a" * 10})
        self.assertEqual(stored["params_by_architecture"], {"wan": {"network_dim": 16}, "krea2": {}})
        self.assertTrue(stored["updated_at"])
        # The draft lives in the data root, so every client sees the same one.
        other = TrainingService(self.store, self.engines, self.datasets)
        self.assertEqual(other.draft()["draft"]["installation_id"], "inst")
        # Oversized payloads are refused instead of silently trimmed.
        for bad in ({**draft, "name": "x" * 201},
                    {**draft, "params": {"k": "x" * 3000}},
                    {**draft, "params": {f"k{i}": 1 for i in range(502)}},
                    {**draft, "params_by_architecture": {f"a{i}": {} for i in range(70)}}):
            with self.assertRaises(ServiceError, msg=str(bad)[:60]):
                self.training.save_draft(bad)
        self.assertEqual(self.training.draft()["draft"]["name"], "aki")
        self.training.clear_draft()
        self.assertIsNone(self.training.draft()["draft"])

    def test_kohya_single_stage_command(self):
        a = self.ready()
        t = self.training.submit(self.draft(a))
        recorded = json.loads((self.store.root / "jobs" / t["task_id"] / "command.json").read_text(encoding="utf-8"))
        self.assertEqual(len(recorded["commands"]), 1)
        Worker(self.store).execute(self.training.task(t["task_id"]))
        self.assertEqual(self.training.task(t["task_id"])["state"], "succeeded")

    def test_output_protects_inputs_and_state(self):
        a = self.ready(); d = self.draft(a)
        for folder in (a["source_path"], self.dataset_path / "output", self.model.parent / "output", self.store.root / "jobs"):
            self.assertFalse(self.training.validate({**d, "output_dir": str(folder)})["ok"])

    def test_nested_dataset_is_explicitly_rejected(self):
        sub = self.dataset_path / "nested"; sub.mkdir()
        Image.new("RGB", (128, 128)).save(sub / "sub.png")
        self.datasets.scan(self.dataset["dataset_id"])
        self.assertFalse(self.training.validate(self.draft(self.ready()))["ok"])

    def test_task_binding_is_fixed_after_default_switch(self):
        a = self.ready(); b = self.ready("another")
        task = self.training.submit(self.draft(a))
        self.engines.set_default(b["installation_id"])
        actual = self.training.task(task["task_id"])
        self.assertEqual(actual["binding"]["installation_id"], a["installation_id"])
        self.assertEqual(actual["binding"]["revision"], a["revision"])
        self.assertNotIn("binding", task)
        self.assertEqual(Path(task["output_dir"]).name, task["task_id"])

    def test_queued_cancel_is_immediate_and_cannot_be_claimed(self):
        t = self.training.submit(self.draft(self.ready()))
        stale = self.training.task(t["task_id"])
        self.training.stop(t["task_id"], False)
        Worker(self.store).execute(stale)
        self.assertEqual(self.training.task(t["task_id"])["state"], "stopped")
        self.assertFalse(Path(t["output_dir"]).exists())

    def execute(self, script=FAKE_ENGINE):
        a = self.ready(script=script)
        t = self.training.submit(self.draft(a))
        Worker(self.store).execute(self.training.task(t["task_id"]))
        return t

    def test_worker_real_subprocess_success_metrics_and_publish(self):
        t = self.execute()
        self.assertEqual(self.training.task(t["task_id"])["state"], "succeeded")
        self.assertEqual(self.training.metrics(t["task_id"])[0]["loss"], 0.25)
        self.assertTrue(any("avr_loss" in l for l in self.training.log(t["task_id"], 10)))
        a = self.training.artifacts(t["task_id"])[0]
        self.assertTrue(a["complete"])
        target = self.root / "comfy-lora"; target.mkdir()
        self.training.publish(a["artifact_id"], str(target), "trained.safetensors")
        self.assertEqual((target / "trained.safetensors").read_bytes(), Path(a["path"]).read_bytes())
        with self.assertRaises(ServiceError): self.training.publish(a["artifact_id"], str(target), "trained.safetensors")
        with self.assertRaises(ServiceError): self.training.publish(a["artifact_id"], str(target), "../escape.safetensors")
        Path(a["path"]).write_bytes(b"truncated")
        self.assertFalse(checkpoint_complete(Path(a["path"])))
        with self.assertRaises(ServiceError): self.training.publish(a["artifact_id"], str(target), "broken.safetensors")
        self.assertFalse((target / "broken.safetensors").exists())

    def test_worker_nonzero_exit_never_claims_success(self):
        t = self.execute("import sys; print('failure'); sys.exit(7)")
        self.assertEqual(self.training.task(t["task_id"])["state"], "failed")
        self.assertIn("7", self.training.task(t["task_id"])["error_summary"])

    def test_worker_refuses_changed_config_source_and_environment(self):
        for change in ("config", "source", "environment"):
            name = "engine-" + change
            a = self.ready(name)
            t = self.training.submit(self.draft(a))
            if change == "config":
                p = self.store.root / "jobs" / t["task_id"] / "dataset.toml"
                p.write_text(p.read_text() + "\n# edited")
            elif change == "source":
                p = Path(a["source_path"]) / "train_network.py"
                p.write_text(p.read_text() + "\n# edited")
            else:
                p = self.store.root / "manifests" / (a["environment_manifest_id"] + ".json")
                p.write_text(json.dumps({**INFO, "torch_version": "changed"}))
            Worker(self.store).execute(self.training.task(t["task_id"]))
            actual = self.training.task(t["task_id"])
            self.assertEqual(actual["state"], "failed", change)
            self.assertFalse(Path(t["output_dir"]).exists())

    def test_force_stop_running_subprocess(self):
        a = self.ready(script="import time; print('running', flush=True); time.sleep(30)")
        t = self.training.submit(self.draft(a))
        worker = Worker(self.store)
        thread = threading.Thread(target=worker.execute, args=(self.training.task(t["task_id"]),))
        thread.start()
        deadline = time.monotonic() + 10
        while self.training.task(t["task_id"])["state"] != "running" and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        process = worker.process
        try:
            self.training.stop(t["task_id"], True)
            thread.join(20)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.training.task(t["task_id"])["state"], "stopped")
        finally:
            # Keep failed tests from leaking a process or an open log handle.
            if process and process.poll() is None:
                process.kill()
            if process: process.wait(timeout=10)
            thread.join(35)

    def test_failed_stop_requires_manual_confirmation(self):
        a = self.ready(script="import time; print('running', flush=True); time.sleep(1)")
        t = self.training.submit(self.draft(a))
        worker = Worker(self.store)
        thread = threading.Thread(target=worker.execute, args=(self.training.task(t["task_id"]),))
        with patch.object(worker, "interrupt", side_effect=RuntimeError("denied")):
            thread.start()
            deadline = time.monotonic() + 10
            while self.training.task(t["task_id"])["state"] != "running" and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            self.training.stop(t["task_id"], True)
            thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.training.task(t["task_id"])["state"], "connection_lost")
        self.assertIn("人工核实", self.training.task(t["task_id"])["error_summary"])

    def test_orphan_requires_explicit_resolution(self):
        t = self.training.submit(self.draft(self.ready()))
        with self.assertRaises(ServiceError): self.training.acknowledge_exit(t["task_id"])
        self.store.patch("task", t["task_id"], {"state": "connection_lost"})
        self.training.acknowledge_exit(t["task_id"])
        self.assertEqual(self.training.task(t["task_id"])["state"], "stopped")

    def test_lock_is_exclusive(self):
        a = ProcessLock(self.root / "lock"); b = ProcessLock(self.root / "lock")
        try:
            self.assertTrue(a.acquire()); self.assertFalse(b.acquire())
            a.close(); self.assertTrue(b.acquire())
        finally: a.close(); b.close()

    # ---- musubi tuner: architecture-scoped defaults and enumerations ----

    def test_musubi_timestep_sampling_default_follows_architecture(self):
        adapter = ADAPTERS["musubi_tuner"]
        a = self.musubi_arch_engine("musubi-arches", (
            "wan_train_network.py", "wan_cache_latents.py", "wan_cache_text_encoder_outputs.py",
            "ideogram4_train_network.py", "ideogram4_cache_latents.py", "ideogram4_cache_text_encoder_outputs.py",
            "minimax_h3_train_network.py", "minimax_h3_cache_latents.py", "minimax_h3_cache_text_encoder_outputs.py",
        ))
        cases = (
            ("wan", {"vae": str(self.root / "vae.safetensors"), "wan_t5": str(self.root / "t5.safetensors")}, "sigma"),
            ("ideogram4", {"vae": str(self.root / "vae.safetensors"),
                           "ideogram_text_encoder": str(self.root / "te.safetensors")}, "ideogram4_shift"),
            ("minimax_h3", {"h3_video_vae": str(self.root / "video.safetensors"),
                            "h3_audio_vae": str(self.root / "audio.safetensors"),
                            "h3_text_encoder": str(self.root / "te.safetensors")}, "uniform"),
        )
        for architecture, params, expected in cases:
            errors = [i.message for i in adapter.validate(domain(a), self.musubi_config(a, architecture, params))
                      if i.severity == "error"]
            self.assertEqual(errors, [], architecture)
            job, native, values = self.render_musubi(a, architecture, params)
            self.assertEqual(values["timestep_sampling"], expected, architecture)
            self.assertEqual(native.read_text(encoding="utf-8").count("timestep_sampling"), 1, architecture)
            # Cache stages must never assume the cache directory exists.
            self.assertTrue((job / "cache" / architecture).is_dir(), architecture)
            self.assertEqual(len(json.loads((job / "stages.json").read_text(encoding="utf-8"))), 3, architecture)

    def test_musubi_dit_dtype_is_only_sent_when_chosen(self):
        a = self.musubi_arch_engine("musubi-dtype", (
            "hv_1_5_train_network.py", "hv_1_5_cache_latents.py", "hv_1_5_cache_text_encoder_outputs.py"))
        params = {"vae": str(self.root / "vae.safetensors"), "hv15_text_encoder": str(self.root / "te.safetensors"),
                  "hv15_byt5": str(self.root / "byt5.safetensors")}
        # An fp16 checkpoint hard-fails if the trainer forces bf16/fp32 on it, so the
        # default must leave dtype detection to the engine.
        _, _, values = self.render_musubi(a, "hunyuan_video_15", params, name="dtype-auto")
        self.assertNotIn("dit_dtype", values)
        _, _, values = self.render_musubi(a, "hunyuan_video_15", {**params, "dit_dtype": "fp16"}, name="dtype-set")
        self.assertEqual(values["dit_dtype"], "fp16")

    def test_musubi_parameter_enums_match_engine_sources(self):
        params = {p.key: p for p in MUSUBI_PARAMS}
        self.assertEqual(params["h3_task"].enum_values, ["t2va", "fl2va", "ref2va"])
        self.assertEqual(params["h3_timestep_sampling"].default, "uniform")
        self.assertEqual(params["ideogram4_timestep_sampling"].default, "ideogram4_shift")
        self.assertEqual(params["timestep_sampling"].default, "sigma")
        presets = params["ideogram_sampler_preset"]
        self.assertEqual(presets.default, "V4_DEFAULT_20")
        self.assertIn("V4_TURBO_12", presets.enum_values)
        self.assertNotIn("V4_TURBO_10", presets.enum_values)
        self.assertEqual(params["kandinsky_task"].default, "k5-lite-t2i-hd")
        self.assertIn("k5-lite-t2i-hd", params["kandinsky_task"].enum_values)
        for value in ("t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC"):
            self.assertIn(value, params["wan_task"].enum_values)
        self.assertEqual(params["flux2_model_version"].enum_values,
                         ["dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"])
        # Exactly one timestep_sampling parameter may apply, whatever the architecture.
        for architecture, expected in (("wan", "timestep_sampling"),
                                       ("ideogram4", "ideogram4_timestep_sampling"),
                                       ("minimax_h3", "h3_timestep_sampling")):
            applying = [p.key for p in MUSUBI_PARAMS
                        if (p.native_key or p.key) == "timestep_sampling" and p.applies_to(architecture)]
            self.assertEqual(applying, [expected], architecture)

    def test_musubi_validate_reports_a_missing_pipeline(self):
        adapter = ADAPTERS["musubi_tuner"]
        a = self.musubi_arch_engine("musubi-pipeline", ("fpack_train_network.py",))
        config = self.musubi_config(a, "framepack", {})
        with patch("adapters.musubi_tuner.adapter.pipeline_for", return_value=None):
            issues = adapter.validate(domain(a), config)
        self.assertIn("尚未声明缓存与训练流水线", "；".join(i.message for i in issues))

    # ---- artifact steps ----

    def test_artifact_steps_come_from_the_adapter_not_the_filename(self):
        out = self.root / "step-outputs"; out.mkdir()
        write_safetensors(out / "lora-step00000200.safetensors")
        write_safetensors(out / "lora-000002.safetensors")
        Image.new("RGB", (64, 64), "red").save(out / "sample.png")
        job = self.store.root / "jobs" / "step-task"; job.mkdir(parents=True)
        (job / "job.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")

        musubi = {a.path.name: a.step for a in ADAPTERS["musubi_tuner"].collect_artifacts(job)}
        self.assertEqual(musubi["lora-step00000200.safetensors"], 200)
        # "{name}-{epoch:06d}" is an epoch file, never a training step.
        self.assertIsNone(musubi["lora-000002.safetensors"])
        kohya = {a.path.name: a.step for a in ADAPTERS["kohya"].collect_artifacts(job)}
        self.assertEqual(kohya["lora-000002.safetensors"], 2)
        self.assertIsNone(kohya["lora-step00000200.safetensors"])

        task = dict(task_id="step-task", engine_id="musubi_tuner", name="steps", output_dir=str(out),
                    architecture="wan", draft={"base_model_path": str(self.model)})
        register_artifacts(self.store, task)
        stored = {a["file_name"]: a["step"] for a in self.store.list("artifact")}
        self.assertEqual(stored["lora-step00000200.safetensors"], 200)
        self.assertIsNone(stored["lora-000002.safetensors"])
        self.assertIsNone(stored["sample.png"])
        self.assertTrue(all(a["complete"] for a in self.store.list("artifact")))

    def test_unknown_engine_id_never_aborts_artifact_registration(self):
        out = self.root / "unknown-outputs"; out.mkdir()
        write_safetensors(out / "lora.safetensors")
        # The job directory is created by submit before the worker ever runs.
        (self.store.root / "jobs" / "unknown-task").mkdir(parents=True)
        task = dict(task_id="unknown-task", engine_id="not_an_engine", name="x", output_dir=str(out),
                    architecture="sd1", draft={"base_model_path": str(self.model)})
        register_artifacts(self.store, task)
        job = self.store.root / "jobs" / "unknown-task"
        self.assertIn("产物收集失败", (job / "stdout.log").read_text(encoding="utf-8"))
        self.assertEqual(self.store.list("artifact"), [])

    # ---- log pipeline faults are never fatal ----

    def test_log_faults_do_not_kill_healthy_training(self):
        script = (
            "import json, struct, sys, tomllib\n"
            "from pathlib import Path\n"
            "sys.stdout.buffer.write(b'bad \\xff\\xfe bytes\\n'); sys.stdout.buffer.flush()\n"
            "print('steps: 100%|xxx| 2/2 [00:01<00:00, 2.00it/s, avr_loss=0.25]', flush=True)\n"
            "config = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))\n"
            "out = Path(config['output_dir']); out.mkdir(exist_ok=True)\n"
            "header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()\n"
            "(out / 'lora.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\\x00' * 4)\n"
        )
        a = self.ready(name="log-fault", script=script)
        adapter = ADAPTERS["kohya"]
        original = adapter.parse_log

        def broken(line):
            if "avr_loss" in line:
                raise ValueError("synthetic parse fault")
            return original(line)

        t = self.training.submit(self.draft(a))
        with patch.object(adapter, "parse_log", broken):
            Worker(self.store).execute(self.training.task(t["task_id"]))
        task = self.training.task(t["task_id"])
        # The engine exited 0, so the task succeeded despite the log pipeline fault.
        self.assertEqual(task["state"], "succeeded")
        self.assertIsNone(task["error_summary"])
        log = (self.store.root / "jobs" / t["task_id"] / "stdout.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("\ufffd", log)                 # raw bytes are preserved, lossily decoded
        self.assertIn("avr_loss=0.25", log)          # stdout was drained, never left to block
        self.assertIn("日志解析/写入异常", log)          # degraded loudly instead of silently
        self.assertTrue(self.training.artifacts(t["task_id"]))

    # ---- environment purification for the training child ----

    def test_training_child_env_is_purified(self):
        probe = self.root / "env-probe.json"
        script = (
            "import json, os, struct, sys, tomllib\n"
            "from pathlib import Path\n"
            "leaked = {name: os.environ.get(name) for name in\n"
            "          ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PIP_INDEX_URL', 'UV_INDEX_URL')}\n"
            "leaked['PYTHONNOUSERSITE'] = os.environ.get('PYTHONNOUSERSITE')\n"
            "leaked['PATH'] = bool(os.environ.get('PATH'))\n"
            "leaked['SystemRoot'] = bool(os.environ.get('SystemRoot'))\n"
            "Path(os.environ['LP_ENV_PROBE']).write_text(json.dumps(leaked), encoding='utf-8')\n"
            "config = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))\n"
            "out = Path(config['output_dir']); out.mkdir(exist_ok=True)\n"
            "header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()\n"
            "(out / 'lora.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\\x00' * 4)\n"
        )
        a = self.ready(name="env-probe", script=script)
        t = self.training.submit(self.draft(a))
        poisoned = {"PYTHONPATH": "poison", "PYTHONHOME": "poison", "VIRTUAL_ENV": "poison",
                    "CONDA_PREFIX": "poison", "PIP_INDEX_URL": "poison", "UV_INDEX_URL": "poison",
                    "LP_ENV_PROBE": str(probe)}
        with patch.dict(os.environ, poisoned):
            Worker(self.store).execute(self.training.task(t["task_id"]))
        self.assertEqual(self.training.task(t["task_id"])["state"], "succeeded")
        leaked = json.loads(probe.read_text(encoding="utf-8"))
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PIP_INDEX_URL", "UV_INDEX_URL"):
            self.assertIsNone(leaked[name], name)
        self.assertEqual(leaked["PYTHONNOUSERSITE"], "1")
        # PATH/SystemRoot must survive or CUDA/DLL loading breaks.
        self.assertTrue(leaked["PATH"]); self.assertTrue(leaked["SystemRoot"])

    # ---- engine listing stays cheap and side-effect free ----

    def test_engine_list_is_read_only_and_never_rescans(self):
        make_engine(self.engine_root / "readonly")
        self.engines.rescan()
        record = self.store.list("engine")[0]
        # A legacy record that the old list() would have silently repaired.
        self.store.patch("engine", record["installation_id"],
                         {"engine_id": None, "candidate_engines": [], "issues": ["无法唯一识别引擎类型，请手动确认"]})
        before = self.store.list("engine")
        with patch.object(self.engines, "_detect_candidates", side_effect=AssertionError("list() must not scan")):
            listed = self.engines.list()
        self.assertIsNone(listed[0]["engine_id"])
        self.assertEqual(self.store.list("engine"), before)
        # The explicit refresh path still repairs it.
        self.assertEqual(self.engines.rescan()[0]["engine_id"], "kohya")

    def test_dataset_duplicate_summary_matches_flagged_images(self):
        key = self.dataset["dataset_id"]
        for name in ("dup-a.png", "dup-b.png", "dup-c.png"):
            Image.new("RGB", (128, 128), "red").save(self.dataset_path / name)
        self.datasets.scan(key)
        summary = self.datasets.get(key)
        counted = {i["kind"]: i["count"] for i in summary["issues"]}["duplicate"]
        flagged = self.datasets.images(key, 0, 200, "duplicate")["total"]
        self.assertEqual(summary["image_count"], 4)
        # Every image sharing content is flagged once, and the summary agrees.
        self.assertEqual(counted, flagged)
        self.assertEqual(counted, 4)

    # ---- protected shutdown ----

    def test_supervisor_exits_on_shutdown_marker(self):
        self.store.put("meta", "shutdown", {"requested_at": "test"})
        thread = threading.Thread(target=Worker(self.store).run, daemon=True)
        thread.start()
        thread.join(10)
        self.assertFalse(thread.is_alive())

    def test_artifacts_appear_while_the_job_is_still_running(self):
        # The authoritative pass runs after the last stage; without a live pass
        # the results/overview pages stay empty for the whole run.
        script = (
            "import json, struct, sys, time, tomllib\n"
            "from pathlib import Path\n"
            "config = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))\n"
            "out = Path(config['output_dir']); out.mkdir(exist_ok=True)\n"
            "header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()\n"
            "(out / 'lora-000002.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\\x00' * 4)\n"
            "print('steps: 50%|xxx| 1/2 [00:01<00:01, 1.00it/s]', flush=True)\n"
            "time.sleep(25)\n"
        )
        a = self.ready(name="live-artifacts", script=script)
        t = self.training.submit(self.draft(a))
        worker = Worker(self.store)
        worker.ARTIFACT_POLL_SECONDS = 0.2
        thread = threading.Thread(target=worker.execute, args=(self.training.task(t["task_id"]),), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            found = []
            while time.monotonic() < deadline and thread.is_alive():
                found = self.training.artifacts(t["task_id"])
                if found:
                    break
                time.sleep(0.1)
            # Still running, yet the checkpoint is already registered and usable.
            self.assertEqual(self.training.task(t["task_id"])["state"], "running")
            self.assertTrue(found, "no artifact was registered while the job was running")
            checkpoint = next(x for x in found if x["kind"] == "checkpoint")
            self.assertTrue(checkpoint["complete"])
            self.assertEqual(checkpoint["step"], 2)
            self.assertIsNone(checkpoint["preview_url"])
        finally:
            self.training.stop(t["task_id"], True)
            thread.join(40)

    def test_checkpoint_rejects_invalid_tensor_shapes(self):
        p = self.root / "invalid.safetensors"
        for shape in ([2], [-1]):
            header = json.dumps({"t": {"dtype": "F32", "shape": shape, "data_offsets": [0, 4]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            self.assertFalse(checkpoint_complete(p))


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-api-tests-")
        self.paths = AppPaths.for_workspace(Path(self.temp.name))
        self.paths.engine_root.mkdir()
        self.app = create_app(self.paths, token="unit-secret", start_worker=False)
        self.client = LocalClient(self.app)

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def login(self):
        res = self.client.post("/api/v1/session", headers={"Origin": self.client.base_url})
        self.assertEqual(res.status_code, 204)
        self.assertIn("HttpOnly", res.headers["set-cookie"])
        self.assertIn("SameSite=strict", res.headers["set-cookie"])

    def test_authentication_origin_host_and_fetch_checks(self):
        unauthorized = self.client.get("/api/v1/settings")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.json()["detail"], "本机会话失效，请重新连接应用")
        self.assertEqual(self.client.post("/api/v1/session").status_code, 401)
        self.assertEqual(self.client.post("/api/v1/session", headers={"Origin": "https://hostile.example"}).status_code, 403)
        self.login()
        self.assertEqual(self.client.get("/api/v1/settings").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/settings", headers={"Host": "hostile.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/v1/settings", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        self.assertEqual(self.client.get("/api/v1/settings", headers={"Origin": "null"}).status_code, 403)
        self.assertEqual(self.client.get("/api/v1/settings", headers={"X-LP-Session": "wrong"}).status_code, 401)

    def test_dataset_routes_and_empty_204(self):
        self.login()
        root = self.paths.project_root / "images"; root.mkdir()
        Image.new("RGB", (128, 128)).save(root / "one.png")
        res = self.client.post("/api/v1/datasets", json={"path": str(root), "name": "My images"})
        key = res.json()["dataset_id"]
        res = self.client.post(f"/api/v1/datasets/{key}/scan")
        self.assertEqual(res.status_code, 204); self.assertEqual(res.content, b"")
        image = self.client.get(f"/api/v1/datasets/{key}/images").json()["items"][0]
        self.assertEqual(self.client.put(f"/api/v1/datasets/{key}/images/{image['image_id']}/caption", json={"caption": "test"}).status_code, 204)
        self.assertEqual(self.client.get(image["thumbnail_url"]).headers["content-type"], "image/jpeg")
        self.assertEqual(self.client.get(f"/api/v1/datasets/{key}/images?limit=9999").status_code, 422)
        self.assertEqual(self.client.get(f"/api/v1/datasets/{key}/images?issue=arbitrary").status_code, 400)

    def test_settings_and_no_hot_data_migration(self):
        self.login()
        settings = self.client.get("/api/v1/settings").json()
        self.assertEqual(self.client.put("/api/v1/settings", json=settings).status_code, 200)
        self.assertEqual(self.client.put("/api/v1/settings", json={**settings, "data_root": str(self.paths.project_root / "elsewhere")}).status_code, 409)
        self.assertEqual(self.client.put("/api/v1/settings", json={**settings, "unexpected": 1}).status_code, 422)
        self.app.state.store.patch("meta", "settings", {"data_root": "old-path"})
        migrated = create_app(self.paths, start_worker=False)
        self.assertEqual(migrated.state.store.get("meta", "settings")["data_root"], str(self.paths.data_root))

    def test_acknowledgement_requires_explicit_confirmation(self):
        self.login()
        self.assertEqual(self.client.post("/api/v1/tasks/x/acknowledge-exit", json={"confirmed_exited": False}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/tasks/x/acknowledge-exit", json={"confirmed_exited": True}).status_code, 404)

    def test_gpu_is_unknown_without_monitoring_and_no_static_build(self):
        self.login()
        status = self.client.get("/api/v1/system/status").json()
        self.assertIsNone(status["gpu"])
        self.assertEqual(status["installation_workflow_version"], INSTALLATION_WORKFLOW_VERSION)
        missing_frontend = self.client.get("/")
        self.assertEqual(missing_frontend.status_code, 503)
        self.assertEqual(missing_frontend.json()["detail"], "前端未构建，请在 frontend 中执行 npm run build")

    def test_shutdown_is_refused_while_work_is_unresolved(self):
        self.login()
        store = self.app.state.store

        class FakeServer:
            should_exit = False

        self.app.state.server = FakeServer()
        for state in ("queued", "preparing", "running", "stopping"):
            store.put("task", "active-" + state, {"task_id": "active-" + state, "state": state})
            self.assertEqual(self.client.post("/api/v1/system/shutdown").status_code, 409, state)
            store.delete("task", "active-" + state)
        store.put("task", "orphan", {"task_id": "orphan", "state": "connection_lost"})
        self.assertEqual(self.client.post("/api/v1/system/shutdown").status_code, 409)
        store.delete("task", "orphan")
        self.assertFalse(self.app.state.server.should_exit)
        self.assertIsNone(store.get("meta", "shutdown"))

    def test_shutdown_is_graceful_when_idle(self):
        self.login()
        store = self.app.state.store

        class FakeServer:
            should_exit = False

        self.app.state.server = FakeServer()
        response = self.client.post("/api/v1/system/shutdown")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "shutting_down")
        # Both halves of the request are recorded: the supervisor's marker and
        # uvicorn's own unwind flag.
        self.assertTrue(store.get("meta", "shutdown")["requested_at"])
        self.assertTrue(self.app.state.server.should_exit)
        # Work must not be accepted once the supervisor is on its way out.
        draft = dict(name="late", installation_id="any", architecture="sd1", base_model_path="m",
                     dataset_id="d", output_dir="o", params={})
        refusal = self.client.post("/api/v1/training/submit", json=draft)
        self.assertEqual(refusal.status_code, 409)
        self.assertIn("正在退出", refusal.json()["detail"])

    def test_stale_shutdown_marker_is_cleared_on_restart(self):
        # A leftover marker must not block submissions once the app is restarted,
        # even if no supervisor has started yet to clear it.
        self.app.state.store.put("meta", "shutdown", {"requested_at": "old", "backend_pid": 1})
        restarted = create_app(self.paths, token="unit-secret", start_worker=False)
        self.assertIsNone(restarted.state.store.get("meta", "shutdown"))
        draft = dict(name="after restart", installation_id="any", architecture="sd1", base_model_path="m",
                     dataset_id="d", output_dir="o", params={})
        # Reaches normal validation again instead of the shutdown refusal.
        self.assertNotEqual(self.client.post("/api/v1/training/submit", json=draft).status_code, 409)

    def test_training_draft_endpoints_share_one_stored_form(self):
        self.login()
        self.assertIsNone(self.client.get("/api/v1/training/draft").json()["draft"])
        body = dict(name="staged", installation_id="i", architecture="wan", base_model_path="m",
                    dataset_id="d", output_dir="o", params={"network_dim": 32, "bad": [1]},
                    params_by_architecture={"wan": {"network_dim": 32}})
        self.assertEqual(self.client.put("/api/v1/training/draft", json=body).status_code, 200)
        got = self.client.get("/api/v1/training/draft").json()
        # Values an engine could not accept are dropped, the rest of the draft is kept.
        self.assertEqual(got["draft"]["params"], {"network_dim": 32})
        self.assertEqual(got["params_by_architecture"], {"wan": {"network_dim": 32}})
        # Another client of the same data root reads the same staged form.
        restarted = create_app(self.paths, start_worker=False)
        self.assertEqual(restarted.state.store.get("meta", "training_draft")["draft"]["name"], "staged")
        # Unknown fields and over-long text are rejected, not stored.
        self.assertEqual(self.client.put("/api/v1/training/draft", json={**body, "unexpected": 1}).status_code, 422)
        self.assertEqual(self.client.put("/api/v1/training/draft", json={**body, "name": "x" * 201}).status_code, 422)
        self.assertEqual(self.client.request("DELETE", "/api/v1/training/draft").status_code, 204)
        self.assertIsNone(self.client.get("/api/v1/training/draft").json()["draft"])

    def test_desktop_reconnect_checks_data_identity(self):
        root = self.paths.data_root
        with patch("urllib.request.build_opener") as opener:
            response = opener.return_value.open.return_value.__enter__.return_value
            response.status = 204
            with patch("desktop.launcher.json.load", side_effect=[{"backend_version": "0.1.0", "installation_workflow_version": INSTALLATION_WORKFLOW_VERSION}, {"data_root": str(root)}]):
                self.assertTrue(backend_ready("http://127.0.0.1:8765", root))
            with patch("desktop.launcher.json.load", side_effect=[{"backend_version": "0.1.0", "installation_workflow_version": INSTALLATION_WORKFLOW_VERSION}, {"data_root": str(root / "wrong")} ]):
                with self.assertRaises(RuntimeError): backend_ready("http://127.0.0.1:8765", root)
        self.assertEqual(
            [m for m in dir(DesktopBridge()) if not m.startswith("_")],
            ["begin_window_resize", "get_window_state", "open_in_explorer", "pick_path", "window_action"],
        )


if __name__ == "__main__":
    unittest.main()
