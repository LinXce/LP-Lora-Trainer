"""No GPU, upstream engines, GUI or persistent processes are needed by these tests."""
import json
import os
import struct
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
from fastapi.testclient import TestClient
from app.config import AppPaths
from app.api.models import TrainingInput
from app.api.server import create_app
from app.services.common import ServiceError, contained, identity
from app.services.datasets import DatasetService
from app.services.engines import EngineService, revision
from app.services.training import TrainingService
from app.storage.state import StateStore
from adapters.registry import ADAPTERS, domain
from desktop.bridge import DesktopBridge
from desktop.launcher import backend_ready
from supervisor.worker import Worker, ProcessLock, checkpoint_complete

INFO = dict(python_version=sys.version.split()[0], executable=sys.executable, torch_version="test", cuda_wheel="test",
            cuda_available=True, diffusers=True, yaml=True, toml=True)
DIAG = "print(" + repr("LP_DIAG=" + json.dumps(INFO)) + ")"
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

    def tearDown(self):
        self.diag_patch.stop()
        self.temp.cleanup()

    def ready(self, name="copied source", script=FAKE_ENGINE, nested=False):
        make_engine(self.engine_root / name, script, nested)
        self.engines.rescan()
        record = next(r for r in self.engines.list() if r["label"] == name)
        self.engines.bind(record["installation_id"], sys.executable)
        return self.engines.get(record["installation_id"])

    def draft(self, installation):
        return dict(name="Test task", installation_id=installation["installation_id"], architecture="sd1",
                    base_model_path=str(self.model), dataset_id=self.dataset["dataset_id"], output_dir=str(self.root / "outputs"), params={})

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

        repaired = self.engines.list()
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
        self.assertEqual(self.engines.capabilities(ai["installation_id"])["architectures"], [])
        self.assertFalse(self.training.validate(self.draft(ai))["ok"])

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
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def login(self):
        res = self.client.post("/api/v1/session", headers={"Origin": "http://127.0.0.1:8765"})
        self.assertEqual(res.status_code, 204)
        self.assertIn("HttpOnly", res.headers["set-cookie"])
        self.assertIn("SameSite=strict", res.headers["set-cookie"])

    def test_authentication_origin_host_and_fetch_checks(self):
        self.assertEqual(self.client.get("/api/v1/settings").status_code, 401)
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
        self.assertIsNone(self.client.get("/api/v1/system/status").json()["gpu"])
        self.assertEqual(self.client.get("/").status_code, 503)

    def test_desktop_reconnect_checks_data_identity(self):
        root = self.paths.data_root
        with patch("urllib.request.build_opener") as opener:
            response = opener.return_value.open.return_value.__enter__.return_value
            response.status = 204
            with patch("desktop.launcher.json.load", side_effect=[{"backend_version": "0.1.0"}, {"data_root": str(root)}]):
                self.assertTrue(backend_ready("http://127.0.0.1:8765", root))
            with patch("desktop.launcher.json.load", side_effect=[{"backend_version": "0.1.0"}, {"data_root": str(root / "wrong")} ]):
                with self.assertRaises(RuntimeError): backend_ready("http://127.0.0.1:8765", root)
        self.assertEqual([m for m in dir(DesktopBridge()) if not m.startswith("_")], ["open_in_explorer", "pick_path"])


if __name__ == "__main__":
    unittest.main()
