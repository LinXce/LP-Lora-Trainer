"""History actions never require an engine, GPU, or training environment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from app.api.server import create_app
from app.config import AppPaths
from app.services.common import ServiceError
from app.services.installation import InstallationService
from app.services.training import TrainingService
from app.storage.state import StateStore
from tests.local_client import LocalClient


class HistoryActionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = StateStore(self.root / "data")
        self.training = TrainingService(self.store, Mock(), Mock())
        self.installations = InstallationService(self.store, Mock(), self.root)
        self.key = "a" * 32
        self.output = self.root / "output" / self.key
        self.output.mkdir(parents=True)
        self.task = dict(task_id=self.key, state="succeeded", pid=None, name="历史训练", engine_id="musubi_tuner",
            installation_id="original-engine", architecture="krea2", output_dir=str(self.output),
            draft=dict(name="历史训练", installation_id="original-engine", architecture="krea2", dataset_id="dataset",
                       base_model_path="H:/model.safetensors", dataset_path="H:/dataset", output_dir=str(self.output),
                       params={"network_dim": 16, "enabled": False, "seed": 0, "resolution": "512,1024"}),
            binding={"private": True})
        self.store.put("task", self.key, self.task)
        self.job = self.store.root / "jobs" / self.key
        self.job.mkdir(parents=True)
        (self.job / "stdout.log").write_text("training log", encoding="utf-8")
        (self.job / "job.json").write_text("{}", encoding="utf-8")

    def assert_error(self, status, function, *args):
        with self.assertRaises(ServiceError) as caught:
            function(*args)
        self.assertEqual(caught.exception.status, status)

    def test_configuration_roundtrip_keeps_all_values_and_restores_output_root(self):
        for engine in ("kohya", "ai_toolkit", "musubi_tuner"):
            with self.subTest(engine=engine):
                self.store.patch("task", self.key, {"engine_id": engine})
                config = self.training.configuration(self.key)
                self.assertEqual(config["engine_id"], engine)
                self.assertEqual(config["format"], "lp-training-parameters")
                self.assertEqual(config["version"], 1)
                self.assertEqual(config["output_dir"], str(self.output.parent))
                self.assertEqual(config["dataset_path"], "H:/dataset")
                self.assertEqual(config["params"], self.task["draft"]["params"])
                self.assertNotIn("binding", config)
                config["params"]["network_dim"] = 99
                self.assertEqual(self.training.task(self.key)["draft"]["params"]["network_dim"], 16)
        self.store.patch("task", self.key, {"draft": {**self.task["draft"], "output_dir": "H:/different-root"}})
        self.assertEqual(self.training.configuration(self.key)["output_dir"], "H:/different-root")

    def test_delete_task_is_persistent_and_preserves_artifacts_and_source_files(self):
        (self.output / "lora.safetensors").write_bytes(b"weights")
        self.store.put("artifact", "artifact", dict(artifact_id="artifact", task_id=self.key, path=str(self.output / "lora.safetensors")))
        self.training.delete(self.key)
        self.assertEqual(self.training.tasks(), [])
        self.assertFalse(self.job.exists())
        self.assertTrue((self.output / "lora.safetensors").is_file())
        self.assertEqual(self.training.artifact_output_dir(self.key), str(self.output))
        self.assertEqual(len(self.training.artifacts(self.key)), 1)
        self.assertTrue(self.training.result_history()[0]["history_deleted"])
        self.assertNotIn("draft", self.training.result_history()[0])
        reopened = TrainingService(StateStore(self.store.root), Mock(), Mock())
        self.assertEqual(reopened.tasks(), [])
        self.assert_error(404, reopened.configuration, self.key)
        self.assert_error(404, reopened.delete, self.key)

    def test_active_unknown_and_stale_pid_tasks_cannot_be_deleted(self):
        for state in ("queued", "preparing", "running", "stopping", "connection_lost", "unknown"):
            with self.subTest(state=state):
                self.store.patch("task", self.key, {"state": state})
                self.assert_error(409, self.training.delete, self.key)
                self.assertTrue(self.job.exists())
        self.store.patch("task", self.key, {"state": "failed", "pid": 123})
        self.assert_error(409, self.training.delete, self.key)

    def test_delete_all_final_task_states_without_log_directory(self):
        import shutil
        shutil.rmtree(self.job)
        for state in ("succeeded", "failed", "stopped"):
            self.store.put("task", self.key, {**self.task, "state": state})
            self.training.delete(self.key)
            self.assertIsNone(self.store.get("task", self.key))

    def test_failed_cleanup_retains_history(self):
        with patch("app.services.training.shutil.rmtree", side_effect=PermissionError("busy")):
            with self.assertRaises(PermissionError): self.training.delete(self.key)
        self.assertIsNotNone(self.store.get("task", self.key))

    def test_invalid_task_key_never_deletes_another_directory(self):
        self.store.put("task", "../outside", {**self.task, "task_id": "../outside"})
        self.assert_error(422, self.training.delete, "../outside")
        self.assertTrue(self.job.exists())

    def test_resolved_delete_targets_are_checked_before_recursive_cleanup(self):
        with patch("app.services.training.contained", return_value=self.output), patch("app.services.training.shutil.rmtree") as remove:
            self.assert_error(403, self.training.delete, self.key)
            remove.assert_not_called()
        self.make_session()
        with patch("app.services.installation.contained", return_value=self.output), patch("app.services.installation.shutil.rmtree") as remove:
            self.assert_error(403, self.installations.delete, self.key)
            remove.assert_not_called()

    def test_linked_log_directory_cannot_delete_outputs(self):
        import shutil
        shutil.rmtree(self.job)
        try: self.job.symlink_to(self.output, target_is_directory=True)
        except OSError: self.skipTest("Creating directory symlinks is unavailable")
        self.addCleanup(lambda: self.job.unlink(missing_ok=True))
        (self.output / "keep.txt").write_text("keep", encoding="utf-8")
        self.assert_error(403, self.training.delete, self.key)
        self.assertTrue((self.output / "keep.txt").exists())

    def make_session(self, state="succeeded"):
        record = dict(session_id=self.key, installation_id="engine", state=state)
        self.store.put("installation", self.key, record)
        log = self.installations.log_path(self.key)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("installation log", encoding="utf-8")
        return log

    def test_delete_installation_history_and_log_but_not_engine_environment(self):
        environment = self.root / "engine" / "musubi-tuner" / ".venv"
        environment.mkdir(parents=True)
        (environment / "keep.txt").write_text("environment", encoding="utf-8")
        for state in ("succeeded", "failed", "cancelled", "interrupted"):
            log = self.make_session(state)
            self.installations.delete(self.key)
            self.assertFalse(log.parent.exists())
            self.assertEqual(self.installations.list(), [])
            self.assertTrue((environment / "keep.txt").exists())
        reopened = InstallationService(StateStore(self.store.root), Mock(), self.root)
        self.assertEqual(reopened.list(), [])
        self.assert_error(404, reopened.delete, self.key)

    def test_installation_busy_states_and_pending_worker_are_protected(self):
        for state in ("queued", "running", "stopping", "verifying", "unknown"):
            log = self.make_session(state)
            self.assert_error(409, self.installations.delete, self.key)
            self.assertTrue(log.exists())
        self.make_session()
        self.installations.workers[self.key] = {}
        self.assert_error(409, self.installations.delete, self.key)
        self.installations.workers.clear()

    def test_authenticated_api_configuration_delete_and_preserved_preview(self):
        (self.root / "engine").mkdir()
        app = create_app(AppPaths.for_workspace(self.root), start_worker=False)
        self.assertEqual(app.state.store.root, self.store.root)
        Image.new("RGB", (8, 8)).save(self.output / "sample.png")
        self.store.put("artifact", "sample", dict(artifact_id="sample", task_id=self.key, kind="sample", complete=True,
                                                 path=str(self.output / "sample.png")))
        self.make_session()
        client = LocalClient(app)
        self.addCleanup(client.close)
        headers = {"X-LP-Session": app.state.session_token}
        self.assertEqual(client.request("DELETE", "/api/v1/tasks/" + self.key).status_code, 401)
        config = client.get("/api/v1/tasks/" + self.key + "/configuration", headers=headers)
        self.assertEqual(config.status_code, 200)
        self.assertEqual(config.json()["params"]["seed"], 0)
        self.store.patch("task", self.key, {"state": "running"})
        self.assertEqual(client.request("DELETE", "/api/v1/tasks/" + self.key, headers=headers).status_code, 409)
        self.store.patch("task", self.key, {"state": "succeeded"})
        self.assertEqual(client.request("DELETE", "/api/v1/tasks/" + self.key, headers=headers).status_code, 204)
        self.assertEqual(client.get("/api/v1/tasks", headers=headers).json(), [])
        self.assertEqual(client.get("/api/v1/tasks/" + self.key + "/configuration", headers=headers).status_code, 404)
        self.assertEqual(client.get("/api/v1/artifacts?task_id=" + self.key, headers=headers).status_code, 200)
        self.assertTrue(client.get("/api/v1/artifacts/tasks", headers=headers).json()[0]["history_deleted"])
        self.assertEqual(client.get("/api/v1/artifacts/sample/preview", headers=headers).status_code, 200)
        self.assertEqual(client.request("DELETE", "/api/v1/terminal/sessions/" + self.key, headers=headers).status_code, 204)
        self.assertEqual(client.get("/api/v1/terminal/sessions", headers=headers).json(), [])
        self.assertTrue((self.output / "sample.png").is_file())
