"""Output ownership across data stores, including jobs submitted by older LP versions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from adapters.registry import ADAPTERS, domain
from app.services import output_ownership
from app.services.common import ServiceError
from app.storage.state import StateStore
from supervisor.worker import Worker
from tests import test_functionality as fixtures


class ReservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lp-output-reservations-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stores = [StateStore(self.root / name) for name in ("data-A", "data-B")]
        self.target = self.root / "outputs" / "portrait"

    def test_independent_stores_racing_for_one_target_have_exactly_one_owner(self):
        start = threading.Barrier(2)
        def reserve(index):
            start.wait(timeout=5)
            try:
                return output_ownership.reserve(self.stores[index], str(index), self.target)
            except ServiceError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, (0, 1)))
        owners = [i for i, result in enumerate(results) if isinstance(result, Path)]
        self.assertEqual(len(owners), 1)
        owner = owners[0]
        other = results[1 - owner]
        self.assertIsInstance(other, ServiceError)
        self.assertEqual(other.status, 409)
        metadata = json.loads(results[owner].read_text(encoding="utf-8"))
        self.assertEqual(metadata["task_id"], str(owner))
        self.assertEqual(metadata["data_root"], str(self.stores[owner].root))
        self.assertEqual(metadata["output_dir"], str(self.target))
        self.assertFalse(self.target.exists())

    def test_new_task_refuses_missing_wrong_task_and_wrong_data_root_marker(self):
        store = self.stores[0]
        key = "1" * 32
        marker = output_ownership.reserve(store, key, self.target)
        expected = json.loads(marker.read_text(encoding="utf-8"))
        job = self.root / "jobs" / key
        job.mkdir(parents=True)
        task = dict(task_id=key, output_dir=str(self.target), binding={})
        (job / "job.json").write_text(json.dumps(dict(output_dir=str(self.target), output_layout="task_id")),
                                      encoding="utf-8")
        for change in (None, dict(task_id="2" * 32), dict(data_root=str(self.stores[1].root))):
            with self.subTest(change=change):
                if change is None:
                    marker.unlink()
                else:
                    marker.write_text(json.dumps({**expected, **change}), encoding="utf-8")
                with self.assertRaises(ServiceError) as error:
                    output_ownership.before_launch(store, task, job)
                self.assertEqual(error.exception.status, 409)
                self.assertFalse(self.target.exists())
                if change:
                    self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), {**expected, **change})
                else:
                    self.assertFalse(marker.exists())


class LegacyQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ServiceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.training = self.fixture.training
        self.engine = self.fixture.ready()
        self.target = self.fixture.root.resolve() / "old-output" / "Test task"

    def legacy_job(self):
        # Keep the real current environment/adapter baseline, but reconstruct
        # the bound files exactly as an old job: original output path, no layout
        # flag, no output_verified field and no marker for its legacy target.
        submitted = self.training.submit(self.fixture.draft(self.engine))
        key = submitted["task_id"]
        job = self.fixture.store.root / "jobs" / key
        task = self.training.task(key)
        task.pop("output_verified")
        task.pop("recovery", None)
        config = task["draft"]
        config.pop("output_layout")
        config["output_dir"] = task["output_dir"] = str(self.target)
        (job / "job.json").write_text(json.dumps(config), encoding="utf-8")
        ADAPTERS["kohya"].write_native_config(domain(self.engine), config, job)
        binding = task["binding"]
        binding["config_hashes"] = {name: hashlib.sha256((job / name).read_bytes()).hexdigest()
                                    for name in binding["config_hashes"]}
        (job / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
        self.fixture.store.put("task", key, task)
        return task

    def execute_blocked(self, task, expected_error):
        worker = Worker(self.fixture.store)
        with patch.object(worker, "_run_stage") as launch:
            worker.execute(task)
        launch.assert_not_called()
        record = self.training.task(task["task_id"])
        self.assertEqual(record["state"], "failed")
        self.assertFalse(record["output_verified"])
        self.assertIn(expected_error, record["error_summary"])
        self.assertEqual(self.training.artifacts(task["task_id"]), [])

    def test_upgrade_preserves_existing_nested_weight_and_does_not_index_it(self):
        task = self.legacy_job()
        folder = self.target / "epoch-1" / "network"
        folder.mkdir(parents=True)
        historical = fixtures.write_safetensors(folder / "old.safetensors")
        before = historical.read_bytes()
        self.execute_blocked(task, "已经存在")
        self.assertEqual(historical.read_bytes(), before)
        self.assertEqual([p for p in self.target.rglob("*") if p.is_file()], [historical])

    def test_two_old_jobs_bound_to_the_same_unused_directory_are_both_blocked(self):
        first = self.legacy_job()
        second = self.legacy_job()
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual(first["output_dir"], second["output_dir"])
        self.execute_blocked(first, second["task_id"])
        self.execute_blocked(second, first["task_id"])
        self.assertFalse(self.target.exists())

    def test_uncontested_old_job_runs_in_its_bound_directory(self):
        task = self.legacy_job()
        Worker(self.fixture.store).execute(task)
        record = self.training.task(task["task_id"])
        self.assertEqual(record["state"], "succeeded", record.get("error_summary"))
        self.assertTrue(record["output_verified"])
        self.assertEqual(Path(record["output_dir"]), self.target)
        result, = self.training.artifacts(task["task_id"])
        self.assertTrue(result["complete"])
        self.assertEqual(Path(result["path"]), self.target / "lora.safetensors")
