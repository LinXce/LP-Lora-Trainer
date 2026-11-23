"""Exercise the exit protocol with private SQLite roots and simulated engine processes."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.api.server import check_exit_protocol, create_app
from app.config import AppPaths
from app.services import queue_control as queue
from app.services.common import ServiceError
from app.storage.state import StateStore
from supervisor.worker import ProcessLock, UnverifiedChildren, Worker
from tests.local_client import LocalClient


class GracefulExitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.paths = AppPaths.for_workspace(Path(temp.name))
        self.store = StateStore(self.paths.data_root)

    def task(self, key, state):
        task = dict(task_id=key, name=key, state=state, stop_requested=None,
                    pid=123 if state == "running" else None, engine_id="kohya",
                    output_dir=str(self.paths.project_root / "output"),
                    recovery={"graceful_stop": True})
        self.store.put("task", key, task)
        return task

    def client(self):
        app = create_app(self.paths, token="test", start_worker=False)
        client = LocalClient(app)
        self.addCleanup(client.close)
        client.post("/api/v1/session", headers={"Origin": client.base_url})
        return client

    def test_api_restart_reuses_worker_and_final_shutdown_preserves_fifo(self):
        self.task("live", "running")
        self.task("next", "queued")
        self.task("last", "queued")
        base = "/api/v1/system/shutdown"
        first = self.client()
        before = first.get(base + "/status").json()
        self.assertEqual((before["status"], before["reason"], before["backend_pid"]), ("idle", "training", os.getpid()))
        self.assertIsNone(self.store.get("meta", "shutdown"))
        self.assertEqual(first.post(base + "?graceful=true&expected_reason=queued").status_code, 409)
        self.assertEqual(first.post(base + "?graceful=true&expected_task_id=live").json()["status"], "draining")
        self.assertIsNone(queue.claim(self.store, "next"))
        self.assertEqual(self.store.get("task", "live")["stop_requested"], "graceful_exit")

        second = self.client()
        self.assertEqual(second.get(base + "/status").json()["status"], "draining")
        self.assertEqual(self.store.get("task", "live")["state"], "running")
        verdict = dict(status="no_savepoint", sources=[], message="未发现保存点")
        worker = Worker(self.store)
        with patch.object(worker, "_stop_result", return_value=verdict):
            worker.finish_task("live", {"state": "stopped", "pid": None})
        resolved = second.get(base + "/status").json()
        self.assertEqual((resolved["status"], resolved["save_result"]), ("ready_to_exit", verdict))
        self.assertEqual(second.post(base + "?graceful=true").json()["status"], "shutting_down")
        self.assertIsNone(queue.claim(self.store, "next"))
        create_app(self.paths, start_worker=False)
        self.assertEqual([t["task_id"] for t in self.store.list("task") if t["state"] == "queued"], ["next", "last"])
        self.assertEqual(self.store.get("task", "live")["stop_result"], verdict)
        self.assertIsNotNone(queue.claim(self.store, "next"))

    def test_timeout_requires_separate_confirmation_and_correct_task_id(self):
        self.task("live", "running")
        with patch.object(queue, "SHUTDOWN_TIMEOUT", -1):
            queue.request(self.store, graceful=True)
        timed_out = queue.status(self.store)
        self.assertEqual((timed_out["status"], timed_out["force_allowed"]), ("blocked", True))
        for key in (None, "another"):
            with self.assertRaises(ServiceError):
                queue.request(self.store, graceful=True, force=True, expected_task_id=key)
        self.assertEqual(self.store.get("task", "live")["stop_requested"], "graceful_exit")
        queue.request(self.store, graceful=True, force=True, expected_task_id="live")
        self.assertEqual(self.store.get("task", "live")["stop_requested"], "force")

    def test_legacy_owner_is_refused_without_writing_a_stop_request(self):
        self.task("live", "running")
        lock = ProcessLock(self.store.root / "supervisor.lock")
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.close)
        (self.store.root / "supervisor.json").write_text(json.dumps({"heartbeat": time.time(), "pid": 123}))
        with self.assertRaises(ServiceError):
            check_exit_protocol(self.store.root)
        self.assertIsNone(self.store.get("task", "live")["stop_requested"])

    def test_only_new_supervisor_marks_unverified_and_cancel_never_replays(self):
        self.task("active", "running")
        self.task("next", "queued")
        queue.request(self.store, graceful=True)
        worker = Worker(self.store)
        with patch.object(worker, "shutdown_requested", return_value=True):
            worker.run()
        self.assertEqual(queue.status(self.store)["reason"], "unverified")
        with self.assertRaises(ServiceError):
            queue.request(self.store, graceful=True, force=True, expected_task_id="active")
        queue.cancel(self.store)
        self.assertEqual(self.store.get("task", "next")["state"], "queued")
        self.assertEqual(self.store.get("task", "active")["state"], "connection_lost")

    def test_enqueue_and_claim_share_freeze_transaction(self):
        self.task("next", "queued")
        # Either the claimant wins, so it is the draining target, or shutdown
        # wins and the task remains queued. A new process cannot slip in between.
        start = threading.Barrier(2)
        result = {}
        def claim():
            start.wait()
            result["claim"] = queue.claim(self.store, "next")
        thread = threading.Thread(target=claim)
        thread.start()
        start.wait()
        shutdown = queue.request(self.store, graceful=True)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(shutdown["status"], "draining" if result["claim"] else "shutting_down")
        with self.assertRaises(ServiceError):
            queue.enqueue(self.store, dict(task_id="late", state="queued"))
        self.assertIsNone(self.store.get("task", "late"))

    def test_stage_does_not_escalate_after_graceful_delivery_failure(self):
        task = self.task("live", "running")
        self.store.patch("task", "live", {"stop_requested": "graceful_exit"})
        process = Mock(pid=123, returncode=0)
        process.poll.side_effect = (None, None, None, 0)
        worker = Worker(self.store)
        with patch.object(worker, "_start_stage", return_value=process), \
                patch.object(worker, "heartbeat"), patch.object(worker, "poll_artifacts"), \
                patch.object(worker, "read_output"), patch.object(worker, "log"), \
                patch("supervisor.worker.time.sleep"), \
                patch("supervisor.worker.time.monotonic", side_effect=(1, 1000, 2000, 3000)), \
                patch.object(worker, "interrupt", return_value=False) as interrupt:
            worker._run_stage(task, None, ["fake"], self.paths.project_root, {})
        interrupt.assert_called_once_with(False, graceful_only=True)
        self.assertTrue(self.store.get("task", "live")["graceful_exit_issue"])

    def test_report_available_state_and_inspection_error_without_claiming_a_new_save(self):
        task = self.task("live", "running")
        state = Path(task["output_dir"]) / "save"
        state.mkdir(parents=True)
        adapter = Mock()
        adapter.resume_sources.return_value = [SimpleNamespace(kind="state", path=state, step=21, label="step 21")]
        adapter.collect_artifacts.return_value = ()
        with patch.dict("supervisor.worker.ADAPTERS", {"kohya": adapter}):
            worker = Worker(self.store)
            result = worker._stop_result(task)
            self.assertEqual((result["status"], result["sources"][0]["step"]), ("full_state", 21))
            self.assertEqual(result["sources"][0]["change"], "unknown")
            task["exit_savepoint_baseline"] = {}
            self.assertEqual(worker._stop_result(task)["sources"][0]["change"], "new")
            saved = state / "optimizer.bin"
            saved.write_bytes(b"old")
            baseline = {s["path"]: s["signature"] for s in worker._savepoints(task)}
            task["exit_savepoint_baseline"] = baseline
            self.assertEqual(worker._stop_result(task)["sources"][0]["change"], "existing")
            replacement = state / "temp.bin"
            replacement.write_bytes(b"new")
            replacement.replace(saved)
            self.assertEqual(worker._stop_result(task)["sources"][0]["change"], "updated")
            adapter.resume_sources.side_effect = OSError("unavailable")
            self.assertEqual(worker._stop_result(task)["status"], "unknown")

    def test_weights_on_disk_even_when_adapter_has_no_resume_source(self):
        from tests.test_functionality import write_safetensors
        task = self.task("live", "running")
        output = Path(task["output_dir"])
        output.mkdir()
        weights = write_safetensors(output / "weights.safetensors")
        adapter = Mock()
        adapter.resume_sources.return_value = ()
        adapter.collect_artifacts.return_value = (SimpleNamespace(kind="checkpoint", path=weights, step=12),)
        with patch.dict("supervisor.worker.ADAPTERS", {"kohya": adapter}):
            result = Worker(self.store)._stop_result(task)
        self.assertEqual((result["status"], result["sources"][0]["change"]), ("weights", "unknown"))

    def test_sse_disconnects_for_shutdown_even_with_a_window_still_listening(self):
        client = self.client()
        self.assertEqual(client.post("/api/v1/system/shutdown?graceful=true").status_code, 202)
        response = client.get("/api/v1/events")
        self.assertEqual((response.status_code, response.content), (200, b""))

    def test_exited_parent_and_abnormally_closed_pipe_cannot_complete_drain(self):
        task = self.task("live", "running")
        queue.request(self.store, graceful=True)
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        worker = Worker(self.store)
        with patch.object(worker, "_start_stage", return_value=process), \
                patch.object(worker, "read_output", return_value=None), \
                self.assertRaises(UnverifiedChildren):
            worker._run_stage(task, None, ["fake"], self.paths.project_root, {})
        self.assertNotEqual(queue.status(self.store)["status"], "ready_to_exit")

    @unittest.skipUnless(os.name == "nt", "requires Windows process accounting")
    def test_job_detects_descendant_after_parent_and_pipe_exit_without_killing_it(self):
        from supervisor.windows_interrupt import stage_process_options
        from supervisor.windows_stage_job import StageJob
        root = self.paths.project_root
        child = ("import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                 "(p/'started').touch(); deadline=time.monotonic()+10\n"
                 "while not (p/'leave').exists() and time.monotonic()<deadline: time.sleep(.05)\n"
                 "(p/'finished').touch()")
        parent = ("import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],"
                  "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
                  "creationflags=subprocess.DETACHED_PROCESS)")
        job = StageJob()
        try:
            process = job.start([sys.executable, "-c", parent, child, str(root)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **stage_process_options())
            output, _ = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, output)
            deadline = time.monotonic() + 4
            while not (root / "started").exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((root / "started").exists())
            worker = Worker(self.store)
            worker.stage_job = job
            with self.assertRaises(UnverifiedChildren):
                worker._verify_stage_tree()
            job.close()
        finally:
            job.close()
            (root / "leave").touch()
            deadline = time.monotonic() + 3
            while not (root / "finished").exists() and time.monotonic() < deadline:
                time.sleep(.02)
        self.assertTrue((root / "finished").exists())
