import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from adapters.sd_scripts.adapter import KohyaAdapter
from adapters.sd_scripts.run_control import stop_command
from app.storage.state import StateStore
from supervisor.worker import Worker


class RunControlTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.output = self.root / "portrait--id"
        self.run_dir = self.output / "portrait_run"
        self.run_dir.mkdir(parents=True)
        self.task = dict(task_id="id", state="running", engine_id="kohya", output_verified=True,
                         output_dir=str(self.output), pid=123, recovery={"graceful_stop": True})
        self.config = self.root / "config.native.toml"
        self.config.write_text(f"output_dir = {json.dumps(str(self.output))}\noutput_name = 'portrait'\n")
        self.argv = ("python", "train_network.py", "--config_file", str(self.config), "--run_control")

    def write_status(self, pid=123, state="running"):
        (self.run_dir / "status.json").write_text(json.dumps({"pid": pid, "state": state}))

    def test_unready_old_finished_or_foreign_trainer_receives_no_command(self):
        self.assertFalse(stop_command(self.task, self.argv))
        for pid, state in ((123, "stopped"), (456, "running")):
            self.write_status(pid, state)
            self.assertFalse(stop_command(self.task, self.argv))
        self.write_status()
        self.assertFalse(stop_command(self.task, self.argv[:-1]))
        self.assertFalse(stop_command(dict(self.task, output_dir=str(self.root)), self.argv))
        self.assertFalse(stop_command(dict(self.task, output_verified=False), self.argv))
        self.assertFalse((self.run_dir / "commands").exists())

    def stop_stage(self, stop):
        task = dict(self.task, pid=None, stop_requested=stop)
        store = StateStore(self.root / "db")
        store.put("task", "id", task)
        worker = Worker(store)
        process = Mock(pid=123, returncode=0)
        process.poll.side_effect = (None, 0)
        with patch.object(worker, "_start_stage", return_value=process), \
                patch.object(worker, "_savepoints", return_value=[]), \
                patch.object(worker, "poll_artifacts"), patch.object(worker, "read_output"), \
                patch.object(worker, "log"), patch.object(worker, "interrupt") as interrupt:
            worker._run_stage(task, KohyaAdapter(), self.argv, self.root, {})
        return interrupt

    def test_supervisor_prefers_native_save_and_stop_for_ready_trainer(self):
        for stop in ("stop", "graceful_exit"):
            with self.subTest(stop=stop):
                self.write_status()
                self.stop_stage(stop).assert_not_called()
                commands = list((self.run_dir / "commands").glob("*.json"))
                self.assertEqual(len(commands), 1)
                self.assertEqual(json.loads(commands[0].read_text()),
                                 {"command": "stop", "save": True, "save_state": True})
                commands[0].unlink()

    def test_supervisor_uses_normal_interrupt_before_controller_start(self):
        self.stop_stage("graceful_exit").assert_called_once_with(False, graceful_only=True)
        self.assertFalse((self.run_dir / "commands").exists())
