"""Exercise the real Windows console boundary with two isolated mock engines."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from supervisor.windows_interrupt import send_interrupt, stage_process_options


@unittest.skipUnless(os.name == "nt", "Windows console control events")
class WindowsEngineInterruptTests(unittest.TestCase):
    def test_only_requested_stage_receives_ctrl_c_and_can_save(self):
        script = """
import signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def stop(sig, frame):
    (root / 'checkpoint').write_text('saved')
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
(root / 'started').touch()
while True: time.sleep(.05)
"""
        with tempfile.TemporaryDirectory() as directory:
            children = []
            try:
                roots = [Path(directory) / name for name in ("owned", "unrelated")]
                for root in roots:
                    root.mkdir()
                    children.append(subprocess.Popen(
                        [sys.executable, "-c", script, str(root)], stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **stage_process_options()))
                deadline = time.monotonic() + 10
                while not all((root / "started").exists() for root in roots):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                self.assertTrue(send_interrupt(children[0]))
                self.assertEqual(children[0].wait(timeout=10), 0)
                self.assertEqual((roots[0] / "checkpoint").read_text(), "saved")
                self.assertIsNone(children[1].poll())
                self.assertFalse((roots[1] / "checkpoint").exists())
            finally:
                for process in children:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
