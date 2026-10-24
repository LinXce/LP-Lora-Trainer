"""Engine terminal must use its own pip, not a system Python's pip."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop.bridge import DesktopBridge, terminal_shims


class TerminalShimTests(unittest.TestCase):
    def test_pip_shim_targets_the_bound_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "python.exe"
            exe.write_bytes(b"fake")
            shims = terminal_shims(exe)
            try:
                pip = (shims / "pip.cmd").read_text(encoding="utf-8")
                self.assertIn(f'"{exe}" -m pip --no-input --disable-pip-version-check %*', pip)
            finally:
                for name in ("pip.cmd", "pip3.cmd", "python.cmd", "python3.cmd"):
                    (shims / name).unlink(missing_ok=True)
                shims.rmdir()

    def test_open_terminal_prepends_the_shim_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "python.exe"
            exe.write_bytes(b"fake")
            with patch("desktop.bridge.subprocess.Popen") as popen:
                DesktopBridge().open_terminal(str(exe))
                kwargs = popen.call_args.kwargs
                path = kwargs["env"]["PATH"].split(";")
                self.assertIn("lp-terminal-shims", path[0])
                self.assertEqual(kwargs["cwd"], str(exe.resolve().parent))
                self.assertIn(f'"{exe.resolve()}" -V', popen.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
