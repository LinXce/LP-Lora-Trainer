"""Kogine launch prefix: prefer the console launcher, survive a moved project."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter

LAUNCHER = "kogine.exe" if os.name == "nt" else "kogine"


def write_launcher(path: Path, interpreter: Path) -> None:
    """A distlib-shaped launcher: executable stub + shebang + appended zip."""
    payload = b"MZstub" + b"#!" + str(interpreter).encode("utf-8") + b"\r\nPK\x03\x04payload"
    path.write_bytes(payload)


class KogineLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environment = self.root / "engine" / ".lp" / "python"
        self.scripts = self.environment / ("Scripts" if os.name == "nt" else "bin")
        self.scripts.mkdir(parents=True)
        self.python = self.environment / ("python.exe" if os.name == "nt" else "python")
        self.python.write_bytes(b"")
        self.launcher = self.scripts / LAUNCHER

    def tearDown(self):
        self.temp.cleanup()

    def test_current_launcher_is_used_directly(self):
        write_launcher(self.launcher, self.python)
        self.assertEqual(
            KohakuMossTrainerAdapter._kogine_launcher(str(self.python)),
            (str(self.launcher),),
        )

    def test_quoted_interpreter_path_with_spaces_is_current(self):
        environment = self.root / "dir with spaces"
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        scripts.mkdir(parents=True)
        spaced = environment / ("python.exe" if os.name == "nt" else "python")
        spaced.write_bytes(b"")
        launcher = scripts / LAUNCHER
        launcher.write_bytes(b"MZstub" + b'#!"' + str(spaced).encode("utf-8") + b'"\r\nPK\x03\x04')
        self.assertEqual(
            KohakuMossTrainerAdapter._kogine_launcher(str(spaced)),
            (str(launcher),),
        )

    def test_launcher_from_previous_location_falls_back_to_interpreter(self):
        write_launcher(self.launcher, self.root / "old-drive" / "python.exe")
        self.assertEqual(
            KohakuMossTrainerAdapter._kogine_launcher(str(self.python)),
            (
                str(self.python),
                "-c",
                "from kohakuengine.cli import main; raise SystemExit(main())",
            ),
        )

    def test_missing_launcher_falls_back_to_interpreter(self):
        self.assertEqual(
            KohakuMossTrainerAdapter._kogine_launcher(str(self.python)),
            (
                str(self.python),
                "-c",
                "from kohakuengine.cli import main; raise SystemExit(main())",
            ),
        )

    def test_preview_commands_flatten_the_launcher_prefix(self):
        write_launcher(self.launcher, self.python)
        adapter = KohakuMossTrainerAdapter()
        installation = SimpleNamespace(
            python_executable=str(self.python), source_path=str(self.root / "engine")
        )
        commands = adapter.preview_commands(installation, {"params": {}}, str(self.root / "config.py"))
        self.assertTrue(commands)
        for command in commands:
            self.assertEqual(command[0], str(self.launcher))
            self.assertEqual(command[1], "run")
        self.assertEqual(commands[-1][2], str(self.root / "engine" / "scripts" / "train" / "moss.py"))

    def test_preview_commands_fall_back_after_a_move(self):
        write_launcher(self.launcher, self.root / "old-drive" / "python.exe")
        adapter = KohakuMossTrainerAdapter()
        installation = SimpleNamespace(
            python_executable=str(self.python), source_path=str(self.root / "engine")
        )
        commands = adapter.preview_commands(installation, {"params": {}}, str(self.root / "config.py"))
        self.assertTrue(commands)
        for command in commands:
            self.assertEqual(command[:2], (str(self.python), "-c"))
            self.assertEqual(command[3], "run")
