"""The embedded (relocatable) engine environment.

These tests build a real embedded CPython from the project's managed interpreter and
move it, because relocation is the whole point: a `.pth`-only or venv-based environment
passes creation and fails the move.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ensure_engine_venv import create_embedded  # noqa: E402
from scripts.normalize_editable import _rewrite_embedded_pth  # noqa: E402

UV = ROOT / "runtime" / "uv" / "uv.exe"
MANAGED = ROOT / "runtime" / "uv" / "python"


def _managed_available() -> bool:
    return os.name == "nt" and UV.is_file() and any(MANAGED.glob("cpython-3.12*"))


@unittest.skipUnless(_managed_available(), "no managed CPython 3.12 in this project")
class EmbeddedEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="lp-embedded-"))
        self.engine = self.root / "engine" / "demo"
        self.engine.mkdir(parents=True)
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(MANAGED)
        self.environment = self.engine / ".lp" / "python"
        create_embedded(self.engine, self.environment, "3.12", UV)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _python(self, environment: Path) -> Path:
        return environment / "python.exe"

    def test_layout_has_no_venv_metadata(self):
        self.assertTrue(self._python(self.environment).is_file())
        self.assertFalse((self.environment / "pyvenv.cfg").is_file())
        # A managed CPython ships its own Scripts/, but never a venv activator.
        self.assertFalse(any(self.environment.glob("Scripts/activate*")))
        self.assertTrue((self.environment / "python312._pth").is_file())

    def test_interpreter_reports_itself(self):
        probe = subprocess.run(
            [str(self._python(self.environment)), "-I", "-c", "import sys; print(sys.prefix)"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(Path(probe.stdout.strip()), self.environment)

    def test_environment_still_runs_after_a_move(self):
        moved = self.root / "moved" / "demo"
        shutil.move(str(self.engine), str(moved))
        after = moved / ".lp" / "python" / "python.exe"
        probe = subprocess.run([str(after), "-I", "-c", "print('ok')"],
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout.strip(), "ok")


class EmbeddedPthRewriteTests(unittest.TestCase):
    def test_duplicate_editable_roots_are_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / ".lp" / "python"
            environment.mkdir(parents=True)
            pth = environment / "python312._pth"
            pth.write_text(
                ".\nLib\nLib/site-packages\n../..\n../..  # LP editable: relative import root\n",
                encoding="utf-8",
            )
            _rewrite_embedded_pth(pth, environment, ["../..", "../.."])
            # A second run must replace the generated block, not append to it.
            _rewrite_embedded_pth(pth, environment, ["../.."])
            lines = pth.read_text(encoding="utf-8").splitlines()
            editable = [line for line in lines if "# LP editable:" in line]
            self.assertEqual(editable, ["../..  # LP editable: relative import root"])
            self.assertEqual(lines.count("Lib"), 1)


if __name__ == "__main__":
    unittest.main()
