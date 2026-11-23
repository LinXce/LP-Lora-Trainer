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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import ensure_engine_venv  # noqa: E402
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
        env = patch.dict(os.environ, {"UV_PYTHON_INSTALL_DIR": str(MANAGED), "UV_OFFLINE": "1"})
        env.start()
        self.addCleanup(env.stop)
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

    def test_reuse_repairs_missing_pip_without_another_download_or_copy(self):
        site = self.environment / "Lib/site-packages"
        (site / "user_dependency.py").write_text("enabled = True")
        (site / "pip/__main__.py").unlink()
        with patch.object(ensure_engine_venv, "_managed_python_dir", side_effect=AssertionError("unexpected download")):
            create_embedded(self.engine, self.environment, "3.12", UV)
        subprocess.run([str(self._python(self.environment)), "-I", "-c", "import pip, user_dependency"], check=True)
        self.assertFalse(list(self.environment.parent.glob("python.broken-*")))

    def test_changed_version_is_checked_before_replacing_the_existing_tree(self):
        with patch.object(ensure_engine_venv, "_managed_python_dir", return_value=self.environment):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                create_embedded(self.engine, self.environment, "3.11", UV)
        self.assertTrue(ensure_engine_venv._runtime_matches(self.environment, "3.12"))
        self.assertFalse(list(self.environment.parent.glob("python.broken-*")))


class ManagedPythonAcquisitionTests(unittest.TestCase):
    def test_empty_cache_downloads_once_with_the_selected_mirror(self):
        env = {"UV_PYTHON_INSTALL_MIRROR": "https://mirror.invalid/python"}
        uv = Path("uv")
        with patch.object(ensure_engine_venv, "_find_managed", side_effect=[None, Path("downloaded")]), \
                patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(ensure_engine_venv._managed_python_dir(uv, "3.13", env), Path("downloaded"))
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["uv", "python", "install"])
        self.assertIn("--no-bin", command)
        if os.name == "nt":
            self.assertIn("--no-registry", command)
        self.assertEqual(run.call_args.kwargs["env"], env)

    def test_reuses_per_version_cache_from_an_older_install(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root) / "3.11"
            legacy.mkdir()
            with patch.object(ensure_engine_venv, "_find_managed", side_effect=[None, legacy]) as find, \
                    patch.object(subprocess, "run", side_effect=AssertionError("must stay offline")):
                source = ensure_engine_venv._managed_python_dir(Path("uv"), "3.11", {"UV_PYTHON_INSTALL_DIR": root})
            self.assertEqual(source, legacy)
            self.assertEqual(find.call_args.args[-1]["UV_PYTHON_INSTALL_DIR"], str(legacy))


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
            self.assertEqual(lines.count("../.."), 1)
            self.assertFalse(any("../..  #" in line for line in lines))
            self.assertEqual(lines.count("Lib"), 1)


if __name__ == "__main__":
    unittest.main()
