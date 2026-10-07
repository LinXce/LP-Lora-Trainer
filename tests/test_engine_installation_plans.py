"""Engine installation plans must delegate to each trainer's own installer."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from adapters.ai_toolkit.adapter import AiToolkitAdapter
from adapters.kohya.adapter import KohyaAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter


class EngineInstallationPlanTests(unittest.TestCase):
    def test_musubi_uses_project_uv_sync_and_declared_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "musubi_tuner").mkdir(parents=True)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "musubi-tuner"\n[project.optional-dependencies]\n'
                'cu124 = []\ncu128 = []\ncu130 = []\ncu132 = []\n', encoding="utf-8"
            )
            (root / "flux_train_network.py").write_text("", encoding="utf-8")
            adapter = MusubiTunerAdapter()

            for source, expected in (
                ("cu124", ("uv.exe", "sync", "--extra", "cu124")),
                ("cu128", ("uv.exe", "sync", "--extra", "cu128")),
                ("cu130", ("uv.exe", "sync", "--extra", "cu130")),
                ("cu132", ("uv.exe", "sync", "--extra", "cu132")),
                ("existing", ("uv.exe", "sync")),
            ):
                plan = adapter.installation_plan(root, "ignored", source, uv="uv.exe")
                self.assertEqual(plan.commands, (expected,))
                self.assertFalse(any("pip" in part for part in expected))

            with self.assertRaises(ValueError):
                adapter.installation_plan(root, "ignored", "cu126", uv="uv.exe")

    @unittest.skipUnless(os.name == "nt", "Kohya installer is Windows-only")
    def test_kohya_matches_official_setup_and_bootstraps_script_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup = root / "setup"
            setup.mkdir()
            (setup / "setup_common.py").write_text("VALUE = 'official-helper'\n", encoding="utf-8")
            (setup / "setup_windows.py").write_text(
                "import setup_common, sys\nprint(setup_common.VALUE, sys.argv[1:])\n", encoding="utf-8"
            )
            plan = KohyaAdapter().installation_plan(root, sys.executable, "existing")
            self.assertEqual(plan.commands[0], (
                sys.executable, "-u", "-m", "pip", "install", "--require-virtualenv",
                "--no-input", "-q", "setuptools",
            ))
            self.assertEqual(plan.commands[1][-2:], (str(setup / "setup_windows.py"), "--headless"))
            # Execute only the tiny fixture entry point, never pip or upstream setup.
            result = subprocess.run(plan.commands[1], cwd=root, capture_output=True,
                                    text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("official-helper ['--headless']", result.stdout)

    @unittest.skipUnless(os.name == "nt", "Kohya installer is Windows-only")
    def test_kohya_missing_official_installer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "setup/setup_windows.py"):
                KohyaAdapter().installation_plan(Path(directory), sys.executable, "existing")

    def test_ai_toolkit_calls_its_native_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "toolkit").mkdir()
            (root / "jobs").mkdir()
            (root / "run.py").write_text("", encoding="utf-8")
            (root / "manager").mkdir()
            (root / "manager" / "__main__.py").write_text("", encoding="utf-8")
            plan = AiToolkitAdapter().installation_plan(
                root, "python_runtime/python.exe", "cu124"
            )
            self.assertEqual(
                plan.commands,
                (("python_runtime/python.exe", "-u", str(root / "manager" / "__main__.py"), "install"),),
            )
            self.assertNotIn("pip", plan.commands[0])


if __name__ == "__main__":
    unittest.main()
