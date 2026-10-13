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
    def test_kohya_plan_installs_the_declared_requirements_with_uv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("numpy==1.26.4\n-e ./sd-scripts\n", encoding="utf-8")
            (root / "requirements_pytorch_windows.txt").write_text("torch==2.7.0+cu128\n", encoding="utf-8")
            (root / "sd-scripts").mkdir()
            plan = KohyaAdapter().installation_plan(
                root, sys.executable, "existing", uv="uv.exe",
                torch_mirror_url="https://mirror.example/pytorch-wheels/cu128",
            )
            self.assertEqual(len(plan.commands), 1)
            install = plan.commands[0]
            self.assertEqual(install[:4], ("uv.exe", "pip", "install", "--python"))
            self.assertIn(sys.executable, install)
            self.assertIn("--no-build-isolation", install)
            self.assertIn("setuptools<82", install)
            self.assertIn("numpy==1.26.4", install)
            self.assertIn("torch==2.7.0+cu128", install)
            self.assertIn("https://mirror.example/pytorch-wheels/cu128/torch/", install)
            self.assertIn(str((root / "sd-scripts").resolve()), install)
            # Neither the engine's installer nor its requirement files: those
            # carry the extra index that made pip go silent for minutes.
            self.assertFalse(any("setup_windows.py" in str(argument) for argument in plan.commands))
            self.assertNotIn("-r", install)

    def test_kohya_plan_falls_back_to_pip_without_uv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
            plan = KohyaAdapter().installation_plan(root, sys.executable, "existing")
            self.assertEqual(plan.commands[0][:5], (
                sys.executable, "-u", "-m", "pip", "install",
            ))
            self.assertIn("numpy==1.26.4", plan.commands[0])

    def test_kohya_declares_the_pinned_upstream_reference(self):
        from adapters.registry import adapter_catalog
        adapters = {a["engine_id"]: a for a in adapter_catalog()["adapters"]}
        kohya = adapters["kohya"]
        self.assertTrue(kohya["clone_url"].endswith("kohya_ss.git"))
        self.assertEqual(kohya["clone_branch"], "main")
        self.assertEqual(kohya["clone_commit"], "45088f04af78e11cec5407ff4652ea3ed2c14422")
        # An adapter that pins nothing reports empty strings, never a guess.
        self.assertEqual(adapters["musubi_tuner"]["clone_commit"], "")

    def test_kohya_source_repair_fetches_a_declared_submodule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kohya_gui.py").write_text("# wrapper", encoding="utf-8")
            (root / "kohya_gui").mkdir()
            (root / ".git").mkdir()
            (root / ".gitmodules").write_text(
                '[submodule "sd-scripts"]\n\tpath = sd-scripts\n\turl = https://example.invalid/sd-scripts.git\n',
                encoding="utf-8",
            )
            adapter = KohyaAdapter()
            self.assertTrue(adapter.source_issues(root))
            self.assertTrue(adapter.source_repair_label)
            self.assertEqual(adapter.source_repair(root), (
                "git", "-c", "core.fsmonitor=false", "submodule", "update", "--init", "--recursive",
            ))
            # A complete tree needs no repair step.
            scripts = root / "sd-scripts"
            (scripts / "library").mkdir(parents=True)
            (scripts / "train_network.py").write_text("# training", encoding="utf-8")
            (scripts / "library" / "train_util.py").write_text("# library", encoding="utf-8")
            self.assertEqual(adapter.source_issues(root), ())
            self.assertIsNone(adapter.source_repair(root))

    def test_kohya_source_repair_clones_into_a_copied_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kohya_gui.py").write_text("# wrapper", encoding="utf-8")
            (root / "kohya_gui").mkdir()
            # Copied without .git (an unpacked archive), but the declared URL is
            # still usable, so a plain clone fills the empty submodule folder.
            (root / ".gitmodules").write_text(
                '[submodule "sd-scripts"]\n\tpath = sd-scripts\n\turl = https://example.invalid/sd-scripts.git\n',
                encoding="utf-8",
            )
            (root / "sd-scripts").mkdir()
            adapter = KohyaAdapter()
            self.assertTrue(adapter.source_issues(root))
            self.assertEqual(adapter.source_repair(root), (
                "git", "-c", "core.fsmonitor=false", "clone", "--depth", "1",
                "https://example.invalid/sd-scripts.git", str(root / "sd-scripts"),
            ))

    def test_kohya_source_repair_needs_a_declared_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kohya_gui.py").write_text("# wrapper", encoding="utf-8")
            (root / "kohya_gui").mkdir()
            # Declared without a URL and without a checkout: nothing to fetch from.
            (root / ".gitmodules").write_text(
                '[submodule "sd-scripts"]\n\tpath = sd-scripts\n',
                encoding="utf-8",
            )
            adapter = KohyaAdapter()
            self.assertTrue(adapter.source_issues(root))
            self.assertIsNone(adapter.source_repair(root))

    def test_other_adapters_declare_no_source_repair(self):
        from adapters.ai_toolkit.adapter import AiToolkitAdapter
        from adapters.musubi_tuner.adapter import MusubiTunerAdapter
        for adapter in (AiToolkitAdapter(), MusubiTunerAdapter()):
            with self.subTest(engine=adapter.engine_id):
                self.assertEqual(adapter.source_repair_label, "")
                self.assertIsNone(adapter.source_repair(Path(".")))

    def test_kohya_reports_local_requirement_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text(
                "numpy==1.26.4\n"
                "-r requirements_pytorch_windows.txt\n"
                "# for kohya_ss sd-scripts library\n"
                "-e ./sd-scripts\n",
                encoding="utf-8",
            )
            (root / "requirements_pytorch_windows.txt").write_text(
                "torch==2.7.0+cu128\n",
                encoding="utf-8",
            )
            adapter = KohyaAdapter()
            self.assertEqual(adapter.local_requirement_paths(root), ("./sd-scripts",))

    def test_kohya_reports_no_local_paths_without_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
            self.assertEqual(KohyaAdapter().local_requirement_paths(root), ())
            self.assertEqual(KohyaAdapter().local_requirement_paths(root / "missing"), ())

    def test_other_adapters_declare_no_local_requirements(self):
        from adapters.musubi_tuner.adapter import MusubiTunerAdapter
        self.assertEqual(MusubiTunerAdapter().local_requirement_paths(Path(".")), ())

    def test_kohya_reports_the_exact_wheel_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements_pytorch_windows.txt").write_text(
                "# comment\n"
                "--extra-index-url https://download.pytorch.org/whl/cu128\n"
                "torch==2.7.0+cu128\n"
                "torchvision==0.22.0+cu128\n"
                "numpy==1.26.4\n"
                "xformers==0.0.29.post3\n",
                encoding="utf-8",
            )
            adapter = KohyaAdapter()
            # Only the local-version pins: those exist solely on the engine's
            # own wheel index, and only they make the resolver search it.
            self.assertEqual(adapter.pinned_wheel_requirements(root),
                             ("torch==2.7.0+cu128", "torchvision==0.22.0+cu128"))
            self.assertEqual(adapter.torch_index_tag(root), "cu128")

    def test_kohya_reports_no_pins_without_local_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements_pytorch_windows.txt").write_text("torch==2.7.0\n", encoding="utf-8")
            self.assertEqual(KohyaAdapter().pinned_wheel_requirements(root), ())
            self.assertEqual(KohyaAdapter().pinned_wheel_requirements(root / "missing"), ())

    def test_other_adapters_declare_no_wheel_pins(self):
        from adapters.musubi_tuner.adapter import MusubiTunerAdapter
        self.assertEqual(MusubiTunerAdapter().pinned_wheel_requirements(Path(".")), ())

    def test_kohya_reports_the_torch_tag_its_requirements_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements_pytorch_windows.txt").write_text(
                "# comment\n"
                "--extra-index-url https://download.pytorch.org/whl/cu128\n"
                "torch==2.7.0+cu128\n"
                "torchvision==0.22.0+cu128\n",
                encoding="utf-8",
            )
            self.assertEqual(KohyaAdapter().torch_index_tag(root), "cu128")

    def test_kohya_reports_no_tag_without_a_pinned_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements_pytorch_windows.txt").write_text("torch==2.7.0\n", encoding="utf-8")
            self.assertIsNone(KohyaAdapter().torch_index_tag(root))
            self.assertIsNone(KohyaAdapter().torch_index_tag(root / "missing"))

    def test_other_adapters_declare_no_torch_tag(self):
        from adapters.musubi_tuner.adapter import MusubiTunerAdapter
        self.assertIsNone(MusubiTunerAdapter().torch_index_tag(Path(".")))

    @unittest.skipUnless(os.name == "nt", "Kohya installer is Windows-only")
    def test_kohya_plan_without_requirements_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requirements"):
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
