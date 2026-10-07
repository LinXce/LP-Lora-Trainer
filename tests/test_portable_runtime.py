"""Portable distribution contract: relative paths, dependency closure and relocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.build_python_runtime import package_file, resolve_dependencies, write_path_config, _packaging_types

try:
    _packaging_types()
    HAS_BUILD_DEPS = True
except RuntimeError:
    HAS_BUILD_DEPS = False


WORKSPACE = Path(__file__).resolve().parents[1]


def distribution(version="1.0", requires=()):
    return SimpleNamespace(version=version, requires=requires)


class RuntimeBuilderTests(unittest.TestCase):
    def test_record_paths_cannot_escape_site_packages(self):
        for value in ("../../Scripts/pip.exe", "../other.py", "/outside.py", "C:/outside.py"):
            self.assertIsNone(package_file(value))
        self.assertEqual(str(package_file("namespace/package/data.json")), "namespace/package/data.json")
        self.assertIsNone(package_file("package/__pycache__/old.pyc"))
        self.assertIsNone(package_file("example.dist-info/direct_url.json"))

    def test_editable_and_path_hooks_are_rejected(self):
        for value in ("example.pth", "__editable__.example.py"):
            with self.assertRaises(ValueError):
                package_file(value)

    @unittest.skipUnless(HAS_BUILD_DEPS, "Release-builder packaging dependency is not part of the app runtime")
    def test_dependency_extras_propagate_but_unused_extras_do_not(self):
        installed = {
            "root": distribution(requires=("child[feature]>=1", "never; extra == 'test'")),
            "child": distribution(requires=("included; extra == 'feature'",)),
            "included": distribution(),
        }
        selected = resolve_dependencies(["root>=1"], lookup=installed.__getitem__)
        self.assertEqual(set(selected), {"root", "child", "included"})

    @unittest.skipUnless(HAS_BUILD_DEPS, "Release-builder packaging dependency is not part of the app runtime")
    def test_later_extra_reprocesses_an_already_resolved_dependency(self):
        installed = {
            "child": distribution(requires=("included; extra == 'feature'",)),
            "included": distribution(),
        }
        selected = resolve_dependencies(["child[feature]", "child"], lookup=installed.__getitem__)
        self.assertEqual(set(selected), {"child", "included"})

    @unittest.skipUnless(HAS_BUILD_DEPS, "Release-builder packaging dependency is not part of the app runtime")
    def test_version_and_source_requirements_fail_closed(self):
        with self.assertRaises(ValueError):
            resolve_dependencies(["example>=2"], lookup=lambda _: distribution("1.0"))
        with self.assertRaises(ValueError):
            resolve_dependencies(["example @ https://example.invalid/example.whl"])
        with self.assertRaises(ValueError):
            resolve_dependencies(["lp-lora-trainer"])

    def test_path_config_contains_only_relative_paths_and_disables_site(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_path_config(root, (3, 12))
            text = (root / "python312._pth").read_text(encoding="utf-8")
            lines = [value for value in text.splitlines() if not value.startswith("#")]
            self.assertEqual(lines, [".", "Lib", "Lib/site-packages", "DLLs", ".."])
            self.assertNotIn("import site", text)
            self.assertTrue(all(not Path(value).is_absolute() for value in lines))

    def test_windows_launchers_never_fall_back_to_system_python(self):
        for name in ("start.cmd", "start-api.cmd", "check-runtime.cmd"):
            text = (WORKSPACE / name).read_text(encoding="ascii")
            self.assertIn('"%~dp0python_runtime\\python.exe"', text)
            self.assertNotIn("activate", text.lower())
            self.assertNotIn("H:\\", text)
            self.assertNotIn("D:\\", text)


@unittest.skipUnless(os.name == "nt" and (WORKSPACE / "python_runtime" / "python.exe").is_file(),
                     "Build the Windows portable runtime to enable relocation checks")
class PortableRelocationTests(unittest.TestCase):
    def test_copy_to_new_non_ascii_directory_without_system_python(self):
        with tempfile.TemporaryDirectory(prefix="LP portable ") as directory:
            root = Path(directory) / "portable copy \u4e2d\u6587"
            root.mkdir()
            for name in ("python_runtime", "app", "desktop", "supervisor", "adapters"):
                shutil.copytree(WORKSPACE / name, root / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (root / "scripts").mkdir()
            shutil.copy2(WORKSPACE / "scripts" / "check_runtime.py", root / "scripts" / "check_runtime.py")
            env = os.environ.copy()
            env.update(PYTHONHOME="Z:\\not-installed", PYTHONPATH="Z:\\missing-libraries",
                       PYTHONUSERBASE="Z:\\missing-user-site")
            env["PATH"] = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
            exe = root / "python_runtime" / "python.exe"
            result = subprocess.run([str(exe), str(root / "scripts" / "check_runtime.py")],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["ok"])
            self.assertEqual(Path(report["runtime"]).resolve(), exe.parent.resolve())
            # Both production entrypoints resolve even from an unrelated cwd.
            for module in ("desktop.launcher", "app.api.server", "supervisor.worker"):
                result = subprocess.run([str(exe), "-m", module, "--help"], cwd=directory,
                                        env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
