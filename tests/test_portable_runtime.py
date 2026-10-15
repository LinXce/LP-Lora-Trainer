"""Portable distribution contract: relative paths, dependency closure and relocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import call
import unittest
from unittest.mock import patch

from scripts.build_python_runtime import child_exists, package_file, preflight_runtime_publish, publish_runtime, resolve_dependencies, secure_staging_directory, write_path_config, _packaging_types

try:
    _packaging_types()
    HAS_BUILD_DEPS = True
except RuntimeError:
    HAS_BUILD_DEPS = False


WORKSPACE = Path(__file__).resolve().parents[1]


def distribution(version="1.0", requires=()):
    return SimpleNamespace(version=version, requires=requires)


class RuntimeBuilderTests(unittest.TestCase):
    def test_publish_preflight_round_trips_staging_and_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            staging = workspace / ".python-runtime-build-test"
            output = workspace / "python_runtime"
            staging.mkdir()
            output.mkdir()
            (staging / "new.txt").write_text("new", encoding="utf-8")
            (output / "old.txt").write_text("old", encoding="utf-8")

            preflight_runtime_publish(workspace, staging, output)

            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse(any(workspace.glob(".python-runtime-publish-check-*")))

    def test_publish_preflight_stops_before_build_if_runtime_cannot_move(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            staging = workspace / ".python-runtime-build-test"
            output = workspace / "python_runtime"
            staging.mkdir()
            output.mkdir()
            original_rename = Path.rename

            def deny_runtime_move(path, target):
                if path == output:
                    raise PermissionError(5, "Access is denied")
                return original_rename(path, target)

            with patch.object(Path, "rename", deny_runtime_move):
                with self.assertRaisesRegex(PermissionError, "No runtime files were changed"):
                    preflight_runtime_publish(workspace, staging, output)

            self.assertTrue(staging.is_dir())
            self.assertTrue(output.is_dir())
            self.assertFalse(any(workspace.glob(".python-runtime-publish-check-*")))

    @unittest.skipUnless(os.name == "nt", "Windows ACLs are required")
    def test_secure_staging_directory_can_be_renamed_under_parent_delete_deny(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            staging = None
            try:
                subprocess.run(
                    ["icacls.exe", str(workspace), "/deny", "*S-1-1-0:(DC)"],
                    check=True, capture_output=True, text=True,
                )
                staging = Path(tempfile.mkdtemp(prefix=".python-runtime-build-", dir=workspace))
                secure_staging_directory(staging)
                destination = workspace / "published-runtime"

                staging.rename(destination)

                self.assertTrue(destination.is_dir())
                self.assertFalse(staging.exists())
            finally:
                if staging is not None:
                    subprocess.run(
                        ["icacls.exe", str(workspace), "/remove:d", "*S-1-1-0"],
                        check=True, capture_output=True, text=True,
                    )

    def test_publish_reports_parent_permission_denial_without_moving_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "python_runtime"
            staging = workspace / ".python-runtime-build-test"
            output.mkdir()
            staging.mkdir()
            (output / "existing.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")

            with patch.object(Path, "rename", side_effect=FileNotFoundError(2, "Path is inaccessible")):
                with self.assertRaisesRegex(PermissionError, "parent still contains the runtime directory"):
                    publish_runtime(workspace, staging, output)

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(any(workspace.glob("python_runtime.previous-*")))

    def test_publish_does_not_assume_file_not_found_means_runtime_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "python_runtime"
            staging = workspace / ".python-runtime-build-test"
            output.mkdir()
            staging.mkdir()
            (output / "existing.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            original_rename = Path.rename

            def rename_with_inaccessible_runtime(path, target):
                if path == output:
                    raise FileNotFoundError(2, "Path is inaccessible")
                return original_rename(path, target)

            with patch.object(Path, "rename", rename_with_inaccessible_runtime):
                with self.assertRaisesRegex(PermissionError, "Unable to move existing runtime"):
                    publish_runtime(workspace, staging, output)

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(any(workspace.glob("python_runtime.previous-*")))

    def test_publish_restores_previous_runtime_and_preserves_staging_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "python_runtime"
            staging = workspace / ".python-runtime-build-test"
            output.mkdir()
            staging.mkdir()
            (output / "existing.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            original_rename = Path.rename

            def rename_with_failed_publish(path, target):
                if path == staging:
                    raise PermissionError(5, "Access is denied")
                return original_rename(path, target)

            with patch.object(Path, "rename", rename_with_failed_publish):
                with self.assertRaisesRegex(PermissionError, "new runtime remains at"):
                    publish_runtime(workspace, staging, output)

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(any(workspace.glob("python_runtime.previous-*")))

    def test_publish_retries_transient_windows_access_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "python_runtime"
            staging = workspace / ".python-runtime-build-test"
            output.mkdir()
            staging.mkdir()
            original_rename = Path.rename
            attempts = 0

            def fail_twice(path, target):
                nonlocal attempts
                if path == staging:
                    attempts += 1
                    if attempts < 3:
                        error = PermissionError("Access is denied")
                        error.winerror = 5
                        raise error
                return original_rename(path, target)

            with patch.object(Path, "rename", fail_twice), patch("scripts.build_python_runtime.time.sleep") as sleep:
                publish_runtime(workspace, staging, output)

            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_args_list, [call(0.5), call(1)])
            self.assertTrue(output.is_dir())
            self.assertFalse(staging.exists())

    def test_publish_moves_old_runtime_to_backup_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            output = workspace / "python_runtime"
            staging = workspace / ".python-runtime-build-test"
            output.mkdir()
            staging.mkdir()
            (output / "existing.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")

            publish_runtime(workspace, staging, output)

            backups = list(workspace.glob("python_runtime.previous-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "existing.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((output / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(staging.exists())

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

    def test_dotnet_config_allows_portable_netfx_assembly_loading(self):
        from scripts.build_python_runtime import write_dotnet_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dotnet_config(root)
            expected = '<loadFromRemoteSources enabled="true" />'
            for executable in ("python.exe", "pythonw.exe"):
                config = (root / f"{executable}.config").read_text(encoding="utf-8")
                self.assertIn(expected, config)
                self.assertTrue(config.startswith('<?xml version="1.0"'))

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
