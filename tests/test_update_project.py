"""update_project.py: the update path must never touch user data."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_project", ROOT / "scripts" / "update_project.py")
update_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_project)


class UpdateProjectTests(unittest.TestCase):
    def test_user_data_is_declared_protected(self):
        for name in ("data/", "output/", "engine/", "runtime/", "python_runtime/", "frontend/dist/"):
            self.assertIn(name, update_project.PROTECTED)

    def test_git_ignores_every_protected_directory(self):
        """A pull can only overwrite tracked files, so these must stay untracked."""
        completed = subprocess.run(
            ["git", "ls-files", *[p.rstrip("/") for p in update_project.PROTECTED]],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if completed.returncode != 0:
            self.skipTest("git is not available here")
        tracked = [line for line in completed.stdout.splitlines() if line.strip()]
        # The repository keeps one placeholder file so an empty directory exists.
        tracked = [line for line in tracked if line != "output/output will be here"]
        self.assertEqual(tracked, [], f"protected paths are tracked: {tracked}")

    def test_dirty_tracked_ignores_untracked_and_ignored_files(self):
        """Untracked user data must not block an update; edited source must."""
        original = update_project.git
        try:
            update_project.git = lambda *args, check=True: (
                " M app/services/engines.py\n"
                "?? output/run.safetensors\n"
                "!! data/state.sqlite\n"
                "M  README.md\n"
            )
            self.assertEqual(
                update_project.dirty_tracked(),
                ["app/services/engines.py", "README.md"],
            )
        finally:
            update_project.git = original

    def test_rebuild_hints_follow_the_changed_paths(self):
        self.assertEqual(update_project.summarize(["README.md"]), [])
        self.assertTrue(any("frontend" in line for line in update_project.summarize(["frontend/src/pages/App.vue"])))
        self.assertTrue(any("LP-Lora-Trainer.exe" in line for line in update_project.summarize(["desktop/launcher_host.cs"])))

    def test_check_mode_needs_a_git_worktree(self):
        original = update_project.run
        try:
            update_project.run = lambda args, check=True: subprocess.CompletedProcess(args, 1, "", "not a repository")
            with self.assertRaises(RuntimeError):
                update_project.ensure_repository()
        finally:
            update_project.run = original


if __name__ == "__main__":
    unittest.main()
