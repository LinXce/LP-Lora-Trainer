"""Engine source updates: fast-forward only, and environments never reach the index."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from adapters.base import EngineAdapter
from adapters.registry import ADAPTERS

GIT = ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test"]


def git(directory, *argv):
    return subprocess.run(GIT + list(argv), cwd=directory, capture_output=True, text=True)


def dirty(directory):
    """The exact check the update uses: tracked changes only."""
    out = subprocess.run(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


class UpdateCommandsTests(unittest.TestCase):
    def test_every_engine_with_upstream_declares_fetch_then_ff_only_merge(self):
        for engine_id, adapter in ADAPTERS.items():
            with self.subTest(engine=engine_id):
                # A real checkout with .git is required before an update is offered.
                if not (Path("engine") / engine_id).is_dir():
                    continue
                source = Path("engine") / engine_id
                commands = adapter.update_commands(source)
                if not adapter.clone_url or not (source / ".git").is_dir():
                    self.assertEqual((), commands)
                    continue
                self.assertEqual(2, len(commands))
                self.assertIn("fetch", commands[0])
                self.assertIn("--ff-only", commands[1])
                for command in commands:
                    for forbidden in ("add", "commit", "stash", "reset", "clean", "push"):
                        self.assertNotIn(forbidden, command, "更新流程不得暂存或改写本地状态")

    def test_engine_without_upstream_offers_no_update(self):
        class Bundled(EngineAdapter):
            engine_id = "bundled"

            def detect(self, source_path):  # pragma: no cover - not exercised
                raise NotImplementedError

            def capabilities(self, installation):  # pragma: no cover
                raise NotImplementedError

            def validate(self, installation, config):  # pragma: no cover
                raise NotImplementedError

            def write_native_config(self, installation, config, job_directory):  # pragma: no cover
                raise NotImplementedError

            def build_launch(self, installation, native_config):  # pragma: no cover
                raise NotImplementedError

            def parse_log(self, line):  # pragma: no cover
                return ()

            def recovery_capabilities(self, installation):  # pragma: no cover
                raise NotImplementedError

            def collect_artifacts(self, job_directory):  # pragma: no cover
                return ()

        self.assertEqual((), Bundled().update_commands(Path(".")))


class DirtyTreeRuleTests(unittest.TestCase):
    """The rule the update relies on, against a real repository."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        git(self.repo, "init", "-q")
        (self.repo / "train.py").write_text("print('hi')\n", encoding="utf-8")
        git(self.repo, "add", "train.py")
        git(self.repo, "commit", "-qm", "init")

    def test_untracked_environment_does_not_count_as_dirty(self):
        (self.repo / ".venv" / "Scripts").mkdir(parents=True)
        (self.repo / ".venv" / "Scripts" / "python.exe").write_bytes(b"x")
        (self.repo / ".lp" / "python").mkdir(parents=True)
        (self.repo / ".lp" / "python" / "marker").write_text("x", encoding="utf-8")
        self.assertEqual("", dirty(self.repo))

    def test_modified_tracked_file_blocks_the_update(self):
        (self.repo / "train.py").write_text("print('patched')\n", encoding="utf-8")
        self.assertNotEqual("", dirty(self.repo))


if __name__ == "__main__":
    unittest.main()
