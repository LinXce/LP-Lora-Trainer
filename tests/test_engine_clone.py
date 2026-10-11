"""Git clone into engine/, against a real local repository (no network)."""
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.common import ServiceError
from app.services.engines import EngineService
from app.services.installation import InstallationService
from app.storage.state import StateStore
from tests.engine_fixtures import cleanup_directory


def git(*args, cwd=None):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


def make_remote(path: Path):
    """A tiny upstream that looks like an ai-toolkit checkout, with a branch and a tag."""
    path.mkdir(parents=True)
    for name in ("toolkit", "jobs", "manager"):
        (path / name).mkdir()
        (path / name / "keep.txt").write_text("x", encoding="utf-8")
    (path / "run.py").write_text("# upstream", encoding="utf-8")
    git("init", "-q", "-b", "main", cwd=path)
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "init", cwd=path)
    git("tag", "v1", cwd=path)
    git("checkout", "-q", "-b", "dev", cwd=path)
    (path / "dev.txt").write_text("dev", encoding="utf-8")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "dev", cwd=path)
    git("checkout", "-q", "main", cwd=path)
    return path


@unittest.skipUnless(InstallationService._git_executable(), "git is not installed")
class EngineCloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="LP clone 中文 ")
        self.root = Path(self.temp.name).resolve()
        self.engine_root = self.root / "engine"
        self.engine_root.mkdir()
        self.remote = make_remote(self.root / "remote" / "ai-toolkit")
        self.url = self.remote.as_uri()
        self.store = StateStore(self.root / "data")
        self.engines = EngineService(self.store, lambda: {"engine_root": str(self.engine_root)}, self.root)
        self.service = InstallationService(self.store, self.engines, self.root)

    def tearDown(self):
        self.wait_idle()
        cleanup_directory(self.temp.name)

    def wait_idle(self, timeout=60):
        deadline = time.monotonic() + timeout
        while self.service.workers and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.service.workers, {})

    def run_clone(self, *args, **kwargs):
        session = self.service.clone(*args, **kwargs)
        self.wait_idle()
        return self.service.get(session["session_id"])

    def log(self, session):
        return self.service.read_log(session["session_id"], 0)["text"]

    def test_clones_into_engine_root_and_registers_the_engine(self):
        done = self.run_clone(self.url, "my-toolkit")
        self.assertEqual(done["state"], "succeeded", done["error"])
        self.assertEqual(done["kind"], "clone")
        target = self.engine_root / "my-toolkit"
        self.assertTrue((target / "run.py").is_file())
        self.assertTrue((target / ".git").is_dir())
        self.assertEqual([p.name for p in self.engine_root.iterdir()], ["my-toolkit"])  # no staging leftovers
        self.assertEqual([e["label"] for e in self.engines.list()], ["my-toolkit"])
        self.assertEqual(self.engines.list()[0]["engine_id"], "ai_toolkit")
        self.assertIn("已拉取到", self.log(done))

    def test_default_folder_name_follows_the_repository(self):
        done = self.run_clone(self.url)
        self.assertEqual(done["state"], "succeeded", done["error"])
        self.assertTrue((self.engine_root / "ai-toolkit" / "run.py").is_file())
        self.assertEqual(InstallationService.default_clone_name("https://github.com/o/musubi-tuner.git"), "musubi-tuner")
        self.assertEqual(InstallationService.default_clone_name("git@github.com:o/Kohya_ss/"), "Kohya_ss")

    def test_branch_and_tag_are_checked_out(self):
        self.assertEqual(self.run_clone(self.url, "dev-copy", "dev")["state"], "succeeded")
        self.assertTrue((self.engine_root / "dev-copy" / "dev.txt").is_file())
        self.assertEqual(self.run_clone(self.url, "tag-copy", "v1")["state"], "succeeded")
        self.assertFalse((self.engine_root / "tag-copy" / "dev.txt").exists())

    def test_existing_folder_is_never_overwritten(self):
        existing = self.engine_root / "ai-toolkit"
        existing.mkdir()
        (existing / "mine.txt").write_text("local work", encoding="utf-8")
        with self.assertRaises(ServiceError) as caught:
            self.service.clone(self.url, "ai-toolkit")
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual((existing / "mine.txt").read_text(encoding="utf-8"), "local work")
        self.assertEqual(self.service.list(), [])

    def test_failed_clone_leaves_no_folder_and_reports_it(self):
        done = self.run_clone((self.root / "missing-repo").as_uri(), "broken")
        self.assertEqual(done["state"], "failed")
        self.assertIn("git clone 失败", done["error"])
        self.assertEqual(list(self.engine_root.iterdir()), [])
        self.assertEqual(self.engines.list(), [])

    def test_unknown_branch_fails_cleanly(self):
        done = self.run_clone(self.url, "nope", "no-such-branch")
        self.assertEqual(done["state"], "failed")
        self.assertEqual(list(self.engine_root.iterdir()), [])

    def test_rejects_unsafe_input_before_running_git(self):
        bad = [
            ("file-looking", "ftp://example.com/x.git", None, None),
            ("option injection url", "--upload-pack=calc", None, None),
            ("ext transport", "ext::sh -c calc", None, None),
            ("whitespace", "https://example.com/a b.git", None, None),
            ("traversal name", self.url, "../escape", None),
            ("absolute name", self.url, "C:/evil", None),
            ("reserved name", self.url, "con", None),
            ("dot name", self.url, "..", None),
            ("option-like ref", self.url, "ok", "--config=evil"),
            ("space in ref", self.url, "ok", "a b"),
        ]
        with patch("app.services.installation.subprocess.Popen") as popen:
            for label, url, name, ref in bad:
                with self.subTest(label), self.assertRaises(ServiceError):
                    self.service.clone(url, name, ref)
            popen.assert_not_called()
        self.assertEqual(list(self.engine_root.iterdir()), [])
        self.assertFalse((self.root / "escape").exists())

    def test_missing_git_is_a_clear_error(self):
        with patch.object(InstallationService, "_git_executable", return_value=None):
            with self.assertRaises(ServiceError) as caught:
                self.service.clone(self.url, "x")
        self.assertIn("git", caught.exception.message)

    def test_only_one_job_at_a_time(self):
        gate = threading.Event()
        real_popen = subprocess.Popen

        def slow(argv, **kwargs):
            gate.wait(10)
            return real_popen(argv, **kwargs)

        with patch("app.services.installation.subprocess.Popen", side_effect=slow):
            first = self.service.clone(self.url, "one")
            with self.assertRaises(ServiceError) as caught:
                self.service.clone(self.url, "two")
            self.assertEqual(caught.exception.status, 409)
            gate.set()
            self.wait_idle()
        self.assertEqual(self.service.get(first["session_id"])["state"], "succeeded")

    def test_cancel_stops_the_clone_and_removes_partial_files(self):
        started = threading.Event()
        real_popen = subprocess.Popen

        def hang(argv, **kwargs):
            # A git that never finishes: the real clone target stays a partial staging dir.
            kwargs.pop("stdout", None)
            process = real_popen(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **{
                k: v for k, v in kwargs.items() if k in ("cwd", "env", "stdin", "creationflags")})
            Path(argv[-1]).mkdir(parents=True)
            (Path(argv[-1]) / "partial.bin").write_bytes(b"x")
            started.set()
            time.sleep(0.3)
            return process

        with patch("app.services.installation.subprocess.Popen", side_effect=hang):
            session = self.service.clone(self.url, "cancelled")
            self.assertTrue(started.wait(10))
            self.service.cancel(session["session_id"])
            self.wait_idle()
        self.assertEqual(self.service.get(session["session_id"])["state"], "cancelled")
        self.assertEqual(list(self.engine_root.iterdir()), [])

    def test_credentials_in_the_url_never_reach_the_stored_session(self):
        secret_url = "https://user:s3cret-token@example.invalid/org/repo.git"
        with patch("app.services.installation.subprocess.Popen") as popen:
            popen.side_effect = OSError("no network in tests")
            session = self.service.clone(secret_url, "secret")
            self.wait_idle()
        stored = self.service.get(session["session_id"])
        dump = repr(stored) + self.log(stored)
        self.assertNotIn("s3cret-token", dump)
        self.assertEqual(stored["repository_url"], "https://example.invalid/org/repo.git")

    def test_restart_removes_a_leftover_staging_folder_and_marks_the_session(self):
        leftover = self.engine_root / ".lp-clone-abcdef123456"
        leftover.mkdir()
        (leftover / "half.bin").write_bytes(b"x")
        self.store.put("installation", "a" * 32, {"session_id": "a" * 32, "installation_id": "", "kind": "clone",
                                                  "state": "running"})
        InstallationService(self.store, self.engines, self.root)
        self.assertFalse(leftover.exists())
        self.assertEqual(self.store.get("installation", "a" * 32)["state"], "interrupted")


if __name__ == "__main__":
    unittest.main()
