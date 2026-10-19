"""Installation state machine, with no network, GPU or large dependencies."""
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.common import ServiceError
from app.services.engines import EngineService
from app.services.installation import InstallationService
from app.storage.state import StateStore
from tests.engine_fixtures import cleanup_directory


def toolkit(root, script=""):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("toolkit", "jobs", "manager"):
        (root / name).mkdir(exist_ok=True)
    (root / "run.py").write_text("# fixture", encoding="utf-8")
    (root / "manager" / "__main__.py").write_text(script, encoding="utf-8")
    return root


class FakeProcess:
    def __init__(self, code=0, output=b"fixture output\n"):
        self.stdout = io.BytesIO(output)
        self.returncode = code
        self.pid = 123

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


class ChunkPipe(io.BytesIO):
    def read1(self, size):
        return self.read(1)


class InstallationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="LP install 中文 ")
        self.root = Path(self.temp.name).resolve()
        self.source = toolkit(self.root / "engine" / "ai-toolkit")
        self.store = StateStore(self.root / "data")
        self.engines = EngineService(self.store, lambda: {"engine_root": str(self.root / "engine")}, self.root)
        self.record = self.engines.rescan()[0]
        self.key = self.record["installation_id"]
        self.service = InstallationService(self.store, self.engines, self.root)
        runtime = self.root / "python_runtime"
        runtime.mkdir()
        (runtime / ("python.exe" if os.name == "nt" else "python")).touch()
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("bootstrap_uv.py", "ensure_engine_venv.py"):
            (scripts / name).touch()
        uv = self.root / "runtime" / "uv"
        uv.mkdir(parents=True)
        (uv / ("uv.exe" if os.name == "nt" else "uv")).touch()

    def tearDown(self):
        self.assertEqual(self.service.workers, {})
        self.assertEqual(self.engines.installing, set())
        cleanup_directory(self.temp.name)

    def start(self, **options):
        with patch("app.services.installation.threading.Thread.start"):
            return self.service.start(self.key, **options)

    def python(self):
        target = self.source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        target.parent.mkdir(parents=True)
        target.touch()
        return target.resolve()

    def run_session(self, session, process=None, diagnose=None, state=None):
        state = state or self.service.workers[session["session_id"]]
        def ready(key, **kwargs):
            self.assertEqual(self.engines.get(key)["python_executable"], str(self.source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
            self.store.patch("engine", key, {"state": "ready"})
        with patch("app.services.installation.subprocess.Popen", return_value=process or FakeProcess()), patch.object(self.engines, "diagnose", side_effect=diagnose or ready):
            self.service._run(session, SimpleNamespace(title="Fixture official installer"), {}, state)
        return self.service.get(session["session_id"])

    def test_success_saves_python_before_diagnosis(self):
        python = self.python()
        session = self.start()
        final = self.run_session(session)
        self.assertEqual(final["state"], "succeeded")
        self.assertEqual(self.engines.get(self.key)["state"], "ready")
        self.assertEqual(self.engines.get(self.key)["python_executable"], str(python))
        self.assertIn("installation complete", self.service.read_log(session["session_id"], 0)["text"])

    def test_nonzero_installer_saves_partial_environment(self):
        python = self.python()
        final = self.run_session(self.start(), FakeProcess(7))
        self.assertEqual(final["state"], "failed")
        self.assertIn("exit code 7", final["error"])
        self.assertEqual(self.engines.get(self.key)["python_executable"], str(python))

    def test_installer_success_without_python_is_failure(self):
        final = self.run_session(self.start())
        self.assertEqual(final["state"], "failed")
        self.assertIn("no engine Python", final["error"])

    def test_diagnostic_failure_keeps_python(self):
        python = self.python()
        final = self.run_session(self.start(), diagnose=ServiceError("CUDA unavailable"))
        self.assertEqual(final["state"], "failed")
        self.assertEqual(final["diagnostic_error"], "CUDA unavailable")
        self.assertEqual(self.engines.get(self.key)["python_executable"], str(python))
        self.assertEqual(self.engines.get(self.key)["state"], "failed")

    def test_process_start_failure_cleans_up(self):
        session = self.start()
        with patch("app.services.installation.subprocess.Popen", side_effect=OSError("cannot launch")):
            self.service._run(session, SimpleNamespace(title="Fixture"), {}, self.service.workers[session["session_id"]])
        self.assertIn("Could not start installer step 1", self.service.get(session["session_id"])["error"])

    def test_bootstrap_failure_prevents_official_installer(self):
        (self.root / "runtime" / "uv" / ("uv.exe" if os.name == "nt" else "uv")).unlink()
        session = self.start()
        self.assertIn("bootstrap_uv.py", session["commands"][0][2])
        with patch("app.services.installation.subprocess.Popen", return_value=FakeProcess(9)) as launch:
            self.service._run(session, SimpleNamespace(title="Fixture"), {}, self.service.workers[session["session_id"]])
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(self.service.get(session["session_id"])["state"], "failed")

    def test_thread_start_failure_rolls_back_registration(self):
        with patch("app.services.installation.threading.Thread.start", side_effect=RuntimeError("thread failure")):
            with self.assertRaises(ServiceError):
                self.service.start(self.key)
        self.assertEqual(self.service.list()[0]["state"], "failed")
        self.assertEqual(self.engines.get(self.key)["state"], "failed")

    def test_final_state_write_failure_still_releases_locks(self):
        self.python()
        session = self.start()
        original = self.store.patch
        def fail_final(kind, key, values, **kwargs):
            if kind == "installation" and "finished_at" in values:
                raise OSError("database unavailable")
            return original(kind, key, values, **kwargs)
        with patch.object(self.store, "patch", side_effect=fail_final):
            self.run_session(session)
        self.assertIn("Failed to persist final", self.service.read_log(session["session_id"], 0)["text"])

    def test_cancel_queued_session_prevents_process_start(self):
        session = self.start()
        self.service.cancel(session["session_id"])
        with patch("app.services.installation.subprocess.Popen") as launch:
            self.service._run(session, SimpleNamespace(title="Fixture"), {}, self.service.workers[session["session_id"]])
        launch.assert_not_called()
        self.assertEqual(self.service.get(session["session_id"])["state"], "cancelled")

    def test_cancel_running_unblocks_output_and_releases_locks(self):
        session = self.start()
        state = self.service.workers[session["session_id"]]
        entered, released = threading.Event(), threading.Event()
        class BlockingPipe(io.BytesIO):
            def read1(self, size):
                entered.set()
                if not released.wait(5):
                    raise TimeoutError("cancel did not terminate installer")
                return b""
        process = FakeProcess()
        process.stdout = BlockingPipe()
        process.returncode = None
        process.wait = lambda: 1
        with patch("app.services.installation.subprocess.Popen", return_value=process), patch.object(self.service, "_terminate", side_effect=lambda p: released.set()):
            thread = threading.Thread(target=self.service._run, args=(session, SimpleNamespace(title="Fixture"), {}, state))
            state["thread"] = thread
            thread.start()
            self.assertTrue(entered.wait(5))
            self.service.cancel(session["session_id"])
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(self.service.get(session["session_id"])["state"], "cancelled")

    def test_second_install_and_active_training_are_blocked(self):
        session = self.start()
        try:
            with self.assertRaises(ServiceError):
                self.service.start(self.key)
        finally:
            self.service.cancel(session["session_id"])
            self.run_session(session)
        self.store.put("task", "active", {"state": "running"})
        with self.assertRaises(ServiceError):
            self.service.start(self.key)

    def test_worker_preserves_utf8_across_single_byte_reads(self):
        self.python()
        process = FakeProcess()
        process.stdout = ChunkPipe("中文日志\n".encode())
        session = self.start()
        self.run_session(session, process)
        text = self.service.read_log(session["session_id"], 0)["text"]
        self.assertIn("中文日志", text)
        self.assertNotIn("�", text)

    def test_log_offset_keeps_partial_multibyte_character(self):
        self.python()
        session = self.start()
        self.run_session(session)
        path = self.service.log_path(session["session_id"])
        data = b"x" * (128 * 1024 - 1) + "中文".encode()
        path.write_bytes(data)
        first = self.service.read_log(session["session_id"], 0)
        self.assertEqual(first["offset"], 128 * 1024 - 1)
        second = self.service.read_log(session["session_id"], first["offset"])
        self.assertEqual(second["text"], "中文")
        self.assertEqual(second["offset"], len(data))

    def test_restart_marks_all_incomplete_sessions_interrupted(self):
        for index, status in enumerate(("queued", "running", "stopping", "verifying")):
            sid = f"{index:032x}"
            self.store.put("installation", sid, {"session_id": sid, "installation_id": self.key, "state": status})
        self.store.patch("engine", self.key, {"state": "preparing"})
        service = InstallationService(self.store, self.engines, self.root)
        self.assertTrue(all(s["state"] == "interrupted" for s in service.list()))
        self.assertEqual(self.engines.get(self.key)["state"], "failed")

    def test_missing_runtime_or_manager_rejected_before_registration(self):
        python = self.root / "python_runtime" / ("python.exe" if os.name == "nt" else "python")
        python.unlink()
        with self.assertRaises(ServiceError):
            self.service.start(self.key)
        python.touch()
        (self.source / "manager" / "__main__.py").unlink()
        with self.assertRaises(ServiceError):
            self.service.start(self.key)
        self.assertEqual(self.service.list(), [])

    def test_environment_variables_do_not_redirect_installation(self):
        from adapters.registry import ADAPTERS
        pollution = {"PYTHONHOME": "wrong", "PYTHONPATH": "wrong", "UV_PYTHON": "wrong", "UV_PROJECT_ENVIRONMENT": "wrong", "VIRTUAL_ENV": "wrong", "UV_NO_SYNC": "1"}
        with patch.dict(os.environ, pollution):
            _, _, env, _ = self.service._install_environment(self.source, ADAPTERS["ai_toolkit"], self.service._runtime_python())
        for key in pollution:
            self.assertNotIn(key, env)
        self.assertEqual(env["UV_PYTHON_PREFERENCE"], "only-managed")
        self.assertTrue(env["UV_PYTHON_INSTALL_DIR"].startswith(str(self.root)))


    def test_selected_mirror_reaches_every_environment_strategy(self):
        from adapters.registry import ADAPTERS
        mirror = 'https://pypi.tuna.tsinghua.edu.cn/simple'
        pollution = {
            'PIP_INDEX_URL': 'https://host.invalid/simple',
            'PIP_EXTRA_INDEX_URL': 'https://host.invalid/extra',
            'PIP_CONFIG_FILE': 'host-pip.ini',
            'UV_INDEX_URL': 'https://host.invalid/simple',
            'UV_DEFAULT_INDEX': 'https://host.invalid/simple',
            'UV_EXTRA_INDEX_URL': 'https://host.invalid/extra',
            'UV_CONFIG_FILE': 'host-uv.toml',
        }
        for adapter in ADAPTERS.values():
            with self.subTest(engine=adapter.engine_id), patch.dict(os.environ, pollution):
                _, _, env, _ = self.service._install_environment(
                    self.source, adapter, self.service._runtime_python(), mirror
                )
                for name in ('PIP_INDEX_URL', 'UV_INDEX_URL', 'UV_DEFAULT_INDEX'):
                    self.assertEqual(env[name], mirror)
                for name in ('PIP_EXTRA_INDEX_URL', 'UV_EXTRA_INDEX_URL', 'UV_CONFIG_FILE'):
                    self.assertNotIn(name, env)
                self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)
                self.assertEqual(os.environ['PIP_INDEX_URL'], pollution['PIP_INDEX_URL'])

    def test_official_selection_does_not_inherit_host_mirror(self):
        from adapters.registry import ADAPTERS
        names = ('PIP_INDEX_URL', 'UV_INDEX_URL', 'UV_DEFAULT_INDEX')
        with patch.dict(os.environ, {name: 'https://host.invalid/simple' for name in names}):
            _, _, env, _ = self.service._install_environment(
                self.source, ADAPTERS['ai_toolkit'], self.service._runtime_python()
            )
        for name in names:
            self.assertNotIn(name, env)

    def test_mirror_saved_in_session_passed_to_worker_and_logged(self):
        mirror = 'https://mirrors.aliyun.com/pypi/simple'
        self.python()
        session = self.start(mirror_url=' ' + mirror + '/ ')
        self.assertEqual(session['mirror_url'], mirror)
        self.assertEqual(self.service.get(session['session_id'])['mirror_url'], mirror)
        worker = self.service.workers[session['session_id']]['thread']
        for name in ('PIP_INDEX_URL', 'UV_INDEX_URL', 'UV_DEFAULT_INDEX'):
            self.assertEqual(worker._args[2][name], mirror)
        self.assertEqual(self.run_session(session)['state'], 'succeeded')
        self.assertIn('[LP] Package index: ' + mirror, self.service.read_log(session['session_id'], 0)['text'])

    def test_invalid_mirror_does_not_register_session_or_lock_engine(self):
        before = self.engines.get(self.key)
        for mirror in ('ftp://mirror/simple', 'https://[broken', 'https://mirror:bad/simple'):
            with self.subTest(mirror=mirror), self.assertRaises(ServiceError):
                self.start(mirror_url=mirror)
        self.assertEqual(self.service.list(), [])
        self.assertEqual(self.service.workers, {})
        self.assertEqual(self.engines.installing, set())
        self.assertEqual(self.engines.get(self.key), before)

    def test_python_download_mirror_is_forwarded_and_logged(self):
        mirror = 'https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone'
        self.python()
        session = self.start(python_mirror_url=' ' + mirror + '/ ')
        self.assertEqual(session['python_mirror_url'], mirror)
        worker = self.service.workers[session['session_id']]['thread']
        self.assertEqual(worker._args[2]['UV_PYTHON_INSTALL_MIRROR'], mirror)
        self.run_session(session)
        self.assertIn(
            '[LP] Python download source: ' + mirror,
            self.service.read_log(session['session_id'], 0)['text'],
        )

    def test_python_download_mirror_is_scrubbed_without_a_choice(self):
        from adapters.registry import ADAPTERS
        with patch.dict(os.environ, {'UV_PYTHON_INSTALL_MIRROR': 'https://host.invalid/mirror'}):
            _, _, env, _ = self.service._install_environment(
                self.source, ADAPTERS['ai_toolkit'], self.service._runtime_python()
            )
        self.assertNotIn('UV_PYTHON_INSTALL_MIRROR', env)

    FAILURE_LOG = (
        "[LP] Python download source: "
        "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone\n"
        "\n> venv\\Scripts\\python.exe -u -m pip install -q setuptools<82\n"
        "\n> venv\\Scripts\\python.exe -u -c runpy setup\\setup_windows.py --headless\n"
        "Traceback (most recent call last):\n"
        "  File \"setup_common.py\", line 8, in <module>\n"
        "ModuleNotFoundError: No module named 'pkg_resources'\n"
    )

    def test_clone_rejects_a_malformed_commit_before_starting_git(self):
        for commit in ('not-a-sha', 'abc', '45088f04af78e11cec5407ff4652ea3ed2c14422ff'):
            with self.subTest(commit=commit), self.assertRaises(ServiceError):
                self.service.clone('https://example.invalid/org/repo.git', commit=commit)

    def test_the_engine_keeps_its_own_uv_and_uses_a_shared_interpreter(self):
        from adapters.registry import ADAPTERS
        _, _, env, _ = self.service._install_environment(
            self.source, ADAPTERS['ai_toolkit'], self.service._runtime_python()
        )
        # Managed interpreters are shared by Python version now: engines re-downloading
        # the same CPython was waste. The engine still carries its own uv binary.
        interpreter = Path(env['UV_PYTHON_INSTALL_DIR'])
        self.assertTrue(interpreter.is_relative_to(self.root / 'runtime' / 'uv' / 'python'))
        name = 'uv.exe' if os.name == 'nt' else 'uv'
        copy = self.source / '.lp' / name
        self.assertTrue(copy.is_file())
        self.assertEqual(copy.stat().st_size, (self.root / 'runtime' / 'uv' / name).stat().st_size)

    def test_proxy_switch_falls_back_to_the_saved_setting(self):
        on = {'proxy_url': 'http://127.0.0.1:7890', 'proxy_enabled': True}
        with patch.object(self.service.engines, 'settings', return_value=on):
            self.assertEqual(self.service._proxy_for(None), 'http://127.0.0.1:7890')
            self.assertEqual(self.service._proxy_for(True), 'http://127.0.0.1:7890')
            self.assertIsNone(self.service._proxy_for(False))
        off = {'proxy_url': 'http://127.0.0.1:7890', 'proxy_enabled': False}
        with patch.object(self.service.engines, 'settings', return_value=off):
            self.assertIsNone(self.service._proxy_for(None))
            self.assertEqual(self.service._proxy_for(True), 'http://127.0.0.1:7890')

    def test_enabling_the_proxy_without_an_address_is_rejected(self):
        with patch.object(self.service.engines, 'settings', return_value={'proxy_url': None, 'proxy_enabled': False}):
            with self.assertRaises(ServiceError):
                self.service._proxy_for(True)

    def test_proxy_reaches_the_installer_environment_only_when_enabled(self):
        from adapters.registry import ADAPTERS
        adapter = ADAPTERS['ai_toolkit']
        proxy = 'http://user:pw@127.0.0.1:7890'
        _, _, env, _ = self.service._install_environment(
            self.source, adapter, self.service._runtime_python(), None, None, None, proxy
        )
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            self.assertEqual(env[name], proxy)
        # Disabled: whatever the host exports is inherited, never replaced.
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://ambient:1'}):
            _, _, env, _ = self.service._install_environment(
                self.source, adapter, self.service._runtime_python()
            )
        self.assertEqual(env['HTTP_PROXY'], 'http://ambient:1')

    def test_source_repair_is_absent_when_the_source_is_complete(self):
        record = next(r for r in self.engines.rescan() if r["label"] == "ai-toolkit")
        self.assertIsNone(record.get("source_repair"))
        with self.assertRaises(ServiceError):
            self.service.repair_source(record["installation_id"])
        self.assertEqual(self.service.list(), [])

    def test_pip_progress_is_forced_on_for_piped_output(self):
        # pip suppresses its bar on a pipe; a silently downloading engine step is
        # indistinguishable from a stuck one, so LP forces it back on.
        from adapters.registry import ADAPTERS
        _, _, env, _ = self.service._install_environment(
            self.source, ADAPTERS['ai_toolkit'], self.service._runtime_python()
        )
        self.assertEqual(env['PIP_PROGRESS_BAR'], 'on')
        # Building a local engine package with an isolated build env can hang;
        # the venv's own setuptools/wheel build it in about a second.
        self.assertEqual(env['PIP_NO_BUILD_ISOLATION'], '0')

    @unittest.skipUnless(os.name == "nt", "Kohya installation plan is Windows-only")
    def test_kohya_plan_owns_the_install_without_lp_extra_steps(self):
        from adapters.registry import ADAPTERS
        kohya = self.root / "engine" / "kohya_plan"
        (kohya / "setup").mkdir(parents=True)
        (kohya / "setup" / "setup_windows.py").write_text("# fixture", encoding="utf-8")
        (kohya / "sd-scripts").mkdir()
        (kohya / "requirements.txt").write_text("numpy==1.26.4\n-e ./sd-scripts\n", encoding="utf-8")
        (kohya / "requirements_pytorch_windows.txt").write_text("torch==2.7.0+cu128\n", encoding="utf-8")
        mirror = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"
        commands, _, _, plan = self.service._build_install(
            kohya, ADAPTERS["kohya"], self.service._runtime_python(), "existing", None, mirror, None,
        )
        # LP adds nothing of its own any more: preparation plus the adapter plan.
        self.assertEqual(commands[-len(plan.commands):], tuple(plan.commands))
        install = commands[-1]
        self.assertIn("numpy==1.26.4", install)
        self.assertIn("torch==2.7.0+cu128", install)
        self.assertIn(f"{mirror}/torch/", install)
        self.assertIn(f"{mirror}/torchvision/", install)
        self.assertIn("-e", install)
        self.assertIn(str(kohya / "sd-scripts"), install)
        self.assertIn("--no-build-isolation", install)
        self.assertNotIn("-r", install)
        # The engine's own installer is no longer invoked at all.
        self.assertFalse(any("setup_windows.py" in str(argument) for argument in commands))

    def test_failure_output_ignores_the_header_and_earlier_steps(self):
        output = InstallationService._failure_output(self.FAILURE_LOG)
        self.assertNotIn("python-build-standalone", output)
        self.assertNotIn("pip install", output)
        self.assertIn("pkg_resources", output)

    def test_a_mirror_named_in_a_header_does_not_trigger_the_fallback(self):
        # Regression: the selected Python mirror URL itself contains
        # "python-build-standalone", which used to look like a mirror failure and
        # retried an unrelated engine error on the official source.
        session = {
            "mirror_url": "https://mirrors.aliyun.com/pypi/simple",
            "torch_mirror_url": None,
            "python_mirror_url": "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone",
        }
        path = self.root / "failure.log"
        path.write_text(self.FAILURE_LOG, encoding="utf-8")
        self.assertIsNone(self.service._official_fallback_reason(session, False, path))

    def test_a_real_download_failure_still_triggers_the_fallback(self):
        session = {"mirror_url": "https://mirrors.aliyun.com/pypi/simple", "torch_mirror_url": None, "python_mirror_url": None}
        path = self.root / "download.log"
        path.write_text(
            "[LP] Package index: https://mirrors.aliyun.com/pypi/simple\n"
            "\n> uv.exe sync --extra cu130\n"
            "error: No matching distribution found for torch==2.14.1+cu130\n",
            encoding="utf-8",
        )
        self.assertEqual(self.service._official_fallback_reason(session, False, path), "依赖源缺少所需包或版本")
        self.assertIsNone(self.service._official_fallback_reason(session, True, path))

    def test_refresh_repairs_legacy_toolkit_notices_without_hiding_errors(self):
        from adapters.registry import ADAPTERS
        damaged = ['AI Toolkit \ufffd old notice', 'AI Toolkit ?????', '?????????????']
        errors = ['Official installer failed (step 1, exit code 2)',
                  'AI Toolkit Could not reach remote (offline?)', '真实安装错误']
        self.store.patch('engine', self.key, {'issues': damaged + errors})
        # ``rescan`` is the repair path; ``list`` must stay read-only.
        for record in (self.engines.rescan()[0], self.engines.list()[0]):
            issues = record['issues']
            for notice in damaged:
                self.assertNotIn(notice, issues)
            for error in errors:
                self.assertIn(error, issues)
            self.assertEqual(issues.count(ADAPTERS['ai_toolkit'].training_notice), 1)
            self.assertEqual(record['engine_id'], 'ai_toolkit')
        persisted = self.store.get('engine', self.key)['issues']
        self.assertEqual(persisted, self.engines.list()[0]['issues'])
        self.assertEqual((self.source / 'manager' / '__main__.py').read_text(encoding='utf-8'), '')

    def test_legacy_notice_cleanup_does_not_touch_other_trainers(self):
        notices = ['AI Toolkit ?????', 'Official installer failed (step 1, exit code 2)']
        issues = self.engines._type_issues(self.source, 'musubi_tuner', ['musubi_tuner'], existing=notices)
        for notice in notices:
            self.assertIn(notice, issues)


if __name__ == "__main__":
    unittest.main()
