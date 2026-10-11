"""Lifecycle/export regression tests; no real GUI, port or user process is touched."""
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from desktop import launcher
from desktop.bridge import DesktopBridge
from desktop.window import open_window
from test_window_chrome import FakeWindow


class DesktopLifetimeTests(unittest.TestCase):
    def test_normal_close_requests_shutdown_and_waits_owned_child(self):
        lifecycle = launcher.DesktopBackend('http://test.invalid')
        process = Mock()
        lifecycle.prepare(lambda: process)
        with patch.object(launcher, 'request_shutdown', return_value='stopped') as shutdown, \
                patch.object(launcher, 'kill_tree') as kill:
            self.assertTrue(lifecycle.closing(Mock()))
            lifecycle.cleanup()
        shutdown.assert_called_once_with('http://test.invalid')
        process.wait.assert_called_once_with(timeout=15)
        kill.assert_not_called()

    def test_busy_close_is_vetoed_regardless_of_dialog_answer(self):
        for answer in (False, True):
            lifecycle = launcher.DesktopBackend('http://test.invalid')
            process, window = Mock(), Mock()
            window.create_confirmation_dialog.return_value = answer
            lifecycle.prepare(lambda: process)
            with patch.object(launcher, 'request_shutdown', return_value='busy'), \
                    patch.object(launcher, 'kill_tree') as kill:
                self.assertFalse(lifecycle.closing(window))
                lifecycle.cleanup()  # surface failure transfers ownership to console
            process.wait.assert_not_called()
            kill.assert_not_called()

    def test_reused_backend_is_never_force_killed(self):
        lifecycle = launcher.DesktopBackend('http://test.invalid')
        lifecycle.prepare(lambda: None)
        with patch.object(launcher, 'request_shutdown') as shutdown, patch.object(launcher, 'kill_tree') as kill:
            lifecycle.cleanup()
        shutdown.assert_not_called()
        kill.assert_not_called()

    def test_hung_owned_backend_is_killed_after_grace_period(self):
        lifecycle = launcher.DesktopBackend('http://test.invalid')
        lifecycle.process = Mock(pid=123, wait=Mock(side_effect=subprocess.TimeoutExpired('test', 15)))
        with patch.object(launcher, 'request_shutdown', return_value='stopped'), patch.object(launcher, 'kill_tree') as kill:
            lifecycle.cleanup()
        kill.assert_called_once_with(123)

    def test_startup_timeout_kills_only_new_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=321)
            process.poll.return_value = None
            with patch.object(launcher, 'backend_ready', return_value=False), \
                    patch.object(launcher.subprocess, 'Popen', return_value=process), \
                    patch.object(launcher.time, 'monotonic', side_effect=[0, 31]), \
                    patch.object(launcher, 'kill_tree') as kill:
                with self.assertRaises(RuntimeError):
                    launcher.ensure_backend('http://test.invalid', None, 1, Path(tmp), Path(tmp))
            kill.assert_called_once_with(321)

    def test_window_failure_falls_back_without_error_dialog(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'open_window', side_effect=RuntimeError('unavailable')), \
                patch.object(launcher, 'spawn_console') as fallback, \
                patch.object(launcher, '_startup_error') as report:
            lock.return_value.acquire.return_value = True
            launcher.run()
        fallback.assert_called_once()
        report.assert_not_called()

    def test_supervisor_child_is_waited_before_lock_probe(self):
        from app.api.server import wait_supervisor_exit
        order = []
        process = Mock()
        process.wait.side_effect = lambda **_: order.append('wait')
        with patch('supervisor.worker.ProcessLock') as lock:
            lock.return_value.acquire.side_effect = lambda: order.append('probe') or True
            wait_supervisor_exit(Path('unused'), processes=(process,))
        self.assertEqual(order, ['wait', 'probe'])
        lock.return_value.close.assert_called_once()


class SplashRaceTests(unittest.TestCase):
    def test_close_during_bootstrap_waits_and_skips_navigation(self):
        window, started, finish = FakeWindow(), threading.Event(), threading.Event()
        def prepare():
            started.set()
            if not finish.wait(2):
                raise AssertionError('test bootstrap blocked')
        def start(**_):
            window.events.shown.fire()
            self.assertTrue(started.wait(2))
            window.events.closed.fire()
            finish.set()
        fake = SimpleNamespace(create_window=Mock(return_value=window), start=start)
        with patch.dict(sys.modules, {'webview': fake}):
            open_window('http://test.invalid', prepare=prepare)
        window.load_url.assert_not_called()

    def test_delayed_shown_cannot_start_backend_after_ui_loop_exits(self):
        window, prepare = FakeWindow(), Mock()
        fake = SimpleNamespace(create_window=Mock(return_value=window), start=Mock())
        with patch.dict(sys.modules, {'webview': fake}):
            open_window('http://test.invalid', prepare=prepare)
            window.events.shown.fire()
        prepare.assert_not_called()

    def test_navigation_failure_closes_window_and_reraises(self):
        window = FakeWindow()
        window.load_url.side_effect = RuntimeError('navigation failed')
        def start(**_):
            window.events.shown.fire()
            # The bootstrap runs on its own thread; emulate the UI loop staying
            # alive until navigation failure closes the window.
            for _ in range(200):
                if window.destroy.called:
                    return
                threading.Event().wait(0.005)
            raise AssertionError('navigation failure did not close window')
        fake = SimpleNamespace(create_window=Mock(return_value=window), start=start)
        with patch.dict(sys.modules, {'webview': fake}), self.assertRaisesRegex(RuntimeError, 'navigation failed'):
            open_window('http://test.invalid', prepare=Mock())
        window.destroy.assert_called_once()


class NativeExportTests(unittest.TestCase):
    def test_utf8_save_uses_user_dialog_and_appends_missing_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chosen'
            window = FakeWindow()
            window.create_file_dialog = Mock(return_value=(str(path),))
            bridge = DesktopBridge()
            bridge._attach(window)
            fake = SimpleNamespace(FileDialog=SimpleNamespace(SAVE='save'))
            text = '\u8bad\u7ec3\u53c2\u6570\n{"rank": 16}'
            with patch.dict(sys.modules, {'webview': fake}):
                result = bridge.save_text_file('training.json', text)
            self.assertEqual(result, str(path.with_suffix('.json')))
            self.assertEqual(Path(result).read_text(encoding='utf-8'), text)
            window.create_file_dialog.assert_called_once_with('save', save_filename='training.json',
                file_types=('Configuration (*.json)', 'All files (*.*)'))

    def test_cancel_does_not_write(self):
        window = FakeWindow()
        window.create_file_dialog = Mock(return_value=None)
        bridge = DesktopBridge()
        bridge._attach(window)
        with patch.dict(sys.modules, {'webview': SimpleNamespace(FileDialog=SimpleNamespace(SAVE='save'))}), \
                patch.object(Path, 'write_text') as write:
            self.assertIsNone(bridge.save_text_file('test.toml', 'rank=16'))
        write.assert_not_called()

    def test_bad_filename_format_and_size_rejected_before_dialog(self):
        bridge = DesktopBridge()
        for name, text in [('../escape.json', '{}'), ('run.exe', ''), ('a\x00.json', ''),
                           ('test.json', 'x' * (4 * 1024 * 1024 + 1)), ('test.json', 12)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                bridge.save_text_file(name, text)


    def test_permission_failure_explains_restricted_process_without_claiming_success(self):
        window = FakeWindow()
        window.create_file_dialog = Mock(return_value=(str(Path(tempfile.gettempdir()) / 'blocked.json'),))
        bridge = DesktopBridge()
        bridge._attach(window)
        fake = SimpleNamespace(FileDialog=SimpleNamespace(SAVE='save'))
        with patch.dict(sys.modules, {'webview': fake}), \
                patch.object(Path, 'write_text', side_effect=PermissionError('blocked')), \
                patch('desktop.permissions.process_integrity_level', return_value=0x1000):
            with self.assertRaisesRegex(PermissionError, '受限权限'):
                bridge.save_text_file('test.json', '{}')

    def test_permission_failure_for_normal_process_identifies_destination(self):
        window = FakeWindow()
        window.create_file_dialog = Mock(return_value=(str(Path(tempfile.gettempdir()) / 'blocked.json'),))
        bridge = DesktopBridge()
        bridge._attach(window)
        fake = SimpleNamespace(FileDialog=SimpleNamespace(SAVE='save'))
        with patch.dict(sys.modules, {'webview': fake}), \
                patch.object(Path, 'write_text', side_effect=PermissionError('blocked')), \
                patch('desktop.permissions.process_integrity_level', return_value=0x2000):
            with self.assertRaisesRegex(PermissionError, 'blocked.json.*重新选择'):
                bridge.save_text_file('test.json', '{}')


class AdapterPathTests(unittest.TestCase):
    def test_all_declared_local_paths_have_browse_metadata(self):
        from adapters.ai_toolkit import params as ai
        from adapters.kohya import params as kohya
        from adapters.musubi_tuner import params as musubi
        for module in (ai, kohya, musubi):
            specs = {p.key: p.spec() for p in module.PARAMS}
            for key, kind in module._PATH_KINDS.items():
                with self.subTest(module=module.__name__, key=key):
                    self.assertEqual(specs[key]['path_kind'], kind)
            for param in module.PARAMS:
                if param.key in ('network_module', 'optimizer_type', 'huggingface_path_in_repo', 'huggingface_token'):
                    self.assertNotIn('path_kind', param.spec())

    def test_path_metadata_remains_optional_and_positional_compatible(self):
        from adapters.capabilities import Param
        self.assertNotIn('path_kind', Param('key', 'label', 'text').spec())
        self.assertEqual(Param('key', 'label', 'text', path_kind='file', file_types=('*.txt',)).spec()['file_types'], ['*.txt'])
