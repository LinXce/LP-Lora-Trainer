"""No-GUI desktop startup and portable EXE contract tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from desktop import console, launcher, window


class DesktopLauncherTests(unittest.TestCase):
    def test_matching_backend_protocol_is_reusable(self):
        with patch.object(launcher.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value.__enter__.return_value.status = 204
            with patch.object(launcher.json, 'load', return_value={
                'backend_version': '0.1.0',
                'installation_workflow_version': launcher.INSTALLATION_WORKFLOW_VERSION,
            }):
                self.assertTrue(launcher.backend_ready('http://127.0.0.1:8765'))

    def test_old_or_incompatible_backend_is_not_silently_reused(self):
        for status in [
            {'backend_version': '0.1.0'},
            {'backend_version': '0.1.0', 'installation_workflow_version': 999},
            {'backend_version': 'old', 'installation_workflow_version': launcher.INSTALLATION_WORKFLOW_VERSION},
        ]:
            with self.subTest(status=status), patch.object(launcher.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value.__enter__.return_value.status = 204
                with patch.object(launcher.json, 'load', return_value=status):
                    with self.assertRaisesRegex(RuntimeError, '旧版或不兼容'):
                        launcher.backend_ready('http://127.0.0.1:8765')

    def test_incompatible_backend_never_spawns_another_backend(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'backend_ready', side_effect=RuntimeError('旧版或不兼容')), \
                patch.object(launcher.subprocess, 'Popen') as spawn, \
                patch.object(launcher, 'open_window') as window_mock:
            lock.return_value.acquire.return_value = True
            with self.assertRaisesRegex(RuntimeError, '旧版或不兼容'):
                launcher.main()
            spawn.assert_not_called()
            window_mock.assert_not_called()
            lock.return_value.close.assert_called_once()

    def test_working_desktop_window_never_shows_a_console(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'ensure_backend') as backend, \
                patch.object(launcher, 'spawn_console') as terminal, \
                patch.object(launcher, 'open_window') as window_mock:
            lock.return_value.acquire.return_value = True
            launcher.main()
            window_mock.assert_called_once()
            terminal.assert_not_called()
            backend.assert_called_once()
            lock.return_value.close.assert_called_once()

    def test_broken_bridge_falls_back_to_terminal_with_reason(self):
        reason = 'RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize'
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=reason), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'ensure_backend') as backend, \
                patch.object(launcher, 'spawn_console') as terminal, \
                patch.object(launcher, 'open_window') as window_mock:
            launcher.main()
            # The console owns the lock and the backend, so nothing is orphaned here.
            lock.assert_not_called()
            backend.assert_not_called()
            window_mock.assert_not_called()
            terminal.assert_called_once()
            self.assertEqual(terminal.call_args.args[2], reason)

    def test_browser_flag_forces_the_terminal_path(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher', '--browser']), \
                patch.object(launcher, 'desktop_bridge_error') as probe, \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'spawn_console') as terminal:
            launcher.main()
            probe.assert_not_called()
            lock.assert_not_called()
            self.assertIn('--browser', terminal.call_args.args[2])

    def test_desktop_bridge_error_reports_the_failure_verbatim(self):
        def broken():
            raise RuntimeError('Failed to resolve Python.Runtime.Loader.Initialize')

        if os.name == 'nt':
            self.assertIn('Python.Runtime.Loader', window.desktop_bridge_error(broken))
        self.assertIsNone(window.desktop_bridge_error(lambda: None))

    def test_console_stops_the_backend_it_owns(self):
        class FakeProcess:
            pid = 4242
            def __init__(self): self.waited = False
            def wait(self, timeout=None): self.waited = True; return 0

        for outcome, owned, answer, expect_kill in (
            ('stopped', True, '', False),
            ('busy', True, 'y', True),
            ('busy', True, 'n', False),
            ('busy', True, None, False),      # no console input must never mean "yes"
            ('busy', False, 'y', False),      # never kill a backend owned elsewhere
            ('unreachable', True, '', True),
            ('unreachable', False, '', False),
        ):
            with self.subTest(outcome=outcome, owned=owned, answer=answer):
                process = FakeProcess() if owned else None
                with patch.object(console, 'request_shutdown', return_value=outcome), \
                        patch.object(console, 'kill_tree') as killer:
                    console.finish(process, 'http://127.0.0.1:8765', note=lambda *_: None,
                                   ask=lambda *_: answer)
                self.assertEqual(killer.called, expect_kill)
                if owned and outcome == 'stopped':
                    self.assertTrue(process.waited)

    @unittest.skipUnless(os.name == 'nt', 'Windows terminal fallback')
    def test_fallback_uses_the_system_terminal_and_shows_the_logo(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'assets').mkdir()
            (workspace / 'assets' / 'logo.txt').write_text('LP ASCII LOGO', encoding='utf-8')
            (workspace / 'runtime').mkdir()
            with patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher.time, 'sleep'):
                spawn.return_value.poll.return_value = None
                launcher.spawn_console(5900, workspace / 'data', 'reason: C:\\x "quoted"', workspace)
            argv = spawn.call_args.args[0]
            # The window is the machine's own terminal, not a Python-owned console.
            self.assertEqual(Path(argv[0]).name, 'cmd.exe')
            script = Path(argv[2]).read_text(encoding='ascii')
            self.assertIn('chcp 65001', script)
            self.assertIn('type', script)
            self.assertIn(str(workspace / 'assets' / 'logo.txt'), script)
            self.assertIn('--reason-file', script)
            # The reason can hold Windows paths and quotes, so it never reaches the cmd line.
            self.assertNotIn('C:\\x', script)
            self.assertEqual((workspace / 'runtime' / 'console-reason.txt').read_text(encoding='utf-8'),
                             'reason: C:\\x "quoted"')

    def test_unavailable_console_input_is_not_an_exit_request(self):
        # The launcher hands this process a NUL stdin; reading that as "the user
        # pressed Enter" would tear the backend down as soon as it came up.
        with patch.object(console.sys, 'stdin', None):
            self.assertIsNone(console.ConsoleLog(Path('unused')).ask(''))
        with patch.object(console, 'input', side_effect=EOFError):
            self.assertIsNone(console.ConsoleLog(Path('unused')).ask(''))

    def test_console_log_survives_a_missing_console(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'console.log'
            log = console.ConsoleLog(path)
            with patch.object(console, 'print', side_effect=OSError('no console')):
                log.note('后端与监管进程已就绪')
            self.assertIn('后端与监管进程已就绪', path.read_text(encoding='utf-8'))

    def test_exception_and_failed_exit_surface_startup_error(self):
        for error in [RuntimeError('frontend missing'), SystemExit('backend failed')]:
            with patch.object(launcher, 'main', side_effect=error), patch.object(launcher, '_startup_error') as report:
                launcher.run()
                report.assert_called_once_with(error)

    def test_help_and_duplicate_window_are_not_crashes(self):
        for error in [SystemExit(0), SystemExit(None), SystemExit('LP LoRA Trainer 桌面窗口已打开')]:
            with patch.object(launcher, 'main', side_effect=error), patch.object(launcher, '_startup_error') as report:
                launcher.run()
                report.assert_not_called()

    def test_startup_error_logs_traceback_and_displays_log_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            module = root / 'desktop' / 'launcher.py'
            module.parent.mkdir()
            module.touch()
            with patch.object(launcher, '__file__', str(module)):
                if os.name == 'nt':
                    with patch('ctypes.windll.user32.MessageBoxW') as message:
                        launcher._startup_error(RuntimeError('启动失败 fixture'))
                        self.assertIn(str(root / 'runtime' / 'desktop.log'), message.call_args.args[1])
                else:
                    launcher._startup_error(RuntimeError('启动失败 fixture'))
            text = (root / 'runtime' / 'desktop.log').read_text(encoding='utf-8')
            self.assertIn('RuntimeError', text)
            self.assertIn('启动失败 fixture', text)

    def test_exe_launches_only_relative_pythonw_without_console(self):
        root = Path(__file__).resolve().parents[1]
        host = (root / 'desktop' / 'launcher_host.cs').read_text(encoding='utf-8-sig')
        self.assertIn('AppDomain.CurrentDomain.BaseDirectory', host)
        self.assertIn('Path.Combine(root, "python_runtime", "pythonw.exe")', host)
        self.assertIn('UseShellExecute = false', host)
        self.assertIn('CreateNoWindow = true', host)
        self.assertIn('desktop.launcher', host)
        self.assertNotIn('cmd.exe', host.lower())
        self.assertIn('desktop.launcher:run', (root / 'pyproject.toml').read_text(encoding='utf-8-sig'))
        if os.name == 'nt':
            self.assertEqual((root / 'LP-Lora-Trainer.exe').read_bytes()[:2], b'MZ')


if __name__ == '__main__':
    unittest.main()
