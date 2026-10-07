"""No-GUI desktop startup and portable EXE contract tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from desktop import launcher


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
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'backend_ready', side_effect=RuntimeError('旧版或不兼容')), \
                patch.object(launcher.subprocess, 'Popen') as spawn, \
                patch.object(launcher, 'open_window') as window:
            lock.return_value.acquire.return_value = True
            with self.assertRaisesRegex(RuntimeError, '旧版或不兼容'):
                launcher.main()
            spawn.assert_not_called()
            window.assert_not_called()
            lock.return_value.close.assert_called_once()

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
