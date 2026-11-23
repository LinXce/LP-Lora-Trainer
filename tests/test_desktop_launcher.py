"""No-GUI desktop startup and portable EXE contract tests."""
import ctypes
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from desktop import console, launcher, window


def run_prepare(url, prepare=None, **_):
    """Stand-in for open_window: run the work the real splash window runs."""
    if prepare is not None:
        prepare()


class WindowResizeTests(unittest.TestCase):
    """A frameless window has no sizing border unless WS_THICKFRAME is restored."""

    class FakeUser32:
        def __init__(self, hwnd=4242, style=0x00CF0000):
            self.hwnd, self.style, self.calls = hwnd, style, []

        def FindWindowW(self, window_class, title):
            self.calls.append(('find', window_class, title))
            return self.hwnd

        def GetWindowLongW(self, hwnd, index):
            self.calls.append(('get', hwnd, index))
            return self.style

        def SetWindowLongW(self, hwnd, index, style):
            self.calls.append(('set', hwnd, index, style))
            return self.style

        def SetWindowPos(self, hwnd, *args):
            self.calls.append(('pos', hwnd, args))
            return 1

    class FakeDwmapi:
        def __init__(self):
            self.calls = []

        def DwmSetWindowAttribute(self, hwnd, attribute, value_ptr, size):
            value = ctypes.cast(value_ptr, ctypes.POINTER(ctypes.c_uint32)).contents.value
            self.calls.append((hwnd, attribute, value, size))
            return 0

    def test_dwm_border_and_caption_colours_are_the_page_colour(self):
        api, dwm = self.FakeUser32(), self.FakeDwmapi()
        window.enable_window_resize('LP LoRA Trainer', user32=api, dwmapi=dwm)
        attributes = {call[1]: call[2] for call in dwm.calls}
        self.assertEqual(attributes[window.DWMWA_BORDER_COLOR], 0x262626)
        self.assertEqual(attributes[window.DWMWA_CAPTION_COLOR], 0x262626)
        self.assertTrue(all(call[0] == api.hwnd for call in dwm.calls))

    def test_sizing_border_and_maximize_box_are_added(self):
        api = self.FakeUser32(style=0x00CF0000)
        self.assertTrue(window.enable_window_resize('LP LoRA Trainer', user32=api, dwmapi=self.FakeDwmapi()))
        hwnd, index, style = next(c[1:] for c in api.calls if c[0] == 'set')
        self.assertEqual(hwnd, api.hwnd)
        self.assertEqual(index, window.GWL_STYLE)
        self.assertTrue(style & window.WS_THICKFRAME, 'sizing border must be restored')
        self.assertTrue(style & window.WS_MAXIMIZEBOX)
        # Existing styles are preserved, not replaced.
        self.assertTrue(style & 0x00CF0000)
        flags = next(c[2][-1] for c in api.calls if c[0] == 'pos')
        self.assertTrue(flags & window.SWP_FRAMECHANGED, 'the new frame must be applied')

    def test_a_missing_window_is_reported_without_raising(self):
        self.assertFalse(window.enable_window_resize(
            'LP LoRA Trainer', user32=self.FakeUser32(hwnd=0), dwmapi=self.FakeDwmapi()))


class DesktopLauncherTests(unittest.TestCase):
    def test_matching_backend_protocol_is_reusable(self):
        with patch.object(launcher.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value.__enter__.return_value.status = 204
            with patch.object(launcher.json, 'load', return_value={
                'backend_version': launcher.APP_VERSION,
                'installation_workflow_version': launcher.INSTALLATION_WORKFLOW_VERSION,
            }):
                self.assertTrue(launcher.backend_ready('http://127.0.0.1:8765'))

    def test_old_or_incompatible_backend_is_not_silently_reused(self):
        for status in [
            {'backend_version': '0.0.9'},
            # A resident pre-catalog backend must not serve the new frontend.
            {'backend_version': '0.0.9', 'installation_workflow_version': 2},
            {'backend_version': '0.0.9', 'installation_workflow_version': 999},
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
                patch.object(launcher, 'backend_ready', side_effect=launcher.IncompatibleBackend('旧版或不兼容')), \
                patch.object(launcher.subprocess, 'Popen') as spawn, \
                patch.object(launcher, 'open_window', side_effect=run_prepare) as window_mock:
            lock.return_value.acquire.return_value = True
            with self.assertRaisesRegex(launcher.IncompatibleBackend, '旧版或不兼容'):
                launcher.main()
            spawn.assert_not_called()
            # The splash window opened, but it never navigated to the incompatible API.
            window_mock.assert_called_once()
            lock.return_value.close.assert_called_once()

    def test_working_desktop_window_never_shows_a_console(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'ensure_backend') as backend, \
                patch.object(launcher, 'spawn_console') as terminal, \
                patch.object(launcher, 'open_window', side_effect=run_prepare) as window_mock:
            lock.return_value.acquire.return_value = True
            launcher.main()
            window_mock.assert_called_once()
            terminal.assert_not_called()
            backend.assert_called_once()
            lock.return_value.close.assert_called_once()
            self.assertEqual(window_mock.call_args.kwargs['bounds_file'].name, 'window-bounds.json')

    def test_window_start_failure_falls_back_to_the_terminal(self):
        # WebView2/pywebview can fail after the bridge check passed; the user must
        # still get a usable surface instead of only an error box.
        order, reasons = [], []

        def fake_console(port, data_root, reason, workspace=None):
            order.append('console')
            reasons.append(reason)

        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock') as lock, \
                patch.object(launcher, 'open_window', side_effect=RuntimeError('WebView2 运行时缺失')), \
                patch.object(launcher, 'spawn_console', side_effect=fake_console):
            lock.return_value.acquire.return_value = True
            lock.return_value.close.side_effect = lambda: order.append('unlock')
            launcher.main()
        # The desktop lock is released first, otherwise the console cannot own it.
        self.assertEqual(order, ['unlock', 'console'])
        self.assertIn('WebView2 运行时缺失', reasons[0])
        self.assertTrue(lock.return_value.close.called)

    def test_window_and_terminal_failures_are_both_reported(self):
        with patch.object(launcher.sys, 'argv', ['desktop.launcher']), \
                patch.object(launcher, 'desktop_bridge_error', return_value=None), \
                patch.object(launcher, 'ProcessLock'), \
                patch.object(launcher, 'open_window', side_effect=RuntimeError('窗口失败')), \
                patch.object(launcher, 'spawn_console', side_effect=RuntimeError('终端失败')):
            with self.assertRaisesRegex(RuntimeError, '窗口失败.*终端回退也失败.*终端失败'):
                launcher.main()

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

    @unittest.skipUnless(os.name == 'nt', 'Windows terminal fallback')
    def test_fallback_uses_the_system_terminal_and_shows_the_logo(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'assets').mkdir()
            (workspace / 'assets' / 'logo.txt').write_text('LP ASCII LOGO', encoding='utf-8')
            (workspace / 'runtime').mkdir()
            data_root = workspace / 'data'
            with patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher.time, 'sleep'):
                spawn.return_value.poll.return_value = None
                launcher.spawn_console(5900, data_root, 'reason: C:\\x "quoted"', workspace)
            argv = spawn.call_args.args[0]
            # The window is the machine's own terminal, not a Python-owned console.
            self.assertEqual(Path(argv[0]).name, 'cmd.exe')
            script = Path(argv[2]).read_text(encoding='ascii')
            self.assertIn('chcp 65001', script)
            self.assertIn('type', script)
            # Everything is addressed through the script's own directory (%~dp0),
            # so no absolute workspace path is embedded.
            self.assertIn('%~dp0', script)
            self.assertIn('"%~dp0..\\assets\\logo.txt"', script)
            self.assertIn('--reason-file', script)
            self.assertNotIn(str(workspace), script)
            self.assertNotIn(str(data_root), script)
            # The reason can hold Windows paths and quotes, so it never reaches the cmd line.
            self.assertNotIn('C:\\x', script)
            # The data root travels in the environment instead of the script.
            self.assertEqual(spawn.call_args.kwargs['env'][console.CONSOLE_DATA_ROOT_ENV],
                             str(data_root.resolve()))
            self.assertEqual((workspace / 'runtime' / 'console-reason.txt').read_text(encoding='utf-8'),
                             'reason: C:\\x "quoted"')

    @unittest.skipUnless(os.name == 'nt', 'Windows terminal fallback')
    def test_non_ascii_workspace_still_produces_a_pure_ascii_script(self):
        # A workspace under a Chinese folder used to raise UnicodeEncodeError while
        # writing browser-mode.cmd, failing the very fallback that had just started.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / '中文 工作区'
            (workspace / 'assets').mkdir(parents=True)
            (workspace / 'assets' / 'logo.txt').write_text('LP ASCII LOGO', encoding='utf-8')
            (workspace / 'runtime').mkdir()
            data_root = workspace / '数据目录'
            with patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher.time, 'sleep'):
                spawn.return_value.poll.return_value = None
                launcher.spawn_console(5901, data_root, '原因：桌面窗口失败', workspace)
            script = (workspace / 'runtime' / 'browser-mode.cmd').read_text(encoding='ascii')
            self.assertIn('%~dp0', script)
            self.assertNotIn('中文', script)
            self.assertNotIn(str(workspace), script)
            self.assertEqual(spawn.call_args.kwargs['env'][console.CONSOLE_DATA_ROOT_ENV],
                             str(data_root.resolve()))
            self.assertEqual((workspace / 'runtime' / 'console-reason.txt').read_text(encoding='utf-8'),
                             '原因：桌面窗口失败')

    @unittest.skipUnless(os.name == 'nt', 'Windows terminal fallback')
    def test_no_data_root_leaves_the_environment_variable_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'assets').mkdir()
            (workspace / 'runtime').mkdir()
            with patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher.time, 'sleep'), \
                    patch.dict(os.environ, {console.CONSOLE_DATA_ROOT_ENV: 'D:/stale'}, clear=False):
                spawn.return_value.poll.return_value = None
                launcher.spawn_console(5900, None, 'reason', workspace)
            self.assertNotIn(console.CONSOLE_DATA_ROOT_ENV, spawn.call_args.kwargs['env'])

    def test_console_reads_the_data_root_from_the_environment(self):
        # The launcher passes the root through the environment because the .cmd file
        # must stay ASCII; console.main() has to consume it when --data-root is absent.
        with patch.object(console, 'run', return_value=0) as run, \
                patch.object(console.sys, 'argv', ['desktop.console']), \
                patch.dict(os.environ, {console.CONSOLE_DATA_ROOT_ENV: 'D:/lp-data'}, clear=False):
            with self.assertRaises(SystemExit):
                console.main()
        self.assertEqual(run.call_args.args[1], 'D:/lp-data')

    def test_console_argument_data_root_wins_over_the_environment(self):
        with patch.object(console, 'run', return_value=0) as run, \
                patch.object(console.sys, 'argv', ['desktop.console', '--data-root', 'E:/explicit']), \
                patch.dict(os.environ, {console.CONSOLE_DATA_ROOT_ENV: 'D:/lp-data'}, clear=False):
            with self.assertRaises(SystemExit):
                console.main()
        self.assertEqual(run.call_args.args[1], 'E:/explicit')

    def test_console_without_a_data_root_uses_the_workspace_default(self):
        with patch.object(console, 'run', return_value=0) as run, \
                patch.object(console.sys, 'argv', ['desktop.console']), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop(console.CONSOLE_DATA_ROOT_ENV, None)
            with self.assertRaises(SystemExit):
                console.main()
        self.assertIsNone(run.call_args.args[1])

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
