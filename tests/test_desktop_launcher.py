"""No-GUI desktop startup and portable EXE contract tests."""
import ctypes
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
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


class KillOnCloseJobTests(unittest.TestCase):
    """The backend tree must die with the window, even when it is force-killed."""

    class FakeKernel32:
        def __init__(self, create=101, configure=True, assign=True):
            self.create, self.configure, self.assign = create, configure, assign
            self.calls = []
            self.limit_flags = None

        def CreateJobObjectW(self, attributes, name):
            self.calls.append(("create", name))
            return self.create

        def SetInformationJobObject(self, handle, info_class, info, size):
            self.calls.append(("configure", handle, info_class))
            self.limit_flags = info._obj.BasicLimitInformation.LimitFlags
            return 1 if self.configure else 0

        def CloseHandle(self, handle):
            self.calls.append(("close", handle))
            return 1

        def AssignProcessToJobObject(self, handle, raw):
            self.calls.append(("assign", handle, raw))
            return 1 if self.assign else 0

    def test_job_is_configured_to_kill_members_on_close(self):
        api = self.FakeKernel32()
        handle = console.create_kill_on_close_job(kernel32=api)
        self.assertEqual(handle, 101)
        self.assertEqual(api.limit_flags, console.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        self.assertEqual(next(c[2] for c in api.calls if c[0] == "configure"),
                         console.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION)

    def test_an_unusable_job_is_reported_not_raised(self):
        failing_create = self.FakeKernel32(create=0)
        self.assertIsNone(console.create_kill_on_close_job(kernel32=failing_create))
        failing_configure = self.FakeKernel32(configure=False)
        self.assertIsNone(console.create_kill_on_close_job(kernel32=failing_configure))
        self.assertIn(("close", 101), failing_configure.calls)

    def test_the_child_is_moved_into_the_job(self):
        api = self.FakeKernel32()
        process = SimpleNamespace(_handle=4242)
        self.assertTrue(console.assign_to_job(101, process, kernel32=api))
        self.assertIn(("assign", 101, 4242), api.calls)

    def test_assignment_is_skipped_without_a_job_or_handle(self):
        api = self.FakeKernel32()
        self.assertFalse(console.assign_to_job(None, SimpleNamespace(_handle=1), kernel32=api))
        self.assertFalse(console.assign_to_job(101, SimpleNamespace(), kernel32=api))
        self.assertFalse(console.assign_to_job(101, None, kernel32=api))
        self.assertEqual(api.calls, [])


class DesktopClosingTests(unittest.TestCase):
    """A busy backend is a question, not a refusal (installations are resumable)."""

    class FakeWindow:
        def __init__(self, answer=True):
            self.answer, self.dialogs = answer, []

        def create_confirmation_dialog(self, title, text):
            self.dialogs.append(text)
            return self.answer

    def closing(self, outcomes, answer=True):
        backend = launcher.DesktopBackend("http://127.0.0.1:5900")
        backend.ready.set()
        window = self.FakeWindow(answer)
        # Patch where the launcher looks it up, so no real HTTP call is made.
        with patch.object(launcher, "request_shutdown", side_effect=outcomes) as request:
            result = backend.closing(window)
        return result, backend, window, request

    def test_a_busy_backend_is_asked_and_forced_when_confirmed(self):
        result, backend, window, request = self.closing(["busy", "stopped"], answer=True)
        self.assertTrue(result)
        self.assertTrue(backend.shutdown_accepted)
        self.assertEqual(request.call_args_list[-1].kwargs, {"force": True})
        self.assertIn("强制停止环境安装并退出", window.dialogs[0])

    def test_declining_keeps_the_window_open(self):
        result, backend, window, request = self.closing(["busy", "stopped"], answer=False)
        self.assertFalse(result)
        self.assertFalse(backend.shutdown_accepted)
        self.assertEqual(request.call_count, 1)  # never forced without consent

    def test_training_still_cannot_be_forced_away(self):
        result, backend, window, request = self.closing(["busy", "busy"], answer=True)
        self.assertFalse(result)
        self.assertFalse(backend.shutdown_accepted)
        self.assertIn("训练任务", window.dialogs[-1])

    def test_a_clean_shutdown_asks_nothing(self):
        result, backend, window, request = self.closing(["stopped"])
        self.assertTrue(result)
        self.assertTrue(backend.shutdown_accepted)
        self.assertEqual(window.dialogs, [])
        self.assertEqual(request.call_count, 1)


class DesktopLauncherTests(unittest.TestCase):
    def setUp(self):
        # Launcher lifecycle tests must never contact the user's resident API.
        patcher = patch.object(launcher, 'request_shutdown', return_value='stopped')
        self.shutdown = patcher.start()
        self.addCleanup(patcher.stop)

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
            # A resident pre-catalog backend must not serve the new frontend.
            {'backend_version': '0.1.0', 'installation_workflow_version': 2},
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
