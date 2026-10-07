"""Desktop window controls without opening a GUI or touching running training."""
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from desktop.bridge import DesktopBridge
from desktop import window_chrome
from desktop.window import open_window


class FakeEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self):
        for handler in self.handlers:
            handler()


class FakeWindow:
    def __init__(self):
        self.native = None
        self.events = SimpleNamespace(**{
            name: FakeEvent() for name in (
                "before_show", "before_load", "loaded", "maximized", "restored", "closed",
            )
        })
        self.minimize = Mock()
        self.destroy = Mock(side_effect=self.events.closed.fire)
        self.evaluate_js = Mock()
        self.maximize = Mock(side_effect=self.events.maximized.fire)
        self.restore = Mock(side_effect=self.events.restored.fire)


class DesktopBridgeTests(unittest.TestCase):
    def setUp(self):
        self.window = FakeWindow()
        self.bridge = DesktopBridge()
        self.bridge._attach(self.window)

    def test_unattached_bridge_rejects_actions(self):
        bridge = DesktopBridge()
        with self.assertRaises(RuntimeError):
            bridge.window_action("minimize")
        with self.assertRaises(RuntimeError):
            bridge.get_window_state()

    def test_action_allowlist(self):
        for action in ("exit_backend", "run", "maximize", "", None, [], {}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.bridge.window_action(action)
        self.window.destroy.assert_not_called()
        self.window.minimize.assert_not_called()

    def test_minimize(self):
        state = self.bridge.window_action("minimize")
        self.window.minimize.assert_called_once_with()
        self.assertFalse(state["maximized"])

    def test_toggle_maximize_and_restore(self):
        self.assertTrue(self.bridge.window_action("toggle_maximize")["maximized"])
        self.window.maximize.assert_called_once_with()
        self.assertFalse(self.bridge.window_action("toggle_maximize")["maximized"])
        self.window.restore.assert_called_once_with()

    def test_close_only_destroys_window(self):
        self.window.events.loaded.fire()
        with patch("desktop.bridge.subprocess.Popen") as spawn:
            self.assertIsNone(self.bridge.window_action("close"))
            spawn.assert_not_called()
        self.window.destroy.assert_called_once_with()
        self.assertFalse(self.bridge._page_loaded)

    def test_native_events_notify_page_and_synchronize_icon(self):
        self.window.events.maximized.fire()
        self.window.evaluate_js.assert_not_called()
        self.window.events.loaded.fire()
        script = self.window.evaluate_js.call_args.args[0]
        self.assertIn("lp-window-state", script)
        self.assertIn('"maximized": true', script)
        self.window.events.restored.fire()
        self.assertIn('"maximized": false', self.window.evaluate_js.call_args.args[0])

    def test_reload_does_not_send_to_old_page(self):
        self.window.events.loaded.fire()
        self.window.events.before_load.fire()
        self.window.evaluate_js.reset_mock()
        self.window.events.maximized.fire()
        self.window.evaluate_js.assert_not_called()
        self.window.events.loaded.fire()
        self.assertIn('"maximized": true', self.window.evaluate_js.call_args.args[0])

    def test_notification_close_race_is_safe(self):
        self.window.evaluate_js.side_effect = RuntimeError("Window closed")
        self.window.events.loaded.fire()
        self.assertTrue(self.bridge._page_loaded)

    def test_before_show_configures_frame(self):
        with patch("desktop.window_chrome.fit_work_area") as fit:
            self.window.events.before_show.fire()
            fit.assert_called_once_with(self.window)

    def test_resize_delegates_only_to_attached_window(self):
        with patch("desktop.window_chrome.begin_resize") as resize:
            self.bridge.begin_window_resize("se")
            resize.assert_called_once_with(self.window, "se")

    def test_native_state_overrides_out_of_order_event(self):
        with patch("desktop.window_chrome.is_maximized", return_value=False):
            self.window.events.maximized.fire()
            self.assertFalse(self.bridge.get_window_state()["maximized"])


class WindowsFrameTests(unittest.TestCase):
    def test_resize_allowlist(self):
        for edge in ("move", "close", None, [], ""):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                window_chrome.begin_resize(FakeWindow(), edge)

    def test_native_resize_posts_single_message_without_mousemove_loop(self):
        user32 = Mock()
        user32.PostMessageW.return_value = True

        def cursor_position(pointer):
            pointer._obj.x, pointer._obj.y = -123, 40
            return True

        user32.GetCursorPos.side_effect = cursor_position
        handle = SimpleNamespace(ToInt64=lambda: 12345)
        window = SimpleNamespace(native=SimpleNamespace(Handle=handle))
        with patch("desktop.window_chrome.os.name", "nt"), \
                patch("desktop.window_chrome._user32", return_value=user32), \
                patch("desktop.window_chrome.is_maximized", return_value=False):
            window_chrome.begin_resize(window, "se")
        user32.ReleaseCapture.assert_called_once_with()
        user32.PostMessageW.assert_called_once_with(12345, 0x00A1, 17, (-123 & 0xFFFF) | (40 << 16))

    def test_resize_ignored_when_maximized(self):
        window = SimpleNamespace(native=object())
        with patch("desktop.window_chrome.os.name", "nt"), \
                patch("desktop.window_chrome.is_maximized", return_value=True), \
                patch("desktop.window_chrome._user32") as api:
            window_chrome.begin_resize(window, "se")
            api.assert_not_called()

    def test_window_creation_is_frameless_and_drag_is_restricted(self):
        fake_webview = SimpleNamespace(create_window=Mock(return_value=FakeWindow()), start=Mock())
        with patch.dict(sys.modules, {"webview": fake_webview}):
            open_window("http://127.0.0.1:8765")
        options = fake_webview.create_window.call_args.kwargs
        self.assertTrue(options["frameless"])
        self.assertFalse(options["easy_drag"])
        self.assertTrue(options["resizable"])
        self.assertEqual(options["min_size"], (1000, 700))
        self.assertEqual(options["background_color"], "#262626")
        fake_webview.start.assert_called_once_with(
            gui="edgechromium" if os.name == "nt" else None, debug=False,
            icon=str(Path(__file__).resolve().parents[1] / "assets" / "logo.ico"),
        )


if __name__ == "__main__":
    unittest.main()
