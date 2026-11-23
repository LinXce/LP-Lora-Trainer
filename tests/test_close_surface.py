"""Close protocols with fake window, process and HTTP transport. No real ports."""
import threading
import unittest
from unittest.mock import Mock, call, patch

from desktop import console
from desktop.bridge import DesktopBridge
from desktop.launcher import DesktopBackend
from desktop.shutdown import close_surface


class CloseSurfaceTests(unittest.TestCase):
    def setUp(self):
        client = patch("desktop.shutdown.ShutdownClient").start()
        self.addCleanup(patch.stopall)
        patch("desktop.shutdown.time.sleep").start()
        self.request = client.return_value.call
        self.process = Mock(pid=42)
        self.process.poll.return_value = None
        self.ask = Mock(return_value=True)
        self.progress = Mock()

    @staticmethod
    def state(reason="training", status="idle", **extra):
        return dict(backend_pid=42, reason=reason, status=status, task_id="A", **extra)

    def decide(self, owned=True):
        return close_surface("http://test.invalid", self.process if owned else None, self.ask, self.progress)

    def test_foreign_active_cancel_and_detach_only_read(self):
        for answer in (False, True):
            self.request.reset_mock()
            self.request.return_value = self.state()
            self.ask.return_value = answer
            self.assertEqual(self.decide(False), answer)
            self.request.assert_called_once_with()
            self.assertEqual(self.ask.call_args.args[0], "foreign")

    def test_foreign_idle_closes_without_touching_backend(self):
        self.request.return_value = self.state("idle")
        self.assertTrue(self.decide(False))
        self.request.assert_called_once_with()
        self.ask.assert_not_called()

    def test_replacement_for_exited_owned_backend_is_foreign(self):
        self.process.poll.return_value = 0
        self.request.return_value = dict(self.state(), backend_pid=99)
        for answer in (False, True):
            self.ask.return_value = answer
            self.assertEqual(self.decide(), answer)
            self.assertEqual(self.ask.call_args.args[0], "foreign")
        self.assertEqual(self.request.call_args_list, [call(), call()])

    def test_unknown_or_changed_pid_never_writes(self):
        self.request.side_effect = OSError("offline")
        self.assertFalse(self.decide())
        self.assertTrue(self.decide(False))
        self.request.side_effect = None
        self.request.return_value = dict(self.state(), backend_pid=99)
        self.assertFalse(self.decide())
        self.assertEqual(self.request.call_args_list, [call(), call(), call()])

    def test_initial_cancel_issues_no_post(self):
        self.request.return_value = self.state()
        self.ask.return_value = False
        self.assertFalse(self.decide())
        self.request.assert_called_once_with()

    def test_training_exits_after_drain_and_saved_result(self):
        self.request.side_effect = [
            self.state(),
            self.state(status="draining"),
            self.state(status="ready_to_exit", save_result=dict(message="已找到完整状态")),
            self.state(status="shutting_down"),
        ]
        self.assertTrue(self.decide())
        self.assertEqual(self.request.call_args_list, [
            call(), call("?graceful=true&expected_reason=training&expected_task_id=A", "POST"),
            call(), call("?graceful=true&expected_reason=training&expected_task_id=A", "POST"),
        ])
        self.progress.assert_any_call("已找到完整状态", False)
        self.process.wait.assert_called_once()

    def test_force_requires_a_second_confirmation(self):
        self.ask.side_effect = [True, False]
        self.request.side_effect = [self.state(), self.state(status="blocked", force_allowed=True), {}]
        self.assertFalse(self.decide())
        self.assertEqual([event.args[0] for event in self.ask.call_args_list], ["training", "force"])
        self.assertEqual(self.request.call_args_list[-1], call("/cancel", "POST"))
        self.assertFalse(any("force=true" in str(c) for c in self.request.call_args_list))

    def test_queued_only_uses_a_state_precondition(self):
        self.request.side_effect = [dict(self.state("queued"), task_id=None), self.state(status="shutting_down")]
        self.assertTrue(self.decide())
        self.assertEqual(self.ask.call_args.args[0], "queued")
        self.assertEqual(self.request.call_args_list[-1], call("?graceful=true&expected_reason=queued", "POST"))

    def test_console_enter_or_missing_consent_preserves_active_training(self):
        self.request.return_value = self.state()
        for reply in (None, "", "n"):
            self.assertFalse(console.finish(self.process, "http://test.invalid", Mock(), Mock(return_value=reply)))
        self.assertEqual(self.request.call_args_list, [call(), call(), call()])
        self.assertTrue(console.finish(None, "http://test.invalid", Mock(), Mock(return_value="y")))
        self.assertEqual(self.request.call_args_list[-1], call())

    def test_cancel_wait_releases_fence_instead_of_finalizing(self):
        self.request.side_effect = [self.state(status="draining"), {}]
        self.assertFalse(close_surface("http://test.invalid", self.process, self.ask, self.progress, lambda: True))
        self.assertEqual(self.request.call_args_list, [call(), call("/cancel", "POST")])
        self.process.wait.assert_not_called()

    def test_confirmed_force_only_targets_the_previously_blocked_task(self):
        self.request.side_effect = [
            self.state(), self.state(status="blocked", force_allowed=True), self.state(status="shutting_down")
        ]
        self.assertTrue(self.decide())
        self.assertEqual(self.request.call_args_list[-1],
                         call("?graceful=true&expected_reason=training&expected_task_id=A&force=true", "POST"))
        self.assertEqual([event.args[0] for event in self.ask.call_args_list], ["training", "force"])

    def test_both_gestures_share_a_nonblocking_decision(self):
        for gesture in ("titlebar", "os"):
            for owned in (False, True):
                with self.subTest(gesture=gesture, owned=owned):
                    self.request.reset_mock()
                    entered, release = threading.Event(), threading.Event()
                    def read(*_):
                        entered.set()
                        self.assertTrue(release.wait(3))
                        return self.state()
                    self.request.side_effect = read
                    backend = DesktopBackend("http://test.invalid")
                    backend.prepare(lambda: self.process if owned else None)
                    window = Mock()
                    window.evaluate_js.return_value = False
                    window.create_confirmation_dialog.return_value = not owned
                    bridge = DesktopBridge()
                    bridge._window = window
                    bridge.close_decider = backend.closing
                    if gesture == "titlebar":
                        bridge.window_action("close")
                    else:
                        self.assertFalse(backend.closing(window))
                    self.assertTrue(entered.wait(3))
                    thread = backend._thread
                    self.assertFalse(backend.closing(window))
                    release.set()
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
                    self.request.assert_called_once_with()
                    window.create_confirmation_dialog.assert_called_once()
                    self.assertEqual(window.destroy.called, not owned)
