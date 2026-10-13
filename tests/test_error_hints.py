"""Failure explanations for finished engine sessions."""
import unittest

from app.services.error_hints import (
    MAX_HINTS,
    analyze,
    mirror_fallback_notice,
    retry_reason,
)


class HintRuleTests(unittest.TestCase):
    def hints(self, log, state="failed", **kwargs):
        return analyze(state, kwargs.pop("error", None), kwargs.pop("diagnostic_error", None), log, **kwargs)

    def test_managed_python_download_wins_over_the_generic_network_rule(self):
        log = (
            "[LP] H:\\uv.exe venv engine\\venv --python 3.11 --managed-python --seed\n"
            "error: Request failed after 3 retries in 8.9s\n"
            "  cause: Failed to download `https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.11.17.tar.gz`\n"
            "  cause: tcp connect error\n"
            "  cause: 由于目标计算机积极拒绝，无法连接。 (os error 10061)\n"
        )
        hints = self.hints(log)
        self.assertEqual(hints[0]["id"], "python_download")
        self.assertEqual([h["id"] for h in hints].count("python_download"), 1)
        self.assertIn("python-build-standalone", hints[0]["evidence"][0])
        self.assertIn("Python 下载源", hints[0]["suggestion"])

    def test_rules_cover_the_common_installer_failures(self):
        cases = [
            ("network_unreachable", "error: connection refused\ncause: os error 10061"),
            ("network_timeout", "error: Read timed out after 30s"),
            ("index_missing", "ERROR: No matching distribution found for torch==2.14.1+cu130"),
            ("cuda_driver", "RuntimeError: CUDA driver version is insufficient for CUDA runtime version"),
            ("disk_space", "OSError: [Errno 28] No space left on device"),
            ("permission", "PermissionError: [WinError 5] Access is denied: 'cache'"),
            ("python_env", "[LP] Failed to create engine environment: uv venv failed with exit code 2"),
            ("git_repository", "remote: Repository not found.\nfatal: could not read Username"),
            ("installer_exit", "[LP] Installation failed: Official installer failed (step 1, exit code 1)"),
        ]
        for expected, log in cases:
            with self.subTest(rule=expected):
                hints = self.hints(log, kind="clone" if expected == "git_repository" else None)
                self.assertIn(expected, [hint["id"] for hint in hints])

    def test_evidence_is_deduplicated_and_capped(self):
        log = "no space left on device\nno space left on device\nno space left on device\nOSError: not enough space"
        hints = self.hints(log)
        self.assertEqual(hints[0]["id"], "disk_space")
        self.assertEqual(len(hints[0]["evidence"]), 2)

    def test_diagnostic_failure_is_reported_first(self):
        hints = self.hints("boom", diagnostic_error="Engine entry point is missing")
        self.assertEqual(hints[0]["id"], "diagnostic")
        self.assertEqual(hints[0]["detail"], "Engine entry point is missing")

    def test_only_finished_sessions_are_explained(self):
        for state in ("succeeded", "queued", "running", "stopping", "verifying"):
            with self.subTest(state=state):
                self.assertEqual(self.hints("no space left on device", state=state), [])

    def test_stopped_sessions_get_a_resume_hint_not_an_error(self):
        for state, title in (("cancelled", "安装已停止"), ("interrupted", "安装被中断")):
            with self.subTest(state=state):
                hints = self.hints("error: boom", state=state)
                self.assertEqual([hint["id"] for hint in hints], [state])
                self.assertEqual(hints[0]["title"], title)

    def test_unknown_failure_falls_back_to_the_error_text(self):
        hints = self.hints("something entirely unexpected", error="Installation failed: step 1")
        self.assertEqual(hints[0]["id"], "unknown")
        self.assertEqual(hints[0]["detail"], "Installation failed: step 1")
        self.assertEqual(hints[0]["title"], "运行失败，原因未识别")

    def test_clone_sessions_use_pull_wording(self):
        hints = self.hints("", kind="clone", error="git clone failed")
        self.assertEqual(hints[0]["title"], "拉取失败，原因未识别")

    def test_hint_count_is_capped(self):
        log = "\n".join([
            "error: Read timed out",
            "ERROR: No matching distribution found for torch",
            "RuntimeError: CUDA driver version is insufficient for CUDA runtime version",
            "OSError: No space left on device",
            "PermissionError: Access is denied",
            "fatal: could not resolve host",
            "Official installer failed (step 3, exit code 1)",
        ])
        hints = self.hints(log)
        self.assertEqual(len(hints), MAX_HINTS)


class MirrorFallbackTests(unittest.TestCase):
    def test_only_source_related_failures_are_retried(self):
        cases = [
            ("error: Failed to download python-build-standalone", "无法下载受管 Python"),
            ("ERROR: No matching distribution found for torch==2.14.1+cu130", "依赖源缺少所需包或版本"),
            ("cause: tcp connect error\ncause: os error 10061", "无法连接依赖源"),
            ("error: Read timed out after 30s", "下载依赖超时"),
        ]
        for log, reason in cases:
            with self.subTest(log=log):
                self.assertEqual(retry_reason(log), reason)

    def test_missing_pkg_resources_is_explained_but_never_retried(self):
        hints = analyze("failed", None, None, "ModuleNotFoundError: No module named 'pkg_resources'")
        self.assertEqual(hints[0]["id"], "missing_pkg_resources")
        self.assertIn("setuptools<82", hints[0]["suggestion"])
        self.assertIsNone(retry_reason("ModuleNotFoundError: No module named 'pkg_resources'"))

    def test_engine_failures_are_not_retried(self):
        for log in (
            "Official installer failed (step 2, exit code 1)",
            "Traceback (most recent call last):\nValueError: bad config",
            "",
        ):
            with self.subTest(log=log):
                self.assertIsNone(retry_reason(log))

    def test_notice_reports_every_outcome(self):
        base = mirror_fallback_notice("无法连接依赖源")
        self.assertEqual(base["id"], "mirror_fallback")
        self.assertEqual(base["evidence"], [])
        self.assertIn("正在改用官方源重试", base["detail"])
        self.assertIn(
            "已改用官方源完成安装",
            mirror_fallback_notice("无法连接依赖源", "succeeded")["detail"],
        )
        self.assertIn(
            "仍然失败",
            mirror_fallback_notice("无法连接依赖源", "failed")["detail"],
        )


if __name__ == '__main__':
    unittest.main()
