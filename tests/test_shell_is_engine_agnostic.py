"""The shell must not know engine names: all engine knowledge lives in adapters/.

This is the executable form of "every engine loads modularly on both sides": the
backend (``app/``) and the frontend (``frontend/src/``) may only reach an engine
through the adapter catalog, never through a hard-coded id or a per-engine branch.
Comments are ignored — only code (quoted literals) is scanned.
"""

import re
import unittest
from pathlib import Path

from adapters.registry import ADAPTERS

ROOT = Path(__file__).resolve().parents[1]
SHELL_DIRS = (ROOT / "app", ROOT / "frontend" / "src")
# Demo fixtures must name engines: they stand in for real catalog data.
FIXTURES = (ROOT / "frontend" / "src" / "api" / "demo.ts",)


def shell_files():
    for directory in SHELL_DIRS:
        for path in directory.rglob("*"):
            if path.suffix in {".py", ".ts", ".vue"} and path.is_file() and path not in FIXTURES:
                yield path


class ShellStaysEngineAgnosticTests(unittest.TestCase):
    def test_no_engine_id_is_hard_coded_in_the_shell(self):
        patterns = {
            engine_id: re.compile(r"""['"]%s['"]""" % re.escape(engine_id))
            for engine_id in ADAPTERS
        }
        offenders = []
        for path in shell_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for engine_id, pattern in patterns.items():
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(ROOT)}: {engine_id}")
        self.assertEqual([], offenders, "主程序/前端出现了硬编码的引擎标识")

    def test_shell_files_are_actually_scanned(self):
        """Guard against the scan silently matching nothing after a move."""
        files = list(shell_files())
        self.assertTrue(any(p.suffix == ".py" for p in files))
        self.assertTrue(any(p.suffix == ".vue" for p in files))


if __name__ == "__main__":
    unittest.main()
