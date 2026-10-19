"""Relocation regression tests: the cases the first migration pass missed.

Each case here corresponds to a defect found in a real rename, not to a hypothetical
one: a quoted ``pyvenv.cfg`` value, a path spelled with an 8.3 short name, and a
clone renamed onto a name that already existed (directory present, environment stale).
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import repair_engine_venv as relocate  # noqa: E402
from app.services.engines import _clean_locator  # noqa: E402


class LocatorTests(unittest.TestCase):
    def test_quoted_recorded_path_is_unwrapped(self):
        self.assertEqual(r"H:\A\python.exe", _clean_locator(r'"H:\A\python.exe"'))
        self.assertEqual(r"H:\A\python.exe", _clean_locator("'H:\\A\\python.exe'"))
        self.assertEqual(r"H:\A\python.exe", _clean_locator("  H:\\A\\python.exe  "))

    def test_inner_quotes_are_left_alone(self):
        self.assertEqual(r"H:\A\'weird\'", _clean_locator(r"H:\A\'weird\'"))


class RebaseIfInsideTests(unittest.TestCase):
    def test_plain_spelling_is_rebased(self):
        old, new = Path(r"H:\P\engine\kohya_old"), Path(r"H:\P\engine\kohya_new")
        value = str(old / ".lp" / "python" / "cpython-3.11.17" / "python.exe")
        self.assertEqual(str(new / ".lp" / "python" / "cpython-3.11.17" / "python.exe"),
                         relocate.rebase_if_inside(value, old, new))

    def test_case_difference_is_rebased(self):
        old, new = Path(r"H:\P\engine\Kohya_Old"), Path(r"H:\P\engine\kohya_new")
        value = str(old).lower() + r"\.lp\python\python.exe"
        self.assertTrue(relocate.rebase_if_inside(value, old, new).startswith(str(new)))

    def test_short_name_spelling_is_rebased(self):
        """The first pass compared only literal spellings and silently kept the old path."""
        base = Path(tempfile.mkdtemp())
        old, new = base / "engine" / "kohya_old", base / "engine" / "kohya_new"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        short = str(old).replace("Administrator", "ADMINI~1") if "Administrator" in str(old) else str(old)
        value = short + r"\.lp\python\python.exe"
        result = relocate.rebase_if_inside(value, old, new)
        self.assertIn(str(new), result)
        self.assertNotIn("kohya_old", result)

    def test_path_outside_the_old_root_is_kept(self):
        old, new = Path(r"H:\P\engine\kohya_old"), Path(r"H:\P\engine\kohya_new")
        self.assertEqual(r"C:\Other\python.exe",
                         relocate.rebase_if_inside(r"C:\Other\python.exe", old, new))


class PyvenvConfigTests(unittest.TestCase):
    def test_quoted_home_is_rewritten_bare_and_rebased(self):
        base = Path(tempfile.mkdtemp())
        old, new = base / "engine" / "kohya_old", base / "engine" / "kohya_new"
        venv = new / "venv"
        (venv / "Lib" / "site-packages").mkdir(parents=True)
        (venv / "Scripts").mkdir()
        (new / ".lp" / "python" / "cpython-3.11.17").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            'home = "' + str(old / ".lp" / "python" / "cpython-3.11.17") + '"\nversion_info = 3.11.17\n',
            encoding="utf-8",
        )
        argv, sys.argv = sys.argv, ["repair", str(venv), "--old-root", str(old), "--new-root", str(new)]
        try:
            relocate.main()
        finally:
            sys.argv = argv
        text = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        self.assertNotIn('"', text, "pyvenv.cfg 的值不能带引号，否则解释器报 No Python at")
        # The script writes the resolved spelling, so compare against it, not the raw temp path.
        self.assertIn(str((new / ".lp" / "python" / "cpython-3.11.17").resolve()), text)
        self.assertNotIn("kohya_old", text)


if __name__ == "__main__":
    unittest.main()
