"""Relocation repair must never rewrite paths outside the known old project."""
import ast
import csv
from pathlib import Path
import tempfile
import unittest

from scripts.repair_environment import rebase_path, repair_finder, refresh_records


class RelocationTests(unittest.TestCase):
    def test_only_paths_below_old_root_are_rebased(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            self.assertEqual(rebase_path(str(old / "app"), old, new), str(new / "app"))
            with self.assertRaises(ValueError):
                rebase_path(str(root / "old-other" / "app"), old, new)

    def test_finder_repair_and_repeat_are_safe(self):
        with tempfile.TemporaryDirectory(prefix="lp relocate ") as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            for package in ("app", "desktop", "adapters", "supervisor"):
                (new / package).mkdir(parents=True)
            mapping = {package: str(old / package) for package in ("app", "desktop", "adapters", "supervisor")}
            namespaces = {"adapters.kohya.templates": [str(old / "adapters" / "kohya" / "templates")]}
            source = f"MAPPING: dict[str, str] = {mapping!r}\nNAMESPACES: dict[str, list[str]] = {namespaces!r}\n# preserve machinery\n"
            repaired, previous = repair_finder(source, new)
            self.assertEqual(previous, old)
            self.assertNotIn(str(old).replace("\\", "\\\\"), repaired)
            self.assertIn("# preserve machinery", repaired)
            ast.parse(repaired)
            again, previous = repair_finder(repaired, new)
            self.assertEqual(previous, new)
            self.assertEqual(again, repaired)

    def test_foreign_namespace_is_not_silently_relocated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new" / "app").mkdir(parents=True)
            source = f"MAPPING: dict = {{'app': {str(root / 'old' / 'app')!r}}}\nNAMESPACES: dict = {{'external': [{str(root / 'external')!r}]}}\n"
            with self.assertRaises(ValueError):
                repair_finder(source, root / "new")

    def test_changed_installed_file_hashes_are_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "Lib" / "site-packages"
            metadata = site / "test.dist-info"
            metadata.mkdir(parents=True)
            launcher = root / "Scripts" / "command.exe"
            launcher.parent.mkdir()
            launcher.write_bytes(b"new launcher")
            untouched = site / "untouched.py"
            untouched.write_bytes(b"unchanged")
            record = metadata / "RECORD"
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows([
                    ["../../Scripts/command.exe", "sha256=old", "1"],
                    ["untouched.py", "sha256=preserved", "9"],
                    ["test.dist-info/RECORD", "", ""],
                ])
            refresh_records(site, [launcher])
            with record.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertNotEqual(rows[0][1], "sha256=old")
            self.assertEqual(rows[0][2], str(len(b"new launcher")))
            self.assertEqual(rows[1], ["untouched.py", "sha256=preserved", "9"])
            self.assertEqual(rows[2], ["test.dist-info/RECORD", "", ""])
            before = record.read_bytes()
            refresh_records(site, [launcher])
            self.assertEqual(before, record.read_bytes())

    def test_missing_packages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = f"MAPPING: dict = {{'app': {str(root / 'old' / 'app')!r}}}\nNAMESPACES: dict = {{}}\n"
            with self.assertRaises(ValueError):
                repair_finder(source, root / "missing")


if __name__ == "__main__":
    unittest.main()
