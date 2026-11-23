"""Offline relocation coverage for engine package installers' editable import hooks."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import normalize_editable as editable


class EditableNormalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / "original with spaces #1"
        self.env = self.checkout / ".lp/python"
        self.site = self.env / "Lib/site-packages"
        self.site.mkdir(parents=True)
        self.layout = self.env / "python312._pth"
        self.layout.write_text(".\nLib/site-packages\n", encoding="utf-8")

    def dist(self, name, source, *record_paths, editable=True):
        info = self.site / f"{name}-1.0.dist-info"
        info.mkdir()
        (info / "direct_url.json").write_text(json.dumps({
            "url": source.as_uri(), "dir_info": {"editable": editable}}), encoding="utf-8")
        with (info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows((p, "", "") for p in record_paths)

    def module(self, root, name):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("ORIGIN = __file__\n", encoding="utf-8")
        return folder

    def normalize(self, env=None):
        env = env or self.env
        with patch.object(editable, "_site_packages", return_value=env / "Lib/site-packages"):
            self.assertEqual(editable.normalize(env), 0)

    def probe(self, site, modules, *, embedded=False):
        if embedded:
            layout = site.parents[1] / "python312._pth"
            startup = ("from pathlib import Path; import sys; layout = Path(sys.argv[1]); "
                       "sys.path.extend(str((layout.parent / p).resolve()) for p in "
                       "layout.read_text(encoding='utf-8').splitlines() if p and not p.startswith('#')); ")
        else:
            layout = site
            startup = "import site, sys; site.addsitedir(sys.argv[1]); "
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", startup +
             "import importlib,json; print(json.dumps([importlib.import_module(n).__file__ for n in sys.argv[2:]]))",
             str(layout), *modules],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return [Path(entry).resolve() for entry in json.loads(result.stdout)]

    def test_copy_uses_only_copied_sources_for_mixed_build_backends_in_both_startup_modes(self):
        alpha = self.module(self.checkout / "nonstandard source", "alpha")
        beta = self.module(self.checkout / "nested" / "src", "beta")
        gamma = self.module(self.checkout / "other", "gamma")
        finder = self.site / "__editable___alpha_1_finder.py"
        finder.write_text(
            f"from pathlib import Path\nimport sys\nMAPPING: dict = {{'alpha': {str(alpha)!r}}}\n"
            f"NAMESPACES = {{'alpha': [{str(alpha)!r}]}}\n"
            "def install():\n    sys.path.append(str(Path(MAPPING['alpha']).parent))\n", encoding="utf-8")
        hook = self.site / "__editable__.alpha-1.pth"
        hook.write_text(f"import {finder.stem}; {finder.stem}.install()\n", encoding="utf-8")
        self.dist("alpha", self.checkout, hook.name, finder.name)
        for module, pth_name in ((beta, "ordinary-backend.pth"), (gamma, "_editable_impl_gamma.pth")):
            (self.site / pth_name).write_text(str(module.parent) + "\n", encoding="utf-8")
            self.dist(module.name, module.parent, pth_name, "../unsafe.pth", "..\\unsafe.pth")
        untouched = self.site / "third-party.pth"
        original = b"# unrelated runtime behavior\nimport os; os.environ['lp_editable_probe'] = 'yes'\n"
        untouched.write_bytes(original)
        self.dist("regular", self.checkout, untouched.name, editable=False)
        self.normalize()

        copied = self.root / "copy with a different name"
        shutil.copytree(self.checkout, copied)
        new_env = copied / ".lp/python"
        new_site = new_env / "Lib/site-packages"
        expected = [copied / module.relative_to(self.checkout) / "__init__.py" for module in (alpha, beta, gamma)]
        self.assertEqual(self.probe(new_site, ["alpha", "beta", "gamma"]), expected)
        self.assertEqual(self.probe(new_site, ["alpha", "beta", "gamma"], embedded=True), expected)
        before = {p.relative_to(copied): p.read_bytes() for p in [
            new_env / self.layout.name, new_site / finder.name, new_site / hook.name, new_site / "ordinary-backend.pth"]}
        self.normalize(new_env)
        for name, content in before.items():
            self.assertEqual((copied / name).read_bytes(), content)
        self.assertEqual((new_site / untouched.name).read_bytes(), original)
        self.assertEqual(self.probe(new_site, ["alpha", "beta", "gamma"]), expected)

    def test_missing_finder_and_deleted_plain_hook_are_recovered_from_each_editable(self):
        first = self.module(self.checkout / "src", "first")
        second = self.module(self.checkout / "inner", "second")
        hook = self.site / "__editable__.first-1.pth"
        hook.write_text("import __editable___first_1_finder; __editable___first_1_finder.install()\n",
                        encoding="utf-8")
        self.dist("first", self.checkout, hook.name, "__editable___first_1_finder.py")
        self.dist("second", second.parent, "_editable_impl_second.pth")
        self.normalize()
        self.assertNotIn("import ", hook.read_text(encoding="utf-8"))
        self.assertEqual(self.probe(self.site, ["first", "second"]),
                         [first / "__init__.py", second / "__init__.py"])

    def test_normalizing_an_already_copied_install_matches_the_original_environment(self):
        source = self.module(self.checkout / "src", "first")
        hook = self.site / "first-backend.pth"
        hook.write_text(str(source.parent), encoding="utf-8")
        self.dist("first", self.checkout, hook.name)
        copied = self.root / "renamed checkout"
        shutil.copytree(self.checkout, copied)
        env = copied / ".lp/python"
        self.normalize(env)
        self.assertEqual(self.probe(env / "Lib/site-packages", ["first"]), [copied / "src/first/__init__.py"])
        self.assertEqual(hook.read_text(encoding="utf-8"), str(source.parent))

    def test_recovers_old_engine_layout_only_with_verified_top_level_modules(self):
        module = self.module(self.checkout, "library")
        self.dist("library", self.checkout / "missing-old-subdirectory", "__editable__.library-1.pth")
        (self.site / "library-1.0.dist-info/top_level.txt").write_text("library\n")
        self.normalize()
        self.assertEqual(self.probe(self.site, ["library"]), [module / "__init__.py"])

    def test_full_path_from_native_file_url_and_unlisted_third_party_pth(self):
        direct = self.site / "direct_url.json"
        direct.write_text(json.dumps({"dir_info": {"editable": True}, "url": self.checkout.as_uri()}), encoding="utf-8")
        self.assertEqual(editable._editable_source(direct), self.checkout)
        legacy = self.site / "__editable__.legacy-1.pth"
        legacy.write_text(str(self.checkout) + "\n", encoding="utf-8")
        third = self.site / "unowned.pth"
        third.write_text(str(self.checkout) + "\n", encoding="utf-8")
        content = third.read_bytes()
        self.normalize()
        self.assertEqual(third.read_bytes(), content)
        self.assertEqual((self.site / legacy.read_text(encoding="utf-8").strip()).resolve(), self.checkout)
        self.assertNotIn(str(self.checkout), legacy.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
