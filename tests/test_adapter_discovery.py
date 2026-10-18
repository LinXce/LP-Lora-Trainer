from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from adapters.registry import ADAPTERS, ADAPTER_LOAD_ERRORS, discover_adapters


class AdapterDiscoveryTests(unittest.TestCase):
    def test_builtin_adapters_are_discovered_without_errors(self):
        self.assertEqual(set(ADAPTERS), {"kohya", "ai_toolkit", "musubi_tuner", "anima_lora"})
        self.assertEqual([], ADAPTER_LOAD_ERRORS)

    def test_new_local_adapter_is_discovered_and_bad_plugins_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "pluginpkg"
            root.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            good = root / "good"
            good.mkdir()
            (good / "__init__.py").write_text("", encoding="utf-8")
            (good / "adapter.py").write_text(textwrap.dedent("""
                from adapters.musubi_tuner.adapter import MusubiTunerAdapter
                class FourthTrainer(MusubiTunerAdapter):
                    engine_id = "fourth_trainer"
                    display_name = "Fourth Trainer"
            """), encoding="utf-8")
            broken = root / "broken"
            broken.mkdir()
            (broken / "__init__.py").write_text("", encoding="utf-8")
            (broken / "adapter.py").write_text("raise RuntimeError('broken plugin')\n", encoding="utf-8")
            ignored = root / "not_an_adapter"
            ignored.mkdir()
            (ignored / "README.md").write_text("ignored", encoding="utf-8")
            sys.path.insert(0, str(root.parent))
            try:
                found, errors = discover_adapters(root, "pluginpkg")
            finally:
                sys.path.remove(str(root.parent))
                for name in list(sys.modules):
                    if name == "pluginpkg" or name.startswith("pluginpkg."):
                        del sys.modules[name]
            self.assertEqual(found["fourth_trainer"].display_name, "Fourth Trainer")
            self.assertEqual([e["directory"] for e in errors], ["broken"])

    def test_duplicate_ids_disable_both_plugins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dupes"
            root.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            for folder, label in (("first", "First"), ("second", "Second")):
                directory = root / folder
                directory.mkdir()
                (directory / "__init__.py").write_text("", encoding="utf-8")
                (directory / "adapter.py").write_text(textwrap.dedent(f"""
                    from adapters.musubi_tuner.adapter import MusubiTunerAdapter
                    class Plugin(MusubiTunerAdapter):
                        engine_id = "same_engine"
                        display_name = "{label}"
                """), encoding="utf-8")
            sys.path.insert(0, str(root.parent))
            try:
                found, errors = discover_adapters(root, "dupes")
            finally:
                sys.path.remove(str(root.parent))
                for name in list(sys.modules):
                    if name == "dupes" or name.startswith("dupes."):
                        del sys.modules[name]
            self.assertNotIn("same_engine", found)
            self.assertEqual(len(errors), 1)
            self.assertIn("重复 engine_id", errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
