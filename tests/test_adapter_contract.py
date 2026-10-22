"""Adapter invariants the shell and the UI rely on.

These are the cheap checks that previously needed manual inspection: a duplicated
label inside one form (two lists rendered together), a parameter registered for an
architecture the adapter does not declare, or an architecture without a training
entry. Every check runs against the real engine checkouts when they are present.
"""

import ast
import re
import unittest
from collections import Counter
from pathlib import Path

from adapters.registry import ADAPTERS
from app.schemas.engine import EngineInstallation, EngineRevision, ManagementMode

# Engine checkouts in this workspace; missing ones are skipped, never failed.
SOURCES = {
    "kohya": "engine/sd-scripts",
    "musubi_tuner": "engine/musubi-tuner",
    "ai_toolkit": "engine/ai-toolkit",
    "anima_lora": "engine/anima_lora",
}


def installation_for(engine_id, source):
    return EngineInstallation(
        installation_id=f"test-{engine_id}",
        engine_id=engine_id,
        source_path=source,
        management_mode=ManagementMode.USER_MANAGED,
        revision=EngineRevision(),
    )


class AdapterContractTests(unittest.TestCase):
    def cases(self):
        """(engine_id, adapter, source, capabilities) for every usable checkout."""
        for engine_id, adapter in sorted(ADAPTERS.items()):
            source = Path(SOURCES.get(engine_id, ""))
            if not SOURCES.get(engine_id) or not source.is_dir():
                continue
            yield engine_id, adapter, source, adapter.capabilities(installation_for(engine_id, source))

    def test_every_engine_has_at_least_one_architecture(self):
        for engine_id, adapter, source, capabilities in self.cases():
            with self.subTest(engine=engine_id):
                self.assertTrue(capabilities["architectures"], f"{engine_id} 未声明任何底模类型")

    def test_architectures_declare_a_training_entry(self):
        for engine_id, adapter, source, capabilities in self.cases():
            declared = {a.id: a for a in adapter.architectures(source)}
            for architecture in capabilities["architectures"]:
                with self.subTest(engine=engine_id, architecture=architecture):
                    self.assertTrue(declared[architecture].script, "架构未声明训练入口脚本")

    def test_base_model_optional_architectures_reach_the_ui(self):
        """An engine that trains from its own model directory declares it here.

        The form hides the generic 底模文件 field only from this capability; without
        it the page shows two model inputs (通用底模 + 引擎模型目录) that can disagree.
        """
        for engine_id, adapter, source, capabilities in self.cases():
            expected = set(getattr(adapter, "base_model_optional_architectures", ()))
            declared = set(capabilities.get("base_model_optional_architectures") or ())
            with self.subTest(engine=engine_id):
                self.assertEqual(expected, declared, f"{engine_id} 未把独立模型目录的底模类型传给界面")
                self.assertLessEqual(declared, set(capabilities["architectures"]))

    def test_parameter_architectures_are_declared(self):
        for engine_id, adapter, source, capabilities in self.cases():
            known = set(capabilities["architectures"])
            for spec in capabilities["params"]:
                for architecture in spec.get("architectures", []):
                    with self.subTest(engine=engine_id, key=spec["key"]):
                        self.assertIn(architecture, known, f"{spec['key']} 指向了未声明的底模类型")

    def test_parameter_keys_are_unique_per_engine(self):
        for engine_id, adapter, source, capabilities in self.cases():
            keys = [spec["key"] for spec in capabilities["params"]]
            with self.subTest(engine=engine_id):
                duplicates = [key for key, count in Counter(keys).items() if count > 1]
                self.assertEqual([], duplicates, f"{engine_id} 参数键重复")

    def test_labels_are_unique_within_one_architecture(self):
        """Two lists rendered together show up as duplicate labels in one form."""
        for engine_id, adapter, source, capabilities in self.cases():
            for architecture in capabilities["architectures"]:
                visible = [
                    spec for spec in capabilities["params"]
                    if not spec.get("architectures") or architecture in spec["architectures"]
                ]
                labels = [spec["label"] for spec in visible]
                with self.subTest(engine=engine_id, architecture=architecture):
                    duplicates = sorted({label for label, count in Counter(labels).items() if count > 1})
                    self.assertEqual([], duplicates, f"{engine_id}/{architecture} 参数标签重复")

    def test_anima_lora_native_keys_exist_in_the_engine(self):
        """Every native key must be a flag/field the engine itself defines.

        MOSS parameters are dataclass fields in ``library/moss/requests.py``; the
        Anima line's flags are declared somewhere under the engine tree. A key that
        exists nowhere would be silently ignored by the engine at run time.
        """
        source = Path(SOURCES["anima_lora"])
        if not source.is_dir():
            self.skipTest("anima_lora 源码不存在")
        requests_text = (source / "library" / "moss" / "requests.py").read_text(encoding="utf-8")
        moss = {
            target.target.id
            for node in ast.parse(requests_text).body
            if isinstance(node, ast.ClassDef)
            for target in node.body
            if isinstance(target, ast.AnnAssign)
        }
        declared = set()
        for path in source.rglob("*.py"):
            if ".venv" in path.parts or ".lp" in path.parts:
                continue
            declared.update(re.findall(r"""['"]--([a-zA-Z0-9_]+)['"]""", path.read_text(encoding="utf-8", errors="ignore")))
        adapter = ADAPTERS["anima_lora"]
        capabilities = adapter.capabilities(installation_for("anima_lora", source))
        for spec in capabilities["params"]:
            if not spec["native"]:
                continue
            key = spec.get("native_key") or spec["key"]
            with self.subTest(key=spec["key"]):
                if spec.get("architectures") == ["moss_image"]:
                    self.assertIn(key, moss, f"MOSS 参数 {key} 不是引擎字段")
                else:
                    self.assertIn(key, declared, f"Anima 参数 {key} 不是引擎声明的开关")


if __name__ == "__main__":
    unittest.main()
