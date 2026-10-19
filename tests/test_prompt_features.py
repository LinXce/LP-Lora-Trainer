import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

from adapters.sd_scripts.adapter import KohyaAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter
from adapters.ai_toolkit.adapter import AiToolkitAdapter
from adapters.sampling import prompt_text
from app.api.server import create_app
from app.config import AppPaths
from app.storage.state import StateStore
from tests.local_client import LocalClient


class PromptFeatureTests(unittest.TestCase):
    def test_inline_prompt_is_written_only_for_native_file_adapters(self):
        for adapter, architecture in ((KohyaAdapter(), "sdxl"), (MusubiTunerAdapter(), "krea2")):
            with self.subTest(adapter=adapter.engine_id), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                config = {"architecture": architecture, "base_model_path": "model.safetensors",
                          "dataset_path": "dataset", "output_dir": "output",
                          "params": {"sample_prompts": "one prompt\nsecond prompt"}}
                before = {**config, "params": dict(config["params"])}
                source = directory / "engine"
                source.mkdir()
                if adapter.engine_id == "musubi_tuner":
                    scripts = source / "src" / "musubi_tuner"
                    scripts.mkdir(parents=True)
                    for name in ("krea2_cache_latents.py", "krea2_cache_text_encoder_outputs.py", "krea2_train_network.py"):
                        (scripts / name).touch()
                installation = SimpleNamespace(source_path=source, python_executable=Path("python"))
                target = adapter.write_native_config(installation, config, directory)
                self.assertTrue(target.is_file())
                self.assertEqual((directory / "sample_prompts.txt").read_text(encoding="utf-8"), "one prompt\nsecond prompt\n")
                self.assertEqual(prompt_text(config), "one prompt\nsecond prompt")
                self.assertEqual(config, before)

    def test_ai_toolkit_receives_prompt_lines_without_a_path(self):
        config = {"architecture": "flux1", "base_model_path": "model.safetensors",
                  "dataset_path": "dataset", "output_dir": "output",
                  "params": {"sample_prompts": "one prompt\n# ignored\nsecond prompt"}}
        output = AiToolkitAdapter().native_config_text(config, None)
        self.assertIn("one prompt", output)
        self.assertIn("second prompt", output)
        self.assertNotIn("sample_prompts.txt", output)

    def test_legacy_prompt_path_is_rejected_by_shared_validation(self):
        with self.assertRaisesRegex(ValueError, "文件路径"):
            prompt_text({"params": {"sample_prompts": "H:/prompts.txt"}})

    def test_prompt_template_api_persists_and_requires_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(AppPaths.for_workspace(root), start_worker=False)
            client = LocalClient(app)
            self.addCleanup(client.close)
            self.assertEqual(client.get("/api/v1/prompt-templates").status_code, 401)
            headers = {"X-LP-Session": app.state.session_token}
            created = client.post("/api/v1/prompt-templates", json={"name": "人物", "prompt": "portrait\nsoft light"}, headers=headers)
            self.assertEqual(created.status_code, 201)
            item = created.json()
            self.assertEqual(client.get("/api/v1/prompt-templates", headers=headers).json()[0]["template_id"], item["template_id"])
            self.assertEqual(client.put(f"/api/v1/prompt-templates/{item['template_id']}", json={"name": "人物2", "prompt": "new"}, headers=headers).status_code, 200)
            self.assertEqual(client.delete(f"/api/v1/prompt-templates/{item['template_id']}", headers=headers).status_code, 204)
            self.assertEqual(client.get(f"/api/v1/prompt-templates/{item['template_id']}", headers=headers).status_code, 404)
            self.assertEqual(client.post("/api/v1/prompt-templates", json={"name": "", "prompt": "x"}, headers=headers).status_code, 422)
            self.assertEqual(client.post("/api/v1/prompt-templates", json={"name": "???", "prompt": "H:/prompts.txt"}, headers=headers).status_code, 422)


if __name__ == "__main__":
    unittest.main()

