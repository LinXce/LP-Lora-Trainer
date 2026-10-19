"""Native parameter profiles are read-only and do not need installed engines."""
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.api.server import create_app
from app.config import AppPaths
from app.services.common import ServiceError
from app.services.training import TrainingService
from tests.local_client import LocalClient


class ParameterExportTests(unittest.TestCase):
    def setUp(self):
        self.engines, self.datasets, self.store = Mock(), Mock(), Mock()
        self.training = TrainingService(self.store, self.engines, self.datasets)
        self.config = dict(engine_id="musubi_tuner", name='角色 "test"', installation_id="removed-engine",
                           architecture="krea2", base_model_path='H:\\模型\\a"b.safetensors',
                           dataset_id="missing", dataset_path="H:/数据集/10_char", output_dir="H:/output",
                           params=dict(network_dim=33, seed=0, gradient_checkpointing=True,
                                       optimizer_args="weight_decay=0.01\nbetas=0.9,0.99"))

    def test_musubi_native_types_lists_paths_and_no_preview_placeholders(self):
        before = json.loads(json.dumps(self.config))
        result = self.training.export_parameters(self.config)
        values = tomllib.loads(result["text"])
        self.assertEqual(result["native_format"], "toml")
        self.assertEqual(values["network_dim"], 33)
        self.assertEqual(values["seed"], 0)
        self.assertIs(values["gradient_checkpointing"], True)
        self.assertEqual(values["optimizer_args"], ["weight_decay=0.01", "betas=0.9,0.99"])
        self.assertEqual(values["dit"], self.config["base_model_path"])
        self.assertEqual(values["output_name"], self.config["name"])
        self.assertEqual(values["output_dir"], "H:/output")
        self.assertNotIn("<task-id>", result["text"])
        self.assertNotIn("dataset_config", values)
        self.assertIn("dataset_config", " ".join(result["warnings"]))
        self.assertEqual(self.config, before)
        self.assertEqual(self.engines.mock_calls, [])
        self.assertEqual(self.datasets.mock_calls, [])
        self.assertEqual(self.store.mock_calls, [])

    def test_kohya_profile_has_native_keys_and_no_dangling_dataset_file(self):
        config = {**self.config, "engine_id": "kohya", "architecture": "sdxl",
                  "params": {"seed": 0, "gradient_checkpointing": False, "network_dim": 24}}
        result = self.training.export_parameters(config)
        values = tomllib.loads(result["text"])
        self.assertEqual(result["engine_name"], "kohya sd-scripts")
        self.assertEqual(values["pretrained_model_name_or_path"], config["base_model_path"])
        self.assertEqual(values["network_dim"], 24)
        self.assertEqual(values["seed"], 0)
        self.assertIs(values["gradient_checkpointing"], False)
        self.assertEqual(values["output_name"], config["name"])
        self.assertNotIn("dataset_config", values)
        self.assertNotIn("<task-id>", result["text"])
        self.assertTrue(result["warnings"])

    def test_ai_toolkit_yaml_has_nested_job_structure_and_quoted_paths(self):
        config = {**self.config, "engine_id": "ai_toolkit", "architecture": "flux1",
                  "params": {"steps": 1200, "gradient_checkpointing": False}}
        result = self.training.export_parameters(config)
        self.assertEqual(result["native_format"], "yaml")
        text = result["text"]
        self.assertTrue(text.startswith('job: "train"\nconfig:\n'))
        self.assertIn('  process:\n    - type: "sd_trainer"', text)
        self.assertIn('      datasets:\n', text)
        self.assertIn('folder_path: ' + json.dumps(config["dataset_path"], ensure_ascii=False), text)
        self.assertIn('name_or_path: ' + json.dumps(config["base_model_path"], ensure_ascii=False), text)
        self.assertIn('steps: 1200', text)
        self.assertIn('gradient_checkpointing: false', text)
        self.assertIn('arch: "flux"', text)
        self.assertNotIn("<task-id>", text)

    def test_unknown_adapter_and_unsupported_architecture_are_explicit_errors(self):
        for config in ({**self.config, "engine_id": "unknown"}, {**self.config, "architecture": "sdxl"}):
            with self.subTest(config=config):
                with self.assertRaises(ServiceError) as caught:
                    self.training.export_parameters(config)
                self.assertEqual(caught.exception.status, 422)

    def test_adapter_declares_export_format_without_frontend_engine_whitelist(self):
        adapter = Mock(engine_id="future_engine", display_name="Future engine", native_format="json")
        adapter.export_parameters.return_value = ('{"native_key": 7}\n', ())
        with patch.dict("app.services.training.ADAPTERS", {"future_engine": adapter}):
            result = self.training.export_parameters({**self.config, "engine_id": "future_engine"})
        self.assertEqual(result["native_format"], "json")
        self.assertEqual(json.loads(result["text"]), {"native_key": 7})

    def test_authenticated_export_api_requires_no_engine_or_dataset_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = AppPaths.for_workspace(root)
            paths.engine_root.mkdir()
            app = create_app(paths, start_worker=False)
            client = LocalClient(app)
            try:
                endpoint = "/api/v1/training/export-parameters"
                self.assertEqual(client.post(endpoint, json=self.config).status_code, 401)
                headers = {"X-LP-Session": app.state.session_token}
                before = set(root.rglob("*"))
                response = client.post(endpoint, json=self.config, headers=headers)
                self.assertEqual(response.status_code, 200, response.content)
                self.assertEqual(tomllib.loads(response.json()["text"])["network_dim"], 33)
                self.assertEqual(before, set(root.rglob("*")))
                self.assertEqual(client.post(endpoint, json={**self.config, "engine_id": "missing"}, headers=headers).status_code, 422)
                self.assertEqual(client.post(endpoint, json={**self.config, "architecture": "invalid"}, headers=headers).status_code, 422)
                self.assertEqual(client.post(endpoint, json={**self.config, "format": "lp-training-parameters"}, headers=headers).status_code, 422)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
