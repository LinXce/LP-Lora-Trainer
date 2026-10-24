"""sd-scripts architectures that need extra frozen model components."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.sd_scripts.adapter import KohyaAdapter


class SdModelInputTests(unittest.TestCase):
    def test_flux_lumina_hunyuan_require_their_extra_encoder(self):
        adapter = KohyaAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.safetensors"
            model.write_bytes(b"")
            installation = SimpleNamespace(source_path=str(root), python_executable="python")
            for architecture, expected in (
                ("flux1", "ae"), ("lumina", "gemma2"), ("hunyuan_image", "byt5"),
            ):
                config = {
                    "architecture": architecture,
                    "base_model_path": str(model),
                    "dataset_path": str(root),
                    "output_dir": str(root / "out"),
                    "name": "probe",
                    "params": {},
                }
                fields = {issue.field for issue in adapter.validate(installation, config)
                          if issue.severity == "error"}
                self.assertIn(expected, fields, architecture)

    def test_new_component_paths_are_written_natively(self):
        adapter = KohyaAdapter()
        config = {
            "architecture": "lumina",
            "base_model_path": "model.safetensors",
            "dataset_path": "data",
            "output_dir": "out",
            "name": "probe",
            "params": {"ae": "vae.safetensors", "gemma2": "gemma2"},
        }
        text = adapter.native_config_text(config, Path("dataset.toml"))
        self.assertIn('ae = "vae.safetensors"', text)
        self.assertIn('gemma2 = "gemma2"', text)


if __name__ == "__main__":
    unittest.main()
