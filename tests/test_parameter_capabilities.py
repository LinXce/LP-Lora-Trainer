"""Regression coverage for parameter UI metadata without importing engines."""
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.capabilities import Param
from adapters.musubi_tuner.adapter import MusubiTunerAdapter
from adapters.musubi_tuner.params import ARCHITECTURES, PARAMS as MUSUBI_PARAMS
from adapters.musubi_tuner.pipeline import PIPELINES
from adapters.sd_scripts.params import PARAMS as KOHYA_PARAMS
from adapters.ai_toolkit.params import PARAMS as AI_PARAMS


class ParameterCapabilityTests(unittest.TestCase):
    def test_metadata_preserves_positional_arguments(self):
        p = Param("vae", "VAE", "string", "", "advanced", "模型与编码器",
                  groups=("basic", "native"), display_order=0, required_for=("wan",))
        spec = p.spec()
        self.assertEqual(spec["group"], "advanced")
        self.assertEqual(spec["groups"], ["basic", "native"])
        self.assertEqual(spec["required_for"], ["wan"])
        self.assertEqual(spec["display_order"], 0)
        self.assertTrue(spec["native"])
        self.assertEqual(p.default, "")
        self.assertNotIn("block", spec)
        nested = Param("vae_path", "VAE", "string", block="model").spec()
        self.assertEqual(nested["block"], "model")
        self.assertFalse(Param("cache", "Cache", "string", native=False).spec()["native"])

    def capabilities_for(self, architectures):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for architecture in ARCHITECTURES:
                if architecture.id in architectures:
                    (root / architecture.script).touch()
            installation = SimpleNamespace(source_path=root, installation_id="test")
            return MusubiTunerAdapter().capabilities(installation)

    def test_required_metadata_matches_every_pipeline(self):
        caps = self.capabilities_for(set(PIPELINES))
        specs = {p["key"]: p for p in caps["params"]}
        for architecture, pipeline in PIPELINES.items():
            with self.subTest(architecture=architecture):
                required = {key for key, p in specs.items()
                            if architecture in p.get("required_for", [])}
                self.assertEqual(required, set(pipeline.required))
        self.assertIn("wan", specs["vae"]["required_for"])
        self.assertIn("krea2", specs["vae"]["required_for"])
        self.assertNotIn("minimax_h3", specs["vae"]["required_for"])

    def test_required_metadata_only_uses_available_architectures(self):
        caps = self.capabilities_for({"wan"})
        specs = {p["key"]: p for p in caps["params"]}
        self.assertEqual(caps["architectures"], ["wan"])
        self.assertEqual(specs["vae"]["required_for"], ["wan"])
        self.assertEqual(specs["wan_t5"]["required_for"], ["wan"])
        self.assertNotIn("required_for", specs["krea2_text_encoder"])

    def test_model_parameters_shared_without_duplicate_keys(self):
        for params, key in ((MUSUBI_PARAMS, "vae"), (MUSUBI_PARAMS, "wan_t5"),
                            (KOHYA_PARAMS, "vae"), (AI_PARAMS, "vae_path")):
            with self.subTest(key=key):
                self.assertEqual(len(params), len({p.key for p in params}))
                p = next(p for p in params if p.key == key)
                self.assertTrue({"basic", "native"}.issubset({p.group, *p.groups}))
                self.assertLess(p.display_order, 100)
                self.assertTrue(p.help)

    def test_all_parameters_have_readable_sections(self):
        for params in (MUSUBI_PARAMS, KOHYA_PARAMS, AI_PARAMS):
            for p in params:
                with self.subTest(key=p.key):
                    self.assertTrue(p.section.strip())
                    self.assertNotIn("?", p.section)
                    self.assertNotIn("\ufffd", p.section + p.help)

    def test_preview_parameters_have_their_own_tab(self):
        engines = (
            (AI_PARAMS, {"sample_every", "sample_start_step", "sample_steps", "sample_prompts", "guidance_scale", "width", "height", "sampler"}),
            (KOHYA_PARAMS, {"sample_every_n_steps", "sample_every_n_epochs", "sample_prompts", "sample_sampler"}),
            (MUSUBI_PARAMS, {"sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts"}),
        )
        for params, keys in engines:
            by_key = {p.key: p for p in params}
            for key in keys:
                with self.subTest(key=key):
                    self.assertEqual(by_key[key].spec()["group"], "preview")
                    self.assertNotIn("advanced", by_key[key].groups)
            self.assertEqual({p.key for p in params if p.group == "preview"}, keys)
        # Musubi guidance_scale also affects training, so is not preview-only.
        self.assertEqual(next(p for p in MUSUBI_PARAMS if p.key == "guidance_scale").group, "advanced")

    def test_bucket_presets_are_metadata_for_the_shared_parameter(self):
        p = next(p for p in MUSUBI_PARAMS if p.key == "bucket_resolutions")
        presets = [512, 640, 768, 896, 1024, 1280, 1536, 2048]
        self.assertEqual(p.spec()["resolution_presets"], presets)
        self.assertFalse(p.native)
        self.assertTrue(p.multiline)
        self.assertNotIn("resolution_presets", Param("test", "Test", "string").spec())
        values = "\n".join(f"{size}x{size}" for size in presets)
        self.assertEqual(MusubiTunerAdapter.parse_resolutions(values), [(size, size) for size in presets])
        config = {"dataset_path": "dataset", "params": {"bucket_resolutions": values + "\n960x544"}}
        dataset = tomllib.loads(MusubiTunerAdapter()._dataset_toml(config, Path("cache")))
        self.assertEqual(dataset["general"]["resolution"], [presets[0], presets[0]])
        self.assertNotIn("resolution_presets", dataset["general"])
        expected = [[size, size] for size in presets] + [[960, 544]]
        self.assertEqual([entry["resolution"] for entry in dataset["datasets"]], expected)
        caches = [entry["cache_directory"] for entry in dataset["datasets"]]
        self.assertEqual(len(caches), len(set(caches)))
        for entry, (width, height) in zip(dataset["datasets"], expected):
            self.assertEqual(Path(entry["cache_directory"]), Path("cache") / f"{width}x{height}")
            self.assertEqual(entry["image_directory"], "dataset")

    def test_empty_bucket_uses_explicit_resolution(self):
        config = {
            "dataset_path": "dataset",
            "params": {
                "resolution_width": 960,
                "resolution_height": 544,
                "bucket_resolutions": "",
            },
        }
        dataset = tomllib.loads(MusubiTunerAdapter()._dataset_toml(config, Path("cache")))
        self.assertEqual(dataset["general"]["resolution"], [960, 544])

    def test_single_resolution_keeps_cache_path_and_deduplicates(self):
        for value in ("1024x1024", "1024x1024\n1024 1024", "960x544"):
            with self.subTest(value=value):
                config = {"dataset_path": "dataset", "params": {"bucket_resolutions": value}}
                dataset = tomllib.loads(MusubiTunerAdapter()._dataset_toml(config, Path("custom-cache")))
                self.assertEqual(len(dataset["datasets"]), 1)
                self.assertEqual(Path(dataset["datasets"][0]["cache_directory"]), Path("custom-cache"))
                self.assertEqual(dataset["general"]["resolution"], dataset["datasets"][0]["resolution"])
                self.assertTrue(all(type(size) is int for size in dataset["general"]["resolution"]))

    def test_shared_ui_metadata_does_not_duplicate_native_output(self):
        text = MusubiTunerAdapter()._render_train({
            "architecture": "wan", "base_model_path": "dit.safetensors", "output_dir": "output",
            "params": {"vae": "vae.safetensors", "wan_t5": "t5.safetensors"},
        })
        values = tomllib.loads(text)
        self.assertEqual(values["vae"], "vae.safetensors")
        self.assertEqual(values["t5"], "t5.safetensors")
        self.assertEqual(text.count("\nvae = "), 1)
        self.assertNotIn("groups", values)
        self.assertNotIn("display_order", values)
        self.assertNotIn("required_for", values)


if __name__ == "__main__":
    unittest.main()

