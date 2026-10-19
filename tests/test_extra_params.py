"""Supplementary engine arguments: parsing, duplicate detection, merging and wiring."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters import extra_params
from adapters.sd_scripts.adapter import KohyaAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter


class ParseTests(unittest.TestCase):
    def test_values_are_typed(self):
        self.assertEqual(extra_params.parse("--network_dim 32 --network_alpha 16"),
                         {"network_dim": 32, "network_alpha": 16})
        self.assertEqual(extra_params.parse("--learning_rate 1e-5 --noise_offset 0.05"),
                         {"learning_rate": 1e-5, "noise_offset": 0.05})

    def test_bare_flag_switches_on(self):
        self.assertEqual(extra_params.parse("--cache_latents"), {"cache_latents": True})
        self.assertEqual(extra_params.parse("--cache_latents --sdpa"),
                         {"cache_latents": True, "sdpa": True})

    def test_quotes_group_a_value_and_windows_paths_survive(self):
        self.assertEqual(extra_params.parse('--network_args "conv_dim=8 conv_alpha=8"'),
                         {"network_args": "conv_dim=8 conv_alpha=8"})
        self.assertEqual(extra_params.parse(r"--vae H:\models\vae.safetensors"),
                         {"vae": r"H:\models\vae.safetensors"})

    def test_quoted_numbers_stay_text(self):
        self.assertEqual(extra_params.parse('--output_name "123"'), {"output_name": "123"})
        self.assertEqual(extra_params.parse("--output_name 123"), {"output_name": 123})

    def test_negative_numbers_are_values_not_flags(self):
        self.assertEqual(extra_params.parse("--min_timestep -1"), {"min_timestep": -1})

    def test_broken_syntax_is_reported(self):
        with self.assertRaisesRegex(ValueError, "引号"):
            extra_params.parse('--network_args "conv_dim=8')
        with self.assertRaisesRegex(ValueError, "缺少参数名"):
            extra_params.parse("--network_dim 32 stray")

    def test_duplicates_are_reported(self):
        self.assertEqual(extra_params.duplicates("--sdpa --sdpa --cache_latents"), ["sdpa"])
        self.assertEqual(extra_params.duplicates("--sdpa --cache_latents"), [])


class IssueTests(unittest.TestCase):
    def test_clean_text_has_no_problems(self):
        self.assertEqual(extra_params.problems("--sdpa --network_dim 32", ["learning_rate"]), [])

    def test_empty_text_has_no_problems(self):
        self.assertEqual(extra_params.problems("  \n ", ["learning_rate"]), [])

    def test_conflict_with_a_provided_key_is_a_warning(self):
        problems = extra_params.problems("--learning_rate 1e-5", ["learning_rate", "output_dir"])
        self.assertEqual([severity for severity, _ in problems], ["warning"])
        self.assertIn("--learning_rate", problems[0][1])
        self.assertIn("会覆盖上面已设置的字段", problems[0][1])

    def test_repeated_key_is_an_error(self):
        problems = extra_params.problems("--sdpa --sdpa", [])
        self.assertEqual([severity for severity, _ in problems], ["error"])
        self.assertIn("给了多次", problems[0][1])

    def test_unparseable_text_is_an_error(self):
        problems = extra_params.problems('--network_args "conv_dim=8', [])
        self.assertEqual([severity for severity, _ in problems], ["error"])

    def test_merge_only_extends_the_config(self):
        self.assertEqual(extra_params.merge({"a": 1}, "--b 2"), {"a": 1, "b": 2})
        self.assertEqual(extra_params.merge({"a": 1}, ""), {"a": 1})

    def test_provided_keys_follow_the_engine_default_rule(self):
        class Param:
            def __init__(self, key, default, native=True):
                self.key, self.default, self.native, self.native_key = key, default, native, None

            def applies_to(self, architecture):
                return True

        params = [Param("learning_rate", 1e-4), Param("keep_tokens", None), Param("display_only", 1, native=False)]
        keys = extra_params.provided_keys(params, "sd1", {}, skip_false=True, extra=("output_dir",))
        self.assertEqual(keys, {"learning_rate", "output_dir"})

    def test_provided_keys_can_keep_false_values(self):
        class Param:
            def __init__(self, key, default):
                self.key, self.default, self.native, self.native_key = key, default, True, None

            def applies_to(self, architecture):
                return True

        keys = extra_params.provided_keys([Param("full_bf16", False)], "sd1", {}, skip_false=False)
        self.assertEqual(keys, {"full_bf16"})


class AdapterWiringTests(unittest.TestCase):
    def test_kohya_config_carries_the_extra_keys(self):
        config = {"architecture": "sd1", "base_model_path": "model.safetensors", "output_dir": "output",
                  "params": {"extra_params": "--alpha_mask --huber_scale 1.0"}}
        text = KohyaAdapter().native_config_text(config, Path("dataset.toml"))
        self.assertIn("alpha_mask = true", text)
        self.assertIn("huber_scale = 1.0", text)
        self.assertNotIn("extra_params", text)

    def test_musubi_config_carries_the_extra_keys(self):
        config = {"architecture": "wan", "base_model_path": "dit.safetensors", "output_dir": "output",
                  "params": {"extra_params": "--offload_dit_during_sampling"}}
        text = MusubiTunerAdapter().native_config_text(config, Path("dataset.toml"))
        self.assertIn("offload_dit_during_sampling = true", text)
        self.assertNotIn("extra_params", text)

    def test_kohya_validate_warns_about_overriding_its_own_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "train_network.py").touch()
            installation = SimpleNamespace(source_path=source)
            config = {"architecture": "sd1",
                      "params": {"learning_rate": 1e-4, "extra_params": "--learning_rate 5e-5"}}
            warnings = [i for i in KohyaAdapter().validate(installation, config) if i.field == "extra_params"]
        self.assertEqual([i.severity for i in warnings], ["warning"])
        self.assertIn("会覆盖上面已设置的字段", warnings[0].message)

    def test_kohya_validate_rejects_broken_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "train_network.py").touch()
            installation = SimpleNamespace(source_path=source)
            config = {"architecture": "sd1", "params": {"extra_params": '--network_args "conv_dim=8'}}
            problems = [i for i in KohyaAdapter().validate(installation, config) if i.field == "extra_params"]
        self.assertEqual([i.severity for i in problems], ["error"])
        self.assertIn("引号", problems[0].message)


if __name__ == "__main__":
    unittest.main()
