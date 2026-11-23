"""Follow the native file/CLI each runnable adapter writes to its weight path."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest

from adapters.registry import ADAPTERS


def assignments(path):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.Assign)}


class BoundOutputTests(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        self.root = Path(sandbox.name).resolve()
        self.installation = SimpleNamespace(source_path=self.root / "source",
                                            python_executable=self.root / "engine-python")
        self.dataset = self.root / "pictures"
        self.dataset.mkdir()

    def config(self, architecture, serial, params=None):
        return dict(name="same-name", output_name="weights", architecture=architecture,
                    output_layout="task_id", dataset_path=str(self.dataset),
                    base_model_path=str(self.root / "input.safetensors"),
                    output_dir=str(self.root / "outputs" / ("same-name--" + serial * 32)),
                    params=params or {})

    def weight_path(self, engine_id, cfg, job):
        adapter = ADAPTERS[engine_id]
        job.mkdir(parents=True)
        native = adapter.write_native_config(self.installation, cfg, job)
        # collect_artifacts / resume_sources read the frozen job, as with older tasks.
        (job / "job.json").write_text(json.dumps(cfg), encoding="utf-8")
        if engine_id == "kohaku_moss_trainer":
            knobs = assignments(native)
            directory = Path(knobs["OUTPUT_DIR"]) / knobs["RUN_NAME"]
            self.assertEqual(Path(knobs["CACHE_DIR"]), Path(cfg["output_dir"]) / "cache")
            return directory / "export" / (knobs["RUN_NAME"] + ".safetensors")
        if engine_id == "anima_lora" and cfg["architecture"] == "moss_image":
            command = adapter.build_launch(self.installation, native).argv
            return Path(command[command.index("--output") + 1])
        knobs = tomllib.loads(native.read_text(encoding="utf-8"))
        return Path(knobs["output_dir"]) / (knobs["output_name"] + ".safetensors")

    def test_native_output_of_each_pipeline_isolates_repeated_names(self):
        cases = (("kohya", "sd1"), ("musubi_tuner", "wan"), ("anima_lora", "anima"),
                 ("anima_lora", "moss_image"), ("kohaku_moss_trainer", "moss_image"))
        for engine_id, architecture in cases:
            with self.subTest(engine=engine_id, arch=architecture):
                jobs = self.root / "jobs" / engine_id / architecture
                former = None
                for serial in ("a", "b"):
                    cfg = self.config(architecture, serial)
                    cfg["output_dir"] = str(self.root / engine_id / architecture / Path(cfg["output_dir"]).name)
                    job = jobs / serial
                    target = self.weight_path(engine_id, cfg, job)
                    self.assertTrue(target.is_relative_to(Path(cfg["output_dir"])), target)
                    self.assertEqual(target.name, "weights.safetensors")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(serial.encode() * 1024)
                    self.assertEqual([a.path for a in ADAPTERS[engine_id].collect_artifacts(job)
                                      if a.kind == "checkpoint"], [target])
                    if former is None:
                        former = target
                        original = hashlib.sha256(target.read_bytes()).digest()
                    else:
                        self.assertNotEqual(former, target)
                        self.assertEqual(hashlib.sha256(former.read_bytes()).digest(), original)

    def test_path_like_names_remain_one_filename_in_the_bound_job(self):
        for i, (key, arch) in enumerate((("kohya", "sd1"), ("anima_lora", "anima"),
                ("musubi_tuner", "wan"), ("anima_lora", "moss_image"), ("kohaku_moss_trainer", "moss_image"))):
            with self.subTest(engine=key, arch=arch):
                cfg = self.config(arch, "a")
                cfg["output_name"] = r"..\..\former/weights"
                if key == "kohaku_moss_trainer":
                    cfg["params"]["RUN_NAME"] = str(self.root / "external" / "bad")
                dest = self.weight_path(key, cfg, self.root / "jobs" / str(i))
                self.assertTrue(dest.resolve().is_relative_to(Path(cfg["output_dir"])), dest)
                self.assertNotIn("..", dest.relative_to(cfg["output_dir"]).parts)

    def test_managed_configs_reject_secondary_output_overrides(self):
        for i, (key, arch, extra) in enumerate((
                ("kohya", "sdxl", "--output_dir ../former"),
                ("musubi_tuner", "wan", "--output_name ../former"),
                ("anima_lora", "anima", "--dataset_config elsewhere.toml"),
                ("kohaku_moss_trainer", "moss_image", 'OUTPUT_DIR = "../former"'),
                ("kohaku_moss_trainer", "moss_image", 'RUN_NAME = "../former"'),
                ("kohaku_moss_trainer", "moss_image", 'CACHE_DIR = "../former"'))):
            with self.subTest(engine=key, extra=extra):
                cfg = self.config(arch, "a", {"extra_params": extra})
                with self.assertRaisesRegex(ValueError, "自定义参数"):
                    self.weight_path(key, cfg, self.root / "jobs" / str(i))
                self.assertTrue(any(v.field == "extra_params" and v.severity == "error" for v in
                                    ADAPTERS[key].validate(self.installation, cfg)))

    def test_kohaku_legacy_layout_and_shared_cache_are_readable(self):
        adapter = ADAPTERS["kohaku_moss_trainer"]
        config = self.config("moss_image", "a", {"CACHE_DIR": str(self.root / "shared")})
        self.assertEqual(adapter.custom_write_dirs(config), {"CACHE_DIR": str(self.root / "shared")})
        native = self.root / "bound.py"
        native.write_text(adapter.native_config_text(config, None), encoding="utf-8")
        self.assertEqual(assignments(native)["CACHE_DIR"], str(self.root / "shared"))
        legacy = dict(config, name="legacy", output_name="", output_dir=str(self.root / "old" / "legacy"))
        del legacy["output_layout"]
        native.write_text(adapter.native_config_text(legacy, None), encoding="utf-8")
        knobs = assignments(native)
        self.assertEqual(Path(knobs["OUTPUT_DIR"]) / knobs["RUN_NAME"], Path(legacy["output_dir"]))
        job = self.root / "old-job"
        job.mkdir()
        (job / "job.json").write_text(json.dumps(legacy), encoding="utf-8")
        old_checkpoint = Path(legacy["output_dir"]) / "checkpoints" / "step_00000011"
        old_checkpoint.mkdir(parents=True)
        (old_checkpoint / "state.json").write_text('{"step":11}', encoding="utf-8")
        self.assertEqual([source.path for source in adapter.resume_sources(job)], [old_checkpoint])

    def test_preflight_refuses_legacy_redirects_but_admits_normal_queued_jobs(self):
        for engine, arch in (("kohya", "sd1"), ("anima_lora", "anima"), ("musubi_tuner", "wan")):
            with self.subTest(engine=engine):
                cfg = self.config(arch, "a")
                cfg.pop("output_layout")
                adapter = ADAPTERS[engine]
                self.assertIsNone(adapter.legacy_output_issue(cfg))
                for extra in ("--output_dir ../other", "--output_name ../history/model",
                              "--config_file other.toml"):
                    self.assertTrue(adapter.legacy_output_issue(dict(cfg, params={"extra_params": extra})))
                with_params = dict(cfg, params={"extra_params": "--learning_rate 0.002"})
                self.assertIsNone(adapter.legacy_output_issue(with_params))
        sd = ADAPTERS["kohya"]
        self.assertTrue(sd.legacy_output_issue(dict(cfg, output_name=r"..\old\model")))
        kohaku = ADAPTERS["kohaku_moss_trainer"]
        cfg = dict(cfg, output_dir=str(self.root / "outputs" / "probe"), name="probe", output_name="")
        self.assertIsNone(kohaku.legacy_output_issue(cfg))
        self.assertTrue(kohaku.legacy_output_issue(dict(cfg, params={"RUN_NAME": "old-run"})))
        self.assertTrue(kohaku.legacy_output_issue(dict(cfg, params={"extra_params": 'OUTPUT_DIR = "old"'})))


if __name__ == "__main__":
    unittest.main()
