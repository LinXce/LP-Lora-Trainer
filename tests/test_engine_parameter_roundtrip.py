"""Exercise the form's actual HTTP contract against installed checkouts, without a worker."""
import ast
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from PIL import Image

from adapters.registry import ADAPTERS
from app.api.server import create_app
from app.config import AppPaths
from app.services.engines import revision
from tests.local_client import LocalClient


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "anima_lora": ("anima_lora", "anima", "network_dim", 32),
    "kohaku_moss_trainer": ("KohakuMossTrainer", "moss_image", "LR", 0.0001),
    "kohya": ("sd-scripts", "sd1", "network_dim", 4),
    "musubi_tuner": ("musubi-tuner", "krea2", "network_dim", 4),
}


class EngineParameterRoundTrip(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.paths = AppPaths(ROOT, ROOT / "engine", self.root / "data", ROOT / "frontend/dist")
        if not all((self.paths.engine_root / case[0]).is_dir() for case in CASES.values()):
            self.skipTest("requires the four installed upstream checkouts")
        self.app = create_app(self.paths, start_worker=False)
        self.client = LocalClient(self.app)
        self.addCleanup(self.client.close)
        self.headers = {"Origin": self.client.base_url}
        self.client.post("/api/v1/session", headers=self.headers)
        images = self.root / "images"
        images.mkdir()
        Image.new("RGB", (64, 64)).save(images / "one.png")
        (images / "one.txt").write_text("cat", encoding="utf-8")
        dataset = self.app.state.datasets.add(str(images), "test")
        self.dataset = dataset["dataset_id"]
        self.app.state.datasets.scan(self.dataset)
        self.model = self.root / "models" / "model.safetensors"
        self.model.parent.mkdir()
        self.model.touch()
        self.weights = self.root / "weights"
        for name in ("transformer", "vae", "text_encoder"):
            (self.weights / name).mkdir(parents=True)
        for key, (folder, _, _, _) in CASES.items():
            source = self.paths.engine_root / folder
            executable = source / ".lp/python/python.exe"
            if key == "anima_lora" and not executable.is_file():
                self.skipTest("the Anima default attention check requires its installed environment")
            self.app.state.store.put("engine", key, dict(
                engine_id=key, installation_id=key, label=folder, source_path=str(source),
                revision=revision(source), management_mode="user_managed", state="ready",
                verification="verified", environment_id="test", environment_manifest_id="test",
                python_executable=str(executable) if executable.is_file() else sys.executable,
                issues=[], is_default=False, candidate_engines=[key],
            ))

    def selection(self, key):
        _, arch, _, _ = CASES[key]
        caps = self.client.get(f"/api/v1/engines/{key}/capabilities?architecture={arch}").json()
        return {p["key"]: p["default"] for p in caps["params"]}

    def inputs(self, key):
        file = str(self.model)
        return {
            "anima_lora": dict(pretrained_model_name_or_path=file, vae=file, qwen3=file),
            "kohaku_moss_trainer": dict(WEIGHTS=str(self.weights)),
            "kohya": dict(resolution=512),
            "musubi_tuner": dict(vae=file, krea2_text_encoder=file, optimizer_type="AdamW", sdpa=True, caption_extension=".txt"),
        }[key]

    def draft(self, key):
        return dict(name=key, installation_id=key, architecture=CASES[key][1],
                    dataset_id=self.dataset, base_model_path=str(self.model),
                    output_dir=str(self.root / "out"), params=self.selection(key) | self.inputs(key))

    def test_form_preview_matches_queued_native_file_and_commands(self):
        for key, (_, _, knob, expected) in CASES.items():
            with self.subTest(engine=key):
                draft = self.draft(key)
                self.assertEqual(draft["params"][knob], expected)
                checked = self.client.post("/api/v1/training/validate", json=draft, headers=self.headers).json()
                self.assertTrue(checked.get("ok"), checked.get("issues", checked))
                response = self.client.post("/api/v1/training/submit", json=draft, headers=self.headers)
                self.assertLess(response.status_code, 300, response.content)
                task = response.json()
                job = self.paths.data_root / "jobs" / task["task_id"]
                stored = json.loads((job / "job.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["params"], draft["params"])
                actual = (job / ADAPTERS[key].native_filename).read_text(encoding="utf-8")
                def normalize(text):
                    return text.replace(str(job).replace("\\", "\\\\"), "<task-id>").replace(
                        str(job), "<task-id>").replace(job.as_posix(), "<task-id>").replace(task["task_id"], "<task-id>")
                self.assertEqual(normalize(actual), checked["native_config"])
                commands = json.loads((job / "command.json").read_text(encoding="utf-8"))["commands"]
                self.assertEqual([[normalize(arg) for arg in stage] for stage in commands], checked["pipeline"])
                if key == "kohaku_moss_trainer":
                    assignments = {s.targets[0].id: ast.literal_eval(s.value)
                                   for s in ast.parse(actual).body if isinstance(s, ast.Assign)}
                    self.assertEqual(assignments[knob], expected)
                else:
                    self.assertEqual(tomllib.loads(actual)[knob], expected)

    def test_invalid_form_values_never_enter_queue(self):
        for key, (_, _, knob, _) in CASES.items():
            with self.subTest(engine=key):
                draft = self.draft(key)
                draft["params"][knob] = "bad-number"
                check = self.client.post("/api/v1/training/validate", json=draft, headers=self.headers).json()
                self.assertFalse(check["ok"])
                self.assertIn(knob, {i["field"] for i in check["issues"] if i["level"] == "error"})
                self.assertIsNone(check["native_config"])
                self.assertEqual(self.client.post("/api/v1/training/submit", json=draft, headers=self.headers).status_code, 422)
        self.assertEqual(self.app.state.store.list("task"), [])

    def test_other_architectures_settings_cannot_be_silently_submitted(self):
        for key, field, value in (("anima_lora", "moss_blocks_to_swap", 7),
                                  ("musubi_tuner", "hv_text_encoder1", str(self.model))):
            with self.subTest(engine=key):
                draft = self.draft(key)
                draft["params"][field] = value
                checked = self.client.post("/api/v1/training/validate", json=draft, headers=self.headers).json()
                self.assertIn(field, {i["field"] for i in checked["issues"] if i["level"] == "error"})
                self.assertEqual(self.client.post("/api/v1/training/submit", json=draft, headers=self.headers).status_code, 422)
        self.assertEqual(self.app.state.store.list("task"), [])
