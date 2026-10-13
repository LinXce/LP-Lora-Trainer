"""Hardware-aware install recommendation, mirror presets and one-click prefill."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.registry import ADAPTERS
from app.api.server import create_app
from app.config import AppPaths
from app.services import hardware, install_plan
from app.services.engines import EngineService
from app.services.installation import InstallationService
from app.storage.state import StateStore
from tests.local_client import LocalClient

HW = dict(os="Windows", arch="AMD64", gpu_name="NVIDIA GeForce RTX 4080",
          driver_version="566.03", cuda_version="12.6", compute_cap="8.9", cuda_usable=True)
MUSUBI_SOURCES = ("cu118", "cu121", "cu124", "cu126")


def record(engine_id, confirmed=True):
    return dict(installation_id="inst", engine_id=engine_id, source_path=".",
                engine_type_confirmed=confirmed)


class CudaRankTests(unittest.TestCase):
    def test_cuda_rank(self):
        self.assertEqual(hardware.cuda_rank("cu124"), (12, 4))
        self.assertEqual(hardware.cuda_rank("cu118"), (11, 8))
        self.assertEqual(hardware.cuda_rank("cu130"), (13, 0))
        self.assertEqual(hardware.cuda_rank("cu12.1"), (12, 1))
        self.assertIsNone(hardware.cuda_rank("existing"))
        self.assertIsNone(hardware.cuda_rank(None))

    def test_driver_rank_and_extra(self):
        self.assertEqual(hardware.driver_rank("12.6"), (12, 6))
        self.assertIsNone(hardware.driver_rank(None))
        self.assertEqual(hardware.cuda_extra((12, 6)), "cu126")
        self.assertIsNone(hardware.cuda_extra(None))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        hardware.reset_cache()

    def tearDown(self):
        hardware.reset_cache()

    def test_classic_header(self):
        def run(args):
            if args == ["nvidia-smi"]:
                return "| NVIDIA-SMI 566.03  Driver Version: 566.03  CUDA Version: 12.6 |\n"
            if "compute_cap" in args[1]:
                return "NVIDIA GeForce RTX 4090, 566.03, 8.9\n"
            return None

        with patch("app.services.hardware._run", side_effect=run):
            info = hardware.probe()
        self.assertEqual(info["cuda_version"], "12.6")
        self.assertEqual(info["driver_version"], "566.03")
        self.assertEqual(info["compute_cap"], "8.9")
        self.assertTrue(info["cuda_usable"])

    def test_new_header_without_compute_cap(self):
        # Newer drivers say "KMD Version" / "CUDA UMD Version" and may reject compute_cap.
        def run(args):
            if args == ["nvidia-smi"]:
                return "| NVIDIA-SMI 616.64   KMD Version: 616.64   CUDA UMD Version: 13.4 |\n"
            if "compute_cap" in args[1]:
                return None
            return "NVIDIA GeForce RTX 5080, 616.64\n"

        with patch("app.services.hardware._run", side_effect=run):
            info = hardware.probe()
        self.assertEqual(info["cuda_version"], "13.4")
        self.assertEqual(info["driver_version"], "616.64")
        self.assertIsNone(info["compute_cap"])
        self.assertTrue(info["cuda_usable"])

    def test_no_nvidia(self):
        with patch("app.services.hardware._run", return_value=None):
            info = hardware.probe()
        self.assertIsNone(info["cuda_version"])
        self.assertFalse(info["cuda_usable"])


class PlanTests(unittest.TestCase):
    def build(self, engine_id, hw, sources=(), confirmed=True):
        adapter = ADAPTERS[engine_id]
        with patch.object(adapter, "installation_sources", return_value=sources), \
                patch.object(adapter, "install_blockers", return_value=()):
            return install_plan.build(adapter, record(engine_id, confirmed), hw)

    def test_picks_highest_supported(self):
        plan = self.build("musubi_tuner", HW, MUSUBI_SOURCES)
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["torch_source"], "cu126")
        self.assertFalse(plan["engine_decides_cuda"])
        self.assertIn("12.6", plan["torch_source_reason"])

    def test_lower_driver(self):
        self.assertEqual(self.build("musubi_tuner", {**HW, "cuda_version": "12.1"}, MUSUBI_SOURCES)["torch_source"], "cu121")
        self.assertEqual(self.build("musubi_tuner", {**HW, "cuda_version": "11.8"}, MUSUBI_SOURCES)["torch_source"], "cu118")

    def test_all_too_new_blocks(self):
        plan = self.build("musubi_tuner", {**HW, "cuda_version": "12.0"}, ("cu124", "cu126"))
        self.assertFalse(plan["supported"])
        self.assertIn("cu124", plan["blocked_reason"])

    def test_engine_decides_cuda(self):
        plan = self.build("kohya", HW)
        self.assertTrue(plan["engine_decides_cuda"])
        self.assertEqual(plan["torch_source"], "existing")
        self.assertEqual(plan["torch_source_options"], ["existing"])

    def test_detected_type_needs_no_manual_confirmation(self):
        # A uniquely detected type (engine_id set) is usable without a separate confirm step.
        plan = self.build("kohya", HW, confirmed=False)
        self.assertTrue(plan["supported"])
        self.assertIsNone(plan["blocked_reason"])
        self.assertFalse(plan["engine_type_confirmed"])

    def test_unknown_type_is_blocked(self):
        plan = install_plan.build(None, record(None), HW)
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["blocked_reason"], "无法唯一识别引擎类型，请先确认")

    def test_no_cuda_blocks(self):
        plan = self.build("musubi_tuner", {**HW, "cuda_version": None, "cuda_usable": False}, MUSUBI_SOURCES)
        self.assertFalse(plan["supported"])
        self.assertIn("NVIDIA", plan["blocked_reason"])

    def test_torch_mirror_urls_follow_cuda_tag(self):
        plan = self.build("musubi_tuner", HW, MUSUBI_SOURCES)
        self.assertEqual(plan["cuda_tag"], "cu126")
        official = next(p for p in plan["torch_mirror_presets"] if p["value"] == "official")
        self.assertTrue(official["url"].endswith("cu126"))

    def test_requested_source_is_honoured(self):
        with patch.object(ADAPTERS["musubi_tuner"], "installation_sources", return_value=MUSUBI_SOURCES):
            plan = install_plan.build(ADAPTERS["musubi_tuner"], record("musubi_tuner"), HW, "cu121")
        self.assertEqual(plan["torch_source"], "cu121")

    def test_kohya_is_windows_only(self):
        with patch("os.name", "posix"):
            self.assertTrue(ADAPTERS["kohya"].install_blockers(HW))
        with patch("os.name", "nt"):
            self.assertEqual(ADAPTERS["kohya"].install_blockers(HW), ())


class EngineTagTests(unittest.TestCase):
    """Wheel mirrors must use the tag the engine pins, not the driver's maximum."""

    def plan(self, locale_value):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements_pytorch_windows.txt").write_text(
                "--extra-index-url https://download.pytorch.org/whl/cu128\n"
                "torch==2.7.0+cu128\n",
                encoding="utf-8",
            )
            record = {"installation_id": "k", "engine_id": "kohya",
                      "source_path": str(root), "engine_type_confirmed": True}
            hardware_info = {"cuda_usable": True, "cuda_version": "13.4", "os": "Windows"}
            with patch.dict(os.environ, {"LC_ALL": "", "LC_MESSAGES": "", "LANG": locale_value}), \
                    patch.object(install_plan.locale, "getlocale", return_value=(None, None)):
                return install_plan.build(ADAPTERS["kohya"], record, hardware_info)

    def test_mirror_urls_follow_the_engine_tag(self):
        plan = self.plan("zh_CN.UTF-8")
        # The driver supports 13.4, but this engine installs torch+cu128.
        self.assertEqual(plan["cuda_tag"], "cu128")
        self.assertEqual(plan["cuda_tag_source"], "engine")
        sjtu = next(p for p in plan["torch_mirror_presets"] if p["value"] == "sjtu")
        self.assertEqual(sjtu["url"], "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128")
        self.assertIn("cu128", plan["torch_source_reason"])

    def test_a_chinese_locale_preselects_the_working_mirror(self):
        self.assertEqual(self.plan("zh_CN.UTF-8")["torch_mirror_preset"], "sjtu")
        self.assertEqual(self.plan("en_US.UTF-8")["torch_mirror_preset"], "project_default")

    def test_presets_avoid_the_dead_aliyun_wheel_path(self):
        plan = self.plan("zh_CN.UTF-8")
        self.assertNotIn("aliyun", [p["value"] for p in plan["torch_mirror_presets"]])


class EnvModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-install-plan-")
        self.root = Path(self.temp.name)
        self.store = StateStore(self.root / "data")
        self.engines = EngineService(self.store, lambda: {"engine_root": str(self.root / "engine")})
        # _install_environment reads scripts/bootstrap_uv.py from the project root.
        self.service = InstallationService(self.store, self.engines, Path(__file__).resolve().parents[1])

    def tearDown(self):
        self.temp.cleanup()

    def test_env_mode_never_injects_the_wheel_mirror_as_an_index(self):
        source = self.root / "kohya"; source.mkdir()
        _, _, env, _ = self.service._install_environment(
            source, ADAPTERS["kohya"], "python.exe",
            "https://pypi.example/simple", "https://whl.example/cu124")
        self.assertEqual(env.get("PIP_INDEX_URL"), "https://pypi.example/simple")
        # The adapter's plan passes it as --find-links instead: a CUDA-tagged
        # index URL answers every ordinary package lookup with a 404 and only
        # slows the resolution down.
        self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
        self.assertNotIn("UV_EXTRA_INDEX_URL", env)

    def test_named_mode_leaves_env_untouched(self):
        source = self.root / "musubi"; source.mkdir()
        _, _, env, _ = self.service._install_environment(
            source, ADAPTERS["musubi_tuner"], "python.exe",
            "https://pypi.example/simple", "https://whl.example/cu124")
        self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
        self.assertNotIn("UV_EXTRA_INDEX_URL", env)


class InstallPlanApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-install-plan-api-")
        self.paths = AppPaths.for_workspace(Path(self.temp.name))
        root = self.paths.engine_root / "kohya_ss"
        (root / "library").mkdir(parents=True)
        (root / "library" / "train_util.py").write_text("# fixture\n", encoding="utf-8")
        (root / "train_network.py").write_text("# fixture\n", encoding="utf-8")
        (root / "sdxl_train_network.py").write_text("# fixture\n", encoding="utf-8")
        self.hw_patch = patch("app.services.hardware.probe", return_value=HW)
        self.hw_patch.start()
        self.app = create_app(self.paths, token="unit-secret", start_worker=False)
        self.client = LocalClient(self.app)

    def tearDown(self):
        self.client.close()
        self.hw_patch.stop()
        self.temp.cleanup()

    def test_hardware_and_install_plan_endpoints(self):
        self.client.post("/api/v1/session", headers={"Origin": self.client.base_url})
        info = self.client.get("/api/v1/system/hardware").json()
        self.assertEqual(info["cuda_version"], "12.6")
        records = self.client.post("/api/v1/engines/rescan").json()
        key = next(r["installation_id"] for r in records if r["label"] == "kohya_ss")
        plan = self.client.get(f"/api/v1/engines/{key}/install-plan").json()
        self.assertEqual(plan["engine_id"], "kohya")
        self.assertTrue(plan["mirror_presets"])
        self.assertEqual(plan["python_mirror_preset"], "github")
        self.assertIsNone(next(p for p in plan["python_mirror_presets"] if p["value"] == "github")["url"])
        self.assertTrue(next(p for p in plan["python_mirror_presets"] if p["value"] == "nju")["url"])
        self.assertIn(plan["torch_source"], plan["torch_source_options"])
        self.assertEqual(plan["os"], "Windows")
