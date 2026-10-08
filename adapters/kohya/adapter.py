"""Conservative, single-GPU sd-scripts LoRA adapter. No engine imports here."""
import json
import re
from pathlib import Path
from adapters.base import InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact
from adapters.capabilities import check_value, parse_progress, specs
from adapters.kohya.params import ARCHITECTURES, NETWORK_MODULES, PARAMS
from app.schemas.training import LaunchSpec


def scripts_root(path):
    path = Path(path)
    return path / "sd-scripts" if (path / "sd-scripts" / "train_network.py").is_file() else path


# sd-scripts entry per LoRA network module (sd1/sd2/sdxl share ``networks.lora``).
MODULE_SCRIPTS = {
    "networks.lora_sd3": "sd3_train_network.py",
    "networks.lora_flux": "flux_train_network.py",
    "networks.lora_hunyuan_image": "hunyuan_image_train_network.py",
    "networks.lora_lumina": "lumina_train_network.py",
    "networks.lora_anima": "anima_train_network.py",
}


class KohyaAdapter(EngineAdapter):
    engine_id = "kohya"
    adapter_version = "0.1.0"
    native_format = "toml"
    submittable = True
    environment_strategy = "precreate_venv"
    environment_dir = "venv"
    environment_candidates = ("venv",)
    python_version = "3.11"

    def installation_plan(self, source_path, python, torch_source, uv=None):
        import os
        if os.name != "nt":
            raise ValueError("当前 Kohya 安装接口仅支持 Windows")
        installer = source_path / "setup" / "setup_windows.py"
        if not installer.is_file():
            raise ValueError("缺少 Kohya 官方 setup/setup_windows.py；不会猜测安装命令")
        # The official installer initializes the sd-scripts submodule itself.
        # Do not reject a normal clone before it gets a chance to run; source
        # completeness is checked again after installation and during diagnosis.
        # Portable _pth Python does not add the script directory automatically.
        bootstrap = "import runpy,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name='__main__')"
        # Match setup.bat's explicit prerequisite, including on a reused venv
        # where uv's initial seed packages may have been removed.
        return InstallationPlan("Kohya 官方 Windows headless 安装器（CUDA 由引擎决定）",
                                ((python, "-u", "-m", "pip", "install", "--require-virtualenv",
                                  "--no-input", "-q", "setuptools"),
                                 (python, "-u", "-c", bootstrap, str(installer), "--headless")))

    def detect(self, source_path):
        root = scripts_root(source_path)
        complete = (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()
        # A plain clone can identify the wrapper without containing the training submodule.
        modules = source_path / ".gitmodules"
        wrapper = (source_path / "kohya_gui.py").is_file() and (source_path / "kohya_gui").is_dir()
        declared = False
        if wrapper and modules.is_file():
            from configparser import ConfigParser, Error
            config = ConfigParser(interpolation=None)
            try:
                config.read(modules, encoding="utf-8")
                declared = any(config.get(section, "path", fallback="").strip() == "sd-scripts"
                               for section in config.sections())
            except (OSError, UnicodeError, Error):
                pass
        return DetectionResult(self.engine_id, complete or (wrapper and declared),
                               "sd-scripts training source or Kohya wrapper with declared submodule")

    def source_issues(self, source_path):
        root = scripts_root(source_path)
        if (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file():
            return ()
        return ("Kohya 缺少完整 sd-scripts 训练源码；请在该引擎目录手动运行 git submodule update --init --recursive，或复制完整 sd-scripts 后刷新发现",)

    def architectures(self, source_path):
        root = scripts_root(source_path)
        return tuple(a for a in ARCHITECTURES if (root / a.script).is_file())

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable)

    def validate(self, installation, config):
        issues = []
        if config.get("architecture") not in {a.id for a in self.architectures(installation.source_path)}:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in PARAMS:
            if not p.native and p.unsupported_reason:
                continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
        resolution = values.get("resolution", 512)
        if isinstance(resolution, int) and resolution % 64:
            issues.append(ValidationIssue("resolution", "分辨率必须是 64 的倍数"))
        if config.get("architecture") == "sdxl":
            issues.append(ValidationIssue("architecture", "SDXL 建议使用 1024 分辨率；当前适配仅支持单 GPU LoRA", "warning"))
        return tuple(issues)

    def render(self, config, dataset_config):
        params = config.get("params", {})
        architecture = config.get("architecture")
        values = {}
        for p in PARAMS:
            if not p.native or not p.applies_to(architecture):
                continue
            value = params.get(p.key, p.default)
            if value is None or (isinstance(value, str) and value == ""):
                continue
            values[p.key] = value
        values.update(
            pretrained_model_name_or_path=config["base_model_path"],
            dataset_config=str(dataset_config),
            output_dir=config["output_dir"],
            output_name="lora",
            network_module=NETWORK_MODULES.get(architecture, "networks.lora"),
        )
        if architecture == "sd2":
            values["v2"] = True
        if architecture == "sdxl":
            values["network_train_unet_only"] = True
        return "\n".join(f"{k} = {str(v).lower() if isinstance(v, bool) else json.dumps(v, ensure_ascii=False)}" for k, v in values.items()) + "\n"

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        params = config.get("params", {})
        dataset = job_directory / "dataset.toml"
        # One subset explicitly names the dataset path; no repeat-folder naming convention required.
        text = "[general]\n"
        text += "caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += f'enable_bucket = {str(bool(params.get("enable_bucket", True))).lower()}\n\n'
        text += "[[datasets]]\n"
        text += f'resolution = {params.get("resolution") or 512}\n'
        text += "[[datasets.subsets]]\n"
        text += f'num_repeats = {params.get("num_repeats") or 1}\n'
        text += "image_dir = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
        atomic_text(dataset, text)
        target = job_directory / "config.native.toml"
        atomic_text(target, self.render(config, dataset))
        return target

    def preview_argv(self, installation, config, native_config):
        root = scripts_root(installation.source_path)
        spec = next((a for a in ARCHITECTURES if a.id == config.get("architecture")), None)
        script = root / (spec.script if spec else "train_network.py")
        return (str(installation.python_executable), "-u", str(script), "--config_file", str(native_config))

    def build_launch(self, installation, native_config):
        import tomllib
        values = tomllib.loads(native_config.read_text(encoding="utf-8"))
        script = MODULE_SCRIPTS.get(values.get("network_module"))
        if not script:
            script = "sdxl_train_network.py" if values.get("network_train_unet_only") else "train_network.py"
        root = scripts_root(installation.source_path)
        return LaunchSpec((str(installation.python_executable), "-u", str(root / script), "--config_file", str(native_config)), root,
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

    def parse_log(self, line):
        # tqdm's progress is explicit; no epoch/ETA guesses and no fabricated loss points.
        return parse_progress(line)

    def recovery_capabilities(self, installation):
        # Stop is process interruption, not a promise of checkpoint-on-stop or resumption.
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        config = json.loads((job_directory / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        found = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                # sd-scripts step saves are "{name}-{step:06d}"; final saves
                # ("lora.safetensors", "{name}.safetensors") carry no step.
                match = re.search(r"-(\d{6})$", path.stem)
                found.append(Artifact(path, "checkpoint", True, int(match[1]) if match else None))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
