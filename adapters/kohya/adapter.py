"""Conservative, single-GPU sd-scripts LoRA adapter. No engine imports here."""
import json
import math
import re
from pathlib import Path
from adapters.base import EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact
from app.schemas.training import LaunchSpec, TrainingEvent


def scripts_root(path):
    path = Path(path)
    return path / "sd-scripts" if (path / "sd-scripts" / "train_network.py").is_file() else path


def param(key, label, kind, default, group="basic", **limits):
    return dict(key=key, label=label, type=kind, default=default, group=group,
                section="训练参数" if group == "basic" else "高级参数", **limits)


PARAMS = [
    param("max_train_steps", "训练步数", "int", 1000, min=1, max=10000000),
    param("learning_rate", "学习率", "float", 0.0001, min=0.00000001, max=1),
    param("network_dim", "LoRA Rank", "int", 16, min=1, max=1024),
    param("network_alpha", "LoRA Alpha", "int", 16, min=1, max=1024),
    param("train_batch_size", "Batch Size", "int", 1, min=1, max=128),
    param("resolution", "训练分辨率", "int", 512, min=64, max=4096, step=64),
    param("mixed_precision", "混合精度", "enum", "fp16", options=[dict(value=v, label=v) for v in ("no", "fp16", "bf16")]),
    param("save_every_n_steps", "保存间隔（步）", "int", 200, min=1, max=10000000),
    param("seed", "随机种子", "int", 42, "advanced", min=0, max=4294967295),
    param("gradient_checkpointing", "梯度检查点", "bool", True, "advanced"),
    param("cache_latents", "缓存 Latents", "bool", True, "advanced"),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 0, "advanced", min=0, max=32),
    param("lr_scheduler", "学习率策略", "enum", "constant", "advanced", options=[dict(value=v, label=v) for v in ("constant", "cosine", "linear")]),
]


class KohyaAdapter(EngineAdapter):
    engine_id = "kohya"
    adapter_version = "0.1.0"

    def detect(self, source_path):
        root = scripts_root(source_path)
        match = (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()
        return DetectionResult(self.engine_id, match, "sd-scripts training entry and library")

    def capabilities(self, installation):
        root = scripts_root(installation.source_path)
        arch = ["sd1", "sd2"] if (root / "train_network.py").is_file() else []
        if (root / "sdxl_train_network.py").is_file():
            arch.append("sdxl")
        return dict(installation_id=installation.installation_id, architectures=arch, params=PARAMS)

    def validate(self, installation, config):
        issues = []
        if config.get("architecture") not in self.capabilities(installation)["architectures"]:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p["key"] for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in PARAMS:
            v = values.get(p["key"], p["default"])
            valid = True
            if p["type"] in ("int", "float"):
                valid = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                valid = valid and (p["type"] != "int" or isinstance(v, int))
                valid = valid and p.get("min", -math.inf) <= v <= p.get("max", math.inf)
            elif p["type"] == "bool":
                valid = isinstance(v, bool)
            elif p["type"] == "enum":
                valid = v in [o["value"] for o in p["options"]]
            if not valid:
                issues.append(ValidationIssue(p["key"], "参数类型或取值范围不正确"))
        if isinstance(values.get("resolution"), int) and values["resolution"] % 64:
            issues.append(ValidationIssue("resolution", "分辨率必须是 64 的倍数"))
        if config.get("architecture") == "sdxl":
            issues.append(ValidationIssue("architecture", "SDXL 建议使用 1024 分辨率；当前适配仅支持单 GPU LoRA", "warning"))
        return tuple(issues)

    def render(self, config, dataset_config):
        values = {p["key"]: config.get("params", {}).get(p["key"], p["default"]) for p in PARAMS}
        values.pop("resolution")
        values.update(pretrained_model_name_or_path=config["base_model_path"],
                      dataset_config=str(dataset_config), output_dir=config["output_dir"],
                      output_name="lora", network_module="networks.lora", save_model_as="safetensors")
        if config["architecture"] == "sd2":
            values["v2"] = True
        if config["architecture"] == "sdxl":
            values["network_train_unet_only"] = True
        return "\n".join(f"{k} = {str(v).lower() if isinstance(v, bool) else json.dumps(v, ensure_ascii=False)}" for k,v in values.items()) + "\n"

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        dataset = job_directory / "dataset.toml"
        # One subset explicitly names the dataset path; no repeat-folder naming convention required.
        text = '[general]\ncaption_extension = ".txt"\nenable_bucket = true\n\n[[datasets]]\n'
        text += f'resolution = {config.get("params", {}).get("resolution", 512)}\n'
        text += '[[datasets.subsets]]\nnum_repeats = 1\nimage_dir = ' + json.dumps(config["dataset_path"], ensure_ascii=False) + '\n'
        atomic_text(dataset, text)
        target = job_directory / "config.native.toml"
        atomic_text(target, self.render(config, dataset))
        return target

    def build_launch(self, installation, native_config):
        import tomllib
        values = tomllib.loads(native_config.read_text(encoding="utf-8"))
        architecture = "sdxl" if values.get("network_train_unet_only") else "sd1"
        root = scripts_root(installation.source_path)
        script = root / ("sdxl_train_network.py" if architecture == "sdxl" else "train_network.py")
        return LaunchSpec((str(installation.python_executable), "-u", str(script), "--config_file", str(native_config)), root,
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

    def parse_log(self, line):
        # tqdm's progress is explicit; no epoch/ETA guesses and no fabricated loss points.
        progress = re.search(r"\|\s*(\d+)\s*/\s*(\d+)", line)
        if not progress:
            return ()
        payload = {"step": int(progress[1]), "total_steps": int(progress[2])}
        loss = re.search(r"(?:avr_loss|avg_loss|loss)[=:]\s*([0-9.eE+-]+)", line)
        if loss:
            try:
                value = float(loss[1])
                if math.isfinite(value): payload["loss"] = value
            except ValueError: pass
        speed = re.search(r"([0-9.]+)it/s", line)
        if speed: payload["it_per_sec"] = float(speed[1])
        return (TrainingEvent("progress", payload),)

    def recovery_capabilities(self, installation):
        # Stop is process interruption, not a promise of checkpoint-on-stop or resumption.
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        config = json.loads((job_directory / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        return tuple(Artifact(p, "checkpoint" if p.suffix == ".safetensors" else "sample", True)
                     for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".safetensors", ".png", ".jpg", ".webp"))
