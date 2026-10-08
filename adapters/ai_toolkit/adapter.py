"""Static AI Toolkit discovery, parameter surface and native-config preview.

Training submission is intentionally not implemented yet: the adapter advertises
its architectures/parameters and renders a preview, but cannot queue a job.
"""
import json

from adapters.base import InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities
from adapters.capabilities import check_value, specs
from adapters.ai_toolkit.params import ARCHITECTURES, ARCH_BY_ID, PARAMS


def _scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _dump_yaml(value, indent=0):
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return pad + "{}"
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(item) if not isinstance(item, (dict, list)) else _dump_yaml(item, indent)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return pad + "[]"
        lines = []
        for item in value:
            if isinstance(item, dict) and item:
                body = _dump_yaml(item, indent + 1).split("\n")
                lines.append(f"{pad}- {body[0].strip()}")
                lines.extend(body[1:])
            else:
                lines.append(f"{pad}- {_scalar(item) if not isinstance(item, (dict, list)) else _dump_yaml(item, indent)}")
        return "\n".join(lines)
    return f"{pad}{_scalar(value)}"


class AiToolkitAdapter(EngineAdapter):
    engine_id = "ai_toolkit"
    adapter_version = "0.1.0"
    training_notice = "AI Toolkit 已支持参数登记与配置预览；训练提交尚未适配"
    native_format = "yaml"
    submittable = False

    environment_strategy = "native"
    environment_candidates = (".venv", "venv")

    def installation_plan(self, source_path, python, torch_source, uv=None):
        manager = source_path / "manager" / "__main__.py"
        if not manager.is_file():
            raise ValueError("缺少 AI Toolkit 官方 manager/__main__.py")
        # The manager performs hardware detection, uv/Python provisioning,
        # PyTorch selection and requirements synchronization itself. Do not
        # replace that flow with a generic pip install guessed by LP.
        return InstallationPlan(
            "AI Toolkit 官方 manager install 环境安装流程",
            ((python, "-u", str(manager), "install"),),
        )

    def detect(self, source_path):
        matched = (source_path / "run.py").is_file() and (source_path / "toolkit").is_dir() and (source_path / "jobs").is_dir()
        return DetectionResult(self.engine_id, matched, "AI Toolkit entry, toolkit and jobs")

    def architectures(self, source_path):
        return tuple(ARCHITECTURES)

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable)

    def validate(self, installation, config):
        issues = []
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            issues.append(ValidationIssue("architecture", "未知的底模类型"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in PARAMS:
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
        if not issues:
            issues.append(ValidationIssue("installation_id", "参数已补齐，训练提交尚未实现；当前仅提供配置预览", "warning"))
        return tuple(issues)

    def render(self, config, dataset_config):
        params = config.get("params", {})
        architecture = config.get("architecture")
        spec = ARCH_BY_ID.get(architecture)
        blocks: dict[str, dict] = {"network": {}, "save": {}, "logging": {}, "train": {}, "model": {}, "sample": {}, "dataset": {}}
        for p in PARAMS:
            if not p.applies_to(architecture):
                continue
            value = params.get(p.key, p.default)
            if value is None or (isinstance(value, str) and value == ""):
                continue
            if p.key == "text_encoder_bits":
                value = int(value)
            blocks.setdefault(p.block or "train", {})[p.key] = value

        dataset = dict(blocks["dataset"])
        dataset["folder_path"] = config["dataset_path"]
        dataset.setdefault("caption_ext", ".txt")

        model = dict(blocks["model"])
        model["name_or_path"] = config["base_model_path"]
        model["arch"] = spec.native if spec else architecture

        process = dict(
            type="sd_trainer",
            training_folder=config["output_dir"],
            device="cuda:0",
            network=blocks["network"],
            save=blocks["save"],
            logging=blocks["logging"],
            datasets=[dataset],
            train=blocks["train"],
            model=model,
            sample=blocks["sample"],
        )
        document = dict(job="train", config=dict(name=config.get("name") or "lora", process=[process]))
        return _dump_yaml(document) + "\n"

    def preview_argv(self, installation, config, native_config):
        return (str(installation.python_executable), str(installation.source_path / "run.py"), str(native_config))

    def write_native_config(self, installation, config, job_directory):
        raise NotImplementedError(self.training_notice)

    def build_launch(self, installation, native_config):
        raise NotImplementedError(self.training_notice)

    def parse_log(self, line):
        return ()

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        return ()
