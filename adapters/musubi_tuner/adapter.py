"""Static Musubi Tuner discovery; no engine imports or training promises."""
from adapters.base import InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities


class MusubiTunerAdapter(EngineAdapter):
    engine_id = "musubi_tuner"
    adapter_version = "0.1.0"
    training_notice = "Musubi Tuner 已支持版本登记和环境诊断；训练配置适配尚未实现"

    def installation_plan(self, source_path, python, torch_source):
        if not self.detect(source_path).matched:
            raise ValueError("Musubi Tuner 源码不完整")
        commands = []
        if torch_source != "existing":
            commands.append((python, "-m", "pip", "--isolated", "install", "torch", "torchvision",
                             "--index-url", "https://download.pytorch.org/whl/" + torch_source))
        commands.append((python, "-m", "pip", "--isolated", "install", "-e", "."))
        return InstallationPlan("Musubi 官方 pip 安装流程（不安装可选依赖）", tuple(commands))

    def detect(self, source_path):
        import tomllib
        metadata = source_path / "pyproject.toml"
        try:
            project = tomllib.loads(metadata.read_text(encoding="utf-8")).get("project", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return DetectionResult(self.engine_id, False, "Missing or invalid Musubi project metadata")
        package = source_path / "src" / "musubi_tuner"
        entries = any(source_path.glob("*_train_network.py"))
        matched = project.get("name") == "musubi-tuner" and package.is_dir() and entries
        return DetectionResult(self.engine_id, matched, "Musubi project metadata, package and training entry")

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id, architectures=[], params=[])

    def validate(self, installation, config):
        return (ValidationIssue("installation_id", self.training_notice),)

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
