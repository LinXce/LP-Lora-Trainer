"""Static Musubi Tuner discovery; no engine imports or training promises."""
from adapters.base import InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities


class MusubiTunerAdapter(EngineAdapter):
    engine_id = "musubi_tuner"
    adapter_version = "0.1.0"
    environment_strategy = "uv_project"
    environment_dir = ".venv"
    environment_candidates = (".venv", "venv")
    python_version = "3.10"
    diagnostic_modules = ("accelerate", "transformers", "safetensors", "diffusers", "toml")

    def installation_sources(self, source_path):
        import re
        import tomllib
        try:
            metadata = tomllib.loads((source_path / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return ()
        extras = metadata.get("project", {}).get("optional-dependencies", {})
        return tuple(sorted(key for key in extras if re.fullmatch(r"cu\d{3}", key)))
    training_notice = "Musubi Tuner 已支持版本登记和环境诊断；训练配置适配尚未实现"

    def installation_plan(self, source_path, python, torch_source, uv=None):
        if not self.detect(source_path).matched:
            raise ValueError("Musubi Tuner 源码不完整")
        if not uv:
            raise ValueError("Musubi Tuner 的官方安装流程需要 uv")

        # Musubi owns its dependency graph in pyproject.toml. LP only invokes
        # the project's documented uv workflow; it must not install torch or
        # requirements through a shared, guessed pip command.
        sources = self.installation_sources(source_path)
        if torch_source != "existing" and torch_source not in sources:
            raise ValueError(
                f"当前 Musubi Tuner 未声明 {torch_source} 安装配置；"
                f"可用配置：{', '.join(sources) or '无 CUDA extra'}"
            )
        extra = None if torch_source == "existing" else torch_source

        command = [uv, "sync"]
        if extra:
            command.extend(("--extra", extra))
        return InstallationPlan(
            "Musubi Tuner 官方 uv 项目安装流程",
            (tuple(command),),
        )

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
