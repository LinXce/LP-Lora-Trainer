"""Static AI Toolkit discovery; training profiles require version-specific adaptation."""
from adapters.base import EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities


class AiToolkitAdapter(EngineAdapter):
    engine_id = "ai_toolkit"
    adapter_version = "0.1.0"

    def detect(self, source_path):
        matched = (source_path / "run.py").is_file() and (source_path / "toolkit").is_dir() and (source_path / "jobs").is_dir()
        return DetectionResult(self.engine_id, matched, "AI Toolkit entry, toolkit and jobs")

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id, architectures=[], params=[])

    def validate(self, installation, config):
        return (ValidationIssue("installation_id", "AI Toolkit 已支持登记和环境诊断，训练配置尚未适配；不会提交占位任务"),)

    def write_native_config(self, installation, config, job_directory):
        raise NotImplementedError("AI Toolkit training profile is not implemented")

    def build_launch(self, installation, native_config):
        raise NotImplementedError("AI Toolkit training profile is not implemented")

    def parse_log(self, line):
        return ()

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        return ()
