"""Contract shared by statically registered engine adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from adapters.capabilities import ArchSpec
from app.schemas.engine import EngineInstallation, EngineRevision
from app.schemas.training import LaunchSpec, TrainingEvent


@dataclass(frozen=True)
class DetectionResult:
    engine_id: str
    matched: bool
    reason: str
    revision: EngineRevision | None = None


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class RecoveryCapabilities:
    graceful_stop: bool = False
    resume_weights: bool = False
    resume_full_state: bool = False


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    complete: bool
    # Explicit training step when the adapter can determine it (else None; the
    # supervisor falls back to a filename heuristic).
    step: int | None = None


@dataclass(frozen=True)
class InstallationPlan:
    title: str
    commands: tuple[tuple[str, ...], ...]


class EngineAdapter(ABC):
    engine_id: str
    adapter_version: str
    training_notice: str | None = None
    # Preview format of the rendered native config, and whether the adapter can
    # actually queue a training job. Adapters that only describe parameters set
    # ``submittable = False`` and keep ``write_native_config`` unimplemented.
    native_format: str = "toml"
    submittable: bool = True

    def architectures(self, source_path: Path) -> tuple[ArchSpec, ...]:
        """Base-model types this clone can train, each with its entry script."""
        return ()

    def preview_argv(
        self, installation: EngineInstallation, config: dict[str, object], native_config: Path
    ) -> tuple[str, ...]:
        """Literal argv shown before submit; must not require submit support."""
        raise NotImplementedError("该引擎尚未提供启动预览")

    def preview_commands(
        self, installation: EngineInstallation, config: dict[str, object], native_config: Path
    ) -> tuple[tuple[str, ...], ...]:
        """Ordered commands the supervisor runs; the last one is training."""
        return (self.preview_argv(installation, config, native_config),)

    # Installation ownership is explicit per engine:
    # - ``native``: the copied engine's manager creates and syncs its environment.
    # - ``uv_project``: the engine's pyproject/uv workflow creates and syncs it.
    # - ``precreate_venv``: LP only creates the empty venv required by the
    #   engine's own installer, then delegates dependency installation to it.
    environment_strategy: str = "precreate_venv"
    environment_dir: str | None = None
    environment_candidates: tuple[str, ...] = ()
    python_version: str | None = None
    diagnostic_modules: tuple[str, ...] = ("accelerate", "transformers", "safetensors", "diffusers", "toml", "yaml")

    def installation_sources(self, source_path: Path) -> tuple[str, ...]:
        """CUDA extras declared by this particular clone, not directory labels."""
        return ()

    def installation_plan(
        self, source_path: Path, python: str, torch_source: str, uv: str | None = None
    ) -> InstallationPlan:
        """Use this engine's own installer or documented dependency entrypoints."""
        raise ValueError("该引擎尚未提供安装接口")

    def source_issues(self, source_path: Path) -> tuple[str, ...]:
        """Report incomplete source trees without importing or running them."""
        return ()

    @abstractmethod
    def detect(self, source_path: Path) -> DetectionResult:
        """Static discovery only; must not execute engine code."""

    @abstractmethod
    def capabilities(self, installation: EngineInstallation) -> dict[str, object]:
        """Describe supported architectures, parameters and version constraints."""

    @abstractmethod
    def validate(
        self, installation: EngineInstallation, config: dict[str, object]
    ) -> tuple[ValidationIssue, ...]:
        """Reject unsupported configurations before queue submission."""

    @abstractmethod
    def write_native_config(
        self,
        installation: EngineInstallation,
        config: dict[str, object],
        job_directory: Path,
    ) -> Path:
        """Write native config into the job directory, not the engine source."""

    @abstractmethod
    def build_launch(
        self, installation: EngineInstallation, native_config: Path
    ) -> LaunchSpec:
        """Build explicit Python/CLI argv and cwd without using a shell."""

    @abstractmethod
    def parse_log(self, line: str) -> tuple[TrainingEvent, ...]:
        """Unrecognized output remains in the raw log; do not invent metrics."""

    @abstractmethod
    def recovery_capabilities(
        self, installation: EngineInstallation
    ) -> RecoveryCapabilities:
        """Declare stop and recovery semantics separately."""

    @abstractmethod
    def collect_artifacts(self, job_directory: Path) -> tuple[Artifact, ...]:
        """Identify artifacts; incomplete writes must not be presented as ready."""
