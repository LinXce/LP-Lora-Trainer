"""Contract shared by engine adapters; implementations are not yet provided."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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


class EngineAdapter(ABC):
    engine_id: str
    adapter_version: str

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
