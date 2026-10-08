"""Plain process and task contracts; no training framework dependencies."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .engine import EngineRevision


class TaskState(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CONNECTION_LOST = "connection_lost"


@dataclass(frozen=True)
class TaskBinding:
    installation_id: str
    revision: EngineRevision
    environment_id: str
    adapter_version: str
    schema_version: str
    # References an immutable, non-secret environment manifest.
    environment_manifest_id: str


@dataclass(frozen=True)
class LaunchSpec:
    """Supervisor receives argv, never a shell command string.

    ``pre_argv`` holds commands that must run, in order, before ``argv`` (e.g.
    musubi-tuner's latent/Text-Encoder caching stages). Everything shares the
    same ``cwd`` and environment.
    """

    argv: tuple[str, ...]
    cwd: Path
    # Overrides are runtime-only; secrets must not be exported in command.json.
    environment_overrides: dict[str, str] = field(default_factory=dict)
    pre_argv: tuple[tuple[str, ...], ...] = ()

    def commands(self) -> tuple[tuple[str, ...], ...]:
        return (*self.pre_argv, self.argv)


@dataclass(frozen=True)
class TrainingEvent:
    kind: str
    payload: dict[str, object]
