"""Engine code identity is separate from an executable installation."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ManagementMode(str, Enum):
    USER_MANAGED = "user_managed"
    APPLICATION_MANAGED = "application_managed"
    EXTERNAL_REFERENCE = "external_reference"


class InstallationState(str, Enum):
    DISCOVERED = "discovered"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"
    MISSING = "missing"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    EXPERIMENTAL = "experimental"
    VERIFIED = "verified"


@dataclass(frozen=True)
class EngineRevision:
    source: str | None = None
    requested_ref: str | None = None
    commit: str | None = None
    fingerprint: str | None = None


@dataclass(frozen=True)
class EngineInstallation:
    installation_id: str
    engine_id: str
    source_path: Path
    management_mode: ManagementMode
    revision: EngineRevision
    state: InstallationState = InstallationState.DISCOVERED
    verification: VerificationState = VerificationState.UNVERIFIED
    environment_id: str | None = None
    python_executable: Path | None = None
