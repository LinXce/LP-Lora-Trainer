"""Application paths only; construction does not create or scan directories."""

from dataclasses import dataclass
from pathlib import Path

# Single source of truth for the loopback API/browser port (desktop launcher,
# terminal fallback and the backend itself must never disagree).
DEFAULT_API_PORT = 5900


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    engine_root: Path
    data_root: Path
    frontend_dist: Path

    @classmethod
    def for_workspace(cls, project_root: Path) -> "AppPaths":
        root = project_root.resolve()
        return cls(
            project_root=root,
            engine_root=root / "engine",
            data_root=root / "data",
            frontend_dist=root / "frontend" / "dist",
        )
