"""The job-directory file contract in one place.

A job directory holds ``job.json`` (the submitted config), ``stages.json`` (the argv
per stage) and ``config.native.<format>``. Parsing and writing them here keeps the
supervisor and every adapter on one shape, so a format change lands in one file.
"""

import json
from pathlib import Path

from app.services.common import atomic_text


class JobFiles:
    """Paths and atomic writers for one job directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @property
    def job(self) -> Path:
        return self.directory / "job.json"

    @property
    def stages(self) -> Path:
        return self.directory / "stages.json"

    @property
    def command(self) -> Path:
        return self.directory / "command.json"

    def native_config(self, extension: str) -> Path:
        return self.directory / f"config.native.{extension}"

    def read_job(self) -> dict:
        return json.loads(self.job.read_text(encoding="utf-8"))

    def read_command(self) -> dict:
        return json.loads(self.command.read_text(encoding="utf-8"))

    def write_command(self, commands, cwd) -> None:
        atomic_text(self.command, json.dumps(
            dict(commands=[list(argv) for argv in commands], cwd=str(cwd)),
            ensure_ascii=False, indent=2,
        ))

    def read_stages(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(stage) for stage in json.loads(self.stages.read_text(encoding="utf-8")))

    def write_stages(self, stages) -> None:
        atomic_text(self.stages, json.dumps([list(stage) for stage in stages], ensure_ascii=False, indent=2))
