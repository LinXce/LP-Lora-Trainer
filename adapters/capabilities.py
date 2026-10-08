"""Shared capability descriptors for engine adapters.

Adapters declare base-model types (:class:`ArchSpec`) and their full parameter
surface (:class:`Param`) without importing any engine code. The frontend renders
parameters generically; the backend filters them by architecture for preview and
native-config rendering.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

from app.schemas.training import TrainingEvent


@dataclass(frozen=True)
class ArchSpec:
    """A base-model type ("底模类型") and the training entry it selects.

    ``id`` is the canonical id shared across engines so the catalog can merge
    them. ``script`` is the relative training entry; ``native`` is the engine's
    own architecture value when it differs from ``id`` (ai-toolkit's ``arch``).
    """

    id: str
    label: str
    script: str = ""
    native: str = ""


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    type: str
    default: Any = None
    group: str = "basic"
    section: str = ""
    # Native-config block this parameter belongs to (ai-toolkit nests blocks;
    # kohya/musubi write a flat file and leave it empty).
    block: str = ""
    # Empty tuple means the parameter applies to every architecture.
    architectures: tuple[str, ...] = ()
    # CLI/config key when it differs from the (unique) UI key.
    native_key: str = ""
    # False keeps the parameter UI-only (e.g. consumed by the dataset config).
    native: bool = True
    # True when the engine argument is a list (nargs="*"), rendered as a TOML array.
    list_arg: bool = False
    # True when the UI should offer a multi-line input (list arguments, plus any
    # parameter whose value is naturally a list of lines, e.g. resolutions).
    multiline: bool = False
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[tuple[str, str], ...] = ()
    help: str = ""
    unsupported_reason: str = ""

    def applies_to(self, architecture: str | None) -> bool:
        return not self.architectures or architecture in self.architectures

    @property
    def enum_values(self) -> list[str]:
        return [value for value, _ in self.options]

    def spec(self) -> dict:
        out: dict[str, Any] = dict(
            key=self.key,
            label=self.label,
            type=self.type,
            group=self.group,
            section=self.section,
            default=self.default,
            options=[dict(value=value, label=label) for value, label in self.options],
        )
        if self.native_key:
            out["native_key"] = self.native_key
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.help:
            out["help"] = self.help
        if self.unsupported_reason:
            out["unsupported_reason"] = self.unsupported_reason
        if self.list_arg:
            out["list_arg"] = True
        if self.multiline or self.list_arg:
            out["multiline"] = True
        if self.architectures:
            out["architectures"] = list(self.architectures)
        return out


def param(key, label, kind, default, group="basic", section="", **kwargs) -> Param:
    return Param(key=key, label=label, type=kind, default=default, group=group, section=section, **kwargs)


def enum(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    return tuple((value, value) for value in values)


def specs(params: Iterable[Param]) -> list[dict]:
    return [p.spec() for p in params]


def check_value(p: Param, value: Any) -> str | None:
    """Return an error message when ``value`` is invalid for ``p``, else None.

    ``None`` always means "unset" and is accepted so adapter defaults win.
    """
    if value is None:
        return None
    if p.type in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "参数类型或取值范围不正确"
        if p.type == "int" and not isinstance(value, int):
            return "参数类型或取值范围不正确"
        if not (p.min is None or value >= p.min):
            return "参数类型或取值范围不正确"
        if not (p.max is None or value <= p.max):
            return "参数类型或取值范围不正确"
    elif p.type == "bool":
        if not isinstance(value, bool):
            return "参数类型或取值范围不正确"
    elif p.type == "enum":
        if value not in p.enum_values:
            return "参数类型或取值范围不正确"
    elif p.type == "string":
        if not isinstance(value, str):
            return "参数类型或取值范围不正确"
    return None


_PROGRESS = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)")
_LOSS = re.compile(r"(?:avr_loss|avg_loss|loss)[=:]\s*([0-9.eE+-]+)")
_SPEED = re.compile(r"([0-9.]+)it/s")


def parse_progress(line: str) -> tuple[TrainingEvent, ...]:
    """Parse a tqdm progress line shared by kohya sd-scripts and musubi-tuner.

    Neither engine's per-step output is guessed; only explicit values are read.
    """
    progress = _PROGRESS.search(line)
    if not progress:
        return ()
    payload: dict[str, object] = {"step": int(progress[1]), "total_steps": int(progress[2])}
    loss = _LOSS.search(line)
    if loss:
        try:
            value = float(loss[1])
            if math.isfinite(value):
                payload["loss"] = value
        except ValueError:
            pass
    speed = _SPEED.search(line)
    if speed:
        payload["it_per_sec"] = float(speed[1])
    return (TrainingEvent("progress", payload),)
