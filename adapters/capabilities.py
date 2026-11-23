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
    # Presentation of `help` in the form: "" (muted) or "info" (highlighted).
    help_style: str = ""
    unsupported_reason: str = ""
    # Extra tabs for the same value, in addition to the primary group.
    groups: tuple[str, ...] = ()
    # Lower values are displayed earlier in the parameter form.
    display_order: int = 100
    # Architecture ids for which the parameter must be supplied before launch.
    required_for: tuple[str, ...] = ()
    # (key, values): the parameter only applies when draft.params[key] is one of values.
    # Used by engines whose algorithms change what a knob means (or drop it entirely).
    visible_when: tuple[str, tuple[str, ...]] = ()
    # A newline-backed set of resolution pairs; presets share the editable value.
    resolution_presets: tuple[int, ...] = ()
    # Local paths are adapter-declared, never guessed from labels or key names.
    path_kind: str = ""
    file_types: tuple[str, ...] = ()
    # Long-form explanation shown in the side drawer behind the ⓘ button: what the
    # knob does, which value shape it takes, and worked examples.
    detail: str = ""

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
            native=self.native,
            display_order=self.display_order,
            default=self.default,
            options=[dict(value=value, label=label) for value, label in self.options],
        )
        if self.path_kind:
            out["path_kind"] = self.path_kind
        if self.file_types:
            out["file_types"] = list(self.file_types)
        if self.resolution_presets:
            out["resolution_presets"] = list(self.resolution_presets)
        if self.groups:
            out["groups"] = list(self.groups)
        if self.visible_when:
            key, values = self.visible_when
            out["visible_when"] = {"key": key, "values": list(values)}
        if self.block:
            out["block"] = self.block
        if self.native_key:
            out["native_key"] = self.native_key
        if self.help_style:
            out["help_style"] = self.help_style
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.help:
            out["help"] = self.help
        if self.detail:
            out["detail"] = self.detail
        if self.unsupported_reason:
            out["unsupported_reason"] = self.unsupported_reason
        if self.list_arg:
            out["list_arg"] = True
        if self.multiline or self.list_arg:
            out["multiline"] = True
        if self.architectures:
            out["architectures"] = list(self.architectures)
        if self.required_for:
            out["required_for"] = list(self.required_for)
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
    if p.key == "sample_prompts":
        from adapters.sampling import prompt_error
        return prompt_error(value)
    if value is None:
        return None
    if p.type in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{p.label} 需要数字，当前不是数字"
        if isinstance(value, float) and not math.isfinite(value):
            return f"{p.label} 必须是有限数值"
        if p.type == "int" and not isinstance(value, int):
            return f"{p.label} 需要整数，当前是 {value}"
        if p.min is not None and value < p.min:
            return f"{p.label} 小于允许的最小值 {p.min}（当前 {value}）"
        if p.max is not None and value > p.max:
            return f"{p.label} 大于允许的最大值 {p.max}（当前 {value}）"
    elif p.type == "bool":
        if not isinstance(value, bool):
            return f"{p.label} 需要开关值（true/false）"
    elif p.type == "enum":
        if value not in p.enum_values:
            return f"{p.label} 不在可选范围内：{'、'.join(str(v) for v in p.enum_values[:8])}"
    elif p.type == "string":
        if not isinstance(value, str):
            return f"{p.label} 需要文本值"
    return None


def form_value_errors(specs, values):
    """Check the selected installation's effective field types before dependent rules."""
    for spec in specs:
        key = spec["key"]
        if key not in values:
            continue
        p = Param(key, spec["label"], spec["type"], min=spec.get("min"), max=spec.get("max"),
                  options=tuple((o["value"], o["label"]) for o in spec.get("options", ())))
        value = values[key]
        if spec.get("unsupported_reason") and value not in (None, "", spec.get("default")):
            yield key, spec["unsupported_reason"]
        elif message := check_value(p, value):
            yield key, message


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAIN_BAR = re.compile(r"^(steps|train):\s*\d+%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)\s*\[")
_EPOCH = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)\b")
_TRAIN_TOTAL = re.compile(r"(num epochs|total optimization steps)(?:\s*/[^:=]*)?\s*[:=]\s*(\d+)$")
_LOSS = re.compile(r"\b(?:avr_loss|avg_loss|loss)[=:]\s*(" + NUMBER + r")(?=[,\s\]])")
_LR = re.compile(r"\blr[=:]\s*(" + NUMBER + r")(?=[,\s\]])")
_RATE = re.compile(r"(?<![\w.])(" + NUMBER + r")\s*(it/s|step/s|s/it|s/step)\b")
_POSTFIX_SECONDS = re.compile(r"\bs/it=(" + NUMBER + r")(?=[,\s\]])")
_ETA = re.compile(r"(?:\beta\s+|\[\d+(?::\d{2}){1,2}<)(?:(\d+):)?(\d+):(\d{2})\b")


def plain_log(line: str) -> str:
    return _ANSI.sub("", line).strip()


def parse_eta(text: str) -> int | None:
    """Read MOSS 'eta H:MM:SS' or tqdm's '[elapsed<remaining, rate]'."""
    match = _ETA.search(text)
    if not match:
        return None
    return int(match[1] or 0) * 3600 + int(match[2]) * 60 + int(match[3])


def read_number(match: re.Match | None) -> float | None:
    if match:
        value = float(match[1])
        if math.isfinite(value):
            return value
    return None


def training_rate(text: str) -> float | None:
    """Convert a reported duration/rate into iterations per second."""
    match = _RATE.search(text)
    value = read_number(match)
    if value is not None and value > 0 and match[2] in ("it/s", "step/s"):
        return value
    if value is None or value <= 0:
        value = read_number(_POSTFIX_SECONDS.search(text))
    if value is not None and value > 0:
        rate = 1 / value
        return rate if math.isfinite(rate) else None
    return None


def epoch_counts(match: re.Match) -> dict[str, int]:
    return {"epoch": int(match[1]), "total_epochs": int(match[2])}


def parse_tqdm(text: str, label: str, *, kind: str = "progress") -> tuple[TrainingEvent, ...]:
    # Loading weights, caching latents and generating samples also draw full tqdm
    # bars. Each supported training loop sets an explicit, stable description.
    progress = _TRAIN_BAR.match(text)
    if not progress or progress[1] != label:
        return ()
    payload = {"step": int(progress[2]), "total_steps": int(progress[3])}
    for key, value in (("loss", read_number(_LOSS.search(text))),
                       ("lr", read_number(_LR.search(text))),
                       ("it_per_sec", training_rate(text)), ("eta_seconds", parse_eta(text))):
        if value is not None:
            payload[key] = value
    return (TrainingEvent(kind, payload),)


def parse_progress(line: str, *, bar_kind: str = "progress") -> tuple[TrainingEvent, ...]:
    """Training announcements and steps bars shared by sd-scripts, musubi and Anima."""
    text = plain_log(line)
    epoch = _EPOCH.fullmatch(text)
    if epoch:
        return (TrainingEvent("progress", epoch_counts(epoch)),)
    total = _TRAIN_TOTAL.fullmatch(text)
    if total:
        key = "total_epochs" if total[1] == "num epochs" else "total_steps"
        return (TrainingEvent("progress", {key: int(total[2])}),)
    return parse_tqdm(text, "steps", kind=bar_kind)
