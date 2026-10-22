"""Save-point discovery shared by the engines that use Accelerate state dirs.

Anima, sd-scripts and Musubi Tuner all write ``accelerate.save_state`` directories
named ``<run>-state`` / ``<run>-step00000100-state`` / ``<run>-000003-state`` next to
their weights. The directory only counts as resumable when the Accelerate state
files are present, so a half-written or unrelated folder is never offered.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapters.base import ResumeSource

# ``accelerate.save_state`` writes the model weights and the RNG/step pickle into
# every state directory. Both are required to resume: without the RNG file
# Accelerate cannot restore the step/RNG state, and without the weights there is
# nothing to continue from. Optimizer/scheduler files are also written but
# Accelerate tolerates their absence, so they are not part of the completeness test.
_RNG_FILE = re.compile(r"^random_states_\d+\.pkl$")
_MODEL_FILE = re.compile(r"^(?:pytorch_model|model)(?:_\d+)?\.(?:bin|safetensors)$")

_STEP = re.compile(r"-step(?P<step>\d+)-state$")
_EPOCH = re.compile(r"-(?P<epoch>\d{6})-state$")


def is_accelerate_state(directory: Path) -> bool:
    """True when ``directory`` is a complete ``accelerate.save_state`` directory."""
    if not directory.is_dir():
        return False
    names = [entry.name for entry in directory.iterdir()]
    return (any(_RNG_FILE.match(name) for name in names)
            and any(_MODEL_FILE.match(name) for name in names))


def state_label(name: str) -> tuple[int | None, str]:
    """``(step, label)`` for a state directory name; step is None for epoch/last."""
    match = _STEP.search(name)
    if match:
        step = int(match.group("step"))
        return step, f"第 {step} 步的状态"
    match = _EPOCH.search(name)
    if match:
        return None, f"第 {int(match.group('epoch'))} 轮的状态"
    if name.endswith("-checkpoint-state"):
        return None, "上次停止时保存的状态"
    return None, "训练结束时的状态"


def accelerate_resume_sources(root: Path) -> tuple[ResumeSource, ...]:
    """Every complete Accelerate state directory under ``root``, newest first."""
    root = Path(root)
    if not root.is_dir():
        return ()
    found = []
    for path in root.rglob("*-state"):
        if not is_accelerate_state(path):
            continue
        step, label = state_label(path.name)
        found.append(ResumeSource(path=path, kind="state", label=label, step=step))
    # Most recently written save point first; the step breaks ties when a run
    # writes an epoch and its step state within the same timestamp tick.
    found.sort(key=lambda source: (source.path.stat().st_mtime, source.step or -1), reverse=True)
    return tuple(found)
