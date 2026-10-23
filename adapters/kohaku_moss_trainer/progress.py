"""Progress lines of KohakuMossTrainer's log callbacks, read from their own wording.

    [23:15:21] [training.callbacks.monitor] [INFO] step 1 [loss=0.2752, grad_norm=0.02899, lr=1e-06]
    [23:15:41] [training.callbacks.kernels] [INFO] step 1 18.371 s/step, steps 2-10 median 1.806 s/step (10.2x)

No value is guessed: a payload is produced only from fields the line actually prints.
"""

import math
import re

from adapters.capabilities import TrainingEvent

_STEP = re.compile(r"\bstep (\d+)(?:/(\d+))?\b")
_RANGE = re.compile(r"training from step (\d+) to (\d+)")
_LOSS = re.compile(r"\bloss=([-+0-9.eE]+)")
_LR = re.compile(r"\blr=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_MEDIAN = re.compile(r"median ([-+0-9.eE]+) s/step")
_SINGLE = re.compile(r"\bstep \d+ ([-+0-9.eE]+) s/step")


def parse_progress(line):
    """One training line -> a progress event, or () when it carries no measurement."""
    payload = {}
    # `training from step 0 to 1500` announces the run: it is the only total it prints.
    span = _RANGE.search(line)
    if span:
        payload["step"] = int(span[1])
        payload["total_steps"] = int(span[2])
    step = _STEP.search(line)
    if step:
        payload["step"] = int(step[1])
        if step[2]:
            payload["total_steps"] = int(step[2])
    loss = _LOSS.search(line)
    if loss:
        value = float(loss[1])
        if math.isfinite(value):
            payload["loss"] = value
    lr = _LR.search(line)
    if lr:
        value = float(lr[1])
        if math.isfinite(value):
            payload["lr"] = value
    # The median of the last window is the honest rate: step 1 includes kernel tuning.
    speed = _MEDIAN.search(line) or _SINGLE.search(line)
    if speed:
        value = float(speed[1])
        if value > 0:
            payload["it_per_sec"] = 1 / value
    return (TrainingEvent("progress", payload),) if payload else ()
