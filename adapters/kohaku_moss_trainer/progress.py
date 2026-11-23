"""Progress lines of KohakuMossTrainer's log callbacks, read from their own wording.

    [23:15:21] [training.callbacks.monitor] [INFO] step 1 [loss=0.2752, grad_norm=0.02899, lr=1e-06]
    [23:15:41] [training.callbacks.kernels] [INFO] step 1 18.371 s/step, steps 2-10 median 1.806 s/step (10.2x)

No value is guessed: a payload is produced only from fields the line actually prints.
"""

import re

from adapters.capabilities import NUMBER, TrainingEvent, read_number, training_rate

_RANGE = re.compile(r"training from step (\d+) to (\d+)")
_STEP = re.compile(r"step (\d+)\s+\[")
_LOSS = re.compile(r"\bloss=(" + NUMBER + r")(?=[,\s\]])")
_LR = re.compile(r"\blr=(" + NUMBER + r")(?=[,\s\]])")
_TIMING = re.compile(r"step \d+ (" + NUMBER + r"\s+s/step)(?:, steps \d+-\d+ median (" + NUMBER + r"\s+s/step))?")
_PREFIX = re.compile(r"^\[[^\]]+\]\s+\[(training\.(?:entry|callbacks\.(?:kernels|monitor)))\]\s+\[INFO\]\s+")


def parse_progress(line):
    """Read training callback messages, excluding sampling, inference and kernel diagnostics."""
    text = _PREFIX.sub("", line.strip(), count=1) if line.lstrip().startswith("[") else line.strip()
    span = _RANGE.fullmatch(text)
    if span:
        return (TrainingEvent("progress", {"step": int(span[1]), "total_steps": int(span[2])}),)
    timing = _TIMING.match(text)
    if timing:
        # 'step 1 ..., steps 2-10 median ...' is a retrospective measurement
        # relative to this process, even when it resumed from global step 1500.
        rate = training_rate(timing[2] or timing[1])
        return (TrainingEvent("progress", {"it_per_sec": rate}),) if rate is not None else ()
    step = _STEP.match(text)
    if not step:
        return ()
    payload = {"step": int(step[1])}
    for key, pattern in (("loss", _LOSS), ("lr", _LR)):
        value = read_number(pattern.search(text))
        if value is not None:
            payload[key] = value
    return (TrainingEvent("progress", payload),)
