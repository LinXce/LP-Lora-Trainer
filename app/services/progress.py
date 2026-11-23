"""Merge explicit training measurements for both the live worker and log replay."""

LOG_PROGRESS_VERSION = 2
PROGRESS = dict.fromkeys(("step", "total_steps", "epoch", "total_epochs", "loss", "lr", "it_per_sec", "eta_seconds"))


def merge_progress(progress, payload, kind="progress"):
    step = payload.get("step")
    if kind == "remaining_steps" and step is not None:
        # These trainers start tqdm at zero when continuing. Its denominator
        # counts the remaining updates; startup output gives the global target.
        span = payload["total_steps"]
        total = progress.get("total_steps")
        offset = max(0, total - span) if total is not None else progress.get("resume_step", 0)
        step += offset
        payload = {**payload, "step": step, "total_steps": span + offset}
    if step is not None and progress.get("step") is not None and step < progress["step"]:
        return False
    progress.update(payload)
    return True
