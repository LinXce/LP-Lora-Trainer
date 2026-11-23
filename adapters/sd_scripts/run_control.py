"""Send sd-scripts' step-boundary stop command to this task's running trainer."""
import json
from pathlib import Path
import time
import tomllib

from app.services.common import atomic_text


def stop_command(task, argv):
    if "--run_control" not in argv or not task.get("output_verified") or not task.get("pid"):
        return False
    try:
        config_path = Path(argv[argv.index("--config_file") + 1])
        with config_path.open("rb") as f:
            config = tomllib.load(f)
        root = Path(task["output_dir"])
        if not root.is_absolute() or root != root.resolve() or Path(config["output_dir"]) != root:
            return False
        name = config["output_name"]
        if not name or Path(name).name != name or name in {".", ".."}:
            return False
        run_dir = root / (name + "_run")
        status_path = run_dir / "status.json"
        commands = run_dir / "commands"
        if status_path.resolve() != status_path or commands.resolve() != commands:
            return False
        # The trainer publishes its first status after expiring stale commands.
        # An early or foreign status must never enqueue a stop for another run.
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict) or status.get("pid") != task["pid"]:
            return False
        if status.get("state") not in {"starting", "running", "paused"}:
            return False
        atomic_text(commands / f"{time.time_ns()}_stop.json",
                    json.dumps({"command": "stop", "save": True, "save_state": True}))
        return True
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError):
        return False
