"""Audit an engine's real CLI surface against the parameters LP declares.

Dev tool. For every architecture of an installed engine it runs the training
script's ``--help`` with that engine's own interpreter and diffs the flags it
accepts against the adapter's parameter catalog. Gaps then show up as data
instead of being found one at a time in the UI.

    python_runtime\\python.exe scripts\\audit_params.py            # every engine
    python_runtime\\python.exe scripts\\audit_params.py kohya      # one engine

Two lists per architecture:

* **未提供** — the engine accepts the flag but the adapter declares nothing, so
  the UI cannot set it (a missing feature).
* **多余** — the adapter would pass the flag but the engine does not accept it,
  so submitting that configuration fails (a bug: fix these first).
"""
from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.registry import ADAPTERS  # noqa: E402  (import needs the path above)

FLAG = re.compile(r"--([a-zA-Z0-9_][a-zA-Z0-9_-]*)")
ENV_PYTHONS = (
    "venv/Scripts/python.exe", ".venv/Scripts/python.exe",
    "venv/bin/python", ".venv/bin/python",
)
# Present in every console script, or deliberately not exposed by LP.
IGNORED_FLAGS = {"help", "version"}
MISSING_PREVIEW = 40


def engine_python(engine_dir: Path) -> Path | None:
    for relative in ENV_PYTHONS:
        candidate = engine_dir / relative
        if candidate.is_file():
            return candidate
    return None


def find_script(engine_dir: Path, name: str) -> Path | None:
    """Resolve a script name for the layouts the adapters support."""
    for candidate in (engine_dir / name, engine_dir / "sd-scripts" / name,
                      engine_dir / "src" / "musubi_tuner" / name):
        if candidate.is_file():
            return candidate
    for candidate in engine_dir.rglob(name):
        if "build" not in candidate.parts and ".lp" not in candidate.parts:
            return candidate
    return None


def help_flags(python: Path, script: Path) -> tuple[set[str] | None, str]:
    """Flags the script accepts, or ``(None, reason)`` when it cannot be asked."""
    try:
        proc = subprocess.run(
            [str(python), str(script), "--help"], capture_output=True, text=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"无法运行：{exc}"
    text = (proc.stdout or "") + (proc.stderr or "")
    flags = set(FLAG.findall(text)) - IGNORED_FLAGS
    if not flags:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "无输出")
        return None, f"--help 未列出参数（{first[:70]}）"
    return flags, ""


def declared_flags(params, architecture: str) -> set[str]:
    """Flags LP would pass for one architecture (the same rule the UI filters by)."""
    keys = set()
    for param in params:
        if param.architectures and architecture not in param.architectures:
            continue
        if not param.native:
            continue  # LP-managed or display-only: never a CLI flag
        keys.add(param.native_key or param.key)
    return keys


def params_module(engine_id: str):
    try:
        module = importlib.import_module(f"adapters.{engine_id}.params")
    except ImportError:
        return ()
    return getattr(module, "PARAMS", ())


def audit(engine_id: str, engine_dir: Path) -> int:
    adapter = ADAPTERS[engine_id]
    python = engine_python(engine_dir)
    print(f"\n=== {engine_id}  ({engine_dir})")
    if python is None:
        print("  跳过：该目录下没有引擎自己的解释器（环境未安装？）")
        return 0
    params = params_module(engine_id)
    if not params:
        print("  跳过：适配器没有声明参数目录")
        return 0
    extra_total = 0
    for architecture in adapter.architectures(engine_dir):
        script = find_script(engine_dir, architecture.script)
        if script is None:
            print(f"  {architecture.id:<18} 跳过：找不到 {architecture.script}")
            continue
        flags, note = help_flags(python, script)
        if flags is None:
            print(f"  {architecture.id:<18} {note}")
            continue
        declared = declared_flags(params, architecture.id)
        missing = sorted(flags - declared)
        extra = sorted(declared - flags)
        extra_total += len(extra)
        print(f"  {architecture.id:<18} 引擎 {len(flags):>3} / LP {len(declared):>3} 参数"
              f" | 未提供 {len(missing):>3} | 多余 {len(extra):>2}"
              + ("   ← 提交会失败" if extra else ""))
        if extra:
            print(f"      多余: {', '.join('--' + name for name in extra)}")
        if missing:
            shown = ", ".join("--" + name for name in missing[:MISSING_PREVIEW])
            more = f" …（另 {len(missing) - MISSING_PREVIEW} 个）" if len(missing) > MISSING_PREVIEW else ""
            print(f"      未提供: {shown}{more}")
    return extra_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比引擎 CLI 与 LP 参数目录")
    parser.add_argument("engines", nargs="*", help="引擎类型；默认全部已适配引擎")
    parser.add_argument("--engine-root", type=Path, default=ROOT / "engine")
    args = parser.parse_args(argv)
    wanted = args.engines or list(ADAPTERS)
    total = 0
    for engine_id in wanted:
        if engine_id not in ADAPTERS:
            print(f"未知引擎类型 {engine_id}（已适配：{', '.join(ADAPTERS)}）")
            continue
        if not args.engine_root.is_dir():
            print(f"引擎目录不存在：{args.engine_root}")
            return 1
        for directory in sorted(p for p in args.engine_root.iterdir() if p.is_dir()):
            if ADAPTERS[engine_id].detect(directory).matched:
                total += audit(engine_id, directory)
    print(f"\n结论：适配器会传、但引擎不接受的参数共 {total} 个 —— 这几个要优先修（提交即报错）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
