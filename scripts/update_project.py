"""Fast-forward the project from its git remote without touching user data.

Only tracked source files are updated. ``data/``, ``output/``, ``engine/*``,
``runtime/``, ``python_runtime/`` and ``frontend/dist/`` are ignored by git, so a
pull cannot overwrite datasets, task outputs, engine sources or installed runtimes.

Usage:
    python scripts/update_project.py [--check] [--rebuild] [--yes]

    --check    report whether an update is available, change nothing
    --rebuild  rebuild the frontend (and the launcher when its sources changed)
    --yes      do not ask for confirmation

Exit codes: 0 up to date or updated, 1 refused (dirty tree, no remote, failed pull).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths git never manages here. Listed for the report and for the dirty-tree check,
# so a user can see up front what an update will not touch.
PROTECTED = (
    "data/",
    "output/",
    "engine/",
    "runtime/",
    "python_runtime/",
    "frontend/dist/",
    "frontend/node_modules/",
    ".venv/",
    "LP-Lora-Trainer.exe",
)


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=check,
    )


def git(*args: str, check: bool = True) -> str:
    completed = run(["git", *args], check=check)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "git failed").strip())
    return completed.stdout.strip()


def ensure_repository() -> None:
    probe = run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        raise RuntimeError("当前目录不是 git 工作区：请用完整源码目录，或直接下载新的发行包")
    if not git("remote"):
        raise RuntimeError("没有配置 git 远端：无法检查更新")


def dirty_tracked() -> list[str]:
    """Tracked files with local edits. Ignored user data is deliberately not listed."""
    changed = []
    for line in git("status", "--porcelain").splitlines():
        # " M path" / "M  path" / "MM path" — untracked ("??") and ignored ("!!") are
        # not tracked changes and must never block an update.
        if not line.strip() or line.startswith(("??", "!!")):
            continue
        changed.append(line[3:].strip())
    return changed


def current_branch() -> str:
    branch = git("symbolic-ref", "--short", "-q", "HEAD", check=False)
    return branch or ""


def upstream(branch: str) -> str:
    remote = git("remote").splitlines()[0].strip() if git("remote") else "origin"
    return f"{remote}/{branch}"


def behind_count(upstream_ref: str) -> int:
    count = git("rev-list", "--count", f"HEAD..{upstream_ref}")
    return int(count or 0)


def changed_paths(upstream_ref: str) -> list[str]:
    output = git("diff", "--name-only", f"HEAD..{upstream_ref}")
    return [line for line in output.splitlines() if line]


def summarize(paths: list[str]) -> list[str]:
    notes = []
    if any(p.startswith("frontend/src/") or p.startswith("frontend/package") for p in paths):
        notes.append("前端源码有变化 → 需要重建 frontend/dist（npm run build）")
    if any(p.startswith("desktop/") and p.endswith(".cs") for p in paths):
        notes.append("桌面启动器源码有变化 → 需要重新编译 LP-Lora-Trainer.exe")
    if any(p.startswith("python_runtime/") for p in paths):
        notes.append("打包运行时来源有变化 → 需要重建 python_runtime")
    return notes


def rebuild(frontend: bool, launcher: bool) -> None:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    if frontend:
        print("\n[build] npm ci && npm run build")
        run([npm, "ci"], check=True)
        run([npm, "run", "build"], check=True)
    if launcher:
        print("\n[build] scripts/build_launcher.ps1")
        run(["powershell", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(PROJECT_ROOT / "scripts" / "build_launcher.ps1")], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="只检查是否有更新")
    parser.add_argument("--rebuild", action="store_true", help="更新后重建前端/启动器")
    parser.add_argument("--yes", action="store_true", help="不询问确认")
    args = parser.parse_args()

    try:
        ensure_repository()
    except RuntimeError as exc:
        print(f"[update] {exc}", file=sys.stderr)
        return 1

    branch = current_branch()
    if not branch:
        print("[update] 当前处于游离 HEAD 状态，请先切回分支（如 main）", file=sys.stderr)
        return 1

    print(f"[update] 分支 {branch}，正在获取远端信息…")
    try:
        git("fetch", "--prune")
    except RuntimeError as exc:
        print(f"[update] 获取远端失败：{exc}", file=sys.stderr)
        return 1

    upstream_ref = upstream(branch)
    ahead_check = run(["git", "rev-parse", "--verify", "--quiet", upstream_ref], check=False)
    if ahead_check.returncode != 0:
        print(f"[update] 分支 {branch} 没有对应的远端分支 {upstream_ref}", file=sys.stderr)
        return 1

    pending = behind_count(upstream_ref)
    if pending == 0:
        print(f"[update] 已是最新（{branch} 与 {upstream_ref} 一致）")
        return 0

    paths = changed_paths(upstream_ref)
    print(f"[update] 有 {pending} 个新提交，涉及 {len(paths)} 个文件：")
    for line in summarize(paths):
        print(f"  · {line}")
    print("\n[update] 以下目录由 git 忽略，更新不会改动它们：")
    for name in PROTECTED:
        print(f"  · {name}")

    if args.check:
        print("\n[update] --check 模式：未做任何修改")
        return 0

    dirty = dirty_tracked()
    if dirty:
        print("\n[update] 已跟踪的源码有本地修改，为避免覆盖已停止：", file=sys.stderr)
        for name in dirty[:20]:
            print(f"  ✗ {name}", file=sys.stderr)
        if len(dirty) > 20:
            print(f"  … 共 {len(dirty)} 个", file=sys.stderr)
        print("  请先提交、还原或挪走这些改动后重试（用户数据目录不在上面的列表里）", file=sys.stderr)
        return 1

    if not args.yes:
        answer = input("\n继续更新？[y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("[update] 已取消")
            return 0

    print(f"\n[update] git merge --ff-only {upstream_ref}")
    merge = run(["git", "merge", "--ff-only", upstream_ref], check=False)
    if merge.returncode != 0:
        print((merge.stderr or merge.stdout).strip(), file=sys.stderr)
        print("[update] 快进合并失败：本地提交与远端分叉，请手动处理后再更新", file=sys.stderr)
        return 1
    print(merge.stdout.strip())

    if args.rebuild:
        rebuild(
            frontend=any(p.startswith("frontend/") for p in paths),
            launcher=any(p.startswith("desktop/") and p.endswith(".cs") for p in paths),
        )

    print("\n[update] 完成。请关闭 LP 窗口后重新打开；后台服务如果仍在运行，请先在设置页退出后端与 supervisor。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
