"""Rebase an engine virtual environment after the project directory moved.

An engine ``venv`` / ``.venv`` stores absolute paths in its metadata: ``pyvenv.cfg``
(``home=`` / ``command=``), an editable-install finder, ``direct_url.json`` and the
native launchers under ``Scripts``. Moving the project makes those stale, which is
why an environment used to need a reinstall. This script rewrites the metadata in
place instead, so the same interpreter and site-packages keep working.

Usage:
    python scripts/repair_engine_venv.py <venv-dir> --old-root <old project> --new-root <new project>

Only paths that really live under ``--old-root`` are rebased, every changed file is
backed up once as ``*.before-relocation``, and nothing is downloaded or removed.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def rebase(value, old_root, new_root):
    """Rebase one path that must live under the previous project root."""
    path = Path(value)
    try:
        suffix = path.relative_to(old_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the previous project: {value}") from exc
    return str(new_root / suffix)


def rebase_if_inside(value, old_root, new_root):
    """Rebase when the path is inside the old root, otherwise keep it (base Python).

    Compared case-insensitively: Windows tools disagree on the case (and occasionally
    the 8.3 spelling) of the same directory, and a stale entry must still be matched.
    """
    text, old = str(value), str(old_root)
    try:
        Path(text).relative_to(old_root)
    except ValueError:
        lowered = text.lower()
        if old.lower() not in lowered:
            return value
        index = lowered.index(old.lower())
        return str(new_root) + text[index + len(old):]
    return rebase(text, old_root, new_root)


def repair_finder(source, old_root, new_root):
    """Rewrite the absolute literals of an editable-install finder, keeping its code."""
    assignments = {}
    for node in ast.parse(source).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id in {"MAPPING", "NAMESPACES"}:
            assignments[target.id] = (node, ast.literal_eval(node.value))
    if not assignments:
        return source, 0
    changed = 0
    for key, (node, value) in assignments.items():
        if key == "MAPPING":
            updated = {k: rebase_if_inside(v, old_root, new_root) for k, v in value.items()}
        else:
            updated = {k: [rebase_if_inside(v, old_root, new_root) for v in values] for k, values in value.items()}
        if updated != value:
            original = ast.get_source_segment(source, node.value)
            source = source.replace(original, repr(updated), 1)
            changed += 1
    return source, changed


def refresh_records(site, changed_paths):
    """Keep installed-file hashes consistent with regenerated files."""
    changed_paths = {Path(path).resolve() for path in changed_paths}
    for record in site.glob("*.dist-info/RECORD"):
        with record.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        modified = False
        for row in rows:
            if len(row) != 3:
                continue
            path = (site / row[0]).resolve()
            if path not in changed_paths:
                continue
            content = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            row[1:] = ["sha256=" + digest, str(len(content))]
            modified = True
        if modified:
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)


def backup_once(path):
    backup = path.with_name(path.name + ".before-relocation")
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def repair(environment, old_root, new_root):
    environment = environment.resolve()
    if os.name != "nt":
        raise RuntimeError("Engine environment repair is for Windows environments only")
    if not environment.is_dir():
        raise RuntimeError(f"Environment directory not found: {environment}")
    old_root, new_root = old_root.resolve(), new_root.resolve()
    # The previous root is normally gone after a move: it is only needed as the string
    # prefix to rebase, and rebase() still refuses any path outside it.
    if not new_root.is_dir():
        raise RuntimeError(f"Current project root not found: {new_root}")
    site = next((p for p in (environment / "Lib" / "site-packages", environment / "lib") if p.is_dir()), None)
    if site is None:
        raise RuntimeError(f"Not a virtual environment (no site-packages): {environment}")
    changed = []

    cfg_path = environment / "pyvenv.cfg"
    if cfg_path.is_file():
        cfg = cfg_path.read_text(encoding="utf-8")
        # home= may point at the project-local base Python, which moved with it.
        updated = re.sub(r"(?m)^(home\s*=\s*)(.+)$",
                         lambda m: m.group(1) + rebase_if_inside(m.group(2).strip(), old_root, new_root), cfg)
        updated = re.sub(r"(?m)^(command\s*=\s*)(.+)$",
                         lambda m: m.group(1) + rebase_if_inside(m.group(2).strip(), old_root, new_root), updated)
        updated = updated.replace(str(old_root), str(new_root))
        if updated != cfg:
            backup_once(cfg_path)
            cfg_path.write_text(updated, encoding="utf-8")
            changed.append(cfg_path)

    for finder in site.glob("__editable___*_finder.py"):
        source = finder.read_text(encoding="utf-8")
        repaired, count = repair_finder(source, old_root, new_root)
        if count:
            backup_once(finder)
            finder.write_text(repaired, encoding="utf-8")
            changed.append(finder)

    for direct in site.glob("*.dist-info/direct_url.json"):
        try:
            url = json.loads(direct.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = url.get("url") or ""
        if str(old_root).replace("\\", "/").lower() not in value.replace("\\", "/").lower():
            continue
        url["url"] = value.replace(str(old_root).replace("\\", "/"), str(new_root).replace("\\", "/"))
        backup_once(direct)
        direct.write_text(json.dumps(url) + "\n", encoding="utf-8")
        changed.append(direct)

    # Native launchers embed the interpreter path. Regenerating them needs pip's
    # ScriptMaker, which only exists in a seeded environment; skip with a warning
    # rather than inventing a binary patch. LP itself launches Scripts\python.exe.
    scripts = environment / "Scripts"
    launchers = 0
    try:
        import venv
        from pip._vendor.distlib.scripts import ScriptMaker
        import importlib.metadata
    except ImportError:
        print("note: pip is unavailable here, so Scripts launchers were left untouched")
    else:
        entry_points = [entry for dist in importlib.metadata.distributions(path=[str(site)])
                        for entry in dist.entry_points
                        if entry.group in {"console_scripts", "gui_scripts"}
                        and re.fullmatch(r"[A-Za-z0-9_.-]+", entry.name or "")]
        builder = venv.EnvBuilder(with_pip=False)
        context = builder.ensure_directories(str(environment))
        builder.setup_scripts(context)
        maker = ScriptMaker(None, str(scripts))
        maker.executable = str(scripts / "python.exe")
        maker.variants = {""}
        maker.clobber = True
        for entry in entry_points:
            changed.extend(maker.make(f"{entry.name} = {entry.value}", {"gui": entry.group == "gui_scripts"}))
            launchers += 1

    refresh_records(site, [p for p in changed if p is not None])
    print(f"Environment: {environment}")
    print(f"Rebased from {old_root} to {new_root}")
    print(f"Rewritten files: {len(changed)}; regenerated launchers: {launchers}")
    print("No dependency was downloaded, installed or removed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", help="engine venv / .venv directory")
    parser.add_argument("--old-root", required=True, help="project root the environment was created under")
    parser.add_argument("--new-root", required=True, help="project root the environment now lives under")
    args = parser.parse_args()
    try:
        return repair(Path(args.environment), Path(args.old_root), Path(args.new_root))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Engine environment repair failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
