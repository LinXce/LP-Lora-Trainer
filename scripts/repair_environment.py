"""Offline repair for this project's copied Windows .venv; no engine/data edits."""
import ast
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys


def rebase_path(value, old_root, new_root):
    """Rebase only paths under the known previous editable-install root."""
    path = Path(value)
    try:
        suffix = path.relative_to(old_root)
    except ValueError as exc:
        raise ValueError(f"Unexpected editable path outside the old project: {value}") from exc
    return str(new_root / suffix)


def repair_finder(source, new_root):
    assignments = {}
    for node in ast.parse(source).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id in {"MAPPING", "NAMESPACES"}:
            assignments[target.id] = (node, ast.literal_eval(node.value))
    if "MAPPING" not in assignments or "NAMESPACES" not in assignments:
        raise ValueError("Unsupported editable-install finder; recreate the application environment")
    mapping = assignments["MAPPING"][1]
    if "app" not in mapping:
        raise ValueError("This finder does not belong to LP LoRA Trainer")
    old_root = Path(mapping["app"]).parent
    updated = {key: rebase_path(value, old_root, new_root) for key, value in mapping.items()}
    if any(not Path(value).is_dir() for value in updated.values()):
        raise ValueError("Project packages are missing at the new location")
    namespaces = {
        key: [rebase_path(value, old_root, new_root) for value in values]
        for key, values in assignments["NAMESPACES"][1].items()
    }
    for key, value in (("MAPPING", updated), ("NAMESPACES", namespaces)):
        node = assignments[key][0]
        original = ast.get_source_segment(source, node.value)
        # Replace only the literals; preserve generated import machinery.
        source = source.replace(original, repr(value), 1)
    return source, old_root



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


def repair(workspace):
    import venv
    from pip._vendor.distlib.scripts import ScriptMaker

    if os.name != "nt":
        raise RuntimeError("This repair tool is for the project's Windows .venv only")
    workspace = workspace.resolve()
    environment = workspace / ".venv"
    if Path(sys.prefix).resolve() != environment or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with the project's .venv\\Scripts\\python.exe")
    if not Path(sys._base_executable).is_file():
        raise RuntimeError("The base Python is missing; recreate .venv with Python 3.11+")
    site = environment / "Lib" / "site-packages"
    finders = list(site.glob("__editable___lp_lora_trainer_*_finder.py"))
    metadata = list(site.glob("lp_lora_trainer-*.dist-info"))
    if len(finders) != 1 or len(metadata) != 1:
        raise RuntimeError("Expected one editable installation of LP LoRA Trainer")
    finder = finders[0]
    repaired, old_root = repair_finder(finder.read_text(encoding="utf-8"), workspace)
    url_path = metadata[0] / "direct_url.json"
    url = json.loads(url_path.read_text(encoding="utf-8"))
    if not url.get("dir_info", {}).get("editable"):
        raise RuntimeError("The project must be installed in editable mode")
    url["url"] = workspace.as_uri()
    cfg_path = environment / "pyvenv.cfg"
    cfg = cfg_path.read_text(encoding="utf-8")
    # Preserve home, base interpreter, prompt and system-site-packages settings.
    cfg = re.sub(r"(?m)^command = .*", lambda _: f"command = {sys._base_executable} -m venv {environment}", cfg)

    entry_points = []
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        for entry in distribution.entry_points:
            if entry.group not in {"console_scripts", "gui_scripts"}:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", entry.name):
                raise RuntimeError(f"Unsupported command name: {entry.name}")
            entry_points.append(entry)

    # Keep first originals for troubleshooting. No dependency downloads/removal.
    for path in (finder, url_path, cfg_path):
        backup = path.with_name(path.name + ".before-relocation")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    finder.write_text(repaired, encoding="utf-8")
    url_path.write_text(json.dumps(url) + "\n", encoding="utf-8")
    cfg_path.write_text(cfg, encoding="utf-8")

    builder = venv.EnvBuilder(with_pip=False)
    context = builder.ensure_directories(str(environment))
    builder.setup_scripts(context)
    # Regenerate native launchers rather than replacing strings in EXE binaries.
    maker = ScriptMaker(None, str(environment / "Scripts"))
    maker.executable = str(environment / "Scripts" / "python.exe")
    maker.variants = {""}
    maker.clobber = True
    changed_paths = [finder, url_path]
    for entry in entry_points:
        options = {"gui": entry.group == "gui_scripts"}
        changed_paths.extend(maker.make(f"{entry.name} = {entry.value}", options))
        if entry.name == "pip":
            changed_paths.extend(maker.make(f"pip{sys.version_info.major}.{sys.version_info.minor} = {entry.value}"))
    refresh_records(site, changed_paths)
    print(f"Project: {workspace}")
    print(f"Previous editable root: {old_root}")
    print(f"Repaired activation, editable paths and {len(entry_points)} command entries.")
    print("Dependencies, engine sources and application data were not changed.")


if __name__ == "__main__":
    try:
        repair(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"Environment repair failed: {exc}")
