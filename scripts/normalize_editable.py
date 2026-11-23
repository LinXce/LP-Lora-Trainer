"""Make editable engine imports follow a copied checkout (embedded CPython or venv)."""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import url2pathname


def _site_packages(environment: Path) -> Path | None:
    """Ask the environment's own interpreter where its purelib is."""
    python = environment / ("python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        python = environment / "Scripts" / "python.exe"
    if not python.is_file():
        return None
    probe = subprocess.run(
        [str(python), "-I", "-X", "utf8", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    text = (probe.stdout or "").strip()
    return Path(text) if probe.returncode == 0 and text else None


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} and {base} must be on the same drive for portable imports") from exc


def _write_changed(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Cannot rewrite a linked editable hook: {path}")
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    # uv may have hardlinked a wheel's file out of its cache. Replace this
    # instance's directory entry without modifying another installation.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(content, encoding="utf-8")
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _editable_source(direct: Path) -> Path | None:
    data = json.loads(direct.read_text(encoding="utf-8"))
    if not data.get("dir_info", {}).get("editable"):
        return None
    url = urlsplit(data["url"])
    if url.scheme != "file":
        return None
    prefix = "//" + url.netloc if url.netloc and url.netloc.lower() != "localhost" else ""
    path = Path(url2pathname(prefix + url.path))
    return path if path.is_absolute() else None


@dataclass(frozen=True)
class EditableSource:
    recorded: Path
    current: Path

    def rebase(self, path: Path) -> Path:
        return self.current / path.relative_to(self.recorded) if path.is_relative_to(self.recorded) else path


def _locate_source(environment: Path, direct: Path, recorded: Path) -> EditableSource:
    checkout = environment.parent.parent
    if environment.parent.name == ".lp" and direct.is_relative_to(checkout):
        if not recorded.is_relative_to(checkout):
            # A copy may still have its original interpreter and PEP 610 metadata.
            # Match the actual old environment before redirecting its source.
            metadata = direct.relative_to(checkout)
            for ancestor in (recorded, *recorded.parents):
                old_info = ancestor / metadata
                if old_info.is_file() and _editable_source(old_info) == recorded:
                    current = checkout / recorded.relative_to(ancestor)
                    if current.is_dir():
                        return EditableSource(recorded, current)
        if not recorded.is_dir():
            # Some older engine archives lost one level of checkout directory.
            # Recover only when every declared top-level module exists locally.
            top_level = direct.parent / "top_level.txt"
            names = top_level.read_text(encoding="utf-8").split() if top_level.is_file() else []
            if names and all(name.isidentifier() for name in names):
                for root in (checkout, checkout / "src"):
                    if all((root / name).is_dir() or (root / (name + ".py")).is_file() for name in names):
                        return EditableSource(recorded, checkout)
    return EditableSource(recorded, recorded)


def _owned_hooks(info: Path) -> set[str]:
    """Only top-level .pth files in this editable wheel's RECORD belong to it."""
    record = info / "RECORD"
    if not record.is_file():
        return set()
    with record.open(encoding="utf-8", newline="") as stream:
        return {row[0] for row in csv.reader(stream)
                if row and row[0].endswith(".pth") and PureWindowsPath(row[0]).name == row[0]}


_FINDER_LINE = re.compile(r"import[ \t]+(?P<name>__editable__\w+_finder);\s*(?P=name)\.install\(\)\s*;?\s*")
_PATH_HELPER = "_lp_editable_path"
_HELPER_BODY = """def _lp_editable_path(relative):
    from pathlib import Path
    return str((Path(__file__).resolve().parent / relative).resolve())
"""


def _finder(finder: Path, origin: EditableSource | None) -> tuple[str, list[Path]]:
    """Keep setuptools' finder and namespaces intact. Rewrite just its path literals."""
    source = finder.read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = []
    changed = False
    first_mapping = None
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id in {"MAPPING", "NAMESPACES"} for t in targets):
            continue
        mapping = statement.value
        if not isinstance(mapping, ast.Dict):
            raise ValueError(f"Unsupported editable mapping in {finder}")
        if first_mapping is None:
            first_mapping = index
        for position, (key, value) in enumerate(zip(mapping.keys, mapping.values)):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(f"Missing module name in {finder}")
            items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            for i, item in enumerate(items):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    path = Path(item.value)
                    if not path.is_absolute():
                        raise ValueError(f"Expected an absolute editable path in {finder}: {path}")
                    if origin:
                        path = origin.rebase(path)
                    relative = _relative(path, finder.parent)
                    call = ast.Call(ast.Name(_PATH_HELPER, ast.Load()), [ast.Constant(relative)], [])
                    if isinstance(value, (ast.List, ast.Tuple)):
                        value.elts[i] = call
                    else:
                        mapping.values[position] = call
                    changed = True
                elif (isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                      and item.func.id == _PATH_HELPER and len(item.args) == 1):
                    path = (finder.parent / ast.literal_eval(item.args[0])).resolve()
                else:
                    raise ValueError(f"Unknown editable path expression in {finder}")
                # A no-site interpreter can import ordinary packages and modules
                # by path alone. Never expose an arbitrary ancestor for an alias.
                package = path.with_suffix("") if path.suffix == ".py" else path
                parts = key.value.split(".")
                if tuple(parts) == package.parts[-len(parts):]:
                    roots.append(package.parents[len(parts) - 1])
    if first_mapping is None:
        raise ValueError(f"Missing setuptools editable mapping in {finder}")
    if changed:
        if not any(isinstance(node, ast.FunctionDef) and node.name == _PATH_HELPER for node in tree.body):
            tree.body.insert(first_mapping, ast.parse(_HELPER_BODY).body[0])
        source = ast.unparse(tree) + "\n"
    return source, roots


def _fallback(source: Path) -> Path:
    # Older LP releases removed both the wheel's hook and its generated finder.
    if not source.is_dir():
        raise ValueError(f"Cannot recover missing editable hook for {source}. Reinstall the engine package.")
    return source / "src" if (source / "src").is_dir() else source


def _rewrite_hook(hook: Path, origin: EditableSource | None, writes: dict[Path, str]) -> list[Path]:
    roots = []
    lines = []
    if not hook.exists():
        if origin is None:
            return roots
        original = str(_fallback(origin.current))
    else:
        original = hook.read_text(encoding="utf-8-sig")
    for line in original.splitlines():
        stripped = line.strip()
        match = _FINDER_LINE.fullmatch(stripped)
        if match:
            finder = hook.parent / (match["name"] + ".py")
            if finder.is_file():
                writes[finder], finder_roots = _finder(finder, origin)
                roots.extend(finder_roots)
            elif origin is not None:
                recovered = _fallback(origin.current)
                roots.append(recovered)
                line = _relative(recovered, hook.parent)
            else:
                raise ValueError(f"Cannot recover missing finder {finder}. Reinstall the engine package.")
        elif stripped and not stripped.startswith(("#", "import ", "import\t")):
            path = Path(stripped)
            if path.is_absolute():
                if origin:
                    path = origin.rebase(path)
                line = _relative(path, hook.parent)
            else:
                path = (hook.parent / path).resolve()
            roots.append(path)
        lines.append(line)
    writes[hook] = "\n".join(lines) + "\n"
    return roots


def _rewrite_embedded_pth(pth: Path, environment: Path, entries: list[str]) -> None:
    lines = []
    generated = False
    for line in pth.read_text(encoding="utf-8").splitlines():
        if line == "# LP editable: begin":
            generated = True
        elif line == "# LP editable: end":
            generated = False
        elif not generated:
            # ._pth only supports full-line comments. Repair legacy inline markers.
            if "# LP editable:" in line:
                line = line.partition("# LP editable:")[0].strip()
                if not line or line in lines:
                    continue
            lines.append(line)
    extra = [entry for entry in dict.fromkeys(entries) if entry not in lines]
    if extra:
        lines.extend(["# LP editable: begin", *extra, "# LP editable: end"])
    _write_changed(pth, "\n".join(lines) + "\n")


def normalize(environment: Path, extra: tuple[Path, ...] = ()) -> int:
    environment = environment.resolve()
    site = _site_packages(environment)
    if site is None or not site.is_dir():
        print(f"[LP] No site-packages found under {environment}", file=sys.stderr)
        return 1

    owned: dict[Path, EditableSource | None] = {}
    for direct in sorted(site.glob("*.dist-info/direct_url.json")):
        source = _editable_source(direct)
        if source is None:
            continue
        origin = _locate_source(environment, direct, source)
        names = _owned_hooks(direct.parent)
        dist = direct.parent.name.partition("-")[0]
        if not names:
            # Some editable wheels ship incomplete RECORDs. Only match their
            # generated hooks, never an unowned ordinary third-party .pth.
            names = {p.name for p in site.glob("*.pth")
                     if re.fullmatch(rf"(?:__editable__[._]?{re.escape(dist)}(?:[-_].+)?|_editable_impl_{re.escape(dist)})\.pth",
                                     p.name, re.IGNORECASE)}
        if not names:
            names = {f"__lp_editable_{dist}.pth"}
        for name in sorted(names):
            owned[site / name] = origin

    # Recognize setuptools hooks even when a legacy installer omitted PEP 610.
    for hook in sorted(site.glob("__editable__*.pth")):
        owned.setdefault(hook, None)

    roots = [root.resolve() for root in extra if root.is_dir()]
    writes = {}
    for hook, source in owned.items():
        roots.extend(_rewrite_hook(hook, source, writes))
    embedded = next(iter(sorted(environment.glob("python*._pth"))), None)
    entries = [_relative(root, environment) for root in roots] if embedded else []
    if embedded is None and extra:
        writes[site / "__lp_explicit_roots.pth"] = "".join(
            _relative(root.resolve(), site) + "\n" for root in extra if root.is_dir())
    for path, content in writes.items():
        _write_changed(path, content)
    if embedded:
        _rewrite_embedded_pth(embedded, environment, entries)
    print(f"[LP] Normalized {len(owned)} editable import hook(s) in {site}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    parser.add_argument("--root", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        return normalize(args.environment, tuple(args.root))
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"[LP] Failed to normalize editable installs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
