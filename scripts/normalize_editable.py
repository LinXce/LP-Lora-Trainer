"""Replace pip's editable-install hooks with relative `.pth` files.

`pip install -e` writes ``__editable__*_finder.py`` (a MAPPING of absolute paths) and
``__editable__*.pth``. Both break the moment the project is copied to another drive.
This helper reads the absolute roots out of the generated metadata and rewrites them as
one relative ``.pth`` per package: a ``.pth`` line is resolved against the directory the
file lives in, so ``../../../src`` keeps working after a move.

Usage:
    python scripts/normalize_editable.py <environment-root>

The environment root is the copied CPython (``<engine>/.lp/python``) or a venv; the
site-packages directory is located from the interpreter, not guessed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _site_packages(environment: Path) -> Path | None:
    """Ask the environment's own interpreter where its purelib is."""
    python = environment / ("python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        python = environment / "Scripts" / "python.exe"
    if not python.is_file():
        return None
    probe = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    text = (probe.stdout or "").strip()
    return Path(text) if probe.returncode == 0 and text else None


def _editable_roots(site: Path) -> dict[str, list[Path]]:
    """{distribution: [import roots]} from pip's editable metadata.

    The finder's ``MAPPING`` maps *package* names to their directories, so the import
    root is the mapped directory's parent. ``direct_url.json`` is the fallback for
    installs that used a different editable backend.
    """
    roots: dict[str, list[Path]] = {}
    import ast

    for finder in site.glob("__editable__*_finder.py"):
        try:
            tree = ast.parse(finder.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        paths: list[Path] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not targets or targets[0] not in {"MAPPING", "NAMESPACES"}:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for value in node.value.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    candidate = Path(value.value)
                    # MAPPING points at the package; NAMESPACES points at a root.
                    for option in (candidate.parent, candidate):
                        if option.is_dir():
                            paths.append(option.resolve())
                            break
        if paths:
            name = finder.name[len("__editable__"):].split("_finder")[0]
            roots.setdefault(name.split("-")[0], []).extend(dict.fromkeys(paths))

    if roots:
        return roots

    from urllib.parse import unquote, urlparse
    for direct in site.glob("*.dist-info/direct_url.json"):
        try:
            data = json.loads(direct.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("dir_info", {}).get("editable"):
            continue
        url = str(data.get("url") or "")
        if not url.startswith("file:"):
            continue
        path = Path(unquote(urlparse(url).path.lstrip("/")))
        if os.name == "nt" and url.startswith("file:///"):
            path = Path(unquote(urlparse(url).path[1:]))
        if path.is_dir():
            roots.setdefault(direct.parent.name.split("-")[0], []).append(path.resolve())
    return roots


def _relative(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:  # different drives on Windows
        return ""


def _pth_file(environment: Path) -> Path | None:
    """The interpreter's ._pth, when this is an embedded (no-site) environment.

    ``.pth`` files in site-packages are only read by ``site``, and an embedded
    interpreter deliberately starts without it, so its import roots have to live in
    ``pythonXY._pth`` next to the executable instead.
    """
    for candidate in sorted(environment.glob("python*._pth")):
        return candidate
    return None


def _rewrite_embedded_pth(pth: Path, environment: Path, entries: list[str]) -> None:
    """Keep the base layout lines and replace LP's editable block.

    Several editable distributions may resolve to the same import root (a package
    installed from a workspace root and its tooling, for example). The block is
    deduplicated so repeated installs cannot grow duplicate paths in the ``._pth``.
    """
    lines = []
    for line in pth.read_text(encoding="utf-8", errors="replace").splitlines():
        # Generated lines start with the relative root, not with `#`, so the marker
        # has to be detected anywhere in the line; startswith() kept appending a new
        # copy on every reinstall.
        if "# LP editable:" in line:
            continue
        if line.strip() in entries:
            continue
        lines.append(line)
    for entry in dict.fromkeys(entries):
        lines.append(f"{entry}  # LP editable: relative import root")
    pth.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def normalize(environment: Path) -> int:
    site = _site_packages(environment)
    if site is None or not site.is_dir():
        print(f"[LP] No site-packages found under {environment}", file=sys.stderr)
        return 1

    roots = _editable_roots(site)
    if not roots:
        print("[LP] No editable installs to normalize", flush=True)
        return 0

    written = 0
    embedded_pth = _pth_file(environment)
    embedded_entries: list[str] = []
    for distribution, paths in roots.items():
        entries = []
        for path in paths:
            relative = _relative(path, site)
            if not relative:
                print(f"[LP] Skipped {distribution}: {path} is on another drive", file=sys.stderr)
                continue
            entries.append(relative)
        if not entries:
            continue
        if embedded_pth is not None:
            # No site: the ._pth carries the roots, entries are relative to the
            # interpreter directory, so convert from site-relative to env-relative.
            for relative in entries:
                embedded_entries.append(_relative((site / relative).resolve(), environment))
            # Write the replacement FIRST; deleting pip's hooks before the new roots
            # exist would leave the package unimportable if the write fails.
            _rewrite_embedded_pth(embedded_pth, environment, embedded_entries)
        else:
            marker = site / f"__lp_editable_{distribution}.pth"
            marker.write_text("\n".join(entries) + "\n", encoding="utf-8")
        written += 1

        # Drop pip's absolute hooks only once the relative replacement exists. The
        # hook names differ per backend (`__editable__<dist>-<ver>_finder.py`,
        # `_editable_impl_<dist>.pth`), so the test is "contains an absolute path",
        # not a name pattern.
        for hook in list(site.glob("__editable__*")) + list(site.glob("_editable_impl_*.pth")):
            try:
                text = hook.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            absolute = any(line.strip() and not line.lstrip().startswith(("#", "import "))
                           and (":\\" in line or line.lstrip().startswith("/"))
                           for line in text.splitlines())
            if not absolute and hook.suffix != ".pth":
                # finders keep their paths inside a Python literal, not per line
                absolute = ":\\" in text or "/home/" in text or "/Users/" in text
            if absolute:
                try:
                    hook.unlink()
                except OSError:
                    pass
        print(f"[LP] {distribution} -> {', '.join(entries)}", flush=True)

    if embedded_pth is not None and embedded_entries:
        _rewrite_embedded_pth(embedded_pth, environment, embedded_entries)
        print(f"[LP] Wrote import roots into {embedded_pth.name}", flush=True)

    print(f"[LP] Normalized {written} editable install(s) in {site}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # `--root` adds import roots that pip's metadata no longer describes (a finder that
    # was written by an older backend, for example).
    extra: list[Path] = []
    cleaned: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--root" and index + 1 < len(args):
            extra.append(Path(args[index + 1]).resolve())
            index += 2
            continue
        cleaned.append(args[index])
        index += 1
    if len(cleaned) != 1:
        print("usage: normalize_editable.py <environment-root> [--root DIR ...]", file=sys.stderr)
        return 2
    environment = Path(cleaned[0]).resolve()
    if not environment.is_dir():
        print(f"[LP] Environment does not exist: {environment}", file=sys.stderr)
        return 2
    try:
        code = normalize(environment)
        if code == 0 and extra:
            site = _site_packages(environment)
            pth = _pth_file(environment) if site is not None else None
            if site is not None and pth is not None:
                entries = [_relative(root, environment) for root in extra if root.is_dir()]
                entries = [e for e in entries if e]
                if entries:
                    _rewrite_embedded_pth(pth, environment, entries)
                    for hook in list(site.glob("__editable__*")) + list(site.glob("_editable_impl_*.pth")):
                        try:
                            hook.unlink()
                        except OSError:
                            pass
                    print(f"[LP] Added {len(entries)} explicit root(s) to {pth.name}", flush=True)
        return code
    except OSError as exc:
        print(f"[LP] Failed to normalize editable installs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
