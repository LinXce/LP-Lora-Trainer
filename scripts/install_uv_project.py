"""Install an upstream uv solution into the selected embedded CPython."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # Musubi's Python 3.10 includes the seeded pip.
    from pip._vendor import tomli as tomllib


def mirror_for(index, torch_mirror, pypi_mirror):
    host = urlsplit(index).hostname
    if host in {"pypi.org", "www.pypi.org"}:
        return pypi_mirror
    if host == "download.pytorch.org":
        return torch_mirror
    return None


def locked_requirements(exported, upstream_lock, torch_mirror, pypi_mirror):
    """Keep uv's selected markers, direct references and hashes bound to their indexes.

    Some Anima registries publish wheels without a hash. PEP 751 export tries to
    download them to compute one, even for unrelated target platforms. The native
    requirements export preserves the available upstream hashes without downloads.
    """
    with upstream_lock.open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    indexes = {}
    for package in packages:
        source = package.get("source", {}).get("registry")
        if source:
            key = package["name"], package["version"]
            indexes.setdefault(key, set()).add(source)
    sections = {}
    for line in re.split(r"(?<!\\)\n", exported.read_text(encoding="utf-8")):
        if not line.strip():
            continue
        match = re.match(r"^([\w.-]+)==([^ ;\\\n]+)", line)
        destination = ""
        if match:
            sources = indexes.get(match.groups(), set())
            if len(sources) != 1:
                raise ValueError(f"Cannot determine the locked registry for {match[0]}")
            index = next(iter(sources))
            destination = mirror_for(index, torch_mirror, pypi_mirror) or index
        sections.setdefault(destination, []).append(line)
    steps = []
    for number, (index, lines) in enumerate(sections.items()):
        target = exported.parent / f"source-{number}.txt"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        steps.append((["--default-index", index] if index else [], target))
    return steps


def install(uv, project, python, groups, no_default_groups, torch_index, extras=(), keep_installed_torch=False):
    if not uv.is_file() or not python.is_file() or not (project / "pyproject.toml").is_file():
        raise ValueError("Missing uv executable, target Python or project metadata")
    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="lp-uv-plan-") as directory:
        frozen = (project / "uv.lock").is_file()
        lock_path = Path(directory) / ("requirements.txt" if frozen else "pylock.toml")
        if frozen:
            command = [str(uv), "export", "--frozen", "--no-emit-project", "--no-header", "--no-annotate"]
            if no_default_groups:
                command.append("--no-default-groups")
        else:
            # Older checkouts can lack uv.lock. Compile for this interpreter into
            # the temporary directory; never generate a lock in the user's clone.
            command = [str(uv), "pip", "compile", "pyproject.toml"]
            with (project / "pyproject.toml").open("rb") as stream:
                metadata = tomllib.load(stream)
            config = metadata.get("tool", {}).get("uv", {})
            if not no_default_groups:
                declared = metadata.get("dependency-groups", {})
                default = config.get("default-groups", ["dev"] if "dev" in declared else [])
                groups = list(dict.fromkeys([*(declared if default == "all" else default), *groups]))
            if torch_index:
                sources = config.get("sources", {}).get("torch", [])
                if isinstance(sources, dict):
                    sources = [sources]
                for item in sources:
                    if item.get("index") and (item.get("group") in groups or item.get("extra") in extras):
                        command += ["--index", f"{item['index']}={torch_index}"]
        for group in groups:
            command += ["--group", group]
        for extra in extras:
            command += ["--extra", extra]
        if keep_installed_torch:
            # A generic, no-extra lock can name the PyPI CPU build. Preserve a
            # working accelerator runtime when the user selected "existing".
            probe = subprocess.run([str(python), "-I", "-c",
                                    "from importlib.metadata import distributions; "
                                    "print(' '.join(d.metadata.get('Name','').lower() for d in distributions() "
                                    "if d.metadata.get('Name','').lower() in ('torch','torchvision')))"],
                                   capture_output=True, text=True, env=env, check=True, timeout=30,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for name in sorted(set(probe.stdout.split()) & {"torch", "torchvision"}):
                command += ["--prune" if frozen else "--no-emit-package", name]
        command += ["--format", lock_path.name, "--python", str(python), "--output-file", str(lock_path)]
        kwargs = dict(cwd=project, env=env, check=True,
                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print("[LP] Resolving the selected upstream dependency set", flush=True)
        subprocess.run(command, stdout=subprocess.DEVNULL, **kwargs)
        pypi = env.get("UV_DEFAULT_INDEX") or env.get("UV_INDEX_URL")
        steps = (locked_requirements(lock_path, project / "uv.lock", torch_index, pypi) if frozen
                 else [([], lock_path)])
        for indexes, path in steps:
            # This is the entire selected transitive graph. A second resolution
            # here or for the editable project can replace the selected CUDA build.
            subprocess.run([str(uv), "--no-config", "pip", "install", "--python", str(python),
                            "--no-deps", *indexes, "-r", str(path)], **kwargs)
        print("[LP] Installing the project into the selected interpreter", flush=True)
        subprocess.run([str(uv), "pip", "install", "--python", str(python), "--no-deps",
                        "-e", str(project)], **kwargs)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--no-default-groups", action="store_true")
    parser.add_argument("--keep-installed-torch", action="store_true")
    parser.add_argument("--torch-index", "--find-links", help="PyTorch simple index mirror for the selected CUDA tag")
    args = parser.parse_args(argv)
    try:
        return install(args.uv.resolve(), args.project.resolve(), args.python.resolve(),
                       args.group, args.no_default_groups, args.torch_index, args.extra, args.keep_installed_torch)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (OSError, ValueError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
