"""Build a relocatable Windows application runtime from trusted local dependencies.

This is a release-builder tool, not an installer: it downloads nothing, preserves
licenses, and never copies the venv redirector, editable hooks or console EXEs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import time
import tempfile
import tomllib

# packaging is a release-builder dependency only.  Keep it out of the delivered
# python_runtime so the portable application can import this module for checks.
def _packaging_types():
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.utils import canonicalize_name
        except ImportError as exc:
            raise RuntimeError(
                "构建便携运行时需要构建机安装 packaging（或 pip）；应用运行环境不需要它"
            ) from exc
    return Requirement, canonicalize_name


def active_requirement(requirement, extras=()):
    return requirement.marker is None or any(
        requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)
    )


def resolve_dependencies(requirements, lookup=metadata.distribution):
    """Resolve the installed dependency closure, including propagated extras."""
    Requirement, canonicalize_name = _packaging_types()
    pending = [Requirement(value) for value in requirements]
    selected = {}
    processed_extras = {}
    while pending:
        requirement = pending.pop()
        if not active_requirement(requirement):
            continue
        if requirement.url:
            raise ValueError(f"Direct-URL dependencies are not supported: {requirement}")
        name = canonicalize_name(requirement.name)
        if name == "lp-lora-trainer":
            raise ValueError("Application source must not be bundled as an editable package")
        dist = selected.get(name) or lookup(requirement.name)
        if not requirement.specifier.contains(dist.version, prereleases=True):
            raise ValueError(f"Installed {name}=={dist.version} does not satisfy {requirement}")
        extras = processed_extras.get(name, set()) | set(requirement.extras)
        if name in selected and extras == processed_extras[name]:
            continue
        selected[name] = dist
        processed_extras[name] = extras
        for value in dist.requires or ():
            dependency = Requirement(value)
            if active_requirement(dependency, extras):
                # Its marker refers to the *parent's* extras, not the child's.
                dependency.marker = None
                pending.append(dependency)
    return dict(sorted(selected.items()))


def package_file(value):
    """Reject RECORD paths escaping site-packages; do not ship venv launchers."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    if ":" in path.parts[0]:
        return None
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        return None
    if path.name == "direct_url.json":
        return None
    if path.suffix == ".pth" or path.name.startswith("__editable__"):
        raise ValueError(f"Dependency has an unsupported path hook: {value}")
    return path


def copy_dependencies(distributions, destination):
    destination.mkdir(parents=True)
    owners = {}
    for name, dist in distributions.items():
        if dist.files is None:
            raise ValueError(f"Missing installed file inventory for {name}")
        for value in dist.files:
            relative = package_file(value)
            if relative is None:
                continue
            source = Path(dist.locate_file(value))
            if not source.is_file():
                raise ValueError(f"Missing installed dependency file: {name}/{relative}")
            if source.is_symlink():
                raise ValueError(f"Dependency symlinks cannot be copied: {source}")
            key = str(relative).casefold()
            if key in owners and owners[key] != name:
                raise ValueError(f"Overlapping dependency files: {relative}")
            owners[key] = name
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def write_path_config(runtime, version):
    # DLL-named _pth overrides registry, Python environment variables and cwd.
    # No 'import site': user-site and editable/.pth hooks must never be executed.
    (runtime / f"python{version[0]}{version[1]}._pth").write_text(
        "# Paths are relative to this runtime, not the current directory.\n"
        ".\nLib\nLib/site-packages\nDLLs\n..\n", encoding="utf-8"
    )


def write_dotnet_config(runtime):
    """Allow the bundled .NET Framework bridge to load from a portable folder.

    .NET Framework can classify assemblies on a copied/relocated volume as a
    remote source and reject them with COR_E_NOTSUPPORTED. pythonnet's netfx
    loader uses AppDomain.Load, so this setting must live beside both Python
    executables that can host the application (console and windowed).
    """
    config = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <runtime>
    <loadFromRemoteSources enabled="true" />
  </runtime>
</configuration>
"""
    for executable in ("python.exe", "pythonw.exe"):
        (runtime / f"{executable}.config").write_text(config, encoding="utf-8")


def copy_cpython(base, destination):
    required = ("python.exe", "pythonw.exe", "python3.dll",
                f"python{sys.version_info.major}{sys.version_info.minor}.dll", "LICENSE.txt")
    for name in required:
        source = base / name
        if not source.is_file():
            raise ValueError(f"A full Windows CPython installation is required: {source}")
        shutil.copy2(source, destination / name)
    for source in base.glob("vcruntime*.dll"):
        shutil.copy2(source, destination / source.name)
    # The app does not use Tk, IDLE, pip, stdlib test fixtures or venv creation.
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "site-packages",
                                  "test", "tests", "idlelib", "tkinter", "turtledemo",
                                  "ensurepip", "venv")
    shutil.copytree(base / "Lib", destination / "Lib", ignore=skip)
    dlls = destination / "DLLs"
    dlls.mkdir()
    for source in (base / "DLLs").iterdir():
        if source.suffix.lower() not in {".dll", ".pyd"}:
            continue
        if source.name.startswith(("_test", "_ctypes_test", "_tkinter", "tcl", "tk")):
            continue
        shutil.copy2(source, dlls / source.name)
    write_path_config(destination, sys.version_info)
    write_dotnet_config(destination)


def isolated_environment():
    env = os.environ.copy()
    env["PYTHONHOME"] = str(Path(tempfile.gettempdir()) / "LP-invalid-python-home")
    env["PYTHONPATH"] = str(Path(tempfile.gettempdir()) / "LP-invalid-python-path")
    env["PYTHONUSERBASE"] = str(Path(tempfile.gettempdir()) / "LP-invalid-user-site")
    # Only OS tools remain visible. Python, pip and node must not come from PATH.
    env["PATH"] = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
    return env


def secure_staging_directory(staging):
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    account = subprocess.run(
        [str(system32 / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    sid = next(csv.reader([account]))[-1]
    subprocess.run(
        [
            str(system32 / "icacls.exe"), str(staging), "/inheritance:r",
            "/grant:r", f"*{sid}:(OI)(CI)F",
            "/grant:r", "*S-1-5-18:(OI)(CI)F",
            "/grant:r", "*S-1-5-32-544:(OI)(CI)F",
        ],
        check=True, capture_output=True, text=True, timeout=10,
    )


def child_exists(parent, name):
    try:
        with os.scandir(parent) as entries:
            return any(entry.name.casefold() == name.casefold() for entry in entries)
    except OSError as exc:
        raise PermissionError(f"Unable to inspect workspace directory {parent}: {exc}") from exc


def preflight_runtime_publish(workspace, staging, output):
    probe = workspace / (".python-runtime-publish-check-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    staging.rename(probe)
    probe.rename(staging)
    if not child_exists(workspace, output.name):
        return

    probe = workspace / (".python-runtime-publish-check-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        output.rename(probe)
    except OSError as exc:
        raise PermissionError(
            f"Cannot safely replace existing runtime {output} due to directory permissions: {exc}. "
            "No runtime files were changed."
        ) from exc
    try:
        probe.rename(output)
    except OSError as exc:
        raise OSError(
            f"Runtime publish preflight moved the existing runtime to {probe}, but could not "
            f"restore it to {output}: {exc}."
        ) from exc


def publish_runtime(workspace, staging, output):
    backup = workspace / ("python_runtime.previous-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    moved = False
    if output.resolve().parent != workspace or output.is_symlink():
        raise ValueError("Existing runtime is not a direct workspace directory")
    try:
        output.rename(backup)
        moved = True
    except FileNotFoundError as exc:
        if child_exists(workspace, output.name):
            raise PermissionError(
                f"Unable to move existing runtime {output} ({exc}). The workspace parent still "
                "contains the runtime directory; check its Delete permission and the parent's "
                "DeleteSubdirectoriesAndFiles permission. No ACL changes were made."
            ) from exc
    except PermissionError as exc:
        raise PermissionError(
            f"Windows denied moving existing runtime {output} ({exc}). Check Delete permission "
            "on the runtime and DeleteSubdirectoriesAndFiles permission on its parent; "
            "no ACL changes were made."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Unable to move existing runtime {output} ({exc}). "
            f"The old runtime and new runtime remain at {output} and {staging}."
        ) from exc

    publish_error = None
    retry_delays = (0.5, 1, 2, 4, 8)
    for attempt in range(len(retry_delays) + 1):
        try:
            staging.rename(output)
            publish_error = None
            break
        except PermissionError as exc:
            publish_error = exc
            if getattr(exc, "winerror", None) != 5 or attempt == len(retry_delays):
                break
            time.sleep(retry_delays[attempt])
    if publish_error is not None:
        if moved:
            try:
                backup.rename(output)
            except OSError as restore_error:
                raise OSError(
                    f"Unable to publish runtime {output} ({publish_error}); the previous runtime remains "
                    f"at {backup} because restoring it also failed ({restore_error}). "
                    f"The new runtime remains at {staging}."
                ) from publish_error
        if isinstance(publish_error, PermissionError):
            raise PermissionError(
                f"Windows denied publishing runtime to {output} ({publish_error}). Check Delete permission "
                "on the build directory and DeleteSubdirectoriesAndFiles permission on its parent; "
                f"no ACL changes were made. The new runtime remains at {staging}."
            ) from publish_error
        raise OSError(
            f"Unable to publish runtime to {output} ({publish_error}). "
            f"The new runtime remains at {staging}."
        ) from publish_error
    if moved:
        print(f"Previous runtime retained: {backup}")


def build(workspace, replace=False):
    if os.name != "nt" or platform.python_implementation() != "CPython":
        raise ValueError("This builder currently supports Windows CPython only")
    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11 or newer is required")
    if struct.calcsize("P") != 8 or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("This release layout currently supports Windows x64 only")
    project = tomllib.loads((workspace / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["desktop"]
    distributions = resolve_dependencies(requirements)
    output = workspace / "python_runtime"
    if output.exists() and not replace:
        raise ValueError("python_runtime already exists; use --replace to retain it as a backup")
    # All generated/renamed paths are direct children of the resolved workspace.
    staging = Path(tempfile.mkdtemp(prefix=".python-runtime-build-", dir=workspace)).resolve()
    if staging.parent != workspace:
        raise ValueError("Staging directory escaped workspace")
    try:
        secure_staging_directory(staging)
        preflight_runtime_publish(workspace, staging, output)
        copy_cpython(Path(sys.base_prefix), staging)
        copy_dependencies(distributions, staging / "Lib" / "site-packages")
        manifest = {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "platform": "win32",
            "architecture": "AMD64",
            "application_version": project["version"],
            "requirements": requirements,
            "packages": {name: dist.version for name, dist in distributions.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout": "relative-_pth-no-site",
        }
        (staging / "runtime-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # A sandbox-marked build directory can silently create low-integrity
        # Python executables. Normalize before testing or publishing the runtime.
        subprocess.run(
            [str(Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" /
                 "v1.0" / "powershell.exe"), "-NoLogo", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File",
             str(workspace / "scripts" / "repair_startup_integrity.ps1"),
             "-SkipLauncher", "-RuntimeDirectory", str(staging)],
            cwd=workspace, check=True, timeout=30,
        )
        subprocess.run(
            [str(staging / "python.exe"), str(workspace / "scripts" / "check_runtime.py")],
            cwd=tempfile.gettempdir(), env=isolated_environment(), check=True, timeout=90,
        )
        publish_runtime(workspace, staging, output)
    except Exception:
        # Never recursively delete user paths, even after a failed build.
        print(f"Incomplete build retained for inspection: {staging}", file=sys.stderr)
        raise
    print(f"Portable Python ready: {output}")
    print(f"CPython {manifest['python_version']}; {len(distributions)} application distributions")
    print("Build scripts\\build_launcher.ps1, then launch LP-Lora-Trainer.exe. Copy the whole project including frontend/dist to another Windows x64 PC.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Keep old runtime as a backup before replacement")
    args = parser.parse_args()
    try:
        build(Path(__file__).resolve().parents[1], replace=args.replace)
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Portable runtime build failed: {exc}\n")


if __name__ == "__main__":
    main()
