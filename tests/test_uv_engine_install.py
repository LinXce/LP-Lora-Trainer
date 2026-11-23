"""Offline acceptance for installing locked projects without modifying their checkout."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from adapters.sd_scripts.adapter import KohyaAdapter
from scripts.install_uv_project import locked_requirements

ROOT = Path(__file__).resolve().parents[1]
UV = ROOT / "runtime" / "uv" / ("uv.exe" if os.name == "nt" else "uv")

# An offline build backend with no dependencies: proves isolation works on an
# empty interpreter without testing the host's pip/setuptools installation.
BACKEND = '''
from pathlib import Path
from zipfile import ZipFile

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    name = "lp_dummy-1.0-py3-none-any.whl"
    info = "lp_dummy-1.0.dist-info/"
    with ZipFile(Path(wheel_directory) / name, "w") as archive:
        archive.writestr(info + "WHEEL", "Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
        archive.writestr(info + "METADATA", "Metadata-Version: 2.3\\nName: lp-dummy\\nVersion: 1.0\\n")
        archive.writestr(info + "RECORD", "")
    return name

build_wheel = build_editable
'''


def simple_wheel(directory, name, version):
    distribution = name.replace("-", "_")
    wheel = directory / f"{distribution}-{version}-py3-none-any.whl"
    info = f"{distribution}-{version}.dist-info/"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(info + "METADATA", f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n")
        archive.writestr(info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(info + "RECORD", "")
    return wheel


class ProjectInstallTests(unittest.TestCase):
    def test_hashes_and_markers_stay_bound_to_the_original_source_or_its_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "uv.lock"
            lock.write_text(
                '[[package]]\nname="torch"\nversion="1.0+cu128"\nsource={registry="https://download.pytorch.org/whl/cu128"}\n'
                '[[package]]\nname="other"\nversion="1.0"\nsource={registry="https://private.example"}\n', encoding="utf-8")
            export = root / "requirements.txt"
            requirement = "torch==1.0+cu128 ; sys_platform == 'win32' " + chr(92) + "\n    --hash=sha256:" + "a" * 64
            export.write_text(requirement + "\nother==1.0\n", encoding="utf-8")
            steps = locked_requirements(export, lock, "https://mirror.example/cu128", "https://pypi-mirror.example")
            self.assertEqual(steps[0][0], ["--default-index", "https://mirror.example/cu128"])
            self.assertEqual(steps[0][1].read_text(encoding="utf-8").strip(), requirement)
            self.assertEqual(steps[1][0], ["--default-index", "https://private.example"])

    @unittest.skipUnless(UV.is_file(), "requires bundled uv")
    def test_real_uv_locked_and_unlocked_checkouts_install_in_clean_interpreters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            project = root / "project"
            project.mkdir()
            wheels = root / "wheels"
            wheels.mkdir()
            artifact = simple_wheel(wheels, "lp-dependency", "1.0")
            cuda = simple_wheel(wheels, "torch", "2.6+cu124")
            base = simple_wheel(wheels, "torch", "2.14")
            (project / "backend.py").write_text(BACKEND, encoding="utf-8")
            (project / "pyproject.toml").write_text(
                '[project]\nname="lp-dummy"\nversion="1.0"\nrequires-python=">=3.10"\n'
                f'dependencies=["torch @ {base.as_uri()}"]\n'
                f'[project.optional-dependencies]\nchosen=["lp-dependency @ {artifact.as_uri()}"]\n'
                '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n', encoding="utf-8")
            env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", UV_CACHE_DIR=str(root / "cache"))
            for var in ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX", "UV_PROJECT_ENVIRONMENT"):
                env.pop(var, None)
            def run(command):
                p = subprocess.run(command, cwd=project, env=env, capture_output=True, encoding="utf-8",
                                   errors="replace", timeout=40)
                self.assertEqual(p.returncode, 0, p.stdout[-500:] + p.stderr[-2000:])
                return p.stdout
            for locked in (False, True):
                with self.subTest(locked=locked):
                    target = root / f"target-{locked}"
                    run([UV, "venv", "--python", sys.executable, target])
                    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    run([UV, "pip", "install", "--python", python, "--no-deps", cuda])
                    lock = project / "uv.lock"
                    if locked:
                        run([UV, "lock", "--python", python])
                    before = lock.read_bytes() if locked else None
                    run([sys.executable, "-B", ROOT / "scripts/install_uv_project.py", "--uv", UV,
                         "--python", python, "--project", project, "--extra", "chosen", "--keep-installed-torch"])
                    self.assertEqual(lock.read_bytes() if lock.exists() else None, before)
                    self.assertEqual(run([python, "-c", "from importlib.metadata import version; "
                                          "print(version('lp-dependency'), version('lp-dummy'), version('torch'))"]).strip(),
                                     "1.0 1.0 2.6+cu124")

    def test_sd_scripts_declared_dependencies_include_markers_and_root_editable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text(
                'numpy\ntensorboard; platform_machine != "ARM64"\ntensorboardX; platform_machine == "ARM64"\n'
                'adapter[fast]>=1.0,<3\n-e .\n', encoding="utf-8")
            a = KohyaAdapter()
            self.assertEqual(a.requirement_entries(root), ("numpy", 'tensorboard; platform_machine != "ARM64"',
                             'tensorboardX; platform_machine == "ARM64"', "adapter[fast]>=1.0,<3"))
            self.assertEqual(a.local_requirement_paths(root), (".",))

    @unittest.skipUnless(UV.is_file(), "requires bundled uv")
    def test_locked_cuda_package_installs_from_selected_local_mirror(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            package = root / "index/lp-accelerator"
            package.mkdir(parents=True)
            wheel = simple_wheel(package, "lp-accelerator", "1.0+cu128")
            (package / "index.html").write_text(f'<a href="{wheel.name}">{wheel.name}</a>', encoding="utf-8")
            lock = root / "uv.lock"
            lock.write_text('[[package]]\nname="lp-accelerator"\nversion="1.0+cu128"\n'
                            'source={registry="https://download.pytorch.org/whl/cu128"}\n', encoding="utf-8")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            export = root / "requirements.txt"
            export.write_text("lp-accelerator==1.0+cu128 --hash=sha256:" + digest, encoding="utf-8")
            [(flags, pinned)] = locked_requirements(export, lock, package.parent.as_uri(), None)
            target = root / "target"
            env = dict(os.environ, UV_OFFLINE="1", UV_CACHE_DIR=str(root / "cache"),
                       UV_DEFAULT_INDEX="https://unused.invalid/simple", UV_INDEX_URL="https://unused.invalid/simple")
            subprocess.run([UV, "--no-config", "venv", "--python", sys.executable, target], check=True, env=env)
            python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run([UV, "--no-config", "pip", "install", "--python", python, "--no-deps", *flags, "-r", pinned],
                           env=env, check=True)
            version = subprocess.check_output([python, "-c", "from importlib.metadata import version; print(version('lp-accelerator'))"],
                                              text=True)
            self.assertEqual(version.strip(), "1.0+cu128")

    @unittest.skipUnless(UV.is_file(), "requires bundled uv")
    def test_installed_upstream_locks_export_cuda_versions_offline(self):
        for name, flags, tag in (
                ("anima_lora", ["--group", "cuda-windows"], "cu132"),
                ("musubi-tuner", ["--extra", "cu128"], "cu128")):
            project = ROOT / "engine" / name
            if not (project / "uv.lock").is_file():
                self.skipTest(f"{name} upstream lock is absent")
            with self.subTest(project=name), tempfile.TemporaryDirectory() as temp:
                dest = Path(temp) / "requirements.txt"
                subprocess.run([UV, "export", "--offline", "--frozen", "--no-cache", "--quiet",
                                "--no-header", "--no-annotate", "--no-default-groups", "--no-emit-project",
                                *flags, "-o", dest], cwd=project, check=True, stdout=subprocess.DEVNULL)
                packages = locked_requirements(dest, project / "uv.lock", "https://fixture.invalid/torch", None)
                wheel_source = next(req for indexes, req in packages if "https://fixture.invalid/torch" in indexes)
                pins = [line.split()[0] for line in wheel_source.read_text(encoding="utf-8").splitlines()
                        if line.startswith(("torch==", "torchvision=="))]
                self.assertEqual(len(pins), 2)
                self.assertTrue(all(pin.endswith("+" + tag) for pin in pins), pins)
