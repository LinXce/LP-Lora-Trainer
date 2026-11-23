"""Kohaku's wheel selection and editable install, exercised without a network or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter
from app.services.installation import begin_package_install, finish_package_install
from scripts import ensure_engine_venv


ROOT = Path(__file__).resolve().parents[1]
UV = ROOT / "runtime" / "uv" / ("uv.exe" if os.name == "nt" else "uv")
MANAGED = ROOT / "runtime" / "uv" / "python"
TORCH = "torch>=1.4,<2"
WHEEL = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"


def make_project(source):
    (source / "scripts/train").mkdir(parents=True)
    (source / "scripts/train/moss.py").touch()
    (source / "src/kohakumoss").mkdir(parents=True)
    (source / "src/kohakumoss/__init__.py").write_text("value = 'from checkout'\n", encoding="utf-8")
    tooling = source / "editable_tooling/kohaku_fixture_tool"
    tooling.mkdir(parents=True)
    (tooling / "__init__.py").write_text("value = 'from tools'\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        f'[project]\nname="kohakumoss"\nversion="1.0"\ndependencies=["{TORCH}"]\n'
        '[project.optional-dependencies]\nkernels=["lp-kernel"]\n'
        '[build-system]\nrequires=["lp-build-tool==1.0"]\nbuild-backend="fixture"\nbackend-path=["."]\n',
        encoding="utf-8")
    # The local build-only wheel must be importable inside uv's isolated PEP 517 environment.
    metadata = (f"Metadata-Version: 2.3\nName: kohakumoss\nVersion: 1.0\nRequires-Dist: {TORCH}\n"
                'Provides-Extra: kernels\nRequires-Dist: lp-kernel; extra == "kernels"\n')
    (source / "fixture.py").write_text(
        "from pathlib import Path\nfrom zipfile import ZipFile\nimport lp_build_tool\n"
        "def build_editable(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    name = 'kohakumoss-1.0-py3-none-any.whl'\n"
        "    info = 'kohakumoss-1.0.dist-info/'\n"
        "    with ZipFile(Path(wheel_directory)/name,'w') as w:\n"
        "        root = Path(__file__).parent\n"
        "        mapping = {name: str(root / layout / name) for name, layout in "
        "[('kohakumoss', 'src'), ('kohaku_fixture_tool', 'editable_tooling')]}\n"
        "        finder = 'MAPPING = '+repr(mapping)+'\\nfrom pathlib import Path\\nimport sys\\n' "
        "+'def install():\\n    sys.path.extend(str(Path(path).parent) for path in MAPPING.values())\\n'\n"
        "        w.writestr('__editable__kohakumoss_finder.py', finder)\n"
        "        w.writestr('__editable__kohakumoss.pth', "
        "'import __editable__kohakumoss_finder; __editable__kohakumoss_finder.install()\\n')\n"
        f"        w.writestr(info+'WHEEL', {WHEEL!r})\n"
        f"        w.writestr(info+'METADATA', {metadata!r})\n"
        "        w.writestr(info+'RECORD', '')\n"
        "    return name\n", encoding="utf-8")


def add_wheel(index, name, version):
    package = index / name
    package.mkdir(parents=True, exist_ok=True)
    stem = name.replace("-", "_")
    filename = f"{stem}-{version}-py3-none-any.whl"
    info = f"{stem}-{version}.dist-info/"
    with ZipFile(package / filename, "w") as archive:
        files = {
            stem + "/__init__.py": f"version = {version!r}\n",
            info + "METADATA": f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n",
            info + "WHEEL": WHEEL,
        }
        for path, data in files.items():
            archive.writestr(path, data)
        archive.writestr(info + "RECORD", "\n".join(path + ",," for path in [*files, info + "RECORD"]))
    (package / "index.html").write_text("\n".join(
        f'<a href="{path.name}">{path.name}</a>' for path in sorted(package.glob("*.whl"))
    ), encoding="utf-8")


class KohakuInstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root / "engine with spaces"
        make_project(self.source)
        self.adapter = KohakuMossTrainerAdapter()
        self.adapter.installation_extras = ("kernels",)
        self.target = self.root / "target"
        self.python = self.target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def test_commands_scope_reinstall_to_the_stale_wheel_and_protect_pip_target(self):
        site = self.target / "Lib/site-packages"
        (site / "torch-1.5+cu128.dist-info").mkdir(parents=True)
        (site / "torchvision-0.5+cu126.dist-info").mkdir()
        editable = str(self.source) + "[kernels]"
        mirror = "https://wheels.example/cu128"
        plan = self.adapter.installation_plan(self.source, self.python, "cu128", uv="uv", torch_mirror_url=mirror)
        wheels, project = plan.commands
        self.assertEqual(wheels, ("uv", "pip", "install", "--python", str(self.python), "--reinstall-package", "torchvision",
                                  "--index-url", mirror, TORCH, "torchvision"))
        self.assertIn(editable, project)
        self.assertIn(TORCH, project)
        self.assertNotIn("--reinstall-package", project)
        pip_wheels, pip_project = self.adapter.installation_plan(self.source, self.python, "cu128", torch_mirror_url=mirror).commands
        self.assertEqual(pip_wheels[:5], (str(self.python), "-u", "-m", "pip", "install"))
        self.assertIn("--require-virtualenv", pip_wheels)
        self.assertIn("--force-reinstall", pip_wheels)
        self.assertNotIn("--force-reinstall", pip_project)
        self.assertIn(editable, pip_project)
        self.assertIn(TORCH, pip_project)
        existing, = self.adapter.installation_plan(self.source, self.python, "existing").commands
        self.assertNotIn("--index-url", existing)
        self.assertNotIn("--extra-index-url", existing)
        self.assertNotIn("--force-reinstall", existing)
        self.assertIn(TORCH, existing)
        self.adapter.embedded_mode = True
        embedded, = self.adapter.installation_plan(self.source, self.python, "existing").commands
        self.assertNotIn("--require-virtualenv", embedded)

    @unittest.skipUnless(UV.is_file(), "requires bundled uv")
    def test_real_uv_installs_switches_cuda_and_reuses_matching_distributions(self):
        self.run_uv_roundtrip(embedded=False)

    @unittest.skipUnless(os.name == "nt" and UV.is_file() and any(MANAGED.glob("cpython-3.12*/python.exe")),
                         "requires bundled uv and a local Windows managed CPython 3.12")
    def test_real_uv_installs_and_imports_editable_in_embedded_windows_python(self):
        self.run_uv_roundtrip(embedded=True)

    def run_uv_roundtrip(self, embedded):
        indexes = self.root / "indexes"
        for tag in ("cu128", "cu130"):
            add_wheel(indexes / tag, "torch", "1.5+" + tag)
            add_wheel(indexes / tag, "torch", "3.0+" + tag)  # outside the checkout's constraint
            add_wheel(indexes / tag, "torchvision", "0.5+" + tag)
        add_wheel(indexes / "common", "lp-kernel", "1.0")
        add_wheel(indexes / "common", "lp-build-tool", "1.0")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "UV_"))}
        env.update(UV_OFFLINE="true", UV_PYTHON_DOWNLOADS="never", UV_CACHE_DIR=str(self.root / "cache"),
                   UV_LINK_MODE="copy", UV_INDEX_URL=(indexes / "common").as_uri(), PYTHONDONTWRITEBYTECODE="1")
        def run(command, cwd=None):
            completed = subprocess.run(command, cwd=cwd or self.source, env=env, capture_output=True,
                                       encoding="utf-8", errors="replace", timeout=45)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("Error processing line", completed.stderr)
            self.assertNotIn("Remainder of file ignored", completed.stderr)
            return completed.stdout.strip()

        if embedded:
            self.target = self.source / ".lp" / "python"
            self.python = self.target / "python.exe"
            self.adapter.embedded_mode = True
            with patch.dict(os.environ, {**env, "UV_PYTHON_INSTALL_DIR": str(MANAGED)}, clear=True):
                ensure_engine_venv.create_embedded(self.source, self.target, "3.12", UV)
        else:
            run([UV, "venv", self.target, "--python", sys.executable])

        def install(tag):
            commands = self.adapter.installation_plan(
                self.source, self.python, tag, uv=UV, torch_mirror_url=(indexes / tag).as_uri()).commands
            if embedded:
                begin_package_install(self.target / "python312._pth")
            try:
                for command in commands:
                    run(command)
            finally:
                if embedded:
                    finish_package_install(self.target / "python312._pth")
            if embedded:
                run([sys.executable, "-I", ROOT / "scripts/normalize_editable.py", self.target], cwd=self.root)
            installed = json.loads(run([self.python, "-I", "-c",
                "import json,torch,torchvision,kohakumoss,kohaku_fixture_tool; from importlib.metadata import distribution,version; "
                "assert kohaku_fixture_tool.value == 'from tools'; "
                "print(json.dumps([torch.version, torchvision.version,version('lp-kernel'), "
                "json.loads(distribution('kohakumoss').read_text('direct_url.json'))['dir_info']['editable'], "
                "torch.__file__, kohakumoss.__file__]))"], cwd=self.root))
            self.assertEqual(installed[:4], ["1.5+" + tag, "0.5+" + tag, "1.0", True])
            self.assertEqual(Path(installed[5]).resolve(), self.source / "src/kohakumoss/__init__.py")
            return Path(installed[4])

        install("cu128")
        module = install("cu130")
        os.utime(module, (1000000000, 1000000000))
        timestamp = module.stat().st_mtime_ns
        install("cu130")
        self.assertEqual(module.stat().st_mtime_ns, timestamp, "matching wheel was unnecessarily reinstalled")

        if embedded:
            # Move the checkout and interpreter together. All editable import roots must move with them.
            moved = self.root / "relocated engine"
            self.source.rename(moved)
            relocated_python = moved / ".lp/python/python.exe"
            polluted_path = self.root / "foreign-pythonpath"
            polluted_path.mkdir()
            (polluted_path / "lp_foreign_path.py").touch()
            userbase = self.root / "foreign-userbase"
            usersite = userbase / "Python312/site-packages"
            usersite.mkdir(parents=True)
            (usersite / "lp_foreign_usersite.py").touch()
            env.update(PYTHONPATH=str(polluted_path), PYTHONUSERBASE=str(userbase))
            # Omit -I here to exercise ordinary startup under the interpreter's own isolation policy.
            startup = json.loads(run([relocated_python, "-c",
                "import sys,site,json,kohakumoss,kohaku_fixture_tool; from importlib.util import find_spec; "
                "print(json.dumps([sys.flags.isolated,sys.flags.ignore_environment,sys.flags.no_site,"
                "sys.flags.no_user_site,site.ENABLE_USER_SITE,find_spec('lp_foreign_path') is not None,"
                "find_spec('lp_foreign_usersite') is not None,kohakumoss.__file__,kohaku_fixture_tool.__file__]))"
            ], cwd=self.root))
            isolated, ignore_env, no_site, no_user_site, user_site_enabled, from_env, from_user, *paths = startup
            self.assertEqual((isolated, ignore_env, from_env, from_user), (1, 1, False, False))
            if not no_site:
                self.assertEqual((no_user_site, user_site_enabled), (1, False))
            self.assertEqual([Path(path).resolve() for path in paths], [
                moved / "src/kohakumoss/__init__.py", moved / "editable_tooling/kohaku_fixture_tool/__init__.py"])
