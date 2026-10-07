"""Installer bootstrap tests without network or training dependencies."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from scripts import bootstrap_uv, ensure_engine_venv
from app.services.common import ServiceError
from app.services.installation import InstallationService


class UvBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.target = self.root / 'uv' / 'uv.exe'
        self.asset = 'uv-x86_64-pc-windows-msvc.zip'

    def release(self, assets=None):
        return io.BytesIO(json.dumps({'assets': assets if assets is not None else [
            {'name': self.asset, 'browser_download_url': 'https://example.invalid/uv.zip'}
        ]}).encode())

    def run_install(self, entries=None, download=None):
        def archive(url, destination):
            with zipfile.ZipFile(destination, 'w') as bundle:
                for name, content in (entries or {'uv.exe': b'fixture executable'}).items():
                    bundle.writestr(name, content)
        with patch.object(bootstrap_uv, '_asset_name', return_value=self.asset), patch.object(bootstrap_uv.urllib.request, 'urlopen', return_value=self.release()), patch.object(bootstrap_uv, '_download', side_effect=download or archive):
            bootstrap_uv.install(self.target)

    def test_installs_only_executable_without_archive_path_traversal(self):
        self.run_install({'../../uv.exe': b'fixture', '../../escaped.txt': b'bad', 'uvx.exe': b'other'})
        self.assertEqual(self.target.read_bytes(), b'fixture')
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertFalse((self.root / 'escaped.txt').exists())

    def test_existing_binary_is_not_downloaded_or_replaced(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'existing')
        with patch.object(bootstrap_uv.urllib.request, 'urlopen') as network:
            bootstrap_uv.install(self.target)
        network.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'existing')

    def test_missing_release_asset_has_useful_error(self):
        with patch.object(bootstrap_uv, '_asset_name', return_value=self.asset), patch.object(bootstrap_uv.urllib.request, 'urlopen', return_value=self.release([])):
            with self.assertRaisesRegex(RuntimeError, self.asset):
                bootstrap_uv.install(self.target)
        self.assertFalse(self.target.exists())

    def test_invalid_zip_or_missing_binary_leaves_no_partial_target(self):
        def invalid(url, destination):
            destination.write_bytes(b'not zip')
        for entries, download, error in [({'readme': b'text'}, None, RuntimeError), (None, invalid, zipfile.BadZipFile)]:
            with self.subTest(error=error), self.assertRaises(error):
                self.run_install(entries, download)
            self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_download_and_replace_failure_clean_staging(self):
        def fail(*args):
            raise OSError('network failed')
        with self.assertRaises(OSError):
            self.run_install(download=fail)
        with patch.object(bootstrap_uv.os, 'replace', side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                self.run_install()
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_architecture_is_explicit(self):
        for arch, suffix in [('AMD64', 'x86_64'), ('ARM64', 'aarch64')]:
            with patch.dict(os.environ, {'PROCESSOR_ARCHITECTURE': arch, 'PROCESSOR_ARCHITEW6432': ''}):
                self.assertEqual(bootstrap_uv._asset_name(), f'uv-{suffix}-pc-windows-msvc.zip')
        with patch.dict(os.environ, {'PROCESSOR_ARCHITECTURE': 'x86', 'PROCESSOR_ARCHITEW6432': ''}):
            with self.assertRaises(RuntimeError):
                bootstrap_uv._asset_name()


class EngineEnvironmentBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.engine = self.root / 'engine'
        self.engine.mkdir()
        self.environment = self.engine / 'venv'
        self.uv = self.root / 'uv.exe'
        self.uv.write_bytes(b'fixture')

    def create(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ensure_engine_venv.create_environment(self.engine, self.environment, '3.11', self.uv)

    def test_rejects_escape_and_engine_root_before_spawning(self):
        with patch.object(ensure_engine_venv.subprocess, 'run') as run:
            for target in [self.root / 'outside', self.engine, self.engine / '..' / 'outside']:
                with self.assertRaises(ValueError):
                    ensure_engine_venv.create_environment(self.engine, target, '3.11', self.uv)
            run.assert_not_called()

    def test_existing_environment_is_idempotent(self):
        python = ensure_engine_venv._python_path(self.environment)
        python.parent.mkdir(parents=True)
        python.write_bytes(b'existing')
        with patch.object(ensure_engine_venv.subprocess, 'run') as run:
            self.create()
        run.assert_not_called()

    def test_managed_seeded_python_and_external_environment_scrubbing(self):
        def finish(command, **kwargs):
            self.assertEqual(command, [str(self.uv), 'venv', str(self.environment), '--python', '3.11', '--managed-python', '--seed'])
            self.assertEqual(kwargs['cwd'], str(self.engine))
            self.assertEqual(kwargs['env']['UV_PYTHON_PREFERENCE'], 'only-managed')
            for name in ['PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'UV_PYTHON']:
                self.assertNotIn(name, kwargs['env'])
            python = ensure_engine_venv._python_path(self.environment)
            python.parent.mkdir(parents=True)
            python.write_bytes(b'fixture')
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, {'PYTHONHOME':'bad', 'PYTHONPATH':'bad', 'VIRTUAL_ENV':'bad', 'UV_PYTHON':'bad', 'UV_PROJECT_ENVIRONMENT':'bad'}), patch.object(ensure_engine_venv.subprocess, 'run', side_effect=finish):
            self.create()

    def test_nonzero_exit_and_missing_python_are_errors(self):
        for code, message in [(7, 'exit code 7'), (0, 'did not create')]:
            with patch.object(ensure_engine_venv.subprocess, 'run', return_value=SimpleNamespace(returncode=code)):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.create()

    def test_invalid_inputs_fail_without_spawning(self):
        with patch.object(ensure_engine_venv.subprocess, 'run') as run:
            for version in ['', '3.11 --system']:
                with self.assertRaises(ValueError):
                    ensure_engine_venv.create_environment(self.engine, self.environment, version, self.uv)
            with self.assertRaises(FileNotFoundError):
                ensure_engine_venv.create_environment(self.engine, self.environment, '3.11', self.root / 'missing')
            run.assert_not_called()


class MirrorURLTests(unittest.TestCase):
    def test_official_and_normalized_http_sources(self):
        cases = [
            (None, None), ('', None), ('   ', None),
            (' https://pypi.tuna.tsinghua.edu.cn/simple/ ', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
            ('https://mirrors.aliyun.com/pypi/simple', 'https://mirrors.aliyun.com/pypi/simple'),
            ('http://localhost:8080/simple/', 'http://localhost:8080/simple'),
            ('https://[::1]:8443/simple', 'https://[::1]:8443/simple'),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(InstallationService._validate_mirror_url(value), expected)

    def test_invalid_sources_are_client_errors(self):
        for value in (
            'ftp://mirror/simple', 'mirror/simple', 'https:///simple',
            'https://user:password@mirror/simple', 'https://@mirror/simple',
            'https://mirror/sim ple', 'https://mirror/sim\nple',
            'https://mirror/sim\x00ple', 'https://mirror/sim\x7fple',
            'https://[::1/simple', 'https://mirror:abc/simple',
            'https://mirror:65536/simple', 'https://mirror:0/simple',
            r'https://mirror\other/simple',
        ):
            with self.subTest(value=value), self.assertRaises(ServiceError) as caught:
                InstallationService._validate_mirror_url(value)
            self.assertEqual(caught.exception.status, 400)


if __name__ == '__main__':
    unittest.main()
