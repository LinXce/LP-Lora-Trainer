"""Real localhost HTTP -> real official-entry subprocess -> logs -> diagnosis.

The miniature trainer only provisions a test interpreter. It installs no packages
and performs no network/GPU work; dependency diagnostics are explicitly a fixture.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.server import create_app
from app.config import AppPaths
from tests.local_client import LocalClient
from tests.test_functionality import DIAG
from tests.test_installation_lifecycle import toolkit


INSTALLER = r"""
import json, os, shutil, sys
from pathlib import Path
assert sys.argv[1:] == ['install']
assert 'PYTHONHOME' not in os.environ
assert os.environ['UV_PYTHON_PREFERENCE'] == 'only-managed'
print('LP_MIRROR_ENV=' + json.dumps({name: os.environ.get(name) for name in ('PIP_INDEX_URL', 'UV_INDEX_URL', 'UV_DEFAULT_INDEX')}), flush=True)
root = Path(__file__).resolve().parents[1]
print('官方安装入口 fixture 开始', flush=True)
runtime = Path(RUNTIME)
target = root / '.venv' / 'Scripts'
target.mkdir(parents=True, exist_ok=True)
shutil.copy2(runtime / 'python.exe', target / 'python.exe')
for dll in runtime.glob('*.dll'):
    shutil.copy2(dll, target / dll.name)
version = f'{sys.version_info.major}{sys.version_info.minor}'
paths = [str(target), str(runtime / 'Lib'), str(runtime / 'DLLs'), str(runtime / f'python{version}.zip')]
(target / f'python{version}._pth').write_text('\n'.join(paths) + '\n', encoding='utf-8')
print('独立解释器 fixture 完成', flush=True)
"""


@unittest.skipUnless(os.name == "nt", "Mini portable interpreter fixture uses Windows layout")
class InstallationAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="LP HTTP install 中文 ")
        self.root = Path(self.temp.name).resolve()
        self.source = toolkit(self.root / "engine" / "ai-toolkit", "RUNTIME = " + repr(str(Path(sys.executable).parent)) + "\n" + INSTALLER)
        uv = self.root / "runtime" / "uv"
        uv.mkdir(parents=True)
        (uv / "uv.exe").touch()  # manager fixture does not use uv or the network
        self.app = create_app(AppPaths.for_workspace(self.root), token="fixture", start_worker=False)
        self.runtime_patch = patch.object(self.app.state.installations, "_runtime_python", return_value=Path(sys.executable))
        self.diag_patch = patch("app.services.engines.DIAGNOSTIC_CODE", DIAG)
        self.runtime_patch.start()
        self.diag_patch.start()
        self.client = LocalClient(self.app)
        self.assertEqual(self.client.post("/api/v1/session", headers={"Origin": self.client.base_url}).status_code, 204)
        records = self.client.post("/api/v1/engines/rescan").json()
        self.key = records[0]["installation_id"]

    def tearDown(self):
        self.client.close()
        self.diag_patch.stop()
        self.runtime_patch.stop()
        self.temp.cleanup()

    def install(self, **options):
        response = self.client.post(f"/api/v1/engines/{self.key}/install", json={"confirmed": True, **options})
        self.assertEqual(response.status_code, 202, response.content)
        return response.json()["session_id"]

    def finished(self, sid):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            session = next(s for s in self.client.get("/api/v1/terminal/sessions").json() if s["session_id"] == sid)
            if session["state"] not in {"queued", "running", "verifying", "stopping"}:
                return session
            time.sleep(0.03)
        self.fail("Installer did not reach a final state")

    def test_real_http_install_log_diagnose_and_refresh(self):
        self.assertEqual(self.client.post(f"/api/v1/engines/{self.key}/install", json={"confirmed": False}).status_code, 422)
        sid = self.install()
        final = self.finished(sid)
        self.assertEqual(final["state"], "succeeded", final)
        log = self.client.get(f"/api/v1/terminal/sessions/{sid}/log?offset=0").json()
        self.assertIn("官方安装入口 fixture 开始", log["text"])
        self.assertIn("独立解释器 fixture 完成", log["text"])
        self.assertIn("diagnosis passed", log["text"])
        self.assertGreater(log["offset"], 0)
        empty = self.client.get(f"/api/v1/terminal/sessions/{sid}/log?offset={log['offset']}").json()
        self.assertEqual(empty["text"], "")
        engines = self.client.post("/api/v1/engines/rescan").json()
        engine = next(r for r in engines if r["installation_id"] == self.key)
        self.assertEqual(engine["engine_id"], "ai_toolkit")
        self.assertEqual(engine["state"], "ready")
        self.assertTrue(Path(engine["python_executable"]).is_relative_to(self.source))
        self.assertTrue(engine["environment_manifest_id"])
        self.assertEqual(self.client.post(f"/api/v1/terminal/sessions/{sid}/stop").status_code, 409)
        self.assertEqual(self.client.get(f"/api/v1/terminal/sessions/{sid}/log?offset=-1").status_code, 422)

    def test_real_nonzero_exit_is_visible_and_install_can_be_retried(self):
        manager = self.source / "manager" / "__main__.py"
        original = manager.read_text(encoding="utf-8")
        manager.write_text("print('official failure', flush=True)\nraise SystemExit(17)\n", encoding="utf-8")
        sid = self.install()
        final = self.finished(sid)
        self.assertEqual(final["state"], "failed")
        self.assertIn("exit code 17", final["error"])
        self.assertIn("official failure", self.client.get(f"/api/v1/terminal/sessions/{sid}/log").json()["text"])
        manager.write_text(original, encoding="utf-8")
        self.assertEqual(self.finished(self.install())["state"], "succeeded")

    def test_real_process_tree_cancel(self):
        manager = self.source / "manager" / "__main__.py"
        manager.write_text("import time\nprint('waiting for stop', flush=True)\ntime.sleep(30)\n", encoding="utf-8")
        sid = self.install()
        deadline = time.monotonic() + 5
        while "waiting for stop" not in self.client.get(f"/api/v1/terminal/sessions/{sid}/log").json()["text"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.03)
        self.assertEqual(self.client.post(f"/api/v1/terminal/sessions/{sid}/stop").status_code, 204)
        self.assertEqual(self.finished(sid)["state"], "cancelled")
        self.assertEqual(self.app.state.installations.workers, {})
        self.assertEqual(self.app.state.engines.installing, set())

    def test_external_interpreter_and_unknown_engine_rejected(self):
        response = self.client.put(f"/api/v1/engines/{self.key}/python", json={"python_executable": sys.executable})
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.app.state.engines.get(self.key)["python_executable"])
        self.app.state.store.patch("engine", self.key, {"python_executable": sys.executable})
        response = self.client.post(f"/api/v1/engines/{self.key}/diagnose")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.app.state.engines.get(self.key)["state"], "failed")
        unknown = self.root / "engine" / "unknown"
        unknown.mkdir()
        records = self.client.post("/api/v1/engines/rescan").json()
        key = next(r for r in records if r["label"] == "unknown")["installation_id"]
        response = self.client.post(f"/api/v1/engines/{key}/install", json={"confirmed": True})
        self.assertEqual(response.status_code, 400)


    def test_mirror_request_reaches_real_official_installer_subprocess(self):
        mirror = 'https://pypi.tuna.tsinghua.edu.cn/simple'
        sid = self.install(mirror_url=mirror + '/')
        final = self.finished(sid)
        self.assertEqual(final['state'], 'succeeded', final)
        self.assertEqual(final['mirror_url'], mirror)
        text = self.client.get(f'/api/v1/terminal/sessions/{sid}/log').json()['text']
        marker = next(line for line in text.splitlines() if line.startswith('LP_MIRROR_ENV='))
        self.assertEqual(json.loads(marker.split('=', 1)[1]), {
            'PIP_INDEX_URL': mirror, 'UV_INDEX_URL': mirror, 'UV_DEFAULT_INDEX': mirror,
        })
        self.assertIn('[LP] Package index: ' + mirror, text)
        self.assertNotIn('\ufffd', text)

    def test_invalid_mirrors_return_400_without_installation_side_effects(self):
        for mirror in ('ftp://mirror/simple', 'https://[broken', 'https://mirror:bad/simple',
                       'https://user:password@mirror/simple'):
            with self.subTest(mirror=mirror):
                response = self.client.post(f'/api/v1/engines/{self.key}/install',
                                            json={'confirmed': True, 'mirror_url': mirror})
                self.assertEqual(response.status_code, 400, response.content)
        self.assertEqual(self.client.get('/api/v1/terminal/sessions').json(), [])
        self.assertEqual(self.app.state.installations.workers, {})
        self.assertEqual(self.app.state.engines.installing, set())


if __name__ == "__main__":
    unittest.main()

