"""Real Chromium UI, HTTP validation, queue submission and persisted native parameters."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import tomllib
import unittest

from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from adapters.registry import ADAPTERS
from tests import test_engine_parameter_roundtrip as fixture


def browser_executable():
    candidates = [os.environ.get("LP_TEST_BROWSER"), shutil.which("chromium"), shutil.which("google-chrome")]
    for base, relative in (("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
                           ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe")):
        if base in os.environ:
            candidates.append(Path(os.environ[base]) / relative)
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


class BrowserParameterRoundTrip(unittest.TestCase):
    def test_initialized_form_matches_checked_submitted_and_native_values(self):
        browser = browser_executable()
        if not browser or not (fixture.ROOT / "frontend/dist/index.html").is_file():
            self.skipTest("requires a built frontend and Chrome, Chromium or Edge")
        for key in fixture.CASES:
            with self.subTest(engine=key):
                self.check_engine(browser, key)

    def check_engine(self, browser, key):
        # Each case owns a fresh session, database and draft, with the worker disabled.
        api = fixture.EngineParameterRoundTrip()
        try:
            api.setUp()
            self.check_form(api, browser, key)
        finally:
            api.doCleanups()

    def check_form(self, api, browser, key):
        _, architecture, knob, expected = fixture.CASES[key]
        labels = {row["id"]: row["label"] for row in api.client.get("/api/v1/base-models").json()}
        test = dict(engine=key, archLabel=labels[architecture], knob=knob, expected=expected, inputs=api.inputs(key),
                    model=str(api.model), dataset=api.dataset, output=str(api.root / "out"))
        page = (api.paths.frontend_dist / "index.html").read_text(encoding="utf-8")
        driver = Path(__file__).with_name("browser_parameter_roundtrip.js").read_text(encoding="utf-8")
        callback = """const report = body => fetch('/__form_result', {method:'POST', body:JSON.stringify(body)});
            exerciseTrainingForm(TEST).then(v => report({value:v}), e => report({error:String(e)}));"""
        page = page.replace("</body>", "<script>\n" + driver + "\n" + callback.replace(
            "TEST", json.dumps(test, ensure_ascii=True)) + "\n</script></body>")
        ready, report = threading.Event(), {}

        async def serve_page(request):
            return HTMLResponse(page)

        async def receive_report(request):
            report.update(await request.json())
            ready.set()
            return Response(status_code=204)

        api.app.router.routes[:0] = [Route("/__form", serve_page), Route("/__form_result", receive_report, methods=["POST"])]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            proc = subprocess.Popen([browser, "--headless=new", "--no-first-run", "--disable-extensions",
                                     "--disable-background-networking", "--disable-gpu", "--disable-dev-shm-usage",
                                     "--user-data-dir=" + profile, api.client.base_url + "/__form#/train"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.assertTrue(ready.wait(55), f"{key}: browser returned no result, exit={proc.poll()}")
            finally:
                proc.terminate()
                proc.wait(timeout=10)
        self.assertNotIn("error", report, report.get("error"))
        result = report["value"]
        self.assertEqual((result["shown"], result["after"]), (expected, expected))
        self.assertEqual(result["initialized"], api.selection(key))
        checked, submitted = result["checked"], result["submitted"]
        self.assertEqual(checked["body"], submitted["body"])
        self.assertEqual(submitted["body"]["params"], api.selection(key) | api.inputs(key))
        task = submitted["result"]
        self.assertEqual(task["state"], "queued")
        self.assertEqual(len(api.app.state.store.list("task")), 1)
        job = api.paths.data_root / "jobs" / task["task_id"]
        stored = json.loads((job / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["params"], submitted["body"]["params"])
        native = (job / ADAPTERS[key].native_filename).read_text(encoding="utf-8")

        def normalize(value):
            return value.replace(str(job).replace("\\", "\\\\"), "<task-id>").replace(str(job), "<task-id>").replace(
                job.as_posix(), "<task-id>").replace(task["task_id"], "<task-id>")

        self.assertEqual(normalize(native), checked["result"]["native_config"])
        commands = json.loads((job / "command.json").read_text(encoding="utf-8"))["commands"]
        self.assertEqual([[normalize(v) for v in command] for command in commands], checked["result"]["pipeline"])
        if key == "kohaku_moss_trainer":
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(native).body if isinstance(n, ast.Assign)}
        else:
            values = tomllib.loads(native)
        self.assertEqual(values[knob], expected)
