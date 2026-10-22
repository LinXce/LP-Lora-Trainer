"""The release packager: engine selection and the generated package README.

The packer is a Windows PowerShell script, so this test builds a tiny fake project
and lets the real script copy it; it skips when PowerShell is unavailable.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell required")
class DeliveryPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lp-pack-")
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("pack_delivery.ps1", "delivery_readme_core.md", "delivery_readme_full.md"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        for name in ("app", "desktop", "supervisor", "adapters", "tests", "assets",
                     "frontend/dist", "python_runtime", "runtime/uv"):
            (self.root / name).mkdir(parents=True)
        (self.root / "app" / "__init__.py").write_text('APP_VERSION = "9.9.9"\n', encoding="utf-8")
        (self.root / "app" / "main.py").write_text("module\n", encoding="utf-8")
        (self.root / "python_runtime" / "python.exe").write_bytes(b"")
        (self.root / "frontend" / "dist" / "index.html").write_text("<html></html>\n", encoding="utf-8")
        (self.root / "runtime" / "uv" / "uv.exe").write_bytes(b"")
        (self.root / "LP-Lora-Trainer.exe").write_bytes(b"")
        for name in ("alpha", "beta"):
            (self.root / "engine" / name).mkdir(parents=True)
        (self.root / "engine" / "alpha" / ".lp" / "python").mkdir(parents=True)
        (self.root / "engine" / "alpha" / ".lp" / "python" / "python.exe").write_bytes(b"")
        (self.root / "engine" / "alpha" / ".git").mkdir()
        (self.root / "engine" / "alpha" / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
        (self.root / "engine" / "beta" / "run.py").write_text("run\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def pack(self, target, engines):
        return subprocess.run(
            [POWERSHELL, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(self.root / "scripts" / "pack_delivery.ps1"),
             "-Target", str(target), "-Engines", engines, "-NoPause"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        )

    def test_selected_engine_and_generated_readme(self):
        target = Path(self.temp.name) / "delivery"
        result = self.pack(target, "alpha")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        core = target / "LP-Lora-Trainer-core"
        full = target / "LP-Lora-Trainer-full"
        self.assertTrue((core / "LP-Lora-Trainer.exe").is_file())
        self.assertFalse((core / "engine").exists())
        self.assertTrue((full / "engine" / "alpha").is_dir())
        self.assertFalse((full / "engine" / "beta").exists())
        self.assertFalse((full / "engine" / "alpha" / ".git").exists())
        self.assertTrue((full / "engine" / "alpha" / ".lp" / "python" / "python.exe").is_file())
        readme = (full / "README-FIRST.md").read_text(encoding="utf-8")
        self.assertIn("| alpha | embedded |", readme)
        self.assertIn("9.9.9", readme)
        self.assertNotIn("{{", readme)
        core_readme = (core / "README-FIRST.md").read_text(encoding="utf-8")
        self.assertNotIn("{{", core_readme)

    def test_none_builds_core_only(self):
        target = Path(self.temp.name) / "core-only"
        result = self.pack(target, "none")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((target / "LP-Lora-Trainer-core" / "LP-Lora-Trainer.exe").is_file())
        self.assertFalse((target / "LP-Lora-Trainer-full").exists())


if __name__ == "__main__":
    unittest.main()
