"""Windows release labels and real user-directory exports; no GUI or UAC prompts."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from desktop.permissions import MEDIUM_INTEGRITY, process_integrity_level

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows integrity controls")
class StartupIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        cls.csc = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        if not cls.csc.is_file():
            raise unittest.SkipTest(".NET Framework compiler unavailable")
        if process_integrity_level() < MEDIUM_INTEGRITY:
            raise unittest.SkipTest("A restricted parent cannot test ordinary desktop launches")
        cls.storage = tempfile.TemporaryDirectory(prefix="lp-integrity-tests-")
        cls.workspace = Path(cls.storage.name)
        (cls.workspace / "scripts").mkdir()
        (cls.workspace / "desktop").mkdir()
        (cls.workspace / "python_runtime").mkdir()
        for relative in ("scripts/repair_startup_integrity.ps1", "desktop/windows_integrity.cs"):
            shutil.copyfile(ROOT / relative, cls.workspace / relative)
        source = cls.workspace / "probe.cs"
        source.write_text('using System; using System.Diagnostics; class Probe { static int Main(string[] args) { '
                          'if(args.Length >= 2 && args[0] == "spawn") { '
                          'var o=new ProcessStartInfo { FileName=args[1],Arguments=args.Length>2?args[2]:"",UseShellExecute=false,'
                          'CreateNoWindow=true,RedirectStandardOutput=true }; '
                          'using(var p=Process.Start(o)) { Console.Write(p.StandardOutput.ReadToEnd()); '
                          'p.WaitForExit(); return p.ExitCode; } } '
                          'Console.WriteLine(LPWindowsIntegrity.ProcessLevel()); return 0; } }', encoding="utf-8")
        cls.probe = cls.workspace / "probe.exe"
        cls.run_command([cls.csc, "/nologo", "/target:exe", "/out:" + str(cls.probe),
                         source, cls.workspace / "desktop/windows_integrity.cs"])
        cls.targets = [cls.workspace / "LP-Lora-Trainer.exe",
                       cls.workspace / "python_runtime/python.exe",
                       cls.workspace / "python_runtime/pythonw.exe"]
        cls.run_command([cls.csc, "/nologo", "/target:winexe", "/out:" + str(cls.targets[0]),
                         "/reference:System.Windows.Forms.dll", ROOT / "desktop/launcher_host.cs",
                         cls.workspace / "desktop/windows_integrity.cs"])
        for target in cls.targets[1:]:
            shutil.copyfile(cls.probe, target)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    @classmethod
    def run_command(cls, args, check=True):
        return subprocess.run([str(arg) for arg in args], check=check, capture_output=True,
                              text=True, errors="replace", timeout=30,
                              creationflags=subprocess.CREATE_NO_WINDOW)

    @classmethod
    def ps(cls, script, check=True):
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return cls.run_command([cls.powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                                "-EncodedCommand", encoded], check=check)

    @classmethod
    def label(cls, path, value):
        cls.run_command([Path(os.environ["SystemRoot"]) / "System32/icacls.exe", path,
                         "/setintegritylevel", value, "/Q"])

    def repair(self, *arguments, check=True):
        return self.run_command([self.powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                                 "-ExecutionPolicy", "Bypass", "-File",
                                 self.workspace / "scripts/repair_startup_integrity.ps1", *arguments], check=check)

    def setUp(self):
        for target in self.targets:
            self.label(target, "M")
        self.label(self.workspace / "python_runtime", "M")

    def dacl(self, target):
        path = str(target).replace("'", "''")
        return self.ps("(Get-Acl -LiteralPath '" + path + "').GetSecurityDescriptorSddlForm("
                       "[System.Security.AccessControl.AccessControlSections]::Access)").stdout.strip()

    def test_native_and_python_token_readers_report_unrestricted_processes(self):
        self.label(self.probe, "M")
        self.assertGreaterEqual(int(self.run_command([self.probe]).stdout), MEDIUM_INTEGRITY)
        self.assertGreaterEqual(process_integrity_level(), MEDIUM_INTEGRITY)

    def test_normal_native_launcher_can_start_child_without_elevation(self):
        self.label(self.probe, "M")
        result = self.run_command([self.probe, "spawn", self.probe])
        self.assertEqual(int(result.stdout), MEDIUM_INTEGRITY)

    def test_medium_native_startup_loads_bridge_and_exports_to_user_directory(self):
        self.label(self.probe, "M")
        with tempfile.TemporaryDirectory(prefix="lp-medium-export-") as temporary:
            target = Path(temporary) / "training.json"
            code = ("import json; from types import SimpleNamespace; "
                    "from desktop.permissions import process_integrity_level; "
                    "from desktop.window import desktop_bridge_error; "
                    "from desktop.bridge import DesktopBridge; "
                    "error=desktop_bridge_error(); b=DesktopBridge(); "
                    f"b._window=SimpleNamespace(create_file_dialog=lambda *a,**k:({str(target)!r},)); "
                    "saved=b.save_text_file('training.json',json.dumps({'rank':32})); "
                    "print(json.dumps({'integrity':process_integrity_level(),'error':error,'saved':saved}))")
            arguments = subprocess.list2cmdline(["-c", code])
            result = self.run_command([self.probe, "spawn", sys.executable, arguments])
            data = json.loads(result.stdout)
            self.assertEqual(data["integrity"], MEDIUM_INTEGRITY)
            self.assertIsNone(data["error"])
            self.assertEqual(Path(data["saved"]), target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"rank":32})

    def test_low_image_reproduces_restricted_process_and_release_check_refuses_it(self):
        self.label(self.targets[1], "L")
        self.assertLess(int(self.run_command([self.targets[1]]).stdout), MEDIUM_INTEGRITY)
        result = self.repair("-CheckOnly", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Low-integrity release file", result.stderr)
        self.assertLess(int(self.run_command([self.targets[1]]).stdout), MEDIUM_INTEGRITY)

    def test_repair_restores_process_and_preserves_discretionary_access(self):
        targets = [*self.targets, self.workspace / "python_runtime"]
        for target in targets:
            self.label(target, "L")
        before = {str(target): self.dacl(target) for target in targets}
        self.repair()
        self.repair("-CheckOnly")
        self.assertGreaterEqual(int(self.run_command([self.targets[1]]).stdout), MEDIUM_INTEGRITY)
        self.assertEqual(before, {str(target): self.dacl(target) for target in targets})

    def test_native_launcher_detects_damaged_python_before_starting_backend(self):
        self.assertEqual(self.run_command([self.targets[0], "--check-startup-permissions"]).returncode, 0)
        self.label(self.targets[2], "L")
        result = self.run_command([self.targets[0], "--check-startup-permissions"], check=False)
        self.assertEqual(result.returncode, 3)
        self.repair()
        self.assertEqual(self.run_command([self.targets[0], "--check-startup-permissions"]).returncode, 0)

    def test_native_launcher_detects_its_own_low_label(self):
        self.label(self.targets[0], "L")
        self.assertEqual(self.run_command([self.targets[0], "--check-startup-permissions"], check=False).returncode, 3)
        self.repair()
        self.assertEqual(self.run_command([self.targets[0], "--check-startup-permissions"]).returncode, 0)

    def test_repair_refuses_external_runtime_before_changing_targets(self):
        self.label(self.targets[1], "L")
        result = self.repair("-RuntimeDirectory", str(self.workspace.parent), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("direct application runtime", result.stderr)
        self.assertLess(int(self.run_command([self.targets[1]]).stdout), MEDIUM_INTEGRITY)

    def test_runtime_build_directory_is_supported_without_existing_launcher(self):
        staging = self.workspace / ".python-runtime-build-test"
        staging.mkdir(exist_ok=True)
        for filename in ("python.exe", "pythonw.exe"):
            target = staging / filename
            shutil.copyfile(self.probe, target)
            self.label(target, "L")
        self.repair("-SkipLauncher", "-RuntimeDirectory", str(staging))
        self.assertGreaterEqual(int(self.run_command([staging / "python.exe"]).stdout), MEDIUM_INTEGRITY)

    def test_desktop_shell_runs_real_bridge_export_outside_project(self):
        runtime = ROOT / "python_runtime/pythonw.exe"
        if not runtime.is_file():
            self.skipTest("Portable runtime absent")
        with tempfile.TemporaryDirectory(prefix="lp-normal-export-") as temporary:
            target = Path(temporary) / "chosen.json"
            code = ("import json; from types import SimpleNamespace; "
                    "from desktop.bridge import DesktopBridge; "
                    "from desktop.permissions import process_integrity_level; "
                    "b=DesktopBridge(); "
                    f"b._window=SimpleNamespace(create_file_dialog=lambda *a,**k:({str(target)!r},)); "
                    "b.save_text_file('chosen.json',json.dumps({'integrity':process_integrity_level(),'rank':16}))")
            arguments = subprocess.list2cmdline(["-c", code])
            quote = lambda text: "'" + str(text).replace("'", "''") + "'"
            self.ps("$shell=New-Object -ComObject Shell.Application; $windows=$shell.Windows(); "
                    "$location=0; $rootLocation=0; $hwnd=0; "
                    "$desktop=$windows.FindWindowSW([ref]$location,[ref]$rootLocation,8,[ref]$hwnd,1); "
                    "$document=$desktop.Document; $desktopShell=$document.Application; "
                    "try { $desktopShell.ShellExecute(" +
                    ",".join([quote(runtime), quote(arguments), quote(ROOT), "'open'", "0"]) +
                    ") } finally { foreach($c in @($desktopShell,$document,$desktop,$windows,$shell)) { "
                    "[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($c) } }")
            deadline = time.monotonic() + 10
            while not target.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(target.is_file(), "Normal desktop user could not export outside project")
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["integrity"], MEDIUM_INTEGRITY)
            self.assertEqual(data["rank"], 16)
