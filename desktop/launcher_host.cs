using System;
using System.Diagnostics;
using System.ComponentModel;
using System.IO;
using System.Text;
using System.Windows.Forms;

internal static class LauncherHost
{
    private static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, "python_runtime", "pythonw.exe");
        if (!File.Exists(python))
        {
            ShowError("Bundled Python is missing. Please use a complete portable release.");
            return 1;
        }

        string[] commandLine = Environment.GetCommandLineArgs();
        try
        {
            bool lowProcess = LPWindowsIntegrity.ProcessLevel() < LPWindowsIntegrity.Medium;
            bool lowFiles = LPWindowsIntegrity.FileLevel(Path.Combine(root, "LP-Lora-Trainer.exe")) < LPWindowsIntegrity.Medium
                || LPWindowsIntegrity.FileLevel(python) < LPWindowsIntegrity.Medium
                || LPWindowsIntegrity.FileLevel(Path.Combine(root, "python_runtime", "python.exe")) < LPWindowsIntegrity.Medium;
            if (commandLine.Length == 2 && commandLine[1] == "--check-startup-permissions")
                return lowProcess || lowFiles ? 3 : 0;
            if (lowFiles)
            {
                // A normal process first tries repair without elevation. A legacy
                // low-integrity executable needs a separate Windows-approved helper.
                int repaired = RepairIntegrity(root, commandLine, lowProcess, lowProcess);
                if (repaired == 0) return 1;
                if (repaired == 2) return 0; // The desktop shell starts a fresh normal process.
            }
            else if (lowProcess)
            {
                ShowError("当前应用由受限进程启动，无法写入普通用户目录。请从资源管理器打开应用。" +
                    "\n应用不会绕过启动它的沙盒或安全策略。");
                return 1;
            }
        }
        catch (Exception error)
        {
            ShowError("无法完成启动权限检查。\n\n" + error.Message);
            return 1;
        }
        var arguments = new StringBuilder();
        arguments.Append(Quote("-m"));
        arguments.Append(" ");
        arguments.Append(Quote("desktop.launcher"));
        for (int i = 1; i < commandLine.Length; i++)
        {
            arguments.Append(" ");
            arguments.Append(Quote(commandLine[i]));
        }

        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = python,
                Arguments = arguments.ToString(),
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            });
            return 0;
        }
        catch (Exception error)
        {
            ShowError("Unable to start LP LoRA Trainer.\n\n" + error.Message);
            return 1;
        }
    }


    private static int RepairIntegrity(string root, string[] commandLine, bool elevated, bool relaunch)
    {
        string script = Path.Combine(root, "scripts", "repair_startup_integrity.ps1");
        if (!File.Exists(script)) throw new FileNotFoundException("Application repair script is missing", script);
        string powershell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),
            "WindowsPowerShell", "v1.0", "powershell.exe");
        var forwarded = new StringBuilder();
        for (int i = 1; i < commandLine.Length; i++)
        {
            if (i > 1) forwarded.Append(" ");
            forwarded.Append(Quote(commandLine[i]));
        }
        string arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script);
        if (relaunch)
            arguments += " -Relaunch -LaunchArgumentsBase64 " + Quote(Convert.ToBase64String(
                Encoding.UTF8.GetBytes(forwarded.ToString())));
        var options = new ProcessStartInfo
        {
            FileName = powershell,
            Arguments = arguments,
            WorkingDirectory = root,
            UseShellExecute = elevated,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden
        };
        if (elevated) options.Verb = "runas";
        try
        {
            using (Process repair = Process.Start(options))
            {
                repair.WaitForExit();
                if (repair.ExitCode == 0) return relaunch ? 2 : 1;
            }
            if (!elevated) return RepairIntegrity(root, commandLine, true, true);
            ShowError("应用启动权限修复失败，未启动训练服务。请检查应用目录是否允许修改。");
            return 0;
        }
        catch (Win32Exception error)
        {
            if (error.NativeErrorCode != 1223) throw;
            ShowError("已取消启动权限修复。应用尚未启动；再次打开时可重试。" +
                "\n修复仅处理本应用启动文件，不会修改用户文件夹权限。");
            return 0;
        }
    }

    private static string Quote(string value)
    {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                slashes++;
                continue;
            }
            if (character == '"')
            {
                result.Append('\\', slashes * 2 + 1);
                result.Append('"');
                slashes = 0;
                continue;
            }
            result.Append('\\', slashes);
            slashes = 0;
            result.Append(character);
        }
        result.Append('\\', slashes * 2);
        result.Append('"');
        return result.ToString();
    }

    private static void ShowError(string message)
    {
        MessageBox.Show(message, "LP LoRA Trainer", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
