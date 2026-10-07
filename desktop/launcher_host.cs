using System;
using System.Diagnostics;
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
