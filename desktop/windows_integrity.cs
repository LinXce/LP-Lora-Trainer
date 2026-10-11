using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Principal;

// Shared by the native launcher and the release/repair script. No DACL changes.
public static class LPWindowsIntegrity
{
    public const int Medium = 0x2000;
    private const uint LabelInformation = 0x10;

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode)]
    private static extern uint GetNamedSecurityInfo(string name, int type, uint information,
        out IntPtr owner, out IntPtr group, out IntPtr dacl, out IntPtr sacl, out IntPtr descriptor);
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetAce(IntPtr acl, uint index, out IntPtr ace);
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetTokenInformation(IntPtr token, int information,
        IntPtr buffer, int size, out int needed);
    [DllImport("advapi32.dll")]
    private static extern IntPtr GetSidSubAuthorityCount(IntPtr sid);
    [DllImport("advapi32.dll")]
    private static extern IntPtr GetSidSubAuthority(IntPtr sid, uint index);
    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr memory);

    private static int SidLevel(IntPtr sid)
    {
        byte count = Marshal.ReadByte(GetSidSubAuthorityCount(sid));
        if (count == 0) throw new InvalidOperationException("Invalid integrity SID");
        return Marshal.ReadInt32(GetSidSubAuthority(sid, (uint)(count - 1)));
    }

    public static int ProcessLevel()
    {
        using (WindowsIdentity identity = WindowsIdentity.GetCurrent())
        {
            int needed;
            GetTokenInformation(identity.Token, 25, IntPtr.Zero, 0, out needed);
            if (needed == 0) throw new Win32Exception(Marshal.GetLastWin32Error());
            IntPtr memory = Marshal.AllocHGlobal(needed);
            try
            {
                if (!GetTokenInformation(identity.Token, 25, memory, needed, out needed))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                return SidLevel(Marshal.ReadIntPtr(memory));
            }
            finally { Marshal.FreeHGlobal(memory); }
        }
    }

    public static int FileLevel(string path)
    {
        IntPtr owner, group, dacl, sacl, descriptor;
        uint error = GetNamedSecurityInfo(path, 1, LabelInformation,
            out owner, out group, out dacl, out sacl, out descriptor);
        if (error != 0) throw new Win32Exception((int)error, "Cannot inspect integrity: " + path);
        try
        {
            if (sacl == IntPtr.Zero) return Medium; // Unlabelled Windows objects are medium.
            int count = (ushort)Marshal.ReadInt16(sacl, 4);
            int level = Int32.MaxValue;
            for (uint index = 0; index < count; index++)
            {
                IntPtr ace;
                if (!GetAce(sacl, index, out ace)) throw new Win32Exception(Marshal.GetLastWin32Error());
                // SYSTEM_MANDATORY_LABEL_ACE; skip inherit-only entries.
                if (Marshal.ReadByte(ace) == 0x11 && (Marshal.ReadByte(ace, 1) & 0x08) == 0)
                    level = Math.Min(level, SidLevel(IntPtr.Add(ace, 8)));
            }
            return level == Int32.MaxValue ? Medium : level;
        }
        finally { LocalFree(descriptor); }
    }

    public static void RejectReparsePath(string path)
    {
        // Refuse junctions/symlinks in any component, including ancestors.
        string cursor = Path.GetFullPath(path);
        while (!String.IsNullOrEmpty(cursor))
        {
            if ((File.GetAttributes(cursor) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidOperationException("Permission repair refuses linked paths: " + cursor);
            DirectoryInfo parent = Directory.GetParent(cursor);
            cursor = parent == null ? null : parent.FullName;
        }
    }
}
