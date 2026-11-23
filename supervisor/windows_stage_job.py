"""Account for every descendant of a newly launched Windows training stage.

The unnamed job has no kill-on-close or breakaway flags. The primary thread is
suspended until assignment, so even a very fast launcher cannot spawn a child
before the OS starts tracking the tree.
"""
import ctypes
from ctypes import wintypes as w
import subprocess
import time


CREATE_SUSPENDED = 0x00000004
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1


class _ThreadEntry(ctypes.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("thread_id", w.DWORD),
                ("owner_pid", w.DWORD), ("base_priority", w.LONG),
                ("priority_delta", w.LONG), ("flags", w.DWORD)]


class _Accounting(ctypes.Structure):
    _fields_ = [("user", ctypes.c_int64), ("kernel", ctypes.c_int64),
                ("period_user", ctypes.c_int64), ("period_kernel", ctypes.c_int64),
                ("page_faults", w.DWORD), ("total", w.DWORD), ("active", w.DWORD),
                ("terminated", w.DWORD)]


def _kernel():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, result, args in (
        ("CreateJobObjectW", w.HANDLE, [w.LPVOID, w.LPCWSTR]),
        ("AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE]),
        ("QueryInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPVOID]),
        ("CreateToolhelp32Snapshot", w.HANDLE, [w.DWORD, w.DWORD]),
        ("Thread32First", w.BOOL, [w.HANDLE, ctypes.POINTER(_ThreadEntry)]),
        ("Thread32Next", w.BOOL, [w.HANDLE, ctypes.POINTER(_ThreadEntry)]),
        ("OpenThread", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
        ("ResumeThread", w.DWORD, [w.HANDLE]),
        ("CloseHandle", w.BOOL, [w.HANDLE]),
    ):
        fn = getattr(api, name)
        fn.argtypes, fn.restype = args, result
    return api


def _error():
    return ctypes.WinError(ctypes.get_last_error())


class StageJob:
    def __init__(self):
        self.api = _kernel()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise _error()

    def _resume_initial_thread(self, pid):
        # Popen owns the live process handle but already closed the initial
        # thread handle. Locate that thread while the new process is suspended.
        snapshot = self.api.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snapshot == w.HANDLE(-1).value:
            raise _error()
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = self.api.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.owner_pid == pid:
                    thread = self.api.OpenThread(THREAD_SUSPEND_RESUME, False, entry.thread_id)
                    if not thread:
                        raise _error()
                    try:
                        previous = self.api.ResumeThread(thread)
                        if previous == 0xFFFFFFFF:
                            raise _error()
                    finally:
                        self.api.CloseHandle(thread)
                    if previous == 1:
                        return
                    if previous > 1:
                        raise OSError("初始线程仍被其他组件挂起，取消启动训练")
                entry.size = ctypes.sizeof(entry)
                found = self.api.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("无法取得新建训练进程的初始线程")
        finally:
            self.api.CloseHandle(snapshot)

    def start(self, argv, **options):
        options["creationflags"] = options.get("creationflags", 0) | CREATE_SUSPENDED
        process = subprocess.Popen(argv, **options)
        try:
            if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise _error()
            self._resume_initial_thread(process.pid)
        except BaseException:
            # No engine code executed without a job. Release only the process
            # just created suspended, never an existing or recovered PID.
            process.kill()
            process.wait()
            if process.stdout:
                process.stdout.close()
            raise
        return process

    def empty(self, timeout=0):
        deadline = time.monotonic() + timeout
        while True:
            stats = _Accounting()
            if not self.api.QueryInformationJobObject(self.handle, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                                                      ctypes.byref(stats), ctypes.sizeof(stats), None):
                raise _error()
            if stats.active == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
