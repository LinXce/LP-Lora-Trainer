"""Opt-in Windows four-case close acceptance. Run with the bundled python.exe.

Starts real WebView2, FastAPI, supervisor and a CPU-only fixture trainer. Uses a
fresh copy of the application code, session profile, data store and random port
for each cell. The driver targets only its own form, never the foreground window.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ID = "a" * 32
QUEUED_ID = "b" * 32


def wait_for(fn, label, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.15)
    raise AssertionError(f"Timeout waiting for {label}")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def case_job():
    """Kill only this test's process group if an assertion/timeout fires."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    handle = wintypes.HANDLE(handle)
    class BasicLimits(ctypes.Structure):
        _fields_ = [("per_process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("max_processes", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class Limits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io_counters", ctypes.c_ulonglong * 6),
                    ("memory_limits", ctypes.c_size_t * 4)]
    limits = Limits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle


def portable_python(target, search_paths):
    runtime = ROOT / "python_runtime"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("python.exe", "python.exe.config"):
        shutil.copy2(runtime / name, target / name)
    for library in runtime.glob("*.dll"):
        shutil.copy2(library, target / library.name)
    source_pth, = runtime.glob("python*._pth")
    (target / source_pth.name).write_text(
        "\n".join(map(str, [target, *search_paths, runtime / "Lib", runtime / "DLLs"])),
        encoding="utf-8",
    )
    return target / "python.exe"


def stage(workspace, dist):
    for package in ("app", "adapters", "desktop", "supervisor"):
        for path in (ROOT / package).rglob("*"):
            if path.suffix in {".py", ".json"} and path.is_file():
                copy = workspace / path.relative_to(ROOT)
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy)
    shutil.copytree(dist, workspace / "frontend" / "dist")
    (workspace / "runtime").mkdir()
    (workspace / "engine").mkdir()
    runtime = ROOT / "python_runtime"
    return portable_python(workspace / "python_runtime", [workspace, runtime / "Lib" / "site-packages"])


ENGINE = """
import json, os, struct, time
from pathlib import Path
job = Path(os.environ['ANIMA_DAEMON_JOB_DIR'])
out = Path(json.loads((job / 'job.json').read_text(encoding='utf-8'))['output_dir'])
out.mkdir(parents=True, exist_ok=True)
step = 0
while step < 3000:
    step += 1
    (out / 'ticks').write_bytes(str(step).encode())
    if (job / 'pause.request').is_file():
        time.sleep(0.7)
        header = json.dumps({'fixture':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        (out / 'fixture.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(4))
        (out / 'graceful-exit').write_text(str(os.getpid()))
        break
    time.sleep(0.1)
"""


def seed_jobs(workspace):
    from adapters.registry import ADAPTERS
    from app.services.common import now
    from app.services.engines import DIAGNOSTIC_CODE, revision
    from app.services.training import PROGRESS
    from app.storage.state import StateStore

    source = workspace / "engine" / "fixture"
    python = portable_python(source / ".lp" / "python", [])
    (python.parent / "torch.py").write_text(
        "from types import SimpleNamespace\n__version__='fixture'\n"
        "version=SimpleNamespace(cuda='fixture')\ncuda=SimpleNamespace(is_available=lambda:True)\n"
    )
    (source / "train.py").write_text(ENGINE, encoding="utf-8")
    store = StateStore(workspace / "data")
    adapter = ADAPTERS["anima_lora"]
    command = [str(python), "-u", str(source / "train.py")]
    diagnostics = subprocess.check_output(
        [python, "-I", "-c", DIAGNOSTIC_CODE, *adapter.diagnostic_modules, *adapter.optional_modules],
        cwd=source, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    write_json(store.root / "manifests" / "fixture.json", json.loads(diagnostics.split("LP_DIAG=")[1]))
    installation = dict(installation_id="fixture", engine_id="anima_lora", source_path=str(source),
                        management_mode="user_managed", revision=revision(source), state="ready",
                        verification="experimental", environment_id="fixture", python_executable=str(python))
    for key in (ACTIVE_ID, QUEUED_ID):
        job = store.root / "jobs" / key
        output = str(workspace / "output" / key)
        draft = dict(name=key, architecture="anima", base_model_path="fixture",
                     output_dir=output, params={"torch_compile": False})
        write_json(job / "job.json", draft)
        (job / "config.native.toml").write_text("torch_compile = false\n")
        write_json(job / "stages.json", [command])
        write_json(job / "command.json", {"commands": [command], "cwd": str(source)})
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in job.iterdir()}
        binding = dict(installation_id="fixture", environment_id="fixture", environment_manifest_id="fixture",
                       revision=installation["revision"], adapter_version=adapter.adapter_version,
                       schema_version="1", installation=installation, config_hashes=hashes)
        write_json(job / "binding.json", binding)
        store.put("task", key, dict(task_id=key, name=key, state="queued", engine_id="anima_lora",
                  installation_id="fixture", installation_label="fixture", architecture="anima",
                  dataset_name="fixture", output_dir=output, created_at=now(), started_at=None,
                  finished_at=None, progress=PROGRESS.copy(), error_summary=None, draft=draft, binding=binding,
                  recovery={"graceful_stop": True, "resume_full_state": True, "resume_from_weights": False},
                  stop_requested=None, pid=None))


def launch_gui(workspace, port, owner, gesture, allow_system_command=False, idle=False):
    import webview
    from desktop import launcher, window as desktop_window
    from app.storage.state import StateStore

    assert launcher.__file__.startswith(str(workspace) + os.sep)
    native_create = webview.create_window
    native_resize = desktop_window.enable_window_resize
    title = f"LP Close Acceptance {port}"
    # Cosmetic title lookup must never select another LP window on this desktop.
    desktop_window.enable_window_resize = lambda *_: native_resize(title)
    store = StateStore(workspace / "data")
    out = workspace / "output" / ACTIVE_ID
    requests = []
    engine_handle = None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    http_open = urllib.request.OpenerDirector.open

    def record_request(opener, request, *args, **kwargs):
        if isinstance(request, urllib.request.Request):
            path = urllib.parse.urlsplit(request.full_url).path
            if path.startswith("/api/v1/system/shutdown"):
                requests.append((request.get_method(), path))
        return http_open(opener, request, *args, **kwargs)

    urllib.request.OpenerDirector.open = record_request

    def tick():
        try:
            return int((out / "ticks").read_bytes())
        except (OSError, ValueError):
            return 0

    def exercise(window):
        nonlocal engine_handle
        try:
            close_button = ".window-bar__button--close"
            dialog = "dialog[open] .modal__foot"
            node = lambda selector: "document.querySelector(" + json.dumps(selector) + ")"
            js = window.evaluate_js
            wait_for(lambda: js(f"!!{node(close_button)}"), "built Vue title bar and desktop bridge")
            js("""window.exitFrames = {frames: 0, moving: false, reduced: matchMedia('(prefers-reduced-motion: reduce)').matches};
                function sampleExitFrame() {
                    const spinner = document.querySelector('dialog.closing-veil[open] .closing-spinner');
                    if (spinner && spinner.getBoundingClientRect().width > 0) {
                        const transform = getComputedStyle(spinner).transform;
                        const seen = window.exitFrames;
                        if (seen.frames && seen.last !== transform) seen.moving = true;
                        seen.last = transform;
                        seen.frames++;
                    }
                    requestAnimationFrame(sampleExitFrame);
                }
                requestAnimationFrame(sampleExitFrame); true""")
            wait_for(lambda: (store.root / "supervisor.json").is_file(), "isolated supervisor")
            if not idle:
                seed_jobs(workspace)
                wait_for(lambda: tick() >= 3, "fake engine process emitting ticks")
                engine_pid = store.get("task", ACTIVE_ID)["pid"]
                engine_handle = ctypes.c_void_p(kernel.OpenProcess(0x00100000, False, engine_pid))
                assert engine_handle.value, "Could not open fixture engine synchronization handle"
            routes = []

            def request_close():
                if gesture == "x":
                    js(f"setTimeout(() => {node(close_button)}.click(), 0); true")
                    routes.append("Vue.click")
                    return
                from ctypes import wintypes
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                hwnd = wintypes.HWND(window.native.Handle.ToInt64())
                owner_pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
                assert owner_pid.value == os.getpid()
                # WinForms reads the modifier state of its UI thread. A bare
                # WM_SYSKEYDOWN is not a reliable Alt+F4 on a background form.
                # Set only this thread's key state, never send global input.
                from System import Action
                def shortcut():
                    window.native.Activate()
                    user32.SetForegroundWindow(hwnd)
                    user32.SetActiveWindow(hwnd)
                    original = (ctypes.c_ubyte * 256)()
                    assert user32.GetKeyboardState(original)
                    held = (ctypes.c_ubyte * 256).from_buffer_copy(original)
                    held[0x12] = held[0xa4] = 0x80
                    assert user32.SetKeyboardState(held)
                    try:
                        user32.SendMessageW(hwnd, 0x104, 0x73, 0x203e0001)
                        user32.SendMessageW(hwnd, 0x105, 0x73, 0xe03e0001)
                    finally:
                        user32.SetKeyboardState(original)
                window.native.Invoke(Action(shortcut))
                # An inactive desktop may discard a synthetic keyboard gesture
                # instead of translating it. Exercise the keyboard-origin system
                # command directly in that case, and report the exact test level.
                def received():
                    selector = ".closing-veil[open]" if idle else dialog
                    return (workspace / "exit-animation.json").exists() or js(f"!!{node(selector)}")
                try:
                    wait_for(received, "Alt+F4 translation", 4)
                except AssertionError:
                    # Some systems filter a directly delivered system key on an
                    # inactive WinForms window. Let Windows' own default window
                    # procedure translate that same Alt+F4 into its system command.
                    def default_alt_f4():
                        user32.DefWindowProcW(hwnd, 0x0104, 0x73, 0x203e0001)
                    window.native.Invoke(Action(default_alt_f4))
                    try:
                        wait_for(received, "DefWindowProcW Alt+F4 translation", 4)
                        routes.append("DefWindowProcW/Alt+F4")
                        return
                    except AssertionError:
                        if not allow_system_command:
                            raise
                    user32.PostMessageW(hwnd, 0x0112, 0xF060, 0)  # WM_SYSCOMMAND / SC_CLOSE
                    routes.append("keyboard-SC_CLOSE")
                else:
                    routes.append("WM_SYSKEYDOWN/F4")

            def ask():
                request_close()
                wait_for(lambda: js(f"!!{node(dialog)}"), f"{gesture} close confirmation")
                label = js(f"{node(dialog + ' button:last-child')}.textContent.trim()")
                if owner == "foreign":
                    assert "仅关闭" in label, label
                else:
                    assert "训练" in label and "退出" in label, label
                return label

            if idle:
                request_close()
                write_json(workspace / "answered.json", {"routes": routes})
                return
            label = ask()
            js(f"{node(dialog + ' .btn--ghost')}.click()")
            after_cancel = tick()
            wait_for(lambda: tick() > after_cancel + 2 and not js(f"!!{node(dialog)}"), "cancel retained window and trainer")
            assert js("window.exitFrames.frames") == 0, "cancel displayed the exit animation"
            assert store.get("task", ACTIVE_ID)["stop_requested"] is None
            assert not any(method == "POST" for method, _ in requests), requests
            assert store.get("task", QUEUED_ID)["state"] == "queued"
            assert ask() == label
            write_json(workspace / "answered.json", {"label": label, "tick_after_cancel": after_cancel, "routes": routes})
            js(f"setTimeout(() => {node(dialog + ' button:last-child')}.click(), 0); true")
        except BaseException:
            write_json(workspace / "http-at-error.json", requests)
            (workspace / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    def create(_title, *args, **kwargs):
        window = native_create(title, *args, **kwargs, focus=False)
        destroy = window.destroy
        def record_animation_then_destroy():
            try:
                write_json(workspace / "exit-animation.json", window.evaluate_js("window.exitFrames"))
            finally:
                destroy()
        window.destroy = record_animation_then_destroy
        window.events.shown += lambda: threading.Thread(target=exercise, args=(window,), daemon=True).start()
        def closed():
            write_json(workspace / "window-closed.json", {
                "active": store.get("task", ACTIVE_ID), "queued": store.get("task", QUEUED_ID),
                "checkpoint": (out / "graceful-exit").exists(), "ticks": tick(),
                "engine_exited": bool(engine_handle) and kernel.WaitForSingleObject(engine_handle, 0) == 0,
            })
        window.events.closed += closed
        return window

    webview.create_window = create
    # Browser fallback is a separate product and would invalidate a desktop test.
    launcher.spawn_console = lambda *args: (_ for _ in ()).throw(AssertionError("GUI startup failed: " + str(args[2])))
    sys.argv[:] = ["desktop.launcher", "--port", str(port), "--data-root", str(store.root)]
    try:
        launcher.main()
    finally:
        if engine_handle:
            kernel.CloseHandle(engine_handle)
    write_json(workspace / "http-requests.json", requests)


def case(workspace, dist, owner, gesture, allow_system_command=False, idle=False):
    from app.storage.state import StateStore
    from desktop import launcher

    python = stage(workspace, dist)
    store = StateStore(workspace / "data")
    store.put("meta", "settings", dict(engine_root=str(workspace / "engine"), gpu_monitor=False, log_tail_lines=500,
                                       proxy_enabled=False, proxy_url=None))
    store.put("meta", "gpu_monitor_initialized", True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert port != 5900
    url = f"http://127.0.0.1:{port}"
    kernel, job = case_job()
    children = []
    with (workspace / "acceptance.log").open("ab", buffering=0) as log:
        def spawn(args):
            process = subprocess.Popen([python, "-B", *args], cwd=workspace, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            children.append(process)
            if not kernel.AssignProcessToJobObject(job, ctypes.c_void_p(int(process._handle))):
                process.kill()
                raise ctypes.WinError(ctypes.get_last_error())
            return process

        try:
            if owner == "foreign":
                foreign = spawn(["-m", "app.api.server", "--port", str(port), "--data-root", str(store.root)])
                wait_for(lambda: launcher.backend_ready(url, store.root), "dedicated reusable backend")
            test_script = Path(__file__).resolve()
            gui = spawn([str(test_script), "--gui", str(workspace), str(port), owner, gesture]
                        + (["--allow-system-command"] if allow_system_command else [])
                        + (["--idle"] if idle else []))
            def done():
                error = workspace / "error.txt"
                if error.exists():
                    raise AssertionError(error.read_text(encoding="utf-8"))
                return gui.poll() is not None
            wait_for(done, f"{owner}/{gesture} desktop exit", 120)
            assert gui.returncode == 0, (workspace / "acceptance.log").read_text(encoding="utf-8", errors="replace")[-2000:]
            closed = json.loads((workspace / "window-closed.json").read_text(encoding="utf-8"))
            animation = json.loads((workspace / "exit-animation.json").read_text(encoding="utf-8"))
            assert animation["frames"] >= 2 and (animation["reduced"] or animation["moving"]), animation
            routes = json.loads((workspace / "answered.json").read_text(encoding="utf-8"))["routes"]
            requests = json.loads((workspace / "http-requests.json").read_text(encoding="utf-8"))
            if idle:
                assert not store.list("task")
                assert launcher.backend_ready(url, store.root) == (owner == "foreign")
                assert any(method == "POST" for method, _ in requests) == (owner == "owned")
                print(f"PASS idle {owner}/{gesture} animation={animation} routes={routes}", flush=True)
                return
            assert closed["queued"]["state"] == store.get("task", QUEUED_ID)["state"] == "queued"
            assert not (workspace / "output" / QUEUED_ID).exists()
            if owner == "owned":
                assert closed["engine_exited"] and closed["checkpoint"] and closed["active"]["state"] == "stopped", closed
                artifacts = store.list("artifact")
                assert any(item["kind"] == "checkpoint" and item["complete"] for item in artifacts)
                assert any(m == "POST" for m, _ in requests)
                assert not launcher.backend_ready(url, store.root)
            else:
                assert foreign.poll() is None and launcher.backend_ready(url, store.root)
                assert closed["active"]["state"] == "running" and not closed["checkpoint"] and not closed["engine_exited"]
                ticks_path = workspace / "output" / ACTIVE_ID / "ticks"
                def still_training():
                    value = ticks_path.read_bytes()
                    return value.isdigit() and int(value) > closed["ticks"] + 3
                wait_for(still_training, "trainer advanced after foreign desktop process exit")
                assert store.get("task", ACTIVE_ID)["stop_requested"] is None
                assert not any(m == "POST" for m, _ in requests), requests
            print(f"PASS {owner}/{gesture} port={port} close_tick={closed['ticks']} "
                  f"task={closed['active']['state']} engine_exited={closed['engine_exited']} "
                  f"checkpoint={closed['checkpoint']} queue=queued shutdown_POSTs="
                  f"{sum(m == 'POST' for m, _ in requests)} animation={animation} routes={routes}", flush=True)
        finally:
            kernel.CloseHandle(job)
            for child in children:
                child.wait(timeout=15)


def main():
    if os.name != "nt":
        raise SystemExit("This acceptance driver requires Windows WebView2.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", nargs=4, help=argparse.SUPPRESS)
    parser.add_argument("--dist", type=Path, default=ROOT / "frontend" / "dist")
    parser.add_argument("--case", choices=("owned/x", "owned/alt-f4", "foreign/x", "foreign/alt-f4"))
    parser.add_argument("--idle", action="store_true", help="Exercise the immediate exit path with no training tasks.")
    parser.add_argument("--allow-system-command", action="store_true",
                        help="Allow separately labelled SC_CLOSE fallback if the desktop filters synthetic keys.")
    args = parser.parse_args()
    if args.gui:
        root, port, owner, gesture = args.gui
        launch_gui(Path(root), int(port), owner, gesture, args.allow_system_command, args.idle)
        return
    if not (args.dist / "index.html").is_file():
        raise SystemExit("Build the current frontend first or pass --dist.")
    for name in ([args.case] if args.case else ["owned/x", "owned/alt-f4", "foreign/x", "foreign/alt-f4"]):
        workspace = Path(tempfile.mkdtemp(prefix="lp-close-check-")).resolve()
        print(f"{name} isolated workspace: {workspace}", flush=True)
        try:
            case(workspace, args.dist.resolve(), *name.split("/"), args.allow_system_command, args.idle)
        except BaseException:
            print(f"Retained private test logs at {workspace}", flush=True)
            raise
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
