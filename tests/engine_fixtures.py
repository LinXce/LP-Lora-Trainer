"""Small real engine-local interpreter fixtures, without training dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _terminate_under(directory):
    """Stop processes whose executable lives inside the fixture directory.

    A real installer subprocess may still be shutting down when a test ends; while
    it lives, its copied python.exe cannot be deleted. Only processes started from
    that exact directory are touched.
    """
    try:
        listing = subprocess.run(["wmic", "process", "get", "ProcessId,ExecutablePath", "/format:csv"],
                                 capture_output=True, text=True, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return
    prefix = str(Path(directory)).casefold()
    for line in listing.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[2].isdigit() or not parts[1]:
            continue
        if parts[1].casefold().startswith(prefix):
            try:
                subprocess.run(["taskkill", "/PID", parts[2], "/T", "/F"], capture_output=True, timeout=15,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError):
                pass


def cleanup_directory(directory, attempts=6, delay=0.5):
    """Delete a fixture directory, retrying while Windows still holds files.

    Fixture interpreters are copies of python.exe that were just executed, so the
    image can stay locked for a moment after the child exits; a plain
    ``TemporaryDirectory.cleanup()`` then fails in tearDown. Some harnesses also
    redirect TEMP into the workspace, where a failed cleanup would leave visible
    junk. Cleanup trouble is never a product defect, so this retries, stops any
    process still running from inside the directory, and finally gives up quietly.
    """
    for attempt in range(attempts):
        if attempt == 1:
            _terminate_under(directory)
        shutil.rmtree(str(directory), ignore_errors=True)
        if not Path(directory).exists():
            return True
        for path in Path(directory).rglob("*"):
            try:
                path.chmod(0o700)
            except OSError:
                pass
        time.sleep(delay)
    return not Path(directory).exists()


def engine_python(source):
    environment = Path(source) / '.venv'
    if os.name != 'nt':
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)], check=True)
        return environment / 'bin' / 'python'
    # Test-only portable Python: copy the executable/DLLs, share read-only stdlib.
    # No application dependencies are installed or used as a training environment.
    runtime = Path(sys.executable).parent
    target = environment / 'Scripts'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.executable, target / 'python.exe')
    for dll in runtime.glob('*.dll'):
        shutil.copy2(dll, target / dll.name)
    version = f'{sys.version_info.major}{sys.version_info.minor}'
    paths = [str(target), str(runtime / 'Lib'), str(runtime / 'DLLs'), str(runtime / f'python{version}.zip')]
    (target / f'python{version}._pth').write_text('\n'.join(paths) + '\n', encoding='utf-8')
    return target / 'python.exe'
