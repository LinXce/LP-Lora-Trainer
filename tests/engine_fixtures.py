"""Small real engine-local interpreter fixtures, without training dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


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
