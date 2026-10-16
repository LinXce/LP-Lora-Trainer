<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · A Windows desktop LoRA training workbench

A standalone desktop application for managing LoRA training. Training outputs are primarily intended for ComfyUI, but ComfyUI is not required, installed, or launched.

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue)
![python](https://img.shields.io/badge/Python-3.11%2B-blue)
![vue](https://img.shields.io/badge/Vue-3-42b883)
![version](https://img.shields.io/badge/version-0.1.0-red)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## Features

- **Multi-engine training**: put engine source in `engine/`, scan and identify it, install its environment and train. Versions coexist, and a task is pinned to the installation it was submitted with.
- **Dataset management**: register a folder and scan its images, filter samples by problem type, and view or edit each caption with automatic backup before overwriting.
- **Parameters and config**: see the parameters available for the chosen engine and base model, then preview the final native config and command before submitting.
- **Training monitoring**: follow steps, loss, speed and logs live, in a single-GPU queue.
- **Job control**: cancel queued jobs, interrupt or force-stop running ones; a task that loses supervision is flagged and needs manual confirmation.
- **Results and delivery**: register checkpoints and sample images, verify that files are complete, and publish to a ComfyUI LoRA directory without overwriting existing files.

## Screenshots

### Overview and engine status

The overview summarizes recent tasks, the training queue and engine environments. The sidebar leads to datasets, prompt templates, new training, tasks, results, engine management and the terminal. The GPU card shows live device monitoring once it is enabled.

![Overview: task summary, GPU status and engine environments](assets/sample0.png)

### Dataset browser

The dataset page shows registered folders, scan results and paginated thumbnails; selecting an image lets you read or edit its caption. Scanning only builds an application index and never copies the dataset into the application directory.

![Dataset page: dataset list, thumbnails and caption editor](assets/sample2.png)

### Training task monitoring

The task detail page summarizes steps, loss, speed and elapsed time and offers raw logs and task information. Interrupt and force-stop actions are constrained by task state and process supervision; finished tasks can be reviewed on the results page.

![Training task page: loss chart, live logs and task status](assets/sample1.png)

### Typical workflow

1. In engine management, scan and confirm trusted engine source placed under `engine/`, then install and diagnose its environment.
2. In datasets, register a data folder, scan the images and confirm samples and captions.
3. In new training, pick a ready engine, base model and dataset, review parameters and the native config preview, then submit.
4. In tasks, follow progress, loss and logs; afterwards check models on the results page and publish what you need.

Adaptation coverage and open gaps are listed under "Engine support and limitations" below.

### Engine support and limitations

| Engine | Current support |
| --- | --- |
| Kohya / sd-scripts | Native single-GPU LoRA CLI training for SD1 / SD2 / SDXL, with config generation, log parsing and artifact registration. |
| AI Toolkit | Static discovery, instance/version management, environment diagnostics and its adapter install flow; **training submission is not implemented yet and cannot be submitted**. |
| Musubi Tuner | Cache pipeline (latents → text encoder → training) plus single-GPU LoRA training for 13 architectures, with config generation, log parsing and artifact registration. |

Kohya and Musubi Tuner training chains are covered by real-subprocess automated tests using simulated engines, but have **not** been validated end to end with real GPUs, models and upstream engines. Arbitrary upstream versions are not guaranteed to work, checkpoints are not guaranteed to survive interruption, and training-state recovery is not provided.

The application never downloads, updates or deletes engine repositories for you. After cloning or copying source, confirm "install engine environment" on the engine page without binding Python first: the app prepares its project-local uv and then runs that engine's own installation flow, streaming and persisting logs. When installation finishes it discovers and saves the engine interpreter, runs diagnostics and writes the state back; a failed diagnosis is never reported as success. The bundled `python_runtime` only runs standard-library bootstrap scripts and never holds training dependencies; the base Python is managed by the project-local uv, and each engine keeps its own `venv` / `.venv`. Kohya's official installer may initialize sd-scripts submodules, and AI Toolkit's official manager may install the Node or ffmpeg it needs.

## Build from source after cloning

The repository contains source code only: `python_runtime/`, `frontend/dist/`, `runtime/` and `engine/*` are all in `.gitignore` and **do not come with a clone**, so the project must be built once after cloning.

**Prerequisites**

- Windows x64 (the only supported target)
- Node.js 20+ and npm for the frontend build
- .NET Framework 4.x (`csc.exe`) to compile `LP-Lora-Trainer.exe`
- WebView2 Runtime to run the desktop window
- Git for cloning engine source and Kohya submodules
- Network access to GitHub and PyPI, and a few GB of free disk space
- **No system Python is required.** `install.cmd` fetches a CPython through the project-local uv.

**Two commands**

```bat
install.cmd   :: bootstrap uv -> fetch CPython -> create .venv -> install deps -> run build.cmd
build.cmd     :: build frontend -> build python_runtime -> compile EXE -> self-check
start.cmd     :: start the desktop window from python_runtime (or double-click LP-Lora-Trainer.exe)
```

- Build outputs: `frontend/dist/`, `python_runtime/`, `LP-Lora-Trainer.exe`.
- To **use the app on a machine without Python** instead of building from source, use a complete distribution folder and copy the whole directory including those three outputs, as described below.

## Portable use (Windows x64, no system Python)

The app ships a complete `python_runtime/`; it is not a `.venv` that depends on a system Python. Copy the **complete distribution directory** to another Windows x64 computer and double-click the windowed `LP-Lora-Trainer.exe`; no Python, pip, Node.js or environment activation is needed.

```text
LP-Lora-Trainer/
├─ LP-Lora-Trainer.exe         # windowed desktop entry, locates its interpreter by its own path
├─ start.cmd                   # command-line compatible entry (use the EXE on the desktop)
├─ start-api.cmd               # run only the loopback API
├─ check-runtime.cmd           # offline runtime self-check
├─ python_runtime/             # full CPython, standard library, app dependencies and licenses
│  ├─ python.exe
│  ├─ python312.dll            # named after the actual build version
│  ├─ python312._pth           # relative paths only; no site, env vars or user site
│  ├─ Lib/site-packages/
│  ├─ DLLs/
│  └─ runtime-manifest.json    # recorded Python and dependency versions
├─ app/, desktop/, supervisor/, adapters/
├─ frontend/dist/              # the built UI must travel with the app; no Node service runs
├─ scripts/
└─ engine/                     # engine source and each engine's own environment
```

**Update note:** closing the window does not stop the API or the supervisor. The launcher checks the backend install protocol, refuses to reconnect an older or incompatible backend and never starts a second one on the same port. After updating, stop installations and training, confirm the processes have exited, then exit the old API and supervisor of this project and reopen the EXE. Never force-kill the background while tasks are active.

The entry points **never fall back to a system Python**. The backend and the supervisor start child processes with the same bundled interpreter, so moving the folder to another drive needs no Python path repairs. The bundled version is CPython 3.12.15 with 22 application dependencies (see `python_runtime/runtime-manifest.json`); upgrading means rebuilding, and this record is not meant as "the newest Python".

**If the desktop window does not open:** the launcher first checks the .NET/WebView bridge the desktop rendering needs. When it is available only the desktop window is shown; when it is missing (or WebView2 initialization / backend bootstrap fails after the window opened) it switches to **terminal + browser** mode: a window from the **system terminal** prints the `assets/logo.txt` wordmark, explains the reason and opens `http://127.0.0.1:5900` in the default browser, with everything logged to `runtime/console.log` (never failing silently). **Closing that terminal window (or Ctrl+C / Enter) shuts the backend and supervisor down.** `--browser` forces this mode. The only case without fallback is a port already serving an older backend or one using another data directory (exit it as prompted first). To quit everything including the supervisor, use "exit backend and supervisor" on the settings page, which is allowed only when no task is active or unverified.

```powershell
.\LP-Lora-Trainer.exe
.\start-api.cmd
# environment self-check without pausing
.\python_runtime\python.exe scripts\check_runtime.py
# custom data directory
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
```

The Windows desktop still requires WebView2 Runtime and a usable .NET Framework; training additionally needs GPU drivers and each engine's own dependencies. **No system Python does not mean no operating-system components**; the project does not install those for you.

The default API/UI address is `http://127.0.0.1:5900`. Closing the window does not end the API, the supervisor or an active training run. Stop training from the task page (interrupt or force-stop); a lost-supervision task must have its original process and children verified as exited before "confirm exited" is enabled.

### Development and producing a distribution

Only a build/development machine needs a full Windows x64 Python 3.11+; building the frontend additionally needs Node.js and npm. Ordinary users just copy a finished directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,test]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
# generate the runtime offline from the local full CPython and installed dependencies
.\.venv\Scripts\python.exe scripts\build_python_runtime.py
powershell -ExecutionPolicy Bypass -File scripts\build_launcher.ps1
.\LP-Lora-Trainer.exe
```

`build_python_runtime.py` copies the real CPython interpreter, standard library, required native libraries and the application dependency closure, keeping licenses and the version manifest while excluding the virtual-environment interpreter, editable mappings, pip and training dependencies. It refuses to overwrite an existing runtime by default; with `--replace` it self-checks the new one first and keeps the old one as a backup. Maintenance details are in `scripts/README.md`.

`LP-Lora-Trainer.exe`, `python_runtime/` and `frontend/dist/` are build artifacts: **downloading the source repository alone cannot be run by double-clicking**. A distribution must include them and does not need the development `.venv` or `frontend/node_modules`. The launcher source is `desktop/launcher_host.cs`, rebuildable with `scripts/build_launcher.ps1`.

### Moving the project directory or changing computers

The portable entry resolves directories from its actual location and binds to no drive letter. Stop training and exit the related background processes before copying the whole directory; the portable runtime needs no activation and no `repair_environment.py`.

Registered engines, Python interpreters, datasets, models, outputs and publish destinations inside the database may still be absolute paths and have to be re-confirmed and rebound in the UI; do not bulk-replace the pinned bindings or native configs of historical tasks.

**Engine portability is a separate matter:** the base Python is managed by the project-local uv and does not depend on a system Python, but an installed `venv` / `.venv` may still hold absolute paths and is not guaranteed to work after moving to another computer or drive. Reinstall, or rebind and diagnose. No fully relocatable training-environment package is produced today, and training dependencies are never folded into the application runtime.

If the development `.venv` only moved drive letters and its base Python is still present, `scripts/repair_environment.py` can repair it; that is not part of the portable delivery flow.

### Adding an engine

```text
engine/
├─ kohya_ss/          # contains sd-scripts/, or place sd-scripts source directly
├─ kohya_ss-old/      # another version, still served by adapters/kohya/
├─ ai-toolkit/        # environment install/management; training adaptation not implemented
└─ musubi-tuner/      # environment install/management; cache pipeline + training (13 architectures)
```

1. Copy trusted engine source into `engine/<any instance name>/`.
2. Rescan on the engine page and confirm the detected type.
3. Click "install engine environment" and confirm you trust the source; there is no need to pick Python first. For Musubi, CUDA extras show only what that version declares, while Kohya and AI Toolkit follow their official installers.
4. Watch or stop the installation in the terminal. The flow discovers and saves the engine interpreter, then diagnoses it; on failure check the logs and retry, or bind an existing environment manually and diagnose it. Only engines that implement training adaptation and pass diagnosis can accept submissions.
5. Add and scan a dataset, choose a local base model, then validate and submit from the new-training page.

Kohya dataset adaptation currently requires images to sit directly in the registered directory; subdirectories are explicitly refused rather than silently dropping samples. The output directory may not live inside engine source, a dataset, a base model or the application `data/`, and every task gets its own output subdirectory.

### Data and settings

- `data/state.sqlite`: index of engines, datasets, tasks and artifacts.
- `data/jobs/<task-id>/`: configs, bindings, command records, logs, metrics and artifact records.
- `data/manifests/`: environment diagnosis records.
- `data/cache/`: thumbnails and other caches.
- `data/installations/<session-id>/output.log`: complete installation logs; clearing the terminal view does not delete them.
- `runtime/desktop.log`, `runtime/backend.log`: desktop errors and backend startup logs.
- `runtime/uv/`: project-local uv, base Python and download cache, kept apart from application dependencies.
- GPU monitoring is off by default; when enabled it uses low-frequency `nvidia-smi` queries and reports unknown when unavailable.

`data_root` cannot be switched while the service is running. To migrate, stop training and the background services, copy the data, then point at the new directory:

```powershell
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
```

## Validation

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
.\python_runtime\python.exe scripts\check_runtime.py
npm.cmd --prefix frontend run build
```

Windows process-tree stop tests require the system to permit `taskkill`. Tests cover real HTTP, installation subprocesses, logs, failure retries, stopping a process tree, environment path write-back and diagnostics; GPU and dependency diagnosis use fixtures and never download large training packages. The portable runtime carries no build tools, so three builder-only tests are skipped there; on a build machine run them with a Python that has the application dependencies and `packaging` installed: `python -m unittest tests.test_portable_runtime.RuntimeBuilderTests -v`. The EXE has been verified to start the desktop and reconnect the backend, and the frontend type check and production build pass; none of that is a full upstream installation, GPU-training or performance validation.

## License

[Apache License 2.0](LICENSE)
