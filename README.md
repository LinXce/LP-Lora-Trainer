<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · A Windows desktop LoRA training workbench

A standalone desktop application for managing LoRA training. Training outputs are primarily intended for ComfyUI, but ComfyUI is not required, installed, or launched by this application.

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue)
![python](https://img.shields.io/badge/Python-3.11%2B-blue)
![vue](https://img.shields.io/badge/Vue-3-42b883)
![version](https://img.shields.io/badge/version-0.1.0-red)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## Features

- Discover engine installations, confirm engine types, bind Python interpreters, diagnose environments, and track source revisions.
- Register and scan datasets, browse paginated thumbnails, filter scan issues, and edit captions with automatic backups before overwriting.
- Validate training parameters, preview native configs, submit and monitor jobs, and inspect logs and loss metrics.
- Cancel queued jobs, interrupt or force-stop running jobs, and require manual confirmation when process supervision is lost.
- Register checkpoints and sample images, verify selected artifacts, and publish them atomically to a ComfyUI LoRA directory without overwriting existing files.
- Save settings, use native path pickers, open paths in Explorer, reconnect the desktop UI, and refresh task history after SSE reconnects.

## Screenshots

### Overview and engine status

The overview summarizes recent tasks, the training queue, and engine environments. Navigation includes datasets, prompt templates, training setup, tasks, results, engine management, and the terminal. GPU monitoring is shown when enabled.

![Overview: task summary, GPU status, and engine environments](assets/sample0.png)

### Dataset browser

Browse registered folders, scan results, and paginated image thumbnails. Select an image to view or edit its caption. Scanning creates an application index; it does not copy the whole dataset into the application directory.

![Dataset page: dataset list, thumbnails, and caption editor](assets/sample2.png)

### Training task monitoring

The task detail page shows steps, loss, speed, elapsed time, and raw logs. Interrupt and force-stop actions are constrained by task state and process supervision. Finished artifacts can be reviewed on the results page.

![Training task page: loss chart, live logs, and task status](assets/sample1.png)

## Typical workflow

1. Put trusted engine source code under `engine/`, then scan, confirm, install, and diagnose its environment in Engine Management.
2. Register and scan a dataset; review images and captions.
3. Select a ready engine, base model, and dataset in New Training. Validate parameters and review the native config before submitting.
4. Monitor progress, loss, and logs in Tasks. Review checkpoints and publish selected outputs from Results.

## Engine support and limitations

| Engine | Current support |
| --- | --- |
| Kohya / sd-scripts | Native single-GPU LoRA training for SD1, SD2, and SDXL, with config generation, log parsing, and artifact registration. Other declared architectures are subject to adapter support and validation. |
| AI Toolkit | Static discovery, instance/version management, environment diagnostics, and its native installation flow. **Training submission is still being adapted and is not available.** |
| Musubi Tuner | Cache pipeline (latents → text encoder → training) and single-GPU LoRA training for 13 architectures, with config generation, log parsing, and artifact registration. |

Kohya and Musubi training flows have automated subprocess tests using simulated engines. They have **not** been end-to-end validated with real GPUs, models, and upstream engine installations. Arbitrary upstream versions are not guaranteed to work. Checkpoint behavior on interruption and automatic training-state recovery are not guaranteed.

The application does not download, update, or delete engine repositories. Engine source is placed under `engine/`; the app prepares its project-local `uv`, invokes the engine's own installation flow, streams and saves installation logs, then discovers and diagnoses the engine interpreter. A failed diagnosis is never reported as success. Each engine keeps its own `venv` or `.venv`; training dependencies are not installed into the application's bundled runtime. Kohya's installer may initialize sd-scripts submodules. AI Toolkit's manager may install Node or ffmpeg. See [Installation and runtime details](docs/development/运行与安装.md).

## Build from source after cloning

The repository contains source code only. `python_runtime/`, `frontend/dist/`, `runtime/`, and `engine/*` are ignored by Git and are not included in a fresh clone. Build the project before launching it.

### Prerequisites

- Windows x64
- Node.js 20+ and npm for the frontend build
- .NET Framework 4.x (`csc.exe`) to compile the desktop launcher
- WebView2 Runtime to run the desktop window
- Git for cloning engine repositories and initializing Kohya submodules
- Network access to GitHub and PyPI; allow several GB of free disk space
- **A system Python is not required.** `install.cmd` bootstraps project-local `uv` and downloads CPython.

### Install, build, and run

```bat
install.cmd
```

This bootstraps `uv`, installs managed CPython into the project runtime area, creates `.venv`, installs the app dependencies, and runs `build.cmd`. Then launch with:

```bat
start.cmd
```

Or build explicitly after preparing the environment:

```bat
install.cmd --no-build
build.cmd
```

Useful options:

- `install.cmd --no-build` prepares the development environment without building deliverables.
- `install.cmd --index <package-index-url>` uses an explicitly selected package index, for example `install.cmd --index https://pypi.org/simple`.
- `build.cmd --no-pause` is intended for automation.

`requirements.lock` pins the validated application dependency versions used to build the bundled runtime. `install.cmd` prefers it when present; removing it makes `uv` resolve the ranges in `pyproject.toml` instead. The lock does not contain machine-specific paths.

The build produces `frontend/dist/`, `python_runtime/`, and `LP-Lora-Trainer.exe`. `uv` is stored at `runtime/uv/uv.exe` and is also used by the engine installation service. If the automatic `uv` download fails, `install.cmd` prints the manual download location and target path.

## Portable use

A portable distribution must include the complete application directory, especially `LP-Lora-Trainer.exe`, `python_runtime/`, and `frontend/dist/`. Copy the whole directory to another Windows x64 computer and launch the EXE. No system Python, pip, Node.js, or activated environment is needed at runtime.

```text
LP-Lora-Trainer/
├─ LP-Lora-Trainer.exe
├─ start.cmd
├─ start-api.cmd
├─ check-runtime.cmd
├─ python_runtime/       # CPython, standard library, app dependencies, manifest
├─ app/ desktop/ supervisor/ adapters/
├─ frontend/dist/        # built UI
├─ scripts/
└─ engine/               # engine source and per-engine environments
```

The launcher does not fall back to a system Python. If the desktop bridge cannot initialize, it can fall back to a terminal and browser workflow. WebView2 Runtime and a usable .NET Framework are still required for the desktop window. GPU drivers and engine-specific dependencies are required for training; the app does not install Windows system components.

Closing the window does not necessarily stop the API, supervisor, or active training. Use the task page to interrupt or force-stop training. A task reported as `connection_lost` must be manually verified before it can be cleared.

The application does not update or delete engine repositories. Engine environments can contain absolute paths and may not be portable across computers or drive letters; reinstall or rebind and diagnose them after migration. Database records for engines, datasets, models, outputs, and publishing destinations may also need to be reselected after migration.

## Data locations

- `data/state.sqlite`: engine, dataset, task, and artifact indexes.
- `data/jobs/<task-id>/`: task configs, bindings, commands, logs, metrics, and artifact records.
- `data/manifests/`: environment diagnosis records.
- `data/cache/`: thumbnails and other application caches.
- `data/installations/<session-id>/output.log`: persistent installation logs.
- `runtime/desktop.log` and `runtime/backend.log`: desktop and backend startup logs.
- `runtime/uv/`: project-local uv, managed Python, and download cache.

The `data_root` cannot be changed while the service is running. Stop training and background services before moving data, then start with a new data directory:

```powershell
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
```

## Validation

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
.\python_runtime\python.exe scripts\check_runtime.py
npm.cmd --prefix frontend run build
```

Windows process-tree tests require permission to run `taskkill`. Tests use fixtures for GPU and dependency diagnosis and do not download large training packages. Three builder-only tests are skipped in the portable runtime, which does not include build tools; on a build machine with app dependencies and `packaging` installed, run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_portable_runtime.RuntimeBuilderTests -v
```

These checks do not constitute validation of upstream engine installation, real-GPU training, or performance.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
