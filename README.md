<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · A Windows desktop LoRA training workbench

A standalone desktop application for managing LoRA training. Training outputs are primarily intended for ComfyUI, but ComfyUI is not required, installed, or launched.

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue?style=flat)
![python](https://img.shields.io/badge/python-3.12%20bundled-blue?style=flat)
![vue](https://img.shields.io/badge/vue-3-42b883?style=flat)
![version](https://img.shields.io/badge/version-1.1.0-red?style=flat)
![license](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## Features

- **Multi-engine training**: put engine source in `engine/`, scan and identify it, install its environment and train. Versions coexist, and a task is pinned to the installation it was submitted with.
- **Dataset management**: register a folder and scan its images, filter samples by problem type, and view or edit each caption with automatic backup before overwriting.
- **Parameters and config**: see the parameters available for the chosen engine and base model, search them by name or engine key, preview the final native config and command before submitting.
- **Training monitoring**: follow steps, loss, speed and logs live, in a single-GPU queue.
- **Job control**: cancel queued jobs, interrupt or force-stop running ones; a task that loses supervision is flagged and needs manual confirmation.
- **Results and delivery**: register checkpoints and sample images, verify that files are complete, and publish to a ComfyUI LoRA directory without overwriting existing files.

## Screenshots

### Overview

Recent tasks, the training queue and engine environments. The GPU card shows live device monitoring once it is enabled.

![Overview: recent tasks, GPU card and engine environment](assets/sample0.png)

### Dataset browser

Registered folders, scan results and paginated thumbnails; selecting an image lets you read or edit its caption. Scanning only builds an application index and never copies the dataset into the application directory.

![Dataset page: dataset list, thumbnails and caption editor](assets/sample1.png)

### New training

Pick a base-model type, an engine instance and a dataset, then search the parameter list and jump straight to a field. The right column shows the engine's native config; staged values are remembered per engine, base-model type and checkpoint. “Initialize parameters” restores that whole parameter set to the engine defaults while the task name, paths, dataset and base-model choice stay as they are.

![New training: base model, dataset, parameter search and native config preview](assets/sample2.png)

### Training task monitoring

Steps, loss, speed and elapsed time with the raw logs beside them. The loss chart carries a moving-average line. Pause (yellow) asks the engine to stop normally and keep a resumable state; Continue (blue) creates a new run from a saved point; force-stop still follows task state and process supervision.

![Training task page: loss chart, live logs and task status](assets/sample3.png)

### Engine management

Each directory under `engine/` is one installation instance; the folder name is only a label. The right-hand panel shows the interpreter, environment strategy, Triton availability and the actions for that instance.

![Engine management: instance list, Triton status and per-instance actions](assets/sample4.png)

## Typical workflow

1. In engine management, scan and confirm trusted engine source placed under `engine/`, then install and diagnose its environment.
2. In datasets, register a data folder, scan the images and confirm samples and captions.
3. In new training, pick a ready engine, base model and dataset, review parameters and the native config preview, then submit.
4. In tasks, follow progress, loss and logs; afterwards check models on the results page and publish what you need.

## Engine support and limitations

| Engine | Current support |
| --- | --- |
| Kohya / sd-scripts | Native single-GPU LoRA CLI training for SD1 / SD2 / SDXL, with config generation, log parsing and artifact registration. |
| Musubi Tuner | Cache pipeline (latents → text encoder → training) plus single-GPU LoRA training for 13 architectures. |
| KohakuMossTrainer / Anima LoRA | MOSS-Image and Anima LoRA training, including previews, checkpoints and loss reporting. |
| AI Toolkit | Static discovery, instance/version management, environment diagnostics and its adapter install flow; **training submission is not implemented yet**. |

Kohya and Musubi Tuner training chains are covered by real-subprocess automated tests using simulated engines, but have **not** been validated end to end with real GPUs, models and upstream engines. Arbitrary upstream versions are not guaranteed to work and an interrupted run is not guaranteed to leave a complete checkpoint. With “保存训练状态” enabled, Anima, Kohya, Musubi and KohakuMoss can pause and continue from a training-state save point (Continue on the task page); without a saved state there is nothing to resume, and a force-stop can still leave incomplete files.

The application never downloads, updates or deletes engine repositories for you. After cloning or copying source, click "install engine environment" on the engine page without binding Python first: the app prepares its project-local uv and runs that engine's own installation flow, streaming and persisting logs. When installation finishes it discovers and saves the engine interpreter, runs diagnostics and writes the state back; a failed diagnosis is never reported as success.

## Install and update

```bat
install.cmd   :: bootstrap uv -> fetch CPython -> create .venv -> install deps -> run build.cmd
build.cmd     :: build frontend -> build python_runtime -> compile EXE -> self-check
update.cmd    :: fast-forward from git; user data, engines and runtimes are never touched
start.cmd     :: start the desktop window from python_runtime (or double-click LP-Lora-Trainer.exe)
```

**Prerequisites** (build machine only): Windows x64, Node.js 20+, .NET Framework 4.x (`csc.exe`), Git, and network access to GitHub and PyPI. **No system Python is required** — `install.cmd` fetches a CPython through the project-local uv.

The repository contains source only. `python_runtime/`, `frontend/dist/`, `runtime/` and `engine/*` are git-ignored and do not come with a clone, so build once with `install.cmd` or `build.cmd`.

`update.cmd` only moves tracked source files: `data/`, `output/`, `engine/*`, `runtime/`, `python_runtime/` and `frontend/dist/` stay exactly as they are. It refuses to run when tracked source has local edits, and `--check` reports what would change without touching anything. Add `--rebuild` to rebuild the frontend (and the launcher) when those sources changed.

## Portable use (Windows x64, no system Python)

The app ships a complete `python_runtime/`; it is not a `.venv` that depends on a system Python. Copy the **complete distribution directory** to another Windows x64 computer and double-click the windowed `LP-Lora-Trainer.exe`; no Python, pip, Node.js or environment activation is needed.

```text
LP-Lora-Trainer/
├─ LP-Lora-Trainer.exe         # windowed desktop entry, locates its interpreter by its own path
├─ start.cmd / start-api.cmd   # command-line entry points
├─ update.cmd / check-runtime.cmd
├─ python_runtime/             # full CPython, standard library, app dependencies and licenses
│  ├─ python.exe
│  ├─ python312._pth           # relative paths only; no site, env vars or user site
│  └─ runtime-manifest.json    # recorded Python and dependency versions
├─ app/, desktop/, supervisor/, adapters/
├─ frontend/dist/              # the built UI must travel with the app; no Node service runs
├─ scripts/
└─ engine/                     # engine source and each engine's own environment
```

The entry points **never fall back to a system Python**: backend and supervisor start child processes with the same bundled interpreter, so moving the folder to another drive needs no Python repairs. The bundled version is CPython 3.12 with the application dependency closure (see `python_runtime/runtime-manifest.json`); upgrading means rebuilding.

**If the desktop window does not open:** when the .NET/WebView bridge is missing, the launcher switches to **terminal + browser** mode and opens `http://127.0.0.1:5900`, logging to `runtime/console.log`. Closing that terminal shuts the backend and supervisor down. `--browser` forces this mode.

```powershell
.\LP-Lora-Trainer.exe
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
.\python_runtime\python.exe scripts\check_runtime.py
```

The default API/UI address is `http://127.0.0.1:5900`. Closing the window does not end the API, the supervisor or an active training run; stop tasks from the task page, and use "exit backend and supervisor" on the settings page when no task is active or unverified.

### Moving the project directory or changing computers

Stop training and exit the background processes, then copy the whole directory. The portable runtime needs no activation and no path repair.

**Engine environments are relocatable.** Every engine except AI Toolkit installs into a copied CPython at `engine/<name>/.lp/python` (`embedded` strategy): no `pyvenv.cfg`, no venv `activate` script or `Scripts\python.exe`, and a relative `pythonXY._pth` that finds `Lib` / `Lib/site-packages` / `DLLs` after the project is copied. Pip-generated console scripts (for example `kogine.exe`) still live under `Scripts` and embed an absolute interpreter path, so they can go stale after a move; LP does not depend on them for launching: KohakuMossTrainer falls back to starting kogine through the bound interpreter when its launcher no longer matches, and the other engines run their entry scripts with the interpreter directly. Editable installs are rewritten into relative import roots. Engine records are adopted by folder name after a move and environment ids are computed without absolute paths, so manifests and bindings keep matching. An older `venv` / `.venv` engine can be reinstalled from the engine page, or converted with `scripts/migrate_env_to_embedded.py` / `scripts/transfer_env_to_embedded.py`.

Registered engines, interpreters, datasets, models, outputs and publish destinations in the database may still hold absolute paths and have to be re-confirmed in the UI; do not bulk-replace the pinned bindings or native configs of historical tasks.

### Adding an engine

```text
engine/
├─ kohya_ss/          # contains sd-scripts/, or place sd-scripts source directly
├─ musubi-tuner/      # cache pipeline + training (13 architectures)
├─ KohakuMossTrainer/ # MOSS-Image / Anima LoRA
└─ ai-toolkit/        # environment install/management; training adaptation not implemented
```

1. Copy trusted engine source into `engine/<any instance name>/`.
2. Rescan on the engine page and confirm the detected type.
3. Click "install engine environment" and confirm you trust the source; there is no need to pick Python first.
4. Watch or stop the installation in the terminal. The flow discovers and saves the interpreter, then diagnoses it; only engines that implement training adaptation and pass diagnosis accept submissions.
5. Add and scan a dataset, choose a local base model, then validate and submit from the new-training page.

Kohya dataset adaptation currently requires images to sit directly in the registered directory. The output directory may not live inside engine source, a dataset, a base model or the application `data/`, and every task gets its own output subdirectory.

## Data and settings

- `data/state.sqlite`: index of engines, datasets, tasks and artifacts. Artifact rows whose file was deleted are pruned at startup and every 6 hours (index only; files are never touched).
- `data/jobs/<task-id>/`: configs, bindings, command records, logs, metrics and artifact records.
- `data/manifests/`: environment diagnosis records.
- `data/cache/`: thumbnails and other caches.
- `data/installations/<session-id>/output.log`: complete installation logs.
- `runtime/desktop.log`, `runtime/backend.log`: desktop errors and backend startup logs.
- `runtime/uv/`: project-local uv, base Python and download cache.
- GPU monitoring is off by default; when enabled it uses low-frequency `nvidia-smi` queries.

`data_root` cannot be switched while the service is running. To migrate, stop training and the background services, copy the data, then point at the new directory with `--data-root`.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,test]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe scripts\build_python_runtime.py --replace
powershell -ExecutionPolicy Bypass -File scripts\build_launcher.ps1
```

`build_python_runtime.py` copies the real CPython interpreter, standard library, native libraries and the application dependency closure, keeping licenses and the version manifest while excluding the virtual-environment interpreter, editable mappings, pip and training dependencies. With `--replace` it self-checks the new runtime first and keeps the old one as a backup. Maintenance details are in `scripts/README.md`.

## Validation

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
.\python_runtime\python.exe scripts\check_runtime.py
npm.cmd --prefix frontend run build
```

Tests cover real HTTP, installation subprocesses, logs, failure retries, process-tree stopping, environment path write-back and diagnostics; GPU and dependency diagnosis use fixtures and never download large training packages. The portable runtime carries no build tools, so three builder-only tests are skipped there.

## Acknowledgements

This project is a workbench around other people's training code. It does not vendor or reimplement their trainers; it discovers them, manages their environments and drives their own entry points.

- **[kohya-ss / sd-scripts](https://github.com/kohya-ss/sd-scripts)** — the LoRA training foundation most engines here follow; dataset config, keys and CLI conventions come from it.
- **[kohya-ss / kohya_ss](https://github.com/bmaltais/kohya_ss)** — the GUI distribution whose official installer this app drives.
- **[kohya-ss / musubi-tuner](https://github.com/kohya-ss/musubi-tuner)** — its cache-then-train pipeline and 13-architecture support.
- **[ostris / ai-toolkit](https://github.com/ostris/ai-toolkit)** — engine discovery and its own manager-based installation flow.
- **[KohakuMossTrainer](https://github.com/KohakuBlueleaf)** and **Anima LoRA** — MOSS-Image / Anima training lines, including the ComfyUI-compatible kohya key layout.
- **[lycoris](https://github.com/KohakuBlueleaf/LyCORIS)** — the LoRA algorithm family behind the exported `ss_network_module` metadata.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — the target runtime for the exported LoRAs.
- **[FastAPI](https://fastapi.tiangolo.com/)**, **[pywebview](https://pywebview.flowrl.com/)**, **[Vue 3](https://vuejs.org/)** and **[uv](https://github.com/astral-sh/uv)** — the application stack.

Thanks as well to everyone maintaining the upstream trainers and to the users who tested early builds against real GPUs, models and datasets.

## License

[Apache License 2.0](LICENSE)
