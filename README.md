<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · A Windows desktop LoRA training workbench

A standalone desktop app for managing LoRA training. Outputs are made for ComfyUI, but ComfyUI is never required, installed, or launched.

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue)
![python](https://img.shields.io/badge/Python-3.11%2B-blue)
![vue](https://img.shields.io/badge/Vue-3-42b883)
![version](https://img.shields.io/badge/version-0.1.0-red)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## What you can do

- Scan and diagnose training engines placed under `engine/`, each with its own environment.
- Register and scan datasets, browse thumbnails, and edit captions with automatic backups.
- Validate parameters, preview the native config, submit jobs, and follow loss, logs, and artifacts.
- Cancel, interrupt, or force-stop jobs, and publish finished LoRAs to a ComfyUI folder.

## Engines

| Engine | Support |
| --- | --- |
| Kohya / sd-scripts | LoRA training for SD1, SD2 and SDXL on a single GPU. |
| Musubi Tuner | Cache pipeline plus LoRA training for 13 architectures on a single GPU. |
| AI Toolkit | Discovery, install and diagnostics only; training submission is still being adapted. |

Kohya and Musubi have automated subprocess tests with simulated engines, but no real-GPU end-to-end validation yet. Arbitrary upstream versions are not guaranteed to work. The app never downloads, updates, or deletes engine repositories.

## Screenshots

![Overview](assets/sample0.png)

![Dataset browser](assets/sample2.png)

![Training task](assets/sample1.png)

## Install and run

Requires Windows x64, Node.js 20+, .NET Framework 4.x, WebView2 Runtime and network access to GitHub and PyPI. A system Python is not required.

```bat
install.cmd
start.cmd
```

Build without starting:

```bat
install.cmd --no-build
build.cmd
```

The build creates `frontend/dist/`, `python_runtime/` and `LP-Lora-Trainer.exe`.

## Portable use

To use the app on another Windows x64 computer, copy the whole folder — including `LP-Lora-Trainer.exe`, `python_runtime/` and `frontend/dist/`. No Python, pip, Node.js or activated environment is needed to run it. The desktop window still needs WebView2 Runtime and .NET Framework.

Closing the window does not always stop the API or an active training. Use the task page to interrupt or force-stop training, and verify a `connection_lost` task manually before clearing it.

Engine environments can contain absolute paths, so reinstall or rebind them after moving the folder, and reselect any engine, dataset, model or output paths the app reports as missing.

## License

[Apache License 2.0](LICENSE)
