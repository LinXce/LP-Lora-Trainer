<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · Windows 桌面端的 LoRA 训练工作台

独立桌面 LoRA 训练管理应用，训练产物面向 ComfyUI 使用，但不依赖、不安装、也不启动 ComfyUI。

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue)
![python](https://img.shields.io/badge/Python-3.11%2B-blue)
![vue](https://img.shields.io/badge/Vue-3-42b883)
![version](https://img.shields.io/badge/version-0.1.0-red)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## 能做什么

- 扫描并诊断放在 `engine/` 下的训练引擎，每个引擎使用独立环境。
- 登记与扫描数据集，浏览缩略图，编辑 caption（覆盖前自动备份）。
- 校验参数、预览原生配置、提交任务，并跟踪 Loss、日志与产物。
- 取消排队、中断或强制结束任务，并把训练好的 LoRA 发布到 ComfyUI 目录。

## 引擎支持

| 引擎 | 支持情况 |
| --- | --- |
| Kohya / sd-scripts | 单 GPU 的 SD1 / SD2 / SDXL LoRA 训练。 |
| Musubi Tuner | 缓存流水线 + 13 个架构的单 GPU LoRA 训练。 |
| AI Toolkit | 仅识别、安装与诊断；训练提交仍在适配中。 |

Kohya 与 Musubi 已有模拟引擎的真实子进程自动化测试，但**尚未**用真实 GPU、模型与上游引擎做端到端验收；不保证任意上游版本兼容。应用不会下载、更新或删除引擎仓库。

## 界面预览

![概览](assets/sample0.png)

![数据集浏览](assets/sample2.png)

![训练任务](assets/sample1.png)

## 安装与运行

需要 Windows x64、Node.js 20+、.NET Framework 4.x、WebView2 Runtime，以及可访问 GitHub 与 PyPI 的网络。**不需要预装 Python。**

```bat
install.cmd
start.cmd
```

只构建不启动：

```bat
install.cmd --no-build
build.cmd
```

构建产物为 `frontend/dist/`、`python_runtime/` 与 `LP-Lora-Trainer.exe`。

## 便携使用

在另一台 Windows x64 电脑上使用：复制**整个文件夹**（含 `LP-Lora-Trainer.exe`、`python_runtime/`、`frontend/dist/`）即可，运行不需要 Python、pip、Node.js 或激活环境；桌面窗口仍需要 WebView2 Runtime 与 .NET Framework。

关闭窗口不一定结束后端或正在进行的训练：请在任务页中断或强制结束训练；对 `connection_lost` 的任务要先人工核实再清除。

引擎环境内部可能保存绝对路径，迁移后请重新安装或重新绑定并诊断；界面提示缺失的引擎、数据集、模型或输出路径也需要重新选择。

## 许可证

[Apache License 2.0](LICENSE)
