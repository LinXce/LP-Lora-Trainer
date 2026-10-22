# LP LoRA Trainer — core 发行包

> 版本 {{VERSION}} · 生成于 {{DATE}}

本目录是 **core 发行包**：包含桌面程序、内置 Python 运行时、已构建前端与应用源码，**不包含训练引擎源码或引擎环境**。

## 目录

| 路径 | 内容 |
| --- | --- |
| `LP-Lora-Trainer.exe` | 无控制台桌面入口 |
| `start.cmd` / `start-api.cmd` | 命令行兼容入口 / 仅启动本机 API |
| `python_runtime/` | 内置 CPython 与应用依赖，不需要系统 Python |
| `frontend/dist/` | 已构建前端 |
| `app/`、`desktop/`、`supervisor/`、`adapters/`、`scripts/`、`assets/` | 应用源码、适配器与资源 |
| `data/` | 首次启动自动创建，本包未包含 |
| `engine/` | 本包未包含，需要自行添加引擎源码 |

## 系统要求

- Windows 10/11 x64，WebView2 Runtime、可用的 .NET Framework。
- NVIDIA CUDA 显卡；AMD、Intel、Apple GPU 不支持。不同引擎的 CUDA 版本决定支持范围：
  - anima_lora / musubi-tuner / KohakuMossTrainer：CUDA 13.x，Turing（RTX 20 / GTX 16）及更新，NVIDIA 驱动 r580 或更新。
  - sd-scripts：CUDA 12.8，Maxwell / Pascal / Volta 及更新，NVIDIA 驱动 r570 或更新。
- 不需要安装 Python、pip 或 Node.js；引擎环境需要联网安装。

## 首次使用

1. 双击 `LP-Lora-Trainer.exe`，或运行 `start.cmd`。
2. 进入引擎页：添加引擎源码（应用内克隆入口，或把源码放到 `engine/<名称>/`），然后点「安装引擎环境」。安装需要网络，并按目标显卡选择 CUDA 来源。
3. 安装完成后应用会自动保存解释器并执行诊断；诊断通过后再添加数据集与基础模型，新建训练任务。
4. 每个引擎环境位于该引擎目录下的 `.lp/python`，随程序目录一起移动；升级程序时保留 `engine/` 即可。

## 数据与隐私

- 本包不包含 `data/`、`output/`、数据集、基础模型、输出模型、任务历史、引擎记录或日志；首次启动会创建新的 `data/`。
- 建议用 `LP-Lora-Trainer.exe --data-root "D:\LPData"` 将数据放到程序目录之外，这样替换程序文件不会影响训练数据。
- 不要把他人的 `data/` 或带个人路径的任务记录混入本包。

## 更新

- 本包**不包含 `.git`**，`update.cmd` / git 快进更新不适用于发行包。
- 更新方式：停止 LP 后，用新版 core 发行包替换程序文件；保留 `data/`、`output/`、`engine/`（以及外置的数据目录）。
- 应用与引擎的版本兼容以发布说明为准；替换前建议备份 `data/`。

## 许可与再分发

- LP 应用本身为 Apache-2.0，详见 `LICENSE`。
- 各训练引擎、PyTorch、CUDA/cuDNN 等组件遵循其各自许可证；本包不包含模型权重或训练数据。再分发前请确认目标组件的许可证要求。

## 常见问题

- **SmartScreen 提示**：EXE 未签名，选择「仍要运行」。
- **端口 5900 被占用**：确认没有其他 LP 实例在运行。
- **没有 NVIDIA 显卡**：界面可打开，但训练引擎无法运行。
- **路径含空格或中文**：项目本身支持；若第三方引擎报错，可先尝试较短的英文路径。
- **日志**：`runtime/` 下保存后端日志；反馈问题时请附上。
