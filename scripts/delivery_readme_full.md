# LP LoRA Trainer — full 发行包

> 版本 {{VERSION}} · 生成于 {{DATE}}

本目录是 **full 发行包**：包含 core 发行包的全部内容，并附带打包时选中的 **{{ENGINE_COUNT}} 个训练引擎**及其 Python 环境。

## 随包引擎

| 引擎目录 | 内置环境 |
| --- | --- |
{{ENGINES_TABLE}}

环境状态说明：

- `embedded`：已带 `.lp/python` 复制版 CPython，随目录一起移动，通常无需重新安装。
- `venv`：旧式虚拟环境，可能仍含绝对路径；迁移后建议重装或转换为 embedded。
- `none`：只有引擎源码，没有环境；首次使用需在引擎页安装环境。

## 目录

| 路径 | 内容 |
| --- | --- |
| `LP-Lora-Trainer.exe` | 无控制台桌面入口 |
| `start.cmd` / `start-api.cmd` | 命令行兼容入口 / 仅启动本机 API |
| `python_runtime/` | 内置 CPython 与应用依赖，不需要系统 Python |
| `frontend/dist/` | 已构建前端 |
| `app/`、`desktop/`、`supervisor/`、`adapters/`、`scripts/`、`assets/` | 应用源码、适配器与资源 |
| `engine/` | 随包引擎源码与环境（见上表） |
| `data/` | 首次启动自动创建，本包未包含 |

## 系统要求

- Windows 10/11 x64，WebView2 Runtime、可用的 .NET Framework。
- 仅支持 NVIDIA CUDA 显卡；AMD、Intel、Apple GPU 不支持。随包引擎的 CUDA 构建决定支持范围：
  - anima_lora / musubi-tuner / KohakuMossTrainer：CUDA 13.x，Turing（RTX 20 / GTX 16）及更新，NVIDIA 驱动 r580 或更新。
  - sd-scripts：CUDA 12.8，Maxwell / Pascal / Volta 及更新，NVIDIA 驱动 r570 或更新。
- 不需要安装 Python、pip 或 Node.js。若引擎状态为 `none`，安装环境需要网络。

## 首次使用

1. 双击 `LP-Lora-Trainer.exe`，或运行 `start.cmd`。
2. 进入引擎页：执行 rescan，绑定解释器（状态为 `embedded` 的环境一般会自动发现），然后执行 diagnose。
3. 状态为 `none` 的引擎需要先安装环境：联网后点「安装引擎环境」，按目标显卡选择 CUDA 来源。
4. 诊断通过后，添加数据集与基础模型，新建训练任务。
5. 内嵌环境 `.lp/python` 随目录迁移，**不需要执行 venv 迁移/修复脚本**；不要删除 `engine/` 下的 `.lp`。

## 数据与隐私

- 本包不包含 `data/`、`output/`、数据集、基础模型、输出模型、任务历史、引擎记录或日志；首次启动会创建新的 `data/`。
- 建议用 `LP-Lora-Trainer.exe --data-root "D:\LPData"` 将数据放到程序目录之外；替换程序或更新引擎源码时保留该数据目录。
- 不要把他人的 `data/`、模型、数据集或带个人路径的任务记录混入本包。

## 更新

- 本包**不包含 `.git`**：`update.cmd` / git 快进更新不适用于发行包。
- 应用更新：停止 LP 后，用新版发行包替换程序文件；保留 `data/`、`output/`、`engine/`（含 `.lp` 环境）和外置数据目录。
- 引擎更新：发行包不带引擎仓库的 `.git`，无法在应用内做 git 更新；需要更新时重新克隆引擎源码并重装其环境，或使用包含新引擎源码的新版 full 包；旧 `.lp` 环境可在重新绑定后继续使用。
- 替换或更新前建议备份 `data/` 与重要输出。

## 许可与再分发

- LP 应用本身为 Apache-2.0，详见 `LICENSE`。
- 各训练引擎、PyTorch、CUDA/cuDNN 等组件遵循其各自许可证；本包不包含模型权重或训练数据。再分发前请确认目标组件的许可证要求。

## 常见问题

- **SmartScreen 提示**：EXE 未签名，选择「仍要运行」。
- **引擎显示 `none` 或无法绑定**：先安装引擎环境；已有环境时在引擎页 rescan / 手动绑定解释器后 diagnose。
- **CUDA 不可用**：确认是 NVIDIA 显卡且驱动达到上表要求；笔记本需使用独立显卡运行。
- **端口 5900 被占用**：确认没有其他 LP 实例在运行。
- **路径含空格或中文**：项目本身支持；若第三方引擎报错，可先尝试较短的英文路径。
- **日志**：`runtime/` 下保存后端日志；反馈问题时请附上。
