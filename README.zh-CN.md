<div align="center">

<img src="assets/home.png" alt="LP LoRA Trainer" width="240" />

### LP LoRA Trainer · Windows 桌面 LoRA 训练工作站

一个独立运行的 LoRA 训练桌面应用。产物主要面向 ComfyUI，但不依赖、不安装、也不启动 ComfyUI。

![platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue?style=flat)
![python](https://img.shields.io/badge/python-3.12%20bundled-blue?style=flat)
![vue](https://img.shields.io/badge/vue-3-42b883?style=flat)
![version](https://img.shields.io/badge/version-2.0.2-red?style=flat)
![license](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

## 功能

- **多引擎训练**：把引擎源码放进 `engine/`，扫描识别后安装环境即可训练；不同版本可并存，任务绑定提交时所用的安装实例。
- **数据集管理**：登记数据目录并扫描图片，按问题类型筛选样本，逐张查看与编辑 caption（覆盖前自动备份）。
- **参数与配置**：按所选引擎与底模给出可用参数，可按中文名或引擎键名搜索并跳转，提交前预览最终原生配置与启动命令。
- **训练监控**：实时查看步数、Loss、速度与日志。队列一次执行一项训练。
- **任务控制**：取消排队、中断或强制结束；后台失联的任务会显式标记，需人工核实后放行。
- **训练结果**：按任务查看 checkpoint 和采样图，检查文件完整性，在输出目录中定位所选权重。
- **版本化引擎基线**：每个 LP 版本固定记录各引擎适配时的 git 分支/commit；引擎页显示基线并在代码已移动时提示。拉取新引擎并改完适配后，用 `scripts/sync_engine_versions.py` 记录新基线并升级项目版本。

## 版本与引擎基线

`app/engine_compat.json` 保存项目版本到各引擎 git 基线（`ref` + 完整 `commit`）及适配器版本的对应关系；一个项目版本可以包含多个引擎条目。引擎页会显示当前安装实例对应的适配基线，commit 或适配器版本不一致时显示警告。

拉取新引擎代码并完成适配后：

```powershell
python scripts/sync_engine_versions.py --bump patch
```

只核对当前检出的引擎是否仍匹配当前项目基线：

```powershell
python scripts/sync_engine_versions.py --check
```

## 界面预览

### 概览

最近任务、训练队列和引擎环境状态；启用监控后显示 GPU 卡片。

![概览页：最近任务、GPU 卡片和引擎环境](assets/sample0.png)

### 数据集浏览

已登记目录、扫描结果和分页缩略图；选择图片后可查看或编辑 caption。扫描只建立应用索引，不会把数据集复制进应用目录。

![数据集页：数据集列表、图片缩略图和 Caption 编辑面板](assets/sample1.png)

### 新建训练

选择底模类型、引擎实例与数据集，可搜索参数并直接跳到对应字段；右栏实时显示该引擎的原生配置，参数暂存按「引擎 × 底模类型 × 底模」各存一份。右上角「初始化参数」会把当前引擎与底模的参数整体恢复为引擎默认值，任务名称、路径、数据集与底模选择保持不变。

![新建训练页：底模、数据集、参数搜索与原生配置预览](assets/sample2.png)

### 训练任务监控

任务页根据引擎训练日志显示 Step、Epoch、Loss 和最近一次训练速度。速度较慢时显示每步秒数（s/it），较快时显示每秒步数（it/s）。引擎未报告的进度显示「未知」。旁边可查看耗时、原始日志和带滑动平均线的 Loss 曲线。暂停（黄色）请求引擎正常停止并尽量保存可续训状态，继续（蓝色）从保存点创建新的续训任务。强制结束仍受任务状态和进程监管约束。

![训练任务页：Loss 曲线、实时日志和任务状态](assets/sample3.png)

### 引擎管理

`engine/` 下每个一级目录是一个安装实例，目录名只是标签；右侧面板显示解释器、环境策略、Triton 可用性与对应操作。

![引擎管理页：实例列表、Triton 状态与单实例操作](assets/sample4.png)

## 典型流程

1. 在引擎管理页扫描并确认 `engine/` 下的可信引擎源码，安装并诊断独立环境。
2. 在数据集页登记数据目录并扫描图片，确认样本和 caption。
3. 在新建训练页选择就绪引擎、底模与数据集，检查参数与原生配置预览后提交。
4. 在训练任务页跟踪进度与日志，到结果页查看 checkpoint、对比采样图并定位权重文件。

## 引擎支持与限制

| 引擎 | 当前支持 |
| --- | --- |
| Kohya / sd-scripts | 支持 SD1、SD2、SDXL、SD3、FLUX.1、HunyuanImage、Lumina、Anima 和 MOSS-Image。0.13 训练器支持多行 caption 和原生保存后停止，Anima、MOSS-Image 还可配置单进程多卡副本。 |
| Musubi Tuner | 缓存流水线（latents → 文本编码器 → 训练）与 13 种架构的单卡 LoRA 训练。 |
| KohakuMossTrainer | MOSS-Image 原生缓存及训练流程，支持教师蒸馏、混合尺寸采样和第 0 步预览。 |
| Anima LoRA | Anima 预处理与训练，以及 MOSS-Image Base/SFT 缓存和训练，支持教师蒸馏和按步、按轮保存权重。 |
| AI Toolkit | 静态发现、实例/版本管理、环境诊断与其适配安装流程；**训练提交尚未实现**。 |

Kohya 与 Musubi 的训练链路由真实子进程 + 模拟引擎的自动化测试覆盖，但**尚未**用真实 GPU、模型和上游引擎做端到端验证。任意上游版本不保证可用，中断后不保证 checkpoint 完整。训练参数开启「保存训练状态」后，Anima、Kohya、Musubi 与 KohakuMoss 可暂停并从状态保存点继续（任务页「继续」）；没有保存状态时无法恢复，强制结束仍可能留下不完整文件。

克隆或复制引擎源码后，在引擎页点「安装引擎环境」，核对该实例的计算后端和可选组件，按需选择下载镜像，再开始安装。四个训练引擎共享下载缓存，各自保留独立的 Python 和依赖。终端会显示安装进度，失败后可点「调整配置并重试」带回上次选项。引擎通过诊断后，界面才显示安装成功。

## 安装与更新

```bat
install.cmd   :: 自举 uv → 取 CPython → 建 .venv → 安装依赖 → 调用 build.cmd
build.cmd     :: 构建前端 → 构建 python_runtime → 编译 EXE → 自检
update.cmd    :: 从 git 快进更新；用户数据、引擎与运行时一律不动
start.cmd     :: 用 python_runtime 启动桌面窗口（也可直接双击 LP-Lora-Trainer.exe）
```

**前置条件**（仅构建机需要）：Windows x64、Node.js 20+、.NET Framework 4.x（`csc.exe`）、Git，以及可访问 GitHub 与 PyPI 的网络。**不需要预装 Python** —— `install.cmd` 会用项目内的 uv 取一份 CPython。

仓库只含源码：`python_runtime/`、`frontend/dist/`、`runtime/`、`engine/*` 都在 `.gitignore` 中，clone 之后先用 `install.cmd` 或 `build.cmd` 构建一次。

`update.cmd` 只更新被 git 跟踪的源码：`data/`、`output/`、`engine/*`、`runtime/`、`python_runtime/`、`frontend/dist/` 保持原样。若已跟踪源码存在本地修改，脚本会**拒绝执行**；`--check` 只报告有哪些更新、不改动任何文件；`--rebuild` 会在更新后按需重建前端与启动器。

## 打包发行包（维护者）

`scripts\pack_delivery.cmd` 默认在 `_delivery` 下生成两个**无 `.git` 的发行目录**：

```bat
scripts\pack_delivery.cmd
scripts\pack_delivery.cmd D:\LP-Delivery
scripts\pack_delivery.cmd D:\LP-Delivery -Engines anima_lora,musubi-tuner
scripts\pack_delivery.cmd D:\LP-Delivery -Engines none
scripts\pack_delivery.cmd -ListEngines
```

- `LP-Lora-Trainer-core`：程序、`python_runtime/`、`frontend/dist/`、应用源码；不含引擎。
- `LP-Lora-Trainer-full`：core + 选中的 `engine/<名称>/` 及其 `.lp` 环境。
- `-Engines` 按目录名选择，默认全部；`none` 只生成 core。每个包都会生成中文 `README-FIRST.md`，写明随包引擎、GPU/驱动要求、首次启动与更新策略。
- release 包排除 `.git`、`data/`、`output/`、日志、缓存、`node_modules/` 和 `docs/`；发行包不能用 `update.cmd` 更新，更新方式是用新版发行包替换程序文件并保留 `data/`、`output/`、`engine/`。git 快进更新只适用于完整源码目录。

## 便携运行（Windows x64，无需系统 Python）

应用自带完整 `python_runtime/`，不是依赖系统 Python 的 `.venv`。把**整个交付目录**复制到另一台 Windows x64 机器，双击 `LP-Lora-Trainer.exe` 即可，无需 Python、pip、Node.js 或任何激活步骤。

```text
LP-Lora-Trainer/
├─ LP-Lora-Trainer.exe         # 桌面入口，按自身位置定位解释器
├─ start.cmd / start-api.cmd   # 命令行入口（API-only 亦可）
├─ update.cmd / check-runtime.cmd
├─ python_runtime/             # 完整 CPython、标准库、应用依赖与许可证
│  ├─ python.exe
│  ├─ python312._pth           # 只有相对路径；无 site、环境变量与用户目录
│  └─ runtime-manifest.json    # 记录的 Python 与依赖版本
├─ app/, desktop/, supervisor/, adapters/
├─ frontend/dist/              # 构建好的 UI 必须随应用一起拷走
├─ scripts/
└─ engine/                     # 引擎源码与各引擎自己的环境
```

所有入口**永不回退到系统 Python**：后端与 supervisor 用同一个内置解释器启动子进程，所以整个目录换盘后不需要修路径。内置版本为 CPython 3.12 加应用依赖闭包（见 `python_runtime/runtime-manifest.json`），升级意味着重建。

**桌面窗口打不开时**：缺少 .NET/WebView 桥接会切到**终端 + 浏览器**模式并打开 `http://127.0.0.1:5900`，日志写在 `runtime/console.log`；关闭该终端即结束后端与 supervisor。`--browser` 可强制此模式。

```powershell
.\LP-Lora-Trainer.exe
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
.\python_runtime\python.exe scripts\check_runtime.py
```

默认地址为 `http://127.0.0.1:5900`。桌面窗口的 × 和 Alt+F4 使用同一退出流程。若窗口启动了后端，有训练时可以取消，或确认「终止训练并退出」。确认后暂停队列，等待引擎正常停止、登记结果，再退出。是否生成新的可续训保存点取决于引擎、配置及当前阶段。排队任务保留到下次启动。若窗口复用已有后端，「仅关闭本窗口」会让训练和队列继续运行。

### 换目录 / 换电脑

先停止训练并退出后台进程，再整体复制目录。便携运行时不需要激活，也不需要修路径。

**引擎环境同样可搬迁**：除 AI Toolkit 外，引擎都把依赖装进 `engine/<名称>/.lp/python` 的复制版 CPython（`embedded` 策略）——没有 `pyvenv.cfg`、没有 venv 的 `activate` / `Scripts\python.exe`，只有相对路径的 `pythonXY._pth` 定位 `Lib` / `Lib/site-packages` / `DLLs`。pip 生成的 console-script（例如 `kogine.exe`）仍位于 `Scripts` 且内嵌绝对解释器路径，整体复制后可能失效；LP 启动不依赖它：KohakuMossTrainer 在启动器指向的解释器不再匹配时，改用当前绑定解释器调用 kogine 入口，其他引擎则直接由解释器运行入口脚本。editable 安装会改写成 `._pth` 里的相对导入根。移动后引擎记录按目录名认领，环境 id 不含绝对路径，清单与绑定继续匹配。旧的 `venv` / `.venv` 环境可在引擎页重装，或用 `scripts/migrate_env_to_embedded.py` / `scripts/transfer_env_to_embedded.py` 转换。

数据库里登记的引擎、解释器、数据集、模型和输出目录可能仍是绝对路径。请在界面里重新确认这些路径，并保留历史任务固定的绑定和原生配置。

### 接入引擎

```text
engine/
├─ kohya_ss/          # 内含 sd-scripts/，也可直接放 sd-scripts 源码
├─ musubi-tuner/      # 缓存流水线 + 训练（13 种架构）
├─ KohakuMossTrainer/ # MOSS-Image / Anima LoRA
└─ ai-toolkit/        # 环境安装/管理；训练适配尚未实现
```

1. 把可信引擎源码放进 `engine/<任意实例名>/`。
2. 引擎页重新扫描并确认识别出的类型。
3. 点「安装引擎环境」并确认信任来源；不需要先选 Python。
4. 在终端观察或停止安装。流程会自动发现并保存解释器，再做诊断；只有实现训练适配且诊断通过的引擎才能提交任务。
5. 添加并扫描数据集、选择本地底模，然后在新建训练页校验并提交。

Kohya 数据集目前要求图片直接位于登记目录。新建训练时，把输出根目录选在引擎源码、数据集、底模和应用 `data/` 之外。

每次提交都使用 `<输出根目录>/<安全任务名>--<完整 task_id>/`。管理器会处理任务名里不适合 Windows 目录的字符，并在提交时分配 ID。因此，即使两次训练使用相同任务名和输出模型名称，权重也分别写入各自的目录。目录中的权重文件仍使用配置的模型名称作为前缀。复用历史配置会带回原来的输出根目录。「继续」从历史保存点读取状态，将新权重和保存点写入续训任务自己的目录。旧任务可在「训练任务 → 任务信息」查看其原有输出路径。

## 数据与设置

- `data/state.sqlite`：引擎、数据集、任务与产物的索引；后端启动时及每 6 小时清理已删除文件的产物记录（只删索引，不删文件）。
- `data/jobs/<task-id>/`：配置、绑定、命令记录、日志、指标与产物记录。
- `data/manifests/`：环境诊断记录。
- `data/cache/`：缩略图等缓存。
- `data/installations/<session-id>/output.log`：完整安装日志。
- `runtime/desktop.log`、`runtime/backend.log`：桌面错误与后端启动日志。
- `runtime/uv/`：项目内 uv、基础 Python 与下载缓存。
- GPU 监控默认关闭；开启后用低频 `nvidia-smi` 查询。

服务运行时不能切换 `data_root`：停止训练与后台服务、复制数据后，用 `--data-root` 指向新目录。

## 开发

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,test]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe scripts\build_python_runtime.py --replace
powershell -ExecutionPolicy Bypass -File scripts\build_launcher.ps1
```

`build_python_runtime.py` 复制真实 CPython、标准库、原生库与应用依赖闭包，保留许可证与版本清单，排除虚拟环境解释器、editable 映射、pip 与训练依赖；加 `--replace` 会先自检新运行时并把旧的留作备份。维护细节见 `scripts/README.md`。

## 验证

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
.\python_runtime\python.exe scripts\check_runtime.py
npm.cmd --prefix frontend run build
```

测试覆盖真实 HTTP、安装子进程、日志、失败重试、进程树停止、环境路径回写与诊断；GPU 与依赖诊断使用固定数据，不下载大型训练包。便携运行时不带构建工具，因此有三个构建专用测试在便携环境中跳过。

## 致谢

本项目是围绕他人训练代码的工作台：不内嵌、也不重写这些训练器，只负责发现、管理环境并调用它们自己的入口。

- **[kohya-ss / sd-scripts](https://github.com/kohya-ss/sd-scripts)** —— 本应用多个引擎沿用的 LoRA 训练基础；数据集配置、键名与 CLI 约定都源自它。
- **[kohya-ss / kohya_ss](https://github.com/bmaltais/kohya_ss)** —— 应用调用其官方安装器完成环境部署。
- **[kohya-ss / musubi-tuner](https://github.com/kohya-ss/musubi-tuner)** —— 缓存后再训练的流水线与 13 种架构支持。
- **[ostris / ai-toolkit](https://github.com/ostris/ai-toolkit)** —— 引擎发现与其自带 manager 的安装流程。
- **[KohakuMossTrainer](https://github.com/KohakuBlueleaf)** 与 **Anima LoRA** —— MOSS-Image / Anima 训练线，以及 ComfyUI 可读的 kohya 键名布局。
- **[lycoris](https://github.com/KohakuBlueleaf/LyCORIS)** —— 导出元数据中 `ss_network_module` 对应的 LoRA 算法家族。
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** —— 导出 LoRA 的目标运行环境。
- **[FastAPI](https://fastapi.tiangolo.com/)**、**[pywebview](https://pywebview.flowrl.com/)**、**[Vue 3](https://vuejs.org/)**、**[uv](https://github.com/astral-sh/uv)** —— 应用本身的技术栈。

也感谢维护这些上游训练器的作者，以及在真实 GPU、模型与数据集上测试早期版本的用户。

## 许可证

[Apache License 2.0](LICENSE)
