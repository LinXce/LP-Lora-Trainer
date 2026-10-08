# LP LoRA Trainer

独立桌面 LoRA 训练管理应用，训练产物主要面向 ComfyUI 使用，但不依赖、安装或启动 ComfyUI。

## 实现方案

- 窗口：pywebview，Windows 使用 WebView2；单窗口、单 WebView。
- 界面：Vue 3 + TypeScript + Vite；保留现有布局、主题和组件样式，原菱形徽标替换为 **LP Logo**。
- 后端：FastAPI + Uvicorn，本机 HTTP API + SSE。
- 训练：独立 supervisor 调用引擎原生 CLI，单 GPU 顺序队列；管理进程不加载训练模型。
- 引擎：直接复制源码到 `engine/`，同类实例共享 `adapters/<engine>/` 接口适配。
- 版本：多安装实例共存，记录源码 revision、解释器、环境诊断 manifest 和 adapter 版本；提交任务后绑定固定，不随默认实例切换。
- 存储：SQLite 索引与任务文件分离；模型和数据集使用本地路径，不上传、不默认复制整个数据集。

## 已实现功能

- 引擎扫描、类型确认、Python 解释器绑定、环境诊断、默认实例选择及版本记录。
- 数据集登记、扫描、问题筛选、分页缩略图、caption 编辑；覆盖 caption 前自动备份。
- 训练参数校验、原生配置预览、任务提交、状态监控、原始日志及 Loss 指标。
- 取消排队、中断和强制结束；监管失联或无法确认退出时进入 `connection_lost`，人工核实后才能解除队列阻塞。
- 模型及采样图登记；选定模型完整性校验、无覆盖原子发布，可交付到 ComfyUI LoRA 目录。
- 设置保存、原生路径选择、资源管理器定位、桌面重连、SSE 重连后快照及当前任务历史刷新。

### 引擎支持边界

| 引擎 | 当前能力 |
| --- | --- |
| Kohya / sd-scripts | 已实现 SD1 / SD2 / SDXL 单 GPU LoRA 原生 CLI 训练适配、配置生成、日志解析与产物登记 |
| AI Toolkit | 静态识别、版本/实例管理、环境诊断及适配器安装流程；**尚未实现真实训练适配，不能提交训练** |
| Musubi Tuner | 静态识别、版本/实例管理、环境诊断及适配器安装流程；**尚未实现真实训练适配，不能提交训练** |

Kohya 功能已经过模拟引擎的真实子进程自动化测试，**尚未使用真实 GPU、模型和上游引擎完成端到端训练验收**。不保证任意上游版本兼容，不保证中断保存 checkpoint，也不提供自动恢复训练状态。

应用不替用户下载、更新或删除引擎主仓库。clone/复制源码后，无需先绑定 Python，直接在引擎页确认“安装引擎环境”；应用准备项目内 uv，再调用该引擎的安装流程，终端实时显示并持久化日志。安装结束后自动发现并保存引擎解释器、执行诊断、回写状态；诊断未通过不会显示成功。应用 `python_runtime` 只运行标准库引导脚本，不安装训练依赖；基础 Python 由项目内 uv 管理，各引擎的 `venv` / `.venv` 独立。Kohya 官方安装器可能初始化 sd-scripts 子模块，AI Toolkit 官方 manager 可能安装其所需的 Node/ffmpeg。详见 [运行与安装](docs/运行与安装.md)。

## 便携运行（Windows x64，无需系统 Python）

应用自带完整的 `python_runtime/`，不是依赖系统 Python 的 `.venv`。复制**完整交付目录**到另一台 Windows x64 电脑后，双击无控制台窗口的 `LP-Lora-Trainer.exe` 打开桌面窗口；不需要安装 Python、pip、Node.js 或激活环境。

```text
LP-Lora-Trainer/
├─ LP-Lora-Trainer.exe         # 无控制台桌面入口，按自身位置定位解释器
├─ start.cmd                   # 兼容命令行启动入口（桌面使用 EXE）
├─ start-api.cmd               # 仅运行本机 API
├─ check-runtime.cmd           # 离线运行环境自检
├─ python_runtime/             # 完整 CPython、标准库、应用依赖与许可证
│  ├─ python.exe
│  ├─ python312.dll            # 按实际构建版本命名
│  ├─ python312._pth           # 仅相对路径；禁用系统/user site 与环境变量
│  ├─ Lib/site-packages/
│  ├─ DLLs/
│  └─ runtime-manifest.json    # Python 和应用依赖版本记录
├─ app/、desktop/、supervisor/、adapters/
├─ frontend/dist/              # 必须携带已构建界面，无 Node 常驻服务
├─ scripts/
└─ engine/                     # 引擎源码及各自独立运行环境
```

**更新提示：** 关闭窗口不会退出 API/supervisor。启动器检查后台安装协议，拒绝重连旧版或不兼容后台，不会另起同端口后端。更新后先停止安装/训练并确认进程已退出，再退出本项目旧 API 和 supervisor，最后重新打开 EXE。请勿在活动任务期间强制结束后台。

入口**不回退到系统 Python**。应用后端与 supervisor 使用同一个随包解释器启动子进程；移动盘符后也不需要修复 Python 的硬编码路径。当前本地生成版本为 CPython 3.12.7（构建机版本），22 个应用依赖；升级需要重新构建，不能把这份版本记录理解为最新 Python 版本。

**桌面窗口打不开时：** 启动器会先检查桌面渲染所需的 .NET/WebView 桥接。可用时只显示桌面窗口；不可用时（或窗口打开后 WebView2 初始化 / 后端引导仍然失败时）自动切换到**终端 + 浏览器**模式——用**系统自带终端**开一个窗口，先显示 `assets/logo.txt` 的 LP 文字字标，再写明原因，并用默认浏览器打开 `http://127.0.0.1:5900`；全部输出写入 `runtime/console.log`（不会静默失败）。**关闭该终端窗口（或按 Ctrl+C / 回车）即结束后端与监管进程。** 也可用 `--browser` 强制该模式。唯一不回退的情况：端口上已有旧版或使用其他数据目录的后端（先按提示退出它）。需要整体退出（含监管进程）时用设置页「退出后端与监督」：只有无活动任务、无失联任务时才允许。

```powershell
.\LP-Lora-Trainer.exe
.\start-api.cmd
# 不暂停的环境自检
.\python_runtime\python.exe scripts\check_runtime.py
# 自定义数据目录
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
```

Windows 桌面仍要求 WebView2 Runtime 和可用的 .NET Framework；训练还需 GPU 驱动及引擎自己的依赖。**无需系统 Python 不等于无需操作系统组件。** 本项目不自动安装这些系统组件。

默认 API/界面地址 `http://127.0.0.1:5900`。关闭窗口不会结束 API、supervisor 或活动训练。结束训练使用任务页的中断/强制结束；失联任务必须先核实原进程及子进程已经退出，再点击“确认已退出”。

### 开发与生成交付环境

只有构建/开发机需要完整 Windows x64 Python 3.11+；构建前端还需 Node.js / npm。普通使用者只复制制作好的完整目录。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,test]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
# 从本地完整 CPython 和已安装依赖离线生成运行时
.\.venv\Scripts\python.exe scripts\build_python_runtime.py
powershell -ExecutionPolicy Bypass -File scripts\build_launcher.ps1
.\LP-Lora-Trainer.exe
```

`build_python_runtime.py` 复制真正的 CPython 解释器、标准库、必要原生库和应用依赖闭包；保留许可证和版本清单，不复制虚拟环境解释器、editable 映射、pip 或训练依赖。默认拒绝覆盖既有运行时；使用 `--replace` 时先自检新版本，再将旧版本保留为备份。详细维护说明见 `scripts/README.md`。

`LP-Lora-Trainer.exe`、`python_runtime/` 和 `frontend/dist/` 是交付生成物，**仅下载源码仓库不能直接双击运行**。制作交付包时必须包含它们，不需要包含开发 `.venv` 或 `frontend/node_modules`。启动器源码在 `desktop/launcher_host.cs`，可用 `scripts/build_launcher.ps1` 重建。

### 移动项目目录或更换电脑

便携入口根据实际位置解析目录，不绑定 D/H 盘。迁移前先停止训练并退出相关后台进程，再复制完整目录；便携运行时无需激活或执行 `repair_environment.py`。

数据库内已登记的引擎、Python、数据集、模型、输出及发布目录仍可能是绝对路径，需在界面重新确认和绑定；不批量替换历史任务的固定绑定或原生配置。

**引擎便携是独立事项：** 安装时基础 Python 由项目内 uv 管理，不依赖系统 Python；但已安装的 `venv` / `.venv` 仍可能保存绝对路径，换电脑/盘符后不能承诺复制即用。应重新安装或重新绑定并诊断，不批量篡改历史任务绑定。当前不制作可任意迁移的整套训练环境包，也不将训练依赖塞进应用运行时。

开发 `.venv` 若仅变更盘符且原基础 Python 仍可用，可运行 `scripts/repair_environment.py` 修复；这不属于便携交付流程。

### 接入引擎

```text
engine/
├─ kohya_ss/          # 内含 sd-scripts/，或直接放 sd-scripts 源码
├─ kohya_ss-old/      # 另一个版本，同样使用 adapters/kohya/
├─ ai-toolkit/        # 环境安装/管理；训练适配尚未实现
└─ musubi-tuner/      # 环境安装/管理；训练适配尚未实现
```

1. 将可信引擎源码复制到 `engine/<任意实例名>/`。
2. 引擎页点击重新扫描，确认识别类型。
3. 点击“安装引擎环境”，确认信任源码；无需先选择 Python。Musubi 的 CUDA extra 只显示该版本声明的选项，Kohya / AI Toolkit 由官方安装器决定。
4. 自动进入终端查看/停止安装。流程自动发现引擎解释器、保存路径并诊断；失败可查看日志后重试，已有独立环境也可手动绑定并诊断。只有实现训练适配且诊断就绪的引擎才能提交训练。
5. 添加并扫描数据集，选择本地基础模型，在新建训练页校验后提交。

当前 Kohya 数据集适配要求图片直接位于所登记目录的一级；含子目录图片会明确拒绝提交，避免静默遗漏样本。输出目录不能位于引擎源码、数据集、基础模型或应用 `data/` 内部，每个任务使用独立输出子目录。

### 数据与设置

- `data/state.sqlite`：引擎、数据集、任务及产物索引。
- `data/jobs/<task-id>/`：配置、绑定、命令记录、日志、指标和产物记录。
- `data/manifests/`：环境诊断记录。
- `data/cache/`：缩略图等缓存。
- `data/installations/<session-id>/output.log`：完整安装日志，清空终端显示不会删除文件。
- `runtime/desktop.log`、`runtime/backend.log`：桌面错误及后端启动日志。
- `runtime/uv/`：项目内 uv、基础 Python 和下载缓存，与应用依赖分离。
- GPU 监控默认关闭，启用后使用低频 `nvidia-smi` 查询；不可用时显示未知。

运行中不能直接切换 `data_root`。迁移时先停下训练及后台服务，复制数据后指定新目录：

```powershell
.\LP-Lora-Trainer.exe --data-root "H:\LPData"
```

## 验证

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
.\python_runtime\python.exe scripts\check_runtime.py
npm.cmd --prefix frontend run build
```

Windows 进程树停止测试需要系统允许执行 `taskkill`。测试覆盖真实 HTTP、安装子进程、日志、失败重试、停止进程树、环境路径回写与诊断；GPU/依赖诊断使用 fixture，不下载大型训练包。便携运行时不携带构建工具，3 项 builder-only 测试会跳过；构建机另用已安装应用依赖与 `packaging` 的 Python 执行 `python -m unittest tests.test_portable_runtime.RuntimeBuilderTests -v`。EXE 已实测桌面启动及后端重连，前端类型检查和生产构建已通过；这些不是上游完整安装、GPU 训练或性能对照验证。

## 文档

- [项目计划](docs/项目计划.md)：目标、当前交付、后续阶段及功能范围。
- [项目架构](docs/项目架构.md)：模块边界、接口、进程生命周期与数据流。
- [运行与安装](docs/运行与安装.md)：完整启动/安装链路、环境隔离、日志、重试与验证边界。
- [前端说明](frontend/README.md)：开发命令与 API 对接。
