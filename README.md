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
| AI Toolkit | 已实现静态识别、实例/版本管理及环境诊断；**尚未实现真实训练适配，不能提交训练** |

Kohya 功能已经过模拟引擎的真实子进程自动化测试，**尚未使用真实 GPU、模型和上游引擎完成端到端训练验收**。不保证任意上游版本兼容，不保证中断保存 checkpoint，也不提供自动恢复训练状态。

应用不自动下载、更新、执行安装脚本或删除用户复制的引擎。只有源码不代表依赖齐全，应绑定引擎自己的 Python 环境并诊断；应用环境与训练环境分开维护。

## 安装与启动（Windows / PowerShell）

需要 Python 3.11+，前端构建需要 Node.js / npm；Windows 桌面窗口需要 WebView2 Runtime。训练所需 CUDA / PyTorch 等由引擎环境提供。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,test]"
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m desktop.launcher
```

构建后运行不需要常驻 Node 服务。桌面入口会启动或连接本机后端，并验证数据目录身份。

仅启动 API：

```powershell
.\.venv\Scripts\python.exe -m app.api.server
```

默认地址 `http://127.0.0.1:8765`。关闭桌面窗口不会停止后台 API 或 supervisor，也不会结束活动训练。退出训练请使用任务页的中断/强制结束；失联任务必须先在系统中核实原进程及其子进程已退出，再点击“确认已退出”。

### 接入引擎

```text
engine/
├─ kohya_ss/          # 内含 sd-scripts/，或直接放 sd-scripts 源码
├─ kohya_ss-old/      # 另一个版本，同样使用 adapters/kohya/
└─ ai-toolkit/        # 当前仅管理与诊断
```

1. 将可信引擎源码复制到 `engine/<任意实例名>/`。
2. 引擎页点击重新扫描，确认识别类型。
3. 绑定该引擎可用的 Python 解释器，并执行环境诊断。
4. 添加并扫描数据集，选择本地基础模型，在新建训练页校验后提交。

当前 Kohya 数据集适配要求图片直接位于所登记目录的一级；含子目录图片会明确拒绝提交，避免静默遗漏样本。输出目录不能位于引擎源码、数据集、基础模型或应用 `data/` 内部，每个任务使用独立输出子目录。

### 数据与设置

- `data/state.sqlite`：引擎、数据集、任务及产物索引。
- `data/jobs/<task-id>/`：配置、绑定、命令记录、日志、指标和产物记录。
- `data/manifests/`：环境诊断记录。
- `data/cache/`：缩略图等缓存。
- `runtime/`：桌面锁与后端启动日志。
- GPU 监控默认关闭，启用后使用低频 `nvidia-smi` 查询；不可用时显示未知。

运行中不能直接切换 `data_root`。迁移时先停下训练及后台服务，复制数据后指定新目录：

```powershell
.\.venv\Scripts\python.exe -m desktop.launcher --data-root "D:\LPData"
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm.cmd --prefix frontend run build
```

Windows 进程树停止测试需要系统允许执行 `taskkill`。当前后端 25 项测试通过，前端类型检查及生产构建通过；这些不是 GPU 训练或性能对照验证。

## 文档

- [项目计划](docs/项目计划.md)：目标、当前交付、后续阶段及功能范围。
- [项目架构](docs/项目架构.md)：模块边界、接口、进程生命周期与数据流。
- [前端说明](frontend/README.md)：开发命令与 API 对接。
