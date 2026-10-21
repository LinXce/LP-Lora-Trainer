# 便携 Python 与桌面 EXE 启动

## 用户运行（不需要系统 Python）

完整 Windows x64 交付目录必须包含 `LP-Lora-Trainer.exe`、`python_runtime/`、`frontend/dist/` 和应用源码。

- `LP-Lora-Trainer.exe`：无控制台桌面窗口；支持 `--port`、`--data-root` 参数。`start.cmd` 仅保留为兼容命令行入口。
- `start-api.cmd`：只运行本机 API。
- `check-runtime.cmd`：离线检查解释器隔离、依赖、原生扩展、.NET 桥接及子进程。

桌面入口使用无控制台的 `LP-Lora-Trainer.exe`，由它定位同目录的 `python_runtime\pythonw.exe` 并启动 `desktop.launcher`，不会额外弹出命令行窗口。命令行兼容入口仍使用 `%~dp0python_runtime\python.exe`，不尝试 PATH 上的 Python，不激活 `.venv`，不自动安装系统软件。

## 构建桌面 EXE 启动器

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_launcher.ps1
```

启动器使用系统已有的 .NET Framework 编译为无控制台窗口 EXE；运行时仍使用项目内的 `python_runtime\pythonw.exe`。

## 开发 / 制作便携运行环境

推荐用 `install.cmd` → `build.cmd`：前者自举 uv 与 CPython、建好 `.venv` 并安装依赖，后者按序构建前端、`python_runtime` 与 EXE；`build.cmd` 在 `.venv` 缺失时会自动调用 `install.cmd --no-build`。以下是等价的手动步骤，适用于已有完整 Windows x64 CPython 和应用依赖的构建机：

```powershell
.\.venv\Scripts\python.exe scripts\build_python_runtime.py
# 重建时保留旧运行时作为 python_runtime.previous-时间戳 备份
.\.venv\Scripts\python.exe scripts\build_python_runtime.py --replace
# 命令行检查，不暂停
.\python_runtime\python.exe scripts\check_runtime.py
```

- 从 `sys.base_prefix` 复制完整 CPython 的 EXE、DLL、标准库和 Python 许可证，而非 `.venv` 的重定向解释器。
- 根据 `pyproject.toml` 计算应用和 desktop 依赖的已安装闭包，校验版本范围、递归 extras 和平台 marker；缺失依赖立即失败。
- 复制依赖文件及 `.dist-info`（含上游许可证），不复制 editable、`.pth` hooks、系统安装路径、旧 EXE 入口、pip 或测试工具。
- 生成相对路径 `pythonXY._pth`；不启用 `site`，忽略环境变量、注册表路径、用户 site-packages 和启动工作目录。
- 使用污染的 Python 环境变量及仅含系统工具的 PATH 进行构建自检；成功后才发布新运行时。失败产物保留为 `.python-runtime-build-*`，便于排查，不删除用户目录。
- `runtime-manifest.json` 保存 Python、应用及依赖版本；不写入构建机绝对路径。依赖升级需要在构建机更新后重新生成，消费端不原地 pip install。

构建脚本不下载任何内容，不更改系统 Python，不复制训练引擎，不自动构建前端。生成目录不纳入 Git；从源码仓库下载并不等于拿到了完整便携发行版。

## 便携边界

当前目标是 Windows x64，不是跨操作系统运行。Windows 桌面仍需 WebView2 Runtime 和可用的 .NET Framework；构建自检不会打开 GUI，也不验证目标电脑的 WebView2 安装。GPU 驱动及训练依赖由引擎负责。

引擎安装使用项目内 uv 管理基础 Python，不依赖系统 Python。**引擎环境默认是可迁移的 embedded 布局**：依赖直接装进 `engine/<名称>/.lp/python` 里复制的 CPython（无 `pyvenv.cfg` / shim / `activate`，靠相对 `._pth` 定位），editable 安装被改写成 `._pth` 中的相对导入根，所以复制整个目录后不需要重定基。本次改动之前创建的 `venv` / `.venv` 仍含绝对路径，需重装或运行 `migrate_env_to_embedded.py`。应用运行时仍不混入大型训练依赖。

迁移前停止训练、退出 API/supervisor 后复制整个目录。数据库内显式保存的引擎、数据集、模型等绝对路径需在界面重新确认，不能盲目替换历史任务绑定。

## 开发 `.venv` 的盘符迁移修复

仅当开发环境的基础 Python 仍在时执行：

```powershell
.\.venv\Scripts\python.exe scripts\repair_environment.py
```

该工具修复本项目 editable 映射、激活脚本和已安装 Windows 命令入口，保留首次修改前的 `*.before-relocation`。它是开发维护工具，**不是无需系统 Python 的解决方案**；便携运行时无需执行它。

若只是想让 `.venv` 恢复可用，更简单的是删掉它再运行 `install.cmd --no-build`（uv 会复用 `runtime/uv/` 内的解释器与缓存），不需要修复旧的绝对路径。

## 引擎安装引导（应用会自动调用）

- `bootstrap_uv.py <target>`：使用随包 Python 标准库，从 Astral 官方 GitHub release 下载当前架构的 uv，只提取 `uv.exe` 并原子发布。已有 uv 不覆盖；下载/解压失败清理暂存目录并退出非零，不遗留半个 EXE。缺 uv 时安装服务自动执行。
- `ensure_engine_venv.py`：两种模式。默认（`precreate_venv`）为引擎在自身目录内创建 `venv`，调用项目 uv 的 `venv --python <ver> --managed-python --seed`；`--embedded` 则复制受管 CPython 到 `<引擎>/.lp/python`、写相对 `._pth`、删除 PEP 668 的 `EXTERNALLY-MANAGED`、从应用运行时播种 pip（缺 setuptools 时按镜像补装），**不创建 venv**。拒绝引擎目录之外或源码根目录的目标，已有解释器则复用。
- `normalize_editable.py <environment>`：把 `pip install -e` 生成的绝对 finder / `.pth` 换成相对导入根；embedded 环境写进 `pythonXY._pth`（`.pth` 只在 `site` 初始化时读取，而 embedded 不加载 `site`），venv 环境写 `site-packages/__lp_editable_*.pth`。安装完成后由服务自动调用。
- `migrate_env_to_embedded.py --engine-root … --from … --python-version … --uv … [--verify MODULE]`：冻结旧 venv 依赖 → 建 embedded 环境 → 装入同样依赖 → 校验模块 → 才把旧环境改名为 `<env>.migrated`。任一步失败都不动旧环境，从不删除文件。
- `install_uv_project.py --uv … --project … --python … [--group G] [--no-default-groups]`：把 uv 项目的依赖组 `uv export` 后装进非 venv 解释器（供 `uv sync` 不可用的 embedded 引擎使用）。
- AI Toolkit 调用自己的 `manager/__main__.py install`，自行创建环境、选择依赖，可能安装 Node / ffmpeg；Musubi 官方流程是 `uv sync [--extra cuXYZ]`，embedded 时改为 `uv pip install -e "<源>[extra]"`，依赖集合仍以该版本 `pyproject.toml` 为准。
- 安装服务配置 `runtime/uv/uv.exe`、`runtime/uv/cache/`、`runtime/uv/python/`，清理继承的 Python/PIP/UV/Conda 污染，避免落入系统环境。引擎仓库已存在的本地 uv 仍由其官方 manager 优先使用。
- 第一次安装需要网络及官方脚本要求的 Git 等工具。GitHub 连接失败在终端明确显示 failed；可修复网络后重试，或手动从 Astral 官方 uv release 解压对应架构的 `uv.exe` 到 `runtime/uv/uv.exe` 后重试，不需 pip，不改用不明镜像。

```powershell
# 可选：单独准备小型 uv 工具，不下载 PyTorch 或训练依赖
.\python_runtime\python.exe scripts\bootstrap_uv.py runtime\uv\uv.exe
# 运行应用与安装链路测试（无需大型训练包）
.\python_runtime\python.exe -m unittest discover -s tests -v
```

完整流程及失败恢复见 `docs/development/运行与安装.md`。不要在 `python_runtime` 内执行训练依赖的 pip install；引擎安装完成后应用会自动保存解释器并诊断，无需手动修改数据库。

## 启动与导出权限

发布构建会自动检查、规范化启动文件完整性标签；运行时自检包含项目外用户目录写入。EXE 对旧副本提供限定范围的自动修复，正常发行包不需要管理员运行。详细流程与系统授权边界见 `docs/development/启动权限与导出修复.md`。

## 设计要点

以下为实现方案，面向维护者；README 只描述用户可见的功能与用法。

- 窗口：pywebview，Windows 使用 WebView2；单窗口、单 WebView。
- 界面：Vue 3 + TypeScript + Vite；保留现有布局、主题和组件样式，原菱形徽标替换为 **LP Logo**。
- 后端：FastAPI + Uvicorn，本机 HTTP API + SSE。
- 训练：独立 supervisor 调用引擎原生 CLI，单 GPU 顺序队列；管理进程不加载训练模型。
- 引擎：直接复制源码到 `engine/`，同类实例共享 `adapters/<engine>/` 接口适配。
- 版本：多安装实例共存，记录源码 revision、解释器、环境诊断 manifest 和 adapter 版本；提交任务后绑定固定，不随默认实例切换。
- 存储：SQLite 索引与任务文件分离；模型和数据集使用本地路径，不上传、不默认复制整个数据集。
