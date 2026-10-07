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

在已有完整 Windows x64 CPython 和应用依赖的构建机执行：

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

引擎安装使用项目内 uv 管理基础 Python，不依赖系统 Python；但已安装的 `venv` / `.venv` 仍可能含绝对路径，复制后可能失效。迁移后重新安装或绑定并诊断。应用运行时不混入大型训练依赖；当前不自动制作可任意迁移的引擎便携包。

迁移前停止训练、退出 API/supervisor 后复制整个目录。数据库内显式保存的引擎、数据集、模型等绝对路径需在界面重新确认，不能盲目替换历史任务绑定。

## 开发 `.venv` 的盘符迁移修复

仅当开发环境的基础 Python 仍在时执行：

```powershell
.\.venv\Scripts\python.exe scripts\repair_environment.py
```

该工具修复本项目 editable 映射、激活脚本和已安装 Windows 命令入口，保留首次修改前的 `*.before-relocation`。它是开发维护工具，**不是无需系统 Python 的解决方案**；便携运行时无需执行它。

## 引擎安装引导（应用会自动调用）

- `bootstrap_uv.py <target>`：使用随包 Python 标准库，从 Astral 官方 GitHub release 下载当前架构的 uv，只提取 `uv.exe` 并原子发布。已有 uv 不覆盖；下载/解压失败清理暂存目录并退出非零，不遗留半个 EXE。缺 uv 时安装服务自动执行。
- `ensure_engine_venv.py`：为 Kohya 在引擎内部创建 `venv`，调用项目 uv 的 `venv --python 3.11 --managed-python --seed`；拒绝引擎目录之外或源码根目录的环境目标，已有解释器则复用。后续安装计划执行官方 setuptools 前置步骤和 headless 安装器。
- AI Toolkit 调用自己的 `manager/__main__.py install`，自行创建环境、选择依赖，可能安装 Node / ffmpeg；Musubi 调用自己的 `uv sync [--extra cuXYZ]`，extra 以该版本 `pyproject.toml` 为准，不猜通用 requirements。
- 安装服务配置 `runtime/uv/uv.exe`、`runtime/uv/cache/`、`runtime/uv/python/`，清理继承的 Python/PIP/UV/Conda 污染，避免落入系统环境。引擎仓库已存在的本地 uv 仍由其官方 manager 优先使用。
- 第一次安装需要网络及官方脚本要求的 Git 等工具。GitHub 连接失败在终端明确显示 failed；可修复网络后重试，或手动从 Astral 官方 uv release 解压对应架构的 `uv.exe` 到 `runtime/uv/uv.exe` 后重试，不需 pip，不改用不明镜像。

```powershell
# 可选：单独准备小型 uv 工具，不下载 PyTorch 或训练依赖
.\python_runtime\python.exe scripts\bootstrap_uv.py runtime\uv\uv.exe
# 运行应用与安装链路测试（无需大型训练包）
.\python_runtime\python.exe -m unittest discover -s tests -v
```

完整流程及失败恢复见 `docs/运行与安装.md`。不要在 `python_runtime` 内执行训练依赖的 pip install；引擎安装完成后应用会自动保存解释器并诊断，无需手动修改数据库。
