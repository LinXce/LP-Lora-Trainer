# 引擎适配器

[English](./README.md) · **中文**

LP 能驱动的每个训练引擎，都是这里的一个目录。适配器负责该引擎的**全部**专属知识 ——
识别、安装、参数面、原生配置渲染、启动命令、日志解析、产物收集。主程序与前端**从不出现
引擎名**，只渲染适配器目录提供的内容；这一点有测试强制（`tests/test_shell_is_engine_agnostic.py`）。

## 目录结构

| 路径 | 作用 |
|---|---|
| `base.py` | 适配契约（`EngineAdapter`）与 `ADAPTER_CONTRACT_VERSION`。 |
| `capabilities.py` | `ArchSpec`（底模类型 → 训练入口脚本）与 `Param`（单个参数）。 |
| `registry.py` | 按目录发现适配器、隔离损坏插件、处理重复 id。 |
| `extra_params.py` | 「自定义参数」桥接（供需要它的适配器共用）。 |
| `sampling.py` | 采样提示词文件的共用处理。 |
| `<engine_id>/adapter.py` | 含且仅含一个具体 `EngineAdapter` 子类。 |
| `<engine_id>/params.py` | 该引擎的底模与参数表（约定目录）。 |

`engine_id` 必须匹配 `[a-z][a-z0-9_]*`。发现过程是**静态**的：不导入、不执行引擎源码，
也绝不从 `engine/` 动态加载插件。

## 契约

必须实现（`base.py` 中的抽象方法）：

`detect` · `capabilities` · `validate` · `write_native_config` · `build_launch` ·
`parse_log` · `recovery_capabilities` · `collect_artifacts`

常用覆盖项：

- **安装** —— `installation_plan`、`installation_sources`、`source_issues`、
  `environment_strategy`（`embedded` 可迁移 / `precreate_venv` / `uv_project` / `native`）、
  `environment_dir`、`python_version`、`torch_index_mode`（`named` / `env` / `none`）。
  声明 `embedded` 时，`installation_plan` 必须能在**没有 venv** 的解释器上工作：
  用 `uv pip install --python <解释器>` 或 `pip`（此时 `embedded_mode` 为真，不要再传
  `--require-virtualenv`），并把任何可执行文件按「解释器目录 → Scripts/bin」两处定位。
- **行为声明** —— `supports_nested_images`、`creates_output_dir`、`submittable`、
  `native_format`、`diagnostic_modules`。
- **界面元数据** —— `display_name`、`adapter_status`、`training_notice`、
  `installation_help`、`clone_url` / `clone_branch` / `clone_commit`。
- **扩展点** —— `preview_commands`、`native_config_text`、`export_parameters`、
  `export_parameter_files`、`caption_cache_notices`、`finalize_artifacts`
  （训练结束后的收尾，例如生成第三方工具可读的副本）。

`finalize_artifacts` 只在最后一个阶段**成功退出**后调用一次；它的返回文本会写进任务日志，
**永远不会**把成功的训练改判为失败。

## 参数表

- `ARCHITECTURES: tuple[ArchSpec, ...]` —— 每个底模类型一条，并声明训练入口脚本。
- `PARAMS: tuple[Param, ...]` —— 引擎真实的参数面：
  - `architectures=("id",)` 把参数限定到对应底模；**留空表示该引擎的全部底模都适用**
    （同一训练器 + 多个底模时成立；同一克隆里两条不同训练线时是错误的用法）。
  - `native_key` 把界面键映射到引擎自己的开关/字段名。
  - `section` 与 `display_order` 决定渲染的区块与顺序。
  - `min` / `max` / `options` 在提交前参与校验。
  - 除数为零会崩引擎的项（保存/采样间隔等）必须 `min=1`。

测试强制的硬规则：参数键不重复、**同一底模下标签不重复**、每个 `native_key` 必须存在于
引擎自身的定义中（`tests/test_adapter_contract.py`）。

## 新增一个引擎

1. 把引擎源码放到 `engine/<name>/`。
2. 新建 `adapters/<engine_id>/adapter.py`（一个具体子类）与 `params.py`
   （`ARCHITECTURES` + `PARAMS`）。
3. 实现八个必须方法，按需覆盖声明项（环境策略、`clone_url`、状态文案等）。
4. 让 `detect` **只认本引擎**：校验它自己的包名与入口脚本，避免误判结构相近的同类引擎。
5. 运行 `python -m unittest discover -s tests`。契约测试会**自动**覆盖新引擎；
   正常情况下 **不需要改 `app/` 或 `frontend/`**。

## 实践中踩过的坑

- **引擎默认 `True` 的布尔开关**必须显式传 `--no-<flag>`；什么都不传等于保持开启。
- 某些引擎**把缓存放在输出目录里** —— 必须在 `collect_artifacts` 中排除，否则缓存文件会被
  当成 checkpoint 列出来。
- **同名≠同义**：不同引擎的同名参数默认值/取值范围不同，不要把两个训练器的参数合并成一个。
- **不修改引擎自己的文件**；只通过开关与生成的配置文件从外部适配。
