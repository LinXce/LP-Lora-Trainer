# 引擎接口适配

一个子目录对应一种引擎；同一引擎的多个版本共享该目录，通过版本规则区分。

- `base.py`：公共适配契约。
- `capabilities.py`：参数与底模类型（底模类型 → 训练脚本）描述。
- `registry.py`：在 `adapters/` 下按目录发现适配器；每个目录一个 `adapter.py`，含且仅含一个具体 `EngineAdapter` 子类。加载失败的插件被隔离；重复 `engine_id` 时冲突双方均被禁用。
- `kohya/`：Kohya / sd-scripts 适配（可提交训练）。
- `musubi_tuner/`：Musubi Tuner 适配（缓存流水线 + 训练，可提交）。
- `ai_toolkit/`：AI Toolkit 适配（**正在适配**：参数登记与配置预览可用，训练提交尚未实现）。
- 各引擎 `profiles/`：版本能力、环境依赖与诊断档案。
- 各引擎 `templates/`：原生训练配置模板。

发现只做静态检查，不能执行未知源码，也不从 `engine/` 动态加载插件。
