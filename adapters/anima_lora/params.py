"""anima_lora parameter surface.

Source of truth: ``engine/anima_lora/train.py`` (``setup_parser``) plus the argument
adders it calls — ``library/config/cli_args.py``, ``library/anima/training.py``,
``library/training/losses.py``, ``library/log.py``, ``library/models/sai_spec.py``.
The engine loads a flat TOML (``--config_file``), so sections here are cosmetic.

Only flags that exist in this engine are declared. In particular it has no
``--fused_backward_pass``, no ``--cpu_offload_checkpointing``, no ``--lowvram`` /
``--medvram`` (low VRAM is a preset there), no ``--cache_latents`` /
``--cache_text_encoder_outputs`` (renamed to ``--use_vae_cache`` / ``--use_text_cache``)
and no ``--sdpa`` / ``--xformers`` / ``--flash_attn`` / ``--mem_eff_attn``.

MOSS-Image is a separate training line in the same clone: it trains from a
pre-cached folder through ``scripts/moss/train.py``, not from a dataset config.
Its flags live in ``library/moss/requests.py`` and are declared once that file is
read; until then only its architecture entry and the cache folder are offered.
"""

from dataclasses import replace
import ast
from pathlib import Path
from typing import Any
import tomllib

from adapters.capabilities import ArchSpec, Param, enum, param

# Engine config chain for the LoRA line. Defaults are read from source on every
# capabilities call, so an engine update that changes these files also updates the
# form defaults without an LP code change.
_ENGINE_DEFAULT_LAYERS = (
    ("configs/base.toml", None),
    ("configs/presets.toml", "default"),
    ("configs/methods/lora.toml", None),
)
# Model paths in base.toml are relative to the engine checkout; LP fills those from
# its own pickers. torch.compile stays opt-in on Windows (needs MSVC cl.exe).
# `logging_dir` is excluded because base.toml points it at the engine checkout
# (`output/logs`): the adapter renders a task-local directory instead, so log
# files never land inside the engine source tree.
# `vocab_pack` is excluded on purpose: base.toml enables the CJK pack, but LP's
# preprocess stage (`tasks.py preprocess-config`) cannot forward it to the TE cache
# pass, so inheriting it would train on caches built with the stock tokenizer while
# the sampler used the pack. The parameter help states the engine default, the exact
# engine-relative prefix, and how to enable it consistently; `validate` warns when a
# user enables it with caching on.
_ENGINE_DEFAULT_EXCLUDE = {
    "torch_compile", "pretrained_model_name_or_path", "vae", "qwen3",
    "llm_adapter_path", "t5_tokenizer_path", "vocab_pack", "network_weights",
    "sample_prompts", "mask_dir", "pooled_text_proj", "ema_resume_path",
    "logging_dir",
}

# Keys the config chain carries but LP owns as job paths/identity. They must never
# be materialized into the generated TOML, or a run would read/write engine-relative
# folders instead of the task's own output.
_ENGINE_EXTRAS_EXCLUDE = _ENGINE_DEFAULT_EXCLUDE | {
    "output_dir", "output_name", "resized_image_dir", "lora_cache_dir",
    "dataset_config", "config_file", "method", "preset", "methods_subdir",
    "save_model_as", "training_comment",
}


def engine_defaults(source_path: str | Path) -> dict[str, Any]:
    """Flat scalar defaults from the engine's base → preset → lora method chain.

    The files are read, never imported: capabilities stay cheap and an engine's
    Python is not entered just to render a form. Unreadable/invalid files fall back
    to the static defaults in this module.
    """
    root = Path(source_path)
    values: dict[str, Any] = {}
    for relative, section in _ENGINE_DEFAULT_LAYERS:
        try:
            with (root / relative).open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            continue
        if section:
            data = data.get(section) or {}
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            # Datasets/sections are not form defaults.
            if isinstance(value, (dict, list)):
                continue
            values[str(key)] = value
    return values


def _coerce_default(p: Param, value: Any) -> Any:
    if p.type == "bool":
        return bool(value) if isinstance(value, bool) else p.default
    if p.type == "int":
        if isinstance(value, bool):
            return p.default
        try:
            return int(value)
        except (TypeError, ValueError):
            return p.default
    if p.type == "float":
        if isinstance(value, bool):
            return p.default
        try:
            return float(value)
        except (TypeError, ValueError):
            return p.default
    if p.type == "enum":
        allowed = {option for option, _label in p.options}
        return value if isinstance(value, str) and value in allowed else p.default
    if p.type == "string":
        return value if isinstance(value, str) else p.default
    return value


def apply_engine_defaults(params: tuple[Param, ...], source_path: str | Path) -> tuple[Param, ...]:
    """Overlay the engine's current config defaults on a static parameter set."""
    defaults = engine_defaults(source_path)
    if not defaults:
        return params
    updated = []
    for p in params:
        key = p.native_key or p.key
        if not p.native or p.path_kind or key in _ENGINE_DEFAULT_EXCLUDE or key not in defaults:
            updated.append(p)
            continue
        updated.append(replace(p, default=_coerce_default(p, defaults[key])))
    return tuple(updated)


_MOSS_DEFAULT_EXCLUDE = {"compile"}


def moss_engine_defaults(source_path: str | Path) -> dict[str, Any]:
    """Defaults declared by MOSS ``CacheRequest`` / ``TrainRequest`` dataclasses.

    Parsed statically instead of importing request modules: capabilities must not
    enter the engine Python just to render defaults. Literal defaults that reference
    class constants stay on the adapter's static fallback.
    """
    path = Path(source_path) / "library" / "moss" / "requests.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return {}
    values: dict[str, Any] = {}
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in ("CacheRequest", "TrainRequest"):
            continue
        for field in node.body:
            if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                continue
            call = field.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_f"):
                continue
            if not call.args:
                continue
            arg = call.args[0]
            if isinstance(arg, ast.Name) and arg.id in constants:
                values[field.target.id] = constants[arg.id]
                continue
            try:
                values[field.target.id] = ast.literal_eval(arg)
            except (ValueError, TypeError, SyntaxError):
                continue
    return values


def apply_moss_defaults(params: tuple[Param, ...], source_path: str | Path) -> tuple[Param, ...]:
    defaults = moss_engine_defaults(source_path)
    if not defaults:
        return params
    updated = []
    for p in params:
        key = p.native_key or p.key
        if "moss_image" not in p.architectures or key in _MOSS_DEFAULT_EXCLUDE or key not in defaults:
            updated.append(p)
            continue
        updated.append(replace(p, default=_coerce_default(p, defaults[key])))
    return tuple(updated)


def engine_chain_extras(source_path: str | Path, architecture: str = "anima") -> dict[str, Any]:
    """Config-chain defaults the form does not model, for a ``--config_file`` run.

    ``train.py --config_file`` does **not** merge ``base.toml → presets → method``
    (``library/config/io.py::read_config_from_file`` only walks the chain when no
    config file is given), so every value the form does not manage would silently
    fall back to the argparse default. Materializing the chain scalars the rendered
    file is missing keeps the LoRA method defaults (``down_init``, ``use_repa``,
    ``use_timestep_mask``, …) in effect for the actual run.
    """
    defaults = engine_defaults(source_path)
    if not defaults:
        return {}
    modeled = {
        (p.native_key or p.key) for p in PARAMS
        if p.native and p.applies_to(architecture)
    }
    return {
        key: value for key, value in defaults.items()
        if key not in modeled and key not in _ENGINE_EXTRAS_EXCLUDE
    }


def default_values(source_path: str | Path) -> dict[str, Any]:
    """Effective defaults for every parameter, including engine/config values.

    One source of truth for capabilities, validation, preview rendering and the
    generated job config; using only capabilities made API/export paths fall back
    to the static table after an engine update.
    """
    params = apply_engine_defaults(PARAMS, source_path)
    params = apply_moss_defaults(params, source_path)
    return {p.key: p.default for p in params}

ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("anima", "Anima", "train.py"),
    ArchSpec("moss_image", "MOSS-Image", "scripts/moss/train.py"),
)

# Values --optimizer_type dispatches in library/training/optimizers.py; the shipped
# configs/base.toml default is AdamW (channel scaling relies on Adam's step size).
OPTIMIZERS = enum([
    "AdamW", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit",
    "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit",
    "Prodigy", "DAdaptAdam", "DAdaptAdan", "DAdaptLion", "DAdaptSGD",
    "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree",
])

# Values ``library/training/schedulers.py`` accepts through
# ``transformers.optimization.SchedulerType`` (plus the diffusers
# ``piecewise_constant`` gate the engine handles explicitly). ``adafactor``/``rex``
# are diffusers-only names: this engine raises ``ValueError`` on them.
LR_SCHEDULERS = enum([
    "constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
    "cosine_with_min_lr", "linear", "polynomial", "inverse_sqrt",
    "piecewise_constant",
])

ATTENTION_MODES = enum(["", "torch", "flash", "flex", "sdpa"])

# Flags this engine commit declares but never reads. A form field for one of them is a
# knob that cannot do anything, so the spec marks it unavailable: the frontend disables
# the field and shows the reason, and ``native=False`` keeps the dead key out of the
# generated config. Drop a marker once the engine starts consuming the flag.
_UNAVAILABLE_EMA = (
    "当前引擎只声明了 EMA 开关，训练/保存路径从未读取它"
    "（library/training/ema.py 的 make_ema 没有调用者），设置不会生效。"
)
_UNAVAILABLE_GROUP_LR = (
    "当前引擎的分组学习率只在 get_anima_param_groups 中定义，训练入口没有调用它，设置不会生效。"
)
_UNAVAILABLE_METADATA = (
    "当前引擎只读取标题/作者/描述/许可/标签；该字段虽在 argparse 中声明，但未写入模型元数据。"
)
_UNAVAILABLE_LLM_ADAPTER_PATH = (
    "当前训练入口不读取该参数：llm_adapter 权重始终从底模权重读取。"
)

_PARAM_HELP = {
    "pretrained_model_name_or_path": "Anima 底模权重；必须包含 blocks.N.* 与 llm_adapter.* 键，缺少 llm_adapter 的权重引擎会直接报错。",
    "vae": "与底模匹配的 VAE 权重路径，用于潜空间缓存。",
    "qwen3": "Qwen3-0.6B 文本编码器权重（文件或目录）；引擎仓库不自带，需要单独准备。",
    "llm_adapter_path": "可选的独立 llm_adapter 权重；留空则从底模权重中读取。",
    "t5_tokenizer_path": "可选的 T5 分词器目录；留空使用引擎自带的 library/anima/configs/t5_old，无需另下。",
    "max_train_steps": "目标训练步数；填写训练轮数时由引擎按轮数计算训练长度。",
    "max_train_epochs": "可选训练轮数；留空使用训练步数，填写后覆盖按步数设置的训练长度。",
    "mixed_precision": "引擎默认 bf16；fp16 / no 在引擎文档中标注为未测试。",
    "save_precision": "输出权重的保存精度，与训练混合精度分别设置。",
    "save_every_n_epochs": "每隔多少轮保存一次权重；必须 ≥ 1，留空表示不按轮保存。",
    "save_every_n_steps": "每隔多少步保存一次权重；必须 ≥ 1，留空表示不按步保存。",
    "network_dim": "LoRA 的 Rank，决定低秩网络容量；通常越大，训练参数、显存和输出文件也越大。",
    "network_alpha": "LoRA 缩放参数，与 Rank 配合控制更新的缩放强度。",
    "network_args": "可选网络附加参数，按 key=value 输入；空格或换行分隔，必须由当前网络模块支持。",
    "learning_rate": "引擎 CLI 默认 2.0e-6，当前 lora 方法预设 2.0e-5；应结合底模与数据量调整。",
    "optimizer_type": "引擎默认 AdamW，且默认开启的通道缩放按 Adam 的步长特性设计，换优化器前建议先做对比。",
    "blocks_to_swap": "引擎标为实验性；交换块数量受模型层数限制，且不能与 unsloth 卸载检查点同时启用。",
    "unsloth_offload_checkpointing": "把激活异步卸载到 CPU 内存；引擎说明不能与 blocks_to_swap 同时使用。",
    "attn_mode": "留空使用 torch；sageattn 只支持推理、不能训练；sdpa 为向后兼容保留。",
    "torch_compile": "引擎默认开启（configs/base.toml）；需要 CUDA 13.2 工具链，关闭后速度明显下降。",
    "activation_memory_budget": "Inductor 保存用于反向的激活上限（0–1），用于换取显存余量；留空用引擎默认。",
    "use_vae_cache": "缓存 VAE 潜空间（引擎内部即 cache_latents / _to_disk）；改动图片或分辨率后需要清空缓存。",
    "use_text_cache": "缓存文本编码器输出（引擎内部即 cache_text_encoder_outputs / _to_disk）；改动字幕不会自动重算。",
    "caption_extension": "引擎默认是 .caption，但仓库示例与数据都用 .txt；两边必须一致，否则读不到字幕。",
    "caption_separator": "字幕标签分隔符，引擎默认是英文逗号。",
    "num_repeats": "该子集图片的重复次数，写入数据集配置。",
    "recursive": "是否递归读取子目录，写入数据集配置。",
    "resolution_width": "训练宽度；本引擎的数据集子集没有分辨率键，LP 的预处理阶段也按引擎的分档（512–1536）自动决定尺寸，"
                        "因此该值当前不参与训练/预处理，不会写进数据集配置。",
    "resolution_height": "训练高度；本引擎的数据集子集没有分辨率键，LP 的预处理阶段也按引擎的分档（512–1536）自动决定尺寸，"
                         "因此该值当前不参与训练/预处理，不会写进数据集配置。",
    "gradient_checkpointing": "以重算换取较低显存占用，通常会降低速度；引擎只把低显存预设默认打开它。",
    "sample_prompts": "可选；每行一个提示词，配合采样间隔使用；本引擎通过 --sample_prompts 读取，训练与采样格式需一致。",
    "progress_jsonl": "结构化进度输出，便于持续读取步数、损失与速度；留空则沿用引擎的文本日志。",
    "extra_params": "补充 LP 未提供的引擎参数，会合并进引擎配置文件；空格或换行分隔，含空格的值用引号，单独的 --开关 表示开启；与上面已提供的参数重复会被拒绝。",
}

PARAMS: tuple[Param, ...] = (
    # ---- basic / 训练 ----
    param("max_train_steps", "训练步数", "int", None, min=1, max=10000000, section="训练"),
    param("max_train_epochs", "训练轮数", "int", 8, min=1, max=100000, section="训练"),
    param("seed", "随机种子", "int", None, min=0, max=4294967295, section="训练"),
    param("gradient_checkpointing", "梯度检查点", "bool", False, section="训练"),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024, section="训练"),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 1, min=0, max=64, section="训练"),
    param("mixed_precision", "混合精度", "enum", "bf16", options=enum(["no", "fp16", "bf16"]), section="训练"),
    param("save_precision", "保存精度", "enum", "bf16", options=enum(["None", "float", "fp16", "bf16"]),
          section="训练"),

    # ---- basic / 数据集（写入数据集配置或影响预处理）----
    param("resolution_width", "训练宽度", "int", 1024, min=64, max=4096, step=16, native=False, section="数据集"),
    param("resolution_height", "训练高度", "int", 1024, min=64, max=4096, step=16, native=False, section="数据集"),
    param("batch_size", "Batch Size", "int", 1, min=1, max=128, native=False, section="数据集"),
    param("caption_extension", "Caption 扩展名", "enum", ".txt", native=False,
          options=enum([".txt", ".caption"]), section="数据集"),
    param("caption_separator", "标签分隔符", "enum", ",", native=False,
          options=enum([",", ";", "|"]), section="数据集"),
    param("caption_dropout_rate", "字幕丢弃率", "float", 0.1, "advanced", "数据增强",
          min=0.0, max=1.0, architectures=("anima",),
          help="按概率清空字幕（CFG 的 uncond）；引擎 lora 方法预设为 0.1。"),
    param("use_shuffled_caption_variants", "打乱字幕变体", "bool", True, "advanced", "数据增强",
          architectures=("anima",),
          help="训练时从缓存的 tag-dropped 变体中抽取；需要预处理阶段生成变体，引擎 lora 方法预设开启。"),
    param("cache_llm_adapter_outputs", "缓存 LLM Adapter 输出", "bool", True, "advanced", "数据增强",
          architectures=("anima",),
          help="把冻结 LLM adapter 的输出也写入文本缓存；仅在启用文本缓存时有效。"),
    param("num_repeats", "图片重复次数", "int", 1, min=1, max=1000, native=False, section="数据集"),
    param("recursive", "递归读取子目录", "bool", True, native=False, section="数据集"),
    param("use_vae_cache", "缓存 VAE 潜空间", "bool", True, section="数据集"),
    param("use_text_cache", "缓存文本编码器输出", "bool", True, section="数据集"),

    # ---- basic / 模型与编码器 ----
    # Order matches the other engines: base model, then VAE, then text encoders.
    # The shared frontend only knows the generic model-file keys, so this order is
    # declared here instead of adding engine keys to the shell.
    param("pretrained_model_name_or_path", "底模路径", "string", "", native_key="pretrained_model_name_or_path",
          architectures=("anima",), section="模型与编码器", display_order=0),
    param("vae", "VAE 路径", "string", "", architectures=("anima",), section="模型与编码器", display_order=10),
    param("qwen3", "Qwen3 文本编码器路径", "string", "", architectures=("anima",), section="模型与编码器",
          display_order=20),
    param("llm_adapter_path", "llm_adapter 路径", "string", "", architectures=("anima",), section="模型与编码器",
          display_order=30, native=False, unsupported_reason=_UNAVAILABLE_LLM_ADAPTER_PATH),
    param("t5_tokenizer_path", "T5 分词器路径", "string", "", architectures=("anima",), section="模型与编码器",
          display_order=40),
    param("vocab_pack", "CJK 词表包路径", "string", "", native_key="vocab_pack", architectures=("anima",),
          section="模型与编码器", display_order=50,
          help="引擎 configs/base.toml 自 v2 起默认开启 CJK 词表包，值是相对引擎目录的前缀 "
               "models/vocab_packs/anima_cjk_vocab_pack_jp_v1（成对的 .safetensors/.json，缺失时由引擎自行下载）。"
               "本适配器默认留空（=关闭）：LP 的预处理阶段 tasks.py preprocess-config 无法把该词表包传给 "
               "TE 缓存步骤，继承引擎默认会让训练读到的缓存与采样提示词使用不同词表。"
               "要启用：填写该前缀，并先在引擎内用词表包重建 TE 缓存（make preprocess-te ARGS=--overwrite）。"),

    # ---- basic / 网络 ----
    param("network_module", "网络模块", "string", "networks.lora_anima", architectures=("anima",), section="网络"),
    param("network_dim", "LoRA Rank", "int", 32, min=1, max=1024, architectures=("anima",), section="网络"),
    param("network_alpha", "LoRA Alpha", "float", 128.0, min=0.0, max=1024.0, architectures=("anima",), section="网络"),
    param("network_dropout", "网络 Dropout", "float", None, min=0.0, max=1.0, architectures=("anima",), section="网络"),
    param("network_weights", "初始权重路径", "string", "", native_key="network_weights",
          architectures=("anima",), section="网络"),
    param("network_args", "网络附加参数", "string", "", native_key="network_args", list_arg=True, multiline=True,
          architectures=("anima",), section="网络"),

    # ---- basic / 优化 ----
    param("optimizer_type", "优化器", "enum", "AdamW", options=OPTIMIZERS, architectures=("anima",), section="优化"),
    param("optimizer_args", "优化器附加参数", "string", "", native_key="optimizer_args", list_arg=True,
          multiline=True, architectures=("anima",), section="优化"),
    param("learning_rate", "学习率", "float", 0.00002, min=0.00000001, max=1.0, architectures=("anima",),
          section="优化"),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0.0, max=100.0, architectures=("anima",), section="优化"),
    param("lr_scheduler", "学习率策略", "enum", "cosine", options=LR_SCHEDULERS, architectures=("anima",),
          section="优化"),
    param("lr_scheduler_type", "自定义调度器模块", "string", "", architectures=("anima",), section="优化"),
    param("lr_scheduler_args", "调度器附加参数", "string", "", native_key="lr_scheduler_args", list_arg=True,
          multiline=True, architectures=("anima",), section="优化"),
    param("lr_warmup_steps", "预热步数/比例", "float", 0.05, min=0, max=1000000, architectures=("anima",), section="优化",
          help="留空使用引擎默认；0–1 为训练步数比例，大于 1 按步数解释。"),
    param("lr_scheduler_num_cycles", "重启次数", "int", 1, min=1, max=100, architectures=("anima",), section="优化"),

    # ---- advanced / 保存 ----
    param("save_every_n_epochs", "保存间隔（轮）", "int", 8, "advanced", "保存",
          architectures=("anima",), min=1, max=10000),
    param("save_every_n_steps", "保存间隔（步）", "int", None, "advanced", "保存",
          architectures=("anima",), min=1, max=10000000),
    param("save_state", "保存训练状态", "bool", False, "advanced", "保存", architectures=("anima",)),
    param("save_state_on_train_end", "训练结束保存状态", "bool", False, "advanced", "保存",
          architectures=("anima",)),
    param("checkpointing_epochs", "自动续训间隔（轮）", "int", 8, "advanced", "保存",
          architectures=("anima",), min=1, max=10000),
    param("resume", "续训状态路径", "string", "", "advanced", "保存", native_key="resume",
          path_kind="directory", architectures=("anima",)),

    # ---- advanced / 内存与加速 ----
    param("blocks_to_swap", "交换块数量", "int", 0, "advanced", "内存与加速", min=0, max=200),
    param("unsloth_offload_checkpointing", "卸载检查点到 CPU", "bool", False, "advanced", "内存与加速"),
    param("highvram", "高显存模式", "bool", False, "advanced", "内存与加速"),
    param("lowram", "低内存模式", "bool", False, "advanced", "内存与加速"),
    param("attn_mode", "注意力实现", "enum", "flash", "advanced", "内存与加速", options=ATTENTION_MODES),
    param("torch_compile", "torch.compile", "bool", False, "advanced", "内存与加速",
          help="引擎的 configs/base.toml 默认开启（配合 compile_dynamic_seq），但需要 CUDA 13.2 工具链与 Windows 上的 MSVC；未安装 cl.exe 时会以 InvalidCxxCompiler 失败，所以这里默认关闭。"),
    param("compile_dynamic_seq", "动态序列轴", "bool", True, "advanced", "内存与加速",
          help="仅在启用 torch.compile 时生效；把序列长度标记为动态，避免按 tier 反复重编译（引擎 base.toml 默认开启）。"),
    param("compile_seq_bands", "按 token 区间分带", "bool", False, "advanced", "内存与加速",
          help="需要同时开启动态序列轴；按 token 数分带各自编译，单 tier 数据集无收益。"),
    param("compile_inductor_mode", "Inductor 模式", "string", "", "advanced", "内存与加速",
          help="转发给 torch.compile(mode=…)，留空用引擎默认（inductor）。'reduce-overhead' 会启用 CUDAGraphs，与块交换不兼容。"),
    param("activation_memory_budget", "激活显存预算", "float", 0.99, "advanced", "内存与加速", min=0.0, max=1.0,
          help=_PARAM_HELP["activation_memory_budget"]),
    param("vae_chunk_size", "VAE 分块大小", "int", 64, "advanced", "内存与加速", min=1, max=512),
    param("no_half_vae", "VAE 不使用半精度", "bool", False, "advanced", "内存与加速"),

    # ---- advanced / 日志 ----
    param("logging_dir", "日志目录", "string", "", "advanced", "日志", path_kind="directory"),
    param("log_with", "日志后端", "enum", "", "advanced", "日志", options=enum(["", "tensorboard", "wandb", "all"])),
    param("log_every_n_steps", "日志间隔（步）", "int", None, "advanced", "日志", min=1, max=1000000),
    param("progress_jsonl", "结构化进度文件", "string", "", "advanced", "日志"),

    # ---- advanced / 时间步与采样 ----
    # Engine default after the upstream pull: sigmoid (library/anima/training.py:352).
    param("timestep_sampling", "时间步采样", "enum", "sigmoid", "advanced", "时间步与采样",
          options=enum(["sigma", "uniform", "sigmoid", "shift", "flux_shift"])),
    param("discrete_flow_shift", "离散流偏移", "float", None, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("sigmoid_scale", "Sigmoid 缩放", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=20.0),
    # Mirrored so the pair stays visible; only used by timestep_sampling=sigmoid.
    param("sigmoid_bias", "Sigmoid 偏置", "float", 0.0, "advanced", "时间步与采样", min=-10.0, max=10.0),
    param("weighting_scheme", "加权方案", "enum", "none", "advanced", "时间步与采样",
          options=enum(["logit_normal", "mode", "cosmap", "sigma_sqrt", "min_snr", "uniform", "none"])),
    param("logit_mean", "Logit 均值", "float", 0.0, "advanced", "时间步与采样", min=-10.0, max=10.0),
    param("logit_std", "Logit 标准差", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=10.0),
    param("min_snr_gamma", "Min-SNR Gamma", "float", None, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("sample_every_n_epochs", "采样间隔（轮）", "int", None, "preview", "时间步与采样", min=1, max=10000),
    param("sample_every_n_steps", "采样间隔（步）", "int", None, "preview", "时间步与采样", min=1, max=10000000),
    param("sample_at_first", "训练前采样", "bool", False, "preview", "时间步与采样"),
    param("sample_prompts", "采样提示词", "string", "", "preview", "时间步与采样",
          native_key="sample_prompts",
          help="每行一个提示词；留空且启用了采样间隔时，自动改用数据集第一张图的 caption。"),

    # ---- advanced / 元数据与输出 ----
    param("metadata_title", "标题", "string", "", "advanced", "元数据"),
    param("metadata_author", "作者", "string", "", "advanced", "元数据"),
    param("metadata_description", "描述", "string", "", "advanced", "元数据"),
    param("metadata_license", "许可", "string", "", "advanced", "元数据"),
    param("metadata_tags", "标签", "string", "", "advanced", "元数据"),
    param("no_metadata", "不写入元数据", "bool", False, "advanced", "元数据"),
    param("save_model_as", "输出格式", "enum", "safetensors", "advanced", "元数据",
          options=enum(["safetensors", "ckpt", "pt"]),
          help="引擎默认 safetensors；其他格式兼容性更差，一般不需要改。"),

    # ---- advanced / 自定义参数 ----
    param("extra_params", "自定义参数", "string", "", "advanced", "自定义参数", multiline=True, native=False),
)

# MOSS-Image trains from a pre-cached folder. Every flag below is a field of
# library/moss/requests.py (TrainRequest / CacheRequest); the engine takes CLI flags
# only — it has no config file. Its VAE and text encoder are not separate flags: both
# live inside the model directory, one diffusers folder per variant
# (transformer/, text_encoder/, vae/, tokenizer/, scheduler/).
MOSS_PARAMS: tuple[Param, ...] = (
    # ---- basic / 数据集（缓存阶段）----
    param("moss_cache_directory", "MOSS 缓存目录", "string", "", "basic", "数据集", native=False,
          path_kind="directory", architectures=("moss_image",),
          help="先由引擎的 scripts/moss/cache.py 生成（图片目录 + .txt 字幕），训练直接读取该目录；留空写入任务目录内。"),
    param("moss_resolution", "缓存分辨率上限", "int", 1024, "basic", "数据集", min=256, max=4096, step=16,
          architectures=("moss_image",), native_key="resolution",
          help="缓存阶段的像素面积上限（按方形边长表示），保持原图长宽比、不放大。"),
    param("moss_caption_format", "字幕格式", "enum", "layout_json", "basic", "数据集",
          options=enum(["layout_json", "text"]), architectures=("moss_image",), native_key="caption_format",
          help="layout_json 把 Anima 风格标签映射为 MOSS 的布局 JSON（推荐）；训练与采样必须使用同一种格式。"),
    # New in MOSS Toolkit 0.5.0: the cache pass got its own precision/IO controls.
    param("moss_cache_text_encoder", "缓存 TE 精度", "enum", "int8", "advanced", "数据集",
          options=enum(["int8", "fp8", "bf16"]), architectures=("moss_image",), native_key="text_encoder",
          help="int8 为引擎默认，会使用单独的 INT8 量化文件（首次自动下载，显存约 9.2 GB）；"
               "已有完整 MOSS 模型且显存足够时选 bf16，可直接读取模型目录里的 text_encoder，不再下载量化文件。"),
    param("moss_cache_vae_dtype", "缓存 VAE 精度", "enum", "fp32", "advanced", "数据集",
          options=enum(["fp32", "bf16"]), architectures=("moss_image",), native_key="vae_dtype",
          help="缓存阶段 FLUX.2 VAE 的计算精度。"),
    param("moss_cache_max_text_tokens", "缓存 TE 截断长度", "int", 4096, "advanced", "数据集",
          min=1, max=65536, architectures=("moss_image",), native_key="max_text_tokens",
          help="缓存字幕编码的最大 token 数；超过后截断。"),
    param("moss_cache_overwrite", "重建缓存", "bool", False, "advanced", "数据集",
          architectures=("moss_image",), native_key="overwrite",
          help="覆盖已有缓存；修改分辨率、模型或字幕格式后需要启用。"),
    param("moss_cache_skip_text", "跳过文本缓存", "bool", False, "advanced", "数据集",
          architectures=("moss_image",), native_key="skip_text",
          help="只重建潜空间缓存，保留已有文本缓存。"),
    param("moss_cache_skip_latents", "跳过潜空间缓存", "bool", False, "advanced", "数据集",
          architectures=("moss_image",), native_key="skip_latents",
          help="只重建文本缓存，保留已有潜空间缓存。"),
    param("moss_cache_save_crops", "保存裁切图", "bool", False, "advanced", "数据集",
          architectures=("moss_image",), native_key="save_crops",
          help="把缓存时的裁切结果另存，便于检查构图；会额外占用磁盘。"),
    param("moss_ref_scale", "i2i 参考图缩放", "float", 1.0, "advanced", "数据集",
          min=0.0, max=8.0, architectures=("moss_image",), native_key="ref_scale",
          help="i2i 缓存参考图的缩放系数；1 = 原尺寸。"),
    param("moss_fixed_caption", "i2i 固定字幕", "string", "", "advanced", "数据集",
          architectures=("moss_image",), native_key="fixed_caption",
          help="i2i 缓存时对所有样本使用同一条字幕；留空使用每张图自己的字幕。"),

    # ---- basic / 模型 ----
    param("moss_model_dir", "MOSS-Image-Base 路径", "string", "", "basic", "模型与编码器", path_kind="directory",
          native=False, architectures=("moss_image",), display_order=0, required_for=("moss_image",),
          help="该底模类型唯一的模型输入（通用「底模文件」在此类型下不显示）：选择 MOSS-Image-Base 这个 "
               "diffusers 文件夹本身（内含 transformer、text_encoder、vae、tokenizer、scheduler）；"
               "提交时自动取其父目录作为引擎的 model_dir，并固定使用 base 变体。"),

    # ---- basic / 训练 ----
    param("moss_epochs", "训练轮数", "int", 8, "basic", "训练", min=1, max=10000,
          architectures=("moss_image",), native_key="epochs"),
    param("moss_max_steps", "训练步数上限", "int", 0, "basic", "训练", min=0, max=10000000,
          architectures=("moss_image",), native_key="max_steps",
          help="大于 0 时提前结束，覆盖训练轮数；0 表示由轮数决定。"),
    param("moss_grad_accum", "梯度累积", "int", 1, "basic", "训练", min=1, max=1024,
          architectures=("moss_image",), native_key="grad_accum",
          help="每个优化器更新累积的样本数；步数/轮数/日志仍按样本计数，预热按更新计数。"),
    param("moss_seed", "随机种子", "int", 0, "basic", "训练", min=0, max=4294967295,
          architectures=("moss_image",), native_key="seed"),

    # ---- basic / 网络（LoRA）----
    param("moss_algo", "适配器类型", "enum", "lora", "basic", "网络",
          options=enum(["lora", "lokr", "loha"]), architectures=("moss_image",), native_key="algo",
          help="lora 为常规 LoRA；lokr / loha 为 LyCORIS 变体（Kronecker / Hadamard 乘积）。"),
    param("moss_rank", "LoRA Rank", "int", 32, "basic", "网络", min=1, max=1024,
          architectures=("moss_image",), native_key="rank",
          help="引擎默认 32（lokr 时为较大 Kronecker 因子的秩）。"),
    param("moss_factor", "LoKr 因子上限", "int", -1, "advanced", "网络", min=-1, max=1024,
          architectures=("moss_image",), native_key="factor",
          help="仅 lokr：较小 Kronecker 因子边长的上限；-1 表示近似均分。"),
    param("moss_down_init", "down 初始化", "enum", "kaiming", "advanced", "网络",
          options=enum(["kaiming", "weight_svd", "weight_svd_nodc"]), architectures=("moss_image",), native_key="down_init",
          help="lora 的 down 初始化：kaiming（引擎默认）或 weight_svd / weight_svd_nodc（以底模权重奇异向量初始化；nodc 会去掉校准 DC 方向）。"),
    param("moss_svd_slice", "SVD 切片", "int", 0, "advanced", "网络", min=0, max=1024,
          architectures=("moss_image",), native_key="svd_slice",
          help="lora + weight_svd：用第 k 段右奇异向量初始化 down（0 = 顶部 r 个）；不同切片互相正交，便于后续合并。"),
    param("moss_svd_slice_count", "SVD 切片交错数", "int", 0, "advanced", "网络", min=0, max=1024,
          architectures=("moss_image",), native_key="svd_slice_count",
          help="大于 0 时按交错方式切分奇异向量；需要大于 SVD 切片值。0 = 连续窗口。"),
    param("moss_svd_drop_loud", "SVD 去强通道数", "int", 0, "advanced", "网络", min=0, max=1024,
          architectures=("moss_image",), native_key="svd_drop_loud",
          help="lora + weight_svd：把校准后平均幅值最大的 k 个输入通道从 SVD 基中剔除，使其 down 列从零开始；0 = 关闭。"),
    param("moss_channel_stats", "通道统计文件", "string", "library/moss/calibration/channel_stats.safetensors",
          "advanced", "网络", path_kind="file", architectures=("moss_image",), native_key="channel_stats",
          help="SVD 去强通道 / channel_stats 校准文件；留空使用引擎自带路径（相对引擎目录）。"),
    param("moss_channel_mean", "通道均值文件", "string", "library/moss/calibration/channel_mean.safetensors",
          "advanced", "网络", path_kind="file", architectures=("moss_image",), native_key="channel_mean",
          help="weight_svd_nodc 使用的带符号 token 均值校准文件；留空使用引擎自带路径。"),
    param("moss_optimizer", "优化器", "enum", "muon", "advanced", "优化",
          options=enum(["muon", "adamw"]), architectures=("moss_image",), native_key="optimizer",
          help="引擎新默认 muon（Newton–Schulz 正交化更新）；lokr/loha 会自动使用 adamw。"),
    param("moss_ema_window", "EMA 窗口", "float", 0.1, "advanced", "优化", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="ema_window",
          help="adapter 权重的 EMA 跨度（占优化更新数的比例）；0 = 关闭。保存写 EMA，原始权重写到 <name>_raw。"),
    param("moss_ema_extra", "额外 EMA 窗口", "string", "", "advanced", "优化",
          architectures=("moss_image",), native_key="ema_extra",
          help="额外跟踪的 EMA 跨度，逗号分隔的比例（如 0.05,0.3）；每个都会保存为 <name>_ema<span>。"),
    param("moss_swap_h2d_only", "仅 H2D 块交换", "bool", True, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="swap_h2d_only",
          help="引擎默认开启：块交换只做 host→device 拷贝，CPU 保留页锁定副本，通常更快；"
               "部分 Windows/驱动会在 cudaHostRegister 失败，此时请关闭本项改用经典块交换。"),
    param("moss_region_weight", "i2i 编辑区权重", "float", 1.0, "advanced", "训练",
          min=0.0, max=100.0, architectures=("moss_image",), native_key="region_weight",
          help="i2i：编辑区域 token 的 loss 乘数；1 = 普通 MSE。"),
    param("moss_rest_weight", "i2i 非编辑区权重", "float", 1.0, "advanced", "训练",
          min=0.0, max=100.0, architectures=("moss_image",), native_key="rest_weight",
          help="i2i：编辑区域之外的 loss 乘数；1 = 普通 MSE。"),
    param("moss_region_threshold", "i2i 编辑区阈值", "float", 0.3, "advanced", "训练",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="region_threshold"),
    param("moss_region_min_tokens", "i2i 最小编辑 token", "int", 5, "advanced", "训练",
          min=0, max=100000, architectures=("moss_image",), native_key="region_min_tokens"),
    param("moss_region_dilate", "i2i 编辑区膨胀", "int", 1, "advanced", "训练",
          min=0, max=100, architectures=("moss_image",), native_key="region_dilate"),
    param("moss_tone_weight", "色调损失权重", "float", 0.0, "advanced", "训练",
          min=0.0, max=100.0, architectures=("moss_image",), native_key="tone_weight",
          help="i2i 非编辑 token 的 Lab 色调损失；0 = 关闭。"),
    param("moss_tone_log", "色调损失日志", "bool", False, "advanced", "训练",
          architectures=("moss_image",), native_key="tone_log"),
    param("moss_tone_pool", "色调池大小", "int", 8, "advanced", "训练",
          min=1, max=1024, architectures=("moss_image",), native_key="tone_pool"),
    param("moss_tone_dilate", "色调膨胀", "int", 2, "advanced", "训练",
          min=0, max=100, architectures=("moss_image",), native_key="tone_dilate"),
    param("moss_tone_dirs", "色调方向文件", "string", "library/moss/calibration/tone_dirs.safetensors",
          "advanced", "训练", path_kind="file", architectures=("moss_image",), native_key="tone_dirs"),
    param("moss_plain_caption_rate", "纯文本字幕比例", "float", 0.2, "advanced", "数据增强",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="plain_caption_rate",
          help="按该概率改用纯文本字幕而非 layout JSON；需要缓存阶段以 layout_json 生成（默认）。"),
    param("moss_caption_variant_rate", "字幕变体比例", "float", 0.0, "advanced", "数据增强",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="caption_variant_rate",
          help="使用 embeds_v{k} 缓存的概率（占非纯文本步数）；引擎预设为 0，需要彩色化探针缓存时才调高。"),
    param("moss_ref_timestep", "参考 token 时间步", "float", 0.0, "advanced", "网络",
          architectures=("moss_image",), native_key="ref_timestep",
          help="参考 token 的调制时间步；小于 0 表示跟随图片自身的 σ。引擎预设为 0。"),
    param("moss_ref_isolate", "参考 token 隔离", "bool", False, "advanced", "网络",
          architectures=("moss_image",), native_key="ref_isolate",
          help="参考 token 只在自己之间做注意力；引擎预设关闭。"),
    param("moss_repa_weight", "REPA 权重", "float", 0.0, "advanced", "网络",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="repa_weight",
          help="REPA 正则权重；0 = 关闭（引擎预设）。"),
    param("moss_repa_layer", "REPA 对齐层", "int", 11, "advanced", "网络", min=0, max=200,
          architectures=("moss_image",), native_key="repa_layer",
          help="REPA 使用第几个 block 的输出做 Gram 对齐；梯度只会到达该层及之前的 LoRA。"),
    param("moss_repa_pe_root", "PE-Spatial 根目录", "string", "post_image_dataset/lora", "advanced", "网络",
          path_kind="directory", architectures=("moss_image",), native_key="repa_pe_root",
          help="Anima PE-Spatial sidecar 的根目录（<root>/<source>/{stem}_anima_pe_spatial.safetensors）；启用 REPA 前需要先生成。"),
    param("moss_targets", "训练目标正则", "string", "", "advanced", "网络",
          architectures=("moss_image",), native_key="targets",
          help="线性层名匹配的正则；留空使用引擎默认的 q/k/v/o/ffn_up_gate/ffn_down。",
          detail=r"""作用：决定哪些线性层挂 LoRA —— 也就是「只训部分 DiT 层」的开关。留空时用引擎默认正则，
训练全部 42 个 block 各 6 个线性层（q / k / v / o / ffn_up_gate / ffn_down，共 252 层）。

取值：一条 Python 正则，引擎对「模块全名」做 fullmatch，例如 layers.17.proj.ffn_down

示例（直接抄进输入框）
  · 全部层的注意力：
    layers\.\d+\.proj\.(q|k|v|o)
  · 只训第 10–29 层（注意力 + FFN）：
    layers\.(1[0-9]|2[0-9])\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)
  · 只训最后 12 层（30–41）：
    layers\.(3[0-9]|4[01])\.proj\..*
  · 只训第 0–9 层与第 20 层：
    layers\.([0-9]|20)\.proj\..*
  · 只训 10–29 层的 FFN、且不要 down 投影（引擎没有排除项，只能正面写）：
    layers\.(1[0-9]|2[0-9])\.proj\.ffn_up_gate

注意
  · fullmatch：整条模块名必须被完整匹配；写错时引擎在起跑阶段报 no nn.Linear matched
  · 没有 exclude / 排除列表，也没有「最后 N 层」「前 50% 深度」这类相对写法，索引要写死
  · block 索引是 0–41，写 42 及以上匹配不到任何层
  · 只选部分层只减少可训参数与梯度，不省激活显存；省显存请用「交换块数量」
  · rank / alpha 是全网络统一值，不能按层不同
  · 训练时这条正则会写进 LoRA 的 metadata，推理 / 合并按它重建，所以子集 LoRA 能正常加载
  · LP 会把它原样作为 --targets 传给引擎的 scripts/moss/train.py"""),
    # ---- advanced / 时间步与采样（噪声分布，引擎默认值）----
    param("moss_sigma_mu", "噪声分布均值 μ", "float", -1.09, "advanced", "时间步与采样", min=-10.0, max=10.0,
          architectures=("moss_image",), native_key="sigma_mu",
          help="Z ~ N(μ, s²)，σ = 1 − sigmoid(Z)；μ 越负 → σ 越高（结构/构图/画风），"
               "μ 越正 → σ 越低（细节/线宽）。默认 -1.09 是预训练最后一期的取值。"),
    param("moss_sigma_s", "噪声分布标准差 s", "float", 0.95, "advanced", "时间步与采样", min=0.01, max=10.0,
          architectures=("moss_image",), native_key="sigma_s", help="Z 的标准差。"),
    param("moss_sigma_cutoff", "σ 截断阈值", "float", 0.95, "advanced", "时间步与采样", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="sigma_cutoff",
          help="截断 σ 后重映射到 1/1000 网格；调低会把剩余 σ 整体抬高（偏结构/高噪），不是细节旋钮；"
               "1.0 表示不截断。"),
    param("moss_sigma_uniform", "离散均匀混合权重", "float", 0.01, "advanced", "时间步与采样", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="sigma_uniform",
          help="离散均匀 σ 采样的混合比例；提高会增加全频带/低噪覆盖，默认 0.01。"),
    param("moss_alpha", "LoRA Alpha", "float", None, "basic", "网络", min=0.0, max=1024.0,
          architectures=("moss_image",), native_key="alpha",
          help="留空时按引擎默认：lora 用 128（rank 32 时缩放 4），lokr/loha 用 rank。"),
    param("moss_lora_dtype", "LoRA 保存精度", "enum", "bf16", "basic", "网络",
          options=enum(["bf16", "fp16", "fp32"]), architectures=("moss_image",), native_key="lora_dtype",
          help="训练时用 fp32 主权重、rank 矩阵按 bf16 计算；该选项只决定保存精度。"),

    # ---- basic / 优化 ----
    param("moss_lr", "学习率", "float", 0.00005, "basic", "优化", min=0.00000001, max=1.0,
          architectures=("moss_image",), native_key="lr",
          help="引擎默认 5e-5（配合默认开启的 DC 隔离与通道缩放；1e-4 在 DC 隔离下会过拟合）。"),
    param("moss_warmup_ratio", "预热比例", "float", 0.05, "basic", "优化", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="warmup_ratio",
          help="线性预热占总步数的比例。"),
    param("moss_max_grad_norm", "梯度裁剪", "float", 1.0, "basic", "优化", min=0.0, max=100.0,
          architectures=("moss_image",), native_key="max_grad_norm"),

    # ---- advanced / 保存 ----
    param("moss_save_every_epochs", "保存间隔（轮）", "int", 0, "advanced", "保存", min=0, max=10000,
          architectures=("moss_image",), native_key="save_every_epochs",
          help="0 表示只在结束时保存最终 LoRA。"),

    # ---- advanced / 内存与加速 ----
    param("moss_blocks_to_swap", "交换块数量", "int", 36, "advanced", "内存与加速", min=-1, max=200,
          architectures=("moss_image",), native_key="blocks_to_swap",
          help="-1 表示按空闲显存自动决定；引擎默认 36（rank 32、16 GB 卡、0.5 激活预算）。换块越多越吃 PCIe 时间。"),
    param("moss_activation_memory_budget", "激活显存预算", "float", 0.5, "advanced", "内存与加速",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="activation_memory_budget",
          help="引擎默认 0.5；仅在启用 compile 且关闭梯度检查点时生效。"),
    param("moss_grad_checkpointing", "梯度检查点", "bool", False, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="grad_checkpointing",
          help="引擎默认关闭（激活预算 + 块交换更快）；启用时建议配合关闭 compile 或把交换块数设为 0。"),
    param("moss_compile", "torch.compile", "bool", False, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="compile",
          help="引擎默认开启，但 Windows 上需要已安装 Visual Studio 的 C++ 生成工具（cl.exe）；未安装时会以 InvalidCxxCompiler 失败，因此这里默认关闭。装好 MSVC 后可开启以提速。"),
    param("moss_fit_report", "峰值显存报告", "bool", False, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="fit_report",
          help="用最长字幕样本跑第 1 步并报告峰值显存与交换块建议；关闭时 epoch 1 正常打乱。"),
    param("moss_expandable_segments", "可扩展显存段", "bool", True, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="expandable_segments",
          help="在 CUDA 初始化前设置 expandable_segments；长字幕/多分辨率缓存可缓解显存碎片。"),

    # ---- advanced / 日志 ----
    param("moss_log_every", "日志间隔（步）", "int", 10, "advanced", "日志", min=1, max=100000,
          architectures=("moss_image",), native_key="log_every"),

    # ---- advanced / 数据增强 ----
    param("moss_caption_dropout", "字幕丢弃概率", "float", 0.0, "advanced", "数据增强", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="caption_dropout",
          help="大于 0 时按该概率用空提示词训练（CFG 的 uncond），会在缓存目录写入 _empty 编码。"),
)

# Parameters the engine exposes but the original form did not surface. Defaults are
# intentionally None/False where the engine has a CLI default: apply_engine_defaults
# overlays the current config chain, and anything still unset is omitted from the
# generated config so the engine's own default applies.
EXTRA_ANIMA_PARAMS: tuple[Param, ...] = (
    # ---- advanced / 数据集 ----
    param("sample_ratio", "采样比例", "float", 1.0, "advanced", "数据集", min=0.0, max=1.0,
          help="全局覆盖每个子集的采样比例；1.0 表示使用各子集自己的值。"),
    param("path_pattern", "路径过滤", "string", "*", "advanced", "数据集",
          help="按图片相对路径过滤，支持 | 分隔多个模式；留空使用 *。"),
    param("artists_shard", "画师分片", "string", "", "advanced", "数据集",
          help="只训练画师子目录的某个分片，格式 k_N（如 1_4）；与路径过滤互斥。"),
    param("artist_filter", "画师过滤", "string", "", "advanced", "数据集",
          help="只训练字幕含 @画师 标签的图片；输出会写入 ckpt-artist 目录。"),
    param("resize_interpolation", "缩放插值", "enum", None, "advanced", "数据集",
          options=enum(["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"])),
    param("vae_batch_size", "VAE 缓存批大小", "int", None, "advanced", "数据集", min=1, max=256),
    param("text_encoder_batch_size", "TE 缓存批大小", "int", None, "advanced", "数据集", min=1, max=256),
    param("skip_cache_check", "跳过缓存校验", "bool", None, "advanced", "数据集",
          help="跳过缓存内容校验以加快启动；缓存损坏时可能直接训练失败。"),
    param("vae_disable_cache", "禁用 VAE 缓存", "bool", None, "advanced", "内存与加速",
          help="关闭 VAE 内部缓存；显存更低但更慢，且与官方行为不同。"),
    param("masked_loss", "启用掩码损失", "bool", False, "advanced", "数据增强",
          help="读取 mask_dir 下的掩码并只计算掩码区域损失。"),
    param("mask_dir", "掩码目录", "string", "", "advanced", "数据增强", path_kind="directory"),
    param("prior_loss_weight", "正则图损失权重", "float", None, "advanced", "训练", min=0.0, max=100.0),
    param("caption_dropout_every_n_epochs", "定期清空字幕", "int", None, "advanced", "数据增强", min=1, max=10000),
    param("caption_tag_dropout_rate", "标签丢弃率", "float", None, "advanced", "数据增强", min=0.0, max=1.0),
    param("weighted_captions", "加权字幕", "bool", False, "advanced", "数据增强",
          help="启用 (token:1.3) 形式的字幕权重语法。"),
    param("use_shuffled_caption_variants_only", "仅使用打乱变体", "bool", False, "advanced", "数据增强",
          help="只从打乱/去标签变体中取样，不训练原始字幕；需要缓存包含变体。"),
    param("use_randomized_caption_variants", "随机化字幕变体", "bool", False, "advanced", "数据增强"),
    param("use_randomized_caption_variants_only", "仅使用随机化变体", "bool", False, "advanced", "数据增强"),
    # ---- advanced / 训练与损失 ----
    param("save_n_epoch_ratio", "按比例保存", "int", None, "advanced", "保存", min=1, max=100),
    param("persistent_data_loader_workers", "持久化 DataLoader", "bool", True, "advanced", "训练"),
    param("dataloader_pin_memory", "锁定 DataLoader 内存", "bool", True, "advanced", "训练"),
    param("dataloader_prefetch_factor", "DataLoader 预取", "int", None, "advanced", "训练", min=1, max=64),
    param("ip_noise_gamma", "输入扰动噪声", "float", None, "advanced", "训练", min=0.0, max=1.0),
    param("ip_noise_gamma_random_strength", "随机扰动强度", "bool", False, "advanced", "训练"),
    param("t_min", "最小 σ", "float", None, "advanced", "时间步与采样", min=0.0, max=1.0),
    param("t_max", "最大 σ", "float", None, "advanced", "时间步与采样", min=0.0, max=1.0),
    param("deterministic", "确定性模式", "bool", False, "advanced", "训练",
          help="确定性反向/算子与 cuDNN，便于成对 A/B；速度略慢。"),
    param("paired_step_rng", "配对随机数", "bool", False, "advanced", "时间步与采样"),
    param("loss_type", "损失函数", "enum", "l2", "advanced", "训练",
          options=enum(["l1", "l2", "huber", "smooth_l1", "pseudo_huber"])),
    param("huber_schedule", "Huber 调度", "enum", "snr", "advanced", "训练",
          options=enum(["constant", "exponential", "snr"])),
    param("huber_c", "Huber 衰减", "float", None, "advanced", "训练", min=0.0, max=100.0),
    param("huber_scale", "Huber 缩放", "float", None, "advanced", "训练", min=0.0, max=100.0),
    param("pseudo_huber_c", "Pseudo-Huber c", "float", None, "advanced", "训练", min=0.0, max=100.0),
    param("multiscale_loss_weight", "多尺度损失权重", "float", None, "advanced", "训练", min=0.0, max=100.0),
    param("debiased_estimation_loss", "去偏估计损失", "bool", False, "advanced", "训练"),
    # ---- advanced / 优化 ----
    param("mode_scale", "Mode 权重缩放", "float", None, "advanced", "时间步与采样", min=0.0, max=100.0),
    # Group learning rates: declared and parsed by the engine, but
    # get_anima_param_groups() has no caller in this commit, so they never reach the
    # optimizer. Marked unavailable instead of silently doing nothing.
    param("llm_adapter_lr", "LLM Adapter 学习率", "float", None, "advanced", "优化", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_GROUP_LR),
    param("self_attn_lr", "自注意力学习率", "float", None, "advanced", "优化", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_GROUP_LR),
    param("cross_attn_lr", "交叉注意力学习率", "float", None, "advanced", "优化", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_GROUP_LR),
    param("mlp_lr", "MLP 学习率", "float", None, "advanced", "优化", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_GROUP_LR),
    param("mod_lr", "AdaLN 学习率", "float", None, "advanced", "优化", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_GROUP_LR),
    # EMA: the flags exist, the implementation does not (make_ema is never called and
    # train.py never reads args.ema*). Marked unavailable rather than pretending.
    param("ema", "启用 EMA", "bool", False, "advanced", "保存",
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_decay", "EMA 衰减", "float", None, "advanced", "保存", min=0.0, max=1.0,
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_device", "EMA 设备", "enum", "gpu", "advanced", "保存", options=enum(["cpu", "gpu"]),
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_use_num_updates", "EMA 更新数", "bool", False, "advanced", "保存",
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_use_feedback", "EMA 反馈", "bool", False, "advanced", "保存",
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_param_multiplier", "EMA 参数倍数", "float", None, "advanced", "保存", min=0.0, max=10.0,
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    param("ema_resume_path", "EMA 续训路径", "string", "", "advanced", "保存", path_kind="file",
          native=False, unsupported_reason=_UNAVAILABLE_EMA),
    # ---- advanced / 网络与 tokenizer ----
    param("network_train_unet_only", "仅训练 U-Net", "bool", True, "advanced", "网络"),
    param("network_train_text_encoder_only", "仅训练文本编码器", "bool", None, "advanced", "网络"),
    param("dim_from_weights", "从权重推断 Rank", "bool", False, "advanced", "网络"),
    param("scale_weight_norms", "权重范数缩放", "float", None, "advanced", "网络", min=0.0, max=100.0),
    param("attn_softmax_scale", "注意力 Softmax 缩放", "float", None, "advanced", "网络", min=0.0, max=1.0),
    param("pooled_text_proj", "pooled_text_proj 路径", "string", "", "advanced", "模型与编码器", path_kind="file"),
    param("qwen3_max_token_length", "Qwen3 最大 token", "int", None, "advanced", "模型与编码器", min=1, max=65536),
    param("t5_max_token_length", "T5 最大 token", "int", None, "advanced", "模型与编码器", min=1, max=65536),
    # ---- advanced / 元数据 ----
    # Only title / author / description / license / tags are read by
    # library/train_util.py; these four argparse flags have no consumer, so they are
    # marked unavailable instead of accepting values that are then dropped.
    param("metadata_usage_hint", "使用提示", "string", "", "advanced", "元数据",
          native=False, unsupported_reason=_UNAVAILABLE_METADATA),
    param("metadata_thumbnail", "缩略图", "string", "", "advanced", "元数据",
          native=False, unsupported_reason=_UNAVAILABLE_METADATA),
    param("metadata_merged_from", "合并来源", "string", "", "advanced", "元数据",
          native=False, unsupported_reason=_UNAVAILABLE_METADATA),
    param("metadata_trigger_phrase", "触发词", "string", "", "advanced", "元数据",
          native=False, unsupported_reason=_UNAVAILABLE_METADATA),
    param("training_comment", "训练备注", "string", "", "advanced", "元数据"),
    # ---- advanced / compile ----
    param("dynamo_backend", "compile 后端", "enum", None, "advanced", "内存与加速",
          options=enum(["eager", "inductor"])),
    param("partitioner_recompute_views", "重算 view 算子", "bool", None, "advanced", "内存与加速"),
    param("partitioner_aggressive_recomputation", "激进重算", "bool", None, "advanced", "内存与加速"),
)

# Every non-MOSS parameter here describes the Anima line and only Anima accepts those
# flags; the MOSS line has its own set (MOSS_PARAMS). Leaving them unscoped made both
# lists render together, with duplicated step / epoch / seed fields in one section.
# extra_params stays shared because it is meaningful for either line.
MOSS_KEYS = {p.key for p in MOSS_PARAMS}
PARAMS = tuple(
    __import__("dataclasses").replace(
        p,
        help=_PARAM_HELP.get(p.key, p.help),
        architectures=p.architectures if (p.architectures or p.key in MOSS_KEYS or p.key == "extra_params") else ("anima",),
    )
    for p in PARAMS + EXTRA_ANIMA_PARAMS + MOSS_PARAMS
)
