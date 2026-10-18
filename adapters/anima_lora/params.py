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

from adapters.capabilities import ArchSpec, Param, enum, param

ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("anima", "Anima", "train.py"),
    ArchSpec("moss_image", "MOSS-Image（alpha）", "scripts/moss/train.py"),
)

# Values --optimizer_type dispatches in library/training/optimizers.py; the shipped
# configs/base.toml default is AdamW (channel scaling relies on Adam's step size).
OPTIMIZERS = enum([
    "AdamW", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit",
    "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit",
    "Prodigy", "DAdaptAdam", "DAdaptAdan", "DAdaptLion", "DAdaptSGD",
    "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree",
])

LR_SCHEDULERS = enum([
    "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
    "polynomial", "inverse_sqrt", "adafactor", "rex",
])

ATTENTION_MODES = enum(["", "torch", "flash", "flex", "sdpa"])

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
    "learning_rate": "引擎默认 2.0e-6；应结合底模与数据量调整。",
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
    "resolution_width": "训练宽度；本引擎的数据集子集没有分辨率键，该值用于预处理/目标分辨率，不会写进数据集配置。",
    "resolution_height": "训练高度；本引擎的数据集子集没有分辨率键，该值用于预处理/目标分辨率，不会写进数据集配置。",
    "gradient_checkpointing": "以重算换取较低显存占用，通常会降低速度；引擎只把低显存预设默认打开它。",
    "sample_prompts": "可选；每行一个提示词，配合采样间隔使用；本引擎通过 --sample_prompts 读取，训练与采样格式需一致。",
    "progress_jsonl": "结构化进度输出，便于持续读取步数、损失与速度；留空则沿用引擎的文本日志。",
    "extra_params": "补充 LP 未提供的引擎参数，会合并进引擎配置文件；空格或换行分隔，含空格的值用引号，单独的 --开关 表示开启；与上面已提供的参数重复会被拒绝。",
}

PARAMS: tuple[Param, ...] = (
    # ---- basic / 训练 ----
    param("max_train_steps", "训练步数", "int", 1600, min=1, max=10000000, section="训练"),
    param("max_train_epochs", "训练轮数", "int", None, min=1, max=100000, section="训练"),
    param("seed", "随机种子", "int", None, min=0, max=4294967295, section="训练"),
    param("gradient_checkpointing", "梯度检查点", "bool", False, section="训练"),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024, section="训练"),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 0, min=0, max=64, section="训练"),
    param("mixed_precision", "混合精度", "enum", "bf16", options=enum(["no", "fp16", "bf16"]), section="训练"),
    param("save_precision", "保存精度", "string", "bf16", section="训练"),

    # ---- basic / 数据集（写入数据集配置或影响预处理）----
    param("resolution_width", "训练宽度", "int", 1024, min=64, max=4096, step=16, native=False, section="数据集"),
    param("resolution_height", "训练高度", "int", 1024, min=64, max=4096, step=16, native=False, section="数据集"),
    param("batch_size", "Batch Size", "int", 1, min=1, max=128, native=False, section="数据集"),
    param("caption_extension", "Caption 扩展名", "enum", ".txt", native=False,
          options=enum([".txt", ".caption"]), section="数据集"),
    param("caption_separator", "标签分隔符", "enum", ",", native=False,
          options=enum([",", ";", "|"]), section="数据集"),
    param("num_repeats", "图片重复次数", "int", 1, min=1, max=1000, native=False, section="数据集"),
    param("recursive", "递归读取子目录", "bool", True, native=False, section="数据集"),
    param("use_vae_cache", "缓存 VAE 潜空间", "bool", False, section="数据集"),
    param("use_text_cache", "缓存文本编码器输出", "bool", False, section="数据集"),

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
          display_order=30),
    param("t5_tokenizer_path", "T5 分词器路径", "string", "", architectures=("anima",), section="模型与编码器",
          display_order=40),
    param("vocab_pack", "CJK 词表包路径", "string", "", native_key="vocab_pack", architectures=("anima",),
          section="模型与编码器", display_order=50),

    # ---- basic / 网络 ----
    param("network_module", "网络模块", "string", "networks.lora_anima", architectures=("anima",), section="网络"),
    param("network_dim", "LoRA Rank", "int", 16, min=1, max=1024, architectures=("anima",), section="网络"),
    param("network_alpha", "LoRA Alpha", "float", 1.0, min=0.0, max=1024.0, architectures=("anima",), section="网络"),
    param("network_dropout", "网络 Dropout", "float", None, min=0.0, max=1.0, architectures=("anima",), section="网络"),
    param("network_weights", "初始权重路径", "string", "", native_key="network_weights",
          architectures=("anima",), section="网络"),
    param("network_args", "网络附加参数", "string", "", native_key="network_args", list_arg=True, multiline=True,
          architectures=("anima",), section="网络"),

    # ---- basic / 优化 ----
    param("optimizer_type", "优化器", "enum", "AdamW", options=OPTIMIZERS, architectures=("anima",), section="优化"),
    param("optimizer_args", "优化器附加参数", "string", "", native_key="optimizer_args", list_arg=True,
          multiline=True, architectures=("anima",), section="优化"),
    param("learning_rate", "学习率", "float", 0.000002, min=0.00000001, max=1.0, architectures=("anima",),
          section="优化"),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0.0, max=100.0, architectures=("anima",), section="优化"),
    param("lr_scheduler", "学习率策略", "enum", "constant", options=LR_SCHEDULERS, architectures=("anima",),
          section="优化"),
    param("lr_scheduler_type", "自定义调度器模块", "string", "", architectures=("anima",), section="优化"),
    param("lr_scheduler_args", "调度器附加参数", "string", "", native_key="lr_scheduler_args", list_arg=True,
          multiline=True, architectures=("anima",), section="优化"),
    param("lr_warmup_steps", "预热步数", "int", 0, min=0, max=1000000, architectures=("anima",), section="优化"),
    param("lr_scheduler_num_cycles", "重启次数", "int", 1, min=1, max=100, architectures=("anima",), section="优化"),

    # ---- advanced / 保存 ----
    param("save_every_n_epochs", "保存间隔（轮）", "int", None, "advanced", "保存",
          architectures=("anima",), min=1, max=10000),
    param("save_every_n_steps", "保存间隔（步）", "int", None, "advanced", "保存",
          architectures=("anima",), min=1, max=10000000),
    param("save_state", "保存训练状态", "bool", False, "advanced", "保存", architectures=("anima",)),
    param("save_state_on_train_end", "训练结束保存状态", "bool", False, "advanced", "保存",
          architectures=("anima",)),
    param("checkpointing_epochs", "自动续训间隔（轮）", "int", None, "advanced", "保存",
          architectures=("anima",), min=1, max=10000),
    param("resume", "续训状态路径", "string", "", "advanced", "保存", native_key="resume",
          path_kind="directory", architectures=("anima",)),

    # ---- advanced / 内存与加速 ----
    param("blocks_to_swap", "交换块数量", "int", None, "advanced", "内存与加速", min=0, max=200),
    param("unsloth_offload_checkpointing", "卸载检查点到 CPU", "bool", False, "advanced", "内存与加速"),
    param("highvram", "高显存模式", "bool", False, "advanced", "内存与加速"),
    param("lowram", "低内存模式", "bool", False, "advanced", "内存与加速"),
    param("attn_mode", "注意力实现", "enum", "", "advanced", "内存与加速", options=ATTENTION_MODES),
    param("torch_compile", "torch.compile", "bool", False, "advanced", "内存与加速"),
    param("activation_memory_budget", "激活显存预算", "float", None, "advanced", "内存与加速", min=0.0, max=1.0,
          help=_PARAM_HELP["activation_memory_budget"]),
    param("vae_chunk_size", "VAE 分块大小", "int", None, "advanced", "内存与加速", min=1, max=512),
    param("no_half_vae", "VAE 不使用半精度", "bool", False, "advanced", "内存与加速"),

    # ---- advanced / 日志 ----
    param("logging_dir", "日志目录", "string", "", "advanced", "日志", path_kind="directory"),
    param("log_with", "日志后端", "enum", "", "advanced", "日志", options=enum(["", "tensorboard", "wandb", "all"])),
    param("log_every_n_steps", "日志间隔（步）", "int", None, "advanced", "日志", min=1, max=1000000),
    param("progress_jsonl", "结构化进度文件", "string", "", "advanced", "日志"),

    # ---- advanced / 时间步与采样 ----
    param("timestep_sampling", "时间步采样", "enum", "sigma", "advanced", "时间步与采样",
          options=enum(["sigma", "uniform", "sigmoid", "shift", "flux_shift", "logsnr"])),
    param("discrete_flow_shift", "离散流偏移", "float", None, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("sigmoid_scale", "Sigmoid 缩放", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("weighting_scheme", "加权方案", "enum", "none", "advanced", "时间步与采样",
          options=enum(["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"])),
    param("logit_mean", "Logit 均值", "float", 0.0, "advanced", "时间步与采样", min=-10.0, max=10.0),
    param("logit_std", "Logit 标准差", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=10.0),
    param("min_snr_gamma", "Min-SNR Gamma", "float", None, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("sample_every_n_epochs", "采样间隔（轮）", "int", None, "preview", "时间步与采样", min=1, max=10000),
    param("sample_every_n_steps", "采样间隔（步）", "int", None, "preview", "时间步与采样", min=1, max=10000000),
    param("sample_at_first", "训练前采样", "bool", False, "preview", "时间步与采样"),
    param("sample_prompts", "采样提示词", "string", "", "preview", "时间步与采样"),

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

    # ---- basic / 模型 ----
    param("moss_model_dir", "MOSS-Image-Base 路径", "string", "", "basic", "模型与编码器", path_kind="directory",
          native=False, architectures=("moss_image",), display_order=0,
          help="选择 MOSS-Image-Base 这个 diffusers 文件夹本身（内含 transformer、text_encoder、vae、tokenizer、scheduler）；提交时自动取其父目录作为引擎的 model_dir，并固定使用 base 变体。"),

    # ---- basic / 训练 ----
    param("moss_epochs", "训练轮数", "int", 8, "basic", "训练", min=1, max=10000,
          architectures=("moss_image",), native_key="epochs"),
    param("moss_max_steps", "训练步数上限", "int", 0, "basic", "训练", min=0, max=10000000,
          architectures=("moss_image",), native_key="max_steps",
          help="大于 0 时提前结束，覆盖训练轮数；0 表示由轮数决定。"),
    param("moss_seed", "随机种子", "int", 0, "basic", "训练", min=0, max=4294967295,
          architectures=("moss_image",), native_key="seed"),

    # ---- basic / 网络（LoRA）----
    param("moss_rank", "LoRA Rank", "int", 16, "basic", "网络", min=1, max=1024,
          architectures=("moss_image",), native_key="rank"),
    param("moss_alpha", "LoRA Alpha", "float", None, "basic", "网络", min=0.0, max=1024.0,
          architectures=("moss_image",), native_key="alpha", help="留空时引擎按 rank 取值。"),
    param("moss_lora_dtype", "LoRA 保存精度", "enum", "bf16", "basic", "网络",
          options=enum(["bf16", "fp16", "fp32"]), architectures=("moss_image",), native_key="lora_dtype",
          help="训练时用 fp32 主权重、rank 矩阵按 bf16 计算；该选项只决定保存精度。"),

    # ---- basic / 优化 ----
    param("moss_lr", "学习率", "float", 0.0001, "basic", "优化", min=0.00000001, max=1.0,
          architectures=("moss_image",), native_key="lr", help="AdamW 学习率。"),
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
    param("moss_blocks_to_swap", "交换块数量", "int", 24, "advanced", "内存与加速", min=-1, max=200,
          architectures=("moss_image",), native_key="blocks_to_swap",
          help="-1 表示按空闲显存自动决定；官方默认 24，具体最优值随显卡与 PCIe 带宽变化。"),
    param("moss_activation_memory_budget", "激活显存预算", "float", 0.2, "advanced", "内存与加速",
          min=0.0, max=1.0, architectures=("moss_image",), native_key="activation_memory_budget",
          help="仅在启用 compile 且关闭梯度检查点时生效。"),
    param("moss_grad_checkpointing", "梯度检查点", "bool", False, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="grad_checkpointing",
          help="引擎默认关闭（激活预算 + 块交换更快）；启用时建议配合关闭 compile 或把交换块数设为 0。"),
    param("moss_compile", "torch.compile", "bool", False, "advanced", "内存与加速",
          architectures=("moss_image",), native_key="compile",
          help="引擎默认开启，但 Windows 上需要已安装 Visual Studio 的 C++ 生成工具（cl.exe）；未安装时会以 InvalidCxxCompiler 失败，因此这里默认关闭。装好 MSVC 后可开启以提速。"),

    # ---- advanced / 日志 ----
    param("moss_log_every", "日志间隔（步）", "int", 10, "advanced", "日志", min=1, max=100000,
          architectures=("moss_image",), native_key="log_every"),

    # ---- advanced / 数据增强 ----
    param("moss_caption_dropout", "字幕丢弃概率", "float", 0.0, "advanced", "数据增强", min=0.0, max=1.0,
          architectures=("moss_image",), native_key="caption_dropout",
          help="大于 0 时按该概率用空提示词训练（CFG 的 uncond），会在缓存目录写入 _empty 编码。"),
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
    for p in PARAMS + MOSS_PARAMS
)
