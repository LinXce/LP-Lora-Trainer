"""Kohya sd-scripts LoRA parameter surface.

Source of truth: ``engine/kohya_ss/kohya_gui/class_*.py`` argument definitions,
``engine/kohya_ss/config example.toml`` and ``engine/kohya_ss/docs/LoRA/options.md``.
The ``sd-scripts`` submodule itself is not vendored, so keys mirror the sd-scripts
CLI/config names exactly.
"""

from dataclasses import replace

from adapters.capabilities import ArchSpec, Param, enum, param

ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("sd1", "SD 1.5", "train_network.py"),
    ArchSpec("sd2", "SD 2.x", "train_network.py"),
    ArchSpec("sdxl", "SDXL", "sdxl_train_network.py"),
    ArchSpec("sd3", "SD 3 / 3.5", "sd3_train_network.py"),
    ArchSpec("flux1", "FLUX.1", "flux_train_network.py"),
    ArchSpec("hunyuan_image", "HunyuanImage", "hunyuan_image_train_network.py"),
    ArchSpec("lumina", "Lumina Image", "lumina_train_network.py"),
    ArchSpec("anima", "Anima", "anima_train_network.py"),
)

# LoRA module per architecture; the adapter selects it, so it is displayed but not editable.
NETWORK_MODULES = {
    "sd1": "networks.lora",
    "sd2": "networks.lora",
    "sdxl": "networks.lora",
    "sd3": "networks.lora_sd3",
    "flux1": "networks.lora_flux",
    "hunyuan_image": "networks.lora_hunyuan_image",
    "lumina": "networks.lora_lumina",
    "anima": "networks.lora_anima",
}

PARAMS: tuple[Param, ...] = (
    # ---- basic / 训练 ----
    param("max_train_steps", "训练步数", "int", 1000, min=1, max=10000000, section='训练'),
    param("max_train_epochs", "训练轮数", "int", None, min=1, max=100000, help="设置后覆盖训练步数", section='训练'),
    param("train_batch_size", "Batch Size", "int", 1, min=1, max=128, section='训练'),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024, section='训练'),
    param("seed", "随机种子", "int", 42, min=0, max=4294967295, section='训练'),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0, max=100, section='训练'),
    param("gradient_checkpointing", "梯度检查点", "bool", True, help="降低显存占用，训练速度略降", section='训练'),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 0, min=0, max=32, section='训练'),
    param("persistent_data_loader_workers", "持久化 DataLoader", "bool", False, section='训练'),

    # ---- basic / 优化器 ----
    param("learning_rate", "学习率", "float", 0.0001, min=0.00000001, max=1.0, section='优化器'),
    param("text_encoder_lr", "Text Encoder 学习率", "float", 0.0001, min=0.0, max=1.0, section='优化器'),
    param("optimizer_type", "优化器", "enum", "AdamW8bit", options=enum([
        "AdamW", "AdamW8bit", "Adafactor", "AdamWScheduleFree", "Lion", "Lion8bit",
        "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptLion", "DAdaptSGD",
        "Prodigy", "pytorch_optimizer.CAME", "RAdamScheduleFree", "SGDNesterov",
    ]), section='优化器'),
    param("optimizer_args", "优化器附加参数", "string", "", section='优化器'),
    param("lr_scheduler", "学习率策略", "enum", "constant", options=enum([
        "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
        "polynomial", "piecewise_constant", "inverse_sqrt", "adafactor", "rex",
    ]), section='优化器'),
    param("lr_warmup_steps", "预热步数", "int", 0, min=0, max=1000000, section='优化器'),
    param("lr_scheduler_num_cycles", "重启次数", "int", 1, min=1, max=100, section='优化器'),
    param("lr_scheduler_power", "多项式幂", "float", 1.0, min=0.0, max=10.0, section='优化器'),
    param("lr_scheduler_type", "自定义调度器模块", "string", "", section='优化器'),
    param("lr_scheduler_args", "调度器附加参数", "string", "", section='优化器'),

    # ---- basic / 网络 ----
    param("network_dim", "LoRA Rank", "int", 16, min=1, max=1024, section='网络'),
    param("network_alpha", "LoRA Alpha", "int", 16, min=1, max=1024, section='网络'),
    param("network_dropout", "网络 Dropout", "float", None, min=0.0, max=1.0, section='网络'),
    param("network_weights", "初始权重路径", "string", "", section='网络'),
    param("dim_from_weights", "从权重推断 Rank", "bool", False, section='网络'),
    param("scale_weight_norms", "权重缩放系数", "float", None, min=0.0, max=100.0, section='网络'),
    param("network_args", "网络附加参数", "string", "", help="如 \"conv_dim=8 conv_alpha=8\"", section='网络'),
    param("network_train_unet_only", "仅训练 UNet", "bool", False, section='网络'),
    param("network_train_text_encoder_only", "仅训练 Text Encoder", "bool", False, section='网络'),

    # ---- basic / 数据集 ----
    param("resolution", "训练分辨率", "int", 512, min=64, max=4096, step=64, native=False, section='数据集'),
    param("enable_bucket", "启用分桶", "bool", True, native=False, section='数据集'),
    param("min_bucket_reso", "最小分桶分辨率", "int", 256, min=64, max=4096, section='数据集'),
    param("max_bucket_reso", "最大分桶分辨率", "int", 2048, min=64, max=8192, section='数据集'),
    param("bucket_reso_steps", "分桶步长", "int", 64, min=1, max=512, section='数据集'),
    param("bucket_no_upscale", "分桶不放大", "bool", True, section='数据集'),
    param("caption_extension", "Caption 扩展名", "enum", ".txt", native=False,
          options=enum([".txt", ".caption", ".cap"]), section='数据集'),
    param("num_repeats", "图片重复次数", "int", 1, min=1, max=1000, native=False, section='数据集'),
    param("keep_tokens", "保留 Token 数", "int", 0, min=0, max=100, section='数据集'),
    param("max_token_length", "最大 Token 长度", "int", 150, min=75, max=225, step=75, section='数据集'),
    param("shuffle_caption", "打乱 Caption", "bool", False, section='数据集'),
    param("caption_dropout_rate", "Caption Dropout 比例", "float", 0.0, min=0.0, max=1.0, section='数据集'),
    param("caption_dropout_every_n_epochs", "每 N 轮 Caption Dropout", "int", 0, min=0, max=100, section='数据集'),
    param("weighted_captions", "加权 Caption", "bool", False, section='数据集'),
    param("flip_aug", "水平翻转增强", "bool", False, section='数据集'),
    param("color_aug", "颜色增强", "bool", False, section='数据集'),
    param("random_crop", "随机裁剪", "bool", False, section='数据集'),
    param("masked_loss", "掩码损失", "bool", False, section='数据集'),
    # 1, not 0: the current DiT latent caches use this value as the worker-pool cap,
    # and 0 makes ThreadPoolExecutor raise "max_workers must be greater than 0".
    param("vae_batch_size", "VAE Batch Size", "int", 1, min=0, max=128, section='数据集',
          help="VAE 批处理大小（1 表示逐张处理）。设为 0 时部分底模的 latent 缓存会直接报错，因此默认 1。"),

    # ---- advanced / 保存与采样 ----
    # 0 would make the engine divide by zero ((epoch + 1) % 0, global_step % 0);
    # "off" means leaving the key out, which is what a blank field does.
    param("save_every_n_epochs", "保存间隔（轮）", "int", 1, "advanced", "保存", min=1, max=10000),
    param("save_every_n_steps", "保存间隔（步）", "int", 200, "advanced", "保存", min=1, max=10000000),
    # Left unset by default: the engine only skips its cleanup when the key is absent.
    # 0 is not "off" — it keeps just the newest checkpoint, which deleted the rest.
    param("save_last_n_steps", "保留最近 N 步", "int", None, "advanced", "保存", min=0, max=10000000,
          help="留空保留全部存档（默认）；填 N 只保留最近 N 步内的存档；填 0 只保留最新一份。"),
    param("save_last_n_steps_state", "保留最近 N 步状态", "int", None, "advanced", "保存", min=0, max=10000000,
          help="对训练状态存档同样生效；留空保留全部，填 0 只保留最新一份。"),
    param("save_state", "保存训练状态", "bool", False, "advanced", "保存"),
    param("save_state_on_train_end", "训练结束保存状态", "bool", False, "advanced", "保存"),
    param("save_model_as", "保存格式", "enum", "safetensors", "advanced", "保存",
          options=enum(["safetensors", "ckpt", "diffusers", "diffusers_safetensors"])),
    param("training_comment", "训练备注", "string", "", "advanced", "保存"),
    # 0 不是"关闭"：引擎在轮末做 epoch % 0（这里没有 None 判断），会直接崩溃。
    # 关闭采样必须不写这两个键，因此默认留空、最小值 1。
    param("sample_every_n_steps", "采样间隔（步）", "int", None, "preview", "采样", min=1, max=10000000),
    param("sample_every_n_epochs", "采样间隔（轮）", "int", None, "preview", "采样", min=1, max=10000),
    param("sample_prompts", "采样提示词", "string", "", "preview", "采样"),
    param("sample_sampler", "采样器", "enum", "euler_a", "preview", "采样",
          options=enum(["euler_a", "euler", "ddim", "dpm++_2m", "dpm++_2m_karras", "lms", "pndm"])),

    # ---- advanced / 精度与显存 ----
    param("mixed_precision", "混合精度", "enum", "fp16", "advanced", "精度与显存",
          options=enum(["no", "fp16", "bf16"])),
    param("save_precision", "保存精度", "enum", "bf16", "advanced", "精度与显存",
          options=enum(["float", "fp32", "fp16", "bf16"])),
    param("full_fp16", "全 fp16", "bool", False, "advanced", "精度与显存"),
    param("full_bf16", "全 bf16", "bool", False, "advanced", "精度与显存"),
    # Anima disables this with a warning, so it is not offered there.
    param("fp8_base", "FP8 底模", "bool", False, "advanced", "精度与显存",
          architectures=("sd1", "sd2", "sdxl", "sd3", "flux1", "hunyuan_image", "lumina")),
    param("cache_latents", "缓存 Latents", "bool", True, "advanced", "精度与显存"),
    param("cache_latents_to_disk", "Latents 缓存到磁盘", "bool", False, "advanced", "精度与显存"),
    param("xformers", "CrossAttention", "enum", "sdp", "advanced", "精度与显存",
          options=enum(["none", "sdp", "xformers"]),
          help="sdp 用 PyTorch 原生注意力（默认，无需额外依赖）；xformers 需要引擎环境已安装 xformers；none 交给引擎默认。"),
    param("mem_eff_attn", "内存高效注意力", "bool", False, "advanced", "精度与显存"),
    param("sdpa", "SDPA 注意力", "bool", False, "advanced", "精度与显存"),
    # Declared here, but the presentation rewrite below moves it to 基础 / 模型与编码器
    # (display_order 0); it must stay declared for that rewrite to find it.
    param("vae", "VAE 路径", "string", "", "advanced", "精度与显存"),
    param("v_pred_like_loss", "v-pred 类似损失权重", "float", 0.0, "advanced", "精度与显存", min=0.0, max=100.0),
    param("scale_v_pred_loss_like_noise_pred", "按噪声预测缩放 v-pred 损失", "bool", False, "advanced", "精度与显存"),

    # ---- advanced / 损失 ----
    param("loss_type", "损失类型", "enum", "l2", "advanced", "损失", options=enum(["l2", "huber", "smooth_l1"])),
    param("min_snr_gamma", "Min SNR Gamma", "float", 0.0, "advanced", "损失", min=0.0, max=20.0, step=0.5),
    param("noise_offset", "Noise Offset", "float", 0.0, "advanced", "损失", min=0.0, max=1.0, step=0.001),
    param("noise_offset_random_strength", "Noise Offset 随机强度", "bool", False, "advanced", "损失"),
    param("adaptive_noise_scale", "自适应噪声尺度", "float", 0.0, "advanced", "损失", min=0.0, max=1.0),
    param("multires_noise_iterations", "多分辨率噪声迭代", "int", 0, "advanced", "损失", min=0, max=100),
    param("multires_noise_discount", "多分辨率噪声折扣", "float", 0.0, "advanced", "损失", min=0.0, max=1.0),
    param("ip_noise_gamma", "IP Noise Gamma", "float", 0.0, "advanced", "损失", min=0.0, max=1.0),
    param("ip_noise_gamma_random_strength", "IP Noise Gamma 随机强度", "bool", False, "advanced", "损失"),
    param("min_timestep", "最小时间步", "int", 0, "advanced", "损失", min=0, max=1000),
    param("max_timestep", "最大时间步", "int", 1000, "advanced", "损失", min=1, max=1000),
    param("debiased_estimation_loss", "Debiased Estimation Loss", "bool", False, "advanced", "损失"),
    param("huber_c", "Huber 损失参数", "float", 0.1, "advanced", "损失", min=0.0, max=10.0, step=0.01),
    param("huber_schedule", "Huber 调度", "enum", "snr", "advanced", "损失",
          options=enum(["snr", "constant", "exponential"])),
    param("prior_loss_weight", "正则损失权重", "float", 1.0, "advanced", "损失", min=0.0, max=10.0),

    # ---- advanced / 日志与上传 ----
    param("logging_dir", "日志目录", "string", "", "advanced", "日志与上传"),
    param("log_with", "日志后端", "enum", "", "advanced", "日志与上传",
          options=enum(["", "tensorboard", "wandb", "all"])),
    param("log_prefix", "日志前缀", "string", "", "advanced", "日志与上传"),
    param("log_tracker_name", "Tracker 名称", "string", "", "advanced", "日志与上传"),
    param("wandb_run_name", "WandB Run 名称", "string", "", "advanced", "日志与上传"),
    param("wandb_api_key", "WandB API Key", "string", "", "advanced", "日志与上传"),
    param("huggingface_repo_id", "HF 仓库 ID", "string", "", "advanced", "日志与上传"),
    param("huggingface_token", "HF Token", "string", "", "advanced", "日志与上传"),
    param("huggingface_repo_type", "HF 仓库类型", "enum", "", "advanced", "日志与上传",
          options=enum(["", "model", "dataset", "space"])),
    param("huggingface_path_in_repo", "HF 仓库内路径", "string", "", "advanced", "日志与上传"),
    param("huggingface_repo_visibility", "HF 可见性", "enum", "", "advanced", "日志与上传",
          options=enum(["", "public", "private"])),
    param("async_upload", "异步上传", "bool", False, "advanced", "日志与上传"),
    param("save_state_to_huggingface", "上传状态到 HF", "bool", False, "advanced", "日志与上传"),
    param("resume_from_huggingface", "从 HF 恢复", "string", "", "advanced", "日志与上传"),

    # ---- advanced / 元数据 ----
    param("metadata_title", "标题", "string", "", "advanced", "元数据"),
    param("metadata_author", "作者", "string", "", "advanced", "元数据"),
    param("metadata_description", "描述", "string", "", "advanced", "元数据"),
    param("metadata_license", "许可", "string", "", "advanced", "元数据"),
    param("metadata_tags", "标签", "string", "", "advanced", "元数据"),
    param("no_metadata", "不写入元数据", "bool", False, "advanced", "元数据"),

    # ---- native / 架构专属 ----
    param("network_module", "网络模块", "string", "", "native", "架构专属",
          native=False, unsupported_reason="按底模类型自动选择"),
    # Accepted by every kohya train_network script (checked with --help), but the
    # Anima docs list it under "Incompatible or Unsupported Options".
    param("clip_skip", "Clip Skip", "int", 1, groups=("basic", "native"), section="模型与编码器",
          architectures=("sd1", "sd2", "sdxl", "sd3", "flux1", "hunyuan_image", "lumina"),
          min=1, max=12),
    # Text-encoder model paths. These architectures take them on the command
    # line (library/sd3_train_utils.py, library/flux_train_utils.py,
    # hunyuan_image_train_network.py); SD1/2/SDXL read the TE from the base model.
    param("clip_l", "CLIP-L Text Encoder 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("sd3", "flux1")),
    param("clip_g", "CLIP-G Text Encoder 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("sd3",)),
    param("t5xxl", "T5XXL Text Encoder 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("sd3", "flux1")),
    param("text_encoder", "Text Encoder 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("hunyuan_image",)),
    # Anima's text encoder is an LLM plus a T5 tokenizer that the base model does
    # not carry, so it needs its own adapter path and tokenizer settings
    # (verified against `anima_train_network.py --help`).
    param("qwen3", "Qwen3 Text Encoder 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("anima",),
          help="Anima 的文本编码器（Qwen3-0.6B）：.safetensors 文件或 HuggingFace 模型目录；官方文档标为训练必需。"),
    param("llm_adapter_path", "LLM Adapter 路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("anima",)),
    param("t5_tokenizer_path", "T5 分词器路径", "string", "", groups=("basic", "native"),
          section="模型与编码器", architectures=("anima",)),
    param("t5_max_token_length", "T5 最大 Token 长度", "int", 512, groups=("basic", "native"),
          section="模型与编码器", architectures=("anima",), min=1, max=4096),
    param("llm_adapter_lr", "LLM Adapter 学习率", "float", None, section='优化器',
          architectures=("anima",), min=0.0, max=1.0),
    # Anima-specific text encoder and VAE knobs; the official Anima guide uses
    # them in its example command (anima_train_network.md).
    param("qwen3_max_token_length", "Qwen3 最大 Token 长度", "int", 512, groups=("basic", "native"),
          section="模型与编码器", architectures=("anima",), min=1, max=4096),
    param("vae_chunk_size", "VAE 分块大小", "int", None, "native", "精度与显存",
          architectures=("anima",), min=1, max=4096,
          help="Qwen-Image VAE 分块处理以降低显存，代价是变慢；官方示例使用 64。"),
    param("vae_disable_cache", "VAE 不使用内部缓存", "bool", False, "native", "精度与显存",
          architectures=("anima",), help="关闭 Qwen-Image VAE 的内部缓存以降低显存。"),
    param("qwen_image_vae_2d", "使用 2D Qwen-Image VAE", "bool", False, "native", "精度与显存",
          architectures=("anima",),
          help="图像专用的 2D VAE：约快 2 倍、峰值显存约为三分之一，官方推荐用于 latent 缓存。"),
    # These follow the engine, not one architecture: every train_network script
    # accepts no_half_vae / fused_backward_pass, and all but SD1/2 accept the
    # text-encoder cache and the mmap switch (checked via --help).
    param("cache_text_encoder_outputs", "缓存 TE 输出", "bool", False, "native", "精度与显存",
          architectures=("sdxl", "sd3", "flux1", "hunyuan_image", "lumina", "anima")),
    param("no_half_vae", "VAE 不使用半精度", "bool", False, "native", "精度与显存"),
    param("disable_mmap_load_safetensors", "禁用 mmap 加载", "bool", False, "native", "精度与显存",
          architectures=("sdxl", "sd3", "flux1", "hunyuan_image", "lumina", "anima")),
    param("fused_backward_pass", "融合反向传播", "bool", False, "native", "精度与显存",
          help="反向传播与优化器步融合以降低显存；仅在优化器为 Adafactor 时生效。"),
    # DiT models only: SD1/2/SDXL have no block swap in sd-scripts.
    param("blocks_to_swap", "Block Swap 块数", "int", None, "native", "精度与显存",
          architectures=("sd3", "flux1", "hunyuan_image", "lumina", "anima"), min=1,
          help="[引擎标为实验性] 把 DiT 的部分块换出显存以降低占用，代价是训练变慢；上限取决于模型块数（引擎要求 ≤ 块数 − 2），超限时引擎直接报错。"),
    param("cpu_offload_checkpointing", "梯度检查点卸载到 CPU", "bool", False, "native", "精度与显存",
          help="把梯度检查点的激活值放到内存，进一步降显存；与 Block Swap、Anima 的 Unsloth 卸载互斥（引擎会直接报错）。"),
    param("unsloth_offload_checkpointing", "Unsloth 卸载梯度检查点", "bool", False, "native", "精度与显存",
          architectures=("anima",),
          help="用异步非阻塞传输把梯度检查点的激活值卸载到内存，比 CPU 卸载更快（Anima 专属）；与 CPU 卸载、Block Swap 互斥，引擎会直接报错；启用它时引擎会自动打开梯度检查点。"),
    param("cache_text_encoder_outputs_to_disk", "TE 输出缓存到磁盘", "bool", False, "native", "精度与显存",
          architectures=("sdxl", "sd3", "flux1", "hunyuan_image", "lumina", "anima"),
          help="配合“缓存 TE 输出”使用，把文本编码器输出写到磁盘而不是内存。"),
    param("highvram", "高显存模式", "bool", False, "native", "精度与显存",
          help="不做低显存优化（不清理 CUDA 缓存），显存充足时更快。"),
    param("lowram", "低内存模式", "bool", False, "native", "精度与显存",
          help="把模型常驻显存，减少系统内存占用。"),

    # ---- advanced / 自定义参数 ----
    param("extra_params", "自定义参数", "string", "", "advanced", "自定义参数", multiline=True, native=False,
          help="补充 LP 未提供的引擎参数，会合并进引擎配置文件；例如 --clip_skip 2。空格或换行分隔，含空格的值用引号，单独的 --开关 表示开启；与上面已提供的参数重复会被拒绝。"),
)

# Presentation only; repeated tabs never duplicate configuration values.
# ``group`` (singular) is what the UI unions with ``groups``, so both must be
# replaced — leaving ``group`` at "advanced" kept a duplicate VAE field there.
PARAMS = tuple(
    replace(p, group="basic", groups=("basic", "native"), section="模型与编码器", display_order=0,
            help='可选的独立 VAE 权重路径；留空不传递此项，交由引擎使用底模自带的 VAE。') if p.key == 'vae' else p
    for p in PARAMS
)


# Native browse controls for every local path; remote repository names stay text.
_PATH_KINDS = {'network_weights': 'file',
 'vae': 'file',
 'qwen3': 'file_or_directory',
 'logging_dir': 'directory'}
PARAMS = tuple(replace(p, path_kind=_PATH_KINDS.get(p.key, p.path_kind)) for p in PARAMS)

# User-facing sampling input is text, not a filesystem path.
PARAMS = tuple(replace(p, multiline=True, path_kind=None, help="可选；直接输入提示词，每行生成一个采样。可从模板复制或保存为模板；留空不提供提示词。Kohya / Musubi 可使用当前引擎支持的 --n、--w、--h 等行内选项；启用预览还需设置采样间隔。") if p.key == "sample_prompts" else p for p in PARAMS)
