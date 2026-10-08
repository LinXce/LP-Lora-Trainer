"""Kohya sd-scripts LoRA parameter surface.

Source of truth: ``engine/kohya_ss/kohya_gui/class_*.py`` argument definitions,
``engine/kohya_ss/config example.toml`` and ``engine/kohya_ss/docs/LoRA/options.md``.
The ``sd-scripts`` submodule itself is not vendored, so keys mirror the sd-scripts
CLI/config names exactly.
"""

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
    param("max_train_steps", "训练步数", "int", 1000, min=1, max=10000000),
    param("max_train_epochs", "训练轮数", "int", None, min=1, max=100000, help="设置后覆盖训练步数"),
    param("train_batch_size", "Batch Size", "int", 1, min=1, max=128),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024),
    param("seed", "随机种子", "int", 42, min=0, max=4294967295),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0, max=100),
    param("gradient_checkpointing", "梯度检查点", "bool", True, help="降低显存占用，训练速度略降"),
    param("stop_text_encoder_training", "停止训练 TE 的步数", "int", 0, min=0, max=100000),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 0, min=0, max=32),
    param("persistent_data_loader_workers", "持久化 DataLoader", "bool", False),
    param("num_cpu_threads_per_process", "每进程 CPU 线程数", "int", 2, min=1, max=64),

    # ---- basic / 优化器 ----
    param("learning_rate", "学习率", "float", 0.0001, min=0.00000001, max=1.0),
    param("text_encoder_lr", "Text Encoder 学习率", "float", 0.0001, min=0.0, max=1.0),
    param("optimizer_type", "优化器", "enum", "AdamW8bit", options=enum([
        "AdamW", "AdamW8bit", "Adafactor", "AdamWScheduleFree", "Lion", "Lion8bit",
        "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptLion", "DAdaptSGD",
        "Prodigy", "pytorch_optimizer.CAME", "RAdamScheduleFree", "SGDNesterov",
    ])),
    param("optimizer_args", "优化器附加参数", "string", ""),
    param("lr_scheduler", "学习率策略", "enum", "constant", options=enum([
        "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
        "polynomial", "piecewise_constant", "inverse_sqrt", "adafactor", "rex",
    ])),
    param("lr_warmup_steps", "预热步数", "int", 0, min=0, max=1000000),
    param("lr_warmup", "预热比例(%)", "float", 0.0, min=0.0, max=100.0),
    param("lr_scheduler_num_cycles", "重启次数", "int", 1, min=1, max=100),
    param("lr_scheduler_power", "多项式幂", "float", 1.0, min=0.0, max=10.0),
    param("lr_scheduler_type", "自定义调度器模块", "string", ""),
    param("lr_scheduler_args", "调度器附加参数", "string", ""),

    # ---- basic / 网络 ----
    param("network_dim", "LoRA Rank", "int", 16, min=1, max=1024),
    param("network_alpha", "LoRA Alpha", "int", 16, min=1, max=1024),
    param("network_dropout", "网络 Dropout", "float", None, min=0.0, max=1.0),
    param("network_weights", "初始权重路径", "string", ""),
    param("dim_from_weights", "从权重推断 Rank", "bool", False),
    param("scale_weight_norms", "权重缩放系数", "float", None, min=0.0, max=100.0),
    param("network_args", "网络附加参数", "string", "", help="如 \"conv_dim=8 conv_alpha=8\""),
    param("network_train_unet_only", "仅训练 UNet", "bool", False),
    param("network_train_text_encoder_only", "仅训练 Text Encoder", "bool", False),

    # ---- basic / 数据集 ----
    param("resolution", "训练分辨率", "int", 512, min=64, max=4096, step=64, native=False),
    param("enable_bucket", "启用分桶", "bool", True, native=False),
    param("min_bucket_reso", "最小分桶分辨率", "int", 256, min=64, max=4096),
    param("max_bucket_reso", "最大分桶分辨率", "int", 2048, min=64, max=8192),
    param("bucket_reso_steps", "分桶步长", "int", 64, min=1, max=512),
    param("bucket_no_upscale", "分桶不放大", "bool", True),
    param("caption_extension", "Caption 扩展名", "enum", ".txt", native=False,
          options=enum([".txt", ".caption", ".cap"])),
    param("num_repeats", "图片重复次数", "int", 1, min=1, max=1000, native=False),
    param("keep_tokens", "保留 Token 数", "int", 0, min=0, max=100),
    param("max_token_length", "最大 Token 长度", "int", 150, min=75, max=225, step=75),
    param("shuffle_caption", "打乱 Caption", "bool", False),
    param("caption_dropout_rate", "Caption Dropout 比例", "float", 0.0, min=0.0, max=1.0),
    param("caption_dropout_every_n_epochs", "每 N 轮 Caption Dropout", "int", 0, min=0, max=100),
    param("weighted_captions", "加权 Caption", "bool", False),
    param("flip_aug", "水平翻转增强", "bool", False),
    param("color_aug", "颜色增强", "bool", False),
    param("random_crop", "随机裁剪", "bool", False),
    param("masked_loss", "掩码损失", "bool", False),
    param("vae_batch_size", "VAE Batch Size", "int", 0, min=0, max=128),

    # ---- advanced / 精度与显存 ----
    param("mixed_precision", "混合精度", "enum", "fp16", "advanced", "精度与显存",
          options=enum(["no", "fp16", "bf16"])),
    param("save_precision", "保存精度", "enum", "bf16", "advanced", "精度与显存",
          options=enum(["float", "fp32", "fp16", "bf16"])),
    param("full_fp16", "全 fp16", "bool", False, "advanced", "精度与显存"),
    param("full_bf16", "全 bf16", "bool", False, "advanced", "精度与显存"),
    param("fp8_base", "FP8 底模", "bool", False, "advanced", "精度与显存"),
    param("cache_latents", "缓存 Latents", "bool", True, "advanced", "精度与显存"),
    param("cache_latents_to_disk", "Latents 缓存到磁盘", "bool", False, "advanced", "精度与显存"),
    param("xformers", "CrossAttention", "enum", "xformers", "advanced", "精度与显存",
          options=enum(["none", "sdp", "xformers"])),
    param("mem_eff_attn", "内存高效注意力", "bool", False, "advanced", "精度与显存"),
    param("sdpa", "SDPA 注意力", "bool", False, "advanced", "精度与显存"),
    param("vae_dir", "VAE 目录", "string", "", "advanced", "精度与显存"),
    param("v_pred_like_loss", "v-pred 类似损失权重", "float", 0.0, "advanced", "精度与显存", min=0.0, max=100.0),
    param("scale_v_pred_loss_like_noise_pred", "按噪声预测缩放 v-pred 损失", "bool", False, "advanced", "精度与显存"),

    # ---- advanced / 保存与采样 ----
    param("save_every_n_epochs", "保存间隔（轮）", "int", 1, "advanced", "保存与采样", min=0, max=10000),
    param("save_every_n_steps", "保存间隔（步）", "int", 200, "advanced", "保存与采样", min=0, max=10000000),
    param("save_last_n_steps", "保留最近 N 步", "int", 0, "advanced", "保存与采样", min=0, max=10000000),
    param("save_last_n_steps_state", "保留最近 N 步状态", "int", 0, "advanced", "保存与采样", min=0, max=10000000),
    param("save_state", "保存训练状态", "bool", False, "advanced", "保存与采样"),
    param("save_state_on_train_end", "训练结束保存状态", "bool", False, "advanced", "保存与采样"),
    param("save_model_as", "保存格式", "enum", "safetensors", "advanced", "保存与采样",
          options=enum(["safetensors", "ckpt", "diffusers", "diffusers_safetensors"])),
    param("training_comment", "训练备注", "string", "", "advanced", "保存与采样"),
    param("sample_every_n_steps", "采样间隔（步）", "int", 0, "advanced", "保存与采样", min=0, max=10000000),
    param("sample_every_n_epochs", "采样间隔（轮）", "int", 0, "advanced", "保存与采样", min=0, max=10000),
    param("sample_prompts", "采样提示词文件", "string", "", "advanced", "保存与采样"),
    param("sample_sampler", "采样器", "enum", "euler_a", "advanced", "保存与采样",
          options=enum(["euler_a", "euler", "ddim", "dpm++_2m", "dpm++_2m_karras", "lms", "pndm"])),

    # ---- advanced / 损失 ----
    param("loss_type", "损失类型", "enum", "l2", "advanced", "损失", options=enum(["l2", "huber", "smooth_l1"])),
    param("min_snr_gamma", "Min SNR Gamma", "float", 0.0, "advanced", "损失", min=0.0, max=20.0, step=0.5),
    param("noise_offset", "Noise Offset", "float", 0.0, "advanced", "损失", min=0.0, max=1.0, step=0.001),
    param("noise_offset_random_strength", "Noise Offset 随机强度", "bool", False, "advanced", "损失"),
    param("noise_offset_type", "Noise Offset 类型", "enum", "Original", "advanced", "损失",
          options=enum(["Original", "Multires"])),
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
    param("clip_skip", "Clip Skip", "int", 1, "native", "SD 1.5 / 2.x",
          architectures=("sd1", "sd2"), min=1, max=12),
    param("sdxl_cache_text_encoder_outputs", "缓存 TE 输出", "bool", False, "native", "SDXL",
          architectures=("sdxl",)),
    param("sdxl_no_half_vae", "VAE 不使用半精度", "bool", False, "native", "SDXL", architectures=("sdxl",)),
    param("disable_mmap_load_safetensors", "禁用 mmap 加载", "bool", False, "native", "SDXL",
          architectures=("sdxl",)),
    param("fused_backward_pass", "融合反向传播", "bool", False, "native", "SDXL", architectures=("sdxl",)),
    param("learning_rate_te1", "Text Encoder 1 学习率", "float", 0.0001, "native", "SDXL",
          architectures=("sdxl",), min=0.0, max=1.0),
    param("learning_rate_te2", "Text Encoder 2 学习率", "float", 0.0001, "native", "SDXL",
          architectures=("sdxl",), min=0.0, max=1.0),
)
