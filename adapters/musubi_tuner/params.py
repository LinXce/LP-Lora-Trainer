"""Musubi Tuner LoRA parameter surface.

Source of truth: ``engine/musubi-tuner/src/musubi_tuner/training/parser_common.py``
(shared training args), each ``*_setup_parser`` for model-specific args, and the
``*_cache_*.py`` scripts for the mandatory pre-caching stages. Musubi loads a flat
TOML training config, so sections are purely cosmetic here.
"""

from adapters.capabilities import ArchSpec, Param, enum, param

ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("hunyuan_video", "HunyuanVideo", "hv_train_network.py"),
    ArchSpec("hunyuan_video_15", "HunyuanVideo 1.5", "hv_1_5_train_network.py"),
    ArchSpec("wan", "Wan 2.1 / 2.2", "wan_train_network.py"),
    ArchSpec("flux2", "FLUX.2", "flux_2_train_network.py"),
    ArchSpec("flux1_kontext", "FLUX.1 Kontext", "flux_kontext_train_network.py"),
    ArchSpec("framepack", "FramePack", "fpack_train_network.py"),
    ArchSpec("qwen_image", "Qwen-Image", "qwen_image_train_network.py"),
    ArchSpec("z_image", "Z-Image", "zimage_train_network.py"),
    ArchSpec("hidream_o1", "HiDream-O1", "hidream_o1_train_network.py"),
    ArchSpec("ideogram4", "Ideogram 4", "ideogram4_train_network.py"),
    ArchSpec("kandinsky5", "Kandinsky 5", "kandinsky5_train_network.py"),
    ArchSpec("krea2", "Krea 2", "krea2_train_network.py"),
    ArchSpec("minimax_h3", "MiniMax-H3", "minimax_h3_train_network.py"),
)

# Training network module per architecture (from each architecture's docs).
NETWORK_MODULES = {
    "hunyuan_video": "networks.lora",
    "hunyuan_video_15": "networks.lora_hv_1_5",
    "wan": "networks.lora_wan",
    "flux2": "networks.lora_flux_2",
    "flux1_kontext": "networks.lora_flux",
    "framepack": "networks.lora_framepack",
    "qwen_image": "networks.lora_qwen_image",
    "z_image": "networks.lora",
    "hidream_o1": "networks.lora_hidream_o1",
    "ideogram4": "networks.lora_ideogram4",
    "kandinsky5": "networks.lora_kandinsky",
    "krea2": "networks.lora_krea2",
    "minimax_h3": "networks.lora_minimax_h3",
}

# timestep_sampling is architecture-sensitive: MiniMax-H3 hard-requires "uniform" and
# Ideogram 4 defaults to "ideogram4_shift", so the shared parameter excludes them and
# each declares its own override (same native_key).
TIMESTEP_SAMPLING_VALUES = enum([
    "sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "ideogram4_shift",
    "qwen_shift", "krea2_shift", "logsnr", "qinglong_flux", "qinglong_qwen",
])
_TIMESTEP_ARCHES = (
    "hunyuan_video", "hunyuan_video_15", "wan", "flux2", "flux1_kontext", "framepack",
    "qwen_image", "z_image", "hidream_o1", "kandinsky5", "krea2",
)
KANDINSKY_TASKS = enum([
    "k5-lite-t2i-hd", "k5-lite-i2i-hd", "k5-lite-t2v-5s-sd", "k5-lite-t2v-10s-sd",
    "k5-lite-i2v-5s-sd", "k5-pro-t2v-5s-sd", "k5-pro-t2v-5s-hd", "k5-pro-t2v-10s-sd",
    "k5-pro-t2v-10s-hd", "k5-pro-i2v-5s-sd", "k5-pro-i2v-5s-hd", "k5-lite-t2v-5s-distil-sd",
    "k5-lite-t2v-10s-distil-sd", "k5-lite-t2v-5s-nocfg-sd", "k5-lite-t2v-10s-nocfg-sd",
    "k5-lite-t2v-5s-pretrain-sd", "k5-lite-t2v-10s-pretrain-sd",
])

# Argument groups added by each architecture's *_setup_parser. Cache-only arguments
# (used solely by the pre-caching stages) are marked native=False so they never
# leak into the training config file.
ARCH_PARAMS: tuple[Param, ...] = (
    # HunyuanVideo
    param("dit_dtype", "DiT 精度", "enum", None, "native", "HunyuanVideo",
          architectures=("hunyuan_video", "hunyuan_video_15"), options=enum(["fp32", "fp16", "bf16"])),
    param("hv_dit_in_channels", "DiT 输入通道", "int", 16, "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="dit_in_channels", min=1, max=64),
    param("hv_fp8_llm", "FP8 LLM", "bool", False, "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="fp8_llm"),
    param("hv_text_encoder1", "文本编码器 1", "string", "", "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="text_encoder1"),
    param("hv_text_encoder2", "文本编码器 2", "string", "", "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="text_encoder2"),
    param("hv_vae_tiling", "VAE 分块", "bool", False, "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="vae_tiling"),
    param("hv_vae_chunk_size", "VAE 分块大小", "int", None, "native", "HunyuanVideo", architectures=("hunyuan_video",),
          native_key="vae_chunk_size", min=1, max=512),
    param("hv_vae_spatial_tile_sample_min_size", "VAE 空间分块最小尺寸", "int", None, "native", "HunyuanVideo",
          architectures=("hunyuan_video",), native_key="vae_spatial_tile_sample_min_size", min=1, max=4096),
    param("hv15_task", "任务", "enum", "t2v", "native", "HunyuanVideo 1.5", architectures=("hunyuan_video_15",),
          native_key="task", options=enum(["t2v", "i2v"])),
    param("hv15_i2v", "图生视频（缓存）", "bool", False, "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native=False),
    param("hv15_text_encoder", "文本编码器", "string", "", "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native_key="text_encoder"),
    param("hv15_fp8_vl", "FP8 文本编码器", "bool", False, "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native_key="fp8_vl"),
    param("hv15_byt5", "ByT5 路径", "string", "", "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native_key="byt5"),
    param("hv15_image_encoder", "图像编码器路径", "string", "", "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native_key="image_encoder"),
    param("hv15_vae_sample_size", "VAE 采样尺寸", "int", 128, "native", "HunyuanVideo 1.5",
          architectures=("hunyuan_video_15",), native_key="vae_sample_size", min=16, max=1024),

    # Wan 2.1 / 2.2
    param("wan_task", "任务/模型", "enum", "t2v-14B", "native", "Wan", architectures=("wan",),
          native_key="task", options=enum([
              "t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B", "flf2v-14B",
              "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC", "i2v-A14B", "t2v-A14B",
          ])),
    param("wan_i2v", "图生视频（缓存）", "bool", False, "native", "Wan", architectures=("wan",), native=False),
    param("wan_t5", "T5 路径", "string", "", "native", "Wan", architectures=("wan",), native_key="t5"),
    param("wan_fp8_t5", "FP8 T5", "bool", False, "native", "Wan", architectures=("wan",), native_key="fp8_t5"),
    param("wan_clip", "CLIP 路径", "string", "", "native", "Wan", architectures=("wan",), native_key="clip"),
    param("wan_vae_cache_cpu", "VAE 缓存到 CPU", "bool", False, "native", "Wan", architectures=("wan",),
          native=False),
    param("wan_offload_inactive_dit", "卸载空闲 DiT", "bool", False, "native", "Wan", architectures=("wan",),
          native_key="offload_inactive_dit"),
    param("wan_dit_high_noise", "高噪声 DiT 路径", "string", "", "native", "Wan", architectures=("wan",),
          native_key="dit_high_noise"),
    param("wan_timestep_boundary", "时间步边界", "int", None, "native", "Wan", architectures=("wan",),
          native_key="timestep_boundary", min=0, max=1000),

    # FLUX.2
    param("flux2_model_version", "模型版本", "enum", "dev", "native", "FLUX.2", architectures=("flux2",),
          native_key="model_version", options=enum(["dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"])),
    param("flux2_text_encoder", "文本编码器路径", "string", "", "native", "FLUX.2", architectures=("flux2",),
          native_key="text_encoder"),
    param("flux2_fp8_text_encoder", "FP8 文本编码器", "bool", False, "native", "FLUX.2", architectures=("flux2",),
          native_key="fp8_text_encoder"),

    # FLUX.1 Kontext
    param("kontext_text_encoder1", "文本编码器 1", "string", "", "native", "FLUX.1 Kontext",
          architectures=("flux1_kontext",), native_key="text_encoder1"),
    param("kontext_text_encoder2", "文本编码器 2", "string", "", "native", "FLUX.1 Kontext",
          architectures=("flux1_kontext",), native_key="text_encoder2"),
    param("kontext_fp8_t5", "FP8 T5", "bool", False, "native", "FLUX.1 Kontext",
          architectures=("flux1_kontext",), native_key="fp8_t5"),

    # Qwen-Image
    param("qwen_text_encoder", "文本编码器路径", "string", "", "native", "Qwen-Image", architectures=("qwen_image",),
          native_key="text_encoder"),
    param("qwen_fp8_vl", "FP8 文本编码器", "bool", False, "native", "Qwen-Image", architectures=("qwen_image",),
          native_key="fp8_vl"),
    param("qwen_num_layers", "层数", "int", None, "native", "Qwen-Image", architectures=("qwen_image",),
          native_key="num_layers", min=1, max=200),
    param("qwen_model_version", "模型版本", "enum", "original", "native", "Qwen-Image", architectures=("qwen_image",),
          native_key="model_version", options=enum(["original", "layered", "edit", "edit-2509", "edit-2511"])),

    # Z-Image
    param("zimage_text_encoder", "文本编码器路径", "string", "", "native", "Z-Image", architectures=("z_image",),
          native_key="text_encoder"),
    param("zimage_fp8_llm", "FP8 LLM", "bool", False, "native", "Z-Image", architectures=("z_image",),
          native_key="fp8_llm"),
    param("zimage_use_32bit_attention", "32 位注意力", "bool", False, "native", "Z-Image",
          architectures=("z_image",), native_key="use_32bit_attention"),

    # HiDream-O1
    param("hidream_model_type", "模型类型", "enum", "full", "native", "HiDream-O1",
          architectures=("hidream_o1",), native_key="model_type", options=enum(["full", "dev"])),
    param("hidream_task", "任务", "enum", "t2i", "native", "HiDream-O1", architectures=("hidream_o1",),
          native_key="task", options=enum(["t2i", "i2i"])),

    # Ideogram 4
    param("ideogram_text_encoder", "文本编码器路径", "string", "", "native", "Ideogram 4",
          architectures=("ideogram4",), native_key="text_encoder"),
    param("ideogram_sampler_preset", "采样预设", "enum", "V4_DEFAULT_20", "native", "Ideogram 4",
          architectures=("ideogram4",), native_key="sampler_preset",
          options=enum(["V4_DEFAULT_20", "V4_QUALITY_48", "V4_TURBO_12"])),
    param("ideogram4_timestep_sampling", "时间步采样", "enum", "ideogram4_shift", "native", "Ideogram 4",
          architectures=("ideogram4",), native_key="timestep_sampling", options=TIMESTEP_SAMPLING_VALUES),

    # Kandinsky 5
    param("kandinsky_task", "任务", "enum", "k5-lite-t2i-hd", "native", "Kandinsky 5",
          architectures=("kandinsky5",), native_key="task", options=KANDINSKY_TASKS),
    param("kandinsky_i2v_mode", "I2V 模式", "enum", "first", "native", "Kandinsky 5",
          architectures=("kandinsky5",), native_key="i2v_mode", options=enum(["first", "first_last"])),
    param("kandinsky_text_encoder_qwen", "Qwen 文本编码器", "string", "", "native", "Kandinsky 5",
          architectures=("kandinsky5",), native_key="text_encoder_qwen"),
    param("kandinsky_text_encoder_clip", "CLIP 文本编码器", "string", "", "native", "Kandinsky 5",
          architectures=("kandinsky5",), native_key="text_encoder_clip"),

    # Krea 2
    param("krea2_text_encoder", "文本编码器路径", "string", "", "native", "Krea 2", architectures=("krea2",),
          native_key="text_encoder"),
    param("krea2_convrot_int8", "ConvRot INT8", "bool", False, "native", "Krea 2", architectures=("krea2",),
          native_key="convrot_int8"),
    param("krea2_convrot_int8_bwd", "ConvRot 反向模式", "enum", "bf16", "native", "Krea 2",
          architectures=("krea2",), native_key="convrot_int8_bwd", options=enum(["bf16", "int8"]),
          help="bf16 最精确；int8 复用 int8 GEMM（更快，需要 triton）"),
    param("krea2_turbo_dit", "Turbo DiT 路径", "string", "", "native", "Krea 2", architectures=("krea2",),
          native_key="turbo_dit", help="用 RAW 训练、采样时切换到 Turbo 权重的蒸馏 DiT 路径"),
    param("krea2_turbo_dit_cache", "Turbo DiT 常驻内存", "bool", False, "native", "Krea 2",
          architectures=("krea2",), native_key="turbo_dit_cache",
          help="把 Turbo 权重常驻 CPU 内存并乒乓换入（多用约 1x CPU、更快）"),
    param("krea2_turbo_lora", "Turbo LoRA 路径", "string", "", "native", "Krea 2", architectures=("krea2",),
          native_key="turbo_lora", help="与 turbo_dit 互斥的另一种 Turbo 方案"),
    param("krea2_turbo_lora_multiplier", "Turbo LoRA 系数", "float", 1.0, "native", "Krea 2",
          architectures=("krea2",), native_key="turbo_lora_multiplier", min=0.0, max=10.0),

    # MiniMax-H3
    param("h3_task", "任务", "enum", "t2va", "native", "MiniMax-H3", architectures=("minimax_h3",),
          native_key="task", options=enum(["t2va", "fl2va", "ref2va"])),
    param("h3_timestep_sampling", "时间步采样", "enum", "uniform", "native", "MiniMax-H3",
          architectures=("minimax_h3",), native_key="timestep_sampling", options=TIMESTEP_SAMPLING_VALUES),
    param("h3_video_vae", "视频 VAE 路径", "string", "", "native", "MiniMax-H3", architectures=("minimax_h3",),
          native_key="video_vae"),
    param("h3_audio_vae", "音频 VAE 路径", "string", "", "native", "MiniMax-H3", architectures=("minimax_h3",),
          native_key="audio_vae"),
    param("h3_text_encoder", "文本编码器路径", "string", "", "native", "MiniMax-H3", architectures=("minimax_h3",),
          native_key="text_encoder"),
    param("h3_one_frame", "单帧模式", "bool", False, "native", "MiniMax-H3", architectures=("minimax_h3",),
          native_key="one_frame"),

    # FramePack
    param("fpack_latent_window_size", "潜空间窗口大小", "int", 9, "native", "FramePack",
          architectures=("framepack",), native_key="latent_window_size", min=1, max=128),
    param("fpack_image_encoder", "图像编码器路径", "string", "", "native", "FramePack",
          architectures=("framepack",), native_key="image_encoder"),
    param("fpack_text_encoder1", "文本编码器 1", "string", "", "native", "FramePack",
          architectures=("framepack",), native_key="text_encoder1"),
    param("fpack_text_encoder2", "文本编码器 2", "string", "", "native", "FramePack",
          architectures=("framepack",), native_key="text_encoder2"),
    param("fpack_fp8_llm", "FP8 LLM", "bool", False, "native", "FramePack", architectures=("framepack",),
          native_key="fp8_llm"),
    param("fpack_f1", "F1 模式", "bool", False, "native", "FramePack", architectures=("framepack",), native_key="f1"),
    param("fpack_one_frame", "单帧模式", "bool", False, "native", "FramePack", architectures=("framepack",),
          native_key="one_frame"),

    # Shared training options from training/parser_common.py that every musubi
    # architecture (including Krea 2) accepts. Deliberately excludes options that
    # are commented out upstream for this path (--full_fp16/--full_bf16,
    # --flow_shift exist only in per-architecture scripts).
    param("gc_cpu_offload", "梯度检查点 CPU 卸载", "bool", False, "advanced", "内存与加速",
          native_key="gradient_checkpointing_cpu_offload",
          help="把梯度检查点的激活值卸载到 CPU，进一步降低显存占用"),
    param("block_swap_h2d_only", "块交换仅 H2D", "bool", False, "advanced", "内存与加速",
          native_key="block_swap_h2d_only",
          help="只做 Host→Device 拷贝（保留 CPU 主副本）"),
    param("block_swap_ring_size", "块交换环形缓冲", "int", None, "advanced", "内存与加速",
          native_key="block_swap_ring_size", min=1, max=8,
          help="配合“块交换仅 H2D”：2 = 双缓冲重叠，1 = 最小显存"),
    param("vae_dtype", "VAE 精度", "enum", "", "advanced", "内存与加速", native_key="vae_dtype",
          options=enum(["", "float16", "bfloat16", "float32"]), help="留空由模型与引擎决定"),
    param("flash3", "FlashAttention 3", "bool", False, "advanced", "内存与加速", native_key="flash3"),
    param("cuda_allow_tf32", "允许 TF32", "bool", False, "advanced", "内存与加速",
          native_key="cuda_allow_tf32"),
    param("cuda_cudnn_benchmark", "cuDNN benchmark", "bool", False, "advanced", "内存与加速",
          native_key="cuda_cudnn_benchmark"),

    # 时间步与采样（共享项）
    param("mode_scale", "Mode 加权缩放", "float", None, "advanced", "时间步与采样", native_key="mode_scale",
          min=0.0, max=10.0, help="仅 weighting_scheme=mode 时生效"),
    param("preserve_distribution_shape", "保持分布形状", "bool", False, "advanced", "时间步与采样",
          native_key="preserve_distribution_shape"),
    param("num_timestep_buckets", "时间步分桶数", "int", None, "advanced", "时间步与采样",
          native_key="num_timestep_buckets", min=1, max=10000,
          help="设置后按时间步分桶采样，而非连续采样"),
    param("sample_every_n_epochs", "采样间隔（轮）", "int", None, "advanced", "时间步与采样",
          native_key="sample_every_n_epochs", min=1, max=10000),

    # 保存与元数据（共享项）
    param("save_last_n_epochs", "保留最近 N 轮", "int", None, "advanced", "保存与日志",
          native_key="save_last_n_epochs", min=1, max=10000),
    param("resume", "续训状态路径", "string", "", "advanced", "保存与日志", native_key="resume",
          help="从 save_state 生成的 state 目录继续训练"),
    param("training_comment", "训练备注", "string", "", "advanced", "元数据", native_key="training_comment"),

    # 优化器调度（共享项）
    param("lr_scheduler_timescale", "调度器时间尺度", "int", None, "advanced", "优化",
          native_key="lr_scheduler_timescale", min=1, max=100000),
    param("lr_scheduler_min_lr_ratio", "最小学习率比例", "float", None, "advanced", "优化",
          native_key="lr_scheduler_min_lr_ratio", min=0.0, max=1.0),
)

PARAMS: tuple[Param, ...] = (
    # ---- basic / 训练 ----
    param("max_train_steps", "训练步数", "int", 1600, min=1, max=10000000),
    param("max_train_epochs", "训练轮数", "int", None, min=1, max=100000, help="设置后覆盖训练步数"),
    param("seed", "随机种子", "int", None, min=0, max=4294967295),
    param("gradient_checkpointing", "梯度检查点", "bool", False, help="降低显存占用，训练速度略降"),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024),
    param("max_data_loader_n_workers", "数据加载进程数", "int", 8, min=0, max=64),
    param("persistent_data_loader_workers", "持久化 DataLoader", "bool", False),
    param("mixed_precision", "混合精度", "enum", "bf16", options=enum(["no", "fp16", "bf16"])),
    param("save_precision", "保存精度", "enum", "bf16", options=enum(["float", "fp32", "fp16", "bf16"])),

    # ---- basic / 数据集 ----
    param("resolution_width", "训练宽度", "int", 960, min=64, max=4096, step=16, native=False),
    param("resolution_height", "训练高度", "int", 544, min=64, max=4096, step=16, native=False),
    param("batch_size", "Batch Size", "int", 1, min=1, max=128, native=False),
    param("enable_bucket", "启用分桶", "bool", False, native=False),
    param("bucket_no_upscale", "分桶不放大", "bool", False, native=False),
    # Musubi derives its buckets from the resolution set; there is no min/max
    # bucket reso (that is a kohya option). Listing several sizes is the
    # equivalent control, so each line becomes one bucket resolution.
    param("bucket_resolutions", "多分辨率分桶", "string", "", native=False, multiline=True,
          help="每行一个尺寸（如 960x544）；填写后覆盖上面的宽高，作为分桶分辨率集合"),
    param("caption_extension", "Caption 扩展名", "enum", ".txt", native=False,
          options=enum([".txt", ".caption", ".cap"])),
    param("num_repeats", "图片重复次数", "int", 1, min=1, max=1000, native=False),
    param("cache_directory", "缓存目录", "string", "", native=False, help="留空则写入任务目录内的 cache"),

    # ---- basic / 模型 ----
    param("vae", "VAE 路径", "string", "", native_key="vae"),

    # ---- basic / 网络 ----
    param("network_module", "网络模块", "string", "", native=False, unsupported_reason="按底模类型自动选择"),
    param("network_dim", "LoRA Rank", "int", 16, min=1, max=1024),
    param("network_alpha", "LoRA Alpha", "float", 1.0, min=0.0, max=1024.0),
    param("network_dropout", "网络 Dropout", "float", None, min=0.0, max=1.0),
    param("network_weights", "初始权重路径", "string", "", native_key="network_weights"),
    param("network_args", "网络附加参数", "string", "", native_key="network_args", list_arg=True, multiline=True),
    param("dim_from_weights", "从权重推断 Rank", "bool", False),
    param("scale_weight_norms", "权重缩放系数", "float", None, min=0.0, max=100.0),

    # ---- basic / 优化 ----
    param("optimizer_type", "优化器", "enum", "AdamW8bit", options=enum([
        "AdamW", "AdamW8bit", "Adafactor", "Lion", "Lion8bit", "Prodigy",
        "pytorch_optimizer.CAME", "RAdamScheduleFree", "SGDNesterov",
    ])),
    param("optimizer_args", "优化器附加参数", "string", "", native_key="optimizer_args", list_arg=True, multiline=True),
    param("learning_rate", "学习率", "float", 0.000002, min=0.00000001, max=1.0),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0.0, max=100.0),
    param("lr_scheduler", "学习率策略", "enum", "constant", options=enum([
        "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
        "polynomial", "inverse_sqrt", "adafactor", "rex",
    ])),
    param("lr_warmup_steps", "预热步数", "int", 0, min=0, max=1000000),
    param("lr_decay_steps", "衰减步数", "int", 0, min=0, max=1000000),
    param("lr_scheduler_num_cycles", "重启次数", "int", 1, min=1, max=100),
    param("lr_scheduler_power", "多项式幂", "float", 1.0, min=0.0, max=10.0),
    param("lr_scheduler_type", "自定义调度器模块", "string", ""),
    param("lr_scheduler_args", "调度器附加参数", "string", "", native_key="lr_scheduler_args", list_arg=True, multiline=True),

    # ---- advanced / 内存与加速 ----
    param("fp8_base", "FP8 底模", "bool", False, "advanced", "内存与加速"),
    param("fp8_scaled", "FP8 缩放", "bool", False, "advanced", "内存与加速"),
    param("blocks_to_swap", "交换块数量", "int", None, "advanced", "内存与加速", min=0, max=100),
    param("use_pinned_memory_for_block_swap", "交换使用锁页内存", "bool", False, "advanced", "内存与加速"),
    param("img_in_txt_in_offloading", "卸载 img_in/txt_in", "bool", False, "advanced", "内存与加速"),
    param("disable_numpy_memmap", "禁用 numpy memmap", "bool", False, "advanced", "内存与加速"),
    param("sdpa", "SDPA 注意力", "bool", True, "advanced", "内存与加速"),
    param("flash_attn", "FlashAttention", "bool", False, "advanced", "内存与加速"),
    param("sage_attn", "SageAttention", "bool", False, "advanced", "内存与加速"),
    param("xformers", "xformers", "bool", False, "advanced", "内存与加速"),
    param("split_attn", "分割注意力", "bool", False, "advanced", "内存与加速"),
    param("compile", "torch.compile", "bool", False, "advanced", "内存与加速"),
    param("compile_backend", "compile 后端", "string", "inductor", "advanced", "内存与加速"),
    param("compile_mode", "compile 模式", "enum", "default", "advanced", "内存与加速",
          options=enum(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])),
    param("compile_fullgraph", "compile Fullgraph", "bool", False, "advanced", "内存与加速"),
    param("dynamo_backend", "Dynamo 后端", "enum", "NO", "advanced", "内存与加速",
          options=enum(["NO", "EAGER", "AOT_EAGER", "INDUCTOR", "ONNXRT", "NVFUSER", "TORCHXLA_TRACE_ONCE", "TENSORRT"])),

    # ---- advanced / 时间步与采样 ----
    param("guidance_scale", "Guidance Scale", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=30.0),
    param("timestep_sampling", "时间步采样", "enum", "sigma", "advanced", "时间步与采样",
          architectures=_TIMESTEP_ARCHES, options=TIMESTEP_SAMPLING_VALUES),
    param("discrete_flow_shift", "离散流偏移", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("sigmoid_scale", "Sigmoid 缩放", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=20.0),
    param("weighting_scheme", "加权方案", "enum", "none", "advanced", "时间步与采样",
          options=enum(["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"])),
    param("logit_mean", "Logit 均值", "float", 0.0, "advanced", "时间步与采样", min=-10.0, max=10.0),
    param("logit_std", "Logit 标准差", "float", 1.0, "advanced", "时间步与采样", min=0.0, max=10.0),
    param("min_timestep", "最小时间步", "int", None, "advanced", "时间步与采样", min=0, max=999),
    param("max_timestep", "最大时间步", "int", None, "advanced", "时间步与采样", min=1, max=1000),
    param("sample_every_n_steps", "采样间隔（步）", "int", None, "advanced", "时间步与采样", min=1, max=10000000),
    param("sample_at_first", "训练前采样", "bool", False, "advanced", "时间步与采样"),
    param("sample_prompts", "采样提示词文件", "string", "", "advanced", "时间步与采样"),

    # ---- advanced / 保存与日志 ----
    param("save_every_n_epochs", "保存间隔（轮）", "int", None, "advanced", "保存与日志", min=1, max=10000),
    param("save_every_n_steps", "保存间隔（步）", "int", 200, "advanced", "保存与日志", min=1, max=10000000),
    param("save_last_n_steps", "保留最近 N 步", "int", None, "advanced", "保存与日志", min=1, max=10000000),
    param("save_state", "保存训练状态", "bool", False, "advanced", "保存与日志"),
    param("save_state_on_train_end", "训练结束保存状态", "bool", False, "advanced", "保存与日志"),
    param("logging_dir", "日志目录", "string", "", "advanced", "保存与日志"),
    param("log_with", "日志后端", "enum", "", "advanced", "保存与日志",
          options=enum(["", "tensorboard", "wandb", "all"])),
    param("wandb_run_name", "WandB Run 名称", "string", "", "advanced", "保存与日志"),
    param("huggingface_repo_id", "HF 仓库 ID", "string", "", "advanced", "保存与日志"),
    param("async_upload", "异步上传", "bool", False, "advanced", "保存与日志"),

    # ---- advanced / 元数据 ----
    param("metadata_title", "标题", "string", "", "advanced", "元数据"),
    param("metadata_author", "作者", "string", "", "advanced", "元数据"),
    param("metadata_description", "描述", "string", "", "advanced", "元数据"),
    param("metadata_license", "许可", "string", "", "advanced", "元数据"),
    param("metadata_tags", "标签", "string", "", "advanced", "元数据"),
    param("no_metadata", "不写入元数据", "bool", False, "advanced", "元数据"),
) + ARCH_PARAMS
