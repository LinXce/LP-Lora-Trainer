"""AI Toolkit LoRA parameter surface.

Source of truth: ``engine/ai-toolkit/toolkit/config_modules.py`` dataclasses
(NetworkConfig / SaveConfig / LoggingConfig / TrainConfig / ModelConfig /
DatasetConfig / SampleConfig) and ``engine/ai-toolkit/config/examples/*.yaml``.
``block`` names the YAML section the value is written into.
"""

from adapters.capabilities import ArchSpec, Param, enum, param

# native = the value passed to ModelConfig.arch
ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("sd1", "SD 1.5", "run.py", "sd1"),
    ArchSpec("sd2", "SD 2.x", "run.py", "sd2"),
    ArchSpec("sd3", "SD 3 / 3.5", "run.py", "sd3"),
    ArchSpec("sdxl", "SDXL", "run.py", "sdxl"),
    ArchSpec("pixart", "PixArt-α", "run.py", "pixart"),
    ArchSpec("pixart_sigma", "PixArt-Σ", "run.py", "pixart_sigma"),
    ArchSpec("auraflow", "AuraFlow", "run.py", "auraflow"),
    ArchSpec("flux1", "FLUX.1", "run.py", "flux"),
    ArchSpec("flex1", "Flex.1", "run.py", "flex1"),
    ArchSpec("flex2", "Flex.2", "run.py", "flex2"),
    ArchSpec("lumina", "Lumina Image 2.0", "run.py", "lumina2"),
    ArchSpec("vega", "Vega", "run.py", "vega"),
    ArchSpec("ssd", "SSD", "run.py", "ssd"),
    ArchSpec("wan21", "Wan 2.1", "run.py", "wan21"),
    ArchSpec("anima", "Anima", "run.py", "anima"),
)

ARCH_BY_ID = {a.id: a for a in ARCHITECTURES}

PARAMS: tuple[Param, ...] = (
    # ---- basic / 训练 ----
    param("steps", "训练步数", "int", 1000, min=1, max=10000000, block="train"),
    param("batch_size", "Batch Size", "int", 1, min=1, max=128, block="train"),
    param("gradient_accumulation_steps", "梯度累积步数", "int", 1, min=1, max=1024),
    param("gradient_checkpointing", "梯度检查点", "bool", True, help="降低显存占用，训练速度略降"),
    param("max_grad_norm", "梯度裁剪", "float", 1.0, min=0.0, max=100.0),
    param("optimizer", "优化器", "enum", "adamw", options=enum(
        ["adamw", "adamw8bit", "adafactor", "prodigy", "lion", "sgd"])),
    param("lr", "学习率", "float", 0.000001, min=0.00000001, max=1.0),
    param("unet_lr", "UNet 学习率", "float", None, min=0.0, max=1.0),
    param("text_encoder_lr", "Text Encoder 学习率", "float", None, min=0.0, max=1.0),
    param("lr_scheduler", "学习率策略", "enum", "constant", options=enum(
        ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"])),
    param("train_unet", "训练 UNet", "bool", True),
    param("train_text_encoder", "训练 Text Encoder", "bool", False),

    # ---- basic / 网络 ----
    param("type", "网络类型", "enum", "lora", block="network", options=enum(["lora", "locon", "lorm", "lokr"])),
    param("rank", "LoRA Rank", "int", 4, block="network", min=1, max=1024),
    param("alpha", "LoRA Alpha", "float", 1.0, block="network", min=0.0, max=1024.0),
    param("conv", "Conv Rank", "int", None, block="network", min=1, max=1024),
    param("dropout", "网络 Dropout", "float", None, block="network", min=0.0, max=1.0),
    param("transformer_only", "仅 Transformer 层", "bool", True, block="network"),

    # ---- basic / 模型 ----
    param("dtype", "模型精度", "enum", "float16", block="model", options=enum(["float16", "bfloat16", "float32"])),
    param("vae_path", "VAE 路径", "string", "", block="model"),
    param("quantize", "量化模型", "bool", False, block="model"),
    param("qtype", "量化类型", "enum", "qfloat8", block="model",
          options=enum(["qfloat8", "float8", "qint8", "qint4"])),
    param("low_vram", "低显存模式", "bool", False, block="model"),

    # ---- advanced / 保存与采样 ----
    param("save_every", "保存间隔（步）", "int", 1000, "advanced", "保存与采样", block="save", min=1, max=10000000),
    param("max_step_saves_to_keep", "最多保留存档数", "int", 5, "advanced", "保存与采样", block="save", min=1, max=1000),
    param("save_format", "保存格式", "enum", "safetensors", "advanced", "保存与采样", block="save",
          options=enum(["safetensors", "diffusers"])),
    param("push_to_hub", "推送到 HuggingFace", "bool", False, "advanced", "保存与采样", block="save"),
    param("hf_repo_id", "HF 仓库 ID", "string", "", "advanced", "保存与采样", block="save"),
    param("sample_every", "采样间隔（步）", "int", 100, "advanced", "保存与采样", block="sample", min=0, max=10000000),
    param("sample_start_step", "开始采样步数", "int", 0, "advanced", "保存与采样", block="sample", min=0, max=10000000),
    param("sample_steps", "采样步数", "int", 20, "advanced", "保存与采样", block="sample", min=1, max=200),
    param("guidance_scale", "Guidance Scale", "float", 7.0, "advanced", "保存与采样", block="sample", min=1.0, max=30.0),
    param("width", "采样宽度", "int", 512, "advanced", "保存与采样", block="sample", min=64, max=4096, step=64),
    param("height", "采样高度", "int", 512, "advanced", "保存与采样", block="sample", min=64, max=4096, step=64),
    param("sampler", "采样器", "enum", "ddpm", "advanced", "保存与采样", block="sample",
          options=enum(["ddpm", "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde", "flowmatch"])),

    # ---- advanced / 数据集 ----
    param("caption_ext", "Caption 扩展名", "enum", ".txt", "advanced", "数据集", block="dataset",
          options=enum([".txt", ".caption", ".cap"])),
    param("resolution", "训练分辨率", "int", 512, "advanced", "数据集", block="dataset",
          min=64, max=4096, step=64),
    param("buckets", "启用分桶", "bool", True, "advanced", "数据集", block="dataset"),
    param("bucket_tolerance", "分桶容差", "int", 64, "advanced", "数据集", block="dataset", min=0, max=512),
    param("num_repeats", "图片重复次数", "int", 1, "advanced", "数据集", block="dataset", min=1, max=1000),
    param("caption_dropout_rate", "Caption Dropout 比例", "float", 0.0, "advanced", "数据集",
          block="dataset", min=0.0, max=1.0),
    param("shuffle_tokens", "打乱 Token", "bool", False, "advanced", "数据集", block="dataset"),
    param("flip_x", "水平翻转增强", "bool", False, "advanced", "数据集", block="dataset"),
    param("flip_y", "垂直翻转增强", "bool", False, "advanced", "数据集", block="dataset"),
    param("cache_latents", "缓存 Latents", "bool", True, "advanced", "数据集", block="dataset"),
    param("num_workers", "数据加载进程数", "int", 2, "advanced", "数据集", block="dataset", min=0, max=32),

    # ---- advanced / 日志 ----
    param("log_every", "日志间隔（步）", "int", 100, "advanced", "日志", block="logging", min=1, max=100000),
    param("verbose", "详细日志", "bool", False, "advanced", "日志", block="logging"),
    param("use_wandb", "使用 WandB", "bool", False, "advanced", "日志", block="logging"),
    param("project_name", "项目名称", "string", "ai-toolkit", "advanced", "日志", block="logging"),
    param("run_name", "Run 名称", "string", "", "advanced", "日志", block="logging"),

    # ---- native / 训练扩展 ----
    param("noise_scheduler", "噪声调度器", "enum", "ddpm", "native", "训练扩展", block="train",
          options=enum(["ddpm", "euler", "euler_a", "dpmpp_2m", "flowmatch"])),
    param("timestep_type", "时间步类型", "enum", "sigmoid", "native", "训练扩展", block="train",
          options=enum(["sigmoid", "linear", "lognorm_blend", "next_sample", "weighted", "one_step"])),
    param("loss_type", "损失类型", "enum", "mse", "native", "训练扩展", block="train",
          options=enum(["mse", "mae", "wavelet", "pixelspace", "mean_flow", "pseudo_huber"])),
    param("min_snr_gamma", "Min SNR Gamma", "float", None, "native", "训练扩展", block="train", min=0.0, max=20.0),
    param("snr_gamma", "SNR Gamma", "float", None, "native", "训练扩展", block="train", min=0.0, max=20.0),
    param("noise_offset", "Noise Offset", "float", 0.0, "native", "训练扩展", block="train", min=0.0, max=1.0),
    param("train_refiner", "训练 Refiner", "bool", True, "native", "训练扩展", block="train"),
    param("train_turbo", "训练 Turbo", "bool", False, "native", "训练扩展", block="train"),
    param("content_or_style", "内容/风格", "enum", "balanced", "native", "训练扩展", block="train",
          options=enum(["balanced", "style", "content"])),
    param("do_cfg", "启用 CFG", "bool", False, "native", "训练扩展", block="train"),
    param("cfg_scale", "CFG Scale", "float", 1.0, "native", "训练扩展", block="train", min=0.0, max=30.0),
    param("do_random_cfg", "随机 CFG", "bool", False, "native", "训练扩展", block="train"),
    param("prompt_dropout_prob", "Prompt Dropout 概率", "float", 0.0, "native", "训练扩展", block="train", min=0.0, max=1.0),
    param("single_item_batching", "单样本批处理", "bool", False, "native", "训练扩展", block="train"),
    param("disable_sampling", "禁用采样", "bool", False, "native", "训练扩展", block="train"),
    param("unload_text_encoder", "卸载 Text Encoder", "bool", False, "native", "训练扩展", block="train"),
    param("cache_text_embeddings", "缓存文本嵌入", "bool", False, "native", "训练扩展", block="train"),

    # ---- native / 模型扩展 ----
    param("use_text_encoder_1", "使用 TE1", "bool", True, "native", "模型扩展", block="model"),
    param("use_text_encoder_2", "使用 TE2", "bool", True, "native", "模型扩展", block="model"),
    param("text_encoder_bits", "TE 量化位宽", "enum", "16", "native", "模型扩展", block="model",
          options=enum(["16", "8", "4"])),
    param("attn_masking", "注意力掩码", "bool", False, "native", "模型扩展", block="model"),
    param("layer_offloading", "层卸载", "bool", False, "native", "模型扩展", block="model"),
    param("compile", "torch.compile", "bool", False, "native", "模型扩展", block="model"),
    param("compile_fullgraph", "compile Fullgraph", "bool", False, "native", "模型扩展", block="model"),
    param("compile_dynamic", "compile 动态形状", "bool", True, "native", "模型扩展", block="model"),
)
