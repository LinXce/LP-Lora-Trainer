"""Version-dependent options for the Anima CLI and MOSS typed cache/train requests."""

from adapters.capabilities import enum, param


def _anima(name, label, kind, default, section, **kwargs):
    return param(name, label, kind, default, "advanced", section, architectures=("anima",), **kwargs)


def _moss(name, label, kind, default, section, **kwargs):
    return param(f"moss_{name}", label, kind, default, "advanced", section,
                 architectures=("moss_image",), native_key=name, **kwargs)


ANIMA_OPTIONS = (
    _anima("clip_mode", "梯度裁剪模式", "enum", "fixed", "优化", options=enum(["fixed", "median"])),
    _anima("clip_k", "中位数裁剪倍数", "float", 2, "优化"),
    _anima("clip_window", "梯度范数窗口", "int", 50, "优化"),
    _anima("clip_per_micro", "逐微批裁剪", "bool", False, "优化"),
    _anima("clip_sigma_bands", "裁剪 σ 分段边界", "string", None, "优化"),
    _anima("grad_sigma_log", "记录梯度与 σ", "bool", False, "优化"),
    _anima("int8", "DiT INT8 模式", "enum", "off", "内存与加速", options=enum(["off", "fwd", "full", "stream"])),
    _anima("int8_file", "INT8 校准文件或 rtn", "string", None, "内存与加速"),
    _anima("stream", "残差流精度", "enum", "fp32", "内存与加速", options=enum(["bf16", "fp32"])),
    _anima("ema_window", "EMA 窗口比例", "float", 0.1, "保存"),
    _anima("ema_save_raw", "同时保存原始权重", "bool", False, "保存"),
    _anima("ema_extra", "额外 EMA 窗口比例", "string", "", "保存"),
    _anima("ema_card", "EMA 保存在显存", "bool", True, "保存"),
)


MOSS_CACHE_OPTIONS = (
    _moss("pairs", "i2i 图像配对清单", "string", "", "缓存", path_kind="file",
          help="选择 pairs.jsonl；引擎按清单中的目标图、参考图和字幕建立缓存。"),
    _moss("ref_scale_alt", "备用参考图缩放", "int", 0, "缓存"),
    _moss("ref_aspect", "参考图长宽比", "enum", "target", "缓存", options=enum(["target", "native"])),
    _moss("text_boxes", "缓存文字框", "bool", True, "缓存"),
    _moss("caption_variants", "缓存字幕变体", "bool", False, "缓存"),
    _moss("distill_teacher", "缓存教师嵌入", "bool", False, "缓存"),
    _moss("token_split", "Token 拆分方式", "enum", "off", "缓存", options=enum(["off", "syllable"])),
    _moss("prompt_tail", "提示词结尾", "enum", "sure", "缓存", options=enum(["off", "sure"])),
)


MOSS_TRAIN_OPTIONS = (
    _moss("mod_rank", "调制层 Rank", "int", 0, "网络"),
    _moss("mod_alpha", "调制层 Alpha", "float", None, "网络"),
    _moss("init_lora", "初始 LoRA 文件", "string", "", "网络", path_kind="file"),
    _moss("save_every_steps", "保存间隔（步）", "int", 0, "保存"),
    _moss("sample_prompts", "采样提示词或提示词文件", "string", "", "预览采样", multiline=True),
    _moss("sample_every_epochs", "采样间隔（轮）", "int", 0, "预览采样"),
    _moss("sample_at_first", "首步前采样", "bool", False, "预览采样"),
    _moss("sample_width", "采样宽度", "int", 1024, "预览采样", min=16, step=16),
    _moss("sample_height", "采样高度", "int", 1024, "预览采样", min=16, step=16),
    _moss("sample_steps", "采样步数", "int", 28, "预览采样"),
    _moss("sample_cfg", "采样引导强度", "float", 3, "预览采样"),
    _moss("sample_seed", "采样种子", "int", 42, "预览采样"),
    _moss("sample_prompt_format", "采样字幕格式", "enum", "auto", "预览采样", options=enum(["auto", "text", "layout_json", "tags"])),
    _moss("delta_rows", "增量更新行", "enum", "all", "网络", options=enum(["all", "text", "caption", "trigger"])),
    _moss("trigger", "触发词", "string", "", "网络"),
    _moss("trigger_registers", "触发寄存器个数", "int", 0, "网络"),
    _moss("register_lr", "寄存器学习率", "float", 0.002, "优化"),
    _moss("preserve_weight", "保真损失权重", "float", 0, "网络"),
    _moss("distill", "蒸馏教师", "enum", "off", "网络", options=enum(["off", "base", "ema"])),
    _moss("distill_sigma_min", "蒸馏最低 σ", "float", 0, "网络"),
    _moss("distill_sigma_ramp", "蒸馏 σ 渐变区间", "float", 0, "网络"),
    _moss("distill_ema_decay", "蒸馏教师 EMA 衰减", "float", 0.9999, "网络"),
    _moss("distill_mix", "蒸馏混合比例", "float", 0, "网络"),
    _moss("distill_add", "蒸馏附加权重", "float", 0, "网络"),
    _moss("distill_balance", "平衡蒸馏梯度", "bool", False, "网络"),
    _moss("distill_gate", "启用蒸馏门控", "bool", False, "网络"),
    _moss("distill_extra_sigma", "附加蒸馏 σ", "float", 0, "网络"),
    _moss("distill_extra_weight", "附加蒸馏权重", "float", 1, "网络"),
    _moss("ref_pos", "参考图位置模式", "enum", "aligned", "网络", options=enum(["aligned", "offset", "canvas"])),
    _moss("clause_hw", "分句位置权重", "float", 0, "网络"),
    _moss("clause_centres", "分句中心", "string", "", "网络"),
    _moss("clause_header", "启用分句标题", "bool", False, "网络"),
    _moss("text_box_hw", "文字框位置权重", "float", 0, "网络"),
    _moss("ref_adapters", "参考图适配层", "enum", "off", "网络", options=enum(["off", "kv", "all"])),
    _moss("ref_rank", "参考图 Rank", "int", 32, "网络"),
    _moss("ref_gate", "参考图门控", "enum", "off", "网络", options=enum(["off", "cut"])),
    _moss("gate_holes", "门控区域", "string", "", "网络"),
    _moss("gate_rate", "门控概率", "float", 1, "网络"),
    _moss("region_weight_end", "末步区域权重", "float", None, "网络"),
    _moss("tone_b_weight", "色调 B 权重", "float", 1, "网络"),
    _moss("tone_start", "色调开始比例", "float", 0, "网络"),
    _moss("tone_b_start", "色调 B 开始比例", "float", 0, "网络"),
    _moss("int8", "DiT INT8 模式", "enum", "stream", "内存与加速", options=enum(["off", "fwd", "full", "stream"])),
    _moss("int8_file", "INT8 校准文件或 rtn", "string", "", "内存与加速"),
    _moss("stream", "残差流精度", "enum", "fp32", "内存与加速", options=enum(["bf16", "fp32"])),
    _moss("ema_card", "EMA 保存在显存", "bool", True, "保存"),
)
