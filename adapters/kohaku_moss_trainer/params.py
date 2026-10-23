"""KohakuMossTrainer parameter surface.

Source of truth: ``engine/KohakuMossTrainer/scripts/train/moss.py`` — every knob is a
module-level global there, so ``native_key`` is the global's own name and the rendered
config is a **Python** file (``native_format = "python"``). Values that are dicts/tuples
in the entry are exposed as JSON text and written back as Python literals.

Supported base models: this engine trains **MOSS-Image only** (Base / SFT), so there is a
single architecture; ``VARIANT`` selects Base or SFT.
"""

from adapters.capabilities import ArchSpec, Param, enum, param

ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("moss_image", "MOSS-Image", "scripts/train/moss.py"),
)

# Groups used by the entry script, kept in the UI's basic/advanced split.
OPTIMIZERS = enum(["adamw", "adamw16", "adamw8bit", "adamw8bit_kernel", "muon", "prodigy", "schedule_free"])
ALGOS = enum(["lora", "locon", "loha", "lokr", "oft", "boft", "glora", "ia3", "dylora", "tlora", "full"])
PRESETS = enum(["attn", "attn-mlp", "full-block", "all"])
CACHES = enum(["off", "offline", "online"])


def _json(key: str, label: str, default: str, section: str, help: str, group: str = "advanced",
          detail: str = "") -> Param:
    """A dict/tuple knob: JSON text in the form, a Python literal in the config."""
    return param(key, label, "string", default, group, section, multiline=True, help=help,
                 detail=detail)


RANKED_ALGOS = ("lora", "locon", "loha", "lokr", "oft", "boft", "glora", "dylora", "tlora")

PARAMS: tuple[Param, ...] = (
    # ---- basic / 模型 ----
    param("WEIGHTS", "MOSS-Image 权重", "string", "", "basic", "模型与编码器",
          path_kind="file_or_directory", file_types=("Safetensors (*.safetensors)", "All files (*.*)"),
          native_key="WEIGHTS", required_for=("moss_image",),
          help="两种导入方式：\n"
               "1) 给 MOSS-Image 目录（含 transformer/、text_encoder/、vae/）—— 编码器路径自动推导；\n"
               "2) 给单个 transformer .safetensors —— 必须同时填写下面的「文本编码器」与「VAE」。"),
    param("VARIANT", "模型变体", "enum", "base", "basic", "模型与编码器",
          options=enum(["base", "sft"]), native_key="VARIANT"),
    param("TEXT_ENCODER", "文本编码器路径", "string", "", "basic", "模型与编码器", path_kind="file_or_directory",
          native_key="TEXT_ENCODER", help="留空使用 <WEIGHTS>/text_encoder（Qwen3.5）。"),
    param("VAE", "VAE 路径", "string", "", "basic", "模型与编码器", path_kind="file",
          native_key="VAE", help="留空使用 <WEIGHTS>/vae。"),
    param("VAE_DTYPE", "VAE 精度", "enum", "auto", "advanced", "模型与编码器",
          options=enum(["auto", "bf16", "fp16", "fp32"]), native_key="VAE_DTYPE"),
    param("PRECISION", "训练精度", "enum", "bf16", "basic", "模型与编码器",
          options=enum(["bf16", "fp16", "auto"]), native_key="PRECISION"),

    # ---- basic / 数据 ----
    param("RESOLUTION_MODE", "分辨率模式", "enum", "resize", "basic", "数据集",
          options=enum(["resize", "bucket", "shifted_square", "square_crop"]), native_key="RESOLUTION_MODE",
          help="resize 保持原生宽高比、面积约等于目标面积。"),
    param("TARGET_AREA", "目标面积（像素）", "int", 1048576, "basic", "数据集", min=65536, max=16777216,
          native_key="TARGET_AREA"),
    # ---- 字幕（拆分字段；组装成引擎的 CAPTION）----
    param("CAPTION_TEMPLATE", "字幕模板", "enum", "tags", "advanced", "字幕", native=False,
          options=enum(["tags", "nl", "mix", "json"]), help_style="info",
          help="tags = 标签模板（读 .txt）；nl = 自然语言（读 .caption）；mix = 按 p_nl 混用；json = 结构化输出。"
               "切换后只显示该模板用得到的字段。"),
    param("CAPTION_KEEP", "置前字段", "string", "artist", "advanced", "字幕", native=False,
          visible_when=("CAPTION_TEMPLATE", ("tags", "nl", "mix")),
          help="逗号分隔的 CatalogRow 字段名（artist / tags / caption_nl / rating），会放在字幕最前。"),
    param("CAPTION_P_NL", "mix 自然语言概率", "float", 0.5, "advanced", "字幕", native=False,
          min=0.0, max=1.0, visible_when=("CAPTION_TEMPLATE", ("mix",)),
          help="0.5 = 一半用 .caption、一半用标签。"),
    # 以下为“多余参数”：默认收起，需在参数区点「显示字幕高级字段」。
    param("CAPTION_KEEP_FIRST", "固定保留前 N 个标签", "int", 0, "advanced", "字幕", native=False,
          min=0, max=64, visible_when=("CAPTION_TEMPLATE", ("tags", "mix")),
          help="0 = 全部参与打乱；设为 1 可让触发词永远排第一。"),
    param("CAPTION_TAG_DROPOUT", "标签丢弃概率", "float", 0.0, "advanced", "字幕", native=False,
          min=0.0, max=1.0, visible_when=("CAPTION_TEMPLATE", ("tags", "mix"))),
    param("CAPTION_GROUP_DROPOUT", "整组丢弃概率", "float", 0.0, "advanced", "字幕", native=False,
          min=0.0, max=1.0, visible_when=("CAPTION_TEMPLATE", ("tags",))),
    param("CAPTION_SHUFFLE", "打乱标签顺序", "bool", True, "advanced", "字幕", native=False,
          visible_when=("CAPTION_TEMPLATE", ("tags", "mix"))),
    param("CAPTION_UNDERSCORES", "下划线转空格", "bool", True, "advanced", "字幕", native=False,
          visible_when=("CAPTION_TEMPLATE", ("tags",))),
    param("CAPTION_UNESCAPE", "反转义括号", "bool", True, "advanced", "字幕", native=False,
          visible_when=("CAPTION_TEMPLATE", ("tags",))),
    param("CAPTION_SEPARATORS", "分隔符与连接符", "string",
          '{"keep_sep": "|||", "group_sep": "%%", "joiner": ", "}', "advanced", "字幕",
          native=False, visible_when=("CAPTION_TEMPLATE", ("tags",)),
          help="JSON：keep_sep / group_sep / joiner。"),
    param("CAPTION_AFFIX", "首尾附加文本", "string", '{"prefix": "", "suffix": ""}',
          "advanced", "字幕", native=False, visible_when=("CAPTION_TEMPLATE", ("tags",)),
          help="JSON：prefix / suffix。"),
    param("CAPTION_FALLBACK", "nl 兜底模板", "string", '{"template": "tags"}',
          "advanced", "字幕", native=False, visible_when=("CAPTION_TEMPLATE", ("nl",)),
          help="某行没有自然语言文本时使用的子模板（JSON）。"),
    param("CAPTION_FIELDS", "json 输出字段", "string", "artist,rating,tags,caption_nl",
          "advanced", "字幕", native=False, visible_when=("CAPTION_TEMPLATE", ("json",)),
          help="逗号分隔；json 模板会输出这些字段。"),
    param("TEXT_DROP", "空提示词概率", "float", 0.1, "advanced", "数据集", min=0.0, max=1.0, native_key="TEXT_DROP"),
    param("LATENT_CACHE", "VAE 潜空间缓存", "enum", "offline", "basic", "数据集", options=CACHES,
          native_key="LATENT_CACHE", help="off / offline（训练前缓存）/ online（训练中编码）。"),
    param("TEXT_CACHE", "文本编码缓存", "enum", "offline", "basic", "数据集", options=CACHES, native_key="TEXT_CACHE"),
    # scripts/data/cache_text.py:50 -- captions per encoder pass; a big value leaves the
    # progress bar at 0 for a long time on a large/offloaded text encoder.
    param("ENCODE_BATCH", "文本编码批大小", "int", 512, "advanced", "数据集",
          min=1, max=4096, native_key="ENCODE_BATCH",
          help="每次编码推向文本编码器的标题数；显存/内存紧张或长时间没进度时调小（如 64）。"),
    param("VAE_ENCODE", "VAE 编码计划", "string", "{}", "advanced", "数据集", native_key="VAE_ENCODE",
          help="引擎 VaePlan 字段（cache_latents.py:54）：memory: resident|offload|stream|auto|cpu、"
               "batch、ahead、tile、compile: auto|on|off。默认 {} = 引擎默认；"
               "没有可用的 triton 时把 compile 设为 \"off\"，否则会报 TritonMissing。"),
    param("CACHE_DIR", "缓存目录", "string", "", "basic", "数据集", path_kind="directory", native_key="CACHE_DIR",
          help="留空为 <数据集>/cache。"),
    param("SOURCE", "数据集读取方式", "enum", "kohya", "advanced", "数据集", native=False,
          options=enum(["kohya", "folder", "jsonl", "parquet", "webdataset", "moss_shard"]),
          help_style="info",
          help="写入 DATASETS 条目的 source（三个训练阶段共用同一份 DATASETS）：\n"
               "kohya = <重复次数>_<名称>/图片 + 同名 .txt / .caption（LP 数据集目录的常规布局）\n"
               "folder = 任意目录树，旁车文件 .txt / .caption / .json\n"
               "jsonl = 每行一条记录，图片键 image / file / path\n"
               "parquet = 路径列 image_col 或二进制列 bytes_col（需要 pyarrow）\n"
               "webdataset = 未压缩 .tar，path 用 glob，成员 <key>.<ext>\n"
               "moss_shard = 原生格式（catalog.db + shards/，需先用 importer 生成）"),
    param("TXT_AS", "字幕字段类型", "enum", "tags", "advanced", "数据集", native=False,
          options=enum(["tags", "nl"]), help_style="info",
          help="决定 .txt 进入哪个字段：tags = 按逗号拆成标签（默认，配 template: tags）；"
               "nl = 整段文本作为自然语言（配 template: nl 或 mix）。"),
    param("NUM_WORKERS", "数据加载进程数", "int", 4, "advanced", "数据集", min=0, max=64, native_key="NUM_WORKERS"),
    param("SEED", "随机种子", "int", 0, "basic", "训练", min=0, max=4294967295, native_key="SEED"),

    # ---- basic / 算法 ----
    param("ALGO", "适配器算法", "enum", "lora", "basic", "网络", options=ALGOS, native_key="ALGO",
          help="来自 LyCORIS：lora/locon/loha/lokr/oft/boft/glora/ia3/dylora/tlora，或 full（全量微调）。"),
    param("PRESET", "作用层预设", "enum", "attn-mlp", "basic", "网络", options=PRESETS, native_key="PRESET",
          detail=r"""作用：选择适配器（LoRA 等）挂到哪些层。四档只按「模块类型」选，覆盖全部 42 层、
不区分层号；要按层号只训一部分，请用下面的「自定义作用层」。

attn        每层注意力的 q / k / v / o 投影
attn-mlp    再加每层 FFN 的 ffn_up_gate / ffn_down（默认）
full-block  再加 2 个 text-refiner 块的线性层
all         变换器里所有线性层（含 time / shared / head modulation、video embedder / head）

引擎还认 +text-refiner / +modulation / +embed 三个修饰符（写作 attn-mlp+modulation）：
  · +text-refiner 由「训练文本细化块」开关自动追加，不用在这里选
  · +modulation、+embed 在 LP 没有对应选项，需要时请到「自定义作用层」直接写正则

示例：只训第 10–29 层的注意力（容量和显存都省），填在「自定义作用层」：
{"target_name": ["layers\\.(1[0-9]|2[0-9])\\.proj\\.(q|k|v|o)$"]}

注意：算法选 full（全量微调）时不建适配器，本项与「自定义作用层」都不生效。"""),
    param("PRESET_CUSTOM", "自定义作用层", "string", "", "advanced", "网络", native=False,
          help="留空则用上面的预设。填写 LyCORIS 风格的目标字典（JSON 或 Python 字面量），会覆盖预设。",
          detail=r"""作用：只训练模型里匹配指定路径的层，而不是整组层。留空时使用「作用层预设」；
填写后覆盖它（整体替换，不是叠加）。

取值：一个字典，键与 LyCORIS 预设一致
  · target_name    要训练模块的正则列表（必填）
  · exclude_name   要排除的正则列表（可选）
  · use_fnmatch    为 true 时按 fnmatch 通配符匹配，而不是正则（可选）
  · module_algo_map / name_algo_map   按模块或名字指定算法（可选，进阶）

匹配对象是模块路径，例如：
  layers.14.proj.q           第 14 层的注意力 q 投影
  layers.14.proj.ffn_down    第 14 层的前馈下投影
  text_embedder.blocks.3.qkv 文本编码器第 3 块

示例一 只训第 10–29 层的注意力（显存和容量都省）：
{"target_name": ["layers\\.(1[0-9]|2[0-9])\\.proj\\.(q|k|v|o)$"]}

示例二 只训前馈、但排除下投影：
{"target_name": ["layers\\.\\d+\\.proj\\.ffn_up_gate$"], "exclude_name": [".*ffn_down$"]}

示例三 只在第 30 层以后训练注意力与 MLP：
{"target_name": ["layers\\.(3[0-9]|[4-9][0-9])\\.proj\\."]}

示例四 通配符写法（use_fnmatch）：
{"target_name": ["layers.*.proj.q"], "use_fnmatch": true}

注意：
  · 上面是 JSON 写法，反斜杠要写两次（\\.）；改用 Python 字面量时只写一次（\.）
  · 正则写错会在「检测参数」时直接报错，不会等到训练阶段
  · 填了本项后，上面的「作用层预设」不再生效，也不会自动追加 +text-refiner；
    要连文字细化块一起训，请把 text_embedder 的正则也写进 target_name（写法同上）
  · 引擎用 re.match 从头匹配，结尾务必写 $，否则 layers\.1 会连第 10–19 层一起命中
  · exclude_name 同样从头匹配整条路径：要排除 ffn_down 得写 .*ffn_down$，
    只写 ffn_down$ 匹配不到（引擎不会报错，只是排除不生效）
  · 字典里没有 norm_name 键时，「训练归一化层」开关不会训任何 norm；
    要训 norm 请写 "norm_name": ["layers\\.\\d+\\.(norm1|norm2)$"]
  · 匹配到的层越少，可训练参数越少：收敛更快，但学到的细节也更少"""),
    param("DIM", "Rank / 块大小", "int", 32, "basic", "网络", min=1, max=1024, native_key="DIM",
          visible_when=("ALGO", RANKED_ALGOS),
          help="lora/locon/loha/lokr/glora/dylora/tlora 下是低秩 rank；oft/boft 下是 OFT 块大小。"),
    param("ALPHA", "Alpha", "float", 32.0, "basic", "网络", min=0.0, max=1024.0, native_key="ALPHA",
          visible_when=("ALGO", RANKED_ALGOS),
          help="低秩算法下是缩放系数（强度 ≈ alpha/dim）；oft/boft 下是 constraint。"),
    _json("ALGO_KWARGS", "算法附加参数", "{}", "网络", "JSON；如 lokr 的 factor、oft 的 constraint。", group="basic"),
    param("DORA", "DoRA", "bool", False, "advanced", "网络", native_key="DORA"),
    _json("DROPOUT", "Dropout", '{"input": 0.0, "rank": 0.0, "module": 0.0}', "网络", "JSON；输入/秩/模块三种丢弃率。"),
    param("ADAPTER_MODE", "适配器前向", "enum", "bypass", "advanced", "网络",
          options=enum(["bypass", "merge"]), native_key="ADAPTER_MODE"),
    param("LORAPLUS_RATIO", "LoRA+ 学习率比", "float", 1.0, "advanced", "网络", min=0.0, max=10.0,
          native_key="LORAPLUS_RATIO", help="1.0 表示关闭。"),

    # ---- basic / 优化 ----
    param("OPTIMIZER", "优化器", "enum", "adamw", "basic", "优化", options=OPTIMIZERS, native_key="OPTIMIZER"),
    param("LR", "学习率", "float", 0.0001, "basic", "优化", min=0.00000001, max=1.0, native_key="LR"),
    param("WEIGHT_DECAY", "权重衰减", "float", 0.01, "advanced", "优化", min=0.0, max=1.0, native_key="WEIGHT_DECAY"),
    param("WD_RULES", "衰减规则", "enum", "moss", "advanced", "优化", options=enum(["moss", "none", "all"]),
          native_key="WD_RULES", help="moss：bias/norm/1-D 不衰减，最后一个 q/k norm 向 1 衰减。"),
    _json("BETAS", "Adam β", "[0.9, 0.95]", "优化", "JSON 数组。"),
    _json("SCHEDULE", "学习率调度", '{"mode": "cosine", "warmup": 100, "end": -1}', "优化",
          "JSON；mode/warmup/end。"),
    param("GRAD_CLIP", "梯度裁剪", "float", 1.0, "basic", "优化", min=0.0, max=100.0, native_key="GRAD_CLIP"),
    param("EMA", "EMA 衰减", "float", 0.0, "advanced", "优化", min=0.0, max=0.99999, native_key="EMA",
          help="0 表示关闭。"),
    param("GRAD_ACC", "梯度累积步数", "int", 1, "basic", "训练", min=1, max=1024, native_key="GRAD_ACC"),
    param("BATCHING", "批次方式", "enum", "tokens", "basic", "训练",
        options=enum(["tokens", "tokens_ffd", "count"]), native_key="BATCHING",
        help="tokens=按 token 数拼批（默认）；count=按 BATCH_SIZE 固定样本数。"),
    param("MAX_TOKENS", "每批最大 token", "int", 16384, "basic", "训练",
        min=256, max=131072, native_key="MAX_TOKENS",
        help="每个批次（图像+文本）的目标 token 数；显存不足时先降低它（如 4096）。"),
    param("BATCH_SIZE", "每批样本数", "int", 4, "basic", "训练",
        min=1, max=256, native_key="BATCH_SIZE",
        help="仅当批次方式为 count 时生效。"),
    param("MAX_SAMPLES", "每批样本上限", "int", 0, "basic", "训练",
        min=0, max=4096, native_key="MAX_SAMPLES",
        help="0 表示不限制。"),
    param("MICROBATCHES", "流水线微批次数", "int", 4, "advanced", "训练",
        min=1, max=64, native_key="MICROBATCHES",
        help="仅流水线并行（pipeline）时生效。"),
    param("STEPS", "训练步数", "int", 1000, "basic", "训练", min=1, max=10000000, native_key="STEPS"),

    # ---- 数据集（引擎 scripts/train/moss.py 的完整 DATA 段）----
    param("SIZE_STEP", "像素步长", "int", 16, "advanced", "数据集", min=1, max=64, native_key="SIZE_STEP"),
    param("ASPECT_RANGE", "宽高比范围", "string", "(0.25, 4.0)", "advanced", "数据集", native_key="ASPECT_RANGE",
          help="Python 元组，如 (0.25, 4.0)。"),
    param("UPSCALE", "放大策略", "enum", "never", "advanced", "数据集",
          options=enum(["never", "to_area"]), native_key="UPSCALE"),
    param("OVERLONG", "超长样本处理", "enum", "drop", "advanced", "数据集",
          options=enum(["drop", "truncate_text", "drop_tokens"]), native_key="OVERLONG"),
    param("TOKEN_BUDGET", "token 预算方式", "enum", "estimate", "advanced", "数据集",
          options=enum(["estimate", "render", "exact"]), native_key="TOKEN_BUDGET"),
    param("CACHE_DTYPE", "缓存精度", "enum", "bf16", "advanced", "数据集",
          options=enum(["bf16", "fp8"]), native_key="CACHE_DTYPE"),
    param("CAPTION_VARIANTS", "标题变体数", "int", 4, "advanced", "字幕", min=1, max=64,
          native_key="CAPTION_VARIANTS"),
    param("FLIP", "水平翻转增强", "bool", False, "advanced", "数据集", native_key="FLIP"),
    param("CROP", "裁剪方式", "enum", "center", "advanced", "数据集",
          options=enum(["center", "random"]), native_key="CROP"),
    param("PLANNER_KWARGS", "规划器覆盖项", "string", "{}", "advanced", "数据集", native_key="PLANNER_KWARGS",
          help="Python 字典，如 {'window': 64, 'drop_last': True}。"),
    param("LATENT_ENCODER_ID", "潜空间编码器 ID", "string", "flux2-vae", "advanced", "数据集", native_key="LATENT_ENCODER_ID"),
    param("LATENT_DRAW", "潜空间采样方式", "enum", "mode", "advanced", "数据集",
          options=enum(["mode", "sample"]), native_key="LATENT_DRAW"),
    param("TEXT_ENCODER_ID", "文本编码器 ID", "string", "qwen3.5-9b", "advanced", "数据集", native_key="TEXT_ENCODER_ID"),
    param("BACKGROUND", "透明图背景色", "string", "white", "advanced", "数据集", native_key="BACKGROUND"),
    param("RESAMPLE", "缩放滤镜", "string", "lanczos", "advanced", "数据集", native_key="RESAMPLE"),
    param("TEXT_CHARS_PER_TOKEN", "每 token 字符数", "float", 3.5, "advanced", "数据集", native_key="TEXT_CHARS_PER_TOKEN"),
    param("TEXT_OVERHEAD_TOKENS", "标题模板开销", "int", 12, "advanced", "数据集", min=0, max=1024, native_key="TEXT_OVERHEAD_TOKENS"),
    param("MAX_TEXT_TOKENS", "标题最大 token", "int", 4096, "advanced", "数据集", min=16, max=65536, native_key="MAX_TEXT_TOKENS"),
    param("PREFETCH", "预取批次数", "int", 2, "advanced", "数据集", min=0, max=64, native_key="PREFETCH"),
    param("PIN_MEMORY", "数据批次锁页内存", "bool", True, "advanced", "数据集", native_key="PIN_MEMORY",
          help="把数据加载器的批次放进锁页内存以加速 H2D 拷贝（引擎 PIN_MEMORY）。"
               "与「内存与加速」里的 PINNED（模型卸载用的锁页内存）不是同一个开关。"),
    param("DATA_RESCAN", "数据重扫策略", "enum", "false", "advanced", "数据集",
          options=enum(["false", "true", "never"]), native_key="DATA_RESCAN",
          help="false=只探测变化；true=全量重扫；never=不遍历（引擎仅接受 False/True/'never'）。"),
    param("DATA_SCAN_WORKERS", "扫描进程数", "int", 8, "advanced", "数据集", min=1, max=64, native_key="DATA_SCAN_WORKERS"),
    param("DATA_WAIT", "等待目录写入秒数", "float", 3600.0, "advanced", "数据集", native_key="DATA_WAIT"),

    # ---- 适配器 / 目标 / 优化（补全）----
    param("TRAIN_TEXT_REFINER", "训练文本细化块", "bool", False, "advanced", "训练", native_key="TRAIN_TEXT_REFINER"),
    param("DORA_DETACH", "DoRA 分离范数", "bool", False, "advanced", "训练", native_key="DORA_DETACH"),
    param("TRAIN_NORMS", "训练归一化层", "bool", False, "advanced", "训练", native_key="TRAIN_NORMS"),
    param("LOSS_KWARGS", "损失参数", "string", "{}", "advanced", "优化", native_key="LOSS_KWARGS"),
    param("OPTIMIZER_KWARGS", "优化器参数", "string", "{}", "advanced", "优化", native_key="OPTIMIZER_KWARGS"),
    param("EPS", "优化器 epsilon", "float", 1e-8, "advanced", "优化", native_key="EPS"),
    param("KEEP_FP32", "保持 FP32 的模块", "string", '("time_embedding", "shared_block_modulation", "video_head_modulation")',
          "advanced", "优化", native_key="KEEP_FP32", help="Python 元组，如 (\"time_embedding\",)。"),
    param("FP16_FIX", "FP16 修复", "enum", "none", "advanced", "优化",
          options=enum(["none", "online", "fp32_residual"]), native_key="FP16_FIX"),
    param("FP16_FIX_KWARGS", "FP16 修复参数", "string", "{}", "advanced", "优化", native_key="FP16_FIX_KWARGS"),

    # ---- 内存（补全）----
    param("STEP_RELEASE", "步间显存释放", "string", "none", "advanced", "内存与加速", native_key="STEP_RELEASE",
          help="none / empty_cache / 每 N 步（填数字）。"),
    param("FUSED_STEP", "反向内更新参数", "bool", False, "advanced", "内存与加速", native_key="FUSED_STEP"),
    param("BASE_PRECISION", "基础模型精度", "enum", "compute", "advanced", "内存与加速",
          options=enum(["compute", "int8", "fp8", "fp8_compute", "mxfp8", "quant"]), native_key="BASE_PRECISION"),
    param("FP8_RECIPE", "FP8 缩放策略", "enum", "row", "advanced", "内存与加速",
          options=enum(["tensor", "row", "block"]), native_key="FP8_RECIPE"),
    param("FP8_DGRAD", "FP8 输入梯度", "enum", "fp8", "advanced", "内存与加速",
          options=enum(["fp8", "16bit"]), native_key="FP8_DGRAD"),
    param("MEMORY_OVERRIDES", "显存计划覆盖项", "string", "{}", "advanced", "内存与加速", native_key="MEMORY_OVERRIDES",
          help="Python 字典，如 {'checkpoint': 'sub-block', 'stream': 'ring'}。"),

    # ---- 并行 / 内核（补全）----
    param("DIST_BACKEND", "分布式后端", "enum", "auto", "advanced", "并行", 
          options=enum(["auto", "nccl", "gloo"]), native_key="DIST_BACKEND"),
    param("PIPELINE_SCHEDULE", "流水线调度", "enum", "1f1b", "advanced", "并行",
          options=enum(["gpipe", "1f1b", "interleaved-1f1b", "looped-bfs", "interleaved-zb", "zbv", "dualpipe-v", "async"]),
          native_key="PIPELINE_SCHEDULE"),
    param("PIPELINE_SPLIT", "流水线切分", "string", "measured", "advanced", "并行", native_key="PIPELINE_SPLIT",
          help="measured / flops / even，或层数元组，如 (11, 10, 11, 10)。"),
    param("FSDP_REDUCE", "FSDP 归约时机", "enum", "step", "advanced", "并行",
          options=enum(["step", "micro"]), native_key="FSDP_REDUCE"),
    param("DDP_BACKEND", "DDP 实现", "enum", "kohakumoss", "advanced", "并行",
          options=enum(["kohakumoss", "torch"]), native_key="DDP_BACKEND"),
    param("DDP_REDUCE_DTYPE", "DDP 梯度精度", "enum", "fp32", "advanced", "并行",
          options=enum(["fp32", "param"]), native_key="DDP_REDUCE_DTYPE"),
    param("DIST_TIMEOUT", "集合通信超时秒", "int", 600, "advanced", "并行", min=10, max=86400, native_key="DIST_TIMEOUT"),
    param("DIST_GATE_TIMEOUT", "集合点超时秒", "int", 7200, "advanced", "并行", min=10, max=86400, native_key="DIST_GATE_TIMEOUT"),
    param("DIST_BIND", "进程组绑定", "enum", "lazy", "advanced", "并行",
          options=enum(["lazy", "device"]), native_key="DIST_BIND"),
    param("KERNELS", "内核后端覆盖", "string", "{}", "advanced", "并行", native_key="KERNELS",
          help="Python 字典，如 {'attention_varlen': 'triton'}。"),
    param("TUNE", "内核调优", "enum", "table", "advanced", "并行",
          options=enum(["table", "off"]), native_key="TUNE"),
    param("RETUNE_STEPS", "重调优步数", "int", 3, "advanced", "并行", min=0, max=100, native_key="RETUNE_STEPS"),
    param("DRIVER", "运行驱动", "enum", "native", "advanced", "并行",
          options=enum(["native", "lightning"]), native_key="DRIVER"),

    # ---- 保存 / 检查点（对齐 musubi 的「保存」分类）----
    param("SAVE_EVERY", "保存间隔步数", "int", 500, "advanced", "保存", min=0, max=1000000, native_key="SAVE_EVERY"),
    param("SAVE_MINUTES", "保存间隔分钟", "float", 0.0, "advanced", "保存", native_key="SAVE_MINUTES"),
    param("KEEP_LAST", "保留检查点数", "int", 3, "advanced", "保存", min=1, max=100, native_key="KEEP_LAST"),
    param("RESUME", "断点续训来源", "string", "", "advanced", "保存", native_key="RESUME",
          help="检查点目录，或填 latest。"),
    param("EXPORT_FORMATS", "导出格式", "string", '("comfy",)', "advanced", "保存", native_key="EXPORT_FORMATS",
          help="Python 元组，如 (\"comfy\", \"lycoris\")。"),
    param("EXPORT_READY", "训练后补导出可用 LoRA", "bool", True, "advanced", "保存", native=False,
          help="默认开启：训练成功后，LP 用引擎自己的导出器把每个 checkpoint 转成可直接加载的 LoRA。"
               "引擎只在训练结束时导出一次（entry.py:335），训练中的检查点由 LP 补做；关闭可省时间与磁盘。"),

    # ---- 日志（对齐 musubi 的「日志」分类）----
    param("PROJECT", "日志项目名", "string", "kohakumoss", "advanced", "日志", native_key="PROJECT"),
    param("PROGRESS", "显示进度条", "bool", True, "advanced", "日志", native_key="PROGRESS"),
    param("LOG_EVERY", "日志间隔（步）", "int", 10, "advanced", "日志", min=1, max=100000, native_key="LOG_EVERY"),
    param("LOGGER", "日志后端", "enum", "console", "advanced", "日志",
          options=enum(["console", "board", "wandb"]), native_key="LOGGER"),
    param("RUN_NAME", "运行名称", "string", "", "advanced", "日志", native_key="RUN_NAME"),
    param("NONFINITE_PATIENCE", "非有限值容忍步数", "int", 8, "advanced", "日志", min=0, max=1000, native_key="NONFINITE_PATIENCE"),
    param("TRACE_STEPS", "profiler 步区间", "string", "()", "advanced", "日志", native_key="TRACE_STEPS",
          help="Python 元组，如 (10, 20)；空 () = 关闭。"),
    param("PROFILE_STEPS", "CUPTI 步区间", "string", "()", "advanced", "日志", native_key="PROFILE_STEPS"),
    # ---- 采样（对齐 musubi：「采样」分类，预览图 tab 的 preview 组）----
    param("SAMPLE_PROMPTS", "采样提示词", "string", "[]", "preview", "采样", native_key="SAMPLE_PROMPTS",
          help="Python 列表，如 ['a cat']；空表示不生成预览。"),
    param("SAMPLE_NEGATIVE", "负向提示词", "string", "", "preview", "采样", native_key="SAMPLE_NEGATIVE"),
    param("SAMPLE_EVERY", "每 N 步出图", "int", 0, "preview", "采样", min=0, max=100000, native_key="SAMPLE_EVERY",
          help="0 表示关闭。"),
    param("SAMPLE_STEPS", "采样步数", "int", 50, "preview", "采样", min=1, max=200, native_key="SAMPLE_STEPS"),
    param("SAMPLE_CFG", "CFG", "float", 3.0, "preview", "采样", min=0.0, max=30.0, native_key="SAMPLE_CFG"),
    param("SAMPLE_SHIFT", "shift", "float", 5.0, "preview", "采样", native_key="SAMPLE_SHIFT"),
    param("SAMPLE_SKIP_LAYERS", "跳层", "string", "(22,)", "preview", "采样", native_key="SAMPLE_SKIP_LAYERS"),
    param("SAMPLE_SIZE", "出图尺寸", "string", "(1024, 1024)", "preview", "采样", native_key="SAMPLE_SIZE"),
    param("SAMPLE_SEEDS", "种子列表", "string", "(0,)", "preview", "采样", native_key="SAMPLE_SEEDS"),

    # ---- 验证（跟随检查点，放在「保存」）----
    param("VALIDATE_EVERY", "每 N 步验证", "int", 0, "advanced", "保存", min=0, max=100000, native_key="VALIDATE_EVERY"),
    param("VALIDATION", "验证集配置", "string", "{}", "advanced", "保存", native_key="VALIDATION"),

    # ---- advanced / 目标函数 ----
    param("TIMESTEP", "时间步采样", "enum", "moss", "advanced", "时间步与采样",
          options=enum(["moss", "uniform", "logit_normal"]), native_key="TIMESTEP",
          detail="""决定每一步抽到哪个噪声强度 σ（σ→1 高噪：构图、大形；σ→0 低噪：细节、画风纹理）。

moss           引擎自定义分布：截断 logit-normal + 均匀混合 + 平移，参数见下一项
uniform        σ 在 [0,1] 均匀，所有噪声强度等概率
logit_normal   标准 logit-normal，不受 TIMESTEP_KWARGS 影响

偏画风/细节：选 moss，把 mu 调低、cutoff 调低（见「时间步参数」的示例）。
偏构图/大形：把 mu 调高、cutoff 调高。"""),
    _json("TIMESTEP_KWARGS", "时间步参数", '{"mu": -1.09, "s": 0.95, "shift": 1.0, "cutoff": 0.95, "uniform_mix": 0.01}',
          "时间步与采样", "JSON；mu/s/shift/cutoff/uniform_mix。",
          detail="""只在 TIMESTEP = moss 时生效。默认值就是引擎预训练最后一期的取值。

mu           分布中心：越负越偏低噪（细节/画风），越正越偏高噪（构图/大形）
             引擎历程：0.0 → -1.45 → -1.09（当前默认）
s            分布宽度：越小越集中，越大越分散
shift        静态平移，默认 1.0（不改即为引擎默认）
cutoff       截断上限：σ 高于它的步不再抽到；0.95 = 丢掉最噪的 5%
             调到 0.85 会让低噪占比明显上升
uniform_mix  均匀成分占比：0.01 = 1% 的步仍均匀抽样，保留少量全频带覆盖

示例 偏画风（低噪更多）：
{"mu": -1.5, "s": 0.8, "shift": 1.0, "cutoff": 0.9, "uniform_mix": 0.02}

示例 偏构图（高噪更多）：
{"mu": -0.7, "s": 1.0, "shift": 1.0, "cutoff": 1.0, "uniform_mix": 0.01}

注意：偏得太狠会让另一半能力退化 —— 角色训练建议 mu 在 -1.3 ~ -1.6 之间试。"""),
    param("LOSS", "损失函数", "enum", "mse", "advanced", "时间步与采样",
          options=enum(["mse", "huber", "l1", "masked_mse"]), native_key="LOSS"),
    param("WEIGHTING", "加权方式", "enum", "none", "advanced", "时间步与采样",
          options=enum(["none", "x0", "eps"]), native_key="WEIGHTING",
          detail="""按噪声强度给损失加权，不改抽样分布，改的是每一步的"分量"。

none   等权（引擎默认）
x0     按 x0 预测加权：高噪步权重低、低噪步权重高
       → 想强化细节/画风、又不想缩小 σ 覆盖时用它
eps    按 eps 预测加权：与 x0 相反，高噪步占更大分量

注意：
· 加权方式不能和 LOSS=huber/l1 的掩码语义混用同一套预期，切换后建议先跑 100 步看 loss 曲线
· 与「时间步参数」的 mu 是两个独立杠杆：mu 改"抽到多少低噪"，x0 改"低噪算多少分"
· 与「权重衰减 WEIGHT_DECAY」无关（那个是优化器的正则项）"""),

    # ---- advanced / 显存与加速 ----
    param("MEMORY_PROFILE", "显存档位", "enum", "balanced", "advanced", "内存与加速",
          options=enum(["speed", "balanced", "low", "minimal", "auto", "manual"]), native_key="MEMORY_PROFILE",
          help="选 non-manual 档位时，下面的检查点/流式/固定内存/优化器状态等手动项会被忽略（引擎要求）。"),
    # ---- advanced / 显存与加速（手动项；仅在 MEMORY_PROFILE=manual 时生效）----
    param("CHECKPOINT", "检查点策略（manual）", "enum", "", "advanced", "内存与加速",
          options=enum(["", "none", "block", "sub-block", "selective"]), native_key="CHECKPOINT",
          help="仅显存档位为 manual 时生效；否则由档位决定。"),
    param("STREAM", "块权重流式（manual）", "enum", "", "advanced", "内存与加速",
          options=enum(["", "off", "ring", "swap"]), native_key="STREAM"),
    param("ACT_OFFLOAD", "激活卸载（manual）", "bool", False, "advanced", "内存与加速", native_key="ACT_OFFLOAD"),
    param("OPTIM_STATE", "优化器状态位置（manual）", "enum", "", "advanced", "内存与加速",
          options=enum(["", "gpu", "stream", "cpu"]), native_key="OPTIM_STATE"),
    param("MEMORY_AUTO_FORCE", "auto 无匹配时仍运行", "bool", False, "advanced", "内存与加速", native_key="MEMORY_AUTO_FORCE"),
    param("PINNED", "固定内存（manual）", "bool", True, "advanced", "内存与加速", native_key="PINNED",
          help="锁页主机内存；主机内存不足时会出现 cuMemHostAlloc 失败，请关闭。"),
    param("STREAM_RESIDENT", "流式设备槽位（manual）", "int", 0, "advanced", "内存与加速", min=0, max=64, native_key="STREAM_RESIDENT"),
    param("TEXT_ENCODE", "文本编码器计划", "string", "{}", "advanced", "内存与加速", native_key="TEXT_ENCODE",
          help="TEXT_ENCODE 只填计算字段（dtype/precision）；放置方式请用下面的「文本编码器放置」。"),
    param("GRAD_OFFLOAD", "梯度卸载（manual）", "bool", False, "advanced", "内存与加速", native_key="GRAD_OFFLOAD",
          help="跨 GRAD_ACC 微批次把梯度在主机 fp32 累加；省显存、慢一些。"),
    param("TEXT_DEVICE", "文本编码器设备", "string", "", "advanced", "内存与加速", native_key="TEXT_DEVICE",
          help="留空 = 引擎默认（cuda）；可填 cpu。"),
    param("IMAGE_ENCODER", "VAE 编码器工厂", "string", "kohakumoss.encoders.vae_encoder",
          "advanced", "数据集", native_key="IMAGE_ENCODER",
          help="引擎默认 kohakumoss.encoders.vae_encoder；只有替换编码器实现时才需要改。"),
    param("TEXT_ENCODER_PLAN", "文本编码器放置", "string", "{}", "advanced", "内存与加速",
          native_key="TEXT_ENCODER_KWARGS.plan",
          help="默认 {} = 引擎默认（cache_text.py:46）。JSON 或 Python 字面量均可，"
               "如 {\"pinned\": false} 或 {'pinned': False}；可选 memory: resident | stream | offload | cpu | auto；"
               "主机锁页内存不足时加 pinned: false。"),
    param("COMPILE", "torch.compile", "enum", "off", "advanced", "内存与加速",
          options=enum(["off", "block", "model"]), native_key="COMPILE"),
    param("DEVICES", "使用的 GPU", "string", "[0]", "advanced", "内存与加速", native_key="DEVICES",
          help="JSON 数组，如 [0] 或 [0, 1]。"),
    param("PARALLEL", "并行方式", "enum", "none", "advanced", "内存与加速",
          options=enum(["none", "ddp", "fsdp", "pipeline", "threaded"]), native_key="PARALLEL"),

    # ---- advanced / 自定义参数 ----
    param("extra_params", "自定义参数", "string", "", "advanced", "自定义参数", multiline=True, native=False,
          help="以 GLOBAL=value 的形式追加到生成的配置末尾（值为 Python 字面量）；与上面的键重复会被拒绝。"),
)
