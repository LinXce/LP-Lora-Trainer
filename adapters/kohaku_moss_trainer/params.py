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
          help="四种布局都能直接用（与引擎 weights/layout.py 一致）：\n"
               "1) ComfyUI models/ 目录（含 diffusion_models/、text_encoders/、vae/）—— 引擎发布配置的默认布局；\n"
               "2) HF 布局目录（含 transformer/、text_encoder/、vae/）—— 编码器路径自动推导；\n"
               "3) 变体根目录（含 base/ 或 sft/ 子目录）；\n"
               "4) 单个 DiT .safetensors —— 必须同时填写下面的「文本编码器」与「VAE」。"),
    param("VARIANT", "模型变体", "enum", "base", "basic", "模型与编码器",
          options=enum(["base", "sft"]), native_key="VARIANT"),
    param("TRANSFORMER", "DiT 权重路径", "string", "", "basic", "模型与编码器",
          path_kind="file_or_directory", native_key="TRANSFORMER",
          help="留空 = 由 WEIGHTS 自动选择（ComfyUI 布局取 diffusion_models/ 下的默认文件）。"
               "可填 diffusion_models/ 下的文件名（含或不含 .safetensors）、单个 DiT 文件或 HF transformer 目录。"),
    _json("ARCH_OVERRIDES", "模型结构覆盖", "{}", "模型与编码器",
          "Python/JSON 字典；覆盖 MossArchConfig 字段，留空使用引擎默认。"),
    param("TEXT_ENCODER", "文本编码器路径", "string", "", "basic", "模型与编码器", path_kind="file_or_directory",
          native_key="TEXT_ENCODER",
          help="留空由 WEIGHTS 推导：ComfyUI 布局取 text_encoders/ 下的默认文件，HF 布局取 <WEIGHTS>/text_encoder。"
               "可填 text_encoders/ 下的文件名、编码器目录或单个 .safetensors。"),
    param("VAE", "VAE 路径", "string", "", "basic", "模型与编码器", path_kind="file_or_directory",
          native_key="VAE",
          help="留空由 WEIGHTS 推导：ComfyUI 布局取 vae/ 下的默认文件，HF 布局取 <WEIGHTS>/vae。"
               "可填 vae/ 下的文件名、VAE 目录或 .safetensors。"),
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
          options=enum(["tags", "nl", "mix", "json", "gbc"]), help_style="info",
          help="tags = 标签模板（读 .txt）；nl = 自然语言（读 .caption）；mix = 按 p_nl 混用；json = 结构化输出；"
               "gbc = KohakuGBC 图的字幕层（需要 kohakugbc-data，并用下方「原始 CAPTION 配置」）。"),
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
    param("CAPTION_JSON", "原始 CAPTION 配置", "string", "", "advanced", "字幕", native=False,
          help="可选：JSON / Python 字典，直接作为引擎的 CAPTION。留空时由上面的字段自动组装；"
               "GBC 模板需要在这里填完整配置（template/meta/tags/nl/graph/formats）。"),
    param("TEXT_DROP", "空提示词概率", "float", 0.1, "advanced", "数据集", min=0.0, max=1.0, native_key="TEXT_DROP"),
    param("LATENT_CACHE", "VAE 潜空间缓存", "enum", "offline", "basic", "数据集", options=CACHES,
          native_key="LATENT_CACHE", help="off / offline（训练前缓存）/ online（训练中编码）。"),
    param("TEXT_CACHE", "文本编码缓存", "enum", "offline", "basic", "数据集", options=CACHES, native_key="TEXT_CACHE"),
    # scripts/data/cache_text.py:50 -- captions per encoder pass; a big value leaves the
    # progress bar at 0 for a long time on a large/offloaded text encoder.
    param("ENCODE_BATCH", "文本编码批大小", "int", 512, "advanced", "数据集",
          min=1, max=4096, native_key="ENCODE_BATCH",
          help="每次编码推向文本编码器的标题数；显存/内存紧张或长时间没进度时调小（如 64）。"),
    param("ENCODE_DEVICES", "编码器 GPU", "string", "()", "advanced", "数据集", native_key="ENCODE_DEVICES",
          help="独立编码进程使用的 GPU，如 (4, 5)；留空 () 表示与训练同进程/同卡。"
               "填了之后文本编码器和 VAE 在编码进程运行，DEVICES 只跑训练。"),
    param("ENCODE_DEPTH", "编码预取深度", "int", 32, "advanced", "数据集", min=1, max=512,
          native_key="ENCODE_DEPTH", help="编码 feed：每条数据流领先训练 rank 的 micro-batch 数。"),
    param("ENCODE_CHUNK", "编码请求批数", "int", 8, "advanced", "数据集", min=1, max=128,
          native_key="ENCODE_CHUNK", help="编码 feed：每次编码请求包含的 micro-batch 数。"),
    param("VAE_ENCODE_BATCH", "潜空间编码批大小", "int", 8, "advanced", "数据集", min=1, max=512,
          native=False, help="cache_latents.py 的 ENCODE_BATCH；与文本编码批大小分开设置。"),
    param("VAE_ENCODE", "VAE 编码计划", "string", "{}", "advanced", "数据集", native_key="VAE_ENCODE",
          help="引擎 VaePlan 字段（cache_latents.py:54）：memory: resident|offload|stream|auto|cpu、"
               "batch、ahead、tile、compile: auto|on|off。默认 {} = 引擎默认；"
               "没有可用的 triton 时把 compile 设为 \"off\"，否则会报 TritonMissing。"),
    param("CACHE_DIR", "缓存目录", "string", "", "basic", "数据集", path_kind="directory", native_key="CACHE_DIR",
          help="留空为 <数据集>/cache。"),
    param("SOURCE", "数据集读取方式", "enum", "folder", "advanced", "数据集", native=False,
          options=enum(["kohya", "folder", "jsonl", "parquet", "webdataset", "moss_shard", "gbc"]),
          help_style="info",
          help="写入 DATASETS 条目的 source（三个训练阶段共用同一份 DATASETS）：\n"
               "folder = 任意目录树（默认，适合 LP 的递归扫描与平铺数据集），旁车文件 .txt / .caption / .json\n"
               "kohya = <重复次数>_<名称>/图片 + 同名 .txt / .caption（仅扫描一级子目录）\n"
               "jsonl = 每行一条记录，图片键 image / file / path\n"
               "parquet = 路径列 image_col 或二进制列 bytes_col（需要 pyarrow）\n"
               "webdataset = 未压缩 .tar，path 用 glob，成员 <key>.<ext>\n"
               "moss_shard = 原生格式（catalog.db + shards/，需要先用 importer 生成）\n"
               "gbc = KohakuGBC 数据集（dataset.json + shards/，需要 kohakugbc-data 和 gbc 字幕模板）"),
    param("TXT_AS", "字幕字段类型", "enum", "tags", "advanced", "数据集", native=False,
          options=enum(["tags", "nl"]), help_style="info",
          help="决定 .txt 进入哪个字段：tags = 按逗号拆成标签（默认，配 template: tags）；"
               "nl = 整段文本作为自然语言（配 template: nl 或 mix）。"),
    # 数据源参数：引擎 sources/kohya.py / sources/folder.py 的构造参数，只写进 DATASETS 条目。
    param("DATASET_INCLUDE_ROOT", "包含数据集根目录图片", "bool", False, "advanced", "数据集", native=False,
          visible_when=("SOURCE", ("kohya",)),
          help="kohya 源默认只扫一级子目录；打开后根目录里的图片也收录（重复次数按 1）。"),
    param("DATASET_REPEAT_PREFIX", "重复次数前缀解析", "enum", "first", "advanced", "数据集", native=False,
          options=enum(["first", "product", "ignore"]), visible_when=("SOURCE", ("kohya",)),
          help="文件夹名 <重复>_<名称> 的解析方式：first = 取第一个整数，product = 各整数相乘，"
               "ignore = 全部按 1（不做重复加权）。"),
    param("DATASET_ARTIST_FROM_FOLDER", "从文件夹名取画师", "string", "", "advanced", "数据集", native=False,
          visible_when=("SOURCE", ("kohya",)),
          help="留空 = 不取；name = 文件夹名整段作为画师；也可填正则（取第一个分组）。"
               "取到的画师会进入字幕最前的置前字段。"),
    param("DATASET_HASH_CONTENT", "按文件内容做缓存键", "bool", False, "advanced", "数据集", native=False,
          visible_when=("SOURCE", ("kohya", "folder", "jsonl", "parquet")),
          help="默认用路径+大小+mtime 做缓存键；打开后按内容哈希，移动文件或改时间不会让缓存失效，"
               "但首次扫描要读完整文件（大目录更慢）。jsonl / parquet 源同样支持。"),
    param("DATASET_RECURSIVE", "递归扫描子目录", "bool", True, "advanced", "数据集", native=False,
          visible_when=("SOURCE", ("folder",)),
          help="folder 源默认递归整棵目录树；关闭后只收数据集目录第一层的图片。"),
    param("DATASET_GBC_SHARDS", "GBC 分片列表", "string", "", "advanced", "数据集", native=False,
          visible_when=("SOURCE", ("gbc",)),
          help="可选：只读取这些分片，逗号分隔；留空读取全部分片。"),
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
          options=enum(["bf16", "fp16", "fp32", "fp8"]), native_key="CACHE_DTYPE"),
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
          help="none / empty_cache / 每 N 步释放一次；填数字（如 20）会写成整数 20，其余按字符串写入。"),
    param("FUSED_STEP", "反向内更新参数", "bool", None, "advanced", "内存与加速", native_key="FUSED_STEP"),
    param("BASE_PRECISION", "基础模型精度", "enum", "auto", "advanced", "内存与加速",
          options=enum(["auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8", "quant"]),
          native_key="BASE_PRECISION"),
    param("FP8_RECIPE", "FP8 缩放策略", "enum", None, "advanced", "内存与加速",
          options=enum(["tensor", "row", "block"]), native_key="FP8_RECIPE"),
    param("FP8_DGRAD", "FP8 输入梯度", "enum", None, "advanced", "内存与加速",
          options=enum(["fp8", "16bit"]), native_key="FP8_DGRAD"),
    param("INT8_COMPUTE", "INT8 计算", "bool", None, "advanced", "内存与加速", native_key="INT8_COMPUTE",
          help="int8 基础模型用 int8 GEMM（激活按行量化）；留空跟随引擎（False）。"),
    param("INT8_DGRAD", "INT8 输入梯度", "enum", None, "advanced", "内存与加速",
          options=enum(["int8", "16bit"]), native_key="INT8_DGRAD",
          help="INT8 计算下的输入梯度格式；留空跟随引擎（int8）。"),
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
    param("FSDP_RESHARD", "FSDP 重分片时机", "enum", "forward", "advanced", "并行",
          options=enum(["forward", "backward"]), native_key="FSDP_RESHARD",
          help="forward = ZeRO-3；backward = ZeRO-2，反向更省通信但更占显存。"),
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
    param("KEEP_LAST", "保留检查点数", "int", 3, "advanced", "保存", min=0, max=100, native_key="KEEP_LAST",
          help="只保留最新 N 个完整检查点；0 = 不删除（保留全部）。"),
    param("FINAL_CHECKPOINT", "结束时写续训检查点", "enum", "auto", "advanced", "保存",
          options=enum(["auto", "true", "false"]), native_key="FINAL_CHECKPOINT",
          help="auto = 仅当 SAVE_EVERY/SAVE_MINUTES 开启时写；true/false 强制。"),
    param("RESUME", "断点续训来源", "string", "", "advanced", "保存", native_key="RESUME",
          help="检查点目录，或填 latest。"),
    param("SAVE_FORMAT", "导出格式", "enum", "comfy", "advanced", "保存",
          options=enum(["comfy", "hf", "both", "none"]), native_key="SAVE_FORMAT"),
    param("SAVE_MERGED", "同时导出合并 DiT", "bool", False, "advanced", "保存", native_key="SAVE_MERGED"),
    param("SAVE_FUSED_QKV", "导出融合 QKV", "enum", "false", "advanced", "保存",
          options=enum(["auto", "true", "false"]), native_key="SAVE_FUSED_QKV",
          help="引擎默认 false（独立 q / k / v 键）；auto = 跟随训练 DiT 文件；true 强制融合成一个 proj.qkv。"),
    param("CHECKPOINT_EXPORT", "每个检查点都导出", "bool", True, "advanced", "保存", native_key="CHECKPOINT_EXPORT",
          help="开启时引擎把导出的 LoRA/DiT 也写进每个检查点的 export/，这是引擎自带行为，LP 不再另做补导出。"),
    param("EXPORT_FORMATS", "旧版导出格式", "string", None, "advanced", "保存", native_key="EXPORT_FORMATS",
          help="旧版 Python 元组（comfy / lycoris / merged / diffusers）；设置后由引擎映射并覆盖 "
               "SAVE_FORMAT/SAVE_MERGED，建议留空使用新字段。"),

    # ---- 日志（对齐 musubi 的「日志」分类）----
    param("PROJECT", "日志项目名", "string", "kohakumoss", "advanced", "日志", native_key="PROJECT"),
    param("PROGRESS", "显示进度条", "bool", True, "advanced", "日志", native_key="PROGRESS"),
    param("LOG_EVERY", "日志间隔（步）", "int", 10, "advanced", "日志", min=1, max=100000, native_key="LOG_EVERY"),
    param("METRICS_EVERY", "指标间隔（步）", "int", None, "advanced", "日志", min=1, max=100000,
          native_key="METRICS_EVERY", help="留空 = 跟随 LOG_EVERY。"),
    param("LOGGER", "日志后端", "enum", "console", "advanced", "日志",
          options=enum(["console", "wandb"]), native_key="LOGGER",
          help="console = 控制台 + metrics.jsonl；wandb 需要引擎环境的 wandb 组件（安装时可勾选）。"),
    param("RUN_NAME", "运行名称", "string", "", "advanced", "日志", native_key="RUN_NAME"),
    param("NONFINITE_PATIENCE", "非有限值容忍步数", "int", 8, "advanced", "日志", min=0, max=1000, native_key="NONFINITE_PATIENCE"),
    param("TRACE_STEPS", "profiler 步区间", "string", "()", "advanced", "日志", native_key="TRACE_STEPS",
          help="Python 元组，如 (10, 20)；空 () = 关闭。"),
    param("PROFILE_STEPS", "CUPTI 步区间", "string", "()", "advanced", "日志", native_key="PROFILE_STEPS"),
    # ---- 采样（对齐 musubi：「采样」分类，预览图 tab 的 preview 组）----
    param("SAMPLE_PROMPTS", "采样提示词", "string", "[]", "preview", "采样", native_key="SAMPLE_PROMPTS",
          help="Python 列表，如 ['a cat']（也可一行一条）；空表示不生成预览。"
               "SAMPLE_EVERY > 0 时至少需要一条（数据集有字幕时会自动补第一条）。"),
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
          detail="""决定每一步抽到哪个噪声强度 σ（σ→1 高噪：结构、构图、大形、画风；σ→0 低噪：细节、线宽、局部纹理）。

moss           引擎自定义分布：截断 logit-normal + 均匀混合 + 平移，参数见下一项
uniform        σ 在 [0,1] 均匀，所有噪声强度等概率
logit_normal   标准 logit-normal，不受 TIMESTEP_KWARGS 影响

偏结构/构图/画风：把 mu 调负（如 -1.4 ~ -1.7），或把 cutoff 调低。
偏细节/线宽：把 mu 调正（如 -0.7 ~ -0.3），或把 uniform_mix 提高到 5%–10%。"""),
    _json("TIMESTEP_KWARGS", "时间步参数", '{"mu": -1.09, "s": 0.95, "shift": 1.0, "cutoff": 0.95, "uniform_mix": 0.01}',
          "时间步与采样", "JSON；mu/s/shift/cutoff/uniform_mix。",
          detail="""只在 TIMESTEP = moss 时生效。默认值就是引擎预训练最后一期的取值。

mu           分布中心：越负 → σ 越高（结构/构图/大形/画风）；越正 → σ 越低（细节/线宽）
             引擎历程：0.0 → -1.45 → -1.09（当前默认）
s            分布宽度：越小越集中，越大越分散
shift        静态平移，默认 1.0（不改即为引擎默认）
cutoff       截断并重映射：调低会把剩余 σ 整体抬高（不是细节旋钮）；0.95 = 默认
uniform_mix  均匀成分占比：0.01 = 1% 的步均匀抽样；提高到 5%–10% 会增加低噪/全频带覆盖

示例 偏结构/构图/画风（高噪更多）：
{"mu": -1.5, "s": 0.8, "shift": 1.0, "cutoff": 0.9, "uniform_mix": 0.02}

示例 偏细节/线宽（低噪更多）：
{"mu": -0.3, "s": 1.0, "shift": 1.0, "cutoff": 1.0, "uniform_mix": 0.05}

注意：偏得太狠会让另一半能力退化 —— 角色训练建议 mu 在 -1.3 ~ -1.6 之间试。"""),
    param("LOSS", "损失函数", "enum", "mse", "advanced", "时间步与采样",
          options=enum(["mse", "huber", "l1", "masked_mse"]), native_key="LOSS"),
    param("WEIGHTING", "加权方式", "enum", "none", "advanced", "时间步与采样",
          options=enum(["none", "x0", "eps"]), native_key="WEIGHTING",
          detail="""按噪声强度给损失加权，不改抽样分布，改的是每一步的"分量"。

none   等权（引擎默认）
x0     按 x0 预测加权：高噪步权重低、低噪步权重高
       → 想强化细节/线宽、又不想缩小 σ 覆盖时用它
eps    按 eps 预测加权：与 x0 相反，高噪步占更大分量（偏结构/构图/画风）

注意：
· 加权方式不能和 LOSS=huber/l1 的掩码语义混用同一套预期，切换后建议先跑 100 步看 loss 曲线
· 与「时间步参数」的 mu 是两个独立杠杆：mu 改"抽到多少低噪"，x0 改"低噪算多少分"
· 与「权重衰减 WEIGHT_DECAY」无关（那个是优化器的正则项）"""),

    # ---- advanced / 显存与加速 ----
    param("MEMORY_PROFILE", "显存档位", "enum", "balanced", "advanced", "内存与加速",
          options=enum(["speed", "balanced", "low", "minimal", "auto", "manual"]), native_key="MEMORY_PROFILE",
          help="选 non-manual 档位时，下面的检查点/流式/固定内存/优化器状态等手动项会被忽略（引擎要求）。"
               "引擎还支持 catalog 档位名（如 ckpt-ring16-fused），用下面的「自定义档位名」填。"),
    param("MEMORY_PROFILE_NAME", "自定义档位名", "string", "", "advanced", "内存与加速", native=False,
          help="留空 = 用上面的下拉。填引擎 catalog 里的档位名（含 -ring<N> / -fused / -ckpt 等组合）会覆盖下拉；"
               "可运行 kohakumoss-config list --all（在引擎环境里）查看全部合法名字，名字写错会在训练阶段报错。"),
    # ---- advanced / 显存与加速（手动项；仅在 MEMORY_PROFILE=manual 时生效）----
    param("CHECKPOINT", "检查点策略（manual）", "enum", None, "advanced", "内存与加速",
          options=enum(["", "none", "block", "sub-block", "selective"]), native_key="CHECKPOINT",
          help="仅显存档位为 manual 时生效；否则由档位决定。要按比例检查点请用下面的「检查点比例」。"),
    param("CHECKPOINT_PARTIAL", "检查点比例（manual）", "float", None, "advanced", "内存与加速",
          min=0.01, max=1.0, native=False,
          help="填 0.5 = 只对一半 block 做检查点（写入引擎的 CHECKPOINT='partial:0.5'），"
               "与上面的「检查点策略」二选一：用了本项就把检查点策略留空（或保持 block）。"
               "仅显存档位为 manual 时生效。"),
    param("STREAM", "块权重流式（manual）", "enum", None, "advanced", "内存与加速",
          options=enum(["", "off", "ring", "swap"]), native_key="STREAM"),
    param("ACT_OFFLOAD", "激活卸载（manual）", "bool", None, "advanced", "内存与加速", native_key="ACT_OFFLOAD"),
    param("OPTIM_STATE", "优化器状态位置（manual）", "enum", None, "advanced", "内存与加速",
          options=enum(["", "gpu", "stream", "cpu"]), native_key="OPTIM_STATE"),
    param("MEMORY_AUTO_FORCE", "auto 无匹配时仍运行", "bool", False, "advanced", "内存与加速", native_key="MEMORY_AUTO_FORCE"),
    param("PINNED", "固定内存（manual）", "bool", None, "advanced", "内存与加速", native_key="PINNED",
          help="锁页主机内存；主机内存不足时会出现 cuMemHostAlloc 失败，请关闭。留空 = 由显存档位决定。"),
    param("STREAM_RESIDENT", "流式设备槽位（manual）", "int", None, "advanced", "内存与加速", min=0, max=64, native_key="STREAM_RESIDENT"),
    param("TEXT_ENCODE", "文本编码器计划", "string", "{}", "advanced", "内存与加速", native_key="TEXT_ENCODE",
          help="TEXT_ENCODE 只填计算字段（dtype/precision）；放置方式请用下面的「文本编码器放置」。"),
    param("GRAD_OFFLOAD", "梯度卸载（manual）", "bool", None, "advanced", "内存与加速", native_key="GRAD_OFFLOAD",
          help="跨 GRAD_ACC 微批次把梯度在主机 fp32 累加；省显存、慢一些。"),
    param("TEXT_DEVICE", "文本编码器设备", "string", "", "advanced", "内存与加速", native_key="TEXT_DEVICE",
          help="留空 = 引擎默认（cuda）；可填 cpu（文本编码器放主机内存）。"
               "该设置同时用于文本缓存阶段（cache_text），把 TE 从显存挪到内存。"),
    param("IMAGE_ENCODER", "VAE 编码器工厂", "string", "kohakumoss.encoders.vae_encoder",
          "advanced", "数据集", native_key="IMAGE_ENCODER",
          help="引擎默认 kohakumoss.encoders.vae_encoder；只有替换编码器实现时才需要改。"),
    param("TEXT_ENCODER_PLAN", "文本编码器放置", "string", "{}", "advanced", "内存与加速",
          native_key="TEXT_ENCODER_KWARGS.plan",
          help="默认 {} = 引擎默认（cache_text.py）。JSON 或 Python 字面量均可，"
               "如 {\"pinned\": false} 或 {'pinned': False}；可选 memory: resident | stream | offload | cpu | auto；"
               "主机锁页内存不足时加 pinned: false。"),
    param("COMPILE", "torch.compile", "enum", "off", "advanced", "内存与加速",
          options=enum(["off", "block", "model"]), native_key="COMPILE",
          help="block = 逐块编译。当前 Torch 与激活检查点（balanced/low/minimal 或 CHECKPOINT）同时开启时，"
               "反向重算会抛 CheckpointError；请保持 off，或改用无激活检查点的 MEMORY_PROFILE=speed。"),
    param("DEVICES", "使用的 GPU", "string", "[0]", "advanced", "内存与加速", native_key="DEVICES",
          help="JSON 数组，如 [0] 或 [0, 1]。"),
    param("PARALLEL", "并行方式", "enum", "none", "advanced", "内存与加速",
          options=enum(["none", "ddp", "fsdp", "pipeline", "threaded"]), native_key="PARALLEL"),

    # ---- advanced / 自定义参数 ----
    param("extra_params", "自定义参数", "string", "", "advanced", "自定义参数", multiline=True, native=False,
          help="以 GLOBAL=value 的形式追加到生成的配置末尾（值为 Python 字面量）；与上面的键重复会被拒绝。"),
)
