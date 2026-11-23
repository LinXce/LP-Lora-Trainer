"""Checks that depend on the Anima trainer or its on-disk resources."""

from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess

from adapters.base import ValidationIssue


_SCHEMA_COMMAND = """
import json, sys
sys.path.insert(0, sys.argv[1])
from library.anima.trainer.cli import setup_parser, build_network_extras
from library.config import schema
schema.populate_schema(setup_parser(), extras=build_network_extras())
print(json.dumps({k: [v.type, v.choices, v.source] for k, v in schema.get_schema().items()}))
"""


def _run_validation_probe(python, source, code, timeout, *, settings=None):
    result = subprocess.run(
        [python, "-B", "-X", "utf8", "-c", code, source], cwd=source,
        input=settings, capture_output=True, check=True, encoding="utf-8", timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return json.loads(result.stdout.splitlines()[-1])


@lru_cache(maxsize=12)
def _extra_schema(python, source, revision):
    # Only custom flags need introspection. The engine parser does not load models.
    return _run_validation_probe(python, source, _SCHEMA_COMMAND, 45)


def validate_extra_keys(installation, custom):
    source = Path(installation.source_path)
    try:
        parser_sources = ("library/anima/trainer/cli.py", "library/config/schema.py",
                          "library/config/cli_args.py", "library/anima/training.py", "networks/__init__.py")
        revision = tuple((source / name).stat().st_mtime_ns for name in parser_sources)
        schema = _extra_schema(str(installation.python_executable), str(source.resolve()), revision)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return [ValidationIssue("extra_params", "无法读取当前 Anima 的参数定义，请检查引擎安装后重新验证自定义参数")]
    issues = []
    for key, value in custom.items():
        if key not in schema:
            issues.append(ValidationIssue("extra_params", f"当前 Anima 原生配置不支持 --{key}"))
            continue
        kind, choices, origin = schema[key]
        if origin == "network_module":
            # Network kwargs are stringified by the trainer, including booleans.
            continue
        expected = {"int": (int,), "float": (float, int), "bool": (bool,),
                    "str": (str,), "path": (str,)}.get(kind)
        wrong_type = (expected and (not isinstance(value, expected)
                                   or isinstance(value, bool) and kind in ("int", "float")))
        if wrong_type or kind.startswith("list[") or (choices and value not in choices):
            issues.append(ValidationIssue("extra_params", f"--{key} 的值不符合引擎参数定义（{kind}；允许值 {choices or '见参数类型'}）"))
    return issues


@lru_cache(maxsize=8)
def _accelerators(python, source, stamp):
    check = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
import torch
from library.runtime.backend import resolve_attention_mode
from networks.attention_dispatch import flash_attn_func
print(json.dumps([resolve_attention_mode("flash", torch) != "flash" or flash_attn_func is not None,
                  importlib.util.find_spec("triton") is not None]))
"""
    return _run_validation_probe(python, source, check, 30)


def accelerator_issues(installation, flash, compile_key):
    root = Path(installation.source_path)
    dispatch = root / "networks/attention_dispatch.py"
    if not (flash or compile_key) or not dispatch.is_file() or not getattr(installation, "python_executable", None):
        return []
    python = Path(installation.python_executable)
    site = python.parent / "Lib/site-packages"
    if not site.is_dir():
        site = python.parent
    try:
        flash_ready, triton_ready = _accelerators(str(python.resolve()), str(root.resolve()),
                                                  (dispatch.stat().st_mtime_ns, site.stat().st_mtime_ns))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return [ValidationIssue("attn_mode" if flash else compile_key, "引擎解释器无法加载 PyTorch 运行环境，请修复引擎安装")]
    issues = []
    if flash and not flash_ready:
        issues.append(ValidationIssue("attn_mode", "此引擎环境无法加载 Flash Attention 2。请安装与显卡和 PyTorch 匹配的上游构建，或手动选择 torch"))
    if compile_key and not triton_ready:
        issues.append(ValidationIssue(compile_key, "Inductor 编译需要 Triton，请修复该引擎的专属环境，或手动关闭 torch.compile"))
    return issues


@lru_cache(maxsize=24)
def _native_overrides(python, source, settings, stamp):
    command = """
import json, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from library.anima.trainer.cli import NETWORK_KWARG_ALLOWLIST
opts = json.load(sys.stdin)
try:
    if opts.get("network_module") == "networks.lora_anima":
        from networks.lora_anima.config import LoRANetworkCfg
        args = dict(x.split("=", 1) for x in (opts.get("network_args") or "").split())
        unknown = args.keys() - set(NETWORK_KWARG_ALLOWLIST)
        if unknown:
            raise ValueError("network_args: " + ", ".join(sorted(unknown)))
        for k in NETWORK_KWARG_ALLOWLIST:
            if k in opts and opts[k] is not None:
                args.setdefault(k, str(opts[k]))
        LoRANetworkCfg.from_kwargs(args, network_dim=opts.get("network_dim"),
            network_alpha=opts.get("network_alpha"), neuron_dropout=opts.get("network_dropout"),
            module_class=object, grad_basis_dict={})
    if opts.get("sigma_lowres") or opts.get("sigma_lowres_yarnsig"):
        from library.anima.trainer import sigma_route
        ns = SimpleNamespace(**opts)
        if opts.get("sigma_lowres"):
            sigma_route.validate_rules(ns)
            sigma_route.span_params(ns)
            sigma_route.span_params(ns,2)
        elif sigma_route.yarnsig_params(ns) is not None:
            raise ValueError("sigma_lowres_yarnsig requires sigma_lowres")
        sigma_route.yarnsig_params(ns)
except (KeyError, ValueError, TypeError) as ex:
    print(json.dumps(str(ex)))
else:
    print("null")
"""
    return _run_validation_probe(python, source, command, 45, settings=settings)


def _vocab_issue(source, supplied):
    if not supplied:
        return None
    raw = str(supplied).strip()
    if not raw:
        return None
    path = source / raw
    if path.is_dir():
        pairs = [p for p in path.glob("*.safetensors") if p.with_suffix(".json").is_file()]
        if len(pairs) == 1:
            return None
        return ValidationIssue("vocab_pack", "词表包目录必须包含唯一一对同名 .json 与 .safetensors 文件，也可以指定文件前缀")
    if path.suffix in (".safetensors", ".json"):
        path = path.with_suffix("")
    if all(path.with_suffix(suffix).is_file() for suffix in (".safetensors", ".json")):
        return None
    shipped = source / "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
    if path.resolve() == shipped.resolve():
        return ValidationIssue("vocab_pack", "官方 CJK 词表包尚未就绪，首次运行时引擎将尝试下载。离线运行请先在引擎目录执行 python tasks.py download-vocab-pack 或选择本地词表包。", "warning")
    return ValidationIssue("vocab_pack", f"找不到完整的 CJK 词表包，请检查 {path} 的 .json 与 .safetensors 文件")


def anima_issues(installation, values, check_native_overrides):
    def positive(key):
        item = values.get(key)
        return not isinstance(item, bool) and isinstance(item, (int, float)) and item > 0

    compile_key = "torch_compile" if values.get("torch_compile") and values.get("dynamo_backend") != "eager" else None
    issues = accelerator_issues(installation, values.get("attn_mode") == "flash", compile_key)
    vocab = _vocab_issue(Path(installation.source_path), values.get("vocab_pack"))
    if vocab:
        issues.append(vocab)
    if check_native_overrides or values.get("network_args"):
        source = Path(installation.source_path)
        parser = source / "networks/lora_anima/config.py"
        if parser.is_file() and getattr(installation, "python_executable", None):
            try:
                error = _native_overrides(str(Path(installation.python_executable).resolve()), str(source.resolve()),
                                          json.dumps(values, ensure_ascii=False, sort_keys=True), parser.stat().st_mtime_ns)
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                error = "无法在引擎解释器中验证，请检查引擎安装"
            if error:
                issues.append(ValidationIssue("network_args" if values.get("network_args") else "extra_params",
                                              f"原生 LoRA 配置校验失败：{error}"))
    if values.get("sigma_lowres") and not values.get("use_vae_cache"):
        issues.append(ValidationIssue("use_vae_cache", "σ 分辨率路由需要启用 VAE 缓存以生成低分辨率潜空间"))
    if values.get("masked_loss"):
        selected = str(values.get("mask_dir") or "").strip()
        if not selected or not (Path(installation.source_path) / selected).is_dir():
            issues.append(ValidationIssue("mask_dir", "启用掩码损失需要选择训练图像对应的掩码目录"))
    if values.get("network_train_text_encoder_only"):
        issues.append(ValidationIssue("network_train_text_encoder_only", "Anima 训练仅支持在 DiT 上训练适配器"))
    if positive("blocks_to_swap"):
        if values.get("unsloth_offload_checkpointing"):
            issues.append(ValidationIssue("unsloth_offload_checkpointing", "CPU 卸载检查点要求将交换块数设为 0"))
        if values.get("mem_reweight_mode"):
            issues.append(ValidationIssue("blocks_to_swap", "记忆重加权不支持块交换，请将交换块数设为 0"))
    if values.get("target_res"):
        tiers = str(values["target_res"]).split()
        if any(t not in {"512", "768", "896", "1024", "1280", "1536"} for t in tiers):
            issues.append(ValidationIssue("target_res", "分辨率档位应以空格分隔，可选 512、768、896、1024、1280、1536"))
    warmup = values.get("lr_warmup_steps")
    if isinstance(warmup, float) and warmup >= 1 and not warmup.is_integer():
        issues.append(ValidationIssue("lr_warmup_steps", "预热步数超过 1 时需要整数"))
    for key in ("network_args", "lr_scheduler_args", "optimizer_args"):
        text = values.get(key)
        if text and any(not arg.partition("=")[0] or not arg.partition("=")[2] for arg in str(text).split()):
            issues.append(ValidationIssue(key, "请按 key=value 填写每个附加参数"))
    if values.get("use_text_cache"):
        network = dict(arg.split("=", 1) for arg in str(values.get("network_args") or "").split() if "=" in arg)
        if values.get("cache_llm_adapter_outputs") and str(network.get("train_llm_adapter", values.get("train_llm_adapter"))).lower() == "true":
            issues.append(ValidationIssue("cache_llm_adapter_outputs", "训练 LLM adapter 需要实时编码，请关闭其输出缓存"))
        for key in ("qwen3_max_token_length", "t5_max_token_length"):
            if values.get(key) not in (None, 512):
                issues.append(ValidationIssue(key, "原生文本缓存按 512 token 编码，自定义长度需要关闭文本缓存"))
        if values.get("caption_extension") != ".txt":
            issues.append(ValidationIssue("caption_extension", "文本缓存阶段读取与图片同名的 .txt 字幕，请使用 .txt 或关闭文本缓存"))
        if positive("caption_tag_dropout_rate") or positive("token_warmup_step"):
            issues.append(ValidationIssue("use_text_cache", "文本缓存不能与实时标签丢弃或 token 预热同时使用"))
    if values.get("use_vae_cache") and (values.get("color_aug") or values.get("random_crop")):
        issues.append(ValidationIssue("use_vae_cache", "VAE 缓存不能与颜色增强或随机裁剪同时使用"))
    for key in ("sigma_lowres_route", "sigma_lowres_route2"):
        if values.get(key):
            try:
                native, demote = map(int, str(values[key]).split(":"))
                if native <= demote or demote <= 0:
                    raise ValueError
            except ValueError:
                issues.append(ValidationIssue("extra_params", f"--{key} 需要 NATIVE:DEMOTE，且 NATIVE > DEMOTE > 0"))
    mode = values.get("clip_mode", "fixed")
    if mode == "median" or values.get("clip_per_micro"):
        if mode == "median" and not positive("clip_k"):
            issues.append(ValidationIssue("clip_k", "中位数裁剪倍数必须大于 0"))
        window = values.get("clip_window", 50)
        if isinstance(window, (float, int)) and window < 10:
            issues.append(ValidationIssue("clip_window", "裁剪窗口必须覆盖引擎的 10 步预热"))
        if mode != "median" and values.get("clip_sigma_bands"):
            issues.append(ValidationIssue("clip_sigma_bands", "σ 分段裁剪需要使用中位数裁剪模式"))
    if values.get("clip_sigma_bands"):
        try:
            bands = [float(x) for x in str(values["clip_sigma_bands"]).split(",")]
            if not all(math.isfinite(x) and 0 < x < 1 for x in bands) or len(bands) != len(set(bands)):
                raise ValueError
        except ValueError:
            issues.append(ValidationIssue("clip_sigma_bands", "请填写不重复的 σ 边界，以英文逗号分隔，每个值应介于 0 和 1 之间"))
    return issues


def moss_issues(installation, config, values):
    issues = accelerator_issues(installation, False, "moss_compile" if values.get("moss_compile") else None)
    rows = values.get("moss_delta_rows", "all")
    token_split = values.get("moss_token_split", "off")
    grounded = values.get("moss_clause_hw") or values.get("moss_text_box_hw")
    if grounded:
        if values.get("moss_distill", "off") != "off" or values.get("moss_preserve_weight") or values.get("moss_trigger_registers"):
            issues.append(ValidationIssue("moss_clause_hw", "分句与文本框定位要求关闭蒸馏、保留损失和触发寄存器"))
        if token_split != "off":
            issues.append(ValidationIssue("moss_token_split", "分句与文本框定位要求使用默认分词"))
    if values.get("moss_clause_centres") and not values.get("moss_clause_hw"):
        issues.append(ValidationIssue("moss_clause_centres", "自定义分句中心需要启用分句位置权重"))
    if (rows == "trigger" or values.get("moss_trigger_registers")) and token_split != "off":
        issues.append(ValidationIssue("moss_token_split", "触发行与触发寄存器不支持拆分 token"))
    if rows != "all" and values.get("moss_mod_rank"):
        issues.append(ValidationIssue("moss_mod_rank", "分行更新不能同时训练调制层"))
    if values.get("moss_ref_adapters", "off") != "off":
        timestep = values.get("moss_ref_timestep")
        if not values.get("moss_ref_isolate") or isinstance(timestep, (int, float)) and timestep < 0:
            issues.append(ValidationIssue("moss_ref_adapters", "参考图适配层需要隔离参考图，且参考图时间步应大于等于 0"))
        if values.get("moss_algo") != "lora":
            issues.append(ValidationIssue("moss_ref_adapters", "参考图适配层仅支持 LoRA"))
    for key in ("moss_sample_steps", "moss_sample_every_epochs"):
        val = values.get(key)
        if isinstance(val, (float, int)) and val < (1 if key == "moss_sample_steps" else 0):
            issues.append(ValidationIssue(key, "采样步数应至少为 1，采样间隔应大于等于 0"))
    if values.get("moss_ema_extra"):
        try:
            spans = [float(s.strip()) for s in str(values["moss_ema_extra"]).split(",") if s.strip()]
            if not all(0 < s <= 1 for s in spans):
                raise ValueError
        except ValueError:
            issues.append(ValidationIssue("moss_ema_extra", "EMA 窗口需要以英文逗号分隔，且每个值应大于 0、小于等于 1"))
    if values.get("moss_sample_at_first") or values.get("moss_sample_every_epochs"):
        sample = str(values.get("moss_sample_prompts") or "").strip()
        if "\n" not in sample and sample.endswith((".txt", ".json")):
            try:
                path = Path(installation.source_path) / sample
                contents = path.read_text(encoding="utf-8")
                if sample.endswith(".json"):
                    data = json.loads(contents)
                    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
                        raise ValueError
                    contents = "".join(data)
                if not contents.strip():
                    raise ValueError
            except (OSError, ValueError):
                issues.append(ValidationIssue("moss_sample_prompts", "采样提示词文件必须可读并包含提示词，JSON 文件应为文本列表"))
    for key, suffix in (("moss_cache_skip_text", "te"), ("moss_cache_skip_latents", "latent")):
        if values.get(key):
            folder = Path(str(values.get("moss_cache_directory") or Path(config.get("output_dir") or "") / "moss-cache"))
            if not folder.is_dir() or not next(folder.rglob(f"*.moss_{suffix}.safetensors"), None):
                issues.append(ValidationIssue(key, "所选缓存目录没有对应的现有缓存，请先生成缓存或开启这个阶段"))
    return issues
