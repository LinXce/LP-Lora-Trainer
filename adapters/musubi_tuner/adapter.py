"""Musubi Tuner adapter: caching pipeline + LoRA training.

Musubi requires pre-computed latent and Text-Encoder caches, so a job runs three
stages in order (latents -> text encoder -> train). No engine imports here.
"""
import ast
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path

from adapters.base import (
    InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact,
    ResumeSource,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.recovery import accelerate_resume_sources
from adapters.sampling import prompt_file, write_prompts, exported_prompt_files
from adapters.musubi_tuner.params import ARCHITECTURES, NETWORK_MODULES, PARAMS, architecture_params
from adapters.musubi_tuner.extra_options import EXTRA_CHOICES, EXTRA_OPTIONS, EXTRA_RANGES
from adapters.musubi_tuner.pipeline import PIPELINES, Pipeline, pipeline_for
from adapters.musubi_tuner.precision import weight_dtype
from adapters import extra_params
from app.schemas.training import LaunchSpec
from app.services.common import safe_file_stem
from app.services.jobs import JobFiles

# The upstream trainer needs an explicit attention backend.
_ATTENTION_KEYS = ("sdpa", "flash_attn", "sage_attn", "xformers", "flash3")
# Engine defaults used when a value is absent from the form.
_PARAM_DEFAULTS = {p.key: p.default for p in PARAMS}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".jxl"}
# Windows-reserved characters that must never reach the model file stem. Quotes are
# not translated: the pure export contract keeps names such as `角色 "test"` verbatim.
_UNSAFE_FILENAME = re.compile(r'[<>:/\\|?*\x00-\x1f]')


class MusubiTunerAdapter(EngineAdapter):
    custom_write_dir_keys = ("logging_dir", "cache_directory")
    model_input_keys = tuple(p.key for p in PARAMS if p.path_kind in ("file", "file_or_directory")) + (
        "control_directory", "krea2_tokenizer",
    )

    def legacy_output_issue(self, config):
        return extra_params.legacy_output_issue(config, self._output_name(config))

    engine_id = "musubi_tuner"
    display_name = "Musubi Tuner"
    clone_url = "https://github.com/kohya-ss/musubi-tuner.git"
    installation_source_help = "根据当前项目声明的 CUDA extra 选择对应的 PyTorch 版本。"
    installation_help = "按项目锁定的版本和下载源安装到引擎解释器；保留其他已安装的包。"
    default_installation_source = "existing"
    adapter_version = "0.2.0"
    environment_strategy = "embedded"
    environment_dir = ".lp/python"
    environment_candidates = (".venv", "venv")
    python_version = "3.10"
    torch_index_mode = "named"
    # Accelerate state dirs (save_state) end with this suffix and hold the full model
    # and optimizer, so their weights are never LoRA artifacts.
    _STATE_DIR_SUFFIX = "-state"
    # Musubi's trainers run os.makedirs(output_dir, exist_ok=True) right before the
    # first checkpoint/state save and create "<output_dir>/sample" for the first
    # preview image, so the task folder only appears once there is real output.
    creates_output_dir = True
    diagnostic_modules = ("accelerate", "transformers", "safetensors", "diffusers", "toml")
    native_format = "toml"
    submittable = True
    training_notice = "Musubi Tuner 支持缓存流水线与训练；缓存写入任务目录"

    def installation_sources(self, source_path):
        import re
        import tomllib
        try:
            metadata = tomllib.loads((source_path / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return ()
        extras = metadata.get("project", {}).get("optional-dependencies", {})
        return tuple(sorted(key for key in extras if re.fullmatch(r"cu\d{3}", key)))

    @staticmethod
    def _torch_index_for_extra(source_path, extra):
        """Return the named uv index declared for the selected CUDA extra.

        The upstream clone owns this mapping.  Keeping it dynamic means LP can
        work with newer Musubi releases without hard-coding their index names.
        """
        import tomllib

        try:
            metadata = tomllib.loads(
                (Path(source_path) / "pyproject.toml").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return None

        source_entries = (
            metadata.get("tool", {})
            .get("uv", {})
            .get("sources", {})
            .get("torch", [])
        )
        if isinstance(source_entries, dict):
            source_entries = [source_entries]
        for entry in source_entries:
            if isinstance(entry, dict) and entry.get("extra") == extra:
                index = entry.get("index")
                if isinstance(index, str) and index.strip():
                    return index.strip()
        return None

    def installation_plan(
        self,
        source_path,
        python,
        torch_source,
        uv=None,
        torch_mirror_url=None,
    ):
        if not self.detect(source_path).matched:
            raise ValueError("Musubi Tuner 源码不完整")
        if not uv:
            raise ValueError("Musubi Tuner 的官方安装流程需要 uv")

        # Musubi owns its dependency graph in pyproject.toml. LP only invokes
        # the project's documented uv workflow; it must not install torch or
        # requirements through a shared, guessed pip command.
        sources = self.installation_sources(source_path)
        if torch_source != "existing" and torch_source not in sources:
            raise ValueError(
                f"当前 Musubi Tuner 未声明 {torch_source} 安装配置；"
                f"可用配置：{', '.join(sources) or '无 CUDA extra'}"
            )
        extra = None if torch_source == "existing" else torch_source

        if getattr(self, "embedded_mode", False):
            helper = Path(__file__).resolve().parents[2] / "scripts" / "install_uv_project.py"
            command = [python, "-u", str(helper), "--uv", str(uv), "--project",
                       str(source_path.resolve()), "--python", str(python), "--no-default-groups"]
            if extra:
                command.extend(["--extra", extra])
            else:
                command.append("--keep-installed-torch")
            if torch_mirror_url:
                if not extra:
                    raise ValueError("未选择 CUDA extra 时不能指定 PyTorch 镜像源")
                command.extend(["--torch-index", torch_mirror_url.rstrip("/")])
            return InstallationPlan("Musubi Tuner：按所选 CUDA 配置安装项目依赖", (tuple(command),))

        command = [uv, "sync"]
        if extra:
            command.extend(("--extra", extra))
            if torch_mirror_url:
                index_name = self._torch_index_for_extra(source_path, extra)
                if not index_name:
                    raise ValueError(
                        f"Musubi Tuner 未为 {extra} 声明可覆盖的 PyTorch uv index，无法为该 CUDA extra 指定自定义 PyTorch/CUDA wheel 镜像源"
                    )
                # uv's --index NAME=URL overrides the named index without
                # changing the copied project's pyproject.toml or uv.lock.
                command.extend(("--index", f"{index_name}={torch_mirror_url}"))
        elif torch_mirror_url:
            raise ValueError("未选择 CUDA extra（沿用现有环境）时不能指定 PyTorch/CUDA wheel 镜像源")
        return InstallationPlan(
            "Musubi Tuner 官方 uv 项目安装流程",
            (tuple(command),),
        )

    def detect(self, source_path):
        import tomllib
        metadata = source_path / "pyproject.toml"
        try:
            project = tomllib.loads(metadata.read_text(encoding="utf-8")).get("project", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return DetectionResult(self.engine_id, False, "Missing or invalid Musubi project metadata")
        package = source_path / "src" / "musubi_tuner"
        entries = any(source_path.glob("*_train_network.py"))
        matched = project.get("name") == "musubi-tuner" and package.is_dir() and entries
        return DetectionResult(self.engine_id, matched, "Musubi project metadata, package and training entry")

    def _script_roots(self, source_path):
        # The package path is what the upstream docs invoke; the repo root may only
        # hold mirrored entry stubs, so prefer src/musubi_tuner when it exists.
        return (Path(source_path) / "src" / "musubi_tuner", Path(source_path))

    def _script(self, source_path, name):
        for root in self._script_roots(source_path):
            if (root / name).is_file():
                return root / name
        return Path(source_path) / "src" / "musubi_tuner" / name

    def architectures(self, source_path):
        roots = self._script_roots(source_path)
        return tuple(a for a in ARCHITECTURES if any((r / a.script).is_file() for r in roots))

    def capabilities(self, installation):
        return self.capabilities_for_architecture(installation, None)

    def capabilities_for_architecture(self, installation, architecture):
        available = tuple(a.id for a in self.architectures(installation.source_path))
        surface = architecture_params(architecture) if architecture else PARAMS
        # Aggregate by parameter, not by pipeline: shared keys such as VAE
        # are required by several architectures, and later pipelines must not
        # overwrite the requirements of earlier ones.
        required_for = {
            p.key: tuple(
                architecture for architecture in available
                if architecture in PIPELINES and (p.key in ("optimizer_type", "caption_extension")
                                                   or p.key in PIPELINES[architecture].required)
            )
            for p in surface
        }
        params = tuple(
            replace(p, required_for=required_for.get(p.key, p.required_for))
            for p in surface
        )
        module_architectures = {}
        for arch in available:
            module_architectures.setdefault(NETWORK_MODULES[arch], []).append(arch)
        return dict(installation_id=installation.installation_id,
                    architectures=list(available),
                    architecture_by_network_module={module: members[0] for module, members in module_architectures.items()
                                                    if len(members) == 1},
                    params=specs(params), submittable=self.submittable)

    @staticmethod
    def _param(key):
        return next((p for p in PARAMS if p.key == key), None)

    @staticmethod
    def _output_name(config):
        """Model file stem: the engine appends ``-step…``/``-state`` to it.

        Path separators and reserved characters are replaced so a task name can
        never create subdirectories or escape ``output_dir`` when the trainer
        saves; quotes are left verbatim for the pure export contract.
        """
        name = config.get("output_name") or config.get("name") or "lora"
        if config.get("output_layout") == "task_id":
            return safe_file_stem(name)
        text = _UNSAFE_FILENAME.sub("_", str(name))
        text = text.rstrip(" .")
        return text or "lora"

    @staticmethod
    def _effective_params(config):
        """Apply TOML overrides to settings shared by cache and training stages."""
        values = dict(config.get("params") or {})
        extra = extra_params.merge({}, str(values.get(extra_params.PARAM_KEY) or ""),
                                   bound_output=config.get("output_layout") == "task_id")
        for p in architecture_params(config.get("architecture")):
            name = p.native_key or p.key
            if p.native and name in extra:
                values[p.key] = extra[name]
        return values, extra

    @staticmethod
    def _extra_issues(architecture, surface, extra):
        modeled = {p.native_key or p.key for p in surface if p.native}
        kinds = {name: kind for (scope, kind), names in EXTRA_OPTIONS.items()
                 if scope in ("common", architecture) for name in names}
        for key, value in extra.items():
            if key == "sample_prompts":
                yield ValidationIssue("extra_params", "请在采样提示词字段填写文本，LP 会为各阶段生成对应的提示词文件。")
                continue
            if architecture == "qwen_image" and key in ("edit", "edit_plus") and value is True:
                yield ValidationIssue("extra_params", f"请使用模型版本字段选择编辑版本，--{key} 会被 model_version 覆盖。")
                continue
            if key in modeled:
                continue  # validated with its effective field value
            kind = kinds.get(key)
            if kind is None:
                yield ValidationIssue("extra_params", f"当前架构不接受训练附加参数 --{key}。")
                continue
            scalar = kind.removesuffix("[]")
            valid = isinstance(value, {"bool": bool, "int": int, "float": (float, int), "str": str}[scalar])
            if scalar in ("int", "float"):
                valid = valid and not isinstance(value, bool) and (not isinstance(value, float) or math.isfinite(value))
            if not valid or (key in EXTRA_CHOICES and value not in EXTRA_CHOICES[key]):
                yield ValidationIssue("extra_params", f"--{key} 的值与上游参数类型或可选值不匹配（{kind}）。")
            elif key in EXTRA_RANGES:
                low, high = EXTRA_RANGES[key]
                if value < low or (high is not None and value > high):
                    limit = f"{low} ～ {high}" if high is not None else f"至少 {low}"
                    yield ValidationIssue("extra_params", f"--{key} 超出上游范围（{limit}）。")

    @staticmethod
    def _derived_values(architecture, values):
        """Cache-stage flags implied by the architecture choice.

        Musubi's cache scripts take a separate ``--i2v`` from the training
        ``--task`` (and enable it silently when CLIP/image encoder is given), so
        deriving it keeps both stages consistent instead of caching the wrong
        condition.
        """
        if architecture == "wan":
            task = str(values.get("wan_task") or "")
            if "i2v" in task or "flf2v" in task or values.get("wan_clip"):
                return {"wan_i2v": True}
        elif architecture == "hunyuan_video_15":
            if values.get("hv15_task") == "i2v" or values.get("hv15_image_encoder"):
                return {"hv15_i2v": True}
        return {}

    @staticmethod
    def _overlaps(first, second):
        """True when two paths are equal or one contains the other (same drive).

        Comparing path prefixes is case-insensitive through ``os.path.normcase`` so a
        Windows ``H:/Data`` and ``h:/data`` are recognised as the same place.
        """
        try:
            left = Path(first).resolve().parts
            right = Path(second).resolve().parts
        except (OSError, RuntimeError, ValueError):
            return False
        if not left or not right or os.path.normcase(left[0]) != os.path.normcase(right[0]):
            return False
        common = min(len(left), len(right))
        return tuple(os.path.normcase(part) for part in left[:common]) == tuple(
            os.path.normcase(part) for part in right[:common])

    def _cache_directory_issues(self, installation, config):
        """Reject a custom cache dir that would pollute the inputs or the engine tree.

        Musubi writes ``*_<arch>.safetensors`` cache files into that directory. A cache
        inside the output folder is therefore picked up as a checkpoint artifact, one
        inside the dataset or the engine source writes into read-only inputs.
        """
        custom = str((config.get("params") or {}).get("cache_directory") or "").strip()
        if not custom:
            # The default lives in the job directory, which never collides.
            return ()
        try:
            cache = Path(custom).resolve()
        except (OSError, RuntimeError, ValueError):
            return (ValidationIssue("cache_directory", "缓存目录路径无效，请重新选择"),)
        if cache.parent == cache:
            return (ValidationIssue("cache_directory", f"缓存目录不能是磁盘根目录：{cache}"),)
        targets = (
            ("数据集目录", config.get("dataset_path")),
            ("控制图片目录", (config.get("params") or {}).get("control_directory")),
            ("输出目录", config.get("output_dir")),
            ("引擎源码", getattr(installation, "source_path", "")),
        )
        issues = []
        for label, target in targets:
            if not target:
                continue
            if self._overlaps(cache, target):
                issues.append(ValidationIssue(
                    "cache_directory", f"缓存目录不能与{label}相同或互相包含：{cache}"))
        return tuple(issues)

    # Musubi parses these as ``key=value`` tokens (optimizer/scheduler also
    # ``ast.literal_eval`` the value); anything else raises inside the trainer.
    _LIST_ARGUMENT_KEYS = ("optimizer_args", "lr_scheduler_args", "network_args")

    @classmethod
    def _list_argument_issues(cls, values):
        issues = []
        for key in cls._LIST_ARGUMENT_KEYS:
            raw = values.get(key)
            if raw in (None, "", False):
                continue
            tokens = raw if isinstance(raw, list) else str(raw).split()
            for token in tokens:
                text = str(token).strip()
                if not text:
                    continue
                parts = text.split("=")
                if len(parts) != 2 or not parts[0].strip():
                    issues.append(ValidationIssue(
                        key, f"每一项必须是 key=value（引擎按等号拆成两项，多一个或少一个都会报错）：{text}"))
                elif key in ("optimizer_args", "lr_scheduler_args"):
                    try:
                        ast.literal_eval(parts[1])
                    except (SyntaxError, ValueError, RecursionError):
                        issues.append(ValidationIssue(key, f"上游需要可解析的 Python 字面值：{text}"))
        return issues

    def validate(self, installation, config):
        issues = []
        architecture = config.get("architecture")
        available = {a.id for a in self.architectures(installation.source_path)}
        if architecture not in available:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        try:
            values, extra = self._effective_params(config)
        except ValueError as exc:
            return (ValidationIssue("extra_params", str(exc)),)
        surface = architecture_params(architecture)
        issues.extend(self._extra_issues(architecture, surface, extra))
        submitted = config.get("params", {})
        legacy_vae = architecture in ("minimax_h3", "hidream_o1")
        for p in PARAMS:
            if legacy_vae and p.key == "vae":
                continue  # Existing tasks can carry the old common VAE; report the migration warning below.
            if not p.applies_to(architecture) and submitted.get(p.key) not in (None, "", p.default):
                issues.append(ValidationIssue(p.key, "此参数不适用于所选底模类型"))
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in surface:
            if not p.native and p.unsupported_reason:
                continue
            if p.key == "optimizer_type" and "optimizer_type" in extra:
                custom = extra["optimizer_type"]
                if isinstance(custom, str) and "." in custom and all(part.isidentifier() for part in custom.split(".")):
                    continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
        if issues:
            return tuple(issues)
        if not values.get("optimizer_type"):
            issues.append(ValidationIssue("optimizer_type", "请选择优化器，Musubi 未提供可执行的默认优化器。"))
        if not values.get("lr_scheduler_type"):
            scheduler = values.get("lr_scheduler") or "constant"
            if scheduler == "adafactor" and values.get("optimizer_type") != "Adafactor":
                issues.append(ValidationIssue("lr_scheduler", "Adafactor 调度器需要 Adafactor 优化器。"))
            if scheduler == "constant" and values.get("lr_warmup_steps") and values.get("optimizer_type") != "Adafactor":
                issues.append(ValidationIssue("lr_warmup_steps", "constant 调度器不接受预热步数，请选择带预热的调度器或设为 0。"))
        suffix = values.get("caption_extension")
        images = config.get("dataset_path")
        if not suffix:
            issues.append(ValidationIssue("caption_extension", "请选择图片同名标注文件的扩展名，Musubi 的图片目录必须提供标注。"))
        elif images and Path(images).is_dir():
            try:
                targets = [p for p in Path(images).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
                           and p.is_file() and p.with_suffix(suffix).is_file()]
            except OSError:
                targets = []
            if not targets:
                issues.append(ValidationIssue("caption_extension", f"图片目录顶层没有找到同名 {suffix} 标注与图片的组合。"))
            controls = values.get("control_directory")
            if controls:
                try:
                    control_names = {p.stem for p in Path(controls).iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES}
                except OSError:
                    control_names = set()
                stems = {p.stem for p in targets}
                matched = set()
                for name in control_names:
                    base, _, index = name.rpartition("_")
                    if name in stems:
                        matched.add(name)
                    elif index.isdigit() and base in stems:
                        matched.add(base)
                if missing := sorted(stems - matched):
                    issues.append(ValidationIssue("control_directory", f"找不到与目标图 {missing[0]} 配对的控制图。"))
        # Bucket resolutions are engine-specific text, so they get their own check.
        for line in self.invalid_resolutions(values.get("bucket_resolutions")):
            issues.append(ValidationIssue("bucket_resolutions",
                                          f"分辨率格式不正确：{line}（应为 960x544 这样的尺寸）"))
        pipeline = pipeline_for(architecture)
        if pipeline is None:
            # The entry script exists but the cache/train pipeline is undeclared:
            # submitting would run an engine script without its mandatory caches.
            if architecture in available:
                issues.append(ValidationIssue("architecture", "该架构尚未声明缓存与训练流水线，无法提交训练"))
        else:
            for key in pipeline.required:
                p = self._param(key)
                # Required native inputs can lack an executable upstream default.
                value = values.get(key, p.default if p else None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(ValidationIssue(key, f"该架构需要填写：{p.label if p else key}"))
            if pipeline.latents == "" or pipeline.text_encoder == "":
                issues.append(ValidationIssue("architecture", "该架构尚未声明缓存脚本"))
        # Engine-wide, not per base model: the epoch retention maths multiplies by
        # --save_every_n_epochs, so leaving that empty crashes the trainer (None * N).
        if not values.get("save_every_n_epochs") and (values.get("save_last_n_epochs") or values.get("save_last_n_epochs_state")):
            issues.append(ValidationIssue(
                "save_every_n_epochs", "使用「保留最近 N 轮」前请先填写「保存间隔（轮）」：引擎的保留计算会用到它，未设置时会直接报错"))
        # The step-based retention has the same None arithmetic (train_utils.
        # get_remove_step_no uses --save_every_n_steps as a modulus).
        if not values.get("save_every_n_steps") and (values.get("save_last_n_steps") or values.get("save_last_n_steps_state")):
            issues.append(ValidationIssue(
                "save_every_n_steps", "使用「保留最近 N 步」前请先填写「保存间隔（步）」：引擎的保留计算会用到它，未设置时会直接报错"))
        selected = [key for key in _ATTENTION_KEYS if values.get(key)]
        if len(selected) != 1:
            issues.append(ValidationIssue(
                "sdpa", "请选择一种训练注意力实现（SDPA、FlashAttention、xFormers 或该架构支持的 FlashAttention 3）。"))
        if values.get("sage_attn"):
            issues.append(ValidationIssue("sage_attn", "当前 Musubi 训练入口不支持 SageAttention。"))
        if values.get("flash3") and architecture == "hunyuan_video":
            issues.append(ValidationIssue(
                "flash3", "HunyuanVideo 尚未支持 FlashAttention 3，请改用 SDPA 或 FlashAttention"))
        # Wan checks the user's precision before applying Accelerator's defaults.
        if architecture == "wan":
            stored = weight_dtype(config.get("base_model_path"))
            if stored in ("fp16", "bf16") and values.get("mixed_precision") not in (stored, "no"):
                issues.append(ValidationIssue("mixed_precision", f"Wan 底模为 {stored}，请选择 {stored} 或 no。"))
            if stored == "fp8" and values.get("fp8_scaled"):
                issues.append(ValidationIssue("fp8_scaled", "Wan 无法再次缩放已经是 FP8 的权重。"))
        if values.get("fp8_scaled") and not values.get("fp8_base"):
            issues.append(ValidationIssue("fp8_scaled", "FP8 缩放需要同时开启 FP8 底模。"))
        if architecture in ("krea2", "hidream_o1") and values.get("fp8_base") and not values.get("fp8_scaled"):
            issues.append(ValidationIssue("fp8_base", "该架构的 FP8 底模需要同时开启 FP8 缩放。"))
        if architecture == "minimax_h3":
            if not values.get("h3_one_frame"):
                issues.append(ValidationIssue("h3_one_frame", "MiniMax-H3 的图片目录需要开启单帧模式。"))
            if values.get("h3_task") in ("fl2va", "ref2va"):
                issues.append(ValidationIssue("h3_task", "当前数据集为普通图片目录，仅支持 t2va；其他任务需要带控制图或引用的数据集。"))
            if values.get("batch_size", 1) not in (None, 1):
                issues.append(ValidationIssue("batch_size", "MiniMax-H3 每批只支持 1 张图片，可用梯度累积增加有效批量。"))
            if extra.get("h3_teacher_matching"):
                issues.append(ValidationIssue("extra_params", "单帧教师匹配需要带参考图的教师条件缓存，当前普通图片目录无法生成。"))
            for name in ("h3_teacher_loss_dc_weight", "h3_teacher_loss_mag_weight", "h3_teacher_preservation_weight"):
                if name in extra and extra[name] != 1 and not extra.get("h3_teacher_matching"):
                    issues.append(ValidationIssue("extra_params", f"--{name} 仅可在教师匹配训练中设置。"))
            if extra.get("h3_guidance_loss_scale", 0) > 0 and not extra.get("h3_guidance_loss_uncond_cache"):
                issues.append(ValidationIssue("extra_params", "启用引导损失需要 h3_guidance_loss_uncond_cache 无条件缓存。"))
            if extra.get("h3_timestep_focus_prob", 0) > 0:
                low = (values.get("min_timestep") or 0) / 1000
                high = 1 if values.get("max_timestep") is None else values["max_timestep"] / 1000
                if not low <= extra.get("h3_timestep_focus_min", 0.4) < extra.get("h3_timestep_focus_max", 0.8) <= high:
                    issues.append(ValidationIssue("extra_params", "H3 时间步聚焦区间必须递增且位于训练时间步范围内。"))
            if values.get("fp8_base"):
                issues.append(ValidationIssue("fp8_base", "MiniMax-H3 不支持 FP8 底模。"))
            if values.get("block_swap_h2d_only") and values.get("blocks_to_swap") and not values.get("gradient_checkpointing"):
                issues.append(ValidationIssue("block_swap_h2d_only", "MiniMax-H3 的单向换块训练需要梯度检查点。"))
        if architecture == "framepack" and not values.get("fpack_one_frame"):
            issues.append(ValidationIssue("fpack_one_frame", "FramePack 的图片目录需要开启单帧模式。"))
        if architecture == "krea2":
            if values.get("krea2_convrot_int8") and (values.get("fp8_base") or values.get("krea2_turbo_dit")):
                issues.append(ValidationIssue("krea2_convrot_int8", "ConvRot INT8 与 FP8 底模、Turbo DiT 不兼容。"))
            if values.get("krea2_convrot_int8_bwd") == "int8" and not values.get("krea2_convrot_int8"):
                issues.append(ValidationIssue("krea2_convrot_int8_bwd", "INT8 反向传播需要开启 ConvRot INT8。"))
            if values.get("krea2_turbo_dit") and (values.get("blocks_to_swap") or values.get("krea2_turbo_lora")):
                issues.append(ValidationIssue("krea2_turbo_dit", "Turbo DiT 不支持换块或同时使用 Turbo LoRA。"))
            if values.get("krea2_turbo_dit_cache") and not values.get("krea2_turbo_dit"):
                issues.append(ValidationIssue("krea2_turbo_dit_cache", "常驻 Turbo 权重需要指定 Turbo DiT。"))
        # The cache stage derives --i2v from the task, but the CLIP conditioning it
        # writes cannot be produced without the CLIP encoder (Wan2.1 I2V/FLF2V only;
        # Wan2.2 A14B models do not use CLIP for I2V).
        wan_task = str(values.get("wan_task") or "")
        if architecture in ("wan", "hunyuan_video_15"):
            is_wan = architecture == "wan"
            flag = "wan_i2v" if is_wan else "hv15_i2v"
            encoder = "wan_clip" if is_wan else "hv15_image_encoder"
            needs_i2v = ("i2v" in wan_task or "flf2v" in wan_task) if is_wan else values.get("hv15_task") == "i2v"
            if needs_i2v and values.get(flag) is False:
                issues.append(ValidationIssue(flag, "所选图生视频任务需要启用图生视频缓存。"))
            if not needs_i2v and values.get(encoder):
                issues.append(ValidationIssue(encoder, "该编码器会让上游缓存切换至图生视频，请选择匹配的任务或清空该项。"))
        if architecture == "wan" and ("i2v" in wan_task or "flf2v" in wan_task) and "A14B" not in wan_task:
            if not str(values.get("wan_clip") or "").strip():
                issues.append(ValidationIssue(
                    "wan_clip", f"Wan 的 {wan_task} 任务需要 CLIP 图像条件，缓存阶段没有 --clip 就无法写入 I2V 条件"))
        if architecture == "hunyuan_video_15" and values.get("hv15_task") == "i2v":
            if not str(values.get("hv15_image_encoder") or "").strip():
                issues.append(ValidationIssue(
                    "hv15_image_encoder", "HunyuanVideo 1.5 的 i2v 任务需要图像编码器；缓存阶段的 --i2v 会因缺少 --image_encoder 直接断言失败"))
        # The cache flag and the training task are independent knobs; the adapter
        # derives the flag from the task, but an explicit mismatch means the cache
        # and the trainer would prepare different conditions.
        if architecture == "hunyuan_video_15" and values.get("hv15_i2v") and values.get("hv15_task") != "i2v":
            issues.append(ValidationIssue(
                "hv15_i2v", "已勾选「图生视频（缓存）」但任务为 t2v：缓存会按 i2v 编码，训练却按 t2v 进行"))
        if architecture == "wan" and values.get("wan_i2v") and "i2v" not in wan_task and "flf2v" not in wan_task:
            issues.append(ValidationIssue(
                "wan_i2v", f"已勾选「图生视频（缓存）」但任务为 {wan_task}：缓存会按 i2v 编码，训练却按该任务进行"))
        # These two trainers never read --vae, so a saved value would be ignored.
        if legacy_vae and str(values.get("vae") or "").strip():
            issues.append(ValidationIssue(
                "vae", "该架构不使用通用 VAE（MiniMax-H3 用视频/音频 VAE，HiDream-O1 用像素缓存），该路径会被忽略", "warning"))
        # --dim_from_weights needs the weights it should inspect (trainer raises).
        if values.get("dim_from_weights") and not str(values.get("network_weights") or "").strip():
            issues.append(ValidationIssue("dim_from_weights", "「从权重推断 Rank」需要同时填写「初始权重路径」"))
        # H3 raises for an empty [min, max] range; an inverted range is meaningless
        # for any architecture.
        if values.get("min_timestep") is not None and values.get("max_timestep") is not None:
            if values["min_timestep"] >= values["max_timestep"]:
                issues.append(ValidationIssue("min_timestep", "「最小时间步」必须小于「最大时间步」"))
        # Wan2.2 high/low models: block swap and inactive-DiT offloading are exclusive
        # (the trainer asserts on the combination).
        if architecture == "wan" and values.get("wan_dit_high_noise"):
            if values.get("blocks_to_swap") and values.get("wan_offload_inactive_dit"):
                issues.append(ValidationIssue(
                    "wan_offload_inactive_dit", "Wan2.2 高低噪双模型使用块交换时不能开启「卸载空闲 DiT」（引擎会断言失败）"))
            # Only the Wan2.2 task configs define a boundary; everything else makes
            # the trainer raise "timestep_boundary is not specified for high noise model".
            if (values.get("wan_timestep_boundary") is None
                    and str(values.get("wan_task") or "") not in ("t2v-A14B", "i2v-A14B")):
                issues.append(ValidationIssue(
                    "wan_timestep_boundary", "使用高噪声 DiT 时必须填写「时间步边界」（Wan2.2 双模型的配置自带默认值，其它任务没有）"))
        # Preview sampling without prompts does nothing but log an engine error.
        sampling = (values.get("sample_every_n_steps") or values.get("sample_every_n_epochs")
                    or values.get("sample_at_first"))
        if sampling and not str(values.get("sample_prompts") or "").strip():
            issues.append(ValidationIssue(
                "sample_prompts", "已启用预览采样但未填写采样提示词，引擎会跳过采样", "warning"))
        # A custom cache dir must not land in the dataset, the output folder (its
        # *.safetensors would be collected as checkpoints) or the engine source.
        issues.extend(self._cache_directory_issues(installation, config))
        issues.extend(self._list_argument_issues(values))
        for severity, message in extra_params.problems(
            str(values.get(extra_params.PARAM_KEY) or ""),
            extra_params.provided_keys(
                surface, architecture, config.get("params") or {}, skip_false=True,
                extra=("dit", "dataset_config", "output_dir", "output_name", "network_module", "config_file",
                       *(p.native_key or p.key for p in surface
                         if p.native and (config.get("params") or {}).get(p.key) is False)),
            ),
            bound_output=config.get("output_layout") == "task_id",
        ):
            issues.append(ValidationIssue(extra_params.PARAM_KEY, message, severity))
        return tuple(issues)

    def caption_cache_notices(self):
        return (
            ("reuse_latent_cache", "已开启「复用已有 Latents 缓存」：改过图片或分辨率请先清空缓存目录再提交"),
            ("reuse_te_cache", "已开启「复用已有 TE 缓存」：改动 caption 不会重算，请先清空缓存目录（或换一个缓存目录）再提交"),
        )

    # ---- native config rendering ----

    @staticmethod
    def _toml_value(value, list_arg=False):
        if list_arg:
            tokens = value if isinstance(value, list) else str(value).split()
            return "[" + ", ".join(json.dumps(str(t)) for t in tokens) + "]"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.dumps(str(value), ensure_ascii=False)

    def _render_train(self, config, prompt_directory=Path(".")):
        architecture = config.get("architecture")
        pipeline = pipeline_for(architecture)
        params, extra = self._effective_params(config)
        values = {}
        surface = architecture_params(architecture)
        for p in surface:
            if not p.native:
                continue
            value = params.get(p.key, p.default)
            if value is None or value == "" or (value is False and p.key not in params):
                continue
            if p.key in ("lr_warmup_steps", "lr_decay_steps") and isinstance(value, float) and value.is_integer():
                # The engine reads a float as "ratio of train steps": 500.0 would be
                # 500x the run. An integral input must stay an integer in the TOML.
                value = int(value)
            if p.key == "sample_prompts":
                value = prompt_file(config, prompt_directory)
                if not value:
                    continue
            values[p.native_key or p.key] = (value, p.list_arg)
        values["dit"] = (config["base_model_path"], False)
        values["network_module"] = (pipeline.module if pipeline else NETWORK_MODULES.get(architecture, "networks.lora"), False)
        values["output_dir"] = (config["output_dir"], False)
        values["output_name"] = (self._output_name(config), False)
        # Supplementary arguments land in the same config file, so preview, export
        # and the real run all see them.
        modeled = {p.native_key or p.key for p in surface if p.native}
        for name, value in extra.items():
            if name not in modeled:
                values[name] = ([value] if name in ("base_weights", "base_weights_multiplier") else value, False)
        return "".join(f"{key} = {self._toml_value(value, list_arg)}\n" for key, (value, list_arg) in values.items())

    def _cache_dir(self, config, job_directory):
        custom = config.get("params", {}).get("cache_directory")
        if isinstance(custom, str) and custom.strip():
            return Path(custom)
        return Path(job_directory) / "cache" / str(config.get("architecture"))

    def _dataset_toml(self, config, cache_dir):
        params, _ = self._effective_params(config)
        def option(key):
            value = params.get(key)
            return _PARAM_DEFAULTS[key] if value is None else value

        width = option("resolution_width")
        height = option("resolution_height")
        resolutions = self.parse_resolutions(params.get("bucket_resolutions"))
        text = "[general]\n"
        if resolutions:
            # Musubi accepts one scalar or one [width, height] pair per dataset.
            # Represent multiple sizes as separate dataset entries, never as
            # a nested general.resolution array.
            width, height = resolutions[0]
        text += f"resolution = [{width}, {height}]\n"
        if option("caption_extension"):
            text += "caption_extension = " + json.dumps(option("caption_extension")) + "\n"
        text += f"batch_size = {option('batch_size')}\n"
        text += f"enable_bucket = {str(option('enable_bucket')).lower()}\n"
        text += f"bucket_no_upscale = {str(option('bucket_no_upscale')).lower()}\n\n"
        for dataset_width, dataset_height in resolutions or [(width, height)]:
            # Sharing caches across resolutions would mix latent shapes and
            # make training consume files from the wrong dataset.
            dataset_cache = Path(cache_dir)
            if len(resolutions) > 1:
                dataset_cache /= f"{dataset_width}x{dataset_height}"
            text += "[[datasets]]\n"
            text += f"resolution = [{dataset_width}, {dataset_height}]\n"
            text += "image_directory = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
            if params.get("control_directory"):
                text += "control_directory = " + json.dumps(params["control_directory"], ensure_ascii=False) + "\n"
            if config.get("architecture") == "framepack":
                text += f"fp_latent_window_size = {option('fpack_latent_window_size')}\n"
                if params.get("fp_1f_target_index") is not None:
                    text += f"fp_1f_target_index = {params['fp_1f_target_index']}\n"
            text += "cache_directory = " + json.dumps(str(dataset_cache), ensure_ascii=False) + "\n"
            text += f"num_repeats = {option('num_repeats')}\n\n"
        return text

    @staticmethod
    def parse_resolutions(value):
        """Parse the multi-line bucket resolution list ("960x544", "960 544", …)."""
        if not isinstance(value, str) or not value.strip():
            return []
        parsed = []
        for line in value.replace(",", "\n").replace("*", "x").splitlines():
            token = line.strip().lower().replace(" ", "x")
            if not token:
                continue
            parts = token.split("x")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                return []  # validated separately; never render a half-parsed list
            width, height = int(parts[0]), int(parts[1])
            if not (64 <= width <= 4096 and 64 <= height <= 4096):
                return []
            if (width, height) not in parsed:
                parsed.append((width, height))
        return parsed

    @staticmethod
    def invalid_resolutions(value):
        """Lines that are not usable `<width>x<height>` pairs (for validation)."""
        if not isinstance(value, str) or not value.strip():
            return []
        bad = []
        for line in value.replace(",", "\n").replace("*", "x").splitlines():
            token = line.strip().lower().replace(" ", "x")
            if not token:
                continue
            parts = token.split("x")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                bad.append(line.strip())
            elif not (64 <= int(parts[0]) <= 4096 and 64 <= int(parts[1]) <= 4096):
                bad.append(line.strip())
        return bad

    # ---- launch stages ----

    def _stage_args(self, pipeline_args, values):
        args = []
        for cli, key in pipeline_args:
            value = values.get(key)
            if value is None or value is False or (isinstance(value, str) and value == ""):
                continue
            if isinstance(value, bool):
                args.append("--" + cli)
            else:
                args.extend(("--" + cli, str(value)))
        return args

    def _stages(self, installation, config, job_directory):
        architecture = config.get("architecture")
        pipeline = pipeline_for(architecture)
        if pipeline is None:
            raise ValueError(f"未声明 {architecture} 的 musubi 流水线")
        params, _ = self._effective_params(config)
        values = {p.key: params.get(p.key, p.default) for p in architecture_params(architecture)}
        # The cache stages take their own --i2v flag; derive it from the task/inputs
        # so the cached condition always matches what the trainer will use.
        for key, value in self._derived_values(architecture, values).items():
            if params.get(key) is None:
                values[key] = value
        python = str(installation.python_executable)
        dataset = str(Path(job_directory) / "dataset.toml")
        native = str(Path(job_directory) / "config.native.toml")
        source = installation.source_path

        # Cache reuse is a user choice, not a fixed behaviour: without it every run
        # recomputes from the current dataset, so an edited caption is never served
        # from a stale cache.
        reuse_latents = ("--skip_existing",) if values.get("reuse_latent_cache") else ()
        latents = (
            python,
            "-u",
            str(self._script(source, pipeline.latents)),
            "--dataset_config",
            dataset,
            *reuse_latents,
            *self._stage_args(pipeline.latents_args, values),
        )
        if pipeline.text_encoder_launcher:
            launcher = Path(__file__).with_name(pipeline.text_encoder_launcher)
            text_encoder_prefix = (
                python,
                "-u",
                str(launcher),
                "--engine_source",
                str(source),
                "--engine_script",
                pipeline.text_encoder,
            )
        else:
            text_encoder_prefix = (
                python,
                "-u",
                str(self._script(source, pipeline.text_encoder)),
            )
        reuse_te = ("--skip_existing",) if values.get("reuse_te_cache") else ()
        text_encoder = (
            *text_encoder_prefix,
            "--dataset_config",
            dataset,
            *reuse_te,
            *self._stage_args(pipeline.te_args, values),
        )
        if pipeline.train_launcher_args:
            # Sample generation re-encodes prompts during training with the same
            # tokenizer, so the training stage needs the same local redirect.
            train = [python, "-u", str(Path(__file__).with_name(pipeline.text_encoder_launcher)),
                     "--engine_source", str(source), "--engine_script", pipeline.script,
                     *self._stage_args(pipeline.train_launcher_args, values)]
        else:
            train = [python, "-u", str(self._script(source, pipeline.script))]
        train += ["--config_file", native, "--dataset_config", dataset]
        for key in pipeline.cli_args:
            p = self._param(key)
            value = values.get(key)
            if value not in (None, "") and p is not None:
                train.extend(("--" + (p.native_key or p.key), str(value)))
        return (tuple(latents), tuple(text_encoder), tuple(train))

    def preview_commands(self, installation, config, native_config):
        return self._stages(installation, config, Path(native_config).parent)

    def preview_argv(self, installation, config, native_config):
        return self.preview_commands(installation, config, native_config)[-1]

    def export_parameters(self, config):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 Musubi 适配器支持")
        notice = "此文件只包含训练参数；Musubi 运行时还需单独提供 dataset_config，执行对应底模的缓存阶段，并配置预览提示词（如启用预览）。"
        output_name = self._output_name(config)
        text = self._render_train(config).replace('output_name = "lora"\n', "output_name = " + json.dumps(output_name, ensure_ascii=False) + "\n")
        if exported_prompt_files(config):
            notice += " 启用采样时会同时导出 sample_prompts.txt 配套文件；运行原生配置时请将它与参数文件放在同一工作目录。"
        return "# " + notice + "\n" + text, (notice,)

    def native_config_text(self, config, dataset_config):
        # Preview/write/export all render the training config here; the dataset
        # config is generated separately by _dataset_toml.
        return self._render_train(config, Path(dataset_config).parent)

    def export_parameter_files(self, config):
        return exported_prompt_files(config)

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        job_directory = Path(job_directory)
        cache_dir = self._cache_dir(config, job_directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(job_directory / "dataset.toml", self._dataset_toml(config, cache_dir))
        target = job_directory / "config.native.toml"
        write_prompts(config, job_directory)
        atomic_text(target, self.native_config_text(config, job_directory / "dataset.toml"))
        stages = self._stages(installation, config, job_directory)
        JobFiles(job_directory).write_stages(stages)
        return target

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        files = JobFiles(native_config.parent)
        if not files.stages.is_file():
            raise RuntimeError("缺少 musubi 流水线定义 stages.json")
        commands = files.read_stages()
        if not commands:
            raise RuntimeError("musubi 流水线为空")
        return LaunchSpec(commands[-1], Path(installation.source_path),
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}, commands[:-1])

    def parse_log(self, line):
        return parse_progress(line)

    def recovery_capabilities(self, installation, architecture=None):
        # save_state writes an Accelerate state dir that --resume can restore; the
        # engine is shared by every architecture, so this is not architecture-specific.
        # Only full-state continuation is offered; a fresh run from LoRA weights is a
        # different user intent and is not presented as "continue".
        return RecoveryCapabilities(graceful_stop=True, resume_full_state=True)

    def resume_sources(self, job_directory):
        try:
            config = JobFiles(job_directory).read_job()
        except (OSError, ValueError, KeyError):
            return ()
        return accelerate_resume_sources(Path(config.get("output_dir") or ""))

    def resume_parameters(self, source: ResumeSource):
        return {"resume": str(source.path)}

    @classmethod
    def _inside_state_dir(cls, path, root):
        """True for a file inside an Accelerate ``*-state`` directory.

        Musubi writes ``{name}-step00000100-state/model.safetensors`` next to the
        LoRA checkpoints. That file is the full model/optimizer state, so it must
        never be reported as a trained LoRA.
        """
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            return False
        return any(part.endswith(cls._STATE_DIR_SUFFIX) for part in parts[:-1])

    def collect_artifacts(self, job_directory):
        config = JobFiles(job_directory).read_job()
        root = Path(config["output_dir"])
        found = []
        if not root.is_dir():
            # Musubi creates the folder on its first save; before that there is nothing to collect.
            return ()
        for path in root.rglob("*"):
            if not path.is_file() or self._inside_state_dir(path, root):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                # musubi step files are "{name}-step{step:08d}"; epoch files use "{name}-{epoch:06d}".
                match = re.search(r"-step(\d+)", path.stem)
                found.append(Artifact(path, "checkpoint", True, int(match[1]) if match else None))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
            elif suffix in (".mp4", ".webm"):
                # Video architectures write previews as mp4 (the image gallery and the
                # preview endpoint only handle stills, so these are indexed as video).
                found.append(Artifact(path, "video", True))
        return tuple(found)
