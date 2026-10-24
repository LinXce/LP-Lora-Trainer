"""KohakuMossTrainer adapter: MOSS-Image LyCORIS adapters and full finetune.

The clone is a Python package (``kohakumoss``) whose scripts are **kogine** scripts: every
setting is a module-level global, and a config file is a Python module of the same names.
A run therefore generates ``config.py`` and calls
``kogine run <script> --config config.py``.

Caches are a separate preparation: ``scripts/data/cache_text.py`` then
``scripts/data/cache_latents.py`` run before the trainer, like musubi-tuner's stages.
No engine imports here.
"""

import ast
import json
import os
import re
import tomllib
from pathlib import Path

from adapters.base import (
    Artifact,
    DetectionResult,
    EngineAdapter,
    InstallationPlan,
    RecoveryCapabilities,
    ResumeSource,
    ValidationIssue,
)
from adapters.capabilities import check_value, specs
from adapters.kohaku_moss_trainer.progress import parse_progress
from adapters.kohaku_moss_trainer.params import ARCHITECTURES, PARAMS
from app.schemas.training import LaunchSpec
from app.services.jobs import JobFiles

TRAIN_SCRIPT = "scripts/train/moss.py"
CACHE_STAGES = ("scripts/data/cache_text.py", "scripts/data/cache_latents.py")

# The engine's weights layouts (kohakumoss.weights.layout): a ComfyUI models folder,
# an HF pipeline folder, a variant root holding base/ and sft/, or a single DiT file.
VARIANTS = ("base", "sft")
COMPONENT_DIRS = {"transformer": "transformer", "text_encoder": "text_encoder", "vae": "vae"}
COMPONENT_LABELS = {"transformer": "DiT 权重", "text_encoder": "文本编码器", "vae": "VAE"}
COMFY_DIRS = {"transformer": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}
COMFY_DEFAULTS = {
    "transformer": ("moss_image_{variant}_int8_convrot", "moss_image_{variant}_bf16"),
    "text_encoder": ("qwen3.5_9b_int8_convrot", "qwen3.5_9b_bf16", "qwen3.5_9b_w4a8_gptq"),
    "vae": ("flux2-vae_small", "flux2-vae"),
}
WEIGHTS_SUFFIX = ".safetensors"

# Dataset / planner values the engine rejects at run time (kohakumoss_data.config
# `data_problems`, resolution.geometry LATENT_FACTOR, loader.decode enums).
LATENT_FACTOR = 16
BACKGROUNDS = ("white", "black", "gray")
RESAMPLE_FILTERS = ("lanczos", "bicubic", "bilinear", "box")

# Container-valued engine knobs: the form takes JSON or a Python literal, and only a
# value of the right container shape renders as the object the engine reads.
LITERAL_PARAMS = {
    "ARCH_OVERRIDES": dict, "ALGO_KWARGS": dict, "DROPOUT": dict, "SCHEDULE": dict,
    "TIMESTEP_KWARGS": dict, "KERNELS": dict, "MEMORY_OVERRIDES": dict,
    "PLANNER_KWARGS": dict, "VALIDATION": dict, "TEXT_ENCODE": dict, "VAE_ENCODE": dict,
    "LOSS_KWARGS": dict, "OPTIMIZER_KWARGS": dict, "FP16_FIX_KWARGS": dict,
    "BETAS": (list, tuple), "DEVICES": (list, tuple), "KEEP_FP32": (list, tuple),
    "ASPECT_RANGE": (list, tuple), "TRACE_STEPS": (list, tuple),
    "PROFILE_STEPS": (list, tuple),
}

# Every parameter's declared default, so a partial draft renders like the form.
PARAM_DEFAULTS = {spec.key: spec.default for spec in PARAMS}

# LP-only knobs translated into an engine key at render time (never written under
# their own name): a fractional checkpoint, a catalog memory profile, and the dataset
# source's constructor arguments.
LP_ONLY_KEYS = (
    "CHECKPOINT_PARTIAL", "MEMORY_PROFILE_NAME",
    "DATASET_INCLUDE_ROOT", "DATASET_REPEAT_PREFIX", "DATASET_ARTIST_FROM_FOLDER",
    "DATASET_HASH_CONTENT", "DATASET_RECURSIVE",
)
REPEAT_PREFIXES = ("first", "product", "ignore")
# kohakumoss_data.registry SOURCES names, and the image suffixes kohya's walk accepts.
DATASET_SOURCES = ("kohya", "folder", "jsonl", "parquet", "webdataset", "moss_shard")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jxl", ".avif")
# app/services/training.py's task folder name (path characters replaced), mirrored so
# the renderer can tell whether OUTPUT_DIR already is that folder.
TASK_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# The engine paints log values when it writes colour (FORCE_COLOR / a TTY); the parsed
# text must be matched without the escape sequences.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Engine-enforced runtime requirements (pyproject.toml + adapters/lycoris/version.py).
# LP cannot import the engine's interpreter, so a rule is judged only against the
# versions a diagnosis reported; an absent version is never assumed to be satisfied.
DEPENDENCY_RULES = (
    ("lycoris-lora", ">=4.1.3", "引擎在导入适配器时硬校验 lycoris 版本与 kernels API"),
    ("torch", ">=2.13,<2.15", "引擎 pyproject 的 torch 固定区间"),
    ("transformers", ">=5.3", "Qwen3.5 文本编码器"),
    ("diffusers", ">=0.38", "VAE 与管线加载"),
)

_RELEASE = re.compile(r"^v?(\d+(?:\.\d+)*)(.*)$")
_CLAUSE = re.compile(r"^(==|!=|>=|<=|>|<)\s*(\S+)$")


def _release(text):
    """``"4.1.3"`` / ``"2.14.1+cu130"`` / ``"4.1.3rc1"`` -> ``((4, 1, 3), rank)``; rank
    orders a pre-release before its release and drops any local build suffix."""
    match = _RELEASE.match(str(text).strip().lower())
    if match is None:
        return None
    parts = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    return parts, -1 if suffix.startswith(("a", "b", "rc", "dev", "pre", "alpha", "beta")) else 0


def _compare(left, right):
    """-1 / 0 / 1 for two ``_release`` values (zero-padded parts, then the rank)."""
    parts, rank = left
    other, other_rank = right
    width = max(len(parts), len(other))
    parts += (0,) * (width - len(parts))
    other += (0,) * (width - len(other))
    if parts != other:
        return 1 if parts > other else -1
    if rank != other_rank:
        return 1 if rank > other_rank else -1
    return 0


def _satisfies(version, specifier):
    """True / False for ``version`` against a comma-separated specifier; None when
    either side cannot be parsed (never guessed)."""
    current = _release(version)
    if current is None:
        return None
    for clause in str(specifier).split(","):
        match = _CLAUSE.match(clause.strip())
        if match is None:
            return None
        operator, wanted_text = match.groups()
        wanted = _release(wanted_text)
        if wanted is None:
            return None
        order = _compare(current, wanted)
        ok = {"==": order == 0, "!=": order != 0, ">=": order >= 0,
              "<=": order <= 0, ">": order > 0, "<": order < 0}[operator]
        if not ok:
            return False
    return True


class KohakuMossTrainerAdapter(EngineAdapter):
    engine_id = "kohaku_moss_trainer"
    display_name = "KohakuMossTrainer"
    adapter_version = "0.1.0"
    installation_help = (
        "使用引擎自己的 Python 包：建独立 venv 后 pip install -e . 再安装 kogine 运行器；"
        "该引擎的脚本都是 kogine 脚本，缺少运行器无法启动。"
    )
    installation_source_help = "由引擎 pyproject 声明依赖；PyTorch/CUDA wheel 镜像通过 pip 索引注入。"
    # The engine declares these as optional-dependencies; LP installs one only when the
    # user picks it, so the base environment stays the engine's default set.
    optional_extras = (
        ("kernels", "CUDA 内核组件（triton-windows / tilelang；启用 torch.compile 时需要）"),
        ("quant", "8-bit 优化器（bitsandbytes）：OPTIMIZER=adamw8bit 时需要"),
        ("optim", "额外优化器（prodigyopt / schedulefree）：OPTIMIZER=prodigy / schedule_free 时需要"),
        ("lightning", "Lightning 驱动：DRIVER=lightning 时需要"),
        ("wandb", "Weights & Biases 日志：LOGGER=wandb 时需要"),
    )
    # Set by the installation service from the request's ticked extras before the plan
    # is built; empty means the engine's plain default install.
    installation_extras: tuple[str, ...] = ()
    # Probed by the environment diagnosis (find_spec) and stored on the engine record.
    optional_modules = (
        "triton", "tilelang",
        "bitsandbytes", "prodigyopt", "schedulefree", "wandb", "lightning",
    )
    # A parameter value that needs one optional distribution: the extra to install and
    # the failure the engine reports at the training stage when it is missing.
    OPTIONAL_FEATURES = {
        "OPTIMIZER": {
            "adamw8bit": ("bitsandbytes", "quant", "8-bit 优化器"),
            "prodigy": ("prodigyopt", "optim", "Prodigy 优化器"),
            "schedule_free": ("schedulefree", "optim", "Schedule-Free 优化器"),
        },
        "LOGGER": {"wandb": ("wandb", "wandb", "W&B 日志")},
        "DRIVER": {"lightning": ("lightning", "lightning", "Lightning 驱动")},
    }
    # Removed knobs a stored draft may still carry; ignored instead of reported as
    # unsupported so an older task can be re-submitted.
    LEGACY_PARAMS = ("EXPORT_READY",)
    default_installation_source = "cu130"
    environment_strategy = "embedded"
    environment_dir = ".lp/python"
    environment_candidates = (".venv", "venv")
    # The engine needs >=3.10; 3.11 matches the cp311 torch wheels the existing
    # environments were built with, so a transferred or reinstalled environment keeps
    # working. Bumping this to 3.12 invalidates every cp311 binary (torch_python.dll
    # fails to load), so it is a deliberate pin, not a default.
    python_version = "3.11"
    torch_index_mode = "env"
    native_format = "python"
    submittable = True
    creates_output_dir = True
    supports_nested_images = True
    # Import names, measured in a working install: kogine is the CLI of kohaku-engine,
    # so the module to check is kohakuengine (there is no importable "kogine").
    diagnostic_modules = ("torch", "transformers", "safetensors", "lycoris", "kohakuengine")
    clone_url = "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git"
    # Trains from WEIGHTS (a model directory), so the generic base-model file is hidden.
    base_model_optional_architectures = ("moss_image",)
    clone_branch = "main"
    training_notice = "MOSS-Image 专用：先缓存文本与潜空间，再训练；配置以 Python 文件生成。"

    # ---- detection / install ----

    def detect(self, source_path):
        source = Path(source_path)
        try:
            project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return DetectionResult(self.engine_id, False, "缺少或无法解析 pyproject.toml")
        matched = project.get("name") == "kohakumoss" and (source / TRAIN_SCRIPT).is_file()
        return DetectionResult(self.engine_id, matched, "kohakumoss 项目元数据与 scripts/train/moss.py")

    def architectures(self, source_path):
        source = Path(source_path)
        return tuple(a for a in ARCHITECTURES if (source / a.script).is_file())

    def installation_sources(self, source_path):
        """CUDA indexes this engine can install torch from (its own guide's mapping).

        Driver CUDA >= 13.0 -> cu130, >= 12.8 -> cu128, >= 12.6 -> cu126. LP ranks them
        against the driver; the chosen tag also seeds the wheel-mirror URLs.
        """
        return ("cu130", "cu128", "cu126")

    def installation_plan(self, source_path, python, torch_source, uv=None, torch_mirror_url=None):
        source = Path(source_path)
        if not self.detect(source).matched:
            raise ValueError("KohakuMossTrainer 源码不完整")
        # Mirrors the engine's own guide: torch first from the index that matches the
        # driver, then the package with that index as an *extra* index so the torch
        # build survives. kogine is the CLI of `kohaku-engine`, a normal dependency —
        # there is no distribution called "kogine".
        existing = torch_source in ("existing", "", None)
        tag = "" if existing else (torch_source if re.fullmatch(r"cu\d{3}", str(torch_source)) else "")
        if not existing and not tag:
            raise ValueError(f"不支持的 CUDA 源：{torch_source}；可用 cu130 / cu128 / cu126")
        index = torch_mirror_url.rstrip("/") if torch_mirror_url else (
            f"https://download.pytorch.org/whl/{tag}" if tag else "")
        stages = []
        # An embedded interpreter is not a venv: `--require-virtualenv` would refuse
        # to install into the copied CPython that lives in the engine directory.
        venv_flag = [] if getattr(self, "embedded_mode", False) else ["--require-virtualenv"]
        if tag:
            stages.append(tuple([python, "-u", "-m", "pip", "install", *venv_flag,
                                 "--no-input", "torch>=2.13,<2.15", "torchvision",
                                 "--index-url", index]))
        package = [python, "-u", "-m", "pip", "install", *venv_flag, "--no-input",
                   "--no-build-isolation"]
        if index:
            # pip has no --index-strategy (that is a uv flag); the extra index is what
            # keeps the already installed cu… torch build.
            package += ["--extra-index-url", index]
        # Optional pip extras ticked in the install dialog (e.g. `kernels` pulls
        # triton-windows). Empty by default: the engine's own base set only.
        extras = tuple(e for e in getattr(self, "installation_extras", ()) if e)
        editable = f"{source}[{','.join(extras)}]" if extras else str(source)
        package += ["-e", editable, "torch>=2.13,<2.15"]
        stages.append(tuple(package))
        suffix = f" + {', '.join(extras)}" if extras else ""
        return InstallationPlan("KohakuMossTrainer：torch + 可编辑安装（kogine 随 kohaku-engine）" + suffix,
                                tuple(stages))

    @staticmethod
    def _site_packages(environment):
        """The site-packages directory of an embedded CPython (`Lib/` on Windows,
        `lib/pythonX.Y/` elsewhere), or None."""
        candidates = [environment / "Lib" / "site-packages",
                      environment / "lib" / "site-packages"]
        candidates += sorted(environment.glob("lib/python*/site-packages"))
        return next((path for path in candidates if path.is_dir()), None)

    @staticmethod
    def _distribution_versions(site_packages):
        """{distribution: version} read from a site-packages tree's dist-info names."""
        versions = {}
        for info in site_packages.glob("*.dist-info"):
            name, _, found = info.stem.partition("-")
            if name and found:
                versions.setdefault(name.lower().replace("_", "-"), found)
        return versions

    def embedded_environment_issues(self, source_path):
        """Version-rule violations of the adapter's own embedded environment.

        The diagnosis probes modules by import name only, so it cannot see that an
        installed distribution is older than the engine requires (the engine raises on
        import, after two cache stages). This reads the dist-info names of the
        environment LP created for the engine — no import, no subprocess — and is
        skipped when a venv-style layout makes the bound interpreter ambiguous.
        """
        source = Path(source_path)
        if not self.environment_dir:
            return ()
        environment = source / self.environment_dir
        site_packages = self._site_packages(environment) if environment.is_dir() else None
        if site_packages is None:
            return ()
        if any((source / name).is_dir() for name in self.environment_candidates):
            return ()
        versions = self._distribution_versions(site_packages)
        return tuple(
            f"环境 {environment} 的 {name}={versions[name]} 不满足引擎要求 {specifier}"
            f"（{why}）；在该环境执行 pip install -U \"{name}{specifier}\" 后重新诊断"
            for name, specifier, why in DEPENDENCY_RULES
            if versions.get(name) and _satisfies(versions[name], specifier) is False
        )

    def source_issues(self, source_path):
        source = Path(source_path)
        issues = []
        if not (source / TRAIN_SCRIPT).is_file():
            issues.append("缺少 scripts/train/moss.py 训练入口")
        issues.extend(self.embedded_environment_issues(source))
        return tuple(issues)

    # ---- parameters / validation ----

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable,
                    base_model_optional_architectures=list(self.base_model_optional_architectures))

    # ---- weights layout / components / dependency rules ----

    @staticmethod
    def _weights_layout(root):
        """``(layout, problem)`` of a WEIGHTS value: ``hf`` / ``comfy`` / ``variants`` /
        ``file``, mirroring ``kohakumoss.weights.layout``; ``(None, reason)`` when the
        path is missing or is no MOSS-Image weights layout."""
        path = Path(str(root))
        if path.is_file():
            return "file", ""
        if not path.is_dir():
            return None, "权重路径不存在"
        if (path / COMPONENT_DIRS["transformer"]).is_dir():
            return "hf", ""
        comfy = [name for name in COMFY_DIRS.values()
                 if any((path / name).glob("*" + WEIGHTS_SUFFIX))]
        if comfy:
            return "comfy", ""
        for variant in VARIANTS:
            folder = path / variant
            if folder.is_dir() and any((folder / name).is_dir() for name in COMPONENT_DIRS.values()):
                return "variants", ""
        return None, (
            "目录不是可识别的 MOSS-Image 权重：需要 ComfyUI 布局（diffusion_models/、"
            "text_encoders/、vae/）、HF 布局（transformer/、text_encoder/、vae/）、"
            "base|sft 变体根目录，或单个 DiT .safetensors")

    @classmethod
    def _comfy_file(cls, folder, component, value, variant):
        """The file of `component` in a ComfyUI models folder: `value` (a name or a path)
        when given, else the first engine default that exists."""
        name = str(value or "").strip()
        if name:
            if not name.lower().endswith(WEIGHTS_SUFFIX):
                name += WEIGHTS_SUFFIX
            return folder / name
        for candidate in COMFY_DEFAULTS[component]:
            path = folder / (candidate.format(variant=variant) + WEIGHTS_SUFFIX)
            if path.is_file():
                return path
        return folder / (COMFY_DEFAULTS[component][0].format(variant=variant) + WEIGHTS_SUFFIX)

    @classmethod
    def _component_path(cls, weights, value, component, layout, variant="base"):
        """A loadable path for `component` in the cache stages; the engine resolves the
        training stage itself (`kohakumoss.weights.layout.component_path`)."""
        text = str(value or "").strip()
        if text and Path(text).exists():
            return str(Path(text))
        if layout == "file":
            # One DiT file: nothing to derive, the knob itself names the component.
            return text
        root = Path(str(weights or ""))
        if layout == "comfy":
            return str(cls._comfy_file(root / COMFY_DIRS[component], component, text, variant))
        if text:
            return text
        if layout == "variants":
            for base in dict.fromkeys((variant, "base", "")):
                path = root / base / COMPONENT_DIRS[component] if base else root / COMPONENT_DIRS[component]
                if path.is_dir():
                    return str(path)
        return str(root / COMPONENT_DIRS[component])

    @staticmethod
    def _sample_every(values) -> int:
        try:
            return int(values.get("SAMPLE_EVERY") or 0)
        except (TypeError, ValueError):
            return 0

    def _component_needs(self, values):
        """``(parameter, component, whether this run reads it)`` for TEXT_ENCODER / VAE.

        A cache stage always reads its encoder, and the trainer reads it for an online
        cache or for sample prompts; only then must a single-file WEIGHTS name it.
        """
        text_cache = str(values.get("TEXT_CACHE") or "offline").strip()
        latent_cache = str(values.get("LATENT_CACHE") or "offline").strip()
        sampling = self._sample_every(values) > 0
        return (
            ("TEXT_ENCODER", "text_encoder", text_cache != "off" or sampling),
            ("VAE", "vae", latent_cache != "off" or sampling),
        )

    @classmethod
    def dependency_issues(cls, versions, *, require_all=False):
        """Issues for the engine's own required distributions.

        ``versions`` is ``{distribution: version}`` as a diagnosis reported it. A
        distribution absent from it is only reported with ``require_all`` (the caller
        claims the set is complete); an unparsable version is reported, never assumed
        to be fine.
        """
        if not versions:
            return ()
        issues = []
        for name, specifier, why in DEPENDENCY_RULES:
            found = versions.get(name)
            if found in (None, ""):
                if require_all:
                    issues.append(ValidationIssue(
                        name, f"环境未报告 {name} 版本，无法确认满足 {specifier}（{why}）"))
                continue
            ok = _satisfies(found, specifier)
            if ok is False:
                issues.append(ValidationIssue(
                    name,
                    f"{name}={found} 不满足引擎要求 {specifier}（{why}）；"
                    f"该环境启动训练时会在导入阶段失败，请按引擎文档升级 {name}",
                    "error"))
            elif ok is None:
                issues.append(ValidationIssue(
                    name, f"无法解析 {name} 版本 {found!r}（引擎要求 {specifier}）", "warning"))
        return tuple(issues)

    @staticmethod
    def _recorded_modules(installation):
        """Optional-module probe results stored on the engine record (empty when the
        installation object does not carry them)."""
        modules = installation.get("optional_modules") if isinstance(installation, dict) else None
        if modules is None:
            modules = getattr(installation, "optional_modules", None)
        return dict(modules) if isinstance(modules, dict) else {}

    @classmethod
    def _environment_issues(cls, installation):
        """Dependency rules judged against whatever the diagnosis recorded."""
        if isinstance(installation, dict):
            recorded = installation.get("dependency_versions")
            torch_version = installation.get("torch_version")
        else:
            recorded = getattr(installation, "dependency_versions", None)
            torch_version = getattr(installation, "torch_version", None)
        versions = dict(recorded) if isinstance(recorded, dict) else {}
        if torch_version and not versions.get("torch"):
            versions["torch"] = str(torch_version)
        return cls.dependency_issues(versions)

    @classmethod
    def _feature_issues(cls, values, installation):
        """Warnings for a selected value whose optional distribution may be missing."""
        installed = cls._recorded_modules(installation)
        issues = []
        for param, choices in cls.OPTIONAL_FEATURES.items():
            chosen = str(values.get(param) or "").strip()
            feature = choices.get(chosen)
            if feature is None:
                continue
            module, extra, what = feature
            if installed.get(module) is True:
                continue
            issues.append(ValidationIssue(
                param,
                f"{what}（{param}={chosen}）需要引擎环境安装「{module}」；请在引擎页重装环境时"
                f"勾选 {extra} 组件，否则运行到训练阶段会报缺少 {module}",
                "warning"))
        return issues

    @staticmethod
    def _dataset_prompt(dataset_path, values) -> str:
        """First caption of the dataset, mirroring the engine's SAMPLE_PROMPTS default
        (cli/config/starter.py: dataset_prompts -> first caption of the first dataset)."""
        root = Path(str(dataset_path or "").strip())
        if not root.is_dir():
            return ""
        tags_mode = str(values.get("TXT_AS") or "caption").strip() == "tags"
        suffixes = (".txt",) if tags_mode else (".caption", ".txt")
        for image in sorted(root.rglob("*")):
            if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            for suffix in suffixes:
                path = image.with_suffix(suffix)
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _flag(values, key) -> bool:
        """A boolean form value, which arrives as a bool or as text."""
        value = values.get(key)
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _literal_value(text):
        """The parsed container of a form value (JSON first, then a Python literal);
        ``None`` when it parses as neither."""
        stripped = str(text).strip() if text is not None else ""
        if not stripped:
            return {}
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(stripped)
            except (ValueError, SyntaxError, TypeError):
                continue
        return None

    @classmethod
    def _prompts(cls, values):
        """The sample prompts the render will emit: a list literal, or one per line."""
        raw = str(values.get("SAMPLE_PROMPTS") or "").strip()
        if not raw:
            return []
        if not raw.startswith(("[", "(")):
            return [row.strip() for row in raw.splitlines() if row.strip()]
        parsed = cls._literal_value(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        return []

    @classmethod
    def _memory_and_sample_issues(cls, values, dataset_path=None):
        """Constraints the engine enforces at run time, checked at submit time.

        The engine rejects a profile combined with manual plan knobs, wants one
        seed per sample prompt, requires positive sampling steps/shifts, needs
        image sizes that are multiples of its 16-pixel latent grid, needs prompts
        for sampling, and ignores input dropout under DoRA.
        """
        issues = []
        profile = (str(values.get("MEMORY_PROFILE_NAME") or "").strip()
                   or str(values.get("MEMORY_PROFILE") or "").strip())
        if profile and profile != "manual":
            blocked = [key for key in ("CHECKPOINT", "CHECKPOINT_PARTIAL", "STREAM",
                                       "STREAM_RESIDENT", "PINNED", "ACT_OFFLOAD",
                                       "OPTIM_STATE", "GRAD_OFFLOAD", "FUSED_STEP",
                                       "MEMORY_OVERRIDES")
                       if str(values.get(key) or "").strip() not in ("", "{}", "False", "0")]
            if blocked:
                # The engine ignores these under a profile, so this is a hint, not a
                # reason to refuse submission (a warning is never blocking).
                issues.append(ValidationIssue(
                    "MEMORY_PROFILE",
                    f"显存档位为 {profile} 时这些手动项不会生效：{', '.join(blocked)}；"
                    "要使用它们请把档位改成 manual，否则可以留空",
                    "warning"))
        prompts = cls._prompts(values)
        seeds = cls._literal_value(values.get("SAMPLE_SEEDS") or "(0,)")
        if prompts and isinstance(seeds, (list, tuple)):
            if len(seeds) != len(prompts):
                issues.append(ValidationIssue(
                    "SAMPLE_SEEDS", f"引擎要求每个提示词一个种子：{len(prompts)} 个提示词，{len(seeds)} 个种子"))
        if cls._sample_every(values) > 0 and not prompts and not cls._dataset_prompt(dataset_path, values):
            issues.append(ValidationIssue(
                "SAMPLE_PROMPTS",
                "SAMPLE_EVERY > 0 时必须至少有一个采样提示词；数据集没有可用字幕时也无法自动补齐"))
        for key in ("SAMPLE_STEPS", "SAMPLE_SHIFT"):
            try:
                if float(values.get(key)) <= 0:
                    issues.append(ValidationIssue(key, "必须为正数"))
            except (TypeError, ValueError):
                pass
        # memory/release.py: "none" | "empty_cache" | an int N >= 1.
        release = str(values.get("STEP_RELEASE") or "none").strip()
        if release not in ("none", "empty_cache") and not (
                release.isdigit() and int(release) >= 1):
            issues.append(ValidationIssue(
                "STEP_RELEASE", "只能是 none / empty_cache / 正整数（每 N 步释放一次）"))
        # memory/profiles.py: "weight streaming needs stream_resident >= 2".
        if str(values.get("STREAM") or "").strip() in ("ring", "swap"):
            try:
                resident = int(values.get("STREAM_RESIDENT") or 0)
            except (TypeError, ValueError):
                resident = 0
            if resident < 2:
                issues.append(ValidationIssue(
                    "STREAM_RESIDENT", "流式模式下引擎要求设备槽位 >= 2（当前为空或 0）"))
        try:
            size = ast.literal_eval(str(values.get("SAMPLE_SIZE") or "(1024, 1024)"))
            if isinstance(size, (list, tuple)) and len(size) == 2:
                if any(int(v) <= 0 or int(v) % 16 for v in size):
                    issues.append(ValidationIssue("SAMPLE_SIZE", "尺寸必须是 16 的正整数倍"))
        except (ValueError, SyntaxError, TypeError):
            issues.append(ValidationIssue("SAMPLE_SIZE", "请填写 Python 元组，如 (1024, 1024)"))
        # adapters/build/config.py: "DORA ignores input dropout".
        dropout = cls._literal_value(values.get("DROPOUT") or {})
        if cls._flag(values, "DORA") and isinstance(dropout, dict):
            try:
                input_dropout = float(dropout.get("input") or 0.0)
            except (TypeError, ValueError):
                input_dropout = 0.0
            if input_dropout > 0:
                issues.append(ValidationIssue(
                    "DROPOUT", "DORA 会忽略输入 dropout：请把 DROPOUT['input'] 设为 0"))
        return issues

    @classmethod
    def _pipeline_issues(cls, values):
        """What keeps every planned batch from fitting a pipeline micro-batch
        (kohakumoss_data.config `pipeline_problems`)."""
        issues = []
        if str(values.get("BATCHING") or "tokens").strip() == "count":
            issues.append(ValidationIssue(
                "BATCHING", "流水线并行需要 BATCHING=tokens 或 tokens_ffd（不能用 count）"))
        try:
            max_samples = int(values.get("MAX_SAMPLES") or 0)
        except (TypeError, ValueError):
            max_samples = 0
        if max_samples < 1:
            issues.append(ValidationIssue(
                "MAX_SAMPLES", "流水线并行需要 MAX_SAMPLES >= 1（微批次的样本槽位）"))
        if str(values.get("TOKEN_BUDGET") or "estimate").strip() != "exact":
            issues.append(ValidationIssue(
                "TOKEN_BUDGET", "流水线并行需要 TOKEN_BUDGET=exact（文本长度取自缓存）"))
        if str(values.get("TEXT_CACHE") or "offline").strip() != "offline":
            issues.append(ValidationIssue(
                "TEXT_CACHE", "流水线并行需要 TEXT_CACHE=offline"))
        return issues

    @classmethod
    def _data_issues(cls, values):
        """Data-knob combinations kohakumoss_data.config `data_problems` rejects; the
        engine reports them only after the cache stages have already run."""
        issues = []
        latent = str(values.get("LATENT_CACHE") or "offline").strip()
        text = str(values.get("TEXT_CACHE") or "offline").strip()
        crop = str(values.get("CROP") or "center").strip()
        mode = str(values.get("RESOLUTION_MODE") or "resize").strip()
        if latent == "offline" and (crop == "random" or mode == "shifted_square"):
            issues.append(ValidationIssue(
                "CROP",
                "LATENT_CACHE=offline 只能用中心裁剪：把裁剪方式改回 center 并把分辨率模式"
                "改回 resize/bucket，否则 cache_latents 阶段会报错"))
        if str(values.get("TOKEN_BUDGET") or "estimate").strip() == "exact" and text != "offline":
            issues.append(ValidationIssue(
                "TOKEN_BUDGET", "TOKEN_BUDGET=exact 需要 TEXT_CACHE=offline（文本长度来自缓存）"))
        try:
            size_step = int(values.get("SIZE_STEP") or LATENT_FACTOR)
        except (TypeError, ValueError):
            size_step = LATENT_FACTOR
        if size_step % LATENT_FACTOR:
            issues.append(ValidationIssue(
                "SIZE_STEP", f"像素步长必须是 latent factor（{LATENT_FACTOR}）的整数倍，当前 {size_step}"))
        try:
            if int(values.get("MAX_TOKENS") or 0) < 2:
                issues.append(ValidationIssue("MAX_TOKENS", "必须 >= 2"))
        except (TypeError, ValueError):
            pass
        background = values.get("BACKGROUND")
        if isinstance(background, str) and background.strip() and background.strip() not in BACKGROUNDS:
            issues.append(ValidationIssue(
                "BACKGROUND", f"透明图背景色只能是 {' / '.join(BACKGROUNDS)} 或 (r, g, b)"))
        resample = str(values.get("RESAMPLE") or "").strip()
        if resample and resample not in RESAMPLE_FILTERS:
            issues.append(ValidationIssue(
                "RESAMPLE", f"缩放滤镜只能是 {' / '.join(RESAMPLE_FILTERS)}"))
        if str(values.get("PARALLEL") or "none").strip() == "pipeline":
            issues.extend(cls._pipeline_issues(values))
        return issues

    @classmethod
    def _dataset_issues(cls, values, extras, dataset_path=None):
        """The DATASETS entry: values the source constructor or the engine would reject."""
        issues = []
        source = str(values.get("SOURCE") or PARAM_DEFAULTS.get("SOURCE") or "folder").strip()
        if source not in DATASET_SOURCES:
            issues.append(ValidationIssue(
                "SOURCE", f"未知数据源 {source!r}，可选 {'/'.join(DATASET_SOURCES)}"))
            return issues
        txt_as = str(values.get("TXT_AS") or "tags").strip()
        if txt_as not in ("tags", "nl"):
            issues.append(ValidationIssue("TXT_AS", "只能是 tags 或 nl"))
        repeat = str(extras.get("DATASET_REPEAT_PREFIX") or "first").strip()
        if source == "kohya":
            if repeat not in REPEAT_PREFIXES:
                issues.append(ValidationIssue(
                    "DATASET_REPEAT_PREFIX", f"只能是 {' / '.join(REPEAT_PREFIXES)}"))
            artist = str(extras.get("DATASET_ARTIST_FROM_FOLDER") or "").strip()
            if artist and artist != "name":
                try:
                    re.compile(artist)
                except re.error as exc:
                    issues.append(ValidationIssue(
                        "DATASET_ARTIST_FROM_FOLDER", f"正则无效：{artist}（{exc}）"))
            if not cls._flag(extras, "DATASET_INCLUDE_ROOT"):
                # The kohya source walks one level of `<repeats>_<name>/` folders only.
                root = Path(str(dataset_path or "").strip())
                images = []
                try:
                    if root.is_dir():
                        images = [entry for entry in root.iterdir()
                                  if entry.suffix.lower() in IMAGE_SUFFIXES]
                except OSError:
                    images = []
                if images:
                    issues.append(ValidationIssue(
                        "SOURCE",
                        f"数据集根目录有 {len(images)} 张图片，kohya 源只扫一级子目录，它们不会被收录："
                        "勾选「包含数据集根目录图片」，或把数据源改成 folder",
                        "warning"))
        return issues

    @classmethod
    def _plan_issues(cls, values, extras):
        """The manual memory plan: a catalog profile name and a fractional checkpoint."""
        issues = []
        name = str(extras.get("MEMORY_PROFILE_NAME") or "").strip()
        if name and any(char.isspace() for char in name):
            issues.append(ValidationIssue(
                "MEMORY_PROFILE_NAME",
                "档位名不能带空格；catalog 名字形如 ckpt-ring16-fused"
                "（在引擎环境运行 kohakumoss-config list --all 查看全部合法名字）"))
        partial = extras.get("CHECKPOINT_PARTIAL")
        if isinstance(partial, bool):
            issues.append(ValidationIssue("CHECKPOINT_PARTIAL", "需要 0–1 之间的数字，例如 0.5"))
            return issues
        if partial in (None, "", 0):
            return issues
        try:
            fraction = float(partial)
        except (TypeError, ValueError):
            issues.append(ValidationIssue("CHECKPOINT_PARTIAL", "需要 0–1 之间的数字，例如 0.5"))
            return issues
        if not 0.0 < fraction <= 1.0:
            issues.append(ValidationIssue(
                "CHECKPOINT_PARTIAL", f"比例必须在 (0, 1]，当前 {fraction:g}"))
        strategy = str(values.get("CHECKPOINT") or "").strip()
        if strategy not in ("", "block"):
            issues.append(ValidationIssue(
                "CHECKPOINT_PARTIAL",
                f"与「检查点策略」冲突：当前为 {strategy}；用比例检查点请把策略留空或选 block"))
        return issues

    def validate(self, installation, config):
        issues = []
        architecture = config.get("architecture")
        if architecture not in {a.id for a in self.architectures(installation.source_path)}:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        legacy = set(self.LEGACY_PARAMS)
        for key in values.keys() - known - legacy:
            issues.append(ValidationIssue(key, "不支持的参数"))
        # WEIGHTS is a ComfyUI models folder, an HF pipeline folder, a base|sft variant
        # root or one DiT file (kohakumoss.weights.layout); only the single-file form
        # cannot derive its encoders, so it must name them here.
        weights = str(values.get("WEIGHTS") or "").strip()
        if not weights:
            issues.append(ValidationIssue(
                "WEIGHTS",
                "该引擎需要 MOSS-Image 权重：ComfyUI models/ 目录、HF 布局目录（含 transformer/）、"
                "base|sft 变体根目录，或单个 DiT .safetensors（后者需另填文本编码器与 VAE）"))
        else:
            layout, problem = self._weights_layout(weights)
            if layout is None:
                issues.append(ValidationIssue("WEIGHTS", f"{problem}：{weights}"))
            else:
                variant = str(values.get("VARIANT") or "base").strip() or "base"
                for key, component, needed in self._component_needs(values):
                    if not needed:
                        continue
                    named = str(values.get(key) or "").strip()
                    if layout == "file":
                        if not named:
                            issues.append(ValidationIssue(
                                key,
                                f"WEIGHTS 是单个文件时，必须单独填写{COMPONENT_LABELS[component]}"
                                "（目录或单个文件），否则该阶段加载不到它"))
                        elif not Path(named).exists():
                            issues.append(ValidationIssue(
                                key, f"找不到{COMPONENT_LABELS[component]}：{named}"))
                        continue
                    path = Path(self._component_path(weights, named, component, layout, variant))
                    if not path.exists():
                        where = (f"（ComfyUI 布局请填 {COMFY_DIRS[component]}/ 下的文件名）"
                                 if layout == "comfy" else "")
                        issues.append(ValidationIssue(
                            key, f"找不到{COMPONENT_LABELS[component]}：{path}{where}"))
                if layout != "file":
                    dit = Path(self._component_path(
                        weights, values.get("TRANSFORMER"), "transformer", layout, variant))
                    if not dit.exists():
                        issues.append(ValidationIssue(
                            "WEIGHTS", f"找不到 DiT 权重：{dit}；可用「DiT 权重路径」指定文件名或路径"))
        issues.extend(self._memory_and_sample_issues(values, config.get("dataset_path")))
        extras = {key: values.get(key, PARAM_DEFAULTS.get(key)) for key in LP_ONLY_KEYS}
        issues.extend(self._data_issues(values))
        issues.extend(self._dataset_issues(values, extras, config.get("dataset_path")))
        issues.extend(self._plan_issues(values, extras))
        issues.extend(self._feature_issues(values, installation))
        issues.extend(self._environment_issues(installation))
        # A custom target map must parse and compile; otherwise the engine would fail
        # when it builds the adapter, long after submission.
        custom_preset = str(values.get("PRESET_CUSTOM") or "").strip()
        if custom_preset:
            parsed = None
            for parse in (json.loads, ast.literal_eval):
                try:
                    candidate = parse(custom_preset)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            if parsed is None:
                issues.append(ValidationIssue(
                    "PRESET_CUSTOM",
                    "自定义作用层必须是一个字典（JSON 或 Python 字面量），例如 "
                    '{"target_name": ["layers\\\\.1[0-9]\\\\.proj\\\\.q$"]}',
                    "error"))
            else:
                for key in ("target_name", "exclude_name"):
                    patterns = parsed.get(key) or []
                    if isinstance(patterns, str):
                        patterns = [patterns]
                    for pattern in patterns:
                        try:
                            re.compile(str(pattern))
                        except re.error as exc:
                            issues.append(ValidationIssue(
                                "PRESET_CUSTOM", f"{key} 里的正则无效：{pattern}（{exc}）", "error"))
        # torch.compile paths and explicit triton kernel backends all need the engine's
        # `kernels` extra (triton-windows). LP cannot import the engine's interpreter
        # from here, so this warns instead of blocking: the run still starts, but the
        # first compiled or triton-backed op raises TritonMissing.
        needs_triton = []
        compile_mode = str(values.get("COMPILE") or "off").strip().lower()
        if compile_mode not in ("", "off", "false"):
            needs_triton.append(f"COMPILE={compile_mode}")
        vae_encode = str(values.get("VAE_ENCODE") or "").strip().lower()
        if vae_encode not in ("", "{}", "off", "false") and "compile" in vae_encode:
            needs_triton.append("VAE_ENCODE.compile")
        kernels = str(values.get("KERNELS") or "").strip().lower()
        if "triton" in kernels:
            needs_triton.append("KERNELS 指定了 triton 后端")
        if needs_triton:
            issues.append(ValidationIssue(
                "COMPILE" if compile_mode not in ("", "off") else "KERNELS",
                "；".join(needs_triton) + " 需要引擎环境的 kernels 组件（triton-windows）。"
                "若未安装，首个使用内核的算子会报 TritonMissing；可在引擎页重装环境并勾选"
                "「CUDA 内核组件」，或把这些项改回 off / 空",
                "warning"))
        # Informational: LP fills the fix at render time, so the run itself is fine.
        if (str(values.get("PRECISION") or "").strip() == "fp16"
                and str(values.get("VARIANT") or "base").strip() == "base"
                and str(values.get("FP16_FIX") or "none").strip() == "none"):
            issues.append(ValidationIssue(
                "FP16_FIX",
                "fp16 + base 变体需要残差流修正：提交时会自动写入 FP16_FIX='online' 与 "
                "FP16_FIX_KWARGS={'c': 0.0625}；如不想自动补，可手动改成 fp32_residual",
                "warning"))
        for key, kind in LITERAL_PARAMS.items():
            raw = values.get(key)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue
            if not isinstance(self._literal_value(raw), kind):
                shape = "字典" if kind is dict else "元组/列表"
                example = "{}" if kind is dict else "(0, 1)"
                issues.append(ValidationIssue(
                    key, f"需要{shape}写法（JSON 或 Python 字面量），例如 {example}"))
        for p in PARAMS:
            if not p.native:
                continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
        return tuple(issues)

    # ---- native config (Python module) ----

    @staticmethod
    def _literal(value):
        """A Python literal for the generated config (JSON text becomes a real object)."""
        if isinstance(value, str):
            text = value.strip()
            if text[:1] in "{[(":
                # JSON first, then a Python literal: both spellings are accepted in the
                # form, and a container must never end up quoted as a string.
                for parse in (json.loads, ast.literal_eval):
                    try:
                        return repr(parse(text))
                    except (ValueError, SyntaxError):
                        continue
            return repr(value)
        return repr(value)

    @staticmethod
    def _safe_folder_name(name):
        """The task folder app/services/training.py creates for `name`."""
        value = TASK_FOLDER_CHARS.sub("_", str(name or "").strip())
        return value.rstrip(" .")[:200]

    @classmethod
    def _dataset_entry(cls, config, extras):
        """The one DATASETS entry: its source plus that source's own arguments.

        `kohakumoss_data.dataset.source_kwargs` hands every non-dataset-level key to the
        source constructor, so only the chosen source's arguments may be written.
        """
        params = config.get("params", {})
        source = str(params.get("SOURCE") or PARAM_DEFAULTS.get("SOURCE") or "folder").strip()
        entry = {
            "source": source,
            "path": str(config["dataset_path"]),
            "weight": 1.0,
            # sources/base.py `apply_sidecars` -- `.txt` goes to tags or to caption_nl.
            "txt_as": str(params.get("TXT_AS") or "tags"),
        }
        if source == "kohya":
            entry["include_root"] = cls._flag(extras, "DATASET_INCLUDE_ROOT")
            repeat = str(extras.get("DATASET_REPEAT_PREFIX") or "first").strip()
            entry["repeat_prefix"] = repeat if repeat in REPEAT_PREFIXES else "first"
            artist = str(extras.get("DATASET_ARTIST_FROM_FOLDER") or "").strip()
            if artist:
                entry["artist_from_folder"] = artist
            if cls._flag(extras, "DATASET_HASH_CONTENT"):
                entry["hash_content"] = True
        elif source == "folder":
            entry["recursive"] = cls._flag(extras, "DATASET_RECURSIVE")
            if cls._flag(extras, "DATASET_HASH_CONTENT"):
                entry["hash_content"] = True
        elif source in ("jsonl", "parquet"):
            # Both constructors take `hash_content`; their column/root arguments are
            # dataset-format specific and not exposed (see the parameter help).
            if cls._flag(extras, "DATASET_HASH_CONTENT"):
                entry["hash_content"] = True
        return entry

    def _render(self, config):
        params = config.get("params", {})
        values = {p.key: params.get(p.key, p.default) for p in PARAMS if p.native}
        # LP-only knobs never reach the engine under their own name; read them with
        # their declared defaults so a partial/older draft renders like the form shows.
        extras = {key: params.get(key, PARAM_DEFAULTS.get(key)) for key in LP_ONLY_KEYS}
        output = Path(config["output_dir"])
        run_name = str(config.get("output_name") or "").strip()
        if not run_name:
            run_name = str(config.get("name") or "moss").strip() or "moss"
            # app/services/training.py already puts the job in `<root>/<safe task name>`
            # and the engine nests once more by RUN_NAME, so an empty model name
            # repeated the task folder. Hoist to the parent: the engine still gets its
            # one OUTPUT_DIR/RUN_NAME level, now the same folder the job was given.
            if (output.name and output.parent != output
                    and output.name == self._safe_folder_name(config.get("name"))):
                output = output.parent
        lines = [
            '"""Generated by LP; a kogine config: module-level knobs."""',
            f"WEIGHTS = {self._literal(values.get('WEIGHTS') or '')}",
            f"VARIANT = {self._literal(values.get('VARIANT') or 'base')}",
            "DATASETS = " + repr([self._dataset_entry(config, extras)]),
            f"OUTPUT_DIR = {self._literal(str(output))}",
            f"RUN_NAME = {self._literal(run_name)}",
        ]
        # The engine refuses a profile together with the manual plan knobs: with a
        # profile they must stay None. Emitting both made every run invalid (and
        # PINNED=True pinned the whole text encoder, exhausting host page-locked
        # memory: cuMemHostAlloc -> CUDA_ERROR_OUT_OF_MEMORY at cache_text).
        profile_from_name = str(extras.get("MEMORY_PROFILE_NAME") or "").strip()
        profile = profile_from_name or str(values.get("MEMORY_PROFILE") or "").strip()
        # Keys this shared config must not carry: the cache scripts own ENCODE_BATCH /
        # IMAGE_ENCODER and bind TEXT_ENCODER to a dotted factory (a WEIGHTS path written
        # under that name made cache_text import it as a module), CAPTION is composed
        # below, and WEIGHTS / VARIANT lead the file. preview_commands re-adds the
        # trainer-only ones on the training stage's command line.
        skip = {"WEIGHTS", "VARIANT", "CAPTION", "TEXT_ENCODER_PLAN",
                "TEXT_ENCODER", "VAE", "PRESET_CUSTOM",
                "TEXT_ENCODER_KWARGS", "IMAGE_ENCODER_KWARGS", "IMAGE_ENCODER",
                "ENCODE_BATCH", "VAE_ENCODE_BATCH"}
        if profile_from_name:
            # An engine catalog profile name replaces the dropdown value.
            lines.append(f"MEMORY_PROFILE = {self._literal(profile_from_name)}")
            skip.add("MEMORY_PROFILE")
        partial = extras.get("CHECKPOINT_PARTIAL")
        if (not isinstance(partial, bool) and partial not in (None, "", 0)
                and profile in ("", "manual")):
            # The engine spells a fractional checkpoint as CHECKPOINT="partial:<0..1>".
            try:
                value = float(partial)
            except (TypeError, ValueError):
                value = 0.0
            if 0.0 < value <= 1.0:
                lines.append(f"CHECKPOINT = 'partial:{value:g}'")
                skip.add("CHECKPOINT")
        # Engine rule: fp16 with VARIANT="base" overflows the residual stream unless an
        # fp16 fix is set (checks/precision.py). Fill the engine's own remedy so the run
        # does not fail at stage 3.
        if (str(values.get("PRECISION") or "").strip() == "fp16"
                and str(values.get("VARIANT") or "base").strip() == "base"
                and str(values.get("FP16_FIX") or "none").strip() == "none"):
            values["FP16_FIX"] = "online"
            if str(values.get("FP16_FIX_KWARGS") or "").strip() in ("", "{}"):
                values["FP16_FIX_KWARGS"] = '{"c": 0.0625}'
        # The prompt editor in the UI stores one prompt per line; the engine knob is a
        # list literal, so convert plain text before the JSON/literal parsing below.
        raw_prompts = str(values.get("SAMPLE_PROMPTS") or "").strip()
        if raw_prompts and not raw_prompts.startswith(("[", "(")):
            rows = [row.strip() for row in raw_prompts.splitlines() if row.strip()]
            values["SAMPLE_PROMPTS"] = repr(rows) if rows else "[]"
        # The engine fills SAMPLE_PROMPTS from the dataset's first caption when the user
        # gives none (cli/config/starter.py); do the same so previews are not empty.
        if str(values.get("SAMPLE_PROMPTS") or "").strip() in ("", "[]", "()"):
            prompt = self._dataset_prompt(config.get("dataset_path"), values)
            if prompt:
                values["SAMPLE_PROMPTS"] = repr([prompt])
        # A custom target map replaces the enum preset: the engine accepts a preset
        # dict directly (`resolve_preset`), so only the container has to be emitted as
        # a Python dict rather than a string.
        # `values` only carries native params, and PRESET_CUSTOM is LP-only, so it is
        # read from the raw params map.
        custom_preset = str(config.get("params", {}).get("PRESET_CUSTOM") or "").strip()
        if custom_preset:
            parsed_preset = None
            for parse in (json.loads, ast.literal_eval):
                try:
                    candidate = parse(custom_preset)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(candidate, dict):
                    parsed_preset = candidate
                    break
            if parsed_preset is not None:
                values["PRESET"] = parsed_preset
        # A key left out of the config is absent from every stage, so preview_commands
        # re-adds the trainer-only ones on its command line. Under a profile the manual
        # plan knobs must stay None (the engine errors otherwise), so they are dropped.
        if profile and profile != "manual":
            skip |= {"CHECKPOINT", "STREAM", "STREAM_RESIDENT", "PINNED", "ACT_OFFLOAD",
                     "OPTIM_STATE", "GRAD_OFFLOAD", "FUSED_STEP", "MEMORY_OVERRIDES"}
        for key, value in values.items():
            if key in skip or value is None or value == "":
                continue
            if key == "DEVICES":
                text = str(value).strip().strip("[]()")
                indices = [part.strip() for part in text.split(",") if part.strip().isdigit()]
                lines.append("DEVICES = (" + ", ".join(indices or ["0"]) + ",)")
                continue
            if key == "STEP_RELEASE":
                # The engine takes "none" | "empty_cache" | an int N; a digit entry is
                # the int form, not the string that repr() would write.
                text = str(value).strip()
                if text.isdigit():
                    lines.append(f"STEP_RELEASE = {int(text)}")
                    continue
            if key == "DATA_RESCAN":
                # Source: kohakumoss_data/scan/runner.py -- (False, True, "never").
                text = str(value).strip().lower()
                lines.append("DATA_RESCAN = " + ("'never'" if text == "never"
                                                  else ("True" if text == "true" else "False")))
                continue
            if key == "SAVE_FUSED_QKV":
                text = str(value).strip().lower()
                lines.append("SAVE_FUSED_QKV = " + (
                    "'auto'" if text == "auto" else "True" if text == "true" else "False"))
                continue
            if key == "FINAL_CHECKPOINT":
                text = str(value).strip().lower()
                lines.append("FINAL_CHECKPOINT = " + (
                    "'auto'" if text == "auto" else "True" if text == "true" else "False"))
                continue
            lines.append(f"{key} = {self._literal(value)}")
        caption = self._caption_config(config)
        if caption:
            lines.append(f"CAPTION = {self._literal(caption)}")
        extra = str(config.get("params", {}).get("extra_params") or "").strip()
        if extra:
            lines.append("# 自定义参数")
            lines.extend(extra.splitlines())
        return "\n".join(lines) + "\n"

    @staticmethod
    def _caption_config(config):
        """Assemble the engine's CAPTION dict from the split form fields.

        Each template accepts a different shape (TagTemplate / NaturalTemplate /
        MixTemplate / JsonTemplate), so only the fields of the chosen one are read.
        """
        params = config.get("params", {})
        defaults = {p.key: p.default for p in PARAMS}

        def value(key):
            """The form value, or the parameter's own default when the key is absent
            (an older task or a partially filled draft)."""
            return params.get(key, defaults.get(key))

        def text(key, fallback=""):
            raw = value(key)
            return "" if raw is None else (str(raw).strip() or fallback)

        def number(key, fallback, cast=float):
            try:
                return cast(value(key))
            except (TypeError, ValueError):
                return fallback

        def flag(key, fallback=True):
            raw = value(key)
            if isinstance(raw, bool):
                return raw
            spelling = str(raw if raw is not None else "").strip().lower()
            if spelling in ("1", "true", "yes", "on"):
                return True
            if spelling in ("0", "false", "no", "off"):
                return False
            return fallback

        def json_object(key):
            raw = text(key)
            if not raw:
                return {}
            for parse in (json.loads, ast.literal_eval):
                try:
                    value = parse(raw)
                except (ValueError, SyntaxError):
                    continue
                return value if isinstance(value, dict) else {}
            return {}

        template = text("CAPTION_TEMPLATE", "tags")
        if template not in ("tags", "nl", "mix", "json"):
            template = "tags"
        keep = [item.strip() for item in text("CAPTION_KEEP").split(",") if item.strip()]
        caption = {"template": template}
        if template in ("tags", "nl"):
            caption["keep"] = keep
        if template == "tags":
            caption.update({
                "keep_first": number("CAPTION_KEEP_FIRST", 0, int),
                "tag_dropout": number("CAPTION_TAG_DROPOUT", 0.0),
                "group_dropout": number("CAPTION_GROUP_DROPOUT", 0.0),
                "shuffle": flag("CAPTION_SHUFFLE", True),
                "underscores": flag("CAPTION_UNDERSCORES", True),
                "unescape": flag("CAPTION_UNESCAPE", True),
                **json_object("CAPTION_SEPARATORS"),
                **json_object("CAPTION_AFFIX"),
            })
        elif template == "nl":
            caption["fallback"] = json_object("CAPTION_FALLBACK") or {"template": "tags"}
        elif template == "mix":
            caption["p_nl"] = number("CAPTION_P_NL", 0.5)
            caption["nl"] = {"template": "nl", "keep": keep}
            caption["tags"] = {
                "template": "tags",
                "keep": keep,
                "keep_first": number("CAPTION_KEEP_FIRST", 0, int),
                "shuffle": flag("CAPTION_SHUFFLE", True),
                "tag_dropout": number("CAPTION_TAG_DROPOUT", 0.0),
            }
        else:
            fields = [f.strip() for f in text("CAPTION_FIELDS", "artist,rating,tags,caption_nl").split(",") if f.strip()]
            caption["fields"] = fields
        return caption

    def native_config_text(self, config, dataset_config):
        return self._render(config)

    def export_parameters(self, config):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 KohakuMossTrainer 适配器支持")
        notice = ("此文件是 kogine 的 Python 配置；运行时还需同目录的缓存阶段与 kogine 运行器，"
                  "并给训练阶段补上 TEXT_ENCODER / VAE 的 --set（见文件末尾注释）。")
        sets = self._component_sets(self._values(config))
        hint = ""
        if sets:
            pairs = [f"--set {sets[index + 1]}" for index in range(0, len(sets), 2)]
            hint = ("\n\n# 训练阶段另需（config.py 不写这两个键：缓存脚本把同名键绑成自己的工厂）：\n"
                    + "\n".join(f"#   {pair}" for pair in pairs))
        return "# " + notice + "\n" + self._render(config) + hint, (notice,)

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        job_directory = Path(job_directory)
        target = job_directory / "config.py"
        atomic_text(target, self._render(config))
        stages = self.preview_commands(installation, config, target)
        atomic_text(job_directory / "stages.json",
                    json.dumps([list(stage) for stage in stages], ensure_ascii=False, indent=2))
        return target

    # ---- launch ----

    def _values(self, config):
        params = config.get("params", {})
        return {p.key: params.get(p.key, p.default) for p in PARAMS}

    @staticmethod
    def _launcher_target(launcher):
        """Interpreter a generated console-script launcher was built for.

        distlib launchers append an ASCII ``#!<interpreter>`` line before their
        zip payload; POSIX console scripts are text with the same first line.
        Returns None when the file does not look like a generated launcher.
        """
        try:
            data = launcher.read_bytes()
        except OSError:
            return None
        # The shebang sits before the appended zip archive; searching the zip bytes
        # too could match compressed data, so stop at the archive header when present.
        limit = data.find(b"PK\x03\x04")
        marker = data.rfind(b"#!", 0, limit if limit >= 0 else len(data))
        if marker < 0:
            return None
        end = len(data)
        for terminator in (b"\r", b"\n", b"\x00"):
            index = data.find(terminator, marker + 2)
            if index != -1:
                end = min(end, index)
        shebang = data[marker + 2:end].decode("utf-8", "replace").strip()
        # distlib quotes an interpreter path that contains spaces: `#!"C:\...\python.exe"`.
        return shebang.strip('"').strip("'") or None

    @classmethod
    def _kogine_launcher(cls, python):
        """argv prefix that starts kogine, independent of where the project lives.

        Normally the generated console script is used directly. Its embedded
        interpreter path is absolute, so after the project is copied or moved the
        launcher still points at the old location even though the embedded CPython
        itself relocated fine. In that case call the very same entry point through
        the bound interpreter instead of requiring a reinstall; the engine source
        and environment layout stay untouched.
        """
        if not python:
            raise RuntimeError("该引擎尚未绑定 Python 解释器，无法启动 kogine")
        python = str(python)
        name = "kogine.exe" if os.name == "nt" else "kogine"
        directory = Path(python).parent
        expected = os.path.normcase(str(Path(python).resolve()))

        def launcher_is_current(candidate):
            target = cls._launcher_target(candidate)
            if not target:
                return False
            try:
                return os.path.normcase(str(Path(target).resolve())) == expected
            except OSError:
                return os.path.normcase(str(Path(target))) == expected

        for candidate in (directory / name,
                          directory / ("Scripts" if os.name == "nt" else "bin") / name):
            if candidate.is_file() and launcher_is_current(candidate):
                return (str(candidate),)
        return (python, "-c",
                "from kohakuengine.cli import main; raise SystemExit(main())")

    def preview_commands(self, installation, config, native_config):
        python = installation.python_executable
        if not python:
            raise RuntimeError("该引擎尚未绑定 Python 解释器，无法启动 kogine")
        launcher = self._kogine_launcher(python)
        source = Path(installation.source_path)
        config_path = str(native_config)
        # Stages are the cache passes the run actually reads. The engine's cache scripts
        # force their own half on regardless of the config, so an "off" cache must not
        # queue a stage that would build a cache nothing reads.
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        weights = str(values.get("WEIGHTS") or "").strip()
        layout, _ = self._weights_layout(weights) if weights else (None, "")
        variant = str(values.get("VARIANT") or "base").strip() or "base"
        text_cache = str(values.get("TEXT_CACHE") or "offline").strip()
        latent_cache = str(values.get("LATENT_CACHE") or "offline").strip()
        # TEXT_DEVICE is the trainer's placement knob, but the cache pass is where the
        # encoder is loaded: mirror it there, and "cpu" means the plan runs on the host.
        text_device = str(values.get("TEXT_DEVICE") or "").strip() or "cuda"
        text_plan = self._literal_value(values.get("TEXT_ENCODER_PLAN") or {})
        if not isinstance(text_plan, dict):
            text_plan = {}
        if text_device.lower().startswith("cpu"):
            text_plan.setdefault("memory", "cpu")
        stages = []
        if text_cache == "offline":
            sets = [
                "--set", "TEXT_ENCODER_KWARGS=" + repr({
                    "path": self._component_path(
                        weights, values.get("TEXT_ENCODER"), "text_encoder", layout, variant),
                    "device": text_device, "plan": text_plan}),
                "--set", "ENCODE_BATCH=" + repr(int(values.get("ENCODE_BATCH") or 512)),
            ]
            stages.append((*launcher, "run", str(source / CACHE_STAGES[0]),
                           "--config", config_path, *sets))
        if latent_cache == "offline":
            sets = [
                "--set", "IMAGE_ENCODER_KWARGS=" + repr({
                    "path": self._component_path(
                        weights, values.get("VAE"), "vae", layout, variant),
                    "device": "cuda"}),
                # IMAGE_ENCODER exists only in cache_latents.py, never in the trainer.
                # kogine takes the value as-is: a repr() here would arrive as a quoted string.
                "--set", "IMAGE_ENCODER=" + str(values.get("IMAGE_ENCODER")
                                                or "kohakumoss.encoders.vae_encoder"),
                "--set", "ENCODE_BATCH=" + repr(int(values.get("VAE_ENCODE_BATCH") or 8)),
            ]
            stages.append((*launcher, "run", str(source / CACHE_STAGES[1]),
                           "--config", config_path, *sets))
        # TEXT_ENCODER / VAE cannot live in the shared config (the cache scripts bind the
        # names to their own factories), so the trainer gets them on its command line.
        # A bare ComfyUI file name is passed through: the engine resolves it.
        train = [*launcher, "run", str(source / TRAIN_SCRIPT), "--config", config_path,
                 *self._component_sets(values)]
        stages.append(tuple(train))
        return tuple(stages)

    @staticmethod
    def _component_sets(values):
        """The ``--set`` pairs the training stage needs for the component knobs the
        shared config cannot carry (``TEXT_ENCODER`` is a factory in cache_text)."""
        sets = []
        for key in ("TEXT_ENCODER", "VAE"):
            text = str(values.get(key) or "").strip()
            if text:
                sets += ["--set", f"{key}={text}"]
        return sets

    def preview_argv(self, installation, config, native_config):
        return self.preview_commands(installation, config, native_config)[-1]

    @staticmethod
    def _config_flag(native_config, name):
        """A boolean knob of the rendered config, or None when it is absent/unreadable."""
        try:
            tree = ast.parse(Path(native_config).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            return None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(target, "id", "") == name for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                    return node.value.value
        return None

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        stages_path = native_config.parent / "stages.json"
        if not stages_path.is_file():
            raise RuntimeError("缺少 KohakuMossTrainer 运行阶段定义 stages.json")
        commands = tuple(tuple(stage) for stage in json.loads(stages_path.read_text(encoding="utf-8")))
        if not commands:
            raise RuntimeError("KohakuMossTrainer 运行阶段为空")
        environment = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        # The engine only draws its tqdm bar on a TTY or with FORCE_COLOR, and the job's
        # stdout is a pipe: without this the progress bar never appears. The log lines
        # then carry ANSI, which parse_log strips again.
        if self._config_flag(native_config, "PROGRESS") is not False:
            environment["FORCE_COLOR"] = "1"
        return LaunchSpec(commands[-1], Path(installation.source_path),
                          environment, commands[:-1])

    # ---- logs / artifacts ----

    def parse_log(self, line):
        # Its callbacks print `step N [loss=…]` and `steps A-B median X s/step`; the
        # launcher asks for colour (the progress bar needs it), so values arrive wrapped
        # in ANSI and have to be matched without the escape sequences.
        return parse_progress(ANSI_ESCAPE.sub("", line))

    def recovery_capabilities(self, installation, architecture=None):
        # A checkpoint directory holds the adapter, optimiser, RNG and step, so
        # RESUME restores the whole run (not merely a fresh start from weights).
        return RecoveryCapabilities(graceful_stop=True, resume_full_state=True)

    def resume_sources(self, job_directory):
        try:
            config = JobFiles(job_directory).read_job()
        except (OSError, ValueError, KeyError):
            return ()
        root = Path(str(config.get("output_dir") or ""))
        if not root.is_dir():
            return ()
        found = []
        # The engine writes `<OUTPUT_DIR>/<RUN_NAME>/checkpoints/step_XXXXXXXX/`
        # and renames a partially-written directory only when it is complete.
        for path in root.rglob("checkpoints/step_*"):
            if not path.is_dir() or path.name.endswith(".partial"):
                continue
            state = path / "state.json"
            if not state.is_file():
                continue
            step = None
            try:
                payload = json.loads(state.read_text(encoding="utf-8"))
                value = payload.get("step") if isinstance(payload, dict) else None
                step = int(value) if isinstance(value, int) else None
            except (OSError, ValueError, TypeError):
                pass
            if step is None:
                match = re.search(r"step_(\d+)", path.name)
                step = int(match[1]) if match else None
            label = f"第 {step} 步检查点" if step is not None else "训练检查点"
            found.append(ResumeSource(path=path, kind="state", label=label, step=step))
        found.sort(key=lambda source: source.path.stat().st_mtime, reverse=True)
        return tuple(found)

    def resume_parameters(self, source: ResumeSource):
        return {"RESUME": str(source.path)}

    def finalize_artifacts(self, job_directory, python, source_path):
        """No post-run export: the engine writes every export itself.

        ``training/entry.py`` writes the final files under ``<output>/export/`` and, with
        ``CHECKPOINT_EXPORT`` (its default), the same files into every checkpoint's
        ``export/``. Re-running the exporter from here duplicated that work and drifted
        from ``SAVE_FORMAT`` / ``SAVE_FUSED_QKV``; ``collect_artifacts`` surfaces the
        engine's own files.
        """
        return ()

    def collect_artifacts(self, job_directory):
        config = json.loads((Path(job_directory) / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        if not root.is_dir():
            return ()
        found = []
        for path in root.rglob("*"):
            # `"cache" in parts` compared whole names, so `moss-cache` slipped through:
            # match the substring on every parent directory instead.
            if not path.is_file() or any(
                    "cache" in part.lower() for part in path.relative_to(root).parts[:-1]):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                found.append(Artifact(path, "checkpoint", True))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
