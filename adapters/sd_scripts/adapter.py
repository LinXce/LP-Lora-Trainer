"""Adapter for the native sd-scripts LoRA training entry points."""
import ast
import json
import math
import re
from pathlib import Path
from adapters.base import (
    InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact,
    ResumeSource,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.recovery import accelerate_resume_sources
from adapters.sd_scripts.params import ARCHITECTURES, NETWORK_MODULES, PARAMS, RELEASE_KEYS, parameters_for
from adapters.sd_scripts.native_options import options_for
from adapters import extra_params
from adapters.sampling import prompt_file, write_prompts, exported_prompt_files
from app.schemas.training import LaunchSpec
from app.services.common import safe_file_stem
from app.services.jobs import JobFiles


def scripts_root(path):
    path = Path(path)
    return path / "sd-scripts" if (path / "sd-scripts" / "train_network.py").is_file() else path


def _strip_weight_prefix(name):
    """Mirror the engine's rename hook, so prefix variants compare equal."""
    for prefix in ("net.", "model.diffusion_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def safetensors_keys(path):
    """Tensor names in a .safetensors file, or None when it is not one.

    Only the header is read. The engine fails much later with "Missing keys in
    checkpoint" when the base model is the wrong kind of file, which is worth
    catching before a run starts.
    """
    try:
        p = Path(path or "")
        if p.suffix.lower() != ".safetensors" or not p.is_file():
            return None
        with p.open("rb") as f:
            size = int.from_bytes(f.read(8), "little")
            if size <= 0 or size > 64 * 1024 * 1024:
                return None
            header = json.loads(f.read(size))
        return {_strip_weight_prefix(name) for name in header if name != "__metadata__"}
    except (OSError, ValueError, UnicodeError):
        return None


# sd-scripts entry per LoRA network module (sd1/sd2/sdxl share ``networks.lora``).
MODULE_SCRIPTS = {
    "networks.lora_sd3": "sd3_train_network.py",
    "networks.lora_flux": "flux_train_network.py",
    "networks.lora_hunyuan_image": "hunyuan_image_train_network.py",
    "networks.lora_lumina": "lumina_train_network.py",
    "networks.lora_anima": "anima_train_network.py",
}

# build_launch only receives the rendered config file, and SD1/SD2/SDXL share
# one network module, so the entry script is recorded as a comment instead of
# being inferred from a parameter users can toggle (the old SDXL inference read
# ``network_train_unet_only``, which is exactly the flag that must not be forced).
_ARCHITECTURE_MARKER = re.compile(r"^#\s*lp-architecture:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)

# Pinned requirements that the README installs from the PyTorch wheel index.
_TORCH_REQUIREMENT_RE = re.compile(r"^(?:torch|torchvision)(?=$|[\s<>=~!;(@\[])", re.IGNORECASE)

# Periodic LoRA saves are "{name}-step{step:08d}"; epoch saves are
# "{name}-{epoch:06d}" and carry no training step.
_ARTIFACT_STEP_RE = re.compile(r"-step(\d+)$")


def list_values(value):
    """Split one UI text value into the engine's ``nargs="*"`` list items.

    The engine declares optimizer_args / lr_scheduler_args / network_args with
    nargs="*" and iterates them; a TOML string was walked character by character
    and crashed on the first "key=value" split. Quotes group values containing
    spaces; any whitespace (including newlines) separates items.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item) != ""]
    tokens, current, quote = [], [], ""
    for char in str(value or ""):
        if quote:
            if char == quote:
                quote = ""
            else:
                current.append(char)
        elif char in "'\"":
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if quote:
        raise ValueError(f"引号 {quote} 没有成对出现")
    if current:
        tokens.append("".join(current))
    return tokens


def native_list(key, value):
    if key == "base_weights":
        # The extra-argument parser has already removed grouping quotes. Preserve a path containing spaces.
        return value if isinstance(value, list) else [value]
    if isinstance(value, bool):
        raise ValueError("需要数值")
    items = value if isinstance(value, (list, tuple)) else str(value).split()
    return [float(item) for item in items]


def toml_value(value):
    """Render one Python value the way the engine's TOML config expects it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


class KohyaAdapter(EngineAdapter):
    custom_write_dir_keys = ("logging_dir",)
    model_input_keys = ("network_weights", "vae", "qwen3", "text_encoder", "clip_l", "clip_g",
                        "t5xxl", "ae", "gemma2", "byt5", "llm_adapter_path")

    def legacy_output_issue(self, config):
        return extra_params.legacy_output_issue(config, config.get("output_name") or config.get("name") or "lora")

    engine_id = "kohya"
    # sd-scripts creates the output folder itself when it first saves
    # (os.makedirs(..., exist_ok=True)), so LP must not pre-create it: doing that
    # with exist_ok=False failed with WinError 183 on a re-run of the same name.
    creates_output_dir = True
    display_name = "kohya sd-scripts"
    installation_help = (
        "由 LP 在引擎目录内准备独立 Python：按 sd-scripts README 的步骤先安装与该显卡匹配的 "
        "torch/torchvision（cu124 或 cu128），再按 requirements.txt 安装其余依赖；"
        "该引擎目录本身就是训练代码，没有额外的 GUI 包装层。"
    )
    default_installation_source = "cu124"
    adapter_version = "0.3.0"
    native_format = "toml"
    submittable = True
    # Declared explicitly: the base default is empty so a new engine never inherits a
    # list of modules it does not use. torchvision is a training-time import
    # (library/dataset.py), so a missing one must fail diagnosis, not the first run.
    # "yaml" is deliberately absent: sd-scripts imports it nowhere and its
    # requirements.txt does not install pyyaml, so requiring it failed every real
    # environment while the fake one passed.
    diagnostic_modules = ("accelerate", "transformers", "safetensors", "diffusers", "toml", "torchvision")
    environment_strategy = "embedded"
    environment_dir = ".lp/python"
    environment_candidates = (".venv", "venv")
    python_version = "3.11"
    torch_index_mode = "env"
    installation_source_help = (
        "torch/torchvision 从官方 PyTorch wheel 源（或所选 wheel 镜像的 find-links 目录）按该 CUDA "
        "标签安装；其余依赖走 PyPI 镜像。请选择不高于显卡驱动支持上限的 CUDA 版本。"
    )
    # CUDA wheel tags named in the sd-scripts README, with the torch / torchvision
    # versions it installs: ``requirements.txt`` deliberately has no PyTorch
    # entries, so LP installs these first instead of letting diffusers pull a
    # CPU-only torch from PyPI and leaving torchvision absent.
    TORCH_WHEEL_PINS = {
        "cu124": ("2.6.0", "0.21.0"),
        "cu128": ("2.8.0", "0.23.0"),
    }
    TORCH_WHEEL_INDEX = "https://download.pytorch.org/whl/{tag}"

    def install_blockers(self, hardware):
        import os
        if os.name != "nt":
            return ("Kohya 的 PyTorch CUDA wheel 与适配器安装流程仅支持 Windows",)
        return ()

    def installation_plan(self, source_path, python, torch_source, uv=None, torch_mirror_url=None):
        """Install the README's PyTorch step, then the declared requirements.

        The upstream README installs ``torch``/``torchvision`` from a CUDA wheel
        index first and only then ``requirements.txt`` (which has no PyTorch
        entries); this reproduces exactly those two steps so LP does not leave a
        CPU-only torch behind. A legacy kohya_ss wrapper requirements file that
        pins torch is still honoured when no CUDA source is selected.
        """
        import os
        if os.name != "nt":
            raise ValueError("当前 Kohya 安装接口仅支持 Windows")
        entries = list(self.requirement_entries(source_path))
        if not entries:
            raise ValueError("缺少 Kohya requirements.txt 或其中没有可安装的依赖条目")
        torch_pins, extra_index = self._torch_packages(torch_source)
        if torch_pins:
            # One explicit CUDA pin: never let a legacy requirements file add a
            # second, different torch/torchvision requirement to the same call.
            entries = [entry for entry in entries if not _TORCH_REQUIREMENT_RE.match(entry)]
        # The selected wheel mirror serves the CUDA-tagged files as flat dirs.
        links = []
        if torch_mirror_url:
            for package in ("torch", "torchvision"):
                links += ["--find-links", f"{torch_mirror_url.rstrip('/')}/{package}/"]
            # The mirror is the source for the wheels: requiring the official
            # index as well would fail on machines that can only reach the mirror.
            extra_index = []
        packages = [*torch_pins, *entries]
        if (source_path / "setup/setup_common.py").is_file():
            # Legacy wrappers still import pkg_resources, removed by setuptools 82.
            packages.append("setuptools<82")
        root = source_path.resolve()
        for relative in self.local_requirement_paths(source_path):
            target = (root / relative).resolve()
            if target.is_dir() and target.is_relative_to(root):
                packages += ["-e", str(target)]
        if uv:
            # uv resolves this dependency graph in seconds where pip can sit
            # silent for minutes, and it is LP's own tool, so the command stays
            # ours; the selected PyPI mirror reaches it through UV_INDEX_URL.
            return InstallationPlan("Kohya 依赖安装（由 LP 执行，使用 uv 与所选镜像源）",
                                    ((str(uv), "pip", "install", "--python", str(python),
                                      *links, *extra_index, *packages),))
        fallback_flag = [] if getattr(self, "embedded_mode", False) else ["--require-virtualenv"]
        return InstallationPlan("Kohya 依赖安装（由 LP 执行，pip 降级路径）",
                                ((python, "-u", "-m", "pip", "install", *fallback_flag,
                                  "--no-input", *links, *extra_index, *packages),))

    def _torch_packages(self, torch_source):
        """``(pins, index_args)`` for the selected CUDA tag; empty for "existing"."""
        tag = str(torch_source or "").strip().lower()
        if tag in ("", "existing"):
            return [], []
        versions = self.TORCH_WHEEL_PINS.get(tag)
        if versions is None:
            raise ValueError(
                f"不支持的 CUDA 源：{torch_source}；可用 {' / '.join(self.TORCH_WHEEL_PINS)} 或 existing")
        torch_version, torchvision_version = versions
        return (
            [f"torch=={torch_version}+{tag}", f"torchvision=={torchvision_version}+{tag}"],
            ["--extra-index-url", self.TORCH_WHEEL_INDEX.format(tag=tag)],
        )

    def detect(self, source_path):
        # The engine is sd-scripts itself: the training entry plus its library must be
        # in this directory. The old GUI-wrapper layout is no longer accepted, because
        # the wrapper is not the engine.
        root = scripts_root(source_path)
        complete = (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()
        return DetectionResult(self.engine_id, complete,
                               "sd-scripts training source (train_network.py + library)")

    # Requirements files a legacy kohya_ss wrapper clone pins PyTorch in. A
    # standalone sd-scripts clone has none of them: its README installs PyTorch
    # separately, which is what TORCH_WHEEL_PINS above reproduces.
    TORCH_REQUIREMENTS = (
        "requirements_pytorch_windows.txt",
        "requirements_windows_torch2.txt",
        "requirements_pytorch.txt",
    )
    _CUDA_TAG_RE = re.compile(r"\bcu\d{3}\b", re.IGNORECASE)
    # ``-e ./sd-scripts`` and plain ``./sd-scripts``; anything else is a package.
    _LOCAL_REQUIREMENT_RE = re.compile(r"^(?:-e\s+)?(\.|\.{1,2}[/\\][^\s#]+)$")

    def torch_index_tag(self, source_path):
        """Read the CUDA tag this engine's requirements install torch from.

        Mirrors must use the same tag: a mirror built for the driver's maximum
        (e.g. cu134) cannot serve ``torch==2.7.0+cu128`` at all.
        """
        for name in self.TORCH_REQUIREMENTS:
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if not any(marker in line for marker in ("--index-url", "--extra-index-url", "+cu")):
                    continue
                match = self._CUDA_TAG_RE.search(line)
                if match:
                    return match.group(0).lower()
        return None

    def pinned_wheel_requirements(self, source_path):
        """The ``+cuXXX`` pins from the torch requirements file, in order."""
        for name in self.TORCH_REQUIREMENTS:
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            pins = []
            for line in text.splitlines():
                entry = line.strip()
                if not entry or entry.startswith(("#", "-")) or "==" not in entry:
                    continue
                if "+cu" in entry.lower():
                    pins.append(entry)
            if pins:
                return tuple(pins)
        return ()

    def requirement_files(self, source_path):
        """Requirement files LP installs, in order.

        A standalone sd-scripts clone only ships ``requirements.txt`` (its
        README installs PyTorch separately). The wrapper-era torch file stays
        first for legacy kohya_ss copies that still carry one.
        """
        names = ("requirements_pytorch_windows.txt", "requirements.txt")
        return tuple(name for name in names if (source_path / name).is_file())

    def local_requirement_paths(self, source_path):
        """Entries like ``-e ./sd-scripts``: the engine installs its own tree."""
        paths = []
        for name in ("requirements.txt",) + self.TORCH_REQUIREMENTS:
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                match = self._LOCAL_REQUIREMENT_RE.match(line.strip())
                if match:
                    paths.append(match.group(1).replace("\\", "/"))
        return tuple(dict.fromkeys(paths))

    def installation_sources(self, source_path):
        """CUDA tags LP can install PyTorch from for this engine.

        ``requirements.txt`` deliberately carries no PyTorch entries, so the
        README's documented tags are the install sources. A legacy wrapper
        requirements file that pins a tag still wins, so old clones keep their
        own choice.
        """
        pinned = self.torch_index_tag(source_path)
        if pinned:
            return (pinned,)
        return tuple(self.TORCH_WHEEL_PINS)

    def requirement_entries(self, source_path):
        """Named pip requirements, with upstream ranges, extras and markers intact."""
        entries = []
        for name in self.requirement_files(source_path):
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw in text.splitlines():
                line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
                if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(?:$|[<>=~!;(@\[])", line):
                    continue
                entries.append(line)
        return tuple(dict.fromkeys(entries))

    def requirement_skips(self, source_path):
        # Source index flags and local editables are handled separately by the installer.
        return ()

    def source_issues(self, source_path):
        root = scripts_root(source_path)
        issues = []
        if not ((root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()):
            issues.append(
                "Kohya 缺少完整 sd-scripts 训练源码（train_network.py + library/train_util.py）；"
                "请把完整的 sd-scripts 仓库复制到该目录，或使用“Git 拉取引擎”重新拉取。"
                "若这是 kohya_ss 包装仓库，可点击“补齐训练源码”初始化其 sd-scripts 子模块"
            )
        return tuple(issues)

    source_repair_label = "补齐 sd-scripts 训练源码"

    clone_url = "https://github.com/kohya-ss/sd-scripts.git"
    # The upstream version this adapter is written against.
    clone_branch = "main"
    clone_commit = "8c088eba9cd5d10a5d00dd58f0d83063fa9d57ca"

    @staticmethod
    def _declared_submodule(source_path):
        """``(path, url)`` for the sd-scripts entry in a legacy wrapper's .gitmodules."""
        from configparser import ConfigParser, Error
        config = ConfigParser(interpolation=None)
        try:
            config.read(source_path / ".gitmodules", encoding="utf-8")
        except (OSError, UnicodeError, Error):
            return None
        for section in config.sections():
            if config.get(section, "path", fallback="").strip() == "sd-scripts":
                url = (config.get(section, "url", fallback="") or "").strip()
                return "sd-scripts", url or None
        return None

    def source_repair(self, source_path):
        """Complete a legacy kohya_ss wrapper checkout from its declared submodule.

        A standalone sd-scripts clone has no .gitmodules entry, so this returns
        None and the UI simply asks the user to copy the full repository. For a
        real wrapper checkout the recorded submodule commit is restored; a
        copied wrapper without ``.git`` gets the declared URL cloned instead.
        """
        if not self.source_issues(source_path):
            return None
        declared = self._declared_submodule(source_path)
        if declared is None:
            return None
        _, url = declared
        if (source_path / ".git").exists():
            return ("git", "-c", "core.fsmonitor=false", "submodule", "update", "--init", "--recursive")
        scripts = source_path / "sd-scripts"
        try:
            empty = not scripts.exists() or not any(scripts.iterdir())
        except OSError:
            return None
        if url and empty:
            return ("git", "-c", "core.fsmonitor=false", "clone", "--depth", "1", url, str(scripts))
        return None

    def architectures(self, source_path):
        root = scripts_root(source_path)
        return tuple(a for a in ARCHITECTURES if (root / a.script).is_file())

    def capabilities(self, installation):
        return self.capabilities_for_architecture(installation, None)

    def capabilities_for_architecture(self, installation, architecture):
        root = scripts_root(installation.source_path)
        params = parameters_for(architecture)
        if not (root / "moss_image_train_network.py").is_file():
            params = tuple(p for p in params if p.key not in RELEASE_KEYS)
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(params), submittable=self.submittable)

    def validate(self, installation, config):
        """Validation in three tiers, so a rule only runs where it belongs.

        * generic: holds for every engine and every base model (known keys, types,
          ranges, enum membership)
        * engine: this engine across all of its base models (how it renders the
          config, packages its own defaults and options need)
        * base model: one architecture of this engine only (that model family's
          layout and limits)
        """
        issues = []
        architecture = config.get("architecture")
        entry = next((a for a in ARCHITECTURES if a.id == architecture), None)
        if not entry:
            return (ValidationIssue("architecture", "当前源码没有对应的训练入口"),)
        if not (scripts_root(installation.source_path) / entry.script).is_file():
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        submitted = config.get("params", {})
        malformed = self.generic_param_issues(submitted, architecture)
        if malformed:
            issues.extend(malformed)
            return tuple(issues)
        values = {p.key: p.default for p in parameters_for(architecture)}
        values.update({k: v for k, v in submitted.items() if v is not None})
        if not (scripts_root(installation.source_path) / "moss_image_train_network.py").is_file():
            for option in parameters_for(architecture):
                if option.key in RELEASE_KEYS and submitted.get(option.key) not in (None, "", option.default):
                    issues.append(ValidationIssue(option.key, "该选项需要 sd-scripts 0.13 或更新的引擎源码"))
        extras = str(values.get(extra_params.PARAM_KEY) or "")
        for severity, message in extra_params.problems(extras, extra_params.provided_keys(
                parameters_for(architecture), architecture, values, skip_false=False,
                extra=("pretrained_model_name_or_path", "dataset_config", "output_dir", "output_name", "network_module",
                       "config_file", "v2")), bound_output=config.get("output_layout") == "task_id"):
            issues.append(ValidationIssue(extra_params.PARAM_KEY, message, severity))
        if extras and not any(i.field == extra_params.PARAM_KEY and i.severity == "error" for i in issues):
            additions = extra_params.parse(extras)
            source_options = options_for(scripts_root(installation.source_path), entry.script)
            for key, value in additions.items():
                message = self.custom_value_issue(key, value, source_options, architecture)
                if message:
                    issues.append(ValidationIssue(extra_params.PARAM_KEY, f"--{key}：{message}"))
                elif key == "xformers":
                    values[key] = "xformers" if value else "none"
                else:
                    values[key] = value
        if any(i.field == extra_params.PARAM_KEY and i.severity == "error" for i in issues):
            return tuple(issues)
        issues += self.engine_param_issues(installation, config, values)
        issues += self.base_model_param_issues(installation, config, values)
        return tuple(issues)

    # ---- 通用参数校验：所有引擎、所有底模 ----
    def generic_param_issues(self, values, architecture=None):
        issues = []
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in PARAMS:
            if not p.applies_to(architecture) and values.get(p.key) not in (None, "", p.default):
                issues.append(ValidationIssue(p.key, "此参数不适用于所选底模类型"))
        for p in parameters_for(architecture):
            if not p.native and p.unsupported_reason:
                if values.get(p.key) not in (None, "", p.default):
                    issues.append(ValidationIssue(p.key, p.unsupported_reason))
                continue
            value = values.get(p.key, p.default)
            message = check_value(p, value)
            # Anima gets the engine-specific max_workers explanation below; do not
            # also report the generic range error for the same value.
            if message and not (p.key == "vae_batch_size" and architecture == "anima" and value == 0):
                issues.append(ValidationIssue(p.key, message))
        resolution = values.get("resolution")
        if isinstance(resolution, int) and resolution % 64:
            issues.append(ValidationIssue("resolution", "分辨率必须是 64 的倍数"))
        return issues

    @staticmethod
    def custom_value_issue(key, value, source_options, architecture):
        if key in {"network_module", "v2", "pretrained_model_name_or_path", "dataset_config", "config_file",
                   "resolution", "caption_extension", "enable_bucket", "extra_params", "run_control", "run_dir"}:
            return "该设置由架构、数据集表单或任务运行器管理"
        form = next((p for p in parameters_for(architecture) if p.key == key), None)
        if not form and any(p.key == key for p in PARAMS):
            return "该表单参数不适用于所选架构"
        if source_options is not None:
            if key not in source_options:
                return "该训练入口没有此参数"
            kind, choices, _ = source_options[key]
            if key in ("base_weights", "base_weights_multiplier", "text_encoder_lr"):
                try:
                    items = native_list(key, value)
                except (ValueError, TypeError):
                    return "参数列表含有无效值"
            else:
                items = [value]
            for item in items:
                if kind in ("int", "int_or_float", "float"):
                    if isinstance(item, bool) or not isinstance(item, (int, float)) or (isinstance(item, float) and not math.isfinite(item)):
                        return "需要有限数值"
                    if kind == "int" and not isinstance(item, int):
                        return "需要整数"
                if kind == "bool" and type(item) is not bool:
                    return "需要 true 或 false"
                if kind in ("str", "Path") and not isinstance(item, str):
                    return "需要文本值"
                if choices and item not in choices:
                    return "不在引擎的可选范围内"
        if key == "xformers":
            return None if isinstance(value, bool) else "需要 true 或 false"
        if form:
            if key == "text_encoder_lr":
                try:
                    return next((issue for v in native_list(key, value) if (issue := check_value(form, v))), None)
                except ValueError:
                    return "需要有效的学习率"
            if key == "optimizer_type" and isinstance(value, str) and re.fullmatch(r"(?:[a-zA-Z_]\w*\.)+[a-zA-Z_]\w*", value):
                return None
            return check_value(form, value)
        if key in {"token_warmup_step", "lr_decay_steps"} and isinstance(value, (float, int)) and value < 0:
            return "必须大于或等于 0"
        return None

    # ---- 同引擎参数校验：本引擎的所有底模 ----
    def engine_param_issues(self, installation, config, values):
        issues = []
        if values.get("network_train_unet_only") and values.get("network_train_text_encoder_only"):
            issues.append(ValidationIssue("network_train_text_encoder_only", "不能同时仅训练 UNet 和仅训练文本编码器"))
        if values.get("persistent_data_loader_workers") and not values.get("max_data_loader_n_workers"):
            issues.append(ValidationIssue("persistent_data_loader_workers", "持久化 DataLoader 要求数据加载进程数至少为 1"))
        if values.get("adaptive_noise_scale") is not None and values.get("noise_offset") is None:
            issues.append(ValidationIssue("adaptive_noise_scale", "自适应噪声尺度需要同时设置 Noise Offset"))
        if values.get("v_pred_like_loss") and values.get("v_parameterization"):
            issues.append(ValidationIssue("v_pred_like_loss", "该损失不能与 v_parameterization 同时启用"))
        if values.get("lr_scheduler") == "adafactor" and values.get("optimizer_type") != "Adafactor":
            issues.append(ValidationIssue("lr_scheduler", "adafactor 调度器要求使用 Adafactor 优化器"))
        for key in ("lr_warmup_steps", "lr_decay_steps"):
            steps = values.get(key)
            if isinstance(steps, float) and steps >= 1 and not steps.is_integer():
                issues.append(ValidationIssue(key, "大于或等于 1 时必须填写整数步数"))
        if (values.get("lr_warmup_steps") and values.get("lr_scheduler") == "constant"
                and not values.get("lr_scheduler_type") and "schedulefree" not in str(values.get("optimizer_type")).lower()):
            issues.append(ValidationIssue("lr_warmup_steps", "constant 调度器不接受预热步数，请选择支持预热的调度器"))
        if (values.get("use_8bit_adam") or values.get("use_lion_optimizer")) and (
                (values.get("use_8bit_adam") and values.get("use_lion_optimizer")) or values.get("optimizer_type") != "AdamW"):
            issues.append(ValidationIssue("optimizer_type", "旧版优化器开关与所选优化器冲突"))
        if values["min_bucket_reso"] > values["max_bucket_reso"]:
            issues.append(ValidationIssue("max_bucket_reso", "最大分桶分辨率须大于或等于最小分桶分辨率"))
        if values.get("enable_bucket") and values.get("resolution"):
            step = values["bucket_reso_steps"]
            minimum = values["min_bucket_reso"] // step * step
            maximum = (values["max_bucket_reso"] + step - 1) // step * step
            if not minimum <= values["resolution"] <= maximum:
                issues.append(ValidationIssue("resolution", f"启用分桶后，训练分辨率须在 {minimum} 到 {maximum} 之间"))
        if (values.get("min_timestep") or 0) >= (values.get("max_timestep") if values.get("max_timestep") is not None else 1000):
            issues.append(ValidationIssue("max_timestep", "结束时间步必须大于起始时间步"))
        for key in ("network_args", "optimizer_args", "lr_scheduler_args"):
            try:
                for item in list_values(values.get(key)):
                    name, raw = item.split("=")
                    if not name or not raw:
                        raise ValueError
                    if key != "network_args":
                        ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                issues.append(ValidationIssue(key, "每项请填写 参数名=值；优化器及调度器的值必须是有效的 Python 字面量"))
        # Optional packages this engine's own defaults and choices need.
        if str(values.get("optimizer_type") or "").endswith("8bit") and self.site_package_installed(installation, "bitsandbytes") is False:
            issues.append(ValidationIssue(
                "optimizer_type", "该优化器需要 bitsandbytes，但引擎环境里没有：请改用 AdamW，或根据引擎 requirements.txt 安装 bitsandbytes"))
        if str(values.get("xformers") or "") == "xformers" and self.site_package_installed(installation, "xformers") is False:
            issues.append(ValidationIssue(
                "xformers", "该注意力实现需要 xformers，但引擎环境里没有：请改用 SDPA（sdp），或在该引擎环境里自行安装 xformers"))
        # The engine reads the TOML values through an argparse Namespace, which
        # does not re-validate choices, so the accepted values are enforced here.
        if values.get("max_token_length") not in (None, 75, 150, 225):
            issues.append(ValidationIssue(
                "max_token_length", "引擎只接受 150 或 225（75 表示引擎默认，会省略该键）"))
        if values.get("fused_backward_pass"):
            if str(values.get("optimizer_type") or "") != "Adafactor":
                issues.append(ValidationIssue(
                    "fused_backward_pass", "引擎要求 fused_backward_pass 与优化器 Adafactor 搭配，否则直接断言失败"))
            if values.get("gradient_accumulation_steps", 1) != 1:
                issues.append(ValidationIssue(
                    "fused_backward_pass", "引擎要求 fused_backward_pass 时梯度累积步数必须为 1"))
        if values.get("masked_loss"):
            issues.append(ValidationIssue(
                "masked_loss",
                "引擎只在数据带 alpha 掩码或 conditioning 图时应用掩码损失；LP 当前不提供这些输入，开启后不会生效",
                "warning"))
        if values.get("cache_text_encoder_outputs_to_disk") and not values.get("cache_text_encoder_outputs"):
            issues.append(ValidationIssue(
                "cache_text_encoder_outputs", "引擎会自动启用「缓存 TE 输出」；建议显式勾选两者以免误解", "warning"))
        # Precision combinations the engine asserts on (train_network.py:1221-1240).
        # They matter more now that mixed_precision defaults to the engine's "no".
        precision = str(values.get("mixed_precision") or "no")
        if values.get("full_fp16") and precision != "fp16":
            issues.append(ValidationIssue(
                "full_fp16", "引擎要求 full_fp16 与混合精度 fp16 搭配，否则直接断言失败"))
        if values.get("full_bf16") and precision != "bf16":
            issues.append(ValidationIssue(
                "full_bf16", "引擎要求 full_bf16 与混合精度 bf16 搭配，否则直接断言失败"))
        if values.get("fp8_base") and precision == "no":
            issues.append(ValidationIssue(
                "fp8_base", "引擎要求 fp8_base 时混合精度为 fp16 或 bf16，否则直接断言失败"))
        return issues

    # ---- 引擎下底模参数校验：仅本引擎的某个底模 ----
    def base_model_param_issues(self, installation, config, values):
        issues = []
        architecture = config.get("architecture")
        if architecture == "sdxl":
            issues.append(ValidationIssue("architecture", "SDXL 建议使用 1024 分辨率；当前适配仅支持单 GPU LoRA", "warning"))
        # sd-scripts asserts this combination away, so fail here instead of in the engine.
        if values.get("blocks_to_swap") and values.get("cpu_offload_checkpointing"):
            issues.append(ValidationIssue(
                "cpu_offload_checkpointing", "Block Swap 与梯度检查点卸载到 CPU 不能同时启用；请只保留一个"))
        # Both offloading implementations exclude CPU checkpoints and block swap.
        if architecture in ("anima", "moss_image") and values.get("unsloth_offload_checkpointing"):
            if values.get("cpu_offload_checkpointing"):
                issues.append(ValidationIssue(
                    "unsloth_offload_checkpointing", "Unsloth 卸载与梯度检查点卸载到 CPU 不能同时启用；请只保留一个"))
            if values.get("blocks_to_swap"):
                issues.append(ValidationIssue(
                    "unsloth_offload_checkpointing", "Unsloth 卸载与 Block Swap 不能同时启用；请只保留一个"))
        # Invariants the engine enforces with asserts (see library/dataset.py and
        # the per-architecture assert_extra_args). Reporting them here keeps a
        # run from dying after model loading.
        required_steps = {"sd1": 64, "sd2": 64, "sdxl": 32, "sd3": 32, "flux1": 32,
                          "hunyuan_image": 32, "lumina": 16, "anima": 16, "moss_image": 16}.get(architecture)
        steps = values.get("bucket_reso_steps", 64)
        if required_steps and isinstance(steps, int) and steps % required_steps:
            issues.append(ValidationIssue(
                "bucket_reso_steps",
                f"该底模要求分桶步长能被 {required_steps} 整除，否则引擎直接断言；当前为 {steps}"))
        cache_latents = values.get("cache_latents") or values.get("cache_latents_to_disk")
        if cache_latents and values.get("train_inpainting"):
            issues.append(ValidationIssue("cache_latents", "随机生成修复掩码需要原图，不能同时缓存 Latents"))
        if cache_latents and (values.get("color_aug") or values.get("random_crop")):
            issues.append(ValidationIssue(
                "cache_latents", "缓存 Latents 与颜色增强/随机裁剪不能同时启用，引擎会直接断言；请关闭其一"))
        cache_text = values.get("cache_text_encoder_outputs") or values.get("cache_text_encoder_outputs_to_disk")
        dropout_allowed = architecture == "moss_image" or (
            architecture == "anima" and (scripts_root(installation.source_path) / "moss_image_train_network.py").is_file())
        if cache_text and (values.get("shuffle_caption") or values.get("caption_tag_dropout_rate")
                or values.get("token_warmup_step") or
                (not dropout_allowed and (values.get("caption_dropout_rate") or 0) > 0)):
            issues.append(ValidationIssue(
                "cache_text_encoder_outputs",
                "缓存 TE 输出与当前架构所选的 Caption 随机化方式不兼容"))
        if architecture in {"flux1", "lumina", "hunyuan_image", "anima", "moss_image"} and (
                values.get("block_swap_h2d_only") and not (values.get("gradient_checkpointing") or
                                                          values.get("unsloth_offload_checkpointing"))):
            issues.append(ValidationIssue("block_swap_h2d_only", "仅向 GPU 传输换出块需要梯度检查点"))
        if architecture in {"flux1", "hunyuan_image", "anima", "moss_image"} and (
                values.get("timestep_sampling") == "truncated_logit_normal" and values.get("timestep_cutoff") == 0):
            issues.append(ValidationIssue("timestep_cutoff", "截断分布上限必须大于 0"))
        if architecture in ("anima", "moss_image"):
            if values.get("compile") and values.get("torch_compile"):
                issues.append(ValidationIssue("compile", "分块编译不能与 torch_compile 同时启用"))
            if values.get("compile") and values.get("compile_fullgraph") and values.get("split_attn"):
                issues.append(ValidationIssue("compile_fullgraph", "整图编译与分段注意力不能同时使用"))
        if architecture == "moss_image":
            if (cache_text or values.get("text_encoder_blocks_to_swap")) and not values.get("network_train_unet_only"):
                issues.append(ValidationIssue("network_train_unet_only", "MOSS 文本缓存或文本编码器换块需要仅训练 DiT"))
            if values.get("fp8_base"):
                issues.append(ValidationIssue("fp8_base", "MOSS-Image 使用 ConvRot INT8 量化，不支持 FP8 底模"))
        if values.get("replica_devices") and architecture in ("moss_image", "anima"):
            devices = [s.strip() for s in str(values["replica_devices"]).split(",") if s.strip()]
            if not devices or not all(re.fullmatch(r"(?:cuda:)?[0-9]+", d) for d in devices):
                issues.append(ValidationIssue("replica_devices", "填写附加 CUDA 显卡编号，例如 1,2 或 cuda:1,cuda:2"))
            if not (cache_latents and cache_text and values.get("network_train_unet_only")):
                issues.append(ValidationIssue("replica_devices", "副本训练要求缓存 Latents、缓存 TE 输出并仅训练 DiT"))
            if values.get("mixed_precision") == "fp16" or values.get("fp8_base"):
                issues.append(ValidationIssue("replica_devices", "副本训练支持 bf16 或全精度，无法使用 fp16 或 FP8 底模"))
        if cache_text and architecture in {"anima", "sdxl"} and not values.get("network_train_unet_only"):
            issues.append(ValidationIssue("network_train_unet_only", "该架构缓存 TE 输出时要求仅训练 UNet / DiT"))
        if values.get("scale_v_pred_loss_like_noise_pred") and not values.get("v_parameterization"):
            issues.append(ValidationIssue(
                "scale_v_pred_loss_like_noise_pred",
                "引擎仅在启用 v_parameterization 时允许该项；LP 未提供该开关，请用「自定义参数」写 --v_parameterization 或改用原生配置"))
        if architecture == "anima":
            # The latent cache caps its image-loading pool with this value, so 0 makes
            # the engine die with "max_workers must be greater than 0".
            if values.get("vae_batch_size", 1) == 0:
                issues.append(ValidationIssue(
                    "vae_batch_size", "VAE Batch Size 不能为 0：该底模的 latent 缓存会把它当作并发上限，引擎会报 max_workers must be greater than 0"))
            # The DiT always builds its LLM adapter and the engine aborts on missing keys.
            keys = safetensors_keys(config.get("base_model_path"))
            if keys is not None:
                if not any(key.startswith("x_embedder.") for key in keys):
                    issues.append(ValidationIssue(
                        "base_model_path", "该文件不包含 Anima DiT 权重（缺少 x_embedder 等键）：请确认没有把 VAE、文本编码器或 LoRA 当成底模"))
                elif not any(key.startswith("llm_adapter.") for key in keys) and not str(values.get("llm_adapter_path") or "").strip():
                    issues.append(ValidationIssue(
                        "llm_adapter_path", "该 Anima 权重不含 LLM Adapter（缺少 llm_adapter.*）：请在此填入适配器文件，或换用包含 llm_adapter.out_proj.weight 的 DiT 权重"))
        # These entries hard-require extra frozen components; leaving the field empty
        # makes the engine call load_safetensors(None) while loading the model.
        for p in parameters_for(architecture):
            visible = not p.visible_when or values.get(p.visible_when[0]) in p.visible_when[1]
            if visible and architecture in p.required_for and not str(values.get(p.key) or "").strip():
                issues.append(ValidationIssue(p.key, f"请填写{p.label}；该训练入口需要此项"))
        if architecture == "flux1" and values.get("model_type") == "chroma" and not values.get("apply_t5_attn_mask"):
            issues.append(ValidationIssue("apply_t5_attn_mask", "Chroma 需要启用 T5 注意力掩码"))
        return issues

    def site_package_installed(self, installation, name):
        """Whether the engine environment has a package, without importing it.

        ``None`` means "cannot tell", which must never be reported as missing.
        """
        try:
            exe = Path(installation.python_executable).resolve()
        except (AttributeError, TypeError, OSError):
            return None
        candidates = (exe.parent / "Lib" / "site-packages", exe.parent.parent / "Lib" / "site-packages")
        site = next((path for path in candidates if path.is_dir()), None)
        if site is None:
            return None
        return any(site.glob(name + "*"))

    def caption_cache_notices(self):
        # kohya writes caches next to the images and validates their content by
        # default, so a caption change is usually picked up automatically.
        return (
            ("cache_text_encoder_outputs", "启用 TE 缓存后，改动 caption 需要重建缓存才会生效：删除数据集目录里对应的 *_te.npz（kohya 默认会校验缓存内容，一般会自动重建）"),
            ("cache_text_encoder_outputs_to_disk", "TE 缓存到磁盘会跨任务复用，改动 caption 需要重建缓存才会生效"),
        )

    def export_parameters(self, config):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 Kohya 适配器支持")
        # A parameter profile must not reference a generated LP task directory.
        text = self.native_config_text(config, Path("dataset.toml"))
        text = "\n".join(line for line in text.splitlines() if not line.startswith("dataset_config =")) + "\n"
        output_name = config.get("output_name") or config.get("name") or "lora"
        text = text.replace('output_name = "lora"\n', "output_name = " + json.dumps(output_name, ensure_ascii=False) + "\n")
        notice = "此文件只包含训练参数；在 Kohya 中另行选择数据集或提供 dataset_config，再检查本机模型和输出路径。"
        if exported_prompt_files(config):
            notice += " 启用采样时会同时导出 sample_prompts.txt 配套文件；运行原生配置时请将它与参数文件放在同一工作目录。"
        return "# " + notice + "\n" + text, (notice,)

    def export_parameter_files(self, config):
        return exported_prompt_files(config)

    def native_config_text(self, config, dataset_config):
        params = config.get("params", {})
        architecture = config.get("architecture")
        values = {}
        for p in parameters_for(architecture):
            if not p.native:
                continue
            value = params.get(p.key, p.default)
            # For MOSS an explicitly empty layer list disables skipping in CFG.
            if value is None or (value == "" and p.key != "sample_uncond_skip_layers"):
                continue
            # The native parser supplies these defaults. Omitting them keeps
            # exported configs usable with checkouts predating the new switches.
            if p.key in RELEASE_KEYS and value == p.default:
                continue
            if p.key == "sample_prompts":
                value = prompt_file(config, Path(dataset_config).parent)
                if not value:
                    continue
            if p.key == "xformers":
                # Handled after the loop: kohya takes flags here, not a value, and a
                # truthy string would silently select xformers for every choice.
                continue
            if p.key == "clip_skip" and value == 1:
                # 1 is the engine's effective default; writing it also triggers the
                # engine's "v2 with clip_skip" warning on SD2 for no reason.
                continue
            if p.key == "max_token_length" and value == 75:
                # Omitting the key is exactly what 75 means to the engine.
                continue
            if p.list_arg:
                value = list_values(value)
                if not value:
                    continue
            values[p.key] = value
        # The plain sdpa switch is written by the loop above with its own default, so the
        # attention choice has to be applied after it to take effect.
        if params.get("xformers") == "xformers":
            values["xformers"] = True
        elif params.get("xformers") == "sdp":
            values["sdpa"] = True
        bound_output = config.get("output_layout") == "task_id"
        output_name = config.get("output_name") or config.get("name") or "lora"
        values.update(
            pretrained_model_name_or_path=config["base_model_path"],
            dataset_config=str(dataset_config),
            output_dir=config["output_dir"],
            output_name=safe_file_stem(output_name) if bound_output else output_name,
            network_module=NETWORK_MODULES.get(architecture, "networks.lora"),
        )
        if architecture == "sd2":
            values["v2"] = True
        # Supplementary arguments go into the same config file, so preview, export
        # and the real run all see them.
        values = extra_params.merge(values, str(params.get(extra_params.PARAM_KEY) or ""),
                                    bound_output=bound_output)
        if (values.get("use_8bit_adam") or values.get("use_lion_optimizer")) and values.get("optimizer_type") == "AdamW":
            values.pop("optimizer_type")
        for key in ("lr_warmup_steps", "lr_decay_steps"):
            number = values.get(key)
            if isinstance(number, float) and number >= 1 and number.is_integer():
                values[key] = int(number)
        for key in ("base_weights", "base_weights_multiplier", "text_encoder_lr"):
            if key in values and (key != "text_encoder_lr" or not isinstance(values[key], (int, float))):
                values[key] = native_list(key, values[key])
        # A custom argument can target a list argument too; normalise it so the
        # engine never receives a scalar it would iterate character by character.
        for p in PARAMS:
            if not p.list_arg:
                continue
            key = p.native_key or p.key
            if key in values and not isinstance(values[key], list):
                items = list_values(values[key])
                if items:
                    values[key] = items
                else:
                    values.pop(key, None)
        # build_launch only gets this file; record the architecture so SD1/SD2/SDXL
        # (which share networks.lora) still select their own entry script.
        lines = [f"# lp-architecture: {architecture}"] if architecture else []
        lines += [f"{key} = {toml_value(value)}" for key, value in values.items()]
        return "\n".join(lines) + "\n"

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        write_prompts(config, job_directory)
        params = config.get("params", {})
        dataset = job_directory / "dataset.toml"
        # One subset explicitly names the dataset path; no repeat-folder naming convention required.
        text = "[general]\n"
        text += "caption_extension = " + json.dumps(params.get("caption_extension") or ".caption") + "\n"
        text += f'enable_bucket = {str(bool(params.get("enable_bucket"))).lower()}\n\n'
        text += "[[datasets]]\n"
        if params.get("resolution") is not None:
            text += f'resolution = {params["resolution"]}\n'
        text += "[[datasets.subsets]]\n"
        text += f'num_repeats = {params.get("num_repeats") or 1}\n'
        text += "image_dir = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
        atomic_text(dataset, text)
        target = job_directory / "config.native.toml"
        atomic_text(target, self.native_config_text(config, dataset))
        return target

    def preview_argv(self, installation, config, native_config):
        root = scripts_root(installation.source_path)
        spec = next((a for a in ARCHITECTURES if a.id == config.get("architecture")), None)
        script = root / (spec.script if spec else "train_network.py")
        return self._training_argv(installation, script, native_config)

    @staticmethod
    def _training_argv(installation, script, native_config):
        argv = (str(installation.python_executable), "-u", str(script), "--config_file", str(native_config))
        if (script.parent / "library" / "run_control.py").is_file():
            argv += ("--run_control",)
        return argv

    def native_stop(self, task, argv):
        from adapters.sd_scripts.run_control import stop_command
        return stop_command(task, argv)

    def build_launch(self, installation, native_config):
        import tomllib
        text = native_config.read_text(encoding="utf-8")
        values = tomllib.loads(text)
        root = scripts_root(installation.source_path)
        script = MODULE_SCRIPTS.get(values.get("network_module"))
        if not script:
            marker = _ARCHITECTURE_MARKER.search(text)
            spec = next((a for a in ARCHITECTURES if marker and a.id == marker.group(1)), None)
            if spec:
                script = spec.script
            elif values.get("network_train_unet_only"):
                # Configs written before the architecture marker existed: SDXL was
                # the only architecture that ever forced this flag.
                script = "sdxl_train_network.py"
            else:
                script = "train_network.py"
        return LaunchSpec(self._training_argv(installation, root / script, native_config), root,
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

    def parse_log(self, line):
        # tqdm's progress is explicit; no epoch/ETA guesses and no fabricated loss points.
        return parse_progress(line, bar_kind="remaining_steps")

    def recovery_capabilities(self, installation, architecture=None):
        # --save_state writes an Accelerate state dir that --resume restores; an
        # interrupted run without a saved state keeps nothing to continue from.
        return RecoveryCapabilities(graceful_stop=True, resume_full_state=True)

    def resume_sources(self, job_directory):
        try:
            config = JobFiles(job_directory).read_job()
        except (OSError, ValueError, KeyError):
            return ()
        return accelerate_resume_sources(Path(config.get("output_dir") or ""))

    def resume_parameters(self, source: ResumeSource):
        return {"resume": str(source.path)}

    def collect_artifacts(self, job_directory):
        config = JobFiles(job_directory).read_job()
        root = Path(config["output_dir"])
        found = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Accelerate state dirs keep their own copy of the weights; they are
            # resume points, not deployable LoRA files.
            if any(part.endswith("-state") for part in path.relative_to(root).parts[:-1]):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                # Periodic saves are "{name}-step{step:08d}"; epoch saves are
                # "{name}-{epoch:06d}" and detail saves are "{name}".
                match = _ARTIFACT_STEP_RE.search(path.stem)
                found.append(Artifact(path, "checkpoint", True, int(match[1]) if match else None))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
