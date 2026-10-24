"""Conservative, single-GPU sd-scripts LoRA adapter. No engine imports here."""
import json
import re
from pathlib import Path
from adapters.base import (
    InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact,
    ResumeSource,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.recovery import accelerate_resume_sources
from adapters.sd_scripts.params import ARCHITECTURES, NETWORK_MODULES, PARAMS
from adapters import extra_params
from adapters.sampling import prompt_file, write_prompts, exported_prompt_files
from app.schemas.training import LaunchSpec
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
_TORCH_REQUIREMENT_RE = re.compile(r"^(?:torch|torchvision)\b", re.IGNORECASE)

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


def toml_value(value):
    """Render one Python value the way the engine's TOML config expects it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(json.dumps(str(item), ensure_ascii=False) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


class KohyaAdapter(EngineAdapter):
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
    adapter_version = "0.1.0"
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
            raise ValueError("缺少 Kohya requirements.txt 或其中没有可安装的固定版本条目")
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
        # setup_common.py imports pkg_resources, which setuptools 82 removed;
        # keep the pin so anything else importing it still works. bitsandbytes is
        # not in the requirements file, but LP defaults to AdamW8bit.
        packages = ["setuptools<82", "bitsandbytes", *torch_pins, *entries]
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
                                      "--no-build-isolation", *links, *extra_index, *packages),))
        fallback_flag = [] if getattr(self, "embedded_mode", False) else ["--require-virtualenv"]
        return InstallationPlan("Kohya 依赖安装（由 LP 执行，pip 降级路径）",
                                ((python, "-u", "-m", "pip", "install", *fallback_flag,
                                  "--no-input", "--no-build-isolation", *links, *extra_index, *packages),))

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
    _LOCAL_REQUIREMENT_RE = re.compile(r"^(?:-e\s+)?(\.{1,2}[/\\][^\s#]+)$")

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
        """Pinned requirement lines LP installs, in order.

        Only ``name==version`` lines are installed: an unpinned line (``numpy``,
        ``bitsandbytes``, ``tensorboard; platform_machine != "ARM64"``) would drift on
        every reinstall and is reported by ``requirement_skips`` instead of being
        installed behind the user's back.
        """
        entries = []
        for name in self.requirement_files(source_path):
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw in text.splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if self._LOCAL_REQUIREMENT_RE.match(line):
                    continue
                if "==" not in line.split(";", 1)[0]:
                    continue
                entries.append(line)
        return tuple(dict.fromkeys(entries))

    def requirement_skips(self, source_path):
        """Requirement lines that are not pinned, so ``requirement_entries`` skips them."""
        skipped = []
        for name in self.requirement_files(source_path):
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw in text.splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if self._LOCAL_REQUIREMENT_RE.match(line):
                    continue
                if "==" in line.split(";", 1)[0]:
                    continue
                skipped.append(f"{name}: {line}")
        return tuple(dict.fromkeys(skipped))

    def source_issues(self, source_path):
        root = scripts_root(source_path)
        issues = []
        if not ((root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()):
            issues.append(
                "Kohya 缺少完整 sd-scripts 训练源码（train_network.py + library/train_util.py）；"
                "请把完整的 sd-scripts 仓库复制到该目录，或使用“Git 拉取引擎”重新拉取。"
                "若这是 kohya_ss 包装仓库，可点击“补齐训练源码”初始化其 sd-scripts 子模块"
            )
        # Unpinned lines are deliberately not listed as issues: bitsandbytes is added
        # explicitly by installation_plan, and the rest arrive as dependencies of the
        # pinned entries. Reporting them as "skipped" was misleading, not actionable.
        return tuple(issues)

    source_repair_label = "补齐 sd-scripts 训练源码"

    clone_url = "https://github.com/kohya-ss/sd-scripts.git"
    # The upstream version this adapter is written against.
    clone_branch = "main"
    clone_commit = "690ea7f96c23182352ec63def76d431c6120bd2f"

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
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable)

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
        if config.get("architecture") not in {a.id for a in self.architectures(installation.source_path)}:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        issues += self.generic_param_issues(values, config.get("architecture"))
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
            if not p.native and p.unsupported_reason:
                continue
            value = values.get(p.key, p.default)
            message = check_value(p, value)
            # Anima gets the engine-specific max_workers explanation below; do not
            # also report the generic range error for the same value.
            if message and not (p.key == "vae_batch_size" and architecture == "anima" and value == 0):
                issues.append(ValidationIssue(p.key, message))
        resolution = values.get("resolution", 512)
        if isinstance(resolution, int) and resolution % 64:
            issues.append(ValidationIssue("resolution", "分辨率必须是 64 的倍数"))
        return issues

    # ---- 同引擎参数校验：本引擎的所有底模 ----
    def engine_param_issues(self, installation, config, values):
        issues = []
        for severity, message in extra_params.problems(
            str(values.get(extra_params.PARAM_KEY) or ""),
            extra_params.provided_keys(
                PARAMS, config.get("architecture"), values, skip_false=False,
                extra=("pretrained_model_name_or_path", "dataset_config", "output_dir", "output_name",
                       "network_module", "config_file", "v2", "network_train_unet_only"),
            ),
        ):
            issues.append(ValidationIssue(extra_params.PARAM_KEY, message, severity))
        # Optional packages this engine's own defaults and choices need.
        if str(values.get("optimizer_type") or "").endswith("8bit") and self.site_package_installed(installation, "bitsandbytes") is False:
            issues.append(ValidationIssue(
                "optimizer_type", "该优化器需要 bitsandbytes，但引擎环境里没有：请改用 AdamW，或重新安装引擎环境（LP 现在会一并装上 bitsandbytes）"))
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
        # HunyuanImage only supports fp8 through --fp8_scaled; fp8_base is ignored
        # with a warning (hunyuan_image_train_network.py:332-338).
        if config.get("architecture") == "hunyuan_image" and values.get("fp8_base"):
            issues.append(ValidationIssue(
                "fp8_base", "HunyuanImage 不支持 fp8_base（引擎会忽略并告警）；若需要 FP8 请改用 --fp8_scaled", "warning"))
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
        # Unsloth offloading is Anima-only; anima_train_network.py asserts both
        # combinations away, so report them here instead of in the engine.
        if architecture == "anima" and values.get("unsloth_offload_checkpointing"):
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
                          "hunyuan_image": 32, "lumina": 16, "anima": 16}.get(architecture)
        steps = values.get("bucket_reso_steps", 64)
        if required_steps and isinstance(steps, int) and steps % required_steps:
            issues.append(ValidationIssue(
                "bucket_reso_steps",
                f"该底模要求分桶步长能被 {required_steps} 整除，否则引擎直接断言；当前为 {steps}"))
        # cache_latents defaults to True, so the check must use that default: a user
        # enabling colour augmentation without touching the cache field still gets
        # cache_latents=true in the rendered config and the engine assert.
        if values.get("cache_latents", True) and (values.get("color_aug") or values.get("random_crop")):
            issues.append(ValidationIssue(
                "cache_latents", "缓存 Latents 与颜色增强/随机裁剪不能同时启用，引擎会直接断言；请关闭其一"))
        if values.get("cache_text_encoder_outputs") and (
                values.get("shuffle_caption") or (values.get("caption_dropout_rate") or 0) > 0):
            issues.append(ValidationIssue(
                "cache_text_encoder_outputs",
                "缓存 TE 输出与「打乱 Caption」或「Caption Dropout」不能同时启用，引擎会直接断言"))
        if values.get("scale_v_pred_loss_like_noise_pred"):
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
        required_components = {
            "flux1": (("ae", "FLUX 需要 VAE/AE 路径"),),
            "lumina": (("ae", "Lumina 需要 VAE/AE 路径"), ("gemma2", "Lumina 需要 Gemma2 编码器路径")),
            "hunyuan_image": (("byt5", "HunyuanImage 需要 ByT5 编码器路径"),),
        }
        for key, message in required_components.get(architecture, ()):
            if not str(values.get(key) or "").strip():
                issues.append(ValidationIssue(key, message))
        return issues

    def site_package_installed(self, installation, name):
        """Whether the engine environment has a package, without importing it.

        ``None`` means "cannot tell", which must never be reported as missing.
        """
        try:
            exe = Path(installation.python_executable).resolve()
        except (AttributeError, TypeError, OSError):
            return None
        site = exe.parent.parent / "Lib" / "site-packages"
        if not site.is_dir():
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
        for p in PARAMS:
            if not p.native or not p.applies_to(architecture):
                continue
            value = params.get(p.key, p.default)
            if value is None or (isinstance(value, str) and value == ""):
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
        values.update(
            pretrained_model_name_or_path=config["base_model_path"],
            dataset_config=str(dataset_config),
            output_dir=config["output_dir"],
            output_name=config.get("output_name") or config.get("name") or "lora",
            network_module=NETWORK_MODULES.get(architecture, "networks.lora"),
        )
        if architecture == "sd2":
            values["v2"] = True
        # Supplementary arguments go into the same config file, so preview, export
        # and the real run all see them.
        values = extra_params.merge(values, str(params.get(extra_params.PARAM_KEY) or ""))
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
        text += "caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += f'enable_bucket = {str(bool(params.get("enable_bucket", True))).lower()}\n\n'
        text += "[[datasets]]\n"
        text += f'resolution = {params.get("resolution") or 512}\n'
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
        return (str(installation.python_executable), "-u", str(script), "--config_file", str(native_config))

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
        return LaunchSpec((str(installation.python_executable), "-u", str(root / script), "--config_file", str(native_config)), root,
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

    def parse_log(self, line):
        # tqdm's progress is explicit; no epoch/ETA guesses and no fabricated loss points.
        return parse_progress(line)

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
