"""anima_lora adapter: Anima DiT LoRA training plus the MOSS-Image line.

Two entries share one uv-locked clone:

* ``train.py`` — Anima. Reads its hyperparameters from a flat TOML
  (``--config_file``) and its dataset from a kohya-style ``--dataset_config``.
* ``scripts/moss/train.py`` — MOSS-Image. Trains from a **pre-cached folder**, so a
  job runs ``scripts/moss/cache.py`` first (same two-stage shape as musubi-tuner).

The engine installs through ``uv sync`` against its committed lock file; its
CUDA/ROCm/torch selection is expressed as dependency groups, never as pip extras.
No engine imports here.
"""

import json
import math
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

from adapters import extra_params
from adapters.base import (
    Artifact,
    DetectionResult,
    EngineAdapter,
    InstallationPlan,
    RecoveryCapabilities,
    ResumeSource,
    ValidationIssue,
)
from adapters.capabilities import (
    NUMBER, check_value, epoch_counts, parse_eta, parse_progress, plain_log, read_number, specs, training_rate,
)
from adapters.recovery import accelerate_resume_sources
from adapters.sampling import PROMPT_FILENAME, exported_prompt_files, prompt_file, write_prompts
from adapters.anima_lora.params import (
    ARCHITECTURES, PARAMS, VERSIONED_KEYS, default_values, engine_chain_extras, installation_params,
)
from adapters.anima_lora.upstream_options import MOSS_CACHE_OPTIONS, MOSS_TRAIN_OPTIONS
from app.services.common import atomic_text, safe_file_stem
from app.services.jobs import JobFiles
from app.schemas.training import LaunchSpec, TrainingEvent

# Assets the Anima entry cannot run without: the DiT weights, its Qwen3 text
# encoder, and the VAE (the T5 tokenizer is bundled by the engine).
ANIMA_REQUIRED = ("pretrained_model_name_or_path", "qwen3", "vae")

# The engine's library/moss/loader.py reads these folders from
# ``<model_dir>/MOSS-Image-Base/``; a directory missing one cannot train.
MOSS_MODEL_SUBDIRS = ("transformer", "text_encoder", "vae", "tokenizer")

# Characters a real Windows path cannot contain. A render target containing one is a
# display placeholder (``"<task-id>"``) from the preview path, never a writable
# directory: writing there raises OSError before any config is shown.
_PLACEHOLDER_CHARS = frozenset('<>:"|?*')

# Characters a checkpoint file stem cannot contain. The task name is user text; the
# engine appends ``-step…``/``-state`` to this stem and writes it under output_dir.
_UNSAFE_FILENAME = re.compile(r'[<>:/\\|?*\x00-\x1f]')


def _custom_values(text, bound_output=False):
    aliases = {
        "cache_latents": "use_vae_cache", "cache_latents_to_disk": "use_vae_cache",
        "cache_text_encoder_outputs": "use_text_cache", "cache_text_encoder_outputs_to_disk": "use_text_cache",
    }
    normalized = {}
    for key, value in extra_params.merge({}, text, bound_output=bound_output).items():
        key = aliases.get(key, key)
        if key in normalized and normalized[key] != value:
            raise ValueError(f"自定义参数对同一缓存开关 --{key} 指定了不同值")
        normalized[key] = value
    return normalized


def _renderable_directory(path) -> bool:
    """False when ``path`` is a placeholder rather than a directory we can write to."""
    for index, part in enumerate(Path(path).parts):
        # The drive root ("C:\\") legitimately contains ':' and is not a name.
        if index == 0 and (part in ("/", "\\") or re.fullmatch(r"[A-Za-z]:[\\/]", part)):
            continue
        if set(part) & _PLACEHOLDER_CHARS:
            return False
    return True


def _requests_compile(values: dict, native_text: str = "") -> bool:
    """True when the job asks for torch.compile on either training line.

    The flag lives under different names per line (``moss_compile`` for MOSS,
    ``torch_compile`` for Anima) and may only exist in the rendered config file when
    ``build_launch`` runs before ``job.json`` is written. MOSS is rendered as command
    lines (one per stage), so the switch is matched as a whole token — never as the
    ``--compile`` substring of ``--no-compile``.
    """
    if values.get("moss_compile") or values.get("torch_compile"):
        return True
    for line in native_text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if {"--compile", "--torch_compile"} & set(stripped.split()):
            return True
        key, separator, raw = stripped.partition("=")
        if separator and key.strip() in {"compile", "torch_compile"} and raw.strip().lower() == "true":
            return True
    return False



def _msvc_available() -> bool:
    """True when a C++ toolchain is installed, whether or not it is on PATH.

    ``cl.exe`` only appears on PATH inside a Developer prompt, so a PATH lookup alone
    reports "missing" on machines that have Visual Studio installed. ``vswhere`` ships
    with every VS installer and answers for Build Tools too.
    """
    if shutil.which("cl") or shutil.which("cl.exe"):
        return True
    if os.name != "nt":
        return False
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles") or ""
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(vswhere), "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(completed.stdout.strip())


def _msvc_environment() -> dict[str, str]:
    """Environment from vcvars64.bat, or an empty dict when unavailable.

    torch.compile on Windows shells out to cl.exe and needs INCLUDE/LIB/PATH prepared
    the way a Developer prompt prepares them; installing Visual Studio alone is not
    enough. The supervisor passes these overrides only to the training process.
    """
    if os.name != "nt":
        return {}
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles") or ""
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return {}
    try:
        located = subprocess.run(
            [str(vswhere), "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    lines = [line.strip() for line in located.stdout.splitlines() if line.strip()]
    for install in reversed(lines):
        vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars.is_file():
            continue
        try:
            prepared = subprocess.run(
                f'"{vcvars}" >nul && set',
                shell=True,
                capture_output=True, text=True, timeout=120,
                # cmd emits the OEM code page (GBK on Chinese Windows), not UTF-8.
                encoding="mbcs", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        out = {}
        for line in prepared.stdout.splitlines():
            key, sep, value = line.partition("=")
            if not sep or not key or key.startswith("=") or key in {"PROMPT", "VSCMD_ARG_TGT_ARCH"}:
                continue
            out["PATH" if key.upper() == "PATH" else key] = value
        return out
    return {}


class AnimaLoraAdapter(EngineAdapter):
    model_input_keys = ("vae", "qwen3", "network_weights", "moss_model_dir", "pooled_text_proj",
                        "moss_init_lora", "moss_pairs")

    def custom_write_dirs(self, config):
        if config.get("architecture") == "moss_image":
            return self._configured_paths(config, ("moss_cache_directory",))
        return self._configured_paths(config, ("cache_directory", "logging_dir"))

    def legacy_output_issue(self, config):
        # The MOSS argv uses the sanitized output stem and has no extra-config
        # merge. The Anima TOML path does merge supplementary arguments.
        if config.get("architecture") == "moss_image":
            return None
        return extra_params.legacy_output_issue(config, self._output_name(config))

    engine_id = "anima_lora"
    display_name = "Anima LoRA"
    clone_url = "https://github.com/sorryhyun/anima_lora.git"
    # Anima needs a base-model file; its MOSS line trains from a model directory.
    base_model_optional_architectures = ("moss_image",)
    adapter_version = "0.3.0"
    installation_source_help = "根据当前项目声明的依赖组安装相应的 PyTorch 和加速组件。"
    installation_help = (
        "使用 uv 将项目锁定的依赖安装到引擎解释器（Python 3.13 + torch 2.14 + CUDA 13.2）。"
        "默认组在 Windows 已包含 CUDA 栈；AMD 需切换为 rocm-windows 组，"
        "旧 torch 需改用 torch-213 / torch-212。"
    )
    default_installation_source = "existing"
    environment_strategy = "embedded"
    environment_dir = ".lp/python"
    environment_candidates = (".venv",)
    python_version = "3.13"
    # The clone binds torch/torchvision to named uv indexes per dependency group
    # ([tool.uv.sources] + [[tool.uv.index]]), so a mirror is injected as a named
    # index override exactly like musubi-tuner's.
    torch_index_mode = "named"
    diagnostic_modules = ("accelerate", "transformers", "safetensors", "diffusers", "toml")
    native_format = "toml"
    submittable = True
    # The dataset config holds no resolution key: the engine resizes to its own
    # tier/band model, and nested subsets are supported through ``recursive``.
    supports_nested_images = True
    # checkpoints.py runs os.makedirs(..., exist_ok=True) on the first save.
    creates_output_dir = True
    training_notice = "Anima 训练需要底模、Qwen3 文本编码器与 VAE；首次运行请先在引擎内准备好模型资产。"

    # ---- detection / install ----

    def detect(self, source_path):
        source = Path(source_path)
        entry = source / "train.py"
        try:
            project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return DetectionResult(self.engine_id, False, "缺少或无法解析 pyproject.toml")
        # A kohya sd-scripts checkout also ships a pyproject.toml and network modules;
        # the package name plus the root train.py is what separates this engine from it.
        matched = (
            project.get("name") == "anima-lora"
            and entry.is_file()
            and (source / "library").is_dir()
        )
        return DetectionResult(self.engine_id, matched, "anima-lora 项目元数据、library 与 train.py")

    @staticmethod
    def _cuda_tag(source_path):
        """CUDA tag of the clone's own PyTorch index name (``pytorch-cu132`` -> cu132)."""
        try:
            metadata = tomllib.loads((Path(source_path) / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return None
        for entry in metadata.get("tool", {}).get("uv", {}).get("index", []):
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name.startswith("pytorch-"):
                tag = name[len("pytorch-"):]
                if re.fullmatch(r"cu\d{3}", tag):
                    return tag
        return None

    def installation_sources(self, source_path):
        """Install sources this clone offers, in the kohya/musubi shape (``cuNNN``).

        The clone selects its stack through dependency groups, but the CUDA tag it
        installs from comes from its own named wheel index. Reporting the group names
        instead would make the driver gate drop every source as unparsable and block
        an install that is in fact compatible. ``rocm-windows`` is kept selectable
        for AMD, whose rank is never derived from a CUDA tag.
        """
        sources = []
        tag = self._cuda_tag(source_path)
        if tag:
            sources.append(tag)
        if "rocm-windows" in self._declared_groups(source_path):
            sources.append("rocm-windows")
        return tuple(sources)

    def _declared_groups(self, source_path):
        try:
            metadata = tomllib.loads((Path(source_path) / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return set()
        groups = metadata.get("dependency-groups", {})
        declared = set(groups if isinstance(groups, dict) else {})
        declared |= set(metadata.get("project", {}).get("optional-dependencies", {}))
        return declared

    def torch_index_tag(self, source_path):
        """The wheel index tag the clone itself installs from."""
        return self._cuda_tag(source_path)

    @staticmethod
    def _torch_index_for_group(source_path, group):
        """Return the named uv index this clone binds torch to for ``group``.

        The upstream clone owns this mapping, so reading it keeps LP working when
        the engine moves to another CUDA index name.
        """
        try:
            metadata = tomllib.loads((Path(source_path) / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return None
        entries = metadata.get("tool", {}).get("uv", {}).get("sources", {}).get("torch", [])
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, dict) and entry.get("group") == group:
                index = entry.get("index")
                if isinstance(index, str) and index.strip():
                    return index.strip()
        return None

    def installation_plan(self, source_path, python, torch_source, uv=None, torch_mirror_url=None):
        source = Path(source_path)
        if not self.detect(source).matched:
            raise ValueError("Anima LoRA 源码不完整")
        if not uv:
            raise ValueError("Anima LoRA 的官方安装流程需要 uv")

        # Embedded environments have no venv to sync into. The clone's dependency
        # groups still own the versions, so they are exported from its own metadata
        # and installed into the copied interpreter instead of being re-guessed.
        if getattr(self, "embedded_mode", False):
            tag = self._cuda_tag(source)
            if torch_source == "existing":
                groups: list[str] = []
            elif torch_source == tag:
                groups = ["cuda-windows" if os.name == "nt" else "torch-214"]
            else:
                groups = [torch_source]
            for group in groups:
                if group not in self._declared_groups(source):
                    raise ValueError(
                        f"当前 Anima LoRA 未声明 {group} 依赖组；可用："
                        f"{', '.join(sorted(self._declared_groups(source))) or '无'}"
                    )
            helper = Path(__file__).resolve().parents[2] / "scripts" / "install_uv_project.py"
            command = [python, "-u", str(helper), "--uv", str(uv),
                       "--project", str(source.resolve()), "--python", str(python)]
            if groups:
                command.append("--no-default-groups")
                for group in groups:
                    command.extend(["--group", group])
            if torch_mirror_url:
                command.extend(["--torch-index", torch_mirror_url.rstrip("/")])
            return InstallationPlan(
                "Anima LoRA 依赖安装（embedded：按引擎依赖组导出后装入解释器）",
                (tuple(command),),
            )

        command = [uv, "sync"]
        if torch_source == "existing":
            # The default groups already carry the CUDA stack on Windows, so a plain
            # sync is the CUDA path. A torch mirror cannot be aimed at a group that
            # was not selected, so it is refused instead of silently ignored.
            if torch_mirror_url:
                raise ValueError("未选择 torch 依赖组（沿用现有环境）时不能指定 PyTorch/CUDA wheel 镜像源")
            return InstallationPlan("Anima LoRA 官方 uv 项目安装流程（默认组）", (tuple(command),))
        # The CUDA tag maps to the clone's own group for this platform: the Windows
        # stack lives in cuda-windows, the Linux one rides the torch-2xx default.
        if torch_source == self._cuda_tag(source):
            group = "cuda-windows" if os.name == "nt" else "torch-214"
        else:
            group = torch_source
        if group not in self._declared_groups(source):
            raise ValueError(
                f"当前 Anima LoRA 未声明 {group} 依赖组；可用：{', '.join(sorted(self._declared_groups(source))) or '无'}"
            )
        if group == "rocm-windows":
            # The two backend groups are declared mutually exclusive upstream.
            command.extend(("--no-group", "cuda-windows", "--group", "rocm-windows"))
        else:
            # Selecting an explicit stack replaces the default groups entirely.
            command.extend(("--no-default-groups", "--group", group))
        if torch_mirror_url:
            index_name = self._torch_index_for_group(source, group)
            if not index_name:
                raise ValueError(
                    f"Anima LoRA 未为 {group} 声明可覆盖的 PyTorch uv 索引，"
                    "无法为该依赖组指定自定义 PyTorch/CUDA wheel 镜像源"
                )
            # uv's --index NAME=URL overrides the named index without editing the
            # copied project's pyproject.toml or uv.lock.
            command.extend(("--index", f"{index_name}={torch_mirror_url}"))
        return InstallationPlan(f"Anima LoRA 官方 uv 项目安装流程（{group}）", (tuple(command),))

    def source_issues(self, source_path):
        source = Path(source_path)
        issues = []
        if not (source / "uv.lock").is_file():
            issues.append("缺少 uv.lock：官方安装流程需要一个可复现的锁文件")
        if not (source / "train.py").is_file():
            issues.append("缺少训练入口 train.py")
        issues.extend(self._anime_tools_issues(source))
        return tuple(issues)

    @staticmethod
    def _anime_tools_issues(source: Path):
        """Check the checkout's anime-tools requirement before building MOSS caches."""
        site = source / ".lp" / "python" / "Lib" / "site-packages"
        if not site.is_dir():
            return ()
        if any((source / name).is_dir() for name in (".venv", "venv", "env")):
            return ()
        found = None
        for info in site.glob("anime_tools-*.dist-info"):
            name, _, version = info.stem.partition("-")
            if name.casefold() == "anime_tools" and version:
                found = version
                break
        try:
            metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
            pinned = metadata.get("tool", {}).get("uv", {}).get("sources", {}).get("anime-tools", {})
            tag = pinned.get("tag", "") if isinstance(pinned, dict) else ""
        except (OSError, ValueError):
            tag = ""
        required = tag.removeprefix("v") if re.fullmatch(r"v?\d+\.\d+\.\d+", tag) else "0.8.1"
        if found is None:
            return (f"缺少 anime-tools；此引擎的 MOSS 缓存需要 anime-tools>={required}",)
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", found)
        if not match or tuple(map(int, match.groups())) < tuple(map(int, required.split("."))):
            return (f"anime-tools={found} 过旧；此引擎的 MOSS 缓存需要 >={required}，请更新引擎环境",)
        return ()

    def install_blockers(self, hardware):
        # torch.compile + Triton want the CUDA 13.2 toolkit; the default path is
        # also much slower without Flash Attention, so say it before installing.
        return ()

    # ---- architectures / parameters ----

    def architectures(self, source_path):
        source = Path(source_path)
        return tuple(a for a in ARCHITECTURES if (source / a.script).is_file())

    def capabilities(self, installation):
        available = tuple(a.id for a in self.architectures(installation.source_path))
        # Parameter keys each architecture must have before launch. ``required_for``
        # on a Param is a list of *architecture ids*, so this table is inverted and
        # aggregated by key: a shared parameter lists every architecture that needs
        # it, and a parameter with a declared requirement keeps it when listed here.
        required = {
            "anima": ANIMA_REQUIRED,
            # MOSS-Image has no generic base-model file: its model directory is the
            # only model input, so the form must mark it as required.
            "moss_image": ("moss_model_dir",),
        }
        required_for: dict[str, list[str]] = {}
        for architecture in available:
            for key in required.get(architecture, ()):
                required_for.setdefault(key, []).append(architecture)
        # Only parameters that apply to an architecture this checkout actually has:
        # a spec must never point at an architecture the capabilities do not declare.
        params = tuple(
            replace(p, required_for=tuple(required_for.get(p.key, p.required_for)))
            for p in installation_params(installation.source_path)
            if any(p.applies_to(architecture) for architecture in available)
        )
        return dict(
            installation_id=installation.installation_id,
            architectures=list(available),
            params=specs(params),
            submittable=self.submittable,
            # MOSS-Image trains from its own model directory, so the generic
            # 底模文件 field must stay hidden: two model inputs would disagree.
            base_model_optional_architectures=[
                architecture for architecture in self.base_model_optional_architectures
                if architecture in available
            ],
        )

    def _param(self, key):
        return next((p for p in PARAMS if p.key == key), None)

    def _defaults(self, installation):
        return default_values(installation.source_path)

    def validate(self, installation, config):
        issues = []
        parameters = installation_params(installation.source_path)
        defaults = {p.key: p.default for p in parameters}
        architecture = config.get("architecture")
        available = {a.id for a in self.architectures(installation.source_path)}
        if architecture not in available:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for key in (values.keys() & VERSIONED_KEYS) - defaults.keys():
            issues.append(ValidationIssue(key, "此版本引擎不支持该参数，请更新引擎或移除该项"))
        applicable = tuple(p for p in parameters if p.applies_to(architecture))
        for p in parameters:
            if not p.applies_to(architecture) and values.get(p.key) not in (None, "", p.default):
                issues.append(ValidationIssue(p.key, "此参数不适用于所选底模类型"))
        for p in applicable:
            if not p.native and p.unsupported_reason:
                continue
            message = check_value(p, values.get(p.key, defaults.get(p.key, p.default)))
            if message:
                issues.append(ValidationIssue(p.key, message))
        custom = {}
        try:
            custom = _custom_values(str(values.get(extra_params.PARAM_KEY) or ""))
        except ValueError as exc:
            issues.append(ValidationIssue(extra_params.PARAM_KEY, str(exc)))
        check_native_overrides = False
        if architecture == "anima":
            by_native = {(p.native_key or p.key): p for p in applicable if p.native}
            unknown = {k: v for k, v in custom.items() if k not in by_native}
            if unknown:
                from adapters.anima_lora.validation import validate_extra_keys
                custom_issues = validate_extra_keys(installation, unknown)
                issues.extend(custom_issues)
                check_native_overrides = not custom_issues
            if custom.get("artist_filter") and config.get("output_layout") == "task_id":
                issues.append(ValidationIssue("extra_params", "--artist_filter 会改变引擎的输出目录，请使用数据集路径过滤保留该任务的输出位置"))
            for key in custom.keys() & by_native.keys():
                p = by_native[key]
                message = check_value(p, custom[key])
                if message or p.list_arg:
                    issues.append(ValidationIssue("extra_params", f"--{key}：{message or '请在表单中填写列表参数'}"))
        elif custom:
            issues.append(ValidationIssue("extra_params", "MOSS 的缓存和训练使用独立参数，请使用对应阶段的表单字段"))
        extras = engine_chain_extras(installation.source_path) if architecture == "anima" else {}
        values = {**extras, **defaults, **values, **custom}
        for key in ANIMA_REQUIRED if architecture == "anima" else ():
            value = values.get(key, defaults.get(key, self._param(key).default if self._param(key) else None))
            # The job's base-model picker is what the config file actually writes, so
            # a selected base model satisfies this field even when the form left the
            # parameter empty (and vice versa, the field alone is not enough).
            if key == "pretrained_model_name_or_path" and key not in custom and str(config.get("base_model_path") or "").strip():
                value = config["base_model_path"]
            if value is None or (isinstance(value, str) and not value.strip()):
                label = self._param(key).label if self._param(key) else key
                issues.append(ValidationIssue(key, f"该底模类型需要填写：{label}"))
            elif isinstance(value, str) and not (Path(installation.source_path) / value).is_file():
                issues.append(ValidationIssue(key, f"找不到选择的模型文件：{value}"))
        if architecture == "anima" and values.get("torch_compile") and values.get("dynamo_backend") != "eager" and os.name == "nt" and not _msvc_available():
            # Same MSVC requirement as MOSS: torch.compile shells out to cl.exe on
            # Windows. build_launch injects the vcvars environment, but a machine
            # without the C++ workload can never satisfy it — say so before queuing.
            issues.append(ValidationIssue(
                "torch_compile",
                "该机器找不到 C++ 编译器（cl.exe）：开启 torch.compile 会在第一次 forward 时以 "
                "InvalidCxxCompiler 失败。请关闭该项，或安装 Visual Studio 的「使用 C++ 的桌面开发」",
                "error",
            ))
        if architecture == "anima":
            from adapters.anima_lora.validation import anima_issues
            issues.extend(anima_issues(installation, values, check_native_overrides))
        if architecture == "moss_image":
            from adapters.anima_lora.validation import moss_issues
            issues.extend(moss_issues(installation, config, values))
            distill = values.get("moss_distill") or "off"
            rows = values.get("moss_delta_rows") or "all"
            ref_adapters = values.get("moss_ref_adapters") or "off"
            if distill != "off":
                if rows != "all" or values.get("moss_trigger_registers") or ref_adapters == "all":
                    issues.append(ValidationIssue("moss_distill", "蒸馏要求 delta_rows=all、关闭触发寄存器及 ref_adapters=all 模式"))
                if values.get("moss_distill_extra_sigma") and values.get("moss_repa_weight"):
                    issues.append(ValidationIssue("moss_distill_extra_sigma", "额外蒸馏 σ 与 REPA 不能同时启用"))
                if values.get("moss_distill_balance") and values.get("moss_distill_add"):
                    issues.append(ValidationIssue("moss_distill_balance", "附加蒸馏损失自行平衡，请关闭梯度平衡或将附加权重设为 0"))
                if values.get("moss_distill_gate") and not values.get("moss_distill_add") and (
                    values.get("moss_distill_mix") or values.get("moss_distill_sigma_ramp")
                ):
                    issues.append(ValidationIssue("moss_distill_gate", "替换模式的蒸馏门控要求混合比例与 σ 渐变区间均为 0"))
            if values.get("moss_algo") == "loha" and (values.get("moss_int8") or defaults.get("moss_int8", "off")) != "off":
                issues.append(ValidationIssue("moss_int8", "LoHa 需要 bf16 底模，请将 DiT INT8 模式设为 off"))
            if values.get("moss_sample_at_first") or values.get("moss_sample_every_epochs"):
                if not str(values.get("moss_sample_prompts") or "").strip():
                    issues.append(ValidationIssue("moss_sample_prompts", "开启训练采样时需要填写提示词"))
                if values.get("moss_pairs") or values.get("moss_trigger_registers") or rows != "all":
                    issues.append(ValidationIssue("moss_sample_prompts", "训练中采样不支持 i2i 配对、触发寄存器或分行更新"))
                for key in ("moss_sample_width", "moss_sample_height"):
                    dimension = values.get(key)
                    if isinstance(dimension, int) and dimension % 16:
                        issues.append(ValidationIssue(key, "采样尺寸必须是 16 的整倍数"))
            if values.get("moss_mod_rank") and values.get("moss_algo", "lora") != "lora":
                issues.append(ValidationIssue("moss_mod_rank", "调制层 Rank 只适用于 LoRA"))
            if (values.get("moss_delta_rows") == "trigger" or values.get("moss_trigger_registers")) and not str(values.get("moss_trigger") or "").strip():
                issues.append(ValidationIssue("moss_trigger", "触发行更新及触发寄存器需要指定触发词"))
            if (values.get("moss_distill_balance") or values.get("moss_distill_gate")) and distill == "off":
                issues.append(ValidationIssue("moss_distill", "平衡蒸馏梯度及蒸馏门控需要选择教师模型"))
            # The selected folder is the MOSS-Image-Base diffusers directory itself
            # (its parent becomes --model_dir). Validate its shape here, so a wrong
            # pick fails at submit time instead of inside cache.py / train.py.
            chosen = values.get("moss_model_dir")
            chosen = chosen.strip() if isinstance(chosen, str) else ""
            if not chosen:
                issues.append(ValidationIssue(
                    "moss_model_dir", "该底模类型需要选择 MOSS-Image-Base 模型目录"))
            else:
                model_dir = Path(chosen)
                if not model_dir.exists():
                    issues.append(ValidationIssue("moss_model_dir", f"模型目录不存在：{chosen}"))
                elif not model_dir.is_dir():
                    issues.append(ValidationIssue("moss_model_dir", f"模型路径不是目录：{chosen}"))
                else:
                    if model_dir.name.casefold() not in {"moss-image-base", "moss-image-sft"}:
                        issues.append(ValidationIssue("moss_model_dir", "请选择 MOSS-Image-Base 或 MOSS-Image-SFT 模型目录"))
                    for name in MOSS_MODEL_SUBDIRS:
                        assets = model_dir.parent / "MOSS-Image-Base" if name != "transformer" else model_dir
                        if not (assets / name).is_dir():
                            issues.append(ValidationIssue(
                                "moss_model_dir", f"模型目录缺少 {assets.name}/{name}/ 子目录"))
            # torch.compile needs MSVC (cl.exe) on Windows; without it the first
            # forward aborts with InvalidCxxCompiler, which is expensive to diagnose
            # from a stack trace. Catch it before the job is queued. A normal Visual
            # Studio install is not on PATH, so vswhere is consulted as well.
            if values.get("moss_compile") and os.name == "nt" and not _msvc_available():
                issues.append(ValidationIssue(
                    "moss_compile",
                    "该机器找不到 C++ 编译器（cl.exe）：开启 torch.compile 会在第一次 forward 时以 "
                    "InvalidCxxCompiler 失败。请关闭该项，或安装 Visual Studio 的「使用 C++ 的桌面开发」",
                    "error",
                ))
            # Current MOSS supports compile and block swap together: the block
            # swapper keeps the frozen base weights page-locked and the LoRA
            # modules are compiled separately. Do not reject the engine's own
            # official LoRA recipe (compile=True, blocks_to_swap=20).
            caption_format = str(values.get("moss_caption_format") or "")
            try:
                plain_caption_rate = float(values.get("moss_plain_caption_rate") or 0.0)
            except (TypeError, ValueError):
                plain_caption_rate = 0.0
            if caption_format != "layout_json" and plain_caption_rate > 0:
                issues.append(ValidationIssue(
                    "moss_plain_caption_rate",
                    "text 格式的缓存没有 plain Anima caption，plain_caption_rate 不会生效；"
                    "请把它设为 0，或把字幕格式改为 layout_json 后重建缓存",
                    "warning",
                ))
            if caption_format == "layout_json" and not self._tag_csv_present(installation):
                issues.append(ValidationIssue(
                    "moss_caption_format",
                    "layout_json 需要 Danbooru tag CSV（danbooru_tags_classified.csv）。"
                    "请在引擎环境执行：python -m anime_tools.downloads danbooru_tags",
                    "error",
                ))
            # int8/fp8 quantized text encoders are fetched from HuggingFace on first
            # use. A complete local bf16 text_encoder can avoid that download when
            # the network is unavailable; warn before the cache stage hangs on HF.
            te_precision = str(values.get("moss_cache_text_encoder") or "int8")
            if te_precision in {"int8", "fp8"}:
                cached = Path(installation.source_path) / "models" / "moss_image" / te_precision
                local_dir = values.get("moss_model_dir")
                local_te = Path(str(local_dir)).expanduser() / "text_encoder" if local_dir else None
                if not (cached / "config.json").is_file() and local_te and local_te.is_dir():
                    issues.append(ValidationIssue(
                        "moss_cache_text_encoder",
                        f"当前 TE 精度为 {te_precision}，本机还没有对应的量化文件，缓存阶段会从 "
                        "HuggingFace 下载（约 9.2 GB）。如果网络不可达，可改用 bf16，直接读取本地 "
                        "MOSS-Image-Base/text_encoder，无需下载；修改后请清空 moss-cache 重新生成。",
                        "warning",
                    ))
            if values.get("moss_grad_checkpointing"):
                issues.append(ValidationIssue(
                    "moss_activation_memory_budget",
                    "启用梯度检查点时 activation_memory_budget 不参与显存规划（该预算只用于 "
                    "torch.compile 且关闭梯度检查点的组合），当前值会被忽略",
                    "warning",
                ))
        for severity, message in extra_params.problems(
            str(values.get(extra_params.PARAM_KEY) or ""),
            extra_params.provided_keys(
                PARAMS, architecture, values, skip_false=True,
                extra=("dataset_config", "output_dir", "output_name", "config_file", "method", "preset"),
            ),
            bound_output=config.get("output_layout") == "task_id",
        ):
            issue = ValidationIssue(extra_params.PARAM_KEY, message, severity)
            if issue not in issues:
                issues.append(issue)
        return tuple(issues)

    def caption_cache_notices(self):
        return (
            ("use_text_cache", "已开启「缓存文本编码器输出」：改动 caption 不会自动重算，请先清空对应缓存目录再提交"),
            ("use_vae_cache", "已开启「缓存 VAE 潜空间」：改过图片或分辨率请先清空对应缓存目录再提交"),
        )

    @staticmethod
    def _tag_csv_present(installation):
        """Whether layout_json can find the Danbooru tag CSV.

        The engine runs with the source tree as its home, so ``<source>/models`` is
        the authoritative place. ``ANIME_TOOLS_MODELS`` and ``ANIMA_HOME`` override
        that only when the caller explicitly set them.
        """
        candidates = []
        models = os.environ.get("ANIME_TOOLS_MODELS")
        if models:
            candidates.append(Path(models))
        home = os.environ.get("ANIMA_HOME")
        if home:
            candidates.append(Path(home) / "models")
        source = Path(installation.source_path)
        candidates.append(source / "models")
        name = "danbooru_tags_classified.csv"
        return any((folder / name).is_file() for folder in candidates)

    # ---- native files ----

    @staticmethod
    def _toml_value(value, list_arg=False):
        if list_arg:
            tokens = value if isinstance(value, list) else str(value).split()
            return "[" + ", ".join(json.dumps(str(t)) for t in tokens) + "]"
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value), ensure_ascii=False)

    def _output_name(self, config):
        """Model file stem: the engine appends ``-step…``/``-state``/``.safetensors``.

        Path separators and Windows-reserved characters are replaced so a task name
        can never create subdirectories or escape ``output_dir`` when the trainer
        saves (the output *folder* is sanitized by the service, the file stem is not).
        Quotes are left verbatim, matching the other adapters' export contract.
        """
        name = config.get("output_name") or config.get("name") or "lora"
        if config.get("output_layout") == "task_id":
            return safe_file_stem(name)
        text = _UNSAFE_FILENAME.sub("_", str(name))
        text = text.rstrip(" .")
        return text or "lora"

    def _preview_prompt_fallback(self, config, defaults=None):
        """First caption of the dataset, used when sampling is on but no prompt is set.

        The engine draws previews only when --sample_prompts is present; without this
        a user who enables "sample every N steps" would get no images at all.
        """
        params = config.get("params", {})
        defaults = defaults or {}

        def effective(key):
            return params.get(key, defaults.get(key))

        if str(effective("sample_prompts") or "").strip():
            return ""
        if not (effective("sample_every_n_steps") or effective("sample_every_n_epochs")
                or effective("sample_at_first")):
            return ""
        dataset = Path(str(config.get("dataset_path") or ""))
        if not dataset.is_dir():
            return ""
        extension = str(effective("caption_extension") or ".txt")
        for image in sorted(dataset.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            caption = image.with_suffix(extension)
            if not caption.is_file():
                continue
            text = caption.read_text(encoding="utf-8-sig", errors="replace").strip()
            if text:
                return text.splitlines()[0].strip()
        return ""

    def _render_train(self, config, prompt_directory, defaults=None, extras=None):
        """Flat TOML the engine reads through --config_file.

        ``library/config/io.py`` flattens sections, so every key is written at the
        top level; the dataset is a separate --dataset_config file.
        """
        architecture = config.get("architecture")
        params = config.get("params", {})
        defaults = defaults or {}
        if architecture == "moss_image":
            return self._render_moss_train(config, defaults)
        # The config references the prompt file by path, so it must exist even for a
        # preview or an export. write_prompts is a no-op when no prompt text is set.
        prompt_directory = Path(prompt_directory)
        # A preview can be rendered against a display placeholder ("<task-id>"), which
        # Windows cannot create as a directory; writing there raised OSError and lost
        # the whole preview. Render the reference but skip the side file.
        can_write_prompts = prompt_directory != Path(".") and _renderable_directory(prompt_directory)
        if can_write_prompts:
            write_prompts(config, prompt_directory)
        values = {}
        for p in PARAMS:
            if not p.native or not p.applies_to(architecture):
                continue
            if p.key in VERSIONED_KEYS and p.key not in defaults and p.key not in params:
                continue
            value = params.get(p.key, defaults.get(p.key, p.default))
            if p.key == "sample_prompts":
                # Handled before the empty-value skip: the engine only draws previews
                # when a prompt file is given, and the fallback may supply one.
                value = prompt_file(config, prompt_directory)
                if not value:
                    fallback = self._preview_prompt_fallback(config, defaults)
                    if not fallback or not can_write_prompts:
                        continue
                    target = prompt_directory / PROMPT_FILENAME
                    atomic_text(target, fallback + "\n")
                    value = str(target)
            elif value is None or (isinstance(value, str) and value == ""):
                continue
            elif p.key == "target_res":
                value = [int(x) for x in str(value).split()]
            elif p.key == "lr_warmup_steps" and isinstance(value, float) and value >= 1 and value.is_integer():
                value = int(value)
            elif value is False:
                # An explicit False must survive for knobs the engine defaults to true
                # (use_vae_cache / use_text_cache / use_shuffled_caption_variants …):
                # dropping the key would silently turn the feature back on. For args
                # whose default is already false the key stays out, as before.
                if p.type != "bool" or (defaults.get(p.key, p.default) is not True):
                    continue
            values[p.native_key or p.key] = (value, p.list_arg)
        # Job-level keys only exist for a queued job; a pure parameter export has
        # neither the base model path nor the output folder, so they are written
        # only when present. The export path must never require them.
        if str(config.get("base_model_path") or "").strip():
            values["pretrained_model_name_or_path"] = (config["base_model_path"], False)
        if str(config.get("output_dir") or "").strip():
            values["output_dir"] = (config["output_dir"], False)
            values["output_name"] = (self._output_name(config), False)
        # A logging backend without a directory makes prepare_accelerator raise
        # ("logging_dir is required when log_with is tensorboard"). base.toml points
        # logging_dir at the engine checkout, so that relative path is excluded from
        # the defaults; render a task-local directory unless the user picked one.
        log_with = str(values.get("log_with", ("", False))[0] or "")
        if log_with in {"tensorboard", "wandb", "all"} and "logging_dir" not in values:
            root = str(config.get("output_dir") or "").strip()
            values["logging_dir"] = (str(Path(root) / "logs") if root else "logs", False)
        for name, value in _custom_values(str(params.get(extra_params.PARAM_KEY) or ""),
                                          bound_output=config.get("output_layout") == "task_id").items():
            if name == "target_res":
                value = [int(x) for x in str(value).split()]
            values[name] = (value, False)
        # With --config_file the engine skips the base → preset → method merge, so the
        # LoRA method defaults the form does not model are pinned here explicitly.
        for name, value in (extras or {}).items():
            values.setdefault(name, (value, False))
        return "".join(f"{key} = {self._toml_value(value, list_arg)}\n" for key, (value, list_arg) in values.items())

    # Flag lists in argv order. The cache stage and the train stage are rendered by
    # the same builders below, so the native view, the export and the real run cannot
    # disagree about which flags (and which --no- forms) the trainer receives.
    _MOSS_CACHE_KEYS = (
        "moss_resolution", "moss_caption_format", "moss_cache_text_encoder",
        "moss_cache_vae_dtype", "moss_cache_max_text_tokens", "moss_cache_overwrite",
        "moss_cache_skip_text", "moss_cache_skip_latents", "moss_cache_save_crops",
        "moss_ref_scale", "moss_fixed_caption",
    ) + tuple(p.key for p in MOSS_CACHE_OPTIONS)
    _MOSS_TRAIN_KEYS = (
        "moss_algo", "moss_epochs", "moss_max_steps", "moss_grad_accum", "moss_seed",
        "moss_rank", "moss_alpha", "moss_factor", "moss_down_init",
        "moss_svd_slice", "moss_svd_slice_count", "moss_svd_drop_loud",
        "moss_channel_stats", "moss_channel_mean", "moss_optimizer",
        "moss_lora_dtype", "moss_lr", "moss_warmup_ratio",
        "moss_max_grad_norm", "moss_save_every_epochs", "moss_blocks_to_swap",
        "moss_activation_memory_budget", "moss_grad_checkpointing",
        "moss_log_every", "moss_caption_dropout", "moss_plain_caption_rate",
        "moss_caption_variant_rate", "moss_ref_timestep", "moss_ref_isolate",
        "moss_repa_weight", "moss_repa_layer", "moss_repa_pe_root",
        "moss_targets", "moss_sigma_mu", "moss_sigma_s", "moss_sigma_cutoff",
        "moss_sigma_uniform", "moss_fit_report", "moss_ema_window", "moss_ema_extra",
        "moss_region_weight", "moss_rest_weight", "moss_region_threshold",
        "moss_region_min_tokens", "moss_region_dilate",
        "moss_tone_weight", "moss_tone_log", "moss_tone_pool", "moss_tone_dilate",
        "moss_tone_dirs",
    ) + tuple(p.key for p in MOSS_TRAIN_OPTIONS)

    def _moss_cache_flags(self, config, values, cache_dir, model_dir):
        """argv tail of the cache stage (everything after the script path)."""
        flags = ["--src", str(config.get("dataset_path") or ""), "--out", str(cache_dir)]
        flags += list(self._cli(values, self._MOSS_CACHE_KEYS))
        if model_dir:
            flags += ["--model_dir", str(model_dir)]
        return flags

    def _moss_train_flags(self, config, values, cache_dir, output_file, model_dir):
        """argv tail of the train stage (everything after the script path).

        The selected Diffusers directory determines the DiT variant.
        """
        chosen = Path(str(values.get("moss_model_dir") or ""))
        variant = "sft" if chosen.name.casefold() == "moss-image-sft" else "base"
        flags = ["--cache", str(cache_dir), "--output", str(output_file), "--model", variant]
        flags += list(self._cli(values, self._MOSS_TRAIN_KEYS))
        # These three switches default to ON in the current engine. State the off
        # form explicitly so the form value is what the trainer receives.
        flags.append("--swap_h2d_only" if values.get("moss_swap_h2d_only", True)
                     else "--no-swap_h2d_only")
        flags.append("--expandable_segments" if values.get("moss_expandable_segments", True)
                     else "--no-expandable_segments")
        # The engine defaults compile to ON too: always state the switch, otherwise
        # a machine without cl.exe aborts with InvalidCxxCompiler.
        flags.append("--compile" if values.get("moss_compile") else "--no-compile")
        if model_dir:
            flags += ["--model_dir", str(model_dir)]
        return flags

    @staticmethod
    def _join_flags(flags):
        """Render argv tokens as one readable line, quoting tokens with spaces."""
        return " ".join(
            f'"{token}"' if any(char.isspace() for char in str(token)) else str(token)
            for token in flags
        )

    def _render_moss_train(self, config, defaults=None):
        """MOSS takes CLI flags only — its request dataclasses write argv, not a file.

        Both stages are rendered from the same flag builders ``preview_commands``
        uses, so the preview/export lists every argument the trainer will receive —
        including ``--src/--out``, ``--cache/--output/--model``, ``--model_dir`` and
        the always-explicit ``--no-compile`` / swap / expandable forms. Only the
        interpreter and script paths are left out.
        """
        values = self._values_from(config, defaults)
        cache_dir = self._cache_directory(config, values)
        model_dir = self._moss_model_dir(config, values)
        output_file = self._moss_output_file(config)
        stages = (
            ("1/2", "scripts/moss/cache.py",
             self._moss_cache_flags(config, values, cache_dir, model_dir)),
            ("2/2", "scripts/moss/train.py",
             self._moss_train_flags(config, values, cache_dir, output_file, model_dir)),
        )
        lines = ["# MOSS-Image 没有配置文件：以下为各阶段实际收到的参数（不含解释器与脚本路径）"]
        for ordinal, script, flags in stages:
            lines.append(f"# 阶段 {ordinal}：{script}")
            lines.append(self._join_flags(flags))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _cache_flags(params, defaults=None):
        """Effective (use_vae_cache, use_text_cache); unset means the engine default."""
        defaults = defaults or {}

        def flag(key):
            value = params.get(key, defaults.get(key))
            return True if value is None else bool(value)

        return flag("use_vae_cache"), flag("use_text_cache")

    def _preprocess_settings(self, source, config, values):
        path = Path(source) / "configs/preprocess.toml"
        if not path.is_file():
            return {}
        source_recipe = tomllib.loads(path.read_text(encoding="utf-8"))
        # Both caption-stage roots must belong to the current task. Curation tools
        # that edit the master dataset are outside a training job.
        recipe = {k: v for k, v in source_recipe.items() if k in {
            "caption_shuffle_variants", "caption_tag_dropout_rate", "caption_tag_randomize_rate",
            "caption_position_clauses", "caption_implied", "caption_correct_order", "freefit_max_ratio",
        }}
        recipe.update(source_image_dir=config["dataset_path"], resized_image_dir=str(self._staging_directory(config)))
        if values.get("target_res"):
            recipe["target_res"] = [int(x) for x in str(values["target_res"]).split()]
        if not values.get("use_shuffled_caption_variants"):
            recipe["caption_shuffle_variants"] = 0
        return recipe

    def _staging_directory(self, config):
        """Per-task folder bucket-resized images land in (the subset's image_dir).

        The resize stage never resizes ``--src`` in place, so the dataset config the
        trainer reads must point at its own resized tree. Keeping it inside the task
        output also makes ``collect_artifacts`` prune it as an input.
        """
        return Path(config["output_dir"]) / "resized"

    def _dataset_toml(self, config, defaults=None):
        """kohya-style dataset config, the only shape this engine accepts.

        Supported subset keys are image_dir/cache_dir/num_repeats/recursive and the
        caption settings; there is no resolution or bucket key, so resolution tiers
        belong to the training and resize requests.
        """
        params = self._values_from(config, defaults)
        vae_cache, text_cache = self._cache_flags(params, defaults)
        # With a preprocess stage the trainer reads the resized tree; with both caches
        # off (live encoding, no preprocess) it must read the originals instead.
        image_dir = (
            self._staging_directory(config) if (vae_cache or text_cache)
            else Path(config["dataset_path"])
        )
        text = "[general]\n"
        if config.get("dataset_config_path"):
            text += "dataset_config = " + json.dumps(str(config["dataset_config_path"]), ensure_ascii=False) + "\n"
        text += "\n[[datasets]]\n"
        text += f"batch_size = {int(params.get('batch_size') or 1)}\n"
        text += "\n  [[datasets.subsets]]\n"
        text += "  image_dir = " + json.dumps(str(image_dir), ensure_ascii=False) + "\n"
        cache = params.get("cache_directory")
        if not (isinstance(cache, str) and cache.strip()):
            cache = str(Path(config["output_dir"]) / "cache")
        text += "  cache_dir = " + json.dumps(str(cache), ensure_ascii=False) + "\n"
        text += "  num_repeats = " + str(int(params.get("num_repeats") or 1)) + "\n"
        text += "  recursive = " + str(bool(params.get("recursive", True))).lower() + "\n"
        text += "  caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += "  caption_separator = " + json.dumps(params.get("caption_separator") or ",") + "\n"
        mask_dir = params.get("mask_dir")
        if isinstance(mask_dir, str) and mask_dir.strip():
            text += "  mask_dir = " + json.dumps(mask_dir, ensure_ascii=False) + "\n"
        return text

    def engine_context(self, installation, architecture=None):
        """``(defaults, extras)`` a rendered preview/export needs to match a real job.

        A queued job renders with the engine chain because ``write_native_config``
        has the installation. The preview (``validate``) and the parameter export
        only have the draft, so they used to fall back to the adapter's static
        defaults and lost the chain values (``learning_rate``,
        ``discrete_flow_shift``, the pinned method keys, …). Callers that do have an
        installation pass it through here and both renderings agree.
        """
        source = getattr(installation, "source_path", installation)
        architecture = architecture or "anima"
        extras = {} if architecture == "moss_image" else engine_chain_extras(source, architecture)
        return default_values(source), extras

    @staticmethod
    def _installation_of(value):
        """Accept an installation-like object where a defaults dict is expected."""
        if value is None or isinstance(value, dict):
            return None
        return value if getattr(value, "source_path", None) else None

    def native_config_text(self, config, dataset_config, defaults=None, extras=None,
                           installation=None):
        # Preview, write and export share this one renderer so the shown text, the
        # written file and the exported file can never diverge.
        installation = installation or self._installation_of(defaults)
        if installation is not None:
            chain_defaults, chain_extras = self.engine_context(
                installation, config.get("architecture"))
            defaults = chain_defaults if defaults is None or self._installation_of(defaults) else defaults
            extras = chain_extras if extras is None else extras
        return self._render_train(
            config,
            Path(dataset_config).parent if dataset_config else Path("."),
            defaults,
            extras,
        )

    def export_parameters(self, config, defaults=None, installation=None):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 Anima LoRA 适配器支持")
        notice = (
            "此文件只包含训练参数；Anima 运行时还需同目录下的 dataset.toml"
            "（通过 --dataset_config 提供），并确认底模、Qwen3 与 VAE 已就绪。"
        )
        text = self.native_config_text(config, None, defaults, installation=installation)
        if exported_prompt_files(config):
            notice += " 启用采样时会同时导出 sample_prompts.txt 配套文件。"
        return "# " + notice + "\n" + text, (notice,)

    def export_parameter_files(self, config):
        return exported_prompt_files(config)

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text

        job_directory = Path(job_directory)
        target = job_directory / "config.native.toml"
        # The same (defaults, extras) pair a preview/export gets when handed the
        # installation, so the written file and the shown file are byte-identical.
        defaults, extras = self.engine_context(installation, config.get("architecture"))
        write_prompts(config, job_directory)
        if config.get("architecture") == "moss_image":
            atomic_text(target, self.native_config_text(config, None, defaults))
            atomic_text(job_directory / "stages.json", json.dumps(
                [list(stage) for stage in self.preview_commands(installation, config, target)],
                ensure_ascii=False, indent=2,
            ))
            return target
        recipe = self._preprocess_settings(installation.source_path, config, self._values_from(config, defaults))
        if recipe:
            atomic_text(job_directory / "preprocess.toml",
                        "".join(f"{k} = {self._toml_value(v)}\n" for k, v in recipe.items()))
        atomic_text(job_directory / "dataset.toml", self._dataset_toml(config, defaults))
        # --config_file skips the engine's base → preset → method merge, so the chain
        # values the form does not model are pinned into the file itself.
        atomic_text(target, self.native_config_text(
            config, job_directory / "dataset.toml", defaults, extras))
        atomic_text(job_directory / "stages.json", json.dumps(
            [list(command) for command in self.preview_commands(installation, config, target)],
            ensure_ascii=False, indent=2,
        ))
        return target

    # ---- launch ----

    def _values_from(self, config, defaults=None):
        """Every parameter's effective value, hidden ones included (flag builders skip)."""
        params = config.get("params", {})
        defaults = defaults or {}
        values = {p.key: params.get(p.key, defaults.get(p.key, p.default)) for p in PARAMS
                  if p.key not in VERSIONED_KEYS or p.key in defaults or p.key in params}
        if config.get("architecture") == "anima":
            if config.get("base_model_path"):
                values["pretrained_model_name_or_path"] = config["base_model_path"]
            values.update(_custom_values(str(params.get(extra_params.PARAM_KEY) or ""),
                                         bound_output=config.get("output_layout") == "task_id"))
        return values

    def _values(self, config, installation=None):
        if installation is None:
            return self._values_from(config)
        defaults, extras = self.engine_context(installation, config.get("architecture"))
        return {**extras, **self._values_from(config, defaults)}

    def _cli(self, values, keys):
        args = []
        for key in keys:
            value = values.get(key)
            if value is None or (isinstance(value, str) and value == ""):
                continue
            p = self._param(key)
            name = (p.native_key or p.key) if p else key
            if isinstance(value, bool):
                if value:
                    args.append("--" + name)
                elif p.default is True:
                    args.append("--no-" + name)
                continue
            if p.type == "int" and isinstance(value, float) and value.is_integer():
                value = int(value)  # Legacy forms stored whole reference scales as floats.
            args.extend(("--" + name, str(value)))
        return tuple(args)

    def _cache_directory(self, config, values=None):
        values = self._values_from(config) if values is None else values
        custom = values.get("moss_cache_directory")
        if isinstance(custom, str) and custom.strip():
            return Path(custom)
        return Path(str(config.get("output_dir") or "")) / "moss-cache"

    def _moss_model_dir(self, config, values=None):
        """Both variants share the Base VAE and tokenizer under this parent."""
        values = self._values_from(config) if values is None else values
        chosen = values.get("moss_model_dir")
        if isinstance(chosen, str) and chosen.strip():
            return Path(chosen.strip()).parent
        return None

    def _moss_output_file(self, config):
        """``--output`` is a FILE path. A pure export has no job folder yet, so the
        view gets a placeholder it cannot be confused with a real path."""
        name = self._output_name(config) + ".safetensors"
        root = str(config.get("output_dir") or "").strip()
        return Path(root) / name if root else Path("<output_dir>") / name

    def preview_commands(self, installation, config, native_config):
        python = str(installation.python_executable)
        source = str(installation.source_path)
        if config.get("architecture") == "moss_image":
            values = self._values(config, installation)
            cache_dir = self._cache_directory(config, values)
            model_dir = self._moss_model_dir(config, values)
            output_file = self._moss_output_file(config)
            cache = (
                python, "-u", str(Path(source) / "scripts" / "moss" / "cache.py"),
                *self._moss_cache_flags(config, values, cache_dir, model_dir),
            )
            train = (
                python, "-u", str(Path(source) / "scripts" / "moss" / "train.py"),
                *self._moss_train_flags(config, values, cache_dir, output_file, model_dir),
            )
            return (cache, train)
        values = self._values(config, installation)
        commands = []
        vae_cache, text_cache = self._cache_flags(values)
        if vae_cache or text_cache:
            root = Path(source)
            original = str(config["dataset_path"])
            staged = str(self._staging_directory(config))
            cache = str(values.get("cache_directory") or Path(config["output_dir"]) / "cache")
            recursive = bool(values.get("recursive"))
            commands.append((python, "-u", str(root / "tasks.py"), "preprocess-resize",
                             "--src", original, "--dst", staged, "--recursive" if recursive else "--no-recursive",
                             *(("--target_res", *str(values["target_res"]).split()) if values.get("target_res") else ())))
            recurse = ("--recursive",) if recursive else ()
            recipe = self._preprocess_settings(root, config, values)
            official_captions = bool(recipe) and values.get("caption_extension") == ".txt" and (
                recipe.get("caption_implied", True) or recipe.get("caption_shuffle_variants", 4)
                or recipe.get("caption_position_clauses") or recipe.get("caption_correct_order"))
            if official_captions:
                commands.append((python, "-u", str(root / "tasks.py"), "preprocess-captions"))
            if vae_cache:
                latent = root / "scripts" / "preprocess" / "cache_latents.py"
                encode = (python, "-u", str(latent), "--dir", staged, "--cache_dir", cache,
                          "--vae", str(values.get("vae") or ""), "--batch_size", "1",
                          "--chunk_size", "0", "--no_half_vae", *recurse)
                commands.append(encode)
                if values.get("sigma_lowres"):
                    for key in ("sigma_lowres_route", "sigma_lowres_route2"):
                        if values.get(key):
                            commands.append((*encode, "--sigma_demote", str(values[key])))
            if text_cache:
                te = root / "scripts" / "preprocess" / "cache_text_embeddings.py"
                text_source = staged if official_captions else original
                flags = ["--dir", text_source, "--cache_dir", cache, "--match_images_from", staged,
                         "--qwen3", str(values.get("qwen3") or ""), *recurse]
                if values.get("cache_llm_adapter_outputs"):
                    flags.extend(("--dit", str(values.get("pretrained_model_name_or_path") or "")))
                for key in ("vocab_pack", "t5_tokenizer_path"):
                    if values.get(key):
                        flags.extend(("--" + key, str(values[key])))
                if values.get("use_shuffled_caption_variants"):
                    for key, fallback in (("caption_shuffle_variants", 4), ("caption_tag_dropout_rate", 0.1),
                                          ("caption_tag_randomize_rate", 0.0)):
                        flags.extend(("--" + key, str(recipe.get(key, fallback))))
                if values.get("cache_directory") or Path(cache).is_dir():
                    commands.append((python, "-u", str(Path(__file__).with_name("cache_checks.py")), "verify",
                                     text_source, staged, cache, str(values.get("vocab_pack") or ""), str(int(recursive))))
                commands.append((python, "-u", str(te), *flags))
            elif not official_captions:
                commands.append((python, "-u", str(Path(__file__).with_name("cache_checks.py")), "captions",
                                 original, staged, str(values.get("caption_extension") or ".txt")))
        train = (
            python, "-u", str(Path(source) / "train.py"),
            "--config_file", str(native_config),
            "--dataset_config", str(Path(native_config).parent / "dataset.toml"),
            # --config_file makes read_config_from_file skip the base → preset →
            # method merge, so the intended chain is stated explicitly; the values the
            # form does not model are also pinned in the config file itself.
            "--method", "lora",
            "--preset", "default",
        )
        commands.append(train)
        return tuple(commands)

    def preview_argv(self, installation, config, native_config):
        return self.preview_commands(installation, config, native_config)[-1]

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        files = JobFiles(native_config.parent)
        if not files.stages.is_file():
            raise RuntimeError("缺少 Anima LoRA 运行阶段定义 stages.json")
        commands = files.read_stages()
        if not commands:
            raise RuntimeError("Anima LoRA 运行阶段为空")
        environment = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            job = files.read_job() or {}
        except (FileNotFoundError, ValueError):
            job = {}
        # submit() writes job.json after build_launch(), so the architecture is not
        # always available here. The MOSS stage is recognizable by its script path;
        # everything else on this adapter is the Anima line.
        is_moss = any(
            Path(str(part)).as_posix().endswith("scripts/moss/train.py")
            for part in commands[-1]
        )
        # Anima's own release-pause protocol: writing pause.request makes train.py
        # save a resumable state at the next optimizer step and exit cleanly. MOSS
        # has no such watcher; it must not receive this environment.
        if job.get("architecture") != "moss_image" and not is_moss:
            environment["ANIMA_DAEMON_JOB_DIR"] = str(native_config.parent)
            preprocess = native_config.parent / "preprocess.toml"
            if preprocess.is_file():
                environment["CONFIG_FILE"] = str(preprocess)
                environment["CAPTION_AUTOTAG"] = "0"
        # torch.compile shells out to cl.exe; give the training process the same
        # INCLUDE/LIB/PATH a Developer prompt would. Covers both lines (MOSS
        # ``--compile`` / ``moss_compile`` and Anima ``torch_compile = true``); MOSS
        # defaults compile to ON and Anima's base.toml does too, so the rendered
        # config is authoritative when job.json was not written yet.
        values = job.get("params", {})
        try:
            text = native_config.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if _requests_compile(values, text):
            environment.update(_msvc_environment())
        return LaunchSpec(
            commands[-1],
            Path(installation.source_path),
            environment,
            commands[:-1],
        )

    # ---- logs / artifacts ----

    # MOSS prints plain lines instead of a tqdm bar, so the shared kohya/musubi
    # patterns never match them:
    #   "  step 6/1770 loss 0.3960 |g| 0.052 sigma 0.837 lr … 3.78s/step"
    #   "  progress 6/1770 epoch 1/15 loss 0.3813 eta 1:51:07"
    _RESUMING = re.compile(r"\bresuming at epoch (\d+), step (\d+)\b")
    _SKIPPED_STEPS = re.compile(r"\bskipping (\d+) steps: \d+ full epochs? \+ \d+ batches of epoch (\d+)\b")
    _MOSS_TOTAL = re.compile(r"(\d+) steps \((\d+) epochs x \d+\)")
    _MOSS_PROGRESS = re.compile(r"(?:progress|step)\s+(\d+)/(\d+)\s+(?:loss |epoch )")
    _MOSS_EPOCH = re.compile(r"\bepoch\s+(\d+)/(\d+)\b")
    _MOSS_EPOCH_SUMMARY = re.compile(r"epoch\s+(\d+)/(\d+): loss\b")
    _MOSS_LOSS = re.compile(r"\bloss\s+(" + NUMBER + r")(?=\s|$)")
    _MOSS_LR = re.compile(r"\blr\s+(" + NUMBER + r")(?=\s|$)")

    def parse_log(self, line):
        text = plain_log(line)
        progress = self._MOSS_PROGRESS.match(text)
        if progress:
            payload = {"step": int(progress[1]), "total_steps": int(progress[2])}
            epoch = self._MOSS_EPOCH.search(text)
            if epoch:
                payload.update(epoch_counts(epoch))
            for key, value in (("loss", read_number(self._MOSS_LOSS.search(text))),
                               ("lr", read_number(self._MOSS_LR.search(text))),
                               ("it_per_sec", training_rate(text)), ("eta_seconds", parse_eta(text))):
                if value is not None:
                    payload[key] = value
            return (TrainingEvent("progress", payload),)
        total = self._MOSS_TOTAL.match(text)
        if total:
            return (TrainingEvent("progress", {"total_steps": int(total[1]), "total_epochs": int(total[2])}),)
        epoch = self._MOSS_EPOCH_SUMMARY.match(text)
        if epoch:
            # This is the epoch mean, not the loss of its final optimizer step.
            return (TrainingEvent("progress", epoch_counts(epoch)),)
        if not text.startswith("{"):
            resume = self._RESUMING.search(text)
            if resume:
                step = int(resume[2])
                return (TrainingEvent("progress", {"epoch": int(resume[1]), "step": step, "resume_step": step}),)
            skipped = self._SKIPPED_STEPS.search(text)
            if skipped:
                # A mid-first-epoch resume emits this skip plan without the
                # "resuming at epoch" banner. Its first number is optimizer steps.
                step = int(skipped[1])
                return (TrainingEvent("progress", {"epoch": int(skipped[2]), "step": step, "resume_step": step}),)
            return parse_progress(text, bar_kind="remaining_steps")
        try:
            record = json.loads(text)
        except ValueError:
            return ()
        if not isinstance(record, dict):
            return ()
        event = record.get("ev")
        if event not in (None, "run_start", "step", "run_end"):
            return ()
        payload = {}
        step = record.get("global_step", record.get("step"))
        if event == "run_end":
            step = record.get("final_step")
        for key, value in (("step", step), ("epoch", record.get("epoch")),
                           ("total_steps", record.get("total_steps")), ("total_epochs", record.get("total_epochs"))):
            if type(value) is int and value >= 0:
                payload[key] = value
        if "step" in payload:
            for key, value in (("loss", record.get("loss", record.get("loss/current"))),
                               ("lr", record.get("lr", record.get("learning_rate")))):
                if type(value) in (float, int) and math.isfinite(value):
                    payload[key] = float(value)
        return (TrainingEvent("progress", payload),) if payload else ()

    def recovery_capabilities(self, installation, architecture=None):
        # The Anima entry saves Accelerate state (--save_state / --resume). LP also
        # enables the engine's release-pause watcher for this line, so "pause" saves a
        # resumable state before exiting. The MOSS line has no state save at all, so
        # stopping it cannot be continued and it must not be presented as a pause.
        if architecture == "moss_image":
            return RecoveryCapabilities(graceful_stop=False)
        return RecoveryCapabilities(graceful_stop=True, resume_full_state=True)

    def resume_sources(self, job_directory):
        try:
            config = JobFiles(job_directory).read_job()
        except (OSError, ValueError, KeyError):
            return ()
        if config.get("architecture") == "moss_image":
            return ()
        return accelerate_resume_sources(Path(config.get("output_dir") or ""))

    def resume_parameters(self, source: ResumeSource):
        return {"resume": str(source.path)}

    # A run keeps its caches inside the task folder (<output>/moss-cache, <output>/cache,
    # <output>/resized) and those files are .safetensors/images too. They are inputs, not
    # results, so they must never be listed as checkpoints or samples. Accelerator
    # ``*-state`` folders are resumable state, not LoRA weights, for the same reason.
    _CACHE_DIRS = {"cache", "moss-cache", "latent-cache", "te-cache", "_sample_te_cache", "resized"}
    _STATE_SUFFIX = "-state"

    @classmethod
    def _is_input_directory(cls, name: str) -> bool:
        lowered = name.lower()
        return lowered in cls._CACHE_DIRS or lowered.endswith(cls._STATE_SUFFIX)

    # The engine writes the kohya layout (``lora_unet_layers_N_proj_<name>.lora_*``),
    # which is exactly what ComfyUI's MOSS loader reads, so no key rewriting is needed.

    def finalize_artifacts(self, job_directory, python, source_path):
        """Nothing to do: the engine already writes ComfyUI-readable keys.

        Its LoRAs use the kohya layout (``lora_unet_layers_N_proj_<name>.lora_*``,
        verified against the kohaku export that ComfyUI loads), so a converted copy
        would only duplicate the file without changing what ComfyUI reads.
        """
        return ()

    def collect_artifacts(self, job_directory):
        config = JobFiles(job_directory).read_job()
        root = Path(config["output_dir"])
        if not root.is_dir():
            # The engine creates the folder on its first save.
            return ()
        found = []
        moss_stem = self._output_name(config) if config.get("architecture") == "moss_image" else None
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(self._is_input_directory(part) for part in path.relative_to(root).parts[:-1]):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                step = None
                final = True if moss_stem and path == root / (moss_stem + ".safetensors") else None
                if moss_stem and path.parent == root / moss_stem:
                    match = re.fullmatch(re.escape(moss_stem) + r"_([se])(\d+)", path.stem)
                    if match:
                        final = False
                        if match[1] == "s":
                            step = int(match[2])
                    elif path.stem == moss_stem + "_raw" or re.fullmatch(
                        re.escape(moss_stem) + r"_ema[\d.]+(?:e[+-]?\d+)?", path.stem
                    ):
                        final = True  # The final save can include raw and additional EMA copies.
                found.append(Artifact(path, "checkpoint", True, step=step, final=final))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
