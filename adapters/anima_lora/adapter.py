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
    ValidationIssue,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.sampling import PROMPT_FILENAME, exported_prompt_files, prompt_file, write_prompts
from adapters.anima_lora.params import ARCHITECTURES, MOSS_PARAMS, PARAMS
from app.services.common import atomic_text
from app.services.jobs import JobFiles
from app.schemas.training import LaunchSpec, TrainingEvent

# Assets the Anima entry cannot run without: the DiT weights, its Qwen3 text
# encoder, and the VAE (the T5 tokenizer is bundled by the engine).
ANIMA_REQUIRED = ("pretrained_model_name_or_path", "qwen3", "vae")


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
    engine_id = "anima_lora"
    display_name = "Anima LoRA"
    clone_url = "https://github.com/sorryhyun/anima_lora.git"
    # Anima needs a base-model file; its MOSS line trains from a model directory.
    base_model_optional_architectures = ("moss_image",)
    adapter_version = "0.1.0"
    installation_source_help = "只列出当前 pyproject.toml 声明的依赖组；LP 原样调用官方 uv 流程，不自行拼装 torch。"
    installation_help = (
        "使用官方 uv sync 按提交的 uv.lock 创建环境（Python 3.13 + torch 2.14 + CUDA 13.2）。"
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
                command.extend(["--find-links", torch_mirror_url.rstrip("/")])
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
        return tuple(issues)

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
        required = {
            "anima": ANIMA_REQUIRED,
            "moss_image": (),
        }
        params = tuple(
            replace(p, required_for=required.get(architecture, ()))
            for architecture in ("anima", "moss_image")
            for p in PARAMS
            if p.applies_to(architecture)
        )
        # A parameter that applies to every architecture is returned once, with the
        # requirements of the architectures that actually need it.
        deduped = {}
        for p in params:
            previous = deduped.get(p.key)
            if previous is None:
                deduped[p.key] = p
            else:
                deduped[p.key] = replace(
                    p, required_for=tuple(sorted(set(previous.required_for) | set(p.required_for)))
                )
        return dict(
            installation_id=installation.installation_id,
            architectures=list(available),
            params=specs(tuple(deduped.values())),
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

    def validate(self, installation, config):
        issues = []
        architecture = config.get("architecture")
        available = {a.id for a in self.architectures(installation.source_path)}
        if architecture not in available:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        applicable = tuple(p for p in PARAMS if p.applies_to(architecture))
        for p in applicable:
            if not p.native and p.unsupported_reason:
                continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
        for key in ANIMA_REQUIRED if architecture == "anima" else ():
            value = values.get(key, self._param(key).default if self._param(key) else None)
            # The job's base-model picker is what the config file actually writes, so
            # a selected base model satisfies this field even when the form left the
            # parameter empty (and vice versa, the field alone is not enough).
            if key == "pretrained_model_name_or_path" and str(config.get("base_model_path") or "").strip():
                value = config["base_model_path"]
            if value is None or (isinstance(value, str) and not value.strip()):
                label = self._param(key).label if self._param(key) else key
                issues.append(ValidationIssue(key, f"该底模类型需要填写：{label}"))
        if architecture == "moss_image":
            # torch.compile needs MSVC (cl.exe) on Windows; without it the first
            # forward aborts with InvalidCxxCompiler, which is expensive to diagnose
            # from a stack trace. Catch it before the job is queued. A normal Visual
            # Studio install is not on PATH, so vswhere is consulted as well.
            if values.get("moss_compile") and not _msvc_available():
                issues.append(ValidationIssue(
                    "moss_compile",
                    "该机器找不到 C++ 编译器（cl.exe）：开启 torch.compile 会在第一次 forward 时以 "
                    "InvalidCxxCompiler 失败。请关闭该项，或安装 Visual Studio 的「使用 C++ 的桌面开发」",
                    "error",
                ))
        for severity, message in extra_params.problems(
            str(values.get(extra_params.PARAM_KEY) or ""),
            extra_params.provided_keys(
                PARAMS, architecture, values, skip_false=True,
                extra=("dataset_config", "output_dir", "output_name", "config_file", "method", "preset"),
            ),
        ):
            issues.append(ValidationIssue(extra_params.PARAM_KEY, message, severity))
        return tuple(issues)

    def caption_cache_notices(self):
        return (
            ("use_text_cache", "已开启「缓存文本编码器输出」：改动 caption 不会自动重算，请先清空对应缓存目录再提交"),
            ("use_vae_cache", "已开启「缓存 VAE 潜空间」：改过图片或分辨率请先清空对应缓存目录再提交"),
        )

    # ---- native files ----

    @staticmethod
    def _toml_value(value, list_arg=False):
        if list_arg:
            tokens = value if isinstance(value, list) else str(value).split()
            return "[" + ", ".join(json.dumps(str(t)) for t in tokens) + "]"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value), ensure_ascii=False)

    def _output_name(self, config):
        return config.get("output_name") or config.get("name") or "lora"

    def _preview_prompt_fallback(self, config):
        """First caption of the dataset, used when sampling is on but no prompt is set.

        The engine draws previews only when --sample_prompts is present; without this
        a user who enables "sample every N steps" would get no images at all.
        """
        params = config.get("params", {})
        if str(params.get("sample_prompts") or "").strip():
            return ""
        if not (params.get("sample_every_n_steps") or params.get("sample_every_n_epochs")
                or params.get("sample_at_first")):
            return ""
        dataset = Path(str(config.get("dataset_path") or ""))
        if not dataset.is_dir():
            return ""
        extension = str(params.get("caption_extension") or ".txt")
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

    def _render_train(self, config, prompt_directory):
        """Flat TOML the engine reads through --config_file.

        ``library/config/io.py`` flattens sections, so every key is written at the
        top level; the dataset is a separate --dataset_config file.
        """
        architecture = config.get("architecture")
        params = config.get("params", {})
        if architecture == "moss_image":
            return self._render_moss_train(config)
        # The config references the prompt file by path, so it must exist even for a
        # preview or an export. write_prompts is a no-op when no prompt text is set.
        write_prompts(config, prompt_directory)
        values = {}
        for p in PARAMS:
            if not p.native or not p.applies_to(architecture):
                continue
            value = params.get(p.key, p.default)
            if p.key == "sample_prompts":
                # Handled before the empty-value skip: the engine only draws previews
                # when a prompt file is given, and the fallback may supply one.
                value = prompt_file(config, prompt_directory)
                if not value:
                    fallback = self._preview_prompt_fallback(config)
                    if not fallback:
                        continue
                    target = Path(prompt_directory) / PROMPT_FILENAME
                    atomic_text(target, fallback + "\n")
                    value = str(target)
            elif value is None or value is False or (isinstance(value, str) and value == ""):
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
        for name, value in extra_params.merge({}, str(params.get(extra_params.PARAM_KEY) or "")).items():
            values[name] = (value, False)
        return "".join(f"{key} = {self._toml_value(value, list_arg)}\n" for key, (value, list_arg) in values.items())

    def _render_moss_train(self, config):
        """MOSS takes CLI flags only — its request dataclasses write argv, not a file.

        The native view therefore lists the flags the trainer receives, so a preview,
        an export and the real run all show the same thing.
        """
        params = config.get("params", {})
        lines = ["# MOSS-Image 没有配置文件：以下为 scripts/moss/train.py 会收到的参数"]
        for p in MOSS_PARAMS:
            value = params.get(p.key, p.default) if p.native else None
            if value is None or value is False or (isinstance(value, str) and value == ""):
                continue
            flag = f"--{p.native_key or p.key}"
            # BooleanOptionalAction switches are bare flags, so no value is printed.
            lines.append(flag if value is True else f"{flag} {value}")
        return "\n".join(lines) + "\n"

    def _moss_args(self, values):
        args = []
        for key, value in values:
            if isinstance(value, bool):
                if value:
                    args.append("--" + key)
                continue
            args.extend(("--" + key, str(value)))
        return args

    def _dataset_toml(self, config):
        """kohya-style dataset config, the only shape this engine accepts.

        Supported subset keys are image_dir/cache_dir/num_repeats/recursive and the
        caption settings; there is no resolution or bucket key, so the UI values for
        width/height are not written here.
        """
        params = config.get("params", {})
        text = "[general]\n"
        if config.get("dataset_config_path"):
            text += "dataset_config = " + json.dumps(str(config["dataset_config_path"]), ensure_ascii=False) + "\n"
        text += "\n[[datasets]]\n"
        text += f"batch_size = {int(params.get('batch_size') or 1)}\n"
        text += "\n  [[datasets.subsets]]\n"
        text += "  image_dir = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
        cache = params.get("cache_directory")
        if not (isinstance(cache, str) and cache.strip()):
            cache = str(Path(config["output_dir"]) / "cache")
        text += "  cache_dir = " + json.dumps(str(cache), ensure_ascii=False) + "\n"
        text += "  num_repeats = " + str(int(params.get("num_repeats") or 1)) + "\n"
        text += "  recursive = " + str(bool(params.get("recursive", True))).lower() + "\n"
        text += "  caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += "  caption_separator = " + json.dumps(params.get("caption_separator") or ",") + "\n"
        return text

    def native_config_text(self, config, dataset_config):
        # Preview, write and export share this one renderer so the shown text, the
        # written file and the exported file can never diverge.
        return self._render_train(config, Path(dataset_config).parent if dataset_config else Path("."))

    def export_parameters(self, config):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 Anima LoRA 适配器支持")
        notice = (
            "此文件只包含训练参数；Anima 运行时还需同目录下的 dataset.toml"
            "（通过 --dataset_config 提供），并确认底模、Qwen3 与 VAE 已就绪。"
        )
        text = self.native_config_text(config, None)
        if exported_prompt_files(config):
            notice += " 启用采样时会同时导出 sample_prompts.txt 配套文件。"
        return "# " + notice + "\n" + text, (notice,)

    def export_parameter_files(self, config):
        return exported_prompt_files(config)

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text

        job_directory = Path(job_directory)
        target = job_directory / "config.native.toml"
        write_prompts(config, job_directory)
        if config.get("architecture") == "moss_image":
            atomic_text(target, self.native_config_text(config, None))
            atomic_text(job_directory / "stages.json", json.dumps(
                [list(stage) for stage in self.preview_commands(installation, config, target)],
                ensure_ascii=False, indent=2,
            ))
            return target
        atomic_text(job_directory / "dataset.toml", self._dataset_toml(config))
        atomic_text(target, self.native_config_text(config, job_directory / "dataset.toml"))
        atomic_text(job_directory / "stages.json", json.dumps(
            [list(self.preview_argv(installation, config, target))], ensure_ascii=False, indent=2,
        ))
        return target

    # ---- launch ----

    def _values(self, config):
        params = config.get("params", {})
        return {p.key: params.get(p.key, p.default) for p in PARAMS}

    def _cli(self, values, keys):
        args = []
        for key in keys:
            value = values.get(key)
            if value is None or value is False or (isinstance(value, str) and value == ""):
                continue
            p = self._param(key)
            name = (p.native_key or p.key) if p else key
            if isinstance(value, bool):
                if value:
                    args.append("--" + name)
                continue
            args.extend(("--" + name, str(value)))
        return tuple(args)

    def _cache_directory(self, config):
        values = self._values(config)
        custom = values.get("moss_cache_directory")
        if isinstance(custom, str) and custom.strip():
            return Path(custom)
        return Path(config["output_dir"]) / "moss-cache"

    def _moss_model_dir(self, config):
        """The engine wants the folder that *contains* MOSS-Image-Base/.

        The form asks for the variant folder itself (the only variant this version
        trains), so its parent is passed to --model_dir and the variant is fixed.
        """
        chosen = self._values(config).get("moss_model_dir")
        if isinstance(chosen, str) and chosen.strip():
            return Path(chosen.strip()).parent
        return None

    def preview_commands(self, installation, config, native_config):
        python = str(installation.python_executable)
        source = str(installation.source_path)
        if config.get("architecture") == "moss_image":
            values = self._values(config)
            cache_dir = self._cache_directory(config)
            model_dir = self._moss_model_dir(config)
            model_args = ("--model_dir", str(model_dir)) if model_dir else ()
            # Stage 1 encodes the dataset: the cache request owns the resolution cap,
            # the caption format and the model directory.
            cache = (
                python, "-u", str(Path(source) / "scripts" / "moss" / "cache.py"),
                "--src", str(config["dataset_path"]), "--out", str(cache_dir),
            ) + self._cli(values, ("moss_resolution", "moss_caption_format")) + model_args
            # Stage 2 trains from the cache. --output is a FILE path, not a folder;
            # this version only ships the base variant.
            output_file = Path(config["output_dir"]) / (self._output_name(config) + ".safetensors")
            train = [
                python, "-u", str(Path(source) / "scripts" / "moss" / "train.py"),
                "--cache", str(cache_dir),
                "--output", str(output_file),
                "--model", "base",
            ]
            train += self._cli(values, (
                "moss_algo", "moss_epochs", "moss_max_steps", "moss_seed",
                "moss_rank", "moss_alpha", "moss_factor", "moss_down_init",
                "moss_channel_scaling_alpha", "moss_channel_stats", "moss_channel_mean",
                "moss_lora_dtype", "moss_lr", "moss_warmup_ratio",
                "moss_max_grad_norm", "moss_save_every_epochs", "moss_blocks_to_swap",
                "moss_activation_memory_budget", "moss_grad_checkpointing",
                "moss_log_every", "moss_caption_dropout", "moss_plain_caption_rate",
                "moss_caption_variant_rate", "moss_ref_timestep", "moss_repa_weight",
                "moss_targets", "moss_sigma_mu", "moss_sigma_s", "moss_sigma_cutoff",
                "moss_sigma_uniform",
            ))
            # dc_isolate defaults to ON in the engine, so turning it off needs the
            # explicit --no- form; omitting the flag would silently keep it enabled.
            train.append("--dc_isolate" if values.get("moss_dc_isolate") else "--no-dc_isolate")
            # The engine defaults compile to ON, so the switch is always explicit:
            # without a C++ compiler (cl.exe) Inductor aborts with InvalidCxxCompiler.
            train.append("--compile" if values.get("moss_compile") else "--no-compile")
            return (tuple(cache), tuple(train) + model_args)
        train = (
            python, "-u", str(Path(source) / "train.py"),
            "--config_file", str(native_config),
            "--dataset_config", str(Path(native_config).parent / "dataset.toml"),
        )
        return (train,)

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
        # torch.compile shells out to cl.exe; give the training process the same
        # INCLUDE/LIB/PATH a Developer prompt would. Only when compile is requested.
        # job.json is written after this call, so a missing record falls back to the
        # native flag list that write_native_config already produced.
        try:
            values = (files.read_job() or {}).get("params", {})
        except (FileNotFoundError, ValueError):
            values = {}
        compile_requested = values.get("moss_compile")
        if compile_requested is None:
            try:
                text = native_config.read_text(encoding="utf-8")
            except OSError:
                text = ""
            compile_requested = any(line.strip() == "--compile" for line in text.splitlines())
        if compile_requested:
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
    _MOSS_PROGRESS = re.compile(r"(?:^|\s)(\d+)\s*/\s*(\d+)")
    _MOSS_LOSS = re.compile(r"\bloss\s+([0-9.]+)")
    _MOSS_STEP_SECONDS = re.compile(r"([0-9.]+)\s*s/step")

    def parse_log(self, line):
        events = list(parse_progress(line))
        if events:
            return tuple(events)
        text = line.strip()
        progress = self._MOSS_PROGRESS.search(text)
        if progress and (text.startswith("step ") or text.startswith("progress ")
                         or " step " in " " + text or " progress " in " " + text):
            payload: dict[str, object] = {"step": int(progress[1]), "total_steps": int(progress[2])}
            loss = self._MOSS_LOSS.search(text)
            if loss:
                value = float(loss[1])
                if math.isfinite(value):
                    payload["loss"] = value
            # The unit here is seconds per step, the inverse of the shared it/s value.
            seconds = self._MOSS_STEP_SECONDS.search(text)
            if seconds:
                value = float(seconds[1])
                if value > 0:
                    payload["it_per_sec"] = 1 / value
            return (TrainingEvent("progress", payload),)
        # --progress_jsonl records one object per update; only keys that are actually
        # present are read, so an unknown schema yields no event instead of a guess.
        if not text.startswith("{"):
            return ()
        try:
            record = json.loads(text)
        except (ValueError, TypeError):
            return ()
        if not isinstance(record, dict):
            return ()
        payload = {}
        step = record.get("step")
        if isinstance(step, int):
            payload["step"] = step
        loss = record.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            payload["loss"] = float(loss)
        return (TrainingEvent("progress", payload),) if payload else ()

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities(graceful_stop=True, resume_weights=True, resume_full_state=True)

    # A run keeps its caches inside the task folder (<output>/moss-cache, <output>/cache)
    # and those files are .safetensors too. They are inputs, not results, so they must
    # never be listed as checkpoints.
    _CACHE_DIRS = {"cache", "moss-cache", "latent-cache", "te-cache"}

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
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part.lower() in self._CACHE_DIRS for part in path.relative_to(root).parts[:-1]):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                found.append(Artifact(path, "checkpoint", True))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
