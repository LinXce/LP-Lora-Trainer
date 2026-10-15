"""Conservative, single-GPU sd-scripts LoRA adapter. No engine imports here."""
import json
import re
from pathlib import Path
from adapters.base import InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact
from adapters.capabilities import check_value, parse_progress, specs
from adapters.kohya.params import ARCHITECTURES, NETWORK_MODULES, PARAMS
from adapters import extra_params
from adapters.sampling import prompt_file, write_prompts, exported_prompt_files
from app.schemas.training import LaunchSpec


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


class KohyaAdapter(EngineAdapter):
    engine_id = "kohya"
    # sd-scripts creates the output folder itself when it first saves
    # (os.makedirs(..., exist_ok=True)), so LP must not pre-create it: doing that
    # with exist_ok=False failed with WinError 183 on a re-run of the same name.
    creates_output_dir = True
    display_name = "Kohya"
    installation_help = "由项目内 uv 准备独立 Python，再调用官方 setup/setup_windows.py --headless；官方安装器初始化 sd-scripts 子模块并选择 CUDA 依赖。"
    default_installation_source = "cu124"
    adapter_version = "0.1.0"
    native_format = "toml"
    submittable = True
    environment_strategy = "precreate_venv"
    environment_dir = "venv"
    environment_candidates = ("venv",)
    python_version = "3.11"
    torch_index_mode = "env"
    installation_source_help = "由官方 setup_windows.py 自选 CUDA；PyTorch/CUDA wheel 镜像仅作为附加 pip 索引注入，官方安装器若显式指定 index 可能不生效。"

    def install_blockers(self, hardware):
        import os
        if os.name != "nt":
            return ("Kohya 官方安装器（setup/setup_windows.py）仅支持 Windows",)
        return ()

    def installation_plan(self, source_path, python, torch_source, uv=None, torch_mirror_url=None):
        """Install the declared requirements with LP's own command.

        The official installer cannot be driven safely: its command line is
        fixed, it re-emits pip's output through a pipe it only drains on stdout,
        and it resolves against three indexes (including its own
        download.pytorch.org), which is what made it sit silent for minutes.
        Everything it did in the headless flow is reproduced here from the
        requirements it declares, so LP keeps control of mirrors and output.
        """
        import os
        if os.name != "nt":
            raise ValueError("当前 Kohya 安装接口仅支持 Windows")
        entries = list(self.requirement_entries(source_path))
        if not entries:
            raise ValueError("缺少 Kohya requirements.txt 或其中没有可安装的固定版本条目")
        # The engine pins torch/torchvision to a CUDA-tagged wheel that only its
        # own index serves; the selected mirror provides the same files.
        links = []
        if torch_mirror_url:
            for package in ("torch", "torchvision"):
                links += ["--find-links", f"{torch_mirror_url.rstrip('/')}/{package}/"]
        # setup_common.py imports pkg_resources, which setuptools 82 removed;
        # keep the pin so anything else importing it still works. bitsandbytes is
        # not in the requirements file, but LP defaults to AdamW8bit, which imports it.
        packages = ["setuptools<82", "bitsandbytes", *entries]
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
                                      "--no-build-isolation", *links, *packages),))
        return InstallationPlan("Kohya 依赖安装（由 LP 执行，pip 降级路径）",
                                ((python, "-u", "-m", "pip", "install", "--require-virtualenv",
                                  "--no-input", "--no-build-isolation", *links, *packages),))

    def detect(self, source_path):
        root = scripts_root(source_path)
        complete = (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file()
        # A plain clone can identify the wrapper without containing the training submodule.
        wrapper = (source_path / "kohya_gui.py").is_file() and (source_path / "kohya_gui").is_dir()
        declared = wrapper and self._declared_submodule(source_path) is not None
        return DetectionResult(self.engine_id, complete or (wrapper and declared),
                               "sd-scripts training source or Kohya wrapper with declared submodule")

    # Requirements files that pin the PyTorch wheel index this engine installs from.
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
        """Requirement files the headless installer installs from, in order."""
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

    def source_issues(self, source_path):
        root = scripts_root(source_path)
        if (root / "train_network.py").is_file() and (root / "library" / "train_util.py").is_file():
            return ()
        return ("Kohya 缺少完整 sd-scripts 训练源码；可在本页点击“补齐训练源码”自动初始化子模块，或手动运行 git submodule update --init --recursive 后刷新发现",)

    source_repair_label = "补齐 sd-scripts 训练源码"

    clone_url = "https://github.com/bmaltais/kohya_ss.git"
    # The upstream version this adapter is written against.
    clone_branch = "main"
    clone_commit = "45088f04af78e11cec5407ff4652ea3ed2c14422"

    @staticmethod
    def _declared_submodule(source_path):
        """Return ``(path, url)`` for the sd-scripts entry in ``.gitmodules``."""
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
        """Fetch the sd-scripts training source declared by .gitmodules.

        A real checkout restores the recorded submodule commit. A copied
        directory has no ``.git`` to submodule-update from, so it clones the
        declared URL instead — the same result as pasting in a complete copy.
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
        issues += self.generic_param_issues(values)
        issues += self.engine_param_issues(installation, config, values)
        issues += self.base_model_param_issues(installation, config, values)
        return tuple(issues)

    # ---- 通用参数校验：所有引擎、所有底模 ----
    def generic_param_issues(self, values):
        issues = []
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        for p in PARAMS:
            if not p.native and p.unsupported_reason:
                continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
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
        if architecture == "sdxl":
            values["network_train_unet_only"] = True
        # Supplementary arguments go into the same config file, so preview, export
        # and the real run all see them.
        values = extra_params.merge(values, str(params.get(extra_params.PARAM_KEY) or ""))
        return "\n".join(f"{k} = {str(v).lower() if isinstance(v, bool) else json.dumps(v, ensure_ascii=False)}" for k, v in values.items()) + "\n"

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
        values = tomllib.loads(native_config.read_text(encoding="utf-8"))
        script = MODULE_SCRIPTS.get(values.get("network_module"))
        if not script:
            script = "sdxl_train_network.py" if values.get("network_train_unet_only") else "train_network.py"
        root = scripts_root(installation.source_path)
        return LaunchSpec((str(installation.python_executable), "-u", str(root / script), "--config_file", str(native_config)), root,
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

    def parse_log(self, line):
        # tqdm's progress is explicit; no epoch/ETA guesses and no fabricated loss points.
        return parse_progress(line)

    def recovery_capabilities(self, installation):
        # Stop is process interruption, not a promise of checkpoint-on-stop or resumption.
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        config = json.loads((job_directory / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        found = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                # sd-scripts step saves are "{name}-{step:06d}"; final saves
                # ("lora.safetensors", "{name}.safetensors") carry no step.
                match = re.search(r"-(\d{6})$", path.stem)
                found.append(Artifact(path, "checkpoint", True, int(match[1]) if match else None))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
