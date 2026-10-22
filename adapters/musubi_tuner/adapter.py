"""Musubi Tuner adapter: caching pipeline + LoRA training.

Musubi requires pre-computed latent and Text-Encoder caches, so a job runs three
stages in order (latents -> text encoder -> train). No engine imports here.
"""
from adapters.sampling import prompt_file, write_prompts, exported_prompt_files
import json
import re
from dataclasses import replace
from pathlib import Path

from adapters.base import (
    InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact,
    ResumeSource,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.recovery import accelerate_resume_sources
from adapters.musubi_tuner.params import ARCHITECTURES, NETWORK_MODULES, PARAMS
from adapters.musubi_tuner.pipeline import PIPELINES, Pipeline, pipeline_for
from adapters import extra_params
from app.schemas.training import LaunchSpec
from app.services.jobs import JobFiles


class MusubiTunerAdapter(EngineAdapter):
    engine_id = "musubi_tuner"
    display_name = "Musubi Tuner"
    clone_url = "https://github.com/kohya-ss/musubi-tuner.git"
    installation_source_help = "仅显示当前 pyproject.toml 声明的 CUDA extra；LP 原样调用官方 uv 项目流程。"
    installation_help = "使用官方 uv sync 创建并同步当前引擎的独立环境；不指定 extra 时按项目默认依赖同步，可能移除未声明的包。"
    default_installation_source = "existing"
    adapter_version = "0.1.0"
    environment_strategy = "embedded"
    environment_dir = ".lp/python"
    environment_candidates = (".venv", "venv")
    python_version = "3.10"
    torch_index_mode = "named"
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

        # Embedded environments have no venv to sync into, so the same dependency set
        # is installed straight from the copied project into the copied interpreter.
        # `-e <src>[<extra>]` is the pip-level equivalent of `uv sync --extra`.
        if getattr(self, "embedded_mode", False):
            project = f"{source_path.resolve()}" + (f"[{extra}]" if extra else "")
            command = [uv, "pip", "install", "--python", python, "-e", project]
            if torch_mirror_url:
                command.extend(("--find-links", torch_mirror_url.rstrip("/")))
            return InstallationPlan("Musubi Tuner 依赖安装（embedded：直接装入引擎解释器）",
                                    (tuple(command),))

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
        available = tuple(a.id for a in self.architectures(installation.source_path))
        # Aggregate by parameter, not by pipeline: shared keys such as VAE
        # are required by several architectures, and later pipelines must not
        # overwrite the requirements of earlier ones.
        required_for = {
            p.key: tuple(
                architecture for architecture in available
                if architecture in PIPELINES and p.key in PIPELINES[architecture].required
            )
            for p in PARAMS
        }
        params = tuple(
            replace(p, required_for=required_for.get(p.key, p.required_for))
            for p in PARAMS
        )
        return dict(installation_id=installation.installation_id,
                    architectures=list(available),
                    params=specs(params), submittable=self.submittable)

    @staticmethod
    def _param(key):
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
        for p in PARAMS:
            if not p.native and p.unsupported_reason:
                continue
            message = check_value(p, values.get(p.key, p.default))
            if message:
                issues.append(ValidationIssue(p.key, message))
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
                # Honour the declared default (e.g. kandinsky_task/h3_task); a
                # default of "" still counts as missing. Type checking above
                # already falls back to defaults, so this must too.
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
        for severity, message in extra_params.problems(
            str(values.get(extra_params.PARAM_KEY) or ""),
            extra_params.provided_keys(
                PARAMS, architecture, values, skip_false=True,
                extra=("dit", "vae", "dataset_config", "output_dir", "output_name",
                       "network_module", "config_file"),
            ),
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
        return json.dumps(str(value), ensure_ascii=False)

    def _render_train(self, config, prompt_directory=Path(".")):
        architecture = config.get("architecture")
        pipeline = pipeline_for(architecture)
        params = config.get("params", {})
        values = {}
        for p in PARAMS:
            if not p.native or not p.applies_to(architecture):
                continue
            value = params.get(p.key, p.default)
            if value is None or (isinstance(value, str) and value == "") or value is False:
                continue
            if p.key == "sample_prompts":
                value = prompt_file(config, prompt_directory)
                if not value:
                    continue
            values[p.native_key or p.key] = (value, p.list_arg)
        # fp8_scaled requires fp8_base in the trainer's own validation.
        if values.get("fp8_scaled", (False, False))[0] and "fp8_base" not in values:
            values["fp8_base"] = (True, False)
        values["dit"] = (config["base_model_path"], False)
        values["network_module"] = (pipeline.module if pipeline else NETWORK_MODULES.get(architecture, "networks.lora"), False)
        values["output_dir"] = (config["output_dir"], False)
        values["output_name"] = (config.get("output_name") or config.get("name") or "lora", False)
        # Supplementary arguments land in the same config file, so preview, export
        # and the real run all see them.
        for name, value in extra_params.merge({}, str(params.get(extra_params.PARAM_KEY) or "")).items():
            values[name] = (value, False)
        return "".join(f"{key} = {self._toml_value(value, list_arg)}\n" for key, (value, list_arg) in values.items())

    def _cache_dir(self, config, job_directory):
        custom = config.get("params", {}).get("cache_directory")
        if isinstance(custom, str) and custom.strip():
            return Path(custom)
        return Path(job_directory) / "cache" / str(config.get("architecture"))

    def _dataset_toml(self, config, cache_dir):
        params = config.get("params", {})
        width = int(params.get("resolution_width") or 960)
        height = int(params.get("resolution_height") or 544)
        resolutions = self.parse_resolutions(params.get("bucket_resolutions"))
        text = "[general]\n"
        if resolutions:
            # Musubi accepts one scalar or one [width, height] pair per dataset.
            # Represent multiple sizes as separate dataset entries, never as
            # a nested general.resolution array.
            width, height = resolutions[0]
        text += f"resolution = [{width}, {height}]\n"
        text += "caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += f"batch_size = {int(params.get('batch_size') or 1)}\n"
        text += f"enable_bucket = {str(bool(params.get('enable_bucket', False))).lower()}\n"
        text += f"bucket_no_upscale = {str(bool(params.get('bucket_no_upscale', False))).lower()}\n\n"
        for dataset_width, dataset_height in resolutions or [(width, height)]:
            # Sharing caches across resolutions would mix latent shapes and
            # make training consume files from the wrong dataset.
            dataset_cache = Path(cache_dir)
            if len(resolutions) > 1:
                dataset_cache /= f"{dataset_width}x{dataset_height}"
            text += "[[datasets]]\n"
            text += f"resolution = [{dataset_width}, {dataset_height}]\n"
            text += "image_directory = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
            text += "cache_directory = " + json.dumps(str(dataset_cache), ensure_ascii=False) + "\n"
            text += f"num_repeats = {int(params.get('num_repeats') or 1)}\n\n"
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
        params = config.get("params", {})
        values = {p.key: params.get(p.key, p.default) for p in PARAMS}
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
        output_name = config.get("output_name") or config.get("name") or "lora"
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

    def collect_artifacts(self, job_directory):
        config = JobFiles(job_directory).read_job()
        root = Path(config["output_dir"])
        found = []
        if not root.is_dir():
            # Musubi creates the folder on its first save; before that there is nothing to collect.
            return ()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                # musubi step files are "{name}-step{step:08d}"; epoch files use "{name}-{epoch:06d}".
                match = re.search(r"-step(\d+)", path.stem)
                found.append(Artifact(path, "checkpoint", True, int(match[1]) if match else None))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                found.append(Artifact(path, "sample", True))
        return tuple(found)
