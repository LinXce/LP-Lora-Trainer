"""Musubi Tuner adapter: caching pipeline + LoRA training.

Musubi requires pre-computed latent and Text-Encoder caches, so a job runs three
stages in order (latents -> text encoder -> train). No engine imports here.
"""
import json
import re
from pathlib import Path

from adapters.base import (
    InstallationPlan, EngineAdapter, DetectionResult, ValidationIssue, RecoveryCapabilities, Artifact,
)
from adapters.capabilities import check_value, parse_progress, specs
from adapters.musubi_tuner.params import ARCHITECTURES, NETWORK_MODULES, PARAMS
from adapters.musubi_tuner.pipeline import Pipeline, pipeline_for
from app.schemas.training import LaunchSpec


class MusubiTunerAdapter(EngineAdapter):
    engine_id = "musubi_tuner"
    adapter_version = "0.1.0"
    environment_strategy = "uv_project"
    environment_dir = ".venv"
    environment_candidates = (".venv", "venv")
    python_version = "3.10"
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

    def installation_plan(self, source_path, python, torch_source, uv=None):
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

        command = [uv, "sync"]
        if extra:
            command.extend(("--extra", extra))
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
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable)

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
        return tuple(issues)

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

    def _render_train(self, config):
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
            values[p.native_key or p.key] = (value, p.list_arg)
        # fp8_scaled requires fp8_base in the trainer's own validation.
        if values.get("fp8_scaled", (False, False))[0] and "fp8_base" not in values:
            values["fp8_base"] = (True, False)
        values["dit"] = (config["base_model_path"], False)
        values["network_module"] = (pipeline.module if pipeline else NETWORK_MODULES.get(architecture, "networks.lora"), False)
        values["output_dir"] = (config["output_dir"], False)
        values["output_name"] = ("lora", False)
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
            # Musubi builds one bucket per listed size; this is its equivalent of
            # bucket reso limits (there is no --min/max_bucket_reso here).
            text += "resolution = [" + ", ".join(f"[{w}, {h}]" for w, h in resolutions) + "]\n"
        else:
            text += f"resolution = [{width}, {height}]\n"
        text += "caption_extension = " + json.dumps(params.get("caption_extension") or ".txt") + "\n"
        text += f"batch_size = {int(params.get('batch_size') or 1)}\n"
        text += f"enable_bucket = {str(bool(params.get('enable_bucket', False))).lower()}\n"
        text += f"bucket_no_upscale = {str(bool(params.get('bucket_no_upscale', False))).lower()}\n\n"
        text += "[[datasets]]\n"
        text += "image_directory = " + json.dumps(config["dataset_path"], ensure_ascii=False) + "\n"
        text += "cache_directory = " + json.dumps(str(cache_dir), ensure_ascii=False) + "\n"
        text += f"num_repeats = {int(params.get('num_repeats') or 1)}\n"
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

        latents = (python, "-u", str(self._script(source, pipeline.latents)), "--dataset_config", dataset,
                   "--skip_existing", *self._stage_args(pipeline.latents_args, values))
        text_encoder = (python, "-u", str(self._script(source, pipeline.text_encoder)), "--dataset_config", dataset,
                        "--skip_existing", *self._stage_args(pipeline.te_args, values))
        train = [python, "-u", str(self._script(source, pipeline.script)), "--config_file", native,
                 "--dataset_config", dataset]
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

    def render(self, config, dataset_config):
        # Preview text is the training config; the dataset config is written separately.
        return self._render_train(config)

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        job_directory = Path(job_directory)
        cache_dir = self._cache_dir(config, job_directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(job_directory / "dataset.toml", self._dataset_toml(config, cache_dir))
        target = job_directory / "config.native.toml"
        atomic_text(target, self._render_train(config))
        stages = self._stages(installation, config, job_directory)
        atomic_text(job_directory / "stages.json", json.dumps([list(s) for s in stages], ensure_ascii=False, indent=2))
        return target

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        stages_path = native_config.parent / "stages.json"
        if not stages_path.is_file():
            raise RuntimeError("缺少 musubi 流水线定义 stages.json")
        commands = tuple(tuple(stage) for stage in json.loads(stages_path.read_text(encoding="utf-8")))
        if not commands:
            raise RuntimeError("musubi 流水线为空")
        return LaunchSpec(commands[-1], Path(installation.source_path),
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}, commands[:-1])

    def parse_log(self, line):
        return parse_progress(line)

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities()

    def collect_artifacts(self, job_directory):
        config = json.loads((Path(job_directory) / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        found = []
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
