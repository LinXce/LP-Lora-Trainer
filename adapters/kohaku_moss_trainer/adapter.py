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
import subprocess
import tomllib
from pathlib import Path

from adapters.base import (
    Artifact,
    DetectionResult,
    EngineAdapter,
    InstallationPlan,
    RecoveryCapabilities,
    ValidationIssue,
)
from adapters.capabilities import check_value, specs
from adapters.kohaku_moss_trainer.progress import parse_progress
from adapters.kohaku_moss_trainer.params import ARCHITECTURES, PARAMS
from app.schemas.training import LaunchSpec
from app.services.jobs import JobFiles

TRAIN_SCRIPT = "scripts/train/moss.py"
CACHE_STAGES = ("scripts/data/cache_text.py", "scripts/data/cache_latents.py")


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
    )
    default_installation_source = "cu130"
    environment_strategy = "precreate_venv"
    environment_dir = ".venv"
    environment_candidates = (".venv", "venv")
    python_version = "3.12"
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
        if tag:
            stages.append(tuple([python, "-u", "-m", "pip", "install", "--require-virtualenv",
                                 "--no-input", "torch>=2.13,<2.15", "torchvision",
                                 "--index-url", index]))
        package = [python, "-u", "-m", "pip", "install", "--require-virtualenv", "--no-input",
                   "--no-build-isolation"]
        if index:
            # pip has no --index-strategy (that is a uv flag); the extra index is what
            # keeps the already installed cu… torch build.
            package += ["--extra-index-url", index]
        package += ["-e", str(source), "torch>=2.13,<2.15"]
        stages.append(tuple(package))
        return InstallationPlan("KohakuMossTrainer：torch + 可编辑安装（kogine 随 kohaku-engine）",
                                tuple(stages))

    def source_issues(self, source_path):
        source = Path(source_path)
        issues = []
        if not (source / TRAIN_SCRIPT).is_file():
            issues.append("缺少 scripts/train/moss.py 训练入口")
        return tuple(issues)

    # ---- parameters / validation ----

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable,
                    base_model_optional_architectures=list(self.base_model_optional_architectures))

    @staticmethod
    def _memory_and_sample_issues(values):
        """Constraints the engine enforces at run time, checked at submit time.

        The engine rejects a profile combined with manual plan knobs, wants one
        seed per sample prompt, requires positive sampling steps/shifts, and needs
        image sizes that are multiples of its 16-pixel latent grid.
        """
        issues = []
        profile = str(values.get("MEMORY_PROFILE") or "").strip()
        if profile and profile != "manual":
            blocked = [key for key in ("CHECKPOINT", "STREAM", "STREAM_RESIDENT", "PINNED",
                                       "ACT_OFFLOAD", "OPTIM_STATE", "GRAD_OFFLOAD",
                                       "FUSED_STEP", "MEMORY_OVERRIDES")
                       if str(values.get(key) or "").strip() not in ("", "{}", "False", "0")]
            if blocked:
                # The engine ignores these under a profile, so this is a hint, not a
                # reason to refuse submission (a warning is never blocking).
                issues.append(ValidationIssue(
                    "MEMORY_PROFILE",
                    f"显存档位为 {profile} 时这些手动项不会生效：{', '.join(blocked)}；"
                    "要使用它们请把档位改成 manual，否则可以留空",
                    "warning"))
        try:
            prompts = ast.literal_eval(str(values.get("SAMPLE_PROMPTS") or "[]"))
            seeds = ast.literal_eval(str(values.get("SAMPLE_SEEDS") or "(0,)"))
        except (ValueError, SyntaxError):
            prompts, seeds = None, None
        if isinstance(prompts, (list, tuple)) and isinstance(seeds, (list, tuple)) and prompts:
            if len(seeds) != len(prompts):
                issues.append(ValidationIssue(
                    "SAMPLE_SEEDS", f"引擎要求每个提示词一个种子：{len(prompts)} 个提示词，{len(seeds)} 个种子"))
        for key in ("SAMPLE_STEPS", "SAMPLE_SHIFT"):
            try:
                if float(values.get(key)) <= 0:
                    issues.append(ValidationIssue(key, "必须为正数"))
            except (TypeError, ValueError):
                pass
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
        return issues

    def validate(self, installation, config):
        issues = []
        architecture = config.get("architecture")
        if architecture not in {a.id for a in self.architectures(installation.source_path)}:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        # The engine derives the text encoder / VAE from WEIGHTS, so an empty WEIGHTS is
        # unusable even though an empty string is a valid type: refuse it at submit time
        # instead of failing later inside cache_text.py.
        weights = str(values.get("WEIGHTS") or "").strip()
        if not weights:
            issues.append(ValidationIssue("WEIGHTS", "该引擎需要 MOSS-Image 模型目录（含 transformer/、text_encoder/、vae/）"))
        else:
            root = Path(weights)
            if not root.is_dir():
                issues.append(ValidationIssue("WEIGHTS", f"模型目录不存在：{weights}"))
            elif not (root / "transformer").is_dir():
                issues.append(ValidationIssue("WEIGHTS", "模型目录缺少 transformer/ 子目录"))
        issues.extend(self._memory_and_sample_issues(values))
        for p in PARAMS:
            if not p.native:
                continue
            value = values.get(p.key, p.default)
            if p.key in ("ALGO_KWARGS", "DROPOUT", "BETAS", "SCHEDULE", "TIMESTEP_KWARGS",
                         "CAPTION", "DEVICES"):
                try:
                    json.loads(value if isinstance(value, str) and value.strip() else "null")
                except ValueError:
                    issues.append(ValidationIssue(p.key, f"{p.label} 需要合法的 JSON"))
                continue
            message = check_value(p, value)
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

    def _render(self, config):
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS if p.native}
        output = Path(config["output_dir"])
        lines = [
            '"""Generated by LP; a kogine config: module-level knobs."""',
            f"WEIGHTS = {self._literal(values.get('WEIGHTS') or '')}",
            f"VARIANT = {self._literal(values.get('VARIANT') or 'base')}",
            "DATASETS = " + repr([{
                "source": str(config.get("params", {}).get("SOURCE") or "kohya"),
                "path": str(config["dataset_path"]),
                "weight": 1.0,
                # sources/base.py:37-47 -- `.txt` goes to tags or to caption_nl.
                # TXT_AS is native=False (a DATASETS field), so read the raw params.
                "txt_as": str(config.get("params", {}).get("TXT_AS") or "tags"),
            }]),
            f"OUTPUT_DIR = {self._literal(str(output))}",
            f"RUN_NAME = {self._literal(config.get('output_name') or config.get('name') or 'moss')}",
        ]
        # Source: scripts/data/cache_text.py:46 -- the plan is TEXT_ENCODER_KWARGS["plan"]
        # (`text_encoder_kwargs` merges TEXT_ENCODE's compute fields into it), and
        # `pinned` lives on that plan (offload/encoders/text.py: plan.pinned). VAE is the
        # path; VAE_ENCODE is its plan, so a path there is ignored.
        weights = Path(str(values.get("WEIGHTS") or ""))
        if str(weights):
            # The text-encoder plan is passed as TEXT_ENCODER_KWARGS['plan'] on the cache
            # stage; VAE is the training script's path knob (IMAGE_ENCODER_KWARGS covers
            # the latent stage the same way).
            lines.append("VAE = " + self._literal(str(weights / "vae")))
        # The engine wants a tuple of indices, not a JSON string: DEVICES = (0,).
        # The engine refuses a profile together with the manual plan knobs: with a
        # profile they must stay None. Emitting both made every run invalid (and
        # PINNED=True pinned the whole text encoder, exhausting host page-locked
        # memory: cuMemHostAlloc -> CUDA_ERROR_OUT_OF_MEMORY at cache_text).
        profile = str(values.get("MEMORY_PROFILE") or "").strip()
        # These three are knobs of the cache scripts only; the trainer rejects unknown
        # keys, so they travel as `--set` on the cache stages instead (see preview_commands).
        skip = {"WEIGHTS", "VARIANT", "CAPTION", "TEXT_ENCODER_PLAN",
                "TEXT_ENCODER_KWARGS", "IMAGE_ENCODER_KWARGS", "IMAGE_ENCODER", "ENCODE_BATCH"}
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
            if key == "DATA_RESCAN":
                # Source: kohakumoss_data/scan/runner.py -- (False, True, "never").
                text = str(value).strip().lower()
                lines.append("DATA_RESCAN = " + ("'never'" if text == "never"
                                                  else ("True" if text == "true" else "False")))
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

        def text(key, default=""):
            return str(params.get(key) or default).strip()

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
                "keep_first": int(params.get("CAPTION_KEEP_FIRST") or 0),
                "tag_dropout": float(params.get("CAPTION_TAG_DROPOUT") or 0.0),
                "group_dropout": float(params.get("CAPTION_GROUP_DROPOUT") or 0.0),
                "shuffle": bool(params.get("CAPTION_SHUFFLE", True)),
                "underscores": bool(params.get("CAPTION_UNDERSCORES", True)),
                "unescape": bool(params.get("CAPTION_UNESCAPE", True)),
                **json_object("CAPTION_SEPARATORS"),
                **json_object("CAPTION_AFFIX"),
            })
        elif template == "nl":
            caption["fallback"] = json_object("CAPTION_FALLBACK") or {"template": "tags"}
        elif template == "mix":
            caption["p_nl"] = float(params.get("CAPTION_P_NL") or 0.5)
            caption["nl"] = {"template": "nl", "keep": keep}
            caption["tags"] = {
                "template": "tags",
                "keep": keep,
                "keep_first": int(params.get("CAPTION_KEEP_FIRST") or 0),
                "shuffle": bool(params.get("CAPTION_SHUFFLE", True)),
                "tag_dropout": float(params.get("CAPTION_TAG_DROPOUT") or 0.0),
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
        notice = "此文件是 kogine 的 Python 配置；运行时还需同目录的缓存阶段与 kogine 运行器。"
        return "# " + notice + "\n" + self._render(config), (notice,)

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

    def preview_commands(self, installation, config, native_config):
        python = installation.python_executable
        if not python:
            raise RuntimeError("该引擎尚未绑定 Python 解释器，无法启动 kogine")
        kogine = Path(python).parent / ("kogine.exe" if os.name == "nt" else "kogine")
        runner = str(kogine if kogine.is_file() else "kogine")
        source = Path(installation.source_path)
        config_path = str(native_config)
        # Cache-only knobs: the trainer's own check_script rejects them, so they are
        # passed with --set on the two preparation stages (cache_text.py:3-4 documents it).
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        weights = Path(str(values.get("WEIGHTS") or ""))
        plan_text = str(values.get("TEXT_ENCODER_PLAN") or "").strip() or "{}"
        text_plan = {}
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(plan_text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                text_plan = parsed
                break
        cache_sets = [
            "--set", "TEXT_ENCODER_KWARGS=" + repr({
                "path": str(weights / "text_encoder"), "device": "cuda", "plan": text_plan}),
            "--set", "IMAGE_ENCODER_KWARGS=" + repr({
                "path": str(weights / "vae"), "device": "cuda"}),
            # IMAGE_ENCODER exists only in cache_latents.py, never in the trainer.
            # kogine takes the value as-is: a repr() here would arrive as a quoted string.
            "--set", "IMAGE_ENCODER=" + str(values.get("IMAGE_ENCODER")
                                            or "kohakumoss.encoders.vae_encoder"),
            "--set", "ENCODE_BATCH=" + repr(int(values.get("ENCODE_BATCH") or 512)),
        ]
        stages = []
        for script in CACHE_STAGES:
            stages.append((runner, "run", str(source / script), "--config", config_path, *cache_sets))
        stages.append((runner, "run", str(source / TRAIN_SCRIPT), "--config", config_path))
        return tuple(stages)

    def preview_argv(self, installation, config, native_config):
        return self.preview_commands(installation, config, native_config)[-1]

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        stages_path = native_config.parent / "stages.json"
        if not stages_path.is_file():
            raise RuntimeError("缺少 KohakuMossTrainer 运行阶段定义 stages.json")
        commands = tuple(tuple(stage) for stage in json.loads(stages_path.read_text(encoding="utf-8")))
        if not commands:
            raise RuntimeError("KohakuMossTrainer 运行阶段为空")
        return LaunchSpec(commands[-1], Path(installation.source_path),
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}, commands[:-1])

    # ---- logs / artifacts ----

    def parse_log(self, line):
        # Its callbacks print `step N [loss=…]` and `steps A-B median X s/step`.
        return parse_progress(line)

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities(graceful_stop=True, resume_weights=True)

    @staticmethod
    def _export_formats(value):
        """The adapter-only export formats of a stored job value.

        jobs store the (str, int, bool) draft, so a tuple arrives as its literal text.
        """
        if isinstance(value, str):
            text = value.strip()
            for parse in (json.loads, ast.literal_eval):
                try:
                    parsed = parse(text)
                except (ValueError, SyntaxError):
                    continue
                value = parsed
                break
            else:
                value = (text,) if text else ()
        if isinstance(value, str):
            value = (value,)
        return tuple(str(item) for item in (value or ()) if str(item) in ("comfy", "lycoris"))

    def finalize_artifacts(self, job_directory, python, source_path):
        """Export every saved checkpoint as a usable LoRA.

        The engine exports only the final state (training/entry.py:335), so per-step
        usable files are produced here with the engine's own exporter
        (scripts/export/export.py, ADAPTER = a run checkpoint's trainable.safetensors).
        A failed export is a notice: the run itself stays successful.
        """
        config = JobFiles(job_directory).read_job()
        if config.get("architecture") != "moss_image":
            return ()
        # The switch defaults to on; off skips the whole pass (no export cost).
        if config.get("params", {}).get("EXPORT_READY") is False:
            return ()
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        weights = str(values.get("WEIGHTS") or "").strip()
        root = Path(config["output_dir"])
        # The engine nests OUTPUT_DIR/<RUN_NAME>/checkpoints, so search recursively.
        checkpoints = sorted(root.rglob("checkpoints/step_*/trainable.safetensors"))
        if not weights or not checkpoints:
            return ()
        python = str(python or "")
        if not python:
            return ("未找到引擎解释器，跳过逐检查点导出",)
        kogine = Path(python).parent / ("kogine.exe" if os.name == "nt" else "kogine")
        runner = str(kogine if kogine.is_file() else "kogine")
        script = str(Path(str(source_path)) / "scripts" / "export" / "export.py")
        formats = self._export_formats(values.get("EXPORT_FORMATS"))
        if not formats:
            return ()
        notices = []
        export_dir = root / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        name = config.get("output_name") or config.get("name") or root.name
        for checkpoint in checkpoints:
            step = checkpoint.parent.name.replace("step_", "step")
            for fmt in formats:
                target = export_dir / f"{name}-{step}-{fmt}.safetensors"
                if target.is_file():
                    continue
                command = [runner, "run", script, "--set", f"WEIGHTS={weights}",
                           "--set", f"ADAPTER={checkpoint}", "--set", f"OUT={target}",
                           "--set", f"FORMAT={fmt}"]
                try:
                    result = subprocess.run(command, cwd=str(source_path), capture_output=True,
                                            timeout=3600, text=True, encoding="utf-8", errors="replace")
                except (OSError, subprocess.SubprocessError) as exc:
                    notices.append(f"{step} 导出 {fmt} 失败：{exc}")
                    continue
                if result.returncode or not target.is_file():
                    tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:]
                    notices.append(f"{step} 导出 {fmt} 失败：{' '.join(tail)[:200]}")
                else:
                    notices.append(f"{step} 已导出可用 LoRA：{target.name}")
        return tuple(notices)

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
