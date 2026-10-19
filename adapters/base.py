"""Contract shared by automatically discovered engine adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from adapters.capabilities import ArchSpec
from app.schemas.engine import EngineInstallation, EngineRevision
from app.schemas.training import LaunchSpec, TrainingEvent

# Bumped when the adapter contract itself changes (a required method, a changed
# return shape). Adapters report it through ``adapter_version``; a mismatch is a
# plugin-load problem, not a training problem.
ADAPTER_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class DetectionResult:
    engine_id: str
    matched: bool
    reason: str
    revision: EngineRevision | None = None


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class RecoveryCapabilities:
    graceful_stop: bool = False
    resume_weights: bool = False
    resume_full_state: bool = False


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    complete: bool
    # Explicit training step when the adapter can determine it (else None; the
    # supervisor falls back to a filename heuristic).
    step: int | None = None


@dataclass(frozen=True)
class InstallationPlan:
    title: str
    commands: tuple[tuple[str, ...], ...]


class EngineAdapter(ABC):
    engine_id: str
    adapter_version: str
    display_name: str = ""
    installation_help: str = "由适配器调用该引擎自身的安装流程。"
    installation_source_help: str = "安装来源由当前引擎源码及适配器声明。"
    default_installation_source: str = "existing"
    # How a PyTorch/CUDA wheel mirror is applied:
    # - "named": the adapter consumes it as a named uv index in ``installation_plan``.
    # - "env": LP exports it as an extra PIP/UV index for the engine's installer.
    # - "none": unused.
    torch_index_mode: str = "none"
    # Conservative default: adapters must explicitly opt into recursive datasets.
    supports_nested_images: bool = False
    # True when the engine itself creates the task output folder (and tolerates it
    # already existing) at the moment it first writes a checkpoint or sample. The
    # supervisor then leaves the folder alone instead of pre-creating it.
    creates_output_dir: bool = False

    def validate_dataset(self, installation, dataset, images):
        """Engine-specific dataset constraints; override for other layouts."""
        if not self.supports_nested_images and any("/" in i["file_name"] or "\\" in i["file_name"] for i in images):
            return (ValidationIssue("dataset_id", f"当前 {self.display_name or self.engine_id} 数据集适配要求图片位于数据集目录一级，请分别登记子目录"),)
        return ()

    def preserve_issue(self, issue: str) -> bool:
        """Filter obsolete adapter notices without dropping real diagnostics."""
        return True
    training_notice: str | None = None
    # Short, adapter-owned badge label for the engine list (e.g. "正在适配").
    # Empty means the shell shows nothing, so other adapters need no change.
    adapter_status: str = ""
    # Preview format of the rendered native config, and whether the adapter can
    # actually queue a training job. Adapters that only describe parameters set
    # ``submittable = False`` and keep ``write_native_config`` unimplemented.
    native_format: str = "toml"
    submittable: bool = True

    def export_parameter_files(self, config: dict[str, object]) -> list[dict[str, str]]:
        """Optional companion text files for a pure, native parameter export."""
        return []

    def export_parameters(self, config: dict[str, object]) -> tuple[str, tuple[str, ...]]:
        """Pure parameter export; no engine imports, environment checks or file writes.

        A native profile may require external dataset/cache files. Adapters must
        explain those requirements and must not export preview job placeholders.
        """
        raise NotImplementedError("该适配器暂未提供原生参数导出")

    def architectures(self, source_path: Path) -> tuple[ArchSpec, ...]:
        """Base-model types this clone can train, each with its entry script."""
        return ()

    def preview_argv(
        self, installation: EngineInstallation, config: dict[str, object], native_config: Path
    ) -> tuple[str, ...]:
        """Literal argv shown before submit; must not require submit support."""
        raise NotImplementedError("该引擎尚未提供启动预览")

    def preview_commands(
        self, installation: EngineInstallation, config: dict[str, object], native_config: Path
    ) -> tuple[tuple[str, ...], ...]:
        """Ordered commands the supervisor runs; the last one is training."""
        return (self.preview_argv(installation, config, native_config),)

    def native_config_text(self, config: dict[str, object], dataset_config: Path | None) -> str:
        """Render the engine's native configuration as text (pure; no file writes).

        ``dataset_config`` is where the dataset config will live, or ``None`` for
        engines that embed the dataset in the native config. The submit preview,
        ``write_native_config`` and parameter export must all go through this one
        function, so the text shown, written and exported cannot diverge. It must
        not import or execute engine code.
        """
        raise NotImplementedError("该适配器暂未提供原生配置渲染")

    # Installation ownership is explicit per engine:
    # - ``native``: the copied engine's manager creates and syncs its environment.
    # - ``uv_project``: the engine's pyproject/uv workflow creates and syncs it.
    # - ``precreate_venv``: LP only creates the empty venv required by the
    #   engine's own installer, then delegates dependency installation to it.
    environment_strategy: str = "precreate_venv"
    environment_dir: str | None = None
    environment_candidates: tuple[str, ...] = ()
    python_version: str | None = None
    # Empty by default on purpose: an inherited list made a new engine report modules
    # it never uses as "missing". Each adapter declares exactly what it imports.
    diagnostic_modules: tuple[str, ...] = ()
    # Base-model types that train from a model directory instead of "a base model file"
    # (MOSS lines use WEIGHTS/model_dir). The UI hides the generic field for them.
    base_model_optional_architectures: tuple[str, ...] = ()
    # Optional dependency groups this engine declares (pip extras). LP installs one only
    # when the user ticks it; the label is what the install dialog shows.
    optional_extras: tuple[tuple[str, str], ...] = ()

    def installation_sources(self, source_path: Path) -> tuple[str, ...]:
        """CUDA extras declared by this particular clone, not directory labels."""
        return ()

    def install_blockers(self, hardware: dict[str, object]) -> tuple[str, ...]:
        """Reasons this machine cannot install this engine (OS, driver, CUDA…).

        Non-empty blocks the one-click install; the empty tuple allows it. The
        hardware dict is :func:`app.services.hardware.probe`'s result.
        """
        return ()

    def installation_plan(
        self,
        source_path: Path,
        python: str,
        torch_source: str,
        uv: str | None = None,
        torch_mirror_url: str | None = None,
    ) -> InstallationPlan:
        """Use this engine's own installer or documented dependency entrypoints."""
        raise ValueError("该引擎尚未提供安装接口")

    def source_issues(self, source_path: Path) -> tuple[str, ...]:
        """Report incomplete source trees without importing or running them."""
        return ()

    def torch_index_tag(self, source_path: Path) -> str | None:
        """CUDA tag (``cu128``) the engine's own requirements pin, if any.

        The engine decides which wheel index it installs from, so mirrors must
        be built for that tag rather than the driver's maximum.
        """
        return None

    def pinned_wheel_requirements(self, source_path: Path) -> tuple[str, ...]:
        """Exact requirements that only the engine's own wheel index can serve.

        LP may fetch these directly from a mirror before the engine's installer
        runs, so pip's resolver never has to search several large indexes for
        them (the case that makes it spin without transferring anything).
        """
        return ()

    def local_requirement_paths(self, source_path: Path) -> tuple[str, ...]:
        """Requirements pointing inside the engine tree (e.g. ``-e ./sd-scripts``).

        LP installs these itself with build isolation switched off, so the
        engine's installer finds them satisfied and never builds them.
        """
        return ()

    def requirement_files(self, source_path: Path) -> tuple[str, ...]:
        """Requirement files the engine's installer installs from, in order.

        LP may install them ahead of time so the engine's own run finds every
        entry already satisfied instead of resolving them all again.
        """
        return ()

    def requirement_entries(self, source_path: Path) -> tuple[str, ...]:
        """The pinned entries of :meth:`requirement_files`, without directives.

        Passing entries as arguments (rather than ``-r <file>``) keeps the
        engine's own ``--extra-index-url`` lines out of the resolution, which is
        what makes pip search several large indexes and go silent for minutes.
        """
        entries, seen = [], set()
        for name in self.requirement_files(source_path):
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                entry = line.split("#", 1)[0].strip()
                if not entry or entry.startswith("-") or "==" not in entry or entry in seen:
                    continue
                seen.add(entry)
                entries.append(entry)
        return tuple(entries)

    def requirement_skips(self, source_path: Path) -> tuple[str, ...]:
        """Requirement lines that are *not* pinned, and are therefore never installed.

        ``requirement_entries`` only keeps ``name==version`` lines: passing an unpinned
        entry to pip would let the resolver search several large indexes. A clone that
        switches to ``>=`` / extras / hashes would silently lose those packages, so
        adapters surface them as source issues instead.
        """
        skipped = []
        for name in self.requirement_files(source_path):
            try:
                text = (source_path / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                entry = line.split("#", 1)[0].strip()
                if not entry or entry.startswith("-") or "==" in entry:
                    continue
                skipped.append(f"{name}: {entry}")
        return tuple(skipped)

    # Label for the completion step, shown in the engine detail panel.
    source_repair_label = ""

    # Upstream reference a fresh `git clone` should land on: the URL the UI can
    # offer as a preset, plus the branch and commit LP is developed against.
    # Empty strings mean "the repository default branch, no commit pin".
    clone_url = ""
    clone_branch = ""
    clone_commit = ""

    def source_repair(self, source_path: Path) -> tuple[str, ...] | None:
        """argv that completes a partially fetched source tree, or None.

        LP runs it inside the engine directory with argv[0] resolved from PATH
        and refreshes detection afterwards. Return None when nothing is missing
        or when the adapter cannot complete the tree automatically.
        """
        return None

    def update_commands(self, source_path: Path) -> tuple[tuple[str, ...], ...]:
        """argv steps that bring this clone to its upstream reference, or empty.

        LP checks first that the working tree is clean, then runs these in order and
        reports the new revision. ``--ff-only`` is deliberate: an update must never
        rewrite local history or discard local changes. Adapters whose engine has no
        upstream (or is bundled) return an empty tuple, which the UI shows as
        "不可更新".
        """
        if not self.clone_url:
            return ()
        # A copied tree (no .git) has no upstream to merge from, even when the adapter
        # knows the repository URL.
        if not (Path(source_path) / ".git").is_dir():
            return ()
        branch = self.clone_branch or "HEAD"
        # FETCH_HEAD always exists after that fetch, unlike origin/<branch> in a
        # shallow or hand-initialised clone. Submodule recursion is off: an update must
        # never fetch or check out submodules (sd-scripts carries its own environment).
        quiet = ("-c", "submodule.recurse=false")
        return (
            ("git", *quiet, "fetch", "--prune", "origin", branch),
            ("git", *quiet, "merge", "--ff-only", "FETCH_HEAD"),
        )

    @abstractmethod
    def detect(self, source_path: Path) -> DetectionResult:
        """Static discovery only; must not execute engine code."""

    @abstractmethod
    def capabilities(self, installation: EngineInstallation) -> dict[str, object]:
        """Describe supported architectures, parameters and version constraints."""

    @abstractmethod
    def validate(
        self, installation: EngineInstallation, config: dict[str, object]
    ) -> tuple[ValidationIssue, ...]:
        """Reject unsupported configurations before queue submission."""

    @abstractmethod
    def write_native_config(
        self,
        installation: EngineInstallation,
        config: dict[str, object],
        job_directory: Path,
    ) -> Path:
        """Write native config into the job directory, not the engine source."""

    def caption_cache_notices(self) -> tuple[tuple[str, str], ...]:
        """``(parameter, message)`` for caches that outlive one job and hold caption output.

        Reported as a warning once the dataset changed after its last scan: those
        caches are reused by file existence, so a caption edit alone does not
        invalidate them. Adapters without such a cache return nothing.
        """
        return ()

    @abstractmethod
    def build_launch(
        self, installation: EngineInstallation, native_config: Path
    ) -> LaunchSpec:
        """Build explicit Python/CLI argv and cwd without using a shell."""

    @abstractmethod
    def parse_log(self, line: str) -> tuple[TrainingEvent, ...]:
        """Unrecognized output remains in the raw log; do not invent metrics."""

    @abstractmethod
    def recovery_capabilities(
        self, installation: EngineInstallation
    ) -> RecoveryCapabilities:
        """Declare stop and recovery semantics separately."""

    @abstractmethod
    def collect_artifacts(self, job_directory: Path) -> tuple[Artifact, ...]:
        """Identify artifacts; incomplete writes must not be presented as ready."""

    def finalize_artifacts(self, job_directory: Path, python: str, source_path: Path) -> tuple[str, ...]:
        """Post-run work on a successful job (optional); returns user-facing notices.

        Called once after the last stage exits successfully, with the engine's
        interpreter and source directory so adapters can run their own tooling (format
        conversion, preview rendering). Returning text never fails the run: the trained
        result already exists.
        """
        return ()
