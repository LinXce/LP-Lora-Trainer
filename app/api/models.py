from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SettingsInput(Input):
    engine_root: str
    data_root: str
    comfyui_lora_dir: str | None = None
    gpu_monitor: bool = True
    log_tail_lines: int = Field(default=500, ge=10, le=10000)
    # Optional HTTP(S) proxy for engine downloads. Credentials are allowed
    # (http://user:pass@host:port) and never written to logs or sessions.
    proxy_url: str | None = Field(default=None, max_length=500)
    proxy_enabled: bool = False


class PromptTemplateInput(Input):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=100000)


class DatasetInput(Input):
    path: str
    name: str = Field(min_length=1, max_length=200)


class CaptionInput(Input):
    caption: str = Field(max_length=100000)
    # Natural-language caption (.caption). None leaves that file untouched.
    caption_nl: str | None = Field(default=None, max_length=100000)


class PythonInput(Input):
    python_executable: str


class TerminalInput(Input):
    """Open a console from browser mode (no pywebview native bridge)."""

    python: str = Field(min_length=1, max_length=32767)
    arguments: str | None = Field(default=None, max_length=2000)


class InstallEnvironmentInput(Input):
    confirmed: Literal[True]
    python_executable: str | None = None
    # Optional pip extras ticked in the dialog; validated against the adapter's
    # declared `optional_extras` before they reach a command line.
    extras: list[str] = Field(default_factory=list, max_length=8)
    torch_source: str = Field(default="cu124", max_length=64, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    # Optional PyPI-compatible index used by the engine's own installer.
    # None keeps the package manager's official default.
    mirror_url: str | None = Field(default=None, max_length=500)
    # Optional PyTorch/CUDA wheel index; kept separate from the ordinary PyPI index.
    torch_mirror_url: str | None = Field(default=None, max_length=500)
    # Optional base URL for uv's managed-Python downloads
    # (astral-sh/python-build-standalone releases); maps to
    # UV_PYTHON_INSTALL_MIRROR. Required when GitHub is unreachable and the
    # installer needs a Python version that is not installed locally yet.
    python_mirror_url: str | None = Field(default=None, max_length=500)
    # Per-install override; None follows the saved proxy setting.
    use_proxy: bool | None = None


class EngineCloneInput(Input):
    url: str = Field(min_length=1, max_length=500)
    # Folder under engine/; empty uses the repository name, like `git clone`.
    name: str | None = Field(default=None, max_length=100)
    # Optional branch or tag; empty clones the remote default branch.
    ref: str | None = Field(default=None, max_length=200)
    # Optional commit to check out after cloning, pinning the fetched version.
    commit: str | None = Field(default=None, max_length=64)
    # Per-pull override; None follows the saved proxy setting.
    use_proxy: bool | None = None


class EngineTypeInput(Input):
    engine_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")


class SourceRepairInput(Input):
    # Per-repair override; None follows the saved proxy setting.
    use_proxy: bool | None = None


class TrainingInput(Input):
    name: str = Field(max_length=200)
    output_name: str = Field(default="", max_length=200)
    installation_id: str
    architecture: str
    base_model_path: str
    dataset_id: str
    output_dir: str
    params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class TrainingExportInput(TrainingInput):
    engine_id: str = Field(min_length=1, max_length=100)
    dataset_path: str | None = Field(default=None, max_length=4096)


class TrainingDraftInput(Input):
    """A partially filled training form.

    Every field is optional on purpose: a draft is saved while the user is still
    filling it, so it must not be rejected for being incomplete. Parameter values
    are typed loosely here and converged by the service layer, which drops what an
    engine could not accept; rejecting the whole save would silently lose the rest
    of what the user typed. Real submission still goes through ``TrainingInput``
    and full validation.
    """
    name: str = Field(default="", max_length=200)
    output_name: str = Field(default="", max_length=200)
    installation_id: str = Field(default="", max_length=500)
    architecture: str = Field(default="", max_length=200)
    base_model_path: str = Field(default="", max_length=4096)
    dataset_id: str = Field(default="", max_length=500)
    output_dir: str = Field(default="", max_length=4096)
    params: dict[str, Any] = Field(default_factory=dict)
    params_by_architecture: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StopInput(Input):
    force: bool = False


class ResumeInput(Input):
    """One save point returned by ``GET /tasks/{id}/resume``; it is re-verified."""

    source: str = Field(min_length=1, max_length=4096)


class AcknowledgeInput(Input):
    confirmed_exited: Literal[True]


class PublishInput(Input):
    target_dir: str
    file_name: str = Field(min_length=1, max_length=200)
