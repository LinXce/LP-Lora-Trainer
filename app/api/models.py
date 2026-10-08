from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SettingsInput(Input):
    engine_root: str
    data_root: str
    comfyui_lora_dir: str | None = None
    gpu_monitor: bool = False
    log_tail_lines: int = Field(default=500, ge=10, le=10000)


class DatasetInput(Input):
    path: str
    name: str = Field(min_length=1, max_length=200)


class CaptionInput(Input):
    caption: str = Field(max_length=100000)


class PythonInput(Input):
    python_executable: str


class InstallEnvironmentInput(Input):
    confirmed: Literal[True]
    python_executable: str | None = None
    torch_source: str = Field(default="cu124", pattern=r"^(cu\d{3}|existing)$")
    # Optional PyPI-compatible index used by the engine's own installer.
    # None keeps the package manager's official default.
    mirror_url: str | None = Field(default=None, max_length=500)


class EngineTypeInput(Input):
    engine_id: Literal["kohya", "ai_toolkit", "musubi_tuner"]


class TrainingInput(Input):
    name: str = Field(max_length=200)
    installation_id: str
    architecture: str
    base_model_path: str
    dataset_id: str
    output_dir: str
    params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


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
    installation_id: str = Field(default="", max_length=500)
    architecture: str = Field(default="", max_length=200)
    base_model_path: str = Field(default="", max_length=4096)
    dataset_id: str = Field(default="", max_length=500)
    output_dir: str = Field(default="", max_length=4096)
    params: dict[str, Any] = Field(default_factory=dict)
    params_by_architecture: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StopInput(Input):
    force: bool = False


class AcknowledgeInput(Input):
    confirmed_exited: Literal[True]


class PublishInput(Input):
    target_dir: str
    file_name: str = Field(min_length=1, max_length=200)
