from typing import Literal
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
    torch_source: Literal["cu124", "cu126", "cu128", "existing"] = "cu124"


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


class StopInput(Input):
    force: bool = False


class AcknowledgeInput(Input):
    confirmed_exited: Literal[True]


class PublishInput(Input):
    target_dir: str
    file_name: str = Field(min_length=1, max_length=200)
