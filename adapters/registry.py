from pathlib import Path
from app.schemas.engine import EngineInstallation, EngineRevision, ManagementMode, InstallationState, VerificationState
from adapters.kohya.adapter import KohyaAdapter
from adapters.ai_toolkit.adapter import AiToolkitAdapter
from adapters.musubi_tuner.adapter import MusubiTunerAdapter

ADAPTERS = {a.engine_id: a for a in (KohyaAdapter(), AiToolkitAdapter(), MusubiTunerAdapter())}


def domain(record):
    return EngineInstallation(
        installation_id=record["installation_id"], engine_id=record["engine_id"], source_path=Path(record["source_path"]),
        management_mode=ManagementMode(record["management_mode"]), revision=EngineRevision(**record["revision"]),
        state=InstallationState(record["state"]), verification=VerificationState(record["verification"]),
        environment_id=record["environment_id"], python_executable=Path(record["python_executable"]) if record["python_executable"] else None)
