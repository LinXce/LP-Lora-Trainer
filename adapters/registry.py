"""Discover trusted LP adapters; never import copied engine source trees."""
import importlib
import inspect
import logging
import re
from pathlib import Path

from adapters.base import ADAPTER_CONTRACT_VERSION, EngineAdapter
from app.schemas.engine import EngineInstallation, EngineRevision, ManagementMode, InstallationState, VerificationState

log = logging.getLogger(__name__)


def discover_adapters(root: Path | None = None, package: str = "adapters"):
    """One local concrete EngineAdapter in each <directory>/adapter.py.

    Loaded once at process startup. A broken plugin is isolated; duplicate ids
    disable all conflicting plugins rather than silently replacing an engine.
    ``root`` and ``package`` allow isolated plugin packages in regression tests.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    adapters, errors, owners = {}, [], {}
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        entry = directory / "adapter.py"
        if (not directory.is_dir() or directory.is_symlink() or entry.is_symlink()
                or not entry.is_file() or not re.fullmatch(r"[a-z][a-z0-9_]*", directory.name)):
            continue
        try:
            module = importlib.import_module(f"{package}.{directory.name}.adapter")
            classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass)
                       if cls.__module__ == module.__name__ and issubclass(cls, EngineAdapter)
                       and cls is not EngineAdapter and not inspect.isabstract(cls)]
            if len(classes) != 1:
                raise ValueError("adapter.py 必须定义且仅定义一个具体 EngineAdapter 子类")
            adapter = classes[0]()
            engine_id = getattr(adapter, "engine_id", "")
            if not isinstance(engine_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", engine_id):
                raise ValueError("engine_id 必须是非空的小写字母、数字或下划线标识符")
            declared = getattr(adapter, "contract_version", ADAPTER_CONTRACT_VERSION)
            if declared != ADAPTER_CONTRACT_VERSION:
                raise ValueError(f"适配器契约版本不匹配：{declared} != {ADAPTER_CONTRACT_VERSION}")
            if engine_id in owners:
                adapters.pop(engine_id, None)
                raise ValueError(f"重复 engine_id {engine_id}：{owners[engine_id]} 与 {directory.name}；冲突适配器均已禁用")
            owners[engine_id] = directory.name
            adapters[engine_id] = adapter
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append(dict(directory=directory.name, message=message))
            log.error("Adapter %s failed to load: %s", directory.name, message)
    return adapters, errors


ADAPTERS, ADAPTER_LOAD_ERRORS = discover_adapters()


def adapter_catalog():
    """Authoritative metadata used by engine names, confirmation and installation UI."""
    return dict(adapters=[dict(
        engine_id=a.engine_id, display_name=a.display_name or a.engine_id,
        adapter_version=a.adapter_version, submittable=a.submittable,
        # Adapter-owned status text (e.g. "正在适配"); the UI renders it verbatim so
        # engine-specific wording never has to live in the shell.
        training_notice=a.training_notice,
        adapter_status=a.adapter_status,
        environment_strategy=a.environment_strategy,
        installation_help=a.installation_help,
        installation_source_help=a.installation_source_help,
        optional_extras=[{"name": name, "label": label} for name, label in a.optional_extras],
        default_installation_source=a.default_installation_source,
        torch_index_mode=a.torch_index_mode,
        clone_url=a.clone_url,
        clone_branch=a.clone_branch,
        clone_commit=a.clone_commit,
    ) for a in ADAPTERS.values()], errors=list(ADAPTER_LOAD_ERRORS))

def domain(record):
    return EngineInstallation(
        installation_id=record["installation_id"], engine_id=record["engine_id"], source_path=Path(record["source_path"]),
        management_mode=ManagementMode(record["management_mode"]), revision=EngineRevision(**record["revision"]),
        state=InstallationState(record["state"]), verification=VerificationState(record["verification"]),
        environment_id=record["environment_id"], python_executable=Path(record["python_executable"]) if record["python_executable"] else None)
