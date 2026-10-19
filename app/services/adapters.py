"""Single access point for the discovered engine adapters.

Services and the supervisor import from here instead of ``adapters.registry``: the
lookup, the "plugin missing" error and the re-exports live in one place, so changing
how adapters are discovered never reaches the call sites.
"""

from adapters.registry import ADAPTERS, ADAPTER_LOAD_ERRORS, adapter_catalog, domain
from app.services.common import ServiceError


def adapter_for(engine_id: str):
    """The adapter registered for ``engine_id``; a missing plugin is a clear error."""
    adapter = ADAPTERS.get(engine_id)
    if adapter is None:
        raise ServiceError(f"未找到引擎适配器：{engine_id}")
    return adapter


__all__ = ["ADAPTERS", "ADAPTER_LOAD_ERRORS", "adapter_catalog", "adapter_for", "domain"]
