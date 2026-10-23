"""Project version ↔ engine git baseline compatibility.

One LP version may pin several engines (one entry per engine). The manifest lives
next to the backend package so a release carries it, and is edited by
``scripts/sync_engine_versions.py`` when an engine is pulled and the adapter updated.
"""

from __future__ import annotations

import json
from pathlib import Path

from app import APP_VERSION

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "engine_compat.json"


def load_manifest() -> dict:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": 1, "versions": []}
    return data if isinstance(data, dict) else {"schema": 1, "versions": []}


def version_entry(version: str | None = None) -> dict | None:
    wanted = version or APP_VERSION
    for entry in load_manifest().get("versions") or []:
        if isinstance(entry, dict) and entry.get("version") == wanted:
            return entry
    return None


def _commit_matches(expected: str | None, current: str | None) -> bool:
    if not expected or not current:
        return False
    expected, current = expected.lower(), current.lower()
    return current.startswith(expected) or expected.startswith(current)


def engine_compat(record: dict, adapter_version: str | None = None, *, version: str | None = None) -> dict:
    """Compare one installation with the current project's engine baseline."""
    entry = version_entry(version)
    current = record.get("revision") or {}
    current_value = {
        "ref": current.get("requested_ref"),
        "commit": current.get("commit"),
        "fingerprint": current.get("fingerprint"),
    }
    if entry is None:
        return {"version": version or APP_VERSION, "status": "unmapped",
                "baseline": None, "current": current_value, "reasons": ["当前项目版本没有引擎基线"]}
    expected = next(
        (item for item in entry.get("engines") or []
         if isinstance(item, dict) and item.get("engine_id") == record.get("engine_id")),
        None,
    )
    if expected is None:
        return {"version": entry.get("version"), "status": "unlisted",
                "baseline": None, "current": current_value,
                "reasons": ["当前项目版本未登记该引擎"]}
    reasons = []
    if not current.get("commit"):
        reasons.append("安装实例没有可核对的 git commit")
    elif not _commit_matches(str(expected.get("commit") or ""), str(current.get("commit") or "")):
        reasons.append("引擎 commit 与项目适配基线不一致")
    expected_ref = str(expected.get("ref") or "").strip()
    current_ref = str(current.get("requested_ref") or "").strip()
    if expected_ref and current_ref and expected_ref != current_ref:
        reasons.append("引擎分支与项目适配基线不一致")
    expected_adapter = str(expected.get("adapter_version") or "").strip()
    if expected_adapter and adapter_version and expected_adapter != adapter_version:
        reasons.append("适配器版本与项目适配基线不一致")
    return {
        "version": entry.get("version"),
        "status": "supported" if not reasons else "mismatch",
        "baseline": {
            "ref": expected.get("ref"),
            "commit": expected.get("commit"),
            "adapter_version": expected.get("adapter_version"),
        },
        "current": current_value,
        "adapter_version": adapter_version,
        "reasons": reasons,
    }
