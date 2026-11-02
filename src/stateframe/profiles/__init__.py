"""Shipped switch profiles. ``r2hc3`` is the default: the r2HC3 effective manifest."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = "r2hc3"
PRIVILEGED_PREFIXES = ("STATEFRAME_SHELL", "STATEFRAME_WEB", "STATEFRAME_TOOLS",
                       "STATEFRAME_HOST_MODE", "STATEFRAME_HOST_AUTHORIZATION",
                       "STATEFRAME_HOST_SANDBOX", "STATEFRAME_RUN_ENV", "STATEFRAME_GIT_TOOL",
                       "STATEFRAME_TOOL_SETTINGS", "STATEFRAME_HOOK")


def available() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.json"))


def load(name: str) -> dict[str, str]:
    """Switch values of a profile (a name from ``available()`` or a path to a profile file)."""
    path = Path(name)
    if not path.suffix:
        path = HERE / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(available())}")
    document = json.loads(path.read_text())
    if document.get("schema") != "stateframe-product-profile/1":
        raise ValueError(f"{path.name} is not a StateFrame product profile")
    switches = document["switches"]
    if not isinstance(switches, dict):
        raise ValueError(f"{path.name}: switches must be an object")
    privileged = [key for key in switches if isinstance(key, str) and key.startswith(PRIVILEGED_PREFIXES)]
    if privileged:
        raise ValueError(f"{path.name}: privileged tool settings belong on the host CLI, not in profiles: "
                         + ", ".join(sorted(privileged)))
    # The product requires sealed typed frames; lean builds an append-only
    # transcript instead. Refuse before materialization, not at request 1.
    if str(switches.get("STATEFRAME_LEAN", "off")).strip().lower() != "off":
        raise ValueError("REFUSED LEAN_UNSUPPORTED: product runs require typed frames; "
                         "set STATEFRAME_LEAN=off and use the composed loop")
    return {str(k): str(v) for k, v in switches.items()}


def load_routes(name: str) -> object:
    """Read a profile's optional routing table separately from engine switches."""
    path = Path(name)
    if not path.suffix:
        path = HERE / f"{name}.json"
    document = json.loads(path.read_text())
    return document.get("routes", {})


def engine_source_summary() -> dict[str, object]:
    path = HERE.parent / "engine" / "ENGINE_SOURCE.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    return {k: record.get(k) for k in ("source_tree", "source_hash", "shipped_hash", "file_count", "release_manifest")}
