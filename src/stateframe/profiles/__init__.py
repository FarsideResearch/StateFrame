"""Shipped switch profiles. ``r2hc3`` is the default: the r2HC3 effective manifest."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = "r2hc3"


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
    return {str(k): str(v) for k, v in document["switches"].items()}


def engine_source_summary() -> dict[str, object]:
    path = HERE.parent / "engine" / "ENGINE_SOURCE.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    return {k: record.get(k) for k in ("source_tree", "source_hash", "shipped_hash", "file_count", "release_manifest")}
