"""Deterministic, explicitly ordered tool-menu composition."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache


# Stages describe the complete per-request menu pipeline.  A stage may be
# omitted for an arm, but an installed stage always runs at this fixed point.
MENU_STAGE_ORDER = (
    "work_record_seed",
    "continuation",
    "generated",
    "decision_state",
    "tool_applicability",
    "repeat_guard",
    "work_record_restore",
    "terminal_green_lock",
    "closure_reserve",
)
_MENU_STAGES = frozenset(MENU_STAGE_ORDER)
_MAX_MEMO_BYTES = 256 * 1024


@lru_cache(maxsize=1)
def _reuse_switch() -> Callable[[], bool] | None:
    try:
        from stateframe import _startup_optimization_enabled as enabled
    except ModuleNotFoundError as error:
        if error.name != "stateframe":
            raise
        return None
    return enabled


def schema_reuse_enabled() -> bool:
    enabled = _reuse_switch()
    return enabled is not None and enabled()


def _canonical_menu_digest(payload: dict) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=4)
def _menu_digest_for_bytes(ordered: bytes) -> str:
    return _canonical_menu_digest(json.loads(ordered))


def freeze_menu_payload(payload: dict, *, reuse: bool) -> dict:
    """Detach a wire menu and bind it to its canonical digest."""
    if reuse:
        try:
            ordered = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            snapshot = json.loads(ordered) if len(ordered) <= _MAX_MEMO_BYTES else None
        except (TypeError, ValueError, RecursionError):
            snapshot = None
        # The ordered bytes include the effective schemas, schema version,
        # allow-set and optional done decision key. Use the eager copy if the
        # JSON round trip would coerce values such as tuples or object keys.
        if snapshot == payload:
            snapshot["digest"] = _menu_digest_for_bytes(ordered)
            return snapshot
    snapshot = copy.deepcopy(payload)
    snapshot["digest"] = _canonical_menu_digest(snapshot)
    return snapshot


def compose_menu(
    tools: Sequence[dict],
    stages: Mapping[str, Callable[[list[dict]], Sequence[dict]]],
) -> list[dict]:
    """Apply declared stages in the one canonical order.

    Registration order is intentionally irrelevant. Unknown stages are an
    error so a new mutator cannot silently bypass the declared pipeline.
    """
    unknown = sorted(set(stages) - _MENU_STAGES)
    if unknown:
        raise ValueError(f"unknown menu stages: {unknown}")
    current = list(tools)
    for name in MENU_STAGE_ORDER:
        apply = stages.get(name)
        if apply is None:
            continue
        result = apply(current)
        if result is None:
            raise TypeError(f"menu stage {name!r} returned None")
        current = list(result)
    return current


__all__ = ["MENU_STAGE_ORDER", "compose_menu"]
