"""Single-owner post-green no-progress stop for token-lab v19."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_TOKENLAB_POSTGREEN_NO_PROGRESS"
THRESHOLD_ENV = "STATEFRAME_TOKENLAB_POSTGREEN_NO_PROGRESS_N"
DEFAULT_THRESHOLD = 3
SCHEMA = "stateframe-v19-postgreen-no-progress/1"
PRODUCER = "kernel_tokenlab_postgreen_no_progress"
EVENT = "tokenlab_v19_postgreen_no_progress_stop"
OWNER = "v19"
V20_MODE_ENV = "STATEFRAME_TOKENLAB_V20_FAIL_CLOSED_COMPLETION"
V20_SCHEMA = "stateframe-v20-fail-closed-completion/1"
V20_BLOCK_EVENT = "tokenlab_v20_no_progress_quieting_blocked"

_STATE_ATTR = "_tokenlab_v19_postgreen_no_progress"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    from kernel_capability_floor import enabled as capability_floor_enabled

    if capability_floor_enabled(runtime):
        # A settled-action heuristic cannot own a terminal outcome while the
        # stable recovery floor is active.
        return False
    configured = getattr(runtime, "stateframe_tokenlab_postgreen_no_progress_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def v20_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_v20_fail_closed_completion_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    value = str(os.environ.get(V20_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{V20_MODE_ENV} must be off or on")
    return value == "on"


def threshold(runtime: object | None = None) -> int:
    configured = getattr(runtime, "stateframe_tokenlab_postgreen_no_progress_n", None)
    raw = configured if configured is not None else os.environ.get(THRESHOLD_ENV)
    if raw in (None, ""):
        return DEFAULT_THRESHOLD
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{THRESHOLD_ENV} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{THRESHOLD_ENV} must be a positive integer")
    return value


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {"streak": 0, "owned": False, "emitted": False, "advisories": ()}
        setattr(runtime, _STATE_ATTR, state)
    return state


def _green(runtime: object) -> bool:
    lock = getattr(runtime, "green_lock", None)
    return bool(lock is not None and getattr(lock, "_green", False))


def _record(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return
    try:
        emitter(kind, producer=PRODUCER, event_schema=SCHEMA, dispatch=False, **fields)
    except Exception:  # noqa: BLE001 - stop authority must not fail on telemetry
        return


def _record_v20(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return
    try:
        emitter(kind, producer=PRODUCER, event_schema=V20_SCHEMA, dispatch=False, **fields)
    except Exception:  # noqa: BLE001 - stop telemetry must not break the run
        return


def _v20_blockers(runtime: object, *, phase: str) -> tuple[dict[str, object], ...]:
    if not v20_enabled(runtime):
        return ()
    try:
        from kernel_tokenlab_v20_completion_guard import guard
        return tuple(guard(runtime, phase=phase))
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as error:
        # A missing fail-closed evaluator is itself insufficient evidence for a
        # quieting stop.  Preserve the normal completion authority instead.
        _record_v20(runtime, V20_BLOCK_EVENT, phase=phase, allowed=False,
                     reason="completion_guard_unavailable",
                     error_type=type(error).__name__)
        return ({"id": "V20_GUARD_UNAVAILABLE",
                 "clause": "completion guard unavailable",
                 "reason": "fail-closed evaluator unavailable",
                 "concrete_counterexample": False},)


def _advisories(runtime: object) -> tuple[str, ...]:
    rows = getattr(runtime, "_tokenlab_postgreen_advisories", ())
    if isinstance(rows, Mapping):
        rows = rows.keys()
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return tuple(str(item) for item in rows if str(item).strip())
    return ()


def _collect_advisories(runtime: object) -> tuple[str, ...]:
    try:
        from kernel_counterexample_visibility import collect_counterexamples, counterexample_ids
        rows = collect_counterexamples(runtime)
        ids = tuple(str(item) for item in counterexample_ids(rows))
        if ids:
            return ids
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    rows = getattr(runtime, "live_counterexamples", ())
    if isinstance(rows, Mapping):
        rows = rows.keys()
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return tuple(str(item) for item in rows)
    return ()


def observe_turn(runtime: object, record: Mapping[str, object] | None = None) -> int:
    """Count closed green turns with no file change and no gating check closure."""
    if not enabled(runtime) or not _green(runtime) or owned(runtime):
        return int(_state(runtime).get("streak", 0) or 0)
    row = dict(record or {})
    changed = bool(row.get("file_changed", row.get("changed", False)))
    check_closed = bool(row.get("gating_check_closed", row.get("check_closed", False)))
    if changed or check_closed:
        _state(runtime)["streak"] = 0
        return 0
    state = _state(runtime)
    streak = int(state.get("streak", 0) or 0) + 1
    state["streak"] = streak
    if streak >= threshold(runtime):
        _stop(runtime, state)
    return streak


def _stop(runtime: object, state: dict[str, object]) -> None:
    if bool(state.get("owned")):
        return
    blockers = _v20_blockers(runtime, phase="quieting")
    if blockers:
        state["v20_blockers"] = tuple(dict(row) for row in blockers)
        if not state.get("v20_block_event_emitted"):
            state["v20_block_event_emitted"] = True
            _record_v20(runtime, V20_BLOCK_EVENT, phase="quieting", allowed=False,
                        blocker_count=len(blockers), blockers=[dict(row) for row in blockers])
        return
    advisories = _collect_advisories(runtime)
    state["owned"] = True
    state["advisories"] = advisories
    setattr(runtime, "_tokenlab_postgreen_advisories", advisories)
    if not state.get("emitted"):
        state["emitted"] = True
        _record(runtime, EVENT, owner=OWNER, threshold=threshold(runtime),
                streak=int(state.get("streak", 0) or 0),
                advisory_check_ids=list(advisories),
                advisory_count=len(advisories),
                reason="no file change and no gating check closed")


def owned(runtime: object) -> bool:
    return enabled(runtime) and bool(_state(runtime).get("owned"))


def done_override(runtime: object) -> dict[str, object] | None:
    if not owned(runtime):
        return None
    blockers = _v20_blockers(runtime, phase="done")
    if blockers:
        state = _state(runtime)
        state["owned"] = False
        state["v20_blockers"] = tuple(dict(row) for row in blockers)
        _record_v20(runtime, V20_BLOCK_EVENT, phase="done", allowed=False,
                    blocker_count=len(blockers), blockers=[dict(row) for row in blockers])
        return None
    advisories = _advisories(runtime)
    detail = (f"post-green no-progress stop owned by {OWNER}: "
              f"{len(advisories)} remaining counterexample(s) recorded as advisory")
    state = _state(runtime)
    if not state.get("done_emitted"):
        state["done_emitted"] = True
        _record(runtime, "tokenlab_v19_postgreen_no_progress_applied", owner=OWNER,
                advisory_check_ids=list(advisories), advisory_count=len(advisories))
    return {"admissible": True, "owner": OWNER, "detail": detail,
            "advisories": advisories}


__all__ = ["DEFAULT_THRESHOLD", "EVENT", "MODE_ENV", "OWNER", "PRODUCER", "SCHEMA",
           "THRESHOLD_ENV", "V20_BLOCK_EVENT", "V20_MODE_ENV", "V20_SCHEMA",
           "done_override", "enabled", "mode", "observe_turn", "owned", "threshold",
           "v20_enabled"]
