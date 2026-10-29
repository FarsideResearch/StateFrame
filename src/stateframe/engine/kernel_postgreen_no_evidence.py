"""Terminal post-green guard for a clean current revision.

``STATEFRAME_TOKENLAB_POSTGREEN_NO_EVIDENCE=on`` is deliberately narrower than a
turn cap or a generic done waiver.  It applies only while the terminal green lock
is engaged, every current-revision host check is PASS/closed, a current-revision
green test exists, and no live counterexample or pending verification exists.
In that state another read, probe, test, or edit cannot add evidence.  The normal
``done`` action remains available, and a real counterexample keeps the existing
repair window open.

Off is a strict no-op.  Every applied denial emits ``tokenlab_v16_postgreen_no_evidence``
from this module so replay can distinguish policy action from model choice.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_TOKENLAB_POSTGREEN_NO_EVIDENCE"
EVENT = "tokenlab_v16_postgreen_no_evidence"
PRODUCER = "kernel_postgreen_no_evidence"
RULE = "postgreen_no_new_evidence"
V20_MODE_ENV = "STATEFRAME_TOKENLAB_V20_FAIL_CLOSED_COMPLETION"

WORK_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
    "read", "grep", "probe", "run_tests", "test", "pytest", "predict",
})
_CLOSED = frozenset({
    "SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED",
    "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED",
})
def _missing(value: object) -> bool:
    return value is None or value == "" or value == "UNKNOWN" or value == "MISSING" or value == "UNAVAILABLE"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_postgreen_no_evidence_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def v20_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_v20_fail_closed_completion_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    value = str(os.environ.get(V20_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{V20_MODE_ENV} must be off or on")
    return value == "on"


def _row(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _rows(runtime: object) -> list[dict[str, object]]:
    raw = getattr(runtime, "host_closure_assessments", None)
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [row for item in raw if (row := _row(item))]


def _value_revision(value: object) -> str | None:
    row = _row(value)
    for key in ("product_revision", "artifact", "current_revision", "revision", "artifact_revision"):
        candidate = row.get(key)
        if not _missing(candidate):
            return str(candidate)
    for key in ("binding", "after_binding", "current_binding"):
        nested = row.get(key)
        nested_row = _row(nested)
        for child in ("product_revision", "artifact", "revision"):
            candidate = nested_row.get(child)
            if not _missing(candidate):
                return str(candidate)
    return None


def _current_revision(runtime: object) -> str | None:
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = _value_revision(binding())
        except (AttributeError, OSError, TypeError, ValueError):
            value = None
        if value is not None:
            return value
    for key in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, key, None)
        if not _missing(value):
            return str(value)
    return None


def _green_revision(lock: object) -> str | None:
    value = getattr(lock, "_green_revision", None)
    if isinstance(value, Mapping):
        return _value_revision(value)
    if not _missing(value):
        return str(value)
    return None


def _status(row: Mapping[str, object]) -> str:
    for key in ("status", "verdict", "result_class", "result", "state"):
        value = row.get(key)
        if not _missing(value):
            return str(value).strip().upper()
    receipt = row.get("receipt")
    receipt_row = _row(receipt)
    for key in ("status", "result_class", "result", "state"):
        value = receipt_row.get(key)
        if not _missing(value):
            return str(value).strip().upper()
    return "UNKNOWN"


def _has_counterexample(row: Mapping[str, object]) -> bool:
    for key in ("counterexample", "live_counterexample", "open_counterexample"):
        value = row.get(key)
        if value not in (None, False, [], {}, ""):
            return True
    receipt = _row(row.get("receipt"))
    return _has_counterexample(receipt) if receipt else False


def _assessment_current(runtime: object, row: Mapping[str, object], revision: str) -> bool:
    """Bind typed assessments to product revision or their current host snapshot."""
    explicit_revision = _value_revision(row)
    if explicit_revision is not None:
        return explicit_revision == revision
    current_snapshot = getattr(runtime, "host_closure_snapshot", None)
    receipts = row.get("receipts")
    if current_snapshot in (None, "") or not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        return False
    snapshots = []
    for item in receipts:
        receipt = _row(item)
        value = receipt.get("snapshot_digest", receipt.get("snapshot"))
        if value not in (None, ""):
            snapshots.append(str(value))
    return bool(snapshots) and all(value == str(current_snapshot) for value in snapshots)


def _live_counterexamples(runtime: object) -> bool:
    try:
        from kernel_live_counterexample_challenge import live_rows
        if live_rows(runtime):
            return True
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    for key in ("live_counterexamples", "open_counterexamples"):
        value = getattr(runtime, key, None)
        if value:
            return True
    return any(_has_counterexample(row) for row in _rows(runtime))


def _pending_verification(runtime: object) -> bool:
    state = getattr(runtime, "_post_green_menu_state", None)
    return isinstance(state, Mapping) and state.get("pending") is not None


def _green_test_exists(runtime: object, revision: str) -> bool:
    receipts = getattr(runtime, "test_receipts", ())
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        return False
    for item in receipts:
        row = _row(item)
        if row.get("exit_code") != 0:
            continue
        row_revision = _value_revision(row)
        if row_revision == revision:
            return True
        binding = _value_revision(row.get("after_binding"))
        if binding == revision and row.get("binding_stable") in (None, True):
            return True
    return False


def clean_current_revision(lock: object) -> bool:
    """Return true only when redundant post-green work is provably non-evidentiary."""
    runtime = getattr(lock, "runtime", None)
    if runtime is None or not enabled(runtime) or not bool(getattr(lock, "terminal_engaged", False)):
        return False
    revision = _current_revision(runtime)
    green_revision = _green_revision(lock)
    if revision is None or green_revision is None or revision != green_revision:
        return False
    if v20_enabled(runtime):
        try:
            from kernel_tokenlab_v20_completion_guard import guard
            if guard(runtime, phase="quieting"):
                return False
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            # Missing v20 evidence is fail-closed: the old no-evidence path
            # must not quiet a run when its defense cannot evaluate.
            return False
    rows = _rows(runtime)
    if not rows or _pending_verification(runtime) or _live_counterexamples(runtime):
        return False
    for row in rows:
        if not _assessment_current(runtime, row, revision) or _status(row) not in _CLOSED or _has_counterexample(row):
            return False
    return _green_test_exists(runtime, revision)


def decide(lock: object, action: str) -> tuple[bool, str] | None:
    """Deny only redundant work; ``None`` preserves the v15 decision path."""
    runtime = getattr(lock, "runtime", None)
    if runtime is not None and enabled(runtime):
        try:
            from kernel_tokenlab_postgreen_no_progress import owned
            if owned(runtime):
                return None
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
    if action not in WORK_ACTIONS or not clean_current_revision(lock):
        return None
    runtime = lock.runtime
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(EVENT, producer=PRODUCER, dispatch=False, action=action, rule=RULE,
             revision=_green_revision(lock), evidence="current_revision_all_checks_pass",
             live_counterexamples=False, pending_verification=False)
    return False, RULE


__all__ = [
    "EVENT", "MODE_ENV", "PRODUCER", "RULE", "WORK_ACTIONS", "clean_current_revision",
    "V20_MODE_ENV", "decide", "enabled", "mode", "v20_enabled",
]
