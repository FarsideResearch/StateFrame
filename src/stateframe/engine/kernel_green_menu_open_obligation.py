"""Keep the repair surface open while current host counterexamples exist.

The green/coherence projection is allowed to share the public-green fact with
the terminal surface, but an executed counterexample is a separate obligation.
This module is the small policy boundary for that obligation.  It deliberately
reads raw host receipts before admission reduction, so an observation that is
not admitted for completion still keeps the model's repair tools reachable.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_GREEN_COHERENCE"
COUNTEREXAMPLE = "COUNTEREXAMPLE"
EXECUTED = "EXECUTED"
PASS_RESULTS = frozenset({"PASS", "PASSED", "SATISFIED", "GREEN", "CERTIFIED"})
EDIT_TOOLS = frozenset({"edit", "str_replace"})
RUN_TESTS_TOOL = "run_tests"
TRIPWIRE_THRESHOLD = 2
SUBMISSION_BLOCKED_PREFIX = "SUBMISSION BLOCKED"
TRIPWIRE_ERROR_CODE = "OPEN_COUNTEREXAMPLE_MENU_MISSING_EDIT"


class MenuObligationError(RuntimeError):
    """Raised when two consecutive model-visible menus strand repair."""


# Keep this release importable on the v46 Python 3.9 runner; ``slots=`` was
# added to ``dataclasses.dataclass`` in Python 3.10.
@dataclass(frozen=True)
class OpenCounterexample:
    """One current-revision executed counterexample exposed to policy."""

    check_id: str
    revision: object = None
    receipt_id: str = ""
    source: str = ""
    observed: object = None
    expected: object = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-open-counterexample/1",
            "check_id": self.check_id,
            "revision": copy.deepcopy(self.revision),
            "receipt_id": self.receipt_id,
            "source": self.source,
            "observed": copy.deepcopy(self.observed),
            "expected": copy.deepcopy(self.expected),
        }


def enabled(runtime: object | None = None) -> bool:
    """Return whether the coherence-gated obligation is active."""
    configured = getattr(runtime, "stateframe_green_coherence_mode", None)
    if configured is not None:
        return configured == "on"
    return os.environ.get(MODE_ENV, "off").strip().lower() == "on"


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return dict(converted)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            converted = dataclasses.asdict(value)
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _value(row: object, *names: str, default: object = None) -> object:
    mapped = _mapping(row)
    for name in names:
        if name in mapped:
            return mapped[name]
        if hasattr(row, name):
            return getattr(row, name)
    return default


def _rows(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "check_id", "check", "status", "result", "result_class",
                "underlying_status", "execution")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _status(row: object) -> str:
    for key in ("underlying_status", "underlying_result", "status", "result", "result_class"):
        value = _value(row, key)
        if value not in (None, ""):
            text = str(value).upper()
            if text != "REUSED":
                return text
    return ""


def _execution(row: object, *, compact_host_projection: bool = False) -> str:
    value = _value(row, "execution", default=None)
    if value not in (None, ""):
        return str(value).upper()
    # The bounded host projection omits execution because its receipt rows are
    # selected from executed records.  Do not make that presentation shape
    # silently erase a counterexample from the menu policy.
    return EXECUTED if compact_host_projection else ""


def _receipt_revision(row: object) -> object:
    for key in (
        "snapshot_digest", "snapshot", "revision", "current_revision",
        "product_revision", "repository_revision", "artifact_revision",
    ):
        value = _value(row, key, default=None)
        if value not in (None, "", "UNKNOWN", "MISSING"):
            return value
    for key in ("binding", "after_binding", "current_binding"):
        nested = _value(row, key, default=None)
        if isinstance(nested, Mapping):
            for name in ("product_revision", "artifact", "revision", "current_revision"):
                value = nested.get(name)
                if value not in (None, "", "UNKNOWN", "MISSING"):
                    return value
    return None


def _current_revisions(runtime: object | None, frame: Mapping[str, object] | None) -> tuple[object, ...]:
    values: list[object] = []
    snapshot = getattr(runtime, "host_closure_snapshot", None) if runtime is not None else None
    if snapshot not in (None, ""):
        values.append(snapshot)
    if isinstance(frame, Mapping):
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping) and closure.get("snapshot") not in (None, ""):
            values.append(closure["snapshot"])
        for key in ("current_revision", "product_revision", "revision"):
            if frame.get(key) not in (None, ""):
                values.append(frame[key])
        binding = frame.get("binding")
        if isinstance(binding, Mapping):
            for key in ("product_revision", "artifact", "revision", "current_revision"):
                if binding.get(key) not in (None, ""):
                    values.append(binding[key])
    if runtime is not None:
        binding_method = getattr(runtime, "binding", None)
        binding = None
        if callable(binding_method):
            try:
                binding = binding_method()
            except (AttributeError, OSError, TypeError, ValueError):
                binding = None
        if isinstance(binding, Mapping):
            for key in ("product_revision", "artifact", "revision", "current_revision"):
                if binding.get(key) not in (None, ""):
                    values.append(binding[key])
        for key in ("current_revision", "product_revision", "revision"):
            value = getattr(runtime, key, None)
            if value not in (None, ""):
                values.append(value)
    result: list[object] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _revision_equal(left: object, right: object) -> bool:
    if left == right or str(left) == str(right):
        return True
    left_text = str(left).split(":", 1)[-1]
    right_text = str(right).split(":", 1)[-1]
    return (len(left_text) >= 12 and len(right_text) >= 12
            and (left_text.startswith(right_text[:12])
                 or right_text.startswith(left_text[:12])))


def _is_current(row: object, revisions: Sequence[object]) -> bool:
    receipt_revision = _receipt_revision(row)
    if receipt_revision is None or not revisions:
        # A raw current host receipt has no reason to be treated as stale when
        # the host has not published a comparable identity.  Explicitly
        # mismatched identities remain excluded below.
        return True
    return any(_revision_equal(receipt_revision, current) for current in revisions)


def _receipt_sources(runtime: object | None, frame: Mapping[str, object] | None):
    if runtime is not None:
        for name in (
            "host_check_receipts", "host_repair_check_receipts",
            "host_closure_receipts", "counterexample_receipts", "test_receipts",
            "probe_receipts", "receipts",
        ):
            for row in _rows(getattr(runtime, name, ())):
                yield row, name, False, ""
        for index, assessment in enumerate(getattr(runtime, "host_closure_assessments", ()) or ()):
            for row in _rows(_value(assessment, "receipts", default=())):
                yield row, "host_closure_assessment", False, str(
                    _value(assessment, "obligation_id", "id", default=""))
    if isinstance(frame, Mapping):
        for name in (
            "host_check_receipts", "host_repair_check_receipts", "receipts",
            "test_receipts", "probe_receipts",
        ):
            for row in _rows(frame.get(name, ())):
                yield row, name, False, ""
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            for row in _rows(closure.get("receipts", ())):
                yield row, "host_closure", True, ""
            for assessment in _rows(closure.get("assessments", ())):
                for row in _rows(_value(assessment, "receipts", default=())):
                    yield row, "host_closure_assessment", True, str(
                        _value(assessment, "obligation_id", "id", default=""))
        for assessment in _rows(frame.get("obligation_assessments", ())):
            for row in _rows(_value(assessment, "receipts", default=())):
                yield row, "obligation_assessment", False, str(
                    _value(assessment, "obligation_id", "id", default=""))


def _candidate_id(row: object, fallback: str, ordinal: int) -> str:
    value = _value(row, "check_id", "check", "id", default=None)
    if value in (None, ""):
        value = _value(row, "obligation_id", "obligation", default=None)
    if value in (None, ""):
        value = fallback
    if value in (None, ""):
        value = f"counterexample-{ordinal}"
    return str(value)


def _fingerprint(row: object) -> str:
    mapped = _mapping(row)
    try:
        return json.dumps(mapped, sort_keys=True, ensure_ascii=False, default=str,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(row)


def open_counterexamples(runtime: object | None = None,
                         frame: Mapping[str, object] | None = None) -> tuple[OpenCounterexample, ...]:
    """Return latest current-revision executed counterexamples by check id.

    With the readiness authority on, the open set IS the authority's actionable
    set: blockers plus typed repairable advisories.  Non-repairable advisories
    remain visible in the frame but cannot keep tools open.
    """
    if not enabled(runtime):
        return ()
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
            or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"):
        from kernel_readiness_authority import runtime_snapshot
        decision = runtime_snapshot(runtime, frame)
        if decision is None:
            return ()
        # Menu/override/tripwire scope: only the same rows that the action gate
        # can address keep the repair surface; only blockers refuse done.
        return tuple(OpenCounterexample(
            check_id=str(row.get("check_id") or row.get("obligation_id") or ""),
            revision=row.get("revision", decision.revision),
            receipt_id=str(row.get("receipt_id") or row.get("evidence_ref") or ""),
            source="readiness_authority",
            observed=copy.deepcopy(row.get("observed")),
            expected=copy.deepcopy(row.get("expected")),
        ) for row in (*decision.blockers, *decision.repairable_advisories))
    revisions = _current_revisions(runtime, frame)
    candidates: list[tuple[str, OpenCounterexample, str, int]] = []
    seen: set[tuple[str, str, str]] = set()
    for ordinal, (raw, source, compact, fallback) in enumerate(
            _receipt_sources(runtime, frame), start=1):
        if _status(raw) != COUNTEREXAMPLE:
            continue
        if _execution(raw, compact_host_projection=compact) != EXECUTED:
            continue
        if not _is_current(raw, revisions):
            continue
        check_id = _candidate_id(raw, fallback, ordinal)
        receipt_id = str(_value(raw, "receipt_id", "evidence_ref", default="") or "")
        revision = _receipt_revision(raw)
        row_key = (check_id, receipt_id, _fingerprint(raw))
        if row_key in seen:
            continue
        seen.add(row_key)
        candidates.append((
            check_id,
            OpenCounterexample(
                check_id=check_id,
                revision=revision,
                receipt_id=receipt_id,
                source=source,
                observed=copy.deepcopy(_value(raw, "observed", "observed_values")),
                expected=copy.deepcopy(_value(raw, "expected")),
            ),
            _status(raw), ordinal,
        ))

    # A later current PASS for the same check closes an earlier receipt.  Host
    # reruns normally change the snapshot, but this also handles same-revision
    # replay fixtures without leaving an old failure permanently open.
    latest: dict[str, tuple[str, object, int]] = {}
    for ordinal, (raw, source, compact, fallback) in enumerate(
            _receipt_sources(runtime, frame), start=1):
        if not _is_current(raw, revisions):
            continue
        check_id = _candidate_id(raw, fallback, ordinal)
        latest[check_id] = (_status(raw), raw, ordinal)
    result: list[OpenCounterexample] = []
    for check_id, candidate, _status_value, ordinal in candidates:
        latest_value = latest.get(check_id)
        if latest_value is not None and latest_value[2] >= ordinal \
                and latest_value[0] in PASS_RESULTS:
            continue
        if not any(item.check_id == check_id for item in result):
            result.append(candidate)
    return tuple(result)


def counterexample_ids(runtime: object | None = None,
                       frame: Mapping[str, object] | None = None) -> tuple[str, ...]:
    return tuple(item.check_id for item in open_counterexamples(runtime, frame))


def blocking_counterexamples(runtime: object | None = None,
                             frame: Mapping[str, object] | None = None
                             ) -> tuple[OpenCounterexample, ...]:
    """Open counterexamples that BLOCK submission (r38 convergence).

    The counterexample-visibility classifier is the single validity authority:
    a check it demotes to advisory (unbound entry point, self-contradicting,
    invalid binding) stays visible and keeps the repair surface, but never
    refuses done.  With the readiness authority on this is its blocker set.
    """
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
            or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"):
        from kernel_readiness_authority import runtime_snapshot
        decision = runtime_snapshot(runtime, frame)
        if decision is None:
            return ()
        ids = set(decision.blocker_ids)
        return tuple(item for item in open_counterexamples(runtime, frame)
                     if item.check_id in ids)
    result = open_counterexamples(runtime, frame)
    if not result:
        return ()
    try:
        from kernel_counterexample_visibility import collect_counterexamples
        advisory = {
            str(row.get("check_id") or row.get("obligation_id") or "")
            for row in collect_counterexamples(runtime, frame)
            if not bool(row.get("counterexample_authority"))}
    except (AttributeError, ImportError, TypeError, ValueError):
        advisory = set()
    return tuple(item for item in result if item.check_id not in advisory)


def blocking_counterexample_ids(runtime: object | None = None,
                                frame: Mapping[str, object] | None = None) -> tuple[str, ...]:
    return tuple(item.check_id for item in blocking_counterexamples(runtime, frame))


def submission_blocked_text(ids: Sequence[object]) -> str:
    values = [str(item) for item in ids if str(item).strip()]
    return f"{SUBMISSION_BLOCKED_PREFIX}: open counterexamples " \
        + json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def ensure_open_menu(current_tools: Sequence[dict], all_tools: Sequence[dict] | None,
                     runtime: object | None = None,
                     frame: Mapping[str, object] | None = None) -> list[dict]:
    """Restore edit and test affordances after any terminal-menu narrowing."""
    if not open_counterexamples(runtime, frame):
        return list(current_tools)
    source = list(all_tools) if all_tools is not None else list(current_tools)
    result = list(current_tools)
    names = {
        tool.get("function", {}).get("name") for tool in result
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    for tool in source:
        if not isinstance(tool, Mapping) or not isinstance(tool.get("function"), Mapping):
            continue
        name = tool["function"].get("name")
        if name in EDIT_TOOLS or name == RUN_TESTS_TOOL:
            if name not in names:
                result.append(tool)
                names.add(name)
    return result


def reset_override_state_if_evidence_changed(lock: object, *, runtime: object | None = None,
                                             frame: Mapping[str, object] | None = None) -> None:
    """Reset a coherence refusal override on revision/receipt/counterexample churn."""
    runtime = runtime if runtime is not None else getattr(lock, "runtime", None)
    if not enabled(runtime):
        return
    receipts = []
    for raw, _source, _compact, _fallback in _receipt_sources(runtime, frame):
        receipts.append(_fingerprint(raw))
    revisions = tuple(str(value) for value in _current_revisions(runtime, frame))
    evidence = (revisions, tuple(receipts), counterexample_ids(runtime, frame))
    prior = getattr(lock, "_green_menu_open_obligation_evidence", None)
    if prior is None:
        setattr(lock, "_green_menu_open_obligation_evidence", evidence)
        return
    if prior == evidence:
        return
    reset = getattr(lock, "_reset_refusal_state", None)
    if callable(reset):
        reset()
    else:
        for name, value in (
            ("_override_action_kind", None), ("_refused_action_kind", None),
            ("_first_action_refused", None), ("_first_action_overridden", False),
            ("_refusal_count", 0), ("_consecutive_refusals", 0),
        ):
            if hasattr(lock, name):
                setattr(lock, name, value)
    setattr(lock, "_green_menu_open_obligation_evidence", evidence)


def refusal_override(lock: object) -> str | None:
    reset_override_state_if_evidence_changed(lock)
    runtime = getattr(lock, "runtime", None)
    if counterexample_ids(runtime):
        # A refusal override is a terminal-surface escape hatch.  It must not
        # become the authority while repair evidence is still open.
        reset = getattr(lock, "_reset_refusal_state", None)
        if callable(reset):
            reset()
        return None
    value = getattr(lock, "_override_action_kind", None)
    return str(value) if value else None


def done_refusal(runtime: object | None = None,
                 frame: Mapping[str, object] | None = None) -> str | None:
    ids = blocking_counterexample_ids(runtime, frame)
    return submission_blocked_text(ids) if ids else None


def observe_model_menu(runtime: object, tools: Sequence[dict]) -> None:
    """Trip on two consecutive model-visible menus without an edit affordance."""
    if not enabled(runtime):
        return
    open_ids = counterexample_ids(runtime)
    if not open_ids:
        setattr(runtime, "_green_menu_open_obligation_missing_edit", 0)
        setattr(runtime, "_green_menu_open_obligation_open_ids", ())
        return
    previous_ids = tuple(getattr(runtime, "_green_menu_open_obligation_open_ids", ()))
    if previous_ids != open_ids:
        missing = 0
    else:
        missing = int(getattr(runtime, "_green_menu_open_obligation_missing_edit", 0))
    names = {
        tool.get("function", {}).get("name") for tool in tools
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    if names.intersection(EDIT_TOOLS):
        missing = 0
    else:
        missing += 1
    setattr(runtime, "_green_menu_open_obligation_open_ids", open_ids)
    setattr(runtime, "_green_menu_open_obligation_missing_edit", missing)
    if missing < TRIPWIRE_THRESHOLD:
        return
    detail = submission_blocked_text(open_ids)
    journal = getattr(runtime, "journal", None)
    if journal is not None and callable(getattr(journal, "emit", None)):
        journal.emit(
            "kernel_error",
            producer="kernel_green_menu_open_obligation",
            error_code=TRIPWIRE_ERROR_CODE,
            detail=detail,
            check_ids=list(open_ids),
            consecutive_responses=missing,
            menu=sorted(str(name) for name in names if name),
            dispatch=False,
        )
    raise MenuObligationError(
        f"{TRIPWIRE_ERROR_CODE}: {detail}; edit affordance missing for "
        f"{missing} consecutive responses"
    )


__all__ = [
    "COUNTEREXAMPLE", "EDIT_TOOLS", "EXECUTED", "MenuObligationError",
    "OpenCounterexample", "PASS_RESULTS", "RUN_TESTS_TOOL", "SUBMISSION_BLOCKED_PREFIX",
    "blocking_counterexample_ids", "blocking_counterexamples",
    "TRIPWIRE_ERROR_CODE", "counterexample_ids", "done_refusal", "enabled",
    "ensure_open_menu", "observe_model_menu", "open_counterexamples",
    "refusal_override", "reset_override_state_if_evidence_changed", "submission_blocked_text",
]
