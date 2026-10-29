"""One host-owned terminal projection for green-lock and completion rendering.

The legacy kernels intentionally keep their older projections available.  This
module is the narrow bridge used when terminal green coherence is selected: it
reads the already trusted public-test verdict and host assessments, then gives
all terminal-facing consumers the same immutable status projection.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from kernel_counterexample_visibility import (
    COUNTEREXAMPLE_LOCK_LABEL,
    COUNTEREXAMPLE_PUBLIC_FIELD,
    COUNTEREXAMPLES_FIELD,
    READY_WITH_COUNTEREXAMPLES,
    REPAIR_ACTION,
    collect_counterexamples,
    counterexample_ids,
    render_counterexample_blocks,
)
from kernel_verdict import NO_HOST_CHECKS


MODE_ENV = "STATEFRAME_GREEN_COHERENCE"
K_CONSECUTIVE_REFUSALS = 3
_CLOSED = frozenset({"done", "closed", "dropped", "satisfied", "resolved"})
_TEST_KEYS = frozenset({
    "latest_receipts", "recent_test_feedback", "test_receipts", "probe_receipts",
    "test_results", "receipts", "applicable_check_statuses", "check_statuses",
})
_CHURN_KEYS = frozenset({
    "action_history", "refusal_history", "refusals", "action_refusals",
    "green_lock_refusals", "terminal_refusals",
    "step", "receipt_count", "budget", "closure_budget", "last_observation",
})


def mode(lock_mode: str | None = None) -> str:
    """Return the strict coherence mode.

    Default OFF (r37t integration): the history arm also runs
    STATEFRAME_GREEN_LOCK=terminal, so an implicit terminal opt-in would change
    history behaviour.  The frame arm sets STATEFRAME_GREEN_COHERENCE=on.
    ``on`` takes effect only with a terminal green lock: when ``lock_mode`` is
    supplied and is not ``terminal`` the projection stays off.
    """
    value = os.environ.get(MODE_ENV)
    if value is None:
        return "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    if value == "on" and lock_mode is not None and lock_mode != "terminal":
        return "off"
    return value


def enabled(lock_mode: str | None = None) -> bool:
    return mode(lock_mode) == "on"


def runtime_enabled(runtime: object) -> bool:
    configured = getattr(runtime, "stateframe_green_coherence_mode", None)
    if configured is not None:
        return configured == "on"
    lock = getattr(runtime, "green_lock", None)
    lock_mode = getattr(lock, "mode", None)
    try:
        return enabled(lock_mode)
    except ValueError:
        return False


def render_enabled(runtime: object) -> bool:
    """Whether a runtime has opted into the new textual terminal projection.

    Hand-built legacy lock fixtures do not carry the runtime marker.  Keeping
    them on the old compact handoff preserves their contract while real
    ``StateFrameRuntime`` instances get the default terminal projection.
    """
    return getattr(runtime, "stateframe_green_coherence_mode", None) == "on"


def _counterexample_visibility_enabled(runtime: object | None) -> bool:
    """Keep the receipt projection inside the coherence switch."""
    if getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on":
        return True
    if getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on":
        return True
    if (getattr(runtime, "stateframe_repair_blocks_done_mode", None) is None
            and os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"):
        return True
    if getattr(runtime, "stateframe_requirement_alignment_mode", "off") == "on":
        return True
    try:
        if runtime is None:
            return enabled()
        return runtime_enabled(runtime)
    except (AttributeError, TypeError, ValueError):
        return False


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            result = dataclasses.asdict(value)
        except (TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    return {}


def _rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "id", "obligation_id", "check_id", "selector", "status", "verdict")):
            return [dict(value)]
        result = []
        for key, item in value.items():
            row = _mapping(item)
            if row:
                row.setdefault("id", str(key))
                result.append(row)
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            row = _mapping(item)
            if row:
                result.append(row)
        return result
    return []


def _selectors(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _required(runtime: object | None, frame: Mapping[str, object] | None) -> list[str]:
    if runtime is not None:
        method = getattr(runtime, "required_public_selectors", None)
        if callable(method):
            try:
                values = _selectors(method())
            except (AttributeError, TypeError, ValueError):
                values = []
            if values:
                return list(dict.fromkeys(values))
        owner = getattr(runtime, "owner", None)
        for value in (getattr(owner, "f2p_ids", ()), getattr(owner, "f2p_inworld", ())):
            values = _selectors(value)
            if values:
                return list(dict.fromkeys(values))
    if isinstance(frame, Mapping):
        values = _selectors(frame.get("required_public_checks"))
        if values:
            return list(dict.fromkeys(values))
        contract = frame.get("completion_contract")
        if isinstance(contract, Mapping):
            return list(dict.fromkeys(_selectors(contract.get("required_public_checks"))))
    return []


def _binding(runtime: object | None, frame: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if runtime is not None:
        method = getattr(runtime, "binding", None)
        if callable(method):
            try:
                value = method()
            except (AttributeError, TypeError, ValueError):
                value = None
            if isinstance(value, Mapping):
                return value
    if isinstance(frame, Mapping):
        for key in ("binding", "current_binding", "revision_binding"):
            if isinstance(frame.get(key), Mapping):
                return frame[key]
        world = frame.get("world")
        if isinstance(world, Mapping) and isinstance(world.get("revision_binding"), Mapping):
            return world["revision_binding"]
    return None


def _revision(value: object) -> object:
    row = value if isinstance(value, Mapping) else {}
    for key in ("product_revision", "artifact", "revision", "current_revision"):
        candidate = row.get(key)
        if candidate not in (None, "", "MISSING", "UNKNOWN"):
            return candidate
    return None


def _frame_revision(frame: Mapping[str, object]) -> object:
    for key in ("current_revision", "product_revision", "revision"):
        if frame.get(key) not in (None, "", "UNKNOWN", "MISSING"):
            return frame[key]
    binding = _binding(None, frame)
    current = _revision(binding)
    if current is not None:
        return current
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping) and len(revisions) == 1:
        return next(iter(revisions.values()))
    verification = frame.get("verification")
    if isinstance(verification, Mapping):
        return _revision(verification)
    return None


def _receipt_rows(runtime: object | None, frame: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if runtime is not None:
        value = getattr(runtime, "test_receipts", ())
        rows = [row for row in _rows(value) if isinstance(row, Mapping)]
        if rows:
            return rows
    result: list[Mapping[str, object]] = []
    if isinstance(frame, Mapping):
        for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
            result.extend(row for row in _rows(frame.get(key)) if isinstance(row, Mapping))
    return result


def _verdict(runtime: object | None, frame: Mapping[str, object] | None,
             required: Sequence[str], receipts: Sequence[Mapping[str, object]]) -> object:
    if runtime is not None:
        method = getattr(runtime, "public_test_verdict", None)
        if callable(method):
            try:
                return method()
            except (AttributeError, TypeError, ValueError):
                pass
    from kernel_verdict import evaluate_required_tests
    current = _binding(runtime, frame)
    return evaluate_required_tests(current, required, receipts)


def _verdict_state(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("state", value.get("status", "UNRESOLVED"))).upper()
    return str(getattr(value, "state", "UNRESOLVED")).upper()


def _assessment_blocking(value: object) -> bool:
    from kernel_obligation_assess import assessment_is_blocking
    try:
        if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
            from kernel_green_lock_repair import is_advisory_assessment
            if is_advisory_assessment(value):
                return False
        return bool(assessment_is_blocking(value))
    except (AttributeError, TypeError, ValueError):
        return bool(getattr(value, "blocking", False))


def _assessment_id(value: object) -> str | None:
    row = _mapping(value)
    for key in ("obligation_id", "id", "check_id", "item"):
        candidate = row.get(key)
        if candidate not in (None, "") and not isinstance(candidate, (Mapping, list, tuple)):
            return str(candidate)
    for key in ("obligation_id", "id", "check_id"):
        candidate = getattr(value, key, None)
        if candidate not in (None, ""):
            return str(candidate)
    return None


def _is_unknown_assessment(value: object) -> bool:
    row = _mapping(value)
    status = str(row.get("verdict", row.get("status", row.get("state", "")))).upper()
    if not row:
        status = str(getattr(value, "verdict", getattr(
            value, "status", getattr(value, "state", "")))).upper()
    if status not in {"UNKNOWN", "AMBIGUOUS", "UNRESOLVED"}:
        return False
    keys = set(row)
    if keys.intersection({
        "verdict", "reason", "reason_code", "check_ids", "receipts",
        "coverage_required", "interpretation", "assessment", "status", "blocking",
    }):
        return True
    return any(getattr(value, key, None) is not None for key in (
        "verdict", "reason", "reason_code", "check_ids", "receipts", "blocking"))


def _host_assessments(runtime: object | None, frame: Mapping[str, object] | None) -> list[object]:
    result: list[object] = []
    if runtime is not None:
        result.extend(getattr(runtime, "host_closure_assessments", ()) or ())
    if isinstance(frame, Mapping):
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            result.extend(closure.get("assessments", ()) or ())
        result.extend(frame.get("obligation_assessments", ()) or ())
    return result


def _counterexamples(runtime: object | None, frame: Mapping[str, object] | None
                     ) -> tuple[dict[str, object], ...]:
    if not _counterexample_visibility_enabled(runtime):
        return ()
    try:
        return collect_counterexamples(runtime, frame)
    except (AttributeError, ImportError, TypeError, ValueError):
        # Presentation must not end a run when an optional receipt adapter is
        # malformed.  The ordinary public verdict remains available.
        return ()


def _decision_row(runtime: object | None, frame: Mapping[str, object] | None) -> dict[str, object]:
    value = getattr(runtime, "_last_completion_decision", None) if runtime is not None else None
    if value is None and isinstance(frame, Mapping):
        for key in ("completion_reducer_decision", "completion_decision", "completion_status",
                    "atomic_completion"):
            if frame.get(key) is not None:
                value = frame[key]
                break
    return _mapping(value)


def _open_ids(runtime: object | None, frame: Mapping[str, object] | None,
              decision: Mapping[str, object]) -> tuple[list[str], list[object]]:
    assessments = _host_assessments(runtime, frame)
    def advisory_assessment(item: object) -> bool:
        del item
        return False
    if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
        try:
            from kernel_green_lock_repair import is_advisory_assessment
            advisory_assessment = is_advisory_assessment
        except (ImportError, TypeError, ValueError):
            pass
    nonblocking_ids = {
        ident for item in assessments
        if (advisory_assessment(item)
            or (_is_unknown_assessment(item) and not _assessment_blocking(item)))
        for ident in (_assessment_id(item),) if ident
    }
    values: list[object] = []
    if runtime is not None:
        candidate = getattr(runtime, "open_obligations", None)
        if callable(candidate):
            try:
                values.extend(_rows(candidate()) or list(candidate()))
            except (AttributeError, TypeError, ValueError):
                pass
        work_record = getattr(runtime, "work_record", None)
        snapshot = work_record.snapshot() if callable(getattr(work_record, "snapshot", None)) else work_record
        if isinstance(snapshot, Mapping):
            values.extend(snapshot.get("open_obligations", ()) or ())
    if isinstance(frame, Mapping):
        for key in ("open_work", "pending_multi_definition_edits", "unresolved_obligations",
                    "obligations", "open_obligations"):
            values.extend(_rows(frame.get(key)) if isinstance(frame.get(key), Mapping)
                          else list(frame.get(key, ()) or ())
                          if isinstance(frame.get(key), Sequence) and not isinstance(
                              frame.get(key), (str, bytes, bytearray)) else
                          [frame[key]] if isinstance(frame.get(key), str) else [])
    needs = decision.get("needs_work")
    if isinstance(needs, Mapping):
        values.extend(_rows(needs.get("blocking", needs.get("obligations", ()))))
    ids = decision.get("needs_work_ids", ())
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, bytearray)):
        values.extend(ids)
    result: list[str] = []
    blocking: list[object] = []
    for item in values:
        row = _mapping(item)
        ident = _assessment_id(item)
        if ident is None and not row and item not in (None, ""):
            ident = str(item)
        if not ident or ident.lower() in _CLOSED or ident in nonblocking_ids:
            continue
        if row and _is_unknown_assessment(row) and not _assessment_blocking(row):
            continue
        if ident not in result:
            result.append(ident)
        if row and _assessment_blocking(row):
            blocking.append(row)
    # Only typed blocking assessments belong in the terminal open-work list.
    for item in assessments:
        if not _assessment_blocking(item):
            continue
        ident = _assessment_id(item)
        if ident and ident not in result:
            result.append(ident)
        if ident:
            blocking.append(_mapping(item) or item)
    return result, blocking


def _lock_green(runtime: object | None, frame: Mapping[str, object] | None) -> bool | None:
    lock = getattr(runtime, "green_lock", None) if runtime is not None else None
    if lock is not None:
        try:
            return bool(lock._green_now(frame))
        except (AttributeError, TypeError, ValueError):
            return bool(getattr(lock, "green", False))
    return None


def green_now(lock: object, frame: Mapping[str, object] | None = None) -> bool:
    """Evaluate the same current-receipt predicate used by terminal status.

    ``GreenLock`` supplies the revision/edit binding predicate; this helper
    supplies the shared verdict predicate.  Keeping the call in one function
    prevents the renderer from treating a stale frame field as fresher than a
    trusted receipt.
    """
    runtime = getattr(lock, "runtime", None)
    try:
        from kernel_obligation_assess import assessment_is_blocking
        if (os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() != "on"
                and getattr(runtime, "stateframe_host_closure_mode", "off") == "on"
                and any(assessment_is_blocking(item)
                        for item in getattr(runtime, "host_closure_assessments", ()) )):
            return False
    except (ImportError, TypeError, ValueError):
        return False
    current = _binding(runtime, frame)
    revision = _revision(current)
    if revision is None:
        return False
    changed = getattr(lock, "_product_changed_after_edit", None)
    if callable(changed):
        try:
            if not changed(revision, frame):
                return False
        except (AttributeError, TypeError, ValueError):
            return False
    required = _required(runtime, frame if isinstance(frame, Mapping) else {})
    if not required:
        return False
    method = getattr(runtime, "public_test_verdict", None)
    if callable(method):
        try:
            return _verdict_state(method()) == "SATISFIED"
        except (AttributeError, TypeError, ValueError):
            # Small replay fixtures may expose the runtime method without its
            # optional integrity sidecars.  Fall back to the same typed
            # receipt evaluator used by the lock's legacy boundary.
            pass
    receipts = _receipt_rows(runtime, frame if isinstance(frame, Mapping) else {})
    if receipts:
        from kernel_verdict import evaluate_required_tests
        return evaluate_required_tests(current, required, receipts).state == "SATISFIED"
    if isinstance(frame, Mapping):
        from kernel_green_lock import _frame_is_green
        return _frame_is_green(frame, revision)
    return False


@dataclass(frozen=True, slots=True)
class TerminalStatus(Mapping[str, object]):
    """The sole terminal status projection consumed by renderers."""

    green: bool
    public_check_status: str
    verdict_state: str
    current_revision: object
    selectors: tuple[str, ...]
    open_ids: tuple[str, ...]
    blocking: tuple[object, ...]
    submission_state: str
    next_feasible_action: str | None
    receipt_revision: object = None
    counterexamples: tuple[Mapping[str, object], ...] = ()
    repair_action: Mapping[str, object] | None = None
    counterexample_visibility_enabled: bool = False
    readiness_authority_enabled: bool = False
    repair_blocks_done_enabled: bool = False
    repair_blocking_counterexamples: tuple[Mapping[str, object], ...] = ()
    
    @property
    def counterexample_authority(self) -> tuple[Mapping[str, object], ...]:
        if not self.counterexample_visibility_enabled:
            return ()
        return tuple(item for item in self.counterexamples
                     if bool(item.get("counterexample_authority")))
    open_counterexample_ids: tuple[str, ...] = ()
    no_host_checks: bool = False

    @property
    def state(self) -> str:
        if self.no_host_checks and not self.blocking:
            return NO_HOST_CHECKS
        if not self.green or self.blocking:
            return "BLOCKED"
        if self.counterexample_authority or self.repair_blocking_counterexamples:
            # Readiness authority: a live admitted counterexample is NOT ready.
            return ("BLOCKED" if (self.readiness_authority_enabled
                                  or self.repair_blocks_done_enabled)
                    else READY_WITH_COUNTEREXAMPLES)
        return "READY"

    @property
    def blocked_by_checks(self) -> tuple[str, ...]:
        rows = (*self.counterexample_authority, *self.repair_blocking_counterexamples)
        return tuple(dict.fromkeys(str(item.get("check_id", item.get("obligation_id", "")))
                     for item in rows
                     if item.get("check_id", item.get("obligation_id", ""))))

    def submission(self) -> dict[str, object]:
        row: dict[str, object] = {
            "state": self.submission_state,
            "blocking": (list(self.open_ids)
                          if (not self.green or self.open_counterexample_ids) else []),
            "authority": "terminal_status",
        }
        if ((self.readiness_authority_enabled or self.repair_blocks_done_enabled)
                and self.blocked_by_checks):
            row["authority"] = ("readiness_authority"
                                 if self.readiness_authority_enabled
                                 else "repair_blocks_done")
            row["blocking"] = list(dict.fromkeys([*row["blocking"], *self.blocked_by_checks]))
        if self.counterexample_visibility_enabled and self.counterexamples:
            row[COUNTEREXAMPLE_PUBLIC_FIELD] = copy.deepcopy(list(self.counterexamples))
        if self.open_counterexample_ids:
            from kernel_green_menu_open_obligation import submission_blocked_text
            row["message"] = submission_blocked_text(self.open_counterexample_ids)
        if self.next_feasible_action is not None:
            row["next_feasible_action"] = self.next_feasible_action
        return row

    def lock_text(self) -> str:
        selectors = json.dumps(list(self.selectors), ensure_ascii=False, separators=(",", ":"))
        revision = json.dumps(self.current_revision, ensure_ascii=False, separators=(",", ":"), default=str)
        obligations = (json.dumps(list(self.open_ids), ensure_ascii=False, separators=(",", ":"))
                       if self.open_ids else "none")
        counterexample_names = []
        if self.counterexample_visibility_enabled:
            counterexample_names = [
                str(item.get("check_id", item.get("obligation_id", "")))
                for item in self.counterexamples
                if item.get("check_id", item.get("obligation_id", ""))]
        if counterexample_names and not self.open_ids:
            # Do not claim that the open-work list is empty on the same line
            # as a live current observation.  The advisory list is the repair
            # frontier even when completion admission was demoted.
            obligations = "[]"
        if self.no_host_checks:
            result = ("NO HOST CHECKS: no public host check exists for this task; "
                      "completion may be accepted with that limitation disclosed; "
                      "requirement evidence or model-owned checks remain available")
        elif self.green:
            result = ("GREEN LOCK: public checks green for selectors " + selectors
                      + " at revision " + revision + "; open obligations " + obligations)
        else:
            result = ("GREEN LOCK: public checks not green for selectors " + selectors
                      + " at revision " + revision + "; open obligations " + obligations)
        if counterexample_names:
            result += "; " + COUNTEREXAMPLE_LOCK_LABEL + " " + json.dumps(
                counterexample_names, ensure_ascii=False, separators=(",", ":"))
        return result

    def header_lines(self) -> list[str]:
        lines = [
            "public_check_status: " + self.public_check_status,
            "public_check_revision: " + str(self.current_revision),
        ]
        if self.counterexample_visibility_enabled:
            lines.append("terminal_status: " + self.state)
        return lines

    def footer_lines(self) -> list[str]:
        lines = ["OPEN WORK: " + json.dumps(list(self.open_ids), ensure_ascii=False,
                                               separators=(",", ":")),
                 "SUBMISSION: " + self.submission_state]
        if ((self.readiness_authority_enabled or self.repair_blocks_done_enabled)
                and self.blocked_by_checks):
            from kernel_readiness_authority import BLOCKED_BY_LABEL
            lines.append(BLOCKED_BY_LABEL + ": " + json.dumps(
                list(self.blocked_by_checks), ensure_ascii=False, separators=(",", ":")))
        elif self.open_counterexample_ids:
            from kernel_green_menu_open_obligation import submission_blocked_text
            lines.append(submission_blocked_text(self.open_counterexample_ids))
        if self.next_feasible_action is not None:
            lines.append("next_feasible_action: " + self.next_feasible_action)
        return lines

    def to_dict(self) -> dict[str, object]:
        row = {
            "schema": "stateframe-terminal-status/1",
            "state": self.state,
            "green": self.green,
            "no_host_checks": self.no_host_checks,
            "host_check_state": NO_HOST_CHECKS if self.no_host_checks else "REQUIRED_PUBLIC_CHECKS",
            "public_check_status": self.public_check_status,
            "verdict_state": self.verdict_state,
            "current_revision": self.current_revision,
            "receipt_revision": self.receipt_revision,
            "selectors": list(self.selectors),
            "open_ids": list(self.open_ids),
            "blocking": copy.deepcopy(list(self.blocking)),
            "submission": self.submission(),
            "next_feasible_action": self.next_feasible_action,
            "lock_text": self.lock_text(),
        }
        if self.counterexample_visibility_enabled:
            row.update({
                COUNTEREXAMPLES_FIELD: copy.deepcopy(list(self.counterexamples)),
                COUNTEREXAMPLE_PUBLIC_FIELD: copy.deepcopy(list(self.counterexamples)),
                "counterexample_ids": list(counterexample_ids(self.counterexamples)),
                "counterexample_authority": copy.deepcopy(list(self.counterexample_authority)),
                "repair_action": copy.deepcopy(self.repair_action),
            })
        if self.repair_blocks_done_enabled:
            row["repair_blocks_done"] = {
                "blocking_counterexamples": copy.deepcopy(
                    list(self.repair_blocking_counterexamples)),
                "bound": 6,
            }
        if self.open_counterexample_ids:
            row.setdefault("counterexample_ids", list(self.open_counterexample_ids))
        return row

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def terminal_status(runtime: object | None = None, frame: Mapping[str, object] | None = None,
                    *, verdict: object = None, receipts: Sequence[Mapping[str, object]] | None = None,
                    completion: object = None) -> TerminalStatus:
    source = frame if isinstance(frame, Mapping) else {}
    required = _required(runtime, source)
    rows = list(receipts) if receipts is not None else _receipt_rows(runtime, source)
    authority = verdict if verdict is not None else _verdict(runtime, source, required, rows)
    verdict_state = _verdict_state(authority)
    lock_green = _lock_green(runtime, source)
    no_host_checks = not required or verdict_state == NO_HOST_CHECKS
    green = False if no_host_checks else (lock_green if lock_green is not None
                                          else verdict_state == "SATISFIED")
    decision = _mapping(completion) if completion is not None else _decision_row(runtime, source)
    open_ids, blocking = _open_ids(runtime, source, decision)
    counterexample_visibility_enabled = _counterexample_visibility_enabled(runtime)
    readiness_authority_enabled = False
    repair_blocks_done_enabled = (
        getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"
        or (getattr(runtime, "stateframe_repair_blocks_done_mode", None) is None
            and os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"))
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
            or repair_blocks_done_enabled):
        # The authority never runs blind to counterexamples.
        readiness_authority_enabled = counterexample_visibility_enabled = True
    counterexamples = _counterexamples(runtime, source)
    current = _revision(_binding(runtime, source))
    if current is None:
        current = _frame_revision(source)
    repair_blocking_counterexamples: tuple[Mapping[str, object], ...] = ()
    if repair_blocks_done_enabled:
        try:
            from kernel_repair_blocks_done import blocking_rows
            repair_blocking_counterexamples = blocking_rows(
                counterexamples, revision=current, active=True)
        except (ImportError, TypeError, ValueError):
            repair_blocking_counterexamples = ()
    alignment_repairable = False
    if (os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on"
            and not repair_blocks_done_enabled):
        try:
            from kernel_requirement_alignment import is_repairable_literal_counterexample
            alignment_repairable = any(
                is_repairable_literal_counterexample(item) for item in counterexamples)
        except (ImportError, TypeError, ValueError):
            alignment_repairable = False
    elif repair_blocks_done_enabled:
        alignment_repairable = bool(repair_blocking_counterexamples)
    authoritative_counterexamples = tuple(
        item for item in counterexamples if bool(item.get("counterexample_authority")))
    counterexample_values: tuple[str, ...] = ()
    if runtime_enabled(runtime) and not readiness_authority_enabled:
        try:
            from kernel_green_menu_open_obligation import (
                blocking_counterexample_ids as gm_counterexample_ids,
            )
            counterexample_values = gm_counterexample_ids(runtime, source)
        except (ImportError, TypeError, ValueError, AttributeError):
            counterexample_values = ()
    for ident in (() if readiness_authority_enabled else counterexample_values):
        if ident not in open_ids:
            open_ids.append(ident)
        if not any(_assessment_id(item) == ident for item in blocking):
            blocking.append({
                "schema": "stateframe-open-counterexample/1",
                "check_id": ident,
                "obligation_id": ident,
                "status": "COUNTEREXAMPLE",
            })
    # A stale reducer decision must not overwrite a current passing receipt;
    # only explicit host blockers remain terminal blockers in that case.
    if no_host_checks and not blocking:
        submission_state = NO_HOST_CHECKS
        next_action = "done"
    elif green and not blocking:
        submission_state = (("BLOCKED" if (readiness_authority_enabled
                                             or repair_blocks_done_enabled)
                             else READY_WITH_COUNTEREXAMPLES)
                            if (authoritative_counterexamples
                                or repair_blocking_counterexamples) else "READY")
        next_action = (REPAIR_ACTION if (authoritative_counterexamples or alignment_repairable)
                       else "done")
    else:
        submission_state = ("BLOCKED" if (
                                not green or counterexample_values
                                or repair_blocking_counterexamples)
                            else "NEEDS_WORK")
        # Green but blocked only by open counterexamples: the recovery is repair.
        next_action = REPAIR_ACTION if (
            green and (counterexample_values or repair_blocking_counterexamples)) else None
    receipt_revision = None
    for row in reversed(rows):
        candidate = _revision(row.get("after_binding", row.get("binding")))
        if candidate is not None:
            receipt_revision = candidate
            break
    if no_host_checks:
        public_status = NO_HOST_CHECKS
    elif green:
        public_status = "PASS"
    elif verdict_state == "VIOLATED":
        public_status = "FAIL"
    else:
        public_status = "BLOCKED"
    repair_action = None
    repair_rows = authoritative_counterexamples or repair_blocking_counterexamples
    if repair_rows:
        target = repair_rows[0]
        target_file = str(target.get("target_file", ""))
        if target_file:
            repair_action = {
                "tool": "edit", "arguments": {"path": target_file},
                "reason": "repair the current counterexample",
            }
    return TerminalStatus(
        green=bool(green), public_check_status=public_status,
        verdict_state=verdict_state, current_revision=current,
        selectors=tuple(required), open_ids=tuple(open_ids),
        blocking=tuple(blocking), submission_state=submission_state,
        next_feasible_action=next_action, receipt_revision=receipt_revision,
        counterexamples=tuple(counterexamples), repair_action=repair_action,
        counterexample_visibility_enabled=counterexample_visibility_enabled,
        readiness_authority_enabled=readiness_authority_enabled,
        repair_blocks_done_enabled=repair_blocks_done_enabled,
        repair_blocking_counterexamples=repair_blocking_counterexamples,
        open_counterexample_ids=counterexample_values,
        no_host_checks=no_host_checks,
    )


def _row_selector(row: Mapping[str, object]) -> str:
    for key in ("selector", "target", "suite", "test", "check_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    values = row.get("selectors")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return ",".join(str(item) for item in values)
    return "__unkeyed__"


def _row_revision(row: Mapping[str, object]) -> object:
    for key in ("revision", "current_revision", "product_revision", "artifact_revision"):
        if row.get(key) is not None:
            return row[key]
    return _revision(row.get("after_binding", row.get("binding")))


def _row_order(row: Mapping[str, object], position: int) -> tuple[int, int, int]:
    def number(*keys: str) -> int:
        for key in keys:
            value = row.get(key)
            if type(value) is int:
                return value
        return -1
    return (number("sequence", "step", "turn", "test_turn"),
            number("attempt", "index"), position)


def _row_selectors(row: Mapping[str, object]) -> tuple[str, ...]:
    values = row.get("selectors", row.get("selector", row.get("target")))
    return tuple(_selectors(values))


def _selectors_match_required(selectors: Sequence[str], required: Sequence[str]) -> bool:
    if not required:
        return True
    if set(selectors).intersection(required):
        return True
    base_selectors = {str(item).split("@", 1)[0] for item in selectors}
    base_required = {str(item).split("@", 1)[0] for item in required}
    return bool(base_selectors.intersection(base_required))


def _normalize_current_test_row(row: dict[str, object], *, required: Sequence[str],
                                current_row: bool) -> dict[str, object]:
    """Expose a bound zero-exit public check as PASS on the coherent surface."""
    if not current_row or row.get("exit_code") != 0:
        return row
    selectors = _row_selectors(row)
    if not _selectors_match_required(selectors, required):
        return row
    explicit = [str(row.get(key, "")).upper()
                for key in ("result", "result_class", "status", "verdict", "outcome")
                if row.get(key) is not None]
    if any(value in {"FAIL", "FAILED", "RED", "ERROR", "VIOLATED", "BLOCKED"}
           or value.endswith("_FAIL") or value.endswith("_RED") for value in explicit):
        return row
    row["result"] = "PASS"
    for key in ("result_class", "status", "verdict"):
        if key in row and str(row[key]).upper() in {"UNKNOWN", "UNRESOLVED", "NOT_RUN", "PENDING"}:
            row[key] = "PASS"
    return row


def _revision_alias(frame: Mapping[str, object], current: object) -> object:
    for key in ("current_revision_alias", "revision_alias", "product_revision_alias"):
        if frame.get(key) not in (None, ""):
            return frame[key]
    aliases = frame.get("revision_aliases")
    if isinstance(aliases, Mapping):
        for key in (current, str(current)):
            if key in aliases:
                return aliases[key]
    return current


def collapse_test_rows(frame: Mapping[str, object], *, current_revision: object = None,
                       receipts: object = ()) -> dict[str, object]:
    """Keep the newest current row per selector and label its real revision.

    Older rows remain available as bounded stale evidence when no current row
    exists, but they are never allowed to masquerade as the current check.
    """
    public = copy.deepcopy(dict(frame))
    current = current_revision if current_revision is not None else _frame_revision(public)
    alias = _revision_alias(public, current)
    required = tuple(_required(None, public))
    for key in _TEST_KEYS:
        value = public.get(key)
        if isinstance(value, Mapping):
            if any(field in value for field in (
                    "id", "check_id", "selector", "selectors", "status", "result",
                    "verdict", "exit_code")):
                rows = [dict(value)]
                mapping_row = True
            else:
                # ``latest_receipts`` is already keyed by check kind.  Keep
                # that public shape while normalising each current row.
                mapped = {}
                for subkey, item in value.items():
                    row = dict(item) if isinstance(item, Mapping) else item
                    if isinstance(row, Mapping):
                        revision = _row_revision(row)
                        if (current is not None and revision is not None
                                and str(revision) == str(current)):
                            row = dict(row)
                            row["revision"] = alias
                            row["revision_alias"] = alias
                            row = _normalize_current_test_row(
                                row, required=required, current_row=True)
                    mapped[subkey] = row
                public[key] = mapped
                continue
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows = [dict(row) for row in value if isinstance(row, Mapping)]
            mapping_row = False
        else:
            continue
        if not rows:
            continue
        selected: dict[str, tuple[tuple[int, int, int], dict[str, object], bool]] = {}
        for position, row in enumerate(rows):
            revision = _row_revision(row)
            current_row = (current is not None and revision is not None
                           and str(revision) == str(current))
            if current_row:
                row["revision"] = alias
                row["revision_alias"] = alias
                row = _normalize_current_test_row(
                    row, required=required, current_row=True)
            selector = _row_selector(row)
            rank = (1 if current_row else 0, *_row_order(row, position))
            prior = selected.get(selector)
            if prior is None or rank >= prior[0]:
                selected[selector] = (rank, row, current_row)
        ordered = [item[1] for item in sorted(selected.values(), key=lambda item: item[0])]
        public[key] = (ordered[0] if mapping_row and ordered else {}) if mapping_row else ordered
    return public


def _filter_open_value(value: object, *, hidden: set[str], visible: set[str] | None = None) -> object:
    visible = visible or set()
    if isinstance(value, Mapping):
        if any(key in value for key in ("id", "obligation_id", "status", "verdict", "reason")):
            row = dict(value)
            ident = _assessment_id(row)
            if (ident in hidden
                    or (_is_unknown_assessment(row) and not _assessment_blocking(row)
                        and ident not in visible)):
                return None
            return row
        result = {}
        for key, item in value.items():
            if str(key) in hidden:
                continue
            child = _filter_open_value(item, hidden=hidden, visible=visible)
            if child is not None:
                result[key] = child
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            child = _filter_open_value(item, hidden=hidden, visible=visible)
            if child is not None:
                result.append(child)
        return result
    return value


def _cohere_result_value(value: object, *, public_status: str,
                         required: Sequence[str]) -> object:
    if not isinstance(value, Mapping):
        return value
    row = copy.deepcopy(dict(value))
    predicate = str(row.get("predicate", "")).lower()
    nested = row.get("value")
    if isinstance(nested, Mapping) and predicate in {"test_result", "host_check_result"}:
        nested = copy.deepcopy(dict(nested))
        if nested.get("exit_code") == 0 and public_status == "PASS":
            nested["result"] = "PASS"
        row["value"] = nested
    selectors = _row_selectors(row)
    if row.get("exit_code") == 0 and public_status == "PASS" \
            and _selectors_match_required(selectors, required):
        row["result"] = "PASS"
        if str(row.get("status", "")).upper() in {"UNKNOWN", "UNRESOLVED", "NOT_RUN", "PENDING"}:
            row["status"] = "PASS"
    return row


def _cohere_result_collection(value: object, *, public_status: str,
                              required: Sequence[str]) -> object:
    if isinstance(value, Mapping):
        if any(key in value for key in ("predicate", "value", "exit_code", "selector", "selectors")):
            return _cohere_result_value(value, public_status=public_status, required=required)
        return {key: _cohere_result_collection(item, public_status=public_status,
                                                required=required)
                for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_cohere_result_collection(item, public_status=public_status,
                                           required=required) for item in value]
    return value


def project_frame(frame: Mapping[str, object], *, runtime: object | None = None,
                  lock: object | None = None) -> dict[str, object]:
    """Apply the shared status to every public control surface in a frame."""
    from kernel_terminal_surface_unify import (
        ADVISORY_OBLIGATIONS_FIELD, REFUSAL_LEAD,
    )
    source = copy.deepcopy(dict(frame))
    runtime_rows = []
    if runtime is not None:
        runtime_rows = [row for row in _rows(getattr(runtime, "test_receipts", ()))
                        if isinstance(row, Mapping)]
        if runtime_rows:
            source_rows = _rows(source.get("test_receipts"))
            source["test_receipts"] = [*source_rows, *copy.deepcopy(runtime_rows)]
    public = collapse_test_rows(source, current_revision=_revision(_binding(runtime, source)),
                                receipts=_receipt_rows(runtime, source))
    status = terminal_status(runtime, public)
    readiness = None
    if (runtime is not None
            and (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
                 or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"
                 or (getattr(runtime, "stateframe_repair_blocks_done_mode", None) is None
                     and os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"))):
        method = getattr(runtime, "readiness_decision", None)
        if callable(method):
            try:
                readiness = method(public)
            except (AttributeError, TypeError, ValueError):
                readiness = None
    alignment_repairable = False
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
        try:
            from kernel_requirement_alignment import is_repairable_literal_counterexample
            alignment_repairable = any(
                is_repairable_literal_counterexample(item) for item in status.counterexamples)
        except (ImportError, TypeError, ValueError):
            alignment_repairable = False
    if lock is not None and getattr(runtime, "green_lock", None) is None:
        # A small fixture may pass a lock without installing it on runtime.
        try:
            status = terminal_status(runtime, public)
        except (AttributeError, TypeError, ValueError):
            pass
    hidden: set[str] = set()
    advisory: list[dict[str, object]] = []
    visible_counterexample_ids = set(counterexample_ids(status.counterexamples))
    for item in _host_assessments(runtime, public):
        is_advisory = False
        if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
            try:
                from kernel_green_lock_repair import is_advisory_assessment
                is_advisory = is_advisory_assessment(item)
            except (ImportError, TypeError, ValueError):
                is_advisory = False
        if is_advisory or (_is_unknown_assessment(item) and not _assessment_blocking(item)):
            ident = _assessment_id(item)
            if ident and ident not in visible_counterexample_ids:
                hidden.add(ident)
            row = _mapping(item)
            if not row and ident:
                row = {"obligation_id": ident}
            if row:
                row.setdefault("status", "ADVISORY")
                row.setdefault("advisory_verdict", str(
                    row.get("verdict", row.get("state", "UNKNOWN"))).upper())
                advisory.append(row)
    for item in _rows(public.get(ADVISORY_OBLIGATIONS_FIELD)):
        if item not in advisory:
            advisory.append(item)
    for key in ("open_work", "pending_multi_definition_edits", "unresolved_obligations"):
        if key in public:
            public[key] = _filter_open_value(
                public[key], hidden=hidden, visible=visible_counterexample_ids)
    for key in ("obligation_assessments", "assessments", "obligations"):
        if key in public:
            public[key] = _filter_open_value(
                public[key], hidden=hidden, visible=visible_counterexample_ids)
    closure = public.get("host_closure")
    if isinstance(closure, Mapping):
        closure = dict(closure)
        for key in ("assessments", "open_work", "open_obligations"):
            if key in closure:
                closure[key] = _filter_open_value(
                    closure[key], hidden=hidden, visible=visible_counterexample_ids)
        public["host_closure"] = closure
    work_record = public.get("work_record")
    if isinstance(work_record, Mapping):
        work_record = dict(work_record)
        for key in ("open_obligations", "obligations", "open_work"):
            if key in work_record:
                work_record[key] = _filter_open_value(
                    work_record[key], hidden=hidden, visible=visible_counterexample_ids)
        public["work_record"] = work_record
    if advisory:
        public[ADVISORY_OBLIGATIONS_FIELD] = advisory
    else:
        public.pop(ADVISORY_OBLIGATIONS_FIELD, None)
    public_status = status.public_check_status
    for key in ("host_facts", "current_results", "test_results", "latest_test_results"):
        if key in public:
            public[key] = _cohere_result_collection(
                public[key], public_status=public_status, required=status.selectors)
    for key in ("verification", "current_public_checks"):
        value = public.get(key)
        if isinstance(value, Mapping):
            value = copy.deepcopy(dict(value))
            value["status"] = public_status
            value["result"] = public_status
            value["revision"] = status.current_revision
            value["selectors"] = list(status.selectors)
            public[key] = value
    contract = public.get("completion_contract")
    if isinstance(contract, Mapping):
        contract = copy.deepcopy(dict(contract))
        current_checks = contract.get("current_public_checks")
        if isinstance(current_checks, Mapping):
            current_checks = copy.deepcopy(dict(current_checks))
            current_checks["status"] = public_status
            current_checks["result"] = public_status
            current_checks["revision"] = status.current_revision
            current_checks["selectors"] = list(status.selectors)
            contract["current_public_checks"] = current_checks
        contract["requirements_review_status"] = (
            "COMPLETE" if status.green else "REQUIRED")
        contract["completion_submission_status"] = status.submission_state
        contract["next_feasible_action"] = status.next_feasible_action
        public["completion_contract"] = contract
    public["requirements_review_status"] = "COMPLETE" if status.green else "REQUIRED"
    public["completion_submission_status"] = status.submission_state
    public["next_feasible_action"] = status.next_feasible_action
    if readiness is not None:
        public["readiness"] = readiness.to_dict()
        if alignment_repairable:
            # r39 single authority: the repair prompt follows the gate's repairability, so the
            # text never asks for a repair the menu does not offer.
            from kernel_requirement_alignment import is_repairable_literal_counterexample
            alignment_repairable = any(is_repairable_literal_counterexample(item)
                                       for item in readiness.repairable_advisories)
    if readiness is not None and readiness.blockers:
        from kernel_readiness_authority import BLOCKED_PENDING_DECISION
        public["pending_decision"] = BLOCKED_PENDING_DECISION
        public["next_action"] = REPAIR_ACTION
    elif alignment_repairable:
        # r39 single policy: an open ticket-literal counterexample is repairable; never "Call done".
        from kernel_requirement_alignment import suppress_done_prompt
        public["pending_decision"] = suppress_done_prompt(
            public.get("pending_decision", ""), status.counterexamples)
        public["next_action"] = REPAIR_ACTION
        resume = public.get("resume")
        if isinstance(resume, Mapping):
            resume = copy.deepcopy(dict(resume))
            resume["recommended_next_action"] = REPAIR_ACTION
            public["resume"] = resume
    elif readiness is not None and readiness.repairable_advisories:
        public["pending_decision"] = readiness.refusal_text()
        public["next_action"] = REPAIR_ACTION  # r39: never direct done while a repairable counterexample is open
    elif status.green:
        public["pending_decision"] = (readiness.refusal_text()
                                       if readiness is not None else REFUSAL_LEAD)
        public["next_action"] = "done"
        resume = public.get("resume")
        if isinstance(resume, Mapping):
            resume = copy.deepcopy(dict(resume))
            resume["recommended_next_action"] = "done"
            public["resume"] = resume
    public["terminal_status"] = status.to_dict()
    public["public_check_status"] = status.public_check_status
    public["submission"] = status.submission()
    public["open_work_ids"] = list(status.open_ids)
    if status.counterexample_visibility_enabled:
        public[COUNTEREXAMPLE_PUBLIC_FIELD] = copy.deepcopy(list(status.counterexamples))
    if status.open_ids:
        existing_rows = _rows(public.get("open_work"))
        existing_ids = {_assessment_id(row) for row in existing_rows}
        missing = [{"schema": "stateframe-open-work/1",
                    "source": "terminal_status", "obligation_id": ident}
                   for ident in status.open_ids if ident not in existing_ids]
        if missing:
            public["open_work"] = [*existing_rows, *missing]
    if status.green:
        public["green_lock_message"] = status.lock_text()
    else:
        public.pop("green_lock_message", None)
    return public


def render_terminal_facts(status: TerminalStatus, *, lock_line: bool = True,
                          include_footer: bool = True) -> list[str]:
    lines = [*status.header_lines()]
    if lock_line:
        lines.append(status.lock_text())
    if status.counterexample_visibility_enabled and status.counterexamples:
        lines.extend(render_counterexample_blocks(status.counterexamples).splitlines())
    if include_footer:
        lines.extend(status.footer_lines())
    return lines


def stall_frame_identity(frame: Mapping[str, object]) -> dict[str, object]:
    """Return the F7 logical frame without refusal/action-history churn."""
    def project(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): project(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if str(key) not in _CHURN_KEYS
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [project(item) for item in value]
        return value
    return project(frame)  # type: ignore[return-value]


def refusal_override(lock: object) -> str | None:
    if getattr(getattr(lock, "runtime", None),
               "stateframe_green_coherence_mode", "off") == "on":
        from kernel_green_menu_open_obligation import counterexample_ids as open_ids
        from kernel_green_menu_open_obligation import refusal_override as obligation_override
        narrowed = obligation_override(lock)
        if open_ids(getattr(lock, "runtime", None)):
            return narrowed
    if (getattr(lock, "_host_refusal_count", 0)
            or (getattr(lock, "_refused_action_kind", None) == "run_tests"
                and getattr(lock, "_coherence_enabled", lambda: False)())):
        return "done"
    value = getattr(lock, "_override_action_kind", None)
    return str(value) if value else None


__all__ = [
    "K_CONSECUTIVE_REFUSALS", "MODE_ENV", "READY_WITH_COUNTEREXAMPLES", "TerminalStatus",
    "collapse_test_rows",
    "enabled", "green_now", "mode", "project_frame", "refusal_override", "render_enabled",
    "render_terminal_facts", "runtime_enabled", "stall_frame_identity", "terminal_status",
]
