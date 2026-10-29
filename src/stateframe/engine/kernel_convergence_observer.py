"""Universal settled-action convergence observer for the v46 frame.

The observer is deliberately advisory.  It fingerprints semantic evidence at
the settled-receipt boundary, not model prose, receipt IDs, or timestamps.  A
repeat gets one compact, actionable summary; it never grants completion and
never removes the repair route for a current RED assertion.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_CONVERGENCE_OBSERVER"
SCHEMA = "stateframe-convergence-observer/1"
EVENT = "convergence_observer_summary"
MENU_EVENT = "convergence_observer_menu_projected"
RESET_EVENT = "convergence_observer_reset"
DEFAULT_THRESHOLD = 2
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_VOLATILE_KEYS = frozenset({
    "receipt_id", "full_output_handle", "check_output_receipt", "receipt_handle",
    "event_id", "step_record_id", "request_id", "call_id", "run_id",
    "sequence", "ordinal", "turn", "step", "timestamp", "created_at", "updated_at",
    "observed_at", "started_at", "finished_at", "time", "ts",
})
_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_READ_TOOLS = frozenset({"read", "grep", "search", "locate"})
_CHECK_TOOLS = frozenset({"run_tests", "test", "pytest", "verification"})
_RED = frozenset({"RED", "FAIL", "FAILED", "ERROR", "REJECTED", "BLOCKED"})
_GREEN = frozenset({"GREEN", "PASS", "PASSED", "SUCCESS", "OK", "CERTIFIED"})
_REFUSED = frozenset({
    "REFUSED", "REFUSED_PREDISPATCH", "REJECTED", "NO_OP", "NOOP",
    "NO_EXECUTION_RECEIPT", "REJECTED_CONTRACT", "BLOCKED",
})
_ASSERTION_RE = re.compile(r"(?:assert(?:ionerror)?|E\s+assert|FAILED)\b.*", re.IGNORECASE)


def enabled(runtime: object | None = None) -> bool:
    """Return the explicit observer switch without changing the OFF path."""
    configured = getattr(runtime, "stateframe_convergence_observer_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _TRUTHY
    return str(os.environ.get(MODE_ENV, "off") or "off").strip().lower() in _TRUTHY


def threshold(runtime: object | None = None) -> int:
    configured = getattr(runtime, "stateframe_convergence_observer_threshold", None)
    raw = configured if configured is not None else os.environ.get(
        "STATEFRAME_CONVERGENCE_OBSERVER_N", str(DEFAULT_THRESHOLD))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise ValueError("STATEFRAME_CONVERGENCE_OBSERVER_N must be a positive integer") from error
    if value < 2:
        raise ValueError("STATEFRAME_CONVERGENCE_OBSERVER_N must be at least 2")
    return value


def _stable(value: object, *, key: str = "") -> object:
    """Canonicalize JSON-like evidence while dropping identity-only metadata."""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = str(raw_key)
            lowered = name.lower()
            if (lowered in _VOLATILE_KEYS or lowered.endswith("_receipt_id")
                    or lowered.endswith("_timestamp") or lowered.endswith("_timestamp_ns")
                    or lowered.endswith("_at")):
                continue
            if lowered in {"why", "reasoning", "model_text"} and key == "arguments":
                continue
            result[name] = _stable(child, key=name)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable(item, key=key) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(_stable(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _first(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value not in (None, "", (), [], {}):
            return value
    return None


def _tool(row: Mapping[str, object], action: object) -> str:
    nested = _mapping(row.get("action"))
    value = _first(row, "tool", "action_kind", "operation")
    if value is None:
        value = _first(nested, "tool", "action_kind", "operation")
    text = str(value or action or "unknown").strip().lower()
    return text.split(None, 1)[0] if text else "unknown"


def _arguments(row: Mapping[str, object]) -> Mapping[str, object]:
    nested = _mapping(row.get("action"))
    for source in (row, nested):
        for name in ("canonical_arguments", "action_parameters", "arguments", "parameters"):
            value = source.get(name)
            if isinstance(value, Mapping):
                return value
    return {}


def _status(row: Mapping[str, object], observation: object) -> str:
    value = _first(row, "result_class", "result", "status", "outcome", "disposition")
    text = str(value or observation or "UNKNOWN").strip().upper().replace("-", "_")
    return text.split(None, 1)[0] if text else "UNKNOWN"


def _text(row: Mapping[str, object], observation: object) -> str:
    values: list[str] = []
    for source in (row, _mapping(row.get("detail")), _mapping(row.get("observation"))):
        for name in ("assertion_context", "failure_excerpt", "failure_tail", "observation",
                     "output", "message", "detail", "error", "refusal_reason"):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                values.append(value)
    if isinstance(observation, str) and observation.strip():
        values.append(observation)
    return "\n".join(dict.fromkeys(values))


def _assertion(text: str) -> str:
    rows = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    matches = [match.group(0) for line in rows if (match := _ASSERTION_RE.search(line))]
    return (matches[-1] if matches else (rows[-1] if rows else "NO ASSERTION CONTEXT"))[:360]


def _part(row: Mapping[str, object], names: Sequence[str]) -> object:
    values = {name: row[name] for name in names if row.get(name) not in (None, "", (), [], {})}
    return _stable(values)


def _bindings(row: Mapping[str, object]) -> dict[str, object]:
    generic = _mapping(_first(row, "binding", "after_binding", "current_binding"))
    product = _first(row, "product_binding", "product_revision", "current_revision",
                     "source_revision", "revision")
    if product is None:
        product = generic.get("product_revision", generic.get("product")) or generic
    harness = _first(row, "harness_binding", "harness_revision", "harness")
    if harness is None:
        harness = generic.get("harness_revision", generic.get("harness"))
    environment = _first(row, "environment_binding", "environment_revision", "environment")
    if environment is None:
        environment = generic.get("environment_revision", generic.get("environment"))
    return {
        "product": _stable(product),
        "harness": _stable(harness),
        "environment": _stable(environment),
    }


def _coverage(row: Mapping[str, object]) -> object:
    coverage = _first(row, "coverage", "check_coverage", "scope", "selectors", "selector",
                      "target", "test_selector")
    return _stable(coverage)


def _spans(row: Mapping[str, object]) -> object:
    return _part(row, (
        "delivered_spans", "delivered_span", "span", "read_evidence", "source_span",
        "line", "start_line", "end_line", "column", "start_column", "end_column",
        "content_hash", "content_revision", "file_revision",
    ))


def _unresolved(runtime: object, row: Mapping[str, object]) -> object:
    value = _first(row, "unresolved_evidence", "unresolved", "open_obligation_ids",
                   "open_ids", "open_clauses", "remaining_gaps")
    if value is not None:
        return _stable(value)
    values: list[str] = []
    for source in (
            getattr(runtime, "open_obligation_ids", ()),
            getattr(runtime, "open_ids", ()),
            getattr(runtime, "unresolved_evidence", ())):
        if isinstance(source, Mapping):
            source = source.keys()
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            values.extend(str(item) for item in source if str(item).strip())
    return sorted(set(values))


def _handle(runtime: object, row: Mapping[str, object]) -> str:
    value = _first(row, "full_output_handle", "check_output_receipt", "receipt_handle")
    if value is not None:
        return str(value)
    latest = getattr(runtime, "_latest_check_receipt", None)
    if isinstance(latest, Mapping):
        value = latest.get("full_output_handle") or latest.get("receipt_id")
        if value:
            return str(value)
    return ""


def _real_change(row: Mapping[str, object], tool: str, status: str) -> bool:
    if tool not in _EDIT_TOOLS or status not in {"APPLIED", "PASS", "SUCCESS", "OK"}:
        return False
    if "byte_changed" in row:
        return bool(row["byte_changed"])
    if "file_changed" in row:
        return bool(row["file_changed"])
    before = _mapping(row.get("before_binding")).get("product_files")
    after = _mapping(row.get("after_binding")).get("product_files")
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return dict(before) != dict(after)
    changes = row.get("changes")
    if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes, bytearray)):
        return any(isinstance(item, Mapping) and item.get("before") != item.get("after")
                   for item in changes)
    return False


def _route(status: str, unresolved: object, row: Mapping[str, object], tool: str) -> tuple[str, ...]:
    supplied = _first(row, "repair_route", "clearing_route", "next_actions")
    if isinstance(supplied, str):
        supplied = (supplied,)
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes, bytearray)):
        values = tuple(dict.fromkeys(str(item) for item in supplied if str(item).strip()))
        if values:
            return values
    if status in _RED or status in _REFUSED:
        if tool in _EDIT_TOOLS:
            return ("str_replace", "edit", "read_check_receipt")
        return ("edit", "str_replace", "read_check_receipt")
    if unresolved not in (None, "", (), [], {}):
        return ("read", "probe", "done")
    return ("done",)


def _row_identity(runtime: object, row: Mapping[str, object], action: object,
                  observation: object) -> tuple[dict[str, object], str]:
    tool = _tool(row, action)
    status = _status(row, observation)
    unresolved = _unresolved(runtime, row)
    arguments = dict(_arguments(row))
    if tool == "read_check_receipt":
        # Handles identify storage, not evidence.  Validity/scope remains in
        # the settled status and assertion fields, so a rotated receipt does
        # not manufacture progress.
        arguments.pop("handle", None)
    parts = {
        "canonical_action": {"tool": tool, "arguments": _stable(arguments, key="arguments")},
        "coverage": _coverage(row),
        "bindings": _bindings(row),
        "delivered_spans": _spans(row),
        "unresolved_evidence": unresolved,
        "status": status,
        "assertion": _assertion(_text(row, observation)) if status in _RED else "",
        "repair_route": _route(status, unresolved, row, tool),
    }
    return parts, _digest(parts)


@dataclass(frozen=True)
class ConvergenceDecision:
    """One stable semantic repeat, with an honest route out of it."""

    identity_digest: str
    repeat_count: int
    tool: str
    status: str
    assertion: str
    receipt_handle: str
    unresolved_evidence: object
    clearing_route: tuple[str, ...]
    summary: str

    @property
    def red(self) -> bool:
        return self.status in _RED

    def as_dict(self) -> dict[str, object]:
        return {
            "identity_digest": self.identity_digest,
            "repeat_count": self.repeat_count,
            "tool": self.tool,
            "status": self.status,
            "assertion": self.assertion,
            "receipt_handle": self.receipt_handle,
            "unresolved_evidence": self.unresolved_evidence,
            "clearing_route": list(self.clearing_route),
            "summary": self.summary,
        }


class ConvergenceObserver:
    """Reduce settled receipts to one semantic repeat streak."""

    def __init__(self, *, repeat_threshold: int = DEFAULT_THRESHOLD) -> None:
        if repeat_threshold < 2:
            raise ValueError("repeat_threshold must be at least 2")
        self.repeat_threshold = repeat_threshold
        self.identity_digest = ""
        self.identity_parts: dict[str, object] = {}
        self.repeat_count = 0
        self.last_tool = ""
        self.last_status = "UNKNOWN"
        self.last_row: dict[str, object] = {}
        self.last_observation = ""
        self.summary: ConvergenceDecision | None = None
        self.emitted: set[str] = set()

    def _reset(self, runtime: object, *, reason: str) -> None:
        if self.identity_digest or self.repeat_count:
            _emit(runtime, RESET_EVENT, reason=reason, previous_identity=self.identity_digest)
        setattr(runtime, "_convergence_notice", None)
        self.identity_digest = ""
        self.identity_parts = {}
        self.repeat_count = 0
        self.summary = None

    def observe(self, runtime: object, row: Mapping[str, object], *, action: object = "",
                observation: object = "") -> ConvergenceDecision | None:
        parts, identity = _row_identity(runtime, row, action, observation)
        tool = _tool(row, action)
        status = _status(row, observation)
        if _real_change(row, tool, status):
            self._reset(runtime, reason="real_product_change")
        if identity != self.identity_digest:
            if self.identity_digest:
                self._reset(runtime, reason="semantic_evidence_changed")
            self.identity_digest = identity
            self.identity_parts = parts
            self.repeat_count = 1
        else:
            self.repeat_count += 1
        self.last_tool = tool
        self.last_status = status
        self.last_row = dict(row)
        self.last_observation = str(observation or "")
        if self.repeat_count < self.repeat_threshold or identity in self.emitted:
            return None
        unresolved = parts["unresolved_evidence"]
        route = tuple(str(item) for item in parts["repair_route"])
        assertion = str(parts["assertion"])
        handle = _handle(runtime, row)
        unresolved_text = json.dumps(unresolved, ensure_ascii=False, sort_keys=True, default=str)
        route_text = " | ".join(route) or "no clearing route recorded"
        receipt_text = f"; receipt={handle}" if handle else "; receipt=unavailable"
        summary_text = (
            f"[CONVERGE] unchanged evidence repeat={self.repeat_count}: action={tool}; "
            f"status={status}; assertion={assertion}{receipt_text}; "
            f"unresolved={unresolved_text}; clearing_route={route_text}. "
            "Receipt IDs/timestamps are not progress."
        )
        decision = ConvergenceDecision(
            identity_digest=identity, repeat_count=self.repeat_count, tool=tool,
            status=status, assertion=assertion, receipt_handle=handle,
            unresolved_evidence=unresolved, clearing_route=route, summary=summary_text,
        )
        self.emitted.add(identity)
        self.summary = decision
        # The final presentation boundary consumes this single compact notice.
        # It is intentionally separate from STALLWIRE's legacy mechanism line
        # so a diagnostic summary cannot be silently replaced by a stall veto.
        setattr(runtime, "_convergence_notice", summary_text)
        _emit(runtime, EVENT, **decision.as_dict())
        return decision


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_convergence_observer", event_schema=SCHEMA,
             dispatch=False, **fields)


def _owner(runtime: object) -> ConvergenceObserver:
    observer = getattr(runtime, "_convergence_observer", None)
    if isinstance(observer, ConvergenceObserver):
        return observer
    observer = ConvergenceObserver(repeat_threshold=threshold(runtime))
    setattr(runtime, "_convergence_observer", observer)
    return observer


def observe_closed_action(runtime: object, action: object, observation: object,
                          *, receipt: Mapping[str, object] | None = None
                          ) -> ConvergenceDecision | None:
    """Consume one settled action when enabled; model prose is never evidence."""
    if not enabled(runtime):
        return None
    row = dict(receipt) if isinstance(receipt, Mapping) else {}
    row.setdefault("action", action)
    row.setdefault("observation", observation)
    return _owner(runtime).observe(runtime, row, action=action, observation=observation)


def decision(runtime: object) -> ConvergenceDecision | None:
    value = getattr(runtime, "_convergence_observer", None)
    return value.summary if isinstance(value, ConvergenceObserver) else None


def notice(runtime: object) -> str | None:
    current = decision(runtime)
    return current.summary if current is not None else None


def apply_menu(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Remove only the exact repeated trigger while retaining a truthful route."""
    selected = [dict(tool) for tool in tools]
    current = decision(runtime)
    if not enabled(runtime) or current is None:
        return selected
    repeated = current.tool
    if repeated == "read_check_receipt":
        # Receipt retrieval is always a legal read-only route, even when the
        # same request was refused earlier by a completion/green-lock owner.
        return selected
    candidate = [tool for tool in selected
                 if str(tool.get("function", {}).get("name", "")) != repeated]
    candidate_names = [str(tool.get("function", {}).get("name", "")) for tool in candidate]
    if current.red and repeated in _EDIT_TOOLS and not set(candidate_names) & _EDIT_TOOLS:
        candidate = selected
    if current.red and not set(candidate_names) & _EDIT_TOOLS:
        # A red assertion must retain at least one repair action.  Do not turn
        # convergence into a completion veto or an accidental done-only menu.
        candidate = selected
    if candidate == selected:
        return selected
    _emit(runtime, MENU_EVENT, repeated_tool=repeated, status=current.status,
          offered=candidate_names, repair_route=list(current.clearing_route))
    return candidate


__all__ = [
    "ConvergenceDecision", "ConvergenceObserver", "DEFAULT_THRESHOLD", "EVENT",
    "MENU_EVENT", "MODE_ENV", "RESET_EVENT", "SCHEMA", "apply_menu", "decision",
    "enabled", "notice", "observe_closed_action", "threshold",
]
