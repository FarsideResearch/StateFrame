"""Small, host-filled just-in-time clearing lines for StateFrame gates."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_JIT_CLEARING_LINES"
SCHEMA = "stateframe-jit-clearing-lines/1"
_KEYS = (
    "open_obligation", "typed_witness_request", "witness_io_contract",
    "first_receipt", "first_done_refusal", "host_only_binding_failure",
    "completion_route", "advisory_degraded",
)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_jit_clearing_lines_mode", None) if runtime is not None else None
    return (str(value).strip().lower() == "on" if value is not None else mode() == "on")


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _field(value: object, *names: str) -> str:
    for name in names:
        candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if candidate not in (None, "", [], {}):
            return " ".join(str(candidate).split())
    return ""


_REVISION_KEYS = ("product_revision", "artifact", "revision", "content_digest")
_MAX_REVISION_CHARS = 96


def _revision_identity(receipt: object) -> str:
    """A short revision identity for a receipt, never a dump of its binding.

    L125b oracles: ``str(binding)`` put every harness-file hash (124 KB) into
    one clearing line, the request exceeded the hard ceiling 3x and the relay
    dropped it.  Bindings are mappings; only their revision identity is shown.
    """
    for name in ("binding", "after_binding"):
        value = receipt.get(name) if isinstance(receipt, Mapping) else getattr(receipt, name, None)
        if isinstance(value, Mapping):
            for key in _REVISION_KEYS:
                if value.get(key) not in (None, "", [], {}):
                    return str(value[key])[:_MAX_REVISION_CHARS]
    plain = _field(receipt, "revision", "product_revision")
    return plain[:_MAX_REVISION_CHARS]


def _assessments(runtime: object) -> tuple[object, ...]:
    rows: list[object] = []
    for source_name in ("host_closure_assessments", "behavior_witness_assessments", "obligation_assessments"):
        source = getattr(runtime, source_name, ())
        for item in _sequence(source):
            rows.append(item)
    return tuple(rows)


def _receipts(runtime: object) -> tuple[object, ...]:
    rows: list[object] = []
    for source_name in ("test_receipts", "probe_receipts", "observation_probe_receipts"):
        source = getattr(runtime, source_name, ())
        for item in _sequence(source):
            rows.append(item)
    return tuple(rows)


def _first_unresolved(runtime: object) -> object | None:
    from kernel_completion_owner import _assessment_is_context_only

    for row in _assessments(runtime):
        if _assessment_is_context_only(row):
            continue
        verdict = _field(row, "verdict", "status").upper()
        if verdict not in {"PASS", "PASSED", "SATISFIED", "GREEN", "COMPLETED"}:
            if _field(row, "obligation_id", "id") and _field(row, "clause", "requirement", "description"):
                return row
    return None


def _action_text(action: Mapping[str, object]) -> str:
    tool = _field(action, "tool")
    arguments = action.get("arguments")
    if tool == "probe" and isinstance(arguments, Mapping):
        clause_id = _field(arguments, "clause_id")
        entrypoint = _field(arguments, "entrypoint")
        if clause_id and entrypoint:
            return (f'probe(clause_id="{clause_id}", entrypoint="{entrypoint}", '
                    "input=INPUT_OBJECT, expected=EXPECTED_OBJECT, "
                    'rationale="derive from the requirement")')
    return tool + "()" if tool else "the offered clearing action"


def _candidate_lines(runtime: object) -> dict[str, str]:
    from kernel_completion_owner import (
        advisory_degraded_notice,
        completion_route_decision,
        completion_route_notice,
    )

    lines: dict[str, str] = {}
    route = completion_route_decision(
        runtime, getattr(runtime, "_last_completion_decision", None),
        tools=getattr(runtime, "_completion_owner_full_tools", ()),
    )
    row = _first_unresolved(runtime)
    actions = route.actions
    if row is not None and route.reason_category != "EVIDENCE_GAP":
        identifier = _field(row, "obligation_id", "id")
        requirement = _field(row, "clause", "requirement", "description")
        evidence = _field(row, "missing", "evidence", "reason")
        action = next((item for item in actions if _field(item, "tool") != "done"), None)
        if identifier and requirement and evidence and action is not None:
            lines["open_obligation"] = (
                f"Open {identifier}: {requirement}. Missing: {evidence}. "
                f"To supply it: {_action_text(action)}."
            )
        entrypoint = _field(row, "entrypoint", "public_entrypoint")
        input_shape = _field(row, "input_shape", "input_schema")
        expected_shape = _field(row, "expected_shape", "expected_schema", "result_shape")
        if identifier and entrypoint and any(_field(item, "tool") == "probe" for item in actions):
            lines["typed_witness_request"] = (
                f'Submit one case with probe(clause_id="{identifier}", entrypoint="{entrypoint}", '
                "input=INPUT_OBJECT, expected=EXPECTED_OBJECT, "
                'rationale="derive this expectation from the requirement"); the host runs it and binds '
                "the result to this revision."
            )
        if entrypoint and input_shape and expected_shape:
            lines["witness_io_contract"] = (
                f"For {entrypoint}, input is {input_shape}; expected is {expected_shape}; "
                "compare the full result, not a partial assertion."
            )

    receipts = _receipts(runtime)
    if receipts:
        receipt = receipts[0]
        receipt_id = _field(receipt, "receipt_id", "id")
        action = _field(receipt, "tool", "kind", "action")
        result = _field(receipt, "result", "result_class", "status", "outcome")
        revision = _revision_identity(receipt)
        if receipt_id and action and result and revision:
            lines["first_receipt"] = (
                f"Receipt {receipt_id} records {action}/{result} for {revision}; "
                "cite this receipt for that observation, not as proof of other requirements."
            )

    route_notice = completion_route_notice(runtime, route)
    if route_notice:
        lines["completion_route"] = route_notice
    degraded_notice = advisory_degraded_notice(runtime)
    if degraded_notice:
        lines["advisory_degraded"] = degraded_notice
    if route.route == "HOST_BLOCKER" and any(
            token in " ".join(route.reason_codes).upper()
            for token in ("BINDING", "DELIVERY", "PENDING")):
        lines["host_only_binding_failure"] = route_notice
    return lines


def append_lines(runtime: object, prepared: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Append each applicable line once, retaining host metadata across turns."""
    from kernel_completion_owner import OWNER_MESSAGE_BLOCK_KEY

    outgoing = [dict(message) for message in prepared]
    if not enabled(runtime):
        return outgoing
    rendered = getattr(runtime, "_stateframe_jit_lines_rendered", None)
    if not isinstance(rendered, set):
        rendered = set()
        setattr(runtime, "_stateframe_jit_lines_rendered", rendered)
    digests = getattr(runtime, "_stateframe_jit_line_digests", None)
    if not isinstance(digests, dict):
        digests = {}
        setattr(runtime, "_stateframe_jit_line_digests", digests)
    for key, line in _candidate_lines(runtime).items():
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        if digests.get(key) == digest or any(message.get("content") == line for message in outgoing):
            rendered.add(key)
            digests[key] = digest
            continue
        old_prefix = f"jit:{key}:"
        if digests.get(key) is not None:
            outgoing = [
                message for message in outgoing
                if not str(message.get(OWNER_MESSAGE_BLOCK_KEY, "")).startswith(old_prefix)
            ]
        outgoing.append({
            "role": "user", "content": line,
            OWNER_MESSAGE_BLOCK_KEY: f"jit:{key}:{digest}",
        })
        rendered.add(key)
        digests[key] = digest
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit("jit_clearing_line_rendered", producer="kernel_jit_clearing_lines",
                 event_schema=SCHEMA, line_kind=key, content=line, dispatch=False)
    return outgoing


__all__ = ["MODE_ENV", "SCHEMA", "append_lines", "enabled", "mode"]
