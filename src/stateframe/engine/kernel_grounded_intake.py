"""RF10: first-turn grounded intake projection.

The module consumes only public intake facts.  It does not classify vague text,
ask a question, or create a second menu owner.  ``StateFrameRuntime`` applies
the returned projection after its existing menu authorities have composed the
dispatch surface, and LEANMODE renders the same bounded fact line.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_FL_GROUNDED_INTAKE"
SCHEMA = "stateframe-fl-grounded-intake/1"
DISCOVERY_MENU = ("read", "grep")
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_UNKNOWN_CHECKS = frozenset({"", "UNKNOWN", "UNRUN", "NOT_RUN", "NOT RUN", "PENDING"})
_CHECKED_STATES = frozenset({
    "PASS", "PASSED", "SATISFIED", "SUCCESS", "OK", "FAIL", "FAILED",
    "ERROR", "RED", "GREEN", "VIOLATED", "UNSATISFIED",
})
_PROVENANCE_KEYS = frozenset({
    "request", "prior_clarification", "clarification", "public_discovery",
    "discovery", "checks", "public_intake_state",
})
_PUBLIC_STATE_MARKER = "[Public intake state]"
_UNRESOLVED_LOCATOR = "No concrete public locator is resolved by the request yet."
_ENTRY_POINT_RE = re.compile(r"The request names this public entry point:\s*(.+?)\.")
_UNRESOLVED_FIELD_RE = re.compile(r"unresolved material field:\s*([^;\n]+)", re.IGNORECASE)


class GroundedIntakeError(ValueError):
    """Raised when the RF10 flag or a public fact contract is malformed."""


@dataclass(frozen=True)
class IntakeFacts:
    """Sanitized public facts used by the RF10 predicate."""

    grounded: bool
    decision_index: int | None
    checks_run: bool | None
    checks_state: str
    referents_unresolved: bool | None
    unresolved_field: str | None
    discovery_referents: tuple[str, ...]
    authorized_creation: bool | None
    environment_fault: bool
    provenance: tuple[str, ...]
    source: str
    schema: str


@dataclass(frozen=True)
class IntakeDecision:
    """The one RF10 decision consumed by menu/text projections."""

    facts: IntakeFacts
    active: bool
    unresolved: bool


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise GroundedIntakeError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_fl_grounded_intake_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _TRUTHY
    return mode() == "on"


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        if isinstance(converted, Mapping):
            return converted
    return None


def _value(row: Mapping[str, object] | None, *names: str) -> object:
    if row is None:
        return None
    for name in names:
        if name in row:
            return row[name]
    return None


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if isinstance(value, str) and value.strip():
        return (value,)
    return ()


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _bool_value(row: Mapping[str, object] | None, *names: str) -> bool | None:
    value = _value(row, *names)
    return value if type(value) is bool else None


def _decision_index(runtime: object, row: Mapping[str, object] | None) -> int | None:
    if row is not None:
        for name in ("decision_index", "decision_idx", "turn_index"):
            if name in row:
                value = row[name]
                return value if type(value) is int and value >= 0 else None
    for candidate in (
        getattr(runtime, "decision_index", None),
        getattr(runtime, "turn_index", None),
        getattr(getattr(runtime, "owner", None), "current_step", None),
    ):
        if type(candidate) is int and candidate >= 0:
            return candidate
    return None


def _checks(row: Mapping[str, object] | None) -> tuple[bool | None, str]:
    explicit = _bool_value(row, "checks_run", "checks_executed", "public_checks_run")
    checks_value = _value(row, "checks_state", "check_state", "checks_status")
    checks = _mapping(_value(row, "public_checks"))
    if checks_value is None:
        checks_value = _value(checks, "state", "status", "verdict")
    status = _text(checks_value).upper()
    if explicit is not None:
        return explicit, status
    if status in _UNKNOWN_CHECKS:
        return False, status or "UNKNOWN"
    if status in _CHECKED_STATES:
        return True, status
    return None, status


def _referent_state(row: Mapping[str, object] | None) -> tuple[bool | None, str | None]:
    unresolved = _bool_value(
        row,
        "referents_unresolved",
        "referent_unresolved",
        "unresolved_referent",
    )
    resolved = _bool_value(
        row,
        "resolved_public_locator",
        "public_locator_resolved",
        "target_resolved",
    )
    if unresolved is None and resolved is not None:
        unresolved = not resolved
    field = _value(
        row,
        "unresolved_field",
        "material_field",
        "material_request_field",
    )
    if type(field) is str:
        field_text = _text(field) or None
    elif field is True:
        field_text = "material request field"
    else:
        field_text = None
    return unresolved, field_text


def _candidate_state(runtime: object, frame: object | None) -> tuple[Mapping[str, object] | None, str]:
    for name in ("public_intake_state", "grounded_intake_state", "intake_state"):
        row = _mapping(getattr(runtime, name, None))
        if row is not None:
            return row, f"runtime.{name}"
    frame_row = _mapping(frame)
    if frame_row is not None:
        for name in ("public_intake_state", "grounded_intake_state", "intake_state", "public_state"):
            row = _mapping(frame_row.get(name))
            if row is not None:
                return row, f"frame.{name}"
    return None, ""


def _message_state(messages: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or _PUBLIC_STATE_MARKER not in content:
            continue
        checks_run = False if "No public checks have run" in content else None
        referents_unresolved = True if _UNRESOLVED_LOCATOR in content else None
        if _ENTRY_POINT_RE.search(content) or "Selected public artifact:" in content:
            referents_unresolved = False
        field_match = _UNRESOLVED_FIELD_RE.search(content)
        return {
            "schema": "stateframe-public-intake-message/1",
            "grounded": True,
            "checks_run": checks_run,
            "checks_state": "UNKNOWN" if checks_run is False else None,
            "referents_unresolved": referents_unresolved,
            "unresolved_field": field_match.group(1).strip() if field_match else None,
            "provenance": {
                "request": "public intake state message",
                "checks": "public intake state message",
            },
        }
    return None


def _provenance(row: Mapping[str, object] | None) -> tuple[str, ...]:
    value = _value(row, "provenance", "fact_provenance", "sources")
    source = _mapping(value)
    if source is not None:
        return tuple(sorted(
            str(key) for key in source
            if str(key).strip().lower() in _PROVENANCE_KEYS
        ))
    return tuple(
        item for item in (_text(item).lower() for item in _items(value))
        if item in _PROVENANCE_KEYS
    )


def facts(
    runtime: object | None = None,
    frame: object | None = None,
    messages: Sequence[Mapping[str, object]] = (),
) -> IntakeFacts:
    """Extract only explicit public intake facts; missing facts stay unknown."""
    host = runtime if runtime is not None else object()
    row, source = _candidate_state(host, frame)
    if row is None:
        row = _message_state(messages)
        source = "messages.public_intake_state" if row is not None else ""
    if row is None:
        return IntakeFacts(
            False, _decision_index(host, None), None, "", None, None, (), None,
            False, (), "", "",
        )

    grounded = _bool_value(row, "grounded", "publicly_grounded")
    provenance = _provenance(row)
    if grounded is None:
        grounded = bool(provenance)
    else:
        grounded = grounded and bool(provenance)
    checks_run, checks_state = _checks(row)
    referents_unresolved, unresolved_field = _referent_state(row)
    referents = tuple(
        value for value in (_text(item) for item in _items(_value(row, "discovery_referents", "public_referents")))
        if value
    )
    authorized_creation = _bool_value(
        row,
        "authorized_creation",
        "creation_authorized",
        "request_authorizes_creation",
    )
    environment_fault = bool(_bool_value(row, "environment_fault", "host_environment_fault") is True)
    return IntakeFacts(
        bool(grounded),
        _decision_index(host, row),
        checks_run,
        checks_state,
        referents_unresolved,
        unresolved_field,
        referents,
        authorized_creation,
        environment_fault,
        provenance,
        source,
        _text(_value(row, "schema")),
    )


def decide(
    runtime: object | None = None,
    frame: object | None = None,
    messages: Sequence[Mapping[str, object]] = (),
) -> IntakeDecision:
    current = facts(runtime, frame, messages)
    unresolved = current.referents_unresolved is True or bool(current.unresolved_field)
    active = (
        enabled(runtime)
        and current.grounded
        and current.decision_index == 0
        and current.checks_run is False
        and not current.environment_fault
        and current.authorized_creation is not True
        and (unresolved or current.referents_unresolved is False)
    )
    return IntakeDecision(current, active, unresolved)


def _tool_name(tool: object) -> str:
    row = _mapping(tool)
    if row is None:
        return ""
    direct = row.get("name")
    if direct:
        return str(direct)
    function = _mapping(row.get("function"))
    return str(function.get("name", "")) if function else ""


def _names(tools: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(name for name in (_tool_name(tool) for tool in tools) if name)


def project_menu(
    selected: Sequence[Mapping[str, object]],
    decision: IntakeDecision,
) -> list[dict[str, object]]:
    """Return exact existing tool definitions in RF10's discovery order."""
    current = [copy.deepcopy(dict(tool)) for tool in selected]
    if not decision.active or not decision.unresolved:
        return current
    by_name = {_tool_name(tool): tool for tool in current if _tool_name(tool)}
    if any(name not in by_name for name in DISCOVERY_MENU):
        return current
    return [copy.deepcopy(by_name[name]) for name in DISCOVERY_MENU]


def apply_menu(
    runtime: object,
    selected: Sequence[Mapping[str, object]],
    messages: Sequence[Mapping[str, object]],
    *,
    frame: object | None = None,
) -> list[dict[str, object]]:
    """Apply RF10 after the existing menu pipeline, preserving one ownership seam."""
    decision = decide(runtime, frame, messages)
    projected = project_menu(selected, decision)
    setattr(runtime, "_grounded_intake_last_decision", decision)
    if decision.active:
        names = list(_names(projected))
        if hasattr(runtime, "completion_menu"):
            runtime.completion_menu = tuple(names)
        if hasattr(runtime, "_question_done_available"):
            runtime._question_done_available = "done" in names
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "grounded_intake_fired",
                producer="kernel_grounded_intake",
                intake_schema=SCHEMA,
                unresolved=decision.unresolved,
                checks_state=decision.facts.checks_state,
                decision_index=decision.facts.decision_index,
                offered=names,
                dispatch=False,
            )
    return projected


def presentation_line(
    runtime: object | None = None,
    frame: object | None = None,
    messages: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render the D+RF10 line from public facts, or an empty string when inactive."""
    decision = decide(runtime, frame, messages)
    if not decision.active:
        return ""
    referents = ", ".join(decision.facts.discovery_referents) or "none supplied"
    if decision.unresolved:
        if "vague" not in decision.facts.schema.lower():
            return (
                "RF10 grounded intake: public intake facts show failed checks with no "
                "changed-file locator resolved; unresolved material field: target or "
                "desired behavior is not resolved in this first turn; read/grep first, "
                "with no edit yet."
            )
        field = decision.facts.unresolved_field or "target or desired behavior"
        if field == "target or desired behavior":
            suffix = (
                "is not supplied by the request"
                if "vague" in decision.facts.schema.lower()
                else "is not resolved in this first turn"
            )
            field = f"{field} {suffix}"
        return (
            "RF10 grounded intake: public read/grep discovery found concrete referents: "
            f"{referents}; unresolved material field: {field}; read/grep first, with no edit yet."
        )
    return (
        "RF10 grounded intake: public read/grep discovery found concrete referents: "
        f"{referents}; the request is explicit, so direct productive action is allowed."
    )


def append_off_notice(
    runtime: object,
    messages: Sequence[Mapping[str, object]],
    *,
    frame: object | None = None,
) -> list[dict[str, object]]:
    """Add RF10 to the public intake message when LEANMODE is disabled."""
    line = presentation_line(runtime, frame, messages)
    copied = [copy.deepcopy(dict(message)) for message in messages]
    if not line or any(line in str(message.get("content", "")) for message in copied):
        return copied
    target = next(
        (
            index
            for index, message in enumerate(copied)
            if message.get("role") == "user" and _PUBLIC_STATE_MARKER in str(message.get("content", ""))
        ),
        None,
    )
    if target is None:
        target = next(
            (index for index in range(len(copied) - 1, -1, -1) if copied[index].get("role") == "user"),
            None,
        )
    if target is None:
        return copied
    content = str(copied[target].get("content", ""))
    copied[target]["content"] = f"{content}\n{line}" if content else line
    return copied


__all__ = [
    "DISCOVERY_MENU", "GroundedIntakeError", "IntakeDecision", "IntakeFacts", "MODE_ENV",
    "SCHEMA", "append_off_notice", "apply_menu", "decide", "enabled", "facts", "mode",
    "presentation_line", "project_menu",
]
