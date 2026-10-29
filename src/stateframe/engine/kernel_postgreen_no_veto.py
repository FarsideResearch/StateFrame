"""Single post-green no-veto owner for frame product work (r45 F1 bet 1).

With ``STATEFRAME_POSTGREEN_NO_VETO=on`` and a terminal lock engaged after
public green, product ``edit``/``str_replace``/``read`` actions are admitted
for every product path. The only remaining host refusal is one ``done``
refusal per RUN, and only when the host can name
an executable counterexample: a named failing check or a SIGNATURE_GAP. An untouched named layer is NOT an
executable counterexample. Continuation (unresolved-effect) serialization, runner capability,
path containment, protected-file and malformed-edit checks stay with their owners. Green-lock, repair, readiness, and named-layer callers defer
to this module instead of growing parallel post-green policy.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace


MODE_ENV = "STATEFRAME_POSTGREEN_NO_VETO"
PLUGGED_KEY = "stateframe_postgreen_no_veto"
ADMITTED_EVENT = "postgreen_no_veto_admitted"
DONE_REFUSED_EVENT = "postgreen_no_veto_done_refused"
RULE_ADMITTED = "postgreen_no_veto"
RULE_DONE_ALLOWED = "postgreen_no_veto_done_allowed"
RULE_DONE_REFUSED = "postgreen_no_veto_done_refused"

EDIT_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
READ_ACTIONS = frozenset({"read"})
PRODUCT_ACTIONS = EDIT_ACTIONS | READ_ACTIONS

_FAILURE_VALUES = frozenset({
    "FAIL", "FAILED", "ERROR", "BROKEN", "BLOCKED", "COUNTEREXAMPLE",
    "SIGNATURE_GAP",
})
_IDENTITY_KEYS = (
    "check_id", "obligation_id", "clause_id", "requirement_id", "id", "check",
)
_TARGET_KEYS = (
    "target_file", "target_path", "file", "path", "target_symbol", "symbol",
)
_NESTED_KEYS = (
    "counterexample", "assessment", "receipt", "observed", "observed_values",
    "detail", "targets", "failure", "evidence", "receipts",
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The owner result consumed by green-lock authority and fallback guards."""

    allowed: bool
    rule: str
    message: str | None = None
    witness: Mapping[str, object] | None = None


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_postgreen_no_veto_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _runtime(lock: object) -> object | None:
    return getattr(lock, "runtime", None)


def _public_green(lock: object) -> bool:
    return (getattr(lock, "mode", "off") == "terminal"
            and bool(getattr(lock, "terminal_engaged", False)))


def _emit(runtime: object | None, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_postgreen_no_veto", dispatch=False, **fields)


def _revision(lock: object) -> str:
    value = getattr(lock, "_green_revision", None)
    return "<unknown>" if value is None else str(value)


def _path(arguments: Mapping[str, object] | None, path: object = None) -> object:
    if path is not None:
        return path
    if isinstance(arguments, Mapping):
        return arguments.get("path")
    return None


def _is_product_path(path: object) -> bool:
    from kernel_green_lock import is_product_path
    return bool(is_product_path(path))


def _state(runtime: object | None) -> dict[str, object]:
    state = getattr(runtime, "_postgreen_no_veto_state", None)
    if not isinstance(state, dict):
        state = {"done_refused_revisions": []}
        try:
            setattr(runtime, "_postgreen_no_veto_state", state)
        except (AttributeError, TypeError):
            pass
    return state


def _row(value: object) -> dict[str, object]:
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
    return {}


def _walk_rows(value: object, parent: Mapping[str, object] | None = None,
               depth: int = 0) -> list[dict[str, object]]:
    """Flatten host evidence while retaining parent check/path identity."""
    current = _row(value)
    if not current:
        return []
    merged = {**(dict(parent) if isinstance(parent, Mapping) else {}), **current}
    rows = [merged]
    if depth >= 2:
        return rows
    for key in _NESTED_KEYS:
        nested = current.get(key)
        if isinstance(nested, Mapping):
            rows.extend(_walk_rows(nested, merged, depth + 1))
        elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            for item in nested:
                if isinstance(item, Mapping):
                    rows.extend(_walk_rows(item, merged, depth + 1))
    return rows


def _evidence_rows(lock: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        snapshot = lock._readiness_snapshot()
    except (AttributeError, TypeError, ValueError, OSError):
        snapshot = None
    for name in ("blockers", "advisory", "repairable_advisories"):
        values = getattr(snapshot, name, ()) if snapshot is not None else ()
        for value in values or ():
            rows.extend(_walk_rows(value))
    if not rows:
        try:
            values = lock._repair_evidence()
        except (AttributeError, TypeError, ValueError, OSError):
            values = ()
        for value in values or ():
            rows.extend(_walk_rows(value))
    return rows


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _upper(value: object) -> str:
    return _text(value).upper().replace("-", "_")


def _first_text(row: Mapping[str, object], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, Mapping):
            nested = _first_text(value, ("id", "name", "value", "status", "path", "file"))
            if nested:
                return nested
        elif _text(value):
            return _text(value)
    return ""


def _check_id(row: Mapping[str, object]) -> str:
    value = _first_text(row, _IDENTITY_KEYS)
    if not value:
        values = row.get("check_ids")
        if isinstance(values, str):
            value = values.strip()
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            value = next((_text(item) for item in values if _text(item)), "")
    return value if value.lower() not in {"unknown", "none", "null"} else ""


def _target(row: Mapping[str, object]) -> str:
    direct = _first_text(row, _TARGET_KEYS)
    if direct:
        return direct
    for name in ("files", "file_paths", "untouched_files"):
        values = row.get(name)
        if isinstance(values, str) and values.strip():
            return values.strip()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for value in values:
                if _text(value):
                    return _text(value)
    targets = row.get("targets")
    if isinstance(targets, Mapping):
        values = targets.get("paths") or targets.get("files") or ()
        if isinstance(values, str):
            return values.strip()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for value in values:
                if _text(value):
                    return _text(value)
    return ""


def _signature_gap(row: Mapping[str, object]) -> str:
    reason = _upper(row.get("demotion_reason") or row.get("reason"))
    classification = _upper(row.get("classification"))
    binding = _upper(row.get("binding"))
    status = _upper(row.get("status"))
    detail = _first_text(row, ("signature_gap", "binding_error", "arity_error"))
    if ("SIGNATURE_GAP" in {reason, classification, binding, status}
            or bool(row.get("signature_gap"))):
        return detail or "resolved entry point rejects the ticket call signature"
    return ""


def _failing_check(row: Mapping[str, object]) -> bool:
    status = _upper(_first_text(row, (
        "status", "verdict", "result_class", "result", "outcome", "execution", "reason",
    )))
    return (bool(row.get("counterexample_authority"))
            or status in _FAILURE_VALUES
            or bool(row.get("executable_counterexample")))


def executable_counterexample(lock: object) -> Mapping[str, str] | None:
    """Return only a named, executable witness suitable for one done refusal."""
    for row in _evidence_rows(lock):
        check_id = _check_id(row)
        target = _target(row)
        if not check_id or not target:
            continue
        signature_gap = _signature_gap(row)
        if signature_gap:
            return {"check_id": check_id, "target": target, "kind": "SIGNATURE_GAP",
                    "detail": signature_gap}
        if _failing_check(row):
            detail = _first_text(row, ("error", "message", "clause", "expected", "detail",
                                       "diff_from_start"))
            return {"check_id": check_id, "target": target, "kind": "failing_check",
                    "detail": detail or "the named check is still failing"}
    return None


def _done_verdict(lock: object, *, for_menu: bool) -> Verdict:
    if for_menu:
        return Verdict(True, RULE_DONE_ALLOWED)
    witness = executable_counterexample(lock)
    if witness is None:
        return Verdict(True, RULE_DONE_ALLOWED)
    revision = _revision(lock)
    state = _state(_runtime(lock))
    refused = state.setdefault("done_refused_revisions", [])
    if not isinstance(refused, list):
        refused = []
        state["done_refused_revisions"] = refused
    if refused:   # run-wide: the host refuses done at most ONCE per run
        return Verdict(True, RULE_DONE_ALLOWED)
    refused.append(revision)
    message = (f"done refused once: executable counterexample {witness['check_id']} at "
               f"{witness['target']} ({witness['kind']}): {witness['detail']}. "
               "Apply the repair and retry done.")
    _emit(_runtime(lock), DONE_REFUSED_EVENT, action="done", product_revision=revision,
          check_id=witness["check_id"], target=witness["target"], witness_kind=witness["kind"],
          detail=witness["detail"])
    return Verdict(False, RULE_DONE_REFUSED, message, witness)


def decide(lock: object, action: object, arguments: Mapping[str, object] | None = None,
           *, path: object = None, for_menu: bool = False) -> Verdict | None:
    """Return this owner's verdict, or ``None`` when legacy policy still owns it."""
    runtime = _runtime(lock)
    if not enabled(runtime) or not _public_green(lock):
        return None
    name = _norm(action)
    if name == "done":
        return _done_verdict(lock, for_menu=for_menu)
    if name not in PRODUCT_ACTIONS:
        return None
    target = _path(arguments, path)
    if not for_menu:
        # The runner's write guard passes absolute paths; judge the workspace-relative path like the legacy gates
        # (a path outside the workspace keeps its legacy owner -- containment is never waived).
        workspace_path = getattr(lock, "_workspace_path", None)
        if callable(workspace_path) and target not in (None, ""):
            try:
                target = workspace_path(target)
            except (OSError, TypeError, ValueError):
                return None
        if not _is_product_path(target):
            return None
    witness = {"path": _text(target)} if target is not None else None
    revision = _revision(lock)
    if not for_menu:
        _emit(runtime, ADMITTED_EVENT, action=name, path=_text(target), product_revision=revision,
              stage="acted")
    return Verdict(True, RULE_ADMITTED, witness=witness)


def defer_readiness(decision: object, lock: object) -> object:
    """Mark readiness product dispositions as delegated to this owner."""
    runtime = _runtime(lock)
    if not enabled(runtime) or not _public_green(lock):
        return decision
    dispositions = dict(getattr(decision, "dispositions", {}) or {})
    for name in ("edit", "str_replace", "read"):
        dispositions[name] = "allow"
    try:
        return replace(decision, dispositions=dispositions, post_green_no_veto=True)
    except TypeError:
        return replace(decision, dispositions=dispositions)


__all__ = [
    "ADMITTED_EVENT", "EDIT_ACTIONS", "DONE_REFUSED_EVENT", "MODE_ENV", "PLUGGED_KEY",
    "PRODUCT_ACTIONS", "READ_ACTIONS", "RULE_ADMITTED", "RULE_DONE_ALLOWED",
    "RULE_DONE_REFUSED", "Verdict", "decide", "defer_readiness", "enabled",
    "executable_counterexample", "mode",
]
