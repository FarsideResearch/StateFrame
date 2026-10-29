"""R-012: one opt-in boundary for applying real runner effects.

``kernel_effect_boundary`` owns the older organ/evaluator contract.  This module
is the small adapter for effects that cross the repair runner's external edge:
editing a workspace, running a test subprocess, and accepting a submission.
The adapter is deliberately dormant unless ``STATEFRAME_EFFECT_BOUNDARY`` is
set to ``off``, ``shadow``, or ``enforce``.  In the default path it calls the
legacy dispatcher directly and creates no receipt or render sidecar.

The enabled ordering is:

    immutable snapshot -> evaluate -> authorize -> dispatch/refuse
    -> authoritative update -> receipt -> render

Evaluation receives a recursively frozen value, so a decision cannot mutate
the live snapshot before authorization.  A dispatch exception still produces
a CRASHED receipt before the exception is re-raised; this keeps an effect cut
distinct from a missing receipt.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path


MODE_ENV = "STATEFRAME_EFFECT_BOUNDARY"
VALID_MODES = {"off", "shadow", "enforce"}
RENDER_HEADER = "[EFFECT_BOUNDARY]"


class Mode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclasses.dataclass(frozen=True)
class EffectReceipt:
    """The host-owned receipt for one effect attempt."""

    receipt_id: str
    operation: str
    mode: str
    status: str
    dispatch_started: bool
    dispatched: bool
    effect_applied: bool | None
    authoritative_updated: bool
    before_hash: str
    after_hash: str
    proposal_hash: str
    snapshot: object = ()
    error_type: str | None = None
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "effect-application-receipt/1",
            "receipt_id": self.receipt_id,
            "operation": self.operation,
            "mode": self.mode,
            "status": self.status,
            "dispatch_started": self.dispatch_started,
            "dispatched": self.dispatched,
            "effect_applied": self.effect_applied,
            "authoritative_updated": self.authoritative_updated,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "proposal_hash": self.proposal_hash,
            "snapshot": _jsonable(self.snapshot),
            "error_type": self.error_type,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass(frozen=True)
class ApplicationResult:
    """The legacy value plus the boundary outcome."""

    value: object = None
    executed: bool = False
    receipt: EffectReceipt | None = None
    rendered: str | None = None


def mode() -> Mode:
    raw = os.environ.get(MODE_ENV, "off").strip().lower() or "off"
    if raw in {"0", "false", "no"}:
        raw = "off"
    elif raw in {"1", "true", "yes"}:
        raw = "enforce"
    if raw not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off, shadow, or enforce")
    return Mode(raw)


def enabled() -> bool:
    return mode() is not Mode.OFF


def freeze_snapshot(value: object) -> object:
    """Return a recursively immutable, deterministic snapshot representation."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), freeze_snapshot(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_snapshot(item) for item in value), key=repr))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return freeze_snapshot(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return freeze_snapshot(dataclasses.asdict(value))
    if isinstance(value, (str, int, float, bool, bytes)) or value is None:
        return value
    return repr(value)


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest(value: object) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def render_receipt(receipt: EffectReceipt) -> str:
    if receipt.status in {"UNKNOWN", "CRASHED"} or receipt.effect_applied is None:
        dispatch = "UNKNOWN"
    elif receipt.effect_applied is True:
        dispatch = "APPLIED"
    elif receipt.effect_applied is False:
        dispatch = "REFUSED"
    else:
        dispatch = "UNKNOWN"
    return (
        f"{RENDER_HEADER} operation={receipt.operation} mode={receipt.mode.upper()} "
        f"status={receipt.status} dispatch={dispatch} receipt={receipt.receipt_id} "
        f"before={receipt.before_hash} after={receipt.after_hash}"
    )


def _runtime_owner(owner: object | None) -> object | None:
    if owner is not None:
        return owner
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        return ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        return None


def _publish(owner: object | None, receipt: EffectReceipt) -> None:
    sink = _runtime_owner(owner)
    if sink is None:
        return
    records = getattr(sink, "_effect_application_receipts", None)
    if records is None:
        records = sink._effect_application_receipts = []
    records.append(receipt)
    pending = getattr(sink, "_effect_application_pending", None)
    if pending is None:
        pending = sink._effect_application_pending = []
    pending.append(receipt)
    journal = getattr(sink, "journal", None) or getattr(sink, "audit_journal", None)
    if journal is not None and callable(getattr(journal, "emit", None)):
        journal.emit("effect_application_receipt", receipt=receipt.as_dict(), dispatch=False)


def _make_receipt(*, operation: str, selected_mode: Mode, status: str,
                  dispatch_started: bool, dispatched: bool,
                  effect_applied: bool | None, authoritative_updated: bool,
                  frozen_snapshot: object, before_state: object, proposal: object,
                  after_snapshot: object, error_type: str | None = None,
                  journal: object = None) -> EffectReceipt:
    from kernel_wire_identity import next_wire_id
    receipt_id = (next_wire_id(journal, "effect-application", operation,
                               selected_mode.value, status, _digest(proposal))
                  if journal is not None else uuid.uuid4().hex)
    return EffectReceipt(
        receipt_id=receipt_id,
        operation=operation,
        mode=selected_mode.value,
        status=status,
        dispatch_started=dispatch_started,
        dispatched=dispatched,
        effect_applied=effect_applied,
        authoritative_updated=authoritative_updated,
        before_hash=_digest(before_state),
        after_hash=_digest(after_snapshot),
        proposal_hash=_digest(proposal),
        snapshot=frozen_snapshot,
        error_type=error_type,
        timestamp=time.time(),
    )


def apply_effect(*, operation: str, snapshot: object, dispatch: Callable[[], object],
                 owner: object | None = None,
                 evaluate: Callable[[object], object] | None = None,
                 authoritative_update: Callable[[object], object] | None = None,
                 before_snapshot: Callable[[], object] | None = None,
                 after_snapshot: Callable[[], object] | None = None,
                 shadow_value: object = None,
                 effect_applied: Callable[[object], bool] | None = None,
                 render: Callable[[EffectReceipt], str] | None = None) -> ApplicationResult:
    """Apply one effect through OFF/SHADOW/ENFORCE.

    The OFF branch intentionally has no evaluation, snapshot, journal, or
    render work.  It is the legacy call with a small result wrapper for callers
    that need to inspect the enabled outcome.
    """
    selected_mode = mode()
    if selected_mode is Mode.OFF:
        return ApplicationResult(value=dispatch(), executed=True)

    sink = _runtime_owner(owner)
    journal = (getattr(sink, "journal", None) or getattr(sink, "audit_journal", None)
               if sink is not None else None)

    frozen_snapshot = freeze_snapshot(snapshot)
    before_state = (freeze_snapshot(before_snapshot())
                    if before_snapshot is not None else frozen_snapshot)
    proposal = evaluate(frozen_snapshot) if evaluate is not None else frozen_snapshot
    if selected_mode is Mode.SHADOW:
        receipt = _make_receipt(
            operation=operation, selected_mode=selected_mode, status="SHADOW",
            dispatch_started=False, dispatched=False, effect_applied=False,
            authoritative_updated=False,
            frozen_snapshot=frozen_snapshot, before_state=before_state,
            proposal=proposal, after_snapshot=before_state, journal=journal,
        )
        _publish(owner, receipt)
        rendered = render(receipt) if render is not None else render_receipt(receipt)
        return ApplicationResult(value=shadow_value, executed=False,
                                 receipt=receipt, rendered=rendered)

    dispatch_started = False
    dispatched = False
    applied: bool | None = None
    authoritative_updated = False
    value = None
    after = before_state
    try:
        dispatch_started = True
        value = dispatch()
        dispatched = True
        applied = True if effect_applied is None else bool(effect_applied(value))
        if applied:
            if authoritative_update is None:
                authoritative_updated = True
            else:
                authoritative_update(value)
                authoritative_updated = True
        if after_snapshot is not None:
            after = freeze_snapshot(after_snapshot())
    except Exception as error:
        # A dispatch can mutate its destination and then fail before returning.
        # Always sample the destination after a started dispatch; a changed hash
        # makes the outcome UNKNOWN instead of falsely calling it REFUSED.
        if after_snapshot is not None and dispatch_started:
            try:
                after = freeze_snapshot(after_snapshot())
            except Exception:
                after = before_state
        if dispatch_started and _digest(after) != _digest(before_state):
            status = "UNKNOWN"
        else:
            status = "CRASHED"
        receipt = _make_receipt(
            operation=operation, selected_mode=selected_mode, status=status,
            dispatch_started=dispatch_started, dispatched=dispatched,
            effect_applied=applied, authoritative_updated=authoritative_updated,
            frozen_snapshot=frozen_snapshot, before_state=before_state,
            proposal=proposal,
            after_snapshot=after, error_type=type(error).__name__, journal=journal,
        )
        _publish(owner, receipt)
        raise

    receipt = _make_receipt(
        operation=operation, selected_mode=selected_mode,
        status=("APPLIED" if applied else
                "UNKNOWN" if _digest(after) != _digest(before_state) else "REFUSED"),
        dispatch_started=dispatch_started, dispatched=dispatched,
        effect_applied=applied, authoritative_updated=authoritative_updated,
        frozen_snapshot=frozen_snapshot, before_state=before_state,
        proposal=proposal,
        after_snapshot=after, journal=journal,
    )
    _publish(owner, receipt)
    rendered = render(receipt) if render is not None else render_receipt(receipt)
    return ApplicationResult(value=value, executed=bool(applied), receipt=receipt,
                             rendered=rendered)


def append_pending_render(prepared: Sequence[dict], runtime: object,
                          *, provider_request: bool = True) -> list[dict]:
    """Append and consume receipts at the final request presentation boundary."""
    if not provider_request or mode() is Mode.OFF:
        return list(prepared)
    pending = list(getattr(runtime, "_effect_application_pending", ()))
    if not pending:
        return list(prepared)
    rendered = "\n".join(
        render_receipt(receipt) for receipt in pending
    )
    output = [*prepared, {"role": "user", "content": rendered}]
    runtime._effect_application_pending = []
    journal = getattr(runtime, "journal", None)
    if journal is not None and callable(getattr(journal, "emit", None)):
        journal.emit(
            "effect_application_rendered",
            receipt_ids=[receipt.receipt_id for receipt in pending],
            content=rendered,
            dispatch=False,
        )
    return output


__all__ = [
    "ApplicationResult", "EffectReceipt", "MODE_ENV", "Mode", "RENDER_HEADER",
    "VALID_MODES", "append_pending_render", "apply_effect", "enabled", "freeze_snapshot",
    "mode", "render_receipt",
]
