"""Fail-open handling for request revalidation drift after first wire truth."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_ONE_SNAPSHOT"
_EVENT_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,95}\Z")
_SNAPSHOT_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _safe_event_label(value: str) -> str:
    """Keep journal labels bounded and free of exception messages or paths."""
    return value if _EVENT_LABEL.fullmatch(value) else "OTHER"


def _safe_snapshot_id(value: str) -> str:
    """Only emit content-free snapshot digests in the journal."""
    return value if _SNAPSHOT_ID.fullmatch(value) else "UNKNOWN"


def enabled() -> bool:
    """Return true only for an explicit opt-in; the release default is OFF."""
    return os.environ.get(MODE_ENV, "off").strip().lower() == "on"


def snapshot_id(preparation: object, wire: bytes) -> str:
    """Bind an ID to the frozen preparation input and its serialized body."""
    digest = hashlib.sha256()
    for name in ("request_id", "step", "body", "binding", "source_plan",
                 "request_source_plan", "sources"):
        try:
            value = getattr(preparation, name)
        except (AttributeError, TypeError):
            value = None
        if value is None:
            raw = b""
        elif isinstance(value, bytes):
            raw = value
        else:
            raw = str(value).encode("utf-8", errors="replace")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    digest.update(len(wire).to_bytes(8, "big"))
    digest.update(wire)
    return "sha256:" + digest.hexdigest()


def has_sendable_body(wire: object) -> bool:
    """Only a missing or malformed serialized message list remains terminal."""
    if not isinstance(wire, bytes) or not wire:
        return False
    try:
        body = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    messages = body.get("messages") if isinstance(body, Mapping) else None
    return isinstance(messages, list) and bool(messages)


def degraded_visibility_receipt(wire: bytes) -> dict[str, object]:
    """Do not credit source delivery from a later body after yielding bytes."""
    return {
        "result": "DEGRADED",
        "missing_ranges": [],
        "delivered_sources": [],
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }


def record_disagreement(
        capture: object, *, validator: str, first_verdict: str,
        second_verdict: str, first_snapshot_id: str,
        second_snapshot_id: str, flag_enabled: bool | None = None,
) -> bool:
    """Journal one disagreement without allowing observability to block send.

    Each event is a per-run alarm input: consumers count ``revalidation_disagreement``
    rows by the journal's run ID. The exact request bytes and exception text are omitted.
    """
    try:
        first_wire = getattr(capture, "_one_snapshot_first_wire", None)
        active = enabled() if flag_enabled is None else flag_enabled
        if not active or not has_sendable_body(first_wire):
            return False
        request_id = str(getattr(capture, "request_id", ""))
        journal = getattr(capture, "journal", None)
        if journal is not None:
            try:
                journal.emit(
                    "revalidation_disagreement",
                    producer="kernel_one_snapshot_preparation",
                    request_id=request_id,
                    step=getattr(capture, "step", None),
                    validator=_safe_event_label(validator),
                    first_verdict=_safe_event_label(first_verdict),
                    second_verdict=_safe_event_label(second_verdict),
                    snapshot_ids={
                        "first": _safe_snapshot_id(first_snapshot_id),
                        "second": _safe_snapshot_id(second_snapshot_id),
                    },
                    dispatch=False,
                )
            except Exception:
                # Audit output is best-effort at this fail-open seam.
                pass
            if (str(first_verdict).upper() not in {"PASS", "DISABLED"}
                    or str(second_verdict).upper() not in {"PASS", "DISABLED"}):
                try:
                    journal.emit(
                        "wire_invariant_violated",
                        producer="kernel_one_snapshot_preparation",
                        request_id=request_id,
                        step=getattr(capture, "step", None),
                        validator=_safe_event_label(validator),
                        first_verdict=_safe_event_label(first_verdict),
                        degraded_validation=_safe_event_label(validator),
                        dispatch=False,
                    )
                except Exception:
                    pass
        if not getattr(capture, "_one_snapshot_degraded_validation", None):
            try:
                setattr(capture, "_one_snapshot_degraded_validation",
                        _safe_event_label(validator))
            except Exception:
                pass
        if (str(first_verdict).upper() not in {"PASS", "DISABLED"}
                or str(second_verdict).upper() not in {"PASS", "DISABLED"}):
            try:
                setattr(capture, "_one_snapshot_wire_invariant_violated", True)
            except Exception:
                pass
        return True
    except Exception:
        # This guard must never turn an observational failure into a request failure.
        return False


__all__ = [
    "MODE_ENV", "degraded_visibility_receipt", "enabled", "has_sendable_body",
    "record_disagreement", "snapshot_id",
]
