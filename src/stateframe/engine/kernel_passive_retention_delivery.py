"""Journal-derived proof that passive conclusions reached a later provider request.

The retention store owns capture and bounded rendering.  This module owns only
the opt-in delivery measurement, and deliberately counts the serialized
``request_submitted`` boundary rather than a preceding render intention.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_PASSIVE_RETENTION_DELIVERY"
SCHEMA = "stateframe-passive-retention-delivery/1"


def mode() -> str:
    """Return the strict opt-in switch value for delivery accounting."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be 'off' or 'on'")
    return value


def enabled() -> bool:
    return mode() == "on"


def _short_reference(value: str) -> str:
    return value if len(value) <= 48 else "…" + value[-47:]


def _journal_events(source: object) -> list[Mapping[str, object]]:
    """Read a journal path/object or accept already-decoded event records."""
    path: Path | None = None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
    else:
        candidate = getattr(source, "path", None)
        if isinstance(candidate, (str, os.PathLike)):
            path = Path(candidate)
    if path is not None:
        from kernel_measurement import resolve_journal_blob_refs

        events: list[Mapping[str, object]] = []
        blob_cache: dict[str, object] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (TypeError, ValueError, UnicodeError) as error:
                    raise ValueError(f"invalid journal JSON at line {line_number}") from error
                if not isinstance(event, Mapping):
                    raise ValueError(f"journal event at line {line_number} is not an object")
                try:
                    resolved = resolve_journal_blob_refs(event, path, blob_cache)
                except (OSError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid journal blob at line {line_number}") from error
                if not isinstance(resolved, Mapping):
                    raise ValueError(
                        f"journal event at line {line_number} did not resolve to an object"
                    )
                events.append(resolved)
        return events
    if isinstance(source, Mapping):
        return [source]
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
        return [event for event in source if isinstance(event, Mapping)]
    raise TypeError("delivery metric requires a journal path, journal, or event iterable")


def _sequence(event: Mapping[str, object], fallback: int) -> int:
    value = event.get("sequence")
    return value if type(value) is int and value >= 0 else fallback


def _message_contents(event: Mapping[str, object]) -> list[str]:
    payload = event.get("payload")
    messages = payload.get("messages") if isinstance(payload, Mapping) else event.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []
    return [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, Mapping)
    ]


def _passive_block_text(event: Mapping[str, object], header: str) -> str:
    blocks = []
    for content in _message_contents(event):
        if header in content:
            blocks.append(content[content.index(header):])
    return "\n".join(blocks)


def _records_from_state(state: object) -> list[Mapping[str, object]]:
    if not isinstance(state, Mapping):
        return []
    rows = state.get("work_items")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    records: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        conclusions = row.get("conclusions")
        if not isinstance(conclusions, Sequence) or isinstance(
                conclusions, (str, bytes, bytearray)):
            continue
        for conclusion in conclusions:
            if isinstance(conclusion, Mapping):
                records.append(conclusion)
    return records


def delivery_metric(source: object) -> dict[str, object]:
    """Derive passive-conclusion delivery from journal state and provider events.

    A record is delivered only when its response reference appears in a later
    submitted payload that contains the passive-conclusion block.  A
    ``request_prepared`` event is intentionally insufficient: it proves a
    renderer produced a candidate, not that the provider received it.
    """
    from kernel_passive_retention import PASSIVE_HEADER

    events = _journal_events(source)
    stored: dict[str, dict[str, object]] = {}
    retention_state: Mapping[str, object] | None = None
    submitted: list[tuple[int, str, str]] = []
    for index, event in enumerate(events, 1):
        event_sequence = _sequence(event, index)
        if event.get("kind") == "passive_retention_state":
            state = event.get("state")
            if isinstance(state, Mapping):
                retention_state = state
            else:
                delta = event.get("state_delta")
                if isinstance(delta, Mapping):
                    from kernel_passive_retention import apply_snapshot_delta
                    retention_state = apply_snapshot_delta(retention_state, delta)
            for record in _records_from_state(retention_state):
                response_ref = record.get("response_ref")
                if not isinstance(response_ref, str) or not response_ref:
                    continue
                record_id = record.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    record_id = f"{response_ref}:{len(stored)}"
                stored.setdefault(record_id, {
                    "sequence": event_sequence,
                    "response_ref": response_ref,
                    "text": record.get("text"),
                })
        if event.get("kind") == "request_submitted":
            # Restrict body matching to messages containing the rendered passive
            # block. Earlier or later ordinary history cannot masquerade as delivery.
            text = _passive_block_text(event, PASSIVE_HEADER)
            if text:
                request_id = event.get("request_id")
                submitted.append((event_sequence, str(request_id or event.get("event_id", "")), text))

    delivered: set[str] = set()
    reference_only: set[str] = set()
    provider_requests: set[str] = set()
    for record_id, record in stored.items():
        response_ref = str(record["response_ref"])
        marker = f"response={_short_reference(response_ref)}"
        for event_sequence, request_id, text in submitted:
            if event_sequence <= int(record["sequence"]) or marker not in text:
                continue
            provider_requests.add(request_id)
            body = record.get("text")
            if isinstance(body, str) and body and body in text:
                delivered.add(record_id)
            else:
                reference_only.add(record_id)
            break

    retained = len(stored)
    delivered_count = len(delivered)
    return {
        "schema": SCHEMA,
        "retained_conclusions": retained,
        "delivered_conclusions": delivered_count,
        "reference_only_conclusions": len(reference_only),
        "provider_requests_with_retention": len(provider_requests),
        "delivery_rate": (delivered_count / retained if retained else 0.0),
    }
