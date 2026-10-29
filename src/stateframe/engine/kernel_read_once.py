"""Read-once note at the first repeated read of an unchanged range (soak CONFIRMED lead C3).

Qualification (multi-turn read-loop anchor, real r2j journal cut after the 3rd unchanged
re-read, exact token-parity placebo): T 6/12, C 2/12, P 1/12. Timing matters: at the 6th
re-read it was weak, so the note fires at the FIRST repeat, once per run.

Placement: appended to that read's tool result as a ``[kernel]`` note (the established
host-note position; a separate user message between native tool results would break
tool-call ordering). The text is verbatim. Language-generic: it names no language or task.

Default ON (soak 10-10 transfer check, tool_result placement): benchmark T4/8 vs C1/8 P1/8,
benchmark neutral, benchmark/benchmark real journals contain no unchanged pre-edit repeat so it cannot
fire there (no harm path). Kill switch: STATEFRAME_READ_ONCE=off.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os

ENV = "STATEFRAME_READ_ONCE"
MESSAGE = ("Do not re-read a line range you have already read unless you edited that file "
           "since; the earlier tool result is still valid.")
NOTE = "[kernel] " + MESSAGE
_REPEAT_STATUSES = frozenset({"UNCHANGED_VISIBLE", "CACHED_BYTES", "REEXECUTED"})
_NOTED_FIELD = "_read_once_noted"


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    return value if value in {"on", "off"} else "off"


def adviser_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Read-once advice is hidden in LEAN so delivered history stays the authority."""
    values = os.environ if environ is None else environ
    from kernel_lean import enabled as lean_enabled
    return mode(values) == "on" and not lean_enabled(values)


PLACEMENT_ENV = "STATEFRAME_READ_ONCE_PLACEMENT"
_PENDING_FIELD = "_read_once_pending"


_REPEATS_FIELD = "_read_once_repeats"
_FINAL_PENDING_FIELD = "_read_once_final_pending"
_WIRE_PENDING_FIELD = "_read_once_wire_pending"
_SEEN_READS_FIELD = "_read_once_seen_ranges"
_SEEN_READ_LIMIT = 128
FINAL_AT_REPEAT = 3


def placement(environ: Mapping[str, str] | None = None) -> str:
    """tool_result: suffix on the repeated read's result (screened: benchmark T4/8 vs C1/8).
    user_message: separate user message after the tool batch (screened: no effect; opt-in only).
    final_message_at_3: tool_result at the first repeat PLUS one bare-sentence FINAL message
    (after all host wire notes) at the 3rd unchanged repeat (screened per stage: benchmark 3rd
    repeat T8/20 vs C3/20 P2/20; the two-stage pair itself is untested)."""
    values = os.environ if environ is None else environ
    value = str(values.get(PLACEMENT_ENV, "tool_result") or "tool_result").strip().lower()
    return value if value in {"tool_result", "user_message", "final_message_at_3"} else "tool_result"


def _read_identity(delivery: object, source_path: object) -> str | None:
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    if not isinstance(dedup, Mapping) or source_path is None:
        return None
    revision = dedup.get("revision")
    if not isinstance(revision, str) or not revision.startswith("sha256:"):
        return None
    try:
        if len(bytes.fromhex(revision[7:])) != 32:
            return None
        path = str(Path(source_path).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    interval = dedup.get("effective_range")
    if not isinstance(interval, Mapping):
        return None
    start, end = interval.get("start_line"), interval.get("end_line")
    if (type(start) is not int or type(end) is not int
            or start < 1 or end < start):
        return None
    identity = json.dumps(
        [path, revision, start, end], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _repeat_detected(
    runtime: object, delivery: object, source_path: object
) -> bool:
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    if not isinstance(dedup, Mapping):
        return False
    from kernel_gate_switches import deliver_requested_bytes_enabled
    if deliver_requested_bytes_enabled():
        reuse = delivery.get("read_reuse") if isinstance(delivery, Mapping) else None
        proof = reuse.get("proof") if isinstance(reuse, Mapping) else None
        delivered_bytes = proof.get("delivered_bytes") if isinstance(proof, Mapping) else None
        return bool(
            delivery.get("delivery_status") == "REUSED_PROVEN"
            and isinstance(reuse, Mapping)
            and reuse.get("status") in {"CACHED_BYTES", "UNCHANGED_VISIBLE"}
            and isinstance(proof, Mapping)
            and proof.get("wire_contains_exact") is True
            and bool(str(proof.get("final_message_location") or "").strip())
            and type(delivered_bytes) is int and delivered_bytes > 0
        )
    if dedup.get("status") in _REPEAT_STATUSES:
        return True
    if dedup.get("status") not in {"EXECUTED", "CACHE_DELIVERED"}:
        return False
    identity = _read_identity(delivery, source_path)
    if identity is None:
        return False
    seen = getattr(runtime, _SEEN_READS_FIELD, ())
    if not isinstance(seen, (tuple, list)):
        seen = ()
    repeated = identity in seen
    if not repeated:
        try:
            setattr(runtime, _SEEN_READS_FIELD, (*seen, identity)[-_SEEN_READ_LIMIT:])
        except (AttributeError, TypeError):
            pass
    return repeated


def _observe_noread(
    runtime: object, delivery: object, source_path: object, *, repeat_confirmed: bool
) -> None:
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    if not isinstance(dedup, Mapping):
        return
    from kernel_noread import arm_on_repeat, observe_read

    observe_read(runtime, delivery, source_path=source_path)
    arm_on_repeat(
        runtime, delivery, source_path=source_path,
        repeat_confirmed=repeat_confirmed,
    )


def note_for(
    runtime: object, delivery: object, *, source_path: object = None
) -> str | None:
    """Process unchanged-repeat policy and return the optional read-once note.

    Noread is an independent opt-in; it observes this detector even when the
    legacy read-once note is disabled.
    """
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    status = dedup.get("status") if isinstance(dedup, Mapping) else None
    identity = _read_identity(delivery, source_path)
    repeated = _repeat_detected(runtime, delivery, source_path)
    if isinstance(dedup, Mapping):
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        capture = getattr(runtime, "_active_capture", None)
        if callable(emit):
            emit(
                "read_repeat_assessed",
                producer="kernel_read_once",
                dispatch=False,
                request_id=getattr(capture, "request_id", None),
                trigger_status=status,
                repeat_detected=repeated,
                identity_sha256=identity,
            )
    _observe_noread(
        runtime, delivery, source_path, repeat_confirmed=repeated)
    if not adviser_enabled() or not repeated:
        return None
    try:
        repeats = int(getattr(runtime, _REPEATS_FIELD, 0) or 0) + 1
        setattr(runtime, _REPEATS_FIELD, repeats)
        if placement() == "final_message_at_3" and repeats == FINAL_AT_REPEAT:
            setattr(runtime, _FINAL_PENDING_FIELD, True)
    except (AttributeError, TypeError, ValueError):
        return None
    if getattr(runtime, _NOTED_FIELD, False):
        return None
    try:
        setattr(runtime, _NOTED_FIELD, True)
        if placement() == "user_message":
            setattr(runtime, _PENDING_FIELD, True)
            return None
    except (AttributeError, TypeError):
        return None
    try:
        setattr(runtime, _WIRE_PENDING_FIELD, True)
    except (AttributeError, TypeError):
        pass
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    capture = getattr(runtime, "_active_capture", None)
    if callable(emit):
        emit(
            "read_once_note_queued",
            producer="kernel_read_once",
            dispatch=False,
            request_id=getattr(capture, "request_id", None),
            placement="tool_result",
            wire_pending=bool(getattr(runtime, _WIRE_PENDING_FIELD, False)),
        )
    return NOTE


def take_pending_message(runtime: object) -> dict[str, str] | None:
    """Pop the queued user-message note; call only when every tool call has its result."""
    if not getattr(runtime, _PENDING_FIELD, False):
        return None
    try:
        setattr(runtime, _PENDING_FIELD, False)
    except (AttributeError, TypeError):
        return None
    return {"role": "user", "content": NOTE}


def prepare_messages(runtime: object, messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Runtime prepare hook (the frame is re-rendered there, so raw appends are lost).

    A queued note becomes ONE trailing user message; a message list that ends with a tool
    call still awaiting results is left untouched and the note stays queued.
    """
    if not adviser_enabled() or not getattr(runtime, _PENDING_FIELD, False) or not messages:
        return messages
    last = messages[-1]
    if isinstance(last, Mapping) and last.get("role") == "assistant" and last.get("tool_calls"):
        return messages
    note = take_pending_message(runtime)
    return messages if note is None else [*messages, note]


def append_final_message(runtime: object, messages: object) -> object:
    """Send-time hook after the host wire notes: append the queued bare final message once."""
    if (not adviser_enabled() or placement() != "final_message_at_3"
            or not getattr(runtime, _FINAL_PENDING_FIELD, False) or not isinstance(messages, list)):
        return messages
    try:
        setattr(runtime, _FINAL_PENDING_FIELD, False)
    except (AttributeError, TypeError):
        return messages
    return [*messages, {"role": "user", "content": MESSAGE}]


def ensure_wire_note(
    runtime: object | None, messages: object
) -> object:
    """Keep the first repeat-read note in the serialized request after frame budgeting."""
    if (not adviser_enabled() or not getattr(runtime, _WIRE_PENDING_FIELD, False)
            or not isinstance(messages, list)):
        return messages
    if (messages and isinstance(messages[-1], Mapping)
            and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls")):
        return messages
    present = any(
        isinstance(row, Mapping)
        and isinstance(row.get("content"), str)
        and NOTE in row["content"]
    for row in messages
    )
    result = messages if present else [
        *messages, {"role": "user", "content": NOTE},
    ]
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    capture = getattr(runtime, "_active_capture", None)
    if callable(emit):
        emit(
            "read_once_wire_note_reconciled",
            producer="kernel_read_once",
            dispatch=False,
            request_id=getattr(capture, "request_id", None),
            result="ALREADY_PRESENT" if present else "INSERTED",
        )
    return result


def confirm_wire_delivery(runtime: object | None, receipt: object) -> bool:
    """Clear a queued note only after its exact request bytes reached transport."""
    if not getattr(runtime, _WIRE_PENDING_FIELD, False) or not isinstance(receipt, Mapping):
        return False
    body_sha256 = receipt.get("body_sha256")
    if not isinstance(body_sha256, str) or not body_sha256.startswith("sha256:"):
        return False
    try:
        if len(bytes.fromhex(body_sha256[7:])) != 32:
            return False
        setattr(runtime, _WIRE_PENDING_FIELD, False)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def final_wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Authenticate the bare final message as the LAST message of the exact bytes."""
    if not adviser_enabled():
        return None
    try:
        messages = json.loads(wire).get("messages")
    except (TypeError, ValueError, AttributeError):
        return None
    if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "user" or messages[-1].get("content") != MESSAGE):
        return None
    return {"message_index": len(messages) - 1, "placement": "final_message_at_3",
            "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest()}


def wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Authenticate the note in exact serialized request bytes (one per request carrying it)."""
    if not adviser_enabled():
        return None
    try:
        messages = json.loads(wire).get("messages")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(messages, list):
        return None
    indexes = [index for index, row in enumerate(messages)
               if isinstance(row, Mapping) and isinstance(row.get("content"), str)
               and NOTE in row["content"]]
    if len(indexes) != 1:
        return None
    return {"message_index": indexes[0], "role": messages[indexes[0]].get("role"),
            "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest()}
