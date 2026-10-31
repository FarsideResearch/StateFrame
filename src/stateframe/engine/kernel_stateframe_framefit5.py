"""FrameFit5: replace prior assistant/tool turns with one typed StateFrame."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from kernel_stateframe_f4 import HANDOFF_HEADER, latest_complete_tool_exchange
from kernel_stateframe_envelope import STATE_HEADER, serialize_stateframe_payload
from kernel_greenfact import GREENFACT_TEXT
from kernel_wire_stability import VOLATILE_TAIL_MARKER
from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
)

SCHEMA = "stateframe-framefit5/1"
PRODUCT_MODE = "FRAME_ONLY"
MAX_SITE_LINES = 80
MAX_SITE_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 32 * 1024
MAX_OPAQUE_RECEIPT_BYTES = 4 * 1024
MAX_PRODUCT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_REFRESH_ATTEMPTS = 3


class FrameFit5Error(RuntimeError):
    """The frame-only request cannot prove its typed replacement state."""


class FrameFit5ProducerError(FrameFit5Error):
    """A StateFrame JSON payload is invalid, with its producer identified."""

    def __init__(self, producer: str) -> None:
        self.producer = producer
        super().__init__(f"frame-only StateFrame JSON is invalid; producer={producer}")


class _CandidateSourceChanged(FrameFit5Error):
    """Candidate bytes moved while a frame-only source window was being rendered."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_text(value: object) -> str:
    """Serialize structured StateFrame data without interpolating source text."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _serialize_state_message(payload: Mapping[str, object]) -> str:
    """Serialize one complete StateFrame payload after its structured fields are assembled."""
    return serialize_stateframe_payload(payload)


def _json_value(value: object) -> object:
    """Normalize a value through JSON while preserving strings and their exact characters."""
    return json.loads(_json_text(value))


def _headcarry_mode() -> str:
    value = (os.environ.get("STATEFRAME_HEADCARRY", "off") or "off").strip().lower()
    return value if value in {"off", "shadow", "on"} else "off"


def _update_headcarry(
    frame: dict[str, object],
    *,
    runtime: object,
    root: Path,
    open_sites: Sequence[object],
    applied_edits: Sequence[object],
    messages: Sequence[Mapping[str, object]],
    initial: bool = False,
) -> None:
    mode = _headcarry_mode()
    already_attached = isinstance(frame.get("headcarry"), Mapping)
    if mode == "off" and not already_attached:
        return
    from kernel_stateframe_headcarry import headcarry_owed_reads_enabled, owed_read_rows

    # T2FIX: reads the frame still owes ride the same typed carrier, so the
    # read-delivery truth can mark them present instead of "re-read to see it".
    # Evaluated outside the renderer guard: a misspelled switch must fail loudly,
    # not silently disable every head.
    owed_reads = (owed_read_rows(frame.get("read_facts"))
                  if headcarry_owed_reads_enabled() else [])
    try:
        from kernel_stateframe_headcarry import (
            record_headcarry_rendered,
            record_headcarry_shadow_mismatches,
            render_headcarry,
        )

        panel = _json_value(render_headcarry(
            root,
            open_sites,
            applied_edits,
            messages=messages,
            wire_sites=open_sites,
            owed_reads=owed_reads,
        ))
        if not isinstance(panel, dict):
            raise TypeError("head-carry renderer did not return a JSON object")
    except Exception:
        # Head carry is enrichment; a renderer fault must leave the legacy frame usable.
        frame.pop("headcarry", None)
        try:
            setattr(runtime, "_framefit5_headcarry_failed", True)
        except Exception:
            pass
        return

    try:
        setattr(runtime, "_framefit5_headcarry_failed", False)
    except Exception:
        pass
    if mode == "shadow" and not already_attached:
        record_headcarry_shadow_mismatches(getattr(runtime, "journal", None), panel)
        return
    frame["headcarry"] = panel
    if initial and mode == "on":
        record_headcarry_rendered(getattr(runtime, "journal", None), panel)


def _pathless_source_path(value: object) -> bool:
    """Missing/blank handles confer no source identity; other values need validation."""
    return value is None or (isinstance(value, str) and not value.strip())


def _safe_relative_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if (not raw or not path.parts or any(char in raw for char in ("\x00", "\n", "\r"))
            or path.is_absolute() or any(
            part in {"", ".", ".."} or part.casefold() == "backups"
            for part in path.parts)):
        raise FrameFit5Error("frame-only source path is unsafe")
    return path.as_posix()


def _first_content_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("text"), str):
        return str(value["text"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for part in value:
            text = _first_content_text(part)
            if text is not None:
                return text
    return None


def _is_volatile_tail_message(message: Mapping[str, object]) -> bool:
    return (
        message.get("role") == "user"
        and (_first_content_text(message.get("content")) or "").startswith(
            VOLATILE_TAIL_MARKER
        )
    )


def _is_regenerated_wire_fact(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith("[WIRE_TRUTH_FACT ")
    )


def _strip_regenerated_tail_fact(
        message: Mapping[str, object],
) -> dict[str, object] | None:
    if _is_volatile_tail_message(message) or _is_regenerated_wire_fact(message):
        return None
    copied = copy.deepcopy(dict(message))
    content = copied.get("content")
    if (copied.get("role") == "user" and isinstance(content, str)
            and content.endswith(GREENFACT_TEXT)):
        copied["content"] = content[:-len(GREENFACT_TEXT)]
        if not copied["content"]:
            return None
    return copied


def _task_and_latest_turn(
        messages: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], tuple[dict[str, object], ...], int, int]:
    user_rows = [
        (index, message) for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    task_row = next(
        ((index, message) for index, message in user_rows
         if str(message.get("content") or "").startswith("Task:")),
        user_rows[0] if user_rows else None,
    )
    if task_row is None:
        raise FrameFit5Error("frame-only request has no task message")
    task_index, task_message = task_row
    last_action_index = max(
        (index for index, message in enumerate(messages)
         if message.get("role") in {"assistant", "tool"}),
        default=-1,
    )
    latest_start = max(task_index + 1, last_action_index + 1)
    latest_rows: list[dict[str, object]] = []
    for message in messages[latest_start:]:
        if message.get("role") != "user" or not index_safe_message(message):
            continue
        content = message.get("content")
        if isinstance(content, str) and (
                content.startswith(STATE_HEADER + "\n")
                or content.startswith(HANDOFF_HEADER)):
            continue
        regenerated = _strip_regenerated_tail_fact(message)
        if regenerated is not None:
            latest_rows.append(regenerated)
    latest = tuple(latest_rows)
    return copy.deepcopy(dict(task_message)), latest, task_index, last_action_index


def index_safe_message(message: Mapping[str, object]) -> bool:
    return isinstance(message.get("content"), (str, list, type(None)))


def _is_legacy_generated_stateframe_unavailable(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    if not isinstance(content, str):
        return False
    lines = content.splitlines()
    return (
        message.get("role") == "user"
        and len(lines) == 3
        and lines[0] == "[Host fact: StateFrame unavailable]"
        and lines[1].startswith("Reason: ")
        and lines[2] == (
            "Automatic StateFrame and carried source sections are omitted; a current target "
            "section is included only after its bytes are verified. Continue from recorded "
            "conversation history."
        )
    )


def _is_legacy_generated_vacuity_certificate(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    suffix = " Observation only; it never blocks done or certifies repair."
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith("[VERIFICATION CERTIFICATE] ")
        and content.endswith(suffix)
        and all(token in content for token in (
            "; selectors=", "; execution_ids=", "; receipt_ids=", "; counts=",
        ))
    )


def _is_framefit5_source_notice(message: Mapping[str, object]) -> bool:
    if message.get("role") != "user":
        return False
    if _is_legacy_generated_stateframe_unavailable(message):
        return True
    if message.get("_request_budget_section") != "stage_evidence":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return content.startswith((
        "[Deferred required source paths]",
        "[Request diet: dropped source ranges]",
        "[Host fact]",
    ))


_FRAMEFIT5_HOST_CONTEXT_PREFIXES = (
    "[Unified stop authority]",
    "[MODULE HEAD |",
    "[INITIALIZATION / CONSTRUCTION FACT]",
    "[LITERAL SITE INVENTORY |",
    "[OPEN UNKNOWNS]",
    "[CONTINUATION LIMITS |",
    "[COMPLETION OWNER]",
    "[WIRE_READ_CONTRACT]",
    "[NATIVE BATCH RECEIPTS]",
)


def _is_framefit5_host_context_panel(message: Mapping[str, object]) -> bool:
    """Recognize current host panels that must accompany the selected exchange."""
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(_FRAMEFIT5_HOST_CONTEXT_PREFIXES)
    )


def _wire_truth_facts(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if not content.startswith("[WIRE_TRUTH_FACT"):
            continue
        body = content.splitlines()[0].split("]", 1)[-1].strip()
        name, separator, value = body.partition(":")
        if not separator or not name.strip() or not value.strip():
            continue
        facts.append({
            "name": name.strip(),
            "value": value.strip(),
            "source": "recorded_wire_truth",
            "message_index": index,
        })
    return facts


def _prior_user_context(
        messages: Sequence[Mapping[str, object]], task_index: int,
        last_action_index: int,
) -> list[dict[str, object]]:
    start = task_index + 1
    stop = last_action_index + 1
    rows: list[dict[str, object]] = []
    total = 0
    for index in range(start, stop):
        message = messages[index]
        if (message.get("role") != "user"
                or _is_framefit5_source_notice(message)
                or _is_legacy_generated_vacuity_certificate(message)):
            continue
        # A replay may contain the previous generated envelope. Its typed
        # state has its own authority; nesting it as user prose duplicates
        # owner pins and makes historical host bytes part of the minimum.
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n"):
            try:
                _state_payload(message)
            except FrameFit5Error:
                pass  # Marker-like original prose remains opaque user input.
            else:
                continue
        content_value = message.get("content")
        if not isinstance(content_value, str):
            raise FrameFit5Error("prior user context is not plain text")
        raw = content_value.encode("utf-8")
        total += len(raw)
        if total > MAX_CONTEXT_BYTES:
            raise FrameFit5Error("typed prior user context exceeds its byte limit")
        header = content_value.splitlines()[0] if content_value else ""
        rows.append({
            "kind": "prior_user_context",
            "header": header[:240],
            "content": content_value,
            "content_sha256": _sha256(raw),
            "message_index": index,
        })
    return rows


def _row_call_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        call_id
        for row in rows
        for call_id in row.get("call_ids", ())
        if isinstance(call_id, str) and call_id
    }


def _opaque_actions(
        turns: object,
        journal: object = None,
) -> tuple[list[dict[str, object]], set[str]]:
    from kernel_stateframe_action_ledger import (
        MAX_ACTION_ARGUMENT_BYTES,
        _bounded_action_receipt,
        _call_argument_bytes,
        _receipt_line,
        _tool_name,
    )

    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes, bytearray)):
        raise FrameFit5Error("unclassified action turns are malformed")
    rows: list[dict[str, object]] = []
    ids: set[str] = set()
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes, bytearray)):
            raise FrameFit5Error("unclassified action turn is malformed")
        results = {
            str(item.get("tool_call_id")): item
            for item in turn
            if isinstance(item, Mapping) and item.get("role") == "tool"
        }
        assistant_rows = [
            item for item in turn
            if isinstance(item, Mapping) and item.get("role") == "assistant"
        ]
        if not assistant_rows:
            raise FrameFit5Error("unclassified action turn has no assistant action")
        for assistant in assistant_rows:
            text = str(assistant.get("content") or "")
            if text:
                encoded = text.encode("utf-8")
                if len(encoded) > MAX_OPAQUE_RECEIPT_BYTES:
                    rows.append({
                        "kind": "assistant_note_receipt",
                        "turn_index": turn_index,
                        "byte_size": len(encoded),
                        "sha256": _sha256(encoded),
                        "first_line": _receipt_line(text),
                    })
                else:
                    rows.append({
                        "kind": "assistant_note",
                        "turn_index": turn_index,
                        "text": text,
                        "text_sha256": _sha256(encoded),
                    })
            calls = assistant.get("tool_calls") or ()
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
                raise FrameFit5Error("unclassified assistant tool calls are malformed")
            for call in calls:
                if not isinstance(call, Mapping):
                    raise FrameFit5Error("unclassified tool call is malformed")
                call_id = str(call.get("id") or "")
                if not call_id or call_id in ids:
                    raise FrameFit5Error("unclassified tool call id is invalid")
                ids.add(call_id)
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise FrameFit5Error("unclassified tool call has no function")
                raw_arguments = function.get("arguments", {})
                if isinstance(raw_arguments, str):
                    try:
                        arguments: object = json.loads(raw_arguments)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        arguments = raw_arguments
                else:
                    arguments = copy.deepcopy(raw_arguments)
                argument_bytes = _call_argument_bytes(call)
                result = results.get(call_id)
                if not isinstance(result, Mapping):
                    raise FrameFit5Error("unclassified tool result is missing")
                from kernel_wire_identity import public_wire_text
                result_text = public_wire_text(str(result.get("content") or ""), journal)
                result_bytes = result_text.encode("utf-8")
                if len(argument_bytes) > MAX_ACTION_ARGUMENT_BYTES:
                    receipt = _bounded_action_receipt(
                        call_id,
                        _tool_name(call) or "unknown",
                        arguments,
                        argument_bytes,
                        result_text,
                        turn_index=turn_index,
                        journal=journal,
                    )
                    rows.append(_validated_bounded_action_receipt(receipt))
                    continue
                first_line = result_text.splitlines()[0] if result_text else ""
                rows.append({
                    "kind": "unclassified_tool_action",
                    "turn_index": turn_index,
                    "call_id": call_id,
                    "tool": str(function.get("name") or "unknown"),
                    "arguments": arguments,
                    "result_receipt": first_line[:240],
                    "result_chars": len(result_text),
                    "result_sha256": _sha256(result_bytes),
                })
    return rows, ids


def _valid_bounded_receipt_path(path: object) -> bool:
    if path is None:
        return True
    if not isinstance(path, str):
        return False
    try:
        if len(path.encode("utf-8")) > 512:
            return False
        return _safe_relative_path(path) == path
    except (FrameFit5Error, UnicodeError):
        return False


def _validated_bounded_action_receipt(value: object) -> dict[str, object]:
    required = {
        "schema", "kind", "call_id", "tool", "path", "outcome",
        "argument_size_bytes", "argument_sha256", "result_first_line",
        "result_size_bytes", "result_sha256",
    }
    optional = {"call_id_sha256", "tool_sha256", "path_sha256", "turn_index"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise FrameFit5Error("typed oversized action receipt is malformed")
    if set(value) - required - optional:
        raise FrameFit5Error("typed oversized action receipt has unknown fields")

    def valid_digest(candidate: object) -> bool:
        return (
            isinstance(candidate, str)
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        )

    from kernel_stateframe_action_ledger import MAX_ACTION_ARGUMENT_BYTES

    call_id = value.get("call_id")
    tool = value.get("tool")
    path = value.get("path")
    argument_size = value.get("argument_size_bytes")
    result_size = value.get("result_size_bytes")
    first_line = value.get("result_first_line")
    path_valid = _valid_bounded_receipt_path(path)
    if (
        value.get("schema") != "stateframe-bounded-action-receipt/1"
        or value.get("kind") != "bounded_action_receipt"
        or not isinstance(call_id, str) or not call_id or len(call_id) > 160
        or not isinstance(tool, str) or not tool or len(tool) > 120
        or not path_valid
        or value.get("outcome") not in {"APPLIED", "REJECTED", "CLOSED"}
        or type(argument_size) is not int or argument_size <= MAX_ACTION_ARGUMENT_BYTES
        or not valid_digest(value.get("argument_sha256"))
        or type(result_size) is not int or result_size < 0
        or not valid_digest(value.get("result_sha256"))
        or not isinstance(first_line, str) or len(first_line) > 240
    ):
        raise FrameFit5Error("typed oversized action receipt fields are invalid")
    for field in ("call_id_sha256", "tool_sha256", "path_sha256"):
        if field in value and not valid_digest(value[field]):
            raise FrameFit5Error("typed oversized action receipt digest is invalid")
    turn_index = value.get("turn_index")
    if "turn_index" in value and (type(turn_index) is not int or turn_index < 0):
        raise FrameFit5Error("typed oversized action receipt index is invalid")
    return copy.deepcopy(dict(value))


def _ledger_projection(action_context: Mapping[str, object]) -> dict[str, object]:
    raw_ledger = action_context.get("ledger")
    if not isinstance(raw_ledger, Mapping):
        ledger_rows: list[dict[str, object]] = []
    else:
        source_rows = raw_ledger.get("rows")
        if not isinstance(source_rows, Sequence) or isinstance(
                source_rows, (str, bytes, bytearray)):
            raise FrameFit5Error("typed action ledger rows are malformed")
        ledger_rows = [copy.deepcopy(dict(row)) for row in source_rows
                       if isinstance(row, Mapping)]
        if len(ledger_rows) != len(source_rows):
            raise FrameFit5Error("typed action ledger contains an invalid row")
    reads = []
    non_file_reads = []
    for row in ledger_rows:
        if row.get("kind") != "read":
            continue
        target = row.get("path")
        if _pathless_source_path(target):
            # Retain the action and its call coverage, without source-file credit.
            non_file_reads.append({"kind": "non_file_read_receipt",
                                   "call_ids": list(row.get("call_ids", ())), "receipt": row})
        else:
            reads.append(row)
    edits = [row for row in ledger_rows if row.get("kind") == "edit"]
    tests = [row for row in ledger_rows if row.get("kind") == "test"]
    for row in reads:
        if (not isinstance(row.get("path"), str)
                or not isinstance(row.get("lines"), Mapping)
                or not isinstance(row.get("revision"), str)):
            raise FrameFit5Error("typed read receipt lacks path, range, or revision")
    for row in edits:
        if (not isinstance(row.get("path"), str)
                or not isinstance(row.get("lines"), Mapping)
                or not isinstance(row.get("post_edit_revision"), str)):
            raise FrameFit5Error("typed edit receipt lacks path, span, or post-edit revision")
        receipt = row.get("bounded_action_receipt")
        if receipt is not None:
            row["bounded_action_receipt"] = _validated_bounded_action_receipt(receipt)
    latest_test = copy.deepcopy(tests[-1]) if tests else None
    return {
        "schema": "stateframe-framefit3-action-ledger/1",
        "reads": reads,
        "edits": edits,
        "last_test": latest_test,
        "call_ids": sorted(_row_call_ids(ledger_rows)),
        **({"non_file_reads": non_file_reads} if non_file_reads else {}),
    }


def _journal_truth_projection(
        action_context: Mapping[str, object], runtime: object,
        messages: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = action_context.get("journal_receipt_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise FrameFit5Error("FRAMEFIT5 requires a journal receipt projection")
    if any(not isinstance(record, Mapping) for record in records):
        raise FrameFit5Error("FRAMEFIT5 journal receipt projection is malformed")
    from kernel_stateframe_action_ledger import _candidate_revision
    from kernel_vacuity_truth import journal_applied_edit_rows, project_journal_truth

    binding_method = getattr(runtime, "binding", None)
    binding = binding_method() if callable(binding_method) else {}
    candidate_revision = _candidate_revision(messages)
    truth = project_journal_truth(
        records, binding, current_revision=candidate_revision or None,
    )
    status = truth.last_status
    if status is None:
        verdict_line = "GATE_NEVER_RAN"
    else:
        verdict_line = f"LAST {status} at rev {truth.tested_rev or 'UNKNOWN'}"
        if truth.changed_since_check and truth.changed_by:
            verdict_line += f"; changed by edit {truth.changed_by}"
    verdict: dict[str, object] = {
        "schema": "stateframe-vacuity-truth/1",
        "status": status,
        "tested_revision": truth.tested_rev,
        "current_revision": truth.current_rev,
        "changed_since_check": truth.changed_since_check,
        "changed_by": truth.changed_by,
        "check_receipt_id": truth.check_receipt_id,
        "line": verdict_line,
    }
    # Journal positions remain available to audit/replay consumers. The frame
    # references the canonical edit receipt alone, independent of other events.
    applied_edits = [
        {key: value for key, value in row.items() if key != "journal_event_id"}
        for row in journal_applied_edit_rows(records)
    ]
    return applied_edits, verdict


def _mark_stale_check_after_edit(
        framefit5: dict[str, object], rows: Sequence[Mapping[str, object]],
) -> None:
    """Do not present a pre-edit check as a verdict on the current revision."""
    ledger = framefit5.get("action_ledger")
    verdict = framefit5.get("last_verdict")
    if not isinstance(ledger, dict) or not isinstance(verdict, dict):
        return
    last_test = ledger.get("last_test")
    if not isinstance(last_test, dict):
        return
    latest_test_at = max((i for i, row in enumerate(rows)
                          if row.get("kind") == "test"), default=-1)
    later_edit = any(row.get("kind") == "edit" and i > latest_test_at
                     for i, row in enumerate(rows)) if latest_test_at >= 0 else False
    tested = verdict.get("tested_revision") or last_test.get("revision")
    current = verdict.get("current_revision")
    drift = (isinstance(tested, str) and isinstance(current, str)
             and current not in {"", "UNKNOWN"} and tested != current)
    # Journal check receipts are the ordering authority when present. A later
    # typed edit matters only if no journal check can supersede it.
    if not (verdict.get("changed_since_check") or drift
            or (later_edit and not verdict.get("check_receipt_id"))):
        return
    last_test["stale"] = True
    last_test["current_revision"] = current
    verdict["changed_since_check"] = True
    line = str(verdict.get("line") or "")
    if "STALE" not in line:
        verdict["line"] = (f"{line}; STALE check at rev {tested or 'UNKNOWN'}"
                           f"; current rev {current or 'UNKNOWN'}")


def _current_file_revision(root: Path, path: str) -> str | None:
    from kernel_stateframe_action_ledger import _read_open_site_file_bytes

    try:
        raw = _read_open_site_file_bytes(root, path)
    except (OSError, RuntimeError, ValueError):
        return None
    return "sha256:" + _sha256(raw)


def _journal_read_facts(
        action_context: Mapping[str, object], root: Path,
        action_ledger: Mapping[str, object],
) -> list[dict[str, object]]:
    records = action_context.get("journal_receipt_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise FrameFit5Error("FRAMEFIT5 requires journal read-receipt records")
    if any(not isinstance(record, Mapping) for record in records):
        raise FrameFit5Error("FRAMEFIT5 journal read-receipt records are malformed")

    from kernel_framefit_receipt_history import read_receipt, validate_context

    history = validate_context(action_context)
    receipts_by_path: dict[str, list[dict[str, object]]] = {}
    for event in records:
        read = read_receipt(event)
        if read is not None:
            path, receipt = read
            receipts_by_path.setdefault(path, []).append(receipt)

    raw_reads = action_ledger.get("reads")
    if not isinstance(raw_reads, Sequence) or isinstance(raw_reads, (str, bytes, bytearray)):
        raise FrameFit5Error("FRAMEFIT5 typed read ledger is malformed")
    read_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in raw_reads:
        if not isinstance(row, Mapping):
            raise FrameFit5Error("FRAMEFIT5 typed read ledger contains an invalid row")
        target = row.get("path")
        # Handle/receipt lookups have no source path and confer no file credit.
        # Keep validation strict for every present, nonempty file path.
        if _pathless_source_path(target):
            continue
        path = _safe_relative_path(target)
        read_rows.setdefault(path, []).append(row)

    facts: list[dict[str, object]] = []
    for path, receipts in receipts_by_path.items():
        matched_rows = read_rows.get(path, [])
        source_read = matched_rows[-1] if matched_rows else None
        read_revision = (
            source_read.get("revision")
            if isinstance(source_read, Mapping)
            and isinstance(source_read.get("revision"), str)
            else None
        )
        current_revision = _current_file_revision(root, path)
        unchanged = bool(
            isinstance(read_revision, str)
            and current_revision is not None
            and current_revision == read_revision
        )
        fact: dict[str, object] = {
            "path": path,
            "read_revision": read_revision,
            "current_revision": current_revision,
            "revision_scope": "source_file" if read_revision is not None else "unavailable",
            "unchanged_since_read": unchanged,
            "status": "UNCHANGED" if unchanged else (
                "CHANGED" if read_revision is not None and current_revision is not None
                else "UNVERIFIABLE"
            ),
            "receipt_count": len(receipts),
            "receipts": receipts,
        }
        if history is not None:
            summary = history["reads"][path]
            historical = summary["historical_receipts"]
            if summary["receipt_count"] != len(receipts) + historical["count"]:
                raise FrameFit5Error("journal read receipt accounting is incomplete")
            fact["receipt_count"] = summary["receipt_count"]
            if historical["count"]:
                fact["historical_receipts"] = copy.deepcopy(historical)
        if isinstance(source_read, Mapping):
            lines = source_read.get("lines")
            if isinstance(lines, Mapping):
                fact["lines"] = copy.deepcopy(dict(lines))
            content_sha256 = source_read.get("content_sha256")
            if isinstance(content_sha256, str):
                fact["content_sha256"] = content_sha256
        if not matched_rows:
            fact["journal_workspace_revisions"] = sorted({
                item["workspace_revision"] for item in receipts
                if item["workspace_revision"]
            })
        facts.append(fact)
    return facts


def _site_context_is_unverifiable(action_context: Mapping[str, object]) -> bool:
    site_context = action_context.get("site_context")
    return (
        isinstance(site_context, Mapping)
        and site_context.get("status") == "UNVERIFIABLE"
    )


def _fallback_site_anchor(action_context: Mapping[str, object]) -> tuple[str, int] | None:
    if _site_context_is_unverifiable(action_context):
        return None

    open_site = action_context.get("open_edit_site")
    if isinstance(open_site, Mapping):
        path = _safe_relative_path(open_site.get("path"))
        lines = open_site.get("lines")
        # A rendered site has an exact anchor plus a surrounding excerpt range.
        # Reusing the excerpt's start walks the anchor backwards on each refresh.
        line = open_site.get("line")
        if type(line) is not int or line < 1:
            line = lines.get("start") if isinstance(lines, Mapping) else None
        if type(line) is int and line > 0:
            return path, line

    ledger = action_context.get("ledger")
    rows = ledger.get("rows", ()) if isinstance(ledger, Mapping) else ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    row = next(
        (item for item in reversed(rows)
         if isinstance(item, Mapping) and item.get("kind") in {"read", "edit"}
         and not (item.get("kind") == "read" and _pathless_source_path(item.get("path")))),
        None,
    )
    if not isinstance(row, Mapping):
        return None
    path = _safe_relative_path(row.get("path"))
    lines = row.get("lines")
    line = lines.get("start") if isinstance(lines, Mapping) else None
    if type(line) is not int or line < 1:
        raise FrameFit5Error("last read or edit site has no valid line anchor")
    return path, line


def _workspace_root(runtime: object) -> Path:
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        raise FrameFit5Error("frame-only workspace root is unavailable")
    return Path(root_value).resolve(strict=True)


def _render_next_site(
    root: Path,
    path: str,
    line: int,
    *,
    source: str,
    literal: str = "",
    expected_text: str | None = None,
) -> dict[str, object]:
    from kernel_stateframe_action_ledger import _read_open_site_file_bytes

    try:
        raw = _read_open_site_file_bytes(root, path)
    except (OSError, RuntimeError) as error:
        raise _CandidateSourceChanged(
            "candidate source became unavailable during frame rendering"
        ) from error
    file_lines = raw.splitlines(keepends=True)
    if not file_lines:
        raise FrameFit5Error("next site file is empty")
    if expected_text is None:
        line = min(line, len(file_lines))
    elif line > len(file_lines):
        raise _CandidateSourceChanged("next remaining site moved outside the current file")
    observed = file_lines[line - 1].decode("utf-8").rstrip("\r\n")
    if expected_text is not None and observed != expected_text:
        raise _CandidateSourceChanged("next remaining site changed during frame construction")
    start = max(1, min(line - 39, len(file_lines) - MAX_SITE_LINES + 1))
    end = min(len(file_lines), start + MAX_SITE_LINES - 1)
    excerpt = b"".join(file_lines[start - 1:end])
    if len(excerpt) > MAX_SITE_BYTES:
        raise FrameFit5Error("next site exceeds its byte limit")
    return {
        "schema": "stateframe-framefit4-open-edit-site/1",
        "source": source,
        "literal": literal,
        "path": path,
        "line": line,
        "lines": {"start": start, "end": end, "total": len(file_lines)},
        "revision": "sha256:" + _sha256(raw),
        "content_sha256": _sha256(excerpt),
        "current_bytes": excerpt.decode("utf-8"),
        "current_line": observed,
        "read_this_request": True,
        "action_guidance": {
            "preferred_tool": "str_replace",
            "first_matching_tool": "str_replace",
            "max_calls_to_edit": 2,
            "unchanged_rereads": 0,
        },
    }


def _next_sites(runtime: object, action_context: Mapping[str, object]) -> list[dict[str, object]]:
    from kernel_remaining_sites import remaining_site_inventory

    root = _workspace_root(runtime)
    for _attempt in range(MAX_SOURCE_REFRESH_ATTEMPTS):
        inventory = remaining_site_inventory(runtime)
        if not inventory:
            fallback = _fallback_site_anchor(action_context)
            if fallback is None:
                return []
            path, line = fallback
            try:
                site = _render_next_site(
                    root, path, line, source="last_read_or_edit_region")
                _verify_next_site_bytes(site, root)
                return [site]
            except _CandidateSourceChanged:
                continue

        rendered: list[dict[str, object]] = []
        retry = False
        for candidate in inventory:
            path = _safe_relative_path(candidate.get("path"))
            line = candidate.get("line")
            expected_text = candidate.get("text")
            expected_revision = candidate.get("revision")
            if (type(line) is not int or line < 1
                    or not isinstance(expected_text, str)
                    or not isinstance(expected_revision, str)):
                raise FrameFit5Error("remaining edit-site inventory is malformed")
            try:
                site = _render_next_site(
                    root, path, line, source="remaining_sites_inventory",
                    literal=str(candidate.get("literal") or ""),
                    expected_text=expected_text,
                )
                if site.get("revision") != expected_revision:
                    retry = True
                    break
                _verify_next_site_bytes(site, root)
            except _CandidateSourceChanged:
                retry = True
                break
            rendered.append(site)
        if not retry:
            return rendered
    raise FrameFit5Error("candidate source kept changing during final frame rendering")


def _next_site(runtime: object, action_context: Mapping[str, object]) -> dict[str, object] | None:
    sites = _next_sites(runtime, action_context)
    return sites[0] if sites else None


def build_typed_state(
        messages: Sequence[Mapping[str, object]], action_context: Mapping[str, object],
        runtime: object,
) -> dict[str, object]:
    """Build complete typed state; unsupported turns become typed receipts, never messages."""
    task, _latest, task_index, last_action_index = _task_and_latest_turn(messages)
    ledger = _ledger_projection(action_context)
    applied_edits, last_verdict = _journal_truth_projection(
        action_context, runtime, messages,
    )
    ledger["applied_edits"] = applied_edits
    ledger["journal_verdict"] = copy.deepcopy(last_verdict)
    latest_check = next(
        (record for record in reversed(action_context["journal_receipt_records"])
         if isinstance(record, Mapping) and record.get("kind") == "check_receipt"),
        None,
    )
    if isinstance(latest_check, Mapping):
        latest_test = ledger.get("last_test")
        latest_test = dict(latest_test) if isinstance(latest_test, Mapping) else {}
        receipt_status = str(latest_check.get("status") or "UNKNOWN").strip().upper()
        if receipt_status in {"PASS", "PASSED", "GREEN"}:
            latest_test["verdict"] = "GREEN"
        elif receipt_status in {"FAIL", "FAILED", "RED", "BEHAVIOR_FAIL"}:
            latest_test["verdict"] = "RED"
        latest_test["revision"] = last_verdict["tested_revision"] or "UNKNOWN"
        selector = latest_check.get("selector")
        if isinstance(selector, str) and selector:
            latest_test["target"] = selector
        ledger["last_test"] = latest_test
    source_ledger = action_context.get("ledger")
    source_rows = source_ledger.get("rows") if isinstance(source_ledger, Mapping) else None
    if isinstance(source_rows, list):
        _mark_stale_check_after_edit(
            {"action_ledger": ledger, "last_verdict": last_verdict}, source_rows,
        )
    opaque, opaque_ids = _opaque_actions(
        action_context.get("unclassified_turns", ()), getattr(runtime, "journal", None))
    opaque.extend(ledger.pop("non_file_reads", ()))
    ledger_ids = set(ledger["call_ids"])
    source_call_ids = {
        str(call.get("id"))
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or ())
        if isinstance(call, Mapping) and call.get("id")
    }
    superseded_test_ids = {
        value for value in action_context.get("superseded_test_call_ids", ())
        if isinstance(value, str)
    }
    missing = source_call_ids - ledger_ids - opaque_ids - superseded_test_ids
    if missing:
        raise FrameFit5Error("one or more prior tool actions lack typed state")
    ledger.pop("call_ids", None)
    ledger["unclassified_actions"] = opaque
    ledger["superseded_test_count"] = len(superseded_test_ids)
    open_sites = _next_sites(runtime, action_context)
    next_site = open_sites[0] if open_sites else None
    owner_root = _workspace_root(runtime)
    read_facts = _journal_read_facts(action_context, owner_root, ledger)
    wire_facts = _wire_truth_facts(messages)
    owner = getattr(runtime, "owner", None)
    task_row = getattr(owner, "task_row", {})
    task_row = task_row if isinstance(task_row, Mapping) else {}
    ticket_text = task_row.get("ticket")
    ticket_clauses: list[dict[str, object]] = []
    if isinstance(ticket_text, str):
        paragraphs = [part.strip() for part in ticket_text.split("\n\n") if part.strip()]
        if paragraphs and paragraphs[0].startswith("# "):
            paragraphs = paragraphs[1:]
        ticket_clauses = [
            {
                "clause_id": f"C{index}",
                "text": clause,
                "status": "OPEN",
                "source": "runtime_ticket",
                "sha256": _sha256(clause.encode("utf-8")),
            }
            for index, clause in enumerate(paragraphs, 1)
        ]
    open_requirements = next(
        (fact["value"] for fact in wire_facts
         if fact.get("name") == "open_requirement_ids"),
        "UNKNOWN",
    )
    open_work: dict[str, object] = {
        "next_site_available": next_site is not None,
        "open_edit_site_count": len(open_sites),
        "next_site_source": next_site.get("source") if next_site else "unavailable",
        "open_requirement_ids": open_requirements,
        "ticket_clauses": ticket_clauses,
        "ticket_clause_completion": "OPEN" if ticket_clauses else "not_inferred",
    }
    site_context = action_context.get("site_context")
    if isinstance(site_context, Mapping) and site_context.get("status") == "UNVERIFIABLE":
        open_work["site_context_status"] = "UNVERIFIABLE"
        open_work["site_context_reason"] = "latest_file_action_has_no_exact_current_site"
    state = {
        "schema": SCHEMA,
        "mode": PRODUCT_MODE,
        "task": {
            "source": "original_task_message",
            "sha256": _sha256(str(task.get("content") or "").encode("utf-8")),
            "ticket_sha256": (
                _sha256(ticket_text.encode("utf-8")) if isinstance(ticket_text, str) else None
            ),
        },
        "next_site": next_site,
        "open_sites": open_sites,
        "read_facts": read_facts,
        "applied_edits": applied_edits,
        "last_verdict": last_verdict,
        "action_ledger": ledger,
        "established_facts": wire_facts,
        "prior_user_context": _prior_user_context(
            messages, task_index, last_action_index),
        "open_work": open_work,
    }
    from kernel_agent_tools import service as agent_tools_service
    host_tools = agent_tools_service()
    if host_tools is not None and host_tools.jobs.jobs:
        state["jobs"] = host_tools.snapshot()
    from kernel_skills import service as skills_service
    skills = skills_service()
    if skills is not None:
        state["skills"] = skills.catalog()
    from kernel_framefit_receipt_history import attach_history

    attach_history(state, action_context)
    _update_headcarry(
        state,
        runtime=runtime,
        root=owner_root,
        open_sites=open_sites,
        applied_edits=applied_edits,
        messages=messages,
        initial=True,
    )
    from kernel_stateframe_capacity import bound_typed_state

    state = bound_typed_state(state, semantic_messages=messages, action_context=action_context)
    serialized = _json_text(state)
    if len(serialized.encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise FrameFit5Error("typed frame exceeds its product byte limit")
    return state


def attach_typed_state(
        action_context: Mapping[str, object], typed_state: Mapping[str, object],
) -> dict[str, object]:
    """Attach FrameFit5 data and select the inventory's next site as F4's open site."""
    result = copy.deepcopy(dict(action_context))
    result["framefit5"] = copy.deepcopy(dict(typed_state))
    result.pop("open_edit_site", None)
    site = typed_state.get("next_site")
    if isinstance(site, Mapping):
        result["open_edit_site"] = copy.deepcopy(dict(site))
    result["verbatim_turns"] = ()
    result["protected_call_ids"] = ()
    result["keep_k_call_ids"] = ()
    return result


def _state_payload(message: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    text = message.get("content")
    if not isinstance(text, str) or not text.startswith(STATE_HEADER + "\n"):
        raise FrameFit5Error("frame-only request lacks a StateFrame block")
    try:
        payload = json.loads(text.split("\n", 1)[1])
    except (TypeError, ValueError, RecursionError) as error:
        producer = (
            "kernel_request_budget._bound_oversized_panels"
            if "[panel excerpt]" in text
            else "kernel_stateframe_framefit5._serialize_state_message"
        )
        raise FrameFit5ProducerError(producer) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        # Name the envelope shape (schema field names only, never values) so a fail-closed projection is diagnosable.
        keys = sorted(str(key) for key in payload)[:8] if isinstance(payload, dict) else [type(payload).__name__]
        schema = payload.get("schema") if isinstance(payload, dict) and isinstance(payload.get("schema"), str) else None
        raise FrameFit5Error(
            "frame-only StateFrame state is invalid"
            f" (schema={str(schema)[:40] if schema else 'none'}; keys={','.join(keys)[:120]})"
        )
    state = payload["state"]
    framefit5 = state.get("framefit5")
    if not isinstance(framefit5, Mapping) or framefit5.get("mode") != PRODUCT_MODE:
        raise FrameFit5Error("StateFrame does not contain the frame-only product state")
    return payload, state


def fit_headcarry_to_request_budget(
        messages: list[object],
        *,
        max_request_bytes: int | None = None,
        measure_request_bytes: Callable[[], int] | None = None,
        max_request_tokens: int | None = None,
        measure_request_tokens: Callable[[], int] | None = None,
) -> int:
    """Trim typed headcarry to active request ceilings while preserving valid JSON."""
    if (max_request_bytes is None) != (measure_request_bytes is None):
        raise ValueError("head-carry byte ceiling and measurement must be provided together")
    if (max_request_tokens is None) != (measure_request_tokens is None):
        raise ValueError("head-carry token ceiling and measurement must be provided together")
    if _headcarry_mode() != "on":
        return 0
    state_messages = [
        message for message in messages
        if isinstance(message, dict)
        and str(message.get("content") or "").startswith(STATE_HEADER + "\n")
        and ('"framefit5"' in str(message.get("content") or "")
             or "stateframe-framefit5/1" in str(message.get("content") or ""))
    ]
    if len(state_messages) != 1:
        return 0
    state_message = state_messages[0]
    payload, state = _state_payload(state_message)
    frame = state.get("framefit5")
    panel = frame.get("headcarry") if isinstance(frame, dict) else None
    if not isinstance(frame, dict) or not isinstance(panel, dict):
        return 0
    original_omitted = panel.get("omitted_count")
    if type(original_omitted) is not int or original_omitted < 0:
        raise FrameFit5Error("typed head-carry omitted count is invalid")

    from kernel_stateframe_headcarry import fit_headcarry_panel_for_request

    original_content = str(state_message["content"])

    def with_candidate(
            candidate: Mapping[str, object],
            measure_request: Callable[[], int],
    ) -> int:
        frame["headcarry"] = dict(candidate)
        state_message["content"] = _serialize_state_message(payload)
        try:
            return measure_request()
        finally:
            frame["headcarry"] = panel
            state_message["content"] = original_content

    fitted = fit_headcarry_panel_for_request(
        panel,
        max_request_bytes=max_request_bytes,
        measure_request_bytes=(
            lambda candidate: with_candidate(candidate, measure_request_bytes)
            if measure_request_bytes is not None else None
        ) if max_request_bytes is not None else None,
        max_request_tokens=max_request_tokens,
        measure_request_tokens=(
            lambda candidate: with_candidate(candidate, measure_request_tokens)
            if measure_request_tokens is not None else None
        ) if max_request_tokens is not None else None,
    )
    removed = fitted["omitted_count"] - original_omitted
    if removed <= 0:
        return 0
    frame["headcarry"] = fitted
    payload["state"] = state
    state_message["content"] = _serialize_state_message(payload)
    _state_payload(state_message)
    return removed


def validate_no_verbatim_turns(turns: object) -> None:
    """Reject every transcript-preservation path in a FRAMEFIT5 product request."""
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes, bytearray)):
        raise FrameFit5Error("frame-only prior turns are malformed")
    if turns:
        raise FrameFit5Error("frame-only product cannot retain prior turns")


_FrameSourceRange = tuple[int, int] | tuple[tuple[int, int], ...]


def _framefit_source_identity(
    message: Mapping[str, object],
) -> tuple[str, str, str, _FrameSourceRange | None] | None:
    """Read presentation identity using the source format's own metadata parser.

    This selects carriers, not delivery proof. The final source verifier still
    checks revisions and exact bytes before any allocation can be published.
    """
    from kernel_request_diet import _source_identity
    from kernel_request_source_window import parse_model_source_window, parse_window

    content = message.get("content")
    metadata = parse_model_source_window(content) or parse_window(content)
    if isinstance(metadata, Mapping):
        path, revision, ranges = (
            metadata.get("path"), metadata.get("revision"), metadata.get("ranges"),
        )
        if isinstance(path, str) and isinstance(revision, str) and isinstance(ranges, list):
            spans = tuple(
                (row.get("start_line"), row.get("end_line"))
                for row in ranges if isinstance(row, Mapping)
            )
            if spans and len(spans) == len(ranges) and all(
                type(start) is int and type(end) is int and 1 <= start <= end
                for start, end in spans
            ):
                return "source", path.replace("\\", "/"), revision, spans
    return _source_identity(message)


def _prepared_current_source_messages(
        messages: Sequence[Mapping[str, object]],
        *,
        task: Mapping[str, object],
        newest: Sequence[Mapping[str, object]],
        structured_sources: bool = False,
) -> list[dict[str, object]]:
    """Keep only F4-selected, revision-bound source excerpts in the product request."""
    from kernel_request_diet import section_name, _source_identity

    # Closure-backed candidates need structured identities. Legacy rendering
    # without source authority retains its original source-selection contract.
    source_identity = _framefit_source_identity if structured_sources else _source_identity

    excluded = [task, *newest]
    newest_source_identities = {
        (path, revision, source_range)
        for message in newest
        if message.get("role") == "user"
        for identity in (source_identity(message),)
        if identity is not None
        for kind, path, revision, source_range in (identity,)
        if kind == "source" and path and revision and source_range is not None
    }
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, str, _FrameSourceRange, str]] = set()
    for index, message in enumerate(messages):
        if message.get("role") != "user" or message in excluded:
            continue
        content = message.get("content")
        if (not isinstance(content, str)
                or content.startswith(STATE_HEADER + "\n")
                or section_name(message, index) != "file_excerpts"):
            continue
        identity = source_identity(message)
        if identity is None:
            continue
        kind, path, revision, source_range = identity
        if kind != "source" or not path or not revision or source_range is None:
            continue
        # A selected excerpt already present in the newest user suffix stays in
        # that original position; adding its prepared copy ahead of the suffix
        # duplicates source panels and changes the pinned frame projection.
        if (path, revision, source_range) in newest_source_identities:
            continue
        key = (path, revision, source_range, content)
        if key in seen:
            continue
        seen.add(key)
        selected.append(copy.deepcopy(dict(message)))
    return selected


def _restore_live_hypotheses(
        state_message: Mapping[str, object], live_frame: Mapping[str, object],
) -> dict[str, object]:
    """Restore the bounded hypotheses carrier from the live journal projection."""
    if "hypotheses" not in live_frame:
        return copy.deepcopy(dict(state_message))
    payload, state = _state_payload(state_message)
    from kernel_stateframe_frame_continuity import project_continuity

    candidate, _mismatches = project_continuity(live_frame, payload)
    projected_state = candidate.get("state")
    if not isinstance(projected_state, Mapping) or "hypotheses" not in projected_state:
        return copy.deepcopy(dict(state_message))
    restored_state = copy.deepcopy(state)
    for key in (
            "hypotheses", "hypotheses_omitted", "hypothesis_state",
            "hypothesis_state_omitted"):
        if key in projected_state:
            restored_state[key] = copy.deepcopy(projected_state[key])
        else:
            restored_state.pop(key, None)
    payload["state"] = restored_state
    return {
        **dict(state_message),
        "content": _serialize_state_message(payload),
    }


def _allocated_task_suffix_exchange(
        layout: Mapping[str, object],
        task_message: Mapping[str, object] | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], int]:
    captured_task = layout.get("task_message")
    raw_suffix = layout.get("newest_user_suffix")
    raw_exchange = layout.get("newest_tool_exchange")
    suffix_start = layout.get("newest_user_suffix_start")
    if (not isinstance(captured_task, Mapping) or not isinstance(raw_suffix, list)
            or not isinstance(raw_exchange, list) or type(suffix_start) is not int):
        raise FrameFit5Error("frame-only allocation has an invalid message layout")

    task = copy.deepcopy(dict(captured_task))
    if task_message is not None:
        if (not isinstance(task_message, Mapping) or task_message.get("role") != "user"
                or not index_safe_message(task_message)
                or (isinstance(task_message.get("content"), str)
                    and str(task_message.get("content")).startswith(STATE_HEADER + "\n"))
                or dict(task_message) != task):
            raise FrameFit5Error("captured frame-only task changed after allocation")

    suffix = [
        copy.deepcopy(dict(message))
        for message in raw_suffix if isinstance(message, Mapping)
    ]
    exchange = [
        copy.deepcopy(dict(message))
        for message in raw_exchange if isinstance(message, Mapping)
    ]
    if len(suffix) != len(raw_suffix) or len(exchange) != len(raw_exchange):
        raise FrameFit5Error("frame-only allocation contains malformed messages")
    return task, suffix, exchange, suffix_start


def _allocated_prefix_messages(
        layout: Mapping[str, object], prefix_start: int, suffix_start: int,
) -> list[dict[str, object]]:
    raw_facts = layout.get("current_user_facts")
    raw_carriers = layout.get("carriers")
    if not isinstance(raw_facts, list) or not isinstance(raw_carriers, list):
        raise FrameFit5Error("frame-only allocation has invalid current facts or carriers")

    rows: list[tuple[object, object]] = []
    rows.extend(
        (fact.get("location"), fact.get("message"))
        for fact in raw_facts if isinstance(fact, Mapping)
    )
    rows.extend(
        (carrier.get("location"), carrier.get("message"))
        for carrier in raw_carriers
        if isinstance(carrier, Mapping) and carrier.get("kind") == "prepared_source"
    )
    expected_count = len(raw_facts) + sum(
        isinstance(carrier, Mapping) and carrier.get("kind") == "prepared_source"
        for carrier in raw_carriers
    )
    if len(rows) != expected_count:
        raise FrameFit5Error("frame-only allocation contains malformed prefix rows")

    by_location: dict[int, dict[str, object]] = {}
    for location, raw_message in rows:
        if (type(location) is not int or not isinstance(raw_message, Mapping)
                or raw_message.get("role") != "user"
                or not index_safe_message(raw_message)):
            raise FrameFit5Error("frame-only allocation contains an invalid prefix message")
        content = raw_message.get("content")
        if (isinstance(content, str)
                and (content.startswith(STATE_HEADER + "\n")
                     or content.startswith(HANDOFF_HEADER)
                     or _is_volatile_tail_message(raw_message))):
            raise FrameFit5Error("frame-only allocation prefix contains a reserved message")
        if location in by_location:
            raise FrameFit5Error("frame-only allocation repeats a prefix location")
        by_location[location] = copy.deepcopy(dict(raw_message))

    if (suffix_start < prefix_start
            or sorted(by_location) != list(range(prefix_start, suffix_start))):
        raise FrameFit5Error("frame-only allocation prefix locations are not contiguous")
    return [by_location[index] for index in range(prefix_start, suffix_start)]


def _validate_allocated_composition(
        result: Sequence[Mapping[str, object]],
        exchange: Sequence[Mapping[str, object]],
) -> None:
    action_messages = [
        message for message in result if message.get("role") in {"assistant", "tool"}
    ]
    if action_messages != list(exchange):
        raise FrameFit5Error("frame-only allocation retained messages outside newest exchange")
    if (not result or not _is_volatile_tail_message(result[-1])
            or any(str(message.get("content") or "").startswith(HANDOFF_HEADER)
                   for message in result)):
        raise FrameFit5Error("frame-only allocation did not preserve final wire structure")
    if len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise FrameFit5Error("frame-only request exceeds its product byte limit")


def _compose_allocated_frame_only(
        systems: Sequence[Mapping[str, object]],
        task_message: Mapping[str, object] | None,
        state_block: Mapping[str, object],
        allocation: FrameRequestAllocation,
) -> list[dict[str, object]]:
    try:
        allocation.validate()
    except FrameRequestAllocationError as exc:
        raise FrameFit5Error("frame-only allocation failed integrity validation") from exc

    layout = allocation.source_layout
    task, suffix, exchange, suffix_start = _allocated_task_suffix_exchange(
        layout, task_message,
    )
    prefix_start = len(systems) + 2
    prefix = _allocated_prefix_messages(layout, prefix_start, suffix_start)
    result = [
        *[copy.deepcopy(dict(message)) for message in systems],
        task,
        copy.deepcopy(dict(state_block)),
        *prefix,
        *suffix,
        *exchange,
        {"role": "user", "content": VOLATILE_TAIL_MARKER},
    ]
    _validate_allocated_composition(result, exchange)
    return result


def compose_frame_only(
        prepared: Sequence[Mapping[str, object]],
        source_messages: Sequence[Mapping[str, object]],
        *, live_frame: Mapping[str, object] | None = None,
        task_message: Mapping[str, object] | None = None,
        host_text_messages: Sequence[Mapping[str, object]] = (),
        frame_request_allocation: FrameRequestAllocation | None = None,
        structured_sources: bool = False,
        volatile_micrograph: bool = False,
) -> list[dict[str, object]]:
    """Assemble the StateFrame, current user blocks, newest exchange, and volatile tail."""
    systems = [copy.deepcopy(dict(message)) for message in prepared
               if message.get("role") == "system"]
    if not systems:
        systems = [copy.deepcopy(dict(message)) for message in source_messages
                   if message.get("role") == "system"]
    if frame_request_allocation is None:
        task, newest, _task_index, _last_action_index = _task_and_latest_turn(source_messages)
        if isinstance(task_message, Mapping):
            task_content = task_message.get("content")
            if (task_message.get("role") != "user"
                    or not index_safe_message(task_message)
                    or (isinstance(task_content, str)
                        and task_content.startswith(STATE_HEADER + "\n"))):
                raise FrameFit5Error("captured FrameFit5 task message is invalid")
            task = copy.deepcopy(dict(task_message))
    state_blocks = [message for message in prepared
                    if isinstance(message, Mapping)
                    and str(message.get("content") or "").startswith(STATE_HEADER + "\n")]
    if len(state_blocks) != 1:
        header_mentions = sum(
            str(message.get("content") or "").count(STATE_HEADER)
            for message in prepared
        )
        roles = tuple(str(message.get("role") or "") for message in prepared)
        raise FrameFit5Error(
            f"frame-only projection requires one StateFrame, found {len(state_blocks)}; "
            f"header_mentions={header_mentions}; roles={roles}"
        )
    state_block = copy.deepcopy(dict(state_blocks[0]))
    _payload, _state = _state_payload(state_block)
    if frame_request_allocation is not None:
        from kernel_frame_optional_allocation import render_optional_selection
        # Legacy hand-built allocations need no optional projection; production
        # snapshots carry the exact semantic site records and original panel.
        if "sites" in frame_request_allocation.site_inventory:
            _state["framefit5"] = render_optional_selection(
                _state["framefit5"], frame_request_allocation.optional_heads,
                frame_request_allocation.site_inventory,
            )
            state_block["content"] = _serialize_state_message(_payload)
    if isinstance(live_frame, Mapping):
        state_block = _restore_live_hypotheses(state_block, live_frame)
    if volatile_micrograph:
        payload, state = _state_payload(state_block)
        if "micrograph" in state:
            state.pop("micrograph")
            state_block["content"] = _serialize_state_message(payload)
    if frame_request_allocation is not None:
        return _compose_allocated_frame_only(
            systems, task_message, state_block, frame_request_allocation,
        )
    current_sources = _prepared_current_source_messages(
        prepared, task=task, newest=newest, structured_sources=structured_sources,
    )
    source_notices = []
    seen_source_notices: set[str] = set()
    for message in prepared:
        content = message.get("content")
        if (message.get("role") != "user"
                or not isinstance(content, str)
                or message in newest
                or not _is_framefit5_source_notice(message)
                or content in seen_source_notices):
            continue
        seen_source_notices.add(content)
        source_notices.append(copy.deepcopy(dict(message)))
    host_context_panels = []
    seen_host_context_panels: set[str] = set()
    for message in host_text_messages:
        content = message.get("content")
        if (message.get("role") != "user"
                or not isinstance(content, str)
                or message in newest
                or not _is_framefit5_host_context_panel(message)
                or content in seen_host_context_panels):
            continue
        seen_host_context_panels.add(content)
        host_context_panels.append(copy.deepcopy(dict(message)))
    latest_exchange = latest_complete_tool_exchange(source_messages)
    if latest_exchange is None:
        raise FrameFit5Error("latest assistant tool-call exchange is incomplete or mismatched")
    task_is_state_frame = (
        task.get("role") == "user"
        and isinstance(task.get("content"), str)
        and str(task.get("content")).startswith(STATE_HEADER + "\n")
    )
    task_messages = [] if task_is_state_frame else [task]
    from kernel_skills import service as skills_service
    skills = skills_service()
    skill_messages = skills.messages() if skills is not None else []
    result = [
        *systems, *task_messages, state_block, *skill_messages, *current_sources, *source_notices,
        *host_context_panels, *newest,
        *(copy.deepcopy(dict(message)) for message in latest_exchange),
        {"role": "user", "content": VOLATILE_TAIL_MARKER},
    ]
    action_messages = [message for message in result
                       if message.get("role") in {"assistant", "tool"}]
    if action_messages != list(latest_exchange):
        raise FrameFit5Error("frame-only projection retained messages outside the newest tool exchange")
    if (len(result) < len(latest_exchange) + 1
            or result[-len(latest_exchange) - 1:-1] != list(latest_exchange)
            or not _is_volatile_tail_message(result[-1])):
        raise FrameFit5Error("frame-only projection must place one volatile tail after its newest exchange")
    if any(str(message.get("content") or "").startswith(HANDOFF_HEADER) for message in result):
        raise FrameFit5Error("frame-only projection retained an F4 handoff")
    if len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise FrameFit5Error("frame-only request exceeds its product byte limit")
    return result


def assemble_framefit5_request(
        prepared: Sequence[Mapping[str, object]],
        source_messages: Sequence[Mapping[str, object]],
        *, live_frame: Mapping[str, object] | None = None,
        task_message: Mapping[str, object] | None = None,
        source_plan: Mapping[str, object] | None = None,
        sources: Mapping[str, str] | None = None,
        allocation_plan: dict[str, object] | None = None,
        runtime: object | None = None,
        owner_contract: object = None,
        owner_source_messages: Sequence[Mapping[str, object]] | None = None,
        owner_presentation: object = None,
        projection_audit: dict[str, object] | None = None,
        host_text_messages: Sequence[Mapping[str, object]] = (),
        frame_request_allocation: FrameRequestAllocation | None = None,
) -> list[dict[str, object]]:
    """Build one complete FrameFit5 provider surface for prep and every recovery path."""
    prepared_for_composition: Sequence[Mapping[str, object]] = prepared
    if frame_request_allocation is None:
        _task, original_newest, _task_index, _action_index = _task_and_latest_turn(source_messages)
    else:
        # Allocated assembly must consume its captured ownership, including
        # callers that intentionally supply no mutable transcript at all.
        original_newest = frame_request_allocation.source_layout["newest_user_suffix"]
    if frame_request_allocation is None and isinstance(source_plan, Mapping):
        from kernel_frame_source_snapshot import retire_f4_deferred_carriers

        prepared_for_composition = retire_f4_deferred_carriers(
            prepared, source_messages, source_plan, sources,
        )
    required_deferred_paths: list[object] = []
    if isinstance(source_plan, Mapping):
        deferred_paths = source_plan.get("deferred_paths", ())
        if (isinstance(deferred_paths, Sequence)
                and not isinstance(deferred_paths, (str, bytes, bytearray))
                and deferred_paths):
            required_deferred_paths = list(deferred_paths)
            if frame_request_allocation is None:
                from kernel_request_source_window import upsert_deferred_source_notice

                prepared_for_composition = [
                    copy.deepcopy(dict(message)) for message in prepared_for_composition
                ]
                from kernel_workspace_truth import enabled as workspace_truth_enabled
                if workspace_truth_enabled() and runtime is not None:
                    from kernel_read_reuse_visibility import _runtime_workspace_context
                    workspace_root, revision, journal = _runtime_workspace_context(runtime)
                    upsert_deferred_source_notice(
                        prepared_for_composition, deferred_paths,
                        workspace_root=workspace_root, revision=revision, journal=journal,
                    )
                else:
                    upsert_deferred_source_notice(prepared_for_composition, deferred_paths)

    preserved_tail_content: object = None
    preserve_tail = False
    if (has_framefit5_state(prepared_for_composition)
            and latest_complete_tool_exchange(prepared_for_composition) is not None
            and any(_is_volatile_tail_message(message)
                    for message in prepared_for_composition)):
        normalized_input = finalize_frame_only_wire_order(prepared_for_composition)
        preserved_tail_content = copy.deepcopy(normalized_input[-1].get("content"))
        preserve_tail = True

    result = compose_frame_only(
        prepared_for_composition, source_messages, live_frame=live_frame,
        task_message=task_message, host_text_messages=host_text_messages,
        frame_request_allocation=frame_request_allocation,
        structured_sources=isinstance(source_plan, Mapping),
        volatile_micrograph=getattr(runtime, "micrograph_wire", None) is not None,
    )
    projected_count = 0
    if frame_request_allocation is None and isinstance(source_plan, Mapping):
        from kernel_request_source_window import project_model_source_messages

        projected_count = project_model_source_messages(
            result,
            source_plan,
            allocation_plan=allocation_plan,
            materialize_framefit5_closure_units=True,
            sources=sources,
            preserve_source_messages=original_newest,
        )
    if runtime is not None and isinstance(owner_contract, Mapping):
        result = attach_owner_protocol(
            result,
            owner_source_messages if owner_source_messages is not None else source_messages,
            runtime,
            owner_contract,
            owner_presentation=owner_presentation,
        )
    if preserve_tail:
        volatile_indexes = [
            index for index, message in enumerate(result)
            if _is_volatile_tail_message(message)
        ]
        if len(volatile_indexes) != 1:
            raise FrameFit5Error(
                "FrameFit5 assembly could not preserve the existing volatile tail"
            )
        result[volatile_indexes[0]]["content"] = preserved_tail_content
    if frame_request_allocation is None:
        from kernel_stateframe_capacity import refresh_read_ranges

        refresh_read_ranges(result, getattr(runtime, "_framefit_action_context", None))
    # Allocated layouts already pin source notices as ordered facts. Rebuilding
    # them from mutable runtime evidence here would move captured carriers.
    source_notice_requested = frame_request_allocation is None and (
        bool(required_deferred_paths)
        or any(
            _is_framefit5_source_notice(message)
            and message not in original_newest
            for message in prepared_for_composition
        )
    )
    if source_notice_requested:
        state_messages = [
            message for message in result
            if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
        ]
        if len(state_messages) != 1:
            raise FrameFit5Error(
                "source notice reconciliation requires one StateFrame"
            )
        _payload, frame_state = _state_payload(state_messages[0])
        framefit5_state = frame_state.get("framefit5")
        if not isinstance(framefit5_state, dict):
            raise FrameFit5Error(
                "source notice reconciliation requires mutable FRAMEFIT5 state"
            )
        from kernel_reread_result_truth import reconcile_source_truth

        reconcile_source_truth(
            result, framefit5_state,
            required_deferred_paths=required_deferred_paths,
            include_host_fact=True,
        )
    micrograph_fact = None
    wire = getattr(runtime, "micrograph_wire", None)
    if wire is not None:
        from kernel_measurement import canonical_bytes

        micrograph_fact = "micrograph: " + canonical_bytes(
            wire.projection()).decode("utf-8")
    result = finalize_frame_only_wire_order(result, micrograph_fact=micrograph_fact)
    if frame_request_allocation is not None:
        from kernel_request_source_window import project_model_source_messages

        projected_count = project_model_source_messages(
            result,
            source_plan,
            materialize_framefit5_closure_units=True,
            sources=sources,
            frame_request_allocation=frame_request_allocation,
        )
    if wire is not None:
        state_messages = [message for message in result
                          if str(message.get("content") or "").startswith(STATE_HEADER + "\n")]
        if len(state_messages) != 1 or not isinstance(state_messages[0], dict):
            raise FrameFit5Error("MicroGraph projection requires one mutable StateFrame")
        if len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")) > MAX_PRODUCT_BYTES:
            raise FrameFit5Error("frame-only request exceeds its product byte limit")
    if projection_audit is not None:
        projection_audit["projected_count"] = projected_count
    return result


def _split_marker_string(content: str) -> object | None:
    if not content.startswith(VOLATILE_TAIL_MARKER):
        raise FrameFit5Error("volatile tail message lost its marker")
    tail = content[len(VOLATILE_TAIL_MARKER):]
    if tail.endswith(GREENFACT_TEXT):
        prefix = tail[:-len(GREENFACT_TEXT)].lstrip("\r\n")
        return prefix + GREENFACT_TEXT
    tail = tail.lstrip("\r\n")
    return tail or None


def _split_volatile_tail_content(content: object) -> object | None:
    if isinstance(content, str):
        return _split_marker_string(content)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = copy.deepcopy(list(content))
        for index, part in enumerate(parts):
            text = part if isinstance(part, str) else (
                part.get("text") if isinstance(part, Mapping) else None
            )
            if not isinstance(text, str) or not text.startswith(VOLATILE_TAIL_MARKER):
                continue
            tail = text[len(VOLATILE_TAIL_MARKER):].lstrip("\r\n")
            if isinstance(part, str):
                if tail:
                    parts[index] = tail
                else:
                    del parts[index]
            elif tail:
                updated = copy.deepcopy(dict(part))
                updated["text"] = tail
                parts[index] = updated
            else:
                del parts[index]
            return parts or None
    raise FrameFit5Error("volatile tail content has an unsupported shape")


def _strip_greenfact_suffix(content: object) -> tuple[object, int]:
    if isinstance(content, str):
        if content.endswith(GREENFACT_TEXT):
            return content[:-len(GREENFACT_TEXT)], 1
        return content, 0
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = copy.deepcopy(list(content))
        for index in range(len(parts) - 1, -1, -1):
            part = parts[index]
            text = part if isinstance(part, str) else (
                part.get("text") if isinstance(part, Mapping) else None
            )
            if not isinstance(text, str) or not text.endswith(GREENFACT_TEXT):
                continue
            stripped = text[:-len(GREENFACT_TEXT)]
            if isinstance(part, str):
                if stripped:
                    parts[index] = stripped
                else:
                    del parts[index]
            elif stripped:
                updated = copy.deepcopy(dict(part))
                updated["text"] = stripped
                parts[index] = updated
            else:
                del parts[index]
            return parts, 1
        return parts, 0
    return content, 0


def _volatile_tail_content(
        marker_payload: object | None, wire_facts: Sequence[str], greenfact_count: int,
        micrograph_fact: str | None = None,
) -> object:
    if greenfact_count not in {0, 1}:
        raise FrameFit5Error("final frame-only request contains duplicate GREENFACT tails")
    structured_payload = (
        marker_payload is not None
        and not isinstance(marker_payload, str)
    )
    if not structured_payload:
        chunks: list[str] = []
        if isinstance(marker_payload, str) and marker_payload:
            chunks.append(marker_payload)
        chunks.extend(wire_facts)
        if micrograph_fact is not None:
            chunks.append(micrograph_fact)
        content = VOLATILE_TAIL_MARKER
        if chunks:
            content += "\n\n" + "\n\n".join(chunks)
        if greenfact_count:
            content += GREENFACT_TEXT
        return content

    has_payload = bool(marker_payload)
    has_following = has_payload or bool(wire_facts) or micrograph_fact is not None
    content_parts: list[object] = [{
        "type": "text",
        "text": VOLATILE_TAIL_MARKER + ("\n\n" if has_following else ""),
    }]
    if isinstance(marker_payload, Sequence) and not isinstance(
            marker_payload, (str, bytes, bytearray)):
        content_parts.extend(copy.deepcopy(list(marker_payload)))
    elif marker_payload is not None:
        content_parts.append(copy.deepcopy(marker_payload))
    content_parts.extend(
        {"type": "text", "text": "\n\n" + fact} for fact in wire_facts
    )
    if micrograph_fact is not None:
        content_parts.append({"type": "text", "text": "\n\n" + micrograph_fact})
    if greenfact_count:
        content_parts.append({"type": "text", "text": GREENFACT_TEXT})
    return content_parts


def _is_micrograph_projection(line: str) -> bool:
    prefix = "micrograph: "
    if not line.startswith(prefix):
        return False
    try:
        prior = json.loads(line[len(prefix):])
    except (TypeError, ValueError):
        return False
    return isinstance(prior, dict) and prior.get("schema") == "stateframe-micrograph-wire/1"


def _strip_projection_lines(text: str) -> str:
    """Drop every whole line that is an assembler micrograph projection; other bytes are untouched."""
    lines = text.split("\n")
    kept = [line for line in lines if not _is_micrograph_projection(line)]
    if len(kept) == len(lines):
        return text
    joined = "\n".join(kept)
    while "\n\n\n" in joined:
        joined = joined.replace("\n\n\n", "\n\n")
    return joined.rstrip("\n") if not text.endswith("\n") else joined


def _strip_prior_micrograph_fact(payload: object) -> object:
    """Remove every prior assembler projection, even after host facts were appended."""
    def strip_text(text: str) -> str:
        segments = text.split("\n\n")
        retained = [segment for segment in segments if not _is_micrograph_projection(segment)]
        joined = text if len(retained) == len(segments) else "\n\n".join(retained)
        return _strip_projection_lines(joined)

    if isinstance(payload, str):
        return strip_text(payload) or None
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        parts: list[object] = []
        for part in payload:
            if isinstance(part, str):
                stripped = strip_text(part)
                if stripped or stripped == part:
                    parts.append(stripped)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                stripped = strip_text(part["text"])
                if stripped or stripped == part["text"]:
                    updated = copy.deepcopy(dict(part))
                    updated["text"] = stripped
                    parts.append(updated)
            else:
                parts.append(copy.deepcopy(part))
        return parts or None
    return payload


def finalize_frame_only_wire_order(
        messages: Sequence[Mapping[str, object]], *,
        micrograph_fact: str | None = None,
) -> list[dict[str, object]]:
    """Place the newest exchange before one final volatile user tail."""
    result = [copy.deepcopy(dict(message)) for message in messages]
    latest_exchange = latest_complete_tool_exchange(result)
    if latest_exchange is None:
        raise FrameFit5Error("final frame-only request has an incomplete tool exchange")
    action_messages = [message for message in result
                       if message.get("role") in {"assistant", "tool"}]
    if action_messages != list(latest_exchange):
        raise FrameFit5Error(
            "final frame-only request contains messages outside the newest tool exchange"
        )
    exchange_start = (
        next(index for index, message in enumerate(result) if message is latest_exchange[0])
        if latest_exchange else len(result)
    )
    exchange_end = exchange_start + len(latest_exchange)
    trailing_facts = result[exchange_end:]
    if any(message.get("role") != "user" for message in trailing_facts):
        raise FrameFit5Error(
            "final frame-only request has unsupported messages after the tool exchange"
        )

    marker_indexes = [
        index for index, message in enumerate(result)
        if _is_volatile_tail_message(message)
    ]
    if len(marker_indexes) != 1:
        raise FrameFit5Error(
            f"final frame-only request requires one volatile tail, found {len(marker_indexes)}"
        )
    marker_index = marker_indexes[0]
    marker_message = copy.deepcopy(result[marker_index])
    marker_payload = _split_volatile_tail_content(marker_message.get("content"))
    marker_payload, greenfact_count = _strip_greenfact_suffix(marker_payload)
    if micrograph_fact is not None:
        marker_payload = _strip_prior_micrograph_fact(marker_payload)

    wire_facts: list[str] = []
    ordinary_messages: list[dict[str, object]] = []
    for index, message in enumerate(result):
        if exchange_start <= index < exchange_end or index == marker_index:
            continue
        copied = copy.deepcopy(dict(message))
        if copied.get("role") == "user":
            content, found_greenfact = _strip_greenfact_suffix(copied.get("content"))
            greenfact_count += found_greenfact
            copied["content"] = content
            if (isinstance(content, str)
                    and content.startswith("[WIRE_TRUTH_FACT ")):
                # Wire facts can carry an earlier projection on a separate line.
                wire_facts.append(_strip_projection_lines(content) if micrograph_fact is not None
                                  else content)
                continue
        ordinary_messages.append(copied)

    if greenfact_count not in {0, 1}:
        raise FrameFit5Error("final frame-only request contains duplicate GREENFACT tails")
    volatile_tail = copy.deepcopy(marker_message)
    volatile_tail["content"] = _volatile_tail_content(
        marker_payload, wire_facts, greenfact_count, micrograph_fact,
    )
    return [*ordinary_messages, *latest_exchange, volatile_tail]


def _owner_menu_names(owner_contract: Mapping[str, object]) -> tuple[str, ...]:
    advertised = owner_contract.get("advertised_actions")
    action_names = tuple(
        value for value in advertised
        if isinstance(value, str) and value
    ) if isinstance(advertised, Sequence) and not isinstance(
        advertised, (str, bytes, bytearray)
    ) else ()
    schemas = owner_contract.get("admitted_tool_schemas")
    schema_names: list[str] = []
    if isinstance(schemas, Sequence) and not isinstance(schemas, (str, bytes, bytearray)):
        for schema in schemas:
            if not isinstance(schema, Mapping):
                continue
            function = schema.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str) and name:
                schema_names.append(name)
    if action_names and schema_names and action_names != tuple(schema_names):
        raise FrameFit5Error("owner action menu differs from admitted tool schemas")
    names = action_names or tuple(schema_names)
    if len(names) != len(set(names)):
        raise FrameFit5Error("owner action menu contains duplicate tool names")
    return names


def _replace_stale_host_menu(content: str, current_menu: str) -> str:
    lowered = content.casefold()
    if "host_blocker" not in lowered and "no model action is currently admissible" not in lowered:
        return content
    route_start = content.rfind("Typed completion route:")
    if route_start >= 0 and "host_blocker" in content[route_start:].casefold():
        route_end = content.find("\n", route_start)
        route_end = len(content) if route_end < 0 else route_end
        return content[:route_start] + current_menu + content[route_end:]
    completion_owner_start = content.rfind("[COMPLETION OWNER]")
    if completion_owner_start >= 0:
        return content[:completion_owner_start] + current_menu
    choice_start = content.rfind("[TERMINAL CHOICE]")
    if choice_start >= 0 and "no model action is currently admissible" in lowered:
        return content[:choice_start] + current_menu
    raise FrameFit5Error("stale host blocker cannot be reconciled with the admitted menu")


def attach_owner_protocol(
        messages: Sequence[Mapping[str, object]],
        source_messages: Sequence[Mapping[str, object]],
        runtime: object,
        owner_contract: object,
        *, owner_presentation: object = None,
) -> list[dict[str, object]]:
    """Move required owner protocol text inside the typed StateFrame."""
    result = [copy.deepcopy(dict(message)) for message in messages]
    if not isinstance(owner_contract, Mapping):
        return result
    state_messages = [
        message for message in result
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    if len(state_messages) != 1:
        raise FrameFit5Error(
            "owner protocol requires one frame-only StateFrame; "
            f"found {len(state_messages)}"
        )
    payload, state = _state_payload(state_messages[0])
    framefit5 = state.get("framefit5")
    if not isinstance(framefit5, dict):
        raise FrameFit5Error("owner protocol requires mutable frame-only state")
    protocol: dict[str, object] = {"schema": "stateframe-framefit5-owner/1"}
    menu_names = _owner_menu_names(owner_contract)
    if menu_names:
        route = str(owner_contract.get("route") or "ACTION")
        protocol["route"] = route
        protocol["admitted_actions"] = list(menu_names)
        protocol["menu_source"] = "request_action_contract"
        current_menu = (
            f"[Current admitted host menu: route={route}; "
            f"available actions: {', '.join(menu_names)}.]"
        )
        prior_context = framefit5.get("prior_user_context")
        if isinstance(prior_context, list):
            for row in prior_context:
                if not isinstance(row, dict):
                    continue
                content = row.get("content")
                if not isinstance(content, str):
                    continue
                reconciled = _replace_stale_host_menu(content, current_menu)
                if reconciled == content:
                    continue
                row["content"] = reconciled
                row["header"] = reconciled.splitlines()[0][:240]
                row["content_sha256"] = _sha256(reconciled.encode("utf-8"))
        for message in result:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if (message.get("role") in {"system", "developer"}
                    or content.lstrip().startswith((
                        "[TERMINAL CHOICE]", "[COMPLETION OWNER]",
                    ))):
                message["content"] = _replace_stale_host_menu(content, current_menu)
    terminal_line = owner_contract.get("terminal_line")
    if isinstance(terminal_line, str) and terminal_line.strip():
        protocol["terminal_line"] = terminal_line.strip()
    from kernel_completion_owner import (
        _contract_evidence_pin_value,
        _wire_continuation_limits,
    )
    if owner_presentation is not None:
        from kernel_request_seal import OwnerPresentation
        if not isinstance(owner_presentation, OwnerPresentation):
            raise FrameFit5Error("invalid explicit owner presentation")
        owner_presentation.validate(owner_contract)
    evidence_pin = (owner_presentation.evidence_pin if owner_presentation is not None
                    else _contract_evidence_pin_value(runtime))
    if isinstance(evidence_pin, str) and evidence_pin:
        protocol["evidence_pin"] = evidence_pin
    continuation_limits = (owner_presentation.continuation_limits if owner_presentation is not None
                           else _wire_continuation_limits(source_messages))
    if isinstance(continuation_limits, str) and continuation_limits:
        protocol["continuation_limits"] = continuation_limits
    framefit5["owner_protocol"] = protocol
    state["framefit5"] = framefit5
    payload["state"] = state
    state_messages[0]["content"] = _serialize_state_message(payload)
    if len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")) > MAX_PRODUCT_BYTES:
        raise FrameFit5Error("owner protocol exceeds the frame-only request byte limit")
    return result




def _verify_next_site_bytes(site: Mapping[str, object], workspace_root: str | Path) -> None:
    raw_path = site.get("path")
    path = _safe_relative_path(raw_path)
    line = site.get("line")
    line_range = site.get("lines")
    current_bytes = site.get("current_bytes")
    revision = site.get("revision")
    missing_fields = []
    if not isinstance(line_range, Mapping):
        missing_fields.append("lines")
    if not isinstance(current_bytes, str):
        missing_fields.append("current_bytes")
    if not isinstance(revision, str):
        missing_fields.append("revision")
    if type(line) is not int:
        missing_fields.append("line")
    if missing_fields:
        raise FrameFit5Error(
            "typed next-site receipt is incomplete: " + ", ".join(missing_fields)
        )
    from kernel_stateframe_action_ledger import _read_open_site_file_bytes

    try:
        raw = _read_open_site_file_bytes(workspace_root, path)
    except (OSError, RuntimeError) as error:
        raise _CandidateSourceChanged(
            "candidate source became unavailable before dispatch"
        ) from error
    actual_revision = "sha256:" + _sha256(raw)
    if actual_revision != revision:
        raise _CandidateSourceChanged("typed next-site revision changed before dispatch")
    raw_lines = raw.splitlines(keepends=True)
    start, end = line_range.get("start"), line_range.get("end")
    if (type(start) is not int or type(end) is not int
            or start < 1 or end < start or end > len(raw_lines)):
        raise FrameFit5Error("typed next-site range is invalid")
    excerpt = b"".join(raw_lines[start - 1:end])
    if excerpt.decode("utf-8") != current_bytes:
        raise _CandidateSourceChanged("typed next-site bytes changed before dispatch")
    if line is not None:
        if line < start or line > end:
            raise FrameFit5Error("typed next-site line is outside its source window")
        expected_line = raw_lines[line - 1].decode("utf-8").rstrip("\r\n")
        if expected_line != site.get("current_line"):
            raise _CandidateSourceChanged("typed next-site line changed before dispatch")
    if site.get("content_sha256") != _sha256(excerpt):
        raise FrameFit5Error("typed next-site excerpt digest is invalid")

def _framefit5_deferred_paths(
        messages: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[int]]:
    from kernel_request_diet import (
        _DROPPED_SOURCE_RANGES_HEADER,
        _parse_dropped_source_notice,
    )
    from kernel_request_source_window import (
        DEFERRED_SOURCE_NOTICE_HEADER,
        deferred_source_notice_paths,
    )

    paths: list[str] = []
    indexes: list[int] = []
    for index, message in enumerate(messages):
        if not _is_framefit5_source_notice(message):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith(DEFERRED_SOURCE_NOTICE_HEADER):
            notices = deferred_source_notice_paths([dict(message)])
            paths.extend(path for notice in notices for path in notice)
        elif content.startswith(_DROPPED_SOURCE_RANGES_HEADER):
            paths.extend(
                str(row["path"])
                for row in _parse_dropped_source_notice(content)
                if isinstance(row.get("path"), str)
            )
        indexes.append(index)
    normalized: list[str] = []
    for path in paths:
        safe_path = _safe_relative_path(path)
        if safe_path not in normalized:
            normalized.append(safe_path)
    return normalized, indexes


def _framefit5_delivered_source_ranges(
        messages: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Count source ranges only when source page bytes are present in message text."""
    from kernel_request_diet import (
        _LEGACY_SOURCE_PATTERN,
        _SOURCE_PATTERN,
        _source_delivery_ranges,
        source_window_path,
    )

    delivered: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        page_header = content.lstrip()
        is_source_page = (
            _SOURCE_PATTERN.match(page_header) is not None
            or _LEGACY_SOURCE_PATTERN.match(page_header) is not None
            or page_header.casefold().startswith("[read ")
            or source_window_path(content) is not None
        )
        if not is_source_page:
            continue

        # The evidence envelope can describe source coverage without carrying
        # any source bytes. The host fact is about delivered pages, so derive it
        # from the visible message body alone.
        visible_message = dict(message)
        visible_message.pop("_framebuild_evidence_envelope", None)
        for identity, spans in _source_delivery_ranges([visible_message]).items():
            delivered.setdefault(identity, []).extend(spans)
    return delivered


def _reconcile_framefit5_source_notices(
        messages: list[dict[str, object]], framefit5: Mapping[str, object],
        *, required_deferred_paths: Sequence[object] = (),
) -> None:
    deferred_paths, notice_indexes = _framefit5_deferred_paths(messages)
    required_paths = {_safe_relative_path(path) for path in required_deferred_paths}
    for raw_path in required_deferred_paths:
        path = _safe_relative_path(raw_path)
        if path not in deferred_paths:
            deferred_paths.append(path)
    last_verdict = framefit5.get("last_verdict")
    if isinstance(last_verdict, Mapping):
        verdict_status = str(last_verdict.get("status") or "").upper()
        verdict_line = str(last_verdict.get("line") or "")
        if verdict_status in {"PASS", "FAIL"} and verdict_line.startswith(
                f"LAST {verdict_status} at rev "):
            for index, message in enumerate(messages):
                content = message.get("content")
                if (isinstance(content, str)
                        and _is_legacy_generated_vacuity_certificate(message)
                        and verdict_line not in content):
                    notice_indexes.append(index)
    notice_indexes = sorted(set(notice_indexes))
    ranges = _framefit5_delivered_source_ranges(messages)
    attached_paths = {path for path, _revision in ranges}
    range_lines = [
        f"{path} lines {start}-{end} revision {revision or 'unavailable'}"
        for (path, revision), spans in sorted(ranges.items())
        for start, end in spans
    ]

    raw_sites = framefit5.get("open_sites")
    if not isinstance(raw_sites, list):
        next_site = framefit5.get("next_site")
        raw_sites = [next_site] if isinstance(next_site, Mapping) else []
    open_sites = [site for site in raw_sites if isinstance(site, Mapping)]
    next_site = framefit5.get("next_site")
    if isinstance(next_site, Mapping) and next_site not in open_sites:
        open_sites = [next_site, *open_sites]
    site_lines: list[str] = []
    for site in open_sites:
        # Compacted location inventories retain identities but not source bytes.
        if not isinstance(site.get("current_bytes"), str) or not site.get("current_bytes"):
            continue
        path = _safe_relative_path(site.get("path"))
        line = site.get("line")
        lines = site.get("lines")
        start = lines.get("start") if isinstance(lines, Mapping) else None
        end = lines.get("end") if isinstance(lines, Mapping) else None
        revision = str(site.get("revision") or "unavailable")
        site_lines.append(
            f"{path}:{line} lines {start}-{end} revision {revision}"
        )

    raw_facts = framefit5.get("read_facts")
    read_facts = raw_facts if isinstance(raw_facts, list) else []
    from kernel_reread_result_truth import read_is_owed

    current_read_paths: set[str] = set()
    read_lines: list[str] = []
    for fact in read_facts:
        if not isinstance(fact, Mapping):
            continue
        path = _safe_relative_path(fact.get("path"))
        read_revision = fact.get("read_revision")
        current_revision = fact.get("current_revision")
        receipts = fact.get("receipts")
        receipt_ids = [
            str(receipt.get("receipt_id"))
            for receipt in receipts
            if isinstance(receipt, Mapping) and receipt.get("receipt_id")
        ] if isinstance(receipts, list) else []
        receipt_text = ", ".join(receipt_ids) if receipt_ids else "receipt id unavailable"
        owed = read_is_owed(fact)
        if not owed:
            read_lines.append(
                f"{path} was read at rev {read_revision or 'unavailable'}; "
                f"current rev {current_revision or 'unavailable'}; file edited since read, "
                f"so old read-body delivery is no longer owed; journal receipts {receipt_text}"
            )
        elif (fact.get("status") == "UNCHANGED"
                and fact.get("unchanged_since_read") is True
                and isinstance(read_revision, str)
                and read_revision == current_revision
                and receipt_ids):
            if fact.get("body_delivery") == "present":
                current_read_paths.add(path)
            read_lines.append(
                f"{path} already read at rev {read_revision}, unchanged since; "
                f"current rev {current_revision}; journal receipts {receipt_text}"
            )
        elif isinstance(read_revision, str) and isinstance(current_revision, str):
            read_lines.append(
                f"{path} read at rev {read_revision}; current rev {current_revision}; "
                f"changed since read; journal receipts {receipt_text}"
            )
        elif isinstance(read_revision, str):
            read_lines.append(
                f"{path} read at rev {read_revision}; current source revision is "
                f"unavailable, so unchanged status cannot be confirmed; "
                f"journal receipts {receipt_text}"
            )
        else:
            workspace_revisions = fact.get("journal_workspace_revisions")
            revision_text = (
                ", ".join(str(revision) for revision in workspace_revisions)
                if isinstance(workspace_revisions, list) and workspace_revisions
                else "unavailable"
            )
            read_lines.append(
                f"{path} has a journal read receipt at workspace revision "
                f"{revision_text}; per-file read revision is unavailable; "
                f"current file revision {current_revision or 'unavailable'}; "
                f"journal receipts {receipt_text}"
            )
        if owed and fact.get("body_delivery") == "not_shown":
            notice = fact.get("delivery_notice")
            if not isinstance(notice, str):
                from kernel_reread_result_truth import eviction_notice

                notice = eviction_notice(path, current_revision)
            read_lines[-1] += "; " + notice

    other_paths = [
        path for path in deferred_paths
        if path in required_paths
        or (path not in attached_paths and path not in current_read_paths)
    ]
    lines = ["[Host fact]"]
    if range_lines:
        lines.append("Current source bytes attached to this request: " + "; ".join(range_lines))
    else:
        lines.append("No revision-bound source windows are attached outside the StateFrame.")
    if read_lines:
        lines.append("Journal-backed source read facts: " + "; ".join(read_lines))
    if site_lines:
        lines.append("Open edit sites verified against current workspace bytes: " + "; ".join(site_lines))
    if other_paths:
        lines.append(
            "Other required source paths are not attached; read one only if your edit needs it: "
            + ", ".join(other_paths)
        )

    deferred_notice = None
    if other_paths:
        from kernel_request_source_window import _render_truthful_deferred_source_notice

        delivered_ranges = [
            (path, revision, start, end)
            for (path, revision), spans in sorted(ranges.items())
            for start, end in spans
        ]
        deferred_notice = {
            "role": "user",
            "content": _render_truthful_deferred_source_notice(
                other_paths, delivered_source_ranges=delivered_ranges,
            ),
            "_request_budget_section": "stage_evidence",
        }

    notice = {
        "role": "user",
        "content": "\n".join(lines),
        "_request_budget_section": "stage_evidence",
    }
    remove = set(notice_indexes)
    cleaned = [message for index, message in enumerate(messages) if index not in remove]
    latest_exchange = latest_complete_tool_exchange(cleaned)
    if latest_exchange is None:
        exchange_start = len(cleaned)
    elif latest_exchange:
        exchange_start = next(
            index for index, message in enumerate(cleaned) if message is latest_exchange[0]
        )
    else:
        marker_indexes = [
            index for index, message in enumerate(cleaned)
            if _is_volatile_tail_message(message)
        ]
        exchange_start = marker_indexes[0] if len(marker_indexes) == 1 else len(cleaned)
    state_indexes = [
        index for index, message in enumerate(cleaned)
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    state_end = state_indexes[0] + 1 if len(state_indexes) == 1 else 0
    requested_insert = notice_indexes[0] if notice_indexes else exchange_start
    insert_at = min(exchange_start, max(state_end, requested_insert))
    cleaned.insert(insert_at, notice)
    if deferred_notice is not None:
        cleaned.insert(insert_at + 1, deferred_notice)
    if isinstance(messages, list):
        messages[:] = cleaned


def _refresh_framefit5_sources(
        messages: list[dict[str, object]], runtime: object,
        *, frame_request_allocation: FrameRequestAllocation | None = None,
) -> None:
    if frame_request_allocation is not None:
        from kernel_frame_allocation_validation import validate_allocation_authority
        validate_allocation_authority(frame_request_allocation, runtime, messages)
        return
    state_messages = [
        message for message in messages
        if isinstance(message, Mapping)
        and str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    if len(state_messages) != 1 or not isinstance(state_messages[0], dict):
        raise FrameFit5Error("final source refresh requires one mutable StateFrame")
    state_message = state_messages[0]
    payload, state = _state_payload(state_message)
    current_frame = state.get("framefit5")
    if not isinstance(current_frame, dict):
        raise FrameFit5Error("final source refresh requires mutable FRAMEFIT5 state")

    action_context = getattr(runtime, "_framefit_action_context", None)
    root = _workspace_root(runtime)
    if isinstance(action_context, Mapping):
        sites = _next_sites(runtime, action_context)
        ledger = current_frame.get("action_ledger")
        if not isinstance(ledger, Mapping):
            raise FrameFit5Error("final source refresh lacks its typed action ledger")
        read_facts = _journal_read_facts(action_context, root, ledger)
        # Fresh journal facts replace the prior projection, including its fold
        # summary; retaining that summary would count the same history twice.
        current_frame.pop("historical_read_facts", None)
        history = action_context.get("journal_receipt_history")
        if isinstance(history, Mapping) and history["historical_read_facts"]["count"]:
            current_frame["historical_read_facts"] = copy.deepcopy(history["historical_read_facts"])
    else:
        raw_sites = current_frame.get("open_sites")
        if isinstance(raw_sites, list):
            sites = [copy.deepcopy(dict(site)) for site in raw_sites
                     if isinstance(site, Mapping)]
        else:
            site = current_frame.get("next_site")
            sites = [copy.deepcopy(dict(site))] if isinstance(site, Mapping) else []
        raw_facts = current_frame.get("read_facts")
        read_facts = copy.deepcopy(raw_facts) if isinstance(raw_facts, list) else []

    from kernel_open_site_capacity import LOCATION_MODE, expand_location_inventory
    if (not isinstance(action_context, Mapping)
            and current_frame.get("site_inventory_encoding") == LOCATION_MODE):
        sites = expand_location_inventory(current_frame, root)
    current_frame.pop("site_inventory_encoding", None)
    for site in sites:
        _verify_next_site_bytes(site, root)
    next_site = sites[0] if sites else None
    current_frame["open_sites"] = sites
    current_frame["open_edit_site_count"] = len(sites)
    current_frame["read_facts"] = read_facts
    current_frame["next_site"] = next_site
    open_work = current_frame.get("open_work")
    if not isinstance(open_work, dict):
        open_work = {}
    open_work["next_site_available"] = next_site is not None
    open_work["open_edit_site_count"] = len(sites)
    open_work["next_site_source"] = (
        next_site.get("source") if isinstance(next_site, Mapping) else "unavailable"
    )
    current_frame["open_work"] = open_work
    applied_edits = current_frame.get("applied_edits")
    if not isinstance(applied_edits, list):
        applied_edits = []
    if frame_request_allocation is None:
        _update_headcarry(
            current_frame,
            runtime=runtime,
            root=root,
            open_sites=sites,
            applied_edits=applied_edits,
            messages=messages,
        )
    if frame_request_allocation is not None:
        from kernel_frame_optional_allocation import render_optional_selection, identity
        inventory = frame_request_allocation.site_inventory
        if [identity(site) for site in sites] != inventory["inventory_identity"]:
            from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
            raise FrameAllocationReplanNeeded("semantic site inventory changed")
        current_frame = render_optional_selection(
            current_frame, frame_request_allocation.optional_heads, inventory, root=root,
        )
    state["framefit5"] = current_frame
    payload["state"] = state
    state_message["content"] = _serialize_state_message(payload)
    refreshed_ledger = current_frame.get("action_ledger")
    refreshed_reads = refreshed_ledger.get("reads") if isinstance(refreshed_ledger, Mapping) else None
    if isinstance(refreshed_reads, list) and any(
            isinstance(row, Mapping) and row.get("range_delivery") == "same_request"
            for row in refreshed_reads):
        from kernel_stateframe_capacity import refresh_read_ranges

        refresh_read_ranges(messages, action_context)
        current_frame = _state_payload(state_message)[1]["framefit5"]

    from kernel_reread_result_truth import reconcile_source_truth

    reconcile_source_truth(
        messages, current_frame, include_host_fact=frame_request_allocation is None,
    )
    state["framefit5"] = current_frame
    payload["state"] = state
    state_message["content"] = _serialize_state_message(payload)

    if isinstance(action_context, dict) and frame_request_allocation is None:
        action_context["framefit5"] = copy.deepcopy(current_frame)
        if isinstance(next_site, Mapping):
            action_context["open_edit_site"] = copy.deepcopy(dict(next_site))
        else:
            action_context.pop("open_edit_site", None)


def _validated_exchange_and_volatile_tail(
        messages: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], int, int]:
    latest_exchange = latest_complete_tool_exchange(messages)
    if latest_exchange is None:
        raise FrameFit5Error("latest assistant tool-call exchange is incomplete or mismatched")
    action_messages = [message for message in messages
                       if message.get("role") in {"assistant", "tool"}]
    if action_messages != list(latest_exchange):
        raise FrameFit5Error(
            "frame-only request may contain only the newest complete tool exchange"
        )
    marker_indexes = [
        index for index, message in enumerate(messages)
        if _is_volatile_tail_message(message)
    ]
    if len(marker_indexes) != 1:
        raise FrameFit5Error(
            f"frame-only request requires one volatile tail, found {len(marker_indexes)}"
        )
    tail_index = marker_indexes[0]
    if tail_index != len(messages) - 1:
        raise FrameFit5Error("frame-only volatile tail must be the final user message")
    exchange_start = tail_index - len(latest_exchange)
    if exchange_start < 0 or list(messages[exchange_start:tail_index]) != list(latest_exchange):
        raise FrameFit5Error(
            "frame-only request must place the newest complete tool exchange immediately before its volatile tail"
        )
    return latest_exchange, exchange_start, tail_index


def validate_product_request(
        messages: Sequence[Mapping[str, object]], runtime: object,
        tools: object = None, *, refresh_headcarry: bool = True,
        frame_request_allocation: FrameRequestAllocation | None = None,
        frame_selection: object = None,
) -> dict[str, object]:
    """Verify frame-only shape, newest tool pairing, task identity, and candidate bytes."""
    from kernel_frame_allocation_validation import (
        require_active_allocation, validate_allocation_authority,
    )
    require_active_allocation(runtime, frame_request_allocation)
    if frame_request_allocation is not None:
        validate_allocation_authority(frame_request_allocation, runtime, messages, tools)
        refresh_headcarry = False
    from kernel_cache_layout_view import is_cache_layout, semantic_messages
    if is_cache_layout(messages):
        messages = semantic_messages(messages)
        refresh_headcarry = False
    if not is_frame_only(messages):
        _raise_specific_framefit5_errors_before_shape(messages)
        raise FrameFit5Error("request does not satisfy the frame-only message contract")
    if any(
            "STATEFRAME_FRAMEFIT_KEEP_K" in str(message.get("content") or "")
            for message in messages):
        raise FrameFit5Error("diagnostic KEEP_K leaked into the product request")
    roles = [message.get("role") for message in messages]
    system_count = next(
        (index for index, role in enumerate(roles) if role != "system"), len(roles)
    )
    if system_count < 1 or any(role == "system" for role in roles[system_count:]):
        raise FrameFit5Error("frame-only request requires leading system messages")
    if any(role not in {"system", "user", "assistant", "tool"} for role in roles):
        raise FrameFit5Error("frame-only request contains an unsupported message role")
    if len(messages) < system_count + 2:
        raise FrameFit5Error("frame-only request is missing task or StateFrame")
    task = messages[system_count]
    state_message = messages[system_count + 1]
    if (task.get("role") != "user"
            or not str(task.get("content") or "").startswith("Task:")):
        raise FrameFit5Error("frame-only request must place the original task after system")
    if (state_message.get("role") != "user"
            or not str(state_message.get("content") or "").startswith(STATE_HEADER + "\n")):
        raise FrameFit5Error("frame-only request must place StateFrame after the task")
    latest_exchange, exchange_start, _tail_index = _validated_exchange_and_volatile_tail(
        messages,
    )
    if any(message.get("role") != "user"
           for message in messages[system_count + 2:exchange_start]):
        raise FrameFit5Error("frame-only current facts must use user messages before the tool exchange")
    if any(str(message.get("content") or "").startswith(HANDOFF_HEADER)
           for message in messages):
        raise FrameFit5Error("frame-only request contains an F4 decision handoff")
    _payload, state = _state_payload(state_message)
    framefit5 = state["framefit5"]
    task_state = framefit5.get("task")
    if (not isinstance(task_state, Mapping)
            or task_state.get("sha256") != _sha256(
                str(task.get("content") or "").encode("utf-8"))):
        raise FrameFit5Error("typed frame task identity does not match the submitted task")
    ledger = framefit5.get("action_ledger")
    if not isinstance(ledger, Mapping):
        raise FrameFit5Error("typed frame action ledger is missing")
    if not isinstance(ledger.get("reads"), list) or not isinstance(ledger.get("edits"), list):
        raise FrameFit5Error("typed frame action ledger lists are malformed")
    from kernel_stateframe_capacity import validate_read_ranges, validate_coverage

    if isinstance(framefit5.get("headcarry"), Mapping) and isinstance(state_message, dict):
        # verified heads are read-delivery carriers. Allocation paths re-render the
        # optional heads after the last delivery reconcile, so derive delivery claims from the
        # carriers in THIS final frame before validating it (the validator still runs below).
        from kernel_reread_result_truth import reconcile_frame

        before_facts = copy.deepcopy(framefit5.get("read_facts"))
        reconcile_frame(framefit5, messages)
        if framefit5.get("read_facts") != before_facts:
            _payload["state"] = state
            state_message["content"] = _serialize_state_message(_payload)
    validate_read_ranges(framefit5, messages)
    validate_coverage(framefit5, getattr(runtime, "_framefit_action_context", None))
    if not isinstance(framefit5.get("applied_edits"), list):
        raise FrameFit5Error("typed frame journal applied edits are missing")
    verdict = framefit5.get("last_verdict")
    if not isinstance(verdict, Mapping) or not isinstance(verdict.get("line"), str):
        raise FrameFit5Error("typed frame journal verdict is missing")
    if verdict.get("status") is not None and "GATE_NEVER_RAN" in str(verdict.get("line")):
        raise FrameFit5Error("typed frame verdict contradicts its journal status")
    next_site = framefit5.get("next_site")
    raw_open_sites = framefit5.get("open_sites")
    if raw_open_sites is None:
        if not isinstance(next_site, Mapping):
            raise FrameFit5Error("typed frame has no current next-site or last-action fallback")
        open_sites = [next_site]
    elif (not isinstance(raw_open_sites, list)
            or any(not isinstance(site, Mapping) for site in raw_open_sites)):
        raise FrameFit5Error("typed frame open edit-site inventory is malformed")
    else:
        open_sites = raw_open_sites
    from kernel_open_site_capacity import (
        LOCATION_MODE, expand_location_inventory, validate_location_routes,
    )
    compact_inventory = framefit5.get("site_inventory_encoding") == LOCATION_MODE
    verified_sites = open_sites
    if compact_inventory:
        verified_sites = expand_location_inventory(framefit5, _workspace_root(runtime))
        validate_location_routes(framefit5, runtime, tools)
    headcarry_mode = _headcarry_mode()
    current_headcarry = framefit5.get("headcarry")
    headcarry_changed = False
    if (headcarry_mode == "on" or isinstance(current_headcarry, Mapping)):
        if not getattr(runtime, "_framefit5_headcarry_failed", False):
            try:
                selection = frame_request_allocation or frame_selection
                if selection is not None:
                    from kernel_frame_optional_allocation import render_optional_selection
                    projected = render_optional_selection(
                        framefit5, selection.optional_heads,
                        selection.site_inventory, root=_workspace_root(runtime),
                    )
                    fresh_headcarry = projected.get("headcarry")
                else:
                    from kernel_stateframe_headcarry import (
                        headcarry_owed_reads_enabled,
                        owed_read_rows,
                        render_headcarry,
                    )

                    # T2FIX: same owed-read inputs as _update_headcarry, or this final
                    # re-render would silently drop the heads the frame just promised.
                    fresh_headcarry = _json_value(render_headcarry(
                        _workspace_root(runtime),
                        open_sites,
                        framefit5["applied_edits"],
                        messages=messages,
                        wire_sites=[next_site] if compact_inventory else open_sites,
                        owed_reads=(owed_read_rows(framefit5.get("read_facts"))
                                    if headcarry_owed_reads_enabled() else []),
                    ))
                    if not isinstance(fresh_headcarry, dict):
                        raise TypeError("head-carry renderer did not return a JSON object")
            except Exception as error:
                if not refresh_headcarry:
                    raise FrameFit5Error(
                        "final typed head-carry could not be verified"
                    ) from error
                framefit5.pop("headcarry", None)
                headcarry_changed = current_headcarry is not None
                try:
                    setattr(runtime, "_framefit5_headcarry_failed", True)
                except Exception:
                    pass
                if isinstance(state_message, dict):
                    _payload["state"] = state
                    state_message["content"] = _serialize_state_message(_payload)
            else:
                if current_headcarry != fresh_headcarry:
                    if not refresh_headcarry:
                        raise FrameFit5Error(
                            "final typed head-carry changed after assembly"
                        )
                    framefit5["headcarry"] = fresh_headcarry
                    headcarry_changed = True
                    payload = {
                        **_payload,
                        "state": state,
                    }
                    state_message["content"] = _serialize_state_message(payload)
    if headcarry_changed:
        # Headcarry is an exact read carrier. If host verification changes its
        # bytes, the previously checked delivery facts no longer describe the
        # final serialized frame.
        from kernel_reread_result_truth import reconcile_frame

        reconcile_frame(framefit5, messages)
        if not isinstance(state_message, dict):
            raise FrameFit5Error("refreshed head-carry cannot be serialized")
        _payload["state"] = state
        state_message["content"] = _serialize_state_message(_payload)
        validate_read_ranges(framefit5, messages)
    open_work = framefit5.get("open_work")
    if not isinstance(open_work, Mapping):
        raise FrameFit5Error("typed frame open-work state is missing")
    if raw_open_sites is not None:
        site_count = open_work.get("open_edit_site_count")
        if open_sites:
            if not isinstance(next_site, Mapping) or dict(verified_sites[0]) != dict(next_site):
                raise FrameFit5Error(
                    "typed next-site alias is not the first open inventory site"
                )
            if (open_work.get("next_site_available") is not True
                    or type(site_count) is not int or site_count != len(open_sites)):
                raise FrameFit5Error("typed frame open-work inventory is inconsistent")
        elif (next_site is not None
                or open_work.get("next_site_available") is not False
                or type(site_count) is not int or site_count != 0):
            raise FrameFit5Error("typed frame empty open-site inventory is inconsistent")
    elif not isinstance(next_site, Mapping):
        raise FrameFit5Error("typed frame has no current next-site or last-action fallback")
    if open_sites:
        owner_root = _workspace_root(runtime)
        for site in verified_sites:
            if site.get("source") not in {
                "remaining_sites_inventory", "last_read_or_edit_region",
            }:
                raise FrameFit5Error(
                    "typed next-site source is not an admitted candidate source"
                )
            if not isinstance(site.get("revision"), str):
                raise FrameFit5Error("final typed next-site revision is missing")
            _verify_next_site_bytes(site, owner_root)
    raw_read_facts = framefit5.get("read_facts", [])
    if (not isinstance(raw_read_facts, list)
            or any(not isinstance(fact, Mapping) for fact in raw_read_facts)):
        raise FrameFit5Error("typed journal read facts are malformed")
    for fact in raw_read_facts:
        if fact.get("status") == "UNCHANGED" and (
                fact.get("unchanged_since_read") is not True
                or not isinstance(fact.get("read_revision"), str)
                or fact.get("read_revision") != fact.get("current_revision")
                or not isinstance(fact.get("receipts"), list)
                or not fact["receipts"]):
            raise FrameFit5Error("typed journal read fact overstates unchanged source bytes")
    if tools is not None:
        if (not isinstance(tools, Sequence)
                or isinstance(tools, (str, bytes, bytearray))
                or any(not isinstance(tool, Mapping) for tool in tools)):
            raise FrameFit5Error("frame-only tool schemas are malformed")
    return {
        "checked_before_provider": True,
        "presentation_arm": "frame",
        "message_count": len(messages),
        "history_roles": roles,
        "automatic_frame_present": True,
        "automatic_working_source_present": False,
        "framefit5_schema": SCHEMA,
        "framefit5_mode": PRODUCT_MODE,
        "tool_schema_count": len(tools) if tools is not None else 0,
        "next_site_path": next_site.get("path") if isinstance(next_site, Mapping) else None,
        "next_site_revision": next_site.get("revision") if isinstance(next_site, Mapping) else None,
        "tools_separate_from_messages": tools is not None,
    }


def _raise_specific_framefit5_errors_before_shape(
        messages: Sequence[Mapping[str, object]],
) -> None:
    """Keep task/site failures visible when the final-wire shape check also fails."""
    roles = [message.get("role") for message in messages]
    system_count = next(
        (index for index, role in enumerate(roles) if role != "system"), len(roles)
    )
    if (system_count < 1 or any(role == "system" for role in roles[system_count:])
            or len(messages) < system_count + 2):
        return
    task = messages[system_count]
    state_message = messages[system_count + 1]
    if (task.get("role") != "user"
            or not str(task.get("content") or "").startswith("Task:")
            or state_message.get("role") != "user"
            or not str(state_message.get("content") or "").startswith(STATE_HEADER + "\n")):
        return
    try:
        _payload, state = _state_payload(state_message)
    except FrameFit5Error:
        return
    framefit5 = state.get("framefit5")
    if not isinstance(framefit5, Mapping):
        return
    task_state = framefit5.get("task")
    if (not isinstance(task_state, Mapping)
            or task_state.get("sha256") != _sha256(
                str(task.get("content") or "").encode("utf-8"))):
        raise FrameFit5Error("typed frame task identity does not match the submitted task")
    if (framefit5.get("open_sites") is None
            and not isinstance(framefit5.get("next_site"), Mapping)):
        raise FrameFit5Error("typed frame has no current next-site or last-action fallback")


def refresh_final_product_inputs(
        runtime: object,
        messages: list[dict[str, object]],
        *, frame_request_allocation: FrameRequestAllocation | None = None,
) -> None:
    """Refresh live FrameFit5 source bindings before shared final assembly."""
    from kernel_frame_allocation_validation import (
        require_active_allocation, validate_allocation_authority,
    )
    require_active_allocation(runtime, frame_request_allocation)
    if frame_request_allocation is not None:
        validate_allocation_authority(
            frame_request_allocation, runtime, messages,
        )
        return
    if any(not isinstance(row, dict) for row in messages):
        raise FrameFit5Error("final frame-only messages are not refreshable")

    last_change: _CandidateSourceChanged | None = None
    for _attempt in range(MAX_SOURCE_REFRESH_ATTEMPTS):
        try:
            if frame_request_allocation is None:
                _refresh_framefit5_sources(messages, runtime)
            else:
                _refresh_framefit5_sources(
                    messages, runtime, frame_request_allocation=frame_request_allocation,
                )
            return
        except _CandidateSourceChanged as error:
            last_change = error
    if last_change is not None:
        raise FrameFit5Error(
            "candidate source kept changing before final frame assembly"
        ) from last_change


def validate_final_product_boundary(
        runtime: object,
        messages: Sequence[Mapping[str, object]],
        tools: object,
        *,
        refresh_sources: bool = True,
        refresh_headcarry: bool = True,
        frame_request_allocation: FrameRequestAllocation | None = None,
        frame_selection: object = None,
) -> dict[str, object]:
    """Validate final dispatch messages; optional refreshes must precede assembly."""
    from kernel_context import token_estimate

    if frame_request_allocation is not None:
        refresh_sources = False
        refresh_headcarry = False

    ceiling = getattr(runtime, "frame_ceiling", None)
    if type(ceiling) is not int or ceiling <= 0:
        raise FrameFit5Error("frame-only request has no valid token ceiling")
    if not isinstance(messages, list) or any(not isinstance(row, dict) for row in messages):
        raise FrameFit5Error("final frame-only messages are not refreshable")

    if refresh_sources:
        last_change: _CandidateSourceChanged | None = None
        for _attempt in range(MAX_SOURCE_REFRESH_ATTEMPTS):
            try:
                if frame_request_allocation is None:
                    _refresh_framefit5_sources(messages, runtime)
                else:
                    _refresh_framefit5_sources(
                        messages, runtime, frame_request_allocation=frame_request_allocation,
                    )
                validation = validate_product_request(
                    messages, runtime, tools=tools,
                    refresh_headcarry=refresh_headcarry,
                    frame_request_allocation=frame_request_allocation,
                    frame_selection=frame_selection,
                )
                last_change = None
                break
            except _CandidateSourceChanged as error:
                last_change = error
        if last_change is not None:
            raise FrameFit5Error(
                "candidate source kept changing at the final dispatch boundary"
            ) from last_change
    else:
        validation = validate_product_request(
            messages, runtime, tools=tools,
            refresh_headcarry=refresh_headcarry,
            frame_request_allocation=frame_request_allocation,
            frame_selection=frame_selection,
        )

    estimate = token_estimate({"messages": messages, "tools": tools})
    if estimate > ceiling:
        raise FrameFit5Error("final frame-only request exceeds its token ceiling")
    from kernel_stateframe_f4 import assert_f4_invariants
    assert_f4_invariants(messages)
    return validation


def has_framefit5_state(messages: Sequence[Mapping[str, object]]) -> bool:
    """Recognize product state even when another check finds its JSON malformed."""
    for message in messages:
        content = message.get("content")
        if (message.get("role") != "user" or not isinstance(content, str)
                or not content.startswith(STATE_HEADER + "\n")):
            continue
        if '"framefit5"' in content or SCHEMA in content:
            return True
    return False


def is_frame_only(messages: Sequence[Mapping[str, object]]) -> bool:
    state_blocks = [message for message in messages
                    if str(message.get("content") or "").startswith(STATE_HEADER + "\n")]
    if len(state_blocks) != 1 or not has_framefit5_state(messages):
        return False
    try:
        _state_payload(state_blocks[0])
        _validated_exchange_and_volatile_tail(messages)
    except FrameFit5Error:
        return False
    return not any(
        str(message.get("content") or "").startswith(HANDOFF_HEADER)
        for message in messages
    )


__all__ = [
    "FrameFit5Error", "FrameFit5ProducerError", "MAX_PRODUCT_BYTES",
    "PRODUCT_MODE", "SCHEMA", "assemble_framefit5_request", "attach_owner_protocol",
    "attach_typed_state", "build_typed_state", "compose_frame_only", "fit_headcarry_to_request_budget",
    "finalize_frame_only_wire_order", "has_framefit5_state", "is_frame_only",
    "validate_final_product_boundary",
    "validate_no_verbatim_turns", "validate_product_request",
]
