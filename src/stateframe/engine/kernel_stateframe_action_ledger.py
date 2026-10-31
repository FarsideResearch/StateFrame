"""Typed, bounded action evidence for the opt-in StateFrame FRAMEFIT product.

Only closed tool exchanges with complete receipts are eligible for ledger coverage.
Unsupported exchanges are returned as private inputs for typed receipts; they are
never appended to the product request. KEEP_K feeds a separate diagnostic only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

SCHEMA = "stateframe-framefit3-action-ledger/1"
OPEN_EDIT_SITE_SCHEMA = "stateframe-framefit4-open-edit-site/1"
KEEP_K_ENV = "STATEFRAME_FRAMEFIT_KEEP_K"
MAX_KEEP_K = 32
MAX_LEDGER_BYTES = 16_384
MAX_EDIT_TEXT_BYTES = 4_096
MAX_ACTION_ARGUMENT_BYTES = 4_096
MAX_OPEN_EDIT_SITE_LINES = 80
MAX_OPEN_EDIT_SITE_FILE_BYTES = 8 * 1024 * 1024
MAX_OPEN_EDIT_SITE_BYTES = 64 * 1024
FRAMEFIT5_ACTION_CONTEXT_ENV = "STATEFRAME_FRAMEFIT5_ACTION_CONTEXT"
ACTION_CONTEXT_MODE_ENV = FRAMEFIT5_ACTION_CONTEXT_ENV
MISSING_EXACT_CURRENT_SITE = "missing_exact_current_site"
_READ_HEADER = re.compile(
    r"^\[read page\]\s+path=(.*?)\s+revision=(sha256:[0-9a-f]{64})\s+"
    r"lines=(\d+)-(\d+)/(\d+)(.*?)$",
    re.IGNORECASE,
)


class ActionLedgerError(ValueError):
    """The source transcript cannot be segmented into safe complete turns."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def framefit5_action_context_mode(environ: Mapping[str, str] | None = None) -> str:
    """Select the opt-in missing-site degradation; invalid configuration is legacy-off."""
    source = os.environ if environ is None else environ
    value = (source.get(FRAMEFIT5_ACTION_CONTEXT_ENV, "off") or "off").strip().lower()
    return value if value in {"off", "shadow", "on"} else "off"


def framefit_keep_k(environ: Mapping[str, str] | None = None) -> int:
    """Parse a bounded, non-negative keep-last-K count; the default disables it."""
    source = os.environ if environ is None else environ
    raw = source.get(KEEP_K_ENV, "0")
    if not isinstance(raw, str) or not re.fullmatch(r"\d+", raw.strip()):
        raise ValueError(f"{KEEP_K_ENV} must be an integer from 0 to {MAX_KEEP_K}")
    value = int(raw.strip())
    if value > MAX_KEEP_K:
        raise ValueError(f"{KEEP_K_ENV} must be an integer from 0 to {MAX_KEEP_K}")
    return value


def _safe_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    if not raw or any(char in raw for char in ("\x00", "\n", "\r")):
        return ""
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(
        part == ".." or part.casefold() == "backups" for part in candidate.parts
    ):
        return ""
    normalized = candidate.as_posix()
    return "" if normalized == "." else normalized


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt_line(result_text: str) -> str:
    line = result_text.splitlines()[0] if result_text else ""
    printable = "".join(character if character.isprintable() else " " for character in line)
    return " ".join(printable.split())[:240]


def _call_argument_bytes(call: Mapping[str, object]) -> bytes:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return b""
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, str):
        return raw_arguments.encode("utf-8")
    return json.dumps(
        raw_arguments, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")


def _bounded_action_outcome(name: str, path: str, result_text: str) -> str:
    normalized = result_text.casefold()
    if (
        name in {"str_replace", "edit", "edit_file"}
        and normalized.startswith(("str_replace applied.", "edit applied."))
    ):
        return "APPLIED"
    if (
        name == "file_create"
        and path
        and result_text.startswith(f"created {path}\n")
    ):
        return "APPLIED"
    first_line = _receipt_line(result_text).casefold()
    if any(marker in first_line for marker in (
        "reject", "refus", "fail", "error", "unable", "could not",
        "not applied", "no match", "does not match",
    )):
        return "REJECTED"
    return "CLOSED"


def _bounded_action_receipt(
    call_id: str,
    name: str,
    arguments: object,
    argument_bytes: bytes,
    result_text: str,
    *,
    turn_index: int | None = None,
    journal: object = None,
) -> dict[str, object]:
    from kernel_wire_identity import public_wire_text

    result_text = public_wire_text(result_text, journal)
    raw_path = (
        arguments.get("path", arguments.get("target_path"))
        if isinstance(arguments, Mapping) else None
    )
    path = _safe_path(raw_path)
    receipt: dict[str, object] = {
        "schema": "stateframe-bounded-action-receipt/1",
        "kind": "bounded_action_receipt",
        "call_id": call_id[:160],
        "tool": name[:120] or "unknown",
        "path": path if path and len(path.encode("utf-8")) <= 512 else None,
        "outcome": _bounded_action_outcome(name, path, result_text),
        "argument_size_bytes": len(argument_bytes),
        "argument_sha256": hashlib.sha256(argument_bytes).hexdigest(),
    }
    if turn_index is not None:
        receipt["turn_index"] = turn_index
    if len(call_id) > 160:
        receipt["call_id_sha256"] = _sha256(call_id)
    if len(name) > 120:
        receipt["tool_sha256"] = _sha256(name)
    if path and len(path.encode("utf-8")) > 512:
        receipt["path_sha256"] = _sha256(path)
    result_bytes = result_text.encode("utf-8")
    receipt.update({
        "result_first_line": _receipt_line(result_text),
        "result_size_bytes": len(result_bytes),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    })
    return receipt


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _arguments(call: Mapping[str, object]) -> dict[str, object] | None:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    raw = function.get("arguments")
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _tool_name(call: Mapping[str, object]) -> str:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip().rsplit(".", 1)[-1].casefold()


def _candidate_revision(messages: Sequence[Mapping[str, object]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        if "WIRE_TRUTH_FACT" not in text or "candidate_revision:" not in text:
            continue
        match = re.search(r"candidate_revision:\s*(sha256:[0-9a-f]{64})", text, re.I)
        if match:
            return match.group(1).lower()
    return ""


def _authenticated_footer_read(result_text: str, path: str, revision: str) -> str | None:
    """Use the source verifier's authenticated extent, never strip a guessed footer.

    A complete page proves its own revision. A partial historical page without
    its full frozen source remains opaque; the capacity owner still covers it.
    """
    from kernel_request_source_window import _read_page_source_range

    try:
        parsed = _read_page_source_range(result_text, {}, {path: revision})
    except UnicodeError:
        return None
    if parsed is None or parsed[0] != path or not isinstance(parsed[2], str):
        return None
    return parsed[2]


def _read_fact(
    arguments: Mapping[str, object], result_text: str, call_id: str
) -> tuple[dict[str, object], str] | None:
    header, first_sep, rest = result_text.partition("\n")
    _notice, second_sep, content = rest.partition("\n")
    match = _READ_HEADER.fullmatch(header)
    if not first_sep or not second_sep or match is None:
        return None
    path = _safe_path(match.group(1))
    revision = match.group(2).lower()
    start, end, total = (int(match.group(index)) for index in (3, 4, 5))
    metadata = match.group(6)
    chars = re.search(r"\bchars=(\d+)\b", metadata)
    requested_path = _safe_path(arguments.get("path", arguments.get("file")))
    if (
        not path
        or (requested_path and requested_path != path)
        or start < 1
        or end < start
        or total < end
        or chars is None
        or "clip=none" not in metadata.casefold()
        or "decoded_with_replacement=true" in metadata.casefold()
    ):
        return None
    declared_chars = int(chars.group(1))
    if len(content) > declared_chars:
        authenticated = _authenticated_footer_read(result_text, path, revision)
        if authenticated is None:
            return None
        content = authenticated
    if len(content) != declared_chars:
        missing_terminal_lf = (
            declared_chars == len(content) + 1
            and end < total
            and bool(content)
            and not content.endswith(chr(10))
            and content.count(chr(10)) == end - start
        )
        if not missing_terminal_lf:
            return None
        content += chr(10)
    if len(content.splitlines(keepends=True)) != end - start + 1:
        return None
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    fact: dict[str, object] = {
        "kind": "read",
        "call_ids": [call_id],
        "path": path,
        "lines": {"start": start, "end": end, "total": total},
        "revision": revision,
        "content_sha256": _sha256(content),
    }
    return fact, content


def _read_site_window(
    arguments: Mapping[str, object], result_text: str,
) -> tuple[dict[str, object], str] | None:
    """Parse an exact delivered read window without treating it as a full-file fact."""
    header, first_sep, rest = result_text.partition("\n")
    _notice, second_sep, content = rest.partition("\n")
    match = _READ_HEADER.fullmatch(header)
    if not first_sep or not second_sep or match is None:
        return None
    path = _safe_path(match.group(1))
    revision = match.group(2).lower()
    start, end, total = (int(match.group(index)) for index in (3, 4, 5))
    metadata = match.group(6)
    requested_path = _safe_path(arguments.get("path", arguments.get("file")))
    chars = re.search(r"\bchars=(\d+)\b", metadata)
    byte_count = re.search(r"\bbytes=(\d+)\b", metadata)
    clip = re.search(r"\bclip=([^\s]+)", metadata, re.IGNORECASE)
    truncated = re.search(r"\btruncated=(true|false)\b", metadata, re.IGNORECASE)
    if (
        not path
        or not requested_path
        or requested_path != path
        or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
        or start < 1
        or end < start
        or total < end
        or chars is None
        or clip is None
        or clip.group(1).casefold() not in {"none", "max_lines"}
        or "decoded_with_replacement=true" in metadata.casefold()
        or (truncated is not None and clip.group(1).casefold() == "none"
            and truncated.group(1).casefold() != "false")
        or (clip.group(1).casefold() == "max_lines"
            and (truncated is None or truncated.group(1).casefold() != "true"))
    ):
        return None
    declared_chars = int(chars.group(1))
    declared_bytes = int(byte_count.group(1)) if byte_count is not None else None
    if len(content) > declared_chars:
        authenticated = _authenticated_footer_read(result_text, path, revision)
        if authenticated is None:
            return None
        content = authenticated
    if len(content) != declared_chars:
        missing_terminal_lf = (
            declared_chars == len(content) + 1
            and end < total
            and bool(content)
            and not content.endswith("\n")
            and content.count("\n") == end - start
        )
        if not missing_terminal_lf:
            return None
        content += "\n"
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if (
        (declared_bytes is not None and len(encoded) != declared_bytes)
        or len(content.splitlines(keepends=True)) != end - start + 1
    ):
        return None
    return {
        "path": path,
        "lines": {"start": start, "end": end, "total": total},
        "revision": revision,
    }, content


def _edit_fact(
    name: str,
    arguments: Mapping[str, object],
    result_text: str,
    call_id: str,
    current_reads: Mapping[str, tuple[dict[str, object], str]],
    candidate_revision: str,
    *,
    bounded_action_receipt: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    path = _safe_path(arguments.get("path", arguments.get("target_path")))
    if not path:
        return None
    if name == "file_create":
        content = arguments.get("content")
        if not isinstance(content, str) or not result_text.startswith(f"created {path}\n"):
            return None
        digest = _sha256(content)
        lines = max(1, len(content.splitlines()))
        return {
            "kind": "edit",
            "operation": "create",
            "call_ids": [call_id],
            "path": path,
            "lines": {"start": 1, "end": lines},
            "receipt": _receipt_line(result_text),
            "old_text_sha256": hashlib.sha256(b"").hexdigest(),
            "new_text_sha256": digest,
            "new_text": content,
            "post_edit_revision": "sha256:" + digest,
            "post_revision_source": "created_file_bytes",
        }
    old_text = arguments.get("old_str")
    new_text = arguments.get("new_str")
    if (
        name not in {"str_replace", "edit", "edit_file"}
        or not isinstance(old_text, str)
        or not isinstance(new_text, str)
        or not result_text.startswith(("str_replace applied.", "edit applied."))
        or not candidate_revision
    ):
        return None
    source = current_reads.get(path)
    if source is None:
        return None
    read_fact, source_text = source
    first = source_text.find(old_text)
    if first < 0 or source_text.find(old_text, first + 1) >= 0:
        return None
    read_lines = read_fact.get("lines")
    if not isinstance(read_lines, Mapping):
        return None
    start_line = int(read_lines["start"]) + source_text.count("\n", 0, first)
    covered_lines = max(1, len(old_text.splitlines()))
    fact: dict[str, object] = {
        "kind": "edit",
        "operation": name,
        "call_ids": [call_id],
        "path": path,
        "lines": {"start": start_line, "end": start_line + covered_lines - 1},
        "receipt": _receipt_line(result_text),
        "old_text_sha256": _sha256(old_text),
        "new_text_sha256": _sha256(new_text),
        "post_edit_revision": candidate_revision,
        "post_revision_source": "candidate_snapshot",
    }
    edit_text_bytes = len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
    if edit_text_bytes <= MAX_EDIT_TEXT_BYTES:
        fact["old_text"] = old_text
        fact["new_text"] = new_text
    else:
        fact["text_elided"] = True
    if bounded_action_receipt is not None:
        fact["bounded_action_receipt"] = copy.deepcopy(dict(bounded_action_receipt))
    return fact


def _edit_action_fact(
    call: Mapping[str, object],
    name: str,
    arguments: Mapping[str, object],
    result_text: str,
    call_id: str,
    current_reads: Mapping[str, tuple[dict[str, object], str]],
    candidate_revision: str,
    *,
    turn_index: int,
    journal: object = None,
) -> dict[str, object] | None:
    argument_bytes = _call_argument_bytes(call)
    bounded_receipt = (
        _bounded_action_receipt(
            call_id, name, arguments, argument_bytes, result_text,
            turn_index=turn_index,
            journal=journal,
        )
        if name in {"str_replace", "edit", "edit_file"}
        and len(argument_bytes) > MAX_ACTION_ARGUMENT_BYTES
        else None
    )
    return _edit_fact(
        name, arguments, result_text, call_id, current_reads, candidate_revision,
        bounded_action_receipt=bounded_receipt,
    )


def _test_fact(
    arguments: Mapping[str, object], result_text: str, call_id: str,
    candidate_revision: str,
) -> dict[str, object] | None:
    target = arguments.get("target")
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
        targets = [_safe_path(item) for item in target]
        targets = [item for item in targets if item]
        target_text = ", ".join(targets)
    else:
        target_text = _safe_path(target)
    if not target_text or not candidate_revision:
        return None
    upper = result_text.upper()
    passed = (
        ("ALL TESTS PASSED" in upper or "STATUS=PASS" in upper)
        and ("EXIT 0" in upper or "STATUS=PASS" in upper)
    )
    failed = (
        "STATUS=FAIL" in upper
        or re.search(r"PYTEST EXIT [1-9]\d*", upper) is not None
        or "TESTS FAILED" in upper
    )
    if not passed and not failed:
        return None
    return {
        "kind": "test",
        "call_ids": [call_id],
        "target": target_text,
        "verdict": "GREEN" if passed else "RED",
        "revision": candidate_revision,
    }


def _turn_groups(
    messages: Sequence[Mapping[str, object]],
    *,
    journal: object = None,
) -> tuple[list[dict[str, object]], set[int]]:
    def drop(reason: str, start: int, end: int, call_count: int = 0) -> None:
        consumed.update(range(start, end))
        emit = getattr(journal, "emit", None)
        if callable(emit):
            try:
                emit(
                    "action_group_dropped",
                    producer="kernel_stateframe_action_ledger",
                    reason=reason,
                    message_index=start,
                    call_count=call_count,
                    result_count=(end - start if reason == "orphan_tool_result"
                                  else max(0, end - start - 1)),
                    dispatch=False,
                )
            except Exception:
                # Diagnostic journaling must not make frame construction fail.
                pass

    groups: list[dict[str, object]] = []
    consumed: set[int] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, Mapping):
            index += 1
            continue
        role = message.get("role")
        if role == "tool":
            drop("orphan_tool_result", index, index + 1)
            index += 1
            continue
        if role != "assistant":
            index += 1
            continue
        calls = message.get("tool_calls")
        if calls is None or (
            isinstance(calls, Sequence)
            and not isinstance(calls, (str, bytes, bytearray))
            and not calls
        ):
            if str(message.get("content") or "").strip():
                standalone = copy.deepcopy(dict(message))
                groups.append({
                    "key": (f"assistant-text:{index}",),
                    "call_ids": (),
                    "messages": (standalone,),
                    "real": False,
                    "standalone": True,
                    "index": index,
                })
                consumed.add(index)
            index += 1
            continue
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            cursor = index + 1
            while (
                cursor < len(messages)
                and isinstance(messages[cursor], Mapping)
                and messages[cursor].get("role") == "tool"
            ):
                cursor += 1
            drop("malformed_tool_calls", index, cursor)
            index = cursor
            continue
        call_ids = [
            str(call.get("id") or "")
            for call in calls
            if isinstance(call, Mapping)
        ]
        if (
            len(call_ids) != len(calls)
            or any(not call_id for call_id in call_ids)
            or len(set(call_ids)) != len(call_ids)
        ):
            cursor = index + 1
            while (
                cursor < len(messages)
                and isinstance(messages[cursor], Mapping)
                and messages[cursor].get("role") == "tool"
            ):
                cursor += 1
            drop("malformed_tool_calls", index, cursor, len(calls))
            index = cursor
            continue
        cursor = index + 1
        results: list[Mapping[str, object]] = []
        while cursor < len(messages):
            candidate = messages[cursor]
            if not isinstance(candidate, Mapping) or candidate.get("role") != "tool":
                break
            results.append(candidate)
            cursor += 1
        result_ids = [str(item.get("tool_call_id") or "") for item in results]
        if len(results) != len(call_ids) or set(result_ids) != set(call_ids):
            drop("incomplete_tool_results", index, cursor, len(call_ids))
            index = cursor
            continue
        span = [copy.deepcopy(dict(message))]
        span.extend(copy.deepcopy(dict(item)) for item in results)
        ids = tuple(call_ids)
        groups.append({
            "key": ids,
            "call_ids": ids,
            "messages": tuple(span),
            "real": any(":auto_verify" not in call_id for call_id in ids),
            "index": index,
        })
        consumed.update(range(index, cursor))
        index = cursor
    return groups, consumed


def _prior_ledger(value: object) -> dict[str, object]:
    empty = {"schema": SCHEMA, "rows": []}
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        return empty
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or any(not isinstance(row, Mapping) for row in raw_rows):
        return empty
    rows = [copy.deepcopy(dict(row)) for row in raw_rows]
    for row in rows:
        kind = row.get("kind")
        call_ids = row.get("call_ids")
        if (
            not isinstance(kind, str)
            or kind not in {"read", "edit", "test"}
            or not isinstance(call_ids, list)
            or not call_ids
            or any(not isinstance(call_id, str) or not call_id for call_id in call_ids)
            or len(set(call_ids)) != len(call_ids)
        ):
            return empty
        if kind in {"read", "edit"}:
            path = _safe_path(row.get("path"))
            lines = row.get("lines")
            if (
                not path
                or path != row.get("path")
                or not isinstance(lines, Mapping)
                or type(lines.get("start")) is not int
                or type(lines.get("end")) is not int
                or lines["start"] < 1
                or lines["end"] < lines["start"]
            ):
                return empty
        if kind == "read":
            revision = row.get("revision")
            digest = row.get("content_sha256")
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                return empty
        elif kind == "edit":
            revision = row.get("post_edit_revision")
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
            ):
                return empty
        elif kind == "test":
            target = row.get("target")
            revision = row.get("revision")
            if (
                not isinstance(target, str)
                or not target.strip()
                or not isinstance(row.get("verdict"), str)
                or row["verdict"] not in {"GREEN", "RED"}
                or not isinstance(revision, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
            ):
                return empty
    ledger = {"schema": SCHEMA, "rows": rows}
    if len(_json_bytes(ledger)) > MAX_LEDGER_BYTES:
        return empty
    return ledger


def _row_call_ids(rows: Sequence[object]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        call_ids = row.get("call_ids")
        if isinstance(call_ids, Sequence) and not isinstance(call_ids, (str, bytes, bytearray)):
            result.update(item for item in call_ids if isinstance(item, str) and item)
    return result


def _emit_action_context_shadow_mismatch(
    journal: object | None,
    field: str,
    *,
    legacy_outcome: str,
    candidate_outcome: str,
) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    safe_field = field if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", field) else "projection"
    try:
        emit(
            f"framefit5_action_context_shadow_mismatch_{safe_field}",
            producer="kernel_stateframe_action_ledger",
            field=safe_field,
            legacy_outcome=legacy_outcome,
            candidate_outcome=candidate_outcome,
            dispatch=False,
        )
    except Exception:
        # Shadow evidence is advisory and must not alter the legacy outcome.
        pass


def build_framefit_action_context(
    messages: Sequence[Mapping[str, object]],
    *,
    keep_k: int = 0,
    prior_ledger: Mapping[str, object] | None = None,
    workspace_root: str | Path | None = None,
    prior_open_edit_site: Mapping[str, object] | None = None,
    journal: object | None = None,
) -> dict[str, object]:
    """Build the producer inventory and attach journal-owned call coverage."""
    from kernel_stateframe_call_coverage import bind_index

    context = _build_framefit_action_context(
        messages, keep_k=keep_k, prior_ledger=prior_ledger,
        workspace_root=workspace_root, prior_open_edit_site=prior_open_edit_site,
        journal=journal,
    )
    return bind_index(context, journal)


def _build_framefit_action_context(
    messages: Sequence[Mapping[str, object]],
    *,
    keep_k: int = 0,
    prior_ledger: Mapping[str, object] | None = None,
    workspace_root: str | Path | None = None,
    prior_open_edit_site: Mapping[str, object] | None = None,
    journal: object | None = None,
) -> dict[str, object]:
    """Apply the default-off FRAMEFIT5 missing-site policy with legacy fallback."""
    mode = framefit5_action_context_mode()
    arguments = {
        "keep_k": keep_k,
        "prior_ledger": prior_ledger,
        "workspace_root": workspace_root,
        "prior_open_edit_site": prior_open_edit_site,
        "journal": journal,
    }
    if mode == "off":
        return build_action_context(messages, **arguments)

    if mode == "shadow":
        try:
            legacy = build_action_context(messages, **arguments)
        except ActionLedgerError as error:
            if error.code != MISSING_EXACT_CURRENT_SITE:
                raise
            try:
                candidate = build_action_context(
                    messages, **arguments, degrade_missing_exact_site=True,
                )
            except Exception:
                _emit_action_context_shadow_mismatch(
                    journal, "site_context", legacy_outcome="error",
                    candidate_outcome="error",
                )
                raise error
            _emit_action_context_shadow_mismatch(
                journal, "open_edit_site", legacy_outcome="error",
                candidate_outcome="unverifiable",
            )
            if candidate.get("site_context") != {
                "status": "UNVERIFIABLE",
                "reason": MISSING_EXACT_CURRENT_SITE,
            }:
                _emit_action_context_shadow_mismatch(
                    journal, "site_context", legacy_outcome="error",
                    candidate_outcome="changed",
                )
            raise error
        try:
            candidate = build_action_context(
                messages, **arguments, degrade_missing_exact_site=True,
            )
        except Exception as error:
            _emit_action_context_shadow_mismatch(
                journal, "build_result", legacy_outcome="value",
                candidate_outcome=type(error).__name__,
            )
            return legacy
        for field in sorted(set(legacy) | set(candidate)):
            if field not in legacy or field not in candidate or legacy[field] != candidate[field]:
                _emit_action_context_shadow_mismatch(
                    journal, field, legacy_outcome="value",
                    candidate_outcome="value",
                )
        return legacy

    try:
        return build_action_context(
            messages, **arguments, degrade_missing_exact_site=True,
        )
    except Exception:
        # If the opt-in projection has any unexpected failure, run the exact
        # legacy computation before allowing an error to escape.
        return build_action_context(messages, **arguments)


def _within_utf8_limit(value: str, limit: int) -> bool:
    try:
        return len(value.encode("utf-8")) <= limit
    except UnicodeEncodeError:
        return False


def _normalise_open_site_locator(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    path = _safe_path(value.get("path"))
    raw_lines = value.get("lines")
    if not path or not isinstance(raw_lines, Mapping):
        return None
    start, end = raw_lines.get("start"), raw_lines.get("end")
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        return None
    action = value.get("last_action")
    if action not in {"read", "edit"}:
        return None
    read_revision = value.get("last_read_revision", "")
    if not isinstance(read_revision, str) or (
        read_revision
        and re.fullmatch(r"sha256:[0-9a-f]{64}", read_revision) is None
    ):
        return None
    anchor_line = value.get("anchor_line", "")
    anchor_offset = value.get("anchor_offset", 0)
    window_count = value.get("window_line_count", end - start + 1)
    if (
        not isinstance(anchor_line, str)
        or type(anchor_offset) is not int
        or anchor_offset < 0
        or type(window_count) is not int
        or window_count < 1
    ):
        return None
    observed_text = value.get("observed_text", "")
    expected_text = value.get("expected_text", "")
    if not isinstance(observed_text, str) or not isinstance(expected_text, str):
        return None
    if (
        not _within_utf8_limit(observed_text, MAX_OPEN_EDIT_SITE_BYTES)
        or not _within_utf8_limit(expected_text, MAX_OPEN_EDIT_SITE_BYTES)
    ):
        return None
    return {
        "path": path,
        "lines": {"start": start, "end": end},
        "last_action": action,
        "last_read_revision": read_revision,
        "anchor_line": anchor_line,
        "anchor_offset": anchor_offset,
        "window_line_count": min(window_count, MAX_OPEN_EDIT_SITE_LINES),
        "observed_text": observed_text,
        "expected_text": expected_text,
    }


def _matching_site_anchor(
    lines: Sequence[bytes], sources: Sequence[tuple[str, int]]
) -> tuple[int, int, bytes] | None:
    current = [line.rstrip(b"\r\n") for line in lines]
    for source, base_offset in sources:
        try:
            candidates = source.encode("utf-8").splitlines()
        except UnicodeEncodeError:
            continue
        if not candidates:
            continue
        first = candidates[0].rstrip(b"\r\n")
        if first.lstrip().startswith((b"def ", b"async def ", b"class ")):
            matches = [index for index, line in enumerate(current) if line == first]
            if len(matches) == 1:
                return matches[0], base_offset, first
        ranked = sorted(
            (
                (len(candidate.strip()), source_index, candidate.rstrip(b"\r\n"))
                for source_index, candidate in enumerate(candidates)
                if len(candidate.strip()) >= 12
            ),
            reverse=True,
        )
        for _length, source_index, candidate in ranked:
            matches = [index for index, line in enumerate(current) if line == candidate]
            if len(matches) == 1:
                return matches[0], base_offset + source_index, candidate
    return None


def _declaration_end(lines: Sequence[bytes], start_index: int) -> int | None:
    if start_index < 0 or start_index >= len(lines):
        return None
    header = lines[start_index].rstrip(b"\r\n")
    if not header.lstrip().startswith((b"def ", b"async def ", b"class ")):
        return None
    indentation = len(header) - len(header.lstrip(b" \t"))
    stop = min(len(lines), start_index + MAX_OPEN_EDIT_SITE_LINES)
    end = stop - 1
    for index in range(start_index + 1, stop):
        candidate = lines[index].rstrip(b"\r\n")
        if candidate.strip() and len(candidate) - len(candidate.lstrip(b" \t")) <= indentation:
            end = index - 1
            break
    while end > start_index and not lines[end].strip():
        end -= 1
    return end


def _read_open_site_file_bytes(
    workspace_root: str | Path, relative_path: str
) -> bytes:
    try:
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise ActionLedgerError("open edit site workspace is unavailable")
        target = (root / relative_path).resolve(strict=True)
        target.relative_to(root)
        if not target.is_file():
            raise ActionLedgerError("open edit site is not a regular file")
        with target.open("rb") as source:
            raw = source.read(MAX_OPEN_EDIT_SITE_FILE_BYTES + 1)
    except ActionLedgerError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ActionLedgerError("open edit site could not be read safely") from error
    if len(raw) > MAX_OPEN_EDIT_SITE_FILE_BYTES:
        raise ActionLedgerError("open edit site exceeds the bounded file size")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ActionLedgerError("open edit site is not UTF-8 text") from error
    return raw


def _exact_site_window(raw: bytes, source_text: str) -> tuple[int, int, str] | None:
    encoded = source_text.encode("utf-8")
    position = raw.find(encoded)
    if not encoded or position < 0 or raw.find(encoded, position + 1) >= 0:
        return None
    start = raw[:position].count(b"\n") + 1
    count = max(1, len(encoded.splitlines(keepends=True)))
    anchor = encoded.splitlines(keepends=True)[0].rstrip(b"\r\n").decode("utf-8")
    return start, count, anchor


def _anchored_site_window(
    lines: Sequence[bytes], locator: Mapping[str, object]
) -> tuple[int, int, str, int] | None:
    candidates = [
        (text, offset)
        for text, offset in (
            (str(locator.get("expected_text") or ""), 0),
            (str(locator.get("observed_text") or ""), 0),
            (
                str(locator.get("anchor_line") or ""),
                int(locator.get("anchor_offset") or 0),
            ),
        )
        if text
    ]
    match = _matching_site_anchor(lines, candidates)
    if match is None:
        return None
    current_index, offset, matched = match
    start = current_index - offset + 1
    if start < 1:
        return None
    count = int(locator.get("window_line_count") or 1)
    if str(locator["last_action"]) == "edit":
        declaration_end = _declaration_end(lines, start - 1)
        if declaration_end is not None:
            count = declaration_end - start + 2
    return start, count, matched.decode("utf-8"), offset


def _bounded_site_window(
    lines: Sequence[bytes], start: int, count: int, anchor: str, offset: int
) -> tuple[int, int, str, int]:
    total = len(lines)
    if total < 1:
        raise ActionLedgerError("open edit site is empty")
    if start > total:
        start = max(1, total - MAX_OPEN_EDIT_SITE_LINES + 1)
    start = max(1, min(start, total))
    count = max(1, min(count, MAX_OPEN_EDIT_SITE_LINES))
    end = min(total, start + count - 1)
    anchor = anchor or lines[start - 1].rstrip(b"\r\n").decode("utf-8")
    return start, end, anchor, offset


def _open_site_window(
    raw: bytes, lines: Sequence[bytes], locator: Mapping[str, object]
) -> tuple[int, int, str, int]:
    requested = locator["lines"]
    start = int(requested["start"])
    count = int(locator.get("window_line_count") or (
        int(requested["end"]) - start + 1
    ))
    anchor = ""
    offset = 0
    exact = None
    for source_text in (
        str(locator.get("expected_text") or ""),
        str(locator.get("observed_text") or ""),
    ):
        if source_text:
            exact = _exact_site_window(raw, source_text)
            if exact is not None:
                break
    if exact is not None:
        start, count, anchor = exact
    else:
        anchored = _anchored_site_window(lines, locator)
        if anchored is None:
            raise ActionLedgerError("open edit site could not be anchored to current bytes")
        start, count, anchor, offset = anchored
    return _bounded_site_window(lines, start, count, anchor, offset)


def _open_site_change_status(
    locator: Mapping[str, object], excerpt: bytes, current_revision: str
) -> tuple[bool | None, str, str]:
    read_revision = str(locator.get("last_read_revision") or "")
    observed_text = str(locator.get("observed_text") or "")
    last_action = str(locator["last_action"])
    if (
        last_action == "edit"
        and read_revision
        and current_revision != read_revision
    ):
        return (
            True,
            "CHANGED_SINCE_LAST_READ",
            "The site changed after its last read and a successful edit targeted it; "
            "these current bytes were freshly read from disk.",
        )
    if not read_revision:
        return (
            None,
            "NO_PRIOR_READ_REVISION",
            "No earlier read revision is available; these current bytes were freshly read from disk.",
        )
    changed = (
        excerpt != observed_text.encode("utf-8")
        if observed_text else current_revision != read_revision
    )
    if changed:
        return (
            True,
            "CHANGED_SINCE_LAST_READ",
            "The site changed after its last read; these current bytes were freshly read from disk.",
        )
    return (
        False,
        "UNCHANGED_SINCE_LAST_READ",
        "The site still matches its last read; these current bytes were freshly read from disk.",
    )


def _read_current_open_site(
    workspace_root: str | Path, locator_value: object
) -> tuple[dict[str, object], dict[str, object]]:
    locator = _normalise_open_site_locator(locator_value)
    if locator is None:
        raise ActionLedgerError("open edit site locator is invalid")
    raw = _read_open_site_file_bytes(workspace_root, str(locator["path"]))
    lines = raw.splitlines(keepends=True)
    start, end, anchor_line, anchor_offset = _open_site_window(raw, lines, locator)
    excerpt = b"".join(lines[start - 1:end])
    if len(excerpt) > MAX_OPEN_EDIT_SITE_BYTES:
        raise ActionLedgerError("open edit site exceeds the bounded excerpt size")
    content = excerpt.decode("utf-8")
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    last_read_revision = str(locator.get("last_read_revision") or "")
    changed, status, notice = _open_site_change_status(locator, excerpt, revision)
    site = {
        "schema": OPEN_EDIT_SITE_SCHEMA,
        "path": str(locator["path"]),
        "lines": {"start": start, "end": end, "total": len(lines)},
        "revision": revision,
        "content_sha256": hashlib.sha256(excerpt).hexdigest(),
        "current_bytes": content,
        "read_this_request": True,
        "last_read_revision": last_read_revision or None,
        "changed_since_last_read": changed,
        "status": status,
        "notice": notice,
        "action_guidance": {
            "preferred_tool": "str_replace",
            "first_matching_tool": "str_replace",
            "max_calls_to_edit": 2,
            "unchanged_rereads": 0,
            "directive": (
                "Use str_replace first when its match is exact; complete the edit within "
                "two calls and do not reread an unchanged site."
            ),
        },
    }
    next_locator = {
        "path": str(locator["path"]),
        "lines": {"start": start, "end": end},
        "last_action": "read",
        "last_read_revision": revision,
        "anchor_line": anchor_line,
        "anchor_offset": anchor_offset,
        "window_line_count": max(1, end - start + 1),
        "observed_text": content,
        "expected_text": "",
    }
    return site, next_locator


def _is_open_site_action(call: Mapping[str, object]) -> bool:
    return _tool_name(call) in {
        "read", "str_replace", "edit", "edit_file", "file_create"
    }


def _read_open_site_locator(
    call: Mapping[str, object], arguments: Mapping[str, object], result_text: str
) -> dict[str, object] | None:
    parsed = _read_site_window(arguments, result_text)
    if parsed is None:
        return None
    fact, content = parsed
    source_lines = content.splitlines(keepends=True)
    bounded_lines = source_lines[:MAX_OPEN_EDIT_SITE_LINES]
    bounded_content = "".join(bounded_lines)
    if (
        not bounded_content
        or len(bounded_content.encode("utf-8")) > MAX_OPEN_EDIT_SITE_BYTES
    ):
        return None
    lines = fact["lines"]
    start = int(lines["start"])
    end = min(int(lines["end"]), start + len(bounded_lines) - 1)
    anchors = [
        (line.rstrip("\r\n"), offset)
        for offset, line in enumerate(bounded_lines)
        if line.strip()
    ]
    anchor_line, anchor_offset = max(anchors, key=lambda item: len(item[0])) if anchors else ("", 0)
    return {
        "path": str(fact["path"]),
        "lines": {"start": start, "end": end},
        "last_action": "read",
        "last_read_revision": str(fact["revision"]),
        "anchor_line": anchor_line,
        "anchor_offset": anchor_offset,
        "observed_text": bounded_content,
        "expected_text": "",
        "window_line_count": max(1, len(bounded_lines)),
    }


def _fallback_edit_open_site_locator(
    name: str,
    arguments: Mapping[str, object],
    result_text: str,
    current_reads: Mapping[str, tuple[dict[str, object], str]],
    candidate_revision: str,
) -> dict[str, object] | None:
    if name not in {"str_replace", "edit", "edit_file"}:
        return None
    if not result_text.startswith(("str_replace applied.", "edit applied.")):
        return None
    if re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_revision) is None:
        return None
    path = _safe_path(arguments.get("path", arguments.get("target_path")))
    old_text, new_text = arguments.get("old_str"), arguments.get("new_str")
    if (
        not path
        or not isinstance(old_text, str)
        or not isinstance(new_text, str)
        or not old_text
        or not new_text
        or not _within_utf8_limit(old_text, MAX_OPEN_EDIT_SITE_BYTES)
        or not _within_utf8_limit(new_text, MAX_OPEN_EDIT_SITE_BYTES)
    ):
        return None
    if len(old_text.encode("utf-8")) + len(new_text.encode("utf-8")) > MAX_OPEN_EDIT_SITE_BYTES:
        return None
    prior_read = current_reads.get(path)
    read_fact, read_text = prior_read if prior_read is not None else ({}, "")
    read_lines = read_fact.get("lines")
    start = int(read_lines.get("start", 1)) if isinstance(read_lines, Mapping) else 1
    count = max(1, len(new_text.splitlines()))
    return {
        "path": path,
        "lines": {"start": start, "end": start + count - 1},
        "last_action": "edit",
        "last_read_revision": str(read_fact.get("revision") or ""),
        "observed_text": read_text,
        "expected_text": new_text,
        "window_line_count": count,
    }


def _edit_open_site_locator(
    call: Mapping[str, object],
    arguments: Mapping[str, object],
    result_text: str,
    current_reads: Mapping[str, tuple[dict[str, object], str]],
    candidate_revision: str,
) -> dict[str, object] | None:
    name = _tool_name(call)
    call_id = str(call.get("id") or "")
    fact = _edit_fact(
        name, arguments, result_text, call_id, current_reads, candidate_revision
    )
    if fact is None:
        return _fallback_edit_open_site_locator(
            name, arguments, result_text, current_reads, candidate_revision
        )
    lines = fact["lines"]
    prior_read = current_reads.get(str(fact["path"]))
    read_fact = prior_read[0] if prior_read is not None else {}
    read_text = prior_read[1] if prior_read is not None else ""
    expected_text = (
        arguments.get("content") if name == "file_create"
        else arguments.get("new_str")
    )
    return {
        "path": str(fact["path"]),
        "lines": {"start": int(lines["start"]), "end": int(lines["end"])},
        "last_action": "edit",
        "last_read_revision": str(read_fact.get("revision") or ""),
        "observed_text": read_text,
        "expected_text": expected_text if isinstance(expected_text, str) else "",
        "window_line_count": max(1, int(lines["end"]) - int(lines["start"]) + 1),
    }


def _open_site_candidate_for_group(
    group: Mapping[str, object],
    current_reads: Mapping[str, tuple[dict[str, object], str]],
    candidate_revision: str,
) -> tuple[bool, dict[str, object] | None]:
    messages = group.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        return False, None
    assistant = messages[0]
    if not isinstance(assistant, Mapping):
        return False, None
    calls = assistant.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        return False, None
    site_calls = [
        call for call in calls
        if isinstance(call, Mapping) and _is_open_site_action(call)
    ]
    if not site_calls:
        return False, None
    if len(site_calls) != 1:
        return True, None
    call = site_calls[0]
    call_id = str(call.get("id") or "")
    arguments = _arguments(call)
    results = {
        str(item.get("tool_call_id") or ""): item
        for item in messages[1:]
        if isinstance(item, Mapping) and item.get("role") == "tool"
    }
    result = results.get(call_id)
    if arguments is None or not isinstance(result, Mapping):
        return True, None
    result_text = str(result.get("content") or "")
    locator = (
        _read_open_site_locator(call, arguments, result_text)
        if _tool_name(call) == "read"
        else _edit_open_site_locator(
            call, arguments, result_text, current_reads, candidate_revision
        )
    )
    return True, locator


def build_action_context(
    messages: Sequence[Mapping[str, object]],
    *,
    keep_k: int = 0,
    prior_ledger: Mapping[str, object] | None = None,
    workspace_root: str | Path | None = None,
    prior_open_edit_site: Mapping[str, object] | None = None,
    degrade_missing_exact_site: bool = False,
    journal: object = None,
) -> dict[str, object]:
    """Build action receipts, typed-source turns, and optional diagnostic-only history."""
    if type(keep_k) is not int or not 0 <= keep_k <= MAX_KEEP_K:
        raise ValueError(f"keep_k must be an integer from 0 to {MAX_KEEP_K}")
    groups, _consumed = _turn_groups(messages, journal=journal)
    prior = _prior_ledger(prior_ledger)
    prior_rows = list(prior.get("rows", []))
    prior_ids = _row_call_ids(prior_rows)
    candidate_revision = _candidate_revision(messages)
    current_reads: dict[str, tuple[dict[str, object], str]] = {}
    read_contents: dict[str, str] = {}
    edited_paths: set[str] = set()
    parsed_groups: list[tuple[dict[str, object], list[dict[str, object]] | None]] = []
    future_reads: dict[tuple[str, int], str] = {}
    latest_site_locator: dict[str, object] | None = None
    site_action_seen = False
    opaque_site_invalidated = False

    for group in groups:
        if group.get("standalone"):
            parsed_groups.append((group, None))
            continue
        facts: list[dict[str, object]] = []
        group_reads: dict[str, tuple[dict[str, object], str]] = {}
        reads_before_group = dict(current_reads)
        # Shell can replace every byte of a previously read site. Its output is
        # an observation, never a source anchor, even when the command succeeds.
        shell_call = any(
            isinstance(call, Mapping) and _tool_name(call) == "shell"
            for call in group["messages"][0].get("tool_calls", ())
        )
        has_site_action, group_site_locator = _open_site_candidate_for_group(
            group, reads_before_group, candidate_revision
        )
        if has_site_action:
            site_action_seen = True
            latest_site_locator = group_site_locator
            opaque_site_invalidated = False
        if shell_call:
            latest_site_locator = None
            site_action_seen = False
            opaque_site_invalidated = True
            current_reads.clear()
        supported = True
        assistant = group["messages"][0]
        # Narration beside a tool call is not itself evidence. Closed calls are
        # projected only from their matched arguments and tool results below;
        # assistant prose is never copied into the product ledger.
        results = {
            str(item.get("tool_call_id")): item
            for item in group["messages"][1:]
            if isinstance(item, Mapping)
        }
        for call in assistant.get("tool_calls", ()):
            if not isinstance(call, Mapping):
                supported = False
                continue
            call_id = str(call.get("id") or "")
            result = results.get(call_id)
            arguments = _arguments(call)
            result_text = str(result.get("content") or "") if isinstance(result, Mapping) else ""
            if arguments is None or result is None:
                supported = False
                continue
            name = _tool_name(call)
            fact: dict[str, object] | None = None
            if name == "read":
                parsed_read = _read_fact(arguments, result_text, call_id)
                if parsed_read is not None:
                    fact, content = parsed_read
                    path = str(fact["path"])
                    group_reads[path] = (fact, content)
                    read_contents[call_id] = content
                    future_reads[(path, int(group["index"]))] = str(fact["revision"])
            elif name in {"str_replace", "edit", "edit_file", "file_create"}:
                path = _safe_path(arguments.get("path", arguments.get("target_path")))
                if path:
                    edited_paths.add(path)
                fact = _edit_action_fact(
                    call, name, arguments, result_text, call_id, reads_before_group,
                    candidate_revision, turn_index=int(group["index"]), journal=journal,
                )
            elif name in {"run_tests", "pytest", "test"}:
                fact = _test_fact(arguments, result_text, call_id, candidate_revision)
            if fact is None:
                supported = False
            else:
                facts.append(fact)
        current_reads.update(group_reads)
        parsed_groups.append((group, facts if supported else None))

    # Bind edits to the first later exact read of that file when available.
    for group, facts in parsed_groups:
        if facts is None:
            continue
        for fact in facts:
            if fact.get("kind") != "edit" or fact.get("operation") == "create":
                continue
            path = str(fact.get("path") or "")
            later = next(
                (
                    revision
                    for (read_path, read_index), revision in sorted(
                        future_reads.items(), key=lambda item: item[0][1]
                    )
                    if read_path == path and read_index > int(group["index"])
                ),
                "",
            )
            if later:
                fact["post_edit_revision"] = later
                fact["post_revision_source"] = "subsequent_read"

    # Exact source ranges are retained for every read that became an edit anchor.
    for _group, facts in parsed_groups:
        if facts is None:
            continue
        for fact in facts:
            if fact.get("kind") == "read" and fact.get("path") in edited_paths:
                call_ids = fact.get("call_ids")
                call_id = call_ids[0] if isinstance(call_ids, list) and call_ids else ""
                if isinstance(call_id, str) and call_id in read_contents:
                    fact["pinned_range"] = read_contents[call_id]

    latest_test_group: tuple[str, ...] | None = None
    superseded_test_groups: set[tuple[str, ...]] = set()
    for group, facts in parsed_groups:
        if facts is not None and any(fact.get("kind") == "test" for fact in facts):
            latest_test_group = tuple(group["key"])
    if latest_test_group is not None:
        for index, (group, facts) in enumerate(parsed_groups):
            has_test = facts is not None and any(
                fact.get("kind") == "test" for fact in facts
            )
            if has_test and tuple(group["key"]) != latest_test_group:
                superseded_test_groups.add(tuple(group["key"]))
                parsed_groups[index] = (group, None)
        prior_rows = [
            row for row in prior_rows
            if not isinstance(row, Mapping) or row.get("kind") != "test"
        ]
        prior_ids = _row_call_ids(prior_rows)

    uncovered_groups: list[dict[str, object]] = []
    for group, facts in parsed_groups:
        ids = tuple(group["call_ids"])
        if ids and set(ids).issubset(prior_ids):
            continue
        if facts is None and tuple(group["key"]) not in superseded_test_groups:
            uncovered_groups.append(group)

    # Bound only newly added evidence; prior evidence is immutable and never evicted.
    rows = [copy.deepcopy(dict(row)) for row in prior_rows]
    overflowed: set[tuple[str, ...]] = set()
    additions: list[tuple[tuple[str, ...], list[dict[str, object]]]] = []
    for group, facts in parsed_groups:
        ids = tuple(group["call_ids"])
        if facts is None or not ids:
            continue
        new_facts = [
            fact for fact in facts
            if not set(fact.get("call_ids", ())).issubset(prior_ids)
        ]
        if new_facts:
            additions.append((ids, copy.deepcopy(new_facts)))
    for ids, facts in additions:
        rows.extend(facts)
    while additions and len(_json_bytes({"schema": SCHEMA, "rows": rows})) > MAX_LEDGER_BYTES:
        ids, facts = additions.pop(0)
        overflowed.add(ids)
        remove_ids = set(ids)
        rows = [
            row for row in rows
            if not (
                isinstance(row, Mapping)
                and remove_ids.intersection(
                    item for item in row.get("call_ids", ())
                    if isinstance(item, str)
                )
            )
        ]
    if overflowed:
        for group, facts in parsed_groups:
            if tuple(group["call_ids"]) in overflowed and group not in uncovered_groups:
                uncovered_groups.append(group)

    ledger: dict[str, object] = {"schema": SCHEMA, "rows": rows}
    covered_ids = _row_call_ids(rows)
    # A call group is covered only when every call id appears in typed state.
    covered_groups = {
        tuple(group["call_ids"])
        for group, facts in parsed_groups
        if group["call_ids"] and set(group["call_ids"]).issubset(covered_ids)
    }
    for group in groups:
        ids = tuple(group["call_ids"])
        if ids and set(ids).issubset(prior_ids):
            covered_groups.add(ids)

    unclassified_turns = tuple(
        tuple(copy.deepcopy(dict(message)) for message in group["messages"])
        for group in sorted(uncovered_groups, key=lambda row: int(row["index"]))
    )
    diagnostic_groups = (
        [group for group in groups if bool(group.get("real"))][-keep_k:]
        if keep_k else []
    )
    diagnostic_turns = tuple(
        tuple(copy.deepcopy(dict(message)) for message in group["messages"])
        for group in diagnostic_groups
    )
    diagnostic_ids = tuple(
        call_id
        for group in diagnostic_groups
        for call_id in group["call_ids"]
        if isinstance(call_id, str) and call_id
    )
    superseded_test_call_ids = tuple(
        call_id
        for group in groups if tuple(group["key"]) in superseded_test_groups
        for call_id in group["call_ids"] if isinstance(call_id, str) and call_id
    )
    selected_site_locator = (
        None if opaque_site_invalidated else latest_site_locator if site_action_seen
        else _normalise_open_site_locator(prior_open_edit_site)
    )
    open_edit_site: dict[str, object] | None = None
    refreshed_site_locator: dict[str, object] | None = None
    site_context: dict[str, str] | None = None
    if opaque_site_invalidated:
        site_context = {
            "status": "UNVERIFIABLE",
            "reason": "shell_execution_requires_current_source_read",
        }
    if workspace_root is not None and selected_site_locator is not None:
        open_edit_site, refreshed_site_locator = _read_current_open_site(
            workspace_root, selected_site_locator
        )
    elif workspace_root is not None and site_action_seen:
        if not degrade_missing_exact_site:
            raise ActionLedgerError(
                "latest file action has no exact current site",
                code=MISSING_EXACT_CURRENT_SITE,
            )
        site_context = {
            "status": "UNVERIFIABLE",
            "reason": MISSING_EXACT_CURRENT_SITE,
        }

    result: dict[str, object] = {
        "ledger": ledger if rows else None,
        "covered_call_ids": tuple(sorted(covered_ids)),
        "covered_groups": tuple(sorted(covered_groups)),
        # Kept empty for compatibility. Product requests never carry prior turns.
        "verbatim_turns": (),
        "unclassified_turns": unclassified_turns,
        "diagnostic_turns": diagnostic_turns,
        "diagnostic_label": "DIAGNOSTIC_ONLY" if diagnostic_turns else None,
        "protected_call_ids": (),
        "diagnostic_protected_call_ids": diagnostic_ids,
        "uncovered_call_ids": tuple(
            call_id for group in uncovered_groups for call_id in group["call_ids"]
        ),
        "superseded_test_call_ids": superseded_test_call_ids,
        "keep_k_call_ids": diagnostic_ids,
        "open_edit_site": open_edit_site,
        "open_edit_site_locator": refreshed_site_locator,
    }
    if site_context is not None:
        result["site_context"] = site_context
    return result


__all__ = [
    "ActionLedgerError",
    "ACTION_CONTEXT_MODE_ENV",
    "FRAMEFIT5_ACTION_CONTEXT_ENV",
    "MISSING_EXACT_CURRENT_SITE",
    "KEEP_K_ENV",
    "MAX_KEEP_K",
    "MAX_OPEN_EDIT_SITE_LINES",
    "MAX_ACTION_ARGUMENT_BYTES",
    "MAX_LEDGER_BYTES",
    "OPEN_EDIT_SITE_SCHEMA",
    "SCHEMA",
    "build_action_context",
    "build_framefit_action_context",
    "framefit_keep_k",
    "framefit5_action_context_mode",
]
