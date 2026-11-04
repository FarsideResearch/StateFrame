"""Compare raw no-server request files against the 0.2.3 source tree that test_no_mcp_release_comparison pins.

Run with --baseline /path/to/33c6c51 (0.2.3) --candidate /path/to/this/tree.
Both runs use the same temporary repository and run-directory path in sequence.
Wire IDs are masked by JSON path so unrelated hex in prompt content stays part of parity checks.
The candidate pins the settings whose defaults changed after 0.2.3 (pytest runner, all tests
protected, stable tool head off), so the request files are expected to be byte-identical;
pass --current-defaults to see how today's defaults differ instead.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stateframe.request_records import list_request_records, read_request_record  # noqa: E402


def source_env(tree: Path, stable_tool_head: str | None, *,
               source_on_pythonpath: bool = True) -> dict[str, str]:
    environment = dict(os.environ)
    if source_on_pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join((str(tree / "src"),
            str(tree / "src/stateframe/engine")))
    else:
        environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    return environment


# Generated IDs and the tool-result hash derived from them are masked by JSON path,
# so unrelated hex remains part of the comparison. StateFrame and read-contract
# payloads are embedded JSON; their markers are path steps, not a malformed-body fallback.
WIRE_ID_PATHS = (
    # The CLI preflight capture records the active request ID outside messages.
    (("request_id",), "request_id", None, (20,)),
    # Request-diet journal payloads name check receipts in their evidence list.
    (("required_evidence_ids", "*"), "check_receipt", "receipt:", (24,)),
    # Provider tool-call IDs are referenced by later tool messages through tool_call_id.
    (("messages", "*", "tool_calls", "*", "id"), "tool_call_id", "native_", (20,)),
    # Tool replies carry the matching provider tool-call ID.
    (("messages", "*", "tool_call_id"), "tool_call_id", "native_", (20,)),
    # The action ledger carries the receipt and request ID for each applied edit.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "applied_edits", "*", "edit_receipt"), "edit_receipt", None, (20,)),
    # The top-level StateFrame projection repeats applied-edit receipt IDs.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "applied_edits", "*", "edit_receipt"), "edit_receipt", None, (20,)),
    # The action ledger carries the request ID that minted each applied-edit receipt.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "applied_edits", "*", "request_id"), "request_id", None, (20,)),
    # The top-level StateFrame projection repeats those applied-edit request IDs.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "applied_edits", "*", "request_id"), "request_id", None, (20,)),
    # Native call IDs in edit action rows derive from the request ID and action index.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "edits", "*", "call_ids", "*"), "native_call_id(request_id)", "native_", (20,)),
    # The history digest preserves native IDs for calls that have left the recent action rows.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "history_digest", "call_ids", "*"), "native_call_id(request_id)", "native_", (20,)),
    # Coverage summaries carry the public run ID derived from the task identity.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "history_digest", "coverage_run_id"), "coverage_run_id", None, (20,)),
    # Read action rows retain the provider call IDs that produced each read receipt.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "reads", "*", "call_ids", "*"), "native_call_id(request_id)", "native_", (20,)),
    # Unclassified actions record their host-derived native call ID in call_id.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "unclassified_actions", "*", "call_id"),
     "native_call_id(request_id)", "native_", (20,)),
    # The oracle result hash covers the sent tool-result text, including generated
    # CHECK_FAILURE_SUMMARY execution_id and full_output_handle receipt IDs.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "unclassified_actions", "*", "result_sha256"),
     "result_sha256(tool_result)", None, (64,)),
    # Read receipts are carried in the StateFrame's read-fact receipt rows.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "read_facts", "*", "receipts", "*", "receipt_id"), "receipt_id", None, (20,)),
    # Read-contract receipts are serialized as a separate tagged JSON payload in message content.
    (("messages", "*", "content", "[WIRE_READ_CONTRACT]", "receipt_id"),
     "receipt_id", None, (20,)),
    # The action ledger's journal verdict references the host-minted check receipt.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "action_ledger", "journal_verdict", "check_receipt_id"), "check_receipt", "check-receipt:", (24,)),
    # The latest verdict projection repeats the host-minted check receipt ID.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "last_verdict", "check_receipt_id"), "check_receipt", "check-receipt:", (24,)),
    # Check execution IDs appear in the evidence pin as labeled text.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "owner_protocol", "evidence_pin"),
     "check_execution", "execution_id=check-execution:", (20,)),
    # The evidence pin's bounded excerpt may truncate this duplicate handle after its label.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "owner_protocol", "evidence_pin"),
     "check_receipt", "full_output_handle=check-receipt:", (24,)),
    # The visible check-output receipt repeats that same host receipt handle.
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "owner_protocol", "evidence_pin"),
     "check_receipt", "check_output_receipt=check-receipt:", (24,)),
    (("messages", "*", "content", "[StateFrame]", "state", "framefit5",
      "owner_protocol", "evidence_pin"),
     "check_receipt", "receipt=check-receipt:", (24,)),
    # Check-finding IDs use the check:<run-id>:<ordinal> form in the history projection.
    (("messages", "*", "content", "[StateFrame]", "state", "history_findings", "latest", "id"),
     "check_finding_id", "check:", (20,)),
    # Omitted-fact handles embed the request ID that fixed the fact's cutoff.
    (("messages", "*", "content"), "wire_truth_handle(request_id)", "wire-truth:", (20,)),
    # The check-execution reference lists host execution IDs in the volatile-truth tail.
    (("messages", "*", "content"), "check_execution_ref", "check_execution_ref: ", (20,)),
    # The generated deferred-read notice lists the same read-fact receipt IDs.
    (("messages", "*", "content"), "receipt_id", "journal receipts ", (20,)),
    # Traceback paths can include the generated continuation-workspace ID; fixtures use <workspace>.
    (("messages", "*", "content"), "continuation_workspace_dir",
     "continuation-workspaces/", (20, 32)),
    # The oracle's sent tool result repeats host-minted check execution and receipt IDs.
    (("messages", "*", "content"), "check_execution", "CHECK_FAILURE_SUMMARY execution_id=check-execution:", (20,)),
    (("messages", "*", "content"), "check_receipt", "full_output_handle=check-receipt:", (24,)),
)

# Used only to verify sent result hashes; masking itself uses WIRE_ID_PATHS exclusively.
_RESULT_SHA = re.compile(rb'(result_sha256\\*":\s*\\*")([0-9a-f]{64})')


def _replace_path(value, path: tuple[str, ...], *, text_marker: str | None = None,
                  text_lengths: tuple[int, ...] = ()) -> tuple[object, list[str]]:
    """Replace the values selected by one explicit JSON path."""
    if not path:
        if not isinstance(value, str) or not value:
            return value, []
        if text_marker is not None:
            return _replace_marked_hex(value, text_marker, text_lengths)
        if text_lengths and (len(value) not in text_lengths
                             or any(char not in "0123456789abcdef" for char in value)):
            return value, []
        return "<wire-id>", [value]

    segment, rest = path[0], path[1:]
    if segment in {"[StateFrame]", "[WIRE_READ_CONTRACT]"}:
        if not isinstance(value, str):
            return value, []
        marker = segment + ("\n" if segment == "[StateFrame]" else " ")
        return _replace_embedded_json(value, marker, rest,
                                      text_marker=text_marker, text_lengths=text_lengths)
    if segment == "*":
        if isinstance(value, dict):
            rebuilt, matches = {}, []
            for key, child in value.items():
                rebuilt_child, child_matches = _replace_path(
                    child, rest, text_marker=text_marker, text_lengths=text_lengths)
                rebuilt[key] = rebuilt_child
                matches.extend(child_matches)
            return rebuilt, matches
        if isinstance(value, list):
            rebuilt, matches = [], []
            for child in value:
                rebuilt_child, child_matches = _replace_path(
                    child, rest, text_marker=text_marker, text_lengths=text_lengths)
                rebuilt.append(rebuilt_child)
                matches.extend(child_matches)
            return rebuilt, matches
        return value, []
    if not isinstance(value, dict) or segment not in value:
        return value, []
    rebuilt = dict(value)
    rebuilt[segment], matches = _replace_path(value[segment], rest, text_marker=text_marker,
                                               text_lengths=text_lengths)
    return rebuilt, matches


def _replace_embedded_json(content: str, marker: str, path: tuple[str, ...], *,
                           text_marker: str | None,
                           text_lengths: tuple[int, ...]) -> tuple[str, list[str]]:
    """Apply a JSON path to each JSON payload following a known message-content marker."""
    decoder = json.JSONDecoder()
    cursor = 0
    matches: list[str] = []
    pieces: list[str] = []
    while True:
        marker_start = content.find(marker, cursor)
        if marker_start < 0:
            pieces.append(content[cursor:])
            break
        payload_start = marker_start + len(marker)
        json_start = payload_start
        while json_start < len(content) and content[json_start].isspace():
            json_start += 1
        try:
            payload, payload_end = decoder.raw_decode(content, json_start)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON after embedded {marker.strip()} marker") from error
        updated, payload_matches = _replace_path(
            payload, path, text_marker=text_marker, text_lengths=text_lengths)
        if payload_matches:
            encoded = json.dumps(updated, separators=(",", ":"), ensure_ascii=True)
            pieces.extend((content[cursor:json_start], encoded))
            cursor = payload_end
            matches.extend(payload_matches)
        else:
            pieces.append(content[cursor:payload_end])
            cursor = payload_end
    return "".join(pieces), matches


def _replace_marked_hex(value: str, marker: str,
                        lengths: tuple[int, ...]) -> tuple[str, list[str]]:
    """Replace hex IDs immediately following one known text label, never arbitrary hex text."""
    if marker == "check_execution_ref: ":
        return _replace_check_execution_refs(value, marker, lengths)
    if marker == "journal receipts ":
        return _replace_journal_receipts(value)
    if marker == "native_" and re.fullmatch(
            r"native_[0-9a-f]{20}_[0-9]+(?::auto_verify)?", value) is None:
        return value, []
    output = value
    cursor = 0
    matches: list[str] = []
    while True:
        marker_start = output.find(marker, cursor)
        if marker_start < 0:
            break
        start = marker_start + len(marker)
        matched_length = next((length for length in lengths
                               if start + length <= len(output)
                               and all(char in "0123456789abcdef"
                                       for char in output[start:start + length])
                               and (start + length == len(output)
                                    or output[start + length] not in "0123456789abcdef")), None)
        if matched_length is None:
            if marker == "full_output_handle=check-receipt:":
                matches.append("")  # The fixture truncates this labeled value after the prefix.
            elif (marker == "continuation-workspaces/"
                  and output.startswith("<workspace>/", start)):
                matches.append("")  # The fixture replaces the generated ID with this stable placeholder.
                cursor = start + len("<workspace>/")
                continue
            cursor = start
            continue
        end = _legacy_sequence_end(output, start + matched_length)
        matches.append(output[start:end])
        output = output[:start] + "<wire-id>" + output[end:]
        cursor = start + len("<wire-id>")
    return output, matches


def _legacy_sequence_end(value: str, token_end: int) -> int:
    """Include the old journal-sequence suffix when it is part of a wire-ID rendering."""
    if token_end >= len(value) or value[token_end] != ":":
        return token_end
    end = token_end + 1
    while end < len(value) and value[end].isdigit():
        end += 1
    return end if end > token_end + 1 else token_end


def _replace_check_execution_refs(value: str, marker: str,
                                   lengths: tuple[int, ...]) -> tuple[str, list[str]]:
    """Mask check execution IDs and the old journal-sequence suffix, if present."""
    spans: list[tuple[int, int]] = []
    matches: list[str] = []
    cursor = 0
    while True:
        marker_start = value.find(marker, cursor)
        if marker_start < 0:
            break
        token_start = marker_start + len(marker)
        list_cursor = token_start
        while True:
            length = next((candidate for candidate in lengths
                           if list_cursor + candidate <= len(value)
                           and all(char in "0123456789abcdef"
                                   for char in value[list_cursor:list_cursor + candidate])
                           and (list_cursor + candidate == len(value)
                                or value[list_cursor + candidate] in ":, \t\r\n")), None)
            if length is None:
                break
            token_end = list_cursor + length
            suffix_end = token_end
            if suffix_end < len(value) and value[suffix_end] == ":":
                suffix_end += 1
                ordinal_start = suffix_end
                while suffix_end < len(value) and value[suffix_end].isdigit():
                    suffix_end += 1
                if suffix_end == ordinal_start:
                    break
            spans.append((list_cursor, suffix_end))
            matches.append(value[list_cursor:suffix_end])
            if suffix_end >= len(value) or value[suffix_end] != ",":
                break
            list_cursor = suffix_end + 1
        cursor = max(token_start, list_cursor)
    output = value
    for start, end in reversed(spans):
        output = output[:start] + "<wire-id>" + output[end:]
    return output, matches


def _replace_journal_receipts(value: str) -> tuple[str, list[str]]:
    """Mask only the generated comma-separated receipt list after its label."""
    matches: list[str] = []
    pattern = re.compile(r"journal receipts ([0-9a-f]{20}(?:, [0-9a-f]{20})*)(?![0-9a-f])")

    def replace(match: re.Match[str]) -> str:
        ids = match.group(1).split(", ")
        matches.extend(ids)
        return "journal receipts " + ", ".join("<wire-id>" for _ in ids)

    return pattern.sub(replace, value), matches


def _mask_one_json_body(body: bytes) -> tuple[bytes, list[tuple[str, list[str]]]]:
    request = json.loads(body)
    if not isinstance(request, (dict, list)) or (isinstance(request, list)
                                                  and not all(isinstance(row, dict) for row in request)):
        raise ValueError("request body must be a JSON object or a list of records")
    records = request if isinstance(request, list) else [request]
    values_by_path = []
    for path, kind, text_marker, text_lengths in WIRE_ID_PATHS:
        matches = []
        for index, record in enumerate(records):
            records[index], found = _replace_path(record, path, text_marker=text_marker,
                                                   text_lengths=text_lengths)
            matches.extend(found)
        values_by_path.append((kind, matches))
    if not isinstance(request, list):
        request = records[0]
    encoded = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return encoded, values_by_path


def _mask_wire_ids_with_values(body: bytes) -> tuple[bytes, list[tuple[str, list[str]]]]:
    """Return one request's path-masked bytes and the original values selected at each path."""
    return _mask_one_json_body(body)


def mask_wire_ids(body: bytes) -> bytes:
    """Mask one wire body or consecutive pretty-printed CLI preflight records.

    capture() joins requests/NNNN.json records with a newline. Each record is
    itself multiline JSON, so splitting the stream into lines loses its framing.
    """
    source = body.decode("utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    masked = []
    while offset < len(source):
        while offset < len(source) and source[offset].isspace():
            offset += 1
        if offset == len(source):
            break
        _, end = decoder.raw_decode(source, offset)
        masked.append(_mask_one_json_body(source[offset:end].encode("utf-8"))[0])
        offset = end
    if not masked:
        raise ValueError("request capture contains no JSON objects")
    return b"\n".join(masked)


def result_hashes(run_dir: Path) -> tuple[list[str], set[str]]:
    """The result_sha256 values a run sent, in order, and the hashes of its sent message texts."""
    sent, texts = [], set()
    for path in sorted(run_dir.rglob("*.request.json")):
        raw = path.read_bytes()
        sent += [match.group(2).decode() for match in _RESULT_SHA.finditer(raw)]
        for message in json.loads(raw).get("messages", []):
            if isinstance(message.get("content"), str):
                texts.add(hashlib.sha256(message["content"].encode("utf-8")).hexdigest())
    return sent, texts


def unbacked_result_hash_changes(before_run: Path, after_run: Path) -> list[str]:
    """Changed result_sha256 values whose hashed text was not sent, so masking them could hide a difference."""
    before_hashes, before_texts = result_hashes(before_run)
    after_hashes, after_texts = result_hashes(after_run)
    if len(before_hashes) != len(after_hashes):
        return [f"{len(before_hashes)} result hashes before, {len(after_hashes)} after"]
    return [f"{before} -> {after}" for before, after in zip(before_hashes, after_hashes)
            if before != after and (before not in before_texts or after not in after_texts)]


# Every field the receipt-determinism change (11-03) is allowed to move. Release parity tests
# fail on a changed ID in any other field, and on any change outside an ID.
MOVED_ID_FIELDS = frozenset({
    "tool_call_id", "wire_truth_handle(request_id)", "request_id", "native_call_id(request_id)",
    "edit_receipt", "check_receipt", "check_execution", "check_execution_ref",
    "check_finding_id", "receipt_id", "continuation_workspace_dir",
})
DERIVED_RESULT_FIELDS = frozenset({"result_sha256(tool_result)"})


def wire_id_drift(before: list[bytes], after: list[bytes]) -> tuple[list[str], dict[str, int]]:
    """Problems outside declared generated fields, plus which fields changed.

    Bodies must be byte-identical after mask_wire_ids. Callers that mask result_sha256
    also verify changed digests hash sent tool-result messages in each run.
    """
    problems: list[str] = []
    kinds: dict[str, int] = {}
    if len(before) != len(after):
        return [f"{len(before)} requests before, {len(after)} after"], kinds
    for index, (old, new) in enumerate(zip(before, after)):
        masked_old, old_values = _mask_one_json_body(old)
        masked_new, new_values = _mask_one_json_body(new)
        if masked_old != masked_new:
            offset = next((i for i, (x, y) in enumerate(zip(masked_old, masked_new)) if x != y),
                          min(len(masked_old), len(masked_new)))
            problems.append(f"request {index}: non-ID bytes differ at masked offset {offset}: "
                            f"{masked_old[max(0, offset - 60):offset + 40]!r} -> "
                            f"{masked_new[max(0, offset - 60):offset + 40]!r}")
            continue
        for (kind, old_ids), (_new_kind, new_ids) in zip(old_values, new_values):
            for old_id, new_id in zip(old_ids, new_ids):
                if old_id != new_id:
                    kinds[kind] = kinds.get(kind, 0) + 1
    return problems, kinds


def changed_id_chars(before: list[bytes], after: list[bytes]) -> list[int]:
    """Per request, how many characters sit in wire-ID tokens whose value changed. A token count
    can follow the ID's characters, not only its length, so this bounds token movement."""
    changed = []
    for old, new in zip(before, after):
        _, old_values = _mask_one_json_body(old)
        _, new_values = _mask_one_json_body(new)
        chars = sum(len(new_id) for (_kind, old_ids), (_new_kind, new_ids)
                    in zip(old_values, new_values)
                    for old_id, new_id in zip(old_ids, new_ids) if old_id != new_id)
        changed.append(chars)
    return changed


def id_length_deltas(before: list[bytes], after: list[bytes]) -> tuple[list[int], list[str]]:
    """Per request, how much the wire-ID tokens changed in length, and any body whose size
    change is NOT fully explained by them (older releases suffixed some IDs with ':<sequence>')."""
    deltas, problems = [], []
    for index, (old, new) in enumerate(zip(before, after)):
        _, old_values = _mask_one_json_body(old)
        _, new_values = _mask_one_json_body(new)
        delta = sum(len(new_id) - len(old_id)
                    for (_kind, old_ids), (_new_kind, new_ids) in zip(old_values, new_values)
                    for old_id, new_id in zip(old_ids, new_ids))
        deltas.append(delta)
        if len(new) - len(old) != delta:
            problems.append(f"request {index}: size changed by {len(new) - len(old)}, ID tokens by {delta}")
    return deltas, problems


_WIRE_TRUTH_HANDLE = re.compile(rb"wire-truth:([0-9a-f]{20}):")


def request_id_handle_mismatches(run_dir: Path) -> list[str]:
    """Each wire-truth handle a request sends must name that request's own recorded request_id."""
    # Request records may be loose files or sealed into requests.pack.xz; read both forms.
    from stateframe.request_records import list_request_records, read_request_record

    problems = []
    names = list_request_records(run_dir)
    if not names:
        return [f"{run_dir}: no request records, so nothing was checked"]
    for name in names:
        request_id = json.loads(read_request_record(run_dir, name)).get("request_id")
        body = (run_dir / "wire" / f"{Path(name).stem}.request.json").read_bytes()
        handles = {match.group(1).decode() for match in _WIRE_TRUTH_HANDLE.finditer(body)}
        if handles - {request_id}:
            problems.append(f"{name}: request_id {request_id} but handles {sorted(handles)}")
    return problems


def planted_body_change_is_caught(body: bytes) -> bool:
    """Check role and 24-hex message-content changes on a real request."""
    request = json.loads(body)
    for message in request.get("messages", []):
        if (isinstance(message, dict) and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)):
            message["role"] += "-planted"
            planted = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode()
            if not wire_id_drift([body], [planted])[0]:
                return False
            message["content"] += "\nparity-control=" + "c" * 24
            first = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode()
            message["content"] = message["content"][:-24] + "d" * 24
            second = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode()
            return bool(wire_id_drift([first], [second])[0])
    return False


def write_launcher(path: Path) -> None:
    path.write_text("import runpy, secrets\n"
                    "_sequence = 0\n"
                    "def fixed_hex(size=32):\n"
                    "    global _sequence\n"
                    "    _sequence += 1\n"
                    "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
                    "secrets.token_hex = fixed_hex\n"
                    "runpy.run_module('stateframe', run_name='__main__')\n")


def capture(tree: Path, python: str, launcher: Path, script: Path,
            repo: Path, run_dir: Path, *, legacy_mode: bool = False,
            stable_tool_head: str | None,
            source_on_pythonpath: bool = True,
            test_python: str | None = None) -> bytes:
    # stable_tool_head has no default (see tools/changed_defaults.py): every caller states its arm.
    environment = source_env(tree, stable_tool_head=stable_tool_head,
                             source_on_pythonpath=source_on_pythonpath)
    # Each capture gets its own empty home. The current tree reads enabled powerups and user skills from
    # STATEFRAME_HOME and old releases do not, so a home inherited from the test session (where an earlier test
    # may have left a powerup enabled) changed only one side's request bytes (11-04, 0.2.8 parity in full runs).
    environment["STATEFRAME_HOME"] = tempfile.mkdtemp(prefix=f"{run_dir.name}-home-", dir=run_dir.parent)
    endpoint = subprocess.Popen([python, "-m", "stateframe.scripted_endpoint",
        "--script", str(script), "--port", "0"], cwd=tree, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert endpoint.stdout is not None
        ready, _, _ = select.select([endpoint.stdout], [], [], 15)
        if not ready:
            raise RuntimeError("scripted endpoint did not start")
        url = endpoint.stdout.readline().strip()
        if not url.startswith("http://127.0.0.1:"):
            raise RuntimeError(f"unexpected scripted endpoint URL: {url!r}")
        command = [python, str(launcher), "run", "--repo", str(repo),
            "--task", "Fix the pricing code so the tests pass.", "--endpoint", url,
            "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", test_python or python,
            "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "32",
            "--no-run-env", "--quiet",
            *(("--runner-kind", "pytest", "--protect-tests", "all") if legacy_mode else ())]
        result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                text=True, timeout=300, check=False)
        # The comparison is about request bytes, so a run whose checks fail still counts;
        # only a run that never wrote its summary or requests is a capture failure.
        names = list_request_records(run_dir)
        if not names or not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"{tree}: run produced no requests or summary ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return b"\n".join(read_request_record(run_dir, name) for name in names)
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="untouched source tree of 0.2.3 (33c6c51)")
    parser.add_argument("--candidate", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that runs the example's tests (each tree runs under its own)")
    parser.add_argument("--current-defaults", action="store_true",
                        help="run the candidate with today's defaults instead of the pinned 0.2.3 settings")
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    script = candidate / "examples/discount_bug.script.json"
    # The engine child drops PYTHONPATH, so a shared interpreter would import whichever stateframe it has
    # installed. Give each tree its own interpreter and prove what it imports before comparing bytes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from acceptance._support import assert_tree_imports, tree_python
    with tempfile.TemporaryDirectory(prefix="sf-no-mcp-diff-") as temporary:
        root = Path(temporary)
        repo = root / "repo"
        shutil.copytree(candidate / "examples/discount_bug", repo)
        launcher = root / "fixed_seed_run.py"
        write_launcher(launcher)
        run_dir = root / "run"
        pythons = {}
        for label, tree in (("baseline", baseline), ("candidate", candidate)):
            pythons[label] = tree_python(tree / "src", root / f"{label}-python")
            assert_tree_imports(pythons[label], tree / "src", root)
        previous = capture(baseline, pythons["baseline"], launcher, script, repo, run_dir,
                           stable_tool_head=None, source_on_pythonpath=False, test_python=args.python)
        run_dir.rename(root / "baseline-run")
        current = capture(candidate, pythons["candidate"], launcher, script, repo, run_dir,
                          legacy_mode=not args.current_defaults,
                          stable_tool_head=None if args.current_defaults else "off",
                          source_on_pythonpath=False, test_python=args.python)
        for label, body in (("baseline", previous), ("candidate", current)):
            print(f"{label}: {len(body)} bytes sha256={hashlib.sha256(body).hexdigest()}")
        if previous == current:
            print("Model request files are byte-identical.")
            return 0
        before = previous.decode(errors="replace").splitlines(keepends=True)
        after = current.decode(errors="replace").splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(before, after, fromfile="baseline",
                                                   tofile="candidate", n=3))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
