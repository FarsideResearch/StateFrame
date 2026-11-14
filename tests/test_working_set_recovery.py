"""Omission carriers require an executable, interval-bound frozen read route."""
from dataclasses import replace
import hashlib
import json

import pytest

from kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
from kernel_source_delivery import deliver_source
from kernel_working_set_snapshot import SourceSlot
from kernel_working_set_recovery import RecoveryRouteUnavailable, validate_recovery
from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_stop import ReasonCode
from test_working_set_prepare import snapshot, source


def schema(*fields, required=("path",), **updates):
    props = {name: {"type": "string" if name in {"path", "cursor", "why", "expected_revision"}
                    else "integer"} for name in fields}
    result = {"type": "object", "properties": props, "required": list(required),
              "additionalProperties": False}
    result.update(updates)
    return json.dumps(result).encode()


def test_original_disjoint_range_is_rebuilt_from_captured_interval():
    captured = replace(source("a"), recovery_arguments=b'{"path":"a.py","start_line":99,"end_line":101}')
    args = validate_recovery(snapshot(captured), captured)
    assert args["path"] == "a.py" and args["start_line"] == 1 and args["end_line"] == 1


def test_restricted_max_lines_schema_gets_an_executable_variant():
    captured = source("a")
    snap = replace(snapshot(captured), recovery_schema=schema("path", "start_line", "max_lines", "max_chars"))
    args = validate_recovery(snap, captured)
    assert args["start_line"] == 1 and args["max_lines"] == 1
    assert "end_line" not in args


def test_schema_required_original_reason_is_preserved():
    captured = replace(source("a"), recovery_arguments=b'{"path":"a.py","why":"check edit"}')
    snap = replace(snapshot(captured), recovery_schema=schema("path", "start_line", "end_line", "max_chars", "why",
                                                             required=("path", "why")))
    assert validate_recovery(snap, captured)["why"] == "check edit"


def test_lossy_columns_are_preserved_in_the_recovery_route():
    captured = source("a", "tail\n")
    identity = replace(captured.identity, byte_start=None, byte_end=None,
        lossy_span=DecodedSpan(17, 22, "utf8-replacement/1"))
    captured = replace(captured, identity=identity)
    args = validate_recovery(snapshot(captured), captured)
    assert args["start_line"] == 1 and args["column"] == 17


def test_captured_cursor_is_checked_against_revision_and_coordinates():
    captured = source("a")
    cursor = "read1:" + captured.identity.revision.removeprefix("sha256:") + ":1:0"
    captured = replace(captured, recovery_arguments=json.dumps({"path":"a.py", "cursor":cursor}).encode())
    snap = replace(snapshot(captured), recovery_schema=schema("path", "cursor", "max_lines", "max_chars"))
    assert validate_recovery(snap, captured)["cursor"] == cursor
    for invalid in (cursor.replace(":1:0", ":2:0"), cursor.replace("1" * 64, "2" * 64)):
        bad = replace(captured, recovery_arguments=json.dumps({"path":"a.py", "cursor":invalid}).encode())
        with pytest.raises(RecoveryRouteUnavailable):
            validate_recovery(replace(snap, sources=(bad,)), bad)


@pytest.mark.parametrize("changes", [
    {"recovery_allowed": False},
    {"recovery_owner_allowed": False},
    {"current_revisions": (("a.py", "sha256:" + "2" * 64),)},
    {"recovery_schema": schema("path", required=("path", "unavailable"))},
])
def test_tool_owner_revision_or_schema_refusal_cannot_mint_an_omission(changes):
    captured = source("a")
    snap = replace(snapshot(captured), **changes)
    with pytest.raises(RecoveryRouteUnavailable):
        validate_recovery(snap, captured)


def test_wrong_path_arguments_never_become_recovery_authority():
    captured = replace(source("a"), recovery_arguments=b'{"path":"elsewhere.py"}')
    with pytest.raises(RecoveryRouteUnavailable):
        validate_recovery(snapshot(captured), captured)


def test_schema_path_enum_cannot_authorize_another_receipt():
    captured = source("a")
    read_schema = json.loads(schema("path", "start_line", "end_line", "max_chars"))
    read_schema["properties"]["path"]["enum"] = ["elsewhere.py"]
    with pytest.raises(RecoveryRouteUnavailable):
        validate_recovery(replace(snapshot(captured), recovery_schema=json.dumps(read_schema).encode()), captured)


@pytest.mark.parametrize("lossy", [False, True])
def test_recovery_reexecutes_the_actual_pager_at_the_captured_column(lossy):
    # A saved decoded-span receipt can contain a replacement character. Live
    # invalid UTF-8 is BINARY_FILE on this release and cannot produce a page.
    raw = b"header\n" + b"prefix " + ("\ufffd".encode("utf-8") if lossy else b"z") + b"x" * 3000 + b"\n"
    original = {"path": "a.py", "start_line": 2, "column": 4, "max_chars": 512}
    page = deliver_source(raw, "a.py", 0, args=original)
    metadata = page.metadata
    effective = metadata["effective_range"]
    span = (DecodedSpan(effective["start_column"], effective["end_column"],
                        metadata["presentation_version"]) if lossy else None)
    identity = ReceiptIdentity("a.py", metadata["revision"],
        effective["start_line"], effective["end_line"],
        None if lossy else metadata["byte_start"], None if lossy else metadata["byte_end"],
        hashlib.sha256(page.content.encode()).hexdigest(), span)
    digests, numbers = line_digests_with_numbers(page.content, identity.first_line)
    captured = SourceSlot("a", ("a",), identity, page.content, page.text,
        digests, numbers, json.dumps(original).encode(), 1, 1, "RESIDENT")
    arguments = validate_recovery(snapshot(captured), captured)
    recovered = deliver_source(raw, "a.py", 0, args=arguments)
    assert recovered.result_class == "OK"
    assert recovered.content == page.content
    assert recovered.metadata["effective_range"] == page.metadata["effective_range"]


def test_recovery_failure_remains_a_typed_shadow_diagnostic():
    captured = source("a", state="RETIRING")
    result = prepare_candidate(replace(snapshot(captured), recovery_owner_allowed=False))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.RECOVERY_ROUTE_UNAVAILABLE


def test_optional_without_an_executable_omission_route_keeps_full_bytes():
    captured = source("a")
    result = prepare_candidate(replace(snapshot(captured), recovery_owner_allowed=False))
    assert isinstance(result, PreparedCandidate)
    assert dict(result.rendered.certificate.assignment) == {"a": "FULL"}
