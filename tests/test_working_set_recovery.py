"""Recovery routes are executable; unavailable read surfaces are explicit."""
import hashlib
import json
from dataclasses import replace

import pytest
from kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
from kernel_source_delivery import deliver_source
from kernel_working_set_allocator import DEFERRED
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_proof import verify_certificate
from kernel_working_set_recovery import RecoveryRouteUnavailable, validate_recovery
from kernel_working_set_render import render_assignment
from kernel_working_set_snapshot import SourceSlot
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


@pytest.mark.parametrize("raw,original,path", [
    (b"hello world\n", {"start_line": 1}, "a.py"),
    (b"x" * 1000 + b"\n", {"start_line": 1}, "a.py"),
    (b"head\n" + b"x" * 3000 + b"\n",
     {"start_line": 2, "max_chars": 1513}, "a.py"),
    (b"head\n" + b"x" * 3000 + b"\n",
     {"start_line": 2, "column": 731, "max_chars": 1513}, "a.py"),
    (("head\n" + "\u03bb" * 2000 + "\n").encode(),
     {"start_line": 2, "column": 19, "max_chars": 1513}, "a.py"),
    (b"head\n" + b"x\n" * 130 + b"tail\n",
     {"start_line": 2, "end_line": 131}, "a.py"),
    (b"x" * 2000 + b"\n", {"start_line": 1, "max_chars": 1513},
     "/".join(["long_directory" * 3] * 6) + "/a.py"),
    (b"x" * 70000 + b"\n", {"start_line": 1, "max_chars": 64000}, "a.py"),
], ids=["complete-small", "complete-1001", "partial-1001", "partial-column",
        "partial-unicode", "complete-many-lines", "long-path", "maximum-page"])
def test_recovery_restores_exact_complete_and_partial_intervals(raw, original, path):
    original = {"path": path, **original}
    page = deliver_source(raw, path, 0, args=original)
    assert page.result_class == "OK"
    effective = page.metadata["effective_range"]
    identity = ReceiptIdentity(path, page.metadata["revision"],
        effective["start_line"], effective["end_line"],
        effective["byte_start"], effective["byte_end"],
        hashlib.sha256(page.content.encode()).hexdigest(), None)
    digests, numbers = line_digests_with_numbers(page.content, identity.first_line)
    captured = SourceSlot("a", ("a",), identity, page.content, page.text,
        digests, numbers, json.dumps(original).encode(), 1, 1, "RESIDENT")

    rendered = render_assignment(snapshot(captured), {captured.key: DEFERRED})
    assert verify_certificate(rendered.wire, rendered.certificate)
    arguments = json.loads(rendered.certificate.witnesses[0].recovery_arguments)
    recovered = deliver_source(raw, path, 0, args=arguments)

    assert recovered.result_class == "OK"
    assert recovered.content == page.content
    assert recovered.metadata["effective_range"] == effective
    assert recovered.metadata["revision"] == identity.revision


def test_recovery_refuses_payload_that_only_fits_without_header_reserve():
    captured = source("a", "x" * (64000 - 511))
    with pytest.raises(RecoveryRouteUnavailable, match="exceeds a single read page"):
        validate_recovery(snapshot(captured), captured)


def test_recovery_refuses_schema_that_cannot_budget_the_whole_page():
    captured = source("a")
    read_schema = json.loads(schema("path", "start_line", "end_line", "max_chars"))
    read_schema["properties"]["max_chars"]["maximum"] = len(captured.payload)
    snap = replace(snapshot(captured), recovery_schema=json.dumps(read_schema).encode())
    with pytest.raises(RecoveryRouteUnavailable):
        validate_recovery(snap, captured)


@pytest.mark.parametrize("state,previous", [("DEGRADED", "D"), ("RETIRING", "F")])
def test_retirement_without_owner_read_has_a_nonrecoverable_notice(state, previous):
    captured = replace(source("a", state=state), previous_mode=previous)
    result = prepare_candidate(replace(snapshot(captured), recovery_owner_allowed=False))
    assert isinstance(result, PreparedCandidate)
    assert dict(result.rendered.certificate.assignment) == {"a": "DEFERRED"}
    assert b"Read recovery unavailable" in result.wire


def test_retiring_resident_without_a_route_is_kept_full():
    # Wave G1 (A7): reserve retirement without a valid retirement representation
    # keeps the resident instead of proving an empty domain.
    captured = replace(source("a", state="RETIRING"), previous_mode="F")
    snap = replace(snapshot(captured),
                   recovery_schema=schema("path", required=("path", "unavailable")))
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate)
    assert dict(result.rendered.certificate.assignment) == {"a": "FULL"}


def test_due_without_an_executable_omission_route_keeps_full_bytes():
    captured = source("a", state="DUE")
    result = prepare_candidate(replace(snapshot(captured), recovery_owner_allowed=False))
    assert isinstance(result, PreparedCandidate)
    assert dict(result.rendered.certificate.assignment) == {"a": "FULL"}
