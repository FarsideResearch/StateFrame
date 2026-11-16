"""kernel_edit_repair_request: every refusal code yields a next request carrying the current
bytes, fresh handles and the proposal verbatim; the fingerprint ladder escalates
RETRY -> NARROW -> BLOCKED; everything is deterministic from the request ordinal."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_handles import DeliveredSpan, HandleMap, mint_handles  # noqa: E402
from kernel_edit_refusal_codes import HANDLE_PATTERN, Refusal, RefusalCode  # noqa: E402
from kernel_edit_repair_request import (  # noqa: E402
    BLOCKED_AT, BLOCKED_MARKER, NARROW_AT, REFUSED_MARKER, SCHEMA, SOURCE_HEADER_PREFIX,
    EscalationLevel, FingerprintLedger, ProposedEdit, RepairRequest, build, level_for,
)
from kernel_edit_source_batch import CoveredSpan, validate_edit_source_batch  # noqa: E402

PATH = "pkg/mod.py"
SOURCE = (
    "import os\n"            # 0..10
    "\n"                     # 10..11
    "def alpha():\n"         # 11..24
    "    return 1\n"         # 24..37
    "\n"                     # 37..38
    "def beta():\n"          # 38..50
    "    return 2\n"         # 50..63
    "\n"                     # 63..64
    "def gamma():\n"         # 64..77
    "    return 3\n"         # 77..90
)
RAW = SOURCE.encode("utf-8")
REV = "sha256:" + hashlib.sha256(RAW).hexdigest()
AUTH = dict(current_revision=REV, authority_ref="authority:test",
            inventory_ref="inventory:test", source_receipt="receipt:test")


def minter(request_ordinal: int, path: str, source: str) -> HandleMap:
    return mint_handles(request_ordinal, [DeliveredSpan.from_source(path, source)])


def py_parse(raw: bytes) -> str | None:
    try:
        ast.parse(raw.decode("utf-8"))
    except SyntaxError as error:
        return str(error)
    return None


def whole_file(handle: str = "q1h1", **overrides) -> CoveredSpan:
    fields = dict(handle=handle, path=PATH, revision=REV, start=0, end=len(RAW))
    fields.update(overrides)
    return CoveredSpan(**fields)


def alpha_span(handle: str = "q1h1") -> CoveredSpan:
    return CoveredSpan(handle, PATH, REV, 11, 37)


def beta_span(handle: str = "q1h2") -> CoveredSpan:
    return CoveredSpan(handle, PATH, REV, 38, 63)


def refuse(edits, spans, source=RAW, **kw):
    result = validate_edit_source_batch(edits, spans, source, **{**AUTH, **kw})
    assert not result.ok, "fixture must induce a refusal"
    return result


def covered_from(repair: RepairRequest) -> list[CoveredSpan]:
    """Translate the fresh handle map into validator spans, as the host executor would."""
    assert repair.handle_map is not None
    return [CoveredSpan(record.handle, record.path, record.revision,
                        record.span.byte_start, record.span.byte_end, kind=record.kind.value)
            for record in repair.handle_map.records]


# ----------------------------------------------------------- one case per refusal code

# (code, proposal, spans, extra validator kwargs, corrected batch written against q2h1)
CASES = {
    RefusalCode.TOO_MANY_EDITS: (
        [("q1h1", f"return {n}", f"return {n}0") for n in (1, 2, 3)]
        + [("q1h1", "import os", "import sys"), ("q1h1", "nope", "x")],
        [whole_file()], {}, [("q2h1", "return 1", "return 10")]),
    RefusalCode.MIXED_FILE_BATCH: (
        [("q1h1", "return 1", "return 10"), ("q1h9", "x", "y")],
        [alpha_span(), CoveredSpan("q1h9", "pkg/other.py", REV, 0, 5)], {},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.STALE_HANDLE: (
        [("q0h1", "return 1", "return 10")], [alpha_span()], {},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.SOURCE_CHANGED: (
        [("q1h1", "return 1", "return 10")], [alpha_span()], {"current_revision": "sha256:old"},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.NOT_EDITABLE: (
        [("q1h1", "return 1", "return 10")], [CoveredSpan("q1h1", PATH, REV, 11, 37, editable=False)], {},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.EMPTY_OLD_TEXT: (
        [("q1h1", "", "inserted\n")], [whole_file()], {},
        [("q2h1", "import os\n", "import os\ninserted\n")]),
    RefusalCode.DISPLAY_MARKER: (
        [("q1h1", "[CURRENT SOURCE]\nreturn 1", "x")], [whole_file()], {},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.OLD_NOT_FOUND: (
        [("q1h1", "return 99", "return 1")], [alpha_span()], {},
        [("q2h1", "return 1", "return 10")]),
    RefusalCode.UNCOVERED_SOURCE: (
        [("q1h1", "return 2", "return 20")], [alpha_span()], {},
        [("q2h1", "return 2", "return 20")]),
    RefusalCode.AMBIGUOUS_MATCH: (
        [("q1h1", "return", "yield")], [whole_file()], {},
        [("q2h1", "    return 1", "    yield 1")]),
    RefusalCode.OVERLAPPING_EDITS: (
        [("q1h1", "def alpha():\n    return 1", "def alpha():\n    return 10"),
         ("q1h1", "return 1\n", "return 11\n")], [whole_file()], {},
        [("q2h1", "def alpha():\n    return 1", "def alpha():\n    return 10")]),
    RefusalCode.PARSE_BREAK: (
        [("q1h1", "return 1", "return (")], [alpha_span()], {"parse_check": py_parse},
        [("q2h1", "return 1", "return (1)")]),
}


def test_cases_cover_the_closed_catalog():
    assert set(CASES) == set(RefusalCode)


@pytest.mark.parametrize("code", list(CASES), ids=lambda c: c.value)
def test_each_code_yields_a_repair_request_with_current_bytes_fresh_handles_and_proposal(code):
    proposal, spans, kwargs, corrected = CASES[code]
    result = refuse(proposal, spans, **kwargs)
    assert code in result.codes

    repair = build(result.refusals, RAW, minter, proposal, path=PATH, request_ordinal=2)

    # Current bytes, verbatim, and the revision they hash to.
    assert repair.current_source == SOURCE
    assert repair.current_revision == REV
    assert SOURCE in repair.render()
    # Fresh handles belong to the request being built and never to the old one.
    assert repair.handles == ("q2h1",)
    assert "q1h1" not in repair.handle_map.render()
    assert repair.handle_map.render() in repair.render()
    # The proposal is carried verbatim (including the stale handle the model used).
    assert [edit.as_dict() for edit in repair.rejected_proposal] == [
        {"handle": h, "old_text": o, "new_text": n} for h, o, n in proposal]
    assert repair.render_proposal() in repair.render()
    # The code list and public text are in the request; codes and fingerprints in the payload.
    assert code in repair.codes
    for refusal in repair.refusals:
        assert f"{refusal.code.value}: {refusal.public_text}" in repair.render()
    assert repair.as_dict()["codes"] == [c.value for c in repair.codes]
    assert repair.fingerprints == tuple(r.fingerprint for r in result.refusals)
    # First refusal: plain retry.
    assert repair.level is EscalationLevel.RETRY and not repair.blocked
    assert repair.render().startswith(REFUSED_MARKER)
    assert BLOCKED_MARKER not in repair.render()

    # Applying the corrected batch against the fresh handles succeeds with zero reads:
    # the bytes come from the repair request, not from a read tool.
    fixed = validate_edit_source_batch(corrected, covered_from(repair), repair.current_source,
                                       current_revision=repair.current_revision,
                                       authority_ref="authority:test",
                                       inventory_ref="inventory:test",
                                       source_receipt="receipt:test", parse_check=py_parse)
    assert fixed.ok, fixed.as_dict()


def test_observation_carries_only_codes_and_fingerprints():
    result = refuse([("q1h1", "return 99", "return 1")], [alpha_span()])
    repair = build(result.refusals, RAW, minter, [("q1h1", "return 99", "return 1")],
                   path=PATH, request_ordinal=7)
    observation = repair.observation()
    assert "OLD_NOT_FOUND (edit 0)" in observation
    assert result.refusals[0].fingerprint in observation
    assert HANDLE_PATTERN.search(observation) is None
    assert "return 99" not in observation and "def alpha" not in observation
    assert PATH not in observation
    assert observation.startswith(REFUSED_MARKER) and "refused" in observation


def test_source_header_is_a_renderer_marker_so_copying_it_is_refused():
    result = refuse([("q1h1", "return 99", "return 1")], [alpha_span()])
    repair = build(result.refusals, RAW, minter, [("q1h1", "return 99", "return 1")],
                   path=PATH, request_ordinal=2)
    header = repair.render_source().split("\n", 1)[0]
    assert header.startswith(SOURCE_HEADER_PREFIX) and PATH in header and REV in header
    copied = validate_edit_source_batch([("q2h1", header + "\nimport os", "x")],
                                        covered_from(repair), RAW, **AUTH)
    assert copied.codes == (RefusalCode.DISPLAY_MARKER,)


# -------------------------------------------------------------------- escalation ladder


def test_level_for_ladder_constants():
    assert (NARROW_AT, BLOCKED_AT) == (2, 3)
    assert level_for(1) is EscalationLevel.RETRY
    assert level_for(2) is EscalationLevel.NARROW
    assert level_for(3) is EscalationLevel.BLOCKED
    assert level_for(9) is EscalationLevel.BLOCKED
    with pytest.raises(ValueError):
        level_for(0)


def test_identical_fingerprint_escalates_retry_narrow_blocked_across_handle_renames():
    proposal = [("q1h1", "return 99", "return 1")]
    first = refuse(proposal, [alpha_span()])
    r1 = build(first.refusals, RAW, minter, proposal, path=PATH, request_ordinal=2)
    assert (r1.level, r1.repeat_count) == (EscalationLevel.RETRY, 1)

    # Same mistake again, now spelled with the fresh handle: fingerprint ignores handles.
    retry = [("q2h1", "return 99", "return 1")]
    second = refuse(retry, covered_from(r1))
    assert second.refusals[0].fingerprint == first.refusals[0].fingerprint
    r2 = build(second.refusals, RAW, minter, retry, path=PATH, request_ordinal=3, ledger=r1.ledger)
    assert (r2.level, r2.repeat_count) == (EscalationLevel.NARROW, 2)
    assert "exactly one edit" in r2.instruction() and r2.instruction() in r2.render()
    assert not r2.blocked and BLOCKED_MARKER not in r2.render()

    third_try = [("q3h1", "return 99", "return 1")]
    third = refuse(third_try, covered_from(r2))
    r3 = build(third.refusals, RAW, minter, third_try, path=PATH, request_ordinal=4, ledger=r2.ledger)
    assert (r3.level, r3.repeat_count) == (EscalationLevel.BLOCKED, 3)
    assert r3.blocked
    assert r3.render().startswith(BLOCKED_MARKER)
    assert r3.observation().startswith(BLOCKED_MARKER)
    assert "blocked" in r3.instruction()
    assert r3.ledger.count(first.refusals[0].fingerprint) == 3


def test_a_different_mistake_resets_to_retry_but_the_ledger_remembers():
    proposal = [("q1h1", "return 99", "return 1")]
    first = refuse(proposal, [alpha_span()])
    r1 = build(first.refusals, RAW, minter, proposal, path=PATH, request_ordinal=2)
    other = [("q2h1", "return", "yield")]
    second = refuse(other, [whole_file("q2h1")])
    assert second.codes == (RefusalCode.AMBIGUOUS_MATCH,)
    r2 = build(second.refusals, RAW, minter, other, path=PATH, request_ordinal=3, ledger=r1.ledger)
    assert (r2.level, r2.repeat_count) == (EscalationLevel.RETRY, 1)
    assert r2.ledger.count(first.refusals[0].fingerprint) == 1
    assert r2.ledger.count(second.refusals[0].fingerprint) == 1


def test_mixed_batch_escalates_on_its_worst_fingerprint():
    stale = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path=PATH, old_text="a")
    fresh = Refusal.build(RefusalCode.AMBIGUOUS_MATCH, edit_index=1, path=PATH, old_text="b")
    ledger = FingerprintLedger().record([stale.fingerprint]).record([stale.fingerprint])
    repair = build([fresh, stale], RAW, minter, [("q1h1", "a", "x"), ("q1h1", "b", "y")],
                   path=PATH, request_ordinal=5, ledger=ledger)
    assert repair.level is EscalationLevel.BLOCKED and repair.repeat_count == 3
    # Refusals are ordered by edit index regardless of the order handed in.
    assert [r.edit_index for r in repair.refusals] == [0, 1]


def test_ledger_round_trips_through_plain_data():
    ledger = FingerprintLedger().record(["b" * 64, "a" * 64]).record(["a" * 64])
    restored = FingerprintLedger.from_dict(json.loads(json.dumps(ledger.as_dict())))
    assert restored == ledger
    assert restored.count("a" * 64) == 2 and restored.count("zz") == 0
    assert ledger.counts == (("a" * 64, 2), ("b" * 64, 1))
    with pytest.raises(ValueError):
        FingerprintLedger((("x", 0),))
    with pytest.raises(ValueError):
        FingerprintLedger((("b", 1), ("a", 1)))


# ------------------------------------------------------------------------- determinism


def test_payload_is_deterministic_and_replay_remints_identically():
    proposal = [("q1h1", "return 99", "return 1")]
    result = refuse(proposal, [alpha_span()])
    a = build(result.refusals, RAW, minter, proposal, path=PATH, request_ordinal=42)
    b = build(list(result.refusals), bytearray(RAW), minter, [dict(handle="q1h1", old_text="return 99", new_text="return 1")],
              path=PATH, request_ordinal=42)
    assert a.as_dict() == b.as_dict()
    assert a.render() == b.render() and a.observation() == b.observation()
    assert a.digest() == b.digest()
    assert a.handles == ("q42h1",)
    # A different request ordinal mints different handles and nothing else changes.
    c = build(result.refusals, RAW, minter, proposal, path=PATH, request_ordinal=43)
    assert c.handles == ("q43h1",)
    assert c.fingerprints == a.fingerprints and c.ledger == a.ledger
    json.dumps(a.as_dict())  # journal-safe
    assert a.as_dict()["schema"] == SCHEMA


def test_bytes_with_bom_and_crlf_are_carried_verbatim():
    source = "﻿def f():\r\n    return 1\r\n"
    raw = source.encode("utf-8")
    rev = "sha256:" + hashlib.sha256(raw).hexdigest()
    result = validate_edit_source_batch([("q1h1", "return 9", "x")],
                                        [CoveredSpan("q1h1", PATH, rev, 0, len(raw))], raw,
                                        **{**AUTH, "current_revision": rev})
    repair = build(result.refusals, raw, minter, [("q1h1", "return 9", "x")],
                   path=PATH, request_ordinal=2)
    assert repair.current_source == source and repair.current_revision == rev
    assert repair.handle_map.records[0].text(source) == source


def test_empty_file_gets_no_handles_but_a_clear_statement():
    refusal = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path=PATH, old_text="x")
    repair = build([refusal], b"", minter, [("q1h1", "x", "y")], path=PATH, request_ordinal=2)
    assert repair.handle_map is None and repair.handles == ()
    assert "The file is empty" in repair.render()
    assert repair.as_dict()["handle_map"] is None


def test_proposal_edits_accept_mappings_tuples_and_objects_verbatim():
    class Edit:
        handle, old_text, new_text = "q1h1", "  spaced \n", "x"
    edits = [ProposedEdit.coerce(v, i) for i, v in enumerate(
        [Edit(), ("q1h1", "  spaced \n", "x"), {"handle": "q1h1", "old_text": "  spaced \n", "new_text": "x"}])]
    assert len(set(edits)) == 1 and edits[0].old_text == "  spaced \n"
    with pytest.raises(ValueError):
        ProposedEdit.coerce(("q1h1", 1, "x"), 0)
    with pytest.raises(ValueError):
        ProposedEdit.coerce("q1h1", 0)


@pytest.mark.parametrize("kwargs", [
    dict(refusals=[]),
    dict(refusals=["OLD_NOT_FOUND"]),
    dict(path=""),
    dict(request_ordinal=-1),
    dict(request_ordinal=2.0),
    dict(minter=None),
    dict(rejected_proposal=[]),
    dict(current_bytes=b"\xff\xfe"),
    dict(current_bytes=12),
])
def test_malformed_inputs_raise(kwargs):
    refusal = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path=PATH, old_text="x")
    params = dict(refusals=[refusal], current_bytes=RAW, minter=minter,
                  rejected_proposal=[("q1h1", "x", "y")], path=PATH, request_ordinal=2)
    params.update(kwargs)
    with pytest.raises(ValueError):
        build(**params)


def test_minter_returning_another_ordinal_is_rejected():
    refusal = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path=PATH, old_text="x")
    with pytest.raises(ValueError):
        build([refusal], RAW, lambda n, p, s: minter(n + 1, p, s), [("q1h1", "x", "y")],
              path=PATH, request_ordinal=2)
    with pytest.raises(ValueError):
        build([refusal], RAW, lambda n, p, s: object(), [("q1h1", "x", "y")],
              path=PATH, request_ordinal=2)


# ------------------------------------------------------------------------------ purity


def test_module_is_pure_and_unwired():
    text = (ENGINE / "kernel_edit_repair_request.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not imported & {"os", "sys", "io", "pathlib", "time", "subprocess", "random"}
    assert "os.environ" not in text and "open(" not in text
    # The handle module is injected, never imported, so legacy requests stay byte-identical.
    assert not re.search(r"^\s*(from|import)\s+kernel_edit_handles\b", text, re.MULTILINE)
    assert "kernel_edit_source_batch" not in text
    # The repair hook is the single wiring point; nothing else imports the builder.
    for py in (ROOT / "src").rglob("*.py"):
        if py.name not in {"kernel_edit_repair_request.py", "kernel_edit_repair_hook.py"}:
            assert "kernel_edit_repair_request" not in py.read_text(encoding="utf-8"), py
