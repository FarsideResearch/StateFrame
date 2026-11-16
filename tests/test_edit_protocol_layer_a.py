"""Edit protocol Layer A fixtures (CPU, no model): one area per refusal code.

Each area walks the whole protocol round trip the way the host executor will run it:

  request q1  the host delivers source and mints handles (kernel_edit_handles);
  proposal    the model sends an ``edit_source`` batch that the batch validator
              (kernel_edit_source_batch) refuses with exactly the area's code;
  request q2  the refusal becomes the next request (kernel_edit_repair_request): the
              current bytes, fresh handles and the rejected proposal, verbatim;
  correction  the model answers with a batch written ONLY from that payload (handles from
              the fresh map, old_text copied from the delivered current source); the host
              validates it, binds the handles through EditAuthority (kernel_revision_bound_edit)
              and composes the candidate. No read tool call happens anywhere in between.

The modules stay unwired: everything here is glued together inside the test, with the
two adapters the host will need (minted handle record -> validator span; current bytes ->
current revision) written once at the top.
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import kernel_revision_bound_edit as rbe  # noqa: E402
from kernel_edit_handles import DeliveredSpan, HandleMap, HandleRecord, mint_handles  # noqa: E402
from kernel_edit_refusal_codes import HANDLE_PATTERN, RefusalCode, fingerprint  # noqa: E402
from kernel_edit_repair_request import (  # noqa: E402
    EscalationLevel, FingerprintLedger, RepairRequest, build, source_revision,
)
from kernel_edit_source_batch import (  # noqa: E402
    UNRESOLVED_PATH, BatchValidation, CoveredSpan, validate_edit_source_batch,
)

PATH = "pkg/mod.py"
OTHER = "pkg/other.py"
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
OTHER_SOURCE = "VALUE = 1\n"
# Same length as SOURCE so the delivered span still fits: only the digest can tell.
DRIFTED_SOURCE = SOURCE.replace("return 2", "return 9")
AUTHORITY_REF = "authority:layer-a"
INVENTORY_REF = "inventory:layer-a"
# The host protects the import block of this file (lines 1-2); the body is editable.
PROTECTED_LINES = {PATH: (1, 2)}


def py_parse(raw: bytes) -> str | None:
    try:
        ast.parse(raw.decode("utf-8"))
    except SyntaxError as error:
        return str(error)
    return None


# ------------------------------------------------------------------ host adapters


def current_revision_of(raw: bytes) -> str:
    """The revision the validator compares against is computed from the bytes it is
    handed, never taken from the span (that is what SOURCE_CHANGED detects)."""
    return source_revision(raw.decode("utf-8"))


def covered_span(record: HandleRecord, *, editable: bool = True) -> CoveredSpan:
    """Minted handle record (nested byte span, enum kind) -> validator span."""
    return CoveredSpan(record.handle, record.path, record.revision,
                       record.span.byte_start, record.span.byte_end,
                       kind=record.kind.value, editable=editable)


def host_spans(path: str, source: str) -> list[DeliveredSpan]:
    """What the host delivers for a file: the protected block (if any) and the rest,
    so a protected region gets its own handle that the validator marks read-only."""
    total = len(source.splitlines(keepends=True))
    protected = PROTECTED_LINES.get(path)
    if protected is None or protected[1] >= total:
        return [DeliveredSpan.from_source(path, source)]
    return [DeliveredSpan.from_source(path, source, start_line=protected[0], end_line=protected[1]),
            DeliveredSpan.from_source(path, source, start_line=protected[1] + 1, end_line=total)]


def is_protected(record: HandleRecord) -> bool:
    protected = PROTECTED_LINES.get(record.path)
    return protected is not None and (record.span.start_line, record.span.end_line) == protected


def spans_for_validator(handle_map: HandleMap) -> list[CoveredSpan]:
    return [covered_span(record, editable=not is_protected(record))
            for record in handle_map.records]


@dataclass
class Host:
    """The executor side: files, the request counter, minted maps and the authority.

    ``model_reads`` counts read tool calls the model made; Layer A requires it to stay
    at zero after the first delivery."""

    root: Path
    files: dict[str, bytes]
    maps: dict[int, HandleMap] = field(default_factory=dict)
    model_reads: int = 0

    def deliver(self, request_ordinal: int, *paths: str) -> HandleMap:
        """Mint handles for the files a request shows the model (a read tool result)."""
        spans: list[DeliveredSpan] = []
        for path in paths:
            spans.extend(host_spans(path, self.files[path].decode("utf-8")))
        self.maps[request_ordinal] = mint_handles(request_ordinal, spans)
        self.model_reads += 1
        return self.maps[request_ordinal]

    def mint(self, request_ordinal: int, path: str, source: str) -> HandleMap:
        """Minter handed to the repair-request builder: same policy as a delivery, but
        it is not a read by the model (the host already holds the bytes)."""
        assert self.files[path].decode("utf-8") == source
        self.maps[request_ordinal] = mint_handles(request_ordinal, host_spans(path, source))
        return self.maps[request_ordinal]

    def known_spans(self, *ordinals: int) -> list[CoveredSpan]:
        spans: list[CoveredSpan] = []
        for ordinal in ordinals:
            spans.extend(spans_for_validator(self.maps[ordinal]))
        return spans

    def validate(self, edits: Sequence[object], spans: Sequence[CoveredSpan],
                 path: str, **kw: object) -> BatchValidation:
        raw = self.files[path]
        params = dict(current_revision=current_revision_of(raw), authority_ref=AUTHORITY_REF,
                      inventory_ref=INVENTORY_REF, source_receipt=f"receipt:{path}",
                      parse_check=py_parse)
        params.update(kw)
        return validate_edit_source_batch(edits, spans, raw, **params)

    def authority(self, path: str) -> rbe.EditAuthority:
        return rbe.EditAuthority(root=self.root, sources={path: self.files[path]},
                                 base_revision=current_revision_of(self.files[path]),
                                 authority_ref=AUTHORITY_REF)


@pytest.fixture
def host(tmp_path: Path) -> Host:
    for path, data in {PATH: SOURCE, OTHER: OTHER_SOURCE}.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data, encoding="utf-8")
    return Host(root=tmp_path, files={PATH: SOURCE.encode("utf-8"),
                                      OTHER: OTHER_SOURCE.encode("utf-8")})


# ---------------------------------------------------------------------- the areas
#
# Each area: how the host/model reach the refusal, and how the model corrects it from
# the repair payload alone. ``correction`` is written against the *repair* map and is
# given the fresh handle map so it can pick handles the way a model reading the rendered
# map would (first handle of the editable body, etc.).


@dataclass(frozen=True)
class Area:
    code: RefusalCode
    # (host, q1 map) -> (proposal edits, spans known to request q1)
    induce: object
    # (repair map) -> corrected batch of (handle, old_text, new_text)
    correct: object
    # expected bytes of PATH after the correction
    expected: bytes


def body(handle_map: HandleMap) -> HandleRecord:
    """The editable body handle of PATH in a map (the model reads this off the map)."""
    records = [r for r in handle_map.for_path(PATH) if not is_protected(r)]
    assert records, "the body of the file must be delivered"
    return records[0]


def header(handle_map: HandleMap) -> HandleRecord:
    records = [r for r in handle_map.for_path(PATH) if is_protected(r)]
    assert records, "the protected block must be delivered"
    return records[0]


Batch = list[tuple[str, str, str]]


def plain(host: Host, q1: HandleMap, *edits: tuple[str, str]) -> tuple[Batch, list[CoveredSpan]]:
    return [(body(q1).handle, old, new) for old, new in edits], host.known_spans(1)


AREAS = [
    Area(RefusalCode.TOO_MANY_EDITS,
         lambda host, q1: plain(host, q1, ("return 1", "return 10"), ("return 2", "return 20"),
                                ("return 3", "return 30"), ("def gamma", "def delta"),
                                ("def beta", "def epsilon")),
         lambda q2: [(body(q2).handle, "return 1", "return 10"),
                     (body(q2).handle, "return 2", "return 20"),
                     (body(q2).handle, "return 3", "return 30"),
                     (body(q2).handle, "def gamma", "def delta")],
         SOURCE.replace("return 1", "return 10").replace("return 2", "return 20")
         .replace("return 3", "return 30").replace("def gamma", "def delta").encode()),
    Area(RefusalCode.MIXED_FILE_BATCH,
         lambda host, q1: ([(body(q1).handle, "return 1", "return 10"),
                            (q1.for_path(OTHER)[0].handle, "VALUE = 1", "VALUE = 2")],
                           host.known_spans(1)),
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.STALE_HANDLE,
         # The model reuses a handle from request q0, which q1 retired.
         lambda host, q1: ([(body(host.maps[0]).handle, "return 1", "return 10")],
                           host.known_spans(1)),
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.SOURCE_CHANGED,
         # The file drifted (same length) between delivery and the edit.
         lambda host, q1: (host.files.__setitem__(PATH, DRIFTED_SOURCE.encode()),
                           plain(host, q1, ("return 1", "return 10")))[1],
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         DRIFTED_SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.NOT_EDITABLE,
         lambda host, q1: ([(header(q1).handle, "import os", "import sys")], host.known_spans(1)),
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.EMPTY_OLD_TEXT,
         lambda host, q1: plain(host, q1, ("", "inserted = True\n")),
         lambda q2: [(body(q2).handle, "def alpha():\n", "inserted = True\n\n\ndef alpha():\n")],
         SOURCE.replace("def alpha():\n", "inserted = True\n\n\ndef alpha():\n").encode()),
    Area(RefusalCode.DISPLAY_MARKER,
         lambda host, q1: plain(host, q1, ("[CURRENT SOURCE]\n    return 1", "    return 10")),
         lambda q2: [(body(q2).handle, "    return 1", "    return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.OLD_NOT_FOUND,
         lambda host, q1: plain(host, q1, ("return 99", "return 1")),
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.UNCOVERED_SOURCE,
         # Text that exists in the file, but inside the protected block, not the body.
         lambda host, q1: plain(host, q1, ("import os", "import sys")),
         lambda q2: [(body(q2).handle, "return 1", "return 10")],
         SOURCE.replace("return 1", "return 10").encode()),
    Area(RefusalCode.AMBIGUOUS_MATCH,
         lambda host, q1: plain(host, q1, ("return", "yield")),
         lambda q2: [(body(q2).handle, "    return 1", "    yield 1")],
         SOURCE.replace("    return 1", "    yield 1").encode()),
    Area(RefusalCode.OVERLAPPING_EDITS,
         lambda host, q1: plain(host, q1,
                                ("def alpha():\n    return 1", "def alpha():\n    return 10"),
                                ("return 1\n", "return 11\n")),
         lambda q2: [(body(q2).handle, "def alpha():\n    return 1", "def alpha():\n    return 10"),
                     (body(q2).handle, "return 2\n", "return 20\n")],
         SOURCE.replace("return 1", "return 10").replace("return 2", "return 20").encode()),
    Area(RefusalCode.PARSE_BREAK,
         lambda host, q1: plain(host, q1, ("return 1", "return (")),
         lambda q2: [(body(q2).handle, "return 1", "return (1)")],
         SOURCE.replace("return 1", "return (1)").encode()),
]


def test_areas_cover_the_closed_catalog_once_each():
    assert {area.code for area in AREAS} == set(RefusalCode)
    assert len(AREAS) == len(RefusalCode) == 12


# ------------------------------------------------------------------- the round trip


def run_round_trip(host: Host, area: Area) -> tuple[RepairRequest, BatchValidation, bytes]:
    # q0: an earlier request that delivered the file; q1 retires it.
    host.deliver(0, PATH)
    q1 = host.deliver(1, PATH, OTHER)
    reads_after_delivery = host.model_reads

    # The model's proposal is refused with exactly this area's code.
    proposal, spans = area.induce(host, q1)
    refused = host.validate(proposal, spans, PATH)
    # The authority is the host's view of the file at the time the batch is judged.
    authority = host.authority(PATH)
    authority.consume(0)
    assert not refused.ok and refused.plan is None
    assert set(refused.codes) == {area.code}, refused.codes
    assert all(HANDLE_PATTERN.search(r.public_text) is None for r in refused.refusals)

    # The refusal becomes request q2: current bytes, fresh handles, proposal verbatim.
    repair = build(refused.refusals, host.files[PATH], host.mint, proposal,
                   path=PATH, request_ordinal=2)
    assert repair.level is EscalationLevel.RETRY and repair.repeat_count == 1
    assert repair.current_source == host.files[PATH].decode("utf-8")
    assert repair.current_revision == current_revision_of(host.files[PATH])
    assert repair.handle_map is not None and repair.handle_map.request_ordinal == 2
    assert [e.handle for e in repair.rejected_proposal] == [e[0] for e in proposal]
    assert HANDLE_PATTERN.search(repair.observation()) is None
    assert area.code.value in repair.observation()
    rendered = repair.render()
    for handle in repair.handles:
        assert handle in rendered
    authority.consume(1)

    # The model corrects from the payload alone: fresh handles, text from current source.
    corrected = area.correct(repair.handle_map)
    assert 1 <= len(corrected) <= 4
    for handle, old_text, _new in corrected:
        assert handle in repair.handles
        assert old_text and old_text in repair.current_source
        assert old_text in rendered
    assert host.model_reads == reads_after_delivery

    # Host side 1: the batch validator accepts it against the fresh spans and current bytes.
    accepted = host.validate(corrected, spans_for_validator(repair.handle_map), PATH)
    assert accepted.ok, accepted.codes
    assert accepted.plan is not None and accepted.plan.base_revision == repair.current_revision
    admission = authority.admit(accepted.plan)
    assert admission.accepted, admission.diagnostics
    via_validator = rbe.compose_candidate(authority, accepted.plan)[PATH]

    # Host side 2: the same edits bound handle-by-handle through EditAuthority.
    refs = []
    for handle, old_text, new_text in corrected:
        record = repair.handle_map.lookup(handle)
        assert record is not None
        refs.append((authority.bind_handle(record, old_text=old_text), new_text))
    plan = rbe.EditPlan(base_revision=authority.base_revision, authority_ref=AUTHORITY_REF,
                        inventory_ref=INVENTORY_REF,
                        edits=tuple(rbe.EditOperation("replace", ref, new, order=i)
                                    for i, (ref, new) in enumerate(refs)))
    admission = authority.admit(plan)
    assert admission.accepted, admission.diagnostics
    via_authority = rbe.compose_candidate(authority, plan)[PATH]

    assert via_validator == via_authority == area.expected
    assert py_parse(via_authority) is None
    return repair, accepted, via_authority


@pytest.mark.parametrize("area", AREAS, ids=[area.code.value for area in AREAS])
def test_refusal_round_trips_to_an_applied_edit_with_zero_reads(host: Host, area: Area):
    repair, accepted, result = run_round_trip(host, area)
    assert result != host.files[PATH]  # something was actually changed
    # Replay: the same inputs rebuild the identical payload (same handles, same digest).
    again = build(repair.refusals, host.files[PATH], host.mint, repair.rejected_proposal,
                  path=PATH, request_ordinal=2)
    assert again.digest() == repair.digest()
    assert again.handles == repair.handles


@pytest.mark.parametrize("area", AREAS, ids=[area.code.value for area in AREAS])
def test_fingerprints_ignore_handle_spelling_and_drive_the_ladder(host: Host, area: Area):
    repair, _accepted, _result = run_round_trip(host, area)
    # A fingerprint is sha256(code|path|old_text): recomputing from the proposal matches.
    # The path is the one the refused edit resolved to: the other file for the stray edit
    # of a mixed batch, UNRESOLVED_PATH for a handle no request minted, else this file.
    path = {RefusalCode.STALE_HANDLE: UNRESOLVED_PATH,
            RefusalCode.MIXED_FILE_BATCH: OTHER}.get(area.code, PATH)
    expected = {fingerprint(r.code, path, repair.rejected_proposal[r.edit_index].old_text)
                for r in repair.refusals}
    assert set(repair.fingerprints) == expected
    # Resending the identical mistake under q3 handles escalates: NARROW then BLOCKED.
    ledger: FingerprintLedger = repair.ledger
    second = build(repair.refusals, host.files[PATH], host.mint, repair.rejected_proposal,
                   path=PATH, request_ordinal=3, ledger=ledger)
    assert second.level is EscalationLevel.NARROW and second.fingerprints == repair.fingerprints
    third = build(repair.refusals, host.files[PATH], host.mint, repair.rejected_proposal,
                  path=PATH, request_ordinal=4, ledger=second.ledger)
    assert third.blocked and "[EDIT BLOCKED]" in third.observation()


# Codes that EditAuthority.bind_handle can also judge on its own (the rest are batch
# shape, span policy or whole-file results that only the batch validator sees).
HANDLE_LEVEL = {RefusalCode.STALE_HANDLE, RefusalCode.SOURCE_CHANGED,
                RefusalCode.EMPTY_OLD_TEXT, RefusalCode.OLD_NOT_FOUND,
                RefusalCode.UNCOVERED_SOURCE, RefusalCode.AMBIGUOUS_MATCH}


@pytest.mark.parametrize("area", [a for a in AREAS if a.code in HANDLE_LEVEL],
                         ids=[a.code.value for a in AREAS if a.code in HANDLE_LEVEL])
def test_authority_refuses_the_same_proposal_with_the_same_code_and_fingerprint(
        host: Host, area: Area):
    host.deliver(0, PATH)
    q1 = host.deliver(1, PATH, OTHER)
    proposal, spans = area.induce(host, q1)
    refused = host.validate(proposal, spans, PATH)
    assert refused.codes == (area.code,)
    authority = host.authority(PATH)
    authority.consume(0)
    handle, old_text, _new = proposal[0]
    record = next(r for m in host.maps.values() for r in m.records if r.handle == handle)
    with pytest.raises(rbe.HandleRefusal) as info:
        authority.bind_handle(record, old_text=old_text)
    assert info.value.code == area.code.value
    if area.code is RefusalCode.STALE_HANDLE:
        # Known divergence: the validator cannot resolve an unknown handle to a path and
        # fingerprints it under UNRESOLVED_PATH; the authority has the record and uses
        # its path. Same code, different fingerprint.
        assert info.value.refusal(edit_index=0).fingerprint != refused.refusals[0].fingerprint
        assert info.value.refusal(edit_index=0).code == refused.refusals[0].code
    else:
        assert info.value.refusal(edit_index=0) == refused.refusals[0]


def test_stale_handle_is_also_refused_once_the_request_was_consumed(host: Host):
    host.deliver(1, PATH)
    authority = host.authority(PATH)
    record = body(host.maps[1])
    authority.bind_handle(record, old_text="return 1")
    authority.consume(1)
    with pytest.raises(rbe.HandleRefusal) as info:
        authority.bind_handle(record, old_text="return 1")
    assert info.value.code == RefusalCode.STALE_HANDLE.value


def test_bom_and_crlf_round_trip_byte_exactly(host: Host, monkeypatch):
    monkeypatch.delitem(PROTECTED_LINES, PATH)
    source = "\ufeffdef f():\r\n    return 1\r\n"
    host.files[PATH] = source.encode("utf-8")
    q1 = host.deliver(1, PATH)
    authority = host.authority(PATH)
    proposal = [(q1.records[0].handle, "return 7", "return 1")]
    refused = host.validate(proposal, host.known_spans(1), PATH, parse_check=None)
    assert refused.codes == (RefusalCode.OLD_NOT_FOUND,)
    repair = build(refused.refusals, host.files[PATH], host.mint, proposal,
                   path=PATH, request_ordinal=2)
    assert repair.current_source == source
    assert repair.handle_map is not None
    corrected = [(repair.handles[0], "return 1\r\n", "return 2\r\n")]
    accepted = host.validate(corrected, spans_for_validator(repair.handle_map), PATH,
                             parse_check=None)
    assert accepted.ok and accepted.plan is not None, accepted.codes
    ref = authority.bind_handle(repair.handle_map.records[0], old_text="return 1\r\n")
    plan = rbe.EditPlan(base_revision=authority.base_revision, authority_ref=AUTHORITY_REF,
                        inventory_ref=INVENTORY_REF,
                        edits=(rbe.EditOperation("replace", ref, "return 2\r\n"),))
    expected = "\ufeffdef f():\r\n    return 2\r\n".encode("utf-8")
    assert rbe.compose_candidate(authority, plan)[PATH] == expected
    assert rbe.compose_candidate(authority, accepted.plan)[PATH] == expected
