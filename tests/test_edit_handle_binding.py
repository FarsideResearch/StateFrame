"""EditAuthority handle binding: request-scoped handles become SiteReferences, are single
use per request, refuse with closed codes, and compose byte-exactly. The legacy admit
diagnostics stay byte-identical when the handle path is not used."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import kernel_revision_bound_edit as rbe  # noqa: E402
from kernel_edit_refusal_codes import HANDLE_PATTERN, Refusal, RefusalCode  # noqa: E402
from kernel_paged_source import source_revision  # noqa: E402

SOURCE = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
PATH = "pkg/mod.py"


def authority(tmp_path: Path, sources: dict[str, bytes] | None = None) -> rbe.EditAuthority:
    sources = sources or {PATH: SOURCE.encode()}
    for path, data in sources.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return rbe.EditAuthority(root=tmp_path, sources=sources, base_revision="base:1",
                             authority_ref="authority:test")


def record(handle: str = "q7h1", *, start: int = 0, end: int | None = None,
           revision: str | None = None, kind: str = "SPAN", path: str = PATH,
           source: str = SOURCE) -> rbe.HandleRecord:
    return rbe.HandleRecord(handle=handle, path=path,
                            revision=revision or source_revision(source), start=start,
                            end=len(source.encode()) if end is None else end, kind=kind)


def plan(authority: rbe.EditAuthority, *edits: tuple[rbe.SiteReference, str]) -> rbe.EditPlan:
    return rbe.EditPlan(base_revision=authority.base_revision, authority_ref=authority.authority_ref,
                        inventory_ref="inventory:test",
                        edits=tuple(rbe.EditOperation("replace", ref, new) for ref, new in edits))


def test_bind_handle_narrows_to_old_text_inside_span(tmp_path):
    auth = authority(tmp_path)
    ref = auth.bind_handle(record(), old_text="return 1")
    assert isinstance(ref, rbe.SiteReference)
    assert SOURCE.encode()[ref.start:ref.end] == b"return 1"
    assert ref.source_receipt == "q7h1" and ref.permitted_operations == ("replace",)
    assert ref.file_revision == source_revision(SOURCE)
    assert ref.site_id in auth.sites
    assert auth.admit(plan(auth, (ref, "return 10"))).accepted


def test_bind_handle_accepts_mapping_records_and_rewrite_kind(tmp_path):
    auth = authority(tmp_path)
    mapping = {"handle": "q3h2", "path": PATH, "revision": source_revision(SOURCE),
               "start": 0, "end": 24, "kind": "REWRITE"}
    ref = auth.bind_handle(mapping)
    assert (ref.start, ref.end) == (0, 24)
    with pytest.raises(rbe.HandleRefusal) as info:
        auth.bind_handle(mapping, old_text="return 1")
    assert info.value.code == "UNCOVERED_SOURCE"


def test_stale_handle_after_consume_and_for_earlier_requests(tmp_path):
    auth = authority(tmp_path)
    ref = auth.bind_handle(record("q7h1"), old_text="return 1")
    assert auth.consume(7) == (ref.site_id,)
    assert ref.site_id not in auth.sites
    with pytest.raises(rbe.HandleRefusal) as info:
        auth.bind_handle(record("q7h1"), old_text="return 1")
    assert info.value.code == "STALE_HANDLE"
    # An earlier request's handle is stale too; a later one is still live.
    with pytest.raises(rbe.HandleRefusal) as earlier:
        auth.bind_handle(record("q6h4"), old_text="return 2")
    assert earlier.value.code == "STALE_HANDLE"
    assert auth.bind_handle(record("q8h1"), old_text="return 2").path == PATH
    assert auth.consume(7) == ()  # idempotent


def test_source_changed_when_revision_moves(tmp_path):
    auth = authority(tmp_path)
    with pytest.raises(rbe.HandleRefusal) as info:
        auth.bind_handle(record(revision=source_revision(SOURCE + "# tail\n")), old_text="return 1")
    assert info.value.code == "SOURCE_CHANGED"
    refusal = info.value.refusal(edit_index=2)
    assert isinstance(refusal, Refusal) and refusal.code is RefusalCode.SOURCE_CHANGED
    assert refusal.edit_index == 2 and HANDLE_PATTERN.search(refusal.public_text) is None
    # Fingerprint ignores the handle spelling.
    with pytest.raises(rbe.HandleRefusal) as again:
        auth.bind_handle(record("q9h3", revision=source_revision(SOURCE + "# tail\n")),
                         old_text="return 1")
    assert again.value.refusal().fingerprint == refusal.fingerprint


def test_uncovered_source_outside_span_and_outside_file(tmp_path):
    auth = authority(tmp_path)
    with pytest.raises(rbe.HandleRefusal) as outside:
        auth.bind_handle(record(start=0, end=24), old_text="return 2")
    assert outside.value.code == "UNCOVERED_SOURCE"
    with pytest.raises(rbe.HandleRefusal) as beyond:
        auth.bind_handle(record(start=0, end=len(SOURCE) + 5), old_text="return 1")
    assert beyond.value.code == "UNCOVERED_SOURCE"


@pytest.mark.parametrize("old_text,code", [
    ("return 3", "OLD_NOT_FOUND"),
    ("return", "AMBIGUOUS_MATCH"),
    ("", "EMPTY_OLD_TEXT"),
])
def test_old_text_defects_map_to_closed_codes(tmp_path, old_text, code):
    auth = authority(tmp_path)
    with pytest.raises(rbe.HandleRefusal) as info:
        auth.bind_handle(record(), old_text=old_text)
    assert info.value.code == code
    assert RefusalCode(code)


def test_not_editable_outside_authority_and_malformed_handle(tmp_path):
    auth = authority(tmp_path)
    with pytest.raises(rbe.HandleRefusal) as info:
        auth.bind_handle(record(path="other/file.py"), old_text="x")
    assert info.value.code == "NOT_EDITABLE"
    with pytest.raises(rbe.EditPlanError) as bad:
        auth.bind_handle({"handle": "site:abc", "path": PATH, "revision": "sha256:0",
                          "start": 0, "end": 1})
    assert bad.value.code == "STALE_HANDLE"


def test_compose_preserves_bom_crlf_and_untouched_bytes(tmp_path):
    source = "﻿def f():\r\n    return 1\r\n\r\n\r\ndef g():\r\n    return 2\r\n"
    raw = source.encode("utf-8")
    auth = authority(tmp_path, {PATH: raw})
    ref = auth.bind_handle(record(source=source), old_text="return 1")
    out = rbe.compose_candidate(auth, plan(auth, (ref, "return 100")))[PATH]
    assert out.startswith(b"\xef\xbb\xbf")
    assert out == raw.replace(b"return 1\r\n", b"return 100\r\n", 1)
    assert b"\n" not in out.replace(b"\r\n", b"")


def test_handle_refusals_translate_admit_diagnostics(tmp_path):
    auth = authority(tmp_path)
    one = auth.bind_handle(record(), old_text="return 1")
    two = auth.bind_handle(record("q7h2"), old_text="def f():\n    return 1")
    edits = plan(auth, (one, "return 9"), (two, "def f():\n    return 8"))
    refusals = auth.handle_refusals(edits)
    assert [r.code for r in refusals] == [RefusalCode.OVERLAPPING_EDITS]
    # A changed file after delivery -> SOURCE_CHANGED (STALE_SOURCE and STALE_REVISION fold).
    changed = {PATH: (SOURCE + "# tail\n").encode()}
    refusals = auth.handle_refusals(plan(auth, (one, "return 9")), changed)
    assert [r.code for r in refusals] == [RefusalCode.SOURCE_CHANGED]
    assert refusals[0].edit_index == 0
    assert refusals[0].fingerprint == Refusal.build(
        RefusalCode.SOURCE_CHANGED, edit_index=0, path=PATH, old_text="return 1").fingerprint
    # A clean plan has no refusals and admit still accepts.
    assert auth.handle_refusals(plan(auth, (one, "return 9"))) == ()


@pytest.mark.parametrize("diagnostic,expected", [
    ("STALE_SOURCE: pkg/mod.py no longer matches site:abc[1]", ("SOURCE_CHANGED", 1)),
    ("STALE_REVISION: file revision changed for pkg/mod.py[0]", ("SOURCE_CHANGED", 0)),
    ("STALE_SPAN: preimage changed for site:abc[3]", ("UNCOVERED_SOURCE", 3)),
    ("OVERLAP: operations 0 and 1 overlap in pkg/mod.py", ("OVERLAPPING_EDITS", None)),
    ("MATCH[2]: legacy old_str must match exactly once", ("OLD_NOT_FOUND", 2)),
    ("OMISSION_MARKER[1]: ...", ("DISPLAY_MARKER", 1)),
    ("AMBIGUOUS[0]: site site:abc was selected twice", ("AMBIGUOUS_MATCH", 0)),
    ("OUT_OF_AUTHORITY: path is outside edit authority: x", ("NOT_EDITABLE", None)),
    ("garbage", ("NOT_EDITABLE", None)),
])
def test_refusal_code_for_diagnostic(diagnostic, expected):
    assert rbe.refusal_code_for_diagnostic(diagnostic) == expected


def test_legacy_admit_diagnostics_unchanged_when_handles_unused(tmp_path):
    auth = authority(tmp_path)
    ref = auth.bind_site(path=PATH, start=4, end=5, node_kind="name", source_receipt="r",
                         permitted_operations=["replace"])
    stale = {PATH: SOURCE.replace("def f", "def h").encode()}
    result = auth.admit(plan(auth, (ref, "k")), stale)
    assert not result.accepted
    assert result.diagnostics == (
        f"STALE_SOURCE: {PATH} no longer matches {ref.site_id}[0]",
        f"STALE_REVISION: file revision changed for {PATH}[0]",
        f"STALE_SPAN: preimage changed for {ref.site_id}[0]",
    )
    assert auth._handle_sites == {} and auth.consume(0) == ()


def test_minted_handle_record_binds_without_reshaping(tmp_path: Path) -> None:
    """A record straight from the handle minter (enum kind, nested byte span) binds."""
    import kernel_edit_handles as handles

    span = handles.DeliveredSpan.from_source(PATH, SOURCE)
    minted = handles.mint_handles(7, [span]).records[0]
    coerced = rbe.HandleRecord.coerce(minted)
    assert (coerced.kind, coerced.start, coerced.end) == ("SPAN", span.byte_start, span.byte_end)
    assert rbe.HandleRecord.coerce(minted.as_dict()) == coerced
    ref = authority(tmp_path).bind_handle(minted)
    assert isinstance(ref, rbe.SiteReference)
