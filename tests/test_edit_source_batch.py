"""kernel_edit_source_batch: every refusal code reachable, all issues reported at once,
clean batches become replace-only EditPlans bound to the current bytes."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_refusal_codes import RefusalCode, fingerprint  # noqa: E402
from kernel_edit_source_batch import (  # noqa: E402
    MAX_EDITS, UNRESOLVED_PATH, BatchInputError, BatchValidation, CoveredSpan,
    compose, display_marker_in, validate_edit_source_batch,
)
from kernel_revision_bound_edit import EditPlan  # noqa: E402

PATH = "pkg/mod.py"
REV = "rev-7"
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
AUTH = dict(current_revision=REV, authority_ref="authority:test",
            inventory_ref="inventory:test", source_receipt="receipt:test")


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


def run(edits, spans, source=RAW, **kw) -> BatchValidation:
    params = {**AUTH, **kw}
    return validate_edit_source_batch(edits, spans, source, **params)


# ---------------------------------------------------------------- one case per code


def test_too_many_edits_reports_each_excess_edit_and_still_checks_the_first_four():
    edits = [("q1h1", f"return {n}", f"return {n}0") for n in (1, 2, 3)]
    edits += [("q1h1", "import os", "import sys"), ("q1h1", "nope", "x"), ("q1h1", "import os", "y")]
    result = run(edits, [whole_file()])
    assert MAX_EDITS == 4
    assert [(r.code, r.edit_index) for r in result.refusals] == [
        (RefusalCode.TOO_MANY_EDITS, 4), (RefusalCode.TOO_MANY_EDITS, 5)]
    assert result.plan is None


def test_mixed_file_batch_refuses_edits_whose_handle_names_another_file():
    other = CoveredSpan("q1h9", "pkg/other.py", REV, 0, 5)
    result = run([("q1h1", "return 1", "return 10"), ("q1h9", "x", "y")],
                 [alpha_span(), other])
    assert result.codes == (RefusalCode.MIXED_FILE_BATCH,)
    assert result.refusals[0].edit_index == 1
    assert result.refusals[0].fingerprint == fingerprint(
        RefusalCode.MIXED_FILE_BATCH, "pkg/other.py", "x")
    assert result.path == PATH


def test_stale_handle_when_the_handle_was_not_delivered_in_this_request():
    result = run([("q0h1", "return 1", "return 10")], [alpha_span()])
    assert result.codes == (RefusalCode.STALE_HANDLE,)
    assert result.refusals[0].fingerprint == fingerprint(
        RefusalCode.STALE_HANDLE, UNRESOLVED_PATH, "return 1")


def test_source_changed_on_revision_drift_digest_drift_or_shrunk_file():
    drift = run([("q1h1", "return 1", "return 10")], [alpha_span()], current_revision="rev-8")
    assert drift.codes == (RefusalCode.SOURCE_CHANGED,)
    pinned = whole_file(source_digest="sha256:" + hashlib.sha256(b"other").hexdigest())
    digest = run([("q1h1", "return 1", "return 10")], [pinned])
    assert digest.codes == (RefusalCode.SOURCE_CHANGED,)
    shrunk = run([("q1h1", "return 1", "return 10")], [whole_file()], source=RAW[:40])
    assert shrunk.codes == (RefusalCode.SOURCE_CHANGED,)


def test_not_editable_span_is_refused_before_matching():
    result = run([("q1h1", "return 1", "return 10")], [alpha_span().__class__(
        "q1h1", PATH, REV, 11, 37, editable=False)])
    assert result.codes == (RefusalCode.NOT_EDITABLE,)


def test_empty_old_text_is_refused():
    result = run([("q1h1", "", "inserted\n")], [whole_file()])
    assert result.codes == (RefusalCode.EMPTY_OLD_TEXT,)


@pytest.mark.parametrize("old_text", [
    "[CURRENT SOURCE]\nreturn 1",
    "def alpha():\n[... middle omitted ...]\n    return 1",
    "[Paged source: pkg/mod.py]",
    "Real04 source window omitted middle lines",
])
def test_display_marker_text_in_old_text_is_refused(old_text):
    result = run([("q1h1", old_text, "x")], [whole_file()])
    assert result.codes == (RefusalCode.DISPLAY_MARKER,)


def test_marker_literal_that_really_is_file_content_is_not_a_display_marker():
    source = b"TEXT = '[CURRENT SOURCE]'\n"
    assert not display_marker_in("'[CURRENT SOURCE]'", source)
    result = run([("q1h1", "'[CURRENT SOURCE]'", "'x'")],
                 [CoveredSpan("q1h1", PATH, REV, 0, len(source))], source=source)
    assert result.ok


def test_old_not_found_when_text_is_nowhere_in_the_file():
    result = run([("q1h1", "return 99", "return 1")], [alpha_span()])
    assert result.codes == (RefusalCode.OLD_NOT_FOUND,)


def test_uncovered_source_when_text_exists_only_outside_the_handle_span():
    result = run([("q1h1", "return 2", "return 20")], [alpha_span()])
    assert result.codes == (RefusalCode.UNCOVERED_SOURCE,)


def test_ambiguous_match_inside_the_span_only():
    # "return" occurs once inside alpha's span, three times in the file: span scoping
    # makes it exact; a whole-file handle makes it ambiguous.
    scoped = run([("q1h1", "return", "yield")], [alpha_span()])
    assert scoped.ok
    whole = run([("q1h1", "return", "yield")], [whole_file()])
    assert whole.codes == (RefusalCode.AMBIGUOUS_MATCH,)


def test_overlapping_edits_reports_both_edits_and_adjacent_ranges_are_allowed():
    overlap = run([("q1h1", "def alpha():\n    return 1", "def alpha():\n    return 10"),
                   ("q1h1", "return 1\n", "return 11\n")], [whole_file()])
    assert overlap.codes == (RefusalCode.OVERLAPPING_EDITS, RefusalCode.OVERLAPPING_EDITS)
    assert [r.edit_index for r in overlap.refusals] == [0, 1]
    adjacent = run([("q1h1", "def alpha():\n", "def alpha_():\n"),
                    ("q1h1", "    return 1\n", "    return 10\n")], [whole_file()])
    assert adjacent.ok
    assert len(adjacent.plan.edits) == 2


def test_parse_break_attributes_to_the_edit_that_breaks_the_file_alone():
    result = run([("q1h1", "return 1", "return 10"), ("q1h2", "return 2", "return (")],
                 [alpha_span(), beta_span()], parse_check=py_parse)
    assert [(r.code, r.edit_index) for r in result.refusals] == [(RefusalCode.PARSE_BREAK, 1)]


def test_parse_break_from_the_combination_reports_every_edit():
    # Each edit alone parses; together they leave an unmatched bracket.
    result = run([("q1h1", "return 1", "return [1"), ("q1h2", "return 2", "return 2]")],
                 [alpha_span(), beta_span()], parse_check=py_parse)
    assert result.codes == (RefusalCode.PARSE_BREAK, RefusalCode.PARSE_BREAK)


def test_every_catalog_code_is_reachable_from_this_module():
    reachable = {
        RefusalCode.TOO_MANY_EDITS, RefusalCode.MIXED_FILE_BATCH, RefusalCode.STALE_HANDLE,
        RefusalCode.SOURCE_CHANGED, RefusalCode.NOT_EDITABLE, RefusalCode.EMPTY_OLD_TEXT,
        RefusalCode.DISPLAY_MARKER, RefusalCode.OLD_NOT_FOUND, RefusalCode.UNCOVERED_SOURCE,
        RefusalCode.AMBIGUOUS_MATCH, RefusalCode.OVERLAPPING_EDITS, RefusalCode.PARSE_BREAK,
    }
    # EDIT_CONFLICT is produced by the COLLAB-DRIFT resolver (kernel_edit_drift), never by the validator.
    assert reachable | {RefusalCode.EDIT_CONFLICT} == set(RefusalCode)
    text = (ENGINE / "kernel_edit_source_batch.py").read_text(encoding="utf-8")
    for code in reachable:
        assert f"RefusalCode.{code.value}" in text
    drift = (ENGINE / "kernel_edit_drift.py").read_text(encoding="utf-8")
    assert "RefusalCode.EDIT_CONFLICT" in drift


# ---------------------------------------------------------------- all issues at once


def test_all_issues_are_reported_together_sorted_by_edit_index():
    edits = [
        ("q1h1", "", "x"),                       # EMPTY_OLD_TEXT
        ("q1h2", "return 9", "x"),               # OLD_NOT_FOUND
        ("q9h9", "return 1", "x"),               # STALE_HANDLE
        ("q1h1", "[WORKING SOURCE]", "x"),       # DISPLAY_MARKER
        ("q1h2", "return 2", "x"),               # TOO_MANY_EDITS
    ]
    result = run(edits, [alpha_span(), beta_span()])
    assert [(r.code, r.edit_index) for r in result.refusals] == [
        (RefusalCode.EMPTY_OLD_TEXT, 0), (RefusalCode.OLD_NOT_FOUND, 1),
        (RefusalCode.STALE_HANDLE, 2), (RefusalCode.DISPLAY_MARKER, 3),
        (RefusalCode.TOO_MANY_EDITS, 4)]
    assert result.plan is None and not result.ok
    payload = result.as_dict()
    assert payload["ok"] is False and len(payload["refusals"]) == 5 and payload["plan"] is None


def test_refusal_fingerprints_do_not_depend_on_handle_spelling():
    first = run([("q1h1", "return 9", "x")], [alpha_span("q1h1")])
    second = run([("q2h7", "return 9", "x")], [alpha_span("q2h7")])
    assert first.refusals[0].fingerprint == second.refusals[0].fingerprint
    assert "q1h1" not in first.refusals[0].public_text


# ---------------------------------------------------------------- clean batches


def test_insert_keeps_anchor():
    anchor = "    return 1\n"
    result = run([("q1h1", anchor, anchor + "\n\ndef alpha_two():\n    return 11\n")],
                 [alpha_span()], parse_check=py_parse)
    assert result.ok
    op = result.plan.edits[0]
    assert op.operation == "replace"
    assert op.new_text.startswith(anchor)
    assert (op.site_ref.start, op.site_ref.end) == (24, 37)
    assert RAW[op.site_ref.start:op.site_ref.end] == anchor.encode("utf-8")
    composed = compose(RAW, ())
    assert composed == RAW


def test_four_non_overlapping_edits_build_one_replace_only_plan():
    edits = [
        ("q1h1", "return 1", "return 10"),
        ("q1h2", "return 2", "return 20"),
        ("q1h3", "return 3", "return 30"),
        ("q1h4", "import os", "import os\nimport sys"),
    ]
    spans = [alpha_span(), beta_span(), CoveredSpan("q1h3", PATH, REV, 64, 90),
             CoveredSpan("q1h4", PATH, REV, 0, 10)]
    result = run(edits, spans, parse_check=py_parse)
    assert result.ok and result.path == PATH
    plan = result.plan
    assert isinstance(plan, EditPlan)
    assert plan.base_revision == REV
    assert plan.authority_ref == "authority:test" and plan.inventory_ref == "inventory:test"
    assert plan.publish is False
    assert [op.operation for op in plan.edits] == ["replace"] * 4
    assert [op.order for op in plan.edits] == [0, 1, 2, 3]
    for op in plan.edits:
        ref = op.site_ref
        assert ref.path == PATH and ref.file_revision == REV
        assert ref.permitted_operations == ("replace",)
        assert ref.source_digest == "sha256:" + hashlib.sha256(RAW).hexdigest()
        preimage = RAW[ref.start:ref.end]
        assert ref.preimage_digest == "sha256:" + hashlib.sha256(preimage).hexdigest()
    # Round trip through the plan schema so the executor accepts it unchanged.
    assert EditPlan.from_mapping(plan.to_dict()) == plan


def test_byte_offsets_survive_bom_crlf_and_non_ascii():
    source = "﻿name = 'héllo'\r\nvalue = 2\r\n".encode("utf-8")
    span = CoveredSpan("q1h1", PATH, REV, 0, len(source))
    result = run([("q1h1", "'héllo'", "'wörld'"), ("q1h1", "value = 2", "value = 3")],
                 [span], source=source)
    assert result.ok
    first, second = result.plan.edits
    assert source[first.site_ref.start:first.site_ref.end] == "'héllo'".encode("utf-8")
    assert source[second.site_ref.start:second.site_ref.end] == b"value = 2"
    # Composing the plan's ranges keeps BOM and CRLF byte-for-byte.
    from kernel_edit_source_batch import _Match
    composed = compose(source, [
        _Match(0, first.site_ref.start, first.site_ref.end, "'wörld'".encode("utf-8"), ""),
        _Match(1, second.site_ref.start, second.site_ref.end, b"value = 3", "")])
    assert composed == "﻿name = 'wörld'\r\nvalue = 3\r\n".encode("utf-8")


def test_mapping_edits_and_rewrite_kind_spans_are_accepted():
    result = run([{"handle": "q1h1", "old_text": "return 1", "new_text": "return 10"}],
                 [whole_file(kind="REWRITE")])
    assert result.ok


# ---------------------------------------------------------------- input contract


@pytest.mark.parametrize("bad", [[], "not-a-list", [("q1h1", "a")], [("q1h1", None, "b")],
                                 [{"handle": "", "old_text": "a", "new_text": "b"}]])
def test_malformed_batches_raise_rather_than_refuse(bad):
    with pytest.raises(BatchInputError):
        run(bad, [whole_file()])


def test_malformed_spans_raise():
    with pytest.raises(BatchInputError):
        CoveredSpan("q1h1", PATH, REV, 5, 4)
    with pytest.raises(BatchInputError):
        CoveredSpan("q1h1", PATH, REV, 0, 4, kind="WINDOW")
    with pytest.raises(BatchInputError):
        run([("q1h1", "return 1", "x")], [whole_file(), whole_file()])


def test_module_is_pure_and_only_tests_import_it():
    text = (ENGINE / "kernel_edit_source_batch.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not imported & {"os", "sys", "io", "pathlib", "time", "subprocess"}
    assert "os.environ" not in text and "open(" not in text
    for py in ENGINE.rglob("*.py"):
        # Consumers: the edit_source executor and the legacy_safe single-edit route.
        if py.name not in {"kernel_edit_source_batch.py", "kernel_edit_source_tool.py",
                           "kernel_legacy_safe_edit.py", "kernel_edit_drift.py"}:
            assert "kernel_edit_source_batch" not in py.read_text(encoding="utf-8"), py
