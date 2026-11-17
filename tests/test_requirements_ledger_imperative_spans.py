"""Unclassified instruction spans must surface as work; coverage must check real spans.

Regression guard for the requirements ledger: a plain requested action such as
"Sort results by timestamp." used to fall through to context_only and produce no
requirement at all.  coverage_complete() also only validated the schema.
"""
import copy

import pytest

import kernel_requirements_ledger as ledger


def _build(text: str) -> dict:
    ledger.clear_cache()
    return ledger.build_ledger("ticket-r05", text)


def _span_id_for(built: dict, quote: str) -> str:
    matches = [row["id"] for row in built["spans"] if row["quote"] == quote]
    assert len(matches) == 1, f"expected one span for {quote!r}, got {matches}"
    return matches[0]


def _coverage_for(built: dict, span_id: str) -> dict:
    rows = [row for row in built["coverage_dispositions"] if row["span_id"] == span_id]
    assert len(rows) == 1, f"expected exactly one coverage row for {span_id}, got {len(rows)}"
    return rows[0]


@pytest.mark.parametrize("sentence", [
    "Sort results by timestamp.",
    "Encrypt every stored token.",
    "Implement cursor pagination.",
    "Fix the pagination bug.",
    "Delete expired entries.",
])
def test_plain_requested_work_becomes_a_requirement(sentence):
    built = _build(sentence)

    span_id = _span_id_for(built, sentence)
    coverage = _coverage_for(built, span_id)
    assert coverage["disposition"] != "context_only", coverage
    assert coverage["requirement_refs"], coverage

    matching = [req for req in built["requirements"] if sentence in req["exact_clause"]]
    assert matching, "no requirement carries the instruction sentence"
    assert matching[0]["kind"] == "unresolved"
    assert matching[0]["source_span_ids"] == [span_id]


def test_unclassified_span_reason_names_host_owner():
    built = _build("Sort results by timestamp.")

    coverage = _coverage_for(built, _span_id_for(built, "Sort results by timestamp."))
    assert coverage["disposition"] == "unresolved"
    assert "Unclassified instruction span" in coverage["reason"]
    assert "owner: host" in coverage["reason"]


def test_focus_label_with_prose_becomes_a_requirement():
    text = "Focus: Reject stale lease tokens"
    built = _build(text)

    coverage = _coverage_for(built, _span_id_for(built, text))
    assert coverage["disposition"] != "context_only", coverage
    assert coverage["requirement_refs"], coverage


def test_title_label_stays_advisory_context_only():
    text = "Title: Lease cleanup"
    built = _build(text)

    coverage = _coverage_for(built, _span_id_for(built, text))
    assert coverage["disposition"] == "context_only"
    assert coverage["requirement_refs"] == []
    assert not [req for req in built["requirements"] if text in req["exact_clause"]]


@pytest.mark.parametrize("fence_or_rule", ["---", "```"])
def test_punctuation_only_lines_stay_context_only(fence_or_rule):
    built = _build(fence_or_rule)

    coverage = _coverage_for(built, _span_id_for(built, fence_or_rule))
    assert coverage["disposition"] == "context_only"
    assert coverage["requirement_refs"] == []


def test_policy_never_blocks_completion_on_unresolved_work():
    built = _build("Sort results by timestamp.\nEncrypt every stored token.\n")

    assert built["policy"]["unknown_blocks_completion"] is False
    assert any(req["kind"] == "unresolved" for req in built["requirements"])


def test_coverage_complete_true_for_fresh_build():
    built = _build("Sort results by timestamp.\nTitle: Lease cleanup\n---\n")

    assert ledger.coverage_complete(built) is True


def test_coverage_complete_false_when_a_span_row_is_uncovered():
    built = _build("Sort results by timestamp.\nTitle: Lease cleanup\n")
    mutated = copy.deepcopy(built)
    victim = _span_id_for(mutated, "Title: Lease cleanup")
    mutated["coverage_dispositions"] = [
        row for row in mutated["coverage_dispositions"] if row["span_id"] != victim
    ]

    assert ledger.coverage_complete(mutated) is False


def test_coverage_complete_false_when_a_requirement_ref_names_a_missing_id():
    built = _build("Sort results by timestamp.\n")
    mutated = copy.deepcopy(built)
    row = next(row for row in mutated["coverage_dispositions"] if row["requirement_refs"])
    row["requirement_refs"] = [{"id": "R-MISSING", "version": 1}]

    assert ledger.coverage_complete(mutated) is False


def test_coverage_complete_false_when_requirement_names_a_missing_span():
    built = _build("Sort results by timestamp.\n")
    mutated = copy.deepcopy(built)
    mutated["requirements"][0]["source_span_ids"] = ["S-MISSING"]

    assert ledger.coverage_complete(mutated) is False


def test_coverage_complete_false_for_mapped_row_without_refs():
    built = _build("Sort results by timestamp.\n")
    mutated = copy.deepcopy(built)
    row = next(row for row in mutated["coverage_dispositions"] if row["requirement_refs"])
    row["requirement_refs"] = []

    assert ledger.coverage_complete(mutated) is False


def test_coverage_complete_never_raises_on_garbage():
    assert ledger.coverage_complete({}) is False
    assert ledger.coverage_complete({"spans": None, "coverage_dispositions": 3}) is False
