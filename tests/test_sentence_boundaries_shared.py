"""Ticket sentence splitting keeps punctuation inside names and references."""
from __future__ import annotations

import pytest

from stateframe.engine import kernel_obligation_seed as seed
from stateframe.engine import kernel_requirement_alignment as alignment


_CASES = (
    ("Use v1.2 for output.", "v1.2"),
    ("Update file.py:12 now.", "file.py:12"),
    ("Use https://example.com/api?q=1 now.", "https://example.com/api?q=1"),
    ("Use https://example.com/./api now.", "https://example.com/./api"),
    ("Accept e.g. JSON only.", "e.g. JSON only"),
    ("Use formats incl. JSON only.", "incl. JSON only"),
    ("Use `file.py:12` now.", "`file.py:12`"),
)


@pytest.mark.parametrize("mode", ("off", "on"))
@pytest.mark.parametrize(("ticket", "required_text"), _CASES)
def test_seed_keeps_reference_in_one_exact_source_span(monkeypatch, mode, ticket, required_text):
    monkeypatch.setenv(seed.MODE_ENV, mode)
    monkeypatch.setenv(seed.LOSSLESS_MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", "off")

    obligations = seed.extract_obligations(ticket, source_id="sentence-test")

    assert len(obligations) == 1
    obligation = obligations[0]
    assert required_text in obligation.text
    assert len(obligation.source_spans) == 1
    span = obligation.source_spans[0]
    assert ticket.encode("utf-8")[span.start_byte:span.end_byte].decode("utf-8") == obligation.text


@pytest.mark.parametrize(("ticket", "required_text"), _CASES)
def test_alignment_keeps_reference_in_one_source_span(monkeypatch, ticket, required_text):
    monkeypatch.setenv(seed.LOSSLESS_MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")

    spans = alignment.extract_ticket_spans(ticket)

    assert len(spans) == 1
    start, end, sentence = spans[0]
    assert required_text in sentence
    assert ticket[start:end] == sentence == ticket


def test_alignment_keeps_closers_with_the_ending_sentence(monkeypatch):
    monkeypatch.setenv(seed.LOSSLESS_MODE_ENV, "on")
    text = "Return the value?!)]}'\" Next request."
    sentence_end = text.index(" Next request.")

    assert alignment._split_sentences(text) == ["Return the value?!)]}'\"", "Next request."]
    assert alignment.extract_ticket_spans(text) == (
        (0, sentence_end, text[:sentence_end]),
        (sentence_end + 1, len(text), "Next request."),
    )


def test_alignment_keeps_quoted_and_backticked_punctuation_inside_sentence(monkeypatch):
    monkeypatch.setenv(seed.LOSSLESS_MODE_ENV, "on")
    text = 'Keep `file.py:12` and "https://example.com/api?q=1" intact. Then stop.'

    assert alignment._split_sentences(text) == [
        'Keep `file.py:12` and "https://example.com/api?q=1" intact.',
        "Then stop.",
    ]


def test_abbreviation_matching_does_not_swallow_a_real_sentence_boundary(monkeypatch):
    monkeypatch.setenv(seed.LOSSLESS_MODE_ENV, "on")
    text = "Use streams. Reject stale rows."

    assert alignment._split_sentences(text) == ["Use streams.", "Reject stale rows."]
