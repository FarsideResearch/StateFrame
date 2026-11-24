"""Work-count regression checks for validity value parsing."""

import kernel_requirements_ledger as ledger


_VALUE_PATTERNS = {
    r"(?:only|values?\s+(?:are|include))\s+(.+?)\s+(?:are|is)\s+valid",
    r"(.+?)\s+(?:are|is)\s+valid",
}


def test_values_after_valid_skips_wildcard_scans_without_validity_phrase(monkeypatch):
    text = "x" + " " * 99_999
    original_search = ledger.re.search
    subject_searches = []

    def count_subject_searches(pattern, subject, *args, **kwargs):
        if subject is text:
            subject_searches.append((pattern, len(subject)))
            if pattern in _VALUE_PATTERNS:
                return None
        return original_search(pattern, subject, *args, **kwargs)

    monkeypatch.setattr(ledger.re, "search", count_subject_searches)

    assert ledger._values_after_valid(text) == []
    assert subject_searches == [(r"\s(?:are|is)\s+valid", len(text))]


def test_values_after_valid_keeps_quoted_values_without_validity_phrase():
    assert ledger._values_after_valid("'alpha'") == ["alpha"]
