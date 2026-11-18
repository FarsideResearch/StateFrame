"""A read-only call the host suppressed as a repeat never ran; its effect is settled, not unknown.

An unknown read-only effect has no repair for search tools, so continuation blocked the whole run
on the first suppressed grep.
"""
from __future__ import annotations

import pytest

from kernel_continuation_evidence import READ_RESULT_CLASSES, result_class_disposition


@pytest.mark.parametrize("tool", ["read", "text", "grep", "ls"])
def test_repeat_suppressed_read_only_call_is_settled_not_applied(tool):
    assert tool in READ_RESULT_CLASSES
    assert result_class_disposition(tool, "REPEAT_SUPPRESSED") == "settled_not_applied"


def test_unrecognised_read_result_stays_unknown():
    assert result_class_disposition("grep", "SOMETHING_NEW") == "unknown"


def test_admission_refusal_stays_scoped_to_read():
    assert result_class_disposition("grep", "REFUSED_ADMISSION") == "unknown"
