"""Resume equivalence: a run interrupted after turn k and resumed in a fresh process must send the same turn k+1 request."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from enginecheck import HarnessError, Need, Scenario
from enginecheck.resume import ResumeDivergence, assert_resume_equivalent, check_resume

FIRST = json.dumps({"model": "m", "messages": [{"role": "user", "content": "a"}]}).encode()
SECOND = json.dumps({"model": "m", "messages": [{"role": "user", "content": "a"}, {"role": "tool"}]}).encode()
THIRD = json.dumps({"model": "m", "messages": [{"role": "user", "content": "a"}, {"role": "tool"}, {}]}).encode()
UNINTERRUPTED = [FIRST, SECOND, THIRD]


def test_equal_post_resume_request_passes() -> None:
    assert check_resume(UNINTERRUPTED, [SECOND, THIRD], stop_after=1) == "byte-equal"


def test_json_equal_bytes_with_different_key_order_fail_at_byte_level() -> None:
    reordered = b'{"messages": [{"role": "user", "content": "a"}, {"role": "tool"}], "model": "m"}'
    assert json.loads(reordered) == json.loads(SECOND)
    assert reordered != SECOND
    with pytest.raises(ResumeDivergence, match="byte offset") as error:
        check_resume(UNINTERRUPTED, [reordered], stop_after=1)
    assert "JSON-equal but not byte-equal" in str(error.value)


def test_json_equal_bytes_with_different_whitespace_fail_at_byte_level() -> None:
    spaced = json.dumps(json.loads(SECOND), indent=2).encode()
    with pytest.raises(ResumeDivergence, match="byte offset"):
        check_resume(UNINTERRUPTED, [spaced], stop_after=1)


def test_masked_difference_reports_json_equal_after_mask() -> None:
    masked_twin = json.dumps({"model": "m", "messages": [{"role": "user", "content": "z"}, {"role": "tool"}]},
                             indent=1).encode()
    verdict = check_resume(UNINTERRUPTED, [masked_twin], stop_after=1, allowed_volatile=("/messages/0/content",))
    assert verdict == "json-equal-after-mask"


def test_path_difference_fails_naming_the_path() -> None:
    drifted = json.dumps({"model": "m", "messages": [{"role": "user", "content": "b"}, {"role": "tool"}]}).encode()
    with pytest.raises(ResumeDivergence) as error:
        check_resume(UNINTERRUPTED, [drifted], stop_after=1)
    assert "/messages/0/content" in str(error.value)
    assert '"a"' in str(error.value) and '"b"' in str(error.value)


def test_declared_volatile_path_is_masked() -> None:
    drifted = json.dumps({"model": "m", "messages": [{"role": "user", "content": "b"}, {"role": "tool"}]}).encode()
    check_resume(UNINTERRUPTED, [drifted], stop_after=1, allowed_volatile=("/messages/0/content",))


def test_undeclared_difference_still_fails_when_another_path_is_masked() -> None:
    drifted = json.dumps({"model": "x", "messages": [{"role": "user", "content": "b"}, {"role": "tool"}]}).encode()
    with pytest.raises(ResumeDivergence, match="/model"):
        check_resume(UNINTERRUPTED, [drifted], stop_after=1, allowed_volatile=("/messages/0/content",))


def test_no_post_resume_request_fails() -> None:
    with pytest.raises(ResumeDivergence, match="no provider request"):
        check_resume(UNINTERRUPTED, [], stop_after=1)


def test_non_json_difference_fails() -> None:
    with pytest.raises(ResumeDivergence, match="not JSON"):
        check_resume(UNINTERRUPTED, [b"\xff"], stop_after=1)


def test_volatile_pointer_missing_from_expected_request_is_rejected() -> None:
    drifted = json.dumps({"model": "x", "messages": []}).encode()
    with pytest.raises(ValueError, match="/nope"):
        check_resume(UNINTERRUPTED, [drifted], stop_after=1, allowed_volatile=("/nope",))


def test_volatile_pointer_must_be_json_pointer() -> None:
    with pytest.raises(ValueError, match="JSON pointer"):
        check_resume(UNINTERRUPTED, [SECOND], stop_after=1, allowed_volatile=("model",))


def test_too_short_uninterrupted_run_is_a_harness_error() -> None:
    with pytest.raises(HarnessError):
        check_resume([FIRST], [SECOND], stop_after=1)


def _three_turn_scenario() -> Scenario:
    """Two reads then done: three provider requests, with a named output file the engine can write."""
    return Scenario(
        files={
            "target.py": b"def add(a, b):\n    return a + b\n",
            "helper.py": b"VALUE = 2\n",
            "notes.md": b"# notes\n",
        },
        task="Read target.py and helper.py, then update notes.md.",
        steps=[
            {"tool": "read", "required": True, "arguments": {"path": "target.py"}},
            {"tool": "read", "required": True, "arguments": {"path": "helper.py"}},
            {"tool": "done", "required": True, "arguments": {"summary": "Read both files."}},
        ],
        needs=[Need("the engine sent three provider requests",
                    lambda trace: len(trace.requests) >= 3)],
    )


def test_real_engine_resume_after_turn_one_matches_uninterrupted(tmp_path: Path) -> None:
    assert assert_resume_equivalent(_three_turn_scenario(), tmp_path, stop_after=1) == "byte-equal"
