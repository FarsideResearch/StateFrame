"""Fail-loud fixes in the test-generation oracle (test_gen_oracle.py), default path.

Three failure paths used to be swallowed with no raise, log, or typed record:
- the base binding fallback in `synthesize_cases`,
- a source that fails to load in `_load` (and the base load in `_execute_matrix`),
- the direction check in `predicate_symptom_direction`, which dropped a candidate on error.
"""
from __future__ import annotations

import logging

import pytest

MODULE_STEM = "test_gen_oracle"

_CANDIDATES = [
    (1, "    if score > 5:", "    if score < 5:"),
    (1, "    if score > 5:", "    if score >= 5:"),
]
_FN_SOURCE = (
    "def grade(score):\n"
    "    if score > 5:\n"
    "        return 'pass'\n"
    "    return 'fail'\n"
)


def _loud(caplog: pytest.LogCaptureFixture, module_name: str) -> bool:
    """True when the module emitted a WARNING-or-higher record during the call."""
    return any(record.levelno >= logging.WARNING and module_name in record.name
               for record in caplog.records)


def _engine_module():
    import test_gen_oracle as tgo
    return tgo


def test_base_binding_failure_is_loud(monkeypatch: pytest.MonkeyPatch,
                                      caplog: pytest.LogCaptureFixture) -> None:
    tgo = _engine_module()
    import intent_oracle as io

    def refusing_binding(fn):
        raise ValueError("forced: cannot bind the signature")

    monkeypatch.setattr(io, "_base_binding", refusing_binding)
    caplog.set_level(logging.WARNING)
    tgo.synthesize_cases(_FN_SOURCE, _CANDIDATES, "cmp_flip")
    assert _loud(caplog, MODULE_STEM), (
        "base binding failed but the probes were built from fallback defaults silently")


def test_unloadable_source_is_loud() -> None:
    tgo = _engine_module()
    source = "x = 1 / 0\ndef grade(score):\n    return score\n"
    with pytest.raises(tgo.SourceLoadError, match="ZeroDivisionError"):
        tgo._load(source, "grade")


def test_direction_check_error_does_not_eliminate_candidate(monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    import ast
    tgo = _engine_module()
    real = tgo.wr.reverses_direction

    def flaky(old_node, new_node):
        if "flaky_x" in ast.unparse(old_node):
            raise ValueError("forced: direction analysis failed")
        return real(old_node, new_node)

    monkeypatch.setattr(tgo.wr, "reverses_direction", flaky)
    candidates = [
        (10, "    if score > 5:", "    if score < 5:"),
        (11, "    if flaky_x > 5:", "    if flaky_x < 5:"),
        (12, "    if score > 5:", "    if score > 6:"),
    ]
    caplog.set_level(logging.WARNING)
    ok, _why = tgo.predicate_symptom_direction("cmp_flip", candidates, {0, 1, 2})
    assert ok is not None and 1 in ok, (
        "a candidate whose direction check raised was dropped from the survivor set")
    assert _loud(caplog, MODULE_STEM)


# --- regressions: the typed record each fix leaves behind --------------------------------------------


def test_base_binding_failure_is_recorded_in_result_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback still builds cases, and the entry point carries a typed note naming the cause."""
    tgo = _engine_module()
    import intent_oracle as io

    clean = tgo.synthesize_discriminating_test(_FN_SOURCE, _CANDIDATES, "cmp_flip")
    assert clean.notes == []

    def refusing_binding(fn):
        raise ValueError("forced: cannot bind the signature")

    monkeypatch.setattr(io, "_base_binding", refusing_binding)
    notes: list[str] = []
    cases = tgo.synthesize_cases(_FN_SOURCE, _CANDIDATES, "cmp_flip", notes=notes)
    assert cases, "the default-value fallback must still produce probes"
    assert notes == ["base binding unavailable (ValueError); probes use default values"]
    degraded = tgo.synthesize_discriminating_test(_FN_SOURCE, _CANDIDATES, "cmp_flip")
    assert degraded.notes == notes


def test_unloadable_base_refuses_to_discriminate() -> None:
    """An original that cannot load must not make every candidate look discriminating."""
    tgo = _engine_module()
    base = (
        "def grade(score, k=1 / 0):\n"
        "    if score > 5:\n"
        "        return 'pass'\n"
        "    return 'fail'\n"
    )
    # Candidates rewrite the failing default, so they load while the original does not.
    candidates = [
        (0, "def grade(score, k=1 / 0):", "def grade(score, k=1):"),
        (0, "def grade(score, k=1 / 0):", "def grade(score, k=2):"),
    ]
    with pytest.raises(tgo.SourceLoadError) as info:
        tgo._load(base, "grade")
    assert info.value.reason == "ZeroDivisionError while loading"
    with pytest.raises(tgo.SourceLoadError):
        tgo._load(_FN_SOURCE, "missing_name")
    assert callable(tgo._load(_FN_SOURCE, "grade"))

    cases = tgo.synthesize_cases(base, candidates, None)
    with pytest.raises(tgo.SourceLoadError):
        tgo._execute_matrix(base, "grade", candidates, base.splitlines(), cases)

    result = tgo.synthesize_discriminating_test(base, candidates)
    assert result.verdict == tgo.NO_SIGNAL
    assert result.reason == "original function did not load: ZeroDivisionError while loading"
    assert result.survivors == candidates
    assert result.discriminating == []


def test_unloadable_candidate_still_sits_out() -> None:
    """A candidate that cannot load keeps the old contract: None in the matrix, neither killed nor credited."""
    tgo = _engine_module()
    candidates = list(_CANDIDATES) + [(1, "    if score > 5:", "    if score > (1 / 0 if False else 5:")]
    cases = tgo.synthesize_cases(_FN_SOURCE, candidates, "cmp_flip")
    base_fps, _bc, cand_fps, _cc = tgo._execute_matrix(
        _FN_SOURCE, "grade", candidates, _FN_SOURCE.splitlines(), cases)
    assert len(base_fps) == len(cases)
    assert cand_fps[0] is not None and cand_fps[1] is not None
    assert cand_fps[2] is None


def test_direction_check_keeps_unreadable_candidates(monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    """Fail-open per candidate: a non-condition line and a raising check both stay, with the reason recorded."""
    import ast
    tgo = _engine_module()
    candidates = [
        (10, "    if score > 5:", "    if score < 5:"),          # reverses
        (11, "    for i in range(n):", "    for i in range(n + 1):"),  # no condition to read
        (12, "    if score > 5:", "    if score > 6:"),          # operand rewrite
    ]
    ok, why = tgo.predicate_symptom_direction("cmp_flip", candidates, {0, 1, 2})
    assert ok == {0, 1}
    assert why == ("reverses the comparison direction the symptom names; "
                   "kept 1 candidate(s) whose direction could not be read")

    real = tgo.wr.reverses_direction

    def flaky(old_node, new_node):
        if "flaky_x" in ast.unparse(old_node):
            raise ValueError("forced: direction analysis failed")
        return real(old_node, new_node)

    monkeypatch.setattr(tgo.wr, "reverses_direction", flaky)
    caplog.set_level(logging.WARNING)
    flaky_cands = [candidates[0], (11, "    if flaky_x > 5:", "    if flaky_x < 5:"), candidates[2]]
    ok, why = tgo.predicate_symptom_direction("cmp_flip", flaky_cands, {0, 1, 2})
    assert ok == {0, 1}
    assert why.endswith("could not be read (error: ValueError)")
    assert _loud(caplog, MODULE_STEM)

    # Nothing positively reverses: the rung abstains instead of firing on unread candidates alone.
    ok, why = tgo.predicate_symptom_direction("cmp_flip", [flaky_cands[1], candidates[1]], {0, 1})
    assert (ok, why) == (None, "")
