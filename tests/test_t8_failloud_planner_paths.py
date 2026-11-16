"""Fail-loud checks for broad except sites in the planner, comment-fact and line-stage modules.

Scope: comment_facts.py, line_stage.py, unit_planner_v2.py (T8H-PLANNER unit).

Loudness bar: a raised exception, a WARNING-or-higher record from the owning module, or a typed
journal event / typed field that a reader checks. Tags (T8-<MODULE>-<line>) name the broad except each
test covers. Pins cover default-path sites whose only safety argument is a typed field. Opt-in coalescer sites all route
through ``unit_planner_v2._coalesce_error`` (WARNING + ``unit_coalesce_error`` journal event), so one
sink test covers them and the inventory guard keeps every site named.
"""
from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import comment_facts
import line_stage
import unit_planner_v2

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
MODULES = ("comment_facts", "line_stage", "unit_planner_v2")
BROAD = re.compile(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)")


def _loud(caplog: pytest.LogCaptureFixture, module_name: str) -> bool:
    return any(record.levelno >= logging.WARNING and module_name in record.name
               for record in caplog.records)


class _FakeBolt:
    """Minimal bolt surface: the coalescer journals through ``_rec``."""

    def __init__(self, fail: bool = False) -> None:
        self.events: list[tuple[str, str, str]] = []
        self._fail = fail

    def _rec(self, kind: str, event: str, detail: str) -> None:
        if self._fail:
            raise RuntimeError("forced: journal unavailable")
        self.events.append((kind, event, detail))


class _Ask:
    """Stand-in for the line-stage model transport."""

    def __init__(self, probe=None, menu=None) -> None:
        self._probe = probe
        self._menu = menu

    def probe(self, item):
        return self._probe(item)

    def menu(self, items):
        return self._menu(items)


# ---------------------------------------------------------------- comment_facts (default path)

# comment_facts 79/86/284 are DEGRADE-OK: age/comment facts are optional hints and the builder is
# documented fail-soft, so a failure drops the hint (abstain) and never yields a wrong verdict.

def test_blame_subprocess_failure_drops_age_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T8-COMMENT_FACTS-79 (DEGRADE-OK): git blame unavailable -> empty age map, no raise."""
    calls: list[object] = []

    def boom(cmd, **_kwargs):
        calls.append(cmd)
        raise OSError("forced: git unavailable")

    monkeypatch.setattr(comment_facts.subprocess, "run", boom)
    assert comment_facts._blame_committer_times(str(tmp_path), "mod.py", {1}) == {}
    assert len(calls) == 1


def test_blame_undecodable_output_drops_age_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T8-COMMENT_FACTS-86 (DEGRADE-OK): non-UTF-8 porcelain output -> empty age map, no raise."""
    calls: list[object] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\xff\xfe author \xff", stderr=b"")

    monkeypatch.setattr(comment_facts.subprocess, "run", fake_run)
    assert comment_facts._blame_committer_times(str(tmp_path), "mod.py", {1}) == {}
    assert len(calls) == 1


def test_comment_facts_builder_defect_fails_soft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T8-COMMENT_FACTS-284 (DEGRADE-OK): a builder defect returns [] (no hints), never raises."""
    (tmp_path / "mod.py").write_text("def f():\n    # the note\n    return 1\n", encoding="utf-8")
    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("span")
        raise RuntimeError("forced: span builder defect")

    monkeypatch.setattr(comment_facts, "_function_span", boom)
    assert comment_facts.extract_comment_facts(str(tmp_path), "mod.py", "f") == []
    assert calls == ["span"]


# ---------------------------------------------------------------- line_stage (default path)

def _fn_node(src: str = "def f():\n    return 1\n") -> ast.AST:
    return ast.parse(src).body[0]


# T8-LINE_STAGE-484: a missing dataflow layer is logged and typed, not swallowed.
def test_dataflow_import_failure_is_visible(monkeypatch: pytest.MonkeyPatch,
                                            caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setitem(sys.modules, "graphs.dataflow", None)
    caplog.set_level(logging.DEBUG)
    df, why = line_stage._dataflow(_fn_node(), "mod.py", "f")
    assert df is None
    assert why.startswith("dataflow layer unavailable: import failed (")
    assert _loud(caplog, "line_stage")


# T8-LINE_STAGE-492: a dataflow analysis defect is logged and typed with the exception name.
def test_dataflow_analysis_failure_is_visible(monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    import graphs.dataflow as dataflow_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced: def-use analysis defect")

    monkeypatch.setattr(dataflow_module, "analyze_function", boom)
    caplog.set_level(logging.DEBUG)
    df, why = line_stage._dataflow(_fn_node(), "mod.py", "f")
    assert df is None
    assert why == "dataflow analysis failed (RuntimeError)"
    assert _loud(caplog, "line_stage")


_FN_LINES = ["def f(x):", "    y = x + 1", "    if y > 2:", "        return y", "    return 0"]


# Regression: the failure reaches the universe and the certificate as the degraded state + cause.
def test_dataflow_analysis_failure_marks_universe_and_cert_degraded(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import graphs.dataflow as dataflow_module

    def boom(*_args, **_kwargs):
        raise TypeError("forced: def-use analysis defect")

    monkeypatch.setattr(dataflow_module, "analyze_function", boom)
    caplog.set_level(logging.DEBUG)
    universe = line_stage.prune_candidates(_FN_LINES, "f", 1, 5, symptom="y", rel_path="mod.py")
    assert universe.degraded is True
    assert universe.degrade_reason == "dataflow analysis failed (TypeError)"
    assert len(universe) > 0                      # degraded, never empty
    result = line_stage.run_line_stage(_FN_LINES, "f", 1, 5, symptom="y", rel_path="mod.py")
    assert result.cert["universe"]["degraded"] is True
    assert result.cert["universe"]["degrade_reason"] == "dataflow analysis failed (TypeError)"
    assert _loud(caplog, "line_stage")


def test_dataflow_import_failure_marks_universe_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "graphs.dataflow", None)
    universe = line_stage.prune_candidates(_FN_LINES, "f", 1, 5, symptom="y", rel_path="mod.py")
    assert universe.degraded is True
    assert universe.degrade_reason.startswith("dataflow layer unavailable: import failed (")
    assert len(universe) > 0


def test_dataflow_healthy_path_is_not_degraded_and_quiet(caplog: pytest.LogCaptureFixture) -> None:
    """Activation witness for the two tests above: with the real layer the universe is clean."""
    caplog.set_level(logging.DEBUG)
    df, why = line_stage._dataflow(_fn_node("\n".join(_FN_LINES) + "\n"), "mod.py", "f")
    assert df is not None and why == ""
    universe = line_stage.prune_candidates(_FN_LINES, "f", 1, 5, symptom="y", rel_path="mod.py")
    assert universe.degraded is False
    assert universe.degrade_reason == ""
    assert not _loud(caplog, "line_stage")


# T8-LINE_STAGE-1100: probe transport failure becomes MODEL_UNAVAILABLE with the exception type recorded.
def test_probe_transport_failure_is_typed() -> None:
    def boom(_item):
        raise RuntimeError("forced: transport down")

    items = [("c1", "display")]
    state, selected, detail = line_stage._run_probes(_Ask(probe=boom), items)
    assert state == line_stage.MODEL_UNAVAILABLE
    assert selected == ()
    assert detail["transport_error"] == "RuntimeError"


# T8-LINE_STAGE-1130: menu transport failure is typed on the cert detail.
def test_menu_transport_failure_is_typed() -> None:
    def boom(_items):
        raise RuntimeError("forced: transport down")

    items = [("c1", "display")]
    state, selected, detail = line_stage._run_menu(_Ask(menu=boom), items, line_stage.LIMITS)
    assert state == line_stage.MODEL_UNAVAILABLE
    assert selected == ()
    assert detail["transport_error"] == "RuntimeError"


# T8-LINE_STAGE-1151: a failed thin-margin re-ask keeps the first answer and records the error type.
def test_menu_reask_failure_keeps_first_answer_and_is_typed() -> None:
    calls = {"n": 0}

    def menu(_items):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "selected", "line_ids": ["c1"], "confidence": "low", "margin": 0.0}
        raise RuntimeError("forced: re-ask transport down")

    items = [("c1", "display")]
    state, selected, detail = line_stage._run_menu(_Ask(menu=menu), items, line_stage.LIMITS)
    assert state == line_stage.SELECTED
    assert selected == ("c1",)
    assert detail["reasked"] is True
    assert detail["reask_error"] == "RuntimeError"


# T8-LINE_STAGE-1130 (owner decision): a programming error in the menu call is ENGINE_ERROR, not a
# model outage; the exception type is kept and the failure is logged.
@pytest.mark.parametrize("exc_type", [TypeError, AttributeError, KeyError, NotImplementedError])
def test_menu_programming_error_is_engine_error(exc_type: type,
                                                caplog: pytest.LogCaptureFixture) -> None:
    def boom(_items):
        raise exc_type("forced: adapter bug")

    caplog.set_level(logging.DEBUG)
    state, selected, detail = line_stage._run_menu(_Ask(menu=boom), [("c1", "display")],
                                                   line_stage.LIMITS)
    assert state == line_stage.ENGINE_ERROR
    assert selected == ()
    assert detail["engine_error"] == exc_type.__name__
    assert "transport_error" not in detail
    assert _loud(caplog, "line_stage")


# T8-LINE_STAGE-1100: the probe path classifies the same way as the menu path.
def test_probe_programming_error_is_engine_error(caplog: pytest.LogCaptureFixture) -> None:
    def boom(_item):
        raise AttributeError("forced: adapter bug")

    caplog.set_level(logging.DEBUG)
    state, selected, detail = line_stage._run_probes(_Ask(probe=boom), [("c1", "display")])
    assert state == line_stage.ENGINE_ERROR
    assert selected == ()
    assert detail["engine_error"] == "AttributeError"
    assert "transport_error" not in detail
    assert _loud(caplog, "line_stage")


@pytest.mark.parametrize("exc_type", [OSError, ConnectionError, TimeoutError, RuntimeError, ValueError])
def test_menu_transport_class_errors_stay_model_unavailable(exc_type: type) -> None:
    def boom(_items):
        raise exc_type("forced: transport down")

    state, _selected, detail = line_stage._run_menu(_Ask(menu=boom), [("c1", "display")],
                                                    line_stage.LIMITS)
    assert state == line_stage.MODEL_UNAVAILABLE
    assert detail["transport_error"] == exc_type.__name__
    assert "engine_error" not in detail


def test_engine_error_is_non_selecting_and_keeps_every_anchor() -> None:
    """ENGINE_ERROR reorders nothing: the whole universe stays an anchor and the cert is typed."""
    assert line_stage.ENGINE_ERROR in line_stage.NON_SELECTING_STATES

    def boom(_items):
        raise TypeError("forced: adapter bug")

    def probe_boom(_item):
        raise TypeError("forced: adapter bug")

    result = line_stage.run_line_stage(_FN_LINES, "f", 1, 5, symptom="y", rel_path="mod.py",
                                       ask=_Ask(probe=probe_boom, menu=boom))
    assert result.state == line_stage.ENGINE_ERROR
    assert result.cert["model"]["state"] == line_stage.ENGINE_ERROR
    assert result.cert["model"]["engine_error"] == "TypeError"
    assert len(result.anchors) == len(result.candidates)
    assert result.cert["invariant"]["anchors_equal_universe"] is True


# ---------------------------------------------------------------- unit_planner_v2 (default + opt-in)

# T8-UNIT_PLANNER_V2-584: default-path pytest rewrite fails open with a WARNING when the gate read raises.
def test_pytest_command_gate_failure_fails_open_loudly(monkeypatch: pytest.MonkeyPatch,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    def boom(*_args, **_kwargs):
        raise ValueError("forced: bad switch value")

    monkeypatch.setattr(unit_planner_v2, "enabled", boom)
    command = ["python", "-m", "pytest", "-x", "-q", "tests/test_x.py"]
    caplog.set_level(logging.DEBUG)
    assert unit_planner_v2.coalesce_pytest_command(command) == command
    assert _loud(caplog, "unit_planner_v2")


# T8-UNIT_PLANNER_V2-954: default-path weld fails open with a WARNING and leaves the transcript untouched.
def test_weld_gate_failure_leaves_transcript_and_is_loud(monkeypatch: pytest.MonkeyPatch,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    class _Plan:
        pass

    def boom(*_args, **_kwargs):
        raise ValueError("forced: bad switch value")

    monkeypatch.setattr(unit_planner_v2, "UnitPlanV2", _Plan)
    monkeypatch.setattr(unit_planner_v2, "_coalesce_enabled", boom)
    bolt = _FakeBolt()
    bolt.unit_plan = _Plan()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    before = [dict(m) for m in messages]
    caplog.set_level(logging.DEBUG)
    unit_planner_v2.weld_initial_plan(bolt, messages)
    assert messages == before
    assert _loud(caplog, "unit_planner_v2")
    assert bolt.events and bolt.events[0][1] == "unit_coalesce_error"


# Opt-in coalescer sites (573, 600, 623, 636, 702, 757, 797, 815, 833,
# 866, 928, 943, 962) all funnel through _coalesce_error: one WARNING plus one typed journal event.
def test_coalesce_error_sink_is_loud_and_journaled(caplog: pytest.LogCaptureFixture) -> None:
    bolt = _FakeBolt()
    caplog.set_level(logging.DEBUG)
    unit_planner_v2._coalesce_error(bolt, SimpleNamespace(unit_id="u1"), RuntimeError("forced"))
    assert _loud(caplog, "unit_planner_v2")
    assert bolt.events == [("unit_plan", "unit_coalesce_error", "u1 RuntimeError: forced")]


# T8-UNIT_PLANNER_V2-561: when the journal itself fails, the sink still leaves a WARNING.
def test_coalesce_error_journal_failure_still_logs(caplog: pytest.LogCaptureFixture) -> None:
    bolt = _FakeBolt(fail=True)
    caplog.set_level(logging.DEBUG)
    unit_planner_v2._coalesce_error(bolt, None, RuntimeError("forced"))
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("unit_coalesce_error journal" in m for m in messages)


# ---------------------------------------------------------------- guards

def test_planner_broad_excepts_are_triaged() -> None:
    pilot: set[tuple[str, int]] = set()
    triaged = {
        # comment_facts.py (default path)
        ("comment_facts", 79), ("comment_facts", 86), ("comment_facts", 109),
        ("comment_facts", 155), ("comment_facts", 284),
        # line_stage.py (default path)
        ("line_stage", 484), ("line_stage", 492), ("line_stage", 1100),
        ("line_stage", 1130), ("line_stage", 1151),
        # unit_planner_v2.py (default: 584, 954; opt-in: the rest; 561 is the census-LOUD journal sink)
        ("unit_planner_v2", 561), ("unit_planner_v2", 573), ("unit_planner_v2", 584),
        ("unit_planner_v2", 600), ("unit_planner_v2", 623), ("unit_planner_v2", 636),
        ("unit_planner_v2", 702), ("unit_planner_v2", 757), ("unit_planner_v2", 797),
        ("unit_planner_v2", 815), ("unit_planner_v2", 833), ("unit_planner_v2", 866),
        ("unit_planner_v2", 928), ("unit_planner_v2", 943), ("unit_planner_v2", 954),
        ("unit_planner_v2", 962),
    }
    found: set[tuple[str, int]] = set()
    for stem in MODULES:
        path = ENGINE / f"{stem}.py"
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not BROAD.match(text):
                continue
            if re.match(r"^\s*except\s*\(", text) and not re.search(r"Exception|BaseException", text):
                continue  # narrow tuple
            found.add((stem, number))
    assert found == pilot | triaged, (
        f"untriaged: {sorted(found - (pilot | triaged))}; stale: {sorted((pilot | triaged) - found)}")


def test_t8_planner_tags_point_at_broad_except_sites() -> None:
    """Every T8-<MODULE>-<line> tag in this file names an existing broad except line."""
    source = Path(__file__).read_text(encoding="utf-8")
    tags = set(re.findall(r"T8-([A-Z0-9_]+)-(\d+)", source))
    assert tags
    for module, line in sorted(tags):
        lines = (ENGINE / (module.lower() + ".py")).read_text(encoding="utf-8").splitlines()
        assert BROAD.match(lines[int(line) - 1]), (
            f"{module}:{line} is no longer a broad except; retag or promote the test")
