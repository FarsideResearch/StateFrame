"""The opt-in host policy selects only useful, feasible fan-out."""

from __future__ import annotations

from types import SimpleNamespace

import kernel_parallel_suggest
import pytest
from kernel_parallel_suggest import (
    ParallelUnit,
    SuggestionConfig,
    apply_suggestion,
    decide_parallel_suggestion,
    independent_pairs,
    render_suggestion,
)


def _units(*, source_bytes: int = 100, source_lines: int = 10,
           test_bytes: int = 100, test_lines: int = 10) -> tuple[ParallelUnit, ...]:
    return tuple(ParallelUnit(f"src/{name}.py", f"tests/test_{name}.py",
                              source_bytes, test_bytes, source_lines, test_lines)
                 for name in ("a", "b"))


@pytest.mark.parametrize("backend,axis,offered", [
    ("local", "input", True), ("local", "output", True),
    ("local", "both", True), ("local", "none", False),
    ("hosted", "input", True), ("hosted", "output", True),
    ("hosted", "both", True), ("hosted", "none", True),
    ("unknown", "input", True), ("unknown", "output", True),
    ("unknown", "both", True), ("unknown", "none", False),
])
def test_backend_axis_matrix(backend: str, axis: str, offered: bool) -> None:
    units = _units(source_bytes=3000 if axis in ("input", "both") else 100,
                   source_lines=1000 if axis in ("output", "both") else 10)
    decision = decide_parallel_suggestion(
        units, frame_budget_tokens=1000, max_output_tokens=1000,
        configured_slots=2, backend_kind=backend,
        discovered_slots=2 if backend == "local" else None,
    )
    assert decision.recommend is offered
    assert decision.axis == axis
    assert decision.reason.startswith(f"{backend}:axis={axis}:")


@pytest.mark.parametrize("slots", [0, 1])
@pytest.mark.parametrize("backend", ["local", "hosted", "unknown"])
def test_insufficient_slots_stay_serial(backend: str, slots: int) -> None:
    decision = decide_parallel_suggestion(
        _units(source_bytes=3000), frame_budget_tokens=1000,
        max_output_tokens=1000, configured_slots=slots, backend_kind=backend,
        discovered_slots=slots if backend == "local" else None,
    )
    assert not decision.recommend
    assert decision.axis == "input"


def test_local_discovery_failure_uses_conservative_unknown_rules() -> None:
    decision = decide_parallel_suggestion(
        _units(), frame_budget_tokens=1000, max_output_tokens=1000,
        configured_slots=2, backend_kind="local", discovered_slots=None,
    )
    assert not decision.recommend
    assert decision.reason.startswith("unknown:axis=none:")


def test_unit_count_and_exact_input_threshold() -> None:
    config = SuggestionConfig(min_units=2, input_share_pct=50)
    one = decide_parallel_suggestion(
        _units(source_bytes=615, test_bytes=0)[:1], frame_budget_tokens=1000,
        max_output_tokens=1000, configured_slots=2, backend_kind="local",
        discovered_slots=2, config=config)
    edge = decide_parallel_suggestion(
        _units(source_bytes=615, test_bytes=0), frame_budget_tokens=1000,
        max_output_tokens=1000, configured_slots=2, backend_kind="local",
        discovered_slots=2, config=config)
    above = decide_parallel_suggestion(
        _units(source_bytes=616, test_bytes=0), frame_budget_tokens=1000,
        max_output_tokens=1000, configured_slots=2, backend_kind="local",
        discovered_slots=2, config=config)
    assert not one.recommend
    assert not edge.recommend  # combined 1230 bytes = 500 estimated tokens
    assert above.recommend


def test_pair_detection_requires_unambiguous_disjoint_python_checks() -> None:
    sources = {
        "src/a.py": "def a():\n    return 1\n",
        "src/b.py": "def b():\n    return 2\n",
        "tests/test_a.py": "from src.a import a\ndef test_a(): assert a() == 1\n",
        "tests/test_b.py": "from src.b import b\ndef test_b(): assert b() == 2\n",
    }
    pairs = independent_pairs(sources, ("src/a.py", "src/b.py"),
                              ("tests/test_a.py", "tests/test_b.py"))
    assert [(unit.source, unit.test) for unit in pairs] == [
        ("src/a.py", "tests/test_a.py"), ("src/b.py", "tests/test_b.py")]
    sources["src/unrelated.py"] = "def unrelated(): return True\n"
    assert [(unit.source, unit.test) for unit in independent_pairs(
        sources, ("src/a.py", "src/b.py", "src/unrelated.py"),
        ("tests/test_a.py", "tests/test_b.py"))] == [
            ("src/a.py", "tests/test_a.py"), ("src/b.py", "tests/test_b.py")]
    sources["tests/test_b.py"] += "from src.a import a\n"
    assert independent_pairs(sources, ("src/a.py", "src/b.py"),
                             ("tests/test_a.py", "tests/test_b.py")) == ()
    sources["tests/test_b.py"] = "from src.b import b\ndef test_b(): assert b() == 2\n"
    sources["src/b.py"] += "from src.a import a\n"
    assert independent_pairs(sources, ("src/a.py", "src/b.py"),
                             ("tests/test_a.py", "tests/test_b.py")) == ()


def test_render_names_pairs_and_concrete_tool_arguments() -> None:
    fact, guidance = render_suggestion(_units(), backend_kind="local", axis="both")
    assert "src/a.py<->tests/test_a.py" in fact
    assert "src/b.py<->tests/test_b.py" in fact
    assert "`parallel`" in fact
    assert '"owns"' in guidance and '"check"' in guidance
    assert "local" in guidance and "both" in guidance


def test_on_presents_outlines_without_pair_bodies() -> None:
    sources = {"src/a.py": "SECRET_A = 123\ndef a(): return SECRET_A\n",
               "src/b.py": "SECRET_B = 456\ndef b(): return SECRET_B\n",
               "tests/test_a.py": "from src.a import a\ndef test_a(): assert a() == 123\n",
               "tests/test_b.py": "from src.b import b\ndef test_b(): assert b() == 456\n"}
    units = independent_pairs(sources, ("src/a.py", "src/b.py"),
                              ("tests/test_a.py", "tests/test_b.py"))
    decision = decide_parallel_suggestion(
        units, frame_budget_tokens=10, max_output_tokens=1000,
        configured_slots=2, backend_kind="hosted", discovered_slots=None)
    messages = [{"role": "system", "content": "System rules.\nOther rules."},
                {"role": "user", "content": "Fix both pairs."},
                {"role": "user", "content":
                 "[Host fact]\nCurrent source bytes attached to this request: "
                 "src/a.py lines 1-2 revision unavailable; src/b.py lines 1-2 revision unavailable"},
                {"role": "user", "content":
                 "[Deferred required source paths]\nSource ranges delivered in this request:\n"
                 "- src/a.py:1-2 revision=unknown\nRequired source paths are not attached; "
                 "read one only if your edit needs it:\n- src/a.py"},
                *({"role": "user", "content":
                    f"[Current source: {path}]\n{source}"}
                  for path, source in sources.items())]
    plan = {"deferred_paths": [], "units": [{"path": path} for path in sources]}
    apply_suggestion(messages, plan, sources, units, decision)
    rendered = "\n".join(str(message["content"]) for message in messages)
    assert "Independent pairs:" in rendered
    assert rendered.index("Parallel selection:") < rendered.index("System rules.")
    assert "[Parallel outline:" in rendered
    from kernel_request_diet import _source_delivery_ranges
    assert not _source_delivery_ranges(messages)
    assert "SECRET_A = 123" not in rendered
    assert "SECRET_B = 456" not in rendered
    assert "assert a() == 123" not in rendered
    assert "Current source bytes attached to this request: src/a.py" not in rendered
    assert "- src/a.py:1-2 revision=unknown" not in rendered
    assert set(plan["deferred_paths"]) == set(sources)


def test_off_decision_does_not_mutate_wire_messages() -> None:
    messages = [{"role": "system", "content": "System rules."},
                {"role": "user", "content": "Keep exact bytes."}]
    before = repr(messages)
    decision = decide_parallel_suggestion(
        _units(), frame_budget_tokens=1000, max_output_tokens=1000,
        configured_slots=2, backend_kind="local", discovered_slots=2)
    apply_suggestion(messages, {"deferred_paths": []}, {}, _units(), decision)
    assert repr(messages) == before


def test_frame_suggestion_uses_public_row_files_and_checks(tmp_path, monkeypatch) -> None:
    (tmp_path / "tests").mkdir()
    for path, content in {
        "a.py": "def a(): return 1\n", "b.py": "def b(): return 2\n",
        "tests/test_a.py": "from a import a\ndef test_a(): assert a() == 1\n",
        "tests/test_b.py": "from b import b\ndef test_b(): assert b() == 2\n",
    }.items():
        (tmp_path / path).write_text(content)
    spec = {"chat_url": "http://127.0.0.1:9999/v1/chat/completions",
            "parallel_mode": "2", "parallel_limits": {"max_concurrency": 2}}
    monkeypatch.setenv("STATEFRAME_PARALLEL_SUGGEST", "on")
    monkeypatch.setenv("STATEFRAME_PARALLEL_SUGGEST_BACKEND", "hosted")
    monkeypatch.setattr(kernel_parallel_suggest, "_host_spec", spec)
    events = []
    row = {"files": {path: {} for path in ("a.py", "b.py", "tests/test_a.py",
                                         "tests/test_b.py")},
           "public_check_paths": ["tests/test_a.py", "tests/test_b.py"],
           "msg": "Fix a.py and b.py."}
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path, task_row=row),
                              journal=SimpleNamespace(emit=lambda *a, **kw: events.append((a, kw))))
    result = kernel_parallel_suggest.frame_suggestion(
        runtime, {"units": [{"path": "a.py"}]},
        [{"function": {"name": "parallel"}}, {"function": {"name": "read"}}],
        frame_budget_tokens=1000)
    assert result is not None and result[1].recommend
    assert [unit.source for unit in result[0]] == ["a.py", "b.py"]
    assert events[0][0] == ("parallel_suggestion_decision",)
    after_read = kernel_parallel_suggest.frame_suggestion(
        runtime, {"units": [{"path": "a.py"}]},
        [{"function": {"name": "parallel"}}, {"function": {"name": "read"}}],
        frame_budget_tokens=1000, previously_read_paths=("a.py",))
    assert after_read is not None and not after_read[1].recommend
    assert "prior_read" in after_read[1].reason


def test_local_capacity_reuses_idle_slots_and_excludes_parent(monkeypatch) -> None:
    import stateframe.parallel_resources as resources

    monkeypatch.setattr(resources, "idle_slots", lambda _endpoint: [0, 1, 2, 3])
    spec = {"chat_url": "http://127.0.0.1:8000/v1/chat/completions",
            "parallel_mode": "3", "parallel_limits": {"max_concurrency": 4},
            "parallel_slots": [1, 2, 3], "prompt_cache_slot": {"slot": 1}}
    assert kernel_parallel_suggest._capacity(spec, "local") == (3, 2)
    monkeypatch.setattr(resources, "idle_slots", lambda _endpoint: None)
    assert kernel_parallel_suggest._capacity(spec, "local") == (3, None)
