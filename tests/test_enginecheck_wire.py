"""Cache-prefix wire checks: pure rules on hand-built wires, offline traces, a real scripted run, and negative controls."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from enginecheck import HarnessError, Need, Scenario, Trace, VacuousScenario, run
from enginecheck import wire as wire_module
from enginecheck.wire import (
    OfflineTokenizer,
    PairMeasurement,
    Wire,
    WireReport,
    analyze,
    assert_deterministic,
    assert_no_cache_drop,
    assert_off_identical,
    assert_prefix_stable,
    check_prefix_pair,
    common_prefix_length,
    pair_measurements,
    scorewire_cached_tokens,
    shipped_switches,
)
from stateframe.engine import kernel_cache_layout as LAYOUT

IM = 0x110000  # above any ord() of fake content, so <|im_start|> can never collide with text


class FakeTokenizer:
    """Deterministic stand-in for the relay: <|im_start|> is one token, every other character is its ordinal."""

    def render(self, messages: Sequence[Mapping[str, object]], tools: object) -> str:
        del tools
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}\n" for m in messages) + "<|im_start|>assistant\n"

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        tokens: list[int] = []
        for index, part in enumerate(prompt.split("<|im_start|>")):
            if index:
                tokens.append(IM)
            tokens.extend(ord(char) for char in part)
        return tuple(tokens)

    def marker(self) -> int:
        return IM


def _trace(tmp_path: Path, bodies: list[dict[str, object]], raws: list[bytes] | None = None,
           responded: set[int] | None = None, errored: set[int] | None = None) -> Trace:
    """A trace whose wire siblings exist; every send responds unless `responded` names the answered turns."""
    requests = raws if raws is not None else [json.dumps(body, sort_keys=True).encode() for body in bodies]
    answered = set(range(1, len(requests) + 1)) if responded is None else responded
    _write_wire_siblings(tmp_path, count=len(requests), responded=answered, errored=errored or set())
    return Trace(requests=requests, request_json=bodies, events=[], exit_code=0, switches={},
                 run_dir=tmp_path, repo=tmp_path)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


SYSTEM = _msg("system", "Protocol rules.")
TASK = _msg("user", "Task: read the files and update notes.")


def _read_turns(extra_turns: int) -> list[dict[str, object]]:
    """System + task stable, then one read round trip per turn. Each turn only appends history."""
    messages: list[dict[str, str]] = [SYSTEM, TASK]
    bodies: list[dict[str, object]] = [{"messages": list(messages), "tools": None}]
    for index in range(extra_turns):
        messages = [*messages, _msg("assistant", f"read step {index}"), _msg("tool", f"result {index}")]
        bodies.append({"messages": list(messages), "tools": None})
    return bodies


# --- 1. Pure rules on hand-built wires (no engine, no relay) ---

def test_append_inside_stable_message_is_a_boundary_failure() -> None:
    previous = Wire(turn=1, raw=b"", body={"messages": []}, tokens=(1, IM, 2, 3, 4, 5, 6), stable_prefix_end=7)
    current = Wire(turn=2, raw=b"", body={"messages": []}, tokens=(1, IM, 2, 3, 4, 5, 6, 7, 8))
    measurement = check_prefix_pair(previous, current, im_start_token=IM, layout=LAYOUT)
    assert measurement.lcp == 7
    assert measurement.cached_tokens == 1
    assert measurement.failure == "prefix_turn_2_boundary_before_protected_extent_b_1_e_7"


def test_system_change_is_a_band_failure() -> None:
    previous = Wire(turn=1, raw=b"", body={"messages": [SYSTEM, TASK], "tools": None},
                    tokens=(1, 2, IM, 3), stable_prefix_end=3)
    changed_system = _msg("system", "Protocol rules, edited.")
    current = Wire(turn=2, raw=b"", body={"messages": [changed_system, TASK], "tools": None},
                   tokens=(1, 9, IM, 3, 4))
    measurement = check_prefix_pair(previous, current, im_start_token=IM, layout=LAYOUT)
    assert measurement.lcp == 1
    assert measurement.failure == "prefix_turn_2_band_system_offset_1"


def test_refused_middle_turn_is_no_sample_and_never_the_baseline() -> None:
    first = Wire(turn=1, raw=b"", body={"messages": []}, tokens=(1, 2, 3, IM, 4), stable_prefix_end=3)
    refused = Wire(turn=2, raw=b"", body={"messages": []}, no_sample_reason="refused")
    third = Wire(turn=3, raw=b"", body={"messages": []}, tokens=(1, 2, 3, IM, 4, 5), stable_prefix_end=3)
    measurements = pair_measurements([first, refused, third], im_start_token=IM, layout=LAYOUT)
    assert [m.turn for m in measurements] == [2, 3]
    assert measurements[0].failure == "no_sample:2"
    assert measurements[0].cached_tokens == 0
    # Compared with turn 1, not the refused turn 2 (which has no tokens and would give lcp 0).
    assert measurements[1].lcp == 5
    assert measurements[1].failure is None


def test_scorewire_picks_last_boundary_at_or_before_lcp() -> None:
    tokens = (IM, 1, IM, 2, 3)
    assert scorewire_cached_tokens(tokens, 2, IM) == 2  # tokens[lcp] is the boundary itself
    assert scorewire_cached_tokens(tokens, 1, IM) == 0
    assert scorewire_cached_tokens(tokens, 4, IM) == 2
    assert scorewire_cached_tokens((1, 2), 1, IM) == 0
    with pytest.raises(ValueError):
        scorewire_cached_tokens(tokens, -1, IM)


def test_common_prefix_length_stops_at_first_difference() -> None:
    assert common_prefix_length((1, 2, 3), (1, 2, 9)) == 2
    assert common_prefix_length((1, 2), (1, 2, 3)) == 2


# --- Offline traces through analyze (fake tokenizer, no relay) ---

def test_append_only_trace_is_stable_with_cached_tokens(tmp_path: Path) -> None:
    trace = _trace(tmp_path, _read_turns(2))
    report = analyze(trace, tokenizer=FakeTokenizer())  # type: ignore[arg-type]
    assert report.failures == ()
    assert report.cached_tokens > 0
    assert_prefix_stable(trace, tokenizer=FakeTokenizer())  # type: ignore[arg-type]


def test_system_change_in_later_turn_is_reported(tmp_path: Path) -> None:
    bodies = _read_turns(2)
    bodies[2]["messages"][0] = _msg("system", "Protocol rules, edited.")  # type: ignore[index]
    report = analyze(_trace(tmp_path, bodies), tokenizer=FakeTokenizer())  # type: ignore[arg-type]
    assert any("band_system" in failure for failure in report.failures)


def test_off_identical_names_turn_and_byte_offset(tmp_path: Path) -> None:
    off = _trace(tmp_path / "off", [{"a": 1}], [b'{"a":1}'])
    base = _trace(tmp_path / "base", [{"a": 1}], [b'{"a":2}'])
    with pytest.raises(AssertionError, match=r"turn 1.*byte offset 5"):
        assert_off_identical(off, base)
    assert_off_identical(off, _trace(tmp_path / "same", [{"a": 1}], [b'{"a":1}']))


def test_off_identical_flags_count_mismatch(tmp_path: Path) -> None:
    off = _trace(tmp_path / "off", [{"a": 1}, {"a": 2}])
    base = _trace(tmp_path / "base", [{"a": 1}])
    with pytest.raises(AssertionError, match="request count"):
        assert_off_identical(off, base)


def test_deterministic_pair_passes_and_drift_fails(tmp_path: Path) -> None:
    assert_deterministic(_trace(tmp_path / "a", [{"a": 1}]), _trace(tmp_path / "b", [{"a": 1}]))
    with pytest.raises(AssertionError):
        assert_deterministic(_trace(tmp_path / "c", [{"a": 1}]), _trace(tmp_path / "d", [{"a": 2}]))


def _report(cached: int) -> WireReport:
    return WireReport(pairs=(PairMeasurement(2, cached, cached, cached),), cached_tokens=cached,
                      no_sample=(), failures=())


def test_cache_drop_gate_uses_tolerance() -> None:
    assert_no_cache_drop(_report(100), _report(99))
    with pytest.raises(AssertionError):
        assert_no_cache_drop(_report(100), _report(98))
    with pytest.raises(AssertionError, match="vacuous"):
        assert_no_cache_drop(_report(0), _report(0))


# --- 2. One real scripted engine run through relay tokenization ---

def _relay_reachable() -> None:
    if not wire_module.RELAY:
        pytest.skip("tokenizer relay not configured (set STATEFRAME_TEST_TOKENIZER_RELAY)")
    try:
        OfflineTokenizer().marker()
    except HarnessError as error:
        if "tokenizer_relay_unreachable" in str(error):
            pytest.skip(f"offline tokenizer relay unreachable ({wire_module.RELAY}); cannot tokenize the real run")
        raise


def _read_three_scenario() -> Scenario:
    files = {"a.py": b"a = 1\n", "b.py": b"b = 2\n", "c.py": b"c = 3\n", "notes.md": b"# notes\n"}
    steps = [{"tool": "read", "required": True, "arguments": {"path": name}} for name in ("a.py", "b.py", "c.py")]
    steps.append({"tool": "done", "required": True, "arguments": {"summary": "Read three files."}})
    return Scenario(
        files=files,
        task="Read a.py, b.py and c.py, then update notes.md.",
        steps=steps,
        switches=shipped_switches(),
        needs=[Need("the engine sent at least three provider requests", lambda t: len(t.requests) >= 3)],
    )


# Real engine cache break under the SHIPPED profile (captured run, 2026-11-15). Not a harness defect: the check is unchanged and
# pinned to this exact failure tuple. A different failure list raises RuntimeError, which strict xfail does not accept.
SHIPPED_ENGINE_FAILURES: tuple[str, ...] = (
    "prefix_turn_2_tool_menu_added_ungated_added_recall_generated_artifact_removed_-_offset_5024",
    "prefix_turn_3_band_checks_offset_6754",
    "prefix_turn_4_band_checks_offset_6934",
    "prefix_turn_5_band_checks_offset_7138",
    "prefix_turn_6_band_checks_offset_7299",
)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CACHE-CHECKSBAND: shipped-profile engine cache break: " + "; ".join(SHIPPED_ENGINE_FAILURES),
)
def test_real_scripted_run_keeps_stable_prefix(tmp_path: Path) -> None:
    _relay_reachable()
    try:
        trace = run(_read_three_scenario(), tmp_path)
    except VacuousScenario as error:
        # A vacuous run is a harness problem, not the pinned engine break: it must not satisfy the xfail.
        raise RuntimeError(f"real scenario was vacuous, not an engine verdict: {error}") from error
    report = analyze(trace)
    if not report.failures:
        return  # engine fixed: strict xfail reports XPASS, prompting removal of the marker
    message = "; ".join(report.failures)
    pinned = "; ".join(SHIPPED_ENGINE_FAILURES)
    if message != pinned:
        raise RuntimeError(f"unexpected real-run failures (not the pinned engine break): {message}")
    raise AssertionError(f"pinned shipped-profile engine break: {message}")


# --- 3. Negative controls: the gate must reject a broken analyzer, and accept the real one ---

def _gate_is_correct(tmp_path: Path) -> None:
    """The gate passes a clean append-only trace and rejects a system-band change. Raises if either is wrong."""
    tokenizer = FakeTokenizer()  # type: ignore[assignment]
    good = _trace(tmp_path / "good", _read_turns(2))
    bad_bodies = _read_turns(2)
    bad_bodies[2]["messages"][0] = _msg("system", "Protocol rules, edited.")  # type: ignore[index]
    bad = _trace(tmp_path / "bad", bad_bodies)
    assert_prefix_stable(good, tokenizer=tokenizer)  # type: ignore[arg-type]
    try:
        assert_prefix_stable(bad, tokenizer=tokenizer)  # type: ignore[arg-type]
    except AssertionError:
        return
    raise AssertionError("gate accepted a system-band change")


def test_real_gate_is_correct(tmp_path: Path) -> None:
    _gate_is_correct(tmp_path)


def test_always_pass_analyzer_fails_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def always_pass(trace: Trace, tokenizer: object = None) -> WireReport:
        return WireReport(pairs=(), cached_tokens=0, no_sample=(), failures=())

    monkeypatch.setattr(wire_module, "analyze", always_pass)
    with pytest.raises(AssertionError):
        _gate_is_correct(tmp_path)


def test_always_fail_analyzer_fails_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(trace: Trace, tokenizer: object = None) -> WireReport:
        return WireReport(pairs=(), cached_tokens=0, no_sample=(), failures=("forced failure",))

    monkeypatch.setattr(wire_module, "analyze", always_fail)
    with pytest.raises(AssertionError):
        _gate_is_correct(tmp_path)


# --- Tool-menu gate: only an exact-prefix append of GATED_TOOL_NAMES may change the tool list between turns ---

def _tools(names: Sequence[str], schema: str = "object") -> list[dict[str, object]]:
    return [{"type": "function", "function": {"name": name, "parameters": {"type": schema}}} for name in names]


def _menu_failure(before: list[dict[str, object]], after: list[dict[str, object]]) -> str | None:
    """Same tokens on both sides, so only the tool-menu gate can fail the pair."""
    previous = Wire(turn=1, raw=b"", body={"messages": [], "tools": before}, tokens=(1, 2, IM, 3, 4),
                    stable_prefix_end=2)
    current = Wire(turn=2, raw=b"", body={"messages": [], "tools": after}, tokens=(1, 2, IM, 3, 4, 5),
                   stable_prefix_end=2)
    return check_prefix_pair(previous, current, im_start_token=IM, layout=LAYOUT).failure


def test_gated_append_is_allowed() -> None:
    assert _menu_failure(_tools(["read", "edit"]), _tools(["read", "edit", "job"])) is None


def test_ungated_add_fails_and_names_the_tool() -> None:
    failure = _menu_failure(_tools(["read", "edit"]), _tools(["read", "edit", "search"]))
    assert failure == "prefix_turn_2_tool_menu_added_ungated_added_search_removed_-_offset_5"


def test_gated_tool_not_appended_fails() -> None:
    failure = _menu_failure(_tools(["read", "edit"]), _tools(["read", "job", "edit"]))
    assert failure is not None and "gated_not_appended_added_job" in failure


def test_removal_fails_and_names_the_tool() -> None:
    failure = _menu_failure(_tools(["read", "edit"]), _tools(["read"]))
    assert failure is not None and "tool_menu_removed_added_-_removed_edit" in failure


def test_reorder_fails() -> None:
    failure = _menu_failure(_tools(["read", "edit"]), _tools(["edit", "read"]))
    assert failure is not None and "tool_menu_reordered" in failure


def test_schema_edit_fails() -> None:
    failure = _menu_failure(_tools(["read", "edit"]), _tools(["read", "edit"], schema="string"))
    assert failure is not None and "tool_menu_schema_edited" in failure


def test_shipped_switches_exclude_launcher_policy_and_keep_engine_switches() -> None:
    switches = shipped_switches()
    assert "STATEFRAME_PROTECT_TESTS" not in switches
    assert switches["STATEFRAME_F4"] == "on"
    assert switches["STATEFRAME_FRAMEFIT"] == "on"


# --- Repair round: no-sample siblings, malformed tool menus, relay HTTP faults, vacuous real runs ---

def _write_wire_siblings(run_dir: Path, count: int, responded: set[int], errored: set[int] = frozenset()) -> None:
    """Engine wire siblings: a request file per send, a response only for responded sends, an error record if any."""
    wire = run_dir / "wire"
    wire.mkdir(parents=True, exist_ok=True)
    for turn in range(1, count + 1):
        (wire / f"{turn:04d}.request.json").write_bytes(b"{}")
        if turn in responded:
            (wire / f"{turn:04d}.response.json").write_bytes(b"{}")
        if turn in errored:
            (wire / f"{turn:04d}.error.txt").write_text("send failed", encoding="utf-8")


def test_send_without_response_is_no_sample_and_never_the_baseline(tmp_path: Path) -> None:
    # Turn 2 has no response file, so it is no sample. Turn 3 must be judged against turn 1 and pass.
    # Before the repair the sibling files were ignored: no_sample was empty and turn 2 was compared (failure).
    report = analyze(_trace(tmp_path, _read_turns(2), responded={1, 3}),
                     tokenizer=FakeTokenizer())  # type: ignore[arg-type]
    assert report.no_sample == (2,)
    assert report.failures == ("no_sample:2",)


def test_send_with_error_record_is_no_sample(tmp_path: Path) -> None:
    report = analyze(_trace(tmp_path, _read_turns(2), responded={1, 3}, errored={2}),
                     tokenizer=FakeTokenizer())  # type: ignore[arg-type]
    assert report.no_sample == (2,)


def test_tool_menu_with_non_mapping_entry_is_malformed(tmp_path: Path) -> None:
    # Before the repair the junk entry was filtered out, so the menu looked unchanged and passed.
    before = _tools(["read"])
    after = [*_tools(["read"]), "junk"]  # type: ignore[list-item]
    assert _menu_failure(before, after) == "prefix_turn_2_tool_menu_malformed_offset_5"


def test_relay_http_error_is_not_retried_as_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Before the repair HTTPError was caught as URLError: three attempts, then tokenizer_relay_unreachable.
    calls: list[int] = []

    def fake_urlopen(request: object, timeout: float) -> object:
        calls.append(1)
        raise urllib.error.HTTPError("http://relay.invalid/tokenize", 500, "relay fault", {}, None)

    monkeypatch.setattr(wire_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wire_module.time, "sleep", lambda seconds: None)
    with pytest.raises(HarnessError, match="tokenizer_relay_http_error"):
        OfflineTokenizer("http://relay.invalid").tokenize("x")
    assert len(calls) == 1


def test_real_run_vacuous_scenario_cannot_satisfy_the_xfail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Before the repair the VacuousScenario (an AssertionError) escaped and would have satisfied the strict xfail.
    def vacuous(scenario: Scenario, tmp: Path) -> Trace:
        raise VacuousScenario("the engine sent no requests")

    monkeypatch.setitem(globals(), "run", vacuous)
    monkeypatch.setitem(globals(), "_relay_reachable", lambda: None)
    with pytest.raises(RuntimeError, match="vacuous"):
        test_real_scripted_run_keeps_stable_prefix(tmp_path)
