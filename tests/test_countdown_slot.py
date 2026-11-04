"""The visible deadline must not change request fitting or census admission."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_callsite_census as census
import kernel_countdown_slot as countdown
import kernel_request_source_window as source_window
import kernel_wire_stability as wire_stability
from kernel_context import token_estimate
from kernel_read_fixes import ContinuationLimits, ResponseBudget
from kernel_request_diet import RequestTokenMeter
from kernel_request_measurement import prompt_projection


@pytest.mark.parametrize(
    ("seconds", "now", "used", "expected"),
    [
        (0, 100.0, 0, "00000.0"),
        (-10, 100.0, 0, "00000.0"),
        (10**1000, 100.0, 0, "99999.9"),
        (None, 100.0, 0, "unbounded"),
        (60, 60.0, 1, "~0001 turns"),
        (60, 0.01, 1, "~999+ turns"),
        (60, 100.0, 0, "~---- turns"),
    ],
)
def test_countdown_extremes_fit_one_byte_and_token_reservation(
    monkeypatch: pytest.MonkeyPatch, seconds: object, now: float, used: int, expected: str,
) -> None:
    monkeypatch.setattr(countdown.time, "monotonic", lambda: now)
    budget = ResponseBudget(limit=10, mode="response", responses_used=used)
    budget._time_left_clock_start = 0.0
    budget._time_left_decisions_at_start = 0
    visible = countdown.render_time_slot(budget, seconds)
    assert expected in visible
    if seconds is None:
        assert visible == "unbounded"
    else:
        assert len(visible.encode("utf-8")) == countdown.TIME_SLOT_WIDTH
    line = f"[CONTINUATION LIMITS | stateframe-continuation-limits/1] sandbox_seconds_remaining={visible}; next_action_admission=allowed"
    canonical = (line if seconds is None else
                 line.replace(visible, countdown.CANONICAL_TIME_SLOT))
    assert token_estimate(line) == token_estimate(canonical)
    assert countdown.canonical_countdown(line) == canonical


@pytest.mark.parametrize(("ledger_type", "terminal_field"), (
    (ResponseBudget, ""),
    (ContinuationLimits, "funded_terminal_responses_remaining=0; "),
))
def test_no_deadline_limits_keep_prior_wire_and_measurement(
    ledger_type: type[ResponseBudget], terminal_field: str,
) -> None:
    prior_limits = (
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        "model_decisions_remaining=10; tool_attempts_remaining=10; "
        "automatic_checks_remaining=10; terminal_reserves_remaining=2; "
        f"{terminal_field}output_reservation_tokens=4000; "
        "sandbox_seconds_remaining=unbounded; next_action_admission=allowed; "
        "cap_stop_reasons=model_decisions|tool_attempts|automatic_checks|"
        "terminal_reserve|output_reservation|sandbox_deadline"
    )
    rendered = ledger_type(limit=10).render_continuation_limits()
    assert rendered == prior_limits
    assert countdown.canonical_countdown(rendered) == prior_limits
    body = {"messages": [{"role": "user", "content": rendered}], "tools": []}
    assert countdown.canonical_countdown(body) is body
    assert token_estimate(body) == (
        len(json.dumps({"messages": [{"role": "user", "content": prior_limits}],
                        "tools": []}, sort_keys=True, ensure_ascii=False)) + 3
    ) // 4
    seen: list[str] = []

    def tokenizer(content: str) -> list[int]:
        seen.append(content)
        return [1]

    assert RequestTokenMeter(tokenizer=tokenizer).measure(body)[1] == "tokenize"
    assert seen == [prompt_projection(body)]


def test_canonical_projection_leaves_non_numeric_limits_unchanged() -> None:
    line = (
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        "sandbox_seconds_remaining=pending; next_action_admission=allowed"
    )
    assert countdown.canonical_countdown(line) is line


@pytest.mark.parametrize("ledger_type", [ResponseBudget, ContinuationLimits])
def test_exhausted_limits_keep_the_reserved_time_slot(
    monkeypatch: pytest.MonkeyPatch, ledger_type: type[ResponseBudget],
) -> None:
    monkeypatch.setattr(countdown.time, "monotonic", lambda: 1.0)
    rendered = ledger_type(limit=0, sandbox_deadline_monotonic=0.0).render_continuation_limits()
    assert f"sandbox_seconds_remaining={countdown.render_time_slot(object(), 0)};" in rendered
    assert "next_action_admission=denied" in rendered


def test_slow_clock_does_not_change_source_carrier_census_or_other_wire_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = "pkg/sample.py"
    source = "".join(f"def method_{line:02d}(): return {line}\n" for line in range(1, 64))
    full = f"[Current source: {path}]\n{source}"
    budget = ResponseBudget(limit=10, mode="response", responses_used=1)
    budget._time_left_clock_start = 0.0
    budget._time_left_decisions_at_start = 0
    budget._sandbox_seconds_remaining = lambda now=None: 600.0
    runtime = SimpleNamespace()

    packet = census.CensusPacket(
        schema="test", workspace_namespace="test", index_revision="test",
        extractor_versions=(), route_id="test", decision_kind="edit",
        selected_source_handles=(), source_span_set_digest="test", coverage={},
        groups=({"sites": ({"path": path, "shown": True},)},), render={},
        payload="[host source relations; static call sites of constructors used in the shown source]\n[/host source relations]",
        cache_key="test",
    )
    monkeypatch.setattr(census, "_workspace_mode", lambda *_args: ("edit", "on"))
    monkeypatch.setattr(census, "_root", lambda _runtime: tmp_path)

    def build_packet(_runtime: object, _root: Path,
                     shown: Sequence[Mapping[str, object]], **_kwargs: object) -> object:
        # The planted site is available only when the complete carrier is shown.
        return packet if any(message.get("content") == full for message in shown) else None

    monkeypatch.setattr(census, "_build_packet", build_packet)

    def counter(value: object) -> int:
        # A stand-in for a tokenizer whose count varies with equal-width digits.
        return token_estimate(value) + json.dumps(value).count("0")

    ceiling_bytes: int | None = None

    def render_and_fit(now: float, cap: int) -> tuple[str, str, str, object]:
        nonlocal ceiling_bytes
        monkeypatch.setattr(countdown.time, "monotonic", lambda: now)
        limits = budget.render_continuation_limits()
        messages = [{"role": "user", "content": limits},
                    {"role": "user", "content": full}]
        non_source = {"messages": [messages[0]], "tools": []}
        # Full source sits exactly at the carrier boundary for the first clock.
        if ceiling_bytes is None:
            ceiling_bytes = len(json.dumps(non_source).encode()) + 768 + len(full.encode())
        source_window._window_path_messages(
            messages, runtime, path=path, source=source,
            max_request_bytes=ceiling_bytes, tools=(), changed_paths=[], is_target=True,
        )
        delivery = census.prepare_request(
            runtime, SimpleNamespace(), messages, (), frame_ceiling=cap,
            request_token_counter=counter,
        )
        wire = json.dumps({"messages": delivery.messages, "tools": []})
        return limits, str(messages[1]["content"]), wire, delivery

    _, _, _, probe = render_and_fit(100.0, 100_000)
    assert probe.status == "fired"
    cap = counter(countdown.canonical_countdown({"messages": probe.messages, "tools": []}))
    first_limits, first_carrier, first_wire, first = render_and_fit(100.0, cap)
    second_limits, second_carrier, second_wire, second = render_and_fit(1.0, cap)
    assert first_limits != second_limits
    assert [first.status, second.status] == ["fired", "fired"]
    assert first_carrier == second_carrier == full
    assert countdown.canonical_countdown(json.loads(first_wire)) == countdown.canonical_countdown(json.loads(second_wire))


def test_token_meter_counts_canonical_slot_without_changing_wire() -> None:
    seen: list[str] = []

    def tokenizer(content: str) -> list[int]:
        seen.append(content)
        return [1] * (len(content) + content.count("9"))

    meter = RequestTokenMeter(tokenizer=tokenizer)
    bodies = [
        {"messages": [{"role": "user", "content":
            f"[CONTINUATION LIMITS | stateframe-continuation-limits/1] sandbox_seconds_remaining={slot}; x=1"}],
         "tools": []}
        for slot in ("00060.0 (about 0001 minutes left, ~0001 turns at your pace)",
                     "00060.0 (about 0001 minutes left, ~999+ turns at your pace)")
    ]
    results = [meter.measure(body) for body in bodies]
    assert results[0] == results[1]
    assert len(seen) == 1
    assert meter.count_text_exact(bodies[0]["messages"][0]["content"]) == meter.count_text_exact(
        bodies[1]["messages"][0]["content"])
    assert bodies[0] != bodies[1]


@pytest.mark.parametrize("header", (
    "[Current source: pkg/sample.py]", "CURRENT_SOURCE pkg/sample.py",
    "[Model source window]",
))
def test_wire_stability_moves_only_host_limits_and_preserves_source_text(header: str) -> None:
    source = (
        f"{header}\n"
        "sandbox_seconds_remaining=source_literal\n"
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        "sandbox_seconds_remaining=source_literal; next_action_admission=source_literal\n"
    )
    assert countdown.canonical_countdown(source) == source
    source_body = {"messages": [{"role": "user", "content": source}]}
    assert countdown.canonical_countdown(source_body) is source_body
    budget = ResponseBudget(limit=10)
    budget._sandbox_seconds_remaining = lambda now=None: 60.0
    limits = budget.render_continuation_limits()
    assert "00060.0 (about 0001 minutes left" in limits
    assert wire_stability._tail_category(limits) == "time"
    assert wire_stability._volatile_segments(limits) == [("limits", limits)]
    projected = wire_stability.project({"messages": [
        {"role": "user", "content": source},
        {"role": "user", "content": limits},
    ]}, environ={wire_stability.WIRE_STABLE_ENV: "on"})
    messages = projected.body["messages"]
    assert messages[0]["content"] == source
    assert messages[-1]["content"].startswith(wire_stability.VOLATILE_TAIL_MARKER)
    assert limits in messages[-1]["content"]
    changed_action = limits.replace("next_action_admission=allowed", "next_action_admission=denied")
    assert countdown.canonical_countdown(changed_action) != countdown.canonical_countdown(limits)
