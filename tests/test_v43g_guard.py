"""Synthetic contracts for length-capped repeated-tail detection."""
from __future__ import annotations

import json
import random
import string
import sys

import pytest

import kernel_v43g_guard as guard


def _response(
    *,
    finish_reason: str | None = "length",
    content: str = "",
    reasoning: str = "",
    tool_arguments: tuple[str, ...] = (),
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_arguments:
        message["tool_calls"] = [
            {"id": f"call-{index}", "type": "function", "function": {
                "name": "inspect", "arguments": value,
            }}
            for index, value in enumerate(tool_arguments)
        ]
    return {"choices": [{"finish_reason": finish_reason, "message": message}]}


def _cycle(parts: tuple[str, ...], *, repetitions: int = 20, partial: bool = True) -> str:
    unit = " ".join(parts)
    content = "\n\t".join([unit] * repetitions)
    if partial:
        content += "\r\n  " + unit[: max(1, len(unit) // 2)]
    return content


def _normal_finish_shapes() -> tuple[str, ...]:
    rows = range(70)
    return (
        "| event_key | outcome | details |\n" + "\n".join(
            f"| event-{i:03d} | ready | queued task from common workflow {i:03d} |" for i in rows),
        json.dumps([{"type": "audit-event", "description": "shared event processing record", "id": i}
                    for i in rows]),
        "\n".join(
            f"def transform_record_{i:03d}(value):\n    return normalize_shared_record(value, {i})"
            for i in rows),
        "\n".join(
            f"2026-11-23T12:00:{i % 60:02d}Z INFO shared request accepted by scheduler id={i:03d}"
            for i in rows),
        "\n".join(
            f"/workspace/cache/shared-project/component-{i:03d}/records/output.json" for i in rows),
        "\n".join(
            f"from shared_namespace.module_{i:03d} import SharedRecord_{i:03d}" for i in rows),
        "\n".join(
            f"+    result_{i:03d} = normalize_shared_record(input_{i:03d})" for i in rows),
        "record_id,description,status\n" + "\n".join(
            f"{i:03d},shared record processing detail {i:03d},ready" for i in rows),
        "\n".join(
            f"customer_identifier_shared_namespace_{i:06d}" for i in rows),
        "\n".join(
            f"- record: {i:03d}\n  description: shared queue processing detail {i:03d}"
            for i in rows),
        "\n".join(
            f"  File \"/workspace/shared/service/worker_{i:03d}.py\", line {i + 10}, in process_record"
            for i in rows),
        "\n".join(
            "INSERT INTO shared_audit_events "
            "(event_kind, processing_status, record_id) "
            f"VALUES ('queue', 'ready', {i:03d});"
            for i in rows),
    )


@pytest.mark.parametrize("shape_index", range(12))
@pytest.mark.parametrize("with_tool_call", [False, True])
def test_normal_finishes_with_repetitive_shapes_are_never_discarded(
        shape_index: int, with_tool_call: bool) -> None:
    shape = _normal_finish_shapes()[shape_index]
    finish_reason = "tool_calls" if with_tool_call else "stop"
    arguments = (json.dumps({"shape": shape}),) if with_tool_call else ()
    assert guard.detect_response(_response(
        finish_reason=finish_reason, content=shape, tool_arguments=arguments)) is None


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_normal_finish_with_a_real_repeated_tail_is_not_a_runaway(finish_reason: str) -> None:
    loop = "the generated answer has stopped making progress " * 24
    arguments = (loop,) if finish_reason == "tool_calls" else ()
    assert guard.detect_response(_response(
        finish_reason=finish_reason, content=loop, tool_arguments=arguments)) is None


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "max_output_tokens"])
def test_every_named_token_cap_reason_retries_a_runaway(finish_reason: str) -> None:
    loop = "the same capped output is repeating " * 24
    finding = guard.detect_response(_response(finish_reason=finish_reason, content=loop))
    assert finding is not None
    assert finding.source == "content"


@pytest.mark.parametrize("finish_reason", [["length"], {"max_tokens": True}, 1])
def test_a_malformed_finish_reason_never_crashes_or_discards(finish_reason: object) -> None:
    loop = "the same capped output is repeating " * 24
    assert guard.detect_response(_response(finish_reason=finish_reason, content=loop)) is None


@pytest.mark.parametrize("finish_reason", [None, "unknown", "stop", "tool_calls"])
def test_only_normalized_length_finish_reason_can_be_discarded(finish_reason: str | None) -> None:
    loop = "the same capped output is repeating " * 24
    assert guard.detect_response(_response(finish_reason=finish_reason, content=loop)) is None


@pytest.mark.parametrize("parts", [
    ("the model repeats one complete phrase with no new content",),
    ("first alternating phrase with distinctive content", "second alternating phrase differs"),
    ("third loop unit alpha", "third loop unit beta", "third loop unit gamma"),
])
def test_length_capped_one_two_and_three_phrase_cycles_are_runaways(
        parts: tuple[str, ...]) -> None:
    finding = guard.detect_response(_response(content=_cycle(parts)))
    assert finding is not None
    assert finding.source == "content"
    assert finding.repetition_count >= guard.MIN_REPETITIONS - 1


def test_length_capped_long_paragraph_cycle_supports_2000_character_units() -> None:
    paragraphs = tuple(
        (f"Paragraph {index} describes a distinct process with several concrete steps. "
         + "The worker checks each input, records useful evidence, and reports progress. " * 30)[:2000]
        for index in range(3)
    )
    assert all(len(paragraph) == 2000 for paragraph in paragraphs)
    finding = guard.detect_response(_response(content=_cycle(paragraphs)))
    assert finding is not None
    assert finding.repeated_span_chars <= guard.MAX_REPEAT_SPAN_CHARS


def test_length_capped_whitespace_drift_and_final_partial_unit_are_tolerated() -> None:
    unit = "alternating whitespace does not add new answer content"
    separators = ("\n", "  ", "\r\n\t", "\t ")
    content = "".join(unit + separators[index % len(separators)] for index in range(24))
    content += unit[:17]
    finding = guard.detect_response(_response(content=content))
    assert finding is not None
    assert finding.source == "content"


@pytest.mark.parametrize("channel", ["reasoning", "content"])
def test_length_capped_loop_in_either_response_channel_is_detected(channel: str) -> None:
    loop = "the same reasoning or content unit repeats " * 24
    response = _response(
        reasoning=loop if channel == "reasoning" else "new reasoning arrives here",
        content=loop if channel == "content" else "new content arrives here",
    )
    finding = guard.detect_response(response)
    assert finding is not None
    assert finding.source == channel


def test_length_capped_loop_inside_truncated_tool_call_argument_is_detected() -> None:
    unit = "the serialized tool argument is stuck on this value "
    truncated_arguments = '{"payload": "' + unit * 23 + unit[:19]
    finding = guard.detect_response(_response(tool_arguments=(truncated_arguments,)))
    assert finding is not None
    assert finding.source == "tool_arguments"


def test_length_capped_repeated_lone_surrogate_does_not_crash_span_hashing() -> None:
    content = json.loads('"\\ud800abcdefgh"') * 24
    finding = guard.detect_response(_response(content=content))
    assert finding is not None
    assert finding.source == "content"


@pytest.mark.parametrize("content", [
    "\n".join(f"row {i:04d}: shared enumeration description differs at item {i:04d}" for i in range(160)),
    "\n".join(f"step {i:04d}: new decision and new output value {i * 7919}" for i in range(160)),
    "\n".join(f"iteration {i:04d} found a distinct path under module_{i:04d}" for i in range(160)),
])
def test_length_capped_replies_with_genuine_progress_are_not_discarded(content: str) -> None:
    assert guard.detect_response(_response(content=content)) is None


def test_repeated_middle_section_does_not_count_when_reply_ends_with_new_content() -> None:
    content = "old section: " + "repeated middle content " * 30 + "new conclusion with fresh facts"
    assert guard.detect_response(_response(content=content)) is None


def test_finding_keeps_the_recorded_evidence_fields() -> None:
    content = "the same phrase extends through the capped tail " * 24
    finding = guard.detect_response(_response(content=content))
    assert finding is not None
    assert set(finding.as_dict()) == {
        "schema", "source", "total_chars", "stop_chars", "saved_chars",
        "trailing_window_chars", "repeated_span_chars", "repetition_count",
        "repeated_chars", "span_hash",
    }
    assert finding.total_chars == len(content)
    assert finding.stop_chars <= finding.total_chars
    assert finding.span_hash.startswith("sha256:")


def test_seeded_random_cycles_detect_and_unique_items_make_progress() -> None:
    rng = random.Random(20261123)
    alphabet = string.ascii_lowercase + " ./,-"
    for _ in range(64):
        parts = tuple(
            "".join(rng.choice(alphabet) for _ in range(rng.randint(12, 42)))
            for _ in range(rng.randint(1, 3))
        )
        finding = guard.detect_response(_response(content=_cycle(parts)))
        assert finding is not None
        assert finding.source == "content"

        progress = "\n".join(
            f"record {index:04d} {''.join(rng.choice(alphabet) for _ in range(20))}"
            for index in range(80)
        )
        assert guard.detect_response(_response(content=progress)) is None


def _traced_work(text: str) -> int:
    target_codes = {
        guard.detect_text.__code__, guard._best_run.__code__,
        guard._normalize_whitespace.__code__, guard._z_values.__code__,
    }
    line_events = 0

    def trace(frame, event, _arg):
        nonlocal line_events
        if event == "line" and frame.f_code in target_codes:
            line_events += 1
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        guard.detect_text(text)
    finally:
        sys.settrace(previous_trace)
    return line_events


def test_detector_work_scales_linearly_at_ten_times_input() -> None:
    rng = random.Random(90210)
    small = "".join(rng.choice(string.ascii_letters) for _ in range(4000))
    large = "".join(rng.choice(string.ascii_letters) for _ in range(40000))
    small_work = _traced_work(small)
    large_work = _traced_work(large)
    assert small_work > 0
    assert large_work <= small_work * 12 + 100
