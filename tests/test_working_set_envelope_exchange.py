"""Priced envelope variants for execution replies and the narrative transcript.

With the allocator ON, the legacy fitters (fit_execution_replies and the
optional-message trim) are skipped, so an oversized check output or a long
narrative row used to leave the fixed envelope above the cap and preparation
returned FLOOR_INFEASIBLE with no source at fault. Preparation now prices
bounded exchange representations itself; the floor uses the most compact one.
"""
import hashlib
import json

from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_stop import ReasonCode


def source(key, text="x" * 100 + "\n", state="DUE", ordinal=1):
    identity = ReceiptIdentity(key + ".py", "sha256:" + "1" * 64, 1, 1,
        0, len(text.encode()), hashlib.sha256(text.encode()).hexdigest(), None)
    digests, numbers = line_digests_with_numbers(text, 1)
    return SourceSlot(key, (key,), identity, text, "[read page]\n" + text,
        digests, numbers, json.dumps({"path": key + ".py", "start_line": 1,
        "end_line": 1}).encode(), ordinal, ordinal, state)


CHECK_OUTPUT = "$ pytest -q\n" + "".join(f"FAILED tests/test_{i}.py::test_case - AssertionError\n"
                                        for i in range(1500))
assert len(CHECK_OUTPUT.encode()) > 60_000


def exchange(output=CHECK_OUTPUT, name="shell", narrative="working"):
    return [
        {"role": "assistant", "content": narrative,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": name, "arguments": json.dumps({"command": "pytest -q"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": output},
    ]


def snapshot(*sources, byte_cap=100_000, extra=(), tail=None):
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "Task: fix"},
                *extra,
                {"role": "user", "content": "opaque suffix"},
                *(exchange() if tail is None else tail)]
    body = {"model": "local", "messages": messages, "max_tokens": 100}
    return PreparationSnapshot(json.dumps(body).encode(), tuple(sources), "request", 1, byte_cap)


def tool_rows(wire):
    return [row for row in json.loads(wire)["messages"] if row.get("role") == "tool"]


def test_large_check_output_is_bounded_instead_of_blocking():
    snap = snapshot(source("a"), byte_cap=20_000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), getattr(result, "reason", None)
    assert result.total_bytes <= snap.byte_cap
    assert result.floor_bytes <= snap.byte_cap
    (reply,) = tool_rows(result.wire)
    assert reply["content"].startswith("$ pytest -q\n")
    assert "[execution output delivery:" in reply["content"]
    assert "observation bytes omitted" in reply["content"]
    assert "full observation in tool-tape.jsonl call call_1" in reply["content"]
    assert hashlib.sha256(CHECK_OUTPUT.encode()).hexdigest() in reply["content"]
    assert "FAILED tests/test_1499.py" in reply["content"]  # the tail of the output survives
    # The DUE source is still delivered in full ahead of the bounded reply.
    rows = json.loads(result.wire)["messages"]
    assert any(str(row.get("content", "")).startswith("[read page]\n" + "x" * 100) for row in rows)


def test_rich_exchange_is_preferred_when_it_fits():
    snap = snapshot(source("a"), byte_cap=200_000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate)
    (reply,) = tool_rows(result.wire)
    assert reply["content"] == CHECK_OUTPUT
    assert result.floor_bytes < result.total_bytes


def test_floor_infeasible_only_when_the_most_compact_exchange_cannot_fit():
    snap = snapshot(source("a"), byte_cap=1_000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    result.proof.validate()
    # The impossibility proof is built on the bounded exchange, not the flood.
    assert len(result.proof.mandatory_wire) < len(CHECK_OUTPUT.encode())
    assert "[execution output delivery:" in result.proof.mandatory_wire.decode()


def test_non_execution_tool_rows_are_never_bounded():
    page = "[read page]\n" + "y" * 70_000 + "\n"
    snap = snapshot(source("a"), byte_cap=20_000, tail=exchange(output=page, name="read"))
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    rows = json.loads(result.proof.mandatory_wire)["messages"]
    assert {"role": "tool", "tool_call_id": "call_1", "content": page} in rows


def test_long_narrative_rows_are_bounded_with_a_marker():
    narrative = "thinking out loud " * 4_000
    older = {"role": "assistant", "content": narrative}
    snap = snapshot(source("a"), byte_cap=20_000, extra=(older,),
                    tail=exchange(output="$ true\n"))
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), getattr(result, "reason", None)
    assert result.total_bytes <= snap.byte_cap
    rows = json.loads(result.wire)["messages"]
    bounded = next(row for row in rows if row.get("role") == "assistant" and "tool_calls" not in row)
    assert bounded["content"].startswith("thinking out loud ")
    assert "[transcript delivery:" in bounded["content"]
    assert hashlib.sha256(narrative.encode()).hexdigest() in bounded["content"]
    # Fixed carriers and the opaque suffix are untouched.
    assert {"role": "user", "content": "opaque suffix"} in rows
    assert {"role": "user", "content": "Task: fix"} in rows


def test_host_marker_user_rows_and_task_rows_are_never_bounded():
    marker_row = {"role": "user", "content": "[Current source: a.py]\n" + "z" * 70_000}
    snap = snapshot(source("a"), byte_cap=20_000, extra=(marker_row,), tail=exchange(output="$ true\n"))
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, Blocked)
    assert result.reason is not ReasonCode.PREPARATION_CONTRACT_FAILURE
    assert all("[transcript delivery:" not in str(trial) for trial in result.trials)


def test_small_exchanges_add_no_envelope_variants():
    import kernel_working_set_prepare as adapter
    snap = snapshot(source("a"), tail=exchange(output="$ true\nok\n"))
    variants = adapter._envelope_variants(snap)
    assert all("[execution output delivery:" not in variant.fixed_body.decode()
               and "[transcript delivery:" not in variant.fixed_body.decode()
               for variant in variants)
