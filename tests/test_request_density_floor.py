"""Offline checks for the hosted, raise-only decoded-content cold floor."""

from __future__ import annotations

import base64
import math
import random

import pytest

from kernel_frame_allocation_capture import calibrated_estimate
from kernel_request_diet import (
    INITIAL_BYTES_PER_TOKEN,
    TEMPLATE_OVERHEAD_TOKENS,
    RequestTokenMeter,
    _density_floor_tokens,
    _wire_bytes,
)


def _body(content: str, *, tools: list[object] | None = None) -> dict[str, object]:
    return {"model": "hosted", "messages": [{"role": "user", "content": content}],
            "tools": tools or []}


def _old_cold(body: dict[str, object]) -> int:
    return math.ceil(_wire_bytes(body) / INITIAL_BYTES_PER_TOKEN) + TEMPLATE_OVERHEAD_TOKENS


@pytest.mark.parametrize("content,active", [
    ("1234567890" * 1200, True),
    (base64.b64encode(random.Random(173).randbytes(9000)).decode(), True),
    ("\n".join(f"a{i} = b{i} + c{i} * 1234567890" for i in range(700)), True),
    ("[]{}():,;=+-*/!@#$%^&|<>?" * 600, True),
    ("漢字かな交じりの文章" * 500, False),
    ("🙂🚀" * 500, False),
])
def test_character_classes_keep_hosted_and_frame_estimates_above_cold(
        content: str, active: bool,
) -> None:
    body = _body(content)
    meter = RequestTokenMeter(disabled=True)
    tokens, method = meter.measure(body)
    assert method == "calibrated_bytes"
    assert tokens == max(_old_cold(body), _density_floor_tokens(body, TEMPLATE_OVERHEAD_TOKENS))
    assert tokens >= _old_cold(body)
    assert (_density_floor_tokens(body, TEMPLATE_OVERHEAD_TOKENS) > _old_cold(body)) is active
    assert calibrated_estimate(body, meter) >= _density_floor_tokens(body, TEMPLATE_OVERHEAD_TOKENS)


def test_tool_schema_density_is_counted() -> None:
    tool = {"type": "function", "function": {"name": "lookup", "description": "Search.",
            "parameters": {"type": "object", "properties": {
                f"p_{i}": {"type": "integer", "description": "identifier"} for i in range(40)}}}}
    body = _body("Choose a tool.", tools=[tool] * 60)
    assert _density_floor_tokens(body, TEMPLATE_OVERHEAD_TOKENS) > _density_floor_tokens(
        _body("Choose a tool."), TEMPLATE_OVERHEAD_TOKENS)
    assert RequestTokenMeter(disabled=True).measure(body)[0] >= _old_cold(body)


@pytest.mark.parametrize("content", [
    "Please summarize this discussion and explain the next steps in plain language. " * 50,
    "The checkout code handles prices, stock, tax, and rewards for each order. " * 100,
])
def test_ordinary_prose_floor_stays_inactive(content: str) -> None:
    body = _body(content)
    assert _density_floor_tokens(body, TEMPLATE_OVERHEAD_TOKENS) <= _old_cold(body)
    assert RequestTokenMeter(disabled=True).measure(body)[0] == _old_cold(body)


def test_real_corpus_integer_snapshots_cover_prompt_truth() -> None:
    # Captured request byte count, rendered/server prompt truth, and decoded
    # class-floor count. These rows come from each held-out corpus, without
    # any external capture dependency in the shipped test suite.
    rows = (
        ("qwen36_local:csv_adapter:1", 57748, 14620, 16730),
        ("deepseek_w2:tiny_site:1", 36248, 9332, 10948),
        ("deepseek_w1:002", 57003, 14502, 17107),
        ("qwen3_deepinfra:002", 57003, 14679, 17107),
    )
    for label, wire_bytes, truth, floor in rows:
        old = math.ceil(wire_bytes / INITIAL_BYTES_PER_TOKEN) + TEMPLATE_OVERHEAD_TOKENS
        assert max(old, floor) >= truth, label


def test_random_dense_sweep_is_raise_only() -> None:
    rng = random.Random(0xF5)
    alphabet = ("The quick brown fox checks the server. ", "0123456789",
                "QWxhZGRpbjpPcGVuU2VzYW1lMTIzNDU2Nzg5+/=", "a1 = b2 + c3;\n",
                '{"properties":{"value":"integer"}}', "漢字仮名交じり", "🙂🚀",
                "()[]{}:;,./+-*=!@#$%^&|<>?")
    meter = RequestTokenMeter(disabled=True)
    for index in range(10_000):
        unit = alphabet[index % len(alphabet)]
        # Most samples are short; every class also reaches 120 KB.
        count = rng.randrange(0, 400)
        if index < len(alphabet):
            count = 120_000 // len(unit.encode())
        content = unit * count
        if index % len(alphabet) == 4:
            body = _body("Choose a tool.", tools=[{"type": "function", "function": {
                "name": "lookup", "description": content,
                "parameters": {"type": "object", "properties": {
                    "value": {"type": "string"}}}}}])
        else:
            body = _body(content)
        new, method = meter.measure(body)
        assert method == "calibrated_bytes"
        assert new >= _old_cold(body)
