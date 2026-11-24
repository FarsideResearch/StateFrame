"""Behavior and scaling tests for the DSML reply scanner."""

import json
import random
import time
from pathlib import Path

import pytest

from kernel_dsml_tool_calls import parse_dsml


_CORPUS = json.loads(Path(__file__).with_name("dsml_tool_calls_corpus.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("case_name", "case"), list(_CORPUS.items()), ids=list(_CORPUS)
)
def test_corpus_case(case_name, case):
    assert parse_dsml(case["input"]) == case["output"]


def _call(name, body=""):
    return f'<｜DSML｜invoke name="{name}">{body}</｜DSML｜invoke>'


def _param(name, value, mode="true"):
    return f'<｜DSML｜parameter name="{name}" string="{mode}">{value}</｜DSML｜parameter>'


@pytest.mark.parametrize("bad_open", [
    '<｜DSML｜invoke>',
    '<｜DSML｜invoke name="">',
    "<｜DSML｜invoke name='ignored'>",
    '<｜DSML｜invoke other="x" name="ignored">',
])
def test_unusable_invocation_openers_are_skipped(bad_open):
    assert parse_dsml(bad_open + _call("later")) == [
        {"id": "dsml_0", "function": {"name": "later", "arguments": "{}"}}
    ]


def test_unclosed_header_quote_recovers_at_later_opener():
    broken = '<｜DSML｜invoke name="unfinished '
    assert parse_dsml(broken + _call("recovered"))[0]["function"]["name"] == "recovered"


def test_closed_header_quote_can_contain_angle_brackets():
    result = parse_dsml(_call("left<inside>right"))
    assert result[0]["function"]["name"] == "left<inside>right"


def test_every_prefix_of_three_calls_is_monotonic():
    reply = (
        "<｜DSML｜tool_calls>"
        + _call("first", _param("v", "one"))
        + _call("second", _param("v", "2", "false"))
        + _call("third")
        + "</｜DSML｜tool_calls>"
    )
    previous = []
    for end in range(len(reply) + 1):
        current = parse_dsml(reply[:end])
        assert current[:len(previous)] == previous
        previous = current
    assert len(previous) == 3


def test_partial_nested_and_duplicate_parameters():
    partial = '<｜DSML｜invoke name="unfinished"><｜DSML｜parameter name="a" string="true">x'
    assert parse_dsml(partial) == []

    nested = _call("outer", _call("inner"))
    result = parse_dsml(nested)
    assert [item["function"]["name"] for item in result] == ["outer"]
    assert result[0]["function"]["arguments"] == "{}"

    duplicate = _call("dup", _param("x", "first") + _param("x", "last"))
    args = json.loads(parse_dsml(duplicate)[0]["function"]["arguments"])
    assert args == {"x": "last"}

    missing_parameter_close = _call(
        "close", '<｜DSML｜parameter name="x" string="true">value'
    )
    assert json.loads(parse_dsml(missing_parameter_close)[0]["function"]["arguments"]) == {}


def test_json_fallback_nan_unicode_and_none():
    reply = _call(
        "values",
        _param("bad", "{bad", "false")
        + _param("nan", "NaN", "false")
        + _param("unicode", "é漢😀"),
    )
    result = parse_dsml(reply)
    assert result[0]["function"]["arguments"] == (
        '{"bad": "{bad", "nan": NaN, "unicode": "\\u00e9\\u6f22\\ud83d\\ude00"}'
    )
    assert parse_dsml(None) == []


def test_large_raw_value_is_preserved():
    value = "x" * (512 * 1024)
    result = parse_dsml(_call("large", _param("text", value)))
    assert json.loads(result[0]["function"]["arguments"]) == {"text": value}


def _cpu_best(text, repeats=7):
    samples = []
    for _ in range(repeats):
        started = time.process_time_ns()
        parse_dsml(text)
        samples.append(time.process_time_ns() - started)
    return min(samples)


def _to_size(token, size):
    return (token * (size // len(token) + 1))[:size]


@pytest.mark.parametrize(("size", "build"), [
    (1 << 18, lambda n: _to_size('<｜DSML｜invoke name="x">', n)),
    (10_000, lambda n: '<｜DSML｜invoke ' * n),
    (50_000, lambda n: '<｜DSML｜invoke' + ' ' * n + 'name="x"></｜DSML｜invoke>'),
    (4096, lambda n: '｜DSML｜' * n),
    (4096, lambda n: '<｜DSML｜unknown' + '<' * (n // 4)),
    (10_000, lambda n: '<｜DSML｜invoke name="unfinished' * n),
    (1 << 18, lambda n: _to_size('</｜DSML｜invoke>', n)),
], ids=["openers-no-close", "10k-broken-openers", "200k-header-spaces", "markers",
        "stray-less-than", "10k-unclosed-quotes", "closers-no-openers"])
def test_linear_time_growth_ratio(size, build):
    small, large = build(size), build(size * 4)
    ratio = _cpu_best(large) / max(_cpu_best(small), 1)
    assert ratio < 6


def test_random_fuzz_is_deterministic_and_never_raises():
    rng = random.Random(18473)
    alphabet = "abc<>/=\"'｜DSML｜ invoke parameter tool_calls\n😀"
    samples = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(300))) for _ in range(500)]
    for sample in samples:
        assert parse_dsml(sample) == parse_dsml(sample)


def test_streamed_prefixes_never_emit_a_partial_call():
    reply = _call("read", _param("path", "a.py")) + _call("run", _param("n", "3", "false"))
    final = parse_dsml(reply)
    assert len(final) == 2
    for end in range(len(reply)):
        assert parse_dsml(reply[:end]) == final[: reply[:end].count("</｜DSML｜invoke>")]


def test_native_entry_point_delegates_and_keeps_the_marker_rule():
    from probes.react_repair import parse_dsml_tool_calls

    reply = _call("read", _param("path", "a.py"))
    assert parse_dsml_tool_calls(reply) == parse_dsml(reply)
    assert parse_dsml_tool_calls("") == []
    assert parse_dsml_tool_calls(reply[:-3]) == []
