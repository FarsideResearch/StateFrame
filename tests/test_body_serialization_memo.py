"""The whole-request serialization memo must equal fresh serialization always."""
from __future__ import annotations

import copy
import json

import pytest

import kernel_body_serialization_memo as memo
from kernel_frame_optional_allocation import identity


def _bodies() -> list[dict[str, object]]:
    return [
        {"messages": [{"role": "user", "content": "hi é中"}], "tools": []},
        {"messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "read", "arguments": "{\"b\":1,\"a\":2}"}}]},
            {"role": "tool", "content": "x" * 5000, "tool_call_id": "c1"},
        ], "tools": [{"type": "function", "function": {"name": "read",
                      "parameters": {"z": [1, 2.5, True, None], "a": {}}}}],
         "model": "m", "n": 1000},
        {"messages": [], "tools": [], "k": 300},
    ]


def _fresh(body):
    return (json.dumps(body).encode(),
            len(json.dumps(body, sort_keys=True, ensure_ascii=False)),
            identity(body))


@pytest.mark.parametrize("body", _bodies())
def test_memo_matches_fresh_serialization_and_hits(body, monkeypatch):
    want = _fresh(body)
    seen = []

    def compute(value):
        seen.append(1)
        return identity(value)

    got = (memo.plain_wire(body), memo.sorted_chars(body),
           memo.canonical_digest(body, compute))
    assert got == want
    calls = 0
    real = json.dumps

    def counted(*a, **k):
        nonlocal calls
        calls += 1
        return real(*a, **k)

    monkeypatch.setattr(memo.json, "dumps", counted)
    assert (memo.plain_wire(body), memo.sorted_chars(body)) == want[:2]
    assert memo.canonical_digest(body, compute) == want[2]
    assert calls == 0 and len(seen) == 1


@pytest.mark.parametrize("mutate", [
    lambda b: b["messages"][0].__setitem__("content", "changed"),
    lambda b: b["messages"].append({"role": "user", "content": "more"}),
    lambda b: b["messages"].pop(),
    lambda b: b.__setitem__("model", "other"),
    lambda b: b.__setitem__("messages", list(reversed(b["messages"]))),
    lambda b: b["tools"].append({"x": 1}),
    lambda b: b["messages"][0].pop("role"),
])
def test_in_place_edit_never_returns_stale_value(mutate):
    body = copy.deepcopy(_bodies()[1])
    memo.plain_wire(body), memo.sorted_chars(body), memo.canonical_digest(body, identity)
    mutate(body)
    assert (memo.plain_wire(body), memo.sorted_chars(body),
            memo.canonical_digest(body, identity)) == _fresh(body)


def test_nested_edit_and_int_edit_invalidate():
    body = copy.deepcopy(_bodies()[1])
    memo.plain_wire(body)
    body["messages"][0]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert memo.plain_wire(body) == json.dumps(body).encode()
    body["n"] = 1001
    assert memo.plain_wire(body) == json.dumps(body).encode()


def test_unknown_types_bypass_memo_and_cache_is_bounded():
    body = {"messages": [], "x": {1, 2}}
    with pytest.raises(TypeError):
        memo.plain_wire(body)
    for i in range(50):
        memo.plain_wire({"i": i})
    assert len(memo._ENTRIES) <= memo._MAX_ENTRIES


def test_cyclic_body_raises_instead_of_hanging():
    cyc: dict[str, object] = {}
    cyc["c"] = cyc
    with pytest.raises(ValueError):
        memo.plain_wire(cyc)
    with pytest.raises(ValueError):
        memo.sorted_chars(cyc)


def test_over_node_cap_and_huge_text_bypass_memo(monkeypatch):
    monkeypatch.setattr(memo, "_MAX_NODES", 10)
    big = {"messages": [{"c": i} for i in range(20)]}
    assert memo.plain_wire(big) == json.dumps(big).encode()
    assert id(big) not in memo._ENTRIES
    monkeypatch.setattr(memo, "_MAX_NODES", 200_000)
    monkeypatch.setattr(memo, "_MAX_TEXT_CHARS", 100)
    huge = {"messages": [{"content": "y" * 500}]}
    assert memo.sorted_chars(huge) == _fresh(huge)[1]
    assert id(huge) not in memo._ENTRIES


def test_digest_cache_is_keyed_by_compute():
    body = {"a": 1}
    assert memo.canonical_digest(body, lambda _b: "one") == "one"
    assert memo.canonical_digest(body, lambda _b: "two") == "two"


class _S(str):
    pass


class _D(dict):
    pass


@pytest.mark.parametrize("body", [
    {"m": _S("xé"), "k": [1.5, True, False, None, 0.0, -0.0, 1e300, 7]},
    _D({"m": "x", "n": [1, 2]}),
    {"m": _D({"a": 1}), "l": [_S("q")]},
    {"f": [1.0, 2.5e-9, True, 3]},
])
def test_subclass_and_float_bool_leaves_match_fresh(body):
    for _ in range(2):
        assert (memo.plain_wire(body), memo.sorted_chars(body)) == _fresh(body)[:2]
