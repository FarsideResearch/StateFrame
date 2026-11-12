"""Request token metering reuses the tokenization projection of an unchanged body."""
from __future__ import annotations

import copy

import kernel_request_diet as diet
from kernel_request_diet import RequestTokenMeter


def _body(n: int = 6) -> dict[str, object]:
    limits = ("[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
              "sandbox_seconds_remaining=1234.5 (about 0021 minutes left, ~0005 turns at your pace);")
    return {"model": "m", "messages": [
        {"role": "system", "content": "sys"},
        *({"role": "user", "content": f"message {i} " + "x" * 300} for i in range(n)),
        {"role": "user", "content": limits},
    ], "tools": []}


def _meter(calls: list[str]) -> RequestTokenMeter:
    def tok(text: str) -> list[int]:
        calls.append(text)
        return list(range(len(text) // 4))
    return RequestTokenMeter(tok)


def test_unchanged_body_projects_and_tokenizes_once(monkeypatch):
    builds = []
    import kernel_request_measurement as measurement
    real = measurement.prompt_projection_body
    monkeypatch.setattr(measurement, "prompt_projection_body",
                        lambda body: builds.append(1) or real(body))
    calls: list[str] = []
    meter, body = _meter(calls), _body()
    results = [meter.measure(body) for _ in range(5)]
    assert len(set(results)) == 1
    assert len(builds) == 1
    assert len(calls) == 1


def test_content_edit_changes_count_and_projection():
    calls: list[str] = []
    meter, body = _meter(calls), _body()
    before = meter.measure(body)[0]
    body["messages"][1]["content"] += "y" * 4000  # type: ignore[index]
    after = meter.measure(body)[0]
    assert after > before
    assert len(calls) == 2
    body["messages"].append({"role": "user", "content": "z" * 400})  # type: ignore[attr-defined]
    assert meter.measure(body)[0] > after


def test_memoized_projection_equals_fresh_for_each_trim_step():
    meter = _meter([])
    body = _body()
    while len(body["messages"]) > 2:  # type: ignore[arg-type]
        assert diet._tokenization_content(body) == diet._tokenization_content(
            copy.deepcopy(body))
        meter.measure(body)
        body["messages"].pop(1)  # type: ignore[attr-defined]


def test_countdown_slot_still_canonical():
    a, b = _body(), _body()
    b["messages"][-1]["content"] = b["messages"][-1]["content"].replace("1234.5", "0001.0")  # type: ignore[index]
    assert diet._tokenization_content(a) == diet._tokenization_content(b)


def test_replaced_countdown_pattern_is_not_served_from_memo(monkeypatch):
    import re

    import kernel_countdown_slot as countdown
    body = _body()
    canonical = diet._tokenization_content(body)
    monkeypatch.setattr(countdown, "_TIME_FIELD", re.compile(r"(?!x)x"))
    assert diet._tokenization_content(body) != canonical
