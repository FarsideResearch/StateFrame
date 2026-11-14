"""Frozen exact admission pays for complete new segments, never net growth."""
from __future__ import annotations

import copy
import json
import math
import random
from types import SimpleNamespace

import kernel_cache_layout_candidate as layout
import pytest
from kernel_frame_allocation_capture import final_request_tokens
from kernel_frozen_delta_bound import FROZEN_DELTA_SEAM_TOKENS, frozen_delta_bound
from kernel_request_diet import RequestTokenMeter, _tokenization_content
from test_final_cache_layout import _body as layout_body
from test_seal_frozen_counter import _capture, _proof
from tools.frozen_delta_splices import TEXTS, DeterministicBPE, measure


@pytest.fixture
def layout_ready(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")


def _outage(body, *, count=None, overhead=None):
    meter = RequestTokenMeter(tokenizer=count or DeterministicBPE())
    if overhead is not None:
        meter.template_overhead_tokens = overhead
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    assert method == "tokenize"
    capture = _capture(body, meter, method=method, tokens=tokens)
    events = []
    capture.journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    meter._availability = "unavailable"
    return meter, capture, events


def _render(body, meter, capture, *, ceiling=33344, byte_ceiling=200000):
    return layout.render_final_cache_candidate(
        capture, None, body, meter=meter,
        inputs={"token_ceiling": ceiling, "byte_ceiling": byte_ceiling,
                "source_plan": {}, "sources": {}, "receipt": None})


def _admission(events):
    return next(fields for kind, fields in events if kind == "final_cache_layout_admission")


def test_real_layout_retains_exact_source_segments_and_carries_bound(layout_ready, monkeypatch):
    body = layout_body()
    body["messages"][3]["content"] += "x" * 80000
    original = copy.deepcopy(body)
    meter, capture, events = _outage(body, count=lambda _text: 1000)
    frozen_tokens = capture._frame_candidate_counter.tokens
    assert meter.measure(body)[0] > 33344
    projected = _render(body, meter, capture)
    assert projected is not body
    assert body == original
    accounting = _admission(events)
    assert accounting["path"] == "delta_bound"
    assert accounting["frozen_tokens"] == frozen_tokens
    assert accounting["delta_bytes"] < 3000  # The retained 78 KB carrier is not new text.
    assert accounting["seam_tokens"] == accounting["splice_points"] * FROZEN_DELTA_SEAM_TOKENS
    assert capture._frame_candidate_counter.tokens == accounting["prompt_tokens"]
    assert _proof(monkeypatch, projected, meter, capture)["status"] == "READY"


def test_oversized_delta_fails_open_with_plain_body_proof(layout_ready, monkeypatch):
    body = layout_body()
    meter, capture, events = _outage(body, count=lambda _text: 31000)
    frozen = capture._frame_candidate_counter
    projected = layout._checked_final_cache_projection(body)
    bound = frozen_delta_bound(body, projected, meter, frozen)
    assert bound is not None
    assert bound["prompt_tokens"] > 33344
    assert _render(body, meter, capture) is body
    assert capture._frame_candidate_counter is frozen
    assert _admission(events)["path"] == "delta_bound"
    failure = next(fields for kind, fields in events if kind == "final_cache_layout_failed_open")
    assert failure["reason"] == "exceeds_measured_envelope"
    assert failure["prompt_tokens"] == bound["prompt_tokens"]
    assert _proof(monkeypatch, body, meter, capture)["status"] == "READY"


@pytest.mark.parametrize("change", ["reorder", "tools", "role", "delete", "metadata"])
def test_unclean_outage_projection_keeps_whole_body_guard(layout_ready, monkeypatch, change):
    body = layout_body()
    body["messages"][3]["content"] += "x" * 80000
    meter, capture, events = _outage(body, count=lambda _text: 1000)
    changed = copy.deepcopy(body)
    if change == "reorder":
        changed["messages"][2:4] = list(reversed(changed["messages"][2:4]))
    elif change == "tools":
        changed["tools"] = [{"type": "function", "function": {"name": "other"}}]
    elif change == "role":
        changed["messages"][2]["role"] = "system"
    elif change == "delete":
        del changed["messages"][2]
    else:
        changed["messages"][2]["name"] = "other"
    assert frozen_delta_bound(body, changed, meter, capture._frame_candidate_counter) is None
    old_guard = max(final_request_tokens(changed, meter, capture)[0], meter.measure(changed)[0])
    assert old_guard > 33344
    monkeypatch.setattr(layout, "_checked_final_cache_projection", lambda _body: changed)
    assert _render(body, meter, capture) is body
    assert _admission(events)["path"] == "fallback"
    assert _admission(events)["prompt_tokens"] == old_guard
    assert _proof(monkeypatch, body, meter, capture)["status"] == "READY"


def test_replacement_counts_full_new_utf8_segment_and_minimum_ratio():
    body = {"messages": [{"role": "user", "content": "x" * 1000}], "tools": []}
    meter, capture, _events = _outage(body)
    changed = copy.deepcopy(body)
    changed["messages"][0]["content"] = "中文🙂\\\"" * 10
    meter._evictfix_token_pair = (1000, 200)
    meter._minimum_bytes_per_token = 0.3
    bound = frozen_delta_bound(body, changed, meter, capture._frame_candidate_counter)
    assert bound is not None
    expected = len(json.dumps(changed["messages"][0], sort_keys=True,
                              ensure_ascii=False, separators=(",", ":")).encode()) + 1
    assert expected < 1000  # Shrinking replacements must still pay for all their new bytes.
    assert bound["delta_bytes"] == expected
    assert bound["conservative_ratio"] == 0.2
    assert bound["delta_tokens"] == math.ceil(expected / 0.2)
    assert bound["splice_points"] == 2


def test_density_cap_covers_unseen_dense_delta():
    body = {"messages": [{"role": "user", "content": "x" * 1000}], "tools": []}
    meter, capture, _events = _outage(body)
    changed = copy.deepcopy(body)
    changed["messages"].append({"role": "user", "content": "qzjv" * 30})
    bound = frozen_delta_bound(body, changed, meter, capture._frame_candidate_counter)
    assert bound is not None
    assert capture._frame_candidate_counter.ratio > 1
    assert bound["conservative_ratio"] == 0.5
    assert bound["prompt_tokens"] >= DeterministicBPE()(_tokenization_content(changed)) + meter.template_overhead_tokens


def test_changed_base_cannot_borrow_another_body_exact_count():
    body = layout_body()
    meter, capture, _events = _outage(body)
    changed_base = copy.deepcopy(body)
    changed_base["messages"][1]["content"] += "different task"
    projected = layout._checked_final_cache_projection(changed_base)
    assert frozen_delta_bound(changed_base, projected, meter, capture._frame_candidate_counter) is None


def test_non_outage_real_guard_is_unchanged(layout_ready, monkeypatch):
    body = layout_body()

    def count(text):
        return 40000 if "stateframe-authority-overlay/1" in text else 1000

    meter, capture, events = _outage(body, count=count)
    meter._availability = "available"
    projected = layout._checked_final_cache_projection(body)
    current = meter.measure(projected)[0]
    assert current == 40000 + meter.template_overhead_tokens
    assert final_request_tokens(projected, meter, capture)[0] == current
    assert frozen_delta_bound(body, projected, meter, capture._frame_candidate_counter) is None
    assert _render(body, meter, capture) is body
    failure = next(fields for kind, fields in events if kind == "final_cache_layout_failed_open")
    assert failure["prompt_tokens"] == current
    assert _proof(monkeypatch, body, meter, capture)["status"] == "READY"


def test_seeded_admission_never_exceeds_real_bpe_ceiling(layout_ready, monkeypatch):
    count = DeterministicBPE()
    admitted = refused = 0
    for seed in range(240):
        rng = random.Random(seed)
        # Entire message spans are replaced, including edits inside words/JSON,
        # combining marks and four-byte UTF-8. Some cases have 100 tiny segments.
        contents = [rng.choice(TEXTS) for _ in range(100 if seed % 12 == 0 else rng.randint(1, 12))]
        body = {"messages": [{"role": "user", "content": text} for text in contents], "tools": []}
        changed = copy.deepcopy(body)
        for index in rng.sample(range(len(contents)), rng.randint(1, min(6, len(contents)))):
            text = contents[index]
            start = rng.randrange(len(text) + 1)
            end = rng.randrange(start, len(text) + 1)
            changed["messages"][index]["content"] = text[:start] + rng.choice(TEXTS) + text[end:]
        for _ in range(rng.randrange(6)):
            changed["messages"].append({"role": "user", "content": rng.choice(TEXTS)})
        room = rng.randrange(12000)
        overhead = 33344 - count(_tokenization_content(body)) - room
        assert overhead >= 0
        meter, capture, events = _outage(body, count=count, overhead=overhead)
        monkeypatch.setattr(layout, "_checked_final_cache_projection", lambda _body, result=changed: result)
        result = _render(body, meter, capture)
        accounting = _admission(events)
        if result is changed:
            admitted += 1
            assert accounting["path"] == "delta_bound"
            real_tokens = count(_tokenization_content(result)) + overhead
            assert real_tokens <= accounting["prompt_tokens"] <= 33344, seed
        else:
            refused += 1
            assert result is body
            assert accounting["prompt_tokens"] > 33344
    assert admitted > 100
    assert refused > 10


def test_measured_seam_margin_has_headroom():
    experiment = measure(DeterministicBPE())
    assert experiment["max_extra_tokens_per_splice"] == 16
    assert FROZEN_DELTA_SEAM_TOKENS >= 2 * experiment["max_extra_tokens_per_splice"]


def test_seeded_actual_layout_admission_bound(layout_ready):
    count = DeterministicBPE()
    admitted = refused = 0
    for seed in range(80):
        rng = random.Random(12059 + seed)
        body = layout_body(exchange=bool(seed % 2))
        body["messages"][3]["content"] += rng.choice(TEXTS) * rng.randint(1, 5)
        if seed % 4 == 0:
            body["messages"][3:3] = [
                {"role": "user", "content": f"[Current source: pkg/{index}.py]\n{text}"}
                for index, text in enumerate(TEXTS)]
        overhead = 33344 - count(_tokenization_content(body)) - rng.randrange(6000)
        meter, capture, events = _outage(body, count=count, overhead=overhead)
        result = _render(body, meter, capture)
        accounting = _admission(events)
        assert accounting["path"] == "delta_bound"
        if result is body:
            refused += 1
            assert accounting["prompt_tokens"] > 33344
        else:
            admitted += 1
            assert count(_tokenization_content(result)) + overhead <= accounting["prompt_tokens"] <= 33344, seed
    assert admitted > 10
    assert refused > 10


def test_altered_final_layout_cannot_claim_validated_segment_mapping():
    body = layout_body()
    meter, capture, _events = _outage(body)
    projected = layout._checked_final_cache_projection(body)
    projected["messages"][1:3] = list(reversed(projected["messages"][1:3]))
    assert frozen_delta_bound(body, projected, meter, capture._frame_candidate_counter) is None


def test_delta_must_also_fit_independent_byte_ceiling(layout_ready):
    body = layout_body()
    meter, capture, events = _outage(body)
    byte_ceiling = len(json.dumps(body).encode())
    assert _render(body, meter, capture, byte_ceiling=byte_ceiling) is body
    assert _admission(events)["path"] == "delta_bound"
    assert _admission(events)["prompt_tokens"] < 33344
    failure = next(fields for kind, fields in events if kind == "final_cache_layout_failed_open")
    assert failure["bytes"] > byte_ceiling


def test_delta_ceiling_boundary_is_inclusive(layout_ready, monkeypatch):
    body = {"messages": [{"role": "user", "content": "Task: fix it"}], "tools": []}
    changed = copy.deepcopy(body)
    changed["messages"].append({"role": "user", "content": "中文🙂"})
    monkeypatch.setattr(layout, "_checked_final_cache_projection", lambda _body: changed)
    for extra in (0, -1):
        meter, capture, events = _outage(body)
        bound = frozen_delta_bound(body, changed, meter, capture._frame_candidate_counter)
        assert bound is not None
        result = _render(body, meter, capture, ceiling=bound["prompt_tokens"] + extra)
        assert result is (changed if extra == 0 else body)
        assert _admission(events)["prompt_tokens"] == bound["prompt_tokens"]


def test_recovery_at_guard_uses_real_count(layout_ready, monkeypatch):
    body = layout_body()
    meter, capture, events = _outage(body)
    calls = []

    def measure_then_recover(value):
        calls.append(value)
        if len(calls) == 1:
            return 100, "calibrated_bytes"
        meter._availability = "available"
        return 40000, "tokenize"

    monkeypatch.setattr(meter, "measure", measure_then_recover)
    assert _render(body, meter, capture) is body
    assert len(calls) == 2
    assert _admission(events)["path"] == "live_exact"
    assert _admission(events)["prompt_tokens"] == 40000
