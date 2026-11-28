import copy
import json
from unittest.mock import Mock

import pytest

import kernel_request_diet as diet


def count_tokens(text):
    return len(text) // 3


def apply(body):
    return diet.apply_diet(body, stage="FREE", max_request_bytes=100000, token_budget=90000,
                           token_meter=diet.RequestTokenMeter(tokenizer=count_tokens))


@pytest.mark.parametrize("allocator", ["on", "off"])
def test_allocator_skips_normalizer_and_matches_final_wire(monkeypatch, allocator):
    original = {"messages": [{"role": "user", "content": "keep Unicode β"}] * 2, "tools": []}
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", allocator)
    normalize = Mock(wraps=diet._normalized_messages)
    monkeypatch.setattr(diet, "_normalized_messages", normalize)
    results = []
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_ALLOCATOR_SKIP_NORMALIZATION", mode)
        body = copy.deepcopy(original)
        normalize.reset_mock()
        audit = apply(body)
        audit.pop("tokenize_latency_ms")
        results.append((json.dumps(body).encode(), audit))
        assert normalize.call_count == int(allocator == "off" or mode == "off")
    assert results[0] == results[1]
    expected = original if allocator == "on" else {**original, "messages": original["messages"][:1]}
    assert results[0][0] == json.dumps(expected).encode()


@pytest.mark.parametrize("invalid", [{"messages": {}}, {"messages": [None]},
                                     {"messages": [{"content": object()}]}])
def test_allocator_off_and_invalid_inputs_keep_validation(monkeypatch, invalid):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    errors = []
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_ALLOCATOR_SKIP_NORMALIZATION", mode)
        with pytest.raises((TypeError, AttributeError)) as caught:
            apply(copy.deepcopy(invalid))
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]
