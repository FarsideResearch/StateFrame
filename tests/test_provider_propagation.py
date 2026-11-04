from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_provider_dialect import ProviderCapabilityError


def test_history_compaction_does_not_swallow_selected_capability_refusal(monkeypatch):
    import probes.history_compaction as history_compaction

    error = ProviderCapabilityError("history-compaction cache/slot", "llama.cpp")
    monkeypatch.setattr(history_compaction, "split_turns", lambda _messages: [{}, {}])
    compactor = history_compaction.HistoryCompactor(
        enabled=True, context_tokens=4096, threshold=1, keep_turns=1,
        summarize=lambda _messages: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ProviderCapabilityError, match="history-compaction cache/slot"):
        compactor.maybe_compact([{"role": "user", "content": "task"}], force=True)


def test_public_review_does_not_swallow_selected_capability_refusal(monkeypatch):
    import kernel_public_review as public_review

    error = ProviderCapabilityError("public-review cache_prompt", "llama.cpp")

    class Contract:
        def __init__(self, *_args):
            self.design = "design"
            self.bundle = {}
            self.review = "review"

        def draft(self, value):
            return value

        def final(self, value):
            return value

        def provenance(self, _value):
            return None

    monkeypatch.setattr(public_review, "public_bundle", lambda _row: {})
    monkeypatch.setattr(public_review, "PublicContract", Contract)
    runtime = SimpleNamespace(
        bound_public_verification=lambda: True,
        native_batch=None,
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    review = object.__new__(public_review.PublicReview)
    review.runtime = runtime
    review.attempted = False
    review.deadline_policy = None
    review.contract_profile = "test"
    review._model = lambda *_args: {}
    review._assess_validity = lambda **_kwargs: None
    review._execute = lambda: (_ for _ in ()).throw(error)

    with pytest.raises(ProviderCapabilityError, match="public-review cache_prompt"):
        review.completion({}, None, "https://api.example/v1", "key")


def test_public_review_rerun_does_not_swallow_selected_capability_refusal():
    from kernel_effect_boundary import Mode
    from kernel_public_review import PublicReview

    error = ProviderCapabilityError("public-review cache_prompt", "llama.cpp")
    review = object.__new__(PublicReview)
    review.exposed = True
    review.reference = "public_probe:ref"
    review.deadline_policy = None
    review.runtime = SimpleNamespace(
        modes={"public_review": Mode.BOTH},
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    review._execute = lambda: (_ for _ in ()).throw(error)

    with pytest.raises(ProviderCapabilityError, match="public-review cache_prompt"):
        review.rerun("public_probe:ref")


def test_template_capture_preserves_selected_capability_refusal():
    from kernel_template_conformance import capture_submitted

    error = ProviderCapabilityError("template conformance", "llama.cpp")
    capture = SimpleNamespace(
        template_conformance_contract=None,
        template_conformance_mode="on",
        request_id="request-1",
        step=1,
        journal=SimpleNamespace(manifest={"model": "hosted"}),
    )

    def render(*_args, **_kwargs):
        raise error

    with pytest.raises(ProviderCapabilityError, match="template conformance"):
        capture_submitted(
            capture,
            {"messages": [{"role": "user", "content": "task"}], "tools": []},
            "https://api.example/v1/chat/completions", "key", renderer=render,
        )


def test_detreasons_kernel_sensor_does_not_turn_capability_refusal_into_unknown():
    from detreason.kernel import Atom, LlamaSensor

    error = ProviderCapabilityError("detreason/kernel sensor", "LLM_MODEL")
    sensor = LlamaSensor("https://api.example/v1/chat/completions")
    sensor._answer = lambda _atom: (_ for _ in ()).throw(error)

    with pytest.raises(ProviderCapabilityError, match="detreason/kernel sensor"):
        sensor.answer(Atom("a1", "the claim is supported"))


def test_qualified_request_final_seam_scrubs_late_tool_and_message_fields(monkeypatch):
    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_launch_identity_cost import assert_openai_chat_body
    from kernel_provider_dialect import ProviderCapabilities

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("STATEFRAME_KV_CHUNK_REUSE", "off")
    monkeypatch.setenv("STATEFRAME_THINK_PREFILL", "off")
    monkeypatch.setenv("LONG_HORIZON_PROTOCOL", "off")
    capture = SimpleNamespace(
        frame_route=None,
        journal=SimpleNamespace(manifest={}),
    )
    candidate = {
        "model": "hosted-model",
        "messages": [{"role": "user", "content": "task", "native_message_key": True}],
        "tools": [{"type": "function", "native_tool_key": True, "function": {
            "name": "read", "parameters": {"type": "object", "x-native-schema": True},
            "native_function_key": True,
        }}],
        "reasoning_budget_tokens": 4096,
    }

    wire_body = _finalize_provider_request_body(
        capture, candidate, ProviderCapabilities("openai-strict"),
    )

    assert_openai_chat_body(wire_body)
    assert "reasoning_budget_tokens" not in wire_body
    assert "native_message_key" not in wire_body["messages"][0]
    assert "native_tool_key" not in wire_body["tools"][0]
    assert "native_function_key" not in wire_body["tools"][0]["function"]
    assert "x-native-schema" not in wire_body["tools"][0]["function"]["parameters"]
