from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

import kernel_launch_identity_cost as boundary
from kernel_launch_identity_cost import (
    OPENAI_CHAT_BODY_KEYS,
    ProviderBoundaryContractError,
    ReasoningProfile,
    assert_openai_chat_body,
    enforce_provider_body,
)


ROOT = Path(__file__).resolve().parents[1]
_BASELINE = None


def _baseline_boundary():
    global _BASELINE
    if _BASELINE is not None:
        return _BASELINE
    source = subprocess.run(
        ["git", "show", "1bbb8ce:src/stateframe/engine/kernel_launch_identity_cost.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    name = "_stateframe_parent_provider_boundary"
    module = ModuleType(name)
    module.__file__ = "1bbb8ce:src/stateframe/engine/kernel_launch_identity_cost.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    _BASELINE = module
    return module


def _allowed_body(body):
    checked = enforce_provider_body(body, profile=ReasoningProfile(), dialect="openai-strict")
    assert set(checked) <= OPENAI_CHAT_BODY_KEYS
    assert_openai_chat_body(checked)
    return checked


def test_qualified_request_binds_strict_profile_before_transport_closure():
    """The sealed FrameFit5 branch bypasses legacy finalization before provider_call."""
    source = ROOT / "src/stateframe/engine/kernel_measurement.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    capture = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == "RequestCapture")
    send_once = next(node for node in capture.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_send_once")
    qualified_branch = next(node for node in send_once.body
                            if isinstance(node, ast.If)
                            and "has_framefit5_state" in ast.unparse(node.test))
    profile_bindings = [node for node in send_once.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "final_profile"
                                for target in node.targets)]
    assert any(binding.lineno < qualified_branch.lineno for binding in profile_bindings)


def test_shipped_profile_parent_request_shape_is_strict_schema_safe(monkeypatch):
    # NOTE: parallel subframes are not on this base, so only the parent request is covered here. When subframes
    # land, their request bodies and a two-subframe CLI run against the strict mock must be added to this guard.
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REACT_REASONING_BUDGET_TOKENS", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_PROFILE", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_MODE", raising=False)
    from kernel_step_contract import configure_reasoning_budget

    parent = {
        "model": "hosted-model",
        "messages": [{"role": "system", "content": "system"},
                     {"role": "user", "content": "task"}],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "Read a file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            }}}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    configure_reasoning_budget(parent)
    assert parent["reasoning_budget_tokens"] == 4096

    strict = _allowed_body(parent)
    assert "reasoning_budget_tokens" not in strict
    assert "reasoning_mode" not in strict
    assert "chat_template_kwargs" not in strict
    assert "reasoning_effort" not in strict


def test_strict_provider_refit_keeps_selected_dialect_for_passthrough_profile(monkeypatch):
    import kernel_measurement

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    body = {
        "model": "hosted-model",
        "messages": [{"role": "user", "content": "task"}],
        "reasoning_budget_tokens": 4096,
        "reasoning_mode": "work",
        "chat_template_kwargs": {"enable_thinking": True},
    }

    refit = kernel_measurement._finalize_provider_refit(
        body, profile=ReasoningProfile("passthrough"), dialect="openai-strict",
    )

    assert refit == {"model": "hosted-model", "messages": body["messages"]}


def test_every_audited_opt_in_native_field_is_removed_or_mapped(monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    body = {
        "model": "hosted-model",
        "messages": [{"role": "assistant", "content": "thinking",
                      "reasoning_content": "native", "native_message_key": True}],
        "tools": [{"type": "function", "native_tool_key": True, "function": {
            "name": "read", "description": "read", "native_function_key": True,
            "parameters": {"type": "object"},
        }}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
        "reasoning_budget_tokens": 4096,
        "reasoning_mode": "work",
        "chat_template_kwargs": {"enable_thinking": True},
        "cache_prompt": True,
        "id_slot": 0,
        "top_k": 40,
        "min_p": 0.1,
        "samplers": ["temperature"],
        "grammar": "root ::= \"A\"",
        "n_probs": 8,
        "continue_final_message": "reasoning_content",
        "add_generation_prompt": True,
        "repeat_penalty": 1.1,
        "score_domain": "native",
        "n_predict": 1,
    }

    strict = _allowed_body(body)
    assert strict["logprobs"] is True
    assert strict["top_logprobs"] == 8
    assert not ({"reasoning_budget_tokens", "reasoning_mode", "chat_template_kwargs",
                 "cache_prompt", "id_slot", "top_k", "min_p", "samplers", "grammar",
                 "n_probs", "continue_final_message", "add_generation_prompt", "repeat_penalty",
                 "score_domain", "n_predict"} & set(strict))
    assert "reasoning_content" not in strict["messages"][0]
    assert "native_message_key" not in strict["messages"][0]
    assert "native_tool_key" not in strict["tools"][0]
    assert "native_function_key" not in strict["tools"][0]["function"]


def test_openai_boundary_scrubs_unknown_keys_at_nested_message_and_schema_levels(monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    body = {
        "model": "hosted-model",
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {
                "url": "https://example.test/image.png", "native_detail_mode": "fast",
            }, "native_part_key": True}],
            "native_message_key": True,
        }],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "read", "native_function_key": True,
            "parameters": {"type": "object", "x-native-schema-key": True, "nullable": True,
                            "properties": {"path": {"type": "string", "x-inner": True}},
                            "prefixItems": [{"type": "string", "x-prefix": True}]},
        }}],
        "response_format": {"type": "json_schema", "native_format_key": True,
                            "json_schema": {"name": "answer", "schema": {
                                "type": "object", "x-native-schema-key": True,
                            }}},
    }

    strict = _allowed_body(body)
    message = strict["messages"][0]
    assert set(message) == {"role", "content"}
    assert set(message["content"][0]) == {"type", "image_url"}
    assert message["content"][0]["image_url"] == {"url": "https://example.test/image.png"}
    assert strict["tools"][0]["function"]["parameters"] == {
        "type": "object", "properties": {"path": {"type": "string"}},
        "prefixItems": [{"type": "string"}],
    }
    assert strict["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }


def test_openai_schema_guard_rejects_unknown_nested_keys_when_boundary_is_bypassed():
    with pytest.raises(ProviderBoundaryContractError, match=r"unknown OpenAI keys at messages\[0\].content\[0\].image_url"):
        assert_openai_chat_body({
            "model": "hosted-model",
            "messages": [{"role": "user", "content": [{
                "type": "image_url", "image_url": {"url": "x", "native_detail_mode": "fast"},
            }]}],
        })
    with pytest.raises(ProviderBoundaryContractError, match="content must be a string or content-part array"):
        assert_openai_chat_body({
            "model": "hosted-model",
            "messages": [{"role": "user", "content": {"native_key": True}}],
        })


def test_strict_mock_rejects_unknown_nested_message_and_tool_keys():
    from stateframe.scripted_endpoint import _strict_openai_error

    assert _strict_openai_error({
        "model": "hosted",
        "messages": [{"role": "user", "content": "task", "native_message_key": True}],
    }) == "unknown keys at messages[0]: native_message_key"
    assert _strict_openai_error({
        "model": "hosted",
        "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "x-native": True},
        }}],
    }) == "unknown keys at tools[0].function.parameters: x-native"
    assert _strict_openai_error({
        "model": "hosted",
        "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "nullable": True},
        }}],
    }) == "unknown keys at tools[0].function.parameters: nullable"
    assert _strict_openai_error({
        "model": "hosted",
        "messages": [{"role": "user", "content": {"native_key": True}}],
    }) == "messages[0].content must be a string or content-part array"


def test_strict_mock_schema_matches_the_production_openai_boundary():
    from stateframe import scripted_endpoint

    assert scripted_endpoint._STRICT_TOP_LEVEL_KEYS == boundary.OPENAI_CHAT_BODY_KEYS
    for mock_name, boundary_name in (
        ("_STRICT_MESSAGE_KEYS", "_MESSAGE_KEYS"),
        ("_STRICT_CONTENT_KEYS", "_CONTENT_PART_KEYS"),
        ("_STRICT_TOOL_KEYS", "_TOOL_KEYS"),
        ("_STRICT_FUNCTION_KEYS", "_FUNCTION_KEYS"),
        ("_STRICT_LEGACY_FUNCTION_KEYS", "_LEGACY_FUNCTION_KEYS"),
        ("_STRICT_TOOL_CALL_KEYS", "_TOOL_CALL_KEYS"),
        ("_STRICT_TOOL_CALL_FUNCTION_KEYS", "_TOOL_CALL_FUNCTION_KEYS"),
        ("_STRICT_SCHEMA_KEYS", "_JSON_SCHEMA_KEYS"),
        ("_STRICT_RESPONSE_FORMAT_KEYS", "_RESPONSE_FORMAT_KEYS"),
        ("_STRICT_JSON_SCHEMA_FORMAT_KEYS", "_JSON_SCHEMA_FORMAT_KEYS"),
    ):
        assert getattr(scripted_endpoint, mock_name) == getattr(boundary, boundary_name)


def test_openai_schema_fields_keep_valid_cache_and_moderation_options(monkeypatch):
    monkeypatch.setenv("REACT_REASONING_EFFORT", "xhigh")
    body = {
        "model": "hosted-model",
        "messages": [
            {"role": "user", "content": [{
                "type": "text", "text": "Analyze this.",
                "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m", "native": True},
            }]},
            {"role": "assistant", "content": None,
             "function_call": {"name": "read", "arguments": "{}"}},
            {"role": "assistant", "content": None,
             "audio": {"id": "audio-id", "native": True}},
        ],
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m", "native": True},
        "functions": [{"name": "legacy", "parameters": {"type": "object"}, "strict": True}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "description": "A short answer", "schema": {"type": "object"},
        }},
        "moderation": {
            "model": "omni-moderation-latest",
            "policy": {"input": {"mode": "score", "native": True},
                       "output": {"mode": "block"}, "native": True},
            "native": True,
        },
        "reasoning_effort": "high",
    }

    strict = _allowed_body(body)

    assert strict["reasoning_effort"] == "xhigh"
    assert strict["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert strict["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit", "ttl": "30m",
    }
    assert strict["messages"][1]["function_call"] == {"name": "read", "arguments": "{}"}
    assert strict["messages"][2]["audio"] == {"id": "audio-id"}
    assert strict["functions"] == [{"name": "legacy", "parameters": {"type": "object"}}]
    assert strict["response_format"]["json_schema"]["description"] == "A short answer"
    assert strict["moderation"] == {
        "model": "omni-moderation-latest",
        "policy": {"input": {"mode": "score"}, "output": {"mode": "block"}},
    }
    from stateframe.scripted_endpoint import _strict_openai_error
    assert _strict_openai_error(strict) is None


@pytest.mark.parametrize(
    ("site", "native_fields"),
    [
        ("default reasoning", {"reasoning_budget_tokens": 4096}),
        ("reasoning off", {"reasoning_mode": "off", "chat_template_kwargs": {"enable_thinking": False}}),
        ("per-step thinking router", {
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
        }),
        ("prompt cache and pinned slot", {"cache_prompt": True, "id_slot": 2}),
        ("routed static slot", {"cache_prompt": True, "id_slot": 4}),
        ("routed sampler", {"top_k": 30, "min_p": 0.05}),
        ("plain loop body", {"reasoning_budget_tokens": 4096, "top_k": 24}),
        ("KV_CHUNK_REUSE", {"cache_prompt": True, "id_slot": 3}),
        ("THINK_PREFILL=self_plan", {"continue_final_message": "reasoning_content"}),
        ("true prefill continuation", {"add_generation_prompt": True}),
        ("history-compaction cache/slot", {"cache_prompt": False}),
        ("public-review cache_prompt", {"cache_prompt": False}),
        ("kernel_decision_api", {"n_probs": 8, "reasoning_budget_tokens": 0,
                                  "reasoning_mode": "off",
                                  "chat_template_kwargs": {"enable_thinking": False}}),
        ("sys1/adapter sampler and grammar", {
            "top_k": 4, "min_p": 0.1, "samplers": ["temperature"],
            "cache_prompt": True, "grammar": "root",
            "chat_template_kwargs": {"enable_thinking": False},
        }),
        ("detreason/kernel n_probs", {"n_probs": 8}),
        ("line_stage n_probs", {"n_probs": 2}),
        ("fixer grammar", {"grammar": "root ::= \\\"A\\\""}),
        ("template conformance", {"add_generation_prompt": True}),
        ("study capacity preflight", {"add_generation_prompt": True,
                                       "content": "prompt", "add_special": False,
                                       "parse_special": True}),
        ("request tokenizer", {"content": "prompt", "add_special": False,
                               "parse_special": True}),
        ("decision ladder metadata", {
            "rung": "L1", "decision_id": "decision-1", "vote_index": 1, "vote_seed": 7919,
        }),
        ("DECIDE2 micro request", {"rung": "L2", "decision_id": "decision-2"}),
        ("DECIDE2 work request", {"reasoning_budget_tokens": 512,
                                   "reasoning_mode": "work",
                                   "chat_template_kwargs": {"enable_thinking": True}}),
        ("System One internal score fields", {
            "score_domain": "native", "n_predict": 1, "post_sampling_probs": False,
            "completion_reserve": 1, "profile_kind": "scored-decision",
            "top_k": 0, "min_p": 0.0, "cache_prompt": True,
            "reasoning_mode": "off", "reasoning_budget_tokens": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }),
        ("BIND_QUANTITY reasoning", {"reasoning_budget_tokens": 512}),
    ],
)
def test_planted_boundary_bypass_is_red_for_each_body_site(monkeypatch, site, native_fields):
    body = {
        "model": "hosted-model",
        "messages": [{"role": "user", "content": site}],
        "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}}}],
        **native_fields,
    }
    monkeypatch.setattr(boundary, "enforce_provider_body", lambda value, **_kwargs: dict(value))
    bypassed = boundary.enforce_provider_body(
        body, profile=ReasoningProfile(), dialect="openai-strict",
    )
    with pytest.raises(ProviderBoundaryContractError, match="unknown OpenAI"):
        assert_openai_chat_body(bypassed)


def test_shipped_profile_llama_body_is_byte_identical_to_unmodified_parent_code(monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REACT_REASONING_BUDGET_TOKENS", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_PROFILE", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_MODE", raising=False)
    from kernel_step_contract import configure_reasoning_budget
    import kernel_provider_dialect as provider_dialect

    body = {
        "model": "local-model.gguf",
        "messages": [{"role": "system", "content": "system"},
                     {"role": "user", "content": "task"}],
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 512,
        "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}}}],
    }
    configure_reasoning_budget(body)
    baseline = _baseline_boundary()
    baseline_body = baseline.enforce_provider_body(
        body, profile=baseline.reasoning_profile(environ={}),
    )
    monkeypatch.setattr(
        provider_dialect,
        "bounded_json_probe",
        lambda _origin, _key, path: (
            (200, {"default_generation_settings": {"n_ctx": 65536},
                   "model_path": "local-model.gguf"})
            if path == "/props" else (404, None)
        ),
    )
    proven_llama = provider_dialect.capabilities_for(
        "http://127.0.0.1:8080/v1", "", cache={},
    )
    assert proven_llama.dialect == "llama.cpp"
    llama_body = enforce_provider_body(
        body, profile=ReasoningProfile(), dialect=proven_llama.dialect,
    )
    assert json.dumps(llama_body).encode("utf-8") == json.dumps(baseline_body).encode("utf-8")


def test_decide2_work_request_defers_dialect_filter_to_selected_endpoint(monkeypatch):
    from kernel_decide2 import DecisionRequest, build_work_request

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_PROFILE", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_MODE", raising=False)
    body = build_work_request(
        DecisionRequest(
            decision_id="decision-1", question_kind="relevance",
            question="Is the source relevant?", options=("yes", "no", "ABSTAIN"),
        ),
        reason="low confidence",
        max_tokens=64,
    )

    assert body["reasoning_budget_tokens"] == 64
    assert body["reasoning_mode"] == "work"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}

    strict = _allowed_body(body)
    assert "reasoning_budget_tokens" not in strict
    assert "reasoning_mode" not in strict
    assert "chat_template_kwargs" not in strict


def test_strict_reasoning_effort_only_comes_from_operator_setting(monkeypatch):
    body = {"model": "hosted", "messages": [], "reasoning_effort": "high",
            "reasoning_budget_tokens": 4096}
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    assert "reasoning_effort" not in _allowed_body(body)
    monkeypatch.setenv("REACT_REASONING_EFFORT", "medium")
    assert _allowed_body(body)["reasoning_effort"] == "medium"


def test_per_step_thinking_router_drops_native_disable_without_operator_effort(monkeypatch):
    from kernel_stateframe_thinking_router import apply_thinking_off

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    routed = apply_thinking_off({"model": "hosted", "messages": []})
    strict = _allowed_body(routed)

    assert "chat_template_kwargs" not in strict
    assert "reasoning_effort" not in strict

    monkeypatch.setenv("REACT_REASONING_EFFORT", "none")
    assert _allowed_body(routed)["reasoning_effort"] == "none"


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_strict_reasoning_effort_accepts_the_openai_chat_enum(monkeypatch, effort):
    monkeypatch.setenv("REACT_REASONING_EFFORT", effort)

    assert _allowed_body({"model": "hosted", "messages": []})["reasoning_effort"] == effort


def test_n_probs_mapping_enables_logprobs_when_openai_count_is_already_set(monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    strict = _allowed_body({
        "model": "hosted", "messages": [], "n_probs": 8,
        "top_logprobs": 5, "logprobs": False,
    })

    assert strict["logprobs"] is True
    assert strict["top_logprobs"] == 5


def test_strict_reasoning_effort_uses_the_selected_environment_mapping(monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    strict = enforce_provider_body(
        {"model": "hosted", "messages": []}, profile=ReasoningProfile(),
        dialect="openai-strict", environ={"REACT_REASONING_EFFORT": "high"},
    )
    assert strict["reasoning_effort"] == "high"

    off = enforce_provider_body(
        {"model": "hosted", "messages": []}, profile=ReasoningProfile("off"),
        dialect="openai-strict", environ={"REACT_REASONING_EFFORT": "none"},
    )
    assert off["reasoning_effort"] == "none"


def test_empty_reasoning_effort_is_treated_as_unset(monkeypatch):
    from kernel_provider_dialect import reasoning_controls_summary
    from kernel_step_contract import configure_reasoning_budget

    monkeypatch.setenv("REACT_REASONING_EFFORT", "  ")
    monkeypatch.delenv("REACT_REASONING_BUDGET_TOKENS", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_PROFILE", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_MODE", raising=False)
    parent = {"model": "hosted", "messages": []}
    configure_reasoning_budget(parent)
    assert parent["reasoning_budget_tokens"] == 4096
    assert "reasoning_effort" not in parent

    strict = enforce_provider_body(
        {"model": "hosted", "messages": [], "reasoning_effort": "high"},
        profile=ReasoningProfile(), dialect="openai-strict",
        environ={"REACT_REASONING_EFFORT": "  "},
    )

    assert "reasoning_effort" not in strict
    assert reasoning_controls_summary(
        "openai-strict", environ={"REACT_REASONING_EFFORT": "  "},
    ) == ("reasoning controls omitted; proposal side-calls refused: require "
          "operator-confirmed REACT_REASONING_EFFORT=none support")


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_budget_writer_normalizes_explicit_effort(monkeypatch, effort):
    from kernel_step_contract import configure_reasoning_budget

    monkeypatch.setenv("REACT_REASONING_EFFORT", f" {effort.upper()} ")
    monkeypatch.delenv("REACT_REASONING_BUDGET_TOKENS", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_PROFILE", raising=False)
    monkeypatch.delenv("STATEFRAME_REASONING_MODE", raising=False)
    body = {"model": "hosted", "messages": []}

    configure_reasoning_budget(body)

    assert body["reasoning_effort"] == effort


def test_unsupported_reasoning_effort_summary_matches_boundary_refusal():
    from kernel_provider_dialect import reasoning_controls_summary

    assert reasoning_controls_summary(
        "openai-strict", environ={"REACT_REASONING_EFFORT": "extreme"},
    ) == "reasoning controls refused: unsupported REACT_REASONING_EFFORT"
    with pytest.raises(ProviderBoundaryContractError, match="invalid_reasoning_effort"):
        enforce_provider_body(
            {"model": "hosted", "messages": [], "reasoning_effort": "high"},
            profile=ReasoningProfile(), dialect="openai-strict",
            environ={"REACT_REASONING_EFFORT": "extreme"},
        )


@pytest.mark.parametrize("count", [-1, 21, True, False, "8"])
def test_strict_n_probs_refuses_values_that_cannot_map_to_openai(count, monkeypatch):
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    with pytest.raises(ProviderBoundaryContractError, match="invalid_n_probs"):
        _allowed_body({"model": "hosted", "messages": [], "n_probs": count})


def test_strict_global_reasoning_off_requires_explicit_endpoint_acceptance(monkeypatch):
    body = {
        "model": "hosted",
        "messages": [],
        "reasoning_budget_tokens": 0,
        "reasoning_mode": "off",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    with pytest.raises(ProviderBoundaryContractError, match="reasoning_off_unproven"):
        enforce_provider_body(body, profile=ReasoningProfile("off"), dialect="openai-strict")
    strict = enforce_provider_body(
        body, profile=ReasoningProfile("off"), dialect="openai-strict",
        environ={"REACT_REASONING_EFFORT": "none"},
    )
    assert strict == {"model": "hosted", "messages": [], "reasoning_effort": "none"}


def test_sys1_adapter_selects_openai_scoring_or_refuses_unsupported_sampler(monkeypatch):
    from sys1.adapter import build_openai_chat_request
    from sys1.protocol import QualifiedTemplate, ScoreRequest, TokenBinding
    from kernel_provider_dialect import ProviderCapabilityError

    request = ScoreRequest(
        prompt="Choose Y or N.",
        labels=(TokenBinding("yes", "Y", 1, (89,)), TokenBinding("no", "N", 2, (78,))),
        template=QualifiedTemplate("qualified:test", "fingerprint"),
        model="hosted-model",
    )
    strict = build_openai_chat_request(request)
    assert set(strict) <= OPENAI_CHAT_BODY_KEYS
    assert strict["logprobs"] is True
    assert strict["top_logprobs"] == request.top_logprobs
    assert_openai_chat_body(strict)

    unsupported = ScoreRequest(
        prompt=request.prompt, labels=request.labels, template=request.template,
        model=request.model, top_k=4,
    )
    with pytest.raises(ProviderCapabilityError, match="sys1/adapter sampler.*top_k"):
        build_openai_chat_request(unsupported)


def test_system_one_uses_strict_chat_wire_and_refuses_unproven_reasoning_off(monkeypatch):
    from kernel_provider_dialect import ProviderCapabilityError
    from kernel_system_one import SystemOne
    from sys1.protocol import QualifiedTemplate, ScoreRequest, TokenBinding, TransportKind

    request = ScoreRequest(
        prompt="Choose Y or N.",
        labels=(TokenBinding("yes", "Y", 1, (89,)), TokenBinding("no", "N", 2, (78,))),
        template=QualifiedTemplate("qualified:test", "fingerprint"),
        model="hosted-model",
    )
    preparation = SimpleNamespace(request={
        "messages": [{"role": "user", "content": "Choose Y or N."}],
        "n_predict": 1,
        "reasoning_effort": "none",
        "reasoning_mode": "off",
        "reasoning_budget_tokens": 0,
    })
    system = object.__new__(SystemOne)
    system.wrappers = ()
    system.provider_capabilities = None

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    with pytest.raises(ProviderCapabilityError, match="sys1/adapter reasoning-off.*REACT_REASONING_EFFORT=none"):
        system._wire(preparation, request, TransportKind.CHAT)

    monkeypatch.setenv("REACT_REASONING_EFFORT", "none")
    strict = system._wire(preparation, request, TransportKind.CHAT)
    assert set(strict) <= OPENAI_CHAT_BODY_KEYS
    assert strict["reasoning_effort"] == "none"
    assert strict["logprobs"] is True
    assert_openai_chat_body(strict)


def test_plain_loop_transport_maps_strict_body_before_recording_or_sending(monkeypatch):
    import kernel_provider_dialect as provider_dialect
    from kernel_plain_host import _PlainHTTPProvider
    from kernel_provider_dialect import ProviderCapabilities

    monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(
        provider_dialect, "capabilities_for",
        lambda *_args, **_kwargs: ProviderCapabilities(
            "openai-strict", reason="openai-strict because /props was 404"),
    )
    provider = _PlainHTTPProvider("https://api.example/v1/chat/completions", "key", environ={})
    body = {
        "model": "hosted-model",
        "messages": [{"role": "user", "content": "task", "reasoning_content": "native"}],
        "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}}}],
        "temperature": 0,
        "reasoning_budget_tokens": 4096,
        "top_k": 40,
    }

    provider.prepare_body(body)

    assert set(body) <= OPENAI_CHAT_BODY_KEYS
    assert "reasoning_content" not in body["messages"][0]
    assert "reasoning_budget_tokens" not in body
    assert "top_k" not in body
    assert_openai_chat_body(body)


def test_plain_loop_refuses_native_opt_in_before_transport_starts(monkeypatch):
    import kernel_provider_dialect as provider_dialect
    from kernel_plain_host import _PlainHTTPProvider
    from kernel_provider_dialect import ProviderCapabilities, ProviderCapabilityError

    monkeypatch.setattr(
        provider_dialect, "capabilities_for",
        lambda *_args, **_kwargs: ProviderCapabilities("openai-strict"),
    )
    provider = _PlainHTTPProvider(
        "https://api.example/v1/chat/completions", "key",
        environ={"STATEFRAME_KV_CHUNK_REUSE": "on"},
    )
    starts = []
    monkeypatch.setattr(provider.transport, "begin", lambda: starts.append(True))

    with pytest.raises(ProviderCapabilityError, match="KV_CHUNK_REUSE.*llama.cpp"):
        provider({"model": "hosted", "messages": []}, 1)

    assert not starts
