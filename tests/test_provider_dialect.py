from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import kernel_provider_dialect as dialect
import kernel_request_diet
import kernel_template_conformance
from kernel_provider_dialect import (
    ProviderCapabilities,
    ProviderCapabilityError,
    capabilities_for,
    require_feature,
    require_selected_request_features,
)
from kernel_launch_identity_cost import ProviderBoundaryContractError


LLAMA_PROPS = {
    "default_generation_settings": {"n_ctx": 65536},
    "model_path": "local-model.gguf",
}


def test_llama_props_and_slots_prove_independent_capabilities(monkeypatch):
    calls = []

    def probe(origin, key, path):
        calls.append((origin, key, path))
        if path == "/props":
            return 200, LLAMA_PROPS
        return 200, [
            {"id": 0, "n_ctx": 65536, "is_processing": False},
            {"id": 2, "n_ctx": 65536, "is_processing": True},
        ]

    monkeypatch.setattr(dialect, "bounded_json_probe", probe)
    cache = {}
    result = capabilities_for("http://localhost:8080/v1/chat/completions", "secret", cache=cache)

    assert result.dialect == "llama.cpp"
    assert result.reason == "proven by /props"
    assert result.slots is True
    assert result.slot_ids == frozenset({0, 2})
    assert calls == [("http://localhost:8080", "secret", "/props"),
                     ("http://localhost:8080", "secret", "/slots")]


@pytest.mark.parametrize("slots, expected_ids", [([], frozenset()),
    ([{"id": 0}], frozenset({0})),
])
def test_props_proven_llama_keeps_legacy_slot_contract(monkeypatch, slots, expected_ids):
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, path: (200, LLAMA_PROPS) if path == "/props" else (200, slots),
    )

    result = capabilities_for("https://api.example/v1", "key", cache={})

    assert result.dialect == "llama.cpp"
    assert result.reason == "proven by /props"
    assert result.slots is True
    assert result.slot_ids == expected_ids
    assert result.slots_status == 200


def test_populated_llama_slots_prove_dialect_when_props_is_missing(monkeypatch):
    calls = []
    slots = [{"id": 3, "n_ctx": 65536, "is_processing": False}]

    def probe(_origin, _key, path):
        calls.append(path)
        return (404, None) if path == "/props" else (200, slots)

    monkeypatch.setattr(dialect, "bounded_json_probe", probe)
    result = capabilities_for("https://api.example/v1", "key", cache={})

    assert result.dialect == "llama.cpp"
    assert result.reason == "proven by /slots"
    assert result.slots is True
    assert result.slot_ids == frozenset({3})
    assert result.props_status == 404
    assert result.slots_status == 200
    assert calls == ["/props", "/slots"]


@pytest.mark.parametrize("slots", [[], [{"id": 3}], [
    {"id": 3, "n_ctx": 65536, "is_processing": False},
    {"id": 3, "n_ctx": 65536, "is_processing": False},
]])
def test_non_slot_shaped_responses_do_not_prove_llama_dialect(monkeypatch, slots):
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, path: (404, None) if path == "/props" else (200, slots),
    )

    result = capabilities_for("https://api.example/v1", "key", cache={})

    assert result.dialect == "openai-strict"
    assert result.reason == "openai-strict because /props was 404 and /slots was unproven"
    assert result.slots is False


@pytest.mark.parametrize(
    ("props", "reason"),
    [
        ((404, None), "openai-strict because /props was 404 and /slots was unproven"),
        ((200, {"models": ["hosted"]}), "openai-strict because /props was not llama.cpp and /slots was unproven"),
        ((302, None), "openai-strict because /props was not llama.cpp and /slots was unproven"),
        (None, "openai-strict because /props was unreachable and /slots was unproven"),
    ],
)
def test_unproven_props_are_openai_strict(monkeypatch, props, reason):
    monkeypatch.setattr(dialect, "bounded_json_probe", lambda *_args: props)
    result = capabilities_for("https://api.example/v1", "key", cache={})
    assert result.dialect == "openai-strict"
    assert result.reason == reason
    assert result.slots is False


@pytest.mark.parametrize(
    ("props", "expected_reason"),
    [
        ((200, None), "openai-strict because /props was not llama.cpp and /slots was unproven"),
        ((401, None), "openai-strict because /props was not llama.cpp and /slots was unproven"),
        ((200, {"default_generation_settings": {}, "n_ctx": "65536"}),
         "openai-strict because /props was not llama.cpp and /slots was unproven"),
        ((200, {"default_generation_settings": {}, "n_ctx": 0}),
         "openai-strict because /props was not llama.cpp and /slots was unproven"),
    ],
)
def test_unclassifiable_answer_is_not_reported_as_unreachable(monkeypatch, props, expected_reason):
    monkeypatch.setattr(dialect, "bounded_json_probe", lambda *_args: props)
    result = capabilities_for("https://api.example/v1", "key", cache={})
    assert result.dialect == "openai-strict"
    assert result.reason == expected_reason
    assert result.props_status == props[0]


def test_probe_records_redirect_without_following_it_or_reusing_credentials(monkeypatch):
    from kernel_bounded_http import _provider_probe_json_response

    calls = []

    class Response:
        status = 302

        def read(self, _limit):
            return b""

        def close(self):
            pass

    class Connection:
        def __init__(self, host, port, *, timeout):
            calls.append(("connect", host, port, timeout))

        def request(self, method, path, *, headers):
            calls.append((method, path, headers))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("kernel_bounded_http.http.client.HTTPConnection", Connection)
    result = _provider_probe_json_response(("http", "api.example", 443), "/props", "secret", 0.5)

    assert result == [302, None]
    assert calls == [
        ("connect", "api.example", 443, 0.5),
        ("GET", "/props", {"Accept": "application/json", "Authorization": "Bearer secret"}),
    ]


def test_probe_timeout_selects_openai_strict(monkeypatch):
    import socket

    from kernel_bounded_http import _provider_probe_json_response

    class Connection:
        def __init__(self, host, port, *, timeout):
            assert (host, port) == ("api.example", 443)
            assert timeout > 0

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            raise socket.timeout("provider probe timed out")

        def close(self):
            pass

    monkeypatch.setattr("kernel_bounded_http.http.client.HTTPSConnection", Connection)
    response = _provider_probe_json_response(("https", "api.example", 443), "/props", "key", 0.5)
    assert response is None

    monkeypatch.setattr(dialect, "bounded_json_probe", lambda *_args: response)
    result = capabilities_for("https://api.example/v1", "key", cache={})
    assert result.dialect == "openai-strict"
    assert result.reason == "openai-strict because /props was unreachable and /slots was unproven"


def test_scripted_endpoint_emulates_llama_props_and_optional_slots():
    import json
    import urllib.request

    from stateframe.scripted_endpoint import Script, serve

    server = serve(Script({"model": "scripted", "steps": [], "slots": [65536]}))
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(origin + "/props", timeout=2) as response:
            assert response.status == 200
            props = json.loads(response.read())
        with urllib.request.urlopen(origin + "/slots", timeout=2) as response:
            slots = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert props["default_generation_settings"]["n_ctx"] == 65536
    assert props["model_path"] == "scripted-endpoint"
    assert slots == [{"id": 0, "n_ctx": 65536, "is_processing": False}]


def test_recorded_tape_server_emulates_llama_props_and_slots():
    from tests.tape_server import TapeServer

    with TapeServer({"turns": [{"message": {"role": "assistant", "content": "done"}}]}) as tape:
        result = capabilities_for(tape.url, "", cache={})

    assert result.dialect == "llama.cpp"
    assert result.reason == "proven by /props"
    assert result.slots is True
    assert result.slot_ids == frozenset(range(8))


@pytest.mark.parametrize("example", ["discount_bug", "tiny_site", "brief_to_report"])
def test_bundled_identity_scripts_prove_llama_without_slots_or_cache_pin(example):
    from stateframe import runner
    from stateframe.endpoint import Endpoint
    from stateframe.scripted_endpoint import Script, serve

    script_path = Path(__file__).resolve().parents[1] / "examples" / f"{example}.script.json"
    script = Script(json.loads(script_path.read_text(encoding="utf-8")))
    server = serve(script)
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(origin + "/props", timeout=2) as response:
            assert response.status == 200
            props = json.loads(response.read())
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(origin + "/slots", timeout=2)
        capabilities = capabilities_for(origin + "/v1", "", cache={})
        options = runner.RunOptions(
            repo=None, task="task", endpoint=Endpoint(origin + "/v1", "scripted"),
        )
        messages = []
        runner._select_prompt_cache_slot(options, messages.append)
    finally:
        server.shutdown()
        server.server_close()

    assert isinstance(props.get("default_generation_settings"), dict)
    assert props["default_generation_settings"]["n_ctx"] == 65536
    assert props["model_path"] == "scripted-endpoint"
    assert error.value.code == 404
    assert capabilities.dialect == "llama.cpp"
    assert capabilities.slots is False
    assert options._cache_slot is None
    assert options._cache_slot_selection == "not_pinned"
    assert options._cache_slot_degraded is True
    assert options._cache_slot_reason == (
        "provider did not prove llama.cpp /slots; optional prompt-cache pin dropped"
    )
    assert messages == ["cache: prompt-cache pin dropped because provider did not prove llama.cpp /slots"]


def test_strict_endpoint_refuses_native_tokenize_before_post():
    from stateframe.scripted_endpoint import Script, serve

    script = Script({"strict_openai": True, "steps": [], "tokenize": {"mode": "ok"}})
    server = serve(script)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(ProviderCapabilityError, match="request tokenization"):
            kernel_request_diet._http_tokenize(
                "hosted prompt", base + "/tokenize",
                generation_endpoint=base + "/v1/chat/completions",
            )
        assert script.tokenize_calls == 0
    finally:
        server.shutdown()
        server.server_close()


def test_strict_tokenizer_uses_existing_safe_meter_fallback():
    from stateframe.scripted_endpoint import Script, serve

    script = Script({"strict_openai": True, "steps": [], "tokenize": {"mode": "ok"}})
    server = serve(script)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        meter = kernel_request_diet.RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0,
            generation_endpoint=base + "/v1/chat/completions",
        )
        count, method = meter.measure({"messages": [{"role": "user", "content": "hosted prompt"}]})
        assert count > 0
        assert method == "calibrated_bytes"
        assert meter.unavailable_reason() == "provider_unsupported"
        assert script.tokenize_calls == 0
    finally:
        server.shutdown()
        server.server_close()


def test_template_conformance_refuses_strict_endpoint_before_native_post(monkeypatch):
    from stateframe.scripted_endpoint import Script, serve

    script = Script({"strict_openai": True, "steps": []})
    server = serve(script)
    posts = []
    monkeypatch.setattr(
        kernel_template_conformance, "_post_json",
        lambda *args, **kwargs: posts.append((args, kwargs)),
    )
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(ProviderCapabilityError, match="template-conformance /apply-template"):
            kernel_template_conformance.render_endpoint_request(
                {"model": "hosted", "messages": []},
                base + "/v1/chat/completions", "",
            )
        assert posts == []
        assert script.requests == []
    finally:
        server.shutdown()
        server.server_close()


def test_provider_status_summary_includes_configured_routes(monkeypatch):
    from types import SimpleNamespace

    from stateframe import runner

    endpoint = SimpleNamespace(
        chat_url="https://default.example/v1/chat/completions",
        api_key=lambda: "",
    )
    options = SimpleNamespace(
        endpoint=endpoint,
        profile="default",
        _provider_route_dialects={
            "edit": {
                "origin": "https://edit.example",
                "dialect": "llama.cpp",
                "reason": "operator override, /props unreachable",
                "reasoning_controls": "llama.cpp reasoning controls unchanged",
            },
        },
    )
    monkeypatch.setattr(runner, "_provider_environment", lambda _options: {})
    monkeypatch.setattr(
        runner, "_provider_capabilities_for",
        lambda *_args: ProviderCapabilities(
            "openai-strict", reason="openai-strict because /props was 404 and /slots was unproven"),
    )

    summary = runner._provider_preflight_summary(options)

    assert "provider dialect: openai-strict (openai-strict because /props was 404 and /slots was unproven)" in summary
    assert "reasoning controls dropped" in summary
    assert "routes: edit@https://edit.example=llama.cpp (operator override, /props unreachable)" in summary


def test_provider_summary_reports_operator_mapped_reasoning_effort(monkeypatch):
    from types import SimpleNamespace

    from stateframe import runner

    options = SimpleNamespace(
        endpoint=SimpleNamespace(
            chat_url="https://api.example/v1/chat/completions",
            api_key=lambda: "",
        ),
        profile="default",
        _provider_route_dialects={},
    )
    monkeypatch.setattr(
        runner, "_provider_environment",
        lambda _options: {"REACT_REASONING_EFFORT": "high"},
    )
    monkeypatch.setattr(
        runner, "_provider_capabilities_for",
        lambda *_args: ProviderCapabilities(
            "openai-strict", reason="openai-strict because /props was 404 and /slots was unproven"),
    )

    summary = runner._provider_preflight_summary(options)

    assert "reasoning controls mapped to reasoning_effort=high" in summary


def test_reasoning_summary_reports_operator_mapped_effort_and_dropped_default():
    from kernel_provider_dialect import reasoning_controls_summary

    assert reasoning_controls_summary("openai-strict", environ={}) == "reasoning controls dropped"
    assert reasoning_controls_summary(
        "openai-strict", environ={"REACT_REASONING_EFFORT": "high"},
    ) == "reasoning controls mapped to reasoning_effort=high"


def test_provider_override_promotes_only_dialect_and_keeps_slots_proof(monkeypatch):
    calls = []

    def probe(_origin, _key, path):
        calls.append(path)
        return None if path == "/props" else (404, None)

    monkeypatch.setattr(dialect, "bounded_json_probe", probe)
    result = capabilities_for(
        "https://api.example/v1", "key", cache={},
        environ={"STATEFRAME_PROVIDER": "llama.cpp"},
    )
    assert result.dialect == "llama.cpp"
    assert result.reason == "operator override, /props unreachable"
    assert not result.slots
    assert calls == ["/props", "/slots"]
    # The caller explicitly requests a slot capability; the provider override alone is not slot proof.
    with pytest.raises(ProviderCapabilityError, match="slot requires proven /slots"):
        require_feature(result, "slot", slots=True)


def test_provider_override_is_registered_as_an_exact_operator_opt_in(monkeypatch):
    from stateframe import runner

    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_PROVIDER", "LLAMA.CPP")
    assert runner.OPT_IN_SWITCHES["STATEFRAME_PROVIDER"] == ("llama.cpp",)
    assert runner.resolve_opt_in_switches()["STATEFRAME_PROVIDER"] == "llama.cpp"
    assert "STATEFRAME_PROVIDER" not in runner.ignored_shell_settings()

    monkeypatch.setenv("STATEFRAME_PROVIDER", "openai")
    with pytest.raises(runner.RunError, match="STATEFRAME_PROVIDER='openai'"):
        runner.resolve_opt_in_switches()


def test_shell_reasoning_effort_reaches_run_spec_and_profile_keeps_precedence(monkeypatch):
    from stateframe import runner

    monkeypatch.setenv("REACT_REASONING_EFFORT", "high")
    environment = {}
    runner._apply_operator_reasoning_effort(environment)
    assert environment["REACT_REASONING_EFFORT"] == "high"

    profile_environment = {"REACT_REASONING_EFFORT": "medium"}
    runner._apply_operator_reasoning_effort(profile_environment)
    assert profile_environment["REACT_REASONING_EFFORT"] == "medium"


def test_missing_slots_proof_refuses_slot_zero_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        dialect,
        "bounded_json_probe",
        lambda _origin, _key, path: (200, LLAMA_PROPS) if path == "/props" else (404, None),
    )
    capabilities = capabilities_for("https://api.example/v1", "key", cache={})
    sent = []

    # An explicitly supplied id_slot requires proof; optional profile pins degrade in runner preflight.
    with pytest.raises(ProviderCapabilityError, match="id_slot requires proven /slots"):
        require_selected_request_features(
            capabilities, {"id_slot": 0},
        )
        sent.append(True)

    assert capabilities.dialect == "llama.cpp"
    assert not capabilities.slots
    assert sent == []


@pytest.mark.parametrize(
    ("props", "reason"),
    [
        ((200, LLAMA_PROPS), "operator override, /props answered llama.cpp shape"),
        ((404, None), "operator override, /props answered 404"),
        ((200, {"model": "other"}), "operator override, /props answered non-llama shape"),
        ((503, None), "operator override, /props answered HTTP 503"),
    ],
)
def test_provider_override_reason_records_props_answer(monkeypatch, props, reason):
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, path: props if path == "/props" else (404, None),
    )
    result = capabilities_for(
        "https://api.example/v1", "key", cache={},
        environ={"STATEFRAME_PROVIDER": "llama.cpp"},
    )
    assert result.dialect == "llama.cpp"
    assert result.reason == reason
    assert result.slots is False


def test_package_imported_direct_boundary_uses_package_relative_schema(monkeypatch):
    from stateframe.engine import kernel_provider_dialect as package_dialect
    from stateframe.engine.kernel_launch_identity_cost import OPENAI_CHAT_BODY_KEYS

    monkeypatch.setattr(package_dialect, "bounded_json_probe", lambda *_args: (404, None))
    selected = package_dialect.enforce_selected_endpoint_body(
        {
            "model": "hosted",
            "messages": [{"role": "user", "content": "task"}],
            "reasoning_budget_tokens": 4096,
            "top_k": 40,
        },
        "https://api.example/v1/chat/completions",
        "key",
        environ={},
    )

    assert set(selected) <= OPENAI_CHAT_BODY_KEYS
    assert "reasoning_budget_tokens" not in selected
    assert "top_k" not in selected


def test_capability_cache_is_run_owned_and_origin_is_credential_free(monkeypatch):
    calls = []

    def probe(origin, key, path):
        calls.append((origin, key, path))
        return (200, LLAMA_PROPS) if path == "/props" else (200, [{"id": 3}])

    monkeypatch.setattr(dialect, "bounded_json_probe", probe)
    cache = {}
    first = capabilities_for("https://user:password@api.example:443/v1", "first", cache=cache)
    second = capabilities_for("https://api.example/other", "second", cache=cache)
    assert first is second
    assert calls == [("https://api.example", "first", "/props"),
                     ("https://api.example", "first", "/slots")]


def test_origin_cache_normalizes_default_ports_but_keeps_nondefault_ports():
    assert dialect.origin_of("HTTPS://api.example:443/v1") == "https://api.example"
    assert dialect.origin_of("http://api.example:8080/v1") == "http://api.example:8080"


def test_parallel_capability_lookups_probe_an_origin_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from time import sleep

    calls = []
    calls_lock = Lock()
    cache = {}

    def probe(_origin, _key, path):
        with calls_lock:
            calls.append(path)
        sleep(0.01)
        return (200, LLAMA_PROPS) if path == "/props" else (404, None)

    monkeypatch.setattr(dialect, "bounded_json_probe", probe)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _index: capabilities_for("https://api.example/v1", "key", cache=cache),
            range(8),
        ))

    assert calls == ["/props", "/slots"]
    assert all(result is results[0] for result in results)


def test_provider_dialect_journal_marker_is_once_per_origin_per_run():
    from concurrent.futures import ThreadPoolExecutor

    dialect.seed_run_capability_cache({})

    with ThreadPoolExecutor(max_workers=4) as executor:
        marked = list(executor.map(
            lambda _index: dialect.mark_dialect_journaled("https://api.example"),
            range(8),
        ))
    assert marked.count(True) == 1
    assert marked.count(False) == 7

    dialect.seed_run_capability_cache({})
    assert dialect.mark_dialect_journaled("https://api.example") is True


def test_old_spec_without_provider_capabilities_seeds_empty_run_cache():
    old_spec = {"schema": "stateframe-run/1", "prompt_cache_slot": {"slot": None}}
    assert "provider_capabilities" not in old_spec
    dialect.seed_run_capability_cache({
        "https://api.example": ProviderCapabilities("llama.cpp").to_record(),
    })
    assert dialect.RUN_CAPABILITY_CACHE

    dialect.seed_run_capability_cache(old_spec.get("provider_capabilities"))

    assert dialect.RUN_CAPABILITY_CACHE == {}


def test_slot_zero_is_never_synthesized(monkeypatch):
    from kernel_kv_chunk_reuse import _slot_id
    from kernel_self_plan_prefill import self_plan_id_slot

    with pytest.raises(ProviderCapabilityError, match="KV_CHUNK_REUSE.*explicitly selected slot"):
        _slot_id({}, {})
    with pytest.raises(ProviderCapabilityError, match="THINK_PREFILL=self_plan.*explicitly selected slot"):
        self_plan_id_slot({})
    assert _slot_id({}, {"REACT_SLOT_ID": "0"}) == 0
    assert self_plan_id_slot({"REACT_SLOT_ID": "0"}) == 0


def test_id_slot_key_with_null_value_is_not_silently_dropped():
    with pytest.raises(ProviderBoundaryContractError, match="id_slot must be explicitly selected"):
        require_selected_request_features(
            ProviderCapabilities("openai-strict"), {"id_slot": None},
        )


@pytest.mark.parametrize(
    "body",
    [
        {"continue_final_message": False},
        {"continue_final_message": ""},
        {"add_generation_prompt": False},
    ],
)
def test_disabled_continuation_controls_do_not_require_llama_capability(body):
    require_selected_request_features(ProviderCapabilities("openai-strict"), body)


def test_legacy_direct_send_helper_maps_strict_body(monkeypatch):
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, path: (404, None) if path == "/props" else None,
    )
    from kernel_launch_identity_cost import OPENAI_CHAT_BODY_KEYS

    result = dialect.enforce_selected_endpoint_body(
        {
            "model": "hosted",
            "messages": [{"role": "user", "content": "task"}],
            "reasoning_budget_tokens": 4096,
            "top_k": 32,
        },
        "https://api.example/v1/chat/completions", "key", environ={},
    )

    assert set(result) <= OPENAI_CHAT_BODY_KEYS
    assert "reasoning_budget_tokens" not in result
    assert "top_k" not in result


def test_detreason_sensor_selects_strict_chat_model_and_logprobs(monkeypatch):
    from detreason.kernel import LlamaSensor
    from kernel_launch_identity_cost import OPENAI_CHAT_BODY_KEYS, assert_openai_chat_body

    monkeypatch.setattr(dialect, "RUN_CAPABILITY_CACHE", {})
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, _path: (404, None),
    )
    monkeypatch.setenv("LLM_MODEL", "hosted-model")
    captured = []

    class Response:
        def read(self):
            return b'{"choices": []}'

    def urlopen(request, timeout):
        captured.append((request, timeout))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sensor = LlamaSensor(
        "https://detreason-strict.example/v1/chat/completions", key="selected-key",
    )
    sensor._call("question", seed=7)

    request, timeout = captured[0]
    body = json.loads(request.data)
    assert set(body) <= OPENAI_CHAT_BODY_KEYS
    assert_openai_chat_body(body)
    assert body["model"] == "hosted-model"
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 8
    assert "n_probs" not in body
    assert request.get_header("Authorization") == "Bearer selected-key"
    assert timeout == sensor.timeout


def test_detreason_sensor_requires_model_before_strict_dispatch(monkeypatch):
    from detreason.kernel import LlamaSensor

    monkeypatch.setattr(dialect, "RUN_CAPABILITY_CACHE", {})
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, _path: (404, None),
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    sent = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *_args, **_kwargs: sent.append(True),
    )

    sensor = LlamaSensor("https://detreason-no-model.example/v1/chat/completions")
    with pytest.raises(ProviderCapabilityError, match="detreason/kernel sensor.*LLM_MODEL"):
        sensor._call("question", seed=7)
    assert sent == []


def test_decision_api_refuses_missing_model_before_openai_dispatch():
    from kernel_decision_api import score_boolean

    sent = []
    with pytest.raises(ProviderCapabilityError, match="kernel_decision_api.*model"):
        score_boolean(
            "Is the claim supported?",
            request_fn=lambda body: sent.append(dict(body)) or {},
            native_n_probs=True,
        )

    assert sent == []


def test_fixer_strict_request_maps_native_fields_and_requires_model(monkeypatch):
    import fixer
    from kernel_launch_identity_cost import OPENAI_CHAT_BODY_KEYS, assert_openai_chat_body

    monkeypatch.setattr(dialect, "RUN_CAPABILITY_CACHE", {})
    monkeypatch.setattr(
        dialect, "bounded_json_probe",
        lambda _origin, _key, _path: (404, None),
    )
    monkeypatch.setenv("LLM_MODEL", "hosted-model")
    monkeypatch.setenv("LLM_API_KEY", "selected-key")
    sent = []

    class Response:
        def read(self):
            return b'{"choices": []}'

    def urlopen(request, timeout):
        sent.append((request, timeout))
        return Response()

    monkeypatch.setattr(fixer.urllib.request, "urlopen", urlopen)
    fixer._post(
        {
            "messages": [{"role": "user", "content": "question"}],
            "n_probs": 2,
            "grammar": "root ::= \\\"A\\\"",
            "top_k": 32,
        },
        "https://fixer-strict.example/v1/chat/completions",
    )

    request, _timeout = sent[0]
    body = json.loads(request.data)
    assert set(body) <= OPENAI_CHAT_BODY_KEYS
    assert_openai_chat_body(body)
    assert body["model"] == "hosted-model"
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    assert "grammar" not in body
    assert "top_k" not in body
    assert request.get_header("Authorization") == "Bearer selected-key"

    monkeypatch.delenv("LLM_MODEL")
    sent.clear()
    with pytest.raises(ProviderCapabilityError, match="fixer request.*model"):
        fixer._post(
            {"messages": [{"role": "user", "content": "question"}]},
            "https://fixer-no-model.example/v1/chat/completions",
        )
    assert sent == []


@pytest.mark.parametrize("action", ["save", "restore", "erase"])
def test_slot_action_refuses_before_network_without_slots_proof(action):
    from kernel_kv_chunk_reuse import slot_action

    called = []

    def opener(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("slot API must not be called")

    # This is an explicit KV slot operation, so it still requires live slot proof.
    with pytest.raises(ProviderCapabilityError, match=rf"KV_CHUNK_REUSE slot {action}.*\/slots"):
        slot_action("https://api.example/v1", 0, action, "cache.bin", opener=opener,
                    capabilities=ProviderCapabilities("llama.cpp", False))
    assert not called


@pytest.mark.parametrize(
    ("feature", "body", "route_slot", "purpose", "environ"),
    [
        ("KV_CHUNK_REUSE", {}, None, None, {"STATEFRAME_KV_CHUNK_REUSE": "on"}),
        ("THINK_PREFILL=self_plan", {}, None, None, {"STATEFRAME_THINK_PREFILL": "self_plan"}),
        ("true prefill continuation", {"continue_final_message": "reasoning_content"}, None, None, {}),
        ("routed static slot", {}, 7, None, {}),
        ("id_slot", {"id_slot": 0}, None, None, {}),
        ("history-compaction cache/slot", {"cache_prompt": True}, None, "history_compaction",
         {"LONG_HORIZON_PROTOCOL": "on"}),
        ("public-review cache_prompt", {"cache_prompt": False}, None, "public_review", {}),
        ("STATEFRAME_PROMPT_CACHE", {"id_slot": 0, "cache_prompt": True}, None, None,
         {"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "0"}),
        ("STATEFRAME_PROMPT_CACHE", {}, None, None,
         {"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "0"}),
        ("public-review cache_prompt", {
            "cache_prompt": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "public_probe"}},
        }, None, "generate", {}),
    ],
)
def test_unproven_selected_feature_is_typed_and_refuses_before_dispatch(
        feature, body, route_slot, purpose, environ):
    # These cases explicitly select a native feature or slot; automatic profile pins are tested separately.
    sends = []

    def dispatch():
        sends.append("sent")

    with pytest.raises(ProviderCapabilityError, match=feature):
        require_selected_request_features(
            ProviderCapabilities("openai-strict"), body,
            route_slot=route_slot, call_purpose=purpose, environ=environ,
        )
        dispatch()
    assert not sends


def test_history_compaction_cache_requires_an_explicit_reported_slot():
    capabilities = ProviderCapabilities("llama.cpp", True, frozenset({0}))
    # Explicit cache_prompt=True keeps the history-cache contract strict when no slot is selected.
    with pytest.raises(ProviderCapabilityError, match="history-compaction cache/slot.*explicitly selected"):
        require_selected_request_features(
            capabilities,
            {"cache_prompt": True},
            call_purpose="history_compaction",
            environ={"LONG_HORIZON_PROTOCOL": "on"},
        )

    require_selected_request_features(
        capabilities,
        {"cache_prompt": True, "id_slot": 0},
        call_purpose="history_compaction",
        environ={"LONG_HORIZON_PROTOCOL": "on"},
    )


def test_hosted_history_compaction_without_cache_uses_strict_request(monkeypatch):
    import probes.history_compaction as history_compaction
    from kernel_launch_identity_cost import (
        ReasoningProfile, assert_openai_chat_body, enforce_provider_body,
    )

    monkeypatch.setenv("LONG_HORIZON_PROTOCOL", "on")
    monkeypatch.setenv("STATEFRAME_PROMPT_CACHE", "off")
    body = history_compaction.summary_request(
        [{"role": "user", "content": "Earlier conversation."}], model="hosted-model",
    )
    assert body["cache_prompt"] is False
    assert "id_slot" not in body

    require_selected_request_features(
        ProviderCapabilities("openai-strict"), body,
        call_purpose="history_compaction",
        environ={"LONG_HORIZON_PROTOCOL": "on", "STATEFRAME_PROMPT_CACHE": "off"},
    )
    strict = enforce_provider_body(
        body, profile=ReasoningProfile(), dialect="openai-strict",
        environ={"LONG_HORIZON_PROTOCOL": "on", "STATEFRAME_PROMPT_CACHE": "off"},
    )
    assert "cache_prompt" not in strict
    assert "id_slot" not in strict
    assert "chat_template_kwargs" not in strict
    assert "reasoning_effort" not in strict
    assert_openai_chat_body(strict)


def test_plain_loop_journals_prompt_cache_pin_degradation(tmp_path):
    from kernel_plain_contracts import PlainContractError
    from kernel_plain_host import run_plain

    output = tmp_path / "worker"
    output.mkdir()
    (output / "scratch").mkdir()
    reason = "provider did not prove llama.cpp /slots; optional prompt-cache pin dropped"

    with pytest.raises(PlainContractError, match="multi-file public row with declared files"):
        run_plain(
            {"files": {}, "msg": "task"}, output,
            provider=lambda _body, _timeout: {},
            check_runner=lambda *_args: None,
            environ={
                "STATEFRAME_PROMPT_CACHE": "off",
                "SF_PROMPT_CACHE_PIN_DEGRADED_REASON": reason,
            },
        )

    events = [json.loads(line) for line in (output / "plain" / "events.jsonl").read_text().splitlines()]
    notices = [event for event in events if event.get("kind") == "provider_feature_degraded"]
    assert len(notices) == 1
    assert notices[0]["feature"] == "prompt-cache pinned slot"
    assert notices[0]["reason"] == reason


def test_explicit_cli_slot_refuses_when_selected_endpoint_has_no_slots_proof(monkeypatch):
    from stateframe import runner
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_provider_dialect import (
        ProviderCapabilities as RunnerProviderCapabilities,
        ProviderCapabilityError as RunnerProviderCapabilityError,
    )

    options = runner.RunOptions(
        repo=None, task="task", endpoint=Endpoint("http://127.0.0.1:8080/v1", "hosted"),
        slot=0,
    )
    monkeypatch.setattr(
        runner, "_provider_capabilities_for",
        lambda *_args: RunnerProviderCapabilities("openai-strict"),
    )
    # --slot is the existing explicit requirement path and keeps its named refusal.
    with pytest.raises(RunnerProviderCapabilityError, match=r"prompt-cache pinned slot.*/slots"):
        runner._select_prompt_cache_slot(options, lambda *_args: None)


def test_enabled_prompt_cache_requires_its_selected_slot_on_each_request():
    capabilities = ProviderCapabilities("llama.cpp", True, frozenset({2}))
    # A runner-selected REACT_SLOT_ID requires the matching id_slot/cache_prompt fields per request.
    with pytest.raises(ProviderCapabilityError, match="STATEFRAME_PROMPT_CACHE.*request fields"):
        require_selected_request_features(
            capabilities, {"model": "local", "messages": []},
            environ={"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "2"},
        )

    require_selected_request_features(
        capabilities,
        {"model": "local", "messages": [], "id_slot": 2, "cache_prompt": True},
        environ={"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "2"},
    )


def test_enabled_prompt_cache_needs_endpoint_slot_proof_before_dispatch():
    # The shipped profile enables prompt caching, but it is only selected for a
    # request after a slot has been pinned. A strict endpoint can keep running
    # with this optional optimization unpinned.
    require_selected_request_features(
        ProviderCapabilities("openai-strict"), {},
        environ={"STATEFRAME_PROMPT_CACHE": "on"},
    )
    # The request contains id_slot and REACT_SLOT_ID, so this explicit pin stays a hard refusal.
    with pytest.raises(ProviderCapabilityError, match=r"STATEFRAME_PROMPT_CACHE.*llama.cpp.*/slots"):
        require_selected_request_features(
            ProviderCapabilities("openai-strict"),
            {"model": "hosted", "messages": [], "id_slot": 0, "cache_prompt": True},
            environ={"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "0"},
        )
    with pytest.raises(ProviderCapabilityError, match=r"STATEFRAME_PROMPT_CACHE.*/slots"):
        # Even a llama.cpp dialect cannot honor an explicit ID without /slots proof.
        require_selected_request_features(
            ProviderCapabilities("llama.cpp", False),
            {"model": "local", "messages": [], "id_slot": 0, "cache_prompt": True},
            environ={"STATEFRAME_PROMPT_CACHE": "on", "REACT_SLOT_ID": "0"},
        )
