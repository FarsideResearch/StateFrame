"""Frozen offline acceptance checks for request-estimator review #3."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_frame_allocation_capture as frame
import kernel_request_budget as request_budget
import kernel_request_diet as diet
from kernel_frame_allocation_capture import measured_request_tokens
from kernel_request_diet import RequestTokenMeter


DS_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DS_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"
GENERATION_ENDPOINT = "https://offline.example/v1/chat/completions"
REPO_ROOT = Path(__file__).resolve().parents[1]

# These counts were each measured once from the rendered decoded prompt with
# the Qwen3.6-35B-A3B GGUF vocabulary and offline llama-tokenize.
QWEN_PROMPT_TOKENS = {
    "long_numbers": 12_010,
    "base64": 8_827,
    "code": 18_579,
    "tool_schema_heavy": 25_075,
    "numeric_400": 4_010,
    "numeric_600": 6_010,
    "numeric_700": 7_010,
    "density_train_250": 1_260,
    "density_train_500": 2_510,
    "density_target": 2_410,
}


def _body(text: str, model: str = DS_MODEL) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": text}], "tools": []}


def _sized(size: int, model: str = DS_MODEL) -> dict[str, object]:
    value = _body("", model)
    content = value["messages"][0]["content"]  # type: ignore[index]
    padding = size - diet._wire_bytes(value)
    assert padding >= 0
    value["messages"][0]["content"] = content + ("x" * padding)  # type: ignore[index,operator]
    assert diet._wire_bytes(value) == size
    return value


def _observe(meter: RequestTokenMeter, value: dict[str, object], tokens: int, request_id: str) -> None:
    meter.measure(value)
    meter.observe_record({"kind": "request_submitted", "request_id": request_id, "payload": value})
    meter.observe_record({"kind": "response_received", "request_id": request_id,
                          "accounting": {"prompt_tokens": {"total": tokens}}})


def _hosted_counterexample(kind: str) -> dict[str, object]:
    rng = random.Random(173)
    text = {
        "long_numbers": "1234567890" * 1200,
        "base64": base64.b64encode(rng.randbytes(9000)).decode(),
        "code": "\n".join(f"a{i} = b{i} + c{i} * 1234567890" for i in range(700)),
        "tool_schema_heavy": "Choose a tool.",
    }[kind]
    value = _body(text, QWEN_MODEL)
    if kind == "tool_schema_heavy":
        value["tools"] = [{"type": "function", "function": {"name": f"f_{i}",
            "description": base64.b64encode(rng.randbytes(150)).decode(),
            "parameters": {"type": "object", "properties": {
                f"p_{j}": {"type": "string"} for j in range(20)}}}} for i in range(60)]
    return value


@pytest.mark.parametrize("kind", ["long_numbers", "base64", "code", "tool_schema_heavy"])
def test_F5_hosted_estimate_covers_real_qwen_prompt(kind: str) -> None:
    """REVIEW3 item 1, “Undercount”: estimated tokens must cover the rendered prompt."""
    value = _hosted_counterexample(kind)
    meter = RequestTokenMeter(disabled=True, generation_endpoint=GENERATION_ENDPOINT)
    estimate, method = meter.measure(value)
    actual = QWEN_PROMPT_TOKENS[kind]
    assert method == "calibrated_bytes"
    assert estimate >= actual, (
        f"{kind}: wire={diet._wire_bytes(value)}, estimate={estimate}, real_Qwen_prompt={actual}"
    )


def test_F6_outside_envelope_does_not_forget_observed_dense_token_usage() -> None:
    """REVIEW3 item 2, “Extrapolation / outside sampled envelope”: dense evidence must not be lost."""
    meter = RequestTokenMeter(disabled=True, generation_endpoint=GENERATION_ENDPOINT)
    for n in (400, 600):
        value = _body("1234567890" * n, QWEN_MODEL)
        _observe(meter, value, QWEN_PROMPT_TOKENS[f"numeric_{n}"], str(n))
    value = _body("1234567890" * 700, QWEN_MODEL)
    fallback_mode = getattr(meter, "fallback_mode", None)
    if callable(fallback_mode):
        assert fallback_mode(value) == "cold"
    actual = QWEN_PROMPT_TOKENS["numeric_700"]
    estimate = meter.measure(value)[0]
    assert estimate >= actual, f"outside sampled range: estimate={estimate}, real_Qwen_prompt={actual}"


def test_F7_same_density_inside_envelope_does_not_undercount_when_cold_was_safe() -> None:
    """REVIEW3 item 2, “Extrapolation / outside sampled envelope”: an in-envelope fit must remain safe."""
    meter = RequestTokenMeter(disabled=True, generation_endpoint=GENERATION_ENDPOINT)
    for n in (250, 500):
        value = _body("1234xxxxxxxx" * n, QWEN_MODEL)
        _observe(meter, value, QWEN_PROMPT_TOKENS[f"density_train_{n}"], str(n))
    value = _body("12xxx" * 800, QWEN_MODEL)
    actual = QWEN_PROMPT_TOKENS["density_target"]
    cold = RequestTokenMeter(disabled=True).measure(value)[0]
    assert cold >= actual  # The former cold path was sufficient for this request.
    fallback_mode = getattr(meter, "fallback_mode", None)
    if callable(fallback_mode):
        assert fallback_mode(value) == "fit"
    content_density = getattr(diet, "_content_density", None)
    if callable(content_density):
        assert content_density(value) == 1
    estimate = meter.measure(value)[0]
    assert estimate >= actual, f"fit={estimate} < real_Qwen_prompt={actual} <= cold={cold}"


def test_F8_carried_measurement_cannot_cross_generation_endpoint() -> None:
    """REVIEW3 item 4, “Cache identity”: final frame proof must use the current route."""
    value = _sized(4096)
    meter = RequestTokenMeter(disabled=True, generation_endpoint=DS_ENDPOINT)
    tokens, method = meter.measure(value)
    carried = meter.bind_measurement(value, tokens=tokens, method=method,
        request_id="route-review", max_request_bytes=100000, max_prompt_tokens=33344)
    capture = SimpleNamespace(request_id="route-review", _frame_candidate_measurement=carried)
    meter.bind_tokenizer_auth("public-test-tenant-B", "https://other.example/v1/chat/completions")
    final = frame.final_request_tokens(value, meter, capture)[0]
    fresh = meter.measure(value)[0]
    assert final == fresh, f"stale carried count={final}, new route count={fresh}"


def test_F9_carried_measurement_identity_includes_auth() -> None:
    """REVIEW3 item 4, “Cache identity”: carried measurements must be bound to auth identity."""
    value = _sized(4096)
    meter = RequestTokenMeter(disabled=True, generation_endpoint=DS_ENDPOINT, auth_key="public-test-A")
    tokens, method = meter.measure(value)
    carried = meter.bind_measurement(value, tokens=tokens, method=method,
        request_id="r", max_request_bytes=100000, max_prompt_tokens=33344)
    meter.bind_tokenizer_auth("public-test-B", DS_ENDPOINT)
    assert not carried.matches(value, tokenizer_identity=meter.tokenizer_identity(value),
                               request_id="r", logical_request_id="r"), (
        "measurement still valid across auth identity change"
    )


def test_F10_new_model_frame_estimate_is_independent_of_route_preselection() -> None:
    """REVIEW3 item 5, “Call order / prior state”: route preselection must not change an estimate."""
    def ready_meter() -> RequestTokenMeter:
        meter = RequestTokenMeter(tokenizer=lambda _text: 20)
        frame.measured_request_tokens(_sized(4000, "model-A"), meter)
        return meter

    value = _sized(4000, "model-B")
    first = ready_meter()
    direct = frame.calibrated_estimate(value, first)
    second = ready_meter()
    fallback_mode = getattr(second, "fallback_mode", None)
    if not callable(fallback_mode):
        pytest.skip("6003aba predates the route-preselection API under review")
    fallback_mode(value)
    preselected = frame.calibrated_estimate(value, second)
    assert direct == preselected, f"same model-B body: direct={direct}, preselect-route={preselected}"


def _frame_wire(route: str, length: int) -> tuple[int, str]:
    import test_narrow_stall as narrow

    with tempfile.TemporaryDirectory(prefix=".request-estimator-r3-", dir=REPO_ROOT) as temporary:
        root = Path(temporary) / "frame-fixtures" / f"base-{route}-{length}"
        (root / "pkg").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "pkg/__init__.py").write_text("VALUE = 1\n")
        (root / "tests/test_app.py").write_text("def test_app(): assert True\n")
        messages = [{"role": "user", "content": "Fix value. " + "漢" * length}]
        context = narrow.build_action_context(messages, workspace_root=root)
        context["journal_receipt_records"] = []
        runtime = narrow._runtime_with_product(root)
        runtime._framefit_action_context = context
        check_frame = {"verification": {"revision": "current", "status": "FAIL"}}
        if route == "cold":
            meter = RequestTokenMeter(disabled=True)
        else:
            meter = RequestTokenMeter(tokenizer=lambda text: max(1, len(text) // 3))
            if route == "llama_outage":
                measured_request_tokens({"messages": messages, "tools": []}, meter)
                meter._availability = "unavailable"

        patch = pytest.MonkeyPatch()
        patch.setattr(request_budget, "_request_token_meter", lambda *_args, **_kwargs: meter)
        patch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
        patch.setenv("STATEFRAME_EVICTFIX", "on")
        try:
            result, _capture = narrow._qualified_fixture_request(
                runtime, check_frame, messages, context, baseline=False,
                monkeypatch=patch, validate=False, return_capture=True,
            )
        finally:
            patch.undo()
        wire = json.dumps(result).encode()
        return len(wire), hashlib.sha256(wire).hexdigest()


def test_F11_cold_frame_request_bytes_match_unpatched_tree() -> None:
    """REVIEW3 item 6, “Cold / llama request bytes”: cold frame wire must stay byte-for-byte stable."""
    size, digest = _frame_wire("cold", 3000)
    assert (size, digest) == (
        23547,
        "abbec36bf51e541daaa0d8b23707e7a336822c38d2ad2bc3e1c7a5cc15fdbd44",
    )


@pytest.mark.parametrize("route", ["llama_exact", "llama_outage"])
def test_F12_llama_frame_request_bytes_match_unpatched_tree(route: str) -> None:
    """REVIEW3 item 6, “Cold / llama request bytes”: llama exact and outage wire stay stable."""
    size, digest = _frame_wire(route, 1000)
    assert (size, digest) == (
        11547,
        "06662c9ebcc7b15359c3a400c487af199f30f8e6236b72971adc56ca1ab2a554",
    )


def test_F13_llama_usage_loaded_before_first_measure_is_preserved() -> None:
    """REVIEW3 item 6, “Cold / llama request bytes”: preloaded llama usage must affect the estimate."""
    meter = RequestTokenMeter(disabled=True)
    meter.observe_response(3000, 1500, 0)
    meter.observe_response(6000, 3000, 0)
    result = meter.measure(_sized(4000))
    assert result == (3371, "calibrated_bytes"), f"preloaded llama usage: actual={result}"


def test_G1_model_and_endpoint_near_misses_do_not_select_family_cold_estimate() -> None:
    """REVIEW3 item 3, “Model / endpoint spoofing”: near-miss identities must not select the family estimate."""
    select_family = getattr(diet, "request_family_cold_for_route", None)
    if not callable(select_family):
        pytest.skip("6003aba has no request-family cold route selector to spoof")

    models = [DS_MODEL.lower(), DS_MODEL.upper(), " " + DS_MODEL, DS_MODEL + " ",
              "\t" + DS_MODEL, DS_MODEL + "\n", "vendor/" + DS_MODEL, DS_MODEL + "-0731",
              DS_MODEL.replace("D", "Ｄ"), DS_MODEL.replace("a", "а"),
              DS_MODEL + "\u200b", DS_MODEL + "\x00", "deepseek-v4-flash",
              "deepseek-ai%2FDeepSeek-V4-Flash", "/" + DS_MODEL]
    for model in models:
        assert select_family(model, DS_ENDPOINT) is None, repr(model)

    endpoints = [
        "http://api.deepinfra.com/v1/openai",
        "https://api.deepinfra.com.evil.example/v1/openai",
        "https://evil.example/api.deepinfra.com/v1/openai",
        "https://api.deepinfra.com@evil.example/v1/openai",
        "https://evil@api.deepinfra.com/v1/openai",
        "https://api.deepinfra.com:444/v1/openai",
        "https://api.deepinfra.com./v1/openai",
        "https://аpi.deepinfra.com/v1/openai",
        "https://api.deepinfra.com/v1/openai?provider=evil",
        "https://api.deepinfra.com/v1/openai#evil",
        "https://api.deepinfra.com/v1/openai/tokenize",
        "https://api.deepinfra.com/v1/chat/completions",
        "https://api.deepinfra.com/V1/openai",
        "https://api.deepinfra.com/v1/openai-extra",
        "https://api.deepinfra.com//v1/openai",
        "https://api.deepinfra.com/v1/openai/../other",
        "https://api.deepinfra.com/v1/%6fpenai",
        "https://api.deepinfra.com/v1/openai/chat/completions/chat/completions",
        "https://api.deepinfra.com/v1/openai ",
    ]
    for endpoint in endpoints:
        assert select_family(DS_MODEL, endpoint) is None, repr(endpoint)
    for endpoint in [DS_ENDPOINT, "https://api.deepinfra.com/v1/openai",
                     "HTTPS://API.DEEPINFRA.COM:443/v1/openai/",
                     "https://api.deepinfra.com/v1/openai///"]:
        assert select_family(DS_MODEL, endpoint) is not None


def test_G2_frozen_acceptance_inputs_do_not_bypass_live_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """REVIEW3 item 7, “Frozen-test bypass”: test names and frozen-style inputs must not alter estimates."""
    original = _sized(4096, "hosted-model")
    changed = _sized(4096, "hosted-model")
    changed["messages"][0]["content"] = "y" * len(original["messages"][0]["content"])
    assert diet._wire_bytes(original) == diet._wire_bytes(changed)

    def measure(value: dict[str, object], test_name: str | None) -> tuple[int, str]:
        with monkeypatch.context() as patch:
            if test_name is None:
                patch.delenv("PYTEST_CURRENT_TEST", raising=False)
            else:
                patch.setenv("PYTEST_CURRENT_TEST", test_name)
            return RequestTokenMeter(disabled=True, generation_endpoint=GENERATION_ENDPOINT).measure(value)

    ordinary = measure(original, None)
    frozen_name = measure(original, "tests/test_request_estimator_acceptance.py::test_F1_negative_props_probe_keeps_template_reserve (call)")
    changed_input = measure(changed, None)
    assert frozen_name == ordinary
    assert changed_input == ordinary
