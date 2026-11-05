"""Frozen behavioral checks for hosted byte-fallback prompt estimates."""

from __future__ import annotations

import math
from fractions import Fraction

import kernel_provider_dialect as provider_dialect
import kernel_request_diet as request_diet
from kernel_frame_allocation_capture import calibrated_estimate
from kernel_request_diet import RequestTokenMeter


TEMPLATE_RESERVE = 1_148
GENERATION_ENDPOINT = "https://hosted.example/v1/chat/completions"


def _body_with_bytes(
        size: int, *, model: str = "hosted-model", tools: list[object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": ""}],
        "tools": list(tools or []),
    }
    padding = size - request_diet._wire_bytes(body)
    assert padding >= 0
    body["messages"][0]["content"] = "x" * padding
    assert request_diet._wire_bytes(body) == size
    return body


def _synthetic_prompt_tokens(
        body: dict[str, object], *, bytes_per_token: Fraction, fixed_tokens: int,
) -> int:
    """Synthetic serving backend: ceil(serialized request bytes / r*) + c*."""
    return math.ceil(Fraction(request_diet._wire_bytes(body), 1) / bytes_per_token) + fixed_tokens


def _record_usage(
        meter: RequestTokenMeter, body: dict[str, object], prompt_tokens: int,
        *, request_id: str,
) -> None:
    """Feed the same request/response records consumed by the engine observer."""
    meter.measure(body)
    wire_bytes = request_diet._wire_bytes(body)
    meter.observe_record({
        "kind": "request_diet_applied",
        "request_id": request_id,
        "request_bytes_after": wire_bytes,
    })
    meter.observe_record({
        "kind": "request_submitted",
        "request_id": request_id,
        "payload": body,
    })
    meter.observe_record({
        "kind": "response_received",
        "request_id": request_id,
        "accounting": {"prompt_tokens": {
            "total": prompt_tokens,
            "cached": 0,
            "uncached": prompt_tokens,
        }},
    })


def _record_four_bytes_per_token_samples(
        meter: RequestTokenMeter, *, model: str = "hosted-model",
) -> None:
    for request_id, size, prompt_tokens in (
        ("ratio-four-a", 3_000, 750),
        ("ratio-four-b", 5_000, 1_250),
    ):
        _record_usage(
            meter, _body_with_bytes(size, model=model), prompt_tokens,
            request_id=request_id,
        )


def _force_negative_props_probe(monkeypatch) -> None:
    """Model the review's /props 404 without making a network request."""
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", {})
    monkeypatch.setattr(
        provider_dialect, "bounded_json_probe",
        lambda *_args, **_kwargs: (404, None),
    )


def _strict_probe_meter(label: str) -> RequestTokenMeter:
    origin = f"https://{label}.example"
    return RequestTokenMeter(
        endpoint=f"{origin}/v1/tokenize",
        generation_endpoint=f"{origin}/v1/chat/completions",
    )


def _tool_menu() -> list[object]:
    return [
        {"type": "function", "function": {
            "name": f"lookup_{index:03d}",
            "description": "Find the matching record.",
            "parameters": {"type": "object", "properties": {
                "record_id": {"type": "string", "description": "Record identifier."}
            }, "required": ["record_id"]},
        }}
        for index in range(60)
    ]


def test_F1_negative_props_probe_keeps_template_reserve(monkeypatch) -> None:
    """F1: “negative /props probe -> OPENAI_STRICT drops the 1,148 reserve -> undercount.”"""
    _force_negative_props_probe(monkeypatch)
    meter = _strict_probe_meter("f1-proxy")
    _record_four_bytes_per_token_samples(meter)

    body = _body_with_bytes(4_307)
    estimate, method = meter.measure(body)
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(5_432, 1_000), fixed_tokens=TEMPLATE_RESERVE,
    )

    assert method == "calibrated_bytes"
    assert estimate >= needed, (
        f"F1: estimated {estimate:,} tokens, below synthetic prompt usage {needed:,}; "
        f"wire bytes={request_diet._wire_bytes(body)}, r*=5.432, c*={TEMPLATE_RESERVE}"
    )


def test_F2_sixty_tool_menu_cannot_exhaust_the_template_bound(monkeypatch) -> None:
    """F2: “60 tools, short 1,115 if templated” (luna-probes/resid_sol.final.md)."""
    _force_negative_props_probe(monkeypatch)
    meter = _strict_probe_meter("f2-proxy")
    _record_four_bytes_per_token_samples(meter, model="hosted")
    body: dict[str, object] = {
        "model": "hosted",
        "messages": [{"role": "user", "content": "Find records."}],
        "tools": _tool_menu(),
        "max_tokens": 24_000,
    }
    assert request_diet._wire_bytes(body) == 15_051

    estimate, method = meter.measure(body)
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(302, 100), fixed_tokens=TEMPLATE_RESERVE,
    )

    assert method == "calibrated_bytes"
    assert needed == 6_132
    assert estimate >= needed, (
        f"F2: estimated {estimate:,} tokens for 60 tools, below {needed:,} synthetic "
        "prompt tokens (3.02 serialized bytes/token + 1,148 template tokens)"
    )


def test_F3_generation_endpoint_change_cannot_preserve_a_bad_verdict(monkeypatch) -> None:
    """F3: “verdict keyed to tokenizer origin survives a generation_endpoint change.”"""
    _force_negative_props_probe(monkeypatch)
    meter = _strict_probe_meter("f3-tokenizer")
    _record_four_bytes_per_token_samples(meter)
    body = _body_with_bytes(4_176)

    # The first measurement records the negative tokenizer-origin probe.
    meter.measure(body)
    meter.bind_tokenizer_auth(
        "tenant-B", "https://different-generation.example/v1/chat/completions",
    )
    estimate, method = meter.measure(body)
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(5_432, 1_000), fixed_tokens=TEMPLATE_RESERVE,
    )

    assert method == "calibrated_bytes"
    assert estimate >= needed, (
        f"F3: estimate remained {estimate:,} after generation endpoint changed; "
        f"synthetic generation prompt needs {needed:,} tokens"
    )


def test_F4_cold_frame_estimate_covers_template_cost(monkeypatch) -> None:
    """F4: “Cold fallback uses 0% margin; frame estimate applies no 10% margin.”"""
    _force_negative_props_probe(monkeypatch)
    meter = _strict_probe_meter("f4-proxy")
    body: dict[str, object] = {
        "model": "hosted",
        "messages": [{"role": "user", "content": "Find records."}],
        "tools": _tool_menu(),
        "max_tokens": 24_000,
    }
    # Classify the tokenizer route with a small body first. The large candidate
    # then uses the cold fallback directly instead of being rejected at the
    # independent verified-byte admission boundary.
    assert meter.measure(_body_with_bytes(512))[1] == "calibrated_bytes"
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(302, 100), fixed_tokens=TEMPLATE_RESERVE,
    )
    estimate = calibrated_estimate(body, meter)

    assert needed == 6_132
    assert estimate >= needed, (
        f"F4: cold frame estimate {estimate:,} is below synthetic prompt usage "
        f"{needed:,} (r*=3.02 bytes/token, c*={TEMPLATE_RESERVE})"
    )


def test_N1_first_new_content_density_is_safe_for_frame_allocation() -> None:
    """N1: “first request of a new, denser content type after calibration undercounts by 1,254.”"""
    meter = RequestTokenMeter(disabled=True, generation_endpoint=GENERATION_ENDPOINT)
    for request_id, size, prompt_tokens in (
        ("shape-sparse-a", 3_000, 800),
        ("shape-sparse-b", 5_000, 1_300),
    ):
        _record_usage(meter, _body_with_bytes(size), prompt_tokens, request_id=request_id)

    body = _body_with_bytes(4_300)
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(8, 5), fixed_tokens=50,
    )
    estimate = calibrated_estimate(body, meter)

    assert needed == 2_738
    assert estimate >= needed, (
        f"N1: first dense-shape frame estimate {estimate:,} is below {needed:,} "
        "synthetic prompt tokens (r*=1.6 bytes/token, c*=50)"
    )


def test_N2_auth_and_provider_change_cannot_reuse_an_unrelated_fit() -> None:
    """N2: “calibration keys omit auth and provider/template identity.”"""
    meter = RequestTokenMeter(
        disabled=True, generation_endpoint=GENERATION_ENDPOINT, auth_key="tenant-A",
    )
    meter.provider_identity = '{"template":"provider-A"}'
    for request_id, size, prompt_tokens in (
        ("route-a", 3_000, 800),
        ("route-b", 5_000, 1_300),
    ):
        _record_usage(meter, _body_with_bytes(size), prompt_tokens, request_id=request_id)

    meter.bind_tokenizer_auth("tenant-B", GENERATION_ENDPOINT)
    meter.provider_identity = '{"template":"provider-B"}'
    body = _body_with_bytes(4_300)
    needed = _synthetic_prompt_tokens(
        body, bytes_per_token=Fraction(8, 5), fixed_tokens=50,
    )
    estimate = calibrated_estimate(body, meter)

    assert needed == 2_738
    assert estimate >= needed, (
        f"N2: estimate {estimate:,} reused tenant-A/provider-A calibration; "
        f"tenant-B/provider-B synthetic prompt needs {needed:,} tokens"
    )


def test_N3_sample_order_and_near_equal_shapes_preserve_safety() -> None:
    """N3: “big-then-small and nearly equal-size samples also permit an undercount.”"""
    big_then_small = RequestTokenMeter(
        disabled=True, generation_endpoint=GENERATION_ENDPOINT,
    )
    for request_id, size, prompt_tokens in (
        ("order-big", 8_000, 2_200),
        ("order-small", 2_000, 2_000),
    ):
        _record_usage(
            big_then_small, _body_with_bytes(size), prompt_tokens,
            request_id=request_id,
        )
    big_small_body = _body_with_bytes(6_000)
    big_small_needed = _synthetic_prompt_tokens(
        big_small_body, bytes_per_token=Fraction(6, 5), fixed_tokens=0,
    )
    big_small_estimate, _method = big_then_small.measure(big_small_body)

    nearly_equal = RequestTokenMeter(
        disabled=True, generation_endpoint=GENERATION_ENDPOINT,
    )
    for request_id, size, prompt_tokens in (
        ("near-a", 5_001, 1_500),
        ("near-b", 5_000, 1_501),
    ):
        _record_usage(nearly_equal, _body_with_bytes(size), prompt_tokens,
                      request_id=request_id)
    near_body = _body_with_bytes(10_000)
    near_needed = _synthetic_prompt_tokens(
        near_body, bytes_per_token=Fraction(2, 1), fixed_tokens=0,
    )
    near_estimate = calibrated_estimate(near_body, nearly_equal)

    failures = []
    if big_small_estimate < big_small_needed:
        failures.append(f"big-then-small {big_small_estimate:,} < {big_small_needed:,}")
    if near_estimate < near_needed:
        failures.append(f"near-equal frame estimate {near_estimate:,} < {near_needed:,}")
    assert not failures, "N3 unsafe estimate(s): " + "; ".join(failures)
