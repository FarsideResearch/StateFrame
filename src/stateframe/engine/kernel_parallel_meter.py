"""Pinned, conservative first-request estimates for parallel subframes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from urllib.parse import urlsplit

if __package__:
    from .kernel_request_diet import _wire_bytes
else:
    from kernel_request_diet import _wire_bytes


_DEEPINFRA_PROVIDER = "deepinfra"
_DEEPINFRA_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
_DEEPINFRA_PROFILE_ID = "deepinfra/deepseek-ai/deepseek-v4-flash-0731@v1"
_GENERIC_PROFILE_ID = "generic-conservative@v1"
_DEFAULT_HEADROOM_PCT = 10
_MAX_HEADROOM_PCT = 50
_GENERIC_FLOOR_TOKENS_PER_BYTE = 1 / 3

# Arm A measured a max 0.290301 tokens/byte over 80 parent requests. The
# versioned DeepInfra profile uses the design's rounded 0.3193 ratio (10%
# margin); only this exact provider/model pair is eligible for that profile.
_MODEL_PROFILES = {
    (_DEEPINFRA_PROVIDER, _DEEPINFRA_MODEL): {
        "calibration_id": _DEEPINFRA_PROFILE_ID,
        "version": 1,
        "observed_max_tokens_per_byte": 0.2903,
        "tokens_per_byte_at_10_pct": 0.3193,
        "residual_tokens": 0,
    },
}
_GENERIC_OBSERVED_MAX_TOKENS_PER_BYTE = max(
    profile["observed_max_tokens_per_byte"] for profile in _MODEL_PROFILES.values())


@dataclass(frozen=True)
class SubframeCapMeasurement:
    tokens: int
    meter_kind: str
    confidence: str
    calibration_id: str
    effective_tokens_per_byte: float | None
    request_bytes: int
    residual_tokens: int

    def audit_fields(self, cap: int) -> dict[str, object]:
        return {
            "meter_kind": self.meter_kind,
            "confidence": self.confidence,
            "calibration_id": self.calibration_id,
            "effective_tokens_per_byte": self.effective_tokens_per_byte,
            "request_bytes": self.request_bytes,
            "estimated_tokens": self.tokens,
            "cap": cap,
        }


def _identity_mapping(provider_identity: object) -> dict[str, object]:
    if isinstance(provider_identity, Mapping):
        return dict(provider_identity)
    if isinstance(provider_identity, str):
        try:
            parsed = json.loads(provider_identity)
        except ValueError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _identity_pair(provider_identity: object, endpoint: object,
                   body_model: object) -> tuple[str | None, str | None]:
    identity = _identity_mapping(provider_identity)
    model_values = [value for value in (identity.get("model"), body_model)
                    if isinstance(value, str) and value]
    model = model_values[0] if model_values and len(set(model_values)) == 1 else None
    provider = identity.get("provider")
    if isinstance(provider, str) and provider.strip():
        normalized_provider = provider.strip().casefold()
    else:
        endpoint_value = endpoint
        if not isinstance(endpoint_value, str) or not endpoint_value:
            endpoint_value = next((identity.get(key) for key in ("endpoint", "tokenizer", "base_url")
                                   if isinstance(identity.get(key), str)), None)
        try:
            host = (urlsplit(endpoint_value).hostname or "").casefold()
        except (TypeError, ValueError):
            host = ""
        normalized_provider = (_DEEPINFRA_PROVIDER if host == "api.deepinfra.com"
                               or host.endswith(".deepinfra.com") else None)
    return normalized_provider, model


def _headroom_pct() -> int:
    raw = os.environ.get("STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT", "10").strip()
    if not raw.isascii() or not raw.isdecimal():
        return _DEFAULT_HEADROOM_PCT
    value = int(raw)
    return value if 0 <= value <= _MAX_HEADROOM_PCT else _DEFAULT_HEADROOM_PCT


def _generic_tokens_per_byte(headroom_pct: int) -> float:
    observed_with_headroom = _GENERIC_OBSERVED_MAX_TOKENS_PER_BYTE * (1 + headroom_pct / 100)
    return max(_GENERIC_FLOOR_TOKENS_PER_BYTE, observed_with_headroom)


def _generic_calibration(provider: str | None, model: str | None,
                         headroom_pct: int) -> dict[str, object]:
    return {
        "schema": "stateframe-subframe-byte-token-profile/1",
        "calibration_id": _GENERIC_PROFILE_ID,
        "provider": provider,
        "model": model,
        "version": 1,
        "tokens_per_byte": _generic_tokens_per_byte(headroom_pct),
        "residual_tokens": 0,
        "observed_max_tokens_per_byte": _GENERIC_OBSERVED_MAX_TOKENS_PER_BYTE,
        "headroom_pct": headroom_pct,
        "confidence": "default",
    }


def select_subframe_cap_calibration(
        provider_identity: object = None, *, endpoint: object = None,
        model: object = None) -> dict[str, object]:
    """Return a serializable, versioned profile; unknown identities stay conservative."""
    provider, identity_model = _identity_pair(provider_identity, endpoint, model)
    headroom_pct = max(_DEFAULT_HEADROOM_PCT, _headroom_pct())
    selected = os.environ.get("STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION", "auto").strip().lower()
    known = _MODEL_PROFILES.get((provider, identity_model))
    if (known is not None and selected in {"auto", known["calibration_id"]}):
        ratio = (known["tokens_per_byte_at_10_pct"] if headroom_pct == _DEFAULT_HEADROOM_PCT else
                 known["observed_max_tokens_per_byte"] * (1 + headroom_pct / 100))
        return {
            "schema": "stateframe-subframe-byte-token-profile/1",
            "calibration_id": known["calibration_id"],
            "provider": provider,
            "model": identity_model,
            "version": known["version"],
            "tokens_per_byte": ratio,
            "residual_tokens": known["residual_tokens"],
            "observed_max_tokens_per_byte": known["observed_max_tokens_per_byte"],
            "headroom_pct": headroom_pct,
            "confidence": "default",
        }
    return _generic_calibration(provider, identity_model, headroom_pct)


def _pinned_calibration(meter: object, provider_identity: object,
                        endpoint: object, model: object) -> dict[str, object]:
    pinned = getattr(meter, "subframe_cap_calibration", None)
    selected = select_subframe_cap_calibration(provider_identity, endpoint=endpoint, model=model)
    if not isinstance(pinned, Mapping):
        return selected
    ratio = pinned.get("tokens_per_byte")
    residual = pinned.get("residual_tokens")
    calibration_id = pinned.get("calibration_id")
    headroom_pct = pinned.get("headroom_pct")
    observed_max = pinned.get("observed_max_tokens_per_byte")
    if (pinned.get("schema") != "stateframe-subframe-byte-token-profile/1"
            or not isinstance(calibration_id, str)
            or type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0
            or type(residual) is not int or residual < 0
            or type(headroom_pct) is not int
            or not _DEFAULT_HEADROOM_PCT <= headroom_pct <= _MAX_HEADROOM_PCT
            or type(observed_max) not in (int, float)
            or not math.isfinite(observed_max) or observed_max <= 0
            or type(pinned.get("version")) is not int or pinned.get("version") != 1
            or pinned.get("confidence") != "default"):
        raise ValueError("pinned subframe byte-token profile is invalid")
    provider, identity_model = _identity_pair(provider_identity, endpoint, model)
    generic_fallback = _generic_calibration(
        provider, identity_model, max(_DEFAULT_HEADROOM_PCT, _headroom_pct()))
    if calibration_id == _GENERIC_PROFILE_ID:
        minimum_ratio = _generic_tokens_per_byte(headroom_pct)
        if (not math.isclose(observed_max, _GENERIC_OBSERVED_MAX_TOKENS_PER_BYTE,
                             rel_tol=0.0, abs_tol=1e-12)
                or residual != 0 or ratio < minimum_ratio):
            return generic_fallback
        return dict(pinned)

    expected_profile = next(
        (profile for profile in _MODEL_PROFILES.values()
         if profile["calibration_id"] == calibration_id), None)
    if expected_profile is None:
        return generic_fallback
    expected_provider, expected_model = next(
        identity for identity, profile in _MODEL_PROFILES.items()
        if profile["calibration_id"] == calibration_id)
    expected_ratio = (expected_profile["tokens_per_byte_at_10_pct"]
                      if headroom_pct == _DEFAULT_HEADROOM_PCT else
                      expected_profile["observed_max_tokens_per_byte"]
                      * (1 + headroom_pct / 100))
    if (pinned.get("provider") != expected_provider or pinned.get("model") != expected_model
            or provider != pinned.get("provider") or identity_model != pinned.get("model")
            or not math.isclose(observed_max, expected_profile["observed_max_tokens_per_byte"],
                                rel_tol=0.0, abs_tol=1e-12)
            or residual != expected_profile["residual_tokens"]
            or ratio < expected_ratio):
        return generic_fallback
    return dict(pinned)


def _contains_image_input(body: Mapping[str, object]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, (list, tuple)):
        return False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if (isinstance(kind, str) and kind in {"image", "image_url", "input_image"}) or any(
                    key in part for key in ("image_url", "image", "input_image")):
                return True
    return False


def estimate_subframe_request_bytes(request_bytes: int,
                                    calibration: Mapping[str, object]) -> int:
    """Apply a pinned byte ratio with ceiling arithmetic and no usage feedback."""
    ratio = calibration.get("tokens_per_byte")
    residual = calibration.get("residual_tokens")
    if (type(request_bytes) is not int or request_bytes < 0
            or type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0
            or type(residual) is not int or residual < 0):
        raise ValueError("subframe byte-token estimate inputs are invalid")
    return math.ceil(request_bytes * ratio + residual)


def measure_subframe_cap(body: Mapping[str, object], meter: object, *,
                         provider_identity: object = None, cap: int | None = None
                         ) -> SubframeCapMeasurement:
    """Measure an exact tokenizer count when available, else the pinned safe byte profile."""
    size = _wire_bytes(body)
    endpoint = getattr(meter, "endpoint", None)
    identity = provider_identity if provider_identity is not None else getattr(meter, "provider_identity", None)
    calibration = _pinned_calibration(meter, identity, endpoint, body.get("model"))
    if _contains_image_input(body):
        return SubframeCapMeasurement(
            cap + 1 if type(cap) is int and cap >= 0 else 2_147_483_647,
            "unknown_content", "unknown", str(calibration["calibration_id"]), None, size, 0)
    method = getattr(meter, "measure_subframe_cap", None)
    if callable(method):
        tokens, meter_kind = method(
            body, tokens_per_byte=float(calibration["tokens_per_byte"]),
            residual_tokens=int(calibration["residual_tokens"]))
        if meter_kind == "tokenize":
            return SubframeCapMeasurement(
                tokens, "tokenize", "exact", "tokenize-exact", None, size, 0)
        if meter_kind != "calibrated_bytes":
            raise ValueError("request meter returned an unsupported subframe measurement")
    else:
        tokens = estimate_subframe_request_bytes(size, calibration)
        meter_kind = "calibrated_bytes"
    return SubframeCapMeasurement(
        estimate_subframe_request_bytes(size, calibration), meter_kind, "default",
        str(calibration["calibration_id"]), float(calibration["tokens_per_byte"]),
        size, int(calibration["residual_tokens"]))


def _meter_calibration_snapshot(meter: object) -> dict[str, object] | None:
    """Return the meter's own snapshot; a meter without the API stays cold."""
    method = getattr(meter, "calibration_snapshot", None)
    if method is None:
        return None
    snapshot = method()
    if not isinstance(snapshot, dict):
        raise ValueError("request meter calibration snapshot must be an object")
    return snapshot


def _meter_source(meter: object) -> str:
    snapshot = _meter_calibration_snapshot(meter)
    if snapshot is None:
        return 'calibrated_bytes'
    source = snapshot.get('source')
    if not isinstance(source, str) or not source:
        raise ValueError('request meter calibration source must be text')
    return source


def _parent_meter_calibration(runtime: object) -> dict[str, object] | None:
    if __package__:
        from .kernel_request_diet import RequestTokenMeter
    else:
        from kernel_request_diet import RequestTokenMeter
    for owner in (getattr(runtime, "_last_capture", None),
                  getattr(runtime, "_active_capture", None), runtime):
        meter = getattr(owner, "_request_token_meter", None)
        if isinstance(meter, RequestTokenMeter):
            return _meter_calibration_snapshot(meter)
    return None


def _parent_subframe_cap_calibration(runtime: object) -> dict[str, object]:
    """Pin the selected byte profile and numeric values before worker launch."""
    for owner in (getattr(runtime, "_last_capture", None),
                  getattr(runtime, "_active_capture", None), runtime):
        meter = getattr(owner, "_request_token_meter", None)
        if meter is not None:
            return select_subframe_cap_calibration(
                getattr(meter, "provider_identity", None),
                endpoint=getattr(meter, "endpoint", None))
    return select_subframe_cap_calibration()


def _meter_for_unit_request(runtime: object | None, capture: object,
                            existing: object | None) -> object | None:
    """Factory hook: restore once, before any first-request fitting pass."""
    if runtime is None:
        return existing
    policy = getattr(runtime, '_unit_policy', None)
    spec = getattr(policy, 'spec', None)
    if existing is not None:
        if spec is not None:
            setattr(existing, "subframe_cap_calibration",
                    getattr(spec, "subframe_cap_calibration", None))
        return existing
    snapshot = getattr(spec, 'meter_calibration', None)
    if snapshot is None:
        return None
    meter = _meter_from_calibration(snapshot, runtime=runtime, capture=capture)
    setattr(meter, "subframe_cap_calibration",
            getattr(spec, "subframe_cap_calibration", None))
    return meter


def _meter_from_calibration(snapshot: Mapping[str, object] | None, *,
                            runtime: object, capture: object):
    """Install a restored meter in the existing runtime/capture meter slot."""
    if __package__:
        from .kernel_request_diet import (RequestTokenMeter, request_tokenizer_credentials,
                                          resolve_tokenize_url, tokenizer_contract)
        from . import kernel_frame_routes
    else:
        from kernel_request_diet import (RequestTokenMeter, request_tokenizer_credentials,
                                         resolve_tokenize_url, tokenizer_contract)
        import kernel_frame_routes

    endpoint = resolve_tokenize_url(runtime, capture,
                                    endpoint=getattr(capture, "_tokenize_endpoint", None))
    routed = (kernel_frame_routes.ROUTES_ACTIVE
              and getattr(capture, "frame_route", None) is not None)
    route_allow_remote = (bool(capture.frame_route.settings.get("allow_remote"))
                          if routed else None)
    auth_key, generation_endpoint = request_tokenizer_credentials(capture)
    constructor = getattr(RequestTokenMeter, "from_calibration", None)
    if snapshot is not None and constructor is not None:
        meter = constructor(snapshot, endpoint=endpoint, route_allow_remote=route_allow_remote,
                            auth_key=auth_key, generation_endpoint=generation_endpoint)
    else:
        meter = RequestTokenMeter(endpoint=endpoint, route_allow_remote=route_allow_remote,
                                  auth_key=auth_key, generation_endpoint=generation_endpoint)
    meter.provider_identity = tokenizer_contract(runtime, capture)
    request_id = getattr(capture, "request_id", None)
    logical = (getattr(capture, "_tokenize_logical_request_id", None)
               or getattr(capture, "_replacement_lineage", None) or request_id)
    if isinstance(request_id, str) and isinstance(logical, str):
        meter.bind_request(request_id, logical,
                           deadline=getattr(capture, "_tokenize_deadline", None))
    meter.attach_journal(getattr(capture, "journal", None))
    setattr(capture if routed else runtime, "_request_token_meter", meter)
    return meter
