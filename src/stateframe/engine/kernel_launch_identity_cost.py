"""Pure launch-identity, provider-boundary, and host-cost contracts.

This module intentionally has no provider, router, parser, or model dependency.
The final wire owner supplies the effective model/configuration and the exact
serialized body.  A router classification can request a transformation, but it
cannot certify that a request was reasoning-off.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


SCHEMA = "stateframe-launch-identity-cost/1"
CERTIFICATE_SCHEMA = "stateframe-run-certificate/1"
REASONING_PROFILE_ENV = "STATEFRAME_REASONING_PROFILE"
REASONING_MODE_ENV = "STATEFRAME_REASONING_MODE"
UNKNOWN = "UNKNOWN"
ENVIRONMENT = "ENVIRONMENT"
PROVIDER_MALFORMED = "PROVIDER_MALFORMED"
SEMANTIC_ERROR = "SEMANTIC_ERROR"
HARNESS_CONTRACT = "HARNESS_CONTRACT"
PHASES = ("extraction", "binding_indexing", "journaling", "request_delivery")
MAX_COUNTER = 10_000_000
MAX_ELAPSED_NS = 86_400_000_000_000


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def wire_digest(wire: bytes) -> str:
    """Digest exact serialized provider bytes, not a reconstructed object."""
    if not isinstance(wire, bytes):
        raise TypeError("wire must be bytes")
    return "sha256:" + hashlib.sha256(wire).hexdigest()


class LaunchIdentityError(ValueError):
    """Base error for invalid launch identity or provider-boundary data."""


class ProviderBoundaryContractError(LaunchIdentityError):
    """A host-owned request contract failed before provider dispatch."""

    outcome = HARNESS_CONTRACT
    failure_origin = "harness"
    transport_retryable = False
    harness_contract = True

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code).strip() or "provider_boundary_contract"
        self.detail = " ".join(str(detail).split())[:400]
        super().__init__(f"{self.code}: {self.detail}")


class ProviderMalformedResponseError(RuntimeError):
    """The provider returned a response that cannot enter semantic parsing."""

    outcome = PROVIDER_MALFORMED
    failure_origin = "provider"
    transport_retryable = False

    def __init__(self, detail: str) -> None:
        self.detail = " ".join(str(detail).split())[:400]
        super().__init__(self.detail)


@dataclass(frozen=True)
class ReasoningProfile:
    """Host-declared request policy; every default is an explicit mode."""

    mode: str = "on"
    source: str = "default"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"off", "on", "passthrough"}:
            raise ValueError("reasoning profile must be off, on or passthrough")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", str(self.source).strip() or "unknown")

    @property
    def off(self) -> bool:
        return self.mode == "off"

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "source": self.source}


def _profile_value(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("mode", value.get("reasoning_mode"))
    if value is None:
        return None
    result = str(value).strip().lower()
    return result or None


def reasoning_profile(
    *,
    manifest: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ReasoningProfile:
    """Resolve an explicit host profile without inspecting labels or responses."""
    if manifest is not None:
        for key in ("reasoning_profile", "reasoning_mode"):
            value = _profile_value(manifest.get(key))
            if value is not None:
                return ReasoningProfile(value, "manifest:" + key)
    source = os.environ if environ is None else environ
    profile = _profile_value(source.get(REASONING_PROFILE_ENV))
    mode = _profile_value(source.get(REASONING_MODE_ENV))
    if profile is not None and mode is not None and profile != mode:
        raise ValueError(
            f"{REASONING_PROFILE_ENV} and {REASONING_MODE_ENV} disagree"
        )
    if profile is not None:
        return ReasoningProfile(profile, "environment:" + REASONING_PROFILE_ENV)
    if mode is not None:
        return ReasoningProfile(mode, "environment:" + REASONING_MODE_ENV)
    return ReasoningProfile()


def apply_global_reasoning_off(body: Mapping[str, object]) -> dict[str, object]:
    """Return an explicit no-thinking copy for the final provider boundary."""
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    result = copy.deepcopy(dict(body))
    raw_kwargs = result.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: dict[str, object] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = copy.deepcopy(dict(raw_kwargs))
    else:
        raise ProviderBoundaryContractError(
            "malformed_template_kwargs",
            "chat_template_kwargs must be a mapping",
        )
    kwargs["enable_thinking"] = False
    result["chat_template_kwargs"] = kwargs
    result["reasoning_effort"] = "none"
    result["reasoning_budget_tokens"] = 0
    result["reasoning_mode"] = "off"
    return result


def assert_global_reasoning_off(body: Mapping[str, object]) -> None:
    """Prove all required off fields are present in the exact final body."""
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, Mapping) or kwargs.get("enable_thinking") is not False:
        raise ProviderBoundaryContractError(
            "reasoning_template_not_off",
            "final provider body must set chat_template_kwargs.enable_thinking=false",
        )
    if body.get("reasoning_effort") != "none":
        raise ProviderBoundaryContractError(
            "reasoning_effort_not_off",
            "final provider body must set reasoning_effort=none",
        )
    if body.get("reasoning_budget_tokens") != 0:
        raise ProviderBoundaryContractError(
            "reasoning_budget_not_off",
            "final provider body must set reasoning_budget_tokens=0",
        )
    if body.get("reasoning_mode") != "off":
        raise ProviderBoundaryContractError(
            "reasoning_mode_not_off",
            "final provider body must set reasoning_mode=off",
        )


def enforce_provider_body(
    body: Mapping[str, object],
    *,
    profile: ReasoningProfile | None = None,
) -> dict[str, object]:
    """Apply and prove the explicit profile at the last mutable boundary."""
    selected = profile or reasoning_profile()
    if selected.off:
        result = apply_global_reasoning_off(body)
        assert_global_reasoning_off(result)
        return result
    if selected.mode == "passthrough":
        raise ProviderBoundaryContractError(
            "reasoning_profile_unset",
            "final provider body cannot inherit an unspecified reasoning policy",
        )
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    return copy.deepcopy(dict(body))


_UNKNOWN_IDENTITY_VALUES = frozenset({"", "MISSING", "UNKNOWN", "ENVIRONMENT", "PASSTHROUGH"})


def _identity_value(value: object) -> object:
    """Normalize sentinel identity values so incomplete evidence stays incomplete."""
    if value is None:
        return UNKNOWN
    if isinstance(value, str) and value.strip().upper() in _UNKNOWN_IDENTITY_VALUES:
        return UNKNOWN
    return value


def deployed_source_digest(manifest: Mapping[str, object] | None) -> str:
    """Return a manifest-bound source identity, never a shell/config label."""
    if not isinstance(manifest, Mapping):
        return UNKNOWN
    source_digests = manifest.get("source_digests")
    if isinstance(source_digests, Mapping) and source_digests:
        if any(_identity_value(value) == UNKNOWN for value in source_digests.values()):
            return UNKNOWN
        explicit = manifest.get("deployed_source_digest")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return _digest({str(key): str(value) for key, value in source_digests.items()})
    explicit = manifest.get("deployed_source_digest")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return UNKNOWN


@dataclass(frozen=True)
class RunCertificate:
    """One receipt identity joining effective config and exact request bytes."""

    run_id: str
    effective_model: str
    deployed_source_digest: str
    idealpath_mode: str
    reasoning_mode: str
    request_body_digest: str
    served_config: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        fields = {
            "schema": CERTIFICATE_SCHEMA,
            "run_id": self.run_id,
            "effective_model": self.effective_model,
            "deployed_source_digest": self.deployed_source_digest,
            "idealpath_mode": self.idealpath_mode,
            "reasoning_mode": self.reasoning_mode,
            "request_body_digest": self.request_body_digest,
            "served_config": copy.deepcopy(dict(self.served_config)),
        }
        complete = all(
            _identity_value(fields[name]) != UNKNOWN
            for name in (
                "run_id",
                "effective_model",
                "deployed_source_digest",
                "idealpath_mode",
                "reasoning_mode",
                "request_body_digest",
            )
        )
        fields["identity_status"] = "COMPLETE" if complete else UNKNOWN
        return fields


def build_run_certificate(
    *,
    run_id: str,
    body: Mapping[str, object],
    wire: bytes,
    manifest: Mapping[str, object] | None,
    backend_identity: Mapping[str, object] | None,
    profile: ReasoningProfile | None = None,
) -> RunCertificate:
    """Join only host-observed identity with the exact final serialized body."""
    selected = profile or reasoning_profile(manifest=manifest)
    manifest_value = manifest if isinstance(manifest, Mapping) else {}
    backend = backend_identity if isinstance(backend_identity, Mapping) else {}
    configured = backend.get("configured_identity")
    configured = configured if isinstance(configured, Mapping) else {}
    served_config: dict[str, object] = {}
    for key in ("model", "server_build", "tokenizer", "chat_template", "cache_policy"):
        value = backend.get(key, configured.get(key, manifest_value.get(key, UNKNOWN)))
        served_config[key] = _identity_value(value)
    explicit_model = backend.get("effective_model")
    if explicit_model not in (None, ""):
        model = explicit_model
    elif backend.get("router_applied") is True:
        model = backend.get("model", body.get("model", UNKNOWN))
    else:
        # With no independently observed routed backend, the model in the
        # exact final body is the effective request identity.  A configured
        # manifest model must not mask a per-request override.
        model = body.get("model", served_config.get("model", UNKNOWN))
    if model in (None, ""):
        model = UNKNOWN
    idealpath = manifest_value.get(
        "stateframe_idealpath", manifest_value.get("idealpath_mode", UNKNOWN)
    )
    idealpath = str(idealpath).strip().lower() if idealpath not in (None, "") else UNKNOWN
    if idealpath == "environment":
        idealpath = "ENVIRONMENT"
    elif idealpath in {"unknown", "missing"}:
        idealpath = UNKNOWN
    elif idealpath not in {"on", "off"}:
        idealpath = UNKNOWN
    return RunCertificate(
        run_id=str(run_id).strip() or UNKNOWN,
        effective_model=str(_identity_value(model)),
        deployed_source_digest=str(_identity_value(deployed_source_digest(manifest_value))),
        idealpath_mode=idealpath.upper() if idealpath == UNKNOWN.lower() else idealpath,
        reasoning_mode=selected.mode,
        request_body_digest=wire_digest(wire),
        served_config=served_config,
    )


class PhaseMeter:
    """Fixed-shape bounded host instrumentation; no per-fact timing list."""

    def __init__(self) -> None:
        self._rows = {
            phase: {"count": 0, "items": 0, "cache_hits": 0, "elapsed_ns": 0}
            for phase in PHASES
        }

    @staticmethod
    def _bounded(value: object, limit: int = MAX_COUNTER) -> int:
        try:
            numeric = int(value)
        except (OverflowError, TypeError, ValueError):
            return 0
        return max(0, min(limit, numeric))

    def record(
        self,
        phase: str,
        elapsed_ns: int,
        *,
        count: int = 1,
        items: int = 0,
        cache_hit: bool = False,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown host-cost phase: {phase}")
        row = self._rows[phase]
        row["count"] = min(MAX_COUNTER, row["count"] + self._bounded(count))
        row["items"] = min(MAX_COUNTER, row["items"] + self._bounded(items))
        row["cache_hits"] = min(MAX_COUNTER, row["cache_hits"] + int(cache_hit))
        row["elapsed_ns"] = min(
            MAX_ELAPSED_NS,
            row["elapsed_ns"] + self._bounded(elapsed_ns, MAX_ELAPSED_NS),
        )

    @contextmanager
    def measure(
        self,
        phase: str,
        *,
        count: int = 1,
        items: int = 0,
        cache_hit: bool = False,
    ) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(
                phase,
                time.perf_counter_ns() - started,
                count=count,
                items=items,
                cache_hit=cache_hit,
            )

    def merge(self, value: Mapping[str, object]) -> None:
        phases = value.get("phases") if isinstance(value, Mapping) else None
        if not isinstance(phases, Mapping):
            return
        for phase in PHASES:
            row = phases.get(phase)
            if not isinstance(row, Mapping):
                continue
            current = self._rows[phase]
            current["count"] = min(
                MAX_COUNTER,
                current["count"] + self._bounded(row.get("count", 0)),
            )
            current["items"] = min(
                MAX_COUNTER,
                current["items"] + self._bounded(row.get("items", 0)),
            )
            current["cache_hits"] = min(
                MAX_COUNTER,
                current["cache_hits"] + self._bounded(row.get("cache_hits", 0)),
            )
            current["elapsed_ns"] = min(
                MAX_ELAPSED_NS,
                current["elapsed_ns"]
                + self._bounded(row.get("elapsed_ns", 0), MAX_ELAPSED_NS),
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "bounded": True,
            "timing_scope": "instrumented_phase_only",
            "phases": {phase: dict(row) for phase, row in self._rows.items()},
        }


__all__ = [
    "CERTIFICATE_SCHEMA",
    "ENVIRONMENT",
    "HARNESS_CONTRACT",
    "MAX_COUNTER",
    "PHASES",
    "PROVIDER_MALFORMED",
    "SEMANTIC_ERROR",
    "ProviderBoundaryContractError",
    "ProviderMalformedResponseError",
    "ReasoningProfile",
    "RunCertificate",
    "PhaseMeter",
    "UNKNOWN",
    "apply_global_reasoning_off",
    "assert_global_reasoning_off",
    "build_run_certificate",
    "deployed_source_digest",
    "enforce_provider_body",
    "reasoning_profile",
    "wire_digest",
]
