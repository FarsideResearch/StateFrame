"""Launch-resolved advisory policy for the v28 compatibility runner."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_ON = frozenset({"1", "on", "true", "yes", "enabled"})
CONCISE_ACTION_GUIDANCE = "concise_action"
_CONFIG_PREFIXES = ("DS_", "STATEFRAME_")
_SERVED_MODEL_KEYS = (
    "DS_SERVED_MODEL",
    "REACT_SERVED_MODEL",
    "STATEFRAME_SERVED_MODEL",
    "SERVED_MODEL",
)


def _switch(value: object) -> bool:
    return str(value).strip().lower() in _ON


def _allowlist(value: object) -> frozenset[str]:
    return frozenset(item.strip().casefold() for item in str(value or "").split(",")
                     if item.strip())


def _model_families(identity: str) -> frozenset[str]:
    """Return explicit family aliases without turning unknown models on."""
    name = identity.strip().casefold()
    families = {name}
    if name.startswith("minicpm5-2b-q8"):
        families.add("minicpm5-2b-q8")
    if name == "a3b" or any(part == "a3b" for part in name.replace(":", "-").replace("/", "-").split("-")):
        families.add("a3b")
    return frozenset(families)


def _served_model(environ: Mapping[str, str] | None = None,
                  runtime: object | None = None) -> str:
    """Resolve allow-list identity from the verified serving path, not a label."""

    for attr in ("served_model", "served_model_identity", "runtime_model_identity"):
        value = getattr(runtime, attr, None) if runtime is not None else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = os.environ if environ is None else environ
    for key in _SERVED_MODEL_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class ModelPolicy:
    model_name: str
    guidance_by_family: Mapping[str, bool]
    truth_enabled: bool = True
    requested_model_name: str = ""
    served_model_name: str = ""

    def guidance_for(self, family: str) -> bool:
        return bool(self.guidance_by_family.get(str(family).strip().casefold(), False))

    def relaxed(self, _name: str) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {"model_name": self.model_name, "truth_enabled": self.truth_enabled,
                "guidance": dict(self.guidance_by_family),
                "requested_model": self.requested_model_name or self.model_name,
                "served_model": self.served_model_name or None}


def resolve_model_policy(model_name: str | None = None,
                         environ: Mapping[str, str] | None = None) -> ModelPolicy:
    source = os.environ if environ is None else environ
    requested = str(model_name or source.get("STATEFRAME_MODEL_NAME")
                    or source.get("DS_MODEL") or "unknown")
    served = _served_model(source)
    identity = served or requested
    raw = source.get("STATEFRAME_TOKENLAB_COORDINATION_GUIDANCE")
    if raw is None:
        raw = source.get("STATEFRAME_TOKENLAB_COORDINATED_LEDGER", "off")
    # Navigation hints are optional guidance only.  Search coverage, page
    # continuations, source freshness, and read permission never consult this flag.
    navigation = _switch(source.get("STATEFRAME_TOKENLAB_NAVIGATION_GUIDANCE", "off"))
    known_model = identity.strip().casefold() not in {"", "unknown", "none", "null"}
    concise_allowlist = _allowlist(
        source.get("STATEFRAME_TOKENLAB_CONCISE_ACTION_GUIDANCE_MODELS", ""))
    concise = (_switch(source.get("STATEFRAME_TOKENLAB_CONCISE_ACTION_GUIDANCE", "off"))
               and known_model
               and (bool(_model_families(identity) & concise_allowlist) or "*" in concise_allowlist))
    return ModelPolicy(
        identity,
        {
            "coordination": _switch(raw),
            "navigation": navigation,
            CONCISE_ACTION_GUIDANCE: concise,
        },
        requested_model_name=requested,
        served_model_name=served,
    )


def policy_for_runtime(runtime: object | None = None) -> ModelPolicy:
    execution = getattr(runtime, "execution_model_policy", None) if runtime else None
    if isinstance(execution, ModelPolicy):
        return execution
    source = dict(os.environ)
    served = _served_model(runtime=runtime)
    if served:
        source["DS_SERVED_MODEL"] = served
    return resolve_model_policy(
        getattr(runtime, "model_name", None) if runtime else None,
        environ=source,
    )


def execution_policy_for(runtime: object | None, model_name: object, *,
                         provider: str | None = None) -> ModelPolicy:
    """Resolve one selected endpoint policy without changing launch defaults."""
    identity = str(model_name or "").strip() or "unknown"
    allowed = getattr(runtime, "model_router_allowed_models", ()) if runtime is not None else ()
    if allowed and identity not in set(allowed):
        raise ValueError("router selected a model outside the launch policy set")
    source = dict(os.environ)
    source["STATEFRAME_MODEL_NAME"] = identity
    source["DS_MODEL"] = identity
    served = _served_model(runtime=runtime)
    if served:
        source["DS_SERVED_MODEL"] = served
    if provider:
        source["STATEFRAME_PROVIDER"] = str(provider).strip().lower()
    policy = resolve_model_policy(identity, environ=source)
    if runtime is not None:
        setattr(runtime, "execution_model_policy", policy)
    return policy


def _effective_config(runtime: object) -> dict[str, dict[str, str | None]]:
    declared = getattr(runtime, "declared_environment", None)
    declared = declared if isinstance(declared, Mapping) else {}
    sources = getattr(runtime, "environment_sources", None)
    sources = sources if isinstance(sources, Mapping) else {}
    keys = {
        key for key in os.environ if key.startswith(_CONFIG_PREFIXES)
    }
    keys.update(
        str(key) for key in declared
        if str(key).startswith(_CONFIG_PREFIXES)
    )
    return {
        key: {
            "value": os.environ.get(key),
            "source": str(sources.get(key, "spec.json" if key in declared else "process-environment")),
        }
        for key in sorted(keys)
    }


def journal_effective_policy(runtime: object, policy: ModelPolicy | None = None) -> ModelPolicy:
    """Journal or verify one effective policy without appending a resume tail."""
    resolved = policy or policy_for_runtime(runtime)
    if getattr(runtime, "_model_policy_journaled", False):
        return resolved
    journal = getattr(runtime, "journal", None)
    records = getattr(journal, "records", None)
    durable = ([event for event in records()
                if isinstance(event, Mapping) and event.get("kind") == "model_policy_resolved"]
               if callable(records) else [])
    expected = resolved.to_dict()
    effective_config = _effective_config(runtime)
    if durable:
        if any(
            event.get("policy") != expected
            or event.get("effective_config") != effective_config
            for event in durable
        ):
            raise ValueError("durable model policy differs from the resumed launch policy")
        setattr(runtime, "_model_policy_journaled", True)
        return resolved
    if callable(records) and any(
            isinstance(event, Mapping) and event.get("kind") == "continuation_writer_resumed"
            for event in records()):
        raise ValueError("resumed journal has no durable model policy")
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        emitter("model_policy_resolved", producer="kernel_model_policy",
                event_schema="stateframe-effective-config/1", dispatch=False,
                policy=expected, effective_config=effective_config,
                requested_model=expected["requested_model"],
                served_model=expected["served_model"])
    setattr(runtime, "_model_policy_journaled", True)
    return resolved


__all__ = [
    "CONCISE_ACTION_GUIDANCE", "ModelPolicy", "execution_policy_for",
    "journal_effective_policy", "policy_for_runtime", "resolve_model_policy",
]


# v30 telemetry is an overlay at this already-imported policy boundary.  The
# adapter is idempotent and does not alter authority, receipts, or tool policy.
try:
    from kernel_v30_token_contracts import install_telemetry_patch
    install_telemetry_patch()
except ImportError:
    pass
