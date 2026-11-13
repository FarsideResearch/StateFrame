"""Proof-bound endpoint dialect and native feature capabilities."""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

try:  # Host runner imports the helper as a package; the engine loads it top-level.
    from .kernel_bounded_http import bounded_json_probe
    from .kernel_launch_identity_cost import (
        OPENAI_REASONING_EFFORTS, ProviderBoundaryContractError,
        require_strict_side_call_reasoning_off,
    )
except ImportError:
    from kernel_bounded_http import bounded_json_probe
    from kernel_launch_identity_cost import (
        OPENAI_REASONING_EFFORTS, ProviderBoundaryContractError,
        require_strict_side_call_reasoning_off,
    )


LLAMA_CPP = "llama.cpp"
OPENAI_STRICT = "openai-strict"
PROVIDER_OVERRIDE_ENV = "STATEFRAME_PROVIDER"
RUN_CAPABILITY_CACHE: dict[str, "ProviderCapabilities"] = {}
RUN_DIALECT_JOURNALED: set[str] = set()
_ORIGIN_LOCKS: dict[str, threading.Lock] = {}
_ORIGIN_LOCKS_GUARD = threading.Lock()
_JOURNAL_GUARD = threading.Lock()


@dataclass(frozen=True)
class ProviderCapabilities:
    dialect: str = OPENAI_STRICT
    slots: bool = False
    slot_ids: frozenset[int] = frozenset()
    reason: str = "openai-strict because /props was unreachable"
    props_status: int | None = None
    slots_status: int | None = None

    @property
    def continuation(self) -> bool:
        return self.dialect == LLAMA_CPP

    def to_record(self) -> dict[str, object]:
        return {
            "dialect": self.dialect,
            "slots": self.slots,
            "slot_ids": sorted(self.slot_ids),
            "reason": self.reason,
            "props_status": self.props_status,
            "slots_status": self.slots_status,
        }


def seed_run_capability_cache(records: object) -> None:
    """Install host-proven capabilities before the engine starts sending requests."""
    RUN_CAPABILITY_CACHE.clear()
    with _JOURNAL_GUARD:
        RUN_DIALECT_JOURNALED.clear()
    if not isinstance(records, Mapping):
        return
    for origin, value in records.items():
        if not isinstance(origin, str) or origin_of(origin) != origin or not isinstance(value, Mapping):
            continue
        dialect = value.get("dialect")
        slots = value.get("slots")
        slot_ids = value.get("slot_ids")
        reason = value.get("reason")
        props_status = value.get("props_status")
        slots_status = value.get("slots_status")
        if (dialect not in {LLAMA_CPP, OPENAI_STRICT} or type(slots) is not bool
                or not isinstance(slot_ids, list)
                or any(type(slot_id) is not int or slot_id < 0 for slot_id in slot_ids)
                or not isinstance(reason, str)
                or (props_status is not None and type(props_status) is not int)
                or (slots_status is not None and type(slots_status) is not int)):
            continue
        RUN_CAPABILITY_CACHE[origin] = ProviderCapabilities(
            dialect=dialect, slots=slots, slot_ids=frozenset(slot_ids), reason=reason,
            props_status=props_status, slots_status=slots_status,
        )


def mark_dialect_journaled(origin: str) -> bool:
    """Return true once per run when the first request selects an origin."""
    if not origin:
        return False
    with _JOURNAL_GUARD:
        if origin in RUN_DIALECT_JOURNALED:
            return False
        RUN_DIALECT_JOURNALED.add(origin)
        return True


class ProviderCapabilityError(ProviderBoundaryContractError):
    """A selected native feature lacks proof on its selected endpoint."""

    def __init__(self, feature: str, capability: str) -> None:
        self.feature = feature
        self.capability = capability
        super().__init__(
            "provider_capability_missing",
            f"{feature} requires proven {capability} capability",
        )


def origin_of(endpoint: str) -> str:
    """Return a credential-free HTTP origin for probes and journal records."""
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not host:
        return ""
    authority = f"[{host}]" if ":" in host else host
    default_port = 80 if scheme == "http" else 443
    if port is not None and port != default_port:
        authority += f":{port}"
    return urllib.parse.urlunsplit((scheme, authority, "", "", ""))


def _llama_props(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    settings = payload.get("default_generation_settings")
    if not isinstance(settings, Mapping):
        return False
    context = payload.get("n_ctx", settings.get("n_ctx"))
    model_path = payload.get("model_path", settings.get("model_path"))
    return (type(context) is int and context > 0
            or isinstance(model_path, str) and bool(model_path.strip()))


def _slots_proof(
    result: tuple[int, object] | None, *, require_llama_shape: bool = False,
) -> tuple[bool, frozenset[int], int | None]:
    if result is None:
        return False, frozenset(), None
    status, payload = result
    if status != 200 or not isinstance(payload, list):
        return False, frozenset(), status
    if not require_llama_shape:
        # /props already established the dialect; preserve the older slot contract.
        slot_ids = frozenset(
            item["id"] for item in payload
            if isinstance(item, Mapping) and type(item.get("id")) is int and item["id"] >= 0
        )
        return True, slot_ids, status
    if not payload:
        return False, frozenset(), status
    slot_ids: set[int] = set()
    for item in payload:
        if (not isinstance(item, Mapping)
                or type(item.get("id")) is not int or item["id"] < 0
                or type(item.get("n_ctx")) is not int or item["n_ctx"] <= 0
                or type(item.get("is_processing")) is not bool):
            return False, frozenset(), status
        slot_ids.add(item["id"])
    if len(slot_ids) != len(payload):
        return False, frozenset(), status
    return True, frozenset(slot_ids), status


def capabilities_for(
    endpoint: str,
    key: str,
    *,
    deadline: float | None = None,
    require_slots: bool = True,
    cache: MutableMapping[str, ProviderCapabilities] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProviderCapabilities:
    """Classify the selected origin once per run, using its selected API key.

    The engine child owns the default module cache for one run and seeds it from
    the host's route preflight. Standalone adapters can pass their own run cache.
    """
    origin = origin_of(endpoint)
    if not origin:
        return ProviderCapabilities()
    selected_cache = RUN_CAPABILITY_CACHE if cache is None else cache
    cached = selected_cache.get(origin)
    if cached is not None and (
        not require_slots or cached.slots or cached.slots_status is not None
    ):
        return cached
    with _ORIGIN_LOCKS_GUARD:
        origin_lock = _ORIGIN_LOCKS.setdefault(origin, threading.Lock())
    if deadline is None:
        origin_lock.acquire()
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not origin_lock.acquire(timeout=remaining):
            raise TimeoutError("tokenizer request deadline expired")
    try:
        cached = selected_cache.get(origin)
        if cached is not None and (
            not require_slots or cached.slots or cached.slots_status is not None
        ):
            return cached
        cached_llama_proof = None
        if (cached is not None and cached.dialect == LLAMA_CPP
                and not cached.slots and cached.slots_status is None):
            cached_llama_proof = cached

        source = os.environ if environ is None else environ
        override = str(source.get(PROVIDER_OVERRIDE_ENV, "")).strip().casefold() == LLAMA_CPP

        def probe(path: str) -> tuple[tuple[int, object] | None, bool]:
            if deadline is None:
                return bounded_json_probe(origin, key, path), False
            if deadline <= time.monotonic():
                return None, True
            result = bounded_json_probe(origin, key, path, deadline=deadline)
            return result, result is None and time.monotonic() >= deadline

        if cached_llama_proof is not None:
            # A later slot request upgrades the cached dialect proof without
            # redetecting it through an endpoint whose /props may now be absent.
            props, props_cut_short = None, False
            props_status = cached_llama_proof.props_status
            props_is_llama = True
        else:
            props, props_cut_short = probe("/props")
            props_status = props[0] if props is not None else None
            props_is_llama = props is not None and props[0] == 200 and _llama_props(props[1])

        # Dialect proof for tokenization needs /props, never slot discovery.
        slots_probe_skipped = not require_slots
        if slots_probe_skipped:
            slots_result, slots_cut_short = None, False
        else:
            slots_result, slots_cut_short = probe("/slots")
        slots, slot_ids, slots_status = _slots_proof(
            slots_result,
            require_llama_shape=not (override or props_is_llama),
        )

        if cached_llama_proof is not None:
            dialect = LLAMA_CPP
            reason = cached_llama_proof.reason
        elif override:
            dialect = LLAMA_CPP
            if props is None:
                reason = "operator override, /props unreachable"
            elif props_is_llama:
                reason = "operator override, /props answered llama.cpp shape"
            elif props_status == 404:
                reason = "operator override, /props answered 404"
            elif props_status == 200:
                reason = "operator override, /props answered non-llama shape"
            else:
                reason = f"operator override, /props answered HTTP {props_status}"
        elif props_is_llama:
            dialect = LLAMA_CPP
            reason = "proven by /props"
        elif slots:
            # A populated llama.cpp slot array has its own typed fields and remains
            # authoritative when a relay or gateway omits the separate /props route.
            dialect = LLAMA_CPP
            reason = "proven by /slots"
        elif props is None:
            dialect = OPENAI_STRICT
            reason = "openai-strict because /props was unreachable and /slots was unproven"
        elif props_status == 404:
            dialect = OPENAI_STRICT
            reason = "openai-strict because /props was 404 and /slots was unproven"
        else:
            dialect = OPENAI_STRICT
            reason = "openai-strict because /props was not llama.cpp and /slots was unproven"

        result = ProviderCapabilities(
            dialect=dialect,
            slots=slots,
            slot_ids=slot_ids,
            reason=reason,
            props_status=props_status,
            slots_status=slots_status,
        )
        if cached_llama_proof is not None:
            if not slots_cut_short:
                selected_cache[origin] = result
        elif slots_probe_skipped and props_is_llama:
            # Keep a successful /props proof even though this caller did not
            # need /slots. A later caller can upgrade it if slots are required.
            selected_cache[origin] = result
        elif not (props_cut_short or slots_cut_short or slots_probe_skipped):
            selected_cache[origin] = result
        return result
    finally:
        origin_lock.release()


def require_feature(
    capabilities: ProviderCapabilities,
    feature: str,
    *,
    slots: bool = False,
    continuation: bool = False,
    slot_id: int | None = None,
) -> None:
    missing: list[str] = []
    if capabilities.dialect != LLAMA_CPP:
        missing.append(LLAMA_CPP)
    if slots and not capabilities.slots:
        missing.append("/slots")
    if continuation and not capabilities.continuation:
        missing.append("llama.cpp continuation")
    if slot_id is not None and slot_id not in capabilities.slot_ids:
        missing.append(f"/slots id {slot_id}")
    if missing:
        raise ProviderCapabilityError(feature, " and ".join(missing))


def require_routed_static_slot(
    capabilities: ProviderCapabilities,
    slot_id: int | None = None,
) -> None:
    """Require proof for a route's explicitly configured static or automatic slot."""
    require_feature(
        capabilities, "routed static slot", slots=True, slot_id=slot_id,
    )


def require_selected_request_features(
    capabilities: ProviderCapabilities,
    body: Mapping[str, object],
    *,
    route_slot: int | None = None,
    call_purpose: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse explicitly selected native request features before provider dispatch."""
    source = os.environ if environ is None else environ
    if source.get("STATEFRAME_KV_CHUNK_REUSE", "off").strip().lower() in {
            "on", "1", "true", "yes"}:
        require_feature(capabilities, "KV_CHUNK_REUSE", slots=True)
    prompt_cache_enabled = source.get("STATEFRAME_PROMPT_CACHE", "off").strip().lower() in {
        "on", "1", "true", "yes",
    }
    raw_slot = source.get("REACT_SLOT_ID")
    prompt_cache_slot_selected = (
        "id_slot" in body or body.get("cache_prompt") is True
        or raw_slot is not None or type(route_slot) is int
    )
    if prompt_cache_enabled and prompt_cache_slot_selected:
        require_feature(capabilities, "STATEFRAME_PROMPT_CACHE", slots=True)
        if raw_slot is None:
            prompt_cache_slot = body.get("id_slot", route_slot)
        else:
            try:
                prompt_cache_slot = int(raw_slot)
            except (TypeError, ValueError) as error:
                raise ProviderBoundaryContractError(
                    "invalid_id_slot", "REACT_SLOT_ID must be an explicitly selected nonnegative integer",
                ) from error
        if type(prompt_cache_slot) is not int or prompt_cache_slot < 0:
            raise ProviderCapabilityError(
                "STATEFRAME_PROMPT_CACHE",
                "an explicitly selected slot (slot 0 is never inferred)",
            )
        require_feature(
            capabilities, "STATEFRAME_PROMPT_CACHE", slots=True,
            slot_id=prompt_cache_slot,
        )
        if (type(body.get("id_slot")) is not int or body.get("id_slot") != prompt_cache_slot
                or body.get("cache_prompt") is not True):
            raise ProviderCapabilityError(
                "STATEFRAME_PROMPT_CACHE",
                "selected id_slot and cache_prompt=true request fields",
            )
    if type(route_slot) is int:
        require_feature(capabilities, "routed static slot", slots=True, slot_id=route_slot)
    if source.get("STATEFRAME_THINK_PREFILL", "off").strip().lower() == "self_plan":
        require_feature(
            capabilities, "THINK_PREFILL=self_plan", slots=True, continuation=True,
        )
    purpose = str(call_purpose or "").casefold()
    response_format = body.get("response_format")
    schema_format = response_format.get("json_schema") if isinstance(response_format, Mapping) else None
    schema_name = schema_format.get("name") if isinstance(schema_format, Mapping) else None
    public_review = (
        "public_review" in purpose or "public review" in purpose
        or schema_name == "public_probe"
    )
    if public_review:
        # The public review calls this body builder with cache_prompt=False;
        # the schema name keeps the refusal visible if route adaptation removed it.
        require_feature(capabilities, "public-review cache_prompt")
    history_summary_cache = (
        purpose in {"summary", "compaction", "history_compaction", "frame_compaction"}
        and source.get("LONG_HORIZON_PROTOCOL", "off").strip().lower()
        in {"on", "1", "true", "yes"}
        and body.get("cache_prompt") is True
    )
    if history_summary_cache:
        # A routed endpoint can strip a default slot before this boundary. The
        # summary purpose still identifies selected cache use after routing.
        require_feature(capabilities, "history-compaction cache/slot", slots=True)
        selected_slot = body.get("id_slot")
        if type(selected_slot) is not int:
            selected_slot = route_slot
        if type(selected_slot) is not int or selected_slot < 0:
            raise ProviderCapabilityError(
                "history-compaction cache/slot",
                "an explicitly selected /slots id when cache_prompt=true",
            )
        require_feature(
            capabilities, "history-compaction cache/slot",
            slots=True, slot_id=selected_slot,
        )
    continue_final_message = body.get("continue_final_message")
    continuation_requested = (
        continue_final_message not in (None, False, "")
        or body.get("add_generation_prompt") is True
    )
    if continuation_requested:
        require_feature(capabilities, "true prefill continuation", continuation=True)
    slot_id = body.get("id_slot")
    if "id_slot" in body:
        if type(slot_id) is not int or slot_id < 0:
            raise ProviderBoundaryContractError(
                "invalid_id_slot",
                "id_slot must be explicitly selected as a nonnegative integer",
            )
        require_feature(capabilities, "id_slot", slots=True, slot_id=slot_id)
    if body.get("cache_prompt") is True and not (public_review or history_summary_cache):
        require_feature(
            capabilities,
            "cache_prompt",
            slots=slot_id is not None,
            slot_id=slot_id if type(slot_id) is int else None,
        )


def enforce_selected_endpoint_body(
    body: Mapping[str, object],
    endpoint: str,
    key: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Enforce the dialect for legacy direct sends that have no RequestCapture."""
    capabilities = capabilities_for(endpoint, key, environ=environ)
    require_selected_request_features(capabilities, body, environ=environ)
    if capabilities.dialect == OPENAI_STRICT:
        try:
            from .kernel_launch_identity_cost import (
                assert_openai_chat_body, enforce_provider_body, reasoning_profile,
            )
        except ImportError:
            from kernel_launch_identity_cost import (
                assert_openai_chat_body, enforce_provider_body, reasoning_profile,
            )

        selected = enforce_provider_body(
            body, profile=reasoning_profile(environ=environ), dialect=OPENAI_STRICT,
            environ=environ,
        )
        assert_openai_chat_body(selected)
        return selected
    return dict(body)


def reasoning_controls_summary(
    dialect: str,
    *,
    reasoning_off: bool = False,
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    if dialect == LLAMA_CPP:
        return "llama.cpp reasoning controls unchanged"
    raw_effort = source.get("REACT_REASONING_EFFORT")
    effort = raw_effort.strip().lower() if isinstance(raw_effort, str) else None
    if not effort:
        effort = None
    if reasoning_off and effort != "none":
        return "global reasoning-off refused: no proven reasoning_effort=none support"
    try:
        require_strict_side_call_reasoning_off(effort)
    except ProviderBoundaryContractError:
        side_calls = (
            "proposal side-calls refused: require operator-confirmed REACT_REASONING_EFFORT=none support"
        )
    else:
        side_calls = "proposal side-calls reasoning-off operator-confirmed"
    if effort is not None:
        if effort not in OPENAI_REASONING_EFFORTS:
            return "reasoning controls refused: unsupported REACT_REASONING_EFFORT"
        return f"reasoning controls mapped to reasoning_effort={effort}; {side_calls}"
    return f"reasoning controls omitted; {side_calls}"
