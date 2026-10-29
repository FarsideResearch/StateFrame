"""Generate, but never launch, the r45 model-hopping comparison arm."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace


COMPARISON_ENV = "STATEFRAME_MODEL_HOPPING_COMPARISON"
COMPARISON_SCHEMA = "stateframe-model-hopping-comparison/1"
VALID_COMPARISON_MODES = frozenset({"off", "on"})


def comparison_mode() -> str:
    value = os.environ.get(COMPARISON_ENV, "off").strip().lower() or "off"
    if value not in VALID_COMPARISON_MODES:
        raise ValueError(f"{COMPARISON_ENV} must be off or on")
    return value


def _list_of_strings(value: Sequence[object], *, name: str) -> list[str]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise ValueError(f"comparison {name} must be a sequence")
    result = [str(item) for item in value]
    if any(not item.strip() for item in result):
        raise ValueError(f"comparison {name} contains an empty item")
    return result


def _list_of_ints(value: Sequence[object], *, name: str) -> list[int]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise ValueError(f"comparison {name} must be a sequence")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"comparison {name} must contain integers")
        result.append(item)
    return result


def _control_config(config):
    # Keep the control on the same host-owned lease, stop, and adapter policy;
    # only endpoint selection changes.  This is a description, not a launch.
    moe = config.moe()
    # Preserve endpoint transport and credential policy as well as the router
    # policy.  The control may select only MoE; it must not become a cheaper
    # or differently authenticated implementation by accident.
    return replace(
        config,
        endpoints=(moe,),
        default_endpoint=moe.name,
        moe_only=True,
        control_name="moe_only",
    )


def generate_comparison_spec(config, *, release_id: str, tasks: Sequence[object] = (),
                             seeds: Sequence[object] = (),
                             depths: Sequence[object] = ()) -> dict[str, object]:
    """Return the two-endpoint-vs-MoE-only experiment description.

    The opt-in switch gates generation itself.  This function performs no
    network, subprocess, provider, or runner operation.
    """
    if comparison_mode() != "on":
        raise ValueError(
            f"opt-in {COMPARISON_ENV}=on is required for the comparison spec arm")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValueError("comparison release_id must be nonempty")
    endpoints = tuple(config.endpoints)
    if len(endpoints) != 2 or sum(item.moe for item in endpoints) != 1:
        raise ValueError("comparison arm requires exactly two endpoints, including one MoE")
    if config.moe_only or config.stop_policy != "host":
        raise ValueError("comparison arm requires adaptive routing with host stopping")
    moe = config.moe()
    alternate = next(item for item in endpoints if item.name != moe.name)
    if alternate.moe:
        raise ValueError("comparison arm requires one non-MoE endpoint")
    adapter_contract = endpoints[0].adapter.as_dict()
    if any(item.adapter.as_dict() != adapter_contract for item in endpoints[1:]):
        raise ValueError("comparison endpoints must share one adapter contract")

    control = _control_config(config)
    matched = {
        "paired": True,
        "tasks": _list_of_strings(tasks, name="tasks"),
        "seeds": _list_of_ints(seeds, name="seeds"),
        "depths": _list_of_strings(depths, name="depths"),
        "equal_budget": True,
        "equal_host_state": True,
        "equal_tools_and_sampler": True,
    }
    shared = {
        "router_switch": "on",
        "replay_switch": "on",
        "restart_policy": "replay_required",
        "stop_policy": config.stop_policy,
        "lease_actions": config.lease_actions,
        "adapter_contract": adapter_contract,
        "launch": False,
    }
    return {
        "schema": COMPARISON_SCHEMA,
        "release_id": release_id.strip(),
        "kind": "generated-spec",
        "launch": False,
        "shared": shared,
        "matched": matched,
        "arms": [
            {
                "id": "two_endpoint",
                "mode": "on",
                "router_config": config.as_dict(),
                "endpoint_names": [item.name for item in endpoints],
                "launch": False,
            },
            {
                "id": "moe_only",
                "mode": "on",
                "router_config": control.as_dict(),
                "endpoint_names": [moe.name],
                "launch": False,
            },
        ],
        "metrics": [
            "paired_correctness",
            "uncached_prefill_tokens",
            "cached_prompt_tokens",
            "decode_tokens",
            "wall_time",
            "peak_memory",
            "provider_cost",
            "crash_and_resume",
        ],
    }


__all__ = ["COMPARISON_ENV", "COMPARISON_SCHEMA", "comparison_mode", "generate_comparison_spec"]
