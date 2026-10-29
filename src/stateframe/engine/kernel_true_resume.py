"""Explicit opt-in switch for v29 true resume.

The measured v28 environment is left untouched unless the operator sets
``STATEFRAME_TRUE_RESUME=on``.  Once enabled, the lower-level continuation
holders are configured to the strict public checkpoint profile.
"""
from __future__ import annotations

import os
from collections.abc import MutableMapping

TRUE_RESUME_ENV = "STATEFRAME_TRUE_RESUME"
TRUE_RESUME_ERROR = "STATEFRAME_TRUE_RESUME=on is required for worker restoration"
TRUE_RESUME_DEFAULT_ENV = {
    "KERNEL_CONTINUATION_EXECUTIVE": "1",
    "KERNEL_CONTINUATION_RESTART": "1",
    "KERNEL_INTERVENTIONS": "0",
    "REACT_TOOLCALL": "1",
    "REACT_STEP_CONTRACT": "1",
    "STATEFRAME_PROJECTOR": "rich",
    "STATEFRAME_VIEW": "operational",
    "KERNEL_DECISION_STATE": "0",
    "KERNEL_REJECTED_CANDIDATE": "0",
    "FACT_COMPILER": "0",
    "CROSSRUN_MEMORY": "0",
    "STATEFRAME_MEMORY_CAPSULE": "off",
    "FAIR_BRIEF": "0",
    "GROUNDED_BRIEF": "0",
    "KERNEL_MODE": "self_oracle:OFF,navigation:OFF,history_cap:OFF,retry_predispatch:OFF,render_selection:ENFORCE,sidekick:OFF,pilot:OFF,tool_applicability:OFF,public_review:OFF",
    "STATEFRAME_MODEL_ROUTER_REPLAY": "on",
    # These v29 organs have process-local sensors/ledgers without a safe
    # cross-process serializer in the v25 resume contract.  The opt-in strict
    # profile refuses them; OFF leaves the measured v28 environment untouched.
    "STATEFRAME_ENVIRONMENT_STATE": "off",
    "STATEFRAME_TOKENLAB_COORDINATED_LEDGER": "off",
    "STATEFRAME_TOKENLAB_COORDINATION_GUIDANCE": "off",
}


def enabled(environ: MutableMapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(TRUE_RESUME_ENV, "off")).strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{TRUE_RESUME_ENV} must be off or on")
    return value == "on"


def activate(environ: MutableMapping[str, str] | None = None) -> bool:
    """Apply strict lower-level defaults only for an explicit true-resume run."""
    source = os.environ if environ is None else environ
    if not enabled(source):
        return False
    # The explicit opt-in selects a compatible profile, so incompatible v29
    # organ flags cannot leak into a process-isolated restore.
    for key, value in TRUE_RESUME_DEFAULT_ENV.items():
        source[key] = value
    source["KERNEL_CONTINUATION_EXECUTIVE"] = "1"
    return True


def require_enabled(environ: MutableMapping[str, str] | None = None) -> bool:
    """Fail closed when a checkpoint restore is attempted without true resume."""
    if not enabled(environ):
        raise RuntimeError(TRUE_RESUME_ERROR)
    return True


__all__ = [
    "TRUE_RESUME_DEFAULT_ENV", "TRUE_RESUME_ENV", "TRUE_RESUME_ERROR", "activate", "enabled",
    "require_enabled",
]
