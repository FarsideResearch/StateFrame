"""E332b — the single effect-application boundary (WIN-PLAN Phase B/C).

Ordering contract (WIN-PLAN REVISION 3, B1):
  committed state -> action proposal -> evaluation + authorization ->
  dispatch or recorded refusal -> observed execution result ->
  required assurance updates -> committed next state -> select + render.

Evaluators RETURN decisions and effect proposals; they NEVER mutate live state
directly. "Single authorization point" means one logical POLICY boundary, not one
moment per iteration -- a pre-dispatch gate (should this action even run) and a
post-receipt assurance update (self_oracle taint, test relevance after a run) both
pass through `authorize()` below, each at its own point in the loop, because they
have different timing requirements. What must be singular is the code path that
decides evaluate/advise/apply, not the wall-clock moment it runs.

THE B2 BUG THIS CLOSES: `bolt.organ_out("self_oracle", bolt.note_created_test_edit(p))`
evaluates `note_created_test_edit(p)` -- which sets `self._self_oracle_tainted = True`
directly -- as a Python argument BEFORE `organ_out` is even entered. KERNEL_SHADOW only
ever inspected the return value inside `organ_out`; by the time it did, the mutation had
already landed. Suppression touched the advisory TEXT, never the EFFECT. Same shape for
`test_relevance` at kernel_bolt.py:588 (kernel_bolt.py:3179 call site): it flips
`_last_relevant_match` / `_last_green_irrelevant` / `_self_oracle_tainted` as a side
effect of being called, again before any shadow check runs.

FIX: split each mutating method into a pure decision function (`evaluate_*`, defined
alongside the original on KernelBolt) that computes a `Decision` -- text plus a tuple of
zero-argument effect callables, not-yet-invoked -- and route every call through
`authorize()`, which decides evaluate/advise/apply from the resolved Mode BEFORE the
effect callables are ever invoked. The original mutating methods (`note_created_test_edit`,
`test_relevance`) are left in place, byte-for-byte, because `probes/test_kernel_organs.py`
and `test_kernel_composition.py` call them directly and assert on their synchronous
mutation -- rewriting them to be pure would break ~25 existing green tests for no benefit
(those tests exercise the ORGAN's decision logic, not the production wiring). Only the
two `react_repair.py` production call sites (kernel_bolt.py :2889/:2971 self_oracle,
:3179 navigation/test_relevance) are re-pointed at the pure evaluators through this
boundary -- that is where the half-ablation actually lived.

CLASSIFICATION (B2's "mandatory-assurance vs experimental" call): the work order's own
acceptance table pins it -- "Set self_oracle to SHADOW -> the taint flag must NOT be
set." Both `self_oracle` and `test_relevance` (aka `navigation`, its organ_out name) are
therefore EXPERIMENTAL_CONTROL: mode gates BOTH their advisory text and their state
effect. (A component classified MANDATORY_ASSURANCE would still evaluate and apply its
effect even under SHADOW/OFF -- reserved for a future safety invariant that must hold
across every experimental arm; none of the two components in scope for this wave
qualify per the acceptance criteria, so no component currently uses that path, but the
axis exists and is exercised by test.)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Callable
from enum import Enum

# --------------------------------------------------------------------------- modes

class Mode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    ADVISE = "ADVISE"
    ENFORCE = "ENFORCE"
    BOTH = "BOTH"


# axis table: (evaluate, deliver_advice, apply_effects) -- Phase C's three axes.
AXES: dict[Mode, tuple[bool, bool, bool]] = {
    Mode.OFF:     (False, False, False),
    Mode.SHADOW:  (True,  False, False),
    Mode.ADVISE:  (True,  True,  False),
    Mode.ENFORCE: (True,  False, True),
    Mode.BOTH:    (True,  True,  True),
}

# Baseline (pre-Phase-C) behaviour for self_oracle/test_relevance was "always evaluate,
# always mutate, deliver text unless already deduped" -- i.e. Mode.BOTH. Work-order §4:
# "every Phase D toggle defaults to current baseline behaviour" -- this is that default,
# extended to Phase C's switch (Phases A-C "change no default behaviour at all").
DEFAULT_MODE = Mode.BOTH


class ConsequenceType(str, Enum):
    """B3: legitimate consequence types a capability's effects may have."""
    INFORMATION = "information"    # model-facing text only
    MEMORY = "memory"              # persisted state the model does not see directly
    CONTROL = "control"            # gates a later decision (done-gate, retry-gate)
    OBSERVATION = "observation"    # audit/replay fact, no forward effect


class ComponentKind(str, Enum):
    MANDATORY_ASSURANCE = "mandatory_assurance"     # constant across experimental arms
    EXPERIMENTAL_CONTROL = "experimental_control"    # fully through the boundary


@dataclasses.dataclass(frozen=True)
class Decision:
    """What an evaluator returns. Never mutates on its own -- `effects` is a tuple of
    zero-arg callables, not yet invoked. `authorize()` invokes them only if the
    resolved mode's apply-effects axis is True."""
    text: str | None
    effects: tuple[Callable[[], None], ...] = ()
    effect_summary: tuple[str, ...] = ()   # human-readable, goes in the audit record

    @staticmethod
    def empty() -> "Decision":
        return Decision(text=None, effects=(), effect_summary=())


# --------------------------------------------------------------------------- manifest (B3)

@dataclasses.dataclass(frozen=True)
class CapabilityEntry:
    """B3 capability manifest row. One per live capability routed through the boundary."""
    name: str
    trigger: str
    typed_output: str
    intended_consumer: str
    required_phase: str
    allowed_effects: tuple[ConsequenceType, ...]
    kind: ComponentKind
    documentation_version: str
    integration_test: str


_MANIFEST: dict[str, CapabilityEntry] = {}


def register_capability(entry: CapabilityEntry) -> None:
    _MANIFEST[entry.name] = entry


def capability_manifest() -> dict[str, CapabilityEntry]:
    return dict(_MANIFEST)


# --------------------------------------------------------------------------- config

def _parse_mode_env() -> tuple[dict[str, Mode], Mode]:
    """KERNEL_MODE="self_oracle:SHADOW,navigation:ADVISE" -- per-component overrides.
    Back-compat: legacy KERNEL_SHADOW="e325,e317" (or 'all') maps each listed name (or
    every component, for 'all') to SHADOW, matching the old organ_out semantics exactly
    for anyone still setting the old env var."""
    per_component: dict[str, Mode] = {}
    default = DEFAULT_MODE
    raw = os.environ.get("KERNEL_MODE", "")
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, _, mode_s = pair.partition(":")
        name = name.strip()
        mode_s = mode_s.strip().upper()
        try:
            per_component[name] = Mode(mode_s)
        except ValueError:
            continue
    legacy = os.environ.get("KERNEL_SHADOW", "")
    for name in (s.strip() for s in legacy.split(",")):
        if not name:
            continue
        if name == "all":
            default = Mode.SHADOW
            continue
        per_component.setdefault(name, Mode.SHADOW)
    return per_component, default


def resolve_mode(owner: object, name: str) -> Mode:
    """Cached on the owner (one resolution per process per KERNEL_MODE/KERNEL_SHADOW
    value) so authorize() does not re-parse env vars every call."""
    from kernel_intervention_policy import interventions_enabled
    if not interventions_enabled() and name in {"self_oracle", "navigation", "retry_predispatch"}:
        return Mode.OFF
    cache = getattr(owner, "_boundary_mode_cache", None)
    if cache is None:
        cache = owner._boundary_mode_cache = {}
    if "__per_component__" not in cache:
        per_component, default = _parse_mode_env()
        cache["__per_component__"] = per_component
        cache["__default__"] = default
    per_component = cache["__per_component__"]
    default = cache["__default__"]
    return per_component.get(name, default)


def _config_hash(owner: object) -> str:
    from kernel_intervention_policy import interventions_enabled
    cache = getattr(owner, "_boundary_mode_cache", None) or {}
    per_component = cache.get("__per_component__", {})
    default = cache.get("__default__", DEFAULT_MODE)
    blob = json.dumps(
        {"default": default.value, "interventions_enabled": interventions_enabled(),
         "components": {k: v.value for k, v in sorted(per_component.items())}},
        sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- audit (Phase C)

@dataclasses.dataclass
class AuditRecord:
    """Per-decision audit record, Phase C spec: run/task/seed, config hash, component
    version, base-state hash, action fingerprint, proposed effect, authorized effect,
    confirmed applied effect, before/after artifact hashes, receipt ids, rendered-prompt
    hash."""
    run_id: str
    task_id: str
    seed: str | None
    config_hash: str
    component: str
    component_version: str
    step: int | None
    mode: str
    base_state_hash: str
    action_fingerprint: str
    proposed_effect: tuple[str, ...]
    authorized_effect: tuple[str, ...]
    confirmed_applied_effect: tuple[str, ...]
    before_hash: str
    after_hash: str
    receipt_id: str
    rendered_prompt_hash: str | None
    ts: float
    effect_label_hash: str = ""
    input_identity: dict = dataclasses.field(default_factory=dict)
    scoped_before: dict = dataclasses.field(default_factory=dict)
    scoped_after: dict = dataclasses.field(default_factory=dict)
    artifact_hashes: dict = dataclasses.field(default_factory=dict)
    request_id: str | None = None
    outcome: str = "completed"
    error_type: str | None = None
    completed_effect_count: int = 0
    returned_advice: str | None = None
    observed_state_changes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def _state_hash(owner: object, fields: tuple[str, ...]) -> str:
    """A cheap, deterministic proxy for 'base-state hash' / 'artifact hash': hashes the
    subset of owner attributes a component's effects actually touch. Not a full-object
    hash (the bolt carries non-hashable state like file handles/registries) -- scoped to
    what this boundary call can affect, which is what before/after comparison needs."""
    vals = {f: repr(getattr(owner, f, "__unset__")) for f in fields}
    blob = json.dumps(vals, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _record(owner: object, rec: AuditRecord) -> None:
    log = getattr(owner, "_effect_audit_log", None)
    if log is None:
        log = owner._effect_audit_log = []
    log.append(rec)
    journal = getattr(owner, "audit_journal", None)
    if journal is not None:
        journal.emit("effect_decision", decision=rec.as_dict())


STATE_FIELDS: dict[str, tuple[str, ...]] = {
    # Which owner attributes each component's effects can touch -- drives the
    # before/after state-hash proxy above. Extend when a new component is registered.
    "self_oracle": ("_self_oracle_tainted",),
    "navigation": ("_last_relevant_match", "_last_green_irrelevant", "_self_oracle_tainted",
                   "_nav_seen", "_challenged"),
}


def _scoped_state(owner: object, fields: tuple[str, ...]) -> dict:
    # Known state is flags, maps and selector sets; deterministic order matters.
    def normalize(value: object) -> object:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        return value
    return {field: normalize(getattr(owner, field, None)) for field in fields}


def authorize(*, owner: object, name: str, version: str, run_id: str,
              decision_fn: Callable[[], Decision | None],
              seed: str | None = None, rendered_prompt_hash: str | None = None,
              input_identity: object = None) -> str | None:
    """Evaluate/apply/advice independently, recording failures and actual state."""
    from kernel_measurement import digest
    mode = resolve_mode(owner, name)
    evaluate, deliver, apply_fx = AXES[mode]
    fields = STATE_FIELDS.get(name, ())
    before = _scoped_state(owner, fields)
    receipt = (getattr(owner, "_receipts", None) or [{}])[-1]
    identity = dict(input_identity) if isinstance(input_identity, dict) else dict(receipt)
    journal = getattr(owner, "audit_journal", None)
    runtime = getattr(owner, "measurement", None)
    snapshot = getattr(runtime, "snapshot_id", "MISSING")
    request_id = getattr(owner, "current_request_id", None)
    if journal is not None:
        journal.emit("effect_decision_started", component=name,
                     step=getattr(owner, "current_step", None),
                     action_fingerprint=digest(identity), input_identity=identity,
                     mode=mode.value, snapshot_id=snapshot, request_id=request_id)
    decision = Decision.empty()
    completed = 0
    error = None
    try:
        if evaluate:
            decision = decision_fn() or Decision.empty()
        if apply_fx:
            for effect in decision.effects:
                effect()
                completed += 1
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        after = _scoped_state(owner, fields)
        confirmed = []
        if apply_fx:
            for label in decision.effect_summary:
                key, sep, value = label.partition("=")
                field = "_" + key.lstrip("_")
                if sep and field in after and str(after[field]) == value:
                    confirmed.append(label)
        returned = decision.text if deliver and error is None else None
        record = AuditRecord(
            run_id=run_id, task_id=getattr(owner, "task_id", ""), seed=seed,
            config_hash=journal.config_hash if journal is not None else _config_hash(owner),
            component=name, component_version=version, step=getattr(owner, "current_step", None),
            mode=mode.value, base_state_hash=snapshot, action_fingerprint=digest(identity),
            proposed_effect=decision.effect_summary,
            authorized_effect=decision.effect_summary if apply_fx else (),
            confirmed_applied_effect=tuple(confirmed),
            before_hash=digest(before), after_hash=digest(after),
            receipt_id=str(receipt.get("receipt_id") or "MISSING"),
            rendered_prompt_hash=rendered_prompt_hash or getattr(owner, "current_prompt_hash", None),
            ts=time.time(), effect_label_hash=digest(decision.effect_summary),
            input_identity=identity, scoped_before=before, scoped_after=after,
            artifact_hashes=dict(getattr(runtime, "artifact_hashes", {})),
            request_id=request_id, outcome="error" if error else "completed" if evaluate else "off",
            error_type=error, completed_effect_count=completed, returned_advice=returned,
            observed_state_changes=tuple(key for key in fields if before[key] != after[key]))
        _record(owner, record)
        rec_fn = getattr(owner, "_rec", None)
        if callable(rec_fn):
            rec_fn("shadow_ns" if mode is Mode.SHADOW else "boundary", f"authorize:{name}",
                   f"mode={mode.value} applied={completed} outcome={record.outcome}")
    return returned


# --------------------------------------------------------------------------- B3 manifest rows

register_capability(CapabilityEntry(
    name="self_oracle",
    trigger="a self-created test file (created_this_run) is edited after it went RED",
    typed_output="Decision(text: str|None, effects=(set _self_oracle_tainted,))",
    intended_consumer="completion_verdict / done-gate (E322/E325 family)",
    required_phase="B",
    allowed_effects=(ConsequenceType.CONTROL, ConsequenceType.INFORMATION),
    kind=ComponentKind.EXPERIMENTAL_CONTROL,
    documentation_version="B2-2026-09-20",
    integration_test="probes/test_effect_boundary.py::"
                      "test_self_oracle_shadow_does_not_set_taint_flag",
))

register_capability(CapabilityEntry(
    name="navigation",
    trigger="a GREEN test run whose selector matched none of the relevant regression "
            "nodes (test_relevance / E317)",
    typed_output="Decision(text: str|None, "
                 "effects=(set _last_relevant_match/_last_green_irrelevant/"
                 "_self_oracle_tainted,))",
    intended_consumer="completion_verdict / done-gate; navigation advisory to the model",
    required_phase="B",
    allowed_effects=(ConsequenceType.CONTROL, ConsequenceType.INFORMATION),
    kind=ComponentKind.EXPERIMENTAL_CONTROL,
    documentation_version="B2-2026-09-20",
    integration_test="probes/test_effect_boundary.py::"
                      "test_navigation_shadow_does_not_flip_relevance_state",
))
