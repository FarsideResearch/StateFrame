"""Coordinate task-state components under explicit feature configuration.

Components preserve bounded request state and expose their execution evidence
without taking repair selection away from the model.
"""
from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Mapping
from enum import Enum

from kernel_effect_boundary import AXES, Mode
from kernel_context import (
    Exchange, HistoryCapResult, HistoryOverflow, DEFAULT_HISTORY_CAP_TOKENS,
    group_into_exchanges, cap_history, assert_no_orphaned_tool_result,
)
from kernel_context import (
    FRAME_TARGET_TOKENS, FRAME_CEILING_TOKENS, SELECTION_POLICY_VERSION,
    SelectionPriority, SelectableItem, ViewPlan, RenderOverflow, select_view,
)
from kernel_retry_gate import Decision as RetryDecisionKind
from kernel_retry_gate import RetryDecision, RetryGate



# =====================================================================================
# Shared toggle resolution -- Phase D components default OFF, never Mode.BOTH.
#
# This is deliberately a SEPARATE resolver from kernel_effect_boundary.resolve_mode():
# that function's un-overridden default is Mode.BOTH, correct for B2's two components
# because BOTH *was* their pre-Wave-2 baseline (always evaluate, always mutate). None
# of D1/D2/D3/D5 existed live before this wave -- their baseline is "does not run at
# all" -- so the un-overridden default here is Mode.OFF. Same env var, same per-
# component override syntax (`KERNEL_MODE="retry_predispatch:ENFORCE,history_cap:SHADOW"`),
# so one mental model covers both phases.
# =====================================================================================

_D_MODE_CACHE: dict[str, Mode] | None = None


def _parse_d_mode_env() -> dict[str, Mode]:
    per_component: dict[str, Mode] = {}
    raw = os.environ.get("KERNEL_MODE", "")
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, _, mode_s = pair.partition(":")
        try:
            per_component[name.strip()] = Mode(mode_s.strip().upper())
        except ValueError:
            continue
    return per_component


def resolve_d_mode(name: str) -> Mode:
    """Resolve one Phase-D component's mode. Default OFF (current baseline: the
    component does not run). Cached per-process; call :func:`_reset_d_mode_cache_for_tests`
    to force a re-read of the environment (tests that flip KERNEL_MODE mid-run need this;
    production code reads the environment once, matching kernel_effect_boundary's own
    per-process caching)."""
    global _D_MODE_CACHE
    if _D_MODE_CACHE is None:
        _D_MODE_CACHE = _parse_d_mode_env()
    return _D_MODE_CACHE.get(name, Mode.OFF)


def _reset_d_mode_cache_for_tests() -> None:
    global _D_MODE_CACHE
    _D_MODE_CACHE = None


# =====================================================================================
# D1 -- protocol documentation, generated from the ACTUAL emitted payload classes.
#
# "Verify each class against its actual emitted payload first... document
# [kernel-retry] as post-hoc detection, NOT prevention, because that is what the code
# does today." The registry below is hand-verified against kernel_bolt.py's actual
# `organ_out(name, ...)` call sites (grep, quoted in the disconnect test) plus the two
# `authorize()` components from Phase B/C -- not the version we intend to build.
# =====================================================================================

class ExecutionStatus(str, Enum):
    PREVENTION = "prevention"        # blocks the action before it takes effect
    POST_HOC_DETECTION = "post_hoc_detection"  # observes/flags AFTER the effect landed
    INFORMATIONAL = "informational"  # never gates anything, advisory text only


@dataclasses.dataclass(frozen=True)
class ProtocolClass:
    """One documented message class: what emits it, what it means, who may act on it,
    whether it prevents or only detects, and what response is permitted."""
    name: str
    emitter: str
    meaning: str
    authority: str
    execution_status: ExecutionStatus
    permitted_response: str
    demonstration: str = ""


# The registry. Every `organ_out(name, ...)` callsite in kernel_bolt.py, plus the two
# `authorize(name=...)` components, must have an entry here -- enforced by
# probes/test_wave3.py::test_d1_protocol_documentation_covers_every_emitted_payload.
PROTOCOL_CLASSES: tuple[ProtocolClass, ...] = (
    ProtocolClass(
        name="retry_gate",
        emitter="kernel_bolt.KernelBolt.receipt() -> RetryGate.observe()",
        meaning="[kernel-retry] warns or refuses-in-words a repeated edit whose "
                "fingerprint matches a PRIOR FAILED attempt with nothing material "
                "changed.",
        authority="Advisory text only from this call site today. The edit this "
                  "message is ABOUT has ALREADY BEEN APPLIED by the time the message "
                  "is emitted -- `receipt()` runs after `str_replace_edit()` in "
                  "probes/react_repair.py. A denial here does not un-apply the edit.",
        execution_status=ExecutionStatus.POST_HOC_DETECTION,
        permitted_response="The model may revise its NEXT edit in light of the "
                           "warning/denial text; it cannot retract the edit that "
                           "triggered it.",
        demonstration="test_d1_post_edit_warning_edit_remains_applied",
    ),
    ProtocolClass(
        name="retry_predispatch",
        emitter="kernel_wave3.PreDispatchGate.evaluate() -> RetryGate.peek_predispatch()",
        meaning="Forecasts the same WARN/DENY a post-hoc retry_gate receipt would "
                "produce, computed BEFORE the editing tool runs.",
        authority="SHADOW (default/baseline): diagnostic only, decision is computed "
                  "and recorded but the dispatcher is never consulted -- identical to "
                  "not having this component at all. ENFORCE: a DENY blocks dispatch; "
                  "the editing tool is never invoked for that turn.",
        execution_status=ExecutionStatus.PREVENTION,
        permitted_response="Under ENFORCE, the model receives a refusal message in "
                           "place of a tool result and must propose a different "
                           "action. Never silently retried by the kernel.",
        demonstration="test_d3_enforce_blocks_dispatch_disconnect",
    ),
    ProtocolClass(
        name="frontier",
        emitter="kernel_bolt.KernelBolt.organ_out('frontier', ...)",
        meaning="Names the current hypothesis frontier (kernel_hypotheses.frontier()).",
        authority="Advisory text only. Never gates the done-gate or dispatch.",
        execution_status=ExecutionStatus.INFORMATIONAL,
        permitted_response="Model may use it to focus investigation; no obligation.",
    ),
    ProtocolClass(
        name="hypotheses",
        emitter="kernel_bolt.KernelBolt.organ_out('hypotheses', ...)",
        meaning="Renders one hypothesis capsule from the hypothesis-state machine.",
        authority="Advisory text only.",
        execution_status=ExecutionStatus.INFORMATIONAL,
        permitted_response="Model may reference it; no obligation.",
    ),
    ProtocolClass(
        name="invalidation",
        emitter="kernel_bolt.KernelBolt.organ_out('invalidation', ...)",
        meaning="A dependency-invalidation notice: a fact this belief depended on "
                "changed after the belief was formed (kernel_invalidation.py).",
        authority="Advisory text only today (work order: 'wired advisory-only, "
                  "kernel_bolt.py:1107-1120'). Does not retract or block anything.",
        execution_status=ExecutionStatus.POST_HOC_DETECTION,
        permitted_response="Model should treat the referenced belief as stale.",
    ),
    ProtocolClass(
        name="composed",
        emitter="kernel_bolt.KernelBolt.organ_out('composed', ...)",
        meaning="A composed-cognition card (composed_cognition.py); names, in text, "
                "the discriminating action the selector computed.",
        authority="Advisory text only -- the underlying selection is never enforced.",
        execution_status=ExecutionStatus.INFORMATIONAL,
        permitted_response="Model may or may not take the named action.",
    ),
    ProtocolClass(
        name="framebreak",
        emitter="kernel_bolt.KernelBolt.organ_out('framebreak', ...)",
        meaning="A frame-break challenge card.",
        authority="Advisory text only.",
        execution_status=ExecutionStatus.INFORMATIONAL,
        permitted_response="Model may respond to the challenge or not.",
    ),
    ProtocolClass(
        name="closelook",
        emitter="kernel_bolt.KernelBolt.organ_out('closelook', ...)",
        meaning="kernel_close_look.check_edit() output rendered as text.",
        authority="Advisory text only -- work order states this organ has ZERO other "
                  "call sites gating anything; this is display only.",
        execution_status=ExecutionStatus.INFORMATIONAL,
        permitted_response="Model may use it to reconsider the edit; nothing blocks "
                           "dispatch based on it today.",
    ),
    ProtocolClass(
        name="self_oracle",
        emitter="kernel_bolt.KernelBolt.authorize(name='self_oracle', ...) "
                "(kernel_effect_boundary.authorize)",
        meaning="Detects a test file edited immediately after (or instead of) the "
                "product code the failing test targets.",
        authority="Phase C mode-gated: SHADOW computes+records without mutating "
                  "`_self_oracle_tainted` or delivering text; ENFORCE mutates the "
                  "taint flag (which the done-gate reads) without delivering "
                  "advisory text; BOTH (default) does both.",
        execution_status=ExecutionStatus.POST_HOC_DETECTION,
        permitted_response="The taint flag, when set, is read by the done-gate; it "
                           "is not a message the model responds to directly.",
    ),
    ProtocolClass(
        name="navigation",
        emitter="kernel_bolt.KernelBolt.authorize(name='navigation', ...) "
                "(kernel_effect_boundary.authorize, aka test_relevance)",
        meaning="Flags whether the current green test is relevant to the bug target.",
        authority="Phase C mode-gated, same axis semantics as self_oracle.",
        execution_status=ExecutionStatus.POST_HOC_DETECTION,
        permitted_response="Read by downstream organs (`_last_relevant_match`, "
                           "`_last_green_irrelevant`), not a direct model response.",
    ),
)


def protocol_documentation_markdown() -> str:
    """Render PROTOCOL_CLASSES as a markdown table -- the D1 deliverable."""
    lines = [
        "| class | emitter | execution status | authority | permitted response |",
        "|---|---|---|---|---|",
    ]
    for pc in PROTOCOL_CLASSES:
        lines.append(
            f"| `{pc.name}` | {pc.emitter} | **{pc.execution_status.value}** | "
            f"{pc.authority} | {pc.permitted_response} |"
        )
    return "\n".join(lines)


def registered_protocol_names() -> frozenset[str]:
    return frozenset(pc.name for pc in PROTOCOL_CLASSES)


# =====================================================================================
# D2 -- token-based history cap, trimming at COMPLETE exchange boundaries.
#
# Task/constraints/tool-instruction messages (index 0, the system/task preamble) sit
# OUTSIDE the adjustable budget -- never trimmed regardless of cap. The cap applies
# only to the exchange history that follows. An "exchange" is one assistant turn
# (its `tool_calls`, if any) plus every `tool` message answering those calls -- never
# split; a tool result is never retained without its call, and a tool call is never
# retained without its result once one exists.
# =====================================================================================



# =====================================================================================
# D3 -- pre-dispatch retry gate: evaluate AFTER action parsing, BEFORE the editing
# tool executes. Conservative identity ONLY (kernel_retry_gate.AttemptFingerprint --
# action_kind/target/edit_digest/hypothesis/read_set_revision). No semantic
# fingerprints, no failure-message similarity -- that is exactly what
# AttemptFingerprint already refuses to use (see kernel_retry_gate.py's own docstring).
#
# Fail to SHADOW on evaluator error. Disarm for the rest of the run after two
# consecutive denials with no new evidence or action diversity. Denials count against
# budget (an external caller charges its own attempt budget on `.blocks`).
# =====================================================================================

@dataclasses.dataclass
class PreDispatchOutcome:
    decision: RetryDecision
    blocks: bool          # True only under ENFORCE/BOTH with a DENY verdict
    disarmed: bool         # True if THIS call caused the gate to disarm
    evaluator_error: bool = False
    advice: str | None = None


class PreDispatchGate:
    """Wraps a RetryGate's read-only forecast (`peek_predispatch`) with the D3 safety
    invariants: mode gating, fail-to-SHADOW on evaluator error, and disarm-after-two-
    consecutive-denials-with-no-new-evidence-or-action-diversity."""

    def __init__(self, gate: RetryGate, *, component_name: str = "retry_predispatch"):
        self.gate = gate
        self.component_name = component_name
        self._consecutive_denials = 0
        self._last_denied_fingerprint: str | None = None
        self._disarmed = False
        self.denials_charged_to_budget = 0

    @property
    def disarmed(self) -> bool:
        return self._disarmed

    def observe(self, receipt: Mapping[str, object]) -> None:
        """Commit the same attempt that was evaluated, isolated from legacy advice."""
        if not AXES[getattr(self, "configured_mode", resolve_d_mode(self.component_name))][0]:
            return
        record = dict(receipt)
        attempt = record.pop("attempt", None)
        if isinstance(attempt, Mapping):
            record.update(attempt)
        elif record.get("kind") == "edit":
            # Legacy/unidentified actions cannot seed canonical failure history.
            return
        self.gate.observe(record)

    def evaluate(self, record: Mapping[str, object], *, mode: Mode | None = None) -> PreDispatchOutcome:
        resolved = mode if mode is not None else getattr(
            self, "configured_mode", resolve_d_mode(self.component_name))
        outcome = self._evaluate(record, mode=resolved)
        if AXES[resolved][1] and outcome.decision.message:
            outcome.advice = outcome.decision.message
        journal = getattr(self, "audit_journal", None)
        if journal is not None:
            journal.emit("predispatch_decision", step=record.get("turn"),
                         input_identity=dict(record), mode=resolved.value,
                         outcome=dataclasses.asdict(outcome))
        return outcome

    def _evaluate(self, record: Mapping[str, object], *, mode: Mode | None = None) -> PreDispatchOutcome:
        resolved = mode if mode is not None else resolve_d_mode(self.component_name)
        evaluate_axis, _deliver, apply_fx = AXES[resolved]
        if self._disarmed:
            # Disarmed for the rest of the run: never evaluate again, never block.
            return PreDispatchOutcome(
                decision=RetryDecision(decision=RetryDecisionKind.ALLOW, turn=0,
                                       reason="predispatch_gate_disarmed"),
                blocks=False, disarmed=False)
        if not evaluate_axis:
            return PreDispatchOutcome(
                decision=RetryDecision(decision=RetryDecisionKind.ALLOW, turn=0,
                                       reason="predispatch_gate_off"),
                blocks=False, disarmed=False)
        try:
            decision = self.gate.peek_predispatch(record)
        except Exception:  # noqa: BLE001 -- fail to SHADOW, never crash the loop
            return PreDispatchOutcome(
                decision=RetryDecision(decision=RetryDecisionKind.ALLOW, turn=0,
                                       reason="predispatch_evaluator_error_failed_shadow"),
                blocks=False, disarmed=False, evaluator_error=True)

        newly_disarmed = False
        if decision.denied:
            fp = decision.fingerprint
            if fp and fp == self._last_denied_fingerprint:
                self._consecutive_denials += 1
            else:
                self._consecutive_denials = 1
                self._last_denied_fingerprint = fp
            self.denials_charged_to_budget += 1
            if self._consecutive_denials >= 2:
                self._disarmed = True
                newly_disarmed = True
        else:
            # New evidence or action diversity (non-denial outcome) resets the streak.
            self._consecutive_denials = 0
            self._last_denied_fingerprint = None

        blocks = bool(decision.denied and apply_fx)
        return PreDispatchOutcome(decision=decision, blocks=blocks, disarmed=newly_disarmed)


# =====================================================================================
# D5 -- four-priority selection contract + RENDER_OVERFLOW as an explicit result.
#
# Frame budget: 4k is
# the optimisation TARGET, 8k is the actual CEILING. Priorities 1-2 should fit inside
# 4k where possible; 4k-8k is margin absorbing a large target symbol.
# =====================================================================================



# =====================================================================================
# Phase E -- measurement harness building blocks: exposure manifest per step, real
# prefill via cache_n/prompt_ms/prompt_n (NEVER submitted token counts).
# =====================================================================================

@dataclasses.dataclass(frozen=True)
class ExposureManifest:
    """One record per step. `prompt_n`/`cache_n`/`prompt_ms` are meant to be read from
    the inference server's own accounting (llama.cpp `/completion` response carries
    `timings.prompt_n`, `timings.prompt_ms`, and cache-hit token count) -- NEVER
    computed by summing the tokens the harness believes it submitted, because that
    number silently diverges from what was actually prefetched/cached whenever the
    server's own tokenizer, chat template, or KV-cache reuse differs from the harness's
    approximation (the exact `approx_tokens` 4-chars/token estimate used elsewhere in
    this module is a fine BUDGETING heuristic and a WRONG measurement of real prefill)."""
    step: int
    snapshot_id: str
    selector_version: str
    renderer_version: str
    selected_item_ids: tuple[str, ...]
    selection_reasons: tuple[str, ...]
    truncation_markers: tuple[str, ...]
    submitted_token_estimate: int         # harness-side approx, budgeting only
    measured_prompt_n: int | None         # from the inference server, real tokens
    measured_prompt_ms: float | None      # from the inference server, real wall time
    measured_cache_n: int | None          # from the inference server, real cache hits
    prompt_hash: str
    ts: float = dataclasses.field(default_factory=time.time)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def prefill_is_measured(self) -> bool:
        """True only when the server actually reported prefill accounting. A caller
        must never fall back to `submitted_token_estimate` and treat it as measured --
        that is exactly the substitution E's design note forbids."""
        return self.measured_prompt_n is not None and self.measured_prompt_ms is not None


def build_exposure_manifest(
    *, step: int, snapshot_id: str, view: ViewPlan | RenderOverflow,
    renderer_version: str, prompt_hash: str,
    server_timings: Mapping[str, object] | None = None,
) -> ExposureManifest:
    """Assemble one step's exposure manifest. `server_timings`, when provided, is the
    inference server's own `timings` block (llama.cpp shape: `prompt_n`, `prompt_ms`,
    and a cache-hit count under whichever key that server build exposes) -- passed
    through verbatim, never recomputed from the prompt text."""
    if isinstance(view, RenderOverflow):
        selected_ids: tuple[str, ...] = ()
        reasons = (f"RENDER_OVERFLOW:{view.required_group}",)
        markers = (f"RENDER_OVERFLOW:{view.required_group}:"
                  f"{view.required_tokens}>{view.available_tokens}",)
        submitted_estimate = 0
    else:
        selected_ids = tuple(i.item_id for i in view.selected)
        reasons = tuple(f"{i.item_id}:{i.priority.name}" for i in view.selected)
        markers = tuple(
            f"omitted:{i.item_id}:{i.group}:{i.omission_reason or 'not_recorded'}"
            + (f":score={i.omission_score:g}" if i.omission_score is not None else "")
            for i in view.omitted
        )
        submitted_estimate = view.used_tokens

    timings = server_timings or {}
    return ExposureManifest(
        step=step, snapshot_id=snapshot_id,
        selector_version=SELECTION_POLICY_VERSION, renderer_version=renderer_version,
        selected_item_ids=selected_ids, selection_reasons=reasons,
        truncation_markers=markers, submitted_token_estimate=submitted_estimate,
        measured_prompt_n=_maybe_int(timings.get("prompt_n")),
        measured_prompt_ms=_maybe_float(timings.get("prompt_ms")),
        measured_cache_n=_maybe_int(timings.get("cache_n", timings.get("prompt_cache_hit_tokens"))),
        prompt_hash=prompt_hash,
    )


def _maybe_int(v: object) -> int | None:
    try:
        return int(v) if v is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _maybe_float(v: object) -> float | None:
    try:
        return float(v) if v is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class SystemResult(str, Enum):
    """Phase E intent-to-treat outcome. A gate causing deadlock is a TREATMENT
    FAILURE, never an excludable infra incident."""
    VERIFIED_COMPLETION = "verified_completion"
    UNVERIFIED_COMPLETION = "unverified_completion"
    CEILING_REACHED = "ceiling_reached"           # ran out of proposal budget
    TREATMENT_DEADLOCK = "treatment_deadlock"      # an experimental gate blocked all forward motion
    INFRA_INCIDENT = "infra_incident"              # genuinely excludable: e.g. host OOM-killed the run


__all__ = [
    "AXES", "Mode", "resolve_d_mode",
    "ExecutionStatus", "ProtocolClass", "PROTOCOL_CLASSES",
    "protocol_documentation_markdown", "registered_protocol_names",
    "Exchange", "group_into_exchanges", "HistoryCapResult", "HistoryOverflow",
    "DEFAULT_HISTORY_CAP_TOKENS", "cap_history", "assert_no_orphaned_tool_result",
    "PreDispatchOutcome", "PreDispatchGate",
    "FRAME_TARGET_TOKENS", "FRAME_CEILING_TOKENS", "SelectionPriority",
    "SelectableItem", "ViewPlan", "RenderOverflow", "SELECTION_POLICY_VERSION",
    "select_view",
    "ExposureManifest", "build_exposure_manifest", "SystemResult",
]
