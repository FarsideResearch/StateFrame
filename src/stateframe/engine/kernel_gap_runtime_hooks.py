"""Minimal final-boundary hooks for the five lane C mechanisms."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from kernel_gap_failure_cards import build_failure_card, render_failure_card
from kernel_gap_hypothesis_recovery import (
    continuity_projection, recover_hypothesis, render_recovery,
)
from kernel_gap_navigation import build_index, locate, render_navigation
from kernel_gap_surface_coverage import reduce_coverage, render_coverage
from kernel_gap_switches import any_enabled, emit, guidance_enabled, truth_enabled
from kernel_gap_witnesses import render_witness_suite, run_witnesses


PRODUCER = "kernel_gap_runtime_hooks"
MAX_BLOCK_CHARS = 3000


def _approx_tokens(value: str) -> int:
    return max(1, len(value.encode("utf-8", errors="replace")) // 4)


def apply_gap_request_hooks(runtime: object, frame: Mapping[str, object],
                            prepared: Sequence[Mapping[str, object]], tools: object,
                            *, request_id: str, provider_request: bool = True,
                            capacity_tokens: int | None = None,
                            model_name: str | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Project enabled gap facts into the existing request boundary.

    With every switch off, the original sequence is returned unchanged and no
    module is asked to inspect runtime state or emit an event.
    """
    behavior_on = os.environ.get("STATEFRAME_BEHAVIOR_WITNESS", "off").strip().lower() == "on"
    if not provider_request or (not any_enabled() and not behavior_on):
        return prepared if isinstance(prepared, list) else list(prepared), {}
    journal = getattr(runtime, "journal", None)
    revision = str(getattr(runtime, "product_revision", ""))
    blocks: list[str] = []
    audit: dict[str, object] = {"request_id": str(request_id), "truth": [], "guidance": False}

    cases = getattr(runtime, "gap_witness_cases", ())
    invoke = getattr(runtime, "gap_witness_invoke", None)
    if truth_enabled("STATEFRAME_GAPS_WITNESSES") and cases and callable(invoke):
        suite = run_witnesses(cases, invoke, revision=revision, journal=journal)
        blocks.append(render_witness_suite(suite))
        audit["truth"].append("witnesses")
    suite = getattr(runtime, "gap_witness_suite", None)
    if truth_enabled("STATEFRAME_GAPS_WITNESSES") and suite is not None:
        blocks.append(render_witness_suite(suite))
        audit["truth"].append("witnesses")

    if truth_enabled("STATEFRAME_GAPS_NAVIGATION"):
        index = getattr(runtime, "gap_navigation_index", None)
        if index is None:
            sources = getattr(runtime, "gap_navigation_sources", {})
            if isinstance(sources, Mapping) and sources:
                index = build_index(sources, revision=revision, journal=journal)
        query = str(getattr(runtime, "gap_navigation_query", "") or "")
        rendered = render_navigation(locate(
            index, query, journal=journal,
        ) if index is not None and query else None)
        if rendered:
            blocks.append(rendered)
            audit["truth"].append("navigation")

    if truth_enabled("STATEFRAME_GAPS_FAILURE_CARDS"):
        failure = getattr(runtime, "gap_failure_evidence", None)
        if isinstance(failure, Mapping):
            card = build_failure_card(failure, journal=journal,
                                      symbol_index=getattr(runtime, "gap_symbol_index", None))
            rendered = render_failure_card(card)
            if rendered:
                blocks.append(rendered)
                audit["truth"].append("failure_cards")

    if truth_enabled("STATEFRAME_GAPS_SURFACE_COVERAGE"):
        bindings = getattr(runtime, "gap_surface_bindings", ())
        receipts = getattr(runtime, "gap_surface_receipts", ())
        if bindings:
            report = reduce_coverage(bindings, receipts, current_revision=revision,
                                     journal=journal)
            blocks.append(render_coverage(report))
            audit["truth"].append("surface_coverage")

    if behavior_on:
        from kernel_behavior_witness import request_block
        decision = getattr(runtime, "_last_completion_decision", None)
        open_ids = getattr(decision, "needs_work_ids", ())
        block = request_block(runtime, open_ids)
        if block:
            blocks.append(block)
            audit["truth"].append("behavior_witness")

    selected_model = str(model_name or getattr(runtime, "model_name", "")
                         or os.environ.get("STATEFRAME_MODEL_NAME", ""))
    if guidance_enabled(selected_model):
        state = getattr(runtime, "gap_hypothesis_state", None)
        observation = getattr(runtime, "gap_hypothesis_observation", None)
        candidates = getattr(runtime, "gap_hypothesis_candidates", ())
        if isinstance(state, Mapping) and isinstance(observation, Mapping):
            recovery = recover_hypothesis(state, observation, candidates,
                                          model_name=selected_model, journal=journal)
            rendered = render_recovery(recovery)
            if rendered:
                store = getattr(runtime, "decision_continuity", None)
                sync = getattr(store, "sync_organ_projection", None)
                if recovery is not None and callable(sync):
                    try:
                        sync(continuity_projection(recovery),
                             existing_refs=observation.get("evidence_refs", ()))
                    except Exception:  # noqa: BLE001 - advisory sync is fail-open
                        pass
                blocks.append(rendered)
                audit["guidance"] = True

    blocks = [block for block in blocks if block]
    if not blocks:
        return prepared if isinstance(prepared, list) else list(prepared), {}
    content = "\n\n".join(blocks)[:MAX_BLOCK_CHARS]
    available = None if capacity_tokens is None else max(0, int(capacity_tokens))
    if available is not None and _approx_tokens(content) + 16 > available:
        emit(journal, "gaps_runtime_projection_omitted", producer=PRODUCER,
             enabled_value=True, request=str(request_id), reason="capacity",
             projected_tokens=_approx_tokens(content), available_tokens=available)
        return list(prepared), {**audit, "omitted": "capacity"}
    outgoing = [dict(message) for message in prepared]
    outgoing.append({"role": "user", "content": content})
    emit(journal, "gaps_runtime_projection", producer=PRODUCER, enabled_value=True,
         request=str(request_id), truth_count=len(audit["truth"]),
         guidance=bool(audit["guidance"]), projected_tokens=_approx_tokens(content))
    return outgoing, audit


__all__ = ["PRODUCER", "apply_gap_request_hooks"]
