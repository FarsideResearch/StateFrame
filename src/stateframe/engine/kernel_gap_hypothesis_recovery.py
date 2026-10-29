"""Evidence-driven, model-filtered hypothesis recovery guidance."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_gap_switches import emit, guidance_enabled


SWITCH = "STATEFRAME_GAPS_HYPOTHESIS_GUIDANCE"
PRODUCER = "kernel_gap_hypothesis_recovery"
MAX_ACTIONS = 6


def _bounded(value: object, limit: int = 260) -> str:
    text = str(value or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


@dataclass(frozen=True)
class HypothesisRecovery:
    enabled: bool
    invalidated: bool
    hypothesis: str
    predicted_observation: str
    contradictory_observation: str
    evidence_refs: tuple[str, ...]
    rejected_hypotheses: tuple[str, ...]
    next_action_id: str
    next_action: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "invalidated": self.invalidated,
            "hypothesis": self.hypothesis,
            "predicted_observation": self.predicted_observation,
            "contradictory_observation": self.contradictory_observation,
            "evidence_refs": list(self.evidence_refs),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "next_action_id": self.next_action_id,
            "next_action": self.next_action,
            "reason": self.reason,
        }


def recover_hypothesis(state: Mapping[str, object], observation: Mapping[str, object],
                       candidates: Sequence[Mapping[str, object]], *, model_name: str,
                       journal: object | None = None) -> HypothesisRecovery | None:
    if not guidance_enabled(model_name):
        return None
    previous_dependency = str(state.get("dependency_fingerprint", ""))
    current_dependency = str(observation.get("dependency_fingerprint", previous_dependency))
    invalidated = bool(previous_dependency and current_dependency
                       and previous_dependency != current_dependency)
    hypothesis = "" if invalidated else _bounded(state.get("hypothesis"))
    predicted = "" if invalidated else _bounded(state.get("predicted_observation"))
    rejected = [str(item) for item in state.get("rejected_hypotheses", ())
                if str(item).strip()][:4]
    contradiction = _bounded(observation.get("actual", observation.get("observed", "")))
    if observation.get("contradicts") is True and hypothesis:
        rejected.append(hypothesis)
        hypothesis = ""
        predicted = ""
    refs = tuple(str(item) for item in observation.get("evidence_refs", ())
                 if str(item).strip())[:8]
    used = {str(state.get("next_action_id", ""))}
    used.update(str(item.get("action_id", "")) for item in state.get("action_history", ())
                if isinstance(item, Mapping))
    chosen: Mapping[str, object] | None = None
    for candidate in candidates[:MAX_ACTIONS]:
        action_id = str(candidate.get("action_id", ""))
        if action_id and action_id not in used:
            chosen = candidate
            break
    if chosen is None:
        action_id = "escalate-unresolved"
        action = "escalate with the contradictory receipt or mark this surface unresolved"
        reason = "candidate_actions_exhausted"
    else:
        action_id = str(chosen.get("action_id"))
        action = _bounded(chosen.get("action"), 220)
        reason = "dependency_changed" if invalidated else "contradiction_requires_new_observation"
    result = HypothesisRecovery(True, invalidated, hypothesis, predicted, contradiction, refs,
                                tuple(rejected[:4]), action_id, action, reason)
    emit(journal, "gaps_hypothesis_recovery", producer=PRODUCER, enabled_value=True,
         model=str(model_name), invalidated=result.invalidated,
         next_action_id=result.next_action_id, reason=result.reason,
         evidence_count=len(result.evidence_refs))
    return result


def render_recovery(recovery: HypothesisRecovery | None) -> str:
    if recovery is None or not recovery.enabled:
        return ""
    payload = {
        "hypothesis": recovery.hypothesis or "none; do not promote a rejected claim",
        "contradictory_observation": recovery.contradictory_observation,
        "rejected_hypotheses": recovery.rejected_hypotheses,
        "next_action": recovery.next_action,
        "evidence_refs": recovery.evidence_refs,
        "advisory": True,
    }
    return "[Advisory hypothesis recovery]\n" + json.dumps(payload, sort_keys=True,
                                                               separators=(",", ":"))


def continuity_projection(recovery: HypothesisRecovery) -> dict[str, object]:
    """Return only fields accepted by v21's existing continuity organ."""
    projection: dict[str, object] = {
        "evidence_refs": list(recovery.evidence_refs),
        "rejected_hypotheses": [
            {"text": text, "source": PRODUCER, "evidence_refs": list(recovery.evidence_refs)}
            for text in recovery.rejected_hypotheses
        ],
        "next_discriminating_action": {
            "text": recovery.next_action,
            "source": PRODUCER,
        },
    }
    if recovery.hypothesis:
        projection["hypothesis"] = {"text": recovery.hypothesis, "source": PRODUCER}
    return projection


__all__ = [
    "HypothesisRecovery", "PRODUCER", "SWITCH", "continuity_projection",
    "recover_hypothesis", "render_recovery",
]
