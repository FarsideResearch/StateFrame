"""unified task state's single host-owned state and frame projection.

The unified arm is deliberately additive. ``STATEFRAME_UNIFIED=off`` is a
deep-copy/no-op boundary, so the existing native payload remains the versioned
baseline. The ON arm folds the older receipt, S1, boot, investigation, and
completion presentations into five durable objects and renders one bounded
block. Model text is never used as evidence here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kernel_verdict import CONFLICT, SATISFIED, VIOLATED, evaluate_required_tests


MODE_ENV = "STATEFRAME_UNIFIED"
VERSION = "design-20-unified/1"
OBJECTS = ("Snapshot", "Obligation", "Observation", "Decision", "PolicyArtifact")
EVIDENCE_AXES = ("binding", "assessment", "interpretation")
DECISION_EXITS = ("OPEN", "RESOLVED", "NARROWED", "NO_DISCRIMINATION", "INCONCLUSIVE")
_PASS = {"GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED", "REUSED_VERIFICATION", "OK"}
_FAIL = {"RED", "FAIL", "FAILED", "ERROR", "DEFECT"}


def unified_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return unified_mode() == "on"


def _rows(value: object) -> list[dict]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _text(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _revision(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("current_revision", "revision", "product_revision", "artifact_revision",
                    "source_revision", "after_revision", "candidate_revision"):
            if value.get(key) is not None:
                return str(value[key])
        for key in ("binding", "after_binding", "current_binding"):
            found = _revision(value.get(key))
            if found is not None:
                return found
    return None if value is None or isinstance(value, Mapping) else str(value)


def current_revision(frame: Mapping[str, object]) -> str | None:
    for key in ("current_revision", "revision", "product_revision"):
        if frame.get(key) is not None:
            return str(frame[key])
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping):
        values = [value for value in revisions.values() if value is not None]
        if len(values) == 1:
            return str(values[0])
    binding = frame.get("binding")
    return _revision(binding)


def _jobs_block(frame: Mapping[str, object]) -> str | None:
    jobs = frame.get("jobs")
    if jobs is None:
        return None
    from kernel_job_watch import enabled as job_watch_enabled, render_jobs_block
    if not job_watch_enabled():
        return None
    return render_jobs_block(jobs)


def _status(row: Mapping[str, object]) -> str:
    for key in ("result_class", "result", "status", "outcome", "verdict"):
        if row.get(key) is not None:
            return str(row[key]).upper()
    return "UNKNOWN"


def _receipt_rows(frame: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "fault_receipts"):
        rows.extend(_rows(frame.get(key)))
    rows.extend(_rows(frame.get("receipts")))
    latest = frame.get("latest_receipts")
    if isinstance(latest, Mapping):
        rows.extend(_rows(latest))
    last_action = frame.get("last_action")
    if isinstance(last_action, Mapping):
        rows.append(dict(last_action))
    return rows


def _latest_test(frame: Mapping[str, object]) -> dict | None:
    rows = [row for row in _receipt_rows(frame)
            if str(row.get("kind", row.get("tool", ""))).lower()
            in {"test", "run_tests", "pytest", "verification"}
            or any(key in row for key in ("selectors", "selector", "passed", "exit_code", "feedback"))]
    return rows[-1] if rows else None


def _selector(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "UNKNOWN"
    for key in ("target", "suite", "selector", "test"):
        if row.get(key):
            return _text(row[key], 140)
    selectors = row.get("selectors")
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        return _text(", ".join(str(item) for item in selectors), 140) or "UNKNOWN"
    return "UNKNOWN"


def _axis_status(row: Mapping[str, object], current: str | None,
                 *, mapped: bool = True, interpreted: bool = True) -> dict[str, str]:
    observed = _revision(row)
    if observed is None or current is None:
        binding = "UNKNOWN"
    elif observed == current:
        binding = "CURRENT"
    else:
        binding = "STALE"
    result = _status(row)
    if result in _FAIL or row.get("witnessed_defect") is True:
        assessment = "FAIL_WITH_WITNESS"
    elif result in _PASS and mapped:
        assessment = "PASS_ON_LISTED_CASES"
    elif result in _PASS:
        assessment = "EXECUTED_WITHOUT_MAPPED_ASSERTION"
    else:
        assessment = "UNEXERCISED"
    interpretation = "SETTLED" if interpreted and result not in {"UNKNOWN", "NOT_RUN"} else "INTERPRETATION_OPEN"
    return {"binding": binding, "assessment": assessment, "interpretation": interpretation}


def _axis_text(axis: Mapping[str, object]) -> str:
    return " ".join(f"{key}={axis.get(key, 'UNKNOWN')}" for key in EVIDENCE_AXES)


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return "u-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _explicit_commitments(frame: Mapping[str, object]) -> list[str]:
    values: list[object] = []
    for key in ("explicit_proposer_commitments", "proposer_commitments"):
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
        elif value:
            values.append(value)
    decision = frame.get("decision")
    if isinstance(decision, Mapping):
        value = decision.get("proposer_commitments")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
    result = []
    for value in values:
        if isinstance(value, Mapping):
            if value.get("explicit") is not True and value.get("committed") is not True:
                continue
            value = value.get("action") or value.get("text") or value.get("proposal")
        if str(value).strip():
            result.append(_text(value, 160))
    return list(dict.fromkeys(result))


def _decision_input(frame: Mapping[str, object]) -> dict:
    values = []
    for key in ("decision", "decision_ticket", "active_investigation", "investigation"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            values.append(dict(value))
    decision_state = frame.get("decision_state")
    if isinstance(decision_state, Mapping):
        values.append(dict(decision_state))
    merged: dict[str, object] = {}
    for value in values:
        merged.update(value)
    return merged


@dataclass
class DecisionLifecycle:
    """The kernel-owned lifecycle for investigation, tickets, and repetition."""

    status: str = "OPEN"
    evidence_epoch: str | None = None
    repetition_fingerprints: set[str] = field(default_factory=set)
    exposures: dict[str, int] = field(default_factory=dict)
    progress_fingerprints: dict[str, str] = field(default_factory=dict)
    adopted_action: str | None = None
    output_cap_suppressed: bool = False

    def note_output_cap(self) -> None:
        self.output_cap_suppressed = True

    def begin_request(self) -> None:
        # The suppression is explicitly one request wide.
        self.output_cap_suppressed = False

    def consume_output_cap(self) -> None:
        """Consume the one-request post-cap suppression at the wire boundary."""
        self.output_cap_suppressed = False

    def update(self, frame: Mapping[str, object], *, evidence_epoch: object = None,
               progress: object = None) -> dict:
        decision = _decision_input(frame)
        epoch = str(evidence_epoch if evidence_epoch is not None else
                    frame.get("evidence_epoch", current_revision(frame) or "UNKNOWN"))
        if epoch != self.evidence_epoch:
            self.evidence_epoch = epoch
        explicit = _explicit_commitments(frame)
        self.adopted_action = explicit[-1] if explicit else None
        raw_status = str(decision.get("status", decision.get("exit", ""))).upper()
        if raw_status in DECISION_EXITS:
            self.status = raw_status
        elif decision.get("resolved") is True:
            self.status = "RESOLVED"
        elif decision.get("narrowed") is True:
            self.status = "NARROWED"
        elif decision.get("inconclusive") is True:
            self.status = "INCONCLUSIVE"
        elif decision.get("reopen_dependencies") or decision.get("dependencies_current") is False:
            self.status = "OPEN"
        elif decision.get("open") is True or decision:
            self.status = "OPEN"
        cycle_count = decision.get("cycle_count", decision.get("repetition_count", 0))
        try:
            cycle_count = int(cycle_count)
        except (TypeError, ValueError):
            cycle_count = 0
        fingerprint = decision.get("repetition_fingerprint")
        if not fingerprint:
            fingerprint = _digest({
                "question": decision.get("question", frame.get("investigation_question", "")),
                "observation": decision.get("observed", decision.get("observation", "")),
                "obligation": decision.get("obligation", frame.get("obligation", "")),
                "epoch": self.evidence_epoch,
            })
        repeated = bool(decision.get("repeated") or decision.get("repetition")
                        or cycle_count >= 3)
        repetition_first = repeated and str(fingerprint) not in self.repetition_fingerprints
        if repeated:
            self.repetition_fingerprints.add(str(fingerprint))
            if self.status == "OPEN" and decision.get("settled"):
                self.status = "NO_DISCRIMINATION"
        key = str(fingerprint)
        progress_fp = _digest(progress if progress is not None else frame.get("progress", frame.get("receipt_count", 0)))
        if key not in self.progress_fingerprints or self.progress_fingerprints[key] != progress_fp:
            self.exposures[key] = 0
            self.progress_fingerprints[key] = progress_fp
        return {"status": self.status, "fingerprint": key, "repeated": repeated,
                "repetition_first": repetition_first,
                "cycle_count": cycle_count, "settled": decision.get("settled"),
                "question": decision.get("question", frame.get("investigation_question", "")),
                "observed": decision.get("observed", decision.get("observation", "")),
                "adopted_action": self.adopted_action,
                "reopen_dependencies": bool(decision.get("reopen_dependencies")
                                             or decision.get("dependencies_current") is False),
                "output_cap_suppressed": self.output_cap_suppressed}

    def may_render(self, *, trigger_active: bool, decision_ref: object,
                   obligation_ref: object, dependencies_current: bool = True,
                   adds_new_info: bool = True, fits_qualifiers: bool = True,
                   evidence_epoch: object = None, progress: object = None,
                   escalation: bool = False) -> bool:
        if not trigger_active or not decision_ref or not obligation_ref or not dependencies_current:
            return False
        if not adds_new_info or not fits_qualifiers:
            return False
        if evidence_epoch is not None and str(evidence_epoch) != self.evidence_epoch:
            return False
        fingerprint = str(decision_ref)
        progress_fp = _digest(progress if progress is not None else "")
        if self.progress_fingerprints.get(fingerprint) != progress_fp:
            self.exposures[fingerprint] = 0
            self.progress_fingerprints[fingerprint] = progress_fp
        if self.exposures.get(fingerprint, 0) >= 2:
            return False
        if escalation and self.output_cap_suppressed:
            return False
        self.exposures[fingerprint] = self.exposures.get(fingerprint, 0) + 1
        return True


def may_render(*, enabled: bool, trigger_active: bool, decision_ref: object = None,
               obligation_ref: object = None, dependencies_current: bool = True,
               adds_new_info: bool = True, fits_qualifiers: bool = True,
               suppressed_this_evidence_epoch: bool = False,
               exposures_without_progress: int = 0, output_cap_hit: bool = False,
               escalation: bool = False) -> bool:
    """Pure public predicate for optional unified proposals."""
    return bool(enabled and trigger_active and decision_ref and obligation_ref
                and dependencies_current and adds_new_info and fits_qualifiers
                and not suppressed_this_evidence_epoch and exposures_without_progress < 2
                and not (escalation and output_cap_hit))


def _completion(frame: Mapping[str, object], current: str | None, latest: Mapping[str, object] | None) -> dict[str, object]:
    from kernel_sweep_fixes import completion_projection, enabled as sweep_fixes_enabled
    contract = frame.get("completion_contract")
    checks = contract.get("current_public_checks") if isinstance(contract, Mapping) else None
    required = (contract.get("required_public_checks", ())
                if isinstance(contract, Mapping) else frame.get("required_public_checks", ()))
    if sweep_fixes_enabled():
        atomic = completion_projection(frame)
        verdict_state = str(atomic["outcome"])
        reason_codes = list(atomic["reason_codes"])
        checked = "UNKNOWN"
        mandatory = "ATOMIC_" + verdict_state
    else:
        atomic = None
        verdict = evaluate_required_tests(frame, required, _receipt_rows(frame))
        verdict_state = verdict.state
        reason_codes = list(verdict.reason_codes)
        checked = _revision(checks)
        if verdict.state == SATISFIED:
            mandatory = "PASS_CURRENT"
        elif verdict.state in {VIOLATED, CONFLICT}:
            mandatory = "FAIL_CURRENT"
        else:
            mandatory = "MISSING_OR_STALE"
    defects: list[str] = []
    for key in ("witnessed_defects", "known_violations", "known_violation", "host_observed_findings"):
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            defects.extend(_text(item, 180) for item in value if str(item).strip())
        elif value:
            defects.append(_text(value, 180))
    s1 = frame.get("stateframe_v2_s1")
    if isinstance(s1, Mapping):
        for anomaly in _rows(s1.get("anomalies")):
            if str(anomaly.get("kind", "")) in {"live_stub", "public_symbol_execution", "edit_rejection"}:
                defects.append(_text(anomaly.get("id", anomaly.get("symbol", "witnessed defect")), 180))
    defects = list(dict.fromkeys(defects))
    if atomic is not None:
        done = str(atomic["outcome"])
        submission = "ELIGIBLE" if atomic.get("accepted") is True else "BLOCKED"
    else:
        explicit_done = frame.get("done_admission")
        if explicit_done in {"ADMISSIBLE", "REFUSE_ONCE", "BLOCKED"}:
            done = str(explicit_done)
        elif mandatory != "PASS_CURRENT":
            done = "BLOCKED"
        elif defects:
            done = "REFUSE_ONCE"
        else:
            done = "ADMISSIBLE"
        submission = frame.get("submission")
        if submission not in {"ELIGIBLE", "BLOCKED"}:
            submission = "ELIGIBLE" if done == "ADMISSIBLE" else "BLOCKED"
    return {"mandatory_checks": mandatory, "verdict_state": verdict_state,
            "reason_codes": reason_codes,
            "checked_revision": checked or "UNKNOWN",
            "current_revision": current or "UNKNOWN", "witnessed_defects": defects,
            "done_admission": done, "submission": submission}


def render_unified_frame(frame: Mapping[str, object], *, lifecycle: DecisionLifecycle | None = None,
                         controller: DecisionLifecycle | None = None,
                         include_proposal: bool = True) -> str:
    """Render one deterministic unified task state block; completion is always last."""
    if not isinstance(frame, Mapping):
        raise TypeError("unified frame requires a mapping")
    from kernel_completion_status import (
        completion_status_from_frame, enabled as blocker_render_enabled,
        render_completion_status,
    )
    blocker_status = (completion_status_from_frame(frame)
                      if blocker_render_enabled() else None)
    if blocker_status is not None:
        lines = [
            f"[STATEFRAME UNIFIED | {VERSION}]",
            "ASSIST BLOCKS: facts + proposals; no commands",
            "Snapshot | CONTRACT",
            f"task={_text(frame.get('task_id', 'UNKNOWN'), 100)} current_revision={current_revision(frame) or 'UNKNOWN'}",
        ]
        jobs = _jobs_block(frame)
        if jobs is not None:
            lines.append(jobs)
        lines.extend(("Decision | KERNEL", render_completion_status(blocker_status)))
        return "\n".join(lines)
    lifecycle = lifecycle or controller or DecisionLifecycle()
    current = current_revision(frame)
    latest = _latest_test(frame)
    decision = lifecycle.update(frame)
    lines = [f"[STATEFRAME UNIFIED | {VERSION}]", "ASSIST BLOCKS: facts + proposals; no commands"]
    lines.append("Snapshot | CONTRACT")
    lines.append(f"task={_text(frame.get('task_id', 'UNKNOWN'), 100)} current_revision={current or 'UNKNOWN'}")
    boot = frame.get("boot_packet")
    if isinstance(boot, Mapping):
        lines.append(f"boot_capsule=present schema={boot.get('schema', 'stateframe-boot/1')} clauses=bounded")
    lines.append("Obligation | CONTRACT")
    contract = frame.get("completion_contract")
    required = contract.get("required_public_checks", ()) if isinstance(contract, Mapping) else ()
    lines.append(f"mandatory_scope={_text(required, 220) if required else 'UNKNOWN'}")
    lines.append("Observation | OBSERVED")
    if latest is not None:
        axis = _axis_status(latest, current,
                            mapped=bool(latest.get("selector", latest.get("selectors", latest.get("target")))),
                            interpreted=True)
        lines.append(f"receipt={_text(latest.get('receipt_id', latest.get('event_id', 'UNKNOWN')), 60)} "
                     f"selector={_selector(latest)} revision={_revision(latest) or 'UNKNOWN'} {_axis_text(axis)}")
    else:
        lines.append("receipt=none binding=UNKNOWN assessment=UNEXERCISED interpretation=INTERPRETATION_OPEN")
    jobs = _jobs_block(frame)
    if jobs is not None:
        lines.append(jobs)
    s1 = frame.get("stateframe_v2_s1")
    if isinstance(s1, Mapping):
        for component in _rows(s1.get("components"))[:24]:
            state = str(component.get("state", "present"))
            assessment = "FAIL_WITH_WITNESS" if state == "witnessed_defect" else (
                "PASS_ON_LISTED_CASES" if state == "evidence_available" else "UNEXERCISED")
            binding = "CURRENT" if component.get("binding") else "UNKNOWN"
            lines.append(f"map={component.get('id', 'UNKNOWN')} symbol={component.get('name', 'UNKNOWN')} "
                         f"binding={binding} assessment={assessment} interpretation=INTERPRETATION_OPEN")
        for anomaly in _rows(s1.get("anomalies"))[:24]:
            lines.append(f"anomaly={anomaly.get('id', 'UNKNOWN')} kind={anomaly.get('kind', 'UNKNOWN')} "
                         f"symbol={anomaly.get('symbol', 'UNKNOWN')} binding={anomaly.get('last_revision', current) == current and 'CURRENT' or 'STALE'} "
                         "assessment=FAIL_WITH_WITNESS interpretation=INTERPRETATION_OPEN")
    lines.append("Decision | KERNEL")
    repetition = (f"repetition_fingerprint={decision['fingerprint']}"
                  if not decision.get("repeated") or decision.get("repetition_first")
                  else "repetition_fingerprint=RECORDED_THIS_EPOCH")
    lines.append(f"lifecycle={decision['status']} {repetition} epoch={lifecycle.evidence_epoch}")
    if decision.get("settled"):
        lines.append(f"settled={_text(decision['settled'], 80)}")
    if decision.get("reopen_dependencies"):
        lines.append("dependencies=STALE reopen=OPEN")
    if decision.get("adopted_action"):
        lines.append(f"adopted_action={_text(decision['adopted_action'], 180)} proposer=EXPLICIT")
    trigger = bool(frame.get("trigger_active", frame.get("trigger", frame.get("investigation_question"))))
    obligation_ref = required[0] if isinstance(required, Sequence) and not isinstance(required, (str, bytes)) and required else frame.get("obligation", "obligation")
    proposal = lifecycle.may_render(
        trigger_active=trigger, decision_ref=decision["fingerprint"], obligation_ref=obligation_ref,
        dependencies_current=not decision.get("reopen_dependencies", False),
        adds_new_info=bool(frame.get("adds_new_info", True)), fits_qualifiers=bool(frame.get("fits_qualifiers", True)),
        evidence_epoch=lifecycle.evidence_epoch, progress=frame.get("progress", frame.get("receipt_count", 0)),
        escalation=True)
    if include_proposal and proposal:
        question = decision.get("question") or "next host discriminator"
        lines.append(f"proposal=host_observation_needed basis={_text(question, 180)}")
    elif lifecycle.output_cap_suppressed:
        lines.append("proposal=SUPPRESSED_AFTER_OUTPUT_CAP_NEXT_REQUEST")
    policy = frame.get("policy_artifact")
    lines.append("PolicyArtifact | PROPOSED")
    lines.append(f"status={_text(policy.get('status', 'BOUNDARY_FACTS_ONLY') if isinstance(policy, Mapping) else 'BOUNDARY_FACTS_ONLY', 100)}")
    completion = _completion(frame, current, latest)
    lines.append("COMPLETION PROJECTION | KERNEL")
    lines.append(f"mandatory_checks: {completion['mandatory_checks']} (revision={completion['current_revision']})")
    lines.append("witnessed_defects: " + ("; ".join(completion["witnessed_defects"]) if completion["witnessed_defects"] else "none"))
    lines.append(f"done_admission: {completion['done_admission']}")
    lines.append(f"submission: {completion['submission']}")
    return "\n".join(lines)


def append_unified_block(messages: Sequence[dict], frame: Mapping[str, object], *,
                         provider_request: bool = True, lifecycle: DecisionLifecycle | None = None) -> list[dict]:
    result = copy.deepcopy(list(messages))
    if provider_request and unified_mode() == "on":
        content = render_unified_frame(frame, lifecycle=lifecycle)
        result.append({"role": "user", "content": content})
        if lifecycle is not None:
            lifecycle.consume_output_cap()
    return result


__all__ = [
    "DECISION_EXITS", "DecisionLifecycle", "EVIDENCE_AXES", "MODE_ENV", "OBJECTS", "VERSION",
    "append_unified_block", "current_revision", "enabled", "may_render", "render_unified_frame",
    "unified_mode",
]
