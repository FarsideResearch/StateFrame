"""Bounded host-owned progress state for persistent StateFrame series."""
from __future__ import annotations

import copy
import dataclasses
import os
from collections import deque
from collections.abc import Iterable, Mapping
from typing import TypedDict

from kernel_measurement import digest

SCHEMA = "frame-series/1"
EVENT_SCHEMA = "frame-series-event/1"
ENABLE_ENV = "KERNEL_FRAME_SERIES"
CONTROL_ENV = "KERNEL_FRAME_SERIES_CONTROL"
DEFAULT_NO_PROGRESS_THRESHOLD = 3
_PHASES = frozenset({
    "orienting", "diagnosing", "implementing", "repairing", "verifying",
    "reviewing", "revisiting", "blocked", "handoff", "stopped",
    "review_requirements_then_done", "verify_or_repair_current_change",
    "make_a_testable_change",
})
_INVALID_RESULTS = frozenset({
    "REUSED_VERIFICATION", "NO_EXECUTION_RECEIPT", "REJECTED_CONTRACT",
    "REJECTED", "REFUSED", "REFUSED_PREDISPATCH", "DENIED_PATH",
    "INVALID_PROPOSAL", "NOT_EXECUTED", "REPEAT_SUPPRESSED",
    "REFUSED_SERIES",
    "SELECTOR_ERROR", "NO_TESTS", "NO_VERDICT",
})
_MAX_TRANSITIONS = 8
_MAX_RESTRICTIONS = 8
_MAX_SEEN = 64


class FrameSeriesView(TypedDict):
    schema: str
    enabled: bool
    control_enabled: bool
    series_id: str
    objective_revision: str
    scope: dict
    plan: dict
    work: dict
    phase: dict
    progress: dict
    binding: dict
    proof: dict
    missing_information: list[dict]
    phase_transitions: list[dict]
    restrictions: list[dict]
    budget: dict


@dataclasses.dataclass(frozen=True)
class _ActionIdentity:
    tool: str
    arguments_hash: str
    binding_digest: str

    def view(self) -> dict[str, str]:
        return {"tool": self.tool, "arguments_hash": self.arguments_hash,
                "binding_digest": self.binding_digest}


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def default_enabled() -> bool:
    return _flag(ENABLE_ENV)


def default_control_enabled() -> bool:
    return _flag(CONTROL_ENV)


def _text(value: object, limit: int = 200) -> str:
    if value is None:
        return ""
    return value[:limit] if isinstance(value, str) else str(value)[:limit]


def _digest(value: object) -> str:
    return digest(value)


def canonical_argument_digest(
    tool: object, arguments: Mapping | None, *, fallback: str | None = None,
) -> str:
    """Digest dispatch arguments with only proven read-default normalization."""
    name = _text(tool, 64)
    clean = ({key: value for key, value in arguments.items()
              if key not in {"why", "reason"}}
             if isinstance(arguments, Mapping) else {})
    normalized_read = False
    if name == "read":
        path, line = clean.get("path"), clean.get("line")
        if isinstance(path, str) and path:
            normalized_read = True
            if "line" not in clean:
                clean["line"] = 1
            elif type(line) is int and line == 1:
                clean["line"] = 1
    if not normalized_read and isinstance(fallback, str) and fallback:
        return fallback
    return _digest(clean)


def _metadata(value: object, keys: set[str]) -> dict:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    sequence_keys = {"dependencies", "depends_on", "dependency_ids", "gate_checks"}
    for key in keys:
        if key not in value:
            continue
        item = value[key]
        if key in sequence_keys:
            if isinstance(item, (list, tuple)):
                result[key] = [_text(entry, 120) for entry in item[:8]]
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item if not isinstance(item, str) else item[:200]
    return result


def _missing(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if any(key in value for key in ("code", "id", "kind", "status", "summary")):
            value = [value]
        else:
            value = [{"code": entry, "kind": bucket}
                     for bucket, entries in value.items()
                     if isinstance(entries, (list, tuple))
                     for entry in entries]
    elif isinstance(value, str):
        value = [{"code": value}]
    if not isinstance(value, Iterable):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping):
            code = _text(item.get("code") or item.get("id") or item.get("kind"), 120)
            kind = _text(item.get("kind") or "unknown", 80)
            status = _text(item.get("status") or "open", 40)
            summary = _text(item.get("summary") or item.get("detail"), 240)
            ref = _text(item.get("evidence_ref") or item.get("ref"), 160)
        else:
            code, kind, status, summary, ref = _text(item, 120), "unknown", "open", "", ""
        if code:
            result.append({"code": code, "kind": kind, "status": status,
                           "summary": summary, "evidence_ref": ref})
        if len(result) == 16:
            break
    return result


def _binding_ref(binding: Mapping | None) -> dict:
    binding = binding if isinstance(binding, Mapping) else {}
    result = {"digest": _digest(binding)}
    for key in ("product_revision", "test_harness_revision", "environment_revision"):
        value = binding.get(key)
        if isinstance(value, (str, int)):
            result[key] = _text(value, 180)
    return result


def _file_revision(binding: Mapping, path: str) -> object:
    for field in ("product_files", "harness_files"):
        files = binding.get(field)
        if isinstance(files, Mapping) and path in files:
            return files[path]
    return None


def _identity(record: Mapping, binding_digest: str) -> _ActionIdentity:
    tool = _text(record.get("tool") or "unknown", 64)
    args_hash = record.get("arguments_hash")
    parameters = record.get("action_parameters")
    arguments = parameters if isinstance(parameters, Mapping) else record.get("arguments")
    args_hash = canonical_argument_digest(
        tool, arguments if isinstance(arguments, Mapping) else None,
        fallback=args_hash if isinstance(args_hash, str) else None,
    )
    return _ActionIdentity(tool, _text(args_hash, 180), binding_digest)


def _selectors(record: Mapping, proof: Mapping | None) -> list[str]:
    values = proof.get("selectors") if isinstance(proof, Mapping) else None
    if not isinstance(values, (list, tuple)):
        values = record.get("selectors")
    if not isinstance(values, (list, tuple)):
        params = record.get("action_parameters")
        target = params.get("target") if isinstance(params, Mapping) else record.get("target")
        values = [target] if target else []
    return [_text(value, 180) for value in values[:8] if value is not None]


def _result(record: Mapping, proof: Mapping | None) -> str:
    value = record.get("result")
    if value:
        return _text(value, 80)
    if isinstance(proof, Mapping):
        return _text(proof.get("result_class", proof.get("result")) or "UNKNOWN", 80)
    return "UNKNOWN"


def _host_observed(proof: Mapping) -> bool:
    authority = _text(proof.get("authority"), 80).upper()
    return proof.get("host_observed") is True or authority in {
        "HOST_OBSERVED", "HOST_RECEIPT", "HOST_VERIFIED",
    }


class FrameSeries:
    """Bounded series cursor; ParentMeter remains the only execution budget."""

    def __init__(
        self, *, task_id: str, path: str | None = None, symbol: str | None = None,
        scope: Mapping | None = None, plan: Mapping | None = None,
        work: Mapping | None = None, objective_revision: str | None = None,
        parent_budget: object | None = None, enabled: bool | None = None,
        control_enabled: bool | None = None,
        no_progress_threshold: int = DEFAULT_NO_PROGRESS_THRESHOLD,
    ):
        if not 1 <= no_progress_threshold <= 16:
            raise ValueError("no_progress_threshold must be between 1 and 16")
        self.enabled = default_enabled() if enabled is None else bool(enabled)
        self.control_enabled = (default_control_enabled() if control_enabled is None
                                else bool(control_enabled))
        self.no_progress_threshold = no_progress_threshold
        self.parent_budget = parent_budget
        scope_value = {"id": scope} if isinstance(scope, str) else scope
        self.scope = _metadata(scope_value, {
            "id", "unit_id", "parent_id", "kind", "objective_id",
            "dependencies", "dependency_ids",
        })
        self.scope.update(task_id=_text(task_id, 160), path=_text(path, 240),
                          symbol=_text(symbol, 160))
        self.plan = _metadata(plan, {
            "id", "unit_id", "digest", "revision", "dependencies",
            "dependency_ids", "gate_checks",
        })
        self.work = _metadata(work, {
            "id", "version", "status", "objective", "acceptance",
            "next_action", "blocker", "depends_on",
        })
        self.objective_revision = _text(objective_revision, 180) or _digest({
            "scope": self.scope, "plan": self.plan, "work": self.work,
        })
        self.series_id = _digest({
            "task_id": self.scope["task_id"],
            "objective_revision": self.objective_revision, "scope": self.scope,
        })
        self._phase = "orienting"
        self._phase_sequence = 0
        self._transitions: deque[dict] = deque(maxlen=_MAX_TRANSITIONS)
        self._missing: list[dict] = []
        self._binding_digest: str | None = None
        self._binding_view = _binding_ref(None)
        self._proof: dict | None = None
        self._proof_status = "none"
        version = self.work.get("version")
        self._decision_state_version = (
            version if isinstance(version, (str, int)) and not isinstance(version, bool) else None
        )
        self._sequence = 0
        self._action_counts: dict[tuple[str, str, str], int] = {}
        self._seen_reads: deque[str] = deque(maxlen=_MAX_SEEN)
        self._seen_read_set: set[str] = set()
        self._seen_results: deque[str] = deque(maxlen=_MAX_SEEN)
        self._seen_result_set: set[str] = set()
        self._cycle_identities: dict[tuple[str, str, str], _ActionIdentity] = {}
        self._progress = {
            "actions": 0, "meaningful": 0, "binding_changes": 0,
            "bound_reads": 0, "selector_results": 0, "work_versions": 0,
            "no_progress_cycles": 0, "repeated_actions": 0,
        }
        self._last_event: dict | None = None

    @staticmethod
    def _remember(queue: deque[str], seen: set[str], key: str) -> bool:
        if key in seen:
            return False
        if len(queue) >= _MAX_SEEN:
            seen.discard(queue.popleft())
        queue.append(key)
        seen.add(key)
        return True

    def _read_is_new(self, record: Mapping, binding: Mapping, binding_digest: str) -> bool:
        if record.get('result') in _INVALID_RESULTS:
            return False
        evidence = record.get("read_evidence")
        if not isinstance(evidence, Mapping):
            return False
        path, revision = _text(evidence.get("path"), 240), evidence.get("file_revision")
        if not path or not isinstance(revision, str) or not revision or _file_revision(binding, path) != revision:
            return False
        key = _digest({
            "path": path, "revision": revision,
            "start_line": evidence.get("start_line"), "end_line": evidence.get("end_line"),
            "content_hash": evidence.get("content_hash"), "binding": binding_digest,
        })
        return self._remember(self._seen_reads, self._seen_read_set, key)

    def _selector_is_new(
        self, record: Mapping, proof: Mapping | None, binding_digest: str,
    ) -> bool:
        if _text(record.get("tool"), 64) != "run_tests":
            return False
        if not isinstance(proof, Mapping) or not _host_observed(proof):
            return False
        if not (proof.get("receipt_id") or proof.get("receipt_ref")):
            return False
        proof_binding = proof.get("binding")
        if not isinstance(proof_binding, Mapping) or not proof_binding:
            return False
        if _digest(proof_binding) != binding_digest:
            return False
        selectors, result = _selectors(record, proof), _result(record, proof)
        if not selectors or result not in {'GREEN', 'RED'}:
            return False
        if proof.get('result', proof.get('result_class')) != result:
            return False
        return self._remember(
            self._seen_results, self._seen_result_set,
            _digest({"selectors": selectors, "binding": binding_digest, "result": result}),
        )

    def transition(
        self, phase: str, *, reason: str = "", evidence_ref: str | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        phase = _text(phase, 64)
        if phase not in _PHASES:
            raise ValueError(f"unknown frame-series phase: {phase}")
        if phase == self._phase:
            return copy.deepcopy(self._transitions[-1]) if self._transitions else None
        self._phase_sequence += 1
        item = {
            "sequence": self._phase_sequence, "from_phase": self._phase,
            "to_phase": phase, "reason": _text(reason, 240),
            "evidence_ref": _text(evidence_ref, 180),
        }
        item["transition_ref"] = _digest({"series_id": self.series_id, **item})
        self._phase = phase
        self._transitions.append(item)
        return copy.deepcopy(item)

    def set_missing_information(self, value: object) -> list[dict]:
        if not self.enabled:
            return []
        self._missing = _missing(value)
        return copy.deepcopy(self._missing)

    def fold(
        self, record: Mapping, *, binding: Mapping | None = None,
        proof: Mapping | None = None, decision_state_version: str | int | None = None,
        missing_info: object = None, phase: str | None = None,
        phase_reason: str = "", phase_evidence_ref: str | None = None,
    ) -> dict:
        if not isinstance(record, Mapping):
            raise TypeError("frame-series fold requires a closed record mapping")
        if not self.enabled:
            return {"schema": EVENT_SCHEMA, "enabled": False,
                    "meaningful_progress": False, "progress_reasons": [],
                    "no_progress_cycles": 0}
        after = binding if isinstance(binding, Mapping) else record.get("after_binding")
        after = after if isinstance(after, Mapping) else {}
        after_digest = _digest(after)
        before = record.get("before_binding")
        before_digest = _digest(before) if isinstance(before, Mapping) else None
        changed = ((before_digest is not None and before_digest != after_digest)
                   or (self._binding_digest is not None
                       and self._binding_digest != after_digest))
        self._binding_digest, self._binding_view = after_digest, _binding_ref(after)
        if changed:
            self._progress["binding_changes"] += 1
            if self._proof_status != "none":
                self._proof_status = "stale_binding"

        proof = proof if isinstance(proof, Mapping) else record.get("proof")
        proof = proof if isinstance(proof, Mapping) else None
        if proof is not None:
            selectors, result = _selectors(record, proof), _result(record, proof)
            proof_result = proof.get('result', proof.get('result_class'))
            verdict_consistent = result in {'GREEN', 'RED'} and proof_result == result
            proof_binding = proof.get("binding")
            proof_binding_digest = (
                _digest(proof_binding)
                if isinstance(proof_binding, Mapping) and proof_binding else ""
            )
            receipt_ref = _text(
                proof.get("receipt_id") or proof.get("receipt_ref"), 180
            )
            observed = bool(
                receipt_ref and proof_binding_digest and _host_observed(proof)
            )
            self._proof = {
                "receipt_ref": receipt_ref, "selector_digest": _digest(selectors),
                "result": result, "binding_digest": proof_binding_digest,
                "host_observed": observed,
                "proof_ref": _digest({
                    "receipt_id": receipt_ref, "selectors": selectors,
                    "result": result, "binding_digest": proof_binding_digest,
                }),
            }
            self._proof_status = (
                "current_host_observation"
                if observed and proof_binding_digest == after_digest and verdict_consistent
                else "unverified_or_stale"
            )

        action = _identity(record, after_digest)
        action_key = (action.tool, action.arguments_hash, action.binding_digest)
        prior = self._action_counts.get(action_key, 0)
        self._action_counts[action_key] = min(prior + 1, _MAX_SEEN)
        if len(self._action_counts) > _MAX_SEEN:
            self._action_counts.pop(next(iter(self._action_counts)))
        self._progress["actions"] += 1
        self._progress["repeated_actions"] += int(prior > 0)
        reasons = []
        if changed:
            reasons.append("binding_change")
        if self._read_is_new(record, after, after_digest):
            self._progress["bound_reads"] += 1
            reasons.append("new_bound_read")
        if self._selector_is_new(record, proof, after_digest):
            self._progress["selector_results"] += 1
            reasons.append("new_selector_binding_result")
        version = decision_state_version
        if isinstance(version, bool) or not isinstance(version, (str, int)):
            version = None
        if version is not None and version != self._decision_state_version:
            self._progress["work_versions"] += 1
            reasons.append("new_work_version")
            self._decision_state_version = version
            self.work["version"] = version
        meaningful = bool(reasons)
        if meaningful:
            self._progress["meaningful"] += 1
            self._progress["no_progress_cycles"] = 0
            self._cycle_identities.clear()
        else:
            self._progress["no_progress_cycles"] += 1
            self._cycle_identities[action_key] = action
            while len(self._cycle_identities) > _MAX_RESTRICTIONS:
                self._cycle_identities.pop(next(iter(self._cycle_identities)))
        if missing_info is not None:
            self.set_missing_information(missing_info)
        self._sequence += 1
        step_ref = _text(record.get("step_record_id") or record.get("receipt_id")
                         or record.get("step"), 180)
        event = {
            "schema": EVENT_SCHEMA, "enabled": True, "series_id": self.series_id,
            "sequence": self._sequence, "step_record_ref": step_ref,
            "action_identity": action.view(), "meaningful_progress": meaningful,
            "progress_reasons": reasons,
            "no_progress_cycles": self._progress["no_progress_cycles"],
        }
        event["event_ref"] = _digest({
            "series_id": self.series_id, "sequence": self._sequence,
            "step_record_ref": step_ref, "action_identity": action.view(),
        })
        self._last_event = event
        if phase is not None:
            self.transition(
                phase, reason=phase_reason,
                evidence_ref=phase_evidence_ref or step_ref,
            )
        return copy.deepcopy(event)

    def restricted_argument_identities(self) -> list[dict]:
        if (not self.enabled or not self.control_enabled
                or self._progress["no_progress_cycles"] < self.no_progress_threshold):
            return []
        return [{
            **action.view(), "binding": action.binding_digest,
            "identity_ref": _digest(action.view()),
            "reason": "bounded_no_progress_cycle",
            "threshold": self.no_progress_threshold, "scope": "next_decision",
        } for action in self._cycle_identities.values() if action.tool != 'done'][: _MAX_RESTRICTIONS]

    def _budget_view(self) -> dict:
        if self.parent_budget is None:
            return {"authority": "ParentMeter", "provided": False}
        snapshotter = getattr(self.parent_budget, "snapshot", None)
        if not callable(snapshotter) and callable(self.parent_budget):
            snapshotter = self.parent_budget
        if not callable(snapshotter):
            return {"authority": "ParentMeter", "provided": True,
                    "snapshot_status": "unavailable"}
        try:
            snapshot = snapshotter()
        except Exception:
            return {"authority": "ParentMeter", "provided": True,
                    "snapshot_status": "unavailable"}
        if not isinstance(snapshot, Mapping):
            return {"authority": "ParentMeter", "provided": True,
                    "snapshot_status": "invalid"}
        result = {"authority": "ParentMeter", "provided": True,
                  "snapshot_status": "ok"}
        if "parent_id" in snapshot:
            result["parent_id"] = _text(snapshot["parent_id"], 120)
        for key in ("limits", "used"):
            value = snapshot.get(key)
            if isinstance(value, Mapping):
                result[key] = {
                    _text(name, 40): item for name, item in value.items()
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                }
        if isinstance(snapshot.get("enforced"), bool):
            result["enforced"] = snapshot["enforced"]
        return result

    def public(self) -> FrameSeriesView:
        restrictions = self.restricted_argument_identities()
        return {
            "schema": SCHEMA, "enabled": self.enabled,
            "control_enabled": self.control_enabled, "series_id": self.series_id,
            "objective_revision": self.objective_revision,
            "scope": copy.deepcopy(self.scope), "plan": copy.deepcopy(self.plan),
            "work": copy.deepcopy(self.work),
            "phase": {
                "id": self._phase, "sequence": self._phase_sequence,
                "latest_transition_ref": (
                    self._transitions[-1]["transition_ref"] if self._transitions else ""
                ),
            },
            "progress": {
                **self._progress,
                "last_event_ref": self._last_event.get("event_ref") if self._last_event else "",
                "last_meaningful_progress": (
                    self._last_event["meaningful_progress"] if self._last_event else False
                ),
                "work_state_version": self._decision_state_version,
            },
            "binding": copy.deepcopy(self._binding_view),
            "proof": {
                "status": self._proof_status,
                **(copy.deepcopy(self._proof) if self._proof is not None else {}),
            },
            "missing_information": copy.deepcopy(self._missing),
            "phase_transitions": copy.deepcopy(list(self._transitions)),
            "restrictions": restrictions,
            "budget": self._budget_view(),
        }

    view = public
