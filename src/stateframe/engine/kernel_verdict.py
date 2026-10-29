"""The shared, initiator-neutral authority for required public tests.

This module deliberately knows nothing about the action which caused a test
to run.  A model ``run_tests`` receipt and an automatic verification receipt
are equivalent when they carry the same trusted subprocess evidence.

The evaluator accepts a snapshot/binding, the required selectors, and an
event or receipt sequence.  A file selector is covered only by a file-level
result (a result for ``path::node`` does not silently cover ``path``).  A
node selector is covered only by the exact node selector.  This conservative
rule makes suite substitution and zero-case collection unresolved rather than
green.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import os
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Callable

from kernel_obligation_assess import assessment_is_blocking


POLICY_VERSION = "kernel-verdict/1"
COMPLETION_CONTRACT_VERSION = "atomic-completion/1"
SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
UNRESOLVED = "UNRESOLVED"
CONFLICT = "CONFLICT"
NO_HOST_CHECKS = "NO_HOST_CHECKS"
_STATES = {SATISFIED, VIOLATED, UNRESOLVED, CONFLICT, NO_HOST_CHECKS}

# Completion outcomes are deliberately separate from test-verdict states.  A
# passing test is evidence used by the completion reducer; it is not itself a
# stop decision.
ACCEPTED = "ACCEPTED"
REFUSED = "REFUSED"
BASIS_CHANGED = "BASIS_CHANGED"
BLOCKED = "BLOCKED"
NEEDS_WORK = "NEEDS_WORK"
BOUNDED = "BOUNDED"
SEALED = "SEALED"
POST_GREEN_STALL = "POST_GREEN_STALL"
_TEST_KINDS = {
    "test_subprocess_completed",
    "test_subprocess_complete",
    "host_test_receipt",
    "auto_verify_completed",
    "test_receipt",
    "test",
    "run_tests",
    "pytest",
}


def _generator_fixes_enabled() -> bool:
    from kernel_check_generator_fixes import enabled
    return enabled()


def _canonical(value: object) -> bytes:
    """Make a stable, host-side identity for an assessment input."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _identity(value: object) -> str:
    """Preserve an already host-issued digest, otherwise digest the value."""
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    return _digest(value)


@dataclass(frozen=True)
class CompletionBasis:
    """The immutable identity against which one done candidate is assessed.

    Only digests cross the reducer boundary.  This prevents a caller from
    mutating a snapshot, binding, or receipt list after admission and making a
    previously accepted candidate mean something else.
    """

    snapshot_digest: str
    binding_digest: str
    contract_version: str
    evidence_digest: str
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("snapshot_digest", "binding_digest", "contract_version", "evidence_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"completion basis requires {name}")
        if self.snapshot_id is not None and not isinstance(self.snapshot_id, str):
            raise ValueError("completion basis snapshot_id must be a string or None")

    @classmethod
    def capture(cls, snapshot: object, binding: object, *,
                contract_version: str = COMPLETION_CONTRACT_VERSION,
                evidence: object = None) -> "CompletionBasis":
        snapshot_id = snapshot if isinstance(snapshot, str) else None
        return cls(snapshot_digest=_identity(snapshot), binding_digest=_identity(binding),
                   contract_version=str(contract_version), evidence_digest=_identity(evidence),
                   snapshot_id=snapshot_id)

    from_state = capture

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_digest": self.snapshot_digest,
            "binding_digest": self.binding_digest,
            "contract_version": self.contract_version,
            "evidence_digest": self.evidence_digest,
            "snapshot_id": self.snapshot_id,
        }

    def diff(self, other: "CompletionBasis") -> dict[str, dict[str, object]]:
        if not isinstance(other, CompletionBasis):
            raise TypeError("basis diff requires another CompletionBasis")
        before, after = self.to_dict(), other.to_dict()
        return {key: {"before": before[key], "after": after[key]}
                for key in before if before[key] != after[key]}


def _menu_names(menu: object) -> tuple[str, ...]:
    if menu is None:
        return ()
    if isinstance(menu, Mapping):
        menu = menu.values()
    if isinstance(menu, (str, bytes)):
        menu = (menu,)
    if not isinstance(menu, Iterable):
        return ()
    names: list[str] = []
    for item in menu:
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
        else:
            name = getattr(item, "name", None)
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name)
    return tuple(names)


def _action_tool(action: object) -> str | None:
    if not isinstance(action, Mapping):
        return None
    value = action.get("tool", action.get("name"))
    return value if isinstance(value, str) and value.strip() else None


def _copy_action(action: object) -> dict[str, object] | None:
    if not isinstance(action, Mapping):
        return None
    tool = _action_tool(action)
    if tool is None:
        return None
    result: dict[str, object] = {"tool": tool}
    arguments = action.get("arguments")
    if isinstance(arguments, Mapping):
        result["arguments"] = json.loads(_canonical(dict(arguments)).decode())
    else:
        # A small protocol for callers which supply selectors/path directly.
        extras = {key: value for key, value in action.items()
                  if key not in {"tool", "name", "arguments"}}
        if extras:
            result["arguments"] = json.loads(_canonical(extras).decode())
    for key in ("reason", "why", "selector_scope"):
        if key in action:
            result[key] = str(action[key])
    return result


def repair_action_on_menu(action: object, menu: object) -> bool:
    """Return whether a proposed typed repair is currently executable."""
    tool = _action_tool(action)
    return tool is not None and tool in _menu_names(menu)


def _questions_nonveto_enabled() -> bool:
    """The question lane is opt-in; the legacy completion path stays unchanged."""
    return os.environ.get("STATEFRAME_QUESTIONS", "off").strip().lower() == "on"


def _run_tests_can_change(reason_codes: Sequence[str]) -> bool:
    """Limit mechanical-test repairs to gaps which tests can actually change."""
    codes = " ".join(str(code) for code in reason_codes).upper()
    return any(token in codes for token in (
        "TEST", "EVIDENCE", "STALE", "RED", "MISSING", "CONFLICT", "VERIFY",
        "ORACLE", "EFFECT",
    ))


def repair_action_for(reason_codes: Sequence[str] = (), *,
                      menu: object = (), selectors: Sequence[str] = (),
                      path: str | None = None,
                      work_ids: Sequence[str] = ()) -> dict[str, object] | None:
    """Choose one concrete repair which is actually present on ``menu``.

    The reducer never emits a command that the current frame cannot execute.
    Exact selectors and paths are retained in the action arguments so a
    refusal cannot send the worker back to a vague "verify something" loop.
    """
    names = _menu_names(menu)
    codes = " ".join(str(code) for code in reason_codes).upper()
    if "CONTINUATION_BLOCKED" in codes:
        # A host effect with no safe, schema-valid repair must terminate as a
        # typed BLOCKED outcome.  Do not synthesize a generic public-test retry
        # merely because selectors happen to be present in the root contract.
        return None
    clean_selectors = tuple(dict.fromkeys(str(item) for item in selectors if str(item).strip()))
    if (clean_selectors and "run_tests" in names
            and (not codes or any(token in codes for token in (
                "TEST", "EVIDENCE", "STALE", "RED", "MISSING", "CONFLICT", "VERIFY")))):
        return {"tool": "run_tests", "arguments": {"target": " ".join(clean_selectors),
                                                        "selectors": list(clean_selectors)},
                "reason": "refresh the exact required selectors"}
    if path and "read" in names and any(token in codes for token in ("SOURCE", "READ", "STALE")):
        return {"tool": "read", "arguments": {"path": str(path)},
                "reason": "read the exact current source file"}
    if _questions_nonveto_enabled() and not _run_tests_can_change(reason_codes):
        # There is no generic edit/work/probe payload that is safe to invent at
        # this boundary.  Returning None lets the reducer emit typed BLOCKED
        # rather than a REFUSED loop whose run_tests action cannot resolve the
        # non-test finding.
        return None
    # There is no generic, schema-valid "close these ids" payload for either
    # work-record tool: the work lane requires typed operations and the legacy
    # work_update tool requires its full current-version record.  Leave the
    # NEEDS_WORK outcome typed rather than inventing an action that the menu
    # cannot execute safely.
    if "run_tests" in names and clean_selectors:
        return {"tool": "run_tests", "arguments": {"target": " ".join(clean_selectors),
                                                        "selectors": list(clean_selectors)},
                "reason": "run the exact public checks"}
    if path and "read" in names:
        return {"tool": "read", "arguments": {"path": str(path)},
                "reason": "read the exact current source file"}
    return None


def work_record_enabled() -> bool:
    value = os.environ.get("STATEFRAME_WORK_RECORD", "off").lower()
    if value not in {"0", "1", "off", "on"}:
        raise ValueError("STATEFRAME_WORK_RECORD must be 0/1 or off/on")
    return value in {"1", "on"}


def _normalize_open_work(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if isinstance(value.get("work_record"), Mapping):
            return _normalize_open_work(value["work_record"])
        if "goals" in value or "steps" in value:
            return tuple(dict.fromkeys(
                (*_normalize_open_work(value.get("goals", ())),
                 *_normalize_open_work(value.get("steps", ())))))
        for key in ("ids", "open_ids", "open_work", "items", "goals"):
            if key in value:
                return _normalize_open_work(value[key])
        # DecisionState and the work-record lane commonly expose an id ->
        # record mapping.  Treat its values as records, retaining the key when
        # a record does not redundantly carry its id.
        rows = []
        for ident, item in value.items():
            if isinstance(item, Mapping):
                rows.append({"id": item.get("id", ident), **item})
            else:
                rows.append({"id": ident, "status": item})
        return _normalize_open_work(rows)
    if isinstance(value, (str, bytes)):
        return (value,) if str(value).strip() else ()
    if not isinstance(value, Iterable):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            ident = item.get("id", item.get("work_id"))
            status = str(item.get("status", "open")).lower()
            if status in {"completed", "complete", "closed", "superseded", "retired"}:
                continue
        else:
            ident = item
        if isinstance(ident, str) and ident.strip() and ident not in result:
            result.append(ident)
    return tuple(result)


def open_work_ids(state: object) -> tuple[str, ...]:
    """Small protocol bridge for the concurrent work-record lane.

    The hook is intentionally switch-gated.  With the switch absent or off,
    this is a no-op and existing decision-state behavior is unchanged.
    """
    if not work_record_enabled() or state is None:
        return ()
    hooks: list[Callable[..., object]] = []
    hook = getattr(state, "open_work", None)
    if callable(hook):
        hooks.append(hook)
    if isinstance(state, Mapping):
        candidate = state.get("open_work") or state.get("_open_work")
        if callable(candidate):
            hooks.append(candidate)
    # The work-record lane may land as a module after this reducer.  Absence is
    # deliberately a no-op, not an import error or a synthetic blocker.
    try:
        import importlib
        module = importlib.import_module("kernel_work_record")
        candidate = getattr(module, "open_work", None)
        if callable(candidate):
            hooks.append(candidate)
    except (ImportError, ModuleNotFoundError):
        pass
    for candidate in hooks:
        try:
            try:
                value = candidate(state)
            except TypeError:
                value = candidate()
        except Exception:  # a work hook must not turn done into a crash loop
            continue
        ids = _normalize_open_work(value)
        if ids:
            return ids
    if isinstance(state, Mapping):
        return _normalize_open_work(state.get(
            "work_items", state.get("open_work", state.get("items", ()))))
    items = getattr(state, "items", None)
    return _normalize_open_work(items) if items is not None else ()


@dataclass(frozen=True)
class CompletionAssessment:
    """One immutable, host-created assessment input for the done reducer."""

    basis: CompletionBasis
    admissible: bool
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    repair_action: Mapping[str, object] | None = None
    tool_menu: tuple[str, ...] = ()
    selectors: tuple[str, ...] = ()
    path: str | None = None
    open_work_ids: tuple[str, ...] = ()
    artifact_preserved: bool = True
    commit: Callable[[], object] | None = field(default=None, compare=False, repr=False)
    # Host-owned closure is opt-in and intentionally appended so every OFF
    # caller retains the pre-verdict constructor and serialized shape.
    host_closure: bool = False
    obligation_assessments: tuple[object, ...] = ()
    needs_work_payload: Mapping[str, object] | None = None
    repair_episode: Mapping[str, object] | None = None
    # An unanswered model question is an informational completion fact.  It is
    # retained in the reducer event, but it is never itself a blocker.
    unresolved_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.basis, CompletionBasis):
            raise TypeError("completion assessment requires a CompletionBasis")
        if type(self.admissible) is not bool:
            raise TypeError("completion assessment admissible must be boolean")
        object.__setattr__(self, "reason_codes",
                           tuple(dict.fromkeys(str(code) for code in self.reason_codes if str(code))))
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "tool_menu", _menu_names(self.tool_menu))
        object.__setattr__(self, "selectors",
                           tuple(dict.fromkeys(str(item) for item in self.selectors
                                                if str(item).strip())))
        if self.path is not None:
            object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "open_work_ids",
                           tuple(dict.fromkeys(str(item) for item in self.open_work_ids if str(item).strip())))
        action = _copy_action(self.repair_action)
        object.__setattr__(self, "repair_action", action)
        object.__setattr__(self, "obligation_assessments", tuple(self.obligation_assessments or ()))
        if self.needs_work_payload is not None:
            object.__setattr__(self, "needs_work_payload",
                               json.loads(_canonical(dict(self.needs_work_payload)).decode()))
        if self.repair_episode is not None:
            object.__setattr__(self, "repair_episode",
                               json.loads(_canonical(dict(self.repair_episode)).decode()))
        object.__setattr__(self, "unresolved_questions", tuple(dict.fromkeys(
            str(item) for item in (self.unresolved_questions or ()) if str(item).strip())))


@dataclass(frozen=True)
class CompletionDecision(Mapping[str, object]):
    """The sole typed result of a completion admission attempt."""

    outcome: str
    basis: CompletionBasis
    accepted: bool = False
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    repair_action: Mapping[str, object] | None = None
    basis_diff: Mapping[str, object] = field(default_factory=dict)
    needs_work_ids: tuple[str, ...] = ()
    artifact_preserved: bool = True
    sealed: bool = False
    needs_work_payload: Mapping[str, object] | None = None
    repair_episode: Mapping[str, object] | None = None
    # Host-only evidence is carried into the reducer event so UNKNOWN
    # obligations remain visible in the journal even when they do not block.
    obligation_assessments: tuple[object, ...] = ()
    # Same shape as CompletionAssessment; omitted from legacy serialized
    # decisions when empty so STATEFRAME_QUESTIONS=off remains byte-identical.
    unresolved_questions: tuple[str, ...] = ()
    # The done-specific typed authority is absent on legacy decisions so switch-off
    # serialization remains byte-identical.
    done_reason: Mapping[str, object] | None = None
    done_reason_line: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {ACCEPTED, REFUSED, BASIS_CHANGED, BLOCKED,
                                NEEDS_WORK, BOUNDED, SEALED, POST_GREEN_STALL}:
            raise ValueError(f"unknown completion outcome: {self.outcome!r}")
        object.__setattr__(self, "reason_codes",
                           tuple(dict.fromkeys(str(code) for code in self.reason_codes if str(code))))
        object.__setattr__(self, "needs_work_ids", tuple(self.needs_work_ids))
        action = _copy_action(self.repair_action)
        object.__setattr__(self, "repair_action", action)
        object.__setattr__(self, "basis_diff", dict(self.basis_diff))
        if self.needs_work_payload is not None:
            object.__setattr__(self, "needs_work_payload",
                               json.loads(_canonical(dict(self.needs_work_payload)).decode()))
        if self.repair_episode is not None:
            object.__setattr__(self, "repair_episode",
                               json.loads(_canonical(dict(self.repair_episode)).decode()))
        object.__setattr__(self, "obligation_assessments",
                           tuple(self.obligation_assessments or ()))
        object.__setattr__(self, "unresolved_questions", tuple(dict.fromkeys(
            str(item) for item in (self.unresolved_questions or ()) if str(item).strip())))
        if self.done_reason is not None:
            object.__setattr__(self, "done_reason",
                               json.loads(_canonical(dict(self.done_reason)).decode()))
        if self.done_reason_line is not None:
            object.__setattr__(self, "done_reason_line", str(self.done_reason_line)[:320])

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @property
    def status(self) -> str:
        return self.outcome

    @property
    def message(self) -> str:
        return self.detail

    @property
    def needs_work(self) -> Mapping[str, object] | None:
        return self.needs_work_payload

    def to_dict(self) -> dict[str, object]:
        result = {
            "outcome": self.outcome,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "repair_action": _copy_action(self.repair_action),
            "basis": self.basis.to_dict(),
            "basis_digest": self.basis.digest,
            "basis_diff": dict(self.basis_diff),
            "needs_work_ids": list(self.needs_work_ids),
            "artifact_preserved": self.artifact_preserved,
            "sealed": self.sealed,
        }
        # These keys are deliberately absent from OFF decisions.  That is the
        # byte-shape compatibility boundary for the opt-in host closure lane.
        if self.needs_work_payload is not None:
            result["needs_work"] = json.loads(_canonical(dict(self.needs_work_payload)).decode())
        if self.repair_episode is not None:
            result["repair_episode"] = json.loads(_canonical(dict(self.repair_episode)).decode())
        if self.obligation_assessments:
            result["obligation_assessments"] = [
                (item.to_dict() if hasattr(item, "to_dict") else
                 json.loads(_canonical(item).decode()))
                for item in self.obligation_assessments
            ]
        if self.unresolved_questions:
            result["unresolved_questions"] = list(self.unresolved_questions)
        if self.done_reason is not None:
            result["done_reason"] = json.loads(_canonical(dict(self.done_reason)).decode())
        if self.done_reason_line is not None:
            result["done_reason_line"] = self.done_reason_line
        return result


class CompletionReducer:
    """Single reducer for done admission and post-admission sealing."""

    def __init__(self) -> None:
        self._accepted_basis: CompletionBasis | None = None
        self._last_basis: CompletionBasis | None = None
        self._sealed = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def accepted_basis(self) -> CompletionBasis | None:
        return self._accepted_basis

    def decide(self, assessment: CompletionAssessment) -> CompletionDecision:
        if not isinstance(assessment, CompletionAssessment):
            raise TypeError("CompletionReducer.decide requires CompletionAssessment")
        basis = assessment.basis
        self._last_basis = basis

        # Q25 is an informational model commitment.  Under the opt-in
        # questions lane, strip only that reason before the sole reducer sees
        # it.  Other reasons remain authoritative, and host-obligation
        # assessments are evaluated below exactly as before.  Keeping this
        # normalization gated preserves the legacy OFF behavior for callers
        # which still construct a QUESTION_CLAUSE assessment directly.
        if _questions_nonveto_enabled() and "QUESTION_CLAUSE" in assessment.reason_codes:
            remaining = tuple(code for code in assessment.reason_codes
                              if code != "QUESTION_CLAUSE")
            assessment = dataclass_replace(
                assessment,
                admissible=assessment.admissible or not remaining,
                reason_codes=remaining,
                detail=assessment.detail if remaining else "",
            )

        # Once admitted, no later gate gets a veto.  A repeated identical done
        # is accepted without re-running any advisory or policy organ.  A
        # changed basis is explicit and carries a machine-readable diff.
        if self._sealed:
            if basis == self._accepted_basis:
                return CompletionDecision(
                    ACCEPTED, basis, accepted=True,
                    reason_codes=("ADMITTED_BASIS_UNCHANGED",),
                    detail="done accepted; the admitted completion basis is unchanged",
                    artifact_preserved=assessment.artifact_preserved, sealed=True,
                    unresolved_questions=assessment.unresolved_questions)
            return CompletionDecision(
                BASIS_CHANGED, basis, reason_codes=("BASIS_CHANGED",),
                detail="done basis changed after completion admission",
                basis_diff=self._accepted_basis.diff(basis) if self._accepted_basis else {},
                repair_action=assessment.repair_action,
                artifact_preserved=assessment.artifact_preserved, sealed=True,
                unresolved_questions=assessment.unresolved_questions)

        host_assessments = (tuple(assessment.obligation_assessments)
                            if assessment.host_closure else ())
        if host_assessments:
            payload_audit = (assessment.needs_work_payload.get("repair_packet_audit")
                             if isinstance(assessment.needs_work_payload, Mapping) else None)
            packet_unrenderable = (isinstance(payload_audit, Mapping)
                                   and payload_audit.get("valid") is False)
            episode_budget_exhausted = (isinstance(assessment.repair_episode, Mapping)
                                        and assessment.repair_episode.get("budget_exhausted") is True)
            blocking = tuple(item for item in host_assessments
                             if assessment_is_blocking(item)
                             and not packet_unrenderable
                             and not episode_budget_exhausted)
            if blocking:
                payload = assessment.needs_work_payload
                action = (payload.get("next_action")
                          if isinstance(payload, Mapping) else assessment.repair_action)
                if action is None or not repair_action_on_menu(action, assessment.tool_menu):
                    focus = (payload.get("focus_obligation")
                             if isinstance(payload, Mapping) else None)
                    ids = tuple(str(item.get("obligation_id")) for item in blocking
                                 if isinstance(item, Mapping) and item.get("obligation_id"))
                    if not ids:
                        ids = tuple(str(getattr(item, "obligation_id", "")) for item in blocking
                                    if getattr(item, "obligation_id", ""))
                    return CompletionDecision(
                        BLOCKED, basis,
                        reason_codes=("HOST_OBLIGATION", "NO_FEASIBLE_REPAIR_ACTION"),
                        detail=(f"host obligation remains: {focus}" if focus else
                                "host obligations remain, but no current repair action is executable"),
                        needs_work_payload=payload, repair_episode=assessment.repair_episode,
                        needs_work_ids=ids, artifact_preserved=assessment.artifact_preserved,
                        obligation_assessments=host_assessments,
                        unresolved_questions=assessment.unresolved_questions)
                focus = payload.get("focus_obligation") if isinstance(payload, Mapping) else None
                ids = (str(focus),) if focus else tuple(
                    str(getattr(item, "obligation_id", "")) for item in blocking
                    if getattr(item, "obligation_id", ""))
                reason = payload.get("reason") if isinstance(payload, Mapping) else None
                reason_codes = ["NEEDS_WORK", "HOST_OBLIGATION"]
                if reason:
                    reason_codes.append(str(reason))
                return CompletionDecision(
                    NEEDS_WORK, basis, reason_codes=tuple(dict.fromkeys(reason_codes)),
                    detail=(f"host obligation remains: {focus}" if focus else
                            "host obligation remains"),
                    repair_action=action, needs_work_payload=payload,
                    repair_episode=assessment.repair_episode, needs_work_ids=ids,
                    artifact_preserved=assessment.artifact_preserved,
                    obligation_assessments=host_assessments,
                    unresolved_questions=assessment.unresolved_questions)

        # Binding uncertainty is visible host evidence, but it is not a
        # second completion veto.  If an upstream caller labelled the same
        # uncertainty as its only reason for refusal, continue through the
        # ordinary admissibility path as if that reason were absent so a
        # missing repair action cannot manufacture terminal BLOCKED.
        # Only binding-uncertainty reasons may be dropped; a red public test
        # (or any other substantive reason) keeps the assessment refused.
        _uncertainty_only = not (set(assessment.reason_codes) - ({
            "HOST_OBLIGATION", "NO_FEASIBLE_REPAIR_ACTION", "NO_VERIFIER",
            "AMBIGUOUS_CONTRACT", "UNBOUND_ENTRYPOINT", "UNBOUND_INVOCATION",
            "MISSING_CHECKS", "STALE", "NOT_EXECUTED", "HARNESS_ERROR",
            "SETUP_ERROR", "SKIPPED", "CONFLICTING_RECEIPTS",
            "REPAIR_PACKET_INVALID", "REPAIR_BUDGET_EXHAUSTED",
        } | ({"ADVISORY_ONLY"} if _generator_fixes_enabled() else set())))
        if (host_assessments and (packet_unrenderable or episode_budget_exhausted)
                and _uncertainty_only):
            assessment = dataclass_replace(assessment, admissible=True,
                                           reason_codes=(),
                                           detail=("repair packet is advisory only"
                                                   if packet_unrenderable else
                                                   "repair budget exhausted; obligation is unknown"))
        elif (host_assessments and not any(assessment_is_blocking(item)
                                           for item in host_assessments)
                and _uncertainty_only):
            assessment = dataclass_replace(assessment, admissible=True,
                                           reason_codes=(), detail="")

        if assessment.open_work_ids and not assessment.host_closure:
            action = assessment.repair_action
            if action is None:
                action = repair_action_for(("OPEN_WORK",), menu=assessment.tool_menu,
                                           work_ids=assessment.open_work_ids,
                                           selectors=assessment.selectors,
                                           path=assessment.path)
            return CompletionDecision(
                NEEDS_WORK, basis, reason_codes=("NEEDS_WORK",),
                detail=("open work remains: " + ", ".join(assessment.open_work_ids)),
                repair_action=action, needs_work_ids=assessment.open_work_ids,
                artifact_preserved=assessment.artifact_preserved,
                obligation_assessments=host_assessments,
                unresolved_questions=assessment.unresolved_questions)

        if not assessment.admissible:
            action = assessment.repair_action
            if (action is not None and _questions_nonveto_enabled()
                    and _action_tool(action) == "run_tests"
                    and not _run_tests_can_change(assessment.reason_codes)):
                action = None
            if action is None:
                action = repair_action_for(assessment.reason_codes,
                                           menu=assessment.tool_menu,
                                           selectors=assessment.selectors,
                                           path=assessment.path)
            if action is None or not repair_action_on_menu(action, assessment.tool_menu):
                return CompletionDecision(
                    BLOCKED, basis,
                    reason_codes=tuple(assessment.reason_codes) + ("NO_FEASIBLE_REPAIR_ACTION",),
                    detail=assessment.detail or "completion is not admissible",
                    needs_work_ids=assessment.open_work_ids,
                    artifact_preserved=assessment.artifact_preserved,
                    obligation_assessments=host_assessments,
                    unresolved_questions=assessment.unresolved_questions)
            return CompletionDecision(
                REFUSED, basis, reason_codes=assessment.reason_codes,
                detail=assessment.detail or "completion is not admissible",
                repair_action=action, artifact_preserved=assessment.artifact_preserved,
                obligation_assessments=host_assessments,
                unresolved_questions=assessment.unresolved_questions)

        if assessment.commit is not None:
            try:
                committed = assessment.commit()
                if committed is False:
                    raise RuntimeError("completion commit returned false")
            except Exception as error:  # commit is part of admission, never a late veto
                action = assessment.repair_action
                if action is None:
                    action = repair_action_for(("COMMIT",), menu=assessment.tool_menu,
                                               selectors=assessment.selectors,
                                               path=assessment.path)
                return CompletionDecision(
                    BLOCKED, basis, reason_codes=("COMMIT_FAILED",),
                    detail="completion commit could not be sealed: "
                    + f"{type(error).__name__}: {error}", repair_action=action,
                    artifact_preserved=assessment.artifact_preserved,
                    obligation_assessments=host_assessments,
                    unresolved_questions=assessment.unresolved_questions)

        self._accepted_basis = basis
        self._sealed = True
        accepted_reasons = ["ADMITTED"]
        if isinstance(assessment.needs_work_payload, Mapping):
            audit = assessment.needs_work_payload.get("repair_packet_audit")
            if isinstance(audit, Mapping) and audit.get("valid") is False:
                accepted_reasons.append("REPAIR_PACKET_INVALID")
        if isinstance(assessment.repair_episode, Mapping) \
                and assessment.repair_episode.get("budget_exhausted") is True:
            accepted_reasons.append("REPAIR_BUDGET_EXHAUSTED")
        stall = POST_GREEN_STALL in assessment.reason_codes
        if stall:
            accepted_reasons = [POST_GREEN_STALL, *accepted_reasons]
        return CompletionDecision(
            POST_GREEN_STALL if stall else ACCEPTED, basis, accepted=True,
            reason_codes=tuple(dict.fromkeys(accepted_reasons)),
            detail=("post-green stall accepted; current product preserved"
                    if stall else "done accepted and completion sealed"),
            artifact_preserved=assessment.artifact_preserved, sealed=True,
            obligation_assessments=host_assessments,
            unresolved_questions=assessment.unresolved_questions)

    def admit_mutation(self, basis: CompletionBasis) -> CompletionDecision | None:
        """Reject a late edit after acceptance, before it can change the artifact."""
        if not self._sealed:
            return None
        if basis == self._accepted_basis:
            return CompletionDecision(
                SEALED, basis, reason_codes=("COMPLETION_SEALED",),
                detail="the completion is sealed; late edits are rejected",
                artifact_preserved=True, sealed=True)
        return CompletionDecision(
            BASIS_CHANGED, basis, reason_codes=("BASIS_CHANGED", "COMPLETION_SEALED"),
            detail="late edit would change the sealed completion basis",
            basis_diff=self._accepted_basis.diff(basis) if self._accepted_basis else {},
            artifact_preserved=True, sealed=True)

    def bounded(self, basis: CompletionBasis, *, detail: str,
                artifact_preserved: bool = True) -> CompletionDecision:
        return CompletionDecision(BOUNDED, basis, reason_codes=("CLOSURE_UNAFFORDABLE",),
                                  detail=detail, artifact_preserved=artifact_preserved)


DoneReducer = CompletionReducer


def reduce_done(*, admissible: bool, basis: CompletionBasis | None = None,
                snapshot: object = None, binding: object = None,
                contract_version: str = COMPLETION_CONTRACT_VERSION,
                evidence: object = None, reason_codes: Sequence[str] = (),
                detail: str = "", repair_action: Mapping[str, object] | None = None,
                tool_menu: object = (), selectors: Sequence[str] = (),
                path: str | None = None, open_work: object = (),
                reducer: CompletionReducer | None = None,
                commit: Callable[[], object] | None = None,
                unresolved_questions: Sequence[object] = ()) -> CompletionDecision:
    """Convenience entry point used by small host boundaries and fixtures."""
    if basis is None:
        basis = CompletionBasis.capture(snapshot, binding,
                                       contract_version=contract_version, evidence=evidence)
    state = reducer or CompletionReducer()
    assessment = CompletionAssessment(
        basis=basis, admissible=admissible, reason_codes=tuple(reason_codes), detail=detail,
        repair_action=repair_action, tool_menu=_menu_names(tool_menu),
        selectors=tuple(selectors), path=path,
        open_work_ids=open_work_ids(open_work), commit=commit,
        unresolved_questions=tuple(str(item) for item in unresolved_questions
                                   if str(item).strip()))
    return state.decide(assessment)


@dataclass(frozen=True)
class TrustedTestReceipt:
    """Typed, executor-owned test evidence selected by the shared verdict.

    The receipt is deliberately richer than the model-facing verdict mapping.
    A continuation may use it to refresh its dependency state, but only after
    the same evaluator has established that it is current and passing.
    """

    receipt_id: str
    sequence: int | float
    selectors: tuple[str, ...]
    binding: Mapping[str, object]
    after_binding: Mapping[str, object]
    exit_code: int
    binding_stable: bool
    producer: str = "test_subprocess"
    kind: str = "test_subprocess_completed"

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_id, str) or not self.receipt_id.strip():
            raise ValueError("trusted test receipt requires an identity")
        if type(self.sequence) not in {int, float}:
            raise ValueError("trusted test receipt requires an ordered sequence")
        selectors = tuple(str(selector) for selector in self.selectors if str(selector).strip())
        if not selectors:
            raise ValueError("trusted test receipt requires selectors")
        if not isinstance(self.binding, Mapping) or not isinstance(self.after_binding, Mapping):
            raise ValueError("trusted test receipt requires bindings")
        if type(self.exit_code) is not int or self.exit_code != 0 or self.binding_stable is not True:
            raise ValueError("trusted test receipt is not stable passing evidence")
        if self.producer != "test_subprocess" or self.kind not in _TEST_KINDS:
            raise ValueError("trusted test receipt has an untrusted producer or kind")
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "binding", dict(self.binding))
        object.__setattr__(self, "after_binding", dict(self.after_binding))

    @classmethod
    def from_event(cls, row: Mapping[str, object], position: int) -> "TrustedTestReceipt":
        """Build a receipt only after ``evaluate_required_tests`` has vetted ``row``."""
        sequence = row.get("sequence")
        if type(sequence) not in {int, float}:
            sequence = position
        return cls(
            receipt_id=_event_ref(row, position),
            sequence=sequence,
            selectors=tuple(_selectors(row)),
            binding=dict(_mapping(row.get("binding")) or {}),
            after_binding=dict(_mapping(row.get("after_binding")) or {}),
            exit_code=row.get("exit_code"),
            binding_stable=row.get("binding_stable"),
            producer=str(row.get("producer", "")),
            kind=str(row.get("kind") or "test_subprocess_completed"),
        )


@dataclass(frozen=True)
class Verdict(Mapping[str, object]):
    """A structured verdict which cannot be used as an implicit boolean."""

    state: str
    supporting_evidence: list[str]
    reason_codes: list[str]
    policy_version: str = POLICY_VERSION
    # These host-only fields are the typed authority shared by the progress
    # guard and continuation.  They intentionally stay out of ``as_dict`` so
    # model-facing projections cannot turn receipt locators into prose proof.
    binding: Mapping[str, object] | None = field(default=None, compare=False, repr=False)
    trusted_receipts: tuple[TrustedTestReceipt, ...] = field(default_factory=tuple,
                                                               compare=False, repr=False)
    identity: Mapping[str, object] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise ValueError(f"unknown verdict state: {self.state!r}")
        object.__setattr__(self, "supporting_evidence", list(self.supporting_evidence))
        object.__setattr__(self, "reason_codes", list(dict.fromkeys(self.reason_codes)))
        if self.binding is not None:
            if not isinstance(self.binding, Mapping):
                raise ValueError("verdict binding must be a mapping")
            object.__setattr__(self, "binding", dict(self.binding))
        receipts = tuple(self.trusted_receipts)
        if any(not isinstance(receipt, TrustedTestReceipt) for receipt in receipts):
            raise ValueError("verdict trusted receipts must be typed test receipts")
        object.__setattr__(self, "trusted_receipts", receipts)
        if self.identity is not None:
            if not isinstance(self.identity, Mapping):
                raise ValueError("verdict identity must be a mapping")
            object.__setattr__(self, "identity", dict(self.identity))

    def __bool__(self) -> bool:
        raise TypeError("Verdict has no truth value; inspect verdict.state explicitly")

    def __getitem__(self, key: str) -> object:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    @property
    def current_binding(self) -> Mapping[str, object] | None:
        """The exact binding used by this verdict, if one was available."""
        return self.binding

    def as_dict(self, *, identity: object = None) -> dict[str, object]:
        result = {
            "state": self.state,
            "supporting_evidence": list(self.supporting_evidence),
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
        }
        attached = identity if identity is not None else self.identity
        if attached is not None:
            from kernel_verdict_identity import stamp_record
            stamped = stamp_record(result, attached)
            if isinstance(stamped, Mapping):
                return dict(stamped)
        return result

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        return self.as_dict(identity=identity)

    @classmethod
    def from_mapping(cls, value: object) -> "Verdict | None":
        """Rehydrate a previously evaluated, host-owned verdict projection.

        Rich projections may omit the raw receipt stream from the model-facing
        frame.  Consumers can still use the exact authority result that was
        computed while the stream was available; arbitrary mappings are not
        accepted as evidence.
        """
        if not isinstance(value, Mapping):
            return None
        state = value.get("state")
        policy = value.get("policy_version")
        evidence = value.get("supporting_evidence", ())
        reasons = value.get("reason_codes", ())
        if (not isinstance(state, str) or state not in _STATES
                or policy != POLICY_VERSION
                or not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes))
                or not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes))):
            return None
        identity = (value.get("identity") if isinstance(value.get("identity"), Mapping)
                    else {key: value[key] for key in (
                        "run_id", "code_hash", "release_id", "harness_version")
                    if key in value} or None)
        return cls(state, [str(item) for item in evidence],
                   [str(item) for item in reasons], policy, identity=identity)


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _current_binding(snapshot: object) -> Mapping[str, object] | None:
    """Extract the current binding without treating prose as evidence."""
    if not isinstance(snapshot, Mapping):
        return None
    for key in ("current_binding", "snapshot_binding"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            return value
    value = snapshot.get("binding")
    if isinstance(value, Mapping):
        return value
    nested = snapshot.get("snapshot")
    if isinstance(nested, Mapping):
        return _current_binding(nested)
    # A binding itself is also a valid snapshot.  This fallback is useful for
    # callers whose public API already names the value ``binding``.
    binding_keys = {
        "product_revision", "current_revision", "revision", "files_revision",
        "harness_revision", "environment_revision", "source_revision",
        "product_files", "harness_files", "test_harness_revision", "artifact", "environment",
    }
    if binding_keys.intersection(snapshot):
        return snapshot
    return None


def _selector_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _required_selectors(value: object) -> list[str]:
    return list(dict.fromkeys(_selector_values(value)))


def _event_rows(events: object) -> list[tuple[int, Mapping[str, object]]]:
    if isinstance(events, TrustedTestReceipt):
        source = [events]
    elif isinstance(events, Mapping):
        # Accept both a receipt map keyed by reference and one receipt passed
        # directly.  The latter is useful at small host boundaries and avoids
        # silently treating the fields of a single event as event values.
        event_keys = {"producer", "kind", "exit_code", "selectors", "selector",
                      "binding", "after_binding"}
        source = [events] if event_keys.intersection(events) else list(events.values())
    elif isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
        source = list(events)
    elif isinstance(events, Iterable) and not isinstance(events, (str, bytes)):
        source = list(events)
    else:
        source = []
    rows: list[tuple[int, Mapping[str, object]]] = []
    for position, item in enumerate(source):
        if isinstance(item, TrustedTestReceipt):
            item = {
                "receipt_id": item.receipt_id,
                "sequence": item.sequence,
                "selectors": list(item.selectors),
                "binding": item.binding,
                "after_binding": item.after_binding,
                "exit_code": item.exit_code,
                "binding_stable": item.binding_stable,
                "producer": item.producer,
                "kind": item.kind,
            }
        if not isinstance(item, Mapping):
            continue
        # Host APIs sometimes wrap the event in ``receipt``.  Preserve the
        # outer event id/sequence while reading the subprocess fields inside.
        nested = item.get("receipt")
        row: Mapping[str, object] = nested if isinstance(nested, Mapping) else item
        if nested is not None and isinstance(nested, Mapping):
            merged = dict(nested)
            for key in ("event_id", "sequence", "timestamp", "kind"):
                if key in item and key not in merged:
                    merged[key] = item[key]
            row = merged
        rows.append((position, row))
    return rows


def _event_ref(row: Mapping[str, object], position: int) -> str:
    for key in ("event_id", "receipt_id", "id", "ref"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"event@{position}"


def _selectors(row: Mapping[str, object]) -> list[str]:
    for key in ("selectors", "required_selectors"):
        found = _selector_values(row.get(key))
        if found:
            return found
    for key in ("selector", "target", "suite", "test"):
        found = _selector_values(row.get(key))
        if found:
            return found
    return []


def _covers(candidate: str, required: str) -> bool:
    # ``path::node`` is an explicit node selector.  File-level execution is
    # intentionally not inferred from a node-level receipt.
    required_node = "::" in required
    candidate_node = "::" in candidate
    if required_node:
        return candidate_node and candidate == required
    return not candidate_node and candidate == required


def _order_key(row: Mapping[str, object], position: int) -> tuple[int, int | float]:
    sequence = row.get("sequence")
    if type(sequence) is int:
        return (2, sequence)
    for key in ("event_sequence", "timestamp_ns", "timestamp"):
        value = row.get(key)
        if type(value) in {int, float}:
            return (1, value)
    return (0, position)


def _newest(rows: list[tuple[int, Mapping[str, object]]]) -> list[tuple[int, Mapping[str, object]]]:
    if not rows:
        return []
    newest_key = max(_order_key(row, position) for position, row in rows)
    return [(position, row) for position, row in rows
            if _order_key(row, position) == newest_key]


def _binding_is_current(row: Mapping[str, object], current: Mapping[str, object]) -> bool:
    binding = _mapping(row.get("binding"))
    after = _mapping(row.get("after_binding"))
    return (row.get("binding_stable") is True and binding == current and after == current)


def _binding_instability_reason(row: Mapping[str, object]) -> str:
    """Expose the host's typed instability code and changed paths to consumers."""
    detail = row.get("binding_stability")
    if not isinstance(detail, Mapping):
        detail = row.get("binding_instability_reason")
    if not isinstance(detail, Mapping):
        return "STALE_OR_UNSTABLE_BINDING"
    code = str(detail.get("code") or "STALE_OR_UNSTABLE_BINDING")
    paths = detail.get("changed_paths")
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
        normalized = tuple(sorted({str(path) for path in paths if str(path).strip()}))
    else:
        normalized = ()
    return "BINDING_UNSTABLE:" + code + (":" + "|".join(normalized) if normalized else "")


def _exit_state(row: Mapping[str, object]) -> tuple[str, str]:
    from kernel_step_contract import environment_check_reason
    if environment_check_reason(row) is not None:
        return UNRESOLVED, "ENVIRONMENT_BLOCKED"
    for key in ("zero_cases", "no_tests_collected", "empty_collection"):
        if row.get(key) is True:
            return UNRESOLVED, "NO_TESTS_COLLECTED"
    for key in ("collected", "tests_collected", "test_count", "case_count", "num_tests"):
        if type(row.get(key)) is int and row[key] == 0:
            return UNRESOLVED, "NO_TESTS_COLLECTED"
    collection = row.get("collection")
    if isinstance(collection, Mapping):
        for key in ("collected", "tests", "cases", "count"):
            if type(collection.get(key)) is int and collection[key] == 0:
                return UNRESOLVED, "NO_TESTS_COLLECTED"
    result = str(row.get("result_class", row.get("result", row.get("status", "")))).upper()
    if result in {"NO_TESTS", "NO_TESTS_COLLECTED", "SELECTOR_ERROR", "NO_VERDICT"}:
        return UNRESOLVED, "NO_TESTS_COLLECTED" if result != "SELECTOR_ERROR" else "SELECTOR_ERROR"
    exit_code = row.get("exit_code")
    if type(exit_code) is not int:
        return UNRESOLVED, "INVALID_EXIT_CODE"
    if exit_code == 0:
        return SATISFIED, "PASS"
    if exit_code in {4, 5}:
        return UNRESOLVED, "NO_TESTS_COLLECTED" if exit_code == 5 else "SELECTOR_ERROR"
    return VIOLATED, "NONZERO_EXIT"


def _identity_projection(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    from kernel_verdict_identity import enabled as identity_enabled
    if not identity_enabled():
        return None
    current = value.current() if callable(getattr(value, "current", None)) else value
    if callable(getattr(current, "to_dict", None)):
        projected = current.to_dict()
        return dict(projected) if isinstance(projected, Mapping) else None
    return dict(current) if isinstance(current, Mapping) else None


def evaluate_required_tests(
    snapshot: object,
    required_selectors: object,
    events: object,
    *,
    identity: object = None,
) -> Verdict:
    """Evaluate the newest trusted, current result for every required selector.

    Only a ``test_subprocess`` producer is trusted.  Initiator, tool name,
    effect id, annotation, and prose fields are intentionally ignored.  Thus
    the same subprocess receipt has the same result regardless of whether a
    model action or automatic verification caused it.
    """
    verdict_identity = _identity_projection(identity)
    current = _current_binding(snapshot)
    required = _required_selectors(required_selectors)
    if not required:
        return Verdict(
            NO_HOST_CHECKS,
            [],
            ["NO_HOST_CHECKS"],
            binding=dict(current) if current is not None else None,
            identity=verdict_identity,
        )
    if current is None:
        return Verdict(UNRESOLVED, [], ["CURRENT_BINDING_UNAVAILABLE"], identity=verdict_identity)

    if identity is not None:
        # The legacy evaluator remains unchanged unless its caller explicitly
        # supplies the host identity provider.  In the identity arm a receipt
        # from the previous workspace tree is unresolved evidence, never a
        # pass that can discharge the current selector.
        try:
            from kernel_verdict_identity import (
                IdentityError, IdentityMismatch, MixedCodeHashError,
                MixedReleaseError, validate_batch,
            )
            validate_batch(events, current=identity)
        except MixedReleaseError:
            return Verdict(UNRESOLVED, [], ["MIXED_RELEASE_ID"], identity=verdict_identity)
        except (MixedCodeHashError, IdentityMismatch):
            return Verdict(UNRESOLVED, [], ["STALE_CODE_HASH"], identity=verdict_identity)
        except IdentityError:
            return Verdict(UNRESOLVED, [], ["MISSING_RUN_IDENTITY"], identity=verdict_identity)

    rows = _event_rows(events)
    trusted: list[tuple[int, Mapping[str, object]]] = []
    ignored_producer = False
    for position, row in rows:
        producer = row.get("producer")
        if producer != "test_subprocess":
            if _selectors(row):
                ignored_producer = True
            continue
        kind = row.get("kind")
        if kind is not None and str(kind) not in _TEST_KINDS:
            # The producer is authoritative, but arbitrary action receipts
            # are not subprocess completion evidence.
            continue
        trusted.append((position, row))

    reasons: list[str] = []
    supporting: list[str] = []
    winning_rows: list[tuple[int, Mapping[str, object]]] = []
    selector_states: list[str] = []
    for required_selector in required:
        covered: list[tuple[int, Mapping[str, object]]] = []
        for position, row in trusted:
            if not any(_covers(candidate, required_selector) for candidate in _selectors(row)):
                continue
            covered.append((position, row))
        if not covered:
            selector_states.append(UNRESOLVED)
            reasons.append("MISSING_SELECTOR:" + required_selector)
            continue
        # A newer observed result, including an unstable or stale one, revokes
        # an older pass. Otherwise a green receipt could survive an edit or a
        # test-time mutation merely because the newer observation was unusable.
        winners = _newest(covered)
        if any(not _binding_is_current(row, current) for _position, row in winners):
            selector_states.append(UNRESOLVED)
            reasons.append(_binding_instability_reason(next(
                row for _position, row in winners
                if not _binding_is_current(row, current))) + ":" + required_selector)
            supporting.extend(_event_ref(row, position) for position, row in winners)
            continue
        winner_states = [_exit_state(row) for _position, row in winners]
        for position, row in winners:
            supporting.append(_event_ref(row, position))
            winning_rows.append((position, row))
        distinct = {state for state, _reason in winner_states}
        if len(distinct) > 1:
            selector_states.append(CONFLICT)
            reasons.append("CONFLICTING_RESULTS:" + required_selector)
            continue
        state, reason = winner_states[0]
        selector_states.append(state)
        if reason != "PASS":
            reasons.append(reason + ":" + required_selector)

    if any(state == CONFLICT for state in selector_states):
        state = CONFLICT
    elif any(state == VIOLATED for state in selector_states):
        state = VIOLATED
    elif any(state == UNRESOLVED for state in selector_states):
        state = UNRESOLVED
    else:
        state = SATISFIED
        reasons.append("ALL_REQUIRED_TESTS_CURRENT")
    if ignored_producer and state != SATISFIED:
        reasons.append("UNTRUSTED_PRODUCER")
    trusted_receipts: list[TrustedTestReceipt] = []
    if state == SATISFIED:
        seen: set[str] = set()
        for position, row in winning_rows:
            receipt = TrustedTestReceipt.from_event(row, position)
            if receipt.receipt_id not in seen:
                seen.add(receipt.receipt_id)
                trusted_receipts.append(receipt)
    return Verdict(state, list(dict.fromkeys(supporting)), list(dict.fromkeys(reasons)),
                   binding=dict(current), trusted_receipts=tuple(trusted_receipts),
                   identity=verdict_identity)


__all__ = [
    "ACCEPTED", "BASIS_CHANGED", "BLOCKED", "BOUNDED", "COMPLETION_CONTRACT_VERSION",
    "CONFLICT", "CompletionAssessment", "CompletionBasis", "CompletionDecision",
    "CompletionReducer", "DoneReducer", "NEEDS_WORK", "POLICY_VERSION", "REFUSED",
    "NO_HOST_CHECKS", "SATISFIED", "SEALED", "UNRESOLVED", "VIOLATED",
    "TrustedTestReceipt", "Verdict",
    "open_work_ids", "reduce_done", "repair_action_for", "repair_action_on_menu",
    "work_record_enabled",
    "evaluate_required_tests",
]
