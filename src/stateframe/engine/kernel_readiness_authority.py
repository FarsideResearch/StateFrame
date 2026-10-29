"""kernel_readiness_authority.py — the ONE readiness authority.

Verifiers (green lock, green coherence, counterexample visibility, green-lock repair,
host checks) only contribute evidence. This module reduces that evidence into one
immutable ``ReadinessDecision`` snapshot per request and is the only place that may
decide:

* whether the run is READY (public checks green on the current revision AND no live
  admitted counterexample AND no explicit host blocker);
* which tools the model-visible menu must keep (repair tools while not ready) and
  whether done-only narrowing is permitted (ONLY when ready);
* whether certified submission is allowed and with what single, complete refusal text.
  The capability floor can retain ``done`` for an uncertified submission while
  this authority withholds certification.

Frame header/SUBMISSION/footer (``kernel_green_coherence.TerminalStatus``), the tool
menu (``StateFrameRuntime.tool_menu``), lock text, the done handler
(``StateFrameRuntime.completion_decision``) and the continuation header all read the
same snapshot. Switch: ``STATEFRAME_READINESS_AUTHORITY`` (off|on, default off). Off
is byte-identical: no consumer calls into this module unless the switch is on.
"""
from __future__ import annotations

import dataclasses

import hashlib
import json
import os
import copy
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from kernel_bounded_locate_code import OBSERVATION_TOOLS
from kernel_harden8 import record_failure

MODE_ENV = "STATEFRAME_READINESS_AUTHORITY"
SCHEMA = "stateframe-readiness-decision/1"
DECISION_EVENT = "readiness_decision"
MENU_EVENT = "readiness_menu_projected"
VIOLATION_EVENT = "kernel_error"
REFUSAL_CODE = "READINESS_COUNTEREXAMPLE_OPEN"
BLOCKED_BY_LABEL = "CERTIFICATION BLOCKED BY CHECKS"
# Repair tools that must stay reachable while a run is not ready.  ``run_tests`` is the
# re-verification path; edit/str_replace are the scoped-repair paths.
REPAIR_TOOLS = ("edit", "str_replace", "run_tests")
EDIT_TOOLS = frozenset({"edit", "str_replace"})
# Tools that only make sense when blocked by a live counterexample (challenge path).
CHALLENGE_TOOLS = frozenset({"dispute", "read_ticket"})


def _active_repair_tools() -> tuple[str, ...]:
    """Keep creation reachable only for the v21 artifact-aware arm."""
    if os.environ.get("STATEFRAME_TOKENLAB_NONPY_EDITS", "off").strip().lower() == "on":
        return (*REPAIR_TOOLS, "file_create")
    return REPAIR_TOOLS

PHASE_WORKING = "working"
PHASE_BLOCKED = "blocked_counterexample"
PHASE_READY = "ready"
PHASE_REPAIR_OPTIONAL = "repair_optional"
PHASE_NO_HOST_CHECKS = "no_host_checks"
NO_HOST_CHECKS = "NO_HOST_CHECKS"

_NON_REPAIRABLE_REASON_PARTS = (
    "UNBOUND", "SELF_CONTRADICT", "INVALID", "WRONG_BINDING", "NOT_ENTAILED",
    "REBOUND", "UNSATISFIABLE", "CONTRADICTS", "ADVISORY_ONLY",
)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def runtime_enabled(runtime: object | None) -> bool:
    if runtime is None:
        return False
    value = getattr(runtime, "stateframe_readiness_authority_mode", None)
    if value == "on":
        return True
    # The done-blocking arm composes with this authority so a runtime that
    # opts into the stricter policy cannot render an advisory menu.
    repair_mode = getattr(runtime, "stateframe_repair_blocks_done_mode", None)
    if repair_mode is not None:
        return repair_mode == "on"
    return os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"


def runtime_snapshot(runtime: object | None, frame: object | None = None):
    """Return the request snapshot shared by menu and action consumers."""
    if not runtime_enabled(runtime):
        return None
    if frame is None:
        cached = getattr(runtime, "_readiness_request_decision", None)
        if cached is not None:
            return cached
        cached = getattr(runtime, "_readiness_decision", None)
        if cached is not None:
            return cached
    method = getattr(runtime, "readiness_decision", None)
    if not callable(method):
        return None
    try:
        return method(frame)
    except (AttributeError, TypeError, ValueError):
        return None


def _tool_name(tool: object) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name", ""))
        return str(tool.get("name", ""))
    return ""


def _check_id(row: Mapping[str, object]) -> str:
    return str(row.get("check_id") or row.get("obligation_id") or row.get("id") or "")


def _row(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _field(item: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        value = item.get(name)
        if value not in (None, ""):
            return value
    return default


def _upper(value: object) -> str:
    if isinstance(value, Mapping):
        value = _field(value, "status", "value", "name", "reason", "type",
                       "basis", "source", default="")
    return str(value or "").strip().upper().replace("-", "_")


def _reason(row: Mapping[str, object]) -> str:
    value = _field(row, "demotion_reason", "classification", "reason", default="")
    if isinstance(value, Mapping):
        value = _field(value, "reason", "classification", "status", default="")
    return _upper(value)


def _has_mutation_site(row: Mapping[str, object]) -> bool:
    for name in ("mutation_site", "mutation_sites", "mutation_site_refs", "mutation_sites_refs"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if any(item not in (None, "", {}) for item in value):
                return True
    return bool(_field(row, "target_file", "target_path", "target_symbol", "symbol"))


def _literal_aligned(row: Mapping[str, object]) -> bool:
    """True only for host-produced ticket-literal alignment evidence (switch-gated producer)."""
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() != "on":
        return False
    evidence = row.get("alignment_evidence")
    return (bool(row.get("literal_grounded")) and isinstance(evidence, Mapping)
            and bool(evidence.get("matched_literals")))


def _repairable_advisory(row: Mapping[str, object], revision: object) -> bool:
    """Classify an advisory using only bounded, current, executable evidence."""
    if bool(row.get("counterexample_authority")):
        return False
    # r39: host literal alignment (kernel_requirement_alignment) is typed proof that the check
    # is entailed by a ticket sentence, so it outranks a NOT_ENTAILED demotion and stands in for
    # the ticket basis.  Every other gate (binding, execution, revision, site) still applies.
    aligned = _literal_aligned(row)
    reason = _reason(row)
    blocking_parts = tuple(part for part in _NON_REPAIRABLE_REASON_PARTS
                           if not (aligned and part == "NOT_ENTAILED"))
    if any(part in reason for part in blocking_parts):
        return False

    basis = _upper(_field(row, "oracle_basis", "ticket_basis", "provenance", default=""))
    if isinstance(row.get("provenance"), Mapping):
        provenance = row["provenance"]
        basis = _upper(_field(provenance, "oracle_basis", "basis", "source", default=basis))
    if basis not in {"TICKET", "TICKET_DERIVED", "TICKET_BOUND", "REQUIREMENT_TICKET"} and not aligned:
        return False

    binding = _field(row, "binding", "binding_status", "entrypoint_binding", default=None)
    if isinstance(binding, Mapping):
        binding = _field(binding, "status", "verdict", "binding", "value", default=None)
    if binding is not None and _upper(binding) != "BOUND":
        return False
    if binding is None and not _has_mutation_site(row):
        return False

    execution = _upper(_field(row, "execution", "execution_status", default=""))
    if execution != "EXECUTED":
        return False
    reproducible = _field(row, "reproducible", "reproducibility", default=None)
    if isinstance(reproducible, Mapping):
        reproducible = _field(reproducible, "value", "status", "reproducible", default=None)
    if reproducible is not None and reproducible is not True:
        if _upper(reproducible) not in {"TRUE", "YES", "REPRODUCIBLE"}:
            return False

    row_revision = _field(row, "revision", "current_revision", "product_revision", default=None)
    aliases = row.get("revision_aliases")
    alias_set = ({str(item) for item in aliases}
                 if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)) else set())
    if row_revision is not None:
        alias_set.add(str(row_revision))
    if revision is None or str(revision) not in alias_set:
        return False
    return _has_mutation_site(row)


def _annotate_advisory(item: object, revision: object) -> dict[str, object]:
    row = copy.deepcopy(_row(item))
    repairable = _repairable_advisory(row, revision)
    row["repairable"] = repairable
    row["demotion_reason"] = _reason(row) or "ADVISORY"
    return row


def _repair_target(row: Mapping[str, object]) -> str:
    target_file = str(_field(row, "target_file", "target_path", "file", default="") or "")
    target_symbol = str(_field(row, "target_symbol", "symbol", default="") or "")
    if target_file and target_symbol:
        return f"{target_file}:{target_symbol}"
    if target_file:
        return target_file
    if target_symbol:
        return target_symbol
    for key in ("mutation_site", "mutation_site_refs", "mutation_sites"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    path = _field(item, "path", "file", default="")
                    symbol = _field(item, "symbol", "name", default="")
                    if path and symbol:
                        return f"{path}:{symbol}"
                    if path:
                        return str(path)
                elif item not in (None, ""):
                    return str(item)
    return "the repair site"


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    """Immutable decision snapshot; every surface projects from this, nothing else."""

    state_version: int
    candidate_digest: str
    revision: object
    green: bool
    public_check_status: str
    blockers: tuple[Mapping[str, object], ...]
    advisory: tuple[Mapping[str, object], ...]
    host_blocking: tuple[str, ...]
    selectors: tuple[str, ...] = ()
    repair_action: Mapping[str, object] | None = None
    dispositions: Mapping[str, str] = field(default_factory=dict)
    repair_blocks_done_active: bool = False
    # One owner: the tools the greenlock authority's post-green work window admits at this state
    # (kernel_post_green_menu.window_dispositions).  Non-empty = readiness defers menu narrowing while
    # preserving the authority's allow/deny verdicts.  ``ready_to_submit`` is the menu projection;
    # ``host_submittable`` is the actual fail-closed host commit predicate.
    post_green_work: tuple[str, ...] = ()
    post_green_admitted_files: tuple[str, ...] = ()
    # STATEFRAME_POSTGREEN_NO_VETO: product edit/read dispositions are owned by
    # kernel_postgreen_no_veto after public green; this marker drives projection.
    post_green_no_veto: bool = False
    no_host_checks: bool = False
    done_authority: object | None = None

    @property
    def ready(self) -> bool:
        return self.green and not self.blockers and not self.host_blocking

    @property
    def repairable_advisories(self) -> tuple[Mapping[str, object], ...]:
        return tuple(item for item in self.advisory if bool(item.get("repairable")))

    @property
    def repairable_advisory_ids(self) -> tuple[str, ...]:
        return tuple(_check_id(item) for item in self.repairable_advisories
                     if _check_id(item))

    @property
    def ready_to_submit(self) -> bool:
        """Whether the menu may narrow to a done-only submission surface."""
        return self.done_allowed and not self.repairable_advisories and not self.post_green_work

    @property
    def host_submittable(self) -> bool:
        """Whether the host may submit despite optional post-green work remaining."""
        return self.done_allowed and not self.repairable_advisories

    @property
    def phase(self) -> str:
        if self.no_host_checks and self.done_allowed:
            return PHASE_NO_HOST_CHECKS
        if self.ready and self.repairable_advisories:
            return PHASE_REPAIR_OPTIONAL
        if self.ready:
            return PHASE_READY
        if self.blockers:
            return PHASE_BLOCKED
        return PHASE_WORKING

    @property
    def blocker_ids(self) -> tuple[str, ...]:
        return tuple(_check_id(item) for item in self.blockers if _check_id(item))

    @property
    def advisory_ids(self) -> tuple[str, ...]:
        return tuple(_check_id(item) for item in self.advisory if _check_id(item))

    @property
    def done_allowed(self) -> bool:
        return ((self.ready or self.no_host_checks)
                and not self.blockers and not self.host_blocking
                and self.dispositions.get("done") == "allow")

    @property
    def next_recovery_action(self) -> str:
        if self.no_host_checks and self.done_allowed:
            return "done"
        if self.ready_to_submit or (self.ready and self.post_green_work and self.done_allowed):
            return "done"
        if self.repairable_advisories:
            return "repair"
        if self.blockers:
            return "repair"
        if not self.green:
            return "run_tests" if self.public_check_status != "FAIL" else "repair"
        if self.ready and (self.post_green_work or self.post_green_admitted_files):
            for name in ("run_tests", "edit", "str_replace", "read", "probe"):
                if (name in self.post_green_work
                        and self.dispositions.get(name) == "allow"):
                    return name
            return "repair"
        return "resolve_host_blockers"

    def refusal_text(self) -> str:
        """One denial carrying ALL reasons and the recovery path."""
        if self.no_host_checks and self.done_allowed:
            return ("No host checks exist for this task; completion is accepted only with "
                    "that limitation disclosed. Requirement evidence or model-owned checks "
                    "may be used. Next action: done.")
        if self.ready_to_submit:
            return "Public checks pass; no required repair remains. Next action: done."
        if self.repairable_advisories:
            item = self.repairable_advisories[0]
            return (f"Repair allowed at {_repair_target(item)} for {_check_id(item)}")
        if self.ready and self.post_green_work and self.done_allowed:
            work = ", ".join(self.post_green_work)
            available = tuple(name for name in REPAIR_TOOLS
                              if self.dispositions.get(name) == "allow")
            suffix = (f" ({'/'.join(available)} stay available)" if available else "")
            scope = (" Edit is admitted only for: "
                     + ", ".join(self.post_green_admitted_files) + "."
                     if self.post_green_admitted_files else "")
            return ("Public checks pass; optional post-green work remains: " + work
                    + "." + scope + " Next action: done." + suffix)
        parts: list[str] = []
        if self.blockers:
            parts.append(BLOCKED_BY_LABEL + ": " + json.dumps(
                list(self.blocker_ids), ensure_ascii=False, separators=(",", ":")))
            for item in self.blockers:
                clause = str(item.get("clause") or item.get("expected") or "")
                line = "- " + _check_id(item)
                if clause:
                    line += ": " + clause[:200]
                aligned_done_blocker = (
                    self.repair_blocks_done_active
                    and bool(item.get("literal_grounded"))
                    and isinstance(item.get("alignment_evidence"), Mapping))
                if aligned_done_blocker:
                    line += " (repair target: " + _repair_target(item) + ")"
                else:
                    target = str(item.get("target_file") or "")
                    if target:
                        line += " (repair target: " + target + ")"
                if item.get("signature_gap"):   # only set with STATEFRAME_NAMED_LAYER_REPAIR on
                    line += " -- " + str(item.get("signature_gap"))[:300]
                parts.append(line)
            if os.environ.get("STATEFRAME_TOKENLAB_COUNTEREXAMPLE_REFUSAL", "off").strip().lower() == "on":
                try:
                    from kernel_tokenlab_counterexample_refusal import render_evidence
                    evidence = render_evidence(self.blockers)
                except (ImportError, TypeError, ValueError):
                    evidence = ""
                if evidence:
                    parts.append(evidence)
        if not self.green:
            parts.append("public checks are " + self.public_check_status
                         + " at the current revision")
        if self.host_blocking:
            parts.append("host blockers open: " + json.dumps(
                list(self.host_blocking), ensure_ascii=False, separators=(",", ":")))
        if self.post_green_admitted_files:
            parts.append("edit is admitted only for: "
                         + ", ".join(self.post_green_admitted_files))
        available = tuple(name for name in REPAIR_TOOLS
                          if self.dispositions.get(name) == "allow")
        recovery = "next action: " + self.next_recovery_action
        if available:
            recovery += " (" + "/".join(available) + " stay available)"
        parts.append(recovery)
        return "done refused — " + "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema": SCHEMA,
            "state_version": self.state_version,
            "candidate_digest": self.candidate_digest,
            "revision": self.revision,
            "phase": self.phase,
            "ready": self.ready,
            "ready_to_submit": self.ready_to_submit,
            "green": self.green,
            "no_host_checks": self.no_host_checks,
            "host_check_state": NO_HOST_CHECKS if self.no_host_checks else "REQUIRED_PUBLIC_CHECKS",
            "public_check_status": self.public_check_status,
            "blockers": list(self.blocker_ids),
            "advisory": list(self.advisory_ids),
            "advisory_details": [copy.deepcopy(dict(item)) for item in self.advisory],
            "repairable_advisories": list(self.repairable_advisory_ids),
            "host_blocking": list(self.host_blocking),
            "next_recovery_action": self.next_recovery_action,
            "done_allowed": self.done_allowed,
            "dispositions": dict(self.dispositions),
        }
        if self.post_green_work:
            payload["post_green_work"] = list(self.post_green_work)
        if self.post_green_no_veto:
            payload["post_green_no_veto"] = True
        if self.post_green_admitted_files:
            payload["post_green_admitted_files"] = list(self.post_green_admitted_files)
        if self.post_green_work or self.post_green_admitted_files:
            payload["host_submittable"] = self.host_submittable
        if self.done_authority is not None:
            converter = getattr(self.done_authority, "to_dict", None)
            if callable(converter):
                payload["done_authority"] = converter()
        return payload


def _digest(revision: object, green: bool, status: str, blockers: Sequence[str],
            advisory: Sequence[Mapping[str, object]], host: Sequence[str]) -> str:
    payload = json.dumps({"r": revision, "g": green, "s": status, "b": sorted(blockers),
                          "a": [dict(item) for item in advisory], "h": sorted(host)},
                         sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def decide(status: object, *, previous: "ReadinessDecision | None" = None) -> ReadinessDecision:
    """Reduce a coherence ``TerminalStatus`` into the immutable decision snapshot.

    Blockers = counterexamples with ``counterexample_authority`` (valid, current,
    executed; the check classifier already demoted unbindable/self-contradicting/
    invalid checks to advisory).  With the separate done-blocking switch on,
    current executed literal-aligned valid rows are promoted into that same
    blocker set.  Advisory rows remain visible but never block otherwise.
    """
    counterexamples = tuple(getattr(status, "counterexamples", ()) or ())
    revision = getattr(status, "current_revision", None)
    blockers = tuple(item for item in counterexamples
                     if bool(item.get("counterexample_authority")))
    promoted: tuple[Mapping[str, object], ...] = ()
    repair_blocks_done = (
        os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"
        or getattr(status, "stateframe_repair_blocks_done_mode", "off") == "on")
    if repair_blocks_done:
        try:
            from kernel_repair_blocks_done import blocking_rows
            promoted = blocking_rows(counterexamples, revision=revision, active=True)
        except (ImportError, TypeError, ValueError):
            promoted = ()
    blocker_keys = {_check_id(item) for item in blockers if _check_id(item)}
    blockers = blockers + tuple(
        item for item in promoted if _check_id(item) not in blocker_keys)
    promoted_keys = {_check_id(item) for item in promoted if _check_id(item)}
    advisory = tuple(_annotate_advisory(item, getattr(status, "current_revision", None))
                     for item in counterexamples
                     if (not bool(item.get("counterexample_authority"))
                         and _check_id(item) not in promoted_keys))
    public_status = str(getattr(status, "public_check_status", "BLOCKED"))
    no_host_checks = bool(getattr(status, "no_host_checks", False)) \
        or public_status == NO_HOST_CHECKS \
        or str(getattr(status, "verdict_state", "")) == NO_HOST_CHECKS
    green = False if no_host_checks else bool(getattr(status, "green", False))
    host_blocking = tuple(str(item) for item in (getattr(status, "blocking", ()) or ()))
    if no_host_checks:
        public_status = NO_HOST_CHECKS
    ids = [_check_id(item) for item in blockers]
    digest = _digest(revision, green, public_status, ids, advisory, host_blocking)
    version = 1
    if previous is not None:
        version = previous.state_version + (0 if previous.candidate_digest == digest else 1)
    ready = green and not blockers and not host_blocking
    repairable = any(bool(item.get("repairable")) for item in advisory)
    done_ready = ready or (no_host_checks and not blockers and not host_blocking)
    dispositions = {"done": "allow" if done_ready else "deny"}
    for name in _active_repair_tools():
        dispositions[name] = "allow" if (not ready or repairable) else "deny"
    # STATEFRAME_GREEN_EVIDENCE_OWNER: readiness's own freshness read (`green`,
    # via public_test_verdict) can be SATISFIED while the continuation
    # completion gate's extra certification-sequence check still refuses
    # `done` as stale (kernel_continuation_evidence.py, `_public_completion`).
    # Without this, `run_tests` stays denied here while continuation loops
    # refusing `done` for the same reason -- the model has no allowed action
    # left that can change the outcome (r44d 370 ldev_02, 36x). Once the
    # owner has recorded that refusal, offer the one repair action that can
    # actually resolve it, without weakening `ready`/`done` itself.
    if dispositions.get("run_tests") == "deny":
        try:
            from kernel_green_evidence_owner import should_allow_run_tests
        except ImportError:
            should_allow_run_tests = None
        if should_allow_run_tests is not None and should_allow_run_tests():
            dispositions["run_tests"] = "allow"
    return ReadinessDecision(
        state_version=version, candidate_digest=digest, revision=revision, green=green,
        public_check_status=public_status, blockers=blockers, advisory=advisory,
        host_blocking=host_blocking,
        selectors=tuple(getattr(status, "selectors", ()) or ()),
        repair_action=getattr(status, "repair_action", None),
        dispositions=dispositions,
        repair_blocks_done_active=bool(promoted),
        no_host_checks=no_host_checks,
    )


def _pg_work(decision: object) -> tuple[str, ...]:
    """Duck-typed readiness decisions (replay/test doubles) predate the post-green fields: absent = ()."""
    return tuple(getattr(decision, "post_green_work", ()) or ())


def _pg_files(decision: object) -> tuple[str, ...]:
    return tuple(getattr(decision, "post_green_admitted_files", ()) or ())


def _done_allowed(decision: object) -> bool:
    value = getattr(decision, "done_allowed", None)
    return bool(getattr(decision, "ready", False)) if value is None else bool(value)


def project_menu(decision: ReadinessDecision, selected: Sequence[dict],
                 full_tools: Sequence[dict], *,
                 preserved_actions: Sequence[str] = (),
                 capability_admissible: Callable[[str], bool] | None = None,
                 capability_floor_enabled: bool | None = None
                 ) -> tuple[list[dict], dict[str, list[str]]]:
    """Final menu projection; enforces the §71 menu invariants on ANY upstream narrowing.

    * not ready -> every available repair tool (edit/str_replace/run_tests) is present;
    * blocked by a live counterexample -> certification stays denied and challenge
      tools (dispute/read_ticket) are kept when the host offers them;
    * an active capability floor may retain ``done`` for an uncertified submission;
    * ready with no repairable advisory -> done-only;
    * ready with a repairable advisory -> done plus the scoped repair tools.
    """
    menu = list(selected)
    names = [_tool_name(tool) for tool in menu]
    available = {_tool_name(tool): tool for tool in full_tools}
    preserved = {
        str(name) for name in preserved_actions
        if str(name) and str(name) in available
    }
    added: list[str] = []
    removed: list[str] = []

    def finish(target: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
        from kernel_capability_floor import apply_floor

        before = {_tool_name(tool) for tool in target}
        result = apply_floor(
            target, full_tools, admissible=capability_admissible,
            active=capability_floor_enabled is not False,
        )
        added.extend(name for name in (_tool_name(tool) for tool in result)
                     if name not in before)
        return result, {"added": added, "removed": removed}

    def restore_observations(target: list[dict] | None = None) -> None:
        destination = menu if target is None else target
        for name in sorted(OBSERVATION_TOOLS):
            if name not in names and name in available:
                destination.append(available[name])
                names.append(name)
                added.append(name)

    if decision.no_host_checks:
        allowed = {"done", *OBSERVATION_TOOLS, *_active_repair_tools(), *preserved}
        filtered = [tool for tool in menu if _tool_name(tool) in allowed]
        removed.extend(name for name in names if name not in allowed)
        names = [_tool_name(tool) for tool in filtered]
        if decision.done_allowed and "done" not in names and "done" in available:
            filtered.append(available["done"])
            names.append("done")
            added.append("done")
        restore_observations(filtered)
        return finish(filtered)

    if bool(getattr(decision, "post_green_no_veto", False)):
        allowed = {"done", *OBSERVATION_TOOLS, *_active_repair_tools(), *_pg_work(decision), *preserved}
        post_green_order = (
            ("done", "read", *_active_repair_tools(), *_pg_work(decision))
            if "file_create" in _active_repair_tools()
            else ("done", "edit", "str_replace", "read", *_pg_work(decision))
        )
        filtered = [tool for tool in menu if _tool_name(tool) in allowed]
        removed.extend(name for name in names if name not in allowed)
        names = [_tool_name(tool) for tool in filtered]
        for name in post_green_order:
            if name not in names and name in available:
                filtered.append(available[name])
                names.append(name)
                added.append(name)
        restore_observations(filtered)
        return finish(filtered)
    if decision.ready and (_pg_work(decision) or _pg_files(decision)
                           or not _done_allowed(decision)):
        allowed = set(_pg_work(decision)) | set(OBSERVATION_TOOLS) | preserved
        if decision.dispositions.get("edit") == "allow":
            allowed.add("str_replace")
        if _done_allowed(decision):
            allowed.add("done")
        filtered = [tool for tool in menu if _tool_name(tool) in allowed]
        removed.extend(name for name in names if name not in allowed)
        names = [_tool_name(tool) for tool in filtered]
        for name in sorted(allowed):
            if name == "edit" and name not in available and "str_replace" in available:
                continue
            if name not in names and name in available:
                filtered.append(available[name])
                names.append(name)
                added.append(name)
        restore_observations(filtered)
        return finish(filtered)
    if decision.ready_to_submit:
        allowed = {"done", *OBSERVATION_TOOLS, *preserved}
        filtered = [tool for tool in menu if _tool_name(tool) in allowed]
        removed.extend(name for name in names if name not in allowed)
        names = [_tool_name(tool) for tool in filtered]
        if _done_allowed(decision) and "done" not in names and "done" in available:
            filtered.append(available["done"])
            names.append("done")
            added.append("done")
        restore_observations(filtered)
        return finish(filtered)

    # Only a typed, current repairable advisory keeps repair reachable after green.
    if not decision.ready or (decision.ready and decision.repairable_advisories):
        if decision.ready and decision.repairable_advisories:
            allowed = {"done", *OBSERVATION_TOOLS, *_active_repair_tools(), *preserved}
            filtered = [tool for tool in menu if _tool_name(tool) in allowed]
            removed.extend(name for name in names if name not in allowed)
            menu = filtered
            names = [_tool_name(tool) for tool in menu]
            if "done" not in names and "done" in available:
                menu.append(available["done"])
                names.append("done")
                added.append("done")
        for name in _active_repair_tools():
            if name not in names and name in available:
                menu.append(available[name])
                names.append(name)
                added.append(name)
    if not decision.ready:
        if decision.blockers:
            for name in sorted(CHALLENGE_TOOLS):
                if name not in names and name in available:
                    menu.append(available[name])
                    names.append(name)
                    added.append(name)
            allowed = (set(_active_repair_tools()) | set(CHALLENGE_TOOLS)
                       | set(OBSERVATION_TOOLS) | preserved)
            filtered = [tool for tool in menu if _tool_name(tool) in allowed]
            removed.extend(name for name in names if name not in allowed)
            menu = filtered
            kept = [tool for tool in menu if _tool_name(tool) != "done"]
            if len(kept) != len(menu):
                removed.append("done")
                menu = kept
    restore_observations()
    return finish(menu)


def menu_violations(decision: ReadinessDecision, menu: Sequence[dict],
                    full_tools: Sequence[dict], *,
                    preserved_actions: Sequence[str] = (),
                    capability_admissible: Callable[[str], bool] | None = None,
                    capability_floor_enabled: bool | None = None) -> list[str]:
    """Invariants over (decision, model-visible menu); empty list == consistent."""
    names = {_tool_name(tool) for tool in menu}
    available = {_tool_name(tool) for tool in full_tools}
    preserved = {
        str(name) for name in preserved_actions
        if str(name) and str(name) in available
    }
    from kernel_capability_floor import admissible_floor_names, enabled as capability_floor_is_enabled
    floor_on = (capability_floor_is_enabled() if capability_floor_enabled is None
                else capability_floor_enabled)
    floor_names = set(admissible_floor_names(
        full_tools, capability_admissible,
    )) if floor_on else set()
    problems: list[str] = []
    if decision.no_host_checks:
        allowed = {"done", *OBSERVATION_TOOLS, *_active_repair_tools(), *preserved,
                   *floor_names}
        unexpected = sorted(names - allowed)
        if unexpected:
            problems.append("unscoped tools offered with no host checks: " + ",".join(unexpected))
        if decision.done_allowed and "done" in available and "done" not in names:
            problems.append("done missing while no host checks are disclosed")
        missing_floor = sorted(floor_names - names)
        if missing_floor:
            problems.append("capability floor missing: " + ",".join(missing_floor))
        return problems
    if bool(getattr(decision, "post_green_no_veto", False)):
        allowed = {"done", *OBSERVATION_TOOLS, "edit", "str_replace", *_pg_work(decision),
                   *preserved, *floor_names}
        missing = [name for name in ("edit", "str_replace", "read")
                   if name in available and name not in names]
        if missing:
            problems.append("product tools missing after post-green no-veto: " + ",".join(missing))
        unexpected = sorted(names - allowed)
        if unexpected:
            problems.append("unscoped tools offered after post-green no-veto: " + ",".join(unexpected))
    elif decision.ready_to_submit:
        if "done" not in names and "done" in available and "done" not in floor_names:
            problems.append("done missing while ready to submit")
        unexpected = sorted(names - ({"done"} | set(OBSERVATION_TOOLS) | preserved | floor_names))
        if unexpected:
            problems.append("tools offered after ready-to-submit: " + ",".join(unexpected))
    elif decision.ready and (_pg_work(decision) or _pg_files(decision)
                             or not _done_allowed(decision)):
        allowed = set(_pg_work(decision)) | set(OBSERVATION_TOOLS) | preserved | floor_names
        if decision.dispositions.get("edit") == "allow":
            allowed.add("str_replace")
        if _done_allowed(decision):
            allowed.add("done")
        unexpected = sorted(names - allowed)
        if unexpected:
            problems.append("tools offered outside post-green disposition: "
                            + ",".join(unexpected))
        required = set(allowed)
        if "edit" in required and "edit" not in available:
            required.discard("edit")
        missing = sorted(name for name in required if name in available and name not in names)
        if missing:
            problems.append("post-green tools missing: " + ",".join(missing))
    elif decision.ready and decision.repairable_advisories:
        missing = [name for name in REPAIR_TOOLS if name in available and name not in names]
        if missing:
            problems.append("repair tools missing for repairable advisory: " + ",".join(missing))
        unexpected = sorted(names - ({"done"} | set(REPAIR_TOOLS) | set(OBSERVATION_TOOLS)
                                     | preserved | floor_names))
        if unexpected:
            problems.append("unscoped tools offered for repairable advisory: " + ",".join(unexpected))
    elif not decision.ready:
        missing = [name for name in REPAIR_TOOLS if name in available and name not in names]
        if missing:
            problems.append("repair tools missing while not ready or counterexample visible: "
                            + ",".join(missing))
    if not decision.ready:
        if ("done" in names and names <= ({"done"} | set(OBSERVATION_TOOLS))
                and not (names & preserved)):
            problems.append("done-only menu while not ready")
    missing_floor = sorted(floor_names - names)
    if missing_floor:
        problems.append("capability floor missing: " + ",".join(missing_floor))
    missing_preserved = sorted(name for name in preserved if name not in names)
    if missing_preserved:
        problems.append("preserved stage actions missing: " + ",".join(missing_preserved))
    if decision.blockers and "done" in names and "done" not in floor_names:
        problems.append("done offered while a live counterexample is open")
    return problems


BLOCKED_PENDING_DECISION = (
    "CERTIFICATION BLOCKED: a live counterexample is open at the current revision; "
    "repair it (edit/str_replace), then run_tests, or submit as uncertified with "
    "the open obligations listed."
)


def blocked_by_counterexample(runtime: object | None) -> bool:
    """True when the authority is on and the current decision has a live blocker."""
    if not runtime_enabled(runtime):
        return False
    try:
        decision = runtime_snapshot(runtime)
        return True if decision is None else bool(decision.blockers)
    except Exception as error:  # noqa: BLE001 - authority error holds readiness closed
        record_failure(
            runtime,
            "readiness.blocked_by_counterexample",
            error,
            authority=True,
            disposition="hold",
            denied_actions=("done", "submit"),
        )
        return True


__all__ = [
    "BLOCKED_PENDING_DECISION", "blocked_by_counterexample",
    "BLOCKED_BY_LABEL", "CHALLENGE_TOOLS", "DECISION_EVENT", "EDIT_TOOLS", "MENU_EVENT",
    "MODE_ENV", "PHASE_BLOCKED", "PHASE_READY", "PHASE_REPAIR_OPTIONAL", "PHASE_WORKING", "REFUSAL_CODE",
    "REPAIR_TOOLS", "ReadinessDecision", "SCHEMA", "VIOLATION_EVENT", "decide", "enabled",
    "menu_violations", "mode", "project_menu", "runtime_enabled",
    "runtime_snapshot",
]


def defer_to_post_green(
        decision: ReadinessDecision,
        actions: Mapping[str, str] | Sequence[str],
        *,
        admitted_files: Sequence[str] = (),
) -> ReadinessDecision:
    """Project the authority's post-green allow/deny verdicts into readiness.

    A legacy sequence means every named action is allowed.  The mapping form
    is used by the live runtime so a pending-verification ``done`` denial is
    not lost while the work window is deferred.
    """
    if isinstance(actions, Mapping):
        verdicts = {
            str(name).strip(): str(verdict).strip().lower()
            for name, verdict in actions.items()
            if str(name).strip() and str(verdict).strip().lower() in {"allow", "deny"}
        }
    else:
        verdicts = {str(name).strip(): "allow" for name in actions if str(name).strip()}
    if not verdicts or not decision.ready:
        return decision
    dispositions = dict(decision.dispositions)
    for name, verdict in verdicts.items():
        dispositions[name] = verdict
    if "edit" in verdicts and "str_replace" not in verdicts:
        dispositions["str_replace"] = verdicts["edit"]
    work = tuple(sorted(name for name, verdict in verdicts.items()
                        if name != "done" and verdict == "allow"))
    files = tuple(dict.fromkeys(str(item).strip() for item in admitted_files if str(item).strip()))
    return dataclasses.replace(
        decision, dispositions=dispositions, post_green_work=work,
        post_green_admitted_files=files)
