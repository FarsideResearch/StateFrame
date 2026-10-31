"""Single request-scoped allow authority for the terminal green lock (r43).

r42's live replay showed the offered menu and the dispatch gates disagreeing:
``green_lock`` refused ``run_tests`` and the repair gate refused
``str_replace`` while both were on the menu.  The legacy code computes the
menu (``_terminal_tool_menu`` + repair/coherence/get-sure mutators) and the
dispatch verdict (``GreenLock.before_action`` / ``before_mutation``) with two
independent rule chains.

With ``STATEFRAME_GREENLOCK_MENU_AUTHORITY=on`` there is ONE decision
function, :func:`decide`.  It reproduces the dispatch precedence of the
legacy gates (stopped state, unverified-check window, get-sure, coherence done
refusal, repair gate, frame-edit-integrity recovery, relaxed profile, open
counterexamples, coherence override, first-refusal override) as a pure read
of the lock state.  Both consumers use it:

* :func:`project_request_menu` - the final request menu is exactly the set of
  tools ``decide`` allows (projected from the full tool surface while the lock
  is engaged); it never restores a disallowed tool.  An empty allow-set is
  recorded so ``prepare`` ends the attempt with an explicit host terminal
  outcome instead of sending a request whose every tool would be refused.
* :func:`dispatch_action` / :func:`dispatch_mutation` - the lock's dispatch
  hooks; they apply the bookkeeping side effects (refusal counters, override
  flags, repair controller capture) around the same decision.

Scope: terminal green-lock mode only, and never the H+state history arm (that
arm owns its own narrow argument-level rerun rule and its full menu).  With the
switch off nothing here runs and the legacy paths are byte-identical.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kernel_bounded_locate_code import OBSERVATION_TOOLS
from kernel_measurement import profiled_function_span


MODE_ENV = "STATEFRAME_GREENLOCK_MENU_AUTHORITY"
MENU_EVENT = "green_lock_menu_authority"
EMPTY_EVENT = "green_lock_menu_authority_empty"
EMPTY_OUTCOME = "green_lock_no_permitted_action"
EMPTY_OVERFLOW_KIND = "GREEN_LOCK_NO_PERMITTED_ACTION"
EDIT_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "file_delete", "file_move", "shell", "replace_module", "write", "replace",
})
VERIFY_ACTIONS = frozenset({"run_tests", "test", "pytest", "predict"})
WINDOW_ACTIONS = frozenset({"read", "probe"})
# Product capabilities the readiness disposition never removes under hard-budget-only terminals (b1393).
_READINESS_NEVER_STRIPS = frozenset({"edit", "str_replace", "file_create", "file_delete", "file_move", "shell", "run_tests", "read_check_receipt"})
UPSTREAM_REFUSAL = "MENU: {action} is not offered on this request; use an offered tool."


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_greenlock_menu_authority_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def _history_arm(runtime: object | None) -> bool:
    try:
        from kernel_stateframe_hstate_plus import history_arm
        return bool(history_arm(runtime))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


# r45 rebase onto r44c: r44c makes this authority own green_lock/repair_gate UNCONDITIONALLY under a live
# continuation, so authority_under_continuation's composition-ready gate is redundant and was removed.  The
# lane's additional, replay-evidenced behaviours below (per-step continuation allow-set intersection =
# queued-batch menu re-projection, readiness-denial of done) stay behind
# STATEFRAME_GREENLOCK_MENU_AUTHORITY_CONTINUATION (default off; off in every r45 arm).
def applies(lock: object | None) -> bool:
    """True when the authority owns this lock's menu and dispatch."""
    if lock is None or getattr(lock, "mode", "off") != "terminal":
        return False
    runtime = getattr(lock, "runtime", None)
    if not enabled(runtime) or _history_arm(runtime):
        return False
    # Bounded-locate phases own the menu while active; repro-first is composed into
    # this authority so its product-edit predicate cannot diverge from projection.
    # r44b: a LiveContinuation does NOT switch the lock's authority off.  Its
    # intention-level gate (LiveContinuation.authorize) is a separate dispatch layer
    # that never adds a tool; declining here left every frame arm on two divergent
    # legacy rule chains (menu offered read, green_lock refused it: 362-ldev_01).
    return getattr(runtime, "bounded_locate", None) is None


def _repro_first_active(runtime: object | None) -> bool:
    """Same activation test ``guard_product_mutation`` uses for the repro-first edit
    gate (review6): while it enforces, product-edit permission is composed into
    menu projection and dispatch rather than disabling this authority."""
    if os.environ.get("STATEFRAME_REPRO_FIRST_GATE", "off") == "off":
        return False
    from kernel_repro_first import repro_first_gate_mode
    gate = getattr(runtime, "repro_first_gate", None)
    return (repro_first_gate_mode() == "on"
            and bool(getattr(gate, "active", False)))


def _repro_first_edit_allowed(runtime: object, arguments: Mapping[str, object], *,
                              for_menu: bool) -> bool:
    if not _repro_first_active(runtime):
        return True
    from kernel_green_lock import is_product_path
    from kernel_repro_first import product_edit_allowed
    raw_path = arguments.get("path") if isinstance(arguments, Mapping) else None
    is_test = False if for_menu else not is_product_path(raw_path or "")
    current_hash = None
    if not for_menu:
        owner = getattr(runtime, "owner", None)
        root = getattr(runtime, "root", None) or getattr(owner, "root", None)
        if root is not None:
            from kernel_repro_first import code_hash
            current_hash = code_hash(root)
    return product_edit_allowed(
        getattr(runtime, "repro_first_gate", None), code_hash=current_hash, is_test=is_test)


@dataclass(frozen=True)
class Decision:
    """One allow/deny verdict.  ``rule`` names the deciding predicate.

    ``witness`` is the argument set under which a menu query was allowed
    (``None`` = any product path / no arguments needed).
    """

    action: str
    allowed: bool
    rule: str
    witness: dict[str, object] | None = None
    evidence: tuple[Mapping[str, object], ...] = field(default=(), compare=False)
    message: str | None = None


def _norm(action: object) -> str:
    return str(action or "").strip().lower()


def _name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


# --------------------------------------------------------------------------
# State readers (no side effects)
# --------------------------------------------------------------------------

def _stopped(lock: object) -> bool:
    return lock._repair_blocks_done_refusal() is not None


def _window_open(runtime: object) -> bool:
    from kernel_green_lock import _unverified_window_open
    return _unverified_window_open(runtime)


def _get_sure(runtime: object) -> bool:
    from kernel_green_lock import _get_sure_open
    return _get_sure_open(runtime)


def _coherence(lock: object) -> bool:
    return bool(lock._coherence_enabled())


def _done_refusal(runtime: object) -> str | None:
    from kernel_green_menu_open_obligation import done_refusal
    return done_refusal(runtime)


def _open_counterexamples(runtime: object) -> tuple[object, ...]:
    from kernel_green_menu_open_obligation import open_counterexamples
    return tuple(open_counterexamples(runtime))


def _integrity() -> bool:
    from kernel_frame_edit_integrity import enabled as integrity_enabled
    return bool(integrity_enabled())


def _relaxed() -> bool:
    from kernel_model_profile import green_menu_relaxed
    return bool(green_menu_relaxed())


def _linked(action: str, arguments: Mapping[str, object], runtime: object) -> bool:
    try:
        from kernel_green_lock_repair import action_linked_to_counterexample
        return bool(action_linked_to_counterexample(action, arguments, runtime))
    except (ImportError, TypeError, ValueError):
        return False


def _evidence_paths(evidence: Sequence[Mapping[str, object]]) -> list[str]:
    paths: list[str] = []
    for item in evidence:
        targets = item.get("targets") if isinstance(item, Mapping) else None
        values = targets.get("paths", ()) if isinstance(targets, Mapping) else ()
        if isinstance(values, str):
            values = (values,)
        for value in sorted(str(path) for path in (values or ()) if str(path).strip()):
            if value not in paths:
                paths.append(value)
    return paths


def _readiness_on(runtime: object) -> bool:
    return getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"


def _repair_on(lock: object) -> bool:
    """Repair gate active: the lock's repair switch OR the readiness authority.

    r43 review3: the readiness authority rewrites the menu around its blockers
    (repair tools in, done out).  That permission lives HERE, in the one
    decision, instead of in a second competing final menu authority.
    """
    return bool(lock._repair_enabled()) or _readiness_on(lock.runtime)


def _repair_evidence(lock: object) -> tuple[Mapping[str, object], ...]:
    if lock._repair_enabled():
        return tuple(lock._repair_evidence() or ())
    try:
        from kernel_green_lock_repair import readiness_repair_counterexamples
        return tuple(readiness_repair_counterexamples(lock.runtime) or ())
    except (ImportError, TypeError, ValueError):
        return ()


def canonical_action(action: object) -> str:
    """THE action identity shared by decide(), the menu and every guard (r43 review4).

    The real write guard (``guard_product_mutation`` -> ``before_mutation(path)``)
    cannot see which edit alias the provider called (str_replace, file_create, ...);
    it always presents ``edit``.  Every edit alias therefore IS one identity here,
    so a menu offering any alias is executable through that guard.
    """
    name = _norm(action)
    return "edit" if name in EDIT_ACTIONS else name


def _edit_class(action: str) -> str:
    return "edit" if action in EDIT_ACTIONS else action


def _intersect_continuation(lock: object, decision: Decision, *, requested_action: object = None,
                            authorized_action: object = None) -> Decision:
    """Apply the executive gate, except for its exact already-admitted write.

    ``authorize`` creates a pending effect before the host editor reaches its
    path-only guard.  In that one boundary the next-action projection is
    intentionally ``done``; the write guard instead checks the exact dispatch
    fence supplied by ``authorized_action``.  No unrelated pending effect gets
    this exception because callers must prove that fence first.
    """
    runtime = lock.runtime
    continuation = getattr(runtime, "continuation", None)
    if continuation is None:
        return decision
    from kernel_continuation_menu_authority import enabled as continuation_mode_enabled
    from kernel_continuation_menu_authority import per_step_allow_set
    if not continuation_mode_enabled(runtime):
        return decision
    allowed = per_step_allow_set(runtime)
    _continuation_projection_event(runtime, allowed)
    if _norm(decision.action) in OBSERVATION_TOOLS:
        return decision
    if allowed is None or not decision.allowed:
        return decision
    requested = _norm(decision.action if requested_action is None else requested_action)
    admitted = _norm(authorized_action)
    if admitted and requested == admitted:
        return decision
    if requested in allowed:
        return decision
    _continuation_acted_event(runtime, requested, allowed)
    return Decision(decision.action, False, "continuation_gate",
                    decision.witness, decision.evidence)


def _continuation_acted_event(runtime: object, requested: str, allowed: object) -> None:
    """Acted row: the continuation gate narrowed ``requested``.  Once per (action, allow set): a menu build asks
    every tool, and repeated identical rows would only add journal volume."""
    key = (requested, tuple(sorted(str(item) for item in allowed)))
    seen = getattr(runtime, "_r45_continuation_acted_seen", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            setattr(runtime, "_r45_continuation_acted_seen", seen)
        except Exception:  # noqa: BLE001
            return
    if key in seen:
        return
    seen.add(key)
    # Counted, not journaled here: a journal record written inside the decide path perturbs the live prepare's
    # required working group (test_authority_under_continuation.test_live_prepare_uses_continuation_composed_menu
    # overflowed its 9000 ceiling).  The count rides on the next projection row.
    try:
        setattr(runtime, "_r45_continuation_acted", int(getattr(runtime, "_r45_continuation_acted", 0) or 0) + 1)
    except Exception:  # noqa: BLE001
        return


def _continuation_projection_event(runtime: object, allowed: object) -> None:
    """Own event of STATEFRAME_GREENLOCK_MENU_AUTHORITY_CONTINUATION: once per per-step allow set (evaluated when
    the executive imposes no set, applicable when it does).  Deduped on the set so menu builds do not flood."""
    key = None if allowed is None else tuple(sorted(str(item) for item in allowed))
    if getattr(runtime, "_r45_continuation_projection_last", object()) == key:
        return
    try:
        setattr(runtime, "_r45_continuation_projection_last", key)
    except Exception:  # noqa: BLE001
        return
    _emit(runtime, "continuation_menu_projection", stage="evaluated" if key is None else "applicable",
          allowed=list(key) if key is not None else None,
          acted_total=int(getattr(runtime, "_r45_continuation_acted", 0) or 0))


def _owner_terminal_submission_reserved(runtime: object, snapshot: object) -> bool:
    """True when the completion owner has reserved an honest uncertified submission.

    Measured (batch 800 sample task, seq 591/869/922): public tests PASS, but open
    obligations with ``MISSING_CHECKS`` keep readiness ``host_blocking`` so its
    ``done`` disposition is ``deny``. The completion owner then offers its
    terminal choice ("call done now... uncertified; open: [...]. No other action
    is available") while the continuation gate also withholds every authoring
    action. Readiness denying that ``done`` left zero productive actions: the
    run looped done -> green_lock_refused 37x until the request cap.

    The owner's reserved submission discloses the open obligations, so it is the
    honest terminal and readiness must not veto it. Never applies when readiness
    holds a counterexample blocker (a failing check is real, repairable evidence)
    or when the owner has not reserved a final submission.
    """
    if getattr(snapshot, "blockers", ()):
        return False
    if _norm(getattr(runtime, "stateframe_completion_owner_mode", "on") or "on") == "off":
        return False
    # Measured (batch 914 sample task, seq 1037/1079..3209): the owner's no-progress
    # edit cap selected the one-shot uncertified terminal
    # (``_begin_uncertified_submit``; system_outcome UNCERTIFIED_SUBMIT) without
    # ever setting terminal_choice_offered/final_submit_reserved, and the tool
    # surface re-offered ``done``.  Readiness still denied it (MISSING_CHECKS
    # host_blocking, public status BLOCKED) -> gate/done refused 32x to the cap.
    # The host-selected uncertified submit is the same honest terminal.
    if getattr(runtime, "_completion_owner_uncertified_submit", False) is True:
        return True
    state = getattr(runtime, "_completion_owner_state", None)
    if not isinstance(state, Mapping):
        return False
    return (state.get("terminal_choice_offered") is True
            and state.get("final_submit_reserved") is True)


def _readiness_done_deny_is_stale_or_host_debt(runtime: object, snapshot: object) -> bool:
    """True when public tests are green on the CURRENT binding and the snapshot's done=deny carries no evidence
    against the candidate: either the snapshot is pre-green, or its only reason is host debt (host_blocking =
    clause obligations UNKNOWN/MISSING_CHECKS) with NO counterexample blockers. Measured: b1427 seq 1933 and
    b1428 seq 1866 are green=True, blockers=[], host_blocking=[obl-* UNKNOWN MISSING_CHECKS]. Counterexample
    blockers keep refusing. Current green uses the owner's strict check (SATISFIED verdict on the current
    binding, every trusted receipt stable, exit 0, bound to it)."""
    if tuple(getattr(snapshot, "blockers", ()) or ()):
        return False
    try:
        from kernel_completion_owner import _green_current
        if not _green_current(runtime):
            return False
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return (not bool(getattr(snapshot, "green", False))
            or bool(tuple(getattr(snapshot, "host_blocking", ()) or ())))


def _lock_label_stale_readiness(runtime: object, snapshot: object) -> None:
    key = (getattr(snapshot, "state_version", None), str(getattr(snapshot, "revision", ""))[:80])
    seen = getattr(runtime, "_stale_readiness_done_labels", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            setattr(runtime, "_stale_readiness_done_labels", seen)
        except (AttributeError, TypeError):
            return
    if key in seen:
        return
    seen.add(key)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("done_label_readiness_host_debt", producer="kernel_greenlock_menu_authority",
             snapshot_state_version=key[0], snapshot_green=bool(getattr(snapshot, "green", False)),
             dispatch=False)


def _intersect_readiness(lock: object, decision: Decision) -> Decision:
    """r44 P1-4: under continuation composition, honour the readiness disposition.

    While the lock is not engaged, r43 left done/repair-tool shape to the upstream
    menu policies.  The continuation's no-root projection can empty that upstream
    shape, and the recovery path then re-offered ``done`` although the readiness
    authority's journaled disposition denies it before green (a replayed
    ``authority:done`` refusal on the offered menu).  The disposition is therefore
    one more input to the single decision, for menu and dispatch alike.  Engaged
    lock rules already consult readiness and are unchanged.
    """
    if _norm(decision.action) in OBSERVATION_TOOLS:
        return decision
    if decision.rule.startswith("postgreen_no_veto"):
        return decision
    # b1386/b1381: a disengaged green lock let the readiness disposition deny
    # edit/str_replace/run_tests on every green menu while the model still had product work (correct
    # candidates timed out). The capability floor survives that deny; readiness still governs done.
    if decision.rule == "capability_floor" and _norm(decision.action) != "done":
        return decision
    if not decision.allowed or getattr(lock, "terminal_engaged", False):
        return decision
    runtime = lock.runtime
    continuation = getattr(runtime, "continuation", None)
    if continuation is None or not _readiness_on(runtime):
        return decision
    # Before the first selection the executive's no-root projection is done-only;
    # denying that done would empty the very first menu and end the run, so the
    # disposition only narrows once an intention is active.
    try:
        getter = getattr(continuation.engine, "active_intention_id", None)
        active = getter() if callable(getter) else continuation.engine.state.active_id
    except Exception:  # noqa: BLE001 - a broken executive keeps the r43 decision
        return decision
    if active is None:
        return decision
    from kernel_continuation_menu_authority import enabled as continuation_mode_enabled
    if not continuation_mode_enabled(runtime):
        return decision
    snapshot = lock._readiness_snapshot()
    dispositions = getattr(snapshot, "dispositions", None)
    if (_norm(decision.action) == "done"
            and _owner_terminal_submission_reserved(runtime, snapshot)):
        return decision
    if (_norm(decision.action) == "done" and isinstance(dispositions, Mapping)
            and _norm(dispositions.get("done", "")) == "deny"
            and _readiness_done_deny_is_stale_or_host_debt(runtime, snapshot)):
        # The readiness
        # snapshot's done=deny came only from host debt (host_blocking: clause obligations UNKNOWN/MISSING_CHECKS,
        # no counterexample blockers) or predated green, yet it refused every done on the green, changed
        # candidate. That deny never refuses done; it is journaled once per snapshot as a label.
        _lock_label_stale_readiness(runtime, snapshot)
        return decision
    if isinstance(dispositions, Mapping) and _norm(dispositions.get(decision.action, "")) == "deny":
        # b1393 (r2ES live, BUILDFIX2 10-18): readiness_disposition stripped edit/str_replace/run_tests on all
        # 34 post-green menus while the model still had product work. Under hard-budget-only terminals,
        # readiness steers done only; it never removes a product capability. Each override is recorded and
        # journaled as capability_floor_kept by the menu event (never journaled inside decide: see above).
        if _norm(decision.action) in _READINESS_NEVER_STRIPS:
            from kernel_terminal_policy import enabled as hard_budget_only_enabled
            if hard_budget_only_enabled(runtime):
                _note_floor_kept(runtime, _norm(decision.action))
                return decision
        return Decision(decision.action, False, "readiness_disposition",
                        decision.witness, decision.evidence)
    return decision


FLOOR_KEPT_EVENT = "capability_floor_kept"


def _note_floor_kept(runtime: object, action: str) -> None:
    """Record a readiness strip the hard-budget floor overrode; drained by the next menu event (one per tool)."""
    pending = getattr(runtime, "_floor_kept_pending", None)
    if not isinstance(pending, set):
        pending = set()
        try:
            setattr(runtime, "_floor_kept_pending", pending)
        except Exception:  # noqa: BLE001 - a frozen runtime only loses the journal row, never the decision
            return
    pending.add(action)


def _drain_floor_kept(runtime: object) -> None:
    """Journal one capability_floor_kept event per overridden tool for this menu build, recording the reason."""
    pending = getattr(runtime, "_floor_kept_pending", None)
    if not isinstance(pending, set) or not pending:
        return
    request_id = getattr(runtime, "current_request_id", None)
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = getattr(runtime, "_completion_owner_pending_request_id", None)
    for tool in sorted(pending):
        _emit(runtime, FLOOR_KEPT_EVENT, tool=tool, overridden_rule="readiness_disposition",
              decision="kept", reason="hard_budget_only_terminals",
              request_id=request_id)
    pending.clear()


# --------------------------------------------------------------------------
# The single decision
# --------------------------------------------------------------------------

def _postgreen_no_veto(lock: object, name: str, args: Mapping[str, object], *,
                       for_menu: bool) -> Decision | None:
    """Delegate every post-green product edit/read and done refusal to one owner."""
    from kernel_postgreen_no_veto import decide as no_veto_decide
    verdict = no_veto_decide(lock, name, args, for_menu=for_menu)
    if verdict is None:
        return None
    witness = dict(verdict.witness) if isinstance(verdict.witness, Mapping) else None
    return Decision(name, verdict.allowed, verdict.rule, witness, (), verdict.message)

def refresh(lock: object) -> bool:
    """Reconcile lock state once before a menu build or a dispatch.

    This is the same reconciliation the legacy gate performed at dispatch
    (coherence evidence reset, then ``sync``).  Menu and dispatch call it the
    same way, so both decide over identical state.
    """
    if _coherence(lock):
        from kernel_green_menu_open_obligation import reset_override_state_if_evidence_changed
        reset_override_state_if_evidence_changed(lock, runtime=lock.runtime)
    lock.sync()
    return bool(lock.terminal_engaged)


def _decide_edit(lock: object, action: str, arguments: Mapping[str, object], *,
                 for_menu: bool) -> Decision:
    """Legacy ``before_mutation`` precedence (the real edit dispatch gate)."""
    runtime = lock.runtime
    if _stopped(lock):
        return Decision(action, False, "stopped")
    if _get_sure(runtime):
        return Decision(action, True, "get_sure_open")
    if not lock.terminal_engaged:
        return Decision(action, True, "not_engaged")
    target = None
    if not for_menu:
        from kernel_green_lock import is_product_path
        target = lock._workspace_path(arguments.get("path", ""))
        if not is_product_path(target):
            return Decision(action, True, "non_product_path")
    if _relaxed():
        return Decision(action, True, "relaxed_profile")
    if _repair_on(lock):
        evidence = _repair_evidence(lock)
        if for_menu:
            # Path-independent permission (readiness / repair_blocks_done evidence
            # admits any edit) is the SAME predicate the write guard evaluates.
            if evidence and _linked(action, {}, runtime):
                return Decision(action, True, "repair_linked", None, evidence)
            for path in _evidence_paths(evidence):
                if _linked(action, {"path": path}, runtime):
                    return Decision(action, True, "repair_linked", {"path": path}, evidence)
            return Decision(action, False,
                            "repair_unrelated" if evidence else "repair_post_green", None, evidence)
        if _linked(action, {"path": target}, runtime):
            return Decision(action, True, "repair_linked", {"path": target}, evidence)
        return Decision(action, False,
                        "repair_unrelated" if evidence else "repair_post_green", None, evidence)
    if _integrity():
        return Decision(action, True, "frame_edit_integrity_recovery")
    coherence = _coherence(lock)
    open_rows = _open_counterexamples(runtime) if coherence else ()
    if open_rows:
        return Decision(action, True, "open_counterexample_repair")
    override = getattr(lock, "_override_action_kind", None)
    if coherence and override is not None:
        # The override is the escape hatch for one action CLASS: every edit
        # tool reaches the write guard as ``edit``, so compare classes.
        if _edit_class(_norm(override)) != "edit":
            return Decision(action, False, "narrowed_terminal_menu")
        return Decision(action, True, "coherent_override")
    refused = sorted(str(item) for item in getattr(lock, "_refused_targets", ()) or ())
    if for_menu:
        if refused:
            return Decision(action, True, "target_override", {"path": refused[0]})
        return Decision(action, False, "coherent_refusal" if coherence else "first_refusal")
    if target in refused:
        return Decision(action, True, "target_override", {"path": target})
    return Decision(action, False, "coherent_refusal" if coherence else "first_refusal")


def _decide_other(lock: object, action: str, arguments: Mapping[str, object]) -> Decision:
    """Legacy ``before_action`` precedence for every non-edit action."""
    runtime = lock.runtime
    if _repro_first_active(runtime) and action in {"reproduce", "read_check_receipt"}:
        return Decision(action, True, "repro_first_recovery")
    if _stopped(lock):
        return Decision(action, False, "stopped")
    if action in WINDOW_ACTIONS and _window_open(runtime):
        return Decision(action, True, "unverified_checks_window")
    if _get_sure(runtime) and action in VERIFY_ACTIONS:
        return Decision(action, True, "get_sure_open")
    coherence = _coherence(lock)
    if coherence and action == "done" and _done_refusal(runtime) is not None:
        return Decision(action, False, "open_counterexample")
    repair = _repair_on(lock)
    if action == "done" and not repair:
        return Decision(action, True, "done")
    if not lock.terminal_engaged:
        return Decision(action, True, "not_engaged")
    if repair:
        evidence = _repair_evidence(lock)
        if action == "done":
            snapshot = lock._readiness_snapshot()
            if snapshot is not None and not bool(getattr(snapshot, "blockers", ())):
                return Decision(action, True, "readiness_ready", None, evidence)
            if evidence:
                return Decision(action, False, "valid_counterexample", None, evidence)
            return Decision(action, True, "done_no_counterexample", None, evidence)
        if _linked(action, arguments, runtime):
            return Decision(action, True, "repair_linked", dict(arguments), evidence)
        return Decision(action, False,
                        "repair_unrelated" if evidence else "repair_post_green", None, evidence)
    if _integrity():
        return Decision(action, True, "frame_edit_integrity_recovery")
    if _relaxed():
        return Decision(action, True, "relaxed_profile")
    open_rows = _open_counterexamples(runtime) if coherence else ()
    if open_rows and action == "run_tests":
        return Decision(action, True, "open_counterexample_reverification")
    if coherence:
        if (action == "run_tests"
                and getattr(runtime, "stateframe_host_closure_mode", "off") == "on"):
            return Decision(action, True, "host_closure_recheck")
        override = getattr(lock, "_override_action_kind", None)
        if override is not None:
            if action == "run_tests" and getattr(lock, "_refused_action_kind", None) == "run_tests":
                return Decision(action, False, "recheck_menu_done_only")
            if _edit_class(action) == _edit_class(_norm(override)):
                return Decision(action, True, "coherent_override")
            return Decision(action, False, "narrowed_terminal_menu")
        return Decision(action, False, "coherent_refusal")
    if getattr(lock, "_first_action_overridden", False):
        return Decision(action, True, "post_override")
    first = getattr(lock, "_first_action_refused", None)
    if isinstance(first, str) and first.split(":", 1)[0] == action:
        return Decision(action, True, "first_refusal_override")
    return Decision(action, False, "first_refusal")


def _post_green_work(lock: object, name: str,
                     args: Mapping[str, object] | None = None,
                     *, for_menu: bool = False) -> Decision | None:
    """STATEFRAME_POST_GREEN_WORK_MENU rule (kernel_post_green_menu); None when off/inapplicable."""
    if getattr(lock.runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        return None
    from kernel_post_green_menu import decide as post_green_decide
    from kernel_v30_token_contracts import (
        admits_new_verification_binding,
        verification_selector_is_executable,
    )
    action = "edit" if name in EDIT_ACTIONS else name
    selector = args.get("selector", args.get("target")) if isinstance(args, Mapping) else None
    from kernel_named_layer_repair import enabled as named_layer_repair_enabled
    if named_layer_repair_enabled():
        # Dispatch carries the path: the named-layer rule admits only the untouched layer's files.
        path = args.get("path") if isinstance(args, Mapping) and args.get("path") not in (None, "") else None
        verdict = post_green_decide(lock, action, path=path, selector=selector,
                                    for_menu=for_menu)
    else:
        verdict = post_green_decide(lock, action, selector=selector, for_menu=for_menu)
    if (action == "run_tests"
            and admits_new_verification_binding(lock.runtime, args)
            and verification_selector_is_executable(lock.runtime, selector)):
        # A changed binding is executable even when the ordinary work window is
        # closed. Menu and dispatch use this exact selector witness.
        return Decision(name, True, "post_green_new_verification_binding")
    if (action == "run_tests" and admits_new_verification_binding(lock.runtime, args)
            and not verification_selector_is_executable(lock.runtime, selector)):
        return Decision(name, False, "post_green_unavailable_verification")
    if verdict is None:
        return None
    return Decision(name, verdict[0], verdict[1])


def _lock_decide(lock: object, name: str, args: Mapping[str, object], *,
                 for_menu: bool) -> Decision:
    # Q23 FIRST_EDIT: menu and dispatch share the same first-edit/run-tests rule.
    from kernel_greenlock_terminal_fixes import menu_rule
    first_edit = menu_rule(lock, name, args, for_menu=for_menu)
    if first_edit is not None:
        return Decision(name, True, first_edit)
    no_veto = _postgreen_no_veto(lock, name, args, for_menu=for_menu)
    if no_veto is not None:
        return no_veto
    post_green = _post_green_work(lock, name, args, for_menu=for_menu)
    if post_green is not None:
        return post_green
    if name in EDIT_ACTIONS:
        return _decide_edit(lock, name, dict(args), for_menu=for_menu)
    return _decide_other(lock, name, dict(args))


def decide(lock: object, action: object, arguments: Mapping[str, object] | None = None,
           *, for_menu: bool = False, authorized_action: object = None) -> Decision:
    """THE allow predicate for menu and dispatch.  Caller runs :func:`refresh` first.

    ``for_menu`` asks "is this tool usable with SOME arguments at this
    request"; edits then need a product-path witness (a repair target or a
    previously refused target) unless a path-independent rule allows them.
    """
    requested = _norm(action)
    name = canonical_action(requested)
    args = dict(arguments) if isinstance(arguments, Mapping) else {}
    from kernel_tool_surface import dispatch_admissible
    if not dispatch_admissible(name, runtime=lock.runtime):
        # The runner has no execution branch for this capability now (tool-surface
        # signal absent, ledger/closure mode off): never offer it, refuse it.
        # NO_VETO removes policy vetoes, not execution limits.
        return Decision(name, False, "runner_capability_unavailable")
    if name in OBSERVATION_TOOLS:
        return Decision(name, True, "observation_authority")
    if name in EDIT_ACTIONS and not _repro_first_edit_allowed(
            lock.runtime, args, for_menu=for_menu):
        return Decision(name, False, "repro_first_required")
    try:
        from kernel_post_green_menu import capability_floor_needed
        floor_needed = capability_floor_needed(lock.runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        floor_needed = False
    # BUILDFIX 10-18: the floor keeps product capabilities; it never grants done (a repair gate holding a
    # counterexample must still deny done - test_counterexample_b). done stays with the lock + readiness.
    floor_action = name in {
        "read", "grep", "edit", "run_tests", "read_check_receipt",
    }
    decision = (Decision(name, True, "capability_floor")
                if floor_needed and floor_action
                else _lock_decide(lock, name, args, for_menu=for_menu))
    decision = _intersect_readiness(lock, _intersect_continuation(
        lock, decision, requested_action=requested,
        authorized_action=authorized_action))
    # While the lock is NOT engaged the request's other menu policies (diet,
    # tool surface, readiness, budget) own the shape.  That shape is part of
    # this decision (recorded per request), so offered == allowed everywhere.
    upstream = getattr(lock.runtime, "_greenlock_menu_authority_upstream", None)
    quiet_pending_verification = False
    if name == "run_tests" and decision.allowed:
        try:
            from kernel_post_green_menu import (
                host_verified_done,
                hostverified_quiet_enabled,
                pending_verification,
            )
            quiet_pending_verification = (
                hostverified_quiet_enabled(lock.runtime)
                and host_verified_done(lock.runtime)
                and pending_verification(lock.runtime)
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            quiet_pending_verification = False
    if (decision.allowed and decision.rule != "capability_floor"
            and upstream is not None and not lock.terminal_engaged
            and name not in upstream and not quiet_pending_verification):
        decision = Decision(name, False, "upstream_menu_policy")
    if not decision.allowed:
        # Q23 NO_RED_TERMINAL: a tool reopened from a red empty menu must remain
        # dispatchable through the same authority for this request.
        from kernel_greenlock_terminal_fixes import red_reopen_rule
        reopened = red_reopen_rule(lock, name, decision.rule)
        if reopened is not None:
            return Decision(name, True, reopened, decision.witness, decision.evidence)
    return decision


# --------------------------------------------------------------------------
# Menu consumer
# --------------------------------------------------------------------------

def _emit(
    runtime: object,
    kind: str,
    *,
    producer: str = "kernel_greenlock_menu_authority",
    dispatch: bool = False,
    **fields: object,
) -> None:
    # 10-07 live crash (batches 904, 929-932 on L125c; 916 on L126b-min): callers pass dispatch=True, which
    # collided with the hard-coded dispatch=False ("got multiple values for keyword argument 'dispatch'").
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer=producer, dispatch=dispatch, **fields)


def _permitted_on_surface(lock: object, name: str) -> bool:
    """decide() for a menu query, ignoring only the per-request upstream narrowing."""
    from kernel_tool_surface import dispatch_admissible
    canonical = canonical_action(name)
    if canonical in OBSERVATION_TOOLS:
        return bool(canonical and dispatch_admissible(canonical, runtime=lock.runtime))
    decision = _intersect_readiness(lock, _intersect_continuation(
        lock, _lock_decide(lock, canonical, {}, for_menu=True), requested_action=name))
    return bool(canonical and dispatch_admissible(canonical, runtime=lock.runtime)
                and decision.allowed)


REPAIR_PATH_TOOLS = ("read", "probe")


def constrain_repair_paths(offered: Sequence[dict], decisions: Mapping[str, Decision]) -> list[dict]:
    """r45 one owner (r44c waiver W2): under the repair lock an offered read/probe carries ``path.enum`` = the repair
    targets -- the same evidence ``_linked`` checks at dispatch -- so the menu shows the lock's restriction and an
    off-target path is an argument-contract rejection, never a policy refusal of an offered tool.  Tools with any
    other rule, without a ``path`` property, or without target paths are returned unchanged (same object)."""
    out: list[dict] = []
    for tool in offered:
        name = _name(tool)
        decision = decisions.get(name)
        paths = (_evidence_paths(tuple(decision.evidence or ()))
                 if name in REPAIR_PATH_TOOLS and decision is not None and decision.rule == "repair_linked" else [])
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if not paths or not isinstance(properties, Mapping) or not isinstance(properties.get("path"), Mapping):
            out.append(tool)
            continue
        import copy as _copy
        constrained = _copy.deepcopy(tool)
        constrained["function"]["parameters"]["properties"]["path"]["enum"] = list(paths)
        out.append(constrained)
    return out


def menu_decisions(lock: object, surface: Sequence[Mapping[str, object]]) -> dict[str, Decision]:
    result: dict[str, Decision] = {}
    from kernel_v30_token_contracts import verification_menu_arguments
    for tool in surface:
        name = _name(tool)
        if name and name not in result:
            arguments = verification_menu_arguments(lock.runtime) if canonical_action(name) in VERIFY_ACTIONS else {}
            result[name] = decide(lock, name, arguments, for_menu=True)
    return result


@profiled_function_span("green_lock_menu_projection")
def project_request_menu(runtime: object, selected: Sequence[dict],
                         tools: Sequence[dict], *, boundary: bool = False) -> list[dict]:
    """Return the request menu == the authority's allow-set.  Off: ``selected`` unchanged."""
    lock = getattr(runtime, "green_lock", None)
    if not applies(lock):
        return list(selected) if not isinstance(selected, list) else selected
    setattr(runtime, "_greenlock_menu_authority_empty", None)
    if getattr(runtime, "_r47_red_menu_reopened", None) is not None:
        setattr(runtime, "_r47_red_menu_reopened", None)
    try:
        engaged = refresh(lock)
    except Exception as error:  # noqa: BLE001 - dispatch fails open on the same error
        _emit(runtime, MENU_EVENT, engaged=None, boundary=boundary, error_type=type(error).__name__,
              offered=[_name(tool) for tool in selected])
        return list(selected)
    # Full surface: the request tools plus any tool a mutator synthesised
    # (predict, job tools) that is not in the raw surface.
    surface: list[dict] = []
    seen: set[str] = set()
    for tool in [*selected, *tools]:
        name = _name(tool)
        if name and name not in seen:
            seen.add(name)
            surface.append(tool)
    previous = getattr(runtime, "_greenlock_menu_authority_engaged", None)
    setattr(runtime, "_greenlock_menu_authority_engaged", engaged)
    if boundary and previous is not None and previous != engaged:
        # Lock state changed after the upstream menu was built (e.g. terminal
        # fallback while rendering): that shape is stale, use the surface.
        selected = list(surface)
    upstream = None
    if not engaged:
        upstream_names = {canonical_action(_name(tool)) for tool in selected if _name(tool)}
        lock_allowed = {name for name in upstream_names if name and _permitted_on_surface(lock, name)}
        # An upstream shape with nothing the lock permits would strand the
        # model; then the lock's own allow-set is the menu (recovery).
        upstream = frozenset(upstream_names) if lock_allowed else None
    setattr(runtime, "_greenlock_menu_authority_upstream", upstream)
    decisions = menu_decisions(lock, surface)
    allowed = {name for name, decision in decisions.items() if decision.allowed}
    order = [*selected, *surface]
    offered, seen_names = [], set()
    for tool in order:
        name = _name(tool)
        if name in allowed and name not in seen_names:
            seen_names.add(name)
            offered.append(next(item for item in surface if _name(item) == name))
    offered = constrain_repair_paths(offered, decisions)
    if getattr(runtime, "stateframe_tokenlab_stall_breaker_mode", "off") == "on":
        from kernel_tokenlab_stall_breaker import apply_menu as apply_stall_menu
        offered = apply_stall_menu(runtime, offered)
    before_floor = {_name(tool) for tool in offered}
    from kernel_post_green_menu import apply_capability_floor
    offered = apply_capability_floor(runtime, offered, surface)
    for tool in offered:
        name = _name(tool)
        if name and name not in before_floor:
            decisions[name] = Decision(name, True, "capability_floor")
            allowed.add(name)
    names = [_name(tool) for tool in offered]
    before = [_name(tool) for tool in selected]
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") == "on":
        from kernel_post_green_menu import note_menu
        note_menu(runtime, names, {name: decision.rule for name, decision in decisions.items()})
    _emit(runtime, MENU_EVENT, engaged=engaged, boundary=boundary, offered=names,
          allowed=sorted(allowed), removed=[name for name in before if name not in names],
          added=[name for name in names if name not in before],
          rules={name: decision.rule for name, decision in sorted(decisions.items())})
    _drain_floor_kept(runtime)
    if not offered and upstream is not None:
        # review6: an upstream narrowing must never cause termination while the
        # full surface still has permitted tools; drop it and project the surface.
        setattr(runtime, "_greenlock_menu_authority_upstream", None)
        decisions = menu_decisions(lock, surface)
        offered = [tool for tool in surface if decisions[_name(tool)].allowed]
        names = [_name(tool) for tool in offered]
    if not offered and not any(_permitted_on_surface(lock, _name(tool)) for tool in surface):
        # Q23 NO_RED_TERMINAL: a non-green empty menu is recoverable; offer the
        # lock-level allow-set instead of creating a terminal outcome.
        from kernel_greenlock_terminal_fixes import reopen_empty_menu
        from kernel_tool_surface import dispatch_admissible
        reopened = reopen_empty_menu(
            lock, surface, _name,
            lambda name: bool(
                dispatch_admissible(canonical_action(name), runtime=runtime)
                and _lock_decide(lock, canonical_action(name), {}, for_menu=True).allowed))
        if reopened:
            offered = reopened
            names = [_name(tool) for tool in offered]
    if not offered and not any(_permitted_on_surface(lock, _name(tool)) for tool in surface):
        # Terminal only when decide() over the FULL surface permits nothing.
        setattr(runtime, "_greenlock_menu_authority_empty", {
            "engaged": engaged,
            "rules": {name: decision.rule for name, decision in sorted(decisions.items())},
        })
    # Freeze the exact serialized schema and allow-set that this request will
    # dispatch.  Preview calls only replace this snapshot; they never consume
    # post-green admission or charge a refusal.
    frozen_payload = {
        "schema": "stateframe-greenlock-final-menu/1",
        "allow_set": list(names),
        "tools": copy.deepcopy(offered),
    }
    frozen_bytes = json.dumps(
        frozen_payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    setattr(runtime, "_greenlock_frozen_menu", {
        **frozen_payload,
        "digest": "sha256:" + hashlib.sha256(frozen_bytes).hexdigest(),
    })
    # The notice/recovery contract reads this exact final projection.  Update
    # it even when the names happen to equal the upstream menu; otherwise a
    # stale menu can recommend a runner that is absent from this request.
    setattr(runtime, "completion_menu", tuple(names))
    setattr(runtime, "offered_tools", tuple(names))
    setattr(runtime, "_question_done_available", "done" in names)
    return offered


def finalize_wire_menu_authority(
        runtime: object, tools: Sequence[Mapping[str, object]], *,
        preserve_input_surface: bool = False,
) -> tuple[str, ...]:
    """Bind menu and dispatch authority to the exact tool contract sent to the provider.

    Late request transforms can alter the projected menu.  The prepared tools
    are the transport contract, so both the frozen dispatch check and the
    upstream policy snapshot must reflect that final contract. Boundary recovery
    may only refine its already admitted frozen menu; it cannot add routes.
    """
    lock = getattr(runtime, "green_lock", None)
    if not applies(lock):
        return ()

    from kernel_done_readiness_authority import (
        DoneReadinessContractError, assert_refusal_recovery, decision_for_runtime,
        enabled as done_authority_enabled, project_menu,
        uncertified_host_debt_enabled,
    )
    done_decision = None
    readiness_snapshot = getattr(lock, "_readiness_snapshot", None)
    evaluate_done = done_authority_enabled(runtime) or (
        uncertified_host_debt_enabled() and callable(readiness_snapshot)
    )
    if evaluate_done:
        done_decision = decision_for_runtime(
            runtime,
            readiness=readiness_snapshot() if callable(readiness_snapshot) else None,
            full_tools=getattr(runtime, "_completion_owner_full_tools", ()) or tools,
        )
    if done_decision is not None:
        from kernel_done_readiness_authority import abandon_decision
        full_tools = (getattr(runtime, "_completion_owner_full_tools", ()) or tools)
        from kernel_completion_owner import capability_floor_action_admissible
        try:
            projected = project_menu(
                done_decision, tools, full_tools,
                capability_admissible=lambda name: capability_floor_action_admissible(
                    runtime, name),
            )
            if preserve_input_surface:
                admitted_names = {_name(item) for item in tools if _name(item)}
                projected = [
                    item for item in projected if _name(item) in admitted_names
                ]
            if (not isinstance(tools, list)
                    and [_name(item) for item in projected] != [_name(item) for item in tools]):
                raise DoneReadinessContractError(
                    "final wire menu cannot satisfy the shared done decision")
            if done_decision.decision == "REFUSED":
                assert_refusal_recovery(done_decision, projected, full_tools)
        except (AssertionError, DoneReadinessContractError) as error:
            abandon_decision(runtime, done_decision, error)
            done_decision = None
        else:
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(projected)

    from kernel_post_green_menu import apply_capability_floor
    if isinstance(tools, list):
        full_tools = getattr(runtime, "_completion_owner_full_tools", ()) or tools
        tools[:] = apply_capability_floor(runtime, tools, full_tools)
        from kernel_done_is_submit_authority import (
            candidate_exists as done_candidate_exists,
            enabled as done_submit_enabled,
            project_candidate_tools,
        )
        if done_submit_enabled(runtime) and done_candidate_exists(runtime):
            tools[:] = project_candidate_tools(
                runtime, tools, full_tools=full_tools,
            )

    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        name = _name(tool)
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    frozen = getattr(runtime, "_greenlock_frozen_menu", None)
    if isinstance(frozen, Mapping):
        projected_names = {
            str(item) for item in frozen.get("allow_set", ()) if str(item).strip()
        }
    else:
        projected_names = set()
    final_names = set(names)
    if projected_names != final_names:
        _emit(
            runtime,
            "menu_authority_wire_mismatch",
            menu_authority=sorted(projected_names),
            final_wire=sorted(final_names),
            dispatch=False,
        )

    # Keep aliases in the frozen request contract, while policy dispatch uses
    # the same canonical action identities as decide().
    payload = {
        "schema": "stateframe-greenlock-final-menu/1",
        "allow_set": list(names),
        "tools": copy.deepcopy([dict(tool) for tool in tools if isinstance(tool, Mapping)]),
    }
    if done_decision is not None:
        payload["done_decision_key"] = list(done_decision.key)
    frozen_bytes = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    setattr(runtime, "_greenlock_frozen_menu", {
        **payload,
        "digest": "sha256:" + hashlib.sha256(frozen_bytes).hexdigest(),
    })
    setattr(runtime, "_greenlock_menu_authority_upstream", frozenset(
        canonical_action(name) for name in names if canonical_action(name)
    ))
    setattr(runtime, "completion_menu", tuple(names))
    setattr(runtime, "offered_tools", tuple(names))
    setattr(runtime, "_question_done_available", "done" in names)
    return tuple(names)


def empty_menu_outcome(runtime: object) -> dict[str, object] | None:
    """Settle an empty menu only in legacy mode; hard-only mode delivers an advisory."""
    record = getattr(runtime, "_greenlock_menu_authority_empty", None)
    if not isinstance(record, Mapping):
        return None
    lock = getattr(runtime, "green_lock", None)
    if not applies(lock):
        return None
    from kernel_terminal_policy import (
        enabled as hard_budget_only_enabled,
        queue_advisory,
    )
    if hard_budget_only_enabled(runtime):
        queue_advisory(runtime, "greenlock_empty_menu")
        _emit(runtime, EMPTY_EVENT, outcome=None, engaged=record.get("engaged"),
              rules=dict(record.get("rules", {})), advisory=True)
        return None
    if not getattr(runtime, "system_outcome", None):
        setattr(runtime, "system_outcome", EMPTY_OUTCOME)
    _emit(runtime, EMPTY_EVENT, outcome=getattr(runtime, "system_outcome", None),
          engaged=record.get("engaged"), rules=dict(record.get("rules", {})))
    return {"kind": EMPTY_OVERFLOW_KIND, "outcome": getattr(runtime, "system_outcome", None),
            "rules": dict(record.get("rules", {}))}


# --------------------------------------------------------------------------
# Dispatch consumers (the lock's hooks)
# --------------------------------------------------------------------------

def _lock_emit(lock: object, kind: str, **fields: object) -> None:
    lock._emit(kind, authority=True, **fields)


def _post_green(lock: object, action: str, refusal: str, *, readiness: bool = False) -> str:
    if lock._record_post_green_refusal(action):
        stopped = lock._repair_blocks_done_refusal()
        if readiness:
            return stopped or lock._readiness_refusal() or refusal
        return stopped or refusal
    return refusal


def dispatch_action(lock: object, kind: object,
                    arguments: Mapping[str, object] | None = None,
                    *, attempt_id: str | None = None) -> str | None:
    """Authority-mode body of ``GreenLock.before_action``."""
    requested = _norm(kind)
    action = canonical_action(requested)
    args = dict(arguments) if isinstance(arguments, Mapping) else {}
    runtime = lock.runtime
    supplied_attempt = attempt_id or args.pop("_attempt_id", None)
    attempt = str(supplied_attempt or getattr(runtime, "_greenlock_attempt_id", "")).strip()
    attempt_digest = ""
    if attempt:
        attempt_digest = hashlib.sha256(json.dumps(
            {"attempt_id": attempt, "action": requested, "arguments": args},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()
        cache = getattr(runtime, "_greenlock_dispatch_attempts", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(runtime, "_greenlock_dispatch_attempts", cache)
        previous = cache.get(attempt_digest)
        if isinstance(previous, Mapping):
            _emit(runtime, "green_lock_dispatch_idempotent_replay",
                  attempt_id=attempt, action_kind=action,
                  decision=previous.get("decision"), dispatch=False)
            return previous.get("refusal")

    def remember(refusal: str | None, decision: str) -> str | None:
        if attempt_digest:
            cache[attempt_digest] = {
                "decision": decision,
                "allowed": refusal is None,
                "refusal": refusal,
            }
        return refusal

    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    if requested == "done" and done_submit_enabled(runtime):
        _emit(runtime, "done_submit_dispatch_admitted",
              attempt_id=attempt or None, action_kind=action, dispatch=False)
        return remember(None, "done_is_submit")

    done_readiness = None
    if requested == "done":
        from kernel_done_readiness_authority import (
            assert_refusal_recovery, decision_for_runtime,
            enabled as done_authority_enabled, record_done_decision, render_done_reason,
            uncertified_host_debt_enabled,
        )
        readiness_snapshot = getattr(lock, "_readiness_snapshot", None)
        evaluate_done = done_authority_enabled(runtime) or (
            uncertified_host_debt_enabled() and callable(readiness_snapshot)
        )
        if evaluate_done:
            done_readiness = decision_for_runtime(
                runtime,
                readiness=readiness_snapshot() if callable(readiness_snapshot) else None,
                full_tools=getattr(runtime, "_completion_owner_full_tools", ()),
            )

    frozen = getattr(runtime, "_greenlock_frozen_menu", None)
    if isinstance(frozen, Mapping):
        allow_set = {
            str(item) for item in frozen.get("allow_set", ())
            if str(item).strip()
        }
        # The frozen wire allow-set contains the exact requested tool names.
        # Canonical aliases remain one policy identity, but an alias absent
        # from this request must not bypass the serialized menu.
        if requested not in allow_set:
            if (done_readiness is not None
                    and done_readiness.decision == "REFUSED"):
                current_menu = frozen.get("tools", ())
                full_tools = (getattr(runtime, "_completion_owner_full_tools", ())
                              or current_menu)
                from kernel_done_readiness_authority import (
                    DoneReadinessContractError, abandon_decision,
                )
                try:
                    assert_refusal_recovery(done_readiness, current_menu, full_tools)
                except (AssertionError, DoneReadinessContractError) as error:
                    abandon_decision(runtime, done_readiness, error)
                else:
                    record_done_decision(runtime, done_readiness)
                    refusal = render_done_reason(done_readiness)
                    _lock_emit(lock, "refused", action_kind=action,
                               product_revision=lock._green_revision,
                               reason="done_readiness_refused")
                    return remember(refusal, "done_readiness_refused")
            refusal = UPSTREAM_REFUSAL.format(action=action)
            if attempt_digest:
                cache[attempt_digest] = {
                    "decision": "frozen_menu_refusal", "refusal": refusal,
                }
            _emit(runtime, "green_lock_dispatch_refused",
                  attempt_id=attempt or None, action_kind=action,
                  reason="frozen_final_menu", dispatch=True)
            return refusal
    if requested in EDIT_ACTIONS:
        refusal = dispatch_mutation(lock, args.get("path", ""), action_kind=requested)
        if attempt_digest:
            cache[attempt_digest] = {"decision": "mutation", "refusal": refusal}
        return refusal
    try:
        refresh(lock)
    except Exception:  # noqa: BLE001 - mirrors the menu's fail-open on the same error
        return remember(None, "refresh_error")
    dispatch_admission = None
    try:
        from kernel_post_green_menu import capability_floor_needed
        floor_needed = capability_floor_needed(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        floor_needed = False
    if action in VERIFY_ACTIONS and not floor_needed:
        try:
            from kernel_postgreen_protocol import consume_admission
            dispatch_admission = consume_admission(
                runtime, action, args, action_id=attempt or None,
            )
        except (ImportError, AttributeError, TypeError, ValueError):
            dispatch_admission = None
    decision = decide(lock, requested, args)
    if requested == "done":
        if (done_readiness is not None
                and done_readiness.decision == "REFUSED"):
            record_done_decision(runtime, done_readiness)
            if (decision.allowed
                    or decision.rule in {"readiness_disposition", "open_counterexample"}):
                decision = Decision(
                    action, False, "done_readiness_refused", decision.witness,
                    decision.evidence, render_done_reason(done_readiness),
                )
        elif (done_readiness is not None
              and done_readiness.decision == "ACCEPTED_UNCERTIFIED"
                  and decision.rule == "readiness_disposition"):
            # The shared readiness decision is the sole owner of this gate:
            # green current evidence with no callable recovery submits once
            # as uncertified instead of replaying the old readiness veto.
            decision = Decision(action, True, "done_uncertified")
    if (dispatch_admission is not None
            and getattr(dispatch_admission, "allowed", True) is False):
        # The post-green protocol is consumed only at this real dispatch
        # boundary; preview already used for_menu=True and cannot reach here.
        decision = Decision(
            action, False,
            getattr(dispatch_admission, "message", "") or
            getattr(dispatch_admission, "rule", "postgreen_refusal"),
        )
    revision = lock._green_revision
    if attempt_digest:
        cache[attempt_digest] = {
            "decision": decision.rule if decision is not None else "none",
            "allowed": decision.allowed if decision is not None else None,
            "refusal": None,
        }
    if decision.allowed:
        if action in {"done", "submit"}:
            try:
                from kernel_postgreen_protocol import note_terminal_action
                note_terminal_action(runtime, action)
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
        if decision.rule == "unverified_checks_window":
            try:
                from kernel_live_counterexample_challenge import note_window_action
                used = note_window_action(runtime)
            except (ImportError, AttributeError, TypeError, ValueError):
                used = None
            _lock_emit(lock, "action_allowed", action_kind=action, product_revision=revision,
                       reason=decision.rule, window_actions=used)
        elif decision.rule == "repair_linked":
            _lock_emit(lock, "repair_allowed", action_kind=action, product_revision=revision,
                       reason="OBLIGATION_LINKED", evidence_count=len(decision.evidence))
            try:
                from kernel_repair_blocks_done import reset_progress
                reset_progress(runtime)
            except (ImportError, TypeError, ValueError):
                pass
        elif decision.rule == "first_refusal_override":
            lock._first_action_overridden = True
            _lock_emit(lock, "override", action_kind=action, product_revision=revision)
        elif decision.rule == "coherent_override":
            _lock_emit(lock, "override", action_kind=action, product_revision=revision,
                       reason="narrowed_terminal_menu")
        elif decision.rule not in {"done", "not_engaged", "get_sure_open"}:
            _lock_emit(lock, "action_allowed", action_kind=action, product_revision=revision,
                       reason=decision.rule)
        return None
    rule = decision.rule
    if rule == "done_readiness_refused" and done_readiness is not None:
        return remember(decision.message or render_done_reason(done_readiness), rule)
    if rule == "stopped":
        return remember(lock._repair_blocks_done_refusal(), rule)
    if rule == "postgreen_no_veto_done_refused":
        _lock_emit(lock, "refused", action_kind=action, product_revision=revision,
                   reason=rule)
        return remember(decision.message or UPSTREAM_REFUSAL.format(action=action), rule)
    if rule == "readiness_disposition":
        _lock_emit(lock, "refused", action_kind=action, product_revision=revision, reason=rule)
        return remember(lock._readiness_refusal() or UPSTREAM_REFUSAL.format(action=action), rule)
    if rule in {"upstream_menu_policy", "runner_capability_unavailable"}:
        _lock_emit(lock, "refused", action_kind=action, product_revision=revision, reason=rule)
        return remember(UPSTREAM_REFUSAL.format(action=action), rule)
    if rule == "open_counterexample":
        refusal = _post_green(lock, action, _done_refusal(runtime) or lock._terminal_refusal())
    elif rule == "valid_counterexample":
        from kernel_green_lock_repair import repair_refusal
        refusal = _post_green(lock, action, repair_refusal(action, decision.evidence))
    elif rule in {"repair_unrelated", "repair_post_green"}:
        from kernel_green_lock_repair import repair_refusal
        refusal = (repair_refusal(action, decision.evidence)
                   if decision.evidence else lock._terminal_refusal())
        refusal = _post_green(lock, action, refusal, readiness=True)
    elif rule == "coherent_refusal":
        lock._register_coherent_refusal(action)
        lock._first_action_refused = lock._action_identity(action, args)
        refusal = lock._terminal_refusal(action)
    elif rule == "recheck_menu_done_only":
        refusal = lock._terminal_refusal(action)
    elif rule == "post_green_edit_unverified":
        from kernel_v30_token_contracts import refusal_with_route
        refusal = refusal_with_route(
            "done refused: a product edit was made after public tests passed and has not been re-tested yet",
            "a qualifying test receipt for the edited binding",
            runtime,
        )
    elif rule == "postgreen_no_new_evidence":
        from kernel_v30_token_contracts import refusal_with_route
        refusal = refusal_with_route(
            "verification refused: current host checks already pass for this revision",
            "a changed verification binding or a still-open host obligation",
            runtime,
        )
    elif rule == "post_green_unavailable_verification":
        from kernel_v30_token_contracts import refusal_with_route
        refusal = refusal_with_route(
            "verification refused: the requested selector is not an authorized executable check",
            "an authorized local verification selector",
            runtime,
        )
    elif rule in {"postgreen_identical_verification", "postgreen_certified", "postgreen_uncertified"}:
        try:
            from kernel_postgreen_protocol import last_message
            refusal = last_message(runtime) or (
                "POSTGREEN_IDENTICAL_VERIFICATION: repeated verification refused; "
                "use the funded next action shown by the host.")
        except (ImportError, AttributeError, TypeError, ValueError):
            refusal = "POSTGREEN_IDENTICAL_VERIFICATION: repeated verification refused"
    elif rule == "first_refusal":
        if lock._first_action_refused is None:
            lock._first_action_refused = lock._action_identity(action, args)
        refusal = lock._terminal_refusal(action)
    else:
        refusal = lock._terminal_refusal()
    _lock_emit(lock, "refused", action_kind=action, product_revision=revision, reason=rule)
    return remember(refusal, rule)


def dispatch_mutation(lock: object, path: object, *, action_kind: str = "edit") -> str | None:
    """Authority-mode body of ``GreenLock.before_mutation`` (terminal mode)."""
    from kernel_green_lock import REFUSAL_SUFFIX, is_product_path
    runtime = lock.runtime
    requested = _norm(action_kind) or "edit"
    authorized_action = None
    continuation = getattr(runtime, "continuation", None)
    authorized_getter = getattr(continuation, "authorized_write_action", None)
    if callable(authorized_getter):
        try:
            authorized_action = authorized_getter(path, action_kind=requested)
        except (AttributeError, OSError, TypeError, ValueError):
            authorized_action = None
    # The continuation getter has validated the exact fence and carries the
    # original alias across the runner's path-only write guard.  Green-lock
    # predicates still see the canonical edit class.
    requested = _norm(authorized_action) or requested
    action = canonical_action(requested) or "edit"
    try:
        refresh(lock)
    except Exception:  # noqa: BLE001 - mirrors the menu's fail-open on the same error
        return None
    target = lock._workspace_path(path)
    decision = decide(lock, requested, {"path": path},
                      authorized_action=authorized_action)
    revision = lock._green_revision
    if decision.allowed:
        if decision.rule == "repair_linked":
            try:
                from kernel_green_lock_repair import controller_for
                controller = controller_for(runtime)
                if controller is not None:
                    controller.capture_before(path)
            except (ImportError, TypeError, ValueError):
                pass
        if decision.rule == "repair_linked":
            _lock_emit(lock, "repair_allowed", path=target, action_kind=action,
                       product_revision=revision, reason="OBLIGATION_LINKED")
            try:
                from kernel_repair_blocks_done import reset_progress
                reset_progress(runtime)
            except (ImportError, TypeError, ValueError):
                pass
        elif decision.rule == "target_override":
            if _coherence(lock):
                lock._register_coherent_refusal(action)
            lock._overridden_targets.add(target)
            _lock_emit(lock, "override", path=target, action_kind=action,
                       product_revision=revision)
        elif decision.rule == "coherent_override":
            _lock_emit(lock, "override", path=target, action_kind=action,
                       product_revision=revision, reason="narrowed_terminal_menu")
        elif decision.rule in {"frame_edit_integrity_recovery", "open_counterexample_repair"}:
            _lock_emit(lock, "action_allowed", path=target, action_kind=action,
                       product_revision=revision, reason=decision.rule)
        return None
    rule = decision.rule
    if rule == "stopped":
        return lock._repair_blocks_done_refusal()
    if rule == "upstream_menu_policy":
        _lock_emit(lock, "refused", path=target, action_kind=action, product_revision=revision,
                   reason=rule)
        return UPSTREAM_REFUSAL.format(action=action)
    if rule in {"repair_unrelated", "repair_post_green"}:
        from kernel_green_lock_repair import repair_refusal
        refusal = (repair_refusal(action, decision.evidence) if decision.evidence
                   else (lock._readiness_refusal() or lock._green_fact() + REFUSAL_SUFFIX))
        refusal = _post_green(lock, action, refusal, readiness=True)
    else:
        if rule in {"coherent_refusal", "first_refusal"} and is_product_path(target):
            if rule == "coherent_refusal":
                lock._register_coherent_refusal(action)
                lock._first_action_refused = lock._action_identity(action, {"path": target})
            elif lock._first_action_refused is None:
                lock._first_action_refused = lock._action_identity(action, {"path": target})
            lock._refused_targets.add(target)
        refusal = lock._green_fact() + REFUSAL_SUFFIX
    _lock_emit(lock, "refused", path=target, action_kind=action, product_revision=revision,
               reason=rule)
    return refusal


__all__ = [
    "Decision", "EDIT_ACTIONS", "EMPTY_EVENT", "EMPTY_OUTCOME", "EMPTY_OVERFLOW_KIND",
    "MENU_EVENT", "MODE_ENV", "applies", "decide", "dispatch_action", "dispatch_mutation",
    "empty_menu_outcome", "enabled", "menu_decisions", "mode", "project_request_menu", "refresh",
    "finalize_wire_menu_authority",
]
