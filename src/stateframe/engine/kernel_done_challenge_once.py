"""Challenge-once-per-revision done policy for the enforced (frame) arm.

r39 dissection: requirement alignment refused ``done`` ("no repair action is
available") 46-51 times in a row while the green-lock terminal menu offered
ONLY ``done`` -- a deadlock that burned ~1.3M tokens.  This module is the
single post-reducer arbiter that makes that state impossible:

* A soft (policy) refusal of an otherwise admissible ``done`` is a CHALLENGE.
  The first ``done`` at a product revision may be refused; the second ``done``
  at the SAME revision is ACCEPTED and journaled as challenged-and-overridden.
  A new revision opens a new one-challenge window.
* ``repair_blocks_done`` keeps its stricter semantics ONLY while a repair
  action (edit/str_replace) is actually on the offered menu; that path is
  bounded by its own K-response STOPPED_INCOMPLETE stop.
* If the offered menu contains nothing but ``done``, a refusal of ANY origin is
  a challenge (once per revision): no state may refuse every offered action.
* A refusal of a ``done`` the caller itself marked inadmissible (e.g. public
  tests red) with a non-trivial menu is left to the normal path.
* Q23 CAP1 is subordinate to actionable evidence: a runnable behavior witness
  or missing current-revision public checks keeps ``done`` refused until that
  evidence is settled. Only an advisory or an explicitly unrunnable witness
  can reach the second-done cap.

Switch: ``STATEFRAME_DONE_CHALLENGE_ONCE=off|on`` (default off; off returns the
reducer decision object unchanged).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_DONE_CHALLENGE_ONCE"
OVERRIDE_CODE = "DONE_CHALLENGE_OVERRIDDEN"
CHALLENGE_SUFFIX = ("[done challenge] This done was challenged once at this revision. "
                    "Repair if the challenge is right; a second done at this same "
                    "revision is accepted.")
NO_AGREE_TEXT = "Do not change correct code merely to agree with this challenge."
# attempt assessment: run-wide cap that edits cannot replenish (done -> cosmetic edit ->
# new revision -> challenge treadmill).
CHALLENGE_RUN_CAP = 2
# r44 fix DONE: the shared done-path vocabulary lives in kernel_completion_policy (one list).
from kernel_completion_policy import NON_WAIVABLE_CODES, TERMINAL_CODES  # noqa: E402

_COUNTEREXAMPLE_CODES = frozenset({
    "REQUIREMENT_ALIGNMENT_OPEN", "READINESS_COUNTEREXAMPLE_OPEN", "GET_SURE_OPEN_CLAUSE",
    "OPEN_COUNTEREXAMPLE", "GREEN_LOCK_REPAIR_OPEN_COUNTEREXAMPLE", *NON_WAIVABLE_CODES})
# Unified stop (kernel_completion_policy.CODE_VIOLATED) owns its challenge/stop ledger.
_UNIFIED_OWNED_CODE = "UNIFIED_VIOLATED"
# r41b (kernel_live_counterexample_challenge.UNVERIFIED_CODE): challenge class UNVERIFIED, strict
# first-green progress rule.  Only ever produced with STATEFRAME_LIVE_CE_CHALLENGE=on.
_STRICT_UNVERIFIED_CODES = frozenset({"UNVERIFIED_DISCLOSED_CHECKS"})
CLASS_TEXT = {
    "COUNTEREXAMPLE": "COUNTEREXAMPLE: a host check produced the witness quoted above.",
    "UNVERIFIED": ("UNVERIFIED: not a demonstrated failure; check before editing."),
    "CHECK_INCONCLUSIVE": ("CHECK_INCONCLUSIVE: a check or binding problem, not a product "
                           "failure."),
}


def challenge_class(codes: object) -> str:
    values = {str(code) for code in (codes or ())}
    if any("SIGNATURE_GAP" in code for code in values):
        return "COUNTEREXAMPLE"
    if any("INCONCLUSIVE" in code or "INVALID" in code or "UNBOUND" in code
           or "SIGNATURE_MISMATCH" in code for code in values):
        return "CHECK_INCONCLUSIVE"
    if values & _COUNTEREXAMPLE_CODES:
        return "COUNTEREXAMPLE"
    return "UNVERIFIED"


# Mutating repair actions: a menu offering one of these can make progress on a
# repair-blocked row.  run_tests alone cannot fix a counterexample.
REPAIR_EDIT_TOOLS = frozenset({"edit", "str_replace", "write", "file_create", "replace",
                               "replace_module"})
QUALIFYING_ACTIONS = frozenset({
    *REPAIR_EDIT_TOOLS, "read", "grep", "find", "search", "probe", "predict",
})
# V2_TERMINAL (disposition-v2 INCOMPLETE, r44 completion_gaps F2) is a non-accepted terminal
# of the shared stop decision: the arbiter must never convert it into a submission.
_TERMINAL_CODES = TERMINAL_CODES


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def menu_names(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def done_only(menu: Sequence[str]) -> bool:
    names = {str(item).strip().lower() for item in menu if str(item).strip()}
    return bool(names) and names <= {"done"}


def has_qualifying_action(menu: Sequence[str]) -> bool:
    """Whether the offered menu contains an action that can earn FGD progress.

    ``run_tests`` is intentionally absent: repeating the same green check is
    evidence reuse, not a source edit, fresh read, or new probe.
    """
    names = {str(item).strip().lower() for item in menu if str(item).strip()}
    return bool(names & QUALIFYING_ACTIONS)


def repair_hold(runtime: object, menu: Sequence[str]) -> bool:
    """repair_blocks_done may keep refusing only while a repair edit is offered."""
    if getattr(runtime, "stateframe_repair_blocks_done_mode", "off") != "on":
        return False
    readiness = (getattr(runtime, "_readiness_decision", None)
                 or getattr(runtime, "_readiness_request_decision", None))
    if not bool(getattr(readiness, "repair_blocks_done_active", False)):
        return False
    return bool(REPAIR_EDIT_TOOLS & {str(item).strip().lower() for item in menu})


def revision_key(basis: object, runtime: object = None) -> str:
    """The PRODUCT revision of a done basis.  Measured on the r39 deadlock journals:
    ``snapshot_digest`` changes on every done even with zero edits (it carries host
    evidence), so keying on it would make every done a "new revision" and the
    challenge would never be overridden.  ``binding_digest`` is the candidate
    product binding: stable across edit-free dones, changed by a real edit."""
    if runtime is not None:
        try:
            from kernel_challenge_budget import artifact_revision, budget_for
            if budget_for(runtime) is not None:
                return artifact_revision(runtime, basis=basis)
        except (ImportError, OSError, TypeError, ValueError):
            pass
    binding = str(getattr(basis, "binding_digest", "") or "")
    return binding or str(getattr(basis, "snapshot_digest", "") or "")


def arbitrate(runtime: object, decision, *, caller_admissible: bool, rebuild_accepted):
    """Return the final decision under challenge-once.  ``rebuild_accepted`` is a
    zero-argument callable producing the accepted reducer decision for override."""
    from kernel_done_is_submit_authority import accepted_decision, done_sealed

    if done_sealed(runtime) is not None:
        return accepted_decision(runtime) or decision
    first_green = _first_green_enabled(runtime)
    if (getattr(runtime, "stateframe_done_challenge_once_mode", "off") != "on"
            and not first_green):
        return decision
    budget = None
    budget_module = None
    try:
        import kernel_challenge_budget as budget_module
        budget = budget_module.budget_for(runtime)
    except (ImportError, TypeError, ValueError):
        budget = None

    # A durable terminal result is an absorbing boundary. A pending keyed
    # submission must also be reconciled before the accepted fast path; otherwise
    # a fresh reducer acceptance can overwrite its recovered provenance as a
    # natural acceptance and skip destination reconciliation.
    key = revision_key(getattr(decision, "basis", None), runtime)
    submission_plan = (budget_module.plan(runtime, key, request=None)
                       if budget is not None else None)
    if (isinstance(submission_plan, Mapping)
            and submission_plan.get("action") in {"terminal", "terminal_replay"}):
        durable = budget_module.durable_decision(
            runtime, fallback_basis=getattr(decision, "basis", None))
        return durable if durable is not None else decision
    recovering_submission = (
        isinstance(submission_plan, Mapping)
        and submission_plan.get("action") in {"submit", "cap", "reconcile", "replay"}
    )

    if getattr(decision, "accepted", False) and not recovering_submission:
        if budget is not None:
            rows = budget_module.open_clause_rows(decision, runtime=runtime)
            ids = budget_module.clause_ids(decision, rows=rows, runtime=runtime)
            klass = budget_module.challenge_class((), rows=rows)
            assessment = budget_module.assessment_for(klass, ids, rows=rows)
            if not ids and assessment == "no_known_violation":
                klass = "NONE"
            record = budget_module.finalize_accepted(
                runtime, decision, revision=key, assessment=assessment,
                open_clause_ids=ids, challenge_class=klass, request=None,
                override_reason="natural_acceptance")
            if record is not None and record.get("override_reason") == "natural_acceptance":
                _emit(runtime, "done_submission_record", **record)
        return decision
    codes = set(getattr(decision, "reason_codes", ()) or ())
    if codes & _TERMINAL_CODES:
        return decision
    if "PUBLIC_RED_DONE" in codes:
        # Q23 NO_RED_TERMINAL owns this refusal; CAP1 must never waive it.
        return decision
    if codes & NON_WAIVABLE_CODES:
        # r44 fix DONE/D2: a witnessed failure (e.g. N14 import FAIL) is never waived by the
        # challenge cap / sole-done submission; it is refused, then bounded by a terminal.
        return witnessed_failure_bound(runtime, decision)
    if _UNIFIED_OWNED_CODE in codes:
        # r44 fix DONE (consult item 3): unified stop already challenged/stopped this done on
        # its own snapshot ledger; a second challenge here would spend budget twice and
        # promise an override (CHALLENGE_SUFFIX) unified stop will not honour.
        _emit(runtime, "done_challenge_deferred", owner="unified_stop",
              reason_codes=sorted(codes))
        return decision
    menu = menu_names(runtime)
    if (not caller_admissible and not done_only(menu)
            and not (first_green and codes & _COUNTEREXAMPLE_CODES)):
        return decision
    # When both presentation arms are enabled, H+state+ has already applied
    # the shared budget before the outer frame arbiter sees its refusal.  Do
    # not turn one done into two budget transitions; the next done re-enters
    # H+state+ and submits at the same challenged revision.
    if "HSTATE_PLUS_DONE_CHALLENGE" in codes:
        try:
            from kernel_challenge_budget import budget_for
            if budget_for(runtime) is not None:
                return decision
        except (ImportError, TypeError, ValueError):
            pass
    # Q23 CAP1: unverified-only refusals are advisory, and other waivable
    # challenges are accepted on the second done at the same revision.
    from kernel_completion_policy import NEVER_ADMIT_UNCERTIFIED_CODES
    from kernel_greenlock_terminal_fixes import cap_done
    cap_key = revision_key(getattr(decision, "basis", None), runtime)
    if getattr(decision, "accepted", False) and recovering_submission:
        capped = None
    else:
        capped = cap_done(
            runtime, decision, key=cap_key,
            non_waivable=(frozenset(NON_WAIVABLE_CODES) | frozenset(_TERMINAL_CODES)
                          | frozenset(NEVER_ADMIT_UNCERTIFIED_CODES)),
            rebuild_accepted=rebuild_accepted,
        )
    if capped is not None:
        if getattr(capped, "accepted", False):
            _commit_capped_submission(runtime, decision, capped, key=cap_key,
                                      codes=codes, menu=menu)
        return capped
    try:
        from kernel_challenge_budget import (
            CHALLENGE_ENDING, assessment_for, budget_for, challenge_class as budget_class,
            clause_ids, commit_challenge, commit_submission, open_clause_rows, plan,
            remember_decision, replay_decision,
        )
        budget = budget_for(runtime)
    except (ImportError, TypeError, ValueError):
        # The legacy arm remains self-contained if this optional module is not
        # present in an older merged runtime.
        CHALLENGE_ENDING = ""
        assessment_for = budget_for = clause_ids = commit_challenge = commit_submission = None
        plan = remember_decision = replay_decision = None
        budget_class = None
    key = revision_key(getattr(decision, "basis", None), runtime)
    spent = int(getattr(runtime, "_done_challenge_count", 0) or 0)
    same_revision = getattr(runtime, "_done_challenge_revision", None) == key
    # r41b: disclosed-but-host-demoted checks (UNVERIFIED class) follow the same first-green
    # rules as a counterexample: challenge once, second done after one non-redundant action.
    strict_counterexample = bool(first_green and codes & (_COUNTEREXAMPLE_CODES | _STRICT_UNVERIFIED_CODES))
    if strict_counterexample:
        from kernel_first_green_disclosure import progress_since_challenge
        progressed = progress_since_challenge(runtime)
        cap_reached = spent >= CHALLENGE_RUN_CAP
    else:
        progressed = False
        cap_reached = spent >= CHALLENGE_RUN_CAP and not same_revision
    klass = challenge_class(codes)
    # r41 composition (first_green_disclosure x challenge_budget), trials/r41_resolutions.md:
    # when the budget is active its durable counter/cap/challenged revision are the ONE
    # authority; the first-green gates read them instead of the legacy attributes.
    budget_view = plan(runtime, key, request=None) if budget is not None else None
    # STATEFRAME_SUBMIT_IDEMPOTENT: a durable submission intent at this revision
    # means the submit decision was already taken; later dones only reconcile it.
    reconciling = budget_view is not None and budget_view.get("action") == "reconcile"
    if budget_view is not None and budget_view.get("action") not in {"replay", "terminal", "terminal_replay"}:
        # max(): strict repair-hold refusals count on the legacy counter (a hold at an
        # already-challenged revision spends no budget), and that count must still reach
        # the cap -- otherwise a persistent hold is an unbounded refusal loop.
        spent = max(spent, int(budget_view.get("spent", 0)))
        run_cap = int(budget_view.get("cap", CHALLENGE_RUN_CAP))
        same_revision = budget_view.get("action") in {"submit", "reconcile"}
        cap_reached = (spent >= run_cap) if strict_counterexample else (
            spent >= run_cap and not same_revision)

    if repair_hold(runtime, menu) and not (first_green and cap_reached) and not reconciling:
        # The repair-hold path used to return before spending a first-green
        # challenge.  In a persistent hold that made the run-wide cap
        # unreachable.  Count each bounded refusal in this strict path so the
        # same cap that protects new revisions also provides a finite escape.
        if strict_counterexample:
            if budget_view is not None and budget_view.get("action") == "challenge":
                commit_challenge(runtime, budget_view, challenge_class=klass)
            setattr(runtime, "_done_challenge_revision", key)
            setattr(runtime, "_done_challenge_count", spent + 1)
            if not same_revision:
                from kernel_first_green_disclosure import begin_challenge
                begin_challenge(runtime, key)
            _emit(runtime, "done_challenge_once", revision=key,
                  reason_codes=sorted(codes), challenge_class=klass,
                  run_challenge_count=spent + 1,
                  run_challenge_cap=CHALLENGE_RUN_CAP, menu=list(menu),
                  done_only_menu=done_only(menu), repair_hold=True)
        else:
            _emit(runtime, "done_challenge_repair_hold", menu=list(menu),
                  reason_codes=sorted(codes))
        return decision

    no_qualifying_action = not has_qualifying_action(menu)
    if reconciling:
        # The submission already started at this revision; waiting for another
        # action would strand it (STATEFRAME_SUBMIT_IDEMPOTENT only).
        progressed = True
    if (strict_counterexample and same_revision and not progressed and not cap_reached
            and not no_qualifying_action):
        _emit(runtime, "done_challenge_waiting_for_action", revision=key,
              reason_codes=sorted(codes), challenge_class=klass, menu=list(menu),
              counterexample_ids=_counterexample_ids(runtime, codes))
        detail = str(getattr(decision, "detail", "") or "")
        suffix = _strict_waiting_suffix(runtime, codes)
        try:
            import dataclasses
            return dataclasses.replace(
                decision, detail=(detail + "\n" if detail else "") + suffix)
        except (TypeError, ValueError):
            return decision
    evidence = open_clause_rows(decision, runtime=runtime) if budget is not None else ()
    budget_plan = plan(runtime, key, request=None) if budget is not None else None
    if budget_plan is not None and budget_plan.get("action") == "replay":
        cached = replay_decision(runtime, budget_plan)
        if cached is not None:
            return cached
        # A process restart cannot resume a terminal done request.  For a
        # transport retry in the same request, returning the incoming refusal
        # is the conservative replay result and does not spend a challenge.
        if budget_plan.get("previous") == "challenge":
            detail = str(getattr(decision, "detail", "") or "")
            replay_class = (budget_class(codes, rows=evidence)
                            if budget_class is not None else klass)
            suffix = f"{CLASS_TEXT[replay_class]} {CHALLENGE_ENDING}"
            try:
                import dataclasses
                return dataclasses.replace(
                    decision, detail=(detail + "\n" if detail else "") + suffix)
            except (TypeError, ValueError):
                return decision
        accepted = rebuild_accepted()
        remember_decision(runtime, budget_plan, accepted)
        return accepted
    if budget_plan is not None:
        klass = (budget_class(codes, rows=evidence)
                 if budget_class is not None else klass)
        ids = (clause_ids(decision, rows=evidence, runtime=runtime)
               if clause_ids is not None else ())
        assessment = (assessment_for(klass, ids, rows=evidence)
                      if assessment_for is not None else
                      ("known_violation" if klass == "COUNTEREXAMPLE" else "unresolved"))
        if budget_plan.get("action") in {"submit", "cap", "reconcile"}:
            accepted = rebuild_accepted()
            if getattr(accepted, "accepted", False):
                status = "capped" if budget_plan["action"] == "cap" else "submitted"
                reconcile_kwargs = ({"override_reason": "submission_reconciled"}
                                    if budget_plan["action"] == "reconcile" else {})
                record = commit_submission(
                    runtime, budget_plan, attempt_status=status,
                    assessment=assessment, open_clause_ids=ids,
                    challenge_class=klass, terminal_result=accepted, **reconcile_kwargs)
                setattr(runtime, "_done_submission_record", record)
                reason = reconcile_kwargs.get("override_reason") or (
                    "second_done_same_revision" if status == "submitted"
                    else "run_challenge_cap")
                _emit(runtime, "done_challenge_overridden", revision=key, reason=reason,
                      challenge_class=klass, challenged_codes=sorted(codes), menu=list(menu),
                      attempt_status=status, assessment=record.get("assessment"),
                      open_clause_ids=list(record.get("open_clause_ids", ())))
                _emit(runtime, "done_submission_record", **record)
                remember_decision(runtime, budget_plan, accepted)
            return accepted
        if budget_plan.get("action") == "challenge":
            if done_only(menu) and not _post_green_reopen(runtime, key, ids):
                accepted = rebuild_accepted()
                if getattr(accepted, "accepted", False):
                    record = commit_submission(
                        runtime, {**budget_plan, "action": "submit"},
                        attempt_status="submitted", assessment=assessment,
                        open_clause_ids=ids, challenge_class=klass,
                        terminal_result=accepted,
                        override_reason="sole_tool_submission")
                    setattr(runtime, "_done_submission_record", record)
                    _emit(runtime, "done_challenge_overridden", revision=key,
                          reason="sole_tool_submission", challenge_class=klass,
                          challenged_codes=sorted(codes), menu=list(menu),
                          attempt_status="submitted", assessment=record.get("assessment"),
                          open_clause_ids=list(record.get("open_clause_ids", ())))
                    _emit(runtime, "done_submission_record", **record)
                return accepted
            commit_challenge(runtime, budget_plan, challenge_class=klass)
            setattr(runtime, "_done_challenge_revision", key)
            setattr(runtime, "_done_challenge_count", int(budget_plan["spent"]) + 1)
            _emit(runtime, "done_challenge_once", revision=key, reason_codes=sorted(codes),
                  challenge_class=klass, run_challenge_count=int(budget_plan["spent"]) + 1,
                  run_challenge_cap=int(budget_plan["cap"]), menu=list(menu),
                  done_only_menu=done_only(menu), open_clause_ids=list(ids))
            if strict_counterexample:
                from kernel_first_green_disclosure import begin_challenge
                begin_challenge(runtime, key)
            detail = str(getattr(decision, "detail", "") or "")
            if strict_counterexample:
                suffix = (f"{CLASS_TEXT[klass]} {NO_AGREE_TEXT}\n"
                          "[done challenge] This done was challenged at this revision. "
                          f"Before a second done, {ACTION_HINT}.")
            else:
                suffix = f"{CLASS_TEXT[klass]} {CHALLENGE_ENDING}"
            try:
                import dataclasses
                challenged = dataclasses.replace(
                    decision, detail=(detail + "\n" if detail else "") + suffix)
            except (TypeError, ValueError):
                challenged = decision
            remember_decision(runtime, budget_plan, challenged)
            return challenged
    if (same_revision or cap_reached
            or (strict_counterexample and progressed)):
        accepted = rebuild_accepted()
        if getattr(accepted, "accepted", False):
            reason = "run_challenge_cap" if cap_reached else "second_done_same_revision"
            setattr(runtime, "_done_challenge_overrides",
                    int(getattr(runtime, "_done_challenge_overrides", 0) or 0) + 1)
            _emit(runtime, "done_challenge_overridden", revision=key, reason=reason,
                  challenge_class=klass, challenged_codes=sorted(codes), menu=list(menu))
            # attempt assessment: attempt_status is separate from the assessment; an accepted
            # override never turns open items into passes.
            record = {"attempt_status": "submitted",
                      "assessment": ("known_violation" if klass == "COUNTEREXAMPLE"
                                     else "unresolved"),
                      "override_reason": reason, "revision": key,
                      "challenged_codes": sorted(codes)}
            if getattr(runtime, "stateframe_false_completion_mode", "off") == "on":
                try:
                    from kernel_false_completion import accepted_submission_fields
                    record.update(accepted_submission_fields(runtime))
                except (ImportError, OSError, TypeError, ValueError):
                    pass
            setattr(runtime, "_done_submission_record", record)
            _emit(runtime, "done_submission_record", **record)
        return accepted
    setattr(runtime, "_done_challenge_revision", key)
    setattr(runtime, "_done_challenge_count", spent + 1)
    if strict_counterexample:
        from kernel_first_green_disclosure import begin_challenge
        begin_challenge(runtime, key)
    _emit(runtime, "done_challenge_once", revision=key, reason_codes=sorted(codes),
          challenge_class=klass, run_challenge_count=spent + 1,
          run_challenge_cap=CHALLENGE_RUN_CAP, menu=list(menu), done_only_menu=done_only(menu))
    detail = str(getattr(decision, "detail", "") or "")
    if strict_counterexample:
        suffix = (f"{CLASS_TEXT[klass]} {NO_AGREE_TEXT}\n"
                  "[done challenge] This done was challenged at this revision. "
                  f"Before a second done, {ACTION_HINT}.")
    else:
        suffix = f"{CLASS_TEXT[klass]} {NO_AGREE_TEXT}\n{CHALLENGE_SUFFIX}"
    try:
        import dataclasses
        return dataclasses.replace(
            decision, detail=(detail + "\n" if detail else "") + suffix)
    except (TypeError, ValueError):
        return decision


WITNESSED_STOP_TEXT = ("STOPPED_INCOMPLETE: [done] a host-witnessed failure ({codes}) still "
                      "holds after {count} refused done(s); the attempt stops unsubmitted.")
WITNESSED_REFUSAL_TEXT = ("[done] refused: a host-witnessed failure ({codes}) holds on this "
                          "candidate. Repair it (edit/str_replace), then done.")


def witnessed_failure_bound(runtime: object, decision):
    """r44 fix DONE/D2: refusals carrying a NON_WAIVABLE witnessed failure.

    Never accepted.  Each such refused done counts run-wide; while a repair edit is on the
    menu the first CHALLENGE_RUN_CAP refusals stay recoverable, the next one (or any with no
    repair on the menu) is the bounded terminal STOPPED_INCOMPLETE -- never a refuse-all loop.
    """
    from kernel_done_is_submit_authority import accepted_decision, done_sealed

    if done_sealed(runtime) is not None:
        return accepted_decision(runtime) or decision
    codes = set(getattr(decision, "reason_codes", ()) or ())
    if (getattr(decision, "accepted", False) or not codes & NON_WAIVABLE_CODES
            or codes & _TERMINAL_CODES):
        return decision
    import dataclasses
    witnessed = sorted(codes & NON_WAIVABLE_CODES)
    menu = menu_names(runtime)
    count = int(getattr(runtime, "_witnessed_failure_refusals", 0) or 0) + 1
    setattr(runtime, "_witnessed_failure_refusals", count)
    repair = bool(REPAIR_EDIT_TOOLS & set(menu)) or not menu
    detail = str(getattr(decision, "detail", "") or "")
    if repair and count <= CHALLENGE_RUN_CAP:
        # When unified stop already challenged this done it owns the refusal (one veto per
        # done); this arbiter only counts it toward the run-wide terminal bound.
        owner = "unified_stop" if _UNIFIED_OWNED_CODE in codes else "done_challenge_once"
        _emit(runtime, "done_witnessed_failure_refused" if owner == "done_challenge_once"
              else "done_witnessed_failure_counted", reason_codes=sorted(codes),
              witnessed=witnessed, refusal_count=count, cap=CHALLENGE_RUN_CAP, menu=list(menu),
              owner=owner)
        if detail.strip():
            return decision
        return dataclasses.replace(decision, detail=WITNESSED_REFUSAL_TEXT.format(
            codes=", ".join(witnessed)))
    stop = WITNESSED_STOP_TEXT.format(codes=", ".join(witnessed), count=count)
    try:
        setattr(runtime, "system_outcome", "STOPPED_INCOMPLETE")
    except (AttributeError, TypeError):
        pass
    _emit(runtime, "done_witnessed_failure_stop", reason_codes=sorted(codes),
          witnessed=witnessed, refusal_count=count, cap=CHALLENGE_RUN_CAP,
          repair_on_menu=repair, menu=list(menu))
    return dataclasses.replace(
        decision,
        reason_codes=tuple(dict.fromkeys((*getattr(decision, "reason_codes", ()),
                                          "STOPPED_INCOMPLETE"))),
        detail=(detail + "\n" if detail else "") + stop)


def _post_green_reopen(runtime: object, key: object, ids: object) -> bool:
    """STATEFRAME_POST_GREEN_WORK_MENU rule 2: a sole done with open clauses is challenged and
    the work window reopened (once per revision, under the cycle cap) instead of waived."""
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        return False
    named_layer = False
    try:
        # Rule D owns the cap exception only when a confident untouched layer is
        # actually present; the ordinary post-green reopen keeps its cap.
        import kernel_named_layer_repair
        named_layer = bool(
            kernel_named_layer_repair.enabled()
            and kernel_named_layer_repair.untouched_rows(runtime)
        )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        named_layer = False
    if named_layer:
        # Rule D is an advisory reopen, never a second completion blocker. Under
        # STATEFRAME_POSTGREEN_NO_VETO the single done-refusal owner is kernel_postgreen_no_veto (run-wide once),
        # so the named-layer cap exception is recorded, not enforced.
        try:
            from kernel_postgreen_no_veto import enabled as no_veto_enabled
            if no_veto_enabled(runtime):
                import kernel_named_layer_repair
                kernel_named_layer_repair.emit(runtime, "done_reopen", stage="applicable", advisory=True)
                named_layer = False
        except (ImportError, ValueError):
            pass
    from kernel_post_green_menu import reopen_instead_of_sole_submission
    reopened = reopen_instead_of_sole_submission(
        runtime, key, tuple(ids or ()), named_layer=named_layer)
    if named_layer:
        import kernel_named_layer_repair
        kernel_named_layer_repair.emit(runtime, "done_reopen", stage="acted" if reopened else "applicable")
    return reopened


def _commit_capped_submission(runtime: object, decision: object, accepted: object, *,
                              key: object, codes: set[str], menu: Sequence[str]) -> dict[str, object] | None:
    """Finalize a CAP1 acceptance when the durable challenge budget is active."""
    if not getattr(accepted, "accepted", False):
        return None
    try:
        from kernel_challenge_budget import (
            assessment_for, budget_for, challenge_class as budget_class, clause_ids,
            commit_submission, open_clause_rows, plan, remember_decision,
        )
    except ImportError:
        return None
    if budget_for(runtime) is None:
        return None
    budget_plan = plan(runtime, key, request=None)
    if budget_plan is None or budget_plan.get("action") not in {"submit", "cap", "reconcile", "challenge"}:
        return None
    evidence = open_clause_rows(decision, runtime=runtime)
    klass = budget_class(codes, rows=evidence)
    ids = clause_ids(decision, rows=evidence, runtime=runtime)
    record = commit_submission(
        runtime, {**budget_plan, "action": "submit"}, attempt_status="submitted",
        assessment=assessment_for(klass, ids, rows=evidence), open_clause_ids=ids,
        challenge_class=klass, terminal_result=accepted,
        override_reason=("submission_reconciled" if budget_plan.get("action") == "reconcile"
                         else "done_challenge_capped"))
    if not record:
        return None
    setattr(runtime, "_done_submission_record", record)
    _emit(runtime, "done_submission_record", **record)
    remember_decision(runtime, budget_plan, accepted)
    return record


def _emit(runtime: object, event: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(event, producer="kernel_done_challenge_once", dispatch=False, **fields)


ACTION_HINT = ("make a real source edit, read a file not read since the challenge, "
               "or run a new probe before done")


def _first_green_enabled(runtime: object) -> bool:
    try:
        from kernel_first_green_disclosure import enabled
        return enabled(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _counterexample_ids(runtime: object, codes: set[str]) -> list[str]:
    try:
        from kernel_first_green_disclosure import open_counterexample_ids
        identifiers = list(open_counterexample_ids(runtime))
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        identifiers = []
    return identifiers or sorted(codes)


def _strict_waiting_suffix(runtime: object, codes: set[str]) -> str:
    labels = json.dumps(_counterexample_ids(runtime, codes), ensure_ascii=False)
    if codes & _STRICT_UNVERIFIED_CODES and not codes & _COUNTEREXAMPLE_CODES:
        # r41b relabel: host-demoted check problems are "unverified checks", never
        # "open counterexamples" (not a demonstrated failure).
        return ("[done challenge] No qualifying action happened since the challenge; "
                f"unverified checks {labels} remain unverified. Before done, {ACTION_HINT}.")
    return ("[done challenge] No qualifying action happened since the challenge; "
            f"open counterexamples {labels} remain aligned. Before done, {ACTION_HINT}.")


__all__ = ["CHALLENGE_RUN_CAP", "CLASS_TEXT", "NO_AGREE_TEXT", "challenge_class", "CHALLENGE_SUFFIX", "MODE_ENV", "OVERRIDE_CODE", "REPAIR_EDIT_TOOLS",
           "QUALIFYING_ACTIONS", "arbitrate", "done_only", "has_qualifying_action", "menu_names",
           "mode", "repair_hold", "revision_key", "witnessed_failure_bound"]
