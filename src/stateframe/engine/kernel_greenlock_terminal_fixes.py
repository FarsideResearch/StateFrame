"""Opt-in r47 fixes for green-lock terminal and done-challenge paths.

Each switch defaults OFF; disabled paths remain owned by the v33 implementation.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence

FIRST_EDIT_ENV = "STATEFRAME_GREENLOCK_FIRST_EDIT"
NO_RED_TERMINAL_ENV = "STATEFRAME_NO_RED_TERMINAL"
CAP1_ENV = "STATEFRAME_DONE_CHALLENGE_CAP1"

FIRST_EDIT_EVENT = "greenlock_first_edit_applied"
RED_TERMINAL_EVENT = "red_terminal_suppressed"
CAP1_EVENT = "done_challenge_capped"

PRODUCER = "kernel_greenlock_terminal_fixes"

EDIT_ACTIONS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"})
RUN_TESTS_ACTIONS = frozenset({"run_tests", "test", "pytest"})
RULE_FIRST_EDIT = "greenlock_first_edit"
RULE_RUN_TESTS_CAP = "greenlock_run_tests_cap"
RULE_RED_REOPEN = "red_terminal_reopen"

RUN_TESTS_HONEST_NOTE = (
    "[green lock] run_tests was refused ONCE at this revision because the public checks are already "
    "green here; the next run_tests call at this revision will run.")
PUBLIC_RED_CODE = "PUBLIC_RED_DONE"
PUBLIC_RED_DETAIL = (
    "[done refused] the required public checks are RED on the current revision; done cannot be "
    "accepted until they pass. Fix the product code, then run_tests.")
UNVERIFIED_CODES = frozenset({"UNVERIFIED_DISCLOSED_CHECKS"})
WITNESS_REQUIRED_CODE = "BEHAVIOR_WITNESS_REQUIRED"
PUBLIC_CHECKS_NOT_RUN_CODE = "PUBLIC_CHECKS_NOT_RUN_CURRENT_REVISION"
_UNEXECUTED_RESULTS = frozenset({
    "NO_EXECUTION_RECEIPT", "REJECTED", "REJECTED_CONTRACT", "REFUSED", "REFUSED_PREDISPATCH",
})
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def _strict(env: str) -> str:
    value = os.environ.get(env, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{env} must be off or on")
    return value


def _enabled(runtime: object, env: str, attr: str) -> bool:
    configured = getattr(runtime, attr, None) if runtime is not None else None
    if configured is not None:
        return configured == "on"
    return _strict(env) == "on"


def first_edit_mode() -> str:
    return _strict(FIRST_EDIT_ENV)


def no_red_terminal_mode() -> str:
    return _strict(NO_RED_TERMINAL_ENV)


def cap1_mode() -> str:
    return _strict(CAP1_ENV)


def first_edit_enabled(runtime: object = None) -> bool:
    return _enabled(runtime, FIRST_EDIT_ENV, "stateframe_greenlock_first_edit_mode")


def no_red_terminal_enabled(runtime: object = None) -> bool:
    return _enabled(runtime, NO_RED_TERMINAL_ENV, "stateframe_no_red_terminal_mode")


def cap1_enabled(runtime: object = None) -> bool:
    return _enabled(runtime, CAP1_ENV, "stateframe_done_challenge_cap1_mode")


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer=PRODUCER, dispatch=False, **fields)


def _norm(action: object) -> str:
    return str(action or "").strip().lower()


def public_verdict(runtime: object) -> str:
    """Return GREEN, RED, or UNKNOWN from current host-bound public receipts."""
    try:
        from kernel_green_lock import _current_binding, _declared_selectors
        from kernel_verdict import SATISFIED, VIOLATED, evaluate_required_tests
        current = _current_binding(runtime)
        required = _declared_selectors(runtime)
        if current is None or not required:
            return "UNKNOWN"
        state = evaluate_required_tests(
            current, required, list(getattr(runtime, "test_receipts", ()) or ())).state
    except Exception:  # noqa: BLE001 - unreadable host evidence is never RED
        return "UNKNOWN"
    if state == SATISFIED:
        return "GREEN"
    if state == VIOLATED:
        return "RED"
    return "UNKNOWN"


def _applies(lock: object) -> bool:
    return (lock is not None and getattr(lock, "mode", "off") == "terminal"
            and first_edit_enabled(getattr(lock, "runtime", None)))


def _stopped(lock: object) -> bool:
    try:
        return lock._repair_blocks_done_refusal() is not None
    except (AttributeError, TypeError, ValueError):
        return False


def _engaged(lock: object) -> bool:
    try:
        lock.sync()
    except Exception:  # noqa: BLE001 - legacy gate owns lock failures
        return False
    return bool(getattr(lock, "terminal_engaged", False))


def _product_target(lock: object, path: object) -> str | None:
    from kernel_green_lock import is_product_path
    try:
        target = lock._workspace_path(path)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return target if is_product_path(target) else None


def menu_rule(lock: object, action: object, arguments: Mapping[str, object] | None = None,
              *, for_menu: bool) -> str | None:
    """Return the first-edit/run-tests rule, or None when this switch abstains."""
    if not _applies(lock) or not getattr(lock, "terminal_engaged", False) or _stopped(lock):
        return None
    name = _norm(action)
    if name in EDIT_ACTIONS:
        if for_menu:
            return RULE_FIRST_EDIT
        path = arguments.get("path", "") if isinstance(arguments, Mapping) else ""
        return RULE_FIRST_EDIT if _product_target(lock, path) is not None else None
    if name in RUN_TESTS_ACTIONS:
        refused = getattr(lock, "_r47_run_tests_refused_revision", None)
        revision = getattr(lock, "_green_revision", None)
        if refused is not None and refused == revision:
            return RULE_RUN_TESTS_CAP
    return None


def _capture_before(lock: object, path: object) -> None:
    try:
        from kernel_green_lock_repair import controller_for
        controller = controller_for(lock.runtime)
        if controller is not None:
            controller.capture_before(path)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass


def gate_mutation(lock: object, path: object, action_kind: str,
                  impl: Callable[[], str | None]) -> str | None:
    """Allow an engaged terminal-lock product edit when FIRST_EDIT is enabled."""
    if not _applies(lock) or not _engaged(lock) or _stopped(lock):
        return impl()
    target = _product_target(lock, path)
    if target is None:
        return impl()
    _capture_before(lock, path)
    _emit(lock.runtime, FIRST_EDIT_EVENT, stage="edit_allowed", action_kind=_norm(action_kind) or "edit",
          path=target, product_revision=getattr(lock, "_green_revision", None))
    return None


def _honest(refusal: str) -> str:
    from kernel_green_lock import REFUSAL_SUFFIX
    text = str(refusal)
    if text.endswith(REFUSAL_SUFFIX):
        text = text[: -len(REFUSAL_SUFFIX)]
    return text + ("\n" if text else "") + RUN_TESTS_HONEST_NOTE


def gate_action(lock: object, kind: object, arguments: Mapping[str, object] | None,
                impl: Callable[[], str | None]) -> str | None:
    """Wrap before_action with product-edit and once-per-revision test behavior."""
    action = _norm(kind)
    runtime = getattr(lock, "runtime", None)
    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    if action == "done" and done_submit_enabled(runtime):
        _emit(runtime, "done_submit_dispatch_admitted", action_kind=action, dispatch=False)
        return None
    if not _applies(lock):
        return impl()
    args = arguments if isinstance(arguments, Mapping) else {}
    if action in EDIT_ACTIONS:
        return gate_mutation(lock, args.get("path", ""), action, impl)
    if action not in RUN_TESTS_ACTIONS or not _engaged(lock) or _stopped(lock):
        return impl()
    revision = getattr(lock, "_green_revision", None)
    if revision is not None and getattr(lock, "_r47_run_tests_refused_revision", None) == revision:
        _emit(lock.runtime, FIRST_EDIT_EVENT, stage="run_tests_allowed", action_kind=action,
              product_revision=revision)
        return None
    refusal = impl()
    if refusal is not None and revision is not None and getattr(lock, "terminal_engaged", False):
        lock._r47_run_tests_refused_revision = revision
        _emit(lock.runtime, FIRST_EDIT_EVENT, stage="run_tests_refused_once", action_kind=action,
              product_revision=revision)
        return _honest(refusal)
    return refusal


def settle_unexecuted_edit(effect: object, record: Mapping[str, object], current: object) -> str | None:
    """Settle a host-proven unexecuted write as not_applied when NO_RED_TERMINAL is on."""
    if no_red_terminal_mode() != "on" or not isinstance(record, Mapping):
        return None
    if getattr(effect, "read_only", True) or getattr(effect, "tool", None) not in EDIT_ACTIONS:
        return None
    if (record.get("result") not in _UNEXECUTED_RESULTS or record.get("changes")
            or record.get("before_binding") != current or record.get("after_binding") != current):
        return None
    return "not_applied"


def note_settled_by_rule(journal: object, evidence: object, effect_id: object) -> None:
    if effect_id is None or getattr(evidence, "_r47_unexecuted_settled", None) != effect_id:
        return
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(RED_TERMINAL_EVENT, producer=PRODUCER, dispatch=False, site="effect_settled",
             effect_id=effect_id, settled="not_applied")


def reopen_empty_menu(lock: object, surface: list[dict], name_of: Callable[[object], str],
                      lock_allows: Callable[[str], bool]) -> list[dict] | None:
    """Offer lock-level repair tools when NO_RED_TERMINAL prevents a red empty terminal."""
    runtime = getattr(lock, "runtime", None)
    if not no_red_terminal_enabled(runtime) or getattr(lock, "terminal_engaged", False):
        return None
    verdict = public_verdict(runtime)
    if verdict == "GREEN":
        return None
    offered = [tool for tool in surface if name_of(tool) and name_of(tool) != "done"
               and lock_allows(name_of(tool))]
    if not offered:
        return None
    setattr(runtime, "_r47_red_menu_reopened", tuple(name_of(tool) for tool in offered))
    _emit(runtime, RED_TERMINAL_EVENT, site="empty_menu", public_verdict=verdict,
          offered=[name_of(tool) for tool in offered])
    return offered


def red_reopen_rule(lock: object, name: str, decision_rule: str) -> str | None:
    runtime = getattr(lock, "runtime", None)
    reopened = getattr(runtime, "_r47_red_menu_reopened", None)
    reopened_names = {_norm(item) for item in (reopened or ())}
    alias_match = name in reopened_names or (
        name == "edit" and bool(reopened_names.intersection(EDIT_ACTIONS)))
    if (not reopened_names or not alias_match or getattr(lock, "terminal_engaged", False)
            or decision_rule not in {"continuation_gate", "readiness_disposition", "upstream_menu_policy"}):
        return None
    return RULE_RED_REOPEN


def done_red_refusal(runtime: object, admissible: bool) -> tuple[str, str] | None:
    if not admissible or not no_red_terminal_enabled(runtime):
        return None
    if public_verdict(runtime) != "RED":
        return None
    _emit(runtime, RED_TERMINAL_EVENT, site="done_refused_red", public_verdict="RED")
    return PUBLIC_RED_CODE, PUBLIC_RED_DETAIL


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value in (None, ""):
        return ()
    return (value,)


def _witness_context(runtime: object, decision: object) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Collect the typed witness rows carried by the current refusal."""
    ids: list[str] = []
    for source in (
        getattr(decision, "needs_work_ids", ()),
        getattr(decision, "open_work_ids", ()),
        (getattr(runtime, "_stateframe_witness_gate_preflight", None) or {}).get("witness_ids", ())
        if isinstance(getattr(runtime, "_stateframe_witness_gate_preflight", None), Mapping) else (),
    ):
        for item in _sequence(source):
            text = str(item).strip()
            if text and text not in ids:
                ids.append(text)

    assessments: list[object] = []
    for source in (
        getattr(decision, "obligation_assessments", ()),
        getattr(runtime, "host_closure_assessments", ()),
        getattr(runtime, "behavior_witness_assessments", ()),
        getattr(runtime, "obligation_assessments", ()),
    ):
        assessments.extend(_sequence(source))
        for item in _sequence(source):
            if isinstance(item, Mapping):
                ident = str(item.get("obligation_id", item.get("id", ""))).strip()
            else:
                ident = str(getattr(item, "obligation_id", getattr(item, "id", ""))).strip()
            if ident and ident not in ids:
                ids.append(ident)
    return tuple(ids), tuple(assessments)


def _typed_probe_on_surface(runtime: object) -> bool:
    """The witness gate is meaningful only with the real typed probe schema."""
    try:
        from kernel_completion_owner import _typed_probe_tool
        current_menu = getattr(runtime, "completion_menu", None)
        if isinstance(current_menu, Sequence) and not isinstance(current_menu, (str, bytes, bytearray)):
            if not current_menu or "probe" not in {str(item) for item in current_menu}:
                return False
        candidates = (
            getattr(runtime, "_completion_owner_full_tools", ()),
            getattr(runtime, "_completion_owner_dispatch_tools", ()),
        )
        return any(_typed_probe_tool(tool) for group in candidates for tool in _sequence(group))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _real_witness_context(runtime: object, ids: Sequence[str]) -> bool:
    """Require every offered witness id to have a concrete public entrypoint."""
    if not ids:
        return False
    assessments = tuple(
        item
        for source_name in (
            "host_closure_assessments", "behavior_witness_assessments", "obligation_assessments",
        )
        for item in (getattr(runtime, source_name, ()) or ())
    )
    by_id = {
        str(item.get("obligation_id", item.get("id", ""))).strip(): item
        for item in assessments if isinstance(item, Mapping)
    }
    return all(
        isinstance(by_id.get(str(clause_id)), Mapping)
        and bool(_ENTRYPOINT_RE.fullmatch(str(
            by_id[str(clause_id)].get("entrypoint",
                                      by_id[str(clause_id)].get("public_entrypoint", ""))
            or "").strip()))
        and str(by_id[str(clause_id)].get("entrypoint",
                                          by_id[str(clause_id)].get("public_entrypoint", ""))
                or "").strip() != "public.module.function"
        for clause_id in ids
    )


def _pending_behavior_witness(runtime: object, decision: object, codes: set[str]) -> tuple[bool, tuple[str, ...]]:
    ids, assessments = _witness_context(runtime, decision)
    if WITNESS_REQUIRED_CODE not in codes and not assessments:
        return False, ()
    if not _typed_probe_on_surface(runtime):
        return False, ids
    try:
        from kernel_behavior_witness import enabled, pending_public_evidence_work
        if not enabled():
            return False, ids
        pending = pending_public_evidence_work(runtime, ids, assessments)
    except (ImportError, AttributeError, TypeError, ValueError):
        # An inspection failure is host uncertainty, not a model-owned pending
        # witness.  The normal terminal owner will retain the full surface.
        return False, ids
    if WITNESS_REQUIRED_CODE in codes and ids and _real_witness_context(runtime, ids):
        # A compact completion assessment can omit optional clause text. The
        # typed refusal remains pending until a current PASS or UNRUNNABLE
        # receipt settles every requested clause.
        try:
            from kernel_behavior_witness import _json_digest
            binding = runtime.binding() if callable(getattr(runtime, "binding", None)) else None
            revision_digest = _json_digest(binding)
            latest = {
                str(record.get("clause_id", "")): str(record.get("result", "")).upper()
                for record in (getattr(runtime, "behavior_witness_records", ()) or ())
                if isinstance(record, Mapping)
                and str(record.get("revision_digest", "")) == revision_digest
            }
            pending = pending or not all(
                latest.get(clause_id) in {"PASS", "UNRUNNABLE"} for clause_id in ids
            )
        except (AttributeError, ImportError, TypeError, ValueError):
            pending = False
    # The gate can carry the typed request IDs without assessments on a later
    # completion pass.  Preserve that request until a PASS or UNRUNNABLE record
    # makes it advisory.
    settled = False
    if not assessments:
        try:
            from kernel_behavior_witness import _json_digest
            binding = runtime.binding() if callable(getattr(runtime, "binding", None)) else None
            current_digest = _json_digest(binding)
            latest: dict[str, Mapping[str, object]] = {}
            for record in getattr(runtime, "behavior_witness_records", ()) or ():
                if (isinstance(record, Mapping)
                        and str(record.get("revision_digest", "")) == current_digest):
                    latest[str(record.get("clause_id", ""))] = record
            settled = (all(
                str(latest.get(clause_id, {}).get("result", "")).upper()
                in {"PASS", "UNRUNNABLE"}
                for clause_id in ids
            ) if ids else bool(latest) and all(
                str(record.get("result", "")).upper() in {"PASS", "UNRUNNABLE"}
                for record in latest.values()
            ))
            pending = not settled
        except (AttributeError, ImportError, TypeError, ValueError):
            pass
    return bool(pending), ids


def _public_checks_not_current(runtime: object) -> bool:
    """Return whether required public checks lack a green receipt for binding()."""
    try:
        from kernel_green_lock import _declared_selectors
        required = _declared_selectors(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    return bool(required) and public_verdict(runtime) != "GREEN"


def _actionable_cap_block(runtime: object, decision: object, codes: set[str]) -> tuple[str, str] | None:
    pending, ids = _pending_behavior_witness(runtime, decision, codes)
    if pending and _real_witness_context(runtime, ids):
        clause = ", ".join(ids) if ids else "one pending public behavior clause"
        return (
            "behavior_witness_pending",
            "[done refused] use the offered typed probe for clause(s): " + clause + ".",
        )
    if _public_checks_not_current(runtime):
        return (
            "public_checks_not_run",
            "[done refused] tests not run since your last edit; run_tests is required before done.",
        )
    return None


def _with_cap_block_detail(decision: object, code: str, detail: str) -> object:
    current = str(getattr(decision, "detail", "") or "")
    reason_codes = tuple(dict.fromkeys((*tuple(getattr(decision, "reason_codes", ()) or ()), code)))
    combined = (current + ("\n" if current else "") + detail).strip()
    try:
        import dataclasses
        return dataclasses.replace(decision, detail=combined, reason_codes=reason_codes)
    except (TypeError, ValueError):
        try:
            setattr(decision, "detail", combined)
            setattr(decision, "reason_codes", reason_codes)
        except (AttributeError, TypeError):
            pass
        return decision


def cap_done(runtime: object, decision: object, *, key: object, non_waivable: frozenset,
             rebuild_accepted: Callable[[], object]) -> object | None:
    """Return an accepted advisory/capped decision, or a guarded refusal."""
    from kernel_done_is_submit_authority import accepted_decision, done_sealed

    if done_sealed(runtime) is not None:
        return accepted_decision(runtime)
    if not cap1_enabled(runtime) or getattr(decision, "accepted", False):
        return None
    codes = set(getattr(decision, "reason_codes", ()) or ())
    if not codes or codes & non_waivable or PUBLIC_RED_CODE in codes:
        return None
    if public_verdict(runtime) == "RED":
        return None
    challenged = getattr(runtime, "_r47_cap1_challenged", None)
    if not isinstance(challenged, set):
        challenged = set()
        setattr(runtime, "_r47_cap1_challenged", challenged)
    blocked = _actionable_cap_block(runtime, decision, codes)
    if blocked is not None:
        reason, detail = blocked
        challenged.add(key)
        try:
            from kernel_done_challenge_once import CHALLENGE_RUN_CAP
        except (ImportError, AttributeError):
            CHALLENGE_RUN_CAP = 2
        spent = max(0, int(getattr(runtime, "_done_challenge_count", 0) or 0))
        spent = min(int(CHALLENGE_RUN_CAP), spent + 1)
        setattr(runtime, "_done_challenge_count", spent)
        setattr(runtime, "_done_challenge_revision", key)
        _emit(runtime, "done_challenge_cap_blocked", reason=reason, revision=key,
              reason_codes=sorted(codes), run_challenge_count=spent,
              run_challenge_cap=int(CHALLENGE_RUN_CAP))
        refusal = _with_cap_block_detail(decision, (
            WITNESS_REQUIRED_CODE if reason == "behavior_witness_pending"
            else PUBLIC_CHECKS_NOT_RUN_CODE), detail)
        if spent >= int(CHALLENGE_RUN_CAP):
            _emit(runtime, "done_challenge_cap_exhausted", reason=reason, revision=key,
                  reason_codes=sorted(codes), run_challenge_count=spent,
                  run_challenge_cap=int(CHALLENGE_RUN_CAP))
            accepted = rebuild_accepted()
            if getattr(accepted, "accepted", False):
                return accepted
            try:
                import dataclasses
                from kernel_terminal_policy import (
                    enabled as hard_budget_only_enabled,
                    queue_advisory,
                )
                hard_budget_only = hard_budget_only_enabled(runtime)
                if hard_budget_only:
                    queue_advisory(runtime, "done_witness_challenge_cap")
                else:
                    setattr(runtime, "system_outcome", "STOPPED_INCOMPLETE")
                reason_codes = tuple(dict.fromkeys((
                    *getattr(refusal, "reason_codes", ()), "STOPPED_INCOMPLETE")))
                stop_detail = (
                    "witness challenge cap reached; continue with available work"
                    if hard_budget_only else "witness challenge cap exhausted; stopping incomplete"
                )
                detail = (str(getattr(refusal, "detail", "") or "")
                          + f"\n[done] {stop_detail}.").strip()
                try:
                    terminal = dataclasses.replace(
                        refusal, reason_codes=reason_codes, detail=detail)
                except (TypeError, ValueError):
                    # Some legacy reducers use a mutable namespace rather than
                    # a dataclass; the run-wide stop must remain durable there.
                    setattr(refusal, "reason_codes", reason_codes)
                    setattr(refusal, "detail", detail)
                    terminal = refusal
                setattr(runtime, "_last_completion_decision", terminal)
                return terminal
            except (AttributeError, TypeError, ValueError):
                return refusal
        # The completion owner is the sole next-action authority.  Publish the
        # post-cap refusal before the next request is composed; otherwise the
        # green-lock terminal fallback sees stale state and re-offers done.
        setattr(runtime, "_last_completion_decision", refusal)
        return refusal
    if codes <= UNVERIFIED_CODES:
        reason = "unverified_advisory"
    elif key in challenged:
        reason = "second_done_same_revision"
    else:
        challenged.add(key)
        return None
    accepted = rebuild_accepted()
    if not getattr(accepted, "accepted", False):
        return None
    detail = str(getattr(decision, "detail", "") or "")
    _emit(runtime, CAP1_EVENT, reason=reason, revision=key, reason_codes=sorted(codes))
    if reason == "unverified_advisory" and detail:
        try:
            import dataclasses
            accepted = dataclasses.replace(
                accepted, detail=(str(getattr(accepted, "detail", "") or "")
                                  + "\n[advisory, not a refusal] " + detail).strip())
        except (TypeError, ValueError):
            pass
    return accepted


__all__ = [
    "CAP1_ENV", "CAP1_EVENT", "FIRST_EDIT_ENV", "FIRST_EDIT_EVENT", "NO_RED_TERMINAL_ENV",
    "PUBLIC_CHECKS_NOT_RUN_CODE", "PUBLIC_RED_CODE", "RED_TERMINAL_EVENT", "RULE_FIRST_EDIT",
    "RULE_RED_REOPEN", "RULE_RUN_TESTS_CAP", "WITNESS_REQUIRED_CODE",
    "cap1_enabled", "cap1_mode", "cap_done", "done_red_refusal", "first_edit_enabled", "first_edit_mode",
    "gate_action", "gate_mutation", "menu_rule", "no_red_terminal_enabled", "no_red_terminal_mode",
    "note_settled_by_rule", "public_verdict", "red_reopen_rule", "reopen_empty_menu",
    "settle_unexecuted_edit",
]
