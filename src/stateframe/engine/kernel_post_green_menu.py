"""Post-green working menu for the terminal green lock (r44, dissection-r42-hybrid-vs-frame.md).

r42/341: once public tests went green the frame collapsed the menu to done/read/probe and then
to ``done`` alone.  In 16 locked runs 3 passed hidden tests (8/12 when the menu never locked);
in ldev_10/924104 the model reasoned "I need to modify service.py" and had no edit tool.  The
done challenge could not bite either: with ``done`` as the only tool it was waved through as
``sole_tool_submission`` while ticket clauses were still open (16x in 341).

``STATEFRAME_POST_GREEN_WORK_MENU=on`` (default off; off never calls into this module):

1. Work window.  While the terminal lock is engaged (public green at the current revision) and
   any ticket clause is open, unprobed or UNVERIFIED on the current revision, the lock's ONE
   allow authority (kernel_greenlock_menu_authority.decide) also permits the working tools
   edit/str_replace/read/probe/run_tests; ``done`` keeps its legacy verdict.  Menu and dispatch
   use the same predicate, so offered == allowed.
   Anti-churn (r37t green-churn): a product edit made through the window marks the lineage
   *pending verification*; ``done`` is denied (and therefore not offered) until a test receipt
   lands, while run_tests stays allowed.  A red result falls to the existing green-lock repair
   path (the lock is simply not engaged).  Post-green edit cycles are capped per green lineage
   (``CYCLE_CAP`` = 2): the lineage starts at the first green revision reached without a
   window edit and every green reached after a window edit inherits it, so an
   edit -> green -> edit chain cannot mint new budget.  At the cap the menu locks as before.
2. No sole-tool waiver with open clauses.  When ``done`` is the only offered tool and clauses
   are open, the challenge arbiter does not accept it as ``sole_tool_submission``; it spends the
   normal one-per-revision challenge and reopens the window once for that revision.  After the
   cap (or once reopened at a revision) the sole done submits, so no state refuses every tool.
3. Green notice: "public tests passed (public only; N ticket clauses still open: ...)".
4. Ticket clause segmentation is wrap-invariant (get-sure table + oracle-grounding spans).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_POST_GREEN_WORK_MENU"
SCHEMA = "stateframe-post-green-work-menu/1"
CYCLE_CAP = 2
MENU_EVENT = "post_green_work_menu"
EDIT_EVENT = "post_green_edit_cycle"
VERIFY_EVENT = "post_green_edit_verified"
REOPEN_EVENT = "post_green_reopen"
RULE_WORK = "post_green_work"
RULE_VERIFY = "post_green_verify"
RULE_CAPABILITY_FLOOR = "post_green_capability_floor"
RULE_UNVERIFIED_DONE = "post_green_edit_unverified"
RULE_IDENTICAL_RERUN = "post_green_identical_rerun"
RULE_NO_EVIDENCE = "postgreen_no_new_evidence"
V15_IDENTICAL_EVENT = "tokenlab_v15_postgreen_identical_rerun"
# Canonical names (the authority maps every edit alias onto ``edit``).
WORK_ACTIONS = frozenset({"edit", "file_delete", "file_move", "shell", "read", "probe", "run_tests"})
UNVERIFIED_DONE_REFUSAL = (
    "done refused: a product edit was made after public tests passed and has not been "
    "re-tested yet. Next action: run_tests (edit/str_replace/read/probe stay available).")
_CLOSED = frozenset({"SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED",
                     "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_post_green_work_menu_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_post_green_menu_state", None)
    if not isinstance(state, dict):
        state = {"base": None, "lineage": False, "cycles": {}, "pending": None,
                 "reopened": [], "reopened_green": []}
        try:
            setattr(runtime, "_post_green_menu_state", state)
        except (AttributeError, TypeError):
            pass
    return state


def state_snapshot(runtime: object) -> dict[str, object]:
    state = _state(runtime)
    return json.loads(json.dumps(state, default=str))


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_post_green_menu", dispatch=False, **fields)


# --------------------------------------------------------------------------- open clauses

def _row(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _row_id(row: Mapping[str, object]) -> str:
    for key in ("obligation_id", "clause_id", "requirement_id", "check_id", "id"):
        value = row.get(key)
        if value not in (None, "") and str(value).strip():
            return str(value)
    return ""


def _row_closed(row: Mapping[str, object]) -> bool:
    for key in ("verdict", "status", "result_class", "result", "state"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().upper() in _CLOSED
    return False


def open_clause_ids(runtime: object) -> tuple[str, ...]:
    """Ticket clauses not closed on the current revision (open, unprobed or UNVERIFIED).

    Sources, in order: the host-closure obligation assessments (recomputed per snapshot; an
    ``UNKNOWN``/unprobed verdict is open), then the first-green disclosure of the CURRENT
    product revision (unverified check ids), then live readiness blockers.
    """
    ids: list[str] = []
    for item in getattr(runtime, "host_closure_assessments", ()) or ():
        row = _row(item)
        ident = _row_id(row)
        if ident and not _row_closed(row) and ident not in ids:
            ids.append(ident)
    if not ids:
        try:
            from kernel_first_green_disclosure import disclosed_open
            disclosed, _rows = disclosed_open(runtime)
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            disclosed = ()
        ids.extend(item for item in disclosed if item not in ids)
    for name in ("_readiness_request_decision", "_readiness_decision"):
        decision = getattr(runtime, name, None)
        for ident in getattr(decision, "blocker_ids", ()) or ():
            if str(ident) and str(ident) not in ids:
                ids.append(str(ident))
    return tuple(ids)


# --------------------------------------------------------------------------- lineage / caps

def _key(value: object) -> str:
    return "" if value is None else str(value)


def cycles_used(runtime: object) -> int:
    state = _state(runtime)
    return int(dict(state.get("cycles") or {}).get(_key(state.get("base")), 0))


def cap_reached(runtime: object) -> bool:
    return cycles_used(runtime) >= CYCLE_CAP


def note_engaged(lock: object, revision: object) -> None:
    """GreenLock.sync hook: a green reached after a window edit inherits the lineage base."""
    runtime = getattr(lock, "runtime", None)
    if not enabled(runtime):
        return
    state = _state(runtime)
    state["green_revision"] = _key(_product_key(revision))
    state["green_selectors"] = list(_selector_tuple(runtime))
    if not state.get("lineage") or state.get("base") is None:
        state["base"] = _key(revision)


def _stopped(lock: object) -> bool:
    try:
        return lock._repair_blocks_done_refusal() is not None
    except (AttributeError, TypeError, ValueError):
        return False


def _authority_applies(lock: object) -> bool:
    try:
        from kernel_greenlock_menu_authority import applies
        return bool(applies(lock))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def window_open(lock: object) -> bool:
    """The post-green work window: engaged, not stopped, under the cap, clauses open (or
    reopened once at this green revision by the done arbiter)."""
    runtime = getattr(lock, "runtime", None)
    if not enabled(runtime) or not bool(getattr(lock, "terminal_engaged", False)):
        return False
    if _stopped(lock) or cap_reached(runtime):
        return False
    state = _state(runtime)
    if _key(getattr(lock, "_green_revision", None)) in set(state.get("reopened_green") or ()):
        return True
    return bool(open_clause_ids(runtime))


def pending_verification(runtime: object) -> bool:
    return enabled(runtime) and _state(runtime).get("pending") is not None


def hostverified_quiet_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_postgreen_hostverified_quiet_mode", None)
    if configured is not None:
        return configured == "on"
    return os.environ.get("STATEFRAME_POSTGREEN_HOSTVERIFIED_QUIET", "off").strip().lower() \
        in {"on", "1", "true", "yes"}


def pending_verification_active(runtime: object) -> bool:
    """Keep the edit armed while ignoring it as an active repair after host verification."""
    if not pending_verification(runtime):
        return False
    if not hostverified_quiet_enabled(runtime):
        return True
    return not host_verified_done(runtime)


def changed_candidate_paths_outside_selectors(
        runtime: object,
) -> tuple[str, ...] | None:
    """Return changed files outside green selectors, or None when identity is unknown."""
    try:
        import kernel_green_lock as green_lock

        capture = getattr(runtime, "_parent_product_identity_capture", None)
        if not isinstance(capture, Mapping):
            capture = green_lock._capture_parent_product_identity(runtime)
        baseline = capture.get("files") if isinstance(capture, Mapping) else None
        if not isinstance(baseline, Mapping) or not baseline:
            lock = getattr(runtime, "green_lock", None)
            baseline = getattr(lock, "_starting_editable_product_files", None)
        if not isinstance(baseline, Mapping) or not baseline:
            return None
        paths = tuple(str(path) for path in baseline if str(path).strip())
        current_workspace = green_lock._workspace_product_file_digests(runtime, paths)
        current_binding = green_lock._current_binding(runtime)
        current_files = green_lock._product_files_value(current_binding)
        if not isinstance(current_workspace, Mapping):
            current_workspace = current_files
        if not isinstance(current_workspace, Mapping):
            return None
        changed_paths = {
            path for path in paths
            if baseline.get(path) != current_workspace.get(path)
        }

        parent_binding = None
        for holder in (getattr(runtime, "green_lock", None), runtime,
                       getattr(runtime, "owner", None)):
            if holder is None:
                continue
            for name in ("_starting_binding", "starting_binding", "parent_binding", "base_binding"):
                candidate = getattr(holder, name, None)
                if isinstance(candidate, Mapping):
                    parent_binding = candidate
                    break
            if parent_binding is not None:
                break
        parent_files = green_lock._product_files_value(parent_binding)
        if current_files is not None and parent_files is not None:
            missing = object()
            for path in set(parent_files) | set(current_files):
                if parent_files.get(path, missing) != current_files.get(path, missing):
                    changed_paths.add(path)
        elif current_files is not None and set(current_files) != set(baseline):
            # A partial or expanded candidate map has no matching parent identity.
            return None
        elif parent_files is not None and current_files is None:
            return None

        changed = tuple(sorted(changed_paths))
        green_state = _state(runtime)
        selectors = tuple(green_state.get("green_selectors") or _selector_tuple(runtime))
        uncovered = []
        for path in changed:
            covered = False
            for selector in selectors:
                normalized = selector.replace("\\", "/").strip()
                if normalized == path or normalized.startswith(path + "::"):
                    covered = True
                    break
                if normalized.endswith("/") and path.startswith(normalized):
                    covered = True
                    break
            if not covered:
                uncovered.append(path)
        return tuple(uncovered)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def capability_floor_needed(runtime: object) -> bool:
    """Keep the normal product surface while declared or uncovered work remains."""
    from kernel_terminal_policy import enabled as hard_budget_only_enabled

    if not hard_budget_only_enabled(runtime):
        return False
    lock = getattr(runtime, "green_lock", None)
    # b1386: key on observed GREEN, not terminal engagement. A non-terminal lock mode is green
    # with terminal_engaged=False; the continuation gate then stripped edit while the model still had product work.
    if lock is None or not bool(getattr(lock, "green", getattr(lock, "terminal_engaged", False))):
        return False
    try:
        import kernel_completion_owner as owner

        repair_state = owner._declared_repair_state(runtime)
        if isinstance(repair_state, Mapping) and repair_state.get("active") is True:
            return True
        if owner.open_ids(runtime) or owner._green_lock_open_requirement_ids(runtime):
            return True
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # Unknown repair state must not collapse an otherwise active repair surface.
        return True
    if pending_verification_active(runtime):
        return True
    changed = changed_candidate_paths_outside_selectors(runtime)
    return changed is None or bool(changed)


def apply_capability_floor(runtime: object, selected: Sequence[dict],
                           available_tools: Sequence[dict]) -> list[dict]:
    """Add only currently dispatchable capability-floor schemas to a green menu."""
    if not capability_floor_needed(runtime):
        return selected if isinstance(selected, list) else list(selected)
    from kernel_capability_floor import apply_floor
    from kernel_completion_owner import capability_floor_action_admissible

    # BUILDFIX 10-18: the green floor restores product capabilities only; done stays with the lock/readiness
    # decision (a repair gate holding a counterexample must still deny it), matching the decide() floor set.
    return apply_floor(
        selected,
        available_tools,
        admissible=lambda name: name != "done" and capability_floor_action_admissible(runtime, name),
        runtime=runtime,
    )


def _named_layer(lock: object, action: str, path: object) -> tuple[bool, str] | None:
    """STATEFRAME_NAMED_LAYER_REPAIR rule B (autopsy-ldev01-cross-release #1): a confidently-untouched named
    layer keeps edit/read admitted past the cycle cap.  Self-bounding: the edit touches the layer."""
    try:
        import kernel_named_layer_repair as named_layer
        if not named_layer.enabled():
            return None
        if named_layer.admits(getattr(lock, "runtime", None), action, path):
            if path is not None:   # dispatch (menu builds are path-free and would flood)
                named_layer.emit(getattr(lock, "runtime", None), "post_green_admit", stage="acted",
                                 action=action, path=str(path))
            return True, named_layer.RULE_NAMED_LAYER
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return None
    return None


def decide(lock: object, action: str, path: object = None, selector: object = None,
           *, for_menu: bool = False) -> tuple[bool, str] | None:
    """Rule for the authority's single allow predicate.  ``None`` = legacy decides.

    ``action`` is the authority's canonical action (every edit alias is ``edit``).  ``path`` (dispatch
    only, switch-on callers) scopes the named-layer rule to the untouched layer's files.
    """
    runtime = getattr(lock, "runtime", None)
    if not enabled(runtime) or _stopped(lock):
        return None
    if (action == "run_tests" and hostverified_quiet_enabled(runtime)
            and host_verified_done(runtime) and capability_floor_needed(runtime)):
        return True, RULE_CAPABILITY_FLOOR
    # POSTGREEN owns verification progress at the host boundary.  It runs
    # before the older no-evidence/selector predicates so a model cannot
    # obtain another request by presenting the same check through an alias.
    try:
        from kernel_postgreen_protocol import decide as protocol_decide
        protocol_args = {"target": path, "selector": selector} if path is not None or selector is not None else {}
        protocol = protocol_decide(lock, action, protocol_args, for_menu=for_menu)
        if protocol is not None:
            return protocol.allowed, protocol.rule
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # The legacy owner remains available for reduced test doubles and
        # profiles that do not carry the protocol's host state.
        pass
    # v16: once current-revision host checks are all green, do not expose or
    # accept work that cannot add evidence; real counterexamples stay in the
    # existing repair window via the fail-closed predicate in this module.
    if getattr(runtime, "stateframe_tokenlab_postgreen_no_evidence_mode", "off") == "on":
        from kernel_postgreen_no_evidence import decide as no_evidence_decide
        no_evidence = no_evidence_decide(lock, action)
        if no_evidence is not None:
            return no_evidence
    if pending_verification_active(runtime):
        if action == "done" and not host_verified_done(runtime):
            return False, RULE_UNVERIFIED_DONE
        if action == "run_tests":
            return True, RULE_VERIFY
    if action == "run_tests" and _v15_identical_green(lock, selector):
        _emit(runtime, V15_IDENTICAL_EVENT,
              revision=_key(getattr(lock, "_green_revision", None)),
              selectors=list(_selector_tuple(runtime, selector)),
              rule=RULE_IDENTICAL_RERUN)
        return False, RULE_IDENTICAL_RERUN
    if action in WORK_ACTIONS and window_open(lock):
        if action == "run_tests":
            # r45 one owner (r44c 364 ldev_09 seq 920): never offer a rerun the first-green disclosure refuses.
            try:
                from kernel_first_green_disclosure import identical_rerun_refused_next
                import kernel_inflight_fixes
                if kernel_inflight_fixes.enabled():
                    # r45 in-flight fix 3: ONE predicate -- verification already reused at this revision.  The
                    # identical-rerun counter it replaces is only written on the DISPOSITION-v1 path (inert in frame).
                    if kernel_inflight_fixes.reused_at(runtime, kernel_inflight_fixes.verification_key(runtime)):
                        kernel_inflight_fixes.emit(runtime, "reused_rerun_dropped", stage="acted")
                        return False, RULE_IDENTICAL_RERUN
                elif identical_rerun_refused_next(runtime):
                    return False, RULE_IDENTICAL_RERUN
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
        return True, RULE_WORK
    if bool(getattr(lock, "terminal_engaged", False)):
        return _named_layer(lock, action, path)
    return None


def owns_lock(lock: object) -> bool:
    """This feature owns the terminal lock: switch on and the menu authority applies."""
    return enabled(getattr(lock, "runtime", None)) and _authority_applies(lock)


def _product_key(binding: object) -> object:
    """The product revision a binding names (the artifact digest when present), JSON-normal."""
    if isinstance(binding, Mapping):
        for key in ("artifact", "product_revision"):
            if binding.get(key) is not None:
                binding = binding.get(key)
                break
    try:
        return json.loads(json.dumps(binding, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return None


def _selector_tuple(runtime: object, selector: object = None) -> tuple[str, ...]:
    if selector not in (None, "", (), [], {}):
        values = selector if isinstance(selector, (list, tuple, set, frozenset)) else (selector,)
        return tuple(sorted({" ".join(str(item).split()) for item in values if str(item).strip()}))
    for name in ("_post_green_test_selectors", "_last_test_selectors", "test_selectors"):
        value = getattr(runtime, name, None)
        if value:
            return _selector_tuple(runtime, value)
    for source_name in ("_readiness_request_decision", "_readiness_decision"):
        source = getattr(runtime, source_name, None)
        values = getattr(source, "selectors", None)
        if values:
            return _selector_tuple(runtime, values)
    owner = getattr(runtime, "owner", None)
    return _selector_tuple(runtime, getattr(owner, "f2p_ids", ())) if owner is not None else ()


def _same_revision(runtime: object, green_revision: object) -> bool:
    if green_revision in (None, ""):
        return False
    binding = getattr(runtime, "binding", None)
    if not callable(binding):
        return False
    try:
        current = _product_key(binding())
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return current is not None and str(current) == str(_product_key(green_revision))


def _v15_identical_green(lock: object, selector: object = None) -> bool:
    runtime = getattr(lock, "runtime", None)
    if runtime is None or not bool(getattr(lock, "terminal_engaged", False)):
        return False
    if pending_verification_active(runtime):
        return False
    if (hostverified_quiet_enabled(runtime) and host_verified_done(runtime)
            and capability_floor_needed(runtime)):
        # Keep verification in the separately admitted floor when other host
        # blockers remain, without making armed pending the reason for the menu.
        return False
    green_revision = getattr(lock, "_green_revision", None)
    if not _same_revision(runtime, green_revision):
        return False
    state = _state(runtime)
    baseline_revision = state.get("green_revision")
    if baseline_revision not in (None, "") and str(baseline_revision) != str(_product_key(green_revision)):
        return False
    current_selectors = _selector_tuple(runtime, selector)
    baseline_selectors = tuple(state.get("green_selectors") or ())
    if baseline_selectors and current_selectors != baseline_selectors:
        return False
    return current_selectors == baseline_selectors


# Stored as the pending binding when runtime.binding() raised: fail closed, never "no binding".
BINDING_UNREADABLE = "__stateframe_binding_unreadable__"


def _binding_now(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    if not callable(binding):
        return None
    try:
        return _product_key(binding())
    except (TypeError, ValueError, AttributeError, OSError):
        return BINDING_UNREADABLE


def note_edit(lock: object, path: object) -> None:
    """Arm explicit verification for every applied product edit; count only window edits."""
    runtime = getattr(lock, "runtime", None)
    if not owns_lock(lock):
        return
    state = _state(runtime)
    in_window = window_open(lock)
    has_lineage = state.get("base") is not None
    seq = int(state.get("edit_seq") or 0) + 1
    state["edit_seq"] = seq
    pending = {"path": str(path), "from_revision": _key(getattr(lock, "_green_revision", None)),
               "edit_seq": seq, "binding": _binding_now(runtime)}
    request_id = getattr(runtime, "current_request_id", None)
    if isinstance(request_id, str) and request_id.strip():
        pending["request_id"] = request_id
    state["pending"] = pending
    if not in_window:
        if has_lineage:
            _emit(runtime, EDIT_EVENT, path=str(path), base=_key(state.get("base")),
                  cycle=cycles_used(runtime), cap=CYCLE_CAP, window=False, edit_seq=seq)
        return
    base = _key(state.get("base") if state.get("base") is not None
                else getattr(lock, "_green_revision", None))
    state["base"] = base
    cycles = dict(state.get("cycles") or {})
    cycles[base] = int(cycles.get(base, 0)) + 1
    state["cycles"] = cycles
    state["lineage"] = True
    _emit(runtime, EDIT_EVENT, path=str(path), base=base, cycle=cycles[base], cap=CYCLE_CAP,
          window=True, edit_seq=seq, open_clause_ids=list(open_clause_ids(runtime)))


def receipt_verifies(pending: Mapping[str, object] | None,
                     receipt: Mapping[str, object] | None) -> bool:
    """A receipt discharges pending verification only if a test subprocess really executed
    (integer exit code), the binding was stable across the run, and it tested the revision the
    latest edit produced.  Red receipts qualify too (that revision WAS tested)."""
    if not isinstance(pending, Mapping) or not isinstance(receipt, Mapping):
        return False
    expected = pending.get("binding")
    edit_request_id = pending.get("request_id")
    if isinstance(edit_request_id, str) and edit_request_id.strip():
        receipt_request_id = receipt.get("request_id")
        if not isinstance(receipt_request_id, str) or not receipt_request_id.strip():
            return False
        # b1420/b1424 (3/3 false post_green_edit_unverified refusals): the host runs the public tests
        # right after the edit IN THE SAME REQUEST. Same-request evidence is admissible only when the
        # edit-time binding is KNOWN and the receipt's stable binding equals it (checked below), i.e.
        # the receipt provably tested the edited revision. Without a known binding, keep refusing.
        if receipt_request_id == edit_request_id and (expected is None or expected == BINDING_UNREADABLE):
            return False
    exit_code = receipt.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return False
    if receipt.get("binding_stable") is not True:
        return False
    binding = receipt.get("binding")
    if binding is None or binding != receipt.get("after_binding"):
        return False
    expected = pending.get("binding")
    if expected == BINDING_UNREADABLE:
        return False                             # edit-time binding unknown: cannot prove it was tested
    return expected is None or _product_key(binding) == expected


def note_test_receipt(runtime: object, receipt: Mapping[str, object] | None = None) -> None:
    """A qualifying test receipt for the latest edited revision closes pending verification."""
    if not enabled(runtime):
        return
    try:
        from kernel_postgreen_protocol import observe_verification
        observe_verification(runtime, receipt)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    state = _state(runtime)
    pending = state.get("pending")
    if pending is None:
        return
    if not receipt_verifies(pending, receipt):
        _emit(runtime, VERIFY_EVENT, verified=False, pending=dict(pending),
              exit_code=receipt.get("exit_code") if isinstance(receipt, Mapping) else None,
              binding_stable=receipt.get("binding_stable") if isinstance(receipt, Mapping) else None)
        return
    _emit(runtime, VERIFY_EVENT, verified=True, exit_code=receipt.get("exit_code"),
          pending=dict(pending))
    state["pending"] = None


def note_same_request_host_receipt(runtime: object, receipt: Mapping[str, object] | None) -> bool:
    """Discharge pending verification from the host's own test run in the EDIT's request (b1420/b1424).

    The host runs the public checks right after an applied edit, in the same request. Those receipts are
    not "explicit" (the model did not call run_tests), so they never reached note_test_receipt and every
    later done was refused post_green_edit_unverified on an unchanged green candidate (3/3 live dones).
    This path is deliberately narrow: only a stable host receipt from the edit's own request whose binding
    equals the KNOWN edit-time binding (receipt_verifies) clears pending; nothing else is fed.
    """
    if not enabled(runtime) or not isinstance(receipt, Mapping):
        return False
    state = _state(runtime)
    pending = state.get("pending")
    if not isinstance(pending, Mapping):
        return False
    if receipt.get("producer") != "test_subprocess" or receipt.get("request_id") != pending.get("request_id"):
        return False
    if host_verified_done(runtime):
        return False
    if not receipt_verifies(pending, receipt):
        return False
    _emit(runtime, VERIFY_EVENT, verified=True, exit_code=receipt.get("exit_code"),
          pending=dict(pending), source="host_same_request")
    # CH-04 (strict startup smoke): clearing pending here also removed run_tests from the post-green work menu,
    # while a pass-gates DEPENDENCY_VETO still needed run_tests as its resolver -> gate_surface_unqualified ->
    # wire truth BLOCKED. Keep pending armed (menu/pass-gates exactly as before) and record the host
    # verification; only the DONE refusal consults it (host_verified_done).
    state["host_verified"] = {"edit_seq": pending.get("edit_seq"), "binding": pending.get("binding")}
    return True


def host_verified_done(runtime: object) -> bool:
    """The pending post-green edit was verified by the host's own same-request run (done may proceed)."""
    if not enabled(runtime):
        return False
    state = _state(runtime)
    pending, verified = state.get("pending"), state.get("host_verified")
    return (isinstance(pending, Mapping) and isinstance(verified, Mapping)
            and verified.get("edit_seq") == pending.get("edit_seq")
            and verified.get("binding") == pending.get("binding"))


def note_reused_verification(
        runtime: object,
        verdict: object,
        selectors: Sequence[str],
) -> bool:
    """Discharge pending verification from an exact, trusted current-green receipt.

    Reuse is sufficient only when the shared verdict is SATISFIED, its binding
    is the live binding, every requested selector is covered by a trusted
    receipt, and the pending edit points at that same product revision. Any
    uncertainty leaves ``pending`` armed so the real dispatcher must execute.
    """
    if (hostverified_quiet_enabled(runtime) and host_verified_done(runtime)):
        return False
    if not enabled(runtime) or not pending_verification(runtime):
        return False
    binding_fn = getattr(runtime, "binding", None)
    if not callable(binding_fn) or getattr(verdict, "state", None) != "SATISFIED":
        return False
    try:
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    if current is None or getattr(verdict, "current_binding", None) != current:
        return False
    requested = tuple(dict.fromkeys(str(item).strip() for item in selectors if str(item).strip()))
    receipts = tuple(getattr(verdict, "trusted_receipts", ()) or ())
    covered = {
        str(selector)
        for receipt in receipts
        if (getattr(receipt, "binding", None) == current
            and getattr(receipt, "after_binding", None) == current
            and getattr(receipt, "exit_code", None) == 0
            and getattr(receipt, "binding_stable", None) is True)
        for selector in (getattr(receipt, "selectors", ()) or ())
    }
    if not requested or not set(requested).issubset(covered):
        return False
    state = _state(runtime)
    pending = state.get("pending")
    synthetic_receipt = {
        "exit_code": 0,
        "binding_stable": True,
        "binding": current,
        "after_binding": current,
    }
    if not receipt_verifies(pending, synthetic_receipt):
        return False
    state["pending"] = None
    _emit(runtime, VERIFY_EVENT, verified=True, reused=True,
          selectors=list(requested), pending=dict(pending))
    return True


def reopen_instead_of_sole_submission(runtime: object, revision: object,
                                      clause_ids: Sequence[str], *, named_layer: bool = False) -> bool:
    """Done arbiter hook (rule 2): True = do NOT accept the sole done; challenge and reopen.

    Once per revision, only while the authority owns the menu (so the reopened tools really
    reach the model) and the lineage is under the cap.
    """
    if not enabled(runtime):
        return False
    ids = [str(item) for item in clause_ids or () if str(item).strip()]
    if not ids:
        return False
    lock = getattr(runtime, "green_lock", None)
    # named_layer (STATEFRAME_NAMED_LAYER_REPAIR rule D): the reopened tools come from the named-layer rule,
    # which is not bounded by the edit-cycle cap; the once-per-revision latch below still prevents a deadlock.
    capped = cap_reached(runtime) and not named_layer
    if lock is None or not _authority_applies(lock) or _stopped(lock) or capped:
        return False
    state = _state(runtime)
    key = _key(revision)
    if key in set(state.get("reopened") or ()):
        return False
    state["reopened"] = [*list(state.get("reopened") or ()), key]
    green = _key(getattr(lock, "_green_revision", None))
    if green:
        state["reopened_green"] = [*list(state.get("reopened_green") or ()), green]
    _emit(runtime, REOPEN_EVENT, revision=key, green_revision=green, open_clause_ids=ids,
          cycles=cycles_used(runtime), cap=CYCLE_CAP, **({"named_layer": True} if named_layer else {}))
    return True


def note_menu(runtime: object, offered: Sequence[str], rules: Mapping[str, str]) -> None:
    """Journal the sanction the replay gate checks: which offered tools this rule added."""
    if not enabled(runtime):
        return
    sanctioned = sorted(name for name, rule in rules.items()
                        if rule in {RULE_WORK, RULE_VERIFY} and name in offered)
    denied = sorted(name for name, rule in rules.items()
                    if rule in {RULE_UNVERIFIED_DONE, RULE_NO_EVIDENCE})
    if not sanctioned and not denied:
        return
    lock = getattr(runtime, "green_lock", None)
    _emit(runtime, MENU_EVENT, menu_schema=SCHEMA, offered=list(offered), sanctioned=sanctioned,
          denied=denied, open_clause_ids=list(open_clause_ids(runtime)),
          cycles=cycles_used(runtime), cap=CYCLE_CAP,
          pending_verification=pending_verification(runtime),
          green_revision=_key(getattr(lock, "_green_revision", None)))


# --------------------------------------------------------------------------- green notice

def green_notice_head(runtime: object) -> str:
    """Rule 3: the public-green line never reads as a completion verdict."""
    ids = open_clause_ids(runtime)
    noun = "ticket clause" if len(ids) == 1 else "ticket clauses"
    if ids:
        return (f"public tests passed (public only; {len(ids)} {noun} still open: "
                + ", ".join(ids) + ")")
    return "public tests passed (public only; 0 ticket clauses still open)"


# --------------------------------------------------------------------------- clause text

def wrap_invariant_clauses(ticket: str) -> tuple[str, ...]:
    """Rule 4: logical ticket clauses; soft line wraps never split a sentence.

    Reuses the oracle-grounding span extractor (list items and fenced code stay atomic; blank
    lines and Markdown hard breaks remain boundaries).  With this switch on its ordered-list
    detection follows CommonMark: only ``1.``/``1)`` may interrupt a paragraph, so a wrapped
    line that starts with ``404.`` continues the sentence.
    """
    from kernel_oracle_grounding import extract_ticket_clauses
    return extract_ticket_clauses(ticket)


__all__ = [
    "CYCLE_CAP", "EDIT_EVENT", "MENU_EVENT", "MODE_ENV", "REOPEN_EVENT", "RULE_NO_EVIDENCE",
    "RULE_UNVERIFIED_DONE",
    "host_verified_done",
    "RULE_VERIFY", "RULE_WORK", "SCHEMA", "UNVERIFIED_DONE_REFUSAL", "VERIFY_EVENT",
    "V15_IDENTICAL_EVENT",
    "WORK_ACTIONS", "cap_reached", "cycles_used", "decide", "enabled", "green_notice_head",
    "mode", "note_edit", "note_engaged", "note_menu", "note_reused_verification",
    "note_test_receipt", "open_clause_ids",
    "pending_verification", "reopen_instead_of_sole_submission", "state_snapshot",
    "owns_lock", "receipt_verifies", "window_actions", "window_admitted_files",
    "window_dispositions", "window_open",
    "wrap_invariant_clauses",
]


def window_dispositions(lock: object) -> dict[str, str]:
    """Return the post-green authority verdict for each surfaced action.

    ``window_actions`` intentionally exposes only allowed work for legacy menu
    callers.  Readiness also needs an explicit ``done`` denial, so this mapping
    carries both sides of the same ``decide`` predicate.
    """
    if lock is None:
        return {}
    out: dict[str, str] = {}
    for action in (*sorted(WORK_ACTIONS), "done"):
        verdict = decide(lock, action)
        if verdict is not None:
            out[action] = "allow" if verdict[0] else "deny"
    return out


def window_admitted_files(lock: object) -> tuple[str, ...]:
    """Return named-layer files admitted by the same post-green predicate, if scoped.

    The ordinary pre-cap work window is path-independent, so it has no file
    scope to report.  Once that window closes, Rule B/D may admit only the
    untouched named-layer files; exposing those files makes a refusal useful
    without creating a second allow predicate.
    """
    if lock is None:
        return ()
    try:
        import kernel_named_layer_repair as named_layer
        if not named_layer.enabled() or window_open(lock):
            return ()
        verdict = decide(lock, "edit")
        if verdict != (True, named_layer.RULE_NAMED_LAYER):
            return ()
        return tuple(dict.fromkeys(named_layer.untouched_files(getattr(lock, "runtime", None))))
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return ()


def window_actions(lock: object) -> tuple[str, ...]:
    """Actions the post-green rule admits now -- the one predicate readiness defers to (r45 one owner)."""
    if lock is None:
        return ()
    out = []
    for action in sorted(WORK_ACTIONS):
        verdict = decide(lock, action)
        if verdict is not None and verdict[0]:
            out.append(action)
    return tuple(out)
