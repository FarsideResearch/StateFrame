"""Shared, host-owned budget for done challenges.

The two done-challenge arms predate this module and each kept a small counter on
the Python runtime.  That is not enough for a run-wide policy: a restart can
create a fresh counter, and the two arms can disagree about what was spent.
This module is deliberately small and initiator-neutral.  It owns:

* strict parsing of ``STATEFRAME_CHALLENGE_BUDGET``;
* an artifact-byte revision identity;
* the monotone state machine (challenge, submit, or cap);
* request-id deduplication for replayed transport deliveries; and
* a journal/checkpoint-safe host-state projection.

No function in this module is called by the legacy path unless the new budget
is configured.  That keeps the switch-off path byte-identical.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path


MODE_ENV = "STATEFRAME_CHALLENGE_BUDGET"
STATE_SCHEMA = "stateframe-challenge-budget/1"
STATE_EVENT = "challenge_budget_state"
CHALLENGE_ENDING = (
    "Do not change correct code merely to agree. All tools remain available. "
    "Calling done again at this revision submits it."
)

_MISSING = object()
_IGNORED_PARTS = frozenset({
    ".git", ".pytest_cache", "__pycache__", ".test-tmp", ".mypy_cache",
    ".ruff_cache", ".hypothesis", ".stateframe", ".tox", ".nox", "htmlcov",
    "build", "dist", "test-results",
})
_IGNORED_FILES = frozenset({
    ".coverage", "coverage.xml", "coverage.json", "junit.xml", "test-results.xml",
})
_COUNTEREXAMPLE_CODES = frozenset({
    "REQUIREMENT_ALIGNMENT_OPEN", "READINESS_COUNTEREXAMPLE_OPEN",
    "GET_SURE_OPEN_CLAUSE", "OPEN_COUNTEREXAMPLE",
    "GREEN_LOCK_REPAIR_OPEN_COUNTEREXAMPLE", "PRODUCT_VIOLATION",
})
_CLOSED_STATUSES = frozenset({
    "SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED",
    "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED",
})
_VIOLATION_STATUSES = frozenset({
    "VIOLATED", "UNSATISFIED", "FAIL", "FAILED", "RED", "ERROR",
    "COUNTEREXAMPLE", "PRODUCT_RED", "REGRESSION", "UNSATISFIED",
})
_UNCERTAIN_STATUSES = frozenset({
    "UNRESOLVED", "UNKNOWN", "STALE", "INCONCLUSIVE", "INVALID", "UNBOUND",
    "SIGNATURE_MISMATCH", "NO_VERIFIER", "NOT_EXECUTED", "HARNESS_ERROR", "SETUP_ERROR", "SKIPPED",
    "CONFLICT", "CONFLICTING_RECEIPTS", "ADVISORY_ONLY",
})


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def configured_budget(environ: Mapping[str, str] | None = None) -> int | None:
    """Return the configured cap, or ``None`` for the exact off path."""
    source = os.environ if environ is None else environ
    raw = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if raw in {"", "off", "0", "none", "false"}:
        return None
    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{MODE_ENV} must be off or a positive integer") from error
    if value <= 0:
        raise ValueError(f"{MODE_ENV} must be off or a positive integer")
    return value


def budget_for(runtime: object, environ: Mapping[str, str] | None = None) -> int | None:
    """Read the host's frozen run cap, falling back to the process switch.

    Real runtimes freeze the value during construction.  Small unit fixtures
    may only provide the environment, so an absent attribute remains dynamic.
    """
    value = getattr(runtime, "stateframe_challenge_budget", _MISSING)
    if value is _MISSING:
        return configured_budget(environ)
    if value in (None, "", 0, "off", "OFF"):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{MODE_ENV} must be off or a positive integer") from error
    if value <= 0:
        raise ValueError(f"{MODE_ENV} must be off or a positive integer")
    return value


def enabled(runtime: object | None = None) -> bool:
    return configured_budget() is not None if runtime is None else budget_for(runtime) is not None


def _root(runtime: object) -> Path | None:
    owner = getattr(runtime, "owner", None)
    value = getattr(owner, "root", None) if owner is not None else getattr(runtime, "root", None)
    if value in (None, ""):
        return None
    try:
        path = Path(value)
    except (TypeError, ValueError):
        return None
    return path if path.is_dir() else None


def _excluded_paths(runtime: object, root: Path) -> set[Path]:
    excluded: set[Path] = set()
    journal = getattr(runtime, "journal", None)
    journal_path = getattr(journal, "path", None)
    if journal_path is not None:
        try:
            excluded.add(Path(journal_path).resolve())
        except (OSError, TypeError, ValueError):
            pass
    workspace = getattr(runtime, "workspace", None)
    workspace_dir = getattr(workspace, "directory", None)
    if workspace_dir is not None:
        try:
            excluded.add(Path(workspace_dir).resolve())
        except (OSError, TypeError, ValueError):
            pass
    output = getattr(runtime, "_challenge_budget_output", None)
    if output is not None:
        try:
            excluded.add(Path(output).resolve())
        except (OSError, TypeError, ValueError):
            pass
    return {item for item in excluded if item == root or item.is_relative_to(root)}


def _artifact_files(runtime: object, root: Path) -> dict[str, str]:
    """Hash source/config/test bytes, excluding generated/cache and host state."""
    excluded = _excluded_paths(runtime, root)
    result: dict[str, str] = {}
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except (OSError, ValueError):
        return result
    for path in paths:
        try:
            relative = path.relative_to(root)
            resolved = path.resolve()
            if any(part in _IGNORED_PARTS for part in relative.parts):
                continue
            if relative.name in _IGNORED_FILES:
                continue
            if resolved in excluded or any(item in resolved.parents for item in excluded):
                continue
            if path.is_symlink():
                # Do not follow a link outside the task root.  Link identity is
                # still an artifact change and therefore deserves a new revision.
                result[relative.as_posix()] = "symlink:" + str(path.readlink())
            elif path.is_file():
                with path.open("rb") as stream:
                    result[relative.as_posix()] = "sha256:" + hashlib.file_digest(
                        stream, "sha256").hexdigest()
        except (OSError, ValueError, UnicodeError):
            continue
    return result


def _binding_fallback(runtime: object, basis: object = None) -> object:
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
            if isinstance(value, Mapping):
                relevant = {}
                for key in ("product_files", "harness_files", "artifact",
                            "product_revision", "test_harness_revision"):
                    if key in value:
                        relevant[key] = value[key]
                if relevant:
                    if set(relevant) == {"artifact"}:
                        return relevant["artifact"]
                    return relevant
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    if basis is not None:
        value = getattr(basis, "binding_digest", None)
        if value not in (None, ""):
            return value
    return None


def artifact_revision(runtime: object, *, basis: object = None,
                      fallback: object = None) -> str:
    """Hash actual in-scope artifact bytes, never receipts or conversation text."""
    root = _root(runtime)
    if root is not None:
        files = _artifact_files(runtime, root)
        if files:
            return _digest({"files": files})
    target = getattr(runtime, "target", None)
    try:
        target_path = Path(target) if target not in (None, "") else None
        if target_path is not None and target_path.is_file():
            return "sha256:" + hashlib.sha256(target_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        pass
    value = _binding_fallback(runtime, basis)
    if value not in (None, ""):
        return value if isinstance(value, str) and value.startswith("sha256:") else _digest(value)
    if fallback not in (None, ""):
        return str(fallback)
    return "sha256:" + hashlib.sha256(b"<unbound-artifact>").hexdigest()


def new_state(cap: int) -> dict[str, object]:
    return {
        "payload_schema": STATE_SCHEMA,
        "cap": int(cap),
        "spent": 0,
        "challenged_revision": None,
        "last_challenge_class": None,
        "request_outcomes": {},
        "submission_record": None,
        "terminal_result": None,
        "terminal": False,
    }


def _normalize_state(raw: object, cap: int) -> dict[str, object]:
    state = new_state(cap)
    if isinstance(raw, Mapping):
        saved_cap = raw.get("cap")
        if isinstance(saved_cap, int) and saved_cap > 0:
            state["cap"] = saved_cap
        spent = raw.get("spent", 0)
        if isinstance(spent, int) and spent >= 0:
            state["spent"] = min(spent, int(state["cap"]))
        revision = raw.get("challenged_revision")
        state["challenged_revision"] = None if revision in (None, "") else str(revision)
        klass = raw.get("last_challenge_class")
        state["last_challenge_class"] = None if klass in (None, "") else str(klass)
        outcomes = raw.get("request_outcomes")
        if isinstance(outcomes, Mapping):
            state["request_outcomes"] = {
                str(key): str(value) for key, value in list(outcomes.items())[-128:]
                if str(key).strip() and str(value).strip()
            }
        record = raw.get("submission_record")
        state["submission_record"] = copy.deepcopy(dict(record)) if isinstance(record, Mapping) else None
        result = raw.get("terminal_result")
        state["terminal_result"] = copy.deepcopy(dict(result)) if isinstance(result, Mapping) else None
        state["terminal"] = bool(raw.get("terminal", False))
        # STATEFRAME_SUBMIT_IDEMPOTENT: these keys exist only when that switch
        # wrote them, so an off-mode state normalizes exactly as before.
        run = raw.get("submission_run")
        if isinstance(run, str) and run.strip():
            state["submission_run"] = run
        intent = raw.get("submission_intent")
        if isinstance(intent, Mapping) and _valid_slot(intent):
            state["submission_intent"] = copy.deepcopy(dict(intent))
    return state


def _state(runtime: object, *, create: bool = True) -> dict[str, object] | None:
    cap = budget_for(runtime)
    if cap is None:
        return None
    raw = getattr(runtime, "_challenge_budget_state", None)
    if raw is None and not create:
        return None
    state = _normalize_state(raw, cap)
    setattr(runtime, "_challenge_budget_state", state)
    return state


def state_snapshot(runtime: object) -> dict[str, object]:
    state = _state(runtime)
    return copy.deepcopy(state) if state is not None else {}


def request_id(runtime: object, explicit: object = None) -> str | None:
    candidates = [explicit, getattr(runtime, "current_tool_call_id", None)]
    capture = getattr(runtime, "_active_capture", None)
    candidates.append(getattr(capture, "request_id", None))
    action = getattr(runtime, "current_action", None)
    if isinstance(action, Mapping):
        candidates.extend(action.get(key) for key in ("request_id", "tool_call_id", "id"))
    owner = getattr(runtime, "owner", None)
    candidates.extend(getattr(owner, key, None) for key in ("current_tool_call_id", "request_id"))
    for value in candidates:
        if value not in (None, ""):
            return str(value)
    return None


def _idempotent(runtime: object) -> bool:
    try:
        from kernel_submission_idempotency import enabled as idempotency_enabled
    except ImportError:
        return False
    return bool(idempotency_enabled(runtime))


def _valid_slot(slot: object) -> bool:
    if not isinstance(slot, Mapping):
        return False
    key = slot.get("key")
    return (isinstance(key, (list, tuple)) and len(key) == 2
            and all(isinstance(item, str) and item for item in key)
            and slot.get("effect_status") in {"pending", "confirmed"})


def slot_key(slot: Mapping[str, object]) -> tuple[str, str]:
    key = slot["key"]
    return str(key[0]), str(key[1])


def logical_run(runtime: object) -> str:
    """The run half of the submission key; persisted once, reused after restore."""
    state = _state(runtime, create=False) or {}
    run = state.get("submission_run")
    if isinstance(run, str) and run:
        return run
    value = getattr(getattr(runtime, "journal", None), "run_id", None)
    return str(value) if value not in (None, "") else "unbound-run"


def submission_key(runtime: object, revision: object) -> tuple[str, str]:
    return logical_run(runtime), str(revision)


def submission_slot(runtime: object) -> dict[str, object] | None:
    """The durable in-flight submission intent (idempotency switch only)."""
    if not _idempotent(runtime):
        return None
    state = _state(runtime, create=False)
    slot = (state or {}).get("submission_intent")
    return copy.deepcopy(dict(slot)) if _valid_slot(slot) else None


def _json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return repr(value)


def reserve_submission_intent(runtime: object, key: tuple[str, str], *,
                              destination: str, recovery_context: object = None) -> dict[str, object]:
    """Persist a pending intent BEFORE the keyed effect runs."""
    state = _state(runtime)
    if state is None:
        from kernel_submission_idempotency import validate_configuration
        validate_configuration("on", None)
    if state.get("terminal"):
        raise RuntimeError("a terminal submission cannot reserve a new intent")
    state["submission_run"] = str(key[0])
    state["submission_intent"] = {
        "schema": "stateframe-submission-intent/1",
        "key": [str(key[0]), str(key[1])],
        "revision": str(key[1]),
        "effect_status": "pending",
        "result": None,
        "destination": str(destination),
        "recovery_context": _json_safe(recovery_context),
    }
    _persist(runtime, state)
    return copy.deepcopy(state["submission_intent"])


def mark_submission_effect(runtime: object, key: tuple[str, str], *,
                           result: object) -> dict[str, object]:
    """Persist the destination's confirmation; this is not terminal state."""
    state = _state(runtime)
    slot = (state or {}).get("submission_intent")
    if not _valid_slot(slot) or slot_key(slot) != (str(key[0]), str(key[1])):
        raise RuntimeError("no durable submission intent matches the confirmed key")
    slot = copy.deepcopy(dict(slot))
    slot["effect_status"] = "confirmed"
    slot["result"] = _json_safe(result)
    state["submission_intent"] = slot
    _persist(runtime, state)
    return copy.deepcopy(slot)


def note_callback_free_attempt(runtime: object, key: tuple[str, str]) -> int:
    """Durably count callback-free dones against a pending intent."""
    state = _state(runtime)
    slot = (state or {}).get("submission_intent")
    if not _valid_slot(slot) or slot_key(slot) != (str(key[0]), str(key[1])):
        raise RuntimeError("no durable submission intent matches the pending key")
    slot = copy.deepcopy(dict(slot))
    slot["callback_free_attempts"] = int(slot.get("callback_free_attempts", 0) or 0) + 1
    state["submission_intent"] = slot
    _persist(runtime, state)
    return int(slot["callback_free_attempts"])


def terminalize_unresolved(runtime: object, info: Mapping[str, object], decision: object) -> dict:
    """Explicit terminal KEYED_SUBMISSION_UNRESOLVED: never a silent local success."""
    state = _state(runtime)
    slot = state.get("submission_intent")
    record = {
        "attempt_status": "unresolved",
        "assessment": "keyed_submission_unresolved",
        "open_clause_ids": [],
        "revision": str((slot or {}).get("revision", "")),
        "challenge_class": "NONE",
        "override_reason": "keyed_submission_unresolved",
        "run_challenge_count": int(state["spent"]),
        "run_challenge_cap": int(state["cap"]),
        "submission_effect": {"kind": "keyed_unresolved",
                              "intent": copy.deepcopy(slot), **copy.deepcopy(dict(info))},
    }
    state["submission_record"] = record
    state["submission_intent"] = None
    payload = decision.to_dict() if hasattr(decision, "to_dict") else decision
    state["terminal_result"] = copy.deepcopy(dict(payload)) if isinstance(payload, Mapping) else None
    state["terminal"] = True
    _persist(runtime, state)
    setattr(runtime, "_challenge_budget_submission_record", copy.deepcopy(record))
    setattr(runtime, "challenge_budget_submission_record", copy.deepcopy(record))
    setattr(runtime, "challenge_budget_terminal", True)
    setattr(runtime, "_challenge_budget_terminal_result",
            copy.deepcopy(state.get("terminal_result")))
    return copy.deepcopy(record)


def withdraw_submission_intent(runtime: object, key: tuple[str, str], *, reason: str) -> None:
    """Drop an intent the destination proved never took effect."""
    state = _state(runtime)
    slot = (state or {}).get("submission_intent")
    if not _valid_slot(slot) or slot_key(slot) != (str(key[0]), str(key[1])):
        return
    state["submission_intent"] = None
    _persist(runtime, state)


def _terminal_submission_effect(runtime: object, state: dict[str, object],
                                revision: str) -> dict[str, object]:
    """Resolve the in-flight intent into the terminal record (same journal write)."""
    slot = state.get("submission_intent")
    state["submission_intent"] = None
    run = state.get("submission_run")
    if not isinstance(run, str) or not run:
        run = logical_run(runtime)
        state["submission_run"] = run
    if _valid_slot(slot) and slot.get("effect_status") == "confirmed":
        return {"kind": "keyed", "key": list(slot_key(slot)), "result": slot.get("result")}
    effect = {"kind": "local", "key": [run, str(revision)]}
    if _valid_slot(slot):
        # A callback-free acceptance while a keyed intent is still pending: the
        # caller offered no destination to resolve it against.  Record it, do
        # not pretend it was confirmed and do not block liveness.
        effect["unresolved_intent"] = copy.deepcopy(dict(slot))
    return effect


def plan(runtime: object, revision: object, *, request: object = None) -> dict[str, object] | None:
    """Plan a monotone transition without changing state."""
    state = _state(runtime)
    if state is None:
        return None
    key = str(revision)
    request_key = request_id(runtime, request)
    previous = (state["request_outcomes"].get(request_key)
                if request_key is not None else None)
    if state.get("terminal"):
        action = "terminal_replay" if previous is not None else "terminal"
        return {"action": action, "request_id": request_key, "previous": previous,
                "revision": key, "spent": state["spent"], "cap": state["cap"]}
    slot = state.get("submission_intent")
    if _valid_slot(slot) and slot_key(slot)[1] == key and _idempotent(runtime):
        # A submission already started at this revision: any later done here
        # resolves it against the keyed destination, never re-challenges.
        return {"action": "reconcile", "request_id": request_key, "previous": previous,
                "revision": key, "spent": state["spent"], "cap": state["cap"]}
    if previous:
        return {"action": "replay", "request_id": request_key, "previous": previous,
                "revision": key, "spent": state["spent"], "cap": state["cap"]}
    if state.get("challenged_revision") == key:
        action = "submit"
    elif int(state.get("spent", 0)) >= int(state["cap"]):
        action = "cap"
    else:
        action = "challenge"
    return {"action": action, "request_id": request_key, "revision": key,
            "spent": state["spent"], "cap": state["cap"]}


def _persist(runtime: object, state: Mapping[str, object]) -> None:
    frozen = copy.deepcopy(dict(state))
    setattr(runtime, "_challenge_budget_state", frozen)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            STATE_EVENT,
            producer="kernel_challenge_budget",
            payload_schema=STATE_SCHEMA,
            budget_state=frozen,
            spent=int(frozen["spent"]),
            cap=int(frozen["cap"]),
            challenged_revision=frozen.get("challenged_revision"),
            terminal=bool(frozen.get("terminal", False)),
            dispatch=False,
        )


def commit_challenge(runtime: object, plan_result: Mapping[str, object], *,
                     challenge_class: str) -> dict[str, object]:
    state = _state(runtime)
    if (state is None or state.get("terminal")
            or plan_result.get("action") != "challenge"):
        return state_snapshot(runtime)
    state["spent"] = int(state["spent"]) + 1
    state["challenged_revision"] = str(plan_result["revision"])
    state["last_challenge_class"] = str(challenge_class)
    request_key = plan_result.get("request_id")
    if request_key is not None:
        state["request_outcomes"][str(request_key)] = "challenge"
    _persist(runtime, state)
    return copy.deepcopy(state)


def commit_submission(runtime: object, plan_result: Mapping[str, object], *,
                      attempt_status: str, assessment: str,
                      open_clause_ids: tuple[str, ...] = (),
                      challenge_class: str = "UNVERIFIED", terminal_result: object = None,
                      override_reason: str | None = None) -> dict[str, object]:
    state = _state(runtime)
    if state is None:
        return {}
    if state.get("terminal"):
        return copy.deepcopy(state.get("submission_record") or {})
    if attempt_status not in {"submitted", "capped"}:
        raise ValueError("attempt_status must be submitted or capped")
    record = {
        "attempt_status": attempt_status,
        "assessment": assessment,
        "open_clause_ids": list(dict.fromkeys(str(item) for item in open_clause_ids
                                                if str(item).strip())),
        "revision": str(plan_result.get("revision", "")),
        "challenge_class": str(challenge_class),
        "override_reason": (str(override_reason) if override_reason else
                             ("second_done_same_revision"
                              if attempt_status == "submitted" else "run_challenge_cap")),
        "run_challenge_count": int(state["spent"]),
        "run_challenge_cap": int(state["cap"]),
    }
    if getattr(runtime, "stateframe_false_completion_mode", "off") == "on":
        try:
            from kernel_false_completion import accepted_submission_fields
            record.update(accepted_submission_fields(runtime))
        except (ImportError, OSError, TypeError, ValueError):
            pass
    if _idempotent(runtime):
        record["submission_effect"] = _terminal_submission_effect(
            runtime, state, record["revision"])
    state["submission_record"] = record
    if terminal_result is not None:
        if hasattr(terminal_result, "to_dict") and callable(terminal_result.to_dict):
            terminal_result = terminal_result.to_dict()
        state["terminal_result"] = (copy.deepcopy(dict(terminal_result))
                                     if isinstance(terminal_result, Mapping) else None)
    state["terminal"] = True
    request_key = plan_result.get("request_id")
    if request_key is not None:
        state["request_outcomes"][str(request_key)] = attempt_status
    _persist(runtime, state)
    setattr(runtime, "_challenge_budget_submission_record", copy.deepcopy(record))
    setattr(runtime, "challenge_budget_submission_record", copy.deepcopy(record))
    setattr(runtime, "challenge_budget_terminal", True)
    setattr(runtime, "_challenge_budget_terminal_result",
            copy.deepcopy(state.get("terminal_result")))
    return copy.deepcopy(record)


def remember_decision(runtime: object, plan_result: Mapping[str, object], decision: object) -> None:
    request_key = plan_result.get("request_id")
    if request_key is None:
        return
    cache = getattr(runtime, "_challenge_budget_decisions", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(runtime, "_challenge_budget_decisions", cache)
    cache[str(request_key)] = decision


def replay_decision(runtime: object, plan_result: Mapping[str, object]) -> object:
    cache = getattr(runtime, "_challenge_budget_decisions", {})
    if isinstance(cache, Mapping):
        return cache.get(str(plan_result.get("request_id")))
    return None


def restore_state(runtime: object, *, records: object = (), checkpoint_state: object = None) -> dict[str, object]:
    """Restore the latest durable budget state; journal state wins checkpoint state."""
    state = durable_state(runtime, records=records, checkpoint_state=checkpoint_state)
    if state is None:
        return {}
    setattr(runtime, "_challenge_budget_state", state)
    setattr(runtime, "_done_challenge_count", int(state["spent"]))
    setattr(runtime, "_done_challenge_revision", state.get("challenged_revision"))
    setattr(runtime, "_hstate_plus_challenge_count", int(state["spent"]))
    setattr(runtime, "_hstate_plus_challenged_revision", state.get("challenged_revision"))
    record = state.get("submission_record")
    if isinstance(record, Mapping):
        setattr(runtime, "_challenge_budget_submission_record", copy.deepcopy(dict(record)))
        setattr(runtime, "challenge_budget_submission_record", copy.deepcopy(dict(record)))
        setattr(runtime, "_done_submission_record", copy.deepcopy(dict(record)))
        setattr(runtime, "_hstate_plus_submission_record", copy.deepcopy(dict(record)))
    setattr(runtime, "_challenge_budget_terminal_result",
            copy.deepcopy(state.get("terminal_result")))
    setattr(runtime, "challenge_budget_terminal", bool(state.get("terminal", False)))
    return copy.deepcopy(state)


def durable_state(runtime: object, *, records: object = (),
                  checkpoint_state: object = None) -> dict[str, object] | None:
    """Project durable budget state without mutating the runtime."""
    cap = budget_for(runtime)
    if cap is None:
        return None
    latest = checkpoint_state if isinstance(checkpoint_state, Mapping) else None
    if isinstance(records, Mapping):
        records = records.values()
    for event in records if records is not None else ():
        if not isinstance(event, Mapping) or event.get("kind") != STATE_EVENT:
            continue
        candidate = event.get("budget_state")
        if isinstance(candidate, Mapping):
            latest = candidate
    return _normalize_state(latest, cap)


def terminal_result(runtime: object) -> dict[str, object] | None:
    state = _state(runtime, create=False)
    value = (state or {}).get("terminal_result")
    if value is None:
        value = getattr(runtime, "_challenge_budget_terminal_result", None)
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else None


def durable_decision(runtime: object, *, fallback_basis: object = None):
    """Rebuild the typed accepted result from durable data, without admission."""
    payload = terminal_result(runtime)
    if not isinstance(payload, Mapping):
        return None
    try:
        from kernel_verdict import CompletionBasis, CompletionDecision, ACCEPTED
        basis_data = payload.get("basis")
        if isinstance(basis_data, Mapping) and all(
                isinstance(basis_data.get(key), str) and basis_data.get(key)
                for key in ("snapshot_digest", "binding_digest", "contract_version", "evidence_digest")):
            basis = CompletionBasis(
                snapshot_digest=basis_data["snapshot_digest"],
                binding_digest=basis_data["binding_digest"],
                contract_version=basis_data["contract_version"],
                evidence_digest=basis_data["evidence_digest"],
                snapshot_id=basis_data.get("snapshot_id"),
            )
        elif fallback_basis is not None:
            basis = fallback_basis
        else:
            return None
        outcome = str(payload.get("outcome") or ACCEPTED)
        accepted = bool(payload.get("accepted", True))
        return CompletionDecision(
            outcome, basis, accepted=accepted,
            reason_codes=tuple(payload.get("reason_codes") or ()),
            detail=str(payload.get("detail") or ""),
            repair_action=payload.get("repair_action"),
            basis_diff=payload.get("basis_diff") or {},
            needs_work_ids=tuple(payload.get("needs_work_ids") or ()),
            artifact_preserved=bool(payload.get("artifact_preserved", True)),
            sealed=bool(payload.get("sealed", True)),
            needs_work_payload=payload.get("needs_work"),
            repair_episode=payload.get("repair_episode"),
            obligation_assessments=tuple(payload.get("obligation_assessments") or ()),
            unresolved_questions=tuple(payload.get("unresolved_questions") or ()),
        )
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def _row_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, Mapping):
                return copy.deepcopy(dict(result))
        except (TypeError, ValueError):
            pass
    result: dict[str, object] = {}
    for key in ("clause_id", "check_id", "obligation_id", "requirement_id", "id",
                "status", "verdict", "result_class", "result", "observed", "actual",
                "expected", "want", "clause"):
        value = getattr(value, key, _MISSING)
        if value is not _MISSING:
            result[key] = copy.deepcopy(value)
    return result


def _row_id(row: Mapping[str, object]) -> str:
    for key in ("clause_id", "check_id", "obligation_id", "requirement_id", "id"):
        value = row.get(key)
        if value not in (None, "") and str(value).strip():
            return str(value)
    return ""


def _row_status(row: Mapping[str, object]) -> str:
    for key in ("verdict", "status", "result_class", "result", "state"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().upper()
    return ""


def _signature_mismatch(row: Mapping[str, object]) -> bool:
    values = [row.get(key) for key in
              ("binding", "verdict", "status", "result_class", "result", "state",
               "classification", "demotion_reason", "classification_reason")]
    for nested_key in ("observed", "observed_values", "counterexample"):
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            values.append(nested.get("binding"))
    return any("SIGNATURE_MISMATCH" in str(value).upper() for value in values if value not in (None, ""))


def _signature_gap(row: Mapping[str, object]) -> bool:
    return any(str(row.get(key, "")).strip().upper() == "SIGNATURE_GAP"
               for key in ("classification", "demotion_reason", "classification_reason"))


def _row_kind(row: Mapping[str, object]) -> str:
    status = _row_status(row)
    if _signature_gap(row):
        return "violation"
    if _signature_mismatch(row):
        return "uncertain"
    if status in _CLOSED_STATUSES:
        return "closed"
    if status in _VIOLATION_STATUSES or any(token in status for token in (
            "VIOLAT", "COUNTEREXAMPLE", "REGRESS", "UNSATISFIED")):
        return "violation"
    if status in _UNCERTAIN_STATUSES or any(token in status for token in (
            "STALE", "INCONCLUSIVE", "UNBOUND", "SIGNATURE_MISMATCH", "UNKNOWN", "NO_VERIFIER",
            "NOT_EXECUTED", "HARNESS", "CONFLICT")):
        return "uncertain"
    # An explicitly identified item with no verdict is an open uncertainty, not
    # an implicit pass.  This is how readiness/alignment IDs retain their debt.
    return "uncertain" if _row_id(row) else "none"


def _iter_rows(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "clause_id", "check_id", "obligation_id", "requirement_id", "id",
                "status", "verdict", "result_class", "result")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    return (value,)


def open_clause_rows(value: object = None, *, rows: object = (),
                     runtime: object = None) -> tuple[dict[str, object], ...]:
    """Return current open evidence, excluding satisfied/closed assessments."""
    result: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(item: object, *, force_id: bool = False) -> None:
        if isinstance(item, (str, bytes, bytearray)):
            ident = str(item)
            row = {"clause_id": ident, "status": "UNRESOLVED"}
        else:
            row = _row_mapping(item)
            ident = _row_id(row)
        if not ident:
            return
        kind = _row_kind(row)
        if kind == "closed" and not force_id:
            return
        row.setdefault("clause_id", ident)
        if not _row_status(row):
            row["status"] = "UNRESOLVED"
        if ident in seen:
            # Prefer demonstrated evidence over a preceding unresolved locator.
            for index, prior in enumerate(result):
                if (_row_id(prior) == ident and _row_kind(prior) != "violation"
                        and kind == "violation"):
                    result[index] = row
            return
        seen.add(ident)
        result.append(row)

    for item in _iter_rows(rows):
        add(item)
    if value is not None:
        mapping = _row_mapping(value)
        ids = (mapping.get("needs_work_ids") or mapping.get("open_clause_ids")
               or mapping.get("unresolved_gating_items"))
        for item in _iter_rows(ids):
            add(item, force_id=True)
        assessments = mapping.get("obligation_assessments")
        for item in _iter_rows(assessments):
            add(item)
    if runtime is not None:
        has_current_evidence = hasattr(runtime, "_challenge_budget_open_evidence")
        for candidate in (getattr(runtime, "_challenge_budget_open_evidence", ()),
                          getattr(runtime, "_challenge_budget_unresolved_gating_items", ())):
            for item in _iter_rows(candidate):
                add(item, force_id=isinstance(item, (str, bytes, bytearray)))
        for candidate in (getattr(runtime, "_readiness_decision", None),
                          getattr(runtime, "_readiness_request_decision", None)):
            if candidate is None:
                continue
            candidate_mapping = _row_mapping(candidate)
            for item in _iter_rows(candidate_mapping.get("blockers")):
                add(item)
            for item in _iter_rows(candidate_mapping.get("blocker_ids")):
                add(item, force_id=True)
        candidates = (getattr(runtime, "_hstate_plus_challenge_rows", ()),)
        if not has_current_evidence:
            candidates = (getattr(runtime, "host_closure_assessments", ()), *candidates)
        for candidate in candidates:
            for item in _iter_rows(candidate):
                add(item)
    return tuple(result)


def clause_ids(value: object = None, *, rows: object = (), runtime: object = None) -> tuple[str, ...]:
    return tuple(_row_id(row) for row in open_clause_rows(value, rows=rows, runtime=runtime)
                 if _row_id(row))


def challenge_class(codes: object = (), rows: object = ()) -> str:
    if isinstance(codes, (str, bytes, bytearray)):
        codes = (codes,)
    values = {str(code).upper() for code in (codes or ())}
    if isinstance(rows, Mapping):
        row_values = ([rows] if any(key in rows for key in (
            "status", "result_class", "verdict", "observed", "actual", "expected", "want"))
                      else list(rows.values()))
    else:
        row_values = list(rows or ())
    evidence = open_clause_rows(rows=rows)
    if any(_row_kind(row) == "violation" for row in evidence):
        return "COUNTEREXAMPLE"
    if any(_row_kind(row) == "uncertain" for row in evidence):
        return "CHECK_INCONCLUSIVE"
    if any("SIGNATURE_GAP" in str(code) for code in values):
        return "COUNTEREXAMPLE"
    if any(any(token in str(code) for token in ("INCONCLUSIVE", "INVALID", "UNBOUND",
                                                "SIGNATURE_MISMATCH",
                                                "STALE", "BINDING", "NO_VERIFIER"))
           for code in values):
        return "CHECK_INCONCLUSIVE"
    for row in row_values:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status") or row.get("result_class") or row.get("verdict") or "").upper()
        if _signature_gap(row):
            return "COUNTEREXAMPLE"
        if _signature_mismatch(row):
            return "CHECK_INCONCLUSIVE"
        if any(token in status for token in ("INCONCLUSIVE", "UNBOUND", "SIGNATURE_MISMATCH",
                                             "UNKNOWN", "NO_VERIFIER")):
            return "CHECK_INCONCLUSIVE"
        if ((row.get("observed") is not None or row.get("actual") is not None)
                and (row.get("expected") is not None or row.get("want") is not None)):
            return "COUNTEREXAMPLE"
    if values & _COUNTEREXAMPLE_CODES:
        return "COUNTEREXAMPLE"
    return "UNVERIFIED"


def assessment_for(klass: str, open_clause_ids: tuple[str, ...] = (),
                   rows: object = ()) -> str:
    klass = str(klass).upper()
    evidence = open_clause_rows(rows=rows)
    if klass == "COUNTEREXAMPLE" or any(_row_kind(row) == "violation" for row in evidence):
        return "known_violation"
    if (klass == "CHECK_INCONCLUSIVE" or klass == "UNVERIFIED"
            and (open_clause_ids or evidence)):
        return "unresolved"
    return "unresolved" if open_clause_ids else "no_known_violation"


def finalize_accepted(runtime: object, result: object, *, revision: object,
                      assessment: str, open_clause_ids: tuple[str, ...] = (),
                      challenge_class: str = "NONE", request: object = None,
                      override_reason: str = "natural_acceptance") -> dict[str, object] | None:
    """Durably record a clean or repaired accepted submission exactly once."""
    if budget_for(runtime) is None:
        return None
    state = _state(runtime)
    if state is None:
        return None
    if state.get("terminal"):
        return copy.deepcopy(state.get("submission_record"))
    key = request_id(runtime, request)
    return commit_submission(
        runtime, {"revision": str(revision), "request_id": key},
        attempt_status="submitted", assessment=assessment,
        open_clause_ids=open_clause_ids, challenge_class=challenge_class,
        terminal_result=result, override_reason=override_reason)


__all__ = [
    "CHALLENGE_ENDING", "MODE_ENV", "STATE_EVENT", "STATE_SCHEMA", "artifact_revision",
    "assessment_for", "budget_for", "challenge_class", "clause_ids", "commit_challenge",
    "commit_submission", "configured_budget", "durable_decision", "durable_state",
    "enabled", "finalize_accepted", "logical_run", "mark_submission_effect", "new_state",
    "open_clause_rows", "plan", "remember_decision", "replay_decision",
    "reserve_submission_intent", "restore_state", "slot_key", "state_snapshot",
    "submission_key", "submission_slot", "terminal_result", "withdraw_submission_intent",
]
