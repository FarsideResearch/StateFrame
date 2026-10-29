"""Host-owned progress protocol for work after a public green.

Post-green work is useful only when it changes the host's evidence state or
reaches the terminal action.  This module owns that distinction at the
green-lock boundary.  It deliberately uses receipt identity and verification
input identity separately: a new receipt for the same clean check is not new
evidence, while a check of an item dirtied by an edit is.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


SCHEMA = "stateframe-postgreen-progress/1"
RULE_NEW_EVIDENCE = "postgreen_new_evidence"
RULE_IDENTICAL = "postgreen_identical_verification"
RULE_CERTIFIED = "postgreen_certified"
RULE_UNCERTIFIED = "postgreen_uncertified"
RULE_TERMINAL = "postgreen_terminal_action"
REFUSAL_CODE = "POSTGREEN_IDENTICAL_VERIFICATION"
STATE_ATTR = "_postgreen_progress_state"

_PASS = frozenset({"GREEN", "PASS", "PASSED", "SUCCESS", "OK"})
_CLOSED = frozenset({
    "SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED",
    "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED",
})
_RECEIPT_ID_KEYS = ("receipt_id", "action_receipt_id", "event_id", "id")
_DYNAMIC_KEYS = frozenset({"turn", "sequence", "timestamp", "elapsed_ns"})


@dataclass(frozen=True, slots=True)
class Decision:
    """One host decision; ``next_action`` is a funded route, not advice."""

    allowed: bool
    rule: str
    message: str = ""
    next_action: str = ""
    funded: bool = False
    evidence: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CompletionEvaluation:
    """One immutable host evaluation shared by preview, dispatch, and refusal."""

    verification_key: str
    progress_epoch: str
    clean: bool
    reason: str
    open_ids: tuple[str, ...]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _semantic(value: object) -> object:
    """Normalize cache inputs without object identity or repr addresses."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _semantic(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_semantic(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return _semantic(converted)
    return {"type": type(value).__name__, "opaque": True}


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


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


def _rows(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(row for item in value if (row := _row(item)))


def _sequence_token(value: object) -> tuple[object, ...]:
    """Cheap append-only identity for host-owned receipt collections.

    Receipt producers append immutable mapping rows.  The token changes for a
    new collection, append, replacement at either boundary, or truncation.  A
    producer that mutates an older row can explicitly clear the runtime cache;
    the conservative fallback below also avoids caching opaque collections.
    """
    generation = getattr(value, "generation", None)
    if type(generation) is int and generation >= 0:
        return (type(value).__name__, "generation", generation)
    if isinstance(value, Mapping):
        values = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            try:
                converted = converter()
            except (AttributeError, TypeError, ValueError):
                converted = None
            if isinstance(converted, Mapping):
                return (type(value).__name__, _digest(converted))
        # Opaque producer objects have no semantic sequence contract. Object
        # identity would make a fresh wrapper look like new evidence.
        return (type(value).__name__, "opaque")
    # Plain lists are legacy producer surfaces without a generation counter.
    # Digesting their semantic rows keeps interior mutation safe; journal-backed
    # producers take the generation branch above and stay incremental.
    return (type(value).__name__, len(values), _digest(_semantic(values)))


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {
            "seen_receipt_ids": [],
            "attempted_receipt_ids": [],
            "verified_keys": [],
            "requested_keys": [],
            "refusals": {},
            "settled": None,
            "last_decision": None,
        }
        setattr(runtime, STATE_ATTR, state)
    return state


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emitter = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emitter):
        try:
            emitter(kind, producer="kernel_postgreen_protocol", event_schema=SCHEMA,
                    dispatch=False, **fields)
        except Exception:  # noqa: BLE001 - telemetry cannot keep the host in a loop
            return


def _active(lock: object) -> bool:
    runtime = getattr(lock, "runtime", None)
    return bool(
        runtime is not None
        and getattr(runtime, "stateframe_post_green_work_menu_mode", "off") == "on"
        and getattr(lock, "terminal_engaged", False)
    )


def _binding(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            value = None
        if value is not None:
            return value
    lock = getattr(runtime, "green_lock", None)
    return getattr(lock, "_green_revision", None)


def _binding_key(value: object) -> str:
    row = _row(value)
    if row:
        # Product-only revision keys are insufficient: harness, environment,
        # policy, authorization scope, and selectors are part of verification
        # identity even when product bytes did not change.
        fields = (
            "artifact", "product_revision", "revision", "digest",
            "product_files", "test_harness_revision", "harness_revision",
            "harness_files", "environment", "environment_revision",
            "policy", "policy_identity", "policy_revision",
            "authorization_scope", "scope_identity", "tenant_identity", "tenant_id",
        )
        identity = {key: row[key] for key in fields if key in row}
        if identity:
            return _digest(identity)
    return _text(value)


def _selector_value(value: object) -> object:
    if isinstance(value, str):
        return (_text(value),) if _text(value) else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(sorted(dict.fromkeys(_text(item) for item in value if _text(item))))
    return ()


def receipt_identity(receipt: Mapping[str, object]) -> str:
    """Return the host identity, never a model-supplied display label."""
    for key in _RECEIPT_ID_KEYS:
        value = _text(receipt.get(key))
        if value:
            return value[:200]
    stable = {key: value for key, value in receipt.items() if key not in _DYNAMIC_KEYS}
    return _digest(stable)


def verification_key(runtime: object, arguments: Mapping[str, object] | None = None,
                     receipt: Mapping[str, object] | None = None) -> str:
    """Identify a verification binding and its input, excluding request noise."""
    args = dict(arguments or {})
    source = dict(receipt or {})
    selectors = args.get("selector")
    if selectors in (None, "", (), [], {}):
        selectors = args.get("selectors")
    if selectors in (None, "", (), [], {}):
        selectors = args.get("target")
    if selectors in (None, "", (), [], {}):
        selectors = source.get("selectors", source.get("selector", source.get("target")))
    input_value = args.get("input", args.get("arguments"))
    if input_value in (None, "", (), [], {}):
        input_value = source.get("input", source.get("arguments", source.get("command")))
    binding = args.get("binding")
    if binding in (None, "", (), [], {}):
        binding = source.get("binding")
    if binding in (None, "", (), [], {}):
        binding = source.get("after_binding")
    if binding in (None, "", (), [], {}):
        binding = _binding(runtime)
    return _digest({
        "binding": _binding_key(binding),
        "selectors": _selector_value(selectors),
        "input": input_value,
    })


def _receipt_rows(runtime: object, *, cache: bool = True) -> tuple[Mapping[str, object], ...]:
    sources = tuple((name, getattr(runtime, name, ())) for name in (
        "test_receipts", "verification_receipts", "probe_receipts", "receipts"))
    source_tokens = _receipt_source_tokens(runtime, sources)
    revision = _binding_key(_binding(runtime)) or "UNKNOWN"
    inputs_digest = _digest(source_tokens)
    cache_key = (revision, inputs_digest)
    cached = getattr(runtime, "_postgreen_receipt_rows_cache", None)
    if cache and isinstance(cached, Mapping) and cached.get("key") == cache_key:
        return tuple(cached.get("rows", ()))
    rows: list[Mapping[str, object]] = []
    for _name, value in sources:
        for row in _rows(value):
            if row not in rows:
                rows.append(row)
    result = tuple(rows)
    if cache:
        setattr(runtime, "_postgreen_receipt_rows_cache", {
            "key": cache_key, "rows": result,
        })
    return result


def _receipt_source_tokens(
        runtime: object,
        sources: Sequence[tuple[str, object]],
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Return cheap append-only source tokens, rebuilding a full token once per change."""
    prior = getattr(runtime, "_postgreen_source_token_cache", None)
    if not isinstance(prior, dict):
        prior = {}
        setattr(runtime, "_postgreen_source_token_cache", prior)
    result: list[tuple[str, tuple[object, ...]]] = []
    for name, value in sources:
        if isinstance(value, Mapping):
            values = tuple(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = tuple(value)
        else:
            values = ()
        # Receipt rows are normally append-only, but the host amends some rows
        # in place after a subprocess closes.  Include semantic content in the
        # input token so identity-only caching cannot reuse stale verification
        # state or alter a later readiness decision.
        shape = (type(value).__name__, len(values), _digest(_semantic(values)))
        cached = prior.get(name)
        if isinstance(cached, Mapping) and cached.get("shape") == shape:
            token = cached.get("token")
        else:
            token = _sequence_token(value)
            prior[name] = {"shape": shape, "token": token}
        result.append((name, token if isinstance(token, tuple) else (token,)))
    return tuple(result)


def _receipt_successful(runtime: object, row: Mapping[str, object]) -> bool:
    status = _text(row.get("result_class") or row.get("status") or row.get("outcome")
                   or row.get("verdict")).upper()
    result = row.get("result")
    if isinstance(result, Mapping):
        status = _text(result.get("result_class") or result.get("status") or status).upper()
    executed = any(row.get(field) is True for field in (
        "executed", "execution_complete", "executor_returned", "current"))
    exit_code = row.get("exit_code")
    executed = executed or type(exit_code) is int
    current_key = _binding_key(_binding(runtime))
    bindings = [row.get("binding"), row.get("after_binding")]
    binding_rows = [binding for binding in bindings if isinstance(binding, Mapping)]
    binding_current = all(_binding_key(binding) == current_key for binding in binding_rows)
    binding_stable = row.get("binding_stable") is not False
    host_test_completion = (
        row.get("kind") == "test_subprocess_completed"
        and row.get("producer") == "test_subprocess"
    )
    if (not status and host_test_completion and exit_code == 0
            and type(exit_code) is int and row.get("binding_stable") is True
            and len(binding_rows) == 2 and binding_current
            and _binding_key(binding_rows[0]) == _binding_key(binding_rows[1])):
        status = "PASS"
    return (status in _PASS and executed and binding_current and binding_stable
            and (type(exit_code) is not int or exit_code == 0))


def _seed_receipts(runtime: object) -> dict[str, object]:
    state = _state(runtime)
    ids = state.setdefault("seen_receipt_ids", [])
    keys = state.setdefault("verified_keys", [])
    if not isinstance(ids, list) or not isinstance(keys, list):
        raise TypeError("post-green state was corrupted")
    for row in _receipt_rows(runtime):
        identity = receipt_identity(row)
        if identity not in ids:
            ids.append(identity)
        if _receipt_successful(runtime, row):
            key = verification_key(runtime, receipt=row)
            if key not in keys:
                keys.append(key)
    return state


def _preview_state(runtime: object) -> dict[str, object]:
    """Build a read-only decision state without seeding or settling runtime state."""
    existing = getattr(runtime, STATE_ATTR, None)
    state = copy.deepcopy(existing) if isinstance(existing, Mapping) else {
        "seen_receipt_ids": [], "verified_keys": [], "requested_keys": [],
        "refusals": {}, "settled": None, "last_decision": None,
    }
    ids = state.setdefault("seen_receipt_ids", [])
    keys = state.setdefault("verified_keys", [])
    for row in _receipt_rows(runtime):
        identity = receipt_identity(row)
        if identity not in ids:
            ids.append(identity)
        if _receipt_successful(runtime, row):
            key = verification_key(runtime, receipt=row)
            if key not in keys:
                keys.append(key)
    return state


def _preview_next_action(state: Mapping[str, object]) -> str:
    settled = state.get("settled")
    if isinstance(settled, Mapping) and settled.get(SETTLEMENT_KEY) == SETTLEMENT_READY:
        return "done"
    return "run_tests"


def _open_ids(runtime: object) -> tuple[str, ...]:
    values = (getattr(runtime, "host_closure_assessments", ()),
              getattr(runtime, "obligation_assessments", ()))
    revision = _binding_key(_binding(runtime)) or "UNKNOWN"
    token = (revision, _digest(_receipt_source_tokens(
        runtime, (("host_closure", values[0]), ("obligations", values[1])))))
    cached = getattr(runtime, "_postgreen_open_ids_cache", None)
    if isinstance(cached, Mapping) and cached.get("token") == token:
        return tuple(cached.get("ids", ()))
    ids: list[str] = []
    for value in values:
        for row in _rows(value):
            ident = next((_text(row.get(key)) for key in
                          ("obligation_id", "clause_id", "requirement_id", "check_id", "id")
                          if _text(row.get(key))), "")
            status = _text(row.get("verdict") or row.get("status") or row.get("state")).upper()
            if ident and status not in _CLOSED and ident not in ids:
                ids.append(ident)
    result = tuple(ids)
    setattr(runtime, "_postgreen_open_ids_cache", {"token": token, "ids": result})
    return result


def _progress_epoch(runtime: object) -> str:
    """Bind post-green progress to every host-owned invalidation dimension."""
    values = {
        "binding": _binding_key(_binding(runtime)),
        "receipts": _receipt_source_tokens(runtime, tuple(
            (name, getattr(runtime, name, ())) for name in (
                "test_receipts", "verification_receipts", "probe_receipts", "receipts"))),
        "requirements": tuple(getattr(runtime, name, None) for name in (
            "host_closure_assessments", "obligation_assessments")),
        "candidate": getattr(runtime, "candidate", getattr(runtime, "candidate_state", None)),
        "evidence": getattr(runtime, "evidence", getattr(runtime, "evidence_state", None)),
        "effects": getattr(runtime, "effects", getattr(runtime, "effect_state", None)),
        "policy": getattr(runtime, "policy", getattr(runtime, "policy_state", None)),
        "invalidation": getattr(runtime, "_invalidation_gate", None),
        "terminal_seal": getattr(getattr(runtime, "green_lock", None), "terminal_seal", None),
    }
    return _digest(_semantic(values))


def _completion_gate(runtime: object) -> Mapping[str, object]:
    revision = _binding_key(_binding(runtime)) or "UNKNOWN"
    inputs_digest = _digest(_receipt_source_tokens(
        runtime, (("host_closure", getattr(runtime, "host_closure_assessments", ())),
                  ("obligations", getattr(runtime, "obligation_assessments", ())))))
    token = (
        revision,
        inputs_digest,
        _digest(_semantic(getattr(runtime, "_invalidation_gate", None))),
    )
    cached = getattr(runtime, "_postgreen_completion_gate_cache", None)
    if isinstance(cached, Mapping) and cached.get("token") == token:
        result = cached.get("result")
        if isinstance(result, Mapping):
            return result
    try:
        from kernel_invalidation import runtime_completion_gate
        result = runtime_completion_gate(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        result = {
            "admitted": False,
            "reason_code": "DEPGRAPH_ERROR",
            "finding": f"dependency graph unavailable: {type(error).__name__}",
            "next_action": "host_stop",
        }
    if isinstance(result, Mapping):
        result = dict(result)
    else:
        result = {
            "admitted": False,
            "reason_code": "DEPGRAPH_ERROR",
            "finding": "dependency graph returned no decision",
            "next_action": "host_stop",
        }
    setattr(runtime, "_postgreen_completion_gate_cache",
            {"token": token, "result": copy.deepcopy(result)})
    return result


def _clean(runtime: object) -> tuple[bool, str, tuple[str, ...]]:
    # S47 manifests are stricter than the legacy clean-evidence projection:
    # empty/clean bookkeeping cannot certify a missing ticket scenario.
    manifest = getattr(runtime, "_s47_verification_manifest", None)
    if isinstance(manifest, Mapping):
        try:
            from kernel_s47_causal_fix import verify_manifest
            context = getattr(runtime, "_s47_verification_context", {})
            context = context if isinstance(context, Mapping) else {}
            decision = verify_manifest(manifest, context)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            return False, f"S47 verification unavailable: {type(error).__name__}", ()
        if decision.get("status") != "READY_FOR_S7":
            blockers = tuple(str(item) for item in decision.get("blockers", ()) if str(item))
            return False, "; ".join(blockers) or "S47 verification is incomplete", blockers
    finding = _completion_gate(runtime)
    open_ids = _open_ids(runtime)
    if finding.get("admitted") is not True:
        reason = _text(finding.get("finding") or finding.get("reason_code"))
        return False, reason or "dependency graph did not certify current evidence", open_ids
    if open_ids:
        return False, "open requirements: " + ", ".join(open_ids), open_ids
    return True, "all requirements clean on the current revision", ()


def _completion_evaluation(runtime: object, arguments: Mapping[str, object]) -> CompletionEvaluation:
    key = verification_key(runtime, arguments)
    clean, reason, open_ids = _clean(runtime)
    return CompletionEvaluation(key, _progress_epoch(runtime), clean, reason, tuple(open_ids))


SETTLEMENT_KEY = "settlement"
SETTLEMENT_READY = "READY_FOR_COMMON_TERMINAL"
SETTLEMENT_UNCERTIFIED = "UNCERTIFIED"


def _settle(runtime: object, *, certified: bool, reason: str,
            open_ids: Sequence[str] = ()) -> Mapping[str, object]:
    state = _state(runtime)
    progress_epoch = _progress_epoch(runtime)
    existing = state.get("settled")
    if isinstance(existing, Mapping):
        if existing.get("progress_epoch") == progress_epoch:
            return existing
        # Candidate/evidence/requirement/effect/policy or binding changes make
        # prior progress stale. It is never silently reused as a new seal.
        state["settled"] = None
        setattr(runtime, "_postgreen_settlement", None)
    s47_finish = getattr(runtime, "_s47_finish_proposal", None)
    s47_decision = getattr(runtime, "_s47_verification_decision", None)
    s47_artifact = getattr(runtime, "_s47_artifact_receipt", None)
    s47_delivery: str | None = None
    if isinstance(s47_finish, Mapping) and isinstance(s47_decision, Mapping):
        try:
            from kernel_s47_causal_fix import settle_terminal
            terminal_state = getattr(runtime, "_s47_terminal_state", None)
            if not isinstance(terminal_state, dict):
                terminal_state = {}
                setattr(runtime, "_s47_terminal_state", terminal_state)
            terminal = settle_terminal(
                s47_finish, s47_decision,
                artifact=s47_artifact if isinstance(s47_artifact, Mapping) else None,
                state=terminal_state,
            )
            setattr(runtime, "_s47_terminal_disposition", copy.deepcopy(terminal))
            certified = terminal.get("certified") is True
            if terminal.get("delivery") in {"STORED", "NOT_STORED"}:
                s47_delivery = str(terminal["delivery"])
            reason = "; ".join(str(item) for item in terminal.get("reason_codes", ()) if str(item)) or reason
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            certified = False
            reason = "S47 terminal settlement unavailable"
    # This is a progress settlement, not a terminal decision.  In particular,
    # a clean post-green observation may inform the common completion owner but
    # cannot mint DoneFacts, accepted state, or a terminal disposition here.
    outcome = SETTLEMENT_READY if certified else SETTLEMENT_UNCERTIFIED
    result: dict[str, object] = {
        "schema": SCHEMA,
        SETTLEMENT_KEY: outcome,
        "authoritative": False,
        "completion_ready": certified,
        "delivery": s47_delivery,
        "open_ids": list(dict.fromkeys(str(item) for item in open_ids if str(item))),
        "completion_owner_reason": reason,
        "blocking_reason": reason,
        "progress_epoch": progress_epoch,
    }
    state["settled"] = copy.deepcopy(result)
    setattr(runtime, "_postgreen_settlement", copy.deepcopy(result))
    _emit(runtime, "postgreen_host_settled", outcome=outcome, certified=False,
           completion_ready=certified, authoritative=False,
           reason=reason, open_ids=list(open_ids))
    return result


def terminal_projection(runtime: object, result: Mapping[str, object]) -> dict[str, object] | None:
    settled = getattr(runtime, "_postgreen_settlement", None)
    if not isinstance(settled, Mapping):
        return None
    projected = dict(result)
    if settled.get("progress_epoch") != _progress_epoch(runtime):
        projected["postgreen_settlement"] = {
            "schema": SCHEMA,
            "settlement": "STALE",
            "authoritative": False,
            "stale": True,
            "progress_epoch": settled.get("progress_epoch"),
        }
        projected["postgreen_settlement_stale"] = True
        return projected
    # Keep settlement visible for diagnostics while leaving the common terminal
    # owner as the sole writer of acceptance/certification/disposition fields.
    projected["postgreen_settlement"] = copy.deepcopy(dict(settled))
    return projected


def _refusal(runtime: object, action: str, key: str, *, reason: str,
             next_action: str, count: int,
             evaluation: CompletionEvaluation | None = None) -> Decision:
    state = _state(runtime)
    if count >= 2:
        if evaluation is not None:
            clean, block, open_ids = evaluation.clean, evaluation.reason, evaluation.open_ids
        else:
            clean, block, open_ids = _clean(runtime)
        settled = _settle(runtime, certified=clean, reason=block if not clean else reason,
                          open_ids=open_ids)
        # One spelling (rule #223): _settle publishes "settlement"
        # (READY_FOR_COMMON_TERMINAL | UNCERTIFIED); there is no "outcome" key.
        outcome = str(settled.get(SETTLEMENT_KEY) or SETTLEMENT_UNCERTIFIED)
        next_action = "done" if clean else "host_stop"
        message = (f"{REFUSAL_CODE}: identical verification refused after {count} attempts; "
                   f"host settled {outcome}: {block if not clean else 'all requirements clean'}. "
                   f"Next action: {next_action} (funded).")
        rule = RULE_CERTIFIED if clean else RULE_UNCERTIFIED
    else:
        message = (f"{REFUSAL_CODE}: verification input is unchanged and adds no host evidence "
                   f"(key={key[:18]}). Next action: {next_action} (funded); "
                   "provide a new selector or re-check the dirtied item.")
        rule = RULE_IDENTICAL
    decision = Decision(False, rule, message, next_action, True,
                        {"verification_key": key, "refusal_count": count,
                         "reason": reason})
    state["last_decision"] = {
        "rule": decision.rule, "message": decision.message,
        "next_action": decision.next_action, "funded": decision.funded,
    }
    _emit(runtime, "postgreen_typed_refusal", code=REFUSAL_CODE, action=action,
           rule=decision.rule, verification_key=key, refusal_count=count,
           next_action=next_action, funded=True)
    return decision


def _dirty_refs(runtime: object) -> set[str]:
    """Read dirty identifiers from the host dependency graph, never from prose."""
    refs: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                add(item)
            return
        text = _text(value)
        if text:
            refs.add(text.replace("\\", "/"))

    gate = getattr(runtime, "_invalidation_gate", None)
    if isinstance(gate, Mapping):
        for field in ("dirty_node_ids", "evidence_refs", "dirty_paths"):
            add(gate.get(field))
    index = getattr(runtime, "_invalidation_index", None)
    for node in getattr(index, "dirty_nodes", ()):
        add(getattr(node, "node_id", ""))
        add(getattr(node, "path", ""))
        add(getattr(node, "evidence_refs", ()))
    return refs


def _host_checked_dirty_recheck(runtime: object, receipt: Mapping[str, object]) -> bool:
    """Accept dirty-item novelty only when the host graph names that dirty item."""
    candidates: set[str] = set()
    for field in ("rechecked_item", "dirtied_item", "dirty_node_id", "target_path", "path"):
        value = receipt.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidates.update(_text(item).replace("\\", "/") for item in value if _text(item))
        elif _text(value):
            candidates.add(_text(value).replace("\\", "/"))
    if not candidates:
        return False
    dirty = _dirty_refs(runtime)
    if not dirty:
        try:
            from kernel_invalidation import runtime_completion_gate
            runtime_completion_gate(runtime)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        dirty = _dirty_refs(runtime)
    return bool(candidates & dirty)


def decide(lock: object, action: object, arguments: Mapping[str, object] | None = None,
           *, for_menu: bool = False,
           evaluation: CompletionEvaluation | None = None) -> Decision | None:
    """Decide whether a post-green verification is capable of making progress."""
    if not _active(lock):
        return None
    runtime = getattr(lock, "runtime")
    name = _text(action).lower()
    if name in {"done", "submit"}:
        # Completion ownership remains with RF38/CompletionOwner.  This
        # protocol records an admitted terminal action from dispatch, but must
        # not replace the pending-edit refusal used to reach that action.
        return None
    if name not in {"run_tests", "test", "pytest", "probe", "predict"}:
        return None
    args = dict(arguments or {})
    evaluation = evaluation or _completion_evaluation(runtime, args)
    if evaluation.progress_epoch != _progress_epoch(runtime):
        return Decision(
            False, RULE_UNCERTIFIED,
            "host completion evaluation became stale before consumption",
            "run_tests", True,
            {"verification_key": evaluation.verification_key,
             "evaluation_epoch": evaluation.progress_epoch,
             "current_epoch": _progress_epoch(runtime),
             "stale": True},
        )
    state = _preview_state(runtime) if for_menu else _seed_receipts(runtime)
    key = evaluation.verification_key
    verified = key in state["verified_keys"]
    requested = key in state["requested_keys"]
    if verified or requested:
        try:
            from kernel_post_green_menu import pending_verification
            # Menu/dispatch admission follows the armed CH-04 evidence contract.
            # QUIET suppresses the repair surface after a host-verified run, but
            # that run must not remove the explicit reentry action while pending
            # evidence is still armed.
            verification_pending = pending_verification(runtime)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            verification_pending = True
        if verification_pending:
            # A prior host green can share this verification key without
            # satisfying the post-edit receipt contract. Keep one explicit
            # runner action available until that receipt clears the pending edit.
            if not for_menu and key not in state["requested_keys"]:
                state["requested_keys"].append(key)
            decision = Decision(
                True, RULE_NEW_EVIDENCE,
                "a post-green edit still requires an explicit current-revision test receipt",
                name, True, {"verification_key": key,
                             "progress_epoch": evaluation.progress_epoch,
                             "postgreen_verification_pending": True},
            )
            state["last_decision"] = {
                "rule": decision.rule, "message": decision.message,
                "next_action": name, "funded": True,
            }
            return decision
        if for_menu:
            # Rendering a menu is observational.  Seeded duplicate keys must
            # not consume the refusal budget or settle the attempt.
            return Decision(
                False,
                RULE_IDENTICAL,
                "preview: identical verification is refused without consuming host progress",
                _preview_next_action(state),
                True,
                {"verification_key": key, "preview": True,
                 "progress_epoch": evaluation.progress_epoch},
            )
        refusals = state["refusals"]
        # Action aliases share one host identity.  Otherwise a model could
        # rotate run_tests/test/probe and buy one refusal budget per alias.
        fingerprint = key
        count = int(refusals.get(fingerprint, 0) or 0) + 1
        refusals[fingerprint] = count
        clean, reason = evaluation.clean, evaluation.reason
        next_action = "done" if clean else "run_tests"
        return _refusal(runtime, name, key, reason=reason,
                        next_action=next_action, count=count, evaluation=evaluation)
    if key not in state["requested_keys"]:
        if not for_menu:
            state["requested_keys"].append(key)
    decision = Decision(True, RULE_NEW_EVIDENCE,
                        "host requires a new verification receipt or dirty-item re-check",
                        name, True, {"verification_key": key,
                                     "progress_epoch": evaluation.progress_epoch})
    state["last_decision"] = {"rule": decision.rule, "message": decision.message,
                              "next_action": name, "funded": True}
    return decision


def preview_admission(runtime: object, action: object,
                      arguments: Mapping[str, object] | None = None) -> Decision | None:
    """Pure menu projection; it never consumes a request/refusal budget."""
    lock = getattr(runtime, "green_lock", None)
    if lock is None:
        return None
    args = dict(arguments or {})
    return decide(lock, action, args, for_menu=True,
                  evaluation=_completion_evaluation(runtime, args))


def consume_admission(runtime: object, action: object,
                      arguments: Mapping[str, object] | None = None,
                      *, action_id: str | None = None) -> Decision | None:
    """Consume one dispatched admission exactly once per host action identity."""
    lock = getattr(runtime, "green_lock", None)
    if lock is None:
        return None
    args = dict(arguments or {})
    token = _digest({"action_id": action_id, "action": _text(action), "arguments": args})
    consumed = getattr(runtime, "_postgreen_consumed_admissions", None)
    if not isinstance(consumed, dict):
        consumed = {}
        setattr(runtime, "_postgreen_consumed_admissions", consumed)
    previous = consumed.get(token)
    if isinstance(previous, Decision):
        return previous
    result = decide(lock, action, args, for_menu=False,
                    evaluation=_completion_evaluation(runtime, args))
    if isinstance(result, Decision):
        consumed[token] = result
    return result


def observe_verification(runtime: object, receipt: Mapping[str, object] | None) -> bool:
    """Record attempted evidence; only executed/current/successful checks verify."""
    if not isinstance(receipt, Mapping):
        return False
    state = _state(runtime)
    identity = receipt_identity(receipt)
    key = verification_key(runtime, receipt=receipt)
    ids = state["seen_receipt_ids"]
    attempted = state.setdefault("attempted_receipt_ids", [])
    keys = state["verified_keys"]
    novel_receipt = identity not in ids
    novel_dirty = _host_checked_dirty_recheck(runtime, receipt)
    if novel_receipt:
        ids.append(identity)
        attempted.append(identity)
    state["requested_keys"] = [item for item in state["requested_keys"] if item != key]
    status = _text(receipt.get("status") or receipt.get("result_class")
                   or receipt.get("outcome") or receipt.get("verdict")).upper()
    result = receipt.get("result")
    if isinstance(result, Mapping):
        status = _text(result.get("status") or result.get("result_class") or status).upper()
    executed = any(receipt.get(field) is True for field in (
        "executed", "execution_complete", "executor_returned", "current"))
    executed = executed or type(receipt.get("exit_code")) is int
    binding = receipt.get("binding", receipt.get("after_binding"))
    binding_current = (not isinstance(binding, Mapping)
                       or _binding_key(binding) == _binding_key(_binding(runtime)))
    successful = _receipt_successful(runtime, receipt)
    if not successful:
        _emit(runtime, "postgreen_non_success_evidence", receipt_id=identity,
              verification_key=key, status=status or "UNKNOWN",
              executed=executed, binding_current=binding_current,
              novel_receipt=novel_receipt)
        return False
    if (novel_receipt or novel_dirty) and (key not in keys or novel_dirty):
        if key not in keys:
            keys.append(key)
        _emit(runtime, "postgreen_new_evidence", receipt_id=identity,
              verification_key=key, novel_receipt=novel_receipt,
              dirtied_item_rechecked=novel_dirty)
        return True
    _emit(runtime, "postgreen_duplicate_evidence", receipt_id=identity,
          verification_key=key, novel_receipt=novel_receipt,
          dirtied_item_rechecked=novel_dirty)
    return False


def note_terminal_action(runtime: object, action: str) -> None:
    if _text(action).lower() not in {"done", "submit"}:
        return
    from kernel_green_lock import (
        current_public_verdict_is_green,
        note_unchanged_parent_green_suppressed,
        product_revision_change_from_parent,
    )
    if not current_public_verdict_is_green(runtime):
        return
    changed, _revision, _comparison = product_revision_change_from_parent(runtime)
    from kernel_green_lock import PROVEN_PARENT_UNCHANGED as _PROVEN_UNCHANGED
    changed = changed or _comparison not in _PROVEN_UNCHANGED  # unknown parent identity is not "unchanged"
    if not changed:
        note_unchanged_parent_green_suppressed(
            runtime, reason="postgreen_terminal_action")
        return
    _state(runtime)["terminal_action"] = _text(action).lower()
    _emit(runtime, "postgreen_terminal_action_admitted", action=_text(action).lower())


def last_message(runtime: object) -> str:
    value = _state(runtime).get("last_decision")
    return _text(value.get("message")) if isinstance(value, Mapping) else ""


def replay_steps(runtime: object, *, selector: str, limit: int = 8) -> int:
    """Replay a repeated verification request until the host settles."""
    lock = getattr(runtime, "green_lock")
    steps = 0
    while steps < limit and not isinstance(_state(runtime).get("settled"), Mapping):
        steps += 1
        decide(lock, "run_tests", {"selector": selector})
    return steps


__all__ = [
    "CompletionEvaluation", "Decision", "REFUSAL_CODE", "RULE_CERTIFIED", "RULE_IDENTICAL", "RULE_NEW_EVIDENCE",
    "RULE_TERMINAL", "RULE_UNCERTIFIED", "SCHEMA", "consume_admission", "decide",
    "last_message", "preview_admission",
    "note_terminal_action", "observe_verification", "receipt_identity", "replay_steps",
    "terminal_projection", "verification_key",
]
