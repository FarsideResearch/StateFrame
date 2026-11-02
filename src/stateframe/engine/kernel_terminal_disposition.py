"""One host-owned terminal disposition for every completion projection.

The completion owner creates the terminal result, but this receipt owns its
meaning.  Stop taxonomy, final-result adapters, and summaries must consume the
same receipt instead of inferring acceptance from a status string or exit code.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-terminal-disposition/1"
STORED = "STORED"
NOT_STORED = "NOT_STORED"
VERIFIED_FIXED = "VERIFIED_FIXED"
UNRESOLVED = "UNRESOLVED"
ENVIRONMENT_BLOCKED = "ENVIRONMENT_BLOCKED"
CERTIFIED = "CERTIFIED"
UNCERTIFIED = "UNCERTIFIED"

_UNCERTIFIED_RESULTS = frozenset({
    "UNCERTIFIED_SUBMIT",
    "INCOMPLETE",
    "STOPPED_INCOMPLETE",
    "BLOCKED",
    "NEEDS_WORK",
    "HOOK_BLOCKED",
    "HOOK_CONTAINMENT",
    "HARNESS_ERROR",
    "ENVIRONMENT_BROKEN",
})
_ENVIRONMENT_RESULTS = frozenset({
    "HARNESS_ERROR",
    "HARNESS_SUSPECT",
    "ENVIRONMENT_BROKEN",
    "REPRO_ENVIRONMENT",
})


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _binding(runtime: object, result: Mapping[str, object]) -> str:
    explicit = result.get("artifact_digest")
    if _text(explicit):
        return _text(explicit)
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            for key in ("product_revision", "artifact", "revision"):
                if _text(value.get(key)):
                    return _text(value[key])
    for key in ("candidate_revision", "current_revision", "revision"):
        if _text(result.get(key)):
            return _text(result[key])
    return "unknown"


def _stage_epoch(runtime: object, result: Mapping[str, object]) -> int | str:
    value = result.get("stage_epoch")
    if type(value) is int and value >= 1:
        return value
    owner_fn = getattr(runtime, "_idealpath_owner", None)
    if callable(owner_fn):
        try:
            owner = owner_fn()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            owner = None
        epoch = getattr(owner, "epoch", None)
        if type(epoch) is int and epoch >= 1:
            return epoch
    return "unknown"


def _open_ids(result: Mapping[str, object]) -> list[str]:
    values = result.get("open_requirement_ids", result.get("open_ids", ()))
    return list(dict.fromkeys(
        _text(item) for item in _sequence(values) if _text(item)
    ))


def _trusted_refs(result: Mapping[str, object], facts: Mapping[str, object] | None) -> list[str]:
    values: list[str] = []
    for item in _sequence(result.get("trusted_receipt_refs", ())):
        if _text(item):
            values.append(_text(item))
    if facts is not None:
        check = facts.get("last_check_result")
        if isinstance(check, Mapping) and _text(check.get("receipt_id")):
            values.append(_text(check["receipt_id"]))
    return list(dict.fromkeys(values))


def _done_facts(result: Mapping[str, object]) -> Mapping[str, object] | None:
    value = result.get("done_facts")
    return value if isinstance(value, Mapping) else None


def _last_stage(runtime: object, result: Mapping[str, object]) -> str:
    explicit = _text(result.get("last_stage", result.get("stage")))
    if explicit:
        return explicit
    owner_fn = getattr(runtime, "_idealpath_owner", None)
    if callable(owner_fn):
        try:
            stage = _text(getattr(owner_fn(), "stage", ""))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            stage = ""
        if stage:
            return stage
    return "UNKNOWN"


def _stop_reason(result: Mapping[str, object], terminal: str, *, accepted: bool) -> str:
    done_reason = result.get("done_reason")
    if (isinstance(done_reason, Mapping)
            and done_reason.get("decision") == "ACCEPTED_UNCERTIFIED"):
        return "UNCERTIFIED_SUBMIT"
    value = _text(result.get("stop_reason", result.get("system_outcome"))).upper()
    aliases = {
        "DONE_ACCEPTED": "DONE",
        "SUBMITTED": "DONE",
        "DONE": "DONE",
        "STALL": "STALL",
        "BUDGET": "BUDGET_CAP",
        "BUDGET_CAP": "BUDGET_CAP",
        "SCHEMA": "SCHEMA_FAILURE",
        "SCHEMA_FAILURE": "SCHEMA_FAILURE",
        "INTERRUPTED": "INTERRUPTED_CHECK",
        "INTERRUPTED_CHECK": "INTERRUPTED_CHECK",
        "TIMEOUT": "TIMEOUT",
        "RESOURCE": "RESOURCE_EXHAUSTION",
        "RESOURCE_EXHAUSTION": "RESOURCE_EXHAUSTION",
        "FINALIZATION_ERROR": "FINALIZATION_ERROR",
        # r2EP b1378: the legacy v39 uncertified-submit result carries this stop
        # reason with D2 off; without the alias it fell through to FINALIZATION_ERROR and the
        # settlement treated a real (uncertified) submission as an operational error.
        "UNCERTIFIED_SUBMIT": "UNCERTIFIED_SUBMIT",
        "HOOK_BLOCKED": "HOOK_BLOCKED",
        "HOOK_CONTAINMENT": "HOOK_CONTAINMENT",
    }
    if value in aliases:
        return aliases[value]
    if accepted:
        return "DONE"
    if terminal in _ENVIRONMENT_RESULTS:
        return "RESOURCE_EXHAUSTION"
    return "FINALIZATION_ERROR"


def _facts_accept(facts: Mapping[str, object] | None) -> bool | None:
    if facts is None or "accepted" not in facts:
        return None
    return facts.get("accepted") is True


def _facts_current(facts: Mapping[str, object] | None) -> bool | None:
    if facts is None:
        return None
    check = facts.get("last_check_result")
    if isinstance(check, Mapping) and "current" in check:
        return check.get("current") is True
    return None


def _uncertified_candidate_present(runtime: object, result: Mapping[str, object],
                                   facts: Mapping[str, object] | None) -> bool:
    """A candidate exists when the artifact is bound and the product actually changed."""
    if _binding(runtime, result) in {"", "unknown"}:
        return False
    changed = (facts or {}).get("changed_file_paths", (facts or {}).get("changed_files", ()))
    return bool([item for item in _sequence(changed) if _text(item)])


def _sealed_receipt(runtime: object, sealed: Mapping[str, object]) -> dict[str, object]:
    """Project the immutable artifact seal into the shared terminal receipt schema."""
    seal = sealed.get("done_seal")
    seal = seal if isinstance(seal, Mapping) else {}
    labels = sealed.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    open_ids = labels.get("open_item_ids", ())
    if not isinstance(open_ids, Sequence) or isinstance(open_ids, (str, bytes, bytearray)):
        open_ids = ()
    certification = "CERTIFIED" if sealed.get("certified") is True else UNCERTIFIED
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_digest": str(seal.get("artifact_digest") or ""),
        "stage_epoch": _stage_epoch(runtime, sealed),
        "last_stage": "DONE_SUBMISSION",
        "stop_reason": "SUBMITTED",
        "done_facts_ref": None,
        "delivery": STORED,
        "artifact": STORED,
        "repair": UNRESOLVED,
        "verification": "UNVERIFIED",
        "acceptance": "ACCEPTED",
        "certification": certification,
        "accepted": True,
        "open_requirement_ids": [str(item) for item in open_ids if str(item)],
        "trusted_receipt_refs": [],
        "evidence_handles": [],
        "unresolved_ids": [str(item) for item in open_ids if str(item)],
        "reason": "done_submission",
        "done_seal": dict(seal),
    }
    receipt["receipt_digest"] = _digest(receipt)
    return receipt


def mint(runtime: object, result: Mapping[str, object]) -> dict[str, object]:
    """Mint one deterministic receipt from the terminal result and shared host facts."""
    from kernel_done_is_submit_authority import terminal_from_seal
    from kernel_terminal_lattice import join_terminal

    sealed = terminal_from_seal(runtime)
    if sealed is not None:
        return join_terminal(
            runtime, _sealed_receipt(runtime, sealed),
            writer="kernel_terminal_disposition.mint",
        )
    try:
        from kernel_done_readiness_authority import apply_terminal_result, cached_decision
        shared_done_decision = cached_decision(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        shared_done_decision = None
    if shared_done_decision is not None:
        result = apply_terminal_result(runtime, result, shared_done_decision)
    terminal = str(result.get("result", result.get("outcome", ""))).upper()
    facts = _done_facts(result)
    facts_accept = _facts_accept(facts)
    facts_current = _facts_current(facts)
    unresolved = tuple(
        _text(item) for item in _sequence(
            (facts or {}).get("unresolved_gating_items", result.get("unresolved_gating_items", ()))
        ) if _text(item)
    )
    pass_gates = None
    vector_fn = getattr(runtime, "pass_gates_vector", None)
    if callable(vector_fn):
        try:
            vector = vector_fn()
            pass_gates = vector.as_dict() if callable(getattr(vector, "as_dict", None)) else None
            if getattr(vector, "ready", False) is not True:
                unresolved = tuple(dict.fromkeys((
                    *unresolved,
                    *(str(getattr(item, "blocker_id", ""))
                      for item in getattr(vector, "blockers", ())
                      if str(getattr(item, "blocker_id", ""))),
                )))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            unresolved = tuple(dict.fromkeys((*unresolved, "PASS_GATES_VECTOR_UNAVAILABLE")))
    explicitly_accepted = result.get("accepted") is True
    current_ok = facts_current is not False
    facts_ok = facts_accept is not False
    verified = (
        explicitly_accepted
        and current_ok
        and facts_ok
        and not unresolved
        and terminal not in _UNCERTIFIED_RESULTS
    )
    if verified:
        repair = VERIFIED_FIXED
        certification = CERTIFIED if result.get("certified") is True else UNCERTIFIED
        reason = "verified_fixed"
    else:
        repair = ENVIRONMENT_BLOCKED if terminal in _ENVIRONMENT_RESULTS else UNRESOLVED
        certification = UNCERTIFIED
        reason = _text(result.get("completion_owner_reason")) or terminal.lower() or "terminal_unresolved"
    trusted_refs = _trusted_refs(result, facts)
    stop_reason = _stop_reason(result, terminal, accepted=verified)
    last_stage = _last_stage(runtime, result)
    requested_delivery = result.get("delivery")
    if (requested_delivery is None and stop_reason == "UNCERTIFIED_SUBMIT"
            and _uncertified_candidate_present(runtime, result, facts)):
        # r2EP b1378: an uncertified submit IS a submission of the current scoped
        # candidate. Record it as stored so settlement exports and grades it (certification stays
        # UNCERTIFIED and acceptance UNACCEPTED; the open obligations stay attached).
        requested_delivery = STORED
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_digest": _binding(runtime, result),
        "stage_epoch": _stage_epoch(runtime, result),
        "last_stage": last_stage,
        "stop_reason": stop_reason,
        "done_facts_ref": (
            _text(facts.get("evidence_digest")) if facts is not None else None
        ),
        "delivery": requested_delivery if requested_delivery in {STORED, NOT_STORED} else NOT_STORED,
        "artifact": requested_delivery if requested_delivery in {STORED, NOT_STORED} else NOT_STORED,
        "repair": repair,
        "verification": "VERIFIED" if verified else "UNVERIFIED",
        "acceptance": "ACCEPTED" if verified else "UNACCEPTED",
        "certification": certification,
        "accepted": "ACCEPTED" == ("ACCEPTED" if verified else "UNACCEPTED"),
        "open_requirement_ids": _open_ids(result),
        "trusted_receipt_refs": trusted_refs,
        "evidence_handles": trusted_refs,
        "unresolved_ids": list(unresolved),
        "reason": reason,
    }
    done_reason = result.get("done_reason")
    if shared_done_decision is not None:
        shared_reason = shared_done_decision.to_dict()
        if done_reason is not None and done_reason != shared_reason:
            raise ValueError("terminal receipt diverged from the shared done readiness decision")
        done_reason = shared_reason
    if isinstance(done_reason, Mapping):
        receipt["done_reason"] = dict(done_reason)
        receipt["done_reason_line"] = str(result.get("done_reason_line", ""))[:320]
        receipt["stop_classification"] = str(
            result.get("stop_classification", stop_reason))[:80]
    if pass_gates is not None:
        receipt["pass_gates"] = pass_gates
    receipt["receipt_digest"] = _digest(receipt)
    return join_terminal(
        runtime, receipt, writer="kernel_terminal_disposition.mint",
    )


def project(
    result: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    runtime: object = None,
) -> dict[str, object]:
    """Project the joined receipt onto a result, forcing rejected states non-accepted."""
    from kernel_terminal_lattice import join_terminal

    receipt = join_terminal(
        runtime, receipt, writer="kernel_terminal_disposition.project",
    )
    projected = dict(result)
    projected["terminal_disposition"] = dict(receipt)
    certified = receipt.get("certification") == CERTIFIED
    accepted = receipt.get("acceptance") == "ACCEPTED"
    projected["artifact"] = receipt.get("artifact", receipt.get("delivery", NOT_STORED))
    projected["verification"] = receipt.get("verification", "VERIFIED" if certified else "UNVERIFIED")
    projected["acceptance"] = receipt.get("acceptance", "ACCEPTED" if certified else "UNACCEPTED")
    legacy_accepted = projected.get("accepted") is True
    projected["accepted"] = accepted
    projected["certified"] = certified
    projected["uncertified"] = not certified
    if legacy_accepted and not accepted:
        projected["terminal_contract_conflict"] = True
        projected["completion_owner_reason"] = "terminal_contract_conflict"
    return projected


def receipt_from(value: object) -> Mapping[str, object] | None:
    """Return a structurally valid projected receipt, if one is present."""
    if not isinstance(value, Mapping):
        return None
    receipt = value.get("terminal_disposition")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA:
        return None
    if receipt.get("certification") not in {CERTIFIED, UNCERTIFIED}:
        return None
    if receipt.get("repair") not in {VERIFIED_FIXED, UNRESOLVED, ENVIRONMENT_BLOCKED}:
        return None
    if receipt.get("delivery") not in {STORED, NOT_STORED}:
        return None
    if receipt.get("artifact", receipt.get("delivery")) not in {STORED, NOT_STORED}:
        return None
    if receipt.get("verification") not in {"VERIFIED", "UNVERIFIED"}:
        return None
    if receipt.get("acceptance") not in {"ACCEPTED", "UNACCEPTED"}:
        return None
    if receipt.get("accepted") is not (receipt.get("acceptance") == "ACCEPTED"):
        return None
    receipt_digest = receipt.get("receipt_digest")
    if not isinstance(receipt_digest, str) or not receipt_digest:
        return None
    unsigned = {
        key: item for key, item in receipt.items() if key != "receipt_digest"
    }
    if _digest(unsigned) != receipt_digest:
        return None
    return receipt


__all__ = [
    "CERTIFIED", "ENVIRONMENT_BLOCKED", "NOT_STORED", "SCHEMA", "STORED",
    "UNCERTIFIED", "UNRESOLVED", "VERIFIED_FIXED", "mint", "project", "receipt_from",
]
