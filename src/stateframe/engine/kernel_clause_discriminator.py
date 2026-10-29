"""Host-owned executable clause coverage for the V46 witness rail.

The model may choose one public case, but it never chooses what counts as a
passing requirement.  This module compiles the public ticket into tracked
clauses, validates a small typed proposal against that ticket, executes it
through ``kernel_behavior_witness``, and records a host-computed evidence pin.
It is intentionally task-neutral: no fixture names, private paths, or task
specific literals are embedded here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA = "stateframe-clause-discriminator/1"
EVIDENCE_PIN_SCHEMA = "stateframe-evidence-pin/1"
TOKEN_FRACTION = 0.08
MAX_CASES_PER_REQUEST = 1
# One provider turn is enough to exercise a clause.  If that turn cannot be
# funded or admitted, the completion owner must disclose the limitation rather
# than repeatedly reopening the same witness invitation.
MAX_WITNESS_TURNS_PER_CLAUSE = 1
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_PASS = frozenset({"PASS", "PASSED", "SUCCESS", "OK", "COMPLETED"})
_EXECUTED = frozenset({"EXECUTED", "RUN", "RAN", "COMPLETED"})
_RELATION_OPERATORS = frozenset({"equals"})


class ClauseDiscriminatorError(ValueError):
    """Raised for a proposal that is not grounded in the public contract."""


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _witness_turns_used(runtime: object, clause_ref: str) -> int:
    value = getattr(runtime, "_clause_discriminator_witness_turns", {})
    if not isinstance(value, Mapping):
        return 0
    turns = value.get(clause_ref, 0)
    return turns if type(turns) is int and turns >= 0 else 0


def _execution_witness_key(runtime: object, clause_ref: str, *, phase: object = None,
                           check_identity: object = None, revision: object = None) -> str:
    binding = getattr(runtime, "binding", None)
    try:
        bound = binding() if callable(binding) else binding
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        bound = None
    if revision is None:
        revision = (bound.get("product_revision") if isinstance(bound, Mapping) else None)
    if revision is None:
        revision = getattr(runtime, "_clause_discriminator_witness_revision",
                           getattr(runtime, "revision", ""))
    phase_value = phase if phase is not None else getattr(
        runtime, "_clause_discriminator_witness_phase", "execution")
    check_value = check_identity if check_identity is not None else getattr(
        runtime, "_clause_discriminator_check_identity", "")
    return "|".join((str(clause_ref), str(check_value or ""), str(revision or ""),
                      str(phase_value or "execution")))


def _execution_turns_used(runtime: object, key: str) -> int:
    value = getattr(runtime, "_clause_discriminator_witness_turns", {})
    if not isinstance(value, Mapping):
        return 0
    turns = value.get(key, 0)
    return turns if type(turns) is int and turns >= 0 else 0


def note_witness_turn(runtime: object, clause_ref: object, *, admitted: bool = False,
                      executed: bool = False, phase: object = None,
                      check_identity: object = None, revision: object = None) -> bool:
    """Record a witness only after admission and execution both succeeded."""
    ident = str(clause_ref or "").strip()
    if not ident or not admitted or not executed:
        return False
    current = getattr(runtime, "_clause_discriminator_witness_turns", None)
    turns = dict(current) if isinstance(current, Mapping) else {}
    key = _execution_witness_key(runtime, ident, phase=phase,
                                 check_identity=check_identity, revision=revision)
    if _execution_turns_used(runtime, key) >= MAX_WITNESS_TURNS_PER_CLAUSE:
        return False
    turns[key] = min(
        MAX_WITNESS_TURNS_PER_CLAUSE,
        _execution_turns_used(runtime, key) + 1,
    )
    setattr(runtime, "_clause_discriminator_witness_turns", turns)
    return True


def settle_witness_execution(runtime: object, admission: Mapping[str, object],
                             result: Mapping[str, object] | None) -> bool:
    """Charge one witness turn only after an admitted case actually settles.

    Rejected proposals, unavailable runners, environment failures, timeouts,
    and resource/policy refusals remain recoverable ``NOT_RUN`` evidence.  A
    completed observation, including a real counterexample, consumes the
    bounded execution slot under its clause/check/revision/phase identity.
    """
    if not isinstance(admission, Mapping) or admission.get("admitted") is not True:
        return False
    data = result if isinstance(result, Mapping) else {}
    status = str(data.get("receipt_status", data.get("runner_status", data.get("status", "")))).upper()
    outcome = str(data.get("result", "")).upper()
    non_execution = {"", "NOT_RUN", "UNRUNNABLE", "ENVIRONMENT", "TIMEOUT",
                     "TIMED_OUT", "RESOURCE_LIMIT", "POLICY_REFUSED", "RUNNER_ERROR",
                     "REFUSED", "REFUSED_PREDISPATCH", "REJECTED", "REJECTED_CONTRACT",
                     "MALFORMED", "SCHEMA_ERROR", "NO_OP", "NOOP"}
    receipt_id = str(data.get("receipt_id", data.get("witness_id", ""))).strip()
    if (status in non_execution or outcome in {"NOT_RUN", "UNRUNNABLE", "REFUSED", "REJECTED"}
            or not receipt_id):
        return False
    clause_ref = str(admission.get("clause_id", admission.get("clause_ref", ""))).strip()
    if not clause_ref:
        return False
    key = _execution_witness_key(
        runtime, clause_ref, phase=admission.get("phase"),
        check_identity=admission.get("check_identity"),
        revision=admission.get("revision"),
    )
    if _execution_turns_used(runtime, key) >= MAX_WITNESS_TURNS_PER_CLAUSE:
        return False
    current = getattr(runtime, "_clause_discriminator_witness_turns", None)
    turns = dict(current) if isinstance(current, Mapping) else {}
    turns[key] = min(MAX_WITNESS_TURNS_PER_CLAUSE, _execution_turns_used(runtime, key) + 1)
    setattr(runtime, "_clause_discriminator_witness_turns", turns)
    setattr(runtime, "_clause_discriminator_witness_phase", admission.get("phase", "execution"))
    setattr(runtime, "_clause_discriminator_check_identity",
            admission.get("check_identity", ""))
    setattr(runtime, "_clause_discriminator_witness_revision", admission.get("revision", ""))
    setattr(runtime, "_clause_discriminator_current_key", key)
    return True


def witness_route(runtime: object, tools: object = ()) -> tuple[bool, str]:
    """Return whether one executable, funded witness turn remains."""
    rows = open_clauses(runtime)
    if not rows:
        return False, "clause_closed"
    offered = {
        str(tool.get("function", {}).get("name", "")).strip()
        for tool in _sequence(tools) if isinstance(tool, Mapping)
    }
    if not offered.intersection({"probe", "authored_check"}):
        return False, "witness_rail_unavailable"
    clause_ref = str(rows[0]["clause_ref"])
    current_key = _execution_witness_key(runtime, clause_ref)
    if _execution_turns_used(runtime, current_key) >= MAX_WITNESS_TURNS_PER_CLAUSE:
        return False, "witness_turn_limit_exhausted"
    try:
        from kernel_route_budget import _budget
        remaining = _budget(runtime).get("remaining")
    except (ImportError, AttributeError, TypeError, ValueError):
        remaining = getattr(runtime, "probe_remaining", None)
    if type(remaining) is int and remaining <= 0:
        return False, "witness_route_budget_exhausted"

    continuation = getattr(runtime, "continuation_limits", None)
    if continuation is None:
        continuation = getattr(runtime, "decision_budget", None)
    qualify = getattr(continuation, "qualify_admission", None)
    if callable(qualify):
        try:
            if not qualify(model_decisions=1, tool_attempts=1):
                return False, "continuation_limit_exhausted"
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False, "continuation_limit_unavailable"
    else:
        remaining_decisions = getattr(continuation, "model_decisions_remaining", None)
        remaining_tools = getattr(continuation, "tool_attempts_remaining", None)
        if type(remaining_decisions) is int and remaining_decisions <= 0:
            return False, "continuation_limit_exhausted"
        if type(remaining_tools) is int and remaining_tools <= 0:
            return False, "continuation_tool_limit_exhausted"
    return True, "available"


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _text(value: object, limit: int = 1200) -> str:
    return (value if isinstance(value, str) else str(value or ""))[:limit]


def _runtime_revision(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    try:
        value = binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        value = None
    if isinstance(value, Mapping):
        # Binding is a structured identity.  Clause/check receipts bind to the
        # product revision, while artifact/environment digests remain separate
        # receipt fields.  Returning the whole mapping made a valid authored
        # candidate look stale merely because the host exposed extra identity.
        for key in ("product_revision", "current_revision", "revision"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return copy.deepcopy(candidate)
    if value is not None:
        return copy.deepcopy(value)
    for name in ("product_revision", "host_closure_snapshot", "current_revision"):
        value = getattr(runtime, name, None)
        if value is not None:
            return copy.deepcopy(value)
    return "unknown"


def _revision_tokens(runtime: object, ledger: Mapping[str, object]) -> frozenset[str]:
    values: list[str] = []
    for value in (_runtime_revision(runtime), ledger.get("current_revision"),
                  getattr(runtime, "host_closure_snapshot", None)):
        if value is None:
            continue
        values.extend((str(value), _digest(value)))
    return frozenset(item for item in values if item and item != "unknown")


def _receipt_revision(receipt: object) -> str:
    provenance = _field(receipt, "provenance", {})
    authored = (isinstance(provenance, Mapping)
                and _text(provenance.get("kind", "")) == "AUTHORED_CHECK")
    if authored:
        value = _field(receipt, "candidate_revision", None)
        if value is not None and str(value).strip():
            return str(value)
    for name in ("repository_revision", "snapshot_digest", "snapshot", "revision",
                 "revision_digest"):
        value = _field(receipt, name, None)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _receipt_status(receipt: object) -> str:
    for name in ("underlying_status", "underlying_result", "result", "status",
                 "result_class", "outcome"):
        value = _field(receipt, name, None)
        if value is not None and str(value).strip():
            return str(value).upper()
    return ""


def _receipt_passes(receipt: object, revisions: frozenset[str]) -> bool:
    provenance = _field(receipt, "provenance", {})
    if isinstance(provenance, Mapping) and provenance.get("kind") == "AUTHORED_CHECK":
        status = _text(_field(receipt, "status", "")).upper()
        result = _text(_field(receipt, "result", "")).upper()
        if status not in _PASS or result not in {"BEHAVIOR_FIXED", "PASS", "COMPLETED"}:
            return False
        parent = _text(_field(receipt, "parent_revision", ""))
        candidate = _text(_field(receipt, "candidate_revision", ""))
        if (not _receipt_id(receipt) or not _text(_field(receipt, "check_id", ""))
                or not _text(_field(receipt, "clause_id", ""))
                or not _text(_field(receipt, "check_hash", ""))
                or not _text(_field(receipt, "execution_identity", ""))
                or not _text(_field(receipt, "run_identity", ""))
                or not _text(_field(receipt, "candidate_identity", ""))
                or not _text(_field(receipt, "candidate_digest", ""))
                or not _text(_field(receipt, "task_id", ""))
                or not isinstance(_field(receipt, "public_source_span", None), Mapping)
                or not _text(_field(receipt, "environment_identity", ""))
                or not parent or not candidate or parent == candidate
                or candidate not in revisions
                or _field(receipt, "certifies_clause", False) is not True
                or _field(receipt, "observation_only", False) is True
                or _field(receipt, "reached_observation", False) is not True
                or _field(receipt, "integrity", False) is not True):
            return False
        return True
    status = _receipt_status(receipt)
    if status not in _PASS:
        return False
    revision = _receipt_revision(receipt)
    if not revision or (revisions and revision not in revisions):
        return False
    execution = str(_field(receipt, "execution", "EXECUTED")).upper()
    if execution not in _EXECUTED:
        return False
    reached = _field(receipt, "assertion_reached", None)
    return reached is not False


def _receipt_id(receipt: object) -> str:
    return _text(_field(receipt, "receipt_id", _field(receipt, "id", "")), 240)


def _check_ids(requirement: Mapping[str, object]) -> tuple[str, ...]:
    refs = requirement.get("check_refs", ())
    values: list[str] = []
    for ref in _sequence(refs):
        value = ref.get("id") if isinstance(ref, Mapping) else ref
        if str(value).strip():
            values.append(str(value))
    return tuple(dict.fromkeys(values))


def _ledger(runtime: object) -> dict[str, object] | None:
    existing = getattr(runtime, "requirements_ledger", None)
    if isinstance(existing, Mapping) and _sequence(existing.get("requirements")):
        return existing if isinstance(existing, dict) else dict(existing)
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", None)
    if not isinstance(row, Mapping):
        row = getattr(runtime, "task_row", None)
    row = dict(row) if isinstance(row, Mapping) else {}
    # Ordinary task messages are not public specifications.  Require an
    # explicit public-brief/ticket seam before creating obligations; otherwise
    # this lane would turn every unrelated smoke objective into a new clause.
    public_spec = next((row.get(name) for name in (
        "public_brief", "_bridge_ordinary_brief", "ticket_text", "public_ticket",
        "public_spec", "public_contract") if isinstance(row.get(name), str)
                        and row.get(name).strip()), None)
    if public_spec is None:
        public_spec = next((getattr(runtime, name, None) for name in (
            "public_brief", "ticket_text", "public_ticket", "public_spec",
            "public_contract") if isinstance(getattr(runtime, name, None), str)
                           and getattr(runtime, name).strip()), None)
    if public_spec is None and not isinstance(row.get("requirements_ledger"), Mapping):
        embedded = getattr(runtime, "requirements_ledger", None)
        if not isinstance(embedded, Mapping):
            return None
    if public_spec is not None and not row.get("public_brief"):
        row["public_brief"] = public_spec
    if not row.get("id"):
        row["id"] = str(getattr(runtime, "task_id", "public-spec"))
    for field_name in ("files", "public_sources", "public_tests", "public_test_files"):
        if field_name not in row:
            value = getattr(runtime, field_name, None)
            if value is not None:
                row[field_name] = value
    if not row.get("public_brief") and not isinstance(row.get("requirements_ledger"), Mapping):
        return None
    try:
        from kernel_requirements_ledger import build_ledger
        root = getattr(owner, "root", getattr(runtime, "root", None))
        compiled = build_ledger(
            row,
            root=Path(root) if root is not None else None,
            current_revision=_runtime_revision(runtime),
        )
    except (ImportError, OSError, TypeError, ValueError):
        return None
    setattr(runtime, "requirements_ledger", compiled)
    return compiled


def _public_text(runtime: object, ledger: Mapping[str, object]) -> str:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    row = row if isinstance(row, Mapping) else {}
    source = next((item for item in ledger.get("sources", ())
                   if isinstance(item, Mapping) and item.get("kind") == "ticket"), {})
    return "\n".join(
        str(value) for value in (
            source.get("text", "") if isinstance(source, Mapping) else "",
            getattr(runtime, "behavior_witness_brief", ""),
            getattr(runtime, "public_brief", ""),
            row.get("public_brief", row.get("brief", row.get("msg", ""))),
        ) if str(value or "").strip())


def _public_interface(runtime: object) -> str:
    values = [getattr(runtime, name, "") for name in
              ("behavior_witness_interface", "public_interface", "public_api")]
    # The current bound product source is public material too.  This keeps
    # structured/nested rows truthful after an edit without requiring the row
    # author to duplicate current source literals in a prose interface field.
    try:
        from kernel_behavior_witness import build_public_interface_index
        index = build_public_interface_index(runtime)
        if index is not None:
            values.append(index.text)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    if isinstance(row, Mapping):
        values.extend(row.get(name, "") for name in
                      ("public_interface", "interface", "public_source", "files"))
        # Boundary rows may declare their public symbol structurally instead of
        # repeating it in free-form interface text.  Preserve that declaration
        # in the same public-only validation surface used by the typed witness.
        entry = row.get("entry")
        if isinstance(entry, Mapping):
            entry_path = str(entry.get("path", "")).replace("\\", "/").strip()
            entry_fn = str(entry.get("fn", "")).strip()
            if entry_path.endswith(".py") and entry_fn:
                values.append(entry_path[:-3].replace("/", ".") + "." + entry_fn)
    return "\n".join(str(value) for value in values if str(value or "").strip())


def _requirement_rows(ledger: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(row for row in _sequence(ledger.get("requirements", ()))
                 if isinstance(row, Mapping) and str(row.get("id", "")).strip())


def _witness_records(runtime: object) -> tuple[object, ...]:
    values: list[object] = []
    for name in ("_clause_discriminator_receipts", "witness_evidence_receipts",
                 "behavior_witness_records"):
        values.extend(_sequence(getattr(runtime, name, ())))
    return tuple(values)


def _is_authored(receipt: object) -> bool:
    provenance = receipt.get("provenance") if isinstance(receipt, Mapping) else None
    return isinstance(provenance, Mapping) and str(provenance.get("kind", "")).upper() == "AUTHORED_CHECK"


def _authored_bound(receipt: object, clause_ref: str, host_clause: object,
                    host_span: object = None) -> bool:
    """W3: an authored receipt covers a clause only via its host binding."""
    if not isinstance(receipt, Mapping):
        return False
    from kernel_authored_check import binding_certifies
    return binding_certifies(receipt, clause_ref, host_clause, clause_span=host_span,
                             allow_alias=True)


def covered_requirement_ids(
        ledger: Mapping[str, object], receipts: object = (), witnesses: object = (),
        *, current_revision: object = None) -> frozenset[str]:
    """Return clauses covered by an executed current public check or witness."""
    revisions = frozenset(
        item for item in (str(current_revision), _digest(current_revision))
        if current_revision is not None
    )
    if not revisions:
        revisions = frozenset({str(ledger.get("current_revision", ""))})
    all_receipts = tuple(_sequence(ledger.get("receipts", ()))) + tuple(_sequence(receipts))
    passed_checks = {
        str(_field(receipt, "check_id", ""))
        for receipt in all_receipts
        if _receipt_passes(receipt, revisions)
        and _receipt_has_reached_consumers(receipt, ledger)
    }
    covered: set[str] = set()
    for requirement in _requirement_rows(ledger):
        if passed_checks.intersection(_check_ids(requirement)):
            covered.add(str(requirement["id"]))
        authored_clause = str(requirement["id"])
        host_span = _ledger_clause_span(ledger, requirement)
        if any(
                _receipt_passes(receipt, revisions)
                and isinstance(_field(receipt, "provenance", {}), Mapping)
                and _field(receipt, "provenance", {}).get("kind") == "AUTHORED_CHECK"
                and _authored_bound(receipt, authored_clause, requirement.get("exact_clause", ""), host_span)
                and _receipt_has_reached_consumers(receipt, ledger)
                and _receipt_scope_complete(receipt, requirement)
                for receipt in all_receipts):
            covered.add(authored_clause)
    all_witnesses = (
        tuple(_sequence(ledger.get("witness_receipts", ())))
        + tuple(_sequence(witnesses))
        + tuple(
            receipt for receipt in all_receipts
            if isinstance(receipt, Mapping) and receipt.get("certifies_clause") is True
        )
    )
    for witness in all_witnesses:
        if not isinstance(witness, Mapping):
            continue
        if (str(witness.get("result", "")).upper() in _PASS
                and witness.get("certifies_clause") is True
                and _receipt_revision(witness) in revisions
                and _immutable_solver_receipt(witness, ledger)):
            clause_ref = str(witness.get("clause_ref", witness.get("clause_id", "")))
            requirement = next((row for row in _requirement_rows(ledger)
                                if str(row.get("id")) == clause_ref), None)
            if (clause_ref and requirement is not None
                    and _receipt_scope_complete(witness, requirement)
                    and (not _is_authored(witness) or _authored_bound(
                        witness, clause_ref, requirement.get("exact_clause", ""),
                        _ledger_clause_span(ledger, requirement)))):
                covered.add(clause_ref)
    return frozenset(covered)


def _receipt_has_reached_consumers(receipt: object, ledger: Mapping[str, object]) -> bool:
    """Shared obligations need every declared consumer, not one convenient site."""
    if not isinstance(receipt, Mapping):
        return False
    reached = receipt.get("reached_consumers", receipt.get("reached_source_units", ()))
    reached_set = {str(item) for item in _sequence(reached)}
    for requirement in _requirement_rows(ledger):
        check_ids = _check_ids(requirement)
        if str(receipt.get("check_id", "")) not in check_ids:
            continue
        contract = requirement.get("contract", {})
        declared = isinstance(contract, Mapping) and (
            "source_unit_handles" in contract or "consumers" in contract)
        expected = contract.get("source_unit_handles", contract.get("consumers", ())) \
            if isinstance(contract, Mapping) else ()
        expected_set = {str(item) for item in _sequence(expected)}
        if declared and not expected_set:
            return False
        if expected_set:
            return expected_set.issubset(reached_set)
        contract_map = contract if isinstance(contract, Mapping) else {}
        universal = str(contract_map.get("quantifier", "")).casefold() in {
            "universal", "universal_in_stated_scope", "all", "every", "each",
        } or bool(contract_map.get("shared_scope"))
        return not universal or receipt.get("coverage_complete") is True
    return True


def _receipt_scope_complete(receipt: Mapping[str, object], requirement: Mapping[str, object]) -> bool:
    """One case cannot certify a universal/shared clause without scope proof."""

    contract = requirement.get("contract", {})
    contract = contract if isinstance(contract, Mapping) else {}
    expected = contract.get("source_unit_handles", contract.get("consumers", ()))
    expected_set = {str(item) for item in _sequence(expected)}
    reached = receipt.get("reached_consumers", receipt.get("reached_source_units", ()))
    reached_set = {str(item) for item in _sequence(reached)}
    if expected_set:
        return expected_set.issubset(reached_set)
    universal = str(contract.get("quantifier", "")).casefold() in {
        "universal", "universal_in_stated_scope", "all", "every", "each",
    } or bool(contract.get("shared_scope"))
    return not universal or receipt.get("coverage_complete") is True


def _immutable_solver_receipt(receipt: Mapping[str, object], ledger: Mapping[str, object] | None = None) -> bool:
    required = ("source_unit_handle", "revision", "input", "observation",
                "parent_identity", "candidate_identity", "clause_id")
    if not all(key in receipt and (key in {"input", "observation"}
                                  or receipt.get(key) not in (None, ""))
               for key in required):
        return False
    reached = receipt.get("reachability", receipt.get("reachable"))
    if reached is not True or receipt.get("immutable") is not True:
        return False
    if receipt.get("solver_authored") is not True or receipt.get("host_authored") is True:
        return False
    provenance = receipt.get("provenance")
    if isinstance(provenance, Mapping) and str(provenance.get("kind", "")).upper() in {
            "HOST_CHECK", "HOST_AUTHORED", "MODEL_CLAIM"}:
        return False
    source_revision = receipt.get("source_revision")
    if source_revision not in (None, "", receipt.get("revision")):
        return False
    if not _text(receipt.get("check_identity")) or not _text(receipt.get("oracle_identity")):
        return False
    store = ledger.get("execution_store", ledger.get("host_execution_store", ())) if isinstance(ledger, Mapping) else ()
    try:
        from kernel_aha10_honest_loop import _host_execution_record
        host = _host_execution_record(receipt, store)
    except (ImportError, TypeError, ValueError):
        host = None
    if host is None:
        return False
    host_identity_fields = (
        "receipt_id", "check_identity", "oracle_identity", "source_unit_handle",
        "revision", "parent_identity", "candidate_identity", "clause_id",
    )
    if any(
            not _text(host.get(key)) or host.get(key) != receipt.get(key)
            for key in host_identity_fields
    ):
        # The host store is intentionally compact; identity, status, and the
        # immutable receipt digest are the authority boundary.
        return False
    if any(key in host for key in ("input", "observation", "integrity_digest", "integrity")) \
            and dict(host) != dict(receipt):
        return False
    host_status = str(host.get("status", host.get("outcome", ""))).upper()
    if host_status not in {"EXECUTED", "PASS", "PASSED", "FAIL", "FAILED"}:
        return False
    for key in ("check_identity", "oracle_identity", "source_unit_handle", "revision",
                "parent_identity", "candidate_identity", "clause_id"):
        if key in host and host.get(key) != receipt.get(key):
            return False
    digest = receipt.get("integrity_digest", receipt.get("integrity"))
    if not isinstance(digest, str):
        return False
    try:
        from kernel_aha10_honest_loop import _digest
        payload = {key: value for key, value in receipt.items()
                   if key not in {"integrity_digest", "integrity"}}
        return digest == _digest(payload)
    except (ImportError, TypeError, ValueError):
        return False


def clause_rows(runtime: object) -> tuple[dict[str, object], ...]:
    """Return every extracted public clause with host coverage status."""
    ledger = _ledger(runtime)
    if ledger is None:
        return ()
    current = _runtime_revision(runtime)
    covered = covered_requirement_ids(
        ledger,
        (*_sequence(getattr(runtime, "host_check_receipts", ())),
         *_sequence(getattr(runtime, "test_receipts", ()))),
        _witness_records(runtime),
        current_revision=current,
    )
    result: list[dict[str, object]] = []
    for requirement in _requirement_rows(ledger):
        ident = str(requirement["id"])
        assessment = requirement.get("assessment", {})
        evidence_ids = list(assessment.get("evidence_ids", ())) if isinstance(assessment, Mapping) else []
        result.append({
            "clause_ref": ident,
            "obligation_id": ident,
            "clause_id": ident,
            "clause": str(requirement.get("exact_clause", "")),
            "kind": str(requirement.get("kind", "unresolved")),
            "contract": copy.deepcopy(requirement.get("contract", {})),
            "check_ids": list(_check_ids(requirement)),
            "evidence_ids": evidence_ids,
            "coverage_required": True,
            "covered": ident in covered,
            "status": "satisfied" if ident in covered else "open",
            "reason": "current clause-specific receipt" if ident in covered
                      else "no current clause-specific receipt",
        })
    return tuple(result)


def open_clauses(runtime: object) -> tuple[dict[str, object], ...]:
    return tuple(row for row in clause_rows(runtime) if not row["covered"])


def open_requirement_ids(runtime: object) -> tuple[str, ...]:
    return tuple(str(row["clause_ref"]) for row in open_clauses(runtime))


def _literal_rows(value: object, path: str = "value") -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key, item in value.items():
            rows.extend(_literal_rows(item, f"{path}.{key}"))
        return tuple(rows)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_literal_rows(item, f"{path}[{index}]"))
        return tuple(rows)
    return ((path, value),) if value is not None else ()


def _literal_in_public(value: object, public_text: str) -> bool:
    if isinstance(value, bool):
        return str(value).lower() in public_text.lower()
    return str(value) in public_text


def _unknown_partition_allowed(value: object, contract: object) -> bool:
    if not isinstance(contract, Mapping):
        return False
    allowed_values: list[object] = []
    for expectation in _sequence(contract.get("expectations", ())):
        if isinstance(expectation, Mapping) and expectation.get("operator") == "member_of":
            allowed_values.extend(_sequence(expectation.get("expected", ())))
    if not allowed_values or not isinstance(value, Mapping):
        return False
    allowed = {str(item) for item in allowed_values}
    for candidate in value.values():
        if isinstance(candidate, str) and candidate not in allowed:
            return True
    return False


def _normalize_relation(value: object) -> tuple[dict[str, object], object]:
    if not isinstance(value, Mapping):
        raise ClauseDiscriminatorError("expected_relation must be an object")
    operator = str(value.get("operator", "")).strip().lower()
    if operator not in _RELATION_OPERATORS:
        raise ClauseDiscriminatorError("expected_relation operator is not host-admitted")
    expected = value.get("expected", value.get("value"))
    if "expected" not in value and "value" not in value:
        raise ClauseDiscriminatorError("expected_relation needs expected/value")
    if not _json_safe(expected):
        raise ClauseDiscriminatorError("expected_relation value must be JSON")
    return {"operator": operator, "expected": copy.deepcopy(expected)}, copy.deepcopy(expected)


def _public_contract_case(runtime: object, row: Mapping[str, object]) -> dict[str, object] | None:
    """Resolve an immutable source-bound case without trusting model prose."""
    candidates: list[object] = [row.get("public_contract")]
    contract = row.get("contract")
    if isinstance(contract, Mapping):
        candidates.append(contract)
    runtime_cases = getattr(runtime, "public_witness_contracts", None)
    if isinstance(runtime_cases, Mapping):
        candidates.extend((runtime_cases.get(str(row.get("clause_ref", ""))),
                           runtime_cases.get(str(row.get("id", "")))))
    revision_tokens = _revision_tokens(runtime, _ledger(runtime) or {})
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        provenance = candidate.get("source_provenance", candidate.get("provenance"))
        if not isinstance(provenance, Mapping) or provenance.get("immutable") is not True:
            continue
        kind = str(provenance.get("kind", provenance.get("source", ""))).casefold()
        if kind not in {"public_source", "public_contract", "host_derived"}:
            continue
        source_unit = str(provenance.get("source_unit_handle", candidate.get(
            "source_unit_handle", ""))).strip()
        if not source_unit:
            continue
        source_revision = provenance.get("revision", candidate.get("revision"))
        if source_revision is not None and revision_tokens and not ({
                str(source_revision), _digest(source_revision)} & revision_tokens):
            continue
        entrypoint = str(candidate.get("entrypoint", "")).strip()
        input_value = candidate.get("input", candidate.get("args"))
        expected_value = candidate.get("expected_relation", candidate.get("expected"))
        if isinstance(expected_value, Mapping):
            relation, expected = _normalize_relation(expected_value)
        else:
            relation, expected = ({"operator": "equals", "expected": copy.deepcopy(expected_value)},
                                  copy.deepcopy(expected_value))
        if (not (_ENTRYPOINT_RE.fullmatch(entrypoint) or source_unit)
                or input_value is None
                or expected_value is None or not _json_safe(input_value)
                or not _json_safe(expected)):
            continue
        return {"entrypoint": entrypoint, "input": copy.deepcopy(input_value),
                "expected_relation": relation, "expected": expected,
                "source_unit_handle": source_unit, "source_revision": copy.deepcopy(source_revision),
                "expectation_origin": "public_contract_source"}
    return None


def _partition_literals(value: object, public_text: str) -> tuple[object, ...]:
    return tuple(dict.fromkeys(
        literal for _path, literal in _literal_rows(value)
        if not _literal_in_public(literal, public_text)))


def validate_proposal(runtime: object, proposal: object) -> dict[str, object]:
    """Validate one model proposal and return host-normalized execution data."""
    if not isinstance(proposal, Mapping):
        raise ClauseDiscriminatorError("proposal must be an object")
    required = {"clause_ref", "entrypoint", "input", "expected_relation"}
    missing = sorted(required - set(proposal))
    if missing:
        raise ClauseDiscriminatorError("missing proposal fields: " + ", ".join(missing))
    rows = {str(row["clause_ref"]): row for row in open_clauses(runtime)}
    clause_ref = str(proposal.get("clause_ref", "")).strip()
    row = rows.get(clause_ref)
    if row is None:
        raise ClauseDiscriminatorError(
            "clause_ref is not an open public clause (available="
            + repr(tuple(rows)) + ", requested=" + repr(clause_ref) + ")")
    entrypoint = str(proposal.get("entrypoint", "")).strip()
    proposal_source_unit = str(proposal.get("source_unit_handle", "")).strip()
    if not _ENTRYPOINT_RE.fullmatch(entrypoint) and not proposal_source_unit:
        raise ClauseDiscriminatorError("entrypoint needs a public dotted symbol or opaque source-unit handle")
    if not _json_safe(proposal.get("input")):
        raise ClauseDiscriminatorError("input must be JSON")
    relation, expected = _normalize_relation(proposal.get("expected_relation"))
    derived = _public_contract_case(runtime, row)
    derived_match = bool(
        isinstance(derived, Mapping)
        and entrypoint == derived.get("entrypoint")
        and proposal.get("input") == derived.get("input")
        and relation == derived.get("expected_relation")
    )
    public_text = _public_text(runtime, _ledger(runtime) or {}) + "\n" + _public_interface(runtime)
    if entrypoint not in public_text and not derived_match and not proposal_source_unit:
        raise ClauseDiscriminatorError("entrypoint is not present in public material")
    for _path, literal in _literal_rows(expected):
        if not _literal_in_public(literal, public_text) and not derived_match:
            raise ClauseDiscriminatorError("expected_relation contains a non-public literal")
    input_value = proposal.get("input")
    if not all(_literal_in_public(literal, public_text)
               for _path, literal in _literal_rows(input_value)) \
            and not _unknown_partition_allowed(input_value, row.get("contract")) \
            and not derived_match:
        raise ClauseDiscriminatorError("input contains a non-public fitting literal")
    baseline = getattr(runtime, "baseline_tokens", getattr(runtime, "token_baseline", None))
    proposed_tokens = proposal.get("token_count")
    if type(baseline) is int and baseline > 0:
        if type(proposed_tokens) is not int:
            proposed_tokens = max(1, len(json.dumps(proposal, sort_keys=True, default=str)) // 4)
        if proposed_tokens > max(1, int(baseline * TOKEN_FRACTION)):
            raise ClauseDiscriminatorError("proposal exceeds the 0.08x token contract")
    return {
        "clause_ref": clause_ref,
        "entrypoint": entrypoint,
        "input": copy.deepcopy(input_value),
        "expected_relation": relation,
        "expected": expected,
        "clause": str(row["clause"]),
        "row": copy.deepcopy(row),
        "expectation_origin": (derived.get("expectation_origin", "public_clause")
                                if derived_match else "public_clause"),
        "source_unit_handle": (derived.get("source_unit_handle") if derived_match
                                else proposal_source_unit or None),
        "source_revision": derived.get("source_revision") if derived_match else None,
        "scope": {"clause_ref": clause_ref, "span_scope": str(row["clause"])},
        "token_contract": {"max_baseline_fraction": TOKEN_FRACTION,
                           "extra_model_calls": 0, "max_cases": MAX_CASES_PER_REQUEST},
    }


def _evidence_pin(*, receipt_id: str, revision: object, observed: object,
                  expectation_origin: str, scope: object) -> tuple[dict[str, object], str]:
    payload = {
        "schema": EVIDENCE_PIN_SCHEMA,
        "receipt": receipt_id,
        "revision": copy.deepcopy(revision),
        "observed": copy.deepcopy(observed),
        "expectation_origin": expectation_origin,
        "scope": copy.deepcopy(scope),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return payload, "[EVIDENCEPIN]" + encoded + "[/EVIDENCEPIN]"


def evidence_pin(*, receipt_id: str, revision: object, observed: object,
                 expectation_origin: str, scope: object) -> tuple[dict[str, object], str]:
    """Expose the shared REQWITNESS evidence-pin rail to adjacent host witnesses."""
    return _evidence_pin(
        receipt_id=receipt_id,
        revision=revision,
        observed=observed,
        expectation_origin=expectation_origin,
        scope=scope,
    )


def execute_proposal(runtime: object, proposal: object, runner: object) -> dict[str, object]:
    """Execute one validated proposal through the existing typed witness runner."""
    try:
        data = validate_proposal(runtime, proposal)
    except ClauseDiscriminatorError as error:
        return {"schema": SCHEMA, "status": "REJECTED", "reason": str(error)}
    current_revision = _runtime_revision(runtime)
    row = data["row"]
    try:
        from kernel_behavior_witness import admit_proposal, context_for_runtime, execute_admitted
        brief, interface, _unused = context_for_runtime(runtime, row, data["clause_ref"])
        # The legacy witness context may retain the ordinary task message.  A
        # discriminator proposal is grounded in the public clause ledger, so
        # admission must see that exact public text as well.
        public_brief = _public_text(runtime, _ledger(runtime) or {})
        if data["clause"] not in brief and data["clause"] in public_brief:
            brief = public_brief
        if not brief:
            brief = str(data["clause"])
        if not interface:
            interface = _public_interface(runtime)
        if data.get("expectation_origin") == "public_contract_source":
            # Keep the host-derived case visible to the existing public
            # witness adapter without converting it into model-authored prose.
            brief += "\n" + json.dumps({
                "entrypoint": data["entrypoint"], "input": data["input"],
                "expected": data["expected"],
                "source_unit_handle": data.get("source_unit_handle"),
            }, sort_keys=True, ensure_ascii=False, default=str)
        open_obligation = {
            "obligation_id": data["clause_ref"], "clause": data["clause"],
            "kind": data["row"].get("kind", "behavior"),
            "reason": "CLAUSE_COVERAGE", "status": "UNKNOWN",
            "coverage_required": True, "changed_surface": True,
        }
        admission = admit_proposal(
            {
                "clause_id": data["clause_ref"], "entrypoint": data["entrypoint"],
                "input": data["input"], "expected": data["expected"],
                "source_unit_handle": data.get("source_unit_handle"),
                "rationale": "host normalized the expected relation from an immutable public contract",
            },
            clause_id=data["clause_ref"], clause=data["clause"], brief=brief,
            interface=interface, revision=current_revision,
            open_obligation=open_obligation, open_obligations=(open_obligation,),
            journal=getattr(runtime, "journal", None),
            public_partition_literals=_partition_literals(
                data["input"], _public_text(runtime, _ledger(runtime) or {})
                + "\n" + _public_interface(runtime)),
        )
        if admission.get("admitted") is not True:
            return {"schema": SCHEMA, "status": "UNRUNNABLE", "admission": admission,
                    "clause_ref": data["clause_ref"]}
        witness = execute_admitted(runtime, admission, runner,
                                   question="clause discriminator " + data["clause_ref"])
        settle_witness_execution(runtime, admission, witness)
    except Exception as error:  # runner/admission failures remain truthful evidence
        witness = {"result": "UNRUNNABLE", "reason": type(error).__name__,
                   "clause_id": data["clause_ref"], "observed_full": None}
    observed = copy.deepcopy(witness.get("observed_full", witness.get("observed")))
    passed = str(witness.get("result", "")).upper() == "PASS" and observed == data["expected"]
    scope_complete = _receipt_scope_complete(
        {"reached_consumers": witness.get("reached_consumers", witness.get("reached_source_units", ())),
         "coverage_complete": witness.get("coverage_complete") is True},
        row,
    )
    receipt_id = _text(witness.get("receipt_id"), 240) or _digest({
        "clause_ref": data["clause_ref"], "revision": current_revision, "observed": observed,
    })
    pin, pin_text = _evidence_pin(
        receipt_id=receipt_id, revision=current_revision, observed=observed,
        expectation_origin=str(data.get("expectation_origin", "public_clause")), scope=data["scope"],
    )
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED" if passed else "COUNTEREXAMPLE" if witness.get("result") == "FAIL" else "UNRUNNABLE",
        "result": "PASS" if passed else "FAIL" if witness.get("result") == "FAIL" else "UNRUNNABLE",
        "clause_ref": data["clause_ref"], "clause_id": data["clause_ref"],
        "entrypoint": data["entrypoint"], "input": copy.deepcopy(data["input"]),
        "expected_relation": copy.deepcopy(data["expected_relation"]),
        "observed": observed, "receipt_id": receipt_id,
        "revision": copy.deepcopy(current_revision), "revision_digest": _digest(current_revision),
        "expectation_origin": data.get("expectation_origin", "public_clause"),
        "source_unit_handle": data.get("source_unit_handle"),
        "scope": copy.deepcopy(data["scope"]),
        "evidence_pin": pin, "evidence_pin_text": pin_text,
        "certifies_clause": passed and scope_complete,
        "scope_complete": scope_complete,
        "witness": copy.deepcopy(witness),
        "token_contract": copy.deepcopy(data["token_contract"]),
    }
    if passed:
        # Keep the post-receipt fact on the live runtime for menu projections
        # that occur after the witness action has already been closed.
        setattr(runtime, "_clause_discriminator_last_pass_revision",
                {"revision": copy.deepcopy(current_revision),
                 "clause_ref": data["clause_ref"]})
    records = getattr(runtime, "_clause_discriminator_receipts", None)
    if not isinstance(records, list):
        records = []
        setattr(runtime, "_clause_discriminator_receipts", records)
    records.append(copy.deepcopy(result))
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("clause_discriminator_receipt", producer="kernel_clause_discriminator",
             event_schema=SCHEMA, clause_ref=data["clause_ref"], receipt_id=receipt_id,
             result=result["result"], certifies_clause=passed,
             evidence_pin=pin, dispatch=False)
    return result


def admit_authored_check_proposal(
        proposal: Mapping[str, object], *, public_text: str,
        parent_revision: object, candidate_revision: object,
        authorized_paths: Sequence[str] = (), protected_paths: Sequence[str] = (),
        existing_check_ids: Sequence[str] = (),
    ) -> dict[str, object]:
    """Admit the separate authored-check artifact for one public clause."""
    from kernel_behavior_witness import admit_authored_check
    try:
        return admit_authored_check(
            proposal, public_text=public_text, parent_revision=parent_revision,
            candidate_revision=candidate_revision, authorized_paths=authorized_paths,
            protected_paths=protected_paths, existing_check_ids=existing_check_ids,
        )
    except TypeError as error:
        # Keep compatibility with an older loaded witness facade while the canonical
        # authored-check module remains the single admission implementation.
        if "existing_check_ids" not in str(error):
            raise
        from kernel_authored_check import admit_authored_check as direct_admit
        return direct_admit(
            proposal, public_text=public_text, parent_revision=parent_revision,
            candidate_revision=candidate_revision, authorized_paths=authorized_paths,
            protected_paths=protected_paths, existing_check_ids=existing_check_ids,
        )


def _host_authored_context(runtime: object, proposal: Mapping[str, object],
                           ledger: Mapping[str, object]) -> dict[str, object]:
    """Resolve public text, revisions, and scratch authority from the host."""
    from kernel_authored_check_context import (
        AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR,
        read_runtime_authored_check_context,
        write_runtime_authored_check_context,
    )
    context = read_runtime_authored_check_context(runtime)
    authority = context.get("scratch_authority")
    if not isinstance(authority, Mapping):
        authority = getattr(runtime, AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR, None)
    if (not str(context.get("public_text", "") or "").strip()
            or not isinstance(authority, Mapping)):
        bundle = _host_bundle(runtime)
        if bundle is not None:
            write_runtime_authored_check_context(runtime, bundle, ledger)
            context = read_runtime_authored_check_context(runtime)
    # A writer-produced projection contains only host check-bundle and ledger
    # text.  Older host-injected contexts remain compatible with this reader.
    public_text = str(context.get("public_text", "") or "")
    if not public_text.strip():
        public_text = _public_text(runtime, ledger)
    # A dispatcher never promotes proposal text to host authority.  Direct
    # admission remains available for offline callers, but this route requires
    # the runtime/ledger to provide public text explicitly.
    parent = context.get("parent_revision", getattr(runtime, "parent_revision", None))
    candidate = context.get("candidate_revision", _runtime_revision(runtime))
    if (not str(candidate or "").strip()
            or str(candidate).strip().casefold() == "unknown"):
        candidate = None
    authority = context.get("scratch_authority")
    if not isinstance(authority, Mapping):
        authority = getattr(runtime, AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR, None)
    # The model's write_authority is intentionally ignored here.  The host
    # must issue a scratch capability before this dispatcher can admit a check.
    allowed = context.get("authorized_paths", ())
    if (not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes))
            or not allowed):
        allowed = authority.get("allowed_paths", ()) if isinstance(authority, Mapping) else ()
    protected = context.get("protected_paths", getattr(runtime, "protected_test_paths", ()))
    host_proposal = dict(proposal)
    host_proposal.pop("public_text", None)
    host_proposal.pop("parent_revision", None)
    host_proposal.pop("candidate_revision", None)
    if isinstance(authority, Mapping):
        host_proposal["write_authority"] = dict(authority)
    for key in ("source_unit_handle", "reached_consumers", "coverage_complete", "required_partitions",
                "invocation"):
        if key in context:
            host_proposal[key] = context[key]
    return {
        "proposal": host_proposal, "public_text": public_text,
        "parent_revision": parent, "candidate_revision": candidate,
        "context": context,
        "authorized": tuple(str(item) for item in allowed if str(item).strip()),
        "protected": tuple(str(item) for item in protected if str(item).strip())
        if isinstance(protected, Sequence) and not isinstance(protected, (str, bytes)) else (),
    }


def _host_bundle(runtime: object) -> object:
    bundle_fn = getattr(runtime, "_host_check_bundle", None)
    if callable(bundle_fn):
        try:
            return bundle_fn()
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None
    for name in ("host_check_bundle_cache", "_host_check_bundle_cache"):
        bundle = getattr(runtime, name, None)
        if bundle is not None:
            return bundle
    return None


def _ledger_clause_span(ledger: Mapping[str, object], requirement: Mapping[str, object]) -> object:
    """Return one exact host span; absent or multi-span rows remain unresolved."""
    direct = requirement.get("clause_span", requirement.get("source_span"))
    if isinstance(direct, Mapping) and direct:
        return copy.deepcopy(dict(direct))
    refs = tuple(str(item).strip() for item in _sequence(requirement.get("source_span_ids", ()))
                 if str(item).strip())
    if len(refs) != 1:
        return None
    matches = [row for row in _sequence(ledger.get("spans", ()))
               if isinstance(row, Mapping) and str(row.get("id", "")) == refs[0]]
    return copy.deepcopy(dict(matches[0])) if len(matches) == 1 else None


def _host_clause_records(runtime: object) -> tuple[dict[str, object], ...]:
    """Project existing bundle specs and open ledger rows into host records.

    This is a per-call projection, not a second persisted ledger or index.  All
    identities are derived from host clause text and host source spans.
    """
    from kernel_authored_check import clause_digest, clause_identity

    records: list[dict[str, object]] = []
    bundle = _host_bundle(runtime)
    for spec in getattr(bundle, "specs", ()) or ():
        ident = str(_field(spec, "obligation_id", _field(spec, "id", "")) or "").strip()
        clause = str(_field(spec, "clause", "") or "")
        span = _field(spec, "clause_span", None)
        if not span:
            provenance = _field(spec, "provenance", {})
            span = provenance.get("clause_span") if isinstance(provenance, Mapping) else None
        if ident and clause.strip():
            records.append({
                "id": ident, "binding_id": ident, "clause": clause,
                "span": copy.deepcopy(span), "source": "host_obligation_bundle",
                "origin": "bundle", "text_digest": clause_digest(clause),
                "identity": clause_identity(clause, span),
            })

    ledger = _ledger(runtime)
    if ledger is None:
        return tuple(records)
    try:
        open_ids = {str(row.get("clause_ref", "")) for row in open_clauses(runtime)}
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        open_ids = set()
    for requirement in _requirement_rows(ledger):
        ident = str(requirement.get("id", "") or "").strip()
        clause = str(requirement.get("exact_clause", "") or "")
        if not ident or not clause.strip() or ident not in open_ids:
            continue
        span = _ledger_clause_span(ledger, requirement)
        records.append({
            "id": ident, "binding_id": ident, "clause": clause,
            "span": span, "source": "host_requirements_ledger", "origin": "ledger",
            "text_digest": clause_digest(clause),
            "identity": clause_identity(clause, span),
        })
    return tuple(records)


def _host_clause_ambiguous(record: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> bool:
    """Reject duplicate ids and same-text obligations at different spans."""
    ident = str(record.get("id", ""))
    same_id = [item for item in records if str(item.get("id", "")) == ident]
    identities = {str(item.get("identity", "")) for item in same_id}
    if len(same_id) > 1 or len(identities) > 1:
        return True
    if record.get("origin") != "bundle":
        return False
    digest = str(record.get("text_digest", ""))
    bundle_rows = [item for item in records
                   if item.get("origin") == "bundle"
                   and str(item.get("text_digest", "")) == digest]
    bundle_identities = {str(item.get("identity", "")) for item in bundle_rows}
    return len(bundle_identities) > 1


def _resolve_host_clause(runtime: object, requested_id: object) -> dict[str, object] | None:
    """Resolve a frame or bundle id to one unambiguous host clause record."""
    requested = str(requested_id or "").strip()
    if not requested:
        return None
    records = _host_clause_records(runtime)
    candidates = [record for record in records if str(record.get("id", "")) == requested]
    if len(candidates) != 1 or _host_clause_ambiguous(candidates[0], records):
        return None
    candidate = candidates[0]
    bundles = [record for record in records if record.get("origin") == "bundle"]
    ledger_rows = [record for record in records if record.get("origin") == "ledger"]
    if not bundles:
        return candidate
    bundle_groups: dict[str, set[str]] = {}
    for bundle_row in bundles:
        bundle_groups.setdefault(str(bundle_row.get("text_digest", "")), set()).add(
            str(bundle_row.get("identity", "")))
    if any(len(identities) > 1 for identities in bundle_groups.values()):
        return None
    identity = str(candidate.get("identity", ""))
    if not identity:
        return None
    if candidate.get("origin") == "bundle":
        matching_ledger = [row for row in ledger_rows if row.get("identity") == identity]
        return candidate if len(matching_ledger) == 1 else None
    matching_bundle = [row for row in bundles if row.get("identity") == identity]
    return matching_bundle[0] if len(matching_bundle) == 1 else None


def _host_obligation_clauses(runtime: object) -> dict[str, tuple[str, str]]:
    """Return resolved host clause text for compatibility callers."""
    clauses: dict[str, tuple[str, str]] = {}
    for record in _host_clause_records(runtime):
        resolved = _resolve_host_clause(runtime, record["id"])
        if resolved is not None:
            clauses[str(record["id"])] = (str(resolved["clause"]), str(resolved["source"]))
    return clauses


def _host_obligation_binding(runtime: object, proposal: Mapping[str, object]) -> dict[str, object] | None:
    from kernel_authored_check import obligation_binding
    ident = str(proposal.get("requirement_id", proposal.get("clause_id")) or "").strip()
    host = _resolve_host_clause(runtime, ident)
    if host is None:
        return None
    quote = proposal.get("requirement_quote", proposal.get("clause"))
    binding = obligation_binding(
        host["binding_id"], host["clause"], quote, source=host["source"],
        clause_span=host.get("span"),
    )
    if binding is not None:
        # This requested id was accepted only after the host identity join;
        # retain it so later reducers can reject relabelled receipts.
        binding["host_requirement_id"] = ident
    return binding


def execute_authored_check_proposal(
        runtime: object, proposal: Mapping[str, object], *,
        parent_snapshot: Mapping[str, object], candidate_snapshot: Mapping[str, object],
        adapter: object = None, observation_only: bool = False,
) -> dict[str, object]:
    """Dispatch one frozen authored check and retain the canonical host receipt."""
    ledger = _ledger(runtime) or {}
    host = _host_authored_context(runtime, proposal, ledger)
    public_text = host["public_text"]
    parent_revision = host["parent_revision"]
    candidate_revision = host["candidate_revision"]
    authorized = host["authorized"]
    protected = host["protected"]
    # A frozen check may be replayed for later candidate revisions.  The
    # direct admission API still rejects duplicate proposals, while this host
    # dispatcher deliberately lets the immutable check identity/version layer
    # distinguish idempotent replay from changed check bytes.
    existing_ids: tuple[str, ...] = ()
    if not isinstance(authorized, Sequence) or isinstance(authorized, (str, bytes)):
        authorized = ()
    if not isinstance(protected, Sequence) or isinstance(protected, (str, bytes)):
        protected = ()
    admission = admit_authored_check_proposal(
        host["proposal"], public_text=public_text, parent_revision=parent_revision,
        candidate_revision=candidate_revision, authorized_paths=authorized,
        protected_paths=protected, existing_check_ids=existing_ids,
    )
    if admission.get("admitted") is not True:
        return {"schema": SCHEMA, "status": "REJECTED", "admission": admission,
                "certifies_clause": False}
    # W3: certification meaning is host-bound.  The requirement id must name an
    # open obligation of THIS run and the quote must cover that obligation's
    # host clause; quote containment in public text alone is not a binding.
    binding = _host_obligation_binding(runtime, host["proposal"])
    if binding is None:
        return {"schema": SCHEMA, "status": "REJECTED", "result": "UNBOUND_REQUIREMENT",
                "admission": {**admission, "admitted": False, "status": "REJECTED",
                              "code": "UNBOUND_REQUIREMENT",
                              "reason": "requirement_id is not an open host obligation whose "
                                        "clause the requirement_quote covers"},
                "certifies_clause": False, "route": "AUTHORED_CHECK"}
    host_context = host.get("context", {})
    frozen = None
    try:
        from kernel_authored_check_lifecycle import freeze_check, register_frozen_check
        lifecycle_context = dict(host_context) if isinstance(host_context, Mapping) else {}
        lifecycle_context.update({
            "public_text": public_text,
            "parent_revision": parent_revision,
            "candidate_revision": candidate_revision,
            "scratch_authority": host["proposal"].get("write_authority", {}),
            "role": lifecycle_context.get("role", "REPAIR"),
        })
        frozen = register_frozen_check(runtime, freeze_check(lifecycle_context, host["proposal"]))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # A route without a persisted immutable executable cannot certify.  The
        # direct benchmark compatibility path remains executable only when the
        # canonical admission itself is used outside this dispatcher.
        return {"schema": SCHEMA, "status": "UNRUNNABLE", "result": "UNRUNNABLE",
                "admission": admission, "certifies_clause": False,
                "reason": "frozen_check_persistence_unavailable", "route": "AUTHORED_CHECK"}
    if isinstance(host_context, Mapping) and isinstance(host_context.get("binding_stable"), bool):
        parent_snapshot = {**parent_snapshot, "binding_stable": host_context["binding_stable"]}
        candidate_snapshot = {**candidate_snapshot, "binding_stable": host_context["binding_stable"]}
    # Execute the exact frozen artifact through the lifecycle owner.  Calling
    # the legacy pair facade here would make persistence ornamental: fixtures,
    # harness bytes, and manifest completeness would not be part of execution.
    from kernel_authored_check_lifecycle import execute_check, pair_executions
    parent_execution = execute_check(
        frozen, {**parent_snapshot, "side": "parent"}, adapter)
    candidate_execution = execute_check(
        frozen, {**candidate_snapshot, "side": "candidate"}, adapter)
    result = pair_executions(
        parent_execution, candidate_execution,
        role=frozen.role, clause_id=str(admission.get("clause_id", "")), check=frozen,
    )
    result["host_obligation_binding"] = copy.deepcopy(binding)
    if isinstance(host_context, Mapping):
        for key in ("source_unit_handle", "reached_consumers", "coverage_complete",
                    "required_partitions"):
            if key in host_context:
                result[key] = copy.deepcopy(host_context[key])
    if observation_only:
        result["result"] = result["status"] = "OBSERVATION_ONLY"
        result["certifies_clause"] = False
        result["observation_only"] = True
    result["check_version"] = frozen.version
    result["logical_check_identity"] = frozen.identity
    result["clause_ref"] = admission.get("clause_id")
    result["route"] = "AUTHORED_CHECK"
    if isinstance(binding, Mapping) and binding.get("clause_identity"):
        result["host_clause_identity"] = binding["clause_identity"]
    try:
        from kernel_authored_check_lifecycle import attach_receipt
        attached = attach_receipt(runtime, result)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        attached = False
    if not attached:
        result["result"] = result["status"] = "UNRUNNABLE"
        result["certifies_clause"] = False
        result["reason"] = "normalized_receipt_attachment_refused"
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("clause_authored_check_dispatched", producer="kernel_clause_discriminator",
             event_schema=SCHEMA, clause_ref=result.get("clause_ref"),
             receipt_id=result.get("receipt_id"), result=result.get("result"),
             certifies_clause=result.get("certifies_clause") is True, dispatch=True)
    return result


def next_action(runtime: object, tools: object = ()) -> dict[str, object] | None:
    """Return one bounded witness action; no additional model call is required."""
    available, _reason = witness_route(runtime, tools)
    if not available:
        return None
    rows = open_clauses(runtime)
    if not rows:
        return None
    row = rows[0]
    contract = row.get("contract")
    derived = _public_contract_case(runtime, row)
    contract_ready = isinstance(contract, Mapping) and all(
        key in contract for key in ("entrypoint", "input", "expected", "expected_relation")
    )
    if (not contract_ready) and isinstance(derived, Mapping):
        contract = derived
    if not isinstance(contract, Mapping):
        return {
            "tool": "host_blocked",
            "arguments": {"clause_ref": str(row["clause_ref"]),
                          "status": "UNRUNNABLE",
                          "reason": "public contract is not constructible"},
            "reason": "no executable public witness contract is admitted",
            "clears": "nothing; acquire public contract evidence first",
        }
    entrypoint = str(contract.get("entrypoint", row.get("entrypoint", ""))).strip()
    input_value = contract.get("input", contract.get("args"))
    expected_value = contract.get("expected", contract.get("expected_relation"))
    if (not entrypoint or input_value is None or expected_value is None
            or not _json_safe(input_value) or not _json_safe(expected_value)):
        # A menu item with empty generated placeholders is not an executable
        # recovery route.  Leave the clause open for a real public read/check.
        return {
            "tool": "host_blocked",
            "arguments": {"clause_ref": str(row["clause_ref"]),
                          "status": "UNRUNNABLE",
                          "reason": "public contract is not constructible"},
            "reason": "empty generated witness arguments are not executable recovery",
            "clears": "nothing; acquire a typed public contract first",
        }
    return {
        "tool": "probe",
        "arguments": {
            "mode": "clause_discriminator_propose",
            "schema": SCHEMA,
            "clause_ref": str(row["clause_ref"]),
            "entrypoint": entrypoint,
            "input": copy.deepcopy(input_value),
            "expected_relation": copy.deepcopy(expected_value),
            "source_unit_handle": (derived.get("source_unit_handle")
                                    if isinstance(derived, Mapping) else None),
            "source_revision": (derived.get("source_revision")
                                 if isinstance(derived, Mapping) else None),
            "clauses": copy.deepcopy(list(rows[:1])),
            "max_cases": MAX_CASES_PER_REQUEST,
            "token_contract": {"max_baseline_fraction": TOKEN_FRACTION,
                               "extra_model_calls": 0},
        },
        "reason": "exercise one unexercised public clause through the host witness rail",
        "clears": "only the named clause after a current host receipt passes",
    }


__all__ = [
    "ClauseDiscriminatorError", "EVIDENCE_PIN_SCHEMA", "MAX_CASES_PER_REQUEST",
    "MAX_WITNESS_TURNS_PER_CLAUSE", "SCHEMA",
    "TOKEN_FRACTION", "admit_authored_check_proposal", "clause_rows", "covered_requirement_ids",
    "execute_authored_check_proposal", "execute_proposal",
    "evidence_pin", "next_action", "note_witness_turn", "open_clauses", "open_requirement_ids",
    "settle_witness_execution", "validate_proposal", "witness_route",
]
