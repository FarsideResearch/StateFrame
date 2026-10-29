"""One scoped, typed authority for done offer, dispatch, refusal, and terminalization."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from kernel_boundary_recovery import safe_obligation_ids

MODE_ENV = "STATEFRAME_READINESS_SINGLE_AUTHORITY"
UNCERTIFIED_HOST_DEBT_ENV = "STATEFRAME_UNCERTIFIED_SUBMIT_ON_HOST_DEBT"
DECISION_EVENT = "done_decision"
MAX_SPECIFICS = 16
MAX_ARGUMENT_BYTES = 2048
GREEN_STATUSES = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK"})
FAILED_STATUSES = frozenset({"FAIL", "FAILED", "RED", "ERROR"})
HARD_HOST_VERDICTS = frozenset({"VIOLATED", "CONTRADICTED", "FAILED", "FAIL"})
HARD_DONE_GATE_CODES = frozenset({
    "OPEN_COUNTEREXAMPLE", "GREEN_LOCK_REPAIR_OPEN_COUNTEREXAMPLE",
    "PREPUBLISH_DIFF_BLOCKED", "BEHAVIOR_WITNESS_REQUIRED",
    "DETREASON_SKEPTIC", "V20_TICKET_CLAUSE_UNPROVEN", "V20_GUARD_UNAVAILABLE",
    "COMMIT_FAILED", "PUBLIC_RED_DONE", "REQUIRED_EVIDENCE_GAP",
    "PASS_GATES_VECTOR_UNAVAILABLE", "EFFECT_LIVENESS_BLOCKED",
})
DONE_REASON_CODES = frozenset({
    "GREEN_ALL_OBLIGATIONS_SATISFIED",
    "GREEN_UNCERTIFIED_NO_CALLABLE_RECOVERY",
    "OPEN_OBLIGATIONS_RECOVERABLE",
    "GREEN_GATE_RECOVERABLE",
    "TESTS_RED",
    "CHECK_STALE_AFTER_EDIT",
    "NO_CHECK_RUN",
    "REPEATED_DONE_NO_NEW_FACTS",
    "HOST_RECOVERY_UNAVAILABLE",
    "READINESS_GATE_BLOCKED",
    "NO_CALLABLE_RECOVERY",
    "NO_CHANGED_CANDIDATE",
})
HOST_OWNED_DEBT_CODES = frozenset({
    "CATALOG_UNKNOWN", "MISSING_CHECKS", "UNSTABLE_BINDING",
    "NO_CALLABLE_RECOVERY", "HOST_UNSUPPORTED",
})
DoneOutcome = Literal["ACCEPTED_CERTIFIED", "ACCEPTED_UNCERTIFIED", "REFUSED"]


class DoneReadinessContractError(ValueError):
    """The current done surface cannot satisfy the scoped readiness contract."""


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def mode(environ: Mapping[str, object] | None = None) -> str:
    environment = os.environ if environ is None else environ
    # Default OFF: D2 changes the done flow, so mandatory byte-pinned startup smokes
    # (scripted legacy conversations) cannot pass with it on. It ships switchable and is enabled
    # per arm through the operator env (FIVE five_d2_on), like other behavior-changing mechanisms.
    value = str(environment.get(MODE_ENV, "off")).strip().lower() or "off"
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value


def enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_readiness_single_authority_mode", None)
    if value is None:
        active = mode() == "on"
    else:
        active = str(value).strip().lower() == "on"
    if not active or runtime is None:
        return active
    return (callable(getattr(runtime, "readiness_decision", None))
            and callable(getattr(runtime, "pass_gates_vector", None)))


def uncertified_host_debt_enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Read the host-debt settlement switch without changing D2's own activation mode."""
    environment = os.environ if environ is None else environ
    value = str(environment.get(UNCERTIFIED_HOST_DEBT_ENV, "on")).strip().lower() or "on"
    if value not in {"on", "off"}:
        raise ValueError(f"{UNCERTIFIED_HOST_DEBT_ENV} must be on or off")
    return value == "on"


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", "")).strip()
    return str(tool.get("name", "")).strip()


def _tool_parameters(tool: Mapping[str, object]) -> Mapping[str, object]:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("parameters"), Mapping):
        return function["parameters"]
    parameters = tool.get("parameters")
    return parameters if isinstance(parameters, Mapping) else {}


def _tools(runtime: object, fallback: object = ()) -> list[dict[str, object]]:
    candidates = (
        getattr(runtime, "_completion_owner_full_tools", None),
        getattr(runtime, "_readiness_full_tools", None),
        getattr(runtime, "available_tools", None),
        getattr(runtime, "tools", None),
        fallback,
    )
    for candidate in candidates:
        if (isinstance(candidate, Sequence)
                and not isinstance(candidate, (str, bytes, bytearray))):
            values = [dict(item) for item in candidate if isinstance(item, Mapping)]
            if values:
                return values
    return []


def _scope_identity(runtime: object, tools: Sequence[Mapping[str, object]]) -> str:
    scope: object = None
    scope_fn = getattr(runtime, "_catalog_task_scope", None)
    if callable(scope_fn):
        try:
            scope = scope_fn()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            scope = "unavailable"
    tool_contract = []
    for tool in tools:
        name = _tool_name(tool)
        if name:
            tool_contract.append({
                "name": name,
                "parameters": dict(_tool_parameters(tool)),
            })
    tool_contract.sort(key=lambda item: str(item["name"]))
    return _digest({
        "task_scope": scope,
        "scope_lock": getattr(runtime, "_scope_lock_last_result", None),
        "tool_contract": tool_contract,
    })


def _public_status(readiness: object | None) -> str:
    value = str(getattr(readiness, "public_check_status", "") or "").strip().upper()
    return value.replace("-", "_")


def _check_id(row: Mapping[str, object]) -> str:
    for name in ("blocker_id", "obligation_id", "check_id", "id"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _blocker_rows(vector: object | None, readiness: object | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in getattr(vector, "blockers", ()) or ():
        blocker_id = str(getattr(item, "blocker_id", "") or "").strip()
        if blocker_id:
            resolver = getattr(item, "resolver", None)
            row = {
                "id": blocker_id,
                "resolver": dict(resolver) if isinstance(resolver, Mapping) else None,
            }
            if uncertified_host_debt_enabled():
                code = str(getattr(item, "code", "") or "").strip()
                if code:
                    row["code"] = code
            rows.append(row)
    for item in getattr(readiness, "blockers", ()) or ():
        if not isinstance(item, Mapping):
            continue
        blocker_id = _check_id(item)
        if not blocker_id:
            continue
        resolver = item.get("resolver", item.get("repair_action", item.get("recovery")))
        if not isinstance(resolver, Mapping):
            resolver = getattr(readiness, "repair_action", None)
        row = {
            "id": blocker_id,
            "resolver": dict(resolver) if isinstance(resolver, Mapping) else None,
        }
        if uncertified_host_debt_enabled():
            code = str(item.get("code", item.get("reason_code", "")) or "").strip()
            if code:
                row["code"] = code
        rows.append(row)
    for value in getattr(readiness, "host_blocking", ()) or ():
        if isinstance(value, str) and value.strip():
            row = {"id": value.strip(), "resolver": None}
            if uncertified_host_debt_enabled():
                row["host_owned"] = True
            rows.append(row)
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique.setdefault(row["id"], row)
    return list(unique.values())


def _assessment_value(row: object, name: str) -> object:
    if isinstance(row, Mapping):
        return row.get(name, "")
    return getattr(row, name, "")


def _assessment_rows(runtime: object) -> tuple[dict[str, str], ...]:
    snapshots: list[dict[str, str]] = []
    for attribute in (
            "host_closure_assessments", "behavior_witness_assessments",
            "obligation_assessments", "behavior_witness_records"):
        rows = getattr(runtime, attribute, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            raw_id = next((_assessment_value(row, name) for name in (
                "obligation_id", "check_id", "clause_id", "id", "witness_id")
                if _assessment_value(row, name)), "")
            safe_id = safe_obligation_ids((str(raw_id),), limit=1)
            raw_status = next((_assessment_value(row, name) for name in (
                "verdict", "status", "state", "result")
                if _assessment_value(row, name)), "")
            status = getattr(raw_status, "value", raw_status)
            snapshots.append({
                "source": attribute,
                "id": safe_id[0] if safe_id else "",
                "status": str(status).strip().upper()[:40],
            })
    snapshots.sort(key=lambda item: (item["source"], item["id"], item["status"]))
    return tuple(snapshots)


def _hard_host_blockers(runtime: object) -> tuple[str, ...]:
    blockers: list[str] = []
    for row in _assessment_rows(runtime):
        if row["status"] in HARD_HOST_VERDICTS and row["id"]:
            blockers.append(row["id"])
    return tuple(safe_obligation_ids(blockers, limit=MAX_SPECIFICS))


def _unmappable_task_obligation_ids(runtime: object) -> frozenset[str]:
    """Read source labels stamped by the TASK.md obligation parser."""
    values: list[str] = []
    holders = (runtime, getattr(runtime, "owner", None))
    for holder in holders:
        if holder is None:
            continue
        record = getattr(holder, "work_record", None)
        snapshot = getattr(record, "snapshot", None)
        if callable(snapshot):
            try:
                record = snapshot()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
        if isinstance(record, Mapping) and isinstance(record.get("work_record"), Mapping):
            record = record["work_record"]
        steps = record.get("steps", ()) if isinstance(record, Mapping) else ()
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            fields = step.get("fields")
            if (not isinstance(fields, Mapping)
                    or fields.get("source_disposition") != "advisory_unmappable"
                    or fields.get("provenance") != "HOST_BRIEF"):
                continue
            status = str(step.get("status", "")).lower()
            if status not in {"todo", "doing", "blocked"}:
                continue
            ident = fields.get("obligation_id")
            if isinstance(ident, str):
                values.append(ident)
    return frozenset(safe_obligation_ids(values, limit=MAX_SPECIFICS))


def _candidate_changed_from_parent(runtime: object) -> bool:
    """Require a positive, product-only comparison against the captured task parent."""
    from kernel_done_is_submit_authority import candidate_changed
    return candidate_changed(runtime)


def _product_file_identity(binding: object) -> dict[str, object] | None:
    if not isinstance(binding, Mapping):
        return None
    files = binding.get("product_files")
    if not isinstance(files, Mapping):
        return None
    try:
        from kernel_non_product_paths import is_non_product
    except ImportError:
        return None
    result: dict[str, object] = {}
    for raw_path, digest in files.items():
        path = str(raw_path).replace("\\", "/").strip()
        while path.startswith("./"):
            path = path[2:]
        if path and not path.startswith("/") and ".." not in path.split("/") \
                and not is_non_product(path):
            result[path] = digest
    return result or None


def _public_check_current_for_product(
        runtime: object, current_revision: str, check_revision: str,
) -> bool:
    if current_revision and check_revision and current_revision == check_revision:
        return True
    if not current_revision or not check_revision:
        return False
    try:
        binding_method = getattr(runtime, "binding", None)
        current_binding = binding_method() if callable(binding_method) else None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        current_binding = None
    current_files = _product_file_identity(current_binding)
    if current_files is None:
        return False
    for attribute in ("test_receipts", "host_check_receipts", "host_repair_check_receipts"):
        rows = getattr(runtime, attribute, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            status = str(next((row.get(key) for key in (
                "result_class", "result", "status", "outcome", "verdict",
            ) if row.get(key) not in (None, "")), "")).upper()
            revision = str(row.get("revision", row.get("candidate_revision", "")) or "")
            if status not in GREEN_STATUSES or revision != check_revision:
                continue
            for candidate in (row.get("binding"), row.get("after_binding"), row):
                receipt_files = _product_file_identity(candidate)
                if receipt_files is not None and receipt_files == current_files:
                    return True
    return False


def _host_owned_open_ids(runtime: object) -> frozenset[str]:
    raw_values = getattr(runtime, "_v39_uncertified_submit_ids", ())
    if isinstance(raw_values, str):
        values = [raw_values] if raw_values.strip() else []
    elif isinstance(raw_values, Sequence) and not isinstance(
            raw_values, (bytes, bytearray)):
        values = list(raw_values)
    else:
        values = []
    for attribute in (
        "host_closure_assessments", "behavior_witness_assessments",
        "obligation_assessments", "behavior_witness_records",
    ):
        rows = getattr(runtime, attribute, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("code", row.get("reason_code", row.get("reason", ""))) or "")
            code = code.strip().upper().replace("-", "_")
            if code not in HOST_OWNED_DEBT_CODES:
                continue
            ident = next((
                row.get(name) for name in (
                    "obligation_id", "check_id", "clause_id", "id", "witness_id",
                ) if isinstance(row.get(name), str) and str(row.get(name)).strip()
            ), "")
            if ident:
                values.append(ident)
    return frozenset(safe_obligation_ids(values, limit=MAX_SPECIFICS))


def _host_owned_debt_only(
        rows: Sequence[Mapping[str, object]], open_ids: Sequence[str], readiness: object | None,
        tools: Sequence[Mapping[str, object]], runtime: object,
) -> bool:
    metadata_ids = _unmappable_task_obligation_ids(runtime)
    host_open_ids = _host_owned_open_ids(runtime) | metadata_ids
    debt_rows = [dict(row) for row in rows]
    known = {str(row.get("id", "")) for row in debt_rows}
    debt_rows.extend(
        {
            "id": ident, "resolver": None,
            "host_owned": ident in host_open_ids,
        }
        for ident in open_ids if ident not in known
    )
    if not debt_rows:
        return False
    for row in debt_rows:
        blocker_id = str(row.get("id", "")).strip()
        if not blocker_id:
            return False
        candidates = _candidate_recoveries(
            (row,), readiness, tools, runtime, blocker_id=blocker_id,
        )
        if _callable_recovery(candidates, tools) is not None:
            return False
        code = str(row.get("code", blocker_id)).strip().upper().replace("-", "_")
        host_owned = (
            row.get("host_owned") is True
            or code in HOST_OWNED_DEBT_CODES
            or blocker_id in metadata_ids
            or (isinstance(row.get("resolver"), Mapping)
                and str(row["resolver"].get("tool", "")).strip()
                in {"host_contract", "host_blocked"})
        )
        if not host_owned or code in HARD_DONE_GATE_CODES:
            return False
    return True


def _open_ids(runtime: object) -> tuple[str, ...]:
    try:
        from kernel_completion_owner import open_ids
        values = open_ids(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        values = getattr(runtime, "_v39_uncertified_submit_ids", ())
    return tuple(safe_obligation_ids(values, limit=MAX_SPECIFICS))


def _argument_defaults(tool: Mapping[str, object], runtime: object,
                       blocker_id: str) -> dict[str, object] | None:
    try:
        from kernel_completion_owner import _spec_for_tool
        parameters = _spec_for_tool(tool).schema
    except (ImportError, AttributeError, TypeError, ValueError):
        parameters = _tool_parameters(tool)
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = parameters.get("required", ())
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
        return {}
    arguments: dict[str, object] = {}
    selectors = getattr(runtime, "required_public_selectors", None)
    try:
        first_selector = str((selectors() or ("current",))[0])
    except (AttributeError, IndexError, OSError, RuntimeError, TypeError, ValueError):
        first_selector = "current"
    for raw_name in required:
        name = str(raw_name)
        spec = properties.get(name)
        if not isinstance(spec, Mapping):
            return None
        if "default" in spec:
            arguments[name] = spec["default"]
        elif name in {"check_id", "selector", "target"} and _tool_name(tool) == "run_tests":
            arguments[name] = first_selector if name != "target" else "all"
        elif name == "path":
            target = getattr(runtime, "_target_relative", None)
            try:
                path = target() if callable(target) else getattr(runtime, "target", None)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                path = None
            if not isinstance(path, str) or not path.strip():
                return None
            arguments[name] = path.strip()
        else:
            return None
    return arguments


def _callable_recovery(
        candidates: Sequence[object], tools: Sequence[Mapping[str, object]],
) -> "RecoveryAction | None":
    from kernel_completion_owner import validate_tool_arguments

    by_name = {_tool_name(item): item for item in tools if _tool_name(item)}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("tool", "")).strip()
        tool = by_name.get(name)
        arguments = raw.get("arguments", {})
        if tool is None or name in {"host_contract", "host_blocked"}:
            continue
        if not isinstance(arguments, Mapping):
            continue
        safe_args = dict(arguments)
        try:
            errors = validate_tool_arguments((tool,), name, safe_args)
            encoded = json.dumps(
                safe_args, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        except (AttributeError, TypeError, ValueError):
            continue
        if errors or len(encoded) > MAX_ARGUMENT_BYTES:
            continue
        blocker_id = str(raw.get("blocker_id", "")).strip()
        safe_blocker = safe_obligation_ids((blocker_id,), limit=1)
        if not safe_blocker:
            continue
        return RecoveryAction(
            tool=name, arguments_json=encoded.decode("utf-8"), blocker_id=safe_blocker[0],
        )
    return None


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    tool: str
    arguments_json: str
    blocker_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool.strip() or len(self.tool) > 80:
            raise ValueError("recovery tool must be a bounded tool name")
        blocker = safe_obligation_ids((self.blocker_id,), limit=1)
        if not blocker:
            raise ValueError("recovery must name one sanitized blocker id")
        object.__setattr__(self, "blocker_id", blocker[0])
        try:
            arguments = json.loads(self.arguments_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("recovery arguments must be canonical JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("recovery arguments must be an object")
        canonical = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        if canonical != self.arguments_json:
            raise ValueError("recovery arguments are not canonically encoded")

    @property
    def arguments(self) -> dict[str, object]:
        value = json.loads(self.arguments_json)
        return value if isinstance(value, dict) else {}

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "blocker_id": self.blocker_id,
        }


@dataclass(frozen=True, slots=True)
class DoneReadinessDecision:
    artifact_identity: str
    evidence_identity: str
    scope_identity: str
    decision: DoneOutcome
    reason_code: str
    specifics: tuple[str, ...]
    blocker_ids: tuple[str, ...]
    open_obligation_ids: tuple[str, ...]
    recovery: RecoveryAction | None = None
    public_check_status: str = "UNKNOWN"
    current_green: bool = False

    def __post_init__(self) -> None:
        if self.decision not in {"ACCEPTED_CERTIFIED", "ACCEPTED_UNCERTIFIED", "REFUSED"}:
            raise ValueError("invalid done decision")
        if self.reason_code not in DONE_REASON_CODES:
            raise ValueError("done reason_code is not allowlisted")
        specifics = tuple(safe_obligation_ids(self.specifics, limit=MAX_SPECIFICS))
        blockers = tuple(safe_obligation_ids(self.blocker_ids, limit=MAX_SPECIFICS))
        obligations = tuple(safe_obligation_ids(self.open_obligation_ids, limit=MAX_SPECIFICS))
        object.__setattr__(self, "specifics", specifics)
        object.__setattr__(self, "blocker_ids", blockers)
        object.__setattr__(self, "open_obligation_ids", obligations)
        if self.decision == "REFUSED" and (self.recovery is None or not specifics):
            raise ValueError("a done refusal requires blockers and one callable recovery")
        if self.decision != "REFUSED" and self.recovery is not None:
            raise ValueError("accepted done decisions cannot carry a recovery")
        if self.decision == "ACCEPTED_CERTIFIED" and not specifics:
            raise ValueError("a certified done decision requires satisfying evidence ids")
        if self.decision == "ACCEPTED_UNCERTIFIED" and not specifics:
            raise ValueError("an uncertified done decision requires open blocker ids")
        if self.decision.startswith("ACCEPTED_") and not self.current_green:
            raise ValueError("a done submission requires green checks current to the artifact")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.artifact_identity, self.evidence_identity, self.scope_identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_identity": self.artifact_identity,
            "evidence_identity": self.evidence_identity,
            "scope_identity": self.scope_identity,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "specifics": list(self.specifics),
            "blocker_ids": list(self.blocker_ids),
            "open_obligation_ids": list(self.open_obligation_ids),
            "recovery": self.recovery.to_dict() if self.recovery is not None else None,
            "public_check_status": self.public_check_status,
            "current_green": self.current_green,
        }


def _any_check_receipt(runtime: object) -> bool:
    """True when any check (any outcome) has produced a receipt in this run."""
    for attribute in ("test_receipts", "host_check_receipts", "host_repair_check_receipts"):
        rows = getattr(runtime, attribute, ())
        if (isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray))
                and any(isinstance(row, Mapping) for row in rows)):
            return True
    return False


def _evidence_ids(runtime: object, readiness: object | None) -> tuple[str, ...]:
    values: list[str] = []
    for attribute in ("test_receipts", "host_check_receipts", "host_repair_check_receipts"):
        rows = getattr(runtime, attribute, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            status = str(row.get("result", row.get("status", row.get("outcome", "")))).upper()
            if status not in GREEN_STATUSES:
                continue
            for name in ("receipt_id", "check_id", "id", "event_id"):
                value = row.get(name)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
                    break
    if not values:
        values.extend(str(item) for item in getattr(readiness, "selectors", ()) or ())
    return tuple(safe_obligation_ids(values, limit=MAX_SPECIFICS))


def _readiness_for_runtime(runtime: object, tools: Sequence[Mapping[str, object]]) -> object | None:
    try:
        method = getattr(runtime, "readiness_decision", None)
        return method() if callable(method) else getattr(runtime, "_readiness_decision", None)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return getattr(runtime, "_readiness_decision", None)


def _selectors(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _rerun_reuses_current_pass(candidate: object, runtime: object, current_revision: str) -> bool:
    """True when a run_tests recovery would only reuse an existing current PASS receipt."""
    if not isinstance(candidate, Mapping) or str(candidate.get("tool", "")) != "run_tests":
        return False
    arguments = candidate.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    wanted = (_selectors(arguments.get("target")) | _selectors(arguments.get("selectors"))
              | _selectors(arguments.get("path")))
    for attribute in ("test_receipts", "host_check_receipts", "host_repair_check_receipts"):
        rows = getattr(runtime, attribute, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            status = str(row.get("result", row.get("status", row.get("outcome", "")))).upper()
            revision = str(row.get("revision", row.get("candidate_revision", "")) or "")
            if status not in GREEN_STATUSES or not current_revision or revision != current_revision:
                continue
            have = _selectors(row.get("selectors")) | _selectors(row.get("selector"))
            if not wanted or not have or wanted <= have:
                return True
    return False


def _candidate_recoveries(
        rows: Sequence[Mapping[str, object]], readiness: object | None,
        tools: Sequence[Mapping[str, object]], runtime: object, *, blocker_id: str,
        allow_generic: bool = False,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in rows:
        resolver = row.get("resolver")
        if isinstance(resolver, Mapping):
            candidates.append({**dict(resolver), "blocker_id": row["id"]})
    action = getattr(readiness, "repair_action", None)
    if isinstance(action, Mapping):
        candidates.append({**dict(action), "blocker_id": blocker_id})
    if allow_generic:
        next_action = str(getattr(readiness, "next_recovery_action", "") or "")
        if next_action in {"run_tests", "edit", "str_replace"}:
            candidates.append({
                "tool": next_action,
                "arguments": {},
                "blocker_id": blocker_id,
            })
        for name in ("run_tests", "edit", "str_replace", "read"):
            for tool in tools:
                if _tool_name(tool) != name:
                    continue
                arguments = _argument_defaults(tool, runtime, blocker_id)
                if arguments is not None:
                    candidates.append({"tool": name, "arguments": arguments,
                                       "blocker_id": blocker_id})
    return candidates


def _unchanged_candidate_refusal(
        runtime: object, readiness: object | None,
        rows: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]],
        *, artifact_identity: str, evidence_identity: str, scope_identity: str,
        all_debt: Sequence[str], open_ids: Sequence[str], status: str,
) -> DoneReadinessDecision:
    reason = "NO_CHANGED_CANDIDATE"
    recovery_rows = rows or [{"id": reason, "resolver": None}]
    candidates = _candidate_recoveries(
        recovery_rows, readiness, tools, runtime,
        blocker_id=all_debt[0] if all_debt else reason, allow_generic=True,
    )
    candidates = [
        candidate for candidate in candidates
        if str(candidate.get("tool", "")).strip() in {"edit", "str_replace", "read"}
    ]
    recovery = _callable_recovery(candidates, tools)
    if recovery is None:
        raise DoneReadinessContractError("unchanged candidate has no callable recovery")
    blockers = tuple(dict.fromkeys((*all_debt, reason)))
    return DoneReadinessDecision(
        artifact_identity, evidence_identity, scope_identity, "REFUSED",
        reason, blockers, blockers, open_ids, recovery, status, True,
    )


def _decision(
        runtime: object,
        readiness: object | None,
        vector: object | None,
        tools: Sequence[Mapping[str, object]],
        *,
        artifact_identity: str,
        evidence_identity: str,
        scope_identity: str,
) -> DoneReadinessDecision:
    host_debt_mode = uncertified_host_debt_enabled()
    status = _public_status(readiness)
    status_green = bool(getattr(readiness, "green", False)) and status in GREEN_STATUSES
    rows = _blocker_rows(vector, readiness)
    open_ids = _open_ids(runtime)
    blocker_ids = tuple(safe_obligation_ids(
        [row["id"] for row in rows] + list(open_ids), limit=MAX_SPECIFICS,
    ))
    hard_host_ids = _hard_host_blockers(runtime)
    if hard_host_ids:
        candidates = _candidate_recoveries(
            rows, readiness, tools, runtime, blocker_id=hard_host_ids[0],
            allow_generic=True,
        )
        recovery = _callable_recovery(candidates, tools)
        if recovery is None:
            raise DoneReadinessContractError(
                "hard host completion violation has no callable current recovery")
        blockers = tuple(dict.fromkeys((*hard_host_ids, *blocker_ids)))
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity, "REFUSED",
            "READINESS_GATE_BLOCKED", blockers, blockers, open_ids, recovery,
            _public_status(readiness), False,
        )
    bound = getattr(runtime, "binding", None)
    try:
        binding = bound() if callable(bound) else {}
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        binding = {}
    current_revision = str(
        binding.get("product_revision", "") or ""
        if isinstance(binding, Mapping) else ""
    )
    check_revision = str(getattr(readiness, "revision", "") or "")
    if host_debt_mode:
        revision_current = _public_check_current_for_product(
            runtime, current_revision, check_revision,
        )
    else:
        revision_current = not (current_revision and check_revision
                                and current_revision != check_revision)
    green = (status_green and not bool(getattr(readiness, "no_host_checks", False))
             and revision_current)
    synthetic = ""
    if not green:
        if status in FAILED_STATUSES:
            reason = "TESTS_RED"
            synthetic = "TESTS_RED"
        elif ((not revision_current
               or any(token in status for token in ("STALE", "BLOCKED", "DIRTY", "INVALIDATED")))
              and _any_check_receipt(runtime)):
            # Stale only if a check actually ran; before any check the truthful reason is
            # NO_CHECK_RUN (step 0 showed "stale after edit" with no edit and no check).
            reason = "CHECK_STALE_AFTER_EDIT"
            synthetic = "CHECK_STALE_AFTER_EDIT"
        else:
            reason = "NO_CHECK_RUN"
            synthetic = "NO_CHECK_RUN"
        blockers = tuple(dict.fromkeys((*blocker_ids, synthetic)))
        recovery_rows = rows or [{"id": synthetic, "resolver": None}]
        candidates = _candidate_recoveries(
            recovery_rows, readiness, tools, runtime, blocker_id=synthetic,
            allow_generic=True,
        )
        recovery = _callable_recovery(candidates, tools)
        if recovery is None:
            raise DoneReadinessContractError(
                "refused done has no callable run_tests/edit recovery")
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity, "REFUSED", reason,
            blockers, blockers, open_ids, recovery, status, False,
        )

    all_debt = tuple(dict.fromkeys((*blocker_ids, *open_ids)))
    if host_debt_mode and not _candidate_changed_from_parent(runtime):
        return _unchanged_candidate_refusal(
            runtime, readiness, rows, tools,
            artifact_identity=artifact_identity, evidence_identity=evidence_identity,
            scope_identity=scope_identity, all_debt=all_debt,
            open_ids=open_ids, status=status,
        )

    if not all_debt:
        evidence_ids = _evidence_ids(runtime, readiness)
        if not evidence_ids:
            synthetic = ("CHECK_EVIDENCE_UNAVAILABLE",)
            candidates = _candidate_recoveries(
                ({"id": synthetic[0], "resolver": None},), readiness, tools, runtime,
                blocker_id=synthetic[0], allow_generic=True,
            )
            recovery = _callable_recovery(candidates, tools)
            if recovery is None:
                raise DoneReadinessContractError(
                    "green readiness lacks a receipt and callable run_tests recovery")
            return DoneReadinessDecision(
                artifact_identity, evidence_identity, scope_identity, "REFUSED",
                "NO_CHECK_RUN", synthetic, synthetic, (), recovery, status, False,
            )
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity,
            "ACCEPTED_CERTIFIED", "GREEN_ALL_OBLIGATIONS_SATISFIED",
            evidence_ids, (), (), None, status, True,
        )

    candidates = _candidate_recoveries(rows, readiness, tools, runtime,
                                       blocker_id=all_debt[0])
    # D2NOEFFECT: a recovery that only re-runs an already-current passing check reuses the same
    # receipt and cannot clear anything; offering it again is the b1366 refusal loop.
    candidates = [c for c in candidates
                  if not _rerun_reuses_current_pass(c, runtime, current_revision)]
    recovery = _callable_recovery(candidates, tools)
    if recovery is not None:
        reason = "OPEN_OBLIGATIONS_RECOVERABLE" if open_ids else "GREEN_GATE_RECOVERABLE"
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity, "REFUSED", reason,
            all_debt, all_debt, open_ids, recovery, status, True,
        )
    if host_debt_mode and _host_owned_debt_only(
            rows, open_ids, readiness, tools, runtime):
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity,
            "ACCEPTED_UNCERTIFIED", "GREEN_UNCERTIFIED_NO_CALLABLE_RECOVERY",
            all_debt, all_debt, open_ids, None, status, True,
        )
    if host_debt_mode:
        # Unknown debt is fail-closed and names a current callable tool so the
        # model can resolve the missing host classification or refresh evidence.
        fallback_rows = rows or [{"id": all_debt[0], "resolver": None}]
        fallback_candidates = _candidate_recoveries(
            fallback_rows, readiness, tools, runtime,
            blocker_id=all_debt[0], allow_generic=True,
        )
        fallback_candidates = [
            candidate for candidate in fallback_candidates
            if not _rerun_reuses_current_pass(candidate, runtime, current_revision)
        ]
        fallback = _callable_recovery(fallback_candidates, tools)
        if fallback is None:
            raise DoneReadinessContractError(
                "host debt is not classified and has no callable model recovery")
        reason = "OPEN_OBLIGATIONS_RECOVERABLE" if open_ids else "GREEN_GATE_RECOVERABLE"
        return DoneReadinessDecision(
            artifact_identity, evidence_identity, scope_identity, "REFUSED", reason,
            all_debt, all_debt, open_ids, fallback, status, True,
        )
    return DoneReadinessDecision(
        artifact_identity, evidence_identity, scope_identity,
        "ACCEPTED_UNCERTIFIED", "GREEN_UNCERTIFIED_NO_CALLABLE_RECOVERY",
        all_debt, all_debt, open_ids, None, status, True,
    )


def _host_debt_decision_scoped(
        runtime: object, readiness: object | None, vector: object | None,
        tools: Sequence[Mapping[str, object]],
) -> bool:
    """Enable the shared authority under D2-off only for proven green host debt."""
    if not uncertified_host_debt_enabled():
        return False
    status = _public_status(readiness)
    if (not bool(getattr(readiness, "green", False)) or status not in GREEN_STATUSES
            or bool(getattr(readiness, "no_host_checks", False))):
        return False
    if not _candidate_changed_from_parent(runtime):
        return False
    binding_method = getattr(runtime, "binding", None)
    try:
        binding = binding_method() if callable(binding_method) else {}
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        binding = {}
    current_revision = str(
        binding.get("product_revision", "") or "" if isinstance(binding, Mapping) else ""
    )
    check_revision = str(getattr(readiness, "revision", "") or "")
    if not _public_check_current_for_product(runtime, current_revision, check_revision):
        return False
    if _hard_host_blockers(runtime):
        return False
    rows = _blocker_rows(vector, readiness)
    open_ids = _open_ids(runtime)
    return bool((rows or open_ids)
                and _host_owned_debt_only(rows, open_ids, readiness, tools, runtime))


def decision_for_runtime(
        runtime: object,
        readiness: object | None = None,
        *,
        vector: object | None = None,
        full_tools: object = (),
) -> DoneReadinessDecision | None:
    """Return the cached decision for the exact artifact/evidence/task-tool scope."""
    d2_enabled = enabled(runtime)
    if not d2_enabled and not uncertified_host_debt_enabled():
        return None
    tools = _tools(runtime, full_tools)
    # Decision construction is meaningful only when this request can offer the
    # terminal action.  Intermediate completion probes and narrow request
    # surfaces without a done schema remain under their existing authorities;
    # the final done-capable menu will build the shared decision from its full
    # current tool contract.
    if "done" not in {_tool_name(tool) for tool in tools}:
        return None
    if readiness is None:
        readiness = _readiness_for_runtime(runtime, tools)
    if vector is None:
        factory = getattr(runtime, "pass_gates_vector", None)
        if callable(factory):
            try:
                vector = factory(tools)
            except TypeError:
                vector = factory()
    if not d2_enabled and not _host_debt_decision_scoped(
            runtime, readiness, vector, tools):
        return None
    artifact = str(getattr(vector, "artifact_identity", "") or "")
    if not artifact:
        binding = getattr(runtime, "binding", None)
        try:
            bound = binding() if callable(binding) else {}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            bound = {}
        artifact = str((bound or {}).get("product_revision", "unknown"))
    gate_evidence = str(getattr(vector, "evidence_identity", "") or "unavailable")
    readiness_evidence = str(getattr(readiness, "candidate_digest", "") or "unavailable")
    current_blockers = _blocker_rows(vector, readiness)
    evidence = _digest({
        "gate_evidence": gate_evidence,
        "readiness_evidence": readiness_evidence,
        "public_check_status": _public_status(readiness),
        "check_revision": str(getattr(readiness, "revision", "") or ""),
        "current_blockers": current_blockers,
        "open_obligation_ids": _open_ids(runtime),
        "current_assessments": _assessment_rows(runtime),
    })
    scope = _scope_identity(runtime, tools)
    key = (artifact, evidence, scope)
    step_now = _current_step(runtime)
    if step_now is not None and getattr(runtime, "_done_decision_abandoned_step", _UNSET) == step_now:
        # Abandoned this step: every consumer stays on the legacy path together.
        return None
    if getattr(runtime, "_done_decision_unavailable_key", None) == key:
        return None
    cached = getattr(runtime, "_done_readiness_decision", None)
    if isinstance(cached, DoneReadinessDecision) and cached.key == key:
        return cached
    try:
        decision = _decision(
            runtime, readiness, vector, tools,
            artifact_identity=artifact,
            evidence_identity=evidence,
            scope_identity=scope,
        )
    except DoneReadinessContractError as error:
        # A state with no callable recovery must not crash request
        # preparation (a harness-contract stop is worse than the legacy refusal). Journal a
        # typed, reasoned fallback and let the existing authorities decide this step.
        _record_decision_unavailable(runtime, error, key)
        step = _current_step(runtime)
        if step is not None:
            setattr(runtime, "_done_decision_abandoned_step", step)
        setattr(runtime, "_done_readiness_decision", None)
        return None
    setattr(runtime, "_done_readiness_decision", decision)
    return decision


_UNSET = object()


def _current_step(runtime: object) -> object:
    for owner in (runtime, getattr(runtime, "owner", None)):
        step = getattr(owner, "current_step", None)
        if step is not None:
            return step
    capture = getattr(runtime, "_active_capture", None)
    return getattr(capture, "step", None)


def abandon_decision(runtime: object, decision: "DoneReadinessDecision | None",
                     error: Exception) -> None:
    """Drop the shared decision for EVERY consumer for the rest of this step (one authority)."""
    key = getattr(decision, "key", None)
    if isinstance(key, tuple):
        _record_decision_unavailable(runtime, error, key)
    # Without a known step, abandonment stays per-key (never permanent).
    step = _current_step(runtime)
    if step is not None:
        setattr(runtime, "_done_decision_abandoned_step", step)
    setattr(runtime, "_done_readiness_decision", None)


def _record_decision_unavailable(runtime: object, error: Exception, key: tuple[str, str, str]) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit) or getattr(runtime, "_done_decision_unavailable_key", None) == key:
        return
    setattr(runtime, "_done_decision_unavailable_key", key)
    try:
        from kernel_boundary_recovery import safe_error_detail
        emit(
            "done_decision_unavailable",
            producer="kernel_done_readiness_authority",
            decision="FALLBACK_LEGACY",
            reason_code="NO_CALLABLE_RECOVERY",
            specifics=[safe_error_detail(error)],
            recovery=None,
            dispatch=False,
        )
    except Exception:  # journaling the fallback is advisory
        pass


def cached_decision(runtime: object) -> DoneReadinessDecision | None:
    value = getattr(runtime, "_done_readiness_decision", None)
    return value if isinstance(value, DoneReadinessDecision) else None


def attach_readiness_decision(
        runtime: object,
        readiness: object,
        *,
        vector: object,
        full_tools: object,
) -> object:
    """Attach the exact shared done decision to the immutable readiness snapshot."""
    decision = decision_for_runtime(
        runtime, readiness=readiness, vector=vector, full_tools=full_tools,
    )
    if decision is None:
        return readiness
    try:
        attached = replace(readiness, done_authority=decision)
    except (TypeError, ValueError):
        attached = readiness
    if attached is readiness:
        return readiness
    request_snapshot = getattr(runtime, "_readiness_request_decision", None)
    if request_snapshot is readiness:
        setattr(runtime, "_readiness_request_decision", attached)
    setattr(runtime, "_readiness_decision", attached)
    return attached


MENU_PROJECTION_ENV = "STATEFRAME_READINESS_MENU_PROJECTION"


def _menu_projection_disabled() -> bool:
    """Menu projection is off unless explicitly re-enabled for an experiment (default off)."""
    return os.environ.get(MENU_PROJECTION_ENV, "off").strip().lower() != "on"


def project_menu(
        decision: DoneReadinessDecision,
        selected: Sequence[Mapping[str, object]],
        full_tools: Sequence[Mapping[str, object]],
        *,
        capability_admissible=None,
        offer_done: bool = False,
) -> list[dict[str, object]]:
    menu = [dict(tool) for tool in selected if isinstance(tool, Mapping)]
    if _menu_projection_disabled():
        if not uncertified_host_debt_enabled():
            return menu
        available = {_tool_name(tool): tool for tool in full_tools if _tool_name(tool)}
        names = {_tool_name(tool) for tool in menu}
        if decision.decision == "REFUSED":
            menu = [tool for tool in menu if _tool_name(tool) != "done"]
            recovery = decision.recovery
            if recovery is None:
                raise DoneReadinessContractError("refusal lost its recovery")
            if recovery.tool not in names:
                tool = available.get(recovery.tool)
                if (tool is None or capability_admissible is None
                        or not capability_admissible(recovery.tool)):
                    raise DoneReadinessContractError(
                        "refusal recovery is not on the dispatchable capability floor")
                menu.append(dict(tool))
            return menu
        if (decision.decision == "ACCEPTED_UNCERTIFIED" and offer_done
                and "done" not in names):
            tool = available.get("done")
            if (tool is None or capability_admissible is None
                    or not capability_admissible("done")):
                return menu
            menu.append(dict(tool))
        return menu
    available = {_tool_name(tool): tool for tool in full_tools if _tool_name(tool)}
    names = {_tool_name(tool) for tool in menu}
    if decision.decision == "REFUSED":
        menu = [tool for tool in menu if _tool_name(tool) != "done"]
        recovery = decision.recovery
        if recovery is None:
            raise DoneReadinessContractError("refusal lost its recovery")
        tool = available.get(recovery.tool)
        if tool is None:
            raise DoneReadinessContractError("refusal recovery is absent from the full surface")
        if capability_admissible is not None and not capability_admissible(recovery.tool):
            raise DoneReadinessContractError("refusal recovery is not dispatch-admissible")
        arguments_error = _callable_recovery(
            (recovery.to_dict() | {"blocker_id": recovery.blocker_id},), full_tools,
        )
        if arguments_error is None:
            raise DoneReadinessContractError("refusal recovery arguments are no longer valid")
        if recovery.tool not in {_tool_name(item) for item in menu}:
            if (not uncertified_host_debt_enabled() or capability_admissible is None
                    or not capability_admissible(recovery.tool)):
                raise DoneReadinessContractError("refusal recovery is not on the current menu")
            menu.append(dict(tool))
        return menu
    if offer_done and "done" not in names:
        if capability_admissible is not None and not capability_admissible("done"):
            return menu
        done_tool = available.get("done")
        if done_tool is None:
            raise DoneReadinessContractError("accepted done decision has no done schema")
        menu.append(dict(done_tool))
    return menu


def host_debt_disposition_text(decision: object) -> str:
    """Render the shared D2 host-debt outcome in one model-facing sentence."""
    if getattr(decision, "decision", None) != "ACCEPTED_UNCERTIFIED":
        return ""
    values: list[object] = []
    for name in ("blocker_ids", "open_obligation_ids", "specifics"):
        raw = getattr(decision, name, ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(raw)
    ids = safe_obligation_ids(values, limit=MAX_SPECIFICS)
    specifics = ", ".join(ids) or "host-owned items"
    return (
        f"Host-owned items [{specifics}] are recorded on submit and are non-blocking for submission; "
        "they remain unresolved. Store as UNCERTIFIED_SUBMIT with certified=false."
    )


def _decision_obligation_ids(decision: object) -> tuple[str, ...]:
    values: list[object] = []
    for name in ("blocker_ids", "open_obligation_ids", "specifics"):
        raw = getattr(decision, name, ())
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            values.extend(raw)
    return tuple(safe_obligation_ids(values, limit=MAX_SPECIFICS))


def project_f4_submission_state(projection: object, decision: object) -> object:
    """Project cached D2 permission and host-debt status into the F4 model view."""
    notice = host_debt_disposition_text(decision)
    if not notice or not isinstance(projection, Mapping):
        return projection
    projected = copy.deepcopy(dict(projection))
    frame_state = projected.get("state")
    if not isinstance(frame_state, Mapping):
        return projection
    frame_state = copy.deepcopy(dict(frame_state))
    ids = set(_decision_obligation_ids(decision))
    rows = frame_state.get("blockers")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        projected_rows: list[object] = []
        for entry in rows:
            if not isinstance(entry, Mapping):
                projected_rows.append(entry)
                continue
            row = entry.get("value")
            if not isinstance(row, Mapping):
                projected_rows.append(entry)
                continue
            row = dict(row)
            ident = _check_id(row)
            code = str(row.get("code", row.get("reason_code", "")) or "")
            if (ident not in ids
                    and code.strip().upper().replace("-", "_") not in HOST_OWNED_DEBT_CODES):
                projected_rows.append(entry)
                continue
            reason = next((row.get(key) for key in (
                "host_reason_code", "code", "reason_code", "reason",
            ) if row.get(key) not in (None, "")), None)
            if reason is not None:
                row["host_reason_code"] = str(reason)[:80]
            for key in ("code", "reason_code", "reason"):
                row.pop(key, None)
            for key in ("status", "state", "verdict", "blocker_status"):
                if key in row:
                    row[key] = "HOST_OWNED_RECORDED_ON_SUBMIT"
            for key in ("blocking", "blocks_submission", "host_blocking"):
                if key in row:
                    row[key] = False
            row["host_owned"] = True
            row["status"] = "HOST_OWNED_RECORDED_ON_SUBMIT"
            row["unresolved"] = True
            row["submission_blocking"] = False
            row["submission_disposition"] = "UNCERTIFIED_SUBMIT"
            for key in ("detail", "message", "description"):
                if key in row:
                    row[key] = (
                        "Host-owned item is recorded on submit; it remains unresolved and "
                        "non-blocking for submission."
                    )
            projected_rows.append({**dict(entry), "value": row})
        frame_state["blockers"] = projected_rows
    permission = frame_state.get("done_permission")
    if isinstance(permission, Mapping):
        frame_state["done_permission"] = {
            **dict(permission),
            "state": "ALLOW",
            "authority": "shared_done_readiness",
            "decision": "ACCEPTED_UNCERTIFIED",
        }
    submission = frame_state.get("submission")
    if isinstance(submission, Mapping):
        frame_state["submission"] = {
            **dict(submission),
            "state": "UNCERTIFIED_SUBMIT",
            "authority": "shared_done_readiness",
            "decision": "ACCEPTED_UNCERTIFIED",
            "route": "UNCERTIFIED_SUBMIT",
            "host_owned_item_ids": sorted(ids),
            "certified": False,
        }
    frame_state["submission_disposition"] = {
        "decision": "ACCEPTED_UNCERTIFIED",
        "route": "UNCERTIFIED_SUBMIT",
        "host_owned_item_ids": sorted(ids),
        "status": "HOST_OWNED_RECORDED_ON_SUBMIT",
        "submission_blocking": False,
        "certified": False,
    }
    projected["state"] = frame_state
    claims = projected.get("claims")
    if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes, bytearray)):
        updated_claims: list[object] = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                updated_claims.append(claim)
                continue
            predicate = str(claim.get("predicate", ""))
            if predicate == "done_permission":
                updated_claims.append({
                    **dict(claim), "state": "ALLOW",
                    "authority": "shared_done_readiness",
                })
            elif predicate == "submission":
                updated_claims.append({
                    **dict(claim), "state": "UNCERTIFIED_SUBMIT",
                    "authority": "shared_done_readiness",
                    "host_owned_item_ids": sorted(ids), "certified": False,
                })
            else:
                updated_claims.append(claim)
        projected["claims"] = updated_claims
    return projected


def render_done_reason(decision: DoneReadinessDecision) -> str:
    recovery = ""
    if decision.recovery is not None:
        encoded = json.dumps(
            decision.recovery.arguments, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        recovery = f"; recovery={decision.recovery.tool}{encoded}"
    specifics = ",".join(decision.specifics) or "none"
    if decision.decision == "ACCEPTED_UNCERTIFIED":
        return f"done {decision.decision} [{decision.reason_code}]; {host_debt_disposition_text(decision)}"
    return (f"done {decision.decision} [{decision.reason_code}]; "
            f"specifics={specifics}{recovery}")


def assert_done_reason_rendered(result: Mapping[str, object]) -> None:
    reason = result.get("done_reason")
    line = result.get("done_reason_line")
    if not isinstance(reason, Mapping) or not isinstance(line, str) or not line.strip():
        raise AssertionError("model-visible done result omitted its typed reason")
    code = reason.get("reason_code")
    if not isinstance(code, str) or code not in line:
        raise AssertionError("model-visible done result omitted its reason code")
    if reason.get("decision") == "REFUSED":
        recovery = reason.get("recovery")
        if not isinstance(recovery, Mapping) or str(recovery.get("tool", "")) not in line:
            raise AssertionError("model-visible refusal omitted its recovery")
    if reason.get("decision") == "ACCEPTED_UNCERTIFIED":
        lowered = line.lower()
        if "non-blocking for submission" not in lowered:
            raise AssertionError("model-visible host debt was not marked non-blocking")
        if "host-owned" not in lowered or "recorded on submit" not in lowered:
            raise AssertionError("model-visible host debt omitted its submit disposition")
        contradictory = lowered.replace("non-blocking for submission", "")
        if any(phrase in contradictory for phrase in (
                "blocking for submission", "must resolve before submission",
                "cannot submit", "blocks submission")):
            raise AssertionError("model-visible host debt contradicts submit readiness")


def record_done_decision(runtime: object, decision: DoneReadinessDecision) -> None:
    history = getattr(runtime, "_done_decision_event_keys", None)
    if not isinstance(history, set):
        history = set()
        setattr(runtime, "_done_decision_event_keys", history)
    event_key = (decision.key, decision.decision, decision.reason_code, decision.specifics)
    if event_key in history:
        return
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        recovery = decision.recovery.to_dict() if decision.recovery is not None else None
        emit(
            DECISION_EVENT,
            producer="kernel_done_readiness_authority",
            decision=decision.decision,
            reason_code=decision.reason_code,
            specifics=list(decision.specifics),
            recovery=recovery,
            artifact_identity=decision.artifact_identity,
            evidence_identity=decision.evidence_identity,
            scope_identity=decision.scope_identity,
            dispatch=False,
        )
    history.add(event_key)
    if len(history) > 128:
        history.pop()


def uncertified_gate_compatible(
        runtime: object, decision: DoneReadinessDecision | None, *,
        reason_codes: Sequence[object], open_work_ids: Sequence[object] = (),
        unresolved_gating_items: Sequence[object] = (),
        obligation_assessments: Sequence[object] = (),
        gate_vector: object | None = None,
) -> bool:
    """Allow the bounded path only when ordinary gates report this exact soft debt."""
    if decision is None or decision.decision != "ACCEPTED_UNCERTIFIED":
        return False
    if _hard_host_blockers(runtime):
        return False
    debt = set((*decision.blocker_ids, *decision.open_obligation_ids))
    active_ids = set(safe_obligation_ids(
        (*open_work_ids, *unresolved_gating_items), limit=MAX_SPECIFICS,
    ))
    assessment_rows: list[dict[str, str]] = []
    for item in obligation_assessments:
        raw_id = next((_assessment_value(item, name) for name in (
            "obligation_id", "check_id", "clause_id", "id", "witness_id")
            if _assessment_value(item, name)), "")
        safe_id = safe_obligation_ids((str(raw_id),), limit=1)
        raw_status = next((_assessment_value(item, name) for name in (
            "verdict", "status", "state", "result")
            if _assessment_value(item, name)), "")
        status = getattr(raw_status, "value", raw_status)
        assessment_rows.append({
            "id": safe_id[0] if safe_id else "",
            "status": str(status).strip().upper()[:40],
        })
    if any(row["status"] in HARD_HOST_VERDICTS for row in assessment_rows):
        return False
    active_ids.update(row["id"] for row in assessment_rows if row["id"])
    if active_ids - debt:
        return False

    allowed_codes: set[str] = set()
    if gate_vector is None:
        try:
            factory = getattr(runtime, "pass_gates_vector", None)
            gate_vector = factory() if callable(factory) else None
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
    for blocker in getattr(gate_vector, "blockers", ()) or ():
        code = str(getattr(blocker, "code", "") or "").strip()
        if code:
            allowed_codes.add(code)
    readiness = _readiness_for_runtime(runtime, _tools(runtime))
    for blocker in getattr(readiness, "blockers", ()) or ():
        if isinstance(blocker, Mapping):
            code = str(blocker.get("code", blocker.get("reason_code", "")) or "").strip()
            if code:
                allowed_codes.add(code)
    codes = {str(code).strip() for code in reason_codes if str(code).strip()}
    if codes & HARD_DONE_GATE_CODES:
        return False
    generic_debt_codes = {
        "NEEDS_WORK", "OPEN_WORK", "HOST_OBLIGATION", "CLAUSE_COVERAGE",
        # Host done-facts reports the unresolved IDs as one aggregate reason.
        # The IDs still have to be present and covered by this exact DRA debt set.
        "UNRESOLVED_GATING_ITEMS",
    }
    if codes - allowed_codes - generic_debt_codes:
        return False
    if codes & generic_debt_codes and not active_ids:
        return False
    return bool(debt)


def finalize_uncertified_completion(
        runtime: object, completion_decision: object,
        readiness_decision: DoneReadinessDecision | None,
) -> object:
    """Project post-gate soft debt unless the immutable submission seal exists."""
    from kernel_done_is_submit_authority import done_sealed

    if done_sealed(runtime) is not None:
        return completion_decision
    reason_codes = tuple(getattr(completion_decision, "reason_codes", ()) or ())
    if (readiness_decision is None
            or readiness_decision.decision != "ACCEPTED_UNCERTIFIED"
            or "GREEN_UNCERTIFIED_NO_CALLABLE_RECOVERY" not in reason_codes
            or bool(getattr(completion_decision, "accepted", False))):
        return completion_decision

    from kernel_completion_owner import _begin_uncertified_submit
    from kernel_verdict import BOUNDED, CompletionDecision

    _begin_uncertified_submit(runtime)
    obligation_ids = tuple(dict.fromkeys((
        *readiness_decision.open_obligation_ids,
        *readiness_decision.blocker_ids,
    )))
    open_ids = obligation_ids
    line = render_done_reason(readiness_decision)
    runtime._v39_uncertified_submit_ids = obligation_ids
    outcome = {
        "result": "UNCERTIFIED_SUBMIT",
        "outcome": "UNCERTIFIED_SUBMIT",
        "accepted": False,
        "certified": False,
        "uncertified": True,
        "open_ids": list(open_ids),
        "open_ids_status": "known",
        "open_requirement_ids": list(obligation_ids),
        "unresolved_gating_items": list(dict.fromkeys((
            *readiness_decision.blocker_ids, *open_ids,
        ))),
        "stop_reason": "UNCERTIFIED_SUBMIT",
        "stop_classification": "UNCERTIFIED_SUBMIT",
        "system_outcome": "UNCERTIFIED_SUBMIT",
        "completion_owner_reason": "green_uncertified_no_callable_recovery",
        "done_reason": readiness_decision.to_dict(),
        "done_reason_line": line,
    }
    runtime._completion_owner_uncertified_submit_result = outcome
    record_done_decision(runtime, readiness_decision)
    result = CompletionDecision(
        BOUNDED,
        completion_decision.basis,
        accepted=False,
        reason_codes=(readiness_decision.reason_code,),
        detail=line,
        needs_work_ids=open_ids,
        done_reason=readiness_decision.to_dict(),
        done_reason_line=line,
    )
    runtime._last_completion_decision = result
    assert_certification_truth(outcome)
    return result


def apply_terminal_result(
        runtime: object, result: Mapping[str, object],
        decision: DoneReadinessDecision | None = None,
) -> dict[str, object]:
    from kernel_done_is_submit_authority import terminal_from_seal

    sealed = terminal_from_seal(runtime)
    if sealed is not None:
        return sealed
    terminal = dict(result)
    if decision is None:
        decision = cached_decision(runtime)
    if decision is None:
        return terminal
    terminal["done_reason"] = decision.to_dict()
    terminal["done_reason_line"] = render_done_reason(decision)
    if decision.decision == "REFUSED":
        record_done_decision(runtime, decision)
        terminal.update({
            "result": "REFUSED",
            "outcome": "REFUSED",
            "stop_reason": decision.reason_code,
            "stop_classification": "REFUSED",
            "system_outcome": "REFUSED",
            "accepted": False,
            "certified": False,
            "uncertified": False,
        })
        assert_certification_truth(terminal)
        return terminal
    if decision.decision == "ACCEPTED_UNCERTIFIED":
        record_done_decision(runtime, decision)
        obligation_ids = list(dict.fromkeys((
            *decision.open_obligation_ids, *decision.blocker_ids,
        )))
        open_ids = obligation_ids
        terminal.update({
            "result": "UNCERTIFIED_SUBMIT",
            "outcome": "UNCERTIFIED_SUBMIT",
            "stop_reason": "UNCERTIFIED_SUBMIT",
            "stop_classification": "UNCERTIFIED_SUBMIT",
            "system_outcome": "UNCERTIFIED_SUBMIT",
            "accepted": False,
            "certified": False,
            "uncertified": True,
            "open_ids": open_ids,
            "open_ids_status": "known",
            "open_requirement_ids": obligation_ids,
            "unresolved_gating_items": tuple(dict.fromkeys((
                *(str(item) for item in terminal.get("unresolved_gating_items", ())
                  if isinstance(item, str)),
                *open_ids,
                *decision.blocker_ids,
            ))),
        })
        terminal.pop("accepted_correct", None)
        terminal.pop("certified_correct", None)
        for name in ("done_facts", "grading_inputs"):
            facts = terminal.get(name)
            if isinstance(facts, Mapping):
                changed = dict(facts)
                changed.update({
                    "accepted": False,
                    "certified": False,
                    "uncertified": True,
                    "open_requirement_ids": obligation_ids,
                    "unresolved_gating_items": list(terminal["unresolved_gating_items"]),
                })
                changed.pop("accepted_correct", None)
                changed.pop("certified_correct", None)
                terminal[name] = changed
        setattr(runtime, "_completion_owner_uncertified_submit", True)
        setattr(runtime, "_completion_owner_uncertified_submit_used", True)
        setattr(runtime, "_completion_owner_uncertified_submit_result", {
            key: terminal[key] for key in (
                "result", "outcome", "accepted", "certified", "uncertified",
                "open_ids", "open_ids_status", "open_requirement_ids",
                "unresolved_gating_items", "done_reason", "done_reason_line",
                "stop_reason", "system_outcome",
            ) if key in terminal
        })
        assert_certification_truth(terminal)
    elif decision.decision == "ACCEPTED_CERTIFIED":
        if terminal.get("accepted") is not True or terminal.get("certified") is not True:
            raise DoneReadinessContractError(
                "terminal completion rejected the shared certified done decision")
        record_done_decision(runtime, decision)
    return terminal


def assert_certification_truth(result: Mapping[str, object]) -> None:
    if str(result.get("result", result.get("outcome", ""))).upper() != "UNCERTIFIED_SUBMIT":
        return
    if (result.get("accepted") is True
            or result.get("certified") in {True, "CERTIFIED"}
            or result.get("accepted_correct") is True
            or result.get("certified_correct") is True):
        raise AssertionError("uncertified submission claims accepted or certified correctness")
    for name in ("stop_reason", "stop_classification", "system_outcome"):
        if str(result.get(name, "")).upper() in {"DONE", "DONE_ACCEPTED", "CERTIFIED", "CERTIFIED_CORRECT"}:
            raise AssertionError(f"uncertified submission has a certified {name}")
    for name in ("done_facts", "grading_inputs", "terminal_disposition"):
        value = result.get(name)
        if not isinstance(value, Mapping):
            continue
        certified = value.get("certified")
        if (value.get("accepted") is True
                or certified is True or certified == "CERTIFIED"
                or value.get("acceptance") == "ACCEPTED"
                or value.get("certification") == "CERTIFIED"):
            raise AssertionError(f"uncertified submission is certified in {name}")
        if value.get("accepted_correct") is True or value.get("certified_correct") is True:
            raise AssertionError(f"uncertified submission is correct in {name}")


def assert_refusal_recovery(
        decision: DoneReadinessDecision,
        menu: Sequence[Mapping[str, object]],
        full_tools: Sequence[Mapping[str, object]],
) -> None:
    if decision.decision != "REFUSED":
        return
    recovery = decision.recovery
    if recovery is None or recovery.tool not in {_tool_name(item) for item in menu}:
        raise AssertionError("refusal does not name a callable tool on the current menu")
    if _callable_recovery((recovery.to_dict() | {"blocker_id": recovery.blocker_id},
                           ), full_tools) is None:
        raise AssertionError("refusal recovery arguments are invalid")


def assert_same_decision(menu: object, dispatch: object) -> None:
    if menu is not dispatch:
        raise AssertionError("menu and dispatch consumed different readiness decisions")


def validate_reason_ast(source: str) -> tuple[str, ...]:
    """Find empty or unknown literal reason codes in structural guard fixtures."""
    import ast

    errors: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        values: list[ast.expr] = []
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "reason_code":
                    values.append(value)
        if isinstance(node, ast.Call):
            values.extend(
                keyword.value for keyword in node.keywords if keyword.arg == "reason_code"
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "reason_code"
                    for target in targets
            ):
                assigned = node.value
                if assigned is not None:
                    values.append(assigned)
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value not in DONE_REASON_CODES:
                    errors.append(f"invalid reason_code: {value.value!r}")
    return tuple(errors)


DONE_DECISION_SITE_REGISTRY = (
    ("kernel_stateframe_runtime.py", "readiness_decision", "attach_readiness_decision"),
    ("kernel_stateframe_runtime.py", "_readiness_menu", "project_menu"),
    ("kernel_stateframe_runtime.py", "completion_decision", "decision_for_runtime"),
    ("kernel_greenlock_menu_authority.py", "dispatch_action", "decision_for_runtime"),
    ("kernel_greenlock_menu_authority.py", "finalize_wire_menu_authority", "project_menu"),
    ("kernel_completion_owner.py", "terminal_tool_menu", "project_menu"),
    ("kernel_completion_owner.py", "completion_route_decision", "cached_decision"),
    ("kernel_completion_owner.py", "_completion_route_surface_error", "cached_decision"),
    ("kernel_submit_at_readiness.py", "at_request_boundary", "cached_decision"),
    ("kernel_tokens_row6_runtime.py", "observe_done", "cached_decision"),
    ("kernel_completion_owner.py", "terminal_result", "apply_terminal_result"),
    ("kernel_terminal_disposition.py", "mint", "cached_decision"),
    ("kernel_wire_truth_facts.py", "_context_facts", "cached_decision"),
)


def structural_site_violations(source_by_file: Mapping[str, str]) -> tuple[str, ...]:
    """Validate registered done sites all flow through the typed shared authority."""
    import ast

    errors: list[str] = []
    for filename, function_name, required_call in DONE_DECISION_SITE_REGISTRY:
        source = source_by_file.get(filename)
        if not isinstance(source, str):
            errors.append(f"missing done decision source: {filename}")
            continue
        tree = ast.parse(source, filename=filename)
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == function_name]
        if len(functions) != 1:
            errors.append(f"missing or ambiguous done decision site: {filename}:{function_name}")
            continue
        calls = {
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            for node in ast.walk(functions[0]) if isinstance(node, ast.Call)
        }
        if required_call not in calls:
            errors.append(f"{filename}:{function_name} bypasses {required_call}")
    return tuple(errors)


__all__ = [
    "DECISION_EVENT", "DONE_DECISION_SITE_REGISTRY", "DONE_REASON_CODES",
    "DoneReadinessContractError", "DoneReadinessDecision", "HARD_DONE_GATE_CODES",
    "HOST_OWNED_DEBT_CODES", "MODE_ENV", "RecoveryAction",
    "UNCERTIFIED_HOST_DEBT_ENV", "apply_terminal_result", "assert_certification_truth",
    "assert_done_reason_rendered", "assert_refusal_recovery", "assert_same_decision", "cached_decision",
    "attach_readiness_decision", "decision_for_runtime", "enabled", "mode", "project_menu",
    "uncertified_host_debt_enabled",
    "finalize_uncertified_completion", "host_debt_disposition_text", "project_f4_submission_state",
    "record_done_decision", "render_done_reason",
    "structural_site_violations", "uncertified_gate_compatible", "validate_reason_ast",
]
