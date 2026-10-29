"""Generic repair policy for a public-green StateFrame lock.

The existing green lock remains the compatibility boundary.  This module is
the opt-in policy layer used when a host has a current public-green receipt but
also has a typed, executable obligation counterexample.  It deliberately
works from check/obligation data rather than task prose or fixed paths.
"""
from __future__ import annotations

import copy
import ast
import dataclasses
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


MODE_ENV = "STATEFRAME_GREEN_LOCK_REPAIR"
ADVISORY = "ADVISORY"
UNBOUND_ENTRYPOINT = "UNBOUND_ENTRYPOINT"
UNBOUND_INVOCATION = "UNBOUND_INVOCATION"
CONTRADICTS_PUBLIC_TEST = "CONTRADICTS_PUBLIC_TEST"
UNSATISFIABLE_CONTRACT_CELL = "UNSATISFIABLE_CONTRACT_CELL"
PUBLIC_REGRESSION = "PUBLIC_REGRESSION"
PUBLIC_CHECK_UNAVAILABLE = "PUBLIC_CHECK_UNAVAILABLE"

ADVISORY_REASONS = frozenset({
    UNBOUND_ENTRYPOINT,
    UNBOUND_INVOCATION,
    CONTRADICTS_PUBLIC_TEST,
    UNSATISFIABLE_CONTRACT_CELL,
    "ADVISORY_ONLY",
})

_EDIT_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_READ_ACTIONS = frozenset({
    "read", "probe", "run_tests", "test", "pytest", "work",
})
_DESTRUCTIVE_ACTIONS = frozenset({
    "delete", "remove", "rename", "shell", "command", "exec",
})
_PATH_KEYS = frozenset({
    "path", "target_path", "file", "file_path", "source_path", "target_file",
    "target_files", "files", "mentioned_files", "paths", "targets", "location_hint",
})
_SYMBOL_KEYS = frozenset({
    "symbol", "target_symbol", "function", "function_name", "qualname",
    "entry_point", "entrypoint", "module",
})
_ID_KEYS = frozenset({
    "obligation_id", "check_id", "selector", "selectors", "selection", "target",
    "test", "check", "obligation", "instance_key",
})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            result = dataclasses.asdict(value)
        except (TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    return {}


def _norm_path(value: object) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def _path_under_root(value: object, root: object) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    try:
        candidate = Path(raw)
        base = Path(root).resolve()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve().relative_to(base).as_posix()
    except (OSError, TypeError, ValueError):
        return _norm_path(raw)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reason_text(value: object) -> str:
    return str(value or "").upper().replace("-", "_").replace(" ", "_")


def _walk(value: object, *, depth: int = 0):
    if depth > 10:
        return
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child, depth=depth + 1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk(child, depth=depth + 1)


def _contains_flag(value: object, names: Sequence[str]) -> bool:
    wanted = {str(name).lower() for name in names}
    for item in _walk(value):
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).lower() in wanted and child is True:
                    return True
    return False


def _entry_point_state(root: object, entry_point: object) -> str:
    if root is None:
        return "UNKNOWN"
    try:
        from kernel_obligation_assess import _entry_point_state as state
        return state(Path(root).resolve(), entry_point if isinstance(entry_point, str) else None)
    except (ImportError, OSError, TypeError, ValueError):
        return "UNKNOWN"


def _check_reason(check: object, *, starter_root: object = None,
                  public_expectations: object = None) -> str | None:
    basis = _reason_text(getattr(check, "oracle_basis", ""))
    provenance = getattr(check, "provenance", {})
    if basis == ADVISORY:
        policy = provenance.get("green_lock_repair") if isinstance(provenance, Mapping) else None
        reason = _reason_text(policy.get("reason")) if isinstance(policy, Mapping) else "ADVISORY_ONLY"
        return reason if reason in ADVISORY_REASONS else "ADVISORY_ONLY"

    entry_point = getattr(check, "entry_point", None)
    if not isinstance(entry_point, str) or not entry_point.strip() or ":" not in entry_point:
        return UNBOUND_ENTRYPOINT
    if starter_root is not None:
        state = _entry_point_state(starter_root, entry_point)
        if state == "MISSING":
            return UNBOUND_ENTRYPOINT
        if state == "UNRUNNABLE":
            return UNBOUND_ENTRYPOINT

    cell = getattr(check, "contract_cell", None)
    cell_map = _mapping(cell)
    predicate = cell_map.get("predicate", {}) if cell_map else getattr(cell, "predicate", {})
    if _contains_flag(predicate, (
            "contradicts_public_expectation", "contradicts_public_test",
            "public_test_contradiction", "contradiction")):
        return CONTRADICTS_PUBLIC_TEST
    if _contains_flag(predicate, (
            "unsatisfiable", "unsatisfiable_contract", "impossible",
            "self_contradictory")):
        return UNSATISFIABLE_CONTRACT_CELL
    admission = _reason_text(cell_map.get("admission", "")) if cell_map else _reason_text(
        getattr(cell, "admission", ""))
    applicability = _reason_text(cell_map.get("applicability", "")) if cell_map else _reason_text(
        getattr(cell, "applicability", ""))
    if admission == "CONFLICTED":
        return UNSATISFIABLE_CONTRACT_CELL
    if applicability in {"DOES_NOT_APPLY", "UNKNOWN", "APPLICABILITY_UNKNOWN"}:
        return UNSATISFIABLE_CONTRACT_CELL

    if isinstance(provenance, Mapping):
        if _contains_flag(provenance, (
                "contradicts_public_expectation", "contradicts_public_test",
                "public_test_contradiction")):
            return CONTRADICTS_PUBLIC_TEST
        if _contains_flag(provenance, (
                "unsatisfiable", "unsatisfiable_contract", "self_contradictory")):
            return UNSATISFIABLE_CONTRACT_CELL

    # Callers may supply a public expectation table.  Comparing only an
    # explicitly keyed expectation keeps this a source-grounded check rather
    # than guessing from arbitrary prose.
    if isinstance(public_expectations, Mapping):
        candidates = []
        for key in (getattr(check, "check_id", ""), entry_point):
            if key and key in public_expectations:
                candidates.append(public_expectations[key])
        for candidate in candidates:
            rows = candidate if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)) else (candidate,)
            for row in rows:
                row_map = row if isinstance(row, Mapping) else {"expected": row}
                fixture = row_map.get("fixture")
                if fixture is not None and fixture != getattr(check, "fixture", None):
                    continue
                expected = row_map.get("expected")
                if expected is not None and expected != getattr(check, "expected", None):
                    return CONTRADICTS_PUBLIC_TEST
    return None


def public_expectations(tests: object) -> dict[str, list[dict[str, object]]]:
    """Extract explicitly asserted call expectations from public test source."""
    if isinstance(tests, Mapping):
        sources = tests.values()
    elif isinstance(tests, Sequence) and not isinstance(tests, (str, bytes, bytearray)):
        sources = (item.get("source", "") if isinstance(item, Mapping) else item
                   for item in tests)
    else:
        sources = ()
    result: dict[str, list[dict[str, object]]] = {}
    for source in sources:
        if not isinstance(source, str):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        aliases: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}:{item.name}"
            elif isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split(".")[0]] = item.name + ":__module__"

        def call_node(node: ast.AST) -> ast.Call | None:
            if isinstance(node, ast.Call):
                return node
            if isinstance(node, (ast.Subscript, ast.Attribute)):
                return call_node(node.value)
            return None

        def literal(node: ast.AST) -> object:
            try:
                return ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError):
                return None

        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                continue
            call = call_node(node.left)
            if call is None:
                continue
            if isinstance(call.func, ast.Name):
                entry = aliases.get(call.func.id)
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                entry = aliases.get(call.func.value.id)
                if entry:
                    entry = entry.split(":", 1)[0] + ":" + call.func.attr
            else:
                entry = None
            if not entry or entry.endswith(":__module__"):
                continue
            expected_value = literal(node.comparators[0])
            if expected_value is None and not isinstance(node.comparators[0], ast.Constant):
                continue
            selector = None
            if isinstance(node.left, ast.Subscript) and isinstance(node.left.slice, ast.Constant):
                selector = node.left.slice.value
            if isinstance(node.left, ast.Attribute):
                selector = node.left.attr
            expected = ({str(selector): expected_value} if selector else expected_value)
            fixture = None
            if call.args:
                args = [literal(item) for item in call.args]
                if all(item is not None for item in args):
                    fixture = {"args": args}
            result.setdefault(entry, []).append({"expected": expected, "fixture": fixture})
    return result


@dataclass(frozen=True)
class PolicyResult:
    check: object
    accepted: bool
    reason: str = "ADMITTED"
    gating: bool = True


def _advisory_check(check: object, reason: str) -> object:
    provenance = copy.deepcopy(dict(getattr(check, "provenance", {}) or {}))
    provenance["green_lock_repair"] = {
        "schema": "stateframe-green-lock-repair/1",
        "admission": ADVISORY,
        "reason": reason,
    }
    try:
        return dataclasses.replace(check, oracle_basis=ADVISORY, provenance=provenance)
    except (TypeError, ValueError):
        return check


def classify_check(check: object, *, starter_root: str | Path | None = None,
                   public_expectations: object = None) -> PolicyResult:
    """Classify a generated check without making an advisory a gate."""
    if not enabled():
        return PolicyResult(check, True, "ADMITTED", True)
    try:
        from kernel_obligation_assess import CheckDefinition
        if isinstance(check, Mapping):
            check = CheckDefinition.from_mapping(check)
    except (ImportError, TypeError, ValueError):
        pass
    reason = _check_reason(check, starter_root=starter_root,
                           public_expectations=public_expectations)
    if reason is None:
        return PolicyResult(check, True, "ADMITTED", True)
    return PolicyResult(_advisory_check(check, reason), False, reason, False)


def _definition_map(runtime: object = None, frame: object = None) -> dict[str, object]:
    values: list[object] = []
    bundle = getattr(runtime, "host_check_bundle", None) if runtime is not None else None
    if callable(bundle):
        try:
            bundle = bundle()
        except (AttributeError, TypeError, ValueError):
            bundle = None
    if bundle is None and runtime is not None:
        bundle_loader = getattr(runtime, "_host_check_bundle", None)
        if callable(bundle_loader):
            try:
                bundle = bundle_loader()
            except (AttributeError, OSError, TypeError, ValueError):
                bundle = None
    if bundle is not None:
        checks = getattr(bundle, "checks", ()) or ()
        values.extend(checks.values() if isinstance(checks, Mapping) else checks)
    if runtime is not None:
        definitions = getattr(runtime, "host_check_definitions", ()) or ()
        values.extend(definitions.values() if isinstance(definitions, Mapping) else definitions)
    if isinstance(frame, Mapping):
        definitions = frame.get("check_definitions", ()) or ()
        values.extend(definitions.values() if isinstance(definitions, Mapping) else definitions)
    result: dict[str, object] = {}
    for value in values:
        ident = (getattr(value, "check_id", None)
                 or _mapping(value).get("check_id"))
        if ident:
            result[str(ident)] = value
    return result


def _receipt_status(value: object) -> str:
    row = _mapping(value)
    candidates = (
        row.get("underlying_status"), row.get("status"), row.get("result"),
        row.get("result_class"), getattr(value, "status", None),
    )
    for candidate in candidates:
        text = _reason_text(candidate)
        if text and text != "REUSED":
            return text
    return ""


def _receipt_execution(value: object) -> str:
    row = _mapping(value)
    return _reason_text(row.get("execution", getattr(value, "execution", "")))


def _is_advisory_definition(value: object) -> bool:
    row = _mapping(value)
    basis = _reason_text(row.get("oracle_basis", getattr(value, "oracle_basis", "")))
    if basis == ADVISORY:
        return True
    provenance = row.get("provenance", getattr(value, "provenance", {}))
    policy = provenance.get("green_lock_repair") if isinstance(provenance, Mapping) else None
    if isinstance(policy, Mapping) and _reason_text(policy.get("admission")) == ADVISORY:
        return True
    generator = provenance.get("generator_policy") if isinstance(provenance, Mapping) else None
    if isinstance(generator, Mapping) and _reason_text(generator.get("admission", "GATING")) != "GATING":
        return True
    certificate = provenance.get("admission_certificate") if isinstance(provenance, Mapping) else None
    if isinstance(certificate, Mapping) and (
            _reason_text(certificate.get("admission", "GATING")) != "GATING"
            or _reason_text(certificate.get("interpretation", "CONFIRMED")) != "CONFIRMED"
            or certificate.get("evidence_complete", True) is not True):
        return True
    return False


def is_advisory_assessment(value: object, *, definitions: Mapping[str, object] | None = None) -> bool:
    row = _mapping(value)
    reason = _reason_text(row.get("reason", getattr(value, "reason", "")))
    if reason in ADVISORY_REASONS or reason.startswith("ADVISORY"):
        return True
    if _is_advisory_definition(value):
        return True
    check_ids = row.get("check_ids", getattr(value, "check_ids", ()))
    if definitions:
        for check_id in check_ids or ():
            if _is_advisory_definition(definitions.get(str(check_id))):
                return True
    receipts = _assessment_receipts(value)
    if receipts and all(
            _reason_text(_mapping(receipt).get(
                "oracle_basis", getattr(receipt, "oracle_basis", ""))) == ADVISORY
            or _reason_text(_mapping(receipt).get(
                "reason", getattr(receipt, "reason", ""))) in ADVISORY_REASONS
            for receipt in receipts):
        return True
    return False


def _assessment_receipts(value: object) -> tuple[object, ...]:
    row = _mapping(value)
    receipts = row.get("receipts", getattr(value, "receipts", ()))
    if isinstance(receipts, Mapping):
        return (receipts,)
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes, bytearray)):
        return tuple(receipts)
    return ()


def _valid_receipt(receipt: object, definition: object = None) -> bool:
    if definition is not None and _is_advisory_definition(definition):
        return False
    row = _mapping(receipt)
    if _reason_text(row.get("oracle_basis", getattr(receipt, "oracle_basis", ""))) == ADVISORY:
        return False
    reason = _reason_text(row.get("reason", getattr(receipt, "reason", "")))
    if reason in ADVISORY_REASONS or reason.startswith("ADVISORY"):
        return False
    return (_receipt_status(receipt) == "COUNTEREXAMPLE"
            and _receipt_execution(receipt) == "EXECUTED")


def _assessment_is_blocking(value: object) -> bool:
    try:
        from kernel_obligation_assess import assessment_is_blocking
        return bool(assessment_is_blocking(value))
    except (ImportError, TypeError, ValueError):
        row = _mapping(value)
        return str(row.get("verdict", row.get("status", ""))).upper() in {
            "UNSATISFIED", "COUNTEREXAMPLE",
        }


def _target_tokens(value: object, *, root: object = None) -> dict[str, set[str]]:
    result = {"paths": set(), "symbols": set(), "ids": set()}

    def visit(item: object, key: str = "", depth: int = 0) -> None:
        if depth > 10:
            return
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                visit(child, str(raw_key).lower(), depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, key, depth + 1)
            return
        if not isinstance(item, str) or not item.strip():
            return
        text = item.strip()
        if key in _PATH_KEYS:
            path = _path_under_root(text, root) if root is not None else _norm_path(text)
            if path:
                result["paths"].add(path)
        if key in _SYMBOL_KEYS:
            symbol = text.strip()
            result["symbols"].add(symbol)
            if ":" in symbol:
                module, function = symbol.split(":", 1)
                result["symbols"].update({module, function, module.replace(".", "/")})
                result["paths"].add(_norm_path(module.replace(".", "/") + ".py"))
        if key in _ID_KEYS:
            result["ids"].add(text)

    visit(value)
    return result


def valid_gating_counterexamples(runtime: object = None, frame: Mapping[str, object] | None = None,
                                 assessments: object = None) -> tuple[dict[str, object], ...]:
    """Return only current executed counterexamples from admitted checks."""
    definitions = _definition_map(runtime, frame)
    values: list[object] = []
    if assessments is not None:
        if isinstance(assessments, Mapping):
            values.extend(assessments.values())
        elif isinstance(assessments, Iterable) and not isinstance(assessments, (str, bytes)):
            values.extend(assessments)
    elif runtime is not None:
        values.extend(getattr(runtime, "host_closure_assessments", ()) or ())
    if isinstance(frame, Mapping):
        values.extend(frame.get("obligation_assessments", ()) or ())
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            values.extend(closure.get("assessments", ()) or ())

    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for assessment in values:
        if is_advisory_assessment(assessment, definitions=definitions):
            continue
        row = _mapping(assessment)
        obligation_id = str(row.get("obligation_id", row.get("id", "")))
        check_ids = row.get("check_ids", ())
        if isinstance(check_ids, str):
            check_ids = (check_ids,)
        receipts = _assessment_receipts(assessment)
        if not receipts and _valid_receipt(assessment):
            receipts = (assessment,)
        if not receipts or (row and not _assessment_is_blocking(assessment)):
            continue
        for receipt in receipts:
            receipt_row = _mapping(receipt)
            check_id = str(receipt_row.get("check_id", ""))
            if not check_id and check_ids:
                check_id = str(next(iter(check_ids), ""))
            definition = definitions.get(check_id)
            if not _valid_receipt(receipt, definition):
                continue
            key = (obligation_id, check_id or _digest(receipt_row))
            if key in seen:
                continue
            seen.add(key)
            where_used = []
            ledger = getattr(runtime, "where_used_ledger", None) if runtime is not None else None
            open_sites = getattr(ledger, "open_obligations", None)
            if callable(open_sites):
                try:
                    where_used = list(open_sites())
                except (AttributeError, TypeError, ValueError):
                    where_used = []
            result.append({
                "obligation_id": obligation_id,
                "check_id": check_id,
                "assessment": assessment,
                "receipt": receipt,
                "counterexample": receipt_row.get("counterexample", row.get("counterexample")),
                "targets": _target_tokens({
                    "assessment": row,
                    "receipt": receipt_row,
                    "check": _mapping(definition),
                    "where_used": where_used,
                }, root=getattr(getattr(runtime, "owner", None), "root", None)),
            })
    return tuple(result)


def readiness_repair_counterexamples(
        runtime: object = None, frame: Mapping[str, object] | None = None,
        *, include_blockers: bool = True) -> tuple[dict[str, object], ...]:
    """Project the current readiness snapshot into repair-link evidence.

    When readiness authority is on, this is deliberately the only evidence
    path used by both the menu and action gate.  Raw receipts remain the
    compatibility path when that switch is off.
    """
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") != "on"
            and getattr(runtime, "stateframe_repair_blocks_done_mode", "off") != "on"):
        return valid_gating_counterexamples(runtime, frame)
    try:
        from kernel_readiness_authority import runtime_snapshot
        decision = runtime_snapshot(runtime, frame)
    except (AttributeError, ImportError, TypeError, ValueError):
        return ()
    if decision is None:
        return ()
    rows: list[dict[str, object]] = []
    source_rows = list(getattr(decision, "blockers", ()) or ()) if include_blockers else []
    source_rows.extend(getattr(decision, "repairable_advisories", ()) or ())
    root = getattr(getattr(runtime, "owner", None), "root", None)
    for raw in source_rows:
        row = _mapping(raw)
        row["authority"] = "readiness_decision"
        row["state_version"] = getattr(decision, "state_version", None)
        row["targets"] = _target_tokens(row, root=root)
        rows.append(row)
    return tuple(rows)


def _matches_path(candidate: object, expected: Iterable[str], *, root: object = None) -> bool:
    path = _path_under_root(candidate, root) if root is not None else _norm_path(candidate)
    if not path:
        return False
    for item in expected:
        other = _path_under_root(item, root) if root is not None else _norm_path(item)
        if path == other or path.endswith("/" + other) or other.endswith("/" + path):
            return True
    return False


def action_linked_to_counterexample(kind: object, arguments: object = None,
                                    runtime: object = None,
                                    frame: Mapping[str, object] | None = None) -> bool:
    """Check whether an action names the surface of an admitted failure."""
    action = str(kind or "").strip().lower()
    evidence = readiness_repair_counterexamples(runtime, frame)
    if not evidence:
        return False
    if ((getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
         or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on")
            and action in {"dispute", "read_ticket"}):
        return any(bool(item.get("counterexample_authority")) for item in evidence)
    if ((getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
         or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on")
            and (action in _EDIT_ACTIONS or action in {"run_tests", "predict"})):
        return True
    args = arguments if isinstance(arguments, Mapping) else {}
    action_tokens = _target_tokens(args, root=getattr(getattr(runtime, "owner", None), "root", None))
    for item in evidence:
        targets = item.get("targets", {})
        if not isinstance(targets, Mapping):
            continue
        if action in _EDIT_ACTIONS:
            if (action_tokens["paths"] and any(
                    _matches_path(path, targets.get("paths", ()),
                                  root=getattr(getattr(runtime, "owner", None), "root", None))
                    for path in action_tokens["paths"])):
                return True
            if action_tokens["symbols"] & set(targets.get("symbols", ())):
                return True
            continue
        if action in _READ_ACTIONS:
            if action_tokens["ids"] & set(targets.get("ids", ())):
                return True
            if action_tokens["paths"] and any(
                    _matches_path(path, targets.get("paths", ()),
                                  root=getattr(getattr(runtime, "owner", None), "root", None))
                    for path in action_tokens["paths"]):
                return True
            # An unqualified verification/read action is the only available
            # way to address a current check when the host supplies the
            # selector rather than the model.
            if not action_tokens["paths"] and not action_tokens["ids"]:
                return True
    return False


def repair_refusal(kind: object, evidence: Sequence[Mapping[str, object]] = ()) -> str:
    action = str(kind or "action").strip().upper() or "ACTION"
    ids = [str(item.get("obligation_id") or item.get("check_id"))
           for item in evidence if item.get("obligation_id") or item.get("check_id")]
    suffix = json.dumps(list(dict.fromkeys(ids)), ensure_ascii=False, separators=(",", ":"))
    return f"GREEN LOCK REPAIR: {action} refused; valid counterexamples remain {suffix}"


def done_is_gated(runtime: object = None, frame: Mapping[str, object] | None = None) -> bool:
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
            or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"):
        try:
            from kernel_readiness_authority import runtime_snapshot
            decision = runtime_snapshot(runtime, frame)
            return True if decision is None else bool(decision.blockers)
        except (AttributeError, ImportError, TypeError, ValueError):
            return True
    return bool(enabled() and valid_gating_counterexamples(runtime, frame))


def repair_terminal_menu(current_tools: Sequence[dict], all_tools: Sequence[dict], *,
                         runtime: object = None,
                         frame: Mapping[str, object] | None = None) -> list[dict]:
    """Re-open only tools capable of addressing an admitted counterexample."""
    if not enabled() or not readiness_repair_counterexamples(runtime, frame):
        return list(current_tools)
    result = list(current_tools)
    present = {str(item.get("function", {}).get("name")) for item in result
               if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)}
    allowed = _EDIT_ACTIONS | _READ_ACTIONS
    for tool in all_tools:
        if not isinstance(tool, Mapping) or not isinstance(tool.get("function"), Mapping):
            continue
        name = str(tool["function"].get("name", ""))
        if name in allowed and name not in present:
            result.append(tool)
            present.add(name)
    return result


def annotate_preservation_counterexample(root: str | Path, check: object,
                                         counterexample: object,
                                         detail: object = None) -> object:
    """Attach static/dynamic input mutation locations to a failed replay."""
    template = _reason_text(getattr(check, "template", ""))
    if template not in {"PRESERVE_INPUT", "REPLAY_EQUIVALENT"}:
        return counterexample
    result = copy.deepcopy(dict(counterexample)) if isinstance(counterexample, Mapping) else {}
    dynamic: list[object] = []
    for item in _walk(detail):
        if isinstance(item, Mapping):
            for key in ("input_mutations", "mutation_trace", "mutations"):
                value = item.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    dynamic.extend(copy.deepcopy(list(value)))
    sites: list[object] = []
    entry = getattr(check, "entry_point", None)
    if entry:
        try:
            from kernel_obligation_assess import mutation_sites
            sites = list(mutation_sites(root, entry))
        except (ImportError, OSError, TypeError, ValueError):
            sites = []
    site_dicts = []
    refs = []
    for site in sites:
        row = site.to_dict() if callable(getattr(site, "to_dict", None)) else _mapping(site)
        if not row:
            continue
        site_dicts.append(row)
        path = _norm_path(row.get("path"))
        line = row.get("line")
        if path and isinstance(line, int):
            refs.append(f"{path}:{line}")
    if not refs:
        for item in dynamic:
            if isinstance(item, Mapping) and item.get("path") and item.get("line"):
                refs.append(f"{_norm_path(item['path'])}:{item['line']}")
    result["input_mutation_trace"] = dynamic
    result["mutation_sites"] = site_dicts
    result["mutation_site_refs"] = list(dict.fromkeys(refs))
    if refs:
        result["mutation_site"] = refs[0]
    return result


@dataclass
class _EditSnapshot:
    path: Path
    before: bytes | None
    existed: bool
    green: bool
    green_revision: object
    product_edit_applied: bool
    fallback_revision: object
    before_manifest: object = None
    effect_id: str = ""


class RepairController:
    """Capture an admitted edit and verify it before allowing green to be lost."""

    def __init__(self, runtime: object):
        self.runtime = runtime
        self._before: dict[str, _EditSnapshot] = {}

    def _resolve(self, path: object) -> Path:
        owner = getattr(self.runtime, "owner", None)
        root = Path(getattr(owner, "root", Path.cwd())).resolve()
        candidate = Path(str(path))
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            if candidate.is_symlink():
                raise ValueError("symlink product path")
            # macOS temporary roots can contain a system-level /var -> /private
            # link. Normalize that container link before checking containment;
            # the product path itself remains symlink-refused above.
            lexical = candidate.resolve(strict=False)
            if root not in lexical.parents and lexical != root:
                raise ValueError("path outside product root")
            current = root
            for part in lexical.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("symlink product path")
            resolved = lexical.resolve()
            resolved.relative_to(root)
            return resolved
        except (OSError, ValueError) as error:
            raise ValueError("unsafe product path") from error

    def _product_paths(self, candidate: Path) -> tuple[str, ...]:
        owner = getattr(self.runtime, "owner", None)
        values = (
            getattr(self.runtime, "product_paths", None)
            or getattr(self.runtime, "candidate_product_paths", None)
            or getattr(owner, "product_paths", None)
            or getattr(owner, "allowed_product_paths", None)
        )
        if isinstance(values, str):
            values = (values,)
        paths = tuple(dict.fromkeys(str(item) for item in (values or ()) if str(item).strip()))
        if paths:
            return paths
        root = Path(getattr(owner, "root", Path.cwd())).resolve()
        return (candidate.relative_to(root).as_posix(),)

    def _before_manifest(self, candidate: Path) -> object | None:
        try:
            from kernel_candidate_checkpoint_store import ProductManifest
            owner = getattr(self.runtime, "owner", None)
            root = getattr(self.runtime, "root", None) or getattr(owner, "root", Path.cwd())
            return ProductManifest.from_root(root, self._product_paths(candidate))
        except (ImportError, OSError, TypeError, ValueError):
            return None

    def _after_manifest(self, candidate: Path) -> object | None:
        return self._before_manifest(candidate)

    def _effect_id(self, candidate: Path, before: object) -> str:
        explicit = getattr(self.runtime, "current_effect_id", None) or getattr(self.runtime, "effect_id", None)
        if explicit:
            return str(explicit)
        digest = getattr(before, "digest", "unknown")
        return f"product-edit:{candidate.as_posix()}:{digest}"

    def capture_before(self, path: object) -> None:
        try:
            candidate = self._resolve(path)
            before = candidate.read_bytes() if candidate.is_file() else None
        except (OSError, UnicodeError, ValueError):
            return
        lock = getattr(self.runtime, "green_lock", None)
        key = str(candidate)
        if key in self._before:
            return
        before_manifest = self._before_manifest(candidate)
        self._before[key] = _EditSnapshot(
            candidate, before, before is not None,
            bool(getattr(lock, "green", False)),
            getattr(lock, "_green_revision", None),
            bool(getattr(lock, "_product_edit_applied", False)),
            getattr(lock, "_terminal_fallback_revision", None),
            before_manifest=before_manifest,
            effect_id=self._effect_id(candidate, before_manifest),
        )

    def _runner(self):
        for name in ("public_check_runner", "auto_verify_runner", "run_public_checks"):
            candidate = getattr(self.runtime, name, None)
            if callable(candidate):
                return candidate
        return None

    def _selectors(self) -> tuple[str, ...]:
        method = getattr(self.runtime, "required_public_selectors", None)
        if callable(method):
            try:
                values = method()
            except (AttributeError, TypeError, ValueError):
                values = ()
        else:
            owner = getattr(self.runtime, "owner", None)
            values = getattr(owner, "f2p_ids", ()) or getattr(owner, "f2p_inworld", ())
        if isinstance(values, str):
            return (values,) if values.strip() else ()
        if isinstance(values, Sequence):
            return tuple(str(item) for item in values if str(item).strip())
        return ()

    @staticmethod
    def _result(result: object) -> tuple[bool, int | None, object]:
        if isinstance(result, bool):
            return result, 0 if result else 1, result
        if isinstance(result, int):
            return result == 0, result, result
        returncode = getattr(result, "returncode", None)
        if isinstance(returncode, int):
            return returncode == 0, returncode, result
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            if result and isinstance(result[0], int):
                return result[0] == 0, result[0], result
        row = _mapping(result)
        exit_code = row.get("exit_code", row.get("exit", row.get("returncode")))
        if isinstance(exit_code, int):
            return exit_code == 0, exit_code, result
        status = _reason_text(row.get("result_class", row.get("status", row.get("result", ""))))
        if status in {"PASS", "PASSED", "GREEN", "SUCCESS", "OK"}:
            return True, 0, result
        if status in {"FAIL", "FAILED", "RED", "COUNTEREXAMPLE", "ERROR"}:
            return False, 1, result
        return False, None, result

    def _restore(self, snapshot: _EditSnapshot) -> bool:
        """Restore one legacy snapshot and verify bytes before rebinding lock state."""
        try:
            if snapshot.path.is_symlink():
                return False
            if snapshot.before is None:
                if snapshot.path.exists():
                    snapshot.path.unlink()
            else:
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = snapshot.path.with_name(f".{snapshot.path.name}.repair-restore")
                temporary.write_bytes(snapshot.before)
                os.replace(temporary, snapshot.path)
            if snapshot.before is None:
                if snapshot.path.exists():
                    return False
            elif snapshot.path.read_bytes() != snapshot.before:
                return False
        except OSError:
            return False
        lock = getattr(self.runtime, "green_lock", None)
        restore = getattr(lock, "repair_restored", None)
        if callable(restore):
            restore(snapshot)
        return True

    def _inventory(self) -> object:
        return (getattr(self.runtime, "candidate_check_inventory", None)
                or getattr(self.runtime, "public_check_inventory", None)
                or ())

    def _open_requirements(self) -> tuple[str, ...]:
        value = getattr(self.runtime, "open_requirement_ids", ())
        if callable(value):
            try:
                value = value()
            except (AttributeError, TypeError, ValueError):
                value = ()
        if isinstance(value, str):
            return (value,) if value else ()
        return tuple(dict.fromkeys(str(item) for item in (value or ()) if str(item)))

    @staticmethod
    def _observation_count(raw: object) -> int:
        if isinstance(raw, Mapping):
            for key in ("observation_count", "observed_count", "tests_run", "count"):
                value = raw.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    return max(0, value)
            nested = raw.get("observed")
            if isinstance(nested, Mapping):
                return RepairController._observation_count(nested)
        return 0

    @classmethod
    def _verification_unknown(cls, raw: object, exit_code: int | None) -> bool:
        """Keep absent/zero-observation verifier outcomes out of recovery decisions."""
        if cls._observation_count(raw) <= 0:
            return True
        row = _mapping(raw)
        status = _reason_text(row.get("result_class", row.get("status", row.get("result", ""))))
        return status in {"", "UNKNOWN", "ABSENT", "SKIPPED", "REFUSED", "TIMEOUT", "UNAVAILABLE"}

    def _record_public_receipt(self, *, candidate_digest: str, raw: object,
                               selectors: Sequence[str], passed: bool) -> object | None:
        try:
            from kernel_candidate_preservation import enabled, reducer_for
            if not enabled(self.runtime):
                return None
            reducer = reducer_for(self.runtime)
            if reducer is None:
                return None
            observed_count = self._observation_count(raw)
            if isinstance(raw, Mapping):
                default_check = str(raw.get("check_id", ""))
                definition_hash = str(raw.get("definition_hash", raw.get("check_hash", "")) or "")
                environment = str(raw.get("environment_id", raw.get("environment", "default")) or "default")
                check_ids = (default_check,) if default_check else tuple(selectors)
            else:
                definition_hash = ""
                environment = "default"
                check_ids = tuple(selectors)
            result = "PASS" if passed else "FAIL"
            inventory = self._inventory()
            definitions = {}
            if isinstance(inventory, Mapping):
                definitions = {
                    str(key): value for key, value in inventory.items()
                }
            else:
                for definition in getattr(inventory, "definitions", ()) or ():
                    check_key = getattr(definition, "check_id", None)
                    if check_key:
                        definitions[str(check_key)] = definition
            receipt: object | None = None
            for index, check_id in enumerate(check_ids):
                if not check_id:
                    continue
                definition = definitions.get(check_id)
                check_definition_hash = str(getattr(definition, "definition_hash", "") or "")
                check_environment = str(getattr(definition, "environment_id", "") or "")
                row = {
                    "check_id": check_id,
                    "result": result,
                    "candidate_digest": candidate_digest,
                    "definition_hash": definition_hash or check_definition_hash,
                    "environment_id": environment if definition_hash else (check_environment or environment),
                    "execution_id": f"repair:{candidate_digest}:{check_id}:{index}",
                    "observation_count": observed_count,
                    "run_id": reducer.run_id,
                    "task_id": reducer.task_id,
                    "receipt_id": f"repair:{candidate_digest}:{check_id}",
                }
                receipt = reducer.record_check(row)
            return receipt
        except (ImportError, TypeError, ValueError):
            return None

    def _preserve_pass(self, snapshot: _EditSnapshot, candidate: Path, raw: object,
                       selectors: Sequence[str]) -> object | None:
        try:
            from kernel_candidate_preservation import enabled, reducer_for
            if not enabled(self.runtime):
                return None
            reducer = reducer_for(self.runtime)
            before_manifest = snapshot.before_manifest
            after_manifest = self._after_manifest(candidate)
            if reducer is None or before_manifest is None or after_manifest is None:
                return None
            transaction_id = getattr(self.runtime, "candidate_transaction_id", "")
            settled = bool(getattr(self.runtime, "candidate_transaction_settled", True))
            candidate_record = reducer.observe_settled_mutation(
                snapshot.effect_id, before_manifest, after_manifest,
                transaction_id=transaction_id, settled=settled,
                model_owned=True, open_requirements=self._open_requirements(),
                check_inventory=self._inventory(),
            )
            if candidate_record is not None:
                setattr(self.runtime, "_candidate_last_captured", candidate_record.candidate_id)
                setattr(self.runtime, "candidate_preservation_snapshot", reducer.export())
            return candidate_record
        except (ImportError, OSError, TypeError, ValueError):
            return None

    def _recover_failed_candidate(self, snapshot: _EditSnapshot, candidate: Path,
                                  raw: object, selectors: Sequence[str],
                                  after_manifest: object | None) -> dict[str, object] | None:
        try:
            from kernel_candidate_preservation import Candidate, enabled, reducer_for
            if not enabled(self.runtime):
                return None
            reducer = reducer_for(self.runtime)
            if reducer is None or after_manifest is None:
                return None
            current = Candidate(
                candidate_id="current-unpreserved", checkpoint_id="",
                run_id=reducer.run_id, task_id=reducer.task_id,
                parent_revision=str(getattr(snapshot.before_manifest, "digest", "")),
                revision=str(getattr(after_manifest, "digest", "")),
                effect_id=snapshot.effect_id, branch_id="current",
                manifest=after_manifest, created_index=len(reducer.state.candidates) + 1,
            )
            reason = self._record_public_receipt(
                candidate_digest=current.revision, raw=raw,
                selectors=selectors, passed=False)
            reducer.observe_current_revision(current.revision)
            fallback = reducer.eligible_fallback(current, self._inventory())
            if fallback is None or reason is None:
                return None
            plan = reducer.plan_recovery(current.revision, fallback.checkpoint_id, reason, self._inventory())
            receipt = reducer.commit_recovery(
                plan,
                workspace=getattr(self.runtime, "root", None)
                or getattr(getattr(self.runtime, "owner", None), "root", None),
                open_requirements=self._open_requirements(),
            )
            if receipt.status == "RECOVERED":
                from kernel_candidate_preservation import invalidate_revision_handles
                invalidate_revision_handles(self.runtime)
                setattr(self.runtime, "candidate_preservation_snapshot", reducer.export())
            return receipt.as_dict()
        except (ImportError, OSError, TypeError, ValueError):
            return None

    def _record(self, receipt: dict[str, object]) -> dict[str, object]:
        collection = getattr(self.runtime, "repair_receipts", None)
        if isinstance(collection, list):
            collection.append(copy.deepcopy(receipt))
        journal = getattr(self.runtime, "journal", None)
        if journal is not None and callable(getattr(journal, "emit", None)):
            journal.emit("green_lock_repair_receipt", producer="kernel_green_lock_repair",
                         receipt=copy.deepcopy(receipt), dispatch=False)
        callback = getattr(self.runtime, "record_repair_receipt", None)
        if callable(callback):
            try:
                callback(copy.deepcopy(receipt))
            except (AttributeError, TypeError, ValueError):
                pass
        return receipt

    def settle_action(self, record: Mapping[str, object] | None = None) -> None:
        """Close provisional candidate branches only at the settled action boundary."""
        try:
            from kernel_candidate_preservation import reducer_for
            reducer = reducer_for(self.runtime)
            if reducer is None:
                return
            effect_id = record.get("effect_id") if isinstance(record, Mapping) else None
            if effect_id:
                reducer.settle_effect(effect_id)
            else:
                reducer.settle_latest()
        except (ImportError, TypeError, ValueError):
            return

    def _no_veto(self) -> bool:
        try:
            from kernel_postgreen_no_veto import enabled as no_veto_enabled
            return bool(no_veto_enabled(self.runtime))
        except (ImportError, TypeError, ValueError):
            return False

    def after_edit(self, path: object) -> dict[str, object] | None:
        try:
            candidate = self._resolve(path)
        except (OSError, TypeError, ValueError):
            return None
        snapshot = self._before.pop(str(candidate), None)
        if snapshot is None:
            return None
        selectors = self._selectors()
        runner = self._runner()
        before_digest = _digest(snapshot.before.decode("utf-8", "replace")
                                if snapshot.before is not None else None)
        observational = self._no_veto()
        if runner is None:
            if observational:
                return self._record({
                    "schema": "stateframe-green-lock-repair-receipt/1",
                    "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root",
                                                                None)),
                    "status": "UNVERIFIED: public verification unavailable (edit kept; no-veto)",
                    "result_class": "UNVERIFIED",
                    "reason": PUBLIC_CHECK_UNAVAILABLE,
                    "selectors": list(selectors),
                    "before_digest": before_digest,
                })
            restored = self._restore(snapshot) if snapshot.green else False
            return self._record({
                "schema": "stateframe-green-lock-repair-receipt/1",
                "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                "status": ("REVERTED: public verification unavailable" if restored
                           else "UNVERIFIED: public verification unavailable"),
                "result_class": "REVERTED" if restored else "UNVERIFIED",
                "reason": PUBLIC_CHECK_UNAVAILABLE,
                "selectors": list(selectors),
                "before_digest": before_digest,
            })
        try:
            try:
                raw = runner(selectors)
            except TypeError:
                raw = runner()
            passed, exit_code, observed = self._result(raw)
        except Exception as error:  # a verifier failure is fail-closed
            from kernel_harden8 import record_failure
            record_failure(
                self.runtime,
                "green_lock.repair_verifier",
                error,
                authority=True,
                disposition="deny",
                denied_actions=("green", "done", "submit"),
            )
            passed, exit_code, observed = False, None, {"error_type": type(error).__name__,
                                                          "detail": str(error)[:240]}
        after_manifest = self._after_manifest(candidate)
        after_digest = str(getattr(after_manifest, "digest", "")) or _digest(
            candidate.read_bytes() if candidate.is_file() else None)
        preserved = None
        if after_manifest is not None and (not passed or self._observation_count(observed) > 0):
            # Keep the failed branch available to the native restore action; its
            # public receipt still decides whether that branch is recoverable.
            preserved = self._preserve_pass(snapshot, candidate, observed, selectors)
        self._record_public_receipt(
            candidate_digest=after_digest, raw=observed,
            selectors=selectors, passed=passed)
        if self._verification_unknown(observed, exit_code):
            return self._record({
                "schema": "stateframe-green-lock-repair-receipt/1",
                "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                "status": "UNVERIFIED: public verification unavailable (zero or unknown observation)",
                "result_class": "UNVERIFIED",
                "reason": PUBLIC_CHECK_UNAVAILABLE,
                "selectors": list(selectors),
                "exit_code": exit_code,
                "observed": copy.deepcopy(observed),
                "before_digest": before_digest,
                "after_digest": after_digest,
            })
        if not passed and observational:
            # A previous revision passed, but this revision fails. Keep the edit and failing evidence; let the
            # model repair; never silently restore.  The observation marks the current revision unverified.
            test_observation = getattr(self.runtime, "test_observation", None)
            if callable(test_observation):
                try:
                    test_observation(list(selectors), exit_code if exit_code not in (None, 0) else 1,
                                     self.runtime.binding())
                except (AttributeError, TypeError, ValueError, OSError):
                    pass
            return self._record({
                "schema": "stateframe-green-lock-repair-receipt/1",
                "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                "status": "RED: public regression (edit kept; repair allowed; no-veto)",
                "result_class": "PUBLIC_RED",
                "reason": PUBLIC_REGRESSION,
                "selectors": list(selectors),
                "exit_code": exit_code,
                "observed": copy.deepcopy(observed),
                "before_digest": before_digest,
                "after_digest": after_digest,
            })
        if not passed and not observational:
            recovery = self._recover_failed_candidate(
                snapshot, candidate, observed, selectors, after_manifest)
            if isinstance(recovery, Mapping) and recovery.get("status") == "RECOVERED":
                return self._record({
                    "schema": "stateframe-green-lock-repair-receipt/1",
                    "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                    "status": "RECOVERED: public regression",
                    "result_class": "RECOVERED",
                    "reason": PUBLIC_REGRESSION,
                    "selectors": list(selectors),
                    "exit_code": exit_code,
                    "observed": copy.deepcopy(observed),
                    "before_digest": before_digest,
                    "after_digest": after_digest,
                    "recovery": copy.deepcopy(dict(recovery)),
                })
        if not passed and not snapshot.green:
            return self._record({
                "schema": "stateframe-green-lock-repair-receipt/1",
                "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                "status": "RED: public failure (edit kept; no eligible candidate)",
                "result_class": "PUBLIC_RED",
                "reason": PUBLIC_REGRESSION,
                "selectors": list(selectors),
                "exit_code": exit_code,
                "observed": copy.deepcopy(observed),
                "before_digest": before_digest,
                "after_digest": after_digest,
            })
        if not passed:
            restored = self._restore(snapshot)
            return self._record({
                "schema": "stateframe-green-lock-repair-receipt/1",
                "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
                "status": "REVERTED: public regression" if restored else "RECOVERY_INCOMPLETE: public regression",
                "result_class": "REVERTED" if restored else "RECOVERY_INCOMPLETE",
                "reason": PUBLIC_REGRESSION,
                "selectors": list(selectors),
                "exit_code": exit_code,
                "observed": copy.deepcopy(observed),
                "before_digest": before_digest,
                "after_digest": after_digest,
            })
        test_observation = getattr(self.runtime, "test_observation", None)
        if callable(test_observation):
            try:
                test_observation(list(selectors), 0, self.runtime.binding())
            except (AttributeError, TypeError, ValueError, OSError):
                pass
        lock = getattr(self.runtime, "green_lock", None)
        if callable(getattr(lock, "sync", None)):
            try:
                lock.sync()
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            after = candidate.read_bytes() if candidate.is_file() else None
        except OSError:
            after = None
        return self._record({
            "schema": "stateframe-green-lock-repair-receipt/1",
            "path": _path_under_root(candidate, getattr(getattr(self.runtime, "owner", None), "root", None)),
            "status": "AUTO_VERIFIED",
            "result_class": "PASS",
            "reason": "PUBLIC_CHECK_GREEN",
            "selectors": list(selectors),
            "exit_code": exit_code,
            "observed": copy.deepcopy(observed),
            "before_digest": before_digest,
            "after_digest": after_digest or _digest(after.decode("utf-8", "replace") if after is not None else None),
            "candidate_id": getattr(preserved, "candidate_id", None),
            "preserved_while_requirements_open": bool(self._open_requirements()),
        })


def controller_for(runtime: object) -> RepairController | None:
    preservation_on = False
    try:
        from kernel_candidate_preservation import enabled as preservation_enabled
        preservation_on = preservation_enabled(runtime)
    except (ImportError, ValueError):
        preservation_on = False
    if (not enabled() and not preservation_on) or runtime is None:
        return None
    existing = getattr(runtime, "green_lock_repair", None)
    if isinstance(existing, RepairController):
        return existing
    controller = RepairController(runtime)
    try:
        runtime.green_lock_repair = controller
    except AttributeError:
        pass
    return controller


__all__ = [
    "ADVISORY", "ADVISORY_REASONS", "CONTRADICTS_PUBLIC_TEST", "MODE_ENV",
    "PUBLIC_CHECK_UNAVAILABLE", "PUBLIC_REGRESSION", "PolicyResult",
    "RepairController", "UNBOUND_ENTRYPOINT", "UNBOUND_INVOCATION",
    "UNSATISFIABLE_CONTRACT_CELL", "action_linked_to_counterexample",
    "annotate_preservation_counterexample", "classify_check", "controller_for",
    "done_is_gated", "enabled", "is_advisory_assessment", "mode",
    "public_expectations", "repair_refusal", "repair_terminal_menu",
    "readiness_repair_counterexamples", "valid_gating_counterexamples",
]
