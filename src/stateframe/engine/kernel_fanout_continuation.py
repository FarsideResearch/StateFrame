"""Render bounded, typed continuation facts for the latest parallel fan-out."""
from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES


MAX_SECTION_BYTES = 3_072
# Owned-source and failing-test windows share this hard request-frame ceiling.
MAX_PIN_BYTES = 6 * 1_024
MAX_SOURCE_EXCERPT_BYTES = 512
MAX_SOURCE_SCAN_BYTES = 128 * 1_024
MAX_CAPTURE_FILE_BYTES = 16 * 1_024 * 1_024
MAX_UNITS = 1_024
SWITCH_NAME = "STATEFRAME_PARALLEL_FANOUT_CONTINUATION"
_DISPOSITION_KINDS = frozenset({
    "protected_fanout_disposition", "parallel_fanout_disposition",
    "protected_parallel_disposition", "fanout_disposition", "parallel_disposition",
})
_EDIT_TOOLS = frozenset({
    "apply_patch", "create_file", "delete", "delete_file", "edit", "edit_file",
    "file_create", "file_delete", "file_move", "insert_before", "move", "patch",
    "rename_file", "replace", "replace_module", "str_replace", "write", "write_file", *PROTOCOL_EDIT_TOOL_NAMES,
})
_CHECK_TOOLS = frozenset({"pytest", "run_tests", "test"})
# Canonical wire shape of a parallel disposition, written by kernel_parallel and read here.
# Identity rows are [id, status]; detail rows are positional; scope rows (owned paths and check
# selectors, needed for pins) live in the additive unit_scopes section.
WRITER_UNIT_FIELDS = ("id", "status")
WRITER_DETAIL_FIELDS = ("id", "reason", "changed_files", "contract_ok", "check_status", "requests")
WRITER_SCOPE_FIELDS = ("id", "owns", "check_selectors")


class FanoutContinuationError(ValueError):
    """A typed fan-out disposition cannot be rendered safely."""


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _clean(value: object, *, limit: int = 240, replace_semicolons: bool = True) -> str:
    raw = str(value if value is not None else "")
    text = " ".join("".join(char for char in raw if char.isprintable()).split())
    if replace_semicolons:
        text = text.replace(";", ",")
    text = text.replace("\x00", "")
    return text[:limit]


def _relative_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\\", "/")
    if any(not char.isprintable() for char in text):
        return None
    if any(part in {"", ".", ".."} for part in text.split("/")):
        return None
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)
            or "\n" in text or "\r" in text):
        return None
    return path.as_posix()


def _paths(value: object) -> list[str]:
    if isinstance(value, str):
        raw: Sequence[object] = [value]
    elif _is_sequence(value):
        raw = value
    else:
        return []
    result = []
    for item in raw:
        path = _relative_path(item)
        if path and path not in result:
            result.append(path)
    return result


def _selectors(unit: Mapping[str, object]) -> list[str]:
    raw = unit.get("check", unit.get("checks", unit.get("check_selectors", ())))
    if isinstance(raw, Mapping):
        raw = raw.get("selectors", ())
    if isinstance(raw, str):
        raw = [raw]
    return [_clean(item, limit=220) for item in raw if isinstance(item, str) and item.strip()] if _is_sequence(raw) else []


def _writer_detail_map(details: object) -> dict[str, Sequence[object]]:
    if not _is_sequence(details):
        raise FanoutContinuationError("fan-out disposition has invalid subframe details")
    by_id: dict[str, Sequence[object]] = {}
    for detail in details:
        if (not _is_sequence(detail) or not detail or not isinstance(detail[0], str)
                or detail[0] in by_id):
            raise FanoutContinuationError("fan-out disposition has a malformed subframe detail")
        by_id[detail[0]] = detail
    return by_id


def _writer_scope_map(scopes: object) -> dict[str, Sequence[object]]:
    if not _is_sequence(scopes):
        raise FanoutContinuationError("fan-out disposition has invalid subframe scopes")
    by_id: dict[str, Sequence[object]] = {}
    for scope in scopes:
        if not _is_sequence(scope) or len(scope) != len(WRITER_SCOPE_FIELDS):
            raise FanoutContinuationError("fan-out disposition has a malformed subframe scope")
        if not isinstance(scope[0], str) or scope[0] in by_id:
            raise FanoutContinuationError("fan-out disposition has a malformed subframe scope")
        by_id[scope[0]] = scope
    return by_id


def _writer_unit_mapping(
        identity: Sequence[object], details: Mapping[str, Sequence[object]],
        scopes: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    """Map one writer identity row plus its detail and scope rows onto the renderer's unit fields."""
    if (len(identity) != len(WRITER_UNIT_FIELDS) or not isinstance(identity[0], str)
            or not isinstance(identity[1], str)):
        raise FanoutContinuationError("fan-out disposition contains a malformed subframe")
    row: dict[str, object] = dict(zip(WRITER_UNIT_FIELDS, identity))
    scope = scopes.get(identity[0])
    if scope is not None:
        row["owns"] = scope[1]
        row["check_selectors"] = scope[2]
    detail = details.get(identity[0])
    if detail is None:
        return row
    if len(detail) != len(WRITER_DETAIL_FIELDS):
        raise FanoutContinuationError("fan-out disposition has a malformed subframe detail")
    values = dict(zip(WRITER_DETAIL_FIELDS, detail))
    changed = values["changed_files"]
    if not _is_sequence(changed):
        raise FanoutContinuationError("fan-out disposition has malformed changed files")
    row.update({
        "reason": values["reason"],
        "check_status": values["check_status"],
        "requests": values["requests"],
        "contract_ok": values["contract_ok"],
        "changed_paths": [item[0] for item in changed if _is_sequence(item) and item and isinstance(item[0], str)],
    })
    return row


def _canonical_unit_rows(raw_units: Sequence[object], disposition: Mapping[str, object]) -> Sequence[object]:
    """Accept the parallel writer's positional rows and the mapping rows the CLI projects."""
    if not any(_is_sequence(row) for row in raw_units):
        return raw_units
    details = _writer_detail_map(disposition.get("unit_details", ()))
    scopes = _writer_scope_map(disposition.get("unit_scopes", ()))
    return [
        _writer_unit_mapping(row, details, scopes) if _is_sequence(row) else row
        for row in raw_units
    ]


def _unit_rows(disposition: Mapping[str, object]) -> list[dict[str, object]]:
    raw_units = disposition.get("units", disposition.get("unit_dispositions", ()))
    if not _is_sequence(raw_units) or len(raw_units) > MAX_UNITS:
        raise FanoutContinuationError("fan-out disposition has invalid subframe rows")
    raw_units = _canonical_unit_rows(raw_units, disposition)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            raise FanoutContinuationError("fan-out disposition contains a malformed subframe")
        unit_id = _clean(raw.get("id", raw.get("unit_id", "")), limit=120)
        if not unit_id:
            raise FanoutContinuationError("fan-out subframe has no id")
        if unit_id in seen_ids:
            raise FanoutContinuationError("fan-out disposition contains duplicate subframe ids")
        seen_ids.add(unit_id)
        check_result = _clean(
            raw.get("check_result", raw.get("check_status", raw.get("check", "NOT_RUN"))), limit=32
        ).upper()
        if check_result not in {"PASS", "FAIL", "NOT_RUN", "UNKNOWN", "INVALID"}:
            check_result = "UNKNOWN"
        usage = raw.get("usage")
        sends = raw.get("child_sends", raw.get("requests"))
        if sends is None and isinstance(usage, Mapping):
            sends = usage.get("requests")
        if type(sends) is not int or sends < 0:
            sends = 0
        files = raw.get("files")
        changed = raw.get("changed_paths", raw.get("changed_files"))
        if changed is None and _is_sequence(files):
            changed = [item.get("path") for item in files if isinstance(item, Mapping)]
        contract = raw.get("contract", raw.get("contract_ok"))
        if type(contract) is not bool:
            contract = None
        reason = raw.get("reason")
        detail = raw.get("detail")
        if isinstance(reason, str) and reason.strip() and isinstance(detail, str) and detail.strip():
            if detail.strip() not in reason:
                reason = f"{reason}: {detail}"
        rows.append({
            "id": unit_id,
            "owns": _paths(raw.get("owns", raw.get("owned_paths", ()))),
            "check": _selectors(raw),
            "status": _clean(raw.get("status", "UNKNOWN"), limit=48).upper() or "UNKNOWN",
            "check_result": check_result,
            "child_sends": sends,
            "changed_paths": _paths(changed),
            "contract": contract,
            "reason": (_clean(reason, limit=500, replace_semicolons=False)
                       if isinstance(reason, str) and reason.strip() else None),
        })
    return rows


def _initial_remaining(disposition: Mapping[str, object], units: Sequence[Mapping[str, object]]) -> list[str]:
    raw = disposition.get("remaining")
    known = {str(unit["id"]) for unit in units}
    if _is_sequence(raw):
        requested = [item for item in raw if isinstance(item, str) and item in known]
        unresolved = set(requested)
        unresolved.update(
            str(unit["id"]) for unit in units
            if unit.get("status") != "COMMITTED" or unit.get("check_result") != "PASS"
        )
        return [str(unit["id"]) for unit in units if str(unit["id"]) in unresolved]
    return [str(unit["id"]) for unit in units
            if unit.get("status") != "COMMITTED" or unit.get("check_result") != "PASS"]


def disposition_from_records(
        call_id: str, request: Mapping[str, object], result: Mapping[str, object], *,
        turn_index: int = 0,
) -> dict[str, object]:
    """Project durable parallel records into the small renderer input schema."""
    planned = request.get("units", ())
    outcomes = result.get("outcomes", ())
    if not _is_sequence(planned) or len(planned) > MAX_UNITS or not _is_sequence(outcomes):
        raise FanoutContinuationError("parallel result or plan has invalid subframe rows")
    by_id = {
        str(item.get("unit_id", item.get("id", ""))): item
        for item in outcomes if isinstance(item, Mapping)
    }
    admission = result.get("admission")
    admission = admission if isinstance(admission, Mapping) else {}
    admission_refusal = admission.get("refusal")
    units: list[dict[str, object]] = []
    for plan in planned:
        if not isinstance(plan, Mapping):
            raise FanoutContinuationError("parallel request has a malformed subframe")
        unit_id = str(plan.get("id") or "")
        outcome = by_id.get(unit_id)
        if outcome is None:
            refusal = admission_refusal if isinstance(admission_refusal, Mapping) else {}
            status = "REFUSED" if admission.get("status") == "REFUSED" else "ENGINE_ERROR"
            outcome = {
                "status": status,
                "check": "NOT_RUN",
                "contract_ok": None,
                "files": [],
                "usage": {"requests": 0},
                "reason": refusal.get("reason", status),
                "detail": refusal.get("detail"),
            }
        check = plan.get("check")
        units.append({
            "id": unit_id,
            "owns": plan.get("owns", ()),
            "check": check.get("selectors", ()) if isinstance(check, Mapping) else (),
            "status": outcome.get("status", "UNKNOWN"),
            "check_result": outcome.get("check", outcome.get("check_status", "NOT_RUN")),
            "child_sends": ((outcome.get("usage") or {}).get("requests", 0)
                            if isinstance(outcome.get("usage"), Mapping) else 0),
            "changed_paths": [file.get("path") for file in outcome.get("files", ())
                              if isinstance(file, Mapping)] if _is_sequence(outcome.get("files")) else [],
            "contract": outcome.get("contract_ok"),
            "reason": outcome.get("reason"),
            "detail": outcome.get("detail"),
        })
    converted: dict[str, object] = {
        "schema": "stateframe-protected-fanout-disposition/1",
        "kind": "protected_fanout_disposition",
        "call_id": _clean(call_id, limit=160),
        "turn_index": max(0, turn_index),
        "units": units,
        "final_check": result.get("final_check", "UNKNOWN"),
        "integration": result.get("integration", "UNKNOWN"),
        "result_summary": result.get("observation", ""),
    }
    if isinstance(admission_refusal, Mapping):
        converted["admission_refusal"] = {
            "reason": admission_refusal.get("reason"),
            "detail": admission_refusal.get("detail"),
            "hint": admission_refusal.get("hint"),
        }
    converted["units"] = _unit_rows(converted)
    converted["remaining"] = _initial_remaining(converted, converted["units"])
    return converted


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_utf8(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    if limit <= 3:
        return "." * max(0, limit)
    prefix = raw[:limit - 3].decode("utf-8", errors="ignore")
    return prefix + "..."


def _unit_line(unit: Mapping[str, object]) -> str:
    owns = ",".join(_paths(unit.get("owns"))) or "none"
    checks = ",".join(_selectors(unit)) or "NOT_RUN"
    changed = ",".join(_paths(unit.get("changed_paths"))) or "none"
    contract = unit.get("contract")
    contract_text = "True" if contract is True else "False" if contract is False else "None"
    return (
        f"subframe={_clean(unit.get('id'), limit=120)}; owns={_truncate_utf8(owns, 180)}; "
        f"check={_truncate_utf8(checks, 180)}; status={_clean(unit.get('status'), limit=48)}; "
        f"check_result={_clean(unit.get('check_result'), limit=32)}; "
        f"child_sends={int(unit.get('child_sends', 0))}; changed_paths={_truncate_utf8(changed, 180)}; "
        f"contract={contract_text}"
    )


def _failure_for(unit_id: str, disposition: Mapping[str, object]) -> Mapping[str, object] | None:
    single = disposition.get("failure")
    if isinstance(single, Mapping):
        if single.get("unit_id") == unit_id:
            return single
        if single.get("unit_id") is None:
            units = _unit_rows(disposition)
            if _initial_remaining(disposition, units) == [unit_id]:
                return single
    failures = disposition.get("failures")
    if _is_sequence(failures):
        for failure in failures:
            if isinstance(failure, Mapping) and failure.get("unit_id") == unit_id:
                return failure
    return None


def _failure_lines(unit: Mapping[str, object], failure: Mapping[str, object] | None) -> list[str]:
    if failure is None:
        reason = unit.get("reason")
        if not reason and unit.get("check_result") == "FAIL":
            reason = "subframe check failed"
        if not reason:
            return []
        path = next(iter(_selectors(unit)), next(iter(_paths(unit.get("owns"))), "unknown"))
        return [
            f"failure={_truncate_utf8(_clean(reason, limit=200), 200)}; location={path}; "
            f"expected=PASS; observed={_clean(unit.get('status'), limit=48)}",
            "failure_assertion=check result did not pass",
        ]
    nodeid = _clean(failure.get("nodeid", failure.get("failure", "unknown")), limit=260)
    location = _clean(failure.get("location", "unknown"), limit=180)
    expected = _clean(failure.get("expected", "unknown"), limit=100)
    observed = _clean(failure.get("observed", "unknown"), limit=100)
    assertion = _clean(failure.get("assertion", "check reported failure"), limit=300)
    return [
        f"failure={nodeid}; location={location}; expected={expected}; observed={observed}",
        f"failure_assertion={assertion}",
    ]


def _reason_code(reason: object) -> tuple[str, str]:
    text = _clean(reason, limit=500, replace_semicolons=False)
    code, separator, detail = text.partition(":")
    if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code):
        return code, _clean(detail, limit=200, replace_semicolons=False) if separator else ""
    return "REFUSED", text


def _fit_section(
        fixed: list[str], unit_lines: list[tuple[bool, str]], cap: int,
) -> tuple[str, int]:
    rows = list(unit_lines)
    omitted = 0
    while True:
        lines = [fixed[0], *(line for _required, line in rows)]
        if omitted:
            lines.append(f"subframes_truncated={omitted}")
        lines.extend(fixed[1:])
        text = "\n".join(lines)
        if _bytes(text) <= cap:
            return text, omitted
        # Preserve unresolved-unit detail first; compact committed rows are the
        # first section data retired when a caller supplies a smaller envelope.
        candidate = next((i for i in range(len(rows) - 1, -1, -1) if not rows[i][0]), None)
        if candidate is None:
            if rows:
                return "", omitted
            candidate = len(rows) - 1 if rows else None
        if candidate is None:
            compact_fixed = [_truncate_utf8(line, max(1, cap // max(1, len(fixed)))) for line in fixed]
            text = "\n".join(compact_fixed)
            if _bytes(text) <= cap:
                return text, omitted
            return "", omitted
        rows.pop(candidate)
        omitted += 1


def _derive_remaining(
        units: list[dict[str, object]], disposition: Mapping[str, object],
) -> tuple[list[str], int | None, bool]:
    initial = _initial_remaining(disposition, units)
    lifecycle = disposition.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
    passed = lifecycle.get("passed_checks", {})
    passed = passed if isinstance(passed, Mapping) else {}
    fanout_turn = lifecycle.get("fanout_turn_index", disposition.get("turn_index", 0))
    current_turn = lifecycle.get("current_turn_index", fanout_turn)
    fanout_turn = fanout_turn if type(fanout_turn) is int else 0
    current_turn = current_turn if type(current_turn) is int else fanout_turn
    if lifecycle.get("closed") is True:
        return [], None, True
    resolved: dict[str, int] = {}
    active = []
    by_id = {str(unit["id"]): unit for unit in units}
    for unit_id in initial:
        unit = by_id[unit_id]
        selectors = _selectors(unit)
        pass_turns = []
        for selector in selectors:
            turn = passed.get(selector)
            if type(turn) is int and turn > fanout_turn:
                pass_turns.append(turn)
        if selectors and len(pass_turns) == len(selectors):
            resolved[unit_id] = max(pass_turns)
            unit["check_result"] = "PASS"
        else:
            active.append(unit_id)
    if active:
        return active, None, False
    if initial:
        verified_turn = max(resolved.values(), default=fanout_turn)
    else:
        verified_turn = fanout_turn
    if current_turn > verified_turn:
        return [], verified_turn, True
    return [], verified_turn, False


def _source_revision_and_pin(
        disposition: Mapping[str, object], units: Sequence[Mapping[str, object]],
        remaining: Sequence[str], root_value: str | Path | None,
) -> tuple[str, str, str, list[dict[str, object]]]:
    from kernel_fanout_pin import source_revision_and_pin

    return source_revision_and_pin(disposition, units, remaining, root_value)



def render(
        disposition: dict[str, object], workspace: str | Path | None, budget: int,
) -> tuple[str, str, list[dict[str, object]]]:
    """Return the typed section, current remaining-unit windows, and journal events."""
    if not isinstance(disposition, dict):
        raise FanoutContinuationError("fan-out disposition must be an object")
    if type(budget) is not int or budget < 0:
        raise FanoutContinuationError("fan-out continuation budget must be a non-negative byte count")
    cap = min(MAX_SECTION_BYTES, budget)
    units = _unit_rows(disposition)
    remaining, verified_turn, retired = _derive_remaining(units, disposition)
    events: list[dict[str, object]] = []
    if retired:
        return "", "", [{"kind": "fanout_continuation_retired",
                          "call_id": _clean(disposition.get("call_id"), limit=160),
                          "reason": "closed_or_verified_turn_consumed"}]
    if cap < 128:
        return "", "", [{"kind": "fanout_continuation_omitted", "reason": "section_budget_too_small",
                          "budget_bytes": cap}]

    revision_line = ""
    excerpt = ""
    pin = ""
    failures = []
    if remaining:
        revision_line, excerpt, pin, pin_events = _source_revision_and_pin(
            disposition, units, remaining, workspace,
        )
        events.extend(pin_events)
        for unit_id in remaining:
            unit = next(row for row in units if row.get("id") == unit_id)
            failures.extend(_failure_lines(unit, _failure_for(unit_id, disposition)))

    fixed = ["[FANOUT CONTINUATION]"]
    for unit in units:
        for reason in ([unit.get("reason")] if unit.get("status") == "REFUSED" else []):
            code, detail = _reason_code(reason)
            fixed.append(f"refusal={code}: {detail}" if detail else f"refusal={code}")
    # Verification words come only from the recorded parent result; children finishing never implies PASS.
    final_check = _clean(disposition.get("final_check", "UNKNOWN"), limit=40).upper() or "UNKNOWN"
    integration = _clean(
        disposition.get("final_integration", disposition.get("integration", "UNKNOWN")), limit=40
    ).upper() or "UNKNOWN"
    fixed.extend([
        f"final_check={final_check}; final_integration={integration}",
        f"remaining=[{','.join(remaining)}]",
        *failures,
    ])
    if remaining:
        if revision_line:
            fixed.extend([revision_line, _truncate_utf8(excerpt, MAX_SOURCE_EXCERPT_BYTES)])
        else:
            supplied_revision = _clean(disposition.get("source_revision", "unavailable"), limit=100)
            supplied_excerpt = _clean(disposition.get("source_excerpt", "source unavailable"), limit=MAX_SOURCE_EXCERPT_BYTES)
            fixed.extend([f"source_revision={supplied_revision}; excerpt:",
                          _truncate_utf8(supplied_excerpt, MAX_SOURCE_EXCERPT_BYTES)])
        completed = [str(unit["id"]) for unit in units
                     if str(unit["id"]) not in remaining and unit.get("status") == "COMMITTED"
                     and unit.get("check_result") == "PASS"]
        check_paths = [selector for unit in units if str(unit["id"]) in remaining
                       for selector in _selectors(unit)]
        remaining_wording = ",".join(remaining)
        completed_wording = f"{'/'.join(completed)} committed+PASS, don't reopen" if completed else "don't reopen completed subframes"
        checks_wording = ",".join(check_paths) or "the remaining subframe check"
        next_action = f"repair only {remaining_wording}; {completed_wording}; run {checks_wording} then the full check"
    elif final_check == "PASS" and units:
        next_action = "all subframes committed and verified; finish unless a concrete mismatch"
    else:
        next_action = (f"subframes finished but parent final check is {final_check}, not verified; "
                       "resolve the parent final check, then run the full check before finishing")
    fixed.append(f"next_action: {_truncate_utf8(next_action, 500)}")

    required_ids = set(remaining)
    unit_lines = [(str(unit.get("id")) in required_ids, _unit_line(unit)) for unit in units]
    section, omitted = _fit_section(fixed, unit_lines, cap)
    if not section:
        events.append({"kind": "fanout_continuation_omitted", "reason": "required_facts_exceed_budget",
                       "budget_bytes": cap})
    if _bytes(section) > MAX_SECTION_BYTES:
        raise FanoutContinuationError("rendered fan-out section exceeded its hard byte cap")
    if omitted:
        events.append({"kind": "fanout_section_truncated", "omitted_units": omitted,
                       "cap_bytes": cap})
    if not remaining:
        pin = ""
    if pin and _bytes(pin) > MAX_PIN_BYTES:
        raise FanoutContinuationError("rendered fan-out pin exceeded its hard byte cap")
    return section, pin, events


def _decode_arguments(call: Mapping[str, object]) -> Mapping[str, object]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    arguments = function.get("arguments")
    if isinstance(arguments, Mapping):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _tool_name(call: Mapping[str, object]) -> str:
    function = call.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    return str(name or "").strip().rsplit(".", 1)[-1].casefold()


def _target_values(value: object) -> set[str]:
    if isinstance(value, str):
        paths = set(_paths(value))
        paths.update(path.split("::", 1)[0] for path in tuple(paths) if "::" in path)
        return paths
    if isinstance(value, Mapping):
        values: set[str] = set()
        for key in (
            "path", "paths", "target", "selector", "files", "source", "destination",
            "selectors", "public_selectors", "test_selector", "nodeids", "old_path",
            "new_path", "source_path", "destination_path",
        ):
            values.update(_target_values(value.get(key)))
        raw_args = value.get("args")
        if isinstance(raw_args, str):
            try:
                args = shlex.split(raw_args)
            except ValueError:
                args = []
            for arg in args:
                if not arg.startswith("-") and ("/" in arg or arg.endswith(".py")):
                    values.update(_target_values(arg))
        return values
    if _is_sequence(value):
        paths: set[str] = set()
        for item in value:
            paths.update(_target_values(item))
        return paths
    return set()


def _record_matches(record: Mapping[str, object], target: object) -> bool:
    candidate_paths = _target_values(target)
    selector = record.get("selector")
    scopes = record.get("selector_scope", ())
    record_paths = _target_values(selector)
    record_paths.update(_target_values(scopes))
    if candidate_paths & record_paths:
        return True
    return False


def _lifecycle_evidence(
        disposition: Mapping[str, object], messages: Sequence[Mapping[str, object]],
        action_context: Mapping[str, object] | None,
) -> dict[str, object]:
    try:
        from kernel_stateframe_action_ledger import _turn_groups
        groups, _consumed = _turn_groups(messages)
    except (ImportError, TypeError, ValueError):
        groups = []
    fanout_turn = disposition.get("turn_index")
    fanout_turn = fanout_turn if type(fanout_turn) is int else 0
    call_id = disposition.get("call_id")
    current_turn = fanout_turn
    check_calls: list[tuple[int, Mapping[str, object], Mapping[str, object]]] = []
    edit_calls: list[tuple[int, str, set[str]]] = []
    later_parallel_or_done: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping) or type(group.get("index")) is not int:
            continue
        turn = int(group["index"])
        current_turn = max(current_turn, turn)
        group_messages = group.get("messages")
        if not _is_sequence(group_messages) or not group_messages or not isinstance(group_messages[0], Mapping):
            continue
        assistant = group_messages[0]
        calls = assistant.get("tool_calls", ())
        if not _is_sequence(calls):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            tool = _tool_name(call)
            arguments = _decode_arguments(call)
            if tool == "parallel":
                if call.get("id") == call_id:
                    fanout_turn = turn
                else:
                    later_parallel_or_done.append(turn)
            elif tool == "done":
                later_parallel_or_done.append(turn)
            if tool in _CHECK_TOOLS:
                check_calls.append((turn, call, arguments))
            elif tool in _EDIT_TOOLS:
                paths = _target_values(arguments)
                edit_calls.append((turn, tool, paths or {"*"}))
            elif tool == "shell":
                edit_calls.append((turn, tool, {"*"}))

    context = action_context if isinstance(action_context, Mapping) else {}
    records = context.get("journal_receipt_records", ())
    receipts = [row for row in records if isinstance(row, Mapping) and row.get("kind") == "check_receipt"] if _is_sequence(records) else []
    receipts.sort(key=lambda row: row.get("sequence", 0) if type(row.get("sequence")) is int else 0)
    used: set[int] = set()
    outcomes: list[dict[str, object]] = []
    closed = False
    for turn, call, arguments in check_calls:
        if turn <= fanout_turn:
            # Still match pre-fan-out receipts so later records stay aligned.
            allow_pass = False
        else:
            allow_pass = True
        target = arguments.get("target")
        if target is None:
            target = arguments
        target_values = _target_values(target)
        selectors = sorted(target_values)
        match_index = next((index for index, receipt in enumerate(receipts)
                            if index not in used and _record_matches(receipt, target)), None)
        if match_index is None:
            continue
        used.add(match_index)
        receipt = receipts[match_index]
        status = str(receipt.get("status") or "").upper()
        receipt_selector = receipt.get("selector")
        if isinstance(receipt_selector, str) and receipt_selector.strip():
            selectors = sorted(set(selectors) | _target_values(receipt_selector))
        outcomes.append({
            "turn": turn,
            "selectors": selectors,
            "status": status,
            "complete": receipt.get("execution_complete") is not False,
            "allow_pass": allow_pass,
        })

    units = _unit_rows(disposition)

    def aliases(selector: str) -> set[str]:
        path = selector.split("::", 1)[0]
        return {selector, path}

    passed: dict[str, int] = {}
    for unit in units:
        owned = set(_paths(unit.get("owns")))
        for selector in _selectors(unit):
            selector_aliases = aliases(selector)
            matching = [outcome for outcome in outcomes
                        if outcome["allow_pass"] is True
                        and any(selector_aliases & aliases(value)
                                for value in outcome["selectors"]
                                if isinstance(value, str))]
            if not matching:
                continue
            latest = max(matching, key=lambda outcome: int(outcome["turn"]))
            if (latest["status"] not in {"PASS", "PASSED", "GREEN"}
                    or latest["complete"] is not True):
                continue
            check_turn = int(latest["turn"])
            edited_after = False
            for edit_turn, _tool, paths in edit_calls:
                # Calls in one assistant turn have no reliable execution order.
                if edit_turn < check_turn:
                    continue
                if "*" in paths or paths & (owned | selector_aliases):
                    edited_after = True
                    break
            if not edited_after:
                passed[selector] = check_turn

    closed = any(turn > fanout_turn for turn in later_parallel_or_done)
    return {
        "fanout_turn_index": fanout_turn,
        "current_turn_index": current_turn,
        "passed_checks": passed,
        "closed": closed,
    }


def _legacy_disposition(frame: Mapping[str, object]) -> tuple[dict[str, object] | None, bool]:
    ledger = frame.get("action_ledger")
    rows = ledger.get("unclassified_actions") if isinstance(ledger, Mapping) else None
    if not _is_sequence(rows):
        return None, False
    candidates = [row for row in rows if isinstance(row, Mapping)
                  and row.get("tool") == "parallel" and isinstance(row.get("call_id"), str)]
    if not candidates:
        return None, False
    latest = max(candidates, key=lambda row: row.get("turn_index", -1)
                 if type(row.get("turn_index")) is int else -1)
    call_id = str(latest["call_id"])
    try:
        from kernel_parallel import action_receipt_source
        result, request = action_receipt_source(call_id)
        request_json = request.to_json() if request is not None else {"units": []}
        result_json = result.to_json()
        turn = latest.get("turn_index", 0)
        return disposition_from_records(call_id, request_json, result_json,
                                        turn_index=turn if type(turn) is int else 0), False
    except (ImportError, OSError, ValueError, TypeError, RuntimeError):
        return None, True


def _find_disposition(frame: Mapping[str, object]) -> tuple[dict[str, object] | None, bool]:
    candidates: list[Mapping[str, object]] = []
    stack: list[tuple[object, str, int]] = [(frame, "frame", 0)]
    visited = 0
    while stack and visited < 12_000:
        value, parent_key, depth = stack.pop()
        visited += 1
        if depth > 8:
            continue
        if isinstance(value, Mapping):
            kind = str(value.get("kind", "")).casefold()
            has_units = _is_sequence(value.get("units", value.get("unit_dispositions", ())))
            tagged = kind in _DISPOSITION_KINDS or "fanout" in parent_key.casefold()
            if tagged and has_units and ("final_check" in value or "integration" in value):
                candidates.append(value)
            for key, child in value.items():
                key_text = str(key)
                if kind in _DISPOSITION_KINDS and isinstance(child, Mapping):
                    merged = {**dict(value), **dict(child)}
                    merged.pop(key_text, None)
                    stack.append((merged, kind, depth + 1))
                elif (key_text == "unclassified_actions" and _is_sequence(child)):
                    for row in child:
                        if isinstance(row, Mapping) and row.get("kind") in _DISPOSITION_KINDS:
                            stack.append((row, "protected_fanout_disposition", depth + 1))
                        elif isinstance(row, Mapping) and row.get("tool") == "parallel":
                            continue
                elif isinstance(child, Mapping) or _is_sequence(child):
                    stack.append((child, key_text, depth + 1))
        elif _is_sequence(value):
            for child in value:
                if isinstance(child, Mapping) or _is_sequence(child):
                    stack.append((child, parent_key, depth + 1))
    if candidates:
        chosen = max(candidates, key=lambda row: row.get("turn_index", -1)
                     if type(row.get("turn_index")) is int else -1)
        nested = chosen.get("disposition")
        if isinstance(nested, Mapping):
            chosen = {**dict(chosen), **dict(nested)}
            chosen.pop("disposition", None)
        return dict(chosen), False
    return _legacy_disposition(frame)


def attach_to_frame(
        frame: dict[str, object], action_context: Mapping[str, object] | None,
        workspace: str | Path | None, *, budget: int = MAX_SECTION_BYTES,
        semantic_messages: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Attach the latest continuation as typed frame fields; never add a chat message."""
    frame.pop("fanout_continuation", None)
    frame.pop("fanout_continuation_pin", None)
    if os.environ.get(SWITCH_NAME, "on").strip().lower() != "on":
        return []
    disposition, unavailable = _find_disposition(frame)
    if disposition is None:
        if unavailable:
            return [{"kind": "fanout_continuation_unavailable",
                     "reason": "protected_disposition_unreadable"}]
        return []
    try:
        if semantic_messages:
            lifecycle = _lifecycle_evidence(disposition, semantic_messages, action_context)
            existing = disposition.get("lifecycle")
            if isinstance(existing, Mapping):
                lifecycle = {**dict(existing), **lifecycle}
            disposition["lifecycle"] = lifecycle
        section, pin, events = render(disposition, workspace, budget)
    except FanoutContinuationError:
        return [{"kind": "fanout_continuation_unavailable",
                 "reason": "malformed_protected_disposition"}]
    if section:
        frame["fanout_continuation"] = section
    if pin:
        frame["fanout_continuation_pin"] = pin
    return events


def load_recorded_run(run_dir: str | Path) -> tuple[dict[str, object], Path]:
    from kernel_fanout_continuation_cli import load_recorded_run as load

    return load(run_dir)


def render_recorded_run(
        run_dir: str | Path, *, budget: int = MAX_SECTION_BYTES,
) -> dict[str, object]:
    from kernel_fanout_continuation_cli import render_recorded_run as render_captured

    return render_captured(run_dir, budget=budget)


if __name__ == "__main__":
    from kernel_fanout_continuation_cli import _main

    raise SystemExit(_main())
