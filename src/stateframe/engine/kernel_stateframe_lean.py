"""Compact, host-owned StateFrame presentation.

The lean arm is a presentation switch only.  It does not alter the recorded
frame, receipts, generated-artifact index, or recall APIs.  The default is
``off`` so the native ``stateframe-native/2`` payload remains untouched.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence

from kernel_verdict import (CONFLICT, SATISFIED, UNRESOLVED, VIOLATED, Verdict,
                            evaluate_required_tests)

MODE_ENV = "STATEFRAME_LEAN_RENDER"
VERSION = "stateframe-lean/1"


def lean_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return lean_mode() == "on"


def _text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _rows(value: object) -> list[dict]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _current_revision(frame: Mapping[str, object]) -> str:
    def revision_value(value: object) -> str | None:
        if isinstance(value, Mapping):
            for key in ("value", "revision", "product_revision", "current_revision",
                        "binding_revision"):
                candidate = value.get(key)
                if candidate is not None:
                    resolved = revision_value(candidate)
                    if resolved is not None:
                        return resolved
            state = value.get("revision_binding_state")
            if isinstance(state, Mapping):
                for key in ("product", "product_revision"):
                    resolved = revision_value(state.get(key))
                    if resolved is not None:
                        return resolved
            for key in ("binding", "after_binding", "current_binding"):
                resolved = revision_value(value.get(key))
                if resolved is not None:
                    return resolved
            return None
        return None if value is None else str(value)

    # The completion contract is the authority for the checked revision when it
    # records one.  The live projection exposes the current product binding in
    # world.revision_binding_state; use that before file-level artifact hashes.
    contract = frame.get("completion_contract")
    checks = (contract.get("current_public_checks")
              if isinstance(contract, Mapping) else None)
    for value in (
            frame.get("current_revision"), frame.get("product_revision"),
            frame.get("world"), frame.get("binding"), frame.get("current_binding"),
            frame.get("frame_series"), checks, frame.get("verification"),
            frame.get("revision")):
        resolved = revision_value(value)
        if resolved is not None:
            return resolved
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping) and len(revisions) == 1:
        return str(next(iter(revisions.values())))
    return "UNKNOWN"


def _latest_test(frame: Mapping[str, object]) -> dict | None:
    rows: list[dict] = []
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts"):
        rows.extend(_rows(frame.get(key)))
    latest = frame.get("latest_receipts")
    if isinstance(latest, Mapping):
        rows.extend(_rows(latest))
    last = frame.get("last_action")
    if isinstance(last, Mapping) and (
            str(last.get("tool", last.get("kind", ""))).lower() in
            {"test", "run_tests", "pytest", "verification"}):
        rows.append(dict(last))
    from kernel_frame_edit_integrity import collapse_check_results, enabled as frame_integrity_enabled
    if frame_integrity_enabled():
        rows = collapse_check_results(rows)
    return rows[-1] if rows else None


def _status(row: object) -> str:
    if row is None:
        return "NOT_RUN"
    if not isinstance(row, Mapping):
        return str(row).upper()
    for key in ("result_class", "result", "status", "outcome", "verdict"):
        if row.get(key) is not None:
            return str(row[key]).upper()
    exit_code = row.get("exit_code", row.get("exit"))
    return "GREEN" if exit_code == 0 else "RED" if exit_code == 1 else "UNKNOWN"


def _selector(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "UNKNOWN"
    for key in ("target", "suite", "selector", "test"):
        if row.get(key):
            return _text(row[key], 180)
    values = row.get("selectors")
    if isinstance(values, str):
        return _text(values, 180)
    if isinstance(values, Sequence):
        return _text(", ".join(str(item) for item in values), 180)
    return "UNKNOWN"


def _revision(row: object) -> str:
    if row is None:
        return "UNKNOWN"
    if not isinstance(row, Mapping):
        return str(row)
    for key in ("revision", "current_revision", "product_revision", "binding_revision"):
        if row.get(key) is not None:
            value = row[key]
            if isinstance(value, Mapping):
                value = value.get("value", value.get("revision", value.get("product_revision")))
            if value is not None:
                return str(value)
    for key in ("binding", "after_binding", "current_binding", "revision_binding_state"):
        binding = row.get(key)
        if not isinstance(binding, Mapping):
            continue
        product = binding.get("product_revision", binding.get("product"))
        if isinstance(product, Mapping):
            product = product.get("value", product.get("revision"))
        if product is not None:
            return str(product)
    return "UNKNOWN"


def _completion_authority(frame: Mapping[str, object]) -> dict[str, object]:
    """Return the normal renderer's public-completion authority and bindings."""
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        submission = terminal.get("submission")
        ready = (str(terminal.get("state", "")).upper() == "READY"
                 or (isinstance(submission, Mapping)
                     and str(submission.get("state", "")).upper() == "READY"))
        checks = {
            "status": terminal.get("public_check_status", "BLOCKED"),
            "selectors": terminal.get("selectors", ()),
            "revision": terminal.get("current_revision"),
        }
        return {
            "checks": checks,
            "required": terminal.get("selectors", ()),
            "stop_action": terminal.get("next_feasible_action") or "done",
            "authoritative": True,
            "verdict": None,
            "terminal_ready": ready,
        }
    from kernel_sweep_fixes import completion_projection, enabled as sweep_fixes_enabled
    if sweep_fixes_enabled():
        contract = frame.get("completion_contract")
        if isinstance(contract, Mapping):
            checks = contract.get("current_public_checks", frame.get("verification"))
            required = contract.get("required_public_checks", frame.get("required_public_checks", ()))
            stop_action = contract.get("stop_action", frame.get("stop_action", "done"))
        else:
            checks = frame.get("verification")
            required = frame.get("required_public_checks", ())
            stop_action = frame.get("stop_action", "done")
        return {"checks": checks, "required": required, "stop_action": stop_action,
                "authoritative": True, "verdict": None,
                "completion": completion_projection(frame)}
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping):
        has_checks = "current_public_checks" in contract
        checks = contract.get("current_public_checks")
        if not has_checks:
            checks = frame.get("verification")
        required = (contract.get("required_public_checks")
                    if "required_public_checks" in contract
                    else frame.get("required_public_checks", ()))
        stop_action = (contract.get("stop_action")
                       if "stop_action" in contract else frame.get("stop_action", "done"))
        events = []
        for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
            events.extend(_rows(frame.get(key)))
        verdict = _frame_verdict(frame, required)
        return {"checks": checks, "required": required, "stop_action": stop_action,
                "authoritative": has_checks or checks is not None,
                "verdict": verdict}

    checks = frame.get("verification")
    required = frame.get("required_public_checks", ())
    events = []
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
        events.extend(_rows(frame.get(key)))
    verdict = _frame_verdict(frame, required)
    return {"checks": checks, "required": required,
            "stop_action": frame.get("stop_action", "done"),
            "authoritative": checks is not None or frame.get("valid_product_verification") is True,
            "verdict": verdict}


def _frame_verdict(frame: Mapping[str, object], required: object) -> Verdict:
    """Use raw current receipts, or the rich host projection when receipts are omitted."""
    events = []
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
        events.extend(_rows(frame.get(key)))
    if events:
        return evaluate_required_tests(frame, required, events)
    projected = Verdict.from_mapping(frame.get("completion_verdict"))
    if projected is not None:
        return projected
    return evaluate_required_tests(frame, required, events)


def _authority_selector(authority: Mapping[str, object]) -> str:
    checks = authority.get("checks")
    if isinstance(checks, Mapping):
        for key in ("covered_selectors", "selectors", "selector", "target", "suite", "test"):
            if checks.get(key):
                return _selector({"selectors": checks[key]})
    return _selector({"selectors": authority.get("required", ())})


def _authority_revision(frame: Mapping[str, object], authority: Mapping[str, object]) -> str:
    checked = _revision(authority.get("checks"))
    return checked if checked != "UNKNOWN" else _current_revision(frame)


def _required_checks(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value:
        return "[]"
    return _selector({"selectors": value})


def _product_revision(row: Mapping[str, object]) -> str | None:
    revision = _revision(row)
    return None if revision == "UNKNOWN" else revision


def _witness_is_current(row: Mapping[str, object], current_revision: str) -> bool:
    """Reject receipt witnesses that cannot be bound to the current product."""
    if current_revision == "UNKNOWN":
        return True
    revision = _product_revision(row)
    if revision is not None:
        return revision == current_revision
    relevance = str(row.get("relevance", "")).lower()
    return relevance == "same_workspace_revision"


def _is_current_green(frame: Mapping[str, object], row: Mapping[str, object] | None) -> bool:
    if _status(row) not in {"GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED", "REUSED_VERIFICATION", "OK"}:
        return False
    checked = _revision(row)
    current = _current_revision(frame)
    return checked == "UNKNOWN" or current == "UNKNOWN" or checked == current


def _certification(frame: Mapping[str, object], row: Mapping[str, object] | None) -> str:
    if _is_current_green(frame, row) or frame.get("valid_product_verification") is True:
        return "GREEN_CURRENT"
    if _status(row) in {"RED", "FAIL", "FAILED", "ERROR"}:
        return "RED_CURRENT"
    return "UNVERIFIED"


def _authority_receipt(frame: Mapping[str, object], authority: Mapping[str, object]) -> dict | None:
    """Project one receipt from the authority that feeds the normal frame."""
    from kernel_sweep_fixes import certification_for_completion, enabled as sweep_fixes_enabled
    if sweep_fixes_enabled():
        completion = authority.get("completion", {})
        return {"status": certification_for_completion(completion),
                "selector": _authority_selector(authority),
                "revision": _authority_revision(frame, authority)}
    if authority.get("authoritative"):
        checks = authority.get("checks")
        status = _status(checks)
        if status == "NOT_RUN" and frame.get("valid_product_verification") is True:
            status = "CERTIFIED"
        return {"status": status, "selector": _authority_selector(authority),
                "revision": _authority_revision(frame, authority)}
    return _latest_test(frame)


def _authority_certification(frame: Mapping[str, object], authority: Mapping[str, object],
                             receipt: Mapping[str, object] | None) -> str:
    """Derive certification from the same current-check authority as the receipt."""
    if isinstance(frame.get("terminal_status"), Mapping):
        status = str(frame["terminal_status"].get("public_check_status", "")).upper()
        if status == "PASS":
            return "GREEN_CURRENT"
        if status == "FAIL":
            return "RED_CURRENT"
    from kernel_sweep_fixes import certification_for_completion, enabled as sweep_fixes_enabled
    if sweep_fixes_enabled():
        return certification_for_completion(authority.get("completion", {}))
    verdict = authority.get("verdict")
    state = getattr(verdict, "state", None)
    if state == SATISFIED:
        return "GREEN_CURRENT"
    if state == VIOLATED:
        return "RED_CURRENT"
    if state == CONFLICT:
        return "CONFLICT_CURRENT"
    if state == UNRESOLVED:
        return "UNVERIFIED"
    return "UNVERIFIED"


def _last_mutation(frame: Mapping[str, object]) -> str:
    action = frame.get("last_action")
    if isinstance(action, Mapping):
        return _text({key: action.get(key) for key in ("tool", "parameters", "result")
                      if action.get(key) is not None}, 360)
    changes = frame.get("recent_changes")
    rows = _rows(changes)
    return _text(rows[-1], 360) if rows else "none"


def _conclusions(frame: Mapping[str, object], row: Mapping[str, object] | None) -> list[str]:
    lines: list[str] = []
    if row is not None:
        lines.append(f"public check {_status(row)} for {_selector(row)} at revision {_revision(row)}")
    findings = frame.get("host_observed_findings", frame.get("known_violations", ()))
    if isinstance(findings, str):
        findings = [findings]
    if isinstance(findings, Sequence):
        lines.extend("host finding: " + _text(item, 200) for item in findings if str(item).strip())
    phase = frame.get("phase")
    if phase:
        lines.append("phase: " + _text(phase, 180))
    return list(dict.fromkeys(lines))[:4]


def _witnessed_defects(frame: Mapping[str, object]) -> list[str]:
    """Keep only executable failure witnesses bound to the current product."""
    import re

    current_revision = _current_revision(frame)
    rows: list[Mapping[str, object]] = []
    for key in ("recent_test_feedback", "test_receipts", "probe_receipts",
                "recent_negative_actions"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            value = list(value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows.extend(item for item in value if isinstance(item, Mapping))

    details: list[str] = []
    for row in rows:
        if not _witness_is_current(row, current_revision):
            continue
        raw = row.get("feedback", row.get("observation", row.get("detail", "")))
        raw_feedback = str(raw or "")
        feedback = " ".join(raw_feedback.split())
        status = _status(row)
        if not feedback or not (status in {"RED", "FAIL", "FAILED", "ERROR", "REJECTED"}
                                or "TESTS FAILED" in feedback
                                or "pytest exit 1" in feedback):
            continue
        exception_match = None
        for line in raw_feedback.splitlines():
            match = re.search(
                r"\b([A-Za-z_]\w*(?:Error|Exception|Unavailable|Failure)):\s*(.+)$",
                line,
            )
            if match:
                exception_match = match
                break
        location_matches = re.findall(
            r"([A-Za-z0-9_./-]+\.py):(\d+)(?::[^\s]*)?", raw_feedback
        )
        obligations = []
        for line in raw_feedback.splitlines():
            match = re.search(r"(?:assert\s+.+|HOW-COUNTED:\s*.+)", line)
            if match:
                obligations.append(match.group(0).strip())
        if exception_match and location_matches and obligations:
            exception = f"{exception_match.group(1)}: {exception_match.group(2).strip()}"
            location = f"{location_matches[-1][0]}:{location_matches[-1][1]}"
            obligation = " / ".join(dict.fromkeys(item.strip() for item in obligations))
            details.append(f"exception={exception}; file:line={location}; obligation={obligation}")
        else:
            # Preserve the complete witness if a future runner changes its
            # traceback shape; omission is less safe than a slightly longer row.
            details.append("feedback=" + feedback)

    action = frame.get("last_action")
    if (isinstance(action, Mapping)
            and _witness_is_current(action, current_revision)
            and str(action.get("result", "")).upper() in {
            "RED", "FAIL", "FAILED", "ERROR", "REJECTED"}):
        feedback = " ".join(str(action.get("feedback", "")).split())
        if feedback:
            details.append("feedback=" + feedback)

    explicit = frame.get("witnessed_defects")
    if isinstance(explicit, str) and explicit.strip():
        details.append("witness=" + " ".join(explicit.split()))
    elif isinstance(explicit, Mapping):
        if _witness_is_current(explicit, current_revision):
            value = explicit.get("detail", explicit.get("feedback", explicit))
            if str(value).strip():
                details.append("witness=" + " ".join(str(value).split()))
    elif isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        for item in explicit:
            if isinstance(item, Mapping) and not _witness_is_current(item, current_revision):
                continue
            if str(item).strip():
                details.append("witness=" + " ".join(str(item).split()))
    return list(dict.fromkeys(details))


def render_lean_state(frame: Mapping[str, object], *, continuation: Mapping[str, object] | None = None) -> str:
    """Render exactly five compact sections from host-owned state."""
    if not isinstance(frame, Mapping):
        raise TypeError("lean state requires a frame mapping")
    from kernel_frame_edit_integrity import enabled as frame_integrity_enabled
    from kernel_completion_status import (
        completion_status_from_frame, enabled as blocker_render_enabled,
        render_completion_status,
    )
    blocker_status = (completion_status_from_frame(frame)
                      if blocker_render_enabled() else None)
    if blocker_status is not None:
        active = continuation.get("active") if isinstance(continuation, Mapping) else None
        work = frame.get("objective", frame.get("msg", ""))
        if isinstance(work, Mapping):
            work = work.get("public_requirements", work.get("objective", ""))
        if isinstance(active, Mapping) and active.get("pending_decision"):
            work = f"{work} | next={active['pending_decision']}"
        lines = [
            "[StateFrame]",
            f"lean_render: {VERSION}",
            "work: " + _text(work or "UNKNOWN"),
            ("last_mutation: " + _last_mutation(frame)
             if blocker_status.state == "ACCEPTABLE"
             else "last_mutation: suppressed by completion status"),
            render_completion_status(blocker_status),
        ]
        return "\n".join(lines)
    authority = _completion_authority(frame)
    row = _authority_receipt(frame, authority)
    active = continuation.get("active") if isinstance(continuation, Mapping) else None
    work = frame.get("objective", frame.get("msg", ""))
    if isinstance(work, Mapping):
        work = work.get("public_requirements", work.get("objective", ""))
    if isinstance(active, Mapping) and active.get("pending_decision"):
        work = f"{work} | next={active['pending_decision']}"
    cert = _authority_certification(frame, authority, row)
    from kernel_sweep_fixes import (
        admission_for_completion, completion_projection,
        enabled as sweep_fixes_enabled,
    )
    if sweep_fixes_enabled():
        atomic = completion_projection(frame)
        completion = admission_for_completion(atomic)
    else:
        atomic = None
        completion = "ADMISSIBLE" if cert == "GREEN_CURRENT" else "BLOCKED"
    stop_action = _text(authority.get("stop_action") or "UNKNOWN", 120)
    required_checks = _required_checks(authority.get("required", ()))
    # The first line MUST be exactly "[StateFrame]": the study request preflight and the
    # runtime message classifiers key on that prefix. A custom header here rejected every
    # lean request before the provider (batch 112, ERR:ValueError, 0 responses).
    lines = [
        "[StateFrame]",
        f"lean_render: {VERSION}",
        "work: " + _text(work or "UNKNOWN"),
        "last_mutation: " + _last_mutation(frame),
        "public_receipt: " + ("none" if row is None else
            f"status={_status(row)} selector={_selector(row)} revision={_revision(row)}"),
        "conclusions:",
    ]
    if frame_integrity_enabled() and isinstance(row, Mapping):
        superseded = row.get("superseded")
        if isinstance(superseded, Sequence) and not isinstance(superseded, (str, bytes)):
            details = ", ".join(
                f"{item.get('status', 'UNKNOWN')}@{item.get('revision', 'UNKNOWN')}"
                for item in superseded if isinstance(item, Mapping)
            )
            if details:
                lines.append("superseded_results: " + details)
    lines.extend("- " + line for line in (_conclusions(frame, row) or ["none recorded"]))
    defects = _witnessed_defects(frame)
    if defects:
        lines.append("witnessed_defects:")
        lines.extend("- " + defect for defect in defects)
    lines.extend(("completion:", f"certification_status={cert}", f"admission={completion}",
                  f"stop_action={stop_action}", f"required_checks={required_checks}"))
    if atomic is not None:
        lines.append("completion_outcome=" + str(atomic["outcome"]))
    if frame.get("green_lock_message"):
        lines.append(str(frame["green_lock_message"]))
    from kernel_terminal_surface_unify import ADVISORY_OBLIGATIONS_FIELD
    terminal = frame.get("terminal_status")
    coherent_terminal = (isinstance(terminal, Mapping)
                         and terminal.get("schema") == "stateframe-terminal-status/1")
    advisory = frame.get(ADVISORY_OBLIGATIONS_FIELD) if coherent_terminal else None
    if isinstance(advisory, Mapping):
        advisory = list(advisory.values())
    if isinstance(advisory, Sequence) and not isinstance(advisory, (str, bytes)) and advisory:
        lines.append(ADVISORY_OBLIGATIONS_FIELD + ": " + json.dumps(
            list(advisory), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


def render_requested_generated(generated: object) -> str | None:
    """Return only an explicitly recalled proposal; catalogs stay retrievable."""
    requested = getattr(generated, "requested", None)
    if not isinstance(requested, Mapping):
        return None
    result = copy.deepcopy(dict(requested))
    index = getattr(generated, "index", None)
    reference = result.get("reference")
    if index is not None and reference:
        try:
            result.update(index.metadata(reference))
        except (KeyError, TypeError, ValueError):
            pass
    renderer = getattr(type(generated), "render", None)
    return renderer(result) if callable(renderer) else None


def sanitize_history(
    messages: Sequence[Mapping[str, object]], *,
    preserve_assistant_tool_messages: bool = False,
) -> list[dict]:
    """Remove superseded projections, with an opt-in native-history escape hatch.

    The default filtering is unchanged.  The observation-masking history arm
    uses the opt-in so content-based cleanup cannot delete a native assistant
    or tool message before its call/result pairing is validated.
    """
    dropped_prefixes = (
        "[Continuation]", "[WORKING_VIEW", "[STATEFRAME UNIFIED",
        "[STATEFRAME RECEIPTS", "[StateFrame v2 S1", "[BOOT CAPSULE",
        "[StateFrame advisory sidecar]", "[StateFrame]",
    )
    result: list[dict] = []
    for index, raw in enumerate(messages):
        message = copy.deepcopy(dict(raw))
        content = str(message.get("content") or "")
        if (preserve_assistant_tool_messages
                and message.get("role") in {"assistant", "tool"}):
            result.append(message)
            continue
        if index >= 2 and (content.startswith(dropped_prefixes)
                           or "public_checks_certified" in content):
            continue
        # Generated catalogs are intentionally not history state.  A recalled
        # result is a tool result and remains available; the recall tool itself
        # is the durable retrieval path for older proposals.
        if index >= 2 and content.startswith("[Generated artifact catalog"):
            continue
        result.append(message)
    return result


__all__ = ["MODE_ENV", "VERSION", "enabled", "lean_mode", "render_lean_state",
           "render_requested_generated", "sanitize_history"]
