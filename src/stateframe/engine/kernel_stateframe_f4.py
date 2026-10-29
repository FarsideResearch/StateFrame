"""Opt-in F4 StateFrame projection and presentation helpers.

F4 owns only the model-facing presentation.  The host frame remains complete in
the journal, while this module projects one bounded view with explicit
revision/evidence binding, a defeasible decision handoff, and provenance-gated
native exchanges.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from kernel_context import token_estimate
from kernel_stateframe_envelope import STATE_HEADER, serialize_stateframe_payload
from kernel_evictfix_mode import enabled as evictfix_enabled


MODE_ENV = "STATEFRAME_F4"
SOURCE_PAGE_HEADER_V2_ENV = "STATEFRAME_SOURCE_PAGE_HEADER_V2"
KEEP_LATEST_BATCH_ENV = "STATEFRAME_KEEP_LATEST_BATCH"
SOURCE_WORKING_SET_ENV = "STATEFRAME_SOURCE_WORKING_SET"
SCHEMA = "stateframe-f4/1"
CLAUSE_HEADER = "[Get-sure state]"
HANDOFF_HEADER = "[Decision handoff]"
MAX_HANDOFF_TOKENS = 250
MIN_HANDOFF_TOKENS = 100
REAL_EXCHANGE_SCHEMA = "stateframe-f4/real-exchanges/1"
MAX_RETAINED_PROVIDER_BATCHES = 2
SOURCE_WORKING_SET_SCHEMA = "stateframe-f4/source-working-set/1"
MAX_SOURCE_WORKING_SET_BYTES = 16_384
MAX_SOURCE_WORKING_SET_PAGES = 64
MAX_SOURCE_WORKING_SET_EVICTIONS = 8
MAX_PENDING_EDIT_TEXT_BYTES = 4_096

_SOURCE_PAGE_HEADER = re.compile(
    r"^\[read page\]\s+path=(.*?)\s+revision=(sha256:[0-9a-f]{64})\s+"
    r"lines=(\d+)-(\d+)/(\d+)"
    r"(?:\s+requested=\S+)?(?:\s+effective=\S+)?(?:\s+clip=\S+)?"
    r"\s+chars=(\d+)(?:\s+bytes=\d+)?\s+"
    r"truncated=(true|false)\s+decoded_with_replacement=(true|false)$",
    re.IGNORECASE,
)
_SOURCE_PAGE_HEADER_LEGACY = re.compile(
    r"^\[read page\]\s+path=(.*?)\s+revision=(sha256:[0-9a-f]{64})\s+"
    r"lines=(\d+)-(\d+)/(\d+)\s+chars=(\d+)\s+"
    r"truncated=(true|false)\s+decoded_with_replacement=(true|false)$",
    re.IGNORECASE,
)
_SOURCE_CITATION_FIELDS = frozenset({
    "path", "file", "filename", "target", "target_path", "query", "search", "symbol",
})
_EDIT_TOOL_NAMES = frozenset({"edit", "str_replace", "apply_patch", "edit_file", "write_file"})

_PUBLIC_CHECK_PATHS = (
    ("verification",),
    ("valid_product_verification",),
    ("recent_change_verification",),
    ("certification_status",),
    ("world", "certification"),
    ("completion_verdict",),
    ("resume", "readiness", "verification"),
    ("readiness", "green"),
    ("readiness", "public_check_status"),
    ("obligations", "current_public_checks"),
    ("completion_contract", "current_public_checks"),
    ("terminal_status", "public_check_status"),
    ("terminal_status", "verdict_state"),
    ("terminal_status", "green"),
    ("public_check_status",),
    ("current_public_checks",),
)
_DONE_PERMISSION_PATHS = (
    ("done_permission",),
    ("done_allowed",),
    ("readiness", "done_allowed"),
    ("readiness", "dispositions", "done"),
    ("current_state", "permissions", "done"),
)
_COMPLETION_PATHS = (
    ("atomic_completion",),
    ("completion_contract", "atomic_completion"),
    ("completion_decision",),
    ("completion_reducer_decision",),
    ("completion_status",),
    ("completion_authority",),
)
_SUBMISSION_PATHS = (
    ("submission",),
    ("terminal_status", "submission"),
    ("completion_submission_status",),
    ("completion_contract", "completion_submission_status"),
    ("current_state", "submission"),
)
_ALIAS_FIELDS = frozenset({
    "verification", "valid_product_verification", "completion_verdict", "readiness",
    "terminal_status", "completion_authority", "certification_status",
    "recent_change_verification", "completion_status", "completion_contract",
    "completion_reducer_decision", "completion_decision", "atomic_completion",
    "current_public_checks", "public_check_status", "green", "ready", "ready_to_submit",
    "done_allowed", "requirements_review_status", "completion_submission_status",
})

_PASSED = frozenset({
    "PASS", "PASSED", "GREEN", "GREEN_CURRENT", "SATISFIED", "CURRENT", "VERIFIED",
    "CERTIFIED", "TRUE", "OK", "SUCCESS",
})
_FAILED = frozenset({
    "FAIL", "FAILED", "RED", "VIOLATED", "FALSE", "REFUSED", "NEEDS_WORK", "BLOCKED",
})
_UNKNOWN = frozenset({
    "UNKNOWN", "UNVERIFIED", "UNRESOLVED", "NOT_OBSERVED", "PENDING", "REQUIRED",
    "MISSING", "NONE", "NULL", "",
})

# The fixed meaning of each StateFrame predicate.  Rendered per predicate in the model view by
# default; STATEFRAME_PROMPT_COMPACT (kernel_prompt_compact) drops the per-request copies and puts
# this glossary in the system prompt once.  One source for both renderings.
F4_PREDICATE_GLOSSARY: dict[str, str] = {
    "public_checks": ("the named public checks ran on this revision and establish only their "
                      "public-check result; they do not establish permission or accepted completion"),
    "done_permission": ("the active dispatch gate permits or denies the done action on this revision; "
                        "it is not a test verdict or completion seal"),
    "completion": ("the host completion reducer or acceptance authority records whether the "
                   "candidate was accepted; it does not retroactively change check history"),
    "submission": ("the host submission projection records terminal submission status only; "
                   "it is not a completion seal"),
    "blockers": "host-recorded blockers remain open on this revision and explain any denied gate",
    "advisories": "host-recorded advisories are visible but do not by themselves grant permission",
}


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strictly validated F4 switch; the default is byte-preserving off."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def source_page_header_v2_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict source-page header parser switch, defaulting to legacy behavior."""
    source = os.environ if environ is None else environ
    value = str(source.get(SOURCE_PAGE_HEADER_V2_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{SOURCE_PAGE_HEADER_V2_ENV} must be off or on")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    for method in ("to_dict", "as_dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            try:
                result = converter()
            except (AttributeError, TypeError, ValueError):
                result = None
            if isinstance(result, Mapping):
                return result
    return {}


def _at(root: Mapping[str, object], path: Sequence[str]) -> object:
    value: object = root
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _copy_json(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _strings(value: object, *, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _refs(value: object, *, limit: int = 8) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    result: list[str] = []
    for key in ("supporting_evidence", "evidence_refs", "receipt_refs", "receipt_ids",
                "receipt_id", "reference", "fact_ids", "adverse_ref"):
        item = value.get(key)
        values = [item] if isinstance(item, str) else item
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for ref in values:
                text = str(ref).strip()
                if text and text not in result:
                    result.append(text)
                if len(result) >= limit:
                    return result
    return result


def _revision(frame: Mapping[str, object], path: str | None = None) -> str:
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping):
        if path and revisions.get(path) is not None:
            return str(revisions[path])
        values = [str(value) for value in revisions.values() if value not in (None, "")]
        if len(values) == 1:
            return values[0]
        if values:
            return "|".join(sorted(dict.fromkeys(values)))
    for key in ("revision", "current_revision", "workspace_revision", "source_revision"):
        value = frame.get(key)
        if value not in (None, ""):
            return str(value)
    current = _mapping(frame.get("current_state"))
    for key in ("revision", "current_revision", "workspace_revision"):
        value = current.get(key)
        if value not in (None, ""):
            return str(value)
    envelope = _mapping(frame.get("host_envelope"))
    return str(envelope.get("revision") or "UNKNOWN")


def _status(value: object) -> str:
    if isinstance(value, bool):
        return "PASSED" if value else "FAILED"
    if isinstance(value, Mapping):
        if value.get("accepted") is True and any(
                key in value for key in ("outcome", "state", "status")):
            # Completion acceptance is not itself a public-check result.
            value = value.get("outcome", value.get("state", value.get("status")))
        else:
            for key in ("state", "status", "result", "verdict", "outcome"):
                if key in value:
                    value = value[key]
                    break
            else:
                return "UNKNOWN"
    text = str(value or "").strip().upper().replace("-", "_")
    if text in _PASSED:
        return "PASSED"
    if text in _FAILED:
        return "FAILED"
    if text in _UNKNOWN:
        return "UNKNOWN"
    return "UNKNOWN"


def _permission(value: object) -> str:
    if isinstance(value, bool):
        return "ALLOW" if value else "DENY"
    if isinstance(value, Mapping):
        if "done" in value and len(value) == 1:
            value = value["done"]
        elif "allowed" in value:
            value = value["allowed"]
        elif "state" in value:
            value = value["state"]
    text = str(value or "").strip().upper()
    if text in {"ALLOW", "ALLOWED", "TRUE", "READY", "PERMIT", "PERMITTED"}:
        return "ALLOW"
    if text in {"DENY", "DENIED", "FALSE", "BLOCK", "BLOCKED", "REFUSE", "REFUSED"}:
        return "DENY"
    if text in {"CONFLICT", "AMBIGUOUS"}:
        return "CONFLICT"
    return "UNKNOWN"


def _completion(value: object) -> str:
    row = _mapping(value)
    if row:
        if row.get("accepted") is True:
            return "ACCEPTED"
        if row.get("accepted") is False and any(
                key in row for key in ("outcome", "state", "status", "reason_codes")):
            return "NOT_ACCEPTED"
        if row.get("seal") is not None:
            return "ACCEPTED"
        value = row.get("outcome", row.get("state", row.get("status")))
    text = str(value or "").strip().upper().replace("-", "_")
    if text in {"ACCEPTED", "SEALED", "COMPLETE", "COMPLETED"}:
        return "ACCEPTED"
    if text in {"REFUSED", "NEEDS_WORK", "BLOCKED", "BASIS_CHANGED", "BOUNDED",
                "REJECTED", "NOT_ACCEPTED"}:
        return "NOT_ACCEPTED"
    if text in {"PENDING", "ACTIVE", "ACCEPTABLE", "UNKNOWN", "UNRESOLVED", ""}:
        return "UNKNOWN"
    return "UNKNOWN"


def _submission(value: object) -> str:
    row = _mapping(value)
    if row:
        value = row.get("state", row.get("outcome", row.get("status")))
    text = str(value or "UNKNOWN").strip().upper().replace("-", "_")
    return text if text in {
        "PENDING", "ACCEPTABLE", "ACCEPTED", "SEALED", "NEEDS_WORK", "BLOCKED",
        "REFUSED", "UNKNOWN", "CONFLICT",
    } else "UNKNOWN"


def _selector_values(frame: Mapping[str, object], selectors: Sequence[str]) -> list[str]:
    result = _strings(selectors)
    for key in ("required_public_checks", "required_selectors", "test_selectors", "selectors"):
        for item in _strings(frame.get(key)):
            if item not in result:
                result.append(item)
    for row_key in ("test_receipts", "host_check_receipts", "receipts"):
        rows = frame.get(row_key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key in ("selector", "check_id", "test_selector", "name"):
                item = row.get(key)
                if isinstance(item, str) and item.strip() and item not in result:
                    result.append(item.strip())
    return result[:8]


def _receipt_candidates(frame: Mapping[str, object], revision: str) -> list[tuple[str, Mapping[str, object]]]:
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for key in ("test_receipts", "host_check_receipts", "receipts"):
        rows = frame.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_revision = row.get("revision", row.get("workspace_revision", row.get("source_revision")))
            if row_revision not in (None, "") and str(row_revision) != revision:
                continue
            candidates.append((key, row))
    return candidates


def _candidate_statuses(frame: Mapping[str, object], paths: Sequence[Sequence[str]], revision: str,
                        *, include_receipts: bool = False) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in paths:
        value = _at(frame, path)
        if value is None:
            continue
        row = _mapping(value)
        row_revision = row.get("revision", row.get("current_revision", row.get("workspace_revision")))
        result.append({
            "path": _source_label(path), "state": _status(value),
            "revision": str(row_revision) if row_revision not in (None, "") else revision,
            "evidence": _refs(row),
        })
    if include_receipts:
        for source, row in _receipt_candidates(frame, revision):
            result.append({
                "path": "public-check receipt", "state": _status(row.get("result_class", row.get("result", row.get("status")))),
                "revision": str(row.get("revision", revision)), "evidence": _refs(row),
            })
    return result


def _has_current_check_execution(
        frame: Mapping[str, object], candidates: Sequence[Mapping[str, object]],
        revision: str,
) -> bool:
    """Require current run or receipt evidence before exposing a check result."""
    no_check_marker = False
    for path in _PUBLIC_CHECK_PATHS:
        value = _at(frame, path)
        row = _mapping(value)
        row_revision = row.get(
            "revision", row.get("current_revision", row.get("workspace_revision", revision)))
        if str(row_revision) == revision and _contains_no_check_marker(value):
            no_check_marker = True
    for key in ("test_receipts", "host_check_receipts", "receipts"):
        rows = frame.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for value in rows:
            if not isinstance(value, Mapping):
                continue
            row_revision = value.get(
                "revision", value.get("workspace_revision", value.get("source_revision", revision)))
            if str(row_revision) == revision and _contains_no_check_marker(value):
                no_check_marker = True
    current = [
        row for row in candidates
        if str(row.get("revision", revision)) == revision
    ]
    explicit_current_execution = False
    for key in ("test_receipts", "host_check_receipts", "receipts"):
        rows = frame.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for value in rows:
            if not isinstance(value, Mapping):
                continue
            row_revision = value.get(
                "revision", value.get("workspace_revision", value.get("source_revision", revision)))
            if str(row_revision) != revision:
                continue
            if _contains_no_check_marker(value):
                continue
            if _refs(value) or any(value.get(name) not in (None, "", False) for name in (
                    "ran", "execution_id", "run_id", "started_at", "completed_at")):
                explicit_current_execution = True
                break
    if explicit_current_execution:
        return True
    if no_check_marker:
        return False
    if any(str(row.get("state", "")).upper() in {"NO_CHECK", "NO_CHECK_RECORDED", "GATE_NEVER_RAN"}
           for row in current):
        return False
    if any(
        isinstance(row.get("evidence"), Sequence)
        and not isinstance(row.get("evidence"), (str, bytes, bytearray))
        and bool(row.get("evidence"))
        for row in current
    ):
        return True
    return False


def _contains_no_check_marker(value: object) -> bool:
    if not isinstance(value, Mapping):
        return str(value or "").strip().upper().replace("-", "_") in {
            "GATE_NEVER_RAN", "NO_CHECK", "NO_CHECK_RECORDED",
        }
    markers = {"GATE_NEVER_RAN", "NO_CHECK", "NO_CHECK_RECORDED"}
    return any(
        str(value.get(key, "")).strip().upper().replace("-", "_") in markers
        for key in ("state", "status", "result", "verdict", "outcome",
                    "execution_state", "check_state", "gate_state")
    )


def public_checks_were_not_run(frame: Mapping[str, object] | None) -> bool:
    """Detect a current explicit no-run marker without exposing internal results."""
    source = frame if isinstance(frame, Mapping) else {}
    revision = _revision(source)
    return any(
        str(_mapping(value).get(
            "revision", _mapping(value).get("current_revision", revision))) == revision
        and _contains_no_check_marker(value)
        for path in _PUBLIC_CHECK_PATHS
        if (value := _at(source, path)) is not None
    )


def _source_label(path: Sequence[str]) -> str:
    """Keep legacy alias spellings out of the F4 model-facing evidence view."""
    first = str(path[0]) if path else "host"
    last = str(path[-1]) if path else "host"
    if last in {"submission", "completion_submission_status"}:
        return "submission authority"
    if last in {"atomic_completion", "completion_decision", "completion_reducer_decision",
                "completion_authority", "completion_status"}:
        return "completion authority"
    if first in {"verification", "valid_product_verification", "completion_verdict",
                 "recent_change_verification", "certification_status", "world",
                 "public_check_status", "current_public_checks", "terminal_status",
                 "obligations", "completion_contract", "resume", "readiness"}:
        return "public-check authority"
    if first in {"done_permission", "done_allowed", "current_state"}:
        return "dispatch-gate authority"
    if first in {"completion_authority", "completion_decision", "completion_reducer_decision",
                 "atomic_completion"}:
        return "completion authority"
    if first in {"submission", "completion_submission_status"}:
        return "submission authority"
    return first.replace("_", "-")


def _reduce_status(candidates: Sequence[Mapping[str, object]], revision: str,
                   *, unknown: str = "UNKNOWN") -> tuple[str, list[dict[str, object]], list[str]]:
    current = [row for row in candidates if str(row.get("revision", revision)) == revision]
    stale = [row for row in candidates if row not in current]
    chosen = current or list(candidates)
    states = {str(row.get("state", "UNKNOWN")) for row in chosen}
    meaningful = states - {"UNKNOWN"}
    if not current and stale:
        state = "STALE"
    elif len(meaningful) > 1:
        state = "CONFLICT"
    elif meaningful:
        state = next(iter(meaningful))
    elif stale:
        state = "STALE"
    else:
        state = unknown
    evidence: list[dict[str, object]] = []
    for row in chosen:
        evidence.append({
            "source": str(row.get("path", "host")),
            "state": str(row.get("state", "UNKNOWN")),
            "revision": str(row.get("revision", revision)),
            "references": list(row.get("evidence", ())) if isinstance(row.get("evidence"), Sequence) else [],
        })
    return state, evidence, [str(row.get("path", "host")) for row in chosen]


def _rows(frame: Mapping[str, object], keys: Sequence[str]) -> list[object]:
    result: list[object] = []
    for key in keys:
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result.extend(_copy_json(item) for item in value)
        elif isinstance(value, Mapping):
            result.append(_copy_json(value))
    return result


def _unique_rows(rows: Sequence[object]) -> list[object]:
    result: list[object] = []
    seen: set[bytes] = set()
    for row in rows:
        marker = _canonical(row)
        if marker not in seen:
            seen.add(marker)
            result.append(row)
    return result


def _decision_row(value: object) -> Mapping[str, object]:
    return _mapping(value)


def _claim(name: str, *, state: str, revision: str, ran: Sequence[str], evidence: object,
           establishes: str, authority: str) -> dict[str, object]:
    return {
        "predicate": name,
        "state": state,
        "ran": list(ran) or ["not recorded"],
        "revision": revision,
        "evidence": _copy_json(evidence),
        "establishes": establishes,
        "authority": authority,
    }


def project_f4_state(frame: Mapping[str, object] | None, *,
                     active_decisions: object = None, tools: object = None,
                     selectors: Sequence[str] = (), path: str | None = None) -> dict[str, object]:
    """Project the allowlisted, revision-bound agent-facing state.

    Raw host fields deliberately never pass through this function.  The output
    contains one field for each distinct predicate; different authorities for
    the same predicate become ``CONFLICT`` evidence rather than parallel labels.
    """
    source = dict(frame or {})
    revision = _revision(source, path)
    selector_values = _selector_values(source, selectors)
    check_candidates = _candidate_statuses(
        source, _PUBLIC_CHECK_PATHS, revision, include_receipts=True)
    check_state, check_evidence, check_sources = _reduce_status(check_candidates, revision)
    public_check_state = (
        check_state if _has_current_check_execution(source, check_candidates, revision)
        else "NOT_RUN"
    )
    check_claim = _claim(
        "public_checks", state=public_check_state, revision=revision,
        ran=selector_values or check_sources,
        evidence=check_evidence,
        establishes=F4_PREDICATE_GLOSSARY["public_checks"],
        authority="host_receipts",
    )

    permission_candidates: list[dict[str, object]] = []
    for path_bits in _DONE_PERMISSION_PATHS:
        value = _at(source, path_bits)
        if value is None:
            continue
        row = _mapping(value)
        permission_candidates.append({
            "path": _source_label(path_bits), "state": _permission(value),
            "revision": str(row.get("revision", revision)), "evidence": _refs(row),
        })
    readiness = _mapping(source.get("readiness"))
    if not permission_candidates:
        for key in ("ready",):
            if key in readiness:
                permission_candidates.append({
                    "path": "dispatch-gate authority", "state": _permission(readiness[key]),
                    "revision": revision, "evidence": _refs(readiness),
                })
                break
    permission_state, permission_evidence, _ = _reduce_status(
        permission_candidates, revision)
    permission_claim = _claim(
        "done_permission", state=permission_state, revision=revision,
        ran=[str(row["path"]) for row in permission_candidates] or ["dispatch gate not recorded"],
        evidence=permission_evidence,
        establishes=F4_PREDICATE_GLOSSARY["done_permission"],
        authority="active_dispatch_gate",
    )

    completion_candidates: list[dict[str, object]] = []
    for path_bits in _COMPLETION_PATHS:
        value = _at(source, path_bits)
        if value is None:
            continue
        row = _decision_row(value)
        completion_candidates.append({
            "path": _source_label(path_bits), "state": _completion(value),
            "revision": str(row.get("revision", row.get("current_revision", revision))),
            "evidence": _refs(row),
        })
    decision_row = _decision_row(active_decisions)
    if decision_row:
        completion_candidates.append({
            "path": "completion authority", "state": _completion(decision_row),
            "revision": str(decision_row.get("revision", revision)),
            "evidence": _refs(decision_row),
        })
    completion_state, completion_evidence, _ = _reduce_status(
        completion_candidates, revision)
    completion_claim = _claim(
        "completion", state=completion_state, revision=revision,
        ran=[str(row["path"]) for row in completion_candidates] or ["completion decision not recorded"],
        evidence=completion_evidence,
        establishes=F4_PREDICATE_GLOSSARY["completion"],
        authority="completion-seal",
    )

    submission_candidates: list[dict[str, object]] = []
    for path_bits in _SUBMISSION_PATHS:
        value = _at(source, path_bits)
        if value is None:
            continue
        row = _decision_row(value)
        submission_candidates.append({
            "path": _source_label(path_bits), "state": _submission(value),
            "revision": str(row.get("revision", revision)), "evidence": _refs(row),
        })
    submission_state, submission_evidence, _ = _reduce_status(
        submission_candidates, revision)
    submission_claim = _claim(
        "submission", state=submission_state, revision=revision,
        ran=[str(row["path"]) for row in submission_candidates] or ["submission state not recorded"],
        evidence=submission_evidence,
        establishes=F4_PREDICATE_GLOSSARY["submission"],
        authority="host_submission_projection",
    )

    blocker_rows = _rows(source, ("blockers", "open_work", "pending_multi_definition_edits"))
    if readiness.get("blockers") is not None:
        blocker_rows.extend(_rows({"blockers": readiness.get("blockers")}, ("blockers",)))
    host_closure = _mapping(source.get("host_closure"))
    assessments = host_closure.get("assessments")
    if isinstance(assessments, Sequence) and not isinstance(assessments, (str, bytes, bytearray)):
        blocker_rows.extend(
            _copy_json(row) for row in assessments
            if isinstance(row, Mapping) and str(row.get("status", row.get("verdict", ""))).upper()
            in {"BLOCKED", "NEEDS_WORK", "OPEN", "FAIL", "FAILED"}
        )
    blocker_rows = _unique_rows(blocker_rows)
    advisory_rows = _rows(source, ("advisories", "advisory"))
    advisory_rows.extend(_rows({"advisory": readiness.get("advisory")}, ("advisory",)))
    advisory_rows = _unique_rows(advisory_rows)

    def bind_rows(rows: Sequence[object], predicate: str, establishes: str) -> list[dict[str, object]]:
        return [{
            "predicate": predicate, "value": row, "revision": revision,
            "ran": ["host observation"], "establishes": establishes,
        } for row in rows]

    claims = [check_claim, permission_claim, completion_claim, submission_claim]
    state = {
        "revision": revision,
        "step": _copy_json(source.get("step")),
        "public_checks": {
            "state": public_check_state, "selectors": selector_values or ["not recorded"],
            "revision": revision, "evidence": check_evidence,
            "establishes": check_claim["establishes"],
        },
        "blockers": bind_rows(
            blocker_rows, "blockers", F4_PREDICATE_GLOSSARY["blockers"]),
        "advisories": bind_rows(
            advisory_rows, "advisories", F4_PREDICATE_GLOSSARY["advisories"]),
        "done_permission": {
            "state": permission_state, "revision": revision,
            "evidence": permission_evidence,
            "establishes": permission_claim["establishes"],
        },
        "completion": {
            "state": completion_state, "revision": revision,
            "evidence": completion_evidence,
            "establishes": completion_claim["establishes"],
        },
        "submission": {
            "state": submission_state, "revision": revision,
            "evidence": submission_evidence,
            "establishes": submission_claim["establishes"],
        },
    }
    notice = source.get("environment_check_notice")
    if (evictfix_enabled()
            and isinstance(notice, str) and 0 < len(notice) <= 320):
        state["environment_check_notice"] = notice
    # The progress ledger is an explicitly supplied presentation payload.  It
    # is allowlisted as one state row, never interpreted as a predicate, and is
    # absent on the OFF path so the existing F4 bytes remain unchanged.
    progress = source.get("progress_ledger")
    if isinstance(progress, Mapping):
        state["progress_ledger"] = _copy_json(progress)
    # The history-findings organ is optional.  Keep the import behind the
    # producer's field gate so an F4-only/off run does not acquire a module or
    # a new payload shape; when present, use its shared field/schema contract.
    if "history_findings" in source:
        from kernel_stateframe_history_findings import FRAME_FIELD, validate_projection
        history_findings = source.get(FRAME_FIELD)
        if isinstance(history_findings, Mapping):
            validate_projection(history_findings)
            state[FRAME_FIELD] = _copy_json(history_findings)
    if "settled_findings" in source:
        from kernel_stateframe_settled_findings import (
            FRAME_FIELD, mode as settled_findings_mode, validate_projection,
        )
        if settled_findings_mode() == "on":
            settled_findings = source.get(FRAME_FIELD)
            if isinstance(settled_findings, Mapping):
                validate_projection(settled_findings)
                state[FRAME_FIELD] = _copy_json(settled_findings)
    # Keep the allowlist auditable: no raw alias field is allowed into the public view.
    assert not _ALIAS_FIELDS.intersection(state)
    return {
        "schema": SCHEMA,
        "revision": revision,
        "state": state,
        "claims": claims,
        "tool_menu_digest": _digest(tools) if tools is not None else None,
    }


def _slot_map(slots: object) -> Mapping[str, object]:
    root = _mapping(slots)
    nested = root.get("slots")
    return nested if isinstance(nested, Mapping) else root


def _clip(value: object, maximum: int) -> str:
    text = str(value or "").strip()
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 1)] + "…"


def _handoff_text(slots: object, host: object, rationale: object, *, limits: tuple[int, ...],
                  work_carry: object = None) -> str:
    values = _slot_map(slots)
    envelope = _mapping(host)
    hypothesis = _mapping(values.get("hypothesis"))
    basis = _mapping(values.get("basis"))
    rejected = values.get("rejected")
    rejected_row = (rejected[0] if isinstance(rejected, Sequence) and rejected
                    and isinstance(rejected[0], Mapping) else {})
    step = _mapping(values.get("step"))
    probe = _mapping(values.get("probe"))
    fallback = _clip(rationale, 260)
    hypothesis_text = str(hypothesis.get("text") or "").strip() or fallback
    if not hypothesis_text:
        hypothesis_text = "not yet supplied"
    approach = str(step.get("text") or probe.get("action") or "").strip()
    if approach:
        hypothesis_text += "; approach: " + approach
    revision = str(envelope.get("revision") or envelope.get("workspace_revision") or "UNKNOWN")
    support = _strings(basis.get("support_refs"), limit=3)
    adverse = str(basis.get("adverse_ref") or "").strip()
    basis_sentence = str(basis.get("sentence") or "").strip()
    if basis_sentence:
        evidence = basis_sentence
    elif envelope.get("last_action") or envelope.get("last_result"):
        evidence = (f"host recorded {envelope.get('last_action', 'an action')} -> "
                    f"{envelope.get('last_result', 'a result')} on revision {revision}; "
                    "interpretation is not yet supplied")
    else:
        evidence = "not yet supplied; no decisive host evidence is recorded"
    evidence += f" (revision {revision}"
    if support:
        evidence += "; refs " + ", ".join(support)
    if adverse:
        evidence += "; adverse " + adverse
    evidence += ")"
    if rejected_row:
        rejected_text = (f"{rejected_row.get('text', 'not supplied')} "
                         f"({rejected_row.get('disposition', 'unclassified')})")
        reopen = str(rejected_row.get("reconsider_condition") or "contrary evidence appears")
    else:
        rejected_text = "not yet supplied"
        reopen = "the relevant code changes or contrary evidence appears"
    next_text = str(step.get("text") or probe.get("action") or "").strip()
    if not next_text:
        next_text = "supply the missing decision slot and choose the next discriminating action"
    done_when = str(step.get("done_when") or probe.get("predicted_result") or "").strip()
    if done_when:
        next_text += "; expected outcome: " + done_when
    extension = ""
    if isinstance(work_carry, Mapping):
        try:
            from kernel_work_carry import render_extension
            carry_limit = max(80, min(360, limits[0] + limits[1] // 2))
            extension = render_extension(work_carry, max_chars=carry_limit)
        except (ImportError, TypeError, ValueError):
            extension = ""
    text = (
        f"{HANDOFF_HEADER}\n"
        f"Hypothesis/approach: {_clip(hypothesis_text, limits[0])}.\n"
        f"Decisive evidence: {_clip(evidence, limits[1])}; it establishes only the "
        "bounded observation, not a final guarantee.\n"
        f"Rejected alternative: {_clip(rejected_text, limits[2])}; reopen if {_clip(reopen, limits[3])}.\n"
        f"Next decision: {_clip(next_text, limits[4])}."
        + ("\n" + extension if extension else "") + "\n"
        "Defeasible: Revisit if relevant code changes or contrary evidence."
    )
    return text


def validate_handoff(message: Mapping[str, object] | str, *, minimum: int = MIN_HANDOFF_TOKENS,
                     maximum: int = MAX_HANDOFF_TOKENS) -> int:
    content = message if isinstance(message, str) else message.get("content", "")
    estimate = token_estimate({"role": "user", "content": str(content)})
    if estimate > maximum:
        raise ValueError(f"F4 decision handoff exceeds {maximum} tokens: {estimate}")
    if estimate < minimum and "not yet supplied" not in str(content):
        raise ValueError(f"F4 decision handoff is below {minimum} tokens: {estimate}")
    return estimate


def render_f4_handoff(*, slots: object = None, host: object = None,
                      rationale: object = None, minimum: int = MIN_HANDOFF_TOKENS,
                      maximum: int = MAX_HANDOFF_TOKENS,
                      work_carry: object = None) -> dict[str, str | int]:
    """Render the four-slot handoff, preserving whole reopen conditions."""
    complete = bool(_mapping(_slot_map(slots)).get("hypothesis")) and bool(
        _mapping(_slot_map(slots)).get("basis")) and bool(_slot_map(slots).get("rejected"))
    limits = (240, 300, 220, 220, 220)
    text = _handoff_text(slots, host, rationale, limits=limits, work_carry=work_carry)
    estimate = token_estimate({"role": "user", "content": text})
    if complete and estimate < minimum:
        text = text[:-1] + (
            " The current claim is a working model rather than a host fact; retain the named "
            "revision and receipts until the next decision resolves it.")
        estimate = token_estimate({"role": "user", "content": text})
    if estimate > maximum:
        for limits in ((190, 220, 165, 165, 180), (150, 175, 130, 140, 150),
                       (120, 145, 110, 115, 120)):
            text = _handoff_text(slots, host, rationale, limits=limits, work_carry=work_carry)
            estimate = token_estimate({"role": "user", "content": text})
            if estimate <= maximum:
                break
    # r41 review fix F4-R4: this runs inside prepare(); a budget miss must never crash the
    # run (ERR).  Model-authored slots can contain escaped unicode that inflates the JSON
    # estimate, so degrade to a hard field clip and record the miss instead of raising.
    budget_note = None
    if estimate > maximum:
        for limits in ((80, 100, 70, 70, 80), (40, 60, 40, 40, 40)):
            text = _handoff_text(slots, host, rationale, limits=limits, work_carry=work_carry)
            estimate = token_estimate({"role": "user", "content": text})
            if estimate <= maximum:
                break
        if estimate > maximum:
            budget_note = "over_budget"
    if complete and estimate < minimum:
        budget_note = "under_budget"
    result: dict[str, str | int] = {"role": "user", "content": text, "token_estimate": estimate}
    if budget_note is not None:
        result["budget_note"] = budget_note
    return result


def _call_ids(provider_calls: object) -> set[str]:
    if isinstance(provider_calls, Mapping):
        provider_calls = provider_calls.get("call_ids", provider_calls.get("tool_calls", ()))
    if not isinstance(provider_calls, Sequence) or isinstance(provider_calls, (str, bytes, bytearray)):
        return set()
    result: set[str] = set()
    for item in provider_calls:
        if isinstance(item, str):
            if item.strip():
                result.add(item.strip())
        elif isinstance(item, Mapping):
            call_id = item.get("id")
            if isinstance(call_id, str) and call_id.strip():
                result.add(call_id.strip())
    return result


def _pair_messages(messages: Sequence[Mapping[str, object]], allowed: set[str]) -> list[tuple[dict, dict]]:
    result: list[tuple[dict, dict]] = []
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping) or raw.get("role") != "assistant":
            continue
        calls = raw.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            continue
        following: list[Mapping[str, object]] = []
        cursor = index + 1
        while cursor < len(messages):
            candidate = messages[cursor]
            if not isinstance(candidate, Mapping) or candidate.get("role") != "tool":
                break
            following.append(candidate)
            cursor += 1
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or call_id not in allowed or ":auto_verify" in call_id:
                continue
            raw_result = next((item for item in following
                               if isinstance(item, Mapping)
                               and item.get("role") == "tool"
                               and item.get("tool_call_id") == call_id), None)
            if raw_result is None:
                continue
            assistant = copy.deepcopy(dict(raw))
            assistant.pop("reasoning_content", None)
            assistant.pop("response_reasoning_content", None)
            assistant["tool_calls"] = [copy.deepcopy(dict(call))]
            result.append((assistant, copy.deepcopy(dict(raw_result))))
    return result


def _wire_safe_call(call: Mapping[str, object]) -> dict:
    """r41 live fix (smokefull-323-ldev_01, llama.cpp /apply-template HTTP 500 "Failed to parse
    tool call arguments as JSON"): a replayed assistant call whose arguments string is not a
    JSON object (the model's truncated/broken call, which the host already REJECTED) kills
    every later request.  Valid arguments stay byte-identical; invalid ones are carried as a
    JSON object holding the exact raw text, so the model still sees what it sent."""
    result = copy.deepcopy(dict(call))
    function = result.get("function")
    if not isinstance(function, Mapping):
        return result
    function = dict(function)
    raw = function.get("arguments")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
        valid = isinstance(parsed, dict)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        function["arguments"] = json.dumps({"unparsed_arguments": str(raw if raw is not None else "")},
                                           ensure_ascii=False)
    result["function"] = function
    return result


def source_working_set_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict default-on C1b source-context switch."""
    source = os.environ if environ is None else environ
    value = str(source.get(SOURCE_WORKING_SET_ENV, "on")).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{SOURCE_WORKING_SET_ENV} must be on or off")
    return value


def source_working_set_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return source_working_set_mode(environ) == "on"


def _source_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    if not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        return ""
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(
            part == ".." or part.casefold() == "backups" for part in candidate.parts):
        return ""
    normalized = candidate.as_posix()
    return "" if normalized == "." else normalized


def _source_exchange_call(
        pair: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, object], str, str] | None:
    if len(pair) != 2:
        return None
    assistant, result = pair
    result_id = result.get("tool_call_id")
    calls = assistant.get("tool_calls")
    if (not isinstance(result_id, str) or not isinstance(calls, Sequence)
            or isinstance(calls, (str, bytes, bytearray))):
        return None
    call = next((item for item in calls if isinstance(item, Mapping)
                 and item.get("id") == result_id), None)
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    name = str(function.get("name") or "").strip().rsplit(".", 1)[-1].casefold()
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        arguments = dict(parsed) if isinstance(parsed, Mapping) else {}
    else:
        arguments = {}
    content = result.get("content")
    result_text = content if isinstance(content, str) else ""
    assistant_content = assistant.get("content")
    assistant_text = assistant_content if isinstance(assistant_content, str) else ""
    return name, arguments, result_text, assistant_text


def _source_page_from_exchange(
        name: str, arguments: Mapping[str, object], result_text: str,
        *, environ: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    if name != "read":
        return None
    header_v2 = source_page_header_v2_mode(environ) == "on"
    header, first_sep, rest = result_text.partition("\n")
    _notice, second_sep, body = rest.partition("\n")
    header_pattern = _SOURCE_PAGE_HEADER if header_v2 else _SOURCE_PAGE_HEADER_LEGACY
    match = header_pattern.fullmatch(header)
    if not first_sep or not second_sep or match is None:
        return None
    path = _source_path(match.group(1))
    revision = match.group(2).lower()
    start_line, end_line, total_lines, char_count = (
        int(match.group(index)) for index in range(3, 7))
    replaced = match.group(8).lower() == "true"
    if header_v2:
        content, trailer = body[:char_count], body[char_count:]
        char_count_valid = len(content) == char_count and (trailer == "" or trailer.startswith("\n"))
    else:
        content = body
        char_count_valid = char_count == len(content)
    requested_path = _source_path(arguments.get("path", arguments.get("file")))
    if (not path or (requested_path and requested_path != path) or replaced
            or start_line < 1 or end_line < start_line or total_lines < end_line
            or not char_count_valid):
        return None
    content_lines = content.splitlines(keepends=True)
    if not content or len(content_lines) != end_line - start_line + 1:
        return None
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    content_digest = hashlib.sha256(content_bytes).hexdigest()
    role = str(arguments.get("role", arguments.get("owner_role", "")) or "").casefold()
    why = str(arguments.get("why", "") or "").casefold()
    is_edit_anchor = (role in {"edit_anchor", "recently_edited", "current_edit_anchor"}
                      or ("edit" in why and "anchor" in why))
    is_cited = (role in {"helper", "import", "imports"}
                or "helper" in why or "import" in why)
    return {
        "path": path,
        "revision": revision,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "content": content,
        "content_digest": content_digest,
        "byte_length": len(content_bytes),
        "edit_anchor": is_edit_anchor,
        "cited": is_cited,
        "unresolved_failure": False,
    }


def _source_page_identity(page: Mapping[str, object]) -> tuple[str, str, int, int]:
    return (str(page.get("path") or ""), str(page.get("content_digest") or ""),
            int(page.get("start_line", 0)), int(page.get("end_line", 0)))


def _source_page_priority(page: Mapping[str, object]) -> int:
    if page.get("edit_anchor") is True:
        return 0
    if page.get("cited") is True:
        return 1
    if page.get("unresolved_failure") is True:
        return 2
    return 3


class RealExchangeLedger:
    """Bounded, provenance-gated ring of closed provider call/result pairs.

    ``STATEFRAME_KEEP_LATEST_BATCH`` keeps the newest two provider responses as
    whole batches. The older response still feeds the optional ``on_evict`` hook
    pair-by-pair before eviction. With the switch off, the historic two-pair cap
    and snapshot shape remain unchanged.
    """

    def __init__(self, exchanges: Sequence[Sequence[Mapping[str, object]]] = (),
                 *, exchange_batch_ids: Sequence[object] = (),
                 latest_batch_id: str | None = None, on_evict=None, on_pair=None):
        self._exchanges: list[tuple[dict, dict]] = []
        self._exchange_batch_ids: list[str] = []
        self._latest_batch_id: str | None = None
        self._source_pages: dict[tuple[str, str, int, int], dict[str, object]] = {}
        self._source_evictions: list[dict[str, object]] = []
        self._source_sequence = 0
        self._pending_edit: dict[str, object] | None = None
        self._on_evict = on_evict
        # [reasoning-ledger patch] optional observer of EVERY admitted pair (None: byte-identical)
        self._on_pair = on_pair
        self._install(exchanges, exchange_batch_ids=exchange_batch_ids)
        if isinstance(latest_batch_id, str) and latest_batch_id in self._exchange_batch_ids:
            self._latest_batch_id = latest_batch_id
        elif self._exchange_batch_ids:
            self._latest_batch_id = self._exchange_batch_ids[-1]

    def _install(self, exchanges: Sequence[Sequence[Mapping[str, object]]],
                 *, exchange_batch_ids: Sequence[object] = ()) -> None:
        for index, pair in enumerate(exchanges):
            if (isinstance(pair, Sequence) and len(pair) == 2
                    and all(isinstance(item, Mapping) for item in pair)):
                batch_id = (exchange_batch_ids[index]
                            if index < len(exchange_batch_ids) else None)
                self._append((copy.deepcopy(dict(pair[0])), copy.deepcopy(dict(pair[1]))),
                             batch_id=batch_id if isinstance(batch_id, str) else None)

    def _append(self, pair: tuple[dict, dict], *, batch_id: str | None = None) -> None:
        calls = pair[0].get("tool_calls")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
            pair = ({**pair[0], "tool_calls": [_wire_safe_call(call) if isinstance(call, Mapping)
                                                else call for call in calls]}, pair[1])
        call_id = str(pair[1].get("tool_call_id", ""))
        if not call_id or ":auto_verify" in call_id:
            return
        if pair[0].get("role") != "assistant" or pair[1].get("role") != "tool":
            return
        calls = pair[0].get("tool_calls")
        if (not isinstance(calls, Sequence)
                or not any(isinstance(call, Mapping) and call.get("id") == call_id
                           for call in calls)):
            return
        self._observe_source_exchange(pair)
        if keep_latest_batch_enabled() and batch_id:
            existing_index = next((index for index, row in enumerate(self._exchanges)
                                   if row[1].get("tool_call_id") == call_id), None)
            if (existing_index is not None
                    and self._exchange_batch_ids[existing_index] == batch_id):
                self._exchanges[existing_index] = pair
                if getattr(self, "_on_pair", None) is not None:
                    self._on_pair(pair)
                return
        retained = [(row, row_batch) for row, row_batch in
                    zip(self._exchanges, self._exchange_batch_ids)
                    if row[1].get("tool_call_id") != call_id]
        self._exchanges = [row for row, _row_batch in retained]
        self._exchange_batch_ids = [row_batch for _row, row_batch in retained]
        self._exchanges.append(pair)
        self._exchange_batch_ids.append(batch_id or f"legacy:{call_id}")
        if getattr(self, "_on_pair", None) is not None:  # [reasoning-ledger patch]
            self._on_pair(pair)
        if keep_latest_batch_enabled():
            self._trim_provider_batches()
        elif len(self._exchanges) > 2:
            self._evict_indices(range(len(self._exchanges) - 2))

    def _evict_indices(self, indexes: Sequence[int]) -> None:
        evicted_indexes = set(indexes)
        if self._on_evict is not None:
            for index, pair in enumerate(self._exchanges):
                if index in evicted_indexes:
                    self._on_evict(pair)
        self._exchanges = [pair for index, pair in enumerate(self._exchanges)
                           if index not in evicted_indexes]
        self._exchange_batch_ids = [batch_id for index, batch_id
                                    in enumerate(self._exchange_batch_ids)
                                    if index not in evicted_indexes]
        if self._latest_batch_id not in self._exchange_batch_ids:
            self._latest_batch_id = (self._exchange_batch_ids[-1]
                                     if self._exchange_batch_ids else None)

    def _trim_provider_batches(self) -> None:
        batches = list(dict.fromkeys(self._exchange_batch_ids))
        while len(batches) > MAX_RETAINED_PROVIDER_BATCHES:
            oldest = batches.pop(0)
            self._evict_indices([index for index, value
                                 in enumerate(self._exchange_batch_ids)
                                 if value == oldest])

    @staticmethod
    def _argument_text(arguments: Mapping[str, object], names: Sequence[str]) -> str:
        for name in names:
            value = arguments.get(name)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _argument_line(arguments: Mapping[str, object], name: str) -> int | None:
        value = arguments.get(name)
        if type(value) is int and value >= 1:
            return value
        return None

    def _mark_source_citations(self, text: str) -> None:
        if not text:
            return
        counts: dict[str, int] = {}
        for page in self._source_pages.values():
            basename = PurePosixPath(str(page["path"])).name.casefold()
            counts[basename] = counts.get(basename, 0) + 1
        folded = text.casefold()
        for page in self._source_pages.values():
            path = str(page["path"])
            basename = PurePosixPath(path).name.casefold()
            if path.casefold() in folded or (counts.get(basename) == 1 and basename in folded):
                page["cited"] = True

    def _set_latest_edit_anchor(self, key: tuple[str, str, int, int] | None) -> None:
        for page_key, page in self._source_pages.items():
            page["edit_anchor"] = page_key == key

    def _observe_edit(self, arguments: Mapping[str, object]) -> None:
        path = _source_path(arguments.get(
            "path", arguments.get("target_path", arguments.get("file", arguments.get("filename")))))
        if not path:
            return
        old_text = self._argument_text(arguments, ("old_str", "old_text", "old", "search"))
        new_text = self._argument_text(arguments, ("new_str", "new_text", "new", "replacement"))
        try:
            pending_text_bytes = len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
        except UnicodeEncodeError:
            pending_text_bytes = MAX_PENDING_EDIT_TEXT_BYTES + 1
        if pending_text_bytes > MAX_PENDING_EDIT_TEXT_BYTES:
            old_text = ""
            new_text = ""
        start_line = self._argument_line(arguments, "start_line")
        end_line = self._argument_line(arguments, "end_line") or start_line
        candidates: list[tuple[int, int, tuple[str, str, int, int]]] = []
        for key, page in self._source_pages.items():
            if page.get("path") != path:
                continue
            content = str(page.get("content") or "")
            if old_text and old_text in content:
                rank = 0
            elif (start_line is not None and end_line is not None
                  and int(page["start_line"]) <= end_line
                  and int(page["end_line"]) >= start_line):
                rank = 1
            else:
                continue
            candidates.append((rank, -int(page.get("last_seen", 0)), key))
        selected = min(candidates)[2] if candidates else None
        self._set_latest_edit_anchor(selected)
        self._pending_edit = {
            "path": path, "old_text": old_text, "new_text": new_text,
            "start_line": start_line, "end_line": end_line,
        }

    def _record_source_eviction(self, page: Mapping[str, object], reason: str) -> None:
        row = {
            "path": str(page.get("path") or ""),
            "start_line": int(page.get("start_line", 0)),
            "end_line": int(page.get("end_line", 0)),
            "content_digest": str(page.get("content_digest") or ""),
            "reason": str(reason),
        }
        self._source_evictions = [
            old for old in self._source_evictions
            if not (old.get("path") == row["path"]
                    and old.get("start_line") == row["start_line"]
                    and old.get("end_line") == row["end_line"]
                    and old.get("content_digest") == row["content_digest"])
        ]
        self._source_evictions.append(row)
        self._source_evictions = self._source_evictions[-MAX_SOURCE_WORKING_SET_EVICTIONS:]

    def _trim_source_pages(self) -> None:
        while (len(self._source_pages) > MAX_SOURCE_WORKING_SET_PAGES
               or sum(int(page.get("byte_length", 0))
                      for page in self._source_pages.values()) > MAX_SOURCE_WORKING_SET_BYTES):
            victim_key = max(
                self._source_pages,
                key=lambda key: (_source_page_priority(self._source_pages[key]),
                                 -int(self._source_pages[key].get("last_seen", 0)), key),
            )
            victim = self._source_pages.pop(victim_key)
            self._record_source_eviction(victim, "source_working_set_capacity")

    def _remember_source_page(self, page: dict[str, object]) -> None:
        key = _source_page_identity(page)
        existing = self._source_pages.get(key)
        if bool(page.get("edit_anchor")):
            self._set_latest_edit_anchor(None)
        if existing is not None:
            page["cited"] = bool(page.get("cited")) or bool(existing.get("cited"))
            page["unresolved_failure"] = (
                bool(page.get("unresolved_failure"))
                or bool(existing.get("unresolved_failure")))
            page["edit_anchor"] = bool(page.get("edit_anchor")) or bool(existing.get("edit_anchor"))
        self._source_sequence += 1
        page["last_seen"] = self._source_sequence
        self._source_pages[key] = copy.deepcopy(page)
        self._source_evictions = [
            old for old in self._source_evictions
            if not (old.get("path") == page.get("path")
                    and old.get("start_line") == page.get("start_line")
                    and old.get("end_line") == page.get("end_line"))
        ]
        self._trim_source_pages()

    def _observe_source_exchange(self, pair: Sequence[Mapping[str, object]]) -> None:
        if not source_working_set_enabled():
            return
        extracted = _source_exchange_call(pair)
        if extracted is None:
            return
        name, arguments, result_text, assistant_text = extracted
        if name != "read":
            citation_fields = [assistant_text]
            for field in _SOURCE_CITATION_FIELDS:
                value = arguments.get(field)
                if isinstance(value, str):
                    citation_fields.append(value)
                elif isinstance(value, Mapping):
                    citation_fields.extend(str(item) for item in value.values()
                                           if isinstance(item, str))
            self._mark_source_citations("\n".join(citation_fields))
        if name in _EDIT_TOOL_NAMES:
            self._observe_edit(arguments)
            return
        page = _source_page_from_exchange(name, arguments, result_text)
        if page is None:
            return
        if assistant_text and str(page["path"]).casefold() in assistant_text.casefold():
            page["cited"] = True
        pending = self._pending_edit
        if isinstance(pending, Mapping) and pending.get("path") == page.get("path"):
            new_text = pending.get("new_text")
            start_line, end_line = pending.get("start_line"), pending.get("end_line")
            overlaps = (type(start_line) is int and type(end_line) is int
                        and int(page["start_line"]) <= end_line
                        and int(page["end_line"]) >= start_line)
            if (isinstance(new_text, str) and new_text and new_text in str(page["content"])) or overlaps:
                page["edit_anchor"] = True
        if page.get("edit_anchor") is True:
            self._set_latest_edit_anchor(None)
        self._remember_source_page(page)

    @staticmethod
    def _exact_page_location(source: str, content: str, preferred_line: int) -> tuple[int, int] | None:
        expected_lines = content.splitlines(keepends=True)
        if not expected_lines:
            return None
        source_lines = source.splitlines(keepends=True)

        def matches_at(start_line: int) -> bool:
            offset = start_line - 1
            return (offset >= 0 and offset + len(expected_lines) <= len(source_lines)
                    and source_lines[offset:offset + len(expected_lines)] == expected_lines)

        if matches_at(preferred_line):
            return preferred_line, preferred_line + len(expected_lines) - 1
        positions: list[int] = []
        cursor = 0
        while len(positions) < 2:
            offset = source.find(content, cursor)
            if offset < 0:
                break
            end_offset = offset + len(content)
            at_line_start = offset == 0 or source[offset - 1] == "\n"
            at_line_end = (end_offset == len(source) or content.endswith(("\n", "\r"))
                           or source[end_offset:end_offset + 1] in {"\n", "\r"})
            if at_line_start and at_line_end:
                positions.append(source.count("\n", 0, offset) + 1)
            cursor = offset + 1
        if len(positions) != 1:
            return None
        start = positions[0]
        return start, start + len(expected_lines) - 1

    def _rekey_source_page(self, old_key: tuple[str, str, int, int],
                           page: dict[str, object]) -> None:
        self._source_pages.pop(old_key, None)
        new_key = _source_page_identity(page)
        existing = self._source_pages.get(new_key)
        if existing is not None:
            page["cited"] = bool(page.get("cited")) or bool(existing.get("cited"))
            page["edit_anchor"] = bool(page.get("edit_anchor")) or bool(existing.get("edit_anchor"))
            page["unresolved_failure"] = (
                bool(page.get("unresolved_failure"))
                or bool(existing.get("unresolved_failure")))
            page["last_seen"] = max(int(page.get("last_seen", 0)),
                                     int(existing.get("last_seen", 0)))
        self._source_pages[new_key] = page
        self._source_evictions = [
            old for old in self._source_evictions
            if not (old.get("path") == page.get("path")
                    and old.get("start_line") == page.get("start_line")
                    and old.get("end_line") == page.get("end_line"))
        ]

    def refresh_source_pages(self, root: object) -> bool:
        """Rebind retained pages to exact current bytes, shifting only exact unchanged spans."""
        if not source_working_set_enabled() or not self._source_pages:
            return False
        from pathlib import Path
        from kernel_f4_source_ranges import current_source_text
        from kernel_paged_source import source_revision

        before = self._source_working_set_state()
        root_path = root if isinstance(root, Path) else Path(str(root))
        sources: dict[str, str | None] = {}
        pending = self._pending_edit
        for old_key, old_page in list(self._source_pages.items()):
            path = str(old_page["path"])
            if path not in sources:
                sources[path] = current_source_text(root_path, path)
            source = sources[path]
            if source is None:
                self._source_pages.pop(old_key, None)
                self._record_source_eviction(old_page, "current_source_unavailable")
                continue
            current_revision = source_revision(source)
            old_start, old_end = int(old_page["start_line"]), int(old_page["end_line"])
            current_lines = source.splitlines(keepends=True)
            current_segment = ("".join(current_lines[old_start - 1:old_end])
                               if old_start >= 1 and old_end <= len(current_lines) else "")
            expected_content = str(old_page["content"])
            if (current_revision != old_page.get("revision")
                    and old_page.get("edit_anchor") is True
                    and isinstance(pending, Mapping) and pending.get("path") == path):
                old_text, new_text = pending.get("old_text"), pending.get("new_text")
                if (isinstance(old_text, str) and old_text and isinstance(new_text, str)
                        and expected_content.count(old_text) == 1):
                    expected_content = expected_content.replace(old_text, new_text, 1)
            if current_revision == old_page.get("revision") and current_segment:
                location = (old_start, old_end)
                expected_content = current_segment
            elif current_revision == old_page.get("revision"):
                location = None
            else:
                location = self._exact_page_location(source, expected_content, old_start)
                if location is None and expected_content != str(old_page["content"]):
                    location = self._exact_page_location(source, str(old_page["content"]), old_start)
                    if location is not None:
                        expected_content = str(old_page["content"])
            if location is None:
                self._source_pages.pop(old_key, None)
                self._record_source_eviction(old_page, "source_revision_changed")
                continue
            start_line, end_line = location
            segment_lines = source.splitlines(keepends=True)[start_line - 1:end_line]
            content = "".join(segment_lines)
            if not content:
                self._source_pages.pop(old_key, None)
                self._record_source_eviction(old_page, "source_range_unavailable")
                continue
            page = dict(old_page)
            page.update(
                revision=current_revision,
                start_line=start_line,
                end_line=end_line,
                total_lines=len(current_lines),
                content=content,
                content_digest=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                byte_length=len(content.encode("utf-8")),
            )
            self._rekey_source_page(old_key, page)
        self._pending_edit = None
        self._trim_source_pages()
        return before != self._source_working_set_state()

    def mark_unresolved_check(self, path: object = None, *,
                              start_line: int | None = None,
                              end_line: int | None = None) -> None:
        """Prioritize a current source page named by the latest unresolved check."""
        normalized = _source_path(path)
        for page in self._source_pages.values():
            matches = bool(normalized and page.get("path") == normalized)
            if matches and start_line is not None and end_line is not None:
                matches = (int(page["start_line"]) <= end_line
                           and int(page["end_line"]) >= start_line)
            page["unresolved_failure"] = matches

    def source_pages(self) -> tuple[dict[str, object], ...]:
        if not source_working_set_enabled():
            return ()
        ordered = sorted(
            self._source_pages.values(),
            key=lambda page: (_source_page_priority(page), -int(page.get("last_seen", 0)),
                              str(page.get("path") or ""), int(page.get("start_line", 0)),
                              int(page.get("end_line", 0)), str(page.get("revision") or ""),
                              str(page.get("content_digest") or "")),
        )
        return tuple(copy.deepcopy(page) for page in ordered)

    def source_working_set_notices(self) -> tuple[str, ...]:
        if not source_working_set_enabled():
            return ()
        notices = []
        for row in self._source_evictions:
            path = str(row.get("path") or "")
            start_line, end_line = int(row.get("start_line", 0)), int(row.get("end_line", 0))
            if not path or start_line < 1 or end_line < start_line:
                continue
            call = (f"read(path={json.dumps(path, ensure_ascii=False)}, "
                    f"start_line={start_line}, end_line={end_line})")
            notices.append(
                "[Source page evicted]\n"
                f"path={path} lines={start_line}-{end_line} reason={row.get('reason', 'capacity')}\n"
                f"Retrieve current bytes before relying on this page with: {call}."
            )
        return tuple(notices)

    def _source_working_set_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "schema": SOURCE_WORKING_SET_SCHEMA,
            "pages": [copy.deepcopy(page) for page in sorted(
                self._source_pages.values(), key=lambda page: (
                    int(page.get("last_seen", 0)), _source_page_identity(page)))],
            "evictions": copy.deepcopy(self._source_evictions),
            "sequence": self._source_sequence,
        }
        if self._pending_edit is not None:
            state["pending_edit"] = copy.deepcopy(self._pending_edit)
        return state

    def _restore_source_working_set(self, state: object) -> None:
        if (not source_working_set_enabled() or not isinstance(state, Mapping)
                or state.get("schema") != SOURCE_WORKING_SET_SCHEMA):
            return
        pages = state.get("pages", ())
        if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes, bytearray)):
            for raw_page in pages[-MAX_SOURCE_WORKING_SET_PAGES * 4:]:
                if not isinstance(raw_page, Mapping):
                    continue
                path = _source_path(raw_page.get("path"))
                content = raw_page.get("content")
                revision = raw_page.get("revision")
                start, end, total = (raw_page.get("start_line"), raw_page.get("end_line"),
                                     raw_page.get("total_lines"))
                if (not path or not isinstance(content, str) or not content
                        or not isinstance(revision, str)
                        or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
                        or type(start) is not int or type(end) is not int
                        or type(total) is not int or start < 1 or end < start or total < end
                        or len(content.splitlines(keepends=True)) != end - start + 1):
                    continue
                try:
                    content_bytes = content.encode("utf-8")
                except UnicodeEncodeError:
                    continue
                digest = hashlib.sha256(content_bytes).hexdigest()
                if raw_page.get("content_digest") != digest:
                    continue
                byte_length = len(content_bytes)
                if byte_length > MAX_SOURCE_WORKING_SET_BYTES:
                    continue
                last_seen = raw_page.get("last_seen", 0)
                last_seen = last_seen if type(last_seen) is int and last_seen > 0 else 0
                page = {
                    "path": path, "revision": revision, "start_line": start,
                    "end_line": end, "total_lines": total, "content": content,
                    "content_digest": digest, "byte_length": byte_length,
                    "edit_anchor": raw_page.get("edit_anchor") is True,
                    "cited": raw_page.get("cited") is True,
                    "unresolved_failure": raw_page.get("unresolved_failure") is True,
                    "last_seen": last_seen,
                }
                self._source_pages[_source_page_identity(page)] = page
                self._source_sequence = max(self._source_sequence, last_seen)
        evictions = state.get("evictions", ())
        if isinstance(evictions, Sequence) and not isinstance(evictions, (str, bytes, bytearray)):
            for row in evictions[-MAX_SOURCE_WORKING_SET_EVICTIONS:]:
                if not isinstance(row, Mapping):
                    continue
                path = _source_path(row.get("path"))
                start, end = row.get("start_line"), row.get("end_line")
                digest = row.get("content_digest")
                if (path and type(start) is int and type(end) is int and start >= 1
                        and end >= start and isinstance(digest, str)
                        and re.fullmatch(r"[0-9a-f]{64}", digest)):
                    self._record_source_eviction({
                        "path": path, "start_line": start, "end_line": end,
                        "content_digest": digest,
                    }, str(row.get("reason") or "capacity")[:80])
        sequence = state.get("sequence")
        if type(sequence) is int and sequence > self._source_sequence:
            self._source_sequence = sequence
        anchors = [key for key, page in self._source_pages.items()
                   if page.get("edit_anchor") is True]
        if len(anchors) > 1:
            latest = max(anchors, key=lambda key: (
                int(self._source_pages[key].get("last_seen", 0)), key))
            self._set_latest_edit_anchor(latest)
        pending = state.get("pending_edit")
        if isinstance(pending, Mapping):
            path = _source_path(pending.get("path"))
            old_text = pending.get("old_text")
            new_text = pending.get("new_text")
            start, end = pending.get("start_line"), pending.get("end_line")
            old_text = old_text if isinstance(old_text, str) else ""
            new_text = new_text if isinstance(new_text, str) else ""
            try:
                pending_bytes = len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
            except UnicodeEncodeError:
                pending_bytes = MAX_PENDING_EDIT_TEXT_BYTES + 1
            if path and pending_bytes <= MAX_PENDING_EDIT_TEXT_BYTES:
                self._pending_edit = {
                    "path": path, "old_text": old_text, "new_text": new_text,
                    "start_line": start if type(start) is int and start >= 1 else None,
                    "end_line": end if type(end) is int and end >= 1 else None,
                }
        self._trim_source_pages()

    def observe(self, messages: Sequence[Mapping[str, object]], *, provider_calls: object = (),
                batch_id: str | None = None) -> None:
        allowed = _call_ids(provider_calls)
        if not allowed:
            return
        pairs = _pair_messages(messages, allowed)
        if not pairs:
            return
        keep_batch = keep_latest_batch_enabled()
        if keep_batch and (not isinstance(batch_id, str) or not batch_id.strip()):
            batch_id = "calls:" + hashlib.sha256(
                "\0".join(sorted(allowed)).encode("utf-8")).hexdigest()
        if keep_batch and batch_id:
            batch_id = batch_id.strip()
            self._latest_batch_id = batch_id
        for pair in pairs:
            self._append(pair, batch_id=batch_id if keep_batch else None)

    def record(self, pair: Sequence[Mapping[str, object]], *, provider_call_id: str | None = None) -> bool:
        if (not provider_call_id or not isinstance(pair, Sequence) or len(pair) != 2
                or not all(isinstance(item, Mapping) for item in pair)):
            return False
        actual_id = pair[1].get("tool_call_id")
        if actual_id != provider_call_id or ":auto_verify" in str(actual_id):
            return False
        before = len(self._exchanges)
        if keep_latest_batch_enabled():
            self._latest_batch_id = provider_call_id
        self._install([pair], exchange_batch_ids=[provider_call_id])
        return any(row[1].get("tool_call_id") == provider_call_id
                   for row in self._exchanges) and len(self._exchanges) >= min(before, 2)

    def exchanges(self) -> tuple[tuple[dict, dict], ...]:
        return tuple((copy.deepcopy(left), copy.deepcopy(right)) for left, right in self._exchanges)

    def snapshot(self) -> dict[str, object]:
        state: dict[str, object] = {
            "schema": REAL_EXCHANGE_SCHEMA,
            "exchanges": [list(pair) for pair in self.exchanges()],
        }
        if keep_latest_batch_enabled():
            state["exchange_batch_ids"] = list(self._exchange_batch_ids)
            state["latest_batch_id"] = self._latest_batch_id
        if source_working_set_enabled() and (self._source_pages or self._source_evictions
                                             or self._pending_edit is not None):
            state["source_working_set"] = self._source_working_set_state()
        return state

    def latest_batch_call_ids(self) -> tuple[str, ...]:
        if not keep_latest_batch_enabled() or not self._latest_batch_id:
            return ()
        return tuple(
            str(pair[1].get("tool_call_id"))
            for pair, batch_id in zip(self._exchanges, self._exchange_batch_ids)
            if batch_id == self._latest_batch_id
        )

    @classmethod
    def from_state(cls, state: Mapping[str, object] | None) -> "RealExchangeLedger":
        rows = state.get("exchanges", ()) if isinstance(state, Mapping) else ()
        batch_ids = state.get("exchange_batch_ids", ()) if isinstance(state, Mapping) else ()
        latest_batch_id = state.get("latest_batch_id") if isinstance(state, Mapping) else None
        ledger = cls(
            rows if isinstance(rows, Sequence) else (),
            exchange_batch_ids=batch_ids if isinstance(batch_ids, Sequence)
            and not isinstance(batch_ids, (str, bytes, bytearray)) else (),
            latest_batch_id=latest_batch_id if isinstance(latest_batch_id, str) else None,
        )
        if isinstance(state, Mapping):
            ledger._restore_source_working_set(state.get("source_working_set"))
        return ledger


def keep_latest_batch_enabled() -> bool:
    """Return the strict default-on batch retention switch state."""
    value = os.environ.get(KEEP_LATEST_BATCH_ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{KEEP_LATEST_BATCH_ENV} must be on or off")
    return value == "on"


def provider_batch_id(capture: object, provider_ids: Sequence[str]) -> str:
    """Stable response identity without storing provider payload content."""
    event_id = (getattr(capture, "response_event_id", None)
                if not isinstance(capture, Mapping)
                else capture.get("response_event_id"))
    request_id = (getattr(capture, "request_id", None)
                  if not isinstance(capture, Mapping) else capture.get("request_id"))
    if isinstance(event_id, str) and event_id.strip():
        return "response:" + event_id.strip()
    if isinstance(request_id, str) and request_id.strip():
        return "request:" + request_id.strip()
    digest = hashlib.sha256("\0".join(sorted(provider_ids)).encode("utf-8")).hexdigest()
    return "calls:" + digest


def provider_call_ids(capture: object) -> tuple[str, ...]:
    """IDs under which this provider response's calls appear in the worker transcript.

    r41 live fix (smokefull-311 ldev_05: ring empty on every request, model re-read the
    same file 6x blind): with native tool batches (kernel_native_tools) the worker renames
    every provider call to ``native_<origin request_id>_<index>`` before it reaches the
    transcript, so the provider's own ids never match.  Those renamed ids are host-minted
    ONLY from a real provider response, so they carry the same provenance; add them.
    """
    value = getattr(capture, "provider_tool_calls", None)
    if value is None and isinstance(capture, Mapping):
        value = capture.get("provider_tool_calls", capture.get("provider_call_ids", ()))
    ids = _call_ids(value)
    request_id = (getattr(capture, "request_id", None) if not isinstance(capture, Mapping)
                  else capture.get("request_id"))
    count = len(value) if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)) else 0
    if isinstance(request_id, str) and request_id:
        ids.update(f"native_{request_id}_{index}" for index in range(count))
    return tuple(sorted(ids))


def order_messages(messages: Sequence[Mapping[str, object]]) -> Sequence[Mapping[str, object]]:
    """OFF identity helper used by callers that must prove byte parity."""
    if not enabled():
        return messages
    return order_f4_messages(messages)


def _reasoning_anchor_header() -> str:
    # r46: lazy import keeps kernel_stateframe_reasoning_anchor an optional dependency of
    # this module -- unused unless STATEFRAME_REASONING_ANCHOR is exercised.
    from kernel_stateframe_reasoning_anchor import ANCHOR_HEADER
    return ANCHOR_HEADER


def _reasoning_ledger_header() -> str:
    # [reasoning-ledger patch] lazy import, same pattern as the anchor header.
    from kernel_reasoning_ledger import LEDGER_HEADER
    return LEDGER_HEADER


def _pending_intent_header() -> str:
    # r46 STATEFRAME_INTENT_CARRY_F4: lazy import, same pattern as the anchor header.
    from kernel_intent_carry import PENDING_INTENT_HEADER
    return PENDING_INTENT_HEADER


# Retained host-state notices (working sets, pending intent) share one explicit rank so
# they stay adjacent in index order; equals the generic-notice tier, so ordering (and
# every existing request byte) is unchanged.
F4_RETAINED_NOTICE_RANK = 3


def _message_rank(message: Mapping[str, object], index: int) -> tuple[int, int]:
    role = message.get("role")
    content = str(message.get("content") or "")
    if role in {"assistant", "tool"}:
        return 100, index
    if content.startswith(CLAUSE_HEADER):
        return 89, index
    if content.startswith(_reasoning_anchor_header()):
        return 88, index
    if content.startswith(_reasoning_ledger_header()):  # [reasoning-ledger patch]
        return 88, index
    if content.startswith(STATE_HEADER):
        return 90, index
    if content.startswith(HANDOFF_HEADER):
        return 91, index
    if role in {"system", "developer"}:
        return 0, index
    if content.startswith("Task:") or content.startswith("[FRAME_CLARITY_BOOTSTRAP]"):
        return 1, index
    if content.startswith(("[Working source:", "[Paged source:", "[Current source:", "[Observed source:",
                           "[Retained source:", "[Source content:", "CURRENT_SOURCE ")):
        return 2, index
    if content.startswith(("[Working sets | host-retained]", _pending_intent_header())):
        return F4_RETAINED_NOTICE_RANK, index
    return 3, index


def order_f4_messages(messages: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Stable-first, volatile-last order without splitting a native exchange."""
    original = list(messages)
    first_history = next((index for index, item in enumerate(original)
                          if item.get("role") in {"assistant", "tool"}), len(original))
    prefix = original[:first_history]
    tail = original[first_history:]
    ordered_prefix = [item for _index, item in sorted(
        enumerate(prefix), key=lambda pair: _message_rank(pair[1], pair[0]))]
    return [*ordered_prefix, *tail]


def frame_bytes(messages: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize message blocks independently so LCP metrics match block order."""
    return b"".join(_canonical(dict(message)) + b"\n" for message in messages)


# r41 review fix F4-R3: ``claims`` restates every predicate already in ``state`` and
# ``tool_menu_digest`` is host bookkeeping; both stay in the projection/journal, never in
# the model-visible block (one field per predicate).
MODEL_VISIBLE_STATE_KEYS = ("schema", "revision", "state")


def state_message(state: Mapping[str, object]) -> dict[str, str]:
    visible = {key: state[key] for key in MODEL_VISIBLE_STATE_KEYS if key in state}
    return {"role": "user", "content": serialize_stateframe_payload(visible)}


def _append_verbatim_turns(
    messages: list[dict],
    turns: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict]:
    for turn in turns:
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes, bytearray)):
            raise TypeError("F4 verbatim turns must be message sequences")
        if any(not isinstance(item, Mapping) for item in turn):
            raise TypeError("F4 verbatim turns must contain message mappings")
        messages.extend(copy.deepcopy(dict(item)) for item in turn)
    return messages


def compose_messages(*, prefix: Sequence[Mapping[str, object]], path: str, working: str,
                     state: Mapping[str, object], handoff: Mapping[str, object],
                     exchanges: Sequence[Sequence[Mapping[str, object]]] = (),
                     clause_blocks: Sequence[Mapping[str, object]] = (),
                     source_messages: Sequence[Mapping[str, object]] | None = None,
                     notices: Sequence[str] = (),
                     anchor_entries: Sequence[Mapping[str, object]] = (),
                     ledger_block: str = "",
                     verbatim_turns: Sequence[Sequence[Mapping[str, object]]] = ()) -> list[dict]:
    """``source_messages`` (runtime-rendered, header contract owned by kernel_read_fixes)
    replaces the legacy literal "[Working source: ...]" block when supplied.

    ``anchor_entries`` (r46, STATEFRAME_REASONING_ANCHOR): rendered as at most one
    "[Settled findings]" block, and only when the switch is on AND entries is non-empty --
    off-parity is byte-identical to before this parameter existed either way.
    ``verbatim_turns`` appends an explicit legacy F4 suffix; FrameFit5 never passes it.
    """
    messages: list[dict] = []
    for message in prefix:
        content = str(message.get("content") or "")
        if content.startswith((STATE_HEADER, HANDOFF_HEADER, CLAUSE_HEADER,
                               _reasoning_anchor_header(),
                               _reasoning_ledger_header(),
                                "[Working source:", "[Paged source:", "[Current source:", "CURRENT_SOURCE ",
                               "[Observed source:", "[Retained source:")):
            continue
        messages.append(copy.deepcopy(dict(message)))
    if source_messages is not None:
        messages.extend({"role": "user", "content": str(item.get("content", ""))}
                        for item in source_messages if isinstance(item, Mapping))
    elif path:
        messages.append({"role": "user", "content": f"[Working source: {path}]\n{working}"})
    for block in clause_blocks:
        if isinstance(block, Mapping) and str(block.get("content") or "").startswith(CLAUSE_HEADER):
            messages.append({"role": "user", "content": str(block["content"])})
    if anchor_entries:
        from kernel_stateframe_reasoning_anchor import enabled as _anchor_enabled, render_block as _anchor_render
        if _anchor_enabled():
            messages.append({"role": "user", "content": _anchor_render(anchor_entries)})
    ring_text = "\n".join(
        str(item.get("content") or "") for pair in exchanges if isinstance(pair, Sequence)
        for item in pair if isinstance(item, Mapping))
    for notice in notices:
        # host notices already visible in a rendered real exchange are not repeated
        if isinstance(notice, str) and notice and notice not in ring_text:
            messages.append({"role": "user", "content": notice})
    if ledger_block:  # [reasoning-ledger patch] one user message, only when non-empty
        messages.append({"role": "user", "content": str(ledger_block)})
    messages.append(state_message(state))
    messages.append({
        "role": str(handoff.get("role", "user")),
        "content": str(handoff.get("content", "")),
    })
    for pair in exchanges:
        if isinstance(pair, Sequence) and len(pair) == 2:
            messages.extend(copy.deepcopy(dict(item)) for item in pair if isinstance(item, Mapping))
    ordered = [dict(message) for message in order_f4_messages(messages)]
    return _append_verbatim_turns(ordered, verbatim_turns)


def latest_complete_tool_exchange(
        messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...] | None:
    """Return the newest complete assistant tool-call batch, or None if malformed."""
    latest_index = next((index for index in range(len(messages) - 1, -1, -1)
                         if messages[index].get("role") == "assistant"
                         and isinstance(messages[index].get("tool_calls"), list)
                         and bool(messages[index].get("tool_calls"))), None)
    if latest_index is None:
        return ()
    assistant = messages[latest_index]
    calls = assistant.get("tool_calls")
    call_ids: list[str] = []
    for call in calls:
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if not isinstance(call_id, str) or not call_id or call_id in call_ids:
            return None
        call_ids.append(call_id)
    if not call_ids:
        return None
    results: list[Mapping[str, object]] = []
    cursor = latest_index + 1
    while cursor < len(messages) and messages[cursor].get("role") == "tool":
        results.append(messages[cursor])
        cursor += 1
    result_ids = [result.get("tool_call_id") for result in results]
    if (
            len(results) != len(call_ids)
            or not all(isinstance(value, str) and value for value in result_ids)
            or len(set(result_ids)) != len(result_ids)
            or set(result_ids) != set(call_ids)
    ):
        return None
    return (assistant, *results)


def assert_f4_invariants(
    messages: Sequence[Mapping[str, object]],
    *,
    allow_framefit5_intermediate: bool = False,
) -> None:
    state_blocks = [item for item in messages
                    if str(item.get("content") or "").startswith(STATE_HEADER)]
    handoffs = [item for item in messages
                if str(item.get("content") or "").startswith(HANDOFF_HEADER)]
    if len(state_blocks) != 1:
        raise AssertionError(f"F4 requires one state block, found {len(state_blocks)}")
    framefit5_product = False
    content = state_blocks[0].get("content")
    if isinstance(content, str):
        try:
            payload = json.loads(content.split("\n", 1)[1])
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            payload = None
        state = payload.get("state") if isinstance(payload, Mapping) else None
        framefit5 = state.get("framefit5") if isinstance(state, Mapping) else None
        framefit5_product = (
            isinstance(framefit5, Mapping)
            and framefit5.get("schema") == "stateframe-framefit5/1"
            and framefit5.get("mode") == "FRAME_ONLY"
        )
    if framefit5_product and not allow_framefit5_intermediate and handoffs:
        raise AssertionError("FRAMEFIT5 product cannot contain an F4 handoff")
    if framefit5_product and allow_framefit5_intermediate and len(handoffs) != 1:
        raise AssertionError(
            f"FRAMEFIT5 intermediate F4 projection requires one handoff, found {len(handoffs)}"
        )
    if not framefit5_product and len(handoffs) != 1:
        raise AssertionError(f"F4 requires one handoff, found {len(handoffs)}")
    if framefit5_product and not allow_framefit5_intermediate:
        from kernel_wire_stability import _is_marker_message as is_volatile_tail

        latest_exchange = latest_complete_tool_exchange(messages)
        action_messages = [item for item in messages
                           if item.get("role") in {"assistant", "tool"}]
        if latest_exchange is None:
            raise AssertionError("FRAMEFIT5 product has an incomplete newest tool exchange")
        marker_indexes = [
            index for index, item in enumerate(messages) if is_volatile_tail(item)
        ]
        if len(marker_indexes) != 1 or marker_indexes[0] != len(messages) - 1:
            raise AssertionError(
                "FRAMEFIT5 product requires one final volatile user tail"
            )
        marker_index = marker_indexes[0]
        exchange_start = marker_index - len(latest_exchange)
        if (action_messages != list(latest_exchange)
                or exchange_start < 0
                or list(messages[exchange_start:marker_index]) != list(latest_exchange)):
            raise AssertionError(
                "FRAMEFIT5 product requires the newest complete tool exchange immediately before its volatile tail"
            )
    clause_blocks = [item for item in messages
                     if str(item.get("content") or "").startswith(CLAUSE_HEADER)]
    if len(clause_blocks) > 1:
        raise AssertionError(f"F4 allows at most one clause table, found {len(clause_blocks)}")
    intent_blocks = [item for item in messages
                     if item.get("role") not in {"assistant", "tool"}
                     and str(item.get("content") or "").startswith(_pending_intent_header())]
    if len(intent_blocks) > 1:
        raise AssertionError(f"F4 allows at most one pending-intent block, found {len(intent_blocks)}")
    if intent_blocks and (intent_blocks[0].get("role") != "user"
                          or _message_rank(intent_blocks[0], 0)[0] != F4_RETAINED_NOTICE_RANK):
        raise AssertionError("F4 pending-intent block must be a user notice ranked with working sets")
    pending: set[str] = set()
    for item in messages:
        role = item.get("role")
        if role == "tool":
            call_id = item.get("tool_call_id")
            if call_id not in pending:
                raise AssertionError("F4 orphaned tool result")
            pending.remove(call_id)
        elif role == "assistant":
            if pending:
                raise AssertionError("F4 tool exchange is not closed before next assistant")
            calls = item.get("tool_calls") or []
            for call in calls:
                call_id = call.get("id") if isinstance(call, Mapping) else None
                if not isinstance(call_id, str) or call_id in pending:
                    raise AssertionError("F4 invalid tool call id")
                pending.add(call_id)
    if pending:
        raise AssertionError("F4 incomplete tool exchange")


__all__ = [
    "CLAUSE_HEADER", "F4_PREDICATE_GLOSSARY", "HANDOFF_HEADER", "MAX_HANDOFF_TOKENS", "MODEL_VISIBLE_STATE_KEYS", "MIN_HANDOFF_TOKENS", "MODE_ENV", "SCHEMA",
    "KEEP_LATEST_BATCH_ENV", "MAX_RETAINED_PROVIDER_BATCHES", "STATE_HEADER",
    "RealExchangeLedger", "assert_f4_invariants", "compose_messages",
    "enabled", "frame_bytes", "keep_latest_batch_enabled", "mode", "order_f4_messages",
    "order_messages", "project_f4_state", "provider_batch_id", "provider_call_ids",
    "latest_complete_tool_exchange", "public_checks_were_not_run",
    "render_f4_handoff", "state_message",
    "validate_handoff",
]
