"""Revision-bound storage for captured public check streams."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from kernel_measurement import digest


SCHEMA = "stateframe-check-output-receipt/1"
CHECK_EXECUTION_BINDING_ENV = "STATEFRAME_CHECK_EXECUTION_BINDING"
CHECK_RECEIPT_ENRICHMENT_DEDUP_ENV = "STATEFRAME_CHECK_RECEIPT_ENRICHMENT_DEDUP"
HANDLE_PREFIX = "check-receipt:"
DEFAULT_PAGE_CHARS = 6000
MAX_PAGE_CHARS = 12000
_CHECK_RECEIPT_MEMO_LIMIT = 64
CHECK_OUTCOMES = frozenset({
    "PASS", "BEHAVIOR_FAIL", "INVALID_CHECK", "ENVIRONMENT",
    "RESOURCE_LIMIT", "NOT_RUN", "UNKNOWN",
})
FAILURE_PHASES = frozenset({
    "NONE", "HOST", "SETUP", "COLLECTION", "IMPORT", "EXECUTION",
    "ASSERTION", "OBSERVATION", "TIMEOUT", "UNKNOWN",
})
_IMPORT_FAILURE = re.compile(
    r"(?:ImportError|ModuleNotFoundError|No module named|cannot import name|"
    r"collection error|test collection failed)",
    re.IGNORECASE,
)
_RESOURCE_FAILURE = re.compile(r"(?:timeout|timed out|out of memory|resource limit)", re.IGNORECASE)
_TYPED_OBSERVATION_FIELDS = frozenset({
    "outcome", "result", "failure_phase", "phase", "reached_observation",
    "assertion_reached", "product_observed",
})
_STRUCTURED_COUNT_KEYS = frozenset({
    "collected", "passed", "failed", "errored", "skipped", "deselected",
})


def _stable_product_revision(runtime: object, binding: Mapping[str, object]) -> str | None:
    product_files = binding.get("product_files")
    if not isinstance(product_files, Mapping):
        return None
    files = {str(key): value for key, value in product_files.items()}
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    journal = getattr(runtime, "journal", None)
    journal_path = getattr(journal, "path", None)
    if root is not None and journal_path is not None:
        try:
            relative_journal = Path(journal_path).resolve().relative_to(Path(root).resolve())
            prefix = relative_journal.parent.as_posix()
        except (OSError, RuntimeError, ValueError):
            prefix = ""
        if prefix and prefix != ".":
            files = {
                key: value for key, value in files.items()
                if key != prefix and not key.startswith(prefix + "/")
            }
    return digest(files)


def _revision_value(runtime: object, value: object) -> str:
    if isinstance(value, Mapping):
        stable = _stable_product_revision(runtime, value)
        if stable is not None:
            return stable
        for key in ("product_revision", "revision", "artifact"):
            if value.get(key) not in (None, ""):
                return str(value[key])
    return _text(value) or "UNKNOWN"


def current_wire_revision(runtime: object) -> str:
    """Resolve the product revision currently bound to a provider wire."""
    missing = object()
    try:
        binding = getattr(runtime, "binding", missing)
    except Exception:
        return "UNKNOWN"
    if binding is not missing:
        value = binding
        if callable(binding):
            try:
                value = binding()
            except Exception:
                return "UNKNOWN"
        if isinstance(value, Mapping):
            try:
                stable = _stable_product_revision(runtime, value)
                if stable is not None:
                    return stable
                for key in ("product_revision", "revision", "artifact"):
                    if value.get(key) not in (None, ""):
                        return str(value[key])
                # An empty binding carries no product identity: report UNKNOWN so the stale-wire guard fails closed and
                # the menu memo fails open (digest({}) used to masquerade as a real revision).
                if not value:
                    return "UNKNOWN"
                return digest(value)
            except Exception:
                return "UNKNOWN"
        if value not in (None, ""):
            return str(value)
        return "UNKNOWN"
    for key in ("product_revision", "current_revision", "revision"):
        try:
            value = getattr(runtime, key, None)
        except Exception:
            return "UNKNOWN"
        if value not in (None, ""):
            return str(value)
    return "UNKNOWN"


def _current_revision(runtime: object) -> str:
    """Return receipt revision, preferring its immutable preparation snapshot."""
    prepared = getattr(runtime, "prepared_binding", None)
    if isinstance(prepared, Mapping):
        revision = _revision_value(runtime, prepared)
        if revision != "UNKNOWN":
            return revision
    binding = getattr(runtime, "binding", None)
    value = binding
    if callable(binding):
        try:
            value = binding()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            value = None
    revision = _revision_value(runtime, value)
    if revision != "UNKNOWN":
        return revision
    for key in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, key, None)
        if value not in (None, ""):
            return str(value)
    return "UNKNOWN"


def _stateframe_mode(runtime: object | None, attribute: str, environment: str) -> str:
    try:
        configured = getattr(runtime, attribute, None) if runtime is not None else None
        raw = configured if configured is not None else os.environ.get(environment, "off")
        value = str(raw).strip().lower()
    except Exception:
        return "off"
    return value if value in {"off", "shadow", "on"} else "off"


def check_execution_binding_mode(runtime: object | None = None) -> str:
    """Read the opt-in pre-launch check binding mode; invalid values fail open."""
    return _stateframe_mode(
        runtime, "stateframe_check_execution_binding_mode", CHECK_EXECUTION_BINDING_ENV)


def check_receipt_enrichment_dedup_mode(runtime: object | None = None) -> str:
    """Read the opt-in exact-equal receipt enrichment mode; invalid values fail open."""
    return _stateframe_mode(
        runtime, "stateframe_check_receipt_enrichment_dedup_mode",
        CHECK_RECEIPT_ENRICHMENT_DEDUP_ENV)


def _check_execution_context(runtime: object, execution_id: str) -> Mapping[str, object] | None:
    try:
        contexts = getattr(runtime, "_stateframe_check_execution_contexts", None)
        context = contexts.get(execution_id) if isinstance(contexts, Mapping) else None
    except Exception:
        return None
    return context if isinstance(context, Mapping) else None


def _fingerprint(value: object) -> str:
    try:
        return digest(value)
    except Exception:
        return digest(_text(type(value).__name__))


def _text(value: object) -> str:
    return value if isinstance(value, str) else str(value or "")


def _preview(value: str, limit: int = 1200) -> tuple[str, int]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, 0
    preview = encoded[:max(0, limit)].decode("utf-8", "ignore")
    return preview, max(0, len(encoded) - len(preview.encode("utf-8")))


def _bytes(value: str) -> int:
    return len(value.encode("utf-8"))


_OUTCOME_ALIASES = {
    "PASSED": "PASS", "GREEN": "PASS", "OK": "PASS",
    "FAIL": "BEHAVIOR_FAIL", "FAILED": "BEHAVIOR_FAIL", "RED": "BEHAVIOR_FAIL",
    "ENV_ERROR": "ENVIRONMENT", "HOST_ERROR": "ENVIRONMENT",
    "HARNESS_ERROR": "ENVIRONMENT", "HOST_REFUSED": "ENVIRONMENT",
    "NO_TESTS": "NOT_RUN", "NO_TESTS_RAN": "NOT_RUN",
    "NO_TESTS_COLLECTED": "NOT_RUN", "COLLECTION_ONLY": "NOT_RUN",
    "KILLED": "RESOURCE_LIMIT", "SIGKILL": "RESOURCE_LIMIT",
    "TIMEOUT": "RESOURCE_LIMIT", "TIMED_OUT": "RESOURCE_LIMIT",
}


def _normalize_outcome(value: object) -> str | None:
    text = _text(value).strip().upper()
    if not text:
        return None
    normalized = _OUTCOME_ALIASES.get(text, text)
    return normalized if normalized in CHECK_OUTCOMES else None


def _has_typed_observation(value: object) -> bool:
    return isinstance(value, Mapping) and any(
        key in value and value[key] is not None for key in _TYPED_OBSERVATION_FIELDS
    )


def _count_map(value: object) -> dict[str, int]:
    """Keep only explicit, non-negative integer counters from a runner."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): count for key, count in value.items()
        if isinstance(key, str) and key.strip() and type(count) is int and count >= 0
    }


def _selector_scope(selector: str, value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        selected = tuple(str(item).strip() for item in value if str(item).strip())
        if selected:
            return selected
    return (selector,) if selector else ()


def _execution_termination(exit_code: object, reported: object = None) -> str:
    explicit = _text(reported).strip().upper()
    if explicit in {"RESOURCE_LIMIT", "KILLED", "SIGKILL", "OUT_OF_MEMORY"}:
        return "RESOURCE_LIMIT"
    if explicit in {"TIMEOUT", "TIMED_OUT", "WALL_TIMEOUT"}:
        return "TIMEOUT"
    try:
        value = int(exit_code) if exit_code is not None and not isinstance(exit_code, bool) else None
    except (TypeError, ValueError):
        value = None
    if value is None:
        return "UNKNOWN"
    if value in {124}:
        return "TIMEOUT"
    if value in {130, 134, 137, 143, 152, 153} or value in {-2, -6, -9, -15, -24, -25}:
        return "RESOURCE_LIMIT"
    return "EXITED"


def canonical_execution_id(
        runtime: object | None = None, *, execution_id: object = None,
        selector: object = "", sequence: object = None, fresh: bool = False,
) -> str:
    """Return the one opaque host identity shared by every view of one check."""
    explicit = _text(execution_id).strip()
    if explicit:
        return explicit
    pending = getattr(runtime, "_pending_check_execution_id", None) if runtime is not None else None
    if not fresh and pending not in (None, ""):
        return str(pending).strip()
    journal = getattr(runtime, "journal", None) if runtime is not None else None
    from kernel_wire_identity import next_wire_id, stable_wire_id
    owner = getattr(runtime, "owner", None) if runtime is not None else None
    task_id = getattr(owner, "task_id", None)
    step = getattr(owner, "current_step", None)
    if journal is not None:
        # The journal sequence survives checkpoint/crash replay. A runtime-local
        # counter does not, and could repeat an already published check ID.
        digest = next_wire_id(journal, "check-execution", step, selector, sequence)
    else:
        ordinal = int(getattr(runtime, "_check_execution_ordinal", 0) or 0) + 1
        if runtime is not None:
            setattr(runtime, "_check_execution_ordinal", ordinal)
        digest = stable_wire_id(task_id, "check-execution", step, selector,
                                sequence, ordinal)
    return "check-execution:" + digest


def _execution_receipt_id(execution_id: str) -> str:
    digest = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()[:24]
    return HANDLE_PREFIX + digest


def _receipt_is_wrapper_only(
        *, status: object, outcome: object, stdout: str, stderr: str,
        assertion_context: str, reached_observation: object,
        exit_code: object, typed_observation: object,
        failure_phase: object, assertion_reached: object,
        product_observed: object, wrapper_error: object,
) -> bool:
    """Recognize bookkeeping-only updates without inventing a second verdict."""
    selected = _normalize_outcome(outcome)
    status_text = _text(status).strip().upper()
    has_typed_fact = (
        bool(_text(failure_phase).strip())
        or assertion_reached is not None
        or product_observed is not None
        or _has_typed_observation(typed_observation)
    )
    return (
        selected in {None, "UNKNOWN"}
        and (
            status_text in {"", "UNKNOWN", "WRAPPER_ERROR", "PARSE_ERROR"}
            or (
                _text(wrapper_error).strip() != ""
                and status_text in {"PASS", "PASSED", "GREEN", "OK", "FAIL", "FAILED", "RED"}
            )
        )
        and not stdout and not stderr and not assertion_context
        and reached_observation is None and exit_code is None
        and not has_typed_fact
    )


def _public_receipt(private: Mapping[str, object], journal: object = None) -> dict[str, object]:
    from kernel_wire_identity import public_wire_text, public_wire_value

    public = {
        key: value for key, value in private.items()
        if key not in {
            "stdout", "stderr", "assertion_context", "_stateframe_memo_generation",
        }
    }
    # The stored receipt remains lossless. Recompute visible sizes and preview
    # from the projected streams so address width cannot perturb model text.
    stdout = public_wire_text(str(private.get("stdout") or ""), journal)
    stderr = public_wire_text(str(private.get("stderr") or ""), journal)
    context = public_wire_text(str(private.get("assertion_context") or ""), journal)
    context_preview, context_omitted = _preview(context)
    public.update({
        "stdout_bytes": _bytes(stdout),
        "stderr_bytes": _bytes(stderr),
        "stdout_omitted_bytes": _bytes(stdout),
        "stderr_omitted_bytes": _bytes(stderr),
        "assertion_context_bytes": _bytes(context),
        "assertion_context_preview": context_preview,
        "assertion_context_omitted_bytes": context_omitted,
        "runner_tail": _runner_tail(stdout, stderr),
    })
    public = public_wire_value(public, journal, normalize_output=True)
    public["summary"] = _summary(public)
    public["text"] = public["summary"]
    public["source"] = "check_receipt"
    return public


def _runner_tail(stdout: str, stderr: str, limit: int = 240) -> str:
    """The runner's own last non-blank output line (e.g. "1 failed, 45 passed").

    Language-generic: pytest, go test, vitest and maven all end with their
    result line.  Verbatim and bounded; it never decides the outcome.
    """
    for stream in (stdout, stderr):
        for line in reversed(stream.splitlines()):
            if line.strip():
                return line.strip()[:limit]
    return ""


def _capture_metadata(output_truncated: object) -> dict[str, object]:
    if output_truncated is not None and type(output_truncated) is not bool:
        raise ValueError("output_truncated must be a bool or None")
    if output_truncated is None:
        capture_complete = None
        capture_status = "UNKNOWN"
    else:
        capture_complete = not output_truncated
        capture_status = "PARTIAL" if output_truncated else "COMPLETE"
    return {
        "output_truncated": output_truncated,
        "capture_complete": capture_complete,
        "capture_status": capture_status,
    }


def _summary(receipt: Mapping[str, object]) -> str:
    context = str(receipt.get("assertion_context_preview") or "NO ASSERTION CONTEXT")
    return (
        "CHECK_FAILURE_SUMMARY "
        f"execution_id={receipt.get('execution_id', 'UNKNOWN')} "
        f"status={receipt.get('status', 'UNKNOWN')} "
        f"outcome={receipt.get('outcome', 'UNKNOWN')} "
        f"reached_observation={receipt.get('reached_observation')} "
        f"failure_phase={receipt.get('failure_phase', 'UNKNOWN')} "
        f"selector={receipt.get('selector', '')} "
        f"selector_scope={receipt.get('selector_scope', [])!r} "
        f"execution_complete={receipt.get('execution_complete', False)} "
        f"termination_reason={receipt.get('termination_reason', 'UNKNOWN')} "
        f"output_truncated={receipt.get('output_truncated')} "
        f"capture_complete={receipt.get('capture_complete')} "
        f"capture_status={receipt.get('capture_status', 'UNKNOWN')} "
        f"counts={receipt.get('counts', {})!r} "
        f"revision={receipt.get('revision', 'UNKNOWN')} "
        f"assertion_context={context!r} "
        f"runner_tail={str(receipt.get('runner_tail') or '')!r} "
        f"stdout_bytes={receipt.get('stdout_bytes', 0)} "
        f"stderr_bytes={receipt.get('stderr_bytes', 0)} "
        f"assertion_context_omitted_bytes={receipt.get('assertion_context_omitted_bytes', 0)} "
        f"stdout_omitted_bytes={receipt.get('stdout_omitted_bytes', 0)} "
        f"stderr_omitted_bytes={receipt.get('stderr_omitted_bytes', 0)} "
        f"wrapper_parse_error={receipt.get('wrapper_parse_error', '')!r} "
        f"full_output_handle={receipt.get('full_output_handle')}"
    )


def classify_check_outcome(
        status: object,
        *,
        stdout: object = "",
        stderr: object = "",
        reached_observation: object = None,
        failure_phase: object = None,
        outcome: object = None,
        exit_code: object = None,
        collected_tests: object = None,
        assertion_reached: object = None,
        product_observed: object = None,
        typed_observation: object = None,
        counts: object = None,
        execution_complete: object = None,
        termination_reason: object = None,
) -> tuple[str, str, bool | None]:
    """Reduce typed execution facts without letting wrappers rewrite them.

    Structured producer facts win over incidental text.  Text is only a
    fallback for an untyped pre-target import/setup failure; it cannot turn a
    reached assertion into an environment result.
    """
    typed = typed_observation if isinstance(typed_observation, Mapping) else {}
    if outcome is None:
        outcome = typed.get("outcome", typed.get("result"))
    if failure_phase is None:
        failure_phase = typed.get("failure_phase", typed.get("phase"))
    if reached_observation is None:
        reached_observation = typed.get(
            "reached_observation", typed.get("assertion_reached", typed.get("product_observed")))
    if assertion_reached is None:
        assertion_reached = typed.get("assertion_reached")
    if product_observed is None:
        product_observed = typed.get("product_observed")
    if counts is None:
        counts = typed.get("counts")
    if execution_complete is None:
        execution_complete = typed.get("execution_complete")
    if termination_reason is None:
        termination_reason = typed.get("termination_reason")
    typed_phase_supplied = bool(_text(failure_phase).strip()) or (
        isinstance(typed_observation, Mapping)
        and any(_text(typed.get(key)).strip() for key in ("failure_phase", "phase"))
    )
    output = "\n".join(value for value in (stdout, stderr) if isinstance(value, str))
    phase = str(failure_phase or "").upper()
    status_text = str(status or "").upper()
    selected = _normalize_outcome(outcome)
    explicit_unknown = str(outcome or "").strip().upper() == "UNKNOWN"
    try:
        exit_value = int(exit_code) if exit_code is not None and not isinstance(exit_code, bool) else None
    except (TypeError, ValueError):
        exit_value = None
    typed_termination = _execution_termination(exit_code, termination_reason)
    normalized_counts = _count_map(counts)
    if collected_tests is None:
        collected_tests = normalized_counts.get("collected")
    try:
        collected = int(collected_tests) if (
            collected_tests is not None and not isinstance(collected_tests, bool)
        ) else None
    except (TypeError, ValueError):
        collected = None
    if phase not in FAILURE_PHASES:
        phase = ""
    reached = reached_observation if isinstance(reached_observation, bool) else None
    if reached is None and isinstance(assertion_reached, bool):
        reached = assertion_reached
    if reached is None and isinstance(product_observed, bool):
        reached = product_observed
    typed_behavior_failure = (
        selected == "BEHAVIOR_FAIL"
        or phase in {"ASSERTION", "OBSERVATION"}
    )
    # Explicit settled behavior survives incidental termination text. Reach is
    # separate: an UNKNOWN observation that reached its target is still UNKNOWN.
    if selected == "BEHAVIOR_FAIL":
        preserved_phase = phase if phase in {
            "HOST", "EXECUTION", "TIMEOUT", "ASSERTION", "OBSERVATION"} else "ASSERTION"
        return selected, preserved_phase, True
    if explicit_unknown:
        if not phase and typed_termination in {"RESOURCE_LIMIT", "TIMEOUT"}:
            phase = "TIMEOUT" if typed_termination == "TIMEOUT" else "HOST"
        return "UNKNOWN", phase or "UNKNOWN", reached

    # An unfinished process cannot be promoted to PASS. Retain typed reach as a
    # separate fact, including when the host killed a process after target entry.
    if selected == "RESOURCE_LIMIT" or typed_termination == "RESOURCE_LIMIT":
        default_phase = ("TIMEOUT" if typed_termination == "TIMEOUT"
                         or phase == "TIMEOUT" or status_text in {"TIMEOUT", "TIMED_OUT"}
                         else "HOST")
        return "RESOURCE_LIMIT", phase if phase in {"HOST", "TIMEOUT"} else default_phase, reached if reached is not None else False
    if typed_termination == "TIMEOUT" or (selected is None and _RESOURCE_FAILURE.search(output)):
        return "RESOURCE_LIMIT", "TIMEOUT", reached if reached is not None else False
    if execution_complete is False and selected is None and not typed_behavior_failure:
        return "UNKNOWN", phase or "UNKNOWN", reached
    if execution_complete is False and selected == "PASS":
        return "UNKNOWN", phase or "UNKNOWN", reached

    # An explicit typed result is authoritative; status prose is only fallback.
    if selected in {"ENVIRONMENT", "INVALID_CHECK", "NOT_RUN"}:
        default_phase = {
            "ENVIRONMENT": "EXECUTION", "INVALID_CHECK": "COLLECTION",
            "NOT_RUN": "COLLECTION",
        }[selected]
        return selected, phase or default_phase, reached if reached is not None else False
    if collected == 0 or exit_value == 5 or status_text in {
            "NO_TESTS", "NO_TESTS_RAN", "NO_TESTS_COLLECTED", "COLLECTION_ONLY"}:
        return "NOT_RUN", "COLLECTION", False
    if selected == "PASS":
        if reached is None:
            reached = True
        return "PASS", phase or "NONE", reached

    behavior_status = status_text in {"FAIL", "FAILED", "RED", "BEHAVIOR_FAIL"}
    if typed_behavior_failure or (
            behavior_status and (
                reached is True or assertion_reached is True or product_observed is True)):
        preserved_phase = phase if phase in {
            "HOST", "EXECUTION", "TIMEOUT", "ASSERTION", "OBSERVATION"} else (
                "ASSERTION" if assertion_reached is True else "OBSERVATION")
        return "BEHAVIOR_FAIL", preserved_phase, True

    untyped_pre_target_failure = (
        not typed_phase_supplied
        and reached is not True
        and assertion_reached is not True
        and product_observed is not True
        and status_text in {"FAIL", "FAILED", "ERROR", "EXECUTION_ERROR"}
        and _IMPORT_FAILURE.search(output)
    )
    if phase in {"IMPORT", "SETUP"} or untyped_pre_target_failure:
        return "ENVIRONMENT", phase if phase in {"IMPORT", "SETUP"} else "IMPORT", False
    if phase == "COLLECTION" or exit_value in {2, 3, 4} or status_text in {
            "COLLECTION_ERROR", "USAGE_ERROR", "INTERNAL_ERROR"}:
        return "INVALID_CHECK", "COLLECTION", False
    selected = _normalize_outcome(status_text) or {
        "PASS": "PASS", "PASSED": "PASS", "GREEN": "PASS", "OK": "PASS",
        "FAIL": "BEHAVIOR_FAIL", "FAILED": "BEHAVIOR_FAIL", "RED": "BEHAVIOR_FAIL",
        "INVALID_CHECK": "INVALID_CHECK", "ENV_ERROR": "ENVIRONMENT",
        "HOST_REFUSED": "ENVIRONMENT", "HARNESS_ERROR": "ENVIRONMENT",
        "NOT_RUN": "NOT_RUN", "ERROR": "UNKNOWN", "EXECUTION_ERROR": "UNKNOWN",
    }.get(status_text, "UNKNOWN")
    if selected == "BEHAVIOR_FAIL":
        if phase in {"HOST", "EXECUTION", "TIMEOUT"}:
            return selected, phase, reached
        return selected, "ASSERTION", True
    if selected == "PASS" and collected != 0:
        return selected, "NONE", True
    if selected == "ENVIRONMENT":
        return selected, "HOST" if status_text in {"HOST_REFUSED", "HARNESS_ERROR"} else "EXECUTION", False
    if selected == "INVALID_CHECK":
        return selected, "COLLECTION", False
    if selected == "NOT_RUN":
        return selected, "COLLECTION", False
    if selected == "UNKNOWN" and status_text in {"ERROR", "EXECUTION_ERROR"}:
        return selected, "EXECUTION", reached
    return selected, phase or "UNKNOWN", reached


def _emit(runtime: object, receipt: Mapping[str, object]) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            "check_receipt",
            producer="kernel_check_receipts",
            event_schema=SCHEMA,
            receipt_id=receipt["receipt_id"],
            execution_id=receipt["execution_id"],
            host_execution_id=receipt["host_execution_id"],
            full_output_handle=receipt["full_output_handle"],
            selector=receipt["selector"],
            status=receipt["status"],
            outcome=receipt["outcome"],
            reached_observation=receipt["reached_observation"],
            failure_phase=receipt["failure_phase"],
            selector_scope=receipt.get("selector_scope", ()),
            counts=receipt.get("counts", {}),
            execution_complete=receipt.get("execution_complete", False),
            termination_reason=receipt.get("termination_reason", "UNKNOWN"),
            revision=receipt["revision"],
            summary=receipt["summary"],
            stdout_bytes=receipt["stdout_bytes"],
            stderr_bytes=receipt["stderr_bytes"],
            stdout_omitted_bytes=receipt["stdout_omitted_bytes"],
            stderr_omitted_bytes=receipt["stderr_omitted_bytes"],
            assertion_context_omitted_bytes=receipt["assertion_context_omitted_bytes"],
            wrapper_parse_error=receipt.get("wrapper_parse_error", ""),
            dispatch=False,
        )


def _emit_shadow_mismatch(
        runtime: object, *, kind: str, execution_id: str, field: str,
        legacy_value: object, proposed_value: object,
) -> None:
    """Journal field-level shadow differences without including bound source data."""
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(
            kind,
            producer="kernel_check_receipts",
            event_schema=SCHEMA,
            execution_id=execution_id,
            field=field,
            legacy_fingerprint=_fingerprint(legacy_value),
            proposed_fingerprint=_fingerprint(proposed_value),
            dispatch=False,
        )
    except Exception:
        # Shadow telemetry must never turn receipt recording into a failure.
        pass


def _receipt_content_equal(
        candidate: Mapping[str, object], existing: Mapping[str, object],
) -> bool:
    """Compare complete receipts while ignoring the private paging generation."""
    try:
        return (
            {key: value for key, value in candidate.items()
             if key != "_stateframe_memo_generation"}
            ==
            {key: value for key, value in existing.items()
             if key != "_stateframe_memo_generation"}
        )
    except Exception:
        return False


_CANONICAL_STATUS = {"PASS": "PASS", "BEHAVIOR_FAIL": "FAIL"}


def canonical_identity(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The identity every wrapper copies from one check action's canonical receipts.

    Wrappers (owner receipt, test receipt, completion authority, automatic
    result) REFERENCE this; they never re-derive an outcome from exit text.
    Aggregate: any BEHAVIOR_FAIL -> BEHAVIOR_FAIL; all PASS -> PASS; else the
    single child's outcome or UNKNOWN.  ``execution_id`` is the latest child.
    """
    children = [row for row in rows if isinstance(row, Mapping) and row.get("execution_id")]
    if not children:
        return {}
    outcomes = [str(row.get("outcome") or "UNKNOWN").upper() for row in children]
    if "BEHAVIOR_FAIL" in outcomes:
        outcome = "BEHAVIOR_FAIL"
    elif all(item == "PASS" for item in outcomes):
        outcome = "PASS"
    else:
        outcome = outcomes[0] if len(outcomes) == 1 else "UNKNOWN"
    return {
        "execution_id": str(children[-1]["execution_id"]),
        "execution_ids": [str(row["execution_id"]) for row in children],
        "check_receipt_ids": [str(row.get("receipt_id") or "") for row in children],
        "outcome": outcome,
        "status": _CANONICAL_STATUS.get(outcome, "UNKNOWN"),
    }


def record_check_output(
        runtime: object,
        *,
        selector: object,
        status: object,
        stdout: object = "",
        stderr: object = "",
        assertion_context: object = "",
        revision: object | None = None,
        sequence: object | None = None,
        reached_observation: object = None,
        failure_phase: object = None,
        outcome: object = None,
        execution_id: object = None,
        exit_code: object = None,
        collected_tests: object = None,
        assertion_reached: object = None,
        product_observed: object = None,
        typed_observation: object = None,
        counts: object = None,
        selector_scope: object = None,
        execution_complete: object = None,
        termination_reason: object = None,
        output_truncated: object = None,
        producer: object = None,
        wrapper_error: object = None,
        wrapper_parse_error: object = None,
) -> dict[str, object]:
    """Store captured public streams and publish a compact receipt row."""
    capture_metadata = _capture_metadata(output_truncated)
    selector_text = _text(selector)
    status_text = _text(status).upper() or "UNKNOWN"
    stdout_text = _text(stdout)
    stderr_text = _text(stderr)
    context_text = _text(assertion_context)
    typed = typed_observation if isinstance(typed_observation, Mapping) else {}
    if counts is None:
        counts = typed.get("counts")
    if execution_complete is None:
        execution_complete = typed.get("execution_complete")
    if termination_reason is None:
        termination_reason = typed.get("termination_reason")
    scope = _selector_scope(selector_text, selector_scope)
    structured_counts = _count_map(counts)
    if collected_tests is None:
        collected_tests = structured_counts.get("collected")
    ordinal = sequence
    if ordinal is None:
        ordinal = getattr(getattr(runtime, "owner", None), "current_step", 0)
    execution_text = canonical_execution_id(
        runtime, execution_id=execution_id, selector=selector_text, sequence=ordinal)
    execution_binding_mode = check_execution_binding_mode(runtime)
    execution_context = (
        _check_execution_context(runtime, execution_text)
        if execution_binding_mode != "off" else None
    )
    pinned_scope_applied = False
    effective_revision = revision
    if isinstance(execution_context, Mapping):
        pinned_binding = execution_context.get("binding")
        if isinstance(pinned_binding, Mapping):
            try:
                pinned_revision = _revision_value(runtime, pinned_binding)
                legacy_revision = (
                    _revision_value(runtime, revision) if revision is not None
                    else _current_revision(runtime)
                )
            except Exception:
                pinned_revision = "UNKNOWN"
                legacy_revision = "UNKNOWN"
            if execution_binding_mode == "shadow" and pinned_revision != legacy_revision:
                _emit_shadow_mismatch(
                    runtime,
                    kind="check_execution_binding_shadow_mismatch",
                    execution_id=execution_text,
                    field="revision",
                    legacy_value=legacy_revision,
                    proposed_value=pinned_revision,
                )
            if execution_binding_mode == "on" and pinned_revision != "UNKNOWN":
                effective_revision = pinned_binding
        pinned_scope = execution_context.get("selector_scope")
        has_pinned_scope = False
        try:
            has_pinned_scope = pinned_scope not in (None, "")
            proposed_scope = (
                _selector_scope(selector_text, pinned_scope) if has_pinned_scope else scope
            )
        except Exception:
            proposed_scope = scope
        if execution_binding_mode == "shadow" and proposed_scope != scope:
            _emit_shadow_mismatch(
                runtime,
                kind="check_execution_binding_shadow_mismatch",
                execution_id=execution_text,
                field="selector_scope",
                legacy_value=scope,
                proposed_value=proposed_scope,
            )
        if execution_binding_mode == "on":
            scope = proposed_scope
            pinned_scope_applied = has_pinned_scope
    wrapper_text = _text(wrapper_parse_error or wrapper_error).strip()
    if len(wrapper_text) > 1200:
        wrapper_text = wrapper_text[:1200]
    has_native_observation = bool(
        stdout_text or stderr_text or context_text
        or outcome is not None or exit_code is not None
        or reached_observation is not None
        or failure_phase not in (None, "")
        or assertion_reached is not None
        or product_observed is not None
        or _has_typed_observation(typed_observation)
    )
    classification_status = "UNKNOWN" if wrapper_text and not has_native_observation else status
    typed_outcome, typed_phase, observation_reached = classify_check_outcome(
        classification_status,
        stdout=stdout_text,
        stderr=stderr_text,
        reached_observation=reached_observation,
        failure_phase=failure_phase,
        outcome=outcome,
        exit_code=exit_code,
        collected_tests=collected_tests,
        assertion_reached=assertion_reached,
        product_observed=product_observed,
        typed_observation=typed_observation,
        counts=structured_counts,
        execution_complete=execution_complete,
        termination_reason=termination_reason,
    )
    termination = _execution_termination(exit_code, termination_reason)
    if termination == "UNKNOWN" and typed_outcome == "RESOURCE_LIMIT":
        termination = "RESOURCE_LIMIT"
    elif termination == "UNKNOWN" and typed_phase == "TIMEOUT":
        termination = "TIMEOUT"
    complete = execution_complete if isinstance(execution_complete, bool) else (
        termination not in {"UNKNOWN", "RESOURCE_LIMIT", "TIMEOUT"}
        or typed_outcome in {"PASS", "BEHAVIOR_FAIL"}
    )
    if termination in {"RESOURCE_LIMIT", "TIMEOUT"}:
        complete = False
    revision_text = (_revision_value(runtime, effective_revision)
                     if effective_revision is not None else _current_revision(runtime))
    handle = _execution_receipt_id(execution_text)
    context_preview, context_omitted = _preview(context_text)
    private = {
        "schema": SCHEMA,
        "receipt_id": handle,
        "full_output_handle": handle,
        "execution_id": execution_text,
        "host_execution_id": execution_text,
        "selector": selector_text,
        "selector_scope": list(scope),
        "producer": _text(producer).strip(),
        # The status is a projection of the ONE canonical outcome: a settled
        # PASS/BEHAVIOR_FAIL is never re-decided by a wrapper's own status word
        # (b857: PASS/46 -> UNKNOWN, FAIL/26 -> UNKNOWN).  A wrapper's advertised
        # PASS/FAIL without native facts stays unresolved.
        "status": _CANONICAL_STATUS.get(typed_outcome) or (
            "UNKNOWN" if wrapper_text and not has_native_observation else status_text),
        "outcome": typed_outcome,
        "reached_observation": observation_reached,
        "failure_phase": typed_phase,
        "revision": revision_text,
        "sequence": ordinal,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "assertion_context": context_text,
        "exit_code": exit_code,
        "collected_tests": collected_tests,
        "counts": structured_counts,
        "execution_complete": complete,
        "termination_reason": termination,
        **capture_metadata,
        "assertion_reached": assertion_reached,
        "product_observed": product_observed,
        "wrapper_parse_error": wrapper_text,
    }
    private.update({
        "stdout_bytes": _bytes(stdout_text),
        "stderr_bytes": _bytes(stderr_text),
        "assertion_context_bytes": _bytes(context_text),
        "stdout_omitted_bytes": _bytes(stdout_text),
        "stderr_omitted_bytes": _bytes(stderr_text),
        "assertion_context_preview": context_preview,
        "assertion_context_omitted_bytes": context_omitted,
        "runner_tail": _runner_tail(stdout_text, stderr_text),
    })
    private["summary"] = _summary(private)
    receipts = getattr(runtime, "_check_output_receipts", None)
    if not isinstance(receipts, dict):
        receipts = {}
        setattr(runtime, "_check_output_receipts", receipts)
    existing = receipts.get(handle)
    wrapper_only = _receipt_is_wrapper_only(
        status=status, outcome=outcome, stdout=stdout_text, stderr=stderr_text,
        assertion_context=context_text, reached_observation=reached_observation,
        exit_code=exit_code, typed_observation=typed_observation,
        failure_phase=failure_phase, assertion_reached=assertion_reached,
        product_observed=product_observed, wrapper_error=wrapper_text)
    if isinstance(existing, Mapping):
        merged = dict(existing)
        if wrapper_text:
            merged["wrapper_parse_error"] = wrapper_text
        if (selector_scope is not None or pinned_scope_applied) and scope:
            merged["selector_scope"] = list(scope)
        if structured_counts:
            merged["counts"] = structured_counts
            if collected_tests is not None:
                merged["collected_tests"] = collected_tests
        if _text(producer).strip():
            merged["producer"] = _text(producer).strip()
        settled = str(existing.get("outcome") or "").upper() in _CANONICAL_STATUS
        later_is_settled = str(typed_outcome or "").upper() in _CANONICAL_STATUS
        if not wrapper_only and settled and not later_is_settled:
        # A later view of the SAME execution that cannot settle it
            # (exit-text only, no typed facts) never re-decides a settled
            # PASS/BEHAVIOR_FAIL; it may only add metadata above.
            wrapper_only = True
        if not wrapper_only:
            for key, value in private.items():
                if key == "wrapper_parse_error" and not wrapper_text:
                    continue
                if key == "selector_scope" and selector_scope is None:
                    continue
                if key == "counts" and not structured_counts:
                    continue
                if key == "collected_tests" and collected_tests is None:
                    continue
                if key == "producer" and not _text(producer).strip():
                    continue
                same_stored_streams = (
                    stdout_text == existing.get("stdout")
                    and stderr_text == existing.get("stderr")
                )
                if (
                    key in {"output_truncated", "capture_complete", "capture_status"}
                    and output_truncated is None
                    and same_stored_streams
                ):
                    continue
                if key not in {"receipt_id", "full_output_handle", "execution_id", "host_execution_id"}:
                    merged[key] = value
            merged["receipt_id"] = handle
            merged["full_output_handle"] = handle
            merged["execution_id"] = execution_text
            merged["host_execution_id"] = execution_text
        merged.update(_capture_metadata(merged.get("output_truncated")))
        merged["summary"] = _summary(merged)
        private = merged
    enrichment_mode = check_receipt_enrichment_dedup_mode(runtime)
    receipt_unchanged = (
        isinstance(existing, Mapping)
        and _receipt_content_equal(private, existing)
    )
    if enrichment_mode == "shadow" and receipt_unchanged:
        _emit_shadow_mismatch(
            runtime,
            kind="check_receipt_enrichment_dedup_shadow_mismatch",
            execution_id=execution_text,
            field="emission",
            legacy_value="emit",
            proposed_value="suppress",
        )
    suppress_unchanged = enrichment_mode == "on" and receipt_unchanged
    previous_generation = existing.get("_stateframe_memo_generation", 0) \
        if isinstance(existing, Mapping) else 0
    if type(previous_generation) is not int or previous_generation < 0:
        previous_generation = 0
    if suppress_unchanged and isinstance(existing, Mapping):
        private = dict(existing)
    else:
        private["_stateframe_memo_generation"] = previous_generation + 1
        receipts[handle] = private
        _invalidate_check_receipt_memo(runtime, handle)
    public = _public_receipt(private, getattr(runtime, "journal", None))
    public_receipts = getattr(runtime, "host_check_receipts", None)
    if not isinstance(public_receipts, list):
        public_receipts = []
        setattr(runtime, "host_check_receipts", public_receipts)
    updated_public = False
    for index, existing_public in enumerate(public_receipts):
        if isinstance(existing_public, Mapping) and existing_public.get("receipt_id") == handle:
            public_receipts[index] = public
            updated_public = True
            break
    if not updated_public:
        public_receipts.append(public)
    # Keep the lossless run index in parallel with the legacy public list; the
    # list is a compatibility view and must never become the retention policy.
    if not isinstance(existing, Mapping):
        try:
            from kernel_recall import record_runtime_item
            record_runtime_item(
                runtime,
                "receipt",
                public,
                item_id=str(public.get("receipt_id", "")) or None,
                source_key="check-receipt:" + str(public.get("receipt_id", "")),
                step=public.get("sequence"),
            )
        except (ImportError, TypeError, ValueError):
            pass
    if not suppress_unchanged:
        _emit(runtime, public)
    setattr(runtime, "_latest_check_receipt", dict(public))
    _refresh_referencing_rows(runtime, public)
    return dict(public)


def _refresh_referencing_rows(runtime: object, public: Mapping[str, object]) -> None:
    """Push a settled canonical outcome to wrapper rows that reference it.

    A test receipt can be written while its execution is still pending (the
    native detail arrives later); it references the execution id and must not
    keep a stale UNKNOWN snapshot once the one canonical receipt settles.
    """
    outcome = str(public.get("outcome") or "").upper()
    if outcome not in _CANONICAL_STATUS:
        return
    execution = str(public.get("execution_id") or "")
    for collection in ("test_receipts",):
        rows = getattr(runtime, collection, None)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or str(row.get("execution_id") or "") != execution:
                continue
            ids = row.get("execution_ids")
            if isinstance(ids, list) and len(ids) > 1:
                continue  # aggregate rows are re-derived by canonical_identity
            row["outcome"] = outcome
            row["status"] = _CANONICAL_STATUS[outcome]


def retrieve_check_output(
        runtime: object,
        handle: object,
        *,
        stream: str = "combined",
        offset: int = 0,
        limit: int = DEFAULT_PAGE_CHARS,
) -> dict[str, object]:
    """Return a lossless page of stored output; refuse handles from another revision."""
    handle_text = _text(handle).strip()
    receipts = getattr(runtime, "_check_output_receipts", {})
    receipt = receipts.get(handle_text) if isinstance(receipts, dict) else None
    if not isinstance(receipt, Mapping):
        current = _current_revision(runtime)
        valid_handles = sorted(
            key for key, value in receipts.items()
            if isinstance(key, str) and isinstance(value, Mapping)
            and (current == "UNKNOWN" or value.get("revision") in {"UNKNOWN", current})
        ) if isinstance(receipts, dict) else []
        return {
            "schema": SCHEMA, "status": "NOT_FOUND", "full_output_handle": handle_text,
            "observation": f"no such check output: {handle_text}",
            "valid_handles": valid_handles,
        }
    current = _current_revision(runtime)
    if receipt.get("revision") not in {"UNKNOWN", current} and current != "UNKNOWN":
        return {
            "schema": SCHEMA,
            "status": "STALE_REVISION",
            "full_output_handle": handle_text,
            "receipt_revision": receipt.get("revision"),
            "current_revision": current,
        }
    if stream not in {"combined", "stdout", "stderr", "assertion_context"}:
        return {"schema": SCHEMA, "status": "INVALID_STREAM", "stream": stream}
    if type(offset) is not int or offset < 0:
        return {"schema": SCHEMA, "status": "INVALID_OFFSET", "offset": offset}
    if type(limit) is not int or limit < 1:
        return {"schema": SCHEMA, "status": "INVALID_LIMIT", "limit": limit}
    limit = min(limit, MAX_PAGE_CHARS)
    memo_mode = _check_receipt_memo_mode(runtime)
    memo_key: tuple[object, ...] | None = None
    cached_page: dict[str, object] | None = None
    if memo_mode != "off":
        try:
            memo_key = (
                _text(receipt.get("selector")),
                _text(receipt.get("revision")),
                _text(receipt.get("receipt_id") or handle_text),
                receipt.get("_stateframe_memo_generation", 0),
                current,
                stream,
                offset,
                limit,
            )
            memo = getattr(runtime, "_stateframe_check_receipt_memo", None)
            candidate = memo.get(memo_key) if isinstance(memo, dict) else None
            if isinstance(candidate, dict):
                cached_page = copy.deepcopy(candidate)
        except Exception:
            memo_key = None
            cached_page = None
        if memo_mode == "on" and cached_page is not None:
            return cached_page
    if stream == "stdout":
        content = str(receipt.get("stdout", ""))
    elif stream == "stderr":
        content = str(receipt.get("stderr", ""))
    elif stream == "assertion_context":
        content = str(receipt.get("assertion_context", ""))
    else:
        content = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
    from kernel_wire_identity import public_wire_text
    content = public_wire_text(content, getattr(runtime, "journal", None))
    page = content[offset:offset + limit]
    end = offset + len(page)
    page = {
        "schema": SCHEMA,
        "status": "OK",
        "full_output_handle": handle_text,
        "revision": receipt.get("revision"),
        "selector": receipt.get("selector"),
        "stream": stream,
        "offset": offset,
        "limit": limit,
        "total_chars": len(content),
        "next_offset": end if end < len(content) else None,
        "complete": end >= len(content),
        **_capture_metadata(receipt.get("output_truncated")),
        "content": page,
    }
    from kernel_wire_identity import public_wire_value
    page = public_wire_value(page, getattr(runtime, "journal", None))
    if (memo_mode != "off" and memo_key is not None
            and not (memo_mode == "shadow" and cached_page == page)):
        _store_check_receipt_page(runtime, memo_key, page)
    return page


def observe_check_output(runtime: object, owner: object, handle: object, *,
                         stream: str = "combined", offset: int = 0,
                         limit: int = DEFAULT_PAGE_CHARS) -> str:
    """Deliver a lookup outcome and journal its actual acquisition result."""
    feedback = retrieve_check_output(runtime, handle, stream=stream, offset=offset, limit=limit)
    result = ("HOST_OBSERVED" if feedback.get("status") == "OK" else
              "NOT_FOUND" if feedback.get("status") == "NOT_FOUND" else "REJECTED")
    text = json.dumps(feedback, ensure_ascii=False, sort_keys=True)
    owner.receipt("read", "read check receipt", str(handle or ""), result, text[:240])
    runtime.observe("read_check_receipt", text)
    return text


def compact_observation(receipt: Mapping[str, object]) -> str:
    """Render the model-facing summary without silently clipping any stream."""
    return str(receipt.get("summary") or _summary(receipt))


def _check_receipt_memo_mode(runtime: object) -> str:
    try:
        configured = getattr(runtime, "stateframe_check_receipt_memo_mode", None)
        value = str(configured if configured is not None else
                    os.environ.get("STATEFRAME_CHECK_RECEIPT_MEMO", "off")).strip().lower()
    except Exception:
        return "off"
    return value if value in {"off", "shadow", "on"} else "off"


def _invalidate_check_receipt_memo(runtime: object, receipt_id: str) -> None:
    try:
        memo = getattr(runtime, "_stateframe_check_receipt_memo", None)
        if not isinstance(memo, dict):
            return
        for key in tuple(memo):
            if isinstance(key, tuple) and len(key) >= 3 and key[2] == receipt_id:
                memo.pop(key, None)
    except Exception:
        # The optimization is disposable; receipt persistence is authoritative.
        pass


def _store_check_receipt_page(
        runtime: object, key: tuple[object, ...], page: dict[str, object],
) -> None:
    try:
        memo = getattr(runtime, "_stateframe_check_receipt_memo", None)
        if not isinstance(memo, dict):
            memo = {}
            setattr(runtime, "_stateframe_check_receipt_memo", memo)
        memo[key] = copy.deepcopy(page)
        while len(memo) > _CHECK_RECEIPT_MEMO_LIMIT:
            memo.pop(next(iter(memo)))
    except Exception:
        pass


__all__ = [
    "CHECK_EXECUTION_BINDING_ENV", "CHECK_RECEIPT_ENRICHMENT_DEDUP_ENV",
    "DEFAULT_PAGE_CHARS", "HANDLE_PREFIX", "MAX_PAGE_CHARS", "SCHEMA",
    "CHECK_OUTCOMES", "FAILURE_PHASES", "canonical_execution_id",
    "check_execution_binding_mode", "check_receipt_enrichment_dedup_mode",
    "classify_check_outcome",
    "compact_observation", "record_check_output", "retrieve_check_output",
]
