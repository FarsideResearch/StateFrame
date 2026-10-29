"""Thin native/automatic handoff to the canonical check reducer and receipt store."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from kernel_check_receipts import (
    canonical_execution_id,
    check_execution_binding_mode,
    record_check_output,
)
from kernel_vacuity_sentinel import structured_pytest_counts


def begin_check_execution(
        runtime: object, *, selector: object, sequence: object = None,
        execution_id: object = None, binding: Mapping[str, object] | None = None,
        selector_scope: object = None,
) -> str:
    """Allocate a fresh child identity, or preserve an explicitly supplied one."""
    identity = canonical_execution_id(
        runtime, execution_id=execution_id, selector=selector, sequence=sequence,
        fresh=execution_id in (None, ""),
    )
    setattr(runtime, "_pending_check_execution_id", identity)
    pin_check_execution_context(
        runtime, identity, binding=binding, selector=selector,
        selector_scope=selector_scope,
    )
    return identity


def pin_check_execution_context(
        runtime: object, execution_id: object, *,
        binding: Mapping[str, object] | None, selector: object = "",
        selector_scope: object = None,
) -> None:
    """Keep the first detached pre-launch binding for an execution identity."""
    identity = str(execution_id or "").strip()
    if (not identity or not isinstance(binding, Mapping)
            or check_execution_binding_mode(runtime) == "off"):
        return
    try:
        contexts = getattr(runtime, "_stateframe_check_execution_contexts", None)
        if isinstance(contexts, Mapping):
            if identity in contexts:
                return
            contexts = dict(contexts)
        else:
            contexts = {}
        raw_scope = selector_scope
        if isinstance(raw_scope, Sequence) and not isinstance(raw_scope, (str, bytes, bytearray)):
            scope = tuple(str(item).strip() for item in raw_scope if str(item).strip())
        else:
            scope = ()
        if not scope and str(selector or "").strip():
            scope = (str(selector).strip(),)
        contexts[identity] = {
            "binding": copy.deepcopy(dict(binding)),
            "selector_scope": copy.deepcopy(scope),
        }
        setattr(runtime, "_stateframe_check_execution_contexts", contexts)
    except Exception:
        # The optimization is disposable; failure leaves today's revision path intact.
        return


def pending_check_execution_id(runtime: object) -> str | None:
    value = getattr(runtime, "_pending_check_execution_id", None)
    return str(value) if value not in (None, "") else None


def finish_check_execution(runtime: object, execution_id: object) -> None:
    """Close only the matching child context; never clear another attempt."""
    identity = str(execution_id or "")
    try:
        contexts = getattr(runtime, "_stateframe_check_execution_contexts", None)
        if isinstance(contexts, dict):
            contexts.pop(identity, None)
    except Exception:
        pass
    if identity and pending_check_execution_id(runtime) == identity:
        try:
            delattr(runtime, "_pending_check_execution_id")
        except AttributeError:
            pass


def record_native_observation(
        runtime: object, *, execution_id: str, selector: object, status: object,
        stdout: object = "", stderr: object = "", assertion_context: object = "",
        revision: object = None, sequence: object = None,
        reached_observation: object = None, failure_phase: object = None,
        outcome: object = None, exit_code: object = None,
        collected_tests: object = None, assertion_reached: object = None,
        product_observed: object = None,
        typed_observation: Mapping[str, object] | None = None,
        wrapper_parse_error: object = None,
        counts: object = None, selector_scope: object = None,
        execution_complete: object = None, termination_reason: object = None,
        output_truncated: object = None,
        producer: object = None,
        finalize: bool = True,
) -> dict[str, object]:
    """Forward one native observation without creating an adapter-side verdict."""
    row = record_check_output(
        runtime,
        selector=selector,
        status=status,
        stdout=stdout,
        stderr=stderr,
        assertion_context=assertion_context,
        revision=revision,
        sequence=sequence,
        reached_observation=reached_observation,
        failure_phase=failure_phase,
        outcome=outcome,
        execution_id=execution_id,
        exit_code=exit_code,
        collected_tests=collected_tests,
        assertion_reached=assertion_reached,
        product_observed=product_observed,
        typed_observation=typed_observation,
        counts=counts,
        selector_scope=selector_scope,
        execution_complete=execution_complete,
        termination_reason=termination_reason,
        output_truncated=output_truncated,
        producer=producer or "kernel_check_truth_adapter",
        wrapper_parse_error=wrapper_parse_error,
    )
    if finalize:
        finish_check_execution(runtime, execution_id or row.get("execution_id"))
        _refresh_vacuity_view(runtime, row)
    return row


def record_wrapper_parse_error(
        runtime: object, *, execution_id: str, selector: object,
        selector_scope: object = None, error: object,
        sequence: object = None, finalize: bool = False,
) -> dict[str, object]:
    """Attach wrapper diagnostics as metadata on the existing child receipt."""
    row = record_check_output(
        runtime, execution_id=execution_id, selector=selector, selector_scope=selector_scope,
        sequence=sequence, status="UNKNOWN", wrapper_parse_error=error,
        producer="kernel_check_truth_adapter",
    )
    if finalize:
        finish_check_execution(runtime, execution_id)
        _refresh_vacuity_view(runtime, row)
    return row


def receipt_ids(runtime: object) -> frozenset[str]:
    rows = getattr(runtime, "host_check_receipts", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return frozenset()
    return frozenset(
        str(row.get("receipt_id")) for row in rows
        if isinstance(row, Mapping) and row.get("receipt_id") not in (None, "")
    )


def receipts_since(runtime: object, previous_ids: Sequence[str] | set[str] | frozenset[str]) -> tuple[dict[str, object], ...]:
    previous = {str(item) for item in previous_ids}
    rows = getattr(runtime, "host_check_receipts", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    return tuple(
        dict(row) for row in rows
        if isinstance(row, Mapping)
        and str(row.get("receipt_id") or "")
        and str(row.get("receipt_id")) not in previous
    )


def automatic_result_identity(
        runtime: object, previous_ids: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, object]:
    """Canonical identity of the check receipts one automatic run recorded.

    The automatic runner result carries this reference so its consumers never
    re-derive a verdict from the aggregated exit code or observation text.
    """
    from kernel_check_receipts import canonical_identity
    return canonical_identity(receipts_since(runtime, previous_ids))


def handoff_automatic_result(
        runtime: object, *, selectors: Sequence[object], result: object,
        previous_receipt_ids: Sequence[str] | set[str] | frozenset[str],
) -> tuple[dict[str, object], ...]:
    """Connect automatic child receipts to their aggregate selector scope.

    Native producers that already record children keep their IDs.  A generic
    runner may instead return typed ``check_observations`` or per-selector
    ``exit_codes``; each such child is recorded on the same canonical rail.
    Unstructured/malformed output remains UNKNOWN and gets no inferred PASS.
    """
    scope = tuple(str(item).strip() for item in selectors if str(item).strip())
    previous = {str(item) for item in previous_receipt_ids}
    existing_rows = receipts_since(runtime, previous)
    result_map = result if isinstance(result, Mapping) else {}
    wrapper_error = result_map.get("wrapper_parse_error", result_map.get("parse_error"))

    if existing_rows:
        updated: list[dict[str, object]] = []
        for existing in existing_rows:
            execution_id = str(existing.get("execution_id") or "")
            selector = existing.get("selector", ", ".join(scope))
            counts = _counts_for_child(
                result_map, existing, str(selector),
                include_result_stdout=len(existing_rows) == 1)
            row = record_check_output(
                runtime, execution_id=execution_id, selector=selector,
                selector_scope=scope, sequence=existing.get("sequence"),
                status="UNKNOWN", counts=counts,
                producer=str(existing.get("producer") or "stateframe_auto_verify"),
            )
            if wrapper_error:
                row = record_wrapper_parse_error(
                    runtime, execution_id=execution_id, selector=selector,
                    selector_scope=scope, sequence=existing.get("sequence"),
                    error=wrapper_error,
                )
            finish_check_execution(runtime, execution_id)
            updated.append(row)
        return tuple(updated)

    if not scope:
        return ()

    children = result_map.get("check_observations", result_map.get("children"))
    child_rows: list[Mapping[str, object]] = []
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes, bytearray)):
        child_rows.extend(child for child in children if isinstance(child, Mapping))
    else:
        codes = result_map.get("exit_codes")
        if (isinstance(codes, Sequence) and not isinstance(codes, (str, bytes, bytearray))
                and len(codes) == len(scope)):
            child_rows.extend({"selector": selector, "exit_code": code}
                              for selector, code in zip(scope, codes))
        elif any(key in result_map for key in ("outcome", "status", "exit_code")):
            child_rows.append(result_map)

    rows: list[dict[str, object]] = []
    for index, child in enumerate(child_rows):
        selector = str(child.get("selector") or (
            scope[index] if len(child_rows) == len(scope) else ", ".join(scope)))
        execution_id = str(child.get("execution_id") or begin_check_execution(
            runtime, selector=selector, sequence=child.get("sequence")))
        code = child.get("exit_code")
        raw_outcome = child.get("outcome")
        status = child.get("status")
        if status is None:
            status = raw_outcome or _status_from_exit(code)
        typed = child.get("typed_observation")
        typed_observation = typed if isinstance(typed, Mapping) else None
        row = record_native_observation(
            runtime,
            execution_id=execution_id,
            selector=selector,
            selector_scope=scope,
            status=status,
            stdout=child.get("stdout", ""),
            stderr=child.get("stderr", ""),
            assertion_context=child.get("assertion_context", ""),
            sequence=child.get("sequence"),
            reached_observation=child.get("reached_observation"),
            failure_phase=child.get("failure_phase"),
            outcome=raw_outcome,
            exit_code=code,
            collected_tests=child.get("collected_tests"),
            assertion_reached=child.get("assertion_reached"),
            product_observed=child.get("product_observed"),
            typed_observation=typed_observation,
            counts=_counts_for_child(
                result_map, child, selector, include_result_stdout=len(child_rows) == 1),
            execution_complete=child.get("execution_complete"),
            termination_reason=child.get("termination_reason"),
            producer="stateframe_auto_verify",
            wrapper_parse_error=child.get(
                "wrapper_parse_error", child.get("parse_error", wrapper_error)),
        )
        rows.append(row)

    if rows:
        return tuple(rows)

    # A wrapper with no typed child record has not established a product result.
    selector = ", ".join(scope)
    execution_id = pending_check_execution_id(runtime) or begin_check_execution(
        runtime, selector=selector)
    error = wrapper_error or "automatic verification returned no typed child result"
    return (record_wrapper_parse_error(
        runtime, execution_id=execution_id, selector=selector,
        selector_scope=scope, error=error, finalize=True,
    ),)


def _counts_for_selector(result: Mapping[str, object], selector: str) -> object:
    values = result.get("counts_by_selector")
    if isinstance(values, Mapping):
        return values.get(selector)
    children = result.get("check_observations", result.get("children"))
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes, bytearray)):
        return result.get("counts") if len(children) == 1 else None
    codes = result.get("exit_codes")
    if isinstance(codes, Sequence) and not isinstance(codes, (str, bytes, bytearray)):
        return result.get("counts") if len(codes) == 1 else None
    return result.get("counts")


def _counts_for_child(
        result: Mapping[str, object], child: Mapping[str, object], selector: str,
        *, include_result_stdout: bool,
) -> object:
    counts = child.get("counts")
    if counts is None or counts == {}:
        counts = _counts_for_selector(result, selector)
    if counts is not None and counts != {}:
        return counts
    outputs = [child.get("stdout")]
    if include_result_stdout and child is not result:
        outputs.append(result.get("stdout"))
    for output in outputs:
        parsed = structured_pytest_counts(output)
        if parsed is not None:
            return parsed
    return counts


def _status_from_exit(code: object) -> str:
    if type(code) is not int:
        return "UNKNOWN"
    if code == 0:
        return "PASS"
    if code == 1:
        return "FAIL"
    if code == 5:
        return "NOT_RUN"
    if code in {124, 130, 134, 137, 143, 152, 153}:
        return "RESOURCE_LIMIT"
    return "ERROR"


def _refresh_vacuity_view(runtime: object, receipt: Mapping[str, object]) -> None:
    if getattr(runtime, "stateframe_vacuity_sentinel_mode", "off") != "on":
        return
    check_receipts = (receipt,)
    try:
        from kernel_vacuity_sentinel import certificate_from_output, current_view
        private_rows = getattr(runtime, "_check_output_receipts", {})
        private = (private_rows.get(receipt.get("receipt_id"))
                   if isinstance(private_rows, Mapping) else None)
        selectors = receipt.get("selector_scope", (receipt.get("selector", ""),))
        binding_fn = getattr(runtime, "binding", None)
        binding = binding_fn() if callable(binding_fn) else None
        certificate = certificate_from_output(
            private.get("stdout", "") if isinstance(private, Mapping) else "",
            exit_code=receipt.get("exit_code"),
            selectors=selectors if isinstance(selectors, Sequence) else (),
            binding=binding,
            check_receipts=check_receipts,
        )
        from kernel_vacuity_truth import enabled as vacuity_truth_enabled
        try:
            use_journal_truth = vacuity_truth_enabled()
        except ValueError:
            use_journal_truth = False
        if use_journal_truth:
            journal = getattr(runtime, "journal", None)
            # The exact journal projection preserves field presence and values.
            # FrameFit's compact receipt view is lossy and cannot establish
            # vacuity truth for long or nullable revision fields.
            exact_records = getattr(journal, "vacuity_truth_records", None)
            full_records = getattr(journal, "cached_records", None)
            if callable(exact_records) or callable(full_records):
                try:
                    from kernel_check_receipts import _current_revision
                    try:
                        bounded = getattr(journal, "_record_cache_limit", None) is not None
                        records = (exact_records() if bounded and callable(exact_records)
                                   else full_records())
                    except Exception:
                        # An incomplete projection falls back to validated
                        # journal records, never to a compact approximation.
                        if not callable(full_records):
                            raise
                        records = full_records()
                    certificate = current_view(
                        certificate, binding, journal_records=records,
                        current_revision=_current_revision(runtime))
                except Exception:
                    # Journal truth is supplemental; preserve the completed check certificate.
                    pass
        setattr(runtime, "_vacuity_certificate", certificate)
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as error:
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit("check_truth_vacuity_refresh_failed",
                 producer="kernel_check_truth_adapter",
                 error_type=type(error).__name__, dispatch=False)


__all__ = [
    "begin_check_execution", "finish_check_execution", "handoff_automatic_result",
    "pin_check_execution_context",
    "pending_check_execution_id", "receipt_ids", "receipts_since",
    "record_native_observation", "record_wrapper_parse_error",
]
