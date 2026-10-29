"""Observation certificates for verification runs (rulebook candidate N5).

Inert until ``STATEFRAME_VACUITY_SENTINEL=on``.  Precision-first: the
certificate is an OBSERVATION rendered to the model and journaled; it never
vetoes completion.  Rules implemented (scope of this lane):

* canonical typed receipts and structured counts decide status; legacy output
  parsing remains a compatibility fallback when no receipt was supplied;
* ``observed == 0`` (incl. pytest exit 5 "no tests ran") is UNKNOWN, never PASS;
* an unparseable run is UNKNOWN with ``parse_failed``;
* every certificate is bound to the candidate binding it ran against; the
  legacy view uses ``GATE_NEVER_RAN`` after a candidate change, while
  ``STATEFRAME_VACUITY_TRUTH=on`` projects the journal's last check and names
  any later APPLIED edit so stale status remains explicit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final


MODE_ENV: Final = "STATEFRAME_VACUITY_SENTINEL"
SCHEMA: Final = "rulebook-vacuity-sentinel/2"
BLOCK_HEADER: Final = "[VERIFICATION CERTIFICATE]"

PASS: Final = "PASS"
FAIL: Final = "FAIL"
UNKNOWN: Final = "UNKNOWN"
GATE_NEVER_RAN: Final = "GATE_NEVER_RAN"

_SUMMARY = re.compile(
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
_COLLECTED = re.compile(r"collected\s+(?P<count>\d+)\s+items?")


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def binding_key(binding: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        binding, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservationCertificate:
    expected: int
    eligible: int
    observed: int
    excluded: int
    parse_failed: bool
    status: str
    reason: str
    selectors: tuple[str, ...] = ()
    binding: str = ""
    execution_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    structured_counts: tuple[tuple[str, int], ...] = ()
    outcomes: tuple[str, ...] = ()
    failure_phases: tuple[str, ...] = ()
    reached_observations: tuple[bool | None, ...] = ()
    execution_completeness: tuple[bool, ...] = ()
    termination_reasons: tuple[str, ...] = ()
    last_status: str | None = None
    tested_rev: str | None = None
    current_rev: str | None = None
    changed_by: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"expected": self.expected, "eligible": self.eligible,
                "observed": self.observed, "excluded": self.excluded,
                "parse_failed": self.parse_failed, "status": self.status,
                "reason": self.reason, "selectors": list(self.selectors),
                "binding": self.binding, "execution_ids": list(self.execution_ids),
                "receipt_ids": list(self.receipt_ids),
                "counts": dict(self.structured_counts),
                "outcomes": list(self.outcomes),
                "failure_phases": list(self.failure_phases),
                "reached_observations": list(self.reached_observations),
                "execution_completeness": list(self.execution_completeness),
                "termination_reasons": list(self.termination_reasons)}
        if self.last_status is not None and _vacuity_truth_enabled():
            result.update({"last_status": self.last_status,
                           "tested_rev": self.tested_rev,
                           "current_rev": self.current_rev,
                           "changed_by": self.changed_by})
            if _journal_metrics_unavailable(self):
                result.update({"expected": None, "eligible": None, "observed": None,
                               "excluded": None, "counts": None})
        return result


def _summary_line(text: str) -> str:
    """The LAST pytest summary line (``=== 3 passed, 1 skipped in 0.1s ===``)."""
    for line in reversed(text.splitlines()):
        stripped = line.strip().strip("=").strip()
        if not re.search(r"\bin \d+(?:\.\d+)?s\b", stripped):
            continue  # pytest's final line always ends "in <seconds>s" (-q or not)
        if _SUMMARY.search(stripped) or stripped.startswith("no tests ran"):
            return stripped
    return ""


def structured_pytest_counts(output: object) -> dict[str, int] | None:
    """Convert a complete, internally consistent pytest summary to typed counts."""
    if not isinstance(output, str):
        return None
    summary = _summary_line(output)
    if not summary:
        return None
    counts = {
        "passed": 0, "failed": 0, "errored": 0, "skipped": 0,
        "xfailed": 0, "xpassed": 0, "deselected": 0,
    }
    for match in _SUMMARY.finditer(summary):
        kind = match.group("kind")
        key = "errored" if kind.startswith("error") else kind
        counts[key] += int(match.group("count"))
    reported_total = sum(counts.values())
    if reported_total <= 0:
        return None
    collected_match = _COLLECTED.search(output)
    collected = int(collected_match.group("count")) if collected_match else reported_total
    if collected != reported_total:
        return None
    return {"collected": collected, **counts}


def certificate_from_output(output: object, *, exit_code: int | None,
                            selectors: Sequence[str] = (),
                            binding: object = None,
                            check_receipts: Sequence[Mapping[str, object]] | None = None) -> ObservationCertificate:
    """Use canonical host receipts when present; otherwise retain the legacy parser."""
    text = output if isinstance(output, str) else str(output or "")
    selected = tuple(str(item) for item in selectors)
    key = binding_key(binding) if binding is not None else ""
    if check_receipts is not None:
        return _certificate_from_check_receipts(check_receipts, selected, key)
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0,
              "xfailed": 0, "xpassed": 0, "deselected": 0}
    summary = _summary_line(text)
    for match in _SUMMARY.finditer(summary):
        kind = "error" if match.group("kind").startswith("error") else match.group("kind")
        counts[kind] += int(match.group("count"))
    collected_match = _COLLECTED.search(text)
    observed = (counts["passed"] + counts["failed"] + counts["error"]
                + counts["xfailed"] + counts["xpassed"])
    excluded = counts["skipped"] + counts["deselected"]
    eligible = int(collected_match.group("count")) if collected_match else observed + excluded
    no_run = exit_code == 5 or "no tests ran" in summary
    parse_failed = not summary and not no_run
    if exit_code is None:
        status, reason = UNKNOWN, "the host did not provide a completed exit code"
    elif no_run:
        status, reason = UNKNOWN, "the run collected/executed zero tests (pytest exit 5)"
    elif parse_failed:
        status, reason = UNKNOWN, "no pytest summary in the host run output"
    elif observed == 0:
        status, reason = UNKNOWN, "zero tests executed; skipped/deselected are not evidence"
    elif exit_code == 0 and counts["failed"] == 0 and counts["error"] == 0:
        status, reason = PASS, f"{observed} executed, all passed"
    else:
        status, reason = FAIL, f"{counts['failed']} failed, {counts['error']} errors"
    return ObservationCertificate(
        expected=len(selected), eligible=eligible, observed=observed, excluded=excluded,
        parse_failed=parse_failed, status=status, reason=reason, selectors=selected,
        binding=key)


def current_runtime_view(runtime: object) -> ObservationCertificate:
    """Build the runtime certificate view from journal truth only when enabled."""
    binding_method = getattr(runtime, "binding", None)
    binding = binding_method() if callable(binding_method) else {}
    certificate = getattr(runtime, "_vacuity_certificate", None)
    if not _vacuity_truth_enabled():
        return current_view(certificate, binding)

    journal = getattr(runtime, "journal", None)
    reader = getattr(journal, "framefit_receipt_records", None)
    if not callable(reader):
        return current_view(certificate, binding)
    try:
        records = reader()
    except Exception:
        return current_view(certificate, binding)
    current_revision = (
        binding.get("revision") if isinstance(binding, Mapping) else None
    )
    return current_view(
        certificate, binding, journal_records=records,
        current_revision=current_revision,
    )


def current_view(certificate: ObservationCertificate | None,
                 current_binding: object, *,
                 journal_records: Sequence[Mapping[str, object]] | None = None,
                 current_revision: object = None,
                 ) -> ObservationCertificate:
    """Project the certificate at the current candidate, optionally from journal truth."""
    key = binding_key(current_binding)
    if journal_records is not None and _vacuity_truth_enabled():
        try:
            from kernel_vacuity_truth import project_journal_truth
            truth = project_journal_truth(
                journal_records, current_binding, current_revision=current_revision)
            if truth.last_status is None:
                base = ObservationCertificate(
                    0, 0, 0, 0, False, GATE_NEVER_RAN,
                    "no check receipt exists in the journal", (), key)
                return replace(
                    base, last_status=GATE_NEVER_RAN,
                    current_rev=truth.current_rev)
            status = truth.last_status
            if truth.changed_since_check and truth.changed_by:
                tested_rev = truth.tested_rev or "UNKNOWN"
                reason = (f"LAST {status} at rev {tested_rev}; candidate changed by edit "
                          f"{truth.changed_by} since")
            elif truth.changed_since_check:
                reason = (f"LAST {status} at rev {truth.tested_rev or 'UNKNOWN'}; current "
                          f"candidate is rev {truth.current_rev}; no APPLIED edit receipt "
                          "follows the check")
            else:
                reason = f"LAST {status} at rev {truth.tested_rev or 'UNKNOWN'}"
            if certificate is None or certificate.status != status:
                base = ObservationCertificate(
                    0, 0, 0, 0, False, status, reason, (), key)
            else:
                base = replace(certificate, status=status, reason=reason, binding=key)
            return replace(
                base, last_status=status, tested_rev=truth.tested_rev,
                current_rev=truth.current_rev, changed_by=truth.changed_by)
        except Exception:
            # Keep legacy behavior when an optional journal projection is malformed.
            pass
    if certificate is None:
        return ObservationCertificate(0, 0, 0, 0, False, GATE_NEVER_RAN,
                                      "no verification has run yet", (), key)
    if certificate.binding != key:
        return ObservationCertificate(
            certificate.expected, 0, 0, 0, False, GATE_NEVER_RAN,
            f"the candidate changed since the last run (last: {certificate.status}); "
            "no verification at the current head", certificate.selectors, key,
            certificate.execution_ids, certificate.receipt_ids,
            certificate.structured_counts, certificate.outcomes,
            certificate.failure_phases, certificate.reached_observations,
            certificate.execution_completeness, certificate.termination_reasons)
    return certificate


def _certificate_from_check_receipts(
        receipts: Sequence[Mapping[str, object]],
        selectors: tuple[str, ...], binding: str,
) -> ObservationCertificate:
    """Present only typed child outcomes and explicit structured host counts."""
    raw_rows = tuple(row for row in receipts if isinstance(row, Mapping))
    rows_by_execution: dict[str, dict[str, object]] = {}
    identity_missing_rows: list[Mapping[str, object]] = []
    for row in raw_rows:
        execution_id = str(row.get("execution_id") or "").strip()
        if not execution_id:
            identity_missing_rows.append(row)
            continue
        previous = rows_by_execution.get(execution_id)
        if previous is not None:
            merged = dict(previous)
            merged.update(row)
            # A repeated wrapper view is metadata on the same child, not a
            # second execution and not authority to erase its typed verdict.
            if str(row.get("outcome", "UNKNOWN")).upper() == "UNKNOWN" and (
                    str(previous.get("outcome", "UNKNOWN")).upper() not in {"", "UNKNOWN"}):
                for key in (
                        "status", "outcome", "failure_phase", "reached_observation",
                        "execution_complete", "termination_reason", "counts",
                        "selector", "selector_scope"):
                    if key in previous:
                        merged[key] = previous[key]
            rows_by_execution[execution_id] = merged
        else:
            rows_by_execution[execution_id] = dict(row)
    rows = tuple(rows_by_execution.values()) + tuple(identity_missing_rows)
    execution_ids = tuple(
        str(row.get("execution_id")) for row in rows
        if row.get("execution_id") not in (None, ""))
    receipt_ids = tuple(
        str(row.get("receipt_id")) for row in rows
        if row.get("receipt_id") not in (None, ""))
    scope_values: list[str] = list(selectors)
    for row in rows:
        scope = row.get("selector_scope", ())
        values = (scope if isinstance(scope, Sequence)
                  and not isinstance(scope, (str, bytes, bytearray))
                  else (row.get("selector", ""),))
        for value in values:
            selector = str(value).strip()
            if selector and selector not in scope_values:
                scope_values.append(selector)
    scope = tuple(scope_values)
    maps = [row.get("counts") for row in rows]
    valid_counts = bool(rows) and all(
        isinstance(value, Mapping)
        and all(type(count) is int and count >= 0 for count in value.values())
        and {"collected", "passed", "failed"}.issubset(value)
        for value in maps)
    totals: dict[str, int] = {}
    if valid_counts:
        common_keys = set(maps[0])
        for value in maps[1:]:
            common_keys.intersection_update(value)
        totals = {key: sum(value[key] for value in maps) for key in common_keys}
    counts_tuple = tuple(sorted(totals.items()))
    eligible = totals.get("collected", 0) if valid_counts else 0
    excluded = totals.get("skipped", 0) + totals.get("deselected", 0) if valid_counts else 0
    observed = (totals.get("passed", 0) + totals.get("failed", 0)
                + totals.get("errored", 0) + totals.get("xfailed", 0)
                + totals.get("xpassed", 0)) if valid_counts else 0
    parse_failed = not rows or any(row.get("wrapper_parse_error") for row in rows)
    outcomes = [str(row.get("outcome", "UNKNOWN")).upper() for row in rows]
    failure_phases = tuple(str(row.get("failure_phase", "UNKNOWN")) for row in rows)
    reached = tuple(row.get("reached_observation")
                    if isinstance(row.get("reached_observation"), bool) else None
                    for row in rows)
    completeness = tuple(row.get("execution_complete") is True for row in rows)
    termination_reasons = tuple(
        str(row.get("termination_reason", "UNKNOWN")) for row in rows)
    identified_rows = [row for row in rows if row.get("execution_id") not in (None, "")]
    covered_selectors = {
        str(row.get("selector") or "").strip() for row in identified_rows
        if str(row.get("selector") or "").strip()
    }
    missing_scope = set(scope) - covered_selectors
    identity_incomplete = len(identified_rows) != len(rows)
    incomplete = any(
        row.get("execution_complete") is not True
        or outcome in {"UNKNOWN", "NOT_RUN", "RESOURCE_LIMIT", "ENVIRONMENT", "INVALID_CHECK"}
        for row, outcome in zip(rows, outcomes)
    )

    if "BEHAVIOR_FAIL" in outcomes:
        status, reason = FAIL, "a canonical child receipt recorded BEHAVIOR_FAIL"
    elif not rows:
        status, reason = UNKNOWN, "no canonical child receipt was available"
    elif identity_incomplete:
        status, reason = UNKNOWN, "one or more child receipts lack canonical execution identity"
    elif missing_scope:
        status, reason = UNKNOWN, "canonical child receipts do not cover every selector in scope"
    elif incomplete:
        status, reason = UNKNOWN, "one or more child executions are incomplete or unresolved"
    elif any(outcome != "PASS" for outcome in outcomes):
        status, reason = UNKNOWN, "canonical child outcomes do not establish PASS"
    elif not valid_counts:
        status, reason = UNKNOWN, "structured test counts are unavailable"
    elif eligible <= 0 or observed <= 0:
        status, reason = UNKNOWN, "zero tests were observed; this is not PASS evidence"
    elif totals.get("failed", 0) or totals.get("errored", 0):
        status, reason = FAIL, "structured host counts include failed or errored tests"
    elif (totals.get("passed", 0) + totals.get("xfailed", 0)
          + totals.get("xpassed", 0)) != eligible - excluded:
        status, reason = UNKNOWN, "structured counts do not settle every eligible test"
    else:
        status, reason = PASS, f"{observed} structured test observations had no failures"

    return ObservationCertificate(
        expected=len(scope), eligible=eligible, observed=observed, excluded=excluded,
        parse_failed=parse_failed, status=status, reason=reason, selectors=scope,
        binding=binding, execution_ids=execution_ids, receipt_ids=receipt_ids,
        structured_counts=counts_tuple, outcomes=tuple(outcomes),
        failure_phases=failure_phases, reached_observations=reached,
        execution_completeness=completeness, termination_reasons=termination_reasons,
    )


def render_certificate(certificate: ObservationCertificate) -> str:
    truth_text = ""
    metrics_unavailable = (
        certificate.last_status is not None
        and _vacuity_truth_enabled()
        and _journal_metrics_unavailable(certificate)
    )
    observed = "UNAVAILABLE" if metrics_unavailable else certificate.observed
    eligible = "UNAVAILABLE" if metrics_unavailable else certificate.eligible
    excluded = "UNAVAILABLE" if metrics_unavailable else certificate.excluded
    counts = "UNAVAILABLE" if metrics_unavailable else repr(dict(certificate.structured_counts))
    if certificate.last_status is not None and _vacuity_truth_enabled():
        truth_text = (
            f"; last_status={certificate.last_status!r}; "
            f"tested_rev={certificate.tested_rev!r}; "
            f"current_rev={certificate.current_rev!r}; "
            f"changed_by={certificate.changed_by!r}"
        )
    return (f"{BLOCK_HEADER} {certificate.status}: {certificate.reason} "
            f"(observed={observed}, eligible={eligible}, "
            f"excluded={excluded}; selectors={list(certificate.selectors)!r}; "
            f"execution_ids={list(certificate.execution_ids)!r}; "
            f"receipt_ids={list(certificate.receipt_ids)!r}; "
            f"outcomes={list(certificate.outcomes)!r}; "
            f"phases={list(certificate.failure_phases)!r}; "
            f"reached={list(certificate.reached_observations)!r}; "
            f"complete={list(certificate.execution_completeness)!r}; "
            f"termination={list(certificate.termination_reasons)!r}; "
            f"counts={counts}{truth_text}). "
            "Observation only; it never blocks done or certifies repair.")


def _journal_metrics_unavailable(certificate: ObservationCertificate) -> bool:
    return (
        certificate.last_status not in (None, GATE_NEVER_RAN)
        and not certificate.structured_counts
        and certificate.eligible == certificate.observed == certificate.excluded == 0
    )


def _vacuity_truth_enabled() -> bool:
    try:
        from kernel_vacuity_truth import enabled
        return enabled()
    except (ImportError, ValueError):
        return False
