"""Host-owned answer receipts and investigation facts for StateFrame.

This module only formats already-recorded host observations.  It does not run
tests, infer a defect from model prose, or promote a model hypothesis to a
fact.  The caller gates delivery with ``STATEFRAME_RECEIPTS`` so the default
path remains byte-for-byte unchanged.
"""
from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_RECEIPTS"
_VALID_PROBE_STATUSES = {"RESOLVED", "NARROWED", "NO_DISCRIMINATION", "INCONCLUSIVE"}
_PASS_STATUSES = {"GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED", "REUSED_VERIFICATION"}
_RED_STATUSES = {"RED", "FAIL", "FAILED", "ERROR"}
_REVISION_KEYS = ("current_revision", "revision", "product_revision", "artifact_revision")
_COUNT_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
_PASSED_RE = re.compile(r"(?<!\d)(\d+)\s+passed\b", re.IGNORECASE)


def receipts_mode() -> str:
    """Return the strictly validated receipt presentation mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _as_rows(value: object) -> list[dict]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _revision(value: object) -> object:
    if isinstance(value, Mapping):
        for key in _REVISION_KEYS:
            if value.get(key) is not None:
                return value[key]
        for key in ("binding", "after_binding", "current_binding"):
            found = _revision(value.get(key))
            if found is not None:
                return found
    return value if value is not None and not isinstance(value, Mapping) else None


def _current_revision(frame: Mapping[str, object]) -> object:
    for key in _REVISION_KEYS:
        if frame.get(key) is not None:
            return frame[key]
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping) and len(revisions) == 1:
        return next(iter(revisions.values()))
    binding = frame.get("binding")
    return _revision(binding)


def _status(row: Mapping[str, object]) -> str:
    for key in ("result", "result_class", "status", "outcome", "verdict"):
        value = row.get(key)
        if value is not None:
            return str(value).upper()
    return "UNKNOWN"


def _is_test_row(row: Mapping[str, object]) -> bool:
    kind = str(row.get("kind", row.get("tool", ""))).lower()
    return kind in {"test", "run_tests", "pytest", "verification"} or any(
        key in row for key in ("selectors", "selector", "suite", "passed", "exit_code", "feedback")) \
        or str(row.get("result", "")).upper() in _PASS_STATUSES | _RED_STATUSES | {"REUSED_VERIFICATION"}


def _test_rows(frame: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    for key in ("test_receipts", "recent_test_feedback"):
        rows.extend(row for row in _as_rows(frame.get(key)) if _is_test_row(row))
    world = frame.get("world")
    if isinstance(world, Mapping) and isinstance(world.get("last_test"), Mapping):
        rows.append(dict(world["last_test"]))
    latest = frame.get("latest_receipts")
    if isinstance(latest, Mapping):
        rows.extend(row for row in latest.values() if isinstance(row, Mapping)
                    and _is_test_row(row))
    last = frame.get("last_action")
    if isinstance(last, Mapping) and _is_test_row(last):
        rows.append(dict(last))
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import collapse_check_results
        return collapse_check_results(rows)
    return rows


def _latest_test(frame: Mapping[str, object]) -> dict | None:
    rows = _test_rows(frame)
    if not rows:
        return None
    # A public ``last_action`` is a useful fallback, but a receipt/feedback
    # row carries the observation detail needed for counts, reuse, and proof.
    detailed = [row for row in rows if any(
        key in row for key in ("receipt_id", "event_id", "passed", "tests_passed",
                               "feedback", "observation", "detail"))]
    return detailed[-1] if detailed else rows[-1]


def _suite(row: Mapping[str, object]) -> str:
    for key in ("suite", "target", "selector"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    selectors = row.get("selectors")
    if isinstance(selectors, str) and selectors.strip():
        return selectors.strip()
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        values = [str(item) for item in selectors if str(item).strip()]
        if values:
            return ", ".join(values)
    parameters = row.get("parameters")
    if isinstance(parameters, Mapping):
        value = parameters.get("target") or parameters.get("selector")
        if value:
            return str(value)
    return "UNKNOWN_SUITE"


def _counts(row: Mapping[str, object]) -> tuple[int | None, int | None]:
    pairs = (("passed", "total"), ("tests_passed", "tests_total"),
             ("pass_count", "test_count"), ("passed_count", "total_count"))
    for passed_key, total_key in pairs:
        passed, total = row.get(passed_key), row.get(total_key)
        if type(passed) is int and type(total) is int:
            return passed, total
    text = " ".join(str(row.get(key, "")) for key in ("feedback", "observation", "detail"))
    match = _COUNT_RE.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = _PASSED_RE.search(text)
    if match:
        count = int(match.group(1))
        return count, count
    return None, None


def _row_revision(row: Mapping[str, object], frame: Mapping[str, object]) -> object:
    return _revision(row) or _current_revision(frame)


def _receipt_id(row: Mapping[str, object]) -> str:
    return str(row.get("receipt_id") or row.get("event_id") or row.get("id") or "UNKNOWN_RECEIPT")


def _has_witnessed_defect(frame: Mapping[str, object], row: Mapping[str, object] | None) -> bool:
    if row is not None and _status(row) in _RED_STATUSES:
        return True
    for key in ("known_violations", "known_violation", "host_observed_findings", "fault_receipts"):
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    review = frame.get("host_review")
    return isinstance(review, Mapping) and bool(review.get("known_violations"))


def _green_current(frame: Mapping[str, object], row: Mapping[str, object]) -> bool:
    if _status(row) not in _PASS_STATUSES:
        return False
    checked = _row_revision(row, frame)
    current = _current_revision(frame)
    if checked is None or current is None:
        return False
    return str(checked) == str(current)


def _answer_receipt(row: Mapping[str, object], frame: Mapping[str, object]) -> str:
    action = _suite(row)
    status = _status(row)
    revision = _row_revision(row, frame)
    revision_text = "UNKNOWN" if revision is None else str(revision)
    passed, total = _counts(row)
    if status in _PASS_STATUSES and passed is not None and total is not None:
        observed = f"{passed}/{total} tests in {action} passed on revision {revision_text}"
    elif status in _PASS_STATUSES:
        observed = f"tests in {action} passed on revision {revision_text}"
    else:
        from kernel_wire_identity import public_wire_text
        detail = public_wire_text(
            str(row.get("feedback") or row.get("observation") or status), None).splitlines()[0]
        observed = f"{action} reported {status} on revision {revision_text}: {detail[:180]}"
    reused = status == "REUSED_VERIFICATION" or str(row.get("result_class", "")).upper() == "REUSED_VERIFICATION"
    executions = 0 if reused else int(row.get("executions", 1)) if str(row.get("executions", "")).isdigit() else 1
    line = (f"Action {action} completed. Observed: {observed}. Current revision: "
            f"{revision_text}. Scope: those tests; not proof of every requirement. "
            f"Executions: {executions}.")
    if reused:
        line += " No new execution; no new sample."
    superseded = row.get("superseded")
    if isinstance(superseded, Sequence) and not isinstance(superseded, (str, bytes)) and superseded:
        revisions = ", ".join(
            f"{item.get('revision', 'UNKNOWN')} (receipt {item.get('receipt_id', 'UNKNOWN')})"
            for item in superseded
            if isinstance(item, Mapping)
        )
        line += (" Older results are superseded and collapsed; revisions: "
                 + (revisions or "UNKNOWN") + ".")
    return line


def _probe_rows(frame: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    for key in ("probe_receipts", "probe_results", "recent_probe_receipts"):
        rows.extend(_as_rows(frame.get(key)))
    for row in _as_rows(frame.get("receipts")):
        if str(row.get("kind", row.get("tool", ""))).lower() == "probe":
            rows.append(row)
    return rows


def classify_probe_exit(row: Mapping[str, object], *, prior: Mapping[str, object] | None = None) -> str:
    """Classify a probe from explicit host fields and its observed exit.

    A zero exit only means the probe ran.  Resolution/narrowing must be an
    explicit host observation; this avoids treating successful execution as
    proof of the hypothesis.
    """
    explicit = str(row.get("exit_status", row.get("probe_status", row.get("classification", "")))).upper()
    if explicit in _VALID_PROBE_STATUSES:
        return explicit
    if row.get("resolved") is True or row.get("disposition") == "resolved":
        return "RESOLVED"
    if row.get("narrowed") is True or row.get("disposition") == "narrowed":
        return "NARROWED"
    if prior is not None and _probe_identity(row) == _probe_identity(prior):
        return "NO_DISCRIMINATION"
    return "INCONCLUSIVE"


def _probe_identity(row: Mapping[str, object]) -> tuple[str, str, str]:
    source = _revision(row) or row.get("source_revision") or "UNKNOWN"
    test = row.get("test") or row.get("selector") or row.get("target") or row.get("obligation") or "UNKNOWN"
    outcome = row.get("outcome") or row.get("result") or row.get("observation") or row.get("exit_code") or "UNKNOWN"
    from kernel_wire_identity import public_wire_text
    return str(source), str(test), public_wire_text(str(outcome), None)


def _investigation_lines(frame: Mapping[str, object], row: Mapping[str, object] | None) -> tuple[list[str], list[str]]:
    revision = _row_revision(row, frame) if row is not None else _current_revision(frame)
    revision_text = "UNKNOWN" if revision is None else str(revision)
    suite = _suite(row) if row is not None else "UNKNOWN_SUITE"
    observed = "no test receipt observed"
    if row is not None:
        observed = (f"{_status(row)} for {suite} on revision {revision_text}; receipt "
                    f"{_receipt_id(row)}")
    question = str(frame.get("investigation_question") or
                   f"What explains the observed outcome for {suite}?")
    if row is not None and _status(row) in _PASS_STATUSES:
        ruled_out = f"a current failure of {suite} on candidate revision {revision_text}"
        next_discriminator = f"a distinct requirement beyond {suite}, because this receipt covers only those tests"
    elif row is not None and _status(row) in _RED_STATUSES:
        ruled_out = "no additional defect; the receipt establishes only the reported failure"
        next_discriminator = f"a fresh host observation that changes or localizes the outcome for {suite}"
    else:
        ruled_out = "nothing beyond the recorded receipt"
        next_discriminator = "a host observation with an explicit outcome and candidate revision"
    receipt_ids = []
    for candidate in _test_rows(frame):
        receipt_ids.append(_receipt_id(candidate))
    receipt_ids = list(dict.fromkeys(receipt_ids[-4:]))
    basis = f"candidate rev={revision_text}; receipts={','.join(receipt_ids) or 'none'}"
    lines = [
        "ACTIVE INVESTIGATION (host-maintained from receipts):",
        f"question: {question}",
        f"observed: {observed}",
        f"ruled out: {ruled_out}",
        f"next discriminator: {next_discriminator}",
        f"basis: {basis}",
    ]
    probe_rows = _probe_rows(frame)
    # Preserve the ordinary receipt path exactly when inspection is explicitly
    # off.  In particular, do not import or validate the opt-in module merely
    # because every receipt render shares this helper.
    stale_probe_inspect_on = False
    if os.environ.get("STATEFRAME_STALE_PROBE_INSPECT", "off") != "off":
        from kernel_stale_probe_inspect import (
            enabled as stale_probe_inspect_enabled,
            requested_receipt_id,
        )
        stale_probe_inspect_on = stale_probe_inspect_enabled()
    if stale_probe_inspect_on:
        from kernel_stale_probe_inspect import render_inspection_lines
        lines.extend(render_inspection_lines(frame))
        if probe_rows and not requested_receipt_id(frame):
            lines.append(
                f"probe receipts available: {len(probe_rows)}; request one by receipt id for inspection"
            )
    else:
        for index, probe in enumerate(probe_rows):
            prior = probe_rows[index - 1] if index else None
            status = classify_probe_exit(probe, prior=prior)
            lines.append(f"probe exit: {status}; test={probe.get('test', probe.get('selector', probe.get('obligation', 'UNKNOWN')))}")
    hypotheses = []
    for key in ("model_hypotheses", "hypotheses", "hypotheses_not_facts"):
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            hypotheses.extend(value)
    for hypothesis in hypotheses[-4:]:
        text = hypothesis.get("claim", hypothesis.get("text", "")) if isinstance(hypothesis, Mapping) else hypothesis
        if str(text).strip():
            lines.append(f"MODEL_HYPOTHESIS: {str(text).strip()[:240]}")
    advisories: list[str] = []
    if not stale_probe_inspect_on:
        for previous, current in zip(probe_rows, probe_rows[1:]):
            if classify_probe_exit(current, prior=previous) == "NO_DISCRIMINATION":
                source, test, outcome = _probe_identity(current)
                advisories.append(
                    f"ADVISORY: repeated NO_DISCRIMINATION probe; source revision {source}, "
                    f"test {test}, outcome {outcome}; no new discrimination.")
    return lines, list(dict.fromkeys(advisories))


def render_receipt_messages(frame: Mapping[str, object]) -> list[dict]:
    """Render answer-shaped receipts plus the investigation block."""
    if not isinstance(frame, Mapping):
        raise TypeError("receipt rendering requires a frame mapping")
    from kernel_completion_status import (
        completion_status_from_frame, enabled as blocker_render_enabled,
        render_completion_status,
    )
    if blocker_render_enabled():
        status = completion_status_from_frame(frame)
        return [{
            "role": "user",
            "content": "\n".join((
                "[STATEFRAME RECEIPTS | host facts]",
                "Public-check receipts are evidence only; completion status is the sole completion authority.",
                render_completion_status(status),
            )),
        }]
    row = _latest_test(frame)
    content: list[str] = ["[STATEFRAME RECEIPTS | host facts]"]
    if row is not None:
        content.append(_answer_receipt(row, frame))
        if _green_current(frame, row) and not _has_witnessed_defect(frame, row):
            content.append("done is admissible now if you judge the work complete")
    investigation, advisories = _investigation_lines(frame, row)
    content.extend(investigation)
    content.extend(advisories)
    # The control-plane adapter is an opt-in host projection.  It receives
    # already-recorded facts only; it never promotes the legacy receipt view or
    # changes completion authority.  Keeping the import here preserves the
    # default/off path for older releases and lightweight host fixtures.
    if os.environ.get("STATEFRAME_CONTROL_PLANE", "off") == "on":
        from kernel_control_plane_receipts import render_control_plane_message
        control_message = render_control_plane_message(frame)
        if control_message is not None:
            content.append(control_message)
    return [{"role": "user", "content": "\n".join(content)}]


def append_receipt_messages(messages: Sequence[dict], frame: Mapping[str, object], *,
                            provider_request: bool = True) -> list[dict]:
    """Append identical receipt messages to either provider arm when enabled."""
    result = copy.deepcopy(list(messages))
    if provider_request and receipts_mode() == "on":
        result.extend(render_receipt_messages(frame))
    return result


__all__ = [
    "MODE_ENV", "append_receipt_messages", "classify_probe_exit",
    "receipts_mode", "render_receipt_messages",
]
