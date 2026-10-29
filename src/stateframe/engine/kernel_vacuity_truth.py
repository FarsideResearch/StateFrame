"""Journal projections for truthful, non-blocking verification certificates."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


MODE_ENV: Final = "STATEFRAME_VACUITY_TRUTH"


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


@dataclass(frozen=True)
class JournalTruth:
    last_status: str | None
    tested_rev: str | None
    current_rev: str
    changed_by: str | None
    changed_since_check: bool
    check_receipt_id: str | None


def _current_revision(binding: object, explicit_revision: object = None) -> str:
    if isinstance(explicit_revision, (str, int)) and explicit_revision != "":
        return str(explicit_revision)
    if isinstance(binding, Mapping):
        for key in ("product_revision", "revision", "artifact", "current_revision",
                    "current_rev", "rev"):
            value = binding.get(key)
            if isinstance(value, (str, int)) and value != "":
                return str(value)
    elif isinstance(binding, str) and binding:
        return binding
    return "UNKNOWN"


def _applied_edit_receipt(event: Mapping[str, object]) -> Mapping[str, object] | None:
    if event.get("kind") != "action_receipt":
        return None
    receipt = event.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    if str(receipt.get("kind", "")).strip().lower() != "edit":
        return None
    if str(receipt.get("result_class", "")).strip().upper() != "APPLIED":
        return None
    return receipt


def journal_applied_edit_rows(
        records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return one manifest row per canonical APPLIED edit action receipt."""
    result: list[dict[str, object]] = []
    for event in records:
        if not isinstance(event, Mapping):
            continue
        receipt = _applied_edit_receipt(event)
        if receipt is None:
            continue
        row: dict[str, object] = {
            "file": receipt.get("target_path"),
            "edit_receipt": receipt.get("receipt_id"),
            "journal_event_id": event.get("event_id"),
            "action_kind": receipt.get("kind"),
            "result_class": receipt.get("result_class"),
        }
        for key in ("workspace_revision", "request_id", "turn"):
            if key in receipt:
                row[key] = receipt[key]
        result.append(row)
    return result


def project_journal_truth(
        records: Sequence[Mapping[str, object]], current_binding: object,
        *, current_revision: object = None,
) -> JournalTruth:
    """Project latest check facts and any later APPLIED edit from a journal cut."""
    latest_check: Mapping[str, object] | None = None
    changed_since_check = False
    changed_by: str | None = None
    for event in records:
        if not isinstance(event, Mapping):
            continue
        if event.get("kind") == "check_receipt":
            latest_check = event
            changed_since_check = False
            changed_by = None
            continue
        receipt = _applied_edit_receipt(event)
        if latest_check is not None and receipt is not None:
            changed_since_check = True
            receipt_id = receipt.get("receipt_id")
            changed_by = str(receipt_id) if receipt_id not in (None, "") else None

    current_rev = _current_revision(current_binding, current_revision)
    if latest_check is None:
        return JournalTruth(None, None, current_rev, None, False, None)

    status = str(latest_check.get("status", "UNKNOWN")).strip().upper() or "UNKNOWN"
    tested_value = latest_check.get("revision", latest_check.get("tested_rev"))
    tested_rev = (str(tested_value)
                  if isinstance(tested_value, (str, int)) and tested_value != "" else None)
    revision_drift = (
        tested_rev is not None and current_rev != "UNKNOWN" and current_rev != tested_rev
    )
    receipt_id = latest_check.get("receipt_id")
    return JournalTruth(
        status, tested_rev, current_rev, changed_by,
        changed_since_check or revision_drift,
        str(receipt_id) if receipt_id not in (None, "") else None,
    )
