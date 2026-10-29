"""Current delivery requirements, historical evictions, and bounded seal recovery."""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib

from kernel_request_seal import RequestSealError

_RANGE_FIELDS = frozenset({"path", "revision", "start_line", "end_line"})


class SourceVisibilityRecoveryError(RequestSealError):
    """The one final-wire claim reconciliation could not establish delivery."""
    reason_code = "SOURCE_VISIBILITY_RECOVERY_EXHAUSTED"

    def __init__(self, check, receipt):
        super().__init__(self.reason_code + ": candidate source visibility is unqualified")
        self.visibility_check = check
        self.visibility_receipt = receipt


def _range_key(row):
    from kernel_read_reuse_visibility import _SHA256

    if not isinstance(row, Mapping):
        return None
    path, revision = row.get("path"), row.get("revision")
    start, end = row.get("start_line"), row.get("end_line")
    if (not isinstance(path, str) or not path or not isinstance(revision, str)
            or _SHA256.fullmatch(revision) is None
            or type(start) is not int or type(end) is not int or not 1 <= start <= end):
        return None
    return path, revision, start, end


def evicted_ranges(capture, runtime):
    """Use bounded host eviction records, never model-authored notices."""
    records = []
    for owner in (capture, runtime):
        rows = getattr(owner, "_read_reuse_visibility_dropped", None)
        if isinstance(rows, list):
            records.extend(row for row in rows if _range_key(row) is not None)
    return records


def _range_is_owed(row, sources):
    """Use REREAD_B's revision rule for seal visibility requirements."""
    key = _range_key(row)
    if key is None:
        return True
    from kernel_reread_result_truth import read_is_owed

    path, revision, _start, _end = key
    current_revision = revision
    source = sources.get(path) if isinstance(sources, Mapping) else None
    if isinstance(source, str):
        from kernel_request_source_window import source_revision

        current_revision = source_revision(source)
    return read_is_owed({
        "path": path,
        "read_revision": revision,
        "current_revision": current_revision,
    })


def retire_evicted_ranges(check, evicted, sources=None):
    """Retire exact host evictions and edited ranges using shared read truth."""
    result = copy.deepcopy(check)
    selected = {_range_key(row) for row in check.get("candidates", ())}
    retired_keys = {_range_key(row) for row in evicted} - selected - {None}
    retired = list(result.get("retired_ranges", ()))
    recorded_keys = {_range_key(row) for row in retired}
    missing = []
    for row in result.get("missing_ranges", ()):
        key = _range_key(row)
        no_longer_owed = key is not None and not _range_is_owed(row, sources)
        if ((key in retired_keys or no_longer_owed) and isinstance(row, Mapping)
                and set(row) == _RANGE_FIELDS):
            if key not in recorded_keys:
                retired.append(row)
                recorded_keys.add(key)
        else:
            missing.append(row)
    result.update(missing_ranges=missing, retired_ranges=retired, result=not missing)
    return result


def qualify_visibility(capture, runtime, wire, check, delivery, sources):
    """Prove final claims; retry once against unchanged bytes and current claims."""
    from kernel_read_reuse_visibility import verify_final_visibility
    from kernel_request_candidate import selected_visibility

    selected = selected_visibility(check, delivery, wire=wire, sources=sources)
    receipt = verify_final_visibility(wire, selected)
    if receipt["result"] is True:
        return selected, receipt

    emit = capture.journal.emit
    fields = {
        "producer": "kernel_request_visibility", "request_id": capture.request_id,
        "step": capture.step, "attempt": 1, "dispatch": False,
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }
    emit("request_visibility_recovery", **fields, status="attempted",
         missing_range_count=len(receipt["missing_ranges"]))
    recovered = retire_evicted_ranges(check, evicted_ranges(capture, runtime), sources)
    selected = selected_visibility(recovered, delivery, wire=wire, sources=sources)
    receipt = verify_final_visibility(wire, selected)
    succeeded = receipt["result"] is True
    emit("request_visibility_recovery", **fields,
         status="recovered" if succeeded else "exhausted",
         missing_range_count=len(receipt["missing_ranges"]))
    if not succeeded:
        raise SourceVisibilityRecoveryError(selected, receipt)
    return selected, receipt
