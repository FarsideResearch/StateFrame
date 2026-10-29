"""Production adapters for the opt-in instrument-validation gates.

The study workers own process boundaries, while ``kernel_instrument_validation``
owns the evidence vocabulary and validation rules.  This module is the narrow
bridge: it appends a nonce to an already-open run journal and gates a grading
batch loaded from the worker's declared artifact.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


_RESERVED_JOURNAL_FIELDS = frozenset({
    "schema",
    "kind",
    "run_id",
    "task_id",
    "event_id",
    "sequence",
    "timestamp_ns",
})


MODE_ENV = "STATEFRAME_INSTRUMENT_VALIDATION"


def validation_requested(spec: Mapping[str, object] | None = None) -> bool:
    """Resolve process and leaf-spec gates without letting an invalid value pass."""
    values = [os.environ.get(MODE_ENV, "off")]
    if spec is not None:
        environment = spec.get("environment")
        if isinstance(environment, Mapping) and MODE_ENV in environment:
            values.append(str(environment[MODE_ENV]))
    normalized = [str(value).strip().lower() for value in values]
    if any(value not in {"off", "on"} for value in normalized):
        raise ValueError(f"{MODE_ENV} must be 'off' or 'on'")
    return any(value == "on" for value in normalized)


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class _ExistingJournal:
    """Append through the same on-disk event contract after the worker exits."""

    def __init__(self, path: Path):
        from kernel_measurement import resolve_journal_blob_refs

        self.path = path
        records = []
        blob_cache: dict[str, object] = {}
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records.append(resolve_journal_blob_refs(json.loads(line), path, blob_cache))
        if not records or any(not isinstance(record, dict) for record in records):
            raise ValueError("instrument validation requires a non-empty journal")
        run_ids = {record.get("run_id") for record in records}
        task_ids = {record.get("task_id") for record in records}
        if len(run_ids) != 1 or len(task_ids) != 1:
            raise ValueError("instrument validation journal identity is not singular")
        self.run_id = next(iter(run_ids))
        self.task_id = next(iter(task_ids))
        if not isinstance(self.run_id, str) or not isinstance(self.task_id, str):
            raise ValueError("instrument validation journal identity is malformed")
        sequences = [record.get("sequence") for record in records]
        if any(type(sequence) is not int or sequence < 0 for sequence in sequences):
            raise ValueError("instrument validation journal sequence is malformed")
        self.sequence = max(sequences, default=0)

    def emit(self, kind: str, **fields: object) -> Mapping[str, object]:
        if _RESERVED_JOURNAL_FIELDS.intersection(fields):
            raise ValueError("nonce adapter cannot override reserved journal fields")
        from kernel_measurement import externalize_journal_fields

        self.sequence += 1
        record: dict[str, object] = {
            **fields,
            "schema": "stateframe-audit/1",
            "kind": kind,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "event_id": f"{self.run_id}:{self.sequence}",
            "sequence": self.sequence,
            "timestamp_ns": time.time_ns(),
        }
        stored_record = {
            **externalize_journal_fields(fields, self.path),
            **{key: record[key] for key in _RESERVED_JOURNAL_FIELDS},
        }
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(stored_record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def _journal_path(folder: Path) -> Path:
    root = folder / "measurement-runs"
    candidates = sorted(root.glob("*/events.jsonl"))
    if len(candidates) != 1:
        raise ValueError(
            "instrument validation requires exactly one completed measurement journal"
        )
    path = candidates[0]
    if path.is_symlink() or path.resolve().parent.parent != root.resolve():
        raise ValueError("measurement journal must remain inside the attempt folder")
    return path


def emit_worker_nonce_for_run(
    folder: str | Path, spec: Mapping[str, object]
) -> dict[str, object] | None:
    """Append and verify the worker nonce on the real solver journal."""
    if not validation_requested(spec):
        return None
    from kernel_instrument_validation import (
        InstrumentValidationError,
        emit_worker_nonce,
        new_nonce,
        validate_journal,
    )

    attempt = str(spec.get("attempt") or "run").strip()
    worker_id = str(spec.get("instrument_validation_worker_id") or f"worker-{attempt}")
    nonce_value = spec.get("instrument_validation_nonce")
    nonce = str(nonce_value).strip() if nonce_value is not None else new_nonce("r201")
    if not nonce or not worker_id:
        raise ValueError("instrument validation nonce and worker ID must be non-empty")
    path = _journal_path(Path(folder))
    journal = _ExistingJournal(path)
    event = emit_worker_nonce(journal, nonce, worker_id)
    if event is None:
        raise RuntimeError("instrument validation nonce was unexpectedly disabled")
    report = validate_journal(
        path,
        expected_nonce=nonce,
        expected_worker_id=worker_id,
    )
    if not report.valid:
        raise InstrumentValidationError(report)
    return {
        "schema": "instrument-validation-adapter/1",
        "worker_nonce": nonce,
        "worker_id": worker_id,
        "journal": str(path),
        "journal_validation": report.as_dict(),
    }


def _case_artifact(spec: Mapping[str, object]) -> list[Mapping[str, object]]:
    inline = spec.get("instrument_validation_cases")
    if inline is None:
        instrument = spec.get("instrument_validation")
        inline = instrument.get("cases") if isinstance(instrument, Mapping) else None
    if inline is not None:
        raw_cases = inline
    else:
        relative = spec.get("instrument_validation_cases_path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(
                "instrument validation requires instrument_validation_cases or "
                "instrument_validation_cases_path when enabled"
            )
        folder = Path(str(spec["folder"])).resolve()
        path = (folder / relative).resolve()
        if path.is_symlink() or path.parent != folder:
            raise ValueError("instrument validation case artifact must belong to the attempt")
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes, bytearray)):
        raise ValueError("instrument validation case artifact must be a JSON array")
    cases = [case for case in raw_cases if isinstance(case, Mapping)]
    if len(cases) != len(raw_cases):
        raise ValueError("instrument validation case artifact contains a non-object row")
    return cases


def preflight_grading_artifact(
    spec: Mapping[str, object],
) -> dict[str, object] | None:
    """Run the grading gate before the private grader consumes candidate rows."""
    if not validation_requested(spec):
        return None
    from kernel_instrument_validation import preflight_grading_batch

    declared_environment = spec.get("environment")
    declared_on = (
        isinstance(declared_environment, Mapping)
        and str(declared_environment.get(MODE_ENV, "off")).strip().lower() == "on"
    )
    previous_mode = os.environ.get(MODE_ENV)
    changed_mode = declared_on and str(previous_mode or "off").strip().lower() != "on"
    if changed_mode:
        os.environ[MODE_ENV] = "on"

    try:
        pairs_value = spec.get("instrument_validation_aa_pairs")
        pairs: list[tuple[int, int]] | None = None
        if pairs_value is not None:
            if not isinstance(pairs_value, Sequence) or isinstance(
                pairs_value, (str, bytes, bytearray)
            ):
                raise ValueError("instrument validation A/A pairs must be an array")
            pairs = []
            for pair in pairs_value:
                if (
                    not isinstance(pair, Sequence)
                    or isinstance(pair, (str, bytes, bytearray))
                    or len(pair) != 2
                    or type(pair[0]) is not int
                    or type(pair[1]) is not int
                ):
                    raise ValueError("instrument validation A/A pair must contain two integers")
                pairs.append((pair[0], pair[1]))
        report = preflight_grading_batch(_case_artifact(spec), aa_pairs=pairs)
        return {
            "schema": "instrument-validation-adapter/1",
            "grading_validation": report.as_dict(),
        }
    finally:
        if changed_mode:
            if previous_mode is None:
                os.environ.pop(MODE_ENV, None)
            else:
                os.environ[MODE_ENV] = previous_mode


__all__ = [
    "emit_worker_nonce_for_run",
    "preflight_grading_artifact",
    "validation_requested",
]
