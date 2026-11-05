"""Standalone grading adapter for non-coding rows.

The hot ``probes/converge.py`` integration can call ``grade_artifact`` without
knowing any class-specific truth logic.  This file intentionally has no model,
database, network, or pytest dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from build_tasks.noncoding_oracles import all_passed, evaluate_oracle_spec


def load_rows(path: str) -> list[dict[str, object]]:
    """Load noncoding JSONL rows and reject malformed top-level records."""
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: row must be an object")
            rows.append(value)
    return rows


def grade_artifact(
    row: Mapping[str, object],
    artifact: object,
    *,
    steps: int = 1,
    tokens: int = 0,
    latency_ms: int = 0,
) -> dict[str, object]:
    """Grade one model artifact and emit converge-compatible receipt fields."""
    row_id = str(row.get("id") or "unknown")
    source = row.get("input")
    spec = row.get("oracle_spec")
    if artifact is None:
        return {
            "id": row_id,
            "fn": row_id,
            "class": row.get("class"),
            "result": "NOT_ATTEMPTED",
            "behaviour": "not_attempted",
            "steps": steps,
            "edits": 0,
            "tests": 0,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "oracle_results": {},
        }
    if not isinstance(spec, Mapping):
        results = {"oracle_spec": {"pass": False, "reason": "oracle_spec_missing", "evidence": {}}}
    else:
        results = evaluate_oracle_spec(source, artifact, spec)
    passed = all_passed(results)
    return {
        "id": row_id,
        "fn": row_id,
        "class": row.get("class"),
        "result": "PASS" if passed else "FAIL",
        "behaviour": "PASS_behavioural" if passed else "tests_fail",
        "steps": steps,
        "edits": 0,
        "tests": len(results),
        "tokens": tokens,
        "latency_ms": latency_ms,
        "oracle_results": results,
        "artifact": artifact,
    }


__all__ = ["grade_artifact", "load_rows"]
