"""Opt-in release-comparison parity and all-attempt accounting checks.

The normal runtime and existing offline graders do not import this module's
switch-sensitive functions unless ``STATEFRAME_MATCHED_PROTOCOL`` is enabled.
That keeps the historical path byte-identical while giving an explicitly
requested comparison a small, shared contract for identity and denominators.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any


MATCHED_PROTOCOL_SWITCH = "STATEFRAME_MATCHED_PROTOCOL"
PROTOCOL_VERSION = "matched-comparison/1"
REQUIRED_SHARED_FIELDS = ("release", "model", "sampler", "tasks", "seeds")
OPTIONAL_SHARED_FIELDS = ("tools", "budgets", "restart", "grading", "capacity")
EXPECTED_RUNS_FIELD = "expected_runs"


class MatchedProtocolError(ValueError):
    """A comparison cannot support a matched release claim."""


def enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Return whether the comparison protocol was explicitly enabled."""
    values = os.environ if environ is None else environ
    return str(values.get(MATCHED_PROTOCOL_SWITCH, "off")).strip().lower() in {
        "1", "true", "yes", "on"
    }


switch_enabled = enabled


def _jsonable(value: object, *, field: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise MatchedProtocolError(f"{field} has a non-text key")
            result[key] = _jsonable(item, field=f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, field=field) for item in value]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise MatchedProtocolError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    raise MatchedProtocolError(f"{field} is not JSON-compatible")


def _canonical(value: object, *, field: str) -> str:
    try:
        return json.dumps(_jsonable(value, field=field), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, MatchedProtocolError):
            raise
        raise MatchedProtocolError(f"{field} is not canonical JSON") from exc


def _payload(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedProtocolError("matched protocol metadata must be an object")
    for key in ("matched_protocol", "protocol"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and not any(
                field in value for field in REQUIRED_SHARED_FIELDS):
            return nested
    return value


def _text(metadata: Mapping[str, Any], field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MatchedProtocolError(f"{field} is missing or empty")
    return value.strip()


def _shared_sequence(metadata: Mapping[str, Any], field: str) -> list[Any]:
    value = metadata.get(field)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise MatchedProtocolError(f"{field} must be a non-empty sequence")
    values = list(value)
    if not values:
        raise MatchedProtocolError(f"{field} must be a non-empty sequence")
    if field == "tasks":
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise MatchedProtocolError("tasks must contain non-empty strings")
        normalized = [item.strip() for item in values]
        if len(set(normalized)) != len(normalized):
            raise MatchedProtocolError("tasks must not contain duplicates")
        return sorted(normalized)
    if field == "seeds":
        if any(type(item) is not int or item < 0 for item in values):
            raise MatchedProtocolError("seeds must contain non-negative integers")
        if len(set(values)) != len(values):
            raise MatchedProtocolError("seeds must not contain duplicates")
        return sorted(values)
    return values


def normalize_metadata(metadata: Mapping[str, Any], *, expected_release: str | None = None,
                       arm: str | None = None) -> dict[str, Any]:
    """Validate and normalize one arm's shared comparison identity."""
    raw = _payload(metadata)
    normalized: dict[str, Any] = {
        "release": _text(raw, "release"),
        "model": _text(raw, "model"),
        "sampler": _jsonable(raw.get("sampler"), field="sampler"),
        "tasks": _shared_sequence(raw, "tasks"),
        "seeds": _shared_sequence(raw, "seeds"),
    }
    if normalized["sampler"] in (None, "", [], {}):
        raise MatchedProtocolError("sampler is missing or empty")
    if expected_release is not None and normalized["release"] != expected_release:
        label = f" for arm {arm!r}" if arm else ""
        raise MatchedProtocolError(
            f"release mismatch{label}: {normalized['release']!r} != {expected_release!r}"
        )
    for field in OPTIONAL_SHARED_FIELDS:
        if field in raw:
            value = _jsonable(raw[field], field=field)
            if value in (None, "", [], {}):
                raise MatchedProtocolError(f"{field} is present but empty")
            normalized[field] = value
    return normalized


def validate_shared_arms(arms: Mapping[str, Mapping[str, Any]], *,
                         expected_release: str | None = None) -> dict[str, Any]:
    """Require every listed arm to carry the same comparison identity."""
    if not isinstance(arms, Mapping) or not arms:
        raise MatchedProtocolError("matched comparison has no arm metadata")
    normalized_arms: dict[str, dict[str, Any]] = {}
    for arm, metadata in arms.items():
        if not isinstance(arm, str) or not arm.strip():
            raise MatchedProtocolError("matched comparison has an invalid arm name")
        normalized_arms[arm] = normalize_metadata(
            metadata, expected_release=expected_release, arm=arm
        )

    first_arm = next(iter(normalized_arms))
    shared = dict(normalized_arms[first_arm])
    for arm, metadata in normalized_arms.items():
        for field in REQUIRED_SHARED_FIELDS:
            if _canonical(metadata[field], field=field) != _canonical(shared[field], field=field):
                raise MatchedProtocolError(
                    f"{field} mismatch between arms {first_arm!r} and {arm!r}"
                )
    optional = {
        field for metadata in normalized_arms.values() for field in OPTIONAL_SHARED_FIELDS
        if field in metadata
    }
    for field in sorted(optional):
        if any(field not in metadata for metadata in normalized_arms.values()):
            missing = [arm for arm, metadata in normalized_arms.items() if field not in metadata]
            raise MatchedProtocolError(
                f"optional parity field {field} is missing for arm(s): {', '.join(missing)}"
            )
        for arm, metadata in normalized_arms.items():
            if _canonical(metadata[field], field=field) != _canonical(shared.get(field), field=field):
                raise MatchedProtocolError(
                    f"{field} mismatch between arms {first_arm!r} and {arm!r}"
                )
        shared[field] = normalized_arms[first_arm][field]
    return {"version": PROTOCOL_VERSION, "shared": shared, "arms": normalized_arms}


def metadata_from_protocol(
    protocol: Mapping[str, Any], *, fallback_tasks: Sequence[str] = (),
    fallback_seeds: Sequence[int] = (),
) -> dict[str, Any]:
    """Project a generated batch protocol onto the matched identity schema."""
    if not isinstance(protocol, Mapping):
        return {}
    nested = protocol.get("matched_protocol")
    if isinstance(nested, Mapping):
        protocol = nested
    runtime = protocol.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}

    def first(*values: object) -> object:
        return next((value for value in values if value is not None and value != ""), None)

    result: dict[str, Any] = {}
    release = first(runtime.get("release_sha256"), runtime.get("release"),
                    protocol.get("release_sha256"), protocol.get("release"),
                    protocol.get("release_id"))
    model = first(runtime.get("model"), protocol.get("model"))
    sampler = first(runtime.get("sampling"), runtime.get("sampler"),
                    protocol.get("sampling"), protocol.get("sampler"))
    tasks = protocol.get("tasks")
    task_ids: list[str] = []
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        for task in tasks:
            if isinstance(task, Mapping):
                task = task.get("task_id", task.get("id", task.get("task")))
            if isinstance(task, str) and task.strip():
                task_ids.append(task.strip())
    if not task_ids:
        task_ids = [str(task).strip() for task in fallback_tasks if str(task).strip()]
    seeds = protocol.get("seeds")
    seed_values = list(seeds) if isinstance(seeds, Sequence) and not isinstance(
        seeds, (str, bytes, bytearray)) else list(fallback_seeds)
    if release is not None:
        result["release"] = release
    if model is not None:
        result["model"] = model
    if sampler is not None:
        result["sampler"] = sampler
    if task_ids:
        result["tasks"] = task_ids
    if seed_values:
        result["seeds"] = seed_values

    tool_surface = first(runtime.get("tool_surface"), protocol.get("tools"), protocol.get("tool_surface"))
    if tool_surface is not None:
        result["tools"] = tool_surface
    for field in ("budgets", "restart", "restart_conditions", "grading", "context", "capacity"):
        value = protocol.get(field)
        if field == "restart_conditions" and value is not None:
            field = "restart"
        if field == "context" and value is not None and "capacity" not in result:
            field = "capacity"
        if value is not None:
            result[field] = value
    runs = protocol.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes, bytearray)):
        result[EXPECTED_RUNS_FIELD] = list(runs)
    elif runs is not None:
        # Preserve malformed declarations so the opt-in validator fails closed
        # instead of silently falling back to a discovered subset.
        result[EXPECTED_RUNS_FIELD] = runs
    return result


def expected_attempts(metadata: Mapping[str, Any], *, arm: str | None = None) -> list[dict[str, Any]]:
    """Return the frozen attempt cohort, never the set of surviving records."""
    normalized = normalize_metadata(metadata, arm=arm)
    raw = _payload(metadata).get(EXPECTED_RUNS_FIELD)
    if raw is None:
        return [
            {"task": task, "seed": seed}
            for task in normalized["tasks"]
            for seed in normalized["seeds"]
        ]
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise MatchedProtocolError("expected_runs must be a sequence")

    entries: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise MatchedProtocolError("expected_runs contains a non-object")
        item_arm = item.get("arm")
        if arm is not None and item_arm not in (None, arm):
            raise MatchedProtocolError(
                f"expected attempt has arm {item_arm!r}, expected {arm!r}"
            )
        run = item.get("run", item.get("label", item.get("name", item.get("id"))))
        task = item.get("task", item.get("task_id", item.get("fixture_id")))
        seed = item.get("seed")
        if not isinstance(run, str) or not run.strip():
            raise MatchedProtocolError("every expected attempt needs a stable run name")
        run = run.strip()
        if run in seen_runs:
            raise MatchedProtocolError(f"expected_runs contains duplicate run {run!r}")
        if not isinstance(task, str) or not task.strip():
            raise MatchedProtocolError("expected attempt lacks task identity")
        if type(seed) is not int or seed < 0:
            raise MatchedProtocolError(f"expected attempt {run!r} lacks an integer seed")
        task = task.strip()
        if task not in normalized["tasks"] or seed not in normalized["seeds"]:
            raise MatchedProtocolError(f"expected attempt {run!r} is outside the task/seed cohort")
        seen_runs.add(run)
        entries.append({"run": run, "task": task, "seed": seed})
    if not entries:
        raise MatchedProtocolError("expected_runs declares no attempts")

    expected_pairs = set(itertools.product(normalized["tasks"], normalized["seeds"]))
    observed_pairs = {(item["task"], item["seed"]) for item in entries}
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise MatchedProtocolError(
            f"expected_runs does not cover the frozen cohort; missing={missing!r} extra={extra!r}"
        )
    return entries


def expected_attempt_count(metadata: Mapping[str, Any], *, arm: str | None = None) -> int:
    return len(expected_attempts(metadata, arm=arm))


def validate_run_identity(expected: Mapping[str, Any], actual: Mapping[str, Any], *,
                          arm: str | None = None, run: str | None = None) -> dict[str, Any]:
    """Compare effective run settings with the frozen arm declaration."""
    expected_normalized = normalize_metadata(expected, arm=arm)
    actual_normalized = normalize_metadata(actual, expected_release=expected_normalized["release"],
                                           arm=arm)
    for field in REQUIRED_SHARED_FIELDS:
        if _canonical(actual_normalized[field], field=field) != _canonical(
                expected_normalized[field], field=field):
            label = f" for run {run!r}" if run else ""
            raise MatchedProtocolError(f"{field} mismatch{label} in arm {arm!r}")
    return actual_normalized


def _record_protocol(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("matched_protocol", "protocol", "comparison_protocol"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    raise MatchedProtocolError(
        f"attempt {record.get('run', record.get('task', '<unknown>'))!r} lacks matched protocol metadata"
    )


def _record_task_seed(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, int]:
    task = metadata.get("task", record.get("task"))
    seed = metadata.get("seed", record.get("seed"))
    record_task = record.get("task")
    if record_task is not None and task is not None and record_task != task:
        raise MatchedProtocolError("attempt task identity disagrees with its protocol metadata")
    record_seed = record.get("seed")
    if record_seed is not None and seed is not None and record_seed != seed:
        raise MatchedProtocolError("attempt seed identity disagrees with its protocol metadata")
    if not isinstance(task, str) or not task.strip():
        raise MatchedProtocolError("matched attempt lacks task identity")
    if type(seed) is not int or seed < 0:
        raise MatchedProtocolError(f"matched attempt {task!r} lacks an integer seed")
    return task.strip(), seed


def _correct(record: Mapping[str, Any]) -> bool:
    if record.get("grade_fresh") is not True:
        return False
    grade = record.get("grade")
    if not isinstance(grade, Mapping):
        return False
    return str(grade.get("outcome", "")).upper() == "PASS" and grade.get("accepted_done") is True


def _output_cost(record: Mapping[str, Any]) -> int | float:
    perf = record.get("perf")
    if not isinstance(perf, Mapping) or perf.get("output_complete") is not True:
        raise MatchedProtocolError(
            f"attempt {record.get('run', record.get('task', '<unknown>'))!r} lacks complete output cost"
        )
    value = perf.get("output_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise MatchedProtocolError("output cost must be a non-negative number for all attempts")
    if isinstance(value, float) and not math.isfinite(value):
        raise MatchedProtocolError("output cost must be finite for all attempts")
    return value


def validate_release_comparison(
    attempts: Sequence[Mapping[str, Any]], report: Mapping[str, Any], *,
    collection: Mapping[str, Any] | None = None,
    expected_release: str | None = None,
) -> dict[str, Any]:
    """Validate identity, task/seed coverage, and all-attempt cost accounting."""
    if not isinstance(report, Mapping) or not isinstance(report.get("arms"), Mapping):
        raise MatchedProtocolError("comparison report has no arm summaries")
    report_arms = report["arms"]
    if len(report_arms) < 2:
        raise MatchedProtocolError("matched comparison requires at least two arms")
    declared: dict[str, Mapping[str, Any]] = {}
    if isinstance(collection, Mapping):
        envelope = collection.get("matched_protocol")
        if isinstance(envelope, Mapping):
            version = envelope.get("version")
            if version is not None and version != PROTOCOL_VERSION:
                raise MatchedProtocolError(
                    f"unsupported matched protocol version: {version!r}"
                )
            raw_arms = envelope.get("arms")
            if raw_arms is None:
                raw_arms = {key: value for key, value in envelope.items()
                            if key != "version"}
            if not isinstance(raw_arms, Mapping) or not raw_arms:
                raise MatchedProtocolError("matched comparison envelope has no arm metadata")
            for arm, value in raw_arms.items():
                if not isinstance(arm, str) or not isinstance(value, Mapping):
                    raise MatchedProtocolError(
                        "matched comparison envelope has malformed arm metadata"
                    )
                declared[arm] = value
    by_arm: dict[str, list[Mapping[str, Any]]] = {str(arm): [] for arm in report_arms}
    for record in attempts:
        if not isinstance(record, Mapping):
            raise MatchedProtocolError("comparison attempt is not an object")
        arm = str(record.get("arm", ""))
        if arm not in by_arm:
            raise MatchedProtocolError(f"attempt belongs to unknown arm {arm!r}")
        by_arm[arm].append(record)
        record_metadata = _record_protocol(record)
        if arm not in declared:
            declared[arm] = record_metadata

    if set(declared) != set(by_arm):
        raise MatchedProtocolError("declared arm metadata does not match comparison arms")
    identity = validate_shared_arms(declared, expected_release=expected_release)
    normalized_by_arm = identity["arms"]
    observed_pairs: dict[str, set[tuple[str, int]]] = {arm: set() for arm in by_arm}
    expected_pairs_by_arm: dict[str, set[tuple[str, int]]] = {}
    expected_counts_by_arm: dict[str, dict[tuple[str, int], int]] = {}
    for arm, rows in by_arm.items():
        expected_metadata = normalized_by_arm.get(arm)
        if expected_metadata is None:
            raise MatchedProtocolError(f"arm {arm!r} has no protocol metadata")
        raw_declared = declared[arm]
        expected = expected_attempts(raw_declared, arm=arm)
        expected_pairs = {(item["task"], item["seed"]) for item in expected}
        expected_pairs_by_arm[arm] = expected_pairs
        expected_counts: dict[tuple[str, int], int] = {}
        for item in expected:
            pair = (item["task"], item["seed"])
            expected_counts[pair] = expected_counts.get(pair, 0) + 1
        expected_counts_by_arm[arm] = expected_counts
        expected_by_run = {item["run"]: item for item in expected if "run" in item}
        actual_by_run: dict[str, tuple[str, int]] = {}
        for record in rows:
            raw = _record_protocol(record)
            normalized = normalize_metadata(raw, expected_release=expected_release, arm=arm)
            for field in REQUIRED_SHARED_FIELDS:
                if _canonical(normalized[field], field=field) != _canonical(
                        expected_metadata[field], field=field):
                    raise MatchedProtocolError(f"{field} mismatch within arm {arm!r}")
            task, seed = _record_task_seed(record, raw)
            if task not in expected_metadata["tasks"] or seed not in expected_metadata["seeds"]:
                raise MatchedProtocolError(
                    f"attempt {task!r}/{seed} is outside the declared tasks/seeds"
                )
            observed_pairs[arm].add((task, seed))
            if record.get("grade_fresh") is not True:
                raise MatchedProtocolError(
                    f"grade for attempt {record.get('run', task)!r} is not fresh"
                )
            if expected_by_run:
                run = record.get("run")
                if not isinstance(run, str) or not run.strip():
                    raise MatchedProtocolError("matched attempt lacks stable run identity")
                if run in actual_by_run:
                    raise MatchedProtocolError(f"duplicate attempt run {run!r}")
                expected_run = expected_by_run.get(run)
                if expected_run is None:
                    raise MatchedProtocolError(f"attempt run {run!r} is outside the frozen cohort")
                if (task, seed) != (expected_run["task"], expected_run["seed"]):
                    raise MatchedProtocolError(f"attempt run {run!r} has the wrong task/seed")
                actual_by_run[run] = (task, seed)
        if expected_by_run:
            if set(actual_by_run) != set(expected_by_run):
                missing = sorted(set(expected_by_run) - set(actual_by_run))
                extra = sorted(set(actual_by_run) - set(expected_by_run))
                raise MatchedProtocolError(
                    f"arm {arm!r} is missing frozen attempts; missing={missing!r} extra={extra!r}"
                )
        elif len(rows) != len(expected):
            missing = sorted(expected_pairs - observed_pairs[arm])
            extra = sorted(observed_pairs[arm] - expected_pairs)
            raise MatchedProtocolError(
                f"arm {arm!r} does not contain every frozen attempt: "
                f"expected={len(expected)} observed={len(rows)} "
                f"missing={missing!r} extra={extra!r}"
            )
        if observed_pairs[arm] != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs[arm])
            extra = sorted(observed_pairs[arm] - expected_pairs)
            raise MatchedProtocolError(
                f"arm {arm!r} has incomplete task/seed coverage; missing={missing!r} extra={extra!r}"
            )
        if not rows:
            raise MatchedProtocolError(f"arm {arm!r} has no attempts")
    pair_sets = list(expected_pairs_by_arm.values())
    if any(pairs != pair_sets[0] for pairs in pair_sets[1:]):
        raise MatchedProtocolError("arms do not share the same frozen task/seed cohort")
    pair_counts = list(expected_counts_by_arm.values())
    if any(counts != pair_counts[0] for counts in pair_counts[1:]):
        raise MatchedProtocolError("arms do not share the same frozen retry cohort")

    accounting: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        summary = report_arms.get(arm)
        if not isinstance(summary, Mapping):
            raise MatchedProtocolError(f"report has no summary for arm {arm!r}")
        correct = sum(_correct(record) for record in rows)
        total = sum(_output_cost(record) for record in rows)
        if summary.get("attempts") != len(rows):
            raise MatchedProtocolError(f"cost accounting for {arm!r} does not include all attempts")
        if summary.get("correct_deliveries") != correct:
            raise MatchedProtocolError(f"correct denominator mismatch for arm {arm!r}")
        resources = summary.get("resources")
        resource = resources.get("output") if isinstance(resources, Mapping) else None
        if (not isinstance(resource, Mapping) or resource.get("complete") is not True
                or resource.get("missing_attempts") != 0
                or resource.get("total") != total):
            raise MatchedProtocolError(f"output cost for {arm!r} does not include all attempts")
        costs = summary.get("cost_per_correct")
        if not isinstance(costs, Mapping) or "output" not in costs:
            raise MatchedProtocolError(f"cost per correct is missing for arm {arm!r}")
        expected_cost = (float(total) / correct) if correct else None
        actual_cost = costs.get("output")
        if expected_cost is None:
            if actual_cost is not None:
                raise MatchedProtocolError(
                    f"cost per correct for {arm!r} must be undefined when no delivery is correct"
                )
        elif (not isinstance(actual_cost, (int, float)) or isinstance(actual_cost, bool)
              or not math.isclose(float(actual_cost), expected_cost, rel_tol=0, abs_tol=1e-9)):
            raise MatchedProtocolError(f"cost per correct denominator is wrong for arm {arm!r}")
        accounting[arm] = {
            "attempts": len(rows),
            "correct_deliveries": correct,
            "output_total": total,
            "cost_per_correct": expected_cost,
            "observed_pairs": [list(pair) for pair in sorted(observed_pairs[arm])],
        }
    return {
        "version": PROTOCOL_VERSION,
        "switch": MATCHED_PROTOCOL_SWITCH,
        "shared": identity["shared"],
        "arms": accounting,
        "cost": {
            "metric": "output_tokens",
            "numerator": "all_attempts",
            "denominator": "correct_deliveries",
        },
    }


def validate_score_rows(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the long-horizon grader's assigned-row cost summary."""
    if not isinstance(report, Mapping):
        raise MatchedProtocolError("grade report is not an object")
    if any(row.get("assigned") is not True or row.get("excluded") is True for row in rows):
        raise MatchedProtocolError("all assigned attempts must remain in the cost denominator")
    costs = [row.get("cost_proxy_tokens") for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
           or (isinstance(value, float) and not math.isfinite(value))
           for value in costs):
        raise MatchedProtocolError("cost is missing for an assigned attempt")
    total = sum(costs)
    correct = sum(row.get("correct") is True for row in rows)
    expected = float(total) / correct if correct else None
    if report.get("assigned_runs") != len(rows) or report.get("correct_runs") != correct:
        raise MatchedProtocolError("grade summary denominator does not include all attempts")
    if report.get("total_cost_proxy_tokens") != total or report.get("cost_per_correct") != expected:
        raise MatchedProtocolError("grade summary cost per correct is not all-attempt accounting")
    if report.get("cost_per_correct_defined") is not (correct > 0):
        raise MatchedProtocolError("grade summary cost-defined flag is inconsistent")
    return {
        "version": PROTOCOL_VERSION,
        "attempts": len(rows),
        "correct_deliveries": correct,
        "total_cost": total,
        "cost_per_correct": expected,
        "numerator": "all_attempts",
        "denominator": "correct_deliveries",
    }


def validate_grade_protocol(
    protocol: Mapping[str, Any], assignments: Sequence[Mapping[str, Any]], *,
    expected_release: str | None = None,
) -> dict[str, Any]:
    """Validate one grade batch's frozen protocol and assignment coverage."""
    metadata = metadata_from_protocol(protocol)
    result = validate_shared_arms({"batch": metadata}, expected_release=expected_release)
    expected = expected_attempts(metadata, arm=(
        protocol.get("batch_arm") if isinstance(protocol.get("batch_arm"), str) else None
    ))
    expected_by_run = {item["run"]: item for item in expected if "run" in item}
    observed_pairs: set[tuple[str, int]] = set()
    observed_names: set[str] = set()
    for row in assignments:
        task = row.get("task", row.get("task_id", row.get("fixture_id")))
        seed = row.get("seed")
        if (not isinstance(task, str) or not task.strip()
                or type(seed) is not int or seed < 0):
            raise MatchedProtocolError(f"grade assignment is outside the frozen task/seed set: {row!r}")
        pair = (task.strip(), seed)
        if expected_by_run:
            name = row.get("name", row.get("run", row.get("label")))
            if not isinstance(name, str) or name not in expected_by_run:
                raise MatchedProtocolError(f"grade assignment run is outside the frozen cohort: {row!r}")
            if name in observed_names:
                raise MatchedProtocolError(f"duplicate grade assignment: {row!r}")
            observed_names.add(name)
            expected_row = expected_by_run[name]
            if pair != (expected_row["task"], expected_row["seed"]):
                raise MatchedProtocolError(f"grade assignment identity changed: {row!r}")
        else:
            if pair in observed_pairs:
                raise MatchedProtocolError(f"duplicate grade assignment: {row!r}")
            observed_pairs.add(pair)
        batch_arm = protocol.get("batch_arm")
        if isinstance(batch_arm, str) and row.get("arm") not in (None, batch_arm):
            raise MatchedProtocolError(f"grade assignment arm changed: {row!r}")
    if expected_by_run:
        if observed_names != set(expected_by_run):
            raise MatchedProtocolError("grade assignments do not cover the frozen cohort")
    elif observed_pairs != {(item["task"], item["seed"]) for item in expected}:
        raise MatchedProtocolError("grade assignments do not cover the frozen cohort")
    return result


__all__ = [
    "MATCHED_PROTOCOL_SWITCH",
    "EXPECTED_RUNS_FIELD",
    "OPTIONAL_SHARED_FIELDS",
    "PROTOCOL_VERSION",
    "REQUIRED_SHARED_FIELDS",
    "MatchedProtocolError",
    "enabled",
    "metadata_from_protocol",
    "normalize_metadata",
    "expected_attempts",
    "expected_attempt_count",
    "validate_run_identity",
    "switch_enabled",
    "validate_grade_protocol",
    "validate_release_comparison",
    "validate_score_rows",
    "validate_shared_arms",
]
