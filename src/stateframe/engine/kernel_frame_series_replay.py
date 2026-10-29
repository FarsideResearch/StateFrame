"""Checked frame-series reconstruction from an already authenticated journal prefix.

This helper does not authenticate journals or authorize worker resume. The caller
must supply the trusted prefix and a pristine cursor with the original config.
No caller state is mutated on failure; ParentMeter remains the live authority.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

from kernel_frame_series import FrameSeries
from kernel_measurement import digest


class SeriesReplayError(ValueError):
    """Historical series inputs disagree with closed actions or recorded state."""


def replay_series(initial: FrameSeries, folds: list[dict],
                  ledger: list[dict]) -> FrameSeries:
    if initial.public()["progress"]["actions"] != 0:
        raise SeriesReplayError("replay requires a pristine configured cursor")
    if len(folds) != len(ledger):
        raise SeriesReplayError("each closed action needs exactly one fold")
    cursor = FrameSeries(
        task_id=initial.scope["task_id"], path=initial.scope.get("path"),
        symbol=initial.scope.get("symbol"), scope=initial.scope,
        plan=initial.plan, work=initial.work,
        objective_revision=initial.objective_revision,
        parent_budget=initial.parent_budget, enabled=initial.enabled,
        control_enabled=initial.control_enabled,
        no_progress_threshold=initial.no_progress_threshold)
    try:
        for entry, record in zip(folds, ledger):
            inputs = copy.deepcopy(entry["fold_inputs"])
            if (entry["kind"] != "frame_series_folded"
                    or digest(inputs["record"]) != digest(record)
                    or digest(inputs["binding"]) != digest(record["after_binding"])
                    or inputs["phase_evidence_ref"] != record["step_record_id"]):
                raise SeriesReplayError("fold does not match its closed action")
            event = cursor.fold(**inputs)
            if digest(event) != digest(entry["event"]):
                raise SeriesReplayError("replayed event differs")
            actual, expected = cursor.public(), copy.deepcopy(entry["cursor"])
            # Historical budget snapshots are observations, never restoration inputs.
            actual.pop("budget")
            expected.pop("budget")
            if digest(actual) != digest(expected):
                raise SeriesReplayError("replayed cursor differs")
    except (KeyError, TypeError, ValueError) as error:
        raise SeriesReplayError("invalid frame-series replay") from error
    return cursor


def snapshot_series(adapter):
    if adapter is None:
        return None
    view = adapter.cursor.public()
    view.pop("budget")
    result = {"schema": "series-restart/1", "cursor": view,
              "restrictions": copy.deepcopy(adapter.restrictions)}
    if getattr(adapter, "stagnation_cap", None) is not None:
        result["stagnation_cap"] = adapter.stagnation_cap.snapshot()
    return result


def restore_series(runtime, saved, records, boundary):
    from kernel_continuation_restart import WorkerResumeError

    adapter = runtime.frame_series
    value = saved.get("frame_series")
    prefix = [e for e in records if e["sequence"] <= boundary["sequence"]]
    folds = [e for e in prefix if e["kind"] == "frame_series_folded"]
    if adapter is None:
        if value is not None or folds:
            raise WorkerResumeError("series state in disabled profile")
        return
    restored_cap = None
    try:
        closed = [e for e in prefix if e["kind"] == "action_outcome_closed"]
        if [e["record"] for e in closed] != saved["ledger"]:
            raise SeriesReplayError("series ledger differs from journal")
        if len(closed) != len(folds):
            raise SeriesReplayError("missing series fold")
        for i, (action, fold) in enumerate(zip(closed, folds)):
            end = closed[i + 1]["sequence"] if i + 1 < len(closed) else boundary["sequence"]
            if not action["sequence"] < fold["sequence"] < end:
                raise SeriesReplayError("series fold outside action boundary")
        restored = replay_series(adapter.cursor, folds, saved["ledger"])
        view = restored.public()
        view.pop("budget")
        if value["schema"] != "series-restart/1" or digest(view) != digest(value["cursor"]):
            raise SeriesReplayError("series checkpoint differs from journal")
        prepared = [e for e in prefix if e["kind"] == "frame_series_restrictions"]
        restrictions = prepared[-1]["restrictions"] if prepared else []
        if digest(restrictions) != digest(value["restrictions"]):
            raise SeriesReplayError("series restrictions differ from journal")
        if getattr(adapter, "stagnation_cap", None) is not None:
            from kernel_stagnation_cap import StagnationCap
            saved_cap = value.get("stagnation_cap")
            if not isinstance(saved_cap, Mapping):
                raise SeriesReplayError("stagnation cap checkpoint is missing")
            restored_cap = StagnationCap(
                adapter.cursor.objective_revision,
                window=adapter.stagnation_cap.window, enabled=True)
            restored_cap.restore(saved_cap)
        elif value.get("stagnation_cap") is not None:
            raise SeriesReplayError("stagnation cap state in disabled profile")
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("invalid series restoration") from error
    adapter.cursor = restored
    adapter.restrictions = copy.deepcopy(restrictions)
    if restored_cap is not None:
        adapter.stagnation_cap = restored_cap
