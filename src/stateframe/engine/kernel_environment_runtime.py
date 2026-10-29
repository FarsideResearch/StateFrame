"""Minimal glue between environment state and the existing StateFrame runtime."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_environment_observer import EnvironmentObserver
from kernel_environment_panel import EnvironmentPanel, PanelDelivery
from kernel_environment_state import EnvironmentSnapshot, EnvironmentState, enabled


def _receipt_stream(runtime: object) -> tuple[object, ...]:
    owner = getattr(runtime, "owner", None)
    rows: list[object] = list(getattr(owner, "_receipts", ()) or ())
    for name in (
        "test_receipts", "probe_receipts", "observation_probe_receipts",
        "host_check_receipts", "host_repair_check_receipts",
    ):
        rows.extend(getattr(runtime, name, ()) or ())
    return tuple(rows)


def attach(runtime: object) -> EnvironmentState | None:
    """Attach one state owner; no attachment is made while the switch is off."""
    if not enabled():
        return None
    owner = getattr(runtime, "owner", None)
    root = Path(getattr(owner, "root", Path.cwd())).resolve()
    journal = getattr(runtime, "journal", None)
    raw_manifest = getattr(journal, "manifest", {})
    manifest = raw_manifest if isinstance(raw_manifest, Mapping) else {}
    run_id = str(getattr(journal, "run_id", None) or manifest.get("run_id", "run:unknown"))
    task_row = getattr(owner, "task_row", {})
    state = EnvironmentState(
        root,
        journal=journal,
        run_id=run_id,
        workspace_id=str(manifest.get("workspace_id") or "workspace:" + root.name),
    )
    state.bind_job_watch(getattr(runtime, "job_watch", None))
    observer = EnvironmentObserver.from_task_row(state, task_row)
    panel = EnvironmentPanel()
    setattr(runtime, "environment_state", state)
    setattr(runtime, "environment_observer", observer)
    setattr(runtime, "environment_panel", panel)
    setattr(runtime, "_environment_snapshot", None)
    setattr(runtime, "_environment_panel_delivery", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("environment_state_attached", producer="kernel_environment_runtime", dispatch=False)
    return state


def snapshot(runtime: object, *, refresh: bool = True) -> EnvironmentSnapshot | None:
    state = getattr(runtime, "environment_state", None)
    if not isinstance(state, EnvironmentState):
        return None
    if refresh:
        observer = getattr(runtime, "environment_observer", None)
        if isinstance(observer, EnvironmentObserver):
            observer.collect(receipts=_receipt_stream(runtime))
    current = state.snapshot(receipts=_receipt_stream(runtime))
    setattr(runtime, "_environment_snapshot", current)
    return current


def attach_frame(runtime: object, frame: Mapping[str, object]) -> dict[str, object]:
    current = snapshot(runtime)
    projected = dict(frame)
    if current is not None:
        projected["environment"] = current.to_dict()
    return projected


def _current_snapshot(runtime: object, frame: Mapping[str, object] | None = None) -> EnvironmentSnapshot | None:
    current = getattr(runtime, "_environment_snapshot", None)
    if isinstance(current, EnvironmentSnapshot):
        return current
    return snapshot(runtime)


def panel_content(runtime: object, frame: Mapping[str, object] | None = None) -> str | None:
    panel = getattr(runtime, "environment_panel", None)
    if not isinstance(panel, EnvironmentPanel):
        return None
    current = _current_snapshot(runtime, frame)
    if current is None:
        return None
    delivery = panel.prepare(current)
    setattr(runtime, "_environment_panel_delivery", delivery)
    return delivery.content if delivery is not None else None


def acknowledge_panel(runtime: object, prepared: Sequence[Mapping[str, object]], *, delivered: bool) -> None:
    if not delivered:
        return
    panel = getattr(runtime, "environment_panel", None)
    delivery = getattr(runtime, "_environment_panel_delivery", None)
    if not isinstance(panel, EnvironmentPanel) or not isinstance(delivery, PanelDelivery):
        return
    if not any(message.get("content") == delivery.content for message in prepared):
        return
    panel.acknowledge(delivery)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("environment_panel_delivered", producer="kernel_environment_runtime",
             panel_digest=delivery.digest, dispatch=False)


def readiness_snapshot(runtime: object) -> EnvironmentSnapshot | None:
    return _current_snapshot(runtime)


def cleanup(runtime: object) -> None:
    state = getattr(runtime, "environment_state", None)
    if not isinstance(state, EnvironmentState):
        return
    try:
        rows = state.close()
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit("environment_state_closed", producer="kernel_environment_runtime",
                 process_count=len(rows),
                 active_count=sum(
                     row.state == "running" and row.ownership == "owned"
                     for row in rows
                 ),
                 dispatch=False)
    except Exception as error:  # cleanup must not conceal the primary terminal result
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit("environment_cleanup_error", producer="kernel_environment_runtime",
                 error_type=type(error).__name__, dispatch=False)


__all__ = [
    "acknowledge_panel", "attach", "attach_frame", "cleanup", "panel_content",
    "readiness_snapshot", "snapshot",
]
