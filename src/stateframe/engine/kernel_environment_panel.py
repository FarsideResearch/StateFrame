"""One compact, cache-stable environment panel."""
from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping, Sequence

from kernel_environment_state import EnvironmentSnapshot, ProcessRecord, safe_label


PANEL_HEADER = "[ENVIRONMENT STATE]"
PANEL_FOOTER = "[/ENVIRONMENT STATE]"
PANEL_SCHEMA = "stateframe-environment-panel/1"
MAX_PANEL_CHARS = 2200
MAX_DIAGNOSTIC_LINES = 2


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _process_line(row: ProcessRecord) -> str:
    identity = f"{row.launch_id}/g{row.entity_generation}"
    line = f"process {identity} {row.command_label} state={row.state}"
    if row.pid is not None:
        line += f" pid={row.pid}"
    if row.exit_code is not None:
        line += f" exit={row.exit_code}"
    if row.exit_signal is not None:
        line += f" signal={row.exit_signal}"
    if row.output_tail and row.state in {"crashed", "failed", "killed", "exited"}:
        line += " log_tail=" + _json(row.output_tail)
    if row.output_truncated:
        line += " log_tail_truncated=true"
    return line


def render(snapshot: EnvironmentSnapshot) -> str:
    """Render fixed-order public text from one immutable snapshot."""
    lines = [
        PANEL_HEADER,
        f"schema={PANEL_SCHEMA} snapshot={snapshot.snapshot_seq} "
        f"scope={safe_label(snapshot.scope.workspace_id, fallback='workspace')} "
        f"assembled_at={snapshot.assembled_at:.6f}",
    ]
    for process in snapshot.processes:
        lines.append(_process_line(process))
    for fact in snapshot.resource_facts:
        version = fact.get("version") if isinstance(fact, Mapping) else None
        digest = version.get("digest") if isinstance(version, Mapping) else None
        lines.append(
            f"resource {safe_label(fact.get('facet', 'unknown'), fallback='resource')} "
            f"availability={fact.get('availability', 'unknown')} "
            f"freshness={fact.get('freshness', 'unknown')} "
            f"coverage={fact.get('coverage', 'unknown')} "
            f"version={safe_label(digest, fallback='unknown')}"
        )
    for observation in snapshot.observations:
        value = observation.value if isinstance(observation.value, Mapping) else observation.value
        lines.append(
            f"fact {observation.entity_id}/g{observation.entity_generation} "
            f"availability={observation.availability} freshness={observation.freshness} "
            f"coverage={observation.coverage} value={_json(value)}"
        )
    if snapshot.receipts:
        lines.append(f"receipts indexed={len(snapshot.receipts)} owner=existing_receipt_owner")
    if snapshot.required_checks:
        for check in snapshot.required_checks:
            lines.append(
                f"readiness {check.get('check_id', 'unknown')} "
                f"status={check.get('status', 'unknown')} required={str(bool(check.get('required'))).lower()}"
            )
    if snapshot.omissions:
        lines.append("omissions=" + _json(list(snapshot.omissions)))
    lines.append("coverage=" + ",".join(snapshot.coverage))
    lines.append(PANEL_FOOTER)
    text = "\n".join(lines)
    if len(text) <= MAX_PANEL_CHARS:
        return text
    # Preserve all lifecycle lines first.  Diagnostics are optional and never
    # allowed to truncate the identity/state of an owned process.
    required = [PANEL_HEADER, lines[1]]
    required.extend(_process_line(process) for process in snapshot.processes)
    required.append(PANEL_FOOTER)
    compact = "\n".join(required)
    if len(compact) <= MAX_PANEL_CHARS:
        return compact
    # The process count is bounded by the supervisor; this final fallback is
    # explicit about omitted records rather than silently cutting a record.
    return "\n".join((PANEL_HEADER, lines[1], "processes omitted=panel_budget", PANEL_FOOTER))


@dataclass(frozen=True)
class PanelDelivery:
    digest: str
    content: str


class EnvironmentPanel:
    """Delivery-aware dedupe; preparing a panel does not acknowledge it."""

    def __init__(self) -> None:
        self._last_delivered: str | None = None
        self._pending: dict[str, PanelDelivery] = {}

    @property
    def last_delivered(self) -> str | None:
        return self._last_delivered

    def prepare(self, snapshot: EnvironmentSnapshot) -> PanelDelivery | None:
        key = snapshot.semantic_digest
        if key == self._last_delivered:
            return None
        existing = self._pending.get(key)
        if existing is not None:
            return existing
        delivery = PanelDelivery(key, render(snapshot))
        self._pending[key] = delivery
        # Keep retries bounded while retaining the current candidate.
        if len(self._pending) > 4:
            for old_key in tuple(self._pending)[:-4]:
                self._pending.pop(old_key, None)
        return delivery

    def acknowledge(self, delivery: PanelDelivery | None) -> None:
        if delivery is None:
            return
        self._last_delivered = delivery.digest
        self._pending.pop(delivery.digest, None)


def append_if_changed(
    messages: Sequence[Mapping[str, object]],
    snapshot: EnvironmentSnapshot,
    panel: EnvironmentPanel,
) -> tuple[list[dict[str, object]], PanelDelivery | None]:
    delivery = panel.prepare(snapshot)
    if delivery is None:
        return [dict(message) for message in messages], None
    return [*([dict(message) for message in messages]),
            {"role": "user", "content": delivery.content}], delivery


__all__ = [
    "EnvironmentPanel", "MAX_PANEL_CHARS", "PANEL_FOOTER", "PANEL_HEADER",
    "PANEL_SCHEMA", "PanelDelivery", "append_if_changed", "render",
]
