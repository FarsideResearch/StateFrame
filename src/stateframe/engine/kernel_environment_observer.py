"""Read-only, explicitly scoped environment facts.

The observer has no ambient process or repository scan.  Process lifecycle is
owned by :class:`EnvironmentState`; receipts and background jobs are read from
their existing owners.  Endpoint checks are accepted only as explicit local
targets supplied by the host.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import shutil
import socket
from pathlib import Path

from kernel_environment_state import EnvironmentState, EnvironmentObservation


OBSERVER_ID = "kernel_environment_observer"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _endpoint_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


class EnvironmentObserver:
    """Collect only facts this organ is explicitly authorized to inspect."""

    def __init__(
        self,
        state: EnvironmentState,
        *,
        endpoints: Iterable[Mapping[str, object]] = (),
        endpoint_timeout_s: float = 0.05,
    ) -> None:
        if endpoint_timeout_s <= 0:
            raise ValueError("endpoint_timeout_s must be positive")
        self.state = state
        self.endpoints = tuple(dict(row) for row in endpoints)
        self.endpoint_timeout_s = float(endpoint_timeout_s)

    @classmethod
    def from_task_row(cls, state: EnvironmentState, task_row: object) -> "EnvironmentObserver":
        if not isinstance(task_row, Mapping):
            return cls(state)
        return cls(state, endpoints=_endpoint_rows(task_row.get("environment_endpoints", ())))

    def observe_disk(self) -> EnvironmentObservation[object]:
        """Record bounded filesystem capacity for the authorized workspace only."""
        try:
            usage = shutil.disk_usage(Path(self.state.root))
        except OSError as error:
            return self.state.record_observation(
                "disk:workspace",
                {"error_type": type(error).__name__},
                observer_id=OBSERVER_ID,
                availability="error",
                coverage="complete",
                valid_for=5.0,
                journal_value_change=False,
            )
        return self.state.record_observation(
            "disk:workspace",
            {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
            observer_id=OBSERVER_ID,
            valid_for=5.0,
            journal_value_change=False,
        )

    def observe_endpoint(self, row: Mapping[str, object]) -> EnvironmentObservation[object]:
        host = str(row.get("host") or "127.0.0.1")
        label = str(row.get("label") or f"{host}:{row.get('port', '')}")
        port = row.get("port")
        permitted_host = host in LOCAL_HOSTS
        valid_port = isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
        entity_id = "endpoint:" + (label[:80] if permitted_host else "denied")
        if not permitted_host or not valid_port:
            return self.state.record_observation(
                entity_id,
                {"host": host if permitted_host else "<denied>",
                 "port": port if valid_port else None, "listening": None},
                observer_id=OBSERVER_ID,
                availability="denied",
                coverage="complete",
                valid_for=5.0,
            )
        sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.endpoint_timeout_s)
        try:
            result = sock.connect_ex((host, port))
            listening = result == 0
            value = {"host": host, "port": port, "listening": listening}
            availability = "present" if listening else "absent"
        except OSError as error:
            value = {"host": host, "port": port, "listening": None,
                     "error_type": type(error).__name__}
            availability = "unknown"
        finally:
            sock.close()
        return self.state.record_observation(
            entity_id, value, observer_id=OBSERVER_ID, availability=availability,
            coverage="complete", valid_for=5.0,
        )

    def collect(self, *, receipts: Iterable[object] | None = None) -> None:
        self.observe_disk()
        for row in self.endpoints:
            self.observe_endpoint(row)
        if receipts is not None:
            self.state.ingest_receipts(receipts)


__all__ = ["EnvironmentObserver", "LOCAL_HOSTS", "OBSERVER_ID"]
