"""Shared readers for live and sealed StateFrame measurement journals in test runs."""
from __future__ import annotations

import json
from pathlib import Path

from stateframe.engine.kernel_journal_store import journal_paths, open_journal


def measurement_journal_paths(run_dir: str | Path) -> list[Path]:
    """Return measurement-journal paths for a run, whether live or sealed."""
    return journal_paths(Path(run_dir) / "worker" / "measurement-runs")


def measurement_events(run_dir: str | Path) -> list[dict[str, object]]:
    """Decode every nonblank event from each live or sealed measurement journal in a run."""
    events: list[dict[str, object]] = []
    for path in measurement_journal_paths(run_dir):
        with open_journal(path) as source:
            for line in source:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"measurement event in {path} is not a JSON object")
                events.append(event)
    return events
