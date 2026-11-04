"""Shared readers for live and sealed StateFrame measurement journals in test runs."""
from __future__ import annotations

import json
import lzma
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


def _journal_path(path_or_run_dir: str | Path) -> Path:
    path = Path(path_or_run_dir)
    if (path / "worker" / "measurement-runs").is_dir():
        paths = measurement_journal_paths(path)
        if len(paths) != 1:
            raise ValueError(f"expected one measurement journal in {path}, found {len(paths)}")
        return paths[0]
    plain = path / "events.jsonl" if path.is_dir() else path
    if plain.suffix == ".xz":
        plain = plain.with_suffix("")
    actual = plain if plain.is_file() else Path(f"{plain}.xz")
    if not actual.is_file():
        raise FileNotFoundError(f"measurement journal is absent: {plain} or {plain}.xz")
    return actual


def journal_bytes(path_or_run_dir: str | Path, *, decompress: bool = False) -> bytes:
    """Read exact on-disk journal bytes for byte-identity, torn-line, or truncation proofs.

    Sealed journals return compressed bytes by default; ``decompress=True`` returns
    their original JSONL bytes. Use ``measurement_events`` for decoded-event checks.
    """
    path = _journal_path(path_or_run_dir)
    data = path.read_bytes()
    return lzma.decompress(data) if decompress and path.suffix == ".xz" else data


def write_journal_bytes(path: str | Path, data: bytes, *, append: bool = False) -> None:
    """Plant exact fixture bytes for byte-identity, torn-line, or truncation proofs.

    Use decoded readers for assertions that concern events rather than bytes.
    """
    with Path(path).open("ab" if append else "wb") as stream:
        stream.write(data)
