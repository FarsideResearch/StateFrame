"""Read-only green-test evidence injection for deterministic host replays.

The replay lane cannot execute the original repository's test command on this
host.  This adapter consumes only recorded ``test_subprocess_completed`` rows,
keeps their order, and returns the recorded successful result.  The live
runtime still supplies the current binding, so archived machine paths or
revisions never become authority in the replay.
"""
from __future__ import annotations

import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path


class ReplayGreenError(ValueError):
    """The replay evidence is malformed or cannot be read."""


def _records_from_text(
    text: str, *, journal_path: str | Path | None = None
) -> tuple[dict[str, object], ...]:
    resolver = None
    if journal_path is not None:
        from kernel_measurement import resolve_journal_blob_refs

        resolver = resolve_journal_blob_refs
    rows: list[dict[str, object]] = []
    blob_cache: dict[str, object] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError, UnicodeError) as error:
            raise ReplayGreenError(f"invalid replay JSON at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise ReplayGreenError(f"replay row at line {line_number} is not an object")
        if resolver is not None:
            try:
                value = resolver(value, journal_path, blob_cache)
            except (OSError, ValueError) as error:
                raise ReplayGreenError(
                    f"invalid replay blob reference at line {line_number}") from error
            if not isinstance(value, Mapping):
                raise ReplayGreenError(f"replay row at line {line_number} did not resolve to an object")
        rows.append(dict(value))
    return tuple(rows)


def _read_source(source: str) -> str:
    # ``archive.tar.gz::member`` lets the perf lane consume the archived 850
    # journal without unpacking or modifying the workspace.
    if "::" in source and source.endswith("events.jsonl"):
        archive_name, member_name = source.split("::", 1)
        try:
            with tarfile.open(archive_name, "r:gz") as archive:
                member = archive.getmember(member_name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ReplayGreenError("replay archive member is not readable")
                return extracted.read().decode("utf-8")
        except (OSError, KeyError, tarfile.TarError, UnicodeError) as error:
            raise ReplayGreenError("cannot read replay archive member") from error
    try:
        return Path(source).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReplayGreenError("cannot read replay journal") from error


def load_green_results(source: str | Path) -> tuple[dict[str, object], ...]:
    """Load successful subprocess rows, preserving their recorded order."""
    source_text = _read_source(str(source))
    journal_path = Path(source) if "::" not in str(source) else str(source)
    rows = _records_from_text(source_text, journal_path=journal_path)
    result: list[dict[str, object]] = []
    for row in rows:
        if row.get("kind") != "test_subprocess_completed" or row.get("exit_code") != 0:
            continue
        selectors = row.get("selectors", ())
        if not isinstance(selectors, Sequence) or isinstance(selectors, (str, bytes, bytearray)):
            selectors = ()
        result.append({
            "source_sequence": row.get("sequence"),
            "exit_code": 0,
            "selectors": tuple(str(item) for item in selectors if str(item).strip()),
            "binding_stability": row.get("binding_stability"),
        })
    if not result:
        raise ReplayGreenError("replay journal contains no successful test results")
    return tuple(result)


class ReplayGreenCursor:
    """Consume recorded green results exactly once per injected observation."""

    def __init__(self, source: str | Path):
        source_text = _read_source(str(source))
        journal_path = Path(source) if "::" not in str(source) else str(source)
        rows = _records_from_text(source_text, journal_path=journal_path)
        self.results = tuple(
            {
                "source_sequence": row.get("sequence"),
                "exit_code": 0,
                "selectors": tuple(str(item) for item in row.get("selectors", ())
                                    if str(item).strip())
                if isinstance(row.get("selectors", ()), Sequence)
                and not isinstance(row.get("selectors", ()), (str, bytes, bytearray)) else (),
                "binding_stability": row.get("binding_stability"),
            }
            for row in rows
            if row.get("kind") == "test_subprocess_completed" and row.get("exit_code") == 0
        )
        if not self.results:
            raise ReplayGreenError("replay journal contains no successful test results")
        self.postgreen_actions = tuple(
            str(row.get("action", "")).strip().lower()
            for row in rows
            if row.get("kind") == "postgreen_typed_refusal"
            and str(row.get("action", "")).strip()
        )
        self.index = 0
        self.postgreen_index = 0

    def next(self, selectors: Sequence[object] = ()) -> dict[str, object] | None:
        if self.index >= len(self.results):
            return None
        row = dict(self.results[self.index])
        self.index += 1
        row["requested_selectors"] = tuple(str(item) for item in selectors if str(item).strip())
        return row

    def next_postgreen_action(self) -> str | None:
        """Return one recorded duplicate-verification action for explicit replay profiling."""
        if self.postgreen_index >= len(self.postgreen_actions):
            return None
        action = self.postgreen_actions[self.postgreen_index]
        self.postgreen_index += 1
        return action


__all__ = ["ReplayGreenCursor", "ReplayGreenError", "load_green_results"]
