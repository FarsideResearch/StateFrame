"""_initial_frame_tokens reads subframe journals through the journal shim (live or sealed) and refuses symlinks."""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from _journal import write_journal_bytes
from stateframe.engine.kernel_parallel import ParallelError, _initial_frame_tokens

EVENT = {"kind": "unit_frame_tokens", "tokens": 9123}


def _runs(run_dir):
    root = run_dir / "worker" / "measurement-runs" / "m1"
    root.mkdir(parents=True)
    return root


def test_live_journal_is_read(tmp_path):
    root = _runs(tmp_path)
    (root / "events.jsonl").write_text(json.dumps(EVENT) + "\n", encoding="utf-8")
    assert _initial_frame_tokens(tmp_path) == 9123


def test_sealed_journal_is_read(tmp_path):
    # A sealed run keeps only events.jsonl.xz; a raw glob for events.jsonl used to miss it.
    root = _runs(tmp_path)
    write_journal_bytes(root / "events.jsonl.xz", lzma.compress((json.dumps(EVENT) + "\n").encode()))
    assert _initial_frame_tokens(tmp_path) == 9123


def test_symlinked_run_directory_is_refused(tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "events.jsonl").write_text(json.dumps(EVENT) + "\n", encoding="utf-8")
    runs = tmp_path / "worker" / "measurement-runs"
    runs.mkdir(parents=True)
    (runs / "m1").symlink_to(real, target_is_directory=True)
    with pytest.raises(ParallelError, match="symlink"):
        _initial_frame_tokens(tmp_path)


def test_no_journal_means_unknown(tmp_path):
    assert _initial_frame_tokens(tmp_path) is None


def test_parent_journals_include_live_and_sealed_attempts_in_order(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "fixtures"))
    from fixtures.parallel.e2e_support import parent_event_journals, parent_measurement_events

    paths = []
    events = []
    for index, run in enumerate((tmp_path / "attempts/2", tmp_path / "attempts/10", tmp_path)):
        root = _runs(run)
        event = {"kind": "attempt", "index": index}
        data = (json.dumps(event) + "\n\n").encode()
        path = root / ("events.jsonl" if index == 0 else "events.jsonl.xz")
        write_journal_bytes(path, data if index == 0 else lzma.compress(data))
        paths.append(path)
        events.append(event)

    assert parent_event_journals(tmp_path) == sorted(paths)
    assert parent_measurement_events(tmp_path) == events
