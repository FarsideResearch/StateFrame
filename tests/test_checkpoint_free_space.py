"""Checkpoint disk admission at the step boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import kernel_checkpoint as checkpoint
from kernel_release_switch_registry import release_stateframe_switches
from stateframe import runner


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "checkpoints" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _setup(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "on")
    monkeypatch.delenv("STATEFRAME_CHECKPOINT_MIN_FREE_BYTES", raising=False)
    monkeypatch.setattr(checkpoint, "_STATE", {
        "saved": 0, "failed": 0, "restored_step": None, "disabled": None,
        "last_sends": None, "child": None, "child_tmp": None, "consecutive_failed": 0,
    })
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_rss_mb", lambda: 100.0)
    sends = [0]
    saves = []
    monkeypatch.setattr(checkpoint, "_sends", lambda _run_dir: sends[0])

    def save(_frame, step, _run_dir, *, background):
        saves.append((step, background))
        return run_dir / "checkpoints" / f"{step:04d}-saved"

    monkeypatch.setattr(checkpoint, "save", save)

    def step(number: int) -> None:
        sends[0] = number
        checkpoint.save_point(None, number, run_dir / "out")

    return run_dir, saves, step


def test_low_disk_skips_capture_once_per_episode_and_recovers(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)
    free = [512 << 20]
    checked = []

    def statvfs(path):
        checked.append(Path(path))
        return SimpleNamespace(f_bavail=free[0], f_frsize=1)

    monkeypatch.setattr(checkpoint.os, "statvfs", statvfs)
    step(1)
    step(2)
    step(3)
    assert saves == []
    skipped = [row for row in _events(run_dir) if row["event"] == "checkpoint_skipped_low_disk"]
    assert len(skipped) == 1
    assert skipped[0]["free_bytes"] == 512 << 20
    assert skipped[0]["floor_bytes"] == 1 << 30
    assert checked == [run_dir, run_dir / "checkpoints", run_dir / "checkpoints"]

    free[0] = 2 << 30
    step(4)
    assert saves == [(4, False)]
    free[0] = 512 << 20
    step(5)
    skipped = [row for row in _events(run_dir) if row["event"] == "checkpoint_skipped_low_disk"]
    assert len(skipped) == 2
    assert saves == [(4, False)]


def test_previous_stored_state_size_sets_floor(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)
    root = run_dir / "checkpoints"
    saved = root / "0001-prior"
    saved.mkdir(parents=True)
    (saved / "checkpoint.json").write_text(json.dumps({
        "state_codec": "zlib1", "state_bytes": 2 << 30, "state_stored_bytes": 500 << 20}))
    with (saved / "state.pkl").open("wb") as stream:
        stream.truncate(500 << 20)
    (root / "LATEST").write_text(saved.name + "\n")
    monkeypatch.setattr(checkpoint.os, "statvfs",
                        lambda path: SimpleNamespace(f_bavail=1200 << 20, f_frsize=1))
    step(2)
    assert saves == []
    skipped = [row for row in _events(run_dir) if row["event"] == "checkpoint_skipped_low_disk"]
    assert len(skipped) == 1
    assert skipped[0]["floor_bytes"] == 1500 << 20


def test_first_save_uses_memory_estimate(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(checkpoint, "_rss_mb", lambda: 600.0)
    monkeypatch.setattr(checkpoint.os, "statvfs",
                        lambda path: SimpleNamespace(f_bavail=1500 << 20, f_frsize=1))
    step(1)
    assert saves == []
    skipped = [row for row in _events(run_dir) if row["event"] == "checkpoint_skipped_low_disk"]
    assert skipped[0]["floor_bytes"] == 1800 << 20


def test_env_override_and_zero_disable(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_MIN_FREE_BYTES", "200")
    calls = []

    def statvfs(path):
        calls.append(path)
        return SimpleNamespace(f_bavail=199, f_frsize=1)

    monkeypatch.setattr(checkpoint.os, "statvfs", statvfs)
    step(1)
    assert saves == []
    assert [row["floor_bytes"] for row in _events(run_dir)
            if row["event"] == "checkpoint_skipped_low_disk"] == [200]
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_MIN_FREE_BYTES", "0")
    step(2)
    assert saves == [(2, False)]
    assert len(calls) == 1


def test_statvfs_error_proceeds_with_save(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)

    def statvfs(_path):
        raise OSError("free space unavailable")

    monkeypatch.setattr(checkpoint.os, "statvfs", statvfs)
    step(1)
    assert saves == [(1, False)]
    assert [row["event"] for row in _events(run_dir)] == ["checkpoint_saved"]


def test_statvfs_uses_available_blocks_times_fragment_size(monkeypatch, tmp_path):
    run_dir, saves, step = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_MIN_FREE_BYTES", "8192")
    monkeypatch.setattr(checkpoint.os, "statvfs",
                        lambda path: SimpleNamespace(f_bavail=1, f_frsize=4096))
    step(1)
    assert saves == []
    skipped = [row for row in _events(run_dir) if row["event"] == "checkpoint_skipped_low_disk"]
    assert skipped[0]["free_bytes"] == 4096


def test_override_is_forwarded_and_release_registered(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_MIN_FREE_BYTES", "12345")
    assert runner.checkpoint_env()["STATEFRAME_CHECKPOINT_MIN_FREE_BYTES"] == "12345"
    registered, _digest = release_stateframe_switches()
    assert "STATEFRAME_CHECKPOINT_MIN_FREE_BYTES" in registered
