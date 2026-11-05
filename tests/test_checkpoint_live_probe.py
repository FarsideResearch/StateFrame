"""Probe bookkeeping that must remain trustworthy during a live interruption."""

import json
from pathlib import Path
import sys
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import checkpoint_live_probe as probe  # noqa: E402
from checkpoint_live_probe import checkpoint_state, environment, pending_request, report  # noqa: E402


def test_checkpoint_requires_durable_state_and_save_event(tmp_path):
    root = tmp_path / "checkpoints"
    folder = root / "0015"
    folder.mkdir(parents=True)
    (root / "LATEST").write_text("0015")
    (folder / "state.pkl").write_bytes(b"state")
    (folder / "checkpoint.json").write_text(
        json.dumps({"step": 15, "sends": 4, "state_bytes": 5})
    )
    journal = root / "events.jsonl"
    journal.write_text(json.dumps({"event": "checkpoint_started", "step": 15}) + "\n")
    assert checkpoint_state(tmp_path) is None
    journal.write_text(json.dumps({"event": "checkpoint_saved", "step": 15}) + "\n")
    assert checkpoint_state(tmp_path)["state_bytes"] == 5
    assert checkpoint_state(tmp_path)["stored_state_bytes"] == 5
    (folder / "state.pkl").write_bytes(b"short")
    (folder / "checkpoint.json").write_text(
        json.dumps({"step": 15, "sends": 4, "state_bytes": 6})
    )
    with pytest.raises(RuntimeError, match="state size"):
        checkpoint_state(tmp_path)


def test_checkpoint_probe_distinguishes_raw_and_stored_sizes_for_zlib_state(tmp_path):
    root = tmp_path / "checkpoints"
    folder = root / "0015"
    folder.mkdir(parents=True)
    (root / "LATEST").write_text("0015")
    raw = b"checkpoint state" * 1024
    stored = zlib.compress(raw, level=1)
    (folder / "state.pkl").write_bytes(stored)
    (folder / "checkpoint.json").write_text(json.dumps({
        "step": 15, "sends": 4, "state_bytes": len(raw), "state_codec": "zlib1",
        "state_stored_bytes": len(stored),
    }))
    (root / "events.jsonl").write_text(json.dumps({"event": "checkpoint_saved", "step": 15}) + "\n")

    state = checkpoint_state(tmp_path)

    assert state["state_bytes"] == len(raw)
    assert state["stored_state_bytes"] == len(stored)


def test_pending_request_ignores_completed_and_incomplete_wire(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0007.request.json").write_text("{}")
    (wire / "0007.response.json").write_text("{}")
    (wire / "0008.request.json").write_text("{")
    assert pending_request(tmp_path, 7) is None
    (wire / "0008.request.json").write_text("{}")
    assert pending_request(tmp_path, 8) == 8
    assert pending_request(tmp_path, 9) is None


def test_environment_uses_default_checkpoint_settings(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_EVERY", "1")
    monkeypatch.setenv("STATEFRAME_KEEP_CHECKPOINTS", "1")
    assert "STATEFRAME_CHECKPOINT" not in environment()
    assert "STATEFRAME_CHECKPOINT_EVERY" not in environment()
    assert environment(interval=3)["STATEFRAME_CHECKPOINT_EVERY"] == "3"


def test_kill_run_stops_engine_process_group(monkeypatch):
    calls = []

    class Runner:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 30
            calls.append("wait")

    monkeypatch.setattr(probe, "engine_child", lambda parent: 202 if parent == 101 else None)
    monkeypatch.setattr(probe.os, "killpg", lambda group, sig: calls.append(("group", group, sig)))
    monkeypatch.setattr(probe.os, "kill", lambda pid, sig: calls.append(("runner", pid, sig)))
    probe.kill_run(Runner())
    assert calls == [("group", 202, probe.signal.SIGKILL),
                     ("runner", 101, probe.signal.SIGKILL), "wait"]


def test_report_writes_machine_and_readable_results(tmp_path):
    row = {
        "checkpoint_step": 15,
        "kill_request": 8,
        "written_state_bytes": 123,
        "restored": True,
        "verified": True,
        "raw_identical": True,
        "seconds_to_request_checkpoint": 1.25,
        "seconds_to_request_full_replay": 4.5,
        "speedup": 3.6,
        "final_outcome": "completed",
        "tests_passed": True,
        "total_requests": 12,
    }
    report(row, tmp_path)
    assert json.loads((tmp_path / "proof.json").read_text()) == row
    table = (tmp_path / "proof.md").read_text()
    assert "| Checkpoint step | 15 |" in table
    assert "| Full replay seconds | 4.5 |" in table
    assert "| Speedup | 3.6x |" in table


def test_full_replay_uses_original_run_path_and_preserves_checkpoint(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    wire = run / "wire"
    wire.mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "LATEST").write_text("0015")
    (run / "spec.json").write_text(
        json.dumps({"environment": {"STATEFRAME_CHECKPOINT": "save"}})
    )
    original = b'{"request":10}'
    (wire / "0010.request.json").write_bytes(original)
    (wire / "0010.response.json").write_bytes(b"{}")

    def launch(command, log, env):
        assert str(run) in command
        assert run.is_dir()
        assert not (run / "checkpoints").exists()
        assert (
            json.loads((run / "spec.json").read_text())["environment"][
                "STATEFRAME_CHECKPOINT"
            ]
            == "off"
        )
        return object()

    monkeypatch.setattr(probe, "launch", launch)
    monkeypatch.setattr(probe, "rebuilt_request", lambda *args: (original, 2.0))
    monkeypatch.setattr(probe, "kill_run", lambda runner: None)
    control, seconds, identical = probe.replay_control(
        tmp_path, run, 10, original, "http://127.0.0.1:18500/v1", "model"
    )
    assert (control / "wire").is_dir()
    assert (run / "checkpoints" / "LATEST").read_text() == "0015"
    assert (run / "wire" / "0010.request.json").read_bytes() == original
    assert (seconds, identical) == (2.0, True)
