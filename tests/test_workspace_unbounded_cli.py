"""CLI regressions for unbounded repository starts and live workspace scans."""
from __future__ import annotations

import errno
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import cli, profiles, runner, workspace_start_snapshot as snapshot
import test_checkpoint_b_resume_fidelity as checkpoint
from test_workspace_state_oracle import scripted_provider

MIB = 1 << 20


def _repo(root: Path, shape: str) -> None:
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples/discount_bug", root)
    initial_files = [path for path in root.rglob("*") if path.is_file()]
    initial_entries = [path for path in root.rglob("*")]
    initial_bytes = sum(path.stat().st_size for path in initial_files)
    if shape == "files":
        for index in range(5_001 - len(initial_files)):
            (root / f"empty-{index:04d}.bin").touch()
    elif shape == "entries":
        nested = root / f"entries-{'x' * 96}"
        nested.mkdir()
        for index in range(20_001 - len(initial_entries) - 1):
            (nested / f"{index:06d}-{'x' * 248}").touch()
    elif shape == "bytes":
        (root / "bulk").mkdir()
        for index in range(64):
            (root / "bulk" / f"full-{index:02d}.bin").write_bytes(b"x" * MIB)
        tail = 65 * MIB - initial_bytes - 64 * MIB
        (root / "bulk" / "tail.bin").write_bytes(b"x" * tail)
    else:
        raise AssertionError(shape)


def _large_tree_count(root: Path, shape: str) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    entries = sum(1 for _ in root.rglob("*"))
    total_bytes = sum(path.stat().st_size for path in files)
    if shape == "files":
        assert len(files) == 5_001
    elif shape == "entries":
        assert entries == 20_001
    else:
        assert total_bytes == 65 * MIB
        assert max(path.stat().st_size for path in files) <= MIB
    return len(files), entries


@pytest.fixture
def cli_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    # Only the runner's identities: the copy code needs real random names.
    monkeypatch.setattr(runner, "secrets", SimpleNamespace(
        token_hex=lambda size=16: ("0123456789abcdef" * size)[:2 * size]))


@pytest.mark.parametrize("shape", ["files", "entries", "bytes"])
def test_cli_large_non_git_start_resumes(tmp_path, monkeypatch, shape):
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    _repo(repo, shape)
    assert not (repo / ".git").exists()
    input_files, input_entries = _large_tree_count(repo, shape)
    assert input_files > 0 and input_entries >= input_files
    launcher = checkpoint._write_launcher(tmp_path / "launch.py")
    profile = checkpoint._write_profile(tmp_path / "profile.json")
    environment = checkpoint._environment

    def startup_snapshot_environment(home, **kwargs):
        values = environment(home, **kwargs)
        values["STATEFRAME_SEND1_WAVE4"] = "on"
        return values

    monkeypatch.setattr(checkpoint, "_environment", startup_snapshot_environment)
    outcome = checkpoint._run_interrupted(
        "discount_bug", repo, run, launcher, profile, tmp_path / "home",
    )
    assert outcome.pending_request == outcome.resumed_request
    assert (run / "workspace" / "shop" / "pricing.py").is_file()
    if shape == "entries":
        assert (run / "parent-start.json").stat().st_size > 8 * MIB


def test_cli_disk_full_injected_before_provider_start(tmp_path, cli_environment, monkeypatch, capsys):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "answer.py").write_text("VALUE = 1\n")
    run = tmp_path / "run"

    def fail_after_partial(_self, _source, target, _info, **_kwargs):
        with target.open("xb") as stream:
            stream.write(b"V")
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(snapshot._Copies, "file", fail_after_partial)
    with scripted_provider([]) as (endpoint, trace):
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Read answer.py.",
            "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run), "--quiet",
        ])
        trace.script.assert_consumed()
    assert status == 2
    message = capsys.readouterr().err
    assert "repository copy needs 20 bytes;" in message
    assert "bytes are free (No space left on device)" in message
    assert not (run / "workspace").exists()
    assert not (run / "parent").exists()


def test_cli_shell_creates_and_scans_60000_files_mid_run(tmp_path, cli_environment):
    profile_switches = profiles.load(profiles.DEFAULT)
    disabled = {"STATEFRAME_SHIP_TREE_SMOKE", "STATEFRAME_GREEN_LOCK",
                "STATEFRAME_GREEN_LOCK_REPAIR", "STATEFRAME_BEHAVIOR_WITNESS"}
    for key in profile_switches:
        if key in disabled or key.startswith(("STATEFRAME_POST_GREEN", "STATEFRAME_POSTGREEN")):
            profile_switches[key] = "off"
    profile = tmp_path / "scale-profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1",
                                   "switches": profile_switches}), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "create_many.py").write_text(
        "from pathlib import Path\n"
        "def main():\n"
        "    output = Path('generated')\n    output.mkdir()\n"
        "    for index in range(60000): (output / f'{index:05d}.txt').touch()\n"
        "    print('created=60000')\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    steps = [
        {"tool": "shell", "required": True, "arguments": {"command": "python3 create_many.py"}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Created the generated files."}},
    ]
    run = tmp_path / "run"
    with scripted_provider(steps) as (endpoint, trace):
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Create and verify 60000 files under generated/.",
            "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
            "--profile", str(profile),
            "--no-run-env", "--shell-network", "none", "--no-git-tool",
            "--max-turns", "6", "--timeout", "2400", "--quiet",
        ])
        trace.script.assert_consumed()
    summary = json.loads((run / "summary.json").read_text())
    generated = run / "workspace" / "generated"
    assert generated.is_dir() and sum(1 for _ in generated.iterdir()) == 60_000
    assert any(b"created=60000" in body for _, body in trace.requests[1:])
    assert status == 0 and summary.get("engine_error_reason") is None, summary


def test_large_shell_receipt_entities_keep_order_and_deduplicate():
    from kernel_timeline import TimelineIndex, _extract_entities

    paths = [f"generated/{index:05d}.txt" for index in range(60_000)]
    receipt = {
        "changed_paths": paths,
        "effects": [{"path": path, "change": "created"} for path in paths],
    }
    entities = _extract_entities(receipt)
    assert len(entities) == 60_000
    assert entities[0] == "generated/00000.txt"
    assert entities[-1] == "generated/59999.txt"
    event = TimelineIndex("large-receipt").append(
        "receipt", receipt, actor="host", turn=1, content="created",
        entities=receipt, status="APPLIED")
    event_entities = event["entities"]
    assert tuple(event_entities[:60_000]) == entities
    assert event_entities[-1] == "created"
