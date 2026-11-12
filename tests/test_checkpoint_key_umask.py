"""Checkpoint keys remain usable with permissive process umasks."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from _timeouts import scaled
from stateframe import _child, runner
from stateframe.state_dirs import private_directory


def _key_dir(home: Path) -> Path:
    return home / ".stateframe"


def test_child_script_imports_its_host_siblings() -> None:
    result = subprocess.run([sys.executable, str(Path(_child.__file__))],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage: _child.py <run-dir>" in result.stderr
    assert "ImportError" not in result.stderr


def test_fresh_home_under_umask_002_has_private_key(tmp_path: Path) -> None:
    script = """import json, os, pathlib, stat
from stateframe import _child
os.umask(0o002)
directory = pathlib.Path.home() / '.stateframe'
directory.mkdir()
key = _child.checkpoint_key()
print(json.dumps({'usable': key is not None, 'directory': stat.S_IMODE(directory.stat().st_mode),
                  'key': stat.S_IMODE((directory / 'checkpoint.key').stat().st_mode)
                         if (directory / 'checkpoint.key').exists() else None}))
"""
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "HOME": str(tmp_path)},
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"usable": True, "directory": 0o700, "key": 0o600}


def test_state_directories_ignore_umask(tmp_path: Path) -> None:
    script = """import json, os, pathlib, stat
from stateframe.state_dirs import private_directory
os.umask(int(os.environ['TEST_UMASK'], 8))
root = pathlib.Path.home() / '.stateframe'
leaf = private_directory(root, 'runs', 'example')
print(json.dumps([stat.S_IMODE(path.stat().st_mode) for path in (root, root / 'runs', leaf)]))
"""
    for mask in (0o002, 0o777):
        home = tmp_path / f"home-{mask:o}"
        result = subprocess.run([sys.executable, "-c", script],
                                env={**os.environ, "HOME": str(home), "TEST_UMASK": f"{mask:o}"},
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == [0o700, 0o700, 0o700]


def test_existing_owned_group_writable_directory_and_key_are_tightened(
        tmp_path: Path, monkeypatch) -> None:
    directory = _key_dir(tmp_path)
    directory.mkdir(mode=0o770)
    directory.chmod(0o770)
    key = directory / "checkpoint.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o660)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _child.checkpoint_key() == b"k" * 32
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_foreign_owned_directory_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    directory = _key_dir(tmp_path)
    directory.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_child.os, "getuid", lambda: os.stat(directory).st_uid + 1)

    assert _child.checkpoint_key() is None
    assert "checkpoint_key_unusable" in capsys.readouterr().out
    assert not (directory / "checkpoint.key").exists()


def test_symlinked_directory_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    _key_dir(tmp_path).symlink_to(destination, target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _child.checkpoint_key() is None
    assert "checkpoint_key_unusable" in capsys.readouterr().out
    assert not (destination / "checkpoint.key").exists()


def test_foreign_and_linked_keys_are_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    directory = private_directory(_key_dir(tmp_path))
    key = directory / "checkpoint.key"
    target = tmp_path / "target"
    target.write_bytes(b"k" * 32)
    key.symlink_to(target)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_child, "KEY_LINK_SETTLE_S", 0.1)
    assert _child.checkpoint_key() is None
    key.unlink()
    os.link(target, key)
    assert _child.checkpoint_key() is None
    assert capsys.readouterr().out.count("checkpoint_key_unusable") == 2


def _drive_events(tmp_path: Path, monkeypatch, events: list[dict]) -> None:
    child = tmp_path / "child.py"
    child.write_text("\n".join("print(" + repr(runner.LOG_PREFIX + json.dumps(event)) + ")"
                               for event in events) + "\n")
    monkeypatch.setattr(runner, "CHILD", child)
    options = runner.RunOptions(repo=tmp_path, task="check warnings", endpoint=None)
    assert runner.drive(tmp_path, options, printer=lambda *_: None) == 0


def test_host_warns_once_for_disabled_checkpoints(tmp_path: Path, monkeypatch, capsys) -> None:
    _drive_events(tmp_path, monkeypatch, [
        {"kind": "checkpoint_key_unusable", "reason": "key directory is foreign"},
        {"kind": "checkpoint_key_unusable", "reason": "another failure"},
    ])
    assert capsys.readouterr().err.splitlines() == [
        "warning: checkpoints are off for this run: key directory is foreign"]


def test_host_does_not_warn_when_checkpoints_work(tmp_path: Path, monkeypatch, capsys) -> None:
    _drive_events(tmp_path, monkeypatch, [{"kind": "engine_start", "turns": 1}])
    assert capsys.readouterr().err == ""


def test_host_warns_for_engine_save_disable(tmp_path: Path, capsys) -> None:
    events = tmp_path / "checkpoints" / "events.jsonl"
    events.parent.mkdir()
    events.write_text(json.dumps({"event": "checkpoint_disabled", "reason": "three failed saves"}) + "\n")
    runner._warn_checkpoint_journal(tmp_path, set())
    assert capsys.readouterr().err.strip() == (
        "warning: checkpoints are off for this run: three failed saves")


def test_key_seen_during_publish_link_window_is_used(tmp_path: Path, monkeypatch) -> None:
    """publish() links its temp file to the key name, then unlinks the temp. A reader that
    opens the key inside that window sees two links and must wait it out, not give up."""
    import threading

    directory = private_directory(_key_dir(tmp_path))
    key = directory / "checkpoint.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    temp = directory / ".checkpoint.key.publishing"
    os.link(key, temp)
    monkeypatch.setenv("HOME", str(tmp_path))
    remover = threading.Timer(0.3, temp.unlink)
    remover.start()
    try:
        assert _child.checkpoint_key() == b"k" * 32
    finally:
        remover.join()


def test_concurrent_first_use_gives_every_process_the_same_key(tmp_path: Path) -> None:
    """Many children creating the key on a fresh home at once all get one usable key."""
    script = """import sys
from stateframe import _child
key = _child.checkpoint_key()
sys.stdout.write(key.hex() if key else 'NONE')
"""
    processes = [subprocess.Popen([sys.executable, "-c", script],
                                  env={**os.environ, "HOME": str(tmp_path)},
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(16)]
    keys = [process.communicate(timeout=scaled(60))[0].strip().splitlines()[-1] for process in processes]
    assert "NONE" not in keys
    assert len(set(keys)) == 1
    assert os.stat(_key_dir(tmp_path) / "checkpoint.key").st_nlink == 1
