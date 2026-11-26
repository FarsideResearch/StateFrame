import hashlib
import json
import mmap
import os
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_effects as effects
from kernel_measurement import _serialize_final_request_body
from kernel_shell_tool import SHELL_SCHEMA, ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay, end_replay, tape_for_run
from stateframe import runner
from stateframe.engine import kernel_effects as host_effects
from stateframe.parallel_host import ParallelSupervisor
from stateframe.parallel_seals import artifact_operation
from test_parallel_host import _checked_receipt_scenario, _record_submission, _request
from test_shell_manifest_reuse import ProcessBoundary


SWITCH = "STATEFRAME_SCAN_STAT_MEMO"
KEY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")


def scan(root):
    return effects.scan(root, exclude=lambda _: False)


@pytest.fixture(autouse=True)
def fresh_memo(monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    for scanner in (effects, host_effects):
        monkeypatch.setattr(scanner, "_scan_memo", OrderedDict())


@pytest.fixture
def stable_clock(monkeypatch):
    clock = SimpleNamespace(time_ns=lambda: time.time_ns() + 10_000_000_000)
    for scanner in (effects, host_effects):
        monkeypatch.setattr(scanner, "time", clock)


@pytest.fixture
def scan_io(monkeypatch):
    calls = Counter()

    def count(function, name):
        def counted(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return counted

    io = SimpleNamespace(**vars(os))
    for name in ("open", "fstat", "read"):
        setattr(io, name, count(getattr(os, name), name))
    hashing = SimpleNamespace(sha256=count(hashlib.sha256, "sha256"))
    for scanner in (effects, host_effects):
        monkeypatch.setattr(scanner, "os", io)
        monkeypatch.setattr(scanner, "hashlib", hashing)
        if not hasattr(scanner.fcntl, "F_SETLEASE") or not hasattr(scanner.fcntl, "F_SETSIG"):
            # Exercise memo bookkeeping on platforms where production scans hash directly.
            monkeypatch.setattr(scanner, "fcntl", SimpleNamespace(
                F_SETSIG=1, F_SETLEASE=2, F_GETLEASE=3, F_RDLCK=0, F_UNLCK=4,
                fcntl=lambda *_: 0,
            ))
    return io, calls


def file_stat(path):
    info = path.stat()
    return SimpleNamespace(**{name: getattr(info, name) for name in (*KEY_FIELDS, "st_nlink")})


def substitute_stat(io, path, listed, descriptor):
    @contextmanager
    def scandir(directory):
        assert directory == path.parent

        def stat(*, follow_symlinks):
            assert follow_symlinks is False
            return listed()

        yield (SimpleNamespace(name=path.name, path=str(path), stat=stat),)

    counted_fstat = io.fstat

    def fstat(fd):
        counted_fstat(fd)
        return descriptor()

    io.scandir = scandir
    io.fstat = fstat


@pytest.mark.parametrize("value", ["", "1", "true", "ON", " on", "off ", "auto"])
def test_switch_requires_exact_on_or_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv(SWITCH, value)
    with pytest.raises(ValueError, match=f"{SWITCH} must be on or off"):
        scan(tmp_path)
    with pytest.raises(runner.RunError, match=f"{SWITCH} must be on or off"):
        runner.resolve_opt_in_switches()


@pytest.mark.parametrize("value", ["on", "off"])
def test_switch_reaches_the_engine(monkeypatch, value):
    monkeypatch.setenv(SWITCH, value)
    assert runner.resolve_opt_in_switches()[SWITCH] == value


def test_switch_defaults_off(tmp_path, monkeypatch, scan_io):
    monkeypatch.delenv(SWITCH, raising=False)
    (tmp_path / "file").write_bytes(b"contents")
    _, calls = scan_io

    assert scan(tmp_path)["file"].digest == hashlib.sha256(b"contents").hexdigest()
    assert calls["read"] == 2
    assert not effects._scan_memo
    assert SWITCH not in runner.resolve_opt_in_switches()


def test_stable_repeat_opens_and_checks_without_reading_or_hashing(
        tmp_path, monkeypatch, stable_clock, scan_io):
    (tmp_path / "file").write_bytes(b"contents")
    io, calls = scan_io
    counted_open = io.open

    def no_follow_open(path, flags):
        assert flags & os.O_NOFOLLOW
        return counted_open(path, flags)

    io.open = no_follow_open
    expected = scan(tmp_path)
    assert calls["sha256"] == 1 and calls["read"] == 2

    calls.clear()
    assert scan(tmp_path) == expected
    assert calls == {"open": 1, "fstat": 2}

    monkeypatch.setenv(SWITCH, "off")
    calls.clear()
    assert scan(tmp_path) == expected
    assert calls["sha256"] == 1 and calls["read"] == 2


def test_mixed_manifest_parity_and_symlink_target_hashing(
        tmp_path, monkeypatch, stable_clock, scan_io):
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "file").write_bytes(b"payload")
    (tmp_path / "link").symlink_to("missing")
    (tmp_path / "original").touch()
    (tmp_path / "hardlink").hardlink_to(tmp_path / "original")
    (tmp_path / "excluded").write_bytes(b"ignore")

    def manifest():
        return effects.scan(tmp_path, exclude=lambda name: name == "excluded")

    _, calls = scan_io
    monkeypatch.setenv(SWITCH, "on")
    expected = manifest()
    calls.clear()
    assert manifest() == expected
    assert calls["sha256"] == 1 and calls["read"] == 0
    assert expected["link"].digest == hashlib.sha256(b"missing").hexdigest()
    assert expected["hardlink"].kind == "other"
    assert "excluded" not in expected
    monkeypatch.setenv(SWITCH, "off")
    assert manifest() == expected
    assert effects.manifest_digest(manifest()) == effects.manifest_digest(expected)


@pytest.mark.parametrize("field", KEY_FIELDS)
def test_each_stat_field_invalidates_the_digest(tmp_path, stable_clock, scan_io, field):
    path = tmp_path / "file"
    path.write_bytes(b"old")
    current = file_stat(path)
    io, calls = scan_io
    substitute_stat(io, path, lambda: current, lambda: current)
    assert scan(tmp_path)["file"].digest == hashlib.sha256(b"old").hexdigest()

    value = getattr(current, field)
    setattr(current, field, value ^ 0o100 if field == "st_mode" else value + 1)
    payload = b"new!" if field == "st_size" else b"new"
    path.write_bytes(payload)
    calls.clear()
    assert scan(tmp_path)["file"].digest == hashlib.sha256(payload).hexdigest()
    assert calls["read"] == 2


@pytest.mark.parametrize("field", ["st_mtime_ns", "st_ctime_ns"])
def test_racy_timestamps_are_neither_memoized_nor_reused(tmp_path, monkeypatch, scan_io, field):
    path = tmp_path / "file"
    path.write_bytes(b"payload")
    age = 2_000_000_000
    now = 10_000_000_000
    current = file_stat(path)
    current.st_mtime_ns = current.st_ctime_ns = now - 2 * age
    setattr(current, field, now - age)
    monkeypatch.setattr(effects, "time", SimpleNamespace(time_ns=lambda: now))
    io, calls = scan_io
    substitute_stat(io, path, lambda: current, lambda: current)

    expected = scan(tmp_path)
    assert scan(tmp_path) == expected
    assert calls["read"] == 4
    now += 1
    assert scan(tmp_path) == expected  # First eligible scan still needs a digest.
    assert scan(tmp_path) == expected
    assert calls["read"] == 6
    now -= 1
    assert scan(tmp_path) == expected
    assert calls["read"] == 8


def test_scan_detects_shared_mmap_write_with_unchanged_stat(tmp_path, stable_clock):
    path = tmp_path / "file"
    path.write_bytes(b"old")
    with path.open("r+b") as source:
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_WRITE)
    with mapping:
        mapping[:] = b"one"  # Fault in the writable page before the first scan.
        before = file_stat(path)
        assert scan(tmp_path)["file"].digest == hashlib.sha256(b"one").hexdigest()
        mapping[:] = b"two"
        if sys.platform == "linux":
            assert file_stat(path) == before
        assert scan(tmp_path)["file"].digest == hashlib.sha256(b"two").hexdigest()


def test_scan_hashes_each_time_without_read_lease_support(
        tmp_path, monkeypatch, stable_clock, scan_io):
    monkeypatch.setattr(effects, "fcntl", SimpleNamespace())
    (tmp_path / "file").write_bytes(b"contents")
    _, calls = scan_io
    assert scan(tmp_path) == scan(tmp_path)
    assert calls["sha256"] == 2 and calls["read"] == 4
    assert not effects._scan_memo


def test_profile_disables_memo_for_host_receipt_scan(
        tmp_path, monkeypatch, stable_clock, scan_io):
    monkeypatch.delenv(SWITCH, raising=False)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema": "stateframe-product-profile/1", "switches": {SWITCH: "off"},
    }))
    environment = runner._provider_environment(
        SimpleNamespace(profile=str(profile), parallel_mode="1", extra_env={}))
    worker, _, _, _, receipt, _ = _checked_receipt_scenario(tmp_path)
    _record_submission(worker.run_dir, {"pkg/a.py": "VALUE = 2\n"})
    terminal_path = worker.run_dir / "engine-result.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["result"]["check_receipt"] = receipt.to_json()
    terminal_path.write_text(json.dumps(terminal))

    run_dir = tmp_path / "run"
    (run_dir / "spec.json").write_text(json.dumps({
        "parallel_mode": "1", "environment": environment,
    }))
    supervisor = ParallelSupervisor(run_dir)
    supervisor._persist_worker(worker)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    _, calls = scan_io
    for token in ("first", "second"):
        stage = worker.call_root / "host" / "seals" / f"stage-{token}.json"
        calls.clear()
        result = artifact_operation("seal_build", {
            "request": request.to_json(), "call_root": str(worker.call_root),
            "token": token, "stage": str(stage),
        })
        staged = json.loads(Path(result["staged"]["stage_path"]).read_text())
        assert staged["check_receipt"] == receipt.to_json()
        assert calls["read"] == 4  # Both regular files are read through EOF.


def test_descriptor_drift_blocks_cached_digest(tmp_path, stable_clock, scan_io):
    path = tmp_path / "file"
    path.write_bytes(b"old")
    listed = file_stat(path)
    opened = final = listed
    io, calls = scan_io
    substitute_stat(io, path, lambda: listed,
                    lambda: opened if calls["fstat"] == 1 else final)
    scan(tmp_path)

    final = SimpleNamespace(**vars(listed))
    final.st_ctime_ns += 1
    calls.clear()
    with pytest.raises(effects.EffectError, match="workspace changed while scanning file"):
        scan(tmp_path)
    assert calls == {"open": 1, "fstat": 2}

    opened = final
    path.write_bytes(b"new")
    calls.clear()
    assert scan(tmp_path)["file"].digest == hashlib.sha256(b"new").hexdigest()
    assert calls["read"] == 2


def test_same_file_stat_in_another_root_is_a_miss(tmp_path, stable_clock, scan_io):
    real_parent = tmp_path / "real"
    root = real_parent / "workspace"
    root.mkdir(parents=True)
    (root / "file").write_bytes(b"payload")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    other_root = alias / "workspace"
    _, calls = scan_io

    assert scan(root) == scan(other_root)
    assert calls["read"] == 4
    assert scan(root) == scan(other_root)
    assert calls["read"] == 4


def test_memo_limit_evicts_old_roots_and_remains_bounded_under_concurrency(
        tmp_path, monkeypatch, stable_clock, scan_io):
    monkeypatch.setattr(effects, "_SCAN_MEMO_LIMIT", 2)
    roots = [tmp_path / str(index) for index in range(3)]
    for root in roots:
        root.mkdir()
        (root / "file").write_bytes(root.name.encode())
    _, calls = scan_io
    expected = [scan(root) for root in roots]
    assert len(effects._scan_memo) == 2
    assert scan(roots[1]) == expected[1]
    assert calls["read"] == 6
    assert scan(roots[0]) == expected[0]
    assert calls["read"] == 8
    with ThreadPoolExecutor(max_workers=4) as threads:
        assert list(threads.map(scan, roots * 8)) == expected * 8
    assert len(effects._scan_memo) == 2


def test_default_preserves_shell_effect_tape_and_replayed_request_bytes(
        tmp_path, monkeypatch, stable_clock):
    monkeypatch.delenv(SWITCH, raising=False)

    def run_shell(base):
        root = base / "workspace"
        root.mkdir(parents=True)
        run = base / "run"
        run.mkdir()
        source = root / "app.py"
        initial = "VALUE = 1\n"
        command = "python edit.py"
        row = {"files": {"app.py": initial}, "scoped_edits": True,
               "allowed_product_paths": ["app.py"], "new_file_roots": []}
        process = ProcessBoundary(actions={command: lambda _: source.write_text("VALUE = 2\n")})

        def exchange(*, replay):
            source.write_text(initial)
            configure_replay(run, active=replay)
            tool = ShellTool(run, ToolSettings(run_env=False), {}, process)
            tool.bind(root, row, set())
            messages = [{"role": "user", "content": "Update app.py and confirm the change."}]
            for call_id, shell in (("call-1", command), ("call-2", "echo confirmed")):
                arguments = {"command": shell}
                reply = tool.execute(arguments, call_id)
                messages.extend((
                    {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps(arguments)}}]},
                    {"role": "tool", "tool_call_id": call_id, "content": reply.observation},
                ))
            return _serialize_final_request_body(
                {"model": "fixture", "messages": messages, "tools": [SHELL_SCHEMA]}, None, "request-1",
            )

        request = exchange(replay=False)
        tape = tape_for_run()
        journal = tape.path.read_bytes()
        edit = tape.lookup("call-1")
        assert edit.pre_manifest_digest != edit.post_manifest_digest
        assert b"modified: app.py" in request
        try:
            assert exchange(replay=True) == request
        finally:
            end_replay()
        assert tape.path.read_bytes() == journal
        assert process.calls == [command, "echo confirmed"] * 2
        return request, journal

    default = run_shell(tmp_path / "default")
    monkeypatch.setenv(SWITCH, "off")
    assert run_shell(tmp_path / "off") == default
