"""The first request can seal and send independently of cache publication."""
from __future__ import annotations

import errno
import io
import json
import os
import subprocess
import sys
from threading import Event, Thread, current_thread

import pytest

from _child_env import child_env

import kernel_request_diet as diet
from stateframe import _child, runner, send1_probe_cache as probes, startup_cache_writes
from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine.kernel_request_auth_identity import protected_home_identity
from stateframe.endpoint import Endpoint
from stateframe.scripted_endpoint import Script, serve


@pytest.fixture
def launch(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv(probes.SWITCH, "on")
    protected_home_identity(home, initialize=True)
    marker = tmp_path / "requests" / "0001.json"
    marker.parent.mkdir()
    return home, marker


def test_cold_writes_follow_the_durable_marker_and_the_next_process_hits(launch, monkeypatch):
    home, marker = launch
    observed = []
    opens = []
    failures = []
    fsync = os.fsync
    store = probes._store
    foreground = current_thread()

    def track_fsync(fd):
        if current_thread() is not foreground:
            observed.append(marker.read_bytes())
        fsync(fd)

    def track_store(*, create):
        opens.append(create)
        return store(create=create)

    monkeypatch.setattr(os, "fsync", track_fsync)
    monkeypatch.setattr(probes, "_store", track_store)
    token_directory = home / "token-counts-v1"
    with startup_cache_writes.writer_scope(lambda kind, **fields: failures.append((kind, fields))) as writer:
        for _ in range(5):
            assert not probes.reads_available()
            assert probes.lookup(probes.ENGINE_VERSION, "interpreter") is None
        assert opens == [False]
        assert probes.publish(probes.ENGINE_VERSION, "interpreter", {"version": [3, 13, 0]})
        startup_cache_writes.defer("exact-token-count", lambda: diet._publish_strong_count(
            token_directory, "identity", "prompt", "prompt", 7))
        assert not (home / probes._DIRECTORY).exists()
        assert not token_directory.exists()
        assert observed == []
        _child.atomic_bytes(marker, b'{"request": 1}')
        writer.seal()

    assert failures == []
    assert observed == [b'{"request": 1}'] * 4  # file and directory fsync for each store
    observed_next = subprocess.run([sys.executable, "-c", """
from pathlib import Path
import os
from stateframe import send1_probe_cache as probes
from kernel_request_diet import _read_exact_count, _strong_count_slot

assert probes.lookup(probes.ENGINE_VERSION, 'interpreter') == {'version': [3, 13, 0]}
directory = Path(os.environ['STATEFRAME_HOME']) / 'token-counts-v1'
assert _read_exact_count(*_strong_count_slot(directory, 'identity', 'prompt', 'prompt')) == 7
"""], capture_output=True, text=True, timeout=30, env=child_env())
    assert observed_next.returncode == 0, observed_next.stderr


def test_checkpoint_rebinds_writer_without_serializing_runtime_state():
    context = {"hostfns": {}, "participants": {}, "framecells": {}, "framefuncs": {}, "cvars": {}}
    with startup_cache_writes.writer_scope(lambda *_args, **_kwargs: None, child=True) as writer:
        writer.seal()
        assert writer._thread is not None
        raw = checkpoint.dumps({"writer": writer}, context)
        assert b"host-startup-cache-writer" in raw
        assert b"StartupCacheWriter" not in raw
        assert b"_thread" not in raw
        restored = checkpoint._Unpickler(io.BytesIO(raw), context).load()
        assert restored["writer"] is writer


@pytest.mark.parametrize("cache", ["probe", "token"])
def test_stalled_failed_fsync_leaves_send_and_existing_entry_available(launch, monkeypatch, cache):
    home, marker = launch
    token_directory = home / "token-counts-v1"
    if cache == "probe":
        store = home / probes._DIRECTORY
        kind = probes.ENGINE_VERSION

        def publish(patch):
            return probes.publish(kind, "interpreter", {"version": [3, 13, patch]})

        def read():
            return probes.lookup(kind, "interpreter")["version"][-1]
    else:
        store = token_directory
        kind = "exact-token-count"

        def publish(count):
            return startup_cache_writes.defer(kind, lambda: diet._publish_strong_count(
                token_directory, "identity", "prompt", "prompt", count))

        def read():
            return diet._read_exact_count(*diet._strong_count_slot(
                token_directory, "identity", "prompt", "prompt"))

    publish(1)
    entered_fsync, release_fsync, sent = Event(), Event(), Event()
    failures = []
    fsync = os.fsync
    foreground = current_thread()

    def fail_while_sending(fd):
        if current_thread() is foreground:
            return fsync(fd)
        entered_fsync.set()
        if not release_fsync.wait(10):
            raise TimeoutError("foreground did not continue")
        raise OSError(errno.EIO, "injected write failure")

    def send():
        assert entered_fsync.wait(10)
        assert json.loads(marker.read_bytes())["request"] == 1
        assert read() == 1
        sent.set()

    monkeypatch.setattr(os, "fsync", fail_while_sending)
    with startup_cache_writes.writer_scope(lambda event, **fields: failures.append((event, fields))) as writer:
        publish(2)
        _child.atomic_bytes(marker, b'{"request": 1}')
        sender = Thread(target=send)
        sender.start()
        try:
            writer.seal()
            assert sent.wait(10)
        finally:
            release_fsync.set()
            sender.join()
    assert read() == 1
    assert len(list(store.iterdir())) == 1
    assert failures == [("cache_write_failed", {"cache": kind, "error": "OSError"})]


@pytest.mark.host_sandbox
def test_rejected_first_request_does_not_publish_startup_cache(tmp_path, launch, monkeypatch):
    home, _ = launch
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def sample():\n    return 42\n")
    run_dir = tmp_path / "run"
    marker = run_dir / "requests" / "0001.json"
    injection = f"""
import os, sys, threading

def reject_live_validation(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "validate"
            and frame.f_globals.get("__name__") == "kernel_request_seal"
            and type(frame.f_locals.get("self")).__name__ == "QualifiedRequest"
            and os.path.isfile({str(marker)!r})):
        raise RuntimeError("seeded first request validation failure")

sys.setprofile(reject_live_validation)
threading.setprofile(reject_live_validation)
"""
    popen = subprocess.Popen

    def launch_child(command, *args, **kwargs):
        if isinstance(command, list) and command[-1] == "stateframe._child":
            command = [*command[:3], injection + command[3], command[4]]
        elif isinstance(command, list) and len(command) == 4 and command[2].endswith("/_child.py"):
            bootstrap = (
                "import runpy\n"
                "sys.argv.pop(0)\n"
                "sys.path[0] = os.path.dirname(sys.argv[0])\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            command = [*command[:2], "-c", injection + bootstrap, *command[2:]]
        return popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch_child)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    script = Script({"tokenize": {"mode": "ok"}})
    with serve(script) as server:
        try:
            runner.run(runner.RunOptions(
                repo=repo, task="Inspect sample.py.", max_turns=1, run_dir=run_dir,
                endpoint=Endpoint.from_args(f"http://127.0.0.1:{server.server_port}/v1", script.model, None),
                check_cmd="true", test_python=sys.executable,
                tool_settings=runner.ToolSettings(shell="off", run_env=False), quiet=True),
                printer=lambda _line: None)
        finally:
            server.shutdown()

    assert marker.is_file()
    assert "seeded first request validation failure" in (run_dir / "engine.log").read_text()
    assert not script.requests
    assert not (run_dir / "wire" / "0001.request.json").exists()
    assert not (home / probes._DIRECTORY).exists()


def test_runner_first_request_releases_host_writer_and_next_launch_hits(tmp_path, launch, monkeypatch):
    home, _marker = launch
    work = tmp_path / "repo"
    work.mkdir()
    (work / "sample.py").write_text("def sample():\n    return 42\n")
    script = Script({"tokenize": {"mode": "ok"}})
    sent = Event()
    next_call = script.next_call

    def observe_send(offered):
        sent.set()
        return next_call(offered)

    monkeypatch.setattr(script, "next_call", observe_send)
    cold = tmp_path / "cold"
    published = []
    checked = []
    publish = probes._publish
    startup_probe = runner._startup_probe

    def version_probe(python, code, *, label, on_clean_success=None):
        if not label.startswith("engine interpreter"):
            return startup_probe(python, code, label=label, on_clean_success=on_clean_success)
        if on_clean_success is not None:
            on_clean_success()
        return json.dumps(list(sys.version_info[:3])).encode()

    def check_live(*_args):
        checked.append(True)
        return True

    def publish_after_request(kind, *args, **kwargs):
        if kind == probes.ENGINE_VERSION:
            marker = cold / "requests" / "0001.json"
            assert marker.is_file()
            assert sent.wait(10)
            published.append(kind)
        return publish(kind, *args, **kwargs)

    monkeypatch.setattr(probes, "engine_fingerprint", lambda *_args: "interpreter")
    monkeypatch.setattr(probes, "check_sandbox", check_live)
    monkeypatch.setattr(probes, "_publish", publish_after_request)
    monkeypatch.setattr(runner, "_startup_probe", version_probe)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    with serve(script) as server:
        endpoint = Endpoint.from_args(f"http://127.0.0.1:{server.server_port}/v1", script.model, None)

        def run(path):
            runner.run(runner.RunOptions(
                repo=work, task="Inspect sample.py.", endpoint=endpoint, max_turns=1,
                run_dir=path, check_cmd="true", test_python=sys.executable,
                tool_settings=runner.ToolSettings(shell="off", run_env=False), quiet=True),
                printer=lambda _line: None)
            return ((path / "requests" / "0001.json").read_bytes(),
                    (path / "wire" / "0001.request.json").read_bytes())

        try:
            first = run(cold)
            assert published == [probes.ENGINE_VERSION]
            assert not checked
            marker_time = (cold / "requests" / "0001.json").stat().st_mtime_ns
            entries = list((home / probes._DIRECTORY).glob("*.json"))
            assert len(entries) >= 2  # the engine also publishes its registration
            assert all(entry.stat().st_mtime_ns >= marker_time for entry in entries)
            assert run(tmp_path / "warm") == first
            assert checked == [True]
            assert published == [probes.ENGINE_VERSION]
        finally:
            server.shutdown()
