"""INTG regressions: completed observations outrank stale timeout decisions."""
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import subprocess

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_journal as journal_mod
from stateframe.engine import kernel_bounded_http as http
from stateframe.parallel_journal import Journal, JournalFault, checked_response


@pytest.mark.parametrize("kind", ("dialect", "health"))
def test_completed_probe_survives_ready_pipe_and_child_exit_timeouts(monkeypatch, kind):
    """Complete replies survive either side of communicate's read/exit timeout race."""
    proof = ([200, {"default_generation_settings": {"n_ctx": 65536}}]
             if kind == "dialect" else [200, True])
    encoded = json.dumps(proof).encode()
    outputs = [(encoded, None), (None, encoded), (encoded[:9], encoded[9:]),
               (None, b'[200, {"unfinished":'), (encoded, None)]
    cleaned = []

    class Probe:
        returncode = None

        def __init__(self, collected, ready):
            self.collected = collected
            self.stdout = None
            if ready is not None:
                read_fd, write_fd = os.pipe()
                self.stdout = os.fdopen(read_fd, "rb", buffering=0)
                try:
                    os.write(write_fd, ready)
                finally:
                    os.close(write_fd)

        def communicate(self, packet, timeout):
            assert json.loads(packet)["paths"] == (["/props"] if kind == "dialect" else ["/health"])
            raise subprocess.TimeoutExpired("probe", timeout, output=self.collected)

    monkeypatch.setattr(http, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(http, "subprocess", SimpleNamespace(
        Popen=lambda *a, **kw: Probe(*outputs.pop(0)), PIPE=subprocess.PIPE,
        DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(http, "_terminate_child", lambda process: cleaned.append(process))

    def probe():
        if kind == "dialect":
            return http.bounded_json_probe("http://127.0.0.1:8080/v1", "", "/props")
        return http._bounded_probe("https://probe.invalid/v1", "", 102.0, paths=("/health",))

    for _ in range(3):
        assert probe() == (200, proof[1])
    assert probe() is None, "partial JSON must never prove capabilities or health"
    monkeypatch.setattr(http, "_MAX_PROBE_RESPONSE_BYTES", 1)
    assert probe() is None, "expiry recovery must remain bounded"
    assert len(cleaned) == 5 and not outputs
    assert all(process.stdout is None or process.stdout.closed for process in cleaned)


def _reply_after_stale_timeout(monkeypatch, reply):
    """select's final empty result predates bytes written while its caller was paused."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb", buffering=0)
    real_select = journal_mod.select.select
    calls = []

    def select(read, write, error, timeout):
        calls.append(timeout)
        if len(calls) <= 3:
            if len(calls) == 3:
                os.write(write_fd, json.dumps(reply).encode() + b"\n")
            return [], [], []
        return real_select(read, write, error, timeout)

    monkeypatch.setattr(journal_mod, "select", SimpleNamespace(select=select))
    monkeypatch.setattr(journal_mod, "_NO_PROGRESS_SLICES", 3)
    try:
        result = Journal.__new__(Journal)._read_response(SimpleNamespace(stdout=reader), bytearray())
        assert json.loads(result) == reply
        assert calls[-1] == 0.0, "expiry may poll ready bytes but may not extend the wait window"
        return json.loads(result)
    finally:
        reader.close()
        os.close(write_fd)


def test_journal_progress_arriving_after_last_timeout_is_not_killed(monkeypatch):
    assert _reply_after_stale_timeout(monkeypatch, {"ok": True})["ok"] is True


def test_seal_evidence_rejection_at_watchdog_boundary_keeps_its_type(monkeypatch):
    result = _reply_after_stale_timeout(monkeypatch, {
        "ok": False, "error_class": "ParallelHostError", "error": "check evidence digest differs"})
    with pytest.raises(host.ParallelHostError, match="evidence"):
        checked_response(result)


def test_failed_seal_publication_cannot_bypass_its_configured_journal(tmp_path):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    calls = []

    class FaultedJournal:
        def call(self, kind, payload):
            calls.append((kind, payload))
            raise JournalFault("durability unavailable")

    supervisor._journal = FaultedJournal()
    job = supervisor.run_dir / "seal/job.json"
    supervisor._fail_seal_job(job, "invalid job")
    assert calls and calls[0][0] == "write", "configured durability authority must receive the write"
    assert job not in supervisor._sealed_jobs
    assert not job.with_name("job.result.json").exists()
    assert any("deferred" in error for error in supervisor._protocol_errors)
    # Seal admission must not turn the worker journal into the foreground channel lane.
    channel = supervisor.run_dir / "parallel/call/attempt-1/an/001.record.json"
    supervisor._write_json(channel, {"identity": "channel"})
    assert json.loads(channel.read_text()) == {"identity": "channel"}
    assert len(calls) == 1, "foreground channel writes queued behind the faulted worker journal"

    class AckingJournal:
        def call(self, kind, payload):
            assert kind == "write" and payload["replace"] is False and payload["durable"] is True
            assert job not in supervisor._sealed_jobs, "seal completed before its durability ACK"
            assert payload["path"] == str(job.with_name("job.result.json"))
            host._write_json(job.with_name("job.result.json"), payload["value"])
            return {"ok": True}

    supervisor._journal = AckingJournal()
    supervisor._fail_seal_job(job, "invalid job")
    assert job in supervisor._sealed_jobs
    assert json.loads(job.with_name("job.result.json").read_text())["status"] == "ERROR"


def test_channel_discovery_cannot_spend_an_expired_pass_budget(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    expired = [False]
    monkeypatch.setattr(supervisor, "_budget_expired", lambda: expired[0])
    monkeypatch.setattr(supervisor, "_publish_seal_results", lambda: None)
    monkeypatch.setattr(supervisor, "_flush_metrics", lambda: None)

    def discover():
        expired[0] = True

    monkeypatch.setattr(supervisor, "_discover_seals", discover)
    original = type(supervisor.parallel_root).iterdir

    def iterdir(path):
        assert path != supervisor.parallel_root, "channel directory scan began after the budget expired"
        return original(path)

    monkeypatch.setattr(type(supervisor.parallel_root), "iterdir", iterdir)
    supervisor._service_channels()
    assert supervisor._channel_pass_latency.count == 1


def _recorded_exit(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot

    call, run, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    worker = host._Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), run,
                          snapshot, spec.start_manifest_digest, {}, state="RUNNING", pid=4242, pgid=4242)
    supervisor._workers[("call", "a", 1)] = worker
    process_file = run / "proc/1/process.json"
    process_file.parent.mkdir(parents=True)
    identity = {"pid": 4242, "pgid": 4242, "kernel_identity": "darwin:4242:1"}
    process_file.write_text(json.dumps({**identity, "verified_members": {}}))
    # Recovered completion needs the former handle's birth-bound family sweep proof.
    process_file.with_name("family-exited.json").write_text(json.dumps(identity))
    observer = host.ProcessTable().identity(os.getpid())
    leader = replace(observer, pid=4242, pgid=4242, zombie=True)
    scans = []

    def scan(table):
        scans.append(table)
        return {observer.pid: observer, leader.pid: leader}

    monkeypatch.setattr(host.ProcessTable, "snapshot", scan)
    monkeypatch.setattr(host, "_process_identity", lambda pid:
                        (4242, "Z", identity["kernel_identity"], []) if pid == 4242 else None)
    monkeypatch.setattr(host, "_terminal", lambda run: "SUBMITTED")
    monkeypatch.setattr(supervisor, "_persist_worker", lambda worker: None)
    return supervisor, worker, scans, process_file


@pytest.mark.skipif(os.name != "posix" or not host.sys.platform.startswith("linux"), reason="Linux table path")
def test_resumed_exit_reuses_its_fresh_scan_without_private_enumeration(tmp_path, monkeypatch):
    from stateframe.parallel_worker import TickProcessTable

    supervisor, worker, scans, _ = _recorded_exit(tmp_path, monkeypatch)
    real_killpg = host.os.killpg

    def present(pgid, sig):
        if (pgid, sig) == (4242, 0):
            return None  # Presence is not enough to decide whether the group has live rows.
        return real_killpg(pgid, sig)

    monkeypatch.setattr(host.os, "killpg", present)
    supervisor._advance_workers()
    assert worker.state == "EXITED" and worker.terminal == "SUBMITTED"
    assert scans == [supervisor._process_table] * 2, "exit added a private third full-table scan"
    assert host._RECORDED_OBSERVATION.get() is None, "observation scope leaked into later signal paths"
    # Explicit freshness remains fresh, and its result supersedes the tick's old view.
    table = TickProcessTable()
    with table.tick():
        table.tick_snapshot()
        first = table.view_token()
        table.snapshot()
        assert table.view_token() != first
        table.tick_snapshot()
    assert scans[-2:] == [table, table]
    assert len(scans) == 4

    # A member seen live during the final family probe can disappear before its
    # identity recheck. Even the saved sweep proof cannot resolve that uncertainty.
    with monkeypatch.context() as unreadable:
        supervisor, worker, _, _ = _recorded_exit(tmp_path / "unreadable", unreadable)
        unreadable.setattr(host, "_live_group_members", lambda pgid: [4242])
        unreadable.setattr(host, "_process_identity", lambda pid: None)
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.exited_seq is None
        assert host._RECORDED_OBSERVATION.get() is None


@pytest.mark.skipif(not host.sys.platform.startswith("linux"), reason="Linux atomic group query")
@pytest.mark.parametrize("mode", ("absent", "denied", "detached"))
def test_absent_group_avoids_rescan_without_bypassing_saved_births(tmp_path, monkeypatch, mode):
    supervisor, worker, scans, process_file = _recorded_exit(tmp_path, monkeypatch)
    real_killpg = host.os.killpg
    queries = []

    def query(pgid, sig):
        if (pgid, sig) != (4242, 0):
            return real_killpg(pgid, sig)
        queries.append((pgid, sig))
        if mode == "denied":
            raise PermissionError("group query denied")
        raise ProcessLookupError("group absent")

    monkeypatch.setattr(host.os, "killpg", query)
    if mode == "detached":
        record = json.loads(process_file.read_text())
        record["verified_members"] = {"5555": "darwin:5555:1"}
        process_file.write_text(json.dumps(record))
        monkeypatch.setattr(host, "_process_identity", lambda pid:
                            (7777, "R", "darwin:5555:1", []) if pid == 5555 else None)
        monkeypatch.setattr(host, "_terminal", lambda run: pytest.fail("live saved birth reached finalization"))

    supervisor._advance_workers()
    assert scans == [supervisor._process_table] * (2 if mode == "denied" else 1)
    assert queries == [(4242, 0)], "passive exit did not make the atomic absence query"
    assert host._RECORDED_OBSERVATION.get() is None
    if mode == "detached":
        assert worker.state == "RUNNING" and worker.exited_seq is None
    else:
        assert worker.state == "EXITED" and worker.terminal == "SUBMITTED"

    # Atomic group absence (or a denied query) cannot turn an unreadable member
    # observed by the final family probe into evidence of death.
    with monkeypatch.context() as unreadable:
        supervisor, worker, _, _ = _recorded_exit(tmp_path / "unreadable", unreadable)
        unreadable.setattr(host, "_live_group_members", lambda pgid: [4242])
        unreadable.setattr(host, "_process_identity", lambda pid: None)
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.exited_seq is None
        assert host._RECORDED_OBSERVATION.get() is None


def test_request_pipe_progress_after_stale_timeout_keeps_exact_bytes(monkeypatch):
    read_fd, write_fd = os.pipe()
    writer = os.fdopen(write_fd, "wb", buffering=0)
    clock = [0.0]
    calls = []
    real_select = journal_mod.select.select

    def select(read, write, error, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            clock[0] = 100.0
            return [], [], []
        return real_select(read, write, error, timeout)

    monkeypatch.setattr(journal_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(journal_mod, "select", SimpleNamespace(select=select))
    message = b'{"kind":"write","payload":{"value":"exact"}}\n'
    try:
        Journal.__new__(Journal)._write_request(SimpleNamespace(stdin=writer), message)
        assert os.read(read_fd, len(message)) == message
        assert calls[-1] == 0.0
    finally:
        writer.close()
        os.close(read_fd)
