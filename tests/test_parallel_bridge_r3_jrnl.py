"""Real-pipe watchdog coverage and bounded seal retries with fresh owner proofs."""
import fcntl
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_channel as channel_mod
import stateframe.parallel_host as host
import stateframe.parallel_journal as journal_mod
from stateframe.parallel_journal import Journal
from test_parallel_bridge import DelayedJournal
from test_parallel_bridge_journal_progress import PipeProcess, _fast_journal
from test_parallel_bridge_seal_spawn import _reserved_operation


class _UnreadHelper:
    """An alive, non-reading helper with a real request pipe smaller than the request."""

    def __init__(self):
        read_fd, write_fd = os.pipe()
        fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, 4096)
        out_read, out_write = os.pipe()
        self.stdin = os.fdopen(write_fd, "wb", buffering=0)
        self.stdout = os.fdopen(out_read, "rb", buffering=0)
        self.requests = os.fdopen(read_fd, "rb", buffering=0)
        self.responses = os.fdopen(out_write, "wb", buffering=0)
        self.kills = 0

    def kill(self):
        self.kills += 1
        self.requests.close()
        self.responses.close()

    def wait(self, timeout=None):
        return -9

    def close(self):
        self.requests.close()
        self.responses.close()


@pytest.mark.skipif(not hasattr(fcntl, "F_SETPIPE_SZ"), reason="requires Linux pipe capacity control")
def test_request_pipe_backpressure_is_covered_by_progress_watchdog(monkeypatch):
    _fast_journal(monkeypatch)
    spawned = []

    def spawn(journal):
        journal.epoch += 1
        process = _UnreadHelper()
        spawned.append(process)
        return process

    monkeypatch.setattr(Journal, "_spawn", spawn)
    journal = Journal()
    try:
        ticket = journal.submit("write", {"value": "x" * (12 << 10)})
        assert ticket.done.wait(2), "a full request pipe must not bypass the watchdog"
        result = journal.take(ticket)
        assert result["lost"] is True and "no progress" in result["error"]
        assert journal.fault is not None
        assert len(spawned) == 2 and all(process.kills == 1 for process in spawned)
        assert journal._credits._value == journal_mod._MAX_PENDING
    finally:
        journal.close()
        # Unblock the old blocking implementation too, so a red run leaves no stuck reactor.
        for process in spawned:
            process.close()
        journal._thread.join(2)
    assert not journal._thread.is_alive()


@pytest.mark.parametrize("descheduled", (False, True), ids=("steady", "descheduled"))
def test_request_partial_writes_preserve_exact_message_and_count_progress(monkeypatch, descheduled):
    _fast_journal(monkeypatch)
    process = PipeProcess()
    monkeypatch.setattr(Journal, "_spawn", lambda journal: process)
    real_write = os.write
    sent = []
    if descheduled:
        clock = [0.0]

        def monotonic():
            clock[0] += 0.1  # the reactor resumes after a whole no-progress window
            return clock[0]

        monkeypatch.setattr(journal_mod, "time", SimpleNamespace(monotonic=monotonic, sleep=time.sleep))

    def partial_write(descriptor, data):
        # Each write takes less than the no-progress window, but the request takes much longer.
        time.sleep(0.015)
        count = real_write(descriptor, data[:32 if descheduled else 512])
        sent.append(bytes(data[:count]))
        return count

    monkeypatch.setattr(journal_mod.os, "write", partial_write)
    journal = Journal()
    payload = {"value": "x" * (1024 if descheduled else 12 << 10)}
    try:
        ticket = journal.submit("write", payload)
        assert ticket.done.wait(3), "partial writes must finish while making progress"
        assert journal.take(ticket)["ok"] is True and process.kills == 0
        assert len(sent) > 20, "the request must use partial nonblocking writes"
        assert json.loads(b"".join(sent)) == {"id": ticket.id, "kind": "write", "payload": payload}
        assert process.received == [{"id": ticket.id, "kind": "write", "payload": payload}]
    finally:
        journal.close()
        journal._thread.join(2)


class _CrashingHelper(PipeProcess):
    """Accept valid requests, then lose the response at the chosen seal phase."""

    def __init__(self, phase):
        def answer(message):
            if message["kind"] == phase:
                self._responses.close()  # EOF after request delivery, as with a helper crash
                return None
            return {"id": message["id"], "ok": True, "token": message["payload"]["token"],
                    "staged": {}}
        super().__init__(answer)


@pytest.mark.parametrize("phase", ["seal_build", "seal_publish"])
def test_seal_crash_loop_has_cross_ticket_budget_and_ack_gated_release(tmp_path, monkeypatch, phase):
    _fast_journal(monkeypatch)
    supervisor, worker, operation = _reserved_operation(tmp_path)
    helper = Journal(role="seal")
    supervisor._seal_helpers = [helper]
    spawned, nonces = [], []
    clock = [100.0]
    monkeypatch.setattr(host, "time", SimpleNamespace(monotonic=lambda: clock[0],
                                                     monotonic_ns=time.monotonic_ns))

    def spawn(journal):
        journal.epoch += 1
        process = _CrashingHelper(phase)
        spawned.append(process)
        return process

    monkeypatch.setattr(Journal, "_spawn", spawn)
    try:
        # Enough attempts to exceed any supported loss budget; every request has a fresh ticket.
        for _ in range(32):
            if operation.phase == "COMMITTING":
                break
            if operation.ticket is None:
                supervisor._advance_seals()
            ticket = operation.ticket
            assert ticket is not None
            assert ticket.done.wait(2)
            kind = ticket.kind
            supervisor._advance_seals()
            if kind == phase:
                nonces.append(ticket.payload["token"][-1])
                clock[0] = max(clock[0], getattr(operation, "retry_at", 0.0))
        assert operation.phase == "COMMITTING", "helper crashes must exhaust a cross-ticket budget"
        assert 1 < len(spawned) <= 6
        assert len(set(nonces)) == len(nonces), "a lost helper must never reuse its proof nonce"
        assert operation.staged["result"]["status"] == "ERROR"
        assert "seal helper unavailable" in operation.staged["result"]["payload"]["error"]
        assert helper._credits._value == journal_mod._MAX_PENDING

        # Exercise the asynchronous channel ACK fence: visibility alone cannot free admission.
        monkeypatch.setattr(channel_mod, "Journal", DelayedJournal)
        writer = supervisor._channel_writes = channel_mod.ChannelWrites()
        supervisor._channel_thread = threading.current_thread()
        supervisor._publish_seal_results()
        publication = writer.journal.tickets[-1]
        host._write_json(operation.result, publication.payload["value"])
        supervisor._publish_seal_results()
        assert operation.job in supervisor._seal_ops and operation.phase == "COMMITTING"
        publication.result = {"ok": True}
        supervisor._publish_seal_results()
        assert operation.phase == "DONE" and operation.job not in supervisor._seal_ops
        assert operation.job in supervisor._sealed_jobs
        assert host._read_json(operation.result, "seal error")["status"] == "ERROR"
        assert worker is operation.worker
    finally:
        helper.close()
        helper._thread.join(2)


def test_seal_loss_retry_deadline_survives_invalidation(tmp_path, monkeypatch):
    supervisor, worker, operation = _reserved_operation(tmp_path)
    helper = DelayedJournal()
    supervisor._seal_helpers = [helper]
    clock = [100.0]
    monkeypatch.setattr(host, "time", SimpleNamespace(monotonic=lambda: clock[0],
                                                     monotonic_ns=time.monotonic_ns))
    supervisor._advance_seals()
    first = operation.ticket
    first.result = {"ok": False, "lost": True, "error": "seal helper lost"}
    supervisor._advance_seals()
    retry_at = getattr(operation, "retry_at", 0.0)
    assert retry_at > clock[0], "a lost seal helper must arm a retry deadline"
    first_delay = retry_at - clock[0]
    worker.generation += 1  # invalidation must not reset the deadline or loss count
    for _ in range(20):
        supervisor._advance_seals()
    assert len(helper.tickets) == 1 and operation.ticket is None
    assert operation.retry_at == retry_at and operation.helper_losses == 1
    clock[0] = retry_at
    supervisor._advance_seals()
    assert len(helper.tickets) == 2 and operation.ticket is not first
    assert operation.ticket.payload["token"] != first.payload["token"]
    assert operation.proof is not None and operation.accepts(operation.proof, helper.epoch)
    operation.ticket.result = {"ok": False, "lost": True, "error": "seal helper lost again"}
    supervisor._advance_seals()
    assert operation.helper_losses == 2 and operation.ticket is None
    assert operation.retry_at - clock[0] == pytest.approx(2 * first_delay)
