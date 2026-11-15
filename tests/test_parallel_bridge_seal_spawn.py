"""Seal helper spawn failures are bounded and terminal; ERROR results always release."""
import errno
import subprocess
import time
from types import SimpleNamespace

import stateframe.parallel_host as host
import stateframe.parallel_journal as journal_mod
from stateframe.parallel_journal import Journal
from stateframe.parallel_seals import SealOperation
from test_parallel_bridge import DelayedJournal
from test_parallel_bridge_seals import _exited_worker


def _reserved_operation(tmp_path):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    seals = tmp_path / "run" / "host" / "seals"
    seals.mkdir(parents=True)
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), seals / "a-1.json",
                       seals / "a-1.result.json", worker.generation)
    supervisor._seal_ops[op.job] = op
    return supervisor, worker, op


def test_seal_helper_spawn_failure_is_capped_and_releases_reservation(tmp_path, monkeypatch):
    # Load-99 case: fork fails with EAGAIN for every attempt. Without a cap the lifecycle lane
    # resubmits the operation on every visit (a fork storm) and the reservation never leaves.
    monkeypatch.setattr(journal_mod, "_RESTART_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(journal_mod, "_RESTART_BACKOFF_MAX_SECONDS", 0.002)
    attempts = []

    def failing_popen(*args, **kwargs):
        attempts.append(args[0] if args else kwargs.get("args"))
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(subprocess, "Popen", failing_popen)
    supervisor, worker, op = _reserved_operation(tmp_path)
    helpers = supervisor._seal_helpers = [Journal(role="seal"), Journal(role="seal")]
    try:
        for _ in range(60):  # far more passes than the restart budget needs
            supervisor._advance_seals()
            supervisor._publish_seal_results()
            if op.job not in supervisor._seal_ops:
                break
            time.sleep(0.01)
    finally:
        for helper in helpers:
            helper.close()
    cap = journal_mod._MAX_RESTARTS_PER_REQUEST
    assert len(attempts) <= cap * len(helpers), f"fork storm: {len(attempts)} spawn attempts"
    assert op.phase == "DONE" and op.job not in supervisor._seal_ops
    result = host._read_json(op.result, "result")
    assert result["status"] == "ERROR" and "seal helper unavailable" in result["payload"]["error"]
    # The launch gate for the next incarnation only looks at live reservations of this worker.
    assert not any(pending.worker is worker and pending.phase != "DONE"
                   for pending in supervisor._seal_ops.values())


def test_seal_helper_spawn_failure_result_is_marked_terminal(monkeypatch):
    monkeypatch.setattr(journal_mod, "_RESTART_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(journal_mod, "_RESTART_BACKOFF_MAX_SECONDS", 0.002)
    attempts = []

    def failing_popen(*args, **kwargs):
        attempts.append(1)
        raise OSError(errno.ENOMEM, "Cannot allocate memory")

    monkeypatch.setattr(subprocess, "Popen", failing_popen)
    helper = Journal(role="seal")
    try:
        ticket = helper.submit("seal_build", {"token": [1, 0, "n"]})
        assert ticket.done.wait(5.0), "spawn failures must not wait on the helper forever"
        result = helper.take(ticket)
    finally:
        helper.close()
    assert result == {"ok": False, "lost": True, "spawn_failed": True, "error": result["error"]}
    assert len(attempts) == journal_mod._MAX_RESTARTS_PER_REQUEST
    assert "unavailable after" in result["error"]


def test_seal_helper_mid_request_loss_is_not_replayed(monkeypatch):
    # A helper that started and then died holds an unknown share of the proof: the loss is
    # reported at once and never retried on the same ticket (publication permits need a fresh proof).
    monkeypatch.setattr(journal_mod, "_RESTART_BACKOFF_SECONDS", 0.001)
    spawns = []

    from test_parallel_bridge_journal_progress import PipeProcess

    class DeadChild(PipeProcess):
        def __init__(self):
            def crash(message):
                self._responses.close()
                return None
            super().__init__(crash)

    def popen(*args, **kwargs):
        spawns.append(1)
        return DeadChild()

    monkeypatch.setattr(subprocess, "Popen", popen)
    helper = Journal(role="seal")
    try:
        ticket = helper.submit("seal_build", {"token": [1, 0, "n"]})
        assert ticket.done.wait(5.0)
        result = helper.take(ticket)
    finally:
        helper.close()
    assert result == {"ok": False, "lost": True, "error": "seal helper lost"}
    assert len(spawns) == 1


def test_error_seal_result_releases_despite_generation_bump_after_write(tmp_path, monkeypatch):
    # Cancel churn bumps worker.generation between the durable ERROR write and the host lock.
    # The ERROR is already on disk and publishes nothing, so the reservation must be released
    # on this pass rather than returning to PENDING while _launch refuses the next incarnation.
    supervisor, worker, op = _reserved_operation(tmp_path)
    supervisor._seal_helpers = [DelayedJournal()]
    supervisor._complete_seal(op, {"status": "ERROR", "payload": {"error": "seal helper unavailable"}})
    original = supervisor._write_json

    def bumping_write(path, value, **kwargs):
        written = original(path, value, **kwargs)
        worker.generation += 1  # a cancel landed after the write, before the post-write gate
        return written

    monkeypatch.setattr(supervisor, "_write_json", bumping_write)
    supervisor._publish_seal_results()
    assert op.phase == "DONE", f"ERROR result re-gated on generation: phase={op.phase}"
    assert op.job not in supervisor._seal_ops and op.job in supervisor._sealed_jobs
    assert host._read_json(op.result, "result")["status"] == "ERROR"


def test_ok_seal_result_is_still_gated_on_generation_after_write(tmp_path, monkeypatch):
    # The permit path keeps its post-write gate: an OK publication whose owner moved is retried.
    supervisor, worker, op = _reserved_operation(tmp_path)
    supervisor._seal_helpers = [DelayedJournal()]
    supervisor._complete_seal(op, {"status": "OK", "payload": {"proposal_digest": "d"}})
    original = supervisor._write_json

    def bumping_write(path, value, **kwargs):
        written = original(path, value, **kwargs)
        worker.generation += 1
        return written

    monkeypatch.setattr(supervisor, "_write_json", bumping_write)
    supervisor._publish_seal_results()
    assert op.phase == "PENDING" and op.job in supervisor._seal_ops
