"""Engine-free checks for host metrics: what the scale gate bounds, how it is written, and lock hygiene."""
from contextlib import contextmanager
import json
import os
import threading
import time

import pytest

import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor
from stateframe.parallel_channel import ChannelWrites
from stateframe.parallel_journal import Journal
from stateframe.parallel_runtime import lifecycle_loop
from _timeouts import scaled


@contextmanager
def _channel_lane(supervisor):
    """Use the actual publication journal, including its helper process and fsyncs."""
    writer = supervisor._channel_writes = ChannelWrites()
    try:
        yield writer
    finally:
        process = writer.journal._process
        writer.close()
        writer.journal._thread.join(scaled(5))
        assert not writer.journal._thread.is_alive()
        if process is not None:
            assert process.wait(timeout=scaled(5)) == 0
            process.stdout.close()


def _metrics(run):
    return json.loads((run / "parallel" / "host-metrics.json").read_text())


def _channel_with_status_request(run, *, age_seconds):
    from test_parallel_host import _request, _write_request
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    path = rq / "001.json"
    _write_request(path, _request(1, "call", 1, "status", {"unit_id": "a"}))
    arrived = time.time() - age_seconds
    os.utime(path, (arrived, arrived))
    return rq.parent / "an" / "001.json"


# ---- body 8: the gate's metrics measure request-to-answer and pass latency -------------------

def test_metrics_record_answer_latency_and_pass_percentiles(tmp_path, monkeypatch):
    run = tmp_path / "run"
    answer = _channel_with_status_request(run, age_seconds=0.3)
    supervisor = ParallelSupervisor(run)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    with _channel_lane(supervisor):
        supervisor._service_channels()
    assert answer.is_file()
    metrics = _metrics(run)
    answers = metrics["answer_latency_seconds"]
    assert answers["count"] == 1 and answers["max"] >= 0.3
    for series in ("answer_latency_seconds", "channel_pass_seconds", "lifecycle_pass_seconds"):
        assert set(metrics[series]) == {"max", "p99", "p99_9", "count"}
        assert metrics[series]["p99"] <= metrics[series]["p99_9"] <= metrics[series]["max"]
    assert metrics["channel_pass_seconds"]["count"] == 1 and metrics["channel_pass_seconds"]["max"] > 0
    assert set(metrics["phase_seconds"]) == set(metrics["max_phase_seconds"]) == {"commits", "workers", "seals", "seal_discovery", "channels"}
    assert metrics["max_phase_seconds"]["channels"] > 0
    # tick() is memory-only: its latency says nothing about the 0.3 s the client waited.
    assert metrics["max_tick_latency_seconds"] < 0.3 <= answers["max"]


def test_lifecycle_pass_latency_is_recorded_from_the_lifecycle_lane(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor, "_pump_commits", lambda: advance(0.3))
    monkeypatch.setattr(supervisor, "_advance_workers", lambda: advance(0.2))
    monkeypatch.setattr(supervisor, "_advance_seals", lambda: advance(0.1))
    import weakref
    import stateframe.parallel_runtime as runtime
    passes = []

    def sleep(_):
        passes.append(1)
        if len(passes) >= 3:
            supervisor._runtime_stop = True
    monkeypatch.setattr(runtime.time, "sleep", sleep)
    lifecycle_loop(weakref.ref(supervisor))
    assert supervisor._lifecycle_pass_latency.count == 3
    supervisor._service_channels()
    metrics = _metrics(supervisor.run_dir)
    assert metrics["lifecycle_pass_seconds"]["count"] == 3
    assert metrics["lifecycle_pass_seconds"]["max"] == pytest.approx(0.6)
    for phase, seconds in (("commits", 0.3), ("workers", 0.2), ("seals", 0.1)):
        assert metrics["phase_seconds"][phase] == pytest.approx(seconds)
        assert metrics["max_phase_seconds"][phase] == pytest.approx(seconds)


def test_latency_stat_percentiles_are_bounded_and_ordered():
    stat = host._LatencyStat()
    for value in range(1, 1001):
        stat.record(value / 1000)
    snapshot = stat.snapshot()
    assert snapshot["count"] == 1000 and snapshot["max"] == 1.0
    assert snapshot["p99"] == pytest.approx(0.99) and snapshot["p99_9"] == pytest.approx(0.999)
    for _ in range(host._LATENCY_SAMPLES):
        stat.record(0.001)
    rolled = stat.snapshot()
    assert rolled["max"] == 1.0 and rolled["p99_9"] == pytest.approx(0.001)  # window rolled, max kept
    assert host._LatencyStat().snapshot() == {"max": 0.0, "p99": 0.0, "p99_9": 0.0, "count": 0}


# ---- body 13: metrics are written directly, outside the pass's finally --------------------------

def test_metrics_write_never_rides_a_faulted_journal(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    calls = []

    def unavailable(journal, kind, payload):
        calls.append(journal)
        time.sleep(0.3)
        raise host.JournalFault("journal helper unavailable after 6 restarts")

    monkeypatch.setattr(Journal, "submit", unavailable)
    supervisor._journal = Journal()
    try:
        with _channel_lane(supervisor):
            started = time.perf_counter()
            supervisor._service_channels()
            assert time.perf_counter() - started < 0.1
            assert calls == []  # neither the channel journal nor the worker journal
    finally:
        supervisor._journal.close()
        supervisor._journal._thread.join(scaled(5))
        assert not supervisor._journal._thread.is_alive()
    assert _metrics(supervisor.run_dir)["schema"] == "stateframe-parallel-host-metrics/1"


def test_metrics_failure_does_not_mask_the_pass_exception_and_is_logged_once(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "parallel" / "call" / "attempt-1" / "rq").mkdir(parents=True)
    supervisor = ParallelSupervisor(run)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    monkeypatch.setattr(supervisor, "_budget_expired", lambda: False)

    writes = []

    def broken(path, value, **kw):
        writes.append(path)
        raise OSError("metrics disk full")
    monkeypatch.setattr(host, "_write_json", broken)
    monkeypatch.setattr(supervisor, "serve", lambda rq, an: (_ for _ in ()).throw(RuntimeError("pass failed")))
    with pytest.raises(RuntimeError, match="pass failed"):
        supervisor._service_channels()
    monkeypatch.setattr(supervisor, "serve", lambda rq, an: ())
    for _ in range(5):
        supervisor._service_channels()
    logged = [entry for entry in supervisor._protocol_errors if "host metrics not written" in entry]
    assert len(logged) == 1
    assert len(writes) == 1  # deduplication alone must not hide repeated disk I/O
    assert not (run / "parallel" / "host-metrics.json").exists()


# ---- body 14: retain proof for inherited seal/launch lock changes; exercise metrics I/O too ----

def _owned(lock):
    return lock._is_owned()


def test_seal_admission_reads_records_and_results_outside_the_host_lock(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _Worker, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    first = ParallelSupervisor(run)
    monkeypatch.setattr(first, "_start_runtime", lambda: None)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED",
                     exited_seq=1)
    first._persist_worker(worker)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    seals = call / "host" / "seals"
    seals.mkdir(parents=True)
    (seals / "a-1.json").write_text(json.dumps({"schema": "stateframe-parallel-seal-job/2",
                                                "operation_id": host.operation_id(request),
                                                "request": request.to_json()}))
    (seals / "a-1.result.json").write_text(json.dumps({"status": "OK", "payload": {"sealed": True}}))

    resumed = ParallelSupervisor(run)  # hydrates nothing into memory for units without launch records
    resumed._workers.clear()
    locked_reads = []
    real_read = host._read_json

    def spying_read(path, label):
        locked_reads.append((label, _owned(resumed._lock)))
        return real_read(path, label)
    monkeypatch.setattr(host, "_read_json", spying_read)
    real_channel_read = resumed._read_channel_json

    def spying_channel_read(path, label):
        locked_reads.append((label, _owned(resumed._lock)))
        return real_channel_read(path, label)
    monkeypatch.setattr(resumed, "_read_channel_json", spying_channel_read)
    status, _, payload = resumed._enqueue_seal(request, call)
    assert (status, payload) == ("OK", {"sealed": True})
    labels = {label for label, _ in locked_reads}
    assert {"subframe supervisor record", "subframe seal job", "subframe seal result"} <= labels
    assert not any(owned for _, owned in locked_reads), locked_reads


def test_launch_probes_prior_group_liveness_before_taking_the_lock(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _Worker, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED",
                     exited_seq=1)
    worker.pid, worker.pgid = 4242, 4242
    supervisor._workers[("call", "a", 1)] = worker
    probes = []

    def probe(run_dir, pid, pgid, birth, process_file):
        probes.append(_owned(supervisor._lock))
        time.sleep(0.05)  # a strict /proc scan at load
        return False
    monkeypatch.setattr(host, "_read_json", lambda path, label: {"kernel_identity": "probe"})
    monkeypatch.setattr(host, "_recorded_kernel_identity", lambda record: "probe")
    monkeypatch.setattr(host, "_recorded_group_live", probe)
    started = []
    monkeypatch.setattr(supervisor, "_start_preparation", lambda w: started.append(w.unit_id))
    monkeypatch.setattr(supervisor, "_persist_worker", lambda w, **kw: None)
    payload = {"unit_id": "a", "execution_id": "execution", "spec_path": str(spec_path),
               "spec_digest": spec.digest(), "run_dir": str(unit), "incarnation": 2,
               "start_snapshot_path": str(snapshot), "start_manifest_digest": spec.start_manifest_digest,
               "snapshot_manifest_digest": worker.snapshot_manifest_digest}
    # A queued worker waits on the non-blocking lock in _start_waiting_workers while a launch probes.
    queued = _Worker("call", "b", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="STARTING")
    queued.queued = True
    supervisor._workers[("call", "b", 1)] = queued
    waiting_started = []
    blocked = threading.Event()
    passes_completed = threading.Event()

    def waiting():
        if not blocked.wait(5):
            return
        try:
            for _ in range(2):
                supervisor._start_waiting_workers()
                if "b" in started:
                    waiting_started.append(True)
                    return
                time.sleep(0.01)
        finally:
            passes_completed.set()
    thread = threading.Thread(target=waiting, daemon=True)
    thread.start()
    original = host._recorded_group_live

    def probe_and_release(*args):
        blocked.set()
        result = original(*args)
        assert passes_completed.wait(5), "queued-worker passes did not complete during the probe"
        return result
    monkeypatch.setattr(host, "_recorded_group_live", probe_and_release)
    status, _, result = supervisor._launch(_request(1, "call", 2, "launch", payload), call)
    thread.join(5)
    assert (status, result) == ("PENDING", {"state": "STARTING"})
    assert probes == [False], probes  # probed exactly once, with the host lock released
    assert waiting_started, "queued worker could not start while the launch probed /proc"


def test_slow_journal_status_burst_is_visible_in_gate_metrics(tmp_path, monkeypatch):
    """A backed-up publication lane raises answer latency without blocking passes."""
    from test_parallel_host import _request, _write_request

    supervisor = ParallelSupervisor(tmp_path / "run")
    rq = supervisor.parallel_root / "call" / "attempt-1" / "rq"
    for seq in range(1, 6):
        _write_request(rq / f"{seq:03d}.json", _request(1, "call", seq, "status", {"unit_id": "a"}))

    original_read = Journal._read_response

    def delayed_ack(journal, process, buffer):
        time.sleep(0.3)
        return original_read(journal, process, buffer)

    monkeypatch.setattr(Journal, "_read_response", delayed_ack)
    with _channel_lane(supervisor):
        supervisor._channel_thread = threading.current_thread()
        # Keep lifecycle runtime out of this engine-free channel test; tick remains
        # the production memory-only heartbeat while the real channel journal runs.
        monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
        supervisor.tick()
        deadline = time.monotonic() + scaled(5)
        while supervisor._answer_latency.snapshot()["count"] < 5:
            assert time.monotonic() < deadline, list(supervisor._protocol_errors)
            supervisor._service_channels()
            time.sleep(0.01)
    metrics = _metrics(supervisor.run_dir)
    assert metrics["answer_latency_seconds"]["count"] == 5
    for key in ("max", "p99", "p99_9"):
        assert metrics["answer_latency_seconds"][key] >= 0.3
    assert metrics["channel_pass_seconds"]["max"] < 0.1
    assert metrics["max_tick_latency_seconds"] < 0.1
    assert metrics["max_phase_seconds"]["channels"] < 0.1


def test_answer_latency_ends_at_publication_ack_and_counts_partial_pass(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    answer = _channel_with_status_request(supervisor.run_dir, age_seconds=0)
    request = answer.parent.parent / "rq" / answer.name
    os.utime(request, (100.0, 100.0))
    clock = [100.0]
    monkeypatch.setattr(host.time, "time", lambda: 100.0)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    original_take = Journal.take

    def take(journal, ticket):
        result = original_take(journal, ticket)
        if result is not None and ticket.payload["path"] == str(answer):
            clock[0] += 0.25
        return result

    monkeypatch.setattr(Journal, "take", take)

    def serve(rq, an):
        from test_parallel_host import _request
        accepted = _request(1, "call", 1, "status", {"unit_id": "a"})
        supervisor._store_answer(an, 1, supervisor._request_identity(accepted),
                                 supervisor._answer(accepted, "OK", None, {}))
        clock[0] += 10.0  # later work must not extend an already acknowledged answer
        raise RuntimeError("later request failed")

    monkeypatch.setattr(supervisor, "serve", serve)
    with _channel_lane(supervisor):
        with pytest.raises(RuntimeError, match="later request failed"):
            supervisor._service_channels()
    # The exception path records completed answers but performs no metrics-file I/O.
    supervisor._flush_metrics()
    metrics = _metrics(supervisor.run_dir)
    assert metrics["answer_latency_seconds"] == {"max": 0.25, "p99": 0.25, "p99_9": 0.25, "count": 1}


def test_metrics_refresh_each_maximum_and_cadence_without_ticks(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    clock = [100.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    supervisor._answer_latency.record(10.0)
    supervisor._flush_metrics()
    supervisor._channel_pass_latency.record(0.5)
    supervisor._flush_metrics()
    assert _metrics(supervisor.run_dir)["channel_pass_seconds"]["max"] == 0.5
    supervisor._channel_pass_latency.record(0.1)
    clock[0] += host._METRICS_REFRESH_SECONDS
    supervisor._flush_metrics()
    assert _metrics(supervisor.run_dir)["channel_pass_seconds"]["count"] == 2
    assert supervisor.tick_count == 0


def test_metrics_disk_fault_backoff_covers_rising_maxima(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    clock = [100.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    writes = []

    def broken(*args, **kwargs):
        writes.append(clock[0])
        raise OSError("disk full")

    monkeypatch.setattr(host, "_write_json", broken)
    for value in (0.1, 0.2, 0.3):
        supervisor._channel_pass_latency.record(value)
        supervisor._flush_metrics()
    assert writes == [100.0]
    clock[0] += host._METRICS_REFRESH_SECONDS
    supervisor._flush_metrics()
    assert writes == [100.0, 105.0]
    assert supervisor._protocol_errors.total == 2


def test_channel_and_metrics_io_allow_queued_worker_to_progress(tmp_path, monkeypatch):
    """Body 14: the channel lane and metrics writer leave the host lock available."""
    from test_parallel_host import _Worker, _spec_and_snapshot

    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    queued = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="STARTING", queued=True)
    supervisor._workers[("call", "a", 1)] = queued
    started = threading.Event()
    monkeypatch.setattr(supervisor, "_start_preparation", lambda worker: started.set())
    attempts = []
    real_write = host._write_json

    def write(path, value, **kwargs):
        if path.name == "host-metrics.json":
            assert kwargs == {"replace_existing": True, "durable": False}
            # A different thread must acquire the lock during the actual file write.
            thread = threading.Thread(target=supervisor._start_waiting_workers)
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            assert started.is_set()
            attempts.append(1)
        return real_write(path, value, **kwargs)

    monkeypatch.setattr(host, "_write_json", write)
    supervisor._service_channels()
    assert attempts == [1]
    assert _metrics(run)["channel_pass_seconds"]["count"] == 1


def test_request_arriving_during_serve_is_measured_once(tmp_path, monkeypatch):
    from test_parallel_host import _request, _write_request

    supervisor = ParallelSupervisor(tmp_path / "run")
    rq = supervisor.parallel_root / "call" / "attempt-1" / "rq"
    rq.mkdir(parents=True)
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    answer = supervisor._answer(request, "OK", None, {})

    def serve(rq, an):
        path = rq / "001.json"
        _write_request(path, request)
        arrived = time.time() - 0.25
        os.utime(path, (arrived, arrived))
        supervisor._store_answer(an, 1, supervisor._request_identity(request), answer)
        return (answer,)

    monkeypatch.setattr(supervisor, "serve", serve)
    with _channel_lane(supervisor):
        supervisor._service_channels()
    metrics = _metrics(supervisor.run_dir)
    assert metrics["answer_latency_seconds"]["count"] == 1
    assert metrics["answer_latency_seconds"]["max"] >= 0.25
    # An idempotent replay of that answer must not count the request twice.
    monkeypatch.setattr(supervisor, "serve", lambda rq, an: (answer,))
    supervisor._service_channels()
    assert supervisor._answer_latency.snapshot()["count"] == 1
