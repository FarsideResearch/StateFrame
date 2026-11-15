"""Engine-free checks for the channel lane: seal discovery faults, bounded fault logs, metrics cadence."""
import json

import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor
from stateframe.parallel_runtime import channel_loop


def _exited_worker(call, unit, snapshot, spec_path, spec, incarnation):
    from test_parallel_host import _Worker
    return _Worker("call", "a", "execution", incarnation, call, spec_path, spec.digest(), unit,
                   snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")


def _serving_supervisor(run, monkeypatch):
    """A resumed supervisor with one ready channel whose serve() calls are recorded."""
    (run / "parallel" / "call" / "attempt-1" / "rq").mkdir(parents=True, exist_ok=True)
    supervisor = ParallelSupervisor(run)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    monkeypatch.setattr(supervisor, "_budget_expired", lambda: False)  # scheduling jitter is not under test
    served = []
    monkeypatch.setattr(supervisor, "serve", lambda rq, an: served.append(rq) or ())
    return supervisor, served


def _result(call, name):
    return json.loads((call / "host" / "seals" / name).read_text())


def test_superseded_seal_job_gets_error_result_and_channel_keeps_serving(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    first = ParallelSupervisor(run)
    monkeypatch.setattr(first, "_start_runtime", lambda: None)
    first._persist_worker(_exited_worker(call, unit, snapshot, spec_path, spec, 1))
    status, *_ = first._enqueue_seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call)
    assert status == "PENDING" and (call / "host" / "seals" / "a-1.json").exists()
    # The controller died before sealing; the client relaunched the unit, so incarnation 1 is stale.
    first._persist_worker(_exited_worker(call, unit, snapshot, spec_path, spec, 2))

    resumed, served = _serving_supervisor(run, monkeypatch)
    for _ in range(3):
        resumed._service_channels()
    assert len(served) == 3, "one unsatisfiable seal job must not stop channel service"
    result = _result(call, "a-1.result.json")
    assert result["status"] == "ERROR" and "stale" in result["payload"]["error"]
    assert (call / "host" / "seals" / "a-1.json") in resumed._sealed_jobs
    assert not resumed._seal_ops
    # The client's own seal request for the stale incarnation answers with an error, not a hang.
    request = _request(2, "call", 9, "seal", {"unit_id": "a", "incarnation": 1})
    answer = resumed._serve_request(request, call, run / "parallel" / "call" / "attempt-2" / "an", 9)
    assert answer.status == "ERROR"


def test_corrupt_seal_job_is_quarantined_without_aborting_discovery(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    seals = call / "host" / "seals"
    seals.mkdir(parents=True)
    (seals / "a-1.json").write_text("{not json")
    (seals / "a-2.json").write_text(json.dumps({"schema": "stateframe-parallel-seal-job/2", "request": {}, "operation_id": "x"}))
    supervisor, served = _serving_supervisor(run, monkeypatch)
    supervisor._service_channels()
    supervisor._service_channels()
    assert len(served) == 2
    assert _result(call, "a-1.result.json")["status"] == "ERROR"
    assert _result(call, "a-2.result.json")["status"] == "ERROR"
    assert {seals / "a-1.json", seals / "a-2.json"} <= supervisor._sealed_jobs
    assert any("a-1.json" in line for line in supervisor._protocol_errors)


def test_discovery_exception_is_logged_and_channels_are_still_served(tmp_path, monkeypatch):
    run = tmp_path / "run"
    supervisor, served = _serving_supervisor(run, monkeypatch)
    monkeypatch.setattr(supervisor, "_discover_seals", lambda: (_ for _ in ()).throw(RuntimeError("journal lost")))
    supervisor._service_channels()
    assert served, "a discovery fault must not abort serving"
    assert any("seal discovery deferred" in line and "journal lost" in line for line in supervisor._protocol_errors)


def test_seal_publication_fault_retries_without_aborting_the_pass(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor, served = _serving_supervisor(run, monkeypatch)
    supervisor._persist_worker(_exited_worker(call, unit, snapshot, spec_path, spec, 1))
    supervisor._enqueue_seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call)
    (operation,) = supervisor._seal_ops.values()
    supervisor._complete_seal(operation, {"status": "OK", "payload": {"sealed": True}})
    real_write = supervisor._write_json
    failures = [RuntimeError("journal acknowledgment missing")]

    def flaky(path, value, **kw):
        if failures and path.name.endswith(".result.json"):
            raise failures.pop()
        return real_write(path, value, **kw)
    monkeypatch.setattr(supervisor, "_write_json", flaky)
    supervisor._service_channels()
    assert served and operation.phase == "COMMITTING" and operation.job in supervisor._seal_ops
    supervisor._service_channels()
    assert operation.phase == "DONE" and not supervisor._seal_ops
    assert _result(call, "a-1.result.json") == {"status": "OK", "payload": {"sealed": True}}


def test_unavailable_journal_never_marks_a_seal_published(tmp_path, monkeypatch):
    # The host maps an exhausted journal helper to ParallelHostError; that must defer the
    # publication, not be mistaken for an immutable result that already exists.
    import sys
    from test_parallel_host import _spec_and_snapshot, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor, served = _serving_supervisor(run, monkeypatch)
    # Use the exact classes the supervisor's module raises and catches.
    host_module = sys.modules[type(supervisor).__module__]
    ParallelHostError, JournalFault = host_module.ParallelHostError, host_module.JournalFault
    supervisor._persist_worker(_exited_worker(call, unit, snapshot, spec_path, spec, 1))
    supervisor._enqueue_seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call)
    (operation,) = supervisor._seal_ops.values()
    supervisor._complete_seal(operation, {"status": "OK", "payload": {"sealed": True}})
    real_write = supervisor._write_json
    faults = [JournalFault("journal helper unavailable after 6 restarts")]

    def unavailable(path, value, **kw):
        if faults and path.name.endswith(".result.json"):
            fault = faults.pop()
            raise ParallelHostError(f"durable write unavailable: {fault}") from fault
        return real_write(path, value, **kw)
    monkeypatch.setattr(supervisor, "_write_json", unavailable)
    supervisor._service_channels()
    assert operation.phase == "COMMITTING" and operation.job in supervisor._seal_ops
    assert not operation.result.exists()
    supervisor._service_channels()
    assert operation.phase == "DONE" and not supervisor._seal_ops
    assert _result(call, "a-1.result.json") == {"status": "OK", "payload": {"sealed": True}}


def test_protocol_error_log_dedupes_and_bounds(tmp_path):
    supervisor = ParallelSupervisor(tmp_path / "run")
    log = supervisor._protocol_errors
    for _ in range(10_000):
        log.append("channel service deferred: ParallelHostError: stale")
    assert len(log) == 1 and log.total == 10_000
    assert log.repeats("channel service deferred: ParallelHostError: stale") == 10_000
    for index in range(2_000):
        log.append(f"distinct fault {index}")
    assert len(log) <= 256 and log.total == 12_000 and log.dropped == 2_001 - 256
    assert list(log)[-1] == "distinct fault 1999"
    assert log == list(log)  # list-equality contract kept for existing assertions
    assert "distinct fault 0" not in log


def test_channel_loop_persistent_fault_does_not_grow_the_log(monkeypatch):
    import weakref
    from stateframe.parallel_runtime import ProtocolErrorLog

    class Owner:
        _runtime_stop = False
        _protocol_errors = ProtocolErrorLog()
        passes = 0

        def _service_channels(self):
            self.passes += 1
            if self.passes >= 500:
                self._runtime_stop = True
            raise host.ParallelHostError("cannot seal an unfinished or stale subframe")
    owner = Owner()
    import stateframe.parallel_runtime as runtime
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    channel_loop(weakref.ref(owner))
    assert owner.passes == 500
    assert len(owner._protocol_errors) == 1 and owner._protocol_errors.total == 500


def test_host_metrics_refresh_periodically_when_ticks_advance(tmp_path, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    writes = []
    monkeypatch.setattr(supervisor, "_write_metrics", lambda path, value, **kw: writes.append(value))
    clock = [1000.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    supervisor._service_channels()
    assert len(writes) == 1
    supervisor.tick_count = 40
    for _ in range(20):
        clock[0] += 0.1  # two seconds of idle passes: nothing changed enough to rewrite
        supervisor._service_channels()
    assert len(writes) == 1
    clock[0] += host._METRICS_REFRESH_SECONDS
    supervisor._service_channels()
    assert len(writes) == 2 and writes[-1]["ticks"] == 40
    for _ in range(50):
        clock[0] += host._METRICS_REFRESH_SECONDS  # channel/lifecycle samples advance independently of tick()
        supervisor._service_channels()
    assert len(writes) == 52
    assert writes[-1]["ticks"] == 40
    assert writes[-1]["channel_pass_seconds"]["count"] == 72
