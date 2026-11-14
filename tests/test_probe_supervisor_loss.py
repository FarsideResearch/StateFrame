"""Bounded recovery from a host-side supervisor kill, without losing evidence."""
from dataclasses import replace
import base64
import errno
import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

import kernel_exec_session as execution
import kernel_safe_probe as probes
import kernel_ship_tree_smoke as smoke
from kernel_post_edit_checks import Finding, ImportStatus, PostEditTracker, import_status
from test_probe_startup_diagnostics import _linux_supervisor_pid


_WAIT_FOR_RELEASE = (
    "import os, time\nfrom pathlib import Path\nprint('ready', flush=True)\n"
    "while not Path(os.environ['PROBE_GATE']).exists():\n    time.sleep(.01)\n"
)


def _result(**changes):
    return replace(execution.ExecResult(0, False, b"ok", False, 2, 0,
                                      execution.CleanupReport(True, 0, (), "")), **changes)


def _invoke(lane, root):
    if lane == "probe":
        return probes.run_probe([sys.executable, "-I", "-B", "-c",
                                 _WAIT_FOR_RELEASE], root)
    (root / "app.py").write_text(_WAIT_FOR_RELEASE)
    if lane == "smoke":
        return smoke.run_smoke(root, ("app.py",), roots=(str(root),))
    return import_status(root, root / "app.py")


@pytest.mark.host_sandbox
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="host PID namespace injection")
@pytest.mark.parametrize("lane", ["probe", "smoke", "post_edit"])
@pytest.mark.parametrize("losses", [1, 2])
def test_host_sigkill_is_retried_once_with_invalid_attempt_evidence(
        monkeypatch, tmp_path, lane, losses):
    sessions = []
    killed = []
    real_kill = execution.kill_launch_unit
    verified = []

    def checked_cleanup(process, **kwargs):
        survivors = real_kill(process, **kwargs)
        verified.append((process.pid, survivors))
        return survivors

    class Session(execution.ExecSession):
        def __init__(self, directory):
            super().__init__(directory)
            sessions.append(self)

        def run(self, argv, cap, **kwargs):
            gate = tmp_path / f"release-{len(sessions)}"
            if len(sessions) > losses:
                gate.touch()
            cap = replace(cap, env={**cap.env, "PROBE_GATE": str(gate)})
            return super().run(argv, cap, **kwargs)

        def _rpc(self, request):
            result = super()._rpc(request)
            if (request["op"] == "status" and len(killed) < losses
                    and self not in killed and b"ready" in base64.b64decode(result["output"])):
                assert not result["finished"] and not result["timed_out"]
                pid = _linux_supervisor_pid(self._process.pid)
                # This call runs on the host, after the contained code reports readiness.
                os.kill(pid, signal.SIGKILL)
                self._process.wait(timeout=5)
                assert self._process.returncode == 137
                killed.append(self)
            return result

    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", sys.executable)
    monkeypatch.setattr(execution, "kill_launch_unit", checked_cleanup)
    monkeypatch.setattr(execution, "ExecSession", Session)
    monkeypatch.setattr(probes, "ExecSession", Session)
    if losses == 2 and lane == "probe":
        with pytest.raises(execution.ExecSessionError) as caught:
            _invoke(lane, tmp_path)
        result = caught.value
        assert result.outcome == "ENGINE_ERROR"
        evidence = probes.probe_evidence(result)
        for failure in evidence["failures"]:
            assert failure["error"] in str(result)
    else:
        result = _invoke(lane, tmp_path)
        if lane == "probe":
            assert result.exit_code == 0 and not result.timed_out
            evidence = probes.probe_evidence(result)
        else:
            evidence = result.probe_evidence
            if losses == 2:
                assert (result.status if lane == "smoke" else result.outcome) == "ENGINE_ERROR"
                if lane == "post_edit":
                    assert result.finding("app.py") is None
                else:
                    assert result.blocking
                    assert result.as_dict()["probe_evidence"] == evidence
                    baseline = replace(result, status="PASS", probe_evidence=None)
                    assert smoke.apply_baseline(result, baseline) is result
                    for failure in evidence["failures"]:
                        assert failure["error"] in result.environment[0][1]
            elif lane == "smoke":
                assert result.status == "PASS"
                assert result.as_dict()["probe_evidence"] == evidence
            else:
                assert not result.failed
    assert len(sessions) == 2
    assert len(killed) == losses
    assert sessions[0]._directory != sessions[1]._directory
    assert len(verified) == losses and all(not survivors for _, survivors in verified)
    assert evidence["attempt_count"] == 2
    assert len(evidence["failures"]) == losses
    for index, failure in enumerate(evidence["failures"], 1):
        assert failure["attempt"] == index
        assert failure["classification"] == "PROBE_SUPERVISOR_LOST"
        assert failure["status"] == "INVALID"
        assert "execution supervisor" in failure["error"]
        assert "137" in failure["error"]
    assert all(session.containment_verified for session in sessions)


@pytest.mark.parametrize("lane", ["probe", "smoke"])
@pytest.mark.parametrize("outcome", ["timeout", "limit", "output_limit", "nonzero"])
def test_completed_code_outcomes_are_never_retried(monkeypatch, tmp_path, lane, outcome):
    calls = []
    completed = _result(**{
        "timeout": {"timed_out": True, "exit_code": -9},
        "limit": {"exit_code": -signal.SIGXCPU},
        "output_limit": {"output_truncated": True},
        "nonzero": {"exit_code": 23},
    }[outcome])

    class Session:
        def __init__(self, directory):
            calls.append(directory)

        def run(self, *args, **kwargs):
            return completed

        def close(self):
            pass

    monkeypatch.setattr(execution, "ExecSession", Session)
    monkeypatch.setattr(probes, "ExecSession", Session)
    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", sys.executable)
    if lane == "probe":
        result = probes.run_probe([sys.executable, "-c", "pass"], tmp_path)
    else:
        result = smoke._execute_smoke(tmp_path, (), (), 1)
    assert result is completed
    assert len(calls) == 1


@pytest.mark.parametrize("lane", ["probe", "smoke"])
@pytest.mark.parametrize("prior_cleanup", [False, True])
def test_primary_error_survives_cleanup_error_without_retry(monkeypatch, tmp_path, lane, prior_cleanup):
    original = execution.ExecSessionError("original execution failure")
    previous = execution.ExecSessionError("earlier stop failure") if prior_cleanup else None
    original.__cause__ = previous
    cleanup = execution.ExecSessionError("containment unverified")
    calls = []

    class Session:
        containment_verified = False

        def __init__(self, directory):
            calls.append(directory)

        def run(self, *args, **kwargs):
            raise original

        def close(self):
            raise cleanup

    monkeypatch.setattr(execution, "ExecSession", Session)
    monkeypatch.setattr(probes, "ExecSession", Session)
    with pytest.raises(execution.ExecSessionError) as caught:
        if lane == "probe":
            probes.run_probe([sys.executable, "-c", "pass"], tmp_path)
        else:
            smoke._execute_smoke(tmp_path, (), (), 1)
    assert caught.value is original
    assert caught.value.__cause__ is cleanup or caught.value.__context__ is cleanup
    assert caught.value.outcome == "ENGINE_ERROR"
    if previous is not None:
        assert cleanup.__context__ is previous
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "limit", "launch", "unverified", "observed_timeout"])
def test_session_errors_require_loss_and_verified_containment(monkeypatch, tmp_path, failure):
    calls = []
    error = execution.ExecSessionError(failure)
    error.supervisor_lost = failure in {"unverified", "observed_timeout"}
    error.probe_retry_forbidden = failure == "observed_timeout"

    class Session:
        containment_verified = failure != "unverified"

        def __init__(self, directory):
            calls.append(directory)

        def run(self, *args, **kwargs):
            raise error

        def close(self):
            pass

    monkeypatch.setattr(probes, "ExecSession", Session)
    with pytest.raises(execution.ExecSessionError) as caught:
        probes.run_probe([sys.executable, "-c", "pass"], tmp_path)
    assert caught.value is error and len(calls) == 1


def _interrupt_probe_after_observation(monkeypatch, outcome, prior_loss):
    """Exercise real collection, cleanup and retry with a deterministic lost channel."""
    sessions, errors = [], []
    collect = execution.ExecSession._collect

    class Session:
        containment_verified = False

        def __init__(self, directory):
            sessions.append(self)
            self.first_loss = prior_loss and len(sessions) == 1

        def run(self, *args, **kwargs):
            error = execution.ExecSessionError(
                "execution supervisor lost " + ("before observation" if self.first_loss
                                                else f"after {outcome}"))
            error.supervisor_lost = True
            errors.append(error)
            statuses = []

            class Handle:
                def _status(self, consume):
                    if statuses or sessions[-1].first_loss:
                        raise error
                    statuses.append(consume)
                    return {
                        "timed_out": outcome == "timeout", "finished": False,
                        "output": base64.b64encode(b"x" * 65 if outcome == "output_limit"
                                                   else b""),
                        "lost_bytes": 0, "probe_retry_forbidden": outcome == "pending_limit",
                        "exit_code": {"nonzero": 23, "resource_limit": -signal.SIGXCPU}.get(outcome),
                    }

                def stop(self, grace):
                    pass

            return collect(
                self, Handle(), SimpleNamespace(limits=SimpleNamespace(output_bytes=64)))

        def close(self):
            self.containment_verified = self.first_loss or outcome not in {"unverified", "cleanup_error"}
            if outcome == "cleanup_error" and not self.first_loss:
                raise execution.ExecSessionError("launch-unit cleanup unproven")

    monkeypatch.setattr(execution, "ExecSession", Session)
    monkeypatch.setattr(probes, "ExecSession", Session)
    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", sys.executable)
    return sessions, errors


_INTERRUPTED_OUTCOMES = ["timeout", "nonzero", "resource_limit", "output_limit",
                         "pending_limit", "unverified", "cleanup_error"]


@pytest.mark.parametrize("outcome", _INTERRUPTED_OUTCOMES)
@pytest.mark.parametrize("prior_loss", [False, True])
def test_unretryable_supervisor_loss_blocks_real_smoke_done_gate(
        monkeypatch, tmp_path, outcome, prior_loss):
    from kernel_stateframe_runtime import StateFrameRuntime

    (tmp_path / "app.py").write_text("VALUE = 1\n")
    sessions, errors = _interrupt_probe_after_observation(monkeypatch, outcome, prior_loss)
    results = []

    def run_smoke():
        result = smoke.run_smoke(tmp_path, ("app.py",), roots=(str(tmp_path),))
        results.append(result)
        return result

    runtime = SimpleNamespace(stateframe_ship_tree_smoke_mode="on", ship_tree_smoke=run_smoke)
    allowed, detail = StateFrameRuntime.ship_tree_done_gate(runtime)
    result, = results
    assert (result.status, result.blocking, allowed) == ("ENGINE_ERROR", True, False)
    evidence = result.probe_evidence
    assert evidence == {
        "attempt_count": 1 + prior_loss,
        "failures": [execution.ProbeFailure(number, "PROBE_SUPERVISOR_LOST", str(error)).as_dict()
                     for number, error in enumerate(errors, 1)],
    }
    assert len(sessions) == 1 + prior_loss
    assert len(evidence["failures"]) == len(sessions)
    assert errors[-1].probe_retry_forbidden is (outcome not in {"unverified", "cleanup_error"})
    assert result.as_dict()["probe_evidence"] == evidence
    assert "ENGINE_ERROR" in detail and str(errors[-1]) in detail
    baseline = replace(result, status="PASS", probe_evidence=None)
    assert smoke.apply_baseline(result, baseline) is result


@pytest.mark.parametrize("outcome", _INTERRUPTED_OUTCOMES)
@pytest.mark.parametrize("prior_loss", [False, True])
def test_unretryable_supervisor_loss_is_retained_in_post_edit_receipt(
        monkeypatch, tmp_path, outcome, prior_loss):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n")
    sessions, errors = _interrupt_probe_after_observation(monkeypatch, outcome, prior_loss)
    receipt = PostEditTracker(tmp_path).evaluate(
        path, "VALUE = 1\n", "VALUE = 1\n", baseline_import=ImportStatus("app", False))
    evidence = receipt["import_probe"]["candidate"]
    assert evidence["outcome"] == "ENGINE_ERROR"
    assert evidence["attempt_count"] == 1 + prior_loss
    assert evidence["failures"] == [
        execution.ProbeFailure(number, "PROBE_SUPERVISOR_LOST", str(error)).as_dict()
        for number, error in enumerate(errors, 1)]
    assert str(errors[-1]) in evidence["error"]
    assert len(sessions) == 1 + prior_loss
    assert receipt["introduced"] == [] and receipt["resolved"] == []


def test_post_edit_receipt_retains_recovered_invalid_attempt(monkeypatch, tmp_path):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n")
    failure = execution.ProbeFailure(1, "PROBE_SUPERVISOR_LOST", "original broken pipe")
    monkeypatch.setattr(probes, "run_probe", lambda *a, **kw: _result(probe_failures=(failure,)))
    result = PostEditTracker(tmp_path).evaluate(
        path, "VALUE = 1\n", "VALUE = 1\n", baseline_import=ImportStatus("app", False))
    evidence = result["import_probe"]["candidate"]
    assert evidence["attempt_count"] == 2
    assert evidence["failures"] == [failure.as_dict()]


def test_post_edit_receipt_keeps_engine_error_out_of_code_findings(monkeypatch, tmp_path):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n")
    failures = tuple(execution.ProbeFailure(number, "PROBE_SUPERVISOR_LOST", f"loss {number}")
                     for number in (1, 2))

    def lost(*args, **kwargs):
        raise probes.ProbeEngineError(failures, execution.ExecSessionError("loss 2"))

    monkeypatch.setattr(probes, "run_probe", lost)
    result = PostEditTracker(tmp_path).evaluate(
        path, "VALUE = 1\n", "VALUE = 1\n", baseline_import=ImportStatus("app", False))
    evidence = result["import_probe"]["candidate"]
    assert evidence["outcome"] == "ENGINE_ERROR" and evidence["attempt_count"] == 2
    assert evidence["failures"] == [failure.as_dict() for failure in failures]
    assert "loss 1" in evidence["error"] and "loss 2" in evidence["error"]
    assert result["introduced"] == []


def test_unverified_import_cannot_resolve_an_existing_failure(monkeypatch, tmp_path):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.setattr("kernel_post_edit_checks.import_status", lambda *a, **kw:
                        ImportStatus("app", False, "lost", outcome="ENGINE_ERROR"))
    tracker = PostEditTracker(tmp_path)
    finding = Finding("app.py", 1, "import_fails", "app")
    tracker.active.add(finding)
    receipt = tracker.evaluate(path, "VALUE = 1\n", "VALUE = 1\n",
                               baseline_import=ImportStatus("app", True, "existing failure"))
    assert receipt["resolved"] == []
    assert finding in tracker.active and finding not in tracker.resolved_emitted
    assert receipt["import_probe"]["candidate"]["outcome"] == "ENGINE_ERROR"


def test_unverified_baseline_cannot_attribute_an_import_failure_to_the_edit(monkeypatch, tmp_path):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.setattr("kernel_post_edit_checks.import_status", lambda *a, **kw:
                        ImportStatus("app", True, "ImportError: broken dependency"))
    receipt = PostEditTracker(tmp_path).evaluate(
        path, "VALUE = 1\n", "VALUE = 1\n",
        baseline_import=ImportStatus("app", False, "lost", outcome="ENGINE_ERROR"))
    assert receipt["introduced"] == []
    assert receipt["import_probe"]["baseline"]["outcome"] == "ENGINE_ERROR"
    assert receipt["import_probe"]["candidate"]["outcome"] == "FAIL"
    assert receipt["import_probe"]["candidate"]["error"] == "ImportError: broken dependency"


@pytest.mark.parametrize("socket_failure,expected", [(TimeoutError(), False),
                                                    (BrokenPipeError(errno.EPIPE, "lost"), True)])
def test_rpc_loss_classification_is_structural(tmp_path, socket_failure, expected):
    session = execution.ExecSession(tmp_path)

    class Control:
        def sendall(self, payload):
            raise socket_failure

    session._control = Control()
    with pytest.raises(execution.ExecSessionError) as caught:
        session._rpc({"op": "status"})
    assert caught.value.supervisor_lost is expected


def test_collection_preserves_loss_when_stop_also_fails(tmp_path):
    session = execution.ExecSession(tmp_path)
    original = execution.ExecSessionError("lost during collection")
    cleanup = execution.ExecSessionError("stop channel lost")

    class Handle:
        def _status(self, consume):
            raise original

        def stop(self, grace):
            raise cleanup

    cap = SimpleNamespace(limits=SimpleNamespace(output_bytes=64))
    with pytest.raises(execution.ExecSessionError) as caught:
        session._collect(Handle(), cap)
    assert caught.value is original
    assert caught.value.__cause__ is cleanup or caught.value.__context__ is cleanup


@pytest.mark.parametrize("limited", ["timed_out", "output", "nonzero", "resource_limit", "pending_limit"])
def test_collection_remembers_terminal_outcomes_before_channel_loss(tmp_path, limited):
    session = execution.ExecSession(tmp_path)
    original = execution.ExecSessionError("lost after a limit")
    original.supervisor_lost = True
    calls = []

    class Handle:
        def _status(self, consume):
            calls.append(consume)
            if len(calls) == 2:
                raise original
            return {"timed_out": limited == "timed_out", "finished": False,
                    "output": base64.b64encode(b"x" * 65 if limited == "output" else b""),
                    "lost_bytes": 0,
                    "probe_retry_forbidden": limited == "pending_limit",
                    "exit_code": {"nonzero": 23, "resource_limit": -signal.SIGXCPU}.get(limited)}

        def stop(self, grace):
            pass

    cap = SimpleNamespace(limits=SimpleNamespace(output_bytes=64))
    with pytest.raises(execution.ExecSessionError) as caught:
        session._collect(Handle(), cap)
    assert caught.value is original
    assert original.probe_retry_forbidden is True


@pytest.mark.parametrize("purpose", ["probe", "shell"])
@pytest.mark.parametrize("failure", ["limit", "nonzero", "ok"])
def test_supervisor_marks_pending_probe_failures_only(purpose, failure):
    from kernel_exec_supervisor import ManagedProcess

    job = ManagedProcess.__new__(ManagedProcess)
    job.cap = SimpleNamespace(purpose=purpose, limits=SimpleNamespace(kill_on_output=False))
    job.collector = SimpleNamespace(total_bytes=0)
    job.cleanup = None
    job.timed_out = False
    job.timeout_at = None
    job.memory_limit_mechanism = None
    job.ended = None
    job.started = time.monotonic()
    job.exit_code = 23 if failure == "nonzero" else None
    job.limit_detail = "resident memory limit" if failure == "limit" else ""
    status = job.status()
    assert status["finished"] is False and status["exit_code"] is None
    if purpose == "probe" and failure != "ok":
        assert status["probe_retry_forbidden"] is True
    else:
        assert "probe_retry_forbidden" not in status


@pytest.mark.parametrize("survivors", [(), (123,)])
def test_close_exposes_positive_containment_proof_only(monkeypatch, survivors):
    from test_exec_cleanup_evidence import _session

    session = _session(SimpleNamespace(pid=123), SimpleNamespace(close=lambda: None))
    assert not session.containment_verified

    def lost_control(*args, **kwargs):
        raise execution.ExecSessionError("broken channel")

    session.stop_all = lost_control
    monkeypatch.setattr(execution, "_await_supervisor_progress", lambda process: None)
    monkeypatch.setattr(execution, "kill_launch_unit", lambda *a, **kw: survivors)
    if survivors:
        with pytest.raises(execution.ExecSessionError, match="survivors 123"):
            session.close()
    else:
        session.close()
    assert session.containment_verified is (not survivors)
