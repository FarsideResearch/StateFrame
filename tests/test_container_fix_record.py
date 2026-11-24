import json
import sys
import time

from stateframe import outcomes, outer_checks, runner
from stateframe.engine import kernel_execution_boundary


class _CapturedOuterRequest(Exception):
    def __init__(self, packet):
        self.packet = packet


def _capture_outer_request(monkeypatch, tmp_path, **kwargs):
    def capture(_command, **process_options):
        request = process_options["stdin"]
        request.seek(0)
        raise _CapturedOuterRequest(json.load(request))

    monkeypatch.setattr(outer_checks.subprocess, "Popen", capture)
    try:
        outer_checks.run("true", tmp_path, python=sys.executable, **kwargs)
    except _CapturedOuterRequest as captured:
        return captured.packet
    raise AssertionError("outer check did not publish a request")


def _container_boundary():
    return kernel_execution_boundary.Boundary(
        kernel_execution_boundary.BACKEND_CONTAINER,
        kernel_execution_boundary.Detection(
            "container", ("cgroup:docker",), {"pid": "2"},
            kernel_execution_boundary.DETECTOR_VERSION))


def _platform_boundary():
    return kernel_execution_boundary.Boundary(
        kernel_execution_boundary.BACKEND_PLATFORM,
        kernel_execution_boundary.Detection(
            "not_container", (), {"pid": "4026531836"},
            kernel_execution_boundary.DETECTOR_VERSION))


def test_summary_uses_authorized_boundary_and_detects_rewrite(tmp_path):
    authorized = _container_boundary()
    rewritten = authorized.record()
    rewritten["backend"] = kernel_execution_boundary.BACKEND_PLATFORM
    (tmp_path / "spec.json").write_text(json.dumps({"execution_boundary": rewritten}))

    summary = runner.summarize(tmp_path, time.monotonic(), execution_boundary=authorized)

    assert summary["execution_boundary"] == authorized.record()
    assert summary["candidate_error"] == (
        "DETECTED CONTAINER_RESUME_MISMATCH: run record changed")
    projected = outcomes.project({
        "engine_terminal": {"result": "SUBMITTED"}, "child_exit_code": 0,
        "candidate_error": summary["candidate_error"]})
    assert projected["runtime_outcome"] == "engine_error"


def test_summary_detects_deleted_boundary_record(tmp_path):
    authorized = _container_boundary()
    (tmp_path / "spec.json").write_text("{}")

    summary = runner.summarize(tmp_path, time.monotonic(), execution_boundary=authorized)

    assert summary["execution_boundary"] == authorized.record()
    assert summary["candidate_error"] == (
        "DETECTED CONTAINER_RESUME_MISMATCH: run record changed")


def test_outer_check_receipt_uses_explicit_boundary(monkeypatch, tmp_path):
    authorized = _container_boundary().record()
    packet = _capture_outer_request(monkeypatch, tmp_path, execution_boundary=authorized)
    assert packet["execution_boundary"] == authorized

    received = {}
    monkeypatch.setattr(outer_checks, "run", lambda *args, **kwargs: received.update(kwargs) or {})

    runner.run_test_cmd("true", tmp_path, execution_boundary=authorized)

    assert received["execution_boundary"] == authorized
    assert outer_checks._boundary_receipt(received["execution_boundary"]) == {
        "execution_boundary": authorized}


def test_default_outer_check_receipt_has_no_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_execution_boundary, "active", _platform_boundary)
    packet = _capture_outer_request(monkeypatch, tmp_path)
    assert "execution_boundary" not in packet

    received = {}
    monkeypatch.setattr(outer_checks, "run", lambda *args, **kwargs: received.update(kwargs) or {})
    runner.run_test_cmd("true", tmp_path)

    assert "execution_boundary" not in received
    assert outer_checks._boundary_receipt(None) == {}
