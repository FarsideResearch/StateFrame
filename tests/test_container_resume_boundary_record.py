import json
import time

from stateframe import outcomes, outer_checks, runner
from stateframe.engine import kernel_execution_boundary as boundary


def _container(namespaces, signals):
    return boundary.Boundary(
        boundary.BACKEND_CONTAINER,
        boundary.Detection("container", tuple(signals), namespaces, boundary.DETECTOR_VERSION),
    )


def test_resume_summary_accepts_current_container_record(tmp_path):
    recorded = _container({"pid": "100", "mnt": "200"}, ["marker:/.dockerenv"])
    current = _container({"pid": "999", "mnt": "888"}, ["cgroup:docker"])
    (tmp_path / "spec.json").write_text(
        json.dumps({"execution_boundary": recorded.record()}), encoding="utf-8")

    summary = runner.summarize(
        tmp_path, time.monotonic(), execution_boundary=current,
        accepted_execution_boundary=recorded.record())

    assert summary["execution_boundary"] == current.record()
    assert "candidate_error" not in summary
    projected = outcomes.project({
        "engine_terminal": {"result": "SUBMITTED"}, "child_exit_code": 0,
        "candidate_error": summary.get("candidate_error")})
    assert projected["runtime_outcome"] == "completed"


def test_resume_summary_detects_record_edit_after_acceptance(tmp_path):
    recorded = _container({"pid": "100", "mnt": "200"}, ["marker:/.dockerenv"])
    current = _container({"pid": "999", "mnt": "888"}, ["cgroup:docker"])
    accepted = recorded.record()
    edited = recorded.record()
    edited["namespaces"]["pid"] = "101"
    (tmp_path / "spec.json").write_text(
        json.dumps({"execution_boundary": edited}), encoding="utf-8")

    summary = runner.summarize(
        tmp_path, time.monotonic(), execution_boundary=current,
        accepted_execution_boundary=accepted)

    assert summary["execution_boundary"] == current.record()
    assert summary["candidate_error"] == (
        "DETECTED CONTAINER_RESUME_MISMATCH: run record changed")


def test_outer_check_receipt_uses_current_boundary_after_resume(tmp_path, monkeypatch):
    recorded = _container({"pid": "100", "mnt": "200"}, ["marker:/.dockerenv"])
    current = _container({"pid": "999", "mnt": "888"}, ["cgroup:docker"])
    (tmp_path / "spec.json").write_text(
        json.dumps({"execution_boundary": recorded.record()}), encoding="utf-8")
    assert boundary.check_resume(recorded.record(), current) is None
    received = {}
    monkeypatch.setattr(outer_checks, "run", lambda *args, **kwargs: received.update(kwargs) or {})

    runner.run_test_cmd("true", tmp_path, execution_boundary=current.record())

    assert received["execution_boundary"] == current.record()
    assert outer_checks._boundary_receipt(received["execution_boundary"]) == {
        "execution_boundary": current.record()}
