import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_execution_boundary as child_boundary


def selected(module):
    return module.Boundary(
        module.BACKEND_CONTAINER,
        module.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1),
    )


def test_inner_pytest_marks_changed_protected_file_as_tampered(monkeypatch):
    import kernel_check_truth_adapter as truth
    import kernel_execution_policy as policy
    import kernel_test_artifact_policy as artifacts
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair

    monkeypatch.setattr(child_boundary, "_active_boundary", selected(child_boundary))
    monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setattr(truth, "begin_check_execution", lambda *args, **kwargs: None)
    selector = "tests/test_pricing.py"
    snapshots = iter({selector: "sha256:" + hashlib.sha256(content).hexdigest()}
                     for content in (b"before", b"after"))
    monkeypatch.setattr(artifacts, "snapshot_workspace", lambda root: next(snapshots))
    monkeypatch.setattr(policy, "run_sandboxed", lambda argv, *_args, **_kwargs:
                        SimpleNamespace(completed=subprocess.CompletedProcess(argv, 0, "1 passed\n", ""),
                                        timed_out=False))
    records = []
    runtime = SimpleNamespace(binding=lambda: {}, test_observation=lambda *args, **kwargs:
                              records.append((args, kwargs)))
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        reply = react_repair.run_tests(Path("examples/discount_bug").resolve(), "", selector,
                                      row={"runner_kind": "pytest", "protect_tests": "all",
                                           "protected_test_paths": [selector]})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert records[0][1]["run_write_set"] == (selector,)
    assert "CONTAINER TEST TAMPERED" in records[0][1]["observation"]
    assert "status=INVALID_CHECK" in reply and "assertion_context='CONTAINER TEST TAMPERED" in reply
    assert "tampered" in reply.lower() and "ALL TESTS PASSED" not in reply, reply


def test_inner_pytest_keeps_pass_in_default_boundary_mode(monkeypatch):
    import kernel_check_truth_adapter as truth
    import kernel_execution_policy as policy
    import kernel_test_artifact_policy as artifacts
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair

    import stateframe.engine.kernel_execution_boundary as package_boundary

    monkeypatch.setattr(package_boundary, "_active_boundary", None)
    monkeypatch.setattr(child_boundary, "_active_boundary", None)
    assert child_boundary.active().backend == child_boundary.BACKEND_PLATFORM
    monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setattr(truth, "begin_check_execution", lambda *args, **kwargs: None)
    selector = "tests/test_pricing.py"
    snapshots = iter({selector: "sha256:" + hashlib.sha256(content).hexdigest()}
                     for content in (b"before", b"after"))
    monkeypatch.setattr(artifacts, "snapshot_workspace", lambda root: next(snapshots))
    monkeypatch.setattr(policy, "run_sandboxed", lambda argv, *_args, **_kwargs:
                        SimpleNamespace(completed=subprocess.CompletedProcess(argv, 0, "1 passed\n", ""),
                                        timed_out=False))
    records = []
    runtime = SimpleNamespace(binding=lambda: {}, test_observation=lambda *args, **kwargs:
                              records.append((args, kwargs)))
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        reply = react_repair.run_tests(Path("examples/discount_bug").resolve(), "", selector,
                                      row={"runner_kind": "pytest", "protect_tests": "all",
                                           "protected_test_paths": [selector]})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert records[0][1]["observation"] == "1 passed\n\n"
    assert "status=PASS" in reply and "ALL TESTS PASSED" in reply
    assert "TAMPERED" not in reply
