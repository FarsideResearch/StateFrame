"""Container wiring: child activation, launch gating, and receipts."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
import pytest
from stateframe import _child, runner, mutate_host, outer_checks
from stateframe.engine import kernel_execution_boundary as boundary
from stateframe.engine import (kernel_execution_policy, kernel_public_probe_execution,
    protected_tree_guard, kernel_test_isolation, kernel_shell_tool, kernel_mcp_tools, kernel_parallel)
from stateframe.engine.kernel_exec_capability import ExecCapability, ExecLimits
from stateframe.engine.kernel_exec_session import ExecSession, ExecSessionError
from stateframe.engine.kernel_effects import Entry

DETECTED = boundary.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1)
_MISSING = object()
def _container(): return boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED)
def _spec(record=_MISSING, **fields):
    result = {"tool_settings": {"shell": "unrestricted", "shell_network": "full"}, **fields}
    if record is _MISSING: result["execution_boundary"] = _container().record()
    elif record is not None: result["execution_boundary"] = record
    return result

def test_child_activates_only_a_strictly_validated_explicit_spec(monkeypatch):
    monkeypatch.setattr(boundary, "_active_boundary", None); monkeypatch.setattr(boundary, "detect", lambda: DETECTED)
    monkeypatch.setattr(_child.os, "geteuid", lambda: 1000)
    chosen = _child._activate_execution_boundary(_spec(), selected=True)
    assert chosen.backend == boundary.BACKEND_CONTAINER and boundary.active().record() == chosen.record()

@pytest.mark.parametrize(("selected", "record", "code"), [
    (False, _container().record(), "CONTAINER_RESUME_MISMATCH"),
    (True, None, "CONTAINER_UNSUPPORTED:engine-child-spec"),
    (True, {**_container().record(), "detector_version": 99}, "CONTAINER_RESUME_MISMATCH")])
def test_child_rejects_missing_or_invalid_authority(monkeypatch, selected, record, code):
    monkeypatch.setattr(boundary, "_active_boundary", None); monkeypatch.setattr(boundary, "detect", lambda: DETECTED)
    monkeypatch.setattr(_child.os, "geteuid", lambda: 1000)
    with pytest.raises(boundary.BoundaryRefused) as error: _child._activate_execution_boundary(_spec(record), selected=selected)
    assert error.value.code == code and boundary.active().backend == boundary.BACKEND_PLATFORM

@pytest.mark.parametrize(("field", "value", "code"), [
    ("tool_settings", {"shell_network": "none"}, "CONTAINER_NETWORK"),
    ("mutations", {"level": "prebuilt"}, "CONTAINER_MUTATIONS"),
    ("parallel_mode", "single", "CONTAINER_UNSUPPORTED:parallel-child"),
    ("yolo", True, "CONTAINER_CONFLICT"),
    ("tool_settings", {"shell_network": "full", "mcp_files": [{"path": "x", "digest": "a"}]}, "CONTAINER_UNSUPPORTED:mcp")])
def test_child_refuses_unsupported_specs_before_launch(monkeypatch, field, value, code):
    monkeypatch.setattr(boundary, "_active_boundary", None); monkeypatch.setattr(boundary, "detect", lambda: DETECTED)
    monkeypatch.setattr(_child.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(_child.subprocess, "Popen", lambda *_a, **_k: pytest.fail("raw launch"))
    with pytest.raises(boundary.BoundaryRefused) as error: _child._activate_execution_boundary(_spec(**{field: value}), selected=True)
    assert error.value.code == code

def test_child_argv_is_byte_stable_without_selection_and_spec_is_private(tmp_path):
    options = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None)
    command, _ = runner._engine_child_launch(options, tmp_path); py = Path(os.path.abspath(options.python))
    old = ("'stateframe._child.py';import os,runpy,sys;" + f"sys.path[0]={str(runner.CHILD.parent)!r};"
           + f"sys.argv=[{str(runner.CHILD)!r},{str(tmp_path)!r}];os.chdir({str(tmp_path)!r});"
           + f"runpy.run_path({str(runner.CHILD)!r},run_name='__main__')")
    assert command == [f"./{py.name}", "-B", "-c", old, "stateframe._child"]
    options._execution_boundary = _container(); assert "--container-is-sandbox" in runner._engine_child_launch(options, tmp_path)[0][3]
    run = tmp_path / "run"; run.mkdir(mode=0o700); runner._write_run_spec_atomic(run, _spec())
    spec = run / "spec.json"; assert os.stat(spec).st_mode & 0o777 == 0o600
    assert _child._read_run_spec(run, container_selected=True)["execution_boundary"]["backend"] == "container"
    moved = run / "saved.json"; spec.rename(moved); spec.symlink_to(moved.name)
    with pytest.raises(boundary.BoundaryRefused, match="CONTAINER_RESUME_MISMATCH"): _child._read_run_spec(run, container_selected=True)

@pytest.mark.parametrize(("change", "code"), [("mcp", "CONTAINER_UNSUPPORTED:mcp"),
    ("parallel", "CONTAINER_UNSUPPORTED:parallel-child"), ("mutation", "CONTAINER_MUTATIONS"), ("network", "CONTAINER_NETWORK")])
def test_runner_refuses_unsupported_sites_before_prepare(monkeypatch, tmp_path, change, code):
    opt = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None); opt._execution_boundary = _container()
    if change == "mcp": opt.tool_settings = runner.ToolSettings(shell_network="full", mcp_files=({"path": "/tmp/mcp.json", "digest": "sha256:x"},))
    elif change == "parallel": opt.parallel_mode = "single"
    elif change == "mutation": opt.mutations = "prebuilt"
    else: opt.tool_settings = runner.ToolSettings(shell_network="none")
    monkeypatch.setattr(runner, "_run", lambda *_a, **_k: pytest.fail("prepare launched"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("raw launch"))
    with pytest.raises(runner.RunError, match=code.replace(":", r":")): runner.run(opt, printer=lambda *_: None)

@pytest.mark.parametrize("network", ["none", "loopback"])
def test_command_boundary_refuses_network_and_mutations(monkeypatch, tmp_path, network):
    monkeypatch.setattr(boundary, "_active_boundary", _container())
    monkeypatch.setattr(kernel_test_isolation, "sandbox_executable", lambda *_a: pytest.fail("bwrap lookup"))
    cap = ExecCapability("shell", tmp_path, False, (), (), network, {"PATH": "/usr/bin:/bin"}, ExecLimits())
    with pytest.raises(boundary.BoundaryRefused, match="CONTAINER_NETWORK"): kernel_test_isolation.capability_command(["true"], cap)
    mutation = ExecCapability("mutation", tmp_path, False, (), (), "none", cap.env, ExecLimits())
    with pytest.raises(boundary.BoundaryRefused, match="CONTAINER_UNSUPPORTED:mutations"):
        kernel_test_isolation.capability_command(["true"], mutation, mutation_mounts={})

def test_unsupported_launch_sites_refuse_before_raw_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(boundary, "_active_boundary", _container())
    session = object.__new__(ExecSession); session._handles = []
    monkeypatch.setattr(session, "_spawn", lambda *_a, **_k: pytest.fail("job launched"))
    cap = ExecCapability("job", tmp_path, False, (), (), "full", {"PATH": "/usr/bin:/bin"}, ExecLimits())
    with pytest.raises(ExecSessionError, match="CONTAINER_UNSUPPORTED:persistent-jobs"): session.spawn(["true"], cap)
    monkeypatch.setattr(kernel_mcp_tools, "exchange", lambda *_a, **_k: pytest.fail("MCP launched"))
    with pytest.raises(kernel_mcp_tools.McpError, match="CONTAINER_UNSUPPORTED:mcp"):
        kernel_mcp_tools.discover((), tmp_path, tmp_path, shell_network="full", host_authorization=False)
    with pytest.raises(RuntimeError, match="CONTAINER_UNSUPPORTED:mutations"): mutate_host.start({"mutations": {"level": "prebuilt"}}, tmp_path)
    with pytest.raises(kernel_parallel.ParallelError, match="CONTAINER_UNSUPPORTED:parallel-child"): kernel_parallel.install(tmp_path, {"parallel_mode": "single"})
    monkeypatch.setattr(protected_tree_guard.subprocess, "call", lambda *_a, **_k: pytest.fail("guard launched"))
    with pytest.raises(RuntimeError, match="CONTAINER_UNSUPPORTED:protected_tree_guard"): protected_tree_guard.guarded_command(["true"], roots=[tmp_path])

def test_public_probe_refuses_and_runner_probe_uses_scrubbed_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(boundary, "_active_boundary", _container())
    report = kernel_execution_policy.capability_report()
    assert report.backend == "no sandbox binary: container"
    from stateframe.engine.kernel_information_policy import InformationPolicy
    policy = InformationPolicy({})
    policy.isolation_backend = kernel_test_isolation.backend()
    manifest = policy.manifest()
    assert manifest["test_isolation_binary_sha256"] is None
    assert manifest["artifact_boundary"] == "CURRENT_CONTAINER_UNVERIFIED"
    monkeypatch.setattr(kernel_execution_policy, "sandboxed_popen", lambda *_a, **_k: pytest.fail("probe launched"))
    root, output = tmp_path / "repo", tmp_path / "out"; root.mkdir(); output.mkdir()
    source = root / "app.py"; source.write_text("pass\n")
    files = {"app.py": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match="CONTAINER_UNSUPPORTED:public-probe"):
        kernel_public_probe_execution.execute_probe(root, files, "def test_probe(): pass\n", output)
    monkeypatch.setenv("PROVIDER_KEY", "secret-value")
    payload = {name: {"enabled": True, "detail": "available"} for name in ("schema_validation", "yaml", "repo_map_ast")}
    calls = []
    def run(command, root, **kwargs):
        calls.append(kwargs); return SimpleNamespace(completed=SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""), timed_out=False)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", run)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("raw interpreter probe"))
    opt = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None); opt._execution_boundary = _container()
    assert runner._probe_capabilities(opt)["schema_validation"]["enabled"] and "PROVIDER_KEY" not in calls[0]["env"]

def test_protected_tests_are_detected_after_shell_and_check_commands(tmp_path):
    old, new = {"tests/t.py": Entry("file", 0o644, "old", 3)}, {"tests/t.py": Entry("file", 0o644, "new", 3)}
    changed = kernel_shell_tool._container_tampered_protected_tests(old, new, {"tests/t.py"})
    assert changed == ("tests/t.py",) and "DETECTED PROTECTED_TEST_TAMPERED" in kernel_shell_tool._container_test_tamper_notice(changed)
    target = tmp_path / "tests" / "t.py"; target.parent.mkdir(); target.write_text("new")
    digest = hashlib.sha256(b"old").hexdigest()
    assert outer_checks._protected_test_tampering(tmp_path, {"tests/t.py": digest}) == ["tests/t.py"]
