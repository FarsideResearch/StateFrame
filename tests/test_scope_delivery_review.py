"""Exercise scope and candidate delivery across the host and worker handoffs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from stateframe import outcomes, runner, workcopy

from test_delivery_refusal_not_submitted import _execute_prepared, _prepare_execution


def _checked_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _execute_real_check(command: str, cwd: Path, **_kwargs) -> dict[str, object]:
    result = subprocess.run(shlex.split(command), cwd=cwd, capture_output=True, text=True,
                            timeout=30, check=False)
    output = result.stdout + result.stderr
    return {"command": command, "exit_code": result.returncode,
            "passed": result.returncode == 0, "tail": output[-4000:], "output": output}


@pytest.mark.parametrize("protection", ["off", "all"])
@pytest.mark.parametrize(("operation", "expected_paths"), [
    ("edit", ["outside.py"]),
    ("create", ["created.py"]),
    ("delete", ["outside.py"]),
    ("rename", ["outside.py", "renamed.py"]),
])
def test_check_command_out_of_scope_effects_are_not_delivered(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protection: str,
        operation: str, expected_paths: list[str]) -> None:
    action = {
        "edit": "Path('outside.py').write_text('outside = 2\\n')",
        "create": "Path('created.py').write_text('created = True\\n')",
        "delete": "Path('outside.py').unlink()",
        "rename": "Path('outside.py').rename('renamed.py')",
    }[operation]
    code = ("from pathlib import Path; "
            f"(lambda: {action})() if Path('sample.py').read_text() == 'value = 2\\n' else None")
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    options.test_cmd = _checked_command(code)
    options.protect_tests = protection
    spec["protect_tests"] = protection
    monkeypatch.setattr(runner, "run_test_cmd", _execute_real_check)
    digest = runner.candidate_digest
    sealed_digest: list[str] = []

    def preserve_sealed_digest(root: Path, **kwargs) -> str:
        if not sealed_digest:
            sealed_digest.append(digest(root, **kwargs))
        else:
            digest(root, **kwargs)
        return sealed_digest[0]

    monkeypatch.setattr(runner, "candidate_digest", preserve_sealed_digest)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == expected_paths
    assert summary["candidate_files_written"] == ["sample.py"]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()
    assert not (repo / "renamed.py").exists()


def test_retry_workcopy_drift_outside_scope_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": "sha256:" + "a" * 64,
        "artifact_digest": "sha256:" + "a" * 64,
    }}

    def drift_during_retry(run_dir: Path, *_args) -> int:
        (work.copy / "outside.py").write_bytes(b"outside = retry drift\n")
        (run_dir / "engine-result.json").write_text(json.dumps({"result": terminal}))
        return 0

    monkeypatch.setattr(runner, "drive", drift_during_retry)
    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == ["outside.py"]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"


@pytest.mark.parametrize("phase", ["during_run", "at_handoff"])
def test_original_outside_scope_source_drift_is_preserved_and_not_delivered(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    candidate = {"sample.py": b"value = 2\n", "outside.py": b"outside = 1\n"}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task="Only edit sample.py.")
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": "sha256:" + "a" * 64,
        "artifact_digest": "sha256:" + "a" * 64,
    }}
    drift = b"outside = changed by another process\n"

    if phase == "during_run":
        def drift_while_running(run_dir: Path, *_args) -> int:
            (repo / "outside.py").write_bytes(drift)
            (run_dir / "engine-result.json").write_text(json.dumps({"result": terminal}))
            return 0

        monkeypatch.setattr(runner, "drive", drift_while_running)
    else:
        project = outcomes.project
        injected = False

        def drift_at_handoff(*args, **kwargs):
            nonlocal injected
            result = project(*args, **kwargs)
            if not injected:
                (repo / "outside.py").write_bytes(drift)
                injected = True
            return result

        monkeypatch.setattr(outcomes, "project", drift_at_handoff)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["sample.py"]
    assert summary["applied_in_place"] == ["sample.py"]
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert (repo / "outside.py").read_bytes() == drift


@pytest.mark.parametrize("task", ["Only edit sample.py.", "Update the sample."])
def test_late_edit_after_check_is_not_promoted_as_the_sealed_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task=task)
    apply = workcopy.apply_in_place

    def edit_after_promotion_scan(target: workcopy.WorkCopy, **kwargs):
        (work.copy / "sample.py").write_bytes(b"value = 3\n")
        return apply(target, **kwargs)

    monkeypatch.setattr(workcopy, "apply_in_place", edit_after_promotion_scan)

    summary = _execute_prepared(options, run, work, spec)

    assert (repo / "sample.py").read_bytes() == b"value = 1\n", (
        "apply_in_place promoted bytes added after the host-approved promotion scan")
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["apply_error"] == "candidate changed after host verification: sample.py"
    assert summary.get("applied_in_place", []) == []
    assert summary["exit_code"] == 2


@pytest.mark.parametrize("alias", [
    "./outside.py", "outside.PY", "pkg/../outside.py", "newdir/outside.py",
])
def test_noncanonical_or_case_alias_candidate_paths_cannot_expand_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", alias: b"outside = 2\n"},
        task="Only edit sample.py.")

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert alias in summary["candidate_paths_dropped"]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"


def test_untracked_symlink_and_empty_directory_in_retry_copy_are_not_promoted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    (work.copy / "empty-outside-dir").mkdir()
    (work.copy / "outside-link").symlink_to(repo / "outside.py")

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["sample.py"]
    assert summary["applied_in_place"] == ["sample.py"]
    assert not (repo / "empty-outside-dir").exists()
    assert not (repo / "outside-link").exists()
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"


def test_nested_worker_projection_preserves_outer_task_scope(tmp_path: Path) -> None:
    from stateframe.parallel_worker import prepare_worker
    from test_parallel_worker_prep import _unit_fixture

    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path / "first")
    parent_spec["public_row"]["task_file_scope"] = ["pkg/a.py"]
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    first_worker = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))

    nested_snapshot, nested_run, nested_spec_path, nested_spec, nested_parent, _ = _unit_fixture(
        tmp_path / "nested")
    nested_parent["public_row"] = first_worker["public_row"]
    prepare_worker(nested_spec, nested_spec_path, nested_run, nested_snapshot,
                   parent_spec=nested_parent)

    second_worker = json.loads((nested_run / "spec.json").read_text(encoding="utf-8"))
    assert second_worker["public_row"]["task_file_scope"] == ["pkg/a.py"]


def test_apply_in_place_refusal_does_not_leak_directory_descriptors(tmp_path: Path) -> None:
    original = tmp_path / "repo"
    original.mkdir()
    (original / "source.py").write_bytes(b"value = 1\n")
    work = workcopy.create(original, tmp_path / "work")
    external = tmp_path / "external"
    external.mkdir()
    (original / "blocked").symlink_to(external, target_is_directory=True)
    (work.copy / "blocked").mkdir()
    (work.copy / "blocked" / "new.py").write_bytes(b"value = 2\n")
    approved = workcopy.changed_files(work)
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("descriptor count is unavailable on this host")
    before = len(list(fd_dir.iterdir()))

    for _ in range(25):
        with pytest.raises(workcopy.WorkCopyError):
            workcopy.apply_in_place(work, changed=approved)

    assert len(list(fd_dir.iterdir())) == before


def test_scoped_serial_candidate_delivers_only_the_named_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from stateframe import cli
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "outside.py").write_text("OUTSIDE = 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    script = Script({"steps": [
        {"tool": "str_replace", "arguments": {
            "path": "target.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}},
        {"tool": "done", "arguments": {"summary": "Updated target.py."}},
    ]})
    server = serve(script)
    try:
        exit_code = cli.main([
            "run", "--repo", str(repo), "--task",
            "Set target.py to VALUE = 2. Only edit target.py.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--profile", "r2hc3", "--run-dir", str(run_dir), "--max-turns", "4",
            "--parallel", "off", "--python", sys.executable, "--test-python", sys.executable,
            "--shell", "off", "--check-cmd", f"{shlex.quote(sys.executable)} -c pass",
            "--target", "target.py", "--protect-tests", "off", "--in-place", "--quiet", "--json",
        ])
    finally:
        server.shutdown()
        server.server_close()
    script.assert_consumed()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0, capsys.readouterr().out
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["target.py"]
    assert summary["applied_in_place"] == ["target.py"]
    assert (repo / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "outside.py").read_text(encoding="utf-8") == "OUTSIDE = 1\n"


def test_scoped_parallel_candidate_completes_after_parent_resume(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tests" / "fixtures"))
    monkeypatch.syspath_prepend(str(root / "tests" / "fixtures" / "parallel"))
    from barrier_endpoint import BarrierEndpoint
    from e2e_support import interrupt_run, summary_for, tool_step
    from engine_run import start_monitored

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "target.py").write_text("def target_value():\n    return 1\n")
    (repo / "pkg" / "quiet.py").write_text("VALUE = 0\n")
    (repo / "tests" / "test_target.py").write_text(
        "from pkg.target import target_value\n\n"
        "def test_target_value():\n    assert target_value() == 2\n")
    run_dir = tmp_path / "run"
    contracts = [
        {"id": "target", "outcome": "Set target_value to return 2.",
         "owns": ["pkg/target.py"], "creates": [], "exports": [], "consumes": [],
         "reads": [], "requirements": [], "check": {"tests": []}},
        {"id": "quiet", "outcome": "Leave the quiet module unchanged.",
         "owns": ["pkg/quiet.py"], "creates": [], "exports": [], "consumes": [],
         "reads": [], "requirements": [], "check": {"tests": []}},
    ]
    parent = [
        tool_step("parallel", {"subframes": contracts}),
        tool_step("done", {"summary": "The scoped parallel candidate is ready."}),
    ]
    subframes = {
        "target": [
            tool_step("str_replace", {"path": "pkg/target.py", "old_str": "return 1",
                                       "new_str": "return 2"}),
            tool_step("done", {"summary": "The target change is complete."}),
        ],
        "quiet": [tool_step("done", {"summary": "The quiet file is unchanged."})],
    }

    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("STATEFRAME_"):
            environment.pop(name, None)
    environment.update(HOME=str(tmp_path / "home"),
                       STATEFRAME_HOME=str(tmp_path / "stateframe-home"),
                       PYTHONPATH=str(root / "src"))

    with BarrierEndpoint(parent, subframes, slots=2, hold_until={"target": "quiet"},
                         hold_parent_request=2, barrier_timeout=240) as endpoint:
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task",
            "Fix pkg/target.py to return 2. Only edit pkg/target.py and pkg/quiet.py.",
            "--endpoint", endpoint.url, "--model", "scripted", "--profile", "r2hc3",
            "--run-dir", str(run_dir), "--max-turns", "20", "--parallel", "2",
            "--python", sys.executable, "--test-python", sys.executable, "--shell", "off",
            "--check-cmd", f"{shlex.quote(sys.executable)} -c pass", "--target", "pkg/target.py",
            "--edit", "pkg/quiet.py", "--protect-tests", "off", "--in-place", "--quiet", "--json",
        ]
        process = start_monitored(
            command, cwd=root, env=environment, run_dir=run_dir,
            expected_seconds=240, step="scoped parallel resume run",
        )
        resumed = None
        try:
            assert endpoint.wait_for_requests("parent", 2, timeout=60)
            assert endpoint.wait_for_requests("target", 1, timeout=180)
            assert endpoint.wait_for_requests("quiet", 1, timeout=180)
            interrupt_run(process, run_dir)
            endpoint.release_parent_request()
            resumed = start_monitored(
                [sys.executable, "-m", "stateframe", "resume", str(run_dir),
                 "--endpoint", endpoint.url, "--model", "scripted", "--keep-checkpoints", "--json"],
                cwd=root, env=environment, run_dir=run_dir,
                expected_seconds=240, step="scoped parallel resume",
            )
            assert resumed.wait(timeout=600) == 0
            endpoint.assert_consumed()
        finally:
            if resumed is not None and resumed.poll() is None:
                resumed.cancel("review test cleanup")
                resumed.escalate(grace=0)
                resumed.reap(timeout=15)
            if process.poll() is None:
                process.cancel("review test cleanup")
                process.escalate(grace=0)
                process.reap(timeout=15)

    summary = summary_for(run_dir)
    parent_spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    child_specs = list((run_dir / "parallel").glob("*/units/*/spec.json"))
    assert len(child_specs) == 2
    assert all(json.loads(path.read_text(encoding="utf-8"))["public_row"]["task_file_scope"] ==
               ["pkg/target.py", "pkg/quiet.py"] for path in child_specs)
    assert parent_spec["public_row"]["task_file_scope"] == ["pkg/target.py", "pkg/quiet.py"]
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["pkg/target.py"]
    assert summary["applied_in_place"] == ["pkg/target.py"]
    assert (repo / "pkg" / "target.py").read_text() == "def target_value():\n    return 2\n"
    assert (repo / "pkg" / "quiet.py").read_text() == "VALUE = 0\n"
