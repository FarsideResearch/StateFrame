from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe.test_runners import (command_runs_tests, discover_check, initial_failing_tests,
                                     lock_initial_failing_tests, is_test_file,
                                     judging_test_paths, parse_command, protected_paths,
                                     validate_patterns)
from stateframe.test_changes import inspect_test_changes
from stateframe.task_row import build_row
from stateframe.workcopy import create
from stateframe.engine import kernel_generic_check
from stateframe.engine.kernel_host_done_facts import CheckResult, DoneFacts
from stateframe.engine.kernel_completion_owner import host_public_check_arguments
from stateframe.engine.kernel_cache_layout_candidate import registered_done_for_wire
from stateframe.engine.kernel_information_policy import InformationPolicy


@pytest.mark.parametrize(("command", "expected"), [
    ("pytest -q tests/test_app.py", "pytest"),
    ("python -m pytest -q", "pytest"),
    ("/tmp/venv/bin/python -m pytest", "pytest"),
    ("uv run pytest -q", "pytest"),
    ("uv run python -m pytest -q", "pytest"),
    ("/usr/bin/env python3 -m pytest -q", "pytest"),
    ("env PYTHONPATH=src pytest -q", "pytest"),
    ("echo -m pytest", "generic"),
    ("npm test", "generic"),
    ("npx vitest run", "generic"),
    ("go test ./...", "generic"),
    ("cargo test", "generic"),
    ("make test", "generic"),
    ("./scripts/check.sh", "generic"),
])
def test_runner_detection(command, expected):
    assert parse_command(command)[0] == expected


def test_reprepared_check_keeps_bound_done_description_without_schema_drift():
    registered = {"type": "function", "function": {
        "name": "done", "description": "Submit", "parameters": {
            "type": "object", "properties": {"summary": {"type": "string"}}}}}
    projected = json.loads(json.dumps(registered))
    projected["function"]["description"] = "Submit the checked revision"
    tools = [projected]
    capture = SimpleNamespace(request_id="replacement", final_tool_surface={
        "schema": "stateframe-final-tool-surface/1",
        "request_id": "replacement", "tools": tools},
        _stateframe_tool_surface_pending=SimpleNamespace(request_id="original"))

    resolved = registered_done_for_wire(capture, registered, tools)
    assert resolved == projected
    assert resolved is not projected
    assert registered_done_for_wire(capture, registered, list(tools)) is registered
    projected["function"]["parameters"]["properties"]["summary"]["type"] = "integer"
    assert registered_done_for_wire(capture, registered, tools) is registered
    projected["function"]["parameters"]["properties"]["summary"]["type"] = "string"
    capture.final_tool_surface["request_id"] = "original"
    assert registered_done_for_wire(capture, registered, tools) is registered


def test_runner_override_and_invalid_command():
    assert parse_command("./scripts/check.sh", "pytest")[0] == "pytest"
    assert parse_command("pytest", "generic")[0] == "generic"
    with pytest.raises(ValueError, match="executable"):
        parse_command(" ")
    with pytest.raises(ValueError, match="parsed"):
        parse_command("'unterminated")


def test_protected_ecosystem_paths():
    paths = (
        "tests/test_app.py", "conftest.py", "src/math.test.ts", "src/math.spec.tsx",
        "src/math_test.go", "tests/api.rs", "__tests__/app.js", "scripts/check.sh",
        "package.json", "Makefile", "Cargo.toml", "src/app.js",
    )
    pytest_paths = protected_paths(paths, kind="pytest")
    assert "tests/test_app.py" in pytest_paths and "conftest.py" in pytest_paths
    assert "src/math.test.ts" not in pytest_paths
    generic_paths = protected_paths(paths, kind="generic", command=("./scripts/check.sh",))
    assert set(paths) - {"src/app.js"} <= set(generic_paths)
    assert "src/app.js" not in generic_paths


def test_extra_protected_globs_are_validated():
    assert validate_patterns(["specs/**", "specs/**"]) == ("specs/**",)
    for value in ("", "../escape", "/absolute", "~home"):
        with pytest.raises(ValueError, match="relative"):
            validate_patterns([value])


def test_generic_row_uses_whole_command_and_excludes_protected_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src" / "app.ts").write_text("export const value = 'bad';\n")
    (repo / "src" / "app.spec.ts").write_text("expect(value).toBe('good');\n")
    (repo / "tests" / "check.sh").write_text("exit 1\n")
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix the value", runner_kind="generic",
                    test_command=("sh", "tests/check.sh"), target="src/app.ts")
    assert row["test_command"] == ["sh", "tests/check.sh"]
    assert row["public_tests"] == {}
    assert row["protected_test_paths"] == ["src/app.spec.ts", "tests/check.sh"]
    assert set(row["files"]) == {"src/app.ts"}


def test_generic_row_starts_at_nonempty_source_in_package(tmp_path):
    repo = tmp_path / "repo"
    (repo / "shop").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "shop/__init__.py").write_text("")
    (repo / "shop/pricing.py").write_text("PRICE = 10\n")
    (repo / "tests/test_pricing.py").write_text("def test_price():\n    assert True\n")
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix pricing", runner_kind="generic",
                    test_command=("python", "-m", "unittest", "discover", "-s", "tests"),
                    protect_tests="off")
    assert row["entry"]["path"] == "shop/pricing.py"


def test_generic_row_prefers_code_over_readme_without_explicit_target(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("Project notes\n")
    (repo / "src/app.js").write_text("export const value = 1;\n")
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Check the app", runner_kind="generic",
                    test_command=("node", "src/app.js"), protect_tests="off")
    assert row["entry"]["path"] == "src/app.js"


@pytest.mark.parametrize(("exit_code", "timed_out", "expected"), [
    (0, False, 0), (1, False, 1), (126, False, 126),
    (127, False, 127), (0, True, 124),
])
def test_generic_exit_code_is_authoritative(monkeypatch, tmp_path, exit_code, timed_out, expected):
    monkeypatch.setattr(kernel_generic_check, "ACTIVE_RUNTIME", SimpleNamespace(get=lambda: None))
    def fake_run(command, root, **kwargs):
        assert command == ("check",)
        assert root == tmp_path
        assert kwargs["output_limit"] == kernel_generic_check.OUTPUT_LIMIT
        assert kwargs["generic_check"] is True
        return SimpleNamespace(
            completed=SimpleNamespace(returncode=exit_code, stdout="result\n", stderr=""),
            timed_out=timed_out, output_truncated=False)

    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", fake_run)
    text, code = kernel_generic_check.check_result(tmp_path, {
        "test_command": ["check"], "runner_kind": "generic"})
    assert code == expected
    assert text.startswith(f"check exit {expected}")


def test_generic_output_tail_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_generic_check, "ACTIVE_RUNTIME", SimpleNamespace(get=lambda: None))
    payload = "a" * 10_000 + "FAILURE_DETAIL"
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *args, **kwargs:
                        SimpleNamespace(completed=SimpleNamespace(
                            returncode=1, stdout=payload, stderr=""),
                            timed_out=False, output_truncated=False))
    text, code = kernel_generic_check.check_result(tmp_path, {
        "test_command": ["check"], "runner_kind": "generic"})
    assert code == 1
    assert "FAILURE_DETAIL" in text
    assert len(text) < kernel_generic_check.OUTPUT_TAIL + 100


def test_generic_failure_tail_reaches_revision_bound_observation(monkeypatch, tmp_path):
    captured = {}

    class Runtime:
        def binding(self):
            return {"product_revision": "r1"}

        def test_observation(self, selectors, code, binding, **kwargs):
            captured.update(selectors=selectors, code=code, binding=binding, **kwargs)

    monkeypatch.setattr(kernel_generic_check, "ACTIVE_RUNTIME",
                        SimpleNamespace(get=lambda: Runtime()))
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *args, **kwargs:
                        SimpleNamespace(completed=SimpleNamespace(
                            returncode=1, stdout="x" * 5000 + "failure detail", stderr=""),
                            timed_out=False, output_truncated=False))
    _, code = kernel_generic_check.check_result(tmp_path, {
        "test_command": ["check"], "runner_kind": "generic"})
    assert code == 1
    assert captured["selectors"] == [kernel_generic_check.CHECK_ID]
    assert captured["binding"] == {"product_revision": "r1"}
    assert captured["typed_observation"]["outcome"] == "BEHAVIOR_FAIL"
    assert captured["typed_observation"]["assertion_context"].endswith("failure detail")
    assert len(captured["typed_observation"]["assertion_context"]) == 4000


def test_generic_selectors_degrade_to_one_whole_command(monkeypatch, tmp_path):
    import kernel_generic_check as engine_generic_check
    from stateframe.engine.probes import react_repair

    calls = []

    def fake_check(root, row):
        calls.append((root, row))
        return "check exit 0: check passed", 0

    monkeypatch.setattr(engine_generic_check, "check_result", fake_check)
    row = {"runner_kind": "generic", "test_command": ["check"]}
    observation, codes = react_repair.run_tests_for_selectors(
        tmp_path, "", ("missing.py::test_no_such_node",), row=row)
    assert (observation, codes) == ("check exit 0: check passed", (0,))
    assert calls == [(tmp_path, row)]


def test_darwin_check_profile_admits_only_declared_command_binary(monkeypatch, tmp_path):
    import kernel_test_isolation as isolation

    root = tmp_path / "repo"
    root.mkdir()
    target = tmp_path / "installed" / "bin" / "node"
    target.parent.mkdir(parents=True)
    target.write_text("binary")
    alias = tmp_path / "bin" / "node"
    alias.parent.mkdir()
    alias.symlink_to(target)
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(isolation, "selected_test_python", lambda: isolation.sys.executable)
    monkeypatch.setattr(isolation.shutil, "which", lambda name, path=None: str(alias))

    argv, _ = isolation.isolated_command(["node", "--test"], root,
                                          {"PATH": str(alias.parent)}, generic_check=True)
    profile = argv[2]
    assert f'(subpath "{alias}")' in profile
    assert f'(subpath "{target}")' in profile
    assert f'(subpath "{alias.parent}")' not in profile

    pytest_argv, _ = isolation.isolated_command(
        [isolation.sys.executable, "-m", "pytest"], root, {"PATH": str(alias.parent)})
    assert f'(subpath "{isolation.sys.executable}")' not in pytest_argv[2]


def test_generic_check_and_shell_share_system_read_roots(monkeypatch, tmp_path):
    import kernel_test_isolation as isolation
    from kernel_exec_capability import shell_capability, system_runtime_read_roots

    for name in ("repo", "env", "home"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    profile = isolation.isolated_command(
        ["/bin/sh", "scripts/check.sh"], tmp_path / "repo", {"PATH": "/usr/bin"},
        generic_check=True)[0][2]
    shell = shell_capability(workspace=tmp_path / "repo", run_env_dir=tmp_path / "env",
                             scratch_home=tmp_path / "home", network="none", writable=True)
    shared = set(system_runtime_read_roots())
    profile_roots = {root for root in shared
                     if f"(subpath {json.dumps(str(root))})" in profile}
    shell_roots = {root for root in shared if any(
        root == admitted or root.is_relative_to(admitted) for admitted in shell.read_roots)}
    assert profile_roots == shell_roots == shared
    assert "(deny network*)" in profile
    assert not any(root == Path.home() for root in shared)
    assert not any(root == Path.home() for root in shell.write_roots)


def test_relative_check_program_resolves_from_working_copy(monkeypatch, tmp_path):
    import kernel_test_isolation as isolation

    root = tmp_path / "repo"
    script = root / "scripts" / "check.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(isolation.shutil, "which", lambda *args, **kwargs:
                        pytest.fail("relative paths must not search PATH"))
    profile = isolation.isolated_command(
        ["./scripts/check.sh"], root, {"PATH": str(tmp_path)}, generic_check=True)[0][2]
    assert f'(subpath "{script}")' not in profile
    assert f'(subpath "{root}")' in profile


def test_outer_check_receives_runner_kind(monkeypatch, tmp_path):
    from stateframe import runner

    captured = []
    monkeypatch.setattr(runner.outer_checks, "run", lambda *args, **kwargs:
                        captured.append(kwargs["runner_kind"]) or {"passed": True})
    assert runner.run_test_cmd("sh scripts/check.sh", tmp_path,
                               runner_kind="generic") == {"passed": True}
    assert captured == ["generic"]


@pytest.mark.skipif(sys.platform != "darwin", reason="frozen macOS sandbox profile")
def test_pytest_profile_matches_987ae68_fixture(monkeypatch):
    import kernel_test_isolation as isolation

    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    fake_sys = SimpleNamespace(prefix="/fixture/prefix", base_prefix="/fixture/base",
                               executable="/fixture/bin/python", platform="darwin")
    monkeypatch.setattr(isolation, "sys", fake_sys)
    monkeypatch.setattr(isolation, "selected_test_python", lambda: fake_sys.executable)
    monkeypatch.setattr(isolation.site, "getsitepackages", lambda: [])
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory) / "repo"
        root.mkdir()
        argv, _ = isolation.isolated_command(
            [fake_sys.executable, "-m", "pytest", "-q"], root, {"PATH": "/usr/bin"})
        actual = argv[2].replace(str(root.resolve()), "<ROOT>").replace(directory, "<DIR>")
    fixture = Path(__file__).parent / "acceptance/fixtures/pytest_check_profile_987ae68.json"
    assert actual == json.loads(fixture.read_text())


def test_generic_check_cannot_pass_after_mutating_protected_test(monkeypatch, tmp_path):
    protected = tmp_path / "app.test.js"
    protected.write_text("expected good\n")
    monkeypatch.setattr(kernel_generic_check, "ACTIVE_RUNTIME", SimpleNamespace(get=lambda: None))

    def fake_run(*args, **kwargs):
        protected.write_text("expected bad\n")
        return SimpleNamespace(completed=SimpleNamespace(
            returncode=0, stdout="pass", stderr=""),
            timed_out=False, output_truncated=False)

    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", fake_run)
    text, code = kernel_generic_check.check_result(tmp_path, {
        "test_command": ["check"], "protected_test_paths": ["app.test.js"]})
    assert code == 125
    assert "protected test resource changed" in text


def test_generic_done_requires_current_whole_command_pass():
    required = (kernel_generic_check.CHECK_ID,)
    common = {"current_revision": "r2", "changed_files": ("app.js",),
              "empty_diff": False, "public_tests_required": required}
    passed = DoneFacts(**common, last_check_result=CheckResult(
        status="PASS", selectors=required, revision="r2", current=True, exit_code=0))
    assert passed.accepted
    assert passed.host_check_state == "REQUIRED_PUBLIC_CHECKS"
    missing = DoneFacts(**common)
    assert "CHECK_MISSING" in missing.rejection_reasons
    assert "PUBLIC_TESTS_MISSING" in missing.rejection_reasons
    failed = DoneFacts(**common, last_check_result=CheckResult(
        status="FAIL", selectors=required, revision="r2", current=True, exit_code=1))
    assert "CHECK_FAILED" in failed.rejection_reasons


@pytest.mark.parametrize("cache_layout", [False, True])
def test_framefit_preassembly_defers_terminal_wire_proof(monkeypatch, cache_layout):
    import kernel_completion_owner as completion_owner
    from kernel_stateframe_final_message import PAGE_LOG_FINAL_MESSAGE_NAME
    from kernel_stateframe_page_log import _authority_content, _partition_authority

    done = {"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}
    terminal = ("Final step: call done now to submit this revision "
                "(uncertified; open: [uncertifiable host coverage]).")
    protocol = {"schema": "stateframe-framefit5-owner/1", "route": "repair",
                "admitted_actions": ["done"]}
    payload = {"state": {"framefit5": {"mode": "FRAME_ONLY", "owner_protocol": protocol}}}
    messages = [{"role": "system", "content": "System"},
                {"role": "user", "content": "Task: fix app"}]
    if cache_layout:
        cacheable, fields = _partition_authority(payload)
        messages.extend([
            {"role": "user", "content": "[StateFrame]\n" + json.dumps(cacheable)},
            {"role": "user", "name": PAGE_LOG_FINAL_MESSAGE_NAME,
             "content": _authority_content(fields)},
        ])
    else:
        messages.append({"role": "user", "content": "[StateFrame]\n" + json.dumps(payload)})

    contract = {
        "request_id": "terminal-1", "route": "submit", "terminal_line": terminal,
        "admitted_tool_schemas": [done], "advertised_actions": ["done"],
        "tool_schema_digest": completion_owner._digest([done]),
        "owner_blocks": {"menu": {"digest": completion_owner._digest([done])}},
    }
    contract["decision_digest"] = completion_owner._contract_decision_digest(contract)
    monkeypatch.setattr(completion_owner, "_contract_route",
                        lambda *_args, **_kwargs: ("submit", None))
    monkeypatch.setattr(completion_owner, "_contract_revision", lambda _runtime: "r1")
    monkeypatch.setattr(completion_owner, "_contract_unresolved_work", lambda _runtime: [])

    def bind(*, preassembly):
        runtime = SimpleNamespace(owner=SimpleNamespace(task_row={"runner_kind": "generic"}))
        capture = SimpleNamespace(request_action_contract=contract.copy())
        surface = {"request_id": "terminal-1", "tools": [done], "names": ["done"],
                   "tool_schema_digest": completion_owner._digest([done])}
        return completion_owner.bind_final_tool_surface(
            runtime, capture, surface, messages=messages,
            framefit5_preassembly=preassembly)

    assert bind(preassembly=True)
    assert not bind(preassembly=False)
    assert completion_owner._framefit5_owner_protocol(messages) == protocol
    assert not completion_owner._framefit5_terminal_line_present(messages, terminal)


def test_generic_completion_owner_requests_whole_command():
    runtime = SimpleNamespace(owner=SimpleNamespace(task_row={
        "runner_kind": "generic", "test_command": ["sh", "scripts/check.sh"]}))
    assert host_public_check_arguments(runtime) == {
        "target": "", "selectors": [kernel_generic_check.CHECK_ID]}


def test_generic_completion_owner_requires_a_command():
    runtime = SimpleNamespace(owner=SimpleNamespace(task_row={"runner_kind": "generic"}))
    assert host_public_check_arguments(runtime) is None


def test_generic_worker_menu_and_dispatch_survive_edit(monkeypatch, tmp_path):
    import kernel_completion_owner as completion_owner
    from kernel_continuation_runtime import required_acceptance_selectors
    import kernel_generic_check as engine_generic_check
    from stateframe.engine.probes import react_repair

    monkeypatch.delenv("INITIAL_ORACLE_SYMPTOM", raising=False)
    monkeypatch.delenv("LIVE_GRADER_FEEDBACK", raising=False)
    row = {"runner_kind": "generic", "test_command": ["sh", "scripts/check.sh"],
           "protected_test_paths": ["scripts/check.sh"], "repo": "build"}
    policy = InformationPolicy({"information_policy": {
        "information_policy_version": "information-policy/1",
        "initial_oracle_symptom": "OFF", "live_grader_feedback": "DIAGNOSTIC",
        "target_disclosure": "FILE_AND_SYMBOL"}})
    projected = policy.worker_row(row)
    assert {key: projected[key] for key in (
        "runner_kind", "test_command", "protected_test_paths")} == {
        key: row[key] for key in ("runner_kind", "test_command", "protected_test_paths")}

    revision = ["before"]
    runtime = SimpleNamespace(
        owner=SimpleNamespace(task_row=projected),
        binding=lambda: {"product_revision": revision[0]},
        public_test_verdict=lambda: {"current_binding": None},
        required_public_selectors=lambda: [kernel_generic_check.CHECK_ID],
    )
    tool = {"function": {"name": "run_tests"}}
    monkeypatch.setattr(completion_owner, "legal_next_actions", lambda *args, **kwargs: ())
    monkeypatch.setattr(completion_owner, "_route_dispatchable", lambda *args: True)
    for current_revision in ("before", "after-edit"):
        revision[0] = current_revision
        assert required_acceptance_selectors(runtime) == [kernel_generic_check.CHECK_ID]
        actions = completion_owner._postgreen_work_actions(runtime, None, [tool])
        assert actions[0]["tool"] == "run_tests"
        assert actions[0]["arguments"] == {
            "target": "", "selectors": [kernel_generic_check.CHECK_ID]}

    calls = []
    monkeypatch.setattr(engine_generic_check, "check_result", lambda root, task_row: (
        calls.append((root, task_row)) or "check exit 0: check passed", 0))
    assert react_repair.run_tests(tmp_path, "", None, row=projected).startswith("check exit 0")
    assert calls == [(tmp_path, projected)]


@pytest.mark.parametrize(("code", "expected"), [
    (0, "GREEN"), (1, "RED"), (5, "RED"), (42, "RED"),
    (124, "HARNESS_ERROR"), (125, "HARNESS_ERROR"),
    (126, "HARNESS_ERROR"), (127, "HARNESS_ERROR"),
    (None, "HARNESS_ERROR"),
])
def test_generic_continuation_result_class(code, expected):
    assert kernel_generic_check.result_class(code) == expected


def test_generic_success_label_in_summary_and_cli(tmp_path, capsys):
    from stateframe import outcomes
    from stateframe.cli import _print_summary

    summary = {
        "runner_kind": "generic", "child_exit_code": 0,
        "engine_result": "SUBMITTED", "engine_terminal": {
            "result": "SUBMITTED", "submission": {
                "stored": True, "seal_id": "sha256:" + "a" * 64,
                "artifact_digest": "sha256:" + "b" * 64}},
        "candidate_captured": True,
        "candidate_artifact_digest": "sha256:" + "b" * 64,
        "candidate_sha256": "sha256:" + "c" * 64,
        "tests_before": {"passed": False},
        "tests_after": {"passed": True, "exit_code": 0,
                        "command": "sh scripts/check.sh", "candidate_sha256": "sha256:" + "c" * 64},
        "run_dir": str(tmp_path),
    }
    summary.update(outcomes.project(summary))
    assert summary["verification"] == "verified by check command"
    assert summary["verification_detail"] == "exit 0: sh scripts/check.sh"
    summary["succeeded"] = True
    (tmp_path / "final.diff").write_text("")
    _print_summary(summary)
    output = capsys.readouterr().out
    assert "result: verified by check command (exit 0: sh scripts/check.sh)" in output
    assert "\nverified by check command (exit 0: sh scripts/check.sh)\n" in output
    summary["test_lock_note"] = "selector-free task check: all existing test files locked"
    _print_summary(summary)
    assert ("result: verified by check command (exit 0: sh scripts/check.sh); selector-free task check: "
            "all existing test files locked") in capsys.readouterr().out


@pytest.mark.parametrize(("files", "expected", "source"), [
    ({"pyproject.toml": "", "src/app.py": "VALUE = 1\n"},
     "python -m compileall -q src/app.py", "build"),
    ({"package.json": '{"scripts":{"lint":"echo lint","build":"echo build",'
                      '"typecheck":"echo type"}}'}, "npm run typecheck", "build"),
    ({"package.json": '{"scripts":{"build":"echo build"}}', "pnpm-lock.yaml": ""},
     "pnpm run build", "build"),
    ({"go.mod": "module x\n"}, "go build ./...", "build"),
    ({"Cargo.toml": "[package]\nname='x'\n"}, "cargo check", "build"),
    ({"Makefile": "build:\n\ttrue\n"}, "make -f Makefile build", "build"),
    ({"README.md": "hello\n"}, None, "none"),
])
def test_no_test_check_discovery(tmp_path, files, expected, source):
    for path, body in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    assert discover_check(tmp_path) == (expected, source, False)


def test_discovery_never_uses_build_when_tests_exist(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts":{"build":"echo pass"}}')
    (tmp_path / "app.test.js").write_text("test('value', () => {});\n")
    assert discover_check(tmp_path) == (None, "tests", True)


def test_test_file_and_judging_path_detection():
    paths = ("tests/check.sh", "tests/other.sh", "src/value.spec.ts", "src/value_test.go")
    assert all(is_test_file(path) for path in paths)
    assert judging_test_paths(paths, ("sh", "tests/check.sh"), ()) == ("tests/check.sh",)
    assert initial_failing_tests(paths, "FAILED src/value.spec.ts::value") == ("src/value.spec.ts",)


@pytest.mark.parametrize(("kind", "command", "output", "expected"), [
    ("generic", ("npm", "test"), "ok\n", False),
    ("generic", ("yarn", "test"), "ok\n", False),
    ("generic", ("make", "test"), "ok\n", False),
    ("generic", ("node", "--test"), "# pass 2\n# fail 0\n", True),
    ("generic", ("node", "--test"), "ℹ pass 1\nℹ fail 0\n", True),
    ("generic", ("npm", "test"), "Tests: 2 passed, 2 total\n", True),
    ("generic", ("go", "test", "-v", "./..."),
     "=== RUN TestValue\n--- PASS: TestValue (0.01s)\nPASS\nok  example/pkg 0.01s\n", True),
    ("generic", ("cargo", "test"),
     "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n", True),
    ("generic", ("node", "--test"), "# pass 0\n", False),
    ("pytest", ("pytest", "--collect-only"), "2 tests collected\n", False),
    ("pytest", ("pytest", "--version"), "pytest 8.0\n", False),
    ("pytest", ("python", "-m", "pytest", "-q"), "2 passed in 0.01s\n", True),
    ("pytest", ("python", "-m", "pytest", "-q"), "no tests ran\n", False),
])
def test_command_evidence_requires_positive_receipt(tmp_path, kind, command, output, expected):
    paths = {"tests/test_app.sh", "tests/test_app.py", "verify.sh"}
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}')
    assert command_runs_tests(kind, command, paths, output=output, repo=tmp_path) is expected


@pytest.mark.parametrize(("command", "output", "expected"), [
    (("npm", "test"), "ok\n", "verified by check command"),
    (("npm", "test"), "Tests: 1 passed, 1 total\n", "verified by tests"),
    (("pytest", "--collect-only"), "1 test collected\n", "verified by check command"),
    (("pytest", "-q"), "1 passed in 0.01s\n", "verified by tests"),
])
def test_execution_receipt_controls_exact_result_label(tmp_path, command, output, expected):
    from stateframe import outcomes
    kind = "pytest" if command[0] == "pytest" else "generic"
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}')
    digest = "sha256:" + "a" * 64
    summary = {"runner_kind": kind, "protect_tests": "off", "check_source": "tests",
               "child_exit_code": 0, "engine_terminal": {"result": "SUBMITTED",
               "submission": {"stored": True, "seal_id": digest, "artifact_digest": digest}},
               "candidate_artifact_digest": digest, "candidate_captured": True,
               "candidate_sha256": digest,
               "tests_after": {"passed": True, "exit_code": 0, "candidate_sha256": digest,
                               "command": " ".join(command)}}
    assert outcomes.project(summary, check_output=output, check_repo=tmp_path,
                            declared_test_command=True)["verification"] == expected


def test_task_pytest_locks_check_and_command_selected_tests(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 1\n")
    for name in ("a", "b", "c"):
        (repo / "tests" / f"test_{name}.py").write_text(f"def test_{name}():\n    assert True\n")
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix app", checks=["tests/test_a.py"],
                    test_command=("python", "-m", "pytest", "-q", "tests/test_b.py::test_b"),
                    protect_tests="task", target="app.py")
    assert row["protected_test_paths"] == ["tests/test_a.py", "tests/test_b.py"]
    assert "tests/test_c.py" in row["files"]
    assert "tests/test_b.py" not in row["files"]


def test_task_locks_failure_named_before_four_kib_tail():
    row = {"test_file_paths": ["tests/test_a.py", "tests/test_b.py"],
           "protected_test_paths": ["tests/test_a.py"],
           "files": {"tests/test_b.py": {"parent_src": "assert False"}, "app.py": {}}}
    output = "FAILED tests/test_b.py::test_case\n" + "x" * 5000
    lock_initial_failing_tests(row, {"output": output, "tail": output[-4000:],
                                     "output_truncated": False})
    assert row["protected_test_paths"] == ["tests/test_a.py", "tests/test_b.py"]
    assert "tests/test_b.py" not in row["files"]


@pytest.mark.parametrize("before", [
    {"output": "failure without a test path", "output_truncated": False},
    {"output": "FAILED tests/test_a.py::test_case", "output_truncated": True},
])
def test_task_fails_closed_when_starting_failure_cannot_be_fully_attributed(before):
    row = {"test_file_paths": ["tests/test_a.py", "tests/test_b.py"],
           "protected_test_paths": ["tests/test_a.py"],
           "files": {"tests/test_b.py": {}, "app.py": {}}}
    lock_initial_failing_tests(row, before)
    assert row["protected_test_paths"] == ["tests/test_a.py", "tests/test_b.py"]
    assert "tests/test_b.py" not in row["files"]


@pytest.mark.parametrize("command", [("python", "-m", "pytest", "-q"),
                                      ("python", "-m", "pytest", "-k", "case")])
def test_task_pytest_without_file_selector_locks_discovered_tests(tmp_path, command):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 1\n")
    for name in ("a", "b"):
        (repo / "tests" / f"test_{name}.py").write_text("def test_case():\n    assert True\n")
    row = build_row(create(repo, tmp_path / "copy"), task="Fix app",
                    checks=["tests/test_a.py"], test_command=command,
                    protect_tests="task", target="app.py")
    assert row["protected_test_paths"] == ["tests/test_a.py", "tests/test_b.py"]


@pytest.mark.parametrize("command", [("node", "--test"), ("npm", "test"),
                                      ("go", "test", "./...")])
def test_task_generic_without_file_selector_locks_all_even_when_starting_green(tmp_path, command):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.js").write_text("export const value = 1;\n")
    for name in ("a", "b"):
        (repo / "tests" / f"{name}.test.js").write_text("test('value', () => {});\n")
    row = build_row(create(repo, tmp_path / "copy"), task="Fix app", runner_kind="generic",
                    test_command=command, protect_tests="task", target="app.js")
    assert row["protected_test_paths"] == ["tests/a.test.js", "tests/b.test.js"]
    assert row["test_lock_note"] == "selector-free task check: all existing test files locked"
    assert set(row["files"]) == {"app.js"}


@pytest.mark.parametrize(("original", "changed"), [
    ("assert value", "assert True"),
    ("assert value", "assert 1"),
    ("assert value", "assert value or True"),
    ("assert value == expected", "assert True"),
])
def test_python_tautological_assertion_is_weakening(original, changed):
    from stateframe.test_changes import _python_findings
    before = f"def test_value():\n    {original}\n"
    after = f"def test_value():\n    {changed}\n"
    assert {item["reason"] for item in _python_findings("tests/test_app.py", before, after)} >= {
        "assertion made tautological"}


def test_python_assertion_removed_conjunct_is_weakening():
    from stateframe.test_changes import _python_findings

    findings = _python_findings("tests/test_app.py",
                                "def test_value():\n    assert value and ready\n",
                                "def test_value():\n    assert value\n")
    assert any(item["reason"] == "assertion predicate reduced" for item in findings)


@pytest.mark.parametrize(("before", "after", "test"), [
    ("def test_value():\n    assert True\n",
     "import pytest\npytest.skip('later', allow_module_level=True)\ndef test_value():\n    assert True\n",
     "test_value"),
    ("import pytest\nclass TestValue:\n    def test_case(self):\n        assert True\n",
     "import pytest\n@pytest.mark.skipif(True, reason='later')\nclass TestValue:\n    def test_case(self):\n        assert True\n",
     "TestValue::test_case"),
    ("import pytest\ndef test_value():\n    assert True\n",
     "import pytest\npytestmark = pytest.mark.xfail(reason='later')\ndef test_value():\n    assert True\n",
     "test_value"),
    ("import pytest\npytestmark = []\ndef test_value():\n    assert True\n",
     "import pytest\npytestmark = []\npytestmark += [pytest.mark.skip(reason='later')]\n"
     "def test_value():\n    assert True\n",
     "test_value"),
    ("import pytest\ndef test_value():\n    assert True\n",
     "import pytest\ndef test_value():\n    if True:\n        pytest.skip('later')\n    assert True\n",
     "test_value"),
])
def test_python_scope_skip_xfail_weakens_each_affected_test(before, after, test):
    from stateframe.test_changes import _python_findings
    assert {item["reason"] for item in _python_findings("tests/test_app.py", before, after)
            if item["test"] == f"tests/test_app.py::{test}"} == {
                "skip/xfail/only/todo marker added"}


@pytest.mark.parametrize(("original", "changed"), [
    ("assert value > 5", "assert value >= 5"),
    ("assert value == 5", "assert value in limits"),
    ("assert 5 < value < 10", "assert 5 <= value < 10"),
    ("assert value > 5 and value < 10", "assert value > 5"),
    ("assert value > 5", "assert value > 5 or True"),
])
def test_python_assertion_comparison_changes_are_flagged(original, changed):
    from stateframe.test_changes import _python_findings
    before = f"def test_value():\n    {original}\n"
    after = f"def test_value():\n    {changed}\n"
    findings = _python_findings("tests/test_app.py", before, after)
    assert any(item["reason"] == "comparison loosened"
               and item["test"] == "tests/test_app.py::test_value" for item in findings)


def test_python_comparison_weakening_qualifies_exact_summary_value(tmp_path):
    from stateframe import outcomes

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests/test_app.py").write_text("def test_value():\n    assert value > 5\n")
    work = create(repo, tmp_path / "copy")
    (work.copy / "tests/test_app.py").write_text("def test_value():\n    assert value >= 5\n")
    findings = inspect_test_changes(work)[0]["weakening"]
    assert findings == [{"test": "tests/test_app.py::test_value",
                         "reason": "comparison loosened", "method": "heuristic"}]
    digest = "sha256:" + "a" * 64
    summary = {"runner_kind": "generic", "check_source": "tests",
               "test_execution_evidence": True, "child_exit_code": 0,
               "engine_terminal": {"result": "SUBMITTED", "submission": {
                   "stored": True, "seal_id": digest, "artifact_digest": digest}},
               "candidate_artifact_digest": digest, "candidate_captured": True,
               "candidate_sha256": digest,
               "tests_after": {"passed": True, "exit_code": 0, "candidate_sha256": digest,
                               "command": "node --test"},
               "test_weakening_findings": findings}
    assert outcomes.project(summary, check_output="# pass 1\n# fail 0\n",
                            declared_test_command=True)["verification"] == (
        "verified by tests, but 1 test file weakened: tests/test_app.py::test_value "
        "(comparison loosened; heuristic)")


def test_tautological_assertion_qualifies_exact_result(tmp_path):
    from stateframe import outcomes

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests/test_app.py").write_text("def test_value():\n    assert value\n")
    work = create(repo, tmp_path / "copy")
    (work.copy / "tests/test_app.py").write_text("def test_value():\n    assert True\n")
    findings = inspect_test_changes(work)[0]["weakening"]
    assert findings == [{"test": "tests/test_app.py::test_value",
                         "reason": "assertion made tautological", "method": "heuristic"}]
    digest = "sha256:" + "a" * 64
    summary = {"runner_kind": "generic", "check_source": "tests",
               "test_execution_evidence": True, "child_exit_code": 0,
               "engine_terminal": {"result": "SUBMITTED", "submission": {
                   "stored": True, "seal_id": digest, "artifact_digest": digest}},
               "candidate_artifact_digest": digest, "candidate_captured": True,
               "candidate_sha256": digest,
               "tests_after": {"passed": True, "exit_code": 0, "candidate_sha256": digest,
                               "command": "node --test"},
               "test_weakening_findings": findings}
    assert outcomes.project(summary, check_output="# pass 1\n# fail 0\n",
                            declared_test_command=True)["verification"] == (
        "verified by tests, but 1 test file weakened: tests/test_app.py::test_value "
        "(assertion made tautological; heuristic)")


def test_js_tautological_expectation_is_weakening():
    from stateframe.test_changes import _line_findings

    before = "test('value', () => { expect(value).toBe(true); });\n"
    after = "test('value', () => { expect(true).toBe(true); });\n"
    assert any(item["reason"] == "assertion made tautological"
               for item in _line_findings("app.test.js", before, after))


@pytest.mark.parametrize(("mode", "locked", "editable"), [
    ("all", {"tests/check.sh", "tests/other.sh"}, set()),
    ("task", {"tests/check.sh"}, {"tests/other.sh"}),
    ("off", set(), {"tests/check.sh", "tests/other.sh"}),
])
def test_protect_modes_change_edit_scope(tmp_path, mode, locked, editable):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "tests/check.sh").write_text("exit 1\n")
    (repo / "tests/other.sh").write_text("assert good\n")
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix value", runner_kind="generic",
                    test_command=("sh", "tests/check.sh"),
                    protect_tests=mode, target="app.sh")
    assert locked <= set(row["protected_test_paths"])
    assert editable <= set(row["files"])
    assert not editable & set(row["protected_test_paths"])


def test_python_test_weakening_flags_named_assertion_and_removed_function(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    original = "def test_value():\n    assert 2 == 2\n\ndef test_other():\n    assert True\n"
    (repo / "tests/test_app.py").write_text(original)
    work = create(repo, tmp_path / "copy")
    (work.copy / "tests/test_app.py").write_text("def test_value():\n    pass\n")
    changes = inspect_test_changes(work)
    assert changes[0]["path"] == "tests/test_app.py"
    assert {item["test"]: item["reason"] for item in changes[0]["weakening"]} == {
        "tests/test_app.py::test_value": "assertion removed",
        "tests/test_app.py::test_other": "test function removed"}


@pytest.mark.parametrize(("path", "before", "after", "test", "reason"), [
    ("app.test.js", "test('one', () => { expect(x).toBe(1) })\n",
     "test('one', () => { })\n", "one", "assertion removed"),
    ("app_test.go", 'func TestValue(t *testing.T) { t.Fatal("bad") }\n',
     "", "TestValue", "test function removed"),
    ("tests/value.rs", "#[test]\nfn test_value() { assert_eq!(1, 1); }\n",
     "", "test_value", "test function removed"),
])
def test_non_python_weakening_heuristic_names_test(path, before, after, test, reason):
    from stateframe.test_changes import _line_findings
    assert {item["reason"] for item in _line_findings(path, before, after)
            if item["test"] == f"{path}::{test}"} == {reason}


def test_no_check_and_authored_check_outcome_labels():
    from stateframe import outcomes
    base = {"runner_kind": "generic", "check_source": "none", "child_exit_code": 0,
            "engine_terminal": {"result": "SUBMITTED", "submission": {
                "stored": True, "seal_id": "sha256:" + "a" * 64,
                "artifact_digest": "sha256:" + "b" * 64}},
            "candidate_captured": True, "candidate_artifact_digest": "sha256:" + "b" * 64,
            "tests_after": {"passed": None, "exit_code": None}}
    plain = {**base, "engine_result": "SUBMITTED"}
    plain.update(outcomes.project(plain))
    assert plain["verification"] == "unverified: review the diff"
    assert outcomes.exit_code(plain) == 0
    authored = {**base, "authored_check_receipt_id": "receipt-1"}
    authored.update(outcomes.project(authored))
    assert authored["verification"] == "verified by an agent-written check"


def test_agent_written_check_requires_parent_red_and_candidate_green(monkeypatch, tmp_path):
    from stateframe.engine import kernel_agent_authored_check as authored

    def green(_snapshot, _path, _source, identity, _marker, side, _parent):
        return {"check_identity": identity, "side": side, "revision": side,
                "status": "PASS", "reached_target": True, "integrity": True,
                "binding_stable": True}

    monkeypatch.setattr(authored, "_run_side", green)
    path = "scratch/authored-check/check.py"
    proposal = {"check_id": "check", "requirement_id": "value", "clause": "value is good",
                "requirement_quote": "value is good", "path": path, "selector": path,
                "language": "python", "source": "print('__STATEFRAME_AUTHORED_OBSERVED__')\n",
                "write_authority": {"authority_ref": "ignored", "allowed_paths": [path]}}
    result = authored.run_agent_check(
        proposal, task="The value is good", root=tmp_path,
        parent_snapshot={"revision": "parent", "files": {}},
        candidate_snapshot={"revision": "candidate", "files": {}})
    assert result["status"] == "NON_DISCRIMINATING"
    assert result["certifies_clause"] is False
