from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
from contextlib import redirect_stdout

import pytest

from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from stateframe import outer_checks
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]
LEDGER_SWITCH = "STATEFRAME_REQUIRED_TEST_LEDGER"
TEST_IDS = ("tests/test_signal.py::test_removed", "tests/test_signal.py::test_broken",
            "tests/test_signal.py::test_control")
RUN_TIMEOUT_SECONDS = 60
pytestmark = pytest.mark.host_sandbox


def _write_repo(repo: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Delivery Test", "-c", "user.email=delivery-test@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-q", "-m", "synthetic input"],
                   cwd=repo, check=True)


def _write_launchers(tmp_path: Path, label: str) -> tuple[Path, Path]:
    launcher = tmp_path / f"fixed_launcher-{label}.py"
    child_wrapper = tmp_path / f"fixed_child-{label}.py"
    launcher.write_text(textwrap.dedent("""
        import ast
        import json
        import os
        from pathlib import Path
        import re
        import subprocess
        import sys
        import uuid
        from uuid import UUID

        from stateframe import cli

        counter = 0
        def fixed_uuid4(*_args, **_kwargs):
            global counter
            counter += 1
            return UUID(int=counter)
        uuid.uuid4 = fixed_uuid4

        original_popen = subprocess.Popen
        def tracked_popen(command, *args, **kwargs):
            if (len(command) == 5 and list(command[1:3]) == ["-B", "-c"]
                    and command[4] == "stateframe._child"):
                environment = kwargs.get("env") or {}
                capture = os.environ.get("LEDGER_CHILD_ENV_CAPTURE")
                if capture:
                    Path(capture).write_text(json.dumps({
                        "switch_present": "STATEFRAME_REQUIRED_TEST_LEDGER" in environment,
                    }), encoding="utf-8")
                child_path, run_path = ast.literal_eval(
                    re.search(r"sys\\.argv=(\\[.*?\\]);", command[3]).group(1))
                executable = os.path.join(kwargs.get("cwd") or ".", command[0])
                command = [executable, "-B", child_path, run_path]
            if len(command) >= 4 and str(command[2]).endswith("_child.py"):
                wrapper = os.environ["LEDGER_CHILD_WRAPPER"]
                command = [command[0], command[1], wrapper, command[2], command[3]]
            return original_popen(command, *args, **kwargs)
        subprocess.Popen = tracked_popen

        raise SystemExit(cli.main(sys.argv[1:]))
    """), encoding="utf-8")
    child_wrapper.write_text(textwrap.dedent("""
        import runpy
        import sys
        import uuid
        from pathlib import Path
        from uuid import UUID

        counter = 0
        def fixed_uuid4(*_args, **_kwargs):
            global counter
            counter += 1
            return UUID(int=counter)
        uuid.uuid4 = fixed_uuid4
        child, run = sys.argv[1:]
        sys.path.insert(0, str(Path(child).resolve().parent.parent))
        sys.path.insert(0, str(Path(child).resolve().parent))
        sys.argv = [child, run]
        runpy.run_path(child, run_name="__main__")
    """), encoding="utf-8")
    return launcher, child_wrapper


def _profile(tmp_path: Path, witness: str) -> Path:
    document = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text(encoding="utf-8"))
    document["switches"]["STATEFRAME_AUTHORED_WITNESS"] = witness
    path = tmp_path / f"profile-witness-{witness}.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _environment(tmp_path: Path, child_wrapper: Path, switches: dict[str, str],
                 capture: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "LEDGER_CHILD_ENV_CAPTURE": str(capture),
        "LEDGER_CHILD_WRAPPER": str(child_wrapper),
    })
    environment.update(switches)
    return environment


def _run_scripted(tmp_path: Path, repo: Path, *, switches: dict[str, str],
                  files: list[dict], task: str,
                  label: str = "run", profile: str | Path = "r2hc3",
                  test_cmd: str | None = None, protect_tests: str | None = None,
                  expected_returncode: int = 0) -> dict[str, object]:
    launcher, child_wrapper = _write_launchers(tmp_path, label)
    run_dir = tmp_path / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    capture = tmp_path / f"child-environment-{label}.json"
    script = Script({"model": "scripted", "steps": files, "tokenize": {"mode": "ok"}})
    server = serve(script)
    command = [sys.executable, str(launcher), "run", "--repo", str(repo), "--task", task,
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--run-dir", str(run_dir), "--profile", str(profile), "--test-python", sys.executable,
               "--max-turns", "4", "--no-run-env", "--quiet"]
    if test_cmd is not None:
        command.extend(["--test-cmd", test_cmd])
    if protect_tests is not None:
        command.extend(["--protect-tests", protect_tests])
    try:
        result = run_monitored(command, env=_environment(
            tmp_path, child_wrapper, switches, capture),
            expected_seconds=RUN_TIMEOUT_SECONDS, run_dir=run_dir,
            step="required test ledger delivery")
        assert result.returncode == expected_returncode, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    requests = [path.read_bytes() for path in sorted(
        (run_dir / "wire").glob("*.request.json"), key=lambda path: int(path.name.split(".")[0]))]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = measurement_events(run_dir)
    child_environment = json.loads(capture.read_text(encoding="utf-8"))
    return {"requests": requests, "summary": summary, "events": events,
            "output": result.stdout + result.stderr, "child_environment": child_environment,
            "exit_code": result.returncode}


def _read_steps() -> list[dict]:
    return [
        {"tool": "read", "arguments": {"path": "pkg/value.py"}},
        {"tool": "done", "arguments": {"summary": "The synthetic value is three."}},
    ]


@pytest.fixture(scope="module")
def parity_workspace(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ledger-wire-parity")
    repo = tmp_path / "repo"
    _write_repo(repo, {"pkg/value.py": "VALUE = 3\n", "README.md": "Read the neutral value."})
    result_dir = tmp_path / "traces"
    result_dir.mkdir()
    return {"tmp_path": tmp_path, "repo": repo, "result_dir": result_dir}


def _parity_trace(workspace, label: str, switches: dict[str, str], profile=None) -> dict[str, object]:
    """Run a trace once per worker, on demand, so each test can run alone."""
    path = workspace["result_dir"] / f"{label}.json"
    if not path.is_file():
        options = {} if profile is None else {"profile": profile}
        _save_trace(path, _run_scripted(
            workspace["tmp_path"], workspace["repo"], switches=switches, files=_read_steps(),
            task="Read pkg/value.py and summarize its value.", label=label, **options))
    return _load_trace(path)


def _witness_trace(workspace, witness: str, ledger: str) -> dict[str, object]:
    return _parity_trace(workspace, f"witness-{witness}-ledger-{ledger}", {LEDGER_SWITCH: ledger},
                         _profile(workspace["tmp_path"], witness))


def _save_trace(path: Path, trace: dict[str, object]) -> None:
    document = {key: value for key, value in trace.items() if key != "requests"}
    document["requests_hex"] = [request.hex() for request in trace["requests"]]
    path.write_text(json.dumps(document), encoding="utf-8")


def _load_trace(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    document["requests"] = [bytes.fromhex(request) for request in document.pop("requests_hex")]
    return document


def _assert_no_ledger_request_content(requests: list[bytes]) -> None:
    lowered = b"\n".join(requests).lower()
    assert LEDGER_SWITCH.lower().encode() not in lowered
    assert b"required_test_ledger" not in lowered
    for test_id in TEST_IDS:
        assert test_id.encode() not in lowered


def _assert_no_ledger_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert "required_test_ledger" not in str(key).lower()
            assert LEDGER_SWITCH.lower() not in str(key).lower()
            _assert_no_ledger_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_ledger_fields(item)


def _assert_equal_partner(ledger_off: dict[str, object], ledger_on: dict[str, object]) -> None:
    assert ledger_on["exit_code"] == ledger_off["exit_code"]
    assert ledger_on["requests"] == ledger_off["requests"]
    assert ledger_on["summary"]["succeeded"] == ledger_off["summary"]["succeeded"]
    assert ledger_on["summary"]["runtime_outcome"] == ledger_off["summary"]["runtime_outcome"]
    assert ledger_on["summary"]["engine_result"] == ledger_off["summary"]["engine_result"]


def _assert_failure_reason_visible(result: dict[str, object]) -> None:
    reason = result.get("required_test_report_reason")
    assert reason == "pytest report was not produced"
    assert result.get("required_test_report") == {"complete": False, "outcomes": {}}


def _normalized_check_output(value: str) -> str:
    return re.sub(r"\bin \d+(?:\.\d+)?s\b", "in <duration>s", value)


def _embedded_session_finish_reason(tmp_path: Path, monkeypatch, capsys, label: str, *,
                                    selected, outcomes: dict[str, str], finished: set[str],
                                    incomplete: bool = False, collectonly: bool = False,
                                    exit_status: int = 0, report_limit: int = 1024,
                                    report_is_directory: bool = False) -> tuple[str, object | None]:
    from types import ModuleType, SimpleNamespace

    from kernel_required_tests import PYTEST_LEDGER_SOURCE

    captured: dict[str, object] = {}

    def hookimpl(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    fake_pytest = ModuleType("pytest")
    fake_pytest.__file__ = str(tmp_path / "pytest.py")
    fake_pytest.hookimpl = hookimpl

    def main(*_args, **kwargs):
        captured.update(kwargs)
        return 0

    fake_pytest.main = main
    monkeypatch.setitem(sys.modules, "pytest", fake_pytest)
    report_path = tmp_path / f"embedded-report-{label}"
    if report_is_directory:
        report_path.mkdir()
    namespace = {
        "repo": str(tmp_path), "report_path": str(report_path), "report_limit": report_limit,
        "console_script": None, "original_args": [],
    }
    previous_argv = sys.argv
    try:
        sys.argv = ["pytest"]
        with pytest.raises(SystemExit) as stopped:
            exec(PYTEST_LEDGER_SOURCE, namespace)
        assert stopped.value.code == 0
    finally:
        sys.argv = previous_argv
    ledger = captured["plugins"][0]
    ledger.selected = selected
    ledger.outcomes = outcomes
    ledger.finished = finished
    ledger.incomplete = incomplete
    session = SimpleNamespace(config=SimpleNamespace(
        option=SimpleNamespace(collectonly=collectonly), rootpath=tmp_path))
    ledger.pytest_sessionfinish(session, exit_status)
    ledger.write_report()
    diagnostic = capsys.readouterr().err
    report = None
    if not report_is_directory and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return diagnostic, report


@pytest.mark.parametrize(("label", "switches"), (
    ("unset", {}), ("off", {LEDGER_SWITCH: "off"}), ("on", {LEDGER_SWITCH: "on"}),
))
@pytest.mark.xdist_group("ledger_delivery")
def test_off_unset_and_on_send_identical_provider_requests_under_the_shipped_profile(
        parity_workspace, label, switches):
    trace = _parity_trace(parity_workspace, label, switches)
    assert len(trace["requests"]) >= 2
    _assert_no_ledger_request_content(trace["requests"])
    assert trace["child_environment"]["switch_present"] is False

    if label == "on":
        unset = _parity_trace(parity_workspace, "unset", {})
        off = _parity_trace(parity_workspace, "off", {LEDGER_SWITCH: "off"})
        assert unset["requests"] == off["requests"] == trace["requests"]
        mutated_request = parity_workspace["tmp_path"] / "request-mutated.json"
        mutated_request.write_bytes(trace["requests"][0] +
                                    b' {"STATEFRAME_REQUIRED_TEST_LEDGER":"on"}')
        with pytest.raises(AssertionError):
            _assert_no_ledger_request_content([mutated_request.read_bytes()])


@pytest.mark.xdist_group("ledger_delivery")
def test_off_and_unset_summaries_and_engine_journal_have_no_ledger_fields(
        parity_workspace, monkeypatch, tmp_path):
    traces = {name: _parity_trace(parity_workspace, name, switches) for name, switches in (
        ("unset", {}), ("off", {LEDGER_SWITCH: "off"}), ("on", {LEDGER_SWITCH: "on"}))}
    unset, off, on = (traces[name] for name in ("unset", "off", "on"))

    for record in (unset, off):
        _assert_no_ledger_fields(record["summary"])
        _assert_no_ledger_fields(record["events"])
    assert "required_test_ledger" in on["summary"]
    assert set(on["summary"]) - set(off["summary"]) == {"required_test_ledger"}
    assert on["summary"]["required_test_ledger"]["state"] == "unknown"
    assert on["summary"]["required_test_ledger"]["reason"]
    assert all("required_test_ledger" not in json.dumps(event).lower() for event in on["events"])
    assert all(not row["child_environment"]["switch_present"] for row in traces.values())

    from stateframe.parallel_worker import _child_environment

    monkeypatch.setenv(LEDGER_SWITCH, "on")
    parallel_environment = _child_environment({"environment": {LEDGER_SWITCH: "on"}})
    assert LEDGER_SWITCH not in parallel_environment

    # A planted summary field is rejected by the no-field assertion.
    mutated_path = tmp_path / "summary-mutated.json"
    mutated = json.loads(json.dumps(off["summary"]))
    mutated["required_test_ledger"] = {"state": "unknown"}
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_no_ledger_fields(json.loads(mutated_path.read_text(encoding="utf-8")))


@pytest.mark.xdist_group("ledger_delivery")
def test_incomplete_baseline_keeps_its_output_evidence_out_of_ledger_stderr_and_requests_equal(
        tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo, {
        "pkg/value.py": "VALUE = 3\n",
        "README.md": "BASELINE_OUTPUT_SENTINEL",
        "test_signal.py": textwrap.dedent("""
            from pathlib import Path
            import sys

            def pytest_generate_tests(metafunc):
                if metafunc.function.__name__ == "test_case":
                    marker = Path("README.md").read_text(encoding="utf-8")
                    metafunc.parametrize("value", [0, 1], ids=[marker, "later"])

            def test_case(value):
                if value == 0:
                    print(Path("README.md").read_text(encoding="utf-8"), file=sys.stderr)
                    assert False
        """),
    })
    test_cmd = (f"{sys.executable} -m pytest -q -x --basetemp=.test-tmp/pytest-base "
                "test_signal.py")
    traces = {}
    for label, switch in (("baseline-off", "off"), ("baseline-on", "on")):
        traces[switch] = _run_scripted(
            tmp_path, repo, switches={LEDGER_SWITCH: switch}, files=_read_steps(),
            task="Read pkg/value.py and summarize its value.", label=label,
            test_cmd=test_cmd, protect_tests="off", expected_returncode=1)

    off, on = traces["off"], traces["on"]
    assert on["requests"] == off["requests"]
    assert any(b"BASELINE_OUTPUT_SENTINEL" in request for request in on["requests"])
    assert on["summary"]["required_test_ledger"] == {
        "missing": [], "failed": [], "passed": [], "state": "unknown",
        "reason": "baseline: pytest did not finish every collected test; "
                  "current: pytest did not finish every collected test",
    }


@pytest.mark.parametrize(("witness", "ledger"), (
    ("off", "off"), ("off", "on"), ("on", "off"), ("on", "on"),
))
@pytest.mark.xdist_group("ledger_delivery")
def test_pairs_with_another_default_on_feature(parity_workspace, witness, ledger):
    trace = _witness_trace(parity_workspace, witness, ledger)
    assert trace["summary"]["succeeded"] is True

    if ledger == "on":
        off = _witness_trace(parity_workspace, witness, "off")
        _assert_equal_partner(off, trace)
    if witness == "on" and ledger == "on":
        mutated_path = parity_workspace["tmp_path"] / "partner-request-mutated.json"
        mutated_path.write_bytes(trace["requests"][0] + b" extra request")
        mutated = dict(trace)
        mutated["requests"] = [*mutated["requests"], mutated_path.read_bytes()]
        off = _witness_trace(parity_workspace, "on", "off")
        with pytest.raises(AssertionError):
            _assert_equal_partner(off, mutated)


def _host_check(repo: Path, *, required: bool, command: str | None = None,
                timeout_s: float = 30) -> dict[str, object]:
    return outer_checks.run(
        command or f"{sys.executable} -m pytest -q test_signal.py", repo, python=sys.executable,
        test_python=sys.executable, runner_kind="pytest", capture_output=True,
        timeout_s=timeout_s, required_test_ledger=required)


def _run_helper_packet_fallback(monkeypatch, tmp_path: Path, *, timed_out: bool = False) -> dict[str, object]:
    import kernel_execution_policy as policy

    repo = tmp_path / ("packet-helper-timeout-repo" if timed_out else "packet-helper-repo")
    _write_repo(repo, {"test_signal.py": "def test_signal():\n    assert 4 == 4\n"})

    def fake_run_sandboxed(_command, cwd, **_kwargs):
        report_path = next((Path(cwd) / ".test-tmp").glob(".stateframe-pytest-*"))
        report_path.write_text(json.dumps({
            "exit_code": 0, "outcomes": {"test_signal.py::test_signal": "passed"},
        }), encoding="utf-8")
        completed = type("Completed", (), {
            "returncode": -9 if timed_out else 0,
            "stdout": "" if timed_out else "x" * 1000,
            "stderr": "",
        })()
        return type("Result", (), {
            "completed": completed, "timed_out": timed_out, "execution_stop": None,
            "process_group_killed": False, "output_truncated": False,
        })()

    spec = {
        "cwd": str(repo), "command": f"{sys.executable} -m pytest -q",
        "test_env": {}, "test_python": sys.executable, "pytest_plugins": None,
        "runner_kind": "pytest", "required_test_ledger": True, "deadline": None,
        "stall_seconds": 30, "cpu_seconds": None, "capture_output": True,
    }
    captured = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(policy, "require_capabilities", lambda: None)
        patch.setattr(policy, "run_sandboxed", fake_run_sandboxed)
        patch.setattr(outer_checks, "WIRE_LIMIT", 512)
        patch.setattr(outer_checks.sys, "stdin", io.StringIO(json.dumps(spec)))
        with redirect_stdout(captured):
            assert outer_checks._helper() == 0
    records = [json.loads(line) for line in captured.getvalue().splitlines()]
    return next(record["result"] for record in records if "result" in record)


def test_every_ledger_failure_is_visible_as_unknown_with_a_reason(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    _write_repo(repo, {"test_signal.py": textwrap.dedent("""
        from pathlib import Path

        def test_signal():
            reports = list((Path.cwd() / ".test-tmp").glob(".stateframe-pytest-*"))
            if reports:
                reports[0].unlink()
                reports[0].mkdir()
            assert 4 == 4
    """)})
    ordinary = _host_check(repo, required=False)
    observed = _host_check(repo, required=True)

    assert (observed["passed"], observed["exit_code"]) == (ordinary["passed"], ordinary["exit_code"])
    assert _normalized_check_output(observed["output"]) == _normalized_check_output(ordinary["output"])
    assert observed["required_test_report"] == {"complete": False, "outcomes": {}}
    _assert_failure_reason_visible(observed)
    assert "STATEFRAME_REQUIRED_TEST_LEDGER" not in observed["tail"]
    mutated_result_path = tmp_path / "check-result-mutated.json"
    mutated_result = dict(observed)
    mutated_result.pop("required_test_report_reason", None)
    mutated_result["tail"] = ""
    mutated_result["output"] = ""
    mutated_result_path.write_text(json.dumps(mutated_result), encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_failure_reason_visible(json.loads(mutated_result_path.read_text(encoding="utf-8")))

    from kernel_required_tests import (MAX_PYTEST_REPORT_BYTES, RequiredTests,
                                       parse_pytest_report_with_reason)

    incomplete, parse_reason = parse_pytest_report_with_reason(b"{", 0)
    assert incomplete == {"complete": False, "outcomes": {}}
    assert parse_reason == "pytest report was not produced"
    invalid_reports = (
        (None, 0),
        (b" " * (MAX_PYTEST_REPORT_BYTES + 1), 0),
        (b"[]", 0),
        (b'{"exit_code":1,"outcomes":{}}', 0),
        (b'{"exit_code":0,"outcomes":{"x":"unknown"}}', 0),
        (b'{"exit_code":0,"outcomes":{"x":"failed"}}', 0),
    )
    for invalid_report, exit_code in invalid_reports:
        rejected, reason = parse_pytest_report_with_reason(invalid_report, exit_code)
        assert rejected == {"complete": False, "outcomes": {}}
        assert isinstance(reason, str) and reason
    with pytest.raises(ValueError, match="valid JSON"):
        RequiredTests.from_json("{")

    node = "test_signal.py::test_signal"
    finish_cases = (
        ("incomplete", {node}, {node: "passed"}, {node}, True, False, 0, 1024, False,
         "session evidence was incomplete"),
        ("collect-only", {node}, {node: "passed"}, {node}, False, True, 0, 1024, False,
         "collection-only mode"),
        ("no-selection", None, {}, set(), False, False, 0, 1024, False,
         "did not report a collected test set"),
        ("not-finished", {node}, {node: "passed"}, set(), False, False, 0, 1024, False,
         "did not finish every collected test"),
        ("exit-status", {node}, {node: "passed"}, {node}, False, False, 2, 1024, False,
         "unsupported exit status"),
        ("outside-identity", {"../outside/test_signal.py::test_signal"},
         {"../outside/test_signal.py::test_signal": "passed"},
         {"../outside/test_signal.py::test_signal"}, False, False, 0, 1024, False,
         "identity was outside"),
        ("duplicate-identity", {"pkg/../test_signal.py::test_signal", node},
         {"pkg/../test_signal.py::test_signal": "passed", node: "passed"},
         {"pkg/../test_signal.py::test_signal", node}, False, False, 0, 1024, False,
         "identity was outside"),
        ("oversize", {node}, {node: "passed"}, {node}, False, False, 0, 0, False,
         "report exceeded its size bound"),
        ("write-error", {node}, {node: "passed"}, {node}, False, False, 0, 1024, True,
         "report write failed: IsADirectoryError"),
    )
    for (label, selected, outcomes, finished, incomplete_state, collectonly, exit_status,
         report_limit, report_is_directory, expected_reason) in finish_cases:
        diagnostic, report = _embedded_session_finish_reason(
            tmp_path, monkeypatch, capsys, label, selected=selected, outcomes=outcomes,
            finished=finished, incomplete=incomplete_state, collectonly=collectonly,
            exit_status=exit_status, report_limit=report_limit,
            report_is_directory=report_is_directory)
        assert diagnostic == ""
        expected_code = {
            "incomplete": "session_incomplete",
            "collect-only": "collection_only",
            "no-selection": "no_collected_test_set",
            "not-finished": "tests_unfinished",
            "exit-status": "unsupported_exit_status",
            "outside-identity": "invalid_test_identity",
            "duplicate-identity": "invalid_test_identity",
            "oversize": "report_too_large",
            "write-error": None,
        }[label]
        if expected_code is None:
            assert report is None
        else:
            assert report == {"incomplete_reason": expected_code}

    import builtins
    from stateframe import runner

    original_import = builtins.__import__

    def fail_ledger_import(name, *args, **kwargs):
        if name.endswith("kernel_required_tests"):
            raise ImportError("injected ledger import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ledger_import)
    unavailable, import_reason = runner._new_required_test_ledger(True)
    import_report = runner._required_test_ledger_summary(unavailable, import_reason)
    assert import_report["state"] == "unknown"
    assert "ImportError" in import_report["reason"]

    class BrokenLedger:
        def report(self):
            return {"missing": [], "failed": [], "passed": [], "state": "unknown"}

        def observe(self, *_args, **_kwargs):
            raise ValueError("injected observation failure")

    unchanged_check = {"passed": False, "exit_code": 7,
                       "required_test_report": {"complete": True, "outcomes": {}}}
    original_ledger = BrokenLedger()
    broken, observe_reason = runner._observe_required_test_run(
        original_ledger, "current", dict(unchanged_check))
    assert broken is original_ledger
    observe_report = runner._required_test_ledger_summary(broken, observe_reason)
    assert observe_report["state"] == "unknown"
    assert "ValueError" in observe_report["reason"]
    assert unchanged_check["passed"] is False and unchanged_check["exit_code"] == 7
    monkeypatch.setattr(builtins, "__import__", original_import)

    # Missing the pytest report setup is advisory and preserves the normal check result.
    unsupported = f'{sys.executable} -c "print(\'check-value\')"'
    plain = _host_check(repo, required=False, command=unsupported)
    unsupported_result = _host_check(repo, required=True, command=unsupported)
    assert (unsupported_result["passed"], unsupported_result["exit_code"]) == (
        plain["passed"], plain["exit_code"])
    assert unsupported_result["required_test_report"] == {"complete": False, "outcomes": {}}
    assert unsupported_result["required_test_report_reason"] == "pytest report was not produced"

    # A symlinked scratch directory forces the setup fallback without changing pytest's result.
    symlink_repo = tmp_path / "symlink-repo"
    _write_repo(symlink_repo, {"test_signal.py": "def test_signal():\n    assert 4 == 4\n"})
    external_scratch = tmp_path / "external-scratch"
    external_scratch.mkdir()
    (symlink_repo / ".test-tmp").symlink_to(external_scratch, target_is_directory=True)
    symlink_plain = _host_check(symlink_repo, required=False)
    symlink_result = _host_check(symlink_repo, required=True)
    assert (symlink_result["passed"], symlink_result["exit_code"]) == (
        symlink_plain["passed"], symlink_plain["exit_code"])
    assert symlink_result["required_test_report_reason"] == "pytest report was not produced"

    # A failing pytest import falls back to the original command and retains its exit status.
    import_repo = tmp_path / "import-repo"
    _write_repo(import_repo, {
        "pytest.py": "raise RuntimeError('injected import failure')\n",
        "test_signal.py": "def test_signal():\n    assert 4 == 4\n",
    })
    import_plain = _host_check(import_repo, required=False)
    import_result = _host_check(import_repo, required=True)
    assert (import_result["passed"], import_result["exit_code"]) == (
        import_plain["passed"], import_plain["exit_code"])
    assert "pytest import failed" not in import_result["tail"]
    assert import_result["required_test_report_reason"] == "pytest could not be imported"

    # Replacing the report with a symlink forces the outer read exception.
    read_repo = tmp_path / "read-repo"
    _write_repo(read_repo, {"test_signal.py": textwrap.dedent("""
        from pathlib import Path

        def test_signal():
            reports = list((Path.cwd() / ".test-tmp").glob(".stateframe-pytest-*"))
            if reports:
                report = reports[0]
                target = report.with_name("report-target.json")
                report.unlink()
                report.symlink_to(target)
            assert 4 == 4
    """)})
    read_plain = _host_check(read_repo, required=False)
    read_result = _host_check(read_repo, required=True)
    assert (read_result["passed"], read_result["exit_code"]) == (
        read_plain["passed"], read_plain["exit_code"])
    assert read_result["required_test_report_reason"] == "pytest report was not produced"

    # A complete report plus forced packet pressure takes the bounded fallback.
    packet_result = _run_helper_packet_fallback(monkeypatch, tmp_path)
    assert packet_result["passed"] is True and packet_result["exit_code"] == 0
    assert packet_result["required_test_report"] == {"complete": False, "outcomes": {}}
    assert packet_result["required_test_report_reason"] == "pytest report exceeded its size bound"

    timeout_result = _run_helper_packet_fallback(monkeypatch, tmp_path, timed_out=True)
    assert timeout_result["passed"] is False and timeout_result["exit_code"] is None
    assert timeout_result["required_test_report"] == {"complete": False, "outcomes": {}}
    assert timeout_result["required_test_report_reason"] == "pytest report was not produced"

    # A stalled pytest invocation follows the stop fallback and keeps the check outcome.
    stop_repo = tmp_path / "stop-repo"
    _write_repo(stop_repo, {
        "test_signal.py": "import time\ndef test_signal():\n    time.sleep(5)\n",
    })
    stop_plain = _host_check(stop_repo, required=False, timeout_s=2)
    stop_result = _host_check(stop_repo, required=True, timeout_s=2)
    assert (stop_result["passed"], stop_result["exit_code"]) == (
        stop_plain["passed"], stop_plain["exit_code"])
    assert stop_result["required_test_report_reason"] == "pytest report was not produced"


def _load_mutated_required_tests(tmp_path: Path):
    source = ROOT / "src/stateframe/engine/kernel_required_tests.py"
    copied = tmp_path / "kernel_required_tests_mutated.py"
    text = source.read_text(encoding="utf-8")
    original = "        if not complete:\n            return\n"
    assert text.count(original) == 1
    copied.write_text(text.replace(original, "        if False:\n            return\n", 1), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("kernel_required_tests_mutated", copied)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RequiredTests


def test_report_fires_only_on_a_seeded_regression(tmp_path):
    from kernel_required_tests import RequiredTests

    ledger = RequiredTests()
    baseline = dict(zip(TEST_IDS, ("passed", "passed", "passed")))
    ledger.observe("baseline", complete=True, outcomes=baseline)
    ledger.observe("current", complete=True, outcomes={
        TEST_IDS[1]: "failed", TEST_IDS[2]: "passed",
    })
    assert ledger.report() == {
        "missing": [TEST_IDS[0]], "failed": [TEST_IDS[1]],
        "passed": [TEST_IDS[2]], "state": "regressed",
    }

    intact = RequiredTests()
    intact.observe("baseline", complete=True, outcomes=baseline)
    intact.observe("current", complete=True, outcomes=baseline)
    assert intact.report() == {
        "missing": [], "failed": [], "passed": sorted(TEST_IDS), "state": "ok",
    }

    expected_unknown = {"missing": [], "failed": [], "passed": [], "state": "unknown"}
    MutatedRequiredTests = _load_mutated_required_tests(tmp_path)
    mutated = MutatedRequiredTests()
    mutated.observe("baseline", complete=True, outcomes={TEST_IDS[0]: "passed"})
    mutated.observe("current", complete=False, outcomes={TEST_IDS[0]: "failed"})
    with pytest.raises(AssertionError):
        assert mutated.report() == expected_unknown

    original_source = (
        "def test_removed():\n    assert 3 == 3\n\n"
        "def test_broken():\n    assert 5 == 5\n\n"
        "def test_control():\n    assert 2 == 2\n"
    )
    changed_source = (
        "def test_broken():\n    assert 5 == 6\n\n"
        "def test_control():\n    assert 2 == 2\n"
    )
    repo = tmp_path / "repo"
    _write_repo(repo, {
        "tests/test_signal.py": original_source,
    })
    command = f"{sys.executable} -m pytest -q tests/test_signal.py"
    baseline_check = _host_check(repo, required=True, command=command)
    assert baseline_check["passed"] is True and baseline_check["exit_code"] == 0, baseline_check
    ledger.observe("baseline", **baseline_check["required_test_report"])
    (repo / "tests/test_signal.py").write_text(changed_source, encoding="utf-8")
    current_check = _host_check(repo, required=True, command=command)
    assert current_check["passed"] is False and current_check["exit_code"] == 1, current_check
    ledger.observe("current", **current_check["required_test_report"])
    assert ledger.report() == {
        "missing": [TEST_IDS[0]], "failed": [TEST_IDS[1]],
        "passed": [TEST_IDS[2]], "state": "regressed",
    }
