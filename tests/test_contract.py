from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from _timeouts import scaled
from fixtures.process_signals import (
    record_identity,
    signal_run_owned,
    signal_run_verified,
    wait_for_checkpoint_saved,
)

from stateframe import cli
from stateframe.contract import (
    assemble, cli_snapshot, commands, compare, document as contract_document, events,
    outputs, readable_summary, render, schema_version,
)
from stateframe.scripted_endpoint import Script, serve


def _parser_nodes(parser):
    root = parser.prog.replace("\\", "/").rsplit("/", 1)[-1]
    result = [(root, parser)]

    def visit(parent, prefix):
        for action in parent._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in sorted(action.choices.items()):
                    path = (*prefix, name)
                    result.append((" ".join(path), child))
                    visit(child, path)

    visit(parser, ())
    return result


def _field_rows(value):
    return value if isinstance(value, list) else []


def _kind_type(name, value):
    if name.endswith("|null"):
        return value is None or _kind_type(name[:-5], value)
    if name.startswith("array<") and name.endswith(">"):
        return isinstance(value, list) and all(_kind_type(name[6:-1], item) for item in value)
    expected = {"array": list, "object": dict, "string": str, "integer": int,
                "number": (int, float), "boolean": bool}
    if name not in expected:
        raise AssertionError(f"unrecognized contract type: {name}")
    if name == "integer":
        return type(value) is int
    if name == "number":
        return type(value) in (int, float)
    if name == "boolean":
        return type(value) is bool
    return isinstance(value, expected[name])


def _path_values(value, path):
    values = [value]
    for part in path.split("."):
        next_values = []
        for item in values:
            if part.endswith("[]"):
                child = item.get(part[:-2]) if isinstance(item, dict) else None
                if isinstance(child, list):
                    next_values.extend(child)
            elif part.startswith("{") and part.endswith("}"):
                if isinstance(item, dict):
                    next_values.extend(item.values())
            elif isinstance(item, dict) and part in item:
                next_values.append(item[part])
        values = next_values
    return values


def _artifact_records(run, artifact):
    pattern = (artifact["file"].replace("NNNN", "????").replace("<call>", "*")
               .replace("<unit>", "*").replace("<n>", "*").replace("<sha256>", "*"))
    records = []
    for path in run.glob(pattern):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if artifact["format"] == "json":
            try:
                records.append(json.loads(text))
            except ValueError:
                continue
        elif artifact["format"] == "jsonl":
            for line in text.splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
        elif artifact["format"] == "prefixed-jsonl":
            for line in text.splitlines():
                if line.startswith("[sf] "):
                    try:
                        records.append(json.loads(line[5:]))
                    except ValueError:
                        continue
        elif artifact["format"] == "text":
            records.append({"__present__": True})
    return records


_ARTIFACT_TOKENS = {
    "NNNN": r"\d{4}",
    "<call>": r"[^/]+",
    "<unit>": r"[^/]+",
    "<n>": r"\d+",
    "<sha256>": r"[0-9a-f]{64}",
    "<id>": r"[A-Za-z0-9_-]+",
    "<path>": r".+",
}


def _artifact_pattern(file):
    assert file not in {"*", "**", "**/*"}
    pattern = re.escape(file)
    for token, expression in _ARTIFACT_TOKENS.items():
        pattern = pattern.replace(re.escape(token), expression)
    assert all(part not in {"*", "**"} for part in file.split("/"))
    return re.compile(pattern)


def _matches_artifact(file, artifact):
    return _artifact_pattern(artifact["file"]).fullmatch(file) is not None


def _parser_action(parser, command, option):
    command_parser = dict(_parser_nodes(parser))[command]
    return next(action for action in command_parser._actions if option in action.option_strings)


def test_commands_cover_real_argparse_tree_and_flags(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ENDPOINT", "https://private.invalid/model")
    monkeypatch.setenv("STATEFRAME_MODEL", "private-model")
    parser = cli.build_parser()
    result = commands(parser)
    by_name = {row["name"]: row for row in result}
    nodes = _parser_nodes(parser)

    assert [row["name"] for row in result] == sorted(by_name)
    assert set(by_name) == {name for name, _ in nodes}
    assert "project-hook inspect" not in by_name
    assert by_name["contract"]["json_output"] is True
    for name, command_parser in nodes:
        row = by_name[name]
        actual = {flag["name"]: flag for flag in row["flags"]}
        expected = {
            option: action.nargs != 0
            for action in command_parser._actions
            for option in action.option_strings
        }
        assert {flag: item["takes_value"] for flag, item in actual.items()} == expected
        for action in command_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            for option in action.option_strings:
                item = actual[option]
                assert item["required"] is bool(
                    action.required or getattr(action, "contract_required_unless_env", None))
                assert item["repeatable"] is isinstance(
                    action, (argparse._AppendAction, argparse._AppendConstAction,
                             argparse._CountAction))
                expected_choices = action.choices
                if isinstance(expected_choices, dict):
                    expected_choices = sorted(expected_choices)
                elif expected_choices is not None:
                    expected_choices = list(expected_choices)
                assert item["choices"] == expected_choices
                assert "default" in item and isinstance(item["type"], str)
        assert all(name in item["commands"] for item in actual.values())
        assert all(isinstance(item["commands"], list) for item in actual.values())
        assert row["json_output"] is any(
            action.dest == "json" and "--json" in action.option_strings
            for action in command_parser._actions
        )
        promised = row["promised_json"]
        if promised is None:
            assert name not in {"inspect", "resume"}
        else:
            assert isinstance(promised["flags"], list) and promised["flags"]
            assert set(promised["flags"]).issubset(expected)
            assert all((field.get("stable") is False or field.get("stable") is True
                        or field.get("stable") == "type")
                       and (field.get("optional") is False or field.get("optional") is True)
                       for field in promised["fields"])
        assert "exit_codes" in row and "promised_lines" in row
    assert by_name["inspect"]["promised_json"]["flags"] == ["--send", "--frame"]
    assert by_name["resume"]["promised_json"]["flags"] == ["--json"]
    flags_by_name = {
        row["name"]: row for row in by_name["run"]["flags"]
    }
    assert flags_by_name["--repo"]["required"] is True
    assert flags_by_name["--repo"]["type"] == "path"
    assert flags_by_name["--runner-kind"]["choices"] == ["auto", "pytest", "generic"]
    assert flags_by_name["--runner-kind"]["default"] == "auto"
    assert flags_by_name["--check"]["repeatable"] is True
    assert flags_by_name["--check"]["default"] == []
    assert flags_by_name["--endpoint"]["default"] is None
    assert flags_by_name["--endpoint"]["default_kind"] == "runtime"
    assert flags_by_name["--endpoint"]["required"] is True
    assert flags_by_name["--endpoint"]["required_unless_env"] == "STATEFRAME_ENDPOINT"
    assert flags_by_name["--model"]["required_unless_env"] == "STATEFRAME_MODEL"
    assert "required_unless_env" not in flags_by_name["--repo"]
    assert by_name["resume"]["positionals"] == [{
        "name": "run_dir", "required": True, "repeatable": False, "choices": None,
        "default": None, "type": "path",
    }]
    assert by_name["stateframe"]["positionals"][0]["name"] == "command"
    assert by_name["contract"]["mutually_exclusive_groups"] == [{
        "required": False, "members": ["--json", "--schema-version"],
    }]
    snapshot = [{
        "name": row["name"],
        "flags": " ".join(flag["name"] + ("=" if flag["takes_value"] else "")
                           for flag in row["flags"]),
        "exit_codes": [code["code"] for code in row["exit_codes"]],
    } for row in result]
    assert snapshot == cli_snapshot()


@pytest.mark.parametrize(("command", "option", "attribute", "value", "field", "expected"), [
    ("run", "--runner-kind", "choices", ("auto", "pytest", "generic", "custom"),
     "choices", ["auto", "pytest", "generic", "custom"]),
    ("run", "--runner-kind", "default", "generic", "default", "generic"),
    ("run", "--repo", "required", False, "required", False),
])
def test_command_document_tracks_parser_metadata_changes(
    command, option, attribute, value, field, expected,
):
    parser = cli.build_parser()
    before = commands(parser)
    action = _parser_action(parser, command, option)
    setattr(action, attribute, value)

    after = commands(parser)
    before_row = next(row for row in before if row["name"] == command)
    after_row = next(row for row in after if row["name"] == command)
    before_flag = next(row for row in before_row["flags"] if row["name"] == option)
    after_flag = next(row for row in after_row["flags"] if row["name"] == option)
    assert before_flag[field] != expected
    assert after_flag[field] == expected


def test_count_action_is_repeatable():
    parser = argparse.ArgumentParser(prog="stateframe")
    parser.add_argument("--verbose", action="count", default=0)
    row = next(row for row in commands(parser) if row["name"] == "stateframe")
    flag = next(flag for flag in row["flags"] if flag["name"] == "--verbose")
    assert flag["repeatable"] is True


def test_contract_json_is_environment_independent_and_redacts_runtime_values(tmp_path):
    source = str(Path(__file__).resolve().parents[1] / "src")
    outputs = []
    cases = (
        ("case-a", "21", "C", "1", "https://private-a.invalid/model", "private-model-a"),
        ("case-b", "132", "C.UTF-8", "73", "https://private-b.invalid/model", "private-model-b"),
    )
    for name, width, locale, seed, endpoint, model in cases:
        cwd = tmp_path / name
        cwd.mkdir()
        home = tmp_path / f"{name}-home"
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": source,
            "HOME": str(home),
            "STATEFRAME_HOME": str(tmp_path / f"{name}-state"),
            "STATEFRAME_ENDPOINT": endpoint,
            "STATEFRAME_MODEL": model,
            "COLUMNS": width,
            "LC_ALL": locale,
            "PYTHONHASHSEED": seed,
        })
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "contract", "--json"],
            cwd=cwd, env=env, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        assert endpoint.encode() not in result.stdout
        assert model.encode() not in result.stdout
        assert str(home).encode() not in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_contract_cli_json_summary_and_schema_version(capsys):
    parser = cli.build_parser()
    expected = contract_document(parser)

    assert cli.main(["contract", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == render(expected).decode("utf-8")
    assert json.loads(captured.out) == expected
    assert expected["settings"] == []
    assert expected["notes"]["settings"] == "pending"
    assert all(not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).is_absolute()
               for value in _all_strings(expected))

    assert cli.main(["contract"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == readable_summary(expected) + "\n"
    assert len(captured.out.splitlines()) == 1

    assert cli.main(["contract", "--schema-version"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == schema_version() + "\n"
    assert captured.out.strip().isdigit()


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def test_contract_command_imports_contract_only_when_selected():
    code = (
        "import sys; from stateframe.cli import main; "
        "assert 'stateframe.contract' not in sys.modules; "
        "result = main(['contract', '--schema-version']); "
        "assert result == 0 and 'stateframe.contract' in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == b"1\n"


def test_assembly_and_render_are_order_independent_and_path_safe():
    settings = [
        {"key": "zeta", "section": "main", "class": "user", "type": "bool",
         "choices": None, "default": False, "default_measured": True, "env": None,
         "flags": [], "aliases": [], "since": None, "removed": None, "meaning": "toggle"},
        {"key": "alpha", "section": "main", "class": "advanced", "type": "enum",
         "choices": ["b", "a"], "default": "a", "default_measured": True, "env": None,
         "flags": [{"name": "--z", "commands": ["run"], "takes_value": True},
                   {"name": "--a", "commands": ["run"], "takes_value": False}],
         "aliases": [], "since": None, "removed": None, "meaning": "select"},
    ]
    command_rows = commands(cli.build_parser())
    output_rows, event_rows = outputs(), events()
    assert all(isinstance(row["since"], str) for row in output_rows + event_rows)
    all_fields = [field for row in output_rows + event_rows for field in row["fields"]]
    all_fields.extend(field for command in commands(cli.build_parser())
                      if command["promised_json"]
                      for field in command["promised_json"]["fields"])
    assert all(isinstance(field["type"], str) and isinstance(field["since"], str)
               and (field["stable"] is True or field["stable"] is False or field["stable"] == "type")
               and (field["optional"] is True or field["optional"] is False)
               for field in all_fields)
    first = assemble(settings=settings, commands=command_rows, outputs=output_rows, events=event_rows)
    second = assemble(list(reversed(settings)), list(reversed(command_rows)),
                      list(reversed(output_rows)), list(reversed(event_rows)))

    assert render(first) == render(second)
    assert json.loads(render(first)) == first
    assert list(first) == sorted(first)

    def check_strings(value):
        if isinstance(value, str):
            assert not PurePosixPath(value).is_absolute()
            assert not PureWindowsPath(value).is_absolute()
        elif isinstance(value, dict):
            for key, child in value.items():
                check_strings(key)
                check_strings(child)
        elif isinstance(value, list):
            for child in value:
                check_strings(child)

    check_strings(first)
    absolute_default = str(Path.cwd() / "private-config")
    with pytest.raises(ValueError, match="absolute paths"):
        assemble([{"key": "path", "type": "path", "default": absolute_default}], [], [], [])


def _case_environment(monkeypatch, root):
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(root / "state"))
    monkeypatch.delenv("STATEFRAME_ALLOW_REMOTE_ENDPOINT", raising=False)


def _run_case(tmp_path, name, files, steps, monkeypatch, capsys, *, flags=(), check=None,
              task="Check the sample repository.", script_class=Script, route_read=False):
    root = tmp_path / name
    repo = root / "repo"
    repo.mkdir(parents=True)
    _case_environment(monkeypatch, root)
    for name, body in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    run_dir = root / "run"
    script_document = {"model": "scripted", "tokenize": {"mode": "ok"}, "steps": steps}
    if route_read:
        script_document["models"] = ["scripted-read"]
    script = script_class(script_document)
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    shell_flags = () if "--yolo" in flags else ("--shell", "standard")
    argv = ["run", "--repo", str(repo), "--task", task, "--endpoint", endpoint,
            "--model", "scripted", "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--max-turns", "0", "--no-run-env", "--keep-checkpoints", "--no-instructions",
            "--no-project-hooks", *shell_flags, "--quiet", "--json", *flags]
    if route_read:
        argv.extend(["--route", f"read.endpoint={endpoint}", "--route", "read.model=scripted-read"])
    if check is not None:
        argv.extend(check)
    try:
        code = cli.main(argv)
        capsys.readouterr()
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    assert (run_dir / "summary.json").is_file()
    return code, run_dir, script, endpoint


def _run_process_child(parent):
    child_file = Path(f"/proc/{parent}/task/{parent}/children")
    for value in child_file.read_text(encoding="ascii").split():
        pid = int(value)
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"_child.py" in command or b"stateframe._child" in command:
            return pid
    raise AssertionError("the scripted run did not launch its engine child")


def _resume_case(tmp_path, monkeypatch, capsys, *, run_name="resume"):
    root = tmp_path / run_name
    repo = root / "repo"
    repo.mkdir(parents=True)
    _case_environment(monkeypatch, root)
    (repo / "app.py").write_text("assert 1 == 1\n", encoding="utf-8")
    run_dir = root / "run"
    entered, release = threading.Event(), threading.Event()

    class PausingScript(Script):
        def next_call(self, offered):
            if len(self.requests) == 2 and not entered.is_set():
                entered.set()
                release.wait()
            return super().next_call(offered)

    script = PausingScript({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "shell", "arguments": {"command": "printf checkpointed"}},
        {"tool": "done", "arguments": {"summary": "The check passed."}},
    ]})
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Run a small shell command before finishing.", "--endpoint", endpoint,
               "--model", "scripted", "--run-dir", str(run_dir), "--test-python", sys.executable,
               "--check-cmd", "true", "--max-turns", "0", "--no-run-env", "--keep-checkpoints",
               "--no-instructions", "--no-project-hooks", "--shell", "standard", "--quiet", "--json"]
    launched_after = time.time()
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        while not entered.is_set():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError("run ended before the pause point: " + stdout + stderr)
            entered.wait(0.1)
        checkpoint_deadline = time.monotonic() + scaled(120)
        wait_for_checkpoint_saved(run_dir, deadline=checkpoint_deadline,
                                  what="scripted contract resume", process=process)
        child = _run_process_child(process.pid)
        identity = record_identity(child, launched_after=launched_after,
                                   command_fragment="_child.py")
        signal_run_verified(run_dir, child, identity, signal.SIGKILL,
                            deadline=checkpoint_deadline, what="scripted contract resume")
        release.set()
        process.wait()
        stdout, stderr = process.communicate()
        assert process.returncode != 0, stdout + stderr
        with script.lock:
            script.position = 1
            script._states[1] = "pending"
        code = cli.main(["resume", str(run_dir), "--endpoint", endpoint, "--model", "scripted",
                         "--max-turns", "0", "--keep-checkpoints", "--json"])
        resumed_json = json.loads(capsys.readouterr().out)
        assert code == 0, resumed_json
        script.assert_consumed()
        return run_dir, resumed_json
    finally:
        release.set()
        if process.poll() is None:
            signal_run_owned(run_dir, process, signal.SIGKILL,
                             deadline=time.monotonic(), what="scripted contract cleanup",
                             require_checkpoint_saved=False, group=True)
            process.wait()
            process.communicate()
        server.shutdown()
        server.server_close()


def _chat_case(tmp_path, monkeypatch):
    from stateframe import chat, runner
    from stateframe.endpoint import Endpoint

    root = tmp_path / "chat"
    repo = root / "repo"
    repo.mkdir(parents=True)
    _case_environment(monkeypatch, root)
    (repo / "app.py").write_text("assert True\n", encoding="utf-8")
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "done", "arguments": {"summary": "The sample is ready."}},
    ]})
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        options = runner.RunOptions(
            repo=repo, task="", endpoint=Endpoint.from_args(endpoint, "scripted", None),
            max_turns=0, check_cmd="true", runner_kind="generic", test_python=sys.executable,
            no_project_hooks=True, instructions="off", keep_checkpoints=True)
        session = chat.Session(repo=repo, home=root / "session", base=options)
        session.start(printer=lambda *_: None)
        record = session.run_task("Check the sample repository.", printer=lambda *_: None)
        script.assert_consumed()
        return Path(record.run_dir)
    finally:
        server.shutdown()
        server.server_close()


def test_catalog_fields_and_tree_patterns_cover_scripted_runs(tmp_path, monkeypatch, capsys):
    hook_file = tmp_path / "hook-config.json"
    hook_file.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "start", "event": "session_start", "argv": [sys.executable, "-c", "pass"]},
    ]}), encoding="utf-8")
    plain_code, plain, plain_script, endpoint = _run_case(
        tmp_path, "plain", {"app.py": "assert True\n",
                             "tests/test_sample.py": "def test_sample():\n    assert True\n"}, [
            {"tool": "run_tests", "required": True, "arguments": {}},
            {"tool": "shell", "required": True, "arguments": {"command": "printf scripted"}},
            {"tool": "done", "arguments": {"summary": "The check passed."}},
        ], monkeypatch, capsys, check=["--test-cmd", f"{shlex.quote(sys.executable)} -m pytest -q",
                                       "--protect-tests", "off", "--hooks", str(hook_file)],
        task="Run the sample test and report the result.", route_read=True)
    assert plain_code == 0
    assert "routes" in json.loads((plain / "spec.json").read_text(encoding="utf-8"))

    failed_code, failed, _, _ = _run_case(
        tmp_path, "failing", {"app.py": "assert True\n",
                               "tests/test_failed.py": "def test_failed():\n    assert False\n"}, [
            {"tool": "done", "arguments": {"summary": "The check was run."}},
        ], monkeypatch, capsys, check=["--test-cmd", f"{shlex.quote(sys.executable)} -m pytest -q",
                                       "--protect-tests", "off"],
        task="Run the repository check and report its result.")
    assert failed_code != 0
    failed_spec = json.loads((failed / "spec.json").read_text(encoding="utf-8"))
    assert isinstance(failed_spec["public_row"]["test_wrong_failed_nodes"], dict)

    class TwoUnitScript(Script):
        def next_call(self, offered):
            if "subframe_blocked" in offered:
                return {"tool": "subframe_blocked", "arguments": {
                    "reason_code": "OTHER", "detail": "The scripted worker cannot complete this unit."}}
            return super().next_call(offered)

    parallel_code, parallel, parallel_script, _ = _run_case(
        tmp_path, "parallel", {"a.py": "VALUE = 1\n", "b.py": "VALUE = 2\n"}, [
            {"tool": "parallel", "arguments": {"subframes": [
                {"id": "a", "goal": "Read a.py and report its value.", "owns": ["a.py"]},
                {"id": "b", "goal": "Read b.py and report its value.", "owns": ["b.py"]},
            ]}},
            {"tool": "done", "arguments": {"summary": "Both subtasks returned."}},
        ], monkeypatch, capsys, flags=("--parallel", "2"), check=["--check-cmd", "true"],
        task="Complete two independent subtasks.", script_class=TwoUnitScript)
    assert isinstance(parallel_code, int)
    parallel_script.assert_consumed()
    parallel_results = list(parallel.glob("parallel/*/result.json"))
    assert len(parallel_results) == 1
    assert len(json.loads(parallel_results[0].read_text(encoding="utf-8"))["outcomes"]) == 2

    hook_trace_code, hook_trace, _, _ = _run_case(
        tmp_path, "hook-trace", {"app.py": "assert True\n"}, [
            {"tool": "done", "arguments": {"summary": "The check passed."}},
        ], monkeypatch, capsys, flags=("--hooks", str(hook_file)),
        check=["--check-cmd", "true"], task="Check a run with a hook.")
    assert isinstance(hook_trace_code, int)

    subframe_trace_code, subframe_trace, _, _ = _run_case(
        tmp_path, "subframe-trace", {"a.py": "VALUE = 1\n", "b.py": "VALUE = 2\n"}, [
            {"tool": "parallel", "arguments": {"subframes": [
                {"id": "a", "goal": "Read a.py and report its value.", "owns": ["a.py"]},
                {"id": "b", "goal": "Read b.py and report its value.", "owns": ["b.py"]},
            ]}},
            {"tool": "done", "arguments": {"summary": "Both subtasks returned."}},
        ], monkeypatch, capsys, flags=("--parallel", "2"), check=["--check-cmd", "true"],
        task="Complete two independent subtasks.", script_class=TwoUnitScript)
    assert isinstance(subframe_trace_code, int)
    assert list(subframe_trace.glob("parallel/*/result.json"))

    mutation_code, mutation, mutation_script, _ = _run_case(
        tmp_path, "mutation", {"app.py": "assert True\n", "report.txt": "row|41\n"}, [
            {"tool": "str_replace", "required": True,
             "arguments": {"path": "report.txt", "old_str": "row|41", "new_str": "row|42"}},
            {"tool": "mutate", "required": True,
             "arguments": {"op": "enter", "id": "sample_reader",
                           "args": json.dumps({"goal": "Read the report",
                                               "inputs": ["report.txt"]})}},
            {"tool": "run_tests", "required": True, "arguments": {}},
            {"tool": "mutate", "required": True, "arguments": {"op": "leave"}},
            {"tool": "done", "arguments": {"summary": "The report was read."}},
        ], monkeypatch, capsys, flags=("--mutations", "authored"),
        check=["--check-cmd", "true"],
        task="Update a report, open an authoring episode, run the repository check, and close it.")
    assert isinstance(mutation_code, int)
    mutation_script.assert_consumed()
    mutation_journal = [json.loads(line) for line in
                        (mutation / "mutations" / "journal.jsonl").read_text().splitlines()]
    kinds = [row["kind"] for row in mutation_journal]
    assert "episode_enter" in kinds and "episode_leave" in kinds
    inspect_code = cli.main(["inspect", str(mutation), "--send", "2", "--frame"])
    inspect_json = json.loads(capsys.readouterr().out)
    assert inspect_code == 0 and inspect_json["schema"] == "stateframe-f4/1"

    resumed, resume_json = _resume_case(tmp_path, monkeypatch, capsys)
    resumed_trace, _ = _resume_case(tmp_path, monkeypatch, capsys, run_name="resumed-trace")
    chat_run = _chat_case(tmp_path, monkeypatch)
    run_dirs = [plain, failed, parallel, mutation, resumed, chat_run,
                hook_trace, subframe_trace, resumed_trace]

    artifact_rows = outputs()
    for artifact in artifact_rows:
        _artifact_pattern(artifact["file"])
        if artifact["optional"]:
            condition = artifact.get("condition")
            assert isinstance(condition, str) and condition.endswith("."), artifact["file"]
            assert condition.count(".") == 1 and len(condition.split()) <= 16, artifact["file"]

    undeclared = []
    for run in run_dirs:
        for path in run.rglob("*"):
            if path.is_file():
                relative = path.relative_to(run).as_posix()
                if not any(_matches_artifact(relative, artifact) for artifact in artifact_rows):
                    undeclared.append(f"{run.name}/{relative}")
    assert not undeclared, f"run files missing from the artifact catalog: {sorted(undeclared)}"

    final_check = next(row for row in artifact_rows
                       if row["file"] == "parallel/<call>/checks/final/<sha256>.txt")
    final_check_witnessed = any(
        _matches_artifact(path.relative_to(run).as_posix(), final_check)
        for run in run_dirs for path in run.rglob("*") if path.is_file()
    )
    assert final_check_witnessed or final_check.get("stability") == "internal"

    for run in run_dirs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        engine_result = json.loads((run / "engine-result.json").read_text(encoding="utf-8"))
        assert isinstance(summary["runtime_outcome"], str)
        assert summary["requests"] is None or type(summary["requests"]) is int
        assert isinstance(engine_result["result"], dict)

    for artifact in artifact_rows:
        records = [record for run in run_dirs for record in _artifact_records(run, artifact)]
        if not records:
            assert artifact["optional"] is True, artifact["file"]
            continue
        for field in artifact["fields"]:
            values = [value for record in records for value in _path_values(record, field["name"])]
            if not values:
                assert field["optional"] is True, (artifact["file"], field["name"])
                continue
            assert all(_kind_type(field["type"], value) for value in values), (
                artifact["file"], field["name"], field["type"],
                sorted({type(value).__name__ for value in values}))

    log_artifact = next(row for row in outputs() if row["file"] == "engine.log")
    log_records = [record for run in run_dirs for record in _artifact_records(run, log_artifact)]
    for event in events():
        matching = [record for record in log_records if record.get("kind") == event["name"]]
        if not matching:
            assert event["name"] == "send_error"
            continue
        for field in event["fields"]:
            values = [value for record in matching for value in _path_values(record, field["name"])]
            if not values:
                assert field["optional"] is True, (event["name"], field["name"])
                continue
            assert all(_kind_type(field["type"], value) for value in values)

    commands_by_name = {row["name"]: row for row in commands(cli.build_parser())}
    promised = {"inspect": inspect_json, "resume": resume_json}
    for name, value in promised.items():
        for field in commands_by_name[name]["promised_json"]["fields"]:
            values = _path_values(value, field["name"])
            if not values:
                assert field["optional"] is True, (name, field["name"])
            else:
                assert all(_kind_type(field["type"], item) for item in values), (
                    name, field["name"], field["type"],
                    sorted({type(item).__name__ for item in values}))


def test_compare_reports_each_removal_and_repurpose_kind():
    previous = {
        "settings": [
            {"key": "removed-setting", "type": "string", "meaning": "old"},
            {"key": "kept-setting", "type": "string", "meaning": "kept", "env": "OLD_ENV",
             "flags": [{"name": "--old-setting", "takes_value": True}]},
        ],
        "commands": [
            {"name": "removed-command", "flags": []},
            {"name": "run", "json_output": False,
             "flags": [{"name": "--old-command", "takes_value": False}]},
        ],
        "outputs": [
            {"file": "removed.json", "format": "json", "fields": []},
            {"file": "summary.json", "format": "json",
             "fields": [{"name": "old-field", "type": "string"}]},
        ],
        "events": [
            {"name": "removed-event", "fields": []},
            {"name": "call", "fields": [{"name": "old-field", "type": "string"}]},
        ],
    }
    current = {
        "settings": [{"key": "kept-setting", "type": "string", "meaning": "changed"}],
        "commands": [{"name": "run", "json_output": True, "flags": []}],
        "outputs": [{"file": "summary.json", "format": "json", "fields": []}],
        "events": [{"name": "call", "fields": []}],
    }
    changes = compare(previous, current)
    assert {row["kind"] for row in changes} == {
        "setting", "setting_env", "setting_flag", "command", "command_flag",
        "output", "output_field", "event", "event_field",
    }
    assert any(row["kind"] == "setting" and row["change"] == "repurposed" for row in changes)
    assert any(row["kind"] == "command" and row["change"] == "repurposed" for row in changes)


def test_compare_reports_command_exit_code_changes():
    previous = {"settings": [], "commands": [{"name": "run", "exit_codes": [0, 1, 2]}],
                "outputs": [], "events": []}
    current = {"settings": [], "commands": [{"name": "run", "exit_codes": [0, 2]}],
               "outputs": [], "events": []}
    assert compare(previous, current) == [
        {"kind": "command_exit_codes", "name": "run", "change": "changed"},
    ]


def test_compare_accepts_setting_aliases_for_removed_spellings():
    previous = {
        "settings": [{"key": "mode", "type": "string", "meaning": "select mode", "env": "OLD_ENV",
                      "flags": [{"name": "--old", "takes_value": True}]}],
        "commands": [{"name": "run", "flags": [{"name": "--old", "takes_value": True}]}],
        "outputs": [], "events": [],
    }
    current = {
        "settings": [{"key": "mode", "type": "string", "meaning": "select mode", "env": "NEW_ENV",
                      "flags": [{"name": "--new", "takes_value": True}],
                      "aliases": [{"kind": "env", "name": "OLD_ENV"},
                                  {"kind": "flag", "name": "--old"}]}],
        "commands": [{"name": "run", "flags": [{"name": "--new", "takes_value": True}]}],
        "outputs": [], "events": [],
    }
    assert compare(previous, current) == []


def test_compare_reports_removed_alias_and_repurposed_artifact_fields():
    previous = {
        "settings": [{"key": "mode", "type": "string", "meaning": "select mode",
                      "aliases": [{"kind": "flag", "name": "--legacy"}]}],
        "commands": [],
        "outputs": [{"file": "summary.json", "format": "json",
                     "fields": [{"name": "result", "type": "string", "stable": True}]}],
        "events": [{"name": "tool-tape.intent",
                    "fields": [{"name": "args", "type": "object", "stable": True}]}],
    }
    current = {
        "settings": [{"key": "mode", "type": "string", "meaning": "select mode"}],
        "commands": [],
        "outputs": [{"file": "summary.json", "format": "text",
                     "fields": [{"name": "result", "type": "number", "stable": True}]}],
        "events": [{"name": "tool-tape.intent",
                    "fields": [{"name": "args", "type": "string", "stable": True}]}],
    }
    assert compare(previous, current) == [
        {"kind": "event_field", "name": "tool-tape.intent#args", "change": "type_changed",
         "severity": "break"},
        {"kind": "output", "name": "summary.json", "change": "repurposed"},
        {"kind": "output_field", "name": "summary.json#result", "change": "type_changed",
         "severity": "break"},
        {"kind": "setting_alias", "name": "flag:--legacy", "change": "removed"},
    ]


def test_compare_marks_stable_artifact_breaks_and_unstable_notices():
    previous = {
        "settings": [], "commands": [],
        "outputs": [{"file": "summary.json", "format": "json", "fields": [
            {"name": "documented", "type": "string", "stable": True},
            {"name": "observed", "type": "string", "stable": False},
            {"name": "unstable-removed", "type": "boolean", "stable": False},
        ]}],
        "events": [{"name": "request", "fields": [
            {"name": "documented", "type": "integer", "stable": True},
        ]}],
    }
    current = {
        "settings": [], "commands": [],
        "outputs": [{"file": "summary.json", "format": "json", "fields": [
            {"name": "observed", "type": "integer", "stable": False},
        ]}],
        "events": [{"name": "request", "fields": []}],
    }
    changes = compare(previous, current)
    assert changes == [
        {"kind": "event_field", "name": "request#documented", "change": "removed",
         "severity": "break"},
        {"kind": "output_field", "name": "summary.json#documented", "change": "removed",
         "severity": "break"},
        {"kind": "output_field", "name": "summary.json#observed", "change": "type_changed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#unstable-removed", "change": "removed",
         "severity": "notice"},
    ]


def test_compare_checks_promised_json_field_compatibility():
    previous = {"settings": [], "commands": [{"name": "inspect", "promised_json": {
        "flags": ["--send", "--frame"], "fields": [
            {"name": "stable", "type": "string", "stable": True},
            {"name": "observed", "type": "string", "stable": False},
        ]}}], "outputs": [], "events": []}
    current = {"settings": [], "commands": [{"name": "inspect", "promised_json": {
        "flags": ["--send", "--frame"], "fields": [
            {"name": "observed", "type": "integer", "stable": False},
        ]}}], "outputs": [], "events": []}
    assert compare(previous, current) == [
        {"kind": "command_json_field", "name": "inspect#observed", "change": "type_changed",
         "severity": "notice"},
        {"kind": "command_json_field", "name": "inspect#stable", "change": "removed",
         "severity": "break"},
    ]


def test_compare_type_stability_and_stability_changes():
    previous = {
        "settings": [],
        "commands": [{"name": "inspect", "promised_json": {"fields": [
            {"name": "type-change", "type": "string", "stable": "type"},
            {"name": "type-remove", "type": "string", "stable": "type"},
            {"name": "true-to-false", "type": "string", "stable": True},
        ]}}],
        "outputs": [{"file": "summary.json", "format": "json", "fields": [
            {"name": "true-change", "type": "string", "stable": True},
            {"name": "true-remove", "type": "string", "stable": True},
            {"name": "false-change", "type": "string", "stable": False},
            {"name": "false-remove", "type": "string", "stable": False},
            {"name": "type-change", "type": "string", "stable": "type"},
            {"name": "type-remove", "type": "string", "stable": "type"},
            {"name": "true-to-false", "type": "string", "stable": True},
            {"name": "false-to-type", "type": "string", "stable": False},
            {"name": "type-to-true", "type": "string", "stable": "type"},
        ]}],
        "events": [],
    }
    current = {
        "settings": [],
        "commands": [{"name": "inspect", "promised_json": {"fields": [
            {"name": "type-change", "type": "integer", "stable": "type"},
            {"name": "true-to-false", "type": "string", "stable": False},
        ]}}],
        "outputs": [{"file": "summary.json", "format": "json", "fields": [
            {"name": "true-change", "type": "integer", "stable": True},
            {"name": "false-change", "type": "integer", "stable": False},
            {"name": "type-change", "type": "integer", "stable": "type"},
            {"name": "true-to-false", "type": "string", "stable": False},
            {"name": "false-to-type", "type": "string", "stable": "type"},
            {"name": "type-to-true", "type": "string", "stable": True},
        ]}],
        "events": [],
    }
    assert compare(previous, current) == [
        {"kind": "command_json_field", "name": "inspect#true-to-false", "change": "stability_changed",
         "severity": "notice"},
        {"kind": "command_json_field", "name": "inspect#type-change", "change": "type_changed",
         "severity": "break"},
        {"kind": "command_json_field", "name": "inspect#type-remove", "change": "removed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#false-change", "change": "type_changed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#false-remove", "change": "removed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#false-to-type", "change": "stability_changed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#true-change", "change": "type_changed",
         "severity": "break"},
        {"kind": "output_field", "name": "summary.json#true-remove", "change": "removed",
         "severity": "break"},
        {"kind": "output_field", "name": "summary.json#true-to-false", "change": "stability_changed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#type-change", "change": "type_changed",
         "severity": "break"},
        {"kind": "output_field", "name": "summary.json#type-remove", "change": "removed",
         "severity": "notice"},
        {"kind": "output_field", "name": "summary.json#type-to-true", "change": "stability_changed",
         "severity": "notice"},
    ]


def test_summary_error_and_yolo_fields_match_catalog(tmp_path):
    from stateframe.runner import summarize

    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"yolo": True}), encoding="utf-8")
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "ENGINE_ERROR", "error_message": "sample failure"},
        "seconds": 0.1, "exit_status": 1,
    }), encoding="utf-8")

    summary = summarize(run, 0.0)
    assert "error_message" not in summary
    assert summary["yolo"] is True
    assert summary["engine_terminal"]["error_message"] == "sample failure"
    field = next(field for artifact in outputs() if artifact["file"] == "summary.json"
                 for field in artifact["fields"] if field["name"] == "engine_terminal.error_message")
    assert (field["type"], field["stable"], field["optional"]) == ("string", True, True)
    yolo = next(field for artifact in outputs() if artifact["file"] == "summary.json"
                for field in artifact["fields"] if field["name"] == "yolo")
    assert (yolo["type"], yolo["stable"], yolo["optional"]) == ("boolean", True, True)
    spec_yolo = next(field for artifact in outputs() if artifact["file"] == "spec.json"
                     for field in artifact["fields"] if field["name"] == "yolo")
    assert (spec_yolo["type"], spec_yolo["stable"], spec_yolo["optional"]) == (
        "boolean", True, True)


def test_yolo_is_recorded_in_spec_and_summary(tmp_path, monkeypatch, capsys):
    code, run_dir, _, _ = _run_case(
        tmp_path, "yolo", {"app.py": "assert True\n"}, [
            {"tool": "done", "arguments": {"summary": "The task is complete."}},
        ], monkeypatch, capsys, flags=("--yolo",), check=["--check-cmd", "true"])
    assert code == 0
    assert json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))["yolo"] is True
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["yolo"] is True


def test_ordinary_cli_request_path_does_not_load_contract_module():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import stateframe.cli; import stateframe.runner; "
         "assert 'stateframe.contract' not in sys.modules"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0


def test_environment_never_changes_the_document(monkeypatch):
    for name in ("STATEFRAME_ENDPOINT", "STATEFRAME_MODEL"):
        monkeypatch.delenv(name, raising=False)
    bare = render(contract_document(cli.build_parser()))
    monkeypatch.setenv("STATEFRAME_ENDPOINT", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("STATEFRAME_MODEL", "example")
    assert render(contract_document(cli.build_parser())) == bare


def test_version_is_the_installed_package_version():
    from stateframe import __version__

    document = contract_document(cli.build_parser())
    assert document["stateframe_version"] == __version__
    assert document["notes"]["settings"] == "pending" and document["settings"] == []
    assert all(row["promised_lines"] == [] for row in document["commands"])
