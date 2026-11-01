"""Product CLI, routed startup, resume refusal, and summary contracts."""
import json
import time

import pytest

from stateframe import cli, routes, runner
from stateframe.endpoint import Endpoint
from stateframe.scripted_endpoint import Script, make_handler
from http.server import ThreadingHTTPServer
from threading import Thread


def test_config_init_writes_inactive_file_and_refuses_overwrite(tmp_path, capsys):
    path = tmp_path / "nested" / "routes.toml"
    assert cli.main(["config", "init", "--path", str(path)]) == 0
    assert path.read_text() == routes.example()
    assert not routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                              file=path, environ={}).active
    assert cli.main(["config", "init", "--path", str(path)]) == 2
    assert "exists" in capsys.readouterr().err


def test_cli_accepts_route_flags_on_run_chat_resume(tmp_path):
    parser = cli.build_parser()
    for command in ("run", "chat"):
        args = [command, "--repo", str(tmp_path), "--endpoint", "http://127.0.0.1:8080/v1",
                "--model", "base", "--route", "read.model=small", "--routes", str(tmp_path / "r.toml")]
        if command == "run":
            args += ["--task", "fix"]
        parsed = parser.parse_args(args)
        assert parsed.route == ["read.model=small"]
        assert parsed.routes == tmp_path / "r.toml"
    parsed = parser.parse_args(["resume", str(tmp_path), "--route", "edit.model=renamed"])
    assert parsed.route == ["edit.model=renamed"]


def test_route_shell_names_are_recognized_but_unit_spec_names_are_ignored():
    names = {"STATEFRAME_ROUTES": "routes.toml", "STATEFRAME_ROUTE_READ_MODEL": "small",
             "STATEFRAME_ROUTE_UNIT_PATCH_SLOT": "2", "STATEFRAME_ROUTE_SCOPE": "unit",
             "STATEFRAME_ROUTE_UNIT_CLASS": "patch", "STATEFRAME_ROUTE_SLOT": "3"}
    assert runner.ignored_shell_settings(names) == ["STATEFRAME_ROUTE_SCOPE", "STATEFRAME_ROUTE_SLOT",
                                                    "STATEFRAME_ROUTE_UNIT_CLASS"]


@pytest.fixture
def servers():
    active = []
    for model in ("base", "small"):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Script({"model": model})))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        active.append((server, thread))
    try:
        yield [f"http://127.0.0.1:{server.server_port}/v1" for server, _ in active]
    finally:
        for server, thread in active:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


def test_routed_start_lines_and_public_spec(servers, tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    default, small = servers
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def answer():\n    return 1\n")
    (repo / "test_calc.py").write_text("from calc import answer\ndef test_answer(): assert answer() == 1\n")
    options = runner.RunOptions(repo=repo, task="Fix answer", endpoint=Endpoint(default, "base"),
                                run_dir=tmp_path / "run", route_cli=[f"read.endpoint={small}",
                                                                   "read.model=small"])
    lines = []
    runner._probe_routes(options, lines.append)
    assert options._routes is not None
    assert any(line.startswith("route read: small at 127.0.0.1:") for line in lines)
    assert any(line.startswith("route edit: default (base at 127.0.0.1:") for line in lines)
    run_dir, _, spec = runner.prepare(options)
    assert spec["routes"]["digest"] == options._routes.digest
    assert spec["routes"]["roles"]["read"]["model"] == "small"
    assert json.loads((run_dir / "spec.json").read_text())["routes"] == spec["routes"]


def test_start_refusals_name_route_and_repair(servers, monkeypatch):
    default, small = servers
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"))
    options.route_cli = [f"read.endpoint={small}", "read.model=wrong"]
    with pytest.raises(runner.RunError, match="route read endpoint does not serve model"):
        runner._probe_routes(options, lambda *_: None)
    options.route_cli = ["read.endpoint=http://127.0.0.1:1/v1", "read.model=small"]
    with pytest.raises(runner.RunError, match="route read endpoint unreachable"):
        runner._probe_routes(options, lambda *_: None)
    options.route_cli = [f"read.endpoint={small}", "read.model=small", "read.slot=auto"]
    with pytest.raises(runner.RunError, match="no idle slot"):
        runner._probe_routes(options, lambda *_: None)
    options.route_cli = [f"read.endpoint={small}", "read.model=small", f"read.tokenizer={default}"]
    with pytest.raises(runner.RunError, match="tokenizer does not list model"):
        runner._probe_routes(options, lambda *_: None)
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", default + "/tokenize")
    options.route_cli = [f"read.endpoint={small}", "read.model=small"]
    with pytest.raises(runner.RunError, match="LLAMA_CPP_TOKENIZE_URL is set"):
        runner._probe_routes(options, lambda *_: None)


def test_routed_key_reaches_child_without_entering_spec(servers, monkeypatch):
    default, _ = servers
    monkeypatch.setenv("ROUTE_TEST_KEY", "secret-not-for-spec")
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["edit.api_key_env=ROUTE_TEST_KEY"])
    runner._probe_routes(options, lambda *_: None)
    assert runner._child_env(options)["ROUTE_TEST_KEY"] == "secret-not-for-spec"
    assert "secret-not-for-spec" not in json.dumps(options._routes.public())


def test_route_capacity_refusal_names_role(servers):
    default, _ = servers
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(Script({"model": "small", "slots": [32768]})))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        small = f"http://127.0.0.1:{server.server_port}/v1"
        options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                    route_cli=[f"read.endpoint={small}", "read.model=small"])
        with pytest.raises(runner.RunError, match="route read model small has a 32,768-token slot"):
            runner._probe_routes(options, lambda *_: None)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_resume_reuses_resolved_auto_slot(servers, monkeypatch):
    default, _ = servers
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["read.slot=auto"])
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [
        {"id": 0, "n_ctx": 65536, "is_processing": True},
        {"id": 1, "n_ctx": 65536, "is_processing": False}])
    runner._probe_routes(options, lambda *_: None)
    assert options._routes.roles["read"]["slot"] == 1
    runner._probe_routes(options, lambda *_: None, resume_roles={"read": {"slot": 0}})
    assert options._routes.roles["read"]["slot"] == 0


def test_inactive_spec_unchanged_shape(tmp_path):
    # The inactive route table is omitted from the run spec, including a fully commented file.
    endpoint = Endpoint("http://127.0.0.1:8080/v1", "base")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def answer():\n    return 1\n")
    (repo / "test_calc.py").write_text("from calc import answer\ndef test_answer(): assert answer() == 1\n")
    options = runner.RunOptions(repo=repo, task="Fix answer", endpoint=endpoint,
                                run_dir=tmp_path / "run")
    options.routes_file = tmp_path / "routes.toml"
    options.routes_file.write_text(routes.example())
    assert not runner._resolve_routes(options).active
    assert options._routes is None
    _, _, spec = runner.prepare(options)
    assert "routes" not in spec
    assert "routes" not in json.loads((options.run_dir / "spec.json").read_text())


def test_resume_refusal_names_role_and_field():
    recorded = {"roles": {"edit": {"model": "old", "endpoint": "http://127.0.0.1:8080/v1"}}}
    current = routes.RouteTable({"edit": {"model": "new", "endpoint": "http://127.0.0.1:8080/v1"}},
                                {}, True, "digest")
    with pytest.raises(runner.resume_module.ResumeRefused, match="route edit.model changed"):
        runner._compare_resume_routes(recorded, current, [])
    changes = runner._compare_resume_routes(recorded, current, ["edit.model=new"])
    assert changes == [{"role": "edit", "field": "model", "from": "old", "to": "new"}]
    with pytest.raises(runner.resume_module.ResumeRefused, match="only change ROLE.endpoint or ROLE.model"):
        runner._compare_resume_routes(recorded, current, ["edit.temperature=0.5"])


def test_route_summary_uses_wire_usage_and_prices(tmp_path, capsys):
    wire = tmp_path / "wire"
    wire.mkdir()
    (tmp_path / "sends.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"send": 1, "request_id": "read-1", "completion_tokens": 10},
        {"send": 2, "request_id": "edit-1"},
    ]) + "\n")
    (wire / "0001.route.json").write_text(json.dumps({"role": "read", "scope": "main",
        "unit_class": None, "endpoint_host": "127.0.0.1:8081", "model": "small", "escalated_from": None}))
    (wire / "0002.route.json").write_text(json.dumps({"role": "edit", "scope": "main",
        "unit_class": None, "endpoint_host": "127.0.0.1:8080", "model": "base", "escalated_from": "read-1"}))
    for number, prompt, cached, completion in ((1, 100, 40, 10), (2, 200, 50, 20)):
        (wire / f"{number:04d}.response.json").write_text(json.dumps({"usage": {
            "prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens": completion}}))
    rates = {"cost_input_per_mtok": 10.0, "cost_cached_input_per_mtok": 2.0,
             "cost_output_per_mtok": 20.0}
    spec = {"routes": {"roles": {"read": {**rates, "cost_input_per_mtok": 1.0},
                                    "edit": rates}}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    summary = runner.summarize(tmp_path, time.monotonic())
    costs = summary["route_costs"]
    assert costs["roles"]["read"]["escalated"] == 1
    assert costs["total"]["requests"] == 2
    assert costs["total"]["prompt_tokens"] == 300
    assert costs["total"]["cached_tokens"] == 90
    assert costs["total"]["completion_tokens"] == 30
    assert costs["all_at_edit_rates"] > costs["total"]["estimated_cost"]
    (tmp_path / "final.diff").write_text("")
    summary.update({"run_dir": str(tmp_path), "tests_before": {"passed": True},
                    "tests_after": {"command": "pytest"}, "runtime_outcome": "completed",
                    "check_outcome": "pass", "artifact": "captured", "succeeded": False,
                    "files_changed": []})
    cli._print_summary(summary)
    assert "=== model routes ===" in capsys.readouterr().out
