"""Product CLI, routed startup, resume refusal, and summary contracts."""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from stateframe import _child, cli, replay_tape, routes, runner
from stateframe.endpoint import Endpoint, EndpointError
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
    with pytest.raises(runner.RunError, match="LLAMA_CPP_TOKENIZE_URL is set while route read uses"):
        runner._probe_routes(options, lambda *_: None)
    # Regression: decide2's built-in "decision-tier" alias counted as another model, so any active
    # route was refused even after the routed role named its own tokenizer.
    options.route_cli = [f"read.endpoint={small}", "read.model=small", f"read.tokenizer={small}"]
    runner._probe_routes(options, lambda *_: None)
    assert options._routes is not None


def test_unit_overlay_model_is_probed_with_request_endpoint(servers, monkeypatch):
    default, small = servers
    served = {(default, "base"), (default, "large"), (small, "small")}
    checked = []

    def probe(endpoint):
        checked.append((endpoint.base_url, endpoint.model))
        return {"model_found": (endpoint.base_url, endpoint.model) in served}

    monkeypatch.setattr(Endpoint, "probe", probe)
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [{"id": 0, "n_ctx": 65536}])
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           "unit.model=large"])
    with pytest.raises(runner.RunError, match="route read.*unit.*does not serve model"):
        runner._probe_routes(options, lambda *_: None)
    assert (small, "large") in checked


def test_unit_overlay_context_is_checked_on_request_endpoint(servers, monkeypatch):
    default, small = servers
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [
        {"id": 0, "n_ctx": 65536 if self.base_url == small else 131072}])
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           "unit.context_tokens=131072"])
    with pytest.raises(runner.RunError, match="route read.*unit.*65,536-token slot"):
        runner._probe_routes(options, lambda *_: None)


def test_unit_overlay_slot_is_checked_on_request_endpoint(servers, monkeypatch):
    default, small = servers
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [
        {"id": 0 if self.base_url == small else 1, "n_ctx": 65536}])
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           "unit.slot=1"])
    with pytest.raises(runner.RunError, match="route read.*unit.*slot 1 is not reported"):
        runner._probe_routes(options, lambda *_: None)


def test_unit_overlay_tokenizer_is_checked_for_effective_model(servers, monkeypatch):
    default, small = servers
    tokenizer = "http://127.0.0.1:12345/v1"
    served = {(default, "base"), (small, "small"), (small, "base"),
              (tokenizer, "small")}
    monkeypatch.setattr(Endpoint, "probe", lambda self: {
        "model_found": (self.base_url, self.model) in served})
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [{"id": 0, "n_ctx": 65536}])
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           f"read.tokenizer={tokenizer}", "unit.model=base"])
    with pytest.raises(runner.RunError, match="route read.*unit.*tokenizer does not list model"):
        runner._probe_routes(options, lambda *_: None)


def test_decide2_model_exception_does_not_hide_unit_model_refusal(servers):
    default, _ = servers
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["plan.temperature=0.5",
                                           "unit.model=decision-tier"])
    with pytest.raises(runner.RunError, match="route unit endpoint does not serve model"):
        runner._probe_routes(options, lambda *_: None)


def test_default_tokenizer_override_keeps_spec_environment_and_output_inactive(
        servers, tmp_path, capsys):
    default, _ = servers
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def answer():\n    return 1\n")
    (repo / "test_calc.py").write_text("from calc import answer\ndef test_answer(): assert answer() == 1\n")
    baseline = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"))
    same = runner.RunOptions(repo=repo, task="task", endpoint=Endpoint(default, "base"),
                             run_dir=tmp_path / "run",
                             route_cli=[f"read.tokenizer={default}"])
    runner._probe_routes(baseline)
    baseline_output = capsys.readouterr()
    runner._probe_routes(same)
    assert capsys.readouterr() == baseline_output
    assert same._routes is None
    assert runner._child_env(same) == runner._child_env(baseline)
    _, _, spec = runner.prepare(same)
    assert "routes" not in spec
    assert "routes" not in json.loads((same.run_dir / "spec.json").read_text())


def test_routed_key_reaches_child_without_entering_spec(servers, monkeypatch):
    default, _ = servers
    monkeypatch.setenv("ROUTE_TEST_KEY", "secret-not-for-spec")
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["edit.api_key_env=ROUTE_TEST_KEY"])
    runner._probe_routes(options, lambda *_: None)
    assert runner._child_env(options)["ROUTE_TEST_KEY"] == "secret-not-for-spec"
    assert "secret-not-for-spec" not in json.dumps(options._routes.public())


def test_routes_with_distinct_key_envs_are_probed_and_failure_names_role(servers, monkeypatch):
    default, _ = servers
    monkeypatch.setenv("ROUTE_FIRST_KEY", "first-secret")
    monkeypatch.setenv("ROUTE_SECOND_KEY", "second-secret")
    checked = []

    def probe(endpoint):
        checked.append(endpoint.api_key_env)
        if endpoint.api_key_env == "ROUTE_SECOND_KEY":
            raise EndpointError("authentication refused")
        return {"model_found": True}

    monkeypatch.setattr(Endpoint, "probe", probe)
    monkeypatch.setattr(Endpoint, "get_slots", lambda _self: [])
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["read.api_key_env=ROUTE_FIRST_KEY",
                                           "plan.api_key_env=ROUTE_SECOND_KEY"])
    with pytest.raises(runner.RunError, match="route plan authentication refused"):
        runner._probe_routes(options, lambda *_: None)
    assert checked == ["ROUTE_FIRST_KEY", "ROUTE_SECOND_KEY"]


@pytest.mark.parametrize("source", ["test_env", "extra_env"])
def test_spec_environment_cannot_persist_routed_key_value(servers, tmp_path, monkeypatch, source):
    default, _ = servers
    monkeypatch.setenv("ROUTE_TEST_KEY", "secret-not-for-spec")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def answer():\n    return 1\n")
    (repo / "test_calc.py").write_text("from calc import answer\ndef test_answer(): assert answer() == 1\n")
    options = runner.RunOptions(repo=repo, task="Fix answer", endpoint=Endpoint(default, "base"),
                                run_dir=tmp_path / "run", route_cli=["edit.api_key_env=ROUTE_TEST_KEY"],
                                **{source: {"ROUTE_TEST_KEY": "secret-not-for-spec"}})
    runner._probe_routes(options, lambda *_: None)
    with pytest.raises(runner.RunError, match="ROUTE_TEST_KEY"):
        runner.prepare(options)
    assert not (tmp_path / "run" / "spec.json").exists()


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


@pytest.mark.parametrize("slot_size", [65536, 16384])
def test_explicit_context_checks_run_slot_size(servers, monkeypatch, slot_size):
    default, small = servers
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           "read.context_tokens=65536"])
    options._context_tokens = 32768
    options._context_env = {runner.CONTEXT_TOKENS_ENV: "32768"}
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [{"id": 0, "n_ctx":
                          slot_size if self.model == "small" else 32768}])
    with pytest.raises(runner.RunError, match="route read model small has a"):
        runner._probe_routes(options, lambda *_: None)


@pytest.mark.parametrize("slot_entry,accepted", [
    ({"id": True, "n_ctx": 65536}, False),
    ({"id": "bad", "id_slot": 1, "n_ctx": 65536}, True),
])
def test_route_slot_uses_valid_nonnegative_integer_id(servers, monkeypatch, slot_entry, accepted):
    default, _ = servers
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["read.slot=1"])
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [slot_entry])
    if accepted:
        runner._probe_routes(options, lambda *_: None)
        assert options._routes is not None
    else:
        with pytest.raises(runner.RunError, match="is not reported"):
            runner._probe_routes(options, lambda *_: None)


def test_resolved_inactive_auto_slot_emits_no_route_output(servers, monkeypatch, capsys):
    default, _ = servers
    monkeypatch.setenv("STATEFRAME_ALLOW_REMOTE_ENDPOINT", "1")
    options = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=["read.slot=auto"])
    options._cache_slot = 0
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [{"id": 0, "n_ctx": 65536,
                                                                 "is_processing": False}])
    capsys.readouterr()
    runner._probe_routes(options)
    assert options._routes is None
    assert capsys.readouterr().err == ""


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


def _started_read_route(servers, monkeypatch):
    default, small = servers
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    monkeypatch.setattr(Endpoint, "get_slots", lambda self: [{"id": 0, "n_ctx": 65536}])
    started = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small",
                                           "read.slot=0", f"read.tokenizer={small}"])
    runner._probe_routes(started, lambda *_: None)
    recorded = started._routes.public()
    resumed = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"))
    resumed._recorded_routes = recorded
    resumed._recorded_route_default = runner._route_defaults(started)
    return recorded, resumed


def test_resume_uses_recorded_read_route_without_cli_and_with_tokenize_env(servers, monkeypatch):
    recorded, resumed = _started_read_route(servers, monkeypatch)
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", servers[0] + "/tokenize")
    current = runner._resolve_routes(resumed)
    assert runner._compare_resume_routes(recorded, current, []) == []
    runner._probe_routes(resumed, lambda *_: None, resume_roles=recorded["roles"])
    assert resumed._routes.public() == recorded
    assert resumed._routes.sources["read"]["tokenizer"] == "recorded"


def test_resume_default_endpoint_move_keeps_explicit_read_route(servers, monkeypatch):
    recorded, resumed = _started_read_route(servers, monkeypatch)
    old_endpoint = servers[0]
    moved_endpoint = "http://127.0.0.1:18501/v1"
    resumed.endpoint = Endpoint(moved_endpoint, "base")
    current = runner._resolve_routes(resumed)
    changes = runner._compare_resume_routes(recorded, current, [],
                                            old_default=(old_endpoint, "base"),
                                            new_default=(moved_endpoint, "base"))
    assert current.roles["read"] == recorded["roles"]["read"]
    assert current.roles["edit"]["endpoint"] == moved_endpoint
    assert {change["role"] for change in changes} == {"plan", "edit", "finish", "decide2", "unit"}


def test_resume_refuses_conflicting_route_env_model(servers, monkeypatch):
    recorded, resumed = _started_read_route(servers, monkeypatch)
    monkeypatch.setenv("STATEFRAME_ROUTE_READ_MODEL", "other")
    with pytest.raises(runner.resume_module.ResumeRefused, match="route read.model changed"):
        runner._compare_resume_routes(recorded, runner._resolve_routes(resumed), [])


def test_resume_unrouted_run_accepts_inactive_current_routes():
    current = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                             environ={"STATEFRAME_ROUTE_READ_MODEL": "base"})
    assert not current.active
    assert runner._compare_resume_routes(None, current, []) == []


def test_resume_unrouted_run_refuses_env_route_model(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTE_READ_MODEL", "small")
    current = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                             environ={"STATEFRAME_ROUTE_READ_MODEL": "small"})
    assert current.active
    with pytest.raises(runner.resume_module.ResumeRefused,
                       match="this run started without per-step models; route read.model is set now"):
        runner._compare_resume_routes(None, current, [])


def test_resume_unrouted_run_names_effective_change_after_inactive_settings():
    current = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                             cli=["read.allow_remote=true", "read.model=base", "plan.model=small"],
                             environ={})
    assert current.first_change == ("plan", "model")
    with pytest.raises(runner.resume_module.ResumeRefused, match="route plan.model is set now"):
        runner._compare_resume_routes(None, current, [])


def test_resume_unrouted_run_refuses_cli_route_endpoint():
    setting = "read.endpoint=http://127.0.0.1:8081/v1"
    current = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                             cli=[setting], environ={})
    with pytest.raises(runner.resume_module.ResumeRefused,
                       match="this run started without per-step models; --route read.endpoint is set now"):
        runner._compare_resume_routes(None, current, [setting])


def test_resume_without_recorded_tokenizer_still_refuses_tokenize_env(servers, monkeypatch):
    default, small = servers
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    started = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"),
                                route_cli=[f"read.endpoint={small}", "read.model=small"])
    runner._probe_routes(started, lambda *_: None)
    resumed = runner.RunOptions(repo=None, task="task", endpoint=Endpoint(default, "base"))
    resumed._recorded_routes = started._routes.public()
    resumed._recorded_route_default = runner._route_defaults(started)
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", default + "/tokenize")
    with pytest.raises(runner.RunError, match="LLAMA_CPP_TOKENIZE_URL is set while route read uses"):
        runner._probe_routes(resumed, lambda *_: None)


@pytest.mark.parametrize("bad_roles,fragment", [
    ({"read": {"unexpected": 1}}, "read.unexpected"),
    ({"read": {"slot": "zero"}}, "read.slot"),
    ({"read": {"api_key_env": ""}}, "read.api_key_env"),
    ({"reviewer": {"model": "small"}}, "reviewer.model"),
])
def test_resume_refuses_malformed_recorded_routes(servers, monkeypatch, tmp_path, bad_roles, fragment):
    endpoint = Endpoint(servers[0], "base")
    spec = {"repo": str(tmp_path), "public_row": {"msg": "task"}, "chat_url": endpoint.chat_url,
            "model": "base", "environment": {}, "max_turns": 1,
            "routes": {"schema": routes.SCHEMA, "roles": bad_roles}}
    monkeypatch.setattr(runner.resume_module, "assess", lambda *args, **kwargs: {
        "spec": spec, "max_turns": 1})
    monkeypatch.setattr(runner, "_check_python", lambda python: python)
    monkeypatch.setattr(runner, "check_slot_context", lambda *args, **kwargs: None)
    with pytest.raises(runner.resume_module.ResumeRefused, match=fragment):
        runner.resume_run(tmp_path, endpoint=endpoint, printer=lambda *_: None)


def test_resume_run_keeps_pending_across_two_endpoint_moves_and_refusal(tmp_path, monkeypatch):
    scripts = [Script({"model": model}) for model in ("base", "small", "small", "small")]
    active = []
    for script in scripts:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(script))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        active.append((server, thread))
    try:
        default, original, moved, latest = [f"http://127.0.0.1:{server.server_port}/v1"
                                             for server, _ in active]
        monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
        monkeypatch.setattr(runner, "_check_python", lambda python: python)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("value = 1\n")
        (repo / "test_app.py").write_text("from app import value\ndef test_value(): assert value == 1\n")
        run = tmp_path / "run"
        options = runner.RunOptions(repo=repo, task="Inspect app.py", endpoint=Endpoint(default, "base"),
                                    run_dir=run, route_cli=[f"read.endpoint={original}", "read.model=small"])
        runner._probe_routes(options, lambda *_: None)
        runner.prepare(options)

        request = b'{"model": "small", "messages": []}'

        def route(url):
            return {"role": "read", "scope": "run", "unit_class": None,
                    "endpoint": url + "/chat/completions",
                    "endpoint_host": url.removeprefix("http://").removesuffix("/v1"),
                    "model": "small", "escalated_from": None}

        def record(wire, number, url, response=None):
            wire.mkdir(exist_ok=True)
            stem = wire / f"{number:04d}"
            stem.with_suffix(".route.json").write_text(json.dumps(route(url)))
            stem.with_suffix(".request.json").write_bytes(request)
            if response is not None:
                stem.with_suffix(".response.json").write_bytes(response)

        def physical(url):
            with urlopen(Request(url + "/chat/completions", data=request,
                                 headers={"Content-Type": "application/json"}), timeout=5) as reply:
                return reply.read()

        first_response = physical(original)
        record(run / "wire", 1, original, first_response)
        record(run / "wire", 2, original)
        seen = []

        def execute(_options, _started, _printer, run_dir, _work, spec):
            resume_spec = spec["resume"]
            seen.append(resume_spec)
            current = moved if len(seen) == 1 else latest
            replay = replay_tape.Replayer(resume_spec["tape"], pending=resume_spec.get("pending"),
                                          route_changes=resume_spec.get("route_changes"))
            assert replay.answer(request, route=route(current)) == first_response
            assert replay.diverged is None
            record(run_dir / "wire", 1, current, first_response)
            if len(seen) == 2:
                assert resume_spec["pending"]["request_digest"] == replay_tape.normalized_digest(request)
                assert replay.answer(request, route=route(current)) is None
                assert replay.diverged is None and replay.pending is None
                record(run_dir / "wire", 2, current, physical(current))
            return {"result": "INTERRUPTED"}

        monkeypatch.setattr(runner, "_execute", execute)
        runner.resume_run(run, route_cli=[f"read.endpoint={moved}"], printer=lambda *_: None)
        after_first = (run / "spec.json").read_bytes()
        with pytest.raises(runner.RunError, match="route read endpoint does not serve model 'wrong'"):
            runner.resume_run(run, route_cli=["read.model=wrong"], printer=lambda *_: None)
        assert (run / "spec.json").read_bytes() == after_first
        runner.resume_run(run, route_cli=[f"read.endpoint={latest}"], printer=lambda *_: None)

        assert len(seen) == 2
        assert len(seen[0]["route_changes"]) == 1
        assert seen[1]["route_changes"] == [
            {"role": "read", "field": "endpoint", "from": original, "to": moved},
            {"role": "read", "field": "endpoint", "from": moved, "to": latest}]
        assert len(scripts[1].requests) == 1
        assert not scripts[2].requests
        assert len(scripts[3].requests) == 1
        assert not (run / "resume-divergence.json").exists()
    finally:
        for server, thread in active:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


def test_three_resumes_keep_pending_through_chained_endpoint_and_model_moves(tmp_path, monkeypatch):
    # Stub the engine launch and Python probe to choose the kill points. Route validation, attempt
    # archival, the child wire recorder, replay checks, and scripted-server HTTP sends run for real.
    monkeypatch.syspath_prepend(str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    scripts = [Script({"model": model}) for model in ("base", "small", "middle", "latest", "final")]
    active = []
    for script in scripts:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(script))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        active.append((server, thread))
    try:
        urls = [f"http://127.0.0.1:{server.server_port}/v1" for server, _ in active]
        default, original, middle, latest, final = urls
        monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
        monkeypatch.setattr(runner, "_check_python", lambda python: python)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("value = 1\n")
        (repo / "test_app.py").write_text("from app import value\ndef test_value(): assert value == 1\n")
        run = tmp_path / "run"
        options = runner.RunOptions(repo=repo, task="Inspect app.py", endpoint=Endpoint(default, "base"),
                                    run_dir=run, route_cli=[f"read.endpoint={original}", "read.model=small"])
        runner._probe_routes(options, lambda *_: None)
        runner.prepare(options)
        physical = kernel_bounded_http.request_bytes

        def route(url, model):
            return {"role": "read", "scope": "run", "unit_class": None,
                    "endpoint": url + "/chat/completions",
                    "endpoint_host": url.removeprefix("http://").removesuffix("/v1"),
                    "model": model, "escalated_from": None}

        def send(body, url, model):
            return kernel_bounded_http.request_bytes(
                body, url + "/chat/completions", "", deadline=time.monotonic() + 15,
                socket_timeout=5, route_meta=route(url, model))

        def pending(wire, number, body, url, model):
            stem = wire / f"{number:04d}"
            stem.with_suffix(".route.json").write_text(json.dumps(route(url, model)))
            stem.with_suffix(".request.json").write_bytes(body)

        initial = b'{"model": "small", "messages": []}'
        _child.install_wire_recorder(run, Endpoint(default, "base").chat_url,
                                     routed_endpoints={route(original, "small")["endpoint"]})
        recorded_responses = [send(initial, original, "small") for _ in range(2)]
        pending(run / "wire", 3, initial, original, "small")

        destinations = [(middle, "middle"), (latest, "latest"), (final, "final")]
        seen = []

        def execute(_options, _started, _printer, run_dir, _work, spec):
            index = len(seen)
            url, model = destinations[index]
            resume_spec = spec["resume"]
            seen.append(resume_spec)
            monkeypatch.setattr(kernel_bounded_http, "request_bytes", physical)
            replay = replay_tape.Replayer(resume_spec["tape"], pending=resume_spec.get("pending"),
                                          route_changes=resume_spec.get("route_changes"))
            _child.install_wire_recorder(run_dir, spec["chat_url"], replayer=replay,
                                         routed_endpoints={route(url, model)["endpoint"]})
            body = json.dumps({"model": model, "messages": []}).encode()
            count = (1, 2, 3)[index]
            responses = [send(body, url, model) for _ in range(count)]
            assert responses[:min(count, 2)] == recorded_responses[:min(count, 2)]
            assert replay.diverged is None
            if index < 2:
                pending(run_dir / "wire", count + 1, body, url, model)
            else:
                assert replay.pending is None
                assert replay.pending_verified is not None
            return {"result": "INTERRUPTED"}

        monkeypatch.setattr(runner, "_execute", execute)
        for url, model in destinations:
            runner.resume_run(run, route_cli=[f"read.endpoint={url}", f"read.model={model}"],
                              printer=lambda *_: None)

        assert len(seen) == 3
        assert [len(item["tape"]) for item in seen] == [2, 2, 2]
        assert all(item["pending"]["request_digest"] == replay_tape.normalized_digest(initial)
                   for item in seen)
        assert [len(item["route_changes"]) for item in seen] == [2, 4, 6]
        assert [len(script.requests) for script in scripts] == [0, 2, 0, 0, 1]
        recorded_sends = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
        assert [item.get("replayed", False) for item in recorded_sends] == [True, True, False]
        verification = json.loads((run / "resume-pending-verified.json").read_text())
        assert verification["send"] == 3
        assert verification["recorded_route"] == route(original, "small")
        assert verification["route"] == route(final, "final")
        assert not (run / "resume-divergence.json").exists()
    finally:
        for server, thread in active:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


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


def test_resume_refuses_changed_unit_override_mask():
    before = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                            cli=["plan.model=large", "unit.model=base"], environ={})
    after = routes.resolve({"endpoint": "http://127.0.0.1:8080/v1", "model": "base"},
                           cli=["plan.model=large"], environ={})
    assert before.roles == after.roles
    with pytest.raises(runner.resume_module.ResumeRefused, match="route unit.model changed"):
        runner._compare_resume_routes(before.public(), after, [])


def test_unit_send_uses_effective_rate_override(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.route.json").write_text(json.dumps({"role": "read", "scope": "unit",
        "unit_class": "patch", "model": "base", "escalated_from": None}))
    spec = {"roles": {"read": {"cost_input_per_mtok": 1.0,
                              "cost_cached_input_per_mtok": 1.0, "cost_output_per_mtok": 1.0},
                      "edit": {"cost_input_per_mtok": 20.0,
                               "cost_cached_input_per_mtok": 20.0, "cost_output_per_mtok": 20.0},
                      "unit": {"cost_input_per_mtok": 10.0},
                      "unit.patch": {"cost_input_per_mtok": 12.0}},
            "unit_overrides": {"unit": ["cost_input_per_mtok"],
                               "unit.patch": ["cost_input_per_mtok"]}}
    costs = runner._route_costs(tmp_path, spec, [{"send": 1, "request_id": "unit-1",
                                                 "completion_tokens": 10}],
                                [{"total": 100, "cached": 20}])
    assert costs["roles"]["read"]["estimated_cost"] == 0.00099
    assert costs["total"]["estimated_cost"] == 0.00099
    assert costs["all_at_edit_rates"] == 0.00156
    assert costs["estimated_savings"] == 0.00057


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


def test_missing_route_record_makes_summary_totals_unknown(tmp_path, capsys):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.route.json").write_text(json.dumps({"role": "read", "scope": "main"}))
    rates = {"cost_input_per_mtok": 1.0, "cost_cached_input_per_mtok": 1.0,
             "cost_output_per_mtok": 1.0}
    spec = {"roles": {"read": rates, "edit": rates}}
    sends = [{"send": 1, "request_id": "known", "completion_tokens": 5},
             {"send": 2, "request_id": "missing", "completion_tokens": 7}]
    usage = [{"total": 20, "cached": 2}, {"total": 30, "cached": 3}]
    costs = runner._route_costs(tmp_path, spec, sends, usage)
    assert costs["total"]["requests"] == 2
    assert costs["total"]["prompt_tokens"] is None
    assert costs["total"]["estimated_cost"] is None
    assert costs["all_at_edit_rates"] is None
    assert costs["estimated_savings"] is None
    assert costs["unattributed_sends"] == 1
    (tmp_path / "final.diff").write_text("")
    cli._print_summary({"run_dir": str(tmp_path), "tests_before": {"passed": True},
                        "tests_after": {"command": "pytest"}, "runtime_outcome": "completed",
                        "check_outcome": "pass", "artifact": "captured", "succeeded": False,
                        "route_costs": costs})
    output = capsys.readouterr().out
    assert "1 send(s) missing route records; totals unknown" in output
    assert "total             2   unknown  unknown" in output


def test_unreported_cached_tokens_make_cost_unknown_not_a_crash(tmp_path, capsys):
    # Regression: a server that reports no cached_tokens left one role's cost None, and the
    # total summed it (TypeError int + None) after a successful routed run.
    wire = tmp_path / "wire"
    wire.mkdir()
    for index, role in ((1, "read"), (2, "edit")):
        (wire / f"{index:04d}.route.json").write_text(json.dumps({"role": role, "scope": "run"}))
    rates = {"cost_input_per_mtok": 1.0, "cost_cached_input_per_mtok": 0.5,
             "cost_output_per_mtok": 2.0}
    spec = {"roles": {"read": rates, "edit": rates}}
    sends = [{"send": 1, "request_id": "a", "completion_tokens": 5},
             {"send": 2, "request_id": "b", "completion_tokens": 7}]
    usage = [{"total": 20, "cached": None}, {"total": 30, "cached": 3}]
    costs = runner._route_costs(tmp_path, spec, sends, usage)
    assert costs["roles"]["read"]["cached_tokens"] is None
    assert costs["roles"]["read"]["prompt_tokens"] == 20
    assert costs["roles"]["read"]["estimated_cost"] is None
    assert costs["roles"]["edit"]["estimated_cost"] is not None
    assert costs["total"]["cached_tokens"] is None
    assert costs["total"]["prompt_tokens"] == 50
    assert costs["total"]["estimated_cost"] is None
    assert costs["estimated_savings"] is None
    (tmp_path / "final.diff").write_text("")
    cli._print_summary({"run_dir": str(tmp_path), "tests_before": {"passed": True},
                        "tests_after": {"command": "pytest"}, "runtime_outcome": "completed",
                        "check_outcome": "pass", "artifact": "captured", "succeeded": False,
                        "route_costs": costs})
    assert "read              1         0       20  unknown" in capsys.readouterr().out
