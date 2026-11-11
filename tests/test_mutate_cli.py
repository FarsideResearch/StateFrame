"""Human-facing mutation commands stay isolated from the optional engine modules."""
from __future__ import annotations

import json
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import cli, mutations_cli
from stateframe.endpoint import Endpoint
from stateframe.runner import mutation_spec_options


@pytest.fixture(autouse=True)
def _run_cli_logic_with_linux_platform(monkeypatch):
    """Keep CLI/store behavior tests portable; platform refusal has dedicated coverage."""
    monkeypatch.setattr(mutations_cli.sys, "platform", "linux")


def _endpoint() -> Endpoint:
    return Endpoint.from_args("http://127.0.0.1:8080/v1", "test-model", None)


def _parser_args(*extra: str):
    return cli.build_parser().parse_args([
        "run", "--repo", ".", "--task", "inspect this", "--endpoint",
        "http://127.0.0.1:8080/v1", "--model", "test-model", *extra,
    ])


def _fake_engine(monkeypatch, **modules):
    def load(name: str):
        try:
            return modules[name]
        except KeyError as error:
            raise AssertionError(f"unexpected engine import: {name}") from error

    monkeypatch.setattr(mutations_cli, "_engine_module", load)


def test_mutations_parser_registers_commands_and_help_text(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["mutations", "selftest", "--json"])
    assert args.command == "mutations"
    assert args.mutations_command == "selftest"
    assert args.json is True

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["mutations", "--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for phrase in ("list", "show", "enable-level", "selftest", "doctor", "retain",
                   "remove", "promote", "prune"):
        assert phrase in help_text

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["run", "--help"])
    assert raised.value.code == 0
    run_help = capsys.readouterr().out
    for phrase in ("--mutations", "--mutation-network", "--mutation-domain",
                   "--mutation-write", "--mutation-read", "--mutation-retain", "--mutation-audit"):
        assert phrase in run_help


def test_run_and_chat_mutation_flags_map_to_run_options():
    args = _parser_args(
        "--mutations", "authored", "--mutation-network", "public-web",
        "--mutation-domain", "docs.example", "--mutation-domain", "api.example",
        "--mutation-write", "src/**", "--mutation-write", "tests/unit_*.py",
        "--mutation-read", "data/**", "--mutation-retain", "project",
        "--mutation-audit", "digest",
    )
    options = cli._run_options(args, _endpoint())
    assert options.mutations == "authored"
    assert options.mutation_network == "public-web"
    assert options.mutation_domains == ["docs.example", "api.example"]
    assert options.mutation_write == ["src/**", "tests/unit_*.py"]
    assert options.mutation_read == ["data/**"]
    assert options.mutation_retain == "project"
    assert options.mutation_audit == "digest"

    chat_args = cli.build_parser().parse_args([
        "chat", "--repo", ".", "--endpoint", "http://127.0.0.1:8080/v1",
        "--model", "test-model", "--mutations", "prebuilt",
    ])
    assert cli._run_options(chat_args, _endpoint()).mutations == "prebuilt"


def test_mutation_hook_flag_is_repeatable_and_requires_enabled_level():
    args = _parser_args("--mutations", "authored", "--mutation-hook", "observer:post_tool",
                        "--mutation-hook", "observer:session_end")
    options = cli._run_options(args, _endpoint())
    assert options.mutation_hooks == ["observer:post_tool", "observer:session_end"]
    assert mutation_spec_options(options, {})["hooks"] == options.mutation_hooks
    off = _parser_args("--mutations", "off", "--mutation-hook", "observer:post_tool")
    assert mutation_spec_options(cli._run_options(off, _endpoint()), {}) is None


def test_off_level_resolves_to_no_mutation_spec_options():
    args = _parser_args()
    options = cli._run_options(args, _endpoint())
    assert options.mutations is None
    assert mutation_spec_options(options, {}) is None
    assert mutation_spec_options(None, {}) is None
    assert mutation_spec_options(options, {"STATEFRAME_MUTATIONS": "off"}) is None


def test_off_parser_and_spec_resolution_do_not_import_mutation_engine_modules(monkeypatch):
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("kernel_mutate") or ".kernel_mutate" in name:
            raise AssertionError(f"unexpected optional mutation import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    args = _parser_args()
    options = cli._run_options(args, _endpoint())
    assert options.mutations is None
    assert mutation_spec_options(options, {}) is None


def test_default_prebuilt_list_and_show_read_the_shipped_catalog(capsys):
    assert cli.main(["mutations", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in listed["units"]}
    assert {"file_facts", "structured_peek", "archive_list", "sqlite_peek",
            "snapshot_diff", "text_grep_struct", "log_watch", "html_extract"} <= names

    assert cli.main(["mutations", "show", "file_facts", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["name"] == "file_facts"
    assert shown["origin"] == "prebuilt"


def test_list_delegates_to_store_and_emits_stable_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))
    rows = [{"name": "kept_reader", "scope": "user", "digest": "sha256:abc"}]
    calls = []
    _fake_engine(monkeypatch, kernel_mutate_store=SimpleNamespace(
        list_units=lambda home, *, scope, run: calls.append((home, scope, run)) or rows))

    result = cli.main(["mutations", "list", "--scope", "user", "--json"])

    assert result == 0
    assert calls == [(tmp_path / "sf-home" / "mutations", "user", None)]
    assert json.loads(capsys.readouterr().out) == {"units": rows}


def test_show_delegates_to_store(monkeypatch, capsys):
    item = {"name": "kept_reader", "revision": 2, "digest": "sha256:abc"}
    _fake_engine(monkeypatch, kernel_mutate_store=SimpleNamespace(
        show_unit=lambda home, name: item if name == "kept_reader" else None))

    result = cli.main(["mutations", "show", "kept_reader", "--json"])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == item


def test_enable_level_prints_exact_cli_and_environment_lines_without_writing(monkeypatch, capsys, tmp_path):
    state_home = tmp_path / "state-home"
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))

    result = cli.main(["mutations", "enable-level", "authored", "--profile", "r2hc3"])

    output = capsys.readouterr().out
    assert result == 0
    assert "export STATEFRAME_MUTATIONS=authored" in output
    assert "stateframe run --mutations authored" in output
    assert "r2hc3" in output
    assert not state_home.exists()


def test_selftest_calls_qualifier_and_json_output(monkeypatch, capsys, tmp_path):
    report = {"ok": True, "platform": "linux", "checks": [{"name": "boot", "ok": True}]}
    calls = []
    _fake_engine(monkeypatch, kernel_mutate_qualify=SimpleNamespace(
        selftest=lambda home: calls.append(home) or report))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))

    result = cli.main(["mutations", "selftest", "--json"])

    assert result == 0
    assert calls == [tmp_path / "sf-home"]
    assert json.loads(capsys.readouterr().out) == report


def test_doctor_includes_selftest_and_host_facts(monkeypatch, capsys):
    report = {"ok": True, "platform": "linux", "checks": []}
    _fake_engine(monkeypatch, kernel_mutate_qualify=SimpleNamespace(selftest=lambda home: report))

    result = cli.main(["mutations", "doctor", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["selftest"] == report
    assert {"seccomp", "cgroup", "bwrap", "sandbox"} <= set(output["facts"])


def test_retain_and_remove_delegate_scope_changes(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))
    calls = []
    store = SimpleNamespace(
        retain_from_run=lambda home, run, unit, scope: calls.append(("retain", home, run, unit, scope)),
        remove_unit=lambda home, unit, scope: calls.append(("remove", home, unit, scope)),
    )
    _fake_engine(monkeypatch, kernel_mutate_store=store)
    run = tmp_path / "run"

    assert cli.main(["mutations", "retain", str(run), "reader", "--scope", "project"]) == 0
    assert cli.main(["mutations", "remove", "reader", "--scope", "user"]) == 0
    assert calls == [
        ("retain", tmp_path / "sf-home" / "mutations", run, "reader", "project"),
        ("remove", tmp_path / "sf-home" / "mutations", "reader", "user"),
    ]
    assert "retained" in capsys.readouterr().out


def test_promote_refuses_unknown_unit_without_export(monkeypatch, capsys, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "stateframe.powerups", SimpleNamespace(
        supports_isolated_unit_executor=lambda name: name == "mutation",
        validate_isolated_unit_attachment=lambda attachment: True,
        install=lambda path: None))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    result = cli.main(["mutations", "promote", "reader", "--out", str(tmp_path / "export")])
    output = capsys.readouterr()
    assert result == 2
    assert "unknown mutation unit" in output.err
    assert not (tmp_path / "export").exists()


def test_prune_runs_sweep_then_store_gc_and_supports_dry_run(monkeypatch, capsys, tmp_path):
    calls = []
    def sweep_runtime(root, *, run_locks, dry_run=False):
        calls.append(("sweep", root, dry_run))
        return {"cleaned": [], "unresolved": []}

    sweep = SimpleNamespace(sweep=sweep_runtime)
    store = SimpleNamespace(gc=lambda home, *, dry_run: calls.append(("gc", home, dry_run)) or {"removed": 0})
    _fake_engine(monkeypatch, kernel_mutate_sweep=sweep, kernel_mutate_store=store)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))

    result = cli.main(["mutations", "prune", "--dry-run", "--json"])

    assert result == 0
    assert calls[0] == ("sweep", tmp_path / "sf-home" / "rt" / "mutations", True)
    assert calls[1] == ("gc", tmp_path / "sf-home" / "mutations", True)
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_prune_runs_normal_sweep_with_fail_closed_lock_probe(monkeypatch, capsys, tmp_path):
    calls = []

    def sweep_runtime(root, *, run_locks):
        candidate = tmp_path / "sf-home" / "rt" / "mutations" / "unknown-run"
        calls.append(("sweep", root, run_locks(candidate)))
        return {"cleaned": [], "unresolved": []}

    _fake_engine(monkeypatch,
                 kernel_mutate_sweep=SimpleNamespace(sweep=sweep_runtime),
                 kernel_mutate_store=SimpleNamespace(
                     gc=lambda home, *, dry_run: calls.append(("gc", home, dry_run)) or {"removed": 0}))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))

    result = cli.main(["mutations", "prune"])

    assert result == 0
    assert calls == [
        ("sweep", tmp_path / "sf-home" / "rt" / "mutations", True),
        ("gc", tmp_path / "sf-home" / "mutations", False),
    ]
    assert "dry_run: false" in capsys.readouterr().out


def test_prune_dry_run_refuses_when_sweeper_cannot_preview(monkeypatch, capsys, tmp_path):
    calls = []

    def sweep_runtime(root, *, run_locks):
        calls.append(("sweep", root))

    _fake_engine(monkeypatch,
                 kernel_mutate_sweep=SimpleNamespace(sweep=sweep_runtime),
                 kernel_mutate_store=SimpleNamespace(
                     gc=lambda home, *, dry_run: calls.append(("gc", home, dry_run))))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "sf-home"))

    result = cli.main(["mutations", "prune", "--dry-run"])

    assert result == 2
    assert calls == []
    assert "does not support dry-run" in capsys.readouterr().err


def test_real_store_list_show_and_remove(monkeypatch, capsys, tmp_path):
    from kernel_mutate_store import MutationStore
    from test_mutate_store import bundle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files, scope="user")
    assert cli.main(["mutations", "list", "--scope", "user", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["units"][0]["name"] == "x"
    assert cli.main(["mutations", "show", "x", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["digest"] == digest
    assert cli.main(["mutations", "remove", "x", "--scope", "user"]) == 0


def test_real_prune_dry_run_preserves_store(monkeypatch, capsys, tmp_path):
    from kernel_mutate_store import MutationStore
    from test_mutate_store import bundle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    assert cli.main(["mutations", "prune", "--dry-run", "--json"]) == 0
    assert store.get(digest) == files


def test_prune_dry_run_does_not_create_an_absent_store(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    assert cli.main(["mutations", "prune", "--dry-run", "--json"]) == 0
    assert not (tmp_path / "home").exists()


def test_promote_exports_disabled_isolated_unit_without_commands(monkeypatch, capsys, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "stateframe.powerups", SimpleNamespace(
        supports_isolated_unit_executor=lambda name: name == "mutation",
        validate_isolated_unit_attachment=lambda attachment: True,
        install=lambda path: None))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    out = tmp_path / "export"
    assert cli.main(["mutations", "promote", "file_facts", "--out", str(out)]) == 0
    assert (out / "powerup.toml").is_file()
    attachment = json.loads((out / "units.json").read_text())[0]
    assert attachment["schema"] == "stateframe-isolated-unit-attachment/1"
    assert attachment["executor"] == "mutation"
    assert attachment["replay_policy"] == "observation"
    assert not (out / "commands.json").exists()


def test_network_promotion_is_external_and_ceiling_digest_includes_read_grant(monkeypatch, tmp_path):
    import hashlib
    import sys

    monkeypatch.setitem(sys.modules, "stateframe.powerups", SimpleNamespace(
        supports_isolated_unit_executor=lambda name: name == "mutation",
        validate_isolated_unit_attachment=lambda attachment: True,
        install=lambda path: None))

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    out = tmp_path / "network-export"
    assert cli.main(["mutations", "promote", "html_extract", "--out", str(out)]) == 0
    attachment = json.loads((out / "units.json").read_text())[0]
    assert attachment["replay_policy"] == "external"
    ceiling = {"network": "public-web", "writes": [], "read_globs": [], "domains": []}
    encoded = json.dumps(ceiling, sort_keys=True, separators=(",", ":")).encode()
    assert attachment["capability_ceiling_digest"] == "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "case,flags,environment,expected_level,expected,error,expected_hook",
    [
        ("prebuilt-level", ["--mutations", "prebuilt"], {}, "prebuilt", {}, None, None),
        ("authored-level", ["--mutations", "authored"], {}, "authored", {}, None, None),
        ("public-web-full", ["--mutations", "prebuilt", "--mutation-network", "public-web",
                              "--shell-network", "full"], {}, "prebuilt",
         {"network": "public-web", "shell_network": "full"}, None, None),
        ("public-web-refused-without-shell-grant",
         ["--mutations", "prebuilt", "--mutation-network", "public-web"], {}, "prebuilt",
         {"network": "public-web", "shell_network": "loopback"},
         "mutation network requires --shell-network full", None),
        ("mutation-domain",
         ["--mutations", "prebuilt", "--mutation-network", "public-web", "--shell-network", "full",
          "--mutation-domain", "docs.example"], {}, "prebuilt",
         {"network": "public-web", "domains": ["docs.example"]}, None, None),
        ("mutation-write-glob", ["--mutations", "authored", "--mutation-write", "docs/**/*.md"],
         {}, "authored", {"write_globs": ["docs/**/*.md"]}, None, None),
        ("mutation-hook", ["--mutations", "authored", "--mutation-hook", "observer:post_tool"],
         {}, "authored", {"hooks": ["observer:post_tool"]}, None, ("observer", "post_tool")),
        ("mutation-retain", ["--mutations", "prebuilt", "--mutation-retain", "project"],
         {}, "prebuilt", {"retain": "project"}, None, None),
        ("environment-level", [], {"STATEFRAME_MUTATIONS": "prebuilt"}, "prebuilt", {}, None, None),
    ],
    ids=["prebuilt-level", "authored-level", "public-web-full", "public-web-refused-no-full",
         "mutation-domain", "mutation-write-glob", "mutation-hook", "mutation-retain",
         "environment-level"],
)
def test_cli_runner_mutation_options_reach_host_and_resume(
        monkeypatch, tmp_path, case, flags, environment, expected_level, expected, error, expected_hook):
    import contextlib

    import kernel_mutate_policy
    import kernel_mutate_records
    import kernel_mutate_runtime
    import kernel_mutate_store
    import kernel_mutate_sweep
    from stateframe import mutate_host, runner

    monkeypatch.delenv("STATEFRAME_MUTATIONS", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    state_home = tmp_path / "state-home"
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    argv = [
        "run", "--repo", str(repo), "--task", "inspect sample.py", "--endpoint",
        "http://127.0.0.1:8080/v1", "--model", "test-model", "--run-dir", str(run_dir),
        *flags,
    ]
    args = cli.build_parser().parse_args(argv)
    options = cli._run_options(args, _endpoint())
    prepared_run, _, spec = runner.prepare(options, printer=lambda *_: None)
    assert prepared_run == run_dir
    assert spec["mutations"]["level"] == expected_level
    assert spec["mutations"]["shell_network"] == options.tool_settings.shell_network

    resumed_endpoint = Endpoint.from_args("http://127.0.0.1:8081/v1", "test-model", None)
    resumed_spec = runner._resume_spec(run_dir, resumed_endpoint)
    persisted_spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert resumed_spec["mutations"]["shell_network"] == spec["mutations"]["shell_network"]
    assert persisted_spec["mutations"] == spec["mutations"]

    captured = {"policies": [], "runtimes": [], "installed": []}
    real_policy_from = kernel_mutate_policy.policy_from

    def recording_policy_from(serialized):
        policy = real_policy_from(serialized)
        captured["policies"].append((dict(serialized), policy))
        return policy

    class Runtime:
        def __init__(self, **kwargs):
            self.level = kwargs["level"]
            self.policy = kwargs["policy"]
            self.ready = {}
            self.attached_hooks = []
            captured["runtimes"].append(self)

        def attach_hook(self, name, *, event):
            self.attached_hooks.append((name, event))

    class Store:
        def __init__(self, _home):
            pass

        def _locked(self):
            return contextlib.nullcontext()

        def _index_path(self, scope):
            return scope

        def _read_index(self, scope):
            return ["e" * 64] if scope == "user" else []

        def get(self, digest):
            assert digest == "e" * 64
            return {"unit.json": json.dumps({"name": "observer"}).encode("utf-8"),
                    "main.py": b"def handle(input, ctx): return {}\n"}

    monkeypatch.setattr(mutate_host, "mutation_platform_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_mutate_policy, "policy_from", recording_policy_from)
    monkeypatch.setattr(kernel_mutate_records, "Journal", lambda _run: object())
    monkeypatch.setattr(kernel_mutate_runtime, "MutationRuntime", Runtime)
    monkeypatch.setattr(kernel_mutate_runtime, "install",
                        lambda runtime: captured["installed"].append(runtime))
    monkeypatch.setattr(kernel_mutate_sweep, "sweep",
                        lambda *_args, **_kwargs: SimpleNamespace(cleaned=(), unresolved=()))
    monkeypatch.setattr(kernel_mutate_store, "MutationStore", Store)
    monkeypatch.setattr(kernel_mutate_store, "project_id", lambda _repo: "project")

    if error is not None:
        with pytest.raises(ValueError, match=error):
            mutate_host.start(resumed_spec, run_dir)
    else:
        runtime = mutate_host.start(resumed_spec, run_dir)
        assert runtime is captured["installed"][0]
        assert runtime.level == expected_level
        if expected_hook is not None:
            assert runtime.attached_hooks == [expected_hook]
            assert "observer" in runtime.ready

    assert len(captured["policies"]) == 1
    serialized, policy = captured["policies"][0]
    assert serialized == resumed_spec["mutations"]
    assert serialized["level"] == expected_level
    for key, value in expected.items():
        assert serialized[key] == value
    assert policy.level == expected_level
    if "network" in expected:
        assert policy.network == expected["network"]
    if "domains" in expected:
        assert policy.domains == tuple(expected["domains"])
    if "write_globs" in expected:
        assert policy.write_globs == tuple(expected["write_globs"])
    if "retain" in expected:
        assert policy.retain == expected["retain"]


def test_promotion_without_registered_consumer_fails_before_export(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    destination = tmp_path / "unregistered"
    assert cli.main(["mutations", "promote", "file_facts", "--out", str(destination)]) == 2
    assert not destination.exists()
    assert "executor" in capsys.readouterr().err


def test_promotion_consumer_without_attachment_validator_fails_before_export(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "stateframe.powerups", SimpleNamespace(
        supports_isolated_unit_executor=lambda name: True, install=lambda path: None))
    destination = tmp_path / "unvalidated"
    assert cli.main(["mutations", "promote", "file_facts", "--out", str(destination)]) == 2
    assert not destination.exists()
