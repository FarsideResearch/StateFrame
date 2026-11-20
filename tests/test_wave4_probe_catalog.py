"""Startup registration reuse and source-capture parity."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from _child_env import child_env

import kernel_agent_registration as registration
import kernel_agent_schemas
import kernel_agent_tools
import kernel_custom_tools
from kernel_agent_tools import AgentTools
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay, tape_for_run
from probes import react_repair
from test_agent_tools import ProcessBoundary


class ProbeStore:
    """Exercise the consumer with independent reads and a persistent test store."""

    def __init__(self, root):
        self.root = root
        root.mkdir(exist_ok=True)
        self.on = True
        self.events = []

    def enabled(self):
        return self.on

    def host_fingerprint(self):
        return "test-host"

    def lookup(self, kind, fingerprint):
        path = self.root / f"{kind}-{fingerprint}"
        self.events.append((kind, path.exists()))
        return json.loads(path.read_text()) if path.exists() else None

    def publish(self, kind, fingerprint, facts):
        path = self.root / f"{kind}-{fingerprint}"
        path.write_text(json.dumps(facts, sort_keys=True))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    store = ProbeStore(tmp_path / "probes")
    monkeypatch.setitem(sys.modules, "stateframe.send1_probe_cache", store)
    return store


def _definitions(path, description="Print one value", executable="/bin/echo", network="none"):
    definition = {
        "name": "sample_tool", "description": description,
        "parameters": {"type": "object", "properties": {
            "zeta": {"type": "string", "maxLength": 100, "pattern": "[a-z]{1,20}"},
            "alpha": {"type": "boolean"},
        }, "required": ["zeta"]},
        "argv": [executable, "{zeta}", "private-argument"], "network": network,
        "writes": "workspace", "replay": "local", "timeout_seconds": 3,
    }
    path.write_text(json.dumps({"schema": "stateframe-tools/1", "tools": [definition]}))
    return path


def _load(path, *, network="loopback", tokenizer=None, reserved=frozenset({"shell"})):
    return kernel_custom_tools.load_tool_files(
        [path], builtin_names=reserved, shell_network=network, tokenizer=tokenizer)


def test_static_registration_hit_skips_build_and_preserves_order(tmp_path, monkeypatch, store):
    shell = SimpleNamespace(spec={}, settings=ToolSettings())
    custom = _load(_definitions(tmp_path / "tools.json"))
    store.on = False
    reference = registration.registration_data(shell, custom)
    assert all(kind != "agent-registration" for kind, _ in store.events)
    store.on = True
    cold = registration.registration_data(shell, custom)
    assert cold == reference
    assert store.events[-1] == ("agent-registration", False)

    rebuild = registration._build_registration
    builds = []

    def counted(*args):
        builds.append(True)
        return rebuild(*args)

    monkeypatch.setattr(registration, "_build_registration", counted)
    cold.custom_schemas[0]["function"]["description"] = "caller mutation"
    warm = registration.registration_data(shell, custom)
    assert store.events[-1] == ("agent-registration", True)
    assert builds == []
    assert json.dumps(warm.custom_schemas) == json.dumps(reference.custom_schemas)
    assert json.dumps(warm.builtin_schema_snapshot) == json.dumps(reference.builtin_schema_snapshot)
    assert list(warm.custom_schemas[0]["function"]["parameters"]["properties"]) == ["zeta", "alpha", "why"]

    shell.settings = ToolSettings(web=True)
    registration.registration_data(shell, custom)
    assert store.events[-1] == ("agent-registration", False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    registration.registration_data(shell, custom)
    assert store.events[-1] == ("agent-registration", False)

    item = kernel_agent_schemas.BUILTIN_SCHEMAS[0]["function"]
    monkeypatch.setitem(item, "description", item["description"] + " Changed.")
    registration.registration_data(shell, custom)
    assert store.events[-1] == ("agent-registration", False)
    assert len(builds) == 3

    for candidate in store.root.glob("agent-registration-*"):
        candidate.write_text("broken JSON")
    registration.registration_data(shell, custom)
    assert len(builds) == 4
    assert all("private-argument" not in candidate.read_text() for candidate in store.root.iterdir())


def test_normalized_definition_hit_revalidates_executable_and_admission(tmp_path, monkeypatch, store):
    path = _definitions(tmp_path / "tools.json")
    reference = _load(path)
    validate = kernel_custom_tools._parameters
    validations = []

    def counted(*args):
        validations.append(True)
        return validate(*args)

    monkeypatch.setattr(kernel_custom_tools, "_parameters", counted)
    assert _load(path) == reference
    assert store.events[-1] == ("custom-definitions", True)
    assert validations == []
    with pytest.raises(kernel_custom_tools.CustomToolError, match="1501 tokens exceeds 1500"):
        _load(path, tokenizer=lambda wire: 1501)
    with pytest.raises(kernel_custom_tools.CustomToolError, match="duplicates a built-in"):
        _load(path, reserved=frozenset({"sample_tool"}))

    _definitions(path, network="loopback")
    _load(path)
    with pytest.raises(kernel_custom_tools.CustomToolError, match="exceeds --shell-network"):
        _load(path, network="none")

    for candidate in store.root.glob("custom-definitions-*"):
        candidate.write_text('{"definitions":[null]}')
    assert _load(path).tools[0].network == "loopback"
    assert len(validations) == 3


def test_host_validation_remains_eager_with_a_warm_child_receipt(tmp_path, store):
    from stateframe.engine.kernel_custom_tools import load_tool_files

    path = _definitions(tmp_path / "tools.json")
    reference = _load(path)
    store.events.clear()
    validated = load_tool_files([path], builtin_names=frozenset({"shell"}), shell_network="loopback")
    assert validated.digest == reference.digest
    assert validated.schemas() == reference.schemas()
    assert store.events == []


def _new_process(root, path):
    return subprocess.run(
        [sys.executable, "-c", """
import json
import sys
from pathlib import Path
from test_wave4_probe_catalog import ProbeStore, _load
sys.modules["stateframe.send1_probe_cache"] = store = ProbeStore(Path(sys.argv[1]))
import kernel_custom_tools
original = kernel_custom_tools._parameters
checks = []
def validate(*args):
    checks.append(True)
    return original(*args)
kernel_custom_tools._parameters = validate
tools = _load(Path(sys.argv[2]))
tool, = tools.tools
print(json.dumps([store.events, len(checks), tools.digest, str(tool.executable), tools.schemas()]))
""", str(root), str(path)], check=True, capture_output=True, text=True, env=child_env(),
    )


def test_cross_process_definition_executable_and_path_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    local = tmp_path / "bin"
    local.mkdir()
    interpreter = tmp_path / "interpreter"
    interpreter.write_text("#!/bin/sh\n# one\n")
    interpreter.chmod(0o755)
    executable = local / "catalog-test-command"
    executable.write_text(f"#!{interpreter}\necho one\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(local))
    definition = _definitions(tmp_path / "tools.json", executable=executable.name)
    root = tmp_path / "probes"

    def launch():
        return json.loads(_new_process(root, definition).stdout)

    cold = launch()
    assert cold[0] == [["custom-definitions", False]]
    assert cold[1] == 1
    warm = launch()
    assert warm[0] == [["custom-definitions", True]]
    assert warm[1] == 0
    assert warm[2:] == cold[2:]

    executable.write_text(f"#!{interpreter}\necho two\n")
    changed_executable = launch()
    assert changed_executable[1] == 1
    assert changed_executable[2] != cold[2]
    interpreter.write_text("#!/bin/sh\n# two\n")
    changed_dependency = launch()
    assert changed_dependency[1] == 1
    assert changed_dependency[2] == changed_executable[2]
    _definitions(definition, description="Print one item!", executable=executable.name)
    changed_input = launch()
    assert changed_input[0] == [["custom-definitions", False]]
    assert changed_input[4] != cold[4]

    second_path = tmp_path / "other"
    second_path.mkdir()
    second = second_path / executable.name
    second.write_bytes(executable.read_bytes())
    second.chmod(0o755)
    monkeypatch.setenv("PATH", str(second_path))
    changed_path = launch()
    assert changed_path[0] == [["custom-definitions", False]]
    assert changed_path[3] == str(second)


def test_cached_definitions_keep_first_dispatch_and_refusal_live(tmp_path, store):
    path = _definitions(tmp_path / "tools.json")
    fresh = _load(path)
    cached = _load(path)
    assert fresh.tools[0] is not cached.tools[0]
    assert store.events[-1] == ("custom-definitions", True)

    outcomes = []
    for index, custom in enumerate((fresh, cached)):
        run = tmp_path / f"run{index}"
        root = run / "workspace"
        root.mkdir(parents=True)
        process = ProcessBoundary()
        process.action = lambda cwd: (cwd / "app.py").write_text("x = 2\n")
        (root / "app.py").write_text("x = 1\n")
        shell = ShellTool(run, ToolSettings(run_env=False, git_tool=False),
                          {"send1_wave4": bool(index)}, process)
        shell.bind(root, {"files": {"app.py": "x = 1\n"}, "scoped_edits": False}, set())
        configure_replay(run, active=False)
        if index:
            registration.registration_data(shell, custom)
        tools = AgentTools(shell, custom, defer_schema_budget=True)
        args = {"zeta": "hello", "why": "Verify the adapter"}
        reply = tools.execute("sample_tool", args, "first")
        assert (root / "app.py").read_text() == "x = 2\n"
        record = tape_for_run().lookup("first")
        outcomes.append((tools.names, tools.schemas(), process.calls[0][0], reply.observation,
                         record.definition_digest, record.post_manifest_digest))
        refusal = tools.execute("sample_tool", {"zeta": "123"}, "invalid")
        assert refusal.observation.startswith("REFUSED BAD_ARGUMENTS")
        assert len(process.calls) == 1
    assert outcomes[0] == outcomes[1]


def test_provider_registration_and_replay_use_eager_static_inputs(tmp_path, store):
    shell = SimpleNamespace(spec={}, settings=ToolSettings())
    custom = _load(_definitions(tmp_path / "tools.json"))
    cached = registration.registration_data(shell, custom)
    store.events.clear()
    for servers, resume in (([object()], False), ([object(), object()], False), ((), True)):
        assert registration.registration_data(shell, custom, servers, resume_active=resume) == cached
    assert store.events == []


@pytest.mark.parametrize("mode", ["disabled", "enabled", "resume"])
def test_builtin_schema_budget_is_enforced_before_the_menu(tmp_path, monkeypatch, store, mode):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "off" if mode == "disabled" else "on")
    shell = SimpleNamespace(
        spec={"chat_url": "http://127.0.0.1:1/v1/chat/completions",
              "send1_wave4": mode != "disabled"},
        settings=ToolSettings(git_tool=False), run_dir=tmp_path,
    )
    custom = kernel_custom_tools.load_tool_files((), builtin_names=frozenset(), shell_network="none")

    def prepare(*args, **kwargs):
        return kernel_agent_schemas.prepare_schema_measurement(
            *args, **kwargs, tokenizer=lambda _content: 1201)

    monkeypatch.setattr(kernel_agent_tools, "prepare_schema_measurement", prepare)
    with pytest.raises(kernel_custom_tools.CustomToolError, match="schema exceeds 1200 tokens: 1201"):
        tools = AgentTools(shell, custom, resume_active=mode == "resume", defer_schema_budget=True)
        tools.bind()
        tools.schemas()


def test_initial_sources_eager_without_view(tmp_path, monkeypatch, store):
    monkeypatch.setitem(sys.modules, "stateframe.workspace_start_snapshot", None)
    (tmp_path / "app.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "asset.dat").write_bytes(b"\xffx\r")
    (tmp_path / "test_app.py").write_text("skip\n")
    (tmp_path / "alias.py").symlink_to("app.py")
    row = {"files": {"alias.py": {"parent_src": "old"}, "missing.py": {"parent_src": "old"}},
           "tier": "USER-REPO", "scoped_edits": False}
    fileset = react_repair.FileSet.from_row(row)
    paths, sources = registration.initial_product_sources(tmp_path, fileset, row)
    assert paths == ("alias.py", "app.py", "asset.dat", "missing.py")
    assert sources == {"alias.py": "x = 1\n", "app.py": "x = 1\n",
                       "asset.dat": "x\n", "missing.py": None}
    assert store.events == []


def test_admitted_source_membership_skips_walk_and_reads_effective_overlay(tmp_path, monkeypatch, store):
    root = tmp_path / "worker"
    root.mkdir()
    (root / "app.py").write_text("overlay = 1\n")
    (root / "asset.dat").write_bytes(b"\xffa\r\n")
    (root / "test_app.py").write_text("test\n")
    row = {"files": {"app.py": {"parent_src": "declared = 0\n"},
                     "missing.py": {"parent_src": "declared = 0\n"}},
           "tier": "USER-REPO", "scoped_edits": False}
    fileset = react_repair.FileSet.from_row(row)
    eager_paths = react_repair._product_source_paths(root, fileset, row)
    eager_sources = react_repair._initial_source_snapshot(root, eager_paths)
    views = ModuleType("stateframe.workspace_start_snapshot")
    initial = SimpleNamespace(root=root, purpose="initial-product-sources",
                              validate_for_use=lambda purpose: purpose == "initial-product-sources",
                              files={"test_app.py": "hash", "asset.dat": "hash", "app.py": "hash"})
    views.startup_observation = lambda root, purpose: initial
    monkeypatch.setitem(sys.modules, views.__name__, views)

    def unexpected_walk(*args):
        pytest.fail("admitted membership caused another product walk")

    with monkeypatch.context() as bound:
        bound.setattr(react_repair, "_product_source_paths", unexpected_walk)
        paths, sources = registration.initial_product_sources(root, fileset, row)
    assert (paths, sources) == (eager_paths, eager_sources)
    assert sources == {"app.py": "overlay = 1\n", "asset.dat": "a\n", "missing.py": None}

    (root / "app.py").write_text("overlay = 2\n")
    (root / "added.py").write_text("new = True\n")
    initial.validate_for_use = lambda purpose: False
    paths, sources = registration.initial_product_sources(root, fileset, row)
    assert paths == ("added.py", "app.py", "asset.dat", "missing.py")
    assert sources["app.py"] == "overlay = 2\n"
    assert store.events == []
