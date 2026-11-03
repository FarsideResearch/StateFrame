"""Section 14 attachment, trust, and preparation regressions."""

import hashlib
import json
from pathlib import Path

import pytest

from _portable import TRUE

from stateframe import powerups
from stateframe.powerup_attachment import read_attachments
from stateframe.powerup_state import user_state
from test_powerups import unit


def _write(path: Path, name: str, rows: list[dict]) -> None:
    (path / name).write_text(json.dumps({"schema": {
        "commands.json": "stateframe-tools/1",
        "hooks.json": "stateframe-hooks/1",
        "mcp.json": "stateframe-mcp/1",
    }[name], {"commands.json": "tools", "hooks.json": "hooks",
              "mcp.json": "servers"}[name]: rows}))


def _command(**overrides):
    return {"name": "sample_tool", "description": "A sample.",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "argv": [TRUE], "network": "none", "writes": "none",
            "replay": "local", "timeout_seconds": 5, **overrides}


def _mutation_attachment():
    ceiling = {"network": "none", "writes": [], "domains": []}
    digest = hashlib.sha256(json.dumps(ceiling, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return {"schema": "stateframe-isolated-unit-attachment/1",
            "unit_digest": "sha256:" + "a" * 64, "executor": "mutation",
            "capability_ceiling_digest": "sha256:" + digest,
            "capability_ceiling": ceiling, "replay_policy": "observation",
            "binding": {"kind": "command", "name": "mutation_tool"}}


def test_shell_parts_have_canonical_attachment_records(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "commands.json", [_command()])
    _write(source, "hooks.json", [{"name": "note", "event": "post_tool",
                                    "argv": [TRUE]}])
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert len(selected.active) == 1
    records = selected.active[0].record()["attachments"]
    assert {(r["executor"], r["binding"]["kind"], r["binding"]["name"])
            for r in records} == {("shell", "command", "sample_tool"),
                                  ("shell", "hook", "note")}
    assert all(r["schema"] == "stateframe-isolated-unit-attachment/1" for r in records)


def test_writing_hook_attachment_uses_local_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "hooks.json", [{"name": "format", "event": "post_tool",
                                    "argv": [TRUE], "writes": "changed"}])
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert selected.active[0].attachments[0]["replay_policy"] == "local"


def test_unavailable_executor_notice_and_no_shell_downgrade(tmp_path, monkeypatch):
    from stateframe.runner import RunOptions

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "commands.json", [_command()])
    (source / "units.json").write_text(json.dumps([_mutation_attachment()]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    options = RunOptions(repo=tmp_path, task="", endpoint=None)
    notices = []
    powerups.attach_units(options, selected, printer=notices.append)
    assert notices == ["powerup sample disabled: executor mutation unavailable"]
    assert selected.skipped[-1]["code"] == "executor_unavailable"
    assert not options.tool_settings.tools_files


@pytest.mark.parametrize("part,network,replay", [
    ("commands.json", "loopback", "local"),
    ("commands.json", "full", "observation"),
    ("mcp.json", "loopback", "local"),
])
def test_networked_command_or_mcp_is_skipped(tmp_path, monkeypatch, part, network, replay):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    row = _command(network=network, replay=replay) if part == "commands.json" else {
        "name": "sample_server", "trust": "local", "argv": [TRUE],
        "network": network, "writes": "none", "replay": replay,
        "tools": []}
    _write(source, part, [row])
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (),
                                      False, shell_network="full")
    assert not selected.active
    assert "external replay" in selected.skipped[0]["reason"]


def test_promoted_gating_hook_is_filtered_until_digest_bound_trust(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path, "promoted", default="on")
    _write(source, "hooks.json", [
        {"name": "observe", "event": "post_tool", "argv": [TRUE]},
        {"name": "gate", "event": "stop", "argv": [TRUE]},
    ])
    installed = powerups.install(source, provenance="promoted from mutation x in run")
    assert powerups.effective_state(installed, "default")["mode"] == "off"
    powerups.enable("promoted")
    selected = powerups.resolve_units(tmp_path, "", "default", (), (), False)
    assert len(selected.active) == 1
    hooks = json.loads((selected.active[0].materialized_dir / "hooks.json").read_text())
    assert [row["name"] for row in hooks["hooks"]] == ["observe"]
    assert any(row.get("code") == "gating_untrusted" for row in selected.skipped)
    powerups.enable("promoted", trust_gating=True)
    state = user_state()["global"]["promoted"]
    assert state["trust_gating"] == powerups.tree_digest(installed.path)
    selected = powerups.resolve_units(tmp_path, "", "default", (), (), False)
    hooks = json.loads((selected.active[0].materialized_dir / "hooks.json").read_text())
    assert {row["name"] for row in hooks["hooks"]} == {"observe", "gate"}


def test_promoted_session_start_is_never_attached(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path, "promoted")
    _write(source, "hooks.json", [
        {"name": "start", "event": "session_start", "argv": [TRUE]},
        {"name": "end", "event": "session_end", "argv": [TRUE]},
    ])
    powerups.install(source, provenance="promoted from mutation x in run")
    powerups.enable("promoted", trust_gating=True)
    selected = powerups.resolve_units(tmp_path, "", "default", (), (), False)
    hooks = json.loads((selected.active[0].materialized_dir / "hooks.json").read_text())
    assert [row["name"] for row in hooks["hooks"]] == ["end"]
    assert selected.skipped[0]["code"] == "event_unavailable"


def test_tampered_attachment_ceiling_digest_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    record = _mutation_attachment()
    record["capability_ceiling_digest"] = "sha256:" + "0" * 64
    (source / "units.json").write_text(json.dumps([record]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert not selected.active
    assert "digest" in selected.skipped[0]["reason"]


def test_declared_shell_attachment_cannot_claim_phantom_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    record = _mutation_attachment()
    record["executor"] = "shell"
    (source / "units.json").write_text(json.dumps([record]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert not selected.active
    assert "binding" in selected.skipped[0]["reason"]


def test_networked_hook_requires_matching_external_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "hooks.json", [{"name": "network_note", "event": "session_end",
                                    "argv": [TRUE], "network": "loopback"}])
    ceiling = {"network": "public-web", "writes": [], "domains": []}
    digest = hashlib.sha256(json.dumps(ceiling, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    record = {"schema": "stateframe-isolated-unit-attachment/1",
              "unit_digest": "sha256:" + "a" * 64, "executor": "shell",
              "capability_ceiling_digest": "sha256:" + digest,
              "capability_ceiling": ceiling, "replay_policy": "external",
              "binding": {"kind": "hook", "name": "network_note", "event": "session_end"}}
    (source / "units.json").write_text(json.dumps([record]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (),
                                      False, shell_network="loopback")
    assert len(selected.active) == 1
    assert selected.active[0].attachments[0]["replay_policy"] == "external"


def test_attachment_reader_rejects_linked_units_file(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("[]")
    (tmp_path / "units.json").symlink_to(target)
    with pytest.raises(powerups.PowerupError):
        read_attachments(tmp_path)


def test_explicit_and_install_root_links_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(source, target_is_directory=True)
    selected = powerups.resolve_units(tmp_path, "", "default", (str(link),), (), False)
    assert not selected.active
    assert "linked" in selected.skipped[0]["reason"]
    with pytest.raises(powerups.PowerupError, match="linked"):
        powerups.install(link)


def test_promoted_enable_revalidates_filtered_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path, "promoted")
    _write(source, "hooks.json", [{"name": "start", "event": "session_start",
                                    "argv": [TRUE]}])
    installed = powerups.install(source, provenance="promoted from mutation x in run")
    _write(installed.path, "hooks.json", [{"name": "start", "event": "session_start",
                                             "argv": ["/missing/program"]}])
    with pytest.raises(powerups.PowerupError):
        powerups.enable("promoted")


def test_resume_rejects_tampered_attachment_record(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "commands.json", [_command()])
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    record = selected.record()
    record["active"][0]["attachments"][0]["executor"] = "mutation"
    with pytest.raises(powerups.PowerupError, match="POWERUP_CHANGED"):
        powerups.verify_records(record)


def test_registered_mutation_keeps_shell_binding(tmp_path, monkeypatch):
    from stateframe.runner import RunOptions

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "commands.json", [_command()])
    (source / "units.json").write_text(json.dumps([_mutation_attachment()]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    called = []
    monkeypatch.setitem(powerups._EXECUTORS, "mutation", lambda options, unit: called.append(unit.name))
    options = RunOptions(repo=tmp_path, task="", endpoint=None)
    powerups.attach_units(options, selected, printer=lambda _: None)
    assert called == ["sample"]
    assert len(options.tool_settings.tools_files) == 1


def test_hook_only_powerup_does_not_start_user_mcp_for_budget(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_tool_settings import ToolSettings
    from stateframe.runner import RunOptions
    from stateframe.powerup_prepare import prepare_run_inputs

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    _write(source, "hooks.json", [{"name": "note", "event": "session_end",
                                    "argv": [TRUE]}])
    mcp = tmp_path / "user-mcp.json"
    mcp.write_text("invalid")
    options = RunOptions(repo=tmp_path, task="inspect",
                         endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None),
                         powerup_requests=(str(source),),
                         tool_settings=ToolSettings(mcp_files=({"path": str(mcp), "digest": "pending"},)))
    selected = prepare_run_inputs(options, {}, lambda _: None)
    assert len(selected.active) == 1


def test_broken_project_trust_store_skips_project_unit(tmp_path, monkeypatch):
    from stateframe import hook_host, powerup_state

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    unit(repo / ".stateframe" / "powerups")
    def broken():
        raise hook_host.HookTrustError("broken trust store")
    monkeypatch.setattr(powerup_state.hook_host, "_read_trust_store", broken)
    selected = powerups.resolve_units(repo, "", "default", (), (), False)
    assert not selected.active
    assert any("broken trust store" in row["reason"] for row in selected.skipped)


def test_changed_materialized_unit_is_skipped_before_attach(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    original = powerups._validate
    def tamper(manifest, materialized, parts, **kwargs):
        original(manifest, materialized, parts, **kwargs)
        (materialized / "SKILL.md").write_text("changed after validation")
    monkeypatch.setattr(powerups, "_validate", tamper)
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert not selected.active
    assert "between validation and attachment" in selected.skipped[0]["reason"]


def test_setting_is_one_argv_operand(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    with (source / "powerup.toml").open("a") as output:
        output.write('[settings.label]\ntype = "string"\ndefault = "safe"\nmax_length = 100\n')
    _write(source, "commands.json", [_command(argv=["/bin/echo", "{setting:label}"])])
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),),
                                      ("sample.label=hello; echo injected",), False)
    assert len(selected.active) == 1
    command = json.loads((selected.active[0].materialized_dir / "commands.json").read_text())
    assert command["tools"][0]["argv"] == ["/bin/echo", "hello; echo injected"]


def test_promote_rejects_linked_recorded_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    recorded = unit(tmp_path, "recorded")
    link = tmp_path / "linked"
    link.symlink_to(recorded, target_is_directory=True)
    with pytest.raises(powerups.PowerupError, match="linked"):
        powerups.promote(tmp_path / "run", "mut-1", "promoted", lambda *_: link)


def test_powerup_mcp_snapshot_failure_is_isolated(tmp_path):
    from types import SimpleNamespace
    from stateframe.powerup_budget import admit_powerup_units
    from stateframe.workcopy import WorkCopyError

    selection = powerups.Selection(active=[SimpleNamespace(
        name="sample", parts=("mcp",), materialized_dir=tmp_path)])
    def broken(_paths):
        raise WorkCopyError("snapshot unavailable")
    admit_powerup_units(selection, user_tools=(), mcp_schemas=(),
                        shell_network="none", endpoint=None,
                        discover_mcp_schemas=broken, printer=lambda _: None)
    assert not selection.active
    assert "snapshot unavailable" in selection.skipped[0]["reason"]


def test_install_rejects_source_change_after_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    original = powerups._validate
    def mutate(manifest, rendered, parts, **kwargs):
        original(manifest, rendered, parts, **kwargs)
        (source / "commands.json").write_text("{}")
    monkeypatch.setattr(powerups, "_validate", mutate)
    with pytest.raises(powerups.PowerupError, match="changed"):
        powerups.install(source)
    assert not (tmp_path / "home/powerups/sample").exists()


def test_broken_state_does_not_abort_prepare(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.runner import RunOptions, prepare

    home = tmp_path / "home"
    home.mkdir()
    (home / "powerups.json").write_text("{invalid")
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    notices = []
    _, _, spec = prepare(RunOptions(repo=repo, task="inspect", run_dir=tmp_path / "run",
                                   endpoint=Endpoint.from_args(
                                       "http://127.0.0.1:8080/v1", "scripted", None)),
                         printer=notices.append)
    assert spec["powerups"]["skipped"][0]["code"] == "state_invalid"
    assert any("powerup state disabled" in notice for notice in notices)


def test_no_powerups_does_not_load_optional_tools_or_mcp(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.runner import RunOptions
    from stateframe.powerup_prepare import prepare_run_inputs

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError("optional discovery or tool loading ran")
    monkeypatch.setattr("stateframe.powerup_prepare.discover_mcp", forbidden)
    monkeypatch.setattr("stateframe.powerup_prepare.load_tool_files", forbidden)
    options = RunOptions(repo=repo, task="inspect",
                         endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None))
    before = tuple(repo.iterdir())
    selection = prepare_run_inputs(options, {}, lambda _: None)
    assert not selection.active and not selection.skipped
    assert tuple(repo.iterdir()) == before


@pytest.mark.parametrize("event, attached", [("session_end", True), ("post_tool", False)])
def test_networked_hook_attachment_must_match_event(tmp_path, monkeypatch, event, attached):
    # A registered non-shell attachment is the only record that can vouch for a networked
    # hook, so a record for another event must not cover it.
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setitem(powerups._EXECUTORS, "mutation", lambda options, unit: None)
    source = unit(tmp_path)
    _write(source, "hooks.json", [{"name": "network_note", "event": "session_end",
                                    "argv": [TRUE], "network": "loopback"}])
    ceiling = {"network": "public-web", "writes": [], "domains": []}
    digest = hashlib.sha256(json.dumps(ceiling, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    record = {"schema": "stateframe-isolated-unit-attachment/1",
              "unit_digest": "sha256:" + "a" * 64, "executor": "mutation",
              "capability_ceiling_digest": "sha256:" + digest,
              "capability_ceiling": ceiling, "replay_policy": "external",
              "binding": {"kind": "hook", "name": "network_note", "event": event}}
    (source / "units.json").write_text(json.dumps([record]))
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (),
                                      False, shell_network="loopback")
    assert bool(selected.active) is attached
    if not attached:
        assert "external replay attachment" in selected.skipped[0]["reason"]
