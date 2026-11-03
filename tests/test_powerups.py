"""Host powerup admission and pinning contracts."""

import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from _portable import TRUE

from stateframe import powerup_state, powerups


def unit(root: Path, name: str = "sample", *, default: str = "off") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "powerup.toml").write_text(
        f'schema = "stateframe-powerup/1"\nname = "{name}"\nversion = "1.0"\n'
        f'summary = "Useful sample."\ndefault = "{default}"\n'
        "[activate]\nskill = true\ncommands = true\nhooks = true\n"
    )
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Useful sample.\n---\nRead this.\n"
    )
    return path


def test_manifest_refuses_unknown_keys(tmp_path):
    path = unit(tmp_path)
    with (path / "powerup.toml").open("a") as output:
        output.write("surprise = true\n")
    with pytest.raises(powerups.PowerupError, match="surprise"):
        powerups.parse_manifest(path)


def test_manifest_refuses_oversized_or_linked_tree(tmp_path):
    path = unit(tmp_path)
    (path / "large.bin").write_bytes(b"x" * (1 << 20))
    with pytest.raises(powerups.PowerupError, match="1 MiB"):
        powerups.parse_manifest(path)
    (path / "large.bin").unlink()
    (path / "link").symlink_to(path / "SKILL.md")
    with pytest.raises(powerups.PowerupError, match="symlinks"):
        powerups.parse_manifest(path)


@pytest.mark.parametrize("field", ['default = ["on"]', 'default = { mode = "on" }'])
def test_manifest_rejects_non_scalar_mode(tmp_path, field):
    path = unit(tmp_path)
    manifest = path / "powerup.toml"
    manifest.write_text(manifest.read_text().replace('default = "off"', field))
    with pytest.raises(powerups.PowerupError, match="default mode"):
        powerups.parse_manifest(path)


def test_install_is_off_until_enable_and_has_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path, default="on")
    powerup_state.update("sample", mode="on", profile="special")
    installed = powerups.install(source)
    assert installed.name == "sample"
    assert json.loads((installed.path / "install.json").read_text())[
        "provenance"
    ] == str(source.resolve())
    assert powerups.effective_state(installed, "default")["mode"] == "off"
    assert powerups.effective_state(installed, "special")["mode"] == "off"
    powerups.enable("sample", mode="on")
    assert powerups.effective_state(installed, "default")["mode"] == "on"


def test_install_checks_provenance_size_before_publishing(tmp_path, monkeypatch):
    from stateframe.skills import MAX_DIRECTORY_BYTES

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    used = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
    (source / "filler.bin").write_bytes(b"x" * (MAX_DIRECTORY_BYTES - used - 2))
    with pytest.raises(powerups.PowerupError, match="1 MiB"):
        powerups.install(source)
    assert not (tmp_path / "home/powerups/sample").exists()


def test_force_install_preserves_broken_previous_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    installed = powerups.install(source)
    (installed.path / "linked").symlink_to(installed.path / "SKILL.md")
    replacement = powerups.install(source, force=True)
    assert replacement.name == "sample"
    previous = list((tmp_path / "home/powerups").glob("sample.previous-*"))
    assert len(previous) == 1 and (previous[0] / "linked").is_symlink()


def test_remove_can_remove_broken_installed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    installed = powerups.install(unit(tmp_path))
    (installed.path / "powerup.toml").write_text("broken = true\n")
    powerups.remove(installed.name)
    assert not installed.path.exists()
    assert powerup_state.user_state()["global"][installed.name]["mode"] == "off"


def test_offer_attaches_only_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    selected = powerups.resolve_units(
        tmp_path, "read sample", "default", (str(source) + "=offer",), (), False
    )
    assert len(selected.active) == 1
    assert selected.active[0].mode == "offer"
    assert selected.active[0].parts == ("skill",)


def test_empty_attach_leaves_option_objects_untouched(tmp_path):
    from stateframe.runner import RunOptions

    options = RunOptions(repo=tmp_path, task="test", endpoint=None)
    settings = options.tool_settings
    powerups.attach_units(options, powerups.Selection())
    assert options.tool_settings is settings


def test_materialized_digest_refuses_change(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    selected = powerups.resolve_units(
        tmp_path, "read sample", "default", (str(source) + "=offer",), (), False
    )
    record = selected.active[0].record()
    powerups.verify_records({"active": [record], "skipped": []})
    (Path(record["materialized_dir"]) / "SKILL.md").write_text("changed")
    with pytest.raises(powerups.PowerupError, match="POWERUP_CHANGED"):
        powerups.verify_records({"active": [record], "skipped": []})


def test_discovery_project_trust_and_cli_path_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    project = unit(repo / ".stateframe" / "powerups")
    user = unit(tmp_path / "home" / "powerups")
    explicit = unit(tmp_path / "explicit")
    powerup_state.update("sample", mode="on", digest=powerups.tree_digest(user))
    entries = []
    monkeypatch.setattr(
        powerup_state.hook_host, "_read_trust_store", lambda: entries[:]
    )
    monkeypatch.setattr(
        powerup_state.hook_host,
        "_write_trust_store",
        lambda values: entries.__setitem__(slice(None), values),
    )
    assert (
        powerups.resolve_units(repo, "", "default", (), (), False).active[0].source
        == user.resolve()
    )
    powerup_state.trust(repo)
    assert (
        powerups.resolve_units(repo, "", "default", (), (), False).active[0].source
        == project.resolve()
    )
    assert (
        powerups.resolve_units(repo, "", "default", (str(explicit),), (), False)
        .active[0]
        .source
        == explicit.resolve()
    )


def test_state_profile_project_cli_and_no_powerups(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    installed = powerups.install(unit(tmp_path))
    powerup_state.update(
        "sample", mode="on", settings={}, digest=powerups.tree_digest(installed.path)
    )
    powerup_state.update("sample", mode="off", profile="special")
    assert powerups.effective_state(installed, "special")["mode"] == "off"
    assert (
        powerups.effective_state(installed, "special", cli={"sample": "offer"})["mode"]
        == "offer"
    )
    assert (
        powerups.resolve_units(repo, "", "special", ("sample=on",), (), True).active
        == []
    )


def test_trusted_project_state_can_only_enable_project_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    config_dir = repo / ".stateframe"
    config_dir.mkdir(parents=True)
    powerups.install(unit(tmp_path))
    (config_dir / "powerups.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-powerup-state/1",
                "global": {"sample": {"mode": "on"}},
                "profiles": {},
            }
        )
    )
    entries = []
    monkeypatch.setattr(
        powerup_state.hook_host, "_read_trust_store", lambda: entries[:]
    )
    monkeypatch.setattr(
        powerup_state.hook_host,
        "_write_trust_store",
        lambda values: entries.__setitem__(slice(None), values),
    )
    powerup_state.trust(repo)
    assert not powerups.resolve_units(repo, "", "default", (), (), False).active
    unit(config_dir / "powerups")
    powerup_state.trust(repo)
    assert (
        powerups.resolve_units(repo, "", "default", (), (), False).active[0].source
        == (config_dir / "powerups/sample").resolve()
    )


def test_auto_rule_and_setting_render_and_gating_default(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    source = unit(tmp_path)
    with (source / "powerup.toml").open("a") as output:
        output.write(
            '[offer]\nany_files = ["pyproject.toml"]\nloops = ["composed"]\n'
            '[settings.limit]\ntype = "integer"\ndefault = 10\nmin = 1\nmax = 20\n'
        )
    (source / "hooks.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "gate",
                        "event": "stop",
                        "argv": ["/bin/echo", "{setting:limit}"],
                    }
                ],
            }
        )
    )
    (repo / "pyproject.toml").write_text("[project]\n")
    selected = powerups.resolve_units(
        repo, "", "default", (str(source) + "=auto",), ("sample.limit=12",), False
    )
    assert selected.active[0].mode == "on"
    hook = json.loads((selected.active[0].materialized_dir / "hooks.json").read_text())[
        "hooks"
    ][0]
    assert hook["argv"][-1] == "12"
    assert hook["on_error"] == "continue"
    with pytest.raises(powerups.PowerupError, match="out of bounds"):
        powerups.effective_state(
            powerups.parse_manifest(source), "default", cli_settings={"limit": "21"}
        )
    (repo / "pyproject.toml").unlink()
    assert not powerups.resolve_units(
        repo, "", "default", (str(source) + "=auto",), (), False
    ).active


def test_broken_unit_skips_without_disabling_other_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    good = unit(tmp_path, "good")
    broken = unit(tmp_path, "broken")
    (broken / "powerup.toml").write_text("invalid = true\n")
    chosen = powerups.resolve_units(
        tmp_path, "", "default", (str(good), str(broken)), (), False
    )
    assert [item.name for item in chosen.active] == ["good"]
    assert chosen.skipped[0]["name"] == "broken"
    (good / "commands.json").write_text('{"schema":"stateframe-tools/1","tools":null}')
    malformed = powerups.resolve_units(tmp_path, "", "default", (str(good),), (), False)
    assert malformed.skipped[0]["name"] == "good"


def test_changed_enabled_unit_skips_without_aborting(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    installed = powerups.install(unit(tmp_path))
    powerups.enable(installed.name)
    with (installed.path / "SKILL.md").open("a") as output:
        output.write("\nChanged after enabling.\n")
    selected = powerups.resolve_units(repo, "", "default", (), (), False)
    assert not selected.active
    assert "changed since enabled" in selected.skipped[0]["reason"]


def test_unpinned_user_state_cannot_enable_installed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    powerups.install(unit(tmp_path))
    powerup_state.update("sample", mode="on")
    selected = powerups.resolve_units(repo, "", "default", (), (), False)
    assert not selected.active
    assert "not enabled" in selected.skipped[0]["reason"]
    assert powerups.resolve_units(repo, "", "default", ("sample=on",), (), False).active


def test_promote_uses_install_validation_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    recorded = unit(tmp_path, "recorded")
    promoted = powerups.promote(
        tmp_path / "run", "mut-1", "promoted", lambda run, mutation: recorded
    )
    metadata = json.loads((promoted.path / "install.json").read_text())
    assert "promoted from mutation mut-1" in metadata["provenance"]
    assert powerups.effective_state(promoted, "default")["mode"] == "off"
    (recorded / "commands.json").write_text("{}")
    with pytest.raises(powerups.PowerupError):
        powerups.promote(
            tmp_path / "run", "mut-2", "another", lambda run, mutation: recorded
        )


def test_unavailable_mutation_executor_is_skipped_at_attach(tmp_path, monkeypatch):
    from stateframe.runner import RunOptions

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    ceiling = {"network": "none", "writes": [], "domains": []}
    ceiling_sha = hashlib.sha256(
        json.dumps(ceiling, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (source / "units.json").write_text(
        json.dumps(
            [
                {
                    "schema": "stateframe-isolated-unit-attachment/1",
                    "unit_digest": "sha256:" + "a" * 64,
                    "executor": "mutation",
                    "capability_ceiling_digest": "sha256:" + ceiling_sha,
                    "capability_ceiling": ceiling,
                    "replay_policy": "observation",
                    "binding": {"kind": "command", "name": "mutation_tool"},
                }
            ]
        )
    )
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False
    )
    assert len(selected.active) == 1
    options = RunOptions(repo=tmp_path, task="test", endpoint=None)
    powerups.attach_units(options, selected, printer=lambda _: None)
    assert not selected.active
    assert selected.skipped == [{"name": "sample", "code": "executor_unavailable",
                                 "reason": "executor mutation unavailable"}]


def test_invalid_isolated_attachment_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    (source / "units.json").write_text(
        json.dumps(
            [
                {
                    "schema": "stateframe-isolated-unit-attachment/1",
                    "unit_digest": "sha256:" + "a" * 64,
                    "executor": ["mutation"],
                    "capability_ceiling_digest": "sha256:" + hashlib.sha256(
                        json.dumps({"network": "none", "writes": [], "domains": []},
                                   sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "capability_ceiling": {
                        "network": "none",
                        "writes": [],
                        "domains": [],
                    },
                    "replay_policy": "observation",
                    "binding": {"kind": "command", "name": "mutation_tool"},
                }
            ]
        )
    )
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False
    )
    assert not selected.active
    assert "invalid isolated unit executor" in selected.skipped[0]["reason"]


def test_mutation_attachment_rejects_local_replay(tmp_path):
    from stateframe.powerup_attachment import read_attachments

    (tmp_path / "units.json").write_text(
        json.dumps(
            [
                {
                    "schema": "stateframe-isolated-unit-attachment/1",
                    "unit_digest": "sha256:" + "a" * 64,
                    "executor": "mutation",
                    "capability_ceiling_digest": "sha256:" + hashlib.sha256(
                        json.dumps({"network": "none", "writes": [], "domains": []},
                                   sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "capability_ceiling": {
                        "network": "none",
                        "writes": [],
                        "domains": [],
                    },
                    "replay_policy": "local",
                    "binding": {"kind": "command", "name": "mutation_tool"},
                }
            ]
        )
    )
    with pytest.raises(powerups.PowerupError, match="local replay"):
        read_attachments(tmp_path)


def test_promoted_gating_hook_needs_explicit_trust(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path, "promoted")
    (source / "hooks.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "gate",
                        "event": "stop",
                        "argv": ["/bin/echo", "ok"],
                        "on_error": "block",
                    }
                ],
            }
        )
    )
    installed = powerups.install(
        source, provenance="promoted from mutation abc in /tmp/run"
    )
    powerups.enable(installed.name)
    selected = powerups.resolve_units(tmp_path, "", "default", (), (), False)
    assert selected.skipped[0]["code"] == "gating_untrusted"
    assert json.loads((selected.active[0].materialized_dir / "hooks.json").read_text())["hooks"] == []
    powerups.enable(installed.name, mode="offer")
    powerups.enable(installed.name)
    powerups.enable(installed.name, trust_gating=True)
    assert powerups.effective_state(installed, "default")["mode"] == "on"
    assert powerup_state.user_state()["global"][installed.name]["trust_gating"] == powerups.tree_digest(installed.path)


def test_networked_powerup_command_rejects_local_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    (source / "commands.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-tools/1",
                "tools": [
                    {
                        "name": "network_tool",
                        "description": "Network test.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        "argv": ["/bin/echo"],
                        "network": "loopback",
                        "writes": "none",
                        "replay": "local",
                        "timeout_seconds": 5,
                    }
                ],
            }
        )
    )
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False, shell_network="loopback"
    )
    assert not selected.active
    assert "external replay" in selected.skipped[0]["reason"]


def test_networked_powerup_hook_requires_replay_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    (source / "hooks.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "network-note",
                        "event": "session_end",
                        "argv": ["/bin/echo", "ok"],
                        "network": "loopback",
                    }
                ],
            }
        )
    )
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False, shell_network="loopback"
    )
    assert not selected.active
    assert "external replay attachment" in selected.skipped[0]["reason"]


def test_user_tool_keeps_first_budget_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)

    def definition(name):
        props = {"vv": {"type": "string", "maxLength": 10}}
        props.update(
            {
                f"extra{i}": {
                    "type": "string",
                    "maxLength": 10,
                    "description": "x" * 300,
                }
                for i in range(7)
            }
        )
        return {
            "name": name,
            "description": "x" * 300,
            "parameters": {"type": "object", "properties": props, "required": ["vv"]},
            "argv": ["/bin/echo", "{vv}"],
            "network": "none",
            "writes": "none",
            "replay": "local",
            "timeout_seconds": 5,
        }

    user = tmp_path / "user-tools.json"
    user.write_text(
        json.dumps({"schema": "stateframe-tools/1", "tools": [definition("user_tool")]})
    )
    (source / "commands.json").write_text(
        json.dumps(
            {"schema": "stateframe-tools/1", "tools": [definition("power_tool")]}
        )
    )
    from stateframe.engine.kernel_custom_tools import load_tool_files

    assert (
        load_tool_files([user], builtin_names=powerups.BUILTINS, shell_network="none")
        .tools[0]
        .name
        == "user_tool"
    )
    chosen = powerups.resolve_units(
        tmp_path,
        "",
        "default",
        (str(source),),
        (),
        False,
        shell_network="none",
        user_tools=(user,),
    )
    assert not chosen.active
    assert chosen.skipped[0]["name"] == "sample"
    assert "1500" in chosen.skipped[0]["reason"]


def test_rejected_first_unit_does_not_consume_later_schema_capacity(
    tmp_path, monkeypatch
):
    from stateframe import powerup_budget
    from stateframe.engine.kernel_custom_tools import CustomToolError, load_tool_files

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    first, second = unit(tmp_path, "aa"), unit(tmp_path, "bb")
    small = {
        "name": "b_tool",
        "description": "Small command.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "argv": [TRUE],
        "network": "none",
        "writes": "none",
        "replay": "local",
        "timeout_seconds": 5,
    }
    (second / "commands.json").write_text(
        json.dumps({"schema": "stateframe-tools/1", "tools": [small]})
    )
    found = False
    for size in range(5, 301, 5):
        props = {
            f"p{index}": {
                "type": "string",
                "maxLength": 300,
                "description": "d" * 150,
                "enum": [str(choice) + "x" * size for choice in range(3)],
            }
            for index in range(8)
        }
        large = {
            **small,
            "name": "a_tool",
            "description": "Large command.",
            "parameters": {"type": "object", "properties": props, "required": []},
        }
        (first / "commands.json").write_text(
            json.dumps({"schema": "stateframe-tools/1", "tools": [large]})
        )
        try:
            load_tool_files(
                [first / "commands.json"],
                builtin_names=powerups.BUILTINS,
                shell_network="none",
            )
        except CustomToolError:
            break
        try:
            load_tool_files(
                [first / "commands.json", second / "commands.json"],
                builtin_names=powerups.BUILTINS,
                shell_network="none",
            )
        except CustomToolError as error:
            if "exceeds 1500" in str(error):
                found = True
                break
            raise
    assert found
    selected = powerups.resolve_units(
        tmp_path,
        "",
        "default",
        (str(first), str(second)),
        (),
        False,
        shell_network="none",
    )
    assert [item.name for item in selected.active] == ["aa", "bb"]

    class Meter:
        def __init__(self, **_):
            pass

        def measure(self, body):
            names = {schema["function"]["name"] for schema in body["tools"]}
            return (2000 if "a_tool" in names else 100, "test")

    monkeypatch.setattr(powerup_budget, "RequestTokenMeter", Meter)
    powerup_budget.admit_powerup_units(
        selected,
        user_tools=(),
        mcp_schemas=(),
        shell_network="none",
        endpoint=None,
        printer=lambda _: None,
    )
    assert [item.name for item in selected.active] == ["bb"]


def test_user_mcp_schema_keeps_first_budget_claim(tmp_path, monkeypatch):
    from stateframe.powerup_budget import admit_powerup_units

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = Path(__file__).resolve().parents[1] / "examples/powerups/repo-grep"
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False
    )
    assert len(selected.active) == 1
    user_schema = {
        "type": "function",
        "function": {
            "name": "user_mcp",
            "description": "x" * 6000,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    admit_powerup_units(
        selected,
        user_tools=(),
        mcp_schemas=(user_schema,),
        shell_network="none",
        endpoint=None,
        printer=lambda _: None,
    )
    assert not selected.active
    assert selected.skipped[0]["name"] == "repo-grep"
    assert "1500" in selected.skipped[0]["reason"]


def test_powerup_mcp_overflow_skips_whole_unit(tmp_path):
    from types import SimpleNamespace

    from stateframe.powerup_budget import admit_powerup_units

    selected = powerups.Selection(
        active=[
            SimpleNamespace(name="sample", parts=("mcp",), materialized_dir=tmp_path)
        ]
    )
    schema = {
        "type": "function",
        "function": {
            "name": "mcp_sample__huge",
            "description": "x" * 6000,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    admit_powerup_units(
        selected,
        user_tools=(),
        mcp_schemas=(),
        shell_network="none",
        endpoint=None,
        discover_mcp_schemas=lambda paths: ([schema], ()),
        printer=lambda _: None,
    )
    assert not selected.active
    assert selected.skipped[0]["name"] == "sample"
    assert "1500" in selected.skipped[0]["reason"]


def test_powerup_mcp_name_collision_skips_unit(tmp_path, monkeypatch):
    from stateframe.powerup_budget import admit_powerup_units

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = Path(__file__).resolve().parents[1] / "examples/powerups/repo-grep"
    selected = powerups.resolve_units(
        tmp_path, "", "default", (str(source),), (), False
    )
    selected.active[0] = replace(selected.active[0], parts=("skill", "commands", "mcp"))
    collision = {
        "type": "function",
        "function": {
            "name": "grep_repo",
            "description": "Collision.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    admit_powerup_units(
        selected,
        user_tools=(),
        mcp_schemas=(),
        shell_network="none",
        endpoint=None,
        discover_mcp_schemas=lambda paths: ([collision], ()),
        printer=lambda _: None,
    )
    assert not selected.active
    assert "collides" in selected.skipped[0]["reason"]


def test_resume_checks_powerup_before_other_records(tmp_path, monkeypatch):
    from stateframe import runner
    from stateframe.resume import ResumeRefused

    monkeypatch.setattr(
        runner.resume_module,
        "assess",
        lambda *args, **kwargs: {
            "spec": {
                "powerups": {
                    "active": [
                        {
                            "name": "missing",
                            "materialized_dir": str(tmp_path / "missing"),
                            "rendered_dir_sha256": "0" * 64,
                            "source": "x",
                            "tree_sha256": "0" * 64,
                            "mode": "on",
                            "settings": {},
                            "parts": [],
                        }
                    ],
                    "skipped": [],
                }
            }
        },
    )
    with pytest.raises(ResumeRefused, match="POWERUP_CHANGED"):
        runner.resume_run(tmp_path / "run")


def test_cli_management_surface(tmp_path, monkeypatch, capsys):
    from stateframe.cli import build_parser
    from stateframe.powerups_cli import handle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = unit(tmp_path)
    parser = build_parser()
    assert handle(parser.parse_args(["powerups", "install", str(source)])) == 0
    assert handle(parser.parse_args(["powerups", "list"])) == 0
    assert (
        handle(parser.parse_args(["powerups", "enable", "sample", "--mode", "offer"]))
        == 0
    )
    assert handle(parser.parse_args(["powerups", "show", "sample"])) == 0
    shown = capsys.readouterr().out
    assert '"mode": "offer"' in shown
    assert '"mode_source": "user-global"' in shown
    assert handle(parser.parse_args(["powerups", "disable", "sample"])) == 0
    assert (
        powerups.effective_state(
            powerups.parse_manifest(tmp_path / "home/powerups/sample"), "default"
        )["mode"]
        == "off"
    )
    project = tmp_path / "repo"
    unit(project / ".stateframe" / "powerups")
    entries = []
    monkeypatch.setattr(
        powerup_state.hook_host, "_read_trust_store", lambda: entries[:]
    )
    monkeypatch.setattr(
        powerup_state.hook_host,
        "_write_trust_store",
        lambda values: entries.__setitem__(slice(None), values),
    )
    assert handle(parser.parse_args(["powerups", "trust", "--repo", str(project)])) == 0
    assert entries
    assert (
        handle(
            parser.parse_args(["powerups", "trust", "--repo", str(project), "--remove"])
        )
        == 0
    )
    assert not entries
    recorded = unit(tmp_path / "recorded")
    assert (
        handle(
            parser.parse_args(
                [
                    "powerups",
                    "promote",
                    str(tmp_path / "run"),
                    "mut-3",
                    "--name",
                    "promoted",
                ]
            ),
            reader=lambda _run, _id: recorded,
        )
        == 0
    )
    assert handle(parser.parse_args(["powerups", "remove", "promoted"])) == 0


def test_cli_promote_reads_pinned_unit_index(tmp_path, monkeypatch):
    from stateframe.cli import build_parser
    from stateframe.powerups_cli import handle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    run = tmp_path / "run"
    recorded = unit(run / "mutations", "recorded")
    (run / "mutations/index.json").write_text(
        json.dumps(
            {
                "units": {
                    "mut-1": {
                        "directory": "recorded",
                        "sha256": powerups.tree_digest(recorded),
                    }
                }
            }
        )
    )
    args = build_parser().parse_args(
        ["powerups", "promote", str(run), "mut-1", "--name", "promoted"]
    )
    assert handle(args) == 0
    installed = tmp_path / "home/powerups/promoted"
    assert powerups.parse_manifest(installed).name == "promoted"
    assert "promoted from mutation mut-1" in (installed / "install.json").read_text()
    (run / "mutations/index.json").write_text(
        json.dumps(
            {"units": {"mut-2": {"directory": "../../outside", "sha256": "a" * 64}}}
        )
    )
    with pytest.raises(powerups.PowerupError, match="escapes"):
        handle(
            build_parser().parse_args(
                ["powerups", "promote", str(run), "mut-2", "--name", "another"]
            )
        )


def test_cli_list_names_installed_source(tmp_path, monkeypatch, capsys):
    from stateframe.cli import build_parser
    from stateframe.powerups_cli import handle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    installed = powerups.install(unit(tmp_path))
    assert handle(build_parser().parse_args(["powerups", "list"])) == 0
    assert str(installed.path) in capsys.readouterr().out


def test_cli_list_reports_broken_unit_without_hiding_others(tmp_path, monkeypatch, capsys):
    from stateframe.cli import build_parser
    from stateframe.powerups_cli import handle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    broken = powerups.install(unit(tmp_path, "broken"))
    powerups.install(unit(tmp_path, "healthy"))
    (broken.path / "powerup.toml").write_text("broken = true\n")
    assert handle(build_parser().parse_args(["powerups", "list"])) == 0
    rows = capsys.readouterr().out
    assert "broken" in rows and "healthy" in rows
    assert "invalid" in rows


def test_cli_promote_rejects_linked_record_index(tmp_path, monkeypatch):
    from stateframe.cli import build_parser
    from stateframe.powerups_cli import handle

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    run = tmp_path / "run"
    recorded = unit(run / "mutations", "recorded")
    records = run / "mutations/records.json"
    records.write_text(json.dumps({"units": {"mut-1": {
        "directory": "recorded", "sha256": powerups.tree_digest(recorded)}}}))
    (run / "mutations/index.json").symlink_to(records)
    args = build_parser().parse_args(["powerups", "promote", str(run), "mut-1",
                                      "--name", "promoted"])
    with pytest.raises(powerups.PowerupError, match="regular"):
        handle(args)


def test_test_gate_veto_includes_failing_tail(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples/powerups/test-gate/gate.py"
    check = tmp_path / "failing-check"
    check.write_text("#!/bin/sh\necho 'last failure detail' >&2\nexit 1\n")
    check.chmod(0o755)
    outcome = subprocess.run(
        [sys.executable, str(script), "--run", str(check)],
        capture_output=True,
        text=True,
    )
    assert outcome.returncode == 2
    assert "last failure detail" in outcome.stderr
    hooks = json.loads((script.parent / "hooks.json").read_text())["hooks"]
    stop = next(h for h in hooks if h["event"] == "stop")
    assert stop["on_error"] == "block"
    assert stop["timeout"] >= 120
    missing = "/no/such/check-program"
    start = subprocess.run(
        [sys.executable, str(script), "--check-program", missing],
        capture_output=True,
        text=True,
    )
    later = subprocess.run(
        [sys.executable, str(script), "--run", missing],
        capture_output=True,
        text=True,
    )
    assert start.returncode == 0 and "unavailable" in start.stdout
    assert later.returncode != 0 and "not installed" in later.stderr


def test_test_gate_caps_check_output():
    script = Path(__file__).resolve().parents[1] / "examples/powerups/test-gate/gate.py"
    command = f"{shlex.quote(sys.executable)} -c 'import sys; sys.stdout.write(\"x\" * 2000000)'"
    outcome = subprocess.run(
        [sys.executable, str(script), "--run", command],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert outcome.returncode == 2
    assert "output limit" in outcome.stderr


def test_repo_grep_caps_directory_entries(tmp_path, monkeypatch, capsys):
    script = (
        Path(__file__).resolve().parents[1] / "examples/powerups/repo-grep/grep_repo.py"
    )
    spec = importlib.util.spec_from_file_location("powerup_repo_grep", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for index in range(4):
        (tmp_path / f"empty{index}").mkdir()
    monkeypatch.setattr(module, "MAX_ENTRIES", 3)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.sys, "argv", [str(script), "needle", "*.txt"])
    assert module.main() == 0
    assert "search stopped at 3 directory entries" in capsys.readouterr().out


def test_py_format_hook_timeout_covers_formatter_limit():
    path = (
        Path(__file__).resolve().parents[1] / "examples/powerups/py-format/hooks.json"
    )
    hooks = json.loads(path.read_text())["hooks"]
    post_tool = next(h for h in hooks if h["event"] == "post_tool")
    assert post_tool["timeout"] >= 60


@pytest.mark.parametrize("command", ["run", "chat"])
def test_run_and_chat_share_powerup_flags(command):
    from stateframe.cli import build_parser

    args = [
        command,
        "--repo",
        "/tmp/repo",
        "--endpoint",
        "http://127.0.0.1:8080/v1",
        "--model",
        "scripted",
        "--powerup",
        "sample=offer",
        "--powerup-set",
        "sample.limit=12",
        "--no-powerups",
    ]
    if command == "run":
        args += ["--task", "Inspect"]
    parsed = build_parser().parse_args(args)
    assert parsed.powerup == ["sample=offer"]
    assert parsed.powerup_set == ["sample.limit=12"]
    assert parsed.no_powerups is True


def test_runner_prepare_attaches_command_and_pins_spec(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_tool_settings import ToolSettings
    from stateframe.runner import RunOptions, prepare

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_value():\n    assert True\n")
    path = Path(__file__).resolve().parents[1] / "examples/powerups/repo-grep"
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    _, _, spec = prepare(
        RunOptions(
            repo=repo,
            task="Search VALUE",
            endpoint=endpoint,
            run_dir=tmp_path / "run",
            powerup_requests=(str(path),),
            tool_settings=ToolSettings(run_env=False),
        ),
        printer=lambda _: None,
    )
    assert spec["powerups"]["active"][0]["name"] == "repo-grep"
    assert spec["tool_settings"]["tools_files"][0]["path"].endswith(
        "repo-grep/commands.json"
    )
    assert spec["skills"][0]["name"] == "repo-grep"


def test_runner_places_user_skill_tools_before_powerup_commands(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_tool_settings import ToolSettings
    from stateframe.runner import RunOptions, prepare

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_value():\n    assert True\n")
    skill = tmp_path / "user-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: user-skill\ndescription: User skill.\n---\nRead me.\n"
    )
    (skill / "tools.json").write_text(
        json.dumps(
            {
                "schema": "stateframe-tools/1",
                "tools": [
                    {
                        "name": "user_command",
                        "description": "User command.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        "argv": [TRUE],
                        "network": "none",
                        "writes": "none",
                        "replay": "local",
                        "timeout_seconds": 5,
                    }
                ],
            }
        )
    )
    powerup = Path(__file__).resolve().parents[1] / "examples/powerups/repo-grep"
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    _, _, spec = prepare(
        RunOptions(
            repo=repo,
            task="Search VALUE",
            endpoint=endpoint,
            run_dir=tmp_path / "run",
            powerup_requests=(str(powerup),),
            skills=(str(skill),),
            trust_skill_tools=True,
            tool_settings=ToolSettings(run_env=False),
        ),
        printer=lambda _: None,
    )
    paths = [item["path"] for item in spec["tool_settings"]["tools_files"]]
    assert paths[0] == str(skill / "tools.json")
    assert paths[1].endswith("repo-grep/commands.json")


def test_test_gate_refuses_when_check_interpreter_is_missing(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples/powerups/test-gate/gate.py"
    check = tmp_path / "check"
    check.write_text("#!/nonexistent/interpreter\n")
    check.chmod(0o755)
    result = subprocess.run([sys.executable, str(script), "--run", str(check)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "could not start" in result.stderr
