"""B2: with no home at all, a run treats user-scope powerups as empty; writing commands keep the clear error."""
import pwd
from types import SimpleNamespace

import pytest

from stateframe import powerup_state, powerups
from stateframe.powerup_manifest import PowerupError
from stateframe.powerup_prepare import prepare_run_inputs
from stateframe.skills import SkillError, install, resolve_skills

EXPECTED_NOTICE = (
    "stateframe: no home directory found (HOME and STATEFRAME_HOME unset); "
    "user powerups and skills are skipped for this run\n"
)


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)


def test_user_home_or_none_is_none_without_home(no_home):
    assert powerup_state.user_home_or_none() is None
    assert powerup_state.user_state() == {"schema": powerup_state.SCHEMA, "global": {}, "profiles": {}}


def test_writing_commands_keep_the_clear_error_without_home(no_home):
    with pytest.raises(PowerupError, match="set HOME or STATEFRAME_HOME"):
        powerup_state.update("example", mode="on")


def test_unconfigured_run_has_no_powerups_without_home(no_home, tmp_path, capsys):
    options = SimpleNamespace(no_powerups=False, repo=tmp_path, powerup_requests=(),
                              powerup_sets=(), skills=())
    assert prepare_run_inputs(options, {}) == powerups.Selection()
    assert powerups.resolve_units(tmp_path, "", "default", (), (), False).active == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == EXPECTED_NOTICE


def test_resolve_units_without_home_skips_user_scope_without_a_state_error(no_home, tmp_path):
    selected = powerups.resolve_units(tmp_path, "", "default", (), (), False)
    assert selected.active == []
    assert not [row for row in selected.skipped if row.get("code") == "state_invalid"]


def test_user_scope_is_read_when_home_exists(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    assert powerup_state.user_home_or_none() == home / ".stateframe"


@pytest.mark.parametrize("variable", ["HOME", "STATEFRAME_HOME"])
def test_run_with_configured_home_has_no_notice(no_home, monkeypatch, tmp_path, capsys, variable):
    monkeypatch.setenv(variable, str(tmp_path / "user"))
    options = SimpleNamespace(no_powerups=False, repo=tmp_path, powerup_requests=(),
                              powerup_sets=(), skills=())
    assert prepare_run_inputs(options, {}) == powerups.Selection()
    assert capsys.readouterr().err == ""


def test_no_powerups_without_skill_reads_has_no_notice(no_home, capsys):
    options = SimpleNamespace(no_powerups=True, skills=())
    assert prepare_run_inputs(options, {}) == powerups.Selection()
    assert capsys.readouterr().err == ""


def test_named_user_skill_is_skipped_without_home(no_home, capsys):
    options = SimpleNamespace(no_powerups=True, skills=("example-skill",))
    assert prepare_run_inputs(options, {}) == powerups.Selection()
    assert resolve_skills(options.skills, trusted=False) == ()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == EXPECTED_NOTICE


def test_explicit_skill_path_remains_available_without_home(no_home, tmp_path, capsys):
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Example\n---\nUse the example.\n",
        encoding="utf-8",
    )
    options = SimpleNamespace(no_powerups=True, skills=(str(skill_dir),))
    assert prepare_run_inputs(options, {}) == powerups.Selection()
    assert [skill.name for skill in resolve_skills(options.skills, trusted=False)] == ["example-skill"]
    with pytest.raises(SkillError, match="set STATEFRAME_HOME or configure a home directory"):
        install(skill_dir)
    assert capsys.readouterr().err == ""
