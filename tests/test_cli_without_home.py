"""Every CLI command builds the full parser first, so it must not need a home directory (11-04 HOME probe)."""
import pwd

import pytest

from stateframe import cli


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)


def test_build_parser_works_without_home(no_home):
    parser = cli.build_parser()
    args = parser.parse_args(["config", "init"])
    assert args.path is None


def test_version_works_without_home(no_home, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip()


def test_config_init_without_home_and_without_path_is_a_clear_error(no_home, capsys):
    assert cli.main(["config", "init"]) == 2
    assert "set HOME or pass --path" in capsys.readouterr().err


def test_config_init_with_explicit_path_works_without_home(no_home, tmp_path):
    target = tmp_path / "routes.toml"
    assert cli.main(["config", "init", "--path", str(target)]) == 0
    assert target.is_file()


def test_config_init_default_with_home_is_unchanged(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    assert cli.main(["config", "init"]) == 0
    assert (home / ".stateframe" / "routes.toml").is_file()
