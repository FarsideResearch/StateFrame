import json
import os
from pathlib import Path

import pytest

from stateframe import cli, hook_host, profiles
from stateframe.endpoint import Endpoint


def _hook_file(repo: Path) -> Path:
    directory = repo / ".stateframe"
    directory.mkdir(parents=True)
    source = directory / "context.py"
    source.write_text("print('ready')\n", encoding="utf-8")
    config = directory / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{
            "name": "context",
            "event": "session_start",
            "argv": [os.fspath(Path(os.sys.executable)), "context.py"],
            "files": ["context.py"],
        }],
    }), encoding="utf-8")
    return config


def test_run_and_chat_accept_repeatable_hook_files_but_resume_does_not():
    parser = cli.build_parser()
    common = ["--repo", ".", "--endpoint", "http://127.0.0.1:8080/v1", "--model", "test"]
    run_args = parser.parse_args(["run", *common, "--task", "work", "--hooks", "one.json",
                                  "--hooks", "two.json", "--no-project-hooks"])
    chat_args = parser.parse_args(["chat", *common, "--hooks", "one.json", "--hooks", "two.json",
                                   "--no-project-hooks"])
    assert run_args.hooks == ["one.json", "two.json"]
    assert chat_args.hooks == ["one.json", "two.json"]
    assert run_args.no_project_hooks and chat_args.no_project_hooks
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["resume", "run", "--hooks", "one.json"])
    assert error.value.code == 2


def test_hook_and_tool_paths_are_resolved_from_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hook_path = tmp_path / "hooks.json"
    tool_path = tmp_path / "tools.json"
    hook_path.write_text("{}", encoding="utf-8")
    tool_path.write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args([
        "run", "--repo", ".", "--endpoint", "http://127.0.0.1:8080/v1", "--model", "test",
        "--task", "work", "--hooks", "hooks.json", "--tools", "tools.json",
    ])

    options = cli._run_options(args, Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None))

    assert options.hooks_files == (hook_path.resolve(),)
    assert options.tool_settings.tools_files[0]["path"] == str(tool_path.resolve())


def test_trust_command_writes_private_atomic_store_and_remove(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _hook_file(repo)
    replaces = []
    original_replace = os.replace

    def record_replace(*args, **kwargs):
        replaces.append((args, kwargs))
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(hook_host.os, "replace", record_replace)
    assert cli.main(["hooks", "trust", "--repo", str(repo)]) == 0
    store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    document = json.loads(store.read_text(encoding="utf-8"))
    assert document["schema"] == hook_host.TRUST_SCHEMA
    assert document["entries"] == [{
        "repo": str(repo.resolve()), "file": str(config.resolve()),
        "digest": document["entries"][0]["digest"],
    }]
    assert store.stat().st_mode & 0o777 == 0o600
    assert len(replaces) == 1
    output = capsys.readouterr().out
    assert "context" in output and document["entries"][0]["digest"] in output

    assert cli.main(["hooks", "trust", "--repo", str(repo), "--remove"]) == 0
    assert json.loads(store.read_text(encoding="utf-8"))["entries"] == []


def test_project_hooks_require_trust_and_exact_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _hook_file(repo)

    untrusted = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert untrusted.hooks is None
    assert "not trusted" in untrusted.message

    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    selected = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert selected.hooks is not None
    assert selected.digest

    config.write_text(config.read_text(encoding="utf-8").replace("context.py", "changed.py"),
                      encoding="utf-8")
    changed = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert changed.hooks is None
    assert "not trusted" in changed.message

    skipped = hook_host.resolve_project_hooks(repo, shell_network="none", no_project_hooks=True)
    assert skipped.hooks is None and skipped.message is None


@pytest.mark.parametrize("shape", ["file", "symlink_file", "symlink_directory", "unreadable"])
def test_discovery_ignores_directory_without_project_hook_candidate(tmp_path, monkeypatch, shape):
    repo = tmp_path / "repo"
    repo.mkdir()
    project = repo / ".stateframe"
    if shape == "file":
        project.write_text("ordinary repository data")
    elif shape == "symlink_file":
        target = tmp_path / "target"
        target.write_text("ordinary repository data")
        project.symlink_to(target)
    elif shape == "symlink_directory":
        target = tmp_path / "target"
        target.mkdir()
        (target / "hooks.json").write_text("{}")
        project.symlink_to(target, target_is_directory=True)
    else:
        project.mkdir()
        original = Path.lstat

        def unavailable(path):
            if path == project / "hooks.json":
                raise PermissionError("directory is unreadable")
            return original(path)

        monkeypatch.setattr(Path, "lstat", unavailable)
    selected = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert selected.hooks is None and selected.message is None
    with pytest.raises(hook_host.HookConfigError):
        hook_host.trust_project_hooks(repo, printer=lambda *_: None)


def test_discovery_refuses_actual_invalid_project_hook_file(tmp_path):
    repo = tmp_path / "repo"
    project = repo / ".stateframe"
    project.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("{}")
    (project / "hooks.json").symlink_to(target)
    with pytest.raises(hook_host.HookConfigError, match="regular singly linked file"):
        hook_host.resolve_project_hooks(repo, shell_network="none")


@pytest.mark.parametrize("change", ["definition", "resource", "executable"])
def test_project_trust_digest_covers_definition_resources_and_executable(tmp_path, monkeypatch, change):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _hook_file(repo)
    executable = None
    if change == "executable":
        executable = tmp_path / "hook-command"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        config.write_text(json.dumps({
            "schema": "stateframe-hooks/1",
            "hooks": [{"name": "context", "event": "session_start", "argv": [str(executable)]}],
        }), encoding="utf-8")
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    assert hook_host.resolve_project_hooks(repo, shell_network="none").hooks is not None

    if change == "definition":
        config.write_text(config.read_text(encoding="utf-8").replace('"name": "context"', '"name": "other"'),
                          encoding="utf-8")
    elif change == "resource":
        (repo / ".stateframe" / "context.py").write_text("print('changed')\n", encoding="utf-8")
    else:
        assert executable is not None
        executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        executable.chmod(0o755)

    changed = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert changed.hooks is None
    assert "not trusted" in changed.message


def test_profile_rejects_hook_switches(tmp_path):
    profile = tmp_path / "bad.json"
    profile.write_text(json.dumps({
        "schema": "stateframe-product-profile/1",
        "switches": {"STATEFRAME_HOOKS": "off"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="STATEFRAME_HOOKS"):
        profiles.load(str(profile))
