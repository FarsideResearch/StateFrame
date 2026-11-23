import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_project_python_token_requires_retrust_for_different_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _hook_file(repo)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["hooks"][0]["argv"] = ["{python}", "context.py"]
    config.write_text(json.dumps(document), encoding="utf-8")
    first = tmp_path / "first-python"
    second = tmp_path / "second-python"
    first.symlink_to(Path(os.sys.executable))
    second.symlink_to(Path(os.sys.executable))

    monkeypatch.setattr(os.sys, "executable", str(first))
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    assert hook_host.resolve_project_hooks(repo, shell_network="none").hooks is not None

    monkeypatch.setattr(os.sys, "executable", str(second))
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


def test_trust_review_write_failure_preserves_existing_trust(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _hook_file(repo)
    assert cli.main(["hooks", "trust", "--repo", str(repo)]) == 0
    store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    previous = store.read_bytes()
    (repo / ".stateframe" / "context.py").write_text("print('changed')\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "hooks", "trust", "--repo", str(repo)],
            cwd=Path(__file__).resolve().parents[1], env=environment, stdout=write_fd,
            stderr=subprocess.PIPE, text=True, pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)

    assert result.returncode == 2
    assert store.read_bytes() == previous
    assert not list(store.parent.glob(".trusted-hooks-*.tmp"))
    assert result.stderr == (
        "stateframe: error: cannot write project hook review; trust was not saved\n"
    )


def test_trust_review_flush_precedes_store_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _hook_file(repo)
    events = []

    class Output:
        def flush(self):
            events.append("flush")

    monkeypatch.setattr(sys, "stdout", Output())
    original_write = hook_host._write_trust_store

    def record_commit(entries, *, before_replace=None):
        assert events[-1] == "flush"
        assert all(isinstance(item, tuple) for item in events[:-1])
        assert ("review", f"hookset sha256: {entries[0]['digest']}") in events
        original_write(entries, before_replace=before_replace)

    monkeypatch.setattr(hook_host, "_write_trust_store", record_commit)
    hook_host.trust_project_hooks(repo, printer=lambda line: events.append(("review", line)))

    assert events[-1][0] == "review"
    assert "trusted project hooks" in events[-1][1]


def test_trust_rejects_project_hook_change_after_review(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _hook_file(repo)
    assert cli.main(["hooks", "trust", "--repo", str(repo)]) == 0
    store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    previous = store.read_bytes()
    resource = repo / ".stateframe" / "context.py"
    resource.write_text("print('reviewed')\n", encoding="utf-8")
    original = hook_host.trust_project_hooks

    def change_during_review(target, *, remove=False, printer=print):
        def review(line):
            printer(line)
            if line.startswith("  file "):
                resource.write_text("print('changed after review')\n", encoding="utf-8")

        return original(target, remove=remove, printer=review)

    monkeypatch.setattr(hook_host, "trust_project_hooks", change_during_review)
    result = cli.main(["hooks", "trust", "--repo", str(repo)])

    assert result == 2
    assert store.read_bytes() == previous
    assert not list(store.parent.glob(".trusted-hooks-*.tmp"))
    assert capsys.readouterr().err == (
        "stateframe: error: project hooks changed after review; trust not saved\n"
    )


def test_interrupted_trust_commit_keeps_previous_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _hook_file(repo)
    assert cli.main(["hooks", "trust", "--repo", str(repo)]) == 0
    store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    previous = store.read_bytes()
    resource = repo / ".stateframe" / "context.py"
    resource.write_text("print('changed')\n", encoding="utf-8")

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(hook_host.os, "replace", interrupt)
    result = cli.main(["hooks", "trust", "--repo", str(repo)])

    assert result == 130
    assert store.read_bytes() == previous
    assert not list(store.parent.glob(".trusted-hooks-*.tmp"))
    assert capsys.readouterr().err == "stateframe: interrupted\n"


def test_concurrent_project_trust_keeps_all_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repos = []
    for index in range(6):
        repo = tmp_path / f"repo-{index}"
        repo.mkdir()
        _hook_file(repo)
        repos.append(repo)
    start = threading.Barrier(len(repos) + 1)

    def trust(repo):
        start.wait()
        hook_host.trust_project_hooks(repo, printer=lambda *_: None)

    with ThreadPoolExecutor(max_workers=len(repos)) as workers:
        futures = [workers.submit(trust, repo) for repo in repos]
        start.wait()
        for future in futures:
            future.result()

    store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    entries = json.loads(store.read_text(encoding="utf-8"))["entries"]
    assert {entry["repo"] for entry in entries} == {str(repo.resolve()) for repo in repos}


def test_every_trust_store_writer_waits_for_the_store_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from stateframe import hook_host, powerup_state

    events = []

    @contextmanager
    def lock():
        events.append("lock")
        yield
        events.append("unlock")

    monkeypatch.setattr(hook_host, "_trust_store_lock", lock)
    monkeypatch.setattr(hook_host, "_read_trust_store", lambda: events.append("read") or [])
    monkeypatch.setattr(hook_host, "_write_trust_store", lambda *_a, **_k: events.append("write"))
    repo = tmp_path / "repo"
    repo.mkdir()
    powerup_state.trust(repo)
    assert events == ["lock", "read", "write", "unlock"]
    events.clear()
    hook_host.trust_project_hooks(repo, remove=True, printer=lambda _line: None)
    assert events == ["lock", "read", "unlock"]
