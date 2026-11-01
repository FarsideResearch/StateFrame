import json
import sys
from pathlib import Path

import pytest

import kernel_hooks
from kernel_hooks import (
    PROJECT_HOOKS_PATH,
    HookArgvError,
    HookConfigError,
    check_operands,
    hookset_trust_digest,
    load_hook_files,
    load_project_hook_file,
    merge_hooksets,
)


def configured(tmp_path, hooks, *, network="none"):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": hooks}))
    return load_hook_files([config], shell_network=network, repo=repo), config, repo


def basic(**changes):
    return {
        "name": "guard",
        "event": "pre_tool",
        "argv": [sys.executable, "-c", "pass"],
        **changes,
    }


def test_defaults_and_digest(tmp_path):
    hooks, _, _ = configured(tmp_path, [basic()])
    hook = hooks.definitions[0]
    assert hook.timeout == 30 and hook.network == "none" and hook.tools == ("*",)
    assert hook.writes == "none"
    assert hooks.for_event("pre_tool") == (hook,)
    assert len(hook.definition_digest()) == 64


def test_path_dispatching_argv_zero_is_refused(tmp_path):
    with pytest.raises(HookConfigError, match="use an absolute interpreter path"):
        configured(tmp_path, [basic(argv=["/usr/bin/env", "python3", "-c", "pass"])])


def test_writer_mode_implies_changed_and_rejects_full_network(tmp_path):
    hooks, _, _ = configured(
        tmp_path, [basic(event="post_tool", writes="changed")]
    )
    assert hooks.definitions[0].writes == "changed"
    assert hooks.definitions[0].changed is True

    full_config = tmp_path / "full"
    full_config.mkdir()
    with pytest.raises(HookConfigError, match="writes changed requires network"):
        configured(
            full_config,
            [basic(event="post_tool", writes="changed", network="full")],
            network="full",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "BAD"},
        {"event": "unknown"},
        {"timeout": True},
        {"timeout": 591},
        {"network": "full"},
        {"max_blocks": 1},
        {"on_error": "ignore"},
        {"changed": False},
        {"writes": "sometimes"},
        {"writes": "changed"},
        {"paths": ["*.py"], "event": "stop"},
        {"tools": ["shell"], "event": "stop"},
        {"argv": [sys.executable, "{paths}"]},
        {"argv": [sys.executable, "prefix{paths}"]},
        {"files": ["../escape"]},
        {"files": ["/tmp/escape"]},
        {"unknown": 1},
    ],
)
def test_rejects_invalid_fields(tmp_path, changes):
    with pytest.raises(HookConfigError):
        configured(tmp_path, [basic(**changes)])


def test_duplicate_keys_and_nan(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    for raw in (
        '{"schema":"stateframe-hooks/1","schema":"stateframe-hooks/1","hooks":[]}',
        '{"schema":"stateframe-hooks/1","hooks":[],"bad":NaN}',
    ):
        config.write_text(raw)
        with pytest.raises(HookConfigError):
            load_hook_files([config], shell_network="none", repo=repo)


def test_missing_and_invalid_unicode_config_are_load_errors(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    with pytest.raises(HookConfigError, match="unavailable"):
        load_hook_files([config], shell_network="none", repo=repo)
    config.write_text('{"schema":"stateframe-hooks/1","hooks":[],"extra":"\\ud800"}')
    with pytest.raises(HookConfigError, match="invalid hook JSON"):
        load_hook_files([config], shell_network="none", repo=repo)


def test_duplicate_names_cross_file_and_limits(tmp_path):
    hooks, config, repo = configured(tmp_path, [basic()])
    with pytest.raises(HookConfigError, match="duplicate"):
        load_hook_files([config, config], shell_network="none", repo=repo)
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [basic(name=f"g{i}") for i in range(9)],
            }
        )
    )
    with pytest.raises(HookConfigError, match="exceeds"):
        load_hook_files([config], shell_network="none", repo=repo)


def test_undeclared_operand_at_load_and_invocation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "script.py").write_text("pass")
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [basic(argv=[sys.executable, "script.py"])],
            }
        )
    )
    with pytest.raises(HookConfigError, match="undeclared"):
        load_hook_files([config], shell_network="none", repo=repo)
    (repo / "script.py").unlink()
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    (repo / "script.py").write_text("pass")
    with pytest.raises(HookArgvError, match="undeclared"):
        check_operands(hooks.definitions[0], repo)


def test_operands_longer_than_a_file_name_are_not_files(tmp_path):
    # A component over NAME_MAX (255) makes stat fail with ENAMETOOLONG; Path.is_file()
    # raised it before Python 3.14 (Linux gate on 3.11). Such operands are plain strings.
    long = "x" * 4095
    hooks, _, repo = configured(tmp_path, [basic(argv=[sys.executable, "-c", "pass", long])])
    assert hooks.definitions[0].argv[-1] == long
    check_operands(hooks.definitions[0], repo)
    (tmp_path / "abs").mkdir()
    with pytest.raises(HookConfigError, match="executable not found"):
        configured(tmp_path / "abs", [basic(argv=["/" + long, "-c", "pass"])])


def test_files_reject_symlinks_and_special_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    for source in (tmp_path / "link",):
        source.symlink_to(sys.executable)
        config.write_text(
            json.dumps(
                {"schema": "stateframe-hooks/1", "hooks": [basic(files=[source.name])]}
            )
        )
        with pytest.raises(HookConfigError, match="symlink"):
            load_hook_files([config], shell_network="none", repo=repo)


def test_repo_executable_requires_bundle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    executable = repo / "guard.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            {"schema": "stateframe-hooks/1", "hooks": [basic(argv=[str(executable)])]}
        )
    )
    with pytest.raises(HookConfigError, match="inside repository"):
        load_hook_files([config], shell_network="none", repo=repo)


def test_operand_resolved_against_hook_directory_and_absolute(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script = tmp_path / "operand.py"
    script.write_text("pass")
    config = tmp_path / "hooks.json"
    for operand in ("operand.py", str(script)):
        config.write_text(
            json.dumps(
                {
                    "schema": "stateframe-hooks/1",
                    "hooks": [basic(argv=[sys.executable, operand])],
                }
            )
        )
        with pytest.raises(HookConfigError, match="undeclared"):
            load_hook_files([config], shell_network="none", repo=repo)


def test_file_size_special_and_symlink_ancestor(tmp_path):
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    folder = tmp_path / "actual"
    folder.mkdir()
    (folder / "script.py").write_text("pass")
    (tmp_path / "linkdir").symlink_to(folder, target_is_directory=True)
    huge = tmp_path / "huge.py"
    huge.write_bytes(b"x" * ((1 << 20) + 1))
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    for relative, expected in (
        ("linkdir/script.py", "symlink"),
        ("huge.py", "1 MiB"),
        ("pipe", "regular"),
    ):
        config.write_text(
            json.dumps(
                {"schema": "stateframe-hooks/1", "hooks": [basic(files=[relative])]}
            )
        )
        with pytest.raises(HookConfigError, match=expected):
            load_hook_files([config], shell_network="none", repo=repo)


def test_paths_implicit_changed_and_protected_inputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "hooks.json"
    script = repo / "guard.py"
    script.write_text("pass")
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "guard",
                        "event": "post_tool",
                        "argv": [sys.executable, "guard.py", "{paths}"],
                        "files": ["guard.py"],
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    assert hooks.definitions[0].changed is True
    assert hooks.protected_repo_paths(repo) == frozenset({"hooks.json", "guard.py"})
    assert any(str(root) == "/usr" for root in hooks.read_roots())


def test_project_hook_file_absence_symlinks_and_invalid_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert load_project_hook_file(repo, shell_network="none") is None

    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "hooks.json").write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": []})
    )
    project_path.parent.rmdir()
    project_path.parent.symlink_to(external, target_is_directory=True)
    with pytest.raises(HookConfigError, match="must not be a symlink"):
        load_project_hook_file(repo, shell_network="none")

    project_path.parent.unlink()
    project_path.parent.mkdir()
    project_path.symlink_to(external / "hooks.json")
    with pytest.raises(HookConfigError, match="must not be a symlink"):
        load_project_hook_file(repo, shell_network="none")

    project_path.unlink()
    project_path.mkdir()
    with pytest.raises(HookConfigError, match="regular singly linked file"):
        load_project_hook_file(repo, shell_network="none")


def test_project_hook_file_invalid_json_is_a_load_error(tmp_path):
    repo = tmp_path / "repo"
    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir(parents=True)
    project_path.write_text("{")

    with pytest.raises(HookConfigError, match="invalid hook JSON"):
        load_project_hook_file(repo, shell_network="none")


def test_project_hook_loader_refuses_path_replacement_during_load(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir(parents=True)
    project_path.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": []})
    )
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": []})
    )
    original_loader = kernel_hooks.load_hook_files

    def replace_before_parse(paths, *, shell_network, repo):
        project_path.unlink()
        project_path.symlink_to(external)
        return original_loader(paths, shell_network=shell_network, repo=repo)

    monkeypatch.setattr(kernel_hooks, "load_hook_files", replace_before_parse)
    with pytest.raises(HookConfigError, match="changed while loading"):
        load_project_hook_file(repo, shell_network="none")


def test_project_hook_trust_digest_tracks_file_executable_and_file_list(tmp_path):
    repo = tmp_path / "repo"
    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir(parents=True)
    executable = tmp_path / "external" / "formatter"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    first_file = project_path.parent / "formatter.py"
    second_file = project_path.parent / "alternate.py"
    first_file.write_text("print('first')\n")
    second_file.write_text("print('second')\n")

    def write_config(*, timeout=30, file="formatter.py"):
        project_path.write_text(
            json.dumps(
                {
                    "schema": "stateframe-hooks/1",
                    "hooks": [
                        {
                            "name": "format",
                            "event": "post_tool",
                            "argv": [str(executable), file],
                            "files": [file],
                            "timeout": timeout,
                        }
                    ],
                }
            )
        )

    def digest():
        loaded = load_project_hook_file(repo, shell_network="none")
        assert loaded is not None
        return hookset_trust_digest(loaded)

    write_config()
    baseline = digest()
    write_config(timeout=31)
    changed_config = digest()
    assert changed_config != baseline

    executable.write_text("#!/bin/sh\nexit 1\n")
    changed_executable = digest()
    assert changed_executable != changed_config

    first_file.write_text("print('changed')\n")
    changed_file = digest()
    assert changed_file != changed_executable

    write_config(file="alternate.py")
    changed_file_list = digest()
    assert changed_file_list != changed_file


def test_project_hooks_merge_before_explicit_files_and_rebase_indices(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir()
    project_path.write_text(
        json.dumps(
            {"schema": "stateframe-hooks/1", "hooks": [basic(name="project")]}
        )
    )
    explicit_first = tmp_path / "explicit-first.json"
    explicit_second = tmp_path / "explicit-second.json"
    explicit_first.write_text(
        json.dumps(
            {"schema": "stateframe-hooks/1", "hooks": [basic(name="cli_first")]}
        )
    )
    explicit_second.write_text(
        json.dumps(
            {"schema": "stateframe-hooks/1", "hooks": [basic(name="cli_second")]}
        )
    )
    project = load_project_hook_file(repo, shell_network="none")
    explicit = load_hook_files(
        [explicit_first, explicit_second], shell_network="none", repo=repo
    )

    merged = merge_hooksets(project, explicit)

    assert [hook.name for hook in merged.definitions] == [
        "project",
        "cli_first",
        "cli_second",
    ]
    assert [hook.source_index for hook in merged.definitions] == [0, 1, 2]
    assert [Path(item["path"]).name for item in merged.files] == [
        "hooks.json",
        "explicit-first.json",
        "explicit-second.json",
    ]
    assert merged.digest == hookset_trust_digest(merged)


def test_project_hooks_merge_rejects_duplicate_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    project_path = repo / PROJECT_HOOKS_PATH
    project_path.parent.mkdir()
    project_path.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": [basic(name="same")]})
    )
    explicit_path = tmp_path / "explicit.json"
    explicit_path.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": [basic(name="same")]})
    )

    project = load_project_hook_file(repo, shell_network="none")
    explicit = load_hook_files([explicit_path], shell_network="none", repo=repo)
    with pytest.raises(HookConfigError, match="duplicate hook name"):
        merge_hooksets(project, explicit)
