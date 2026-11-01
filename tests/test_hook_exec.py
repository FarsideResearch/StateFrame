import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import kernel_hook_exec
from kernel_effects import ContentStore, WorkspaceEffectPolicy, manifest_digest, scan
from kernel_exec_capability import _macos_developer_roots, symlink_hops
from kernel_exec_session import CleanupReport, ExecResult
from kernel_hook_exec import (
    HookContainmentError,
    HookReplayRefused,
    HookRunner,
    HookTape,
)
from kernel_hooks import build_payload, create_bundle, event_id, load_hook_files


def setup_runner(tmp_path, *, event="pre_tool", on_error="block", replay=lambda: False):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "guard",
                        "event": event,
                        "argv": [sys.executable, "-c", "pass"],
                        **(
                            {"on_error": on_error}
                            if event in ("pre_tool", "prompt_submit", "stop")
                            else {}
                        ),
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / "workspace"
    workspace.mkdir()
    (workspace / "code.py").write_text("original")
    bundle = run / "hook-bundle"
    create_bundle(hooks, bundle)
    return HookRunner(
        run,
        hooks,
        bundle_dir=bundle,
        workspace=workspace,
        shell_network="none",
        run_seed="seed",
        run_id="run",
        version="0.2.1",
        replay=replay,
    ), workspace


def envelope(*, code=0, stdout=b"", stderr=b"", cleanup=True):
    return {
        "schema": "stateframe-hook-envelope/1",
        "exit_code": code,
        "signal": None,
        "timed_out": False,
        "launch_error": None,
        "stdout_b64": base64.b64encode(stdout).decode(),
        "stderr_b64": base64.b64encode(stderr).decode(),
        "stdout_total": len(stdout),
        "stderr_total": len(stderr),
        "cleanup_ok": cleanup,
    }


def fake_session(monkeypatch, *, output=None, mutation=None, cleanup=True):
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def configure(self, cap):
            assert cap.sandboxed and not cap.workspace_writable
            assert len(cap.write_roots) == 1

        def run(self, argv, cap, *, stdin, cwd):
            calls.append((argv, cap, stdin, cwd))
            if mutation:
                mutation(cwd)
            content = json.dumps(output or envelope()).encode()
            return ExecResult(
                0,
                False,
                content,
                False,
                len(content),
                0.1,
                CleanupReport(cleanup, 0, (), "failure" if not cleanup else ""),
            )

        def close(self):
            pass

    monkeypatch.setattr(kernel_hook_exec, "ExecSession", Session)
    return calls


def setup_writer_runner(
    tmp_path,
    scripts,
    initial_files,
    *,
    replay=lambda: False,
    protected_paths=frozenset(),
    edit_scope=None,
    paths=None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    definitions = []
    for name, filename, source in scripts:
        (tmp_path / filename).write_text(source)
        definition = {
            "name": name,
            "event": "post_tool",
            "argv": [sys.executable, filename, "{paths}"],
            "files": [filename],
            "writes": "changed",
        }
        if paths is not None:
            definition["paths"] = paths
        definitions.append(definition)
    config.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": definitions})
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / "workspace"
    workspace.mkdir()
    for relative, value in initial_files.items():
        content, mode = value if isinstance(value, tuple) else (value, 0o644)
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        target.chmod(mode)
    bundle = run / "hook-bundle"
    create_bundle(hooks, bundle)
    policy = WorkspaceEffectPolicy(
        lambda _: False,
        edit_scope if edit_scope is not None else lambda _: True,
        lambda _: True,
    )
    runner = HookRunner(
        run,
        hooks,
        bundle_dir=bundle,
        workspace=workspace,
        shell_network="none",
        run_seed="seed",
        run_id="run",
        version="0.2.1",
        replay=replay,
        effect_policy=policy,
        protected_paths=frozenset(protected_paths),
    )
    return runner, workspace


def real_driver_session(monkeypatch):
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.capability = None

        def configure(self, cap):
            assert cap.sandboxed and cap.workspace_writable
            assert len(cap.write_roots) == 1
            self.capability = cap

        def run(self, argv, cap, *, stdin, cwd):
            assert self.capability is not None
            assert cap.workspace_writable
            calls.append((argv, cap, stdin, cwd))
            completed = subprocess.run(
                argv,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                check=False,
            )
            result = json.loads(completed.stdout)
            return ExecResult(
                completed.returncode,
                False,
                completed.stdout,
                False,
                len(completed.stdout),
                0.01,
                CleanupReport(
                    result["cleanup_ok"] and completed.returncode == 0,
                    0,
                    (),
                    "" if result["cleanup_ok"] and completed.returncode == 0 else "failure",
                ),
            )

        def close(self):
            pass

    monkeypatch.setattr(kernel_hook_exec, "ExecSession", Session)
    return calls


def run_writer_event(runner, changed_paths, *, boundary="post_tool:writer"):
    return runner.run_event(
        "post_tool",
        boundary=boundary,
        data={"tool": "shell", "changed_paths": changed_paths},
        match={"tool": "shell", "changed_paths": changed_paths},
    )


def test_tape_pending_and_corruption(tmp_path):
    tape = HookTape(tmp_path)
    tape.intent("id", {"input": "digest"})
    assert tape.pending() == ("id",)
    assert HookTape(tmp_path).pending() == ("id",)
    tape.complete("id", {"decision": "continue"})
    assert HookTape(tmp_path).lookup("id")["completed"] is True
    with pytest.raises(ValueError, match="duplicate"):
        tape.intent("id", {})
    tape.path.write_text(
        '{"kind":"intent","record_id":"x","identity":{},"identity":{}}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        HookTape(tmp_path)


def test_completed_replay_served_without_execution(tmp_path, monkeypatch):
    active = [False]
    runner, _ = setup_runner(tmp_path, replay=lambda: active[0])
    calls = fake_session(monkeypatch, output=envelope(code=2, stderr=b"blocked"))
    first = runner.run_event(
        "pre_tool",
        boundary="pre_tool:c1",
        data={"tool": "shell"},
        match={"tool": "shell"},
    )
    assert first.decision == "deny" and first.deny_code == "HOOK_DENIED"
    assert (
        runner.records_dir / "out" / f"{first.runs[0].event_id}.err"
    ).read_bytes() == b"blocked"
    assert (
        runner.records_dir / "out" / f"{first.runs[0].event_id}.err"
    ).stat().st_mode & 0o777 == 0o600
    record = json.loads(
        (runner.records_dir / "records.jsonl").read_text().splitlines()[-1]
    )
    assert record["cleanup_ok"] is True
    assert len(record["stream_digests"]["err"]) == 64
    assert len(calls) == 1
    active[0] = True
    second = runner.run_event(
        "pre_tool",
        boundary="pre_tool:c1",
        data={"tool": "shell"},
        match={"tool": "shell"},
    )
    assert second.runs[0].replayed and len(calls) == 1
    with pytest.raises(HookReplayRefused, match="changed"):
        runner.run_event(
            "pre_tool",
            boundary="pre_tool:c1",
            data={"tool": "different"},
            match={"tool": "shell"},
        )
    with pytest.raises(HookReplayRefused, match="no completed"):
        runner.run_event(
            "pre_tool",
            boundary="pre_tool:c2",
            data={"tool": "shell"},
            match={"tool": "shell"},
        )


def test_on_error_continue_and_mutation_is_terminal(tmp_path, monkeypatch):
    runner, workspace = setup_runner(tmp_path, on_error="continue")
    fake_session(monkeypatch, output=envelope(code=1))
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c1", data={}, match={"tool": "shell"}
    )
    assert result.decision == "continue" and result.runs[0].outcome.error_code == "exit"


@pytest.mark.parametrize("event", ["pre_tool", "post_tool"])
def test_injected_workspace_mutation_restored_and_terminal(
    tmp_path, monkeypatch, event
):
    runner, workspace = setup_runner(tmp_path, event=event, on_error="continue")
    fake_session(
        monkeypatch, mutation=lambda cwd: (cwd / "code.py").write_text("changed")
    )
    with pytest.raises(HookContainmentError, match="changed workspace"):
        runner.run_event(
            event, boundary=f"{event}:c1", data={}, match={"tool": "shell"}
        )
    assert (workspace / "code.py").read_text() == "original"
    assert runner.tape.pending()


def test_supervisor_cleanup_failure_is_terminal(tmp_path, monkeypatch):
    runner, _ = setup_runner(tmp_path, event="post_tool")
    fake_session(monkeypatch, cleanup=False)
    with pytest.raises(HookContainmentError, match="cleanup"):
        runner.run_event(
            "post_tool", boundary="post_tool:c1", data={}, match={"tool": "shell"}
        )


def test_writer_in_place_formatter_records_and_persists_authorized_bytes(
    tmp_path, monkeypatch
):
    script = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    path.write_text(path.read_text().upper())
"""
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        {"document.txt": ("draft\n", 0o640)},
    )
    calls = real_driver_session(monkeypatch)

    result = run_writer_event(runner, ["document.txt"])

    assert result.decision == "continue"
    assert (workspace / "document.txt").read_text() == "DRAFT\n"
    assert len(calls) == 1 and calls[0][1].workspace_writable
    write = result.writes[0]
    assert write.hook == "format"
    assert write.restored == ()
    assert len(write.authorized) == 1
    effect = write.authorized[0]
    assert effect.path == "document.txt"
    assert effect.pre_sha256 == hashlib.sha256(b"draft\n").hexdigest()
    assert effect.post_sha256 == hashlib.sha256(b"DRAFT\n").hexdigest()
    assert ContentStore(runner.run_dir / "hook-effects").get(
        effect.post_sha256
    ) == b"DRAFT\n"


def test_writer_temp_then_rename_formatter_preserves_mode_and_is_authorized(
    tmp_path, monkeypatch
):
    script = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
    target = Path(raw)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(target.read_text().upper())
    temporary.chmod(target.stat().st_mode & 0o777)
    temporary.replace(target)
"""
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        {"document.txt": ("draft", 0o640)},
    )
    real_driver_session(monkeypatch)

    result = run_writer_event(runner, ["document.txt"])

    target = workspace / "document.txt"
    assert target.read_text() == "DRAFT"
    assert target.stat().st_mode & 0o777 == 0o640
    assert [effect.path for effect in result.writes[0].authorized] == [
        "document.txt"
    ]
    assert result.writes[0].restored == ()


def test_writer_restores_and_reports_unauthorized_effects_and_mode_changes(
    tmp_path, monkeypatch
):
    script = """from pathlib import Path
allowed = Path('allowed.txt')
allowed.write_text('formatted')
Path('other.txt').write_text('outside change')
Path('new.txt').write_text('new file')
Path('gone.txt').unlink()
cache = Path('cache')
cache.mkdir()
(cache / 'item').write_text('cache data')
mode = Path('mode.txt')
mode.write_text('changed bytes')
mode.chmod(0o600)
Path('exec.txt').chmod(0o755)
Path('protected.txt').write_text('changed protected')
Path('.stateframe/hooks.json').write_text('changed hook file')
Path('base_denied.txt').write_text('base policy denied')
"""
    initial = {
        "allowed.txt": "original",
        "other.txt": "other original",
        "gone.txt": "delete me",
        "mode.txt": ("original mode bytes", 0o640),
        "exec.txt": "not executable",
        "protected.txt": "protected original",
        ".stateframe/hooks.json": "trusted hook file",
        "base_denied.txt": "outside run policy",
    }
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        initial,
        protected_paths={"protected.txt", ".stateframe/hooks.json"},
        edit_scope=lambda path: path != "base_denied.txt",
    )
    real_driver_session(monkeypatch)
    changed_paths = [
        "allowed.txt",
        "mode.txt",
        "exec.txt",
        "protected.txt",
        ".stateframe/hooks.json",
        "base_denied.txt",
    ]

    result = run_writer_event(runner, changed_paths)

    assert result.decision == "continue"
    assert (workspace / "allowed.txt").read_text() == "formatted"
    for relative, value in initial.items():
        if relative == "allowed.txt":
            continue
        content, mode = value if isinstance(value, tuple) else (value, 0o644)
        target = workspace / relative
        assert target.read_text() == content
        assert target.stat().st_mode & 0o777 == mode
    assert not (workspace / "new.txt").exists()
    assert not (workspace / "cache").exists()
    restored = set(result.writes[0].restored)
    assert {
        "other.txt",
        "new.txt",
        "gone.txt",
        "cache",
        "cache/item",
        "mode.txt",
        "exec.txt",
        "protected.txt",
        ".stateframe/hooks.json",
        "base_denied.txt",
    } <= restored
    assert [effect.path for effect in result.writes[0].authorized] == [
        "allowed.txt"
    ]


def test_writer_skips_when_filtered_write_set_is_empty(tmp_path, monkeypatch):
    runner, _ = setup_writer_runner(
        tmp_path,
        [("format", "format.py", "raise SystemExit('must not launch')\n")],
        {"directory/child.txt": "content"},
    )
    calls = real_driver_session(monkeypatch)

    result = run_writer_event(runner, ["directory"])

    assert result.decision == "continue"
    assert result.runs[0].skipped == "empty_write_set"
    assert result.writes == ()
    assert calls == []


def test_writers_run_in_order_and_each_reconcile_before_next(tmp_path, monkeypatch):
    uppercase = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    path.write_text(path.read_text().upper())
"""
    append = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    path.write_text(path.read_text() + '!')
"""
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("uppercase", "upper.py", uppercase), ("append", "append.py", append)],
        {"document.txt": ("alpha", 0o640)},
    )
    calls = real_driver_session(monkeypatch)

    result = run_writer_event(runner, ["document.txt"])

    assert (workspace / "document.txt").read_text() == "ALPHA!"
    assert len(calls) == 2
    assert [write.hook for write in result.writes] == ["uppercase", "append"]
    first, second = result.writes
    assert first.authorized[0].post_sha256 == second.authorized[0].pre_sha256


def test_writer_replay_reapplies_recorded_bytes_without_running_formatter(
    tmp_path, monkeypatch
):
    counter = tmp_path / "counter.txt"
    script = f"""from pathlib import Path
import sys
counter = Path({str(counter)!r})
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
for raw in sys.argv[1:]:
    path = Path(raw)
    path.write_text(path.read_text().upper())
"""
    replaying = [False]
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        {"document.txt": ("draft", 0o640)},
        replay=lambda: replaying[0],
    )
    calls = real_driver_session(monkeypatch)

    first = run_writer_event(runner, ["document.txt"])
    assert counter.read_text() == "1"
    target = workspace / "document.txt"
    target.write_text("draft")
    target.chmod(0o640)
    replaying[0] = True

    second = run_writer_event(runner, ["document.txt"])

    assert second.runs[0].replayed is True
    assert target.read_text() == "DRAFT"
    assert target.stat().st_mode & 0o777 == 0o640
    assert counter.read_text() == "1"
    assert len(calls) == 1
    assert second.writes[0].authorized == first.writes[0].authorized


def test_writer_replay_refuses_pre_digest_mismatch(tmp_path, monkeypatch):
    runner, workspace = setup_writer_runner(
        tmp_path,
        [
            (
                "format",
                "format.py",
                "from pathlib import Path\nimport sys\nfor raw in sys.argv[1:]:\n Path(raw).write_text('formatted')\n",
            )
        ],
        {"document.txt": "draft"},
    )
    calls = real_driver_session(monkeypatch)

    first = run_writer_event(runner, ["document.txt"])
    tape_record = runner.tape.lookup(first.runs[0].event_id)
    tape_record["result"]["writer"]["pre_digests"]["document.txt"] = "0" * 64
    target = workspace / "document.txt"
    target.write_text("draft")
    before = scan(workspace, exclude=lambda _: False)

    with pytest.raises(HookReplayRefused, match="pre-digest changed"):
        runner._replay_writer(
            runner.hooks.definitions[0],
            tape_record["result"]["writer"],
            ("document.txt",),
            before,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["second_file", "after_replace_before_dir_fsync"])
def test_writer_replay_restores_all_files_after_apply_failure(tmp_path, monkeypatch, failure):
    script = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
 path = Path(raw)
 path.write_text(path.read_text().upper())
"""
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        {"first.txt": ("first", 0o640), "second.txt": ("second", 0o600)},
    )
    real_driver_session(monkeypatch)
    first = run_writer_event(runner, ["first.txt", "second.txt"])
    record = runner.tape.lookup(first.runs[0].event_id)["result"]["writer"]
    (workspace / "first.txt").write_text("first")
    (workspace / "second.txt").write_text("second")
    before = scan(workspace, exclude=lambda _: False)
    assert manifest_digest(before) == record["pre_workspace_digest"]
    if failure == "second_file":
        original_write = kernel_hook_exec._write_replay_file
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise HookReplayRefused("injected second-file failure")
            return original_write(*args, **kwargs)

        monkeypatch.setattr(kernel_hook_exec, "_write_replay_file", fail_second)
    else:
        original_fsync = os.fsync
        failed = False

        def fail_directory_once(fd):
            nonlocal failed
            if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
                failed = True
                raise OSError("injected directory fsync failure")
            return original_fsync(fd)

        monkeypatch.setattr(kernel_hook_exec.os, "fsync", fail_directory_once)
    with pytest.raises(HookReplayRefused, match="could not apply recorded bytes"):
        runner._replay_writer(
            runner.hooks.definitions[0], record,
            ("first.txt", "second.txt"), before,
        )
    assert (workspace / "first.txt").read_text() == "first"
    assert (workspace / "second.txt").read_text() == "second"
    assert manifest_digest(scan(workspace, exclude=lambda _: False)) == record["pre_workspace_digest"]


def test_writer_replay_reports_containment_when_rollback_fails(tmp_path, monkeypatch):
    script = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
 path = Path(raw)
 path.write_text(path.read_text().upper())
"""
    runner, workspace = setup_writer_runner(
        tmp_path, [("format", "format.py", script)],
        {"first.txt": "first", "second.txt": "second"},
    )
    real_driver_session(monkeypatch)
    first = run_writer_event(runner, ["first.txt", "second.txt"])
    record = runner.tape.lookup(first.runs[0].event_id)["result"]["writer"]
    (workspace / "first.txt").write_text("first")
    (workspace / "second.txt").write_text("second")
    before = scan(workspace, exclude=lambda _: False)
    original_write = kernel_hook_exec._write_replay_file
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise HookReplayRefused("injected persistent write failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(kernel_hook_exec, "_write_replay_file", fail_after_first)
    with pytest.raises(HookContainmentError, match="writer replay restoration failed"):
        runner._replay_writer(
            runner.hooks.definitions[0], record,
            ("first.txt", "second.txt"), before,
        )


def test_delivery_projection_is_private_and_links_decisions(tmp_path, monkeypatch):
    runner, _ = setup_runner(tmp_path)
    fake_session(monkeypatch, output=envelope(code=2, stderr=b"blocked"))
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:delivery", data={}, match={"tool": "shell"}
    )
    ident = result.runs[0].event_id
    runner.record_delivery([ident], request_id="request-7", admitted_bytes=12,
                           omitted_bytes=4, dropped=False)
    runner.record_delivery([ident], request_id="request-8", admitted_bytes=0,
                           omitted_bytes=16, dropped=True)
    path = runner.records_dir / "delivery.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"event_ids": [ident], "request_id": "request-7", "admitted_bytes": 12,
         "omitted_bytes": 4, "dropped": False},
        {"event_ids": [ident], "request_id": "request-8", "admitted_bytes": 0,
         "omitted_bytes": 16, "dropped": True},
    ]
    invocation = json.loads((runner.records_dir / "records.jsonl").read_text().splitlines()[-1])
    tape = runner.tape.lookup(ident)["result"]
    assert invocation["decision"] == tape["outcome"]["decision"] == "deny"


@pytest.mark.host_sandbox
@pytest.mark.parametrize("binary,argument", [
    ("/usr/bin/make", "-v"), ("/usr/bin/git", "--version"),
])
def test_macos_developer_commands_run_as_hooks_in_real_sandbox(tmp_path, binary, argument):
    if not Path(binary).is_file():
        pytest.skip(f"{binary} is absent")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "developer", "event": "pre_tool", "argv": [binary, argument]}
    ]}))
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / "workspace"
    workspace.mkdir()
    bundle = run / "hook-bundle"
    create_bundle(hooks, bundle)
    runner = HookRunner(run, hooks, bundle_dir=bundle, workspace=workspace,
                        shell_network="none", run_seed="seed", run_id="run",
                        version="0.2.1")
    try:
        result = runner.run_event("pre_tool", boundary=f"pre_tool:{argument}",
                                  data={}, match={"tool": "shell"})
        assert result.runs[0].outcome.exit_code == 0
        assert result.runs[0].outcome.decision == "continue"
    finally:
        runner.close()


@pytest.mark.host_sandbox
def test_sandbox_writer_formats_only_the_changed_file(tmp_path):
    script = """from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    path.write_text(path.read_text().upper())
"""
    runner, workspace = setup_writer_runner(
        tmp_path,
        [("format", "format.py", script)],
        {"document.txt": ("draft", 0o640), "other.txt": "untouched"},
    )

    try:
        result = run_writer_event(runner, ["document.txt"])
        assert result.decision == "continue"
        assert (workspace / "document.txt").read_text() == "DRAFT"
        assert (workspace / "other.txt").read_text() == "untouched"
        assert [effect.path for effect in result.writes[0].authorized] == [
            "document.txt"
        ]
    finally:
        runner.close()


@pytest.mark.host_sandbox
def test_real_sandbox_runner_and_read_only_workspace(tmp_path):
    runner, workspace = setup_runner(tmp_path)
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c1", data={}, match={"tool": "shell"}
    )
    assert result.decision == "continue"
    assert (workspace / "code.py").read_text() == "original"
    runner.close()


def test_labeled_capability_identity_ignores_scratch_path(tmp_path):
    runner, _ = setup_runner(tmp_path)
    hook = runner.hooks.definitions[0]
    first = runner.scratch_root / "first"
    second = runner.scratch_root / "second"
    first.mkdir()
    second.mkdir()
    from kernel_hook_exec import _digest, _labeled

    a = runner._capability(hook, first)
    b = runner._capability(hook, second)

    def identity(cap, scratch):
        labels = {
            str(runner.workspace): "<workspace>",
            str(runner.run_dir): "<run>",
            str(runner.bundle_dir): "<bundle>",
            str(scratch): "<scratch>",
        }
        return _digest(_labeled(cap.as_dict(), labels))

    assert a.digest() != b.digest()
    assert identity(a, first) == identity(b, second)


def test_sandbox_unavailable_is_ordinary_gate_error(tmp_path, monkeypatch):
    runner, _ = setup_runner(tmp_path)
    from kernel_exec_session import ExecSessionError

    class Unavailable:
        def __init__(self, *args, **kwargs):
            pass

        def configure(self, cap):
            pass

        def run(self, *args, **kwargs):
            raise ExecSessionError(
                "execution capability requires sandbox-exec; no unsandboxed fallback"
            )

        def close(self):
            pass

    monkeypatch.setattr(kernel_hook_exec, "ExecSession", Unavailable)
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c1", data={}, match={"tool": "shell"}
    )
    assert result.decision == "deny"
    assert result.deny_code == "HOOK_FAILED"
    assert result.runs[0].outcome.error_code == "sandbox_unavailable"


@pytest.mark.parametrize("code,on_error,decision,deny_code", [
    (2, "block", "deny", "HOOK_DENIED"),
    (1, "block", "deny", "HOOK_FAILED"),
    (1, "continue", "continue", None),
])
def test_stop_gate_distinguishes_veto_error_and_explicit_continue(
        tmp_path, monkeypatch, code, on_error, decision, deny_code):
    runner, _ = setup_runner(tmp_path, event="stop", on_error=on_error)
    fake_session(monkeypatch, output=envelope(code=code, stderr=b"run checks\n"))
    try:
        result = runner.run_event("stop", boundary="stop:1", data={
            "call_id": "call-1", "source": "model", "candidate_digest": "digest",
            "changed_paths": [], "checks": {}, "blocks_so_far": 0,
            "max_blocks": 3,
        })
        assert result.decision == decision
        assert result.deny_code == deny_code
        if code == 2:
            assert result.reason == "run checks"
        elif code == 1 and decision == "deny":
            assert result.reason == "hook exited with status 1"
    finally:
        runner.close()


def test_capability_construction_failure_has_capability_error_code(tmp_path, monkeypatch):
    runner, _ = setup_runner(tmp_path)
    calls = fake_session(monkeypatch)

    def fail_capability(*args, **kwargs):
        raise ValueError("symlink read root lacks canonical target")

    monkeypatch.setattr(runner, "_capability", fail_capability)
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:capability", data={}, match={"tool": "shell"}
    )

    assert result.runs[0].outcome.error_code == "capability"
    assert result.runs[0].skipped == "capability"
    assert calls == []
    record = json.loads((runner.records_dir / "records.jsonl").read_text().splitlines()[-1])
    assert record["error_code"] == "capability"
    assert record["skipped"] == "capability"


@pytest.mark.parametrize("stage", ["constructor", "configure"])
def test_session_setup_failure_has_session_error_code(tmp_path, monkeypatch, stage):
    runner, _ = setup_runner(tmp_path)

    class BrokenSession:
        def __init__(self, *args, **kwargs):
            if stage == "constructor":
                raise PermissionError("supervisor setup denied")
            self.closed = False

        def configure(self, cap):
            raise kernel_hook_exec.ExecSessionError("session configuration denied")

        def run(self, *args, **kwargs):
            raise AssertionError("setup failure must not launch")

        def close(self):
            self.closed = True

    monkeypatch.setattr(kernel_hook_exec, "ExecSession", BrokenSession)
    result = runner.run_event(
        "pre_tool", boundary=f"pre_tool:session-{stage}", data={}, match={"tool": "shell"}
    )

    assert result.runs[0].outcome.error_code == "session"
    assert result.runs[0].skipped == "session"
    record = json.loads((runner.records_dir / "records.jsonl").read_text().splitlines()[-1])
    assert record["error_code"] == "session"
    assert record["skipped"] == "session"


def test_full_network_capability_includes_canonical_system_read_roots(tmp_path):
    runner, _ = setup_runner(tmp_path)
    config = tmp_path / "full-hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "full_network",
                        "event": "pre_tool",
                        "argv": [sys.executable, "-c", "pass"],
                        "network": "full",
                    }
                ],
            }
        )
    )
    runner.hooks = load_hook_files(
        [config], shell_network="full", repo=tmp_path / "repo"
    )
    scratch = runner.scratch_root / "full-network-test"
    scratch.mkdir()

    cap = runner._capability(runner.hooks.definitions[0], scratch)

    roots = set(cap.read_roots)
    for name in ("/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/ssl", "/etc/pki"):
        path = Path(name)
        if path.exists():
            assert path in roots
            assert path.resolve(strict=True) in roots


def test_hook_capability_and_set_include_developer_read_roots(tmp_path):
    runner, _ = setup_runner(tmp_path)
    scratch = runner.scratch_root / "developer-roots"
    scratch.mkdir()
    capability_roots = set(runner._capability(runner.hooks.definitions[0], scratch).read_roots)
    set_roots = set(runner.hooks.read_roots())
    expected = set(_macos_developer_roots())
    gitconfig = Path("/etc/gitconfig")
    if sys.platform == "darwin" and gitconfig.is_file():
        expected.update(symlink_hops(gitconfig))
    assert expected <= capability_roots
    assert expected <= set_roots


def test_operand_rechecked_immediately_before_invocation(tmp_path, monkeypatch):
    runner, workspace = setup_runner(tmp_path)
    calls = fake_session(monkeypatch)
    (workspace / "pass").write_text("appeared after load")
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c1", data={}, match={"tool": "shell"}
    )
    assert result.deny_code == "HOOK_FAILED"
    assert result.runs[0].outcome.error_code == "argv"
    assert calls == []


@pytest.mark.host_sandbox
def test_sandbox_denied_workspace_write_is_ordinary_error(tmp_path):
    runner, workspace = setup_runner(tmp_path)
    config = tmp_path / "hooks.json"
    code = "from pathlib import Path;Path('code.py').write_text('changed')"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "guard",
                        "event": "pre_tool",
                        "argv": [sys.executable, "-c", code],
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=tmp_path / "repo")
    runner.hooks = hooks
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c2", data={}, match={"tool": "shell"}
    )
    assert result.deny_code == "HOOK_FAILED"
    assert result.runs[0].outcome.error_code == "exit"
    assert (workspace / "code.py").read_text() == "original"
    runner.close()


@pytest.mark.host_sandbox
def test_sandbox_cannot_read_run_state(tmp_path):
    runner, _ = setup_runner(tmp_path)
    secret = runner.run_dir / "spec.json"
    secret.write_text("secret-marker")
    config = tmp_path / "hooks.json"
    code = f"from pathlib import Path;print(Path({str(secret)!r}).read_text())"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "guard",
                        "event": "pre_tool",
                        "argv": [sys.executable, "-c", code],
                    }
                ],
            }
        )
    )
    runner.hooks = load_hook_files(
        [config], shell_network="none", repo=tmp_path / "repo"
    )
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c3", data={}, match={"tool": "shell"}
    )
    assert result.deny_code == "HOOK_FAILED"
    assert result.runs[0].outcome.error_code == "exit"
    assert "secret-marker" not in result.reason
    runner.close()


@pytest.mark.host_sandbox
def test_sandbox_network_placement(tmp_path):
    import socket
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(10)
    port = listener.getsockname()[1]
    accepted = []

    # macOS loopback reaches host loopback services; on Linux, loopback is a private
    # network namespace (docs/guide/tools.md), so only `full` reaches this listener.
    expected = 1 if sys.platform.startswith("linux") else 2

    def accept_two():
        for _ in range(expected):
            connection, _ = listener.accept()
            accepted.append(True)
            connection.close()

    thread = threading.Thread(target=accept_two)
    thread.start()
    try:
        runner, _ = setup_runner(tmp_path)
        config = tmp_path / "hooks.json"
        code = f"import socket;socket.create_connection(('127.0.0.1',{port}),timeout=2).close()"
        hooks = [
            {
                "name": network,
                "event": "pre_tool",
                "argv": [sys.executable, "-c", code],
                "network": network,
                "on_error": "continue",
            }
            for network in ("none", "loopback", "full")
        ]
        config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": hooks}))
        runner.hooks = load_hook_files(
            [config], shell_network="full", repo=tmp_path / "repo"
        )
        result = runner.run_event(
            "pre_tool", boundary="pre_tool:network", data={}, match={"tool": "shell"}
        )
        assert [run.outcome.decision for run in result.runs] == [
            "error",
            "error" if expected == 1 else "continue",
            "continue",
        ]
        assert len(accepted) == expected
        runner.close()
    finally:
        listener.close()
        thread.join(timeout=2)


def test_first_gate_deny_short_circuits_later_hooks(tmp_path, monkeypatch):
    runner, _ = setup_runner(tmp_path)
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": name,
                        "event": "pre_tool",
                        "argv": [sys.executable, "-c", "pass"],
                    }
                    for name in ("first", "second")
                ],
            }
        )
    )
    runner.hooks = load_hook_files(
        [config], shell_network="none", repo=tmp_path / "repo"
    )
    calls = fake_session(monkeypatch, output=envelope(code=2, stderr=b"no"))
    result = runner.run_event(
        "pre_tool", boundary="pre_tool:c1", data={}, match={"tool": "shell"}
    )
    assert result.deny_hook == "first"
    assert len(result.runs) == len(calls) == 1


@pytest.mark.host_sandbox
def test_large_payload_and_argv_through_real_execution_rpc(tmp_path):
    runner, _ = setup_runner(tmp_path)
    config = tmp_path / "hooks.json"
    argv = [sys.executable, "-c", "pass"] + ["x"] * 61
    remaining = 65536 - len(
        json.dumps(argv, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    for index in range(3, len(argv)):
        extra = min(4095, remaining)
        argv[index] += "x" * extra
        remaining -= extra
    assert remaining == 0
    assert len(argv) == 64
    assert max(map(len, argv)) <= 4096
    assert len(
        json.dumps(argv, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ) == 65536
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [{"name": "guard", "event": "pre_tool", "argv": argv}],
            }
        )
    )
    runner.hooks = load_hook_files(
        [config], shell_network="none", repo=tmp_path / "repo"
    )
    boundary = "pre_tool:large"
    data_size = 0
    base = build_payload(
        "pre_tool",
        runner.hooks.definitions[0],
        event_id=event_id("seed", boundary, 0),
        run_id="run",
        cwd=str(runner.workspace),
        data={"args": {"blob": ""}},
        version="0.2.1",
    )
    assert base.payload is not None
    data_size = (256 << 10) - len(base.payload)
    data = {"args": {"blob": "x" * data_size}}
    maximum_payload = build_payload(
        "pre_tool",
        runner.hooks.definitions[0],
        event_id=event_id("seed", boundary, 0),
        run_id="run",
        cwd=str(runner.workspace),
        data=data,
        version="0.2.1",
    )
    assert maximum_payload.payload is not None
    assert len(maximum_payload.payload) == 256 << 10
    result = runner.run_event(
        "pre_tool",
        boundary=boundary,
        data=data,
        match={"tool": "shell"},
    )
    assert result.decision == "continue"
    record = json.loads(
        (runner.records_dir / "records.jsonl").read_text().splitlines()[-1]
    )
    assert record["input_bytes"] == 256 << 10
    runner.close()
