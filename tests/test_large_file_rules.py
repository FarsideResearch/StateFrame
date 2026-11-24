"""Files over 1 MiB keep the protection, delivery and copy rules of smaller files."""

import errno
import hashlib
import json
import os
import stat
import subprocess

import pytest

from stateframe import chat, resume, runner, workcopy
from stateframe.endpoint import Endpoint
from stateframe.task_row import editable_large_files
from stateframe.workspace_start_snapshot import pin_parent


def options(repo, run, *, admitted=True, **kwargs):
    opts = runner.RunOptions(
        repo, "Update app.py", Endpoint("http://127.0.0.1:65534/v1", "offline"),
        run_dir=run, no_powerups=True, no_project_hooks=True, **kwargs,
    )
    opts._send1_wave4 = admitted
    return opts


@pytest.fixture
def repo(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n")
    (repo / "test_app.py").write_text("def test_app(): assert True\n")
    return repo


@pytest.mark.parametrize("admitted", [False, True])
def test_ordinary_tree_membership_and_file_mtime(repo, tmp_path, admitted):
    app = repo / "app.py"
    os.chmod(app, 0o751)
    mtime = 946684800123456789
    os.utime(app, ns=(mtime, mtime))
    (repo / "empty").mkdir()
    (repo / "cache-only").mkdir()
    (repo / "cache-only" / "app.pyc").write_bytes(b"cache")

    _, work, _ = runner.prepare(
        options(repo, tmp_path / "run", admitted=admitted), printer=lambda *_: None,
    )
    session = chat.Session(repo, tmp_path / "session", None)
    session.start(printer=lambda *_: None)
    copies = (work.copy, work.snapshot, session.copy, session.start_snapshot)
    actual = []
    for root in copies:
        assert (root / "app.py").read_bytes() == app.read_bytes()
        assert stat.S_IMODE((root / "app.py").stat().st_mode) == 0o751
        actual.append((root.name, (root / "app.py").stat().st_mtime_ns == mtime,
                       sorted(p.name for p in root.iterdir())))
    assert all(not preserved and names == ["app.py", "test_app.py"]
               for _, preserved, names in actual), actual


@pytest.mark.parametrize("selection", ["target", "check"])
def test_large_path_can_be_selected(repo, tmp_path, selection):
    relative = {"target": "app.py", "check": "test_app.py"}[selection]
    marker = "only the size matters"
    with (repo / relative).open("a") as output:
        output.write("# " + marker + "x" * (1 << 20) + "\n")
    flags = {"target": relative} if selection == "target" else {"checks": [relative]}

    _, work, spec = runner.prepare(
        options(repo, tmp_path / "run", **flags), printer=lambda *_: None,
    )
    assert (work.copy / relative).is_file()
    if selection == "target":
        assert spec["public_row"]["entry"]["path"] == relative
    else:
        assert relative in spec["public_row"]["public_check_paths"]


def test_allowed_large_shell_edit_is_delivered_by_host(repo, tmp_path):
    from kernel_done_is_submit_authority import store_candidate
    from kernel_effects import ContentStore, effect_policy_for_run, reconcile, scan

    relative = "payload.txt"
    original = b"value=A\n" + b"x" * (2 << 20) + b"\n"
    changed = b"value=B\n" + original.split(b"\n", 1)[1]
    (repo / relative).write_bytes(original)
    run, work, spec = runner.prepare(
        options(repo, tmp_path / "run"), printer=lambda *_: None,
    )
    row = spec["public_row"]
    assert row["scoped_edits"] is False
    assert relative not in row["files"]
    store = ContentStore(run)
    def manifest():
        return scan(work.copy, exclude=lambda _: False)

    before = manifest()
    store.capture(work.copy, before)
    policy = effect_policy_for_run(work.copy, row, [], baseline_manifest=before)
    (work.copy / relative).write_bytes(changed)
    effects = reconcile(work.copy, before, manifest(), policy, store, None)
    assert [effect.path for effect in effects.applied] == [relative]
    out = run / "worker" / "results" / "react_done_submissions"
    terminal = store_candidate(out, row["task_id"], {relative: changed.decode()})
    (run / "engine-result.json").write_text(json.dumps({
        "result": {
            "result": "SUBMITTED",
            "submission": dict(terminal, seal_id=terminal["artifact_digest"]),
        },
    }))
    candidate = runner.final_candidate(run)
    assert candidate == {relative: changed}

    delivered, dropped, error = runner._apply_sealed_candidate(
        work, candidate, set(row["files"]) | editable_large_files(work, row), printer=lambda *_: None,
    )
    assert "+value=B\n" in workcopy.unified_diff(work)
    assert not dropped and error is None, (delivered, dropped, error)


@pytest.mark.parametrize("padding", [0, 2 << 20])
def test_generic_check_script_stays_protected(repo, tmp_path, padding):
    from kernel_effects import ContentStore, effect_policy_for_run, reconcile, scan

    relative = "verify.sh"
    original = b"exit 23\n#" + b"x" * padding + b"\n"
    (repo / relative).write_bytes(original)
    run, work, spec = runner.prepare(
        options(
            repo, tmp_path / "run", runner_kind="generic",
            test_cmd="bash verify.sh", protect_tests="all",
        ),
        printer=lambda *_: None,
    )
    row = spec["public_row"]
    protected = {str(work.copy / path) for path in row["protected_test_paths"]}

    def check_exit_code():
        return subprocess.run(
            ["bash", "verify.sh"],
            cwd=work.copy, check=False,
        ).returncode

    assert check_exit_code() == 23
    before = scan(work.copy, exclude=lambda _: False)
    store = ContentStore(run)
    store.capture(work.copy, before)
    policy = effect_policy_for_run(work.copy, row, protected, baseline_manifest=before)
    (work.copy / relative).write_bytes(b"exit 0\n")
    effects = reconcile(work.copy, before, scan(work.copy, exclude=lambda _: False),
                        policy, store, store)
    actual = check_exit_code()
    assert actual == 23, (actual, row["protected_test_paths"], effects.applied)
    assert (work.copy / relative).read_bytes() == original


def test_chat_start_cleans_first_copy_after_second_copy_enospc(repo, tmp_path, monkeypatch):
    session = chat.Session(repo, tmp_path / "session", None)
    original = workcopy._Copies.file

    def fail_in_start_snapshot(self, source, target, info, **kwargs):
        if target == session.start_snapshot / "app.py":
            target.write_bytes(b"part")
            raise OSError(errno.ENOSPC, "injected disk full")
        return original(self, source, target, info, **kwargs)

    monkeypatch.setattr(workcopy._Copies, "file", fail_in_start_snapshot)
    with pytest.raises(workcopy.WorkCopyError, match="disk full"):
        session.start(printer=lambda *_: None)
    assert not session.start_snapshot.exists()
    assert not session.copy.exists()


@pytest.mark.parametrize("size", [1 << 20, 20 << 20])
def test_corrupt_rollback_slice_is_rejected_before_publish(tmp_path, size):
    from kernel_effects import BaselineStore, ContentStore, EffectRestoreError
    from kernel_effects import WorkspaceEffectPolicy, reconcile, scan

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "payload.bin"
    with path.open("wb") as output:
        output.truncate(size)
    before = scan(workspace, exclude=lambda _: False)
    baseline_root = tmp_path / "baseline"
    baseline = BaselineStore(baseline_root, before)
    baseline.publish(workspace)
    pack, = baseline_root.glob("*/objects.pack")
    pack.chmod(0o644)
    with pack.open("r+b") as output:
        output.write(b"corrupt")

    with path.open("r+b") as output:
        output.write(b"refused")
    after = scan(workspace, exclude=lambda _: False)
    policy = WorkspaceEffectPolicy(lambda _: True, None, lambda _: True, before)
    with pytest.raises(EffectRestoreError):
        reconcile(workspace, before, after, policy, ContentStore(tmp_path), baseline)
    assert scan(workspace, exclude=lambda _: False) == after


def test_run_killed_during_its_start_copy_says_how_to_go_on(repo, tmp_path, monkeypatch):
    payload = repo / "payload.bin"
    with payload.open("wb") as output:
        output.truncate(20 << 20)
    run = tmp_path / "run"
    original = workcopy._Copies.file

    def stop_after_first_block(self, source, target, info, **kwargs):
        if target.name == payload.name:
            with target.open("xb") as output:
                output.write(os.read(source, 1 << 20))
            os._exit(9)  # no cleanup, like a kill
        return original(self, source, target, info, **kwargs)

    monkeypatch.setattr(workcopy._Copies, "file", stop_after_first_block)
    pid = os.fork()
    if pid == 0:
        try:
            runner.prepare(options(repo, run), printer=lambda *_: None)
        finally:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 9
    monkeypatch.setattr(workcopy._Copies, "file", original)
    partial, = run.glob(".workspace-start-*/workspace/payload.bin")
    assert partial.stat().st_size == 1 << 20
    with pytest.raises(resume.ResumeRefused, match="new run directory"):
        resume.assess(run)
    _, work, _ = runner.prepare(options(repo, tmp_path / "fresh"), printer=lambda *_: None)
    assert (work.copy / payload.name).stat().st_size == 20 << 20


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("grows", ["appended", "truncated"])
@pytest.mark.parametrize("size, starts", [(2 << 20, True), (100, False)])
def test_file_that_keeps_changing_while_copied(repo, tmp_path, monkeypatch, admitted, grows, size, starts):
    log = repo / "nested" / "build.log"
    log.parent.mkdir()
    log.write_bytes(b"x" * size)
    original = workcopy._Copies.file
    seen = []

    def change(self, source, target, info, **kwargs):
        if target.name == log.name:
            seen.append(target)
            with log.open("r+b") as stream:
                if grows == "appended":
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"more")
                else:
                    stream.truncate(max(0, info.st_size // 2))
        return original(self, source, target, info, **kwargs)

    monkeypatch.setattr(workcopy, "_reflink", _no_clone)
    monkeypatch.setattr(workcopy._Copies, "file", change)
    destination = tmp_path / "run" / "workspace"
    if not starts:
        with pytest.raises(workcopy.WorkCopyError, match="nested/build.log"):
            workcopy.create(repo, destination, admit_snapshot=admitted)
        return
    work = workcopy.create(repo, destination, admit_snapshot=admitted)
    rel = "nested/build.log"
    assert len(seen) == 2
    assert work.skip_reasons == {rel: "changed while copying"}
    assert set(work.baseline) == {"app.py", "test_app.py"} and not work.large_files
    if admitted:
        pin_parent(work, tmp_path / "run" / "parent")
        assert not (tmp_path / "run" / "parent" / rel).exists()
    assert not (destination / rel).exists()


@pytest.mark.parametrize("admitted", [False, True])
def test_file_that_changes_once_is_copied_again(repo, tmp_path, monkeypatch, admitted):
    original = workcopy._Copies.file
    seen = []

    def change_once(self, source, target, info, **kwargs):
        result = original(self, source, target, info, **kwargs)
        if target.name == "app.py" and not seen:
            seen.append(target)
            (repo / "app.py").write_text("VALUE = 22\n")
        return result

    monkeypatch.setattr(workcopy._Copies, "file", change_once)
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=admitted)
    assert (work.copy / "app.py").read_text() == "VALUE = 22\n"
    assert work.skipped == []


def _no_clone(_source, _target):
    raise OSError(errno.ENOTSUP, "clone unavailable")


def test_effect_manifest_keeps_a_20_mib_file_descriptor():
    from kernel_workspace_consumers import _live_manifest
    from kernel_workspace_tree import File, directory

    size = 20 << 20
    tree = directory(0o755, ((b"archive.bin", File(0o644, size, hashlib.sha256(b"archive").digest())),))

    manifest = _live_manifest(tree, lambda _: False)

    assert manifest["archive.bin"].size == size
