from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import kernel_challenge_budget as budget
import kernel_live_projection as projection
from kernel_green_lock import GreenLock, product_revision_change_from_parent
from kernel_host_done_facts import workspace_snapshot
from stateframe import _child, workcopy
from stateframe import workspace_start_snapshot as snapshot


@pytest.fixture(autouse=True)
def fresh_attempt(monkeypatch):
    monkeypatch.setattr(snapshot, "_worker", None)
    monkeypatch.setattr(snapshot, "_worker_revoked", False)
    monkeypatch.setattr(_child, "HOST_STATE", {})
    monkeypatch.delenv("STATEFRAME_WORKSPACE_OBJECTS", raising=False)


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    for path, data in {
        "app.py": b"VALUE = 0\r\n",
        "config/tool.sh": b"#!/bin/sh\n",
        "tests/test_app.py": b"assert False\n",
        ".visible-to-notices": b"one\n",
        ".test-tmp/ignored.txt": b"old output\n",
        "deleted.py": b"DELETE = True\n",
        ".env": b"SECRET=never-copy\n",
        "skip.pyc": b"cached",
    }.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (root / "config/tool.sh").chmod(0o751)
    (root / "tests/test_app.py").chmod(0o444)
    (root / "empty").mkdir()
    (root / "alias").symlink_to("app.py")
    (root / "dir-link").symlink_to("config", target_is_directory=True)
    return root


def _create(root, run, *, admitted):
    work = workcopy.create(root, run / "workspace", admit_snapshot=admitted)
    if admitted:
        snapshot.pin_parent(work, run / "parent")
    else:
        work.snapshot = Path(shutil.copytree(work.copy, run / "parent"))
    return work


def _inventory(root):
    members = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        members[path.relative_to(root).as_posix()] = (
            mode, path.read_bytes() if stat.S_ISREG(mode) else None)
    return members


def _worker(work, root, *, admitted, declared=b"VALUE = 1\r\n", public="assert True\n"):
    root.mkdir()
    (root / "app.py").write_bytes(declared)
    (root / "generated").mkdir()
    (root / "generated/value").write_bytes(b"generated data\n")

    def public_materializer(path, fileset, row):
        assert (path / "tests/test_app.py").read_text() == "assert False\n"
        assert stat.S_IMODE((path / "tests/test_app.py").stat().st_mode) == 0o644
        (path / "tests/test_app.py").write_text(public, encoding="utf-8")
        (path / "tests/test_app.py").chmod(0o640)
        return ("tests/test_app.py",)

    rr = SimpleNamespace(_materialize_public_tests=public_materializer)
    _child.install_repo_context(rr, work.snapshot, admitted=admitted)
    assert rr._materialize_public_tests(root, None, {}) == ("tests/test_app.py",)
    return root


def _runtime(root, **kwargs):
    return SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={"editable_paths": ["app.py", "new.py"]}),
        binding=partial(projection.workspace_binding, root, "environment"),
        **kwargs,
    )


@pytest.mark.parametrize("clone_error", [errno.EXDEV, errno.ENOTSUP, errno.EIO])
def test_membership_modes_and_independent_copies(original, tmp_path, monkeypatch, clone_error):
    def failed_clone(source, target):
        os.write(target, b"partial clone with excess bytes\n")
        raise OSError(clone_error, os.strerror(clone_error))

    monkeypatch.setattr(snapshot, "_reflink", failed_clone)
    eager = _create(original, tmp_path / "off", admitted=False)
    admitted = _create(original, tmp_path / "on", admitted=True)
    assert admitted.baseline == eager.baseline
    assert admitted.skipped == eager.skipped
    assert _inventory(admitted.copy) == _inventory(admitted.snapshot) == _inventory(eager.copy)
    worker = _worker(admitted, tmp_path / "worker", admitted=True)
    eager_worker = _worker(eager, tmp_path / "eager", admitted=False)
    assert _inventory(worker) == _inventory(eager_worker)
    for path in ("config/tool.sh", "deleted.py"):
        assert (admitted.copy / path).stat().st_mtime_ns == (admitted.snapshot / path).stat().st_mtime_ns
        assert (worker / path).stat().st_mtime_ns == (admitted.snapshot / path).stat().st_mtime_ns
        assert len({(root / path).stat().st_ino
                    for root in (admitted.copy, admitted.snapshot, worker)}) == 3
    (worker / "config/tool.sh").write_bytes(b"changed")
    (worker / "config/tool.sh").chmod(0o644)
    (admitted.copy / "config/tool.sh").write_bytes(b"changed in workspace")
    (admitted.copy / "config/tool.sh").chmod(0o700)
    assert _inventory(admitted.snapshot) == _inventory(eager.snapshot)


def test_admission_hashes_the_copied_bytes(original, tmp_path, monkeypatch):
    original_bytes = (original / "app.py").read_bytes()

    def replace_after_read(source, target):
        if os.fstat(source).st_size == len(original_bytes):
            (original / "app.py").write_bytes(original_bytes.replace(b"0", b"2"))
        raise OSError(errno.EIO, "no clone")

    monkeypatch.setattr(snapshot, "_reflink", replace_after_read)
    work = _create(original, tmp_path / "run", admitted=True)
    assert (original / "app.py").read_bytes() != original_bytes
    assert work.baseline["app.py"] == hashlib.sha256(original_bytes).hexdigest()
    assert (work.snapshot / "app.py").read_bytes() == original_bytes
    (work.copy / "app.py").write_bytes(b"VALUE = 4\r\n")
    assert "-VALUE = 0\r\n+VALUE = 4" in workcopy.unified_diff(work)


def test_path_replaced_with_symlink_during_admission(original, tmp_path, monkeypatch):
    external = tmp_path / "outside"
    external.write_bytes(b"outside")
    real_open = os.open

    def replace_at_open(path, flags, *args, **kwargs):
        if path == "app.py" and kwargs.get("dir_fd") is not None:
            victim = original / path
            victim.unlink()
            victim.symlink_to(external)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_at_open)
    work = _create(original, tmp_path / "run", admitted=True)
    assert "app.py" in work.skipped
    assert "app.py" not in work.baseline
    assert not (work.copy / "app.py").exists()
    assert not (work.snapshot / "app.py").exists()


def test_limits_refuse_before_publication(original, tmp_path):
    run = tmp_path / "failed"
    with pytest.raises(workcopy.WorkCopyError, match="larger than the run limits"):
        workcopy.create(original, run / "workspace", workcopy.Limits(max_files=1),
                        admit_snapshot=True)
    assert list(run.iterdir()) == []


def test_admission_preserves_empty_and_exact_limit_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    exact = b"\xef\xbb\xbf" + "# café\r\n".encode() + b"\x00\xff"
    files = {"empty.txt": b"", "café.bin": exact}
    for name, data in {**files, "oversize.bin": exact + b"\x00"}.items():
        (source / name).write_bytes(data)

    limits = workcopy.Limits(max_files=2, max_file_bytes=len(exact), max_total_bytes=len(exact))
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    for admitted in (False, True):
        run = tmp_path / str(admitted)
        work = workcopy.create(source, run / "workspace", limits, admit_snapshot=admitted)
        assert work.baseline == expected
        assert work.skipped == ["oversize.bin"]
        copies = [work.copy]
        if admitted:
            copies.append(snapshot.pin_parent(work, run / "parent"))
        for root in copies:
            assert {path.name: path.read_bytes() for path in root.iterdir()} == files


@pytest.mark.parametrize("platform", ["native", "darwin"])
def test_workspace_admission_uses_portable_copy_at_a_relocated_root(tmp_path, monkeypatch, platform):
    if platform == "darwin":
        monkeypatch.setattr(snapshot, "sys", SimpleNamespace(platform="darwin"))

        def unexpected_ioctl(*_args):
            pytest.fail("the macOS copy attempted the Linux clone ioctl")

        monkeypatch.setattr(fcntl, "ioctl", unexpected_ioctl)
    root = tmp_path / "source with spaces é"
    root.mkdir()
    original = root / "entry.sh"
    original.write_bytes(b"#!/bin/sh\nprintf '%s\\n' portable\n")
    original.chmod(0o751)
    work = _create(root, tmp_path / "run with spaces é", admitted=True)

    expected_mtime = (work.copy / original.name).stat().st_mtime_ns
    for copy in (work.copy, work.snapshot):
        path = copy / original.name
        assert path.read_bytes() == original.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o751
        assert path.stat().st_mtime_ns == expected_mtime
        assert not path.samefile(original)


@pytest.mark.parametrize("corruption", ["missing", "malformed", "membership"])
def test_manifest_miss_uses_the_ordinary_overlay(original, tmp_path, corruption):
    work = _create(original, tmp_path / "run", admitted=True)
    if corruption == "missing":
        (work.copy.parent / snapshot._MANIFEST).unlink()
    elif corruption == "malformed":
        (work.copy.parent / snapshot._MANIFEST).write_text("[]")
    else:
        (work.snapshot / "added").write_bytes(b"parent membership changed")
    root = _worker(work, tmp_path / "worker", admitted=True)
    assert _inventory(root) == _inventory(_worker(work, tmp_path / "eager", admitted=False))
    assert snapshot.startup_observation(root, "binding") is None


def test_projections_reuse_copy_hashes_then_read_live_facts(original, tmp_path, monkeypatch):
    work = _create(original, tmp_path / "run", admitted=True)
    root = _worker(work, tmp_path / "worker", admitted=True)
    eager = _worker(work, tmp_path / "eager", admitted=False)
    runtime = _runtime(
        root, workspace=SimpleNamespace(directory=root / "generated"),
        journal=SimpleNamespace(path=root / "deleted.py"),
    )
    eager_runtime = _runtime(
        eager, workspace=SimpleNamespace(directory=eager / "generated"),
        journal=SimpleNamespace(path=eager / "deleted.py"),
    )
    expected_binding = eager_runtime.binding()
    expected_revision = budget.artifact_revision(eager_runtime)
    calls = {"binding": 0, "artifact": 0}
    real_file_value, real_artifact_files = projection._file_value, budget._artifact_files

    def file_value(path):
        calls["binding"] += 1
        return real_file_value(path)

    def artifact_files(runtime, path):
        calls["artifact"] += 1
        return real_artifact_files(runtime, path)

    monkeypatch.setattr(projection, "_file_value", file_value)
    monkeypatch.setattr(budget, "_artifact_files", artifact_files)
    store = projection.WorkspaceViewStore(root, "environment")
    assert store.current_view().as_binding() == expected_binding
    assert budget.artifact_revision(runtime) == expected_revision
    assert calls == {"binding": 0, "artifact": 0}
    binding = snapshot.startup_observation(root, "binding")
    notice = snapshot.startup_observation(root, "notice")
    assert binding.identity != notice.identity
    assert binding.membership_digest == notice.membership_digest
    assert binding.validate_for_use("binding")
    assert not binding.validate_for_use("notice")
    assert binding.files["app.py"] == hashlib.sha256(b"VALUE = 1\r\n").hexdigest()
    assert stat.S_IMODE(binding.modes["tests/test_app.py"]) == 0o640

    # This path contributes to the notice, but not to the binding.
    (root / ".visible-to-notices").write_text("two\n")
    assert snapshot.startup_observation(root, "notice") is None
    assert not notice.validate_for_use("notice")
    assert budget.artifact_revision(runtime) != expected_revision
    assert store.current_view().as_binding() == expected_binding
    assert calls["binding"] > 0
    assert calls["artifact"] == 1


def test_unsafe_boundary_retains_the_live_view_epoch(original, tmp_path):
    work = _create(original, tmp_path / "run", admitted=True)
    root = _worker(work, tmp_path / "worker", admitted=True)
    eager = _worker(work, tmp_path / "eager", admitted=False)
    observed = projection.WorkspaceViewStore(root, "environment")
    ordinary = projection.WorkspaceViewStore(eager, "environment")
    assert observed.current_view().as_binding() == ordinary.current_view().as_binding()
    for path in (root, eager):
        (path / ".visible-to-notices").write_text("two\n")
    current, expected = observed.current_view("external"), ordinary.current_view("external")
    assert current.as_binding() == expected.as_binding()
    assert current.mutation_epoch == expected.mutation_epoch == 1
    assert snapshot.startup_observation(root, "notice") is None


@pytest.mark.parametrize("phase", ["materialization", "observation"])
def test_revocation_rejects_in_flight_workspace_reuse(original, tmp_path, monkeypatch, phase):
    work = _create(original, tmp_path / "run", admitted=True)
    root = tmp_path / "worker"
    if phase == "materialization":
        root.mkdir()
        context = snapshot.materialize_worker(work.snapshot, root, None, {})
        assert context is not None
        operation = partial(snapshot.finalize_worker, context, ())
    else:
        _worker(work, root, admitted=True)
        operation = partial(snapshot.startup_observation, root, "workspace-binding")

    scanned, proceed = Event(), Event()
    inventory = snapshot._inventory

    def pause_after_scan(path):
        result = inventory(path)
        scanned.set()
        assert proceed.wait(10)
        return result

    monkeypatch.setattr(snapshot, "_inventory", pause_after_scan)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(operation)
        try:
            assert scanned.wait(10)
            snapshot.revoke_startup_observation()
        finally:
            proceed.set()
        assert pending.result() is None
    assert snapshot.startup_observation(root, "workspace-binding") is None


@pytest.mark.parametrize("action", ["edit", "shell", "harness"])
def test_before_first_consumer_mutation_uses_eager_facts(original, tmp_path, action):
    work = _create(original, tmp_path / "run", admitted=True)
    root = _worker(work, tmp_path / "worker", admitted=True)
    if action == "edit":
        workcopy.apply_files(workcopy.WorkCopy(original, root), {"app.py": b"VALUE = 2\r\n"},
                             declared={"app.py"})
    elif action == "shell":
        subprocess.run([sys.executable, "-c", "from pathlib import Path; "
                        "Path('app.py').write_bytes(b'VALUE = 2\\r\\n'); "
                        "Path('new.py').write_bytes(b'new'); Path('deleted.py').unlink()"],
                       cwd=root, check=True)
    else:
        (root / "tests/test_app.py").write_text("assert None\n")
    assert snapshot.startup_observation(root, "artifact") is None
    expected = projection.workspace_binding(root, "environment")
    assert projection.WorkspaceViewStore(root, "environment").current_view().as_binding() == expected
    runtime = _runtime(root)
    assert budget.artifact_revision(runtime) == budget._digest({"files": budget._artifact_files(runtime, root)})


@pytest.mark.parametrize("consumer", ["binding", "view", "artifact"])
def test_workspace_consumers_recheck_files_changed_after_admission(original, tmp_path, monkeypatch, consumer):
    work = _create(original, tmp_path / "run", admitted=True)
    root = _worker(work, tmp_path / "worker", admitted=True)
    runtime = _runtime(root)
    observe = snapshot.startup_observation
    changed = False
    updated = b"VALUE = 2\r\n"

    def observe_then_change(path, purpose):
        nonlocal changed
        observed = observe(path, purpose)
        if observed is not None and not changed:
            changed = True
            (root / "app.py").write_bytes(updated)
        return observed

    monkeypatch.setattr(snapshot, "startup_observation", observe_then_change)
    if consumer == "artifact":
        actual = budget.artifact_revision(runtime)
        expected = budget._digest({"files": budget._artifact_files(runtime, root)})
    else:
        binding = (runtime.binding() if consumer == "binding"
                   else projection.WorkspaceViewStore(root, "environment").current_view().as_binding())
        actual = binding["product_files"]["app.py"]
        expected = "sha256:" + hashlib.sha256(updated).hexdigest()
    assert changed
    assert actual == expected


def test_terminal_diff_uses_effective_start_after_notice_preparation(original, tmp_path):
    work = _create(original, tmp_path / "run", admitted=True)
    root = _worker(work, tmp_path / "worker", admitted=True)
    eager = _worker(work, tmp_path / "eager", admitted=False)
    runtime = _runtime(root, _host_done_facts_baseline=workspace_snapshot(root))
    eager_runtime = _runtime(eager, _host_done_facts_baseline=workspace_snapshot(eager))
    lock, eager_lock = GreenLock(runtime, mode="terminal"), GreenLock(eager_runtime, mode="terminal")
    assert lock._starting_binding == eager_lock._starting_binding
    planned = budget.artifact_revision(runtime)
    assert planned == budget.artifact_revision(eager_runtime)
    for path in (root, eager):
        (path / "app.py").write_bytes(b"VALUE = 2\r\n")
        (path / "new.py").write_text("NEW = 1\n")
        (path / "deleted.py").unlink()
    assert budget.artifact_revision(runtime) != planned
    assert budget.artifact_revision(runtime) == budget.artifact_revision(eager_runtime)
    assert lock._terminal_change_diff() == eager_lock._terminal_change_diff()
    assert "-VALUE = 1\n+VALUE = 2" in lock._terminal_change_diff()
    assert product_revision_change_from_parent(runtime)[0]
    binding = runtime.binding()
    runtime.test_receipts = [{
        "producer": "test_subprocess", "selectors": ["check"], "exit_code": 0,
        "binding": binding, "after_binding": binding, "binding_stable": True,
    }]
    runtime.owner.f2p_ids = ["check"]
    lock.applied_product_edit("app.py")
    assert lock.sync()
    assert lock.before_mutation("app.py").startswith("GREEN LOCK:")


def test_recorded_choice_is_host_only(monkeypatch):
    active = []

    @contextmanager
    def child_scope():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setitem(sys.modules, "stateframe.send1_probe_cache",
                        SimpleNamespace(child_scope=child_scope))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(_child, "_RUNNER_HOME_IDENTITY_CAPTURED", False)
    monkeypatch.setattr(_child, "_RUNNER_HOME_IDENTITY_PIN", None)
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    spec = {"environment": {"STATEFRAME_SEND1_WAVE4": "on"}}
    with ExitStack() as lifetime:
        _child.install_environment(spec, lifetime)
        assert "STATEFRAME_SEND1_WAVE4" not in os.environ
        assert not active
    with ExitStack() as lifetime:
        _child.install_environment({**spec, "send1_wave4": True}, lifetime)
        assert "STATEFRAME_SEND1_WAVE4" not in os.environ
        assert active == [True]
    assert not active
