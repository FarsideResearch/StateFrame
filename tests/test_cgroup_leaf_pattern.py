"""Fake cgroup v2 trees exercise the delegated leaf setup and its rollback."""
from __future__ import annotations

import errno
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import sf_launch


LIMIT = 64 << 20


def _leaf_race_child(root, ready, start, setup_ready, child_attempt, results):
    ready.put(("child", os.getpid()))
    if not start.wait(10) or not setup_ready.wait(10):
        results.put(("child", None, "worker_timeout"))
        return
    group = sf_launch._command_cgroup("multiprocess-child", 8, memory_bytes=LIMIT)
    results.put(("child", str(group) if group is not None else None,
                 sf_launch._cgroup_leaf_pattern_status()))


def _leaf_race_parent(root, ready, start, setup_ready, child_attempt, results):
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_leaf_race_child,
        args=(root, ready, start, setup_ready, child_attempt, results),
    )
    child.start()
    ready.put(("parent", os.getpid()))
    if not start.wait(10):
        results.put(("parent", None, "worker_timeout"))
        child.join(10)
        return
    group = sf_launch._command_cgroup("multiprocess-parent", 8, memory_bytes=LIMIT)
    results.put(("parent", str(group) if group is not None else None,
                 sf_launch._cgroup_leaf_pattern_status()))
    child.join(10)
    if child.is_alive():
        child.terminate()
        child.join(2)
        results.put(("child", None, "worker_timeout"))


def _owned_leaf_fork_child(root, ready, start, results):
    ready.put(os.getpid())
    if not start.wait(10):
        results.put((None, "worker_timeout"))
        return
    sf_launch._current_cgroup = lambda: Path(root) / "supervisor"
    group = sf_launch._command_cgroup("fork-owned", 8, memory_bytes=LIMIT)
    results.put((str(group) if group is not None else None,
                 sf_launch._cgroup_leaf_pattern_status()))


@pytest.fixture
def cgroup_tree(monkeypatch, tmp_path):
    root = tmp_path / "delegated"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids")
    (root / "cgroup.subtree_control").write_text("")
    (root / "cgroup.procs").write_text(str(os.getpid()))

    real_mkdir = Path.mkdir
    real_rmdir = Path.rmdir
    real_write_text = Path.write_text

    def virtual_controls(group: Path) -> None:
        (group / "cgroup.procs").touch()
        (group / "cgroup.controllers").write_text("")
        (group / "cgroup.subtree_control").write_text("")
        enabled = set((group.parent / "cgroup.subtree_control").read_text().split())
        if "pids" in enabled:
            (group / "pids.max").touch()
        if "memory" in enabled:
            for name in ("memory.max", "memory.swap.max", "memory.oom.group"):
                (group / name).touch()

    def fake_mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path.parent == root and (path.name == "supervisor"
                                    or path.name.startswith("stateframe-")):
            virtual_controls(path)
        return result

    def fake_rmdir(path):
        if path.parent == root and path.is_dir():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
        return real_rmdir(path)

    def fake_write_text(path, value, *args, **kwargs):
        if path.name == "cgroup.procs" and root in path.parents:
            requested = str(value).split()
            if len(requested) != 1:
                return real_write_text(path, value, *args, **kwargs)
            pid = int(str(value).strip())
            for membership in root.rglob("cgroup.procs"):
                lines = [line for line in membership.read_text().splitlines()
                         if line.strip() and int(line) != pid]
                if lines != membership.read_text().splitlines():
                    real_write_text(membership, "\n".join(lines), encoding="ascii")
            current = path.read_text().splitlines()
            if str(pid) not in current:
                current.append(str(pid))
            return real_write_text(path, "\n".join(current), encoding="ascii")
        if path == root / "cgroup.subtree_control":
            operations = str(value).split()
            if operations and all(not item.startswith(("+", "-")) for item in operations):
                return real_write_text(path, " ".join(operations), encoding="ascii")
            if (root / "cgroup.procs").read_text().strip():
                raise OSError(16, "cgroup contains processes")
            enabled = set(path.read_text().split())
            for operation in operations:
                controller = operation[1:]
                if operation.startswith("+"):
                    enabled.add(controller)
                elif operation.startswith("-"):
                    enabled.discard(controller)
            return real_write_text(path, " ".join(sorted(enabled)), encoding="ascii")
        return real_write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(Path, "rmdir", fake_rmdir)
    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: root)
    sf_launch._reset_cgroup_leaf_pattern_state()
    yield root
    sf_launch._reset_cgroup_leaf_pattern_state()


def _fake_stateframe_scope_processes(monkeypatch, script_path: Path, *, foreign=False):
    cli_pid, engine_pid, helper_pid, sibling_helper_pid, foreign_pid = (
        2_001_001, 2_001_002, 2_001_003, 2_001_004, 2_001_005)
    uid = os.getuid()
    details = {
        cli_pid: (1, uid),
        engine_pid: (cli_pid, uid),
        helper_pid: (engine_pid, uid),
        sibling_helper_pid: (engine_pid, uid),
        foreign_pid: (1, uid),
    }
    argv = {
        cli_pid: (sys.executable, str(script_path), "run"),
        engine_pid: ("python", "-I", "engine.py"),
        helper_pid: ("python", "-I", "sf_launch.py"),
        sibling_helper_pid: ("python", "-I", "outer_checks.py"),
        foreign_pid: ("unrelated",),
    }
    script_path.write_text("from stateframe.cli import main\n")
    monkeypatch.setattr(sf_launch.os, "getpid", lambda: helper_pid)
    monkeypatch.setattr(sf_launch, "_process_details", details.__getitem__)
    monkeypatch.setattr(sf_launch, "_process_argv", argv.__getitem__)
    members = [cli_pid, engine_pid, helper_pid, sibling_helper_pid]
    if foreign:
        members.append(foreign_pid)
    return members, helper_pid, foreign_pid


def test_stateframe_scope_owns_cli_engine_and_helper_processes(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    members, helper_pid, _foreign_pid = _fake_stateframe_scope_processes(
        monkeypatch, root.parent / "stateframe")
    (root / "cgroup.procs").write_text("\n".join(map(str, members)))

    group = sf_launch._command_cgroup("stateframe-scope", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{helper_pid}-stateframe-scope"
    assert set((root / "supervisor" / "cgroup.procs").read_text().split()) == {
        str(pid) for pid in members}
    assert (root / "cgroup.procs").read_text() == ""
    assert (group / "memory.max").read_text() == str(LIMIT)
    assert sf_launch._cgroup_leaf_pattern_status() == "applied"


def test_foreign_scope_member_forces_memory_fallback(monkeypatch, cgroup_tree):
    root = cgroup_tree
    members, helper_pid, foreign_pid = _fake_stateframe_scope_processes(
        monkeypatch, root.parent / "stateframe", foreign=True)
    (root / "cgroup.subtree_control").write_text("pids")
    (root / "cgroup.procs").write_text("\n".join(map(str, members)))

    group = sf_launch._command_cgroup("foreign-stateframe-scope", 8,
                                      memory_bytes=LIMIT)

    assert group == root / f"stateframe-{helper_pid}-foreign-stateframe-scope"
    assert not (root / "supervisor").exists()
    assert set((root / "cgroup.procs").read_text().split()) == {
        str(pid) for pid in members}
    assert str(foreign_pid) in (root / "cgroup.procs").read_text().split()
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:foreign_process"


def test_similarly_named_script_does_not_claim_scope_ownership(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    script = root.parent / "stateframe"
    members, helper_pid, _foreign_pid = _fake_stateframe_scope_processes(
        monkeypatch, script)
    script.write_text("def main():\n    return 0\n")
    (root / "cgroup.subtree_control").write_text("pids")
    (root / "cgroup.procs").write_text("\n".join(map(str, members)))

    group = sf_launch._command_cgroup("impostor-stateframe", 8,
                                      memory_bytes=LIMIT)

    assert group == root / f"stateframe-{helper_pid}-impostor-stateframe"
    assert not (root / "supervisor").exists()
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:foreign_process"


@pytest.mark.parametrize("foreign", [False, True])
def test_fresh_helper_does_not_reuse_unproven_supervisor_from_cli_ancestry(
        monkeypatch, cgroup_tree, foreign):
    root = cgroup_tree
    members, _helper_pid, _foreign_pid = _fake_stateframe_scope_processes(
        monkeypatch, root.parent / "stateframe", foreign=foreign)
    supervisor = root / "supervisor"
    (root / "cgroup.subtree_control").write_text("memory pids")
    supervisor.mkdir()
    supervisor_members = members
    (supervisor / "cgroup.procs").write_text(
        "\n".join(map(str, supervisor_members)))
    (supervisor / "memory.max").write_text(str(LIMIT // 2))
    (supervisor / "pids.max").write_text("2")
    (root / "cgroup.procs").write_text("")
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: supervisor)
    # A separately executed launcher has no process-local creator proof.
    sf_launch._reset_cgroup_leaf_pattern_state()

    group = sf_launch._command_cgroup("fresh-helper", 8, memory_bytes=LIMIT)

    assert group is None
    assert sorted(path.name for path in root.glob("stateframe-*")) == []
    assert (supervisor / "cgroup.procs").read_text().split() == list(
        map(str, supervisor_members))
    assert (supervisor / "memory.max").read_text() == str(LIMIT // 2)
    assert (supervisor / "pids.max").read_text() == "2"
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:supervisor_foreign"


def test_leaf_pattern_moves_process_and_sets_sibling_memory_and_pids_limits(
        cgroup_tree):
    root = cgroup_tree

    group = sf_launch._command_cgroup("applied", 8, memory_bytes=LIMIT)

    assert group is not None
    assert group.parent == root
    assert (root / "cgroup.procs").read_text().strip() == ""
    assert (root / "supervisor" / "cgroup.procs").read_text().splitlines() == [
        str(os.getpid())]
    assert set((root / "cgroup.subtree_control").read_text().split()) >= {
        "memory", "pids"}
    assert (group / "memory.max").read_text() == str(LIMIT)
    assert (group / "memory.swap.max").read_text() == "0"
    assert (group / "pids.max").read_text() == "16"
    assert sf_launch._cgroup_leaf_pattern_status() == "applied"


def test_foreign_process_aborts_pattern_and_uses_existing_pids_fallback(
        cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    (root / "cgroup.procs").write_text(f"{os.getpid()}\n2147483000")

    group = sf_launch._command_cgroup("foreign", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-foreign"
    assert group.parent == root
    assert (root / "supervisor").exists() is False
    assert (root / "cgroup.procs").read_text().splitlines() == [
        str(os.getpid()), "2147483000"]
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status().startswith("failed:")


def test_foreign_process_uses_memory_fallback_even_if_controllers_are_enabled(
        cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("memory pids")
    (root / "cgroup.procs").write_text(f"{os.getpid()}\n2147483000")

    group = sf_launch._command_cgroup("foreign-memory", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-foreign-memory"
    assert (group / "pids.max").read_text() == "16"
    assert (group / "memory.max").read_text() == ""
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:foreign_process"


def test_same_uid_ancestor_is_foreign_and_keeps_existing_fallback(cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    ancestor = os.getppid()
    (root / "cgroup.procs").write_text(f"{os.getpid()}\n{ancestor}")

    group = sf_launch._command_cgroup("ancestor", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-ancestor"
    assert not (root / "supervisor").exists()
    assert set((root / "cgroup.procs").read_text().split()) == {
        str(os.getpid()), str(ancestor)}
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:foreign_process"


def test_nonwritable_cgroup_uses_existing_pids_fallback(monkeypatch, cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    monkeypatch.setattr(sf_launch, "_cgroup_leaf_pattern_writable", lambda _path: False)

    group = sf_launch._command_cgroup("not-delegated", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-not-delegated"
    assert group.parent == root
    assert not (root / "supervisor").exists()
    assert (root / "cgroup.procs").read_text().strip() == str(os.getpid())
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status() == "not_delegated"


def test_second_call_reuses_applied_pattern_and_creates_another_sibling(cgroup_tree):
    root = cgroup_tree

    first = sf_launch._command_cgroup("first", 8, memory_bytes=LIMIT)
    second = sf_launch._command_cgroup("second", 12, memory_bytes=LIMIT)

    assert first is not None and second is not None
    assert first.parent == root and second.parent == root
    assert (root / "supervisor").is_dir()
    assert (root / "cgroup.subtree_control").read_text().split() == ["memory", "pids"]
    assert (second / "pids.max").read_text() == "20"
    assert sf_launch._cgroup_leaf_pattern_status() == "applied"


def test_concurrent_calls_initialize_pattern_once_and_create_siblings(cgroup_tree):
    root = cgroup_tree

    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(
            lambda ident: sf_launch._command_cgroup(ident, 8, memory_bytes=LIMIT),
            ("thread-one", "thread-two")))

    assert all(group is not None and group.parent == root for group in groups)
    assert (root / "supervisor").is_dir()
    assert set((root / "cgroup.subtree_control").read_text().split()) == {
        "memory", "pids"}
    assert sf_launch._cgroup_leaf_pattern_status() == "applied"


def test_multiprocess_setup_rollback_is_consistent(cgroup_tree, monkeypatch):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    setup_ready = context.Event()
    child_attempt = context.Event()
    owner_pid = context.Value("i", 0)
    child_pid = context.Value("i", 0)
    write_text = Path.write_text
    process_lock = sf_launch._cgroup_leaf_pattern_process_lock

    def signal_child_before_lock(cgroup):
        if os.getpid() == child_pid.value:
            child_attempt.set()
        return process_lock(cgroup)

    def fail_after_child_races_setup(path, value, *args, **kwargs):
        if (path == root / "cgroup.subtree_control" and value == "+memory"
                and os.getpid() == owner_pid.value):
            write_text(path, value, *args, **kwargs)
            setup_ready.set()
            if not child_attempt.wait(5):
                raise TimeoutError("child did not race leaf setup")
            raise PermissionError("simulated setup failure")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_after_child_races_setup)
    monkeypatch.setattr(sf_launch, "_cgroup_leaf_pattern_process_lock",
                        signal_child_before_lock)
    parent = context.Process(
        target=_leaf_race_parent,
        args=(root, ready, start, setup_ready, child_attempt, results),
    )
    parent.start()
    announced = dict((name, pid) for name, pid in (ready.get(timeout=10),
                                                     ready.get(timeout=10)))
    owner_pid.value = announced["parent"]
    child_pid.value = announced["child"]
    (root / "cgroup.procs").write_text(
        f"{announced['parent']}\n{announced['child']}")
    start.set()

    parent.join(15)
    if parent.is_alive():
        parent.terminate()
        parent.join(2)
        pytest.fail("multiprocess leaf setup did not settle")
    outcomes = dict((name, (path, status)) for name, path, status in
                    (results.get(timeout=2), results.get(timeout=2)))

    assert {name for name in outcomes} == {"parent", "child"}
    assert all(path is not None and Path(path).parent == root
               for path, _status in outcomes.values())
    assert all(status.startswith("failed:") for _path, status in outcomes.values())
    assert not (root / "supervisor").exists()
    assert set((root / "cgroup.procs").read_text().split()) == {
        str(announced["parent"]), str(announced["child"])}
    assert (root / "cgroup.subtree_control").read_text().split() == ["pids"]
    assert sorted(path.name for path in root.glob("stateframe-*")) == sorted(
        Path(path).name for path, _status in outcomes.values())


def test_preexisting_supervisor_is_foreign_and_falls_back(monkeypatch, cgroup_tree):
    root = cgroup_tree
    supervisor = root / "supervisor"
    supervisor.mkdir()
    (supervisor / "cgroup.controllers").write_text("memory pids")
    (supervisor / "cgroup.subtree_control").write_text("")
    (supervisor / "cgroup.procs").write_text(str(os.getpid()))
    (root / "cgroup.procs").write_text("")
    (root / "cgroup.subtree_control").write_text("memory pids")
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: supervisor)
    sf_launch._reset_cgroup_leaf_pattern_state()

    group = sf_launch._command_cgroup("reused", 8, memory_bytes=LIMIT)

    assert group is None
    assert not (supervisor / "supervisor").exists()
    assert (supervisor / "cgroup.procs").read_text().splitlines() == [str(os.getpid())]
    assert (root / "cgroup.procs").read_text() == ""
    assert (root / "cgroup.subtree_control").read_text().split() == ["memory", "pids"]
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:supervisor_foreign"


def test_current_process_can_recognize_only_its_created_supervisor(cgroup_tree):
    root = cgroup_tree

    group = sf_launch._command_cgroup("owned", 8, memory_bytes=LIMIT)
    supervisor = root / "supervisor"

    assert group is not None
    assert sf_launch._supervisor_parent(supervisor) == root


def test_stateframe_parent_ownership_proof_survives_fork(cgroup_tree):
    root = cgroup_tree
    initial = sf_launch._command_cgroup("owner", 8, memory_bytes=LIMIT)
    assert initial is not None

    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    child = context.Process(
        target=_owned_leaf_fork_child, args=(root, ready, start, results))
    child.start()
    child_pid = ready.get(timeout=10)
    (root / "supervisor" / "cgroup.procs").write_text(
        f"{os.getpid()}\n{child_pid}")
    start.set()
    child.join(10)
    if child.is_alive():
        child.terminate()
        child.join(2)
        pytest.fail("forked StateFrame child did not settle")
    group_path, status = results.get(timeout=2)

    assert group_path is not None
    assert Path(group_path).parent == root
    assert status == "applied"


def test_existing_supervisor_rejects_mount_root_parent(monkeypatch, cgroup_tree):
    root = cgroup_tree
    supervisor = root / "supervisor"
    supervisor.mkdir()
    (supervisor / "cgroup.procs").write_text(str(os.getpid()))
    (root / "cgroup.subtree_control").write_text("memory pids")
    monkeypatch.setattr(sf_launch, "_cgroup_mount_root", lambda: root)

    assert sf_launch._supervisor_parent(supervisor) is None


def test_existing_supervisor_requires_delegated_parent_and_all_controllers(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    supervisor = root / "supervisor"
    supervisor.mkdir()
    (supervisor / "cgroup.procs").write_text(str(os.getpid()))
    (root / "cgroup.subtree_control").write_text("pids")
    assert sf_launch._supervisor_parent(supervisor) is None

    (root / "cgroup.subtree_control").write_text("memory pids")
    monkeypatch.setattr(sf_launch, "_cgroup_leaf_pattern_writable", lambda _path: False)
    assert sf_launch._supervisor_parent(supervisor) is None


def test_existing_supervisor_rejects_foreign_or_ancestor_members(cgroup_tree):
    root = cgroup_tree
    supervisor = root / "supervisor"
    supervisor.mkdir()
    (supervisor / "cgroup.procs").write_text(
        f"{os.getpid()}\n{os.getppid()}")
    (root / "cgroup.subtree_control").write_text("memory pids")

    assert sf_launch._supervisor_parent(supervisor) is None


def test_unrelated_existing_supervisor_is_left_untouched(cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    supervisor = root / "supervisor"
    supervisor.mkdir()
    foreign_pid = 2147483000
    (supervisor / "cgroup.procs").write_text(str(foreign_pid))

    group = sf_launch._command_cgroup("existing-supervisor", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-existing-supervisor"
    assert (root / "cgroup.procs").read_text().splitlines() == [str(os.getpid())]
    assert (supervisor / "cgroup.procs").read_text().splitlines() == [str(foreign_pid)]
    assert (root / "cgroup.subtree_control").read_text().split() == ["pids"]
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:supervisor_exists"


def test_failure_after_move_rolls_back_and_keeps_legacy_fallback(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    write_text = Path.write_text
    failed = False

    def fail_controller_enable(path, value, *args, **kwargs):
        nonlocal failed
        if path == root / "cgroup.subtree_control" and not failed:
            failed = True
            write_text(path, value, *args, **kwargs)
            raise PermissionError("controller write denied")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_controller_enable)

    group = sf_launch._command_cgroup("rollback", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-rollback"
    assert (root / "cgroup.procs").read_text().strip() == str(os.getpid())
    assert not (root / "supervisor").exists()
    assert (root / "cgroup.subtree_control").read_text().split() == ["pids"]
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status().startswith("failed:")


def test_rollback_preserves_controller_enabled_concurrently(monkeypatch, cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    write_text = Path.write_text
    failed = False

    def interleave_controller_enable(path, value, *args, **kwargs):
        nonlocal failed
        if path == root / "cgroup.subtree_control" and value == "+memory" and not failed:
            failed = True
            write_text(path, "+cpu", encoding="ascii")
            write_text(path, value, *args, **kwargs)
            raise PermissionError("simulated partial controller enable")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interleave_controller_enable)

    group = sf_launch._command_cgroup("interleaved-controller", 8, memory_bytes=LIMIT)

    assert group == root / f"stateframe-{os.getpid()}-interleaved-controller"
    assert (root / "cgroup.subtree_control").read_text().split() == ["cpu", "pids"]
    assert (root / "cgroup.procs").read_text().strip() == str(os.getpid())
    assert not (root / "supervisor").exists()
    assert (group / "pids.max").read_text() == "16"
    assert not (group / "memory.max").exists()
    assert sf_launch._cgroup_leaf_pattern_status().startswith("failed:")


def test_failed_rollback_returns_none_so_memory_fallback_cannot_be_skipped(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    (root / "cgroup.subtree_control").write_text("pids")
    write_text = Path.write_text
    controller_failed = False

    def fail_enable_and_return(path, value, *args, **kwargs):
        nonlocal controller_failed
        if path == root / "cgroup.subtree_control" and not controller_failed:
            controller_failed = True
            write_text(path, value, *args, **kwargs)
            raise PermissionError("controller write failed after partial enable")
        if path == root / "cgroup.procs" and str(value).strip() == str(os.getpid()):
            raise PermissionError("process return denied")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_enable_and_return)

    group = sf_launch._command_cgroup("rollback-failed", 8, memory_bytes=LIMIT)

    assert group is None
    assert (root / "supervisor" / "cgroup.procs").read_text().splitlines() == [
        str(os.getpid())]
    assert "memory" not in (root / "cgroup.subtree_control").read_text().split()
    assert not any(path.name.startswith("stateframe-") for path in root.iterdir())
    assert sf_launch._cgroup_leaf_pattern_status().startswith("failed:")


def test_esrch_during_partial_migration_rolls_back_idempotently(
        monkeypatch, cgroup_tree):
    root = cgroup_tree
    members, helper_pid, _foreign_pid = _fake_stateframe_scope_processes(
        monkeypatch, root.parent / "stateframe")
    engine_pid = members[1]
    sibling_pid = members[-1]
    supervisor = root / "supervisor"
    (root / "cgroup.procs").write_text("\n".join(map(str, members)))
    (root / "cgroup.subtree_control").write_text("pids")
    initial_paths = {path.relative_to(root) for path in root.rglob("*")}
    write_text = Path.write_text

    def remove_pid(path: Path, pid: int) -> None:
        remaining = [line for line in path.read_text().splitlines()
                     if line.strip() and int(line) != pid]
        with path.open("w", encoding="ascii") as stream:
            stream.write("\n".join(remaining))

    def exit_during_migration(path, value, *args, **kwargs):
        if path == supervisor / "cgroup.procs":
            pid = int(str(value).strip())
            if pid == engine_pid:
                remove_pid(root / "cgroup.procs", pid)
                raise OSError(errno.ESRCH, "process exited during migration")
        if path == root / "cgroup.procs" and int(str(value).strip()) == sibling_pid:
            remove_pid(supervisor / "cgroup.procs", sibling_pid)
            raise OSError(errno.ESRCH, "process exited during rollback")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exit_during_migration)

    parent, applied, status = sf_launch._try_cgroup_leaf_pattern(root)

    assert not supervisor.exists()
    assert parent == root
    assert not applied
    assert status.startswith("failed:")
    assert set((root / "cgroup.procs").read_text().split()) == {
        str(pid) for pid in members if pid not in {engine_pid, sibling_pid}}
    assert (root / "cgroup.controllers").read_text().split() == [
        "cpu", "memory", "pids"]
    assert (root / "cgroup.subtree_control").read_text().split() == ["pids"]
    assert {path.relative_to(root) for path in root.rglob("*")} == initial_paths
    assert sf_launch._rollback_leaf_pattern(root, supervisor, set()) == (True, "")
