"""bwrap mounts: an identity --ro-bind already shown by an identity --ro-bind ancestor is not mounted again."""
from dataclasses import replace
from pathlib import Path

import pytest

from _portable import TRUE

from kernel_exec_capability import shell_capability
from kernel_test_isolation import _mount_points_from, _uncovered_mounts, capability_command

RO, RW = "--ro-bind", "--bind"
ROOT_ONLY = frozenset({Path("/")})


def test_children_of_a_read_only_ancestor_are_pruned_transitively():
    mounts = {Path("/usr"): RO, Path("/usr/lib"): RO, Path("/usr/lib/python3/os.py"): RO,
              Path("/usr/bin/python3.11"): RO, Path("/opt/tool"): RO}
    assert set(_uncovered_mounts(mounts, ROOT_ONLY)) == {Path("/usr"), Path("/opt/tool")}


def test_read_only_child_of_a_writable_bind_is_kept():
    # The child narrows the writable parent to read-only; dropping it would widen authority.
    mounts = {Path("/w"): RW, Path("/w/ro"): RO, Path("/w/ro/deeper"): RO}
    assert set(_uncovered_mounts(mounts, ROOT_ONLY)) == {Path("/w"), Path("/w/ro")}


def test_writable_child_of_a_read_only_bind_is_kept():
    mounts = {Path("/data"): RO, Path("/data/scratch"): RW}
    assert set(_uncovered_mounts(mounts, ROOT_ONLY)) == {Path("/data"), Path("/data/scratch")}


def test_nearest_mounted_ancestor_decides_not_any_ancestor():
    # /a is read-only, /a/w is writable, /a/w/r is read-only again: /a/w/r narrows /a/w and stays.
    mounts = {Path("/a"): RO, Path("/a/w"): RW, Path("/a/w/r"): RO, Path("/a/x"): RO}
    assert set(_uncovered_mounts(mounts, ROOT_ONLY)) == {Path("/a"), Path("/a/w"), Path("/a/w/r")}


@pytest.mark.parametrize("special", ["/dev", "/proc"])
def test_paths_under_bwrap_dev_and_proc_are_kept(special):
    # bwrap mounts its own /dev and /proc first; a bind under them is not covered by any host bind.
    child = Path(special) / "thing"
    assert _uncovered_mounts({child: RO}, ROOT_ONLY) == [child]
    assert set(_uncovered_mounts({Path(special): RO, child: RO}, ROOT_ONLY)) == {Path(special)}


def test_unrelated_and_sibling_paths_are_kept():
    mounts = {Path("/usr/bin"): RO, Path("/usr/binx"): RO, Path("/usr/lib"): RO}
    assert set(_uncovered_mounts(mounts, ROOT_ONLY)) == set(mounts)


def bwrap_binds(command):
    head = command[:command.index("--")]
    return [(head[i], head[i + 1]) for i, value in enumerate(head) if value in (RO, RW) and head[i + 1] == head[i + 2]]


def test_builder_emits_each_host_tree_once_and_keeps_every_root_visible(tmp_path, monkeypatch):
    monkeypatch.setattr("kernel_test_isolation._bwrap_userns_guard_supported", lambda path: True)
    monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr("kernel_test_isolation._host_mount_points", lambda: ROOT_ONLY)
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir()
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "tool").write_text("#!/bin/sh\n")
    (tree / "lib").mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=False,
                           extra_read_roots=(tree, tree / "bin" / "tool", tree / "lib"))
    command, _ = capability_command([TRUE], cap, platform_backend="bwrap")
    binds = bwrap_binds(command)
    mounted = [Path(path) for _, path in binds]
    assert len(mounted) == len(set(mounted))
    resolved_tree = tree.resolve()
    assert (RO, str(resolved_tree)) in binds
    assert not any(resolved_tree in path.parents for path in mounted)
    # Every read root is still visible: it is mounted, or under a mounted read-only ancestor.
    for root in cap.read_roots:
        target = root.resolve()
        assert target in mounted or any(parent in mounted for parent in target.parents), root
    # Writable roots keep their own writable mount.
    for root in cap.write_roots:
        assert (RW, str(root)) in binds


def test_child_at_or_below_a_host_mount_boundary_is_kept():
    # A recursive bind omits unbindable submounts, so a child across a mount boundary keeps its own bind.
    mounts = {Path("/usr"): RO, Path("/usr/bin"): RO, Path("/usr/bin/python3"): RO,
              Path("/usr/lib/x"): RO, Path("/usr/share/y/z"): RO}
    points = ROOT_ONLY | {Path("/usr/bin"), Path("/usr/share")}
    assert set(_uncovered_mounts(mounts, points)) == {
        Path("/usr"), Path("/usr/bin"), Path("/usr/share/y/z")}
    # The ancestor being a mount point itself is no boundary below it.
    assert set(_uncovered_mounts({Path("/usr"): RO, Path("/usr/lib/x"): RO}, ROOT_ONLY | {Path("/usr")})) == {
        Path("/usr")}


def test_unreadable_mount_table_prunes_nothing():
    mounts = {Path("/usr"): RO, Path("/usr/lib/x"): RO}
    assert _uncovered_mounts(mounts, None) == list(mounts)


def test_builder_without_mount_table_binds_every_root(tmp_path, monkeypatch):
    monkeypatch.setattr("kernel_test_isolation._bwrap_userns_guard_supported", lambda path: True)
    monkeypatch.setattr("kernel_test_isolation.shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr("kernel_test_isolation._host_mount_points", lambda: None)
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir()
    # The capability itself collapses plain paths under a directory root; a symlinked root survives,
    # and its resolved target is what bwrap would bind (the common case behind hundreds of binds).
    tree = tmp_path / "tree"
    (tree / "lib").mkdir(parents=True)
    (tree / "lib" / "tool").write_text("#!/bin/sh\n")
    (tree / "alias").symlink_to(tree / "lib" / "tool")
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=False,
                           extra_read_roots=(tree, tree / "alias"))
    command, _ = capability_command([TRUE], cap, platform_backend="bwrap")
    mounted = [Path(path) for _, path in bwrap_binds(command)]
    assert tree.resolve() / "lib" / "tool" in mounted and tree.resolve() in mounted
    monkeypatch.setattr("kernel_test_isolation._host_mount_points", lambda: ROOT_ONLY)
    command, _ = capability_command([TRUE], cap, platform_backend="bwrap")
    pruned = [Path(path) for _, path in bwrap_binds(command)]
    assert tree.resolve() in pruned and tree.resolve() / "lib" / "tool" not in pruned


def test_mountinfo_parsing_unescapes_and_rejects_malformed_lines():
    text = ("22 1 0:21 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
            "35 22 0:30 / /mnt/with\\040space rw - tmpfs tmpfs rw\n"
            "36 22 0:31 / /tab\\011and\\134slash rw - tmpfs tmpfs rw\n")
    assert _mount_points_from(text) == frozenset({Path("/"), Path("/mnt/with space"), Path("/tab\tand\\slash")})
    assert _mount_points_from("22 1 0:21 /\n") is None
    assert _mount_points_from("") == frozenset()
