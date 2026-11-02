"""Linux command sandboxes emit only the bind mounts that change what the command can see.

Capability read roots list every executable on PATH. Binding each one beneath an already read-only-bound
/usr cost ~425 mounts and 1.5-3 s per sandboxed command on Linux 7.0, long enough that short command
limits expired before the command ran. A redundant read-only bind is dropped; the effective view is not.
"""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import kernel_test_isolation as isolation
from kernel_exec_capability import ExecCapability, ExecLimits, shell_capability


def _binds(argv: list[str]) -> list[tuple[str, str]]:
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 2)
            if argv[i] in ("--ro-bind", "--bind") and argv[i + 1] == argv[i + 2]]


def _command(monkeypatch, cap: ExecCapability) -> list[str]:
    monkeypatch.setattr(isolation, "sandbox_executable", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(isolation, "_bwrap_userns_guard_supported", lambda executable: True)
    argv, _env = isolation.capability_command(["/bin/true"], cap, platform_backend="bwrap", shared_namespace=True)
    return argv


def test_redundant_read_only_binds_are_dropped_and_carve_outs_kept(tmp_path, monkeypatch):
    system = tmp_path / "system"
    tool = system / "bin" / "tool"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/true\n")
    # Merged-/usr shape: a PATH entry reached through a directory symlink (/bin -> usr/bin). The
    # capability keeps such roots (their parent is not canonical), and each resolves under /usr.
    linked = tmp_path / "bin"
    linked.symlink_to(system / "bin")
    workspace = tmp_path / "workspace"
    locked = workspace / "locked"
    locked.mkdir(parents=True)
    writable = system / "cache"          # a writable root beneath a read-only root
    writable.mkdir()
    cap = ExecCapability(purpose="shell", workspace=workspace, workspace_writable=True,
                         read_roots=(system, linked / "tool", locked),
                         write_roots=(writable,), network="none", env={"PATH": str(linked)},
                         limits=ExecLimits())
    assert any(str(root).endswith("bin/tool") for root in cap.read_roots), cap.read_roots
    binds = _binds(_command(monkeypatch, cap))
    paths = {path: mode for mode, path in binds}
    assert paths[str(system.resolve())] == "--ro-bind"
    # Covered by the read-only system bind: no separate mount.
    assert str(tool.resolve()) not in paths, binds
    # Writable roots and read-only carve-outs beneath them change the view: kept.
    assert paths[str(writable.resolve())] == "--bind"
    assert paths[str(workspace.resolve())] == "--bind"
    assert paths[str(locked.resolve())] == "--ro-bind"
    assert len(binds) == len(set(binds))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux bind mounts")
def test_real_shell_capability_emits_a_small_bounded_mount_set(tmp_path, monkeypatch):
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    binds = _binds(_command(monkeypatch, replace(cap, sandboxed=True)))
    # Before the fix this was ~425 (one per PATH executable); the read roots collapse to their
    # covering directories.
    assert len(binds) <= 40, len(binds)
    covered = [path for mode, path in binds if mode == "--ro-bind" and any(
        Path(path).is_relative_to(other) and path != other
        for other_mode, other in binds if other_mode == "--ro-bind")]
    writable_between = {path for mode, path in binds if mode == "--bind"}
    assert not [path for path in covered if not any(
        Path(path).is_relative_to(w) for w in writable_between)], covered
