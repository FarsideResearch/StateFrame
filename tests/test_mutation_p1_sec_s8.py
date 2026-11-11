"""Security regressions for mutation input snapshots."""
from __future__ import annotations

import errno

import kernel_mutate_inputs as inputs
import pytest
from kernel_mutate_policy import MutationPolicy


@pytest.mark.parametrize("failure_point", ("write", "fsync"))
def test_snapshot_enospc_preserves_original_error_and_removes_partial_root(
        tmp_path, monkeypatch, failure_point):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input.txt").write_bytes(b"payload")
    snapshot_root = tmp_path / "snapshot"
    failure = OSError(errno.ENOSPC, "snapshot disk full")

    if failure_point == "write":
        real_write = inputs.os.write
        writes = 0

        def fail_after_partial_write(fd, data):
            nonlocal writes
            writes += 1
            if writes == 1:
                return real_write(fd, data[:2])
            raise failure

        monkeypatch.setattr(inputs.os, "write", fail_after_partial_write)
    else:
        def fail_fsync(_fd):
            raise failure

        monkeypatch.setattr(inputs.os, "fsync", fail_fsync)

    with pytest.raises(OSError) as raised:
        inputs.admit_inputs(repo, ["input.txt"], MutationPolicy(level="prebuilt"),
                            snapshot_root=snapshot_root)

    assert raised.value is failure
    assert not snapshot_root.exists()


def test_snapshot_cleanup_failure_does_not_mask_write_error(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input.txt").write_bytes(b"payload")
    snapshot_root = tmp_path / "snapshot"
    failure = OSError(errno.ENOSPC, "snapshot disk full")

    def fail_write(_fd, _data):
        raise failure

    real_unlink = inputs.Path.unlink

    def fail_snapshot_unlink(path, *args, **kwargs):
        if path == snapshot_root / "in0":
            raise PermissionError(errno.EACCES, "cannot remove snapshot")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(inputs.os, "write", fail_write)
    monkeypatch.setattr(inputs.Path, "unlink", fail_snapshot_unlink)

    with pytest.raises(OSError) as raised:
        inputs.admit_inputs(repo, ["input.txt"], MutationPolicy(level="prebuilt"),
                            snapshot_root=snapshot_root)

    assert raised.value is failure
    assert any("unlink" in note for note in failure.__notes__)
    assert any("rmdir" in note for note in failure.__notes__)
