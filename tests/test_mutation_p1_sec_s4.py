"""Security regressions for mutation input admission."""

import os

import kernel_mutate_inputs
import pytest
from kernel_mutate_policy import MutationPolicy


def test_input_snapshot_refuses_same_size_in_place_edit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "input.txt"
    source.write_bytes(b"AAAA")
    original_read = os.read
    changed = False

    def read_during_edit(fd, count):
        nonlocal changed
        block = original_read(fd, min(count, 2))
        if block and not changed:
            changed = True
            with source.open("r+b") as writer:
                os.pwrite(writer.fileno(), b"BBBB", 0)
            prior_mtime = source.stat().st_mtime_ns
            os.utime(source, ns=(prior_mtime + 1_000_000_000,
                                 prior_mtime + 1_000_000_000))
        return block

    monkeypatch.setattr(kernel_mutate_inputs.os, "read", read_during_edit)
    snapshot_root = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="input changed during snapshot"):
        kernel_mutate_inputs.admit_inputs(
            repo, [source.name], MutationPolicy(level="authored"),
            snapshot_root=snapshot_root,
        )

    assert changed
    assert source.read_bytes() == b"BBBB"
    assert not snapshot_root.exists()
