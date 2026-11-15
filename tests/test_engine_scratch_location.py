"""Engine scratch directories must live under the bound/TMPDIR temp root, not the system /tmp."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import kernel_history_spool
import kernel_run_env
import kernel_safe_probe
from kernel_history_spool import HistorySpool


def test_history_spool_scratch_follows_bound_tempdir(tmp_path: Path, monkeypatch) -> None:
    bound = tmp_path / "bound-scratch"
    bound.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(bound))

    spool = HistorySpool()
    try:
        owned = Path(spool._owned_directory.name)
        assert owned.parent == bound
        assert owned.name.startswith("sf-history-run-")
    finally:
        spool._file.close()
        spool._owned_directory.cleanup()


def test_history_spool_scratch_follows_tmpdir(tmp_path: Path, monkeypatch) -> None:
    env_root = tmp_path / "env-tmp"
    env_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(env_root))

    spool = HistorySpool()
    try:
        assert Path(spool._owned_directory.name).parent == env_root
    finally:
        spool._file.close()
        spool._owned_directory.cleanup()


def test_run_env_provision_scratch_sits_beside_destination(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "run" / "env"
    destination.parent.mkdir(parents=True)
    seen: dict[str, Path] = {}

    def fake_probe(argv, scratch, **_kwargs):
        seen["scratch"] = Path(scratch)
        return SimpleNamespace(timed_out=False, exit_code=0, output_truncated=False, output=b"ok")

    monkeypatch.setattr(kernel_safe_probe, "run_probe", fake_probe)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "system-like"))
    (tmp_path / "system-like").mkdir()

    assert kernel_run_env._run(["true"], workspace=tmp_path, destination=destination) == "ok"
    assert seen["scratch"].parent == destination.parent
    assert seen["scratch"].name.startswith("stateframe-provision-")
