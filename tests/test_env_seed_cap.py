"""Bounded environment seeds preserve active clones and recover cleanly."""
import errno
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from stateframe import runner
from stateframe.engine import kernel_env_seed as seeds


def _source(root: Path, name: str, size: int = 512) -> Path:
    source = root / name
    site = source / "lib/python3.11/site-packages"
    info = site / "x-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: x\nVersion: 1.0\n")
    (info / "RECORD").write_text("x.py,,\n")
    (site / "x.py").write_bytes(name.encode().ljust(size, b"x"))
    (source / "bin").mkdir()
    (source / "bin/python").symlink_to(sys.executable)
    (source / "pyvenv.cfg").write_text("version = 3.11\n")
    return source


def _clone(root: Path, source: Path, name: str, events=None) -> Path:
    return seeds.clone(source, root / name, source / "bin/python", None, (), event=events)


def _size(seed: Path) -> int:
    return sum(row.get("size", 0) for row in json.loads((seed / "seed.json").read_text())["files"].values())


def _cap(monkeypatch, bytes_: int) -> None:
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", format(Decimal(bytes_) / Decimal(10**9), "f"))


def _setup(root: Path, monkeypatch, count: int = 3) -> tuple[list[Path], list[Path], Path]:
    monkeypatch.setenv("STATEFRAME_HOME", str(root))
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", "0")
    monkeypatch.setattr(seeds, "_link", lambda *_: False)
    sources = [_source(root, f"source-{index}") for index in range(count)]
    found = [_clone(root, source, f"run-{index}") for index, source in enumerate(sources[:2])]
    return sources, found, root / "env-seeds"


def test_least_recent_seed_is_evicted_and_reuse_changes_order(tmp_path, monkeypatch):
    sources, found, _ = _setup(tmp_path, monkeypatch)
    first, second = found
    first_sidecar = first.parent / f".{first.name}.bytes"
    assert first_sidecar.read_text() == str(_size(first))
    _cap(monkeypatch, 2 * max(map(_size, found)))
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    third = _clone(tmp_path, sources[2], "run-2")
    assert not first.exists() and not first_sidecar.exists() and second.exists() and third.exists()

    _clone(tmp_path, sources[1], "run-3")
    rebuilt = _clone(tmp_path, sources[0], "run-4")
    assert rebuilt == first and rebuilt.exists() and second.exists() and not third.exists()


def test_locked_seed_is_skipped(tmp_path, monkeypatch):
    sources, (first, second), store = _setup(tmp_path, monkeypatch)
    _cap(monkeypatch, 2 * max(_size(first), _size(second)))
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    code = ("import fcntl,sys; p=open(sys.argv[1], 'a+b'); "
            "fcntl.flock(p, fcntl.LOCK_EX); print('ready', flush=True); sys.stdin.read(1)")
    process = subprocess.Popen([sys.executable, "-c", code, str(store / f".{first.name}.lock")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        third = _clone(tmp_path, sources[2], "run-2")
        assert first.exists() and not second.exists() and third.exists()
    finally:
        process.stdin.write("x")
        process.stdin.flush()
        process.communicate(timeout=10)


def test_external_hardlink_seed_is_skipped(tmp_path, monkeypatch):
    sources, (first, second), _ = _setup(tmp_path, monkeypatch)
    _cap(monkeypatch, 2 * max(_size(first), _size(second)))
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    os.link(first / "lib/python3.11/site-packages/x.py", tmp_path / "external")
    third = _clone(tmp_path, sources[2], "run-2")
    assert first.exists() and not second.exists() and third.exists()


def test_quarantine_and_orphan_temporary_order(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    quarantine = store / "quarantine-old"
    quarantine.mkdir()
    (quarantine / "data").write_bytes(b"q" * 100)
    dead = store / f".{first.name}.tmp-99999999-1"
    dead.mkdir()
    (dead / "data").write_bytes(b"d" * 100)
    live = store / f".{first.name}.tmp-{os.getpid()}-2"
    live.mkdir()
    (live / "data").write_bytes(b"l" * 100)
    events = []
    total = _size(first) + _size(second) + 300
    seeds.evict_to_cap(store, total - 200, second.name, events.append)
    assert not quarantine.exists() and not dead.exists() and live.exists()
    assert first.exists() and second.exists()
    assert [event["reason"] for event in events] == ["quarantine", "orphan_tmp"]


def test_quarantine_with_external_hardlink_is_retained(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    quarantine = store / "quarantine-old"
    quarantine.mkdir()
    payload = quarantine / "data"
    payload.write_bytes(b"q" * 100)
    os.link(payload, tmp_path / "external")
    os.utime(first, ns=(1, 1))
    events = []
    seeds.evict_to_cap(store, _size(second), second.name, events.append)
    assert quarantine.exists() and not first.exists() and second.exists()
    assert (tmp_path / "external").read_bytes() == b"q" * 100


def test_permission_error_keeps_temporary_directory(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    temporary = store / f".{first.name}.tmp-99999999-1"
    temporary.mkdir()
    (temporary / "data").write_bytes(b"d" * 100)
    monkeypatch.setattr(seeds.os, "kill", lambda *_: (_ for _ in ()).throw(PermissionError(errno.EPERM, "denied")))
    seeds.evict_to_cap(store, _size(first) + _size(second), second.name, None)
    assert temporary.exists()


def test_leftover_eviction_target_is_cleaned(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    leftover = store / ".evicting-abandoned-1"
    leftover.mkdir()
    (leftover / "data").write_text("old")
    seeds.evict_to_cap(store, 10**9, second.name, None)
    assert not leftover.exists() and first.exists() and second.exists()


def test_unreadable_seed_manifest_uses_tree_size(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    (store / f".{first.name}.bytes").unlink()
    (first / "seed.json").write_text("{")
    assert seeds._seed_bytes(first) == seeds._tree_bytes(first)
    os.utime(first, ns=(1, 1))
    seeds.evict_to_cap(store, _size(second), second.name, None)
    assert not first.exists() and second.exists()


def test_cap_sizes_seed_store_without_reading_manifests(tmp_path, monkeypatch):
    sources, found, store = _setup(tmp_path, monkeypatch, count=12)
    found.extend(_clone(tmp_path, source, f"run-{index}")
                 for index, source in enumerate(sources[2:], start=2))
    assert all((store / f".{seed.name}.bytes").is_file() for seed in found)
    limit = sum(map(_size, found[:-1])) - 1
    original_read_text = Path.read_text

    def no_manifest_read(path, *args, **kwargs):
        if path.name == "seed.json":
            pytest.fail(f"eviction opened {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_manifest_read)
    result = seeds.evict_to_cap(store, limit, found[-1].name, None)
    assert result["count"] >= 1


def test_missing_sidecar_is_written_after_first_cap_pass(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    sidecar = store / f".{first.name}.bytes"
    sidecar.unlink()
    seeds.evict_to_cap(store, 10**9, second.name, None)
    assert sidecar.read_text() == str(_size(first))


def test_invalid_sidecar_falls_back_to_manifest_and_is_repaired(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    sidecar = store / f".{first.name}.bytes"
    sidecar.write_text("-1")
    expected = _size(first)
    original_read_text = Path.read_text
    reads = []

    def counted_read(path, *args, **kwargs):
        if path == first / "seed.json":
            reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read)
    seeds.evict_to_cap(store, 10**9, second.name, None)
    assert reads == [first / "seed.json"]
    assert sidecar.read_text() == str(expected)


def test_nonregular_sidecar_does_not_block_eviction(tmp_path, monkeypatch):
    _, (first, second), store = _setup(tmp_path, monkeypatch)
    sidecar = store / f".{first.name}.bytes"
    sidecar.unlink()
    os.mkfifo(sidecar)
    assert seeds._seed_bytes(first) == _size(first)
    seeds.evict_to_cap(store, 10**9, second.name, None)


def test_zero_is_unbounded_and_invalid_uses_default(tmp_path, monkeypatch):
    sources, found, _ = _setup(tmp_path, monkeypatch)
    third = _clone(tmp_path, sources[2], "run-2")
    assert all(seed.exists() for seed in [*found, third])
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", "invalid")
    events = []
    _clone(tmp_path, sources[0], "run-3", events.append)
    assert {event["event"] for event in events} == {"env_seed_cap_invalid"}
    assert all(seed.exists() for seed in [*found, third])


def test_concurrent_clones_leave_valid_seeds_below_soft_bound(tmp_path, monkeypatch):
    sources, (first, _), store = _setup(tmp_path, monkeypatch, count=5)
    cap = 2 * _size(first)
    _cap(monkeypatch, cap)
    code = ("import sys; from pathlib import Path; from stateframe.engine import kernel_env_seed as seeds; "
            "sys.stdin.read(1); seeds._link=lambda *_: False; source=Path(sys.argv[1]); "
            "seeds.clone(source, Path(sys.argv[2]), source/'bin/python', None, ())")
    processes = [subprocess.Popen([sys.executable, "-c", code, str(source), str(tmp_path / f"parallel-{index}")],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for index, source in enumerate(sources[1:])]
    for process in processes:
        process.stdin.write("x")
        process.stdin.flush()
    for process in processes:
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stdout + stderr
    remaining = [path for path in store.iterdir() if path.is_dir() and len(path.name) == 64]
    assert all(seeds._valid(seed, seed.name) for seed in remaining)
    assert not list(store.glob(".evicting-*"))
    assert sum(map(_size, remaining)) <= cap + max(map(_size, remaining))
    assert all((tmp_path / f"parallel-{index}/lib/python3.11/site-packages/x.py").is_file()
               for index in range(4))


def test_eviction_failure_does_not_fail_clone(tmp_path, monkeypatch):
    sources, (first, second), _ = _setup(tmp_path, monkeypatch)
    _cap(monkeypatch, 2 * max(_size(first), _size(second)))
    events = []
    monkeypatch.setattr(seeds.shutil, "rmtree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("blocked")))
    third = _clone(tmp_path, sources[2], "run-2", events.append)
    assert third.exists()
    assert any(event["event"] == "env_seed_evict_failed" for event in events)


@pytest.mark.parametrize("raw, expected", [("0", "0"), ("1.25", "1.25"), ("-1", "10"),
                                            ("nan", "10"), ("1e3", "10")])
def test_runner_forwards_validated_cap(monkeypatch, raw, expected):
    assert "STATEFRAME_ENV_SEED_MAX_GB" in runner.OPT_IN_SWITCHES
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", raw)
    assert runner.resolve_opt_in_switches()["STATEFRAME_ENV_SEED_MAX_GB"] == expected
