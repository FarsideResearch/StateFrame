"""GC plans are read-only by default, lock-aware, idempotent and seed-reference safe."""
from __future__ import annotations

import json
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from stateframe import storage_gc
from stateframe.resume import hold_run_lock


def _finished_run(root, name="finished", *, result="SUBMITTED"):
    run = root / name
    (run / "env").mkdir(parents=True)
    (run / "env/data.bin").write_bytes(b"environment" * 100)
    (run / "env/pyvenv.cfg").write_text(
        f"implementation = {sys.implementation.name}\n"
        f"version = {'.'.join(map(str, sys.version_info[:3]))}\n")
    (run / "checkpoints/0001/run").mkdir(parents=True)
    (run / "checkpoints/0001/run/checkpoint.bin").write_bytes(b"checkpoint" * 100)
    (run / "worker/measurement-runs/one/blobs").mkdir(parents=True)
    (run / "worker/measurement-runs/one/events.jsonl").write_bytes(b'{"event":"finished"}\n' * 600)
    for index in range(10):
        digest = f"{index:064x}"
        (run / "worker/measurement-runs/one/blobs" / f"{digest}.zlib").write_bytes(
            (f"blob-{index}-" * 20).encode())
    (run / "requests").mkdir()
    for index in range(1, 6):
        (run / f"requests/{index:04}.json").write_bytes(
            b'{"turn":1,"padding":"' + b"a" * 1000 + b'"}\n')
    (run / "env-sites.json").write_text("[]")
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": result}}))
    (run / "spec.json").write_text(json.dumps({"schema": "stateframe-run/1"}))
    return run


def test_gc_dry_run_reports_artifacts_without_mutating_and_apply_is_idempotent(tmp_path):
    runs = tmp_path / "runs"
    run = _finished_run(runs)

    dry = storage_gc.run_gc([runs], older_than_days=0, now=time.time() + 1)
    entry = next(item for item in dry["runs"] if item["run"] == str(run))
    assert entry["reclaimable"]["env"] > 0
    assert entry["reclaimable"]["checkpoints"] > 0
    assert entry["reclaimable"]["measurement_journals"] > 0
    assert entry["reclaimable"]["blobs"] > 0
    assert entry["reclaimable"]["requests"] > 0
    assert (run / "env/data.bin").exists()
    assert (run / "requests/0001.json").exists()
    assert (run / "worker/measurement-runs/one/events.jsonl").exists()

    applied = storage_gc.run_gc([runs], apply=True, older_than_days=0, now=time.time() + 1)
    assert (run / "env-manifest.json").is_file()
    assert not (run / "env").exists()
    assert not (run / "checkpoints").exists()
    assert (run / "worker/measurement-runs/one/events.jsonl.xz").is_file()
    assert (run / "worker/measurement-runs/one/blobs/pack-0001.sfpk").is_file()
    assert (run / "requests.pack.xz").is_file()

    repeated = storage_gc.run_gc([runs], older_than_days=0, now=time.time() + 1)
    assert all(value == 0 for value in repeated["totals"].values())
    assert applied["totals"]["env"] > 0


def test_measurement_pack_is_attempted_even_if_journal_seal_fails(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "worker/measurement-runs/one"
    directory.mkdir(parents=True)
    calls = []

    class Measurement:
        @staticmethod
        def seal_measurement_run(_directory):
            calls.append("seal")
            raise OSError("injected journal failure")

        @staticmethod
        def pack_blobs(_directory):
            calls.append("pack")

    monkeypatch.setattr(storage_gc, "_measurement_api", lambda: Measurement)

    storage_gc.seal_measurements(tmp_path)

    assert calls == ["seal", "pack"]
    assert "warning:" in capsys.readouterr().out


def test_acceptance_reader_opens_finished_measurement_journal(tmp_path):
    from acceptance._support import read_measurement_events

    run = tmp_path / "run"
    journal = run / "worker/measurement-runs/one/events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"kind":"saved"}\n')

    storage_gc.seal_measurements(run, printer=lambda _line: None)

    assert not journal.exists()
    assert read_measurement_events(run) == ({"kind": "saved"},)


def test_gc_skips_live_run_lock_and_unfinished_by_default(tmp_path):
    runs = tmp_path / "runs"
    run = _finished_run(runs, "locked")
    unfinished = _finished_run(runs, "unfinished", result="INTERRUPTED")

    with hold_run_lock(run):
        plan = storage_gc.run_gc([runs], older_than_days=0, include_unfinished=False, now=time.time() + 1)

    statuses = {item["run"]: item["status"] for item in plan["runs"]}
    assert statuses[str(run)] == "locked"
    assert statuses[str(unfinished)] == "unfinished"
    assert (run / "env/data.bin").exists()
    assert (unfinished / "env/data.bin").exists()


def test_gc_age_threshold_and_keep_last_apply_per_root(tmp_path):
    runs = tmp_path / "runs"
    old = _finished_run(runs, "old")
    middle = _finished_run(runs, "middle")
    newest = _finished_run(runs, "newest")
    now = time.time()
    for run, age_days in ((old, 10), (middle, 3), (newest, 1)):
        timestamp = now - age_days * 86400
        os.utime(run / "engine-result.json", (timestamp, timestamp))

    kept = storage_gc.run_gc([runs], older_than_days=2, keep_last=1, now=now,
                             printer=lambda _line: None)
    statuses = {Path(row["run"]).name: row["status"] for row in kept["runs"]}
    assert statuses == {"old": "eligible", "middle": "eligible", "newest": "kept_last"}

    aged = storage_gc.run_gc([runs], older_than_days=2, now=now, printer=lambda _line: None)
    statuses = {Path(row["run"]).name: row["status"] for row in aged["runs"]}
    assert statuses == {"old": "eligible", "middle": "eligible", "newest": "too_new"}


def test_gc_dry_run_survives_malformed_blob_index(tmp_path):
    runs = tmp_path / "runs"
    run = _finished_run(runs)
    (run / "worker/measurement-runs/one/blobs/pack-0001.idx").write_text(
        json.dumps({"pack_size": 0, "records": None}))

    result = storage_gc.run_gc([runs], older_than_days=0, now=time.time() + 1,
                               printer=lambda _line: None)

    assert result["runs"][0]["status"] == "eligible"


def test_gc_skips_live_engine_lock_and_refuses_symlinked_lock(tmp_path):
    runs = tmp_path / "runs"
    engine_locked = _finished_run(runs, "engine-locked")
    with (engine_locked / ".engine.lock").open("a+") as engine_lock:
        fcntl.flock(engine_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = storage_gc.run_gc([runs], older_than_days=0, now=time.time() + 1)
        statuses = {item["run"]: item["status"] for item in plan["runs"]}
        assert statuses[str(engine_locked)] == "locked"
        fcntl.flock(engine_lock, fcntl.LOCK_UN)

    symlinked = _finished_run(runs, "symlinked-lock")
    target = tmp_path / "lock-target"
    target.write_text("not a lock")
    (symlinked / ".run.lock").symlink_to(target)
    result = storage_gc.run_gc([symlinked], apply=True, older_than_days=0, now=time.time() + 1)

    assert result["runs"][0]["status"] == "locked"
    assert (symlinked / "env/data.bin").exists()
    assert target.read_text() == "not a lock"


def test_gc_include_unfinished_warns_resume_will_be_lost(tmp_path, capsys):
    runs = tmp_path / "runs"
    run = _finished_run(runs, "interrupted", result="INTERRUPTED")

    storage_gc.run_gc([runs], apply=True, older_than_days=0, include_unfinished=True, now=time.time() + 1)

    assert "resume will be lost" in capsys.readouterr().out.lower()
    assert not (run / "env").exists()
    assert not (run / "checkpoints").exists()


def test_gc_retains_referenced_seed_and_removes_unreferenced_and_old_quarantine(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = _finished_run(runs, "seeded", result="INTERRUPTED")
    home = tmp_path / "home"
    seeds = home / "env-seeds"
    referenced = seeds / "referenced"
    unused = seeds / "unused"
    quarantine = seeds / "quarantine-1-dead"
    for seed in (referenced, unused, quarantine):
        seed.mkdir(parents=True)
        (seed / "read-only").write_text(seed.name)
        (seed / "read-only").chmod(0o444)
    (run / "env-seed.json").write_text(json.dumps({"path": str(referenced)}))
    (run / "checkpoints/0001/run/env-seed.json").write_text(json.dumps({"path": str(referenced)}))
    old = time.time() - 10 * 86400
    for seed in (referenced, unused, quarantine):
        os.utime(seed, (old, old))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    storage_gc.run_gc([runs], apply=True, older_than_days=1, include_unfinished=False,
                      prune_seeds=True, now=time.time())

    assert referenced.exists()
    assert not unused.exists()
    assert not quarantine.exists()


def test_gc_does_not_chmod_or_remove_seed_with_hardlinks_outside_scanned_roots(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _finished_run(runs)
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-shared"
    seed.mkdir(parents=True)
    seeded_file = seed / "read-only"
    seeded_file.write_text("shared")
    seeded_file.chmod(0o444)
    outside_env = tmp_path / "outside-run-env"
    os.link(seeded_file, outside_env)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    dry = storage_gc.run_gc([runs], older_than_days=0, now=time.time() + 1)

    assert str(seed) not in dry["seeds"]["paths"]
    assert seeded_file.stat().st_mode & 0o222 == 0
    assert outside_env.stat().st_mode & 0o222 == 0

    storage_gc.run_gc([runs], apply=True, older_than_days=0, prune_seeds=True, now=time.time() + 1)

    assert seed.exists()
    assert seeded_file.stat().st_mode & 0o222 == 0
    assert outside_env.stat().st_mode & 0o222 == 0


def test_gc_dry_run_projects_seeds_released_by_finished_run_retention(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = _finished_run(runs, "seeded")
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-a"
    seed.mkdir(parents=True)
    (seed / "data").write_text("seed")
    (run / "env-seed.json").write_text(json.dumps({"path": str(seed)}))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    dry = storage_gc.run_gc([runs], older_than_days=0, prune_seeds=True, now=time.time() + 1)
    assert dry["seeds"]["paths"] == [str(seed)]
    assert seed.exists()

    storage_gc.run_gc([runs], apply=True, older_than_days=0, prune_seeds=True, now=time.time() + 1)
    assert not seed.exists()


def test_gc_keeps_seed_if_checkpoint_cleanup_fails(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = _finished_run(runs, "seeded")
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-a"
    seed.mkdir(parents=True)
    (seed / "data").write_text("seed")
    (run / "env-seed.json").write_text(json.dumps({"path": str(seed)}))
    (run / "checkpoints/0001/run/env-seed.json").write_text(json.dumps({"path": str(seed)}))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    def fail(_run):
        raise OSError("injected checkpoint cleanup failure")

    monkeypatch.setattr(storage_gc, "remove_checkpoints", fail)
    storage_gc.run_gc([runs], apply=True, older_than_days=0, prune_seeds=True, now=time.time() + 1,
                      printer=lambda _line: None)

    assert seed.exists()
    assert (run / "checkpoints/0001/run/env-seed.json").is_file()


def test_gc_cli_prints_json_inventory_by_default_dry_run(tmp_path, capsys):
    from stateframe.cli import main

    runs = tmp_path / "runs"
    run = _finished_run(runs)

    assert main(["gc", str(runs), "--older-than", "0", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["runs"][0]["run"] == str(run)
    assert result["runs"][0]["status"] == "eligible"
    assert result["totals"]["total"] > 0
    assert (run / "env/data.bin").exists()


def test_gc_uses_runner_runs_home_when_no_root_is_passed(tmp_path, monkeypatch):
    runs = tmp_path / "default-runs"
    run = _finished_run(runs)
    monkeypatch.setattr(storage_gc, "default_runs_home", lambda: runs)

    result = storage_gc.run_gc(older_than_days=0, now=time.time() + 1,
                               printer=lambda _line: None)

    assert result["roots"] == [str(runs)]
    assert result["runs"][0]["run"] == str(run)


def _cap_store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    store = home / "env-seeds"
    store.mkdir(parents=True)
    seeds = []
    for index in range(3):
        seed = store / f"{index:064x}"
        seed.mkdir()
        (seed / "payload").write_bytes(b"x" * 400)
        os.utime(seed, ns=(index + 1, index + 1))
        seeds.append(seed)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", "0.0000008")
    runs = tmp_path / "explicit-runs"
    runs.mkdir()
    return runs, store, seeds


def test_gc_cap_eviction_after_pruning_applies_to_explicit_roots(tmp_path, monkeypatch):
    runs, store, seeds = _cap_store(tmp_path, monkeypatch)
    sidecar = store / f".{seeds[0].name}.bytes"
    sidecar.write_text("400")
    result = storage_gc.run_gc([runs], apply=True, printer=lambda _line: None)
    assert result["seed_cap"]["count"] == 1
    assert result["seed_cap"]["bytes"] == 400
    assert result["seed_cap"]["paths"] == [str(seeds[0])]
    assert not seeds[0].exists() and not sidecar.exists()
    assert all(seed.exists() for seed in seeds[1:])
    assert sum(storage_gc._logical_tree_bytes(seed) for seed in store.iterdir() if seed.is_dir()) <= 800


def test_gc_cap_can_evict_seed_referenced_by_finished_run(tmp_path, monkeypatch):
    runs, _, seeds = _cap_store(tmp_path, monkeypatch)
    run = _finished_run(runs)
    (run / "env-seed.json").write_text(json.dumps({"path": str(seeds[0])}))
    result = storage_gc.run_gc([runs], apply=True, printer=lambda _line: None)
    assert result["seed_cap"]["paths"] == [str(seeds[0])]
    assert not seeds[0].exists() and (run / "env-seed.json").is_file()


def test_gc_cap_uses_sidecars_without_opening_manifests(tmp_path, monkeypatch):
    runs, store, seeds = _cap_store(tmp_path, monkeypatch)
    for seed in seeds:
        (seed / "seed.json").write_text("unreadable")
        (store / f".{seed.name}.bytes").write_text("400")
    original_read_text = Path.read_text

    def no_manifest_read(path, *args, **kwargs):
        if path.name == "seed.json":
            raise AssertionError(f"gc opened {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_manifest_read)
    result = storage_gc.run_gc([runs], apply=True, printer=lambda _line: None)
    assert result["seed_cap"]["bytes"] == 400
    assert not seeds[0].exists()


def test_gc_age_pruning_removes_seed_sidecar(tmp_path, monkeypatch):
    runs, store, seeds = _cap_store(tmp_path, monkeypatch)
    sidecar = store / f".{seeds[0].name}.bytes"
    sidecar.write_text("400")
    result = storage_gc.run_gc([runs], apply=True, older_than_days=1,
                               prune_seeds=True, printer=lambda _line: None)
    assert result["seeds"]["count"] == 3
    assert not seeds[0].exists() and not sidecar.exists()


def test_gc_zero_cap_keeps_seeds(tmp_path, monkeypatch):
    runs, _, seeds = _cap_store(tmp_path, monkeypatch)
    monkeypatch.setenv("STATEFRAME_ENV_SEED_MAX_GB", "0")
    result = storage_gc.run_gc([runs], apply=True, printer=lambda _line: None)
    assert result["seed_cap"] == {"count": 0, "bytes": 0, "paths": []}
    assert all(seed.exists() for seed in seeds)


def test_gc_cap_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    runs, store, seeds = _cap_store(tmp_path, monkeypatch)
    result = storage_gc.run_gc([runs], printer=lambda _line: None)
    assert result["seed_cap"] == {"count": 1, "bytes": 400, "paths": [str(seeds[0])]}
    assert all(seed.exists() for seed in seeds)
    assert not list(store.glob(".*.lock"))


def test_gc_cap_dry_run_accounts_for_age_pruning_first(tmp_path, monkeypatch):
    runs, _, seeds = _cap_store(tmp_path, monkeypatch)
    result = storage_gc.run_gc([runs], older_than_days=1, prune_seeds=True,
                               printer=lambda _line: None)
    assert result["seeds"]["count"] == 3
    assert result["seed_cap"]["count"] == 0
    assert all(seed.exists() for seed in seeds)


def test_gc_cap_skips_in_use_seed(tmp_path, monkeypatch):
    runs, store, seeds = _cap_store(tmp_path, monkeypatch)
    code = ("import fcntl,sys; lock=open(sys.argv[1], 'a+b'); "
            "fcntl.flock(lock, fcntl.LOCK_EX); print('ready', flush=True); sys.stdin.read(1)")
    process = subprocess.Popen([sys.executable, "-c", code, str(store / f".{seeds[0].name}.lock")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        result = storage_gc.run_gc([runs], apply=True, printer=lambda _line: None)
        assert seeds[0].exists() and not seeds[1].exists() and seeds[2].exists()
        assert result["seed_cap"]["paths"] == [str(seeds[1])]
    finally:
        process.stdin.write("x")
        process.stdin.flush()
        process.communicate(timeout=10)


def test_gc_of_explicit_roots_never_prunes_seeds_other_runs_may_use(tmp_path, monkeypatch, capsys):
    """Seeds are shared across every run directory; a gc scoped to some roots cannot see them all."""
    runs = tmp_path / "runs"
    _finished_run(runs)
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-elsewhere"
    seed.mkdir(parents=True)
    (seed / "data").write_text("seed")
    old = time.time() - 30 * 86400
    os.utime(seed, (old, old))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    result = storage_gc.run_gc([runs], apply=True, older_than_days=0, now=time.time() + 1)

    assert seed.exists()
    assert result["seeds"]["count"] == 0
    assert "--prune-seeds" in capsys.readouterr().out


def test_gc_keeps_recently_used_unreferenced_seed(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _finished_run(runs)
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-fresh"
    seed.mkdir(parents=True)
    (seed / "data").write_text("seed")
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    storage_gc.run_gc([runs], apply=True, older_than_days=1, prune_seeds=True, now=time.time())

    assert seed.exists()


def test_clone_marks_the_seed_as_recently_used(tmp_path, monkeypatch):
    from stateframe.engine import kernel_env_seed
    from test_env_seed_store import _source

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    seed = kernel_env_seed.clone(source, tmp_path / "first", source / "bin/python", None, ())
    old = time.time() - 30 * 86400
    os.utime(seed, (old, old))

    kernel_env_seed.clone(source, tmp_path / "second", source / "bin/python", None, ())

    assert seed.stat().st_mtime > old + 86400


def test_record_payload_digest_ignores_installer_artifacts_but_not_payload(tmp_path):
    """pip's own rows (.pyc, INSTALLER, REQUESTED, launcher scripts, order) differ per install."""
    first = tmp_path / "first.RECORD"
    first.write_text("pkg.py,sha256=AAA,10\npkg-1.0.dist-info/INSTALLER,sha256=I1,4\n"
                     "../../../bin/tool,sha256=S1,90\npkg-1.0.dist-info/RECORD,,\n")
    reinstalled = tmp_path / "second.RECORD"
    reinstalled.write_text("__pycache__/pkg.cpython-314.pyc,,\npkg-1.0.dist-info/RECORD,,\n"
                           "../../../bin/tool,sha256=S2,95\npkg.py,sha256=AAA,10\n"
                           "pkg-1.0.dist-info/INSTALLER,sha256=I2,4\n")
    changed = tmp_path / "third.RECORD"
    changed.write_text("pkg.py,sha256=BBB,10\n")

    assert storage_gc.record_payload_sha256(first) == storage_gc.record_payload_sha256(reinstalled)
    assert storage_gc.record_payload_sha256(first) != storage_gc.record_payload_sha256(changed)


def _freeze(directory: Path) -> None:
    (directory / "pkg").mkdir(parents=True)
    (directory / "pkg/__init__.py").write_text("")
    for path in (directory / "pkg/__init__.py", directory / "pkg", directory):
        path.chmod(0o555 if path.is_dir() else 0o444)


def test_finished_run_cleanup_removes_read_only_checkpoint_and_env_trees(tmp_path):
    # Checkpoints hold candidate workspaces the engine leaves read-only; removal must not stop on them.
    from stateframe import runner
    run = _finished_run(tmp_path)
    _freeze(run / "checkpoints/0001/run/workspace")
    _freeze(run / "env/lib")
    seed_file = tmp_path / "seed-file"
    seed_file.write_text("shared")
    seed_file.chmod(0o444)
    os.link(seed_file, run / "env/shared")
    assert runner.clean_finished_checkpoints(run)
    assert storage_gc.remove_run_env(run, require_finished=False)
    assert not (run / "checkpoints").exists() and not (run / "env").exists()
    assert seed_file.stat().st_mode & 0o777 == 0o444  # a hard-linked seed file keeps its mode
    other = _finished_run(tmp_path, "other")
    _freeze(other / "checkpoints/0001/run/workspace")
    assert storage_gc.remove_checkpoints(other) and not (other / "checkpoints").exists()
