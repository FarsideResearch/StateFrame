"""Measure the whole five-send run at quiescent barriers, including the live checkpoint."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import _stateframe_home
import test_checkpoint_b_resume_fidelity as resume_proof
from _timeouts import scaled
from fixtures.process_signals import wait_for_checkpoint_saved
from stateframe import cli, runner, storage_gc, workcopy
from stateframe.engine import kernel_env_seed
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 100_000_000
DISK_FIXTURES = [pytest.param(False, id="library")]
if os.environ.get("SF_TEST_DISK_LOOKUP_TABLES") == "1":
    DISK_FIXTURES.append(pytest.param(True, id="tables"))


def _gate(samples):
    over = {stage: report for stage, report in samples.items()
            if report["run"]["attributable_bytes"] > BUDGET}
    assert not over, f"run allocation exceeds {BUDGET}: {json.dumps(over, sort_keys=True)}"


def _write_tables(repo):
    # 128 importable modules, each with 512 distinct fixed-width rows.
    tables = repo / "tables"
    tables.mkdir()
    rng = random.Random(17)
    for index in range(128):
        rows = [b"    b'" + rng.randbytes(60).hex().encode() + b"',\n" for _ in range(512)]
        (tables / f"table_{index:03}.py").write_bytes(
            b"ROWS = (\n" + b"".join(rows) + b")\n")


def _fixture(repo, *, tables=False):
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    if tables:
        _write_tables(repo)
    # A project venv is discovered by the stock CLI. Use the bundled pure Python wheels so
    # neither pip's cache nor the developer's installed packages affect the fixture size.
    venv = repo / ".venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(venv)], check=True)
    site, = venv.glob("lib/python*/site-packages")
    for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments"):
        wheel, = (ROOT / "wheelhouse").glob(f"{name}-*-py3-none-any.whl")
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)
            if name == "packaging" and not tables:
                for member in archive.namelist():
                    if member.startswith("packaging/") and member.endswith(".py"):
                        archive.extract(member, repo)


def _document():
    doc = json.loads((ROOT / "examples/discount_bug.script.json").read_bytes())
    doc["steps"].insert(0, {"tool": "read", "required": True,
                            "arguments": {"path": "tests/test_pricing.py"}})
    return doc


def _run_args(repo, run, url):
    return ["run", "--repo", str(repo), "--run-dir", str(run), "--model", "scripted",
            "--endpoint", url, "--task", "Fix the pricing code so the tests pass.", "--quiet"]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("private_copy", [
    pytest.param(False, id="default"),
    pytest.param(True, id="copy", marks=pytest.mark.skipif(
        sys.platform != "linux", reason="injects the Linux seed copy fallback")),
])
@pytest.mark.parametrize("tables", DISK_FIXTURES)
def test_five_send_allocation(tmp_path, monkeypatch, private_copy, tables):
    repo, run = tmp_path / "repo", tmp_path / "run"
    _fixture(repo, tables=tables)
    for key in tuple(os.environ):
        if key.startswith("STATEFRAME_"):
            monkeypatch.delenv(key)
    # Each measurement starts with a cold seed store.
    monkeypatch.setattr(_stateframe_home, "_fallback", None)
    home = _stateframe_home.isolated_stateframe_home()
    monkeypatch.setenv("STATEFRAME_HOME", home)
    seeds = Path(home) / "env-seeds"
    if private_copy:
        def unsupported(*_args):
            raise OSError(errno.EOPNOTSUPP, "fixture without reflinks")

        monkeypatch.setattr(workcopy, "_reflink", unsupported)
        monkeypatch.setattr(kernel_env_seed, "_reflink", lambda *_: False)
        monkeypatch.setattr(kernel_env_seed, "_link", lambda *_: False)

    samples = {}

    def sample(label):
        samples[label] = storage_gc.run_allocation(run, seed_root=seeds)
        print(label, json.dumps(samples[label], sort_keys=True), flush=True)

    execute = runner._execute

    def after_prepare(*args, **kwargs):
        seed = Path(json.loads((run / "env-seed.json").read_bytes())["path"])
        assert seed.parent == seeds
        if private_copy:
            package, = (run / "env").glob("lib/python*/site-packages/packaging/version.py")
            assert not os.path.samestat(package.stat(), (seed / package.relative_to(run / "env")).stat())
        for tree in ("workspace", "parent"):
            sources = list((run / tree / ("tables" if tables else "packaging")).rglob("*.py"))
            assert len(sources) == (128 if tables else 17)
            assert sum(path.stat().st_size for path in sources) == (8455552 if tables else 259868)
        sample("prepared")
        return execute(*args, **kwargs)

    monkeypatch.setattr(runner, "_execute", after_prepare)
    script = Script(_document())
    next_call = script.next_call
    checkpoint_errors = []

    def after_checkpoint(offered):
        completed = script.position
        if 1 <= completed <= 4:
            try:
                wait_for_checkpoint_saved(run, deadline=time.monotonic() + scaled(90),
                                          minimum=completed, what=f"disk sample after send {completed}")
                assert resume_proof._saved_checkpoint(run, completed) is not None
            except AssertionError as error:
                # Answer the provider call so the run can finish and report the failed barrier.
                checkpoint_errors.append(str(error))
            else:
                sample(f"send-{completed + 1}")
        return next_call(offered)

    monkeypatch.setattr(script, "next_call", after_checkpoint)
    server = serve(script)
    try:
        status = cli.main(_run_args(repo, run, f"http://127.0.0.1:{server.server_port}/v1"))
    finally:
        server.shutdown()
        server.server_close()
    assert status == 0
    script.assert_consumed()
    assert len(script.requests) == 5
    wires = sorted((run / "wire").glob("*.request.json"))
    sends = [json.loads(line) for line in (run / "sends.jsonl").read_bytes().splitlines()]
    assert len(wires) == len(sends) == 5
    for wire, send, received in zip(wires, sends, script.requests, strict=True):
        data = wire.read_bytes()
        assert hashlib.sha256(data).hexdigest() == send["request_sha256"]
        assert json.loads(data) == received
    assert runner.final_candidate(run)["shop/pricing.py"] == (run / "workspace/shop/pricing.py").read_bytes()
    assert not (run / "env").exists() and not (run / "checkpoints").exists()
    sample("retained")
    assert not checkpoint_errors, checkpoint_errors
    assert list(samples) == ["prepared", "send-2", "send-3", "send-4", "send-5", "retained"]
    for report in samples.values():
        assert report["seed"]["allocated_bytes"] > 0
        for metric in ("logical_bytes", "allocated_bytes", "attributable_bytes"):
            assert sum(row[metric] for row in report["buckets"].values()) == report["run"][metric]
    _gate(samples)


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="interruption harness identifies the child with ps --ppid")
def test_interrupted_checkpoint_reissues_exact_request(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    _fixture(repo)
    # Reuse the identity-checked interruption harness. Launch with the same project
    # environment, shipped profile and default retention as the uninterrupted edit.
    def command(example, repo, run_dir, url, launcher, profile, **kwargs):
        return [sys.executable, str(launcher), *_run_args(repo, run_dir, url)]

    monkeypatch.setattr(resume_proof, "_command", command)
    outcome = resume_proof._run_interrupted(
        "discount_bug", repo, run, resume_proof._write_launcher(tmp_path / "launcher.py"),
        ROOT / "src/stateframe/profiles/r2hc3.json", tmp_path / "home",
        script_document=_document(), completed_actions=1)
    assert outcome.pending_request == outcome.resumed_request
    assert (outcome.checkpoint_dir / "state.pkl").is_file()
    assert (run / "env").is_dir()
    assert (run / "env-seed.json").is_file()


def test_allocation_includes_root_and_deduplicates_links(tmp_path):
    seed, run = tmp_path / "seed", tmp_path / "run"
    seed.mkdir()
    env = run / "env"
    env.mkdir(parents=True)
    package = seed / "package"
    package.write_bytes(b"package" * 1024)
    os.link(package, env / "linked")
    private = run / "capture"
    private.write_bytes(b"capture" * 1024)
    os.link(private, env / "capture")
    (env / "symlink").symlink_to(private)
    (env / "external").symlink_to(tmp_path, target_is_directory=True)
    sparse = run / "sparse"
    with sparse.open("wb") as out:
        out.truncate(BUDGET * 2)

    report = storage_gc.run_allocation(run, seed_root=seed)
    private_size = (private.stat().st_blocks + sparse.stat().st_blocks) * 512
    seed_size = package.stat().st_blocks * 512
    assert report["seed"]["allocated_bytes"] == seed_size
    assert report["run"]["logical_bytes"] == 2 * private.stat().st_size + package.stat().st_size + BUDGET * 2
    assert report["run"]["allocated_bytes"] == report["combined_allocated_bytes"] == private_size + seed_size
    assert report["run"]["attributable_bytes"] == report["buckets"]["."]["attributable_bytes"] == private_size
    assert report["buckets"]["env"]["attributable_bytes"] == 0


def test_allocation_empty_directory_bucket(tmp_path):
    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    (run / "cas" / "objects").mkdir(parents=True)
    (run / "shell-home").mkdir()
    (run / "shell-home" / "alias").symlink_to(run / "requests", target_is_directory=True)
    (run / "alias").symlink_to(run / "requests", target_is_directory=True)

    report = storage_gc.run_allocation(run)
    zero = {"logical_bytes": 0, "allocated_bytes": 0, "attributable_bytes": 0}
    assert report["buckets"] == dict.fromkeys((".", "requests", "cas", "shell-home"), zero)
    assert report["run"] == zero


def test_duplicate_allocation_trips_gate(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    source, copy = run / "capture.bin", run / "duplicate.bin"
    rng = random.Random(406)
    # Two private writes avoid clone-aware copy implementations. The incompressible
    # original passes alone, even on the non-compressing reference filesystem.
    with source.open("wb") as output:
        for _ in range(50):
            output.write(rng.randbytes(1 << 20))
        output.flush()
        os.fsync(output.fileno())
    _gate({"single": storage_gc.run_allocation(run)})
    with source.open("rb") as original, copy.open("xb") as output:
        shutil.copyfileobj(original, output)
        output.flush()
        os.fsync(output.fileno())
    assert not os.path.samestat(source.stat(), copy.stat())
    expected = (source.stat().st_blocks + copy.stat().st_blocks) * 512
    report = storage_gc.run_allocation(run)
    assert report["run"]["attributable_bytes"] == expected
    if expected > BUDGET:
        with pytest.raises(AssertionError, match="run allocation exceeds"):
            _gate({"duplicated": report})

    # Also exercise the exact same inode walk and assertion with uncompressed block
    # counts for these two fully written files, even on a transparently deduplicating FS.
    file_stats = storage_gc._regular_file_stats

    def uncompressed(fd, names):
        for info in file_stats(fd, names):
            assert info.st_size == 50 << 20
            yield SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino,
                                  st_size=info.st_size, st_blocks=info.st_size // 512)

    monkeypatch.setattr(storage_gc, "_regular_file_stats", uncompressed)
    with pytest.raises(AssertionError, match="run allocation exceeds"):
        _gate({"uncompressed duplicate": storage_gc.run_allocation(run)})
