"""Release gates for retained workspace bytes and the physical record format."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time
import zlib
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

import kernel_bridge_evidence as evidence
import pytest

from stateframe import _child
from stateframe.engine import kernel_checkpoint as checkpoint
from tools.workspace_artifact_probe import assert_size_class, compare_runs, observe_artifacts


TURNS = 20
TURN_LIMIT = 10
WORKSPACE_SETTING = "STATEFRAME_WORKSPACE_OBJECTS"


def _payloads() -> tuple[bytes, bytes]:
    rng = random.Random(31729)
    return rng.randbytes(12 << 10), base64.b64encode(rng.randbytes(9 << 10))


def _put(root: Path, name: str, data: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _count_workspace_access(probe, roots: tuple[Path, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    prefixes = tuple(os.fspath(root) for root in roots)

    def observe(operation, kind):
        def counted(path, *args, **kwargs):
            if isinstance(path, int):
                name = _descriptor_path(path)
            else:
                name = os.fsdecode(path)
                directory = kwargs.get("dir_fd")
                if directory is not None and not os.path.isabs(name):
                    name = os.path.join(_descriptor_path(directory), name)
            if name and any(name == root or name.startswith(root + os.sep) for root in prefixes):
                # A pinned snapshot must reopen the workspace root with
                # O_NOFOLLOW to verify its physical binding. Count those
                # separately from source traversal and payload reads.
                category = "root_open" if kind == "os.open" and name in prefixes else kind
                counts[category] += 1
            return operation(path, *args, **kwargs)
        return counted

    probe.setattr(os, "scandir", observe(os.scandir, "scandir"))
    probe.setattr(os, "open", observe(os.open, "os.open"))
    probe.setattr(io, "open", observe(io.open, "io.open"))
    return counts


def _descriptor_path(descriptor: int) -> str:
    try:
        return os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError:
        return ""


def _bind_direct_capture(run: Path, parent: Path, work: Path):
    # The legacy entry points need only the paths. The integrated entry points
    # also need the private host binding that normally comes from the runner.
    if find_spec("kernel_workspace_refs") is None:
        return ()
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_refs import bind_run_store
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    namespace = hashlib.sha256(b"workspace-format-fixture").hexdigest()
    bind_run_store(run, namespace)
    private = run.parent / "fences"
    private.mkdir(mode=0o700)
    bindings = []
    for root in (parent, work):
        state = WorkspaceState(WorkspaceFence.create(root, private / root.name))
        binding = WorkspaceConsumers(state)
        bind_consumers(root, binding)
        bindings.append(binding)
    return tuple(bindings)


def test_twenty_no_edit_boundaries_keep_workspace_payload_once(tmp_path, monkeypatch):
    run, home = tmp_path / "run", tmp_path / "home"
    home.mkdir(mode=0o700)
    run.mkdir(mode=0o700)
    parent, work = run / "parent", run / "workspace"
    candidates = run / "worker" / "candidates"
    monkeypatch.setenv(WORKSPACE_SETTING, "on")
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_KEEP", str(TURNS + 1))
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"format-gate-test-key-0123456789abcd"])
    payloads = _payloads()
    repeated_files = {"inventory/one.bin": payloads[0], "inventory/two.bin": payloads[0],
                      "inventory/text.txt": payloads[1]}
    entries = {f"inventory/{index // 100:02d}/{index:04d}.txt": b"stock\n" for index in range(4997)}
    entries.update(repeated_files)
    for name, data in entries.items():
        _put(parent, name, data)
        _put(work, name, data)

    # Use the existing direct capture entry points and the real state writer.
    # A fixed-size logical frame isolates the workspace tax from model history.
    context = {key: {} for key in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    state = {"frame": {}, "modules": {}, "classes": {}, "patched": {}, "participants": {},
             "context": [], "random": random.Random(31).getstate()}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (state, context))
    _bindings = _bind_direct_capture(run, parent, work)
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    row = {"path": "inventory/text.txt", "files": dict.fromkeys(repeated_files)}
    fixture_inputs = tuple(parent / name for name in repeated_files)

    def capture(step: int) -> float:
        started = time.perf_counter()
        evidence.capture_candidate(str(candidates), work, row, step)
        checkpoint.save(sys._getframe(), step, run)
        return time.perf_counter() - started

    cold = capture(1)
    before = observe_artifacts(run, home, payloads=payloads, fixture_inputs=fixture_inputs)
    with monkeypatch.context() as probe:
        counts = _count_workspace_access(probe, (parent, work))
        elapsed = [capture(step) for step in range(2, TURNS + 2)]
    after = observe_artifacts(run, home, payloads=payloads, fixture_inputs=fixture_inputs)
    print(f"cold={cold:.3f}s warm_total={sum(elapsed):.3f}s warm_max={max(elapsed):.3f}s "
          f"workspace_operations={dict(counts)}\nbefore: {before.counts()}\nafter: {after.counts()}")
    assert_size_class(before, after, unique_payload_bytes=0, changed_paths=0, turns=TURNS)
    assert sum(before.blob_sizes.values()) == sum(map(len, payloads)), (
        f"initial boundary must admit each witness exactly once\n{before.counts()}")
    assert counts["os.open"] == counts["scandir"] == counts["io.open"] == 0, (
        f"clean boundaries traversed or opened source payloads: {counts}")
    assert counts["root_open"] <= 12 * TURNS, (
        f"clean boundaries exceeded the bounded root-identity checks: {counts}")
    assert max(elapsed) < TURN_LIMIT


def test_home_exclusion_discovery_skips_independently_fenced_mount(tmp_path, monkeypatch):
    from kernel_workspace_checkpoint import _RunSide
    from kernel_workspace_refs import bind_run_store
    from kernel_workspace_tree import lookup

    run, home = tmp_path / "run", tmp_path / "home"
    fenced = run / "worker" / "session" / "repo"
    run.mkdir(mode=0o700)
    fenced.mkdir(parents=True)
    home.mkdir(mode=0o700)
    (fenced / "payload.bin").write_bytes(b"source")
    (fenced.parent / "home-link").symlink_to(home, target_is_directory=True)
    (fenced.parent / "retained").write_bytes(b"run artifact")
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    store = bind_run_store(run, "b" * 64)
    with monkeypatch.context() as probe:
        scans = _count_workspace_access(probe, (fenced,))
        side = _RunSide(run, fenced.relative_to(run), home, store)
        try:
            with side.pin() as snapshot:
                assert lookup(snapshot.tree, (b"worker", b"session", b"repo")) is None
                assert lookup(snapshot.tree, (b"worker", b"session", b"home-link")) is None
                assert lookup(snapshot.tree, (b"worker", b"session", b"retained")) is not None
        finally:
            side.close()
    assert scans["scandir"] == scans["io.open"] == scans["os.open"] == 0


def _write_reference_layout(run: Path, home: Path, payloads: tuple[bytes, ...]) -> None:
    namespace = "a" * 64
    for data in payloads:
        oid = hashlib.sha256(data).hexdigest()
        _put(home, f"workspace-state/1/{namespace}/objects/{oid}", data)
    for step in range(1, TURNS + 1):
        _put(run, f"checkpoints/{step:04d}/checkpoint.json", json.dumps({
            "schema": "stateframe-engine-checkpoint/2", "step": step,
        }).encode())
        _put(run, f"worker/candidates/{step:04d}.json", json.dumps({
            "schema": "stateframe-candidate-trees/1", "proposals": step,
        }).encode())


def test_format_gate_accepts_one_object_per_payload(tmp_path):
    run, home = tmp_path / "run", tmp_path / "home"
    payloads = _payloads()
    before = observe_artifacts(run, home, payloads=payloads)
    _write_reference_layout(run, home, payloads)
    _put(run, "workspace/inventory/one.bin", payloads[0])
    started = time.perf_counter()
    after = observe_artifacts(run, home, payloads=payloads)
    print(f"reference layout scan={time.perf_counter() - started:.6f}s {after.counts()}")
    assert_size_class(before, after, unique_payload_bytes=sum(map(len, payloads)), changed_paths=0, turns=TURNS)


@pytest.mark.parametrize("encode", [bytes, base64.b64encode, gzip.compress, zlib.compress])
def test_format_gate_rejects_content_in_an_unknown_sidecar(tmp_path, encode):
    run, home = tmp_path / "run", tmp_path / "home"
    payloads = _payloads()
    before = observe_artifacts(run, home, payloads=payloads)
    _write_reference_layout(run, home, payloads)
    # The sidecar lives outside every expected consumer directory and has no
    # filename, suffix or field that identifies what it holds.
    _put(run, "worker/output/segment-09", encode(payloads[0]))
    after = observe_artifacts(run, home, payloads=payloads)
    with pytest.raises(AssertionError, match="workspace content outside.*segment-09"):
        assert_size_class(before, after, unique_payload_bytes=sum(map(len, payloads)), changed_paths=0, turns=TURNS)


def _write_reference_run(run: Path, body: bytes, seal: str) -> None:
    _put(run, "wire/0001.request.json", body)
    _put(run, "summary.json", json.dumps({
        "engine_result": "SUBMITTED",
        "engine_terminal": {"result": "SUBMITTED", "submission": {"seal_id": seal}},
        "tests_before": {"passed": False}, "tests_after": {"passed": True},
    }).encode())


def test_release_oracle_requires_exact_request_bytes_and_delivery_seal(tmp_path):
    base, trial = tmp_path / "base", tmp_path / "trial"
    body, seal = b'{"model":"fixture"}', "sha256:" + "a" * 64
    _write_reference_run(base, body, seal)
    _write_reference_run(trial, body, seal)
    compare_runs(base, trial)


@pytest.mark.parametrize(("body", "seal", "failure"), [
    (b'{"model":"Fixture"}', "sha256:" + "a" * 64, "provider request 1 differs"),
    (b'{"model":"fixture"}', "sha256:" + "b" * 64, "delivery verdict or seal differs"),
])
def test_release_oracle_rejects_changed_request_or_seal(tmp_path, body, seal, failure):
    base, trial = tmp_path / "base", tmp_path / "trial"
    _write_reference_run(base, b'{"model":"fixture"}', "sha256:" + "a" * 64)
    _write_reference_run(trial, body, seal)
    with pytest.raises(AssertionError, match=failure):
        compare_runs(base, trial)
