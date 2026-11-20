"""Artifact identity parity and work budgets at the notice and challenge entry points."""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import kernel_challenge_budget as challenge
import kernel_stateframe_notice_dedup as notice
import kernel_workspace_tree as tree
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers, write_scope
from kernel_workspace_index import WorkspaceIndex
from kernel_workspace_state import WorkspaceFence, WorkspaceState


TURN_SWITCHES = (
    "ATOMIC_PUBLICATION", "BOUNDED_QUALIFICATION", "DELIVERY_PROOFS",
    "GENERATION_OVERLAP", "INDEXED_READS", "MINIMAL_LAUNCH", "PANELS", "REPO_INTEL", "TIMING",
)


def _switches(monkeypatch, setting):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", setting)
    for name in TURN_SWITCHES:
        monkeypatch.setenv(f"STATEFRAME_TURN_{name}", setting)


def _bind(root):
    state = WorkspaceState(WorkspaceFence.create(root, root.parent / "fence"))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    return consumers


def _runtime(root):
    return SimpleNamespace(owner=SimpleNamespace(root=root), stateframe_notice_dedup_mode="on")


def _render(runtime):
    return notice.render_notice_bundle(
        runtime, coverage_payload=None, turn=1,
        unverified_rows=[{"id": "verify-api", "path": "app.py", "reason": "read the API"}],
    ).encode()


def _instrument(monkeypatch):
    work = Counter()

    def count(owner, name, key):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            work[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    count(Path, "rglob", "artifact_walks")
    count(hashlib, "file_digest", "legacy_blob_hashes")
    count(WorkspaceIndex, "_scan", "directories_scanned")
    count(WorkspaceIndex, "_file_digest", "index_blob_hashes")
    count(challenge, "_digest", "canonical_hashes")
    count(tree, "children", "tree_enumerations")
    original_diff = tree.diff

    def counted_diff(*args, **kwargs):
        for change in original_diff(*args, **kwargs):
            work["changed_entries"] += 1
            yield change

    monkeypatch.setattr(tree, "diff", counted_diff)
    return work


def test_4500_file_notice_revision_has_zero_clean_work(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root = tmp_path / "workspace"
    root.mkdir()
    for index in range(45):
        directory = root / f"pkg{index:02d}"
        directory.mkdir()
        for number in range(100):
            (directory / f"module{number:02d}.py").write_bytes(b"VALUE = 1\n")
    runtime = _runtime(root)
    consumers = _bind(root)
    work = _instrument(monkeypatch)

    started = perf_counter()
    with consumers.state.pin("test-admission") as observed:
        assert observed.validate_for_use("test-admission")
    admission_elapsed = perf_counter() - started
    admission_work = dict(work)

    work.clear()
    started = perf_counter()
    _render(runtime)
    first_revision = notice.state_snapshot(runtime)["revision"]
    first_elapsed = perf_counter() - started
    first_work = dict(work)

    work.clear()
    started = perf_counter()
    rendered = [_render(runtime) for _ in range(5)]
    warm_elapsed = perf_counter() - started
    warm_work = dict(work)
    work.clear()
    assert all(text == rendered[0] and text for text in rendered)
    assert first_revision == challenge.artifact_revision(runtime)
    challenge_work = dict(work)

    changed = root / "pkg00" / "module00.py"
    before = changed.stat()
    with write_scope(changed):
        changed.write_bytes(b"VALUE = 2\n")
        os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = changed.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns,
    )
    work.clear()
    started = perf_counter()
    _render(runtime)
    edit_elapsed = perf_counter() - started
    edit_work = dict(work)
    edited_revision = notice.state_snapshot(runtime)["revision"]
    with monkeypatch.context() as legacy:
        _switches(legacy, "off")
        assert edited_revision == challenge.artifact_revision(runtime)
    assert edited_revision != first_revision

    print(f"admission: {admission_elapsed:.6f}s {admission_work}")
    print(f"first render: {first_elapsed:.6f}s {first_work}")
    print(f"five clean renders: {warm_elapsed:.6f}s {warm_work}")
    print(f"clean challenge: {challenge_work}")
    print(f"same-size, same-mtime edit render: {edit_elapsed:.6f}s {edit_work}")
    assert warm_work == {}
    assert challenge_work == {}
    assert warm_elapsed < 2
    assert edit_work.get("artifact_walks", 0) == edit_work.get("directories_scanned", 0) == 0
    assert edit_work.get("tree_enumerations", 0) == edit_work.get("legacy_blob_hashes", 0) == 0
    assert edit_work.get("index_blob_hashes", 0) <= 1
    assert edit_work.get("changed_entries", 0) <= 1
    assert edit_work.get("canonical_hashes", 0) == 1


def test_projection_matches_legacy_inclusion_and_notice_bytes(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace_output = root / "workspace-output"
    workspace_output.mkdir()
    files = {
        "app.py": b"product",
        "tests/test_protected.py": b"protected test",
        ".git/config": b"ignored",
        "build/report.txt": b"ignored",
        "coverage.xml": b"ignored",
        "nested/coverage.xml/source.py": b"included",
        "journal.jsonl": b"excluded",
        "workspace-output/data": b"excluded",
        "challenge.txt": b"excluded",
    }
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    external = tmp_path / "external"
    external.symlink_to(root / "challenge.txt")
    (root / "product-alias").symlink_to("./app.py")
    (root / "dangling").symlink_to("missing")
    (root / "output-alias").symlink_to("workspace-output/data")
    (root / "external-alias").symlink_to(external)
    (root / "ignored-alias").symlink_to("build/report.txt")
    consumers = _bind(root)
    runtime = _runtime(root)
    legacy_runtime = _runtime(root)
    for owner in (runtime, legacy_runtime):
        owner.journal = SimpleNamespace(path=root / "journal.jsonl")
        owner.workspace = SimpleNamespace(directory=workspace_output)
        owner._challenge_budget_output = root / "challenge.txt"

    def compare():
        _switches(monkeypatch, "off")
        expected = challenge.artifact_revision(legacy_runtime)
        legacy_notice = _render(legacy_runtime)
        _switches(monkeypatch, "on")
        assert challenge.artifact_revision(runtime) == expected
        assert _render(runtime) == legacy_notice
        return expected

    included = {name: "sha256:" + hashlib.sha256(files[name]).hexdigest() for name in
                ("app.py", "tests/test_protected.py", "nested/coverage.xml/source.py")}
    included.update({"product-alias": "symlink:app.py", "dangling": "symlink:missing",
                     "ignored-alias": "symlink:build/report.txt"})
    first = compare()
    assert first == challenge._digest({"files": included}) == compare()

    ignored = (".git/config", "build/report.txt", "coverage.xml", "journal.jsonl",
               "workspace-output/data", "challenge.txt")
    with consumers.state.begin_write(ignored):
        for name in ignored:
            (root / name).write_bytes(b"new bytes")
    assert compare() == first

    with consumers.state.begin_write(("tests/test_protected.py",)):
        (root / "tests/test_protected.py").write_bytes(b"changed test")
    protected = compare()
    assert protected != first

    # The external hop can change exclusion membership without changing the workspace tree.
    external.unlink()
    external.symlink_to(root / "app.py")
    rerouted = compare()
    assert rerouted != protected
    for owner in (runtime, legacy_runtime):
        owner._challenge_budget_output = root / "app.py"
    assert compare() != rerouted

    with consumers.state.begin_write(("dangling", "tests", "new.py")):
        (root / "dangling").unlink()
        (root / "dangling").symlink_to("./new.py")
        (root / "tests").rename(root / "new.py")
    assert compare() != rerouted


def test_unavailable_projection_and_empty_artifact_keep_legacy_fallback(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "app.py"
    path.write_bytes(b"original")
    runtime = _runtime(root)
    original = challenge.artifact_revision(runtime)  # No registration yet.
    consumers = _bind(root)
    assert challenge.artifact_revision(runtime) == original
    with consumers.state.begin_write(("app.py",)):
        path.write_bytes(b"changed!")
        # The writer holds the fence, so the projection cannot supply a pin.
        during_write = challenge.artifact_revision(runtime)
        assert during_write != original
    assert challenge.artifact_revision(runtime) == during_write

    with consumers.state.begin_write(("app.py",)):
        path.unlink()
    assert challenge.artifact_revision(runtime, fallback="legacy fallback") == "legacy fallback"
    runtime.binding = lambda: {"artifact": "sha256:binding"}
    assert challenge.artifact_revision(runtime, fallback="legacy fallback") == "sha256:binding"


def test_off_and_explicit_notice_revision_do_not_request_a_pin(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_bytes(b"product")
    runtime = _runtime(root)
    consumers = _bind(root)

    def unexpected_pin(*args, **kwargs):
        raise AssertionError("the caller already supplied the revision or switched off the projection")

    monkeypatch.setattr(consumers.state, "pin", unexpected_pin)
    _switches(monkeypatch, "off")
    assert _render(runtime)
    assert notice.state_snapshot(runtime)["revision"] == challenge.artifact_revision(runtime)
    _switches(monkeypatch, "on")
    runtime.product_revision = "supplied-revision"
    assert _render(runtime)
    assert notice.state_snapshot(runtime)["revision"] == "supplied-revision"


def test_projection_falls_back_when_installation_home_hides_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    home = root / "private-state"
    home.mkdir(parents=True)
    (root / "app.py").write_bytes(b"product")
    (home / "config.json").write_bytes(b"legacy includes this path")
    consumers = _bind(root)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    runtime = _runtime(root)
    _switches(monkeypatch, "off")
    expected = challenge.artifact_revision(runtime)
    _switches(monkeypatch, "on")
    with consumers.state.pin("partial") as observed:
        assert tree.lookup(observed.tree, (b"private-state",)) is None
    assert challenge.artifact_revision(runtime) == expected
