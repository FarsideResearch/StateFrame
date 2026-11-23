"""Revision-bound census reuse keeps request-specific projection intact."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_callsite_census as census
import kernel_repo_catalog as catalog


LEFT_SOURCE = b"from model import Item\nitem = Item()\n"
RIGHT_SOURCE = b"from model import Item\nitem = Item()\n"


@pytest.fixture(autouse=True)
def isolate_revision_cache():
    with census._CENSUS_REVISION_CACHE_CONDITION:
        census._CENSUS_REVISION_CACHE.clear()
        census._CENSUS_REVISION_CACHE_PENDING.clear()
    yield
    with census._CENSUS_REVISION_CACHE_CONDITION:
        census._CENSUS_REVISION_CACHE.clear()
        census._CENSUS_REVISION_CACHE_PENDING.clear()


def _enable_census(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_CALLSITE_CENSUS", "on")
    monkeypatch.setenv("STATEFRAME_CALLSITE_CENSUS_EDIT", "on")
    monkeypatch.setenv("STATEFRAME_CENSUS_INDEX_REUSE", "on")
    monkeypatch.setenv("STATEFRAME_CENSUS_INCREMENTAL", "off")
    monkeypatch.setenv("STATEFRAME_CATALOG_FULL_COVERAGE", "off")
    monkeypatch.setenv("STATEFRAME_TURN0_PERSIST", "off")
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "catalog"))


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "left.py").write_bytes(LEFT_SOURCE)
    (root / "right.py").write_bytes(RIGHT_SOURCE)
    (root / "model.py").write_text("class Item:\n    pass\n")
    return root


def _page(path: str, source: bytes) -> dict[str, str]:
    digest = sha256(source).hexdigest()
    lines = len(source.splitlines())
    return {
        "role": "tool",
        "content": (f"[read page] path={path} revision=sha256:{digest} lines=1-{lines}/{lines}\n"
                    f"Read complete\n{source.decode()}"),
    }


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, *args, **row):
        self.events.append({"kind": kind, **row})


def _runtime(root):
    return SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={"editable_paths": ["left.py", "right.py"]}),
        journal=_Journal(),
    )


def _prepare(root, path, *, runtime=None):
    runtime = runtime or _runtime(root)
    messages = [_page(path, LEFT_SOURCE if path == "left.py" else RIGHT_SOURCE)]
    result = census.prepare_request(
        runtime, {"decision_kind": "stage_l"}, messages, [], frame_ceiling=32768,
        request_id="same-request", census_token_counter=lambda _body: 1,
    )
    return result, runtime


def _shared_index_lease(monkeypatch, index, *, cache=None):
    lease = SimpleNamespace(
        index=index, cache=cache, changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)


def _site_visibility(packet):
    return {site["path"]: site["shown"] for group in packet.groups for site in group["sites"]}


def test_counted_work_reuses_one_census_across_fresh_prepares(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    left, _ = _prepare(root, "left.py")
    right, _ = _prepare(root, "right.py")

    assert builds == 1
    assert left.status == right.status == "fired"
    assert _site_visibility(left.packet) == {"left.py": True, "right.py": False}
    assert _site_visibility(right.packet) == {"left.py": False, "right.py": True}


def test_revision_change_invalidates_cached_census(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    indexes = [
        catalog.build_structural_index(
            root, priority_paths=("left.py", "right.py"), include_call_graph=False,
            capture_scan_entries=True,
        )
    ]
    _shared_index_lease(monkeypatch, indexes[0])
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    before, _ = _prepare(root, "left.py")
    assert before.status == "fired"

    (root / "right.py").write_bytes(b"from model import Other\nitem = Other()\n")
    indexes.append(catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    ))
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: SimpleNamespace(
        index=indexes[-1], cache=None, changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    ))
    after, _ = _prepare(root, "left.py")

    assert builds == 2
    assert after.status == "route_off"
    assert after.packet is None


def test_compact_cache_hit_does_not_seed_incremental_analysis(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    original_lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: original_lease)
    first, original_runtime = _prepare(root, "left.py")
    assert first.status == "fired", (first.error_type, original_runtime.journal.events)
    assert original_lease.cache.analysis.complete

    incremental_lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: incremental_lease)
    cached, _ = _prepare(root, "right.py")
    assert cached.status == "fired"
    assert incremental_lease.cache.analysis is None

    (root / "model.py").write_text('class Item:\n    """Revised imported definition."""\n    pass\n')
    incremental_lease.index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    incremental_lease.cache.index = incremental_lease.index
    incremental_lease.incremental = True
    incremental_lease.changed_paths = ("model.py",)
    monkeypatch.setenv("STATEFRAME_CENSUS_INCREMENTAL", "on")
    refreshed, _ = _prepare(root, "left.py")

    assert refreshed.status == "fired"
    assert _site_visibility(refreshed.packet) == {"left.py": True, "right.py": False}


def test_concurrent_prepares_share_one_build(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    _shared_index_lease(monkeypatch, index)
    original = census._build_census_structure
    count_lock = threading.Lock()
    builds = 0
    ready = threading.Barrier(4)

    def counted(*args, **kwargs):
        nonlocal builds
        with count_lock:
            builds += 1
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)

    def prepare(index_number):
        ready.wait(timeout=5)
        path = "left.py" if index_number % 2 == 0 else "right.py"
        return _prepare(root, path)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(prepare, range(4)))

    assert builds == 1
    assert all(result.status == "fired" for result in results)
    assert [_site_visibility(result.packet) for result in results] == [
        {"left.py": True, "right.py": False},
        {"left.py": False, "right.py": True},
        {"left.py": True, "right.py": False},
        {"left.py": False, "right.py": True},
    ]


def test_failed_census_build_is_not_cached(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    _shared_index_lease(monkeypatch, index)
    original = census._build_census_structure
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected census build failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", fail_once)
    failed, _ = _prepare(root, "left.py")
    succeeded, _ = _prepare(root, "left.py")

    assert failed.status == "unavailable"
    assert failed.error_type == "RuntimeError"
    assert succeeded.status == "fired"
    assert attempts == 2


def test_cached_projection_keeps_request_and_journal_bytes(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    monkeypatch.setattr(census, "time", SimpleNamespace(perf_counter_ns=lambda: 1234))
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    _prepare(root, "left.py")
    cached, cached_runtime = _prepare(root, "right.py")
    monkeypatch.setattr(census, "_census_revision_cache_key", lambda *_args, **_kwargs: None)
    uncached, uncached_runtime = _prepare(root, "right.py")

    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    assert encode(cached.messages) == encode(uncached.messages)
    assert cached.packet.payload == uncached.packet.payload
    assert cached.packet.coverage == uncached.packet.coverage
    assert encode(cached_runtime.journal.events) == encode(uncached_runtime.journal.events)
    assert cached_runtime.journal.events[0]["kind"] == "callsite_census_outcome"
    assert builds == 2


def test_two_workspaces_do_not_share_revision_cache_entries(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_root = _workspace(first_dir)
    second_root = _workspace(second_dir)
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    _prepare(first_root, "left.py")
    _prepare(second_root, "left.py")
    _prepare(first_root, "right.py")

    assert builds == 2
    assert set(census._CENSUS_REVISION_CACHE) == {
        first_root.resolve().as_posix(), second_root.resolve().as_posix(),
    }


def test_partial_census_is_reused_with_partial_coverage_preserved(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    (root / "broken.py").write_bytes(b"def incomplete(:\n")
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    first, _ = _prepare(root, "left.py")
    second, _ = _prepare(root, "right.py")

    assert first.status == second.status == "fired"
    assert first.packet.coverage["status"] == second.packet.coverage["status"] == "PARTIAL"
    assert first.packet.coverage["files_partial"] == second.packet.coverage["files_partial"] == 1
    assert builds == 1


def test_scoped_analysis_never_consumes_a_full_revision_cache_entry(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    full, _ = _prepare(root, "left.py")
    assert full.status == "fired"
    assert len(census._CENSUS_REVISION_CACHE) == 1

    import kernel_repo_dependencies

    class ScopedAnalysis:
        @staticmethod
        def analysis(*_args, **_kwargs):
            return "scoped", None

    monkeypatch.setattr(kernel_repo_dependencies, "active", ScopedAnalysis())
    scoped, _ = _prepare(root, "right.py")

    assert scoped.status == "fired"
    assert scoped.packet.coverage["scan_scope"] == "scoped"
    assert len(census._CENSUS_REVISION_CACHE) == 1


def test_revision_cache_enforces_entry_source_and_call_count_limits(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    monkeypatch.setattr(census, "_CENSUS_REVISION_CACHE_MAX_ENTRIES", 2)
    for number in range(4):
        root_dir = tmp_path / f"small-{number}"
        root_dir.mkdir()
        _prepare(_workspace(root_dir), "left.py")
    assert len(census._CENSUS_REVISION_CACHE) == 2

    monkeypatch.setattr(census, "MAX_CACHED_ANALYSIS_SOURCE_BYTES", 1)
    large_dir = tmp_path / "source-over-limit"
    large_dir.mkdir()
    _prepare(_workspace(large_dir), "left.py")
    assert len(census._CENSUS_REVISION_CACHE) == 2

    monkeypatch.setattr(census, "MAX_CACHED_ANALYSIS_SOURCE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(census, "_CENSUS_REVISION_CACHE_MAX_CALLS", 1)
    call_heavy_dir = tmp_path / "calls-over-limit"
    call_heavy_dir.mkdir()
    _prepare(_workspace(call_heavy_dir), "left.py")
    assert call_heavy_dir.joinpath("workspace").resolve().as_posix() not in census._CENSUS_REVISION_CACHE


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_single_flight_releases_waiters_after_exception_or_cancellation(
        tmp_path, monkeypatch, failure):
    key = ((tmp_path / "workspace").resolve().as_posix(), "shared-revision")
    index = SimpleNamespace(modules=())
    structure = census._CensusStructure(
        "run-revision", "run-scope", "parser", {}, {}, {}, {"a.py": ()},
        {"a.py": 0}, set(), set(), {},
    )
    builder_started = threading.Event()
    waiter_started = threading.Event()
    release_builder = threading.Event()
    attempts = 0

    def build():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            builder_started.set()
            assert release_builder.wait(timeout=5)
            raise failure("injected single-flight interruption")
        return structure

    condition = census._CENSUS_REVISION_CACHE_CONDITION
    original_wait = condition.wait

    def tracked_wait(timeout=None):
        waiter_started.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", tracked_wait)

    def get_structure():
        return census._census_structure_for_revision(
            key, revision="run-revision", scope_key="run-scope", parser_version="parser",
            index=index, build=build,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(get_structure)
        assert builder_started.wait(timeout=2)
        second = pool.submit(get_structure)
        try:
            assert waiter_started.wait(timeout=2)
        finally:
            release_builder.set()
        with pytest.raises(failure, match="injected single-flight interruption"):
            first.result(timeout=3)
        assert second.result(timeout=3).calls_by_path == {"a.py": ()}

    assert attempts == 2
    assert not census._CENSUS_REVISION_CACHE_PENDING


def test_subprocess_worker_can_build_and_hit_its_own_revision_cache(tmp_path):
    engine = Path(census.__file__).resolve().parent
    source = engine.parents[1]
    code = (
        "import time; from types import SimpleNamespace; "
        "import kernel_callsite_census as census; "
        "calls=[]; "
        "structure=census._CensusStructure('run','scope','parser',{}, {}, {}, "
        "{'a.py':()}, {'a.py':0}, set(), set(), {}); "
        "index=SimpleNamespace(modules=()); key=('isolated-worker','revision'); "
        "results=[census._census_structure_for_revision("
        "key,revision='run',scope_key='scope',parser_version='parser',index=index, "
        "build=lambda: (calls.append(1) or structure)) for _ in range(2)]; "
        "print(len(calls)); time.sleep(0.05)"
    )
    environment = dict(os.environ)
    extra_path = [str(source), str(engine)]
    existing = environment.get("PYTHONPATH")
    if existing:
        extra_path.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(extra_path)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert result.stdout.strip() == "1"


def test_actual_workspace_revert_and_untracked_addition_invalidate_revision_cache(
        tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    original = census._build_census_structure
    builds = 0

    def counted(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted)
    first, _ = _prepare(root, "left.py")
    assert first.status == "fired"

    changed = b"from model import Other\nitem = Other()\n"
    (root / "right.py").write_bytes(changed)
    edited, _ = _prepare(root, "left.py")
    assert edited.status == "route_off"

    (root / "right.py").write_bytes(RIGHT_SOURCE)
    reverted, _ = _prepare(root, "left.py")
    assert reverted.status == "fired"

    (root / "untracked.py").write_bytes(LEFT_SOURCE)
    untracked, _ = _prepare(root, "left.py")
    assert untracked.status == "fired"
    assert untracked.packet.coverage["files_available"] == 4
    assert builds == 4
