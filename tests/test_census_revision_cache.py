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

import kernel_checkpoint as checkpoint
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


def test_revision_facts_do_not_enter_checkpoint_state_or_journal(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    _result, runtime = _prepare(root, "left.py")
    assert census._CENSUS_REVISION_CACHE

    module_state = checkpoint._module_state("kernel_callsite_census", census)
    saved_cache = module_state["_CENSUS_REVISION_CACHE"]
    assert type(saved_cache) is type(census._CENSUS_REVISION_CACHE)
    assert not saved_cache
    assert "model.Item" not in json.dumps(runtime.journal.events, sort_keys=True)


def test_small_legacy_admission_ignores_new_compact_size_limit(tmp_path, monkeypatch):
    key = ((tmp_path / "workspace").resolve().as_posix(), "admitted-revision")
    index = SimpleNamespace(modules=())
    structure = census._CensusStructure(
        "run", "scope", "parser", {}, {}, {}, {"a.py": ()}, {"a.py": 0}, set(), set(), {},
    )
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return structure

    monkeypatch.setattr(census, "_analysis_source_size", lambda _index: 430 * 1024)
    monkeypatch.setattr(
        census, "_compact_census_structure_bytes",
        lambda _structure: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    for _ in range(2):
        census._census_structure_for_revision(
            key, revision="run", scope_key="scope", parser_version="parser",
            index=index, build=build,
        )

    assert builds == 1


def test_small_legacy_hit_keeps_the_v0830_unreadable_source_behavior(tmp_path, monkeypatch):
    key = ((tmp_path / "workspace").resolve().as_posix(), "legacy-unreadable")
    index = SimpleNamespace(modules=(SimpleNamespace(path="a.py"),))
    index.read_content = lambda *_args, **_kwargs: pytest.fail("legacy hit re-read a source")
    structure = census._CensusStructure(
        "run", "scope", "parser", {}, {}, {}, {"a.py": ()}, {"a.py": 0}, set(), set(), {},
    )
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return structure

    monkeypatch.setattr(census, "_analysis_source_size", lambda _index: 430 * 1024)
    census._census_structure_for_revision(
        key, revision="run", scope_key="scope", parser_version="parser", index=index, build=build,
    )
    reused = census._census_structure_for_revision(
        key, revision="run", scope_key="scope", parser_version="parser",
        index=index, build=build,
    )

    assert builds == 1
    assert reused.complete is False


def test_revision_cache_return_does_not_share_mutable_facts(tmp_path):
    key = ((tmp_path / "workspace").resolve().as_posix(), "immutable-revision")
    index = SimpleNamespace(modules=())
    structure = census._CensusStructure(
        "run", "scope", "parser", {}, {}, {}, {"a.py": ()}, {"a.py": 0}, {"bad.py"}, set(), {},
    )
    cached = census._census_structure_for_revision(
        key, revision="run", scope_key="scope", parser_version="parser",
        index=index, build=lambda: structure,
    )
    cached.calls_by_path["a.py"] = ()
    cached.calls_by_path["injected.py"] = ()
    cached.unresolved_by_path["a.py"] = 99
    cached.partial_paths.clear()

    reused = census._census_structure_for_revision(
        key, revision="run", scope_key="scope", parser_version="parser",
        index=index, build=lambda: pytest.fail("cache mutation caused a miss"),
    )

    assert reused.calls_by_path == {"a.py": ()}
    assert reused.unresolved_by_path == {"a.py": 0}
    assert reused.partial_paths == {"bad.py"}


def test_compact_size_walk_counts_all_empty_structure_maps(monkeypatch):
    structure = census._CensusStructure(
        "r", "s", "p", {}, {}, {}, {}, {}, set(), set(), {},
    )
    empty_maps = (structure.parsed, structure.trees, structure.definitions, structure.importers_by_path)
    expected_ids = {id(value) for value in empty_maps}
    seen = set()
    original_getsizeof = sys.getsizeof

    def tracked_getsizeof(value, *args):
        if isinstance(value, dict) and id(value) in expected_ids:
            seen.add(id(value))
        return original_getsizeof(value, *args)

    monkeypatch.setattr(sys, "getsizeof", tracked_getsizeof)
    census._compact_census_structure_bytes(structure)

    assert seen == {id(value) for value in empty_maps}


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


def test_oversized_index_reuses_bounded_facts_without_reparsing_files(
        tmp_path, monkeypatch, request):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    distinct_right = RIGHT_SOURCE + b"# right file identity\n"
    (root / "right.py").write_bytes(distinct_right)
    monkeypatch.setattr(sys.modules[__name__], "RIGHT_SOURCE", distinct_right)
    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    parse_cache = census._parse_cache
    parse_cache.clear()
    request.addfinalizer(parse_cache.clear)
    parse_calls = {}
    original_parse = parse_cache.parse

    def counted_parse(source, filename="<unknown>", *, on_miss=None):
        digest = parse_cache.source_digest(source)
        parse_calls[digest] = parse_calls.get(digest, 0) + 1
        return original_parse(source, filename, on_miss=on_miss)

    monkeypatch.setattr(parse_cache, "parse", counted_parse)
    original_build = census._build_census_structure
    builds = 0

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted_build)
    runtime = _runtime(root)
    left, _ = _prepare(root, "left.py", runtime=runtime)
    right, _ = _prepare(root, "right.py", runtime=runtime)

    assert left.status == right.status == "fired"
    assert _site_visibility(left.packet) == {"left.py": True, "right.py": False}
    assert _site_visibility(right.packet) == {"left.py": False, "right.py": True}
    assert builds == 1
    assert len(parse_calls) == 3
    assert set(parse_calls.values()) == {1}
    cached = next(iter(census._CENSUS_REVISION_CACHE.values()))[1]
    assert census._compact_census_structure_bytes(cached) <= census.MAX_CACHED_ANALYSIS_SOURCE_BYTES


def test_oversized_syntax_partial_keeps_compact_cache_saving(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    (root / "broken.py").write_text("def incomplete(:\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False, reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    original_build = census._build_census_structure
    builds = 0

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted_build)
    first, _ = _prepare(root, "left.py")
    second, _ = _prepare(root, "right.py")

    assert first.packet.coverage["status"] == second.packet.coverage["status"] == "PARTIAL"
    assert builds == 1
    assert lease.cache.analysis is None
    assert census._CENSUS_REVISION_CACHE


def test_oversized_source_read_failure_does_not_admit_compact_entry(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    (root / "broken.py").write_text("class Broken:\n    pass\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False, reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    original_read = index.read_content
    failed = False

    def fail_once(path, **kwargs):
        nonlocal failed
        if path == "broken.py" and not failed:
            failed = True
            raise OSError("injected source read failure")
        return original_read(path, **kwargs)

    monkeypatch.setattr(index, "read_content", fail_once)
    original_build = census._build_census_structure
    builds = 0

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted_build)
    first, _ = _prepare(root, "left.py")
    assert first.packet.coverage["status"] == "PARTIAL"
    assert not census._CENSUS_REVISION_CACHE

    second, _ = _prepare(root, "right.py")
    assert second.status == "fired"
    assert builds == 2
    assert census._CENSUS_REVISION_CACHE


@pytest.mark.parametrize("body_state", ["changed", "missing"])
def test_changed_or_unavailable_indexed_body_does_not_admit_compact_entry(
        tmp_path, monkeypatch, body_state):
    root = _workspace(tmp_path)
    path = root / "broken.py"
    path.write_text("class Broken:\n    pass\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py", "broken.py"), include_call_graph=False,
    )
    if body_state == "changed":
        path.write_text("class Changed:\n    pass\n")
    else:
        path.unlink()
    monkeypatch.setattr(
        census, "_analysis_source_size", lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )

    structure = census._census_structure_for_revision(
        (root.resolve().as_posix(), body_state), revision="run", scope_key="scope",
        parser_version="parser", index=index,
        build=lambda: census._build_census_structure(
            index, scope_key="scope", parser_version="parser",
        ),
    )

    assert structure.read_failed
    assert structure.partial_paths == {"broken.py"}
    assert "broken.py" not in structure.parsed
    assert not census._CENSUS_REVISION_CACHE


@pytest.mark.parametrize("error_type", [LookupError, OSError, catalog.StructuralIndexError, ValueError])
def test_source_paths_marks_every_index_read_failure(error_type):
    index = SimpleNamespace(modules=(SimpleNamespace(path="unavailable.py"),))

    def fail_read(*_args, **_kwargs):
        raise error_type("body unavailable")

    index.read_content = fail_read
    failures = set()

    parsed, trees, partial, _skipped = census._source_paths(index, failures)

    assert not parsed and not trees
    assert partial == failures == {"unavailable.py"}


def test_source_paths_keeps_parser_oserror_partial(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(root, include_call_graph=False)
    parse = census._parse_cached_source

    def fail_one_parse(source, path):
        if path == "left.py":
            raise OSError("parser unavailable")
        return parse(source, path)

    monkeypatch.setattr(census, "_parse_cached_source", fail_one_parse)

    parsed, trees, partial, _skipped = census._source_paths(index)

    assert "left.py" in partial
    assert "left.py" not in parsed and "left.py" not in trees


def test_transient_read_failure_keeps_legacy_partial_analysis(
        tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False, reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    original_read = index.read_content
    failed = False

    def fail_model_once(path, **kwargs):
        nonlocal failed
        if path == "model.py" and not failed:
            failed = True
            raise OSError("injected one-time read failure")
        return original_read(path, **kwargs)

    monkeypatch.setattr(index, "read_content", fail_model_once)
    first, _ = _prepare(root, "left.py")

    assert failed
    assert first.status == "fired"
    assert first.packet.coverage["status"] == "PARTIAL"
    assert lease.cache.analysis is not None
    assert lease.cache.analysis.partial_paths == {"model.py"}
    assert len(census._CENSUS_REVISION_CACHE) == 1

    second, _ = _prepare(root, "right.py")
    assert second.status == "fired"
    assert second.packet.coverage["status"] == "PARTIAL"
    assert second.packet.coverage["files_partial"] == 1


def test_large_compact_hit_reads_sources_without_reparsing(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease = SimpleNamespace(
        index=index, cache=SimpleNamespace(index=index, overlay_index=None, analysis=None),
        changed_paths=(), incremental=False, reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    first, _ = _prepare(root, "left.py")
    assert first.status == "fired"

    original_read = index.read_content
    reads = []
    denied = False

    def counted_read(path, **kwargs):
        nonlocal denied
        reads.append(path)
        if path == "model.py" and not denied:
            denied = True
            raise OSError("injected one-time census read denial")
        return original_read(path, **kwargs)

    monkeypatch.setattr(index, "read_content", counted_read)
    parse_cache = census._parse_cache

    def forbidden_parse(*_args, **_kwargs):
        raise AssertionError("a compact cache hit parsed source again")

    monkeypatch.setattr(parse_cache, "parse", forbidden_parse)
    second, _ = _prepare(root, "right.py")

    assert second.status == "fired"
    assert denied
    assert reads.count("model.py") == 2
    assert set(reads) == {"left.py", "right.py", "model.py"}
    assert _site_visibility(second.packet) == {"left.py": False, "right.py": True}


@pytest.mark.parametrize("setting", ["get_int_max_str_digits", "getrecursionlimit"])
def test_parser_settings_change_census_and_ast_cache_keys(monkeypatch, setting):
    getter = getattr(sys, setting, None)
    if not callable(getter):
        pytest.skip(f"sys.{setting} is unavailable")
    original_value = getter()
    source = "value = 123\n"
    parser_version = census._parser_version()
    tree_key = census._parse_cache.parse_cache_key(source)

    monkeypatch.setattr(sys, setting, lambda: original_value + 1)

    assert census._parser_version() != parser_version
    assert census._parse_cache.parse_cache_key(source) != tree_key


def test_nonadmitted_revision_entry_keeps_incremental_bindings_after_syntax_repair(
        tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    cache = SimpleNamespace(index=index, overlay_index=None, analysis=None)
    lease = SimpleNamespace(
        index=index, cache=cache, changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    )
    runtime = _runtime(root)
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    original_build = census._build_census_structure
    builds = 0

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(census, "_build_census_structure", counted_build)
    monkeypatch.setattr(census, "_CENSUS_REVISION_CACHE_MAX_CALLS", 1)
    first, _ = _prepare(root, "left.py", runtime=runtime)

    assert first.status == "fired"
    assert not census._CENSUS_REVISION_CACHE
    assert cache.analysis is not None and cache.analysis.complete
    original_analysis = cache.analysis
    original_bindings = cache.analysis.definitions["model.py"]

    monkeypatch.setenv("STATEFRAME_CENSUS_INCREMENTAL", "on")
    (root / "right.py").write_text("def incomplete(:\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease.index = index
    cache.index = index
    lease.incremental = True
    lease.changed_paths = ("right.py",)
    _prepare(root, "left.py", runtime=runtime)
    assert cache.analysis is not original_analysis
    assert cache.analysis.partial_paths == {"right.py"}
    assert cache.analysis.definitions["model.py"] == original_bindings

    (root / "right.py").write_bytes(RIGHT_SOURCE + b"# repaired\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease.index = index
    cache.index = index
    lease.changed_paths = ("right.py",)
    repaired, _ = _prepare(root, "left.py", runtime=runtime)

    assert repaired.status == "fired"
    assert cache.analysis.definitions["model.py"] == original_bindings
    assert set(cache.analysis.calls_by_path) == {"left.py", "right.py", "model.py"}
    assert all(
        call.identity.endswith("Item")
        for path in ("left.py", "right.py")
        for call in cache.analysis.calls_by_path[path]
    )
    assert builds == 1


def test_turn0_cache_size_boundary_keeps_legacy_journal_events(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    monkeypatch.setenv("STATEFRAME_TURN0_PERSIST", "on")
    monkeypatch.setattr(census._turn0, "cache_directory", lambda: tmp_path / "turn0-cache")
    monkeypatch.setattr(
        census._turn0, "make_context",
        lambda *_args, **_kwargs: (SimpleNamespace(path=tmp_path / "context.json"), None),
    )
    root = _workspace(tmp_path)
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    monkeypatch.setattr(census, "_runtime_persistence_task_id", lambda _runtime: "task")
    parser_version = census._parser_version()

    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES,
    )
    at_limit = _Journal()
    context, _, _ = census._census_persistence_context(
        index, _runtime(root), parser_version=parser_version, journal=at_limit,
    )
    assert context is not None
    assert at_limit.events == []

    monkeypatch.setattr(
        census, "_analysis_source_size",
        lambda _index: census.MAX_CACHED_ANALYSIS_SOURCE_BYTES + 1,
    )
    over_limit = _Journal()
    context, _, _ = census._census_persistence_context(
        index, _runtime(root), parser_version=parser_version, journal=over_limit,
    )
    assert context is None
    assert over_limit.events == [
        {"kind": "turn0_cache", "name": "census_index", "outcome": "miss", "dispatch": False},
        {"kind": "cache_bypass", "name": "census_index", "reason": "limits", "dispatch": False},
    ]


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


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
def test_partial_analysis_keeps_v0830_refresh_behavior_through_edit_add_delete_repair(
        tmp_path, monkeypatch, mode):
    _enable_census(monkeypatch, tmp_path)
    monkeypatch.setenv("STATEFRAME_CENSUS_INCREMENTAL", mode)
    root = _workspace(tmp_path)
    (root / "broken.py").write_text("def incomplete(:\n")
    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    cache = SimpleNamespace(index=index, overlay_index=None, analysis=None)
    lease = SimpleNamespace(
        index=index, cache=cache, changed_paths=(), incremental=False,
        reuse_status="cold", reuse_reason=None,
    )
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)
    runtime = _runtime(root)
    first, _ = _prepare(root, "left.py", runtime=runtime)
    assert first.status == "fired"
    assert cache.analysis is not None
    assert cache.analysis.partial_paths == {"broken.py"}

    refreshes = []
    original_refresh = census._refresh_census_structure

    def counted_refresh(*args, **kwargs):
        refreshes.append(tuple(kwargs["changed_paths"]))
        return original_refresh(*args, **kwargs)

    monkeypatch.setattr(census, "_refresh_census_structure", counted_refresh)
    steps = (
        ("model.py", lambda: (root / "model.py").write_text(
            "class Item:\n    pass\n\n# edited\n"), {"broken.py"}),
        ("added.py", lambda: (root / "added.py").write_text(
            "from model import Item\nextra = Item()\n"), {"broken.py"}),
        ("added.py", lambda: (root / "added.py").unlink(), {"broken.py"}),
        ("broken.py", lambda: (root / "broken.py").write_text(
            "class Repaired:\n    pass\n"), set()),
    )
    for changed_path, update, expected_partial in steps:
        update()
        index = catalog.build_structural_index(
            root, priority_paths=("left.py", "right.py"), include_call_graph=False,
            capture_scan_entries=True,
        )
        lease.index = index
        cache.index = index
        lease.incremental = True
        lease.changed_paths = (changed_path,)
        result, _ = _prepare(root, "left.py", runtime=runtime)
        assert result.status == "fired"
        assert cache.analysis is not None
        assert cache.analysis.partial_paths == expected_partial

    assert (len(refreshes) == len(steps)) is (mode in {"on", "shadow"})
    shadows = [event for event in runtime.journal.events
               if event["kind"] == "callsite_census_incremental_shadow"]
    assert len(shadows) == (len(steps) if mode == "shadow" else 0)


def test_turn0_persists_partial_analysis_and_keeps_hit_journal_event(tmp_path, monkeypatch):
    _enable_census(monkeypatch, tmp_path)
    monkeypatch.setenv("STATEFRAME_TURN0_PERSIST", "on")
    monkeypatch.setattr(census._turn0, "cache_directory", lambda: tmp_path / "turn0")
    monkeypatch.setattr(census._turn0, "_snapshot_digest", lambda: "fixed-snapshot")
    root = _workspace(tmp_path)
    (root / "broken.py").write_text("def incomplete(:\n")
    monkeypatch.setattr(census, "_runtime_persistence_task_id", lambda _runtime: "partial-task")
    lease = SimpleNamespace(index=None, cache=None, changed_paths=(), incremental=False,
                            reuse_status="cold", reuse_reason=None)
    monkeypatch.setattr(census, "acquire_index", lambda *_args, **_kwargs: lease)

    index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    lease.index = index
    first, first_runtime = _prepare(root, "left.py")
    first_events = [event for event in first_runtime.journal.events
                    if event["kind"] == "turn0_cache"]
    assert [event["outcome"] for event in first_events] == ["miss"]

    lease.index = catalog.build_structural_index(
        root, priority_paths=("left.py", "right.py"), include_call_graph=False,
        capture_scan_entries=True,
    )
    second, second_runtime = _prepare(root, "right.py")
    second_events = [event for event in second_runtime.journal.events
                     if event["kind"] == "turn0_cache"]
    assert [event["outcome"] for event in second_events] == ["hit"]
    monkeypatch.setenv("STATEFRAME_TURN0_PERSIST", "off")
    cold, _ = _prepare(root, "right.py")
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    assert first.packet.coverage["status"] == second.packet.coverage["status"] == "PARTIAL"
    assert encode(second.messages) == encode(cold.messages)
    assert second.packet.payload == cold.packet.payload


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
