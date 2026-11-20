"""Admission work for census during late request preparation."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import pytest

import kernel_callsite_census as census
import kernel_census_index_reuse as reuse
import kernel_repo_catalog as catalog


_TURN_SWITCHES = (
    "ATOMIC_PUBLICATION", "BOUNDED_QUALIFICATION", "DELIVERY_PROOFS",
    "GENERATION_OVERLAP", "INDEXED_READS", "MINIMAL_LAUNCH", "PANELS",
    "REPO_INTEL", "TIMING",
)


def _switches(monkeypatch, enabled: bool) -> None:
    mode = "on" if enabled else "off"
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", mode)
    for switch in _TURN_SWITCHES:
        monkeypatch.setenv("STATEFRAME_TURN_" + switch, mode)
    for switch in ("CALLSITE_CENSUS", "CALLSITE_CENSUS_EDIT", "CENSUS_INDEX_REUSE"):
        monkeypatch.setenv("STATEFRAME_" + switch, "on")
    monkeypatch.setenv("STATEFRAME_CATALOG_FULL_COVERAGE", "off")


def _page(path: str, source: bytes) -> dict[str, str]:
    lines = len(source.splitlines())
    revision = sha256(source).hexdigest()
    return {
        "role": "tool",
        "content": (f"[read page] path={path} revision=sha256:{revision} "
                    f"lines=1-{lines}/{lines}\n"
                    f"Read complete\n{source.decode()}"),
    }


def _prepare(runtime, messages, *, kind="stage_l", request_id=2):
    return census.prepare_request(
        runtime, {"decision_kind": kind}, messages, [],
        frame_ceiling=32768, request_id=request_id, collect_source_facts=True,
    )


@pytest.mark.usefixtures("isolated_pilot_context")
def test_warm_no_route_prepare_work(tmp_path, monkeypatch):
    from kernel_bolt import KernelBolt
    from kernel_stateframe_runtime import StateFrameRuntime

    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_") and name != "STATEFRAME_HOME":
            monkeypatch.delenv(name)
    _switches(monkeypatch, True)
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "catalog"))
    root = tmp_path / "workspace"
    root.mkdir()
    source = b"class Item:\n    pass\n\nitem = Item()\n"
    for folder in range(90):
        package = root / f"pkg_{folder:02d}"
        package.mkdir()
        for number in range(50):
            (package / f"module_{number:02d}.py").write_bytes(source)
    selected = "pkg_89/module_49.py"
    owner = KernelBolt(root, "", "census-admission", defer_symbols=True)
    owner.task_row = {"msg": "Review the change.", "editable_paths": [selected]}
    owner.current_step = 2
    runtime = StateFrameRuntime(
        owner, output=tmp_path / "measurement", target=root / selected,
        model="scripted", tools=[], temperature=0,
        source_resolver=lambda _path, _symbol: None, multi_file=True,
    )

    # Pre-warm the structural cache, as an earlier route could have done.
    reuse.acquire_index(
        runtime, root, scope_key=census._index_scope_key(runtime, root),
        parser_version=f"{census.EXTRACTOR_VERSION}:ast-{sys.version_info.major}."
                       f"{sys.version_info.minor}",
        priority_paths=[selected], enabled=True, build_index_fn=catalog.build_structural_index,
    )
    assert getattr(runtime, reuse.CACHE_ATTR) is not None

    work = Counter()
    acquire = census.acquire_index
    scan = catalog._scan_inventory
    open_file = Path.open
    os_open = os.open
    prepare_census = census.prepare_request
    emit = runtime.journal.emit
    outcomes = []

    def record_outcome(kind, *args, **row):
        if kind == "callsite_census_outcome":
            outcomes.append(row)
        return emit(kind, *args, **row)

    def count_acquire(*args, **kwargs):
        if census_running:
            work["acquisitions"] += 1
        return acquire(*args, **kwargs)

    def count_scan(*args, **kwargs):
        if census_running:
            work["inventory_scans"] += 1
        return scan(*args, **kwargs)

    def count_source(path):
        if isinstance(path, (str, bytes, Path)):
            name = os.fsdecode(path)
            if name.endswith(".py"):
                work["point_reads" if name.endswith(selected) else "unrelated_reads"] += 1

    def count_open(path, *args, **kwargs):
        if census_running:
            count_source(path)
        return open_file(path, *args, **kwargs)

    def count_os_open(path, *args, **kwargs):
        if census_running:
            if kwargs.get("dir_fd") is not None and path == Path(selected).name:
                work["point_reads"] += 1
            else:
                count_source(path)
        return os_open(path, *args, **kwargs)

    def observe_census(*args, **kwargs):
        nonlocal census_running
        census_running = True
        work.clear()
        start = perf_counter()
        try:
            result = prepare_census(*args, **kwargs)
            elapsed = perf_counter() - start
            assert result.status == "route_off", outcomes[-1]
            assert result.messages == tuple(args[2])
            observations.append((dict(work), outcomes[-1]["index_source_bytes_admitted"], elapsed))
            return result
        finally:
            census_running = False

    monkeypatch.setattr(census, "acquire_index", count_acquire)
    monkeypatch.setattr(catalog, "_scan_inventory", count_scan)
    monkeypatch.setattr(Path, "open", count_open)
    monkeypatch.setattr(os, "open", count_os_open)
    monkeypatch.setattr(runtime.journal, "emit", record_outcome)
    monkeypatch.setattr(census, "prepare_request", observe_census)
    census_running = False
    observations = []
    mentioned = [{"role": "user", "content": f"Consider changing {selected}."}]
    prepare_seconds = []
    for step, messages in enumerate((mentioned, [_page(selected, source)]), 2):
        owner.current_step = step
        messages = [{"role": "system", "content": "Review source."}, *messages]
        start = perf_counter()
        prepared = runtime.prepare(messages, uncapped=messages, tools=[])
        prepare_seconds.append(perf_counter() - start)
        assert prepared.overflow is None
    assert len(observations) == 2, observations
    print(f"census prepare work (mention, exact page): {observations}", flush=True)
    print(f"whole prepare seconds: {prepare_seconds}", flush=True)
    for observed, admitted_bytes, seconds in observations:
        assert observed.get("acquisitions", 0) == 0, observations
        assert observed.get("inventory_scans", 0) == 0, observations
        assert observed.get("unrelated_reads", 0) == 0, observations
        assert admitted_bytes == 0, observations
        assert seconds < 2, observations
    assert observations[0][0].get("point_reads", 0) == 0, observations
    assert 1 <= observations[1][0].get("point_reads", 0) <= 2, observations


@pytest.fixture
def relation_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "catalog"))
    root = tmp_path / "workspace"
    root.mkdir()
    caller = b"from model import Item\nitem = Item()\n"
    (root / "left.py").write_bytes(caller)
    (root / "right.py").write_bytes(caller)
    (root / "model.py").write_text("class Item:\n    pass\n")
    return root, caller


def _relation_runtime(root, *editable):
    return SimpleNamespace(owner=SimpleNamespace(root=root, task_row={"editable_paths": editable}))


@pytest.mark.parametrize("reuse_mode", ["off", "on"])
def test_cold_catalog_waits_for_python_carrier(relation_workspace, monkeypatch, reuse_mode):
    root, caller = relation_workspace
    _switches(monkeypatch, False)
    monkeypatch.setenv("STATEFRAME_CENSUS_INDEX_REUSE", reuse_mode)
    runtime = _relation_runtime(root, "left.py", "right.py")
    (root / "answer.json").write_text("{}\n")
    work = Counter()

    def count(module, name):
        original = getattr(module, name)

        def counted(*args, **kwargs):
            work[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    count(reuse, "_build_cold_lease")
    count(census, "build_structural_index")
    count(catalog, "_scan_inventory")

    initial = [{"role": "user", "content": "[Current source: answer.json]\n{}\n"}]
    first = _prepare(runtime, initial, kind="edit", request_id=1)
    assert first.status == "route_off"
    assert first.messages == tuple(initial)
    assert not work

    later = _prepare(runtime, [_page("left.py", caller)], kind="edit", request_id=2)
    assert later.status == "fired"
    assert work["build_structural_index"] == 1
    assert work["_scan_inventory"] > 0
    assert work["_build_cold_lease"] == (reuse_mode == "on")


def test_deferred_catalog_preserves_source_facts(relation_workspace, monkeypatch):
    import kernel_parse_cache as parse_cache

    root, caller = relation_workspace
    _switches(monkeypatch, False)
    runtime = _relation_runtime(root, "left.py", "right.py")
    first = _prepare(runtime, [{"role": "user", "content": "Review the construction sites."}],
                     kind="edit", request_id=1)
    assert first.status == "route_off"

    parse_cache.parse(caller.decode(), "left.py")
    parse_cache.parse((root / "model.py").read_text(), "model.py")
    parse_stats = parse_cache.stats()
    parsed_paths = []

    def shared_parse(source, path):
        parsed_paths.append(path)
        return parse_cache.parse(source, path)

    monkeypatch.setattr(census, "_parse_cache", SimpleNamespace(parse=shared_parse))
    later = _prepare(runtime, [_page("left.py", caller)], kind="edit", request_id=2)
    assert later.status == "fired"
    assert sorted(parsed_paths) == ["left.py", "model.py", "right.py"]
    assert parse_cache.stats()["tree_misses"] == parse_stats["tree_misses"]
    assert parse_cache.stats()["tree_hits"] > parse_stats["tree_hits"]
    assert {row["path"]: row["text"] for row in later.source_facts} == {
        "left.py": caller.decode(), "right.py": caller.decode(),
    }
    assert [(site["path"], site["shown"]) for site in later.packet.groups[0]["sites"]] == [
        ("left.py", True), ("right.py", False),
    ]
    ready_on_first = _prepare(
        _relation_runtime(root, "left.py", "right.py"), [_page("left.py", caller)],
        kind="edit", request_id=1,
    )
    assert ready_on_first.status == "fired"
    assert ready_on_first.packet.groups == later.packet.groups
    assert ready_on_first.messages == later.messages
    assert ready_on_first.source_facts == later.source_facts


@pytest.mark.parametrize("kind", ["edit", "stage_l"])
@pytest.mark.parametrize("cache_defect", ["none", "incomplete", "corrupt"])
def test_census_route_matches_legacy_with_current_handles(
        relation_workspace, monkeypatch, kind, cache_defect):
    root, caller = relation_workspace
    left = {"role": "user", "content": "[Current source: left.py]\n" + caller.decode()}
    right = _page("right.py", caller)
    # In edit, the other file is declared but unread. In stage L, both are shown.
    editable = ("left.py", "right.py") if kind == "edit" else ()
    messages = [left] if kind == "edit" else [left, right]
    runtime = _relation_runtime(root, *editable)
    _switches(monkeypatch, False)
    expected = _prepare(runtime, messages, kind=kind)
    assert expected.status == "fired"
    assert expected.source_facts
    assert expected.packet is not None
    assert expected.packet.groups[0]["shown_sites"] == (1 if kind == "edit" else 2)

    _switches(monkeypatch, True)
    assert _prepare(runtime, messages, kind=kind) == expected
    index_cache = getattr(runtime, reuse.CACHE_ATTR)
    if cache_defect == "incomplete":
        index_cache.index.scan_entries = None
    elif cache_defect == "corrupt":
        index_cache.structural_revision = "corrupt"

    result = _prepare(runtime, messages, kind=kind, request_id=3)
    assert result == expected
    assert json.dumps(result.messages).encode() == json.dumps(expected.messages).encode()
    if cache_defect != "none":
        assert getattr(runtime, reuse.CACHE_ATTR) is not index_cache


@pytest.mark.parametrize("carrier", ["read_page", "window"])
def test_stale_delivered_handle_does_not_route_or_acquire(relation_workspace, monkeypatch, carrier):
    from kernel_request_source_window import MODEL_WINDOW_MARKER, MODEL_WINDOW_SCHEMA
    from kernel_request_source_window import WINDOW_CONTENT_MARKER, WINDOW_END_MARKER

    root, caller = relation_workspace
    if carrier == "read_page":
        message = _page("left.py", caller)
    else:
        digest = sha256(caller).hexdigest()
        metadata = {
            "schema": MODEL_WINDOW_SCHEMA, "path": "left.py", "owner": "entry",
            "revision": f"sha256:{digest}", "sha256": digest,
            "ranges": [{"start_line": 1, "end_line": 2, "trailing_newline": True}],
            "total_lines": 2,
            "status": "COMPLETE", "continuation": {"status": "COMPLETE"},
            "absence": "NONE",
        }
        message = {"role": "user", "content": (
            f"[Current source: left.py]\n{MODEL_WINDOW_MARKER}\n{json.dumps(metadata)}\n"
            f"{WINDOW_CONTENT_MARKER}\n--- source lines 1-2 ---\n"
            f"{caller.decode()}{WINDOW_END_MARKER}")}
    _switches(monkeypatch, True)
    runtime = _relation_runtime(root, "left.py", "right.py")
    assert _prepare(runtime, [message], kind="edit").status == "fired"
    (root / "left.py").write_bytes(caller.replace(b"item =", b"left ="))

    _switches(monkeypatch, False)
    expected = _prepare(_relation_runtime(root, "left.py", "right.py"), [message], kind="edit")
    assert expected.status == "route_off"

    _switches(monkeypatch, True)
    acquisitions = []
    acquire = census.acquire_index

    def count_acquire(*args, **kwargs):
        acquisitions.append(1)
        return acquire(*args, **kwargs)

    monkeypatch.setattr(census, "acquire_index", count_acquire)
    assert _prepare(runtime, [message], kind="edit") == expected
    assert not acquisitions


def test_no_wide_index_for_mentions_or_excluded_carriers(relation_workspace, monkeypatch):
    root, caller = relation_workspace
    messages = [
        {"role": "user", "content": "left.py uses the same constructor as right.py."},
        {**_page("left.py", caller), "role": "assistant"},
        _page("tests/right.py", caller),
    ]
    runtime = _relation_runtime(root, "left.py", "right.py")
    _switches(monkeypatch, False)
    expected = _prepare(runtime, messages, kind="edit")
    assert expected.status == "route_off"

    def unexpected_acquisition(*args, **kwargs):
        pytest.fail("census acquired an index for messages without a source carrier")

    _switches(monkeypatch, True)
    monkeypatch.setattr(census, "acquire_index", unexpected_acquisition)
    assert _prepare(runtime, messages, kind="edit") == expected
