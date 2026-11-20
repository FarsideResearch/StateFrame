"""Per-turn work accounting on a repository-sized, deterministic input."""
from __future__ import annotations

import importlib
import json
import time
from dataclasses import replace
from pathlib import Path

import kernel_env_seed
import kernel_measurement
import kernel_ship_tree_smoke
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession
from repo_intel_core import call_graph

FILE_COUNT = 5_000
PAYLOAD = b"from pathlib import Path\n"


def test_package_and_flat_imports_share_counter_context():
    flat = importlib.import_module("kernel_turn_work_counters")
    packaged = importlib.import_module("stateframe.engine.kernel_turn_work_counters")
    assert packaged is flat


def _source_tree(root: Path) -> None:
    root.mkdir()
    for index in range(FILE_COUNT):
        (root / f"module_{index:04d}.py").write_bytes(PAYLOAD)


def test_turn_counts_actual_repository_work_and_meets_wall_budget(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _source_tree(source)
    graph_root = tmp_path / "empty_graph"
    graph_root.mkdir()
    monkeypatch.setattr(call_graph, "repo_cache_file", lambda _root: graph_root / "cache.json")
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "on")
    recorder = kernel_measurement.TurnTimingRecorder()
    recorder.note_response({"request_id": "first", "step": 1, "elapsed_ns": 0})
    started = time.perf_counter()
    try:
        fingerprint = kernel_env_seed._tree_fingerprint(source)
        source_digest = kernel_env_seed._source_digest(source)
        imports = kernel_ship_tree_smoke._top_level_import_names("import pathlib")
        graph = call_graph._build_graph_uncached(str(graph_root), use_cache=False,
                                                 use_lsp=False)
    finally:
        elapsed = time.perf_counter() - started
        record = recorder.finish_submission({"request_id": "second"})
    assert fingerprint[0] == FILE_COUNT
    assert len(source_digest) == 64
    assert imports == ()
    assert isinstance(graph, call_graph.CodeGraph)
    assert record is not None
    assert record["work_counters"] == {
        "files_read": FILE_COUNT,
        "files_statted": 2 * FILE_COUNT + 1,
        "files_hashed": FILE_COUNT,
        "parses": 1,
        "index_builds": 1,
        "subprocess_launches": 0,
        "bytes_hashed": FILE_COUNT * len(PAYLOAD),
    }
    assert elapsed < 10.0, f"repository work took {elapsed:.3f}s"


def test_timing_off_keeps_request_bytes_and_allocates_no_counters(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.py").write_bytes(PAYLOAD)
    body = {"messages": [{"role": "user", "content": "check"}], "tools": []}
    before = kernel_measurement.canonical_bytes(body)
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "off")
    recorder = kernel_measurement.TurnTimingRecorder()
    recorder.note_response({"request_id": "first", "step": 1})
    kernel_env_seed._tree_fingerprint(source)
    kernel_env_seed._source_digest(source)
    assert recorder.finish_submission({"request_id": "second"}) is None
    assert kernel_measurement.canonical_bytes(body) == before
    assert json.loads(before) == body


def test_process_launch_is_counted_once_per_supervisor(tmp_path, monkeypatch):
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    capability = shell_capability(
        workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
        scratch_home=tmp_path / "home", network="none", writable=False,
    )
    capability = replace(capability, sandboxed=False)
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "on")
    recorder = kernel_measurement.TurnTimingRecorder()
    recorder.note_response({"request_id": "first", "step": 1})
    try:
        with ExecSession(tmp_path / "run") as session:
            first = session.run(["true"], capability)
            second = session.run(["true"], capability)
    finally:
        record = recorder.finish_submission({"request_id": "second"})
    assert first.exit_code == second.exit_code == 0
    assert record is not None
    assert record["work_counters"]["subprocess_launches"] == 1


def test_import_root_resolution_counts_one_read_parse_and_index(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "entry.py").write_text("import missing_package\n", encoding="utf-8")
    (root / "first").mkdir()
    (root / "second").mkdir()
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "on")
    recorder = kernel_measurement.TurnTimingRecorder()
    recorder.note_response({"request_id": "first", "step": 1})
    try:
        roots, internal_only = kernel_ship_tree_smoke._source_roots(
            root, ("entry.py",), "",
        )
    finally:
        record = recorder.finish_submission({"request_id": "second"})
    assert roots == (str(root),)
    assert internal_only
    assert record is not None
    assert record["work_counters"] == {
        "files_read": 1,
        "files_statted": 7,
        "files_hashed": 0,
        "parses": 1,
        "index_builds": 1,
        "subprocess_launches": 0,
        "bytes_hashed": 0,
    }
