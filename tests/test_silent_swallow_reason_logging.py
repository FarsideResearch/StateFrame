"""Fault-injection coverage for the first reported silent fallback sites."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_bolt as bolt
import kernel_callsite_census as census
import kernel_editready_fact as editready
import kernel_green_lock as green_lock
import kernel_reason_log as reason_log
import kernel_repo_intel as repo_intel
import kernel_repo_signals as repo_signals
import kernel_request_diet as request_diet


@pytest.fixture(autouse=True)
def isolated_reason_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv(reason_log.LEVEL_ENV, raising=False)
    monkeypatch.delenv(reason_log.PROBLEMS_ENV, raising=False)
    reason_log.reset_for_tests()
    problems = tmp_path / "problems.jsonl"
    reason_log.configure(level="WARN", problems_path=problems, stream=io.StringIO())
    yield problems
    reason_log.reset_for_tests()


def _assert_reported_once(problems: Path, cause_key: str, count: int) -> None:
    entry = reason_log.run_summary()["by_cause"][f"fallback:{cause_key}"]
    assert entry["count"] == count
    assert entry["first"]["dropped"]
    rows = [json.loads(line) for line in problems.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["cause_key"] == cause_key
    assert rows[0]["fields"]["dropped"]


def test_symbol_view_skips_bad_sources_and_logs_the_cause_once(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    import kernel_turn_facts

    for name in ("healthy.py", "broken_a.py", "broken_b.py"):
        (tmp_path / name).write_text("pass\n", encoding="utf-8")

    def symbols(path: bytes, _source: bytes):
        name = os.fsdecode(path)
        if name.startswith("broken_"):
            raise OSError("unavailable")
        return {"healthy_symbol": ({"path": name},)}

    monkeypatch.setattr(kernel_turn_facts, "_symbols", symbols)
    published = []
    bolt._build_grep_symbol_view(tmp_path, False, published.append, lambda: True)

    assert len(published) == 1
    assert "healthy_symbol" in published[0].definitions
    assert "broken_a_symbol" not in published[0].definitions
    _assert_reported_once(isolated_reason_log, "grep_symbol_source_unavailable", 2)


def test_census_revision_failure_keeps_cache_reuse_disabled(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    monkeypatch.setattr(census, "structural_revision", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError()))
    index = SimpleNamespace(scan_stable=True, scan_entries=())

    assert census._census_revision_cache_key(
        index, tmp_path, parser_version="test", priority_paths=(),
    ) is None
    assert census._census_revision_cache_key(
        index, tmp_path, parser_version="test", priority_paths=(),
    ) is None
    _assert_reported_once(isolated_reason_log, "callsite_census_revision_unavailable", 2)


def test_absolute_editready_path_failure_still_omits_the_path(
        tmp_path: Path, isolated_reason_log: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"

    assert editready._normalize_repo_path(str(outside), tmp_path) is None
    assert editready._normalize_repo_path(str(outside), tmp_path) is None
    _assert_reported_once(isolated_reason_log, "editready_absolute_path_resolution_failed", 2)


def test_editready_symlink_escape_still_omits_the_path(
        tmp_path: Path, isolated_reason_log: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    assert editready._normalize_repo_path("escape/secret.py", tmp_path) is None
    assert editready._normalize_repo_path("escape/secret.py", tmp_path) is None
    _assert_reported_once(isolated_reason_log, "editready_path_containment_failed", 2)


def test_workspace_digest_consumer_failure_still_returns_none(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    def fail_digest(_paths):
        raise OSError("unavailable")

    monkeypatch.setattr(green_lock, "_workspace_consumers", lambda _root: SimpleNamespace(digests=fail_digest))
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))

    assert green_lock._workspace_product_file_digests(runtime, ("app.py",)) is None
    assert green_lock._workspace_product_file_digests(runtime, ("app.py",)) is None
    _assert_reported_once(isolated_reason_log, "green_lock_workspace_digest_unavailable", 2)


def test_suppression_journal_failure_keeps_the_suppression_decision(
        isolated_reason_log: Path) -> None:
    class BrokenJournal:
        calls = 0

        def emit(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("closed")

    journal = BrokenJournal()
    runtime = SimpleNamespace(journal=journal)
    comparison = (False, "rev-1", "digest")

    assert green_lock.note_unchanged_parent_green_suppressed(
        runtime, reason="unchanged", comparison=comparison,
    ) is None
    assert green_lock.note_unchanged_parent_green_suppressed(
        runtime, reason="unchanged", comparison=comparison,
    ) is None
    assert journal.calls == 1
    assert runtime._postgreen_unchanged_parent_suppression_revisions == {"rev-1"}
    _assert_reported_once(isolated_reason_log, "green_lock_suppression_event_failed", 1)


def test_product_source_read_failure_still_omits_invalid_text(
        tmp_path: Path, isolated_reason_log: Path) -> None:
    (tmp_path / "broken.py").write_bytes(b"\xff")

    assert green_lock._read_product_sources(tmp_path, {"broken.py"}) == {}
    assert green_lock._read_product_sources(tmp_path, {"broken.py"}) == {}
    _assert_reported_once(isolated_reason_log, "green_lock_product_source_read_failed", 2)


def test_intake_cache_lookup_failure_still_skips_the_cached_map(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    class BrokenCache:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def reads_available():
            return True

        @staticmethod
        def lookup(*_args, **_kwargs):
            raise OSError("unavailable")

    monkeypatch.setitem(sys.modules, "stateframe.send1_probe_cache", BrokenCache)
    monkeypatch.setitem(sys.modules, "kernel_tool_tape", SimpleNamespace(replay_mode_active=lambda: False))
    monkeypatch.delenv("STATEFRAME_CHECKPOINT_RESTORE", raising=False)

    assert repo_intel._observed_intake_map(tmp_path, "", "") is None
    assert repo_intel._observed_intake_map(tmp_path, "", "") is None
    _assert_reported_once(isolated_reason_log, "repo_intel_intake_cache_lookup_failed", 2)


def test_intake_map_generation_failure_still_skips_the_map(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    class Observation:
        @staticmethod
        def policy():
            return "test-policy"

        def __init__(self, _root: Path) -> None:
            pass

    class Cache:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def reads_available():
            return True

        @staticmethod
        def lookup(*_args, **_kwargs):
            return None

    monkeypatch.setattr(repo_intel._repo_map_module, "IntakeMapObservation", Observation)

    def fail_generation(*_args, **_kwargs):
        raise repo_intel._repo_map_module.IntakeObservationUnavailable

    monkeypatch.setattr(repo_intel._repo_map_module, "_generate_repo_map_with_stats", fail_generation)
    monkeypatch.setitem(sys.modules, "stateframe.send1_probe_cache", Cache)
    monkeypatch.setitem(sys.modules, "kernel_tool_tape", SimpleNamespace(replay_mode_active=lambda: False))
    monkeypatch.delenv("STATEFRAME_CHECKPOINT_RESTORE", raising=False)

    assert repo_intel._observed_intake_map(tmp_path, "", "") is None
    assert repo_intel._observed_intake_map(tmp_path, "", "") is None
    _assert_reported_once(isolated_reason_log, "repo_intel_intake_map_unavailable", 2)


def test_inventory_rejects_outside_entries_and_logs_the_reason(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_reason_log: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        repo_signals.os, "walk",
        lambda *_args, **_kwargs: iter([(str(outside.parent), [], [outside.name])]),
    )

    assert repo_signals._inventory_paths(tmp_path, source_only=True) == []
    assert repo_signals._inventory_paths(tmp_path, source_only=True) == []
    _assert_reported_once(isolated_reason_log, "repo_signals_inventory_path_outside_root", 2)


def test_source_decode_failure_still_returns_empty_text(
        tmp_path: Path, isolated_reason_log: Path) -> None:
    source = tmp_path / "broken.py"
    source.write_bytes(b"\xff")

    assert repo_signals._read_source(source) == ""
    assert repo_signals._read_source(source) == ""
    _assert_reported_once(isolated_reason_log, "repo_signals_source_read_failed", 2)


def test_invalid_tokenizer_endpoint_still_returns_none(
        isolated_reason_log: Path) -> None:
    assert request_diet._tokenize_url_for_endpoint("http://[::1") is None
    assert request_diet._tokenize_url_for_endpoint("http://[::1") is None
    _assert_reported_once(isolated_reason_log, "tokenizer_endpoint_url_parse_failed", 2)
