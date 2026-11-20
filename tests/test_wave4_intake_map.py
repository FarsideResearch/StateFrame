import copy
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import kernel_repo_intel as intel
import kernel_tool_tape as tape
from repo_intel_core import repo_map


@pytest.fixture
def intake(tmp_path, monkeypatch):
    cache = ModuleType("stateframe.send1_probe_cache")
    cache.active = False
    cache.unavailable = False
    cache.records = {}
    cache.lookups = []
    cache.enabled = lambda: cache.active

    def lookup(kind, fingerprint, *, validate):
        cache.lookups.append((kind, fingerprint))
        if cache.unavailable:
            raise OSError("store unavailable")
        facts = cache.records.get((kind, fingerprint))
        return copy.deepcopy(facts) if validate(facts) else None

    def publish(kind, fingerprint, facts, *, validate):
        assert validate(facts)
        cache.records[kind, fingerprint] = copy.deepcopy(facts)

    cache.lookup, cache.publish = lookup, publish
    monkeypatch.setitem(sys.modules, cache.__name__, cache)
    monkeypatch.setattr(intel.background, "enabled", lambda: False)
    monkeypatch.setattr(repo_map, "_TREE_SITTER_AVAILABLE", False)
    monkeypatch.setattr(repo_map, "_MAX_FILES", 2)
    monkeypatch.setattr(intel, "get_current_commit_sha", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(tape, "_replay_active", False)
    monkeypatch.delenv("STATEFRAME_CHECKPOINT_RESTORE", raising=False)
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "graph-cache"))
    root = tmp_path / "repo"
    root.mkdir()

    work = SimpleNamespace(builds=0, parses=0, ranks=0, graph=0, sources=0)
    for module, function, counter in (
        (repo_map, "_generate_repo_map_with_stats", "builds"),
        (repo_map, "_file_entry_with_mode", "parses"),
        (repo_map.IntakeMapObservation, "source", "sources"),
        (intel, "bound_repo_map", "ranks"),
        (intel, "build_graph", "graph"),
    ):
        original = getattr(module, function)

        def counted(*args, _counter=counter, _original=original, **kwargs):
            setattr(work, _counter, getattr(work, _counter) + 1)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, function, counted)

    summaries = []
    monkeypatch.setattr(intel, "_journal_repo_map_receipt",
                        lambda stats, **fields: summaries.append((copy.deepcopy(stats), fields)))

    def run(on, **inputs):
        cache.active = on
        text = intel.intake_orientation(root, **inputs)
        return text, summaries[-1]

    return SimpleNamespace(root=root, run=run, cache=cache, work=work)


def _write(root, relative, source):
    path = root / relative
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(source, encoding="utf-8")


def test_verified_hit_skips_build_parse_and_rank(intake):
    _write(intake.root, "b.py", "def middle():\n    return True\n")
    _write(intake.root, "c.py", "def later():\n    return True\n")
    _write(intake.root, "z/uncounted.py", "def outside_cap():\n    return True\n")
    baseline = intake.run(False)
    assert intake.cache.lookups == []
    assert intake.run(True) == baseline
    before = vars(intake.work).copy()
    assert before == {"builds": 2, "parses": 4, "ranks": 2, "graph": 0, "sources": 2}
    assert intake.run(True) == baseline
    assert vars(intake.work) == {**before, "sources": 4}


def test_current_frontier_focus_and_ranking_match_eager(intake):
    root = intake.root
    _write(root, "b.py", "def middle_old():\n    return True\n")
    _write(root, "c.py", "def later():\n    return True\n")
    _write(root, "z/focus.py", "def focused_old():\n    return True\n")

    def compare(*, miss=True, **inputs):
        expected = intake.run(False, **inputs)
        builds = intake.work.builds
        actual = intake.run(True, **inputs)
        assert actual == expected
        assert intake.work.builds == builds + int(miss)
        assert intake.run(True, **inputs) == expected
        assert intake.work.builds == builds + int(miss)
        return actual[0]

    assert "def later" in compare()
    _write(root, "a/__init__.py", "# This path has no signatures.\n")
    _write(root, "a/early.py", "def early():\n    return True\n")
    earlier = compare()
    assert "def early" in earlier and "def later" not in earlier
    (root / "a/early.py").unlink()
    assert "def later" in compare()
    previous = (root / "b.py").stat()
    _write(root, "b.py", "def middle_new():\n    return True\n")
    os.utime(root / "b.py", ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert "def middle_new" in compare()

    inputs = {"target_path": "z/focus.py", "symbol": "focused_old"}
    assert "def focused_old" in compare(**inputs)
    _write(root, "z/focus.py", "def focused_new():\n    return True\n")
    assert "def focused_new" not in compare(miss=False)
    assert "def focused_new" in compare(**inputs)
    assert "def focused_new" in compare(target_path="z/focus.py", symbol="focused_new")
    by_symbol = compare(symbol="later")
    assert by_symbol.index("\nc.py:") < by_symbol.index("\nb.py:")
    ranked = compare(target_path="c.py", symbol="later")
    assert ranked.index("\nc.py:") < ranked.index("\nb.py:")

    assert intake.work.graph == 0
    runtime = SimpleNamespace(repo_root=root)
    result = json.loads(intel.operation(runtime, "where_defined", {"symbol": "focused_new"}))
    assert result["status"] == "FOUND", result
    assert result["definition"]["file"] == "z/focus.py"


def test_cached_map_rechecks_earlier_files_after_reading_the_focus(intake, monkeypatch):
    root = intake.root
    _write(root, "a.py", "def before_change():\n    return True\n")
    _write(root, "b.py", "def second():\n    return True\n")
    _write(root, "z/focus.py", "def focused():\n    return True\n")
    inputs = {"target_path": "z/focus.py"}
    assert "def before_change" in intake.run(True, **inputs)[0]

    read_source = repo_map.IntakeMapObservation.source
    changed = False

    def read_then_change(observation, path):
        nonlocal changed
        result = read_source(observation, path)
        if path == root / "z/focus.py" and not changed:
            changed = True
            _write(root, "a.py", "def after_change():\n    return True\n")
        return result

    monkeypatch.setattr(repo_map.IntakeMapObservation, "source", read_then_change)
    actual = intake.run(True, **inputs)
    assert changed
    assert "def after_change" in actual[0]
    assert actual == intake.run(False, **inputs)


def test_unknown_parser_mode_and_failures_use_eager_map(intake, monkeypatch):
    _write(intake.root, "entry.py", "def entry():\n    return True\n")
    intake.run(True)
    warmed = intake.cache.records.copy()

    def fail(_source):
        raise LookupError("grammar unavailable")

    monkeypatch.setattr(repo_map, "_parser_available", lambda: True)
    monkeypatch.setattr(repo_map, "_TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr(repo_map, "_get_parser", lambda _language: object())
    monkeypatch.setattr(repo_map, "_extract_python_sigs", fail)
    baseline = intake.run(False)
    assert intake.run(True) == baseline
    assert baseline[1][0]["failure_reasons"] == {"LookupError": 1}
    monkeypatch.setattr(repo_map, "_extract_python_sigs", lambda _source: ["def entry()"])
    parsed = intake.run(False)
    assert intake.run(True) == parsed
    assert parsed[1][0]["tree_sitter"] == 1
    assert intake.cache.records == warmed


def test_uncapped_membership_and_cache_fallbacks(intake, monkeypatch):
    _write(intake.root, "one.py", "def one():\n    return True\n")
    intake.run(True)
    _write(intake.root, "two.py", "def two():\n    return True\n")
    baseline = intake.run(False)
    assert intake.run(True) == baseline
    key = intake.cache.lookups[-1]
    intake.cache.records[key]["map"] = ["damaged"]
    builds = intake.work.builds
    assert intake.run(True) == baseline
    assert intake.work.builds == builds + 1
    intake.cache.unavailable = True
    assert intake.run(True) == baseline
    assert intake.work.builds == builds + 2
    intake.cache.unavailable = False

    with monkeypatch.context() as replay:
        replay.setattr(tape, "_replay_active", True)
        lookups = len(intake.cache.lookups)
        assert intake.run(True) == baseline
        assert len(intake.cache.lookups) == lookups
        assert intake.work.builds == builds + 3
    with monkeypatch.context() as restore:
        restore.setenv("STATEFRAME_CHECKPOINT_RESTORE", "checkpoint")
        assert intake.run(True) == baseline
        assert len(intake.cache.lookups) == lookups
        assert intake.work.builds == builds + 4

    (intake.root / "one.py").unlink()
    (intake.root / "one.py").symlink_to("two.py")
    records = intake.cache.records.copy()
    assert intake.run(True) == intake.run(False)
    assert intake.cache.records == records
