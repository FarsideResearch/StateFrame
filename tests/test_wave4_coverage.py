import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import kernel_named_layer_coverage as layers
from kernel_coverage_reminders import compact_state
from kernel_stateframe_notice_dedup import (
    prepare_with_commit,
    render_coverage_notice,
    state_snapshot,
)
from test_rp3_coverage_work import _admit, _reports, _runtime


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "off")
    gate = ModuleType("stateframe.send1_wave4")
    gate.enabled = lambda: True
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    return gate


def test_current_reports_share_rows_and_keep_live_evidence(tmp_path, monkeypatch, pilot):
    source = tmp_path / "api.py"
    source.write_text("def fetch(): pass\n")
    check = tmp_path / "test_api.py"
    check.write_text("assert status == 400\n")
    runtime, events, _ = _runtime(
        tmp_path, "Update API `api.fetch` to return HTTP 422.")
    _admit(runtime)

    calls = Counter()
    compute_rows = layers._compute_layer_rows
    defines = layers._defines
    read_bytes = Path.read_bytes

    def count_rows(*args):
        calls["rows"] += 1
        return compute_rows(*args)

    def count_match(*args):
        calls["match"] += 1
        return defines(*args)

    def count_read(path):
        if path == source:
            calls["read"] += 1
        return read_bytes(path)

    monkeypatch.setattr(layers, "_compute_layer_rows", count_rows)
    monkeypatch.setattr(layers, "_defines", count_match)
    monkeypatch.setattr(Path, "read_bytes", count_read)

    first = runtime.layer_coverage_report()
    assert calls == {"rows": 1, "match": 1, "read": 1}
    assert first["rows"][0]["files"] == ["api.py"]
    assert first["public_test_evidence"][0]["literal_present"] is False
    first["rows"][0]["files"].clear()

    check.write_text("assert status == 422\n")
    runtime.owner.task_row["public_tests"] = {check.name: check.read_text()}
    named, completion = _reports(runtime)
    assert calls == {"rows": 1, "match": 1, "read": 3}
    assert named["rows"][0]["files"] == ["api.py"]
    assert named["public_test_evidence"][0]["literal_present"] is True
    assert completion["assertion_count"] == 1
    assert completion["untouched_named_layers"] == named["rows"]
    assert len(events) == 2

    restored = pickle.loads(pickle.dumps(runtime._coverage_file_cache))
    runtime._coverage_file_cache = restored
    assert runtime.layer_coverage_report() == named
    assert calls == {"rows": 2, "match": 2, "read": 4}


def test_current_rows_recheck_source_availability_and_proof_inputs(tmp_path, monkeypatch, pilot):
    source = tmp_path / "one.py"
    source.write_text("def fetch(): pass\n")
    cache = layers.CoverageFileCache()
    start = layers.snapshot_workspace(tmp_path)
    current = dict(start)
    ticket = "Update `fetch`."

    def report():
        with layers.use_file_cache(tmp_path, cache):
            return layers.build_report(ticket, tmp_path, start_snapshot=start,
                                       current_snapshot=current)

    def verify(*, confidence, evidence=None, diff=None):
        pilot.enabled = lambda: True
        actual = report()
        pilot.enabled = lambda: False
        assert json.dumps(actual) == json.dumps(report())
        row, = actual["rows"]
        assert row["confidence"] == confidence
        if evidence:
            assert evidence in row["evidence"]
        assert row["diff_from_start"] is diff

    verify(confidence="confident", diff=False)
    start["one.py"] = None
    verify(confidence="CHECK_INCONCLUSIVE", evidence="hash-read-failed")
    del start["one.py"]
    verify(confidence="CHECK_INCONCLUSIVE", evidence="hash-missing")
    start.update(current)

    # The caller's snapshot is a diff proof, even if the independently read
    # definition changed to a different symbol of the same size.
    source.write_text("def patch(): pass\n")
    verify(confidence="UNKNOWN", evidence="unbound")
    source.write_bytes(b"\xffef fetch(): pass\n")
    verify(confidence="CHECK_INCONCLUSIVE", evidence="binding-read-failed")
    source.write_text("def fetch(): pass\n")

    with monkeypatch.context() as unavailable:
        read_bytes = Path.read_bytes

        def fail_read(path):
            if path == source:
                raise PermissionError(path)
            return read_bytes(path)

        unavailable.setattr(Path, "read_bytes", fail_read)
        read_text = Path.read_text

        def fail_read_text(path, *args, **kwargs):
            if path == source:
                raise PermissionError(path)
            return read_text(path, *args, **kwargs)

        unavailable.setattr(Path, "read_text", fail_read_text)
        verify(confidence="CHECK_INCONCLUSIVE", evidence="binding-read-failed")
    verify(confidence="confident", diff=False)
    source.unlink()
    verify(confidence="UNKNOWN", evidence="unbound")

    # Alias names never become module qualifiers. Retargeting a link must
    # still expose only the actual in-tree product path.
    (tmp_path / "real.py").write_text("def fetch(): pass\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def fetch(): pass\n")
    link = tmp_path / "alias.py"
    link.symlink_to(outside)
    ticket = "Update `alias.fetch`."
    verify(confidence="UNKNOWN", evidence="unbound")
    link.unlink()
    link.symlink_to("real.py")
    start.update(layers.snapshot_workspace(tmp_path))
    current.update(start)
    verify(confidence="UNKNOWN", evidence="unbound")
    ticket = "Update `real.fetch`."
    verify(confidence="confident", diff=False)


def test_cached_coverage_rechecks_all_sources_after_the_last_read(tmp_path, monkeypatch, pilot):
    first = tmp_path / "first.py"
    first.write_text("def fetch(): pass\n")
    last = tmp_path / "last.py"
    last.write_text("def unrelated(): pass\n")
    start = layers.snapshot_workspace(tmp_path)
    cache = layers.CoverageFileCache()

    def report():
        with layers.use_file_cache(tmp_path, cache):
            return layers.build_report("Update `fetch`.", tmp_path, start_snapshot=start,
                                       current_snapshot=start)

    assert report()["rows"][0]["confidence"] == "confident"
    read_bytes = Path.read_bytes
    changed = False

    def read_then_change(path):
        nonlocal changed
        data = read_bytes(path)
        if path == last and not changed:
            changed = True
            first.write_text("def renamed(): pass\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    result = report()
    assert changed
    assert result["rows"][0]["confidence"] == "UNKNOWN"
    pilot.enabled = lambda: False
    assert result == report()


def test_current_report_notice_delivery_and_replay(tmp_path, monkeypatch, pilot):
    package = tmp_path / "package"
    package.mkdir()
    api = package / "api.py"
    api.write_text("def fetch(): pass\n")
    cli = package / "cli.py"
    cli.write_text("def audit(): pass\n")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    check = tmp_path / "test_api.py"
    check.write_text("assert status == 400\n")
    guarded = "Do not change CLI `package.cli.audit`."
    ticket = f"Update API `package.api.fetch` to return HTTP 422. {guarded}"
    rules = layers._LAYER_RULES

    def run(enable):
        pilot.enabled = lambda: enable
        runtime, events, binding = _runtime(tmp_path, ticket)
        runtime.stateframe_notice_dedup_mode = "on"
        runtime.product_revision = "start"
        _admit(runtime)

        def reports():
            runtime.owner.task_row["public_tests"] = {check.name: check.read_text()}
            return _reports(runtime)

        def prepare(*, deliver):
            def assemble(owner, messages, **kwargs):
                named, _ = reports()
                text = render_coverage_notice(owner, {"coverage_reminders": compact_state(named)})
                previews.append(text.encode())
                return SimpleNamespace(messages=[{"role": "user", "content": text}] if deliver else [])

            prepare_with_commit(runtime, assemble, [], uncapped=False, tools=[])
            return state_snapshot(runtime)

        previews = []
        dropped = prepare(deliver=False)
        delivered = prepare(deliver=True)
        assert dropped["described"] == []
        assert delivered["described"]
        assert guarded.encode() in previews[0]
        assert previews[0] == previews[1]
        baseline = reports()
        assert baseline[0]["review_reminders"][0]["target_files"] == ["package/cli.py"]
        runtime.stateframe_done_challenge_once_mode = "on"
        runtime.completion_menu = ("read", "done")
        assessment = layers.precommit_assessment(runtime, admissible=True)
        assert assessment["repair_action"]["arguments"] == {"path": "package/api.py"}
        assert events[-1]["kind"] == "named_layer_coverage_evidence"

        moved = alternate / api.name
        api.rename(moved)
        binding.update(product_revision="moved", code_hash="moved")
        runtime.product_revision = "moved"
        relocated = reports()
        assert relocated[0]["rows"][0]["files"] == ["UNKNOWN"]
        runtime.owner.task_row["ticket_text"] = ticket.replace("package.api", "alternate.api")
        check.write_text("assert status == 422\n")
        scoped = reports()
        assert scoped[0]["rows"][0]["files"] == ["alternate/api.py"]
        assert scoped[0]["public_test_evidence"][0]["literal_present"] is True

        runtime.test_receipts.append({
            "clause": guarded, "status": "PASS", "product_revision": "moved",
        })
        checked = reports()
        assert checked[0]["review_reminders"][0]["clause_check_recorded"] is True
        runtime.test_receipts[-1]["product_revision"] = "start"
        stale = reports()
        assert stale[0]["review_reminders"][0]["clause_check_recorded"] is False

        # Change evidence patterns without changing the ticket, files or
        # selected layer. The warm report must adopt the new row evidence.
        amended_rules = tuple((name, (*patterns, r"\bUpdate\b")) if name == "API"
                              else (name, patterns) for name, patterns in rules)
        with monkeypatch.context() as policy:
            policy.setattr(layers, "_LAYER_RULES", amended_rules)
            amended = reports()
            assert r"\bUpdate\b" in amended[0]["rows"][0]["evidence"]
            active_ticket = runtime.owner.task_row["ticket_text"]
            replay, replay_events, replay_binding = _runtime(tmp_path, active_ticket)
            replay_events.extend(events)
            replay_binding.update(binding)
            replay.owner.task_row.update(runtime.owner.task_row)
            replay.test_receipts.extend(runtime.test_receipts)
            _admit(replay, resume=object())
            assert _reports(replay) == amended
            assert replay_events == events
        moved.rename(api)
        check.write_text("assert status == 400\n")
        return json.dumps((events, baseline, assessment, relocated, scoped, checked, stale, amended,
                           dropped, delivered)), previews

    assert run(True) == run(False)
