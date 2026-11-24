"""Compact edit contracts keep even explicitly declared large product sets bounded."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import edit_scope_parity as parity
import kernel_edit_affordance as affordance
from probes import react_repair as rr
from stateframe.task_row import DECLARE_ALL_LIMIT, build_row
from stateframe.workcopy import create

# Digest of parity.surfaces(50) computed at 71739465 (before the compact scope existed).
PARITY_DIGEST_50 = "bad02d163b7e0195feaa6058b57e3f218d5d9bef4b5cc8a2704d7752ae087040"
PARITY_MODE_DIGESTS_50 = {
    "whole_file": "ed224e60d251bf78fd930ed6f638d0b3e3f7c46655ff8179db7f6a95a0bfa826",
    "declared_unmeasured": "c80af611df3ca2079315556f79956316ac949590f0dcdd88284cde12fcb7cff1",
    "entry_function": "6dc7c0709271524061ca7830e67e17fbdf76a0b4ae69347babc62dc74bfd47ce",
    "top_level": "9082fba99d73b00f7e21a1dc6e5a7014c0b1ed40388d148afa251805fbccdd4c",
    "qualified": "5efc8cc7b793fbe498a9911bd48ae0e0cd82cfd92eccca216488f0ba662fb811",
}
EDIT_SCOPE_BOUND = 4096


@pytest.fixture(autouse=True)
def _affordance_env(monkeypatch):
    monkeypatch.setenv("KERNEL_EDIT_AFFORDANCE", "1")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")


def test_limits_stay_in_step():
    assert affordance.COMPACT_EDIT_SCOPE_LIMIT == DECLARE_ALL_LIMIT == rr._SNAPSHOT_STAT_LIMIT


def test_small_whole_repo_surface_is_byte_identical_to_71739465():
    result = parity.surfaces(50)
    assert {name: parity.digest(value) for name, value in result.items()} == PARITY_MODE_DIGESTS_50
    assert parity.digest(result) == PARITY_DIGEST_50


def test_limit_sized_set_is_still_listed():
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=parity.paths_for(DECLARE_ALL_LIMIT))
    assert "editable_all" not in contract and len(contract["editable_paths"]) == DECLARE_ALL_LIMIT


@pytest.mark.parametrize("mode", ["whole_file", "declared_unmeasured", "entry_function",
                                  "top_level", "qualified"])
def test_6800_file_edit_scope_is_bounded_in_every_mode(mode):
    surfaces = parity.surfaces(6800)[mode]
    contract = surfaces["contract"]
    assert len(parity.serialize(contract)) < EDIT_SCOPE_BOUND
    assert len(parity.serialize(surfaces["frame_True"])) < EDIT_SCOPE_BOUND
    assert len(parity.serialize(surfaces["frame_False"])) < EDIT_SCOPE_BOUND
    for key in ("tools_True", "tools_False"):
        assert len(parity.serialize(surfaces[key])) < EDIT_SCOPE_BOUND
    assert contract["editable_all"] is True and contract["editable_count"] == 6800
    assert contract["editable_paths"] == []
    assert len(contract["editable_sample"]) == affordance.EDIT_SCOPE_SAMPLE_SIZE
    authority = contract["edit_authority"]
    assert authority["editable_all"] is True and authority["editable_count"] == 6800
    assert authority["editable_paths"] == []


def test_6800_file_contract_never_asks_for_per_file_measurement():
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=parity.paths_for(6800))
    assert contract["mode"] == "declared_product_files"
    assert contract["whole_file_eligibility"]["eligible"] is False
    assert contract["whole_file_eligibility"]["files_count"] == 6800


def test_deep_file_is_still_authorized_by_membership():
    paths = parity.paths_for(6800)
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=paths)
    deep = "pkg/sub16/mod_06799.py"
    row = {"files": {path: {"parent_src": ""} for path in paths}}
    assert deep not in contract["editable_sample"]
    for holder in (contract, contract["edit_authority"]):
        assert affordance.path_editable(holder, deep, row=row)
        assert affordance.path_editable(holder, deep, product_paths=frozenset(paths))
        assert not affordance.path_editable(holder, "pkg/other/new.py", row=row)
        assert not affordance.path_editable(holder, "tests/test_x.py", row=row)
        assert affordance.resolve_editable_paths(holder, row) == sorted(paths)


def test_listed_scope_membership_is_unchanged():
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=["a.py", "b/c.py"])
    assert affordance.path_editable(contract, "b/c.py")
    assert not affordance.path_editable(contract, "z.py")
    assert affordance.resolve_editable_paths(contract["edit_authority"]) == ["a.py", "b/c.py"]


def test_str_replace_description_has_no_path_list_when_compact():
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=parity.paths_for(6800))
    tools = affordance.tool_surface(json.loads(json.dumps(parity.BASE_TOOLS)), contract,
                                    authority=False)
    text = next(t["function"]["description"] for t in tools if t["function"]["name"] == "str_replace")
    assert "6800" in text and "pkg/sub" not in text


def test_downstream_editable_path_consumers_see_the_product_set(tmp_path):
    from kernel_green_lock import _editable_product_paths
    from kernel_read_reuse_visibility import editable_read_paths
    paths = parity.paths_for(DECLARE_ALL_LIMIT + 50)
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=paths)
    runtime = SimpleNamespace(
        edit_contract=contract,
        owner=SimpleNamespace(task_row={"files": {p: {"parent_src": ""} for p in paths}},
                              root=tmp_path, edit_contract=None))
    assert _editable_product_paths(runtime) == tuple(sorted(paths))
    assert editable_read_paths(runtime) == set(paths)


def test_snapshot_reuses_unchanged_files_and_sees_changes(tmp_path):
    names = [f"m{index:04d}.py" for index in range(rr._SNAPSHOT_STAT_LIMIT + 10)]
    for name in names:
        (tmp_path / name).write_text(f"X = '{name}'\n")
    first = rr._source_snapshot(tmp_path, names)
    plain = {name: (tmp_path / name).read_text() for name in names}
    assert dict(first) == plain
    assert rr._source_map_digest(first) == rr._source_map_digest(plain)
    second = rr._source_snapshot(tmp_path, names)
    assert all(first[name] is second[name] for name in names)  # no re-read of unchanged files
    (tmp_path / names[3]).write_text("X = 'changed content'\n")
    (tmp_path / names[4]).unlink()
    third = rr._source_snapshot(tmp_path, names)
    assert third[names[3]] == "X = 'changed content'\n" and third[names[4]] is None
    assert rr._source_map_digest(third) != rr._source_map_digest(second)
    assert rr._source_map_digest(third) == rr._source_map_digest(dict(third))


def test_small_snapshot_stays_a_plain_dict(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n")
    snap = rr._source_snapshot(tmp_path, ["a.py", "gone.py"])
    assert type(snap) is dict and snap == {"a.py": "A = 1\n", "gone.py": None}


def test_receipt_paths_are_bounded_over_the_limit_only():
    small = [f"a{i}.py" for i in range(DECLARE_ALL_LIMIT)]
    assert rr._bounded_receipt_paths(small) == {"paths": small}
    big = [f"a{i}.py" for i in range(6800)]
    bounded = rr._bounded_receipt_paths(big)
    assert bounded["paths_count"] == 6800 and len(bounded["paths"]) == DECLARE_ALL_LIMIT


def _large_repo_with_tests(tmp_path, extra_tests):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    for index in range(DECLARE_ALL_LIMIT + 5):
        (repo / "pkg" / f"mod_{index:03d}.py").write_text(f"VALUE = {index}\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_main.py").write_text("def test_a():\n    assert True\n")
    for index in range(extra_tests):
        (repo / "tests" / f"test_extra_{index:03d}.py").write_text("X = 1\n")
    (repo / "tests" / "conftest.py").write_text("")
    return create(repo, tmp_path / "copy")


@pytest.mark.parametrize("runner_kind", ["pytest", "generic"])
def test_large_uninferred_row_starts_read_only_discovery(tmp_path, runner_kind):
    work = _large_repo_with_tests(tmp_path, 30)
    row = build_row(
        work, task="Something is broken.", checks=["tests/test_main.py"],
        runner_kind=runner_kind,
        test_command=("python", "-m", "pytest") if runner_kind == "generic" else (),
    )

    assert row["scope_source"] == "discovery"
    assert row["scoped_edits"] is True
    assert row["files"] == {}
    assert row["entry"]["path"] == ""
    assert row["path"] == ""
    assert row["new_file_roots"] == []


def test_explicit_scope_still_builds_a_large_repo_row(tmp_path):
    work = _large_repo_with_tests(tmp_path, 3)
    row = build_row(work, task="Something is broken.", checks=["tests/test_main.py"],
                    target="pkg/mod_000.py")
    assert row["scope_source"] == "explicit"
    assert list(row["files"]) == ["pkg/mod_000.py"]
    assert list(row["public_tests"]) == ["tests/test_main.py"]


def test_inferred_scope_contract_only_authorizes_the_named_file(tmp_path):
    work = _large_repo_with_tests(tmp_path, 0)
    row = build_row(work, task="Something is broken in pkg/mod_000.py.",
                    checks=["tests/test_main.py"])
    assert row["scope_source"] == "inferred" and row["scoped_edits"] is True
    declared = [p for p in row["files"] if p.endswith(".py")]
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=declared)
    assert len(parity.serialize(contract)) < EDIT_SCOPE_BOUND
    assert affordance.path_editable(contract, "pkg/mod_000.py", row=row)
    assert not affordance.path_editable(contract, "pkg/mod_204.py", row=row)


# --- Compact-scope follow-ups: bundle targets, whole-set membership, widening, racy stat cache,
# --- and the repair-packet request audit. ---

def _compact_contract_and_row(extra=50):
    paths = parity.paths_for(DECLARE_ALL_LIMIT + extra)
    contract = affordance.contract(multi_file=True, entry_path=None, entry_fn=None,
                                   product_paths=paths)
    row = {"files": {p: {"parent_src": ""} for p in paths}}
    return paths, contract, row


def test_widened_path_key_is_the_react_repair_spelling():
    assert affordance.WIDENED_PATHS_KEY == rr.EDIT_SCOPE_WIDENED_KEY


def test_compact_row_yields_nonempty_bundle_targets_for_input_preservation():
    paths, contract, row = _compact_contract_and_row()
    assert "editable_paths" not in row and contract.get("editable_all") is True
    targets = affordance.bundle_editable_targets(row, contract)
    assert targets and set(targets) == set(paths)
    # A row that carries its own list still wins, exactly as before.
    assert affordance.bundle_editable_targets({"editable_paths": ["pkg/a.py"]}, contract) == ["pkg/a.py"]


def test_whole_set_repair_packet_marks_only_declared_files_editable():
    import kernel_obligation_assess as obligations
    paths, contract, row = _compact_contract_and_row()
    declared, undeclared = paths[0], "pkg/undeclared_module.py"
    packet = obligations.repair_packet(
        {"focus_obligation": "obl-1"},
        edit_scope=contract, task_row=row,
        source_bytes={declared: "A = 1\n", undeclared: "B = 2\n"})
    editable = {item["path"]: item["editable"] for item in packet["current_source"]["files"]}
    assert editable == {declared: True, undeclared: False}


def test_widened_path_is_in_resolved_list_under_whole_set():
    _paths, contract, row = _compact_contract_and_row()
    widened = "docs_extra/widened_module.py"
    row["_edit_scope_widened"] = [widened]
    assert widened in affordance.resolve_editable_paths(contract, row)
    assert affordance.path_editable(contract, widened, row=row)
    assert not affordance.path_editable(contract, "docs_extra/other_module.py", row=row)


def test_racy_same_size_rewrite_with_same_mtime_is_reread(tmp_path):
    target = tmp_path / "racy.py"
    target.write_text("X = 'aaaa'\n")
    mtime_ns = os.stat(target).st_mtime_ns
    first = rr._stat_source_snapshot(tmp_path, ["racy.py"])
    target.write_text("X = 'bbbb'\n")  # same size, same inode
    os.utime(target, ns=(mtime_ns, mtime_ns))
    second = rr._stat_source_snapshot(tmp_path, ["racy.py"])
    assert second["racy.py"] == "X = 'bbbb'\n"
    assert rr._source_map_digest(second) != rr._source_map_digest(first)


def test_stat_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_SNAPSHOT_CACHE", {})
    monkeypatch.setattr(rr, "_SNAPSHOT_CACHE_LIMIT", 3)
    names = [f"c{index}.py" for index in range(5)]
    for name in names:
        (tmp_path / name).write_text(f"V = '{name}'\n")
    snapshot = rr._stat_source_snapshot(tmp_path, names)
    assert dict(snapshot) == {name: f"V = '{name}'\n" for name in names}
    assert len(rr._SNAPSHOT_CACHE) <= 3


def test_request_audit_checks_only_request_paths_by_membership(monkeypatch):
    from types import SimpleNamespace as NS
    import kernel_stateframe_runtime as runtime_mod
    paths, contract, row = _compact_contract_and_row()
    declared, undeclared = paths[0], "pkg/undeclared_module.py"
    import kernel_obligation_assess as obligations
    packet = obligations.repair_packet(
        {"focus_obligation": "obl-1"}, edit_scope=contract, task_row=row,
        source_bytes={declared: "A = 1\n", undeclared: "B = 2\n"})
    emitted = []
    fake = NS(stateframe_host_closure_mode="on", owner=NS(task_row=row),
              journal=NS(emit=lambda name, **kw: emitted.append(kw)))
    tools = [{"function": {"name": "edit", "description": affordance.AUTHORITY_EDIT_DESCRIPTION}}]
    runtime_mod.StateFrameRuntime._audit_repair_packet_request(
        fake, {"repair_packet": packet}, [{"role": "user", "content": "x"}], tools, "rid-1")
    audit = emitted[0]["audit"]
    assert audit["edit_tool_allows_path"] == {declared: True, undeclared: False}
