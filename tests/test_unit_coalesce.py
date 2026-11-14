"""STATEFRAME_UNIT_COALESCE: dynamic producer+consumer merge when a unit's gate fails at a cross-unit seam."""
from __future__ import annotations

import ast
import json
import os
import pickle
import random
import re
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import get_args, get_type_hints

import chains
import chains_recall
import pytest
import unit_planner_v2 as v2
from test_unit_planner_v2 import (
    CYC2,
    CYC3,
    MODELS,
    PKG,
    RefIndex,
    plan_for,
    row,
    symbols,
)

COALESCE_ENV = "STATEFRAME_UNIT_COALESCE"
BROKEN_MODELS = MODELS.replace("self.has_more = has_more", "self.more = has_more")
ROOT = Path(__file__).resolve().parents[1]


def _history_bytes(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError):
        pytest.fail(
            f"Unit-coalescing parity requires Git history in {root}: git {' '.join(args)} failed. "
            "Fetch the feature-adding commit and its parent (unshallow the clone if needed).",
            pytrace=False,
        )


def _coalesce_parent(root):
    added = _history_bytes(root, "log", "--diff-filter=A", "--format=%H", "-1", "--",
                           "src/stateframe/engine/unit_coalesce.py").decode().strip()
    if not added:
        pytest.fail(
            "Unit-coalescing parity cannot find the feature-adding commit; fetch complete Git history.",
            pytrace=False,
        )
    return _history_bytes(root, "rev-parse", "--verify", f"{added}^").decode().strip()


@pytest.fixture(scope="module")
def base_engine_source():
    parent = _coalesce_parent(ROOT)

    def read(filename):
        return _history_bytes(ROOT, "show", f"{parent}:src/stateframe/engine/{filename}.py")

    return read


def test_parity_base_is_parent_of_feature_addition(monkeypatch):
    calls = []
    added, parent = "a" * 40, "b" * 40

    def check_output(command, **kwargs):
        assert kwargs == {"cwd": ROOT, "stderr": subprocess.PIPE}
        calls.append(command)
        return ((added if command[1] == "log" else parent) + "\n").encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    assert _coalesce_parent(ROOT) == parent
    assert calls == [
        ["git", "log", "--diff-filter=A", "--format=%H", "-1", "--",
         "src/stateframe/engine/unit_coalesce.py"],
        ["git", "rev-parse", "--verify", added + "^"],
    ]


def test_parity_base_fails_clearly_when_addition_history_is_missing(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"")
    with pytest.raises(pytest.fail.Exception, match="feature-adding commit; fetch complete Git history"):
        _coalesce_parent(ROOT)


@pytest.mark.parametrize("operation", ["log", "rev-parse", "show"])
@pytest.mark.parametrize("missing_git", [False, True])
def test_parity_history_errors_fail_instead_of_skipping(monkeypatch, operation, missing_git):
    def check_output(command, **kwargs):
        if command[1] == operation:
            if missing_git:
                raise FileNotFoundError("git unavailable")
            raise subprocess.CalledProcessError(128, command, stderr=b"history unavailable")
        return ("a" * 40 + "\n").encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    with pytest.raises(pytest.fail.Exception, match="Fetch the feature-adding commit and its parent"):
        if operation == "show":
            _history_bytes(ROOT, "show", "parent:src/stateframe/engine/missing.py")
        else:
            _coalesce_parent(ROOT)


def _assert_no_pinned_parity_oracles(source):
    pinned_show = re.compile(
        r'''(?:\bgit\s+show\s+|["']git["']\s*,\s*["']show["']\s*,\s*f?["'])'''
        r"([0-9a-fA-F]{7,40}):"
    )
    assert not pinned_show.findall(source), "Use the feature's parent or a checked-in fixture"
    assert not re.search(r"coalesce_off_[0-9a-fA-F]{7,40}\.json", source), "Use a SHA-free golden name"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assert):
            literals = [part.value for part in ast.walk(node.test)
                        if isinstance(part, ast.Constant) and isinstance(part.value, str)]
            assert not any(re.fullmatch(r"[0-9a-fA-F]{7,40}", value) for value in literals), (
                "Assert golden bytes, not a baseline commit SHA"
            )


def test_no_literal_git_show_or_sha_baseline_oracles():
    _assert_no_pinned_parity_oracles(Path(__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["shell_show", "argv_show", "golden_name", "baseline_assert"])
def test_parity_sha_guard_rejects_pinned_oracles(kind):
    sha = "a" * 40
    source = {
        "shell_show": f'run("git show {sha}:src/example.py")',
        "argv_show": f'run(["git", "show", "{sha}:src/example.py"])',
        "golden_name": f'load("coalesce_off_{sha}.json")',
        "baseline_assert": f'assert golden.pop("baseline_commit") == "{sha}"',
    }[kind]
    with pytest.raises(AssertionError):
        _assert_no_pinned_parity_oracles(source)


def test_parity_sha_guard_accepts_runtime_parent_and_golden_bytes():
    _assert_no_pinned_parity_oracles('''
source = run(["git", "show", f"{parent}:src/example.py"])
golden = load("coalesce_off_golden.json")
assert request.encode("utf-8") == golden["request"].encode("utf-8")
''')


class Bolt:
    """Stub of the kernel bolt surface the hook touches: _rec, organ_out, unit_plan, unit_index."""

    def __init__(self, plan, exit_code=0):
        self.unit_plan, self.last_test_exit, self.events, self.organ_lines = plan, exit_code, [], []
        self.unit_index = 0
        self.measurement = SimpleNamespace()

    def _rec(self, *args):
        self.events.append(args)

    def organ_out(self, name, line):
        self.organ_lines.append((name, line))
        return line


def _plant(tmp_path, files):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)


@pytest.fixture
def v2_on(tmp_path, monkeypatch):
    monkeypatch.setenv(chains_recall.MODE_ENV, "on")
    monkeypatch.setenv(v2.MODE_ENV, "on")
    monkeypatch.setattr(v2, "_index_class", lambda: RefIndex)
    _plant(tmp_path, PKG)
    return tmp_path


@pytest.fixture
def coalesce_on(v2_on, monkeypatch):
    monkeypatch.setenv(COALESCE_ENV, "on")
    return v2_on


def _coalesce():
    import unit_coalesce
    return unit_coalesce


def _planted_break(bolt, root, pointer=1):
    """Page passed; the fetch_all unit renames has_more -> more (tests green, seam broken)."""
    if os.environ.get(COALESCE_ENV) == "on" and not hasattr(bolt, "unit_gate_snapshots"):
        bolt.unit_index = pointer
        v2.gate_and_advance(bolt, root, _pytest_observation(bolt.unit_plan))
    bolt.repair_number = getattr(bolt, "repair_number", 0) + 1
    (root / "pkg/models.py").write_text(BROKEN_MODELS + f"\n# repair {bolt.repair_number}\n")
    bolt.last_test_exit, bolt.unit_index = 0, pointer
    return v2.gate_and_advance(bolt, root, _pytest_observation(bolt.unit_plan))


def _run_break_scenario(root):
    plan = v2.derive_with_index(row(), root)
    bolt = Bolt(plan, 0)
    assert v2.gate_and_advance(bolt, root, _pytest_observation(plan)).passed and bolt.unit_index == 1
    _planted_break(bolt, root)
    _planted_break(bolt, root)
    return bolt


def _unit(plan, symbol):
    return next(u for u in plan.units if u.symbol == symbol)


def _ids(plan):
    return [u.unit_id for u in plan.units]


# ------------------------------------------------------------ 1-2 switch
def test_off_byte_parity(v2_on, monkeypatch):
    sys.modules.pop("unit_coalesce", None)
    monkeypatch.delenv(COALESCE_ENV, raising=False)
    unset = _run_break_scenario(v2_on)
    (v2_on / "pkg/models.py").write_text(MODELS)
    monkeypatch.setenv(COALESCE_ENV, "off")
    off = _run_break_scenario(v2_on)
    golden = ["unit_gate_pass", "unit_gate_fail", "unit_gate_fail"]      # A's test_pointer_advances_only_on_passing_gate
    assert [e[1] for e in unset.events] == golden and unset.events == off.events
    assert unset.unit_plan.plan_digest == off.unit_plan.plan_digest
    assert unset.unit_plan.to_dict() == off.unit_plan.to_dict()
    assert unset.unit_index == off.unit_index == 1
    for key in ("coalesced_from", "gate_attempts", "lineage", "coalesce_events"):
        assert key not in unset.unit_plan.to_dict() and all(key not in u for u in unset.unit_plan.to_dict()["units"])
    assert "unit_coalesce" not in sys.modules
    monkeypatch.setenv(COALESCE_ENV, "on")
    monkeypatch.delenv(v2.MODE_ENV, raising=False)                         # v2 off: nothing to merge, never imported
    assert v2.derive_with_index(row(), v2_on) is None
    assert "unit_coalesce" not in sys.modules


def test_switch_junk_raises():
    uc = _coalesce()
    assert uc.mode({}) == "off" and uc.mode({COALESCE_ENV: "on"}) == "on" and uc.MODE_ENV == COALESCE_ENV
    with pytest.raises(ValueError):
        uc.mode({COALESCE_ENV: "maybe"})


# ------------------------------------------------------------ 3-4 static / unchanged
def test_planted_cycle_is_one_unit_and_coalesce_is_identity():
    uc = _coalesce()
    src = "from a import A\nfrom c import use\ndef test_u(): assert use()\n"
    cyc, _ = plan_for(CYC2, test_src=src, msg="Implement class A, class B and def use.")
    assert [u.kind for u in cyc.units].count("joint") == 1 and cyc.units[0].members == ("A", "B")
    plan, _ = plan_for()
    with pytest.raises(ValueError):
        uc.coalesce(plan, None)
    # a hand-built two-unit plan with A<->B edges through the pure merge: one joint unit
    a = v2.PlannedUnitV2("unit:A:g0", "A", "class", ("test_a",), ("unit:B:g0",), "sc", "ho", ("a.py",), ("A",))
    b = v2.PlannedUnitV2("unit:B:g0", "B", "class", ("test_b",), ("unit:A:g0",), "sc", "ho", ("b.py",), ("B",))
    hand = v2.UnitPlanV2("t", "a.py", (a, b), v2.GATED_HANDOFF, "", "gated", (), "rev", ("a.py", "b.py"))
    hand = v2.UnitPlanV2(**{**hand.__dict__, "plan_digest": v2._digest(hand._basis())})
    out = uc.coalesce(hand, uc.CoalesceTrigger("integration", "unit:B:g0", "unit:A:g0", "a.py:A"))
    assert not out.escalate and len(out.plan.units) == 1
    merged = out.plan.units[0]
    assert merged.members == ("A", "B") and merged.kind == "joint" and merged.depends_on == ()
    assert merged.unit_id == "unit:A+B:g1" and out.plan.plan_digest != hand.plan_digest


def test_acyclic_plan_unchanged_without_failure(coalesce_on):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan, 0)
    digest = plan.plan_digest
    for expected in (1, 2, 3):
        assert v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan)).passed and bolt.unit_index == expected
    assert [e[1] for e in bolt.events] == ["unit_gate_pass"] * 3
    assert bolt.unit_plan is plan and plan.plan_digest == digest and bolt.organ_lines == []


# ------------------------------------------------------------ 5-6 integration merge + lineage
def test_planted_integration_break_merges_producer_consumer(coalesce_on):
    plan = v2.derive_with_index(row(), coalesce_on)
    page_id, fetch_all_id = _unit(plan, "Page").unit_id, _unit(plan, "fetch_all").unit_id
    bolt = Bolt(plan, 0)
    assert v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan)).passed and bolt.unit_index == 1
    bolt.unit_index = 2                                           # fetch (unit 1) passed: pointer at fetch_all
    _planted_break(bolt, coalesce_on, pointer=2)
    assert bolt.unit_plan is plan and bolt.unit_index == 2       # first failure: allowance 2, no merge
    _planted_break(bolt, coalesce_on, pointer=2)
    new = bolt.unit_plan
    assert new is not plan and bolt.unit_index == 0
    merged = new.units[0]
    fetch_id = _unit(plan, "fetch").unit_id
    # fetch consumes Page and feeds fetch_all: left outside it would form a cycle with the merged unit
    assert merged.symbol == "Page+fetch+fetch_all" and merged.kind == "joint"
    assert merged.coalesced_from == (page_id, fetch_id, fetch_all_id)
    assert merged.files == ("pkg/client.py", "pkg/models.py")
    assert merged.gate_tests == ("test_page", "test_fetch_all")  # union in plan order
    assert merged.members == ("Page", "fetch", "fetch_all") and merged.unit_id == "unit:Page+fetch+fetch_all:g1"
    assert symbols(new) == ["Page+fetch+fetch_all", "consume"] and merged.depends_on == ()
    assert _unit(new, "consume").depends_on == (merged.unit_id,)
    assert not any(page_id in u.depends_on or fetch_all_id in u.depends_on for u in new.units)
    event = next(e for e in bolt.events if e[1] == "unit_coalesce")
    assert f"{plan.plan_digest}->{new.plan_digest}" in event[2]
    assert "kind=integration" in event[2] and "surface=pkg/models.py:Page.has_more" in event[2]
    assert bolt.organ_lines and "coalesced Page+fetch_all" in bolt.organ_lines[-1][1]   # partner+failing named


def test_merged_digest_differs_and_records_lineage(coalesce_on):
    uc = _coalesce()
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan, 0)
    bolt.unit_index = 2
    _planted_break(bolt, coalesce_on, pointer=2)
    _planted_break(bolt, coalesce_on, pointer=2)
    new = bolt.unit_plan
    assert new.plan_digest != plan.plan_digest and new.lineage == (plan.plan_digest,)
    assert new.coalesce_events == 1 and new.to_dict()["lineage"] == [plan.plan_digest]
    assert new.plan_digest == v2._digest(new._basis())
    # replaying the same trigger on the original plan reproduces the same digest (resume determinism)
    trigger = uc.CoalesceTrigger("integration", _unit(plan, "fetch_all").unit_id, _unit(plan, "Page").unit_id,
                                 "pkg/models.py:Page.has_more")
    assert bolt.unit_index == 0 and new.units[0].symbol == "Page+fetch+fetch_all"
    files = {p: (coalesce_on / p).read_text() for p in plan.index_paths}
    surfaces = uc.reseam(files, _unit(plan, "fetch_all").files, _unit(plan, "Page").files)
    replay = uc.coalesce(plan, trigger, failures={_unit(plan, "fetch_all").unit_id: 2}, surfaces=surfaces)
    assert replay.plan.plan_digest == new.plan_digest and replay.plan.to_dict() == new.to_dict()


# ------------------------------------------------------------ 7-8 partner mapping
def test_shared_test_regression_maps_to_passed_unit(coalesce_on):
    uc = _coalesce()
    shared = ("from pkg.models import Page\nfrom pkg.client import fetch_all\nfrom pkg.consumer import consume\n"
              "def test_both(): assert fetch_all() == [] and Page([], False)\n"
              "def test_only_page(): assert Page([], True)\n"
              "def test_consume(): assert consume() == []\n")
    plan, _ = plan_for(PKG, test_src=shared)
    page, fetch_all = _unit(plan, "Page"), _unit(plan, "fetch_all")
    verdict = v2.GateVerdict(False, failing_tests=("test_both",))
    files = dict(PKG)
    trigger = uc.locate_partner(plan, fetch_all, verdict, chains.seam_manifest(files))
    assert trigger.kind == "integration" and trigger.partner_unit == page.unit_id
    assert trigger.failing_unit == fetch_all.unit_id and trigger.surface == "test:test_both"


def test_unmapped_failure_reseams_then_merges_predecessor(coalesce_on):
    uc = _coalesce()
    plan = v2.derive_with_index(row(), coalesce_on)
    fetch_all = _unit(plan, "fetch_all")
    files = {p: (coalesce_on / p).read_text() for p in plan.index_paths}
    verdict = v2.GateVerdict(False, failing_tests=("test_fetch_all",))     # unique to fetch_all, no seam row
    trigger = uc.locate_partner(plan, fetch_all, verdict, chains.seam_manifest(files))
    assert trigger.kind == "unmapped" and trigger.partner_unit == _unit(plan, "fetch").unit_id
    surfaces = uc.reseam(files, fetch_all.files, _unit(plan, "fetch").files)
    assert surfaces == ()                                                    # same file: nothing cross-file
    surfaces = uc.reseam(files, ("pkg/client.py",), ("pkg/models.py",))
    assert "pkg/models.py:Page" in surfaces and all(s.startswith("pkg/models.py:") for s in surfaces)
    assert surfaces == tuple(sorted(surfaces)) and surfaces == uc.reseam(files, ("pkg/models.py",), ("pkg/client.py",))
    bolt = Bolt(plan, 1)
    bolt.unit_index = 2
    v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan, ("test_fetch_all",)))
    for i in range(2):
        path = coalesce_on / "pkg/client.py"
        path.write_text(path.read_text() + f"\n# repair {i}\n")
        v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan, ("test_fetch_all",)))
    new = bolt.unit_plan
    assert new is plan and bolt.unit_index == 2
    event = next(e for e in bolt.events if e[1] == "unit_coalesce_skip")
    assert "no_reseam_evidence" in event[2]
    # a pair that shares real seams carries them in the subcontract text
    out = uc.coalesce(plan, uc.CoalesceTrigger("unmapped", fetch_all.unit_id, _unit(plan, "Page").unit_id,
                                               "test:test_fetch_all"), surfaces=("pkg/models.py:Page",))
    assert "pkg/models.py:Page" in out.plan.units[0].subcontract


# ------------------------------------------------------------ 9-11 escalate, budget, caps
def test_second_failure_escalates_not_retries(coalesce_on):
    bolt = _run_break_scenario(coalesce_on)
    merged_plan, merged_id = bolt.unit_plan, bolt.unit_plan.units[0].unit_id
    assert bolt.unit_index == 0 and merged_plan.coalesce_events == 1
    _planted_break(bolt, coalesce_on, pointer=0)                   # merged unit fails once: plain retry
    assert bolt.unit_plan is merged_plan and not hasattr(bolt, "unit_escalate")
    _planted_break(bolt, coalesce_on, pointer=0)                   # second failure after merge: escalate
    assert bolt.unit_plan is merged_plan and bolt.unit_index == 0
    assert bolt.unit_escalate == merged_id and bolt.measurement._unit_coalesce_escalated == merged_id
    assert [e[1] for e in bolt.events].count("unit_coalesce_escalate") == 1
    assert "stop retrying" in bolt.organ_lines[-1][1]
    for _ in range(3):
        _planted_break(bolt, coalesce_on, pointer=0)
    assert bolt.unit_plan is merged_plan and [e[1] for e in bolt.events].count("unit_coalesce") == 1


def test_budget_is_sum_of_leftovers(coalesce_on):
    uc = _coalesce()
    plan = v2.derive_with_index(row(), coalesce_on)
    page_id, fetch_id = _unit(plan, "Page").unit_id, _unit(plan, "fetch_all").unit_id
    trigger = uc.CoalesceTrigger("integration", fetch_id, page_id, "pkg/models.py:Page.has_more")
    fetch = _unit(plan, "fetch").unit_id                     # absorbed by the closure; its leftover pools too
    merged = lambda failures: uc.coalesce(plan, trigger, failures=failures).plan.units[0]
    assert merged({fetch_id: 2, fetch: 2}).gate_attempts == 2                                 # 2+0+0, never 6
    assert merged({fetch_id: 1, page_id: 1, fetch: 2}).gate_attempts == 2
    assert merged({fetch_id: 2, page_id: 2, fetch: 2}).gate_attempts == 2
    # a direct neighbour pair pools exactly its two leftovers
    pair = uc.CoalesceTrigger("integration", fetch, page_id, "pkg/models.py:Page")
    assert uc.coalesce(plan, pair, failures={fetch: 2}).plan.units[0].gate_attempts == 2
    assert uc.coalesce(plan, pair, failures={fetch: 1, page_id: 1}).plan.units[0].gate_attempts == 2
    assert uc.DEFAULT_GATE_ATTEMPTS == 2


def test_merge_size_cap_and_event_cap_escalate(coalesce_on):
    uc = _coalesce()
    files = {**CYC3, "d.py": "from a import A\n\ndef use():\n    return A()\n",
             "e.py": "from d import use\n\ndef more():\n    return use()\n"}
    src = "from a import A\nfrom d import use\nfrom e import more\ndef test_u(): assert use()\ndef test_m(): assert more()\n"
    plan, _ = plan_for(files, test_src=src, msg="Implement class A, class B, def use and def more.")
    joint, use = plan.units[0], _unit(plan, "use")
    assert joint.symbol == "A+B+C"
    one = uc.coalesce(plan, uc.CoalesceTrigger("integration", use.unit_id, joint.unit_id, "a.py:A"), failures={use.unit_id: 2})
    assert not one.escalate and len(one.plan.units[0].members) == 4 and one.plan.coalesce_events == 1
    big = one.plan.units[0]
    two = uc.coalesce(one.plan, uc.CoalesceTrigger("integration", _unit(one.plan, "more").unit_id, big.unit_id, "d.py:use"),
                      failures={_unit(one.plan, "more").unit_id: 2})
    assert two.escalate and two.plan is one.plan and "members" in two.reason                       # 5 > MAX_MERGE_MEMBERS
    assert uc.MAX_MERGE_MEMBERS == 4 and uc.MAX_COALESCE_EVENTS == 2
    # third coalesce in one plan escalates on the event cap
    capped = v2.UnitPlanV2(**{**plan.__dict__, "coalesce_events": 2})
    three = uc.coalesce(capped, uc.CoalesceTrigger("integration", use.unit_id, joint.unit_id, "a.py:A"))
    assert three.escalate and three.plan is capped and "events" in three.reason


# ------------------------------------------------------------ 12-13 order invariant
def _providers_first(plan):
    seen = set()
    for unit in plan.units:
        assert all(dep in seen for dep in unit.depends_on), (unit.unit_id, unit.depends_on)
        seen.add(unit.unit_id)


def test_merged_unit_never_precedes_its_providers_pkg(coalesce_on):
    uc = _coalesce()
    plan = v2.derive_with_index(row(), coalesce_on)
    assert symbols(plan) == ["Page", "fetch", "fetch_all", "consume"]
    page, fetch, fetch_all = (_unit(plan, s).unit_id for s in ("Page", "fetch", "fetch_all"))
    assert uc.merge_closure(plan, {page, fetch_all}) == {page, fetch, fetch_all}   # fetch sits on the Page->fetch_all path
    assert uc.merge_closure(plan, {page, fetch}) == {page, fetch}
    bolt = Bolt(plan, 0)
    assert v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan)).passed
    assert v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan)).passed
    assert bolt.unit_index == 2                                                   # Page, fetch passed
    _planted_break(bolt, coalesce_on, pointer=2)
    _planted_break(bolt, coalesce_on, pointer=2)
    new = bolt.unit_plan
    assert symbols(new) == ["Page+fetch+fetch_all", "consume"] and bolt.unit_index == 0
    _providers_first(new)
    assert new.units[bolt.unit_index].unit_id == new.units[0].unit_id            # pointer is the merged unit
    # a merge whose partner is the direct provider leaves passed units before the pointer untouched
    out = uc.coalesce(plan, uc.CoalesceTrigger("integration", _unit(plan, "consume").unit_id, fetch_all, "x"))
    assert symbols(out.plan) == ["Page", "fetch", "fetch_all+consume"] and out.unit_index == 2
    assert out.plan.units[:2] == plan.units[:2] or [u.unit_id for u in out.plan.units[:2]] == [page, fetch]
    _providers_first(out.plan)
    with pytest.raises(ValueError):
        uc.assert_providers_first((new.units[1], new.units[0]))


def test_every_merge_over_fixtures_keeps_providers_first(coalesce_on):
    uc = _coalesce()
    cases = [plan_for(PKG)[0]]
    for files, src, msg in [
        (CYC2, "from a import A\nfrom c import use\ndef test_u(): assert use()\n", "Implement class A, class B and def use."),
        (CYC3, "from a import A\nfrom d import use\ndef test_u(): assert use()\n", "Implement class A, class B and def use."),
    ]:
        cases.append(plan_for(files, test_src=src, msg=msg)[0])
    checked = 0
    for plan in cases:
        _providers_first(plan)
        for f, failing in enumerate(plan.units):
            for partner in plan.units[:f]:                    # every (passed partner, failing) pair
                out = uc.coalesce(plan, uc.CoalesceTrigger("integration", failing.unit_id, partner.unit_id, "x"))
                if out.escalate:
                    continue
                _providers_first(out.plan)
                assert out.plan.units[out.unit_index].unit_id == out.merged_unit_id
                assert len(out.plan.units) == len(plan.units) - len(out.plan.units[out.unit_index].coalesced_from) + 1
                checked += 1
    assert checked >= 6


# Review regressions: small synthetic plans isolate each contract from index heuristics.
def _small_plan(names, deps=None, paths=None):
    deps, paths = deps or {}, paths or {}
    units = tuple(v2.PlannedUnitV2(
        f"unit:{n}:g0", n, "function", (f"test_{n}",),
        tuple(f"unit:{d}:g0" for d in deps.get(n, ())), "sc", "ho",
        (paths.get(n, f"{n}.py"),), (n,)) for n in names)
    plan = v2.UnitPlanV2("review", "", units, v2.GATED_HANDOFF, "", "gated", (),
                         "revision", tuple(sorted({p for u in units for p in u.files})))
    return replace(plan, plan_digest=v2._digest(plan._basis()))


def _pytest_observation(plan, failed=()):
    # Real pytest stdout -> real _run_tests_impl -> real compact_observation.
    # Cache only immutable fixture observations, never build verbose status strings.
    return _real_pytest_observation(tuple(sorted({t for u in plan.units for t in u.gate_tests})), tuple(failed))


def _seed_pass_sources(bolt, root, observation):
    """Give declared prior passes the source/inventory binding from real pytest."""
    from unit_gate_evidence import pass_snapshot, shared_source_identity
    plan = bolt.unit_plan
    tests = shared_source_identity(plan, root)
    bolt.unit_gate_pass_sources = {unit.unit_id: pass_snapshot(unit, root, observation, tests)
                                  for unit in plan.units if unit.unit_id in bolt.unit_gate_passed}


@lru_cache(maxsize=32)
def _real_pytest_observation(names, failed, *, runtime_present=True, truncated=False):
    import kernel_check_receipts
    import kernel_execution_policy
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair

    with TemporaryDirectory(prefix="sf-gate-") as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        (root / "pytest.ini").write_text("[pytest]\n")
        (root / "test_gate.py").write_text("\n".join(
            f"def {name}(): assert {name not in failed!r}" for name in names))
        runtime = SimpleNamespace(binding=dict, test_observation=lambda *args, **kw: None)
        patch.setenv(v2.MODE_ENV, "on")
        patch.setenv(COALESCE_ENV, "on")
        patch.setenv("STATEFRAME_TEST_RUN_SPANS", "off")
        patch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
        def run(command, cwd, **kwargs):
            completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                       timeout=30, check=False,
                                       env={**kwargs["env"], "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
            return SimpleNamespace(completed=completed, timed_out=False, output_truncated=truncated)
        patch.setattr(kernel_execution_policy, "run_sandboxed", run)
        patch.setattr(kernel_execution_policy, "policy_read_grants", lambda result: ())
        patch.setattr(kernel_check_receipts, "sandbox_open_denial", lambda *args: None)
        token = ACTIVE_RUNTIME.set(runtime if runtime_present else None)
        try:
            observation = react_repair._run_tests_impl(root, "gate", "test_gate.py", row={"_skip_db": True})
        finally:
            ACTIVE_RUNTIME.reset(token)
        assert ("CHECK_FAILURE_SUMMARY" in observation) == runtime_present
        if runtime_present:
            assert not re.search(r"^\S+::test_\w+\s+PASSED", observation, re.MULTILINE)
        elif not failed:
            assert all(f"::{name} PASSED" in observation for name in names)
        return observation


@pytest.mark.parametrize("members", [("helper",), ("C.helper",)])
def test_review_01_partner_matches_member_before_file(members):
    plan = _small_plan(("other", "helper", "use"), paths={"other": "m.py", "helper": "m.py"})
    helper = replace(plan.units[1], members=members)
    plan = replace(plan, units=(plan.units[0], helper, plan.units[2]))
    qualified = "." in members[0]
    verdict = v2.GateVerdict(False, seam_violations=({"name": "helper", "file": "use.py",
        "detail": "class C exposes helper" if qualified else ""},))
    trigger = _coalesce().locate_partner(plan, plan.units[2], verdict, {"definitions": [
        {"name": "C" if qualified else "helper", "kind": "class" if qualified else "function", "file": "m.py"}]})
    assert trigger.partner_unit == helper.unit_id and trigger.kind == "integration"


def test_review_02_unmapped_requires_reseam_evidence(coalesce_on):
    plan = _small_plan(("helper", "use"))
    bolt = Bolt(plan, 1)
    bolt.unit_index = 1
    verdict = v2.GateVerdict(False, ("test_use",))
    for i in range(4):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], verdict,
                                    {"helper.py": "def helper(): return 1\n", "use.py": f"x = {i}\n"}, coalesce_on)
    assert bolt.unit_plan is plan and bolt.unit_index == 1
    assert any(e[1] == "unit_coalesce_skip" and "no_reseam_evidence" in e[2] for e in bolt.events)


def test_review_02_pure_merge_refuses_unsupported_unmapped_pair():
    plan = _small_plan(("a", "b"))
    uc = _coalesce()
    out = uc.coalesce(plan, uc.CoalesceTrigger("unmapped", plan.units[1].unit_id, plan.units[0].unit_id, "test:b"))
    assert out.plan is plan and out.merged_unit_id is None and "no_reseam_evidence" in out.reason


def test_review_02_real_reseam_allows_unmapped_merge_and_seeds_budget(coalesce_on):
    plan = _small_plan(("helper", "use"))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    uc = _coalesce()
    verdict = v2.GateVerdict(False, ("test_use",))
    for i in range(3):
        files = {"helper.py": "def helper(): return 1\n",
                 "use.py": f"from helper import helper\ndef use(): return helper() + {i}\n"}
        uc.on_gate_failure(bolt, plan, plan.units[1], verdict, files, coalesce_on)
    new = bolt.unit_plan
    assert new is not plan and new.units[0].members == ("helper", "use")
    assert "helper.py:helper" in new.units[0].subcontract
    assert any(e[1] == "unit_coalesce" and "kind=unmapped" in e[2] and "fallback=predecessor" in e[2]
               for e in bolt.events)
    for _ in range(3):
        uc.on_gate_failure(bolt, new, new.units[0], verdict, files, coalesce_on)
    assert new.units[0].unit_id not in bolt.unit_gate_failures
    assert not hasattr(bolt, "unit_escalate")


@pytest.mark.parametrize("error", [ValueError("cycle"), KeyError("missing")])
def test_review_03_hook_errors_hold_pointer_and_record(coalesce_on, monkeypatch, error):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan)
    bolt.unit_index = 2
    (coalesce_on / "pkg/models.py").write_text(BROKEN_MODELS)
    def broken(*args):
        raise error
    monkeypatch.setattr(_coalesce(), "on_gate_failure", broken)
    verdict = v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan))
    assert verdict is not None and not verdict.passed and bolt.unit_index == 2
    assert any(e[1] == "unit_coalesce_error" and type(error).__name__ in e[2] for e in bolt.events)


def test_review_03_invalid_switch_fails_open(coalesce_on, monkeypatch):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan)
    bolt.unit_index = 2
    monkeypatch.setenv(COALESCE_ENV, "junk")
    (coalesce_on / "pkg/models.py").write_text(BROKEN_MODELS)
    v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan))
    assert bolt.unit_index == 2
    assert any(e[1] == "unit_coalesce_error" and COALESCE_ENV in e[2] for e in bolt.events)


def test_review_04_counts_changed_complete_repair_attempts(coalesce_on):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan, 1)
    bolt.unit_index = 2
    complete = _pytest_observation(plan, ("test_fetch_all",))
    v2.gate_and_advance(bolt, coalesce_on, complete)  # pre-edit baseline
    assert getattr(bolt, "unit_gate_failures", {}) == {}
    p = coalesce_on / "pkg/client.py"
    p.write_text(p.read_text() + "\n# first repair\n")
    v2.gate_and_advance(bolt, coalesce_on, "FAILED t.py::test_page")  # current gate un-run
    assert getattr(bolt, "unit_gate_failures", {}) == {}
    v2.gate_and_advance(bolt, coalesce_on, complete)
    v2.gate_and_advance(bolt, coalesce_on, complete)  # repeat same tree
    assert bolt.unit_gate_failures == {plan.units[2].unit_id: 1}
    assert bolt.unit_plan is plan


def test_review_04_new_unit_baseline_excludes_previous_unit_edits(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b"))
    _plant(coalesce_on, {"a.py": "x=0\n", "b.py": "y=0\n"})
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    bolt = Bolt(plan)
    v2.seed_coalesce(bolt, coalesce_on)
    (coalesce_on / "a.py").write_text("x=1\n")
    assert v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan)).passed
    assert bolt.unit_index == 1
    bolt.last_test_exit = 1
    assert not v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan, ("test_b",))).passed
    assert getattr(bolt, "unit_gate_failures", {}) == {}


def test_review_04_install_baseline_counts_first_actual_edit(coalesce_on):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan)
    bolt.unit_index = 2
    v2.seed_coalesce(bolt, coalesce_on)
    (coalesce_on / "pkg/models.py").write_text(BROKEN_MODELS)
    v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(plan))
    assert bolt.unit_gate_failures == {plan.units[2].unit_id: 1}


def test_review_13_switch_is_sampled_once_per_gate(coalesce_on, monkeypatch):
    calls = []
    original = v2._coalesce_enabled
    def sampled():
        calls.append(1)
        return original()
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan)
    observation = _pytest_observation(plan)
    monkeypatch.setattr(v2, "_coalesce_enabled", sampled)
    v2.gate_and_advance(bolt, coalesce_on, observation)
    assert calls == [1]


@pytest.mark.parametrize("setting", [None, "off", "on"])
def test_review_04_host_test_command_proves_complete_gate(coalesce_on, monkeypatch, setting):
    """ON regression; OFF/unset cases are parity pins, expected to pass on the base."""
    if setting is None:
        monkeypatch.delenv(COALESCE_ENV, raising=False)
    else:
        monkeypatch.setenv(COALESCE_ENV, setting)
    source = Path(__file__).resolve().parents[1] / "src/stateframe/engine/probes/react_repair.py"
    tree = ast.parse(source.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_tests_impl")
    block = next(n for n in ast.walk(runner) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_cmd" for t in n.targets)
                 and '"--no-header"' in ast.unparse(n).replace("'", '"'))
    parent = next(n for n in ast.walk(runner) if hasattr(n, "body") and block in n.body)
    start = parent.body.index(block)
    nodes = []
    for n in parent.body[start:]:
        if isinstance(n, ast.ImportFrom) and n.module == "kernel_step_contract":
            break
        nodes.append(n)
    code = compile(ast.Module(body=nodes, type_ignores=[]), "host-pytest-command", "exec")
    scope = {"executable": sys.executable, "root": coalesce_on, "targets": ["t.py"],
             "kexpr": None, "_nearest_pytest_config_args": lambda *args: []}
    exec(code, scope)  # noqa: S102 -- execute only the trusted command construction, no subprocess
    cmd = scope["_cmd"]
    if setting == "on":
        assert "-x" in cmd and "-q" not in cmd
        assert "--maxfail=0" not in cmd and "--verbosity=1" in cmd
    else:
        assert cmd == [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"]
    monkeypatch.delenv(v2.MODE_ENV, raising=False)
    exec(code, scope)  # noqa: S102 -- same trusted construction with the v2 switch off
    assert scope["_cmd"] == [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"]


@pytest.mark.parametrize("selector", ["-x", "-q", "test_a or test_b"])
def test_review_04_command_keeps_selector_values(coalesce_on, selector):
    command = [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-k", selector]
    result = v2.coalesce_pytest_command(command)
    assert result == command[:4] + ["--verbosity=1"] + command[5:]
    assert command[3:5] == ["-x", "-q"]


def test_review_03_bad_command_switch_logs_and_fails_open(coalesce_on, monkeypatch, caplog):
    command = [sys.executable, "-m", "pytest", "-x", "-q"]
    monkeypatch.setenv(COALESCE_ENV, "junk")
    assert v2.coalesce_pytest_command(command) == command
    assert "unit_coalesce_error" in caplog.text and COALESCE_ENV in caplog.text


@pytest.mark.parametrize("exit_code,output", [(0, ""), (0, "t.py::test_page PASSED"),
                                             (1, "FAILED t.py::test_page")])
def test_review_04_partial_run_never_passes_unrun_gate(coalesce_on, exit_code, output):
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan, exit_code)
    bolt.unit_index = 2
    assert v2.gate_and_advance(bolt, coalesce_on, output) is None
    assert bolt.unit_index == 2
    assert [e[1] for e in bolt.events] == ["unit_gate_unobserved"]
    assert "test_fetch_all" in bolt.events[0][2]


def test_review_05_merged_allowance_has_two_edited_attempts():
    plan = _small_plan(("helper", "use"))
    a, b = plan.units
    out = _coalesce().coalesce(plan, _coalesce().CoalesceTrigger("integration", b.unit_id, a.unit_id, "m:helper"),
                               failures={a.unit_id: 1, b.unit_id: 2})
    assert out.plan.units[out.unit_index].gate_attempts >= 2


def test_review_06_advisory_escalation_is_documented():
    import kernel_v46_mechanism_wiring as wiring
    text = Path("KNOWN_LIMITATIONS.md").read_text()
    assert "escalated but kept retrying" in text
    assert 'kernel_v46_mechanism_wiring.enabled("escalation")' in text
    assert wiring.FLAGS["escalation"] == "STATEFRAME_KERNEL_ESCALATION"


@pytest.mark.parametrize("owner", ["helper", "use", "other"])
def test_review_07_dangling_dependencies_are_not_laundered(owner):
    plan = _small_plan(("helper", "other", "use"), {owner: ("GHOST",)})
    with pytest.raises(ValueError, match="GHOST"):
        _coalesce().coalesce(plan, _coalesce().CoalesceTrigger("integration", plan.units[2].unit_id,
                                                             plan.units[0].unit_id, "m:helper"))


def test_review_08_passed_in_between_unit_is_not_regated(coalesce_on, monkeypatch):
    plan = _small_plan(("helper", "other", "use"))
    bolt = Bolt(plan, 1)
    bolt.unit_index = 2  # helper and other have passed
    bolt.unit_gate_passed = {unit.unit_id for unit in plan.units[:2]}
    _seed_pass_sources(bolt, coalesce_on, _pytest_observation(plan))
    uc = _coalesce()
    monkeypatch.setattr(uc, "locate_partner", lambda *a, **kw: uc.CoalesceTrigger(
        "integration", plan.units[2].unit_id, plan.units[0].unit_id, "helper.py:helper"))
    verdict = v2.GateVerdict(False, ("test_use",))
    for i in range(3):
        uc.on_gate_failure(bolt, plan, plan.units[2], verdict, {"use.py": f"x={i}\n"}, coalesce_on)
    merged = bolt.unit_plan
    assert merged is not plan and bolt.unit_index == 0 and merged.units[1].symbol == "other"
    monkeypatch.setattr(v2, "build_index", lambda *a: None)
    bolt.last_test_exit = 0
    v2.gate_and_advance(bolt, coalesce_on, _pytest_observation(merged))
    assert bolt.unit_index == len(merged.units)
    assert not any(e[1] == "unit_gate_pass" and plan.units[1].unit_id in e[2] for e in bolt.events)
    assert "all unit gates passed" in v2.render_unit_progress(bolt)


def test_review_08_position_contract_is_honest():
    assert "earliest" in _coalesce().coalesce.__doc__ and "providers first" in _coalesce().coalesce.__doc__


def test_review_09_merge_notice_supersedes_old_plan(coalesce_on):
    plan = _small_plan(("helper", "use"), paths={"helper": "a.py", "use": "b.py"})
    bolt = Bolt(plan)
    bolt.unit_index = 1
    verdict = v2.GateVerdict(False, ("test_helper",))
    for i in range(3):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], verdict,
                                    {"b.py": f"x={i}\n"}, coalesce_on)
    notice = "\n".join(s for _, s in bolt.organ_lines)
    assert f"supersedes digest {plan.plan_digest}" in notice
    assert bolt.unit_plan.plan_digest in notice and "unit:helper+use:g1" in notice


def test_review_09_refreshes_welded_plan_after_merge(coalesce_on):
    from unit_planner import render_plan_block
    plan = _small_plan(("helper", "use", "other"))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task\n\n" + render_plan_block(plan) + "\n\nkeep suffix"}]
    verdict = v2.GateVerdict(False, ("test_helper",))
    for i in range(3):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], verdict, {"use.py": f"x={i}"}, coalesce_on)
    prefix = json.dumps(messages)
    v2.refresh_coalesced_plan(bolt, messages)
    assert json.dumps(messages[:2]) == prefix
    text = messages[-1]["content"]
    assert f"supersedes digest {plan.plan_digest}" in text and f"digest={bolt.unit_plan.plan_digest}" in text
    assert text.count("[PLAN ") == 1 and "unit:helper+use:g1" in text
    before = json.dumps(messages)
    v2.refresh_coalesced_plan(bolt, messages)
    assert json.dumps(messages) == before


def test_review_09_refresh_preserves_task_quoted_plan_blocks(coalesce_on):
    from unit_planner import render_plan_block
    plan = _small_plan(("helper", "use", "other"))
    quoted = "[PLAN unit-plan/1]\ntask=example digest=example\nuser-owned example\n[/PLAN]"
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": quoted + "\n\n" + render_plan_block(plan) + "\nkeep suffix"}]
    bolt = Bolt(plan)
    bolt.unit_index = 1
    verdict = v2.GateVerdict(False, ("test_helper",))
    for i in range(3):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], verdict, {"use.py": f"x={i}"}, coalesce_on)
    prefix = json.dumps(messages)
    v2.refresh_coalesced_plan(bolt, messages)
    assert json.dumps(messages[:2]) == prefix
    assert messages[1]["content"].startswith(quoted + "\n\n")
    assert messages[1]["content"].endswith("keep suffix")
    text = messages[-1]["content"]
    assert f"supersedes digest {plan.plan_digest}" in text
    assert f"digest={bolt.unit_plan.plan_digest}" in text


def test_review_09_main_loop_refreshes_real_gate_messages(coalesce_on):
    from unit_planner import render_plan_block
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/stateframe/engine/probes/react_repair.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.BoolOp)
                 and "unit_plan_v2_on" in ast.unparse(n.test))
    code = compile(ast.Module(body=block.body, type_ignores=[]), "main-loop-unit-gate", "exec")
    plan = v2.derive_with_index(row(), coalesce_on)
    bolt = Bolt(plan)
    bolt.unit_index = 2
    v2.seed_coalesce(bolt, coalesce_on)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task\n" + render_plan_block(plan)}]
    prefix = json.dumps(messages)
    for i in range(2):
        (coalesce_on / "pkg/models.py").write_text(BROKEN_MODELS + f"\n# repair {i}\n")
        # Execute the trusted production gate block, without starting the engine.
        observation = _pytest_observation(plan)
        exec(code, {"bolt": bolt, "root": coalesce_on, "plain_observation": observation,  # noqa: S102
                    "_unit_gate_observation": observation, "msgs": messages})
    assert bolt.unit_plan is not plan
    assert json.dumps(messages[:2]) == prefix
    assert f"supersedes digest {plan.plan_digest}" in messages[-1]["content"]
    assert f"digest={bolt.unit_plan.plan_digest}" in messages[-1]["content"]
    assert messages[-1]["content"].count("[PLAN ") == 1


def test_review_10_measurement_only_bolt_matches_kernel():
    """Kernel measurement contract pin; also passes on the feature parent, not a fix discriminator."""
    from kernel_bolt import KernelBolt
    real = KernelBolt.__new__(KernelBolt)
    real.measurement = SimpleNamespace()
    assert _coalesce()._runtime_of(real) is real.measurement
    bolt = Bolt(_small_plan(("a",)))
    assert not hasattr(bolt, "runtime") and not hasattr(bolt, "owner")
    _coalesce()._escalate(bolt, "unit:a:g0", "review")
    assert bolt.measurement._unit_coalesce_escalated == "unit:a:g0"


def test_review_11_generated_dags_all_pairs_preserve_contracts():
    uc = _coalesce()
    rng = random.Random(68491)
    checked = reversed_pairs = closures = 0
    for _ in range(35):
        names = tuple(f"n{i}" for i in range(rng.randint(3, 7)))
        deps = {n: tuple(p for p in names[:i] if rng.random() < .4) for i, n in enumerate(names)}
        plan = _small_plan(names, deps)
        for failing in plan.units:
            for partner in plan.units:
                if partner == failing:
                    continue
                trigger = uc.CoalesceTrigger("integration", failing.unit_id, partner.unit_id, "generated")
                out = uc.coalesce(plan, trigger)
                if out.escalate:
                    assert out.plan is plan
                    continue
                _providers_first(out.plan)
                merged = out.plan.units[out.unit_index]
                group = [u for u in plan.units if u.unit_id in merged.coalesced_from]
                assert set(merged.members) == {m for u in group for m in u.members}
                assert set(merged.gate_tests) == {t for u in group for t in u.gate_tests}
                assert set(merged.files) == {p for u in group for p in u.files}
                assert out.plan.lineage == (plan.plan_digest,)
                assert out.plan.plan_digest == v2._digest(out.plan._basis())
                assert out == uc.coalesce(plan, trigger)
                checked += 1
                reversed_pairs += plan.units.index(partner) > plan.units.index(failing)
                closures += len(group) > 2
    assert checked > 300 and reversed_pairs > 100 and closures > 20


def test_review_11_hash_seed_merge_bytes():
    source = '''
from test_unit_coalesce import _small_plan
from unit_coalesce import coalesce, CoalesceTrigger
import json
p = _small_plan(("a", "b", "c", "d"), {"b": ("a",), "c": ("b",), "d": ("c",)})
o = coalesce(p, CoalesceTrigger("integration", p.units[0].unit_id, p.units[2].unit_id, "seed"))
print(json.dumps({"plan": o.plan.to_dict(), "index": o.unit_index}, sort_keys=True))
'''
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src/stateframe/engine")))}
    outputs = [subprocess.run([sys.executable, "-c", source], env={**env, "PYTHONHASHSEED": seed},
                              capture_output=True, text=True, check=True, timeout=30).stdout
               for seed in ("0", "1", "42", "12345")]
    assert outputs[0] and len(set(outputs)) == 1


def test_review_12_ambiguous_definition_has_record(coalesce_on):
    plan = _small_plan(("helper", "use"))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    files = {"helper.py": "def helper(): return 1\n", "other.py": "helper = 2\n", "use.py": "x=0\n"}
    verdict = v2.GateVerdict(False, seam_violations=({"name": "helper", "file": "use.py"},))
    for i in range(3):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], verdict, {**files, "use.py": f"x={i}\n"}, coalesce_on)
    assert any(e[1] == "unit_coalesce_skip" and "ambiguous_definition" in e[2] for e in bolt.events)


def test_review_13_trigger_kinds_describe_actual_producers():
    assert get_args(get_type_hints(_coalesce().CoalesceTrigger)["kind"]) == ("integration", "unmapped")


def _record_planner_request(bolt, monkeypatch):
    """Capture real ask_tools transport bytes without a server or a StateFrame run."""
    import unit_planner as v1
    from probes import react_repair as react
    captured = []
    class Reply:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
    def send(req, **kwargs):
        captured.append(req.data.decode())
        return Reply()
    monkeypatch.setattr(react, "MODEL", "coalesce-parity")
    monkeypatch.setattr(react, "API", "http://127.0.0.1:1/v1/chat/completions")
    monkeypatch.setattr(react.urllib.request, "urlopen", send)
    messages = [{"role": "system", "content": "Repair the current unit."},
                {"role": "user", "content": v1.render_plan_block(bolt.unit_plan)},
                {"role": "user", "content": v1.render_current_unit_line(bolt.unit_plan, bolt.unit_index)}]
    assert react.ask_tools(messages, mt=128, temp=0, tools=[]) == ("ok", [])
    assert len(captured) == 1
    return {"request": captured[0], "plan": bolt.unit_plan.to_dict(),
            "events": [list(e) for e in bolt.events], "pointer": bolt.unit_index}


@pytest.mark.parametrize("setting", [None, "off"])
def test_review_11_off_matches_baseline_recorded_request(v2_on, monkeypatch, setting):
    """Recorded request parity pin, captured with coalescing OFF on the feature parent."""
    sys.modules.pop("unit_coalesce", None)
    if setting is None:
        monkeypatch.delenv(COALESCE_ENV, raising=False)
    else:
        monkeypatch.setenv(COALESCE_ENV, setting)
    bolt = _run_break_scenario(v2_on)
    golden = json.loads(Path(__file__).with_name("coalesce_off_golden.json").read_text(encoding="utf-8"))
    recorded = _record_planner_request(bolt, monkeypatch)
    assert recorded.pop("request").encode("utf-8") == golden.pop("request").encode("utf-8")
    assert recorded == golden
    assert "unit_coalesce" not in sys.modules


def test_coalescing_is_documented_as_experimental_and_default_off():
    from kernel_protocol_spec import _SWITCH_REFS
    from stateframe import runner

    switch = next(ref for ref in _SWITCH_REFS if ref.name == COALESCE_ENV)
    assert switch.default == "off"
    assert "experimental" in switch.purpose.lower()
    assert "default off" in switch.purpose.lower()
    assert runner.OPT_IN_SWITCHES[COALESCE_ENV] == ("off", "on")
    root = Path(__file__).resolve().parents[1]
    for path in (root / "docs/guide/running.md", root / "KNOWN_LIMITATIONS.md"):
        section = next(section for section in path.read_text().split("\n## ")
                       if section.startswith(("Experimental unit coalescing", "Opt-in unit coalescing:")))
        assert COALESCE_ENV in section
        assert "**EXPERIMENTAL**" in section and "**off by default**" in section
        assert "live A/B" in section and "by default until a live A/B" in section


@pytest.mark.parametrize("failed", [(), ("test_a",)])
def test_r2_n1_real_runner_compaction_reaches_gate(coalesce_on, monkeypatch, failed):
    plan = _small_plan(("a",))
    bolt = Bolt(plan, int(bool(failed)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    observation = _pytest_observation(plan, failed)
    assert "CHECK_FAILURE_SUMMARY" in observation
    restored = pickle.loads(pickle.dumps(observation))
    assert str(restored) == str(observation)
    assert restored.unit_gate_runs == observation.unit_gate_runs
    verdict = v2.gate_and_advance(bolt, coalesce_on, restored)
    assert verdict is not None and verdict.passed == (not failed)
    assert any(e[1] == ("unit_gate_fail" if failed else "unit_gate_pass") for e in bolt.events)


@pytest.mark.parametrize("status", ["SKIPPED", "XFAIL", "XPASS", "DESELECTED"])
def test_r2_n2_nonpasses_are_observed_failures(coalesce_on, monkeypatch, status):
    plan = _small_plan(("a",))
    bolt = Bolt(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    verdict = v2.gate_and_advance(bolt, coalesce_on, f"test_a.py::TestA::test_a {status} [100%]")
    assert verdict is not None and not verdict.passed and verdict.failing_tests == ("test_a",)
    assert [e[1] for e in bolt.events] == ["unit_gate_fail"]


def test_r2_n2_unnamed_deselection_has_reason_and_no_attempt(coalesce_on):
    bolt = Bolt(_small_plan(("a",)), 5)
    assert v2.gate_and_advance(bolt, coalesce_on, "1 deselected in 0.01s") is None
    assert bolt.unit_index == 0 and not hasattr(bolt, "unit_gate_failures")
    assert [e[1] for e in bolt.events] == ["unit_gate_unobserved"]
    assert "test_a" in bolt.events[0][2] and "deselected" in bolt.events[0][2]


@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError, AttributeError, RecursionError, RuntimeError])
@pytest.mark.parametrize("passed", [False, True])
def test_r2_n3_n6_hook_fail_open_keeps_legitimate_pointer(coalesce_on, monkeypatch, error_type, passed):
    plan = _small_plan(("a", "b"))
    bolt = Bolt(plan)
    bolt.unit_index = 1 if passed else 0  # gate already advanced before the hook
    def broken(*args, **kwargs):
        raise error_type("advisory failure")
    uc = _coalesce()
    monkeypatch.setattr(uc, "observe_tree" if passed else "on_gate_failure", broken)
    v2._coalesce_gate_hook(bolt, plan, plan.units[0], v2.GateVerdict(passed), {}, coalesce_on, 0, True)
    assert bolt.unit_index == (1 if passed else 0)
    assert [e[1] for e in bolt.events] == ["unit_coalesce_error"]
    assert error_type.__name__ in bolt.events[0][2]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_r2_n6_process_control_is_not_swallowed(coalesce_on, monkeypatch, error_type):
    plan = _small_plan(("a",))
    bolt = Bolt(plan)
    def interrupted(*args):
        raise error_type()
    monkeypatch.setattr(_coalesce(), "observe_tree", interrupted)
    with pytest.raises(error_type):
        v2._coalesce_gate_hook(bolt, plan, plan.units[0], v2.GateVerdict(True), {}, coalesce_on, 0, True)
    assert not bolt.events


@pytest.mark.parametrize("operation", ["seed", "command", "results", "refresh", "weld", "capture", "combine"])
def test_r2_n6_other_advisory_helpers_fail_open(coalesce_on, monkeypatch, caplog, operation):
    plan = _small_plan(("a", "b"))
    bolt = Bolt(plan)
    def broken(*args, **kwargs):
        raise TypeError("advisory failure")
    if operation == "refresh":
        bolt.unit_plan_refresh = "notice"
        class BrokenMessages(list):
            append = broken
        v2.refresh_coalesced_plan(bolt, BrokenMessages())
    elif operation == "weld":
        monkeypatch.setattr(bolt, "organ_out", broken)
        v2.weld_initial_plan(bolt, [])
    else:
        monkeypatch.setattr(v2, "_coalesce_enabled", broken)
        if operation == "seed":
            v2.seed_coalesce(bolt, coalesce_on)
        elif operation in {"capture", "combine"}:
            text = "compact receipt"
            result = (v2.coalesce_observation(text, "raw output", 0) if operation == "capture"
                      else v2.coalesce_combined_observation(text, (text,)))
            assert result is text
            assert "unit_coalesce_error" in caplog.text and "TypeError" in caplog.text
            return
        elif operation == "command":
            command = ["python", "-m", "pytest", "-x", "-q"]
            assert v2.coalesce_pytest_command(command) == command
            assert "unit_coalesce_error" in caplog.text and "TypeError" in caplog.text
            return
        else:
            assert v2._coalesce_results(bolt, plan.units[0], {}, "") == (None, False)
    assert [e[1] for e in bolt.events] == ["unit_coalesce_error"]
    assert "TypeError" in bolt.events[0][2]


@pytest.mark.parametrize("quiet", [False, True])
def test_r2_n7_escalation_is_delivered_or_explicitly_skipped(coalesce_on, monkeypatch, quiet):
    bolt = Bolt(_small_plan(("a",)))
    if quiet:
        monkeypatch.setattr(bolt, "organ_out", lambda *args: None)
    _coalesce()._escalate(bolt, bolt.unit_plan.units[0].unit_id, "budget exhausted")
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]
    prefix = json.dumps(messages)
    v2.refresh_coalesced_plan(bolt, messages)
    assert json.dumps(messages[:2]) == prefix
    if quiet:
        assert len(messages) == 2
        assert any(e[1] == "unit_plan_refresh_skipped" and "suppressed" in e[2] for e in bolt.events)
    else:
        assert len(messages) == 3 and "stop retrying this unit" in messages[-1]["content"]
    assert bolt.measurement._unit_coalesce_escalated == bolt.unit_plan.units[0].unit_id


def test_r2_n5_quiet_merge_records_missing_delivery(coalesce_on, monkeypatch):
    plan = _small_plan(("helper", "use"))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    monkeypatch.setattr(bolt, "organ_out", lambda *args: None)
    for i in range(3):
        _coalesce().on_gate_failure(bolt, plan, plan.units[1], v2.GateVerdict(False, ("test_helper",)),
                                    {"use.py": f"x={i}"}, coalesce_on)
    assert bolt.unit_plan is not plan
    assert any(e[1] == "unit_plan_refresh_skipped" and "suppressed" in e[2] for e in bolt.events)


def test_r2_partial13_switch_has_one_shared_parser():
    assert v2._coalesce_enabled is _coalesce().enabled
    for value in ("off", "on", " ON ", "junk", "", "0"):
        env = {COALESCE_ENV: value}
        if value.strip().lower() in {"on", "off"}:
            assert _coalesce().mode(env) == value.strip().lower()
        else:
            with pytest.raises(ValueError):
                _coalesce().mode(env)


@pytest.mark.parametrize("proposal", ["ancestor", "current_tree", "v1"])
def test_r2_n5_resume_install_keeps_coalesced_plan_and_tail(coalesce_on, proposal):

    from unit_planner import render_plan_block
    plan = _small_plan(("a", "b", "c"))
    bolt = Bolt(plan)
    merged = _coalesce().coalesce(plan, _coalesce().CoalesceTrigger(
        "integration", plan.units[1].unit_id, plan.units[0].unit_id, "a.py:a")).plan
    bolt.unit_plan = merged
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task\n" + render_plan_block(plan)},
                {"role": "user", "content": "[planner] supersedes digest " + plan.plan_digest + "\n" + render_plan_block(merged)}]
    bolt, messages = pickle.loads(pickle.dumps((bolt, messages)))
    prefix = json.dumps(messages)
    proposed = plan
    if proposal == "current_tree":
        proposed = replace(plan, index_revision="current edited tree")
        proposed = replace(proposed, plan_digest=v2._digest(proposed._basis()))
        assert proposed.plan_digest not in merged.lineage
    elif proposal == "v1":
        from unit_planner import UnitPlan
        proposed = UnitPlan(plan.task_id, plan.source_path, plan.units, plan.handoff_rule,
                            plan.plan_digest, plan.handoff_policy)
    assert v2.install_unit_plan(bolt, proposed, coalesce_on) is bolt.unit_plan
    assert bolt.unit_plan.plan_digest == merged.plan_digest
    v2.weld_initial_plan(bolt, messages)
    assert json.dumps(messages) == prefix
    assert any(e[1] == "unit_plan_weld_skipped" for e in bolt.events)



def test_r2_n8_parity_and_contract_pins_are_identified():
    assert "contract pin" in (test_review_10_measurement_only_bolt_matches_kernel.__doc__ or "")
    assert "parity pins" in (test_review_04_host_test_command_proves_complete_gate.__doc__ or "")
    assert "parity pin" in (test_review_11_off_matches_baseline_recorded_request.__doc__ or "")


@pytest.mark.parametrize("failed", [(), ("test_a",)])
def test_r2_n1_real_runner_without_runtime(coalesce_on, monkeypatch, failed):
    bolt = Bolt(_small_plan(("a",)), int(bool(failed)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    observation = _real_pytest_observation(("test_a",), failed, runtime_present=False)
    verdict = v2.gate_and_advance(bolt, coalesce_on, observation)
    assert verdict is not None and verdict.passed == (not failed)


@pytest.mark.parametrize("bad_child", [None, "timeout", "truncated"])
def test_r2_n1_multi_selector_retains_each_child_evidence(coalesce_on, monkeypatch, bad_child):
    from probes import react_repair
    plan = _small_plan(("a",))
    plan = replace(plan, units=(replace(plan.units[0], gate_tests=("test_a", "test_b")),))
    bolt = Bolt(plan)
    first = _real_pytest_observation(("test_a",), ())
    second = (_real_pytest_observation(("test_b",), (), truncated=bad_child == "truncated")
              if bad_child != "timeout" else "pytest exit 1 — TIMED OUT")
    monkeypatch.setattr(react_repair, "run_tests", lambda root, fn, selector, *args, **kw:
                        first if selector == "first" else second)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    observation, codes = react_repair.run_tests_for_selectors(coalesce_on, "gate", ("first", "second"))
    bolt.last_test_exit = 1 if bad_child == "timeout" else 0
    verdict = v2.gate_and_advance(bolt, coalesce_on, observation)
    assert codes == ((0, 1) if bad_child == "timeout" else (0, 0))
    if bad_child:
        assert verdict is None and bolt.unit_index == 0
        assert [e[1] for e in bolt.events] == ["unit_gate_unobserved"]
        assert "incomplete_capture" in bolt.events[0][2]
    else:
        assert verdict is not None and verdict.passed


def test_r2_n4_real_fail_fast_does_not_pass_unrun_gate(coalesce_on):
    plan = _small_plan(("a",))
    plan = replace(plan, units=(replace(plan.units[0], gate_tests=("test_a", "test_b")),))
    bolt = Bolt(plan, 1)
    observation = _pytest_observation(plan, ("test_a",))
    assert v2.gate_and_advance(bolt, coalesce_on, observation) is None
    assert bolt.unit_index == 0 and not hasattr(bolt, "unit_gate_failures")
    assert [e[1] for e in bolt.events] == ["unit_gate_unobserved"]
    assert "test_b" in bolt.events[0][2]


@pytest.mark.parametrize("failure", ["render", "organ"])
def test_r2_n6_merge_error_does_not_half_install_plan(coalesce_on, monkeypatch, failure):
    plan = _small_plan(("a", "b", "c"))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    uc = _coalesce()
    uc.observe_tree(bolt, plan, {"b.py": "x=0"})
    def broken(*args):
        raise TypeError("advisory failure")
    monkeypatch.setattr(uc if failure == "render" else bolt,
                        "render_plan_block" if failure == "render" else "organ_out", broken)
    for i in (1, 2):
        v2._coalesce_gate_hook(bolt, plan, plan.units[1], v2.GateVerdict(False, ("test_a",)),
                               {"b.py": f"x={i}"}, coalesce_on, 1, True)
    assert bolt.unit_plan is plan and bolt.unit_index == 1
    assert not getattr(bolt, "unit_plan_refresh", None)
    assert any(e[1] == "unit_coalesce_error" and "TypeError" in e[2] for e in bolt.events)


def test_r2_n5_real_setup_does_not_reweld_stale_ancestor(coalesce_on, monkeypatch):
    from unit_planner import render_plan_block
    plan = _small_plan(("a", "b", "c"))
    bolt = Bolt(plan)
    merged = _coalesce().coalesce(plan, _coalesce().CoalesceTrigger(
        "integration", plan.units[1].unit_id, plan.units[0].unit_id, "a.py:a")).plan
    bolt.unit_plan = merged
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task\n" + render_plan_block(plan)},
                {"role": "user", "content": "[planner] supersedes digest " + plan.plan_digest + "\n" + render_plan_block(merged)}]
    before = json.dumps(messages)
    monkeypatch.setattr(v2, "derive_with_index", lambda *args: plan)
    source = Path("src/stateframe/engine/probes/react_repair.py").read_text()
    setup = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Try)
                 and any(isinstance(stmt, ast.ImportFrom) and stmt.module == "unit_planner"
                         and any(alias.name == "derive_unit_plan" for alias in stmt.names) for stmt in n.body))
    exec(compile(ast.Module(body=[setup], type_ignores=[]), "real-plan-setup", "exec"),  # noqa: S102
         {"d": {"repo": "build"}, "root": coalesce_on, "bolt": bolt, "msgs": messages})
    assert bolt.unit_plan is merged and bolt.unit_plan_v2_on
    assert json.dumps(messages) == before


@pytest.mark.parametrize("setting", [None, "off"])
def test_r2_n1_off_return_channel_preserves_string_identity(monkeypatch, setting):
    monkeypatch.setenv(v2.MODE_ENV, "on")
    if setting is None:
        monkeypatch.delenv(COALESCE_ENV, raising=False)
    else:
        monkeypatch.setenv(COALESCE_ENV, setting)
    text = "pytest exit 0 — compact receipt"
    assert v2.coalesce_observation(text, "raw output", 0) is text
    assert v2.coalesce_combined_observation(text, (text,)) is text
    assert type(v2.coalesce_observation(text, "raw output", 0)) is str


@pytest.mark.parametrize("pointer", [0, 1])
@pytest.mark.parametrize("failure", ["observe_tree", "advance_passed"])
def test_r2_n3_real_pass_survives_advice_error_including_final_unit(coalesce_on, monkeypatch, pointer, failure):
    plan = _small_plan(("a", "b"))
    bolt = Bolt(plan)
    bolt.unit_index = pointer
    observation = _pytest_observation(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    def broken(*args):
        raise TypeError("advisory failure")
    monkeypatch.setattr(_coalesce(), failure, broken)
    verdict = v2.gate_and_advance(bolt, coalesce_on, observation)
    assert verdict is not None and verdict.passed
    assert bolt.unit_index == pointer + 1
    assert plan.units[pointer].unit_id in bolt.unit_gate_passed
    assert any(e[1] == "unit_coalesce_error" for e in bolt.events)
    if pointer == 1:
        assert "all unit gates passed" in v2.render_unit_progress(bolt)
        before = list(bolt.events)
        assert v2.gate_and_advance(bolt, coalesce_on, observation) is None
        assert bolt.events == before



def test_r2_n3_advice_error_still_skips_proven_units(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b", "c"))
    bolt = Bolt(plan)
    bolt.unit_gate_passed = {plan.units[1].unit_id}
    observation = _pytest_observation(plan)
    _seed_pass_sources(bolt, coalesce_on, observation)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    def broken(*args):
        raise TypeError("advisory failure")
    monkeypatch.setattr(_coalesce(), "observe_tree", broken)
    assert v2.gate_and_advance(bolt, coalesce_on, observation).passed
    assert bolt.unit_index == 2
    assert bolt.unit_gate_passed == {plan.units[0].unit_id, plan.units[1].unit_id}
    assert any(e[1] == "unit_coalesce_error" for e in bolt.events)


# Round 3: exercise the production runner, gate dispatch and frame projection.
def _r3_runner(root, bolt, monkeypatch, *, timeout=False, skip_db=True, check_timeout=30):
    import kernel_check_receipts
    import kernel_execution_policy
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair as react

    # Keep nested pytest node IDs relative to its fixture repository, including
    # when the outer test's basetemp lives inside this worktree.
    if not (root / "pytest.ini").exists():
        (root / "pytest.ini").write_text("[pytest]\n")
    runtime = bolt.measurement
    runtime.owner = bolt
    runtime.binding = dict
    runtime.test_observation = lambda *args, **kwargs: None
    monkeypatch.setenv("STATEFRAME_TEST_RUN_SPANS", "off")
    monkeypatch.setattr(react, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setattr(kernel_execution_policy, "policy_read_grants", lambda result: ())
    monkeypatch.setattr(kernel_check_receipts, "sandbox_open_denial", lambda *args: None)

    def run(command, cwd, **kwargs):
        if timeout and "--collect-only" not in command:
            raise subprocess.TimeoutExpired(command, 300, output="partial stdout")
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                   timeout=check_timeout, check=False,
                                   env={**kwargs["env"], "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
        return SimpleNamespace(completed=completed, timed_out=False, output_truncated=False)

    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", run)

    def observe(selector):
        token = ACTIVE_RUNTIME.set(runtime)
        try:
            observation = react._run_tests_impl(root, "gate", selector,
                                               row={"_skip_db": skip_db, "python": sys.executable})
        finally:
            ACTIVE_RUNTIME.reset(token)
        bolt.last_test_exit = react._selector_exit_code(observation)
        return observation
    return observe


def _r3_gate(bolt, root, observation, messages=None):
    source = Path("src/stateframe/engine/probes/react_repair.py").read_text()
    block = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.BoolOp) and "unit_plan_v2_on" in ast.unparse(n.test))
    exec(compile(ast.Module(body=block.body, type_ignores=[]), "production-unit-gate", "exec"),  # noqa: S102
         {"bolt": bolt, "root": root, "plain_observation": str(observation),
          "_unit_gate_observation": observation, "msgs": messages if messages is not None else []})


@pytest.mark.parametrize("notice_kind", ["merge", "escalate"])
@pytest.mark.parametrize("scoped", [False, True])
def test_r3_p1_1_n5_n7_notice_preserves_native_exchange(coalesce_on, monkeypatch, notice_kind, scoped):
    import kernel_stateframe_framefit5 as framefit
    from kernel_stateframe_framefit5 import latest_complete_tool_exchange
    from probes import react_repair as react
    if scoped:
        monkeypatch.setattr(framefit, "has_framefit5_state", lambda messages: True)
    bolt = Bolt(_small_plan(("a", "b")))
    bolt.unit_plan = replace(bolt.unit_plan, units=(bolt.unit_plan.units[0],
                             replace(bolt.unit_plan.units[1], gate_tests=("test_a",))))
    bolt.unit_index = 1
    (coalesce_on / "test_gate.py").write_text("def test_a(): assert False\ndef test_b(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in bolt.unit_plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    v2.seed_coalesce(bolt, coalesce_on)
    if notice_kind == "escalate":
        # The existing merged-unit allowance escalates on its next exhausted gate.
        bolt.unit_plan = replace(bolt.unit_plan, coalesce_events=2)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Task: repair"}]
    prefix = json.dumps(messages)
    from kernel_public_evidence import render_scoped_check_truth
    selectors = ("test_gate.py::test_a", "test_gate.py::test_b")
    scope_line = render_scoped_check_truth(selectors, selectors[:1], {selectors[0]: 1}) if scoped else None
    for i in (1, 2):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"test{i}", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]})
        (coalesce_on / "b.py").write_text(f"x={i}\n")
        observation = observe("test_gate.py")
        _r3_gate(bolt, coalesce_on, observation, messages)
        # Execute the actual fallthrough delivery statements, including any notice enrichment.
        tree = ast.parse(Path(react.__file__).read_text())
        append = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call)
                      and ast.unparse(n).startswith("msgs.append(_tool_reply_for_call(msgs, tool_call_id, delivered_obs)"))
        parent = next(n for n in ast.walk(tree) if isinstance(getattr(n, "body", None), list)
                      and append in n.body)
        start = next(j for j, n in enumerate(parent.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n) == "delivered_obs = obs")
        end = parent.body.index(append)
        delivered_observation = str(observation) + "\n" + scope_line if scope_line else observation
        exec(compile(ast.Module(body=parent.body[start:end + 1], type_ignores=[]), "production-tool-delivery", "exec"),  # noqa: S102
             {"bolt": bolt, "obs": delivered_observation, "msgs": messages, "line": "test test_gate.py",
              "_scope_truth_line": scope_line,
              "tool_call_id": f"test{i}", "_tool_reply_for_call": react._tool_reply_for_call})
        assert latest_complete_tool_exchange(messages) is not None
        if scope_line:
            assert messages[-1]["content"].endswith("\n" + scope_line)
            assert messages[-1]["content"].count(scope_line) == 1
    assert json.dumps(messages[:2]) == prefix
    assert messages[-1]["role"] == "tool"
    expected = "supersedes digest" if notice_kind == "merge" else "stop retrying this unit"
    assert expected in messages[-1]["content"]


def test_r3_p1_2_merge_invalidates_consumers_and_fresh_red_revokes_pass(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b", "c"), deps={"b": ("a",), "c": ("a",)})
    plan = replace(plan, units=(replace(plan.units[0], gate_tests=("test_c",)), *plan.units[1:]))
    bolt = Bolt(plan)
    bolt.unit_index = 2
    bolt.unit_gate_passed = {plan.units[0].unit_id, plan.units[1].unit_id}
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): pass\ndef test_c(): assert False\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    v2.seed_coalesce(bolt, coalesce_on)
    for i in (1, 2):
        (coalesce_on / "c.py").write_text(f"x={i}\n")
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_plan is not plan
    assert plan.units[1].unit_id not in bolt.unit_gate_passed
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_c(): pass\ndef test_b(): assert False\n")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 1 and "all unit gates passed" not in v2.render_unit_progress(bolt)
    # Cached completion must be revoked even if a later full run arrives after completion.
    bolt.unit_gate_passed.add(plan.units[1].unit_id)
    bolt.unit_index = len(bolt.unit_plan.units)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 1 and plan.units[1].unit_id not in bolt.unit_gate_passed


def test_r3_p1_3_partial4_fail_fast_counts_only_edited_repairs(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b"))
    plan = replace(plan, units=(plan.units[0], replace(plan.units[1], gate_tests=("test_a", "test_b"))))
    bolt = Bolt(plan)
    bolt.unit_index = 1
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text("def test_a(): assert False\ndef test_b(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    v2.seed_coalesce(bolt, coalesce_on)
    for i in (0, 1, 1, 2):
        (coalesce_on / "b.py").write_text(f"x={i}\n")
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
        assert getattr(bolt, "unit_gate_failures", {}).get(plan.units[1].unit_id, 0) == min(i, 2)
    assert bolt.unit_plan is not plan


@pytest.mark.parametrize("kind", ["parameter", "class", "file"])
def test_r3_p1_4_partial4_passing_selector_does_not_prove_whole_function(coalesce_on, monkeypatch, kind):
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    if kind == "parameter":
        source = "import pytest\n@pytest.mark.parametrize('value',[True,False],ids=['good','bad'])\ndef test_a(value): assert value\n"
        selector = "test_gate.py::test_a[good]"
    elif kind == "class":
        source = "class TestGood:\n def test_a(self): pass\nclass TestBad:\n def test_a(self): assert False\n"
        selector = "test_gate.py::TestGood::test_a"
    else:
        source = "def test_a(): pass\n"
        (coalesce_on / "test_other.py").write_text("def test_a(): assert False\n")
        selector = "test_gate.py::test_a"
    (coalesce_on / "test_gate.py").write_text(source)
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    _r3_gate(bolt, coalesce_on, observe(selector))
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())
    assert any(e[1] == "unit_gate_unobserved" for e in bolt.events)


def test_r3_p2_5_n5_current_plan_persists_in_real_frame_and_resume(coalesce_on, monkeypatch):
    from kernel_stateframe_framefit5 import build_typed_state
    bolt = _run_break_scenario(coalesce_on)
    plan = bolt.unit_plan
    assert plan.lineage
    from unit_planner import render_plan_block
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: repair\n" + render_plan_block(v2.derive_with_index(row(), coalesce_on))}]
    v2.refresh_coalesced_plan(bolt, messages)
    # A newer action has displaced the ephemeral notice from newest-user context.
    messages.append({"role": "assistant", "content": "inspect next"})
    bolt, messages = pickle.loads(pickle.dumps((bolt, messages)))
    context = {"journal_receipt_records": [], "ledger": {"rows": []}}
    runtime = SimpleNamespace(owner=bolt, binding=dict)
    bolt.root = coalesce_on
    frame = build_typed_state(messages, context, runtime)
    current = frame.get("unit_plan")
    assert current is not None and current["plan_digest"] == plan.plan_digest
    assert current["lineage"] == list(plan.lineage)
    assert current["units"] == plan.to_dict()["units"]
    bolt.unit_escalate = plan.units[0].unit_id
    assert build_typed_state(messages, context, runtime)["unit_plan"]["escalated_unit"] == bolt.unit_escalate


@pytest.mark.parametrize("runtime_present", [False, True])
def test_r3_p2_6_timeout_never_passes_seam_only_unit(coalesce_on, monkeypatch, runtime_present):
    plan = _small_plan(("a",))
    plan = replace(plan, units=(replace(plan.units[0], gate_tests=()),))
    bolt = Bolt(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch, timeout=True)
    if not runtime_present:
        # Exercise the runner's no-runtime branch while retaining the same subprocess.
        from kernel_pilot import ACTIVE_RUNTIME
        from probes import react_repair
        token = ACTIVE_RUNTIME.set(None)
        try:
            observation = react_repair._run_tests_impl(coalesce_on, "gate", "test_gate.py",
                                                       row={"_skip_db": True, "python": sys.executable})
        finally:
            ACTIVE_RUNTIME.reset(token)
        bolt.last_test_exit = 1
    else:
        observation = observe("test_gate.py")
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())
    assert observation.unit_gate_runs and not observation.unit_gate_runs[0].complete


@pytest.mark.parametrize("quiet", [False, True])
def test_r3_partial6_escalation_is_consumed_by_real_loop(coalesce_on, monkeypatch, quiet):
    from kernel_native_tools import NativeToolBatch
    if quiet:
        monkeypatch.setattr(Bolt, "organ_out", lambda *args: None)
    bolt = _run_break_scenario(coalesce_on)
    merged = bolt.unit_plan
    for _ in range(_coalesce().allowance(merged.units[0])):
        _planted_break(bolt, coalesce_on, pointer=0)
    assert bolt.unit_escalate == merged.units[0].unit_id
    tree = ast.parse(Path("src/stateframe/engine/probes/react_repair.py").read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and ast.unparse(n.iter).startswith("step_indices("))
    stop = next((n for n in loop.body if isinstance(n, ast.If)
                 and "consume_coalesce_escalation" in ast.unparse(n.test)), None)
    assert stop is not None, "escalation needs a loop consumer before the next model request"
    # A one-iteration wrapper makes the production break observable without running a task.
    wrapper = ast.For(target=ast.Name(id="iteration", ctx=ast.Store()),
                      iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                      body=[stop, ast.Assign(targets=[ast.Name(id="requested", ctx=ast.Store())], value=ast.Constant(True))],
                      orelse=[])
    batch = NativeToolBatch()
    batch.pending = [{"id": "queued-edit", "type": "function", "function": {"name": "str_replace", "arguments": "{}"}}]
    batch.journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    finish = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "finish_native_batch_transcript")
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Task: repair"}]
    env = {"bolt": bolt, "requested": False, "consume_coalesce_escalation": v2.consume_coalesce_escalation,
           "native_batch": batch, "msgs": messages}
    exec(compile(ast.Module(body=[finish], type_ignores=[]), "production-batch-finish", "exec"), env)  # noqa: S102
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "production-escalation-stop", "exec"), env)  # noqa: S102
    assert not env["requested"]
    assert bolt.measurement.system_outcome == "unit_coalesce_escalated"
    assert batch.finalized and not batch.pending
    assert messages[-1]["tool_call_id"] == "queued-edit" and "NOT EXECUTED" in messages[-1]["content"]
    # Restored attempts retain the stop even if their outcome sink was rebuilt.
    bolt.measurement.system_outcome = None
    assert v2.consume_coalesce_escalation(bolt)
    assert bolt.measurement.system_outcome == "unit_coalesce_escalated"
    assert sum(e[1] == "unit_coalesce_stopped" for e in bolt.events) == 1


def test_r3_p1_2_provider_edit_rechecks_cached_dependents(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b", "c"), deps={"b": ("a",)})
    bolt = Bolt(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): pass\ndef test_c(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 2
    (coalesce_on / "a.py").write_text("x=1\n")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 0
    assert plan.units[1].unit_id not in bolt.unit_gate_passed
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 1
    _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 3 and "all unit gates passed" in v2.render_unit_progress(bolt)


def test_r3_p1_4_selector_evidence_accumulates_and_test_edit_expires_it(coalesce_on, monkeypatch):
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value',[1,2],ids=['one','two'])\ndef test_a(value): assert value > 0\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[one]"))
    assert bolt.unit_index == 0
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[two]"))
    assert bolt.unit_index == 1
    bolt.unit_index = 0
    bolt.unit_gate_passed.clear()
    (coalesce_on / "test_gate.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value',[0,2],ids=['one','two'])\ndef test_a(value): assert value > 0\n")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[two]"))
    assert bolt.unit_index == 0 and not bolt.unit_gate_passed


@pytest.mark.parametrize("selector", ["missing.py::test_a", None])
def test_r3_p2_6_every_unrun_return_has_incomplete_evidence(coalesce_on, monkeypatch, selector):
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair
    bolt = Bolt(_small_plan(("a",)))
    plan = bolt.unit_plan
    bolt.unit_plan = replace(plan, units=(replace(plan.units[0], gate_tests=()),))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    token = ACTIVE_RUNTIME.set(None)
    try:
        observation = react_repair._run_tests_impl(coalesce_on, "missing", selector)
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert observation.unit_gate_runs and not observation.unit_gate_runs[0].complete
    bolt.last_test_exit = react_repair._selector_exit_code(observation)
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 0


@pytest.mark.parametrize("setting", [None, "off"])
@pytest.mark.parametrize("exit_code", [0, 1, 124])
def test_r3_off_real_runner_matches_pre_review_bytes(v2_on, monkeypatch, base_engine_source, setting, exit_code):
    """Switch-OFF parity pins, intentionally green on the baseline and this engine."""
    import kernel_execution_policy
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair
    monkeypatch.delenv(COALESCE_ENV, raising=False)
    if setting is not None:
        monkeypatch.setenv(COALESCE_ENV, setting)
    monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setenv("STATEFRAME_TEST_RUN_SPANS", "off")
    (v2_on / "test_gate.py").write_text("def test_a(): pass\n")
    calls = []
    def run(command, root, **kwargs):
        calls.append(list(command))
        assert "--collect-only" not in command
        if exit_code == 124:
            raise subprocess.TimeoutExpired(command, 300)
        return SimpleNamespace(completed=subprocess.CompletedProcess(command, exit_code,
                               stdout="test_gate.py::test_a PASSED\n1 passed in 0.01s\n", stderr=""),
                               timed_out=False, output_truncated=False)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", run)
    monkeypatch.setattr(kernel_execution_policy, "policy_read_grants", lambda result: ())
    source = base_engine_source("probes/react_repair")
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_run_tests_impl")
    namespace = dict(vars(react_repair))
    exec(compile(ast.Module(body=[original], type_ignores=[]), "pre-review-runner", "exec"), namespace)  # noqa: S102
    token = ACTIVE_RUNTIME.set(None)
    try:
        before = namespace["_run_tests_impl"](v2_on, "gate", "test_gate.py", row={"python": sys.executable})
        after = react_repair._run_tests_impl(v2_on, "gate", "test_gate.py", row={"python": sys.executable})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert type(after) is type(before) is str and after == before
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize("refusal", ["predispatch", "host"])
def test_r3_p2_6_multi_refusal_cannot_borrow_child_pass(coalesce_on, monkeypatch, refusal):
    import kernel_execution_policy
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\n")
    _r3_runner(coalesce_on, bolt, monkeypatch)
    second = "missing.py"
    if refusal == "host":
        second = "test_refused.py"
        (coalesce_on / second).write_text("def test_other(): pass\n")
        original = kernel_execution_policy.run_sandboxed
        def refuse(command, *args, **kwargs):
            if second in command:
                raise RuntimeError("host policy refused test")
            return original(command, *args, **kwargs)
        monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", refuse)
    token = ACTIVE_RUNTIME.set(None)
    try:
        observation, codes = react_repair.run_tests_for_selectors(
            coalesce_on, "gate", ("test_gate.py", second), row={"_skip_db": True, "python": sys.executable})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert codes == (0, -1)
    bolt.last_test_exit = react_repair._selector_exit_code(observation)
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())
    assert observation.unit_gate_runs and any(not run.complete for run in observation.unit_gate_runs)


@pytest.mark.parametrize("setting", [None, "off"])
@pytest.mark.parametrize("branch", ["pass", "fail", "host", "predispatch", "empty", "generic"])
def test_r3_off_selector_aggregation_matches_pre_review_bytes(v2_on, monkeypatch, base_engine_source, setting, branch):
    """Switch-OFF parity pins, intentionally green on the baseline and this engine."""
    import kernel_generic_check
    from probes import react_repair
    monkeypatch.delenv(COALESCE_ENV, raising=False)
    if setting is not None:
        monkeypatch.setenv(COALESCE_ENV, setting)
    first = "pytest exit 0 — passed\ntest_gate.py::test_a PASSED\n"
    second = {"pass": first, "fail": "pytest exit 1 — failed\nFAILED test_gate.py::test_b\n",
              "host": react_repair.HOST_REFUSED_PREFIX + " policy refused",
              "predispatch": "[selector] REFUSED_PREDISPATCH — missing test",
              "generic": first, "empty": first}[branch]
    monkeypatch.setattr(react_repair, "run_tests", lambda root, fn, selector, *args, **kwargs:
                        first if selector == "first" else second)
    generic = "check exit 0 — generic check"
    monkeypatch.setattr(kernel_generic_check, "check_result", lambda *args: (generic, 0))
    source = base_engine_source("probes/react_repair")
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_tests_for_selectors")
    namespace = dict(vars(react_repair))
    exec(compile(ast.Module(body=[original], type_ignores=[]), "pre-review-selectors", "exec"), namespace)  # noqa: S102
    selectors = () if branch == "empty" else ("first", "second")
    row = {"runner_kind": "generic"} if branch == "generic" else {}
    before = namespace["run_tests_for_selectors"](v2_on, "gate", selectors, row=row)
    after = react_repair.run_tests_for_selectors(v2_on, "gate", selectors, row=row)
    assert after == before and type(after[0]) is type(before[0]) is str


def test_r3_p1_4_collection_follows_existing_oracle_conftest_retry(coalesce_on, monkeypatch):
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate_oracle.py").write_text("def test_a(): pass\n")
    (coalesce_on / "conftest.py").write_text("import stateframe_deliberately_missing_fixture_dependency\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch, skip_db=False)
    observation = observe("test_gate_oracle.py")
    assert bolt.last_test_exit == 0
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 1
    assert observation.unit_gate_runs[0].collected == ("test_gate_oracle.py::test_a",)


# Round 4: actual pytest -> compact receipt -> production gate dispatch.
def test_r4_p1_1_merge_preserves_revoked_pass_and_rechecks(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b", "c"))
    plan = replace(plan, units=(*plan.units[:2], replace(plan.units[2], gate_tests=("test_b",))))
    bolt = Bolt(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text(
        "from pathlib import Path\n"
        "def test_a(): assert Path('a.py').read_text() == 'x=0\\n'\n"
        "def test_b(): assert Path('c.py').read_text() == 'x=0\\n'\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    for _ in range(2):
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 2 and len(bolt.unit_gate_passed) == 2
    for i in (1, 2):
        (coalesce_on / "a.py").write_text("x=1\n")
        (coalesce_on / "c.py").write_text(f"x={i}\n")
        _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert bolt.unit_plan is not plan
    assert plan.units[0].unit_id not in bolt.unit_gate_passed
    assert plan.units[0].unit_id in bolt.unit_gate_rechecks
    assert set(bolt.unit_gate_rechecks) <= {u.unit_id for u in bolt.unit_plan.units}
    (coalesce_on / "c.py").write_text("x=0\n")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert bolt.unit_index == 0
    assert "all unit gates passed" not in v2.render_unit_progress(bolt)


@pytest.mark.parametrize("addopts", [
    "--deselect=test_gate.py::test_a[bad]", "-k good", "-m selected",
    "--ignore=test_other.py", "--ignore-glob=*other.py",
])
def test_r4_p1_2_collection_strips_environment_selection(coalesce_on, monkeypatch, addopts):
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value', [True, False], ids=['good','bad'])\n"
        "def test_a(value): assert value\n")
    (coalesce_on / "test_other.py").write_text("def test_a(): assert False\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", addopts)
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    observation = observe("test_gate.py::test_a[good]")
    _r3_gate(bolt, coalesce_on, observation)
    inventory = observation.unit_gate_runs[0].collected
    assert inventory is not None and "test_gate.py::test_a[bad]" in inventory
    assert "test_other.py::test_a" in inventory
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())


def test_r4_p1_3_partial_failure_merges_only_confirmed_owner(coalesce_on, monkeypatch):
    plan = _small_plan(("a", "b", "c"))
    plan = replace(plan, units=(*plan.units[:2], replace(plan.units[2], gate_tests=("test_a", "test_b"))))
    bolt = Bolt(plan)
    bolt.unit_index = 2
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): assert False\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    v2.seed_coalesce(bolt, coalesce_on)
    for i in (1, 1, 2):
        (coalesce_on / "c.py").write_text(f"x={i}\n")
        _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
        assert bolt.unit_gate_failures[plan.units[2].unit_id] == i
    assert bolt.unit_plan is not plan
    merged = next(u for u in bolt.unit_plan.units if u.coalesced_from)
    assert set(merged.coalesced_from) == {plan.units[1].unit_id, plan.units[2].unit_id}
    assert any(e[1] == "unit_coalesce" and "surface=test:test_b" in e[2] for e in bolt.events)


@pytest.mark.parametrize("change", ["source", "inventory", "config", "conftest"])
def test_r4_p1_4_completed_pass_expires_with_test_identity(coalesce_on, monkeypatch, change):
    plan = _small_plan(("a", "b"))
    bolt = Bolt(plan)
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    source = (
        "import os, pytest\n"
        "@pytest.mark.parametrize('value', [True] if os.environ.get('GATE_MORE_CASES') != '1' else [True,False])\n"
        "def test_a(value): assert value\n"
        "def test_b(): pass\n")
    (coalesce_on / "test_gate.py").write_text(source)
    observe = _r3_runner(coalesce_on, bolt, monkeypatch, skip_db=False)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a"))
    assert bolt.unit_index == 1 and plan.units[0].unit_id in bolt.unit_gate_passed
    if change == "source":
        (coalesce_on / "test_gate.py").write_text(source.replace("[True] if", "[True,False] if"))
    elif change == "inventory":
        monkeypatch.setenv("GATE_MORE_CASES", "1")
    elif change == "config":
        (coalesce_on / "pytest.ini").write_text("[pytest]\nxfail_strict = true\n")
    else:
        (coalesce_on / "conftest.py").write_text("# changed harness\n")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert plan.units[0].unit_id not in bolt.unit_gate_passed
    assert plan.units[0].unit_id in bolt.unit_gate_rechecks
    assert bolt.unit_index == 0 and "all unit gates passed" not in v2.render_unit_progress(bolt)


def test_r4_p2_5_collection_timeout_still_executes_requested_selector(coalesce_on, monkeypatch):
    import kernel_execution_policy
    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    original = kernel_execution_policy.run_sandboxed
    calls = []
    def collection_timeout(command, *args, **kwargs):
        calls.append(tuple(command))
        if "--collect-only" in command:
            raise subprocess.TimeoutExpired(command, 300, output="inventory hung")
        return original(command, *args, **kwargs)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", collection_timeout)
    observation = observe("test_gate.py::test_a")
    assert len(calls) == 2 and "--collect-only" not in calls[-1]
    assert bolt.last_test_exit == 0 and "TIMED OUT" not in observation
    assert observation.unit_gate_runs[0].collected is None
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())
    assert any(e[1] == "unit_gate_collection_error" for e in bolt.events)


def test_r4_p1_1_restored_recheck_overrides_completed_pointer(coalesce_on, monkeypatch):
    import io

    import kernel_checkpoint

    bolt = Bolt(_small_plan(("a", "b")))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    for _ in range(2):
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 2
    # A restored earlier attempt may contain both a stale pass and an explicit
    # recheck. The recheck always wins, even past the end of the plan.
    bolt.unit_gate_rechecks = {bolt.unit_plan.units[0].unit_id}
    context = {name: {} for name in ("framecells", "framefuncs", "cvars", "hostfns", "participants")}
    bolt = kernel_checkpoint._Unpickler(io.BytesIO(kernel_checkpoint.dumps(bolt, context)), context).load()
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert bolt.unit_index == 0
    assert bolt.unit_plan.units[0].unit_id not in bolt.unit_gate_passed
    assert "all unit gates passed" not in v2.render_unit_progress(bolt)


@pytest.mark.parametrize("options", [
    ("--deselect", "test_gate.py::test_a[bad]"),
    ("--deselect=test_gate.py::test_a[bad]",),
    ("--ignore", "ignored-tests"), ("--ignore-glob=*other.py",),
    ("-kgood",), ("-mselected",), ("--lf",), ("--stepwise",),
])
def test_r4_p1_2_collection_removes_command_selection(coalesce_on, monkeypatch, options):
    import kernel_execution_policy
    bolt = Bolt(_small_plan(("a",)))
    (coalesce_on / "test_gate.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value',[True,False],ids=['good','bad'])\n"
        "def test_a(value): assert value\n")
    (coalesce_on / "test_other.py").write_text("def test_a(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    original_command = v2.coalesce_pytest_command
    monkeypatch.setattr(v2, "coalesce_pytest_command", lambda command: original_command(command) + list(options))
    original_runner = kernel_execution_policy.run_sandboxed
    inventories = []
    def capture(command, *args, **kwargs):
        if "--collect-only" in command:
            inventories.append(tuple(command))
        return original_runner(command, *args, **kwargs)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", capture)
    observation = observe("test_gate.py::test_a[good]")
    assert len(inventories) == 1
    assert not any(option in inventories[0] for option in options)
    assert observation.unit_gate_runs[0].collected == (
        "test_gate.py::test_a[good]", "test_gate.py::test_a[bad]", "test_other.py::test_a")


def test_r4_p1_4_identity_bound_cannot_retain_completed_passes(coalesce_on, monkeypatch):
    from unit_gate_evidence import MAX_FILE_BYTES

    bolt = Bolt(_small_plan(("a", "b")))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    for _ in range(2):
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 2
    (coalesce_on / "test_added.py").write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert not bolt.unit_gate_passed
    assert bolt.unit_index == 0 and "all unit gates passed" not in v2.render_unit_progress(bolt)
    assert any(e[1] == "unit_coalesce_error" for e in bolt.events)


def test_r4_p1_4_missing_legacy_pass_identity_requires_fresh_evidence(coalesce_on, monkeypatch):
    bolt = Bolt(_small_plan(("a", "b")))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\ndef test_b(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    for _ in range(2):
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_index == 2
    del bolt.unit_gate_pass_sources  # older checkpoints have no test/collection binding
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert not bolt.unit_gate_passed
    assert bolt.unit_index == 0 and "all unit gates passed" not in v2.render_unit_progress(bolt)


# Round 5: runner receipts and production dispatch must qualify the same tree.
@pytest.mark.parametrize("capture", ["truncated", "timeout", "missing_inventory"])
def test_r5_p1_1_incomplete_red_expires_cached_case(coalesce_on, monkeypatch, capture):
    import io

    import kernel_checkpoint
    import kernel_execution_policy
    from unit_gate_evidence import statuses

    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text(
        "import os, pytest\n"
        "@pytest.mark.parametrize('value', [1,2], ids=['one','two'])\n"
        "def test_a(value): assert value != 1 or os.environ.get('GATE_RED') != '1'\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    original = kernel_execution_policy.run_sandboxed

    def incomplete(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if os.environ.get("GATE_RED") == "1":
            if "--collect-only" in command:
                if capture == "missing_inventory":
                    result.output_truncated = True
            elif capture == "timeout":
                raise subprocess.TimeoutExpired(command, 300, output=result.completed.stdout)
            elif capture == "truncated":
                result.output_truncated = True
        return result

    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", incomplete)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[one]"))
    assert bolt.unit_gate_node_evidence[1]["test_gate.py::test_a[one]"] is True
    monkeypatch.setenv("GATE_RED", "1")
    red = observe("test_gate.py::test_a[one]")
    assert statuses(red.unit_gate_runs[0].stdout)["test_gate.py::test_a[one]"] is False
    _r3_gate(bolt, coalesce_on, red)
    monkeypatch.delenv("GATE_RED")
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[two]"))
    assert bolt.unit_index == 0 and not getattr(bolt, "unit_gate_passed", set())
    # A fresh, complete execution can supersede the red; merely running its
    # sibling cannot. Checkpoint serialization must retain that distinction.
    context = {name: {} for name in ("framecells", "framefuncs", "cvars", "hostfns", "participants")}
    bolt = kernel_checkpoint._Unpickler(io.BytesIO(kernel_checkpoint.dumps(bolt, context)), context).load()
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[one]"))
    assert bolt.unit_index == 1 and bolt.unit_gate_passed == {bolt.unit_plan.units[0].unit_id}


@pytest.mark.parametrize("input_path", [
    "settings.py", "settings.json", ".settings.json", "data/value.bin",
    "env/settings.json", ".stateframe/settings.json", "node_modules/library.json",
])
@pytest.mark.parametrize("completed", [False, True])
def test_r5_p1_2_unowned_inputs_revoke_real_derived_pass(coalesce_on, monkeypatch, input_path, completed):
    # settings.py has no defs, so the index includes it but the plan owns no unit.
    _plant(coalesce_on, {
        "a.py": "from pathlib import Path\ndef a(): return Path(" + repr(input_path) + ").read_bytes() == b'VALID=True\\n'\n",
        "b.py": "def b(): return True\n",
        input_path: "VALID=True\n",
        "test_gate.py": "from a import a\nfrom b import b\ndef test_a(): assert a()\ndef test_b(): assert b()\n",
    })
    plan = v2.derive_with_index({"repo": "build", "msg": "Implement a and b",
                                 "test_src": (coalesce_on / "test_gate.py").read_text(),
                                 "public_gate_paths": ["test_gate.py"]}, coalesce_on)
    assert plan is not None
    a = next(unit for unit in plan.units if unit.symbol == "a")
    b = next(unit for unit in plan.units if unit.symbol == "b")
    assert all(input_path not in unit.files for unit in plan.units)
    if input_path.endswith(".py"):
        assert input_path in plan.index_paths
    bolt = Bolt(plan)
    bolt.unit_index = plan.units.index(a)
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a"))
    assert a.unit_id in bolt.unit_gate_passed
    if completed:
        bolt.unit_index = plan.units.index(b)
        _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
        bolt.unit_index = len(plan.units)
    (coalesce_on / input_path).write_text("VALID=False\n")
    if not completed:
        bolt.unit_index = plan.units.index(b)
    _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_b"))
    assert a.unit_id not in bolt.unit_gate_passed
    assert a.unit_id in bolt.unit_gate_rechecks
    assert bolt.unit_index <= plan.units.index(a)
    assert "all unit gates passed" not in v2.render_unit_progress(bolt)
    # The next real run confirms why A's cached pass must not be retained.
    red = observe("test_gate.py::test_a")
    assert bolt.last_test_exit == 1
    _r3_gate(bolt, coalesce_on, red)
    assert a.unit_id not in bolt.unit_gate_passed


def test_r5_p1_3_exhausted_merged_local_failure_stops_real_loop(coalesce_on, monkeypatch):
    plan = _small_plan(("x", "a", "c"))
    uc = _coalesce()
    merged = uc.coalesce(plan, uc.CoalesceTrigger(
        "integration", plan.units[2].unit_id, plan.units[1].unit_id, "test:test_a")).plan
    unit = merged.units[1]
    bolt = Bolt(merged)
    bolt.unit_index = 1
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    for path in plan.index_paths:
        (coalesce_on / path).write_text("x=0\n")
    (coalesce_on / "test_gate.py").write_text("def test_a(): assert False\ndef test_c(): pass\n")
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    v2.seed_coalesce(bolt, coalesce_on)
    for i in range(1, uc.allowance(unit) + 1):
        (coalesce_on / "c.py").write_text(f"x={i}\n")
        _r3_gate(bolt, coalesce_on, observe("test_gate.py"))
    assert bolt.unit_gate_failures[unit.unit_id] == uc.allowance(unit)
    assert getattr(bolt, "unit_escalate", None) == unit.unit_id
    assert v2.consume_coalesce_escalation(bolt)
    assert bolt.measurement.system_outcome == "unit_coalesce_escalated"
    assert any(e[1] == "unit_coalesce_escalate" for e in bolt.events)


@pytest.mark.parametrize("partner", [None, "unit:x:g0"])
def test_r5_p1_3_pure_exhaustion_precedes_partner_and_reseam(coalesce_on, partner):
    uc = _coalesce()
    plan = _small_plan(("x", "a", "c"))
    plan = uc.coalesce(plan, uc.CoalesceTrigger(
        "integration", plan.units[2].unit_id, plan.units[1].unit_id, "test:test_a")).plan
    unit = plan.units[1]
    outcome = uc.coalesce(plan, uc.CoalesceTrigger("unmapped", unit.unit_id, partner, "test:test_a"),
                          failures={unit.unit_id: uc.allowance(unit)})
    assert outcome.escalate and outcome.plan is plan and outcome.unit_index == 1
    assert "out of gate attempts" in outcome.reason


def test_r5_p1_4_large_plan_fits_real_frame_and_resume(coalesce_on, monkeypatch):
    from kernel_stateframe_framefit5 import MAX_PRODUCT_BYTES, build_typed_state
    from unit_planner import render_plan_block

    files = {f"p{i}.py": f"def f{i}(): return True\n" for i in range(v2.MAX_UNITS)}
    tests = "from p0 import f0\n" + "\n".join(
        f"def test_shared_integration_contract_{i}(): assert f0()" for i in range(300))
    # Every test touches all units: a valid worst-case shared integration gate.
    tests = tests.replace("assert f0()", "assert " + " and ".join(f"f{i}()" for i in range(v2.MAX_UNITS)))
    tests = "\n".join(f"from p{i} import f{i}" for i in range(1, v2.MAX_UNITS)) + "\n" + tests
    _plant(coalesce_on, {**files, "test_gate.py": tests})
    plan = v2.derive_with_index({"repo": "build", "msg": "Implement all functions", "test_src": tests,
                                 "public_gate_paths": ["test_gate.py"]}, coalesce_on)
    assert plan is not None and len(plan.units) == v2.MAX_UNITS
    assert all(len(unit.gate_tests) == 300 for unit in plan.units)
    bolt = Bolt(plan)
    bolt.root = coalesce_on
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: repair\n" + render_plan_block(plan)},
                {"role": "assistant", "content": "inspect next"}]
    context = {"journal_receipt_records": [], "ledger": {"rows": []}}
    runtime = SimpleNamespace(owner=bolt, binding=dict)
    before = json.dumps(messages)
    monkeypatch.setenv(COALESCE_ENV, "off")
    assert len(json.dumps(build_typed_state(messages, context, runtime)).encode()) < MAX_PRODUCT_BYTES
    monkeypatch.setenv(COALESCE_ENV, "on")
    frame = build_typed_state(messages, context, runtime)
    current = frame["unit_plan"]
    assert len(json.dumps(current).encode()) <= 16384
    assert current["plan_digest"] == plan.plan_digest and current["unit_index"] == 0
    assert current["units"][0]["unit_id"] == plan.units[0].unit_id
    assert len(bolt.unit_plan.units) == 150 and len(bolt.unit_plan.units[0].gate_tests) == 300
    assert json.dumps(messages) == before
    # Projection is regenerated from a restored host plan and reflects progress.
    bolt = pickle.loads(pickle.dumps(bolt))
    bolt.unit_index = 149
    bolt.unit_escalate = plan.units[-1].unit_id
    runtime.owner = bolt
    resumed = build_typed_state(messages, context, runtime)["unit_plan"]
    assert resumed["unit_index"] == 149 and resumed["escalated_unit"] == plan.units[-1].unit_id
    assert resumed["plan_digest"] == plan.plan_digest
    assert len(json.dumps(resumed).encode()) <= 16384


@pytest.mark.parametrize("workers", ["local", "named"])
def test_r5_p2_5_real_xdist_passes_are_gate_evidence(coalesce_on, monkeypatch, workers):
    import kernel_execution_policy

    bolt = Bolt(_small_plan(("a",)))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    (coalesce_on / "test_gate.py").write_text("def test_a(): pass\n")
    # Four nested workers need their own bounded startup allowance on busy CI.
    observe = _r3_runner(coalesce_on, bolt, monkeypatch, check_timeout=90)
    original = kernel_execution_policy.run_sandboxed
    commands = []

    def distributed(command, *args, **kwargs):
        # Load the installed plugin explicitly; the harness disables autoload.
        actual = list(command) + ["-p", "xdist.plugin", "-n", "4"]
        if workers == "named":
            actual = list(command) + ["-p", "xdist.plugin", "--dist=load"]
            for index in range(4):
                actual += ["--tx", f"popen//id=gate-worker-{index}"]
        commands.append(actual)
        return original(actual, *args, **kwargs)

    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", distributed)
    observation = observe("test_gate.py::test_a")
    assert bolt.last_test_exit == 0, observation
    prefix = r"gw\d+" if workers == "local" else r"gate-worker-\d+"
    assert re.search(r"\[" + prefix + r"\].*PASSED test_gate.py::test_a", observation.unit_gate_runs[0].stdout)
    _r3_gate(bolt, coalesce_on, observation)
    assert bolt.unit_index == 1 and bolt.unit_gate_passed == {bolt.unit_plan.units[0].unit_id}
    assert len(commands) == 2 and all("-x" in command for command in commands if "--collect-only" not in command)


def test_r5_p1_4_projection_bound_counts_json_escape_bytes(coalesce_on):
    from kernel_stateframe_framefit5 import build_typed_state

    # Non-BMP letters are valid Python identifiers, but ASCII JSON expands
    # each to a surrogate pair. Character bounds alone do not bound the wire.
    names = tuple(f"f{i}_" + "\U00010400" * 100 for i in range(v2.MAX_UNITS))
    plan = _small_plan(names)
    last = plan.units[-1]
    last = replace(last, members=names[:4], gate_tests=tuple(f"test_{name}" for name in names[:4]),
                   files=tuple(("\U00010400" * 40 + "/") * 4 + f"p{i}.py" for i in range(4)),
                   depends_on=tuple(unit.unit_id for unit in plan.units[:4]))
    plan = replace(plan, units=(*plan.units[:-1], last))
    plan = replace(plan, plan_digest=v2._digest(plan._basis()))
    bolt = Bolt(plan)
    bolt.root, bolt.unit_index = coalesce_on, len(plan.units) - 1
    bolt.unit_gate_passed = {unit.unit_id for unit in plan.units[:4]}
    bolt.unit_gate_rechecks = {unit.unit_id for unit in plan.units[4:8]}
    bolt.unit_escalate = last.unit_id
    frame = build_typed_state(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "Task: repair"}],
        {"journal_receipt_records": [], "ledger": {"rows": []}}, SimpleNamespace(owner=bolt, binding=dict))
    assert len(json.dumps(frame["unit_plan"]).encode()) <= 16384
    assert frame["unit_plan"]["plan_digest"] == plan.plan_digest


def test_r5_p1_2_journal_exclusion_is_host_bound(coalesce_on, monkeypatch):
    from unit_gate_evidence import collection_command, shared_source_identity

    _plant(coalesce_on, {
        ".stateframe/run/journal.jsonl": "initial\n",
        ".stateframe/settings.json": "true\n",
        "env/pyvenv.cfg": "home = /unused\n",
        "env/settings.json": "true\n",
        "test_gate.py": "def test_a(): pass\n",
    })
    runtime = SimpleNamespace(journal=SimpleNamespace(path=coalesce_on / ".stateframe/run/journal.jsonl"))
    plan = _small_plan(("a",))
    identity = shared_source_identity(plan, coalesce_on, runtime)
    command = [sys.executable, "-m", "pytest", "-x", "-q"]
    inventory_identity = collection_command(command, ("test_gate.py",), coalesce_on, plan, runtime=runtime)[1]
    runtime.journal.path.write_text("later host receipt\n")
    assert shared_source_identity(plan, coalesce_on, runtime) == identity
    assert collection_command(command, ("test_gate.py",), coalesce_on, plan, runtime=runtime)[1] == inventory_identity
    (coalesce_on / ".stateframe/settings.json").write_text("false\n")
    assert shared_source_identity(plan, coalesce_on, runtime) != identity
    identity = shared_source_identity(plan, coalesce_on, runtime)
    # A virtualenv-looking name/marker does not establish a host artifact root.
    (coalesce_on / "env/settings.json").write_text("false\n")
    assert shared_source_identity(plan, coalesce_on, runtime) != identity
    _plant(coalesce_on, {"host-env/pyvenv.cfg": "home = /host\n", "host-env/library.bin": "before"})
    (coalesce_on / ".venv").symlink_to(coalesce_on / "host-env", target_is_directory=True)
    monkeypatch.setattr(sys, "prefix", str(coalesce_on / ".venv"))
    identity = shared_source_identity(plan, coalesce_on, runtime)
    (coalesce_on / "host-env/library.bin").write_text("after")
    assert shared_source_identity(plan, coalesce_on, runtime) == identity


def _r5_legacy_identity(root, *, exclude=()):
    """Freeze pre-fix checkpoint authority without depending on the retired lineage."""
    import hashlib

    # Exact 3768a027 identity algorithm: old checkpoints omitted unowned data
    # and virtualenv-looking directories. Keep this independent of today's hash.
    _PYTEST_CONFIGS = frozenset({"pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini",
                                 "setup.cfg", "pytest.toml", ".pytest.toml"})
    MAX_FILE_BYTES = 16 * 1024 * 1024
    MAX_SOURCE_BYTES = 256 * 1024 * 1024
    excluded = set(exclude)
    sources: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if (path.suffix != ".py" and path.name not in _PYTEST_CONFIGS
                or rel.as_posix() in excluded
                or any(part.startswith(".") or part in {"venv", "env", "node_modules", "__pycache__"}
                       for part in rel.parts[:-1])):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("unit gate source escapes workspace")
        if not path.is_file():
            continue
        if len(sources) >= 20000:
            raise ValueError("unit gate source identity exceeds entry bound")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("unit gate source identity exceeds byte bound")
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        total_bytes += len(data)
        if len(data) > MAX_FILE_BYTES or total_bytes > MAX_SOURCE_BYTES:
            raise ValueError("unit gate source identity exceeds byte bound")
        sources.append((rel.as_posix(), hashlib.sha256(data).hexdigest()))
    return hashlib.sha256(json.dumps(sources, separators=(",", ":")).encode()).hexdigest()

@pytest.mark.parametrize("authority", ["node_cache", "completed_pass"])
def test_r5_p1_1_2_legacy_checkpoint_requalifies_cached_authority(coalesce_on, monkeypatch, authority):
    import io

    import kernel_checkpoint

    bolt = Bolt(_small_plan(("a", "b")))
    monkeypatch.setattr(v2, "build_index", lambda *args: None)
    _plant(coalesce_on, {"a.py": "x=0\n", "b.py": "x=0\n", "test_gate.py": (
        "import pytest\n@pytest.mark.parametrize('value',[1,2],ids=['one','two'])\n"
        "def test_a(value): assert value > 0\ndef test_b(): pass\n")})
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    first = observe("test_gate.py::test_a[one]")
    _r3_gate(bolt, coalesce_on, first)
    assert bolt.unit_index == 0
    if authority == "completed_pass":
        _r3_gate(bolt, coalesce_on, observe("test_gate.py::test_a[two]"))
        unit_id = bolt.unit_plan.units[0].unit_id
        product, _shared, inventory = bolt.unit_gate_pass_sources[unit_id]
        bolt.unit_gate_pass_sources[unit_id] = (
            product, _r5_legacy_identity(coalesce_on, exclude=bolt.unit_plan.index_paths), inventory)
    old_identity = json.dumps([_r5_legacy_identity(coalesce_on)])
    old_key = v2._digest([old_identity, [(path, (coalesce_on / path).read_text())
                                       for path in bolt.unit_plan.index_paths], sorted(first.unit_gate_runs[0].collected)])
    bolt.unit_gate_node_evidence = (old_key, bolt.unit_gate_node_evidence[1])
    # Earlier code could discard an incomplete red and checkpoint these positive
    # markers. An upgraded reader cannot infer the lost red from that checkpoint.
    context = {name: {} for name in ("framecells", "framefuncs", "cvars", "hostfns", "participants")}
    bolt = kernel_checkpoint._Unpickler(io.BytesIO(kernel_checkpoint.dumps(bolt, context)), context).load()
    observe = _r3_runner(coalesce_on, bolt, monkeypatch)
    selector = "test_gate.py::test_a[two]" if authority == "node_cache" else "test_gate.py::test_b"
    _r3_gate(bolt, coalesce_on, observe(selector))
    assert bolt.unit_index == 0
    assert bolt.unit_plan.units[0].unit_id not in getattr(bolt, "unit_gate_passed", set())
