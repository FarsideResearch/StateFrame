"""Excluded carrier modules must remain literal copies of the item's base."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ca = pytest.importorskip("kernel_claim_assessment")
from tools.pinned_tree import pinned_tree


ROOT = Path(__file__).resolve().parents[1]
BASE = "56af0dc6d6"


@pytest.mark.parametrize("setting", ["off", "on"])
@pytest.mark.parametrize("case", ["dotfile", "test_output", "clipped_read"])
def test_decision_slot_host_receipt_behavior_matches_base(tmp_path, tmp_path_factory, setting, case):
    def scenario(tree):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests/decision_claim_parity_scenario.py"),
             str(tree), str(tmp_path / "decision-parity"), setting],
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    expected = scenario(pinned_tree(BASE, tmp_path_factory))
    actual = scenario(ROOT)
    before, withdrawn, restored = expected[case]
    assert not before["withdrawn"]
    assert bool(withdrawn["withdrawn"]) is (setting == "on")
    assert not restored["withdrawn"]
    assert actual[case] == expected[case], case
# Existing non-passive carriers preserve the target stack bytes.
# The framefit, repair and measurement modules have allocator/read-receipt hooks;
# their OFF behavior is covered by the allocator and read-receipt parity suites.
UNCHANGED_CARRIERS = (
    "kernel_invalidation.py",
    "kernel_intent_carry.py",
    "kernel_qgates.py",
    "kernel_reasoning_ledger.py",
    "kernel_recall.py",
    "kernel_stateframe_resume.py",
    "kernel_timeline.py",
    "kernel_wire_string_hygiene.py",
    "probes/history_compaction.py",
)


@pytest.mark.parametrize("setting", ["off", "on"])
@pytest.mark.parametrize("module", UNCHANGED_CARRIERS)
def test_non_passive_store_carrier_module_matches_base_bytes(monkeypatch, module, setting):
    monkeypatch.setenv(ca.MODE_ENV, setting)
    path = "src/stateframe/engine/" + module
    expected = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    assert (ROOT / path).read_bytes() == expected, path


def test_removed_carrier_infrastructure_is_absent():
    engine = ROOT / "src/stateframe/engine"
    assert not (engine / "kernel_passive_boundary.py").exists()
    assert not (engine / "kernel_passive_provenance.py").exists()
    for path in engine.rglob("*.py"):
        text = path.read_text()
        assert "PASSIVE_DERIVED_CARRIERS" not in text, path
        assert "kernel_passive_provenance" not in text, path
        assert "kernel_passive_boundary" not in text, path


def test_runtime_changes_are_limited_to_allocator_request_and_suspension_plumbing():
    path = "src/stateframe/engine/kernel_stateframe_runtime.py"
    before = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)

    def methods(source):
        tree = ast.parse(source)
        runtime = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "StateFrameRuntime")
        return {node.name: ast.dump(node, include_attributes=False) for node in runtime.body
                if isinstance(node, ast.FunctionDef)}

    expected, actual = methods(before), methods((ROOT / path).read_bytes())
    changed = {name for name in expected.keys() | actual.keys()
               if expected.get(name) != actual.get(name)}
    # 0.6.36 already carries the passive-store runtime work; this carry adds only
    # sealed-request admission and resumable allocator suspension handling.
    assert changed == {"prepare", "finish"}
