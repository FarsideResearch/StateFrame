from __future__ import annotations

from types import SimpleNamespace

import kernel_catalog_assume
import kernel_measurement
from kernel_catalog_assume import CatalogVerificationFact, CatalogVerificationReport
from kernel_catalog_contracts import CoverageRef, CoverageStatus
from kernel_stateframe_runtime import StateFrameRuntime


def _runtime(root, assumptions, binding):
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(
        root=root,
        task_id="task-catalog-assumptions",
        task_row={"assumptions": assumptions},
    )
    runtime.binding = binding
    return runtime


def test_empty_assumptions_do_not_encode_workspace_binding_50_times(tmp_path, monkeypatch):
    binding = {"product_files": {f"module_{i}.py": f"sha256:{i}" for i in range(100)}}
    binding_calls = []

    def get_binding():
        binding_calls.append(True)
        return binding

    runtime = _runtime(tmp_path, None, get_binding)
    original = kernel_measurement.canonical_bytes
    encoded_binding = []

    def count_binding_encodes(value):
        if isinstance(value, dict) and value.get("binding") is binding:
            encoded_binding.append(value)
        return original(value)

    monkeypatch.setattr(kernel_measurement, "canonical_bytes", count_binding_encodes)
    for _ in range(50):
        assert runtime._catalog_assumption_receipts() == ()
    assert encoded_binding == []
    assert len(binding_calls) == 50


def test_assumptions_are_rechecked_after_a_clear_and_after_a_workspace_change(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    marker.write_text("before", encoding="utf-8")
    assumptions = [{"kind": "symbol_exists", "target": "probe"}]

    def workspace_binding():
        text = marker.read_text(encoding="utf-8")
        return {
            "product_revision": kernel_measurement.digest({"marker.txt": text}),
            "test_harness_revision": kernel_measurement.digest({}),
            "environment_revision": "env",
            "product_files": {"marker.txt": text},
            "harness_files": {},
        }

    runtime = _runtime(tmp_path, assumptions, workspace_binding)
    observed = []

    def verify(root, raw_assumptions):
        text = (root / "marker.txt").read_text(encoding="utf-8")
        observed.append((text, raw_assumptions))
        coverage = CoverageRef(f"coverage:{text}", "assumption", CoverageStatus.UNKNOWN)
        fact = CatalogVerificationFact(
            "UNKNOWN", "symbol_exists", "probe", f"workspace contains {text}", coverage,
        )
        return CatalogVerificationReport((fact,))

    monkeypatch.setattr(kernel_catalog_assume, "verify_indexed_assumptions", verify)
    first = runtime._catalog_assumption_receipts()
    runtime.owner.task_row["assumptions"] = []
    assert runtime._catalog_assumption_receipts() == ()
    runtime.owner.task_row["assumptions"] = assumptions
    after_clear = runtime._catalog_assumption_receipts()
    assert after_clear == first

    marker.write_text("after", encoding="utf-8")
    after_mutation = runtime._catalog_assumption_receipts()
    assert after_mutation != first
    assert [item[0] for item in observed] == ["before", "before", "after"]
