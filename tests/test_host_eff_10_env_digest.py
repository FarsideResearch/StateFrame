"""Parent and worker must compute the same check environment digest, so PASS receipts reuse."""
from __future__ import annotations

from pathlib import Path

import pytest

import kernel_check_permits
import kernel_effects
import kernel_parallel
import kernel_unit_policy
from kernel_parallel_records import CheckReceipt, CheckRecipe, ParallelLimits, UnitContract, UnitSpec


@pytest.fixture(autouse=True)
def _permit(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_check_permits, "_installed", (tmp_path / "permits", 1))


def _spec() -> UnitSpec:
    recipe = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py::test_value",), "pytest")
    contract = UnitContract("a", "fix a", ("pkg/a.py",), (), (), (), (), (), recipe)
    return UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "tests/test_a.py"), (), "", 0, 0, "manifest", "policy",
                    ParallelLimits())


def _candidate(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/a.py").write_text("value = 1\n")
    (root / "tests/test_a.py").write_text("from pkg.a import value\n")
    return root


def _manifest_digests(candidate: Path, spec: UnitSpec) -> tuple[str, str]:
    # Same scan and digest the parent uses for reuse decisions.
    source = kernel_effects.scan(candidate, exclude=kernel_unit_policy._check_ignored)
    return (kernel_unit_policy._manifest_digest(source),
            kernel_unit_policy._manifest_digest(
                source, set(spec.contract.check.source_paths)))


def _worker_receipt(spec: UnitSpec, candidate: Path, **overrides) -> CheckReceipt:
    product, harness = _manifest_digests(candidate, spec)
    values = {
        "recipe_digest": spec.contract.check.digest(),
        "product_manifest_digest": product,
        "harness_manifest_digest": harness,
        "environment_digest": kernel_unit_policy._check_environment_digest(),
        "policy_digest": spec.policy_digest,
        "status": "PASS",
        "invalid_reason": None,
        "wrote_paths": (),
        "raw_evidence_ref": "evidence.txt",
        "projected_observation": "pytest exit 0",
    }
    values.update(overrides)
    return CheckReceipt(**values)


def test_parent_and_worker_environment_digests_match_in_one_process(monkeypatch):
    # Same process, same environment: parent helper must equal the shared worker function.
    assert kernel_parallel._parent_environment_digest() == \
        kernel_unit_policy._check_environment_digest()
    # Changing the inherited environment must move both sides together.
    monkeypatch.setenv("HOST_EFF_10_PROBE", "1")
    assert kernel_parallel._parent_environment_digest() == \
        kernel_unit_policy._check_environment_digest()


def test_matching_pass_worker_receipt_is_reused_without_fresh_check(tmp_path, monkeypatch):
    spec = _spec()
    candidate = _candidate(tmp_path / "candidate")
    receipt = _worker_receipt(spec, candidate)
    calls: list[object] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return receipt

    monkeypatch.setattr(kernel_parallel, "_check", spy)
    chosen = kernel_parallel._candidate_check(spec, candidate, tmp_path / "root", None, receipt)
    assert calls == []
    assert chosen is receipt


@pytest.mark.parametrize("overrides", [
    {"status": "FAIL"},
    {"status": "INVALID", "invalid_reason": "CHECK_WROTE_FILES"},
    {"environment_digest": "0" * 64},
    {"policy_digest": "other-policy"},
    {"recipe_digest": "0" * 64},
    {"product_manifest_digest": "0" * 64},
    {"harness_manifest_digest": "0" * 64},
])
def test_mismatched_worker_receipt_still_falls_back_to_fresh_check(
        tmp_path, monkeypatch, overrides):
    # Reject paths are unchanged: any mismatch still runs a fresh check.
    spec = _spec()
    candidate = _candidate(tmp_path / "candidate")
    receipt = _worker_receipt(spec, candidate, **overrides)
    calls: list[object] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return "fresh"

    monkeypatch.setattr(kernel_parallel, "_check", spy)
    chosen = kernel_parallel._candidate_check(spec, candidate, tmp_path / "root", None, receipt)
    assert len(calls) == 1
    assert chosen == "fresh"


def test_no_worker_receipt_runs_fresh_check(tmp_path, monkeypatch):
    spec = _spec()
    candidate = _candidate(tmp_path / "candidate")
    calls: list[object] = []
    monkeypatch.setattr(kernel_parallel, "_check",
                        lambda *a, **k: calls.append(a) or "fresh")
    assert kernel_parallel._candidate_check(spec, candidate, tmp_path / "root", None, None) == "fresh"
    assert len(calls) == 1
