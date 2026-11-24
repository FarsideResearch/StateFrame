"""Work and parity at the public verdict and completion evidence boundaries."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import pickle
from time import perf_counter
from types import SimpleNamespace

import kernel_completion_owner as completion
import kernel_host_cache as host_cache
import kernel_release_switch_registry as switch_registry
import kernel_test_artifact_policy as artifact_policy
import kernel_verdict as verdict
import pytest
from kernel_test_integrity import TestIntegrityGuard
from kernel_verdict_identity import RunIdentityProvider
from kernel_host_done_facts import workspace_snapshot
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_state import WorkspaceFence, WorkspaceState


pytestmark = pytest.mark.usefixtures("isolated_pilot_context")

TURN_SWITCHES = (
    "ATOMIC_PUBLICATION", "BOUNDED_QUALIFICATION", "DELIVERY_PROOFS",
    "GENERATION_OVERLAP", "INDEXED_READS", "MINIMAL_LAUNCH", "PANELS",
    "REPO_INTEL", "TIMING",
)
SELECTOR = "test_product.py"
RECEIPT_COUNT = 200


def _switches(monkeypatch, setting):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", setting)
    monkeypatch.setenv("STATEFRAME_PUBLIC_TEST_VERDICT_FACT", "on")
    monkeypatch.setenv("STATEFRAME_HOST_DONE_FACTS", "on")
    for name in TURN_SWITCHES:
        monkeypatch.setenv(f"STATEFRAME_TURN_{name}", setting)


def _runtime(tmp_path, monkeypatch, file_count=1):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "file_0000.py"
    target.write_text("VALUE = 0\n")
    harness = root / SELECTOR
    harness.write_text("def test_product(): assert True\n")

    monkeypatch.setattr(switch_registry, "enforce_stateframe_switch_registry",
                        lambda *_args, **_kwargs: {})
    owner = SimpleNamespace(root=root, task_id="completion-test", current_step=1,
                            task_row={}, f2p_ids=[SELECTOR],
                            authorize=lambda *_args, **_kwargs: None)
    runtime = StateFrameRuntime(
        owner, output=tmp_path / "journal", target=target, model="scripted",
        tools=[], temperature=0, source_resolver=lambda _path, _symbol: None,
    )

    # Size the measured consumers independently of runtime bootstrap.
    if file_count > 1:
        for index in range(1, file_count):
            (root / f"file_{index:04}.py").write_text(f"VALUE = {index}\n")
        runtime._host_done_facts_baseline = workspace_snapshot(root)

    def binding(**_kwargs):
        return {
            "product_revision": "product", "artifact": "product",
            "test_harness_revision": harness.read_text(),
            "environment_revision": "environment", "environment": "environment",
        }

    runtime.binding = binding
    runtime.required_public_selectors = lambda: [SELECTOR]
    original = binding()
    runtime.test_receipts.extend({
        "kind": "test_subprocess_completed", "producer": "test_subprocess",
        "receipt_id": f"check-{index}", "exit_code": 0, "selectors": [SELECTOR],
        "binding": original, "after_binding": original, "binding_stable": True,
        "sequence": index + 1, "status": "PASS",
    } for index in range(RECEIPT_COUNT))
    return runtime


def _pin_workspace(tmp_path, root):
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"),
                           feed=WorkspaceFeed(root, enabled=True))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    with state.pin("fixture") as observation:
        assert observation.validate_for_use("fixture")
    return consumers


def test_repeated_completion_consumers_bound_evidence_work(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _runtime(tmp_path, monkeypatch, file_count=5000)
    root = runtime.owner.root
    consumers = _pin_workspace(tmp_path, root)
    counts = Counter()
    normalize = artifact_policy.normalize_public_test_bindings_for_mode
    canonical = verdict._canonical
    entry_digest = host_cache._entry_digest
    read_bytes = Path.read_bytes
    audit = runtime._audit_test_integrity
    observe = consumers.state._observe

    def count_normalize(*args):
        counts["normalizations"] += 1
        return normalize(*args)

    def count_canonical(value):
        if isinstance(value, dict) and "required_selectors" in value and "test_receipts" in value:
            counts["completion_encodes"] += 1
        return canonical(value)

    def count_digest(value):
        if (isinstance(value, tuple) and len(value) == 4
                and isinstance(value[2], (list, tuple)) and len(value[2]) == RECEIPT_COUNT):
            counts["history_hashes"] += 1
        return entry_digest(value)

    def count_read(path):
        if path.is_relative_to(root):
            counts["workspace_byte_reads"] += 1
        return read_bytes(path)

    def count_observe(*args, **kwargs):
        tree = observe(*args, **kwargs)
        if consumers.state.work.content_reads:
            counts["workspace_byte_reads"] += consumers.state.work.content_reads
        return tree

    def count_audit():
        counts["integrity_audits"] += 1
        return audit()

    event_sequence = 0

    def count_event(kind, **_fields):
        nonlocal event_sequence
        event_sequence += 1
        if kind == "public_test_verdict_fact_normalized":
            counts["verdict_events"] += 1
        return {"kind": kind, "sequence": event_sequence}

    monkeypatch.setattr(artifact_policy, "normalize_public_test_bindings_for_mode", count_normalize)
    monkeypatch.setattr(verdict, "_canonical", count_canonical)
    monkeypatch.setattr(host_cache, "_entry_digest", count_digest)
    monkeypatch.setattr(Path, "read_bytes", count_read)
    monkeypatch.setattr(runtime, "_audit_test_integrity", count_audit)
    monkeypatch.setattr(consumers.state, "_observe", count_observe)
    monkeypatch.setattr(runtime.journal, "emit", count_event)

    # Isolate unrelated gate producers; the live route and its verdict checks
    # still run through the same check used by physical-send qualification.
    runtime.pass_gates_vector = lambda _tools: SimpleNamespace(ready=True)
    started = perf_counter()
    bases = []
    cycle_times = []
    for _ in range(6):
        cycle_started = perf_counter()
        bases.append(runtime._completion_basis())
        assert completion._green_current(runtime)
        assert completion._completion_route_surface_error_uncached(runtime, ()) is None
        assert runtime.compute_done_facts().empty_diff
        cycle_times.append(perf_counter() - cycle_started)
    elapsed = perf_counter() - started

    assert consumers.state.fence.root == root
    assert all(basis == bases[0] for basis in bases)
    audits = counts["integrity_audits"]
    assert counts["verdict_events"] == 1 and audits >= 18
    print(f"completion consumers: total={elapsed:.6f}s max_cycle={max(cycle_times):.6f}s {dict(counts)}")
    assert dict(counts) == {
        "normalizations": 1, "completion_encodes": 1,
        "integrity_audits": audits, "verdict_events": 1,
    }
    assert max(cycle_times) < 10.0


@pytest.mark.parametrize("setting", ["off", "on"])
def test_amendment_defensive_copy_and_restart(tmp_path, monkeypatch, setting):
    _switches(monkeypatch, setting)
    runtime = _runtime(tmp_path, monkeypatch)
    consumers = _pin_workspace(tmp_path, runtime.owner.root)
    first = runtime._completion_basis()
    caller_verdict = runtime.public_test_verdict()
    assert caller_verdict.state == verdict.SATISFIED
    caller_verdict.reason_codes.append("CALLER_ONLY")
    caller_verdict.binding["environment"] = "caller-only"
    caller_verdict.trusted_receipts[0].binding["artifact"] = "caller-only"
    assert runtime._completion_basis() == first
    assert completion._green_current(runtime)

    receipt = runtime.test_receipts[-1]
    receipt["exit_code"] = 1
    assert runtime.public_test_verdict().state == verdict.VIOLATED
    amended = runtime._completion_basis()
    assert amended.evidence_digest != first.evidence_digest

    runtime.test_receipts = pickle.loads(pickle.dumps(runtime.test_receipts))
    assert runtime._completion_basis() == amended
    assert runtime.public_test_verdict().state == verdict.VIOLATED
    runtime.test_receipts[-1]["binding"]["test_harness_revision"] = "amended"
    assert runtime.public_test_verdict().state == verdict.UNRESOLVED
    assert consumers.state.fence.root == runtime.owner.root


def test_checkpoint_rebuilds_receipts_without_saving_attempt_local_projections(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    rows = [{"receipt_id": "check"}]
    runtime.test_receipts = host_cache.completion_receipts(rows)
    runtime._completion_binding_generation = lambda: None
    runtime._live_decision_panels = lambda: None

    state = runtime.__getstate__()
    assert type(state["test_receipts"]) is list
    restored = pickle.loads(pickle.dumps(runtime))
    assert host_cache.completion_receipt_state(restored.test_receipts).value == rows
    assert not hasattr(restored, "_completion_binding_generation")
    assert not hasattr(restored, "_live_decision_panels")


def test_checkpoint_rebuilds_coverage_projection_without_losing_initial_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "pricing.py"
    target.write_text("VALUE = 1\n")
    consumers = _pin_workspace(tmp_path, root)
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    initial = runtime._coverage_start_snapshot(root)
    assert "pricing.py" in initial
    assert runtime._coverage_workspace_projection.state is consumers.state

    restored = pickle.loads(pickle.dumps(runtime))
    assert not hasattr(restored, "_coverage_workspace_projection")
    with consumers.state.begin_write(("pricing.py",)):
        target.write_text("VALUE = 2\n")
    current = restored._project_coverage(root, lambda projection, _: projection.initial_snapshot())
    assert current["pricing.py"] != initial["pricing.py"]
    assert restored._coverage_start_snapshot(root) == initial


@pytest.mark.parametrize("setting", ["off", "on"])
def test_changed_harness_still_blocks_fresh_pass(tmp_path, monkeypatch, setting):
    _switches(monkeypatch, setting)
    runtime = _runtime(tmp_path, monkeypatch)
    root = runtime.owner.root
    (root / SELECTOR).write_text("def test_product(): assert 1 == 1\n")
    runtime.test_integrity = TestIntegrityGuard(root, enabled_override=True)
    consumers = _pin_workspace(tmp_path, root)
    for receipt in runtime.test_receipts:
        receipt["binding"] = receipt["after_binding"] = runtime.binding()
    assert runtime.public_test_verdict().state == verdict.SATISFIED
    old = runtime._completion_basis()
    (root / SELECTOR).write_text("def test_product(): pass\n")
    for receipt in runtime.test_receipts:
        receipt["binding"] = receipt["after_binding"] = runtime.binding()
    after = runtime.public_test_verdict()
    assert after.state == verdict.UNRESOLVED
    assert "TEST_WEAKENED" in after.reason_codes
    assert runtime._completion_basis().evidence_digest != old.evidence_digest
    assert consumers.state.fence.root == root


@pytest.mark.parametrize("provider_kind", ["pinned", "versioned", "live"])
def test_provider_rechecks_identity_and_workspace_projections(tmp_path, monkeypatch, provider_kind):
    _switches(monkeypatch, "on")
    monkeypatch.setenv("STATEFRAME_VERDICT_IDENTITY", "on")
    runtime = _runtime(tmp_path, monkeypatch)
    root = runtime.owner.root
    (root / "file_0000.py").chmod(0o751)
    (root / "linked.py").symlink_to("file_0000.py")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "ignored").write_text("cache\n")
    excluded = root / "external.log"
    excluded.write_text("excluded\n")
    consumers = _pin_workspace(tmp_path, root)
    real_provider = RunIdentityProvider(
        root, "run", "release", "harness", excluded_paths=(excluded,))
    original = real_provider.current()
    for row in runtime.test_receipts:
        row.update(original.to_dict())
    reads = []
    normalize = artifact_policy.normalize_public_test_bindings_for_mode
    calls = []

    class LiveProvider(RunIdentityProvider):
        def current(self):
            reads.append(provider_kind)
            return original if not changed else real_provider.current()

    class VersionedProvider(LiveProvider):
        @property
        def generation(self):
            return generation[0]

    generation = [0]
    changed = []
    runtime.verdict_identity_provider = {
        "pinned": real_provider,
        "versioned": VersionedProvider(root, "run", "release", "harness"),
        "live": LiveProvider(root, "run", "release", "harness"),
    }[provider_kind]

    def count_normalize(*args):
        calls.append(provider_kind)
        return normalize(*args)

    monkeypatch.setattr(artifact_policy, "normalize_public_test_bindings_for_mode", count_normalize)
    for _ in range(3):
        current = runtime.public_test_verdict()
        assert current.state == verdict.SATISFIED
        assert current.identity == original.to_dict()
    if provider_kind != "pinned":
        assert len(reads) >= 3

    generation[0] += 1
    assert runtime.public_test_verdict().state == verdict.SATISFIED
    expected_normalizations = {"pinned": 1, "versioned": 2, "live": 4}
    assert len(calls) == expected_normalizations[provider_kind]

    (root / "file_0000.py").write_text("VALUE = 5\n")
    changed.append(True)
    # Even a provider whose generation was not advanced is sampled on every call.
    current = runtime.public_test_verdict()
    assert current.state == verdict.UNRESOLVED
    assert current.identity == real_provider.current().to_dict()
    assert consumers.state.fence.root == root


def test_pinned_done_facts_matches_legacy_after_workspace_changes(tmp_path, monkeypatch):
    from kernel_host_done_facts import compute_done_facts

    _switches(monkeypatch, "on")
    runtime = _runtime(tmp_path, monkeypatch)
    root = runtime.owner.root
    consumers = _pin_workspace(tmp_path, root)
    runtime.binding = dict

    def both():
        baseline = runtime._host_done_facts_baseline
        legacy = compute_done_facts(
            root, baseline, test_receipts=runtime.test_receipts,
            required_public_tests=runtime.required_public_selectors(),
        )
        pinned = runtime.compute_done_facts()
        assert pinned.to_dict() == legacy.to_dict()
        return pinned

    assert both().empty_diff
    (root / "file_0000.py").write_text("new line\nsecond line\n")
    (root / SELECTOR).unlink()
    (root / "link.py").symlink_to("file_0000.py")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "ignored").write_text("cache\n")
    assert both().changed_files == ("file_0000.py", SELECTOR)
    assert consumers.state.fence.root == root
