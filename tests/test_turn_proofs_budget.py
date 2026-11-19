"""Budgets and parity for early completion guards and delivery proofs."""
from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import kernel_completion_owner as completion
import kernel_green_lock as green_lock
import kernel_ship_tree_smoke as smoke
import pytest


@pytest.fixture
def large_tree(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    paths = tuple(f"mod_{index:04}.py" for index in range(5_000))
    for path in paths:
        (root / path).write_bytes(b"VALUE = 1\n")
    return root, paths


def test_clean_guard_defers_delivery_only_proofs(large_tree, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", "on")
    root, paths = large_tree
    parent = green_lock._workspace_product_file_digests(
        SimpleNamespace(owner=SimpleNamespace(root=root)), paths
    )
    assert parent is not None and len(parent) == 5_000
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root, current_step=0),
        green_lock=SimpleNamespace(
            _starting_editable_product_files=parent,
            _starting_editable_product_paths=paths,
        ),
        binding=lambda: {"product_revision": "parent"},
        test_receipts=[],
        max_steps=201,
    )
    counts: Counter[str] = Counter()
    full_proof = green_lock._workspace_product_file_digests
    read_bytes = Path.read_bytes

    def count_proof(*args, **kwargs):
        counts["full_proofs"] += 1
        return full_proof(*args, **kwargs)

    def count_read(path):
        if path.parent == root:
            counts["source_reads"] += 1
        return read_bytes(path)

    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", count_proof)
    monkeypatch.setattr(Path, "read_bytes", count_read)

    durations = []
    for step in range(200):
        runtime.owner.current_step = step
        before = perf_counter()
        assert not completion._product_parent_unchanged(runtime, reason="completion_route")
        durations.append(perf_counter() - before)
        assert counts == Counter(), f"turn {step + 1}: {dict(counts)} in {durations[-1]:.6f}s"

    print(
        f"proof guard: {len(durations)} turns, {dict(counts)}, "
        f"first={durations[0]:.6f}s max={max(durations):.6f}s sum={sum(durations):.6f}s"
    )
    assert max(durations) < 0.25


def test_green_parent_route_parity(tmp_path, monkeypatch):
    product = tmp_path / "product.py"
    product.write_text("VALUE = 1\n")
    events = []
    binding = {"product_revision": "current"}
    receipt = SimpleNamespace(binding=binding, after_binding=binding, exit_code=0,
                              binding_stable=True)
    verdict = SimpleNamespace(state="SATISFIED", current_binding=binding,
                              trusted_receipts=(receipt,))
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, current_step=1),
        green_lock=SimpleNamespace(_starting_editable_product_paths=("product.py",)),
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
        public_test_verdict=lambda: verdict,
        binding=lambda: binding,
        test_receipts=[],
        max_steps=4,
    )
    runtime.green_lock._starting_editable_product_files = (
        green_lock._workspace_product_file_digests(runtime, ("product.py",)))

    for source, expected in (("VALUE = 1\n", True), ("VALUE = 2\n", False)):
        product.write_text(source)
        results = []
        for switch in ("off", "on"):
            monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", switch)
            events.clear()
            runtime._postgreen_unchanged_parent_suppression_revisions = set()
            unchanged = completion._product_parent_unchanged(runtime, reason="completion_route")
            results.append((unchanged, completion.postgreen_done_offer(runtime), events.copy()))
        assert results[0] == results[1]
        assert results[0][:2] == (expected, (not expected, "current"))


def test_smoke_resolver_bounds_probes_and_rechecks_missing_providers(tmp_path, monkeypatch):
    checked = []
    for name in ("application", "sibling"):
        (tmp_path / name).mkdir()
    (tmp_path / "application" / "local_module.py").touch()
    for index in range(100):
        relative = f"application/module_{index}.py"
        (tmp_path / relative).write_text("import local_module\nimport missing_dependency\n")
        checked.append(relative)
    counts: Counter[str] = Counter()
    provides = smoke._provides

    def probe(directory, name):
        counts["provider_probes"] += 1
        return provides(directory, name)

    monkeypatch.setattr(smoke, "_provides", probe)
    monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", "off")
    before = perf_counter()
    legacy = smoke.source_roots(tmp_path, checked)
    legacy_seconds = perf_counter() - before
    legacy_probes = counts["provider_probes"]
    # A new provider has to be visible even if its name missed previously.
    (tmp_path / "sibling" / "missing_dependency.py").touch()
    changed_legacy = smoke.source_roots(tmp_path, checked)
    (tmp_path / "sibling" / "missing_dependency.py").unlink()
    monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", "on")
    counts.clear()
    before = perf_counter()
    assert smoke.source_roots(tmp_path, checked) == legacy
    print(f"smoke resolver: OFF={legacy_seconds:.6f}s ({legacy_probes} probes), "
          f"ON={perf_counter() - before:.6f}s ({counts['provider_probes']} probes)")
    assert counts == {"provider_probes": 5}
    (tmp_path / "sibling" / "missing_dependency.py").touch()
    assert smoke.source_roots(tmp_path, checked) == changed_legacy
    assert changed_legacy == (*legacy, str(tmp_path / "sibling"))


@pytest.mark.parametrize("switch, repeated", [("off", 2), ("on", 1)])
def test_delivery_proves_sealed_input_with_current_harness(tmp_path, monkeypatch, switch, repeated):
    proof = pytest.importorskip("kernel_turn_proofs")
    from kernel_turn_contracts import (
        Basis,
        Context,
        Corrupt,
        Dep,
        Head,
        Key,
        Ref,
        Snapshot,
        Stale,
    )
    from kernel_verdict import SATISFIED, evaluate_required_tests

    monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", switch)
    blobs = {}

    def stage(codec, data):
        ref = Ref(codec, sha256(data).hexdigest(), len(data))
        blobs[ref] = data
        return ref

    def load(ref):
        data = blobs.get(ref)
        if data is None or len(data) != ref.size or sha256(data).hexdigest() != ref.sha256:
            raise Corrupt("missing or corrupt stored bytes")
        return data

    publication = SimpleNamespace(stage=stage, load=load, transaction=nullcontext,
                                  head=lambda: None)
    workspace = SimpleNamespace()
    ctx = Context("principal", "run", "attempt", tmp_path, "profile")
    at = Basis("parent", "live", 1, Head(201, "journal"), 200, "authority")
    scope = stage("turn/scope/1", b"{}")
    harness = tmp_path / "test_product.py"
    harness.write_text("assert True\n")
    key = Key("workspace-path", b"test_product.py")
    counts: Counter[str] = Counter()
    pending = ["verify"]

    def revision():
        return sha256(harness.read_bytes()).hexdigest()

    def current(basis, deps):
        assert basis == at
        counts["freshness_checks"] += 1
        return deps == (Dep(key, revision()),)

    @contextmanager
    def pin(ref):
        assert ref == scope
        counts["pins"] += 1
        yield Snapshot(at.live, scope, at.epoch)

    workspace.pin = pin
    port = proof.open_port(ctx, workspace=workspace, facts=SimpleNamespace(current=current),
                           publication=publication)
    original = {"product_revision": "live", "test_harness_revision": revision(),
                "environment_revision": "interpreter"}
    receipt = {"kind": "test_subprocess_completed", "producer": "test_subprocess",
               "receipt_id": "original-check", "exit_code": 0, "selectors": ["test_product.py"],
               "binding": original, "after_binding": original, "binding_stable": True}

    def legacy(candidate, guard):
        assert candidate in (b"sealed candidate", b"another sealed candidate")
        assert guard["blockers"] == pending
        counts["legacy_verifier"] += 1
        binding = {**original, "test_harness_revision": revision()}
        verdict = evaluate_required_tests(binding, ["test_product.py"], [receipt])
        return {"accepted": verdict.state == SATISFIED, "reasons": list(verdict.reason_codes)}

    proof.bind_legacy_verifier(port, legacy)

    def guard(use):
        evidence = {name: stage(f"turn/{name}/1", b"{}")
                    for name in ("selector_coverage", "public_harness", "menu", "obligations")}
        inputs = proof.stage_guard_inputs(
            ctx, publication, at, use, scope=scope, deps=(Dep(key, revision()),),
            evidence=evidence, allowed_actions=("read",), blockers=tuple(pending),
        )
        return port.guard(at, use, inputs)

    menu_guard = guard("menu")
    assert json.loads(load(menu_guard.value))["blockers"] == ["verify"]
    # The host completed automatic verification before freezing the next send.
    pending.clear()
    send_guard = guard("send")
    assert json.loads(load(send_guard.value))["blockers"] == []
    assert counts == {"freshness_checks": 2, "pins": 2}

    old_guard = guard("act")
    sealed = stage("turn/candidate/1", b"sealed candidate")
    candidate = proof.stage_candidate(publication, at, old_guard, sealed)
    accepted = port.delivery(at, candidate, old_guard)
    assert json.loads(load(accepted.value)) == {
        "accepted": True, "reasons": ["ALL_REQUIRED_TESTS_CURRENT"],
    }
    assert port.delivery(at, candidate, old_guard) == accepted
    assert counts["legacy_verifier"] == repeated
    other_sealed = stage("turn/candidate/1", b"another sealed candidate")
    other = proof.stage_candidate(publication, at, old_guard, other_sealed)
    assert port.delivery(at, other, old_guard) == accepted
    blobs[other_sealed] = b"corrupt"
    with pytest.raises(Corrupt, match="corrupt stored bytes"):
        port.delivery(at, other, old_guard)
    assert counts["legacy_verifier"] == repeated + 1

    harness.write_text("assert False\n")
    with pytest.raises(Stale, match="dependencies changed"):
        port.delivery(at, candidate, old_guard)
    assert counts["legacy_verifier"] == repeated + 1

    new_guard = guard("act")
    refreshed = proof.stage_candidate(publication, at, new_guard, sealed)
    refused = port.delivery(at, refreshed, new_guard)
    legacy_refusal = legacy(b"sealed candidate", {"blockers": []})
    assert not legacy_refusal["accepted"] and legacy_refusal["reasons"]
    assert json.loads(load(refused.value)) == legacy_refusal

    # A new proof port has no cached verifier results after resume.
    resumed = proof.open_port(ctx, workspace=workspace, facts=SimpleNamespace(current=current),
                              publication=publication)
    proof.bind_legacy_verifier(resumed, legacy)
    prior_calls = counts["legacy_verifier"]
    assert resumed.delivery(at, refreshed, new_guard) == refused
    assert counts["legacy_verifier"] == prior_calls + 1
