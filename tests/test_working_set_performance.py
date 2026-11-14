"""Deterministic randomized checks and wall limits for the recorded 98-slot layout."""
import itertools
import json
import random
from dataclasses import replace
from time import perf_counter

import pytest
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_cache import _ACTIVE, independent, transaction
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_render import domain, render_assignment
from test_working_set_prepare import snapshot, source
from working_set_benchmark import admission, admit_request, recorded_snapshot

def random_cases():
    rng = random.Random(50834)
    cases = []
    for index in range(32):
        sources = []
        for ordinal in range(1, rng.randrange(1, 7)):
            slot = source(str(ordinal), rng.choice(['x\n', 'quote " \\ λ\n', 'long ' * 150 + '\n']),
                          state=rng.choice(["DUE", "RESIDENT", "CLOSURE", "RETIRING"]),
                          ordinal=ordinal)
            if sources and rng.randrange(4) == 0:
                slot = replace(sources[0], key=slot.key, obligation_ids=slot.obligation_ids,
                               admission_ordinal=ordinal)
            slot = replace(slot, plan_only=rng.choice([True, False]),
                           previous_mode=rng.choice([None, None, "F", "O"]))
            sources.append(slot)
        snap = snapshot(*sources, layouts=("plain", "final_cache"),
                        reasoning_mode="off" if index % 7 == 0 else "on")
        body = json.loads(snap.fixed_body)
        if index % 3 == 0:
            from test_final_cache_layout import _body
            body = _body(exchange=True)
        if sources and index % 4 == 0:
            # Exact fixed sharing plus an advisory suffix must preserve indexes.
            body["messages"].append({"role": "tool", "content": sources[0].page + "\nadvisory"})
        snap = replace(snap, fixed_body=json.dumps(body).encode(),
                       layouts=(("plain", "final_cache") if index % 3 == 0 and index % 4 != 0
                                else ("plain",)),
                       current_revisions=tuple(sorted({s.identity.path: s.identity.revision
                                                      for s in sources}.items())))
        baseline = {s.key: domain(snap, s)[-1] for s in snap.sources}
        cost = len(render_assignment(snap, baseline).wire)
        snap = replace(snap, byte_cap=max(1, cost + rng.choice([-1, 0, 50, 1500])),
                       prompt_limit=100 if index % 5 == 0 else 33_344)
        assignments = tuple(dict(zip((s.key for s in sources), modes))
                            for modes in itertools.product(*(domain(snap, s) for s in sources)))
        cases.append((snap, assignments))
    return cases


def test_randomized_assignments_are_deterministic_and_certified():
    cases = [(recorded_snapshot(), ()), *random_cases()]
    for snap, assignments in cases:
        actual = prepare_candidate(snap)
        assert actual == prepare_candidate(snap)
        assert tuple(render_assignment(snap, assignment, layout)
                     for assignment in assignments for layout in snap.layouts) == tuple(
            render_assignment(snap, assignment, layout)
            for assignment in assignments for layout in snap.layouts)
        if isinstance(actual, PreparedCandidate):
            assert verify_certificate(actual.wire, actual.rendered.certificate)


def test_recorded_98_slot_wall_limits():
    snap = recorded_snapshot()
    assert len(snap.sources) == 98
    started = perf_counter()
    result = prepare_candidate(snap)
    preparation_seconds = perf_counter() - started
    assert isinstance(result, PreparedCandidate)
    assert preparation_seconds < 10
    started = perf_counter()
    delivered, events = admit_request(snap)
    admission_seconds = perf_counter() - started
    assert admission_seconds < 4
    assert delivered.result_class == "OK" and events[-1][1]["decision"] == "FULL"
    print(f"98 slots: prepare={preparation_seconds:.3f}s admission={admission_seconds:.3f}s")


def test_proof_recomputes_without_warm_proposal_cache(monkeypatch):
    import kernel_working_set_proof as proof
    from kernel_working_set_cache import trial_render
    snap = snapshot(source("a", state="DUE"))
    modes = {"a": FULL}

    @transaction
    def exercise():
        cached = trial_render(snap, modes, "plain", render_assignment)
        assert _ACTIVE.get() is not None
        original = proof.render_assignment
        calls = []

        def changed(*args):
            assert _ACTIVE.get() is None
            calls.append(True)
            rebuilt = original(*args)
            return replace(rebuilt, wire=rebuilt.wire + b" ")

        monkeypatch.setattr(proof, "render_assignment", changed)
        with pytest.raises(CertificateError, match="recomputation"):
            verify_certificate(cached.wire, cached.certificate)
        assert calls == [True]
        assert _ACTIVE.get() is not None

    exercise()
    assert _ACTIVE.get() is None


def test_transaction_cleanup_and_nested_proof_on_failure():
    @transaction
    def fail():
        current = _ACTIVE.get()
        assert current is not None
        assert transaction(lambda: _ACTIVE.get())() is current
        assert independent(lambda: _ACTIVE.get())() is None
        assert _ACTIVE.get() is current
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        fail()
    assert _ACTIVE.get() is None


def test_fragment_serializer_preserves_json_numeric_types_and_nested_rows(monkeypatch):
    import kernel_working_set_cache as cache
    monkeypatch.setattr(cache, "MAX_MESSAGE_BYTES", 200)
    snap = snapshot()

    @transaction
    def exercise():
        for value in (True, 1, -0.0, 0.0, 1.0, None, '" λ \\ \n', "x" * 1000):
            body = {"model": "local", "messages": [{"role": "user", "content": "text", "value": value},
                    {"role": "assistant", "tool_calls": [{"id": "nested"}]}]}
            assert cache.wire_bytes(body, snap, "plain") == json.dumps(body).encode()
            assert cache._ACTIVE.get().message_bytes <= 200

    exercise()


def test_render_cache_byte_eviction_changes_only_work(monkeypatch):
    import kernel_working_set_cache as cache
    monkeypatch.setattr(cache, "MAX_RENDER_BYTES", 1)
    snap = snapshot(source("a", state="DUE"))

    @transaction
    def exercise():
        first = cache.trial_render(snap, {"a": FULL}, "plain", render_assignment)
        second = cache.trial_render(snap, {"a": FULL}, "plain", render_assignment)
        assert first == second and first is not second
        assert not cache._ACTIVE.get().renders
        assert cache._ACTIVE.get().render_bytes == 0
        assert verify_certificate(second.wire, second.certificate)

    exercise()


def test_wire_trials_defer_witnesses_and_bind_immutable_assignment(monkeypatch):
    import kernel_working_set_render as renderer
    from kernel_working_set_allocator import DEFERRED
    snap = snapshot(source("a", 'quote " λ\n'))
    calls = []
    original = renderer.render_assignment

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(renderer, "render_assignment", counted)

    @transaction
    def exercise():
        modes = {"a": FULL}
        preview = renderer.preview_assignment(snap, modes)
        expected = original(snap, modes)
        assert preview.wire == expected.wire and calls == []
        modes["a"] = DEFERRED
        assert renderer.materialize(preview) == expected
        assert calls == [True]
        assert renderer.materialize(preview) == expected and calls == [True]
        bad = replace(preview, wire=preview.wire + b" ", candidate=None)
        with pytest.raises(ValueError, match="trial bytes"):
            renderer.materialize(bad)

    exercise()


@pytest.mark.parametrize("changes,expected", [
    ({"protected_paths": ("a.py",)}, (FULL,)),
    ({"current_revisions": (("a.py", "sha256:" + "2" * 64),)}, (DEFERRED,)),
    ({"recovery_allowed": False}, (FULL, DEFERRED)),
    ({"recovery_owner_allowed": False}, (FULL, DEFERRED)),
    ({"recovery_schema": b'{"type":"object","additionalProperties":false}'}, (FULL,)),
])
def test_source_fragment_cache_binds_recovery_and_protection(changes, expected):
    original = source("a")
    snap = snapshot(original)

    @transaction
    def exercise():
        assert domain(snap, original) == (FULL, DEFERRED)
        changed = replace(snap, **changes)
        assert domain(changed, original) == expected
        # A different immutable capture with the same key cannot reuse a route.
        due = replace(original, state="DUE")
        assert domain(snap, due) == (FULL,)

    exercise()
