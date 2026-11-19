"""Exact baseline differential and wall limits for the recorded 98-slot layout."""
import gzip
import itertools
import json
import os
import pickle
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from time import perf_counter

import pytest
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_cache import _ACTIVE, independent, transaction
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_render import domain, render_assignment
from test_working_set_prepare import snapshot, source
from _stateframe_home import isolated_stateframe_home
from tools.pinned_tree import pinned_tree
from working_set_benchmark import admission, admit_request, recorded_read, recorded_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


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


def oracle_value(value):
    """Lossless JSON projection: include every field, byte and container/type tag."""
    kind = type(value).__module__ + "." + type(value).__qualname__
    if isinstance(value, Enum):
        return {"$enum": kind, "value": oracle_value(value.value)}
    if is_dataclass(value):
        return {"$type": kind, "fields": {
            field.name: oracle_value(getattr(value, field.name)) for field in fields(value)
            if not (kind == "kernel_working_set_snapshot.PreparationSnapshot"
                    and field.name == "protected_ranges" and not value.protected_ranges)}}
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, tuple):
        return {"$tuple": [oracle_value(item) for item in value]}
    if isinstance(value, list):
        return [oracle_value(item) for item in value]
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value), "oracle requires string keys"
        return {key: oracle_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported oracle value: {kind}")


@pytest.fixture(scope="module")
def admission_reference(tmp_path_factory):
    """Keep frozen raw oracles; price promoted reads against the carried parent."""
    raw = recorded_snapshot()
    _, display, _, page = recorded_read()
    revisions = dict(raw.current_revisions)
    revisions[display] = page.metadata["revision"]
    protected = replace(raw, protected_paths=tuple(sorted(set(raw.protected_paths) | {display})),
                        current_revisions=tuple(sorted(revisions.items())))
    base = pinned_tree("d7fd3aad93", tmp_path_factory)
    work = tmp_path_factory.mktemp("protected-admission-reference")
    inputs, outputs = work / "inputs.pickle", work / "outputs.json"
    inputs.write_bytes(pickle.dumps((raw, protected)))
    driver = """
import json, pickle, sys
from pathlib import Path
from kernel_working_set_prepare import prepare_candidate
from test_working_set_performance import oracle_value
from working_set_benchmark import admission, admit_request
raw, protected = pickle.loads(Path(sys.argv[1]).read_bytes())
results = {name: {"prepared": oracle_value(prepare_candidate(snap)),
                  "admission": oracle_value(admission(snap)),
                  "read": oracle_value(admit_request(snap))}
           for name, snap in (("raw", raw), ("protected", protected))}
Path(sys.argv[2]).write_text(json.dumps(results, sort_keys=True))
"""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment["PYTHONPATH"] = os.pathsep.join(str(base / path) for path in
        ("src/stateframe/engine", "src", "tests"))
    result = subprocess.run([sys.executable, "-c", driver, str(inputs), str(outputs)],
                            cwd=base, env=environment, capture_output=True, text=True,
                            timeout=600, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return protected, json.loads(outputs.read_text())


@pytest.mark.parametrize("baseline", ["7467debf1", "911db220e"])
def test_pinned_baseline_exact_results_and_randomized_assignments(baseline, admission_reference):
    # Captured once by the old isolated reference_results path, before its removal.
    # JSON is sorted/compact; gzip mtime=0 follows the recorded 98-slot fixture.
    expected = json.loads(gzip.decompress(
        (FIXTURES / f"working_set_reference_{baseline}.json.gz").read_bytes()))
    assert expected["schema"] == "working-set-reference/1"
    assert expected["baseline"].startswith(baseline)
    # Keep the old allocator oracle, with read metadata bound to the target
    # release. Its four later text-inspection fields remain in provenance.
    assert expected["read_base"] == "1de34aa46b01deb0aecf2d43287e45df4134809f"
    assert set(expected["source_read_metadata"]) == {
        "bom", "newline_style", "newline_styles", "valid_utf8"}
    cases = [(recorded_snapshot(), ()), *random_cases()]
    for index, ((snap, assignments), reference) in enumerate(zip(cases, expected["cases"], strict=True)):
        actual = prepare_candidate(snap)
        assert oracle_value(actual) == reference["prepared"], (baseline, index)
        renders = (render_assignment(snap, assignment, layout)
                   for assignment in assignments for layout in snap.layouts)
        for ordinal, (rendered, pinned) in enumerate(zip(renders, reference["renders"], strict=True)):
            assert oracle_value(rendered) == pinned, (baseline, index, ordinal)
        if isinstance(actual, PreparedCandidate):
            assert verify_certificate(actual.wire, actual.rendered.certificate)
    protected, parent = admission_reference
    # Point 8 still reproduces the frozen pre-promotion admission results.
    assert parent["raw"]["admission"] == expected["admission"]
    assert parent["raw"]["read"] == expected["read"]
    # Compare complete preparation/admission with the same promoted inputs,
    # as the floor fix requires, without depending on retired Git objects.
    prepared = prepare_candidate(protected)
    assert oracle_value(prepared) == parent["protected"]["prepared"]
    if isinstance(prepared, PreparedCandidate):
        assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    assert oracle_value(admission(protected)) == parent["protected"]["admission"]
    assert oracle_value(admit_request(protected)) == parent["protected"]["read"]
    print(f"exact differential: {len(cases)} frozen preparations plus promoted parent, "
          f"{sum(len(a) for _, a in cases)} assignments, cold and fresh admission floors")


def test_recorded_read_matches_target_release_in_every_field(tmp_path_factory):
    base = pinned_tree("56af0dc6d6", tmp_path_factory)
    raw, display, request, page = recorded_read()
    driver = '''
import json, sys
from dataclasses import asdict
from kernel_source_delivery import deliver_source, normalize_read_request
data = json.loads(sys.stdin.read())
page = deliver_source(bytes.fromhex(data["raw"]), data["display"], 1,
                      args=normalize_read_request(data["arguments"]))
print(json.dumps(asdict(page), sort_keys=True))
'''
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment["STATEFRAME_HOME"] = isolated_stateframe_home()
    environment["PYTHONPATH"] = os.pathsep.join(map(str, (
        base / "src/stateframe/engine", base / "src")))
    result = subprocess.run([sys.executable, "-c", driver], cwd=base, env=environment,
                            input=json.dumps({"raw": raw.hex(), "display": display,
                                              "arguments": dict(request.arguments)}),
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout) == asdict(page)


def test_oracle_value_preserves_fields_types_and_exact_bytes():
    class Mode(str, Enum):
        FULL = "FULL"

    @dataclass(frozen=True)
    class Result:
        wire: bytes
        details: tuple

    value = Result(b'"\\\x00\xff', (Mode.FULL, ["λ", None, True, 1, -0.0]))
    encoded = oracle_value(value)
    assert encoded == {
        "$type": type(value).__module__ + "." + type(value).__qualname__,
        "fields": {"wire": {"$bytes": "225c00ff"}, "details": {"$tuple": [
            {"$enum": Mode.__module__ + "." + Mode.__qualname__, "value": "FULL"},
            ["λ", None, True, 1, -0.0]]}}}
    assert json.loads(json.dumps(encoded, sort_keys=True, allow_nan=False)) == encoded
    assert oracle_value(("a",)) != oracle_value(["a"])
    assert oracle_value(Mode.FULL) != oracle_value("FULL")
    assert oracle_value(replace(value, wire=value.wire + b" ")) != encoded
    assert oracle_value({"result": value}) == {"result": encoded}
    with pytest.raises(TypeError, match="unsupported oracle value"):
        oracle_value(object())
    with pytest.raises(AssertionError, match="string keys"):
        oracle_value({1: "not a JSON key"})


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
