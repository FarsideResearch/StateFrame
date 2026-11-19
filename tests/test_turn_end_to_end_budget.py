"""Phase and composed launch-to-submission budgets for a large workspace."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from test_turn_qualification_budget import repository  # noqa: F401


@pytest.fixture
def _qualification_fixture(repository):
    contracts = pytest.importorskip("kernel_turn_contracts")
    qualification = pytest.importorskip("kernel_turn_qualification")
    host = pytest.importorskip("kernel_turn_host")
    work = Counter()
    state = SimpleNamespace(held=False)
    blobs = {}
    submissions = []

    def stage(codec, data):
        ref = contracts.Ref(codec, sha256(data).hexdigest(), len(data))
        if codec == qualification.CERTIFICATE_CODEC:
            assert json.loads(data)["guard_deps"] == [["workspace-path", selector.hex(), "present"]]
        blobs[ref] = data
        work[codec] += 1
        return ref

    def load(ref):
        work["loads"] += 1
        return blobs[ref]

    @contextmanager
    def pin(scope):
        assert not state.held and scope == scope_ref
        yield contracts.Snapshot(state.basis.live, scope_ref, state.basis.epoch)

    @contextmanager
    def authority(snapshot):
        assert snapshot.root == state.basis.live
        state.held = True
        try:
            work["observations"] += 1
            yield state.basis
        finally:
            state.held = False

    @contextmanager
    def transaction():
        assert not state.held
        yield

    def guard(at, use, inputs):
        assert not state.held and use == "send" and inputs == guard_inputs
        if at != state.basis:
            raise contracts.Stale("qualification basis changed")
        work["guards"] += 1
        return state.guard

    def count(meter, content, deadline):
        assert not state.held and (deadline is None or time.monotonic() < deadline)
        work["counts"] += 1
        return contracts.Count(meter, sha256(content.encode()).hexdigest(), 90)

    def submit(wire):
        assert state.held
        submissions.append(wire)

    def make_candidate(step, *, route="read", content=None, max_tokens=200,
                       method="tokenize"):
        body = {"model": "synthetic", "messages": [
            {"role": "user", "content": content or f"Read result {step}, café"},
        ], "tools": [], "max_tokens": max_tokens}
        wire = json.dumps(body).encode()
        meter = contracts.MeterId(route, body["model"], "verified", "count")
        fit = qualification.stage_fit_inputs(
            publication, at=state.basis, scope=scope_ref, guard_inputs=guard_inputs,
            guard=state.guard, request_id=f"request-{step}", meter=meter, budget={
                "status": "READY", "final_validation_failures": [],
                "measure_method": method, "body_sha256": "sha256:" + sha256(wire).hexdigest(),
                "request_bytes": len(wire), "max_request_bytes": 5000,
                "max_prompt_tokens": 100, "measured_prompt_tokens": 93,
                "template_overhead_tokens": 3,
            },
        )
        return contracts.Candidate(wire, fit, state.guard, meter)

    scope_ref = stage("turn/scope/1", b"{}")
    guard_inputs = stage("turn/guard-inputs/1", b"{}")
    # Charge fixture admission once. Guard observations below carry the admitted
    # generation. Production Live must provide equivalent reconciliation.
    digest = sha256()
    for path in sorted(repository.glob("*/*.py")):
        digest.update(path.read_bytes())
        work["reads"] += 1
    root = digest.hexdigest()
    selector = b"source-\xff.py"
    dep = contracts.Dep(contracts.Key("workspace-path", selector), "present")
    state.guard = contracts.Artifact(stage("turn/guard/1", b"{}"), (dep,))
    state.basis = contracts.Basis(root, root, 1, contracts.Head(0, "genesis"), 0, "initial")
    publication = SimpleNamespace(load=load, stage=stage, transaction=transaction)
    live = host.WorkspaceLive(authority)
    with live.bound(SimpleNamespace(pin=pin)):
        port = qualification.open_port(
            contracts.Context("principal", "run", "attempt", repository.parent, "profile"),
            live=live, panels=SimpleNamespace(),
            proofs=SimpleNamespace(guard=guard), meter=SimpleNamespace(count=count),
            publication=publication,
        )
        yield state, port, work, submissions, submit, make_candidate


def test_fenced_qualification_over_200_turns(_qualification_fixture, record_property):
    state, port, work, submissions, submit, candidate = _qualification_fixture
    elapsed = []
    for step in range(201):
        state.basis = state.basis._replace(step=step)
        started = time.perf_counter()
        item = candidate(step)
        approved = port.qualify(state.basis, item, deadline=None)
        assert port.qualify(state.basis, item, deadline=None) is approved
        port.handoff(approved, submit)
        elapsed.append(time.perf_counter() - started)
        assert submissions[-1] is item.wire
    record_property("fenced_send_one_seconds", elapsed[0])
    record_property("fenced_max_turn_seconds", max(elapsed[1:]))
    record_property("fenced_work", str(dict(work)))
    assert elapsed[0] < 0.60 and max(elapsed[1:]) < 0.65
    assert work["counts"] == work["turn/certificate/1"] == work["loads"] == 201
    assert work["guards"] == 402
    assert work["reads"] == 5000


def test_fenced_handoff_rechecks_rewrites_routes_and_retries(_qualification_fixture):
    contracts = pytest.importorskip("kernel_turn_contracts")
    state, port, work, submissions, submit, candidate = _qualification_fixture
    original = candidate(0)
    approved = port.qualify(state.basis, original, deadline=None)
    # Changing output allocation changes the wire but preserves the exact prompt.
    changed_output = candidate(0, max_tokens=100)
    same_prompt = port.qualify(state.basis, changed_output, deadline=None)
    assert same_prompt.wire != approved.wire
    assert work["counts"] == 1

    rewritten = candidate(0, content="A hook rewrote the user message")
    final = port.qualify(state.basis, rewritten, deadline=None)
    assert work["counts"] == 2
    with pytest.raises(contracts.Stale, match="selected candidate"):
        port.handoff(approved, submit)
    port.handoff(final, submit)
    port.handoff(final, submit)
    assert submissions == [rewritten.wire, rewritten.wire]

    state.basis = state.basis._replace(authority="other route")
    with pytest.raises(contracts.Stale, match="basis changed"):
        port.handoff(final, submit)
    moved = candidate(0, route="finish", content="A hook rewrote the user message")
    moved_certificate = port.qualify(state.basis, moved, deadline=None)
    assert work["counts"] == 3
    state.guard = state.guard._replace(value=state.guard.value._replace(size=1))
    with pytest.raises(contracts.Stale, match="guard changed"):
        port.handoff(moved_certificate, submit)
    assert submissions == [rewritten.wire, rewritten.wire]


def test_fenced_qualification_rejects_estimates_and_expired_receipts(_qualification_fixture):
    contracts = pytest.importorskip("kernel_turn_contracts")
    state, port, work, submissions, submit, candidate = _qualification_fixture
    exact = candidate(0)
    approval = port.qualify(state.basis, exact, deadline=None)
    with pytest.raises(contracts.Unavailable, match="requires exact"):
        port.qualify(state.basis, candidate(0, method="calibrated_bytes"), deadline=None)
    with pytest.raises(contracts.Stale, match="selected candidate"):
        port.handoff(approval, submit)
    with pytest.raises(TimeoutError, match="deadline expired"):
        port.qualify(state.basis, exact, deadline=time.monotonic() - 1)
    assert work["counts"] == 1 and not submissions


@pytest.mark.host_sandbox
@pytest.mark.skipif(os.environ.get("STATEFRAME_TEST_TURN_SYSTEM") != "on",
                    reason="requires the composed turn adapters and a verified tokenizer")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_every_send_under_five_seconds(tmp_path, monkeypatch, warm):
    from _journal import measurement_events
    from test_workspace_state_budget import _large_repo
    from test_workspace_state_oracle import scripted_provider

    from stateframe import runner
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_tool_settings import ToolSettings

    repo = tmp_path / "repo"
    _large_repo(repo)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    paths = ["answer.py", *(path.relative_to(repo).as_posix()
                            for path in sorted((repo / "modules").glob("*/*.py"))[:200])]
    reads = [{"tool": "read", "arguments": {"path": path}} for path in paths]
    switches = (
        "WORKSPACE", "PANELS", "DELIVERY_PROOFS", "MINIMAL_LAUNCH", "INDEXED_READS",
        "BOUNDED_QUALIFICATION", "ATOMIC_PUBLICATION", "GENERATION_OVERLAP", "LOCAL_EXACT_METER",
    )
    extra_env = {f"STATEFRAME_TURN_{switch}": "on" for switch in switches}

    def run_trajectory(name, steps, url):
        run = tmp_path / name
        options = runner.RunOptions(
            repo=repo, task="Read the relevant source before fixing the failing check.",
            endpoint=Endpoint.from_args(url, "scripted", None), run_dir=run,
            target="answer.py", test_python=sys.executable, max_turns=steps,
            tool_settings=ToolSettings(run_env=False), timeout_s=steps * 5 + 10, quiet=True,
            extra_env=extra_env,
        )
        started = time.monotonic_ns()
        runner.run(options)
        return run, started

    with scripted_provider([reads[0], *reads] if warm else reads) as (url, trace):
        if warm:
            run_trajectory("seed", 1, url)
        run, started = run_trajectory("measured", len(reads), url)
    trace.script.assert_consumed()
    requests = [stamp for stamp, _ in trace.requests[int(warm):]]
    responses = trace.responses[int(warm):]
    assert len(requests) == len(responses) == len(reads)
    gaps = [(requests[0] - started) / 1e9, *[
        (request - response) / 1e9 for request, response in zip(requests[1:], responses)
    ]]
    prepare = [row["host_seconds"] for row in measurement_events(run)
               if row.get("kind") == "request_prepare_timing"]
    assert len(prepare) >= len(reads)
    print(f"launch={gaps[0]:.3f}s max_gap={max(gaps):.3f}s max_prepare={max(prepare):.3f}s")
    assert max(gaps) < 5
    assert max(prepare) < 5
