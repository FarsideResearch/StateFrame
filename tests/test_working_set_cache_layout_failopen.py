"""G8: a final_cache closure-proof failure must fall open to the plain layout under ON.

Mainline render_final_cache_candidate fails open to plain when the final-layout source
proof fails and refuses (cache_layout_refused) only when plain fails too. The allocator
path picks the layout before the closure proof runs, so prepare_live has to apply the
same fail-open itself instead of raising an untyped RequestSealError.
"""
import json

import pytest

from test_working_set_cutover import hook as hook  # Re-export the shared fixture.


def _rows(capture):
    return [json.loads(line) for line in capture.journal.path.read_bytes().splitlines()]


def _prefer_final_cache(monkeypatch):
    """Make the final_cache candidate win whenever the snapshot still admits it.

    The real selector takes the smallest wire; on the fixture body plain is smaller, so
    pin the choice while honouring the snapshot's admitted layouts (the fallback must
    restrict those, not just re-run the same search)."""
    import kernel_working_set_prepare as prep
    original = prep._layout_candidates

    def pick(snapshot, modes):
        candidates = original(snapshot, modes)
        cache = [c for c in candidates if c.certificate.layout == "final_cache"]
        return cache or candidates
    monkeypatch.setattr(prep, "_layout_candidates", pick)


def _install_layout_proof(monkeypatch, failing_layouts):
    """Inject a closure-proof verdict keyed on the layout the certified body carries."""
    import kernel_working_set_live as live
    seen = []
    original = live.assert_certified_body

    def record(body, prepared):
        seen.append(prepared.rendered.certificate.layout)
        return original(body, prepared)
    monkeypatch.setattr(live, "assert_certified_body", record)

    def verify(plan, messages, **kwargs):
        if seen[-1] in failing_layouts:
            return {"status": "INCOMPLETE", "proof_status": "FAILED",
                    "proof_failures": [{"kind": "injected_" + seen[-1]}]}
        return {"status": "COMPLETE", "proof_status": "VERIFIED"}
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery", verify)
    return seen


def _prepare(hook, monkeypatch):
    from kernel_working_set_live import prepare_live
    from test_final_cache_layout import _body
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    _run, capture, runtime, body = hook
    # A body the final-cache projection accepts; the cutover hook body cannot project.
    messages = _body(exchange=True)["messages"]
    frame = next(m for m in messages if m["content"].startswith("[StateFrame]\n"))
    payload = json.loads(frame["content"].split("\n", 1)[1])
    payload["state"]["framefit5"]["mode"] = "FRAME_ONLY"
    frame["content"] = "[StateFrame]\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    body = {**body, "messages": messages}
    return lambda: prepare_live(capture, runtime, body, dialect="llama.cpp",
                                reasoning_mode="off"), capture, runtime


def test_final_cache_candidate_is_chosen_when_it_wins(hook, monkeypatch):
    prepare, capture, _ = _prepare(hook, monkeypatch)
    _prefer_final_cache(monkeypatch)
    seen = _install_layout_proof(monkeypatch, set())
    _body, prepared = prepare()
    assert seen == ["final_cache"]
    assert prepared.rendered.certificate.layout == "final_cache"
    assert not any(row["kind"] == "final_cache_layout_failed_open" for row in _rows(capture))


def test_cache_layout_proof_failure_falls_open_to_plain(hook, monkeypatch):
    from kernel_cache_layout_view import semantic_messages
    prepare, capture, _ = _prepare(hook, monkeypatch)
    _prefer_final_cache(monkeypatch)
    seen = _install_layout_proof(monkeypatch, {"final_cache"})
    body, prepared = prepare()
    assert seen == ["final_cache", "plain"]
    assert prepared.rendered.certificate.layout == "plain"
    assert capture._working_set_certificate is prepared.rendered.certificate
    assert capture.request_source_final_delivery["status"] == "COMPLETE"
    # The plain body is the certified wire, not a re-projection of the failed one.
    assert body == json.loads(prepared.wire)
    assert semantic_messages(body["messages"]) == body["messages"]
    rows = _rows(capture)
    failed_open = [row for row in rows if row["kind"] == "final_cache_layout_failed_open"]
    assert [row["reason"] for row in failed_open] == ["closure_proof_failed"]
    assert failed_open[0]["failures"] == ["injected_final_cache"]
    from kernel_continuation_restart import SAFE_TAIL, resume_event_rejection_reason
    assert failed_open[0]["kind"] in SAFE_TAIL
    assert resume_event_rejection_reason(failed_open[0]["kind"]) is None
    prepared_rows = [row for row in rows if row["kind"] == "working_set_prepared"]
    assert len(prepared_rows) == 1 and prepared_rows[0]["body_sha256"] == \
        prepared.rendered.certificate.body_sha256


def test_plain_failure_after_fallback_is_a_typed_refusal(hook, monkeypatch):
    from kernel_request_seal import RequestSealError
    prepare, capture, _ = _prepare(hook, monkeypatch)
    _prefer_final_cache(monkeypatch)
    seen = _install_layout_proof(monkeypatch, {"final_cache", "plain"})
    with pytest.raises(RequestSealError, match="^cache_layout_refused: .*injected_plain") as refused:
        prepare()
    assert refused.value.harness_contract is True
    assert refused.value.outcome == "HARNESS_CONTRACT"
    assert seen == ["final_cache", "plain"]
    assert not any(row["kind"] == "working_set_prepared" for row in _rows(capture))


def test_plain_proof_failure_without_cache_layout_still_raises(hook, monkeypatch):
    from kernel_request_seal import RequestSealError
    prepare, capture, _ = _prepare(hook, monkeypatch)
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    seen = _install_layout_proof(monkeypatch, {"plain"})
    with pytest.raises(RequestSealError, match="^certified working-set failed closure proof"):
        prepare()
    assert seen == ["plain"]
    assert not any(row["kind"] == "final_cache_layout_failed_open" for row in _rows(capture))


def test_fallback_rebuilds_closure_from_pristine_plan(hook, monkeypatch):
    import copy
    from kernel_request_source_window import source_revision

    prepare, capture, runtime = _prepare(hook, monkeypatch)
    _prefer_final_cache(monkeypatch)
    seen = _install_layout_proof(monkeypatch, {"final_cache"})
    plan = runtime._request_source_closure
    content = "VALUE = 1\n"
    (runtime.owner.root / "a.py").write_text(content)
    plan["paths"] = ["a.py"]
    plan["units"] = [{"path": "a.py", "start_line": 1, "end_line": 1}]
    plan["source_contents"] = {"a.py": content}
    plan["revisions"] = {"a.py": source_revision(content)}
    original = copy.deepcopy(plan)
    # Reject a per-attempt deferral as well as the cache layout. No mutation of
    # that rejected closure may leak into either the source plan or the retry.
    from kernel_request_source_window import verify_final_source_delivery as verdict
    closures = []

    def verify(plan, messages, **kwargs):
        if seen[-1] == "final_cache":
            plan["deferred_paths"] = ["a.py"]
        closures.append(copy.deepcopy(plan))
        return verdict(plan, messages, **kwargs)

    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery", verify)

    _body, prepared = prepare()

    assert seen == ["final_cache", "plain"]
    assert [closure["deferred_paths"] for closure in closures] == [["a.py"], []]
    assert runtime._request_source_closure == original
    closure = capture._frame_candidate_inputs["source_plan"]
    assert closure["deferred_paths"] == []
    assert capture._request_source_closure_override is closure
    assert capture._frame_candidate_inputs["receipt"] == capture.request_source_final_delivery
    assert prepared.rendered.certificate.layout == "plain"


def test_blocked_plain_fallback_suspends_without_publishing(hook, monkeypatch):
    import kernel_working_set_prepare as prep
    from kernel_working_set_state import WorkingSetState
    from kernel_working_set_stop import ReasonCode, WorkingSetSuspended

    prepare, capture, runtime = _prepare(hook, monkeypatch)
    _prefer_final_cache(monkeypatch)
    seen = _install_layout_proof(monkeypatch, {"final_cache"})
    original = prep.prepare_candidate
    layouts = []

    def prepare_or_block(snapshot, **kwargs):
        layouts.append(snapshot.layouts)
        if snapshot.layouts == ("plain",):
            return prep.Blocked(ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED,
                                0, 0, snapshot.byte_cap, ())
        return original(snapshot, **kwargs)

    monkeypatch.setattr(prep, "prepare_candidate", prepare_or_block)
    with pytest.raises(WorkingSetSuspended) as stopped:
        prepare()

    assert layouts == [("plain", "final_cache"), ("plain",)]
    assert seen == ["final_cache"]
    assert stopped.value.outcome()["result"] == "BLOCKED"
    saved = runtime.working_set_state.checkpoint()
    assert WorkingSetState.from_checkpoint(saved).checkpoint() == saved
    rows = _rows(capture)
    assert any(row["kind"] == "working_set_blocked" for row in rows)
    assert not any(row["kind"] in {"working_set_prepared", "qualified_request_sealed"}
                   for row in rows)
    assert getattr(capture, "request_source_final_delivery", None) is None


def test_off_provider_bytes_match_group_base(tmp_path, tmp_path_factory):
    from test_working_set_astra_off_parity import test_off_provider_bytes_match_cutover_base

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, "1f6b7e9bc1")
