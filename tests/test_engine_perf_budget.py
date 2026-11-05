"""Deterministic count budgets for request fitting and preparation hot paths."""
from __future__ import annotations

import hashlib
import inspect
import json
import pytest

import kernel_code_bookmarks
import kernel_frame_source_snapshot as source_snapshot
import kernel_measurement as measurement
import kernel_request_budget
import kernel_stateframe_call_coverage as call_coverage
from kernel_frame_source_snapshot import fit_recent_read_windows
from kernel_measurement import canonical_bytes, externalize_journal_fields
from kernel_continuation_journal import ContinuationJournal
import kernel_request_diet as request_diet
from kernel_request_diet import RequestTokenMeter
from kernel_frame_allocation_capture import (
    FrozenRequestCounter, _source_candidate_nonfit, final_request_tokens,
)
from kernel_final_wire_budget import reconcile_frozen_budget
from kernel_request_measurement import RequestMeasurement, prompt_projection
from types import SimpleNamespace


def _body() -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": "task"}],
        "tools": [],
        "max_tokens": 4096,
    }


def _windows(count: int = 22, *, chars: int = 512) -> list[dict[str, object]]:
    return [
        {
            "path": f"pkg/module_{index:02}.py",
            "content": f"[fixture source {index:02}]\n" + "x" * chars,
            "source_chars": chars,
            "bookmark": True,
            "rank": index + 1,
        }
        for index in range(count)
    ]


def _fit_kwargs(candidate_nonfit=None):
    kwargs = {"fits": lambda _candidate: False}
    # c60ec35 has no proof-pruning hook. Exercise its exhaustive behavior so
    # this planted count guard fails by assertion, not by an unsupported kwarg.
    if candidate_nonfit is not None and "candidate_nonfit" in inspect.signature(
            fit_recent_read_windows).parameters:
        kwargs["candidate_nonfit"] = candidate_nonfit
    return kwargs


def test_candidate_trials_detach_each_request_from_the_base(monkeypatch):
    monkeypatch.setattr(kernel_code_bookmarks, "enabled", lambda: True)
    calls = 0
    fit_calls = 0
    original = source_snapshot.copy.deepcopy

    def counted_deepcopy(value, memo=None):
        nonlocal calls
        calls += 1
        return original(value, memo) if memo is not None else original(value)

    monkeypatch.setattr(source_snapshot.copy, "deepcopy", counted_deepcopy)
    body = _body()
    def reject(_candidate):
        nonlocal fit_calls
        fit_calls += 1
        return False

    fit_recent_read_windows(body, _windows(), fits=reject)

    # The detached base preserves return isolation; each mutable callback input
    # needs its own recursively detached request as well.
    assert fit_calls > 1
    assert calls >= fit_calls + 1


def test_tokenize_calls_per_22_source_fit_use_only_proven_byte_rejections(monkeypatch):
    monkeypatch.setattr(kernel_code_bookmarks, "enabled", lambda: True)
    exact_prompts: set[str] = set()

    def counted_exact_fit(candidate):
        # Model the run meter's exact-content memo: one remote count per unique
        # projected prompt even though coverage variants and both exhaustive
        # passes may present an identical body.
        exact_prompts.add(prompt_projection(candidate))
        return False

    token_budget = 400

    def proven_nonfit(candidate):
        wire_size = len(json.dumps(candidate).encode())
        return wire_size > token_budget * 16

    fit_recent_read_windows(
        _body(), _windows(chars=8192),
        **{
            **_fit_kwargs(proven_nonfit),
            "fits": counted_exact_fit,
        },
    )

    # The empty proposal is below 400*16 bytes. Each nonempty proposal adds
    # at least one 8 KiB ASCII source, so all 22 source additions are proven
    # over budget by the verified envelope. At most the empty prompt is counted.
    assert len(exact_prompts) <= 1


def test_send_measurement_reuses_the_prepared_wire(monkeypatch):
    body = _body()
    prepared_wire = json.dumps(body).encode()
    dumps_of_body = 0
    original_dumps = json.dumps

    def counted_dumps(value, *args, **kwargs):
        nonlocal dumps_of_body
        if value is body and not kwargs.get("sort_keys", False):
            dumps_of_body += 1
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", counted_dumps)
    kwargs = {
        "tokenizer_identity": "fixture-tokenizer",
        "request_id": "perf-r1",
        "logical_request_id": "perf-r1",
        "tokens": 10,
        "method": "tokenize",
        "template_overhead_tokens": 0,
        "max_prompt_tokens": 100,
        "max_request_bytes": 1600,
    }
    if "wire" in inspect.signature(RequestMeasurement.bind).parameters:
        kwargs["wire"] = prepared_wire
    measurement = RequestMeasurement.bind(body, **kwargs)

    # The caller already produced this exact transport serialization. Binding
    # measurement may hash/count it, but must not serialize the mutable body again.
    assert measurement.request_bytes == len(prepared_wire)
    assert measurement.body_sha256 == "sha256:" + hashlib.sha256(prepared_wire).hexdigest()
    assert dumps_of_body == 0


def test_final_wire_preparation_serializes_transport_body_once(monkeypatch):
    body = _body()
    meter = RequestTokenMeter(tokenizer=lambda _text: 10)
    meter.bind_request("perf-r1", "perf-r1")
    tokens, method = meter.measure(body)
    capture = SimpleNamespace(
        request_id="perf-r1",
        request_action_contract={"admitted_tool_schemas": []},
        request_budget_final_wire={},
        _frame_candidate_inputs={"token_ceiling": 33344},
        _frame_candidate_counter=FrozenRequestCounter.capture(
            body, meter, tokens, method, "perf-r1"),
        _frame_candidate_fallback_fit=False,
    )
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: meter)
    dumps_of_body = 0
    original_dumps = json.dumps

    def counted_dumps(value, *args, **kwargs):
        nonlocal dumps_of_body
        if value is body and not kwargs.get("sort_keys", False):
            dumps_of_body += 1
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", counted_dumps)
    reconcile_frozen_budget(capture, SimpleNamespace(), body)

    # This send seam creates one transport snapshot. Measurement and source
    # delivery consume that same immutable byte string.
    assert dumps_of_body == 1


def test_frozen_candidate_measurement_reuses_prepared_wire_size(monkeypatch):
    initial = _body()
    meter = RequestTokenMeter(tokenizer=lambda _text: [1, 2, 3],
                              template_overhead_tokens=0)
    meter.bind_request("perf-r1", "perf-r1")
    initial_tokens, initial_method = meter.measure(initial)
    capture = SimpleNamespace(
        request_id="perf-r1",
        _frame_candidate_counter=FrozenRequestCounter.capture(
            initial, meter, initial_tokens, initial_method, "perf-r1"),
    )
    candidate = _body()
    candidate["messages"][0]["content"] = "changed candidate"
    prepared_wire = json.dumps(candidate).encode()
    byte_serializations = 0
    original_wire_bytes = request_diet._wire_bytes

    def counted_wire_bytes(value):
        nonlocal byte_serializations
        if value is candidate:
            byte_serializations += 1
        return original_wire_bytes(value)

    monkeypatch.setattr(request_diet, "_wire_bytes", counted_wire_bytes)
    kwargs = {"wire_bytes": len(prepared_wire)} if "wire_bytes" in inspect.signature(
        final_request_tokens).parameters else {}
    tokens, method = final_request_tokens(candidate, meter, capture, **kwargs)

    # The fit callback passes its just-serialized body size through both frozen
    # counter checks, so exact counting does not serialize that body again.
    assert (tokens, method) == (3, "tokenize")
    assert byte_serializations == 0


@pytest.mark.parametrize(("endpoint", "expected_contexts"), [
    ("http://127.0.0.1:8080/tokenize", 0),
    ("https://127.0.0.1:8443/tokenize", 1),
])
def test_tokenizer_ssl_contexts_are_built_once_per_meter(
        monkeypatch, endpoint: str, expected_contexts: int):
    import http.client
    import ssl
    import urllib.error
    import kernel_provider_dialect

    # Isolate transport setup from endpoint classification: this test exercises
    # SSL context lifetime, while provider refusal is covered separately.
    monkeypatch.setattr(
        kernel_provider_dialect,
        "capabilities_for",
        lambda *_args, **_kwargs: SimpleNamespace(dialect="llama.cpp"),
    )

    contexts = 0
    original_factory = ssl._create_default_https_context

    def counted_context(*args, **kwargs):
        nonlocal contexts
        contexts += 1
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(ssl, "_create_default_https_context", counted_context)

    def fail_connect(_connection):
        raise OSError("synthetic local tokenizer connection failure")

    monkeypatch.setattr(http.client.HTTPConnection, "connect", fail_connect)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", fail_connect)
    meter = RequestTokenMeter(endpoint=endpoint, route_allow_remote=False)
    for prompt in ("first", "second"):
        with pytest.raises((OSError, urllib.error.URLError)):
            meter._count_tokens_uncached(prompt)

    # HTTP needs no TLS context; HTTPS builds one trusted context per run and
    # reuses it while each request gets a fresh opener.
    assert contexts == expected_contexts


def test_checkpoint_drops_and_rebuilds_journal_suffix_cache(monkeypatch, tmp_path):
    from collections import OrderedDict
    import kernel_checkpoint

    journal = ContinuationJournal(tmp_path, "checkpoint-suffix-cache", {"origin": "perf-test"})
    try:
        journal.emit("checkpoint_suffix_probe", payload={"value": 1})
        journal.cached_records_since(0)
        assert journal._records_since_cache
        monkeypatch.setattr(kernel_checkpoint, "_chunk_list", lambda *_args, **_kwargs: None)
        state = kernel_checkpoint._journal_state(journal)
        state["_fd"] = None
        assert "_records_since_cache" not in state
        assert "_records_since_cache_bytes" not in state

        restored = ContinuationJournal.__new__(ContinuationJournal)
        kernel_checkpoint._rebuild_journal(restored, state)
        assert restored._records_since_cache == OrderedDict()
        assert restored._records_since_cache_bytes == 0
    finally:
        journal.close()


def test_checkpointed_meter_rebuilds_its_tls_context(monkeypatch):
    import ssl
    import kernel_checkpoint

    contexts = 0
    original_factory = ssl._create_default_https_context

    def counted_context(*args, **kwargs):
        nonlocal contexts
        contexts += 1
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(ssl, "_create_default_https_context", counted_context)
    meter = RequestTokenMeter(
        endpoint="https://127.0.0.1:8443/tokenize", route_allow_remote=False,
    )
    meter_context = meter._get_http_ssl_context()
    assert contexts == 1
    state = kernel_checkpoint._runtime_state(
        meter, kernel_checkpoint.DROP_ATTRS[("kernel_request_diet", "RequestTokenMeter")]
    )
    assert "_http_opener" not in state[2]
    assert "_http_ssl_context" not in state[2]
    assert "_http_ssl_context_lock" not in state[2]

    restored = RequestTokenMeter.__new__(RequestTokenMeter)
    restored.__dict__.update(state[2])
    opener = restored._get_http_opener()
    assert opener is not None
    assert restored._get_http_opener() is not opener
    restored_context = restored._get_http_ssl_context()
    assert restored_context is not meter_context
    assert restored._get_http_ssl_context() is restored_context
    assert contexts == 2


def test_tokenizer_builds_fresh_openers_with_one_meter_context(monkeypatch):
    import ssl

    contexts = 0
    original_factory = ssl._create_default_https_context

    def counted_context(*args, **kwargs):
        nonlocal contexts
        contexts += 1
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(ssl, "_create_default_https_context", counted_context)
    meter = RequestTokenMeter(endpoint="https://127.0.0.1:8443/tokenize")
    context = meter._get_http_ssl_context()
    first_opener = meter._get_http_opener()
    second_opener = meter._get_http_opener()
    assert first_opener is not second_opener
    assert contexts == 1

    first_context_provider = next(
        handler._context_provider for handler in first_opener.handlers
        if isinstance(handler, request_diet._TokenizeHTTPSHandler)
    )
    second_context_provider = next(
        handler._context_provider for handler in second_opener.handlers
        if isinstance(handler, request_diet._TokenizeHTTPSHandler)
    )
    assert first_context_provider() is context
    assert second_context_provider() is context

    meter.endpoint = "https://localhost:8443/tokenize"
    third_opener = meter._get_http_opener()
    assert third_opener is not second_opener
    meter.endpoint = "http://127.0.0.1:8080/tokenize"
    fourth_opener = meter._get_http_opener()
    assert fourth_opener is not third_opener
    assert meter._get_http_ssl_context() is context
    assert contexts == 1


@pytest.mark.parametrize("seed", [0, 1, 7, 22])
@pytest.mark.parametrize("adversarial", [False, True])
def test_byte_pruned_fitting_matches_exhaustive_nonmonotone_selection(
        monkeypatch, seed: int, adversarial: bool):
    import random

    monkeypatch.setattr(kernel_code_bookmarks, "enabled", lambda: True)
    rng = random.Random(seed)
    windows = _windows(chars=512)
    for index, row in enumerate(windows):
        row["content"] += "y" * rng.randrange(0, 150)
        row["source_chars"] = len(str(row["content"]))
        row["rank"] = 1 if index == 0 else index + 2
    token_budget = 400

    def reference_token_count(candidate):
        # An exhaustive reference tokenizer consumes at most 16 wire bytes per
        # token, with a message-shape penalty that can make larger proposals
        # cheaper. It computes a count independently of the pruning predicate.
        wire = json.dumps(candidate).encode()
        count = 0
        offset = 0
        while offset < len(wire):
            count += 1
            offset += min(16, 1 + wire[offset] % 16)
        messages = candidate.get("messages", ())
        if len(messages) % 2 == 0:
            count += 500 if adversarial else 160
        return count

    smaller = _body()
    smaller["messages"].append({"role": "user", "content": "probe"})
    larger = _body()
    larger["messages"].extend([
        {"role": "user", "content": "probe"},
        {"role": "user", "content": "probe"},
    ])
    assert len(json.dumps(larger).encode()) > len(json.dumps(smaller).encode())
    assert reference_token_count(larger) < reference_token_count(smaller)

    def make_exact_fits():
        measured = []

        def exact_fits(candidate):
            measured.append(candidate)
            return reference_token_count(candidate) <= token_budget

        return exact_fits, measured

    baseline_fits, baseline_measured = make_exact_fits()
    optimized_fits, optimized_measured = make_exact_fits()
    proven_rejections = 0

    def proven_nonfit(candidate):
        nonlocal proven_rejections
        wire_size = len(json.dumps(candidate).encode())
        nonfit = wire_size > token_budget * 16
        proven_rejections += nonfit
        return nonfit

    baseline = fit_recent_read_windows(_body(), windows, fits=baseline_fits)
    optimized = fit_recent_read_windows(_body(), windows, **{
        **_fit_kwargs(proven_nonfit),
        "fits": optimized_fits,
    })
    assert optimized == baseline
    assert json.dumps(optimized).encode() == json.dumps(baseline).encode()
    assert proven_rejections > 0
    assert len(optimized_measured) < len(baseline_measured)


def test_candidate_trials_do_not_alias_body_or_each_other(monkeypatch):
    monkeypatch.setattr(kernel_code_bookmarks, "enabled", lambda: True)
    body = _body()
    body["messages"][0]["metadata"] = {"labels": ["source"]}
    body["messages"][0]["tool_calls"] = [{"function": {"name": "original_call"}}]
    body["tools"] = [{"function": {"name": "original_tool"}}]
    original = json.loads(json.dumps(body))
    observed = []

    def mutating_fit(candidate):
        observed.append((
            candidate["messages"][0]["metadata"]["labels"][0],
            candidate["messages"][0]["tool_calls"][0]["function"]["name"],
            candidate["tools"][0]["function"]["name"],
        ))
        candidate["messages"][0]["metadata"]["labels"][0] = "mutated metadata"
        candidate["messages"][0]["tool_calls"][0]["function"]["name"] = "mutated call"
        candidate["tools"][0]["function"]["name"] = "mutated tool"
        candidate["messages"][0]["content"] = "changed inside fit callback"
        candidate["messages"].append({"role": "user", "content": "trial only"})
        return False

    fit_recent_read_windows(body, _windows(2), fits=mutating_fit)
    assert len(observed) > 1
    assert set(observed) == {("source", "original_call", "original_tool")}
    assert body == original


def test_nonfit_meter_transition_matches_exhaustive_fallback_and_keeps_deadline(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_REQUESTS", 2)

    def unavailable(_text):
        raise TimeoutError("scripted tokenizer outage")

    prefiltered_meter = RequestTokenMeter(tokenizer=unavailable, template_overhead_tokens=0)
    exhaustive_meter = RequestTokenMeter(tokenizer=unavailable, template_overhead_tokens=0)
    for meter in (prefiltered_meter, exhaustive_meter):
        meter.begin_request("initial-outage")
        assert meter.measure(_body())[1] == "calibrated_bytes"
        meter.begin_request("fallback-wait-1")
        meter.begin_request("fallback-trials")
        assert meter._reprobe_due()

    trials = [
        {"messages": [{"role": "user", "content": f"trial {index}"}], "tools": []}
        for index in range(4)
    ]
    for trial in trials:
        serialized_bytes = request_diet._wire_bytes(trial)
        nonfit, measured = _source_candidate_nonfit(
            prefiltered_meter, trial, serialized_bytes, lambda _method: 1,
            hard_max_request_bytes=100, clip_to_hard_limit=False,
        )
        assert nonfit is True
        assert measured is not None and measured[1] == "calibrated_bytes"
        assert exhaustive_meter.measure(trial)[1] == "calibrated_bytes"

    stable_journal_keys = {
        "tokenize_availability", "tokenize_failure_reason", "tokenize_attempts",
        "tokenizer_calls", "tokenize_consecutive_failures", "fallback_bytes_per_token",
        "fallback_safety_margin", "fallback_calibration_samples", "fallback_rejection_bound",
        "tokenize_reprobe_status", "tokenize_reprobe_attempts",
    }

    def meter_state(meter):
        fields = meter.journal_fields("calibrated_bytes")
        return (
            tuple(sorted((key, fields.get(key)) for key in stable_journal_keys)),
            meter._fallback_pending,
            meter._fallback_reported,
            meter._reprobe_active,
            meter._reprobe_requests_since_attempt,
            frozenset(meter._reprobe_ready_requests),
            frozenset(meter._reprobe_attempted_requests),
            meter._last_tokenizer_error,
        )

    assert meter_state(prefiltered_meter) == meter_state(exhaustive_meter)

    deadline_meter = RequestTokenMeter(tokenizer=lambda _text: [1], template_overhead_tokens=0)
    deadline_meter.bind_request(
        "expired", "expired", deadline=request_diet.time.monotonic() - 1,
    )
    with pytest.raises(TimeoutError, match="tokenizer request deadline expired"):
        _source_candidate_nonfit(
            deadline_meter, _body(), request_diet._wire_bytes(_body()),
            lambda _method: 1, hard_max_request_bytes=100, clip_to_hard_limit=False,
        )


def test_canonical_bytes_cache_is_immutable_only_and_mapping_changes_invalidate(monkeypatch):
    measurement._canonical_immutable_bytes.cache_clear()
    value = "repeated journal label"
    scalar_dumps = 0
    large_int_dumps = 0
    mapping_dumps = 0
    original_dumps = json.dumps
    large_integer = 1 << 2048

    def counted_dumps(candidate, *args, **kwargs):
        nonlocal scalar_dumps, large_int_dumps, mapping_dumps
        if candidate is value:
            scalar_dumps += 1
        if candidate is large_integer:
            large_int_dumps += 1
        if isinstance(candidate, dict) and "value" in candidate:
            mapping_dumps += 1
        return original_dumps(candidate, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", counted_dumps)
    assert canonical_bytes(value) == b'"repeated journal label"'
    assert canonical_bytes(value) == b'"repeated journal label"'
    assert canonical_bytes(large_integer) == canonical_bytes(large_integer)
    assert canonical_bytes(0.0) == b"0.0"
    assert canonical_bytes(-0.0) == b"-0.0"
    mutable = {"value": 1}
    before = canonical_bytes(mutable)
    mutable["value"] = 2
    assert canonical_bytes(mutable) != before
    assert canonical_bytes(mutable) == b'{"value":2}'
    assert scalar_dumps == 1
    # The mapping serializes once before its first mutation and once for each
    # later value; mutable mappings never enter the immutable scalar cache.
    assert mapping_dumps == 3
    assert large_int_dumps == 2


def test_small_journal_normalization_keeps_tuple_and_reserved_key_semantics(tmp_path):
    value = {"nested": (1, {"items": (2, 3)})}
    assert externalize_journal_fields(
        value, tmp_path / "events.jsonl", threshold_bytes=1024,
    ) == {"nested": [1, {"items": [2, 3]}]}
    with pytest.raises(ValueError, match="reserved blob-reference key"):
        externalize_journal_fields(
            {"nested": {"$blob": "caller-value"}},
            tmp_path / "events.jsonl", threshold_bytes=1024,
        )


def test_cached_journal_suffix_extends_incrementally_and_returns_detached_rows(
        tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "suffix-cache", {"origin": "perf-test"})
    try:
        journal.emit("suffix_cache_probe", payload={"value": 1})
        cursor = journal._sequence - 1
        first = journal.cached_records_since(cursor)
        assert len(first) == 1 and first[0]["kind"] == "suffix_cache_probe"
        first[0]["payload"]["value"] = 99

        original_range = journal._record_range
        range_calls = 0

        def counted_range(start, stop, **kwargs):
            nonlocal range_calls
            range_calls += 1
            yield from original_range(start, stop, **kwargs)

        monkeypatch.setattr(journal, "_record_range", counted_range)
        repeated = journal.cached_records_since(cursor)
        assert repeated[0]["payload"]["value"] == 1
        assert range_calls == 0

        journal._clear_records_since_cache()
        invalidated = journal.cached_records_since(cursor)
        assert invalidated[0]["payload"]["value"] == 1
        assert range_calls == 1

        journal.emit("suffix_cache_append", payload={"value": 2})
        before = range_calls
        extended = journal.cached_records_since(cursor)
        assert [row["kind"] for row in extended] == [
            "suffix_cache_probe", "suffix_cache_append",
        ]
        assert range_calls == before
        borrowed = journal._borrow_records_since(cursor)
        assert borrowed[0] is journal._record_cache[cursor]
        assert borrowed[0]["payload"]["value"] == 1
    finally:
        journal.close()


def test_coverage_index_uses_read_only_journal_borrow():
    record = {"sequence": 1, "kind": "non_coverage_event"}
    journal = SimpleNamespace(
        run_id="coverage-perf",
        path="coverage-perf.jsonl",
        receipt_cursor=lambda: SimpleNamespace(sequence=1),
        _borrow_records_since=lambda sequence: (record,) if sequence == 0 else (),
        cached_records_since=lambda _sequence: pytest.fail(
            "coverage sync should use the journal-owned read-only view"
        ),
    )
    index = call_coverage.CoverageIndex(journal)
    index.sync()
    assert index.sequence == 1
    assert record == {"sequence": 1, "kind": "non_coverage_event"}


def test_countdown_projection_is_exact_and_skips_objects_without_markers(monkeypatch):
    import kernel_countdown_slot as countdown
    from kernel_request_diet import _tokenization_content

    ordinary = {"messages": [{"role": "user", "content": "plain"}], "tools": []}
    monkeypatch.setattr(
        countdown, "_canonical_countdown",
        lambda _value: pytest.fail("countdown projection should be skipped without a marker"),
    )
    assert countdown.canonical_countdown(ordinary) is ordinary
    monkeypatch.setattr(
        countdown, "canonical_countdown",
        lambda _value: pytest.fail("countdown walk should be skipped without a marker"),
    )
    assert _tokenization_content(ordinary) == prompt_projection(ordinary)

    marked = {
        "messages": [{
            "role": "user",
            "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] sandbox_seconds_remaining=12.0; pace=live",
        }],
        "tools": [],
    }
    monkeypatch.undo()
    from kernel_countdown_slot import canonical_countdown
    expected = prompt_projection(canonical_countdown(marked))
    dumps = 0
    original_dumps = request_diet.json.dumps

    def counted_dumps(*args, **kwargs):
        nonlocal dumps
        dumps += 1
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(request_diet.json, "dumps", counted_dumps)
    assert _tokenization_content(marked) == expected
    assert dumps == 1


def test_list_token_validation_accepts_only_nonnegative_exact_ints():
    meter = RequestTokenMeter(tokenizer=lambda _text: [0, 1, 2])
    assert meter._count_tokens_uncached("valid") == 3
    for invalid in ([True], [-1], [1.0], ["1"], [1, False]):
        meter = RequestTokenMeter(tokenizer=lambda _text, values=invalid: values)
        with pytest.raises(ValueError, match="invalid injected tokenizer response"):
            meter._count_tokens_uncached("invalid")


def test_finalize_frame_candidate_has_bounded_copies_and_body_serializations(
        monkeypatch, tmp_path):
    import copy as copy_module
    import kernel_code_bookmarks as bookmarks
    import kernel_completion_owner as completion_owner
    import kernel_execution_reply_budget as reply_budget
    import kernel_frame_allocation_capture as allocation_capture
    import kernel_frame_optional_allocation as optional_allocation
    import kernel_stateframe_capacity as capacity
    import kernel_stateframe_framefit5 as framefit5
    import kernel_step_contract as step_contract
    import kernel_request_source_window as source_window

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    bookmark = {
        "path": "pkg/example.py", "symbol": "target", "revision": "sha256:rev",
        "rank": 1, "start_line": 1, "end_line": 1,
    }
    state = {
        "framefit5": {
            "mode": framefit5.PRODUCT_MODE,
            "action_ledger": {},
            "code_bookmarks": [bookmark],
            "read_facts": [{
                "path": bookmark["path"], "current_revision": bookmark["revision"],
                "body_delivery": "present",
            }],
        },
    }
    task = {"role": "user", "content": "Keep the request stable."}
    body = {
        "messages": [task, {
            "role": "user", "content": framefit5._serialize_state_message({"state": state}),
        }],
        "tools": [],
    }
    plan = {"units": [], "paths": [], "deferred_paths": []}
    runtime = SimpleNamespace(
        frame_ceiling=8192, owner=SimpleNamespace(root=tmp_path), _code_bookmark_carry=(),
    )
    capture = SimpleNamespace(
        request_id="perf-finalize-copy-budget", _framefit5_semantic_messages=[task],
        _framefit5_task_message=task, _request_source_closure_override=plan,
    )
    meter = RequestTokenMeter(tokenizer=lambda _text: [1], template_overhead_tokens=0)
    meter.bind_request(capture.request_id, capture.request_id)

    import kernel_request_budget as request_budget
    import kernel_frame_source_snapshot as source_snapshot

    monkeypatch.setattr(request_budget, "_request_token_meter", lambda *_a, **_kw: meter)
    monkeypatch.setattr(request_budget, "_lean_request_limits",
                        lambda *_a, **_kw: (1_000_000, 100_000))
    monkeypatch.setattr(request_budget, "_graceful_trim_optional_messages",
                        lambda *_a, **_kw: None)
    monkeypatch.setattr(completion_owner, "_current_tool_admission",
                        lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(reply_budget, "fit_execution_replies", lambda *_a, **_kw: None)
    monkeypatch.setattr(reply_budget, "project_execution_delivery",
                        lambda semantic, *_a, **_kw: semantic)
    monkeypatch.setattr(capacity, "fit_protected_frame",
                        lambda *_a, **_kw: {"changed": False})
    monkeypatch.setattr(capacity, "fit_protected_source_overhead", lambda *_a, **_kw: None)
    monkeypatch.setattr(capacity, "within_protected_capacity", lambda *_a, **_kw: True)
    monkeypatch.setattr(optional_allocation, "snapshot_optional_inputs",
                        lambda *_a, **_kw: ({}, {}))
    monkeypatch.setattr(framefit5, "_workspace_root", lambda *_a: tmp_path)
    monkeypatch.setattr(source_snapshot, "recent_editable_read_windows", lambda *_a: [])
    monkeypatch.setattr(source_snapshot, "public_check_source_windows", lambda *_a: [])
    monkeypatch.setattr(source_snapshot, "editable_product_windows", lambda *_a: ([], []))
    monkeypatch.setattr(source_snapshot, "frame_source_snapshot", lambda *_a: {})
    monkeypatch.setattr(bookmarks, "resolve", lambda _root, row: {"revision": row["revision"]})
    monkeypatch.setattr(bookmarks, "windows", lambda *_a: [])
    monkeypatch.setattr(bookmarks, "enabled", lambda: False)
    monkeypatch.setattr(source_window, "verify_final_source_delivery",
                        lambda *_a, **_kw: {"status": "COMPLETE", "paths": {}})
    monkeypatch.setattr(source_window, "source_delivery_satisfied", lambda _receipt: True)
    monkeypatch.setattr(step_contract, "reconcile_final_source_access",
                        lambda *_a, **_kw: False)

    deepcopy_calls = 0
    deepcopy_depth = 0
    candidate = None
    original_deepcopy = copy_module.deepcopy

    def counted_deepcopy(value, memo=None):
        nonlocal deepcopy_calls, deepcopy_depth, candidate
        if deepcopy_depth == 0:
            deepcopy_calls += 1
        deepcopy_depth += 1
        try:
            cloned = (original_deepcopy(value) if memo is None
                      else original_deepcopy(value, memo))
        finally:
            deepcopy_depth -= 1
        if value is body:
            candidate = cloned
        return cloned

    candidate_serializations = 0
    original_dumps = json.dumps

    def counted_dumps(value, *args, **kwargs):
        nonlocal candidate_serializations
        if value is candidate:
            candidate_serializations += 1
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(copy_module, "deepcopy", counted_deepcopy)
    monkeypatch.setattr(json, "dumps", counted_dumps)
    result = allocation_capture.finalize_frame_candidate(
        runtime, capture, body, frame_counter=lambda _value: 1,
    )

    assert result is candidate
    assert capture._code_bookmark_transition[0] is True
    assert capture._code_bookmark_transition[1][0]["path"] == bookmark["path"]
    # Four top-level clones remain: caller-body isolation, incoming-plan
    # isolation, the independently retained receipt, and the published closure
    # snapshot. Bookmark/source snapshots and inactive optional inputs are shared.
    # The unchanged body is serialized exactly eight times: meter wire size,
    # measured-pair chars, frozen chars+digest, one source-proof payload,
    # one frozen identity check, and final carried chars+digest. Reuse of the
    # prepared source-proof wire covers all later size checks and measurement bind.
    # Candidate counts are exact on this fixture; the combined assertion also
    # exposes both pass-1 baseline values in the planted control run.
    assert (deepcopy_calls, candidate_serializations) == (4, 8)


def test_source_plan_snapshot_detaches_metadata_and_shares_text_snapshot():
    import kernel_frame_allocation_capture as allocation_capture

    source_contents = {"pkg/example.py": "VALUE = 1\n"}
    plan = {
        "deferred_paths": ["pkg/example.py"],
        "units": [{"path": "pkg/example.py", "start_line": 1, "end_line": 1}],
        "source_contents": source_contents,
    }

    snapshot = allocation_capture._copy_source_plan_snapshot(plan)

    assert snapshot is not plan
    assert snapshot["deferred_paths"] is not plan["deferred_paths"]
    assert snapshot["units"] is not plan["units"]
    assert snapshot["source_contents"] is source_contents
    snapshot["deferred_paths"].append("pkg/other.py")
    assert plan["deferred_paths"] == ["pkg/example.py"]


def test_route_probe_snapshot_isolates_only_delivery_mutation_maps():
    import kernel_frame_allocation_capture as allocation_capture

    delivery = {"pkg/previous.py": {"status": "COMPLETE"}}
    coverage = {
        "delivered_span": {"status": "PARTIAL", "delivered": 0},
        "acquired_span": {"status": "UNKNOWN", "acquired": 0},
        "unchanged": {"value": 1},
    }
    plan = {
        "units": [{"path": "pkg/example.py", "start_line": 1, "end_line": 1}],
        "source_contents": {"pkg/example.py": "VALUE = 1\n"},
        "delivery": delivery,
        "coverage": coverage,
    }

    route_plan = allocation_capture._copy_source_delivery_route_plan(plan)

    assert route_plan is not plan
    assert route_plan["units"] is plan["units"]
    assert route_plan["source_contents"] is plan["source_contents"]
    assert route_plan["delivery"] is not delivery
    assert route_plan["delivery"]["pkg/previous.py"] is delivery["pkg/previous.py"]
    assert route_plan["coverage"] is not coverage
    assert route_plan["coverage"]["delivered_span"] is not coverage["delivered_span"]
    assert route_plan["coverage"]["acquired_span"] is not coverage["acquired_span"]
    assert route_plan["coverage"]["unchanged"] is coverage["unchanged"]

    route_plan["delivery"]["pkg/example.py"] = {"status": "PARTIAL"}
    route_plan["coverage"]["delivered_span"]["delivered"] = 1
    route_plan["coverage"]["acquired_span"]["acquired"] = 1
    assert "pkg/example.py" not in delivery
    assert coverage["delivered_span"]["delivered"] == 0
    assert coverage["acquired_span"]["acquired"] == 0
