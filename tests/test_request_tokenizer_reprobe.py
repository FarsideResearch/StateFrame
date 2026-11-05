"""Tokenizer recovery is bounded and preserves the healthy request path."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from time import monotonic

import pytest

from _stateframe_home import isolated_stateframe_home
import kernel_agent_schemas
import kernel_provider_dialect as provider_dialect
import kernel_request_diet as request_diet
from kernel_shell_tool import SHELL_SCHEMA
from tests import test_allocation_replay_corpus as corpus
from tests.tape_server import TapeServer
from tools.compare_no_mcp_requests import mask_wire_ids


TOKENIZE_TIMEOUT_SECONDS = 0.05
TOKENIZE_DELAY_SECONDS = 0.08
# Matches the fixed-width slot from kernel_countdown_slot.py and older rounded-minute slots.
_CLEAN_METER_COUNTDOWN = re.compile(
    rb"sandbox_seconds_remaining=\d+\.\d \(about (?:\d+|999\+) minutes left"
    rb"(?:, ~(?:\d+|999\+|----) turns at your pace)?\)"
)
_CLEAN_METER_COUNTDOWN_MASK = b"sandbox_seconds_remaining=<masked>"
# An external STATEFRAME_REPLAY_SOURCE records a different environment_revision, so it cannot derive these.
# Countdown masked because it is wall-clock; only requests 1-2 have stable masked bytes.
# Derived with tests/tools/derive_clean_meter_pins.py on an untouched 012e0b5 worktree (2 runs) and on the v0.3.31 RC
# (2 runs); all four agree. Wire-visible IDs (moved by deterministic IDs) and the countdown are masked, so these hashes
# prove every other request-1/2 byte is unchanged since 012e0b5.
CLEAN_METER_MASKED_HASHES_012E0B5 = (
    "64e811d7331b461bb7af6c571ef7eb757707670616db8e87171117cae0a2cb4e",
    "adbfc0415994d9ec3aa377abbc723541f622bd01992a4a535921859ba5f9ffb3",
)
CLEAN_METER_COUNTDOWN_COPIES_012E0B5 = (2, 2, 2, 1, 1, 2)

# Compact request specimen with the current StateFrame field and a near-match decoy.
_COUNTDOWN_MASK_CONTROL_BODY = (
    b'{"model":"allocation-replay-model","messages":[{"role":"user",'
    b'"content":"[StateFrame]\\n{\\"state\\":{},\\"meter\\":\\"'
    b'sandbox_seconds_remaining=120.0 '
    b'(about 2 minutes left, ~4 turns at your pace) '
    b'sandbox_seconds_remaining=120 (about 2 minutes left, ~4 turns at your pace)\\"}"}],'
    b'"tools":[]}'
)


def _masked_wire_sha256(wire_body: bytes) -> str:
    masked = _CLEAN_METER_COUNTDOWN.sub(
        _CLEAN_METER_COUNTDOWN_MASK, mask_wire_ids(wire_body),
    )
    return hashlib.sha256(masked).hexdigest()


def _model_value_start(wire_body: bytes) -> int:
    match = re.search(rb'"model":\s*"', wire_body)
    assert match is not None
    return match.end()


# The countdown may be echoed elsewhere in the wire body. Count the canonical active-frame field
# once per request, while the digest helper masks every formatted copy.
def _active_frame_countdowns(wire_body: bytes) -> list[bytes]:
    payload = json.loads(wire_body)
    if not isinstance(payload, dict):
        raise AssertionError("wire request must decode to an object")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise AssertionError("wire request must contain a messages list")
    active_frames = [
        content.encode("utf-8")
        for message in messages
        if isinstance(message, dict)
        and isinstance((content := message.get("content")), str)
        and content.startswith("[StateFrame]")
    ]
    if len(active_frames) != 1:
        raise AssertionError(f"expected one active StateFrame message, found {len(active_frames)}")
    return _CLEAN_METER_COUNTDOWN.findall(active_frames[0])


def test_countdown_mask_preserves_non_countdown_byte_changes():
    original_hash = _masked_wire_sha256(_COUNTDOWN_MASK_CONTROL_BODY)
    changed = bytearray(_COUNTDOWN_MASK_CONTROL_BODY)
    flip_at = _model_value_start(_COUNTDOWN_MASK_CONTROL_BODY)
    changed[flip_at] ^= 1
    assert _masked_wire_sha256(bytes(changed)) != original_hash


def test_wire_id_mask_preserves_non_id_non_countdown_byte_changes():
    original = _COUNTDOWN_MASK_CONTROL_BODY.replace(
        b'"tools":[]', b'"request_id":"' + b"a" * 20 + b'","tools":[]',
    )
    changed_id = original.replace(b"a" * 20, b"b" * 20)
    assert original != changed_id
    assert _masked_wire_sha256(original) == _masked_wire_sha256(changed_id)

    changed_model = bytearray(changed_id)
    changed_model[_model_value_start(changed_id)] ^= 1
    assert _masked_wire_sha256(bytes(changed_model)) != _masked_wire_sha256(original)


def test_mask_rejects_non_json_after_state_frame_marker():
    body = (
        b'{"model":"allocation-replay-model","messages":[{"role":"user",'
        b'"content":"[StateFrame]\\nnot-json"}],"tools":[]}'
    )
    with pytest.raises(ValueError, match=r"invalid JSON after embedded \[StateFrame\] marker"):
        _masked_wire_sha256(body)


def test_countdown_mask_matches_one_active_frame_occurrence():
    assert len(_active_frame_countdowns(_COUNTDOWN_MASK_CONTROL_BODY)) == 1


@pytest.mark.parametrize("slot", (
    b"00000.0 (about 0000 minutes left, ~---- turns at your pace)",
    b"00060.0 (about 0001 minutes left, ~0001 turns at your pace)",
    b"00060.0 (about 0001 minutes left, ~999+ turns at your pace)",
    b"99999.9 (about 999+ minutes left, ~---- turns at your pace)",
))
def test_countdown_mask_covers_fixed_width_slot(slot: bytes):
    field = b"sandbox_seconds_remaining=" + slot
    assert _CLEAN_METER_COUNTDOWN.sub(_CLEAN_METER_COUNTDOWN_MASK, field) == _CLEAN_METER_COUNTDOWN_MASK


def _body(label: str) -> dict[str, object]:
    return {
        "model": "allocation-replay-model",
        "messages": [{"role": "user", "content": label}],
        "tools": [],
    }


def _schema_meter_startup_digest() -> str:
    source_root = corpus._source_root()
    engine_root = source_root / "stateframe" / "engine"
    # Mirrors kernel_agent_schemas.measure_schemas: startup first tokenizes SHELL_SCHEMA alone.
    script = (
        "import hashlib\n"
        "from kernel_request_diet import _tokenization_content\n"
        "from kernel_shell_tool import SHELL_SCHEMA\n"
        "content = _tokenization_content({'messages': [], 'tools': [SHELL_SCHEMA]})\n"
        "print(hashlib.sha256(content.encode('utf-8')).hexdigest())\n"
    )
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("STATEFRAME_")
    }
    environment.update({
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "PYTHONPATH": os.pathsep.join((str(source_root), str(engine_root))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    })
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
    )
    digest = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AssertionError(f"invalid schema-meter digest output: {digest!r}")
    return digest


def test_slow_burst_recovers_once_with_recorded_event_sequence(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    monkeypatch.setattr(request_diet, "TOKENIZE_QUEUEING_TIMEOUT_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_REQUESTS", 2)
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_SECONDS", 3600.0)
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "tokenize_delay_s": TOKENIZE_DELAY_SECONDS,
        "tokenize_slow_calls": [1, 2, 3],
    }
    event_sequence = []
    with TapeServer(script) as endpoint:
        meter = request_diet.RequestTokenMeter(
            endpoint=f"{endpoint.url}/tokenize", template_overhead_tokens=0,
        )
        for request_id in ("request-1", "request-2", "request-3"):
            meter.bind_request(request_id, request_id, deadline=monotonic() + 3.0)
            _, method = meter.measure(_body(request_id))
            event_sequence.append({
                "kind": "request_diet_applied",
                "request_id": request_id,
                "measure_method": method,
                **meter.journal_fields(method),
            })

    assert endpoint.delayed_tokenize_calls == (1, 2, 3)
    assert endpoint.tokenize_request_count == 4
    assert tuple(entry["call"] for entry in endpoint.tokenize_request_log) == (1, 2, 3, 4)
    assert all(len(str(entry["content_sha256"])) == 64 for entry in endpoint.tokenize_request_log)
    assert [event["measure_method"] for event in event_sequence] == [
        "calibrated_bytes", "calibrated_bytes", "tokenize",
    ]
    assert [event["tokenize_availability"] for event in event_sequence] == [
        "unavailable", "unavailable", "available",
    ]
    assert [event.get("tokenize_reprobe_status") for event in event_sequence] == [
        "waiting", "waiting", "recovered",
    ]
    assert event_sequence[-1]["tokenize_reprobe_attempts"] == 1


def test_reprobe_is_limited_to_one_attempt_per_request(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_REQUESTS", 1)
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_SECONDS", 3600.0)
    calls = 0

    def unavailable(_content: str) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("scripted tokenizer deadline")

    meter = request_diet.RequestTokenMeter(tokenizer=unavailable, template_overhead_tokens=0)
    meter.begin_request("request-1")
    assert meter.measure(_body("first"))[1] == "calibrated_bytes"
    assert calls == request_diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES

    meter.begin_request("request-2")
    assert meter.measure(_body("probe"))[1] == "calibrated_bytes"
    assert calls == request_diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES + 1
    assert meter.measure(_body("same request, another fitting pass"))[1] == "calibrated_bytes"
    assert calls == request_diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES + 1
    fields = meter.journal_fields("calibrated_bytes")
    assert fields["tokenize_reprobe_status"] == "failed"
    assert fields["tokenize_reprobe_attempts"] == 1
    assert fields["tokenize_failure_reason"] == "timeout"
    assert meter.unavailable_reason() == "timeout"


def test_slow_capability_probe_obeys_request_deadline_without_caching(monkeypatch):
    budget = 0.2
    cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", cache)
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "props_delay_s": 0.5,
    }
    with TapeServer(script) as endpoint:
        started = monotonic()
        with pytest.raises(TimeoutError, match="tokenizer request deadline expired"):
            request_diet._http_tokenize(
                "slow capability probe", f"{endpoint.url}/tokenize",
                deadline=started + budget,
            )
        elapsed = monotonic() - started

    origin = provider_dialect.origin_of(endpoint.url)
    assert elapsed <= budget + 0.1
    assert origin not in cache
    assert endpoint.tokenize_request_count == 0


def test_props_proof_survives_a_later_probe_miss_and_slot_probe(monkeypatch):
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
    }
    cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", cache)
    real_probe = provider_dialect.bounded_json_probe
    probe_misses_enabled = False
    missed_paths = []

    def probe_with_later_misses(origin, key, path, **kwargs):
        if probe_misses_enabled:
            missed_paths.append(path)
            return None
        return real_probe(origin, key, path, **kwargs)

    monkeypatch.setattr(provider_dialect, "bounded_json_probe", probe_with_later_misses)

    with TapeServer(script) as endpoint:
        origin = provider_dialect.origin_of(endpoint.url)
        first_tokens = request_diet._http_tokenize(
            "props capability", f"{endpoint.url}/tokenize", deadline=monotonic() + 1.0,
        )
        probe_misses_enabled = True
        slot_miss = provider_dialect.capabilities_for(endpoint.url, "", require_slots=True)
        second_tokens = request_diet._http_tokenize(
            "cached props capability after slot miss",
            f"{endpoint.url}/tokenize", deadline=monotonic() + 1.0,
        )

        props_proof = cache[origin]
        assert props_proof.dialect == provider_dialect.LLAMA_CPP
        assert props_proof.reason == "proven by /props"
        assert props_proof.props_status == 200
        assert props_proof.slots is False
        assert props_proof.slots_status is None
        assert slot_miss.dialect == provider_dialect.LLAMA_CPP
        assert slot_miss.slots is False
        assert missed_paths == ["/slots"]
        assert first_tokens and second_tokens
        assert endpoint.tokenize_request_count == 2

        monkeypatch.setattr(provider_dialect, "bounded_json_probe", real_probe)
        slot_proof = provider_dialect.capabilities_for(endpoint.url, "", require_slots=True)

    assert slot_proof.dialect == provider_dialect.LLAMA_CPP
    assert slot_proof.slots is True
    assert slot_proof.slot_ids == frozenset(range(8))
    assert slot_proof.slots_status == 200
    assert cache[origin] == slot_proof


def test_late_tokenize_success_within_request_budget_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", 0.2)
    monkeypatch.setattr(request_diet, "TOKENIZE_QUEUEING_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(request_diet, "TOKENIZE_RELAY_RTT_SECONDS", 0.0)
    monkeypatch.setattr(request_diet, "TOKENIZE_RELAY_LOAD_MARGIN", 0.0)
    capability_cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", capability_cache)
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "tokenize_delay_s": 0.03,
        "tokenize_slow_calls": [1],
    }
    with TapeServer(script) as endpoint:
        meter = request_diet.RequestTokenMeter(
            endpoint=f"{endpoint.url}/tokenize", template_overhead_tokens=0,
        )
        meter.bind_request("request-1", "request-1", deadline=monotonic() + 0.2)
        _count, method = meter.measure(_body("late but within request budget"))

    assert endpoint.tokenize_request_count == 1
    assert endpoint.delayed_tokenize_calls == (1,)
    assert method == "tokenize"
    assert meter.journal_fields(method)["tokenize_consecutive_failures"] == 0
    assert provider_dialect.origin_of(endpoint.url) not in capability_cache


def test_tool_schema_meter_records_one_initial_retry_group(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    monkeypatch.setattr(request_diet, "TOKENIZE_QUEUEING_TIMEOUT_SECONDS", TOKENIZE_TIMEOUT_SECONDS)
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "tokenize_delay_s": TOKENIZE_DELAY_SECONDS,
        "tokenize_slow_calls": tuple(range(1, 32)),
    }
    with TapeServer(script) as endpoint:
        results = kernel_agent_schemas.measure_schemas(endpoint=f"{endpoint.url}/v1/chat/completions")
        request_log = endpoint.tokenize_request_log

    schema_content = request_diet._tokenization_content({"messages": [], "tools": [SHELL_SCHEMA]})
    schema_digest = hashlib.sha256(schema_content.encode("utf-8")).hexdigest()
    assert tuple(entry["call"] for entry in request_log) == (1, 2, 3)
    assert all(entry["content_sha256"] == schema_digest for entry in request_log)
    assert all(entry["delayed"] is True for entry in request_log)
    assert all(result["method"] == "calibrated_bytes" for result in results.values())


@pytest.mark.host_sandbox
def test_permanent_tokenizer_outage_replay_submits_with_bounded_probes(tmp_path):
    probe_interval = 2
    outcome = corpus._run_case(
        tmp_path,
        "factory-a-econ-flat-r1",
        tokenize_delay_s=TOKENIZE_DELAY_SECONDS,
        tokenize_slow_calls=tuple(range(1, 512)),
        tokenize_timeout_s=TOKENIZE_TIMEOUT_SECONDS,
        tokenize_reprobe_after_requests=probe_interval,
        tokenize_reprobe_after_s=3600.0,
    )

    assert outcome.engine_result == "SUBMITTED", (
        f"result={outcome.engine_result!r}; exit={outcome.returncode}; "
        f"requests={outcome.endpoint_requests}; tokenize calls={outcome.tokenize_request_count}\n"
        f"{outcome.engine_log[-3000:]}\nstdout:\n{outcome.stdout[-3000:]}\nstderr:\n{outcome.stderr[-3000:]}"
    )
    assert not any(error in outcome.evidence for error in corpus.ALLOCATION_ERRORS)
    warning = next(
        (json.loads(line[len("[sf] "):]) for line in outcome.engine_log.splitlines()
         if line.startswith("[sf] ")
         and '"kind":"token_counting_unavailable"' in line),
        None,
    )
    assert warning is not None and warning.get("reason") == "timeout"

    probe_attempts = max(
        (int(event.get("tokenize_reprobe_attempts", 0))
         for event in outcome.measurement_events),
        default=0,
    )
    probe_ids = {
        event.get("request_id")
        for event in outcome.measurement_events
        if event.get("tokenize_reprobe_status") == "failed"
        and isinstance(event.get("request_id"), str)
    }
    logical_request_ids = tuple(dict.fromkeys(
        event["request_id"]
        for event in outcome.measurement_events
        if event.get("kind") == "request_diet_applied"
        and isinstance(event.get("request_id"), str)
    ))
    request_positions = {request_id: index for index, request_id in enumerate(logical_request_ids)}
    probe_positions = sorted(request_positions[request_id] for request_id in probe_ids)
    probe_events = {}
    for event in outcome.measurement_events:
        request_id = event.get("request_id")
        if (event.get("tokenize_reprobe_status") == "failed"
                and isinstance(request_id, str)):
            probe_events.setdefault(request_id, event)
    ordered_probe_events = sorted(
        probe_events.values(), key=lambda event: int(event.get("tokenize_reprobe_attempts", 0)),
    )
    request_initial_attempts = max(
        (int(event.get("tokenizer_calls", 0))
         for event in outcome.measurement_events
         if event.get("request_id") == logical_request_ids[0]
         and event.get("kind") == "request_diet_applied"),
        default=0,
    )
    schema_digest = _schema_meter_startup_digest()
    # schema-meter startup tokenize calls exist on some bases (3 on 89de58c, 0 on rel330b); derive them from the log
    schema_calls = tuple(
        entry for entry in outcome.tokenize_request_log
        if entry.get("content_sha256") == schema_digest
    )
    schema_call_numbers = tuple(int(entry["call"]) for entry in schema_calls)
    request_initial_calls = outcome.tokenize_request_log[len(schema_calls):-probe_attempts]
    probe_calls = outcome.tokenize_request_log[-probe_attempts:]
    initial_request_event = next(
        event for event in outcome.measurement_events
        if event.get("kind") == "request_diet_applied"
        and event.get("request_id") == logical_request_ids[0]
        and int(event.get("tokenizer_calls", 0)) == request_initial_attempts
    )
    sequence = [
        *(f"{entry['call']}: schema-meter initial retry" for entry in schema_calls),
        *(f"{entry['call']}: request-meter initial retry" for entry in request_initial_calls),
        *(f"{entry['call']}: probe at step {event.get('step')}"
          for entry, event in zip(probe_calls, ordered_probe_events)),
    ]
    maximum_probes = max(0, (len(logical_request_ids) - 1) // probe_interval)
    assert probe_attempts > 0
    assert len(probe_ids) == probe_attempts
    assert len(probe_positions) == probe_attempts
    assert probe_positions[0] >= probe_interval
    assert all(
        later - earlier >= probe_interval
        for earlier, later in zip(probe_positions, probe_positions[1:])
    )
    assert probe_attempts <= maximum_probes
    assert schema_call_numbers == tuple(range(1, len(schema_calls) + 1))
    assert all(entry["delayed"] is True for entry in schema_calls)
    assert request_initial_attempts == request_diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES
    assert all(
        int(call["timestamp_ns"]) < int(initial_request_event["timestamp_ns"])
        for call in request_initial_calls
    )
    assert tuple(int(entry["call"]) for entry in outcome.tokenize_request_log) == tuple(
        range(1, outcome.tokenize_request_count + 1)
    )
    assert len(outcome.tokenize_request_log) == (
        len(schema_calls) + request_initial_attempts + probe_attempts
    ), f"recorded /tokenize call sequence: {sequence!r}; log={outcome.tokenize_request_log!r}"
    assert len(request_initial_calls) == request_initial_attempts
    assert tuple(int(entry["call"]) for entry in request_initial_calls) == tuple(
        range(len(schema_calls) + 1, len(schema_calls) + request_initial_attempts + 1)
    )
    assert len(probe_calls) == len(ordered_probe_events) == probe_attempts
    assert tuple(int(entry["call"]) for entry in probe_calls) == tuple(
        range(
            len(schema_calls) + request_initial_attempts + 1,
            len(schema_calls) + request_initial_attempts + probe_attempts + 1,
        )
    )
    assert all(
        int(call["timestamp_ns"]) < int(event["timestamp_ns"])
        for call, event in zip(probe_calls, ordered_probe_events)
    ), f"recorded /tokenize call sequence: {sequence!r}"
    assert outcome.tokenize_request_count <= (
        len(schema_calls) + request_diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES + maximum_probes
    )
    assert outcome.delayed_tokenize_calls == tuple(range(1, outcome.tokenize_request_count + 1))


def _capture_clean_meter_request_bodies(tmp_path):
    # v0.4.0 changes default request order; the 0.3.30 byte oracle uses plain order.
    # Stop at the needed request count instead of replaying all 41 tape turns.
    # Each clean-meter request costs ~71 s of engine-side prep on Linux (measured on e62db60:
    # 34 sends in 2422 s; journal attribution by crew-08: public_test_verdict_fact_normalized,
    # code_bookmark_activation_decision, continuation_transition), so the full replay ran past
    # the engine deadline before submitting. Current capture compares requests 1-6.
    outcome = corpus._run_case(
        tmp_path,
        "clean-meter-22file-r1",
        stop_after_provider_requests=6,
        final_cache_layout="off",
    )
    assert outcome.engine_result == "RUN_INCOMPLETE", (
        f"capture stop result={outcome.engine_result!r}; exit={outcome.returncode}; "
        f"requests={outcome.endpoint_requests}\n{outcome.evidence[-6000:]}"
    )
    assert outcome.returncode == 1, outcome.summary
    assert outcome.summary is not None
    assert outcome.summary.get("runtime_outcome") == "incomplete"
    assert '"kind":"token_counting_unavailable"' not in outcome.engine_log
    assert outcome.endpoint_requests == 6
    assert len(outcome.request_hashes) >= 6
    capture_marker = tmp_path / "run" / "capture-limit-reached"
    assert capture_marker.is_file(), "request-capped wrapper did not record its stop marker"
    assert capture_marker.read_text(encoding="ascii") == "6"
    request_bodies = tuple(
        (tmp_path / "run" / "wire" / f"{index:04d}.request.json").read_bytes()
        for index in range(1, 7)
    )
    return request_bodies


def _clean_meter_fingerprints(request_bodies):
    countdown_copy_counts = tuple(
        len(_CLEAN_METER_COUNTDOWN.findall(wire_body)) for wire_body in request_bodies
    )
    masked_hashes = []
    for index, wire_body in enumerate(request_bodies, start=1):
        countdowns = _active_frame_countdowns(wire_body)
        assert len(countdowns) == 1, (
            f"request {index} active-frame countdown matches={countdowns!r}; "
            f"whole-request countdown-copy counts={countdown_copy_counts!r}"
        )
        masked_hashes.append(_masked_wire_sha256(wire_body))
    return tuple(masked_hashes), countdown_copy_counts


@pytest.mark.host_sandbox
def test_healthy_measurement_preserves_baseline_request_bytes(tmp_path):
    body = _body("wire identity")
    expected = (
        b'{"model":"allocation-replay-model","messages":[{"role":"user",'
        b'"content":"wire identity"}],"tools":[]}'
    )
    expected_wire = (
        b'{"model": "allocation-replay-model", "messages": [{"role": "user", '
        b'"content": "wire identity"}], "tools": []}'
    )
    assert request_diet._wire_bytes(body) == len(expected_wire)
    assert json.dumps(body).encode("utf-8") == expected_wire
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert encoded == expected

    meter = request_diet.RequestTokenMeter(
        tokenizer=lambda _content: [1, 2, 3], template_overhead_tokens=0,
    )
    assert meter.measure(body)[1] == "tokenize"
    assert request_diet._wire_bytes(body) == len(expected_wire)
    assert json.dumps(body).encode("utf-8") == expected_wire
    assert json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8") == expected
    assert "tokenize_reprobe_status" not in meter.journal_fields("tokenize")

    request_bodies = _capture_clean_meter_request_bodies(tmp_path)
    computed_masked_hashes, countdown_copy_counts = _clean_meter_fingerprints(request_bodies)
    assert countdown_copy_counts == CLEAN_METER_COUNTDOWN_COPIES_012E0B5, (
        f"whole-request countdown-copy counts for requests 1-6: {countdown_copy_counts!r}"
    )

    # Prove the whole-body normalizer still notices a byte outside IDs and countdowns.
    changed_body = bytearray(request_bodies[0])
    flip_at = _model_value_start(request_bodies[0])
    changed_body[flip_at] ^= 1
    assert _masked_wire_sha256(bytes(changed_body)) != computed_masked_hashes[0]

    # Pin requests 1-2 only: a callsite census at step 12 fires depending on run timing
    # and changes request 3+ bytes (a determinism leak). Owner: sf-test-runners.
    # Finding: sf-work/lanes/findings/census_timing_1104.md.
    # Full masked identity is checked by ident_pair.sh.
    assert computed_masked_hashes[:2] == CLEAN_METER_MASKED_HASHES_012E0B5, (
        f"masked request SHA-256s for requests 1-2: {computed_masked_hashes[:2]!r}; "
        f"whole-request countdown-copy counts={countdown_copy_counts!r}"
    )
