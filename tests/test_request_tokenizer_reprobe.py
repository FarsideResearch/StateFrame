"""Tokenizer recovery is bounded and preserves the healthy request path."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic

import pytest

from _stateframe_home import isolated_stateframe_home
from _journal import measurement_events
from engine_run import run_monitored
import kernel_agent_schemas
import kernel_provider_dialect as provider_dialect
import kernel_request_diet as request_diet
from kernel_shell_tool import SHELL_SCHEMA
from tests import test_allocation_replay_corpus as corpus
from tests.tape_server import TapeServer
from tools.immediate_baseline import baseline_checkout  # noqa: F401 (pytest fixture)
from tools.compare_no_mcp_requests import mask_wire_ids
from tools.sandbox_temp import sandbox_visible_temp_dir, timeout_scale


TOKENIZE_TIMEOUT_SECONDS = 0.05
TOKENIZE_DELAY_SECONDS = 0.08
# Matches the fixed-width slot from kernel_countdown_slot.py and older rounded-minute slots.
_CLEAN_METER_COUNTDOWN = re.compile(
    rb"sandbox_seconds_remaining=\d+\.\d \(about (?:\d+|999\+) minutes left"
    rb"(?:, ~(?:\d+|999\+|----) turns at your pace)?\)"
)
_CLEAN_METER_COUNTDOWN_MASK = b"sandbox_seconds_remaining=<masked>"

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


def test_cold_tokenize_probe_reserves_post_budget_and_skips_slots(monkeypatch):
    monkeypatch.delenv(provider_dialect.PROVIDER_OVERRIDE_ENV, raising=False)
    cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", cache)
    probes = []

    def probe(_origin, _key, path, *, deadline):
        probes.append((path, deadline))
        return 404, None

    monkeypatch.setattr(provider_dialect, "bounded_json_probe", probe)
    started = monotonic()
    with pytest.raises(provider_dialect.ProviderCapabilityError):
        request_diet._http_tokenize(
            "unproven endpoint", "http://127.0.0.1:9/tokenize", deadline=started + 0.2,
        )

    assert len(probes) == 1
    assert probes[0][0] == "/props"
    assert started < probes[0][1] < started + 0.15
    assert cache == {}


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
        request_budget = 4.0 * timeout_scale()
        first_tokens = request_diet._http_tokenize(
            "props capability", f"{endpoint.url}/tokenize", deadline=monotonic() + request_budget,
        )
        probe_misses_enabled = True
        slot_miss = provider_dialect.capabilities_for(endpoint.url, "", require_slots=True)
        second_tokens = request_diet._http_tokenize(
            "cached props capability after slot miss",
            f"{endpoint.url}/tokenize", deadline=monotonic() + request_budget,
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


def test_cold_props_proof_leaves_time_for_late_tokenize_and_is_cached(monkeypatch):
    monkeypatch.delenv(provider_dialect.PROVIDER_OVERRIDE_ENV, raising=False)
    cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", cache)
    probe_paths = []
    real_probe = provider_dialect.bounded_json_probe

    def record_probe(origin, key, path, **kwargs):
        probe_paths.append(path)
        return real_probe(origin, key, path, **kwargs)

    monkeypatch.setattr(provider_dialect, "bounded_json_probe", record_probe)
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
        # The cold /props probe starts a Python child: allow its full 2s cap
        # under loaded CI, then 2s for POST scheduling and its 30ms response.
        meter.bind_request("request-1", "request-1", deadline=monotonic() + 4.0)
        _count, method = meter.measure(_body("cold props proof and late tokenize"))

    assert method == "tokenize"
    assert endpoint.tokenize_request_count == 1
    assert probe_paths == ["/props"]
    assert cache[provider_dialect.origin_of(endpoint.url)].reason == "proven by /props"


def test_late_tokenize_success_within_request_budget_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", 2.0)
    monkeypatch.setattr(request_diet, "TOKENIZE_QUEUEING_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(request_diet, "TOKENIZE_RELAY_RTT_SECONDS", 0.0)
    monkeypatch.setattr(request_diet, "TOKENIZE_RELAY_LOAD_MARGIN", 0.0)
    # Allow 2s for the cold /props child and about 2s for POST on a loaded host.
    # Its 2.5s delay forces a miss; 30ms tokenization is late versus the 5ms attempt timeout.
    monkeypatch.setenv(provider_dialect.PROVIDER_OVERRIDE_ENV, provider_dialect.LLAMA_CPP)
    capability_cache = {}
    monkeypatch.setattr(provider_dialect, "RUN_CAPABILITY_CACHE", capability_cache)
    script = {
        "model": "allocation-replay-model",
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "props_delay_s": 2.5,
        "tokenize_delay_s": 0.03,
        "tokenize_slow_calls": [1],
    }
    with TapeServer(script) as endpoint:
        meter = request_diet.RequestTokenMeter(
            endpoint=f"{endpoint.url}/tokenize", template_overhead_tokens=0,
        )
        meter.bind_request("request-1", "request-1", deadline=monotonic() + 4.0)
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


def _run_compact_edit_replay(
        scratch: Path, *, probe_interval: int | None = None,
        request_limit: int | None = None, source_root: Path | None = None,
) -> corpus.ReplayOutcome:
    # The allocation corpus's first turn reads five full files. Under the current
    # conservative outage estimate its next mandatory body honestly exceeds the
    # prompt ceiling, so it cannot exercise reprobes or submission. Use the
    # shipped small edit script to keep the complete run inside that ceiling.
    root = Path(__file__).resolve().parents[1]
    repo = scratch / "repo"
    shutil.copytree(root / "examples" / "discount_bug", repo)
    steps = json.loads((root / "examples" / "discount_bug.script.json").read_text())[
        "steps"
    ]
    assert [step["tool"] for step in steps] == [
        "read", "str_replace", "run_tests", "done",
    ]
    if request_limit is not None:
        assert probe_interval is None
        steps = [
            {"tool": "read", "arguments": {"path": f"note-{index}.txt"}}
            for index in range(1, 7)
        ]
        for index in range(1, 7):
            (repo / f"note-{index}.txt").write_text(
                f"Reference note {index}.\n", encoding="utf-8",
            )
    else:
        assert probe_interval is not None
    turns = [{"message": {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": f"call_{index}", "type": "function",
            "function": {
                "name": step["tool"],
                "arguments": json.dumps(step["arguments"]),
            },
        }],
    }} for index, step in enumerate(steps, start=1)]
    script = {"model": "allocation-replay-model", "turns": turns}
    if probe_interval is not None:
        script.update({
            "tokenize_delay_s": TOKENIZE_DELAY_SECONDS,
            "tokenize_slow_calls": list(range(1, 512)),
        })
    source_root = source_root or corpus._source_root()
    stateframe_home = scratch / "stateframe-home"
    stateframe_home.mkdir()
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("STATEFRAME_")
    }
    environment.update({
        "STATEFRAME_HOME": str(stateframe_home),
        "PYTHONPATH": os.pathsep.join((
            str(source_root), str(source_root / "stateframe" / "engine"),
        )),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    })
    if request_limit is not None:
        environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = "off"
    run_dir = scratch / "run"
    engine_timeout = int(240 * timeout_scale())
    if request_limit is not None:
        wrapper = corpus._request_capped_engine_python(
            scratch, request_limit,
            extra_bootstrap=(
                f"import os\nos.environ[{provider_dialect.PROVIDER_OVERRIDE_ENV!r}] = "
                f"{provider_dialect.LLAMA_CPP!r}"
            ),
        )
    else:
        wrapper = corpus._short_tokenize_timeout_engine_python(
            scratch, TOKENIZE_TIMEOUT_SECONDS,
            reprobe_after_requests=probe_interval,
        )
    started = monotonic()
    with TapeServer(script) as endpoint:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
                "--task", "Fix the pricing code so the tests pass.",
                "--endpoint", endpoint.url, "--model", "allocation-replay-model",
                "--run-dir", str(run_dir), "--test-python", sys.executable,
                "--shell-read", str(source_root), "--slot", "1", "--max-turns", "20",
                "--timeout", str(engine_timeout), "--quiet", "--python", str(wrapper),
            ],
            expected_seconds=engine_timeout + 60,
            run_dir=run_dir, step="bounded tokenizer outage replay",
            env=environment, progress_probe=lambda: endpoint.request_count,
        )
        endpoint_requests = endpoint.request_count
        served_steps = endpoint.served_steps
        total_steps = endpoint.total_steps
        terminal_served = endpoint.terminal_served
        tokenize_request_count = endpoint.tokenize_request_count
        tokenize_request_log = endpoint.tokenize_request_log
        delayed_tokenize_calls = endpoint.delayed_tokenize_calls

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    engine_log_path = run_dir / "engine.log"
    engine_log = engine_log_path.read_text(errors="replace") if engine_log_path.is_file() else ""
    sends_path = run_dir / "sends.jsonl"
    sends = (
        tuple(json.loads(line) for line in sends_path.read_text().splitlines() if line.strip())
        if sends_path.is_file() else ()
    )
    return corpus.ReplayOutcome(
        case_id="discount-bug-compact-replay", returncode=process.returncode,
        engine_result=summary.get("engine_result") if isinstance(summary, dict) else None,
        summary=summary, engine_log=engine_log,
        stdout=process.stdout, stderr=process.stderr,
        elapsed_seconds=monotonic() - started,
        endpoint_requests=endpoint_requests, served_steps=served_steps,
        total_steps=total_steps, terminal_served=terminal_served,
        tokenize_request_count=tokenize_request_count,
        delayed_tokenize_calls=delayed_tokenize_calls,
        tokenize_request_log=tokenize_request_log,
        measurement_events=tuple(measurement_events(run_dir)),
        request_hashes=tuple(str(send["request_sha256"]) for send in sends),
        request_bytes=tuple(int(send["request_bytes"]) for send in sends),
    )


@pytest.mark.host_sandbox
def test_permanent_tokenizer_outage_replay_submits_with_bounded_probes():
    probe_interval = 2
    with sandbox_visible_temp_dir() as scratch:
        outcome = _run_compact_edit_replay(Path(scratch), probe_interval=probe_interval)

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


def _capture_clean_meter_request_bodies(
        tmp_path: Path, *, request_limit: int = 6,
        source_root: Path | None = None,
) -> tuple[bytes, ...]:
    outcome = _run_compact_edit_replay(
        tmp_path, request_limit=request_limit, source_root=source_root,
    )
    assert outcome.engine_result == "RUN_INCOMPLETE", (
        f"capture stop result={outcome.engine_result!r}; exit={outcome.returncode}; "
        f"requests={outcome.endpoint_requests}\n{outcome.evidence[-6000:]}"
    )
    assert outcome.returncode == 1, outcome.summary
    assert outcome.summary is not None
    assert outcome.summary.get("runtime_outcome") == "incomplete"
    assert '"kind":"token_counting_unavailable"' not in outcome.engine_log
    assert outcome.endpoint_requests == request_limit
    assert len(outcome.request_hashes) >= request_limit
    capture_marker = tmp_path / "run" / "capture-limit-reached"
    assert capture_marker.is_file(), "request-capped wrapper did not record its stop marker"
    assert capture_marker.read_text(encoding="ascii") == str(request_limit)
    request_bodies = tuple(
        (tmp_path / "run" / "wire" / f"{index:04d}.request.json").read_bytes()
        for index in range(1, request_limit + 1)
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


def _remove_proven_intake_addition(current_wire: bytes, baseline_wire: bytes) -> bytes:
    """Remove only the current release's intake block and its derived task digest."""
    current = json.loads(current_wire)
    baseline = json.loads(baseline_wire)
    current_task = current["messages"][1]["content"]
    baseline_task = baseline["messages"][1]["content"]
    marker = "[NAVBASICS intake orientation]"
    assert marker not in baseline_task
    assert current_task.count(marker) == 1
    insertion_start = current_task.index(marker)
    assert baseline_task.startswith(current_task[:insertion_start])
    suffix = baseline_task[insertion_start:]
    assert suffix and current_task.endswith(suffix)
    addition = current_task[insertion_start:len(current_task) - len(suffix)]
    assert addition.startswith(marker + "\nschema=navbasics-intake-orientation/1\n")
    assert "Repository root: ." in addition
    assert addition.endswith("\n\n")

    current_digest = hashlib.sha256(current_task.encode("utf-8")).hexdigest()
    baseline_digest = hashlib.sha256(baseline_task.encode("utf-8")).hexdigest()
    for body, expected_digest in ((current, current_digest), (baseline, baseline_digest)):
        frame_messages = [
            message["content"] for message in body["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith("[StateFrame]\n")
        ]
        assert len(frame_messages) == 1
        frame = json.loads(frame_messages[0].split("\n", 1)[1])
        assert frame["state"]["framefit5"]["task"]["sha256"] == expected_digest

    encoded_current = json.dumps(current_task).encode("utf-8")
    encoded_baseline = json.dumps(baseline_task).encode("utf-8")
    assert current_wire.count(encoded_current) == 1
    assert current_wire.count(current_digest.encode("ascii")) == 1
    return current_wire.replace(encoded_current, encoded_baseline, 1).replace(
        current_digest.encode("ascii"), baseline_digest.encode("ascii"), 1,
    )


@pytest.mark.host_sandbox
def test_healthy_measurement_preserves_baseline_request_bytes(baseline_checkout):
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

    # Compare against the immediate base release in this same environment (shared discovery:
    # SF_TEST_BASELINE_IMMEDIATE, else the merge base / nearest ancestor release), never a pin.
    baseline_tree = baseline_checkout
    with sandbox_visible_temp_dir() as scratch:
        scratch_root = Path(scratch)
        baseline_capture_root = scratch_root / "baseline-capture"
        baseline_capture_root.mkdir()
        baseline_bodies = _capture_clean_meter_request_bodies(
            baseline_capture_root,
            request_limit=2,
            source_root=baseline_tree / "src",
        )
        request_bodies = _capture_clean_meter_request_bodies(scratch_root)
    baseline_masked_hashes, baseline_copy_counts = _clean_meter_fingerprints(baseline_bodies)
    computed_masked_hashes, countdown_copy_counts = _clean_meter_fingerprints(request_bodies)
    # Every one of the six current requests has exactly one active-frame
    # countdown; only requests 1-2 have an immediate-base byte oracle.
    assert countdown_copy_counts[:2] == baseline_copy_counts[:2], (
        f"whole-request countdown-copy counts for requests 1-2: current={countdown_copy_counts!r}; "
        f"baseline={baseline_copy_counts!r}"
    )
    # Planted control: one extra standalone limits line in request 0001 must change its count.
    extra_line = re.search(
        rb"\[CONTINUATION LIMITS \| stateframe-continuation-limits/1\][^\\\n]*", request_bodies[0])
    assert extra_line is not None
    planted_body = request_bodies[0].replace(
        b'"messages": [', b'"messages": [{"role": "user", "content": "' + extra_line[0] + b'"}, ', 1)
    planted_count = len(_CLEAN_METER_COUNTDOWN.findall(planted_body))
    assert planted_count == countdown_copy_counts[0] + 1
    assert (planted_count, *countdown_copy_counts[1:2]) != baseline_copy_counts[:2]

    # Prove the whole-body normalizer still notices a byte outside IDs and countdowns.
    changed_body = bytearray(request_bodies[0])
    flip_at = _model_value_start(request_bodies[0])
    changed_body[flip_at] ^= 1
    assert _masked_wire_sha256(bytes(changed_body)) != computed_masked_hashes[0]

    # The current release now includes a repository intake block that the
    # immediate base silently omitted. Require that insertion and its task
    # digest to be the only byte differences in requests 1-2.
    assert computed_masked_hashes != baseline_masked_hashes
    normalized_bodies = tuple(
        _remove_proven_intake_addition(current, baseline)
        for current, baseline in zip(request_bodies, baseline_bodies)
    )
    normalized_hashes = tuple(_masked_wire_sha256(wire) for wire in normalized_bodies)
    assert normalized_hashes == baseline_masked_hashes, (
        f"masked request SHA-256s differ from the immediate base {baseline_tree} "
        "after the proven intake addition for requests 1-2: "
        f"current={normalized_hashes!r}; baseline={baseline_masked_hashes!r}; "
        f"whole-request countdown-copy counts={countdown_copy_counts!r}"
    )

    changed_current_hashes = (
        _masked_wire_sha256(_remove_proven_intake_addition(
            bytes(changed_body), baseline_bodies[0],
        )),
        *normalized_hashes[1:2],
    )
    assert changed_current_hashes != baseline_masked_hashes, (
        "baseline-relative masked comparison accepted a one-byte change in current request 0001"
    )


def test_reprobe_schedule_ignores_wall_time(monkeypatch):
    # A resumed run rebuilds earlier requests in moments, while the original
    # attempt waited on the model between them. Probe timing must depend only
    # on the request sequence, or the rebuilt request is measured differently.
    monkeypatch.setattr(request_diet, "TOKENIZE_REPROBE_AFTER_REQUESTS", 5)
    calls = 0

    def unavailable(_content: str) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("scripted tokenizer deadline")

    def run(gap_s: float) -> list[tuple[str, str | None]]:
        nonlocal calls
        calls = 0
        clock = [1_000.0]
        monkeypatch.setattr(request_diet.time, "monotonic", lambda: clock[0])
        meter = request_diet.RequestTokenMeter(tokenizer=unavailable, template_overhead_tokens=0)
        observed = []
        for index in range(1, 4):
            meter.begin_request(f"request-{index}")
            method = meter.measure(_body(f"request {index}"))[1]
            observed.append((method, meter.journal_fields(method).get("tokenize_reprobe_status")))
            clock[0] += gap_s
        return observed + [("calls", str(calls))]

    assert run(gap_s=0.0) == run(gap_s=600.0)
