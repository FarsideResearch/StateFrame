"""Slow /tokenize replies must not invalidate the final source proof."""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pytest

from tests import test_allocation_replay_corpus as corpus
from tests.tape_server import TapeServer


SLOW_CASE_IDS = (
    "legacy-guard22",
    *(case_id for case_id in corpus.CASE_IDS
      if case_id.startswith(("factory-a-econ-", "factory-b-econ-"))),
)
SLOW_TOKENIZE_CALLS = (8, 9, 10)
TOKENIZE_TIMEOUT_SECONDS = 0.05
TOKENIZE_DELAY_SECONDS = 0.08


def test_tape_server_delays_only_configured_tokenize_calls():
    script = {
        "turns": [{"message": {"role": "assistant", "content": "done"}}],
        "tokenize_delay_s": 0.01,
        "tokenize_slow_calls": [2],
    }
    with TapeServer(script) as endpoint:
        for _call in range(3):
            request = Request(
                f"{endpoint.url}/tokenize",
                data=json.dumps({"content": "small request"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=2) as response:
                assert json.loads(response.read())["tokens"]

    assert endpoint.tokenize_request_count == 3
    assert endpoint.delayed_tokenize_calls == (2,)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("case_id", SLOW_CASE_IDS, ids=SLOW_CASE_IDS)
def test_slow_tokenize_fallback_preserves_final_allocation(tmp_path, case_id):
    outcome = corpus._run_case(
        tmp_path,
        case_id,
        tokenize_delay_s=TOKENIZE_DELAY_SECONDS,
        tokenize_slow_calls=SLOW_TOKENIZE_CALLS,
        tokenize_timeout_s=TOKENIZE_TIMEOUT_SECONDS,
    )

    assert outcome.delayed_tokenize_calls == SLOW_TOKENIZE_CALLS, (
        f"{case_id}: replay did not reach all selected slow /tokenize calls; "
        f"calls={outcome.delayed_tokenize_calls}, total={outcome.tokenize_request_count}"
    )
    allocation_errors = [
        error for error in corpus.ALLOCATION_ERRORS if error in outcome.evidence
    ]
    assert not allocation_errors, (
        f"{case_id}: allocation errors={allocation_errors}; "
        f"result={outcome.engine_result!r}; exit={outcome.returncode}; "
        f"endpoint requests={outcome.endpoint_requests}; "
        f"elapsed={outcome.elapsed_seconds:.2f}s\n{outcome.engine_log[-6000:]}\n"
        f"{outcome.stdout[-2000:]}\n{outcome.stderr[-2000:]}"
    )
    assert outcome.engine_result == "SUBMITTED", (
        f"{case_id}: result={outcome.engine_result!r}; exit={outcome.returncode}; "
        f"elapsed={outcome.elapsed_seconds:.2f}s\n{outcome.engine_log[-6000:]}"
    )

    warning_events = []
    for line in outcome.engine_log.splitlines():
        if line.startswith("[sf] "):
            try:
                warning_events.append(json.loads(line[len("[sf] "):]))
            except json.JSONDecodeError:
                continue
    fallback_warning = next(
        (event for event in warning_events
         if event.get("kind") == "token_counting_unavailable"),
        None,
    )
    assert fallback_warning is not None, f"{case_id}: fallback warning was not logged"
    assert fallback_warning.get("reason") == "timeout", (
        f"{case_id}: fallback warning omitted its reason: {fallback_warning!r}"
    )
    assert any(
        event.get("kind") == "request_diet_applied"
        and event.get("measure_method") == "calibrated_bytes"
        and event.get("tokenize_availability") == "unavailable"
        and event.get("tokenize_failure_reason") == "timeout"
        for event in outcome.measurement_events
    ), f"{case_id}: fallback reason was not recorded in measurement events"
