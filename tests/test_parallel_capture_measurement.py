"""Child continuation limits are part of the body that request measurement seals."""

import json

from kernel_request_measurement import RequestMeasurement
from kernel_stateframe_envelope import serialize_stateframe_payload
from kernel_stateframe_framefit5 import carry_scoped_continuation_limits


def test_scoped_continuation_limit_must_precede_body_measurement():
    line = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] decisions=4"
    frame = {
        "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1"},
        "subframe_scope": {"schema": "stateframe-subframe-state-scope/1"},
    }
    body = {"messages": [
        {"role": "user", "content": serialize_stateframe_payload(
            {"schema": "stateframe-f4/1", "state": {"framefit5": frame}})},
        {"role": "user", "content": line},
    ], "tools": []}

    early = RequestMeasurement.bind(
        body, tokenizer_identity="fixture", request_id="r", logical_request_id="r",
        tokens=1, method="calibrated_bytes", template_overhead_tokens=0,
        max_prompt_tokens=33344, max_request_bytes=100000,
    )
    assert carry_scoped_continuation_limits(body["messages"])
    assert not early.matches(body, tokenizer_identity="fixture", request_id="r",
                             logical_request_id="r")

    sealed = RequestMeasurement.bind(
        body, tokenizer_identity="fixture", request_id="r", logical_request_id="r",
        tokens=1, method="calibrated_bytes", template_overhead_tokens=0,
        max_prompt_tokens=33344, max_request_bytes=100000,
    )
    assert sealed.matches(body, tokenizer_identity="fixture", request_id="r",
                          logical_request_id="r")
    assert json.loads(body["messages"][0]["content"].split("\n", 1)[1])[
        "state"]["framefit5"]["owner_protocol"]["continuation_limits"] == line
