"""Measure the common prefix of consecutive certified main requests."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from enginecheck import Scenario, Trace, run_pair
from enginecheck.wire import common_prefix_length, shipped_switches
from kernel_cache_layout import canonical_request_bytes
from kernel_working_set_allocator import FULL
from kernel_working_set_proof import verify_certificate
from kernel_working_set_render import render_assignment
from test_final_cache_layout import _body
from test_working_set_render import snapshot, source


SWITCH = "STATEFRAME_WORKING_SET_STABLE_FIRST"


def _source(prefix: str) -> bytes:
    return "".join(f"{prefix}_{i:03} = {i}  # deterministic values for source context\n"
                   for i in range(1, 161)).encode()


def _scenario() -> Scenario:
    # The release allocator reserves the edit scope before the first send.
    # Fit that mandatory envelope while measuring layout reuse in both arms.
    switches = {**shipped_switches(), SWITCH: "off", "STATEFRAME_CONTEXT_TOKENS": "131072"}
    return Scenario(
        files={"pricing.py": _source("price"), "shipping.py": _source("shipping")},
        task="Update pricing.py and shipping.py. In pricing.py set price_001 to 2. "
             "In shipping.py set shipping_001 to 3.",
        steps=[
            {"tool": "read", "required": True,
             "arguments": {"path": path, "start_line": start, "end_line": start + 9}}
            for path, start in [("pricing.py", 1), ("shipping.py", 1), ("pricing.py", 41)]
        ],
        switches=switches, max_turns=3, cli_args=("--parallel", "off"),
    )


def _prefix_shares(trace: Trace) -> list[float]:
    # Use the tools-first prompt envelope, so a schema change counts as an
    # early miss even when the HTTP body serializes its tools after messages.
    prompts = [canonical_request_bytes(body["messages"], body["tools"])
               for body in trace.request_json]
    return [common_prefix_length(before, after) / len(after)
            for before, after in zip(prompts, prompts[1:])]


def test_stable_first_moves_carriers_without_reordering_witnesses():
    read = source("read", "first\n")
    closure = replace(source("closure", "second\n", first=3, ordinal=2), plan_only=True)
    snap = snapshot(read, closure, body=_body(exchange=True), layouts=("stable_first",),
                    current_revisions=((read.identity.path, read.identity.revision),))
    candidate = render_assignment(snap, {read.key: FULL, closure.key: FULL}, "stable_first")
    read_witness, closure_witness = candidate.certificate.witnesses
    assert (read_witness.slot_id, closure_witness.slot_id) == (read.key, closure.key)
    assert closure_witness.message_index < read_witness.message_index
    messages = json.loads(candidate.wire)["messages"]
    assert messages[closure_witness.message_index]["content"] == closure.page
    assert messages[read_witness.message_index]["content"] == read.page
    assert verify_certificate(candidate.wire, candidate.certificate)


@pytest.mark.host_sandbox
def test_stable_first_reuses_main_request_prefix(tmp_path: Path):
    off, on = run_pair(_scenario(), tmp_path, {SWITCH: "on"})
    assert len(on.requests) == len(off.requests) >= 3
    off_shares, on_shares = _prefix_shares(off), _prefix_shares(on)
    assert min(on_shares) >= 0.60, (off_shares, on_shares)
    assert on_shares[0] > off_shares[0], (off_shares, on_shares)
    for trace in (off, on):
        prepared = {event["body_sha256"] for event in trace.events
                    if event["kind"] == "working_set_prepared" and event["proof_passed"]}
        assert all(sha256(raw).hexdigest() in prepared for raw in trace.requests)
    assert [body["tools"] for body in on.request_json] == [body["tools"] for body in off.request_json]
