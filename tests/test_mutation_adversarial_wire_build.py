"""Adversarial tests for mutation framing, validation and hard budgets."""
from __future__ import annotations

import json
import sys

import pytest

from mutate_leftovers import mutation_root, stage_test_unit, sweep_and_assert


def _stdout_frame(body: dict) -> bytes:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"SFMW1 " + str(len(payload)).encode("ascii") + b"\n" + payload


def _valid_manifest(source: str, *, output_schema: dict | None = None) -> dict:
    return {
        "schema": "stateframe-mutation-unit/1", "name": "contract_probe", "revision": 1,
        "parent": None, "entry": "main.py", "summary": "Contract probe", "kind": "reader",
        "input_schema": {"type": "object"},
        "output_schema": output_schema or {"type": "object"},
        "requires": {"network": False, "writes": False}, "smoke": [],
        "limits": {"wall_s": 2, "cpu_s": 1, "mem_mb": 128},
    }


def test_mutation_strict_json_input_output_and_terminal_eof():
    from kernel_mutate_wire import WireError, build_stdin_doc, parse_stdout, strict_loads

    call = "call-1"
    frame = _stdout_frame({"wire": 1, "call": call, "kind": "result", "result": {"ok": True}})
    parsed = parse_stdout(frame, call=call)
    assert parsed.kind == "result"
    assert parsed.result == {"ok": True}
    assert strict_loads(b'{"items":[1,true,null]}') == {"items": [1, True, None]}

    for malformed in (
        b"prefix" + frame,
        frame + b"trailer",
        frame + frame,
        b"SFMW1 9\n{}",
        b"SFMW1 2\n\xff\xfe",
    ):
        with pytest.raises(WireError):
            parse_stdout(malformed, call=call)

    with pytest.raises(WireError):
        build_stdin_doc(wire=1, call=call, launch="a" * 32, phase=0, role="payload",
                        input={}, inputs=[], responses=[], grants={"network": False,
                        "writes": False}, max_fetch=4, ambient="extra")


def test_mutation_wire_aggregate_and_state_machine_limits():
    from kernel_mutate_wire import WireError, build_stdin_doc, parse_stdout

    call = "aggregate"
    too_many_fetches = _stdout_frame({
        "wire": 1, "call": call, "kind": "need", "fetch": [
            {"method": "GET", "url": f"https://example.com/{n}", "accept": "text/plain"}
            for n in range(5)
        ],
    })
    with pytest.raises(WireError):
        parse_stdout(too_many_fetches, call=call)
    with pytest.raises(WireError):
        parse_stdout(_stdout_frame({"wire": 1, "call": call, "kind": "result",
                                    "result": {"x": "y" * 2048}}), call=call, max_body=64)

    responses = [{"key": f"sha256:{n:064x}", "method": "GET",
                  "url": f"https://example.com/{n}", "accept": "text/plain", "status": 200,
                  "headers": {}, "final_url": f"https://example.com/{n}", "body": f"/resp/{n}",
                  "bytes": 0, "error": None} for n in range(5)]
    with pytest.raises(WireError):
        build_stdin_doc(wire=1, call=call, launch="b" * 32, phase=0, role="payload",
                        input={}, inputs=[], responses=responses,
                        grants={"network": False, "writes": False}, max_fetch=4)


@pytest.mark.parametrize("raw", [
    b'{"x":1,"x":2}',
    b'{"outer":{"x":1,"x":2}}',
    b'{"x":NaN}',
    b'{"x":Infinity}',
    b'{"x":1e999}',
    b'{"x":"\\ud800"}',
    b'{"x":12345678901234567890123456789012345678901}',
    (b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}"),
])
def test_mutation_wire_adversarial_json(raw):
    from kernel_mutate_wire import WireError, strict_loads

    with pytest.raises((WireError, ValueError, RecursionError)):
        strict_loads(raw)


def test_mutation_repair_budget_and_nonconvergent_cycle():
    from kernel_mutate_policy import Budget

    first_attempt = Budget().fold([])
    assert first_attempt.reserve("revisions_per_unit", 1, scope="unit")
    assert first_attempt.reserve("revisions_per_unit", 1, scope="unit")
    # Re-folding persisted charges must not reset the unit revision allowance.
    replayed = Budget().fold([{"kind": "BUDGET_RESERVE", "name": "revisions_per_unit",
                               "amount": 2, "scope": "unit"}])
    assert replayed.remaining(scope="unit")["revisions_per_unit"] == first_attempt.remaining(scope="unit")["revisions_per_unit"]
    while replayed.reserve("revisions_per_unit", 1, scope="unit"):
        pass
    assert not replayed.reserve("revisions_per_unit", 1, scope="unit")


def test_mutation_build_first_use_aggregate_wall_cpu_token_budget():
    from kernel_mutate_policy import Budget

    state = Budget().fold([])
    initial = state.remaining()
    assert initial["invocations"] == 40
    assert state.reserve("invocations", 40)
    assert not state.reserve("invocations", 1)
    assert state.remaining()["invocations"] == 0

    # DESIGN §2.1 defines byte and runtime budgets, not a model-token budget.
    for name, limit in (("payload_cpu_s", 60), ("payload_wall_s", 120),
                        ("authored_source_bytes", 256 << 10), ("output_bytes", 64 << 10)):
        budget = Budget().fold([])
        assert budget.remaining()[name] == limit
        assert budget.reserve(name, limit)
        assert not budget.reserve(name, 1)
        replayed = Budget().fold([{"kind": "BUDGET_RESERVE", "name": name, "amount": limit}])
        assert not replayed.reserve(name, 1)


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate validator payloads",
)
def test_mutation_validator_smoke_import_and_obfuscated_code_isolation(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution

    marker = tmp_path / "validator-imported-unit"
    source = (
        "import builtins\n"
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "def handle(input, ctx):\n"
        "    return eval(compile('{' + repr('ok') + ': True}', '<unit>', 'eval'))\n"
    )
    staged = stage_test_unit(build, mutation_root, source,
                             limits={"wall_s": 10, "cpu_s": 5, "mem_mb": 256})
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    try:
        receipt = build.validate(staged, executor)
        assert receipt.ok
        assert not marker.exists(), "validator imported authored source instead of compiling it"
    finally:
        cleanup = executor.close()
    assert cleanup.ok
    sweep_and_assert(mutation_root)


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap mounts the unit at /sf/unit for repair diagnostics",
)
def test_mutation_mike_syntax_error_is_repairable_not_executed(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution

    marker = tmp_path / "syntax-unit-ran"
    source = f"open({str(marker)!r}, 'w').write('bad')\ndef handle(input, ctx):\n return {{\n"
    staged = stage_test_unit(build, mutation_root, source,
                             limits={"wall_s": 10, "cpu_s": 5, "mem_mb": 256})
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    try:
        receipt = build.validate(staged, executor)
        assert not receipt.ok
        assert receipt.result["errors"]
        assert not marker.exists()
    finally:
        cleanup = executor.close()
        assert cleanup.ok
        sweep_and_assert(mutation_root)


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp execute mutation contract checks",
)
def test_mutation_mike_wrong_contract_is_rejected(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = "def handle(input, ctx):\n    return {'count': True}\n"
    staged = stage_test_unit(build, mutation_root, source,
                             output_schema={"type": "object", "properties": {
                                 "count": {"type": "integer"}}, "required": ["count"],
                                 "additionalProperties": False},
                             limits={"wall_s": 10, "cpu_s": 5, "mem_mb": 256})
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    try:
        with pytest.raises(wire.WireError, match="SCHEMA_TYPE"):
            build.smoke(staged, executor, [{"input": {}, "inputs": {},
                                            "expect": {"count": 1}}])
    finally:
        cleanup = executor.close()
        assert cleanup.ok
        sweep_and_assert(mutation_root)
