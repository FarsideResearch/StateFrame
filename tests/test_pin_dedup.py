"""Opt-in evidence-pin deduplication keeps the authoritative copy intact."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_completion_owner import _wire_evidence_pin, _wire_pin_is_typed
from kernel_evidence_pin import EVIDENCE_END, EVIDENCE_START
from kernel_refusal_compact import append_evidence_pin, render_evidence_pin
from kernel_stateframe_final_message import (
    PAGE_LOG_FINAL_MESSAGE_NAME,
    compose_final_user_message,
)
from kernel_stateframe_page_log import _authority_content, project_request


def _runtime(requirements, *, refusal=None):
    return SimpleNamespace(
        requirements_ledger={"requirements": requirements, "ticket_handle": "ticket-7"},
        owner=SimpleNamespace(task_row={"msg": "original requirement is longer than the outgoing task"}),
        _evidencepin_refusal_state=refusal,
    )


def _requirements(*clauses):
    return [
        {
            "id": f"R{index}", "exact_clause": clause,
            "reference": f"ref-{index}", "source": f"source-{index}",
            "status": "OPEN", "observed": "not checked",
            "missing_category": "behavior",
        }
        for index, clause in enumerate(clauses, 1)
    ]


def _stateframe_message():
    payload = {"state": {"framefit5": {
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1"},
    }}}
    return {"role": "user", "content": "[StateFrame]\n" + json.dumps(payload)}


def _fixture_body(name):
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_owner(body):
    frame_row = next(
        row for row in body["messages"]
        if str(row.get("content", "")).startswith("[StateFrame]\n")
    )
    payload = json.loads(frame_row["content"].split("\n", 1)[1])
    owner = payload["state"]["framefit5"]["owner_protocol"]
    return frame_row, payload, owner


def _canonical_bytes(value):
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _authority(pin=None):
    fields = [] if pin is None else [{
        "path": ["state", "framefit5", "owner_protocol", "evidence_pin"],
        "value": pin,
    }]
    return _authority_content(fields)


def _compose(pin, authority, *, task="Task: satisfy the exact requirement"):
    body = {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": task},
        _stateframe_message(),
        {"role": "user", "content": "Latest note\n\n" + pin},
    ]}
    return compose_final_user_message(
        body, tail_start=3, additions=(authority,),
        final_message_name=PAGE_LOG_FINAL_MESSAGE_NAME,
    )


def _legacy_compose(body, *, tail_start, additions=(), greenfact_text="", final_message_name=None):
    messages = body["messages"]
    contents = [message["content"] for message in messages[tail_start:]]
    extra = list(additions)
    green_occurrences = sum(content.count(greenfact_text) for content in contents) if greenfact_text else 0
    greenfact = ""
    if green_occurrences:
        suffix_indexes = [index for index, content in enumerate(contents) if content.endswith(greenfact_text)]
        assert green_occurrences == 1 and len(suffix_indexes) == 1
        index = suffix_indexes[0]
        contents[index] = contents[index][:-len(greenfact_text)]
        greenfact = greenfact_text
    final_content = "\n\n".join([*contents, *extra]) + greenfact
    output = json.loads(json.dumps(body))
    final_message = {"role": "user", "content": final_content}
    if final_message_name is not None:
        final_message["name"] = final_message_name
    output["messages"] = [*messages[:tail_start], final_message]
    return output


def _pin_rows(pin):
    rows = []
    for line in pin.splitlines():
        if not line.startswith("- "):
            continue
        fields = dict(re.findall(
            r"(?:^| )(reference|source|status|observed|missing_evidence)=([^ ]+)", line,
        ))
        rows.append((line.split(" ", 2)[1], fields))
    return rows


def _pin_body(pin):
    assert pin.startswith(EVIDENCE_START) and pin.endswith(EVIDENCE_END)
    return pin[len(EVIDENCE_START):-len(EVIDENCE_END)]


def _pin_nonrow_lines(pin):
    return [line for line in _pin_body(pin).splitlines() if not line.startswith("- ")]


@pytest.mark.parametrize("case", [0, 1, 2])
def test_switch_off_keeps_composed_request_bytes(case, monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "off")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n[/EVIDENCEPIN]"
    authority = _authority(pin)
    body = {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: satisfy the exact requirement"},
        _stateframe_message(),
        {"role": "user", "content": "Latest note\n\n" + pin},
    ]}
    if case == 0:
        expected = _legacy_compose(
            body, tail_start=3, additions=(authority,),
            final_message_name=PAGE_LOG_FINAL_MESSAGE_NAME,
        )
        actual = compose_final_user_message(
            body, tail_start=3, additions=(authority,),
            final_message_name=PAGE_LOG_FINAL_MESSAGE_NAME,
        )
    elif case == 1:
        body["messages"].append({"role": "user", "content": "done\nGREEN"})
        expected = _legacy_compose(
            body, tail_start=3, additions=("extra",), greenfact_text="GREEN",
        )
        actual = compose_final_user_message(
            body, tail_start=3, additions=("extra",), greenfact_text="GREEN",
        )
    else:
        expected = _legacy_compose(body, tail_start=3)
        actual = compose_final_user_message(body, tail_start=3)
    encode = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    assert encode(actual) == encode(expected)


def test_exact_authority_copy_replaces_one_standalone_suffix(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n[/EVIDENCEPIN]"
    authority = _authority(pin)
    result = _compose(pin, authority)
    final = result["messages"][-1]

    assert final["name"] == PAGE_LOG_FINAL_MESSAGE_NAME
    assert final["content"].count(EVIDENCE_START) == 1
    assert pin not in final["content"]
    assert _wire_evidence_pin(result["messages"]) == (pin, 1)
    assert _wire_pin_is_typed(result["messages"], pin)


@pytest.mark.parametrize("authority_kind", ["different", "missing", "clipped"])
def test_nonidentical_or_unavailable_authority_keeps_standalone(authority_kind, monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n[/EVIDENCEPIN]"
    if authority_kind == "different":
        authority = _authority(pin.replace("PASS", "FAIL", 1))
    elif authority_kind == "missing":
        authority = _authority()
    elif authority_kind == "clipped":
        authority = _authority(pin[len(EVIDENCE_START):])
    result = _compose(pin, authority)
    assert result["messages"][-1]["content"].count(pin) == 1
    # As with the switch off, the typed authority value wins when it is well formed.
    expected = pin.replace("PASS", "FAIL", 1) if authority_kind == "different" else pin
    assert _wire_evidence_pin(result["messages"]) == (expected, 1)
    assert _wire_pin_is_typed(result["messages"], expected)


@pytest.mark.parametrize("quoted", ["overlay", "standalone"])
def test_quoted_file_text_does_not_replace_the_typed_pin(quoted, monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n[/EVIDENCEPIN]"
    other = pin.replace("PASS", "FAIL", 1)
    text = _authority(other) if quoted == "overlay" else other
    result = _compose(pin, _authority(pin), task="Task: fix src/example.py\n" + text)
    assert _wire_evidence_pin(result["messages"]) == (pin, 1)
    assert _wire_pin_is_typed(result["messages"], pin)
    assert not _wire_pin_is_typed(result["messages"], other)


def test_malformed_matching_authority_does_not_remove_raw_pin(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    malformed = EVIDENCE_START + "\ncheck_status=[EVIDENCEPIN]PASS\n" + EVIDENCE_END
    result = _compose(malformed, _authority(malformed))
    assert result["messages"][-1]["content"].count(malformed) == 1


def test_verbatim_elision_uses_the_outgoing_task_row(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    monkeypatch.setenv("STATEFRAME_FRAME_CONTINUITY", "off")
    clause = "retain the exact boundary across the complete outgoing request"
    runtime = _runtime(_requirements(clause))
    runtime.owner.task_row["msg"] = "Task: " + clause + " and more"
    captured = []
    import kernel_completion_owner
    monkeypatch.setattr(kernel_completion_owner, "record_owner_block",
                        lambda _runtime, _kind, value, **_kwargs: captured.append(value))
    messages, panel = append_evidence_pin(
        runtime,
        [{"role": "user", "content": "Task: " + clause}],
        [],
    )

    assert "verbatim=" in render_evidence_pin(runtime, [], task_text="Task: retain the boundary")
    assert "verbatim=" not in panel
    assert captured == [panel]
    assert messages[-1]["content"].endswith(panel)


def test_clause_matching_uses_the_task_after_wire_cleanup(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    monkeypatch.setenv("STATEFRAME_FRAME_CONTINUITY", "off")
    clause = (
        "Document [EVIDENCEPIN]audit log must remain intact[/EVIDENCEPIN] "
        "as the sample payload."
    )
    runtime = SimpleNamespace(requirements_ledger={"requirements": [{
        "id": "R1", "exact_clause": clause,
    }]})
    import kernel_completion_owner
    monkeypatch.setattr(
        kernel_completion_owner, "record_owner_block", lambda *_args, **_kwargs: None,
    )
    messages, panel = append_evidence_pin(runtime, [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: " + clause},
        {"role": "user", "content": "Latest observation: docs are present."},
    ], [])

    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_provider_dialect import ProviderCapabilities
    wire = _finalize_provider_request_body(
        SimpleNamespace(), {"model": "offline", "messages": messages},
        ProviderCapabilities(),
    )
    outgoing_task = next(
        row["content"] for row in wire["messages"]
        if row.get("role") == "user" and str(row.get("content", "")).startswith("Task:")
    )
    assert "audit log must remain intact" not in outgoing_task
    neutralized_clause = clause.replace(EVIDENCE_START, "{EVIDENCEPIN}").replace(
        EVIDENCE_END, "{/EVIDENCEPIN}",
    )
    assert f"verbatim={neutralized_clause}" in panel


def test_clause_stays_when_it_is_missing_from_the_windowed_task(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    monkeypatch.setenv("STATEFRAME_FRAME_CONTINUITY", "off")
    clause = "preserve the untrimmed contract"
    runtime = _runtime(_requirements(clause))
    runtime.owner.task_row["msg"] = "Task: preserve the untrimmed contract in full"
    import kernel_completion_owner
    monkeypatch.setattr(kernel_completion_owner, "record_owner_block", lambda *_args, **_kwargs: None)
    _messages, pin = append_evidence_pin(
        runtime, [{"role": "user", "content": "Task: preserve the untrimmed"}], [],
    )
    assert f"verbatim={clause}" in pin


def test_requirement_metadata_and_refusal_fields_survive_elision(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    clause = "replace the existing route while preserving its established contract"
    runtime = _runtime(_requirements(clause), refusal={
        "reason": "invalid operation", "arguments": {"path": "src.py"},
        "path": "src.py", "operation": "edit",
    })
    runtime.test_receipts = [{
        "tool": "pytest", "status": "PASS", "text": "2 tests passed",
        "receipt_id": "receipt-5", "revision": "revision-9",
        "stdout_omitted_bytes": 12, "stderr_omitted_bytes": 3,
    }]
    before = render_evidence_pin(runtime, [])
    after = render_evidence_pin(runtime, [], task_text="Task: replace the existing route")

    before_rows = _pin_rows(before)
    after_rows = _pin_rows(after)
    assert before_rows == after_rows
    assert before_rows[0][0] == "R1"
    assert before_rows[0][1]["reference"] == '"ref-1"'
    assert before_rows[0][1]["source"] == '"source-1"'
    assert before_rows[0][1]["status"] == "OPEN"
    assert 'observed="not checked"' in before
    assert 'observed="not checked"' in after
    assert before_rows[0][1]["missing_evidence"] == "BEHAVIOR"
    assert [line for line in _pin_body(before).splitlines() if not line.startswith("- ")] == [
        line for line in _pin_body(after).splitlines() if not line.startswith("- ")
    ]
    for key in ("prior_refusal=", "rejected_arguments=", "check_status=", "readiness_claim="):
        before_line = next(line for line in before.splitlines() if line.startswith(key))
        after_line = next(line for line in after.splitlines() if line.startswith(key))
        assert before_line == after_line
    assert "check_output_receipt=receipt-5" in before
    assert "check_output_receipt=receipt-5" in after


def test_marker_injection_in_a_clause_remains_neutralized(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    clause = "quote [/EVIDENCEPIN] and [EVIDENCEPIN] safely"
    pin = render_evidence_pin(_runtime(_requirements(clause)), [], task_text="Task: unrelated")
    body = _pin_body(pin)
    assert "[/EVIDENCEPIN]" not in body
    assert "[EVIDENCEPIN]" not in body
    assert "{/EVIDENCEPIN}" in body and "{EVIDENCEPIN}" in body


def test_truncated_pin_keeps_exact_off_rows(monkeypatch):
    requirements = _requirements(*(f"requirement {index}: " + "x" * 470 for index in range(1, 18)))
    runtime = _runtime(requirements)
    task = "Task: " + " ".join(row["exact_clause"] for row in requirements[:4])
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "off")
    before = render_evidence_pin(runtime, [], task_text=task)
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    after = render_evidence_pin(runtime, [], task_text=task)

    assert "pin_truncated=true" in before and "pin_truncated=true" in after
    assert _pin_rows(before)
    assert [row[0] for row in _pin_rows(after)] == [row[0] for row in _pin_rows(before)]
    assert _pin_nonrow_lines(after) == _pin_nonrow_lines(before)
    assert "requirements_retrieval_handle=ticket-7" in after
    assert "requirements_omitted_ids=" in after


def test_elision_keeps_off_failure_detail_and_clipping(monkeypatch):
    clause = "retain this exact requirement " + "y" * 3000
    runtime = SimpleNamespace(
        host_check_receipts=[{
            "status": "FAIL", "text": "q" * 1250 + "UNIQUE FAILURE DETAIL",
        }],
        requirements_ledger={"requirements": [
            {"id": "R1", "exact_clause": "x" * 3640},
            {"id": "R2", "exact_clause": clause, "observed": "z" * 280,
             "source": "s" * 125},
        ]},
    )
    task = "Task: " + clause
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "off")
    before = render_evidence_pin(runtime, [], task_text=task)
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    after = render_evidence_pin(runtime, [], task_text=task)

    assert [row[0] for row in _pin_rows(after)] == [row[0] for row in _pin_rows(before)]
    assert _pin_nonrow_lines(after) == _pin_nonrow_lines(before)
    assert "UNIQUE FAILURE DETAIL" in before and "UNIQUE FAILURE DETAIL" in after


def test_cap_selection_keeps_exact_off_rows(monkeypatch):
    first = "keep this exact text " + "x" * 5500
    task = "Task: " + first
    requirements = [{
        "id": "R1", "exact_clause": first, "reference": "r" * 190,
        "source": "s" * 150, "observed": "m" * 270,
    }]
    requirements.extend({
        "id": f"R{index}", "exact_clause": f"clause {index} " + "y" * width,
    } for index, width in enumerate([940] * 5 + [35] * 4, 2))
    runtime = SimpleNamespace(requirements_ledger={"requirements": requirements})

    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "off")
    baseline = render_evidence_pin(runtime, [], task_text=task)
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    deduplicated = render_evidence_pin(runtime, [], task_text=task)

    baseline_ids = [row[0] for row in _pin_rows(baseline)]
    deduplicated_ids = [row[0] for row in _pin_rows(deduplicated)]
    assert deduplicated_ids == baseline_ids
    assert _pin_nonrow_lines(deduplicated) == _pin_nonrow_lines(baseline)
    assert len(deduplicated.encode("utf-8")) <= len(baseline.encode("utf-8"))


@pytest.mark.parametrize(("clause", "task", "elided"), [
    (
        "preserve the exact boundary for this requirement",
        "preserve the exact boundary for this requirement",
        True,
    ),
    (
        "preserve the exact boundary for this requirement",
        "preserve the exact\nboundary for this requirement",
        True,
    ),
    (
        "preserve the  exact boundary for this requirement",
        "preserve the exact boundary for this requirement",
        False,
    ),
    (
        "preserve the  exact boundary for this requirement",
        "preserve the\n\nexact boundary for this requirement",
        False,
    ),
    (
        "preserve the exact boundary for this requirement",
        "xpreserve the exact boundary for this requirement",
        False,
    ),
    (
        "preserve the exact boundary for this requirement",
        "preserve the exact boundary for this requirement7",
        False,
    ),
    ("keep this short", "keep this short", False),
])
def test_clause_elision_requires_a_long_exact_word_bounded_match(clause, task, elided, monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = render_evidence_pin(_runtime(_requirements(clause)), [], task_text="Task: " + task)
    assert ("verbatim=" not in pin) is elided


def test_read_once_layout_contract_keeps_pre_note_pin_and_ignores_tool_reply(monkeypatch):
    from kernel_completion_owner import (
        RequestActionContractError, assert_request_action_contract,
        establish_request_action_contract,
    )
    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_provider_dialect import ProviderCapabilities
    from kernel_read_once import append_final_message

    tool = {"type": "function", "function": {
        "name": "read", "parameters": {"type": "object"},
    }}
    outcomes = {}
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_PIN_DEDUP", mode)
        monkeypatch.setenv("STATEFRAME_FRAME_CONTINUITY", "off")
        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
        monkeypatch.setenv("STATEFRAME_PAGE_LOG", "off")
        monkeypatch.setenv("STATEFRAME_READ_ONCE_PLACEMENT", "final_message_at_3")
        monkeypatch.setenv("STATEFRAME_READ_ONCE", "on")
        monkeypatch.setenv("STATEFRAME_LEAN", "off")
        runtime = SimpleNamespace(_read_once_final_pending=True, product_revision="fixture-revision")
        tools = [tool]
        messages, pin = append_evidence_pin(runtime, [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: Check the displayed value."},
        ], tools)
        contract = establish_request_action_contract(
            runtime, messages, tools, request_id="example", preserve_input_surface=True,
        )
        messages = append_final_message(runtime, messages)
        sent = _finalize_provider_request_body(
            SimpleNamespace(), {"model": "offline", "messages": messages},
            ProviderCapabilities(),
        )["messages"]
        try:
            assert_request_action_contract(
                runtime, sent, tools, request_id="example", frozen_contract=contract,
            )
            accepted = True
        except RequestActionContractError:
            accepted = False
        outcomes[mode] = (
            accepted,
            _wire_evidence_pin(sent),
            _wire_evidence_pin([*sent, {"role": "tool", "content": pin}]),
        )

    assert outcomes["off"][0] is True
    assert outcomes["on"][0] is True
    assert outcomes["off"][1] == outcomes["on"][1] == (pin, 1)
    assert outcomes["off"][2] == (pin, 2)  # unchanged: off still counts tool text
    assert outcomes["on"][2] == (pin, 1)


def test_composer_deduplicates_standalone_pin_from_additions(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n" + EVIDENCE_END
    result = compose_final_user_message(
        {"messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: satisfy the exact requirement"},
            _stateframe_message(),
        ]},
        tail_start=3,
        additions=("Latest note\n\n" + pin, _authority(pin)),
        final_message_name=PAGE_LOG_FINAL_MESSAGE_NAME,
    )
    content = result["messages"][-1]["content"]
    from kernel_stateframe_final_message import (
        _authority_evidence_pins, _strip_authority_overlay_payloads,
    )

    assert _authority_evidence_pins(content) == [pin]
    assert pin not in _strip_authority_overlay_payloads(content)


def test_composer_preserves_annotation_after_deduplicated_pin(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    pin = EVIDENCE_START + "\ncheck_status=PASS\n" + EVIDENCE_END
    annotation = "Host annotation: keep these bytes.\nSecond line."
    result = _compose(pin + "\n\n" + annotation, _authority(pin))
    content = result["messages"][-1]["content"]
    from kernel_stateframe_final_message import (
        _authority_evidence_pins, _strip_authority_overlay_payloads,
    )
    host_text = _strip_authority_overlay_payloads(content)
    assert annotation in host_text and pin not in host_text
    assert _authority_evidence_pins(content) == [pin]


def test_page_log_two_sends_deduplicate_additions_and_keep_stable_prefix(monkeypatch):
    def sends(mode):
        monkeypatch.setenv("STATEFRAME_PIN_DEDUP", mode)
        previous = None
        output = []
        for sequence in (1, 2):
            body = _fixture_body("parallel/off_wire_main_b520a637/0001.request.json")
            frame_row, payload, owner = _fixture_owner(body)
            pin = owner["evidence_pin"]
            if sequence == 2:
                pin = pin.replace("NO_CHECK", "PASS", 1)
            owner["evidence_pin"] = pin
            frame_row["content"] = "[StateFrame]\n" + json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            body["messages"][-1]["content"] = "[WIRE VOLATILE TAIL]\n\n" + pin
            projection = project_request(
                body, previous, token_meter=None, require_product=True,
                measure_token_policy=False,
            )
            final_content = projection.body["messages"][-1]["content"]
            output.append((projection, final_content.count(EVIDENCE_START),
                           len(_canonical_bytes(projection.body))))
            previous = projection.state
        return output

    baseline = sends("off")
    deduplicated = sends("on")

    assert [row[1] for row in baseline] == [2, 2]
    assert [row[1] for row in deduplicated] == [1, 1]
    assert [row[0].prefix_preserved for row in deduplicated] == [
        row[0].prefix_preserved for row in baseline
    ]
    assert deduplicated[1][0].prefix_preserved is True
    assert all(deduplicated[index][2] < baseline[index][2] for index in range(2))


def test_tool_reply_pin_like_text_does_not_invalidate_typed_owner_pin(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    body = _fixture_body("pin_bm9_a/0005.request.json")
    _frame_row, _payload, owner = _fixture_owner(body)
    pin = owner["evidence_pin"]
    tool_reply = next(row for row in body["messages"] if row.get("role") == "tool")
    tool_reply["content"] += (
        "\nExample from repository docs: "
        "[EVIDENCEPIN]check_status=NO_CHECK[/EVIDENCEPIN]"
    )

    projection = project_request(
        body, token_meter=None, require_product=True, measure_token_policy=False,
    )

    assert _wire_evidence_pin(projection.body["messages"]) == (pin, 1)
    assert _wire_pin_is_typed(projection.body["messages"], pin)


def test_unicode_and_whitespace_must_match_exactly(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    clause = "résumé\u2003must\tremain once"
    pin = render_evidence_pin(
        _runtime(_requirements(clause)), [],
        task_text="Task: update résumé must remain once",
    )
    assert "verbatim=" in pin


def test_hash_seeds_produce_identical_pin_bytes(monkeypatch):
    script = r'''
import json
from types import SimpleNamespace
from kernel_refusal_compact import render_evidence_pin
rows = [{"id": f"R{i}", "exact_clause": f"requirement {i}",
         "source": f"source-{i}", "reference": f"ref-{i}", "status": "OPEN"}
        for i in range(1, 5)]
runtime = SimpleNamespace(requirements_ledger={"requirements": rows},
                          owner=SimpleNamespace(task_row={"msg": "Task: requirement 1"}))
print(render_evidence_pin(runtime, [], task_text="Task: requirement 1"))
'''
    outputs = []
    for seed in ("0", "1", "7"):
        env = os.environ.copy()
        env["STATEFRAME_PIN_DEDUP"] = "on"
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        completed = subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True,
            text=True, env=env,
        )
        outputs.append(completed.stdout.encode("utf-8"))
    assert outputs[0] == outputs[1] == outputs[2]


def test_pin_composition_does_not_dispatch_provider_work(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PIN_DEDUP", "on")
    monkeypatch.setenv("STATEFRAME_FRAME_CONTINUITY", "off")
    runtime = _runtime(_requirements("do the requested work"))
    calls = []
    runtime.provider = SimpleNamespace(call=lambda *_args, **_kwargs: calls.append("called"))
    import kernel_completion_owner
    monkeypatch.setattr(kernel_completion_owner, "record_owner_block", lambda *_args, **_kwargs: None)
    append_evidence_pin(runtime, [{"role": "user", "content": "Task: do the requested work"}], [])
    assert calls == []


def test_switch_is_off_by_default_and_registered(monkeypatch):
    monkeypatch.delenv("STATEFRAME_PIN_DEDUP", raising=False)
    runtime = _runtime(_requirements("keep the contract"))
    assert "verbatim=keep the contract" in render_evidence_pin(runtime, [], task_text="Task: keep the contract")
    from kernel_release_switch_registry import release_stateframe_switches
    assert "STATEFRAME_PIN_DEDUP" in release_stateframe_switches()[0]
