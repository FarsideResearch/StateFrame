"""Compare no-MCP request bytes with the immediate base release."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from tools.compare_no_mcp_requests import (
    capture, result_hashes, unbacked_result_hash_changes)
from acceptance._support import assert_tree_imports, tree_python

pytest_plugins = ("tools.immediate_baseline",)

ROOT = Path(__file__).resolve().parents[1]
_MISSING_BASELINE = ("SF_TEST_BASELINE_IMMEDIATE must name the immediate base checkout "
                     "for no-MCP parity")


def _nested_sandbox_refusal(data: bytes) -> bool:
    return ((b"sandbox-exec" in data and b"Operation not permitted" in data)
            or b"sandbox_apply" in data)


def _check_mac_sandbox(data: bytes) -> None:
    if sys.platform == "darwin" and _nested_sandbox_refusal(data):
        reason = "INFRA: nested sandbox-exec refused"
        if os.environ.get("SF_GATE_MUST_RUN", "").strip().lower() in {"1", "true", "yes", "on"}:
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)


def _capture_checked(*args, stable_tool_head: str | None,
                     final_cache_layout: str | None, **kwargs) -> bytes:
    try:
        result = capture(*args, stable_tool_head=stable_tool_head,
                         final_cache_layout=final_cache_layout, **kwargs)
    except RuntimeError as error:
        _check_mac_sandbox(str(error).encode("utf-8", errors="replace"))
        raise
    _check_mac_sandbox(result)
    run_dir = args[5]
    summary = run_dir / "summary.json"
    if summary.exists():
        _check_mac_sandbox(summary.read_bytes())
    return result


def _write_traced_launcher(path: Path, trace_path: Path) -> None:
    """Record imported modules after each pinned-tree CLI process exits."""
    path.write_text(
        "import json, runpy, secrets, sys\n"
        "_sequence = 0\n"
        "def fixed_hex(size=32):\n"
        "    global _sequence\n"
        "    _sequence += 1\n"
        "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
        "secrets.token_hex = fixed_hex\n"
        f"_trace_path = {str(trace_path)!r}\n"
        "try:\n"
        "    runpy.run_module('stateframe', run_name='__main__')\n"
        "finally:\n"
        "    with open(_trace_path, 'w', encoding='utf-8') as target:\n"
        "        json.dump(sorted(sys.modules), target)\n",
        encoding="utf-8",
    )


def test_nested_sandbox_refusal_is_infrastructure_not_parity(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    refusal = b"sandbox-exec: Operation not permitted"
    assert _nested_sandbox_refusal(refusal)
    assert _nested_sandbox_refusal(b"sandbox_apply failed")
    assert not _nested_sandbox_refusal(b"request byte mismatch")
    with pytest.raises(pytest.skip.Exception, match="INFRA: nested sandbox-exec refused"):
        _check_mac_sandbox(refusal)
    monkeypatch.setenv("SF_GATE_MUST_RUN", "1")
    with pytest.raises(pytest.fail.Exception, match="INFRA: nested sandbox-exec refused"):
        _check_mac_sandbox(refusal)


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    "SF_TEST_BASELINE_IMMEDIATE" not in os.environ
    and os.environ.get("SF_GATE_MUST_RUN", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason=_MISSING_BASELINE,
)
def test_no_mcp_requests_match_immediate_base(tmp_path, baseline_checkout, must_run_witness):
    """A Mac host that refuses nested sandbox-exec cannot complete this capture.

    The shared capture helper runs the product's check sandbox. Classify that host
    refusal as infrastructure so it cannot be mistaken for changed request bytes.
    """
    baseline = baseline_checkout
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    baseline_trace = tmp_path / "baseline-modules.json"
    current_trace = tmp_path / "candidate-modules.json"
    baseline_launcher = tmp_path / "baseline-run.py"
    current_launcher = tmp_path / "candidate-run.py"
    _write_traced_launcher(baseline_launcher, baseline_trace)
    _write_traced_launcher(current_launcher, current_trace)
    run_dir = tmp_path / "run"
    script = ROOT / "examples/discount_bug.script.json"
    baseline_python = tree_python(baseline / "src", tmp_path / "baseline-python")
    assert_tree_imports(baseline_python, baseline / "src", tmp_path)
    previous = _capture_checked(baseline, baseline_python, baseline_launcher, script, repo, run_dir,
                                stable_tool_head=None, final_cache_layout=None,
                                source_on_pythonpath=False, test_python=sys.executable)
    run_dir.rename(tmp_path / "baseline-run")
    current_python = tree_python(ROOT / "src", tmp_path / "current-python")
    assert_tree_imports(current_python, ROOT / "src", tmp_path)
    current = _capture_checked(ROOT, current_python, current_launcher, script, repo, run_dir,
                               stable_tool_head=None, final_cache_layout=None,
                               source_on_pythonpath=False, test_python=sys.executable)
    # A result_sha256 must be backed by a tool message in the same run.
    assert result_hashes(run_dir)[0], "no result_sha256 found; the hash check below would be vacuous"
    assert unbacked_result_hash_changes(tmp_path / "baseline-run", run_dir) == []
    # Fixed seeds make the complete request stream, including wire IDs, comparable.
    (tmp_path / "baseline-requests.bin").write_bytes(previous)
    (tmp_path / "candidate-requests.bin").write_bytes(current)
    previous_sha = hashlib.sha256(previous).hexdigest()
    current_sha = hashlib.sha256(current).hexdigest()
    assert previous == current, (f"immediate base sha256={previous_sha}; "
                                f"candidate sha256={current_sha}")
    must_run_witness.record_parity(
        baseline_trace=json.loads(baseline_trace.read_text(encoding="utf-8")),
        candidate_trace=json.loads(current_trace.read_text(encoding="utf-8")),
        expected_trace_prefixes=("stateframe",),
        comparison_completed=True,
    )


def test_wire_id_drift_names_moved_fields_and_rejects_other_changes(tmp_path):
    import json

    from tools.compare_no_mcp_requests import (
        planted_body_change_is_caught, request_id_handle_mismatches, wire_id_drift)

    def body(request_id, receipt):
        frame = {"state": {"framefit5": {"action_ledger": {"applied_edits": [
            {"edit_receipt": receipt, "request_id": request_id},
        ]}}}}
        content = ("[StateFrame]\n" + json.dumps(frame, separators=(",", ":"))
                   + f"\nFact handle=wire-truth:{request_id}:green_state")
        return json.dumps({"messages": [{"role": "user", "content": content}]}).encode()

    old, new = body("a" * 20, "b" * 20), body("c" * 20, "d" * 20)
    problems, kinds = wire_id_drift([old], [new])
    assert problems == []
    assert kinds == {"wire_truth_handle(request_id)": 1, "request_id": 1, "edit_receipt": 1}
    legacy_suffix = old.replace(b"wire-truth:" + b"a" * 20 + b":green_state",
                                b"wire-truth:" + b"a" * 20 + b":1067:green_state")
    assert wire_id_drift([legacy_suffix], [old]) == (
        [], {"wire_truth_handle(request_id)": 1})
    refs_before = json.dumps({"messages": [{"role": "user", "content":
        "check_execution_ref: " + "a" * 20 + ":1," + "b" * 20 + ":2"}]}).encode()
    refs_after = json.dumps({"messages": [{"role": "user", "content":
        "check_execution_ref: " + "c" * 20 + ":1," + "d" * 20 + ":2"}]}).encode()
    assert wire_id_drift([refs_before], [refs_after]) == (
        [], {"check_execution_ref": 2})
    changed_ordinal = refs_after.replace(b"c" * 20 + b":1", b"c" * 20 + b":9")
    assert wire_id_drift([refs_before], [changed_ordinal]) == (
        [], {"check_execution_ref": 2})
    no_ordinal = refs_after.replace(b"c" * 20 + b":1", b"c" * 20).replace(
        b"d" * 20 + b":2", b"d" * 20)
    assert wire_id_drift([refs_before], [no_ordinal]) == (
        [], {"check_execution_ref": 2})
    assert wire_id_drift([refs_before], [no_ordinal.replace(b"check_execution_ref", b"changed_ref")])[0]
    def finding(finding_id):
        frame = {"state": {"history_findings": {"latest": {"id": finding_id}}}}
        return json.dumps({"messages": [{"role": "user", "content":
            "[StateFrame]\n" + json.dumps(frame, separators=(",", ":"))}]}).encode()
    assert wire_id_drift([finding("check:" + "a" * 20 + ":616")],
                         [finding("check:" + "b" * 20)]) == (
        [], {"check_finding_id": 1})
    assert wire_id_drift([finding("check:" + "a" * 20 + ":616")],
                         [finding("check:" + "b" * 20 + ":617")]) == (
        [], {"check_finding_id": 1})
    # A change outside the IDs is reported, and the planted control finds such changes.
    assert wire_id_drift([old], [new.replace(b"green_state", b"red_state")])[0]
    non_id_hex_before = json.dumps({"messages": [{"role": "user", "content": "c" * 24}]}).encode()
    non_id_hex_after = json.dumps({"messages": [{"role": "user", "content": "d" * 24}]}).encode()
    assert wire_id_drift([non_id_hex_before], [non_id_hex_after])[0]
    def deferred_read(receipt, suffix):
        return json.dumps({"messages": [{"role": "user", "content":
            f"journal receipts {receipt}; status {suffix}"}]}).encode()
    assert wire_id_drift([deferred_read("a" * 20, "same")],
                         [deferred_read("b" * 20, "same")]) == ([], {"receipt_id": 1})
    assert wire_id_drift([deferred_read("a" * 20, "same")],
                         [deferred_read("b" * 20, "changed")])[0]
    evidence_before = json.dumps({"required_evidence_ids": ["R1", "receipt:" + "a" * 24]}).encode()
    evidence_after = json.dumps({"required_evidence_ids": ["R1", "receipt:" + "b" * 24]}).encode()
    assert wire_id_drift([evidence_before], [evidence_after]) == ([], {"check_receipt": 1})
    assert wire_id_drift([evidence_before], [evidence_after.replace(b'"R1"', b'"R2"')])[0]
    assert planted_body_change_is_caught(new)
    assert wire_id_drift([old], [new, new])[0]

    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    (run / "wire").mkdir()
    (run / "requests" / "0001.json").write_text(json.dumps({"request_id": "c" * 20}))
    (run / "wire" / "0001.request.json").write_bytes(new)
    assert request_id_handle_mismatches(run) == []
    (run / "requests" / "0001.json").write_text(json.dumps({"request_id": "e" * 20}))
    assert request_id_handle_mismatches(run)


def test_release_str_replace_allowance_is_exactly_the_old_batch_schema():
    from tools.compare_no_mcp_requests import release_wire_id_drift

    scalar = {"type": "object", "properties": {
        "path": {"type": "string", "description":
                 "Declared product file path; the single EditAuthority lists eligible paths."},
        "old_str": {"type": "string"}, "new_str": {"type": "string"},
        "why": {"type": "string", "description":
                "one line: what you expect this action to show or achieve, "
                "and how it serves your current plan"},
        "operation": {"type": "string", "enum": ["replace"], "description":
                      "Exact text replacement; insertion and move need a host-issued anchor."},
    }, "required": ["new_str", "old_str", "path"], "additionalProperties": True}
    description = ("Exact text replacement in a path listed by the single EditAuthority. "
                   "The old text must match the current file exactly once.")
    old_tool = {"type": "function", "function": {"name": "str_replace",
        "description": description, "parameters": {**scalar, "properties": {
            **scalar["properties"], "edits": {"type": "array"}}}}}
    new_tool = {"type": "function", "function": {"name": "str_replace",
        "description": description, "parameters": scalar}}
    other = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}

    def body(tool, *, other_tool=other, task="same"):
        return json.dumps({"model": "scripted", "tools": [other_tool, tool],
                           "messages": [{"role": "user", "content": task}]},
                          separators=(",", ":")).encode()

    baseline, current = body(old_tool), body(new_tool)
    assert release_wire_id_drift([baseline], [current]) == ([], {})
    assert release_wire_id_drift([current], [current]) == ([], {})
    assert release_wire_id_drift([baseline, baseline], [current])[0]
    assert release_wire_id_drift([baseline], [body(new_tool, task="changed")])[0]
    changed_other = {**other, "function": {**other["function"], "description": "changed"}}
    assert release_wire_id_drift([baseline], [body(new_tool, other_tool=changed_other)])[0]
    missing_required = {**scalar, "required": ["path", "old_str"]}
    altered_candidate = {**new_tool, "function": {**new_tool["function"],
                                                   "parameters": missing_required}}
    assert release_wire_id_drift([baseline], [body(altered_candidate)])[0]
    altered_batch = {**old_tool["function"]["parameters"], "required": ["edits"]}
    altered_baseline = {**old_tool, "function": {**old_tool["function"],
        "parameters": altered_batch}}
    assert release_wire_id_drift([body(altered_baseline)], [current])[0]


def test_applied_edit_history_digest_masks_only_its_declared_path():
    """The pruned-row digest contains generated IDs; adjacent data remains visible."""
    import json

    from tools.compare_no_mcp_requests import wire_id_drift

    def body(rows, count, note="stable"):
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        frame = {"state": {"framefit5": {"action_ledger": {
            "historical_applied_edits": {"count": count, "sha256": digest},
            "note": note,
        }}}}
        content = "[StateFrame]\n" + json.dumps(frame, separators=(",", ":"))
        return json.dumps({"messages": [{"role": "user", "content": content}]}).encode()

    rows = [{"edit_receipt": "a" * 20, "file": "shop/pricing.py"}]
    moved_ids = [{"edit_receipt": "b" * 20, "file": "shop/pricing.py"}]
    # An older generated edit receipt moves the digest at this exact path.
    assert wire_id_drift([body(rows, 1)], [body(moved_ids, 1)]) == (
        [], {"historical_applied_edits_sha256(edit_ids)": 1})
    # The count is compared too: a different number of pruned edits fails.
    assert wire_id_drift([body(rows, 1)], [body(rows, 2)])[0]
    # A non-ID hex change beside the digest, including another 64-hex hash, fails.
    assert wire_id_drift([body(rows, 1, "c" * 64)],
                         [body(moved_ids, 1, "d" * 64)])[0]


def test_diet_size_counts_move_only_with_the_id_length_change():
    from test_hook_wiring_parity import _size_count_problems, _without_size_counts

    old = {"request_bytes_before": 100, "sections": [{"after_tokens": 10, "section": "a"}]}
    shorter = {"request_bytes_before": 96, "sections": [{"after_tokens": 9, "section": "a"}]}
    assert _size_count_problems(old, shorter, -8, changed_chars=20) == []
    # Tokens follow ID characters, so they may even rise, within the changed characters.
    more_tokens = {**shorter, "sections": [{"after_tokens": 11, "section": "a"}]}
    assert _size_count_problems(old, more_tokens, -8, changed_chars=20) == []
    # Planted: no ID changed at all, so any count change is reported.
    assert _size_count_problems(old, shorter, 0, changed_chars=0)
    # Planted: bytes that moved against the IDs or by more than they did; tokens past the bound.
    assert _size_count_problems(old, {**shorter, "request_bytes_before": 104}, -8, changed_chars=20)
    assert _size_count_problems(old, {**shorter, "request_bytes_before": 80}, -8, changed_chars=20)
    assert _size_count_problems(old, {**shorter, "sections": [{"after_tokens": 40, "section": "a"}]},
                                -8, changed_chars=20)
    # Non-count fields are kept for the masked comparison.
    assert _without_size_counts(shorter) == {"sections": [{"section": "a"}]}


def test_unbacked_result_hash_change_is_reported(tmp_path):
    import json

    def run(name, sent_text, hashed_text):
        digest = hashlib.sha256(hashed_text.encode()).hexdigest()
        (tmp_path / name / "wire").mkdir(parents=True)
        (tmp_path / name / "wire" / "0001.request.json").write_text(json.dumps({
            "messages": [{"role": "tool", "content": sent_text},
                         {"role": "user", "content": json.dumps({"result_sha256": digest})}]}))
        return tmp_path / name

    before = run("before", "result id aaaa", "result id aaaa")
    backed = run("backed", "result id bbbb", "result id bbbb")
    hidden = run("hidden", "result id bbbb", "text that was never sent")
    assert len(result_hashes(before)[0]) == 1  # the hash is found, so the check is not vacuous
    assert unbacked_result_hash_changes(before, backed) == []
    assert len(unbacked_result_hash_changes(before, hidden)) == 1


def test_stale_result_hash_is_backed_by_earlier_oracle_reply(tmp_path):
    """The action ledger hashes a stale label plus an earlier, separately sent reply."""
    import json

    from tools.compare_no_mcp_requests import unbacked_result_hash_changes

    def run(name, receipt_id, *, result_text=None, result_chars=None, prior_role="tool"):
        directory = tmp_path / name
        (directory / "wire").mkdir(parents=True)
        prior = ("pytest exit 1 — multi-selector oracle (1 selectors)\n"
                 f"CHECK_FAILURE_SUMMARY full_output_handle=check-receipt:{receipt_id}")
        prefix = ("[STALE TEST RESULT revision=sha256:" + "a" * 64
                  + "; superseded by GREEN revision=sha256:" + "b" * 64 + "]")
        stale = prefix + "\n" + (prior if result_text is None else result_text)
        digest = hashlib.sha256(stale.encode()).hexdigest()
        row = {"tool": "run_tests", "result_receipt": prefix,
               "result_chars": len(stale) if result_chars is None else result_chars,
               "result_sha256": digest}
        frame = "[StateFrame]\n" + json.dumps({"state": {"framefit5": {
            "action_ledger": {"unclassified_actions": [row]}}}})
        (directory / "wire" / "0001.request.json").write_text(json.dumps({
            "messages": [{"role": prior_role, "content": prior}]}))
        (directory / "wire" / "0002.request.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": frame}]}))
        return directory

    before = run("before", "c" * 24)
    after = run("after", "d" * 24)
    assert unbacked_result_hash_changes(before, after) == []
    # Neither an unsent tail, a false length, nor non-tool text can back the hash.
    hidden = run("hidden", "d" * 24, result_text="pytest exit 1 — unsent result")
    wrong_length = run("wrong-length", "d" * 24, result_chars=0)
    not_tool = run("not-tool", "d" * 24, prior_role="user")
    assert len(unbacked_result_hash_changes(before, hidden)) == 1
    assert len(unbacked_result_hash_changes(before, wrong_length)) == 1
    assert len(unbacked_result_hash_changes(before, not_tool)) == 1


def test_mask_wire_ids_uses_declared_paths_and_preserves_unrelated_hex():
    import json

    from tools.compare_no_mcp_requests import mask_wire_ids

    def body(tool_call_id, content):
        return json.dumps({"messages": [
            {"role": "assistant", "tool_calls": [{"id": tool_call_id,
                "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "user", "content": content},
        ]}).encode()

    unchanged = "this message contains an unrelated value " + "c" * 24
    first = body("native_" + "a" * 20 + "_0", unchanged)
    changed_id = body("native_" + "b" * 20 + "_0", unchanged)
    assert mask_wire_ids(first) == mask_wire_ids(changed_id)

    changed_content = body("native_" + "b" * 20 + "_0",
                           "this message contains an unrelated value " + "d" * 24)
    assert mask_wire_ids(changed_id) != mask_wire_ids(changed_content)
    assert mask_wire_ids(first) != mask_wire_ids(body("native_" + "b" * 20 + "_1", unchanged))
    assert mask_wire_ids(body("call-good", unchanged)) != mask_wire_ids(body("call-bad", unchanged))
    assert mask_wire_ids(body("native_" + "a" * 20 + "_0:auto_verify", unchanged)) == mask_wire_ids(
        body("native_" + "b" * 20 + "_0:auto_verify", unchanged))
    assert mask_wire_ids(body("native_" + "a" * 20 + "_0:auto_verify", unchanged)) != mask_wire_ids(
        body("native_" + "b" * 20 + "_1:auto_verify", unchanged))
    oracle_a = "CHECK_FAILURE_SUMMARY execution_id=check-execution:" + "a" * 20 + " full_output_handle=check-receipt:" + "b" * 24
    oracle_b = "CHECK_FAILURE_SUMMARY execution_id=check-execution:" + "c" * 20 + " full_output_handle=check-receipt:" + "d" * 24
    assert mask_wire_ids(body("call-good", oracle_a)) == mask_wire_ids(body("call-good", oracle_b))
    assert mask_wire_ids(body("call-good", oracle_a)) != mask_wire_ids(body("call-good", oracle_b + " changed"))
    expectation_a = "receipt expectation-receipt:" + "a" * 24 + "."
    expectation_b = "receipt expectation-receipt:" + "b" * 24 + "."
    assert mask_wire_ids(body("call-good", expectation_a)) == mask_wire_ids(
        body("call-good", expectation_b))
    assert mask_wire_ids(body("call-good", expectation_a)) != mask_wire_ids(
        body("call-good", expectation_b + " changed"))
    # Similar looking prose and unrelated hex must not be treated as a minted receipt.
    assert mask_wire_ids(body("call-good", "otherexpectation-receipt:" + "a" * 24)) != mask_wire_ids(
        body("call-good", "otherexpectation-receipt:" + "b" * 24))
    assert mask_wire_ids(body("call-good", "expectation-receipt:" + "a" * 24 + "suffix")) != mask_wire_ids(
        body("call-good", "expectation-receipt:" + "b" * 24 + "suffix"))
    assert mask_wire_ids(body("call-good", expectation_a + " value " + "c" * 24)) != mask_wire_ids(
        body("call-good", expectation_b + " value " + "d" * 24))

    digest = "ab" * 32
    frame = {"state": {"framefit5": {"action_ledger": {"unclassified_actions": [
        {"result_sha256": digest, "candidate_revision": "sha256:" + digest},
    ]}}}}
    content = "[StateFrame]\n" + json.dumps(frame, separators=(",", ":"))
    masked = mask_wire_ids(json.dumps({"messages": [{"role": "user", "content": content}]}).encode())
    assert b"result_sha256" in masked and b"<wire-id>" in masked
    assert masked.count(digest.encode()) == 1
    assert (b"sha256:" + digest.encode()) in masked
    changed_hash = json.loads(json.dumps(frame))
    changed_hash["state"]["framefit5"]["action_ledger"]["unclassified_actions"][0][
        "result_sha256"] = "cd" * 32
    changed_content = "[StateFrame]\n" + json.dumps(changed_hash, separators=(",", ":"))
    assert mask_wire_ids(json.dumps({"messages": [{"role": "user", "content": changed_content}]}).encode()) == masked
    changed_hash["state"]["framefit5"]["action_ledger"]["unclassified_actions"][0][
        "candidate_revision"] = "sha256:" + "cd" * 32
    changed_content = "[StateFrame]\n" + json.dumps(changed_hash, separators=(",", ":"))
    assert mask_wire_ids(json.dumps({"messages": [{"role": "user", "content": changed_content}]}).encode()) != masked


def test_wire_id_paths_match_real_captured_request_fixtures():
    from tools.compare_no_mcp_requests import WIRE_ID_PATHS, _mask_wire_ids_with_values

    fixture_root = Path(__file__).resolve().parent / "fixtures"
    fixtures = sorted(fixture_root.rglob("*.request.json"))
    assert fixtures, "no captured request fixtures available for wire-ID path coverage"
    coverage = [0] * len(WIRE_ID_PATHS)
    for fixture in fixtures:
        _, path_values = _mask_wire_ids_with_values(fixture.read_bytes())
        for index, (_kind, values) in enumerate(path_values):
            coverage[index] += len(values)
    # Provider wire fixtures have no CLI preflight wrapper. Both release and current
    # _child.py write this top-level request_id in requests/NNNN.json records.
    native_call = "native_" + "a" * 20 + "_0"
    preflight = {"request": 1, "turn": 1, "request_id": "a" * 20,
                 "messages": [{"role": "assistant", "tool_calls": [{"id": native_call}]},
                              {"role": "tool", "tool_call_id": native_call,
                               "content": ("CHECK_FAILURE_SUMMARY execution_id=check-execution:"
                                           + "a" * 20 + " full_output_handle=check-receipt:"
                                           + "b" * 24)}],
                 "tools": [], "presentation": {}}
    _, path_values = _mask_wire_ids_with_values(json.dumps(preflight).encode())
    for index, (_kind, values) in enumerate(path_values):
        coverage[index] += len(values)
    _, path_values = _mask_wire_ids_with_values(json.dumps({
        "required_evidence_ids": ["R1", "receipt:" + "a" * 24],
    }).encode())
    for index, (_kind, values) in enumerate(path_values):
        coverage[index] += len(values)
    historical = {"state": {"framefit5": {"action_ledger": {
        "historical_applied_edits": {"count": 1, "sha256": "a" * 64},
    }}}}
    _, path_values = _mask_wire_ids_with_values(json.dumps({"messages": [
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(historical)},
        {"role": "user", "content": "receipt expectation-receipt:" + "a" * 24},
    ]}).encode())
    for index, (_kind, values) in enumerate(path_values):
        coverage[index] += len(values)
    missing = [f"{kind}: {'.'.join(path)}" for index, (path, kind, _marker, _lengths)
               in enumerate(WIRE_ID_PATHS) if not coverage[index]]
    assert not missing, f"wire-ID JSON paths matched no captured request field: {missing}"


def test_mask_wire_ids_rejects_non_json_bodies():
    from tools.compare_no_mcp_requests import mask_wire_ids

    with pytest.raises(json.JSONDecodeError):
        mask_wire_ids(b"request body that is not JSON")


def test_mask_wire_ids_parses_multiline_preflight_capture_and_hook_payloads():
    from tools.compare_no_mcp_requests import mask_wire_ids, wire_id_drift

    def preflight(request_id, content):
        return json.dumps({"request": 1, "request_id": request_id,
                           "messages": [{"role": "user", "content": content}],
                           "tools": [], "presentation": {}}, indent=1).encode()

    old = preflight("a" * 20, "fixed " + "c" * 24)
    new = preflight("b" * 20, "fixed " + "c" * 24)
    assert mask_wire_ids(old + b"\n" + old) == mask_wire_ids(new + b"\n" + new)
    assert mask_wire_ids(old + b"\n" + old) != mask_wire_ids(new + b"\n" +
                                                         preflight("b" * 20, "fixed " + "d" * 24))
    assert wire_id_drift([json.dumps([{"request_id": "a" * 20}]).encode()],
                         [json.dumps([{"request_id": "b" * 20}]).encode()]) == (
        [], {"request_id": 1})
