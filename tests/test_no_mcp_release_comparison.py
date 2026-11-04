"""Compare no-MCP request bytes with the pinned 0.2.3 source tree."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tools.compare_no_mcp_requests import (
    capture, mask_wire_ids, result_hashes, unbacked_result_hash_changes, write_launcher)
from acceptance._support import assert_tree_imports, tree_python

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "33c6c511611cc249b90b47930905a8e60648c05c"
BASELINE_TREE = "59bef5821e45d506309bbb5efb5ed3fb59e52279"


def _materialize_baseline(destination: Path) -> str:
    if shutil.which("git") is None:
        pytest.skip("git is unavailable, so the pinned 0.2.3 tree cannot be materialized")
    present = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e",
                              f"{BASELINE_COMMIT}^{{commit}}"], capture_output=True, check=False)
    if present.returncode != 0:
        pytest.skip("pinned 0.2.3 commit 33c6c51 is unavailable in this repository's git objects")
    tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{BASELINE_COMMIT}^{{tree}}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert tree == BASELINE_TREE, f"pinned 0.2.3 commit has unexpected tree {tree}"
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar",
                              BASELINE_COMMIT], capture_output=True, check=True).stdout
    destination.mkdir()
    subprocess.run(["tar", "-xf", "-", "-C", str(destination)], input=archive, check=True)
    return tree


@pytest.mark.host_sandbox
def test_no_mcp_requests_match_release_0_2_3(tmp_path):
    baseline = tmp_path / "baseline"
    tree = _materialize_baseline(baseline)
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    run_dir = tmp_path / "run"
    script = ROOT / "examples/discount_bug.script.json"
    baseline_python = tree_python(baseline / "src", tmp_path / "baseline-python")
    assert_tree_imports(baseline_python, baseline / "src", tmp_path)
    previous = capture(baseline, baseline_python, launcher, script, repo, run_dir, stable_tool_head=None,
                       source_on_pythonpath=False, test_python=sys.executable)
    run_dir.rename(tmp_path / "baseline-run")
    current_python = tree_python(ROOT / "src", tmp_path / "current-python")
    assert_tree_imports(current_python, ROOT / "src", tmp_path)
    # The default now treats unittest as a generic check and permits test edits.
    # Leave the stable tool head unset here to exercise the new default; pin it off only for byte comparison.
    default = capture(ROOT, current_python, launcher, script, repo, run_dir,
                      stable_tool_head=None, source_on_pythonpath=False,
                      test_python=sys.executable)
    assert default
    run_dir.rename(tmp_path / "default-run")
    current = capture(ROOT, current_python, launcher, script, repo, run_dir,
                      legacy_mode=True, stable_tool_head="off", source_on_pythonpath=False,
                      test_python=sys.executable)
    # A masked result_sha256 must not hide a content change. Each hash that differs
    # between the releases has to hash a tool message sent in the same run, and those
    # messages are compared below after wire-ID masking, so the text behind every
    # masked hash is still checked byte for byte.
    assert result_hashes(run_dir)[0], "no result_sha256 found; the hash check below would be vacuous"
    assert unbacked_result_hash_changes(tmp_path / "baseline-run", run_dir) == []
    # Wire IDs changed derivation on purpose (receipt determinism, 11-03); see mask_wire_ids.
    previous, current = mask_wire_ids(previous), mask_wire_ids(current)
    # Kept next to the run directories so a failure can be diffed directly.
    (tmp_path / "baseline-masked.bin").write_bytes(previous)
    (tmp_path / "candidate-masked.bin").write_bytes(current)
    previous_sha = hashlib.sha256(previous).hexdigest()
    current_sha = hashlib.sha256(current).hexdigest()
    assert previous == current, (f"0.2.3 commit {BASELINE_COMMIT} tree {tree}: "
                                f"sha256={previous_sha}; candidate sha256={current_sha}")


def test_wire_id_drift_names_moved_fields_and_rejects_other_changes(tmp_path):
    import json

    from tools.compare_no_mcp_requests import (
        planted_body_change_is_caught, request_id_handle_mismatches, wire_id_drift)

    def body(request_id, receipt):
        return json.dumps({"messages": [
            {"role": "user", "content": f"fact handle=wire-truth:{request_id}:green_state"},
            {"role": "tool", "content": json.dumps({"edit_receipt": receipt, "status": "ok"})},
        ]}).encode()

    old, new = body("a" * 20, "b" * 20), body("c" * 20, "d" * 20)
    problems, kinds = wire_id_drift([old], [new])
    assert problems == []
    assert kinds == {"wire_truth_handle(request_id)": 1, "edit_receipt": 1}
    # A change outside the IDs is reported, and the planted control finds such changes.
    assert wire_id_drift([old], [new.replace(b"ok", b"no")])[0]
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


def test_applied_edit_history_digest_mask_keeps_count_and_states_its_blind_spot():
    """historical_applied_edits.sha256 is masked, its count is not.

    KNOWN LIMITATION: the digest's preimage is the older applied-edit rows pruned from the
    frame. They are not in the same request, so a release comparison cannot recompute it from
    what was sent; it relies on each pruned row having been sent, and compared after masking,
    in an earlier request while it was among the last three edits. A change that exists ONLY
    inside a pruned row's digest is invisible to the mask. The planted case below shows that
    blind spot on purpose; the count and every other byte stay compared.
    """
    import json

    from tools.compare_no_mcp_requests import wire_id_drift

    def body(rows, count):
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        ledger = {"historical_applied_edits": {"count": count, "sha256": digest}}
        return json.dumps({"messages": [{"role": "user", "content": json.dumps(ledger)}]}).encode()

    rows = [{"edit_receipt": "a" * 20, "file": "shop/pricing.py"}]
    moved_ids = [{"edit_receipt": "b" * 20, "file": "shop/pricing.py"}]
    problems, kinds = wire_id_drift([body(rows, 1)], [body(moved_ids, 1)])
    assert problems == [] and kinds == {"historical_applied_edits_sha256": 1}
    # The count is still compared: a different number of pruned edits fails.
    assert wire_id_drift([body(rows, 1)], [body(rows, 2)])[0]
    # Blind spot, planted: a non-ID change inside a pruned row alters only the digest,
    # so the masked comparison cannot see it.
    planted = [{"edit_receipt": "a" * 20, "file": "shop/discount.py"}]
    assert wire_id_drift([body(rows, 1)], [body(planted, 1)])[0] == []


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


def test_mask_wire_ids_hides_only_ids_and_result_hashes():
    digest = "ab" * 32
    body = (b'{"result_sha256":"' + digest.encode() + b'","escaped":"{\\"result_sha256\\":\\"'
            + digest.encode() + b'\\"}","candidate_revision":"sha256:' + digest.encode()
            + b'","ref":"' + b"c" * 20 + b'","check":"' + b"d" * 24 + b':1067","long":"'
            + b"e" * 40 + b'"}')
    masked = mask_wire_ids(body)
    assert masked.count(b"<result-sha256>") == 2
    assert masked.count(b"<wire-id>") == 2
    # Every other hex value is still compared byte for byte.
    assert b"sha256:" + digest.encode() in masked and b"e" * 40 in masked
