"""Keep unframed tree hashes from reappearing outside the stage 2 inventory."""
from pathlib import Path

from tools.digest_class_guard import scan_source, scan_tree


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {
    "stateframe/engine/chains.py:tree_hash": "legacy in-memory source snapshot; stage 2",
    "stateframe/engine/behavior_bridge.py:_files_hash": "legacy behavior source snapshot; stage 2",
    "stateframe/engine/build_tasks/swebench_env.py:_dependency_manifest_digest": "SWEbench dependency manifests; stage 2",
    "stateframe/engine/kernel_state_owner_inventory.py:_source_revision_vector": "state owner revision vector; stage 2",
    "stateframe/engine/kernel_workspace_truth.py:_symbol_index": "workspace symbol index; stage 2",
    "stateframe/engine/kernel_workspace_truth.py:_current_literal_fact": "workspace literal evidence; stage 2",
    "stateframe/engine/kernel_expert_gap_sidecars.py:_fresh_closure": "dependency-closure freshness, path+NUL+bytes; stage 2",
    "stateframe/engine/kernel_remaining_sites.py:current_workspace_revision": "site-inventory revision, path+NUL+source+NUL; stage 2",
}

# The ten digest-family sites proven to collide on 0.3.18; the guard must flag every one
# of them on an unfixed tree (run against e19c325 in the release evidence).
HISTORICAL = {
    "stateframe/engine/kernel_coordinated_change_ledger.py:_compute_tree_revision",
    "stateframe/engine/kernel_request_evidence_closure.py:dependency_revision",
    "stateframe/engine/kernel_request_source_window.py:_revision12",
    "stateframe/engine/kernel_stateframe_get_sure.py:_workspace_fingerprint",
    "stateframe/engine/kernel_ship_tree_smoke.py:tree_digest",
    "stateframe/engine/realtask_benchmark.py:_tree_digest",
    "stateframe/engine/chains.py:tree_hash",
    "stateframe/engine/behavior_bridge.py:_files_hash",
    "stateframe/engine/build_tasks/swebench_env.py:_dependency_manifest_digest",
    "stateframe/engine/kernel_state_owner_inventory.py:_source_revision_vector",
}


def test_digest_class_guard_current_tree():
    scanned, eligible, violated = scan_tree(ROOT / "src")
    print(f"scanned={scanned} eligible={eligible} violated={len(violated)}")
    assert scanned > 500 and eligible > 0
    assert violated == set(ALLOWLIST), ("new or resolved violation; update the exact allowlist", violated)
    assert all(reason and "stage 2" in reason for reason in ALLOWLIST.values())


def test_guard_flags_planted_path_content_concatenation():
    planted = """import hashlib
def bad(root, path):
    digest = hashlib.sha256()
    digest.update(path.encode())
    digest.update((root / path).read_bytes())
    return digest.hexdigest()
def bad_one(path, content):
    return hashlib.sha256(path.encode() + content).hexdigest()
def bad_join(path, content):
    return hashlib.new('sha256', b''.join((path.encode(), content))).hexdigest()
"""
    eligible, violated = scan_source(planted)
    assert eligible >= 3 and violated == {"bad", "bad_one", "bad_join"}


def test_guard_treats_nul_separators_as_unframed():
    # NUL "framing" is the crafted-collision class: content may itself contain NUL.
    planted = """import hashlib
def nul_framed(files):
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(files[relative].encode("utf-8"))
        digest.update(b"\\0")
    return digest.hexdigest()
"""
    assert scan_source(planted)[1] == {"nul_framed"}


def test_historical_sites_are_flagged_or_fixed():
    # Every historical site is either fixed (gone from the scan) or explicitly allowlisted.
    _scanned, _eligible, violated = scan_tree(ROOT / "src")
    remaining = HISTORICAL & violated
    assert remaining <= set(ALLOWLIST)
    assert not (HISTORICAL - set(ALLOWLIST)) & violated
