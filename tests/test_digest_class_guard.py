"""Keep unframed tree hashes from reappearing outside the stage 2 inventory."""
from pathlib import Path
from functools import lru_cache

from _guard_scan_cache import source_tree_signature
from tools.digest_class_guard import scan_source, scan_tree


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {}

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
    "stateframe/engine/kernel_expert_gap_sidecars.py:_fresh_closure",
    "stateframe/engine/kernel_remaining_sites.py:current_workspace_revision",
    "stateframe/engine/kernel_workspace_truth.py:_symbol_index",
    "stateframe/engine/kernel_workspace_truth.py:_current_literal_fact",
    "stateframe/engine/long_horizon_protocol.py:_staged_digest",
}


@lru_cache(maxsize=2)
def _cached_src_tree_scan(_signature):
    scanned, eligible, violated = scan_tree(ROOT / "src")
    return scanned, eligible, frozenset(violated)


def _src_tree_scan():
    return _cached_src_tree_scan(source_tree_signature(ROOT / "src" / "stateframe"))


def test_digest_class_guard_current_tree():
    scanned, eligible, violated = _src_tree_scan()
    print(f"scanned={scanned} eligible={eligible} violated={len(violated)}")
    assert scanned > 500 and eligible > 0
    assert violated == set(ALLOWLIST), ("new or resolved violation; update the exact allowlist", violated)


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


def test_guard_flags_nul_tagged_ticket_payload():
    planted = '''def bad(manifest_path, brief_path, phases):
    payload = manifest_path.read_bytes() + b"\\0brief\\0" + brief_path.read_bytes()
    for phase in phases:
        payload += b"\\0phase\\0" + phase["ticket"].encode()
    return _sha256_bytes(payload)
'''
    assert scan_source(planted)[1] == {"bad"}


def test_typed_call_cannot_hide_unframed_hash():
    planted = '''import hashlib
def bad(path, content):
    digest = hashlib.sha256()
    digest.update(path.encode())
    digest.update(content)
    tree_digest("unused/2", [])
    return digest.hexdigest()
'''
    assert scan_source(planted)[1] == {"bad"}


def test_historical_sites_are_flagged_or_fixed():
    # Every historical site is either fixed (gone from the scan) or explicitly allowlisted.
    _scanned, _eligible, violated = _src_tree_scan()
    remaining = HISTORICAL & violated
    assert remaining <= set(ALLOWLIST)
    assert not (HISTORICAL - set(ALLOWLIST)) & violated
