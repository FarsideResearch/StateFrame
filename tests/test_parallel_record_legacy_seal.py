"""Legacy nested receipts retain their sealed proposal bytes until changed."""

import json
from pathlib import Path

from kernel_parallel_records import Proposal, canonical_json


def test_legacy_proposal_reencode_changes_only_after_logical_mutation():
    path = Path(__file__).parent / 'fixtures/parallel/recorded_sealed_proposal_v1.json'
    wire = path.read_bytes().rstrip(b'\n')
    proposal = Proposal.from_json(json.loads(wire))

    assert canonical_json(proposal.to_json()) == wire
    assert proposal.check_receipt is not None
    assert proposal.check_receipt.schema == 'stateframe-parallel-check/2'

    proposal.check_receipt.product_manifest_modes['src/sample.py'] = 0o644
    changed = proposal.to_json()
    assert changed['check_receipt']['schema'] == 'stateframe-parallel-check/2'
    assert canonical_json(changed) != wire
