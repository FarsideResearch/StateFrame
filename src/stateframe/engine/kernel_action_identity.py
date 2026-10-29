"""Canonical proposal identity shared by the dispatcher and execution receipt.

The legacy transport fingerprint remains unchanged for baseline organs. This
identity describes the FULL attempted operation, independent of its outcome.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kernel_retry_gate import AttemptFingerprint


def canonical_edit(owner: object, path: Path, *, dialect: str,
                   old: str, new: str) -> dict[str, object]:
    root = Path(getattr(owner, "root")).resolve()
    resolved = path.resolve()
    try:
        target = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("edit target is outside the task sandbox") from exc
    # Do not strip whitespace: it can change Python indentation/string literals.
    material = json.dumps({"target": target, "dialect": dialect,
                           "old": old, "new": new}, sort_keys=True,
                          ensure_ascii=False).encode()
    digest = "sha256:" + hashlib.sha256(material).hexdigest()
    return AttemptFingerprint(
        action_kind="edit", target=target, edit_digest=digest,
        leading_hypothesis_id=str(getattr(owner, "leading_hypothesis_id", "") or ""),
        read_set_revision=int(getattr(owner, "workspace_revision", 0) or 0),
    ).to_dict()
