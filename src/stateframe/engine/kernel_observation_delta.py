"""Lossless host observation deltas for world-knowledge replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationDelta:
    """The minimal replay delta; the full fact remains beside this record."""

    fact_set_id: str
    fact_id: str
    new_evidence_ref: str
    ordinal: int
    observation_count_before: int
    observation_count_after: int
    provenance_refs: tuple[str, ...]
    freshness_before: str | None
    freshness_after: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.fact_set_id, self.fact_id, self.new_evidence_ref)):
            raise ValueError("observation delta identities are required")
        if (type(self.ordinal) is not int or type(self.observation_count_before) is not int
                or type(self.observation_count_after) is not int
                or self.ordinal < 1 or self.observation_count_before < 0):
            raise ValueError("observation delta counters are invalid")
        if self.observation_count_after != self.observation_count_before + 1:
            raise ValueError("observation count delta must advance by one")
        if (not isinstance(self.provenance_refs, tuple)
                or not all(isinstance(value, str) and value.strip() for value in self.provenance_refs)):
            raise ValueError("observation provenance must be immutable strings")
        if len(set(self.provenance_refs)) != len(self.provenance_refs):
            raise ValueError("observation provenance must be unique")
        if self.new_evidence_ref not in self.provenance_refs:
            raise ValueError("new evidence must be present in provenance")
        if (self.freshness_before is not None
                and (not isinstance(self.freshness_before, str) or not self.freshness_before.strip())):
            raise ValueError("observation prior freshness is invalid")
        if not isinstance(self.freshness_after, str) or not self.freshness_after.strip():
            raise ValueError("observation freshness is required")

    @staticmethod
    def fact_set_identity(scope: Mapping[str, object], *, path: str,
                          workspace_revision: int, content_digest: str,
                          extractor: str) -> str:
        return "fact-set:" + _digest({
            "scope": dict(scope), "path": path,
            "workspace_revision": workspace_revision,
            "content_digest": content_digest, "extractor": extractor,
        }).removeprefix("sha256:")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "observation-delta/1",
            "fact_set_id": self.fact_set_id,
            "fact_id": self.fact_id,
            "new_evidence_ref": self.new_evidence_ref,
            "ordinal": self.ordinal,
            "observation_count_before": self.observation_count_before,
            "observation_count_after": self.observation_count_after,
            "provenance_refs": list(self.provenance_refs),
            "freshness_before": self.freshness_before,
            "freshness_after": self.freshness_after,
        }

    def replay(self, state: Mapping[str, object] | None = None) -> dict[str, object]:
        """Apply only this delta to replay state; full fact payload stays separate."""
        current = dict(state or {})
        refs = list(current.get("provenance_refs", ()))
        if current.get("fact_set_id") not in (None, self.fact_set_id):
            raise ValueError("observation delta belongs to another fact set")
        if current.get("fact_id") not in (None, self.fact_id):
            raise ValueError("observation delta belongs to another fact")
        if ("observation_count" in current
                and current["observation_count"] != self.observation_count_before):
            raise ValueError("observation delta count does not match replay state")
        if ("last_ordinal" in current
                and int(current["last_ordinal"]) >= self.ordinal):
            raise ValueError("observation delta is duplicate or out of order")
        if self.new_evidence_ref in refs:
            raise ValueError("observation evidence was already admitted")
        for ref in self.provenance_refs:
            if ref not in refs:
                refs.append(ref)
        current.update({
            "fact_set_id": self.fact_set_id,
            "fact_id": self.fact_id,
            "observation_count": self.observation_count_after,
            "provenance_refs": refs,
            "freshness": self.freshness_after,
            "last_evidence_ref": self.new_evidence_ref,
            "last_ordinal": self.ordinal,
        })
        return current


__all__ = ["ObservationDelta"]
