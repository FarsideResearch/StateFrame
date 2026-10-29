"""Bind retained-read proposals to source obligations and surviving wire carriers."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from kernel_read_reuse_visibility import (
    _candidate_is_predeclared_missing, _full_line_source_message,
    source_delivery_identity, verify_final_visibility,
)
from kernel_request_seal import RequestSealError


_IDENTITY_FIELDS = ("path", "revision", "start_line", "end_line", "content_sha256")


def _atomic_requirements(candidates):
    identities = []
    boundaries = {}
    for candidate in candidates:
        identity = source_delivery_identity(candidate)
        if identity is None or not isinstance(candidate.get("content"), str):
            raise RequestSealError("visibility requirement has an invalid source identity")
        identities.append((candidate, identity))
        boundaries.setdefault((identity["path"], identity["revision"]), set()).update(
            (identity["start_line"], identity["end_line"] + 1))
    for candidate, identity in identities:
        # A later pair of pages can cover an older window. Segment at proposal
        # boundaries so the allocation still has one exact carrier per unit.
        points = sorted(point for point in boundaries[(identity["path"], identity["revision"])]
                        if identity["start_line"] <= point <= identity["end_line"] + 1)
        lines = candidate["content"].splitlines(keepends=True)
        for start, stop in zip(points, points[1:]):
            content = "".join(lines[start - identity["start_line"]:stop - identity["start_line"]])
            yield {**identity, "start_line": start, "end_line": stop - 1,
                   "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def register_visibility_requirements(plan, candidates):
    """Give each verified read window an omission route in the detached plan."""
    units = list(plan.get("units", ()))
    paths = list(plan.get("paths", ()))
    revisions = dict(plan.get("revisions", {}))
    handles = {row.get("unit_handle") or row.get("handle") for row in units
               if isinstance(row, Mapping)}
    added = []
    for identity in _atomic_requirements(candidates):
        path, revision = identity["path"], identity["revision"]
        if path in revisions and revisions[path] != revision:
            raise RequestSealError("visibility requirement differs from source authority")
        handle = "unit:visibility:" + hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if handle in handles:
            continue
        units.append({
            "path": path, "revision": revision,
            "start_line": identity["start_line"], "end_line": identity["end_line"],
            "unit_handle": handle, "handle": handle,
            "role": "read_reuse_visibility", "reason": "retained read delivery requirement",
            "priority": 1,
        })
        handles.add(handle)
        added.append(handle)
        if path not in paths:
            paths.append(path)
        revisions[path] = revision
    if not added:
        return
    plan.update(units=units, paths=paths, revisions=revisions)
    bounded = plan.get("bounded_delivery")
    if isinstance(bounded, Mapping):
        # These are verified acquired reads, independently selected for this
        # request. Keep the original closure's semantic coverage accounting.
        plan["bounded_delivery"] = {
            **bounded,
            "retained_handles": list(dict.fromkeys((*bounded.get("retained_handles", ()), *added))),
        }


def _identity_key(candidate):
    identity = source_delivery_identity(candidate)
    return tuple(identity[name] for name in _IDENTITY_FIELDS) if identity is not None else None


def _matching_overlap(proposal, carrier):
    if (proposal.get("path"), proposal.get("revision")) != (
            carrier.get("path"), carrier.get("revision")):
        return False
    start = max(proposal["start_line"], carrier["start_line"])
    end = min(proposal["end_line"], carrier["end_line"])
    if start > end:
        return False
    left = proposal["content"].splitlines(keepends=True)
    right = carrier["content"].splitlines(keepends=True)
    return left[start - proposal["start_line"]:end - proposal["start_line"] + 1] == (
        right[start - carrier["start_line"]:end - carrier["start_line"] + 1])


def _covering_selection(proposal, available):
    """Select a contiguous exact-byte cover; ranges alone cannot confer truth."""
    eligible = [row for row in available if _matching_overlap(proposal, row)]
    cursor = proposal["start_line"]
    selected = []
    while cursor <= proposal["end_line"]:
        choices = [row for row in eligible if row["start_line"] <= cursor <= row["end_line"]]
        if not choices:
            return []
        chosen = max(choices, key=lambda row: (row["end_line"], -row["start_line"]))
        selected.append(chosen)
        cursor = chosen["end_line"] + 1
    return selected


def qualify_visibility_requirements(original, selected, delivery, wire, sources):
    """Qualify proposals against exact surviving carriers without editing wire.

    A closure may be COMPLETE via a broader newest tool page after generated
    copies are demoted. The unchanged visibility verifier consumes the actual
    carrier identities; the receipt keeps every original proposal and its cover.
    """
    result = copy.deepcopy(selected)
    proposals = original.get("candidates", ())
    deferred = set(selected.get("deferred_paths", ()))
    missing = selected.get("missing_ranges", ())
    pool = [row for row in selected.get("candidates", ())
            if isinstance(row, Mapping) and isinstance(row.get("content"), str)
            and _identity_key(row) is not None]
    proof = verify_final_visibility(wire, {"candidates": pool, "missing_ranges": []})
    proven = {_identity_key(row) for row in proof["delivered_sources"]}
    needed_paths = {row["path"] for row in pool if _identity_key(row) not in proven
                    and not _candidate_is_predeclared_missing(row, missing)}
    for path, row in delivery.get("paths", {}).items():
        if path not in needed_paths or path in deferred or not isinstance(row, Mapping):
            continue
        source = sources.get(path)
        if not isinstance(source, str):
            continue
        for start, end in row.get("ranges", ()):
            rendered = _full_line_source_message({
                "path": path, "revision": row.get("revision"),
                "start_line": start, "end_line": end,
            }, source)
            if rendered is not None:
                pool.append(rendered[0])
    pool = list({_identity_key(row): row for row in pool}.values())
    if needed_paths:
        proof = verify_final_visibility(wire, {"candidates": pool, "missing_ranges": []})
        proven = {_identity_key(row) for row in proof["delivered_sources"]}
    available = [row for row in pool if _identity_key(row) in proven]
    available_by_key = {_identity_key(row): row for row in available}
    requirements = []
    chosen = []
    chosen_keys = set()
    for proposal in proposals:
        identity = source_delivery_identity(proposal) if isinstance(proposal, Mapping) else None
        requirement = {"proposal": identity, "status": "UNQUALIFIED", "candidates": []}
        if identity is not None and identity["path"] in deferred:
            requirement["status"] = "DEFERRED"
        else:
            cover = []
            if (identity is not None and isinstance(proposal.get("content"), str)
                    and not _candidate_is_predeclared_missing(proposal, missing)):
                exact = available_by_key.get(_identity_key(proposal))
                cover = [exact] if exact is not None else _covering_selection(proposal, available)
            if cover:
                requirement.update(status="QUALIFIED", candidates=[source_delivery_identity(row)
                                                                   for row in cover])
            # Failed inputs remain visible to the unchanged final seal refusal.
            for candidate in cover or [proposal]:
                key = _identity_key(candidate) if isinstance(candidate, Mapping) else None
                if key is None or key not in chosen_keys:
                    chosen.append(copy.deepcopy(candidate))
                    chosen_keys.add(key)
        requirements.append(requirement)
    result["candidates"] = chosen
    result["requirements"] = requirements
    return result
