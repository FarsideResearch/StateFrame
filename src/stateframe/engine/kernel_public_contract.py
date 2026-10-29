"""Opt-in public text references; provenance never certifies an assertion."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re

from kernel_public_probe import SCHEMA, validate_draft, validate_proposal

PROFILES = {"literal_single", "referenced_suite"}
MAX_FRAGMENTS = 64
MAX_FRAGMENT_CHARS = 2000
MAX_INPUT_BYTES = 64000


def configured_profile() -> str:
    profile = os.environ.get("KERNEL_PUBLIC_REVIEW_CONTRACT", "literal_single")
    if profile not in PROFILES:
        raise ValueError("unknown public review contract profile")
    return profile


def citation_catalog(task: str) -> dict:
    """Exact spans, not semantic requirement extraction. Full task remains visible."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("public task is required for citations")
    source_hash = hashlib.sha256(task.encode()).hexdigest()
    boundaries = [match.end() for match in re.finditer(r"[.;](?=\s)|\n+", task)]
    boundaries.append(len(task))
    fragments = []
    start = 0
    for end in boundaries:
        while start < end:
            while start < end and task[start].isspace():
                start += 1
            stop = min(end, start + MAX_FRAGMENT_CHARS)
            # Long spans split at whitespace when possible, without dropping text.
            if stop < end:
                space = max(task.rfind(" ", start, stop), task.rfind("\t", start, stop))
                if space > start:
                    stop = space
            trimmed = stop
            while trimmed > start and task[trimmed - 1].isspace():
                trimmed -= 1
            if trimmed > start:
                fragments.append({"id": f"req_{source_hash[:12]}_{len(fragments) + 1:03d}",
                                  "start": start, "end": trimmed, "text": task[start:trimmed]})
                if len(fragments) > MAX_FRAGMENTS:
                    raise ValueError("public citation catalog exceeds fragment bound")
            start = stop
    return {"source_sha256": source_hash, "fragments": fragments}


class PublicContract:
    def __init__(self, bundle: dict, profile: str, design: str, review: str):
        if profile not in PROFILES:
            raise ValueError("unknown public review contract profile")
        self.profile = profile
        self.bundle = copy.deepcopy(bundle)
        self.schema = copy.deepcopy(SCHEMA)
        module_note = (
            ' Import only pytest or exact names in public_modules; original source for each '
            'declared module is in public_interface_sources. Do not import acceptance-test modules.'
            if 'public_modules' in self.bundle else '')
        self.design, self.review = design + module_note, review + module_note
        self.max_tests = 1
        self.references = {}
        self.catalog = None
        if profile == "literal_single":
            return
        catalog = citation_catalog(bundle["task"])
        self.catalog = catalog
        self.references = {item["id"]: item["text"] for item in catalog["fragments"]}
        self.bundle["public_citations"] = catalog
        if len(json.dumps(self.bundle).encode()) > MAX_INPUT_BYTES:
            raise ValueError("public citation input exceeds bounded context")
        self.max_tests = 8
        self.schema = {
            "type": "object", "properties": {
                "test_source": {"type": "string", "maxLength": 8000},
                "requirement_refs": {"type": "array", "maxItems": 8, "uniqueItems": True,
                                     "items": {"type": "string", "enum": list(self.references)}},
                "uncertainties": {"type": "array", "maxItems": 8,
                                  "items": {"type": "string", "minLength": 1, "maxLength": 2000}},
            }, "required": ["test_source", "requirement_refs", "uncertainties"],
            "additionalProperties": False,
        }
        output = (
            "Output contract: return exactly one JSON object with test_source, requirement_refs, "
            "and uncertainties. test_source contains complete executable Python source: imports "
            "and one to eight full def test_... functions with their bodies, each containing an "
            "assertion or pytest.raises check. Total source must fit 8000 UTF-8 bytes. "
            "Never return just a function name, path, summary or code reference. "
            "requirement_refs contains up to eight distinct IDs from public_citations.fragments "
            "supporting the asserted outcomes. Select IDs; do not copy or reconstruct quotations. "
            "The kernel resolves IDs to exact public text. These are citation fragments, not "
            "an interpretation or exhaustive decomposition of requirements. Read the full task "
            "for context. uncertainties lists unresolved assumptions, or [] if none. "
            "If no justified probe remains, return empty test_source and requirement_refs. "
            "Return JSON only without markdown fences."
        )
        self.design = (
            "You are a public-contract test designer. Propose bounded self-contained pytest "
            "checks for stated behaviors or interactions not already covered by supplied tests. "
            "Use only the public request and original interface. Do not implement the product "
            f"or invent requirements. Import only the declared {'public_modules' if module_note else 'public_module'} and pytest. "
            "Use deterministic inputs; no fixtures, decorators, helper functions, filesystem, "
            "network, clock, subprocess, dynamic code, skips or xfails. List ambiguity as "
            "uncertainty instead of asserting a guessed answer. Passing generated tests never "
            "prove completeness. " + output + module_note
        )
        self.review = (
            "Audit every proposed assertion and reference against the full public task. "
            "You have not seen candidate code or evaluator results. Remove unsupported or "
            "ambiguous assertions; correct mistakes only where the public requirement supports "
            "the correction. Return complete corrected source and supporting reference IDs. " + output + module_note
        )

    def _canonical(self, value: object) -> dict:
        if not isinstance(value, dict) or set(value) != set(self.schema["required"]):
            raise ValueError("invalid referenced public probe fields")
        refs = value["requirement_refs"]
        if (not isinstance(refs, list) or len(refs) > 8
                or any(not isinstance(ref, str) or ref not in self.references for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError("unknown or duplicate public citation reference")
        return {"test_source": value["test_source"], "uncertainties": value["uncertainties"],
                "requirement_quotes": [self.references[ref] for ref in refs]}

    def draft(self, value: object) -> dict:
        if self.profile == "literal_single":
            return validate_draft(value)
        validate_draft(self._canonical(value))
        return copy.deepcopy(value)

    def final(self, value: object) -> dict:
        canonical = value if self.profile == "literal_single" else self._canonical(value)
        return validate_proposal(canonical, self.bundle["task"],
                                 public_module=self.bundle["public_module"], max_tests=self.max_tests,
                                 public_modules=self.bundle.get("public_modules"))

    def provenance(self, value: object) -> dict | None:
        """Keep selected span identities separate from the canonical executable proposal."""
        if self.profile == "literal_single":
            return None
        self._canonical(value)
        by_id = {item["id"]: item for item in self.catalog["fragments"]}
        return {"source_sha256": self.catalog["source_sha256"], "selected": [
            {"quote_index": index, **{key: by_id[ref][key] for key in ("id", "start", "end")}}
            for index, ref in enumerate(value["requirement_refs"])]}
