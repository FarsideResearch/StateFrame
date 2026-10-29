"""Catalog-aware ASSUMECHECK adapter.

The legacy request qualification verifier remains available for its Python AST contract.
This module supplies the CATC2 path: assumptions are checked against the
language-neutral structural index, and every answer carries shared content and
coverage handles.  Partial extraction returns UNKNOWN, never a false MISSING.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_catalog_contracts import ContentHandle, CoverageRef, CoverageStatus
from kernel_repo_catalog import StructuralIndex, StructuralIndexError, build_structural_index

_KINDS = frozenset({
    "symbol_exists", "signature", "attribute_exists", "config_key", "caller_passes_arg", "import_path",
})
_STATUS = frozenset({"CONFIRMED", "MISSING", "MISMATCH", "UNKNOWN"})


@dataclass(frozen=True, slots=True)
class CatalogVerificationFact:
    status: str
    kind: str
    target: str
    message: str
    coverage: CoverageRef
    content: ContentHandle | None = None
    matches: tuple[str, ...] = ()
    ambiguous: bool = False
    actual_signature: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUS:
            raise ValueError("unknown catalog verification status")

    def mapping(self) -> dict[str, object]:
        return {
            "status": self.status, "kind": self.kind, "target": self.target,
            "message": self.message, "coverage": self.coverage.to_dict(),
            "content": self.content.to_dict() if self.content is not None else None,
            "matches": list(self.matches), "ambiguous": self.ambiguous,
            "actual_signature": self.actual_signature,
        }


@dataclass(frozen=True, slots=True)
class CatalogVerificationReport:
    facts: tuple[CatalogVerificationFact, ...]
    source: str = "catalog-outline"

    @property
    def status(self) -> str:
        if any(item.status == "MISMATCH" for item in self.facts):
            return "MISMATCH"
        if any(item.status == "MISSING" for item in self.facts):
            return "MISSING"
        if any(item.status == "UNKNOWN" for item in self.facts):
            return "UNKNOWN"
        return "CONFIRMED"

    @property
    def confirmed(self) -> bool:
        return bool(self.facts) and self.status == "CONFIRMED"

    def mapping(self) -> dict[str, object]:
        return {"status": self.status, "source": self.source,
                "facts": [item.mapping() for item in self.facts]}

    def text(self, *, max_bytes: int = 12_000) -> str:
        encoded = json.dumps(self.mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= max_bytes:
            return encoded
        return json.dumps({"status": self.status, "source": self.source, "facts": []}, separators=(",", ":"))

    def rejection_reason(self) -> str:
        if self.confirmed:
            return "catalog assumption verification confirmed"
        details = [f"{item.kind} {item.target}: {item.message}" for item in self.facts if item.status != "CONFIRMED"]
        return "ASSUMPTION_PREFLIGHT: " + " | ".join(details)


def unknown_receipts(
    report: CatalogVerificationReport,
    *,
    required_by: Sequence[str] = (),
    blocking_scope: str = "task",
    sequence: int = 1,
    receipt_prefix: str = "catalog-assumption",
) -> tuple[dict[str, object], ...]:
    """Translate unresolved structural facts into host-admitted catalog receipts.

    The report remains a verifier result; this adapter is the producer boundary
    that gives the pure Catalog A reducer explicit, deduplicable unknown drafts.
    Only ``UNKNOWN`` facts are emitted, so a complete verifier cannot create an
    open task unknown merely because a check was confirmed or definitively absent.
    """
    if not isinstance(report, CatalogVerificationReport):
        raise TypeError("report must be a CatalogVerificationReport")
    scope = str(blocking_scope or "task").strip() or "task"
    required = tuple(str(item).strip() for item in required_by if str(item).strip())
    resolution_by_kind = {
        "import_path": "SEARCH",
        "symbol_exists": "SEARCH",
        "attribute_exists": "READ",
        "signature": "READ",
        "config_key": "READ",
        "caller_passes_arg": "READ",
    }
    receipts: list[dict[str, object]] = []
    for offset, fact in enumerate(report.facts):
        if fact.status != "UNKNOWN":
            continue
        identity = f"{receipt_prefix}:{fact.kind}:{fact.target}:{offset}"
        subject_id = f"assumption:{fact.kind}:{fact.target}"
        receipts.append({
            "receipt_id": identity,
            "seq": sequence + offset,
            "kind": "check",
            "check_id": identity,
            "status": "UNKNOWN",
            "result_class": "UNKNOWN",
            "complete": True,
            "revision": fact.coverage.coverage_id,
            "required_by": list(required),
            "scope": scope,
            "producer": "kernel_catalog_assume",
            "host_admitted": True,
            "unknowns": [{
                "subject": {
                    "entity_id": subject_id,
                    "kind": "assumption",
                    "locator": fact.target,
                },
                "predicate": f"assumption.{fact.kind}",
                "reason": fact.message,
                "resolution": resolution_by_kind.get(fact.kind, "READ"),
                "required_by": list(required),
                "blocking_scope": scope,
            }],
        })
    return tuple(receipts)


def _unknown_coverage(index: StructuralIndex) -> CoverageRef:
    partial = [item for item in index.coverage.values() if item.status is not CoverageStatus.COMPLETE]
    if partial:
        return partial[0]
    return CoverageRef("coverage:assume:unknown", "assumption", CoverageStatus.UNKNOWN)


def _fact(
    index: StructuralIndex,
    status: str,
    kind: str,
    target: str,
    message: str,
    *,
    content: ContentHandle | None = None,
    coverage: CoverageRef | None = None,
    matches: Sequence[str] = (),
    ambiguous: bool = False,
    actual_signature: str | None = None,
) -> CatalogVerificationFact:
    return CatalogVerificationFact(
        status, kind, target, message, coverage or (content and index.coverage.get(content.source_ref)) or _unknown_coverage(index),
        content, tuple(matches), ambiguous, actual_signature,
    )


def _assumptions(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("assumptions must be a sequence")
    if not value or len(value) > 8:
        raise ValueError("assumptions must contain one to eight items")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"assumption {index} must be an object")
        unknown = sorted(str(key) for key in item if str(key) not in {
            "kind", "target", "module", "path", "owner", "type", "expected_signature", "signature",
            "expected_arity", "arity", "caller", "argument", "arg",
        })
        if unknown:
            raise ValueError(f"assumption {index} has unknown field(s): {', '.join(unknown)}")
        if item.get("kind") not in _KINDS or not isinstance(item.get("target"), str) or not str(item["target"]).strip():
            raise ValueError(f"assumption {index} has an invalid kind or target")
        result.append(item)
    return tuple(result)


def _target_path(item: Mapping[str, object]) -> str | None:
    value = item.get("path")
    return str(value).replace("\\", "/") if isinstance(value, str) and value.strip() else None


def _coverage_complete(index: StructuralIndex) -> bool:
    return bool(index.coverage) and all(item.status is CoverageStatus.COMPLETE for item in index.coverage.values())


def _symbols(index: StructuralIndex, target: str, path: str | None) -> tuple[tuple[str, ContentHandle, str], ...]:
    rows: list[tuple[str, ContentHandle, str]] = []
    for module in index.modules:
        if path is not None and module.path != path:
            continue
        for span in index.spans:
            if span.path != module.path or span.kind not in {"function", "type"} or span.name != target:
                continue
            signature = ""
            try:
                signature = index.read_span(span.span_id).body.decode(module.content.encoding, errors="replace").splitlines()[0].strip()
            except (IndexError, UnicodeError, StructuralIndexError):
                signature = span.name
            rows.append((span.span_id, module.content, signature))
    return tuple(rows)


def _arity(signature: str) -> int | None:
    match = re.search(r"\(([^()]*)\)", signature)
    if match is None:
        return None
    body = match.group(1).strip()
    if not body:
        return 0
    return sum(1 for part in body.split(",") if part.strip() and part.strip() not in {"*", "/"})


def _verify_one(index: StructuralIndex, item: Mapping[str, object]) -> CatalogVerificationFact:
    kind = str(item["kind"])
    target = str(item["target"]).strip()
    path = _target_path(item)
    symbols = _symbols(index, target, path)
    if kind == "import_path":
        paths = {module.path for module in index.modules}
        imports = {value for module in index.modules for value in module.imports}
        if target in paths or target in imports or any(target.rstrip(".") in value for value in imports):
            module = next((row for row in index.modules if row.path == target), None)
            return _fact(index, "CONFIRMED", kind, target, "import path is present", content=module.content if module else None)
        return _fact(index, "MISSING" if _coverage_complete(index) else "UNKNOWN", kind, target,
                     "import path is not present in the complete indexed scope")
    if kind == "config_key":
        matches = tuple(item.key for item in index.configs if item.key == target)
        content = next((module.content for module in index.modules if module.module_id == next(
            (row.module_id for row in index.configs if row.key == target), "")), None)
        if matches:
            return _fact(index, "CONFIRMED", kind, target, "configuration key is present", content=content, matches=matches)
        return _fact(index, "MISSING" if _coverage_complete(index) else "UNKNOWN", kind, target,
                     "configuration key was not found in the indexed scope")
    if kind == "attribute_exists":
        owner = str(item.get("owner", item.get("type", "")) or "")
        owners = [row for row in index.types if row.name == owner]
        attributes = [row for row in index.spans if row.kind == "function" and row.name == target
                      and any(owner_row.module_id == next((module.module_id for module in index.modules if module.path == row.path), "")
                              and target in owner_row.attributes for owner_row in owners)]
        if owners and attributes:
            return _fact(index, "CONFIRMED", kind, target, f"attribute/method is present on type {owner}", matches=tuple(row.span_id for row in attributes), ambiguous=len(attributes) > 1)
        return _fact(index, "MISSING" if _coverage_complete(index) else "UNKNOWN", kind, target,
                     f"type {owner!r} or attribute is not present in the indexed scope")
    if kind == "caller_passes_arg":
        caller = str(item.get("caller", item.get("owner", "")) or "")
        argument = str(item.get("argument", item.get("arg", "")) or "")
        for span_id, content, _signature in _symbols(index, caller, path):
            try:
                body = index.read_span(span_id).body.decode(content.encoding, errors="replace")
            except (UnicodeError, StructuralIndexError):
                continue
            if re.search(rf"\b{re.escape(target)}\s*\([^)]*\b{re.escape(argument)}\b", body, re.DOTALL):
                return _fact(index, "CONFIRMED", kind, target, "caller passes the requested argument", content=content)
        return _fact(index, "MISSING" if _coverage_complete(index) else "UNKNOWN", kind, target,
                     "caller argument binding is not established by the indexed structure")
    if not symbols:
        config = next((row for row in index.configs if row.key == target), None)
        if config is not None:
            content = next((module.content for module in index.modules if module.module_id == config.module_id), None)
            return _fact(index, "CONFIRMED", kind, target, "module-level configuration symbol is present", content=content,
                         matches=(config.config_id,))
        return _fact(index, "MISSING" if _coverage_complete(index) else "UNKNOWN", kind, target,
                     "symbol is not present in the indexed scope")
    matches = tuple(row[0] for row in symbols)
    content = symbols[0][1]
    if kind == "signature":
        expected = item.get("expected_signature", item.get("signature"))
        expected_arity = item.get("expected_arity", item.get("arity"))
        if isinstance(expected, str) and not any(signature == expected for _span_id, _content, signature in symbols):
            return _fact(index, "MISMATCH", kind, target, "declared signature differs from indexed source",
                         content=content, matches=matches, ambiguous=len(symbols) > 1, actual_signature=symbols[0][2])
        if isinstance(expected_arity, int) and not any(_arity(signature) == expected_arity for _span_id, _content, signature in symbols):
            return _fact(index, "MISMATCH", kind, target, "declared arity differs from indexed source",
                         content=content, matches=matches, ambiguous=len(symbols) > 1, actual_signature=symbols[0][2])
    message = "symbol is present"
    if len(symbols) > 1:
        message = f"symbol is present with {len(symbols)} ambiguous candidates"
    return _fact(index, "CONFIRMED", kind, target, message, content=content, matches=matches,
                 ambiguous=len(symbols) > 1, actual_signature=symbols[0][2] if kind == "signature" else None)


def verify_indexed_assumptions(
    root: Path,
    raw_assumptions: object,
    *,
    index: StructuralIndex | None = None,
) -> CatalogVerificationReport:
    """Verify assumptions across all OUTLINEGENERIC-supported languages."""

    if raw_assumptions is None or raw_assumptions == []:
        return CatalogVerificationReport((), source="catalog-outline/no-assumptions")
    try:
        assumptions = _assumptions(raw_assumptions)
    except ValueError as exc:
        unknown = CoverageRef("coverage:assume:invalid", "assumption", CoverageStatus.UNKNOWN)
        return CatalogVerificationReport((CatalogVerificationFact("MISMATCH", "assumption_list", "<list>", str(exc), unknown),))
    candidate_paths = tuple(
        path for path in (_target_path(item) for item in assumptions)
        if path is not None and (Path(root) / path).is_file()
    )
    current = index or build_structural_index(root, focus_paths=candidate_paths)
    return CatalogVerificationReport(tuple(_verify_one(current, item) for item in assumptions))


def verify_indexed_edit_references(
    root: Path,
    raw_edits: object,
    *,
    index: StructuralIndex | None = None,
) -> CatalogVerificationReport:
    """Admit helpers introduced in one candidate transaction before writing."""

    if not isinstance(raw_edits, Sequence) or isinstance(raw_edits, (str, bytes, bytearray)):
        unknown = CoverageRef("coverage:edit:invalid", "assumption", CoverageStatus.UNKNOWN)
        return CatalogVerificationReport((CatalogVerificationFact("MISMATCH", "edit_references", "<edits>", "edits must be a sequence", unknown),))
    overrides: dict[str, str] = {}
    for item in raw_edits:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            continue
        path = str(item["path"]).replace("\\", "/")
        old = item.get("old")
        new = item.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        target = (Path(root) / path).resolve()
        if not target.is_relative_to(Path(root).resolve()) or ".." in Path(path).parts:
            continue
        try:
            source = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if source.count(old) == 1:
            overrides[path] = source.replace(old, new, 1)
    if not overrides:
        unknown = CoverageRef("coverage:edit:empty", "assumption", CoverageStatus.UNKNOWN)
        return CatalogVerificationReport((CatalogVerificationFact("MISMATCH", "edit_references", "<edits>", "no valid candidate source was supplied", unknown),))
    baseline = index or build_structural_index(root, focus_paths=tuple(overrides), include_call_graph=False)
    candidate = build_structural_index(root, namespace=baseline.namespace, focus_paths=tuple(overrides), source_overrides=overrides, include_call_graph=False)
    baseline_ids = {span.name for span in baseline.spans if span.kind in {"function", "type"}}
    facts: list[CatalogVerificationFact] = []
    for span in candidate.spans:
        if span.kind not in {"function", "type"} or span.name in baseline_ids:
            continue
        content = candidate.contents[span.content_id]
        facts.append(_fact(candidate, "CONFIRMED", "symbol_exists", span.name,
                           "new same-transaction symbol is admitted by candidate outline",
                           content=content, coverage=span.coverage, matches=(span.span_id,)))
    if not facts:
        facts.append(_fact(candidate, "CONFIRMED", "edit_references", "<edits>",
                           "candidate introduces no unresolved structural symbol reference"))
    return CatalogVerificationReport(tuple(facts), source="catalog-outline/edit")


__all__ = [
    "CatalogVerificationFact", "CatalogVerificationReport", "verify_indexed_assumptions",
    "verify_indexed_edit_references", "unknown_receipts",
]
