"""Deterministic safety signals for function selection and repair authorship.

This module deliberately does not claim to prove a bug is fixed. It identifies
hard contradictions that make a selected function or emitted edit unsafe to
author, while preserving an explicit ``unknown`` outcome for sparse evidence.
"""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_ATTRIBUTE_RE = re.compile(r"\.([a-z_][a-z0-9_]{2,})")

# Only generic language/ticket scaffolding belongs here. Domain vocabulary must
# remain eligible evidence, even when it is a common word in ordinary English.
_STOP = frozenset(
    "the a an and or of to for in on at by is are be was were it its this that "
    "these those with from as if else not but so than then there here what which "
    "who whom find fix bug bugs function functions method responsible should "
    "would could must check checks direction wrong incorrect correct case cases "
    "pass passes fail fails expected value values return returns given code file "
    "files test tests error errors missing none get gets set sets new create "
    "creates update updates make makes when where why how does do doing done use "
    "uses used call calls line lines change changes changed fixed broken issue "
    "problem behaviour behavior"
    .split()
)

_FAMILY_MARKERS = {
    "cmp_flip": re.compile(r"(?:<=|>=|<|>)"),
    "bool_flip": re.compile(r"\b(?:and|or)\b"),
    "boolean_operator": re.compile(r"\b(?:and|or)\b"),
    "none_flip": re.compile(r"\b(?:is|is not)\s+None\b"),
    "none_passthrough_guard": re.compile(r"\b(?:None|continue|return)\b"),
    "arith_flip": re.compile(r"(?<![A-Za-z0-9_])[+\-*/%](?![A-Za-z0-9_=])"),
    "wrong_variable_used": re.compile(r"[A-Za-z_][A-Za-z0-9_]*"),
    "missing_kwarg": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*="),
    "missing_await": re.compile(r"\bawait\b"),
    "early_return_missing": re.compile(r"\b(?:return|raise|break|continue)\b"),
    "wrong_attribute": re.compile(r"\.[A-Za-z_][A-Za-z0-9_]*"),
    "attribute_swap": re.compile(r"\.[A-Za-z_][A-Za-z0-9_]*"),
    "clause_insertion": re.compile(r"\b(?:and|or|if|filter|where)\b"),
    "statement_insert": re.compile(
        r"\b(?:return|raise|yield|continue|break|pass|await)\b|"
        r"[A-Za-z_][A-Za-z0-9_]*\s*="
    ),
    "import_change": re.compile(r"\b(?:from|import)\b"),
    "exception_handler_change": re.compile(r"\b(?:try|except|raise|finally)\b"),
}


def _parts(text: str) -> set[str]:
    """Return lowercase whole and split identifier tokens from source text."""
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        lowered = raw.casefold()
        out.add(lowered)
        for piece in raw.split("_"):
            out.update(part.casefold() for part in _CAMEL_RE.findall(piece))
    return {token for token in out if len(token) > 2}


def explicit_identifier_tokens(task_text: str) -> set[str]:
    """Return task tokens that look like code identifiers or explicit attrs."""
    out = {
        raw.casefold()
        for raw in _TOKEN_RE.findall(task_text or "")
        if "_" in raw or raw[:1].isupper() or raw.startswith("__")
    }
    out.update(match.casefold() for match in _ATTRIBUTE_RE.findall(task_text or ""))
    return out


def symptom_term_tokens(task_text: str, supplied: set[str] | None = None) -> set[str]:
    """Normalize caller-supplied symptom terms, or derive conservative terms."""
    if supplied:
        return {
            token.casefold()
            for term in supplied
            for token in _parts(str(term))
            if token not in _STOP
        }
    return {token for token in _parts(task_text) if token not in _STOP}


@dataclass(frozen=True)
class FunctionEvidence:
    name: str
    identifier_hits: tuple[str, ...]
    term_hits: tuple[str, ...]
    distinctive_term_hits: tuple[str, ...]
    structural_count: int
    support_level: int
    reasons: tuple[str, ...]
    hard_anchor: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "identifier_hits": list(self.identifier_hits),
            "term_hits": list(self.term_hits),
            "distinctive_term_hits": list(self.distinctive_term_hits),
            "structural_count": self.structural_count,
            "support_level": self.support_level,
            "reasons": list(self.reasons),
            "hard_anchor": self.hard_anchor,
        }


def analyze_functions(
    task_text: str,
    sources: dict[str, str],
    *,
    structural_counts: dict[str, int] | None = None,
    symptom_terms: set[str] | None = None,
    pinned: set[str] | None = None,
) -> dict[str, FunctionEvidence]:
    """Build comparable evidence for functions offered by one menu.

    Function-local terms include comments/docstrings because they are part of the
    code's authored contract; a term is *distinctive* only when at most half of
    the offered functions contain it.
    """
    terms = symptom_term_tokens(task_text, symptom_terms)
    explicit = explicit_identifier_tokens(task_text)
    pinned_names = set(pinned or ())
    source_parts = {name: _parts(source) for name, source in sources.items()}
    term_df: dict[str, int] = {}
    for parts in source_parts.values():
        for term in terms & parts:
            term_df[term] = term_df.get(term, 0) + 1
    n_functions = max(1, len(source_parts))
    counts = structural_counts or {}
    structural_names = {
        name for name, count in counts.items() if int(count or 0) > 0
    }

    out: dict[str, FunctionEvidence] = {}
    for name, parts in source_parts.items():
        identifier_hits = tuple(sorted(explicit & parts))
        term_hits = tuple(sorted(terms & parts))
        distinctive = tuple(
            sorted(
                token
                for token in term_hits
                if term_df.get(token, 0) * 2 <= n_functions
            )
        )
        structural_count = max(0, int(counts.get(name, 0) or 0))
        hard_anchor = name in pinned_names
        reasons: list[str] = []
        if hard_anchor:
            reasons.append("hard_anchor")
        if identifier_hits:
            reasons.append("explicit_identifier")
        if distinctive:
            reasons.append("distinctive_symptom_terms")
        if term_hits and not distinctive:
            reasons.append("non_distinctive_symptom_terms")
        if structural_count:
            reasons.append("class_construct")
        level = 0
        if hard_anchor:
            level = 4
        elif identifier_hits:
            level = 3
        elif distinctive:
            level = 2
        elif term_hits or structural_count:
            level = 1
        out[name] = FunctionEvidence(
            name=name,
            identifier_hits=identifier_hits,
            term_hits=term_hits,
            distinctive_term_hits=distinctive,
            structural_count=structural_count,
            support_level=level,
            reasons=tuple(reasons),
            hard_anchor=hard_anchor,
        )

    # A unique class-feasible function is stronger than a generic construct match.
    if len(structural_names) == 1:
        only = next(iter(structural_names))
        evidence = out.get(only)
        if evidence is not None and not evidence.hard_anchor and evidence.support_level < 2:
            out[only] = FunctionEvidence(
                name=evidence.name,
                identifier_hits=evidence.identifier_hits,
                term_hits=evidence.term_hits,
                distinctive_term_hits=evidence.distinctive_term_hits,
                structural_count=evidence.structural_count,
                support_level=2,
                reasons=evidence.reasons + ("unique_class_construct",),
                hard_anchor=evidence.hard_anchor,
            )
    return out


def selection_gate(
    selected: str,
    evidence: dict[str, FunctionEvidence],
) -> dict[str, object]:
    """Detect a selected function dominated by a better-supported alternative.

    ``allow`` with ``unknown`` is intentional: absence of lexical evidence alone
    is not proof of a wrong function, especially for cross-module repairs.
    """
    chosen = evidence.get(selected)
    if chosen is None:
        return {
            "decision": "reject",
            "reason": "selected_function_not_in_evidence_set",
            "selected": selected,
            "alternatives": [],
        }
    alternatives = [item for name, item in evidence.items() if name != selected]
    alternatives.sort(key=lambda item: (-item.support_level, item.name))
    best = alternatives[0] if alternatives else None
    if chosen.hard_anchor:
        reason = "selected_function_has_hard_anchor"
        decision = "allow"
    elif chosen.support_level >= 2:
        reason = "selected_function_has_direct_support"
        decision = "allow"
    elif best is not None and best.support_level >= 2:
        reason = "selected_function_dominated_by_supported_alternative"
        decision = "reject"
    else:
        reason = "function_support_inconclusive"
        decision = "allow"
    return {
        "decision": decision,
        "reason": reason,
        "selected": chosen.as_dict(),
        "strongest_alternative": best.as_dict() if best is not None else None,
        "alternatives": [item.as_dict() for item in alternatives[:8]],
    }


@dataclass(frozen=True)
class PatchEvidence:
    decision: str
    reason: str
    no_op: bool
    family_match: bool | None
    task_overlap: tuple[str, ...]
    changed_tokens: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "no_op": self.no_op,
            "family_match": self.family_match,
            "task_overlap": list(self.task_overlap),
            "changed_tokens": list(self.changed_tokens),
        }


def _changed_tokens(old: str, new: str) -> set[str]:
    return _parts(old) ^ _parts(new)


def _literal_delta(old: str, new: str) -> bool | None:
    try:
        old_tree = ast.parse(textwrap.dedent(old or ""))
        new_tree = ast.parse(textwrap.dedent(new or ""))
    except (SyntaxError, TypeError, ValueError):
        return None
    old_values = [node.value for node in ast.walk(old_tree) if isinstance(node, ast.Constant)]
    new_values = [node.value for node in ast.walk(new_tree) if isinstance(node, ast.Constant)]
    return old_values != new_values


def _family_delta(cls: str | None, old: str, new: str) -> bool | None:
    if not cls:
        return None
    if cls in ("literal_change", "literal_substitution"):
        return _literal_delta(old, new)
    signature_patterns = {
        "cmp_flip": r"<=|>=|==|!=|<|>",
        "bool_flip": r"\b(?:and|or)\b",
        "boolean_operator": r"\b(?:and|or)\b",
        "none_flip": r"\b(?:is not|is)\s+None\b",
        "arith_flip": r"(?<![A-Za-z0-9_])[+\-*/%](?![A-Za-z0-9_=])",
        "wrong_attribute": r"\.[A-Za-z_][A-Za-z0-9_]*",
        "attribute_swap": r"\.[A-Za-z_][A-Za-z0-9_]*",
        "missing_kwarg": r"\b[A-Za-z_][A-Za-z0-9_]*\s*=",
        "missing_await": r"\bawait\b",
        "early_return_missing": r"\b(?:return|raise|break|continue)\b",
        "clause_insertion": r"\b(?:and|or|if|filter|where)\b",
        "statement_insert": (
            r"\b(?:return|raise|yield|continue|break|pass|await)\b|"
            r"[A-Za-z_][A-Za-z0-9_]*\s*="
        ),
        "import_change": r"\b(?:from|import)\b",
        "exception_handler_change": r"\b(?:try|except|raise|finally)\b",
    }
    if cls == "wrong_variable_used":
        return bool(_parts(old) ^ _parts(new))
    pattern = signature_patterns.get(cls)
    if pattern is not None:
        old_signature = tuple(re.findall(pattern, old or ""))
        new_signature = tuple(re.findall(pattern, new or ""))
        # A class witness must describe a delta, not merely survive in the new
        # text. This catches a variable-only edit presented as a comparison
        # flip and a function-definition replacement whose old line happened
        # to contain the diagnosed operator.
        return bool(new_signature) and old_signature != new_signature
    marker = _FAMILY_MARKERS.get(cls)
    if marker is None:
        return None
    # For families without a stable signature, retain a conservative presence
    # check. Looking at the old side as an alternative would accept a
    # replacement such as a function definition over a comparison line.
    return bool(marker.search(new or ""))


def patch_gate(
    old: str,
    new: str,
    *,
    cls: str | None,
    task_text: str,
    symptom_terms: set[str] | None = None,
) -> PatchEvidence:
    """Check whether an authored edit has a class-relevant deterministic delta."""
    old_text, new_text = old or "", new or ""
    no_op = old_text == new_text or old_text.strip() == new_text.strip()
    changed = _changed_tokens(old_text, new_text)
    terms = symptom_term_tokens(task_text, symptom_terms)
    overlap = tuple(sorted(changed & terms))
    if no_op:
        return PatchEvidence("reject", "patch_is_no_op", True, False, overlap, tuple(sorted(changed)))
    family_match = _family_delta(cls, old_text, new_text)
    if family_match is False:
        if not overlap:
            return PatchEvidence(
                "reject",
                "patch_has_no_class_relevant_delta_or_symptom_overlap",
                False,
                False,
                overlap,
                tuple(sorted(changed)),
            )
        return PatchEvidence(
            "allow",
            "patch_has_symptom_token_delta_despite_class_mismatch",
            False,
            False,
            overlap,
            tuple(sorted(changed)),
        )
    if family_match is None:
        return PatchEvidence(
            "allow",
            "patch_family_not_deterministically_classifiable",
            False,
            None,
            overlap,
            tuple(sorted(changed)),
        )
    return PatchEvidence(
        "allow",
        "patch_has_class_relevant_delta",
        False,
        True,
        overlap,
        tuple(sorted(changed)),
    )
