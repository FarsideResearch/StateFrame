"""Deterministic contract-atom extraction for the repair kernel.

The nit-picker is deliberately a standalone organ.  It reads observed pytest
output, an in-world test source, and the current target source; it never reads a
fixed source, executes code, or decides how an edit should be written.  Test
evidence is normative, target evidence is observational, and the joiner only
creates a contradiction when those two surfaces can be compared.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from kernel_semantic_ir import SourceSpan, canonical_json
from kernel_liveness import Evidence, OrganCensusSpec, TriState


ANALYZER_VERSION = "nitpicker/1"
MAX_VISIBLE_ATOMS = 3

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
NitTraceCallback: TypeAlias = Callable[[str, str, str], None]


class AtomKind(str, Enum):
    """Small, directional contract vocabulary.

    ``DEFAULT_POLICY`` is an alias rather than a second atom class: the stamp
    autopsy's default-off/explicit-opt-in rule is a policy contract.
    """

    ORDER = "order"
    BOUND = "bound"
    POLARITY = "polarity"
    WINDOW = "window"
    UNIT = "unit"
    END_SELECTION = "end-selection"
    POLICY = "policy"
    DEFAULT_POLICY = "policy"


class AtomState(str, Enum):
    CONTRADICTED = "CONTRADICTED"
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"


class AtomBasis(str, Enum):
    FAILING_ASSERTION = "FAILING_ASSERTION"
    TEST_AST = "TEST_AST"
    TARGET_AST = "TARGET_AST"


class ResultClass(str, Enum):
    NEW = "NEW"
    CONTRADICTED = "CONTRADICTED"
    SUPPORTED = "SUPPORTED"
    DEDUPED = "DEDUPED"
    UNKNOWN = "UNKNOWN"
    DISARMED = "DISARMED"


# Compatibility names used by callers that describe the same ternary state.
ContractState = AtomState
AtomStatus = AtomState


_KIND_PRIORITY = {
    AtomKind.ORDER: 0,
    AtomKind.END_SELECTION: 1,
    AtomKind.WINDOW: 2,
    AtomKind.BOUND: 3,
    AtomKind.POLARITY: 4,
    AtomKind.UNIT: 5,
    AtomKind.POLICY: 6,
}
_BASIS_PRIORITY = {
    AtomBasis.FAILING_ASSERTION: 0,
    AtomBasis.TEST_AST: 1,
    AtomBasis.TARGET_AST: 2,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_key",
        "diff",
        "fixed",
        "fixed_source",
        "fixed_src",
        "gold",
        "gold_src",
        "patch",
        "solution",
    }
)
_INFRA_MARKERS = (
    "connection refused",
    "could not connect",
    "operationalerror",
    "modulenotfounderror",
    "importerror",
    "permissionerror",
    "docker daemon",
    "no tests ran",
    "usage error",
    "fixture .* not found",
    "timed out",
)
_END_WORDS = {
    "first": "first",
    "head": "first",
    "oldest": "oldest",
    "last": "last",
    "tail": "last",
    "newest": "newest",
    "latest": "latest",
    "recent": "recent",
}
_UNIT_WORDS = frozenset(
    {
        "microseconds",
        "milliseconds",
        "seconds",
        "minutes",
        "hours",
        "days",
    }
)
_TRUE_WORDS = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off", "disabled"})


def _normal(value: object) -> object:
    """Return deterministic JSON-compatible data for identity and display."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normal(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normal(item) for item in value), key=lambda item: canonical_json(item))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _normalized(value: object) -> str:
    return canonical_json(_normal(value))


def _clip(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _state_from_value(value: AtomState | str | None) -> AtomState:
    if value is None:
        return AtomState.UNKNOWN
    if isinstance(value, AtomState):
        return value
    normalized = str(value).strip().upper()
    aliases = {
        "FALSE": AtomState.CONTRADICTED,
        "CONTRADICTION": AtomState.CONTRADICTED,
        "TRUE": AtomState.SUPPORTED,
        "UNKNOWN": AtomState.UNKNOWN,
    }
    return aliases.get(normalized, AtomState(normalized))


def _kind_from_value(value: AtomKind | str) -> AtomKind:
    if isinstance(value, AtomKind):
        return value
    return AtomKind(str(value).strip().lower())


def _basis_from_value(value: AtomBasis | str) -> AtomBasis:
    if isinstance(value, AtomBasis):
        return value
    normalized = str(value).strip().upper()
    aliases = {
        "FAIL": AtomBasis.FAILING_ASSERTION,
        "ASSERTION": AtomBasis.FAILING_ASSERTION,
        "TEST": AtomBasis.TEST_AST,
        "TARGET": AtomBasis.TARGET_AST,
    }
    return aliases.get(normalized, AtomBasis(normalized))


def _derive_state(expected: object, observed: object) -> AtomState:
    if expected is None or observed is None:
        return AtomState.UNKNOWN
    if isinstance(observed, str) and observed.strip().lower() in {"unknown", "unspecified?"}:
        return AtomState.UNKNOWN
    return (
        AtomState.SUPPORTED
        if _normal(expected) == _normal(observed)
        else AtomState.CONTRADICTED
    )


def _line(node: object, fallback: int = 1) -> int:
    value = getattr(node, "lineno", fallback)
    return value if isinstance(value, int) and value > 0 else fallback


def _end_line(node: object, fallback: int) -> int:
    value = getattr(node, "end_lineno", fallback)
    return value if isinstance(value, int) and value >= fallback else fallback


def _span(
    path: str,
    node: object,
    *,
    symbol: str,
    role: str,
    fallback_line: int = 1,
) -> SourceSpan:
    start = _line(node, fallback_line)
    return SourceSpan(
        path=path or "source.py",
        start_line=start,
        end_line=_end_line(node, start),
        symbol=symbol,
        role=role,
    )


def _span_key(span: SourceSpan) -> tuple[str, int, int, str, str]:
    return (span.path, span.start_line, span.end_line or span.start_line, span.symbol, span.role)


def _unique_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    result: list[SourceSpan] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    for span in spans:
        key = _span_key(span)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return tuple(result)


def _same_path(left: str, right: str) -> bool:
    left_norm = left.replace("\\", "/").lstrip("./")
    right_norm = right.replace("\\", "/").lstrip("./")
    return left_norm == right_norm or left_norm.endswith("/" + right_norm) or right_norm.endswith("/" + left_norm)


def _span_label(spans: Sequence[SourceSpan]) -> str:
    if not spans:
        return "unknown span"
    span = spans[0]
    end = span.end_line or span.start_line
    return f"{span.path}:{span.start_line}-{end}"


def _rebuild_atom(atom: "ContractAtom", **changes: object) -> "ContractAtom":
    values = {
        "kind": atom.kind,
        "subject": atom.subject,
        "expected": atom.expected,
        "observed": atom.observed,
        "state": atom.state,
        "basis": atom.basis,
        "provenance": atom.provenance,
        "target_spans": atom.target_spans,
        "task_id": atom.task_id,
        "target_symbol": atom.target_symbol,
        "notes": atom.notes,
    }
    values.update(changes)
    return ContractAtom(**values)


@dataclass(frozen=True)
class ContractAtom:
    """One bounded contract comparison with source provenance."""

    kind: AtomKind | str
    subject: str
    expected: JsonValue
    observed: JsonValue = None
    state: AtomState | str = AtomState.UNKNOWN
    basis: AtomBasis | str = AtomBasis.TEST_AST
    provenance: tuple[SourceSpan, ...] = ()
    target_spans: tuple[SourceSpan, ...] = ()
    task_id: str = ""
    target_symbol: str = ""
    contract_key: str = ""
    fingerprint: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = _kind_from_value(self.kind)
        state = _state_from_value(self.state)
        basis = _basis_from_value(self.basis)
        subject = " ".join(str(self.subject).split())
        provenance = _unique_spans(self.provenance)
        target_spans = _unique_spans(self.target_spans)
        if not subject:
            raise ValueError("contract atom subject is required")
        if not provenance:
            raise ValueError(f"contract atom {subject!r} needs provenance")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "expected", _normal(self.expected))
        object.__setattr__(self, "observed", _normal(self.observed))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "target_spans", target_spans)
        object.__setattr__(self, "notes", tuple(_clip(item, 240) for item in self.notes if item))
        contract_key = self.contract_key or _contract_key(
            self.task_id,
            self.target_symbol,
            kind,
            subject,
            self.expected,
        )
        fingerprint = self.fingerprint or _fingerprint(contract_key, self.observed, state)
        object.__setattr__(self, "contract_key", contract_key)
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def span(self) -> SourceSpan:
        return self.provenance[0]

    @property
    def provenance_spans(self) -> tuple[SourceSpan, ...]:
        return self.provenance

    @property
    def is_contradicted(self) -> bool:
        return self.state is AtomState.CONTRADICTED

    @property
    def is_unknown(self) -> bool:
        return self.state is AtomState.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "subject": self.subject,
            "expected": _normal(self.expected),
            "observed": _normal(self.observed),
            "state": self.state.value,
            "basis": self.basis.value,
            "provenance": tuple(span.to_dict() for span in self.provenance),
            "target_spans": tuple(span.to_dict() for span in self.target_spans),
            "task_id": self.task_id,
            "target_symbol": self.target_symbol,
            "contract_key": self.contract_key,
            "fingerprint": self.fingerprint,
            "notes": self.notes,
        }


def _contract_key(
    task_id: str,
    target_symbol: str,
    kind: AtomKind,
    subject: str,
    expected: object,
) -> str:
    material = "|".join(
        (task_id, target_symbol, kind.value, subject, _normalized(expected))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _fingerprint(contract_key: str, observed: object, state: AtomState) -> str:
    material = "|".join((contract_key, _normalized(observed), state.value))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _atom(
    *,
    kind: AtomKind,
    subject: str,
    expected: JsonValue,
    observed: JsonValue = None,
    state: AtomState | str | None = None,
    basis: AtomBasis,
    provenance: Sequence[SourceSpan],
    target_spans: Sequence[SourceSpan] = (),
    task_id: str = "",
    target_symbol: str = "",
    notes: Sequence[str] = (),
) -> ContractAtom:
    resolved_state = _derive_state(expected, observed) if state is None else _state_from_value(state)
    return ContractAtom(
        kind=kind,
        subject=subject,
        expected=expected,
        observed=observed,
        state=resolved_state,
        basis=basis,
        provenance=tuple(provenance),
        target_spans=tuple(target_spans),
        task_id=task_id,
        target_symbol=target_symbol,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class TestRunEvidence:
    """The safe, observed portion of one pytest invocation."""

    __test__ = False

    output: str
    exit_code: int = 1
    test_path: str = ""
    test_line: int | None = None
    nodeid: str = ""
    output_path: str = "pytest-output"
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", str(self.output or ""))
        object.__setattr__(self, "exit_code", int(self.exit_code))
        if self.test_line is not None and self.test_line < 1:
            raise ValueError("test_line must be positive")

    @property
    def is_genuine_red(self) -> bool:
        return self.exit_code == 1 and not self.truncated

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TestRunEvidence":
        output = value.get(
            "output",
            value.get("stdout", value.get("test_output", value.get("text", ""))),
        )
        return cls(
            output=str(output or ""),
            exit_code=int(value.get("exit_code", value.get("returncode", 1))),
            test_path=str(value.get("test_path", value.get("path", "")) or ""),
            test_line=_optional_int(value.get("test_line", value.get("line"))),
            nodeid=str(value.get("nodeid", value.get("node", "")) or ""),
            output_path=str(value.get("output_path", "pytest-output") or "pytest-output"),
            truncated=bool(value.get("truncated", False)),
        )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class NitEvidence:
    """Inputs accepted by the organ; answer-key fields are intentionally absent."""

    task_id: str
    target_file: str
    target_symbol: str
    target_source: str
    test_run: TestRunEvidence
    target_revision: int = 0
    test_source: str = ""
    test_path: str = ""
    test_symbol: str = ""

    def __post_init__(self) -> None:
        for field_name in ("task_id", "target_file", "target_symbol"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        if not isinstance(self.test_run, TestRunEvidence):
            if isinstance(self.test_run, Mapping):
                object.__setattr__(self, "test_run", TestRunEvidence.from_mapping(self.test_run))
            else:
                raise TypeError("test_run must be TestRunEvidence or a mapping")
        object.__setattr__(self, "target_source", str(self.target_source or ""))
        object.__setattr__(self, "test_source", str(self.test_source or ""))
        if not self.test_path:
            object.__setattr__(self, "test_path", self.test_run.test_path)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NitEvidence":
        dropped = tuple(sorted(_forbidden_paths(value)))
        if dropped:
            names = ", ".join(dropped[:4])
            suffix = " …" if len(dropped) > 4 else ""
            raise ValueError(f"answer-key fields rejected: {names}{suffix}")
        target_source = value.get("target_source", value.get("parent_src", ""))
        test_source = value.get("test_source", value.get("oracle_source", ""))
        run_value = value.get("test_run")
        run = run_value if isinstance(run_value, TestRunEvidence) else TestRunEvidence.from_mapping(
            run_value if isinstance(run_value, Mapping) else value
        )
        return cls(
            task_id=str(value.get("task_id", value.get("id", "task"))),
            target_file=str(value.get("target_file", value.get("path", "source.py"))),
            target_symbol=str(value.get("target_symbol", value.get("fn", "target"))),
            target_source=str(target_source or ""),
            test_run=run,
            target_revision=int(value.get("target_revision", value.get("revision", 0))),
            test_source=str(test_source or ""),
            test_path=str(value.get("test_path", "") or ""),
            test_symbol=str(value.get("test_symbol", "") or ""),
        )


def _forbidden_paths(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            path = f"{prefix}.{key}" if prefix else str(key)
            if normalized in _FORBIDDEN_KEYS or any(
                token in normalized for token in ("fixed_src", "gold_src", "answer_key", "patch_text")
            ):
                found.append(path)
            else:
                found.extend(_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


@dataclass(frozen=True)
class EditProposal:
    """Exact edit bytes and/or target spans used by the pre-edit attention check."""

    path: str
    old_source: str = ""
    new_source: str = ""
    changed_spans: tuple[SourceSpan, ...] = ()
    changed_lines: tuple[int, ...] = ()
    target_symbol: str = ""

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("edit path is required")
        spans = _unique_spans(self.changed_spans)
        lines = tuple(sorted({int(line) for line in self.changed_lines if int(line) > 0}))
        if not lines and self.old_source != self.new_source:
            lines = _changed_line_numbers(self.old_source, self.new_source)
        object.__setattr__(self, "changed_spans", spans)
        object.__setattr__(self, "changed_lines", lines)

    @classmethod
    def from_sources(cls, path: str, old_source: str, new_source: str, **kwargs: object) -> "EditProposal":
        return cls(path=path, old_source=old_source, new_source=new_source, **kwargs)

    @property
    def touched_line_numbers(self) -> tuple[int, ...]:
        from_spans = [
            line
            for span in self.changed_spans
            for line in range(span.start_line, (span.end_line or span.start_line) + 1)
        ]
        return tuple(sorted(set(self.changed_lines).union(from_spans)))


def _changed_line_numbers(old_source: str, new_source: str) -> tuple[int, ...]:
    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()
    changed: set[int] = set()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if new_start == new_end:
            changed.add(max(1, new_start))
        else:
            changed.update(range(new_start + 1, new_end + 1))
    return tuple(sorted(changed))


@dataclass(frozen=True)
class EditCheck:
    advisory: str | None
    apply: bool = True
    touches_atom_lines: bool = False
    considered_atoms: tuple[ContractAtom, ...] = ()


@dataclass(frozen=True)
class Confrontation:
    """A capped model-facing view plus the complete deterministic join result."""

    phase: str
    atoms: tuple[ContractAtom, ...]
    all_atoms: tuple[ContractAtom, ...] = ()
    result_class: ResultClass | str = ResultClass.UNKNOWN
    suppressed_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", str(self.phase).upper())
        object.__setattr__(self, "atoms", tuple(self.atoms[:MAX_VISIBLE_ATOMS]))
        object.__setattr__(self, "all_atoms", tuple(self.all_atoms or self.atoms))
        if not isinstance(self.result_class, ResultClass):
            object.__setattr__(self, "result_class", ResultClass(str(self.result_class)))

    @property
    def visible_atoms(self) -> tuple[ContractAtom, ...]:
        return self.atoms

    @property
    def count(self) -> int:
        return len(self.all_atoms)

    def render(self) -> str | None:
        return render_confrontation(self.atoms, phase=self.phase)

    def render_red(self) -> str | None:
        return render_confrontation(self.atoms, phase="RED")

    def render_green(self) -> str | None:
        return render_confrontation(self.atoms, phase="GREEN")


def _changed_path_lines(proposal: EditProposal) -> tuple[tuple[str, int, int], ...]:
    return tuple((proposal.path, line, line) for line in proposal.touched_line_numbers)


def _atom_sort_key(atom: ContractAtom) -> tuple[object, ...]:
    first = atom.provenance[0]
    # Keep an assertion's normative claim ahead of residual TEST_AST noise.
    return (
        _BASIS_PRIORITY[atom.basis],
        0 if _is_assertion_derived(atom) else 1,
        0 if atom.state is AtomState.CONTRADICTED else 1 if atom.state is AtomState.SUPPORTED else 2,
        _KIND_PRIORITY[atom.kind],
        first.path,
        first.start_line,
        atom.subject,
        atom.fingerprint,
    )


def _is_assertion_derived(atom: ContractAtom) -> bool:
    return any(span.role == "test_assertion" for span in atom.provenance)


def rank_atoms(atoms: Iterable[ContractAtom], *, limit: int | None = MAX_VISIBLE_ATOMS) -> tuple[ContractAtom, ...]:
    ranked = tuple(sorted(atoms, key=_atom_sort_key))
    return ranked if limit is None else ranked[: max(0, limit)]


def _atom_match(left: ContractAtom, right: ContractAtom) -> bool:
    if left.kind is not right.kind:
        return False
    if left.subject == right.subject:
        return True
    generic = {"result", "collection", "sequence", "assertion", "target", "rows"}
    return left.subject in generic or right.subject in generic


def _collapse_same_contract(atoms: Iterable[ContractAtom]) -> tuple[ContractAtom, ...]:
    groups: dict[tuple[str, str, str], ContractAtom] = {}
    for atom in sorted(atoms, key=_atom_sort_key):
        key = (atom.kind.value, atom.subject, _normalized(atom.expected))
        current = groups.get(key)
        if current is None:
            groups[key] = atom
            continue
        chosen = min((current, atom), key=_atom_sort_key)
        other = atom if chosen is current else current
        observed = chosen.observed if chosen.observed is not None else other.observed
        merged = _rebuild_atom(
            chosen,
            observed=observed,
            state=_derive_state(chosen.expected, observed),
            provenance=_unique_spans((*chosen.provenance, *other.provenance)),
            target_spans=_unique_spans((*chosen.target_spans, *other.target_spans)),
            notes=(*chosen.notes, *other.notes),
        )
        groups[key] = merged
    return tuple(groups.values())


def _join_all(assertion_atoms: Iterable[ContractAtom], source_atoms: Iterable[ContractAtom]) -> tuple[ContractAtom, ...]:
    assertions = list(_collapse_same_contract(assertion_atoms))
    sources = list(_collapse_same_contract(source_atoms))
    used: set[int] = set()
    joined: list[ContractAtom] = []
    unmatched_assertions: list[ContractAtom] = []
    for assertion in assertions:
        candidates = [
            (index, source)
            for index, source in enumerate(sources)
            if index not in used and _atom_match(assertion, source)
        ]
        if not candidates:
            unmatched_assertions.append(assertion)
            continue
        # Prefer a subject-exact candidate, then the first deterministic source span.
        index, source = min(
            candidates,
            key=lambda item: (
                0 if item[1].subject == assertion.subject else 1,
                _span_key(item[1].provenance[0]),
            ),
        )
        used.add(index)
        observed = assertion.observed if assertion.observed is not None else source.observed
        target_spans = _unique_spans((*assertion.target_spans, *source.target_spans))
        if not target_spans and source.basis is AtomBasis.TARGET_AST:
            target_spans = source.provenance
        joined.append(
            _rebuild_atom(
                assertion,
                observed=observed,
                state=_derive_state(assertion.expected, observed),
                provenance=_unique_spans((*assertion.provenance, *source.provenance)),
                target_spans=target_spans,
                notes=(*assertion.notes, *source.notes),
            )
        )
    # A target often has one observation for several test-side expressions.  A
    # unique source kind can safely attach to one unmatched assertion, but pick
    # the concrete result subject over a setup `range` subject.  Ambiguous
    # observations remain unjoined and therefore cannot manufacture a claim.
    for kind in AtomKind:
        remaining_sources = [
            (index, source)
            for index, source in enumerate(sources)
            if index not in used and source.kind is kind
        ]
        remaining_assertions = [
            atom for atom in unmatched_assertions if atom.kind is kind
        ]
        if len(remaining_sources) == 1 and remaining_assertions:
            source_index, source = remaining_sources[0]
            assertion = min(
                remaining_assertions,
                key=lambda atom: (
                    1 if atom.subject == "range" else 0,
                    1 if atom.subject in {"result", "collection", "assertion"} else 0,
                    _span_key(atom.provenance[0]),
                ),
            )
            used.add(source_index)
            unmatched_assertions.remove(assertion)
            observed = assertion.observed if assertion.observed is not None else source.observed
            target_spans = _unique_spans((*assertion.target_spans, *source.target_spans))
            if not target_spans and source.basis is AtomBasis.TARGET_AST:
                target_spans = source.provenance
            joined.append(
                _rebuild_atom(
                    assertion,
                    observed=observed,
                    state=_derive_state(assertion.expected, observed),
                    provenance=_unique_spans((*assertion.provenance, *source.provenance)),
                    target_spans=target_spans,
                    notes=(*assertion.notes, *source.notes),
                )
            )
    joined.extend(unmatched_assertions)
    joined.extend(source for index, source in enumerate(sources) if index not in used)
    return _collapse_same_contract(joined)


def join_all_atoms(
    assertion_atoms: Iterable[ContractAtom],
    source_atoms: Iterable[ContractAtom],
) -> tuple[ContractAtom, ...]:
    """Join normative assertion atoms with observational target atoms."""

    return _join_all(assertion_atoms, source_atoms)


def join_atoms(
    assertion_atoms: Iterable[ContractAtom],
    source_atoms: Iterable[ContractAtom],
    *,
    max_atoms: int = MAX_VISIBLE_ATOMS,
) -> tuple[ContractAtom, ...]:
    """Return the ranked, capped confrontation surface.

    The full join remains available through :func:`join_all_atoms`; the default
    surface is intentionally no larger than three atoms.
    """

    return rank_atoms(_join_all(assertion_atoms, source_atoms), limit=max_atoms)


def _safe_expr(node: object, limit: int = 120) -> str:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            return repr(_clip(value, 80))
        return _clip(repr(value), limit)
    try:
        return _clip(ast.unparse(node), limit)  # type: ignore[arg-type]
    except (AttributeError, TypeError):
        return type(node).__name__


def _callee(node: object) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _attribute_text(node: object, depth: int = 0) -> str:
    if depth > 8:
        return type(node).__name__
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attribute_text(node.value, depth + 1)}.{node.attr}"
    return _safe_expr(node)


def _subject(node: object) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and _callee(node.func) == "len" and node.args:
        return _subject(node.args[0])
    if isinstance(node, ast.Attribute):
        return _attribute_text(node)
    label = _safe_expr(node, 72)
    return label or "result"


def _literal(node: object) -> tuple[object, bool]:
    try:
        value = ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError, SyntaxError):
        return None, False
    return _normal(value), True


@dataclass(frozen=True)
class _ParamBinding:
    values: tuple[object, ...] | None


def _param_bindings(tree: ast.AST) -> dict[str, _ParamBinding]:
    bindings: dict[str, _ParamBinding] = {}
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call) or _callee(decorator.func) != "parametrize":
                continue
            if len(decorator.args) < 2:
                continue
            names_value, names_ok = _literal(decorator.args[0])
            values_value, values_ok = _literal(decorator.args[1])
            if isinstance(names_value, str):
                names = (names_value,)
            elif isinstance(names_value, list):
                names = tuple(item for item in names_value if isinstance(item, str))
            else:
                names = ()
            if not names:
                continue
            if values_ok and isinstance(values_value, list):
                bindings.update({name: _ParamBinding(tuple(values_value)) for name in names})
            else:
                bindings.update({name: _ParamBinding(None) for name in names})
    return bindings


def _bound_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for element in node.elts for name in _bound_names(element))
    return ()


def _enclosing_scope(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    root: ast.AST,
) -> ast.AST:
    current: ast.AST | None = node
    while current is not None:
        if current is root or isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return root


def _scope_chain(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    root: ast.AST,
) -> tuple[ast.AST, ...]:
    scopes: list[ast.AST] = []
    current: ast.AST | None = node
    while current is not None:
        if current is root or isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(current)
            if current is root:
                break
        current = parents.get(current)
    return tuple(scopes)


def _assignment_values(
    tree: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> dict[int, dict[str, tuple[ast.AST, ...]]]:
    values: dict[int, dict[str, list[ast.AST]]] = {}
    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: tuple[ast.AST, ...] = ()
        if isinstance(node, ast.Assign):
            value = node.value
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = (node.target,)
        elif isinstance(node, ast.NamedExpr):
            value = node.value
            targets = (node.target,)
        if value is None:
            continue
        scope_values = values.setdefault(id(_enclosing_scope(node, parents, tree)), {})
        for target in targets:
            for name in _bound_names(target):
                scope_values.setdefault(name, []).append(value)
    return {
        scope_id: {name: tuple(items) for name, items in scope_values.items()}
        for scope_id, scope_values in values.items()
    }


def _load_names(nodes: Iterable[ast.AST]) -> tuple[str, ...]:
    return tuple(
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _binding_flow_names(value: ast.AST) -> tuple[str, ...]:
    if isinstance(value, ast.Name):
        return (value.id,)
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return tuple(element.id for element in value.elts if isinstance(element, ast.Name))
    if isinstance(value, ast.Call) and _order_value(value) is None:
        return tuple(argument.id for argument in value.args if isinstance(argument, ast.Name))
    return ()


def _looks_like_expected_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and any(
        marker in node.id.lower() for marker in ("expected", "desired", "oracle", "want")
    )


def _assertion_value_roots(test: ast.AST) -> tuple[ast.AST, ...]:
    if not isinstance(test, ast.Compare) or not test.comparators:
        return (test,)
    if len(test.ops) == 1 and _operator_text(test.ops[0]) == "==":
        left, right = test.left, test.comparators[0]
        if isinstance(left, ast.ListComp) or _looks_like_expected_name(left):
            return (left,)
        return (right,)
    return tuple(test.comparators)


def _assertion_contexts(
    tree: ast.AST,
    *,
    assertion_root: ast.AST | None = None,
) -> tuple[tuple[ast.Assert, tuple[ast.AST, ...]], ...]:
    """Return assertion expressions plus assignment values that feed them."""

    parents = _target_parents(tree)
    assignment_values = _assignment_values(tree, parents)
    contexts: list[tuple[ast.Assert, tuple[ast.AST, ...]]] = []
    scope = tree if assertion_root is None else assertion_root
    for assertion in ast.walk(scope):
        if not isinstance(assertion, ast.Assert):
            continue
        nodes: dict[int, ast.AST] = {}
        scopes = _scope_chain(assertion, parents, tree)

        def add_nodes(root: ast.AST) -> None:
            for child in ast.walk(root):
                nodes.setdefault(id(child), child)

        add_nodes(assertion.test)
        pending = list(_load_names(_assertion_value_roots(assertion.test)))
        seen_names: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            values: tuple[ast.AST, ...] = ()
            for scope in scopes:
                scoped_values = assignment_values.get(id(scope), {})
                if name in scoped_values:
                    values = scoped_values[name]
                    break
            for value in values:
                add_nodes(value)
                pending.extend(_binding_flow_names(value))
        contexts.append((assertion, tuple(nodes.values())))
    return tuple(contexts)


def _value(node: object, bindings: Mapping[str, _ParamBinding]) -> tuple[object, bool]:
    literal, is_literal = _literal(node)
    if is_literal:
        return literal, True
    if isinstance(node, ast.Name) and node.id in bindings:
        binding = bindings[node.id]
        if binding.values is not None:
            return {"parameter": node.id, "values": list(binding.values)}, True
        return {"parameter": node.id}, False
    return _safe_expr(node, 96), False


def _operator_text(operator: ast.AST) -> str:
    operators: dict[type[ast.AST], str] = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.In: "in",
        ast.NotIn: "not in",
        ast.Is: "is",
        ast.IsNot: "is not",
    }
    for kind, text in operators.items():
        if isinstance(operator, kind):
            return text
    return type(operator).__name__.lower()


def _range_spec(call: ast.Call, bindings: Mapping[str, _ParamBinding]) -> tuple[dict[str, object], bool, int | None]:
    if not 1 <= len(call.args) <= 3:
        return {}, False, None
    values = [_value(arg, bindings) for arg in call.args]
    if len(values) == 1:
        start: object = 0
        stop, stop_known = values[0]
        step: object = 1
        known = stop_known
    elif len(values) == 2:
        start, start_known = values[0]
        stop, stop_known = values[1]
        step = 1
        known = start_known and stop_known
    else:
        start, start_known = values[0]
        stop, stop_known = values[1]
        step, step_known = values[2]
        known = start_known and stop_known and step_known
    cardinality: int | None = None
    if known and all(isinstance(item, int) and not isinstance(item, bool) for item in (start, stop, step)) and step != 0:
        try:
            cardinality = len(range(start, stop, step))
        except (OverflowError, ValueError):
            cardinality = None
    spec = {
        "start": start,
        "stop": stop,
        "step": step,
        "stop_inclusive": False,
    }
    if cardinality is not None:
        spec["cardinality"] = cardinality
    return spec, known, cardinality


def _range_direction(spec: Mapping[str, object]) -> str | None:
    step = spec.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        if step < 0:
            return "descending"
        if step > 0:
            return "ascending"
    return None


def _range_calls(node: ast.AST) -> tuple[ast.Call, ...]:
    return tuple(
        child
        for child in sorted(ast.walk(node), key=lambda item: (_line(item), getattr(item, "col_offset", 0), type(item).__name__))
        if isinstance(child, ast.Call) and _callee(child.func) == "range"
    )


def _listcomp_contract(
    node: ast.ListComp,
    *,
    subject: str,
    path: str,
    target_symbol: str,
    task_id: str,
    bindings: Mapping[str, _ParamBinding],
    handled_ranges: set[int],
) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for generator in node.generators:
        if not isinstance(generator.iter, ast.Call) or _callee(generator.iter.func) != "range":
            continue
        call = generator.iter
        handled_ranges.add(id(call))
        spec, known, cardinality = _range_spec(call, bindings)
        provenance = (_span(path, node, symbol=target_symbol, role="test_assertion"), _span(path, call, symbol=target_symbol, role="range_contract"))
        atoms.append(
            _atom(
                kind=AtomKind.BOUND,
                subject=subject,
                expected=spec,
                state=AtomState.UNKNOWN if not known else AtomState.UNKNOWN,
                basis=AtomBasis.TEST_AST,
                provenance=provenance,
                task_id=task_id,
                target_symbol=target_symbol,
                notes=("range stop is exclusive",),
            )
        )
        direction = _range_direction(spec)
        if direction is not None:
            atoms.append(
                _atom(
                    kind=AtomKind.ORDER,
                    subject=subject,
                    expected=direction,
                    basis=AtomBasis.TEST_AST,
                    provenance=provenance,
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
            if cardinality is not None:
                end = "high-index/recent end" if direction == "descending" else "low-index end"
                atoms.append(
                    _atom(
                        kind=AtomKind.END_SELECTION,
                        subject=subject,
                        expected=end,
                        basis=AtomBasis.TEST_AST,
                        provenance=provenance,
                        task_id=task_id,
                        target_symbol=target_symbol,
                    )
                )
    return atoms


def _slice_parts(node: ast.Subscript, bindings: Mapping[str, _ParamBinding]) -> tuple[object, object, object, bool]:
    slice_node = node.slice
    if isinstance(slice_node, ast.Slice):
        start, start_known = (None, True) if slice_node.lower is None else _value(slice_node.lower, bindings)
        stop, stop_known = (None, True) if slice_node.upper is None else _value(slice_node.upper, bindings)
        step, step_known = (None, True) if slice_node.step is None else _value(slice_node.step, bindings)
        return start, stop, step, start_known and stop_known and step_known
    return _safe_expr(slice_node), None, None, False


def _slice_atoms(
    node: ast.Subscript,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    bindings: Mapping[str, _ParamBinding],
    basis: AtomBasis,
    target_spans: Sequence[SourceSpan] = (),
) -> list[ContractAtom]:
    start, stop, step, known = _slice_parts(node, bindings)
    subject = _subject(node.value)
    provenance = (_span(path, node, symbol=target_symbol, role="slice_contract"),)
    observed_or_expected: dict[str, object] = {
        "start": start,
        "stop": stop,
        "step": step,
        "stop_inclusive": False,
    }
    if isinstance(start, int) and start < 0 and stop is None:
        observed_or_expected["count"] = abs(start)
        selection = "last"
    elif isinstance(stop, int) and stop >= 0 and start is None:
        observed_or_expected["count"] = stop
        selection = "first"
    else:
        selection = None
    atoms = [
        _atom(
            kind=AtomKind.BOUND,
            subject=subject,
            expected=observed_or_expected if basis is AtomBasis.TEST_AST else None,
            observed=observed_or_expected if basis is AtomBasis.TARGET_AST else None,
            state=AtomState.UNKNOWN,
            basis=basis,
            provenance=provenance,
            target_spans=target_spans,
            task_id=task_id,
            target_symbol=target_symbol,
            notes=("slice stop is exclusive",),
        )
    ]
    window_value = dict(observed_or_expected)
    if selection is not None:
        window_value["selection"] = selection
    atoms.append(
        _atom(
            kind=AtomKind.WINDOW,
            subject=subject,
            expected=window_value if basis is AtomBasis.TEST_AST else None,
            observed=window_value if basis is AtomBasis.TARGET_AST else None,
            state=AtomState.UNKNOWN,
            basis=basis,
            provenance=provenance,
            target_spans=target_spans,
            task_id=task_id,
            target_symbol=target_symbol,
        )
    )
    if selection is not None:
        atoms.append(
            _atom(
                kind=AtomKind.END_SELECTION,
                subject=subject,
                expected=selection if basis is AtomBasis.TEST_AST else None,
                observed=selection if basis is AtomBasis.TARGET_AST else None,
                state=AtomState.UNKNOWN,
                basis=basis,
                provenance=provenance,
                target_spans=target_spans,
                task_id=task_id,
                target_symbol=target_symbol,
            )
        )
    if isinstance(step, int) and step < 0:
        atoms.append(
            _atom(
                kind=AtomKind.ORDER,
                subject=subject,
                expected="descending" if basis is AtomBasis.TEST_AST else None,
                observed="descending" if basis is AtomBasis.TARGET_AST else None,
                state=AtomState.UNKNOWN,
                basis=basis,
                provenance=provenance,
                target_spans=target_spans,
                task_id=task_id,
                target_symbol=target_symbol,
            )
        )
    if not known:
        atoms = [_rebuild_atom(atom, notes=(*atom.notes, "dynamic slice bound remains UNKNOWN")) for atom in atoms]
    return atoms


def _call_window_atom(
    call: ast.Call,
    *,
    method: str,
    path: str,
    target_symbol: str,
    task_id: str,
    bindings: Mapping[str, _ParamBinding],
    basis: AtomBasis,
    target_spans: Sequence[SourceSpan] = (),
) -> ContractAtom | None:
    if not call.args:
        return None
    value, known = _value(call.args[0], bindings)
    payload: dict[str, object] = {method: value}
    if not known:
        payload["dynamic"] = True
    return _atom(
        kind=AtomKind.WINDOW,
        subject=_subject(call.func.value) if isinstance(call.func, ast.Attribute) else "result",
        expected=payload if basis is AtomBasis.TEST_AST else None,
        observed=payload if basis is AtomBasis.TARGET_AST else None,
        state=AtomState.UNKNOWN,
        basis=basis,
        provenance=(_span(path, call, symbol=target_symbol, role=f"{method}_contract"),),
        target_spans=target_spans,
        task_id=task_id,
        target_symbol=target_symbol,
    )


def _order_value(call: ast.Call) -> str | None:
    name = _callee(call.func)
    if name == "reversed":
        return "descending"
    if name == "sorted" or name == "sort":
        reverse = next((keyword.value for keyword in call.keywords if keyword.arg == "reverse"), None)
        value, known = _literal(reverse) if reverse is not None else (False, True)
        if known and isinstance(value, bool):
            return "descending" if value else "ascending"
        return "unspecified"
    if name in {"order_by", "order", "desc", "descending"}:
        text = " ".join(_safe_expr(arg, 80).lower() for arg in call.args)
        text += " " + " ".join(_safe_expr(keyword.value, 80).lower() for keyword in call.keywords)
        if "desc" in text or name in {"desc", "descending"}:
            return "descending"
        if "asc" in text:
            return "ascending"
        return "unspecified"
    return None


def _order_atoms(
    tree: ast.AST,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    basis: AtomBasis,
    target_spans: Sequence[SourceSpan] = (),
    allowed_node_ids: Collection[int] | None = None,
    assertion_context: bool = False,
) -> list[ContractAtom]:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (allowed_node_ids is None or id(node) in allowed_node_ids)
        and _order_value(node) is not None
    ]
    calls.sort(key=lambda node: (_line(node), getattr(node, "col_offset", 0), type(node).__name__))
    if not calls:
        if basis is AtomBasis.TARGET_AST:
            return [
                _atom(
                    kind=AtomKind.ORDER,
                    subject="result",
                    expected=None,
                    observed="unspecified",
                    state=AtomState.UNKNOWN,
                    basis=basis,
                    provenance=(_span(path, tree, symbol=target_symbol, role="missing_order"),),
                    target_spans=target_spans,
                    task_id=task_id,
                    target_symbol=target_symbol,
                    notes=("missing order is unspecified, never assumed ascending",),
                )
            ]
        return []
    result: list[ContractAtom] = []
    for call in calls:
        direction = _order_value(call)
        if direction is None:
            continue
        subject = _subject(call.args[0]) if call.args else "result"
        span = _span(path, call, symbol=target_symbol, role="order_contract")
        provenance: tuple[SourceSpan, ...] = (span,)
        if assertion_context:
            provenance = (_span(path, call, symbol=target_symbol, role="test_assertion"), span)
        result.append(
            _atom(
                kind=AtomKind.ORDER,
                subject=subject,
                expected=direction if basis is AtomBasis.TEST_AST else None,
                observed=direction if basis is AtomBasis.TARGET_AST else None,
                state=AtomState.UNKNOWN,
                basis=basis,
                provenance=provenance,
                target_spans=target_spans,
                task_id=task_id,
                target_symbol=target_symbol,
            )
        )
    return result


def _comparison_atoms(
    tree: ast.AST,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    basis: AtomBasis,
    bindings: Mapping[str, _ParamBinding],
    target_spans: Sequence[SourceSpan] = (),
) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    comparisons.sort(key=lambda node: (_line(node), getattr(node, "col_offset", 0)))
    for compare in comparisons:
        left = compare.left
        for operator, right in zip(compare.ops, compare.comparators):
            text = _operator_text(operator)
            right_value, right_known = _literal(right)
            polarity = (
                f"{text} None"
                if text in {"is", "is not"} and right_known and right_value is None
                else text
            )
            subject = _subject(left)
            provenance = (_span(path, compare, symbol=target_symbol, role="comparison_contract"),)
            atoms.append(
                _atom(
                    kind=AtomKind.POLARITY,
                    subject=subject,
                    expected=polarity if basis is AtomBasis.TEST_AST else None,
                    observed=polarity if basis is AtomBasis.TARGET_AST else None,
                    state=AtomState.UNKNOWN,
                    basis=basis,
                    provenance=provenance,
                    target_spans=target_spans,
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
            if text in {"<", "<=", ">", ">="}:
                value, known = _value(right, bindings)
                expected = {
                    "operator": text,
                    "value": value,
                    "inclusive": text in {"<=", ">="},
                }
                atoms.append(
                    _atom(
                        kind=AtomKind.BOUND,
                        subject=subject,
                        expected=expected if basis is AtomBasis.TEST_AST else None,
                        observed=expected if basis is AtomBasis.TARGET_AST else None,
                        state=AtomState.UNKNOWN,
                        basis=basis,
                        provenance=provenance,
                        target_spans=target_spans,
                        task_id=task_id,
                        target_symbol=target_symbol,
                        notes=("dynamic comparison bound remains UNKNOWN",) if not known else (),
                    )
                )
            left = right
    return atoms


def _assert_window_atoms(
    node: ast.Assert,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    bindings: Mapping[str, _ParamBinding],
    range_cardinalities: Sequence[int],
    context_nodes: Sequence[ast.AST] | None = None,
) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    assertion_span = _span(path, node, symbol=target_symbol, role="test_assertion")
    nodes = tuple(context_nodes) if context_nodes is not None else tuple(ast.walk(node))
    for child in nodes:
        if isinstance(child, ast.Call) and _callee(child.func) in {"limit", "offset"}:
            atom = _call_window_atom(
                child,
                method=_callee(child.func),
                path=path,
                target_symbol=target_symbol,
                task_id=task_id,
                bindings=bindings,
                basis=AtomBasis.TEST_AST,
            )
            if atom:
                atoms.append(_rebuild_atom(atom, provenance=(assertion_span, *atom.provenance)))
        elif isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Slice):
            atoms.extend(
                _rebuild_atom(atom, provenance=(assertion_span, *atom.provenance))
                for atom in _slice_atoms(
                    child,
                    path=path,
                    target_symbol=target_symbol,
                    task_id=task_id,
                    bindings=bindings,
                    basis=AtomBasis.TEST_AST,
                )
            )
    for compare in (child for child in nodes if isinstance(child, ast.Compare)):
        if len(compare.ops) != 1 or _operator_text(compare.ops[0]) != "==":
            continue
        left, right = compare.left, compare.comparators[0]
        len_node = left if isinstance(left, ast.Call) and _callee(left.func) == "len" else right if isinstance(right, ast.Call) and _callee(right.func) == "len" else None
        if len_node is None or not len_node.args:
            continue
        other = right if len_node is left else left
        expected_value, known = _value(other, bindings)
        payload: dict[str, object] = {"count": expected_value}
        if not known:
            payload["dynamic"] = True
        if range_cardinalities and known and expected_value in range_cardinalities:
            payload["phase"] = "after-ordering"
        atoms.append(
            _atom(
                kind=AtomKind.WINDOW,
                subject=_subject(len_node),
                expected=payload,
                state=AtomState.UNKNOWN,
                basis=AtomBasis.TEST_AST,
                provenance=(assertion_span, _span(path, node, symbol=target_symbol, role="length_contract")),
                task_id=task_id,
                target_symbol=target_symbol,
                notes=("cap follows ordering when paired with the expected ordered cardinality",) if "phase" in payload else (),
            )
        )
    return atoms


def _test_unit_atoms(tree: ast.AST, *, path: str, target_symbol: str, task_id: str) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node.func)
        if name == "timedelta":
            units = [keyword.arg for keyword in node.keywords if keyword.arg in _UNIT_WORDS]
            if units:
                atoms.append(
                    _atom(
                        kind=AtomKind.UNIT,
                        subject="duration",
                        expected=units[0],
                        basis=AtomBasis.TEST_AST,
                        provenance=(_span(path, node, symbol=target_symbol, role="unit_contract"),),
                        task_id=task_id,
                        target_symbol=target_symbol,
                    )
                )
        elif name in {"sleep", "wait", "timeout"} and node.args:
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="duration",
                    expected="seconds",
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="unit_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        elif name in {"ZoneInfo", "timezone"} or (
            isinstance(node.func, ast.Attribute) and node.func.attr in {"astimezone", "now"}
        ):
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="timezone",
                    expected={"timezone_aware": True},
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="timezone_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr.lower() in {"utc", "gmt", "utcnow"}:
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="timezone",
                    expected={"timezone_aware": True},
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="timezone_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.lower() in {"utc", "gmt", "utcnow"}:
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="timezone",
                    expected={"timezone_aware": True},
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="timezone_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
    return atoms


def _target_unit_atoms(tree: ast.AST, *, path: str, target_symbol: str, task_id: str) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node.func)
        if name == "timedelta":
            units = [keyword.arg for keyword in node.keywords if keyword.arg in _UNIT_WORDS]
            if units:
                atoms.append(
                    _atom(
                        kind=AtomKind.UNIT,
                        subject="duration",
                        expected=None,
                        observed=units[0],
                        basis=AtomBasis.TARGET_AST,
                        state=AtomState.UNKNOWN,
                        provenance=(_span(path, node, symbol=target_symbol, role="unit_observation"),),
                        target_spans=(_span(path, node, symbol=target_symbol, role="unit_target"),),
                        task_id=task_id,
                        target_symbol=target_symbol,
                    )
                )
        elif name in {"sleep", "wait", "timeout"} and node.args:
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="duration",
                    expected=None,
                    observed="seconds",
                    basis=AtomBasis.TARGET_AST,
                    state=AtomState.UNKNOWN,
                    provenance=(_span(path, node, symbol=target_symbol, role="unit_observation"),),
                    target_spans=(_span(path, node, symbol=target_symbol, role="unit_target"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        if name in {"ZoneInfo", "timezone"} or (
            isinstance(node.func, ast.Attribute) and node.func.attr in {"astimezone", "now"}
        ):
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="timezone",
                    expected=None,
                    observed={"timezone_aware": True},
                    basis=AtomBasis.TARGET_AST,
                    state=AtomState.UNKNOWN,
                    provenance=(_span(path, node, symbol=target_symbol, role="timezone_observation"),),
                    target_spans=(_span(path, node, symbol=target_symbol, role="timezone_target"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.lower() in {"utc", "gmt", "utcnow"}:
            span = _span(path, node, symbol=target_symbol, role="timezone_observation")
            atoms.append(
                _atom(
                    kind=AtomKind.UNIT,
                    subject="timezone",
                    expected=None,
                    observed={"timezone_aware": True},
                    basis=AtomBasis.TARGET_AST,
                    state=AtomState.UNKNOWN,
                    provenance=(span,),
                    target_spans=(span,),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
    return atoms


def _policy_flag(node: ast.Call) -> str | None:
    name = _callee(node.func)
    if name == "getenv" and node.args:
        value, known = _literal(node.args[0])
        return str(value) if known and isinstance(value, str) else _safe_expr(node.args[0], 80)
    if name == "get" and node.args and isinstance(node.func, ast.Attribute):
        owner = _attribute_text(node.func.value).lower()
        if "environ" not in owner and "config" not in owner and "settings" not in owner:
            return None
        value, known = _literal(node.args[0])
        return str(value) if known and isinstance(value, str) else _safe_expr(node.args[0], 80)
    return None


def _default_value(call: ast.Call) -> object | None:
    if len(call.args) >= 2:
        value, known = _literal(call.args[1])
        if known:
            return value
    for keyword in call.keywords:
        if keyword.arg in {"default", "fallback"}:
            value, known = _literal(keyword.value)
            if known:
                return value
    return None


def _policy_atoms_test(tree: ast.AST, *, path: str, target_symbol: str, task_id: str) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"delenv", "setenv"} and node.args:
            flag, known = _literal(node.args[0])
            if not known or not isinstance(flag, str):
                continue
            if node.func.attr == "delenv":
                expected: JsonValue = {"flag": flag, "default": "off", "opt_in": True}
            else:
                value, value_known = _literal(node.args[1]) if len(node.args) > 1 else (None, False)
                expected = {"flag": flag, "opt_in": value if value_known else "enabled"}
            atoms.append(
                _atom(
                    kind=AtomKind.POLICY,
                    subject=flag,
                    expected=expected,
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="policy_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        elif isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            compare = node.test
            op = _operator_text(compare.ops[0])
            if op not in {"is", "=="}:
                continue
            right = compare.comparators[0]
            right_value, right_known = _literal(right)
            left_label = _safe_expr(compare.left, 80).lower()
            if not right_known or right_value is not False or not (
                any(word in left_label for word in ("enabled", "active", "flag", "metadata"))
                or any(word in _safe_expr(node, 120).lower() for word in ("default", "opt_in", "opt-in", "ships_dark"))
            ):
                continue
            atoms.append(
                _atom(
                    kind=AtomKind.POLICY,
                    subject=_subject(compare.left),
                    expected={"default": "off", "opt_in": True},
                    basis=AtomBasis.TEST_AST,
                    provenance=(_span(path, node, symbol=target_symbol, role="policy_contract"),),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
    return atoms


def _policy_atoms_target(tree: ast.AST, *, path: str, target_symbol: str, task_id: str) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            flag = _policy_flag(node)
            if flag is not None:
                default = _default_value(node)
                if isinstance(default, str):
                    lowered = default.strip().lower()
                    state_label = "on" if lowered in _TRUE_WORDS else "off" if lowered in _FALSE_WORDS else "unspecified"
                elif isinstance(default, bool):
                    state_label = "on" if default else "off"
                else:
                    state_label = "unspecified"
                span = _span(path, node, symbol=target_symbol, role="policy_observation")
                atoms.append(
                    _atom(
                        kind=AtomKind.POLICY,
                        subject=flag,
                        expected=None,
                        observed={"flag": flag, "default": state_label},
                        state=AtomState.UNKNOWN,
                        basis=AtomBasis.TARGET_AST,
                        provenance=(span,),
                        target_spans=(span,),
                        task_id=task_id,
                        target_symbol=target_symbol,
                    )
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                label = _safe_expr(target, 80)
                if not any(word in label.lower() for word in ("enabled", "active", "flag")):
                    continue
                span = _span(path, node, symbol=target_symbol, role="policy_observation")
                atoms.append(
                    _atom(
                        kind=AtomKind.POLICY,
                        subject=label,
                        expected=None,
                        observed={"flag": label, "default": "on" if value.value else "off"},
                        state=AtomState.UNKNOWN,
                        basis=AtomBasis.TARGET_AST,
                        provenance=(span,),
                        target_spans=(span,),
                        task_id=task_id,
                        target_symbol=target_symbol,
                    )
                )
    return atoms


def _end_atoms_target(
    tree: ast.AST,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    order_direction: str | None,
    has_window: bool,
) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        label = ""
        if isinstance(node, ast.Name):
            label = node.id.lower()
        elif isinstance(node, ast.Attribute):
            label = node.attr.lower()
        mapped = _END_WORDS.get(label)
        if mapped is None:
            continue
        span = _span(path, node, symbol=target_symbol, role="end_selection_observation")
        atoms.append(
            _atom(
                kind=AtomKind.END_SELECTION,
                subject="result",
                expected=None,
                observed=mapped,
                basis=AtomBasis.TARGET_AST,
                state=AtomState.UNKNOWN,
                provenance=(span,),
                target_spans=(span,),
                task_id=task_id,
                target_symbol=target_symbol,
            )
        )
    if has_window and not atoms:
        observed = {
            "ascending": "low-index/oldest end",
            "descending": "high-index/recent end",
        }.get(order_direction or "", "unspecified end")
        span = _span(path, tree, symbol=target_symbol, role="end_selection_inference")
        atoms.append(
            _atom(
                kind=AtomKind.END_SELECTION,
                subject="result",
                expected=None,
                observed=observed,
                basis=AtomBasis.TARGET_AST,
                state=AtomState.UNKNOWN,
                provenance=(span,),
                target_spans=(span,),
                task_id=task_id,
                target_symbol=target_symbol,
                notes=("end is inferred from ordering plus finite window",),
            )
        )
    return atoms


def _target_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _ancestor_calls(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> tuple[ast.Call, ...]:
    result: list[ast.Call] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Call):
            result.append(current)
        current = parents.get(current)
    return tuple(result)


def _target_window_phase(node: ast.AST, parents: Mapping[ast.AST, ast.AST], order_calls: Sequence[ast.Call]) -> str:
    descendant_orders = [
        call
        for call in ast.walk(node)
        if call is not node and isinstance(call, ast.Call) and _order_value(call) is not None
    ]
    if descendant_orders:
        return "after-ordering"
    ancestor_orders = [call for call in _ancestor_calls(node, parents) if _order_value(call) is not None]
    if ancestor_orders:
        return "before-ordering"
    line = _line(node)
    prior = [call for call in order_calls if _line(call) < line]
    return "after-ordering" if prior else "before-ordering"


def _target_windows(
    tree: ast.AST,
    *,
    path: str,
    target_symbol: str,
    task_id: str,
    parents: Mapping[ast.AST, ast.AST],
    order_calls: Sequence[ast.Call],
) -> list[ContractAtom]:
    atoms: list[ContractAtom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            start, stop, step, _known = _slice_parts(node, {})
            payload: dict[str, object] = {
                "start": start,
                "stop": stop,
                "step": step,
                "stop_inclusive": False,
                "phase": _target_window_phase(node, parents, order_calls),
            }
            if isinstance(start, int) and start < 0 and stop is None:
                payload["count"] = abs(start)
            elif isinstance(stop, int) and stop >= 0 and start is None:
                payload["count"] = stop
            span = _span(path, node, symbol=target_symbol, role="window_observation")
            atoms.append(
                _atom(
                    kind=AtomKind.WINDOW,
                    subject=_subject(node.value),
                    expected=None,
                    observed=payload,
                    basis=AtomBasis.TARGET_AST,
                    state=AtomState.UNKNOWN,
                    provenance=(span,),
                    target_spans=(span,),
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        elif isinstance(node, ast.Call) and _callee(node.func) in {"limit", "offset"}:
            atom = _call_window_atom(
                node,
                method=_callee(node.func),
                path=path,
                target_symbol=target_symbol,
                task_id=task_id,
                bindings={},
                basis=AtomBasis.TARGET_AST,
                target_spans=(_span(path, node, symbol=target_symbol, role="window_target"),),
            )
            if atom:
                phase = _target_window_phase(node, parents, order_calls)
                observed = dict(atom.observed) if isinstance(atom.observed, Mapping) else {"window": atom.observed}
                observed["phase"] = phase
                atoms.append(_rebuild_atom(atom, observed=observed, state=AtomState.UNKNOWN))
    return atoms


def _function_node(tree: ast.AST, target_symbol: str) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Module:
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        if function.name == target_symbol:
            return function
    return functions[0] if functions else tree if isinstance(tree, ast.Module) else tree


def _assertion_listcomp_subject(assertion: ast.Assert) -> str:
    test = assertion.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = test.left, test.comparators[0]
        if isinstance(right, ast.ListComp):
            return _subject(left)
        if isinstance(left, ast.ListComp):
            return _subject(right)
        return _subject(left)
    return "result"


def _test_function_scope(tree: ast.AST, test_symbol: str, test_line: int | None = None) -> ast.AST:
    name = test_symbol.rsplit("::", 1)[-1].split("[", 1)[0]
    functions = [
        function
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if name:
        for function in functions:
            if function.name == name:
                return function
    if test_line is not None:
        for function in functions:
            if _line(function) <= test_line <= _end_line(function, _line(function)):
                return function
    return tree


def parse_test_ast(
    test_source: str,
    *,
    path: str = "tests/test_contract.py",
    task_id: str = "",
    target_symbol: str = "target",
    test_symbol: str = "",
    test_line: int | None = None,
) -> tuple[ContractAtom, ...]:
    """Extract normative atoms from Python test syntax without executing it."""

    try:
        tree = ast.parse(test_source, filename=path)
    except (SyntaxError, TypeError, ValueError):
        return ()
    bindings = _param_bindings(tree)
    assertion_contexts = _assertion_contexts(
        tree,
        assertion_root=_test_function_scope(tree, test_symbol, test_line),
    )
    # Expected-side context prevents setup loops from becoming test demands.
    assertion_node_ids = frozenset(
        id(context_node)
        for _assertion, context_nodes in assertion_contexts
        for context_node in context_nodes
    )
    atoms: list[ContractAtom] = []
    handled_ranges: set[int] = set()
    handled_listcomps: set[int] = set()
    range_cardinalities: list[int] = []
    for node, context_nodes in assertion_contexts:
        test = node.test
        listcomp_subject = _assertion_listcomp_subject(node)
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and _operator_text(test.ops[0]) == "==":
            left, right = test.left, test.comparators[0]
            if isinstance(right, ast.ListComp):
                listcomp_subject = _subject(left)
            elif isinstance(left, ast.ListComp):
                listcomp_subject = _subject(right)
        for child in context_nodes:
            if not isinstance(child, ast.ListComp) or id(child) in handled_listcomps:
                continue
            handled_listcomps.add(id(child))
            list_atoms = _listcomp_contract(
                child,
                subject=listcomp_subject,
                path=path,
                target_symbol=target_symbol,
                task_id=task_id,
                bindings=bindings,
                handled_ranges=handled_ranges,
            )
            atoms.extend(list_atoms)
            range_cardinalities.extend(
                int(atom.expected["cardinality"])
                for atom in list_atoms
                if atom.kind is AtomKind.BOUND
                and isinstance(atom.expected, Mapping)
                and isinstance(atom.expected.get("cardinality"), int)
            )
        atoms.extend(
            _assert_window_atoms(
                node,
                path=path,
                target_symbol=target_symbol,
                task_id=task_id,
                bindings=bindings,
                range_cardinalities=range_cardinalities,
                context_nodes=context_nodes,
            )
        )
    for call in _range_calls(tree):
        if id(call) in handled_ranges or id(call) not in assertion_node_ids:
            continue
        spec, known, cardinality = _range_spec(call, bindings)
        provenance = (
            _span(path, call, symbol=target_symbol, role="test_assertion"),
            _span(path, call, symbol=target_symbol, role="range_contract"),
        )
        atoms.append(
            _atom(
                kind=AtomKind.BOUND,
                subject="range",
                expected=spec,
                state=AtomState.UNKNOWN if not known else AtomState.UNKNOWN,
                basis=AtomBasis.TEST_AST,
                provenance=provenance,
                task_id=task_id,
                target_symbol=target_symbol,
                notes=("range stop is exclusive",),
            )
        )
        direction = _range_direction(spec)
        if direction is not None:
            atoms.append(
                _atom(
                    kind=AtomKind.ORDER,
                    subject="range",
                    expected=direction,
                    basis=AtomBasis.TEST_AST,
                    provenance=provenance,
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        if cardinality is not None:
            range_cardinalities.append(cardinality)
    atoms.extend(
        _order_atoms(
            tree,
            path=path,
            target_symbol=target_symbol,
            task_id=task_id,
            basis=AtomBasis.TEST_AST,
            allowed_node_ids=assertion_node_ids,
            assertion_context=True,
        )
    )
    atoms.extend(_comparison_atoms(tree, path=path, target_symbol=target_symbol, task_id=task_id, basis=AtomBasis.TEST_AST, bindings=bindings))
    atoms.extend(_test_unit_atoms(tree, path=path, target_symbol=target_symbol, task_id=task_id))
    atoms.extend(_policy_atoms_test(tree, path=path, target_symbol=target_symbol, task_id=task_id))
    return _test_postprocess(atoms, range_cardinalities)


def _test_postprocess(atoms: Sequence[ContractAtom], cardinalities: Sequence[int]) -> tuple[ContractAtom, ...]:
    descending = any(atom.kind is AtomKind.ORDER and atom.expected == "descending" for atom in atoms)
    result: list[ContractAtom] = []
    for atom in atoms:
        if atom.kind is AtomKind.WINDOW and isinstance(atom.expected, Mapping) and "count" in atom.expected and "phase" not in atom.expected:
            count = atom.expected.get("count")
            if descending and isinstance(count, int) and count in cardinalities:
                expected = dict(atom.expected)
                expected["phase"] = "after-ordering"
                atom = _rebuild_atom(atom, expected=expected)
        result.append(atom)
    return _collapse_same_contract(result)


def parse_target_ast(
    target_source: str,
    *,
    path: str = "source.py",
    task_id: str = "",
    target_symbol: str = "target",
) -> tuple[ContractAtom, ...]:
    """Extract low-priority observations and exact target-bearing spans."""

    try:
        tree = ast.parse(target_source, filename=path)
    except (SyntaxError, TypeError, ValueError):
        return ()
    function = _function_node(tree, target_symbol)
    parents = _target_parents(tree)
    function_tree: ast.AST = function
    order_calls = [
        node
        for node in ast.walk(function_tree)
        if isinstance(node, ast.Call) and _order_value(node) is not None
    ]
    order_calls.sort(key=lambda node: (_line(node), getattr(node, "col_offset", 0)))
    order_direction = _order_value(order_calls[-1]) if order_calls else "unspecified"
    atoms: list[ContractAtom] = []
    atoms.extend(_order_atoms(function_tree, path=path, target_symbol=target_symbol, task_id=task_id, basis=AtomBasis.TARGET_AST, target_spans=(_span(path, function, symbol=target_symbol, role="target_function"),)))
    atoms.extend(_target_windows(function_tree, path=path, target_symbol=target_symbol, task_id=task_id, parents=parents, order_calls=order_calls))
    atoms.extend(_comparison_atoms(function_tree, path=path, target_symbol=target_symbol, task_id=task_id, basis=AtomBasis.TARGET_AST, bindings={}))
    atoms.extend(_target_unit_atoms(function_tree, path=path, target_symbol=target_symbol, task_id=task_id))
    atoms.extend(_policy_atoms_target(tree, path=path, target_symbol=target_symbol, task_id=task_id))
    windows = [atom for atom in atoms if atom.kind is AtomKind.WINDOW]
    atoms.extend(
        _end_atoms_target(
            function_tree,
            path=path,
            target_symbol=target_symbol,
            task_id=task_id,
            order_direction=order_direction,
            has_window=bool(windows),
        )
    )
    return _collapse_same_contract(atoms)


def _failure_location(run: TestRunEvidence, lines: Sequence[str]) -> tuple[str, int | None]:
    if run.test_path:
        return run.test_path, run.test_line
    node_path = run.nodeid.split("::", 1)[0] if run.nodeid else ""
    pattern = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^\s:]+\.py):(?P<line>\d+)(?::\d+)?")
    for text in lines:
        match = pattern.search(text)
        if match:
            return match.group("path"), int(match.group("line"))
    return node_path, run.test_line


def _is_infrastructure_output(output: str) -> bool:
    lowered = output.lower()
    has_assertion = "assertionerror" in lowered or re.search(r"\bassert\b", lowered) is not None
    return not has_assertion and any(re.search(marker, lowered) for marker in _INFRA_MARKERS)


def _split_top_level_compare(text: str) -> tuple[str, str, str] | None:
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escape = False
    operators = (" is not ", " not in ", "<=", ">=", "==", "!=", " is ", " in ", "<", ">")
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in depths:
            depths[char] += 1
            index += 1
            continue
        if char in closing:
            depths[closing[char]] = max(0, depths[closing[char]] - 1)
            index += 1
            continue
        if not any(depths.values()):
            for operator in operators:
                if text.startswith(operator, index):
                    left = text[:index].strip()
                    right = text[index + len(operator):].strip()
                    if left and right:
                        return left, operator.strip(), right
        index += 1
    return None


def _literal_output(text: str) -> tuple[object, bool]:
    if "..." in text or "Ellipsis" in text:
        return None, False
    try:
        return _normal(ast.literal_eval(text)), True
    except (ValueError, SyntaxError, TypeError):
        return None, False


def _numeric_tail(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # Do not mistake the separator in ``metric-001`` for a negative sign;
        # a leading sign is valid only at the beginning of a token.
        match = re.search(r"(?<![A-Za-z0-9])-?\d+\D*$", value)
        if match:
            return int(match.group(0).rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.:"))
    return None


def _sequence_direction(value: object) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    numeric = [_numeric_tail(item) for item in value]
    if any(item is None for item in numeric):
        return None
    numbers = [item for item in numeric if item is not None]
    if all(left < right for left, right in zip(numbers, numbers[1:])):
        return "ascending"
    if all(left > right for left, right in zip(numbers, numbers[1:])):
        return "descending"
    return None


def _output_span(
    run: TestRunEvidence,
    *,
    output_line: int,
    test_path: str,
    test_line: int | None,
    role: str,
) -> tuple[SourceSpan, ...]:
    output_span = SourceSpan(
        path=run.output_path or "pytest-output",
        start_line=output_line,
        end_line=output_line,
        symbol=run.nodeid,
        role="pytest_output",
    )
    if test_path and test_line:
        return (
            SourceSpan(
                path=test_path,
                start_line=test_line,
                end_line=test_line,
                symbol=run.nodeid,
                role=role,
            ),
            output_span,
        )
    return (output_span,)


def parse_pytest_output(
    output: str | TestRunEvidence,
    *,
    exit_code: int = 1,
    task_id: str = "",
    target_symbol: str = "target",
    test_path: str = "",
) -> tuple[ContractAtom, ...]:
    """Parse reliable assertion signal from pytest's textual failure output."""

    run = output if isinstance(output, TestRunEvidence) else TestRunEvidence(output=output, exit_code=exit_code, test_path=test_path)
    if not run.is_genuine_red or _is_infrastructure_output(run.output):
        return ()
    lines = run.output.splitlines()
    discovered_path, discovered_line = _failure_location(run, lines)
    atoms: list[ContractAtom] = []
    assertion_pattern = re.compile(r"\bassert\s+(.+)$", re.IGNORECASE)
    for output_index, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        # The indented source assertion is context, not an observed failure.
        # Pytest's evaluated assertion is emitted on an ``E`` line.
        if not re.match(r"^E(?:\s+|$)", line):
            continue
        line = re.sub(r"^E\s*", "", line)
        if not line or "..." in line:
            continue
        match = assertion_pattern.search(line)
        if match:
            expression = match.group(1).strip()
            if expression.lower().startswith("ionerror:"):
                expression = expression.split(":", 1)[1].strip()
            comparison = _split_top_level_compare(expression)
            if comparison:
                left_text, operator, right_text = comparison
                left, left_known = _literal_output(left_text)
                right, right_known = _literal_output(right_text)
                provenance = _output_span(
                    run,
                    output_line=output_index,
                    test_path=discovered_path,
                    test_line=discovered_line,
                    role="failing_assertion",
                )
                if left_known and right_known and isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                    observed_direction = _sequence_direction(left)
                    expected_direction = _sequence_direction(right)
                    subject = left_text if re.fullmatch(r"[A-Za-z_]\w*", left_text) else "asserted sequence"
                    if observed_direction and expected_direction:
                        atoms.append(
                            _atom(
                                kind=AtomKind.ORDER,
                                subject=subject,
                                expected=expected_direction,
                                observed=observed_direction,
                                state=_derive_state(expected_direction, observed_direction),
                                basis=AtomBasis.FAILING_ASSERTION,
                                provenance=provenance,
                                task_id=task_id,
                                target_symbol=target_symbol,
                            )
                        )
                        observed_end = "high-index end" if observed_direction == "descending" else "low-index end"
                        expected_end = "high-index end" if expected_direction == "descending" else "low-index end"
                        atoms.append(
                            _atom(
                                kind=AtomKind.END_SELECTION,
                                subject=subject,
                                expected=expected_end,
                                observed=observed_end,
                                state=_derive_state(expected_end, observed_end),
                                basis=AtomBasis.FAILING_ASSERTION,
                                provenance=provenance,
                                task_id=task_id,
                                target_symbol=target_symbol,
                            )
                        )
                    if len(left) != len(right):
                        atoms.append(
                            _atom(
                                kind=AtomKind.WINDOW,
                                subject=subject,
                                expected={"count": len(right)},
                                observed={"count": len(left)},
                                state=AtomState.CONTRADICTED,
                                basis=AtomBasis.FAILING_ASSERTION,
                                provenance=provenance,
                                task_id=task_id,
                                target_symbol=target_symbol,
                            )
                        )
                elif left_known and right_known and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    if "len" in left_text.lower() or "count" in left_text.lower() or "len" in right_text.lower() or "count" in right_text.lower():
                        atoms.append(
                            _atom(
                                kind=AtomKind.WINDOW,
                                subject="collection",
                                expected={"count": right},
                                observed={"count": left},
                                state=_derive_state(right, left),
                                basis=AtomBasis.FAILING_ASSERTION,
                                provenance=provenance,
                                task_id=task_id,
                                target_symbol=target_symbol,
                            )
                        )
                    else:
                        atoms.append(
                            _atom(
                                kind=AtomKind.BOUND,
                                subject=left_text,
                                expected={"operator": operator, "value": right, "inclusive": operator in {"<=", ">="}},
                                observed={"value": left, "operator": operator},
                                state=AtomState.CONTRADICTED if left != right else AtomState.SUPPORTED,
                                basis=AtomBasis.FAILING_ASSERTION,
                                provenance=provenance,
                                task_id=task_id,
                                target_symbol=target_symbol,
                            )
                        )
                    atoms.append(
                        _atom(
                            kind=AtomKind.POLARITY,
                            subject=left_text,
                            expected=operator,
                            observed=operator,
                            state=AtomState.SUPPORTED,
                            basis=AtomBasis.FAILING_ASSERTION,
                            provenance=provenance,
                            task_id=task_id,
                            target_symbol=target_symbol,
                        )
                    )
            continue
        contains = re.search(r"(?:right|left)\s+contains\s+(\d+)\s+more\s+items?", line, re.IGNORECASE)
        if contains:
            delta = int(contains.group(1))
            provenance = _output_span(
                run,
                output_line=output_index,
                test_path=discovered_path,
                test_line=discovered_line,
                role="failing_assertion",
            )
            atoms.append(
                _atom(
                    kind=AtomKind.WINDOW,
                    subject="collection",
                    expected={"count_delta": delta},
                    observed={"count_delta": 0},
                    state=AtomState.CONTRADICTED,
                    basis=AtomBasis.FAILING_ASSERTION,
                    provenance=provenance,
                    task_id=task_id,
                    target_symbol=target_symbol,
                )
            )
        diff = re.search(r"At index \d+ diff:\s*(.+?)\s+(!=|==)\s+(.+)$", line, re.IGNORECASE)
        if diff and "..." not in diff.group(0):
            # A single diff pair proves a value mismatch but not sequence direction.
            provenance = _output_span(
                run,
                output_line=output_index,
                test_path=discovered_path,
                test_line=discovered_line,
                role="failing_assertion",
            )
            atoms.append(
                _atom(
                    kind=AtomKind.BOUND,
                    subject="asserted value",
                    expected=_clip(diff.group(3), 80),
                    observed=_clip(diff.group(1), 80),
                    state=AtomState.CONTRADICTED,
                    basis=AtomBasis.FAILING_ASSERTION,
                    provenance=provenance,
                    task_id=task_id,
                    target_symbol=target_symbol,
                    notes=("single diff pair does not establish order",),
                )
            )
    return _collapse_same_contract(atoms)


def extract_pytest_atoms(*args: object, **kwargs: object) -> tuple[ContractAtom, ...]:
    return parse_pytest_output(*args, **kwargs)  # type: ignore[arg-type]


def parse_pytest_failure(*args: object, **kwargs: object) -> tuple[ContractAtom, ...]:
    return parse_pytest_output(*args, **kwargs)  # type: ignore[arg-type]


def extract_target_atoms(*args: object, **kwargs: object) -> tuple[ContractAtom, ...]:
    return parse_target_ast(*args, **kwargs)  # type: ignore[arg-type]


def extract_test_atoms(*args: object, **kwargs: object) -> tuple[ContractAtom, ...]:
    return parse_test_ast(*args, **kwargs)  # type: ignore[arg-type]


def _test_symbol_from_nodeid(nodeid: str) -> str:
    return nodeid.rsplit("::", 1)[-1].split("[", 1)[0] if nodeid else ""


def extract_evidence_atoms(evidence: NitEvidence) -> tuple[ContractAtom, ...]:
    output_atoms = parse_pytest_output(
        evidence.test_run,
        task_id=evidence.task_id,
        target_symbol=evidence.target_symbol,
    )
    test_atoms = parse_test_ast(
        evidence.test_source,
        path=evidence.test_path or evidence.test_run.test_path or "tests/test_contract.py",
        task_id=evidence.task_id,
        target_symbol=evidence.target_symbol,
        test_symbol=evidence.test_symbol or _test_symbol_from_nodeid(evidence.test_run.nodeid),
        test_line=evidence.test_run.test_line,
    ) if evidence.test_source else ()
    source_atoms = parse_target_ast(
        evidence.target_source,
        path=evidence.target_file,
        task_id=evidence.task_id,
        target_symbol=evidence.target_symbol,
    ) if evidence.target_source else ()
    return _join_all((*output_atoms, *test_atoms), source_atoms)


def build_edit_proposal(
    path: str,
    old_source: str,
    new_source: str,
    *,
    changed_spans: Sequence[SourceSpan] = (),
    changed_lines: Sequence[int] = (),
    target_symbol: str = "",
) -> EditProposal:
    """Build the exact-byte proposal consumed by the pre-edit check."""

    return EditProposal(
        path=path,
        old_source=old_source,
        new_source=new_source,
        changed_spans=tuple(changed_spans),
        changed_lines=tuple(changed_lines),
        target_symbol=target_symbol,
    )


def _human_value(kind: AtomKind, value: object, *, expected: bool) -> str:
    if isinstance(value, Mapping):
        if kind is AtomKind.WINDOW:
            count = value.get("count")
            phase = value.get("phase")
            if count is not None and phase == "after-ordering":
                return f"{count} rows, cap after ordering"
            if count is not None and phase == "before-ordering":
                return f"{count} rows, cap before ordering"
            if count is not None:
                return f"{count} rows"
            if "limit" in value:
                return f"limit {value['limit']}"
            if "offset" in value:
                return f"offset {value['offset']}"
        if kind is AtomKind.BOUND:
            if "step" in value:
                cardinality = f", cardinality {value['cardinality']}" if "cardinality" in value else ""
                return f"range({value.get('start')}, {value.get('stop')}, step {value.get('step')}){cardinality}; stop exclusive"
            if "operator" in value:
                return f"{value.get('operator')} {value.get('value')}"
        if kind is AtomKind.POLICY:
            flag = value.get("flag", "policy")
            default = value.get("default")
            if default == "off":
                return f"default-off / explicit opt-in for {flag}"
            if default == "on":
                return f"default-on for {flag}"
        if kind is AtomKind.UNIT and value.get("timezone_aware"):
            return "timezone-aware values"
        return _clip(canonical_json(value), 150)
    if kind is AtomKind.ORDER and value not in (None, ""):
        return f"{value} order"
    if kind is AtomKind.END_SELECTION:
        return str(value)
    if kind is AtomKind.UNIT:
        return str(value)
    if kind is AtomKind.POLARITY:
        return f"{value} comparison"
    return _clip(value, 150)


def render_confrontation(
    atoms: Iterable[ContractAtom],
    *,
    phase: str = "RED",
    max_atoms: int = MAX_VISIBLE_ATOMS,
) -> str | None:
    """Render only actionable atoms; UNKNOWN remains receipt-only/silent."""

    phase_name = phase.upper()
    selected = [
        atom
        for atom in rank_atoms(atoms, limit=None)
        if (phase_name == "GREEN" and atom.state is AtomState.SUPPORTED)
        or (phase_name != "GREEN" and atom.state is AtomState.CONTRADICTED)
    ][:max(0, max_atoms)]
    if not selected:
        return None
    clauses: list[str] = []
    for atom in selected:
        expected = _human_value(atom.kind, atom.expected, expected=True)
        observed = _human_value(atom.kind, atom.observed, expected=False)
        span = _span_label(atom.target_spans or atom.provenance)
        if phase_name == "GREEN":
            clauses.append(f"the green run now supports {expected} at <{span}>")
        else:
            clauses.append(f"the test demands {expected}, your code does {observed} at <{span}>")
    prefix = "[kernel-nit] GREEN: " if phase_name == "GREEN" else "[kernel-nit] RED: "
    return prefix + "; ".join(clauses) + "."


def render_red(atoms: Iterable[ContractAtom], *, max_atoms: int = MAX_VISIBLE_ATOMS) -> str | None:
    return render_confrontation(atoms, phase="RED", max_atoms=max_atoms)


def render_edit_attention(atoms: Iterable[ContractAtom]) -> str | None:
    labels = [atom.kind.value for atom in rank_atoms(atoms, limit=None) if atom.state is AtomState.CONTRADICTED]
    if not labels:
        return None
    labels = list(dict.fromkeys(labels))[:MAX_VISIBLE_ATOMS]
    return (
        "[kernel-nit] EDIT ATTENTION: this edit leaves all violated contract atoms untouched "
        f"({', '.join(labels)}); the edit is still being applied. Reconcile the red contract on the next test."
    )


def render_done_challenge(atoms: Iterable[ContractAtom]) -> str | None:
    unresolved = [atom for atom in rank_atoms(atoms, limit=None) if atom.state is AtomState.CONTRADICTED]
    if not unresolved:
        return None
    labels = ", ".join(dict.fromkeys(atom.kind.value for atom in unresolved))
    return (
        "[kernel-nit] DONE CHALLENGE: fixed is not yet certified; "
        f"{len(unresolved)} contradicted contract atoms remain unresolved: {labels}."
    )


def build_confrontation(
    atoms: Iterable[ContractAtom],
    *,
    phase: str = "RED",
    result_class: ResultClass | str | None = None,
) -> Confrontation:
    all_atoms = tuple(atoms)
    visible = rank_atoms(all_atoms, limit=MAX_VISIBLE_ATOMS)
    result = result_class
    if result is None:
        result = ResultClass.CONTRADICTED if any(atom.is_contradicted for atom in all_atoms) else ResultClass.UNKNOWN
    return Confrontation(
        phase=phase,
        atoms=visible,
        all_atoms=all_atoms,
        result_class=result,
        suppressed_count=max(0, len(all_atoms) - len(visible)),
    )


@dataclass
class NitpickerState:
    """Per-task state with bounded output, deduplication, and green generations."""

    task_id: str = "task"
    target_symbol: str = "target"
    armed: bool = True
    green_generation: int = 0
    _atoms: dict[str, ContractAtom] = field(default_factory=dict, init=False, repr=False)
    _red_scopes: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _seen_fingerprints: set[str] = field(default_factory=set, init=False, repr=False)
    _edit_advisories: set[tuple[int, tuple[str, ...]]] = field(default_factory=set, init=False, repr=False)
    _done_challenged: bool = field(default=False, init=False, repr=False)
    trace_cb: NitTraceCallback | None = field(default=None, repr=False, compare=False)
    trace: list[dict[str, object]] = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._record(
            "NITPICKER_ARMED" if self.armed else "NITPICKER_DISARMED",
            f"task={self.task_id} target={self.target_symbol}",
        )

    def _record(self, event: str, detail: object) -> None:
        entry = {
            "cylinder": "nitpicker",
            "event": event,
            "detail": _clip(detail, 240),
            "turn": None,
        }
        self.trace.append(entry)
        callback = self.trace_cb
        if callback is None:
            return
        try:
            callback("nitpicker", event, str(entry["detail"]))
        except Exception as exc:  # noqa: BLE001 - observability cannot control work
            print(f"[kernel-wiring] nitpicker trace failed: {type(exc).__name__}")
            self.trace_cb = None

    def arm(self) -> None:
        self.armed = True
        self._record("NITPICKER_ARMED", "explicit arm")

    def disarm(self) -> None:
        self.armed = False
        self._record("NITPICKER_DISARMED", "explicit disarm")

    @property
    def atoms(self) -> tuple[ContractAtom, ...]:
        return tuple(sorted(self._atoms.values(), key=_atom_sort_key))

    @property
    def contradicted(self) -> tuple[ContractAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.state is AtomState.CONTRADICTED)

    @property
    def supported(self) -> tuple[ContractAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.state is AtomState.SUPPORTED)

    @property
    def unknown(self) -> tuple[ContractAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.state is AtomState.UNKNOWN)

    def _store(self, atoms: Iterable[ContractAtom]) -> tuple[ContractAtom, ...]:
        for atom in atoms:
            previous = self._atoms.get(atom.contract_key)
            if previous is None:
                self._atoms[atom.contract_key] = atom
                continue
            if previous.state is AtomState.CONTRADICTED and atom.state is AtomState.UNKNOWN:
                continue
            if previous.state is AtomState.SUPPORTED and atom.state is AtomState.UNKNOWN:
                continue
            self._atoms[atom.contract_key] = _rebuild_atom(
                atom,
                provenance=_unique_spans((*previous.provenance, *atom.provenance)),
                target_spans=_unique_spans((*previous.target_spans, *atom.target_spans)),
                notes=(*previous.notes, *atom.notes),
            )
        return self.atoms

    def observe_red(self, evidence: NitEvidence) -> Confrontation:
        if not self.armed:
            return Confrontation("RED", (), (), ResultClass.DISARMED)
        if evidence.task_id != self.task_id:
            raise ValueError("evidence task_id does not match nit-picker state")
        if not evidence.test_run.is_genuine_red or _is_infrastructure_output(evidence.test_run.output):
            return Confrontation("RED", (), (), ResultClass.UNKNOWN)
        self._record(
            "NITPICKER_TRIGGER",
            f"task={evidence.task_id} target={evidence.target_symbol}",
        )
        joined = extract_evidence_atoms(evidence)
        # Preserve every contributing RED scope. Unknown scope cannot be discharged.
        scope = evidence.test_run.nodeid or evidence.test_run.test_path
        for atom in joined:
            if atom.state is AtomState.CONTRADICTED:
                self._red_scopes.setdefault(atom.contract_key, set()).add(scope)
        self._store(joined)
        current = [atom for atom in self.contradicted if atom.fingerprint not in self._seen_fingerprints]
        visible = rank_atoms(current, limit=MAX_VISIBLE_ATOMS)
        self._seen_fingerprints.update(atom.fingerprint for atom in current)
        result = ResultClass.NEW if visible else ResultClass.DEDUPED if current == [] and self.contradicted else ResultClass.UNKNOWN
        if result is ResultClass.NEW:
            self._record("NITPICKER_FIRED", f"atoms={len(visible)}")
        elif result is ResultClass.DEDUPED:
            self._record("NITPICKER_DEDUPED", f"atoms={len(current)}")
        else:
            self._record("NITPICKER_SUPPRESSED", f"result={result.value}")
        return Confrontation(
            phase="RED",
            atoms=visible,
            all_atoms=tuple(current),
            result_class=result,
            suppressed_count=max(0, len(current) - len(visible)),
        )

    on_red = observe_red

    def observe_green(
        self,
        evidence: NitEvidence | None = None,
        *,
        relevant: bool = True,
        untainted: bool = True,
    ) -> Confrontation:
        if not self.armed:
            return Confrontation("GREEN", (), (), ResultClass.DISARMED)
        self.green_generation += 1
        if evidence is not None and evidence.test_run.exit_code != 0:
            return Confrontation("GREEN", (), (), ResultClass.UNKNOWN)
        resolved: list[ContractAtom] = []
        if relevant and untainted:
            if evidence is not None:
                self._store(extract_evidence_atoms(evidence))
            for atom in self.contradicted:
                resolved_atom = _rebuild_atom(atom, observed=atom.expected, state=AtomState.SUPPORTED)
                self._atoms[atom.contract_key] = resolved_atom
                if resolved_atom.fingerprint not in self._seen_fingerprints:
                    resolved.append(resolved_atom)
                    self._seen_fingerprints.add(resolved_atom.fingerprint)
        return Confrontation(
            phase="GREEN",
            atoms=rank_atoms(resolved, limit=MAX_VISIBLE_ATOMS),
            all_atoms=tuple(resolved),
            result_class=ResultClass.SUPPORTED if resolved else ResultClass.UNKNOWN,
            suppressed_count=max(0, len(resolved) - MAX_VISIBLE_ATOMS),
        )

    on_green = observe_green

    def reconcile_public_green(self, selectors: tuple[str, ...]) -> tuple[str, ...]:
        """Discharge only known RED scopes covered by host-verified current checks.

        The runtime owns freshness/untainted-harness verification. Unlike the legacy
        observe_green API, unknown or filtered RED scopes never clear here.
        """
        from kernel_nit_reconciliation import selector_covers
        if not self.armed:
            return ()
        resolved = []
        for atom in self.contradicted:
            scopes = self._red_scopes.get(atom.contract_key, set())
            if not scopes or not all(any(selector_covers(green, red) for green in selectors)
                                     for red in scopes):
                continue
            self._atoms[atom.contract_key] = _rebuild_atom(
                atom, observed=atom.expected, state=AtomState.SUPPORTED,
                notes=(*atom.notes, "current scope-bound public check passed"))
            resolved.append(atom.contract_key)
        if resolved:
            self.green_generation += 1
            self._record("NITPICKER_BOUND_GREEN", f"resolved={len(resolved)}")
        return tuple(resolved)

    def check_edit(self, proposal: EditProposal) -> EditCheck:
        violated = self.contradicted
        usable = [
            atom
            for atom in violated
            if any(_same_path(span.path, proposal.path) for span in atom.target_spans)
        ]
        if not self.armed or not violated or not usable:
            return EditCheck(None, True, False, tuple(usable))
        changed_lines = set(proposal.touched_line_numbers)
        touched = any(
            _same_path(span.path, proposal.path)
            and any(
                span.start_line <= line <= (span.end_line or span.start_line)
                for line in changed_lines
            )
            for atom in usable
            for span in atom.target_spans
        )
        return EditCheck(None, True, touched, tuple(usable))

    def before_edit(self, proposal: EditProposal) -> str | None:
        check = self.check_edit(proposal)
        if check.touches_atom_lines or not check.considered_atoms:
            return None
        key = (self.green_generation, tuple(sorted(atom.fingerprint for atom in check.considered_atoms)))
        if key in self._edit_advisories:
            return None
        self._edit_advisories.add(key)
        line = render_edit_attention(check.considered_atoms)
        if line is not None:
            self._record("NITPICKER_EDIT_ATTENTION", f"atoms={len(check.considered_atoms)}")
        return line

    def done_challenge(self) -> str | None:
        if not self.armed or self._done_challenged:
            return None
        line = render_done_challenge(self.contradicted)
        if line is None:
            return None
        self._done_challenged = True
        self._record("NITPICKER_CHALLENGE_ONCE", "done challenge")
        return line

    def completion_counts(self) -> dict[str, int]:
        return {
            "nit_contradicted": len(self.contradicted),
            "nit_supported": len(self.supported),
            "nit_unknown": len(self.unknown),
            "nit_unresolved": len(self.contradicted),
        }

    def summary(self) -> dict[str, object]:
        return {"armed": self.armed, "green_generation": self.green_generation, **self.completion_counts()}


def _nitpicker_trigger(evidence: Evidence) -> TriState:
    """Recognize the genuine-red input that makes the nit-picker eligible."""

    record = evidence.event
    if evidence.source == "trace":
        event = str(record.get("event") or "").strip().lower()
        if not event:
            return TriState.UNKNOWN
        return TriState.YES if event == "nitpicker_trigger" else TriState.NO
    if evidence.source != "receipt":
        return TriState.NO

    kind = str(record.get("kind") or "").strip().lower()
    result = str(record.get("result_class") or record.get("result") or "").strip().upper()
    if kind == "nit":
        if not result:
            return TriState.UNKNOWN
        return TriState.YES if result in {"NEW", "CONTRADICTED", "DEDUPED", "UNKNOWN"} else TriState.NO
    if kind in {"test", "observation"}:
        if result in {"RED", "PRODUCT_RED"}:
            return TriState.YES
        return TriState.UNKNOWN if not result else TriState.NO
    return TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="nitpicker",
    trigger=_nitpicker_trigger,
    fire_events=frozenset(
        {
            "NITPICKER_FIRED",
            "NITPICKER_EDIT_ATTENTION",
            "red_confrontation",
            "red_no_atoms",
            "nit:NEW",
            "nit:CONTRADICTED",
        }
    ),
    suppression_events=frozenset({"NITPICKER_SUPPRESSED"}),
    dedup_events=frozenset({"NITPICKER_DEDUPED", "nit:DEDUPED"}),
    challenge_once_events=frozenset({"NITPICKER_CHALLENGE_ONCE", "challenged_done"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


NitPickerState = NitpickerState
NitPicker = NitpickerState
Nitpicker = NitpickerState


def emit_via_organ_out(bolt: object, line: str | None) -> str | None:
    """Route output through the existing KernelBolt shadow chokepoint."""

    if not line:
        return None
    organ_out = getattr(bolt, "organ_out", None)
    if not callable(organ_out):
        raise TypeError("bolt must expose callable organ_out(name, line)")
    return organ_out("nitpicker", line)


organ_out = emit_via_organ_out
render = render_confrontation


__all__ = [
    "ANALYZER_VERSION",
    "AtomBasis",
    "AtomKind",
    "AtomState",
    "AtomStatus",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "ContractAtom",
    "ContractState",
    "Confrontation",
    "EditCheck",
    "EditProposal",
    "JsonValue",
    "NitEvidence",
    "NitPicker",
    "NitPickerState",
    "Nitpicker",
    "NitpickerState",
    "ResultClass",
    "SourceSpan",
    "TestRunEvidence",
    "build_confrontation",
    "build_edit_proposal",
    "emit_via_organ_out",
    "extract_evidence_atoms",
    "extract_pytest_atoms",
    "extract_target_atoms",
    "extract_test_atoms",
    "join_all_atoms",
    "join_atoms",
    "organ_out",
    "parse_pytest_failure",
    "parse_pytest_output",
    "parse_target_ast",
    "parse_test_ast",
    "rank_atoms",
    "render",
    "render_confrontation",
    "render_done_challenge",
    "render_edit_attention",
    "render_red",
]
