"""Conservative policy for host CHECK GENERATOR output.

The boundary compiler is allowed to propose checks from public material, but a
proposal is not automatically an oracle.  This module is the opt-in policy
layer which grounds literals, keeps fixtures in the vocabulary of the public
task, and executes each generated call against the starter before admission.

It intentionally contains no task answers.  Its inputs are the public brief,
public tests, and public starter sources supplied by the caller.  The policy
is inactive unless ``STATEFRAME_CHECK_GENERATOR_FIXES=on``.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kernel_live_projection import is_harness_path


MODE_ENV = "STATEFRAME_CHECK_GENERATOR_FIXES"
COMPAT_ADMISSION_ENV = "STATEFRAME_TOKENLAB_COMPATIBILITY_ADMISSION"
ADVISORY = "ADVISORY"
POLICY_SCHEMA = "stateframe-check-generator-policy/1"

_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?(?![A-Za-z_])")

# These are protocol/selector fields owned by the check executor.  They are
# not product vocabulary, so their presence does not make a generated fixture
# cross-task.  Product keys nested below them are still checked.
_STRUCTURAL_KEYS = frozenset({
    "args", "kwargs", "calls", "cases", "base", "variant", "entry_point",
    "input_index", "path", "json", "status", "body", "error_contains",
    "contains", "outputs", "selector", "value", "mode", "percent",
    "total_path", "component_paths", "public_value", "input", "fixture",
    "expected", "preserved", "equivalent", "binding", "kind", "surface",
    "field", "old", "new", "alias", "arity", "call", "template",
})
_STRUCTURAL_STRING_KEYS = frozenset({
    "entry_point", "path", "selector", "mode", "kind", "surface", "field",
    "template", "call", "alias", "old", "new",
})
_FORBIDDEN_PUBLIC_PATH_PARTS = frozenset({
    "private", "hidden", "hidden_tests", "near_miss", "gold", "oracle",
    "grader", "graders", "grading", "harness",
})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def compatibility_admission_enabled() -> bool:
    value = os.environ.get(COMPAT_ADMISSION_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{COMPAT_ADMISSION_ENV} must be off or on")
    return value == "on"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _public_text(brief: object, sources: object, tests: object) -> str:
    parts = [str(brief or "")]
    for collection in (sources, tests):
        if isinstance(collection, Mapping):
            parts.extend(str(item) for item in collection.values()
                         if isinstance(item, str))
        elif isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
            parts.extend(str(item) for item in collection if isinstance(item, str))
    return "\n".join(parts)


def _public_literals(text: str) -> tuple[set[str], set[str], set[str]]:
    """Return exact strings, numbers, and identifier vocabulary in public text."""
    strings: set[str] = set()
    numbers: set[str] = set(_NUMBER.findall(text))
    try:
        trees = [ast.parse(text)]
    except SyntaxError:
        trees = []
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str) and node.value.strip():
                    strings.add(node.value)
                elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    numbers.add(str(node.value))
    strings.update(item.strip() for item in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", text)
                   if item.strip())
    words = set(_WORD.findall(text))
    return strings, numbers, words


def _leaf_token(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _literal_in_public(leaf: str, text: str, *, strings: set[str],
                       words: set[str], numbers: set[str]) -> bool:
    if leaf in strings or leaf in words or leaf in numbers:
        return True
    # Identifier-like values must be complete public tokens: ``legacy`` is
    # not grounded merely because the public starter contains ``legacy_id``.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", text) is not None
    return leaf in text


def _fixture_walk(value: object, *, path: tuple[object, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield ("key", key, path)
            yield from _fixture_walk(child, path=(*path, key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _fixture_walk(child, path=(*path, index))
    else:
        yield ("value", value, path)


def _vocabulary_violation(token: object, path: tuple[object, ...], *, kind: str,
                          allowed_words: set[str], allowed_strings: set[str],
                          allowed_numbers: set[str], public_text: str) -> dict[str, object] | None:
    if kind == "key":
        text = str(token)
        if text in _STRUCTURAL_KEYS or text in allowed_words or text in allowed_strings:
            return None
        return {"token": text, "kind": "field", "path": list(path),
                "reason": "ABSENT_FROM_PUBLIC_MATERIAL"}
    if token is None or isinstance(token, bool):
        return None
    leaf = _leaf_token(token)
    if leaf is None:
        return None
    # Executor control values are not task fixtures.  Domain values remain
    # subject to exact/string-token admission below.
    parent = str(path[-1]) if path else ""
    if parent in _STRUCTURAL_STRING_KEYS:
        return None
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        if leaf in allowed_numbers:
            return None
    elif _literal_in_public(leaf, public_text, strings=allowed_strings,
                            words=allowed_words, numbers=allowed_numbers):
        return None
    return {"token": token, "kind": "value", "path": list(path),
            "reason": "ABSENT_FROM_PUBLIC_MATERIAL"}


def fixture_vocabulary_violations(fixture: object, brief: object = "",
                                  sources: object = None, tests: object = None
                                  ) -> tuple[dict[str, object], ...]:
    """Reject fixture fields and values absent from this task's public material."""
    text = _public_text(brief, sources or {}, tests or {})
    allowed_strings, allowed_numbers, allowed_words = _public_literals(text)
    violations: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for kind, token, path in _fixture_walk(fixture):
        violation = _vocabulary_violation(
            token, path, kind=kind, allowed_words=allowed_words,
            allowed_strings=allowed_strings, allowed_numbers=allowed_numbers,
            public_text=text)
        if violation is None:
            continue
        key = (kind, repr(token), repr(path))
        if key not in seen:
            seen.add(key)
            violations.append(violation)
    return tuple(violations)


def _asserted_literal_walk(value: object, *, path: tuple[object, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield ("key", key, path)
            yield from _asserted_literal_walk(child, path=(*path, key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _asserted_literal_walk(child, path=(*path, index))
    else:
        yield ("value", value, path)


def _ungrounded_ticket_literals(check: object, brief: object, tests: object) -> tuple[object, ...]:
    if str(getattr(check, "oracle_basis", "")).upper() != "TICKET":
        return ()
    expected = getattr(check, "expected", None)
    text = _public_text(brief, {}, tests or {})
    allowed_strings, allowed_numbers, allowed_words = _public_literals(text)
    missing: list[object] = []
    for kind, token, path in _asserted_literal_walk(expected):
        if kind == "key":
            text_token = str(token)
            if text_token in _STRUCTURAL_KEYS or text_token in allowed_words or text_token in allowed_strings:
                continue
            missing.append(text_token)
            continue
        if token is None or isinstance(token, bool):
            continue
        leaf = _leaf_token(token)
        if leaf is None:
            continue
        if isinstance(token, (int, float)) and not isinstance(token, bool):
            grounded = leaf in allowed_numbers
        else:
            grounded = _literal_in_public(
                leaf, text, strings=allowed_strings, words=allowed_words,
                numbers=allowed_numbers)
        if not grounded:
            missing.append(token)
    return tuple(dict.fromkeys(repr(item) if not isinstance(item, str) else item
                               for item in missing))


@dataclass(frozen=True, slots=True)
class StarterValidation:
    accepted: bool
    reason: str
    receipt: object = None


@dataclass(frozen=True, slots=True)
class PolicyResult:
    check: object
    accepted: bool
    reason: str = "ADMITTED"
    missing_literals: tuple[object, ...] = ()
    fixture_violations: tuple[dict[str, object], ...] = ()
    starter_validation: StarterValidation | None = None
    events: tuple[dict[str, object], ...] = ()
    admission_tier: str = "GATING"


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    checks: tuple[object, ...]
    rejected: tuple[dict[str, object], ...]
    events: tuple[dict[str, object], ...]
    checked: tuple[object, ...] = ()


def _event(rule: str, status: str, **details: object) -> dict[str, object]:
    return {"schema": POLICY_SCHEMA, "rule": rule, "status": status, **details}


def _with_policy_provenance(check: object, *, missing: Sequence[object] = (),
                            violations: Sequence[Mapping[str, object]] = (),
                            starter: str = "NOT_RUN",
                            admission_tier: str = "GATING",
                            compatibility_starter: Mapping[str, object] | None = None) -> object:
    degraded = bool(missing or violations or starter.startswith("REJECTED")
                    or starter in {"STARTER_VALIDATION_NOT_RUN", "NOT_RUN"}
                    or str(admission_tier).upper() != "GATING")
    effective_basis = (ADVISORY if degraded else
                       str(getattr(check, "oracle_basis", "")))
    provenance = copy.deepcopy(dict(getattr(check, "provenance", {}) or {}))
    provenance["generator_policy"] = {
        "schema": POLICY_SCHEMA,
        "oracle_basis": effective_basis,
        "missing_literals": list(missing),
        "fixture_violations": [dict(item) for item in violations],
        "starter_validation": starter,
        "requirements_admission_tier": str(admission_tier).upper(),
        "admission": "ADVISORY" if degraded else "GATING",
    }
    if compatibility_starter is not None:
        provenance["generator_policy"]["compatibility_starter"] = dict(compatibility_starter)
    if degraded:
        certificate = provenance.get("admission_certificate")
        if isinstance(certificate, Mapping):
            certificate = dict(certificate)
            certificate["admission"] = "ADVISORY"
            certificate["interpretation"] = "OPEN"
            certificate["evidence_complete"] = False
            provenance["admission_certificate"] = certificate
    contract_cell = getattr(check, "contract_cell", None)
    if degraded and contract_cell is not None:
        try:
            from kernel_obligation_assess import ContractCell
            contract_cell = ContractCell(
                obligation=contract_cell.obligation,
                surface=contract_cell.surface,
                case_partition=contract_cell.case_partition,
                fixture=contract_cell.fixture,
                observable_selector=contract_cell.observable_selector,
                predicate=contract_cell.predicate,
                oracle_sources=contract_cell.oracle_sources,
                applicability=contract_cell.applicability,
                admission="PROVISIONAL",
                cell_id=contract_cell.cell_id,
            )
        except (AttributeError, TypeError, ValueError):
            contract_cell = getattr(check, "contract_cell", None)
    try:
        return dataclasses.replace(
            check,
            oracle_basis=effective_basis,
            provenance=provenance,
            contract_cell=contract_cell,
        )
    except (TypeError, ValueError):
        return check


def _nested_error_types(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "exception_type" and isinstance(child, str):
                found.add(child)
            found.update(_nested_error_types(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.update(_nested_error_types(child))
    return found


def _binding_error_present(value: object) -> bool:
    """Return whether an executor result proves a call-shape failure.

    A target may legitimately raise ``ValueError`` for a ticket-driven
    negative case.  The executor's explicit binding marker is therefore more
    authoritative than the exception class alone; runtime ``TypeError`` is
    still treated as a malformed fixture because it is the common signature
    of a bare-string/dict-shape mistake.
    """
    if _nested_error_types(value) & {"BindingError", "TypeError"}:
        return True
    if isinstance(value, Mapping):
        if value.get("binding") not in {None, "BOUND"}:
            return True
        if value.get("binding_error"):
            return True
        exception = value.get("exception")
        if isinstance(exception, Mapping):
            if str(exception.get("type", "")) in {"BindingError", "TypeError"}:
                return True
        return any(_binding_error_present(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_binding_error_present(child) for child in value)
    return False


def validate_check_against_starter(root: str | Path, check: object) -> StarterValidation:
    """Run one generated check against the public starter and classify its evidence."""
    from kernel_obligation_assess import (
        HARNESS_ERROR, NOT_EXECUTED, PASS, COUNTEREXAMPLE, UNBOUND_ENTRYPOINT,
        UNBOUND_INVOCATION, run_check,
    )

    try:
        receipt = run_check(root, check)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return StarterValidation(False, "REJECTED_STARTER_HARNESS", {"error": str(error)})
    reason = str(getattr(receipt, "reason", ""))
    status = str(getattr(receipt, "status", "")).upper()
    observed = getattr(receipt, "observed", None)
    counterexample = getattr(receipt, "counterexample", None)
    binding_error = _binding_error_present((observed, counterexample))
    if reason == UNBOUND_ENTRYPOINT:
        return StarterValidation(False, "REJECTED_STARTER_ENTRYPOINT", receipt)
    if reason == UNBOUND_INVOCATION or status == NOT_EXECUTED or binding_error:
        return StarterValidation(False, "REJECTED_STARTER_CALL_SHAPE", receipt)
    if status == HARNESS_ERROR:
        text = " ".join(str(item) for item in (
            getattr(receipt, "reason", ""),
            getattr(receipt, "counterexample", ""),
        ))
        if "TypeError" in text or "BindingError" in text:
            return StarterValidation(False, "REJECTED_STARTER_CALL_SHAPE", receipt)
        return StarterValidation(False, "REJECTED_STARTER_HARNESS", receipt)
    preservation = (
        str(getattr(check, "oracle_basis", "")).upper() == "PRESERVED_BEHAVIOR"
        or str(getattr(check, "template", "")).upper().startswith("PRESERVE")
        or str(getattr(check, "template", "")).upper() == "REPLAY_EQUIVALENT"
        or str(getattr(check, "template", "")).upper() == "COMPATIBILITY_WITNESS"
    )
    if preservation and status != PASS:
        return StarterValidation(False, "REJECTED_PRESERVATION_STARTER", receipt)
    if status == PASS:
        return StarterValidation(True, "STARTER_PASS", receipt)
    if status == COUNTEREXAMPLE:
        return StarterValidation(True, "STARTER_TARGET_COUNTEREXAMPLE", receipt)
    return StarterValidation(False, "REJECTED_STARTER_EXECUTION", receipt)


def apply_generator_policy(check: object, *, brief: object = "",
                           sources: object = None, tests: object = None,
                           starter_root: str | Path | None = None,
                           requirements_ledger: object = None) -> PolicyResult:
    """Apply all generator admission rules to one candidate check."""
    if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
        try:
            from kernel_green_lock_repair import classify_check, public_expectations
            repair_policy = classify_check(
                check, starter_root=starter_root,
                public_expectations=public_expectations(tests or {}))
        except (ImportError, TypeError, ValueError):
            repair_policy = None
        if repair_policy is not None and not repair_policy.gating:
            event = _event(
                "green_lock_repair_admission", "ADVISORY",
                reason=repair_policy.reason,
                check_id=str(getattr(check, "check_id", "")),
            )
            validation = StarterValidation(False, repair_policy.reason)
            return PolicyResult(
                repair_policy.check, False, repair_policy.reason,
                starter_validation=validation, events=(event,), admission_tier="ADVISORY")
    violations = fixture_vocabulary_violations(
        getattr(check, "fixture", None), brief, sources or {}, tests or {})
    missing = _ungrounded_ticket_literals(check, brief, tests or {})
    tier = admission_tier(
        requirements_ledger,
        check_id=str(getattr(check, "check_id", "")),
        obligation_id=str(getattr(check, "obligation_id", "")),
    )
    events: list[dict[str, object]] = [
        _event("oracle_basis", "DOWNGRADED" if missing else "GROUNDED",
               missing_literals=list(missing),
               basis_before=str(getattr(check, "oracle_basis", "")),
               basis_after=ADVISORY if missing else str(getattr(check, "oracle_basis", ""))),
        _event("fixture_vocabulary", "REJECTED" if violations else "ACCEPTED",
               violations=[dict(item) for item in violations]),
        _event("requirements_ledger", "GATING" if tier == "GATING" else "ADVISORY",
               admission_tier=tier),
    ]
    if starter_root is None:
        validation = StarterValidation(False, "STARTER_VALIDATION_NOT_RUN")
        events.append(_event("starter_validation", "NOT_RUN", reason=validation.reason))
    else:
        validation = validate_check_against_starter(starter_root, check)
        events.append(_event("starter_validation", validation.reason,
                             accepted=validation.accepted))
    template = str(getattr(check, "template", "")).upper()
    compatibility_starter: dict[str, object] | None = None
    if template == "COMPATIBILITY_WITNESS":
        fixture = getattr(check, "fixture", {})
        calls = fixture.get("calls", ()) if isinstance(fixture, Mapping) else ()
        events.append(_event("alias_probe", "KEYWORD_AWARE" if any(
            isinstance(call, Mapping) and call.get("kwargs") for call in calls
        ) else "SIGNATURE_AWARE", call_count=len(calls) if isinstance(calls, Sequence) else 0))
        if (compatibility_admission_enabled()
                and validation.reason.startswith("REJECTED")):
            compatibility_starter = {
                "status": "NON_GATING",
                "reason": validation.reason,
                "starter_failed": True,
            }
            events.append(_event(
                "compatibility_starter_admission", "NON_GATING",
                reason=validation.reason,
                starter_failed=True,
            ))
        elif compatibility_admission_enabled():
            compatibility_starter = {
                "status": "GATING" if validation.accepted else "NON_GATING",
                "reason": validation.reason,
                "starter_failed": not validation.accepted,
            }
            events.append(_event(
                "compatibility_starter_admission",
                "GATING" if validation.accepted else "NON_GATING",
                reason=validation.reason,
                starter_failed=not validation.accepted,
            ))
    updated = _with_policy_provenance(
        check, missing=missing, violations=violations,
        starter=validation.reason, admission_tier=tier,
        compatibility_starter=compatibility_starter)
    accepted = not violations and validation.accepted
    reason = ("REJECTED_FIXTURE_VOCABULARY" if violations else
              validation.reason if not validation.accepted else
              "ADVISORY_ADMISSION_TIER" if tier != "GATING" else
              "DOWNGRADED_ORACLE_BASIS" if missing else "ADMITTED")
    return PolicyResult(updated, accepted, reason, missing, violations, validation,
                        tuple(events), tier)


def _write_public_sources(root: Path, sources: Mapping[str, str]) -> None:
    for raw_path, source in sources.items():
        raw = str(raw_path).replace("\\", "/")
        relative = PurePosixPath(raw)
        if (relative.is_absolute() or ".." in relative.parts
                or "\x00" in raw or (relative.parts and ":" in relative.parts[0])
                or relative.as_posix() != raw
                or relative.suffix != ".py"
                or is_harness_path(Path(raw))
                or any(part.lower() in _FORBIDDEN_PUBLIC_PATH_PARTS
                       or "backup" in part.lower() for part in relative.parts)
                or any(part.startswith(".") for part in relative.parts)):
            raise ValueError("starter validation source path is not public")
        path = root.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(source), encoding="utf-8")


def enforce_generated_checks(checks: Iterable[object], *, brief: object = "",
                             sources: Mapping[str, str] | None = None,
                             tests: Mapping[str, str] | None = None,
                             requirements_ledger: object = None,
                             ) -> EnforcementResult:
    """Filter/admit generated checks after public-starter self-validation."""
    checks = tuple(checks)
    sources = dict(sources or {})
    tests = dict(tests or {})
    admitted: list[object] = []
    checked: list[object] = []
    rejected: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="stateframe_generator_starter_") as directory:
        root = Path(directory)
        _write_public_sources(root, sources)
        for check in checks:
            result = apply_generator_policy(
                check, brief=brief, sources=sources, tests=tests, starter_root=root,
                requirements_ledger=requirements_ledger)
            checked.append(result.check)
            events.extend({**event, "check_ref": str(getattr(check, "check_id", ""))}
                          for event in result.events)
            if result.accepted:
                admitted.append(result.check)
            else:
                rejected.append({
                    "check_id": str(getattr(check, "check_id", "")),
                    "obligation_id": str(getattr(check, "obligation_id", "")),
                    "reason": result.reason,
                    "missing_literals": list(result.missing_literals),
                    "fixture_violations": [dict(item) for item in result.fixture_violations],
                    "starter_reason": (result.starter_validation.reason
                                       if result.starter_validation else None),
                })
    return EnforcementResult(tuple(admitted), tuple(rejected), tuple(events), tuple(checked))


def _ledger_candidate(ledger: object, *, check_id: str = "",
                      obligation_id: str = "", clause: str = "") -> object:
    """Find one ledger row across the supported public serialization shapes."""
    if isinstance(ledger, Sequence) and not isinstance(ledger, (str, bytes, bytearray)):
        for item in ledger:
            if not isinstance(item, Mapping):
                continue
            identifiers = {
                str(item.get("check_id", item.get("check", ""))),
                str(item.get("obligation_id", item.get("obligation", item.get("id", "")))),
                str(item.get("clause", "")),
            }
            requested = {str(value) for value in (check_id, obligation_id, clause) if str(value)}
            identifiers.discard("")
            if requested & identifiers:
                return item
        return None
    if not isinstance(ledger, Mapping):
        return None
    for nested_name in ("obligations", "entries", "checks", "coverage", "dispositions"):
        if nested_name in ledger:
            candidate = _ledger_candidate(
                ledger[nested_name], check_id=check_id,
                obligation_id=obligation_id, clause=clause)
            if candidate is not None:
                return candidate
    for key in (check_id, obligation_id, clause):
        if key and key in ledger:
            return ledger[key]
    if any(key in ledger for key in (
            "check_id", "check", "obligation_id", "obligation", "id", "clause")):
        identifiers = {
            str(ledger.get("check_id", ledger.get("check", ""))),
            str(ledger.get("obligation_id", ledger.get("obligation", ledger.get("id", "")))),
            str(ledger.get("clause", "")),
        }
        requested = {str(value) for value in (check_id, obligation_id, clause) if str(value)}
        identifiers.discard("")
        if requested & identifiers:
            return ledger
    return None


def admission_tier(ledger: object, *, check_id: str = "",
                   obligation_id: str = "", clause: str = "") -> str:
    """Normalize a requirements-ledger admission tier to GATING/ADVISORY."""
    candidate = _ledger_candidate(
        ledger, check_id=check_id, obligation_id=obligation_id, clause=clause)
    if isinstance(candidate, Mapping):
        candidate = next((candidate[key] for key in (
            "admission_tier", "admission", "tier", "authority")
            if key in candidate), None)
    value = str(candidate or "GATING").upper().replace("-", "_")
    if value in {"ADVISORY", "PROVISIONAL", "OPTIONAL", "INFO", "INFORMATIONAL",
                 "NON_GATING", "NONAUTHORITATIVE", "NOT_APPLICABLE", "N_A"}:
        return "ADVISORY"
    return "GATING"


def coverage_disposition(ledger: object, obligation_id: str = "",
                         clause: str = "") -> str:
    """Read an optional requirements-ledger disposition without requiring it."""
    candidate = _ledger_candidate(ledger, obligation_id=obligation_id, clause=clause)
    if isinstance(candidate, Mapping):
        candidate = next((candidate[key] for key in (
            "coverage_disposition", "coverage", "disposition", "tier")
            if key in candidate), None)
    value = str(candidate or "REQUIRED").upper().replace("-", "_")
    if value in {"ADVISORY", "OPTIONAL", "NOT_APPLICABLE", "N_A", "INFO", "INFORMATIONAL"}:
        return value
    return "REQUIRED"


def check_is_gating(check: object) -> bool:
    """Return whether policy provenance permits a check to authorize completion."""
    if str(getattr(check, "oracle_basis", "")).upper() == ADVISORY:
        return False
    provenance = getattr(check, "provenance", {})
    policy = provenance.get("generator_policy") if isinstance(provenance, Mapping) else None
    if not isinstance(policy, Mapping):
        return True
    if str(policy.get("admission", "GATING")).upper() != "GATING":
        return False
    if (compatibility_admission_enabled()
            and str(getattr(check, "template", "")).upper() == "COMPATIBILITY_WITNESS"):
        compatibility = policy.get("compatibility_starter")
        if isinstance(compatibility, Mapping):
            if (str(compatibility.get("status", "")).upper() != "GATING"
                    or bool(compatibility.get("starter_failed"))):
                return False
    starter = str(policy.get("starter_validation", ""))
    return not (starter.startswith("REJECTED")
                or starter in {"NOT_RUN", "STARTER_VALIDATION_NOT_RUN"})


def _public_clauses(brief: object) -> tuple[str, ...]:
    text = str(brief or "")
    if not text.strip():
        return ()
    try:
        from kernel_boundary_checks import _clause_list
        return tuple(item for item in _clause_list(text) if str(item).strip())
    except (ImportError, TypeError, ValueError, SyntaxError):
        return tuple(item.strip() for item in re.split(
            r"(?<=[.!?])\s+|;\s*", text) if item.strip())


def _coverage_status_id(obligation_id: str, clause: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(clause)).strip("-").lower()[:48]
    return f"coverage-{obligation_id}-{slug or 'clause'}"


def enforce_bundle(bundle: object, *, brief: object = "",
                   sources: Mapping[str, str] | None = None,
                   tests: Mapping[str, str] | None = None,
                   requirements_ledger: object = None) -> object:
    """Apply generator policy to every check in a static or augmented bundle.

    The bundle compiler and the frozen-bundle loader share this function so a
    preservation check added after loading cannot bypass starter validation or
    the fixture vocabulary guard.
    """
    if not enabled() or bundle is None:
        return bundle
    from kernel_obligation_assess import ObligationSpec

    original_checks = tuple(getattr(bundle, "checks", ()) or ())
    enforcement = enforce_generated_checks(
        original_checks, brief=brief, sources=sources or {}, tests=tests or {},
        requirements_ledger=requirements_ledger)
    checked_by_id = {
        str(getattr(check, "check_id", "")): check
        for check in enforcement.checked
    }
    original_by_id = {
        str(getattr(check, "check_id", "")): check
        for check in original_checks
    }
    cell_rewrites: dict[str, object] = {}
    for check_id, checked in checked_by_id.items():
        original = original_by_id.get(check_id)
        old_cell = getattr(original, "contract_cell", None)
        new_cell = getattr(checked, "contract_cell", None)
        if old_cell is not None and new_cell is not None:
            cell_rewrites[str(getattr(old_cell, "contract_cell_hash", ""))] = new_cell
    cells = tuple(
        cell_rewrites.get(str(getattr(cell, "contract_cell_hash", "")), cell)
        for cell in (getattr(bundle, "contract_cells", ()) or ())
    )
    admitted_by_obligation: dict[str, list[str]] = {}
    for check in enforcement.checks:
        if check_is_gating(check):
            admitted_by_obligation.setdefault(
                str(getattr(check, "obligation_id", "")), []).append(
                    str(getattr(check, "check_id", "")))

    specs: list[object] = []
    coverage_map: list[dict[str, object]] = []
    existing_ids: set[str] = set()
    for spec in tuple(getattr(bundle, "specs", ()) or ()):
        obligation_id = str(getattr(spec, "obligation_id", ""))
        clause = str(getattr(spec, "clause", ""))
        disposition = coverage_disposition(
            requirements_ledger, obligation_id, clause)
        required = normative_clause(clause, disposition)
        if disposition == "REQUIRED":
            required = bool(getattr(spec, "coverage_required", False)) or required
        ids = tuple(admitted_by_obligation.get(obligation_id, ()))
        spec_cells = tuple(
            cell_rewrites.get(str(getattr(cell, "contract_cell_hash", "")), cell)
            for cell in (getattr(spec, "contract_cells", ()) or ())
        )
        coverage = {
            "obligation_id": obligation_id,
            "clause": clause,
            "normative": required,
            "evidence_check_ids": list(ids),
            "covered": bool(ids),
            "evidence_status": "COVERED" if ids else "MISSING",
            "status_level": True,
            "disposition": disposition,
            "status_check_id": _coverage_status_id(obligation_id, clause),
        }
        provenance = copy.deepcopy(getattr(spec, "provenance", {}) or {})
        provenance["coverage"] = coverage
        specs.append(dataclasses.replace(
            spec, acceptance_policy=ids, provenance=provenance,
            coverage_required=required, coverage_disposition=disposition,
            contract_cells=spec_cells,
        ))
        existing_ids.add(obligation_id)
        coverage_map.append(coverage)

    # Static bundles can predate a clause added to the public brief.  Materialize
    # a blocking empty obligation for it instead of allowing the old bundle to
    # make completion look covered.
    known_clauses = {str(getattr(spec, "clause", "")) for spec in specs}
    for index, clause in enumerate(_public_clauses(brief)):
        obligation_id = f"obl-{index:03d}"
        if obligation_id in existing_ids or clause in known_clauses:
            continue
        disposition = coverage_disposition(requirements_ledger, obligation_id, clause)
        required = normative_clause(clause, disposition)
        if not required:
            continue
        coverage = {
            "obligation_id": obligation_id,
            "clause": clause,
            "normative": True,
            "evidence_check_ids": [],
            "covered": False,
            "evidence_status": "MISSING",
            "status_level": True,
            "disposition": disposition,
            "status_check_id": _coverage_status_id(obligation_id, clause),
        }
        specs.append(ObligationSpec(
            obligation_id,
            {"path": "brief.md", "text": clause},
            "VALIDATED", tuple(sorted(sources or {})), (), clause,
            {"compiler": "stateframe-check-generator-policy/1",
             "coverage": coverage}, (),
            coverage_required=True, coverage_disposition=disposition,
        ))
        coverage_map.append(coverage)

    provenance = copy.deepcopy(getattr(bundle, "provenance", {}) or {})
    old_events = provenance.get("generator_events", ())
    merged_events: list[dict[str, object]] = []
    prior_events = (list(old_events) if isinstance(old_events, Sequence)
                    and not isinstance(old_events, (str, bytes, bytearray)) else [])
    prior_events.extend(enforcement.events)
    prior_events.extend({
        "schema": POLICY_SCHEMA,
        "rule": "clause_coverage",
        "status": "COVERED" if item.get("covered") else "MISSING",
        **{key: copy.deepcopy(value) for key, value in item.items()
           if key not in {"status"}},
    } for item in coverage_map)
    for event in prior_events:
        if not isinstance(event, Mapping):
            continue
        normalized = dict(event)
        if normalized not in merged_events:
            merged_events.append(normalized)
    provenance.update({
        "generator_policy": {
            "schema": POLICY_SCHEMA,
            "enabled": True,
            "rejected_checks": [dict(item) for item in enforcement.rejected],
        },
        "generator_events": merged_events,
        "clause_coverage": coverage_map,
    })
    return dataclasses.replace(
        bundle,
        specs=tuple(specs),
        checks=tuple(enforcement.checks),
        provenance=provenance,
        contract_cells=cells,
    )


def normative_clause(clause: object, disposition: str = "REQUIRED") -> bool:
    if str(disposition).upper() in {"ADVISORY", "OPTIONAL", "NOT_APPLICABLE", "N_A",
                                    "INFO", "INFORMATIONAL"}:
        return False
    text = str(clause or "")
    if re.search(r"\b(?:non[- ]?normative|informational|advisory|not\s+required)\b",
                 text, re.I):
        return False
    if re.search(r"\b(?:optional|may|could|might)\b", text, re.I) and not re.search(
            r"\b(?:must|shall|required|reject|preserve|accept|ensure)\b", text, re.I):
        return False
    return bool(text.strip())


__all__ = [
    "ADVISORY", "COMPAT_ADMISSION_ENV", "EnforcementResult", "MODE_ENV", "POLICY_SCHEMA", "PolicyResult",
    "StarterValidation", "admission_tier", "apply_generator_policy",
    "compatibility_admission_enabled",
    "check_is_gating", "coverage_disposition", "enabled", "enforce_bundle",
    "enforce_generated_checks",
    "fixture_vocabulary_violations", "mode", "normative_clause",
    "validate_check_against_starter",
]
