"""Precision-first admission for generated checks.

The legacy check executor remains the observation engine. This module is the
host policy layer that decides whether an observation can become a blocking
counterexample. It intentionally uses duck-typed check/receipt records so it
can sit beside the existing ``kernel_obligation_assess`` dataclasses without a
module cycle.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


ENV = "STATEFRAME_REQUIREMENTS_LEDGER"
SCHEMA_VERSION = "requirements-admission/1"

REJECTED = "REJECTED"
ADVISORY = "ADVISORY"
SHADOW = "SHADOW"
GATING = "GATING"
TIERS = frozenset({REJECTED, ADVISORY, SHADOW, GATING})

SOURCE_ALIGNMENT = "source_alignment"
BINDING = "binding"
FIXTURE = "fixture"
ORACLE = "oracle"
RUNNER = "runner"
CERTIFICATE_STAGES = (SOURCE_ALIGNMENT, BINDING, FIXTURE, ORACLE, RUNNER)

UPHELD = "UPHELD"
CHECK_INVALID = "CHECK_INVALID"
CHECK_REBOUND = "CHECK_REBOUND"
INTERPRETATION_OPEN = "INTERPRETATION_OPEN"

GROUNDS = frozenset({
    "not_entailed", "wrong_binding", "invalid_fixture", "wrong_observable",
    "harness_error", "stale_evidence", "conflicting_source",
})

_NARROW_TEMPLATES = {
    "OPTIONAL_INPUT": "explicit-default/1",
    "REJECT_WITH_STATUS": "specified-error-mapping/1",
    "ERROR_TRANSLATION": "specified-error-mapping/1",
    "PRESERVE_INPUT": "non-mutation/1",
    "PRESERVE_EXISTING_VALIDATION": "non-mutation/1",
    "PRESERVE_CALL_SHAPE": "compatibility-alias/1",
    "COMPATIBILITY_WITNESS": "compatibility-alias/1",
    "VALUE_TRANSPORT": "pass-through/1",
    "OUTPUT_DELTA": "exact-arithmetic/1",
    "ROUNDING_DISCRIMINATION": "exact-arithmetic/1",
    "SUM_CONSISTENCY": "exact-arithmetic/1",
}
_INTERNAL_HELPER = re.compile(r"(?:models?|model|validation|parser|merge):[^:]*\.", re.I)
_SYMBOL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def mode() -> str:
    value = os.environ.get(ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if dataclasses.is_dataclass(value):
        return copy.deepcopy(dataclasses.asdict(value))
    return {}


def _seq(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return tuple(value)
    return ()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode("utf-8")).hexdigest()


def _provenance(check: object) -> Mapping[str, Any]:
    value = _get(check, "provenance", {})
    return value if isinstance(value, Mapping) else {}


def _entry_point(check: object) -> str:
    return str(_get(check, "entry_point", _get(check, "entry", "")) or "").strip()


def _tail(entry_point: str) -> str:
    value = entry_point.split(":", 1)[1] if ":" in entry_point else entry_point
    return value.rsplit(".", 1)[-1]


def _ticket_source(ledger: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(ledger, Mapping):
        return None
    ticket_id = str(ledger.get("ticket_source_id", ""))
    return next((source for source in ledger.get("sources", ())
                 if isinstance(source, Mapping) and str(source.get("id")) == ticket_id), None)


def _ticket_text(ledger: Mapping[str, Any] | None) -> str:
    source = _ticket_source(ledger)
    return str(source.get("text", "")) if source is not None else ""


def _spans(ledger: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("id")): row for row in (ledger or {}).get("spans", ())
            if isinstance(row, Mapping) and row.get("id")}


def _binding_rows(ledger: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in (ledger or {}).get("bindings", ()) if isinstance(row, Mapping))


def _explicit_names(ledger: Mapping[str, Any] | None) -> set[str]:
    names: set[str] = set()
    for row in _binding_rows(ledger):
        if str(row.get("status", "")) in {"unresolved", "ambiguous"}:
            # An unresolved binding is still a public name if the ticket wrote
            # it explicitly; it is not permission to invent a sibling helper.
            pass
        name = str(row.get("name", "")).strip()
        if name and name != "unresolved-ticket-subject":
            names.add(name.rsplit(".", 1)[-1])
            names.add(name)
    text = _ticket_text(ledger)
    for token in re.findall(r"`([^`\n]+)`", text):
        token = token.strip()
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", token):
            names.add(token)
            names.add(token.rsplit(".", 1)[-1])
    return names


def _requirement_for_check(check: object, ledger: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(ledger, Mapping):
        return None
    provenance = _provenance(check)
    clause = provenance.get("clause_span")
    clause_text = str(clause.get("text", "")) if isinstance(clause, Mapping) else ""
    obligation = str(_get(check, "obligation_id", _get(check, "obligation", "")) or "")
    spans = _spans(ledger)
    for requirement in ledger.get("requirements", ()):
        if not isinstance(requirement, Mapping):
            continue
        exact = str(requirement.get("exact_clause", ""))
        if clause_text and (clause_text == exact or clause_text in exact or exact in clause_text):
            return requirement
        requirement_id = str(requirement.get("id", ""))
        if obligation and obligation == requirement_id:
            return requirement
        if (obligation.startswith("obl-") and obligation[4:].isdigit()
                and requirement_id.startswith("R")
                and requirement_id[1:].isdigit()
                and int(obligation[4:]) + 1 == int(requirement_id[1:])):
            return requirement
        for span_id in requirement.get("source_span_ids", ()):
            span = spans.get(str(span_id))
            if clause_text and isinstance(span, Mapping) and clause_text in str(span.get("quote", "")):
                return requirement
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
        try:
            from kernel_requirement_alignment import align_check_to_requirement
            aligned = align_check_to_requirement(check, ledger)
        except (ImportError, TypeError, ValueError):
            aligned = None
        if isinstance(aligned, Mapping) and isinstance(aligned.get("requirement"), Mapping):
            result = copy.deepcopy(dict(aligned["requirement"]))
            clause = aligned.get("clause")
            if isinstance(clause, str) and clause.strip():
                result["exact_clause"] = clause
            result["alignment_evidence"] = copy.deepcopy(aligned.get("evidence", {}))
            return result
    return None


def _source_alignment(check: object, ledger: Mapping[str, Any] | None) -> tuple[bool, str]:
    provenance = _provenance(check)
    requirement = _requirement_for_check(check, ledger)
    if requirement is not None:
        return True, "check clause aligns with an exact ticket span"
    if str(_get(check, "oracle_basis", "")).upper() == "PUBLIC_TEST":
        public = provenance.get("public_test")
        if public is not None and _get(check, "selection", ()):
            return True, "explicit public acceptance-test source"
    spans = provenance.get("source_span_ids", provenance.get("support_span_ids"))
    if _seq(spans):
        known = set(_spans(ledger))
        if known and set(str(item) for item in spans).issubset(known):
            return True, "host supplied support spans"
    return False, "no exact ticket/public source alignment"


def _binding_alignment(check: object, ledger: Mapping[str, Any] | None) -> tuple[bool, str]:
    entry = _entry_point(check)
    if not entry or ":" not in entry:
        return False, "missing public entrypoint binding"
    tail = _tail(entry)
    names = _explicit_names(ledger)
    if tail in names or entry.replace(":", ".") in names:
        return True, "entrypoint is role-qualified by a named ticket symbol"
    basis = str(_get(check, "oracle_basis", "")).upper()
    provenance = _provenance(check)
    if basis == "PUBLIC_TEST" and provenance.get("public_test") is not None:
        # A public acceptance test may exercise an existing public API without
        # the ticket spelling its Python import.  A dotted model/class helper
        # is still not promoted: it is the held-out false-blocker shape.
        if ":" in entry and "." in entry.split(":", 1)[1]:
            return False, "wrong_binding"
        return True, "public acceptance test binds the tested public surface"
    return False, "not_entailed"


def _fixture_alignment(check: object) -> tuple[bool, str]:
    provenance = _provenance(check)
    override = provenance.get("fixture_certificate")
    if isinstance(override, Mapping) and override.get("verdict") in {"supported", "SUPPORTED"}:
        return True, str(override.get("reason", "host fixture certificate"))
    fixture = _get(check, "fixture", None)
    if fixture is None:
        return False, "fixture is absent; omitted is not null"
    if provenance.get("fixture_valid") is False:
        return False, "host marked the fixture invalid"
    if isinstance(fixture, Mapping) and not fixture:
        return False, "fixture has no invocation data"
    return True, "fixture is present and its varied dimensions are explicit"


def _oracle_alignment(check: object, ledger: Mapping[str, Any] | None) -> tuple[bool, str]:
    provenance = _provenance(check)
    oracle = _get(check, "oracle", None)
    if isinstance(oracle, Mapping):
        if _seq(oracle.get("unsupported_assertions")):
            return False, "oracle contains unsupported assertions"
        assertions = oracle.get("assertions")
        if _seq(assertions) and all(
                isinstance(item, Mapping) and _seq(item.get("support_span_ids"))
                for item in assertions):
            return True, "every oracle assertion has support spans"
        return False, "oracle assertions are not completely source-supported"
    if provenance.get("oracle_valid") is False:
        return False, "host marked the oracle invalid"
    basis = str(_get(check, "oracle_basis", "")).upper()
    template = str(_get(check, "template", "")).upper()
    if basis == "PUBLIC_TEST" and provenance.get("public_test") is not None:
        return True, "public acceptance assertion is the oracle"
    if template in {"PRESERVE_CALL_SHAPE", "COMPATIBILITY_WITNESS", "PRESERVE_INPUT"}:
        if _get(check, "expected", None) is None and _get(check, "fixture", None) is not None:
            return True, "preservation oracle is the named call/input relation"
    if _get(check, "expected", None) is not None:
        requirement = _requirement_for_check(check, ledger)
        if requirement is not None and requirement.get("contract", {}).get("expectations"):
            return True, "expected value is supported by the typed contract"
    return False, "oracle contains an unsupported or unsourced assertion"


def _runner_alignment(check: object, *, current_revision: object = None) -> tuple[bool, str]:
    provenance = _provenance(check)
    if provenance.get("execution_integrity") is True:
        return True, "host recorded target assertion reachability"
    if provenance.get("runner_certificate") is True:
        return True, "host runner certificate is present"
    if str(_get(check, "oracle_basis", "")).upper() == "PUBLIC_TEST":
        selection = _seq(_get(check, "selection", ()))
        if selection and all(str(item).startswith(("tests/", "test/")) for item in selection):
            return True, "public test selector is target-bound"
    if current_revision is not None and provenance.get("checked_revision") == current_revision:
        return True, "check is bound to the current revision"
    return False, "execution integrity certificate is missing"


def _stage_row(check_id: str, stage: str, supported: bool, reason: str,
               *, evidence_ids: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "id": f"K-{check_id}-{stage}", "stage": stage,
        "verdict": "supported" if supported else "unknown",
        "method": "public_fact_validation" if stage in {BINDING, FIXTURE, RUNNER} else "deterministic_template",
        "producer": "host", "producer_version": "design-55/1",
        "evidence_ids": list(evidence_ids) or [f"E-{check_id}-{stage}"],
        "normative_assumptions": [], "limitations": [reason],
    }


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    check_id: str
    tier: str
    reason: str
    certificates: tuple[dict[str, Any], ...] = ()
    authorization: dict[str, Any] | None = None
    requirement_refs: tuple[dict[str, Any], ...] = ()
    symbol: str = ""
    template_id: str | None = None
    repairable: bool = field(default=False, repr=False, compare=False)
    alignment_evidence: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False)

    @property
    def gating(self) -> bool:
        return self.tier == GATING

    @property
    def tier_lower(self) -> str:
        return self.tier.lower()

    @property
    def certificate_map(self) -> dict[str, Mapping[str, Any]]:
        return {str(row.get("stage")): row for row in self.certificates}

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": SCHEMA_VERSION, "check_id": self.check_id,
            "tier": self.tier_lower, "reason": self.reason,
            "certificate_ids": [str(row.get("id")) for row in self.certificates],
            "certificates": copy.deepcopy(list(self.certificates)),
            "authorization": copy.deepcopy(self.authorization),
            "requirement_refs": copy.deepcopy(list(self.requirement_refs)),
            "symbol": self.symbol, "template_id": self.template_id,
        }
        if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
            result.update({
                "repairable": bool(self.repairable),
                "alignment_evidence": copy.deepcopy(self.alignment_evidence),
            })
        return result


def _provided_certificates(check: object, certificates: object = None) -> dict[str, Mapping[str, Any]]:
    raw = certificates
    if raw is None:
        provenance = _provenance(check)
        candidate = provenance.get("certificates")
        if candidate is None:
            candidate = provenance.get("admission_certificates")
        # The boundary checks field certificate is not the five-certificate package.
        if isinstance(candidate, Mapping) and any(stage in candidate for stage in CERTIFICATE_STAGES):
            raw = candidate
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            raw = candidate
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        for stage, row in raw.items():
            if stage in CERTIFICATE_STAGES and isinstance(row, Mapping):
                result[stage] = row
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for row in raw:
            if isinstance(row, Mapping) and str(row.get("stage")) in CERTIFICATE_STAGES:
                result[str(row["stage"])] = row
    return result


def _is_explicit_public_acceptance(check: object) -> bool:
    provenance = _provenance(check)
    return (str(_get(check, "oracle_basis", "")).upper() == "PUBLIC_TEST"
            or str(_get(check, "generator", {}).get("kind", "") if isinstance(_get(check, "generator", {}), Mapping) else "").lower() == "public_test"
            or provenance.get("public_test") is not None)


def _generator_policy_cap(check: object) -> str | None:
    """Return a demotion reason when check_generator_fixes did not admit ``check`` as gating.

    Only consulted when STATEFRAME_CHECK_GENERATOR_FIXES=on; the generator policy can demote a check but
    never promote one, so admission can only ever be the stricter of the two layers.
    """
    try:
        from kernel_check_generator_fixes import check_is_gating, enabled as generator_fixes_enabled
    except ImportError:  # pragma: no cover - policy module absent in minimal trees
        return None
    if not generator_fixes_enabled():
        return None
    policy = _provenance(check).get("generator_policy")
    if not isinstance(policy, Mapping) and str(_get(check, "oracle_basis", "")).upper() != ADVISORY:
        return None
    return None if check_is_gating(check) else "generator_policy_advisory"


def admit_check(check: object, ledger: Mapping[str, Any] | None = None, *,
                certificates: object = None, current_revision: object = None,
                explicit_public_acceptance: bool | None = None,
                ticket: object = None, ticket_text: object = None,
                public_api: object = None, public_contract: object = None,
                contract: object = None, sources: object = None) -> AdmissionDecision:
    """Host-admit one generated check without trusting solver-authored status."""
    check_id = str(_get(check, "check_id", _get(check, "id", "")) or "")
    template = str(_get(check, "template", "") or "").upper()
    entry = _entry_point(check)
    symbol = entry or str(_get(check, "symbol", "") or "")
    public = _is_explicit_public_acceptance(check) if explicit_public_acceptance is None else bool(explicit_public_acceptance)
    alignment, alignment_reason = _source_alignment(check, ledger)
    binding_ok, binding_reason = _binding_alignment(check, ledger)
    fixture_ok, fixture_reason = _fixture_alignment(check)
    oracle_ok, oracle_reason = _oracle_alignment(check, ledger)
    runner_ok, runner_reason = _runner_alignment(check, current_revision=current_revision)
    supplied = _provided_certificates(check, certificates)
    rows: list[dict[str, Any]] = []
    checks = {
        SOURCE_ALIGNMENT: (alignment, alignment_reason), BINDING: (binding_ok, binding_reason),
        FIXTURE: (fixture_ok, fixture_reason), ORACLE: (oracle_ok, oracle_reason),
        RUNNER: (runner_ok, runner_reason),
    }
    # A ticket check needs an explicit host-produced five-certificate package
    # (or a host authorization marker).  Merely having a familiar legacy
    # template is not authorization; this is what keeps an inferred helper
    # assertion in SHADOW/REJECTED territory.
    can_derive = public or (template in _NARROW_TEMPLATES and (
        _provenance(check).get("admission_authorization") is not None
        or set(supplied) == set(CERTIFICATE_STAGES)))
    for stage in CERTIFICATE_STAGES:
        if stage in supplied:
            raw = supplied[stage]
            verdict = str(raw.get("verdict", "unknown")).lower()
            supported = verdict == "supported"
            limitations = _seq(raw.get("limitations", ()))
            reason = str(raw.get("reason") or (limitations[0] if limitations else "certificate supplied by host"))
            method = str(raw.get("method", "public_fact_validation"))
            if method not in {"deterministic_template", "public_fact_validation",
                              "independent_semantic_review", "formal_derivation"}:
                method = "public_fact_validation"
            rows.append({
                "id": str(raw.get("id") or f"K-{check_id}-{stage}"), "stage": stage,
                "verdict": "supported" if supported else "unknown",
                "method": method,
                "producer": "host", "producer_version": str(raw.get("producer_version", "design-55/1")),
                "evidence_ids": [str(item) for item in _seq(raw.get("evidence_ids", ())) if str(item)] or [f"E-{check_id}-{stage}"],
                "normative_assumptions": [str(item) for item in _seq(raw.get("normative_assumptions", ()))],
                "limitations": [reason],
            })
        elif can_derive:
            supported, reason = checks[stage]
            rows.append(_stage_row(check_id, stage, supported, reason))
        else:
            rows.append(_stage_row(check_id, stage, False, "certificate not supplied"))
    requirement = _requirement_for_check(check, ledger)
    refs = ({"id": str(requirement.get("id")), "version": int(requirement.get("version", 1))}
            if requirement is not None else None)
    requirement_refs = (refs,) if refs is not None else ()
    if not alignment:
        tier, reason = REJECTED, "not_entailed"
    elif not binding_ok:
        tier, reason = (REJECTED if binding_reason == "not_entailed" else ADVISORY), binding_reason
    elif not fixture_ok or not oracle_ok:
        tier, reason = ADVISORY, "invalid_fixture" if not fixture_ok else "wrong_observable"
    elif all(str(row.get("verdict")) == "supported" for row in rows) and can_derive:
        tier, reason = GATING, "admitted precision-first check"
    elif any(str(row.get("verdict")) == "unknown" for row in rows):
        tier, reason = SHADOW, "missing_certificate"
    else:
        tier, reason = SHADOW, "certificate_not_supported"
    policy_cap = _generator_policy_cap(check)
    if policy_cap is not None and tier in {GATING, SHADOW}:
        tier, reason = ADVISORY, policy_cap
    repairable = False
    alignment_evidence: dict[str, Any] = {}
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
        try:
            from kernel_requirement_alignment import admission_adjustment
            adjustment = admission_adjustment(check, ledger, tier, reason)
            tier = str(adjustment.get("tier", tier))
            reason = str(adjustment.get("reason", reason))
            repairable = bool(adjustment.get("repairable", False))
            aligned = adjustment.get("alignment")
            if isinstance(aligned, Mapping):
                alignment_evidence = copy.deepcopy(dict(aligned.get("evidence", {})))
                if requirement is None and isinstance(aligned.get("requirement"), Mapping):
                    requirement = aligned["requirement"]
                    refs = {"id": str(requirement.get("id")),
                            "version": int(requirement.get("version", 1))}
                    requirement_refs = (refs,)
        except (ImportError, TypeError, ValueError):
            pass
    literal_reason = None
    if os.environ.get("STATEFRAME_ORACLE_TICKET_LITERALS", "off").strip().lower() == "on":
        from kernel_oracle_literal_canary import admission_reason
        literal_reason = admission_reason(check, ledger)
    if literal_reason is not None:
        # This cap is applied after every other admission adapter so no later
        # alignment policy can promote an unsupported expected atom.
        tier, reason = ADVISORY, literal_reason
    grounding_reason = None
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        try:
            from kernel_oracle_grounding import admission_reason as grounding_admission_reason
            grounding_reason = grounding_admission_reason(
                check, ticket=ticket if ticket is not None else ticket_text,
                ledger=ledger, public_api=public_api,
                public_contract=public_contract, contract=contract, sources=sources)
        except (ImportError, TypeError, ValueError):
            grounding_reason = None
    if grounding_reason is not None:
        # An unbound host expectation is an observation gap, never an
        # admission failure that can become a required repair.
        tier, reason = ADVISORY, grounding_reason
    authorization = None
    if tier == GATING:
        kind = "explicit_public_acceptance_test" if public else "calibrated_template_family"
        authorization = {
            "kind": kind,
            "policy_ref": "explicit-public-acceptance/1" if public else f"{_NARROW_TEMPLATES[template]}",
            "evidence_ids": [str(row.get("id")) for row in rows],
        }
    return AdmissionDecision(
        check_id=check_id, tier=tier, reason=reason,
        certificates=tuple(rows), authorization=authorization,
        requirement_refs=requirement_refs, symbol=symbol,
        template_id=_NARROW_TEMPLATES.get(template), repairable=repairable,
        alignment_evidence=alignment_evidence,
    )


def admit_checks(checks: object, ledger: Mapping[str, Any] | None = None, *,
                 current_revision: object = None, ticket: object = None,
                 ticket_text: object = None, public_api: object = None,
                 public_contract: object = None, contract: object = None,
                 sources: object = None) -> tuple[AdmissionDecision, ...]:
    return tuple(admit_check(
        check, ledger, current_revision=current_revision,
        ticket=ticket, ticket_text=ticket_text, public_api=public_api,
        public_contract=public_contract, contract=contract, sources=sources)
                 for check in _seq(checks))


def _receipt_status(receipt: object) -> str:
    for key in ("underlying_status", "status", "result", "result_class"):
        value = _get(receipt, key, None)
        if value is not None:
            text = str(value).upper()
            if text != "REUSED":
                return text
    return ""


def is_admitted_counterexample(check: object, receipt: object, decision: AdmissionDecision | None = None,
                               *, current_revision: object = None) -> bool:
    if decision is None:
        decision = admit_check(check)
    if decision.tier != GATING or str(_get(check, "lifecycle", "active")).lower() != "active":
        return False
    if _receipt_status(receipt) != "COUNTEREXAMPLE":
        return False
    execution = str(_get(receipt, "execution", "EXECUTED")).upper()
    if execution != "EXECUTED":
        return False
    attribution = _get(receipt, "attribution", None)
    if attribution is not None and str(attribution) != "target_assertion":
        return False
    observed = _get(receipt, "observed", _get(receipt, "observed_values", None))
    expected = _get(receipt, "expected", None)
    counterexample = _get(receipt, "counterexample", None)
    if isinstance(counterexample, Mapping) and any(
            key in counterexample for key in ("missing_symbol", "binding_error", "wrong_binding")):
        signature_gap = False
        if os.environ.get("STATEFRAME_NAMED_LAYER_REPAIR", "off").strip().lower() == "on":
            # autopsy-a3b-r44c-ldev01 A: a ticket-literal signature gap on a resolved target is a product
            # counterexample; missing symbols / wrong bindings / unaligned rows stay refused.
            from kernel_named_layer_repair import is_signature_gap
            row = receipt if isinstance(receipt, Mapping) else {}
            signature_gap = ("missing_symbol" not in counterexample and "wrong_binding" not in counterexample
                             and is_signature_gap(row, row))
        if not signature_gap:
            return False
    if observed is None and counterexample is None:
        return False
    if current_revision is not None:
        revision = _get(receipt, "repository_revision", _get(receipt, "snapshot_digest", None))
        if revision is not None and str(revision) != str(current_revision):
            return False
    return True


def reduce_assessments(assessments: object, *, checks: object = (), ledger: Mapping[str, Any] | None = None,
                       current_revision: object = None, ticket: object = None,
                       ticket_text: object = None, public_api: object = None,
                       public_contract: object = None, contract: object = None,
                       sources: object = None) -> tuple[object, ...]:
    """Demote failures from non-admitted checks to observations.

    The typed legacy assessment is retained for presentation, but its receipt
    collection is cleared on a shadow-only failure so the old ``blocking``
    adapter cannot accidentally veto completion.
    """
    definitions = {str(_get(check, "check_id", _get(check, "id", ""))): check
                   for check in _seq(checks)}
    decisions = {ident: admit_check(
        check, ledger, current_revision=current_revision, ticket=ticket,
        ticket_text=ticket_text, public_api=public_api,
        public_contract=public_contract, contract=contract, sources=sources)
        for ident, check in definitions.items()}
    result: list[object] = []
    oracle_grounding_on = (
        os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on")
    for item in _seq(assessments):
        receipts = _seq(_get(item, "receipts", ()))
        receipt_check_ids = {
            str(_get(receipt, "check_id", "")) for receipt in receipts
        }
        unbound = next(
            (decisions[ident] for ident in receipt_check_ids
             if (ident in decisions
                 and str(_get(decisions[ident], "reason", "")) == "UNBOUND_OBSERVABLE")),
            None,
        )
        admitted = any(
            is_admitted_counterexample(definitions.get(str(_get(receipt, "check_id", ""))), receipt,
                                       decisions.get(str(_get(receipt, "check_id", ""))),
                                       current_revision=current_revision)
            for receipt in receipts
            if definitions.get(str(_get(receipt, "check_id", ""))) is not None
        )
        if (oracle_grounding_on
                and (unbound is not None
                     or (str(_get(item, "reason", "")) == "UNBOUND_OBSERVABLE"
                         and not admitted))):
            if dataclasses.is_dataclass(item):
                try:
                    item = dataclasses.replace(item, verdict="UNKNOWN",
                                               reason="UNBOUND_OBSERVABLE", receipts=())
                except (TypeError, ValueError):
                    item = copy.deepcopy(item)
            elif isinstance(item, Mapping):
                updates = {"verdict": "UNKNOWN", "reason": "UNBOUND_OBSERVABLE",
                           "receipts": []}
                if "status" in item:
                    updates["status"] = "UNKNOWN"
                if "blocking" in item:
                    updates["blocking"] = False
                item = {**item, **updates}
            result.append(item)
            continue
        if (oracle_grounding_on
                and str(_get(item, "reason", "")) == "UNBOUND_OBSERVABLE"
                and admitted):
            if dataclasses.is_dataclass(item):
                try:
                    item = dataclasses.replace(item, verdict="UNSATISFIED",
                                               reason="COUNTEREXAMPLE")
                except (TypeError, ValueError):
                    item = copy.deepcopy(item)
            elif isinstance(item, Mapping):
                updates = {"verdict": "UNSATISFIED", "reason": "COUNTEREXAMPLE"}
                if "status" in item:
                    updates["status"] = "COUNTEREXAMPLE"
                if "blocking" in item:
                    updates["blocking"] = True
                item = {**item, **updates}
        if (str(_get(item, "reason", "")).upper() == "COUNTEREXAMPLE"
                and not admitted and receipts):
            if dataclasses.is_dataclass(item):
                try:
                    item = dataclasses.replace(item, verdict="UNKNOWN",
                                               reason="NON_ADMITTED_OBSERVATION", receipts=())
                except (TypeError, ValueError):
                    item = copy.deepcopy(item)
            elif isinstance(item, Mapping):
                item = {**item, "verdict": "UNKNOWN", "reason": "NON_ADMITTED_OBSERVATION",
                        "receipts": []}
        result.append(item)
    return tuple(result)


def completion_terminal_state(assessments: object, decisions: object = ()) -> str:
    decision_map = {str(_get(row, "check_id", "")): row for row in _seq(decisions)}
    gating_counterexample = False
    open_non_gating = False
    for assessment in _seq(assessments):
        for receipt in _seq(_get(assessment, "receipts", ())):
            check_id = str(_get(receipt, "check_id", ""))
            decision = decision_map.get(check_id)
            tier = str(_get(decision, "tier", "")).upper() if decision is not None else ""
            if tier == GATING and _receipt_status(receipt) == "COUNTEREXAMPLE":
                gating_counterexample = True
            elif _receipt_status(receipt) == "COUNTEREXAMPLE":
                open_non_gating = True
        assessment_status = str(_get(assessment, "verdict", _get(assessment, "status", ""))).upper()
        if assessment_status in {"UNKNOWN", "UNSATISFIED", "OPEN"}:
            open_non_gating = True
    if gating_counterexample:
        return "blocked"
    if open_non_gating:
        return "submitted_with_unverified_requirements"
    return "accepted"


def _evidence_rows(evidence: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(evidence, Mapping):
        return (evidence,)
    return tuple(item for item in _seq(evidence) if isinstance(item, Mapping))


def dispute(check_id: object, ground: object, claim: object, evidence: object = (), *,
            checks: object = (), ledger: Mapping[str, Any] | None = None,
            receipts: object = (), current_revision: object = None) -> dict[str, Any]:
    """Adjudicate a solver challenge using deterministic host facts first."""
    ident = str(check_id or "")
    ground_text = str(ground or "").strip().lower()
    check = next((item for item in _seq(checks)
                  if str(_get(item, "check_id", _get(item, "id", ""))) == ident), None)
    result: dict[str, Any] = {
        "schema": "requirements-dispute/1", "check_id": ident,
        "ground": ground_text, "claim": str(claim or ""),
        "evidence": copy.deepcopy(list(_evidence_rows(evidence))),
        "outcome": INTERPRETATION_OPEN, "status": INTERPRETATION_OPEN,
        "reason": "inconclusive host adjudication; check demoted",
        "satisfies_requirement": False,
    }
    if ground_text not in GROUNDS or check is None:
        return result
    decision = admit_check(check, ledger, current_revision=current_revision)
    if ground_text == "not_entailed":
        if decision.reason == "not_entailed" or not decision.requirement_refs:
            result.update(outcome=CHECK_INVALID, status=CHECK_INVALID,
                          reason="the check is not entailed by an exact ticket span")
        else:
            result.update(outcome=UPHELD, status=UPHELD,
                          reason="ticket span and requirement binding support the check")
    elif ground_text == "wrong_binding":
        binding_ok, reason = _binding_alignment(check, ledger)
        result.update(
            outcome=UPHELD if binding_ok else CHECK_REBOUND,
            status=UPHELD if binding_ok else CHECK_REBOUND,
            reason="binding is role-qualified" if binding_ok else reason,
        )
    elif ground_text == "invalid_fixture":
        valid, reason = _fixture_alignment(check)
        result.update(outcome=UPHELD if valid else CHECK_INVALID,
                      status=UPHELD if valid else CHECK_INVALID, reason=reason)
    elif ground_text == "wrong_observable":
        valid, reason = _oracle_alignment(check, ledger)
        result.update(outcome=UPHELD if valid else CHECK_INVALID,
                      status=UPHELD if valid else CHECK_INVALID, reason=reason)
    elif ground_text == "harness_error":
        harness = any(_receipt_status(item) == "HARNESS_ERROR"
                      for item in _seq(receipts)
                      if str(_get(item, "check_id", "")) == ident)
        result.update(outcome=CHECK_REBOUND if harness else INTERPRETATION_OPEN,
                      status=CHECK_REBOUND if harness else INTERPRETATION_OPEN,
                      reason="runner failure requires a new check revision" if harness
                      else "harness error is not established")
    elif ground_text == "stale_evidence":
        stale = any(current_revision is not None
                    and str(_get(item, "repository_revision", _get(item, "snapshot_digest", "")))
                    != str(current_revision)
                    for item in _seq(receipts) if str(_get(item, "check_id", "")) == ident)
        result.update(outcome=CHECK_REBOUND if stale else INTERPRETATION_OPEN,
                      status=CHECK_REBOUND if stale else INTERPRETATION_OPEN,
                      reason="receipt is bound to an old revision" if stale
                      else "staleness is not established")
    elif ground_text == "conflicting_source":
        result.update(reason="source interpretations conflict; host abstains")
    return result


adjudicate_dispute = dispute


def apply_dispute(ledger: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Record a host dispute result without turning invalidation into satisfaction."""
    updated = copy.deepcopy(dict(ledger))
    rows = updated.setdefault("disputes", [])
    dispute_id = f"D{len(rows) + 1}"
    rows.append({
        "id": dispute_id,
        "target_check_ref": {"id": str(result.get("check_id", "")), "version": 1},
        "raised_at_revision": str(updated.get("current_revision", "unknown")),
        "ground": str(result.get("ground", "conflicting_source")),
        "claim": str(result.get("claim", "host dispute")) or "host dispute",
        "source_span_ids": [], "evidence_ids": [], "reproduction_command": [],
        "alternative_interpretation": None,
        "resolution": {
            "outcome": str(result.get("outcome", INTERPRETATION_OPEN)).lower(),
            "adjudicator": "host", "reason": str(result.get("reason", "")) or "host adjudication",
            "evidence_ids": [], "replacement_check_ref": None,
        },
    })
    # CHECK_INVALID/CHECK_REBOUND changes check lifecycle/admission only.  No
    # requirement assessment is set to satisfied here.
    if str(result.get("outcome", "")) in {CHECK_INVALID, CHECK_REBOUND}:
        for check in updated.get("checks", ()):
            if isinstance(check, Mapping) and str(check.get("id")) == str(result.get("check_id")):
                check["lifecycle"] = "retired"
                refs = {str(ref.get("id")) for ref in check.get("requirement_refs", ())
                        if isinstance(ref, Mapping)}
                for requirement in updated.get("requirements", ()):
                    if not isinstance(requirement, Mapping) or str(requirement.get("id")) not in refs:
                        continue
                    prior = requirement.get("assessment", {})
                    requirement["assessment"] = {
                        **(dict(prior) if isinstance(prior, Mapping) else {}),
                        "status": "open",
                        "reason": "check invalidated; host must recompute the requirement",
                        "basis": "none", "policy_ref": None, "evidence_ids": [],
                    }
    return updated


def _span_ids_for_check(check: object, ledger: Mapping[str, Any]) -> list[str]:
    requirement = _requirement_for_check(check, ledger)
    if requirement is not None and requirement.get("source_span_ids"):
        return [str(item) for item in requirement["source_span_ids"]]
    spans = _spans(ledger)
    clause = _provenance(check).get("clause_span")
    text = clause.get("text") if isinstance(clause, Mapping) else ""
    for ident, row in spans.items():
        if text and str(text) in str(row.get("quote", "")):
            return [ident]
    return [next(iter(spans), "S1")]


def _binding_ids_for_check(check: object, ledger: Mapping[str, Any]) -> list[str]:
    entry = _entry_point(check)
    tail = _tail(entry)
    rows = [str(row.get("id")) for row in _binding_rows(ledger)
            if str(row.get("name", "")).rsplit(".", 1)[-1] == tail]
    return rows or ["B-UNRESOLVED"]


def _predicate_for_check(check: object, span_ids: Sequence[str]) -> dict[str, Any]:
    template = str(_get(check, "template", "")).upper()
    expected = copy.deepcopy(_get(check, "expected", None))
    operator = "relation" if expected is None else "equals"
    return {"observable": "target_assertion", "operator": operator,
            "expected": expected if expected is not None else template,
            "support_span_ids": list(span_ids) or ["S1"]}


def attach_checks(ledger: Mapping[str, Any], checks: object, *,
                  current_revision: object = None) -> tuple[dict[str, Any], tuple[AdmissionDecision, ...]]:
    """Attach schema-shaped check/admission/certificate records to a ledger."""
    result = copy.deepcopy(dict(ledger))
    values = _seq(checks)
    decisions = admit_checks(values, result, current_revision=current_revision)
    by_id = {decision.check_id: decision for decision in decisions}
    schema_checks: list[dict[str, Any]] = []
    certs: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    req_map = {str(row.get("id")): row for row in result.get("requirements", ())
               if isinstance(row, Mapping)}
    for check in values:
        ident = str(_get(check, "check_id", _get(check, "id", "")))
        decision = by_id[ident]
        span_ids = _span_ids_for_check(check, result)
        refs = list(decision.requirement_refs)
        if not refs:
            requirement = _requirement_for_check(check, result)
            if requirement is not None:
                refs = [{"id": str(requirement.get("id")), "version": int(requirement.get("version", 1))}]
        # Never attach an unscoped green check to the first requirement.  A
        # suite-level pass is not clause evidence, and choosing the first row
        # made an unexercised public clause appear satisfied by unrelated work.
        binding_ids = _binding_ids_for_check(check, result)
        artifact = _get(check, "artifact_sha256", "")
        if not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact):
            artifact = hashlib.sha256(json.dumps(_mapping(check), sort_keys=True, default=str).encode()).hexdigest()
        fixture = copy.deepcopy(_get(check, "fixture", {}))
        expected = copy.deepcopy(_get(check, "expected", None))
        span_ids = span_ids or ["S1"]
        admission_payload = {
            "tier": decision.tier_lower, "reason": decision.reason,
            "certificate_ids": [str(row.get("id")) for row in decision.certificates],
            "authorization": copy.deepcopy(decision.authorization),
        }
        schema_checks.append({
            "id": ident, "version": 1, "requirement_refs": refs,
            "lifecycle": "active", "purpose": "change_detection",
            "source_span_ids": span_ids, "binding_ids": binding_ids,
            "fixture": {
                "input": fixture, "setup_evidence_ids": [f"E-{ident}-fixture"],
                "varied_dimensions": [], "validated_precondition_ids": [],
                "reset_strategy": "fresh process", "environment_ref": "host-runner",
            },
            "oracle": {
                "template_id": decision.template_id or str(_get(check, "template", "host")),
                "assertions": [_predicate_for_check(check, span_ids)],
                "scope": "one host check invocation", "unsupported_assertions": [],
            },
            "artifact_path": str(_get(check, "artifact_path", f"generated/{ident}")),
            "artifact_sha256": artifact, "command": ["host-check", ident],
            "generator": {"kind": "public_test" if _is_explicit_public_acceptance(check) else "deterministic_host",
                          "id": "design-55-admission", "version": "1",
                          "input_source_ids": [str(result.get("ticket_source_id", "ticket"))]},
            "admission": admission_payload,
            "base_receipt_ids": [], "current_receipt_ids": [], "supersedes": None,
        })
        for cert in decision.certificates:
            if not any(str(row.get("id")) == str(cert.get("id")) for row in certs):
                certs.append(copy.deepcopy(cert))
            evidence.append({
                "id": str(cert.get("evidence_ids", [f"E-{ident}"])[0]),
                "kind": "calibration" if cert.get("stage") == RUNNER else "inspection",
                "producer": "host", "revision": str(current_revision or result.get("current_revision", "unknown")),
                "locator": None, "span_ids": span_ids, "receipt_id": None,
                "claim": str(cert.get("limitations", ["admission certificate"])[0]),
                "scope": f"check {ident}", "artifact_ref": None,
            })
        alignment_id = ""
        if decision.alignment_evidence:
            alignment_id = f"E-{ident}-alignment"
            alignment = decision.alignment_evidence
            evidence.append({
                "id": alignment_id, "kind": "source_binding", "producer": "host",
                "revision": str(current_revision or result.get("current_revision", "unknown")),
                "locator": None, "span_ids": span_ids, "receipt_id": None,
                "claim": "ticket literal alignment: " + json.dumps(
                    alignment, sort_keys=True, ensure_ascii=False, default=str),
                "scope": str(alignment.get("sentence") or "ticket literal alignment"),
                "artifact_ref": str(alignment.get("target_file") or "") or None,
            })
        for ref in refs:
            requirement = req_map.get(str(ref.get("id")))
            if requirement is not None:
                existing = {str(row.get("id")): row for row in requirement.get("check_refs", ())}
                existing[ident] = {"id": ident, "version": 1}
                requirement["check_refs"] = list(existing.values())
                if alignment_id:
                    assessment = requirement.get("assessment")
                    if isinstance(assessment, Mapping):
                        assessment = copy.deepcopy(dict(assessment))
                        evidence_ids = list(assessment.get("evidence_ids", ()))
                        assessment["evidence_ids"] = list(dict.fromkeys(
                            [*evidence_ids, alignment_id]))
                        requirement["assessment"] = assessment
    result["checks"] = schema_checks
    result["certificates"] = certs
    result["evidence"] = evidence
    # The caller may have supplied receipt rows after this point; this function
    # only changes host registry state and leaves assessment status open.
    from kernel_requirements_ledger import validate_ledger
    validate_ledger(result)
    return result, decisions


def frame_projection(ledger: Mapping[str, Any], *, decisions: object = (), receipts: object = (),
                     actions: Sequence[Mapping[str, Any]] = (),
                     current_revision: object = None) -> dict[str, Any]:
    """Produce the compact model-facing ordering required by the requirements ledger."""
    decision_map = {str(_get(row, "check_id", "")): row for row in _seq(decisions)}
    schema_checks = {str(row.get("id")): row for row in ledger.get("checks", ())
                     if isinstance(row, Mapping)}
    spans = _spans(ledger)
    source = _ticket_source(ledger) or {}
    ticket = {
        "source_id": str(source.get("id", ledger.get("ticket_source_id", ""))),
        "revision": str(source.get("revision", "")),
        "text": str(source.get("text", "")),
    }
    oracle_literal_canary = None
    if os.environ.get("STATEFRAME_ORACLE_TICKET_LITERALS", "off").strip().lower() == "on":
        from kernel_oracle_literal_canary import (
            NOT_IN_TICKET, info_row, is_stale, stale_row,
        )
        oracle_literal_canary = (NOT_IN_TICKET, info_row, is_stale, stale_row)
    oracle_grounding_unknown = None
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        try:
            from kernel_oracle_grounding import UNBOUND_OBSERVABLE
            oracle_grounding_unknown = UNBOUND_OBSERVABLE
        except ImportError:
            oracle_grounding_unknown = None
    effective_revision = current_revision
    if effective_revision is None:
        effective_revision = ledger.get("current_revision")
    if effective_revision is None:
        effective_revision = source.get("revision")
    gating: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for receipt in _seq(receipts):
        check_id = str(_get(receipt, "check_id", ""))
        check = schema_checks.get(check_id, receipt)
        admission = decision_map.get(check_id)
        if admission is None and isinstance(check, Mapping):
            admission = check.get("admission")
        tier = str(_get(admission, "tier", "") or "").upper()
        status = _receipt_status(receipt)
        stale = bool(oracle_literal_canary and is_stale(receipt, effective_revision))
        literal_advisory = bool(
            oracle_literal_canary
            and str(_get(admission, "reason", "") or "") == NOT_IN_TICKET)
        span_ids = _get(check, "source_span_ids", ())
        quote = next((str(spans.get(str(ident), {}).get("quote", "")) for ident in span_ids
                      if str(ident) in spans), "")
        row = {
            "check_id": check_id, "tier": tier or "SHADOW", "quote": quote,
            "expected": copy.deepcopy(_get(receipt, "expected", _get(check, "expected", None))),
            "observed": copy.deepcopy(_get(receipt, "observed", _get(receipt, "observed_values", None))),
            "evidence": [str(_get(receipt, "evidence_ref", _get(receipt, "receipt_id", "")))],
            "actions": [
                {"tool": "inspect", "arguments": {"check_id": check_id}},
                {"tool": "rerun", "arguments": {"check_id": check_id}},
                {"tool": "dispute", "arguments": {"check_id": check_id}},
            ],
        }
        if (oracle_grounding_unknown
                and str(_get(admission, "reason", "") or "") == oracle_grounding_unknown):
            tier = ADVISORY
            row.update({
                "tier": tier,
                "verdict": "UNKNOWN",
                "reason": oracle_grounding_unknown,
                "required_edit": False,
                "actionable": False,
                "blocking": False,
            })
        if stale:
            row = stale_row(row, receipt, effective_revision)
        elif literal_advisory:
            row = info_row(row, message=f"{NOT_IN_TICKET}; cannot block completion")
        if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
            try:
                from kernel_requirement_alignment import admission_adjustment, enrich_counterexample
                alignment_check = copy.deepcopy(dict(check)) if isinstance(check, Mapping) else {}
                alignment_check.setdefault("expected", row["expected"])
                alignment_check.setdefault("entry_point", _get(admission, "symbol", ""))
                adjustment = admission_adjustment(
                    alignment_check, ledger, tier, str(_get(admission, "reason", "")))
                tier = str(adjustment.get("tier", tier)).upper()
                row["tier"] = tier
                row = enrich_counterexample(row, ledger=ledger, check=alignment_check)
            except (ImportError, TypeError, ValueError):
                pass
        if (oracle_grounding_unknown
                and str(row.get("reason", "")) == oracle_grounding_unknown):
            try:
                from kernel_oracle_grounding import cap_unbound_presentation
                row = cap_unbound_presentation(row)
            except ImportError:
                pass
        if literal_advisory:
            tier = ADVISORY
            row["tier"] = tier
            if not stale:
                row = info_row(row, message=f"{NOT_IN_TICKET}; cannot block completion")
        if tier == GATING and status == "COUNTEREXAMPLE" and not stale:
            gating.append(row)
        elif status == "COUNTEREXAMPLE" or stale:
            binding_ids = _get(check, "binding_ids", ())
            binding_id = str(binding_ids[0]) if _seq(binding_ids) else check_id
            symbol = str(_get(check, "binding",
                              _get(check, "entry_point",
                                   _get(admission, "symbol", ""))))
            if not symbol or symbol == check_id:
                symbol = next((str(item.get("name", binding_id))
                               for item in ledger.get("bindings", ())
                               if isinstance(item, Mapping)
                               and str(item.get("id")) == binding_id), binding_id)
            if stale:
                message = row["message"]
            elif literal_advisory:
                message = f"{NOT_IN_TICKET}; cannot block completion"
            elif (oracle_grounding_unknown
                  and str(row.get("reason", "")) == oracle_grounding_unknown):
                message = (f"{oracle_grounding_unknown}; cannot block completion; "
                           "do not treat this host expectation as a required edit")
            elif row.get("repairable"):
                message = ("repairable ticket counterexample; aligned clause is required "
                           "and the bound target remains open")
            else:
                message = f"cannot block completion; do not treat {symbol} as required"
            observation = {**row, "message": message, "blocking": False}
            if (oracle_grounding_unknown
                    and str(row.get("reason", "")) == oracle_grounding_unknown):
                observation.update({
                    "verdict": "UNKNOWN",
                    "required_edit": False,
                    "actionable": False,
                })
            observations.append(observation)
    open_rows: list[dict[str, Any]] = []
    satisfied: list[dict[str, Any]] = []
    try:
        from kernel_clause_discriminator import covered_requirement_ids
        covered_requirements = covered_requirement_ids(
            ledger, receipts,
            current_revision=effective_revision)
    except (ImportError, TypeError, ValueError):
        # Fail closed: inability to establish clause coverage keeps every
        # non-counterexample requirement open in this projection.
        covered_requirements = frozenset()
    for requirement in ledger.get("requirements", ()):
        if not isinstance(requirement, Mapping):
            continue
        assessment = requirement.get("assessment", {})
        status = str(assessment.get("status", "open")) if isinstance(assessment, Mapping) else "open"
        row = {
            "requirement_id": str(requirement.get("id", "")),
            "quote": str(requirement.get("exact_clause", "")),
            "scope": str(assessment.get("scope", "stated ticket span")) if isinstance(assessment, Mapping) else "stated ticket span",
        }
        requirement_id = str(requirement.get("id", ""))
        clause_covered = requirement_id in covered_requirements
        if status != "counterexample" and clause_covered:
            row["status"] = "SATISFIED under policy AP1: specified cases passed"
            satisfied.append(row)
        elif status != "counterexample":
            row["status"] = "open"
            row["reason"] = (
                "host assessment passed but no current clause-specific receipt"
                if status == "satisfied" and not clause_covered else
                str(assessment.get("reason", "host has not established satisfaction"))
                if isinstance(assessment, Mapping) else "host has not established satisfaction")
            open_rows.append(row)
    binding_names = {str(row.get("id")): str(row.get("name", row.get("id", "")))
                     for row in ledger.get("bindings", ()) if isinstance(row, Mapping)}
    for check_id, check in schema_checks.items():
        admission = decision_map.get(check_id) or check.get("admission", {})
        tier = str(_get(admission, "tier", "shadow")).upper()
        if tier == GATING:
            continue
        if not any(row.get("check_id") == check_id for row in observations):
            binding_id = str(check.get("binding_ids", (check_id,))[0]
                             if check.get("binding_ids") else check_id)
            symbol = binding_names.get(binding_id, binding_id)
            observation = {
                "check_id": check_id, "tier": tier,
                "message": f"cannot block completion; do not treat {symbol} as required",
                "blocking": False,
            }
            if (oracle_literal_canary
                    and str(_get(admission, "reason", "") or "") == NOT_IN_TICKET):
                observation = info_row(
                    observation, message=f"{NOT_IN_TICKET}; cannot block completion")
            if (oracle_grounding_unknown
                    and str(_get(admission, "reason", "") or "") == oracle_grounding_unknown):
                observation.update({
                    "verdict": "UNKNOWN",
                    "reason": oracle_grounding_unknown,
                    "required_edit": False,
                    "actionable": False,
                    "blocking": False,
                    "message": (f"{oracle_grounding_unknown}; cannot block completion; "
                                "do not treat this host expectation as a required edit"),
                })
            observations.append(observation)
    read_action = {"tool": "read_ticket", "arguments": {"source_id": ticket["source_id"]},
                   "description": "Return exact ticket spans and host-sliced quotes."}
    terminal_receipts = list(_seq(receipts))
    if oracle_literal_canary:
        terminal_receipts = [
            receipt for receipt in terminal_receipts
            if not is_stale(receipt, effective_revision)
        ]
    return {
        "schema": "requirements-ledger-frame/1", "ticket": ticket,
        "actionable_counterexamples": gating, "open": open_rows,
        "satisfied": satisfied[-8:], "observations": observations,
        "actions": [read_action, *[copy.deepcopy(dict(item)) for item in actions]],
        "terminal_status": completion_terminal_state(
            ({"receipts": terminal_receipts},), decisions),
        "policy": "AP1: only current target-attributed GATING failures can block completion",
    }


project_frame = frame_projection
project_requirements = frame_projection


__all__ = [
    "ADVISORY", "AdmissionDecision", "BINDING", "CHECK_INVALID", "CHECK_REBOUND",
    "FIXTURE", "GATING", "GROUNDS", "INTERPRETATION_OPEN", "ORACLE", "REJECTED",
    "RUNNER", "SHADOW", "SOURCE_ALIGNMENT", "TIERS", "UPHELD", "adjudicate_dispute",
    "admit_check", "admit_checks", "apply_dispute", "attach_checks", "completion_terminal_state",
    "dispute", "enabled", "frame_projection", "is_admitted_counterexample", "mode",
    "project_frame", "project_requirements", "reduce_assessments",
]
