"""Kernel-owned, revision-bound probes for uncovered brief clauses.

The gate is deliberately conservative.  It derives obligations only from the
public brief's existing unit-bearing clause extractor.  A public check is
marked COVERED only when a selected public-test oracle mechanically contains
the clause's quantity markers (or when the host supplies an explicit,
selector-bound coverage record).  Missing evidence is UNKNOWN, never
coverage.

Probe execution remains a worker action.  This module records the distinction
between the host's subprocess observation and the model's expectation; the
latter never changes coverage or completion authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_findings import extract_unit_clauses, lexical_token_count, normalize_test_oracle


MODE_ENV = "STATEFRAME_PROBE_GATE"
PROBE_RENDER_TOKEN_CAP = 120
PROBE_REFUSAL_TOKEN_CAP = 180
MAX_PROBE_CODE_CHARS = 1600
MAX_PROBE_OUTPUT_CHARS = 1200
_UNKNOWN = "UNKNOWN"
_COVERED = "COVERED"
_UNCOVERED = "UNCOVERED"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_PATTERN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?!\w)")

# This is only a coverage marker vocabulary.  Clause extraction itself is
# delegated to kernel_findings.extract_unit_clauses above.
_QUANTITY_MARKERS = {
    "per", "second", "seconds", "sec", "s", "minute", "minutes", "min",
    "hour", "hours", "millisecond", "milliseconds", "ms", "microsecond",
    "microseconds", "us", "token", "tokens", "byte", "bytes", "request",
    "requests", "item", "items", "record", "records", "rate", "interval",
    "duration", "elapsed", "refill", "capacity", "quota", "percent",
    "percentage", "count", "counts", "size", "job", "jobs",
}
class ProbeGateError(ValueError):
    """A probe-gate packet or action was not admissible."""


def probe_gate_mode() -> str:
    """Return the strictly validated feature mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return probe_gate_mode() == "on"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def revision_identity(value: object = None, *, source: object = None) -> str:
    """Normalize a host revision without exposing source content."""
    if isinstance(value, Mapping):
        for key in ("product_revision", "current_revision", "artifact_revision",
                    "revision", "artifact"):
            if value.get(key) is not None:
                return revision_identity(value[key])
        revisions = value.get("artifact_revisions")
        if isinstance(revisions, Mapping):
            pairs = [(str(key), str(item)) for key, item in sorted(revisions.items())]
            if pairs:
                return "revisions:" + hashlib.sha256(_canonical_json(pairs)).hexdigest()[:16]
    if value is not None:
        return str(value)
    if isinstance(source, str):
        return "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    return _UNKNOWN


def _short_revision(value: object) -> str:
    text = revision_identity(value)
    if text.startswith("sha256:"):
        return text[7:15]
    return text if len(text) <= 16 else text[:12]


def _as_list(value: object) -> list[object]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _selector_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("selector", "id", "path", "target"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _normalise_selectors(value: object) -> tuple[str, ...]:
    result = []
    items = list(value.keys()) if isinstance(value, Mapping) else _as_list(value)
    for item in items:
        selector = _selector_text(item)
        if selector and selector not in result:
            result.append(selector)
    return tuple(result)


def _selector_leaf(selector: str) -> str | None:
    if "::" not in selector:
        return None
    leaf = selector.rsplit("::", 1)[1].strip()
    return leaf or None


def _source_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        for key in ("source", "test_source", "test_src", "public_test_source",
                    "public_test_src", "content"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item
    return None


def _source_for_selector(source: object, selector: str) -> str | None:
    if isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        keys = [selector, selector.split("::", 1)[0], _selector_leaf(selector)]
        for key in keys:
            if key and key in source:
                value = _source_value(source[key])
                if value is not None:
                    return value
        for key in ("default", "source", "test_source", "test_src"):
            value = _source_value(source.get(key))
            if value is not None:
                return value
    values = _as_list(source)
    for value in values:
        text = _source_value(value)
        if text is not None:
            return text
    return None


def _selected_oracle(source: str, selector: str) -> list[str]:
    """Select a normalized oracle for one public selector, conservatively."""
    oracle = normalize_test_oracle(source)
    leaf = _selector_leaf(selector)
    if leaf is None:
        return oracle
    selected = [line for line in oracle if line.startswith(leaf + ":")]
    return selected


def _coverage_markers(clause: str) -> tuple[str, ...]:
    words = [word.lower() for word in _WORD_PATTERN.findall(clause)]
    numbers = [number for number in _NUMBER_PATTERN.findall(clause)]
    markers = numbers + [word for word in words if word in _QUANTITY_MARKERS]
    return tuple(dict.fromkeys(markers))


def _oracle_exercises_clause(oracle: Sequence[str], clause: str,
                             target_symbol: str | None = None) -> bool:
    markers = _coverage_markers(clause)
    if not markers:
        return False
    for line in oracle:
        text = str(line).lower()
        if target_symbol and not re.search(r"\b" + re.escape(target_symbol) + r"\b", text):
            continue
        if all(marker.lower() in text for marker in markers):
            return True
    return False


def _explicit_coverage(coverage: object, obligation_id: str,
                       selector: str) -> bool:
    """Accept only host-supplied selector-to-clause records."""
    if not isinstance(coverage, Mapping):
        return False
    direct = coverage.get(obligation_id)
    if direct is True:
        return True
    if direct is not None:
        direct_ids = {str(item) for item in _as_list(direct)}
        if selector in direct_ids:
            return True
    values = coverage.get(selector)
    if values is None:
        values = coverage.get("*")
    if isinstance(values, Mapping):
        values = values.get("covered", values.get("clauses", values.get("obligations")))
    ids = {str(item) for item in _as_list(values)}
    return obligation_id in ids


def _coverage_for_clause(clause: str, obligation_id: str, selectors: Sequence[str],
                         public_test_source: object, public_coverage: object,
                         target_symbol: str | None) -> tuple[str, list[str]]:
    if not selectors:
        return _UNKNOWN, []
    if obligation_id in {str(item) for item in _as_list(public_coverage)}:
        return _COVERED, list(selectors)
    covered_by = []
    source_seen = False
    selector_seen = False
    for selector in selectors:
        if _explicit_coverage(public_coverage, obligation_id, selector):
            covered_by.append(selector)
            continue
        source = _source_for_selector(public_test_source, selector)
        if source is None:
            continue
        source_seen = True
        oracle = _selected_oracle(source, selector)
        known_oracle = [line for line in oracle
                        if str(line) != _UNKNOWN and "(not normalized" not in str(line)]
        if _selector_leaf(selector) is None and known_oracle:
            selector_seen = True
        elif known_oracle:
            selector_seen = True
        if known_oracle and _oracle_exercises_clause(known_oracle, clause, target_symbol):
            covered_by.append(selector)
    if covered_by:
        return _COVERED, covered_by
    if source_seen and selector_seen:
        return _UNCOVERED, []
    return _UNKNOWN, []


def _obligation(index: int, clause: str, coverage: str,
                covered_by: Sequence[str]) -> dict:
    # `expected` is intentionally the exact public brief clause.  It is never
    # populated from product source, a test implementation, or probe code.
    return {
        "id": f"C{index}",
        "clause": clause,
        "expected": clause,
        "expected_authority": "BRIEF",
        "coverage": coverage,
        "covered_by": list(covered_by),
        "probe": None,
    }


def compile_obligations(
    brief: str | None,
    public_selectors: object = (),
    public_test_source: object = None,
    *,
    public_coverage: object = None,
    target_symbol: str | None = None,
    public_checks: object = None,
    source: object = None,
    public_api: object = None,
    public_contract: object = None,
    ledger: object = None,
    public_sources: object = None,
) -> list[dict]:
    """Derive brief obligations using only public brief/check evidence.

    ``source`` is accepted for callers that already pass a target source, but
    is deliberately ignored: product source cannot supply an expected value or
    make a clause covered.  Public-test source belongs in ``public_test_source``.
    """
    del source
    if public_checks is not None and not public_selectors:
        public_selectors = public_checks
    clauses = extract_unit_clauses(brief)
    selectors = _normalise_selectors(public_selectors)
    result = []
    for index, clause in enumerate(clauses, 1):
        coverage, covered_by = _coverage_for_clause(
            clause, f"C{index}", selectors, public_test_source,
            public_coverage, target_symbol)
        result.append(_obligation(index, clause, coverage, covered_by))
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        from kernel_oracle_grounding import ground_probe_obligations
        source_material = public_sources
        if source_material is None:
            source_material = public_test_source
        elif public_test_source is not None:
            source_material = (source_material, public_test_source)
        return ground_probe_obligations(
            result, ticket=brief, ledger=ledger, public_api=public_api,
            public_contract=public_contract, sources=source_material)
    return result


derive_obligations = compile_obligations


def _frame_value(frame: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        if name in frame:
            return frame[name]
    for container_name in ("kernel_findings", "findings_input", "task_row", "row"):
        container = frame.get(container_name)
        if isinstance(container, Mapping):
            for name in names:
                if name in container:
                    return container[name]
    return None


def _grounding_context_from_frame(frame: Mapping[str, object], brief: object = None) -> dict[str, object]:
    """Collect the public binding material carried by a serialized frame."""
    contract = frame.get("completion_contract")
    current = (contract.get("current_public_checks")
               if isinstance(contract, Mapping) else None)
    ticket = brief if isinstance(brief, str) else _frame_value(
        frame, ("task_brief", "brief", "msg", "ticket", "ticket_text"))
    public_api = _frame_value(frame, ("public_api", "repo_public_api", "api"))
    public_contract = _frame_value(
        frame, ("public_contract", "repo_contract", "contract"))
    if public_contract is None and isinstance(contract, Mapping):
        public_contract = contract
    public_sources = _frame_value(
        frame, ("public_sources", "sources", "public_material"))
    ledger = _frame_value(
        frame, ("requirements_ledger", "coverage_dispositions"))
    public_test_source = _frame_value(
        frame, ("public_test_source", "public_test_src", "test_source",
                "test_src", "public_tests", "public_test"))
    if public_test_source is None and isinstance(current, Mapping):
        public_test_source = current.get(
            "public_test_source", current.get("public_test_src", current.get(
                "test_source", current.get("test_src"))))
    if public_sources is None:
        public_sources = public_test_source
    elif public_test_source is not None:
        public_sources = (public_sources, public_test_source)
    return {
        "ticket": ticket if isinstance(ticket, str) else "",
        "ledger": ledger,
        "public_api": public_api,
        "public_contract": public_contract,
        "sources": public_sources,
    }


def obligations_from_frame(frame: Mapping[str, object], *, brief: str | None = None,
                           target_symbol: str | None = None) -> list[dict]:
    """Compile obligations from the same public fields used by findings."""
    if not isinstance(frame, Mapping):
        return []
    existing = frame.get("probe_obligations")
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        result = copy.deepcopy(list(existing))
        if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
            from kernel_oracle_grounding import ground_probe_obligations
            return ground_probe_obligations(result, **_grounding_context_from_frame(
                frame, brief=brief or _frame_value(frame, ("task_brief", "brief", "msg"))))
        return result
    contract = frame.get("completion_contract")
    current = contract.get("current_public_checks") if isinstance(contract, Mapping) else None
    selectors = _frame_value(frame, ("public_check_selectors", "f2p_ids",
                                     "required_public_checks"))
    if selectors is None and isinstance(contract, Mapping):
        selectors = contract.get("required_public_checks")
    if selectors is None and isinstance(current, Mapping):
        selectors = current.get("selectors", current.get("selector"))
    source = _frame_value(frame, ("public_test_source", "public_test_src", "test_source",
                                  "test_src", "public_tests", "public_test"))
    if source is None and isinstance(current, Mapping):
        source = current.get("public_test_source", current.get(
            "public_test_src", current.get("test_source", current.get("test_src"))))
    coverage = _frame_value(frame, ("public_check_coverage", "covered_clauses",
                                    "clause_coverage", "covered_obligations",
                                    "exercised_clauses"))
    if coverage is None and isinstance(current, Mapping):
        coverage = current.get("public_check_coverage", current.get(
            "covered_clauses", current.get("clause_coverage",
                                            current.get("covered_obligations"))))
    if brief is None:
        raw_brief = _frame_value(frame, ("task_brief", "brief", "msg"))
        brief = raw_brief if isinstance(raw_brief, str) else ""
    context = _grounding_context_from_frame(frame, brief=brief)
    return compile_obligations(
        brief, selectors, source, public_coverage=coverage,
        target_symbol=target_symbol, public_api=context["public_api"],
        public_contract=context["public_contract"], ledger=context["ledger"],
        public_sources=context["sources"])


@dataclass(frozen=True)
class DoneDecision:
    accepted: bool
    rejected: bool
    revision: str
    obligations: tuple[dict, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "revision": self.revision,
            "obligations": copy.deepcopy(list(self.obligations)),
            "reason": self.reason,
        }


class ProbeGate:
    """Small state machine implementing one soft challenge per revision."""

    def __init__(self, brief: str | None = "", *, public_selectors: object = (),
                 public_test_source: object = None, public_coverage: object = None,
                 target_symbol: str | None = None, public_checks: object = None,
                 source: object = None, mode: str | None = None,
                 revision: object = None, public_api: object = None,
                 public_contract: object = None, ledger: object = None,
                 public_sources: object = None):
        self.mode = probe_gate_mode() if mode is None else mode
        if self.mode not in {"off", "on"}:
            raise ValueError(f"{MODE_ENV} must be off or on")
        self.brief = brief if isinstance(brief, str) else ""
        if public_checks is not None and not public_selectors:
            public_selectors = public_checks
        self.public_selectors = _normalise_selectors(public_selectors)
        self.public_test_source = public_test_source
        self.public_coverage = public_coverage
        self.target_symbol = target_symbol
        self.public_api = public_api
        self.public_contract = public_contract
        self.ledger = ledger
        self.public_sources = public_sources
        self._revision = revision_identity(revision)
        self._probes: dict[tuple[str, str], dict] = {}
        self._challenged_revisions: set[str] = set()
        self.obligations = compile_obligations(
            self.brief, self.public_selectors, self.public_test_source,
            public_coverage=self.public_coverage, target_symbol=self.target_symbol,
            source=source, public_api=self.public_api,
            public_contract=self.public_contract, ledger=self.ledger,
            public_sources=self.public_sources)

    @classmethod
    def from_frame(cls, frame: Mapping[str, object], *, brief: str | None = None,
                   target_symbol: str | None = None, mode: str | None = None):
        if brief is None:
            candidate = _frame_value(frame, ("task_brief", "brief", "msg"))
            brief = candidate if isinstance(candidate, str) else ""
        contract = frame.get("completion_contract")
        current = (contract.get("current_public_checks")
                   if isinstance(contract, Mapping) else None)
        selectors = _frame_value(frame, (
            "public_check_selectors", "f2p_ids", "required_public_checks"))
        if selectors is None and isinstance(current, Mapping):
            selectors = current.get("selectors", current.get("selector"))
        source = _frame_value(frame, (
            "public_test_source", "public_test_src", "test_source", "test_src"))
        if source is None and isinstance(current, Mapping):
            source = current.get("public_test_source", current.get("test_source"))
        coverage = _frame_value(frame, (
            "public_check_coverage", "covered_clauses", "clause_coverage",
            "covered_obligations"))
        if coverage is None and isinstance(current, Mapping):
            coverage = current.get("covered_clauses", current.get("clause_coverage"))
        context = _grounding_context_from_frame(frame, brief=brief)
        instance = cls(
            brief, public_selectors=selectors, public_test_source=source,
            public_coverage=coverage, target_symbol=target_symbol, mode=mode,
            revision=revision_identity(frame), public_api=context["public_api"],
            public_contract=context["public_contract"], ledger=context["ledger"],
            public_sources=context["sources"])
        if isinstance(frame.get("probe_obligations"), Sequence) and not isinstance(
                frame.get("probe_obligations"), (str, bytes)):
            instance.obligations = obligations_from_frame(frame, brief=brief,
                                                           target_symbol=target_symbol)
        return instance

    def configure(self, *, brief: str | None = None, public_selectors: object = None,
                  public_test_source: object = None, public_coverage: object = None,
                  target_symbol: str | None = None, public_api: object = None,
                  public_contract: object = None, ledger: object = None,
                  public_sources: object = None) -> None:
        """Refresh public context while retaining revision-bound probe history."""
        if brief is not None:
            self.brief = brief if isinstance(brief, str) else ""
        if public_selectors is not None:
            self.public_selectors = _normalise_selectors(public_selectors)
        if public_test_source is not None:
            self.public_test_source = public_test_source
        if public_coverage is not None:
            self.public_coverage = public_coverage
        if target_symbol is not None:
            self.target_symbol = target_symbol
        if public_api is not None:
            self.public_api = public_api
        if public_contract is not None:
            self.public_contract = public_contract
        if ledger is not None:
            self.ledger = ledger
        if public_sources is not None:
            self.public_sources = public_sources
        self.obligations = compile_obligations(
            self.brief, self.public_selectors, self.public_test_source,
            public_coverage=self.public_coverage, target_symbol=self.target_symbol,
            public_api=self.public_api, public_contract=self.public_contract,
            ledger=self.ledger, public_sources=self.public_sources)

    def set_revision(self, revision: object = None, *, source: object = None) -> str:
        self._revision = revision_identity(revision, source=source)
        return self._revision

    @property
    def revision(self) -> str:
        return self._revision

    def _current_probe(self, obligation_id: str) -> dict | None:
        value = self._probes.get((obligation_id, self._revision))
        return copy.deepcopy(value) if value is not None else None

    def open_obligations(self, revision: object = None) -> list[dict]:
        current = revision_identity(revision) if revision is not None else self._revision
        result = []
        for item in self.obligations:
            if (str(item.get("reason", "")) == "UNBOUND_OBSERVABLE"
                    and self.mode == "on"
                    and os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on"):
                continue
            if item.get("coverage") == _COVERED:
                continue
            if self._probes.get((str(item.get("id")), current)) is not None:
                continue
            result.append(copy.deepcopy(item))
        return result

    def frame_view(self) -> dict:
        """Return bounded, source-free state for a working view."""
        if self.mode != "on":
            return {}
        obligations = []
        for item in self.obligations:
            keys = ["id", "clause", "expected", "expected_authority", "coverage", "covered_by"]
            if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
                keys.extend((
                    "status", "verdict", "reason", "required_edit", "actionable",
                    "blocking", "oracle_original_expected",
                    "_oracle_grounding_original_fields", "_oracle_grounding_added_fields",
                ))
            row = {key: copy.deepcopy(item[key]) for key in keys
                   if key in item}
            probe = self._current_probe(str(item.get("id")))
            if probe is not None:
                row["probe"] = {
                    "status": "PROBED",
                    "revision": self._revision,
                    "authority": "HOST_OBSERVED",
                    "expectation_authority": "MODEL_INTERPRETATION",
                }
            else:
                row["probe"] = None
            obligations.append(row)
        frame = {"probe_revision": self._revision,
                 "probe_obligations": obligations,
                 "probe_gate": MODE_ENV + "=on" if self.mode == "on" else MODE_ENV + "=off"}
        if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
            frame["task_brief"] = self.brief
            for key, value in (
                    ("public_api", self.public_api),
                    ("public_contract", self.public_contract),
                    ("public_sources", self.public_sources),
                    ("public_test_source", self.public_test_source)):
                if value is not None:
                    frame[key] = copy.deepcopy(value)
        return frame

    def record_probe(self, obligation: str, host_observed: object,
                     model_interpretation: object = None, *, revision: object = None) -> dict:
        """Record a successful host probe without promoting its interpretation."""
        obligation_id = str(obligation).strip()
        current = revision_identity(revision) if revision is not None else self._revision
        if self._revision == _UNKNOWN and revision is not None:
            self._revision = current
        if current != self._revision:
            return {"accepted": False, "status": "STALE", "reason": "probe revision is stale"}
        if not any(str(item.get("id")) == obligation_id for item in self.obligations):
            return {"accepted": False, "status": "REJECTED", "reason": "unknown obligation"}
        if self.mode != "on":
            return {"accepted": False, "status": "OFF", "reason": "probe gate is off"}
        observed = str(host_observed)
        if len(observed) > MAX_PROBE_OUTPUT_CHARS:
            observed = observed[:MAX_PROBE_OUTPUT_CHARS - 20] + "...[truncated]"
        interpretation = ("" if model_interpretation is None
                          else str(model_interpretation)[:320])
        brief_expected = next(
            (item.get("expected", item.get("clause", _UNKNOWN))
             for item in self.obligations if str(item.get("id")) == obligation_id),
            _UNKNOWN)
        record = {
            "obligation": obligation_id,
            "revision": self._revision,
            "result": observed,
            "authority": "HOST_OBSERVED",
            "expected": brief_expected,
            "expected_authority": "BRIEF",
            "expectation": interpretation,
            "expectation_authority": "MODEL_INTERPRETATION",
        }
        if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
            from kernel_oracle_grounding import ground_probe_expectation
            record = ground_probe_expectation(
                record, ticket=self.brief, ledger=self.ledger,
                public_api=self.public_api, public_contract=self.public_contract,
                sources=self.public_sources or self.public_test_source)
        self._probes[(obligation_id, self._revision)] = record
        return {"accepted": True, "status": "PROBED",
                "record": copy.deepcopy(record)}

    def evaluate_done(self, revision: object = None) -> DoneDecision:
        """Apply the one-challenge-per-current-revision soft gate."""
        current = revision_identity(revision) if revision is not None else self._revision
        if current != self._revision:
            self._revision = current
        if self.mode != "on":
            return DoneDecision(True, False, current)
        open_items = self.open_obligations(current)
        if not open_items:
            return DoneDecision(True, False, current)
        if current in self._challenged_revisions:
            return DoneDecision(True, False, current, tuple(open_items))
        self._challenged_revisions.add(current)
        return DoneDecision(False, True, current, tuple(open_items),
                            render_gate_refusal(open_items, current))

    def done(self, revision: object = None) -> tuple[bool, str]:
        decision = self.evaluate_done(revision)
        return decision.accepted, decision.reason


def _input_obligations(value: object) -> tuple[list[dict], str]:
    if isinstance(value, ProbeGate):
        return copy.deepcopy(value.obligations), value.revision
    if isinstance(value, Mapping):
        raw = value.get("probe_obligations", value.get("obligations", []))
        revision = revision_identity(value.get("probe_revision", value.get("revision")))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [dict(item) for item in raw if isinstance(item, Mapping)], revision
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)], _UNKNOWN
    return [], _UNKNOWN


def _line_within_cap(line: str, cap: int) -> str:
    if lexical_token_count(line) <= cap:
        return line
    tokens = _TOKEN_PATTERN.findall(line)
    if cap <= 0:
        return ""
    return " ".join(tokens[:cap])


def render_probe_obligations(value: object, cap: int = PROBE_RENDER_TOKEN_CAP) -> list[str]:
    """Render source-free obligation facts under a hard lexical-token cap."""
    cap = max(0, int(cap))
    if cap == 0:
        return []
    obligations, revision = _input_obligations(value)
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        obligations = [item for item in obligations
                       if str(item.get("reason", "")) != "UNBOUND_OBSERVABLE"]
    if not obligations:
        return []
    heading = "PROBE OBLIGATIONS (brief-derived; expected authority=BRIEF; revision=" \
              + _short_revision(revision) + "):"
    lines = [heading]
    for item in obligations:
        identifier = str(item.get("id", "UNKNOWN"))
        coverage = str(item.get("coverage", _UNKNOWN))
        probe = item.get("probe")
        status = ("PROBED" if isinstance(probe, Mapping)
                  and revision_identity(probe.get("revision")) == revision else "UNPROBED")
        clause = str(item.get("clause", item.get("expected", _UNKNOWN)))
        prefix = f"{identifier} {coverage}; {status}; expected from BRIEF: "
        available = (cap - lexical_token_count("\n".join(lines))
                     - lexical_token_count(prefix) - 1)
        if available <= 0:
            break
        if lexical_token_count(prefix + clause) <= available:
            clause_text = clause
        else:
            words = _TOKEN_PATTERN.findall(clause)
            clause_text = " ".join(words[:available])
            if len(words) > len(words[:available]):
                clause_text += " ..."
        line = _line_within_cap(prefix + clause_text, cap)
        candidate = "\n".join(lines + [line])
        if lexical_token_count(candidate) > cap:
            break
        lines.append(line)
    if len(lines) == 1:
        return [_line_within_cap(heading, cap)]
    return lines


def render_gate_refusal(obligations: Sequence[Mapping[str, object]],
                        revision: object = None) -> str:
    """Render the one-shot completion challenge in at most six lines."""
    rows = [item for item in obligations if isinstance(item, Mapping)]
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        rows = [item for item in rows
                if str(item.get("reason", "")) != "UNBOUND_OBSERVABLE"]
    row_count = len(rows)
    if len(rows) > 3:
        rows = rows[:3]
        omitted = row_count - len(rows)
    else:
        omitted = 0
    lines = ["[probe-gate] DONE REFUSED once for revision "
             + _short_revision(revision) + ": unprobed brief obligations:"]
    for item in rows:
        line = (f"{item.get('id', 'UNKNOWN')} {item.get('coverage', _UNKNOWN)}: "
                f"{item.get('clause', item.get('expected', _UNKNOWN))}")
        lines.append(_line_within_cap(line, PROBE_REFUSAL_TOKEN_CAP))
    if omitted:
        lines.append(f"... {omitted} more obligation(s) omitted")
    lines.append("Next done is accepted; a probe is valid only on this revision.")
    # Preserve the line-count contract even if a future extractor grows.
    return "\n".join(lines[:6])


def render_probe_findings(frame: Mapping[str, object],
                          cap: int = PROBE_RENDER_TOKEN_CAP) -> list[str]:
    """Working-view adapter; it never reads target implementation source."""
    if isinstance(frame, Mapping) and not frame.get("probe_obligations"):
        obligations = obligations_from_frame(frame)
        if obligations:
            value = {"probe_obligations": obligations,
                     "probe_revision": revision_identity(frame)}
            return render_probe_obligations(value, cap=cap)
    return render_probe_obligations(frame, cap=cap)


__all__ = [
    "DoneDecision", "MAX_PROBE_CODE_CHARS", "MAX_PROBE_OUTPUT_CHARS", "MODE_ENV",
    "PROBE_REFUSAL_TOKEN_CAP", "PROBE_RENDER_TOKEN_CAP", "ProbeGate", "ProbeGateError",
    "compile_obligations", "derive_obligations", "enabled", "obligations_from_frame",
    "probe_gate_mode", "revision_identity", "render_gate_refusal",
    "render_probe_findings", "render_probe_obligations",
]
