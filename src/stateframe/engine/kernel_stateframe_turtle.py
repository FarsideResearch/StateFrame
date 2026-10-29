"""Rabbit/Turtle review at StateFrame commit points.

The rabbit owns the normal actor request.  This module is a small host-owned
review seam for a separate, same-model request.  Turtle input is deliberately
limited to contract/evidence material; rabbit reasoning is never copied into
the request.  Turtle output is advisory data until the host proves a witness
against the current candidate revision.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_TURTLE"
VALID_MODES = {"off", "on"}
VERDICTS = {"CLEAR_SCOPE", "DEFECT", "CHECK", "ABSTAIN"}
HOST_VERDICTS = VERDICTS | {"ADVISE", "CHECK_RESULT"}
READ_ONLY_TOOLS = ("read", "grep", "run_tests")
_GREEN = {"OK", "GREEN", "PASS", "PASSED", "SUCCESS", "APPLIED"}
_RED = {"RED", "FAIL", "FAILED", "ERROR", "FAILURE"}
_STRUCTURAL_KINDS = {
    "duplicate_binding", "live_stub", "missing_public_symbol", "structural",
    "structural_guard", "syntax_error", "protected_oracle_modified",
}
_QUANTITY_WORDS = {
    "per", "second", "seconds", "sec", "s", "minute", "minutes", "min",
    "hour", "hours", "ms", "millisecond", "milliseconds", "token", "tokens",
    "byte", "bytes", "request", "requests", "item", "items", "record",
    "records", "rate", "interval", "duration", "elapsed", "capacity", "quota",
    "percent", "percentage", "count", "counts", "size", "job", "jobs",
}
_COMPARISON_RE = re.compile(r"(?:<=|>=|==|!=|<|>)|\b(?:accept|reject|allow|deny|inclusive|exclusive|compare|comparison)\b", re.I)
_INTERFACE_RE = re.compile(r"\b(?:class|def|function|method|public|interface|api|field|attribute|parameter|return)\b", re.I)
_DIFF_LINE_RE = re.compile(r"^[+\-](?![+\-])", re.M)


def turtle_mode() -> str:
    """Return the strict feature switch; absent means the byte-preserving OFF arm."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return turtle_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def revision_identity(value: object = None, *, source: str | None = None) -> str:
    """Normalize a host revision without deriving authority from product prose."""
    if isinstance(value, Mapping):
        for key in ("candidate_revision", "product_revision", "current_revision",
                    "artifact_revision", "revision"):
            if value.get(key) is not None:
                return revision_identity(value[key])
    if value is not None:
        return str(value)
    if source is not None:
        return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    return "UNKNOWN"


def _as_rows(value: object) -> list[dict]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _result(row: Mapping[str, object]) -> str:
    return str(row.get("result_class", row.get("result", row.get("status", "")))).upper()


def _row_revision(row: Mapping[str, object]) -> str | None:
    for key in ("candidate_revision", "current_revision", "product_revision", "revision",
                "after_revision", "source_revision", "artifact_revision"):
        value = row.get(key)
        if value is not None:
            return str(value)
    binding = row.get("binding")
    if isinstance(binding, Mapping):
        return _row_revision(binding)
    return None


def current_receipts(receipts: object) -> list[dict]:
    """Copy receipt facts while excluding assistant/model rationale fields."""
    result = []
    for row in _as_rows(receipts):
        result.append({key: copy.deepcopy(value) for key, value in row.items()
                       if key not in {"reasoning", "reasoning_content", "rationale",
                                      "assistant_reasoning", "model_thought"}})
    return result


def brief_clauses(brief: str | None) -> list[str]:
    """Return exact clause text from the existing brief extractor when available."""
    text = brief if isinstance(brief, str) else ""
    try:
        from kernel_findings import extract_unit_clauses
        clauses = [str(item) for item in extract_unit_clauses(text) if str(item).strip()]
        if clauses:
            return clauses
    except (ImportError, AttributeError, TypeError):
        pass
    # The fallback is only a transport-preserving clause split.  It does not
    # infer values from source or tests.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text)
            if part.strip()]


def _obligation_rows(brief: str, obligations: object) -> list[dict]:
    rows = _as_rows(obligations)
    if rows:
        return rows
    return [{"id": f"C{index}", "clause": clause, "expected": clause}
            for index, clause in enumerate(brief_clauses(brief), 1)]


def _coverage(row: Mapping[str, object]) -> str:
    return str(row.get("coverage", row.get("evidence_status", row.get("status", "UNKNOWN")))).upper()


def stale_receipts(receipts: object, candidate_revision: object) -> list[dict]:
    revision = revision_identity(candidate_revision)
    stale = []
    for row in _as_rows(receipts):
        observed = _row_revision(row)
        if row.get("stale") is True or row.get("current") is False:
            stale.append(row)
        elif observed is not None and observed != revision:
            stale.append(row)
    return stale


def _covered(rows: Sequence[Mapping[str, object]], receipts: Sequence[Mapping[str, object]],
             candidate_revision: str) -> bool:
    if rows and all(_coverage(row) in {"COVERED", "PASS", "GREEN", "EVIDENCE_AVAILABLE"}
                    for row in rows):
        return True
    wanted = {str(row.get("id")) for row in rows}
    for row in receipts:
        covered = row.get("covered_obligations", row.get("obligations", row.get("covers", ())))
        if isinstance(covered, str):
            covered = [covered]
        if (_result(row) in _GREEN and _row_revision(row) == candidate_revision
                and wanted.intersection(str(item) for item in (covered or ()) or ())):
            return True
    return False


def _structural_failure(findings: object) -> bool:
    for finding in _as_rows(findings):
        kind = str(finding.get("kind", finding.get("class", ""))).lower()
        if kind in _STRUCTURAL_KINDS or str(finding.get("class", "")).upper() == "STRUCTURAL":
            return True
    for finding in findings if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)) else ():
        text = str(finding).lower()
        if any(word in text for word in ("duplicate binding", "live stub", "missing public symbol",
                                         "syntax error", "structural guard")):
            return True
    return False


def _changed_lines(candidate_diff: str) -> str:
    return "\n".join(line for line in str(candidate_diff or "").splitlines()
                     if _DIFF_LINE_RE.match(line))


def _tagged_clause(clause: str) -> bool:
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", clause)}
    return bool(words & _QUANTITY_WORDS or _COMPARISON_RE.search(clause)
                or _INTERFACE_RE.search(clause))


def _diff_touches_clause(candidate_diff: str, clause: str) -> bool:
    changed = _changed_lines(candidate_diff)
    if not changed or not _tagged_clause(clause):
        return False
    clause_words = {word.lower() for word in re.findall(r"[A-Za-z_]\w*", clause)}
    changed_words = {word.lower() for word in re.findall(r"[A-Za-z_]\w*", changed)}
    shared = clause_words & changed_words
    return bool(shared) or bool(_COMPARISON_RE.search(changed) and _COMPARISON_RE.search(clause)) 


def _red_after_green(receipts: Sequence[Mapping[str, object]], revision: str) -> bool:
    seen_green = False
    for row in receipts:
        if _row_revision(row) not in {None, revision}:
            continue
        result = _result(row)
        if result in _GREEN:
            seen_green = True
        elif seen_green and result in _RED:
            return True
    return False


def _repetition_without_gain(receipts: Sequence[Mapping[str, object]]) -> bool:
    if len(receipts) < 2:
        return False
    first, second = receipts[-2:]
    first_identity = (first.get("kind", first.get("tool")), first.get("target", first.get("path")),
                      _row_revision(first), _result(first))
    second_identity = (second.get("kind", second.get("tool")), second.get("target", second.get("path")),
                       _row_revision(second), _result(second))
    if first_identity != second_identity:
        return False
    return not any(row.get("information_gain") is True or row.get("new_evidence") is True
                   for row in (first, second))


def detect_triggers(*, event: str | None = None, candidate_diff: str = "",
                    brief: str = "", obligations: object = (), receipts: object = (),
                    candidate_revision: object = None, host_findings: object = (),
                    expectation: object = None, contradiction: object = None) -> list[dict]:
    """Return host-observable review triggers in deterministic priority order."""
    revision = revision_identity(candidate_revision)
    receipt_rows = current_receipts(receipts)
    obligation_rows = _obligation_rows(brief, obligations)
    triggers: list[dict] = []
    if _structural_failure(host_findings):
        return [{"kind": "structural_guard_failure", "turtle": False,
                 "question": "kernel structural guard failure"}]
    event_name = str(event or "").lower()
    uncovered = [row for row in obligation_rows
                 if _coverage(row) not in {"COVERED", "PASS", "GREEN", "EVIDENCE_AVAILABLE"}]
    if event_name in {"promotion", "candidate_promotion", "promote", "green"}:
        touched = [row for row in obligation_rows
                   if _diff_touches_clause(candidate_diff, str(row.get("clause", row.get("expected", ""))))]
        if touched and not _covered(touched, receipt_rows, revision):
            triggers.append({"kind": "risky_promotion", "turtle": True,
                             "obligation_ids": [str(row.get("id", "UNKNOWN")) for row in touched],
                             "question": "does the candidate satisfy the touched tagged clauses?"})
    if event_name in {"done", "before_done", "completion"}:
        stale = stale_receipts(receipt_rows, revision)
        if uncovered or stale:
            triggers.append({"kind": "done_uncovered_or_stale", "turtle": True,
                             "obligation_ids": [str(row.get("id", "UNKNOWN")) for row in uncovered],
                             "stale_receipt_ids": [str(row.get("receipt_id", row.get("event_id", "UNKNOWN")))
                                                    for row in stale],
                             "question": "which uncovered or stale obligation remains unresolved?"})
    if _red_after_green(receipt_rows, revision) or contradiction is not None or (
            expectation is not None and isinstance(expectation, Mapping)
            and expectation.get("contradicted") is True):
        triggers.append({"kind": "surprise_contradiction", "turtle": True,
                         "question": "what host-observable evidence discriminates the contradiction?"})
    if _repetition_without_gain(receipt_rows):
        triggers.append({"kind": "repetition_without_information_gain", "turtle": True,
                         "question": "what concrete probe would add information beyond the repeated result?"})
    if uncovered and event_name in {"done", "before_done"}:
        if not any(item["kind"] == "done_uncovered_or_stale" for item in triggers):
            triggers.append({"kind": "missing_component_coverage", "turtle": True,
                             "obligation_ids": [str(row.get("id", "UNKNOWN")) for row in uncovered],
                             "question": "which brief component lacks current evidence?"})
    return triggers


def review_key(*, candidate_fingerprint: object, contract_revision: object,
               environment_revision: object, obligation_scope: object,
               question: object) -> str:
    """Stable coalescing key for one candidate/contract/environment/question."""
    return digest({"candidate": candidate_fingerprint, "contract": contract_revision,
                   "environment": environment_revision, "obligation_scope": obligation_scope,
                   "question": question})


def candidate_diff(before: str | None, after: str | None, path: str = "candidate.py") -> str:
    """Host-observed diff used in the Turtle request."""
    return "\n".join(difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(),
        fromfile="a/" + path, tofile="b/" + path, lineterm=""))


def evidence_tuple(*, candidate_revision: object, evidence: object,
                   obligations: object, probe_id: object = None) -> dict:
    """Evidence tuple: revision, host evidence, obligations, and probe identity."""
    return {"candidate_revision": revision_identity(candidate_revision),
            "evidence": copy.deepcopy(evidence),
            "obligations": copy.deepcopy(obligations),
            "probe_id": None if probe_id is None else str(probe_id),
            "authority": "HOST_OBSERVED"}


def turtle_tools() -> list[dict]:
    """Read-only tool declarations; product writes are intentionally absent."""
    return [{"type": "function", "function": {"name": name,
             "description": "Host-executed read-only Turtle tool; product writes denied.",
             "parameters": {"type": "object", "additionalProperties": True}}}
            for name in READ_ONLY_TOOLS]


def append_turtle_sidecar(messages: Sequence[dict], context: object, *,
                          provider_request: bool = True, mode: str | None = None) -> list[dict]:
    """Append a host result only in the ON arm, preserving OFF payload bytes."""
    result = [copy.deepcopy(message) for message in messages]
    selected_mode = turtle_mode() if mode is None else mode
    if selected_mode == "on" and provider_request and context:
        result.append({"role": "user", "content": str(context)})
    return result


def build_turtle_request(*, model: str, obligation: object, candidate_diff: str,
                         brief_clauses: Sequence[str], receipts: object,
                         trigger: Mapping[str, object], candidate_revision: object,
                         environment_revision: object = "UNKNOWN") -> dict:
    """Build the separate request without rabbit reasoning or hidden context."""
    packet = {
        "obligation": copy.deepcopy(obligation),
        "candidate_diff": str(candidate_diff),
        "brief_clauses": [str(clause) for clause in brief_clauses],
        "receipts": current_receipts(receipts),
        "trigger": copy.deepcopy(dict(trigger)),
        "candidate_revision": revision_identity(candidate_revision),
        "environment_revision": str(environment_revision),
    }
    system = ("You are Turtle, a read-only evidence-seeking reviewer. Return exactly one "
              "JSON object with verdict CLEAR_SCOPE, DEFECT, CHECK, or ABSTAIN. "
              "A DEFECT needs an executable witness, exact clause, and candidate_revision. "
              "A CHECK needs a concrete probe built from the brief contract before candidate "
              "behavior; do not propose edits or completion decisions.")
    user = "TURTLE_PACKET\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True)
    return {"model": model, "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
            "tools": turtle_tools()}


def _strict_object(raw: object) -> dict | None:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text in VERDICTS - {"DEFECT", "CHECK"}:
        return {"verdict": text}
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _witness_executable(value: object) -> bool:
    if isinstance(value, Mapping):
        kind = str(value.get("kind", value.get("tool", value.get("type", "")))).lower()
        return kind in READ_ONLY_TOOLS or any(value.get(key) for key in ("command", "selector", "code", "steps"))
    if isinstance(value, str):
        return bool(re.match(r"^(?:read|grep|run_tests|test|probe)\b", value.strip()))
    return False


def parse_turtle_output(raw: object, *, candidate_revision: object = None) -> dict:
    """Strictly parse Turtle output; invalid packets become ABSTAIN."""
    packet = _strict_object(raw)
    if packet is None or set(packet) - {"verdict", "reason", "witness", "clause",
                                        "candidate_revision", "check", "probe", "basis"}:
        return {"verdict": "ABSTAIN", "reason": "malformed turtle output",
                "authority": "HOST_OBSERVED"}
    verdict = packet.get("verdict")
    if verdict not in VERDICTS:
        return {"verdict": "ABSTAIN", "reason": "malformed turtle verdict",
                "authority": "HOST_OBSERVED"}
    result = {key: copy.deepcopy(value) for key, value in packet.items()}
    result["verdict"] = verdict
    result["authority"] = "TURTLE_UNVERIFIED"
    if verdict == "DEFECT":
        if not (_witness_executable(packet.get("witness"))
                and isinstance(packet.get("clause"), str) and packet["clause"].strip()
                and isinstance(packet.get("candidate_revision"), str)
                and packet["candidate_revision"].strip()):
            result.update(verdict="ADVISE", reason="DEFECT lacks executable witness, clause, or candidate revision")
        elif candidate_revision is not None and packet["candidate_revision"] != revision_identity(candidate_revision):
            result.update(verdict="ADVISE", reason="DEFECT witness is bound to a stale candidate revision")
    elif verdict == "CHECK":
        check = packet.get("check", packet.get("probe"))
        if not isinstance(check, Mapping) or not any(check.get(key) for key in ("command", "selector", "code", "steps")):
            return {"verdict": "ABSTAIN", "reason": "CHECK lacks a concrete probe", "authority": "HOST_OBSERVED"}
        if check.get("built_from_contract") is not True and not packet.get("basis"):
            return {"verdict": "ABSTAIN", "reason": "CHECK lacks contract-first basis", "authority": "HOST_OBSERVED"}
        check = dict(check)
        for key in ("clause", "basis"):
            if packet.get(key) is not None:
                check.setdefault(key, packet[key])
        result["check"] = copy.deepcopy(check)
    return result


def execute_check(check: Mapping[str, object], executor: Callable[[Mapping[str, object]], object] | None) -> dict:
    """Run one host-supplied existing executor; never execute Turtle code directly."""
    if not callable(executor):
        return {"status": "ABSTAIN", "reason": "no existing host executor"}
    command = str(check.get("command", ""))
    if re.search(r"(?:>|>>|\brm\b|\bmv\b|\bcp\b|\btouch\b|\bedit\b|\bwrite\b|\bfile_create\b)",
                 command, re.I):
        return {"status": "ABSTAIN", "reason": "Turtle check is not read-only"}
    try:
        # Existing executors may use this directory for generated probe files.
        # The target workspace is never selected as the probe-file location.
        with tempfile.TemporaryDirectory(prefix="stateframe-turtle-probe-") as sandbox:
            safe_check = {**dict(check), "sandbox_dir": str(Path(sandbox).resolve())}
            outcome = executor(safe_check)
    except Exception as error:  # host faults are evidence, not Turtle verdicts
        return {"status": "UNKNOWN", "error_type": type(error).__name__, "reason": str(error)}
    if isinstance(outcome, Mapping):
        return copy.deepcopy(dict(outcome))
    return {"status": "HOST_OBSERVED", "observation": str(outcome)}


def resolve_check(check: Mapping[str, object], outcome: Mapping[str, object], *, candidate_revision: object) -> dict:
    """Mechanically resolve a CHECK when its existing executor gives a typed result."""
    status = str(outcome.get("verdict", outcome.get("status", outcome.get("result", "")))).upper()
    if status in {"CLEAR", "CLEAR_SCOPE", "GREEN", "PASS", "PASSED", "OK"}:
        return {"verdict": "CLEAR_SCOPE", "check": dict(check), "outcome": dict(outcome),
                "authority": "HOST_OBSERVED"}
    if status in {"DEFECT", "RED", "FAIL", "FAILED", "ERROR"}:
        witness = outcome.get("witness", check)
        clause = outcome.get("clause") or check.get("clause") or check.get("basis")
        return parse_turtle_output({"verdict": "DEFECT", "witness": witness,
                                    "clause": clause, "candidate_revision": revision_identity(candidate_revision)},
                                   candidate_revision=candidate_revision) | {"outcome": dict(outcome)}
    return {"verdict": "CHECK_RESULT", "check": dict(check), "outcome": dict(outcome),
            "authority": "HOST_OBSERVED"}


class TurtleReview:
    """Run-local coalescing and accounting for at most one episode per review key."""

    def __init__(self, *, mode: str | None = None, journal: object = None,
                 model: str = "", charge: Callable[[str], object] | None = None):
        self.mode = turtle_mode() if mode is None else mode
        if self.mode not in VALID_MODES:
            raise ValueError("invalid StateFrame Turtle mode")
        self.journal = journal
        self.model = model
        self.charge = charge
        self.episodes: dict[str, dict] = {}

    def _emit(self, kind: str, **fields: object) -> None:
        if self.journal is not None and hasattr(self.journal, "emit"):
            self.journal.emit(kind, **fields)

    def review(self, *, trigger: Mapping[str, object], obligation: object,
               candidate_diff: str, brief: str, receipts: object,
               candidate_revision: object, contract_revision: object,
               environment_revision: object, sender: Callable[[dict], object] | None = None,
               check_executor: Callable[[Mapping[str, object]], object] | None = None,
               obligation_scope: object = None) -> dict:
        if self.mode == "off":
            return {"verdict": "OFF", "authority": "HOST_OBSERVED"}
        if not trigger.get("turtle", True):
            result = {"verdict": "KERNEL_GUARD", "authority": "HOST_OBSERVED"}
            self._emit("turtle_trigger", trigger=dict(trigger), turtle_called=False)
            return result
        revision = revision_identity(candidate_revision)
        clauses = brief_clauses(brief)
        key = review_key(candidate_fingerprint=revision, contract_revision=contract_revision,
                         environment_revision=environment_revision,
                         obligation_scope=obligation_scope if obligation_scope is not None else obligation,
                         question=trigger.get("question", trigger.get("kind", "")))
        dependencies = digest({"candidate_revision": revision, "receipts": current_receipts(receipts),
                               "candidate_diff": candidate_diff})
        self._emit("turtle_trigger", review_key=key, trigger=dict(trigger),
                   candidate_revision=revision, dependencies=dependencies, turtle_called=False)
        prior = self.episodes.get(key)
        if prior is not None and prior.get("dependencies") == dependencies:
            result = {**copy.deepcopy(prior["result"]), "reused": True, "review_key": key}
            self._emit("turtle_verdict", review_key=key, verdict=result.get("verdict"), reused=True)
            return result
        if not callable(sender):
            result = {"verdict": "ABSTAIN", "reason": "separate Turtle sender unavailable",
                      "authority": "HOST_OBSERVED", "review_key": key}
            self.episodes[key] = {"dependencies": dependencies, "result": copy.deepcopy(result)}
            self._emit("turtle_verdict", review_key=key, verdict="ABSTAIN", reused=False)
            return result
        request = build_turtle_request(
            model=self.model, obligation=obligation, candidate_diff=candidate_diff,
            brief_clauses=clauses, receipts=receipts, trigger=trigger,
            candidate_revision=revision, environment_revision=environment_revision)
        self._emit("turtle_request", review_key=key, request_hash=digest(request),
                   candidate_revision=revision, trigger=trigger.get("kind"))
        if callable(self.charge):
            self.charge("TURTLE separate review request")
        try:
            sent = sender(request)
            response = sent[0] if isinstance(sent, tuple) and sent else sent
            sent_meta = sent[1] if isinstance(sent, tuple) and len(sent) > 1 \
                and isinstance(sent[1], Mapping) else {}
            accounting = sent_meta.get("accounting") if isinstance(sent_meta, Mapping) else None
            if isinstance(accounting, Mapping):
                self._emit("turtle_accounting", review_key=key,
                           request_id=sent_meta.get("request_id"),
                           response_event_id=sent_meta.get("response_event_id"),
                           accounting=copy.deepcopy(dict(accounting)),
                           counted_in_run_totals=True)
            if isinstance(response, Mapping) and isinstance(response.get("message"), Mapping):
                response = response["message"].get("content", "")
            parsed = parse_turtle_output(response, candidate_revision=revision)
        except Exception as error:  # malformed/transport failures never veto the rabbit
            parsed = {"verdict": "ABSTAIN", "reason": type(error).__name__ + ": " + str(error),
                      "authority": "HOST_OBSERVED"}
        if parsed.get("verdict") == "DEFECT":
            allowed_clauses = set(clauses)
            for row in _as_rows(obligation):
                clause = row.get("clause", row.get("expected"))
                if isinstance(clause, str):
                    allowed_clauses.add(clause)
            if parsed.get("clause") not in allowed_clauses:
                parsed.update(verdict="ADVISE", reason="DEFECT clause is not a verbatim brief obligation")
        if parsed.get("verdict") == "CHECK":
            outcome = execute_check(parsed["check"], check_executor)
            parsed = resolve_check(parsed["check"], outcome, candidate_revision=revision)
        parsed["review_key"] = key
        parsed["candidate_revision"] = revision
        self.episodes[key] = {"dependencies": dependencies, "result": copy.deepcopy(parsed)}
        self._emit("turtle_verdict", review_key=key, verdict=parsed.get("verdict"),
                   reused=False, authority=parsed.get("authority"),
                   evidence_tuple=evidence_tuple(candidate_revision=revision,
                                                  evidence=parsed.get("outcome", parsed),
                                                  obligations=obligation,
                                                  probe_id=parsed.get("probe_id")))
        return parsed


__all__ = [
    "HOST_VERDICTS", "MODE_ENV", "READ_ONLY_TOOLS", "TurtleReview", "VERDICTS",
    "append_turtle_sidecar", "brief_clauses", "build_turtle_request", "candidate_diff", "current_receipts",
    "detect_triggers", "digest", "enabled", "evidence_tuple", "execute_check",
    "parse_turtle_output", "resolve_check", "review_key", "revision_identity",
    "stale_receipts", "turtle_mode", "turtle_tools",
]
