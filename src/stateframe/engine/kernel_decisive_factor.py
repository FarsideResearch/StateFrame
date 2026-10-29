"""Advisory decisive-factor and hidden-disqualifier records for r44.

The organ is deliberately small and host-owned.  A model can record one
approach, one factor, and one ticket-implied disqualifier; the host renders that
record back and re-checks it at the done boundary.  Neither operation has
completion authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_DECISIVE_FACTOR"
SCHEMA = "stateframe-decisive-factor/1"
PROMPT_HEADER = "[Decisive factor]"
MAX_APPROACH = 180
MAX_FACTOR = 180
MAX_DISQUALIFIER = 180
MAX_RECORDS = 8
_NONE_VALUES = frozenset({
    "none", "none identified", "no disqualifier", "no disqualifier identified",
    "not applicable", "n/a", "na",
})
_DISQUALIFIER_WORDS = re.compile(
    r"\b(?:must|shall|required|preserve|without|invalid|at\s+most|at\s+least|"
    r"exactly|cannot|can't|keep|compatible|compatibility|backward|unless|reject)\b",
    re.IGNORECASE,
)


def mode() -> str:
    value = (os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def ticket_for_owner(owner: object) -> str:
    row = getattr(owner, "task_row", {})
    row = row if isinstance(row, Mapping) else {}
    for key in ("_bridge_ordinary_brief", "msg", "brief", "symptom"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:1200]
    return ""


def _compact(value: object, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError("must not be empty")
    if len(text) > limit:
        raise ValueError(f"must be at most {limit} characters")
    return text


def _binding_digest(binding: object) -> str | None:
    if not isinstance(binding, Mapping):
        return None
    try:
        payload = json.dumps(dict(binding), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ticket_implies_disqualifier(ticket: str) -> bool:
    return bool(_DISQUALIFIER_WORDS.search(ticket or ""))


def normalize_decision(arguments: object) -> dict[str, str]:
    """Validate one compact model decision without granting it authority."""
    if not isinstance(arguments, Mapping):
        raise ValueError("decide arguments must be an object")
    values = {}
    for key, limit in (("approach", MAX_APPROACH),
                       ("decisive_factor", MAX_FACTOR),
                       ("disqualifier", MAX_DISQUALIFIER)):
        if key not in arguments:
            raise ValueError(f"decide requires {key}")
        values[key] = _compact(arguments[key], limit)
    # `why` is a common optional envelope field on the runner's other tools.
    # It is deliberately not part of the durable record or rendered state.
    if "why" in arguments and arguments["why"] is not None:
        _compact(arguments["why"], 220)
    return values


DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "decide",
        "description": (
            "Before committing to an approach, name one metric/property that governs "
            "the choice and the ticket's hidden disqualifier. Advisory only; this "
            "never blocks completion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "approach": {"type": "string", "maxLength": MAX_APPROACH},
                "decisive_factor": {"type": "string", "maxLength": MAX_FACTOR},
                "disqualifier": {"type": "string", "maxLength": MAX_DISQUALIFIER},
                "why": {"type": "string", "maxLength": 220},
            },
            "required": ["approach", "decisive_factor", "disqualifier"],
            "additionalProperties": False,
        },
    },
}
# A descriptive alias makes the host/tool contract easy to discover in probes.
DECISION_TOOL = DECIDE_TOOL


def tool_surface(tools: object) -> object:
    """Add exactly one typed decide tool only in the opt-in arm."""
    if not enabled() or tools is None:
        return tools
    if isinstance(tools, list):
        if not any(isinstance(item, Mapping)
                   and isinstance(item.get("function"), Mapping)
                   and item["function"].get("name") == "decide"
                   for item in tools):
            tools.append(copy.deepcopy(DECIDE_TOOL))
        return tools
    if isinstance(tools, tuple):
        return [*tools, copy.deepcopy(DECIDE_TOOL)]
    return tools


def evidence_rows(test_receipts: object) -> list[dict[str, object]]:
    """Project host test receipts to the public fields the re-check reads."""
    rows = []
    if not isinstance(test_receipts, (list, tuple)):
        return rows
    for row in test_receipts:
        if not isinstance(row, Mapping):
            continue
        exit_code = row.get("exit_code")
        if type(exit_code) is not int:
            continue
        try:
            step = int(row.get("step", 0) or 0)
        except (TypeError, ValueError):
            continue
        after = row.get("after_binding", row.get("binding"))
        rows.append({"step": step, "exit_code": exit_code,
                     "binding_digest": _binding_digest(after)})
    return rows


def _latest_evidence_after(rows: object, decision_step: int) -> dict[str, object] | None:
    if not isinstance(rows, (list, tuple)):
        return None
    latest = None
    for row in rows:
        if (isinstance(row, Mapping) and type(row.get("exit_code")) is int
                and int(row.get("step", 0) or 0) >= decision_step):
            latest = row
    return dict(latest) if latest is not None else None


class DecisiveFactorStore:
    """Bounded host record with an advisory done-time re-check."""

    def __init__(self, ticket: str = "") -> None:
        self.ticket = " ".join(str(ticket or "").split())[:1200]
        self.records: list[dict[str, object]] = []
        self._sequence = 0
        self.last_recheck: dict[str, object] | None = None

    @property
    def latest(self) -> dict[str, object] | None:
        return copy.deepcopy(self.records[-1]) if self.records else None

    def restore(self, events: object) -> None:
        if not isinstance(events, (list, tuple)):
            return
        for event in events:
            if not isinstance(event, Mapping) or event.get("kind") != "decisive_factor_recorded":
                continue
            record = event.get("record")
            if not isinstance(record, Mapping):
                continue
            try:
                normalized = normalize_decision(record)
            except ValueError:
                continue
            sequence = event.get("decision_sequence", len(self.records) + 1)
            try:
                sequence = max(1, int(sequence))
            except (TypeError, ValueError):
                sequence = len(self.records) + 1
            restored = {
                "id": str(event.get("decision_id") or f"decision-{sequence}"),
                **normalized,
                "step": int(event.get("step", 0) or 0),
                "binding_digest": event.get("binding_digest"),
            }
            self.records.append(restored)
            self._sequence = max(self._sequence, sequence)
        self.records = self.records[-MAX_RECORDS:]

    def record(self, arguments: object, *, step: int = 0,
               binding: object = None) -> dict[str, object]:
        try:
            normalized = normalize_decision(arguments)
        except ValueError as error:
            return {
                "status": "ADVISORY",
                "recorded": False,
                "reason": str(error),
                "rendered": f"{PROMPT_HEADER} advisory: {error}; completion not blocked.",
            }
        self._sequence += 1
        record = {
            "id": f"decision-{self._sequence}",
            **normalized,
            "step": int(step or 0),
            "binding_digest": _binding_digest(binding),
        }
        self.records.append(record)
        self.records = self.records[-MAX_RECORDS:]
        return {
            "status": "RECORDED",
            "recorded": True,
            "record": copy.deepcopy(record),
            "rendered": render_record(record),
        }

    def recheck(self, *, step: int = 0, binding: object = None,
                test_evidence: object = ()) -> dict[str, object]:
        """Re-assess the latest decision against current public host evidence.

        The host cannot judge a free-text factor semantically.  It CAN say
        whether the work the decision governed has been exercised by a public
        test run at the current workspace binding after the decision was made:

        - ``MISSING``      no valid decision was recorded;
        - ``UNCONFIRMED``  the ticket signals a disqualifier the record denies;
        - ``UNVERIFIED``   no public test ran after the decision;
        - ``STALE``        the latest post-decision run predates the current work;
        - ``CONTRADICTED`` the latest post-decision run at the current work failed;
        - ``CONSISTENT``   the latest post-decision run at the current work passed.

        Every status is advisory information; none has completion authority.
        """
        record = self.records[-1] if self.records else None
        factor_present = bool(record and record.get("decisive_factor"))
        disqualifier_present = bool(record and record.get("disqualifier"))
        disqualifier = str(record.get("disqualifier", "")) if record else ""
        ticket_signal = _ticket_implies_disqualifier(self.ticket)
        missing_ticket_disqualifier = bool(record) and ticket_signal and (
            disqualifier.casefold() in _NONE_VALUES or not disqualifier)
        current_digest = _binding_digest(binding)
        saved_digest = record.get("binding_digest") if record else None
        work_changed = (saved_digest is not None and current_digest is not None
                        and saved_digest != current_digest)
        evidence = _latest_evidence_after(test_evidence,
                                          int(record.get("step", 0) or 0) if record else 0)
        evidence_current = bool(evidence) and current_digest is not None \
            and evidence.get("binding_digest") == current_digest
        if not (factor_present and disqualifier_present):
            status = "MISSING"
        elif missing_ticket_disqualifier:
            status = "UNCONFIRMED"
        elif evidence is None:
            status = "UNVERIFIED"
        elif not evidence_current:
            status = "STALE"
        elif evidence.get("exit_code") == 0:
            status = "CONSISTENT"
        else:
            status = "CONTRADICTED"
        result = {
            "status": status,
            "advisory_only": True,
            "step": int(step or 0),
            "decision_id": record.get("id") if record else None,
            "factor_present": factor_present,
            "disqualifier_present": disqualifier_present,
            "ticket_disqualifier_signal": ticket_signal,
            "ticket_disqualifier_unconfirmed": missing_ticket_disqualifier,
            "work_changed_since_decision": work_changed,
            "binding_match": not work_changed,
            "evidence_step": evidence.get("step") if evidence else None,
            "evidence_exit_code": evidence.get("exit_code") if evidence else None,
            "evidence_current": evidence_current,
        }
        result["rendered"] = render_recheck(record, result)
        self.last_recheck = copy.deepcopy(result)
        return result

    def public(self) -> dict[str, object]:
        latest = self.latest
        if latest is not None:
            latest.pop("binding_digest", None)
        result = {
            "factor_schema": SCHEMA,
            "status": "recorded" if latest is not None else "awaiting_decision",
            "latest": latest,
        }
        if self.last_recheck is not None:
            result["last_recheck"] = {
                key: self.last_recheck.get(key)
                for key in ("status", "decision_id", "ticket_disqualifier_unconfirmed",
                            "work_changed_since_decision", "evidence_step", "advisory_only")
            }
        return result


def render_record(record: Mapping[str, object]) -> str:
    factor = str(record.get("decisive_factor", ""))
    disqualifier = str(record.get("disqualifier", ""))
    return (f"{PROMPT_HEADER} factor={factor}; disqualifier={disqualifier}\n"
            "Re-check this factor and disqualifier at done; advisory only.")


def render_recheck(record: Mapping[str, object] | None,
                   result: Mapping[str, object]) -> str:
    factor = str(record.get("decisive_factor", "missing")) if record else "missing"
    disqualifier = str(record.get("disqualifier", "missing")) if record else "missing"
    status = str(result.get("status", "UNVERIFIED"))
    if result.get("evidence_step") is None:
        evidence = "no public test since decision"
    else:
        evidence = (f"public test step {result.get('evidence_step')} "
                    f"exit {result.get('evidence_exit_code')}"
                    + ("" if result.get("evidence_current") else " on older work"))
    changed = "; work changed since decision" if result.get("work_changed_since_decision") else ""
    return (f"[Decisive factor re-check] factor={factor}; disqualifier={disqualifier}; "
            f"status={status} ({evidence}{changed}; advisory, completion not blocked).")


def render_prompt(store: DecisiveFactorStore | None) -> str:
    if not enabled():
        return ""
    latest = store.latest if store is not None else None
    if latest is None:
        return (f"{PROMPT_HEADER} Before committing, use `decide` to name one "
                "metric/property governing the choice and the ticket's disqualifier; "
                "advisory only.")
    return render_record(latest)
