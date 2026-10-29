"""Canonical DecisionEvent — the data plane's one record type.

Doctrine this module encodes, in order of how expensively we learned it:

1. ONE KEY, ONE SPELLING. `from_dict` rejects unknown keys outright. A reader
   that accepts three candidate spellings for a field is an admission that
   nobody checked which one the producer emits.
2. NOTHING IS EVER MUTATED. An outcome learned later is a NEW event whose
   `causation_id` points back at the event it is the outcome of. Events are
   frozen; the stream is append-only (see `event_stream.py`, which deliberately
   has no clear/reset/truncate).
3. A GOVERNED VOCABULARY. `decision_type` comes from `DECISION_TYPES`. An
   unknown type raises at construction, not at read time three weeks later.
4. INPUTS ARE CONTENT-ADDRESSED. `content_ref(text)` gives a stable sha256 ref,
   so two events can be compared on whether they saw literally the same input.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0.0"

# ----------------------------------------------------------------- vocabulary
MENU_PICK = "menu_pick"
FILE_LOCATE = "file_locate"
CLASS_INFER = "class_infer"
FIX_PROPOSE = "fix_propose"
GATE_VERDICT = "gate_verdict"
CHAIN_EDIT = "chain_edit"
ABSTAIN = "abstain"
ESCALATE = "escalate"
SENSOR_READ = "sensor_read"
ULEDGER_OPEN = "uledger_open"
ULEDGER_CLOSE = "uledger_close"
OUTCOME_OBSERVED = "outcome_observed"

# -- work-ledger vocabulary (detreason/work_ledger.py) -----------------------
# Additive extension, never a rename: the planning subsystem records durable
# work state on THIS stream rather than inventing a second ledger. Each name
# has exactly one spelling and one producer (work_ledger.WorkLedger / resume).
PLAN_REGISTERED = "plan_registered"   # immutable plan + node ids + manifest
WORK_INTENT = "work_intent"           # WRITE-AHEAD: emitted BEFORE a mutation
WORK_RECEIPT = "work_receipt"         # completion, caused_by its intent
DRIFT_DETECTED = "drift_detected"     # world diverged from the manifest; halt
RESUME_VERDICT = "resume_verdict"     # outcome of a resume reconciliation

WORK_LEDGER_TYPES = frozenset({
    PLAN_REGISTERED, WORK_INTENT, WORK_RECEIPT, DRIFT_DETECTED, RESUME_VERDICT,
})

DECISION_TYPES = frozenset({
    MENU_PICK, FILE_LOCATE, CLASS_INFER, FIX_PROPOSE, GATE_VERDICT,
    CHAIN_EDIT, ABSTAIN, ESCALATE, SENSOR_READ, ULEDGER_OPEN,
    ULEDGER_CLOSE, OUTCOME_OBSERVED,
}) | WORK_LEDGER_TYPES


class DecisionEventError(ValueError):
    """Malformed event. Raised at construction, never swallowed."""


def content_ref(text: Any) -> str:
    """Content-addressed reference to an input: `sha256:<12 hex>`.

    Non-str inputs are stringified deliberately and visibly rather than
    silently skipped — a ref you cannot compute is a ref you must not fake.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def new_event_id() -> str:
    return uuid.uuid4().hex


# ------------------------------------------------------------------- producer
@dataclass(frozen=True)
class Producer:
    """Who emitted this, at what version, under what configuration.

    `bundle_fingerprint` is the config fingerprint the run was executing under.
    causal.py treats a fingerprint as a treatment LABEL (not a description), so
    `DecisionEvent.context['config']` carries the full nested dict when the
    producer has one.
    """
    module: str
    version: str = "0"
    bundle_fingerprint: str | None = None

    def to_dict(self) -> dict:
        return {"module": self.module, "version": self.version,
                "bundle_fingerprint": self.bundle_fingerprint}

    @staticmethod
    def from_dict(d: dict) -> "Producer":
        if not isinstance(d, dict):
            raise DecisionEventError(f"producer must be a dict, got {type(d)}")
        unknown = set(d) - {"module", "version", "bundle_fingerprint"}
        if unknown:
            raise DecisionEventError(f"unknown producer keys: {sorted(unknown)}")
        if not d.get("module"):
            raise DecisionEventError("producer needs a module")
        return Producer(module=d["module"], version=d.get("version", "0"),
                        bundle_fingerprint=d.get("bundle_fingerprint"))


# --------------------------------------------------------------------- event
_ENVELOPE_KEYS = ("event_id", "schema_version", "causation_id",
                  "correlation_id", "occurred_at", "recorded_at", "producer")
_PAYLOAD_KEYS = ("decision_type", "inputs_refs", "options_considered", "choice",
                 "evidence_refs", "certificate_ref", "u_rows_opened",
                 "u_rows_closed", "outcome", "context")
ALL_KEYS = frozenset(_ENVELOPE_KEYS + _PAYLOAD_KEYS)


@dataclass(frozen=True)
class DecisionEvent:
    """One decision, recorded once, never edited.

    `context` carries producer-supplied task identity that the projections need
    and the envelope has no business inventing: `kind`, `key`, `fn`,
    `config` (the nested dict), plus anything a specific adapter attaches
    (e.g. a serialized U-row under `urow`). It is deliberately open-shaped —
    unlike every other field, which is governed.
    """
    # ---- envelope (immutable identity + provenance)
    correlation_id: str
    producer: Producer
    decision_type: str
    event_id: str = field(default_factory=new_event_id)
    schema_version: str = SCHEMA_VERSION
    causation_id: str | None = None
    occurred_at: float = field(default_factory=time.time)
    recorded_at: float = field(default_factory=time.time)

    # ---- payload
    inputs_refs: tuple = ()
    options_considered: tuple = ()      # ({option, score, margin}, ...)
    choice: Any = None
    evidence_refs: tuple = ()
    certificate_ref: str | None = None
    u_rows_opened: tuple = ()
    u_rows_closed: tuple = ()
    outcome: Any = None                 # None until known; a later event tells
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.decision_type not in DECISION_TYPES:
            raise DecisionEventError(
                f"unknown decision_type {self.decision_type!r}; "
                f"governed vocabulary is {sorted(DECISION_TYPES)}")
        if not self.correlation_id:
            raise DecisionEventError(
                "correlation_id is required: an event that belongs to no run "
                "cannot be joined to anything")
        if not isinstance(self.producer, Producer):
            raise DecisionEventError("producer must be a Producer")
        if not self.schema_version:
            raise DecisionEventError("schema_version is required")
        # Sequences are frozen to tuples so an event cannot be edited by
        # mutating a list someone still holds a reference to.
        for f in ("inputs_refs", "options_considered", "evidence_refs",
                  "u_rows_opened", "u_rows_closed"):
            v = getattr(self, f)
            if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
                raise DecisionEventError(f"{f} must be a sequence, got {v!r}")
            object.__setattr__(self, f, tuple(v))
        for i, o in enumerate(self.options_considered):
            if not isinstance(o, dict):
                raise DecisionEventError(
                    f"options_considered[{i}] must be a dict "
                    "with keys option/score/margin")
            unknown = set(o) - {"option", "score", "margin"}
            if unknown:
                raise DecisionEventError(
                    f"options_considered[{i}] unknown keys: {sorted(unknown)}")
        if not isinstance(self.context, dict):
            raise DecisionEventError("context must be a dict")
        for f in ("occurred_at", "recorded_at"):
            v = getattr(self, f)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DecisionEventError(f"{f} must be an epoch float")

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "producer": self.producer.to_dict(),
            "decision_type": self.decision_type,
            "inputs_refs": list(self.inputs_refs),
            "options_considered": [dict(o) for o in self.options_considered],
            "choice": self.choice,
            "evidence_refs": list(self.evidence_refs),
            "certificate_ref": self.certificate_ref,
            "u_rows_opened": list(self.u_rows_opened),
            "u_rows_closed": list(self.u_rows_closed),
            "outcome": self.outcome,
            "context": dict(self.context),
        }

    @staticmethod
    def from_dict(d: dict) -> "DecisionEvent":
        """Strict: unknown keys are an error, not something to route around."""
        if not isinstance(d, dict):
            raise DecisionEventError(f"event must be a dict, got {type(d)}")
        unknown = set(d) - ALL_KEYS
        if unknown:
            raise DecisionEventError(
                f"unknown event keys: {sorted(unknown)} — one key, one spelling")
        for req in ("correlation_id", "producer", "decision_type"):
            if req not in d:
                raise DecisionEventError(f"missing required key {req!r}")
        kw = {k: d[k] for k in ALL_KEYS if k in d}
        kw["producer"] = Producer.from_dict(kw["producer"])
        return DecisionEvent(**kw)

    # ------------------------------------------------------------------ causal
    def caused(self, decision_type: str, **kw) -> "DecisionEvent":
        """A follow-on event caused by this one. Same run, new identity.

        This is how an outcome is recorded: never by mutating the original.
        """
        kw.setdefault("correlation_id", self.correlation_id)
        kw.setdefault("producer", self.producer)
        kw.setdefault("context", dict(self.context))
        return DecisionEvent(decision_type=decision_type,
                             causation_id=self.event_id, **kw)
