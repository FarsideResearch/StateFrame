"""MICRO-CALL CONTRACT LAYER — bounded semantic sensors, never kernel authorities.

Design authority: sol_answers a_microcalls.md (the doctrine + the ranked
shopping list) and a_microcalls_crucible.md (the hostile review, whose
guardrails are MANDATORY here, not advisory).

The one sentence: *a micro-call is a typed, bounded evidence vote.* It may
propose an atom. It may never own authorization, state mutation, budget
accounting, final tie-breaking, or execution.

What this module enforces, in code:

1. THE CONTRACT IS A REGISTRATION GATE, not a convention. `register_site`
   REJECTS a spec missing `verify`, missing `fallback`, or carrying an
   output schema that is not a code-parseable atom. The crucible's rule —
   "if code cannot cheaply verify the atom, there is no micro-call site" —
   is a raised exception, not a comment.

2. THE PAMPHLET RULE. All evidence travels inline in one canonical prompt,
   as `(evidence_id, text)` pairs. No paths, no retrieval instructions, no
   "inspect this file". `build_prompt` is the only prompt producer, so the
   prompt hash is a real pin.

3. FAIL-CLOSED PARSING. One strict JSON atom, governed keys, governed
   answer domain. Anything else — prose, an invented candidate, a second
   key, a bare letter, an empty reply — parses to nothing and takes the
   deterministic path. There is no "best effort" recovery.

4. AN UNVERIFIED ANSWER NEVER PROPAGATES. `verify` runs on every parsed
   answer. False, or a raising verifier, is a REJECTION: the result carries
   the fallback's answer, `verified=False`, and an ABSTAIN DecisionEvent
   naming the rejection reason. The model's answer survives only in `raw`
   and in the event, never in the returned atom.

5. EVERY INVOCATION IS ACCOUNTED. A U-row opens before the call and closes
   after it, on the ledger, under a REAL closure criterion — the accepted
   path closes on `deterministic_check:<site>:verify`, the fallback path on
   `deterministic_check:<site>:fallback`, because the deterministic path IS
   a deterministic check. There is no `model_said_so`, here or in uledger.

6. TRANSPORT IS INJECTED AND DEFAULTS TO NONE. `transport=None` means
   OFFLINE: no network, no call, straight to the fallback with an event
   saying so. Every registered site therefore ships dormant, and the whole
   layer is testable without a model. Wiring guidance for later is in
   `TRANSPORT_CONTRACT` below.

7. KEEP/DROP IS ONE PRE-REGISTERED NUMBER. `SiteRegistry.record_pair`
   books per-site paired outcomes (micro vs the deterministic twin, minus
   call cost) so `cycle_report`-style judgement can compute PNDU later.
   Abstentions and fallbacks are counted; filtering to "confident" calls is
   exactly the Goodhart the crucible warns about, so `record_pair` has no
   way to exclude a case.

The first registered site is #1 on the doctrine's ranked list AND the
crucible's chosen falsifying target: symptom-phrase -> candidate-line
linking, the 52.8% mutation-menu plateau. It is registered dormant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .decision_event import (ABSTAIN, SENSOR_READ, ULEDGER_CLOSE,
                             ULEDGER_OPEN, DecisionEvent, Producer,
                             content_ref)
from .uledger import (COST_CLASSES, DETERMINISTIC_CHECK, UNVERIFIED_CLAIM,
                      ClosureRecord, ULedger, URow)

MODULE = "detreason.microcall"
VERSION = "1"

# The literal a site returns when it declines to answer. Governed, and never
# a valid candidate id (`register_site` enforces that).
ABSTAIN_TOKEN = "UNKNOWN"

# Documented for the later wiring pass; deliberately NOT imported. fixer.py's
# `_post(body, url)` is the ONE network seam in this repo, and `_call` wraps it
# with the temp-0 / fence-strip / retry-on-empty semantics. A transport for this
# layer is any callable with this signature:
#
#     transport(prompt: str) -> (text: str, raw: object)
#
# so the eventual live adapter is a three-line closure over `fixer._call`:
#
#     def make_transport(url, seed=1):
#         from fixer import _call
#         return lambda prompt: _call(prompt, seed=seed, url=url, max_tokens=64,
#                                     n_probs=2)
#
# The crucible's batching guardrail (E45) applies to that adapter: singleton
# execution, pinned decode config, and cross-boot replay before any site is
# promoted out of shadow mode.
TRANSPORT_CONTRACT = "transport(prompt: str) -> (text: str, raw: object)"


class MicroCallError(ValueError):
    """A contract violation. Raised at registration or construction time."""


# --------------------------------------------------------------- output atoms
MENU = "menu"
YES_NO = "yes_no"
BOUNDED_INT = "bounded_int"
SCHEMA_KINDS = (MENU, YES_NO, BOUNDED_INT)

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


@dataclass(frozen=True)
class OutputSchema:
    """The code-parseable atom a site accepts. Nothing else is a schema.

    `menu`        — one of `options` (candidate ids/letters) or ABSTAIN.
    `yes_no`      — YES / NO / ABSTAIN.
    `bounded_int` — an integer in [lo, hi] or ABSTAIN.

    Free text, floats, confidence scores and prose are deliberately absent:
    a self-reported confidence is the model grading itself (crucible Q1,
    the E38 self-coherence trap).
    """
    kind: str
    options: tuple = ()
    lo: int | None = None
    hi: int | None = None

    def __post_init__(self):
        if self.kind not in SCHEMA_KINDS:
            raise MicroCallError(
                f"unknown output schema kind {self.kind!r}; "
                f"code-parseable atoms are {SCHEMA_KINDS}")
        if self.kind == MENU:
            opts = tuple(self.options)
            if len(opts) < 2:
                raise MicroCallError("a menu schema needs at least 2 options")
            if len(set(opts)) != len(opts):
                raise MicroCallError(f"duplicate menu options: {opts}")
            for o in opts:
                if not isinstance(o, str) or not _ID_RE.match(o):
                    raise MicroCallError(
                        f"menu option {o!r} is not a bare identifier; "
                        "candidate ids must be code-comparable atoms")
                if o == ABSTAIN_TOKEN:
                    raise MicroCallError(
                        f"{ABSTAIN_TOKEN} is the abstain token, never an option")
            object.__setattr__(self, "options", opts)
            if self.lo is not None or self.hi is not None:
                raise MicroCallError("menu schema takes no lo/hi bounds")
        elif self.kind == YES_NO:
            if self.options or self.lo is not None or self.hi is not None:
                raise MicroCallError("yes_no schema takes no options or bounds")
            object.__setattr__(self, "options", ("YES", "NO"))
        else:  # BOUNDED_INT
            if self.options:
                raise MicroCallError("bounded_int schema takes no options")
            if not isinstance(self.lo, int) or isinstance(self.lo, bool) or \
                    not isinstance(self.hi, int) or isinstance(self.hi, bool):
                raise MicroCallError("bounded_int schema needs integer lo and hi")
            if self.hi <= self.lo:
                raise MicroCallError(f"bounded_int needs hi > lo, got {self.lo}..{self.hi}")
            if self.hi - self.lo > 64:
                raise MicroCallError(
                    "bounded_int range wider than 64 is not a bounded atom")

    # -- prompt half ---------------------------------------------------------
    def answer_domain(self) -> str:
        if self.kind == MENU:
            return " | ".join(self.options + (ABSTAIN_TOKEN,))
        if self.kind == YES_NO:
            return f"YES | NO | {ABSTAIN_TOKEN}"
        return f"an integer {self.lo}..{self.hi} | {ABSTAIN_TOKEN}"

    # -- parse half ----------------------------------------------------------
    def coerce(self, value: Any) -> Any:
        """The value as a governed atom, or None. Fail-closed by construction."""
        if self.kind == BOUNDED_INT:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value if self.lo <= value <= self.hi else None
            if isinstance(value, str) and value.strip() == ABSTAIN_TOKEN:
                return ABSTAIN_TOKEN
            return None
        if not isinstance(value, str):
            return None
        v = value.strip()
        if v == ABSTAIN_TOKEN:
            return ABSTAIN_TOKEN
        if self.kind == YES_NO:
            return v if v in ("YES", "NO") else None
        return v if v in self.options else None


# ------------------------------------------------------------------ the spec
@dataclass(frozen=True)
class MicroCall:
    """One registered micro-call site. Frozen, hashable by identity of fields.

    `verify` is `(answer, spec) -> bool` and is REQUIRED: the crucible's
    verification asymmetry means an answer code cannot cheaply check is not
    an answer at all. `fallback` is `(spec) -> answer` and is REQUIRED: it
    is the deterministic path this site is trying, and may fail, to beat.
    """
    site_id: str
    question: str
    evidence: tuple                     # ((evidence_id, text), ...)
    output_schema: OutputSchema
    verify: Callable[[Any, "MicroCall"], bool]
    fallback: Callable[["MicroCall"], Any]
    budget_class: str = "cheap"
    context: dict = field(default_factory=dict)   # site-private, prompt-invisible

    def __post_init__(self):
        if not isinstance(self.site_id, str) or not self.site_id.strip():
            raise MicroCallError("site_id is required")
        if not isinstance(self.question, str) or not self.question.strip():
            raise MicroCallError("question is required")
        if self.question.count("?") != 1:
            raise MicroCallError(
                f"site {self.site_id}: a micro-call asks exactly ONE narrow "
                f"question; found {self.question.count('?')} '?' in the ask")
        if not isinstance(self.output_schema, OutputSchema):
            raise MicroCallError(
                f"site {self.site_id}: output_schema must be an OutputSchema; "
                "free-text output is not a micro-call")
        ev = tuple(tuple(e) for e in self.evidence)
        if not ev:
            raise MicroCallError(
                f"site {self.site_id}: the pamphlet rule — a call with no "
                "inline evidence is asking the model to recall, not to read")
        seen = set()
        for e in ev:
            if len(e) != 2:
                raise MicroCallError(
                    f"site {self.site_id}: evidence entries are (id, text)")
            eid, text = e
            if not isinstance(eid, str) or not _ID_RE.match(eid):
                raise MicroCallError(
                    f"site {self.site_id}: evidence id {eid!r} is not a bare "
                    "identifier")
            if eid in seen:
                raise MicroCallError(
                    f"site {self.site_id}: duplicate evidence id {eid!r}")
            seen.add(eid)
            if not isinstance(text, str) or not text.strip():
                raise MicroCallError(
                    f"site {self.site_id}: evidence {eid} carries no text")
        object.__setattr__(self, "evidence", ev)
        if not callable(self.verify):
            raise MicroCallError(
                f"site {self.site_id}: verify is REQUIRED and must be callable "
                "— an unverifiable site is not a micro-call site")
        if not callable(self.fallback):
            raise MicroCallError(
                f"site {self.site_id}: fallback is REQUIRED and must be "
                "callable — there is no site without a deterministic path")
        if self.budget_class not in COST_CLASSES:
            raise MicroCallError(
                f"site {self.site_id}: unknown budget_class "
                f"{self.budget_class!r}; known {sorted(COST_CLASSES)}")
        if not isinstance(self.context, dict):
            raise MicroCallError(f"site {self.site_id}: context must be a dict")

    # -- pins ----------------------------------------------------------------
    @property
    def evidence_ids(self) -> tuple:
        return tuple(eid for eid, _ in self.evidence)

    def evidence_digest(self) -> str:
        return content_ref("\n".join(f"{i}\x1f{t}" for i, t in self.evidence))

    def prompt_hash(self) -> str:
        return content_ref(build_prompt(self))

    def cost(self) -> float:
        return COST_CLASSES[self.budget_class]


# ------------------------------------------------------------------- prompting
_PROMPT_SCHEMA = "microcall/1"


def build_prompt(spec: MicroCall) -> str:
    """The one canonical prompt. Pamphlet rule: everything is in here.

    Deliberately boring and fully ordered: prompt text is executable policy
    (crucible Q1), so it is produced in exactly one place and hashed.
    """
    lines = [
        f"SITE: {spec.site_id}",
        "You are answering ONE narrow question from the evidence below.",
        "Use ONLY the evidence below. Do not infer anything not written here.",
        "",
        "EVIDENCE:",
    ]
    for eid, text in spec.evidence:
        lines.append(f"[{eid}] {text}")
    lines += [
        "",
        f"QUESTION: {spec.question.strip()}",
        "",
        f"ANSWER MUST BE ONE OF: {spec.output_schema.answer_domain()}",
        f'Answer {ABSTAIN_TOKEN} if the evidence does not decide it.',
        "",
        'Reply with only this JSON and nothing else:',
        '{"answer": <one of the above>}',
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- parsing
_PARSE_JSON = "json"
_GOVERNED_REPLY_KEYS = {"answer", "evidence_ids"}


def parse_answer(text: Any, schema: OutputSchema) -> tuple:
    """`(answer, parse_by, reason)`. `answer is None` means REJECTED.

    Fail-closed, and pointedly not lenient: a bare letter, a fenced blob with
    prose around it, an extra key, or an invented candidate all reject. The
    silent-drop family of bugs starts with a parser that "recovers" something
    plausible from output that did not follow the contract.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "", "empty_response"
    raw = text.strip()
    # A single ```-fenced block is a llama.cpp formatting habit, not a content
    # change; stripping whole fence lines is the only tolerance granted.
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines()
                        if not l.strip().startswith("```")).strip()
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None, "", "not_json"
    if not isinstance(obj, dict):
        return None, "", "not_an_object"
    unknown = set(obj) - _GOVERNED_REPLY_KEYS
    if unknown:
        return None, "", f"unknown_reply_keys:{sorted(unknown)}"
    if "answer" not in obj:
        return None, "", "no_answer_key"
    coerced = schema.coerce(obj["answer"])
    if coerced is None:
        return None, "", "answer_outside_domain"
    return coerced, _PARSE_JSON, ""


def _margin_from_raw(raw: Any) -> float | None:
    """Top-1 minus top-2 logprob on the first token, when the transport gives
    us logprobs. `None` is an honest unknown — never a fabricated 1.0."""
    try:
        lp = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        vals = sorted((float(t["logprob"]) for t in lp), reverse=True)
    except (KeyError, TypeError, ValueError):
        return None
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


# ------------------------------------------------------------------- execution
def _producer(spec: MicroCall) -> Producer:
    return Producer(module=MODULE, version=VERSION,
                    bundle_fingerprint=spec.prompt_hash())


def _base_context(spec: MicroCall, extra: dict | None = None) -> dict:
    ctx = {
        "kind": "microcall",
        "key": spec.site_id,
        "site_id": spec.site_id,
        "prompt_schema": _PROMPT_SCHEMA,
        "prompt_hash": spec.prompt_hash(),
        "evidence_digest": spec.evidence_digest(),
        "schema_kind": spec.output_schema.kind,
        "budget_class": spec.budget_class,
    }
    if extra:
        ctx.update(extra)
    return ctx


def execute(spec: MicroCall, transport: Callable | None = None, *,
            correlation_id: str | None = None,
            ledger: ULedger | None = None,
            registry: "SiteRegistry | None" = None) -> dict:
    """Run one micro-call site. Returns the result atom.

    ``{answer, verified, parse_by, margin, raw, events, fallback_invoked,
       reject_reason, site_id, cost}``

    `answer` is ALWAYS safe to propagate: it is either a verified model atom
    or the deterministic fallback's. `verified=False` means the model's atom
    was rejected and is present only in `raw` and in the ABSTAIN event.

    `transport=None` is offline and is the default: the site is registered
    but dormant, no network is touched, and the deterministic path answers.
    """
    if not isinstance(spec, MicroCall):
        raise MicroCallError("execute takes a MicroCall spec")
    cid = correlation_id or f"microcall:{spec.site_id}"
    led = ledger if ledger is not None else ULedger(decision_id=cid)
    prod = _producer(spec)
    events: list = []

    def emit(dtype, **kw):
        kw.setdefault("correlation_id", cid)
        kw.setdefault("producer", prod)
        ev = DecisionEvent(decision_type=dtype, **kw)
        events.append(ev)
        return ev

    # -- U-row opens BEFORE the call. An attempted call is charged whether or
    #    not it returns anything usable; retries are not free evidence.
    row_id = f"u:{spec.site_id}:{spec.evidence_digest()}"
    row = URow(id=row_id, utype=UNVERIFIED_CLAIM,
               claim=f"micro-call {spec.site_id}: {spec.question.strip()}",
               u=1.0, severity=0.5, source=MODULE, owner=MODULE,
               closure_criterion=DETERMINISTIC_CHECK,
               evidence_ids=list(spec.evidence_ids))
    led.open(row)
    open_ev = emit(ULEDGER_OPEN, choice=row_id,
                   inputs_refs=(spec.evidence_digest(),),
                   evidence_refs=spec.evidence_ids,
                   u_rows_opened=(row_id,),
                   context=_base_context(spec, {"urow": row.to_dict()}))

    text, raw, margin = "", None, None
    answer, parse_by, reject_reason = None, "", ""
    transport_error = ""

    if transport is None:
        reject_reason = "offline_no_transport"
    else:
        prompt = build_prompt(spec)
        try:
            out = transport(prompt)
        except Exception as exc:                      # transport is foreign code
            out = None
            transport_error = f"{type(exc).__name__}"
            reject_reason = "transport_error"
        if out is not None:
            if isinstance(out, tuple) and len(out) == 2:
                text, raw = out
            else:
                text, raw = out, None
            margin = _margin_from_raw(raw)
            answer, parse_by, reject_reason = parse_answer(text, spec.output_schema)

    # -- SENSOR_READ records the attempt itself, parsed or not.
    sensor_ev = emit(SENSOR_READ, causation_id=open_ev.event_id,
                     choice=answer,
                     inputs_refs=(spec.prompt_hash(),),
                     evidence_refs=spec.evidence_ids,
                     options_considered=tuple(
                         {"option": o, "score": None,
                          "margin": margin if o == answer else None}
                         for o in spec.output_schema.options),
                     context=_base_context(spec, {
                         "parse_by": parse_by,
                         "raw_text": text if isinstance(text, str) else "",
                         "transport": "offline" if transport is None else "live",
                         "transport_error": transport_error,
                         "margin": margin,
                     }))

    # -- verification. An abstention is a valid atom and NOT an answer.
    verified = False
    if answer is not None and answer != ABSTAIN_TOKEN:
        try:
            verified = bool(spec.verify(answer, spec))
        except Exception as exc:
            verified = False
            reject_reason = f"verifier_raised:{type(exc).__name__}"
        if not verified and not reject_reason:
            reject_reason = "verify_failed"
    elif answer == ABSTAIN_TOKEN:
        reject_reason = "model_abstained"

    if verified:
        final = answer
        fallback_invoked = False
        check_id = f"{spec.site_id}:verify"
    else:
        # The model's atom stops here. It does not reach the caller.
        final = spec.fallback(spec)
        fallback_invoked = True
        check_id = f"{spec.site_id}:fallback"
        emit(ABSTAIN, causation_id=sensor_ev.event_id,
             choice=final,
             evidence_refs=spec.evidence_ids,
             outcome="rejected",
             context=_base_context(spec, {
                 "reject_reason": reject_reason or "unverified",
                 "rejected_answer": answer,
                 "fallback_answer": final,
                 "parse_by": parse_by,
             }))

    # -- U-row closes on a REAL criterion. Both paths end in a deterministic
    #    check: the verifier, or the deterministic fallback itself.
    led.close(row_id, ClosureRecord(
        criterion=DETERMINISTIC_CHECK, actor=MODULE,
        evidence_ids=list(spec.evidence_ids),
        detail={"check_id": check_id, "passed": True},
        residual_u=0.0))
    emit(ULEDGER_CLOSE, causation_id=open_ev.event_id,
         choice=final, u_rows_closed=(row_id,),
         evidence_refs=spec.evidence_ids,
         outcome="verified" if verified else "fallback",
         context=_base_context(spec, {"check_id": check_id,
                                      "cost": spec.cost()}))

    result = {
        "site_id": spec.site_id,
        "answer": final,
        "verified": verified,
        "parse_by": parse_by,
        "margin": margin,
        "raw": text if isinstance(text, str) else "",
        "events": tuple(events),
        "fallback_invoked": fallback_invoked,
        "reject_reason": reject_reason,
        "cost": spec.cost(),
    }
    reg = registry if registry is not None else REGISTRY
    reg.record_execution(spec.site_id, result)
    return result


# -------------------------------------------------------------------- registry
@dataclass
class SiteStats:
    """Per-site bookkeeping. Executions are attempts; pairs are judgement.

    The paired half deliberately has no filter: every eligible case counts,
    including abstentions and fallbacks. Scoring only "confident" calls is
    the Goodhart the crucible names by name.
    """
    site_id: str
    executions: int = 0
    verified: int = 0
    fallbacks: int = 0
    abstentions: int = 0
    parse_failures: int = 0
    pairs: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    net_utility: float = 0.0          # sum(micro - code - cost)

    @property
    def win_rate(self) -> float | None:
        """Paired win rate over ALL pairs. None before any pair is booked."""
        return None if not self.pairs else self.wins / self.pairs

    @property
    def pndu(self) -> float | None:
        """Costed paired net utility: mean(micro - code - call_cost)."""
        return None if not self.pairs else self.net_utility / self.pairs

    @property
    def verified_rate(self) -> float | None:
        return None if not self.executions else self.verified / self.executions

    def to_dict(self) -> dict:
        return {"site_id": self.site_id, "executions": self.executions,
                "verified": self.verified, "fallbacks": self.fallbacks,
                "abstentions": self.abstentions,
                "parse_failures": self.parse_failures,
                "pairs": self.pairs, "wins": self.wins, "losses": self.losses,
                "ties": self.ties, "net_utility": self.net_utility,
                "win_rate": self.win_rate, "pndu": self.pndu,
                "verified_rate": self.verified_rate}


class SiteRegistry:
    """The registered sites and their paired bookkeeping.

    Registration is the contract gate: a spec that reaches `register_site`
    has a verifier, a deterministic fallback, a governed output atom, and
    inline evidence — or it does not get in.
    """

    def __init__(self):
        self._sites: dict = {}
        self._stats: dict = {}

    # -- registration --------------------------------------------------------
    def register_site(self, spec: MicroCall, *, replace: bool = False) -> MicroCall:
        if not isinstance(spec, MicroCall):
            raise MicroCallError(
                "register_site takes a MicroCall; a dict is not a contract")
        # MicroCall.__post_init__ already enforced the contract. Re-assert the
        # three load-bearing halves here so a future refactor that loosens the
        # dataclass still cannot get an unverifiable site onto the registry.
        if not callable(getattr(spec, "verify", None)):
            raise MicroCallError(f"site {spec.site_id}: registration REJECTED, "
                                 "no verify")
        if not callable(getattr(spec, "fallback", None)):
            raise MicroCallError(f"site {spec.site_id}: registration REJECTED, "
                                 "no fallback")
        if not isinstance(getattr(spec, "output_schema", None), OutputSchema):
            raise MicroCallError(f"site {spec.site_id}: registration REJECTED, "
                                 "no code-parseable output schema")
        if spec.site_id in self._sites and not replace:
            raise MicroCallError(
                f"site {spec.site_id} already registered; pass replace=True "
                "deliberately rather than shadowing a live site")
        self._sites[spec.site_id] = spec
        self._stats.setdefault(spec.site_id, SiteStats(site_id=spec.site_id))
        return spec

    def get(self, site_id: str) -> MicroCall:
        try:
            return self._sites[site_id]
        except KeyError:
            raise MicroCallError(f"no registered site {site_id!r}") from None

    def site_ids(self) -> tuple:
        return tuple(sorted(self._sites))

    # -- bookkeeping ---------------------------------------------------------
    def stats(self, site_id: str) -> SiteStats:
        if site_id not in self._stats:
            raise MicroCallError(f"no stats for site {site_id!r}")
        return self._stats[site_id]

    def record_execution(self, site_id: str, result: dict) -> SiteStats:
        st = self._stats.setdefault(site_id, SiteStats(site_id=site_id))
        st.executions += 1
        if result.get("verified"):
            st.verified += 1
        if result.get("fallback_invoked"):
            st.fallbacks += 1
        reason = result.get("reject_reason") or ""
        if reason == "model_abstained":
            st.abstentions += 1
        elif reason in ("not_json", "not_an_object", "no_answer_key",
                        "empty_response", "answer_outside_domain") or \
                reason.startswith("unknown_reply_keys"):
            st.parse_failures += 1
        return st

    def record_pair(self, site_id: str, micro_score: float, code_score: float,
                    *, call_cost: float | None = None) -> SiteStats:
        """Book ONE paired case: the micro path against its deterministic twin.

        Both scores come from outside this layer — held-out labels or an
        independently checked downstream outcome. Never from the same model
        that produced the answer (E38: model agreement is not validation).
        """
        st = self._stats.setdefault(site_id, SiteStats(site_id=site_id))
        for name, v in (("micro_score", micro_score), ("code_score", code_score)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                raise MicroCallError(f"{name} must be a real number, got {v!r}")
        cost = call_cost
        if cost is None:
            spec = self._sites.get(site_id)
            cost = spec.cost() if spec is not None else 0.0
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost != cost:
            raise MicroCallError(f"call_cost must be a real number, got {cost!r}")
        st.pairs += 1
        if micro_score > code_score:
            st.wins += 1
        elif micro_score < code_score:
            st.losses += 1
        else:
            st.ties += 1
        st.net_utility += float(micro_score) - float(code_score) - float(cost)
        return st

    def report(self) -> dict:
        """cycle_report-shaped: one row per site, every case counted."""
        return {sid: self._stats[sid].to_dict() for sid in sorted(self._stats)}


REGISTRY = SiteRegistry()


def register_site(spec: MicroCall, *, replace: bool = False) -> MicroCall:
    """Register on the process-wide registry. See `SiteRegistry.register_site`."""
    return REGISTRY.register_site(spec, replace=replace)


# ============================================================== SITE #1 =======
# symptom-phrase -> candidate-line linking (a_microcalls ranked list item 1,
# and the crucible's chosen falsifying target). The mutation menu's measured
# per-menu accuracy is 52.8% — the worst ceiling we have — and its output is
# bounded to known code spans, which is exactly what makes it cheaply
# verifiable. Nothing else on the list has both properties.
#
# The call proposes WHICH candidate line best explains the symptom. It does not
# construct the candidate set, does not order the menu, and does not decide the
# mutation: code owns candidate identity, ranking and application (crucible Q1,
# E49/E50 — a model-selected ordinal is not evidence).

SYMPTOM_LINE_SITE = "skeptic.symptom_line_link"

_STOPWORDS = frozenset("""
a an and are as at be been but by can did do does for from had has have if in
into is it its not of on or that the then there these this to was were when
which while with you your should would could when's it's don't doesn't
one all any some very just also here now than too such each other
""".split())

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _content_words(text: str) -> set:
    """Lowercased identifier-ish words, minus stopwords. Also splits
    snake_case and camelCase so `parse_header` matches "parse" and "header"."""
    out = set()
    for w in _WORD_RE.findall(text or ""):
        lw = w.lower()
        if len(lw) > 1 and lw not in _STOPWORDS:
            out.add(lw)
        for part in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", w):
            lp = part.lower()
            if len(lp) > 2 and lp not in _STOPWORDS:
                out.add(lp)
    return out


def _overlap(symptom: str, line: str) -> int:
    return len(_content_words(symptom) & _content_words(line))


def symptom_line_link_spec(symptom: str, candidates, *,
                           site_id: str = SYMPTOM_LINE_SITE,
                           budget_class: str = "cheap") -> MicroCall:
    """Build the site-#1 spec.

    `candidates` is an ordered sequence of `(line_id, line_no, text)`. The ids
    are code-minted (`L1..Ln`), never model-minted, so a hallucinated id cannot
    resolve. Line numbers travel in the spec's private context, not the prompt:
    the model must decide from the code, not from a position (E49/E50).

    VERIFY (cheap, deterministic, three checks, all must pass):
      1. the id is one this call actually offered;
      2. its line is inside the declared candidate span — no invented span;
      3. the chosen line shares at least one content word with the symptom.
    (3) is the real bite. It cannot certify correctness — nothing cheap can —
    but it rejects the failure this site exists to catch: a confident pick with
    no lexical connection to the symptom at all, which is what a position-decided
    menu answer looks like from the outside.

    FALLBACK is the deterministic ranking this site must beat: highest content-
    word overlap, ties broken by earliest candidate. Deliberately the same
    signal the verifier uses, so a "win" means the model picked a line the code
    would NOT have picked and that still survived the cross-check.
    """
    if not isinstance(symptom, str) or not symptom.strip():
        raise MicroCallError("symptom_line_link needs a symptom phrase")
    cands = []
    for c in candidates or ():
        c = tuple(c)
        if len(c) != 3:
            raise MicroCallError(
                "candidates are (line_id, line_no, text) triples")
        lid, lno, text = c
        if not isinstance(lno, int) or isinstance(lno, bool) or lno < 0:
            raise MicroCallError(f"candidate {lid!r} needs a line number")
        cands.append((str(lid), lno, str(text)))
    if len(cands) < 2:
        raise MicroCallError(
            "symptom_line_link needs at least 2 candidate lines; a one-option "
            "menu is a decision code already made")

    evidence = [("S1", f"Reported symptom: {symptom.strip()}")]
    evidence += [(lid, f"candidate line: {text.strip()}") for lid, _, text in cands]
    ids = tuple(lid for lid, _, _ in cands)
    line_nos = {lid: lno for lid, lno, _ in cands}
    texts = {lid: text for lid, _, text in cands}
    span = (min(line_nos.values()), max(line_nos.values()))

    def verify(answer, spec) -> bool:
        ctx = spec.context
        if answer not in ctx["candidate_ids"]:
            return False
        lo, hi = ctx["span"]
        if not (lo <= ctx["line_nos"][answer] <= hi):
            return False
        return _overlap(ctx["symptom"], ctx["texts"][answer]) >= 1

    def fallback(spec):
        ctx = spec.context
        best, best_score = None, -1
        for i, lid in enumerate(ctx["candidate_ids"]):
            score = _overlap(ctx["symptom"], ctx["texts"][lid])
            if score > best_score:
                best, best_score = lid, score
        return best

    return MicroCall(
        site_id=site_id,
        question=("Which candidate line best explains the reported symptom?"),
        evidence=tuple(evidence),
        output_schema=OutputSchema(kind=MENU, options=ids),
        verify=verify,
        fallback=fallback,
        budget_class=budget_class,
        context={"symptom": symptom.strip(), "candidate_ids": ids,
                 "line_nos": line_nos, "texts": texts, "span": span},
    )


# The site is registered at import so the registry, the contract gate and the
# bookkeeping are live — but every caller must pass a transport to reach a
# model, and the default is None. It ships DORMANT, per the crucible: shadow
# mode first, promotion only on a positive pre-registered PNDU interval.
register_site(symptom_line_link_spec(
    "placeholder registration probe",
    [("L1", 1, "placeholder registration probe line one"),
     ("L2", 2, "placeholder registration probe line two")],
))
