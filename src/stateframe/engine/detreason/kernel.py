"""DETREASON KERNEL (B1) — the deterministic judgement engine, complete.

verdict = deterministic_kernel(policy, evidence, atom_answers)

The model (or any registered SENSOR) answers leaf ATOMS as TRUE/FALSE/UNKNOWN with
a margin; CODE owns decomposition, elimination, interval scoring, ranking, ties,
verification, certificates, and escalation. Design authority:
DETREASON_DESIGN.md + sol_answers/ (a1 architecture, a3 substrate/tri-logic,
a5 escalation, a_sense2 sensors, a_crossboot margin probe, a_hats aggregation).

Everything here is pure code except Sensor.answer implementations.
"""
from __future__ import annotations
import json, re, urllib.request
from dataclasses import dataclass, field

TRUE, FALSE, UNKNOWN, CONFLICT = "TRUE", "FALSE", "UNKNOWN", "CONFLICT"


# ----------------------------------------------------------------------- atoms
@dataclass
class Atom:
    """One leaf proposition. Positive canonical polarity, explicit subject."""
    id: str
    claim: str                       # one proposition, positive polarity
    family: str = "semantic"         # numeric|schema|semantic|direction|prose
    pivotal: bool = False            # unknown/conflict here always escalates
    evidence: str = ""               # facts-only text shown to LLM sensors
    option_id: str | None = None     # candidate this atom scores, if any
    hard: bool = False               # FALSE eliminates the option outright
    weight: float = 1.0


@dataclass
class AtomResult:
    atom_id: str
    value: str                       # TRUE/FALSE/UNKNOWN
    margin: float                    # calibrated confidence proxy [0,1]
    sensor: str
    raw: str = ""


# --------------------------------------------------------------------- sensors
class Sensor:
    """Registered answerer. Subclasses implement _answer -> (value, margin, raw)."""
    name = "base"
    families: tuple = ()

    def answer(self, atom: Atom) -> AtomResult:
        # P0 fix (crucible): a sensor failure is an UNKNOWN observation, never an
        # exception that escapes the certificate.
        try:
            v, m, raw = self._answer(atom)
        except Exception as e:
            return AtomResult(atom.id, UNKNOWN, 0.0, self.name,
                              f"sensor-error:{type(e).__name__}:{str(e)[:120]}")
        if v not in (TRUE, FALSE, UNKNOWN):
            v, m = UNKNOWN, 0.0
        return AtomResult(atom.id, v, max(0.0, min(1.0, m)), self.name, raw[:200])

    def _answer(self, atom):
        raise NotImplementedError


class ToolSensor(Sensor):
    """Deterministic code predicate. p=1 only because its preconditions are code."""
    name = "tool"

    def __init__(self, fn):
        self.fn = fn  # fn(atom) -> True/False/None

    def _answer(self, atom):
        r = self.fn(atom)
        if r is None:
            return UNKNOWN, 0.0, "tool:none"
        return (TRUE if r else FALSE), 1.0, "tool"


ATOM_Q = """You are answering ONE factual question. Evidence:
{evidence}

CLAIM: {claim}
Is the claim supported by the evidence? Reply with only one JSON object:
{{"answer": "yes"}} or {{"answer": "no"}} or {{"answer": "unknown"}}"""


class LlamaSensor(Sensor):
    """Harnessed-1B ternary sensor with n_probs top-2 logprob margin (the
    fragility detector the A10G-vs-M3 flip measured the need for)."""
    name = "llama-1b"

    def __init__(self, url, seed=31, margin_probe=True, timeout=240):
        self.url, self.seed = url, seed
        self.margin_probe, self.timeout = margin_probe, timeout

    def _call(self, prompt, seed):
        body = {"messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "top_p": 1, "max_tokens": 500, "seed": seed}
        if self.margin_probe:
            body["n_probs"] = 8
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())

    @staticmethod
    def _parse(content):
        m = re.search(r'"answer"\s*:\s*"(yes|no|unknown)"', content or "", re.I)
        if not m:
            low = (content or "").lower()
            hits = [w for w in ("yes", "no", "unknown") if re.search(rf"\b{w}\b", low)]
            if len(set(hits)) != 1:
                return None
            m_val = hits[0]
        else:
            m_val = m.group(1).lower()
        return {"yes": TRUE, "no": FALSE, "unknown": UNKNOWN}[m_val]

    @staticmethod
    def _margin(resp, answer_word):
        """Top-2 logprob gap on the PARSED answer token. P0 fix (crucible):
        unavailable/unmatched margin returns None — the caller must treat that
        as UNKNOWN-confidence, never as 'neutral'; and we only score the token
        that actually carries the parsed answer."""
        try:
            toks = resp["choices"][0]["logprobs"]["content"]
        except (KeyError, TypeError, IndexError):
            return None
        if not isinstance(toks, list):
            return None
        for tok in toks:
            t = (tok.get("token") or "").strip().strip('"').lower()
            if t == answer_word:
                tops = sorted((x["logprob"] for x in tok.get("top_logprobs", [])),
                              reverse=True)
                if len(tops) >= 2:
                    gap = tops[0] - tops[1]          # nats
                    return min(1.0, gap / 2.0)       # 2 nats ~= decisive
                return None
        return None

    def _answer(self, atom):
        evidence = re.sub(r"\{[^}]*\}", "(...)", atom.evidence)  # brace rule
        prompt = ATOM_Q.format(evidence=evidence or "(none provided)",
                               claim=atom.claim)
        resp = self._call(prompt, self.seed)
        content = (resp["choices"][0]["message"].get("content") or "").strip()
        v = self._parse(content)
        if v is None:
            return UNKNOWN, 0.0, content
        word = {TRUE: "yes", FALSE: "no", UNKNOWN: "unknown"}[v]
        m = self._margin(resp, word) if self.margin_probe else None
        if m is None:
            # no calibrated margin available: the VALUE stands but with zero
            # claimed confidence — downstream thresholds will coerce to UNKNOWN
            # unless the atom's policy accepts uncalibrated answers.
            return v, 0.0, content
        return v, m, content


# ------------------------------------------------------------ verification ops
def variant_verify(sensor: Sensor, atom: Atom, n_variants: int) -> list[AtomResult]:
    """Extra probes that CHANGE THE DECISION SURFACE (never same-prompt repeats:
    temp-0 repeats are one vote). Variants: negated claim (answer inverts),
    evidence-order flip."""
    out = []
    if n_variants >= 1:
        neg = Atom(atom.id + "/neg", f"It is NOT the case that: {atom.claim}",
                   atom.family, atom.pivotal, atom.evidence)
        r = sensor.answer(neg)
        inv = {TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[r.value]
        out.append(AtomResult(atom.id, inv, r.margin, r.sensor + "+neg", r.raw))
    if n_variants >= 2 and "\n" in atom.evidence:
        lines = atom.evidence.splitlines()
        flip = Atom(atom.id + "/flip", atom.claim, atom.family, atom.pivotal,
                    "\n".join(reversed(lines)))
        r = sensor.answer(flip)
        out.append(AtomResult(atom.id, r.value, r.margin, r.sensor + "+flip", r.raw))
    return out


def paired_entailment(sensor: Sensor, atom: Atom) -> AtomResult:
    """P1: evidence supports claim? P2: evidence supports the NEGATION?
    supported/refuted/unknown/contradictory — kills the no-evidence-vs-disproven
    conflation (a5 §2)."""
    p1 = sensor.answer(atom)
    neg = Atom(atom.id + "/negation",
               f"The evidence shows the OPPOSITE of: {atom.claim}",
               atom.family, atom.pivotal, atom.evidence)
    p2 = sensor.answer(neg)
    if p1.value == TRUE and p2.value != TRUE:
        return AtomResult(atom.id, TRUE, min(p1.margin, 1 - 0.0), "paired", p1.raw)
    if p2.value == TRUE and p1.value != TRUE:
        return AtomResult(atom.id, FALSE, p2.margin, "paired", p2.raw)
    if p1.value == TRUE and p2.value == TRUE:
        return AtomResult(atom.id, CONFLICT, 0.0, "paired", "both-supported")
    return AtomResult(atom.id, UNKNOWN, 0.0, "paired", "neither-supported")


# ---------------------------------------------------------------------- ledger
@dataclass
class Ledger:
    """P0 fix (crucible): every observation is PRESERVED; the effective value is
    an aggregation, order-independent, and CONFLICT is sticky."""
    observations: dict = field(default_factory=dict)  # atom_id -> [AtomResult]
    results: dict = field(default_factory=dict)       # atom_id -> effective AtomResult

    def record(self, r: AtomResult, unknown_threshold: float):
        if r.value in (TRUE, FALSE) and r.margin < unknown_threshold:
            r = AtomResult(r.atom_id, UNKNOWN, r.margin, r.sensor + "+thin", r.raw)
        self.observations.setdefault(r.atom_id, []).append(r)
        obs = self.observations[r.atom_id]
        decisive = {o.value for o in obs if o.value in (TRUE, FALSE)}
        if len(decisive) > 1:
            eff = AtomResult(r.atom_id, CONFLICT, 0.0, "ledger",
                             f"disagreement across {len(obs)} observations")
        elif decisive:
            best = max((o for o in obs if o.value in decisive),
                       key=lambda o: o.margin)
            eff = best
        else:
            eff = AtomResult(r.atom_id, UNKNOWN, 0.0, "ledger", "no decisive obs")
        self.results[r.atom_id] = eff

    def value(self, atom_id):
        r = self.results.get(atom_id)
        return r.value if r else UNKNOWN


# ------------------------------------------------------------------- decisions
@dataclass
class Certificate:
    verdict: str                       # ACCEPT/ABSTAIN/ESCALATE
    winner: str | None
    reason: str
    scores: dict
    atoms: dict                        # atom_id -> {value, margin, sensor}
    pivotal_issues: list
    config: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__


@dataclass
class Decision:
    """select_best over options via weighted atoms; or a gate (single option)."""
    id: str
    options: list                      # option ids
    atoms: list                        # Atom list (option_id set for per-option)
    win_margin: float = 0.10


def _validate(decision: Decision):
    """P1 fix (crucible): reject malformed decisions before any sensing."""
    if not decision.atoms:
        raise ValueError("decision has no atoms")
    ids = [a.id for a in decision.atoms]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate atom ids")
    if len(set(decision.options)) != len(decision.options):
        raise ValueError("duplicate options")
    opts = set(decision.options)
    for a in decision.atoms:
        if a.option_id is not None and a.option_id not in opts:
            raise ValueError(f"atom {a.id} references unknown option {a.option_id}")
        if not (a.weight == a.weight and 0 < a.weight < 1e6):  # NaN/neg/inf guard
            raise ValueError(f"atom {a.id} has invalid weight {a.weight}")


def collect_answers(decision: Decision, sensor: Sensor, *, unknown_threshold=0.30,
                    paired=False, variants=0) -> Ledger:
    """SENSING half (P0 split, crucible): live, non-deterministic. Returns the
    immutable observation ledger that evaluate() judges."""
    ledger = Ledger()
    for atom in decision.atoms:
        r = paired_entailment(sensor, atom) if paired else sensor.answer(atom)
        ledger.record(r, unknown_threshold)
        if variants and ledger.value(atom.id) in (TRUE, FALSE):
            for vr in variant_verify(sensor, atom, variants):
                ledger.record(vr, unknown_threshold)
    return ledger


def evaluate(decision: Decision, ledger: Ledger, *, config=None) -> Certificate:
    """VERDICT half: pure function of (decision, observations). Deterministic,
    replayable — the only thing allowed to own the verdict."""
    atoms_out = {aid: {"value": r.value, "margin": round(r.margin, 3),
                       "sensor": r.sensor}
                 for aid, r in ledger.results.items()}
    pivotal_issues = [a.id for a in decision.atoms if a.pivotal
                      and ledger.value(a.id) in (UNKNOWN, CONFLICT)]
    if pivotal_issues:
        return Certificate("ESCALATE", None, "pivotal atom unresolved",
                           {}, atoms_out, pivotal_issues, config or {})
    if any(r.value == CONFLICT for r in ledger.results.values()):
        return Certificate("ESCALATE", None, "sensor conflict",
                           {}, atoms_out,
                           [a for a, r in ledger.results.items()
                            if r.value == CONFLICT], config or {})
    # interval scoring per option: lower = earned/total, upper adds unknown mass.
    # P1 fix (crucible): a HARD atom that is UNKNOWN/CONFLICT also blocks — an
    # unverified hard requirement can never let its option win.
    scores = {}
    for opt in decision.options:
        oatoms = [a for a in decision.atoms if a.option_id == opt]
        if any(a.hard and ledger.value(a.id) in (FALSE, UNKNOWN, CONFLICT)
               for a in oatoms):
            scores[opt] = {"eliminated": True, "lo": 0.0, "hi": 0.0}
            continue
        total = sum(a.weight for a in oatoms) or 1.0
        earned = sum(a.weight for a in oatoms if ledger.value(a.id) == TRUE)
        unknown = sum(a.weight for a in oatoms if ledger.value(a.id) == UNKNOWN)
        scores[opt] = {"eliminated": False,
                       "lo": round(earned / total, 4),
                       "hi": round((earned + unknown) / total, 4)}
    live = {o: s for o, s in scores.items() if not s["eliminated"]}
    if not live:
        return Certificate("ESCALATE", None, "all options eliminated",
                           scores, atoms_out, [], config or {})
    best = max(live, key=lambda o: live[o]["lo"])
    others_hi = max((s["hi"] for o, s in live.items() if o != best), default=0.0)
    if live[best]["lo"] > others_hi + decision.win_margin:
        return Certificate("ACCEPT", best, "winner by interval margin",
                           scores, atoms_out, [], config or {})
    return Certificate("ABSTAIN", None,
                       f"no winner: best.lo={live[best]['lo']} vs "
                       f"others.hi={others_hi} (margin {decision.win_margin})",
                       scores, atoms_out, [], config or {})


def decide(decision: Decision, sensor: Sensor, *, unknown_threshold=0.30,
           paired=False, variants=0, config=None) -> Certificate:
    """Convenience: validate -> sense -> evaluate. For replay/audit use the
    split halves directly."""
    _validate(decision)
    ledger = collect_answers(decision, sensor, unknown_threshold=unknown_threshold,
                             paired=paired, variants=variants)
    return evaluate(decision, ledger, config=config)
