"""FRONT DOOR v1 — a vague utterance becomes a typed task contract, deterministically.

The committed design (``sol_answers/a_frontdoor.md``) calls this a *state-grounded
contract compiler, not a classifier*:

    utterance -> referent extraction -> state resolution -> candidate contracts
              -> ambiguity gate -> verdict (PROCEED | ASK_ONE | AMBIGUOUS)

Zero model calls. Everything here is regex, set arithmetic and the retrieval
machinery we already own (``func_index``, ``chains.seam_violations``).

THE ANTI-FALSE-CLOSURE RULE (``sol_answers/w8/a_intent_engine.md``, "low-margin
atoms become ``unknown``, not guesses"): a contract may NOT be trusted merely
because the alternatives scored lower. Winning a weak field is not evidence. So
every contract carries ``confidence_basis`` — the *positive* evidence for the
reading (a matched cue phrase, a unique symbol hit, a matched seam violation) —
and a top contract with an EMPTY basis can never PROCEED, however far ahead it
is. It gets ASK_ONE.

The ONE question is information-maximising and, wherever the candidate set makes
it possible, *discriminating*: its ``outcomes`` map answer -> resulting contract,
and ``discriminating`` is True only when at least two answers lead to different
contracts. A question that cannot change the outcome is not worth asking, and we
say so rather than pretending otherwise.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "frontdoor_eval.jsonl"

CONTRACT_TYPES = ("fix_bug", "investigate", "improve", "build_new", "question")

# ---------------------------------------------------------------------------
# Type cues. Weight is *evidence strength*, not probability: an explicit
# imperative ("fix") is stronger evidence than a symptom word ("every time").
# A cue is a (regex, weight, human-readable label) triple; the label is what
# lands in ``confidence_basis``, so the basis always names something in the
# utterance rather than a vibe.
# ---------------------------------------------------------------------------
_CUES: dict[str, tuple] = {
    "fix_bug": (
        (r"\bfix(es|ed|ing)?\b", 3.0, "explicit repair imperative 'fix'"),
        (r"\bbroken?\b|\bbusted\b|\bblows up\b|\bcrash(es|ing)?\b", 2.5, "failure word"),
        (r"\bbug\b|\bregression\b|\bflaky\b", 2.5, "defect noun"),
        (r"\bevery time\b|\balways\b|\bkeeps\b|\bnever\b", 1.5, "invariant-failure phrasing"),
        (r"\bdouble[- ]?\w+\b|\btwice\b|\bduplicate", 1.5, "duplicate-effect symptom"),
        (r"\bwrong\b|\boff by\b|\bstopped working\b|\bdoesn'?t work\b", 2.0, "incorrect-result symptom"),
    ),
    "investigate": (
        (r"^\s*why\b|\bwhy is\b|\bwhy does\b|\bwhy are\b", 3.0, "'why' — causal question"),
        (r"\bhow come\b|\bwhat'?s going on\b|\bno idea why\b", 3.0, "causal question phrasing"),
        # 'what's the deal with X' is a colloquial CAUSAL COMPLAINT, not an
        # explanatory question: the speaker is reporting that something is off
        # and asking why, which is exactly an investigation.
        (r"\bwhat'?s the deal with\b|\bwhats the deal with\b", 3.0,
         "colloquial causal complaint ('what's the deal with ...')"),
        (r"\bfigure out\b|\bdig into\b|\btrack down\b|\bwork out what\b"
         r"|\binvestigate\b|\blook into\b|\bunderstand what\b", 2.5,
         "investigation imperative"),
        (r"\bwhere is .* going\b|\bwhat'?s eating\b", 2.0, "resource-loss framing"),
    ),
    "improve": (
        (r"\bmake .{0,40}\b(readable|nicer|cleaner|faster|better|less \w+|usable"
         r"|clearer|bigger|smaller|larger|shorter|current|fresher|consistent"
         r"|quicker|simpler|prettier)\b",
         3.0, "'make X <better>' improvement frame"),
        (r"\bspeed (?:it |them |this |that )?up\b|\bmore current\b|\bup to date\b",
         2.5, "improvement request on an existing surface"),
        (r"\bimprove\b|\bpolish\b|\btighten up\b|\bclean up\b|\btidy\b", 3.0, "improvement verb"),
        (r"\bjanky\b|\bugly\b|\bunreadable\b|\bhard to read\b|\bpainful\b|\bhorrible\b",
         2.0, "quality complaint"),
    ),
    "build_new": (
        (r"\b(build|create|add|write|stand up|wire up|give me) (me )?(a|an|the|some)\b",
         3.0, "construction imperative"),
        (r"\bwe need a\b|\bwe should have a\b|\bthere'?s no\b", 2.5, "absence framing"),
        (r"\bnew (app|page|tool|module|command|engine|report|dashboard)\b", 2.0, "'new <artifact>'"),
    ),
    "question": (
        (r"^\s*(what|which|how|does|do we|is there|are there|can we|should we)\b",
         2.5, "interrogative opener"),
        (r"\bwhat does\b|\bhow does\b|\bwhat'?s the\b", 2.0, "explanatory question"),
    ),
}

# When a causal question and an explicit repair imperative co-occur ("why is the
# gym slow, fix it") the investigation is INSTRUMENTAL — the thing asked for is
# the repair. This is a stated rule, not a tiebreak-by-luck.
_INSTRUMENTAL_BONUS = 1.5

# Mutating contracts may not act on an ungrounded referent; inspection may.
_MUTATING_TYPES = frozenset({"fix_bug", "improve", "build_new"})

# v2 clarification: the key under which a scripted/human answerer returns the
# contract type, and the evidence weight a stated type carries. It is set far
# above any cue stack so a stated type is never out-argued by regexes.
TYPE_ANSWER_KEY = "__contract_type__"
STATED_TYPE_BONUS = 100.0

AMBIGUITY_MARGIN = 1.0      # closer than this => competing readings
REFERENT_MARGIN = 0.15       # two resolutions this close => ambiguous referent

_STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being do does did
doing done and or but if then than so to of in on at for with without from by
about into over under again just now here there when where how what which who
whom why i me my we our you your they them their he she his her not no nor can
could should would will shall may might must have has had thing things stuff
something anything everything please make makes made fix fixes fixed get gets
got go goes going keep keeps kept take takes took see sees saw look looks
looked need needs needed want wants wanted work works worked run runs ran
seems seem sometimes always every time never all some any one two more most
less least very really quite pretty kinda sorta again still yet also too
""".split())

# App-behaviour nouns that will never appear as a repo symbol but are the real
# referent of a user request ("lessons double-grade" is about the APP, not the repo).
# Deliberately short: everything else is derived from the repo itself.
_APP_NOUNS = frozenset("""
lesson lessons grading grade signup login dashboard checkout onboarding
learner learners email emails notification notifications queue worker workers
scheduler deploy deploys ui page pages
""".split())

_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_PATHY = re.compile(r"\b[\w./-]+\.(?:py|js|ts|vue|md|json|yaml|yml|sql)\b")
_CAPS = re.compile(r"\b[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*\b")
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list:
    return _WORD.findall((text or "").lower())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Resolution:
    """One candidate meaning for a surface phrase, with its evidence."""
    kind: str                  # seam_violation | repo_symbol | repo_file | activity
    target: str
    score: float
    detail: str = ""

    def as_tuple(self) -> tuple:
        return (self.kind, self.target, self.score)


@dataclass(frozen=True)
class Referent:
    surface_phrase: str
    resolved: tuple = ()       # tuple[Resolution], best first
    note: str = ""

    @property
    def is_resolved(self) -> bool:
        return bool(self.resolved)

    def to_dict(self) -> dict:
        return dict(surface_phrase=self.surface_phrase,
                    resolved=[r.as_tuple() for r in self.resolved],
                    note=self.note)


@dataclass(frozen=True)
class Contract:
    type: str
    score: float
    referents: tuple = ()          # tuple[Referent]
    confidence_basis: tuple = ()   # tuple[str] — POSITIVE evidence only
    missing: tuple = ()            # tuple[Referent] — extracted, unresolved

    @property
    def has_positive_evidence(self) -> bool:
        """The false-closure test. False means this reading leads only because
        the alternatives are weaker — which is not evidence, and never a licence
        to proceed."""
        return bool(self.confidence_basis)

    def to_dict(self) -> dict:
        return dict(type=self.type, score=round(self.score, 3),
                    referents=[r.to_dict() for r in self.referents],
                    confidence_basis=list(self.confidence_basis),
                    missing=[m.to_dict() for m in self.missing])


@dataclass(frozen=True)
class Question:
    text: str
    kind: str                      # contract_type | referent | open
    options: tuple = ()
    outcomes: tuple = ()           # tuple[(option, resulting-contract-key)]
    reason: str = ""

    @property
    def discriminating(self) -> bool:
        """True iff at least two answers lead to DIFFERENT contracts. A question
        whose every answer yields the same contract is not worth asking."""
        return len({o for _, o in self.outcomes}) >= 2

    def to_dict(self) -> dict:
        return dict(text=self.text, kind=self.kind, options=list(self.options),
                    outcomes=[list(o) for o in self.outcomes],
                    discriminating=self.discriminating, reason=self.reason)


@dataclass(frozen=True)
class InterpretationResult:
    utterance: str
    contracts: tuple                 # ranked, best first
    verdict: str                     # PROCEED | ASK_ONE | AMBIGUOUS
    verdict_reason: str
    question: Question | None = None
    deliberation_trigger: dict | None = None
    grounded: bool = False           # was any repo/activity state available?

    @property
    def top(self) -> Contract | None:
        return self.contracts[0] if self.contracts else None

    def to_dict(self) -> dict:
        return dict(utterance=self.utterance, verdict=self.verdict,
                    verdict_reason=self.verdict_reason, grounded=self.grounded,
                    contracts=[c.to_dict() for c in self.contracts],
                    question=self.question.to_dict() if self.question else None,
                    deliberation_trigger=self.deliberation_trigger)


@dataclass(frozen=True)
class NotYetRouted:
    """A typed refusal. Names exactly what is missing rather than pretending."""
    contract_type: str
    missing: tuple
    reason: str

    def to_dict(self) -> dict:
        return dict(routed=False, contract_type=self.contract_type,
                    missing=list(self.missing), reason=self.reason)


# ---------------------------------------------------------------------------
# Referent extraction — deterministic, no model
# ---------------------------------------------------------------------------
def extract_referents(utterance: str, domain_vocab: frozenset = frozenset()) -> list:
    """Surface phrases that could name a thing, best-evidence class first.

    Four deterministic classes, in descending specificity:
      1. quoted phrases   — the speaker delimited it themselves
      2. file paths       — unambiguous
      3. snake_case       — code-shaped by construction
      4. CapitalisedNames — not sentence-initial (that is just grammar)
      5. domain nouns     — a bare word that names something in the repo, or an
                            app-behaviour noun from ``_APP_NOUNS``

    Order within a class is by appearance, so the list is stable.
    """
    text = utterance or ""
    seen, out = set(), []

    def push(phrase: str):
        p = phrase.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)

    for m in re.findall(r"'([^']{2,120})'", text) + re.findall(r'"([^"]{2,120})"', text):
        push(m)
    for m in _PATHY.findall(text):
        push(m)
    for m in _SNAKE.findall(text):
        push(m)
    # Capitalised names, skipping the first token of each sentence.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        head = sentence.strip().split(" ")[0] if sentence.strip() else ""
        for m in _CAPS.findall(sentence):
            if m == head or m.lower() in _STOPWORDS:
                continue
            push(m)
    for w in _words(text):
        if w in _STOPWORDS:
            continue
        if w in _APP_NOUNS or w in domain_vocab:
            push(w)
    return out


# ---------------------------------------------------------------------------
# State: the thing referents resolve AGAINST
# ---------------------------------------------------------------------------
@dataclass
class RepoState:
    """Everything the resolver can point a phrase at. All fields optional — an
    ungrounded call is legal and is reported as such, never silently treated as
    'nothing matched, therefore fine'."""
    files: dict = field(default_factory=dict)          # path -> source
    symbols: dict = field(default_factory=dict)        # name -> [paths]
    violations: list = field(default_factory=list)     # chains.seam_violations rows
    activity: list = field(default_factory=list)       # [{summary, kind, ts?}]

    @property
    def grounded(self) -> bool:
        return bool(self.files or self.symbols or self.violations or self.activity)

    @property
    def vocab(self) -> frozenset:
        """Every word that names something in this repo — the domain lexicon a
        bare noun in the utterance is checked against."""
        bag = set()
        for name in self.symbols:
            bag.update(_words(name))
        for path in self.files:
            bag.update(_words(pathlib.PurePosixPath(path).stem))
        return frozenset(w for w in bag if w not in _STOPWORDS and len(w) > 2)


def _symbols_via_func_index(files: dict) -> dict | None:
    """Reuse ``func_index`` when it imports; None means fall back to an AST walk.

    We never reimplement retrieval that already exists — but we also never let a
    missing optional import turn into a silent empty index.
    """
    try:
        import func_index
    except Exception:                                   # noqa: BLE001
        return None
    try:
        fidx = func_index.build_func_index(files)
    except Exception:                                   # noqa: BLE001
        return None
    out: dict = {}
    for rec in fidx.records:
        if getattr(rec, "is_module", False):
            continue
        out.setdefault(rec.name, []).append(rec.path)
    return {k: sorted(set(v)) for k, v in out.items()}


def _symbols_via_ast(files: dict) -> dict:
    out: dict = {}
    for path, src in sorted(files.items()):
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.setdefault(node.name, []).append(path)
    return {k: sorted(set(v)) for k, v in out.items()}


_SKIP_DIRS = frozenset({"__pycache__", ".git", "node_modules", ".venv", "venv",
                        "site-packages", "build", "dist", ".mypy_cache",
                        ".pytest_cache", "corpora", "results", "bundles"})

BUILD_FILE_CAP = 400


def build_state(repo_root=None, files: dict | None = None,
                activity: list | None = None,
                file_cap: int = BUILD_FILE_CAP) -> RepoState:
    """Assemble the resolution surface. ``files`` wins over ``repo_root``.

    The walk is BOUNDED (``file_cap``, skip-list) on purpose: the front door runs
    in front of a human waiting for an answer, and an unbounded rglob over a
    monorepo turns a sub-second interpretation into minutes. A truncated walk is
    honest — unresolved referents say so — an unbounded one is a hang.
    """
    src = dict(files or {})
    if not src and repo_root:
        root = pathlib.Path(repo_root)
        for p in sorted(root.rglob("*.py")):
            if _SKIP_DIRS & set(p.parts):
                continue
            try:
                src[str(p.relative_to(root))] = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if len(src) >= file_cap:
                break

    symbols = {}
    violations: list = []
    if src:
        symbols = _symbols_via_func_index(src) or _symbols_via_ast(src)
        try:
            import chains
            violations = chains.seam_violations(chains.seam_manifest(src), src)
        except Exception:                               # noqa: BLE001
            violations = []
    return RepoState(files=src, symbols=symbols, violations=violations,
                     activity=list(activity or []))


# ---------------------------------------------------------------------------
# Referent resolution
# ---------------------------------------------------------------------------
def _violation_text(v: dict) -> str:
    return " ".join(str(v.get(k, "")) for k in ("kind", "file", "name", "detail"))


def resolve_referent(phrase: str, state: RepoState) -> Referent:
    """Score one surface phrase against state, in the committed priority order:
    seam violations, then repo symbols, then file paths, then recent activity.

    A seam violation match is the strongest signal we have — it is a *measured*
    inconsistency naming the same thing the human named — and it becomes the
    contract's evidence, not merely a pointer.
    """
    pw = set(_words(phrase))
    if not pw:
        return Referent(surface_phrase=phrase, note="no lexical content")

    res: list = []

    for v in state.violations:
        vw = set(_words(_violation_text(v)))
        name_w = set(_words(str(v.get("name", ""))))
        if pw and pw == name_w:
            score = 1.0
        elif pw and pw & name_w:
            score = 0.8
        elif pw and pw <= vw:
            score = 0.6
        else:
            continue
        res.append(Resolution(
            kind="seam_violation",
            target=f"{v.get('kind')}:{v.get('file')}:{v.get('name')}",
            score=score,
            detail=str(v.get("detail", ""))[:200]))

    for name, paths in sorted(state.symbols.items()):
        nw = set(_words(name))
        if not nw:
            continue
        if pw == nw:
            score = 1.0
        elif pw & nw and (pw <= nw or nw <= pw):
            score = 0.7
        else:
            continue
        for p in paths:
            res.append(Resolution(kind="repo_symbol", target=f"{p}::{name}",
                                  score=score, detail="symbol name match"))

    for path in sorted(state.files):
        stem_w = set(_words(pathlib.PurePosixPath(path).stem))
        if phrase.strip() == path:
            res.append(Resolution("repo_file", path, 1.0, "exact path"))
        elif stem_w and pw == stem_w:
            res.append(Resolution("repo_file", path, 0.6, "file stem match"))

    for i, act in enumerate(state.activity):
        aw = set(_words(str(act.get("summary", act))))
        if not aw:
            continue
        overlap = len(pw & aw) / len(pw)
        if overlap >= 0.5:
            res.append(Resolution(
                kind="activity",
                target=str(act.get("id", f"activity[{i}]")),
                score=round(0.5 + 0.4 * overlap, 3),
                detail=str(act.get("summary", ""))[:200]))

    res.sort(key=lambda r: (-r.score, r.kind, r.target))
    if not res:
        note = ("unresolved against repo state; may name APP behaviour "
                "(pass activity= to resolve) " if state.grounded
                else "no repo/activity state supplied — nothing was checkable")
        return Referent(surface_phrase=phrase, note=note)
    return Referent(surface_phrase=phrase, resolved=tuple(res[:8]))


# ---------------------------------------------------------------------------
# Contract typing
# ---------------------------------------------------------------------------
def _type_evidence(utterance: str) -> dict:
    """type -> (score, [basis strings]). Cue matches only — positive evidence.

    v3 wraps the cue stack in three SPEECH-ACT rules, applied before and after
    the regexes. All three are general facts about how English requests work,
    not per-example patches, and each is exercised by more than one frozen case:

      * NEGATED ACTIONS ARE NOT EVIDENCE. "do not fix the grading thing yet, i
        just want to understand it" contains the word "fix"; v2 counted it as
        3.0 of repair evidence and typed the ask as the exact thing the human
        forbade. A verb inside a forbidden-action span is masked out before any
        cue runs. This is the single worst failure mode a front door can have.
      * A POLITE REQUEST IS NOT A QUESTION. "can we get X", "is there any way to
        make X faster" are conventionalised REQUESTS in English. v2's
        interrogative-opener cue fired on the auxiliary and typed them
        ``question``, which routes an ask for work to an ask for an answer.
      * A NORMALCY CHECK IS A QUESTION even when it is full of failure words.
        "the worker died again, is that expected with the new queue" is asking
        whether the behaviour is correct, not asking for a repair.
    """
    raw = utterance or ""
    # Mask forbidden-action spans: a verb the human negated must never become
    # evidence FOR that verb's contract type.
    forbidden = [c.span for c in extract_constraints(raw)
                 if c.kind == "forbidden_action" and c.span]
    text = (_mask_spans(raw, forbidden) if forbidden else raw).lower()
    out = {}
    for ctype, cues in _CUES.items():
        score, basis = 0.0, []
        for pattern, weight, label in cues:
            if re.search(pattern, text):
                score += weight
                basis.append(f"cue: {label}")
        out[ctype] = (score, basis)

    fix_score, fix_basis = out["fix_bug"]
    inv_score, _ = out["investigate"]
    if fix_score > 0 and inv_score > 0 and re.search(r"\bfix\b", text):
        # Documented rule, not a coin flip: the investigation is instrumental to
        # the repair the speaker explicitly asked for.
        out["fix_bug"] = (fix_score + _INSTRUMENTAL_BONUS,
                          fix_basis + ["rule: causal framing + explicit repair "
                                       "imperative -> investigation is instrumental"])

    # --- v3 speech-act rules ------------------------------------------------
    req = _POLITE_REQUEST.search(text)
    if req and not _NORMALCY_CHECK.search(text):
        verb = next((g for g in (req.group("verb"), req.group("verb2"),
                                 req.group("verb3")) if g), "").lower()
        target = _REQUEST_VERB_TARGETS.get(verb)
        if target is None:
            target = ("improve" if re.search(r"\b(?:faster|better|nicer|cleaner|"
                                             r"readable|smaller|bigger)\b", text)
                      else "build_new")
        if target == "build_new" and re.search(r"\b(?:working|work|so that|so it)\b", text):
            # "can we get X working" is a repair request, not a build request:
            # the thing already exists and is expected to work.
            target = "fix_bug"
        q_score, _ = out["question"]
        if q_score > 0:
            out["question"] = (0.0, [])
        score, basis = out[target]
        out[target] = (score + _REQUEST_FRAME_WEIGHT,
                       basis + [f"rule: '{req.group(0).strip()}' is a polite "
                                f"REQUEST frame, not an interrogative — routed "
                                f"to {target}"])

    if _NORMALCY_CHECK.search(text):
        # An explicit "is that normal/expected" outranks the symptom words that
        # provoked it: the deliverable asked for is an ANSWER.
        q_score, q_basis = out["question"]
        out["question"] = (q_score + _NORMALCY_WEIGHT,
                           q_basis + ["cue: normalcy check ('is that "
                                      "normal/expected') — the ask is for a "
                                      "judgement, not a repair"])
    return out


# A conventionalised request frame: an auxiliary + a subject + a MUTATION verb.
# The verb is captured because it, not the auxiliary, says which contract this
# is: "can we GET a leaderboard" builds, "can we MAKE it faster" improves.
_POLITE_REQUEST = re.compile(
    r"\b(?:can|could|would|will)\s+(?:we|you|i)\s+(?:please\s+)?(?P<verb>get|make|"
    r"add|build|create|have|give|put|do|turn|set|show|support|switch)\b"
    r"|\bis there any way\s+(?:to|we can|you can|for us to)\s+(?P<verb2>get|make|"
    r"add|build|create|have|give|put|do|turn|set|show|support|switch)\b"
    r"|\bany chance\s+(?:we|you)\s+(?:can|could)\s+(?P<verb3>get|make|add|build|"
    r"create|have|give|put|do|turn|set|show|support|switch)\b", re.I)

_REQUEST_VERB_TARGETS = {
    "get": "build_new", "add": "build_new", "build": "build_new",
    "create": "build_new", "have": "build_new", "give": "build_new",
    "put": "build_new", "support": "build_new", "show": "build_new",
    "set": "build_new", "turn": "build_new",
    "make": "improve", "do": "improve", "switch": "improve",
}

# A request frame is strong evidence but weaker than an explicit imperative:
# it says "this is a request for work", the verb says which kind.
_REQUEST_FRAME_WEIGHT = 2.5

_NORMALCY_CHECK = re.compile(
    r"\bis (?:that|this|it) (?:normal|expected|supposed to|a bug|right|correct)\b"
    r"|\bshould it (?:be|do) that\b|\bis that (?:how|what) it(?:'s| is)? "
    r"(?:meant|supposed)\b", re.I)
_NORMALCY_WEIGHT = 3.5


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
def interpret(utterance: str, repo_root=None, extra_context: dict | None = None
              ) -> InterpretationResult:
    """Vague ask -> ranked typed contracts + a verdict. No model calls.

    ``extra_context`` may carry ``files`` (path->source dict), ``activity``
    (list of ``{summary, id}`` records) and ``state`` (a prebuilt ``RepoState``).
    """
    ctx = dict(extra_context or {})
    state = ctx.get("state") or build_state(repo_root=repo_root,
                                            files=ctx.get("files"),
                                            activity=ctx.get("activity"))
    # v2: answers already given by the human. A stated type is USER-SAID
    # evidence — the strongest kind there is — so it outranks every cue.
    clar = dict(ctx.get("clarifications") or {})
    stated_type = str(clar.get(TYPE_ANSWER_KEY, "") or "").strip().lower()
    if stated_type not in CONTRACT_TYPES:
        stated_type = ""

    phrases = extract_referents(utterance, domain_vocab=state.vocab)
    referents = [resolve_referent(p, state) for p in phrases]
    resolved = tuple(r for r in referents if r.is_resolved)
    missing = tuple(r for r in referents if not r.is_resolved)

    # A matched seam violation is positive evidence FOR a bug reading.
    violation_basis = [
        f"seam violation matched '{r.surface_phrase}': {r.resolved[0].target}"
        for r in resolved if r.resolved[0].kind == "seam_violation"]
    unique_symbol_basis = [
        f"unique symbol hit for '{r.surface_phrase}': {r.resolved[0].target}"
        for r in resolved
        if r.resolved[0].kind == "repo_symbol" and len(r.resolved) == 1]

    ev = _type_evidence(utterance)
    contracts = []
    for ctype in CONTRACT_TYPES:
        score, basis = ev[ctype]
        basis = list(basis)
        if ctype == "fix_bug" and violation_basis:
            score += 2.0 * len(violation_basis)
            basis += violation_basis
        if score > 0:
            basis += unique_symbol_basis
        if stated_type:
            if ctype == stated_type:
                score += STATED_TYPE_BONUS
                basis += [f"user answered: this is a {ctype} ask"]
            else:
                # A stated type does not merely outrank the others, it RETIRES
                # them: keeping a rival's cue evidence alive would re-trigger
                # the competing-readings gate on a question already answered.
                score = 0.0
                basis = []
        contracts.append(Contract(type=ctype, score=score,
                                  referents=tuple(referents),
                                  confidence_basis=tuple(basis),
                                  missing=missing))
    # Deterministic ranking: score desc, then the declared type order.
    contracts.sort(key=lambda c: (-c.score, CONTRACT_TYPES.index(c.type)))

    verdict, reason, question, trigger = _adjudicate(contracts, referents, state)
    return InterpretationResult(utterance=utterance, contracts=tuple(contracts),
                                verdict=verdict, verdict_reason=reason,
                                question=question, deliberation_trigger=trigger,
                                grounded=state.grounded)


def _adjudicate(contracts, referents, state):
    """The ambiguity gate. Order matters and each step is a stated policy."""
    top, second = contracts[0], contracts[1]

    # (1) FALSE-CLOSURE GUARD. Winning a weak field is not evidence. A top
    #     contract with no positive basis can never proceed, however far ahead.
    if not top.has_positive_evidence:
        q = _type_question(contracts, reason="no positive evidence for any reading")
        return "ASK_ONE", ("top contract has an EMPTY confidence_basis — it leads "
                           "only because the alternatives are weaker"), q, None

    # (2) COMPETING READINGS -> deliberation.
    if top.score - second.score < AMBIGUITY_MARGIN and second.confidence_basis:
        trigger = dict(kind="competing_contracts",
                       margin=round(top.score - second.score, 3),
                       candidates=[c.type for c in contracts if c.confidence_basis])
        return ("AMBIGUOUS",
                f"{top.type} and {second.type} are within {AMBIGUITY_MARGIN} of "
                "each other and both carry positive evidence",
                _type_question(contracts, reason="competing readings"), trigger)

    # (3) AMBIGUOUS REFERENT — two near-equal meanings for one phrase. This is
    #     the highest-value question available: its answer changes the contract.
    for r in referents:
        if len(r.resolved) >= 2 and (r.resolved[0].score - r.resolved[1].score) < REFERENT_MARGIN:
            return ("ASK_ONE",
                    f"referent '{r.surface_phrase}' has {len(r.resolved)} near-equal meanings",
                    _referent_question(r, top), None)

    # (3b) NOTHING TO ACT ON AT ALL. A mutating contract from which not one
    #      surface phrase could even be EXTRACTED names no object; "fix it" is
    #      not a repair order, it is a pronoun. This holds whether or not state
    #      was supplied — ungroundedness is not what makes it unactionable, the
    #      absence of a named thing is.
    if top.type in _MUTATING_TYPES and not referents:
        return ("ASK_ONE",
                f"{top.type} is a mutating contract and the utterance names no "
                "object at all — not one referent could be extracted",
                Question(text="What should I act on? The ask names an action but "
                              "no thing to perform it on.",
                         kind="open", options=(), outcomes=(),
                         reason="mutating contract with zero extracted referents"),
                None)

    # (4) NOTHING GROUNDED. We had state and still resolved nothing to act ON.
    #     Per the sufficiency gate, low-risk inspection (investigate/question)
    #     may proceed under uncertainty; MUTATING work may not.
    if (state.grounded and top.type in _MUTATING_TYPES
            and not any(r.is_resolved for r in referents)):
        names = ", ".join(f"'{r.surface_phrase}'" for r in referents[:3])
        subject = f"What does {names} refer to?" if names else "What should I act on?"
        return ("ASK_ONE",
                f"{top.type} is a mutating contract and no referent resolved "
                "against the available state",
                Question(text=f"{subject} Nothing in the repo, the seam violations "
                              "or the supplied activity matches it.",
                         kind="open", options=(), outcomes=(),
                         reason="unresolvable referent for a mutating contract"), None)

    return "PROCEED", "clear top contract with positive evidence", None, None


def _type_question(contracts, reason: str) -> Question:
    """Discriminating by construction: each option IS a different contract."""
    live = [c for c in contracts if c.score > 0][:3]
    if len(live) < 2:
        live = list(contracts[:2])
    opts = tuple(c.type for c in live)
    verbs = {"fix_bug": "repair it", "investigate": "find out why it happens",
             "improve": "make it better without a specific defect",
             "build_new": "build something new", "question": "just answer a question"}
    body = " or ".join(f"{verbs[o]} ({o})" for o in opts)
    return Question(text=f"Do you want me to {body}?", kind="contract_type",
                    options=opts, outcomes=tuple((o, o) for o in opts),
                    reason=reason)


def _referent_question(r: Referent, top: Contract) -> Question:
    """Options are the competing meanings; each outcome is a different contract
    (same type, different target), so the answer genuinely changes the work."""
    opts = tuple(res.target for res in r.resolved[:4])
    return Question(
        text=f"When you say '{r.surface_phrase}', do you mean "
             + ", or ".join(opts) + "?",
        kind="referent", options=opts,
        outcomes=tuple((o, f"{top.type}@{o}") for o in opts),
        reason=f"'{r.surface_phrase}' resolves to {len(r.resolved)} near-equal targets")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def to_plan(contract: Contract, files: dict | None = None):
    """fix_bug + a matched seam violation -> real ``plan_synth`` wiring.

    Everything else returns a typed ``NotYetRouted`` naming what is missing —
    per the committed route table, which is explicit that only the bounded-bug
    lane runs end to end in v1.
    """
    if contract.type != "fix_bug":
        return NotYetRouted(
            contract_type=contract.type,
            missing=_ROUTE_GAPS.get(contract.type, ("route adapter",)),
            reason=f"no v1 route from a {contract.type} contract to an executable plan")

    matched = [res for r in contract.referents for res in r.resolved
               if res.kind == "seam_violation"]
    if not matched:
        return NotYetRouted(
            contract_type="fix_bug",
            missing=("app-state incident resolver",
                     "a referent resolved to a measured seam violation"),
            reason="fix_bug contract carries no matched seam violation to compile")
    if not files:
        return NotYetRouted(
            contract_type="fix_bug", missing=("repo files dict",),
            reason="plan synthesis needs the file contents the violation was measured on")

    from detreason import plan_synth
    out = plan_synth.synthesize(files)
    wanted = {m.target for m in matched}
    out["front_door"] = dict(
        routed=True,
        matched_violations=sorted(wanted),
        plans_for_matched=[p for p in out["plans"] if _plan_touches(p, wanted)],
    )
    return out


def _plan_touches(plan_row: dict, wanted: set) -> bool:
    """Does this compiled plan address one of the violations we matched?

    The violation key we resolved against is ``kind:file:name``; a plan's targets
    are ``(kind, file, name)`` triples from the same source, so we compare the
    same three fields rather than guessing at a shared spelling.
    """
    for t in plan_row.get("targets", []):
        if len(t) >= 3 and f"{t[0]}:{t[1]}:{t[2]}" in wanted:
            return True
    return False


_ROUTE_GAPS = {
    "investigate": ("evidence adapters", "sufficiency/termination policy",
                    "provenance-rich bundle"),
    "improve": ("baseline metric", "risk budget", "acceptance criteria",
                "executable chain"),
    "build_new": ("architecture contract", "inventory integration",
                  "multi-file synthesis", "observability"),
    "question": ("cross-source retrieval", "answer-sufficiency gate",
                 "calibrated uncertainty"),
}


# ---------------------------------------------------------------------------
# Request interpretation examples
# ---------------------------------------------------------------------------
def load_corpus(path=CORPUS_PATH) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def evaluate(rows=None, extra_context: dict | None = None) -> dict:
    """Interpretation accuracy over the corpus. Reports per-row, honestly.

    Two measures, kept separate because they fail for different reasons:
      * ``type_accuracy``     — did we pick the intended contract type?
      * ``referent_accuracy`` — did the key referent survive extraction?
    """
    rows = rows if rows is not None else load_corpus()
    detail = []
    for row in rows:
        res = interpret(row["utterance"], extra_context=extra_context)
        got = res.top.type if res.top else None
        surfaces = {p.lower() for c in res.contracts[:1]
                    for p in [r.surface_phrase for r in c.referents]}
        key = str(row.get("key_referent", "")).lower()
        detail.append(dict(
            utterance=row["utterance"], expected=row["expected_type"], got=got,
            type_ok=(got == row["expected_type"]),
            key_referent=row.get("key_referent"),
            referent_ok=bool(key) and key in surfaces,
            verdict=res.verdict))
    n = len(detail) or 1
    return dict(
        n=len(detail),
        type_correct=sum(1 for d in detail if d["type_ok"]),
        type_accuracy=round(sum(1 for d in detail if d["type_ok"]) / n, 4),
        referent_correct=sum(1 for d in detail if d["referent_ok"]),
        referent_accuracy=round(sum(1 for d in detail if d["referent_ok"]) / n, 4),
        by_type={t: dict(
            n=sum(1 for d in detail if d["expected"] == t),
            correct=sum(1 for d in detail if d["expected"] == t and d["type_ok"]))
            for t in CONTRACT_TYPES},
        detail=detail,
    )


# ===========================================================================
# FRONT DOOR v2 — vague ask -> a DETERMINISTIC, COMPLETED contract
#
# v1 stopped at "which contract type is this, and is it safe to proceed". That
# leaves the actual work under-specified: knowing an ask is a ``fix_bug`` does
# not tell you WHAT to repair, what the symptom is, or how far the blast radius
# reaches. v2 adds the missing half:
#
#   1. CONTRACT SCHEMA — each type declares its REQUIRED FIELDS.
#   2. ASK-ELEVATION  — every unresolved field first tries to answer ITSELF from
#                       the utterance, the repo (via ``behavior_bridge``), or a
#                       recorded default. Only what genuinely cannot be resolved
#                       becomes a question.
#   3. THE ASSUMPTION LEDGER — every self-answer is recorded, sourced, and
#                       flagged ``vetoable``. Rule #197: a model-authored field
#                       is NON-AUTHORITATIVE and must be visible as such. There
#                       are no silent defaults anywhere in this module.
#   4. THE COMPLETION LOOP — bounded (3 questions, hard cap), provably
#                       terminating, and deterministic.
#
# THE KERNEL-WIDE PRIMITIVE. ``resolve_or_ask`` is the ladder every field walks,
# exposed as a pure function so ANY engine can use it mid-execution rather than
# re-inventing "should I ask the human about this?" five times:
#
#     resolve_or_ask(field_spec, evidence_context, budget_state)
#         -> FieldVerdict(RESOLVED | SELF_ANSWERED | ASK | PROCEED_WITHOUT | ABSTAIN)
#
#   Intended callers (contract documented here, NOT wired — wiring them is their
#   own change, and a reader with no writer is worse than none, rule #223):
#     * ``sufficiency``  — a GATHER verdict names what evidence is missing; each
#       missing item is a field spec, and this decides ask vs assume vs proceed.
#     * ``deliberation`` — a DISAGREE outcome is a field whose value two engines
#       dispute; ASK is the escalation, PROCEED_WITHOUT the documented risk.
#     * ``adherence``    — at an amendment boundary, "is this change in scope?"
#       is a field; ABSTAIN is the safety-floor answer.
#   All callers land their output in the SAME ledger shape (``AssumptionRecord``
#   / ``Question``) so one surface renders assumptions from every engine.
#
# THE ASK IS PRICED. Whether to spend a question is not a vibe: it goes through
# ``economics.worth_calling(criticality, "human_question", budget_state)``, where
# ``HUMAN_QUESTION`` carries a deliberately large human-minutes cost ("human ask
# = precious"). ``economics`` is imported READ-ONLY and FAIL-OPEN: if it is
# absent or raises, we fall back to asking for critical fields and assuming for
# the rest, and say so in the verdict's reason. A missing cost model must not
# silently turn every question into a guess.
# ===========================================================================

RESOLVED = "RESOLVED"
SELF_ANSWERED = "SELF_ANSWERED"
NEEDS_ASK = "NEEDS_ASK"
PROCEED_WITHOUT = "PROCEED_WITHOUT"
ABSTAIN = "ABSTAIN"

FIELD_STATUSES = (RESOLVED, SELF_ANSWERED, NEEDS_ASK, PROCEED_WITHOUT, ABSTAIN)

# Provenance vocabulary. Every field value in a rendered spec cites exactly one
# of these, so "where did this come from" is never a guess.
SOURCE_USER = "user-said"
SOURCE_REPO = "repo-derived"
SOURCE_ASSUMED = "assumed"

# Criticality is the VOI proxy on economics' [0,1] scale: how much of the work
# is undetermined while this field is unknown. It is NOT a probability.
CRITICAL = 0.9      # without this there is no task at all
MATERIAL = 0.5      # changes the shape of the work
SHAPING = 0.15      # narrows the work; a recorded default is honest


@dataclass(frozen=True)
class FieldSpec:
    """One required field of a contract type.

    ``default``/``default_rationale`` are what makes a self-answer legitimate:
    an assumption with no stated rationale is a guess wearing a value's clothes.
    A spec with ``default=None`` has no honest fallback, so an exhausted budget
    gives ``PROCEED_WITHOUT`` (with the risk named), never an invented value.
    """
    name: str
    contract_type: str
    question: str
    criticality: float = MATERIAL
    resolver: str = "referent"      # referent | symptom | dimension | artifact | focus | none
    default: str | None = None
    default_rationale: str = ""

    @property
    def is_critical(self) -> bool:
        return self.criticality >= CRITICAL


def _fs(name, ctype, question, criticality=MATERIAL, resolver="none",
        default=None, rationale=""):
    return FieldSpec(name=name, contract_type=ctype, question=question,
                     criticality=criticality, resolver=resolver,
                     default=default, default_rationale=rationale)


FIELD_SPECS: dict = {
    "fix_bug": (
        _fs("referent_target", "fix_bug",
            "What exactly is misbehaving - which feature, page or code path?",
            CRITICAL, "referent"),
        _fs("symptom_behavior", "fix_bug",
            "What does it do wrong, and what should it do instead?",
            MATERIAL, "symptom"),
        _fs("scope", "fix_bug",
            "Should the repair stay inside that path, or may it touch callers?",
            SHAPING, "none",
            default="the named target and its direct callers only",
            rationale="narrowest blast radius that can still contain a real fix"),
    ),
    "build_new": (
        _fs("deliverable_kind", "build_new",
            "What kind of thing should this be - a page, an endpoint, an export, a setting?",
            CRITICAL, "artifact"),
        _fs("components", "build_new",
            "Which existing surfaces must it plug into?",
            MATERIAL, "referent"),
        _fs("acceptance_hint", "build_new",
            "How will we know it is done - what must a user be able to do?",
            SHAPING, "none",
            default="the described capability is reachable from the existing UI "
                    "and covered by at least one test",
            rationale="the platform's standing definition of done for a new surface"),
    ),
    "improve": (
        _fs("target", "improve",
            "Which surface should get better?", CRITICAL, "referent"),
        _fs("dimension", "improve",
            "Better along which axis - speed, readability, tone, freshness, cost?",
            MATERIAL, "dimension"),
        _fs("constraint", "improve",
            "Anything that must NOT change while improving it?",
            SHAPING, "none",
            default="no behaviour change beyond the named dimension",
            rationale="an improvement that alters behaviour is a rewrite, not an improvement"),
    ),
    "investigate": (
        _fs("subject", "investigate",
            "What should I investigate?", CRITICAL, "referent"),
        _fs("observed_signal", "investigate",
            "What did you observe that says something is off?",
            MATERIAL, "symptom"),
        _fs("scope", "investigate",
            "How far should I dig before reporting back?", SHAPING, "none",
            default="report the first causal chain that explains the signal, "
                    "without fixing it",
            rationale="investigation terminates at explanation; repair is a separate contract"),
    ),
    "question": (
        _fs("subject", "question",
            "What is the question about?", CRITICAL, "referent"),
        _fs("question_focus", "question",
            "What specifically do you want to know about it?",
            MATERIAL, "focus"),
    ),
}

# Value extractors -----------------------------------------------------------
_SYMPTOM_PATTERNS = (
    r"\bmark(?:ed|s)? [^,.;]{0,60}\bwrong\b",
    r"\bdouble[- ]?\w+\b", r"\btwice\b", r"\bduplicate[sd]?\b",
    r"\bcuts? off\b[^,.;]{0,40}", r"\bnever (?:seems to )?\w+[^,.;]{0,40}",
    r"\bkeeps? [a-z]+ing\b[^,.;]{0,40}",
    r"\bfail(?:ed|s|ing)?\b[^,.;]{0,40}",
    r"\bcrash(?:es|ed|ing)?\b", r"\bhas ?n'?t updated\b[^,.;]{0,40}",
    r"\bdoes ?n'?t (?:actually )?\w+[^,.;]{0,40}",
    r"\bstopped working\b", r"\btaking way too long\b", r"\btakes? forever\b",
    r"\bshowing [^,.;]{0,40}", r"\bshows? back up\b[^,.;]{0,30}",
    r"\bover and over\b", r"\bstill says pending\b",
    r"\bwrong [a-z ]{0,30}",
)

_DIMENSION_WORDS = {
    "faster": "speed", "slow": "speed", "speed": "speed", "forever": "speed",
    "readable": "readability", "unreadable": "readability",
    "clearer": "readability", "cleaner": "readability", "bland": "polish",
    "premium": "polish", "nicer": "polish", "ugly": "polish",
    "janky": "polish", "bigger": "layout", "smaller": "layout",
    "tiny": "layout", "current": "freshness", "fresh": "freshness",
    "nice": "tone", "tone": "tone", "annoying": "ergonomics",
    "remembers": "persistence", "memory": "persistence",
}

_ARTIFACT_WORDS = (
    "leaderboard", "dark mode", "export", "pdf", "dashboard", "report",
    "page", "endpoint", "api", "setting", "toggle", "integration", "app",
    "tool", "command", "module", "engine",
)


def _matched(patterns, text: str) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(0).strip()
    return ""


# ---------------------------------------------------------------------------
# Evidence context — what a field may resolve itself against
# ---------------------------------------------------------------------------
@dataclass
class EvidenceContext:
    """Everything ``resolve_or_ask`` is allowed to consult. Deliberately a plain
    record: any engine can build one without importing the front door's ladder.

    ``answers`` are explicit human answers already given (keyed by field name,
    plus ``TYPE_ANSWER_KEY``) — the only source that outranks the repo.
    """
    utterance: str = ""
    contract: Contract | None = None
    state: RepoState | None = None
    files: dict = field(default_factory=dict)
    answers: dict = field(default_factory=dict)
    notes: str = ""
    # v3: constraints the human STATED. These are not a field with a default —
    # see ``Constraint``. They are carried here so the ``constraint``/``scope``
    # field specs can be answered from what was actually said instead of being
    # overwritten by a recorded default that may assert the opposite.
    constraints: tuple = ()

    def bridge(self, limit: int = 5) -> list:
        """``behavior_bridge.bridge_candidates`` — the referent resolver the
        messy corpus actually needs: 'lessons double-grade' has no repo SYMBOL
        called that, but the functions that WRITE grade fields are exactly the
        thing being complained about. Fail-open: no files, no bridge, no crash.
        """
        if not self.files or not self.utterance:
            return []
        try:
            import behavior_bridge
            return list(behavior_bridge.bridge_candidates(
                self.utterance, files=self.files, limit=limit))
        except Exception:                                   # noqa: BLE001
            return []


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssumptionRecord:
    """A value nobody said out loud. ``vetoable`` is always True: an assumption
    the human cannot overturn is a decision made on their behalf."""
    field: str
    value: str
    source: str
    confidence_basis: str
    vetoable: bool = True

    def to_dict(self) -> dict:
        return dict(field=self.field, value=self.value, source=self.source,
                    confidence_basis=self.confidence_basis,
                    vetoable=self.vetoable)


@dataclass(frozen=True)
class FieldVerdict:
    """The result of one turn of the ladder for one field."""
    field: str
    status: str
    value: str = ""
    source: str = ""
    assumption: AssumptionRecord | None = None
    question: Question | None = None
    risk_note: str = ""
    reason: str = ""
    batched_ok: bool = True

    @property
    def is_open(self) -> bool:
        return self.status == NEEDS_ASK

    def to_dict(self) -> dict:
        return dict(field=self.field, status=self.status, value=self.value,
                    source=self.source, reason=self.reason,
                    risk_note=self.risk_note, batched_ok=self.batched_ok,
                    assumption=self.assumption.to_dict() if self.assumption else None,
                    question=self.question.to_dict() if self.question else None)


_NON_ANSWERS = re.compile(
    r"^\s*(?:idk|i don'?t know|dunno|no idea|whatever|unsure|n/?a|\?+)\s*$", re.I)


def _field_question(spec: FieldSpec) -> Question:
    return Question(text=spec.question, kind="field", options=(),
                    outcomes=((spec.name, f"{spec.contract_type}.{spec.name}"),),
                    reason=f"required field '{spec.name}' of a "
                           f"{spec.contract_type} contract is unresolved")


def _voi_permits_asking(spec: FieldSpec, budget_state) -> tuple:
    """(may_ask, reason). Fail-open on a missing/broken economics module: ask
    for critical fields, assume for the rest, and SAY that the price was not
    checked rather than pretending it was."""
    if budget_state is None:
        return True, ("no budget state supplied - asking is bounded by the "
                      "question cap, not by VOI")
    if getattr(budget_state, "hard_floor_blocked", False):
        return False, "kernel safety floor vetoes the call"
    try:
        from detreason import economics
    except Exception:                                       # noqa: BLE001
        return spec.is_critical, ("economics unavailable - fell back to "
                                  "'ask only for critical fields'")
    try:
        ok = economics.worth_calling(spec.criticality, "human_question",
                                     budget_state)
        return bool(ok), ("VOI of asking exceeds the price of a human question"
                          if ok else
                          "VOI of asking does not clear the price of a human "
                          "question (or the budget is exhausted)")
    except Exception as exc:                                # noqa: BLE001
        return spec.is_critical, (f"economics raised ({type(exc).__name__}) - "
                                  "fell back to 'ask only for critical fields'")


def _resolve_from_evidence(spec: FieldSpec, ctx: EvidenceContext) -> tuple:
    """(value, source, basis) or ('', '', '') — the SELF-ANSWERING half.

    Order is a precedence claim, not a preference: what the human said beats
    what the repo shows, which beats what we would otherwise assume.
    """
    text = ctx.utterance or ""

    # 1. explicit answer already given
    ans = str(ctx.answers.get(spec.name, "") or "").strip()
    if ans and not _NON_ANSWERS.match(ans):
        return ans, SOURCE_USER, "answered directly by the human"

    # 1b. v3: a STATED constraint answers the constraint/scope field outright.
    #     This closes a real hole: v2 gave those fields a SHAPING default, and a
    #     shaping default is applied without asking, so "make it faster WITHOUT
    #     changing what it costs" completed with constraint = "no behaviour
    #     change beyond the named dimension" — a different, weaker promise than
    #     the one the human made, recorded as an assumption nobody would query.
    if spec.name in ("constraint", "scope") and ctx.constraints:
        stated = "; ".join(c.text for c in ctx.constraints)
        kinds = ", ".join(sorted({c.kind for c in ctx.constraints}))
        return (stated, SOURCE_USER,
                f"stated in the ask as a hard constraint ({kinds})")

    # 2. from the utterance itself
    if spec.resolver == "referent" and ctx.contract is not None:
        for r in ctx.contract.referents:
            if r.is_resolved:
                return (f"{r.surface_phrase} -> {r.resolved[0].target}", SOURCE_USER,
                        f"named in the ask and resolved to a {r.resolved[0].kind}")
    if spec.resolver == "symptom":
        hit = _matched(_SYMPTOM_PATTERNS, text)
        if hit:
            return hit, SOURCE_USER, "symptom phrase quoted from the ask"
    if spec.resolver == "dimension":
        low = text.lower()
        for word, dim in sorted(_DIMENSION_WORDS.items()):
            if re.search(rf"\b{re.escape(word)}\b", low):
                return dim, SOURCE_USER, f"quality word '{word}' in the ask"
    if spec.resolver == "artifact":
        low = text.lower()
        for word in _ARTIFACT_WORDS:
            if word in low:
                return word, SOURCE_USER, f"artifact noun '{word}' in the ask"
    if spec.resolver == "focus":
        m = re.search(r"\b(?:what|which|how|why|does|do|is|are|can|will|should)\b[^?.]{3,120}",
                      text, flags=re.I)
        if m:
            return m.group(0).strip(), SOURCE_USER, "interrogative clause from the ask"

    # 3. from the repo, via the behaviour bridge. This is the resolver the messy
    #    corpus needs: the complaint names a BEHAVIOUR, the repo names FUNCTIONS,
    #    and the bridge is the only thing that connects the two.
    if spec.resolver in ("referent", "artifact"):
        cands = ctx.bridge(limit=3)
        if cands:
            c = cands[0]
            return (f"{c.path}::{c.fn}", SOURCE_REPO,
                    f"behaviour bridge ({c.bridge_kind}): {c.evidence}")

    # 4. a caller with STATE but no contract (any engine outside the front door
    #    calling this primitive directly) still gets referent resolution.
    if spec.resolver == "referent" and ctx.contract is None and ctx.state is not None:
        for phrase in extract_referents(text, domain_vocab=ctx.state.vocab):
            r = resolve_referent(phrase, ctx.state)
            if r.is_resolved:
                return (r.resolved[0].target, SOURCE_REPO,
                        f"'{phrase}' resolved against supplied state as a "
                        f"{r.resolved[0].kind}")

    # 5. an UNRESOLVED surface phrase is still what the human said. It is weaker
    #    than a grounded referent, so it is offered only after the bridge.
    if spec.resolver == "referent" and ctx.contract is not None:
        for r in ctx.contract.referents:
            return (r.surface_phrase, SOURCE_USER,
                    "named in the ask (surface phrase, unresolved against state)")

    return "", "", ""


def resolve_or_ask(field_spec: FieldSpec, evidence_context: EvidenceContext,
                   budget_state=None) -> FieldVerdict:
    """THE KERNEL-WIDE QUESTION PRIMITIVE. Pure, deterministic, no I/O.

    The ladder, in order:
      1. the human already said it                 -> RESOLVED   (user-said)
      2. the utterance or the repo says it         -> RESOLVED / SELF_ANSWERED
      3. asking is worth its price                 -> NEEDS_ASK  (a Question)
      4. not worth it, a default exists            -> SELF_ANSWERED (+ ledger)
      5. not worth it, no honest default           -> PROCEED_WITHOUT (+ risk)
      6. a kernel safety floor vetoes the call     -> ABSTAIN

    There is deliberately no branch that produces a value without recording
    where it came from. Steps 4 and 5 are the difference between an assumption
    and a guess: both proceed, only one is auditable and vetoable.
    """
    spec, ctx = field_spec, evidence_context
    value, source, basis = _resolve_from_evidence(spec, ctx)

    if value and source == SOURCE_USER:
        return FieldVerdict(field=spec.name, status=RESOLVED, value=value,
                            source=source, reason=basis)
    if value and source == SOURCE_REPO:
        rec = AssumptionRecord(field=spec.name, value=value, source=SOURCE_REPO,
                               confidence_basis=basis)
        return FieldVerdict(field=spec.name, status=SELF_ANSWERED, value=value,
                            source=SOURCE_REPO, assumption=rec, reason=basis)

    # The human was asked and declined: that IS an answer. Re-asking would be
    # both rude and non-terminating.
    raw = ctx.answers.get(spec.name, None)
    declined = spec.name in ctx.answers and (
        not str(raw or "").strip() or bool(_NON_ANSWERS.match(str(raw))))

    may_ask, why = _voi_permits_asking(spec, budget_state)
    if may_ask and not declined:
        return FieldVerdict(field=spec.name, status=NEEDS_ASK,
                            question=_field_question(spec), reason=why)

    if spec.default is not None:
        rec = AssumptionRecord(
            field=spec.name, value=spec.default, source=SOURCE_ASSUMED,
            confidence_basis=(spec.default_rationale or "declared default") +
            ("; the human was asked and declined to specify" if declined
             else f"; not asked because: {why}"))
        return FieldVerdict(field=spec.name, status=SELF_ANSWERED,
                            value=spec.default, source=SOURCE_ASSUMED,
                            assumption=rec, reason=why)

    if budget_state is not None and getattr(budget_state, "hard_floor_blocked", False):
        return FieldVerdict(
            field=spec.name, status=ABSTAIN, reason=why,
            risk_note=f"'{spec.name}' is unknown, no honest default exists, and a "
                      "kernel safety floor forbids asking - this contract cannot "
                      "be completed")
    return FieldVerdict(
        field=spec.name, status=PROCEED_WITHOUT, reason=why,
        risk_note=(f"proceeding with '{spec.name}' UNKNOWN - no default is honest "
                   "for this field and the question was "
                   + ("declined" if declined else "not affordable")
                   + f"; anything downstream that depends on {spec.name} is "
                     "unvalidated"))


# ---------------------------------------------------------------------------
# complete() — the whole contract, field by field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompletedContract:
    contract: Contract
    verdicts: tuple = ()
    utterance: str = ""

    @property
    def fields(self) -> dict:
        return {v.field: v for v in self.verdicts}

    @property
    def assumption_ledger(self) -> tuple:
        return tuple(v.assumption for v in self.verdicts if v.assumption)

    @property
    def open_asks(self) -> tuple:
        return tuple(v for v in self.verdicts if v.status == NEEDS_ASK)

    @property
    def risks(self) -> tuple:
        return tuple(v.risk_note for v in self.verdicts if v.risk_note)

    @property
    def blocked_fields(self) -> tuple:
        """CRITICAL fields that ended the ladder with no value at all.

        v3 fix for a real completion escape in v2: ``PROCEED_WITHOUT`` and
        ``ABSTAIN`` are terminal statuses that carry NO value, and ``is_complete``
        only looked at open asks — so a budget-exhausted ``referent_target`` (the
        field that says WHAT to act on) made the contract report itself complete
        with nothing to act on. Downstream mutation code reading ``is_complete``
        would have proceeded. The risk note existed; nothing was gated on it.
        """
        return tuple(v for v in self.verdicts
                     if v.status in (PROCEED_WITHOUT, ABSTAIN)
                     and not v.value
                     and _spec_for(self.contract.type, v.field) is not None
                     and _spec_for(self.contract.type, v.field).is_critical)

    @property
    def is_complete(self) -> bool:
        """No open questions AND nothing critical left unknown. Both halves are
        required: a contract with an unknown critical field is not complete, it
        is merely finished being asked about."""
        return not self.open_asks and not self.blocked_fields

    def to_dict(self) -> dict:
        return dict(contract=self.contract.to_dict(), utterance=self.utterance,
                    fields={k: v.to_dict() for k, v in self.fields.items()},
                    assumptions=[a.to_dict() for a in self.assumption_ledger],
                    open_asks=[v.question.to_dict() for v in self.open_asks],
                    risks=list(self.risks), complete=self.is_complete)


def complete(contract: Contract, evidence_context: EvidenceContext | None = None,
             budget_state=None) -> CompletedContract:
    """Walk every required field of ``contract.type`` through the ladder."""
    ctx = evidence_context or EvidenceContext(contract=contract)
    if ctx.contract is None:
        ctx.contract = contract
    specs = FIELD_SPECS.get(contract.type, ())
    verdicts = tuple(resolve_or_ask(s, ctx, budget_state) for s in specs)
    return CompletedContract(contract=contract, verdicts=verdicts,
                             utterance=ctx.utterance)


# ---------------------------------------------------------------------------
# render_spec — the "brought up to par" ask
# ---------------------------------------------------------------------------
def render_spec(completed: CompletedContract) -> str:
    """The elevated ask: same meaning as the utterance, nothing invented.

    Every line cites its source, so a reader can audit the difference between
    what was said, what was measured, and what was assumed WITHOUT reading this
    module. Assumptions get their own block and are marked vetoable — an
    assumption you have to hunt for is a silent one.
    """
    c = completed.contract
    lines = [f"CONTRACT: {c.type}"]
    if completed.utterance:
        lines.append(f"  from: {completed.utterance!r}")
    lines.append("  evidence for this reading:")
    for b in (c.confidence_basis or ("(none - this reading is unevidenced)",)):
        lines.append(f"    - {b}")
    lines.append("  fields:")
    for v in completed.verdicts:
        if v.status in (RESOLVED, SELF_ANSWERED):
            lines.append(f"    {v.field}: {v.value}   [{v.source}]")
        elif v.status == NEEDS_ASK:
            lines.append(f"    {v.field}: <OPEN - asked>")
        else:
            lines.append(f"    {v.field}: <{v.status}>")
    ledger = completed.assumption_ledger
    lines.append(f"  assumptions ({len(ledger)}) - every one vetoable:")
    for a in ledger:
        lines.append(f"    - {a.field} = {a.value}")
        lines.append(f"        source: {a.source}; basis: {a.confidence_basis}")
        lines.append(f"        vetoable: {a.vetoable}")
    if not ledger:
        lines.append("    (none - every field came from what was actually said "
                     "or measured)")
    opens = completed.open_asks
    lines.append(f"  open questions ({len(opens)}):")
    for v in opens:
        lines.append(f"    - {v.question.text}")
    if not opens:
        lines.append("    (none)")
    if completed.risks:
        lines.append("  proceeding-without risks:")
        for r in completed.risks:
            lines.append(f"    ! {r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The completion loop
# ---------------------------------------------------------------------------
MAX_QUESTIONS = 3


@dataclass(frozen=True)
class CompletionResult:
    utterance: str
    interpretation: InterpretationResult
    completed: CompletedContract
    rounds: int
    questions_asked: tuple = ()
    answers: dict = field(default_factory=dict)
    termination: str = ""
    first_pass: InterpretationResult | None = None

    @property
    def clarified(self) -> bool:
        return bool(self.questions_asked)

    def to_dict(self) -> dict:
        return dict(utterance=self.utterance, rounds=self.rounds,
                    termination=self.termination,
                    questions_asked=[q.to_dict() for q in self.questions_asked],
                    answers=dict(self.answers),
                    interpretation=self.interpretation.to_dict(),
                    completed=self.completed.to_dict())


def interpret_complete(utterance: str, answerer=None, *, repo_root=None,
                       extra_context: dict | None = None,
                       max_questions: int = MAX_QUESTIONS,
                       budget_state=None) -> CompletionResult:
    """interpret -> complete -> (ask, incorporate, re-complete)* -> contract.

    ``answerer`` is any callable ``(questions: list[Question]) -> dict`` mapping
    an answer key (``TYPE_ANSWER_KEY`` or a field name) to a string. Questions
    are BATCHED: one round asks everything still open, never one at a time.

    TERMINATION IS PROVABLE, by two independent bounds:
      (a) the loop asks at most ``max_questions`` questions in total, and every
          round that continues asks at least one; and
      (b) a round whose answers are all unusable exits immediately, after
          RECORDING the refusals so the fields do not come back as open asks.
    So an adversarial answerer that returns junk, the same answer twice, or
    nothing at all cannot spin: (b) catches it on the first useless round and
    (a) catches everything else. With ``answerer=None`` the loop runs exactly
    once and returns the contract with its open asks intact.
    """
    ctx = dict(extra_context or {})
    answers: dict = dict(ctx.get("clarifications") or {})
    asked: list = []
    rounds = 0
    first = None
    termination = "no answerer - returned with open asks"
    state = ctx.get("state") or build_state(repo_root=repo_root,
                                            files=ctx.get("files"),
                                            activity=ctx.get("activity"))

    def _pass():
        res = interpret(utterance, repo_root=repo_root,
                        extra_context={**ctx, "clarifications": answers})
        ev = EvidenceContext(utterance=utterance, contract=res.top, state=state,
                             files=dict(ctx.get("files") or state.files),
                             answers=dict(answers),
                             constraints=extract_constraints(utterance))
        done = (complete(res.top, ev, budget_state) if res.top
                else CompletedContract(contract=Contract(type="question", score=0.0),
                                       utterance=utterance))
        return res, done

    while True:
        rounds += 1
        res, done = _pass()
        if first is None:
            first = res

        pending: list = []
        # The type question outranks every field question: a perfectly specified
        # field of the WRONG contract is a good answer to the wrong ask.
        # An "open" question (mutating contract, nothing to act on) means the
        # reading rests on cue words alone with no object behind them — the TYPE
        # is as unsettled as the target, so it is worth a question too. A
        # "referent" question is different: there the type is evidenced and only
        # the target is contested, so we do NOT spend a question re-asking it.
        if (res.verdict in ("ASK_ONE", "AMBIGUOUS") and res.question is not None
                and res.question.kind in ("contract_type", "open")
                and TYPE_ANSWER_KEY not in answers):
            tq = (res.question if res.question.kind == "contract_type"
                  else _type_question(res.contracts,
                                      reason="the ask names an action but no "
                                             "object, so the reading is unsettled"))
            pending.append((TYPE_ANSWER_KEY, tq))
        pending += [(v.field, v.question) for v in done.open_asks]

        if not pending:
            termination = "every required field resolved"
            break
        if answerer is None:
            termination = "no answerer - returned with open asks"
            break
        if len(asked) >= max_questions:
            termination = f"question budget exhausted ({max_questions})"
            break

        # BATCHING RULE. Field questions batch freely with each other — asking
        # "what is broken, what does it do wrong, how far may I go" in one
        # breath is what a good engineer does. The TYPE question is the one
        # exception and is asked ALONE: it decides WHICH fields exist, so
        # batching it with the current guess's fields spends the budget on
        # questions that may belong to the wrong contract entirely.
        if pending[0][0] == TYPE_ANSWER_KEY:
            batch = pending[:1]
        else:
            batch = pending[:max_questions - len(asked)]
        try:
            got = answerer([q for _, q in batch]) or {}
        except Exception:                                   # noqa: BLE001
            termination = "answerer failed - stopped rather than looped"
            break
        asked += [q for _, q in batch]

        new = {}
        for key, q in batch:
            # An unanswered/useless key is RECORDED as an explicit non-answer so
            # the next round does not ask it again. This is bound (b).
            new[key] = str(got.get(key, got.get(q.text, "")) or "").strip()
        usable = [v for v in new.values() if v and not _NON_ANSWERS.match(v)]
        answers.update(new)
        if not usable:
            termination = "a full round produced no usable answer"
            rounds += 1
            res, done = _pass()
            break

    return CompletionResult(utterance=utterance, interpretation=res,
                            completed=done, rounds=rounds,
                            questions_asked=tuple(asked), answers=answers,
                            termination=termination, first_pass=first)


# ---------------------------------------------------------------------------
# Multi-turn scoring over the messy corpus
# ---------------------------------------------------------------------------
MESSY_CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "frontdoor_eval_messy.jsonl"


def load_messy_corpus(path=MESSY_CORPUS_PATH) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def scripted_answerer(row: dict):
    """A human simulator built from the row's OWN labels — nothing else.

    It answers only what a real asker could answer about their own ask: the
    contract type they meant, and the thing they meant by it. It never supplies
    a symptom, a scope or an acceptance criterion the row does not carry, so a
    'clarified pass' cannot be manufactured by feeding in the answer key.
    """
    ref = row.get("expected_key_referent")
    ctype = row.get("expected_type")

    def answer(questions):
        out = {}
        for q in questions:
            if q.kind == "contract_type":
                # A row with expected_type=None is an ask its own author could
                # not type either: the simulated human shrugs, honestly.
                out[TYPE_ANSWER_KEY] = ctype or "idk"
            elif q.kind == "field" and q.outcomes:
                fieldname = q.outcomes[0][0]
                out[fieldname] = (ref or "idk") if fieldname in (
                    "referent_target", "target", "subject", "components",
                    "deliverable_kind") else "idk"
            else:
                out[q.kind] = ref or "idk"
        return out
    return answer


ONE_SHOT = "one_shot_pass"
CLARIFIED = "clarified_pass"
FAIL = "fail"


def score_messy_row(row: dict, extra_context: dict | None = None,
                    max_questions: int = MAX_QUESTIONS) -> dict:
    """Score ONE messy ask under the three-outcome philosophy.

    ``one_shot_pass``  — right type AND acceptable verdict, first try.
    ``clarified_pass`` — it did NOT close falsely (it asked), and after a bounded
                         clarification exchange it landed on the right type.
                         Asking and then being right is a PASS: it is what a good
                         engineer does with a vague ask.
    ``fail``           — anything else, and in particular any confident PROCEED
                         on the wrong type. A false closure is NEVER rescued by
                         clarification, because none was offered.
    """
    ctx = dict(extra_context or {})
    expected = row["expected_type"]
    first = interpret(row["ask"], extra_context=ctx)
    got = first.top.type if first.top else None
    type_ok = (got == expected) if expected is not None else True
    verdict_ok = first.verdict in row["acceptable_verdicts"]

    if type_ok and verdict_ok:
        return dict(ask=row["ask"], expected=expected, outcome=ONE_SHOT,
                    first_type=got, first_verdict=first.verdict,
                    final_type=got, questions=0, note="")

    # Did it ask, or did it close? Only an ask earns a second chance.
    if first.verdict == "PROCEED":
        return dict(ask=row["ask"], expected=expected, outcome=FAIL,
                    first_type=got, first_verdict=first.verdict, final_type=got,
                    questions=0,
                    note="closed confidently on the wrong reading - no "
                         "clarification was offered, so none may rescue it")

    out = interpret_complete(row["ask"], answerer=scripted_answerer(row),
                             extra_context=ctx, max_questions=max_questions)
    final = out.interpretation.top.type if out.interpretation.top else None
    if expected is not None and final == expected:
        return dict(ask=row["ask"], expected=expected, outcome=CLARIFIED,
                    first_type=got, first_verdict=first.verdict,
                    final_type=final, questions=len(out.questions_asked),
                    note=out.termination)
    return dict(ask=row["ask"], expected=expected, outcome=FAIL,
                first_type=got, first_verdict=first.verdict, final_type=final,
                questions=len(out.questions_asked),
                note=f"clarification did not converge ({out.termination})")


def evaluate_messy(rows=None, extra_context: dict | None = None,
                   max_questions: int = MAX_QUESTIONS) -> dict:
    """The three-rate table. All three are reported and none is hidden inside a
    single headline number, because they mean different things about the system:
    one-shot is competence, clarified is honesty, fail is damage."""
    rows = rows if rows is not None else load_messy_corpus()
    detail = [score_messy_row(r, extra_context, max_questions) for r in rows]
    n = len(detail) or 1
    counts = {k: sum(1 for d in detail if d["outcome"] == k)
              for k in (ONE_SHOT, CLARIFIED, FAIL)}
    return dict(
        n=len(detail), counts=counts,
        one_shot_rate=round(counts[ONE_SHOT] / n, 4),
        clarified_rate=round(counts[CLARIFIED] / n, 4),
        fail_rate=round(counts[FAIL] / n, 4),
        pass_rate=round((counts[ONE_SHOT] + counts[CLARIFIED]) / n, 4),
        detail=detail,
    )


# ===========================================================================
# FRONT DOOR v3 — the ask that is not ONE ask
#
# v2 compiles an utterance into ONE typed contract with a flat field vector.
# That schema is adequate right up until the ask gets complicated, and then it
# fails SILENTLY — which is the worst way for a front door to fail, because the
# dropped half never appears anywhere downstream to be noticed (rules #163,
# #181c, #223: the caller gets a plausible answer and believes it worked).
#
# Four things v2 cannot represent, each measured against the frozen v3 corpus:
#
#   1. COMPOUNDS.    "fix X and while you're there add Y" is two mutations. v2
#                    types it fix_bug and Y is gone. No warning, no residue.
#   2. CONSTRAINTS.  "make it faster without changing what it costs" carries a
#                    hard prohibition. v2's ``constraint`` field is a SHAPING
#                    field with a recorded default, so the stated prohibition is
#                    overwritten by the default that says the opposite.
#   3. PRIOR WORK.   "same issue as yesterday, but only on mobile" points at a
#                    contract we do not hold. Every content word is a stopword;
#                    v2 sees an empty ask and types it on cue words alone.
#   4. COUNTER-PROPOSALS. "don't build a blockchain thing, just give me a log"
#                    contains a REJECTED option. v2 keeps the rejected noun as a
#                    referent and loses the rejection — the one part that must
#                    never be forgotten.
#
# v3 adds a STRUCTURAL LAYER above the v2 contract rather than a sixth contract
# type. That is deliberate: ``compound`` is not a kind of work, it is a
# statement about how many pieces of work were asked for, and modelling it as a
# type would make every downstream consumer of ``Contract.type`` handle a value
# that has no obligations, no fields and no route. The v2 ladder is untouched
# and every v2 caller keeps working; ``decompose()`` is the new entry point.
#
# ORDER OF OPERATIONS MATTERS AND IS A RULE, NOT A CONVENIENCE:
#   constraints are extracted FIRST and their spans are MASKED before clause
#   splitting. "dry run first and show me what it would delete" contains the
#   word "and" between two verbs, but the whole phrase is ONE approval gate;
#   splitting before masking manufactures a second contract out of a condition.
# ===========================================================================

MODE_SINGLE = "single"
MODE_COMPOUND = "compound"
MODE_COUNTER_PROPOSAL = "counter_proposal"
BUNDLE_MODES = (MODE_SINGLE, MODE_COMPOUND, MODE_COUNTER_PROPOSAL)

# Edge relations between children. ``must_precede`` is stated ordering ("then",
# "first"); ``parallel`` is the honest default for an unordered "and".
EDGE_MUST_PRECEDE = "must_precede"
EDGE_PARALLEL = "parallel"

CONSTRAINT_KINDS = ("prohibition", "preservation", "platform_scope",
                    "threshold", "approval_gate", "forbidden_action")

# --- constraints -----------------------------------------------------------
# Each entry is (kind, regex, label). The regex must match the WHOLE constraint
# clause, because its span is masked out of the utterance before clause
# splitting; a regex that matches only the trigger word would leave the rest of
# the condition free to be mistaken for a second ask.
_CONSTRAINT_PATTERNS = (
    ("forbidden_action",
     r"\b(?:do ?n'?t|dont|do not|never|please don'?t)\s+"
     r"(?:fix|change|touch|build|add|create|rewrite|refactor|delete|remove|ship)\b"
     r"[^,.;?]{0,60}",
     "explicitly forbidden action"),
    ("forbidden_action",
     r"\bnot a (?:whole )?new \w+\b[^,.;?]{0,40}",
     "explicitly rejected artifact class"),
    ("prohibition",
     r"\bwithout\s+(?:chang\w+|break\w+|touch\w+|affect\w+|alter\w+|regress\w+)\b"
     r"[^,.;?]{0,80}",
     "prohibition ('without changing/breaking ...')"),
    ("prohibition",
     r"\b(?:must never|should never|must not|cannot ever)\b[^,.;?]{0,80}",
     "hard prohibition"),
    ("preservation",
     r"\b(?:must keep|has to keep|keep)\s+\w+[^,.;?]{0,40}"
     r"\b(?:working|as (?:they|it) (?:do|does|is|are) now|exactly as|unchanged)\b"
     r"[^,.;?]{0,40}",
     "preservation of existing behaviour"),
    ("preservation",
     r"\b(?:same|identical) (?:output|behaviou?r|results?|cost|spend)\b[^,.;?]{0,40}",
     "output/cost must not move"),
    ("platform_scope",
     r"\bonly (?:for|on|in|when)\b[^,.;?]{0,50}",
     "scope restricted to a named context"),
    ("platform_scope",
     r"\b(?:on |in )?(?:ipad|iphone|mobile|desktop|tablet|android|web)\s+only\b",
     "platform restricted"),
    ("platform_scope",
     r"\b(?:ipad|iphone|mobile|desktop|tablet|android|web) is fine\b",
     "contrastive platform scope (the other side is explicitly fine)"),
    ("platform_scope",
     r"\bjust for\b[^,.;?]{0,40}",
     "scope restricted ('just for ...')"),
    ("threshold",
     r"\b(?:under|below|less than|no more than|within)\s+\d+\s*"
     r"(?:s|ms|sec|secs|second|seconds|minutes?|%)\b[^,.;?]{0,30}",
     "numeric acceptance threshold"),
    ("threshold",
     r"\bp\d{2}\b[^,.;?]{0,30}",
     "percentile latency target"),
    ("approval_gate",
     r"\bdry[- ]?run\b[^,.;?]{0,80}",
     "dry-run before the destructive action"),
    ("approval_gate",
     r"\bflag\b[^,.;?]{0,60}\bbefore you\b[^,.;?]{0,40}",
     "flag-before-touch gate"),
    ("approval_gate",
     r"\b(?:ask|check with|show) me (?:first|before)\b[^,.;?]{0,40}",
     "human approval required before acting"),
    ("approval_gate",
     r"\bapprove (?:those|them|it|these) separately\b",
     "separate approval required"),
    ("prohibition",
     r"\b(?:but|and) (?:do ?n'?t|dont|do not) \w+[^,.;?]{0,60}",
     "prohibition attached to the request"),
)


@dataclass(frozen=True)
class Constraint:
    """A stated limit on HOW the work may be done.

    A constraint is not a field with a default. v2 modelled "anything that must
    NOT change?" as a SHAPING field carrying a recorded default, which means a
    stated prohibition and an unstated one produce the SAME contract — the
    default silently overwrites what the human actually said. A ``Constraint``
    is only ever created from a span the human wrote, and it has no default.
    """
    kind: str
    text: str
    label: str
    span: tuple = ()

    def to_dict(self) -> dict:
        return dict(kind=self.kind, text=self.text, label=self.label,
                    span=list(self.span))


def extract_constraints(utterance: str) -> tuple:
    """Every stated limit, de-overlapped, in order of appearance.

    Overlap removal keeps the longest match starting earliest and drops any
    later match contained inside it, so "must never include anything from the
    private memory store" is one prohibition rather than a prohibition plus a
    fragment of itself.
    """
    text = utterance or ""
    hits: list = []
    for kind, pattern, label in _CONSTRAINT_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.I):
            hits.append((m.start(), m.end(), kind, m.group(0).strip(), label))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0]), h[2]))
    out: list = []
    taken: list = []
    for start, end, kind, body, label in hits:
        if any(start >= s and end <= e for s, e in taken):
            continue
        taken.append((start, end))
        out.append(Constraint(kind=kind, text=body, label=label,
                              span=(start, end)))
    return tuple(out)


def _mask_spans(text: str, spans) -> str:
    """Blank out constraint spans so they cannot be read as a second ask.

    Same length, so every other span offset in this module stays valid.
    """
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


# --- prior-work / context pointers -----------------------------------------
_CONTEXT_PATTERNS = (
    ("temporal",
     r"\bsame (?:issue|thing|bug|problem|one) as (?:yesterday|last \w+|before|earlier)\b",
     "points at a prior report we do not hold"),
    ("temporal",
     r"\b(?:whatever|what) you (?:changed|did|shipped|deployed) "
     r"(?:this morning|yesterday|last night|earlier|today)\b",
     "points at our own prior action"),
    ("prior_change",
     r"\bthe same (?:fix|change|thing) (?:you|we) did\b[^,.;?]{0,40}",
     "uses a prior CHANGE as the specification"),
    ("anaphoric",
     r"\bthat thing (?:we|you|i) (?:talked about|discussed|mentioned)\b[^,.;?]{0,30}",
     "pure anaphora — the referent lives only in a prior conversation"),
    ("anaphoric",
     r"\b(?:like|as) we (?:discussed|talked about|agreed)\b",
     "defers to an unheld prior agreement"),
    ("anaphoric",
     r"\bpick (?:it|that) back up\b",
     "resumes unnamed prior work"),
)


@dataclass(frozen=True)
class ContextPointer:
    """A reference to work, a report or a conversation the front door does not
    hold. Its whole purpose is to make the gap LOUD: an ask whose specification
    lives elsewhere cannot be completed from the utterance, however confident
    the cue stack is."""
    kind: str
    text: str
    note: str

    def to_dict(self) -> dict:
        return dict(kind=self.kind, text=self.text, note=self.note)


def extract_context_pointers(utterance: str) -> tuple:
    text = utterance or ""
    out, seen = [], set()
    for kind, pattern, note in _CONTEXT_PATTERNS:
        m = re.search(pattern, text, flags=re.I)
        if m and m.group(0).lower() not in seen:
            seen.add(m.group(0).lower())
            out.append(ContextPointer(kind=kind, text=m.group(0).strip(),
                                      note=note))
    return tuple(out)


# --- counter-proposals ------------------------------------------------------
# (regex, frame label). The rejected half is the reason this exists: an
# alternative the human has already ruled out must be recorded, or the next
# planner proposes it again and burns the human's patience proving the same
# "no" twice.
_COUNTER_PATTERNS = (
    (r"\binstead of\s+(?P<rejected>[^,.;?]{3,80}?)\s*[,.]?\s*"
     r"(?:can (?:we|you)|lets|let's|just|please)\s*(?P<proposed>[^.;?]{3,120})",
     "'instead of X, Y'"),
    (r"\brather than\s+(?P<rejected>[^,.;?]{3,80}?)\s*[,.]?\s*"
     r"(?:lets|let's|just|can we|please)\s+(?P<proposed>[^.;?]{3,120})",
     "'rather than X, Y'"),
    (r"\b(?:do ?n'?t|dont|do not)\s+(?P<rejected>[a-z][^,.;?]{3,80}?)\s*[,.]\s*"
     r"(?:just|instead|please)\s*(?P<proposed>[^.;?]{3,120})",
     "'don't X, Y'"),
    (r"\b(?:we should ?n'?t|shouldnt)\s+(?:just\s+)?"
     r"(?P<rejected>[^,.;?]{3,80}?)\s*[,.]\s*(?P<proposed>[^.;?]{3,120})",
     "'we shouldn't X, Y'"),
    (r"\bnot a (?:whole )?(?P<rejected>new \w+)\b[^,.;?]{0,30}[,.]\s*"
     r"(?P<proposed>[^.;?]{3,120})",
     "'not a new X, Y'"),
)


@dataclass(frozen=True)
class CounterProposal:
    rejected: str
    proposed: str
    frame: str

    def to_dict(self) -> dict:
        return dict(rejected=self.rejected, proposed=self.proposed,
                    frame=self.frame)


def extract_counter_proposal(utterance: str):
    """The FIRST matching rejection frame, or None.

    Only one is returned on purpose: a second "instead of" inside the proposed
    half is part of the proposal, not a second rejection, and stacking them
    produces a rejected-list nobody can act on.
    """
    text = utterance or ""
    for pattern, frame in _COUNTER_PATTERNS:
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue
        rejected = (m.group("rejected") or "").strip(" ,.")
        proposed = (m.group("proposed") or "").strip(" ,.")
        if rejected and len(proposed.split()) >= 2:
            return CounterProposal(rejected=rejected, proposed=proposed,
                                   frame=frame)
    return None


# --- clause splitting -------------------------------------------------------
# ADDITIVE connectives explicitly mark "here comes another item"; PLAIN ones are
# ordinary coordination and may just be joining two nouns. The distinction is
# the whole splitting rule: "and also" splits on the marker's own authority, a
# bare "and" has to earn it by carrying an action predicate on BOTH sides.
_ADDITIVE_CONNECTIVES = (
    (r",?\s+(?:and\s+)?also,?\s+", EDGE_PARALLEL, "additive marker 'also'"),
    (r",?\s+while (?:you'?re|your|we'?re|were) (?:in there|at it),?\s+(?:and\s+)?",
     EDGE_PARALLEL, "'while you're in there'"),
    (r",?\s+plus\s+", EDGE_PARALLEL, "additive marker 'plus'"),
)
_ORDERED_CONNECTIVES = (
    (r",?\s+(?:and\s+)?then\s+", EDGE_MUST_PRECEDE, "explicit ordering 'then'"),
    (r",?\s+after (?:that|which),?\s+", EDGE_MUST_PRECEDE, "explicit ordering"),
)
_PLAIN_CONNECTIVES = (
    (r",\s+and\s+", EDGE_PARALLEL, "coordination ', and'"),
    (r"\s+and\s+", EDGE_PARALLEL, "coordination 'and'"),
    (r";\s+", EDGE_PARALLEL, "sentence break ';'"),
)

# An "action predicate" is a verb that asks for WORK. Its presence on BOTH sides
# is what makes a plain "and" a real compound rather than a coordinated noun.
_ACTION_VERBS = (
    r"\b(?:fix|add|build|create|make|write|update|change|remove|delete|export|"
    r"import|investigate|figure out|check|clean up|speed up|improve|polish|"
    r"rename|migrate|wire up|stand up|show|give|refactor|replace|support)\b"
)


def _has_content(text: str) -> bool:
    """At least one content-bearing token — a clause of pure function words is
    not a second ask, it is the tail of the first one."""
    return any(w not in _STOPWORDS and len(w) > 2 for w in _words(text))


def _has_action(text: str) -> bool:
    return bool(re.search(_ACTION_VERBS, text or "", flags=re.I))


@dataclass(frozen=True)
class Clause:
    text: str
    connective: str = ""
    relation: str = EDGE_PARALLEL
    ordering_hint: str = ""      # "first" seen on THIS clause

    def to_dict(self) -> dict:
        return dict(text=self.text, connective=self.connective,
                    relation=self.relation, ordering_hint=self.ordering_hint)


_ORDER_FIRST = re.compile(r"\b(?:first|before (?:that|you|anything))\b", re.I)


def split_clauses(utterance: str, constraints=None) -> tuple:
    """Utterance -> clauses, with constraint spans masked out FIRST.

    Returns a single clause when nothing splits. The masking step is why
    "clean up the stale lesson rows, dry run first and show me what it would
    delete" stays ONE ask: the ``and`` sits inside an approval-gate span, and a
    condition on the work is not a second piece of work.
    """
    text = utterance or ""
    cons = extract_constraints(text) if constraints is None else tuple(constraints)
    masked = _mask_spans(text, [c.span for c in cons if c.span])

    def _first_split(patterns):
        best = None
        for pattern, relation, label in patterns:
            m = re.search(pattern, masked, flags=re.I)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, relation, label)
        return best

    for group, must_earn in ((_ADDITIVE_CONNECTIVES, False),
                             (_ORDERED_CONNECTIVES, False),
                             (_PLAIN_CONNECTIVES, True)):
        found = _first_split(group)
        if not found:
            continue
        m, relation, label = found
        left, right = text[:m.start()].strip(" ,;"), text[m.end():].strip(" ,;")
        if not (_has_content(left) and _has_content(right)):
            continue
        if must_earn and not (_has_action(left) and _has_action(right)):
            # A bare "and" joining a verb to a noun phrase is one deliverable
            # with two parts ("a dark mode toggle and a font size slider"), not
            # two contracts. Refusing to split here is what keeps the
            # false-split rate at zero on the frozen corpus.
            continue
        return (Clause(text=left, connective="", relation=EDGE_PARALLEL,
                       ordering_hint="first" if _ORDER_FIRST.search(left) else ""),
                Clause(text=right, connective=label, relation=relation,
                       ordering_hint="first" if _ORDER_FIRST.search(right) else ""))
    return (Clause(text=text.strip()),)


# --- the bundle -------------------------------------------------------------
@dataclass(frozen=True)
class ChildContract:
    id: str
    contract: Contract
    clause: Clause
    interpretation: InterpretationResult | None = None

    def to_dict(self) -> dict:
        return dict(id=self.id, clause=self.clause.to_dict(),
                    contract=self.contract.to_dict())


@dataclass(frozen=True)
class BundleEdge:
    src: str
    dst: str
    relation: str
    reason: str = ""

    def to_dict(self) -> dict:
        return dict(src=self.src, dst=self.dst, relation=self.relation,
                    reason=self.reason)


@dataclass(frozen=True)
class RequestBundle:
    """What was actually asked for: one or more contracts, plus everything that
    is true of ALL of them.

    ``constraints``, ``context_pointers`` and ``counter_proposal`` live on the
    BUNDLE, not on a child, because they are properties of the REQUEST. A
    prohibition stated once at the end ("...without changing the api") governs
    every child, and copying it into one child would quietly exempt the others.
    """
    utterance: str
    mode: str
    children: tuple = ()
    edges: tuple = ()
    constraints: tuple = ()
    context_pointers: tuple = ()
    counter_proposal: CounterProposal | None = None
    split_reason: str = ""
    verdict: str = ""
    verdict_reason: str = ""
    question: Question | None = None

    @property
    def primary(self):
        return self.children[0] if self.children else None

    @property
    def is_split(self) -> bool:
        return len(self.children) > 1

    @property
    def needs_context(self) -> bool:
        return bool(self.context_pointers)

    def to_dict(self) -> dict:
        return dict(
            utterance=self.utterance, mode=self.mode,
            children=[c.to_dict() for c in self.children],
            edges=[e.to_dict() for e in self.edges],
            constraints=[c.to_dict() for c in self.constraints],
            context_pointers=[p.to_dict() for p in self.context_pointers],
            counter_proposal=(self.counter_proposal.to_dict()
                              if self.counter_proposal else None),
            split_reason=self.split_reason, verdict=self.verdict,
            verdict_reason=self.verdict_reason,
            question=self.question.to_dict() if self.question else None)


def _is_instrumental_pair(left: Clause, right: Clause) -> bool:
    """The v1 instrumental rule, generalised to clauses: an investigation whose
    sibling is a bare repair of the SAME (unnamed) thing is ONE contract.

    "figure out why the queue backs up and fix it" — the right clause names no
    object of its own, so it is the same object, and splitting it would create a
    second contract with no referent at all.
    """
    if not re.search(r"\b(?:fix|repair)\b", right.text, flags=re.I):
        return False
    if not re.search(r"\bwhy\b|\bfigure out\b|\bdig into\b|\binvestigate\b",
                     left.text, flags=re.I):
        return False
    return not _has_content(re.sub(_ACTION_VERBS, " ", right.text, flags=re.I))


def decompose(utterance: str, repo_root=None,
              extra_context: dict | None = None) -> RequestBundle:
    """THE v3 ENTRY POINT: an utterance becomes a bundle of typed contracts.

    A single-clause ask produces a one-child bundle whose child is EXACTLY the
    v2 contract — same type, same score, same basis — so this is a superset of
    ``interpret``, never a competing reading of it.

    Verdict policy, in order:
      * a compound is ASK_ONE. Not because we cannot type the pieces, but
        because the ORDER and the sharing between them are genuinely
        underdetermined by the utterance, and guessing the order of "delete the
        old data and export it first" destroys data.
      * an unheld context pointer is ASK_ONE whatever the cue stack says: the
        specification is somewhere we cannot read, so any reading is a guess
        wearing a contract's clothes.
      * otherwise the primary child's own v2 verdict stands, unchanged.
    """
    ctx = dict(extra_context or {})
    constraints = extract_constraints(utterance)
    pointers = extract_context_pointers(utterance)
    counter = extract_counter_proposal(utterance)
    clauses = split_clauses(utterance, constraints)

    if len(clauses) == 2 and _is_instrumental_pair(clauses[0], clauses[1]):
        clauses = (Clause(text=(utterance or "").strip()),)

    children, edges = [], []
    for i, cl in enumerate(clauses):
        res = interpret(cl.text, repo_root=repo_root, extra_context=ctx)
        top = res.top
        if i > 0 and top is not None and not top.has_positive_evidence:
            # ADDITIVE CONTINUATION. "we need a spanish ui and also voice in
            # spanish" — the second clause carries no cue of its own because it
            # is elliptical: the verb lives in the first clause. Inheriting the
            # sibling's type is the reading the speaker meant, and it is stated
            # as a basis rather than smuggled in, so the false-closure guard can
            # still see exactly why this child is typed the way it is.
            #
            # This never fabricates evidence for a LONE ask: it fires only for a
            # non-first clause of an already-split compound.
            prior = children[0].contract
            top = Contract(type=prior.type, score=prior.score,
                           referents=top.referents,
                           confidence_basis=(
                               f"rule: elliptical continuation of the preceding "
                               f"{prior.type} clause — this clause states no verb "
                               f"of its own",),
                           missing=top.missing)
        children.append(ChildContract(id=f"c{i + 1}", contract=top,
                                      clause=cl, interpretation=res))
    if len(children) == 2:
        a, b = children
        if b.clause.ordering_hint == "first":
            # "delete X and export it FIRST" — the stated order is the reverse
            # of the order of mention. Reading order off position alone is
            # exactly the mistake that deletes the data before the export runs.
            edges.append(BundleEdge(src=b.id, dst=a.id, relation=EDGE_MUST_PRECEDE,
                                    reason="the second clause says 'first'"))
        elif b.clause.relation == EDGE_MUST_PRECEDE:
            edges.append(BundleEdge(src=a.id, dst=b.id, relation=EDGE_MUST_PRECEDE,
                                    reason=b.clause.connective))
        else:
            edges.append(BundleEdge(src=a.id, dst=b.id, relation=EDGE_PARALLEL,
                                    reason="no ordering stated"))

    if len(children) > 1:
        mode = MODE_COMPOUND
        split_reason = f"split on {children[1].clause.connective}"
    elif counter is not None:
        mode = MODE_COUNTER_PROPOSAL
        split_reason = f"rejection frame: {counter.frame}"
    else:
        mode = MODE_SINGLE
        split_reason = "no independent second action predicate"

    primary = children[0].interpretation
    verdict = primary.verdict
    reason = primary.verdict_reason
    question = primary.question

    if mode == MODE_COMPOUND:
        kinds = " + ".join(c.contract.type for c in children)
        verdict = "ASK_ONE"
        reason = (f"this is {len(children)} asks in one ({kinds}); which comes "
                  "first and whether they ship together is not stated")
        question = Question(
            text=("You asked for two things: "
                  + "; ".join(f"({c.id}) {c.clause.text}" for c in children)
                  + ". Should I do them together, or is one of them first?"),
            kind="decomposition",
            options=tuple(c.id for c in children) + ("both",),
            outcomes=tuple((c.id, f"{c.contract.type}@{c.id}") for c in children)
            + (("both", "bundle@all"),),
            reason="compound ask with unstated ordering")
    elif pointers:
        verdict = "ASK_ONE"
        reason = ("the ask refers to work we do not hold ("
                  + "; ".join(p.text for p in pointers) + ")")
        question = Question(
            text=(f"You said {pointers[0].text!r} — I do not have that. "
                  "Which report, change or conversation do you mean?"),
            kind="context", options=(), outcomes=(),
            reason=pointers[0].note)

    return RequestBundle(utterance=utterance, mode=mode, children=tuple(children),
                         edges=tuple(edges), constraints=constraints,
                         context_pointers=pointers, counter_proposal=counter,
                         split_reason=split_reason, verdict=verdict,
                         verdict_reason=reason, question=question)


# ---------------------------------------------------------------------------
# v3 scoring — the structure is scored, not just the type
# ---------------------------------------------------------------------------
# v2's headline was a TYPE/VERDICT number. That is a real measure of one thing
# and a silent non-measure of everything else: a row could score a perfect
# one-shot pass while the second half of a compound was dropped, a stated
# prohibition was lost and a rejected option was forgotten, because nothing
# asserted them (rule #223 — a reader with no writer, in reverse: a field with
# no assertion). v3 scores the STRUCTURE alongside the type, and reports the two
# separately so neither can hide inside the other.
V3_CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "frontdoor_eval_messy_v3.jsonl"

# The structural checks a row may opt into. A row that declares none is scored
# exactly as v2 scored it.
STRUCTURE_KEYS = ("expected_mode", "expected_children", "min_constraints",
                  "expects_context_ref")


def score_structure(row: dict, bundle: RequestBundle) -> dict:
    """Per-dimension structural verdicts for one row. Only declared keys count.

    Every dimension is reported with its own ``n`` so a corpus that exercises
    constraints twenty times and orderings twice cannot present itself as
    uniform coverage.
    """
    out: dict = {}
    if "expected_mode" in row:
        out["mode"] = (bundle.mode == row["expected_mode"])
    if "expected_children" in row:
        got = [c.contract.type for c in bundle.children]
        want = list(row["expected_children"])
        out["child_count"] = (len(got) == len(want))
        out["child_types"] = (got == want)
    if "min_constraints" in row:
        out["constraints"] = (len(bundle.constraints) >= int(row["min_constraints"]))
    if "expects_context_ref" in row:
        out["context_ref"] = (bundle.needs_context == bool(row["expects_context_ref"]))
    return out


def score_v3_row(row: dict, extra_context: dict | None = None,
                 max_questions: int = MAX_QUESTIONS) -> dict:
    """One row, scored on BOTH axes, through the v3 entry point.

    The type/verdict axis keeps the v2 three-outcome philosophy verbatim
    (clarify-then-pass is a PASS), so the numbers are comparable across
    versions. The structural axis is new and is never folded into the headline.
    """
    ctx = dict(extra_context or {})
    bundle = decompose(row["ask"], extra_context=ctx)
    expected = row.get("expected_type")
    got = bundle.primary.contract.type if bundle.primary else None
    type_ok = (got == expected) if expected is not None else True
    verdict_ok = bundle.verdict in row["acceptable_verdicts"]

    structure = score_structure(row, bundle)

    if type_ok and verdict_ok:
        outcome, questions, note = ONE_SHOT, 0, ""
    elif bundle.verdict == "PROCEED":
        outcome, questions = FAIL, 0
        note = ("closed confidently on the wrong reading - no clarification was "
                "offered, so none may rescue it")
    else:
        # It asked. A bounded exchange earns the second chance, exactly as v2.
        out = interpret_complete(row["ask"], answerer=scripted_answerer(row),
                                 extra_context=ctx, max_questions=max_questions)
        final = out.interpretation.top.type if out.interpretation.top else None
        questions = len(out.questions_asked)
        if expected is None or final == expected:
            outcome, note = CLARIFIED, out.termination
            got = final
        else:
            outcome = FAIL
            note = f"clarification did not converge ({out.termination})"
            got = final
    return dict(ask=row["ask"], expected=expected, outcome=outcome,
                final_type=got, verdict=bundle.verdict, questions=questions,
                mode=bundle.mode, structure=structure, note=note)


def evaluate_v3(rows=None, path=None, extra_context: dict | None = None,
                max_questions: int = MAX_QUESTIONS) -> dict:
    """The v3 table: the v2 three-rate headline PLUS per-dimension structure."""
    if rows is None:
        rows = load_messy_corpus(path or V3_CORPUS_PATH)
    detail = [score_v3_row(r, extra_context, max_questions) for r in rows]
    n = len(detail) or 1
    counts = {k: sum(1 for d in detail if d["outcome"] == k)
              for k in (ONE_SHOT, CLARIFIED, FAIL)}
    dims: dict = {}
    for d in detail:
        for key, ok in d["structure"].items():
            slot = dims.setdefault(key, {"n": 0, "ok": 0})
            slot["n"] += 1
            slot["ok"] += 1 if ok else 0
    for key, slot in dims.items():
        slot["rate"] = round(slot["ok"] / (slot["n"] or 1), 4)
    checked = sum(s["n"] for s in dims.values())
    passed = sum(s["ok"] for s in dims.values())
    return dict(
        n=len(detail), counts=counts,
        one_shot_rate=round(counts[ONE_SHOT] / n, 4),
        clarified_rate=round(counts[CLARIFIED] / n, 4),
        fail_rate=round(counts[FAIL] / n, 4),
        pass_rate=round((counts[ONE_SHOT] + counts[CLARIFIED]) / n, 4),
        structure=dims,
        structure_rate=round(passed / (checked or 1), 4),
        structure_checks=checked,
        detail=detail,
    )


def _spec_for(contract_type: str, field_name: str):
    """The declared spec for one field of one contract type, or None."""
    for s in FIELD_SPECS.get(contract_type, ()):
        if s.name == field_name:
            return s
    return None


# ===========================================================================
# THE OUTPUT SEAM: front-door contract -> procedure.TaskSpec
#
# A front-door contract that nothing consumes is a compiler with no back end.
# ``procedure.py`` runs the HYPOTHESIZE -> CHECK -> ACT ladder and starts from a
# ``TaskSpec`` (task_id, task_kind, request, surfaces, declared_scope), where
# ``task_kind`` selects the obligation pack the run must cover. This is the
# function that hands one over.
#
# IT FAILS CLOSED, and that is the whole point (rule #223, #127): an
# underdetermined contract must NOT seed a procedure run. A run started from a
# contract whose CRITICAL field is unknown will dutifully enumerate obligations
# about a thing nobody named, and every downstream artifact will look
# well-formed. Refusing here is the last place the gap is still visible.
# ===========================================================================

class SeamNotReady(ValueError):
    """Raised when a contract is too underdetermined to seed a procedure run.
    Carries the exact blocking reasons rather than a bare failure."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


#: contract type -> procedure task kind. Every mapping is an argument, not a
#: coincidence of naming:
#:   fix_bug    -> bug_fix    (identical intent)
#:   investigate-> bug_fix    the bug_fix pack's first three obligations are
#:                            reproduction / location / root_cause — which IS an
#:                            investigation. Mapping it to ``question`` would
#:                            demand source-sufficiency and freshness while
#:                            demanding no reproduction at all.
#:   improve    -> feature    an improvement changes behaviour on purpose, so it
#:                            owes the feature pack's compatibility, failure
#:                            behaviour and test obligations.
#:   build_new  -> feature    identical intent.
#:   question   -> question   identical intent.
#: A counter-proposal bundle overrides all of the above with ``judgment``: the
#: work is choosing between a rejected option and a proposed one, and only the
#: judgment pack carries ``rejected_alternatives``.
CONTRACT_TO_TASK_KIND = {
    "fix_bug": "bug_fix",
    "investigate": "bug_fix",
    "improve": "feature",
    "build_new": "feature",
    "question": "question",
}

#: Words that mean a procedure SURFACE is in play. Surfaces ADD obligations, so
#: a false positive costs a few extra checks and a false negative costs a missed
#: migration — the asymmetry is why these are deliberately generous.
_SURFACE_CUES = {
    "database": (r"\bdatabase\b|\bmigration\b|\bschema\b|\bsql\b|\btable\b|"
                 r"\bpostgres\b|\brows?\b|\bcolumn\b"),
    "public_api": r"\bapi\b|\bendpoint\b|\bcontract\b|\bclients?\b|\bserializer\b",
    "queue": r"\bqueue\b|\bworker\b|\bsqs\b|\bjob\b|\bbackground\b|\bretry\b|\bretries\b",
    "auth": r"\bauth\b|\blogin\b|\bsignup\b|\bpermission\b|\btenant\b|\btoken\b|\bprivate\b",
    "test": r"\btests?\b|\bfixtures?\b|\bcoverage\b|\bsuite\b",
}


def detect_surfaces(text: str) -> tuple:
    """Procedure surfaces named by the ask, in ``procedure.SURFACES`` order."""
    low = (text or "").lower()
    return tuple(name for name in sorted(_SURFACE_CUES)
                 if re.search(_SURFACE_CUES[name], low))


def _declared_scope(contract: Contract) -> tuple:
    """Repo paths the contract's referents actually resolved to.

    Only ``repo_file``/``repo_symbol``/``seam_violation`` resolutions contribute
    — an activity record is not a file the action may touch.
    """
    paths = set()
    for r in contract.referents:
        for res in r.resolved:
            if res.kind == "repo_file":
                paths.add(res.target)
            elif res.kind == "repo_symbol" and "::" in res.target:
                paths.add(res.target.split("::", 1)[0])
            elif res.kind == "seam_violation":
                parts = res.target.split(":")
                if len(parts) >= 2 and parts[1]:
                    paths.add(parts[1])
    return tuple(sorted(paths))


def to_task_spec(completed: CompletedContract, task_id: str,
                 bundle: RequestBundle | None = None,
                 child_id: str | None = None):
    """A completed contract becomes a ``procedure.TaskSpec``, or raises.

    ``request`` is the RENDERED SPEC, not the raw utterance: the whole value of
    the front door is that the elevated ask carries its provenance, its
    assumption ledger and its constraints, and handing procedure the raw
    utterance would throw all of that away at the seam.

    Blocking conditions, all fail-closed:
      * an open question         — the human was asked and has not answered
      * an unknown CRITICAL field — nothing to act on
      * an unheld context pointer — the specification lives somewhere we cannot
        read, so the request text would be a fiction
      * a compound bundle with no ``child_id`` — a bundle is N task specs, and
        silently seeding a run from only the first child is precisely the
        dropped-half failure v3 exists to prevent.
    """
    from detreason import procedure

    reasons = []
    if completed.open_asks:
        reasons.append("open questions remain: "
                       + ", ".join(v.field for v in completed.open_asks))
    blocked = completed.blocked_fields
    if blocked:
        reasons.append("critical field(s) unknown: "
                       + ", ".join(v.field for v in blocked))
    if bundle is not None:
        if bundle.needs_context:
            reasons.append("the ask points at prior work we do not hold: "
                           + "; ".join(p.text for p in bundle.context_pointers))
        if bundle.is_split and child_id is None:
            reasons.append(
                f"this is a compound of {len(bundle.children)} contracts "
                f"({', '.join(c.id for c in bundle.children)}); name the child "
                "to seed - a bundle is not one task")
    if reasons:
        raise SeamNotReady(reasons)

    ctype = completed.contract.type
    kind = CONTRACT_TO_TASK_KIND.get(ctype)
    if bundle is not None and bundle.mode == MODE_COUNTER_PROPOSAL:
        # Choosing between a rejected option and a proposed one is a JUDGMENT,
        # whatever the proposed half happens to look like.
        kind = "judgment"
    if kind is None:
        raise SeamNotReady([f"no procedure task kind for contract type {ctype!r}"])

    text = completed.utterance or ""
    surfaces = detect_surfaces(text + " " + " ".join(
        c.text for c in (bundle.constraints if bundle else ())))
    request = render_spec(completed)
    if bundle is not None and bundle.constraints:
        request += "\n  HARD CONSTRAINTS (stated by the human, not defaults):"
        for c in bundle.constraints:
            request += f"\n    ! [{c.kind}] {c.text}"
    if bundle is not None and bundle.counter_proposal is not None:
        cp = bundle.counter_proposal
        request += ("\n  REJECTED BY THE HUMAN (do not re-propose): "
                    f"{cp.rejected}\n  PROPOSED INSTEAD: {cp.proposed}")
    return procedure.TaskSpec(task_id=task_id, task_kind=kind, request=request,
                              surfaces=surfaces,
                              declared_scope=_declared_scope(completed.contract))


def bundle_to_task_specs(bundle: RequestBundle, task_id_prefix: str,
                         evidence_context: EvidenceContext | None = None,
                         budget_state=None) -> tuple:
    """Every child of a bundle as its own ``TaskSpec``, in dependency order.

    Ordering comes from the bundle's ``must_precede`` edges, never from the
    order the clauses were written — "delete the old data and export it first"
    states the reverse of its own word order, and running that in written order
    destroys the data.
    """
    order = [c.id for c in bundle.children]
    for e in bundle.edges:
        if e.relation != EDGE_MUST_PRECEDE:
            continue
        if order.index(e.src) > order.index(e.dst):
            order.remove(e.src)
            order.insert(order.index(e.dst), e.src)
    by_id = {c.id: c for c in bundle.children}
    out = []
    base = evidence_context
    for cid in order:
        child = by_id[cid]
        # Each child gets its OWN evidence context. Reusing one context across
        # children would point every child's referent resolver at the FIRST
        # child's contract and clause, so child two would be completed from
        # child one's text — the fields would look filled and describe the wrong
        # half of the ask. A supplied context contributes only its reusable
        # parts (answers, files, state); the contract and clause are per-child.
        ctx = EvidenceContext(
            utterance=child.clause.text,
            contract=child.contract,
            state=getattr(base, "state", None),
            files=dict(getattr(base, "files", {}) or {}),
            answers=dict(getattr(base, "answers", {}) or {}),
            constraints=bundle.constraints)
        done = complete(child.contract, ctx, budget_state)
        out.append(to_task_spec(done, task_id=f"{task_id_prefix}-{cid}",
                                bundle=bundle, child_id=cid))
    return tuple(out)
