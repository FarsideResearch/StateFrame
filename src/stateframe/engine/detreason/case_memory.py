"""CASE_MEMORY — the solved-case base with replay (memory engine v1).

Implements the case-base half of the committed design in
``sol_answers/w8/a_memory_engine.md`` (Q1 class 1 "solved-task certificates",
Q2 case-based reuse, Q3 poisoning defenses). The other five memory classes in
that taxonomy (competence/calibration, blacklists, codebase facts, menu-order
preferences, DecisionEvent history) are OUT OF SCOPE here and are named in
``OUT_OF_SCOPE`` so a reader never has to guess whether a gap is a bug.

The thesis: ``plan_synth`` already proves that for the seam families the kernel
can SEE, the repair is a lookup rather than a reasoning problem. A case base is
the next compression: the SECOND time a structurally identical seam breaks, the
kernel should not re-run the compiler search at all — it should recall the plan
TEMPLATE it already proved once, fill the slots from the live violations, and
re-prove it. Zero model calls either way; the saving is the search, not the AST.

Three properties this module refuses to trade away:

  1. **Verified-only minting.** A case may ONLY be minted from a run whose
     outcome was verified by something OUTSIDE the run: a chain gold-match, or
     a witness replay that PASSED. A fixer's own ``FIX_PROPOSED`` is the model
     grading its own homework and NEVER mints. See ``verification_of``.
  2. **Mandatory revalidation on replay.** An instantiated template is a
     CANDIDATE, never an answer. It must pass ``plan_synth.simulate`` against
     the LIVE files (targets cleared, nothing new introduced, every
     intermediate state parses, idempotent) before ``replay`` returns it. A
     failed revalidation returns ``None`` and counts as a miss. Replay never
     writes; it returns a plan for the caller's ``run_plan_loop``/``run_chain``.
  3. **Fail-closed quarantine.** Two CONSECUTIVE replay failures quarantine a
     case: it is excluded from matching and never deleted. Release requires a
     fresh EXTERNALLY VERIFIED confirmation — an ordinary "it worked" is not
     enough to undo evidence of poisoning.

Names never enter a signature. A rename case minted from
``helper -> helper_v2`` must match ``fetch -> fetch_scoped``, or the case base
is a lookup table of one repo instead of a memory of a seam FAMILY. Signatures
carry violation KINDS and their structural RELATION (which violations share a
file), never identifiers or paths.

Storage is an append-only JSONL event log (``results/case_memory.jsonl``).
Mutations are appended as events and folded into an in-memory index on load;
no line is ever rewritten, so the file doubles as the forensic trail the design
demands.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:                      # pragma: no cover
    sys.path.insert(0, str(_ROOT))

from detreason import plan_synth as ps              # noqa: E402

# ------------------------------------------------------------------ constants
SCHEMA = "case_memory/v1"

#: Default append-only log.
DEFAULT_PATH = _ROOT / "results" / "case_memory.jsonl"

#: Consecutive replay failures that quarantine a case (design Q3: fail-closed).
QUARANTINE_AFTER = 2

#: Memory classes from the committed taxonomy that this module does NOT own.
OUT_OF_SCOPE = {
    "competence_calibration": "per-(atom,context,stage) EWMA lives with the "
                              "skeptic/uledger, not the case base",
    "failed_approach_blacklist": "needs a mutation-family key this module does "
                                 "not mint",
    "codebase_facts": "file-hash-keyed fact cache; separate invalidation "
                      "lifecycle",
    "menu_order_preferences": "ordering preferences may never be derived from "
                              "a case base — a replayed case must not reorder "
                              "atoms inside an active task",
    "decision_events": "detreason.decision_event owns the append-only "
                       "provenance stream; this log is a case store, not it",
}

#: The ONE spelling of each externally-verified outcome. Rule #223: a reader
#: with no writer is a dead feature, so each of these names its producer.
#:
#:   * ``gold_match``      — produced HERE, by comparing ``cert['final_files']``
#:     to ``task['gold_files']`` (the harvested twofile corpus in
#:     ``results/twofile_tasks.jsonl`` carries ``gold_files`` on every row).
#:   * ``witness_pass``    — read from ``cert['witness_verdict'] == 'PASS'``.
#:     CONTRACT NOTE: nothing in the tree stamps that key today. It is accepted
#:     because the design names witness replay as a mint path, and it is
#:     declared here as the exact literal the witness lane must write. Until a
#:     producer exists, only the gold-match path can actually mint — that is a
#:     stated limitation, not a silent one.
VERIFICATION_GOLD = "gold_match"
VERIFICATION_WITNESS = "witness_pass"

#: Chain verdict that is a precondition for minting (never sufficient alone).
_MINTABLE_CHAIN_VERDICT = "CHAIN_COMPLETE"

#: Fixer verdicts that are explicitly NOT verification. Named so the refusal is
#: readable in a rejection reason rather than implied by an absence.
_SELF_REPORTED_VERDICTS = ("FIX_PROPOSED", "LOW_CONFIDENCE_PROPOSAL",
                           "PARTIAL_SALVAGED", "ABSTAIN")


class MintRefused(Exception):
    """Raised when a certificate does not earn a case. Carries the reason."""


# =====================================================================
# 1. SIGNATURES — kinds + relation, never names
# =====================================================================
def _canonical_violations(violations) -> list:
    """One record per SEAM, deterministically ordered.

    ``seam_violations`` reports a site per line, so three stale uses of one
    symbol in one file are three records of ONE seam. ``plan_synth.synthesize``
    collapses them on the ``(kind, file, name)`` triple before deciding, and so
    do we — otherwise the same breakage would sign differently depending on how
    many times the consumer happened to call the symbol.

    Records are kept WHOLE (``lineno`` included): slot resolvers need the line
    to find the import that raised the violation. Nothing here reaches a
    signature — ``violation_signature`` reads only ``kind`` and file GROUPING.
    """
    out, seen = [], set()
    for v in sorted(violations, key=lambda v: (v["kind"], v["file"], v["name"],
                                               v.get("lineno", 0))):
        key = (v["kind"], v["file"], v["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(v))
    return out


def violation_signature(violations, *, level: str = "exact") -> str:
    """Normalized signature of a seam breakage.

    ``level='exact'``      kinds + multiplicity + which violations SHARE a file
                           (files are replaced by their group index, so no path
                           or identifier survives).
    ``level='structural'`` kinds + multiplicity only — the loosest tier the
                           design allows, still never a name.

    A rename in ``cons.py`` of ``helper`` and a rename in ``svc/api.py`` of
    ``fetch`` produce the SAME signature at both levels. That is the point.
    """
    if level not in ("exact", "structural"):
        raise ValueError(f"unknown signature level {level!r}")
    canon = _canonical_violations(violations)
    kinds = sorted(Counter(v["kind"] for v in canon).items())
    if level == "structural":
        body = dict(level="structural", kinds=kinds)
    else:
        # File identity is replaced by first-appearance group index: the
        # RELATION (same file / different file) survives, the path does not.
        groups: dict = {}
        for v in canon:
            groups.setdefault(v["file"], len(groups))
        body = dict(level="exact", kinds=kinds, n_files=len(groups),
                    shape=sorted((v["kind"], groups[v["file"]]) for v in canon))
    return f"{level}:" + hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


# =====================================================================
# 2. SLOTS — the template's only vocabulary
# =====================================================================
#: A slot is ``{{v<i>.<resolver>}}`` where ``i`` indexes the canonically ordered
#: violations of the task. Every resolver is deterministic, code-derived, and
#: returns ``None`` when it cannot PROVE a single answer — an unresolved slot
#: aborts the replay rather than guessing.
def _slot(i: int, resolver: str) -> str:
    return "{{v%d.%s}}" % (i, resolver)


def _r_file(files, viol):
    return viol["file"]


def _r_name(files, viol):
    return viol["name"]


def _r_module(files, viol):
    return ps._importing_module(files, viol)[0]


def _r_producer(files, viol):
    return ps._importing_module(files, viol)[1]


def _r_new_name(files, viol):
    """The unique replacement definition in the producer module, or ``None``.

    Deliberately NOT ``compile_rename``: this is a slot resolver, so it answers
    one question (what did this symbol become?) and abstains on any tie. It
    reuses ``plan_synth``'s uniqueness helpers so the two cannot drift on what
    "plausible replacement" means, but it invokes no compiler and builds no
    menu. Ambiguity here is a replay MISS, which falls through to synthesis.
    """
    if viol["kind"] not in ps.RENAME_EVIDENCE_KINDS:
        return None
    old = viol["name"]
    module, producer = ps._importing_module(files, viol)
    if producer is None or producer not in files:
        return None
    consumer_names = ps._names_bound_in(files, viol["file"])
    already = ps._consumer_imports(files, viol["file"], module)
    wanted = ps._consumer_uses_as(files, viol["file"], old)

    kept, similar = [], []
    for d in ps._module_defs(files, producer):
        if d["name"] == old:
            return None                       # producer still defines it
        if d["name"] in consumer_names or d["name"] in already:
            continue
        if wanted and d["kind"] not in wanted:
            continue
        row = (round(ps._similarity(old, d["name"]), 4), d["name"])
        if ps._keeps_stem(old, d["name"]):
            kept.append(row)
        elif row[0] >= ps.RENAME_SIMILARITY_FLOOR:
            similar.append(row)
    cands = sorted(kept or similar, key=lambda r: (-r[0], r[1]))
    if not cands:
        return None
    if len(cands) > 1:                        # a tie or a rival is a MISS
        return None
    return cands[0][1]


#: Declared registry. Adding a resolver is a deliberate, reviewable act.
SLOT_RESOLVERS = {
    "file": _r_file,
    "name": _r_name,
    "module": _r_module,
    "producer": _r_producer,
    "new_name": _r_new_name,
}


def _resolve_slots(files: dict, violations) -> dict:
    """{slot_token: concrete_value} for every resolvable slot, live."""
    out = {}
    for i, viol in enumerate(_canonical_violations(violations)):
        for res, fn in SLOT_RESOLVERS.items():
            try:
                val = fn(files, viol)
            except Exception:                 # noqa: BLE001 — a resolver bug is
                val = None                    # an abstention, never a guess
            if isinstance(val, str) and val:
                out[_slot(i, res)] = val
    return out


def _substitute(obj, mapping: dict):
    """Recursively replace every key of ``mapping`` inside ``obj``'s strings.

    ONE pass, longest key first. Sequential ``str.replace`` calls are wrong
    here and were caught by a test: the module ``prod`` is a substring of the
    path ``prod.py``, so a second pass chewed a hole in the token the first
    pass had just written (``{{v0.{{v0.module}}ucer}}``). A single regex
    alternation never re-scans text it has already emitted, which is the only
    version of this that is safe in BOTH directions (concrete->slot when
    abstracting a plan, slot->concrete when instantiating one).
    """
    if isinstance(obj, str):
        if not mapping:
            return obj
        order = sorted(mapping, key=lambda k: (-len(k), k))
        pat = re.compile("|".join(re.escape(k) for k in order))
        return pat.sub(lambda m: mapping[m.group(0)], obj)
    if isinstance(obj, list):
        return [_substitute(o, mapping) for o in obj]
    if isinstance(obj, tuple):
        return tuple(_substitute(o, mapping) for o in obj)
    if isinstance(obj, dict):
        return {k: _substitute(v, mapping) for k, v in obj.items()}
    return obj


def abstract_plan(plan: dict, files: dict, violations) -> dict:
    """Concrete plan -> plan SHAPE (the edit template with slots).

    Every value a resolver can reproduce from the live repo is replaced by its
    slot. What survives literally is genuinely constant across the family
    (edit ids, roles, phases, ``synth_op``, prose scaffolding).
    """
    live = _resolve_slots(files, violations)
    # invert: concrete value -> slot token
    inverse: dict = {}
    for token, value in live.items():
        inverse.setdefault(value, token)
    return _substitute(json.loads(json.dumps(plan, sort_keys=True)), inverse)


def instantiate(shape: dict, files: dict, violations):
    """Plan SHAPE + live repo -> concrete plan, or ``None`` if a slot is unmet.

    Fail-closed: if the shape mentions a slot no resolver could prove against
    THESE files, there is no plan. A partially-filled template is exactly the
    "plausible wrong answer" the kernel exists to refuse.
    """
    live = _resolve_slots(files, violations)
    plan = _substitute(json.loads(json.dumps(shape, sort_keys=True)), live)
    leftover = sorted(_unfilled_slots(plan))
    if leftover:
        return None
    return plan


def _unfilled_slots(obj) -> set:
    out = set()
    if isinstance(obj, str):
        rest = obj
        while "{{v" in rest:
            i = rest.index("{{v")
            j = rest.find("}}", i)
            if j < 0:
                break
            out.add(rest[i:j + 2])
            rest = rest[j + 2:]
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            out |= _unfilled_slots(o)
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _unfilled_slots(v)
    return out


# =====================================================================
# 3. VERIFICATION — the mint gate
# =====================================================================
def verification_of(cert: dict, task: dict | None = None) -> dict:
    """Was this outcome verified by something OUTSIDE the run?

    Returns ``{ok, kind, reason}``. ``ok`` is never True without either a
    gold-file match or an explicit witness PASS. THE single authority; nothing
    else in this module inspects verdicts.
    """
    cert = cert or {}
    verdict = cert.get("verdict")
    if verdict != _MINTABLE_CHAIN_VERDICT:
        return dict(ok=False, kind=None, reason=(
            f"chain verdict {verdict!r} is not {_MINTABLE_CHAIN_VERDICT}"
            + (" (a self-reported fixer verdict is not verification)"
               if verdict in _SELF_REPORTED_VERDICTS else "")))

    for check in cert.get("seam_checks") or ():
        if check.get("edit_id") == "final" and not check.get("ok", False):
            return dict(ok=False, kind=None,
                        reason="final seam gate did not pass")

    gold = (task or {}).get("gold_files")
    if gold is not None:
        if cert.get("final_files") == gold:
            return dict(ok=True, kind=VERIFICATION_GOLD, reason="")
        return dict(ok=False, kind=None,
                    reason="chain completed but final files != gold files")

    if cert.get("witness_verdict") == "PASS":
        return dict(ok=True, kind=VERIFICATION_WITNESS, reason="")

    return dict(ok=False, kind=None, reason=(
        "no external verification: task carries no 'gold_files' and cert "
        "carries no witness_verdict=='PASS'"))


# =====================================================================
# 4. CASE STORE
# =====================================================================
def _new_case(case_id, sig_exact, sig_struct, shape, cert_ref, kind) -> dict:
    return dict(
        schema=SCHEMA,
        case_id=case_id,
        violation_signature=sig_exact,
        structural_signature=sig_struct,
        plan_shape=shape,
        certificate_ref=cert_ref,
        outcome=kind,                 # gold_match | witness_pass
        times_replayed=0,
        times_confirmed=0,
        consecutive_failures=0,
        quarantined=False,
    )


class CaseStore:
    """Append-only case base with an in-memory index.

    Every state change is a new line. ``mint`` writes a ``case`` event;
    ``confirm`` writes a ``confirm`` event; quarantine and release are derived
    consequences recorded on that same event. Nothing is ever rewritten or
    deleted, so a poisoned case stays readable for forensics after it is
    excluded from matching.
    """

    def __init__(self, path=DEFAULT_PATH, *, load: bool = True):
        self.path = pathlib.Path(path)
        self.cases: dict = {}
        self.misses = 0                # replays that failed revalidation
        if load:
            self.load()

    # ---------------------------------------------------------- persistence
    def load(self) -> None:
        self.cases = {}
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue               # a corrupt line is skipped, not fatal
            self._fold(ev)

    def _fold(self, ev: dict) -> None:
        op = ev.get("op")
        if op == "case":
            self.cases[ev["case"]["case_id"]] = dict(ev["case"])
        elif op == "confirm":
            c = self.cases.get(ev["case_id"])
            if c is not None:
                c.update(ev["state"])

    def _append(self, ev: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, sort_keys=True, default=str) + "\n")

    def cases_file_hash(self) -> str:
        """Canary for the idle scheduler.

        The idle lane samples this; a change it did not cause means the case
        base was edited out of band, and every case must be re-verified before
        the next replay. Returns the sha256 of the raw log bytes (``"absent"``
        when there is no log yet — an absence is a state, not an error).
        """
        if not self.path.exists():
            return "absent"
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    # ---------------------------------------------------------------- mint
    def mint_from_chain_cert(self, cert: dict, task: dict) -> dict:
        """Mint a case from an EXTERNALLY VERIFIED chain run.

        ``task`` must carry the repo the plan ran against and the plan itself:

            {"id":..., "injected_files": {...}, "plan": {...},
             "gold_files": {...}}   # gold_files optional iff cert has a witness

        Raises ``MintRefused`` (never returns a case) when verification fails.
        This is the attack surface: an unverified certificate that reaches here
        must LEAVE as an exception, not as a memory.
        """
        v = verification_of(cert, task)
        if not v["ok"]:
            raise MintRefused(v["reason"])

        files = task.get("injected_files")
        plan = task.get("plan")
        if not files or not plan:
            raise MintRefused("task carries no 'injected_files'/'plan'; a case "
                              "cannot be abstracted without the run's inputs")

        violations = ps.all_violations(files)
        if not violations:
            raise MintRefused("no baseline violations: nothing to key a case on")

        shape = abstract_plan(plan, files, violations)
        sig = violation_signature(violations, level="exact")
        struct = violation_signature(violations, level="structural")
        cert_ref = cert.get("plan_hash") or hashlib.sha256(
            json.dumps(cert, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        case_id = "case-" + hashlib.sha256(
            json.dumps([sig, shape], sort_keys=True).encode()
        ).hexdigest()[:12]

        existing = self.cases.get(case_id)
        if existing is not None:
            return existing            # idempotent: the same proof twice is one case
        case = _new_case(case_id, sig, struct, shape, cert_ref, v["kind"])
        self.cases[case_id] = case
        self._append(dict(op="case", case=case, task_id=task.get("id", "")))
        return case

    # --------------------------------------------------------------- match
    def match(self, violations) -> list:
        """Candidate cases for a live breakage: exact signature, then structural.

        Quarantined cases are EXCLUDED (never deleted). Order is deterministic
        and confirmation-weighted so a well-proven case is tried first.
        """
        sig = violation_signature(violations, level="exact")
        struct = violation_signature(violations, level="structural")
        live = [c for c in self.cases.values() if not c.get("quarantined")]

        def rank(c):
            return (-c.get("times_confirmed", 0), c["case_id"])

        exact = sorted((c for c in live if c["violation_signature"] == sig), key=rank)
        loose = sorted((c for c in live
                        if c["violation_signature"] != sig
                        and c.get("structural_signature") == struct), key=rank)
        return exact + loose

    # -------------------------------------------------------------- replay
    def replay(self, case: dict, files: dict, violations):
        """Instantiate ``case``'s template against the LIVE repo and re-prove it.

        Returns a plan ONLY when ``plan_synth.simulate`` passes on these files.
        Never writes to the repo: the caller runs it. A failed revalidation
        returns ``None`` and increments ``self.misses``.
        """
        plan = instantiate(case["plan_shape"], files, violations)
        if plan is None:
            self.misses += 1
            return None
        baseline = ps.all_violations(files)
        base_keys = {ps._vkey(v) for v in baseline}
        # Targets are the baseline violations this instantiated plan claims —
        # every baseline violation whose file is inside the plan's closure.
        closure = set(plan.get("allowed_files") or
                      [e["file"] for e in plan.get("edits") or ()])
        targets = tuple(sorted(k for k in base_keys if k[1] in closure))
        if not targets:
            self.misses += 1
            return None
        sim = ps.simulate(plan, files, targets=targets, baseline=baseline)
        if not sim["ok"]:
            self.misses += 1
            return None
        return plan

    # ------------------------------------------------------------- confirm
    def confirm(self, case_id: str, ok: bool, *, verified: bool = False) -> dict:
        """Report a replay OUTCOME back into memory. Mandatory after every replay.

        ``ok=False`` twice IN A ROW quarantines the case. ``ok=True`` resets the
        streak and counts a confirmation, but only ``verified=True`` (a fresh
        externally-verified confirmation, same bar as minting) may RELEASE a
        quarantined case — otherwise the release would be exactly the wrong
        memory re-confirming itself.
        """
        case = self.cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        case["times_replayed"] = case.get("times_replayed", 0) + 1
        if ok:
            case["times_confirmed"] = case.get("times_confirmed", 0) + 1
            case["consecutive_failures"] = 0
            if case.get("quarantined") and verified:
                case["quarantined"] = False
                case["released_by"] = "verified_confirmation"
        else:
            case["consecutive_failures"] = case.get("consecutive_failures", 0) + 1
            if case["consecutive_failures"] >= QUARANTINE_AFTER:
                case["quarantined"] = True
        state = {k: case[k] for k in
                 ("times_replayed", "times_confirmed", "consecutive_failures",
                  "quarantined", "released_by") if k in case}
        self._append(dict(op="confirm", case_id=case_id, ok=bool(ok),
                          verified=bool(verified), state=state))
        return case


# =====================================================================
# 5. THE STANDARD ENTRY POINT
# =====================================================================
def plan_or_recall(files: dict, violations, store: CaseStore) -> dict:
    """Memory first, synthesis second. Pure: nothing is written to the repo.

    Returns ``{source, plan, case_id, detail}``:

      * ``source='memory'`` — a matched case's template instantiated against
        THESE files and re-proven by ``simulate``. Zero compilers invoked.
      * ``source='synth'``  — no case matched (or every candidate failed
        revalidation); ``plan_synth.synthesize`` ran and its first passing plan
        is returned. ``plan`` is ``None`` when synthesis also abstains, and
        ``detail`` carries the honest partition counts.

    The CALLER is responsible for reporting the outcome back via
    ``store.confirm(case_id, ok)`` when ``source == 'memory'``. A replay whose
    outcome is never reported can never be quarantined, which is why the
    contract is stated here rather than left to a comment at the call site.
    """
    tried = []
    for case in store.match(violations):
        plan = store.replay(case, files, violations)
        if plan is not None:
            return dict(source="memory", plan=plan, case_id=case["case_id"],
                        detail=dict(tried=tried,
                                    signature=case["violation_signature"]))
        tried.append(case["case_id"])

    res = ps.synthesize(files)
    plan = res["plans"][0]["plan"] if res["plans"] else None
    return dict(source="synth", plan=plan, case_id=None,
                detail=dict(tried=tried, counts=res["counts"],
                            model_calls=res["model_calls"]))


__all__ = [
    "CaseStore", "MintRefused", "OUT_OF_SCOPE", "QUARANTINE_AFTER", "SCHEMA",
    "SLOT_RESOLVERS", "abstract_plan", "instantiate", "plan_or_recall",
    "verification_of", "violation_signature",
]
