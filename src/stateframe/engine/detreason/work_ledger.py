"""Durable work state + resume protocol (PLANNING V1, module 1).

What this is for: a run that gets killed mid-flight must be able to answer,
from the event stream and the filesystem ALONE, exactly one question — "is it
safe to continue, and from where?" — with no conversation memory, no model
call, and no internal bookkeeping trusted over the world.

The four doctrines this file encodes:

1. WRITE-AHEAD OR NOTHING. The intent event is appended and fsynced BEFORE the
   mutation runs. An intent without a receipt is the ONLY durable signature of
   a crash window; if the mutation could run first, that window is invisible
   forever and resume degenerates into guessing.

2. THE LEDGER IS A CLAIM, THE WORLD IS THE EVIDENCE. Every resume re-hashes
   the manifest files off disk. A receipt saying "done" is a hypothesis until
   the probe agrees with it.

3. AMBIGUITY STOPS. Three-way reconciliation (ledger claim / world / plan) has
   exactly four outcomes and two of them halt. Drift never becomes an
   amendment, and an unreadable probe is never read as "unchanged".

4. ONE STREAM. This builds ON decision_event/event_stream — same append-only
   JSONL, same governed vocabulary (extended additively with the five
   work-ledger types), same "one key one spelling" strictness. There is no
   second ledger to drift out of sync.

Everything here is deterministic: same events + same bytes on disk => same
verdict. No model calls, no clocks in any decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .decision_event import (
    DRIFT_DETECTED, PLAN_REGISTERED, RESUME_VERDICT, WORK_INTENT, WORK_RECEIPT,
    DecisionEvent, Producer,
)
from .event_stream import EventStream

__all__ = [
    "WorkLedgerError", "ManifestError", "PlanRegistration", "IntentRecord",
    "ResumeReport", "WorkLedger", "resume",
    "canonical_json", "sha256_ref", "plan_hash", "node_ids_for_plan",
    "manifest_from_files", "manifest_from_root", "idempotency_key",
    "CLEAN_RESUME", "REPLAY_NEEDED", "EXTERNAL_DRIFT", "AMBIGUOUS", "NO_PLAN",
]

PRODUCER_MODULE = "detreason.work_ledger"
LEDGER_VERSION = "1.0.0"

# ---- resume outcomes (the complete governed set) ---------------------------
CLEAN_RESUME = "CLEAN_RESUME"      # world matches the receipts; continue
REPLAY_NEEDED = "REPLAY_NEEDED"    # intent w/o receipt, world untouched; redo
EXTERNAL_DRIFT = "EXTERNAL_DRIFT"  # world moved under us; HALT + re-manifest
AMBIGUOUS = "AMBIGUOUS"            # intent w/o receipt AND world moved; STOP
NO_PLAN = "NO_PLAN"                # nothing registered; there is no work state

RESUME_OUTCOMES = frozenset({
    CLEAN_RESUME, REPLAY_NEEDED, EXTERNAL_DRIFT, AMBIGUOUS, NO_PLAN})

#: Outcomes after which an automated caller may keep going. Anything else is a
#: hard stop. Callers gate on THIS, never on a string comparison of their own.
SAFE_TO_PROCEED = frozenset({CLEAN_RESUME, REPLAY_NEEDED})


class WorkLedgerError(RuntimeError):
    """Ledger contract violated. Raised, never swallowed into a verdict."""


class ManifestError(WorkLedgerError):
    """A manifest could not be computed. Never degrades to 'unchanged'."""


# ------------------------------------------------------------------ hashing
def canonical_json(obj: Any) -> str:
    """One serialization, one hash. sort_keys so dict order never leaks in."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_ref(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def plan_hash(plan: Mapping) -> str:
    """Content address of the WHOLE plan. Any edit to it is a different plan."""
    if not isinstance(plan, Mapping):
        raise WorkLedgerError(f"plan must be a mapping, got {type(plan)}")
    return sha256_ref(canonical_json(plan))


def node_ids_for_plan(plan: Mapping) -> list[str]:
    """Immutable per-edit node ids, derived from the plan's own edit ids.

    Positional, because chains.edits_from_plan defaults an absent id to its
    position. A node id is stable for a given plan hash and meaningless across
    plan hashes — which is the point: re-planning cannot silently inherit the
    receipts of the old plan's nodes.
    """
    ph = plan_hash(plan)
    out = []
    for i, e in enumerate(plan.get("edits") or []):
        raw = (e.get("id") if isinstance(e, Mapping) else None) or f"e{i + 1}"
        out.append(f"{ph[7:19]}:{i}:{raw}")
    return out


# ----------------------------------------------------------------- manifests
def _digest_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def manifest_from_files(files: Mapping[str, str],
                        allowed_files: Iterable[str] | None = None) -> dict:
    """Scoped manifest over an in-memory {relpath: source} tree.

    A path in scope but ABSENT from the tree records `None` — absence is data,
    not a gap. `manifest_hash` covers the scoped mapping, so a file appearing
    or vanishing changes the hash exactly as a content change does.
    """
    if not isinstance(files, Mapping):
        raise ManifestError(f"files must be a mapping, got {type(files)}")
    scope = sorted(set(allowed_files)) if allowed_files is not None \
        else sorted(files)
    entries: dict[str, str | None] = {}
    for rel in scope:
        src = files.get(rel)
        if src is None:
            entries[rel] = None
            continue
        if not isinstance(src, str):
            raise ManifestError(f"{rel}: source must be str, got {type(src)}")
        entries[rel] = _digest_bytes(src.encode("utf-8"))
    return {"scope": scope, "entries": entries,
            "manifest_hash": sha256_ref(canonical_json(entries)),
            "source": "files"}


def manifest_from_root(root: str | os.PathLike,
                       allowed_files: Sequence[str]) -> dict:
    """Scoped manifest over a REAL directory. This is the world-probe.

    Fails loudly on anything it cannot read or that is not a regular file. An
    unreadable path must never be recorded as absent: "I could not look" and
    "it is not there" are different facts and conflating them is how a resume
    plows ahead through drift.
    """
    base = Path(root)
    if allowed_files is None:
        raise ManifestError("a world probe must be scoped: pass allowed_files")
    scope = sorted(set(allowed_files))
    entries: dict[str, str | None] = {}
    for rel in scope:
        p = base / rel
        try:
            if not p.exists():
                entries[rel] = None
                continue
            if not p.is_file():
                raise ManifestError(
                    f"{p} is in scope but is not a regular file")
            entries[rel] = _digest_bytes(p.read_bytes())
        except OSError as exc:
            raise ManifestError(
                f"cannot probe {p}: {exc}; an unreadable path is never "
                "recorded as unchanged") from exc
    return {"scope": scope, "entries": entries,
            "manifest_hash": sha256_ref(canonical_json(entries)),
            "source": "root", "root": str(base)}


def diff_manifests(before: Mapping, after: Mapping) -> dict:
    """Per-path change report between two manifests over the same scope."""
    b, a = dict(before.get("entries") or {}), dict(after.get("entries") or {})
    changed = sorted(p for p in set(b) & set(a) if b[p] != a[p])
    return {
        "changed": changed,
        "only_in_before": sorted(set(b) - set(a)),
        "only_in_after": sorted(set(a) - set(b)),
        "identical": before.get("manifest_hash") == after.get("manifest_hash"),
    }


# ----------------------------------------------------------- idempotency key
def idempotency_key(*, run_id: str, plan_hash_: str, node_id: str,
                    attempt: int, payload: Any, pre_manifest_hash: str) -> str:
    """Deterministic key for one logical action.

    STABLE across crashes for the same logical intent — that is the whole
    contract. A replay after a crash MUST present the same key so an external
    provider can reject the duplicate. It changes only when something that
    genuinely makes this a different action changes: a new attempt, a different
    payload, a different starting world, a different plan.
    """
    if not run_id or not node_id or not plan_hash_:
        raise WorkLedgerError(
            "idempotency key needs run_id, plan_hash and node_id")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise WorkLedgerError(f"attempt must be a non-negative int, got {attempt!r}")
    return sha256_ref(canonical_json([
        run_id, plan_hash_, node_id, attempt,
        canonical_json(payload), pre_manifest_hash,
    ]))


# --------------------------------------------------------------- data shapes
@dataclass(frozen=True)
class PlanRegistration:
    run_id: str
    plan_hash: str
    node_ids: tuple
    manifest: dict
    allowed_files: tuple
    event: DecisionEvent


@dataclass(frozen=True)
class IntentRecord:
    run_id: str
    plan_hash: str
    node_id: str
    attempt: int
    payload: Any
    pre_manifest: dict
    idempotency_key: str
    event: DecisionEvent


@dataclass(frozen=True)
class ResumeReport:
    """Deterministic verdict. `outcome` is the only thing callers branch on."""
    outcome: str
    run_id: str | None = None
    plan_hash: str | None = None
    reason: str = ""
    recommendation: str = ""
    pending_node_id: str | None = None
    pending_idempotency_key: str | None = None
    open_intents: tuple = ()
    completed_node_ids: tuple = ()
    changed_files: tuple = ()
    expected_manifest_hash: str | None = None
    world_manifest_hash: str | None = None
    events_emitted: tuple = ()

    @property
    def safe_to_proceed(self) -> bool:
        return self.outcome in SAFE_TO_PROCEED

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "events_emitted"}
        for k in ("open_intents", "completed_node_ids", "changed_files"):
            d[k] = list(d[k])
        return d


# ------------------------------------------------------------------- ledger
class WorkLedger:
    """Write-ahead work state on top of an EventStream.

    Usage is deliberately narrow:

        led = WorkLedger(stream, run_id="run-1")
        reg = led.register_plan(plan, files=files)
        with led.node(reg.node_ids[0], payload={...}) as intent:
            ...mutate...
            intent.receipt(post_files=new_files, outcome={"verdict": "ACCEPT"})

    `node()` exists so the write-ahead ordering is STRUCTURAL rather than a
    convention a caller can forget: the intent is durable before the body runs,
    and a body that raises still leaves the intent open (which is exactly the
    signature resume() needs to see).
    """

    def __init__(self, stream: EventStream, *, run_id: str,
                 repo_root: str | os.PathLike | None = None,
                 producer: Producer | None = None):
        if not isinstance(stream, EventStream):
            raise WorkLedgerError(
                f"stream must be an EventStream, got {type(stream)}")
        if not run_id:
            raise WorkLedgerError("run_id is required (it is the correlation id)")
        self.stream = stream
        self.run_id = run_id
        self.repo_root = str(repo_root) if repo_root is not None else None
        self.producer = producer or Producer(module=PRODUCER_MODULE,
                                             version=LEDGER_VERSION)
        self._registration: PlanRegistration | None = None
        self._attempts: dict[str, int] = {}

    # ----------------------------------------------------------- registration
    def register_plan(self, plan: Mapping, *,
                      files: Mapping[str, str] | None = None,
                      allowed_files: Sequence[str] | None = None
                      ) -> PlanRegistration:
        """Freeze the plan: immutable hash, immutable node ids, scoped manifest.

        `allowed_files` defaults to the union of the plan's declared
        allowed_files (plus each edit's own file), i.e. exactly the blast
        radius the plan claims. Manifesting anything wider would make every
        unrelated edit in a shared tree look like drift; manifesting anything
        narrower would let a real in-scope change slip past the probe.
        """
        ph = plan_hash(plan)
        nodes = node_ids_for_plan(plan)
        scope = sorted(set(allowed_files)) if allowed_files is not None \
            else _plan_scope(plan)
        if files is not None:
            manifest = manifest_from_files(files, scope)
        elif self.repo_root is not None:
            manifest = manifest_from_root(self.repo_root, scope)
        else:
            raise WorkLedgerError(
                "register_plan needs either files=... or a repo_root: a plan "
                "with no manifest cannot be resumed")
        ev = self.stream.append(DecisionEvent(
            correlation_id=self.run_id, producer=self.producer,
            decision_type=PLAN_REGISTERED,
            inputs_refs=(ph,), choice=ph,
            context={
                "kind": "work_ledger", "run_id": self.run_id,
                "plan_hash": ph, "node_ids": list(nodes),
                "allowed_files": list(scope), "manifest": manifest,
                "repo_root": self.repo_root, "plan": dict(plan),
                "ledger_version": LEDGER_VERSION,
            }))
        reg = PlanRegistration(run_id=self.run_id, plan_hash=ph,
                               node_ids=tuple(nodes), manifest=manifest,
                               allowed_files=tuple(scope), event=ev)
        self._registration = reg
        return reg

    @property
    def registration(self) -> PlanRegistration:
        if self._registration is None:
            raise WorkLedgerError("no plan registered on this ledger yet")
        return self._registration

    # ------------------------------------------------------------ write-ahead
    def open_intent(self, node_id: str, *, payload: Any,
                    pre_files: Mapping[str, str] | None = None,
                    attempt: int | None = None,
                    expected: Any = None) -> IntentRecord:
        """Append + fsync the intent. Call this BEFORE touching anything."""
        reg = self.registration
        if node_id not in reg.node_ids:
            raise WorkLedgerError(
                f"node {node_id!r} is not in the registered plan "
                f"{reg.plan_hash}; every action binds to exactly one node")
        if attempt is None:
            attempt = self._attempts.get(node_id, 0)
        self._attempts[node_id] = attempt + 1
        pre = self._probe(pre_files, reg.allowed_files)
        key = idempotency_key(run_id=self.run_id, plan_hash_=reg.plan_hash,
                              node_id=node_id, attempt=attempt,
                              payload=payload,
                              pre_manifest_hash=pre["manifest_hash"])
        ev = self.stream.append(DecisionEvent(
            correlation_id=self.run_id, producer=self.producer,
            decision_type=WORK_INTENT, causation_id=reg.event.event_id,
            inputs_refs=(reg.plan_hash, pre["manifest_hash"]), choice=node_id,
            context={
                "kind": "work_ledger", "run_id": self.run_id,
                "plan_hash": reg.plan_hash, "node_id": node_id,
                "attempt": attempt, "payload": payload,
                "expected": expected, "pre_manifest": pre,
                "idempotency_key": key,
            }))
        return IntentRecord(run_id=self.run_id, plan_hash=reg.plan_hash,
                            node_id=node_id, attempt=attempt, payload=payload,
                            pre_manifest=pre, idempotency_key=key, event=ev)

    def close_receipt(self, intent: IntentRecord, *, outcome: Any,
                      post_files: Mapping[str, str] | None = None,
                      ) -> DecisionEvent:
        """Completion receipt, caused_by the intent. Emitted AFTER the action."""
        if not isinstance(intent, IntentRecord):
            raise WorkLedgerError("close_receipt needs the IntentRecord")
        post = self._probe(post_files, self.registration.allowed_files)
        return self.stream.append(DecisionEvent(
            correlation_id=self.run_id, producer=self.producer,
            decision_type=WORK_RECEIPT, causation_id=intent.event.event_id,
            inputs_refs=(intent.idempotency_key,), choice=intent.node_id,
            outcome=outcome,
            context={
                "kind": "work_ledger", "run_id": self.run_id,
                "plan_hash": intent.plan_hash, "node_id": intent.node_id,
                "attempt": intent.attempt,
                "idempotency_key": intent.idempotency_key,
                "intent_event_id": intent.event.event_id,
                "pre_manifest_hash": intent.pre_manifest["manifest_hash"],
                "post_manifest": post,
            }))

    def node(self, node_id: str, *, payload: Any,
             pre_files: Mapping[str, str] | None = None,
             expected: Any = None) -> "_NodeScope":
        return _NodeScope(self, node_id, payload, pre_files, expected)

    # ------------------------------------------------------------------ probe
    def _probe(self, files: Mapping[str, str] | None,
               scope: Sequence[str]) -> dict:
        if files is not None:
            return manifest_from_files(files, scope)
        if self.repo_root is None:
            raise WorkLedgerError(
                "no files given and no repo_root: refusing to record a "
                "manifest that was never observed")
        return manifest_from_root(self.repo_root, scope)


class _NodeScope:
    """Context manager making write-ahead ordering structural, not optional."""

    def __init__(self, ledger: WorkLedger, node_id: str, payload: Any,
                 pre_files, expected):
        self._ledger = ledger
        self._args = (node_id, payload, pre_files, expected)
        self.intent: IntentRecord | None = None
        self.receipt_event: DecisionEvent | None = None

    def __enter__(self) -> "_NodeScope":
        node_id, payload, pre_files, expected = self._args
        self.intent = self._ledger.open_intent(
            node_id, payload=payload, pre_files=pre_files, expected=expected)
        return self

    def receipt(self, *, outcome: Any,
                post_files: Mapping[str, str] | None = None) -> DecisionEvent:
        if self.intent is None:
            raise WorkLedgerError("receipt before the intent was opened")
        self.receipt_event = self._ledger.close_receipt(
            self.intent, outcome=outcome, post_files=post_files)
        return self.receipt_event

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A body that raised leaves the intent OPEN on purpose: that is the
        # crash signature resume() reconciles. Never auto-close it.
        return False


def _plan_scope(plan: Mapping) -> list[str]:
    scope: set[str] = set()
    for e in plan.get("edits") or []:
        if not isinstance(e, Mapping):
            continue
        if e.get("file"):
            scope.add(e["file"])
        for f in e.get("allowed_files") or ():
            scope.add(f)
    if not scope:
        raise WorkLedgerError(
            "plan declares no files; there is nothing to manifest and "
            "therefore nothing that could be resumed safely")
    return sorted(scope)


# ------------------------------------------------------------------- resume
def resume(stream: EventStream, repo_root: str | os.PathLike, *,
           run_id: str | None = None,
           world_files: Mapping[str, str] | None = None,
           emit: bool = True) -> ResumeReport:
    """Reconstruct work state from the stream, probe the world, reconcile.

    Deterministic and read-mostly: the only events it may append are the
    DRIFT_DETECTED halt record and its own RESUME_VERDICT observation. It never
    mutates the tree and never continues past an unverified world.

    `world_files` exists for tests and for in-memory executors; when absent the
    probe reads `repo_root` off disk, which is the production path.
    """
    events = list(stream.iter_events())
    if run_id is None:
        regs = [e for e in events if e.decision_type == PLAN_REGISTERED]
        if not regs:
            return _verdict(stream, None, ResumeReport(
                outcome=NO_PLAN,
                reason="no plan_registered event in the stream",
                recommendation="register a plan before resuming"), emit)
        run_id = regs[-1].correlation_id
    events = [e for e in events if e.correlation_id == run_id]

    regs = [e for e in events if e.decision_type == PLAN_REGISTERED]
    if not regs:
        return _verdict(stream, run_id, ResumeReport(
            outcome=NO_PLAN, run_id=run_id,
            reason=f"run {run_id!r} has no plan_registered event",
            recommendation="register a plan before resuming"), emit)
    reg = regs[-1]                     # latest registration wins; older ones
    ph = reg.context.get("plan_hash")  # belong to superseded plans
    scope = list(reg.context.get("allowed_files") or ())
    base_manifest = reg.context.get("manifest") or {}

    # ---- fold: intents and the receipts that close them
    intents = [e for e in events
               if e.decision_type == WORK_INTENT
               and e.context.get("plan_hash") == ph]
    receipts = [e for e in events
                if e.decision_type == WORK_RECEIPT
                and e.context.get("plan_hash") == ph]
    closed = {r.causation_id for r in receipts}
    open_intents = [i for i in intents if i.event_id not in closed]
    completed = tuple(dict.fromkeys(
        r.context.get("node_id") for r in receipts
        if r.context.get("node_id")))

    # The ledger's CLAIM about the world: the last receipt's post-manifest,
    # or, with no receipt at all, the manifest taken at registration.
    claimed = (receipts[-1].context.get("post_manifest")
               if receipts else base_manifest)

    # ---- world probe (this is the part that outranks every claim above)
    try:
        world = (manifest_from_files(world_files, scope)
                 if world_files is not None
                 else manifest_from_root(repo_root, scope))
    except ManifestError as exc:
        return _verdict(stream, run_id, ResumeReport(
            outcome=AMBIGUOUS, run_id=run_id, plan_hash=ph,
            reason=f"world probe failed: {exc}",
            recommendation=("probe the tree by hand; an unreadable path is "
                            "never assumed unchanged"),
            open_intents=tuple(i.context.get("node_id") for i in open_intents),
            completed_node_ids=completed,
            expected_manifest_hash=claimed.get("manifest_hash")), emit)

    world_hash = world["manifest_hash"]
    matches_claim = world_hash == claimed.get("manifest_hash")
    changed = tuple(diff_manifests(claimed, world)["changed"])

    common = dict(run_id=run_id, plan_hash=ph,
                  open_intents=tuple(i.context.get("node_id")
                                     for i in open_intents),
                  completed_node_ids=completed,
                  changed_files=changed,
                  expected_manifest_hash=claimed.get("manifest_hash"),
                  world_manifest_hash=world_hash)

    if open_intents:
        pending = open_intents[-1]
        common["pending_node_id"] = pending.context.get("node_id")
        common["pending_idempotency_key"] = pending.context.get(
            "idempotency_key")
        # A crash between intent and receipt. The intent's OWN pre-manifest is
        # the reference here, not the last receipt: it is what the world looked
        # like the instant before the action was authorised.
        pre = pending.context.get("pre_manifest") or {}
        if world_hash == pre.get("manifest_hash"):
            return _verdict(stream, run_id, ResumeReport(
                outcome=REPLAY_NEEDED,
                reason=("intent without receipt and the world is byte-identical "
                        "to the intent's pre-state: the action did not land"),
                recommendation=("replay this node reusing idempotency key "
                                f"{pending.context.get('idempotency_key')}"),
                **common), emit)
        return _verdict(stream, run_id, ResumeReport(
            outcome=AMBIGUOUS,
            reason=("intent without receipt AND the world moved: whether the "
                    "action landed, half-landed, or someone else edited the "
                    "tree cannot be decided from here"),
            recommendation=("STOP and ask a human; do not replay and do not "
                            "skip — a duplicate and a lost edit look the same "
                            "from inside the ledger"),
            **common), emit)

    if matches_claim:
        return _verdict(stream, run_id, ResumeReport(
            outcome=CLEAN_RESUME,
            reason="no open intent and the world matches the last receipt",
            recommendation="continue with the next unstarted node",
            **common), emit)

    # No open intent, but the world is not what the receipts claim: something
    # outside this run edited the tree.
    report = ResumeReport(
        outcome=EXTERNAL_DRIFT,
        reason=("world diverged from the manifest with no action of ours in "
                f"flight; changed in scope: {list(changed)}"),
        recommendation=("HALT. Re-manifest against the observed tree and "
                        "re-plan from the new baseline. Drift is not an "
                        "amendment and must not be laundered into one."),
        **common)
    emitted = []
    if emit:
        emitted.append(stream.append(DecisionEvent(
            correlation_id=run_id,
            producer=Producer(module=PRODUCER_MODULE, version=LEDGER_VERSION),
            decision_type=DRIFT_DETECTED, causation_id=reg.event_id,
            inputs_refs=(claimed.get("manifest_hash") or "", world_hash),
            choice="halt",
            context={"kind": "work_ledger", "run_id": run_id, "plan_hash": ph,
                     "changed_files": list(changed),
                     "expected_manifest": claimed, "world_manifest": world,
                     "recommendation": "re-manifest"})))
    return _verdict(stream, run_id, report, emit, extra_events=emitted)


def _verdict(stream: EventStream, run_id: str | None, report: ResumeReport,
             emit: bool, extra_events: Sequence[DecisionEvent] = ()) -> ResumeReport:
    events = list(extra_events)
    if emit and run_id:
        events.append(stream.append(DecisionEvent(
            correlation_id=run_id,
            producer=Producer(module=PRODUCER_MODULE, version=LEDGER_VERSION),
            decision_type=RESUME_VERDICT, choice=report.outcome,
            context={"kind": "work_ledger", "run_id": run_id,
                     "verdict": report.to_dict()})))
    # replace(), not to_dict(): a round-trip through to_dict() would quietly
    # turn every tuple field into a list and change the report's own type
    # contract on the way out.
    return replace(report, events_emitted=tuple(events))
