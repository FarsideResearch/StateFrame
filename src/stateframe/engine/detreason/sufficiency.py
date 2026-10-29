"""SUFFICIENCY — the required-input manifest gate (PLANNING V1, MODULE 4).

Design authority: ``/tmp/solq/w17/a_module4_suff.md``. Where that spec named an
API this tree does not actually have, the BUILT API wins and the deviation is
named in a comment rather than invented into existence.

The claim this module makes is narrow and mechanical: **before a plan is
allowed to mutate the world, every input class its own milestones depend on
must be SATISFIED BY CODE — files that exist and parse, acceptance predicates
that are executable, referenced symbols that actually resolve somewhere in the
world, evidence refs that resolve to matching content, and an independent
oracle per milestone.** A plan that is missing a decision-critical input does
not proceed "carefully"; it BLOCKS.

Why this is a gate and not a warning
------------------------------------
The whole failure mode this kernel exists to kill is *plausible work on a
premise nobody checked*. An agent that starts editing ``worker.py`` because the
plan said to, when the type the plan renames toward does not exist anywhere in
the tree, will produce a green executor certificate, a green receipt, and a
world that is wrong. The seam check passes because the seam was never the
problem. The only place to catch that is BEFORE the first ``WORK_INTENT``.

The U-ledger relationship (the load-bearing asymmetry)
-----------------------------------------------------
Every missing input opens a deterministic ``URow`` so the uncertainty is
CHARGED and visible to the skeptic. That is all it does. A U-row can never
waive the input it charges for:

* a low total U does not unblock — :meth:`ULedger.ready_to_act` is consulted
  for an ADDITIONAL veto, never for permission;
* a row that some caller has already CLOSED does not unblock — closure is a
  statement about uncertainty, not about the existence of a file;
* an empty ledger, or no ledger at all, does not unblock.

``test_sufficiency.py`` drives all three of those as attacks. If someone later
"optimises" the gate by consulting the ledger for a PROCEED, those tests fail.

Deviations from the spec, deliberately
--------------------------------------
1. The spec extends ``adherence.PlanNode`` with typed declaration fields. This
   module does not own ``adherence.py`` and does not touch it. Typed
   declarations are read from the LEGACY PLAN MAPPING (the same mapping
   ``plan_run.run_plan_loop`` already consumes), with the ``adherence.Plan``
   used as the authority for invariants and milestone identity. Same six input
   classes, same verdicts, no edit to a file owned by another lane.
2. The spec derives entities/contracts/invariants through ``chains.seam_manifest()``
   / ``cross_file_seams()`` / ``plan_invariants()``. Those functions do not
   exist in this tree (``chains.py`` exposes ``run_chain`` and is owned by
   another lane), so derivation is done here with the stdlib ``ast`` module
   against the real world. A ``seam_manifest`` provider may be INJECTED if that
   lane later lands one; the fallback is not a stub, it parses real source.
3. The verdict vocabulary is ``PROCEED`` / ``GATHER`` / ``BLOCK`` (the spec's
   two-verdict form plus the caller-visible GATHER list). GATHER carries only
   non-decision-critical items and does not stop the run.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import adherence as _adh
from . import uledger as _u

log = logging.getLogger(__name__)

__all__ = [
    "PROCEED", "GATHER", "BLOCK",
    "PLAN_PREFLIGHT", "NODE_PRE", "NODE_POST", "FINAL", "STAGES",
    "ENTITIES", "CONTRACTS", "INVARIANTS", "EXTERNAL_SEAMS", "EVIDENCE",
    "ORACLE", "INPUT_CLASSES", "DECISION_CRITICAL_CLASSES",
    "RequiredInput", "RequiredInputManifest", "SufficiencyResult",
    "SufficiencyError",
    "derive_required_inputs", "evaluate_required_inputs", "check_sufficiency",
    "world_from_root", "charge_missing_inputs",
]

# ------------------------------------------------------------------ vocabulary

PROCEED = "PROCEED"
GATHER = "GATHER"
BLOCK = "BLOCK"

PLAN_PREFLIGHT = "PLAN_PREFLIGHT"
NODE_PRE = "NODE_PRE"
NODE_POST = "NODE_POST"
FINAL = "FINAL"
STAGES = (PLAN_PREFLIGHT, NODE_PRE, NODE_POST, FINAL)

ENTITIES = "entities"
CONTRACTS = "contracts"
INVARIANTS = "invariants"
EXTERNAL_SEAMS = "external_seams"
EVIDENCE = "evidence"
ORACLE = "oracle"
INPUT_CLASSES = (ENTITIES, CONTRACTS, INVARIANTS, EXTERNAL_SEAMS, EVIDENCE,
                 ORACLE)

#: Missing one of these is a BLOCK, full stop. The two that are NOT here are
#: still charged and still reported — they land in the GATHER list — because an
#: undeclared advisory input is a gap in the paperwork, not a wrong premise.
DECISION_CRITICAL_CLASSES = frozenset(INPUT_CLASSES)

#: What kind of uncertainty each input class is, in the U-ledger's vocabulary.
#: These are the ledger's own types; nothing new is invented here.
_UTYPE_FOR_CLASS = {
    ENTITIES: _u.UNGROUNDED_SYMBOL,
    CONTRACTS: _u.UNGROUNDED_SYMBOL,
    INVARIANTS: _u.UNVERIFIED_CLAIM,
    EXTERNAL_SEAMS: _u.PREDICTED_RISK,
    EVIDENCE: _u.UNKNOWN_ATOM,
    ORACLE: _u.UNVERIFIED_CLAIM,
}

#: The capabilities an external seam must DECLARE before an unattended run may
#: cross it. A seam you cannot probe, cannot repeat safely, and cannot get a
#: receipt from is not a seam an unattended loop may touch.
REQUIRED_SEAM_CAPABILITIES = ("probe", "idempotency", "receipt")

#: Modules that resolve without a file in our world. Deliberately small: an
#: unknown import is a MISSING CONTRACT, not a stdlib module we forgot.
_STDLIB_OK = frozenset(getattr(sys, "stdlib_module_names", ()))


class SufficiencyError(RuntimeError):
    """A malformed gate invocation. NOT a runtime refusal (those are verdicts)."""


# ------------------------------------------------------------------- utilities

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def _sha(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def world_from_root(root: str | os.PathLike, *,
                    extra: Sequence[str] = ()) -> dict:
    """The world the gate judges: every top-level ``.py`` file, read fresh.

    Deliberately the WHOLE tree and not the node's closure. A rename's
    destination usually lives OUTSIDE the closure (that is what makes it a
    contract), so a gate that could only see the closure could never check the
    one thing it exists to check.
    """
    p = Path(root)
    out: dict = {}
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix == ".py":
                try:
                    out[f.name] = f.read_text(encoding="utf-8")
                except OSError as exc:            # unreadable != absent
                    log.warning("sufficiency: cannot read %s: %s", f, exc)
    for rel in extra or ():
        if rel in out:
            continue
        f = p / rel
        if f.is_file():
            try:
                out[rel] = f.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("sufficiency: cannot read %s: %s", f, exc)
    return out


def _module_name(rel: str) -> str:
    return rel[:-3] if rel.endswith(".py") else rel


def _parse(src: str):
    """``(tree, error)``. A file that does not parse is a MISSING entity."""
    try:
        return ast.parse(src or ""), ""
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc.msg} (line {exc.lineno})"


def _top_level_names(tree) -> set:
    """Module-level definitions: what this module can honestly export."""
    names: set = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


def _all_names(tree) -> set:
    """Every identifier the module mentions anywhere (defs, refs, attributes)."""
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[0])
            if node.asname:
                names.add(node.asname)
    return names


def _imported_modules(tree) -> set:
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:                 # relative import: not our world
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


class _WorldIndex:
    """A parsed view of the world. One parse per file, reused by every check.

    This is the fallback for the spec's ``chains.seam_manifest()``. If that lane
    lands one, inject it as ``seam_manifest=`` and this index steps aside.
    """

    def __init__(self, world: Mapping[str, str]):
        self.world = dict(world or {})
        self.trees: dict = {}
        self.parse_errors: dict = {}
        for rel, src in self.world.items():
            tree, err = _parse(src)
            if tree is None:
                self.parse_errors[rel] = err
            else:
                self.trees[rel] = tree
        self._defs: dict = {rel: _top_level_names(t)
                            for rel, t in self.trees.items()}
        self._names: dict = {rel: _all_names(t)
                             for rel, t in self.trees.items()}
        self.modules = {_module_name(rel) for rel in self.world}

    def has_file(self, rel: str) -> bool:
        return rel in self.world

    def defines(self, name: str) -> tuple:
        """Every file whose MODULE LEVEL defines ``name`` (sorted)."""
        return tuple(sorted(rel for rel, ns in self._defs.items()
                            if name in ns))

    def mentions(self, rel: str, name: str) -> bool:
        return name in self._names.get(rel, ())

    def imports_of(self, rel: str) -> set:
        tree = self.trees.get(rel)
        return _imported_modules(tree) if tree is not None else set()


# ------------------------------------------------------------------ data model

@dataclass(frozen=True)
class RequiredInput:
    """One input the plan depends on, and whether CODE could satisfy it."""

    cls: str
    key: str
    node_id: str = ""
    satisfied: bool = False
    reason: str = ""
    remediation: str = ""
    decision_critical: bool = True
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cls not in INPUT_CLASSES:
            raise SufficiencyError(
                f"{self.cls!r} is not an input class; known: {INPUT_CLASSES}")

    @property
    def row_key(self) -> str:
        """Stable identity, independent of wording. Feeds the U-row id."""
        return f"{self.node_id}|{self.cls}|{self.key}"

    def as_dict(self) -> dict:
        return {"cls": self.cls, "key": self.key, "node_id": self.node_id,
                "satisfied": self.satisfied, "reason": self.reason,
                "remediation": self.remediation,
                "decision_critical": self.decision_critical,
                "detail": dict(self.detail)}


@dataclass(frozen=True)
class RequiredInputManifest:
    """The six input classes for one scope, and the verdict evidence."""

    plan_hash: str
    node_id: str
    stage: str
    inputs: tuple = ()
    derivation_notes: tuple = ()

    @property
    def missing(self) -> tuple:
        return tuple(i for i in self.inputs if not i.satisfied)

    @property
    def blocking(self) -> tuple:
        return tuple(i for i in self.missing if i.decision_critical)

    @property
    def gatherable(self) -> tuple:
        return tuple(i for i in self.missing if not i.decision_critical)

    def by_class(self, cls: str) -> tuple:
        return tuple(i for i in self.inputs if i.cls == cls)

    @property
    def manifest_hash(self) -> str:
        """Stable over wording, sensitive to WHAT is required and satisfied."""
        return _sha({
            "plan_hash": self.plan_hash,
            "node_id": self.node_id,
            "inputs": sorted(
                [[i.cls, i.key, i.node_id, i.satisfied, i.decision_critical]
                 for i in self.inputs]),
        })

    def as_dict(self) -> dict:
        return {"plan_hash": self.plan_hash, "node_id": self.node_id,
                "stage": self.stage, "manifest_hash": self.manifest_hash,
                "inputs": [i.as_dict() for i in self.inputs],
                "missing": [i.as_dict() for i in self.missing],
                "derivation_notes": list(self.derivation_notes)}


@dataclass(frozen=True)
class SufficiencyResult:
    """PROCEED / GATHER(list) / BLOCK(reason), with its own evidence."""

    decision: str
    stage: str
    plan_hash: str
    node_id: str = ""
    manifest_hash: str = ""
    reason: str = ""
    missing: tuple = ()          # RequiredInput, decision-critical
    gather: tuple = ()           # RequiredInput, non-critical
    remediation: tuple = ()
    opened_rows: tuple = ()
    ledger_veto: tuple = ()      # rows ready_to_act() named, advisory-blocking
    error: str = ""              # set ONLY on a fail-open gate exception

    @property
    def proceed(self) -> bool:
        return self.decision != BLOCK

    def as_dict(self) -> dict:
        return {"decision": self.decision, "stage": self.stage,
                "plan_hash": self.plan_hash, "node_id": self.node_id,
                "manifest_hash": self.manifest_hash, "reason": self.reason,
                "missing": [i.as_dict() for i in self.missing],
                "gather": [i.as_dict() for i in self.gather],
                "remediation": list(self.remediation),
                "opened_rows": list(self.opened_rows),
                "ledger_veto": list(self.ledger_veto),
                "error": self.error}


# ------------------------------------------------------------------ derivation

def _edits_in_scope(plan: Mapping, node_id: str, node_ids: Sequence[str]):
    """``[(node_id, plan_edit), ...]`` for the requested scope.

    ``node_id == ""`` is whole-plan preflight. Node ids are the LEDGER's node
    ids, zipped to the plan's edits exactly the way ``run_plan_loop`` does it —
    same 1:1 assumption, same order, so a manifest names the node the loop will
    actually run.
    """
    edits = list(plan.get("edits") or ())
    ids = list(node_ids or ())
    if ids and len(ids) != len(edits):
        raise SufficiencyError(
            f"{len(ids)} node ids for {len(edits)} plan edits; the gate cannot "
            "name a node it cannot map to an edit")
    pairs = list(zip(ids, edits)) if ids else [("", e) for e in edits]
    if not node_id:
        return pairs
    hit = [(nid, e) for nid, e in pairs if nid == node_id]
    if not hit:
        raise SufficiencyError(f"node {node_id!r} is not in this plan")
    return hit


def _chain_edits(plan_edit: Mapping) -> list:
    out = [ce for ce in (plan_edit.get("chain_edits") or ())
           if isinstance(ce, Mapping)]
    # A milestone with no sub-plan still names its own file; treat the milestone
    # itself as the unit so the entity check is never vacuously empty.
    return out or [plan_edit]


def _declared_files(edit: Mapping) -> tuple:
    out = set()
    if edit.get("file"):
        out.add(edit["file"])
    out.update(f for f in (edit.get("allowed_files") or ()) if f)
    return tuple(sorted(out))


def _derive_entities(nid: str, plan_edit: Mapping, idx: _WorldIndex) -> list:
    """Files exist and PARSE; every symbol the edit names is really in its file."""
    out = []
    for ce in _chain_edits(plan_edit):
        for rel in _declared_files(ce):
            if not idx.has_file(rel):
                out.append(RequiredInput(
                    ENTITIES, f"file:{rel}", nid, False,
                    f"declared file {rel!r} does not exist in the world",
                    f"create {rel} or remove it from the plan's closure"))
            elif rel in idx.parse_errors:
                out.append(RequiredInput(
                    ENTITIES, f"file:{rel}", nid, False,
                    f"{rel} does not parse: {idx.parse_errors[rel]}",
                    f"fix the syntax error in {rel} before planning against it"))
            else:
                out.append(RequiredInput(ENTITIES, f"file:{rel}", nid, True,
                                         "exists and parses"))
        # The symbol the edit acts ON must be reachable in the file it acts on.
        # An edit whose subject is nowhere is not a "no-op", it is a plan
        # written against a world that does not exist.
        #
        # ONE deliberate exception, and it is not a loophole: if the edit's
        # DESTINATION name is already present, the work has already landed —
        # a re-baselined run after an acknowledged drift, or a crash-resumed
        # one, replays milestones that are already done. Charging those as a
        # missing input would make the gate refuse exactly the recovery paths
        # the ledger exists to support, so an already-landed edit is satisfied
        # and says so.
        target = ce.get("file") or ""
        new = ce.get("new_name")
        for key in ("symbol", "old_name"):
            name = ce.get(key)
            if not name or not target:
                continue
            k = f"symbol:{target}::{name}"
            if any(i.key == k for i in out):
                continue
            if target not in idx.trees:
                continue                      # already charged as a file miss
            if idx.mentions(target, name):
                out.append(RequiredInput(ENTITIES, k, nid, True,
                                         "symbol present in its declared file"))
            elif new and new != name and idx.mentions(target, new):
                out.append(RequiredInput(
                    ENTITIES, k, nid, True,
                    f"{name!r} is gone from {target} but {new!r} is present: "
                    "this edit has already landed",
                    detail={"already_landed": True}))
            else:
                out.append(RequiredInput(
                    ENTITIES, k, nid, False,
                    f"{name!r} does not appear in {target}; the edit's subject "
                    "is not in the world it plans to change",
                    f"re-derive the plan against the current {target}"))
    return out


def _derive_contracts(nid: str, plan_edit: Mapping, idx: _WorldIndex) -> list:
    """The DESTINATION of every move must exist, and every import must resolve.

    This is the check the fixture is built around: ``worker.py`` renames toward
    ``AccountRecord``, which lives in ``contracts.py`` — a file OUTSIDE the
    node's closure. If that destination is not in the world, the edit produces
    a green certificate and a broken tree.
    """
    out = []
    seen = set()
    for ce in _chain_edits(plan_edit):
        new = ce.get("new_name")
        old = ce.get("old_name")
        if new and new != old:
            k = f"defines:{new}"
            if k not in seen:
                seen.add(k)
                owners = idx.defines(new)
                if owners:
                    out.append(RequiredInput(
                        CONTRACTS, k, nid, True,
                        f"defined in {list(owners)}",
                        detail={"defined_in": list(owners)}))
                else:
                    out.append(RequiredInput(
                        CONTRACTS, k, nid, False,
                        f"rename destination {new!r} is not defined anywhere "
                        "in the world",
                        f"land the producer side of {new!r} first, or correct "
                        "the plan's new_name"))
        for rel in _declared_files(ce):
            for mod in sorted(idx.imports_of(rel)):
                if mod in _STDLIB_OK or mod in idx.modules:
                    continue
                k = f"import:{rel}->{mod}"
                if k in seen:
                    continue
                seen.add(k)
                out.append(RequiredInput(
                    CONTRACTS, k, nid, False,
                    f"{rel} imports {mod!r}, which is neither stdlib nor a "
                    "module in this world",
                    f"add {mod}.py to the world or drop the import"))
    # Every import that DID resolve is recorded as satisfied, so the manifest
    # hash moves when the world's seam shape moves — not only when it breaks.
    for ce in _chain_edits(plan_edit):
        for rel in _declared_files(ce):
            for mod in sorted(idx.imports_of(rel)):
                k = f"import:{rel}->{mod}"
                if k in seen or mod in _STDLIB_OK:
                    continue
                seen.add(k)
                out.append(RequiredInput(CONTRACTS, k, nid, True,
                                         f"{mod} resolves in this world"))
    return out


def _derive_invariants(nid: str, plan_edit: Mapping, adh_plan, milestone_id: str
                       ) -> list:
    """The spine must be executable predicates; the milestone must be a gate."""
    out = []
    spine = tuple(getattr(adh_plan, "spine", ()) or ())
    if not spine:
        out.append(RequiredInput(
            INVARIANTS, "spine", nid, False,
            "the plan declares no spine predicates; nothing is fenced off",
            "register the plan with a spine (see plan_run.default_spine)"))
    for pred in spine:
        ok = bool(getattr(pred, "args", ()))
        out.append(RequiredInput(
            INVARIANTS, f"spine:{getattr(pred, 'name', '?')}", nid, ok,
            "executable predicate with arguments" if ok else
            f"spine predicate {getattr(pred, 'name', '?')!r} has no arguments "
            "and can never fire",
            "" if ok else "give the predicate the paths/families it fences"))
    ms = getattr(adh_plan, "milestone_index", {}) or {}
    milestone = ms.get(milestone_id)
    if milestone is None:
        out.append(RequiredInput(
            INVARIANTS, f"milestone:{milestone_id}", nid, False,
            f"no registered milestone {milestone_id!r} for this node",
            "register an executable milestone for the node"))
        return out
    accs = tuple(getattr(milestone, "acceptance", ()) or ())
    executable = tuple(a for a in accs
                       if getattr(a, "kind", None) in _adh.ACCEPTANCE_KINDS)
    out.append(RequiredInput(
        INVARIANTS, f"milestone:{milestone_id}", nid, bool(executable),
        f"{len(executable)} executable acceptance predicate(s)" if executable
        else f"milestone {milestone_id!r} has no executable acceptance",
        "" if executable else "replace the prose milestone with a predicate",
        detail={"kinds": sorted({getattr(a, "kind", "") for a in accs})}))
    return out


def _derive_external_seams(nid: str, plan_edit: Mapping, plan: Mapping) -> list:
    """Declared external seams only. Nothing is INFERRED from prose.

    A plan that declares no external seam requires none — that is honest for a
    pure in-tree refactor. A plan that DOES declare one must show that it can
    probe it, repeat it safely, and get a receipt back, or an unattended loop
    has no business crossing it.
    """
    out = []
    declared = list(plan_edit.get("external_seams") or ()) + \
        list(plan.get("external_seams") or ())
    for seam in declared:
        if not isinstance(seam, Mapping):
            out.append(RequiredInput(
                EXTERNAL_SEAMS, f"malformed:{seam!r}"[:120], nid, False,
                "external seam declaration is not a mapping",
                "declare seams as {id, probe, idempotency, receipt}"))
            continue
        sid = seam.get("id") or seam.get("name") or ""
        if not sid:
            out.append(RequiredInput(
                EXTERNAL_SEAMS, "unnamed", nid, False,
                "external seam declaration has no id",
                "give the seam an id"))
            continue
        gaps = [c for c in REQUIRED_SEAM_CAPABILITIES if not seam.get(c)]
        out.append(RequiredInput(
            EXTERNAL_SEAMS, f"seam:{sid}", nid, not gaps,
            "declares probe, idempotency and receipt" if not gaps
            else f"seam {sid!r} is missing capability {gaps}",
            "" if gaps == [] else
            f"declare {gaps} for {sid} or take the seam out of an "
            "unattended plan",
            detail={"missing_capabilities": gaps}))
    return out


def _derive_evidence(nid: str, plan_edit: Mapping, plan: Mapping,
                     evidence: Mapping, idx: _WorldIndex) -> list:
    """Every declared evidence ref must RESOLVE and match its content hash."""
    out = []
    refs = list(plan_edit.get("evidence_refs") or ()) + \
        list(plan.get("evidence_refs") or ())
    for ref in refs:
        if isinstance(ref, Mapping):
            rid = ref.get("id") or ref.get("ref") or ""
            want = ref.get("content_hash") or ""
            covers = ref.get("covers") or ""
            advisory = bool(ref.get("advisory"))
        else:
            rid, want, covers, advisory = str(ref), "", "", False
        critical = not advisory
        if not rid:
            out.append(RequiredInput(
                EVIDENCE, "unnamed", nid, False,
                "evidence ref has no id", "give the evidence ref an id",
                decision_critical=critical))
            continue
        got = (evidence or {}).get(rid)
        if got is None:
            out.append(RequiredInput(
                EVIDENCE, f"ref:{rid}", nid, False,
                f"evidence ref {rid!r} does not resolve",
                f"supply evidence[{rid!r}] or drop the reference",
                decision_critical=critical))
            continue
        content = got.get("content") if isinstance(got, Mapping) else str(got)
        if content is None and isinstance(got, Mapping) and got.get("file"):
            content = idx.world.get(got["file"])
        if content is None:
            out.append(RequiredInput(
                EVIDENCE, f"ref:{rid}", nid, False,
                f"evidence {rid!r} resolves to nothing readable",
                "point the evidence at real content",
                decision_critical=critical))
            continue
        actual = sha256_text(content)
        if want and want != actual:
            out.append(RequiredInput(
                EVIDENCE, f"ref:{rid}", nid, False,
                f"evidence {rid!r} content hash mismatch: plan says "
                f"{want[:12]}…, content is {actual[:12]}…",
                "re-gather the evidence; it is stale",
                decision_critical=critical,
                detail={"expected": want, "actual": actual}))
            continue
        if covers and covers not in _canonical(plan_edit):
            out.append(RequiredInput(
                EVIDENCE, f"ref:{rid}", nid, False,
                f"evidence {rid!r} claims to cover {covers!r}, which this "
                "node does not name",
                "attach the evidence to the node it actually covers",
                decision_critical=critical))
            continue
        out.append(RequiredInput(EVIDENCE, f"ref:{rid}", nid, True,
                                 "resolves and matches its content hash"))
    return out


def _derive_oracle(nid: str, plan_edit: Mapping, adh_plan, milestone_id: str,
                   holdout_dir, idx: _WorldIndex, oracles: Mapping) -> list:
    """An INDEPENDENT executable oracle per milestone, or nothing runs.

    Independent means three things at once, all checked: it is a ``tests_pass``
    suite (not the executor's own claim about itself), it exists on disk under
    the held-out root, and it is NOT a file in the mutable world — a suite the
    plan could rewrite is a suite the plan grades itself with.
    """
    out = []
    ms = getattr(adh_plan, "milestone_index", {}) or {}
    milestone = ms.get(milestone_id)
    if milestone is None:
        return [RequiredInput(
            ORACLE, f"milestone:{milestone_id}", nid, False,
            f"no milestone {milestone_id!r}, therefore no oracle",
            "register an executable milestone with a held-out suite")]
    suites = [a.args.get("suite") for a in getattr(milestone, "acceptance", ())
              if getattr(a, "kind", None) == _adh.TESTS_PASS
              and isinstance(getattr(a, "args", None), Mapping)]
    suites = [s for s in suites if s]
    if not suites:
        # GATHER, not BLOCK, and the reason is a boundary rather than leniency:
        # ``plan_run``'s witness already fails a non-``tests_pass`` acceptance
        # closed ("not independently executable"), so blocking here would only
        # duplicate a gate that works, and would change a proven behaviour of a
        # module this one does not own. The gate's job is the inputs the loop
        # CANNOT discover for itself.
        return [RequiredInput(
            ORACLE, f"milestone:{milestone_id}", nid, False,
            f"milestone {milestone_id!r} has no tests_pass acceptance; the "
            "witness will fail it closed rather than judge it",
            "give the milestone a held-out suite",
            decision_critical=False)]
    for suite in suites:
        k = f"suite:{milestone_id}:{suite}"
        override = (oracles or {}).get(suite) or (oracles or {}).get(k)
        if isinstance(override, Mapping) and override.get("registered") is False:
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                f"oracle {suite!r} is explicitly not registered",
                "register the oracle"))
            continue
        if suite in idx.world:
            # A decoy verifier sitting in the mutable world. Reported, never
            # blocking: ``witness_milestone`` resolves the suite from the
            # held-out package by path, so the decoy is already inert, and
            # ``test_plan_integration`` proves that. Escalating it here would
            # turn a DEFEATED attack into an outage.
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                f"a file named {suite!r} also exists in the mutable world; the "
                "witness ignores it, but its presence is suspicious",
                "delete the decoy suite from the world",
                decision_critical=False))
            continue
        if holdout_dir is None:
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                "no held-out root supplied; the oracle cannot be located. "
                "'We could not check' is never 'it passed'.",
                "pass holdout_dir to the gate"))
            continue
        p = Path(holdout_dir) / suite
        if not p.is_file():
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                f"held-out suite {suite!r} not found under {holdout_dir}",
                f"add {suite} to the held-out root"))
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except OSError as exc:
            out.append(RequiredInput(
                ORACLE, k, nid, False, f"held-out suite {suite!r} unreadable: {exc}",
                "fix the suite's permissions"))
            continue
        tree, err = _parse(body)
        if tree is None:
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                f"held-out suite {suite!r} does not parse: {err}",
                "fix the suite"))
            continue
        tests = [n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name.startswith("test")]
        if not tests:
            out.append(RequiredInput(
                ORACLE, k, nid, False,
                f"held-out suite {suite!r} collects no tests; a suite that "
                "cannot fail is not an oracle",
                "write at least one test in the suite"))
            continue
        out.append(RequiredInput(ORACLE, k, nid, True,
                                 f"{len(tests)} held-out test(s), outside the "
                                 "mutable world",
                                 detail={"tests": sorted(tests)}))
    return out


def derive_required_inputs(plan: Mapping, *, adh_plan, node_id: str = "",
                           node_ids: Sequence[str] = (),
                           world: Mapping[str, str] = None,
                           evidence: Mapping = None,
                           oracles: Mapping = None,
                           holdout_dir: str | os.PathLike = None,
                           stage: str = PLAN_PREFLIGHT,
                           seam_manifest: Callable | None = None,
                           ) -> RequiredInputManifest:
    """Compute the six required-input classes for a plan, or for one node.

    ``plan`` is the legacy plan MAPPING (see deviation 1 in the module
    docstring); ``adh_plan`` is the :class:`adherence.Plan` that owns the
    invariants and milestone identity. ``node_id=""`` derives the whole-plan
    preflight manifest.
    """
    if stage not in STAGES:
        raise SufficiencyError(f"{stage!r} is not a stage; known: {STAGES}")
    idx = _WorldIndex(world if world is not None else {})
    notes = []
    if seam_manifest is not None:
        # Honest about what an injected provider buys: it replaces the AST
        # fallback for symbol resolution only. Nothing else changes shape.
        try:
            extra = seam_manifest(idx.world) or {}
            for rel, names in (extra.get("definitions") or {}).items():
                idx._defs.setdefault(rel, set()).update(names)
                idx._names.setdefault(rel, set()).update(names)
            notes.append("seam_manifest provider merged")
        except Exception as exc:              # a bad provider must not decide
            notes.append(f"seam_manifest provider failed: {exc!r}")
            log.warning("sufficiency: seam_manifest provider failed: %r", exc)

    inputs: list = []
    for nid, plan_edit in _edits_in_scope(plan, node_id, node_ids):
        if not isinstance(plan_edit, Mapping):
            raise SufficiencyError("plan edits must be mappings")
        mid = plan_edit.get("milestone_id") or plan_edit.get("id") or ""
        inputs += _derive_entities(nid, plan_edit, idx)
        inputs += _derive_contracts(nid, plan_edit, idx)
        inputs += _derive_invariants(nid, plan_edit, adh_plan, mid)
        inputs += _derive_external_seams(nid, plan_edit, plan)
        inputs += _derive_evidence(nid, plan_edit, plan, evidence or {}, idx)
        inputs += _derive_oracle(nid, plan_edit, adh_plan, mid, holdout_dir,
                                 idx, oracles or {})

    # Deduplicate on (node, class, key) — the same spine predicate is required
    # by every node and must be charged once, not once per node.
    seen: dict = {}
    for i in inputs:
        prev = seen.get(i.row_key)
        # An unsatisfied finding always wins over a satisfied duplicate.
        if prev is None or (prev.satisfied and not i.satisfied):
            seen[i.row_key] = i
    ordered = tuple(sorted(seen.values(),
                           key=lambda i: (i.node_id, INPUT_CLASSES.index(i.cls),
                                          i.key)))
    return RequiredInputManifest(
        plan_hash=str(getattr(adh_plan, "plan_hash", "")),
        node_id=node_id, stage=stage, inputs=ordered,
        derivation_notes=tuple(notes))


# ------------------------------------------------------------------ evaluation

def _row_id(manifest: RequiredInputManifest, item: RequiredInput) -> str:
    """Deterministic, so re-evaluating a manifest never duplicates a row."""
    h = hashlib.sha256(
        f"{manifest.plan_hash}|{item.row_key}".encode("utf-8")).hexdigest()[:16]
    return f"suff:{item.cls}:{h}"


def charge_missing_inputs(manifest: RequiredInputManifest, ledger) -> tuple:
    """Open one U-row per missing input. CHARGES the uncertainty; waives nothing.

    Idempotent by construction: the row id is a hash of (plan_hash, node,
    class, key), so a second evaluation of the same manifest finds its rows
    already on the ledger and adds nothing. Rows are never closed here — the
    gate has no closure criterion to offer, and inventing one would be exactly
    the ``model_said_so`` closure the ledger refuses to have.
    """
    if ledger is None:
        return ()
    opened = []
    for item in manifest.missing:
        rid = _row_id(manifest, item)
        if rid in getattr(ledger, "rows", {}):
            continue
        try:
            ledger.open(_u.URow(
                id=rid,
                utype=_UTYPE_FOR_CLASS[item.cls],
                claim=f"[{item.cls}] {item.key}: {item.reason}",
                severity=0.9 if item.decision_critical else 0.4,
                irreversibility=0.0,     # a missing input is not yet damage
                u=1.0,
                correlation_group=f"sufficiency:{manifest.plan_hash[:12]}",
                owner="detreason.sufficiency",
                source="sufficiency_gate",
                reduction_actions=[_u.ReductionAction(
                    action=item.remediation or f"supply {item.key}",
                    cost_class="cheap", expected_du=0.9)]))
            opened.append(rid)
        except Exception as exc:   # ledger trouble must never grant passage
            log.warning("sufficiency: could not charge %s: %r", rid, exc)
    return tuple(opened)


def evaluate_required_inputs(manifest: RequiredInputManifest, *,
                             ledger=None, stage: str = None
                             ) -> SufficiencyResult:
    """Turn a manifest into PROCEED / GATHER / BLOCK.

    The ledger is consulted in ONE direction only. It may add a veto; it can
    never subtract a missing input. Read the order below as a contract: the
    blocking set is computed from the MANIFEST, before the ledger is touched,
    and the ledger's answer is only ever ANDed in.
    """
    stage = stage or manifest.stage
    blocking = manifest.blocking
    gather = manifest.gatherable
    opened = charge_missing_inputs(manifest, ledger)

    veto: tuple = ()
    if ledger is not None:
        try:
            ready, blockers = ledger.ready_to_act()
            if not ready:
                veto = tuple(blockers or ())
        except Exception as exc:
            log.warning("sufficiency: ready_to_act failed: %r", exc)

    if blocking:
        by_class: dict = {}
        for i in blocking:
            by_class.setdefault(i.cls, []).append(i.key)
        reason = ("required inputs are missing: " + "; ".join(
            f"{c}={sorted(ks)}" for c, ks in sorted(by_class.items())))
        return SufficiencyResult(
            BLOCK, stage, manifest.plan_hash, manifest.node_id,
            manifest.manifest_hash, reason, blocking, gather,
            tuple(sorted({i.remediation for i in blocking if i.remediation})),
            opened, veto)
    if veto:
        # Not a missing input — an uncertainty exposure the ledger's own
        # stopping rule refuses to act under. Still a BLOCK, named differently
        # so the two causes are never confused in a post-mortem.
        return SufficiencyResult(
            BLOCK, stage, manifest.plan_hash, manifest.node_id,
            manifest.manifest_hash,
            f"U-ledger is not ready to act; blocking rows: {list(veto)}",
            (), gather, (), opened, veto)
    if gather:
        return SufficiencyResult(
            GATHER, stage, manifest.plan_hash, manifest.node_id,
            manifest.manifest_hash,
            f"{len(gather)} non-critical input(s) outstanding",
            (), gather,
            tuple(sorted({i.remediation for i in gather if i.remediation})),
            opened, veto)
    return SufficiencyResult(PROCEED, stage, manifest.plan_hash,
                             manifest.node_id, manifest.manifest_hash,
                             "every required input is satisfied by code",
                             (), (), (), opened, veto)


def check_sufficiency(plan: Mapping, *, adh_plan, node_ids: Sequence[str] = (),
                      node_id: str = "", world: Mapping[str, str] = None,
                      evidence: Mapping = None, oracles: Mapping = None,
                      holdout_dir=None, ledger=None,
                      stage: str = PLAN_PREFLIGHT,
                      seam_manifest: Callable | None = None,
                      ) -> tuple:
    """``(manifest, result)`` in one call. The loop's single entry point."""
    manifest = derive_required_inputs(
        plan, adh_plan=adh_plan, node_id=node_id, node_ids=node_ids,
        world=world, evidence=evidence, oracles=oracles,
        holdout_dir=holdout_dir, stage=stage, seam_manifest=seam_manifest)
    return manifest, evaluate_required_inputs(manifest, ledger=ledger,
                                              stage=stage)
