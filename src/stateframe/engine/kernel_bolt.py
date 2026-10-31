"""Coordinate deterministic components around an iterative coding worker.

Components supply facts, preserve task state, and reject premature completion.
They do not select or apply repairs. Each records whether it engaged, skipped
work, or failed, together with the reason, so integration faults remain visible.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys
import traceback
from collections.abc import Callable, Mapping

import kernel_effect_boundary as keb
import kernel_workspace_truth as _workspace_truth
from kernel_liveness import Evidence, OrganCensusSpec, TriState
from kernel_verdict import SATISFIED
from kernel_history_spool import history as _history, journal_directory as _journal_directory
from kernel_history_receipts import (
    append_receipt as _append_receipt,
    new_receipt as _new_receipt,
    receipts as _receipt_history,
)

POC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(POC))


class KernelBolt:
    """All cylinders, per-task. Every method is best-effort and self-reporting."""

    def __init__(self, root: pathlib.Path, fn: str, task_id: str) -> None:
        self.root = root
        self.fn = str(fn or "")
        self.target_path: str | None = None
        self.task_id = task_id
        import uuid
        self.run_id = uuid.uuid4().hex
        self.trace = _history()
        self.status: dict[str, str] = {}          # cylinder -> engaged/skipped/error:reason
        self.seen: dict[str, str] = {}            # command -> first observation (ledger)
        self._seen_query_steps: dict[str, int] = {}
        self._repeat_counts: dict[str, int] = {}  # command -> times issued (for escalation)
        # Which objections have already been raised. Every gate here challenges AT MOST ONCE:
        # an unbounded refusal is a trap, and we have shipped three of those already.
        self._challenged: dict[str, bool] = {}
        self.repeats = 0
        self.edits = 0
        self.last_test_exit: int | None = None
        self._last_relevant_match = True
        self.oracle_preflight: Mapping[str, object] = {}
        self._reg = None
        self._graph = None
        self._boot()

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        """Start every cylinder. A failure to start is RECORDED, never swallowed."""
        try:
            from symbol_registry import SymbolRegistry
            self._reg = SymbolRegistry().build(self.root)
            self.status["symbol"] = (f"engaged ({len(self._reg.defs)} symbols, "
                                     f"{self._reg.files_scanned} files)")
        except Exception as e:                                       # noqa: BLE001
            self.status["symbol"] = f"error: {type(e).__name__}: {e}"

        # work_ledger's real API is manifest/plan-hash oriented (canonical_json, plan_hash,
        # sha256_ref) and expects a plan object we do not have in a free-form agent loop. We
        # take its BEHAVIOUR — remember what was done, never redo it blind — with its own
        # hashing, and record honestly that the module itself is not driving.
        try:
            from detreason import work_ledger as _wl
            self._wl = _wl
            self.status["ledger"] = "engaged (behaviour; module imported for hashing)"
        except Exception as e:                                       # noqa: BLE001
            self._wl = None
            self.status["ledger"] = f"degraded: {type(e).__name__}: {e}"

        # micrograph's hypothesis surface lives on HypothesisGraphConsumer, NOT on MicroGraph —
        # MicroGraph is the raw node/edge store (add_node/add_edge) and has no add_entity at
        # all. My first guess (`MicroGraph().add_entity`) failed at runtime, which is exactly
        # what turning every cylinder on at once is for: the body named itself in one run.
        micrograph_setting = os.environ.get("STATEFRAME_MICROGRAPH", "off").strip().lower()
        if micrograph_setting != "off":
            from kernel_micrograph_wire import mode as micrograph_mode
            micrograph_setting = micrograph_mode()
        if micrograph_setting == "on":
            self._mg = None
            self._graph = None
            self.status["graph"] = "routed"
        else:
            try:
                from detreason import micrograph as _mg
                self._mg = _mg
                self._graph = _mg.HypothesisGraphConsumer()
                self.status["graph"] = "engaged (HypothesisGraphConsumer)"
            except Exception as e:                                   # noqa: BLE001
                self._mg = None
                self._graph = None
                self.status["graph"] = f"error: {type(e).__name__}: {e}"

        for name, mod in (("gate", "obligations"), ("skeptic", "skeptic")):
            try:
                __import__(f"detreason.{mod}")
                self.status[name] = "engaged (behaviour; module importable)"
            except Exception as e:                                   # noqa: BLE001
                self.status[name] = f"degraded: {type(e).__name__}: {e}"

        # Liveness lifecycle records are deliberately on their own cylinder so the
        # historical report still distinguishes an engaged-but-never-fired organ.
        for organ, status in self.status.items():
            self._rec(
                "liveness",
                "ORGAN_ARMED" if status.startswith("engaged") else "ORGAN_DISARMED",
                f"organ=bolt.{organ} status={status[:160]}",
            )

    def _target_label(self) -> str:
        """Render an explicit unlocalized state instead of inventing a path."""
        return str(getattr(self, "target_path", "") or "<unlocalized>")

    # ------------------------------------------------------- cylinder 1+2
    def observe(self, cmd: str, obs: str, *, step: int | None = None) -> str:
        """Every command passes through here. Returns a possibly-augmented observation."""
        from kernel_intervention_policy import interventions_enabled
        if not interventions_enabled():
            return obs
        key = " ".join(cmd.split())
        out = obs

        # ONLY QUERIES CAN REPEAT MEANINGLESSLY. `edit` and `test` are ACTIONS whose payload
        # and world-state change between issues, so an identical command line is not an
        # identical operation.
        #
        # Caught before this ever ran against a task: the watcher flagged
        # `edit app/routers/helm_human_work.py` issued NINE times and I read it as thrashing.
        # It was `_ack_decision` iterating — nine DIFFERENT patches to the same file, each a
        # distinct python block, which is precisely the act-observe-revise behaviour the whole
        # loop exists to produce. This cylinder would have blocked patch #2 onward and killed
        # it. A memory that blocks progress is worse than no memory.
        if cmd.split()[0] in ("edit", "test", "done"):
            # AN EDIT CHANGES THE WORLD, SO EVERY PRIOR QUERY ANSWER IS STALE.
            # the audit (trap_grading) demonstrated the trap: `grep x` -> "old", then an
            # edit, then `grep x` -> we hand back "old" and tell the agent repeating it
            # "cannot produce new information". After a write it absolutely can — that is how
            # you confirm a patch landed. Blocking it is the same class as the `done` trap:
            # the kernel suppressing exactly the behaviour the loop exists to produce.
            if cmd.split()[0] == "edit":
                self.seen.clear()
                self._seen_query_steps.clear()
                self._repeat_counts.clear()
                self._rec("ledger", "cleared_on_edit", "world changed")
            return out

        # Explicit reads are evidence requests. Keep the actual freshly executed
        # bytes in the response on every repeat; an advisory may follow them, but
        # a bounded reminder must never replace source content.
        if cmd.split()[0] == "read":
            from kernel_stateframe_m3 import m3_working_context_enabled
            if m3_working_context_enabled():
                previous = self.seen.get(key)
                if previous == obs and key in self.seen:
                    self.repeats += 1
                    self._repeat_counts[key] = self._repeat_counts.get(key, 1) + 1
                    prior_step = self._seen_query_steps.get(key)
                    note = (f"[kernel] unchanged since step {prior_step}; full current bytes are "
                            "included above." if prior_step is not None else
                            "[kernel] unchanged since the prior read; full current bytes are "
                            "included above.")
                    self._rec("ledger", "repeat_read_delivered", f"{key[:160]} (full bytes)")
                    return f"{obs}\n{note}" if obs else note
                if key in self.seen and previous != obs:
                    self._repeat_counts.pop(key, None)
                    self._rec("ledger", "observation_changed", key[:160])
                self.seen[key] = obs
                self._repeat_counts.setdefault(key, 1)
                if type(step) is int and step >= 1:
                    self._seen_query_steps[key] = step
                return obs

        from kernel_gate_switches import repeat_gate_enabled
        if not repeat_gate_enabled():
            # Ablation switch: fresh observations always pass through unchanged.
            return out

        # CYLINDER 2 — negative work, on QUERIES only. Measured live: an agent re-ran an
        # identical grep it had already run three steps earlier. Handing back the previous
        # answer costs nothing and cannot mislead — it is literally what the repo said before.
        # The caller already executed this query. Its fresh observation outranks
        # cached text, including changes made outside the edit tool.
        if key in self.seen and self.seen[key] != obs:
            self.seen.pop(key)
            self._repeat_counts.pop(key, None)
            self._rec("ledger", "observation_changed", key)
        if key in self.seen:
            self.repeats += 1
            # ESCALATE, NEVER REPEAT VERBATIM (defect #37). At temperature 0 a byte-identical
            # replay is a FIXED POINT: same input -> same output -> same query, forever.
            # Measured live (v13 _ack_decision): 10+ consecutive `grep get_setting`, each
            # answered with the identical "[kernel] You already ran..." — the kernel's own
            # static reply was the loop's fuel. Varying the message (count + a concrete next
            # action) changes the model's input every step, which is what breaks determinism.
            n = self._repeat_counts[key] = self._repeat_counts.get(key, 1) + 1
            nudge = ("Try a different command."
                     if n <= 3 else
                     f"You have now issued this identical query {n} times. The answer cannot "
                     f"change until you EDIT something. Choose a DIFFERENT action now: `read` "
                     f"a file you have not read, `edit` the function, or `test`.")
            out = (f"[kernel] (repeat #{n}) You already ran `{key}` and got:\n"
                   f"{self.seen[key][:400]}\n{nudge}")
            self._rec("ledger", "repeat_blocked", f"{key} (#{n})")
            return out
        self.seen[key] = obs

        # CYLINDER 1 — symbol ground truth on a dead search.
        # E306: this NEVER FIRED in production — it watched the literal
        # "(no matches)" while the loop's actual empty-grep marker is "(no grep matches)",
        # and presentation text is the wrong trigger anyway. The loop now sets
        # self.last_grep_hits STRUCTURALLY before calling observe; fire on hit_count == 0,
        # accept either legacy marker as fallback.
        _dead_grep = (getattr(self, "last_grep_hits", None) == 0
                      or "(no matches)" in obs or "(no grep matches)" in obs)
        if self._reg is not None and cmd.startswith("grep") and _dead_grep:
            if _workspace_truth.enabled():
                query = re.sub(r"^grep\s+", "", cmd).strip().strip("\"'`")
                if query:
                    probe = "invalid-grep-command" if query.startswith("-") else "literal"
                    fact = _workspace_truth.probe_live_workspace(
                        self.root, probe, query, journal=self,
                    )
                    out = obs + "\n[kernel] " + _workspace_truth.render_fact(fact)
                    self._rec("symbol", "answered_dead_grep", query[:160])
            else:
                tok = re.sub(r"^grep\s+", "", cmd)
                tok = re.sub(r"^(def|class|async def)\s+", "", tok.strip("\"'`"))
                tok = re.split(r"[^\w]", tok)[0] if tok else ""
                if tok:
                    out = obs + "\n[kernel] " + self._reg.answer(tok)
                    self._rec("symbol", "answered_dead_grep", tok)
        return out

    # --------------------------------------------------------- cylinder 4
    def note(self, kind: str, text: str) -> None:
        """Record a hypothesis/evidence node — state a small model cannot carry 36 steps."""
        micrograph_setting = os.environ.get("STATEFRAME_MICROGRAPH", "off").strip().lower()
        if micrograph_setting != "off":
            from kernel_micrograph_wire import mode as micrograph_mode
            micrograph_setting = micrograph_mode()
        if micrograph_setting == "on":
            # Model prose cannot enter the typed MicroGraph route.
            return
        if self._graph is None:
            return
        try:
            # Real API: add_hypothesis accepts a mapping with hypothesis_id + claim.
            hid = f"{self.task_id}:{kind}:{len(self.trace)}"
            self._graph.add_hypothesis({"hypothesis_id": hid, "claim": text[:400]})
            self._rec("graph", kind, text[:80])
        except Exception as e:                                       # noqa: BLE001
            self.status["graph"] = f"error at runtime: {type(e).__name__}: {e}"

    # ------------------------------------------------------- cylinder 3+5
    def may_finish(self) -> tuple[bool, str]:
        """Is `done` legal right now? The ONLY veto in the whole bolt-on."""
        from kernel_intervention_policy import interventions_enabled
        if not interventions_enabled():
            return True, ""
        # CHALLENGE ONCE, THEN YIELD. adversarial review (inst_antipattern):
        #
        #   "Hard zero-edit and red-test gates encourage dummy edits and pathological loops.
        #    A no-edit completion can be valid when the issue is already fixed, unreproducible,
        #    external, or diagnostic rather than mutative. Both conditions deserve a challenge,
        #    not an absolute refusal."
        #
        # My gate was an UNBOUNDED refusal: an agent that legitimately had nothing to edit
        # could never finish. That is the third inescapable-loop bug in this file, after the
        # exit-4 gate and the stale-query ledger — all three the kernel suppressing behaviour
        # rather than supporting it. Governance here is now bounded and reversible: state the
        # objection once, and if the agent still wants to stop, it has answered.
        # E307 completion_evidence: edits that touched ONLY test files
        # cannot complete a product-bug task — the article run changed tests while leaving
        # production untouched and declared success. One bounded challenge.
        # E322: the gate CONSUMES relevance state — a green that
        # executed no relevant regression node cannot silently close the task. One
        # bounded challenge, then yield (article-v3 finished on exactly this false green).
        # E339 (DoD live, dod_contract.md bounded-challenge): a FIXED claim whose
        # receipts don't support it gets ONE challenge naming the exact missing
        # evidence — then yields (the verdict still stamps UNCERTIFIED; credit is
        # earned, exit is free). Only fires on an explicit FIXED-family claim.
        if not self._challenged.get("dod_evidence"):
            _claim = (getattr(self, "last_done_summary", "") or "").lower()
            if any(w in _claim for w in ("fixed", "resolved", "now correctly",
                                         "corrected")):
                v = self.completion_verdict()
                if not v["success_certified"]:
                    missing = [label for ok, label in (
                        (v["product_edit"], "an APPLIED product-code edit"),
                        (v["red_green_flip"], "a red->green flip on a test"),
                        (v["oracle_untainted"], "an untainted oracle (you weakened a "
                                                "failing test)"),
                        (v["relevant_green"], "a green on a RELEVANT regression node"),
                    ) if not ok]
                    self._challenged["dod_evidence"] = True
                    self._rec("gate", "challenged_done", f"dod missing={len(missing)}")
                    if (not v["red_green_flip"]
                            and self._preflight_proves_oracle()
                            and not self._whole_relevant_oracle_green()):
                        whole_green_files = self._whole_green_oracle_files()
                        oracle_file = next(
                            (
                                file_name
                                for file_name in self._oracle_files()
                                if file_name not in whole_green_files
                                and not self._single_selector_oracle_green(file_name)
                            ),
                            "",
                        )
                        if oracle_file:
                            return False, (
                                f"Run the whole oracle file `{oracle_file}` green, then "
                                "issue `done`. The recorded evidence is missing: "
                                + "; ".join(missing) + ". Restate your outcome honestly "
                                "(e.g. 'unverified' or 'cannot reproduce, because ...') "
                                "if that evidence cannot be produced."
                            )
                    # E386: when the world holds NO discriminating test for this fn,
                    # "produce a red->green flip" is an impossible demand — tell the
                    # model the LEGAL way to make one (write its own repro test; the
                    # self-oracle taint tracker E325 keeps that honest, and grading
                    # restores pristine tests so it cannot game the grade).
                    _blind = ""
                    if not getattr(self, "f2p_inworld", ()):
                        _blind = (" NOTE: no existing test covers this function — "
                                  "WRITE a minimal reproduction test that exercises "
                                  "the buggy behaviour, confirm it FAILS on the "
                                  "original code path, then confirm it PASSES after "
                                  "your fix. That flip is the missing evidence.")
                    return False, ("[kernel] You claim the bug is FIXED, but the "
                                   "recorded evidence is missing: "
                                   + "; ".join(missing) + "." + _blind +
                                   " Produce the missing evidence, or restate your "
                                   "outcome honestly (e.g. 'unverified' or 'cannot "
                                   "reproduce, because ...') and issue `done` again.")
        # E393: unresolved CONTRADICTED contract atoms join the done gate — one bounded
        # challenge naming them (the nit-picker tracks its own challenged flag).
        _nit = getattr(self, "nit", None)
        if _nit is not None:
            try:
                _nchal = _nit.done_challenge()
            except Exception:                                        # noqa: BLE001
                _nchal = None
            if _nchal:
                self._rec("nitpicker", "challenged_done", "contradicted atoms open")
                return False, _nchal
        # E325: a done whose only green evidence is a self-authored test weakened after
        # a red gets one bounded challenge.
        if (getattr(self, "_self_oracle_tainted", False)
                and not self._challenged.get("self_oracle_done")):
            self._challenged["self_oracle_done"] = True
            self._rec("gate", "challenged_done", "self-oracle tainted")
            return False, ("[kernel] Your green evidence comes from a repro test you "
                           "edited AFTER it failed — that green does not verify the "
                           "repair. Restore the original assertion and make it pass, "
                           "run a pre-existing relevant test, or state why the "
                           "assertion was wrong. Then finish.")
        if (getattr(self, "_last_relevant_match", True) is False
                and not self._challenged.get("irrelevant_green_done")):
            self._challenged["irrelevant_green_done"] = True
            self._rec("gate", "challenged_done", "green-but-irrelevant")
            inworld = getattr(self, "f2p_inworld", ())
            # E341 harmonized with E335 (require-k autopsy: the old wording sent the
            # model to a masked green): the node is necessary, the bug report is
            # sufficient — say both in one breath.
            hint = (" The relevant node is: " + "; ".join(sorted(inworld)[:2]) +
                    " — run it, AND remember this checkout's copy may be weaker than "
                    "the grading version: your change must also satisfy the failure "
                    "description in the bug report."
                    if inworld else "")
            if inworld:
                relevant_node = sorted(inworld)[0]
                return False, (
                    f"Run the relevant selector `{relevant_node}` green, then issue "
                    "`done`. The latest green run executed NONE of the relevant "
                    "regression nodes, so it does not verify this repair."
                    + hint
                )
            return False, ("[kernel] Your latest green run executed NONE of the relevant "
                           "regression nodes, so it does not verify this repair." + hint +
                           " Then finish (or state why you cannot).")
        if (self.edits > 0 and getattr(self, "product_edits", None) == 0
                and not self._challenged.get("test_only_edits")):
            self._challenged["test_only_edits"] = True
            self._rec("skeptic", "challenged_done", "test-only edits")
            return False, ("[kernel] Every edit you made touched TEST files; the production "
                           "code is unchanged, so the bug is still present. Edit the product "
                           "function, or state why no product change is needed and issue "
                           "`done` again.")
        if self.edits == 0 and not self._challenged.get("zero_edits"):
            self._challenged["zero_edits"] = True
            self._rec("gate", "challenged_done", "zero edits (once)")
            # E333 diagnosis-to-action bridge (ack, 31 steps, 0 edits): the model
            # correctly NAMED the defect in its done-summary and treated the diagnosis
            # as the deliverable. Quote its own claim back and demand the edit.
            _claim = (getattr(self, "last_done_summary", "") or "")[:220]
            _diagnosed = any(w in _claim.lower() for w in
                             ("caused by", "mismatch", "missing", "incorrect", "bug is",
                              "should", "wrong"))
            if _diagnosed:
                return False, ("[kernel] You have not edited anything, yet your own "
                               f"summary DIAGNOSES a concrete defect: \"{_claim}\". A "
                               "diagnosis is not a repair — APPLY the edit your summary "
                               "describes (str_replace on the product code), run the "
                               "test, then finish. If you now believe no edit is needed, "
                               "say why and issue `done` again.")
            return False, ("[kernel] You have not edited anything, so `done` would submit the "
                           "original bug unchanged. If you believe no edit is needed, say why "
                           "and issue `done` again — it will be accepted.")
        # ONLY A REAL TEST FAILURE MAY BLOCK `done`.
        #
        # pytest exit 1 = tests ran and failed (real evidence against the patch).
        # exit 4/5      = usage error / nothing collected — NO ORACLE EXISTS. Those say
        #                 nothing about the patch, and blocking on them is inescapable: the
        #                 agent can never produce a green run, so `done` is refused forever.
        #
        # Measured live within minutes of wiring this in: 6 consecutive "done REFUSED by
        # kernel" on exit 4, trapping the agent in a loop with no exit. A gate that cannot be
        # satisfied is worse than no gate, and this is exactly the "cylinder harms a working
        # agent" failure identified in review. Challenge only on 1, 2 and 3.
        if self.last_test_exit in (1, 2, 3) and not self._challenged.get("red_test"):
            self._challenged["red_test"] = True
            self._rec("skeptic", "challenged_done", f"exit {self.last_test_exit}")
            return False, (f"[kernel] Your last test run exited {self.last_test_exit} (tests "
                           f"ran and did not pass) and you are about to stop. Either fix what "
                           f"the traceback named, or state why that failure is unrelated.")
        # A repeated challenge is itself a trap: if the agent has already been challenged and
        # still wants to stop, it has answered. Never block the same finish twice.
        return True, ""

    # ---------------------------------------------------- E313 get_sure adapter
    def sure_observe(self, record: dict) -> None:
        """Accumulate a bounded structured observation for the orientation round."""
        if not hasattr(self, "_sure_observations"):
            self._sure_observations = []
        if len(self._sure_observations) < 60:
            self._sure_observations.append(record)

    # Re-run get_sure continuously, rather than only once per task, whenever
    # a NEW uncertainty appears (a red test anchored somewhere it hasn't grounded yet),
    # so the agent cannot keep guessing against stale orientation as the task evolves.
    # Bounds: dedup by anchor signature, hard cap of rounds per task, step cooldown.
    _SURE_MAX_ROUNDS = 6
    _SURE_COOLDOWN_STEPS = 6

    def sure_orient(self, step: int = 0) -> str | None:
        """Bounded get_sure orientation round.

        Fires on each NEW uncertainty signature — the latest red-test anchor (file:line)
        it has not grounded before, or the ungrounded-symbol state when there is no
        anchor. Grounds the target symbol's usage via the deterministic usage index and
        returns a compact observation line — localization ground truth, never a patch.
        Degrades to None on any failure (the signature is consumed before any work, so a
        failing signature never retries)."""
        seen = getattr(self, "_sure_seen", None)
        if seen is None:
            seen = self._sure_seen = set()
        if len(seen) >= self._SURE_MAX_ROUNDS:
            return None
        if step and step - getattr(self, "_sure_last_step", -99) < self._SURE_COOLDOWN_STEPS:
            return None
        obs_list = getattr(self, "_sure_observations", [])
        anchor = next((o.get("primary_anchor") for o in reversed(obs_list)
                       if o.get("kind") == "test" and o.get("primary_anchor")), None)
        sig = (anchor or "ungrounded").split(":")[0] + ":" + \
              ((anchor or "").split(":") + ["?"])[1]
        if sig in seen:
            return None
        seen.add(sig)
        self._sure_last_step = step
        try:
            from detreason import get_sure as gs
            from detreason.knowledge import build_usage_index
            from detreason.uledger import (EVIDENCE_SPAN, UNGROUNDED_SYMBOL, UNKNOWN_ATOM,
                                           ReductionAction, ULedger, URow)
            hits0 = any(o.get("kind") == "grep" and o.get("hit_count") == 0
                        for o in obs_list)
            rnd = len(seen)
            row = URow(
                id=f"react:{self.task_id}:{self._target_label()}:{self.fn}"
                   f"#orientation:{rnd}:{sig}",
                utype=UNGROUNDED_SYMBOL if (hits0 or not anchor) else UNKNOWN_ATOM,
                claim=(f"Static usage around {self._target_label()}:{self.fn}"
                       f" must be grounded for repair orientation; not a patch verdict."),
                severity=0.7, irreversibility=0.0, u=1.0,
                closure_criterion=EVIDENCE_SPAN,
                reduction_actions=[ReductionAction("anchor_by_exact_symbol", "cheap", 0.8)],
                correlation_group=f"react:{self.task_id}:orientation",
                owner="react_repair", source="react_observation")
            ledger = ULedger(f"react:{self.task_id}:sure:{rnd}",
                             gs.policy_for_tier(gs.R2))
            ledger.open(row)
            usage_index = getattr(self, "_sure_usage_index", None)
            if usage_index is None:
                usage_index = self._sure_usage_index = build_usage_index(self.root)
            context = gs.GetSureContext(
                base_facts={"target_path": getattr(self, "target_path", ""),
                            "function": self.fn, "observations": list(obs_list)},
                usage_index=usage_index, witness_inputs={},
                environment_id=f"react:{self.task_id}:sure")
            result, _decisions, evidences = gs.run_get_sure(
                run_id=f"react:{self.task_id}:sure:round{rnd}", ledger=ledger,
                context=context, decide=None,
                budget=gs.GetSureBudget(max_total_cost_milli=50, max_action_count=1,
                                        max_human_requests=0, max_wall_ms=5_000),
                tier=gs.R2,
                catalog={gs.USAGE_INDEX_LOOKUP: gs.CATALOG[gs.USAGE_INDEX_LOOKUP]},
                dry_run=False)
            cites = []
            for ev in evidences:
                for c in list(getattr(ev, "citations", ()))[:4]:
                    p = getattr(c, "source_path", None) or str(c)
                    ln = getattr(c, "line", None)
                    sn = (getattr(c, "verbatim_snippet", "") or "").strip()[:70]
                    cites.append(f"{p}:{ln}: {sn}" if ln else str(c))
            self._rec("skeptic", "sure_orient", str(getattr(result, "status", "?")))
            if cites:
                return ("[kernel] deterministic usage grounding for "
                        f"`{self.fn}` — real references:\n  " + "\n  ".join(cites[:5]))
            return None
        except Exception as e:                                       # noqa: BLE001
            self.status["skeptic"] = f"sure_orient degraded: {type(e).__name__}"
            return None

    # ---------------------------------------------------- E317 test-navigation oracle
    # Navigation help, never answer material. May reveal
    # node identity/relevance/held-out status; never bodies, assertions, or fixtures.
    _NAV_CAP = 3

    @staticmethod
    def _oracle_file(value: object) -> str:
        """Return the repository-relative file portion of an oracle selector."""
        return str(value or "").strip().replace("\\", "/").lstrip("./").split("::", 1)[0]

    @classmethod
    def _same_oracle_file(cls, left: object, right: object) -> bool:
        left_file = cls._oracle_file(left)
        right_file = cls._oracle_file(right)
        return bool(left_file and right_file) and (
            left_file == right_file
            or left_file.endswith("/" + right_file)
            or right_file.endswith("/" + left_file)
        )

    def _oracle_files(self) -> tuple[str, ...]:
        files = {
            self._oracle_file(value)
            for value in getattr(self, "f2p_ids", ())
            if self._oracle_file(value)
        }
        return tuple(sorted(files))

    def _preflight_proves_oracle(self) -> bool:
        """Accept only an explicit harness parent-RED/gold-GREEN proof."""
        proof = getattr(self, "oracle_preflight", None)
        if not isinstance(proof, Mapping):
            return False
        if proof.get("parent_red") is not True or proof.get("gold_green") is not True:
            return False
        proof_ids = proof.get("oracle_ids")
        if not isinstance(proof_ids, (tuple, list, set, frozenset)):
            return False
        oracle_files = self._oracle_files()
        return bool(oracle_files) and all(
            any(self._same_oracle_file(proof_id, oracle_file) for proof_id in proof_ids)
            for oracle_file in oracle_files
        )

    def _whole_relevant_oracle_green(self) -> bool:
        """Require whole-file GREEN, with E582's counted single-selector exception."""
        oracle_files = self._oracle_files()
        if not oracle_files:
            return False
        greens = self._whole_green_oracle_files()
        for oracle_file in oracle_files:
            if self._single_selector_oracle_green(oracle_file):
                continue
            if not any(self._same_oracle_file(green, oracle_file) for green in greens):
                return False
        return True

    def _whole_green_oracle_files(self) -> set[str]:
        return {
            self._oracle_file(record.get("intent"))
            for record in getattr(self, "_receipts", ())
            if record.get("kind") == "test"
            and record.get("result_class") == "GREEN"
            and "::" not in str(record.get("intent") or "")
        }

    def _bare_oracle_files(self) -> tuple[str, ...]:
        return tuple(sorted({
            self._oracle_file(value)
            for value in getattr(self, "f2p_ids", ())
            if self._oracle_file(value) and "::" not in str(value)
        }))

    def _oracle_selector_counts(self) -> Mapping[str, int]:
        """Return selector counts only from explicit wiring or the in-world file."""
        counts: dict[str, int] = {}
        explicit = getattr(self, "oracle_selector_counts", None)
        if isinstance(explicit, Mapping):
            for file_name, value in explicit.items():
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                if count >= 0:
                    counts[self._oracle_file(file_name)] = count

        for oracle_file in self._bare_oracle_files():
            if oracle_file in counts:
                continue
            path = self.root / oracle_file
            if not path.is_file():
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"), filename=oracle_file)
            except (OSError, SyntaxError):
                continue
            counts[oracle_file] = len({
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            })
        return counts

    def _single_selector_oracle_green(self, oracle_file: str) -> bool:
        """Treat one counted bare-file selector GREEN as E582 whole-file evidence."""
        if oracle_file not in self._bare_oracle_files():
            return False
        if self._oracle_selector_counts().get(oracle_file) != 1:
            return False
        return any(
            record.get("kind") == "test"
            and record.get("result_class") == "GREEN"
            and "::" in str(record.get("intent") or "")
            and self._same_oracle_file(record.get("intent"), oracle_file)
            for record in getattr(self, "_receipts", ())
        )

    def evaluate_test_relevance(self, requested: str, exit_code: int) -> "keb.Decision":
        """B2 pure evaluator for `test_relevance` below -- same decision, no mutation
        until `authorize()`'s effects axis allows it. See `evaluate_created_test_edit`
        for why the mutating method stays in place (existing organ-level tests call it
        directly and assert synchronous mutation)."""
        names = getattr(self, "f2p_names", ())
        if exit_code != 0 or not names:
            return keb.Decision.empty()
        req = (requested or "").strip()
        if not req or req == "test":
            return keb.Decision.empty()
        inworld_files = {iw.split("::")[0] for iw in getattr(self, "f2p_inworld", ())}
        relevant = (any(n in req for n in names)
                    or any(f and f in req for f in inworld_files))

        if relevant:
            req_file = req.split("::")[0]
            flip = any(rf and (rf in req or req_file in rf)
                       for rf in getattr(self, "_red_test_files", ()))
            whole_green = self._whole_relevant_oracle_green()
            if flip or (whole_green and self._preflight_proves_oracle()):
                def _apply_verified() -> None:
                    self._last_relevant_match = True
                    self._last_green_irrelevant = False
                    self._self_oracle_tainted = False
                return keb.Decision(
                    text=None, effects=(_apply_verified,),
                    effect_summary=("last_relevant_match=True",
                                     "last_green_irrelevant=False",
                                     "self_oracle_tainted=False"))
            if whole_green:
                def _apply_whole_green() -> None:
                    self._last_relevant_match = True
                    self._last_green_irrelevant = False
                return keb.Decision(
                    text=None, effects=(_apply_whole_green,),
                    effect_summary=("last_relevant_match=True",
                                     "last_green_irrelevant=False"))
            already = bool(self._challenged.get("never_red_msg"))
            text = None if already else (
                "[kernel] That relevant test has never FAILED in this world "
                "during this run — a green that was never red demonstrates "
                "nothing about your repair. Reproduce the failure first (the "
                "bug report's failure lines are the authoritative red), or "
                "state why it cannot fail here.")

            def _apply_never_red(_req: str = req, _already: bool = already) -> None:
                self._last_relevant_match = True
                self._last_green_irrelevant = True
                if not _already:
                    self._challenged["never_red_msg"] = True
                    self._rec("gate", "nav_green_never_red", _req[:80])

            return keb.Decision(
                text=text, effects=(_apply_never_red,),
                effect_summary=("last_relevant_match=True", "last_green_irrelevant=True"))

        # irrelevant green
        seen_now = getattr(self, "_nav_seen", None)
        already_seen = bool(seen_now) and (
            req in seen_now or len(seen_now) >= self._NAV_CAP)
        inworld = getattr(self, "f2p_inworld", ())
        if inworld:
            relevant_node = sorted(inworld)[0]
            text_candidate = (
                f"Run the relevant selector `{relevant_node}` green, then issue `done`. "
                "The latest green run selected 0 of the relevant regression nodes, so "
                "it does not verify this repair. Relevant nodes: "
                + "; ".join(sorted(inworld)[:4]))
        else:
            text_candidate = (
                "[kernel] your green run selected 0 of the relevant regression nodes — "
                "the discriminating regression test is held out of this checkout, so no "
                "in-world green run can verify the repair on its own. Verify against the "
                "failure lines in the bug report instead.")
        text = None if already_seen else text_candidate

        def _apply_irrelevant(_req: str = req) -> None:
            self._last_relevant_match = False
            self._last_green_irrelevant = True
            s = getattr(self, "_nav_seen", None)
            if s is None:
                s = self._nav_seen = set()
            if _req in s or len(s) >= self._NAV_CAP:
                return
            s.add(_req)
            self._rec("gate", "nav_irrelevant_green", _req[:80])

        return keb.Decision(
            text=text, effects=(_apply_irrelevant,),
            effect_summary=("last_relevant_match=False", "last_green_irrelevant=True"))

    def authorize(self, name: str, *, version: str,
                  decision_fn: "Callable[[], keb.Decision]",
                  rendered_prompt_hash: str | None = None,
                  input_identity: Mapping[str, object] | None = None) -> str | None:
        """Single authorization point, bound to the unique attempt and current request."""
        return keb.authorize(owner=self, name=name, version=version,
                              run_id=self.run_id, decision_fn=decision_fn,
                              seed=getattr(self, "measurement_seed", None),
                              rendered_prompt_hash=rendered_prompt_hash,
                              input_identity=input_identity)

    def test_relevance(self, requested: str, exit_code: int) -> str | None:
        """After a GREEN run: if the selector referenced none of the relevant regression
        nodes, say so — a green unrelated test must not be mistaken for verification.
        Advisory only; deduped per selector; capped. f2p_names/f2p_inworld are set by
        the loop at task init (empty = oracle disabled, e.g. no-oracle tasks)."""
        names = getattr(self, "f2p_names", ())
        if exit_code != 0 or not names:
            return None
        req = (requested or "").strip()
        if not req or req == "test":
            return None   # bare run auto-targets the fn's own tests — cannot judge relevance
        inworld_files = {iw.split("::")[0] for iw in getattr(self, "f2p_inworld", ())}
        if any(n in req for n in names) or \
                any(f and f in req for f in inworld_files):
            self._last_relevant_match = True
            # E327 (ack autopsy): a relevant green only VERIFIES when the same file was
            # seen RED earlier this run. An
            # in-world copy that never fails cannot demonstrate the repair.
            req_file = req.split("::")[0]
            flip = any(rf and (rf in req or req_file in rf)
                       for rf in getattr(self, "_red_test_files", ()))
            whole_green = self._whole_relevant_oracle_green()
            if flip or (whole_green and self._preflight_proves_oracle()):
                self._last_green_irrelevant = False
                self._self_oracle_tainted = False
                return None
            if whole_green:
                # The whole oracle is relevant evidence even when this run has no
                # model-staged RED; completion_verdict owns the remaining E386 proof.
                self._last_green_irrelevant = False
                return None
            self._last_green_irrelevant = True
            if not self._challenged.get("never_red_msg"):
                self._challenged["never_red_msg"] = True
                self._rec("gate", "nav_green_never_red", req[:80])
                return ("[kernel] That relevant test has never FAILED in this world "
                        "during this run — a green that was never red demonstrates "
                        "nothing about your repair. Reproduce the failure first (the "
                        "bug report's failure lines are the authoritative red), or "
                        "state why it cannot fail here.")
            return None
        # STATE updates on every irrelevant green, even when the message is deduped —
        # the done-gate (E322) consumes it; dedup below throttles only the advisory text.
        self._last_relevant_match = False
        self._last_green_irrelevant = True
        seen = getattr(self, "_nav_seen", None)
        if seen is None:
            seen = self._nav_seen = set()
        if req in seen or len(seen) >= self._NAV_CAP:
            return None
        seen.add(req)
        self._rec("gate", "nav_irrelevant_green", req[:80])
        inworld = getattr(self, "f2p_inworld", ())
        if inworld:
            relevant_node = sorted(inworld)[0]
            return (
                f"Run the relevant selector `{relevant_node}` green, then issue `done`. "
                "The latest green run selected 0 of the relevant regression nodes, so "
                "it does not verify this repair. Relevant nodes: "
                + "; ".join(sorted(inworld)[:4])
            )
        return ("[kernel] your green run selected 0 of the relevant regression nodes — "
                "the discriminating regression test is held out of this checkout, so no "
                "in-world green run can verify the repair on its own. Verify against the "
                "failure lines in the bug report instead.")

    # ---------------------------------------------------- E329 call-path card 
    # rss patched SSRF at the wrong hop; content proposed the helper's 503 where the
    # gate's 403 was the contract. The traceback's in-repo frames ARE the measured call
    # path — state it, hop by hop, with the one warning that matters: the failure
    # SURFACES at the last hop but the fix belongs at the hop whose CONTRACT broke.
    def call_path_card(self, anchors: list) -> str | None:
        if not anchors or len(anchors) < 2:
            return None                      # single-hop: E312's anchor already covers it
        chain = [a.get("anchor", "?") for a in reversed(anchors)][:5]  # caller -> failure
        key = "|".join(chain)
        seen = getattr(self, "_cp_seen", None)
        if seen is None:
            seen = self._cp_seen = set()
        if key in seen or len(seen) >= 3:
            return None
        seen.add(key)
        self._rec("graph", "call_path_card", chain[-1][:60])
        return ("[kernel] CALL PATH (measured from the traceback, caller -> failure):\n  "
                + " -> ".join(chain) +
                "\n  The failure SURFACES at the last hop, but the fix belongs at the "
                "hop whose CONTRACT the bug report describes — not necessarily the last.")

    # ---------------------------------------------------- E325 self-oracle taint
    # Article round-F: the model CREATED a repro test, and when it went red, edited the
    # failing assertion OUT of it — then presented the green as proof. A self-authored
    # oracle weakened after a red proves nothing. Rule-7 family: a red test is a
    # question, never an instruction to weaken it — including your own.
    def note_test_red(self, path: str) -> None:
        if not hasattr(self, "_red_test_files"):
            self._red_test_files = set()
        self._red_test_files.add(path)

    def evaluate_created_test_edit(self, path: str) -> "keb.Decision":
        """B2 pure evaluator for `note_created_test_edit` below -- computes the SAME
        decision (taint + at-most-one advisory) WITHOUT mutating `self`. Routed through
        `authorize()` at the two react_repair.py call sites so KERNEL_MODE/SHADOW can
        gate the effect itself, not just the text (the bug `note_created_test_edit`'s
        eager-argument-evaluation call sites had). `note_created_test_edit` is left
        untouched below for the existing organ-level tests that call it directly."""
        if path not in getattr(self, "_red_test_files", ()):
            return keb.Decision.empty()
        already = bool(self._challenged.get("self_oracle_taint_msg"))
        text = None if already else (
            "[kernel] You edited your own repro test AFTER it failed. If the edit "
            "weakened or removed the failing assertion, a green run of it no longer "
            "proves the repair. Fix product code to satisfy the original assertion, "
            "or state why the assertion itself was wrong.")

        def _apply(_path: str = path, _already: bool = already) -> None:
            self._self_oracle_tainted = True
            if not _already:
                self._challenged["self_oracle_taint_msg"] = True
                self._rec("skeptic", "self_oracle_taint", _path[:80])

        return keb.Decision(text=text, effects=(_apply,),
                             effect_summary=("self_oracle_tainted=True",))

    def note_created_test_edit(self, path: str) -> str | None:
        """Edit to a created test AFTER that file went red: taint + one advisory."""
        if path not in getattr(self, "_red_test_files", ()):
            return None
        self._self_oracle_tainted = True
        if self._challenged.get("self_oracle_taint_msg"):
            return None
        self._challenged["self_oracle_taint_msg"] = True
        self._rec("skeptic", "self_oracle_taint", path[:80])
        return ("[kernel] You edited your own repro test AFTER it failed. If the edit "
                "weakened or removed the failing assertion, a green run of it no longer "
                "proves the repair. Fix product code to satisfy the original assertion, "
                "or state why the assertion itself was wrong.")

    def clear_self_oracle_taint(self) -> None:
        """A relevant/authoritative green (E317 relevance) clears the taint."""
        self._self_oracle_tainted = False

    # ---------------------------------------------------- E323 contract card 
    # rss/content failed on wrong-LAYER reasoning (503 from the helper vs 403 from the
    # gate; SSRF fixed at the wrong hop). The card states the target's CURRENT contract —
    # signature, raises, returns — plus real caller exemplars. Measured facts only.
    def contract_card(self) -> str | None:
        """Once per task, on the first red anchor: the target function's live contract."""
        if getattr(self, "_contract_done", False):
            return None
        self._contract_done = True
        tp = str(getattr(self, "target_path", "") or "").strip()
        if not tp:
            return None
        p = self.root / tp
        if not p.is_file():
            return None
        try:
            import ast
            tree = ast.parse(p.read_text(errors="replace"))
        except (OSError, SyntaxError):
            return None
        fndef = next((n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == self.fn), None)
        if fndef is None:
            return None
        try:
            sig = f"def {self.fn}({ast.unparse(fndef.args)})" + (
                f" -> {ast.unparse(fndef.returns)}" if fndef.returns else "")
        except Exception:                                            # noqa: BLE001
            sig = f"def {self.fn}(...)"
        raises = sorted({ast.unparse(n.exc).split("(")[0]
                         for n in ast.walk(fndef)
                         if isinstance(n, ast.Raise) and n.exc is not None})[:4]
        returns = sum(isinstance(n, ast.Return) and n.value is not None
                      for n in ast.walk(fndef))
        lines = [f"[kernel] CONTRACT of `{self.fn}` (current source, {tp}):",
                 f"  {sig}"]
        if raises:
            lines.append("  raises: " + ", ".join(raises))
        lines.append(f"  value-returning return statements: {returns}")
        idx = getattr(self, "_sure_usage_index", None)
        if idx is not None:
            try:
                for ex in list(idx.query(self.fn, limit=3)):
                    src = (getattr(ex, "statement", "") or "").strip()[:100]
                    loc = f"{getattr(ex, 'path', '?')}:{getattr(ex, 'line', '?')}"
                    if src:
                        lines.append(f"  caller {loc}: {src}")
            except Exception:                                        # noqa: BLE001
                pass
        self._rec("graph", "contract_card", tp)
        return "\n".join(lines)

    # ---------------------------------------------------- E320 target-guard tracer 
    # Guard/predicate semantics were wrong in 6/6 gap tasks (guesscensus). On a red
    # anchor in product code, print the EXACT guard expressions at/around the anchor
    # from the CURRENT source — measured text, never an interpretation or a patch.
    def guard_at_anchor(self) -> str | None:
        obs_list = getattr(self, "_sure_observations", [])
        anchor = next((o.get("primary_anchor") for o in reversed(obs_list)
                       if o.get("kind") == "test" and o.get("primary_anchor")), None)
        if not anchor:
            return None
        parts = anchor.split(":")
        try:
            afile, aline = parts[0], int(parts[1])
        except (IndexError, ValueError):
            return None
        if "/test" in afile or afile.startswith("test"):
            return None                      # test-side anchors: raw output already shows it
        p = self.root / afile
        if not p.is_file():
            return None
        try:
            import ast
            src = p.read_text(errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            return None
        lines = src.splitlines()
        hits: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and abs(node.lineno - aline) <= 4:
                seg = lines[node.lineno - 1].strip()[:110]
                hits.append(f"L{node.lineno}: {seg}")
            if len(hits) >= 3:
                break
        if not hits:
            return None                      # NO_COMPARISON — say nothing, never guess
        key = f"{afile}:{aline}:" + "|".join(hits)
        seen = getattr(self, "_guard_seen", None)
        if seen is None:
            seen = self._guard_seen = set()
        if key in seen or len(seen) >= 4:
            return None
        seen.add(key)
        self._rec("skeptic", "guard_at_anchor", f"{afile}:{aline}")
        out = ("[kernel] TARGET guards at the failure anchor (current source, "
               f"{afile} near L{aline}):\n  " + "\n  ".join(hits))
        # E340 witness juxtaposition (guard-witness minimal tier): concrete values the
        # FAILURE itself exposed, placed next to the guards they must satisfy. Measured
        # extraction only — never evaluated, never interpreted (negative control:
        # anything dynamic stays UNKNOWN/silent).
        wit = getattr(self, "_last_witness_values", None)
        if wit:
            out += ("\n  concrete values visible in the failure (trace which guard "
                    "branch they take): " + ", ".join(wit[:6]))
        return out

    def note_witness_values(self, obs: str) -> None:
        """Harvest literal values pytest's assertion rewriting / exception reprs expose
        in E-lines: quoted strings, numbers, key=value pairs. Extraction, not inference."""
        vals: list[str] = []
        for ln in obs.splitlines():
            ln = ln.strip()
            if not ln.startswith("E "):
                continue
            vals += re.findall(r"'[^']{1,40}'|\"[^\"]{1,40}\"", ln)[:4]
            vals += re.findall(r"\b\w+=(?:'[^']{1,30}'|[\d.]+|True|False|None)", ln)[:4]
        seen: list[str] = []
        for v in vals:
            if v not in seen:
                seen.append(v)
        if seen:
            self._last_witness_values = seen[:8]

    # ---------------------------------------------------- E319 edit-relevance witness 
    # "Does this edit affect the anchored failure path?" Measured facts only, never a
    # patch; the edit ALWAYS proceeds. UNKNOWN is silent and consumes no warning slot.
    _EDIT_WARN_CAP = 3

    def edit_relevance(self, edit_path: str, edit_src_hint: str = "") -> str | None:
        """Relation of an applied product edit to the latest red-test anchor.
        DIRECT: same file. CALLCHAIN: the anchor file statically references the edited
        module/symbol (one conservative edge). UNRELATED: provably zero references —
        the only case that warns. Anything uncertain -> UNKNOWN -> silent."""
        obs_list = getattr(self, "_sure_observations", [])
        anchor = next((o.get("primary_anchor") for o in reversed(obs_list)
                       if o.get("kind") == "test" and o.get("primary_anchor")), None)
        if not anchor or not edit_path:
            return None
        anchor_file = anchor.split(":")[0]
        ep = str(edit_path)
        if ep.endswith(anchor_file) or anchor_file.endswith(ep):
            return None                                   # DIRECT — never warn
        try:
            af = self.root / anchor_file
            if not af.is_file():
                return None                               # UNKNOWN
            text = af.read_text(errors="replace")
            stem = pathlib.Path(ep).stem
            if stem in text or (edit_src_hint and edit_src_hint in text):
                return None                               # CALLCHAIN (1 edge) — never warn
        except OSError:
            return None                                   # UNKNOWN
        seen = getattr(self, "_editwarn_seen", None)
        if seen is None:
            seen = self._editwarn_seen = set()
        key = f"{anchor_file}|{ep}"
        if key in seen or len(seen) >= self._EDIT_WARN_CAP:
            return None
        seen.add(key)
        self._rec("skeptic", "edit_unrelated", key[:80])
        return (f"[kernel] EDIT WARN only | edited={ep} | failure={anchor} | "
                f"relation=UNRELATED (the failing file has no static reference to "
                f"`{pathlib.Path(ep).stem}`) | action=APPLIED anyway — if this edit is "
                f"right, say how it reaches the failing path.")

    # ---------------------------------------------------- E318 result admissibility 
    # "database refusal becomes INFRA_FAILURE; a collected assertion failure remains
    # PRODUCT_FAILURE." An environmental red must never drive a product edit — that is
    # how the voided round (E314) would have poisoned a live agent.
    _INFRA_MARKERS = (
        "psycopg2.OperationalError", "ConnectionRefusedError", "connection refused",
        "could not connect to server", "sqlalchemy.exc.OperationalError",
        "redis.exceptions.ConnectionError", "socket.gaierror",
        "Temporary failure in name resolution", "errno 61", "Errno 111",
    )

    def result_admissibility(self, obs: str, exit_code: int) -> str | None:
        """One capped advisory when a red run's failure is environmental. An
        AssertionError anywhere in the output keeps the verdict PRODUCT_FAILURE —
        an app-level test may legitimately assert on connection-shaped exceptions."""
        if exit_code == 0 or "AssertionError" in obs:
            return None
        hit = next((m for m in self._INFRA_MARKERS if m in obs), None)
        if hit is None:
            return None
        seen = getattr(self, "_adm_seen", None)
        if seen is None:
            seen = self._adm_seen = set()
        if hit in seen or len(seen) >= 3:
            return None
        seen.add(hit)
        self._rec("skeptic", "infra_failure", hit)
        return ("[kernel] result=INFRA_FAILURE — this red is environmental "
                f"(`{hit}`), not evidence about your patch. Do not edit product code "
                "in response to it; re-run or pick a test that does not need that service.")

    # ---------------------------------------------------- E306 advisory nudges
    def finish_nudge(self, edits: int) -> str | None:
        """Gate, forward gear: tests green + edits landed + nothing edited
        since -> ONE bounded 'consider done' per edit generation. Advisory, never forced."""
        if (self.last_test_exit == 0 and edits > 0
                and self._challenged.get("finish_gen") != edits):
            self._challenged["finish_gen"] = edits
            self._rec("gate", "finish_nudged", f"green at edits={edits}")
            return ("[kernel] Your last test run was GREEN and you have applied edits. If "
                    "you believe the fix is complete, call `done` now; otherwise state what "
                    "is still missing.")
        return None

    def green_freeze_nudge(self, *, green_before_edit: bool | None = None) -> str | None:
        """Offer one advisory only when a green receipt predates the edit."""
        if green_before_edit is not True:
            return None
        gen = self._green_gen = getattr(self, "_green_gen", 0)
        if (self.last_test_exit == 0
                and self._challenged.get("green_freeze") != gen):
            self._challenged["green_freeze"] = gen
            self._rec("gate", "green_freeze_nudged", f"edit after green gen={gen}")
            return ("[kernel] The last test run was GREEN before this edit. "
                    "Run the affected tests before relying on that result.")
        return None

    def take_invalidation_notice(self, action_kind: object) -> str | None:
        """Keep invalidation notices off an edit or terminal submission receipt."""
        action = str(action_kind or "").strip().lower()
        if action in {"edit", "str_replace", "file_create", "delete", "move", "done"}:
            return None
        notice = getattr(self, "_inval_pending", None)
        if not isinstance(notice, str) or not notice.strip():
            return None
        self._inval_pending = None
        return notice

    def churn_nudge(self, edits_since_test: int) -> str | None:
        """Skeptic, in-flight: N edits with no verification is churn, not
        iteration — measured 83-edit/0-test and 101-edit/0-test runs. One nudge per streak."""
        if edits_since_test >= 5 and not self._challenged.get(f"churn_{edits_since_test}"):
            self._challenged[f"churn_{edits_since_test}"] = True
            self._rec("skeptic", "churn_nudged", f"{edits_since_test} edits unverified")
            return (f"[kernel] You have made {edits_since_test} edits without running any "
                    f"test. Run `run_tests` NOW to learn whether you are closer or further.")
        return None

    def plan_summary(self, edits: int, tests: int) -> str:
        """The persistent plan: compact state that survives context
        trims — what the model has DONE, so a trimmed conversation keeps its memory."""
        # E307 context_capsule: a STRUCTURED capsule, not counters — what
        # was read, what was edited, and the latest verdict, so a trimmed conversation
        # keeps its working memory instead of re-investigating.
        reads = [k.split()[1] for k in self.seen if k.startswith(("read ", "grep "))
                 and len(k.split()) > 1][-8:]
        base = (f"CAPSULE — target fn: {self.fn}; edits: {edits} "
                f"({getattr(self, 'product_edits', 0)} product); test runs: {tests}, last "
                f"exit={self.last_test_exit}; recently examined: {', '.join(reads) or '-'}")
        # Record read-only task state evidence for subsequent verification.
        # misremember its own history across a trim. Facts only, ~<400 tokens.
        obs_list = getattr(self, "_sure_observations", [])
        test_recs = [o for o in obs_list if o.get("kind") == "test"][-5:]
        if test_recs:
            base += "\n  tests (recorded): " + "; ".join(
                f"s{o.get('step', '?')} exit={o.get('exit')}"
                + (f" anchor={o.get('primary_anchor')}" if o.get("primary_anchor") else "")
                for o in test_recs)
        anchors = {o.get("primary_anchor") for o in obs_list
                   if o.get("kind") == "test" and o.get("primary_anchor")}
        if anchors:
            base += "\n  distinct failure anchors seen: " + ", ".join(sorted(anchors)[:4])
        neg = [t for t in self.trace if t["event"] in
               ("repeats_blocked", "edit_unrelated", "infra_failure",
                "nav_irrelevant_green")][-4:]
        if neg:
            base += "\n  negative work (do not repeat): " + "; ".join(
                f"{t['event']}:{t['detail'][:60]}" for t in neg)
        # E349 StateFrame frontier (promotion gate cleared at 18.07/20): objective atoms
        # + per-selector verdict ledger, rendered from receipts — the model reads
        # projected state, never writes it. Shadowable like every organ.
        _fr_lines = []
        tp = getattr(self, "target_path", "")
        if tp:
            _fr_lines.append(f"objective: repair `{self.fn}` in {tp}")
        _by_sel: dict = {}
        for r in getattr(self, "_receipts", []):
            if r["kind"] == "test" and r["result_class"] in (
                    "GREEN", "RED", "SELECTOR_ERROR", "NO_TESTS"):
                _by_sel[r["intent"]] = f"{r['result_class']}@s{r.get('turn', '?')}"
        if _by_sel:
            _fr_lines.append("verdict ledger: " + "; ".join(
                f"{s[:60]}={v}" for s, v in list(_by_sel.items())[-4:]))
        if _fr_lines:
            _fr = self.organ_out("frontier", "\n  ".join(_fr_lines))
            if _fr:
                base += "\n  " + _fr
        # E342: hypothesis frontier + kill-list in the capsule (shadowable organ).
        hyp = getattr(self, "hyp", None)
        if hyp is not None:
            try:
                _hc = self.organ_out("hypotheses", hyp.capsule() or None)
                if _hc:
                    base += "\n" + _hc
            except Exception:                                        # noqa: BLE001
                pass
        return base

    # ------------------------------------------------------------- tracing
    # ---------------------------------------------------- E336 CompletionVerdict (shadow)
    # DoD skeleton (dod_contract.md): stop-control and success-certification separate.
    # `done` may always stop the loop; this verdict — computed from RECEIPTS, never the
    # model's prose — says whether the finish deserves credit. Shadow-only for now:
    # recorded as trace + receipt, shown to nobody.
    _CLAIM_WORDS = (
        ("FIXED", ("fixed", "resolved", "repair", "now correctly", "corrected")),
        ("NO_DEFECT", ("already correct", "no defect", "not failing", "code is correct")),
        ("CANNOT_REPRODUCE", ("cannot reproduce", "not present", "fixed in a later",
                              "does not exist")),
        ("BLOCKED_ENV", ("environment", "connection refused", "cannot connect")),
    )

    def completion_verdict(self) -> dict:
        claim_text = (getattr(self, "last_done_summary", "") or "").lower()
        claimed = next((label for label, words in self._CLAIM_WORDS
                        if any(w in claim_text for w in words)), "UNSPECIFIED")
        receipts = getattr(self, "_receipts", [])
        product_edit = any(r["kind"] == "edit" and r["result_class"] == "APPLIED"
                           and "/test" not in r["intent"] and
                           not r["intent"].split()[-1].startswith("test_")
                           for r in receipts)
        greens = [r for r in receipts if r["kind"] == "test"
                  and r["result_class"] == "GREEN"]
        reds = [r for r in receipts if r["kind"] == "test" and r["result_class"] == "RED"]
        untainted = not getattr(self, "_self_oracle_tainted", False)
        relevant = bool(getattr(self, "_last_relevant_match", True))
        observed_flip = bool(reds and greens and
                             max(g["turn"] or 0 for g in greens) >
                             min(r["turn"] or 0 for r in reds))
        preflight_waiver = (
            self._preflight_proves_oracle()
            and self._whole_relevant_oracle_green()
        )
        flip = observed_flip or preflight_waiver
        certified = (claimed == "FIXED" and product_edit and flip
                     and untainted and relevant)
        runtime = getattr(self, 'measurement', None)
        bound_current = None
        authority = None
        if runtime is not None and runtime.projector_kind == 'rich':
            public_verdict = getattr(runtime, "public_test_verdict", None)
            if callable(public_verdict):
                authority = public_verdict()
                bound_current = authority.state == SATISFIED
                # Current original-contract proof does not require manufacturing a
                # historical failure after a correct first repair.
                certified = (bound_current if getattr(runtime, "continuation", None) is not None
                             else claimed == 'FIXED' and bound_current)
            else:
                bound_current = runtime.bound_public_verification()
                certified = (bound_current if getattr(runtime, "continuation", None) is not None
                             else claimed == 'FIXED' and bound_current)
        verdict = {"claimed": claimed, "product_edit": product_edit,
                   "red_green_flip": flip, "oracle_untainted": untainted,
                   "relevant_green": relevant,
                   "preflight_red_green_waiver": preflight_waiver,
                   "success_certified": certified}
        if bound_current is not None:
            verdict['current_bound_public_verification'] = bound_current
            verdict['certification_scope'] = 'public_required_checks_only'
        if authority is not None:
            verdict['public_test_verdict'] = authority.to_dict()
        self._rec("gate", "dod_verdict",
                  f"{claimed} certified={certified} edit={product_edit} flip={flip}"
                  f" waiver={preflight_waiver}")
        self.receipt("verdict", claimed, str(sorted(verdict.items())),
                     "CERTIFIED" if certified else "UNCERTIFIED")
        if runtime is not None and getattr(runtime, "continuation", None) is not None:
            runtime.continuation.certify(verdict)
        return verdict

    # ---------------------------------------------------- E334 action receipts (Session 1)
    # The synthesis' one-source-of-truth substrate: every consequential action leaves a
    # typed, fingerprinted, turn-stamped receipt. Pure telemetry (shadow by construction
    # — nothing model-facing); the spine/CompletionVerdict will consume these.
    def receipt(self, kind: str, intent: str, material: str, result_class: str,
                detail: str = "", anchor: str | None = None,
                why: str | None = None, *,
                attempt: Mapping[str, object] | None = None,
                target_path: str | None = None,
                changed_paths: tuple[str, ...] = (),
                read_contract: Mapping[str, object] | None = None,
                result_set_digest: str | None = None,
                test_output: str | None = None) -> None:
        import hashlib
        import json as _json
        import time
        receipt_why = (why if why is not None else
                       (getattr(self, "last_why", "") or ""))
        receipt_detail = str(detail or "")
        # Typed edit refusals are the model-facing recovery route.  Preserve their complete
        # JSON envelope in the receipt so history and frame projections cannot lose the code,
        # path, or str_replace alternative to the legacy telemetry bound.
        preserve_typed_detail = False
        if kind == "edit" and receipt_detail.lstrip().startswith("{"):
            try:
                parsed_detail, _detail_end = _json.JSONDecoder().raw_decode(
                    receipt_detail.lstrip())
            except (TypeError, ValueError):
                parsed_detail = None
            if (isinstance(parsed_detail, dict) and parsed_detail.get("status") == "REJECTED"
                    and "code" in parsed_detail and "path" in parsed_detail
                    and "alternatives" in parsed_detail):
                receipt_detail = _json.dumps(
                    parsed_detail, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                preserve_typed_detail = True
        preserve_guard_detail = receipt_detail.startswith((
            "REJECTED_SHRINK ", "REJECTED_DESTRUCTIVE_COMMAND "))
        rec = _new_receipt({"kind": kind, "intent": intent[:120],
               "why": receipt_why[:600 if why is not None else 160],
               "fingerprint": hashlib.sha256(
                   f"{kind}|{material}".encode()).hexdigest()[:16],
               "result_class": result_class,
               "detail": (receipt_detail
                          if preserve_typed_detail or preserve_guard_detail
                          else receipt_detail[:150]),
               "turn": getattr(self, "current_step", None),
               "workspace_revision": getattr(self, "workspace_revision", 0),
               "ts": round(time.time(), 2)})
        if target_path is not None:
            rec["target_path"] = target_path
        if changed_paths:
            rec["changed_files"] = [{"path": path} for path in changed_paths]
        if anchor:
            rec["anchor"] = anchor
        # Baseline organs consume the exact legacy record. The experimental retry
        # history has its own canonical identity and cannot alter baseline advice.
        legacy_retry_record = dict(rec)
        if kind in {"grep", "search"} and result_set_digest is not None:
            rec["result_set_digest"] = result_set_digest
        if (kind == "test" and test_output is not None
                and str(result_class).upper() in {"RED", "FAIL", "FAILED", "FAIL_WITH_WITNESS"}):
            from kernel_tool_call_loop_breaker import failing_test_identity
            failure_identity = failing_test_identity(
                test_output, check_outputs=getattr(
                    getattr(self, "measurement", None), "_check_output_receipts", None),
                journal=getattr(self, "audit_journal", None))
            if failure_identity is not None:
                rec["failure_identity"] = failure_identity
        if kind == "read" and read_contract is not None:
            import copy
            from kernel_source_delivery import READ_CONTRACT_FIELDS
            rec.update({field: copy.deepcopy(read_contract.get(field))
                        for field in READ_CONTRACT_FIELDS})
        if attempt is not None:
            rec["attempt"] = dict(attempt)
            rec["attempt_id"] = attempt["fingerprint"]
        from kernel_wire_identity import next_wire_id
        rec["receipt_id"] = next_wire_id(
            getattr(self, "audit_journal", None), "action-receipt",
            getattr(self, "current_step", None), kind, rec["fingerprint"])
        self.last_receipt_id = rec["receipt_id"]
        journal = getattr(self, "audit_journal", None)
        if journal is not None:
            rec["run_id"] = self.run_id
            runtime = getattr(self, "measurement", None)
            continuation = getattr(runtime, "continuation", None) if runtime is not None else None
            dispatch = getattr(continuation, "dispatch", None)
            dispatch_is_current = (
                isinstance(dispatch, Mapping)
                and dispatch.get("accepted") is True
                and dispatch.get("step") == getattr(self, "current_step", None)
            )
            dispatch_request_id = (
                dispatch.get("request_id") if dispatch_is_current else None
            )
            request_id = (
                dispatch_request_id
                if isinstance(dispatch_request_id, str) and dispatch_request_id.strip()
                else getattr(self, "current_request_id", None)
            )
            if request_id is None and runtime is not None:
                request_id = getattr(runtime, "current_request_id", None)
            rec["request_id"] = request_id
            if continuation is not None:
                rec.update(continuation.lineage())
            journal.emit("action_receipt", receipt=rec)
        if not hasattr(self, "_receipts"):
            self._receipts = _receipt_history(directory=_journal_directory(journal))
        rec = _append_receipt(self._receipts, rec)
        runtime = getattr(self, "measurement", None)
        if runtime is not None:
            runtime.completed_receipt(rec, material)
        # E484: the governor folds every receipt into difficulty signals.
        _cgv = getattr(self, "cogov", None)
        if _cgv is not None:
            try:
                _cgv.observe_receipt(legacy_retry_record)
            except Exception:                                        # noqa: BLE001
                pass
        # E491: changed-condition retry gate — an edit whose fingerprint matches a
        # prior FAILED attempt with nothing material changed gets warned (1st) then
        # refused-in-words (2nd). Tests/reads always allowed. (Post-receipt delivery
        # this slice; true pre-dispatch block is slice 2.)
        _rgt = getattr(self, "retrygate", None)
        if _rgt is not None:
            try:
                _rd = _rgt.observe(legacy_retry_record)
                if kind == "edit" and not getattr(_rd, "allowed", True) and _rd.message:
                    _rl = self.organ_out("retry_gate", "[kernel-retry] " + _rd.message[:400])
                    if _rl:
                        self._retry_pending = _rl
                elif kind == "edit" and str(getattr(_rd, "decision", "")).lower().endswith("warn") and _rd.message:
                    _rl = self.organ_out("retry_gate", "[kernel-retry] " + _rd.message[:400])
                    if _rl:
                        self._retry_pending = _rl
            except Exception:                                        # noqa: BLE001
                pass
        _pdg = getattr(self, "predispatch_gate", None)
        if _pdg is not None:
            _pdg.observe(rec)
        # E470: every receipt feeds the invalidation cascade; dirty-belief notices
        # are stashed for same-turn delivery (bounded + deduped in the ledger).
        _inv = getattr(self, "inval", None)
        if _inv is not None:
            try:
                _ir = _inv.observe_receipt(legacy_retry_record)
                for _n in getattr(_ir, "notices", ())[:2]:
                    _il = self.organ_out("invalidation", _n)
                    if _il:
                        self._inval_pending = (
                            (getattr(self, "_inval_pending", None) or "") +
                            ("\n" if getattr(self, "_inval_pending", None) else "")
                            + _il)
            except Exception:                                        # noqa: BLE001
                pass
        # E342: receipts feed the hypothesis graph (kill-list/frontier) when wired.
        hyp = getattr(self, "hyp", None)
        if hyp is not None:
            try:
                hyp.observe(legacy_retry_record)
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] hyp.observe failed: {type(_e).__name__}")
        # E423: composed cognition loop (the long-horizon unlock) — every receipt
        # advances fold->hypotheses->replan->seam-actions->selector; a rendered card
        # (discriminating next action) is stashed for same-turn delivery.
        cc = getattr(self, "composed", None)
        if cc is not None:
            try:
                _adv = cc.advance(legacy_retry_record)
                # Deliver ONLY when the replan trigger fires, and NEVER the raw card
                # JSON (10KB — E394): one compact line naming reason + chosen action.
                _card = None
                _trig = getattr(_adv, "trigger", None)
                if getattr(_trig, "should_fire", False):
                    _sel = getattr(_adv, "selection", None)
                    _aid = getattr(_sel, "selected_action_id", None) or \
                        getattr(_sel, "baseline_action_id", "")
                    _card = ("[kernel-plan] RE-PLAN: "
                             f"{getattr(_trig, 'reason', '')[:120]}. Highest-value "
                             f"next action: {_aid}.")
                if _card:
                    _cline = self.organ_out("composed", _card)
                    if _cline:
                        self._composed_pending = _cline
                        self._rec("composed", "card_rendered", _card[:80])
                    # E466: frame breaker — critique asks "is the answer wrong";
                    # this asks "is the QUESTION wrong". Licensed only after two
                    # same-frame replans; once per frame.
                    try:
                        from kernel_framebreak import build_frame_challenge
                        if not hasattr(self, "_fb_replans"):
                            self._fb_replans, self._fb_challenged = [], []
                        self._fb_replans.append({
                            "should_fire": True,
                            "claim_id": str(_aid or ""),
                            "fingerprint": (rec or {}).get("fingerprint", ""),
                        })
                        _fch = build_frame_challenge(
                            getattr(self, "task_row", {}) or {},
                            getattr(self, "_receipts", ()),
                            self._fb_replans,
                            challenged_frames=self._fb_challenged)
                        if _fch is not None:
                            self._fb_challenged.append(_fch.frame.frame_id)
                            _fl = self.organ_out("framebreak", _fch.render())
                            if _fl:
                                self._composed_pending = (
                                    (self._composed_pending + "\n" + _fl)
                                    if self._composed_pending else _fl)
                                self._rec("framebreak", "challenge_delivered",
                                          _fch.frame.frame_id[:80])
                    except Exception:                                # noqa: BLE001
                        pass
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] composed advance failed: {type(_e).__name__}")
        # Inspect focused evidence in the close-look component for every
        # test/edit receipt feeds the focus state machine; an OPEN decision renders the
        # microscope dossier, stashed for the orchestrator to append to the observation.
        cl = getattr(self, "closelook", None)
        if cl is not None:
            try:
                _dec = cl.process_receipt(legacy_retry_record)
                if getattr(_dec, "opened", False):
                    # E419: the dossier renders from cl.dossier.render() + the
                    # decision's walkthrough demand (render_dossier never existed on
                    # the kernel — collect-5b lost ALL deliveries to that AttributeError).
                    _doss = getattr(cl, "dossier", None)
                    _text = ((_doss.render() if _doss is not None and
                              hasattr(_doss, "render") else "")
                             + "\n" + (getattr(_dec, "message", "") or "")).strip()
                    _line = self.organ_out("closelook", _text) if _text else None
                    if _line:
                        self._closelook_pending = _line
                        self._rec("closelook", "dossier_opened",
                                  getattr(_dec, "trigger", "")[:80])
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] closelook failed: {type(_e).__name__}: {_e}")
        # E350: receipts feed the vitals EKG when wired.
        ekg = getattr(self, "ekg", None)
        if ekg is not None:
            try:
                ekg.observe(legacy_retry_record)
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] ekg.observe failed: {type(_e).__name__}")
        if kind == "edit" and result_class == "APPLIED":
            self.workspace_revision = getattr(self, "workspace_revision", 0) + 1
        rp = getattr(self, "receipts_path", None)
        if rp:
            try:
                with open(rp, "a") as fh:
                    fh.write(_json.dumps(rec) + "\n")
            except OSError as e:
                print(f"    [kernel-wiring] receipt persist failed: {e}")

    def organ_out(self, name: str, line: str | None) -> str | None:
        """E332 shadow mode: the single chokepoint between an organ's decision and the
        model's context. KERNEL_SHADOW (comma list of organ names, or 'all') keeps the
        organ fully computing but suppresses its output, recording a would_fire trace —
        so unproven organs dry-run in production lanes and their counterfactual value is
        measurable from the transcript."""
        if not line:
            return None
        from kernel_intervention_policy import interventions_enabled
        if not interventions_enabled():
            self._rec("liveness", "ORGAN_INTERVENTION_DISABLED", f"organ={name}")
            return None
        import os
        shadow = getattr(self, "_shadow_set", None)
        if shadow is None:
            raw = os.environ.get("KERNEL_SHADOW", "")
            shadow = self._shadow_set = {s.strip() for s in raw.split(",") if s.strip()}
        if name in shadow or "all" in shadow:
            self._rec("shadow", f"would_fire:{name}", line[:150])
            self._rec("liveness", "ORGAN_SHADOWED", f"organ={name} detail={line[:120]}")
            return None
        # E484/E509: cognitive governor — advisory organs go quiet while the task
        # is CRUISING; STUCK may widen enforcement/diagnosis delivery only. The
        # governor's class table keeps advisory dosing at its STANDARD cap.
        _cg = getattr(self, "cogov", None)
        if _cg is not None:
            try:
                if not _cg.should_deliver(name):
                    self._rec("liveness", "ORGAN_DOSED_QUIET",
                              f"organ={name} posture={_cg.posture.value}")
                    return None
            except Exception:                                        # noqa: BLE001
                pass
        self._rec("liveness", "ORGAN_DELIVERED", f"organ={name} detail={line[:120]}")
        # E429: every delivered advisory becomes an advice-ledger row so later turns
        # can classify it FOLLOWED / IGNORED / OVERTAKEN (2-strike escalation cap).
        adv = getattr(self, "advice", None)
        if adv is not None and name != "advice_ledger":
            try:
                import re as _re
                _tgts = _re.findall(
                    r"[\w./-]+\.py(?:::\w+)?|test_\w+", line)[:4]
                adv.deliver(name, getattr(self, "current_step", 0) or 0, line,
                            _tgts,
                            world_revision=getattr(self, "workspace_revision", 0) or 0)
            except Exception:                                        # noqa: BLE001
                pass
        return line

    def emit(self, event: str, **fields: object) -> None:
        if event != "existence_claim_downgraded":
            return
        detail = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        self._rec("symbol", event, detail)

    def _rec(self, cylinder: str, event: str, detail: str) -> None:
        # E331: turn-stamped trace (stateframe_1's deferred instrumentation) — the loop
        # sets bolt.current_step each iteration so the shadow projector can slice the
        # kernel trace per turn instead of run_final_only.
        receipt = {"cylinder": cylinder, "event": event, "detail": detail,
                   "turn": getattr(self, "current_step", None)}
        self.trace.append(receipt)
        # E484/E509: ORGAN_DELIVERED is part of the same receipt tape used by the
        # cognitive governor.  Forward the exact _rec shape so a following read or
        # grep can be recognized as an organ-requested observation during replay and
        # in the live loop.  The governor is best-effort like the rest of this seam.
        if str(event).upper() == "ORGAN_DELIVERED":
            cogov = getattr(self, "cogov", None)
            if cogov is not None:
                try:
                    cogov.observe_receipt(receipt)
                except Exception as exc:                              # noqa: BLE001
                    print(f"    [kernel-wiring] cog-governor delivery receipt failed: "
                          f"{type(exc).__name__}")

    def report(self) -> str:
        lines = [f"KERNEL BOLT — task {self.task_id} ({self.fn})", "  cylinders:"]
        for k in ("symbol", "ledger", "gate", "graph", "skeptic"):
            lines.append(f"    {k:8s} {self.status.get(k, 'NOT STARTED')}")
        fired = {}
        for t in self.trace:
            fired[t["cylinder"]] = fired.get(t["cylinder"], 0) + 1
        lines.append(f"  events: {fired or 'NONE FIRED'}")
        lines.append(f"  repeats blocked: {self.repeats}   edits seen: {self.edits}")
        # The point of the graveyard read: a cylinder that started but never fired is as dead
        # as one that failed to start, and only this line distinguishes them.
        silent = [k for k in self.status
                  if self.status[k].startswith("engaged") and k not in fired]
        if silent:
            lines.append(f"  ENGAGED BUT NEVER FIRED: {', '.join(silent)}")
        return "\n".join(lines)


def _bolt_trigger_for(cylinder: str) -> Callable[[Evidence], TriState]:
    """Own the core-cylinder trigger shape next to KernelBolt's implementation."""

    def trigger(evidence: Evidence) -> TriState:
        record = evidence.event
        if evidence.source == "trace":
            if str(record.get("cylinder") or "").strip().lower() != cylinder:
                return TriState.NO
            event = str(record.get("event") or "").strip().lower()
            if event in {"organ_armed", "organ_disarmed", "boot_fingerprint"}:
                return TriState.NO
            return TriState.YES if event else TriState.UNKNOWN

        if evidence.source != "receipt":
            return TriState.NO
        kind = str(record.get("kind") or "").strip().lower()
        result = str(record.get("result_class") or record.get("result") or "").strip().upper()
        if cylinder == "symbol" and kind == "grep":
            if result == "NO_HITS" or "n=0" in str(record.get("detail") or ""):
                return TriState.YES
            return TriState.NO if result else TriState.UNKNOWN
        if cylinder in {"gate", "skeptic"} and kind == "test":
            return TriState.YES if result in {"RED", "INFRA_FAILURE", "NO_VERDICT"} else TriState.NO
        if cylinder == "ledger" and kind in {"grep", "read"}:
            return TriState.YES if result else TriState.UNKNOWN
        return TriState.NO

    trigger.__name__ = f"bolt_{cylinder.replace('-', '_')}_trigger"
    return trigger


# Each core cylinder keeps its own trigger predicate beside the bolt lifecycle.
# The census engine only invokes these declarations; it does not reproduce them.
CENSUS_SPECS = (
    OrganCensusSpec(
        organ_id="bolt.symbol",
        trigger=_bolt_trigger_for("symbol"),
        fire_events=frozenset({"answered_dead_grep"}),
        suppression_events=frozenset({"symbol_suppressed"}),
        dedup_events=frozenset({"symbol_deduped"}),
        challenge_once_events=frozenset({"symbol_challenge_once"}),
    ),
    OrganCensusSpec(
        organ_id="bolt.ledger",
        trigger=_bolt_trigger_for("ledger"),
        fire_events=frozenset({"ledger_replay"}),
        suppression_events=frozenset({"ledger_suppressed"}),
        dedup_events=frozenset({"repeat_blocked"}),
        challenge_once_events=frozenset({"ledger_challenge_once"}),
    ),
    OrganCensusSpec(
        organ_id="bolt.gate",
        trigger=_bolt_trigger_for("gate"),
        fire_events=frozenset({"nav_irrelevant_green", "nav_green_never_red"}),
        suppression_events=frozenset({"gate_suppressed"}),
        dedup_events=frozenset({"gate_deduped"}),
        challenge_once_events=frozenset({"challenged_done"}),
    ),
    OrganCensusSpec(
        organ_id="bolt.graph",
        trigger=_bolt_trigger_for("graph"),
        fire_events=frozenset({"call_path_card", "contract_card"}),
        suppression_events=frozenset({"graph_suppressed"}),
        dedup_events=frozenset({"graph_deduped"}),
        challenge_once_events=frozenset({"graph_challenge_once"}),
    ),
    OrganCensusSpec(
        organ_id="bolt.skeptic",
        trigger=_bolt_trigger_for("skeptic"),
        fire_events=frozenset({"sure_orient", "guard_at_anchor", "infra_failure", "edit_unrelated"}),
        suppression_events=frozenset({"skeptic_suppressed"}),
        dedup_events=frozenset({"skeptic_deduped"}),
        challenge_once_events=frozenset({"challenged_done"}),
    ),
)


def selftest() -> int:
    """Prove each cylinder fires at least once — planted controls, not hope."""
    configured_root = os.environ.get("STATEFRAME_SELFTEST_ROOT")
    if not configured_root:
        print("self-test repository unavailable; set STATEFRAME_SELFTEST_ROOT")
        return 2
    root = pathlib.Path(configured_root)
    if not root.exists():
        print("repo unavailable; cannot self-test")
        return 2
    kb = KernelBolt(root, "blocking_run_status", "selftest")

    print("--- cylinder 1: dead grep gets ground truth")
    out = kb.observe("grep blocking_run_status", "(no matches)")
    print("   ", "PASS" if "EXISTS" in out else f"FAIL: {out[:120]}")

    print("--- cylinder 2: repeated QUERY blocked")
    kb.observe("grep foo", "hit one")
    out2 = kb.observe("grep foo", "hit one")
    print("   ", "PASS" if "already ran" in out2 else f"FAIL: {out2[:120]}")

    print("--- cylinder 2 NEGATIVE CONTROL: a repeated EDIT is NOT blocked")
    # The control that matters. A model iterating on a fix issues the same command line with
    # different payloads — measured at 9x on one live task. Blocking that would destroy the
    # only behaviour the loop exists to produce.
    kb.observe("edit app/foo.py", "edit applied.")
    out5 = kb.observe("edit app/foo.py", "edit applied.")
    print("   ", "PASS" if "already ran" not in out5 else f"FAIL: blocked an edit -- {out5[:90]}")

    print("--- cylinder 3: done refused with zero edits")
    ok, why = kb.may_finish()
    print("   ", "PASS" if not ok and "not edited" in why else f"FAIL: {ok} {why[:80]}")

    print("--- cylinder 5: done challenged after a red test")
    kb.edits = 1
    kb.last_test_exit = 1
    ok2, why2 = kb.may_finish()
    print("   ", "PASS" if not ok2 and "did not pass" in why2 else f"FAIL: {ok2} {why2[:80]}")

    print("--- cylinder 3: done allowed when edited and green")
    kb.last_test_exit = 0
    ok3, _ = kb.may_finish()
    print("   ", "PASS" if ok3 else "FAIL: refused a legitimate finish")

    print("--- BOUNDED GOVERNANCE: every gate challenges at most ONCE")
    # antipattern review: an unbounded refusal is a trap, and a no-edit or red-test
    # finish can be legitimate (already fixed, unreproducible, external, diagnostic, or an
    # expected-red regression test). Re-asking after a challenge must be ACCEPTED.
    kb2 = KernelBolt(root, "x", "bounded")
    kb2.edits = 0
    first_ok, _ = kb2.may_finish()
    second_ok, _ = kb2.may_finish()
    print("   ", "PASS" if (not first_ok and second_ok)
          else f"FAIL: first={first_ok} second={second_ok} — gate is unbounded")
    kb3 = KernelBolt(root, "x", "bounded2")
    kb3.edits, kb3.last_test_exit = 1, 1
    f3, _ = kb3.may_finish()
    s3, _ = kb3.may_finish()
    print("   ", "PASS" if (not f3 and s3) else f"FAIL: red-test gate unbounded ({f3},{s3})")

    print("--- cylinder 5 NEGATIVE CONTROL: exit 4 (no oracle) must NOT block done")
    # The control that matters most. Blocking on a non-verdict exit code is INESCAPABLE — the
    # agent can never produce a green run when the suite cannot collect, so `done` is refused
    # forever. Measured live at 6 consecutive refusals before this control existed.
    kb.last_test_exit = 4
    ok4, why4 = kb.may_finish()
    print("   ", "PASS" if ok4 else f"FAIL: trapped the agent -- {why4[:90]}")
    kb.last_test_exit = 5
    ok5, _ = kb.may_finish()
    print("   ", "PASS" if ok5 else "FAIL: exit 5 (no tests collected) blocked done")

    print("--- cylinder 4: graph note")
    kb.note("hypothesis", "the guard is inverted")

    print()
    print(kb.report())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(selftest())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
