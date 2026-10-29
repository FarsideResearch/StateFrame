"""E340 HYPOTHESIS TRACKER — the MicroGraph cylinder, finally carrying evidence.

The kernel inventory row that built this: cylinder 4 has been "engaged" since it was
wired and has NEVER produced an evidence edge. `KernelBolt.note()` routes BOTH
`"hypothesis"` and `"evidence"` notes through `add_hypothesis()`, so the graph's
evidence / posture / discriminator APIs — the entire reason the module exists — are
dead weight, and the hypothesis IDs it mints use `len(self.trace)` as identity, which
is not stable across a replay. the wire spec (`sol_answers/cogwire_micrograph.md`)
names the safe boundary; this module implements the hypothesis half of it.

THE BOUNDARY:

  * ACTIVE hypotheses come from kernel-owned MEASURED facts only: a red-test anchor
    the traceback adapter produced. Never from model prose, never from a heuristic
    like "this anchor means the guard is inverted".
  * MODEL PROSE is a proposal receipt. It is recorded for continuity and rendered as
    UNVERIFIED. It never enters the graph, so it can never create, support, or
    contradict anything. `Hypothesis` has no PROPOSED lifecycle and overloading LIVE
    would make a guess look exactly like a measurement.
  * EVIDENCE comes only from E334 receipts — typed, fingerprinted, turn-stamped
    records of things that actually happened. Prose in, nothing out.
  * ELIMINATED requires a VALID HARD contradiction. Non-verdict exits (selector
    error, nothing collected, no verdict) are `validity=unknown` and relate to
    nothing: `update_postures()` skips them by contract, which is precisely the
    exit-4 trap that has already bitten this kernel twice.
  * NOTHING HERE BLOCKS. There is no veto, no retry, no completion effect. The worst
    this module does is hand `plan_summary()` four bounded lines of recorded state.

DEGRADATION: every detreason import happens inside a function body (cycle safety —
`kernel_bolt` imports this, this must never import back at module scope). Any import
or runtime failure flips the tracker to a permanent no-op and prints exactly ONE
`[kernel-wiring] hypotheses degraded: <exc>` line. Wiring standard rule 2: no silent
degradation, ever. A dark organ must be visible in the transcript.

SNITCH: `open_from_prose()` is the declaration channel — the model's stated theory is
recorded with its own id. `observe()` is its later-check: the receipts either produce
a measured hypothesis matching that theory or they do not. The tracker never punishes
an honest proposal; it just refuses to let one masquerade as evidence.
"""
from __future__ import annotations

import os

import hashlib
from typing import Callable, Iterable

# Bounded rendering budgets. The E321 capsule has a ~400-token budget for the WHOLE
# kernel state; four short lines is the graph's entire share of it.
_KILL_MAX_LINES = 4
_LINE_MAX_CHARS = 100
_FRONTIER_LIVE = 2
_FRONTIER_UNKNOWN = 1

# Receipt result classes that carry a real verdict. Everything else — SELECTOR_ERROR,
# NO_TESTS, NO_VERDICT, NO_OP, REJECTED, FAILED — is `validity=unknown` and relates to
# no hypothesis in either direction. Measured trap: a task where exit 4 was read as
# "the patch failed" and the agent thrashed for six turns against a suite that never
# collected. A non-verdict must never move a posture.
_TEST_VALID = {"GREEN", "RED"}
_EDIT_VALID = {"APPLIED"}


def _sha12(*parts: object) -> str:
    """Content-addressed identity: never use `len(trace)` as the identity source.

Stable identities allow receipts to be replayed without depending on log length.
"""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _basename(path: object) -> str:
    text = str(path or "").replace("\\", "/").strip()
    return text.rsplit("/", 1)[-1] if text else ""


def _clip(text: str, limit: int = _LINE_MAX_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _split_anchor(anchor: object) -> tuple[str, str, str]:
    """`app/users.py:88:load_user` -> (path, line, function).

    That is the exact shape `tracebacks.traceback_signal()['primary_anchor']` emits
    (tracebacks.py:1448). Tolerates the 2-part and 1-part forms rather than raising,
    because a partially-resolved anchor is still a usable subject."""
    text = str(anchor or "").strip()
    if not text:
        return "", "", ""
    bits = text.split(":")
    if len(bits) >= 3:
        return bits[0], bits[1], ":".join(bits[2:])
    if len(bits) == 2:
        return bits[0], bits[1], ""
    return bits[0], "", ""


class HypothesisTracker:
    """One MicroGraph per task: measured hypotheses, receipt-derived evidence edges.

    Every public method is best-effort and self-reporting. None of them raise into the
    ReAct loop, and none of them return anything the loop is obliged to act on."""

    def __init__(
        self,
        task_id: str,
        *,
        trace_cb: Callable[[str, str, str], None] | None = None,
        workspace_id: str = "react-repair",
        max_nodes: int = 256,
        max_edges: int = 1024,
    ) -> None:
        self.task_id = str(task_id or "task")
        self._trace_cb = trace_cb
        self.degraded = False
        self._mg = None
        self._consumer = None

        # hypothesis_id -> short human label used in every rendered line.
        self._labels: dict[str, str] = {}
        # hypothesis_id -> {"file", "fn", "family"} — the measured subject it refers to.
        self._subjects: dict[str, dict[str, str]] = {}
        # hypothesis_id -> short evidence id of the contradiction that killed it.
        # Write-once by construction: `eliminate()` refuses to overwrite an entry, so a
        # dead hypothesis can never resurrect even if later evidence supports it.
        self._eliminated: dict[str, str] = {}
        # Stale is distinct from falsified: it requires a fresh check but may
        # not be promoted back into the live frontier without that check.
        self._stale: dict[str, dict[str, object]] = {}
        # short evidence id ("r7") -> graph node, in arrival order. The short id is what
        # the kill-list quotes, so an operator can find the receipt in the E334 tape.
        self._evidence: dict[str, object] = {}
        self._evidence_seq = 0
        # Duplicate receipts must not create duplicate edges.
        self._seen_receipts: set[str] = set()
        # Model prose, kept for continuity and rendered UNVERIFIED. Never graph state.
        self._proposals: list[dict[str, str]] = []
        # The hypothesis an APPLIED edit was acting on, plus the red signature that was
        # standing when it landed — the input to the "edit then IDENTICAL red" test.
        self._open_edit: dict[str, str] | None = None
        self._last_red_sig = ""

        micrograph_setting = os.environ.get("STATEFRAME_MICROGRAPH", "off").strip().lower()
        if micrograph_setting != "off":
            from kernel_micrograph_wire import mode as micrograph_mode
            micrograph_setting = micrograph_mode()
        if micrograph_setting == "on":
            # StateFrameRuntime owns the sole graph in the typed route.
            self.degraded = True
        else:
            self._boot(workspace_id, max_nodes, max_edges)

    # ------------------------------------------------------------------ boot
    def _boot(self, workspace_id: str, max_nodes: int, max_edges: int) -> None:
        """Import inside the body (cycle safety) and arm, or degrade loudly once."""
        try:
            from detreason import micrograph as _mg

            self._mg = _mg
            self._consumer = _mg.HypothesisGraphConsumer(
                _mg.MicroGraph(workspace_id, self.task_id,
                               max_nodes=max_nodes, max_edges=max_edges)
            )
            # Wiring standard rule 1: an ARMED event at init, with config, so a
            # disabled or absent organ is visible in EVERY transcript rather than
            # being indistinguishable from one that simply never had anything to say.
            self._trace("hypotheses_armed",
                        f"nodes<={max_nodes} edges<={max_edges} ws={workspace_id}")
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)

    def _degrade(self, exc: BaseException) -> None:
        """One line, naming the exception, then permanent silence (#113)."""
        if not self.degraded:
            self.degraded = True
            print(f"[kernel-wiring] hypotheses degraded: {type(exc).__name__}: {exc}")
            self._trace("hypotheses_disarmed", f"{type(exc).__name__}: {exc}")
        self._mg = None
        self._consumer = None

    @property
    def graph(self) -> object | None:
        """The one task MicroGraph, exposed for the DEPGRAPH projection."""
        return getattr(self._consumer, "graph", None)

    @property
    def postures(self) -> dict[str, object]:
        """Current graph-derived hypothesis postures for selector pins."""
        if self._consumer is None:
            return {}
        try:
            return dict(self._consumer.update_postures())
        except Exception as exc:                                    # noqa: BLE001
            self._degrade(exc)
            return {}

    def _trace(self, event: str, detail: str) -> None:
        """Every state change is turn-stamped by the bolt's `_rec` (wiring rule 3).

        A failing trace callback must never take the loop down with it — the trace is
        observability, not control flow."""
        cb = self._trace_cb
        if cb is None:
            return
        try:
            cb("graph", event, str(detail)[:150])
        except Exception:                                            # noqa: BLE001
            self._trace_cb = None

    # ------------------------------------------------------ opening hypotheses
    def open_from_anchor(self, anchor: str, family: str = "") -> str | None:
        """Open a KERNEL-DERIVED hypothesis from a measured red-test anchor.

        The claim is deliberately weak — "the defect involves <file>:<fn>" — because an
        anchor proves WHERE the failure surfaced, not which layer OWNS the defect.
        design review rejects the tempting upgrade ("this anchor means the guard is inverted") as
        a heuristic masquerading as a fact; `family` is carried as a label only, never
        as a semantic claim about the cause.

        Idempotent: the id is a content hash of the subject, so the same anchor seen on
        turn 3 and turn 19 is ONE hypothesis accumulating evidence, not two."""
        if self._consumer is None:
            return None
        try:
            path, _line, fn = _split_anchor(anchor)
            if not path:
                return None                      # an empty anchor is not a measurement
            fam = " ".join(str(family or "").split())[:64] or "defect"
            subject = f"{path}:{fn}"
            hid = f"h:{self.task_id}:anchor:{_sha12(path, fn, fam)}"
            if hid in self._eliminated:
                self._trace("hypothesis_reproposal_blocked", f"{hid} is permanently falsified")
                return None
            if hid in self._subjects:
                return hid
            label = _clip(f"{fam}-in-{fn or _basename(path)}", 60)
            self._consumer.add_hypothesis({
                "hypothesis_id": hid,
                "claim": f"the defect involves {subject}",
                # `required` is the predicate the graph matches evidence against. One
                # exact predicate, so a grep miss or an unrelated green cannot resolve
                # it by accident.
                "required": [{"predicate": "defect_site", "path": path,
                              "operator": "equals", "value": fn or path}],
                "provenance": [{"source": "kernel", "kind": "traceback_anchor",
                                "anchor": str(anchor)[:120]}],
            })
            self._labels[hid] = label
            self._subjects[hid] = {"file": path, "fn": fn, "family": fam}
            self._trace("hypothesis_opened", f"{hid} {label}")
            return hid
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return None

    def open_from_prose(self, text: str) -> str | None:
        """Record a MODEL-PROSE theory as an UNVERIFIED proposal. Never a hypothesis.

        This is the whole defect the module fixes: `note("hypothesis", prose)` used to
        push free-form model text straight into `add_hypothesis()`, where it sat beside
        measured facts with identical standing. A proposal here reaches the graph
        NEVER, is excluded from `frontier()` and `kill_list()`, and — critically —
        cannot be supported or contradicted, because prose creates no evidence."""
        if self.degraded:
            return None
        try:
            body = " ".join(str(text or "").split())[:400]
            if not body:
                return None
            pid = f"p:{self.task_id}:{_sha12(body)}"
            if any(p["id"] == pid for p in self._proposals):
                return pid                        # dedup: a repeated theory is one theory
            if len(self._proposals) < 32:          # bounded; prose is cheap to emit
                self._proposals.append({"id": pid, "text": body,
                                        "provenance": "model_prose", "status": "UNVERIFIED"})
            self._trace("proposal_recorded", f"{pid} UNVERIFIED {body[:60]}")
            return pid
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return None

    # ---------------------------------------------------------------- evidence
    def observe(self, receipt: dict) -> str | None:
        """Turn ONE E334 receipt into graph evidence. Returns the short evidence id.

        The receipt tape is the only admissible input: it is typed, fingerprinted and
        turn-stamped, so an evidence node can always be traced back to a thing that
        measurably happened. The relation policy is table:

          RED test, anchored at a hypothesis' file -> SUPPORT (failure still there)
          GREEN test after an APPLIED edit on that
            hypothesis' file -> SUPPORT (the site mattered)
          APPLIED edit then the IDENTICAL red again -> CONTRADICT the hypothesis that
                                                        motivated the edit
          anything non-verdict (exit 2/3/4/5, no-op,
            rejected, failed edit, done, verdict) -> recorded, relates to NOTHING

        The last row is the one that keeps this safe. A selector that collected nothing
        says NOTHING about the patch; treating it as a failure is the inescapable-gate
        bug this kernel has already shipped twice."""
        if self._consumer is None or not isinstance(receipt, dict):
            return None
        try:
            kind = str(receipt.get("kind", "")).strip()
            result = str(receipt.get("result_class", "")).strip().upper()
            # Dedup on the receipt's OWN fingerprint plus its class: replaying the same
            # receipt must not double an edge or double-count support.
            dedup = f"{kind}|{receipt.get('fingerprint', '')}|{result}|{receipt.get('turn')}"
            if dedup in self._seen_receipts:
                self._trace("evidence_deduped", dedup[:80])
                return None
            self._seen_receipts.add(dedup)

            valid = (kind == "test" and result in _TEST_VALID) or \
                    (kind == "edit" and result in _EDIT_VALID)
            validity = "valid" if valid else "unknown"
            eid = self._add_evidence(receipt, kind, result, validity)
            if not valid:
                # Recorded so the tape is complete, related to nothing so it is inert.
                self._trace("evidence_unknown", f"{eid} {kind}/{result or 'none'}")
                return eid

            if kind == "edit":
                self._relate_edit(receipt, eid)
            else:
                self._relate_test(receipt, eid, result)
            return eid
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return None

    def _add_evidence(self, receipt: dict, kind: str, result: str, validity: str) -> str:
        """Mint a bounded evidence node. Short ids ("r7") are for DISPLAY only; the
        graph keys on the node's own content-addressed id."""
        self._evidence_seq += 1
        short = f"r{self._evidence_seq}"
        attrs = {
            "event_kind": kind,
            "event_id": short,
            "result_class": result,
            "step": receipt.get("turn"),
            "source_generation": receipt.get("workspace_revision", 0),
            "validity": validity,
            "payload_sha": str(receipt.get("fingerprint", ""))[:16],
            "producer": "kernel_hypotheses/1",
            # The predicate this evidence speaks to. Present so `_fact_matches()` can
            # resolve a required predicate instead of matching on absence.
            "predicate": "defect_site",
            "operator": "equals",
        }
        anchor_file, _, anchor_fn = _split_anchor(receipt.get("anchor"))
        if anchor_file:
            attrs["path"] = anchor_file
            attrs["value"] = anchor_fn or anchor_file
        node = self._mg.make_node(self._mg.EntityKind.EVIDENCE,
                                  f"{self.task_id}:{short}:{attrs['payload_sha']}", attrs)
        self._consumer.add_evidence(node)
        self._evidence[short] = node
        return short

    def _receipt_files(self, receipt: dict, kind: str) -> set[str]:
        """Which files this receipt is ABOUT, resolved fail-closed.

        For an edit the intent carries the filename (`str_replace guard.py`) so the
        basename is admissible. For a TEST the intent is a pytest SELECTOR — the test
        file, not the product file — so matching on it would attach product evidence to
        the wrong subject every time. A test receipt therefore resolves its file ONLY
        from an explicit `anchor`/`path`, and an unanchored red relates to nothing."""
        files: set[str] = set()
        for key in ("path", "anchor"):
            value = receipt.get(key)
            if value:
                head = _split_anchor(value)[0] if key == "anchor" else value
                if _basename(head):
                    files.add(_basename(head))
        if kind == "edit":
            for token in str(receipt.get("intent", "")).split():
                if token.endswith(".py"):
                    files.add(_basename(token))
        return files

    def _live_ids(self) -> tuple[str, ...]:
        """Hypotheses the graph currently calls LIVE, canonically sorted."""
        postures = self._consumer.update_postures()
        return tuple(sorted(hid for hid, p in postures.items()
                            if p.posture == self._mg.LIVE and hid not in self._eliminated))

    def _match(self, files: set[str], ids: Iterable[str]) -> list[str]:
        return sorted(hid for hid in ids
                      if _basename(self._subjects.get(hid, {}).get("file")) in files)

    def _relate_edit(self, receipt: dict, eid: str) -> None:
        """An APPLIED edit on a hypothesis' file OPENS a falsifiable expectation.

        It is not itself support — a landed patch proves generation advanced, never
        that the patch was correct (design review: "Supports only `generation_advanced`; never
        patch correctness"). What it does is arm the next test: if the identical red
        comes back, the site this edit was acting on is contradicted."""
        if "target_path" in receipt:
            # Executor identity is authoritative; prose cannot broaden its subject.
            target = receipt["target_path"]
            matched = sorted(
                hid for hid in self._live_ids()
                if isinstance(target, str) and target
                and self._subjects.get(hid, {}).get("file") == target
            )
        else:
            matched = self._match(self._receipt_files(receipt, "edit"), self._live_ids())
        if not matched:
            self._trace("edit_unmatched", f"{eid} no live hypothesis for this file")
            return
        hid = matched[0]
        self._open_edit = {"hid": hid, "red_sig": self._last_red_sig, "evidence": eid}
        self._trace("edit_armed", f"{eid} -> {hid} prior_red={self._last_red_sig[:40]}")

    def _relate_test(self, receipt: dict, eid: str, result: str) -> None:
        """RED supports the site it anchors at; GREEN closes an armed edit; an armed
        edit answered by the IDENTICAL red contradicts the site it was acting on."""
        sig = f"{receipt.get('intent', '')}|{receipt.get('anchor', '')}"
        if result == "RED":
            # 1. The armed-edit check FIRST: same selector, same anchor, still red ->
            #    the edit at that site did not move the failure. That is a measured
            #    hard contradiction of "the defect involves <that site>".
            armed = self._open_edit
            if armed and armed["red_sig"] and armed["red_sig"] == sig:
                self._open_edit = None
                self.eliminate(armed["hid"], eid,
                               reason="identical red after an applied edit at this site")
            else:
                # 2. Otherwise it is ordinary reproduction evidence for whatever site
                #    the traceback ANCHORS at — never for the test file it ran from.
                for hid in self._match(self._receipt_files(receipt, "test"),
                                       self._live_ids()):
                    self._consumer.support(hid, self._evidence[eid], validity="valid",
                                           predicate="defect_site")
                    self._trace("evidence_supports", f"{eid} -> {hid} (red reproduced)")
            self._last_red_sig = sig
            return

        # GREEN. It may only close an edit that was armed on a specific hypothesis;
        # a green with nothing armed is a green that proves nothing about a site.
        armed = self._open_edit
        self._open_edit = None
        if not armed:
            self._trace("green_unarmed", f"{eid} no armed edit; no relation")
            return
        hid = armed["hid"]
        if hid in self._eliminated:
            self._trace("green_ignored_eliminated", f"{eid} -> {hid}")
            return
        self._consumer.support(hid, self._evidence[eid], validity="valid",
                               predicate="defect_site")
        self._trace("evidence_supports", f"{eid} -> {hid} (red->green at this site)")

    # -------------------------------------------------------------- elimination
    def eliminate(self, hypothesis_id: str, evidence_id: str, reason: str = "") -> bool:
        """Stamp ELIMINATED with the evidence that did it. WRITE-ONCE.

        MicroGraph recomputes posture from edges rather than storing a stamp, so
        elimination is sticky for as long as a valid hard `contradicts` edge exists —
        and this module never removes an edge. The local `_eliminated` map is the
        safeguard: once a hypothesis is in it, no later support path will touch it and the
        kill-list keeps quoting the ORIGINAL contradiction. trap review flags
        sticky elimination as a real risk, which is exactly why the only door into
        this method is a VALID, HARD, measured contradiction."""
        if self._consumer is None:
            return False
        node = self._evidence.get(str(evidence_id))
        if node is None or hypothesis_id not in self._subjects:
            return False
        if hypothesis_id in self._eliminated:
            return False                        # never re-stamp; never resurrect either
        try:
            self._consumer.contradict(hypothesis_id, node, validity="valid",
                                      soft=False, assessment="contradiction")
            self._eliminated[hypothesis_id] = str(evidence_id)
            self._trace("hypothesis_eliminated",
                        f"{hypothesis_id} e={evidence_id} {reason or 'valid hard contradiction'}")
            return True
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return False

    def mark_stale(
        self, hypothesis_id: str, *, reason: str = "EVIDENCE_STALE",
        evidence_refs: Iterable[str] = (),
    ) -> bool:
        """Mark a live hypothesis for re-check without weakening falsification."""
        if hypothesis_id not in self._subjects or hypothesis_id in self._eliminated:
            return False
        self._stale[hypothesis_id] = {
            "reason": str(reason or "EVIDENCE_STALE"),
            "evidence_refs": tuple(str(item) for item in evidence_refs if str(item)),
        }
        self._trace("hypothesis_stale", f"{hypothesis_id} {reason or 'EVIDENCE_STALE'}")
        return True

    def mark_path_stale(
        self, path: str, *, reason: str = "EVIDENCE_STALE",
        evidence_refs: Iterable[str] = (),
    ) -> tuple[str, ...]:
        """Apply dependency staleness to hypotheses whose measured subject is a path."""
        normalized = str(path or "").replace("\\", "/").strip().lstrip("./")
        changed = [
            hid for hid, subject in self._subjects.items()
            if normalized and (subject.get("file") == normalized
                                or str(subject.get("file", "")).endswith("/" + normalized))
            and self.mark_stale(hid, reason=reason, evidence_refs=evidence_refs)
        ]
        return tuple(changed)

    def recheck_required(self) -> list[dict[str, object]]:
        """Return stale hypotheses in stable order for the next verification step."""
        return [
            {"hypothesis_id": hid, **dict(self._stale[hid])}
            for hid in sorted(self._stale)
            if hid not in self._eliminated
        ]

    def clear_stale(self, hypothesis_id: str) -> bool:
        return self._stale.pop(str(hypothesis_id), None) is not None

    # ----------------------------------------------------------------- rendering
    def kill_list(self) -> list[str]:
        """Bounded negative work for the E321 capsule: what is PROVABLY not the answer.

        The single highest-value thing a trimmed conversation can carry is the list of
        theories already killed — a model that cannot see its own history re-derives
        and re-tries them. Capped at four short lines because the whole capsule budget
        is ~400 tokens; an overflow renders as an explicit omission count so the
        boundedness is never mistaken for completeness."""
        if self._consumer is None or not self._eliminated:
            return []
        try:
            rows = [
                _clip(f"ELIMINATED: {self._labels.get(hid, hid)} "
                      f"(contradicted by receipt {self._eliminated[hid]})")
                for hid in sorted(self._eliminated)
            ]
            if len(rows) > _KILL_MAX_LINES:
                keep = _KILL_MAX_LINES - 1
                rows = rows[:keep] + [_clip(f"ELIMINATED: +{len(rows) - keep} more omitted")]
            return rows
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return []

    def frontier(self) -> list[str]:
        """At most 2 LIVE hypotheses + 1 UNKNOWN, one line each.

        LIVE here means "measured, not killed, and carrying at least one supporting
        receipt". A hypothesis the graph calls LIVE but which nothing has ever touched
        is UNKNOWN in the honest sense, and gets at most one line saying so. UNKNOWN is
        otherwise SILENT: when there is nothing measured, this returns [] rather than a
        placeholder — a kernel that narrates its own emptiness is noise."""
        if self._consumer is None:
            return []
        try:
            postures = self._consumer.update_postures()
            live, unknown = [], []
            for hid in sorted(self._subjects):
                if hid in self._eliminated:
                    continue
                posture = postures.get(hid)
                if posture is None or posture.posture == self._mg.ELIMINATED:
                    continue
                label = self._labels.get(hid, hid)
                n = len(posture.support_evidence_ids)
                if n:
                    live.append(_clip(f"LIVE: {label} ({n} supporting receipt"
                                      f"{'s' if n != 1 else ''})"))
                else:
                    unknown.append(_clip(f"UNKNOWN: {label} (no measured evidence yet)"))
            return live[:_FRONTIER_LIVE] + unknown[:_FRONTIER_UNKNOWN]
        except Exception as exc:                                     # noqa: BLE001
            self._degrade(exc)
            return []

    def proposals(self) -> list[dict[str, str]]:
        """Model prose, verbatim and clearly labelled UNVERIFIED. Deliberately NOT part
        of `frontier()` — a caller that wants to show a guess must ask for it by name."""
        return [dict(p) for p in self._proposals]

    def capsule(self) -> str:
        """The graph's slice of the E321 trim capsule: frontier then kill-list, or ""."""
        rows = self.frontier() + self.kill_list()
        return "\n".join("  " + r for r in rows) if rows else ""
