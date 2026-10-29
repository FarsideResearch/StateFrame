"""Durable host executive projection using the existing single-seat scheduler.

No model tool exposure or worker restart entry point is provided here. Live
StateFrame integration must call this adapter at its actual dispatch boundary.
"""
from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

from detreason.juggler import LaneRegistry, Scheduler, SchedulerPolicy
from detreason.juggler_vocab import PRIORITY_P1_UNBLOCK, PRIORITY_P2_WORK
from kernel_continuation_evidence import HostEvidence, READ_RECEIPT_KINDS, binding_versions
from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_menu_authority import state_allow_set
from kernel_continuation_state import ContinuationError, ContinuationState, transition, unresolved
from kernel_measurement import canonical_bytes, digest


class ContinuationExecutive:
    def __init__(self, journal: ContinuationJournal, *, policy: SchedulerPolicy | None = None,
                 clock_us: Callable[[], int] | None = None):
        if not isinstance(journal, ContinuationJournal):
            raise ContinuationError("executive requires the exclusive resumable journal")
        self.journal = journal
        self.policy = policy or SchedulerPolicy()
        self.clock_us = clock_us or (lambda: time.time_ns() // 1000)
        self._state = ContinuationState()
        # The projected state changes only at the reducer boundary.  Re-hashing
        # the unchanged pre-transition document for both observer and publisher
        # made continuation cost grow with event history.
        self._state_hash = digest(self._state.document())
        self._evidence = HostEvidence(journal.run_id,
            source_refresh=journal.manifest.get("continuation_source_refresh") is True,
            native_prefix=journal.manifest.get("native_prefix_authority") is True,
            directory=journal.path.parent)
        self._sequence = 0
        self._unsettled_effect_ids: set[str] = set()
        self._policy_record: dict | None = None
        self._planned_artifacts = None
        self._return_ledger = None
        self._retention = None
        self._verified_checkpoint: dict | None = None
        if self._retention_enabled():
            from kernel_continuation_retention import ContinuationRetention
            self._retention = ContinuationRetention(journal)
        journal.subscribe("continuation_executive", self._observe)
        with journal.ownership():
            policy_record = {"schema": "continuation-policy/1", "scheduler": asdict(self.policy)}
            if self._policy_record is None:
                journal.emit("continuation_policy", policy=policy_record)
            elif self._policy_record != policy_record:
                raise ContinuationError("scheduler policy changed on resume")

    @property
    def state(self) -> ContinuationState:
        with self.journal.ownership():
            return copy.deepcopy(self._state)

    def active_acceptance_selectors(self) -> tuple[str, ...] | None:
        """Read detached selectors; None preserves the owner's fallback contract."""
        with self.journal.ownership():
            identity = self._state.active_id
            item = self._state.intentions.get(identity) if identity is not None else None
            acceptance = getattr(item, "acceptance", None)
            if isinstance(acceptance, Mapping):
                selectors = acceptance.get("selectors")
                if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
                    return tuple(str(value) for value in selectors if str(value).strip())
            return None

    def active_intention_id(self) -> str | None:
        with self.journal.ownership():
            return self._state.active_id

    def active_menu_state(self) -> tuple[str | None, bool, bool, bool]:
        """Return only identity, active phase, unsettled effects and stale status."""
        with self.journal.ownership():
            identity = self._state.active_id
            item = self._state.intentions.get(identity) if identity is not None else None
            return (identity,
                    item is not None and getattr(item, "phase", "active") == "active",
                    bool(unresolved(self._state, identity)) if item is not None else False,
                    bool(getattr(item, "stale_dependencies", ())))

    def per_step_allow_set(self) -> frozenset[str]:
        """Pure name-level protocol permissions for the current intention.

        This is deliberately separate from ``authorize``: authorization emits
        a dispatch fence and may reconcile a retry, while menu construction
        must only read a stable projection of the executive state.
        """
        with self.journal.ownership():
            return state_allow_set(self._state)

    def has_unsettled_effects(self) -> bool:
        """Return the unresolved-effect summary without copying the full state."""
        with self.journal.ownership():
            return bool(self._unsettled_effect_ids)

    @staticmethod
    def _retention_enabled() -> bool:
        value = os.environ.get("KERNEL_CONTINUATION_RETENTION_GC", "off").strip().lower()
        if value not in {"off", "on"}:
            raise ValueError("KERNEL_CONTINUATION_RETENTION_GC must be off or on")
        if value == "off":
            # The retention module is intentionally omitted from OFF manifests;
            # do not import an optional module merely to discover that it is off.
            return False
        from kernel_continuation_retention import enabled
        return enabled()

    @property
    def retention(self):
        return self._retention

    def register_checkpoint(self, event: dict) -> None:
        """Bind retention to a committed worker checkpoint when restart is on."""
        if self._retention is None:
            return
        if (not isinstance(event, dict)
                or event.get("kind") != "continuation_worker_checkpoint"
                or not isinstance(event.get("checkpoint"), dict)
                or type(event.get("sequence")) is not int
                or event.get("sequence") != self._sequence
                or not isinstance(event.get("record_hash"), str)
                or len(event["record_hash"]) != 71
                or not event["record_hash"].startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in event["record_hash"][7:])
                or event.get("checkpoint_hash") != digest(event["checkpoint"])):
            raise ContinuationError("invalid retention checkpoint")
        self._verified_checkpoint = {
            "sequence": event["sequence"], "head": event["record_hash"],
            "state_hash": self._state_hash,
        }

    def retention_checkpoint(self) -> dict | None:
        """Archive and publish one safe reclamation batch, if one is ready.

        The current projected state and journal head are the verified archive
        boundary for non-restart continuation.  A worker checkpoint, when
        available, is used only while it still matches that exact boundary.
        """
        if self._retention is None:
            return None
        with self.journal.ownership():
            state = self._state
            current_hash = self._state_hash
            marker = self._verified_checkpoint
            if (marker is None or marker["sequence"] != self._sequence
                    or marker["head"] != self.journal.head
                    or marker["state_hash"] != current_hash):
                marker = {"sequence": self._sequence, "head": self.journal.head,
                          "state_hash": current_hash}
            plan = self._retention.plan(state)
            if plan.empty:
                return None
            batch = self._retention.archive(
                plan, checkpoint_sequence=marker["sequence"],
                checkpoint_head=marker["head"])
            compacted = self._retention.compact(state, plan)
            return self.journal.emit(
                "continuation_retention_gc", retention_schema="continuation-retention/1",
                checkpoint_sequence=marker["sequence"], checkpoint_head=marker["head"],
                checkpoint_state_hash=marker["state_hash"],
                batch_id=batch.batch_id, archive_path=str(self._retention.archive_path),
                archive_start=batch.start, archive_end=batch.end,
                archive_head=batch.tail_hash, archive_digest=batch.batch_digest,
                intention_ids=list(plan.intention_ids), effect_ids=list(plan.effect_ids),
                wake_tokens=list(plan.wake_tokens), signal_ids=list(plan.signal_ids),
                return_receipts=list(plan.return_receipts),
                reclaimed_digest=plan.ids_digest(),
                after_state_hash=digest(compacted.document()),
            )

    def retention_summary(self) -> dict | None:
        if self._retention is None:
            return None
        return self._retention.summary(self._state)

    # --------------------------------------------------------- planned artifacts seam
    def enable_planned_artifacts(self, root: str, *, package_roots: tuple[str, ...] = (),
                                 obligation_ledger=None):
        """Attach planned inputs and return records to this existing executive."""
        from kernel_planned_artifacts import PlannedArtifactLedger, ReturnLedger, enabled
        if not enabled():
            return None
        if self._planned_artifacts is None:
            self._planned_artifacts = PlannedArtifactLedger(
                root, workspace_id=self.journal.task_id, session_id=self.journal.run_id,
                package_roots=package_roots, obligation_ledger=obligation_ledger)
            self._return_ledger = ReturnLedger()
        return self._planned_artifacts

    @property
    def planned_artifacts(self):
        return self._planned_artifacts

    @property
    def return_ledger(self):
        return self._return_ledger

    def capture_resumption_cue(self, parent_id: str, *, interrupted_by: str,
                               focus: str, last_evidence=(), return_target: str = "",
                               trigger: str, owed_items=(), waiting_checks=(),
                               priority: int = 0, affinity: int = 0, now: int = 0,
                               proposed_next_action: str = "", snapshot_refs=()):
        if self._return_ledger is None:
            return None
        cue = self._return_ledger.interrupt(
            parent_id, interrupted_by=interrupted_by, focus=focus,
            last_evidence=last_evidence, return_target=return_target,
            trigger=trigger, owed_items=owed_items, waiting_checks=waiting_checks,
            priority=priority, affinity=affinity, now=now,
            proposed_next_action=proposed_next_action, snapshot_refs=snapshot_refs)
        with self.journal.ownership():
            self.journal.emit("continuation_return_cue", parent_id=parent_id,
                              interrupted_by=interrupted_by, cue=cue.to_dict(),
                              priority=priority, affinity=affinity)
        return cue

    def mark_continuation_return(self, child_id: str, *, now: int = 0,
                                 evidence=(), receipt: str = ""):
        if self._return_ledger is None:
            return ()
        parents = self._return_ledger.mark_complete(
            child_id, now=now, evidence=evidence, receipt=receipt)
        with self.journal.ownership():
            self.journal.emit("continuation_return_ready", child_id=child_id,
                              parent_ids=list(parents), evidence=list(evidence),
                              receipt=receipt)
        return parents

    def reconcile_continuation(self, parent_id: str, *, evidence=(), now: int = 0):
        if self._return_ledger is None:
            return None
        record = self._return_ledger.reconcile(parent_id, evidence=evidence, now=now)
        if record is not None:
            with self.journal.ownership():
                self.journal.emit("continuation_return_reconciled",
                                  parent_id=parent_id, stale=record.stale,
                                  evidence=list(evidence))
        return record

    def continuation_projection(self, parent_id: str):
        if self._return_ledger is None:
            return ""
        return self._return_ledger.projection(parent_id)

    def continuation_projection_dict(self, parent_id: str):
        if self._return_ledger is None:
            return {}
        return self._return_ledger.projection_dict(parent_id)

    def planned_completion_blocker(self, *, now: int = 0) -> str | None:
        if self._planned_artifacts is None:
            return None
        blockers = self._planned_artifacts.completion_blockers(now=now)
        return ("[planned artifacts] unresolved required inputs: "
                + ", ".join(blockers)) if blockers else None

    def _observe(self, event: dict) -> None:
        if (event.get("run_id") != self.journal.run_id
                or event.get("sequence") != self._sequence + 1):
            raise ContinuationError("projection event gap or foreign run")
        kind = event["kind"]
        if kind == "continuation_policy":
            if self._policy_record is not None:
                raise ContinuationError("duplicate continuation policy")
            policy_record = event.get("policy")
            if not isinstance(policy_record, dict) or policy_record.get("schema") != "continuation-policy/1":
                raise ContinuationError("invalid continuation policy")
            self._policy_record = copy.deepcopy(policy_record)
        elif kind == "continuation_transition":
            if (self._policy_record is None or event.get("transition_schema") != "continuation-transition/1"
                    or event.get("previous_revision") != self._state.revision
                    or event.get("before_hash") != self._state_hash):
                raise ContinuationError("continuation revision or previous state mismatch")
            operation, data = event.get("operation"), event.get("data")
            if not isinstance(operation, str) or not isinstance(data, dict):
                raise ContinuationError("invalid continuation transition")
            if operation == "begin_effect" and data.get("intent_sequence") != event["sequence"]:
                raise ContinuationError("effect intent sequence mismatch")
            self._evidence.validate(self._state, operation, data)
            staged = transition(self._state, operation, data)
            staged_hash = digest(staged.document())
            if event.get("after_hash") != staged_hash:
                raise ContinuationError("continuation resulting state mismatch")
            if operation == "begin_effect":
                self._unsettled_effect_ids.add(str(data["effect_id"]))
            elif operation in {"settle_effect", "repair_effect"}:
                effect_id = str(data["effect_id"])
                effect = staged.effects.get(effect_id)
                if effect is not None and effect.status in {"pending", "unknown"}:
                    self._unsettled_effect_ids.add(effect_id)
                else:
                    self._unsettled_effect_ids.discard(effect_id)
            self._state = staged
            self._state_hash = staged_hash
        elif kind == "continuation_retention_gc":
            if self._retention is None:
                raise ContinuationError("retention GC journal requires its switch")
            self._state = self._retention.apply_event(
                self._state, event, expected_sequence=self._sequence)
            self._unsettled_effect_ids = {
                effect.effect_id for effect in self._state.effects.values()
                if effect.status in {"pending", "unknown"}
            }
            self._state_hash = digest(self._state.document())
        self._evidence.observe(event)
        self._sequence = event["sequence"]

    def apply(self, operation: str, **fields: object) -> dict:
        """Validate, fsync, then publish via the journal's ordered observer."""
        with self.journal.ownership():
            data = json.loads(canonical_bytes(fields))
            now = self.clock_us()
            if type(now) is not int or now < 0:
                raise ContinuationError("host clock must return nonnegative microseconds")
            data["now_us"] = max(now, self._state.last_time_us)
            if operation == "complete":
                blocker = self.planned_completion_blocker(now=data["now_us"])
                if blocker is not None:
                    self.journal.emit(
                        "continuation_transition_rejected",
                        operation=operation, input_hash=digest(data),
                        state_revision=self._state.revision,
                        intention_id=data.get("intention_id"),
                        reason=blocker,
                    )
                    raise ContinuationError(blocker)
            if operation == "begin_effect":
                data["intent_sequence"] = self._sequence + 1
            try:
                if self._retention is not None:
                    self._retention.validate_operation(operation, data)
                if operation == "settle_effect":
                    _, closed = self._evidence._closed(data.get("receipt_ref"))
                    data["binding_changed"] = closed.get("before_binding") != closed.get("after_binding")
                self._evidence.validate(self._state, operation, data)
                staged = transition(self._state, operation, data)
            except ContinuationError as error:
                self.journal.emit("continuation_transition_rejected", operation=operation,
                                  input_hash=digest(data), state_revision=self._state.revision,
                                  intention_id=data.get("intention_id"), effect_id=data.get("effect_id"),
                                  reason=str(error))
                raise
            before_hash = self._state_hash
            after_hash = digest(staged.document())
            event = self.journal.emit(
                "continuation_transition", transition_schema="continuation-transition/1",
                previous_revision=self._state.revision, before_hash=before_hash,
                after_hash=after_hash, operation=operation, data=data)
            if self._state != staged:
                raise ContinuationError("committed transition was not projected")
            return event

    def current_source_evidence(self, reference: str, current_binding: dict) -> dict:
        """Return validated observed source, without granting completion authority."""
        with self.journal.ownership():
            return copy.deepcopy(self._evidence._read(reference, current_binding))

    def settle_host_effect(self, *, effect_id: str, receipt_ref: str,
                           current_binding: dict) -> dict:
        """Derive outcome under the writer lock, then durably validate and publish."""
        with self.journal.ownership():
            data = json.loads(canonical_bytes({
                "effect_id": effect_id, "receipt_ref": receipt_ref,
                "current_binding": current_binding}))
            outcome = self._evidence.effect_outcome(self._state, data)
            event = self.apply("settle_effect", **data, status=outcome)
            if outcome == "not_applied":
                from kernel_greenlock_terminal_fixes import note_settled_by_rule
                note_settled_by_rule(self.journal, self._evidence, effect_id)
            return event

    def settle_external_effect(self, *, effect_id: str, outcome: str,
                               external_event_ref: str, current_binding: dict) -> dict:
        """Bridge a verified external tri-state into ordinary host evidence."""
        if outcome not in {"PRESENT", "ABSENT"}:
            raise ContinuationError("external outcome must be PRESENT or ABSENT")
        with self.journal.ownership():
            effect = self._state.effects.get(effect_id)
            if effect is None or effect.status not in {"pending", "unknown"}:
                raise ContinuationError("no unresolved effect to settle externally")
            external = next((record for record in self.journal.records()
                             if record.get("event_id") == external_event_ref), None)
            if not isinstance(external, dict):
                raise ContinuationError("missing external outcome evidence")
            if external.get("kind") == "guarded_effect_receipt":
                valid = (external.get("effect_id") == effect_id
                         and external.get("outcome") == outcome)
            elif external.get("kind") == "external_effect_outcome":
                key = external.get("idempotency_key")
                valid = (isinstance(key, list) and effect_id in key
                         and external.get("status") == outcome)
            else:
                valid = False
            if not valid:
                raise ContinuationError("external outcome is not bound to the effect")
            if outcome == "ABSENT":
                result, changes, receipt_kind = "NO_OP", [], "external"
            elif effect.tool in READ_RECEIPT_KINDS:
                result = {"read": "OK", "text": "OK", "grep": "HITS", "ls": "OK",
                          "recall_evidence": "RECALLED",
                          "recall_generated_artifact": "RECALLED",
                          "probe": "COMPLETED"}.get(effect.tool, "OK")
                changes, receipt_kind = [], READ_RECEIPT_KINDS[effect.tool]
            elif effect.tool == "run_tests":
                result, changes, receipt_kind = "GREEN", [], "test"
                self.journal.emit(
                    "test_subprocess_completed", producer="test_subprocess",
                    selectors=[], exit_code=0, binding=current_binding,
                    after_binding=current_binding, binding_stable=True,
                    request_id=effect.request_id, continuation_id=effect.intention_id,
                    continuation_effect_id=effect.effect_id,
                    continuation_dispatch_ref=effect.dispatch_ref)
            else:
                result, changes, receipt_kind = "APPLIED", [external_event_ref], "external"
            receipt_id = "external:" + digest(external_event_ref)
            self.journal.emit("action_receipt", receipt={
                "receipt_id": receipt_id, "run_id": self.journal.run_id,
                "request_id": effect.request_id, "turn": effect.step,
                "kind": receipt_kind, "result_class": result,
                "continuation_id": effect.intention_id,
                "continuation_effect_id": effect.effect_id,
                "continuation_dispatch_ref": effect.dispatch_ref})
            record = {"step": effect.step, "tool": effect.tool, "target": effect.tool,
                      "arguments_hash": effect.arguments_hash, "result": result,
                      "receipt_ids": [receipt_id], "before_binding": current_binding,
                      "after_binding": current_binding, "changes": changes,
                      "continuation_effect_id": effect.effect_id,
                      "continuation_dispatch_ref": effect.dispatch_ref,
                      "continuation_id": effect.intention_id}
            record["step_record_id"] = digest(record)
            closed = self.journal.emit("action_outcome_closed", record=record)
        return self.settle_host_effect(effect_id=effect_id, receipt_ref=closed["event_id"],
                                       current_binding=current_binding)

    def repair_effect(self, *, intention_id: str, effect_id: str, tool: str,
                      current_binding: dict, reason: str) -> dict:
        """Close one safely retryable unknown effect before a typed retry.

        The control receipt and the state transition are one host-owned
        reconciliation.  No model-supplied outcome can use this path, and
        writes are deliberately excluded by the state reducer.
        """
        with self.journal.ownership():
            effect = self._state.effects.get(effect_id)
            if effect is None or effect.status not in {"pending", "unknown"}:
                raise ContinuationError("no unresolved effect to repair")
            control = self.journal.emit(
                "continuation_control", producer="stateframe_runtime",
                command="repair_effect", intention_id=intention_id,
                effect_id=effect_id, tool=tool, reason=reason,
                current_binding=current_binding)
            return self.apply(
                "repair_effect", intention_id=intention_id, effect_id=effect_id,
                tool=tool, receipt_ref=control["event_id"],
                current_binding=current_binding)

    def choose_next(self, current_binding: dict) -> str | None:
        """Keep productive active work; schedule only when it yields or suspends."""
        with self.journal.ownership():
            current_time = self.clock_us()
            if type(current_time) is not int or current_time < 0:
                raise ContinuationError("host clock must return nonnegative microseconds")
            now = max(current_time, self._state.last_time_us)
            for item in list(self._state.intentions.values()):
                if item.phase == "suspended" and item.wake and now >= item.wake.deadline_us:
                    self.apply("expire", intention_id=item.intention_id)
            active = self._state.active_id
            if active is not None:
                if unresolved(self._state, active):
                    return None
                if self._state.intentions[active].turns_left == 0:
                    self.apply("exhaust", intention_id=active)
                else:
                    self.apply("select", intention_id=active,
                               current_dependencies=binding_versions(current_binding),
                               selection_reason="continue_productive_active_work", eligible=[active])
                    return active
            registry = LaneRegistry()
            for item in list(self._state.intentions.values()):
                if item.phase != "eligible":
                    continue
                if item.turns_left == 0:
                    self.apply("exhaust", intention_id=item.intention_id)
                    continue
                lane = registry.register(
                    lane_id=item.intention_id, contract_id="continuation",
                    parent_request_id=item.parent_id or "", now=item.created_us, kind="general",
                    turns=item.turns_left,
                    priority_class=PRIORITY_P1_UNBLOCK if item.return_pending else PRIORITY_P2_WORK)
                lane.last_progress_at = item.last_progress_us
                registry.update_lane(item.intention_id, service_debt=item.quanta)
            scheduler = Scheduler(self.policy)
            ranked = scheduler.rank(registry, now=now)
            if not ranked:
                return None
            selected = ranked[0].lane_id
            self.apply("select", intention_id=selected,
                       current_dependencies=binding_versions(current_binding),
                       selection_reason="existing_juggler_priority_and_service_order",
                       eligible=[item.lane_id for item in ranked])
            return selected

    def unfinished(self) -> list[str]:
        with self.journal.ownership():
            return [item.intention_id for item in self._state.intentions.values()
                    if item.phase != "completed"]
