"""Live continuation authority at the StateFrame request/dispatch boundary."""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

from kernel_continuation_evidence import (
    READ_ONLY, binding_versions, has_execution_receipt_contract,
)
from kernel_continuation_executive import ContinuationExecutive
from kernel_continuation_menu_authority import AUTHORED_CHECK_TOOLS, EDIT_TOOLS, state_allow_set
from kernel_continuation_state import ContinuationError, children_open, unresolved
from kernel_measurement import digest
from kernel_verdict import (
    NO_HOST_CHECKS, SATISFIED, TrustedTestReceipt, Verdict, evaluate_required_tests,
    repair_action_for,
)

CONTROL_TOOLS = frozenset({"done", "continuation_detour", "work_update", "note_update", "work"})
EFFECT_TOOLS = READ_ONLY | {
    "job_stop", *EDIT_TOOLS, "run_tests",
} | AUTHORED_CHECK_TOOLS
# A witness is a host-owned evidence action, not a continuation effect.  It is
# nevertheless a valid way to clear a read-only unknown effect: the old read
# can be reconciled as not-applied before the witness rail executes.
WITNESS_TOOLS = frozenset({"probe", *AUTHORED_CHECK_TOOLS})
SAFE_REPAIR_TOOLS = frozenset({"read", "run_tests", "probe"})


def _guarded_host_outcome(result: object) -> str:
    """Map host close results to the shared external tri-state vocabulary."""
    if isinstance(result, dict):
        result = result.get("outcome", result.get("status", result.get("result")))
    value = str(result).strip().upper().replace("-", "_").replace(" ", "_")
    if value in {"OK", "APPLIED", "PASSED", "GREEN", "SUCCESS", "PRESENT",
                 "REUSED_VERIFICATION"}:
        return "PRESENT"
    if value in {"NO_OP", "REJECTED", "REFUSED", "FAILED", "ABSENT", "NO_TESTS",
                 "SELECTOR_ERROR", "HARNESS_ERROR"}:
        return "ABSENT"
    return "UNKNOWN"


def enabled() -> bool:
    value = os.environ.get("KERNEL_CONTINUATION_EXECUTIVE", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_CONTINUATION_EXECUTIVE must be 0 or 1")
    return value == "1"


def tool_surface(tools: list[dict]) -> list[dict]:
    if any(t["function"]["name"] == "continuation_detour" for t in tools):
        return tools
    return [*tools, {"type": "function", "function": {
        "name": "continuation_detour",
        "description": "Suspend this decision to acquire a specific source range. The kernel returns "
                       "to your original decision after a current, complete read of that range. "
                       "Use ordinary read for simple inspection; use a detour when return must be tracked.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "reason": {"type": "string"}, "pending_decision": {"type": "string"}},
            "required": ["path", "start_line", "end_line", "reason", "pending_decision"],
            "additionalProperties": False}}}]


class LiveContinuation:
    def __init__(self, runtime):
        self.runtime = runtime
        self.journal = runtime.journal
        self.context_scope = getattr(runtime, "context_scope", None)
        from kernel_context_scope import bind_runtime, enabled as context_scope_enabled
        if context_scope_enabled():
            self.context_scope = bind_runtime(runtime)
        self.engine = ContinuationExecutive(self.journal)
        self.guarded_effects = None
        from kernel_guarded_effects import (
            GuardedEffectSequence, enabled as guarded_effects_enabled,
        )
        if guarded_effects_enabled(runtime):
            self.guarded_effects = GuardedEffectSequence(
                self.journal,
                destination=getattr(runtime, "guarded_effect_destination", None),
                destination_provider=lambda _name: getattr(
                    runtime, "guarded_effect_destination", None),
                enabled_override=True,
            )
        # R-041 is an opt-in host seam.  The legacy continuation path does not
        # import, scan, or serialize the adapter when the switch is off.
        self._unknown_outcomes = None
        if self._unknown_outcomes_enabled():
            from kernel_unknown_outcomes import ExternalEffectAdapter
            self._unknown_outcomes = ExternalEffectAdapter(self.journal)
            destination = getattr(runtime, "external_effect_destination", None)
            if destination is not None:
                self._unknown_outcomes.bind(destination)
        # Planned artifacts are an overlay on this same executive.  They are
        # deliberately lazy so the legacy continuation path has no extra
        # filesystem scan, journal event, or scheduler branch when the switch
        # is off.
        if os.environ.get("STATEFRAME_PLANNED_ARTIFACTS", "off").strip().lower() == "on":
            root = getattr(getattr(runtime, "owner", None), "root", None)
            if root is not None:
                self.engine.enable_planned_artifacts(str(root))
                claim_grounding = getattr(runtime, "claim_grounding", None)
                if claim_grounding is not None:
                    claim_grounding.planned_ledger = self.engine.planned_artifacts
        self.dispatch: dict | None = None
        self.effect_id: str | None = None
        # The write call site may expose only a path.  Keep the exact admitted
        # arguments beside the dispatch fence so that path-only admission can
        # prove which edit alias is still in flight.
        self._authorized_arguments: dict | None = None
        self.certification_ref: str | None = None
        self.source_capture = None
        self.source_prepared_ref: str | None = None
        self.completed = False
        self.certified = False
        self._pending_completion_ref: str | None = None
        self._pending_reused_test: tuple[str, Verdict, TrustedTestReceipt] | None = None
        self._stranded_blocker: str | None = None
        self._stranded_revision: object | None = None
        self.terminal_blocked = False

    @staticmethod
    def _unknown_outcomes_enabled() -> bool:
        value = os.environ.get("KERNEL_UNKNOWN_OUTCOMES", "off")
        if value not in {"off", "on"}:
            raise ValueError("KERNEL_UNKNOWN_OUTCOMES must be off or on")
        return value == "on"

    def _unknown_outcome_adapter(self):
        if not self._unknown_outcomes_enabled():
            return None
        adapter = getattr(self, "_unknown_outcomes", None)
        if adapter is None:
            from kernel_unknown_outcomes import ExternalEffectAdapter
            adapter = ExternalEffectAdapter(self.journal)
            self._unknown_outcomes = adapter
        return adapter

    def bind_external_destination(self, destination) -> str | None:
        """Bind a host destination for durable external-effect reconciliation."""
        adapter = self._unknown_outcome_adapter()
        return None if adapter is None else adapter.bind(destination)

    def begin_external_effect(self, key, payload, *, destination=None,
                              destination_name=None) -> dict | None:
        """Record one host-owned external intent before the next real prepare."""
        adapter = self._unknown_outcome_adapter()
        if adapter is None:
            return None
        return adapter.begin(key, payload, destination=destination,
                             destination_name=destination_name)

    def reconcile_external_effects(self) -> set[str]:
        """Settle durable external evidence before scheduling another action.

        r45 merge (single owner of this seam = restart_reconcile's executive bridge):
        guarded_effects receipts settle first, then unknown_outcomes tri-state
        reconciliation for whatever is still pending.  Each adapter is inert
        (returns an empty set, imports nothing new) when its switch is off."""
        reconciled = self._reconcile_guarded_effects()
        reconciled.update(self._reconcile_unknown_outcomes())
        return reconciled

    def _reconcile_guarded_effects(self) -> set[str]:
        sequence = getattr(self, "guarded_effects", None)
        if sequence is None:
            return set()
        settle = getattr(self.engine, "settle_external_effect", None)
        if not callable(settle):
            return set()
        state = self.engine.state
        reconciled: set[str] = set()
        records = list(self.journal.records())
        for effect_id, effect in getattr(state, "effects", {}).items():
            if getattr(effect, "status", None) not in {"pending", "unknown"}:
                continue
            snapshot = sequence.recover(effect_id)
            outcome = snapshot.get("status") if isinstance(snapshot, dict) else None
            if outcome not in {"PRESENT", "ABSENT"}:
                continue
            external_ref = snapshot.get("receipt_ref")
            if not isinstance(external_ref, str):
                matching = [event for event in records
                            if event.get("kind") == "guarded_effect_receipt"
                            and event.get("effect_id") == effect_id]
                external_ref = matching[-1].get("event_id") if matching else None
            if not isinstance(external_ref, str):
                continue
            try:
                settle(effect_id=effect_id, outcome=outcome,
                       external_event_ref=external_ref,
                       current_binding=self.runtime.binding())
            except ContinuationError:
                continue
            reconciled.update(
                event["event_id"] for event in records
                if event.get("event_id") and (
                    event.get("effect_id") == effect_id
                    or event.get("continuation_effect_id") == effect_id
                )
            )
        return reconciled

    def _reconcile_unknown_outcomes(self) -> set[str]:
        adapter = self._unknown_outcome_adapter()
        if adapter is None:
            return set()
        snapshots = adapter.reconcile_pending()
        settle = getattr(self.engine, "settle_external_effect", None)
        if not callable(settle):
            return set()
        state = self.engine.state
        records = list(self.journal.records())
        reconciled: set[str] = set()
        effects = getattr(state, "effects", {})
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            outcome = snapshot.get("status")
            if outcome not in {"PRESENT", "ABSENT"}:
                continue
            key = snapshot.get("idempotency_key")
            if not isinstance(key, list):
                continue
            effect_id = next((part for part in reversed(key)
                              if isinstance(part, str) and part in effects), None)
            if effect_id is None or getattr(effects[effect_id], "status", None) not in {"pending", "unknown"}:
                continue
            external_ref = snapshot.get("outcome_event_ref") or snapshot.get("receipt_ref")
            if not isinstance(external_ref, str):
                continue
            try:
                settle(effect_id=effect_id, outcome=outcome,
                       external_event_ref=external_ref,
                       current_binding=self.runtime.binding())
            except ContinuationError:
                continue
            external_id = snapshot.get("external_effect_id")
            reconciled.update(
                event["event_id"] for event in records
                if event.get("event_id") and (
                    event.get("external_effect_id") == external_id
                    or event.get("event_id") == external_ref
                )
            )
        return reconciled

    def _record_stranded(self, blocker: str) -> None:
        """Keep a recoverable scheduler dead-end visible to the next request."""
        state = self.engine.state
        revision = getattr(state, "revision", None)
        if self._stranded_blocker == blocker and self._stranded_revision == revision:
            return
        identity = state.active_id
        item = state.intentions.get(identity) if identity is not None else None
        unresolved_effects = unresolved(state, identity) if identity is not None else []
        unresolved_details = self._unresolved_effect_details(identity) if identity is not None else []
        self._stranded_blocker = blocker
        self._stranded_revision = revision
        self.journal.emit(
            "continuation_stranded",
            continuation_id=identity,
            state_revision=revision,
            blocker=blocker,
            pending_decision=getattr(item, "pending_decision", None),
            turns_left=getattr(item, "turns_left", None),
            stale_dependencies=getattr(item, "stale_dependencies", []),
            unresolved_effects=unresolved_effects,
            unresolved_details=unresolved_details,
        )

    def _unresolved_effect_details(self, identity: str | None) -> list[dict[str, object]]:
        """Describe unresolved host effects and any schema-valid live repair.

        The continuation state owns the effect identity/status.  The completion
        reducer owns admission, but it needs this typed finding to distinguish a
        runnable verification repair from an unrecoverable effect.  Small test
        doubles may expose only a partial effect shape; those retain the legacy
        blocker rather than receiving invented repair data.
        """
        if not isinstance(identity, str):
            return []
        state = self.engine.state
        item = state.intentions.get(identity)
        if item is None:
            return []
        selectors = item.acceptance.get("selectors", []) if isinstance(item.acceptance, dict) else []
        selectors = tuple(str(selector) for selector in selectors if str(selector).strip())
        details: list[dict[str, object]] = []
        unresolved_ids = unresolved(state, identity)
        # Minimal continuation doubles used by the intervention probes expose
        # this predicate as a boolean.  They cannot provide a trustworthy
        # effect identity, so retain the legacy generic finding.
        if not isinstance(unresolved_ids, (list, tuple)):
            return details
        for effect_id in unresolved_ids:
            effect = state.effects.get(effect_id)
            actual_id = getattr(effect, "effect_id", None)
            tool = getattr(effect, "tool", None)
            status = getattr(effect, "status", None)
            if not isinstance(actual_id, str) or not isinstance(tool, str) or not isinstance(status, str):
                continue
            repair = None
            if tool == "run_tests" and selectors:
                repair = self._registered_test_repair(selectors)
            elif tool == "read":
                path = self._effect_path(effect)
                if path is not None:
                    repair = repair_action_for(
                        ("UNRESOLVED_EFFECT", "READ_EVIDENCE"),
                        menu=self._live_completion_menu(), path=path)
                if repair is None:
                    repair = self._completion_owner_repair_action()
            details.append({
                "effect_id": actual_id,
                "tool": tool,
                "status": status,
                "repair_action": repair,
            })
        return details

    def _registered_test_repair(self, selectors: tuple[str, ...]) -> dict | None:
        """Resolve a test retry before the menu intersects continuation permission.

        A narrowed completion menu cannot prove its own retry unavailable.
        Use the registered schema and capability gate; authorization still
        checks the exact effect and target before reconciling any outcome.
        """
        from kernel_completion_owner import validate_tool_arguments
        from kernel_tool_surface import dispatch_admissible

        surface = getattr(self.runtime, "_completion_owner_full_tools", ())
        if not isinstance(surface, (list, tuple)) or not dispatch_admissible(
                "run_tests", runtime=self.runtime):
            return None
        schemas = [schema for schema in surface if isinstance(schema, Mapping)
                   and isinstance(schema.get("function"), Mapping)
                   and schema["function"].get("name") == "run_tests"]
        arguments = {"target": " ".join(selectors)}
        if len(schemas) != 1:
            return None
        parameters = schemas[0]["function"].get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        target = properties.get("target") if isinstance(properties, Mapping) else None
        if (not isinstance(target, Mapping) or target.get("type") != "string"
                or validate_tool_arguments(schemas, "run_tests", arguments)):
            return None
        return {"tool": "run_tests", "arguments": arguments,
                "reason": "retry the exact unresolved public test selectors"}

    def _effect_path(self, effect: object) -> str | None:
        """Recover the exact read target from the host receipt journal.

        ``Effect`` intentionally stores only hashes, so a read that closes with
        an unknown cursor result cannot be retried from continuation state alone.
        The host action record is the authoritative, already-journaled argument
        source and is available in both live and offline replays.
        """
        effect_id = getattr(effect, "effect_id", None)
        receipt_ref = getattr(effect, "receipt_ref", None)
        records = getattr(self.journal, "records", None)
        if callable(records):
            records = records()
        if not isinstance(records, (list, tuple)):
            return None
        for event in reversed(records):
            if not isinstance(event, Mapping):
                continue
            record = event.get("record")
            if not isinstance(record, Mapping):
                record = event
            effect_matches = record.get("continuation_effect_id") == effect_id
            receipt_matches = event.get("event_id") == receipt_ref
            if not (effect_matches or receipt_matches):
                continue
            parameters = record.get("action_parameters")
            if not isinstance(parameters, Mapping):
                parameters = record.get("arguments")
            path = parameters.get("path") if isinstance(parameters, Mapping) else None
            if isinstance(path, str) and path.strip():
                return path
        return None

    def _completion_owner_repair_action(self) -> dict[str, object] | None:
        """Project one currently admitted owner action onto continuation repair.

        This is deliberately a fallback.  Exact read replay is preferred; the
        owner projection covers a REQWITNESS state where the only admissible
        progress is the typed probe already present on the live menu.
        """
        tools = getattr(self.runtime, "_completion_owner_full_tools", ())
        if not tools:
            return None
        try:
            from kernel_completion_owner import legal_next_actions
            actions = legal_next_actions(
                self.runtime,
                decision=getattr(self.runtime, "_last_completion_decision", None),
                tools=tools,
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        menu = set(self._live_completion_menu())
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            tool = str(action.get("tool", "")).strip()
            arguments = action.get("arguments")
            if tool not in SAFE_REPAIR_TOOLS or tool not in menu or not isinstance(arguments, Mapping):
                continue
            return {"tool": tool, "arguments": dict(arguments),
                    "reason": str(action.get("reason", "use the completion owner's live route"))}
        return None

    def _genuine_stall(self, item: object, revision: object) -> bool:
        """Allow a blocked stop only after exhaustion or a repeated no-progress frame."""
        turns_left = getattr(item, "turns_left", None)
        if type(turns_left) is int and turns_left <= 0:
            return True
        if bool(getattr(self.runtime, "_completion_owner_no_progress_edit_cap", False)):
            return True
        return self._stranded_blocker is not None and self._stranded_revision == revision

    def _live_completion_menu(self) -> tuple[str, ...]:
        menu = getattr(self.runtime, "completion_menu", ())
        if isinstance(menu, (list, tuple, set, frozenset)) and menu:
            return tuple(str(name) for name in menu)
        return ("done", "run_tests")

    def _repair_unresolved_effect(self, tool: str, arguments: dict,
                                  identity: str) -> bool:
        """Apply the one typed retry that the current live menu can execute."""
        if tool not in SAFE_REPAIR_TOOLS:
            return False
        details = self._unresolved_effect_details(identity)
        if len(details) != 1:
            return False
        action = details[0].get("repair_action")
        action_tool = action.get("tool") if isinstance(action, Mapping) else None
        if action_tool != tool:
            return False
        action_arguments = action.get("arguments") if isinstance(action, dict) else None
        effect_id = details[0].get("effect_id")
        effect_tool = details[0].get("tool")
        if not isinstance(effect_id, str) or not isinstance(effect_tool, str):
            return False
        # A read-only unknown can be safely superseded by another typed
        # read/test/witness route.  Unknown test execution itself may only be
        # retried by the same test tool and exact target.
        if effect_tool != "read" and effect_tool != tool:
            return False
        if not isinstance(action_arguments, Mapping):
            return False
        if tool == "run_tests":
            expected = action_arguments.get("target")
            if not isinstance(expected, str) or arguments.get("target") != expected:
                return False
        elif tool == "read":
            expected = action_arguments.get("path")
            if not isinstance(expected, str) or arguments.get("path") != expected:
                return False
        elif tool == "probe":
            expected_mode = action_arguments.get("mode")
            expected_clause = action_arguments.get("clause_ref", action_arguments.get("clause_id"))
            if expected_mode is not None and arguments.get("mode") != expected_mode:
                return False
            if expected_clause is not None:
                actual_clause = arguments.get("clause_ref", arguments.get("clause_id"))
                if actual_clause != expected_clause:
                    return False
        self.engine.repair_effect(
            intention_id=identity, effect_id=effect_id, tool=tool,
            current_binding=self.runtime.binding(),
            reason="typed current-menu verification retry")
        self._stranded_blocker = None
        self._stranded_revision = None
        return True

    def prepare(self, messages) -> None:
        # ``completed`` is reserved for an accepted native ``done``.  A green
        # public-check certification is evidence for that decision, not the
        # decision itself, so it must still expose a model request.
        if self.completed:
            return
        self.reconcile_external_effects()
        from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
        if candidate_reconcile_enabled():
            from kernel_candidate_restart_reconcile import maybe_promote_portfolio
            maybe_promote_portfolio(self.runtime)
        current = self.runtime.binding()
        if self.guarded_effects is not None:
            self.reconcile_external_effects()
        if not self.engine.state.intentions:
            owner = self.runtime.owner
            objective = getattr(owner, "task_row", {}).get("msg")
            if not isinstance(objective, str) or not objective.strip():
                objective = str(messages[1].get("content", "")) if len(messages) > 1 else ""
            required = [s for s in getattr(owner, "f2p_ids", ())
                        if (Path(owner.root) / s.split("::")[0]).is_file()]
            self.engine.apply("create", new_id="root", objective=objective,
                pending_decision="Complete the original objective and verify its public requirements.",
                dependencies=binding_versions(current), turns=self.runtime.step_limit,
                acceptance={
                    "kind": "public_checks" if required else "no_public_checks",
                    "selectors": required,
                    "host_check_state": "REQUIRED_PUBLIC_CHECKS" if required else NO_HOST_CHECKS,
                })
        # Retention runs at the request boundary, before selection can consume
        # a return-pending child.  This keeps source needed for the current
        # parent frame hot while allowing the following boundary to reclaim it.
        self.engine.retention_checkpoint()
        selected = self.engine.choose_next(current)
        if selected is None:
            # A settled, currently verified root has reached the hand-off point:
            # the model must decide whether more work is needed or whether to
            # submit the native ``done`` control.  Keep the root live so the
            # next frame is a real provider request; do not manufacture a
            # continuation terminal here.
            blocker = self.completion_blocker()
            if blocker is None:
                self._stranded_blocker = None
                self._stranded_revision = None
                self.certified = True
                return
            self.certified = False
            details = self._unresolved_effect_details(self.engine.state.active_id)
            self.terminal_blocked = bool(details and
                                         not any(item.get("repair_action") for item in details)
                                         and self._genuine_stall(
                                             self.engine.state.intentions[self.engine.state.active_id],
                                             getattr(self.engine.state, "revision", None)))
            if details and any(item.get("repair_action") for item in details):
                # An unresolved effect with a typed live route is waiting for
                # the next provider action, not stranded work.
                self._stranded_blocker = None
                self._stranded_revision = None
            else:
                self._record_stranded(blocker)
            return
        self._stranded_blocker = None
        self._stranded_revision = None

    def source_prepared(self, event: dict, capture, metadata: dict | None) -> None:
        # A queued frame was never shown to the model and cannot refresh evidence.
        self.source_capture = capture if metadata is not None else None
        self.source_prepared_ref = event["event_id"] if metadata is not None else None

    def refresh_delivered_source(self, current: dict) -> None:
        capture = self.source_capture
        if (capture is None or self.source_prepared_ref is None
                or capture.request_id != self.runtime.current_request_id
                or capture.selected_response_event_id is None):
            return
        try:
            self.engine.apply("revalidate", intention_id=self.engine.state.active_id,
                receipt_ref=capture.selected_response_event_id,
                source_prepared_ref=self.source_prepared_ref,
                request_id=capture.request_id, snapshot_id=capture.snapshot_id, step=capture.step,
                current_binding=current, current_dependencies=binding_versions(current))
        except ContinuationError as error:
            from kernel_boundary_recovery import safe_error_class, safe_error_detail
            self.journal.emit("continuation_source_refresh_refused",
                request_id=capture.request_id, source_prepared_ref=self.source_prepared_ref,
                receipt_ref=capture.selected_response_event_id, dispatch=False,
                error_class=safe_error_class(error), error_detail=safe_error_detail(error))
            return  # The unchanged effect fence still rejects stale authorization.

    def reset_action(self) -> None:
        self.dispatch = None
        self.effect_id = None
        self._authorized_arguments = None
        self.certification_ref = None
        self._pending_completion_ref = None
        self._pending_reused_test = None
        self.certified = False

    def per_step_allow_set(self) -> frozenset[str]:
        """Return pure name-level permissions for the current intention.

        Menu construction must not call :meth:`authorize`: authorization emits
        a dispatch binding and may reconcile a typed retry.  This projection
        carries the durable intention gates into menu authority; argument,
        path, and host-binding checks still run at dispatch.
        """
        getter = getattr(self.engine, "per_step_allow_set", None)
        if callable(getter):
            allowed = set(getter())
        else:  # Small continuation doubles used by offline intervention probes.
            allowed = set(state_allow_set(self.engine.state))
        menu_state = getattr(self.engine, "active_menu_state", None)
        if callable(menu_state):
            identity, active, unsettled, stale = menu_state()
        else:  # Compatibility with snapshot-only continuation doubles.
            state = self.engine.state
            identity = getattr(state, "active_id", None)
            item = getattr(state, "intentions", {}).get(identity) if identity is not None else None
            active = item is not None and getattr(item, "phase", "active") == "active"
            unsettled = bool(unresolved(state, identity)) if active else False
            stale = bool(getattr(item, "stale_dependencies", ()))
        if not active:
            from kernel_tool_surface import dispatch_admissible
            return frozenset(
                name for name in {"done"} & allowed
                if dispatch_admissible(name, runtime=self.runtime)
            )
        if unsettled:
            # ``run_tests`` is menu-dispatchable only when the exact typed
            # repair is available to this live menu. ``done`` remains so the
            # shared completion reducer can issue its typed finding.
            details = self._unresolved_effect_details(identity)
            allowed = {"done"}
            for item in details:
                action = item.get("repair_action")
                if isinstance(action, Mapping) and action.get("tool") in SAFE_REPAIR_TOOLS:
                    allowed.add(str(action["tool"]))
            from kernel_tool_surface import dispatch_admissible
            return frozenset(
                name for name in allowed
                if dispatch_admissible(name, runtime=self.runtime)
            )
        if stale:
            from kernel_stale_edit_policy import allows_stale_repair
            stale_edits = {name for name in EDIT_TOOLS if not allows_stale_repair(name)}
            batch = getattr(self.runtime, "native_batch", None)
            prefix_ok = bool(
                getattr(self.runtime, "native_prefix_authority", False)
                and batch is not None and (
                    (getattr(batch, "index", 0) > 1 and getattr(batch, "selected_ref", None))
                    or (getattr(batch, "index", 0) > 0 and getattr(batch, "pending", None))))
            capture = self.source_capture
            refresh_ok = bool(
                getattr(self.runtime, "continuation_source_refresh", False)
                and capture is not None
                and getattr(capture, "request_id", None) == getattr(
                    self.runtime, "current_request_id", None)
                and getattr(capture, "selected_response_event_id", None) is not None)
            if prefix_ok:
                stale_edits.difference_update(EDIT_TOOLS)
            if refresh_ok:
                stale_edits.clear()
            allowed.difference_update(stale_edits)
        # Host obligations at UNKNOWN/MISSING_CHECKS can only be certified by a
        # solver-authored check; withholding it leaves only uncertified done.
        from kernel_check_authoring_route import CHECK_AUTHORING_TOOLS, authoring_open
        if authoring_open(self.runtime):
            allowed.update(CHECK_AUTHORING_TOOLS)
        from kernel_tool_surface import dispatch_admissible
        return frozenset(
            name for name in allowed
            if dispatch_admissible(name, runtime=self.runtime)
        )

    def authorized_write_action(self, path: object, *, action_kind: str = "edit") -> str | None:
        """Return the exact admitted edit alias for this write boundary.

        ``authorize`` intentionally moves an edit into ``pending`` before the
        product editor runs.  That changes the *next* continuation allow-set to
        ``done``; it must not veto the write that just created that pending
        effect.  This narrow exception is valid only when the current path,
        request fence, binding, step, intention, and pending effect all match
        the admitted dispatch.  An unrelated pending read/edit cannot use it.
        """
        dispatch = getattr(self, "dispatch", None)
        arguments = getattr(self, "_authorized_arguments", None)
        effect_id = getattr(self, "effect_id", None)
        if (not isinstance(dispatch, dict) or not isinstance(arguments, dict)
                or not isinstance(effect_id, str)):
            return None
        action = str(dispatch.get("tool", "")).strip().lower()
        requested = str(action_kind or "").strip().lower()
        if action not in EDIT_TOOLS or requested not in {"edit", action}:
            return None
        expected_path = arguments.get("path")
        if not isinstance(expected_path, str):
            return None
        runtime = self.runtime
        owner = getattr(runtime, "owner", None)
        root = getattr(owner, "root", None)
        if root is None:
            return None
        try:
            root_path = Path(root).resolve()
            actual_path = Path(path)
            if not actual_path.is_absolute():
                actual_path = root_path / actual_path
            if actual_path.resolve() != (root_path / expected_path).resolve():
                return None
            current = runtime.binding()
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if (dispatch.get("accepted") is not True
                or dispatch.get("continuation_effect_id") != effect_id
                or dispatch.get("expected") != dispatch.get("current")
                or current != dispatch.get("expected")
                or dispatch.get("request_id") != getattr(runtime, "current_request_id", None)
                or dispatch.get("step") != getattr(owner, "current_step", None)):
            return None
        state = self.engine.state
        effect = getattr(state, "effects", {}).get(effect_id)
        if (effect is None or effect.status != "pending"
                or effect.tool != action
                or effect.arguments_hash != dispatch.get("arguments_hash")
                or effect.request_id != dispatch.get("request_id")
                or effect.step != dispatch.get("step")
                or effect.intention_id != dispatch.get("continuation_id")
                or effect.dispatch_ref != dispatch.get("event_id")
                or getattr(state, "active_id", None) != dispatch.get("continuation_id")):
            return None
        return action
    def host_done_dispatch(self) -> None:
        """Stage the same live ``done`` fence used by a model tool call.

        This is only used by the opt-in response-budget boundary.  Keeping the
        fence in this class means a host submission still gets the normal
        binding, continuation id, request id, and dispatch journal record; the
        completion reducer remains responsible for deciding whether it seals.
        """
        if self.completed:
            return
        if self.dispatch is not None and self.dispatch.get("tool") == "done":
            return
        self.reset_action()
        feedback = self.authorize("done", {})
        if feedback is not None:
            raise ContinuationError(str(feedback))
        self.journal.emit(
            "continuation_host_done_dispatch",
            producer="stateframe_runtime",
            dispatch_ref=(self.dispatch or {}).get("event_id"),
            continuation_id=(self.dispatch or {}).get("continuation_id"),
            request_id=(self.dispatch or {}).get("request_id"),
            step=(self.dispatch or {}).get("step"),
            dispatch=False,
        )

    def lineage(self) -> dict:
        if self.dispatch is None:
            return {}
        return {"continuation_id": self.dispatch["continuation_id"],
                "continuation_effect_id": self.effect_id,
                "continuation_dispatch_ref": self.dispatch["event_id"]}

    def authorize(self, tool: str, arguments: dict) -> str | None:
        runtime = self.runtime
        state = self.engine.state
        identity = state.active_id
        # ``done`` is assessed atomically with the other completion gates.  Do
        # not reject it here on a partial continuation predicate: that used to
        # create a second veto before the shared reducer saw the immutable
        # basis. Ordinary effects still require an active, settled intention.
        if tool != "done" and identity is None:
            return self.completion_blocker()
        if tool != "done" and identity is not None and unresolved(state, identity):
            if not self._repair_unresolved_effect(tool, arguments, identity):
                return self.completion_blocker()
            state = self.engine.state
        from kernel_check_authoring_route import CHECK_AUTHORING_TOOLS, authoring_open
        authoring = tool in CHECK_AUTHORING_TOOLS and authoring_open(runtime)
        if not has_execution_receipt_contract(tool):
            return "[continuation] Tool has no implemented execution receipt contract."
        from kernel_agent_tools import is_registered
        if tool not in CONTROL_TOOLS | EFFECT_TOOLS | WITNESS_TOOLS and not authoring and not is_registered(tool):
            return "[continuation] Tool has no continuation dispatch contract."
        from kernel_tool_surface import dispatch_admissible
        if not dispatch_admissible(tool, runtime=runtime):
            return "[continuation] Tool is not dispatchable in this frame."
        if tool in EDIT_TOOLS:
            guard = getattr(runtime, "completion_mutation_guard", None)
            if callable(guard):
                feedback = guard(tool)
                if feedback is not None:
                    return feedback
        path_arg = arguments.get("path")
        if path_arg is not None:
            root = Path(runtime.owner.root).resolve()
            if not isinstance(path_arg, str) or not (root / path_arg).resolve().is_relative_to(root):
                return "[continuation] Path outside the bound workspace denied."
        current = runtime.binding()
        if self.context_scope is not None:
            feedback = self.context_scope.authorize_dispatch(tool, arguments)
            if feedback is not None:
                return feedback
        if tool != "done" and current != runtime.prepared_binding:
            return "[continuation] Workspace changed after this frame; request a fresh decision."
        # A completion attempt without a live root is still allowed to reach
        # the shared completion reducer.  The reducer will receive the typed
        # CONTINUATION_NO_ROOT finding; this dispatch boundary must not turn it
        # into a second string veto or a KeyError.
        if tool == "done" and identity is None:
            return None
        item = state.intentions[identity]
        prefix = {}
        batch = getattr(runtime, "native_batch", None)
        if (getattr(runtime, "native_prefix_authority", False) and tool in EDIT_TOOLS
                and batch is not None and batch.index > 1 and batch.selected_ref):
            prefix["native_selected_ref"] = batch.selected_ref
        if (not prefix and self.runtime.continuation_source_refresh and tool in EDIT_TOOLS
                and (item.stale_dependencies or item.dependencies != binding_versions(current))):
            self.refresh_delivered_source(current)
        effect = None if (tool in CONTROL_TOOLS | WITNESS_TOOLS or authoring) else uuid.uuid4().hex
        args_hash = digest({k: v for k, v in arguments.items() if k not in {"why", "reason"}})
        fence = self.journal.emit("step_dispatch_binding", producer="stateframe_runtime",
            snapshot_id=runtime.snapshot_id, accepted=True, expected=current, current=current,
            continuation_id=identity, continuation_effect_id=effect,
            request_id=runtime.current_request_id, step=runtime.owner.current_step,
            tool=tool, arguments_hash=args_hash)
        if effect is not None:
            if self.guarded_effects is not None:
                from kernel_guarded_effects import GuardedEffectError
                try:
                    self.guarded_effects.begin(
                        effect,
                        {"tool": tool, "arguments_hash": args_hash,
                         "request_id": runtime.current_request_id,
                         "snapshot_id": runtime.snapshot_id},
                        origin_ref=fence["event_id"],
                    )
                except GuardedEffectError as error:
                    return "[guarded-effects] Intent refused: " + str(error)
            try:
                self.engine.apply("begin_effect", intention_id=identity, effect_id=effect,
                    action_hash=digest({"tool": tool, "arguments_hash": args_hash}),
                    arguments_hash=args_hash, tool=tool, read_only=tool in READ_ONLY,
                    step=runtime.owner.current_step, request_id=runtime.current_request_id,
                    snapshot_id=runtime.snapshot_id, dispatch_ref=fence["event_id"],
                    current_binding=current, current_dependencies=binding_versions(current), **prefix)
            except ContinuationError as error:
                return "[continuation] Dispatch refused: " + str(error)
            from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
            if tool == "str_replace" and candidate_reconcile_enabled():
                from kernel_candidate_restart_reconcile import persist_intent
                persist_intent(self.runtime, effect_id=effect, arguments=arguments,
                               current=current)
            if self.guarded_effects is not None:
                from kernel_guarded_effects import GuardedEffectError
                try:
                    self.guarded_effects.dispatch(effect, dispatch_ref=fence["event_id"])
                except GuardedEffectError as error:
                    return "[guarded-effects] Dispatch refused: " + str(error)
        self.dispatch, self.effect_id = fence, effect
        self._authorized_arguments = dict(arguments)
        runtime.step_ledger.pending.update(self.lineage())
        return None

    def settle_authored_check(self, receipt: Mapping[str, object]) -> bool:
        """Settle the exact host-authored-check payload admitted for this step."""
        dispatch = self.dispatch
        if (not isinstance(dispatch, Mapping)
                or dispatch.get("tool") != "authored_check"
                or not isinstance(receipt, Mapping)):
            return False
        arguments = self._authorized_arguments or {}
        expected_hash = digest({key: value for key, value in arguments.items()
                                if key not in {"why", "reason"}})
        if expected_hash != dispatch.get("arguments_hash"):
            return False
        payload = receipt.get("receipt") if isinstance(receipt.get("receipt"), Mapping) else receipt
        expected_check = arguments.get("check_id")
        expected_clause = arguments.get("clause_id", arguments.get("requirement_id"))
        if (expected_check and payload.get("check_id") != expected_check) or (
                expected_clause and expected_clause not in {
                    payload.get("clause_id"), payload.get("requirement_id")}):
            return False
        try:
            from kernel_authored_check_lifecycle import attach_receipt
            attached = attach_receipt(self.runtime, payload)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            attached = False
        if not attached:
            return False
        setattr(self.runtime, "_last_authored_check_receipt", dict(payload))
        try:
            from kernel_completion_owner import recompute_authored_completion
            recompute_authored_completion(self.runtime)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            pass
        journal = getattr(self.runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit("continuation_authored_check_settled",
                 producer="stateframe_runtime", receipt_id=payload.get("receipt_id"),
                 result=payload.get("result"), certifies_clause=payload.get("certifies_clause") is True,
                 dispatch=False)
        return True

    def stage_reused_test_receipt(self, verdict: Verdict) -> bool:
        """Stage the evaluator's existing green receipt for a reuse effect.

        A progress-guard message is not evidence.  The guard may decline to
        execute only when the shared typed verdict supplies a current receipt
        which the continuation can later validate as completion evidence.
        """
        if (self.effect_id is None or self.dispatch is None
                or self.dispatch.get("tool") != "run_tests"
                or not isinstance(verdict, Verdict)):
            return False
        public_verdict = getattr(self.runtime, "public_test_verdict", None)
        candidate = public_verdict() if callable(public_verdict) else verdict
        if (not isinstance(candidate, Verdict) or candidate.state != SATISFIED
                or candidate.current_binding != self.runtime.binding()
                or not candidate.trusted_receipts):
            return False
        identity = self.engine.state.active_id
        if identity is None:
            return False
        try:
            evidence = self.engine._evidence
            state = self.engine.state
        except AttributeError:
            return False
        valid_receipts: list[TrustedTestReceipt] = []
        for receipt in candidate.trusted_receipts:
            if not isinstance(receipt, TrustedTestReceipt):
                return False
            try:
                event = evidence.event(receipt.receipt_id, "test_subprocess_completed")
            except ContinuationError:
                return False
            effect = state.effects.get(event.get("continuation_effect_id"))
            if (event.get("producer") != "test_subprocess"
                    or event.get("binding_stable") is not True
                    or event.get("binding") != candidate.current_binding
                    or event.get("after_binding") != candidate.current_binding
                    or type(event.get("exit_code")) is not int
                    or event.get("exit_code") != 0
                    or receipt.binding != candidate.current_binding
                    or receipt.after_binding != candidate.current_binding
                    or effect is None or effect.status != "applied"
                    or effect.intention_id != event.get("continuation_id")
                    or effect.dispatch_ref != event.get("continuation_dispatch_ref")):
                return False
            ancestor = event.get("continuation_id")
            while ancestor in state.intentions and ancestor != identity:
                ancestor = state.intentions[ancestor].parent_id
            if ancestor != identity:
                return False
            valid_receipts.append(receipt)
        if not valid_receipts:
            return False
        self._pending_reused_test = (self.effect_id, candidate, valid_receipts[0])
        return True

    def discard_queued(self, reason: str) -> None:
        batch = getattr(self.runtime, "native_batch", None)
        if batch is not None:
            batch.discard_pending(reason)

    def operation(self, tool: str, arguments: dict) -> str | None:
        if tool != "continuation_detour":
            return None
        if self.dispatch is None:
            raise ContinuationError("detour bypassed live dispatch authorization")
        state = self.engine.state
        parent = state.active_id
        try:
            start, end = arguments.get("start_line"), arguments.get("end_line")
            if type(start) is not int or type(end) is not int or start < 1 or end < start or end - start >= 120:
                raise ContinuationError("source range must contain 1 to 120 lines")
            raw = arguments.get("path")
            root = Path(self.runtime.owner.root).resolve()
            path = (root / str(raw)).resolve()
            if not isinstance(raw, str) or not path.is_relative_to(root) or not path.is_file():
                raise ContinuationError("detour source must be a current workspace file")
            total_lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
            if end > total_lines:
                raise ContinuationError(f"source has {total_lines} lines; choose an available range")
            current = self.runtime.binding()
            relative = path.relative_to(root).as_posix()
            if relative not in current.get("product_files", {}) and relative not in current.get("harness_files", {}):
                raise ContinuationError("detour source lacks a tracked file revision")
            child = uuid.uuid4().hex
            parent_item = state.intentions.get(parent) if parent is not None else None
            if parent_item is not None:
                acceptance = getattr(parent_item, "acceptance", {})
                self.engine.capture_resumption_cue(
                    parent, interrupted_by=child,
                    focus=getattr(parent_item, "pending_decision", "")
                    or getattr(parent_item, "objective", ""),
                    last_evidence=tuple(getattr(parent_item, "dependencies", ())),
                    return_target=parent,
                    trigger="continuation detour completes",
                    owed_items=(getattr(parent_item, "pending_decision", ""),),
                    waiting_checks=tuple(acceptance.get("selectors", ()))
                    if isinstance(acceptance, dict) else (),
                    priority=1 if getattr(parent_item, "return_pending", False) else 0,
                    affinity=1, now=self.engine.clock_us(),
                    proposed_next_action="reconcile the returned source against the current binding",
                    snapshot_refs=(relative, f"{start}-{end}"),
                )
            self.engine.apply("detour", intention_id=parent, reason=arguments.get("reason"),
                wake_token=uuid.uuid4().hex, deadline_us=self.engine.clock_us() + 300_000_000,
                child={"new_id": child, "objective": "Acquire source: " + relative,
                    "pending_decision": arguments.get("pending_decision"),
                    "dependencies": binding_versions(current),
                    "turns": (12 if self.runtime.step_limit is None else
                              max(1, min(12, self.runtime.step_limit - self.runtime.owner.current_step - 1))),
                    "acceptance": {"kind": "source_read", "path": relative,
                                   "start_line": start, "end_line": end}})
        except (ContinuationError, OSError) as error:
            feedback = "[continuation] Detour rejected: " + str(error)
            self.runtime.owner.receipt("state", tool, tool, "REJECTED", feedback)
            return feedback
        self.discard_queued("continuation detour requires a new decision for the child")
        feedback = "[continuation] Original intention suspended; acquire the declared range. Return is automatic."
        self.runtime.owner.receipt("state", tool, tool, "APPLIED", feedback)
        return feedback

    def closed(self, event: dict) -> None:
        record = event["record"]
        effect = record.get("continuation_effect_id")
        if effect is None:
            return
        if self.guarded_effects is not None:
            self.guarded_effects.record_receipt(
                effect,
                 {"tool": record.get("tool"), "result": record.get("result"),
                  "action_ref": event["event_id"]},
                 outcome=_guarded_host_outcome(record.get("result")),
                 receipt_ref=event["event_id"],
            )
        self.engine.settle_host_effect(effect_id=effect, receipt_ref=event["event_id"],
                                      current_binding=self.runtime.binding())
        state = self.engine.state
        if state.effects[effect].status != "applied":
            if (record.get("tool") == "run_tests"
                    and record.get("result") == "REUSED_VERIFICATION"):
                self._revalidate_reused_test(effect, record)
            return
        identity = record["continuation_id"]
        current = self.runtime.binding()
        if record["tool"] == "run_tests" or (record["tool"] == "read" and record["result"] == "OK"):
            try:
                self.engine.apply("revalidate", intention_id=identity, receipt_ref=event["event_id"],
                    current_binding=current, current_dependencies=binding_versions(current))
            except ContinuationError:
                # An empty/truncated/historical read is not a current evidence refresh.
                return
            if record["tool"] == "read" and state.intentions[identity].acceptance.get("kind") == "source_read":
                try:
                    self.engine.apply("complete", intention_id=identity, receipt_ref=event["event_id"],
                        current_binding=current, current_dependencies=binding_versions(current))
                except ContinuationError:
                    return  # Current read is useful, but the declared range remains unfinished.
                self.engine.mark_continuation_return(
                    identity, now=self.engine.clock_us(), evidence=(event["event_id"],),
                    receipt=event["event_id"],
                )
                self.discard_queued("source child completed; return to original pending decision")

    def revalidate_from_green_receipt(
            self, receipt_ref: str, effect_id: str | None, *,
            selectors: tuple[str, ...] | None = None,
            intention_id: str | None = None) -> bool:
        """Refresh the active intention only from a settled current green receipt."""
        if getattr(self.runtime, "stateframe_green_revalidate_mode", "off") != "on":
            return False
        from kernel_green_revalidate import effect_is_settled, trusted_current_receipt
        if not effect_is_settled(self.runtime, effect_id):
            self.journal.emit("continuation_green_revalidation_rejected",
                              producer="kernel_green_revalidate", receipt_ref=receipt_ref,
                              effect_id=effect_id, reason="effect_not_settled", dispatch=False)
            return False
        selected = trusted_current_receipt(self.runtime, receipt_ref)
        identity = self.engine.state.active_id
        if selected is None or identity is None:
            self.journal.emit("continuation_green_revalidation_rejected",
                              producer="kernel_green_revalidate", receipt_ref=receipt_ref,
                              effect_id=effect_id, reason="receipt_not_current_trusted_green",
                              dispatch=False)
            return False
        verdict, receipt = selected
        if selectors is not None or intention_id is not None:
            test_receipts = getattr(self.runtime, "test_receipts", ())
            evidence = next((item for item in reversed(test_receipts)
                             if isinstance(item, dict)
                             and str(item.get("receipt_id", item.get("event_id", "")))
                             == receipt.receipt_id), None)
            same_selectors = (selectors is None
                              or tuple(str(item) for item in evidence.get("selectors", ()))
                              == tuple(selectors)) if isinstance(evidence, dict) else False
            same_effect = (isinstance(evidence, dict)
                           and evidence.get("continuation_effect_id") == effect_id)
            same_intention = (intention_id is None
                              or (isinstance(evidence, dict)
                                  and evidence.get("continuation_id") == intention_id))
            if not (same_selectors and same_effect and same_intention):
                self.journal.emit(
                    "continuation_green_revalidation_rejected",
                    producer="kernel_green_revalidate", receipt_ref=receipt.receipt_id,
                    effect_id=effect_id, reason="receipt_pair_mismatch",
                    selectors=list(selectors or ()), intention_id=intention_id,
                    dispatch=False)
                return False
        current = self.runtime.binding()
        try:
            self.engine.apply("revalidate", intention_id=identity,
                              receipt_ref=receipt.receipt_id, current_binding=current,
                              current_dependencies=binding_versions(current))
        except ContinuationError as error:
            self.journal.emit("continuation_green_revalidation_rejected",
                              producer="kernel_green_revalidate", receipt_ref=receipt.receipt_id,
                              effect_id=effect_id, reason="validation_rejected",
                              error_type=type(error).__name__, detail=str(error)[:180],
                              dispatch=False)
            return False
        self.journal.emit("continuation_green_revalidated",
                          producer="kernel_green_revalidate", event_schema="stateframe-green-revalidate/1",
                          intention_id=identity, effect_id=effect_id,
                          receipt_ref=receipt.receipt_id, current_binding=current,
                          trusted_receipts=[item.receipt_id for item in verdict.trusted_receipts],
                          refresh="freshness_and_dependencies", dispatch=False)
        return True

    def _revalidate_reused_test(self, effect_id: str, record: dict) -> None:
        staged = self._pending_reused_test
        self._pending_reused_test = None
        if staged is None or staged[0] != effect_id:
            return
        _staged_effect, verdict, receipt = staged
        current = self.runtime.binding()
        identity = record.get("continuation_id")
        if (not isinstance(identity, str) or verdict.current_binding != current
                or receipt.binding != current or receipt.after_binding != current):
            return
        try:
            self.engine.apply("revalidate", intention_id=identity,
                receipt_ref=receipt.receipt_id, current_binding=current,
                current_dependencies=binding_versions(current))
        except ContinuationError:
            # The typed receipt is only useful if the durable continuation
            # evidence join accepts the exact same binding and lineage.
            return

    def _keyed_completion_durable(self) -> bool:
        """STATEFRAME_SUBMIT_IDEMPOTENT only: the root is not active because this
        run's keyed ``complete`` for its durable submission intent already landed
        (a crash hit between that effect and the terminal record).  The retried
        done must reach the reducer so it can reconcile, not be refused forever."""
        if getattr(self.runtime, "stateframe_submit_idempotent_mode", "off") != "on":
            return False
        from kernel_challenge_budget import slot_key, submission_slot
        slot = submission_slot(self.runtime)
        return slot is not None and self.submission_result(slot_key(slot)) is not None

    def completion_blocker(self) -> str | None:
        catalog_gate = getattr(self.runtime, "catalog_completion_blocker", None)
        if callable(catalog_gate):
            gate = catalog_gate()
            if isinstance(gate, Mapping):
                return str(gate.get("detail") or "[continuation] Host catalog unknowns remain open.")
        state = self.engine.state
        identity = state.active_id
        if identity is None:
            if self._keyed_completion_durable():
                return None
            return "[continuation] No active root can be completed."
        item = state.intentions[identity]
        if item.parent_id is not None:
            return "[continuation] Finish the declared source range; the kernel will return to the parent."
        if unresolved(state, identity) or children_open(state, identity):
            details = self._unresolved_effect_details(identity)
            if details:
                first = details[0]
                action = first.get("repair_action")
                if action is not None:
                    return (f"[continuation] Unresolved effect {first['effect_id']} "
                            f"(tool={first['tool']}, status={first['status']}) has no settled host "
                            "outcome. Use the typed repair action on the live menu: "
                            + json.dumps(action, sort_keys=True, ensure_ascii=False))
                return (f"[continuation] Unresolved effect {first['effect_id']} "
                        f"(tool={first['tool']}, status={first['status']}) has no typed repair "
                        "action on the live menu; completion is BLOCKED.")
            return "[continuation] Required work or an action outcome remains unresolved."
        if item.stale_dependencies or item.dependencies != binding_versions(self.runtime.binding()):
            return "[continuation] Evidence is stale. Read current source or run current checks before claiming completion."
        acceptance = getattr(item, "acceptance", {})
        required = (acceptance.get("selectors", ())
                    if isinstance(acceptance, dict) else ())
        if not required:
            planned_blocker = (getattr(self.engine, "planned_completion_blocker", None)
                               or (lambda: None))()
            return planned_blocker
        public_verdict = getattr(self.runtime, "public_test_verdict", None)
        if callable(public_verdict):
            verdict = public_verdict()
            if verdict.state != SATISFIED:
                if verdict.state == "VIOLATED":
                    return "[continuation] The newest required public test result is red; repair it before claiming completion."
                if verdict.state == "CONFLICT":
                    return "[continuation] Required public test evidence conflicts; rerun the checks before claiming completion."
                return "[continuation] Run every required public check against the current workspace."
        elif hasattr(self.runtime, "test_receipts"):
            # Production StateFrame runtimes expose ``public_test_verdict``; this
            # direct branch keeps small runtime doubles on the same authority
            # whenever they provide the receipt stream but not the convenience
            # method.  The boolean hook below is retained only for older doubles
            # that predate receipt-backed verification entirely.
            current = self.runtime.binding()
            acceptance = getattr(item, "acceptance", {})
            required = (acceptance.get("selectors", ())
                        if isinstance(acceptance, dict) else ())
            verdict = evaluate_required_tests(
                current, required, getattr(self.runtime, "test_receipts", ()))
            if verdict.state != SATISFIED:
                if verdict.state == "VIOLATED":
                    return "[continuation] The newest required public test result is red; repair it before claiming completion."
                if verdict.state == "CONFLICT":
                    return "[continuation] Required public test evidence conflicts; rerun the checks before claiming completion."
                return "[continuation] Run every required public check against the current workspace."
        elif not self.runtime.bound_public_verification():
            # Minimal offline continuation doubles predate the shared runtime
            # method; production StateFrameRuntime always takes the branch above.
            return "[continuation] Run every required public check against the current workspace."
        planned_blocker = (getattr(self.engine, "planned_completion_blocker", None) or (lambda: None))()
        if planned_blocker is not None:
            return planned_blocker
        return None

    def completion_finding(self) -> dict[str, object] | None:
        """Expose the continuation predicate as a reducer finding.

        The legacy string method remains for callers and transcript stability;
        new completion admission receives a typed code and exact selectors.
        """
        catalog_gate = getattr(self.runtime, "catalog_completion_blocker", None)
        if callable(catalog_gate):
            gate = catalog_gate()
            if isinstance(gate, Mapping):
                return {
                    "code": str(gate.get("reason_code") or "CATALOG_UNKNOWN"),
                    "detail": str(gate.get("detail") or "[continuation] Host catalog unknowns remain open."),
                    "unknown_ids": list(gate.get("unknown_ids", ()) or ()),
                }
        state = self.engine.state
        identity = state.active_id
        if identity is None:
            if self._keyed_completion_durable():
                return None
            return {"code": "CONTINUATION_NO_ROOT",
                    "detail": "[continuation] No active root can be completed."}
        item = state.intentions[identity]
        if item.parent_id is not None:
            return {"code": "CONTINUATION_CHILD_OPEN",
                    "detail": "[continuation] Finish the declared source range; the kernel will return to the parent."}
        if unresolved(state, identity):
            details = self._unresolved_effect_details(identity)
            selectors = item.acceptance.get("selectors", []) if isinstance(item.acceptance, dict) else []
            repairable = bool(details and details[0].get("repair_action"))
            return {"code": "UNRESOLVED_EFFECT" if repairable else "CONTINUATION_BLOCKED",
                    "detail": self.completion_blocker() or
                    "[continuation] Required work or an action outcome remains unresolved.",
                    "selectors": list(selectors) if repairable else [],
                    "unresolved_effects": details}
        if children_open(state, identity):
            return {"code": "OPEN_CONTINUATION_CHILD",
                    "detail": "[continuation] Required work or an action outcome remains unresolved."}
        acceptance = getattr(item, "acceptance", {})
        selectors = (acceptance.get("selectors", [])
                     if isinstance(acceptance, dict) else [])
        if item.stale_dependencies or item.dependencies != binding_versions(self.runtime.binding()):
            return {"code": "STALE_EVIDENCE",
                    "detail": "[continuation] Evidence is stale. Read current source or run current checks before claiming completion.",
                    "selectors": list(selectors)}
        if not selectors:
            planned_blocker = (getattr(self.engine, "planned_completion_blocker", None)
                               or (lambda: None))()
            if planned_blocker is not None:
                return {"code": "PLANNED_ARTIFACTS_UNRESOLVED", "detail": planned_blocker}
            return None
        public_verdict = getattr(self.runtime, "public_test_verdict", None)
        verdict = public_verdict() if callable(public_verdict) else None
        if verdict is None and hasattr(self.runtime, "test_receipts"):
            acceptance = getattr(item, "acceptance", {})
            required = (acceptance.get("selectors", ())
                        if isinstance(acceptance, dict) else ())
            verdict = evaluate_required_tests(
                self.runtime.binding(), required,
                getattr(self.runtime, "test_receipts", ()))
        if verdict is None:
            bound = getattr(self.runtime, "bound_public_verification", None)
            try:
                bound_current = bound() if callable(bound) else False
            except Exception:  # a missing legacy authority is unresolved, not a crash
                bound_current = False
            if not bound_current:
                return {"code": "MISSING_TEST_EVIDENCE",
                        "detail": "[continuation] Run every required public check against the current workspace."}
        if verdict is not None and verdict.state != SATISFIED:
            selectors = item.acceptance.get("selectors", []) if isinstance(item.acceptance, dict) else []
            code = {"VIOLATED": "TESTS_RED", "CONFLICT": "TEST_EVIDENCE_CONFLICT"}.get(
                verdict.state, "MISSING_TEST_EVIDENCE")
            return {"code": code,
                    "detail": self.completion_blocker() or "[continuation] Run every required public check against the current workspace.",
                    "selectors": list(selectors)}
        owner_finding = self._green_evidence_owner_finding(item, verdict)
        if owner_finding is not None:
            return owner_finding
        planned_blocker = (getattr(self.engine, "planned_completion_blocker", None) or (lambda: None))()
        if planned_blocker is not None:
            return {"code": "PLANNED_ARTIFACTS_UNRESOLVED",
                    "detail": planned_blocker}
        return None

    def _green_evidence_owner_finding(self, item, verdict) -> dict[str, object] | None:
        """R4' preflight: the SAME completion-evidence predicate the commit gate
        runs (``kernel_green_evidence_owner.completion_evidence_failure``), minus
        the certification-order fence (no certification exists yet).  Flag off,
        or no SATISFIED verdict / host evidence to read: no finding (unchanged)."""
        try:
            import kernel_green_evidence_owner as owner
        except ImportError:
            return None
        if not owner.enabled() or verdict is None or verdict.state != SATISFIED:
            return None
        evidence = getattr(self.engine, "_evidence", None)
        if evidence is None:
            return None
        current = self.runtime.binding()
        for reference in sorted(getattr(verdict, "supporting_evidence", ()) or ()):
            code = owner.completion_evidence_failure(
                evidence, self.engine.state, reference, item.intention_id, current)
            if code is not None:
                selectors = item.acceptance.get("selectors", []) if isinstance(item.acceptance, dict) else []
                return {"code": "STALE_EVIDENCE", "failed_predicate": code,
                        "detail": "[continuation] " + owner.typed_refusal(code) + ".",
                        "selectors": list(selectors)}
        return None

    def certify(self, verdict: dict) -> None:
        if self.dispatch is None or self.dispatch["tool"] != "done":
            if self._keyed_completion_durable():
                return      # the keyed complete already landed; the done only reconciles
            raise ContinuationError("completion bypassed the live done fence")
        if self.completed:
            return
        current = self.runtime.binding()
        identity = self.dispatch["continuation_id"]
        item = self.engine.state.intentions[identity]
        event = self.journal.emit("continuation_public_certification", producer="stateframe_runtime",
            intention_id=identity, required_selectors=item.acceptance["selectors"],
            request_id=self.dispatch["request_id"], step=self.dispatch["step"],
            arguments_hash=self.dispatch["arguments_hash"], snapshot_id=self.dispatch["snapshot_id"],
            dispatch_ref=self.dispatch["event_id"], binding=current,
            certification_event_ref=self.certification_ref, verdict=verdict)
        # Certification is evidence produced while a done candidate is being
        # evaluated.  The outer completion gates may still refuse that same
        # candidate, so do not complete the continuation (and clear active_id)
        # until the caller explicitly accepts native done.
        self._pending_completion_ref = event["event_id"]
        self.certified = True

    def submission_result(self, key: tuple[str, str]) -> dict | None:
        """Durable confirmation of a keyed ``complete``, or ``None`` (fail closed).

        The effect and its key are one fsynced ``complete`` transition, so this
        lookup is the destination's own record (STATEFRAME_SUBMIT_IDEMPOTENT).
        Journal errors propagate: "unknown" must never read as "not submitted".
        """
        expected = [str(key[0]), str(key[1])]
        records = getattr(self.journal, "records", None)
        if not callable(records):
            raise ContinuationError("keyed completion lookup needs a replayable journal")
        found = None
        for record in records():
            data = record.get("data")
            if (record.get("kind") == "continuation_transition"
                    and record.get("operation") == "complete"
                    and isinstance(data, dict) and data.get("submission_key") == expected):
                found = {"accepted": True, "submission_key": expected,
                         "event_id": record.get("event_id")}
        return found

    def accept_done(self, verdict: dict | None = None, *,
                    idempotency_key: tuple[str, str] | None = None) -> dict | None:
        """Commit a previously certified native ``done`` after all gates pass."""
        if idempotency_key is not None:
            return self._accept_done_keyed(verdict, idempotency_key)
        if self.completed:
            return
        if self.dispatch is None or self.dispatch["tool"] != "done":
            raise ContinuationError("completion bypassed the live done fence")
        # The live bolt may stage certification from its receipt callback. The
        # checked-in companion returns the verdict without that callback, so
        # accept the verified return value at this final boundary as well.
        if self._pending_completion_ref is None and verdict is not None:
            self.certify(verdict)
        if self._pending_completion_ref is None:
            identity = self.dispatch["continuation_id"]
            item = self.engine.state.intentions[identity]
            acceptance = getattr(item, "acceptance", {})
            selectors = (acceptance.get("selectors", ())
                         if isinstance(acceptance, dict) else ())
            if not selectors:
                self.certify({
                    "state": NO_HOST_CHECKS,
                    "reason_codes": [NO_HOST_CHECKS],
                    "detail": "accepted with no host checks; requirement evidence or model-owned checks apply",
                })
        if self._pending_completion_ref is None:
            raise ContinuationError("accepted done lacks current host certification")
        current = self.runtime.binding()
        identity = self.dispatch["continuation_id"]
        self.engine.apply("complete", intention_id=identity,
                          receipt_ref=self._pending_completion_ref,
                          current_binding=current,
                          current_dependencies=binding_versions(current))
        self.completed = True
        self._pending_completion_ref = None

    def _accept_done_keyed(self, verdict: dict | None, key: tuple[str, str]) -> dict:
        """Keyed ``complete``: at most one per key, and the key is in the record."""
        durable = self.submission_result(key)
        if durable is not None:
            self.completed = True
            self._pending_completion_ref = None
            return durable
        if self.completed:
            raise ContinuationError("continuation completed without this submission key")
        if self.dispatch is None or self.dispatch["tool"] != "done":
            raise ContinuationError("completion bypassed the live done fence")
        if self._pending_completion_ref is None and verdict is not None:
            self.certify(verdict)
        if self._pending_completion_ref is None:
            identity = self.dispatch["continuation_id"]
            item = self.engine.state.intentions[identity]
            acceptance = getattr(item, "acceptance", {})
            selectors = (acceptance.get("selectors", ())
                         if isinstance(acceptance, dict) else ())
            if not selectors:
                self.certify({
                    "state": NO_HOST_CHECKS,
                    "reason_codes": [NO_HOST_CHECKS],
                    "detail": "accepted with no host checks; requirement evidence or model-owned checks apply",
                })
        if self._pending_completion_ref is None:
            raise ContinuationError("accepted done lacks current host certification")
        current = self.runtime.binding()
        expected = [str(key[0]), str(key[1])]
        event = self.engine.apply("complete", intention_id=self.dispatch["continuation_id"],
                                  receipt_ref=self._pending_completion_ref,
                                  current_binding=current,
                                  current_dependencies=binding_versions(current),
                                  submission_key=expected)
        self.completed = True
        self._pending_completion_ref = None
        return {"accepted": True, "submission_key": expected,
                "event_id": (event or {}).get("event_id")}

    def keyed_destination(self, verdict: dict | None):
        """The done callback as a keyed destination (idempotency switch on)."""
        from kernel_submission_idempotency import KeyedDestination
        return KeyedDestination(
            lambda key: self.accept_done(verdict, idempotency_key=key),
            self.submission_result, name="continuation", recovery_context=verdict)

    def returned_source(self, active, state) -> dict | None:
        # Deliver the latest completed child's requested range in the next real
        # parent frame. A bare receipt reference would lose the useful product.
        for reference in reversed(active.return_receipts):
            child = next((item for item in state.intentions.values()
                          if item.parent_id == active.intention_id
                          and item.terminal_receipt == reference
                          and item.acceptance.get("kind") == "source_read"), None)
            if child is None:
                continue
            try:
                source = self.engine.current_source_evidence(reference, self.runtime.binding())
            except ContinuationError:
                return {"receipt_ref": reference, "status": "historical_not_current",
                        "path": child.acceptance["path"]}
            start, end = child.acceptance["start_line"], child.acceptance["end_line"]
            lines = source["content"].split("\n")
            content = "\n".join(lines[start - source["start_line"]:end - source["start_line"] + 1])
            return {"receipt_ref": reference, "status": "current_host_observed_source",
                    "path": source["path"], "revision": source["file_revision"],
                    "start_line": start, "end_line": end, "content": content,
                    "content_hash": digest(content)}
        return None

    def context(self) -> str:
        state = self.engine.state
        active = state.intentions.get(state.active_id) if state.active_id is not None else None
        parent = state.intentions.get(active.parent_id) if active is not None else None
        current_public = getattr(self.runtime, "bound_public_verification", None)
        public_checks_certified = self.certified
        if not public_checks_certified and callable(current_public):
            public_checks_certified = current_public() is True
        coherent_ready = False
        acceptance = getattr(active, "acceptance", {}) if active is not None else {}
        active_selectors = (acceptance.get("selectors", ())
                            if isinstance(acceptance, dict) else ())
        host_check_state = "REQUIRED_PUBLIC_CHECKS" if active_selectors else NO_HOST_CHECKS
        if getattr(self.runtime, "stateframe_green_coherence_mode", "off") == "on":
            try:
                from kernel_green_coherence import terminal_status
                coherent_ready = terminal_status(self.runtime).state == "READY"
            except (AttributeError, TypeError, ValueError):
                coherent_ready = False
        if coherent_ready:
            from kernel_terminal_surface_unify import READY_PENDING_DECISION, REFUSAL_LEAD
        else:
            READY_PENDING_DECISION = None
            REFUSAL_LEAD = None
        pending_decision = (
            READY_PENDING_DECISION
            if coherent_ready else getattr(active, "pending_decision", None)
        )
        parent_pending_decision = (
            READY_PENDING_DECISION
            if coherent_ready else getattr(parent, "pending_decision", None)
        )
        value = {"schema": "live-continuation/1", "revision": state.revision,
                 "active": None if active is None else {
                    "id": active.intention_id, "objective": active.objective,
                    "pending_decision": pending_decision, "acceptance": active.acceptance,
                    "stale_dependencies": active.stale_dependencies, "turns_left": active.turns_left},
                 "parent": None if parent is None else {
                    "id": parent.intention_id, "objective": parent.objective,
                    "pending_decision": parent_pending_decision, "reason_stopped": parent.suspension_reason},
                 "return_receipts": [] if active is None else active.return_receipts[-4:],
                  "public_checks_certified": public_checks_certified,
                  "host_check_state": host_check_state,
                 "unresolved_effects": self._unresolved_effect_details(state.active_id),
                 "returned_source": None if active is None else self.returned_source(active, state),
                 "unfinished": self.engine.unfinished(),
                 "instructions": "Work on the active intention. A source detour completes only after "
                    "a current read covers its declared range. The scheduler then returns to its parent. "
                    "After edits invalidate prior evidence, acquire current source or run current checks "
                    "before another mutation or completion. Saved notes and model claims are not completion proof. "
                     "When public checks are certified, decide whether to continue the objective or call native done; "
                     "certification alone does not end the run."}
        if host_check_state == NO_HOST_CHECKS:
            value["instructions"] = (
                "No public host check exists for this task. Work from the requirement evidence, "
                "or author and run your own checks with run_tests when useful. Completion may be "
                "accepted with this no-host-checks limitation disclosed; do not claim host-test coverage."
            )
        if self.guarded_effects is not None:
            value["guarded_effects"] = self.guarded_effects.context_dict()
        if coherent_ready:
            value["instructions"] = REFUSAL_LEAD
        if self._stranded_blocker is not None:
            value["blocker"] = self._stranded_blocker
            value["instructions"] += (
                " The continuation is temporarily stranded; resolve this blocker before claiming completion: "
                + self._stranded_blocker)
        if self.context_scope is not None:
            value["context_scope"] = self.context_scope.public_context()
        adapter = self._unknown_outcome_adapter()
        if adapter is not None:
            value["external_outcomes"] = adapter.context()
        from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
        if candidate_reconcile_enabled():
            from kernel_candidate_restart_reconcile import context_payload
            recovery = context_payload(self.runtime)
            if recovery is not None:
                value["candidate_restart_recovery"] = recovery
        planned = getattr(self.engine, "planned_artifacts", None)
        if planned is not None:
            value["planned_artifacts"] = {
                "schema": "planned-artifacts/1",
                "closure_blockers": list(planned.completion_blockers()),
                "records": [record.to_dict() for record in planned.artifacts.values()],
            }
        returns = getattr(self.engine, "return_ledger", None)
        if returns is not None and active is not None:
            projection = returns.projection_dict(active.intention_id)
            if projection["depth"]:
                value["continuation_return"] = projection
        retention = self.engine.retention_summary()
        if retention is not None:
            value["retention"] = retention
        if self.runtime.continuation_source_refresh:
            value["instructions"] += (
                " Complete current Working source in this model request also counts as a source "
                "observation after the kernel verifies actual delivery. It does not refresh test "
                "results or prove completion. Queued actions do not receive a fresh model observation.")
        return "[Continuation]\n" + json.dumps(value, ensure_ascii=False, sort_keys=True)

    def finish(self) -> None:
        self.journal.emit("continuation_worker_stopped", state=self.engine.state.document(),
                          unfinished=self.engine.unfinished(),
                          worker_resume_implemented=bool(getattr(self.runtime, "restart_enabled", False)))
        if self.engine.unfinished():
            self.runtime.system_outcome = self.runtime.system_outcome or "continuation_incomplete"
