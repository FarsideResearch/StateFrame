"""Opt-in exactly-once submission (STATEFRAME_SUBMIT_IDEMPOTENT=on, default off).

Problem (coverage_reminders review18 #2): the reducer runs the caller's
``commit()`` effect (kernel_verdict) BEFORE the challenge budget persists the
terminal submission.  A crash in between leaves no durable marker, so a later
``done`` at the same product revision with a fresh tool-call id re-runs the
effect.

A local marker alone cannot fix this: a marker written before the effect cannot
tell "never committed" from "committed, then crashed", and a marker written
after it has the same gap.  So when this switch is on the effect destination
must participate:

* An effectful submission must be a :class:`KeyedDestination`.  Its
  ``submit(key)`` performs the effect and writes a durable record carrying the
  idempotency key in ONE atomic write; ``result_for(key)`` returns that
  record's confirmation, or ``None`` when no effect with that key exists.
  Lookup errors must raise (fail closed), never answer ``None``.
* The key is ``(logical run, product revision)``.  The logical run id is
  minted once and persisted in the challenge-budget state, so a restarted
  runtime restored from the journal reuses it.
* Protocol per effect call: resolve any durable intent against the
  destination first; otherwise persist a pending intent, then submit with the
  key, then persist the confirmation.  Terminal state is written by the
  challenge budget's ``commit_submission`` only after the reducer accepted a
  confirmed effect; the confirmation is copied into the terminal record in
  the same journal write.
* A plain (unkeyed) callable is refused with :class:`SubmissionIdempotencyError`
  before the reducer runs.  No silent unkeyed fallback exists in enabled mode.
* No callback at all is a local submission: there is no external effect, and
  the terminal record written by ``commit_submission`` is the submission.
* Enabled without a challenge budget is rejected at runtime construction; the
  budget journal is the only durable state this protocol uses.

Residual risk (stated, not hidden): exactly-once holds only if the
destination's effect and its keyed record really are one atomic write.  The
continuation destination (``LiveContinuation.accept_done``) satisfies this: the
effect IS the fsynced ``complete`` transition, and the key is stored in it.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping


MODE_ENV = "STATEFRAME_SUBMIT_IDEMPOTENT"
SCHEMA = "stateframe-submission-intent/1"
EVENT = "submission_idempotency"


class SubmissionIdempotencyError(RuntimeError):
    """Unsupported configuration or unconfirmed effect.

    Deliberately a RuntimeError, not ValueError/TypeError: several optional
    runtime hooks swallow those, and this error must never be swallowed.
    """


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    if runtime is not None:
        value = getattr(runtime, "stateframe_submit_idempotent_mode", None)
        if value is not None:
            return str(value).strip().lower() == "on"
    return mode() == "on"


def validate_configuration(idempotency_mode: object, challenge_budget: object) -> None:
    """Refuse the half-enabled configuration instead of silently degrading."""
    if str(idempotency_mode).strip().lower() == "on" and challenge_budget is None:
        raise SubmissionIdempotencyError(
            f"{MODE_ENV}=on requires STATEFRAME_CHALLENGE_BUDGET: the challenge-budget "
            "journal holds the durable submission intent")


RESERVED_NAMES = frozenset({"destination", "default"})


class KeyedDestination:
    """An effect destination that deduplicates by idempotency key.

    ``submit(key)`` must perform the effect and durably record ``key`` in one
    atomic write and return a confirmation that is neither ``None`` nor
    ``False`` (``False`` means the destination refused and recorded nothing).
    ``submit(None)`` is the legacy unkeyed effect, used only when the switch
    is off.  ``result_for(key)`` returns the durable confirmation or ``None``.
    """

    def __init__(self, submit: Callable[[object], object],
                 result_for: Callable[[tuple[str, str]], object], *, name: str,
                 recovery_context: object = None):
        if not callable(submit) or not callable(result_for):
            raise TypeError("a keyed destination needs callable submit and result_for")
        # ``name`` is the stable identity of the deduplicating store/namespace.
        # An intent is bound to it; a pending intent is only ever resolved
        # through the destination that carries it (review6 #2: no shared default).
        if not isinstance(name, str) or not name.strip() or name.strip() in RESERVED_NAMES:
            raise ValueError("a keyed destination needs an explicit, stable, non-default identity "
                             f"(got {name!r})")
        self._submit = submit
        self._result_for = result_for
        self.name = name.strip()
        # JSON-safe context the runtime needs to rebuild and submit through this
        # destination later without the caller (e.g. the continuation verdict).
        self.recovery_context = recovery_context

    def submit(self, key: tuple[str, str] | None) -> object:
        return self._submit(key)

    def result_for(self, key: tuple[str, str]) -> object:
        return self._result_for(key)

    def __call__(self) -> object:
        # Reached only when the switch is off: the reducer's legacy zero-arg call.
        return self._submit(None)


def guard_commit(runtime: object, commit):
    """Return the reducer callback for this done.

    Off: ``commit`` unchanged.  On: ``None`` stays ``None`` (local
    submission); a :class:`KeyedDestination` becomes a zero-arg guarded
    callback; anything else is refused before the reducer runs.
    """
    if not enabled(runtime):
        return commit
    if commit is None:
        return _pending_intent_resolver(runtime)
    if isinstance(commit, _GuardedSubmission):
        return commit
    from kernel_challenge_budget import budget_for
    validate_configuration("on", budget_for(runtime))
    if not isinstance(commit, KeyedDestination):
        raise SubmissionIdempotencyError(
            f"{MODE_ENV}=on: an effectful done callback must be a KeyedDestination "
            "(durable submit(key) + result_for(key)); an unkeyed callable cannot be "
            f"made exactly-once across a crash (got {type(commit).__name__})")
    return _GuardedSubmission(runtime, commit)


# Contract (review6): a callback-free done while a keyed intent is PENDING must
# itself be able to finish the operation.  The runtime rebuilds the ORIGINAL
# destination from the intent's durable binding (identity + recovery context)
# through ``submission_destination_provider`` or ``runtime.continuation`` and
# runs the ordinary keyed protocol: an effect it already holds is reconciled,
# otherwise it submits the same key exactly once.
#   * No destination can be built at the moment -> non-terminal
#     KEYED_SUBMISSION_PENDING; the durable intent remains recoverable.
#   * Built but the attempt fails -> KEYED_SUBMISSION_PENDING with the attempt
#     count in telemetry.  A later healthy destination can always resolve it.
CALLBACK_FREE_CAP = 2
PENDING_CODE = "KEYED_SUBMISSION_PENDING"
UNRESOLVED_CODE = "KEYED_SUBMISSION_UNRESOLVED"


class PendingKeyedSubmission(SubmissionIdempotencyError):
    """Non-terminal: the rebuilt destination failed once; retry on the next done."""


class KeyedSubmissionUnresolved(SubmissionIdempotencyError):
    """Reserved for an explicitly recorded non-retryable unresolved state."""


def _original_destination(runtime: object, slot: object):
    """Rebuild the destination an intent is bound to, or (None, reason).

    Construction failures are an unavailable destination, never an escape."""
    slot = slot if isinstance(slot, Mapping) else {}
    name = str(slot.get("destination"))
    context = slot.get("recovery_context")
    try:
        provider = getattr(runtime, "submission_destination_provider", None)
        if callable(provider):
            destination = provider(name)
        elif name == "continuation" and getattr(runtime, "continuation", None) is not None:
            destination = runtime.continuation.keyed_destination(context)
        else:
            destination = None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(destination, KeyedDestination) or destination.name != name:
        return None, f"original destination {name!r} cannot be built by this runtime"
    return destination, None


def _pending_intent_resolver(runtime: object):
    """Reducer callback for a callback-free done while a keyed intent is PENDING."""
    from kernel_challenge_budget import budget_for, submission_slot
    if budget_for(runtime) is None:
        return None
    slot = submission_slot(runtime)
    if slot is None or slot.get("effect_status") != "pending":
        return None                   # genuinely local, or already confirmed

    counted: dict[str, int] = {}

    def resolve():
        import kernel_challenge_budget as budget
        key = budget.slot_key(slot)
        name = str(slot.get("destination"))
        original, reason = _original_destination(runtime, slot)
        info = {"submission_key": list(key), "destination": name}
        if original is None:
            # Destination construction is an infrastructure/recovery failure, not
            # proof that the keyed effect is impossible.  Keep the durable intent
            # so a later healthy host can resolve the original destination rather
            # than turning a temporarily unavailable provider into an absorbing
            # terminal refusal.
            info.update(attempts=0, reason=reason, retryable=True)
            setattr(runtime, "_submission_pending", info)
            _emit(runtime, "pending", **info)
            raise PendingKeyedSubmission(reason)
        _emit(runtime, "resolving", submission_key=list(key), destination=name)
        try:
            result = _GuardedSubmission(runtime, original)()
            if result is False:
                raise SubmissionIdempotencyError("original destination refused the submission")
            # A later re-entry of the reducer in the SAME done succeeded: an earlier
            # failure's projection must not turn this accepted submission into a
            # refusal (review7 #1).
            setattr(runtime, "_submission_pending", None)
            setattr(runtime, "_submission_unresolved", None)
            return result
        except Exception as error:
            # One done counts once, however often the gates re-enter the reducer.
            if "attempts" not in counted:
                current = budget.submission_slot(runtime)
                if current is not None and budget.slot_key(current) == key:
                    counted["attempts"] = budget.note_callback_free_attempt(runtime, key)
                else:
                    counted["attempts"] = CALLBACK_FREE_CAP
            info.update(attempts=counted["attempts"], reason=f"{type(error).__name__}: {error}")
            # A destination can recover after any number of callback-free
            # attempts; the durable intent is the retry contract.  The historical
            # cap remains telemetry for operators, never an absorbing terminal
            # state that permanently refuses a later healthy destination.
            info["retryable"] = True
            setattr(runtime, "_submission_pending", info)
            _emit(runtime, "pending", **info)
            raise PendingKeyedSubmission(info["reason"]) from error
    return resolve


def finalize_decision(runtime: object, decision):
    """Project the resolver's outcome onto the done decision (switch on only)."""
    unresolved = getattr(runtime, "_submission_unresolved", None)
    pending = getattr(runtime, "_submission_pending", None)
    setattr(runtime, "_submission_unresolved", None)
    setattr(runtime, "_submission_pending", None)
    if unresolved is None and pending is None:
        return decision
    import dataclasses
    from kernel_verdict import BLOCKED
    if unresolved is not None:
        detail = (f"{UNRESOLVED_CODE}: the keyed submission {unresolved['submission_key']} "
                  f"to {unresolved['destination']!r} could not be completed "
                  f"({unresolved['reason']}). The run ends unresolved; it is NOT submitted.")
        final = dataclasses.replace(decision, outcome=BLOCKED, accepted=False,
                                    reason_codes=(UNRESOLVED_CODE,), detail=detail,
                                    repair_action=None)
        import kernel_challenge_budget as budget
        budget.terminalize_unresolved(runtime, unresolved, final)
        return final
    detail = (f"{PENDING_CODE}: a keyed submission {pending['submission_key']} is pending at "
              f"{pending['destination']!r}; the attempt failed ({pending['reason']}). Call done "
              "again: the next done retries it once more.")
    return dataclasses.replace(decision, outcome=BLOCKED, accepted=False,
                               reason_codes=(PENDING_CODE,), detail=detail)


def _emit(runtime: object, phase: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(EVENT, producer="kernel_submission_idempotency", phase=phase,
             intent_schema=SCHEMA, dispatch=False, **fields)


def _confirmed(value: object) -> bool:
    return value is not None and value is not False


class _GuardedSubmission:
    """Zero-arg reducer callback enforcing intent -> keyed effect -> confirmation."""

    def __init__(self, runtime: object, destination: KeyedDestination):
        self.runtime = runtime
        self.destination = destination

    def __call__(self) -> object:
        import kernel_challenge_budget as budget
        runtime = self.runtime
        destination = self.destination
        key = budget.submission_key(runtime, budget.artifact_revision(runtime))
        slot = budget.submission_slot(runtime)
        if slot is not None and slot.get("effect_status") == "confirmed":
            # Our own durable record already holds the destination's
            # confirmation: never ask again, never submit again.
            _emit(runtime, "reconciled", submission_key=list(budget.slot_key(slot)),
                  destination=destination.name, source="intent")
            return slot.get("result")
        slot_destination = destination
        if slot is not None:
            # The intent is bound to the destination identity it was reserved
            # against.  Prefer the runtime's authoritative rebuild of that
            # binding; a caller-supplied destination is used only when it carries
            # the same identity (review5 #2, review6 #2).
            rebuilt, reason = _original_destination(runtime, slot)
            if rebuilt is not None:
                slot_destination = rebuilt
            elif str(slot.get("destination")) != destination.name:
                _emit(runtime, "destination_mismatch", submission_key=list(budget.slot_key(slot)),
                      destination=destination.name, bound_to=str(slot.get("destination")))
                raise SubmissionIdempotencyError(
                    f"intent is bound to {slot.get('destination')!r}; {reason}")
        if slot is not None:
            slot_key = budget.slot_key(slot)
            # Destination lookup errors propagate: an unknown answer must never
            # be read as "nothing was submitted".
            found = slot_destination.result_for(slot_key)
            if _confirmed(found):
                budget.mark_submission_effect(runtime, slot_key, result=found)
                _emit(runtime, "reconciled", submission_key=list(slot_key),
                      destination=slot_destination.name)
                return found
            if slot_key != key:
                budget.withdraw_submission_intent(
                    runtime, slot_key, reason="superseded_without_destination_result")
                _emit(runtime, "withdrawn", submission_key=list(slot_key),
                      reason="superseded_without_destination_result")
                slot = None
        if slot is None:
            budget.reserve_submission_intent(runtime, key, destination=destination.name,
                                             recovery_context=destination.recovery_context)
            _emit(runtime, "reserved", submission_key=list(key), destination=destination.name)
        else:
            destination = slot_destination      # same key, same (original) destination
        result = destination.submit(key)
        if result is False:
            # The destination refused and recorded nothing: no misleading intent
            # may survive a refusal.
            budget.withdraw_submission_intent(runtime, key, reason="destination_refused")
            _emit(runtime, "withdrawn", submission_key=list(key), reason="destination_refused")
            return False
        if result is None:
            # Effect state unknown: keep the pending intent so recovery asks the
            # destination by key, and fail this admission.
            _emit(runtime, "unconfirmed", submission_key=list(key), destination=destination.name)
            raise SubmissionIdempotencyError(
                "keyed destination returned no confirmation; the intent stays pending")
        budget.mark_submission_effect(runtime, key, result=result)
        _emit(runtime, "confirmed", submission_key=list(key), destination=destination.name)
        return result


__all__ = [
    "CALLBACK_FREE_CAP", "EVENT", "KeyedDestination", "KeyedSubmissionUnresolved", "MODE_ENV",
    "PENDING_CODE", "PendingKeyedSubmission", "SCHEMA", "SubmissionIdempotencyError",
    "UNRESOLVED_CODE", "finalize_decision",
    "enabled", "guard_commit", "mode", "validate_configuration",
]
