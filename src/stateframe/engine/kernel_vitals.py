"""Receipt-backed behavioral vitals for the repair loop.

The EKG is deliberately an organ, not another repair policy.  It remembers the
trajectory that a small model cannot reliably carry, raises one bounded
challenge when the trajectory has become informationally stagnant, and exposes
counterfactual escalation state for the orchestrator.  A caller that has a
``KernelBolt`` should pass ``bolt.organ_out`` as ``organ_out`` (or wrap the
return value itself) so E332 shadow mode remains the only model-output
chokepoint.

All measurements use receipt order and the receipt ``turn`` field.  ``ts`` is
never read.  Missing or malformed evidence can make a vital ``UNKNOWN`` or the
whole EKG's coverage incomplete, but it never produces an advisory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import re

from kernel_liveness import Evidence, OrganCensusSpec, TriState


OutputHook = Callable[[str, str | None], str | None]


class _Status:
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


@dataclass
class VitalState:
    """Public, small state card for one vital."""

    name: str
    state: str = _Status.UNKNOWN
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state, **self.metrics}

    def __getitem__(self, key: str) -> object:
        if key == "state":
            return self.state
        return self.metrics[key]

    def get(self, key: str, default: object = None) -> object:
        if key == "state":
            return self.state
        return self.metrics.get(key, default)


@dataclass(frozen=True)
class _Action:
    """Normalized E334 action plus derived, revision-aware identities."""

    index: int
    turn: int
    kind: str
    intent: str
    result_class: str
    why: str
    fingerprint: str | None
    target_key: str | None
    file_key: str | None
    inspection_key: str | None
    selector: str | None
    scoped_test: bool
    workspace_revision: int
    anchor: str | None
    new_fact: bool
    inspection_fact_new: bool
    selector_new: bool
    test_result_new: bool
    new_file: bool
    evidence_complete: bool
    consequential: bool


class VitalsEKG:
    """Deterministic, shadow-first trajectory monitor.

    The constructor has no repository or clock dependency.  This is
    intentional: the E334 receipt is the contract, and the monitor can be
    replayed from a JSONL tape in exactly the same way it runs live.
    """

    SCHEMA = "poc.trajectory_ekg/1"
    ORGAN_NAME = "vitals"
    ADVISORY = (
        "your last 8 actions produced no new information; state your current "
        "hypothesis before the next edit"
    )

    # Measured cutoffs from sol_answers/cs_vitals.md.
    SCOPE_WINDOW = 4
    SCOPE_MIN_EDITS = 3
    SCOPE_MIN_TEST_EDITS = 2
    SCOPE_MAX_PRODUCT_EDITS = 1
    ACTION_WINDOW = 8
    MIN_INSPECTIONS = 4
    MIN_SCOPED_TESTS = 3
    MIN_EDITS_SINCE_FACT = 4
    FILE_DWELL_ACTIONS = 8
    FILE_DWELL_FACT_WINDOW = 4
    OSCILLATION_MAX_SPAN = 8
    REREAD_MIN_REPEAT_RATIO = 0.75
    DECLARATION_MIN_COVERAGE = 0.50

    # The archive gives no declaration-refusal cutoff.  Three consecutive
    # omissions is the smallest useful bounded streak and is kept explicit so
    # future receipt measurements can revise it without changing the API.
    DECLARATION_REFUSAL_STREAK = 3
    ESCALATION_WINDOW = 4
    ESCALATION_MIN_NO_FACT = 3

    VITAL_NAMES = (
        "scope_skew",
        "edit_oscillation",
        "reinspection_stall",
        "reread_rate",
        "selector_stall",
        "edit_fact_gap",
        "file_dwell",
        "declaration_refusal",
    )

    # Raw reread rate was measured as unusable because it false-flagged slow
    # winners.  Keep its state for telemetry, but do not let it alone earn the
    # model-facing advisory.
    ADVISORY_VITALS = frozenset(
        {
            "scope_skew",
            "edit_oscillation",
            "reinspection_stall",
            "selector_stall",
            "edit_fact_gap",
            "file_dwell",
            "declaration_refusal",
        }
    )

    _ACTION_KINDS = frozenset(
        {
            "grep",
            "read",
            "edit",
            "test",
            "done",
            "surgery",
            "context_surgery",
            "get_sure",
            "sure",
        }
    )
    _CONSEQUENTIAL_KINDS = frozenset(
        {"grep", "read", "edit", "test", "done"}
    )
    _TEST_PATH = re.compile(
        r"(?:^|/)(?:tests?|specs?)(?:/|$)|"
        r"(?:^|/)test_[^/]+$|(?:^|/)[^/]+_test\.[^/]+$|"
        r"(?:^|/)[^/]+\.test\.[^/]+$",
        re.IGNORECASE,
    )
    _READ_PREFIXES = ("read ", "open ", "cat ")
    _EDIT_PREFIXES = ("replace_fn ", "str_replace ", "file_create ", "edit ")

    def __init__(
        self,
        task_id: str = "",
        *,
        organ_out: OutputHook | None = None,
        organ_name: str = ORGAN_NAME,
    ) -> None:
        self.task_id = task_id
        self.organ_name = organ_name or self.ORGAN_NAME
        self._organ_out = organ_out

        self.receipts: list[dict[str, object]] = []
        self.history: list[_Action] = []
        self.trace: list[dict[str, object]] = []
        # ``events`` is a readable alias used by offline probes.
        self.events = self.trace
        self.vitals: dict[str, VitalState] = {
            name: VitalState(name) for name in self.VITAL_NAMES
        }
        self.per_vital = self.vitals

        self.coverage = "UNKNOWN"
        self.state = _Status.UNKNOWN
        self.active_vitals: tuple[str, ...] = ()
        self.last_fact_delta: tuple[str, ...] = ()
        self.last_receipt: dict[str, object] | None = None
        self.last_decision: dict[str, object] = {
            "event": None,
            "would_fire": False,
            "emitted": False,
            "advisory_ordinal": 0,
        }

        self._action_index = 0
        self._last_turn: int | None = None
        self._last_revision = 0
        self._coverage_complete = True
        self._seen_facts: set[str] = set()
        self._seen_inspections: set[str] = set()
        self._seen_selectors: set[str] = set()
        self._seen_test_results: set[str] = set()
        self._seen_files: set[str] = set()
        self._edits_since_fact = 0
        self._task_open = True

        self._advisory_fired = False
        self.advisory_count = 0
        self._advisory_severity = 0
        self._advisory_action_index: int | None = None
        self._post_advisory_actions: list[_Action] = []
        self._post_advisory_window_complete = False
        self._surgery_done = False
        self._get_sure_used = False
        self.recommend_surgery = False
        self._terminal = False
        self.terminal_event: str | None = None
        self._escalation_state = "NORMAL"
        self._last_active_count = 0

        # E332 remains outside this class by default.  Supplying the hook lets
        # live callers use KernelBolt.organ_out without importing kernel_bolt.
        self._record_event(
            "ARMED",
            turn=None,
            detail=(
                f"schema={self.SCHEMA} shadow=organ_out "
                f"window={self.ACTION_WINDOW} "
                f"scope={self.SCOPE_WINDOW}/{self.SCOPE_MIN_TEST_EDITS}/"
                f"{self.SCOPE_MAX_PRODUCT_EDITS} "
                f"declaration_coverage={self.DECLARATION_MIN_COVERAGE:.2f}"
            ),
        )

    # ------------------------------------------------------------------ public
    def observe(self, receipt_dict: Mapping[str, object]) -> None:
        """Fold one E334 receipt into the EKG.

        Observation is total: malformed input is retained as incomplete
        evidence and never raises into the repair loop.  It also never emits
        model-facing text; only :meth:`check` can return the bounded advisory.
        """

        self._action_index += 1
        self.last_decision = {
            "event": None,
            "would_fire": False,
            "emitted": False,
            "advisory_ordinal": self.advisory_count,
        }
        if not isinstance(receipt_dict, Mapping):
            self._coverage_complete = False
            self.coverage = "INCOMPLETE"
            self.last_fact_delta = ()
            self._record_event(
                "RECEIPT_UNKNOWN",
                turn=self._action_index,
                detail="receipt is not a mapping",
            )
            return

        raw = dict(receipt_dict)
        self.receipts.append(raw)
        self.last_receipt = raw
        action = self._normalize(raw, self._action_index)
        self.history.append(action)
        if len(self.history) > 256:
            del self.history[:-256]

        self.last_fact_delta = self._fact_delta_for(action)
        if action.new_fact:
            self._edits_since_fact = 0
        elif action.kind == "edit" and action.result_class == "APPLIED":
            self._edits_since_fact += 1

        if action.kind == "done" and action.result_class not in {
            "CHALLENGED",
            "REFUSED",
            "REJECTED",
        }:
            # A challenged done receipt is a continuation, exactly as in the
            # live KernelBolt path.  An accepted/ordinary done closes the task,
            # but never gets vetoed by the EKG.
            self._task_open = False

        if action.kind in {"surgery", "context_surgery"}:
            self.note_surgery(turn=action.turn)
        elif action.kind in {"get_sure", "sure"}:
            self._record_get_sure(turn=action.turn)
        elif (
            "context_surgery" in action.intent.lower()
            or "context surgery" in action.intent.lower()
        ):
            self.note_surgery(turn=action.turn)
        elif (
            "get_sure" in action.intent.lower()
            or "sure_orient" in action.intent.lower()
        ):
            self._record_get_sure(turn=action.turn)

        previous_active = self._last_active_count
        self._evaluate_vitals()
        current_active = len(self.active_vitals)

        if not self._terminal:
            recovered = action.new_fact or (
                previous_active > 0 and current_active < previous_active
            )
            if recovered and (self._advisory_fired or self.recommend_surgery):
                self._reset_episode(action.turn, "new fact or lower active-vital count")
            elif current_active > 0 and previous_active == 0:
                self._escalation_state = "NORMAL"

            self._last_active_count = current_active
            self._advance_escalation(action)
            self._maybe_stall(action.turn)
        else:
            self._last_active_count = current_active

    def check(self, organ_out: OutputHook | None = None) -> str | None:
        """Return the one bounded advisory, or ``None``.

        The optional hook is normally ``KernelBolt.organ_out``.  With
        ``KERNEL_SHADOW`` configured there, the hook returns ``None`` while
        retaining the would-fire trace marker.  Without a hook, returning the
        line is useful for direct/offline tests.
        """

        if self._terminal or not self._task_open:
            return None
        if self.coverage != "COMPLETE":
            return None
        if len(self.history) < self.ACTION_WINDOW:
            return None
        window = self.history[-self.ACTION_WINDOW :]
        if any(not action.evidence_complete for action in window):
            return None
        if any(action.new_fact for action in window):
            self._record_event(
                "EKG_SUPPRESSED",
                turn=window[-1].turn,
                detail="new fact in action window",
            )
            return None
        if not self.active_vitals:
            return None
        if self._advisory_fired:
            self._record_event(
                "EKG_DEDUPED",
                turn=window[-1].turn,
                detail="advisory already fired",
            )
            self._record_event(
                "EKG_CHALLENGE_ONCE",
                turn=window[-1].turn,
                detail="one advisory per trajectory episode",
            )
            return None

        # UNKNOWN vitals are absent from active_vitals, so they can never make
        # this branch fire.
        self._advisory_fired = True
        self.advisory_count += 1
        self._advisory_severity = len(self.active_vitals)
        self._advisory_action_index = self.history[-1].index
        self._post_advisory_actions = []
        self._post_advisory_window_complete = False
        self._escalation_state = "ADVISORY"
        turn = window[-1].turn
        self._record_event(
            "EKG_ADVISORY",
            turn=turn,
            detail=(
                f"active={','.join(self.active_vitals)} "
                f"severity={self._advisory_severity}"
            ),
        )

        hook = organ_out or self._organ_out
        if hook is None:
            self.last_decision = {
                "event": "EKG_ADVISORY",
                "would_fire": True,
                "emitted": True,
                "advisory_ordinal": self.advisory_count - 1,
            }
            return self.ADVISORY
        try:
            line = hook(self.organ_name, self.ADVISORY)
            self.last_decision = {
                "event": "EKG_ADVISORY",
                "would_fire": True,
                "emitted": line is not None,
                "advisory_ordinal": self.advisory_count - 1,
            }
            return line
        except Exception as exc:  # noqa: BLE001 - organ must fail open
            self._record_event(
                "WIRING_ERROR",
                turn=turn,
                detail=f"organ_out failed: {type(exc).__name__}",
            )
            print(
                f"    [kernel-wiring] vitals organ_out failed: {type(exc).__name__}"
            )
            self.last_decision = {
                "event": "EKG_ADVISORY",
                "would_fire": True,
                "emitted": False,
                "advisory_ordinal": self.advisory_count - 1,
            }
            return None

    def note_surgery(self, *, turn: int | None = None) -> None:
        """Record the one bounded context-surgery response."""

        if self._terminal or not self.recommend_surgery or self._surgery_done:
            return
        self._surgery_done = True
        self._escalation_state = "SURGERY_RECOMMENDED"
        self._record_event(
            "EKG_CONTEXT_SURGERY",
            turn=turn if turn is not None else self._current_turn(),
            detail="one bounded context-only capsule requested",
        )

    def note_get_sure(self, *, turn: int | None = None) -> None:
        """Record at most one eligible get_sure response after surgery.

        Eligibility is a live-orchestrator concern: callers should invoke this
        only for a fresh RED/NO_VERDICT result with an ungrounded anchor.  The
        EKG records the bounded round; it does not run or synthesize it.
        """

        was_used = self._get_sure_used
        self._record_get_sure(turn=turn)
        if not was_used and self._get_sure_used:
            self._maybe_stall(turn if turn is not None else self._current_turn())

    def _record_get_sure(self, *, turn: int | None = None) -> None:
        """Record a get_sure marker while an enclosing observe is in flight."""

        if self._terminal or not self.recommend_surgery or not self._surgery_done:
            return
        if self._get_sure_used:
            return
        self._get_sure_used = True
        self._escalation_state = "GET_SURE"
        self._record_event(
            "EKG_GET_SURE",
            turn=turn if turn is not None else self._current_turn(),
            detail="one bounded eligible round recorded",
        )

    # Friendly aliases for orchestrator adapters and replay probes.
    mark_surgery = note_surgery
    note_context_surgery = note_surgery
    mark_context_surgery = note_surgery
    record_surgery = note_surgery
    on_surgery = note_surgery
    mark_get_sure = note_get_sure
    record_get_sure = note_get_sure
    on_get_sure = note_get_sure

    @property
    def escalation_state(self) -> str:
        return self._escalation_state

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def stalled(self) -> bool:
        return self._terminal

    @property
    def task_open(self) -> bool:
        return self._task_open

    @property
    def advisory_fired(self) -> bool:
        return self._advisory_fired

    @property
    def surgery_done(self) -> bool:
        return self._surgery_done

    @property
    def get_sure_used(self) -> bool:
        return self._get_sure_used

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly state card for shadow projection/tests."""

        latest = self.history[-1] if self.history else None
        action_card = None
        if latest is not None:
            action_card = {
                "kind": latest.kind,
                "result_class": latest.result_class,
                "target_key": latest.target_key,
                "payload_fingerprint": latest.fingerprint,
            }
        window = self.history[-self.ACTION_WINDOW :]
        return {
            "schema": self.SCHEMA,
            "task_id": self.task_id,
            "coverage": self.coverage,
            "clock": {
                "turn": self._current_turn(),
                "action_index": self._action_index,
                "workspace_revision": self._last_revision,
            },
            "action": action_card,
            "fact_delta": list(self.last_fact_delta),
            "vitals": {name: card.to_dict() for name, card in self.vitals.items()},
            "ekg": {
                "state": self.state,
                "active_vitals": list(self.active_vitals),
                "no_new_information_8": bool(window)
                and len(window) == self.ACTION_WINDOW
                and not any(action.new_fact for action in window),
                "advisory_eligible": self._advisory_eligible(),
            },
            "escalation": {
                "state": self._escalation_state,
                "recommend_surgery": self.recommend_surgery,
                "terminal_event": self.terminal_event,
            },
            "decision": dict(self.last_decision),
        }

    # --------------------------------------------------------------- normalize
    def _normalize(self, receipt: Mapping[str, object], index: int) -> _Action:
        kind = self._text(receipt.get("kind")).lower().replace("-", "_")
        intent = self._text(
            receipt.get("intent")
            or receipt.get("query")
            or receipt.get("selector")
            or ""
        ).strip()
        result = self._text(receipt.get("result_class")).upper().strip()
        why = self._text(receipt.get("why")).strip()
        fingerprint = self._optional_text(
            receipt.get("payload_fingerprint") or receipt.get("fingerprint")
        )
        anchor = self._optional_text(
            receipt.get("anchor") or receipt.get("primary_anchor")
        )
        turn_value = self._as_int(receipt.get("turn"))
        turn_valid = turn_value is None or turn_value >= 0
        if turn_value is None or not turn_valid:
            turn = index
        else:
            turn = turn_value
        if turn_valid and turn_value is not None and self._last_turn is not None:
            turn_valid = turn_value >= self._last_turn
        if turn_value is not None and turn_valid:
            self._last_turn = turn_value

        revision_value = self._as_int(receipt.get("workspace_revision"))
        revision_present = revision_value is not None
        revision = revision_value if revision_value is not None else 0
        self._last_revision = revision

        file_key = self._file_key(receipt, kind, intent, anchor)
        target_key = self._target_key(receipt, file_key, kind, intent)
        selector = self._selector(receipt, kind, intent)
        scoped_test = kind == "test" and self._is_scoped_selector(selector)

        explicit_fact_present = "fact_delta" in receipt
        explicit_fact_keys = self._explicit_fact_keys(receipt, revision)
        derived_keys = self._derived_fact_keys(
            receipt=receipt,
            kind=kind,
            intent=intent,
            file_key=file_key,
            selector=selector,
            result=result,
            revision=revision,
            anchor=anchor,
        )
        fact_keys = (
            explicit_fact_keys
            if explicit_fact_present
            else derived_keys
        )
        unseen_facts = tuple(key for key in fact_keys if key not in self._seen_facts)
        for key in fact_keys:
            self._seen_facts.add(key)
        new_fact = bool(unseen_facts)

        inspection_key = self._inspection_key(
            kind, receipt, intent, file_key, revision
        )
        inspection_fact_new = False
        if inspection_key is not None:
            inspection_fact_new = inspection_key not in self._seen_inspections
            # An explicit empty fact_delta is authoritative: it is useful for
            # replaying a known no-information receipt without inventing a fact.
            if explicit_fact_present and not explicit_fact_keys:
                inspection_fact_new = False
            self._seen_inspections.add(inspection_key)

        selector_new = False
        if kind == "test" and selector:
            selector_new = selector not in self._seen_selectors
            self._seen_selectors.add(selector)

        test_result_new = False
        if kind == "test" and selector and result:
            test_result_key = self._key(
                "test-result", selector, result, str(revision)
            )
            test_result_new = test_result_key not in self._seen_test_results
            self._seen_test_results.add(test_result_key)

        new_file = False
        if file_key:
            new_file = file_key not in self._seen_files
            self._seen_files.add(file_key)

        intent_present = bool(intent) or kind in {
            "surgery",
            "context_surgery",
            "get_sure",
            "sure",
        }
        evidence_complete = (
            kind in self._ACTION_KINDS
            and bool(result)
            and intent_present
            and revision_present
            and turn_value is not None
            and turn_valid
            and not (
                receipt.get("coverage") is not None
                and self._text(receipt.get("coverage")).upper() != "COMPLETE"
            )
        )
        if not evidence_complete:
            self._coverage_complete = False
        self.coverage = "COMPLETE" if self._coverage_complete else "INCOMPLETE"

        return _Action(
            index=index,
            turn=turn,
            kind=kind,
            intent=intent,
            result_class=result,
            why=why,
            fingerprint=fingerprint,
            target_key=target_key,
            file_key=file_key,
            inspection_key=inspection_key,
            selector=selector,
            scoped_test=scoped_test,
            workspace_revision=revision,
            anchor=anchor,
            new_fact=new_fact,
            inspection_fact_new=inspection_fact_new,
            selector_new=selector_new,
            test_result_new=test_result_new,
            new_file=new_file,
            evidence_complete=evidence_complete,
            consequential=kind in self._CONSEQUENTIAL_KINDS,
        )

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        text = cls._text(value).strip()
        return text or None

    @staticmethod
    def _as_int(value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    @classmethod
    def _key(cls, *parts: str) -> str:
        return "\x1f".join(parts)

    @classmethod
    def _field_text(cls, receipt: Mapping[str, object], *names: str) -> str | None:
        for name in names:
            value = cls._optional_text(receipt.get(name))
            if value:
                return value
        return None

    @classmethod
    def _file_key(
        cls,
        receipt: Mapping[str, object],
        kind: str,
        intent: str,
        anchor: str | None,
    ) -> str | None:
        explicit = cls._field_text(
            receipt,
            "file",
            "path",
            "file_path",
            "target_path",
            "target_file",
            "scope",
        )
        if explicit:
            return cls._clean_path(explicit)

        candidate = intent.strip().strip("`'\"")
        if kind == "test" and "::" in candidate:
            candidate = candidate.split("::", 1)[0]
        elif kind == "read":
            for prefix in cls._READ_PREFIXES:
                if candidate.lower().startswith(prefix):
                    candidate = candidate[len(prefix) :].strip()
                    break
            candidate = re.sub(r"@\d+(?:-\d+)?$", "", candidate)
            candidate = candidate.split()[0] if candidate.split() else ""
        elif kind == "edit":
            lowered = candidate.lower()
            for prefix in cls._EDIT_PREFIXES:
                if lowered.startswith(prefix):
                    candidate = candidate[len(prefix) :].strip()
                    break
            candidate = candidate.split()[0] if candidate.split() else ""
        if cls._looks_like_path(candidate):
            return cls._clean_path(candidate)

        if anchor and ":" in anchor:
            anchor_path = anchor.split(":", 1)[0]
            if cls._looks_like_path(anchor_path):
                return cls._clean_path(anchor_path)
        return None

    @classmethod
    def _target_key(
        cls,
        receipt: Mapping[str, object],
        file_key: str | None,
        kind: str,
        intent: str,
    ) -> str | None:
        explicit = cls._field_text(receipt, "target_key", "target")
        if explicit:
            return explicit
        if kind == "test":
            selector = cls._field_text(receipt, "selector", "test_selector")
            return selector or intent or None
        if kind == "grep":
            query = cls._field_text(receipt, "query") or intent
            return cls._key("grep", file_key or "*", query) if query else None
        return file_key

    @classmethod
    def _selector(
        cls, receipt: Mapping[str, object], kind: str, intent: str
    ) -> str | None:
        if kind != "test":
            return None
        return cls._field_text(receipt, "selector", "test_selector") or intent or None

    @classmethod
    def _is_scoped_selector(cls, selector: str | None) -> bool:
        if not selector:
            return False
        clean = selector.strip().lower()
        if clean in {"test", "tests", "bare", "bare-run", "agent-runner"}:
            return False
        if clean.startswith("agent-runner"):
            return False
        return "::" in clean or cls._looks_like_path(clean)

    @classmethod
    def _looks_like_path(cls, value: str) -> bool:
        if not value or any(char.isspace() for char in value):
            return False
        clean = value.strip().strip("`'\"").rstrip(":")
        if not clean:
            return False
        return (
            "/" in clean
            or clean.startswith(".")
            or bool(re.search(r"\.(?:py|pyi|js|ts|tsx|jsx|go|rs|java|json|yaml|yml|md)$", clean, re.I))
        )

    @staticmethod
    def _clean_path(value: str) -> str:
        clean = value.strip().strip("`'\"")
        return clean.lstrip("./") if clean.startswith("./") else clean

    @classmethod
    def _explicit_fact_keys(
        cls, receipt: Mapping[str, object], revision: int
    ) -> tuple[str, ...]:
        value = receipt.get("fact_delta")
        if value is None:
            return ()
        if isinstance(value, Mapping):
            items: list[object] = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
        else:
            items = [value]
        keys: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                identity = cls._field_text(
                    item, "identity_fingerprint", "fingerprint", "identity"
                )
                kind = cls._field_text(item, "kind") or "fact"
                item_revision = cls._as_int(item.get("workspace_revision"))
                rev = revision if item_revision is None else item_revision
                if identity:
                    keys.append(cls._key("explicit", kind, identity, str(rev)))
                else:
                    keys.append(cls._key("explicit", cls._stable(item), str(rev)))
            else:
                keys.append(cls._key("explicit", cls._text(item), str(revision)))
        return tuple(keys)

    @classmethod
    def _derived_fact_keys(
        cls,
        *,
        receipt: Mapping[str, object],
        kind: str,
        intent: str,
        file_key: str | None,
        selector: str | None,
        result: str,
        revision: int,
        anchor: str | None,
    ) -> tuple[str, ...]:
        keys: list[str] = []
        if kind == "read" and file_key:
            span = cls._field_text(receipt, "span", "range", "line")
            if not span:
                start = cls._as_int(receipt.get("start_line"))
                end = cls._as_int(receipt.get("end_line"))
                if start is not None:
                    span = str(start) if end is None else f"{start}-{end}"
            span = span or cls._read_span_from_intent(intent) or "full"
            keys.append(cls._key("read", file_key, span, str(revision)))
        elif kind == "grep" and intent:
            query = cls._field_text(receipt, "query") or intent
            scope = file_key or cls._field_text(receipt, "scope") or "*"
            keys.append(cls._key("grep", scope, query, str(revision)))
        elif kind == "test" and selector and result:
            keys.append(cls._key("test", selector, result, str(revision)))
        if anchor:
            keys.append(cls._key("anchor", anchor, str(revision)))
        return tuple(keys)

    @classmethod
    def _inspection_key(
        cls,
        kind: str,
        receipt: Mapping[str, object],
        intent: str,
        file_key: str | None,
        revision: int,
    ) -> str | None:
        if kind == "read" and file_key:
            span = cls._field_text(receipt, "span", "range", "line")
            span = span or cls._read_span_from_intent(intent) or "full"
            return cls._key("inspection-read", file_key, span, str(revision))
        if kind == "grep" and intent:
            query = cls._field_text(receipt, "query") or intent
            return cls._key("inspection-grep", file_key or "*", query, str(revision))
        return None

    @staticmethod
    def _read_span_from_intent(intent: str) -> str | None:
        match = re.search(r"@(?P<span>\d+(?:-\d+)?)$", intent.strip())
        return match.group("span") if match else None

    @classmethod
    def _stable(cls, value: object) -> str:
        if isinstance(value, Mapping):
            entries = sorted(
                (cls._text(key), cls._stable(item)) for key, item in value.items()
            )
            return "{" + ",".join(f"{key}={item}" for key, item in entries) + "}"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "[" + ",".join(cls._stable(item) for item in value) + "]"
        return cls._text(value)

    def _fact_delta_for(self, action: _Action) -> tuple[str, ...]:
        """Return compact identities used by shadow snapshots and recovery."""

        delta: list[str] = []
        if action.new_fact:
            if action.kind == "read":
                delta.append("NEW_READ_FACT")
            elif action.kind == "grep":
                delta.append("NEW_GREP_FACT")
            elif action.kind == "test":
                delta.append("NEW_TEST_RESULT")
            elif action.anchor:
                delta.append("NEW_FAILURE_ANCHOR")
            else:
                delta.append("NEW_FACT")
        return tuple(delta)

    def _current_turn(self) -> int | None:
        return self.history[-1].turn if self.history else None

    # ------------------------------------------------------------- vital state
    def _evaluate_vitals(self) -> None:
        self._evaluate_scope_skew()
        self._evaluate_oscillation()
        self._evaluate_reinspection()
        self._evaluate_reread_rate()
        self._evaluate_selector_stall()
        self._evaluate_edit_fact_gap()
        self._evaluate_file_dwell()
        self._evaluate_declaration_refusal()

        active = tuple(
            name
            for name in self.VITAL_NAMES
            if name in self.ADVISORY_VITALS
            and self.vitals[name].state == _Status.DEGRADED
        )
        self.active_vitals = active
        if active:
            self.state = _Status.DEGRADED
        elif any(card.state == _Status.NORMAL for card in self.vitals.values()):
            self.state = _Status.NORMAL
        else:
            self.state = _Status.UNKNOWN

    def _set(self, name: str, state: str, **metrics: object) -> None:
        card = self.vitals[name]
        card.state = state
        card.metrics = metrics

    def _evaluate_scope_skew(self) -> None:
        edits = [
            action
            for action in self.history
            if action.kind == "edit" and action.result_class == "APPLIED"
        ][-self.SCOPE_WINDOW :]
        if len(edits) < self.SCOPE_MIN_EDITS:
            self._set(
                "scope_skew",
                _Status.UNKNOWN,
                window_actions=len(edits),
                applied_edits=len(edits),
            )
            return
        classes = [self._edit_class(action) for action in edits]
        if any(kind is None for kind in classes):
            self._set("scope_skew", _Status.UNKNOWN, window_actions=len(edits))
            return
        test_edits = sum(kind == "test" for kind in classes)
        product_edits = sum(kind == "product" for kind in classes)
        degraded = (
            len(edits) >= self.SCOPE_MIN_EDITS
            and test_edits >= self.SCOPE_MIN_TEST_EDITS
            and product_edits <= self.SCOPE_MAX_PRODUCT_EDITS
        )
        self._set(
            "scope_skew",
            _Status.DEGRADED if degraded else _Status.NORMAL,
            window_actions=len(edits),
            applied_edits=len(edits),
            product_edits=product_edits,
            test_edits=test_edits,
        )

    def _evaluate_oscillation(self) -> None:
        actions = self.history[-self.ACTION_WINDOW :]
        edits = [
            action
            for action in actions
            if action.kind == "edit" and action.result_class == "APPLIED"
        ]
        if len(edits) < 3:
            self._set("edit_oscillation", _Status.UNKNOWN, comparable_edits=len(edits))
            return

        comparable = [
            action for action in edits if action.target_key and action.fingerprint
        ]
        if len(comparable) < 3:
            self._set(
                "edit_oscillation",
                _Status.UNKNOWN,
                comparable_edits=len(comparable),
            )
            return

        found = False
        span = None
        for first_index in range(len(edits) - 2):
            first, middle, last = edits[first_index : first_index + 3]
            if not first.target_key or not first.fingerprint:
                continue
            if first.target_key != middle.target_key or first.target_key != last.target_key:
                continue
            if not middle.fingerprint or middle.fingerprint == first.fingerprint:
                continue
            if last.fingerprint != first.fingerprint:
                continue
            span_value = last.index - first.index + 1
            if span_value > self.OSCILLATION_MAX_SPAN:
                continue
            between = [
                action
                for action in actions
                if first.index < action.index < last.index
            ]
            if any(action.new_fact for action in between):
                continue
            found = True
            span = span_value
            break

        self._set(
            "edit_oscillation",
            _Status.DEGRADED if found else _Status.NORMAL,
            span_actions=span,
            facts_between=0 if found else None,
        )

    def _evaluate_reinspection(self) -> None:
        actions = self.history[-self.ACTION_WINDOW :]
        inspections = [
            action
            for action in actions
            if action.kind in {"read", "grep"}
        ]
        if len(inspections) < self.MIN_INSPECTIONS:
            self._set(
                "reinspection_stall",
                _Status.UNKNOWN,
                inspect_actions=len(inspections),
                new_files=sum(action.new_file for action in inspections),
                new_spans_or_queries=sum(
                    action.inspection_fact_new for action in inspections
                ),
            )
            return
        if any(action.inspection_key is None for action in inspections):
            self._set(
                "reinspection_stall",
                _Status.UNKNOWN,
                inspect_actions=len(inspections),
                new_files=sum(action.new_file for action in inspections),
                new_spans_or_queries=None,
            )
            return
        new_inspection_facts = sum(
            action.inspection_fact_new for action in inspections
        )
        new_files = sum(action.new_file for action in inspections)
        degraded = new_inspection_facts == 0 and new_files == 0
        self._set(
            "reinspection_stall",
            _Status.DEGRADED if degraded else _Status.NORMAL,
            inspect_actions=len(inspections),
            new_files=new_files,
            new_spans_or_queries=new_inspection_facts,
        )

    def _evaluate_reread_rate(self) -> None:
        actions = self.history[-self.ACTION_WINDOW :]
        inspections = [action for action in actions if action.kind == "read"]
        if len(inspections) < self.MIN_INSPECTIONS:
            self._set(
                "reread_rate",
                _Status.UNKNOWN,
                read_actions=len(inspections),
                new_files=0,
                repeated_ratio=0.0,
            )
            return
        if any(not action.file_key for action in inspections):
            self._set("reread_rate", _Status.UNKNOWN, read_actions=len(inspections))
            return
        counts = Counter(action.target_key or action.file_key for action in inspections)
        repeated_actions = sum(
            count for count in counts.values() if count > 1
        )
        repeated_ratio = repeated_actions / len(inspections)
        new_files = sum(action.new_file for action in inspections)
        degraded = (
            new_files == 0
            and repeated_ratio >= self.REREAD_MIN_REPEAT_RATIO
            and all(not action.inspection_fact_new for action in inspections)
        )
        self._set(
            "reread_rate",
            _Status.DEGRADED if degraded else _Status.NORMAL,
            read_actions=len(inspections),
            new_files=new_files,
            repeated_ratio=round(repeated_ratio, 3),
        )

    def _evaluate_selector_stall(self) -> None:
        tests = [
            action
            for action in self.history[-self.ACTION_WINDOW :]
            if action.kind == "test" and action.scoped_test
        ]
        if len(tests) < self.MIN_SCOPED_TESTS:
            self._set(
                "selector_stall",
                _Status.UNKNOWN,
                scoped_tests=len(tests),
                new_selectors=sum(action.selector_new for action in tests),
                new_results=sum(action.test_result_new for action in tests),
            )
            return
        if any(not action.result_class for action in tests):
            self._set("selector_stall", _Status.UNKNOWN, scoped_tests=len(tests))
            return
        new_selectors = sum(action.selector_new for action in tests)
        new_results = sum(action.test_result_new for action in tests)
        degraded = new_selectors == 0 and new_results == 0
        self._set(
            "selector_stall",
            _Status.DEGRADED if degraded else _Status.NORMAL,
            scoped_tests=len(tests),
            new_selectors=new_selectors,
            new_results=new_results,
        )

    def _evaluate_edit_fact_gap(self) -> None:
        applied_edits = sum(
            action.kind == "edit" and action.result_class == "APPLIED"
            for action in self.history
        )
        if not applied_edits:
            self._set(
                "edit_fact_gap",
                _Status.UNKNOWN,
                applied_edits_since_fact=self._edits_since_fact,
            )
            return
        self._set(
            "edit_fact_gap",
            _Status.DEGRADED
            if self._edits_since_fact >= self.MIN_EDITS_SINCE_FACT
            else _Status.NORMAL,
            applied_edits_since_fact=self._edits_since_fact,
        )

    def _evaluate_file_dwell(self) -> None:
        if not self.history:
            self._set("file_dwell", _Status.UNKNOWN, consecutive_actions=0)
            return
        last_file = self.history[-1].file_key
        if not last_file:
            self._set("file_dwell", _Status.UNKNOWN, consecutive_actions=0)
            return
        run: list[_Action] = []
        for action in reversed(self.history):
            if action.file_key != last_file:
                break
            run.append(action)
        run.reverse()
        trailing = run[-self.FILE_DWELL_FACT_WINDOW :]
        new_facts = sum(action.new_fact for action in trailing)
        degraded = (
            len(run) >= self.FILE_DWELL_ACTIONS
            and len(trailing) == self.FILE_DWELL_FACT_WINDOW
            and new_facts == 0
        )
        self._set(
            "file_dwell",
            _Status.DEGRADED if degraded else _Status.NORMAL,
            target_key=last_file,
            consecutive_actions=len(run),
            new_facts_trailing_4=new_facts,
        )

    def _evaluate_declaration_refusal(self) -> None:
        if not self.receipts:
            self._set("declaration_refusal", _Status.UNKNOWN, streak=None)
            return
        declared = sum(
            bool(self._text(receipt.get("why")).strip())
            for receipt in self.receipts
        )
        coverage = declared / len(self.receipts)
        streak = 0
        for receipt in reversed(self.receipts):
            if self._text(receipt.get("why")).strip():
                break
            streak += 1
        if coverage < self.DECLARATION_MIN_COVERAGE:
            self._set(
                "declaration_refusal",
                _Status.UNKNOWN,
                coverage=round(coverage, 3),
                declared=declared,
                receipts=len(self.receipts),
                streak=None,
            )
            return
        self._set(
            "declaration_refusal",
            _Status.DEGRADED
            if streak >= self.DECLARATION_REFUSAL_STREAK
            else _Status.NORMAL,
            coverage=round(coverage, 3),
            declared=declared,
            receipts=len(self.receipts),
            streak=streak,
        )

    @classmethod
    def _edit_class(cls, action: _Action) -> str | None:
        if action.file_key is None:
            return None
        return "test" if cls._TEST_PATH.search(action.file_key) else "product"

    # -------------------------------------------------------------- escalation
    def _advisory_eligible(self) -> bool:
        if self._terminal or not self._task_open or self.coverage != "COMPLETE":
            return False
        if len(self.history) < self.ACTION_WINDOW:
            return False
        window = self.history[-self.ACTION_WINDOW :]
        return (
            all(action.evidence_complete for action in window)
            and not any(action.new_fact for action in window)
            and bool(self.active_vitals)
            and not self._advisory_fired
        )

    def _advance_escalation(self, action: _Action) -> None:
        if not self._advisory_fired or self._terminal:
            return
        if (
            not self._post_advisory_actions
            and self._advisory_action_index is not None
            and action.index <= self._advisory_action_index
        ):
            # The action that preceded the advisory is not post-advisory.  Use
            # receipt order here so multiple receipts may safely share a turn.
            return
        if self._post_advisory_window_complete:
            return
        self._post_advisory_actions.append(action)
        if len(self._post_advisory_actions) < self.ESCALATION_WINDOW:
            return
        self._post_advisory_window_complete = True
        no_fact = sum(not item.new_fact for item in self._post_advisory_actions)
        severity = len(self.active_vitals)
        if (
            self._task_open
            and severity > 0
            and severity >= self._advisory_severity
            and no_fact >= self.ESCALATION_MIN_NO_FACT
        ):
            self.recommend_surgery = True
            self._escalation_state = "SURGERY_RECOMMENDED"
            self._record_event(
                "EKG_RECOMMEND_SURGERY",
                turn=self._post_advisory_actions[-1].turn,
                detail=(
                    f"no_fact_actions={no_fact}/{self.ESCALATION_WINDOW} "
                    f"severity={severity}"
                ),
            )

    def _maybe_stall(self, turn: int | None) -> None:
        if (
            self._terminal
            or not self._task_open
            or not self.recommend_surgery
            or not self._surgery_done
            or not self._get_sure_used
            or not self.active_vitals
        ):
            return
        self._terminal = True
        self.terminal_event = "EKG_STALLED"
        self._escalation_state = "STALLED"
        self._record_event(
            "EKG_STALLED",
            turn=turn if turn is not None else self._current_turn(),
            detail="advisory+surgery+one get_sure did not recover trajectory",
        )

    def _reset_episode(self, turn: int | None, reason: str) -> None:
        had_escalation = self._advisory_fired or self.recommend_surgery
        self._advisory_fired = False
        self._advisory_severity = 0
        self._advisory_action_index = None
        self._post_advisory_actions = []
        self._post_advisory_window_complete = False
        self._surgery_done = False
        self._get_sure_used = False
        self.recommend_surgery = False
        self._escalation_state = "NORMAL"
        if had_escalation:
            self._record_event("EKG_RECOVERED", turn=turn, detail=reason)

    def _record_event(self, event: str, *, turn: int | None, detail: str) -> None:
        self.trace.append(
            {
                "cylinder": "vitals",
                "event": event,
                "detail": detail,
                "turn": turn,
            }
        )


def _vitals_trigger(evidence: Evidence) -> TriState:
    """Replay the receipt prefix and ask the EKG whether an advisory is eligible."""

    record = evidence.event
    if evidence.source == "trace":
        event = str(record.get("event") or "").strip().lower()
        if not event:
            return TriState.UNKNOWN
        return TriState.YES if event == "ekg_trigger" else TriState.NO
    if evidence.source != "receipt":
        return TriState.NO

    prefix: list[Mapping[str, object]] = []
    found = False
    for item in evidence.receipts:
        prefix.append(item)
        if item is record or item == record:
            found = True
            break
    if not found:
        return TriState.UNKNOWN

    monitor = VitalsEKG(task_id=evidence.run_id)
    for item in prefix:
        monitor.observe(item)
    if monitor.coverage != "COMPLETE":
        return TriState.UNKNOWN
    return TriState.YES if monitor._advisory_eligible() else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="vitals",
    trigger=_vitals_trigger,
    fire_events=frozenset({"EKG_ADVISORY"}),
    suppression_events=frozenset({"EKG_SUPPRESSED"}),
    dedup_events=frozenset({"EKG_DEDUPED"}),
    challenge_once_events=frozenset({"EKG_CHALLENGE_ONCE"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = ["CENSUS_SPEC", "CENSUS_SPECS", "VitalState", "VitalsEKG"]
