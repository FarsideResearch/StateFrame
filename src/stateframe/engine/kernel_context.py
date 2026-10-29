"""Pure context plans: protected exchanges and dependency-closed frame selection."""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from enum import Enum

from kernel_effect_boundary import AXES, Mode


def _mode(name: str, mode: Mode | None) -> Mode:
    from kernel_wave3 import resolve_d_mode
    return mode if mode is not None else resolve_d_mode(name)


def token_estimate(value: object) -> int:
    """Budget estimate only; never report this as measured model prefill."""
    return (len(json.dumps(value, sort_keys=True, ensure_ascii=False)) + 3) // 4


@dataclasses.dataclass(frozen=True)
class Exchange:
    messages: tuple[Mapping[str, object], ...]

    @property
    def token_estimate(self) -> int:
        return sum(token_estimate(message) for message in self.messages)


def group_into_exchanges(messages: Sequence[Mapping[str, object]], *,
                         protected_prefix_count: int | None = None) -> tuple[
                             tuple[Mapping[str, object], ...], tuple[Exchange, ...]]:
    if protected_prefix_count is None:
        protected_prefix_count = 0
        for message in messages:
            if message.get("role") not in {"system", "user", "developer"}:
                break
            protected_prefix_count += 1
    if not 0 <= protected_prefix_count <= len(messages):
        raise ValueError("protected prefix is outside the message list")
    prefix = tuple(messages[:protected_prefix_count])
    exchanges: list[Exchange] = []
    group: list[Mapping[str, object]] = []
    for message in messages[protected_prefix_count:]:
        if message.get("role") == "assistant" and group:
            exchanges.append(Exchange(tuple(group)))
            group = []
        group.append(message)
    if group:
        exchanges.append(Exchange(tuple(group)))
    return prefix, tuple(exchanges)


@dataclasses.dataclass(frozen=True)
class HistoryOverflow:
    required_tokens: int
    available_tokens: int
    reason: str = "newest_complete_exchange_exceeds_cap"


@dataclasses.dataclass(frozen=True)
class HistoryCapResult:
    messages: tuple[Mapping[str, object], ...]
    dropped_exchanges: int
    kept_exchanges: int
    total_tokens: int  # exchange budget only; protected_prefix_tokens is separate
    applied: bool
    proposed_dropped_exchanges: int = 0
    proposed_messages: tuple[Mapping[str, object], ...] = ()
    overflow: HistoryOverflow | None = None
    protected_prefix_tokens: int = 0


DEFAULT_HISTORY_CAP_TOKENS = 2048


def cap_history(messages: Sequence[Mapping[str, object]], *,
                cap_tokens: int = DEFAULT_HISTORY_CAP_TOKENS,
                mode: Mode | None = None,
                protected_prefix_count: int | None = None) -> HistoryCapResult:
    evaluate, _deliver, apply = AXES[_mode("history_cap", mode)]
    prefix, exchanges = group_into_exchanges(
        messages, protected_prefix_count=protected_prefix_count)
    prefix_tokens = sum(token_estimate(message) for message in prefix)
    total = sum(exchange.token_estimate for exchange in exchanges)
    if not evaluate:
        return HistoryCapResult(tuple(messages), 0, len(exchanges), total, False,
                                protected_prefix_tokens=prefix_tokens)
    if cap_tokens < 0:
        raise ValueError("history cap must be nonnegative")
    assert_no_orphaned_tool_result(messages)
    kept: list[Exchange] = []
    running = 0
    overflow = None
    for exchange in reversed(exchanges):
        cost = exchange.token_estimate
        if running + cost > cap_tokens:
            if not kept:
                overflow = HistoryOverflow(cost, cap_tokens)
            break
        kept.append(exchange)
        running += cost
    kept.reverse()
    proposed = prefix + tuple(message for exchange in kept for message in exchange.messages)
    dropped = len(exchanges) - len(kept)
    return HistoryCapResult(
        proposed if apply else tuple(messages), dropped if apply else 0,
        len(kept) if apply else len(exchanges), running if apply else total, apply,
        proposed_dropped_exchanges=dropped, proposed_messages=proposed,
        overflow=overflow, protected_prefix_tokens=prefix_tokens)


def assert_no_orphaned_tool_result(messages: Sequence[Mapping[str, object]]) -> None:
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise AssertionError("orphaned or duplicate tool result ID")
            pending.remove(call_id)
            continue
        if pending:
            raise AssertionError("tool calls missing results before next message")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, (list, tuple)):
                raise AssertionError("tool_calls must be a sequence")
            for call in calls:
                call_id = call.get("id") if isinstance(call, Mapping) else None
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise AssertionError("missing or duplicate tool call ID")
                pending.add(call_id)
    if pending:
        raise AssertionError("tool calls missing results at exchange end")


FRAME_TARGET_TOKENS = 4096
FRAME_CEILING_TOKENS = 8192
SELECTION_POLICY_VERSION = "wave4-atomic-selection/1"


class SelectionPriority(int, Enum):
    REQUIRED_CONTROL_CONTEXT = 1
    IMMEDIATE_WORKING_SET = 2
    DECISION_LINKED_EVIDENCE = 3
    OPTIONAL_HISTORY = 4


@dataclasses.dataclass(frozen=True)
class SelectableItem:
    item_id: str
    priority: SelectionPriority
    group: str
    text: str
    token_estimate: int
    retrievable_ref: str = ""
    dependencies: tuple[str, ...] = ()
    action_required: bool = False
    revision: str = ""
    omission_reason: str = ""
    omission_score: float | None = None


@dataclasses.dataclass(frozen=True)
class RenderOverflow:
    required_group: str
    available_tokens: int
    required_tokens: int
    retrievable_ref: str
    reason: str = "required_group_exceeds_ceiling"


@dataclasses.dataclass(frozen=True)
class ViewPlan:
    selected: tuple[SelectableItem, ...]
    omitted: tuple[SelectableItem, ...]
    used_tokens: int
    budget_tokens: int
    policy_version: str
    applied: bool = True
    margin_reasons: tuple[str, ...] = ()
    proposal: ViewPlan | RenderOverflow | None = None


def _select(items: Sequence[SelectableItem], target: int, ceiling: int) -> ViewPlan | RenderOverflow:
    if not 0 < target <= ceiling:
        raise ValueError("selection requires 0 < target <= ceiling")
    if any(item.token_estimate < 0 for item in items):
        raise ValueError("negative token estimate")
    unique_items: list[SelectableItem] = []
    duplicate_items: list[SelectableItem] = []
    seen_ids: set[str] = set()
    for item in items:
        if item.item_id in seen_ids:
            duplicate_items.append(dataclasses.replace(
                item, omission_reason="duplicate",
                omission_score=None))
            continue
        seen_ids.add(item.item_id)
        unique_items.append(item)
    by_id = {item.item_id: item for item in unique_items}
    groups: dict[str, list[SelectableItem]] = {}
    for item in unique_items:
        groups.setdefault(item.group or item.item_id, []).append(item)

    def closure(group: str) -> set[str]:
        included: set[str] = set()
        pending = [group]
        while pending:
            key = pending.pop()
            for member in groups[key]:
                if member.item_id in included:
                    continue
                included.add(member.item_id)
                for dependency in member.dependencies:
                    if dependency not in by_id:
                        raise KeyError(dependency)
                    dep = by_id[dependency]
                    pending.append(dep.group or dep.item_id)
        return included

    required_groups = [key for key, group in groups.items() if any(
        item.priority == SelectionPriority.REQUIRED_CONTROL_CONTEXT or item.action_required
        for item in group)]
    selected: set[str] = set()
    for key in required_groups:
        try:
            selected.update(closure(key))
        except KeyError as exc:
            return RenderOverflow(key, ceiling, 0, str(exc.args[0]), "required_dependency_missing")
    used = sum(by_id[key].token_estimate for key in selected)
    if used > ceiling:
        key = required_groups[-1]
        return RenderOverflow(key, ceiling, used, groups[key][0].retrievable_ref)
    margin = ["required_dependency_closure"] if used > target else []
    ordered = sorted(groups, key=lambda key: min(item.priority for item in groups[key]))
    for key in ordered:
        if all(item.item_id in selected for item in groups[key]):
            continue
        try:
            additions = closure(key) - selected
        except KeyError:
            continue  # optional group cannot be rendered without its dependencies
        cost = sum(by_id[item_id].token_estimate for item_id in additions)
        immediate = any(item.priority == SelectionPriority.IMMEDIATE_WORKING_SET
                        for item in groups[key])
        limit = ceiling if immediate else target
        if used + cost <= limit:
            selected.update(additions)
            used += cost
            if used > target:
                margin.append("immediate_atomic_group:" + key)
    # Stable priority ordering, preserving input order within a tier.
    ordered_items = sorted(unique_items, key=lambda item: item.priority)
    omitted_items: list[SelectableItem] = list(duplicate_items)
    for item in unique_items:
        if item.item_id in selected:
            continue
        omitted_items.append(dataclasses.replace(
            item,
            omission_reason="budget",
            omission_score=None,
        ))
    return ViewPlan(tuple(item for item in ordered_items if item.item_id in selected),
                    tuple(omitted_items),
                    used, target, SELECTION_POLICY_VERSION, margin_reasons=tuple(margin))


def select_view(items: Sequence[SelectableItem], *, budget_tokens: int = FRAME_TARGET_TOKENS,
                ceiling_tokens: int = FRAME_CEILING_TOKENS,
                mode: Mode | None = None) -> ViewPlan | RenderOverflow:
    evaluate, _deliver, apply = AXES[_mode("render_selection", mode)]
    proposal = _select(items, budget_tokens, ceiling_tokens) if evaluate else None
    if apply:
        assert proposal is not None
        return proposal
    return ViewPlan(tuple(items), (), sum(item.token_estimate for item in items),
                    budget_tokens, SELECTION_POLICY_VERSION, applied=False, proposal=proposal)
