"""Mutation grants and durable per-run resource ceilings."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Mapping
from types import MappingProxyType

Level = Literal["off", "prebuilt", "authored"]


@dataclass(frozen=True)
class MutationPolicy:
    level: Level = "off"
    network: Literal["none", "public-web"] = "none"
    domains: tuple[str, ...] = ()
    write_globs: tuple[str, ...] = ()
    read_globs: tuple[str, ...] = ()
    retain: Literal["run", "project", "user"] = "run"
    audit: Literal["source", "digest"] = "source"

    def __post_init__(self):
        if (self.level not in ("off", "prebuilt", "authored") or self.network not in ("none", "public-web")
                or self.retain not in ("run", "project", "user") or self.audit not in ("source", "digest")):
            raise ValueError("unknown mutation policy value")
        for name in ("domains", "write_globs", "read_globs"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"invalid {name}")
        if self.level == "off" and (self.network != "none" or self.write_globs):
            raise ValueError("grants require an enabled mutation level")

    @property
    def enabled(self) -> bool:
        return self.level != "off"


def policy_from(options: Mapping) -> MutationPolicy:
    allowed = {"level", "network", "domains", "write_globs", "read_globs", "retain", "audit", "shell_network", "hooks"}
    if options.keys() - allowed:
        raise ValueError("unknown mutation policy option")
    if "shell_network" in options and options["shell_network"] not in ("none", "loopback", "full"):
        raise ValueError("unknown shell network level")
    values = {name: options[name] for name in MutationPolicy.__dataclass_fields__ if name in options}
    for name in ("domains", "write_globs", "read_globs"):
        if name in values:
            if not isinstance(values[name], (tuple, list)) or any(not isinstance(item, str) or not item for item in values[name]):
                raise ValueError(f"invalid {name}")
            values[name] = tuple(values[name])
    result = MutationPolicy(**values)
    if result.level not in ("off", "prebuilt", "authored") or result.network not in ("none", "public-web"):
        raise ValueError("unknown mutation level or network grant")
    if result.retain not in ("run", "project", "user") or result.audit not in ("source", "digest"):
        raise ValueError("unknown mutation retention or audit setting")
    if result.level == "off" and (result.network != "none" or result.write_globs):
        raise ValueError("grants require an enabled mutation level")
    return result


DEFAULTS = {
    "authoring_episodes": 3, "revisions_per_unit": 3, "revisions_total": 8,
    "sends_per_episode": 4, "sends_total": 10, "invocations": 40,
    "payload_cpu_s": 60, "payload_wall_s": 120, "authored_source_bytes": 256 << 10,
    "output_bytes": 64 << 10, "broker_fetches": 10, "broker_bytes": 8 << 20,
    "broker_wall_s": 120, "broker_cpu_s": 60,
    "edit_files": 16, "edit_bytes": 256 << 10, "journal_bytes": 2 << 20,
}
CEILINGS = {
    **DEFAULTS, "authoring_episodes": 6, "revisions_per_unit": 5,
    "revisions_total": 16, "sends_total": 20, "invocations": 200,
    "payload_cpu_s": 300, "payload_wall_s": 600, "authored_source_bytes": 1 << 20,
    "output_bytes": 256 << 10, "broker_fetches": 50, "broker_bytes": 32 << 20,
    "broker_wall_s": 600, "broker_cpu_s": 300,
    "edit_files": 64, "edit_bytes": 1 << 20, "journal_bytes": 8 << 20,
}
SCOPED = frozenset(("revisions_per_unit", "sends_per_episode"))


@dataclass
class BudgetState:
    limits: dict[str, int | float]
    events: list[dict] = field(default_factory=list)
    disabled: bool = False

    def remaining(self, *, scope: str | None = None) -> dict[str, float]:
        used = {name: 0.0 for name in self.limits}
        reserves = {}
        for event in self.events:
            if event.get("kind") == "BUDGET_RESERVE":
                name = event.get("name")
                if name in used:
                    amount = event.get("amount", 0)
                    relevant = name not in SCOPED or event.get("scope") == scope
                    if relevant:
                        used[name] += amount
                    if event.get("id") is not None:
                        reserves[event["id"]] = (name, amount, relevant)
            elif event.get("kind") == "BUDGET_ADMISSION":
                for name, amount in event["charges"].items():
                    if name not in SCOPED or event["scope"] == scope:
                        used[name] += amount
            elif event.get("kind") == "BUDGET_SETTLE" and event.get("id") in reserves:
                name, amount, relevant = reserves.pop(event["id"])
                if relevant:
                    used[name] += event.get("amount", amount) - amount
        return {name: max(0, limit - used[name]) for name, limit in self.limits.items()}

    def reserve(self, kind: str, amount: float, *, id: str | None = None, scope: str | None = None) -> bool:
        if kind not in self.limits or type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid budget reservation")
        if kind in SCOPED and (not isinstance(scope, str) or not scope):
            raise ValueError("scoped budget requires a unit or episode id")
        if id is not None and (not isinstance(id, str) or not id):
            raise ValueError("invalid reservation id")
        if self.disabled or amount > self.remaining(scope=scope)[kind]:
            self.disabled = True
            self.events.append({"kind": "BUDGET_EXHAUSTED", "name": kind, "scope": scope})
            return False
        event = {"kind": "BUDGET_RESERVE", "name": kind, "amount": amount}
        if kind in SCOPED:
            event["scope"] = scope
        if id is not None:
            if any(row.get("id") == id and row.get("kind") == "BUDGET_RESERVE" for row in self.events):
                raise ValueError("duplicate budget reservation id")
            event["id"] = id
        self.events.append(event)
        if self.remaining(scope=scope)[kind] <= 0:
            self.disabled = True
            self.events.append({"kind": "BUDGET_EXHAUSTED", "name": kind, "scope": scope})
        return True

    def settle(self, id: str, measured: float) -> None:
        if type(measured) not in (int, float) or not math.isfinite(measured) or measured < 0:
            raise ValueError("invalid measurement")
        if not any(row.get("id") == id and row.get("kind") == "BUDGET_RESERVE" for row in self.events):
            raise ValueError("unknown reservation")
        if any(row.get("id") == id and row.get("kind") == "BUDGET_SETTLE" for row in self.events):
            raise ValueError("duplicate settlement")
        self.events.append({"kind": "BUDGET_SETTLE", "id": id, "amount": measured})
        reservation = next(row for row in self.events if row.get("id") == id and row.get("kind") == "BUDGET_RESERVE")
        if self.remaining(scope=reservation.get("scope"))[reservation["name"]] <= 0:
            self.disabled = True
            self.events.append({"kind": "BUDGET_EXHAUSTED", "name": "settlement"})


@dataclass(frozen=True)
class Budget:
    limits: Mapping[str, int | float] = field(default_factory=lambda: dict(DEFAULTS))

    def __post_init__(self):
        if set(self.limits) != set(DEFAULTS) or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > CEILINGS[name]
                                                    for name, value in self.limits.items()):
            raise ValueError("invalid budget limit")
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))

    def fold(self, journal_events) -> BudgetState:
        events = list(journal_events)
        reserves = set()
        admissions = set()
        settled = set()
        for row in events:
            if not isinstance(row, dict):
                raise ValueError("invalid budget event")
            kind = row.get("kind")
            if kind == "BUDGET_RESERVE":
                name, amount, token = row.get("name"), row.get("amount"), row.get("id")
                if (name not in self.limits or type(amount) not in (int, float) or not math.isfinite(amount)
                        or amount < 0 or (name in SCOPED and (not isinstance(row.get("scope"), str) or not row["scope"]))
                        or (token is not None and (not isinstance(token, str) or not token or token in reserves))):
                    raise ValueError("invalid budget reservation event")
                if token is not None:
                    reserves.add(token)
            elif kind == "BUDGET_ADMISSION":
                token, scope, charges = row.get("id"), row.get("scope"), row.get("charges")
                if (not isinstance(token, str) or not token or token in admissions
                        or not isinstance(scope, str) or not scope
                        or not isinstance(charges, dict)
                        or set(charges) != {"authored_source_bytes", "revisions_total", "revisions_per_unit"}
                        or any(type(amount) is not int or amount < 0 or amount > self.limits[name]
                               for name, amount in charges.items())
                        or charges["revisions_total"] != 1 or charges["revisions_per_unit"] != 1):
                    raise ValueError("invalid revision admission event")
                admissions.add(token)
            elif kind == "BUDGET_SETTLE":
                token, amount = row.get("id"), row.get("amount")
                if (token not in reserves or token in settled or type(amount) not in (int, float)
                        or not math.isfinite(amount) or amount < 0):
                    raise ValueError("invalid budget settlement event")
                settled.add(token)
            elif kind == "BUDGET_EXHAUSTED":
                name, scope = row.get("name"), row.get("scope")
                if (name not in self.limits and name != "settlement") or (scope is not None and
                    (not isinstance(scope, str) or not scope)) or (name in SCOPED and scope is None):
                    raise ValueError("invalid budget exhaustion event")
        state = BudgetState(dict(self.limits), events)
        state.disabled = any(row.get("kind") == "BUDGET_EXHAUSTED" for row in events)
        if any(value <= 0 for name, value in state.remaining().items() if name not in SCOPED):
            state.disabled = True
        for scope in {row.get("scope") for row in events if row.get("name") in SCOPED} - {None}:
            if any(state.remaining(scope=scope)[name] <= 0 for name in SCOPED):
                state.disabled = True
        return state
