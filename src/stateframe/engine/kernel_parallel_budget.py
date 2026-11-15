"""Durable, run-wide physical-send allowance for parent and subframes."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

if __package__:
    from .kernel_parallel_records import UnitOutcome
else:
    from kernel_parallel_records import UnitOutcome


class BudgetError(ValueError):
    """The parent allowance or its durable accounting is inconsistent."""


class BudgetStore:
    """One parent engine owns the state; worker ledgers supply physical spend."""

    _SCHEMA = "stateframe-parallel-budget/1"
    _KEYS = {"schema", "limit", "parent_steps_spent", "subframe_sends_authorized",
             "held_reservations", "parent_reserve", "free", "calls"}

    def __init__(self, run: Path, *, limit: int | None, parent_reserve: int):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise BudgetError("parent max_turns must be a positive integer or unlimited")
        if type(parent_reserve) is not int or parent_reserve < 0:
            raise BudgetError("subframe parent reserve must be a nonnegative integer")
        self.run = Path(run)
        self.path = self.run / "parallel" / "budget.json"
        self.limit = limit
        self.parent_reserve = parent_reserve

    def _initial(self) -> dict:
        return {"schema": self._SCHEMA, "limit": self.limit, "parent_steps_spent": 0,
                "subframe_sends_authorized": 0, "held_reservations": 0,
                "parent_reserve": self.parent_reserve,
                "free": None if self.limit is None else self.limit - self.parent_reserve,
                "calls": {}}

    def _load(self) -> dict:
        if self.path.is_symlink():
            raise BudgetError("subframe budget path is a symlink")
        if not self.path.exists():
            state = self._initial()
            if self.path.parent.is_dir():
                for call_dir in self.path.parent.iterdir():
                    intent_path = call_dir / "intent.json"
                    if not call_dir.is_dir() or not intent_path.exists():
                        continue
                    if call_dir.is_symlink() or intent_path.is_symlink():
                        raise BudgetError(f"subframe intent path is a symlink: {intent_path}")
                    try:
                        intent = json.loads(intent_path.read_bytes())
                    except (OSError, ValueError) as error:
                        raise BudgetError(f"subframe intent is unreadable: {intent_path}") from error
                    if (not isinstance(intent, dict) or
                            intent.get("schema") != "stateframe-parallel-intent/1" or
                            intent.get("call_id") != call_dir.name or
                            not isinstance(intent.get("admission"), dict)):
                        raise BudgetError(f"subframe intent is invalid: {intent_path}")
                    if intent["admission"].get("status") != "ADMITTED":
                        continue
                    reservations = intent.get("reservations")
                    if (not isinstance(reservations, dict) or
                            any(not isinstance(unit, str) or type(cap) is not int or cap < 1
                                for unit, cap in reservations.items())):
                        raise BudgetError(f"subframe intent reservations are invalid: {intent_path}")
                    state["calls"][call_dir.name] = {"reservations": reservations,
                                                      "terminal": []}
            return state
        try:
            state = json.loads(self.path.read_bytes())
        except (OSError, ValueError) as error:
            raise BudgetError(f"subframe budget state is unreadable: {error}") from error
        if (not isinstance(state, dict) or set(state) != self._KEYS or
                state["schema"] != self._SCHEMA or type(state["limit"]) is not type(self.limit) or
                state["limit"] != self.limit or
                type(state["parent_reserve"]) is not int or
                state["parent_reserve"] != self.parent_reserve or
                not isinstance(state["calls"], dict)):
            raise BudgetError("subframe budget state has a foreign or invalid schema")
        for key in ("parent_steps_spent", "subframe_sends_authorized", "held_reservations"):
            if type(state[key]) is not int or state[key] < 0:
                raise BudgetError(f"subframe budget {key} is invalid")
        expected_free = (None if self.limit is None else self.limit - self.parent_reserve -
                         state["parent_steps_spent"] - state["subframe_sends_authorized"] -
                         state["held_reservations"])
        if state["free"] != expected_free or (expected_free is not None and expected_free < 0):
            raise BudgetError("subframe budget counters violate the parent allowance invariant")
        for call_id, call in state["calls"].items():
            if (not isinstance(call_id, str) or
                    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", call_id) is None or
                    not isinstance(call, dict) or
                    set(call) != {"reservations", "terminal"} or
                    not isinstance(call["reservations"], dict) or
                    not isinstance(call["terminal"], list)):
                raise BudgetError("subframe budget call reservation is invalid")
            units = call["reservations"]
            if (any(not isinstance(unit, str) or
                    re.fullmatch(r"[a-z][a-z0-9_]{0,23}", unit) is None or
                    type(cap) is not int or cap < 1
                    for unit, cap in units.items()) or
                    any(type(unit) is not str for unit in call["terminal"]) or
                    len(call["terminal"]) != len(set(call["terminal"])) or
                    any(unit not in units for unit in call["terminal"])):
                raise BudgetError("subframe budget call units or caps are invalid")
        return state

    def _authorized(self, call_id: str, unit_id: str) -> int:
        path = self.run / "parallel" / call_id / "units" / unit_id / "sends-ledger.jsonl"
        if path.is_symlink():
            raise BudgetError(f"subframe send ledger is a symlink: {path}")
        if not path.exists():
            return 0
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BudgetError(f"subframe send ledger is not a private regular file: {path}")
            fcntl.flock(fd, fcntl.LOCK_SH)
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read()
        finally:
            os.close(fd)
        if data and not data.endswith(b"\n"):
            raise BudgetError(f"subframe send ledger has an incomplete record: {path}")
        authorized: set[str] = set()
        settled: set[str] = set()
        for number, line in enumerate(data.splitlines(), 1):
            try:
                item = json.loads(line)
            except ValueError as error:
                raise BudgetError(f"subframe send ledger line {number} is invalid: {path}") from error
            if not isinstance(item, dict) or not isinstance(item.get("send_id"), str):
                raise BudgetError(f"subframe send ledger line {number} lacks send identity: {path}")
            send_id = item["send_id"]
            state = item.get("state")
            if state == "AUTHORIZED" and send_id not in authorized:
                authorized.add(send_id)
            elif state == "SETTLED" and send_id in authorized and send_id not in settled:
                settled.add(send_id)
            else:
                raise BudgetError(f"subframe send ledger line {number} has an invalid transition: {path}")
        return len(authorized)

    def _refresh(self, state: dict, parent_steps_spent: int) -> dict:
        if type(parent_steps_spent) is not int or parent_steps_spent < 0:
            raise BudgetError("parent step count is invalid")
        state["parent_steps_spent"] = max(state["parent_steps_spent"], parent_steps_spent)
        spent = held = 0
        for call_id, call in state["calls"].items():
            for unit_id, cap in call["reservations"].items():
                count = self._authorized(call_id, unit_id)
                if count > cap:
                    raise BudgetError(f"subframe {unit_id} authorized {count} sends beyond cap {cap}")
                spent += count
                if unit_id not in call["terminal"]:
                    held += cap - count
        state["subframe_sends_authorized"] = spent
        state["held_reservations"] = held
        state["free"] = (None if self.limit is None else self.limit -
                         state["parent_steps_spent"] - spent - held - self.parent_reserve)
        if state["free"] is not None and state["free"] < 0:
            raise BudgetError("subframe budget exceeds the parent request allowance")
        return state

    def _save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise BudgetError("subframe budget path is a symlink")
        data = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=".budget-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def can_reserve(self, reservations: Mapping[str, int], *, parent_steps_spent: int) -> bool:
        if self.limit is not None and self.limit < self.parent_reserve + parent_steps_spent:
            return False
        state = self._refresh(self._load(), parent_steps_spent)
        self._save(state)
        return state["free"] is None or sum(reservations.values()) <= state["free"]

    def reserve(self, call_id: str, reservations: Mapping[str, int], *, parent_steps_spent: int) -> bool:
        if self.limit is not None and self.limit < self.parent_reserve + parent_steps_spent:
            return False
        state = self._refresh(self._load(), parent_steps_spent)
        existing = state["calls"].get(call_id)
        if existing is not None:
            if existing["reservations"] != dict(reservations):
                raise BudgetError(f"subframe call {call_id} has different durable reservations")
            self._save(state)
            return True
        required = sum(reservations.values())
        if state["free"] is not None and required > state["free"]:
            self._save(state)
            return False
        state["calls"][call_id] = {"reservations": dict(reservations), "terminal": []}
        self._save(self._refresh(state, parent_steps_spent))
        return True

    def terminal(self, call_id: str, unit_id: str, *, parent_steps_spent: int) -> None:
        state = self._load()
        call = state["calls"].get(call_id)
        if call is None or unit_id not in call["reservations"]:
            raise BudgetError(f"subframe {unit_id} has no durable parent reservation")
        if unit_id not in call["terminal"]:
            call["terminal"].append(unit_id)
            call["terminal"].sort()
        self._save(self._refresh(state, parent_steps_spent))

    def recover(self, *, parent_steps_spent: int) -> None:
        state = self._load()
        for call_id, call in state["calls"].items():
            outcomes = self.run / "parallel" / call_id / "outcomes.jsonl"
            if outcomes.is_symlink():
                raise BudgetError(f"subframe outcomes log is a symlink: {outcomes}")
            if outcomes.is_file():
                data = outcomes.read_bytes()
                if data and not data.endswith(b"\n"):
                    raise BudgetError(f"subframe outcomes log has an incomplete record: {outcomes}")
                seen: set[str] = set()
                for line in data.splitlines():
                    try:
                        item = UnitOutcome.from_json(json.loads(line))
                    except (ValueError, TypeError) as error:
                        raise BudgetError(f"subframe outcomes log is invalid: {outcomes}") from error
                    unit_id = item.unit_id
                    if unit_id not in call["reservations"] or unit_id in seen:
                        raise BudgetError(f"subframe outcomes log names an unreserved subframe: {outcomes}")
                    seen.add(unit_id)
                    if unit_id not in call["terminal"]:
                        call["terminal"].append(unit_id)
                call["terminal"].sort()
                if set(call["terminal"]) != seen:
                    raise BudgetError(f"subframe budget terminal state lacks a matching outcome: {outcomes}")
            elif call["terminal"]:
                raise BudgetError(f"subframe budget terminal state lacks an outcomes log: {outcomes}")
        self._save(self._refresh(state, parent_steps_spent))
