"""Thin, switchable factual hooks for the plain-loop engine. All default OFF.

STATEFRAME_PLAIN_RESTATE=on        exact kernel_restate_requirement.MESSAGE as a user message
                                   right after the system prefix, only while zero product
                                   edits exist and at least two product files are declared.
STATEFRAME_PLAIN_WIDE_WINDOW=on    a ranged read of a declared Python file that cuts through a
                                   definition is widened to the complete enclosing
                                   definition(s) plus up to 50 context lines, within the
                                   tool-result cap; never narrower than what was asked.
STATEFRAME_PLAIN_SIBLING_SITES=on  after an applied product str_replace, a bounded factual list
                                   of other call/construction sites (in declared files, current
                                   source) of the callables named in the replaced text.

Hooks receive plain data only (never a runtime). Any hook failure leaves the base
request/result unchanged and is journaled as ``hook_error``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ast
import keyword
import os
import re

from kernel_restate_requirement import MESSAGE as RESTATE_MESSAGE

SWITCHES = {"restate": "STATEFRAME_PLAIN_RESTATE", "wide_window": "STATEFRAME_PLAIN_WIDE_WINDOW",
            "sibling_sites": "STATEFRAME_PLAIN_SIBLING_SITES"}
CONTEXT_LINES = 50          # total extra lines around a widened definition (split before/after)
MAX_SITES = 20
MAX_CALLABLES = 5
_NOT_SITES = frozenset({
    "print", "len", "str", "int", "float", "bool", "dict", "list", "set", "tuple", "isinstance",
    "getattr", "setattr", "hasattr", "range", "enumerate", "zip", "min", "max", "sorted", "super",
    "type", "repr", "any", "all", "sum", "open", "iter", "next", "format", "round", "abs",
})


@dataclass(frozen=True)
class HookSettings:
    restate: bool = False
    wide_window: bool = False
    sibling_sites: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "HookSettings":
        values = os.environ if environ is None else environ
        resolved = {}
        for field, key in SWITCHES.items():
            raw = str(values.get(key, "off") or "off").strip().lower()
            if raw not in {"on", "off"}:
                raise ValueError(f"{key} must be on or off")
            resolved[field] = raw == "on"
        return cls(**resolved)

    def receipt(self) -> dict[str, bool]:
        return {"restate": self.restate, "wide_window": self.wide_window, "sibling_sites": self.sibling_sites}


class PlainHooks:
    def __init__(self, settings: HookSettings, declared: Sequence[str], journal, *, out_cap: int):
        self.settings, self.declared, self.journal, self.out_cap = settings, tuple(declared), journal, out_cap

    def _failed(self, hook: str, error: Exception) -> None:
        self.journal.write("hook_error", hook=hook, error_type=type(error).__name__, detail=str(error)[:300])

    # -- restatement ---------------------------------------------------------------------------
    def prepare(self, messages: list, *, product_edits: int) -> list:
        """Wire view only; the stored history never contains the lever."""
        if not self.settings.restate or product_edits > 0 or len(self.declared) < 2:
            return messages
        try:
            index = 0
            while index < len(messages) and messages[index].get("role") == "system":
                index += 1
            if index == 0:
                return messages
            return messages[:index] + [{"role": "user", "content": RESTATE_MESSAGE}] + messages[index:]
        except (AttributeError, TypeError) as error:
            self._failed("restate", error)
            return messages

    # -- wide target window --------------------------------------------------------------------
    def widen(self, relative: str, text: str, start: int, end: int, *, ranged: bool) -> tuple[int, int]:
        """Return a (start, end) at least as wide as requested; unchanged when not applicable."""
        if not (self.settings.wide_window and ranged and relative in self.declared
                and relative.endswith(".py")):
            return start, end
        try:
            lines = text.splitlines()
            spans = _definition_spans(text)
            # Innermost definition cut by each edge of the requested range.
            new_start = max((s for s, e in spans if s < start <= e), default=start)
            new_end = min((e for s, e in spans if s <= end < e), default=end)
            if (new_start, new_end) == (start, end):
                return start, end
            before = CONTEXT_LINES // 2
            new_start = max(1, new_start - before)
            new_end = min(len(lines), new_end + (CONTEXT_LINES - before))
            rendered = sum(len(f"{i}\t{lines[i - 1]}") + 1 for i in range(new_start, new_end + 1))
            if rendered > self.out_cap:
                self.journal.write("hook_skipped", hook="wide_window", reason="budget",
                                   requested=[start, end], widened=[new_start, new_end])
                return start, end
            return new_start, new_end
        except (SyntaxError, ValueError, TypeError, RecursionError) as error:
            self._failed("wide_window", error)
            return start, end

    # -- sibling-site fact ---------------------------------------------------------------------
    def sibling_fact(self, relative: str, old: str, new: str, sources: Mapping[str, str | None],
                     revision: str) -> str | None:
        if not self.settings.sibling_sites or relative not in self.declared:
            return None
        try:
            names = _callables(old)
            if not names:
                return None
            edited = sources.get(relative) or ""
            offset = edited.find(new) if new else -1
            first = edited.count("\n", 0, offset) + 1 if offset >= 0 else 0
            last = first + new.count("\n") if offset >= 0 else -1
            sites: list[str] = []
            scanned = 0
            for path in self.declared:
                source = sources.get(path)
                if source is None:
                    continue
                scanned += 1
                for number, line in enumerate(source.splitlines(), 1):
                    if path == relative and first <= number <= last:
                        continue
                    if any(re.search(r"(?<![\w.])" + re.escape(name) + r"\s*\(", line) for name in names):
                        sites.append(f"{path}:{number}")
            if not sites:
                return None
            shown = sites[:MAX_SITES]
            more = f" (+{len(sites) - len(shown)} more)" if len(sites) > len(shown) else ""
            return (f"[host fact, product revision {revision[:19]}] Other call/construction sites of "
                    + ", ".join(f"`{name}(`" for name in names) + " in the declared files: "
                    + ", ".join(shown) + more
                    + f". Scanned {scanned}/{len(self.declared)} declared files; this is a list, not an instruction.")
        except (TypeError, ValueError, re.error) as error:
            self._failed("sibling_sites", error)
            return None


def _definition_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans.append((first, node.end_lineno or node.lineno))
    return spans


def _callables(text: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"(?<![\w.])([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*\(", text):
        name = match.group(1)
        leaf = name.rsplit(".", 1)[-1]
        if keyword.iskeyword(leaf) or name in _NOT_SITES or name.startswith("self.") and "." not in name[5:]:
            continue
        if name not in found:
            found.append(name)
        if len(found) >= MAX_CALLABLES:
            break
    return found


__all__ = ["CONTEXT_LINES", "HookSettings", "PlainHooks", "RESTATE_MESSAGE", "SWITCHES"]
