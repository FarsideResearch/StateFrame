"""Sparse after-binding revision histories for partial router comparisons."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
import copy


class BindingHistory:
    """Retain each path's changes once, instead of a map per action."""
    def __init__(self):
        self.paths: dict[tuple, list[tuple]] = {}
        self.latest: dict[tuple, object] = {}
        self.changes: list[list[tuple]] = []

    def append(self, binding: object) -> None:
        binding = binding if isinstance(binding, Mapping) else {}
        values = {(group, path): revision
                  for group in ('product_files', 'harness_files')
                  if isinstance(binding.get(group), Mapping)
                  for path, revision in binding[group].items()}
        changed = []
        index = len(self.changes)
        for key in self.latest.keys() | values.keys():
            # Presence matters: an explicit None revision differs from absence.
            old, new = (key in self.latest, self.latest.get(key)), (key in values, values.get(key))
            if old != new:
                self.paths.setdefault(key, []).append((index, copy.deepcopy(new)))
                changed.append(key)
        self.latest = copy.deepcopy(values)
        self.changes.append(changed)

    def replace_last(self, binding: object) -> None:
        for key in self.changes.pop():
            history = self.paths[key]
            history.pop()
            if not history:
                del self.paths[key]
                self.latest.pop(key, None)
            elif history[-1][1][0]:
                self.latest[key] = history[-1][1][1]
            else:
                self.latest.pop(key, None)
        self.append(binding)

    def matches(self, index: int, current: Mapping) -> bool:
        for group in ('product_files', 'harness_files'):
            files = current.get(group)
            if not isinstance(files, Mapping):
                continue
            for path, revision in files.items():
                history = self.paths.get((group, path), ())
                position = bisect_right(history, index, key=lambda entry: entry[0]) - 1
                if position >= 0:
                    present, previous = history[position][1]
                    if present and previous != revision:
                        return False
        return True
