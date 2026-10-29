"""Write-through receipt dictionaries over the immutable history spool.

Only callers holding a receipt keep its decoded tree alive. Weak registries
reuse live views, including nested aliases, without retaining old payloads.
"""
from __future__ import annotations

import copy
import pickle
import weakref

from kernel_history_spool import HistorySpool, legacy
from kernel_measurement import profile_active_host_span


def receipts(directory=None):
    return [] if legacy() else ReceiptHistory(directory=directory)


def _no_history():
    return None


def new_receipt(value):
    """Wrap before publication so journal/consumer aliases keep the same row."""
    if legacy():
        return value
    state = _ReceiptState(None, -1, _raw(value))
    return state.wrap(state.value)


def append_receipt(rows, record):
    if isinstance(rows, ReceiptHistory):
        if isinstance(record, ReceiptDict) and record._state.history() is None:
            state = record._state
            rows._append_encoded(pickle.dumps(state.value, protocol=5))
            state.history, state.index = weakref.ref(rows), len(rows) - 1
            rows._live[state.index] = state
            return record
        rows.append(record)
        return rows[-1]
    rows.append(record)
    return record


def _raw(value):
    if isinstance(value, (ReceiptDict, ReceiptList)):
        value = value._node
    return copy.deepcopy(value)


def _restore_receipts(records):
    if legacy():
        return [pickle.loads(record) for record in records]
    result = ReceiptHistory()
    for record in records:
        result._append_encoded(record)
    return result


def _resolve_view(history, index, path):
    value = history[index]
    for key in path:
        value = value[key]
    return value


def _path(root, target, seen=None):
    if root is target:
        return ()
    seen = set() if seen is None else seen
    if id(root) in seen:
        return None
    seen.add(id(root))
    entries = root.items() if isinstance(root, dict) else enumerate(root) if isinstance(root, (list, tuple)) else ()
    for key, value in entries:
        found = _path(value, target, seen)
        if found is not None:
            return (key,) + found
    return None


class _WeakViews:
    """Weak values whose table capacity follows live references, not history."""
    def __init__(self):
        self._values = {}
        self._peak = 0

    def get(self, key):
        ref = self._values.get(key)
        return None if ref is None else ref()

    def __setitem__(self, key, value):
        owner = weakref.ref(self)

        def expired(ref):
            registry = owner()
            if registry is None or registry._values.get(key) is not ref:
                return
            del registry._values[key]
            size = len(registry._values)
            if not size or (registry._peak > 16 and size <= registry._peak // 2):
                registry._values = dict(registry._values)
                registry._peak = size

        self._values[key] = weakref.ref(value, expired)
        self._peak = max(self._peak, len(self._values))


class _ReceiptState:
    def __init__(self, history, index, value):
        self.history = weakref.ref(history) if history is not None else _no_history
        self.index = index
        self.value = value
        self.views = _WeakViews()

    def wrap(self, value):
        if isinstance(value, tuple):
            return tuple(self.wrap(item) for item in value)
        cls = ReceiptDict if isinstance(value, dict) else ReceiptList if isinstance(value, list) else None
        if cls is None:
            return value
        existing = self.views.get(id(value))
        if existing is not None:
            return existing
        result = cls.__new__(cls)
        result._state, result._node = self, value
        self.views[id(value)] = result
        result._sync()
        return result

    def changed(self):
        owner = self.history()
        if owner is not None:
            owner._replace_encoded(self.index, pickle.dumps(self.value, protocol=5))


class _ReceiptView:
    def _changed(self):
        self._sync()
        self._state.changed()

    def __deepcopy__(self, memo):
        # Existing runtime consumers expect detached plain dict/list snapshots.
        return copy.deepcopy(self._node, memo)

    def __reduce_ex__(self, protocol):
        owner = self._state.history()
        path = _path(self._state.value, self._node)
        if owner is not None and path is not None:
            return _resolve_view, (owner, self._state.index, path)
        cls = dict if isinstance(self, dict) else list
        return cls, (self._node,)


class _Missing:
    """Import-stable sentinel; checkpoint module capture must preserve identity."""


_MISSING = _Missing


class ReceiptDict(_ReceiptView, dict):
    def _sync(self):
        dict.clear(self)
        dict.update(self, ((key, self._state.wrap(value)) for key, value in self._node.items()))

    def __setitem__(self, key, value):
        self._node[key] = _raw(value)
        self._changed()

    def __delitem__(self, key):
        del self._node[key]
        self._changed()

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        self._node.update({key: _raw(value) for key, value in changes.items()})
        self._changed()

    def clear(self):
        self._node.clear()
        self._changed()

    def pop(self, key, default=_MISSING):
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self[key]
        del self[key]
        return value

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, value):
        self.update(value)
        return self


class ReceiptList(_ReceiptView, list):
    def _sync(self):
        list.clear(self)
        list.extend(self, (self._state.wrap(value) for value in self._node))

    def __setitem__(self, key, value):
        self._node[key] = _raw(list(value) if isinstance(key, slice) else value)
        self._changed()

    def __delitem__(self, key):
        del self._node[key]
        self._changed()

    def append(self, value):
        self._node.append(_raw(value))
        self._changed()

    def extend(self, values):
        self._node.extend(_raw(list(values)))
        self._changed()

    def insert(self, index, value):
        self._node.insert(index, _raw(value))
        self._changed()

    def pop(self, index=-1):
        value = self[index]
        self._node.pop(index)
        self._changed()
        return value

    def remove(self, value):
        self._node.remove(value)
        self._changed()

    def clear(self):
        self._node.clear()
        self._changed()

    def reverse(self):
        self._node.reverse()
        self._changed()

    def sort(self, *, key=None, reverse=False):
        self._node.sort(key=key, reverse=reverse)
        self._changed()

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __imul__(self, count):
        self._node *= count
        self._changed()
        return self


class ReceiptHistory(HistorySpool):
    def __init__(self, directory=None):
        super().__init__(directory=directory)
        self._live = _WeakViews()

    def append(self, value):
        if not isinstance(value, dict):
            raise TypeError('receipt history requires dictionary records')
        super().append(_raw(value))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = self._normalize(index)
        state = self._live.get(index)
        if state is None:
            encoded = super()._encoded(index)
            with profile_active_host_span('history_receipt_decode'):
                state = _ReceiptState(self, index, pickle.loads(encoded))
            self._live[index] = state
        return state.wrap(state.value)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _encoded(self, index):
        state = self._live.get(index)
        return (pickle.dumps(state.value, protocol=5) if state is not None
                else super()._encoded(index))

    def __reduce_ex__(self, protocol):
        return super().__reduce_ex__(protocol)
