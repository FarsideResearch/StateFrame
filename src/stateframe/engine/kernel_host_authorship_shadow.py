"""Shadow comparison, pass-boundary marks and the off-wire manifest for host-row identity.

In ``shadow`` mode a converted reader computes its legacy answer and its ledger
answer, reports a difference here and returns the legacy answer. Marks record
which rows existed at each pass boundary, so a row the ledger does not know can
be traced to the pass that produced it. Stdlib only.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import kernel_host_authorship as _authorship
from kernel_host_authorship import HostAuthorship

_JSON_SCALARS = (str, int, float, bool, type(None))


def _row_key(message: object) -> tuple[str, str] | None:
    if not isinstance(message, Mapping):
        return None
    role, content = message.get("role"), message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        return None
    return (role, _authorship._sha(content))


def mark(ledger: HostAuthorship | None, tag: str, messages: Sequence[object]) -> None:
    """Record the set of row hashes at a pass boundary (shadow mode only)."""
    if ledger is None or _authorship.mode() != "shadow":
        return
    ledger.marks.append((tag, frozenset(
        key for key in (_row_key(message) for message in messages) if key is not None)))


def _first_interval(ledger: HostAuthorship, message: object) -> str | None:
    key = _row_key(message)
    previous = ""
    for tag, keys in ledger.marks:
        if key in keys:
            return f"{previous}..{tag}"
        previous = tag
    return None


def _plain(value: object) -> object:
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    return f"<{type(value).__name__}>"


def compare(ledger: HostAuthorship | None, journal: object, reader: str, legacy: object,
            ledger_answer: object, row: object) -> bool:
    """Report a reader whose two answers differ; True when they agree."""
    if legacy == ledger_answer:
        return True
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            "host_identity_shadow", reader=reader, legacy=_plain(legacy),
            ledger=_plain(ledger_answer),
            site=_authorship.row_site(ledger, row),
            mark=_first_interval(ledger, row) if ledger is not None else None,
            step=getattr(ledger, "step", None), request_id=getattr(ledger, "request_id", None),
            dispatch=False,
        )
    return False


def manifest(ledger: HostAuthorship | None, messages: Sequence[object], *,
             wire: bytes | None = None, capture: object = None) -> dict[str, object] | None:
    """Row kinds and data spans of one submitted wire, for readers with no live ledger."""
    if ledger is None:
        return None
    order = _authorship.occurrences(ledger, messages, capture=capture)
    rows = []
    for index, message in enumerate(messages):
        kind = _authorship.row_kind(ledger, message, occurrence=order[index])
        if kind is None:
            continue
        rows.append([index, kind, [[start, end - start] for start, end in
                                   _authorship.data_spans(ledger, message, occurrence=order[index])]])
    return {
        "step": ledger.step,
        "request_id": ledger.request_id,
        "body_sha256": hashlib.sha256(wire).hexdigest() if wire is not None else None,
        "rows": rows,
    }
