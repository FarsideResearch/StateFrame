"""Optional grounding and freshness policy for generated host checks.

The policy is deliberately evidence-only.  It reads the public ticket and
public material already present in a requirements ledger; it never opens a
workspace path and it never treats a check identifier as product vocabulary.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


MODE_ENV = "STATEFRAME_ORACLE_TICKET_LITERALS"
ADVISORY = "ADVISORY"
NOT_IN_TICKET = "not in ticket"
STALE = "STALE"
INFO = "info"
FRESHNESS_STALE = "stale"

# These names describe the host-check wire format, not a product contract.
_STRUCTURAL_KEYS = frozenset({
    "args", "kwargs", "calls", "cases", "base", "variant", "input_index",
    "path", "paths", "json", "status", "body", "data", "output", "result",
    "response", "value", "error", "errors", "message", "detail", "code",
    "type", "ok", "status_code", "http_status", "http_code", "contains",
    "error_contains", "preserved", "equivalent", "outputs", "selector",
    "mode", "percent", "total_path", "component_paths", "public_value",
    "input", "fixture", "expected", "operator", "predicate", "template",
    "entry_point", "command", "selection", "tested_domain", "obligation",
})
_IGNORED_PATH_KEYS = frozenset({
    "args", "kwargs", "calls", "cases", "input_index", "path", "paths",
    "anchor", "total_path", "component_paths", "selector", "selection",
    "command", "entry_point", "template",
})
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "source", "src", "content", "public_source"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return ""


def _text_collection(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        if _text(value):
            return (_text(value),)
        parts: list[str] = []
        for item in value.values():
            parts.extend(_text_collection(item))
        return tuple(parts)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return tuple(text for item in value if (text := _text(item)))
    return ()


def public_material(ledger: Mapping[str, object] | None) -> tuple[str, str]:
    """Return ``(ticket_text, public_code_text)`` without reading the disk."""
    if not isinstance(ledger, Mapping):
        return "", ""
    ticket_id = str(ledger.get("ticket_source_id", ""))
    ticket_parts: list[str] = []
    code_parts: list[str] = []
    sources = ledger.get("sources", ())
    if isinstance(sources, Mapping):
        sources = (sources,) if _text(sources) else tuple(sources.values())
    if isinstance(sources, Iterable) and not isinstance(sources, (str, bytes, bytearray)):
        for source in sources:
            text = _text(source)
            if not text:
                continue
            kind = str(_field(source, "kind", "")).lower()
            ident = str(_field(source, "id", ""))
            if ident == ticket_id or kind == "ticket":
                ticket_parts.append(text)
            else:
                code_parts.append(text)
    if not ticket_parts:
        for key in ("ticket", "ticket_text", "public_ticket", "brief"):
            ticket_parts.extend(_text_collection(ledger.get(key)))
            if ticket_parts:
                break
    for key in ("public_code", "public_sources", "code", "public_tests", "tests"):
        code_parts.extend(_text_collection(ledger.get(key)))
    return "\n".join(ticket_parts), "\n".join(code_parts)


def _iter_expected_literals(value: object, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text not in _STRUCTURAL_KEYS and key_text not in _IGNORED_PATH_KEYS:
                yield "identifier", key_text, path
            yield from _iter_expected_literals(child, (*path, key_text))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _iter_expected_literals(child, (*path, str(index)))
        return
    if path and path[-1] in _IGNORED_PATH_KEYS:
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)) and str(value).strip():
        yield "literal", value, path


def expected_literals(check: object) -> tuple[object, ...]:
    """Return unique product-facing assertion atoms from a generated check."""
    result: list[object] = []
    for _kind, value, _path in _iter_expected_literals(_field(check, "expected")):
        if value not in result:
            result.append(copy.deepcopy(value))
    return tuple(result)


def _forms(value: object) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return ()
    forms = [text]
    if _IDENTIFIER.fullmatch(text) and "." in text:
        forms.extend(part for part in text.split(".") if part)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
            if number.is_integer():
                forms.append(str(int(number)))
            forms.append(str(number))
        except (TypeError, ValueError):
            pass
    return tuple(dict.fromkeys(forms))


def _present(value: object, text: str) -> bool:
    if not text:
        return False
    for form in _forms(value):
        if _NUMBER.fullmatch(form):
            pattern = (rf"(?<![A-Za-z0-9_.]){re.escape(form)}"
                       rf"(?![A-Za-z0-9_]|\.(?=\d))")
            if re.search(pattern, text):
                return True
        elif _IDENTIFIER.fullmatch(form):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(form)}(?![A-Za-z0-9_])"
            if re.search(pattern, text, re.IGNORECASE):
                return True
        elif form.casefold() in text.casefold():
            return True
    return False


@dataclasses.dataclass(frozen=True, slots=True)
class LiteralGrounding:
    applicable: bool
    literals: tuple[object, ...] = ()
    missing: tuple[object, ...] = ()
    ticket_text_present: bool = False
    public_code_present: bool = False

    @property
    def demote(self) -> bool:
        return self.applicable and bool(self.missing)

    @property
    def reason(self) -> str:
        return NOT_IN_TICKET if self.demote else "grounded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "literals": copy.deepcopy(list(self.literals)),
            "missing": copy.deepcopy(list(self.missing)),
            "ticket_text_present": self.ticket_text_present,
            "public_code_present": self.public_code_present,
            "reason": self.reason,
        }


def ground_check(check: object, ledger: Mapping[str, object] | None) -> LiteralGrounding:
    """Compare assertion atoms with public ticket/code material."""
    literals = expected_literals(check)
    ticket, code = public_material(ledger)
    material = f"{ticket}\n{code}" if ticket or code else ""
    if not literals or not material:
        return LiteralGrounding(
            applicable=False, literals=literals,
            ticket_text_present=bool(ticket), public_code_present=bool(code),
        )
    missing = tuple(value for value in literals if not _present(value, material))
    return LiteralGrounding(
        applicable=True, literals=literals, missing=missing,
        ticket_text_present=bool(ticket), public_code_present=bool(code),
    )


def admission_reason(check: object, ledger: Mapping[str, object] | None) -> str | None:
    if not enabled():
        return None
    grounding = ground_check(check, ledger)
    return NOT_IN_TICKET if grounding.demote else None


def _revision(value: object) -> object:
    for key in ("repository_revision", "snapshot_digest", "snapshot", "revision", "tested_revision"):
        candidate = _field(value, key, None)
        if candidate is not None and str(candidate) != "":
            return candidate
    return None


def receipt_revision(receipt: object) -> object:
    return _revision(receipt)


def _identity(value: object) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def revision_is_current(receipt: object, current_revision: object) -> bool:
    observed = receipt_revision(receipt)
    if current_revision is None or observed is None:
        return True
    return (str(observed) == str(current_revision)
            or _identity(observed) == _identity(current_revision))


def is_stale(receipt: object, current_revision: object) -> bool:
    return current_revision is not None and receipt_revision(receipt) is not None and not revision_is_current(
        receipt, current_revision)


def stale_row(row: Mapping[str, object], receipt: object, current_revision: object) -> dict[str, object]:
    """Make the freshness contract explicit in a presentation row."""
    result = copy.deepcopy(dict(row))
    revision = receipt_revision(receipt)
    result.update({
        "status": STALE,
        "revision": str(revision),
        "receipt_revision": str(revision),
        "current_revision": str(current_revision),
        "freshness": FRESHNESS_STALE,
        "render": INFO,
        "blocking": False,
        "message": f"{STALE} host-check receipt at revision {revision}; current revision {current_revision}",
    })
    return result


def info_row(row: Mapping[str, object], *, message: str) -> dict[str, object]:
    result = copy.deepcopy(dict(row))
    result.update({"render": INFO, "severity": INFO, "blocking": False, "message": message})
    return result


__all__ = [
    "ADVISORY", "FRESHNESS_STALE", "INFO", "LiteralGrounding", "MODE_ENV", "NOT_IN_TICKET", "STALE",
    "admission_reason", "enabled", "expected_literals", "ground_check", "info_row",
    "is_stale", "mode", "public_material", "receipt_revision", "revision_is_current",
    "stale_row",
]
