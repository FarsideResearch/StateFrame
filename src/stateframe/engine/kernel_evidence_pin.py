"""Typed evidence pin wire format: build, neutralize and locate the host-authored EVIDENCEPIN block.

A pin is ``[EVIDENCEPIN]<body>[/EVIDENCEPIN]``. Model, tool, task and file text share the host text the wire
check scans, so the scanner must never trust a marker it did not see close. Two rules keep it honest:

* a pin body never contains either marker literal: markers inside untrusted data are neutralized when the pin
  is BUILT (``build_evidence_pin`` / ``neutralize_pin_body``), so a block ends where the host wrote it;
* a block is only counted when it is well formed (no nested start or end marker), so a stray start line in
  task or tool text cannot swallow the real pin.

The wire check then requires the exact typed pin to be the one and only well-formed block on the wire.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

EVIDENCE_START = "[EVIDENCEPIN]"
EVIDENCE_END = "[/EVIDENCEPIN]"
PIN_DEDUP_ENV = "STATEFRAME_PIN_DEDUP"

# Replacement tokens keep the text readable for the model and contain no bracket markers.
_NEUTRAL_START = "{EVIDENCEPIN}"
_NEUTRAL_END = "{/EVIDENCEPIN}"

# A well-formed block: start, then any text that opens no marker, then the first end marker.
_WELL_FORMED_BLOCK = re.compile(
    re.escape(EVIDENCE_START) + r"(?:(?!\[/?EVIDENCEPIN\]).)*?" + re.escape(EVIDENCE_END),
    re.DOTALL,
)


def neutralize_pin_body(body: str) -> str:
    """Replace marker literals inside a pin body so the wrapped block stays parseable."""
    return body.replace(EVIDENCE_START, _NEUTRAL_START).replace(EVIDENCE_END, _NEUTRAL_END)


def build_evidence_pin(body: str) -> str:
    """Wrap a body in the pin markers after neutralizing any marker text the body quotes."""
    return EVIDENCE_START + neutralize_pin_body(body) + EVIDENCE_END


def pin_dedup_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether request composition may elide duplicated evidence-pin text."""
    source = os.environ if environ is None else environ
    value = str(source.get(PIN_DEDUP_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{PIN_DEDUP_ENV} must be off or on")
    return value == "on"



def well_formed_pin_blocks(text: str) -> list[str]:
    """Every pin block in ``text`` that opens and closes without a nested marker, in order."""
    return [match.group(0) for match in _WELL_FORMED_BLOCK.finditer(text)]


def typed_pin_is_sole_wire_block(text: str, typed: str) -> bool:
    """True when ``typed`` appears exactly once in ``text`` and is the only well-formed pin block there."""
    if not typed or text.count(typed) != 1:
        return False
    return well_formed_pin_blocks(text) == [typed]
