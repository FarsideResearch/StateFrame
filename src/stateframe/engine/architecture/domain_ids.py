"""Stable domain identifiers for decision architecture.

Domain identity must be explicit so unrelated decisions cannot share state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Slug normalization
# ---------------------------------------------------------------------------

_SLUG_DISALLOWED: Final = re.compile(r"[^a-z0-9]+")
_SLUG_EDGE_DASH: Final = re.compile(r"^-+|-+$")


def normalize_slug(raw: str) -> str:
    """Lowercase and collapse to ``[a-z0-9-]``, stripping leading/trailing dashes.

    Deterministic and idempotent: ``normalize_slug(normalize_slug(x)) == normalize_slug(x)``.
    Raises on empty input (including input that normalizes to empty) — a blank identity
    component is never valid, it must be caught at construction time, not discovered later
    as an ID that silently collides with every other blank-component ID.
    """
    if raw is None:
        raise ValueError("normalize_slug: raw is None")
    slug = _SLUG_DISALLOWED.sub("-", raw.strip().lower())
    slug = _SLUG_EDGE_DASH.sub("", slug)
    if not slug:
        raise ValueError(f"normalize_slug: {raw!r} normalizes to an empty slug")
    return slug


# ---------------------------------------------------------------------------
# Entity kinds this scheme covers
# ---------------------------------------------------------------------------

# Kept as plain strings (not an Enum) so a new kind can be added by adding one constructor
# function below without touching a central enum every other lane also imports.
KIND_ARCH: Final = "arch"
KIND_COMPONENT: Final = "component"
KIND_CONTRACT: Final = "contract"
KIND_LAYER: Final = "layer"
KIND_RULE: Final = "rule"
KIND_SEAM: Final = "seam"
KIND_DECISION: Final = "decision"

ALL_KINDS: Final = frozenset(
    {
        KIND_ARCH,
        KIND_COMPONENT,
        KIND_CONTRACT,
        KIND_LAYER,
        KIND_RULE,
        KIND_SEAM,
        KIND_DECISION,
    }
)


@dataclass(frozen=True)
class DomainId:
    """A parsed canonical ID: kind + ordered, normalized parts.

    ``str(DomainId(...))`` reproduces the exact canonical string. Two ``DomainId`` values
    compare equal iff they'd serialize to the same string — that's the whole identity
    contract this module exists to guarantee across lanes.
    """

    kind: str
    parts: tuple[str, ...]

    def __str__(self) -> str:
        return ":".join((self.kind, *self.parts))


def _build(kind: str, *parts: str) -> str:
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown domain id kind: {kind!r} (expected one of {sorted(ALL_KINDS)})")
    if not parts:
        raise ValueError(f"domain id kind {kind!r} requires at least one part")
    normalized = tuple(normalize_slug(p) for p in parts)
    return str(DomainId(kind=kind, parts=normalized))


# ---------------------------------------------------------------------------
# Typed constructors — one function per entity kind, matching a_arch_module section 3
# ---------------------------------------------------------------------------


def arch_id(system_slug: str) -> str:
    """``arch:<system-slug>`` — one architecture artifact for one logical system."""
    return _build(KIND_ARCH, system_slug)


def component_id(system_slug: str, bounded_context: str, name: str) -> str:
    """``component:<system-slug>:<bounded-context>:<name>``."""
    return _build(KIND_COMPONENT, system_slug, bounded_context, name)


def contract_id(
    system_slug: str,
    producer: str,
    consumer: str,
    name: str,
    version: int | str,
) -> str:
    """``contract:<system-slug>:<producer>:<consumer>:<name>:v<version>``.

    ``version`` is rendered as ``v<version>`` so the ID sorts and reads the way
    ``a_arch_module`` section 3's examples do (``...create-invoice:v1``).
    """
    version_part = f"v{version}"
    return _build(KIND_CONTRACT, system_slug, producer, consumer, name, version_part)


def layer_id(system_slug: str, name: str) -> str:
    """``layer:<system-slug>:<name>``."""
    return _build(KIND_LAYER, system_slug, name)


def rule_id(system_slug: str, source_layer: str, target_layer: str, rule_name: str) -> str:
    """``rule:<system-slug>:<source-layer>:<target-layer>:<rule-name>``."""
    return _build(KIND_RULE, system_slug, source_layer, target_layer, rule_name)


def seam_id(system_slug: str, seam_key: str) -> str:
    """``seam:<system-slug>:<seam_key>``.

    Wraps ``platform/seam_graph.py``'s own ``seam_key`` (already the canonical join key
    between a route's producer and consumer sides — see ``http_seam_key`` /
    ``SeamEdge.seam_key`` there) with the ``system_slug`` namespace so seam identity is
    comparable across systems without re-deriving anything the seam extractor already
    computed. Deliberately does NOT re-normalize ``seam_key`` beyond slugging it, since
    ``seam_graph.py`` already owns that normalization (path templating, method casing, ...).
    """
    return _build(KIND_SEAM, system_slug, seam_key)


def decision_id(system_slug: str, slug: str) -> str:
    """``decision:<system-slug>:<slug>`` — an architecture decision record."""
    return _build(KIND_DECISION, system_slug, slug)


# ---------------------------------------------------------------------------
# Parsing — the round-trip half of the contract
# ---------------------------------------------------------------------------


def parse_domain_id(value: str) -> DomainId:
    """Parse a canonical ID string back into its kind and normalized parts.

    Raises ``ValueError`` on anything that isn't a well-formed ID from this scheme:
    unknown kind, missing parts, or any part that fails ``normalize_slug`` (which would
    mean the string wasn't produced by this module's own constructors, since every part
    they emit is already normalized and therefore normalize-idempotent).
    """
    if not value or ":" not in value:
        raise ValueError(f"not a domain id: {value!r}")
    kind, _, rest = value.partition(":")
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown domain id kind: {kind!r} in {value!r}")
    raw_parts = rest.split(":")
    if not raw_parts or any(p == "" for p in raw_parts):
        raise ValueError(f"malformed domain id (empty part): {value!r}")
    parts = tuple(normalize_slug(p) for p in raw_parts)
    for raw, normalized in zip(raw_parts, parts):
        if raw != normalized:
            raise ValueError(
                f"domain id {value!r} is not canonical: part {raw!r} != {normalized!r} "
                "(construct IDs via this module's typed constructors, never by hand)"
            )
    return DomainId(kind=kind, parts=parts)


def is_domain_id(value: str) -> bool:
    """Best-effort predicate wrapping :func:`parse_domain_id` for call sites that just
    need a yes/no (e.g. deciding whether a graph-node field is already a canonical ID
    versus raw evidence text that still needs one minted)."""
    try:
        parse_domain_id(value)
    except ValueError:
        return False
    return True
