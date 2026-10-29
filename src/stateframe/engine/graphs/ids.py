"""graphs/ids.py — canonical entity IDs for every graph layer.

ONE identity spine (a_graph_layers.md "Cross-layer identity", line ~803):

    A function does not have l1_function_id / l2_function_id / l5_history_function_id.
    It has ONE `entity_id`, and each layer attaches facts to it.

This module does NOT invent a second ID scheme. Every ID it returns is minted by
``architecture/domain_ids.py`` — the landed canon — through its own typed
constructors. This file is the thin, code-owned mapping from *graph* entity kinds
(file, function, class, seam, commit, token) onto the kinds ``domain_ids``
already covers, so that:

  * ``graphs`` and any future architecture lane produce byte-identical IDs for the
    same logical entity (the w65/a_wave2_prep "cross-repo ID disagreement" risk);
  * nothing here re-normalizes, re-slugs, or re-hashes anything ``domain_ids``
    or ``seam_graph`` already owns (#211 shared-vocabulary ownership).

THE COLLISION PROBLEM, MEASURED NOT ASSUMED
-------------------------------------------
``domain_ids.normalize_slug`` collapses ``[^a-z0-9]+`` to ``-`` and strips edge
dashes. That is deliberate there (``"Invoice Worker"`` and ``"invoice-worker"``
are MEANT to collide — they are the same component named two ways). It is wrong
for a code entity, where ``_get_audience`` and ``get_audience`` are two different
functions that both slug to ``get-audience``.

This is not hypothetical. The first full build of ``mission-control/backend/app``
produced **92 collisions**, every one of them a private/public pair in the same
module (``routers/audiences.py::_get_audience`` vs ``::get_audience``,
``routers/canvas.py::_get_canvas`` vs ``::get_canvas``, ...). w65/a_wave2_prep is
explicit that unknown nodes are acceptable but "ID collisions, silent
reassignment, or unreported L2 truncation are not."

THE FIX: every entity ID carries a DISAMBIGUATOR — a 12-hex-char digest of the
exact raw ``(repo, kind, name)`` triple, appended to the readable slug::

    component:mc:function:routers-audiences-py-get-audience-h4f2c1a90b3de
    component:mc:function:routers-audiences-py--get-audience-h71bd0c4e2a15

The readable half survives for humans and for grep; the digest half makes the ID
injective on the raw key. The result is still a well-formed ``domain_ids`` ID —
``parse_domain_id`` round-trips it, the ``component:`` scheme is unchanged, and
nothing in ``domain_ids`` was edited to accommodate this. We did NOT fork the
scheme; we stopped feeding it a lossy name.

The collision detector stays anyway. It is now a guard that must never fire, and
:class:`EntityIdMinter` raises :class:`EntityIdCollision` if it ever does — a
silent reassignment is the failure this module exists to prevent, and a fix you
cannot detect regressing is not a fix.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict

_POC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC not in sys.path:
    sys.path.insert(0, _POC)

from architecture.domain_ids import (  # noqa: E402
    component_id,
    normalize_slug,
    parse_domain_id,
    seam_id,
)

# ---------------------------------------------------------------------------
# Graph entity kinds -> the bounded-context slot of domain_ids.component_id
# ---------------------------------------------------------------------------

#: ``component_id(system, bounded_context, name)``. We reserve the
#: bounded-context slot for the graph entity kind so an ID reads
#: ``component:<repo>:<kind>:<name>`` and ``parse_domain_id`` recovers the kind
#: without this module owning a parallel registry.
KIND_FILE = "file"
KIND_FUNCTION = "function"
KIND_CLASS = "class"
KIND_MODULE = "module"
KIND_COMMIT = "commit"
KIND_TOKEN = "token"

GRAPH_ENTITY_KINDS = (
    KIND_FILE,
    KIND_FUNCTION,
    KIND_CLASS,
    KIND_MODULE,
    KIND_COMMIT,
    KIND_TOKEN,
)


class EntityIdCollision(ValueError):
    """Two distinct raw keys minted the same canonical entity ID."""

    def __init__(self, entity_id: str, existing_key: str, new_key: str):
        super().__init__(
            f"entity id collision on {entity_id!r}: already minted for "
            f"{existing_key!r}, now requested for {new_key!r}"
        )
        self.entity_id = entity_id
        self.existing_key = existing_key
        self.new_key = new_key


def _raw_key(repo: str, kind: str, name: str) -> str:
    """The pre-normalization identity. Collisions are defined against THIS."""
    return f"{repo}\x00{kind}\x00{name}"


#: Hex chars of the disambiguating digest. 12 hex = 48 bits: at ~50k entities per
#: repo the birthday probability is ~7e-11. Long enough to be safe, short enough
#: that the readable half of the ID still dominates when a human reads it.
DISAMBIGUATOR_HEX = 12

#: Placeholder when a name normalizes to nothing at all (e.g. a symbol named
#: only in punctuation). The digest still separates it from every other such
#: name, so the ID stays injective; the readable half just stops being useful.
_EMPTY_SLUG = "x"


def disambiguated_name(repo: str, kind: str, name: str) -> str:
    """``<readable-slug>-h<digest>`` — the injective name part of an entity ID.

    Pure and deterministic: the same triple always yields the same string, on
    any machine, in any run order. No clock, no counter, no randomness.
    """
    try:
        readable = normalize_slug(name)
    except ValueError:
        readable = _EMPTY_SLUG
    digest = hashlib.sha256(_raw_key(repo, kind, name).encode("utf-8")).hexdigest()
    return f"{readable}-h{digest[:DISAMBIGUATOR_HEX]}"


@dataclass
class EntityIdMinter:
    """Deterministic entity-ID minter with collision detection.

    Same inputs -> same ID, always. No clock, no randomness, no counters in the
    ID itself (a counter would make the ID build-order dependent and destroy
    cross-repo/cross-run comparability).
    """

    system_slug: str
    _by_id: Dict[str, str] = field(default_factory=dict)
    collisions: list = field(default_factory=list)
    #: strict=True is the production setting. strict=False exists ONLY so a
    #: diagnostic run can enumerate every collision instead of stopping at the
    #: first — it must never be how a real build is configured.
    strict: bool = True

    def _mint(self, kind: str, name: str) -> str:
        if kind not in GRAPH_ENTITY_KINDS:
            raise ValueError(f"unknown graph entity kind: {kind!r}")
        eid = component_id(
            self.system_slug, kind, disambiguated_name(self.system_slug, kind, name)
        )
        raw = _raw_key(self.system_slug, kind, name)
        prior = self._by_id.get(eid)
        if prior is not None and prior != raw:
            self.collisions.append((eid, prior, raw))
            if self.strict:
                raise EntityIdCollision(eid, prior, raw)
        else:
            self._by_id[eid] = raw
        return eid

    # -- typed constructors, one per graph entity kind ---------------------

    def file_id(self, rel_path: str) -> str:
        return self._mint(KIND_FILE, rel_path)

    def module_id(self, dotted: str) -> str:
        return self._mint(KIND_MODULE, dotted)

    def function_id(self, rel_path: str, qualname: str) -> str:
        return self._mint(KIND_FUNCTION, f"{rel_path}::{qualname}")

    def class_id(self, rel_path: str, qualname: str) -> str:
        return self._mint(KIND_CLASS, f"{rel_path}::{qualname}")

    def commit_id(self, sha: str) -> str:
        return self._mint(KIND_COMMIT, sha)

    def token_id(self, token: str) -> str:
        return self._mint(KIND_TOKEN, token)

    def seam_entity_id(self, seam_key: str) -> str:
        """``seam:<repo>:<seam_key>`` — delegates to ``domain_ids.seam_id``.

        Deliberately NOT routed through ``_mint``: ``seam_graph.py`` already owns
        seam-key normalization (see domain_ids.seam_id's own docstring), and the
        seam namespace is disjoint from the component namespace so it cannot
        collide with a file/function ID.
        """
        return seam_id(self.system_slug, seam_key)

    # -- introspection -----------------------------------------------------

    @property
    def minted(self) -> int:
        return len(self._by_id)

    def raw_key_for(self, entity_id: str) -> str | None:
        return self._by_id.get(entity_id)


def kind_of(entity_id: str) -> str:
    """Recover the graph entity kind from a canonical ID.

    ``component:<repo>:<kind>:<name>`` -> ``<kind>``; a ``seam:`` ID reports
    ``"seam"``. Raises ``ValueError`` on anything this module did not mint.
    """
    parsed = parse_domain_id(entity_id)
    if parsed.kind == "seam":
        return "seam"
    if parsed.kind != "component" or len(parsed.parts) < 3:
        raise ValueError(f"not a graph entity id: {entity_id!r}")
    kind = parsed.parts[1]
    if kind not in GRAPH_ENTITY_KINDS:
        raise ValueError(f"not a graph entity kind: {kind!r} in {entity_id!r}")
    return kind


def repo_of(entity_id: str) -> str:
    """Recover the repo (system) slug from a canonical entity ID."""
    parsed = parse_domain_id(entity_id)
    return parsed.parts[0]


__all__ = [
    "EntityIdMinter",
    "EntityIdCollision",
    "GRAPH_ENTITY_KINDS",
    "KIND_CLASS",
    "KIND_COMMIT",
    "KIND_FILE",
    "KIND_FUNCTION",
    "KIND_MODULE",
    "KIND_TOKEN",
    "kind_of",
    "normalize_slug",
    "repo_of",
]
