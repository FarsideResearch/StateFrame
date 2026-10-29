"""LAYER REGISTRY — the one canonical, named layer taxonomy for the graph platform.

Named layers reconcile two conflicting numeric taxonomies.

THE CONFLICT (measured, not inferred)
--------------------------------------
Two earlier taxonomies describe what is meant to be the same six-layer
whole-codebase comprehension graph, using two INCOMPATIBLE numeric schemes for
overlapping-but-not-identical concepts:

  * Whole-codebase scheme -- L1..L6
        L1 Structure graph            (imports/calls/exports; "answers: which
                                        symbols exist and how do they reference
                                        each other")
        L2 Dataflow graph             (def-use, local + whole-repo taint mode)
        L3 Seams and contracts graph  (service boundaries, schemas, endpoints)
        L4 Ownership and runtime graph(deploy binding, desired vs observed state)
        L5 History graph              (commits, blame, lineage)
        L6 Intent graph               (docstrings, comments, inferred behavior)

  * Incremental scheme -- L0..L5
        L0 Artifact and repository identity (file/path/hash/line-map)
        L1 Syntax and spans                 (AST/CST subtree, declaration span)
        L2 Symbols and def-use               (definitions, references, calls)
        L3 Contracts and seams                (exports, modules, APIs, schemas)
        L4 Fine semantic analysis             (behavior summaries, taint, tests)
        L5 History and provenance             (commits, snapshots, blame, lineage)

Meanwhile the hierarchical working model (the ZoomSliceV1 authority
-- see zoom_slice.py) uses NAMED levels PLATFORM | APP | SUBSYSTEM, and
the skill and goal convergence layers already use those named scopes.

THE RESOLUTION
---------------------------------------------
    "RECOMMENDATION: named levels win, numbers are retired. Numeric layer IDs
    can be confused across dependent components (#211)."

This module is the code-owned enactment of that resolution for the GRAPH
PLATFORM's six layers specifically (a separate, narrower taxonomy than the
ZoomSliceV1 PLATFORM/APP/SUBSYSTEM scope levels -- the two are not the same
axis; see the note at the bottom of this docstring). Every layer gets one
canonical, named identity. The two legacy numeric schemes are retired to
comment-only mapping tables -- never used as a live identifier again.

An architecture `Layer` type uses named string IDs (`layer:api`) plus
an informative `ordinal` int -- NOT a third L-prefixed numbering scheme. It is
compatible with this resolution; a caller mapping into that doc's `Layer` type
should set `ordinal` to this enum's `CanonicalLayer` position (0-based) and
`layer:<name.lower()>` as the string ID. No code in this module does that
mapping today -- it is documented here so a future integration doesn't
reinvent a numbering scheme by accident.

THE MAPPING
-----------
The two source schemes do NOT describe an identical six concepts under two
offsets (i.e. incremental's L0 is not simply layers-L1 renumbered down by
one). They overlap heavily but diverge in scope for RUNTIME/L4 and diverge
entirely at the edges (incremental's L0 identity layer has no standalone
counterpart in the whole-codebase scheme; that scheme's L6 Intent layer has no
counterpart in the incremental scheme, which stops at L5). The mapping below is
by CONTENT, not by numeric position.

| Canonical  | Whole-codebase scheme (L1-L6)    | Incremental scheme (L0-L5)         |
|------------|------------------------------------|--------------------------------------|
| ARTIFACT   | (no standalone layer -- folded    | L0 "Artifact and repository          |
|            | into the common canonical entity   |    identity" (line ~15)              |
|            | registry, doc section 1, line ~9:  |                                       |
|            | "every layer references the same   |                                       |
|            | canonical entity_id")              |                                       |
| STRUCTURE  | L1 "Structure graph" (line ~100)   | L1 "Syntax and spans" (line ~16)     |
| DATAFLOW   | L2 "Dataflow graph" (line ~193)    | L2 "Symbols and def-use" (line ~17)  |
| SEAM       | L3 "Seams and contracts graph"     | L3 "Contracts and seams" (line ~18)  |
|            |    (line ~325)                     |                                       |
| RUNTIME    | L4 "Ownership and runtime graph"   | L4 "Fine semantic analysis" (line    |
|            |    (line ~436)                     |    ~19) -- incremental's L4 is       |
|            |                                     | BROADER (also covers taint/behavior  |
|            |                                     | summaries that the first scheme puts |
|            |                                     | into its L2 whole-repo taint mode);  |
|            |                                     | RUNTIME is the union, not a strict   |
|            |                                     | subset of either source.             |
| HISTORY    | L5 "History graph" (line ~552)     | L5 "History and provenance" (line    |
|            |                                     |    ~20) -- exact conceptual match.   |
| INTENT     | L6 "Intent graph" (line ~662)      | (no counterpart -- incremental stops |
|            |                                     | at L5; flagged as a known gap below, |
|            |                                     | not silently dropped)                |

Six canonical layers total (ARTIFACT, STRUCTURE, DATAFLOW, SEAM, RUNTIME,
HISTORY) plus INTENT as a seventh, extension-status layer that only one of the
two source docs defines. `CanonicalLayer` enumerates all seven; `CORE_SIX`
names the six in the incremental core so a caller can select that scope explicitly.

Relationship to ZoomSliceV1's PLATFORM/APP/SUBSYSTEM (see zoom_slice.py):
those are SCOPE levels (how much of the repository a slice covers -- whole
platform, one app cluster, or one materialized subsystem) and are orthogonal
to this module's LAYER axis (what kind of fact the graph records -- structure,
dataflow, seams, ...). A ZoomSlice at SUBSYSTEM scope can carry facts from
several of these layers at once. Do not conflate the two axes; the
named-layer resolution applies independently to each.
"""

from __future__ import annotations

import enum
from typing import Dict, Tuple


class CanonicalLayer(enum.Enum):
    """The one named layer taxonomy; numeric scheme conflicts are retired."""

    ARTIFACT = "artifact"
    STRUCTURE = "structure"
    DATAFLOW = "dataflow"
    SEAM = "seam"
    RUNTIME = "runtime"
    HISTORY = "history"
    INTENT = "intent"


# The six layers in the incremental scheme's native span (L0-L5). INTENT is real
# in the whole-codebase scheme (L6) but absent from the incremental scheme, so it
# is kept out of the "core six" set that
# code expecting incremental-rebuild parity should iterate.
CORE_SIX: Tuple[CanonicalLayer, ...] = (
    CanonicalLayer.ARTIFACT,
    CanonicalLayer.STRUCTURE,
    CanonicalLayer.DATAFLOW,
    CanonicalLayer.SEAM,
    CanonicalLayer.RUNTIME,
    CanonicalLayer.HISTORY,
)

# All seven, in canonical dependency order (each layer's construction generally
# depends on the ones before it being resolvable -- ARTIFACT first, INTENT last
# per both source docs' construction-cost sections).
ALL_LAYERS: Tuple[CanonicalLayer, ...] = CORE_SIX + (CanonicalLayer.INTENT,)


def arch_module_ordinal(layer: CanonicalLayer) -> int:
    """0-based ordinal for the architecture `Layer.ordinal` field.

    The architecture `Layer` type is named-string + ordinal already, not a
    third numbering scheme, so it is compatible with this resolution as long
    as ordinal is derived here rather than hand-assigned per call site.
    """
    return ALL_LAYERS.index(layer)


def arch_module_layer_id(layer: CanonicalLayer) -> str:
    """`layer:<name>` string ID matching the architecture layer convention."""
    return f"layer:{layer.value}"


# ---------------------------------------------------------------------------
# Legacy numeric mapping tables -- COMMENT-ONLY DATA, never a live identifier.
# Exist solely so a drift/migration reader (and the tests below) can prove the
# mapping is total and non-contradictory. Do not construct a graph fact keyed
# by these numbers; key it by CanonicalLayer.
# ---------------------------------------------------------------------------

#: Whole-codebase scheme L1-L6 -> canonical; see the module docstring table.
GRAPH_LAYERS_L1_L6: Dict[int, CanonicalLayer] = {
    1: CanonicalLayer.STRUCTURE,
    2: CanonicalLayer.DATAFLOW,
    3: CanonicalLayer.SEAM,
    4: CanonicalLayer.RUNTIME,
    5: CanonicalLayer.HISTORY,
    6: CanonicalLayer.INTENT,
}

#: Incremental scheme L0-L5 -> canonical; see the module docstring table. Has no L6/
#: INTENT entry -- that is the documented gap, not an omission bug.
GRAPH_INCREMENTAL_L0_L5: Dict[int, CanonicalLayer] = {
    0: CanonicalLayer.ARTIFACT,
    1: CanonicalLayer.STRUCTURE,
    2: CanonicalLayer.DATAFLOW,
    3: CanonicalLayer.SEAM,
    4: CanonicalLayer.RUNTIME,
    5: CanonicalLayer.HISTORY,
}


def from_graph_layers_numeral(n: int) -> CanonicalLayer:
    """Map a whole-codebase L1-L6 numeral onto the canonical layer.

    Raises KeyError for any numeral outside 1..6 -- there is no silent
    fallback, per the "unknown edges are dirty edges" discipline both source
    docs use for their own invalidation propagation.
    """
    return GRAPH_LAYERS_L1_L6[n]


def from_graph_incremental_numeral(n: int) -> CanonicalLayer:
    """Map an incremental L0-L5 numeral onto the canonical layer.

    Raises KeyError for any numeral outside 0..5.
    """
    return GRAPH_INCREMENTAL_L0_L5[n]
