"""graphs — multi-layer whole-codebase comprehension graph, v1.

Six named layers on ONE canonical entity spine:

    ARTIFACT  (folded into the entity registry itself)
    STRUCTURE full build      graphs/structure.py
    DATAFLOW  LAZY per module graphs/dataflow.py
    SEAM      adapter-wrapped graphs/seam.py     -> detreason/seam_adapter.py
    RUNTIME   not built in v1  (model.KNOWN_GAPS)
    HISTORY   sampled window  graphs/history.py
    INTENT    full build      graphs/intent.py

Canon this package IMPORTS rather than re-founds:

    detreason/layer_registry.py   the ONE layer enum (W0-6, named levels)
    architecture/domain_ids.py    the ONE entity-ID scheme
    detreason/seam_adapter.py     the ONE seam producer (W0-7)

Nothing here parses a seam, re-numbers a layer, or mints an ID by hand.

Entry points::

    from graphs.build import build_repo, build_all
    from graphs.queries import all_layer_context, callers_of
    from graphs.store import GraphStore
"""

from graphs.model import (  # noqa: F401
    ALL_LAYERS,
    CORE_SIX,
    KNOWN_GAPS,
    CanonicalLayer,
    Edge,
    Evidence,
    Freshness,
    KnowledgeCut,
    LayerGraph,
    Node,
    Receipt,
)

__all__ = [
    "ALL_LAYERS",
    "CORE_SIX",
    "CanonicalLayer",
    "Edge",
    "Evidence",
    "Freshness",
    "KNOWN_GAPS",
    "KnowledgeCut",
    "LayerGraph",
    "Node",
    "Receipt",
]
