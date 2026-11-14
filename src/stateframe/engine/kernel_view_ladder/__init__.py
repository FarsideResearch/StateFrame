"""View ladder: compressed views of delivered reads (full / range,relevant / symbols,outline / stub).

Pure package: rendering (`render`), structure maps (`structure`), artifact state plus the
measured rung choice (`state`) and the relevance selection that feeds the `relevant` rung
(`relevance`). Nothing on the shipped request path imports it yet; the diet, finalizer and
dispatch wiring is a separate item and carries the `STATEFRAME_VIEW_LADDER` switch.
"""
from kernel_view_ladder.relevance import (  # noqa: F401
    EDGE_OBSERVED,
    EDGE_RESOLVED,
    EDGE_UNRESOLVED,
    EDGES,
    KINDS,
    RELEVANCE_VERSION,
    Definition,
    Seed,
    Selection,
    fallback_selection,
    frozen_selection,
    omitted_ranges,
    relevant_selection,
    seeds_from_runtime,
    select,
    selection_bytes,
    selection_signature,
)
from kernel_view_ladder.render import (  # noqa: F401
    DEGRADED_RUNGS,
    READ_VIEW_HEADER,
    REASONS,
    RENDERER_VERSION,
    RUNGS,
    VIEW_LADDER_ENV,
    Artifact,
    Rendered,
    applicable_rungs,
    change_notice,
    enabled,
    is_degraded_view,
    iter_verbatim_rows,
    parse_view_header,
    range_marker,
    render_stub,
    render_stub_line,
    render_view,
    row_matches_slice,
    split_delivered_lines,
    verify_body_sha256,
    view_id,
)
from kernel_view_ladder.state import (  # noqa: F401
    ArtifactKey,
    ArtifactState,
    Keep,
    ViewLadderState,
    candidate_rungs,
    pick_rung,
)
