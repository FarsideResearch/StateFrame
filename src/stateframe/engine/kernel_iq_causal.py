"""Backward paths from host-observed observables.

The path is a bounded receipt over existing S1 observable, causal-frontier,
and lifecycle edge records.  ``authoritative_edges`` is supplied by those
owners; this module does not persist a replacement dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


SCHEMA = "iq-procedural-causal-path/1"


class CausalContractError(ValueError):
    """Raised for malformed path records."""


class CausalRole(str, Enum):
    OBSERVABLE = "OBSERVABLE"
    PRODUCER = "PRODUCER"
    TRIGGER = "TRIGGER"
    PREREQUISITE = "PREREQUISITE"


class EdgeStrength(str, Enum):
    NECESSARY = "NECESSARY"
    SUFFICIENT = "SUFFICIENT"


class CausalStatus(str, Enum):
    OBSERVED = "OBSERVED"
    UNKNOWN = "UNKNOWN"
    INCOMPLETE = "INCOMPLETE"
    UNREACHABLE = "UNREACHABLE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class CausalNode:
    node_id: str
    role: CausalRole
    owner_ref: str
    reachable: bool = True

    def __post_init__(self) -> None:
        _text(self.node_id, "causal node_id")
        object.__setattr__(self, "role", CausalRole(self.role))
        _text(self.owner_ref, "causal node owner_ref")


@dataclass(frozen=True, slots=True)
class CausalLink:
    edge_id: str
    source_id: str
    target_id: str
    strength: EdgeStrength
    owner_ref: str

    def __post_init__(self) -> None:
        for field_name in ("edge_id", "source_id", "target_id", "owner_ref"):
            _text(getattr(self, field_name), field_name)
        if self.source_id == self.target_id:
            raise CausalContractError("causal link cannot self-reference")
        object.__setattr__(self, "strength", EdgeStrength(self.strength))


@dataclass(frozen=True, slots=True)
class CausalPath:
    observable_id: str
    nodes: tuple[CausalNode, ...]
    links: tuple[CausalLink, ...]
    requirement_ref: str

    def __post_init__(self) -> None:
        _text(self.observable_id, "observable_id")
        _text(self.requirement_ref, "causal requirement_ref")
        nodes = tuple(self.nodes)
        if not nodes:
            raise CausalContractError("causal path must contain nodes")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise CausalContractError("causal path node IDs must be unique")
        if (
            nodes[0].node_id != self.observable_id
            or nodes[0].role is not CausalRole.OBSERVABLE
        ):
            raise CausalContractError(
                "causal path must start at the requested observable"
            )
        links = tuple(self.links)
        if len({link.edge_id for link in links}) != len(links):
            raise CausalContractError("causal link IDs must be unique")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "links", links)


@dataclass(frozen=True, slots=True)
class CausalCheckResult:
    status: CausalStatus
    observable_id: str
    first_missing_boundary: str | None
    necessary_edges: tuple[str, ...]
    sufficient_edges: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status.value,
            "observable_id": self.observable_id,
            "first_missing_boundary": self.first_missing_boundary,
            "necessary_edges": list(self.necessary_edges),
            "sufficient_edges": list(self.sufficient_edges),
            "detail": self.detail,
        }


def validate_causal_path(
    path: CausalPath,
    *,
    authoritative_edges: Mapping[tuple[str, str], str],
    authoritative_owners: Mapping[str, str],
    observed_nodes: Sequence[str],
    observed_edges: Sequence[str],
) -> CausalCheckResult:
    """Validate a backward path against host-owned edge and observation records."""

    nodes = path.nodes
    links = path.links
    observed_node_set = set(observed_nodes)
    observed_edge_set = set(observed_edges)
    necessary = tuple(
        link.edge_id for link in links if link.strength is EdgeStrength.NECESSARY
    )
    sufficient = tuple(
        link.edge_id for link in links if link.strength is EdgeStrength.SUFFICIENT
    )
    expected_roles = (
        CausalRole.OBSERVABLE,
        CausalRole.PRODUCER,
        CausalRole.TRIGGER,
        CausalRole.PREREQUISITE,
    )
    if tuple(node.role for node in nodes) != expected_roles:
        return CausalCheckResult(
            CausalStatus.INCOMPLETE,
            path.observable_id,
            _role_boundary(nodes),
            necessary,
            sufficient,
            "the path does not expose the required observable-to-prerequisite roles",
        )
    if len(links) != len(nodes) - 1:
        return CausalCheckResult(
            CausalStatus.INCOMPLETE,
            path.observable_id,
            _role_boundary(nodes),
            necessary,
            sufficient,
            "a producer, trigger, or prerequisite boundary is missing",
        )
    if path.observable_id not in observed_node_set:
        return CausalCheckResult(
            CausalStatus.UNREACHABLE,
            path.observable_id,
            "observable",
            necessary,
            sufficient,
            "the requested observable has no host observation",
        )
    for node in nodes:
        if not node.reachable:
            return CausalCheckResult(
                CausalStatus.UNREACHABLE,
                path.observable_id,
                node.node_id,
                necessary,
                sufficient,
                f"causal node {node.node_id} is unreachable",
            )
        if not node.owner_ref:
            return CausalCheckResult(
                CausalStatus.INVALID,
                path.observable_id,
                node.node_id,
                necessary,
                sufficient,
                "causal node has no owner",
            )
    for index, link in enumerate(links):
        target = nodes[index]
        source = nodes[index + 1]
        boundary = f"{target.node_id}<-{source.node_id}"
        if link.source_id != source.node_id or link.target_id != target.node_id:
            return CausalCheckResult(
                CausalStatus.INVALID,
                path.observable_id,
                boundary,
                necessary,
                sufficient,
                "causal link direction does not match the backward path",
            )
        authoritative_id = authoritative_edges.get((source.node_id, target.node_id))
        if authoritative_id != link.edge_id:
            return CausalCheckResult(
                CausalStatus.INCOMPLETE,
                path.observable_id,
                boundary,
                necessary,
                sufficient,
                "host has no authoritative edge for this proposed cause",
            )
        if authoritative_owners.get(link.edge_id) != link.owner_ref:
            return CausalCheckResult(
                CausalStatus.INVALID,
                path.observable_id,
                boundary,
                necessary,
                sufficient,
                "causal edge owner does not match the host record",
            )
        if link.edge_id in necessary and link.edge_id not in observed_edge_set:
            return CausalCheckResult(
                CausalStatus.UNKNOWN,
                path.observable_id,
                boundary,
                necessary,
                sufficient,
                "necessary edge is not backed by an observed receipt",
            )
    return CausalCheckResult(
        CausalStatus.OBSERVED,
        path.observable_id,
        None,
        necessary,
        sufficient,
        "observable has a host-owned, observed causal route",
    )


def _role_boundary(nodes: Sequence[CausalNode]) -> str | None:
    expected = (
        CausalRole.OBSERVABLE,
        CausalRole.PRODUCER,
        CausalRole.TRIGGER,
        CausalRole.PREREQUISITE,
    )
    if len(nodes) >= len(expected):
        return None
    return (
        f"{nodes[-1].role.value.lower()}<-{expected[len(nodes)].value.lower()}"
        if nodes
        else "observable"
    )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalContractError(f"{field_name} must be non-empty")
    return value.strip()


__all__ = [
    "CausalCheckResult",
    "CausalContractError",
    "CausalLink",
    "CausalNode",
    "CausalPath",
    "CausalRole",
    "CausalStatus",
    "EdgeStrength",
    "SCHEMA",
    "validate_causal_path",
]
