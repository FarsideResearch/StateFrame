"""Bounded plan records and host-checked joins.

The plan is a small projection over existing lifecycle, work, decision, and
invalidation owners.  It does not schedule those owners itself; it only
validates the three-node seam and records host receipts needed for a root
settlement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from kernel_iq_relations import units_compatible


SCHEMA = "iq-procedural-plan/1"


class PlanContractError(ValueError):
    """Raised for malformed plans, cycles, or invalid host state transitions."""


class NodeStatus(str, Enum):
    PLANNED = "PLANNED"
    GREEN = "GREEN"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"


class SettlementStatus(str, Enum):
    ADMITTED = "ADMITTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    type_name: str
    unit: str | None = None
    optional: bool = False
    cardinality: str = "one"

    def __post_init__(self) -> None:
        _text(self.name, "port name")
        _text(self.type_name, "port type")
        if self.unit:
            _text(self.unit, "port unit")
        if self.cardinality not in {"one", "many"}:
            raise PlanContractError("port cardinality must be one or many")


@dataclass(frozen=True, slots=True)
class PlanNode:
    node_id: str
    role: str
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    requirement_refs: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    seam_id: str = ""
    assumptions: tuple[str, ...] = ()
    preservation_refs: tuple[str, ...] = ()
    basis_revision: str = ""
    active: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        _text(self.node_id, "node_id")
        if self.role not in {"root", "leaf", "transform"}:
            raise PlanContractError("node role must be root, leaf, or transform")
        for field_name in (
            "requirement_refs",
            "preconditions",
            "assumptions",
            "preservation_refs",
        ):
            object.__setattr__(
                self, field_name, _text_tuple(getattr(self, field_name), field_name)
            )
        if self.active and not self.seam_id:
            raise PlanContractError(
                f"active node {self.node_id} must expose a checkable seam"
            )
        if self.role != "root" and self.required and not self.requirement_refs:
            raise PlanContractError(
                f"required node {self.node_id} must reference a requirement"
            )
        _unique_ports(self.inputs, f"{self.node_id} inputs")
        _unique_ports(self.outputs, f"{self.node_id} outputs")


@dataclass(frozen=True, slots=True)
class JoinSpec:
    join_id: str
    producer_id: str
    output_name: str
    consumer_id: str
    input_name: str
    shared_assumptions: tuple[str, ...] = ()
    preservation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "join_id",
            "producer_id",
            "output_name",
            "consumer_id",
            "input_name",
        ):
            _text(getattr(self, field_name), field_name)
        if self.producer_id == self.consumer_id:
            raise PlanContractError("a join cannot connect a node to itself")
        object.__setattr__(
            self,
            "shared_assumptions",
            _text_tuple(self.shared_assumptions, "shared assumption"),
        )
        object.__setattr__(
            self,
            "preservation_refs",
            _text_tuple(self.preservation_refs, "join preservation"),
        )


@dataclass(frozen=True, slots=True)
class BoundedPlan:
    root_id: str
    nodes: tuple[PlanNode, ...]
    joins: tuple[JoinSpec, ...] = ()
    plan_revision: str = "plan-1"
    authority_refs: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _text(self.root_id, "root_id")
        _text(self.plan_revision, "plan_revision")
        nodes = tuple(self.nodes)
        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise PlanContractError("plan node IDs must be unique")
        if self.root_id not in by_id or by_id[self.root_id].role != "root":
            raise PlanContractError("plan root_id must name a root node")
        active = tuple(node for node in nodes if node.active)
        if len(active) > 3:
            raise PlanContractError(
                "a bounded plan may have at most three active nodes"
            )
        joins = tuple(self.joins)
        if len({join.join_id for join in joins}) != len(joins):
            raise PlanContractError("join IDs must be unique")
        for join in joins:
            producer = by_id.get(join.producer_id)
            consumer = by_id.get(join.consumer_id)
            if producer is None or consumer is None:
                raise PlanContractError(
                    f"join {join.join_id} references an unknown node"
                )
            if not _port(producer.outputs, join.output_name) or not _port(
                consumer.inputs, join.input_name
            ):
                raise PlanContractError(
                    f"join {join.join_id} references an unknown port"
                )
        _assert_acyclic(joins)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "joins", joins)
        object.__setattr__(self, "authority_refs", dict(self.authority_refs or {}))

    def node(self, node_id: str) -> PlanNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise PlanContractError(f"unknown plan node: {node_id}")

    def join_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        for join in self.joins:
            producer = self.node(join.producer_id)
            consumer = self.node(join.consumer_id)
            output = _port(producer.outputs, join.output_name)
            input_port = _port(consumer.inputs, join.input_name)
            assert output is not None and input_port is not None
            if output.type_name != input_port.type_name:
                blockers.append(f"{join.join_id}: incompatible types")
            if not units_compatible(output.unit, input_port.unit):
                blockers.append(f"{join.join_id}: incompatible units")
            if output.cardinality == "many" and input_port.cardinality == "one":
                blockers.append(f"{join.join_id}: many output cannot feed one input")
            if output.optional and not input_port.optional:
                blockers.append(f"{join.join_id}: optional output feeds required input")
            assumptions = set(join.shared_assumptions)
            if not assumptions.issubset(
                set(producer.assumptions) & set(consumer.assumptions)
            ):
                blockers.append(
                    f"{join.join_id}: shared assumptions are not common to both nodes"
                )
            if not set(join.preservation_refs).issubset(
                set(producer.preservation_refs) & set(consumer.preservation_refs)
            ):
                blockers.append(
                    f"{join.join_id}: preservation seam is not owned by both nodes"
                )
        return tuple(blockers)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "root_id": self.root_id,
            "plan_revision": self.plan_revision,
            "authority_refs": dict(self.authority_refs or {}),
            "nodes": [_node_dict(node) for node in self.nodes],
            "joins": [_join_dict(join) for join in self.joins],
        }


@dataclass(frozen=True, slots=True)
class NodeReceipt:
    node_id: str
    basis_revision: str
    status: NodeStatus
    outputs: Mapping[str, object]
    artifact_ready: bool
    behavior_pass: bool
    assumptions: tuple[str, ...] = ()
    preserved_relations: tuple[str, ...] = ()
    serialized: bool = True
    end_to_end: bool = False

    def __post_init__(self) -> None:
        _text(self.node_id, "receipt node_id")
        _text(self.basis_revision, "receipt basis_revision")
        object.__setattr__(self, "status", NodeStatus(self.status))
        object.__setattr__(
            self, "assumptions", _text_tuple(self.assumptions, "receipt assumption")
        )
        object.__setattr__(
            self,
            "preserved_relations",
            _text_tuple(self.preserved_relations, "preserved relation"),
        )
        if not isinstance(self.outputs, Mapping):
            raise PlanContractError("receipt outputs must be a mapping")


@dataclass(frozen=True, slots=True)
class RootSettlement:
    status: SettlementStatus
    root_id: str
    blockers: tuple[str, ...]
    receipt_ids: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.status is SettlementStatus.ADMITTED

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "iq-procedural-root-settlement/1",
            "status": self.status.value,
            "root_id": self.root_id,
            "blockers": list(self.blockers),
            "receipt_ids": list(self.receipt_ids),
        }


InvalidationSink = Callable[[str, tuple[str, ...]], None]


class PlanHost:
    """Host state machine for scheduling, invalidation, restart, and root join."""

    def __init__(
        self, plan: BoundedPlan, *, invalidation_sink: InvalidationSink | None = None
    ) -> None:
        self.plan = plan
        self._receipts: dict[str, NodeReceipt] = {}
        self._invalidated: dict[str, tuple[str, ...]] = {}
        self._invalidation_sink = invalidation_sink

    def schedule(self) -> tuple[str, ...]:
        blockers = self.plan.join_blockers()
        if blockers:
            return ()
        ready: list[str] = []
        for node in self.plan.nodes:
            if (
                not node.active
                or node.node_id in self._receipts
                or node.node_id in self._invalidated
            ):
                continue
            incoming = [
                join for join in self.plan.joins if join.consumer_id == node.node_id
            ]
            if all(
                self._receipts.get(join.producer_id) is not None
                and self._receipts[join.producer_id].status is NodeStatus.GREEN
                for join in incoming
            ):
                ready.append(node.node_id)
        return tuple(ready)

    def settle(self, receipt: NodeReceipt) -> None:
        node = self.plan.node(receipt.node_id)
        if receipt.node_id in self._invalidated:
            raise PlanContractError(
                "invalidated node cannot settle without a new plan revision"
            )
        if receipt.basis_revision != node.basis_revision and node.basis_revision:
            raise PlanContractError("receipt basis revision does not match node")
        if receipt.status is NodeStatus.GREEN and not receipt.serialized:
            raise PlanContractError("green node receipt must be serializable")
        self._receipts[receipt.node_id] = receipt

    def invalidate(self, node_id: str, changed_basis: Sequence[str]) -> tuple[str, ...]:
        self.plan.node(node_id)
        reasons = _text_tuple(tuple(changed_basis), "changed basis")
        if not reasons:
            raise PlanContractError("invalidation requires at least one changed basis")
        affected: set[str] = set()
        pending = [node_id]
        while pending:
            current = pending.pop()
            if current in affected:
                continue
            affected.add(current)
            pending.extend(
                join.consumer_id
                for join in self.plan.joins
                if join.producer_id == current
            )
        for affected_id in affected:
            self._invalidated[affected_id] = reasons
            self._receipts.pop(affected_id, None)
            if self._invalidation_sink is not None:
                self._invalidation_sink(affected_id, reasons)
        return tuple(sorted(affected))

    def root_settlement(self) -> RootSettlement:
        blockers = list(self.plan.join_blockers())
        if self._invalidated:
            blockers.append("dependent nodes are invalidated")
        active_required = [
            node for node in self.plan.nodes if node.active and node.required
        ]
        for node in active_required:
            receipt = self._receipts.get(node.node_id)
            if receipt is None:
                blockers.append(f"{node.node_id}: no receipt")
                continue
            if receipt.status is not NodeStatus.GREEN or not receipt.behavior_pass:
                blockers.append(f"{node.node_id}: behavior is not green")
            if not receipt.artifact_ready:
                blockers.append(f"{node.node_id}: artifact is not ready")
            if not receipt.serialized:
                blockers.append(f"{node.node_id}: artifact is not serialized")
        for join in self.plan.joins:
            producer = self._receipts.get(join.producer_id)
            consumer = self._receipts.get(join.consumer_id)
            if producer is None or consumer is None:
                continue
            if not set(join.shared_assumptions).issubset(
                set(producer.assumptions) & set(consumer.assumptions)
            ):
                blockers.append(f"{join.join_id}: receipt assumptions do not join")
            if not set(join.preservation_refs).issubset(
                set(producer.preserved_relations) & set(consumer.preserved_relations)
            ):
                blockers.append(f"{join.join_id}: preservation receipt is missing")
        root = self._receipts.get(self.plan.root_id)
        if root is not None and not root.end_to_end:
            blockers.append("root: end-to-end outcome is absent")
        status = SettlementStatus.ADMITTED if not blockers else SettlementStatus.BLOCKED
        return RootSettlement(
            status,
            self.plan.root_id,
            tuple(dict.fromkeys(blockers)),
            tuple(sorted(self._receipts)),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "iq-procedural-plan-snapshot/1",
            "plan": self.plan.to_dict(),
            "plan_digest": _digest(self.plan.to_dict()),
            "receipts": {
                node_id: _receipt_dict(receipt)
                for node_id, receipt in self._receipts.items()
            },
            "invalidated": {
                node_id: list(reasons) for node_id, reasons in self._invalidated.items()
            },
        }

    @classmethod
    def restore(
        cls,
        plan: BoundedPlan,
        snapshot: Mapping[str, object],
        *,
        invalidation_sink: InvalidationSink | None = None,
    ) -> "PlanHost":
        if snapshot.get("schema") != "iq-procedural-plan-snapshot/1":
            raise PlanContractError("unsupported plan snapshot")
        if snapshot.get("plan_digest") != _digest(plan.to_dict()):
            raise PlanContractError(
                "plan snapshot belongs to a different plan revision"
            )
        host = cls(plan, invalidation_sink=invalidation_sink)
        raw_receipts = snapshot.get("receipts", {})
        if not isinstance(raw_receipts, Mapping):
            raise PlanContractError("snapshot receipts must be a mapping")
        for value in raw_receipts.values():
            if not isinstance(value, Mapping):
                raise PlanContractError("snapshot receipt must be a mapping")
            host.settle(
                NodeReceipt(
                    str(value["node_id"]),
                    str(value["basis_revision"]),
                    NodeStatus(str(value["status"])),
                    value.get("outputs", {}),
                    bool(value["artifact_ready"]),
                    bool(value["behavior_pass"]),
                    tuple(str(item) for item in value.get("assumptions", ())),
                    tuple(str(item) for item in value.get("preserved_relations", ())),
                    bool(value.get("serialized", True)),
                    bool(value.get("end_to_end", False)),
                )
            )
        raw_invalidated = snapshot.get("invalidated", {})
        if isinstance(raw_invalidated, Mapping):
            host._invalidated = {
                str(node_id): tuple(str(item) for item in reasons)
                for node_id, reasons in raw_invalidated.items()
            }
        return host


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanContractError(f"{field_name} must be non-empty")
    return value.strip()


def _text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise PlanContractError(f"{field_name} must be a sequence")
    return tuple(dict.fromkeys(_text(value, field_name) for value in values))


def _unique_ports(ports: Sequence[PortSpec], field_name: str) -> None:
    if len({port.name for port in ports}) != len(tuple(ports)):
        raise PlanContractError(f"{field_name} contain duplicate names")


def _port(ports: Sequence[PortSpec], name: str) -> PortSpec | None:
    return next((port for port in ports if port.name == name), None)


def _assert_acyclic(joins: Sequence[JoinSpec]) -> None:
    adjacency: dict[str, list[str]] = {}
    for join in joins:
        adjacency.setdefault(join.producer_id, []).append(join.consumer_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise PlanContractError("plan joins must be acyclic")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in adjacency.get(node_id, ()):
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in tuple(adjacency):
        visit(node_id)


def _node_dict(node: PlanNode) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "role": node.role,
        "inputs": [_port_dict(port) for port in node.inputs],
        "outputs": [_port_dict(port) for port in node.outputs],
        "requirement_refs": list(node.requirement_refs),
        "preconditions": list(node.preconditions),
        "seam_id": node.seam_id,
        "assumptions": list(node.assumptions),
        "preservation_refs": list(node.preservation_refs),
        "basis_revision": node.basis_revision,
        "active": node.active,
        "required": node.required,
    }


def _port_dict(port: PortSpec) -> dict[str, object]:
    return {
        "name": port.name,
        "type_name": port.type_name,
        "unit": port.unit,
        "optional": port.optional,
        "cardinality": port.cardinality,
    }


def _join_dict(join: JoinSpec) -> dict[str, object]:
    return {
        "join_id": join.join_id,
        "producer_id": join.producer_id,
        "output_name": join.output_name,
        "consumer_id": join.consumer_id,
        "input_name": join.input_name,
        "shared_assumptions": list(join.shared_assumptions),
        "preservation_refs": list(join.preservation_refs),
    }


def _receipt_dict(receipt: NodeReceipt) -> dict[str, object]:
    return {
        "node_id": receipt.node_id,
        "basis_revision": receipt.basis_revision,
        "status": receipt.status.value,
        "outputs": dict(receipt.outputs),
        "artifact_ready": receipt.artifact_ready,
        "behavior_pass": receipt.behavior_pass,
        "assumptions": list(receipt.assumptions),
        "preserved_relations": list(receipt.preserved_relations),
        "serialized": receipt.serialized,
        "end_to_end": receipt.end_to_end,
    }


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BoundedPlan",
    "JoinSpec",
    "NodeReceipt",
    "NodeStatus",
    "PlanContractError",
    "PlanHost",
    "PlanNode",
    "PortSpec",
    "RootSettlement",
    "SCHEMA",
    "SettlementStatus",
]
