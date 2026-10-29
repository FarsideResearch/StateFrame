"""Typed host-receipt adapter for the bounded MicroGraph owner.

Only explicit receipt fields are admitted. Model notes, observations, detail,
intent and trace text are never parsed or copied into graph state.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

from detreason.micrograph import GraphDelta, MicroGraph, make_edge, make_node


SCHEMA = "stateframe-micrograph-wire/1"
FIELD = "micrograph"
MAX_NODES = 256
MAX_EDGES = 384
MAX_ROWS_PER_FRAME = 8
MAX_PATHS_PER_RECEIPT = 4
_KINDS = frozenset({"read", "edit", "test"})
_RESULTS = {
    "read": frozenset({"OK", "READ", "SUCCESS"}),
    "edit": frozenset({"APPLIED"}),
    "test": frozenset({"RED", "GREEN"}),
}
_WEIGHTS = {"RED": 5, "APPLIED": 3, "OK": 1, "READ": 1, "SUCCESS": 1, "GREEN": 0}


def mode() -> str:
    value = os.environ.get("STATEFRAME_MICROGRAPH", "off").strip().lower()  # default off; STATEFRAME_MICROGRAPH=on enables
    if value not in {"off", "on"}:
        raise ValueError("STATEFRAME_MICROGRAPH must be off or on")
    return value


_SAFE_PATH = re.compile(r"[A-Za-z0-9_./+@-]+\Z")


def _path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.replace("\\", "/").strip()
    if (not path or len(path) > 256 or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or _SAFE_PATH.fullmatch(path) is None):
        return None
    return path


def _paths(receipt: Mapping[str, object], kind: str) -> tuple[str, ...]:
    if kind != "test":
        values = (receipt.get("target_path"), receipt.get("path"))
    elif receipt.get("result_class", "").upper() == "RED":
        # A suite selector names work requested, not the test that failed.
        # Only individually classified host facts can locate a RED receipt.
        rows = receipt.get("failed_paths")
        values = tuple(row.get("path") for row in rows[:MAX_PATHS_PER_RECEIPT]
                       if isinstance(row, Mapping) and row.get("result_class") == "RED") \
            if isinstance(rows, (tuple, list)) else ()
    else:
        rows = receipt.get("source_paths")
        values = tuple(rows[:MAX_PATHS_PER_RECEIPT]) if isinstance(rows, (tuple, list)) else ()
    return tuple(sorted({path for value in values
                         if (path := _path(value)) is not None
                         and (kind != "test" or path.endswith(".py"))}))[:MAX_PATHS_PER_RECEIPT]


def _revision(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else -1


def individual_failed_paths(rows: object) -> list[dict[str, str]]:
    """Locate only a failed canonical check scoped to one typed selector."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    paths: set[str] = set()
    for row in rows[:32]:
        if not isinstance(row, Mapping) or row.get("outcome") != "BEHAVIOR_FAIL":
            continue
        scope = row.get("selector_scope")
        if not isinstance(scope, (tuple, list)) or len(scope) != 1:
            continue
        selector = scope[0]
        if isinstance(selector, str):
            path = _path(selector.split("::", 1)[0])
            if path is not None and path.endswith(".py"):
                paths.add(path)
    return [{"path": path, "result_class": "RED"}
            for path in sorted(paths)[:MAX_PATHS_PER_RECEIPT]]


class MicroGraphWire:
    """Checkpointable graph plus bounded stream cursors; MicroGraph owns all records."""

    def __init__(self, task_id: str) -> None:
        self.graph = MicroGraph("stateframe", task_id, max_nodes=MAX_NODES,
                                max_edges=MAX_EDGES)
        self.cursors: dict[str, int] = {}
        self.skipped = 0
        self.next_ordinal = 0

    def checkpoint(self) -> dict[str, object]:
        return {"schema": SCHEMA, "graph": self.graph.to_dict(),
                "cursors": dict(sorted(self.cursors.items())),
                "skipped": self.skipped, "next_ordinal": self.next_ordinal}

    @classmethod
    def from_checkpoint(cls, value: object, task_id: str) -> "MicroGraphWire":
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "graph", "cursors", "skipped", "next_ordinal"} \
                or value.get("schema") != SCHEMA:
            raise ValueError("invalid MicroGraph checkpoint")
        raw_graph = value["graph"]
        cursors = value["cursors"]
        if (not isinstance(raw_graph, Mapping)
                or not isinstance(cursors, Mapping) or set(cursors) - {"actions", "tests", "journal"}
                or any(type(count) is not int or count < 0 for count in cursors.values())
                or any(type(value.get(key)) is not int or value[key] < 0
                       for key in ("skipped", "next_ordinal"))
                or len(raw_graph.get("nodes", ())) > MAX_NODES
                or len(raw_graph.get("edges", ())) > MAX_EDGES):
            raise ValueError("invalid MicroGraph checkpoint state")
        graph = MicroGraph.from_dict(raw_graph)
        if (graph.workspace_id != "stateframe" or graph.case_id != task_id
                or graph.max_nodes != MAX_NODES or graph.max_edges != MAX_EDGES):
            raise ValueError("MicroGraph checkpoint scope or bounds mismatch")
        wire = cls(task_id)
        wire.graph = graph
        wire.cursors = dict(cursors)
        wire.skipped = value["skipped"]
        wire.next_ordinal = value["next_ordinal"]
        return wire

    def observe(self, raw: object) -> tuple[dict[str, object], ...]:
        receipt = raw.get("receipt", raw) if isinstance(raw, Mapping) else None
        if not isinstance(receipt, Mapping):
            self.skipped += 1
            return ()
        kind = receipt.get("kind")
        result = receipt.get("result_class")
        evidence_id = receipt.get("receipt_id", raw.get("event_id") if isinstance(raw, Mapping) else None)
        if (not isinstance(kind, str) or kind not in _KINDS or not isinstance(result, str)
                or result.upper() not in _RESULTS[kind]
                or not isinstance(evidence_id, str) or not 1 <= len(evidence_id) <= 128
                or any(ord(char) < 33 or ord(char) > 126 for char in evidence_id)):
            self.skipped += 1
            return ()
        if kind == "test" and "exit_code" in receipt:
            expected = 0 if result.upper() == "GREEN" else 1
            if type(receipt["exit_code"]) is not int or receipt["exit_code"] != expected:
                self.skipped += 1
                return ()
        try:
            paths = _paths(receipt, kind)
        except (AttributeError, TypeError, ValueError):
            self.skipped += 1
            return ()
        if not paths:
            self.skipped += 1
            return ()
        result = result.upper()
        outputs: list[dict[str, object]] = []
        for path in paths:
            try:
                revision = _revision(receipt.get("workspace_revision"))
                evidence_key = f"receipt:{evidence_id}:{kind}:{path}"
                evidence_identity = make_node("evidence", evidence_key).id
                previous = self.graph.node(evidence_identity)
                if previous is not None:
                    if (previous.attrs.get("result") != result
                            or previous.attrs.get("revision") != revision):
                        self.skipped += 1
                    continue
                self.next_ordinal += 1
                retention = self.next_ordinal * 10 + _WEIGHTS[result]
                # One location per evidence keeps endpoint and evidence eviction
                # priorities together. Projection folds the bounded set by path.
                location = make_node("value", f"path:{path}:{evidence_id}:{kind}",
                                     {"path": path, "priority": retention + 0.5})
                evidence = make_node("evidence", f"receipt:{evidence_id}:{kind}:{path}", {
                    "receipt_id": evidence_id, "kind": kind, "result": result,
                    "path": path, "priority": retention, "weight": _WEIGHTS[result],
                    "revision": revision, "ordinal": self.next_ordinal,
                })
                relation = make_edge("suspects", evidence.id, location.id)
                self.graph.apply_delta(GraphDelta((location, evidence), (relation,)))
                outputs.append({"path": path, "evidence_id": evidence.id, "result": result})
            except (TypeError, ValueError, KeyError, OverflowError):
                self.skipped += 1
        return tuple(outputs)

    def observe_stream(self, name: str, rows: object) -> tuple[dict[str, object], ...]:
        if (not isinstance(name, str) or name not in {"actions", "tests", "journal"}
                or not isinstance(rows, Sequence)
                or isinstance(rows, (str, bytes, bytearray))):
            self.skipped += 1
            return ()
        start = min(self.cursors.get(name, 0), len(rows))
        emitted: list[dict[str, object]] = []
        end = min(len(rows), start + MAX_ROWS_PER_FRAME)
        for index in range(start, end):
            emitted.extend(self.observe(rows[index]))
        self.cursors[name] = end
        return tuple(emitted)

    def projection(self) -> dict[str, object]:
        by_path: dict[str, list[object]] = {}
        linked = {edge.source for edge in self.graph.edges if edge.kind == "suspects"}
        for node in self.graph.nodes:
            if node.kind == "evidence" and node.id in linked:
                path = node.attrs.get("path")
                if isinstance(path, str):
                    by_path.setdefault(path, []).append(node)
        suspects = []
        for path, evidence in by_path.items():
            valid = [node for node in evidence
                     if type(node.attrs.get("weight")) is int
                     and type(node.attrs.get("revision")) is int
                     and type(node.attrs.get("ordinal")) is int
                     and node.attrs.get("result") in _WEIGHTS]
            if not valid:
                continue
            revision = max(node.attrs["revision"] for node in valid)
            current = [node for node in valid if node.attrs["revision"] == revision]
            latest_green = max((node.attrs["ordinal"] for node in current
                                if node.attrs["result"] == "GREEN"), default=-1)
            latest_edit = max((node.attrs["ordinal"] for node in current
                               if node.attrs["result"] == "APPLIED"), default=-1)
            ordered = sorted((node for node in current
                              if node.attrs["ordinal"] > latest_green
                              and (node.attrs["result"] != "RED"
                                   or node.attrs["ordinal"] > latest_edit)),
                             key=lambda node: (-node.attrs["weight"],
                                               -node.attrs["ordinal"], node.id))
            if not ordered or ordered[0].attrs["weight"] == 0:
                continue
            suspects.append({"path": path, "score": max(node.attrs["weight"] for node in ordered),
                             "evidence_ids": [node.id for node in ordered[:4]]})
        suspects.sort(key=lambda row: (-row["score"], row["path"]))
        return {"schema": SCHEMA, "suspects": suspects[:5],
                "node_count": self.graph.node_count, "edge_count": self.graph.edge_count,
                "skipped": self.skipped}
