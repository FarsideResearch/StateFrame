"""Per-send visibility for repository dependency facts in the product loop."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from mc_repo_intel_vendor import background

SUFFIX = ".repo-dependencies.json"


class DependencyView:
    """Pin one completed graph per exchange and tape only the facts consumed.

    The sealed send opens the window for the next request. Host-observed
    writes retire its graph immediately. During replay the next send's tape
    is authoritative, and neither worker state nor readiness is restored.
    """

    def __init__(self, replay_request: Callable[[], Path | None]):
        self._replay_request = replay_request
        self._root: Path | None = None
        self._graph = None
        self._current = None
        self._rows: dict[tuple[str, int], dict] = {}
        self._loaded = False
        self._recorded = False

    def advance(self, root: Path | None) -> None:
        self._rows.clear()
        self._loaded = self._recorded = False
        self._root = root
        self._graph, self._current = None, None
        if root is not None:
            # Sample before scheduling. Even a zero-cost job waits until the
            # next boundary to influence a request.
            self._graph, self._current = background.graph_snapshot(root)
            background.request_graph(root)

    def _load_rows(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        request = self._replay_request()
        if request is None:
            return
        self._recorded = True
        record = request.with_name(request.name.removesuffix(".request.json") + SUFFIX)
        if not record.exists():
            return
        from kernel_tool_tape import ToolReplayRefused

        try:
            rows = json.loads(record.read_bytes())
            for row in rows:
                path, revision = row["path"], row["revision"]
                deps = row["dependencies"]
                if (not isinstance(path, str) or type(revision) is not int
                        or not isinstance(deps, list) or any(not isinstance(d, str) for d in deps)
                        or row["source"] not in {"graph", "static", "scoped"}
                        or (path, revision) in self._rows):
                    raise ValueError("invalid dependency fact")
                self._rows[path, revision] = row
        except (OSError, KeyError, TypeError, ValueError, RecursionError) as error:
            raise ToolReplayRefused("recorded repository dependencies are invalid") from error

    def rows_for_send(self) -> list[dict]:
        self._load_rows()
        return list(self._rows.values())

    def fact(self, root: Path, path: str, revision: int) -> tuple[tuple[str, ...], bool]:
        self._load_rows()
        key = (path, revision)
        row = self._rows.get(key)
        if row is None:
            graph = self._graph
            if (not self._recorded and graph is not None and root == self._root
                    and self._current()):
                dependencies = graph.dependency_files(path)
                source = "graph" if getattr(graph, "index_status", "ready") == "ready" else "static"
            else:
                from mc_repo_intel_vendor.call_graph import scoped_test_dependencies

                dependencies = scoped_test_dependencies(root, path)
                source = "scoped"
            row = self._rows[key] = {
                "path": path, "revision": revision, "dependencies": list(dependencies), "source": source,
            }
        return tuple(row["dependencies"]), row["source"] == "graph"


# New-process authority, intentionally outside runtime checkpoints.
active: DependencyView | None = None
