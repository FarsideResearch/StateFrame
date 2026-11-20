"""Per-send visibility for repository analysis in the product loop."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from repo_intel_core import background

SUFFIX = ".repo-dependencies.json"
ANALYSIS_SUFFIX = ".repository-analysis.json"
_ANALYSES = ("claim", "census")


def enabled() -> bool:
    return os.environ.get("STATEFRAME_BACKGROUND_REPOSITORY_ANALYSIS", "off") == "on"


def _produce_analysis(root: Path, kind: str):
    if kind == "claim":
        from kernel_claim_grounding import build_ast_index

        return build_ast_index(root)
    from kernel_callsite_census import build_background_census

    return build_background_census(root)


def _request_analysis(root: Path, kind: str) -> None:
    def work(publish, current):
        if current():
            publish(_produce_analysis(root, kind))

    background.request(root, kind, work)


class DependencyView:
    """Pin completed analysis per exchange and tape the versions and facts used.

    The sealed send opens the window for the next request. Host-observed
    writes retire its snapshots immediately. During replay the next send's tape
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
        self._ready = {}
        self._analysis_rows: dict[str, dict[str, list[str]]] | None = None
        self._analysis_positions: dict[tuple[str, str], int] = {}

    def advance(self, root: Path | None) -> None:
        self._rows.clear()
        self._loaded = self._recorded = False
        self._root = root
        self._graph, self._current = None, None
        self._ready.clear()
        self._analysis_rows = None
        self._analysis_positions.clear()
        if root is not None:
            # Sample before scheduling. Even a zero-cost job waits until the
            # next boundary to influence a request.
            self._graph, self._current = background.graph_snapshot(root)
            self._ready.update((kind, background.snapshot(root, kind)) for kind in _ANALYSES)
            background.request_graph(root)
            if (os.environ.get("STATEFRAME_CLAIM_GROUNDING", "off") == "on"
                    and os.environ.get("STATEFRAME_EXISTENCE_ORACLE", "off") != "on"):
                _request_analysis(root, "claim")
            if (os.environ.get("STATEFRAME_CENSUS_INDEX_REUSE", "on") == "on"
                    and any(os.environ.get(flag, "on") != "off" for flag in
                            ("STATEFRAME_CALLSITE_CENSUS", "STATEFRAME_CALLSITE_CENSUS_EDIT"))):
                _request_analysis(root, "census")

    def analysis_rows_for_send(self) -> dict[str, dict[str, list[str]]]:
        if self._analysis_rows is None:
            self._analysis_rows = {}
            request = self._replay_request()
            if request is not None:
                record = request.with_name(request.name.removesuffix(".request.json") + ANALYSIS_SUFFIX)
                from kernel_tool_tape import ToolReplayRefused

                try:
                    rows = json.loads(record.read_bytes())
                    if not isinstance(rows, dict) or any(
                        kind not in _ANALYSES or not isinstance(choices, dict)
                        or any(
                            not isinstance(modes, list) or not modes
                            or any(mode not in ("full", "scoped") for mode in modes)
                            for modes in choices.values()
                        )
                        for kind, choices in rows.items()
                    ):
                        raise ValueError("invalid repository analysis versions")
                    self._analysis_rows = rows
                except FileNotFoundError:
                    pass  # older runs used the synchronous analysis
                except (OSError, TypeError, ValueError, RecursionError) as error:
                    raise ToolReplayRefused("recorded repository analysis is invalid") from error
        return self._analysis_rows

    def analysis(self, root: Path, kind: str, key: str,
                 *, ready: Callable[[object], bool] = lambda _: True) -> tuple[str, object | None]:
        """Pin a version for the facts this request seeks, including during replay."""
        tape = self.analysis_rows_for_send()
        value, current = self._ready.get(kind, (None, None)) if self._root == root else (None, None)
        if self._replay_request() is not None:
            modes = tape.get(kind, {}).get(key)
            if modes is None:
                return ("scoped" if tape else "legacy"), None
            position = self._analysis_positions.get((kind, key), 0)
            self._analysis_positions[kind, key] = position + 1
            # A repeated preparation may ask for the last decision again.
            mode = modes[min(position, len(modes) - 1)]
            if mode == "full" and not (value is not None and current() and ready(value)):
                value = _produce_analysis(root, kind)
            return mode, value
        full = value is not None and current() and ready(value)
        mode = "full" if full else "scoped"
        tape.setdefault(kind, {}).setdefault(key, []).append(mode)
        return mode, value

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
                from repo_intel_core.call_graph import scoped_test_dependencies

                dependencies = scoped_test_dependencies(root, path)
                source = "scoped"
            row = self._rows[key] = {
                "path": path, "revision": revision, "dependencies": list(dependencies), "source": source,
            }
        return tuple(row["dependencies"]), row["source"] == "graph"


# New-process authority, intentionally outside runtime checkpoints.
active: DependencyView | None = None
