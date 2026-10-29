"""Checks for acceptance test files over non-code output (JSON, CSV-derived data, reports).

Use these inside the pytest file that accepts a task's output, for example::

    from stateframe import checks

    def test_summary_is_well_formed():
        data = json.loads(Path("out/summary.json").read_text())
        checks.require(checks.schema_validity(data, SCHEMA))

Each check returns a receipt ``{"pass": bool, "reason": str, "evidence": {...}}``; ``require`` turns a failing
receipt into an AssertionError that shows the reason and evidence. The checks are the engine's own mechanical
ones (exact schema, exact labels, exact source spans, exact recomputed values). They have not been measured as
task acceptance on the StateFrame path, so treat them as strict building blocks, not as a quality judge.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "engine" / "build_tasks" / "noncoding_oracles.py"


def _load():
    spec = importlib.util.spec_from_file_location("stateframe._noncoding_checks", _SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"stateframe.checks: cannot load {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_oracles = _load()

schema_validity = _oracles.schema_validity
closed_set_membership = _oracles.closed_set_membership
citation_span_grounding = _oracles.citation_span_grounding
deterministic_recomputation = _oracles.deterministic_recomputation

__all__ = ["schema_validity", "closed_set_membership", "citation_span_grounding",
           "deterministic_recomputation", "require"]


def require(result: dict[str, object]) -> dict[str, object]:
    """Assert that a check passed; on failure the message carries its reason and evidence."""
    if result.get("pass") is not True:
        raise AssertionError(f"check failed: {result.get('reason')}; evidence: {result.get('evidence')}")
    return result
