"""Classify a pytest gate run by what it actually ran: PASS / PRODUCT_FAIL / INVALID_ENV / HARNESS_FAIL / INCOMPLETE.

A matching collected count is not a pass. Every mandatory case must end PASSED: a skip, an xfail, an xpass or
a missing case makes the run INCOMPLETE. Zero mandatory cases, or zero tests run, is INCOMPLETE too.

Order of precedence (first match wins):
    HARNESS_FAIL  the run left no finished census, or pytest exited with an internal or usage error (3, 4)
    INVALID_ENV   a collection error is an import failure (the tree did not install or import here)
    PRODUCT_FAIL  any test or any other collection error failed
    INCOMPLETE    a mandatory case did not pass, or nothing mandatory / nothing at all ran
    PASS          otherwise

CLI for gate scripts (exit 0 only on PASS; the verdict is the last stdout line):
    python3 tests/_gate_verdict.py [--platform linux|darwin] [--mandatory <nodeid>]... -- <pytest args>
Without --mandatory the mandatory set is every tests/must_run.json cell whose job for the platform is RUN.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PASS = "PASS"
PRODUCT_FAIL = "PRODUCT_FAIL"
INVALID_ENV = "INVALID_ENV"
HARNESS_FAIL = "HARNESS_FAIL"
INCOMPLETE = "INCOMPLETE"

_IMPORT_FAILURES = ("ModuleNotFoundError", "ImportError")


@dataclass(frozen=True)
class Verdict:
    outcome: str
    reasons: tuple[str, ...]

    def render(self) -> str:
        return self.outcome + ("" if not self.reasons else ": " + "; ".join(self.reasons))


def mandatory_nodeids(root: Path, platform: str) -> list[str]:
    """Node ids of the release matrix cells that must RUN on ``platform``."""
    manifest = json.loads((root / "tests" / "must_run.json").read_text(encoding="utf-8"))
    selected: list[str] = []
    for cell in manifest["cells"]:
        if cell.get("jobs", {}).get(platform) == "RUN":
            selected.extend(cell.get("item_nodeids") or [cell["nodeid"]])
    return sorted(set(selected))


def _matching(nodeid: str, outcomes: dict[str, str]) -> list[str]:
    """Outcomes for a mandatory id: exact, or every parametrized case / test under a base id or file."""
    if nodeid in outcomes:
        return [outcomes[nodeid]]
    return [outcome for ran, outcome in outcomes.items()
            if ran.startswith(nodeid + "[") or ran.startswith(nodeid + "::")]


def classify(document: dict | None, mandatory: list[str], output: str = "") -> Verdict:
    if document is None or document.get("phase") != "finished":
        # A conftest that cannot import stops pytest before any hook writes the census.
        if "ImportError while loading conftest" in output or any(name in output for name in _IMPORT_FAILURES):
            return Verdict(INVALID_ENV, ("the tree did not import before collection",))
        return Verdict(HARNESS_FAIL, ("the run left no finished census",))
    exitstatus = document.get("exitstatus")
    if exitstatus in (3, 4):
        return Verdict(HARNESS_FAIL, (f"pytest exited {exitstatus} (internal or usage error)",))
    errors = [report for report in document.get("reports", ()) if report.get("outcome") == "failed"]
    import_errors = [report["nodeid"] for report in errors
                     if any(name in report.get("detail", "") for name in _IMPORT_FAILURES)]
    if import_errors:
        return Verdict(INVALID_ENV, (f"import failed during collection: {import_errors}",))
    outcomes: dict[str, str] = document.get("outcomes", {})
    failed = sorted(nodeid for nodeid, outcome in outcomes.items() if outcome == "failed")
    if errors or failed:
        return Verdict(PRODUCT_FAIL, tuple(
            [f"collection error: {report['nodeid']}" for report in errors] + [f"failed: {node}" for node in failed]))
    reasons: list[str] = []
    if not outcomes:
        reasons.append("0 tests ran")
    if not mandatory:
        reasons.append("0 mandatory cases")
    for nodeid in mandatory:
        seen = _matching(nodeid, outcomes)
        if not seen:
            reasons.append(f"mandatory case not run: {nodeid}")
        elif any(outcome != "passed" for outcome in seen):
            reasons.append(f"mandatory case {'/'.join(sorted(set(seen)))}: {nodeid}")
    if reasons:
        return Verdict(INCOMPLETE, tuple(reasons))
    return Verdict(PASS, ())


def run_gate(root: Path, pytest_args: list[str], mandatory: list[str], timeout: float) -> Verdict:
    """Run pytest in a child with the census plugin and classify what it ran."""
    with tempfile.TemporaryDirectory(prefix="sf-gate-") as scratch:
        target = Path(scratch) / "census.json"
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(Path(__file__).resolve().parent)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SF_COLLECTION_CENSUS": str(target),
        }
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "_collection_census_plugin",
                 "--rootdir", str(root), *pytest_args],
                cwd=root, env=environment, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return Verdict(HARNESS_FAIL, (f"pytest did not finish within {timeout:.0f} s",))
        document = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
    return classify(document, mandatory, result.stdout + result.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="darwin" if sys.platform == "darwin" else "linux")
    parser.add_argument("--mandatory", action="append", default=None)
    parser.add_argument("--timeout", type=float, default=43200.0)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    pytest_args = options.pytest_args[1:] if options.pytest_args[:1] == ["--"] else options.pytest_args
    mandatory = options.mandatory if options.mandatory is not None else mandatory_nodeids(ROOT, options.platform)
    verdict = run_gate(ROOT, pytest_args, mandatory, options.timeout)
    print(verdict.render())
    return 0 if verdict.outcome == PASS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
