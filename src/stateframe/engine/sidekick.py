"""Sidekick public receipt-review API.

Live predicates are implemented in kernel_sidekick and invoked only at lifecycle
boundaries by StateFrameRuntime. The legacy prose classifier below is retained
under an explicit diagnostic name; its findings have no execution authority.
"""
from __future__ import annotations

import json
import pathlib
import re

# Live authority uses only structured receipts. Legacy prose classifier is diagnostic-only.
from kernel_sidekick import review as review


HARNESS_SIGNATURES = (
    "Unrecognised command",
    "conftest or a pytest plugin fails to import",
    "CANNOT RUN IN THIS SANDBOX",
    "connection refused",
    "Connection refused",
    "OperationalError",
)


def legacy_prose_review(bundle: dict) -> dict:
    """Review one FROZEN attempt bundle. Returns {finding, evidence, detail}.

    bundle = {
        "transcript": [ {role, content}, ... ]     the worker's full exchange
        "final_diff": str                          unified diff of product-code changes
        "test_evidence": str                       the LAST test observation the worker saw
        "claimed_done": bool                       did the worker issue `done` itself
        "edited_test_files": [str]                 test paths the worker edited (from the log)
    }
    """
    diff = bundle.get("final_diff") or ""
    tests = bundle.get("test_evidence") or ""
    transcript = bundle.get("transcript") or []

    # 1. QUARANTINE FIRST. If the evidence stream itself is broken, nothing else in this
    #    bundle means anything, and the model must not be charged for it.
    harness_hits = [s for s in HARNESS_SIGNATURES if s in tests]
    obs_text = " ".join(m.get("content", "") for m in transcript
                        if m.get("role") == "user")
    unrec = obs_text.count("Unrecognised command")
    if harness_hits or unrec > 2:
        return {"finding": "QUARANTINE_HARNESS",
                "evidence": harness_hits or [f"{unrec}x Unrecognised command"],
                "detail": "evidence stream invalid; do not charge the model; fix the harness"}

    # 2. INVALIDATE on objective contradictions.
    if bundle.get("edited_test_files"):
        return {"finding": "INVALIDATE_ATTEMPT",
                "evidence": bundle["edited_test_files"],
                "detail": "forbidden test-file edits (grading overlays pristine f2p tests "
                          "anyway, but the ATTEMPT gamed its own feedback)"}
    if bundle.get("claimed_done") and "pytest exit 0" not in tests and diff.strip():
        m = re.search(r"pytest exit (\d+)", tests)
        if m and m.group(1) in ("1", "2", "3"):
            return {"finding": "INVALIDATE_ATTEMPT",
                    "evidence": [f"claimed done with pytest exit {m.group(1)}"],
                    "detail": "success asserted against red evidence"}

    # 3. CHALLENGE_ONCE for a named missing obligation.
    if diff.strip() and "pytest exit" not in tests:
        return {"finding": "CHALLENGE_ONCE",
                "evidence": ["edits exist, no test run observed after the last edit"],
                "detail": "rerun tests against the final tree before finishing"}

    # 4. Internally consistent -> the grader's move now.
    if diff.strip() and "pytest exit 0" in tests:
        return {"finding": "READY_FOR_GRADING",
                "evidence": ["diff present", "last observed run green"],
                "detail": "hidden fail-to-pass proof still owed by the external grader"}

    return {"finding": "UNVERIFIED",
            "evidence": ["no diff" if not diff.strip() else "no green evidence"],
            "detail": "evidence cannot establish correctness"}


def bundle_from_transcript(path: pathlib.Path) -> dict:
    """Freeze an attempt bundle from a react_transcripts/*.json file."""
    d = json.loads(path.read_text())
    msgs = d.get("messages", [])
    edited_tests = []
    last_test_obs = ""
    claimed_done = False
    for m in msgs:
        c = m.get("content", "")
        if m["role"] == "assistant":
            for ln in c.strip().splitlines():
                s = ln.strip().lstrip("`").strip()
                if s.startswith("edit ") and ("tests/" in s or "test_" in s):
                    edited_tests.append(s[5:].strip())
                if s.startswith("done"):
                    claimed_done = True
                break
        elif c.startswith("pytest exit"):
            last_test_obs = c
    return {"transcript": msgs,
            "final_diff": "x" if d.get("edits") else "",     # presence signal; diff lives in sandbox
            "test_evidence": last_test_obs,
            "claimed_done": claimed_done,
            "edited_test_files": edited_tests}


def legacy_diagnostic_selftest() -> int:
    """Planted controls — positive AND negative, per the standing rule."""
    ok = True

    r = legacy_prose_review({"final_diff": "d", "test_evidence": "pytest exit 4 — THIS REPOSITORY'S TEST "
                "SUITE CANNOT RUN IN THIS SANDBOX", "transcript": [], "claimed_done": True})
    print("harness failure -> QUARANTINE:",
          "PASS" if r["finding"] == "QUARANTINE_HARNESS" else f"FAIL {r}")
    ok &= r["finding"] == "QUARANTINE_HARNESS"

    r = legacy_prose_review({"final_diff": "d", "test_evidence": "pytest exit 1 — TESTS FAILED",
                "transcript": [], "claimed_done": True})
    print("claimed green on red  -> INVALIDATE:",
          "PASS" if r["finding"] == "INVALIDATE_ATTEMPT" else f"FAIL {r}")
    ok &= r["finding"] == "INVALIDATE_ATTEMPT"

    r = legacy_prose_review({"final_diff": "d", "test_evidence": "pytest exit 0 — ALL TESTS PASSED.",
                "transcript": [], "claimed_done": True, "edited_test_files": []})
    print("green + diff          -> READY:     ",
          "PASS" if r["finding"] == "READY_FOR_GRADING" else f"FAIL {r}")
    ok &= r["finding"] == "READY_FOR_GRADING"

    # NEGATIVE control: a clean, honest attempt must never be quarantined or invalidated.
    r = legacy_prose_review({"final_diff": "d", "test_evidence": "pytest exit 0 — ALL TESTS PASSED.",
                "transcript": [{"role": "user", "content": "ok"}], "claimed_done": True,
                "edited_test_files": []})
    print("NEGATIVE (clean run)  -> not punished:",
          "PASS" if r["finding"] == "READY_FOR_GRADING" else f"FAIL {r}")
    ok &= r["finding"] == "READY_FOR_GRADING"

    r = legacy_prose_review({"final_diff": "d", "test_evidence": "pytest exit 0",
                "edited_test_files": ["tests/test_x.py"], "transcript": []})
    print("test-file edit        -> INVALIDATE:",
          "PASS" if r["finding"] == "INVALIDATE_ATTEMPT" else f"FAIL {r}")
    ok &= r["finding"] == "INVALIDATE_ATTEMPT"

    return 0 if ok else 1


def selftest() -> int:
    for text in ("pytest exit 0", "Connection refused", "Unrecognised command " * 4):
        assert review({"test_evidence": text, "final_diff": "x"})["finding"] == "UNVERIFIED"
    print("Receipt-only authority: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
