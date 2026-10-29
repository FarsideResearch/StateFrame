"""Optional parent supervisor for the existing worker, disabled by default.

K=1 unless KERNEL_MODE enables pilot. RetryPolicy owns temperatures and transport
retry counts. ParentMeter survives fresh sandboxes and fresh DB names. Only the
existing grader can resolve a task; budget exhaustion is not a capability claim.
Held-out assertion access requires PILOT_REPAIR_MODE=ASSISTED_REPAIR. Capsules
contain evidence references only and are never automatically replayed.
"""
from __future__ import annotations

import collections
import json
import pathlib
import os
import sys

POC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(POC))
sys.path.insert(0, str(POC / "probes"))

import converge                                                  # noqa: E402
import react_repair as rr                                        # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 21
K = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = POC / "pilot-measurements" / "pilot_runs.jsonl"
RESUME = POC / "pilot-measurements" / "resume_artifacts"


def classify_transcript(fn: str, actions: list[dict] | None = None) -> dict:
    """Compatibility name: typed completed actions replace transcript heuristics."""
    from kernel_pilot import ActionJournal
    journal = ActionJournal()
    for action in actions or ():
        journal.actions.append(action)
    return journal.diagnose()


def reframed_brief(d) -> str | None:
    """Held-out assertion access: permitted only in explicitly labeled assisted repair."""
    if os.environ.get("PILOT_REPAIR_MODE", "ORIGINAL_TASK") != "ASSISTED_REPAIR":
        return None
    ids = converge.fail_to_pass_ids(d)
    if not ids:
        return None
    rep = converge.verify_oracle(d)
    return ("A regression test for this bug fails with the following assertion. Make the code "
            "satisfy it:\n" + rep) if rep else None


def run_task(d, attempts: int = 1) -> dict:
    from kernel_effect_boundary import AXES, Mode
    from kernel_pilot import (ACTIVE_PARENT, POLICY, ParentBudgetExhausted,
                              ParentMeter, retry_capsule)
    mode = Mode.OFF
    for pair in os.environ.get("KERNEL_MODE", "").split(","):
        name, _, value = pair.partition(":")
        if name.strip() == "pilot":
            mode = Mode(value.strip().upper())
    repair_mode = os.environ.get("PILOT_REPAIR_MODE", "ORIGINAL_TASK")
    if repair_mode not in {"ORIGINAL_TASK", "ASSISTED_REPAIR"}:
        raise ValueError("invalid Pilot repair mode")
    if repair_mode == "ASSISTED_REPAIR" and not AXES[mode][2]:
        raise ValueError("assisted repair requires enabled Pilot")
    allowed = attempts if AXES[mode][2] else 1
    parent = ParentMeter(attempts=allowed,
                         steps=rr.parse_step_limit(os.environ.get("PILOT_PARENT_STEPS", rr.MAX_STEPS)),
                         enforce=AXES[mode][2])
    token = ACTIVE_PARENT.set(parent)
    history = []
    original_temp = rr.TEMPERATURE
    original_run_id = os.environ.get("REACT_RUN_ID")
    original_db = os.environ.get("REACT_DB_NAME")
    outcome = "ATTEMPT_BUDGET_EXHAUSTED"
    try:
        for index in range(allowed):
            try:
                parent.charge("attempts", reason="fresh worker sandbox")
                rr.TEMPERATURE = (POLICY.attempt_temperature(index) if AXES[mode][2]
                                  else original_temp)
                if AXES[mode][2]:
                    os.environ["REACT_RUN_ID"] = parent.parent_id[:12] + "_" + str(index + 1)
                row_input = dict(d)
                if index == 2 and repair_mode == "ASSISTED_REPAIR":
                    parent.charge("tool_work", reason="held-out oracle brief access")
                    brief = reframed_brief(d)
                    if brief:
                        row_input["_oracle_symptom"] = brief
                        row_input["_pilot_assisted"] = True
                row = rr.work(row_input)
            except ParentBudgetExhausted as exc:
                history.append({"attempt": index + 1, "result": "ATTEMPT_BUDGET_EXHAUSTED",
                                "category": str(exc)})
                break
            except Exception as exc:
                row = {"result": "HARNESS_ERROR", "error_type": type(exc).__name__}
            completed = parent.completed_attempts[-1] if parent.completed_attempts else {}
            history.append({"attempt": index + 1, "result": row.get("result"),
                            "behaviour": row.get("behaviour"),
                            "mode": "ASSISTED_REPAIR" if row_input.get("_oracle_symptom") else "ORIGINAL_TASK",
                            "run_id": completed.get("run_id"),
                            "diagnostics": classify_transcript(d["fn"], completed.get("actions"))})
            if row.get("result") == "PASS" or row.get("behaviour") == "PASS_behavioural":
                outcome = "RESOLVED"
                break
    finally:
        rr.TEMPERATURE = original_temp
        for key, value in (("REACT_RUN_ID", original_run_id), ("REACT_DB_NAME", original_db)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        ACTIVE_PARENT.reset(token)
    completed = parent.completed_attempts[-1] if parent.completed_attempts else {}
    capsule = retry_capsule(completed.get("actions", []), completed.get("dependencies", {}))
    record = {"fn": d["fn"], "exit": outcome, "attempts": len(history), "history": history,
              "mode": mode.value, "repair_mode": repair_mode, "parent_budget": parent.snapshot(),
              "budget_events": parent.events, "system_denominator_included": True,
              "retry_capsule": capsule}
    RESUME.mkdir(parents=True, exist_ok=True)
    with (RESUME / (parent.parent_id + ".json")).open("x") as handle:
        json.dump(record, handle, indent=1)
    return record


def main() -> int:
    if not rr.KEY:
        print("DEEPINFRA_API_KEY unset — refusing a vacuous zero.")
        return 2
    tasks = [json.loads(line) for line in rr.CORPUS.read_text().splitlines()
             if line.strip()][:N]

    print(f"PILOT — {rr.MODEL}, {len(tasks)} tasks, up to {K} attempts each\n")
    print("  preflighting oracles (fail-to-pass ids, two-sided)...")
    scoreable, oracle_missing = [], []
    for d in tasks:
        oracle_report = converge.verify_oracle(d)
        if oracle_report:
            oracle_ids = converge.fail_to_pass_ids(d)
            scoreable.append({
                **d,
                "_oracle_preflight": {
                    "parent_red": True,
                    "gold_green": True,
                    "oracle_ids": tuple(oracle_ids),
                },
            })
            oracle_status = "ORACLE OK"
        else:
            oracle_missing.append(d)
            oracle_status = "ORACLE-MISSING"
        print(f"    {d['fn'][:32]:34s} "
              f"{oracle_status}", flush=True)

    print(f"\n  scoreable {len(scoreable)} | oracle-missing {len(oracle_missing)} "
          f"(reported, never counted as failures)\n")

    results = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as sink:
        for d in scoreable:
            r = run_task(d, K)
            results.append(r)
            sink.write(json.dumps(r) + "\n")
            sink.flush()

    resolved = [r for r in results if r["exit"] == "RESOLVED"]
    print("\n" + "=" * 74)
    print(f"  RESOLVED            : {len(resolved)}/{len(scoreable)} scoreable")
    print(f"  ORACLE-MISSING      : {len(oracle_missing)} (excluded from denominator, named: "
          f"{[d['fn'] for d in oracle_missing]})")
    if resolved:
        dist = collections.Counter(r["attempts"] for r in resolved)
        print(f"  attempts-to-resolve : {dict(sorted(dist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
