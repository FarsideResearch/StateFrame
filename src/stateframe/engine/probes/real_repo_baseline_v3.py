"""REAL-REPO v3 — kernel-style: evidence cards + mutation-menu fix stage.

Changes vs v2 (same seeded 20 tasks):
  1. FILE MENU WITH EVIDENCE CARDS — each candidate shows WHY it ranked (its top
     matched query identifiers), attacking v2's new failure (rich menus made bare
     path-picking harder).
  2. MUTATION-MENU FIX STAGE (the v5 atom-assembly move): C0 infers the bug CLASS
     from the symptom wording (deterministic 1:1 with our symptom templates, and in
     production a symptom taxonomy); code enumerates every class-matched single-line
     mutation across the picked function (the closed candidate set); the model
     answers ONE menu atom: which mutation matches the symptom. Line-pick and fix
     collapse into a single decision over code-enumerated candidates. Emission: gone.
Oracle: v2's normalized-AST behavioral equivalence, unchanged.
Checkpoint: results/real_repo_baseline_v3.jsonl
"""
import ast, json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from probes.real_repo_baseline import REPO, MENU_K, INJECTORS
from probes.real_repo_baseline_v2 import (
    build_index, bm25_rank, words, FUNC_MENU_Q)
# Single authority for the mutation candidate set (was a local copy here).
# Re-exported so `from probes.real_repo_baseline_v3 import enumerate_mutations`
# (fixer.py) keeps working.
from mutations import enumerate_mutations  # noqa: F401

POC = pathlib.Path(__file__).resolve().parents[1]
OUT = POC / "results" / "real_repo_baseline_v3.jsonl"

CLASS_HINTS = {
    "arith_flip": ["numerically wrong", "shifted the wrong way"],
    "cmp_flip": ["threshold", "boundary", "wrong direction"],
    "bool_flip": ["combines its conditions", "conditions incorrectly"],
    "none_flip": ["missing/none", "none case", "backwards"],
}

FILE_MENU_EV_Q = """TASK: {task}
Candidate files (with the identifiers that matched the task):
{menu}
Which file contains the buggy function? Reply with only: {{"pick": "A"}}"""

MUT_MENU_Q = """TASK: {task}
The bug is a single wrong line in `{fn}`. Candidate corrections (current -> corrected):
{menu}
Which correction fixes the described symptom? Reply with only: {{"pick": "A"}}"""


def infer_class(task_text):
    tl = task_text.lower()
    for cls, hints in CLASS_HINTS.items():
        if any(h in tl for h in hints):
            return cls
    return None
