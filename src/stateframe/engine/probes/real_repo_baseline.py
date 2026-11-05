"""REAL-REPO BASELINE — the harness meets actual production code (AHA backend).

Purpose: first measurement of the gap between the 6-file gauntlet and reality.
- Corpus: every .py under backend/app/ (real MC-style production code, hundreds of
  files) — read-only; nothing is written to the repo.
- Task generation: sample real functions (5-30 lines), inject ONE deterministic
  single-line bug (operator flip, comparison flip, and/or swap, is/is-not None flip,
  dropped guard). Ground truth = the original function.
- Task text: template-generated symptom naming BEHAVIOR, never the file (realistic
  vagueness tier 1). Brace literals sanitized per the new standing rule.
- Ladder: C0 lexical narrowing over identifiers/paths (code-owned, scores every
  file) -> top-8 FILE MENU -> FUNCTION MENU -> line menu + single-line fix
  (the v5-proven shape).
- Oracle: AST-equality of the fixed function vs the original (deterministic,
  no imports/execution of app code needed, fail-closed).
Metrics: file-hit@menu8, file-pick, function-hit, fix-restores-original; per
bug-class breakdown. Checkpoint: results/real_repo_baseline.jsonl
"""
import ast, io, json, pathlib, random, re, sys, time, tokenize as _tokenize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from swarm.client2 import ask as _ask, degenerate

# Own server on 8098 — the stability battery owns 8099 and pkills it between boots
URL = "http://127.0.0.1:8098/v1/chat/completions"


POC = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path("/nonexistent/lab-repo/backend/app")
OUT = POC / "results" / "real_repo_baseline.jsonl"
N_TASKS = 20
MENU_K = 8

# ------------------------------------------------------------------ bug injectors
def flip_arith(line):
    for a, b in (("+", "-"), ("-", "+")):
        m = re.search(rf"\w \{a} \w", line)
        if m and "=" not in line.split(a)[0][-2:]:
            return line.replace(f" {a} ", f" {b} ", 1), "arith_flip"
    return None

def flip_cmp(line):
    for a, b in ((">=", "<"), ("<=", ">"), (" > ", " < "), (" < ", " > ")):
        if a in line:
            return line.replace(a, b if len(b) > 2 else b, 1), "cmp_flip"
    return None

def flip_bool(line):
    if " and " in line:
        return line.replace(" and ", " or ", 1), "bool_flip"
    if " or " in line and " or " != line.strip():
        return line.replace(" or ", " and ", 1), "bool_flip"
    return None

def flip_none(line):
    if " is not None" in line:
        return line.replace(" is not None", " is None", 1), "none_flip"
    if " is None" in line:
        return line.replace(" is None", " is not None", 1), "none_flip"
    return None

INJECTORS = [flip_cmp, flip_none, flip_bool, flip_arith]

SYMPTOMS = {
    "arith_flip": "produces numerically wrong results (values come out shifted the wrong way)",
    "cmp_flip": "applies its threshold/boundary check in the wrong direction",
    "bool_flip": "combines its conditions incorrectly (cases that should pass fail, or vice versa)",
    "none_flip": "handles the missing/None case backwards",
}

# ------------------------------------------------- prose vs executable classifier
# A planted bug whose mutated span lands inside a string/docstring or a comment is
# semantically FAKE: flipping "width or height" to "and" inside a docstring changes
# no behaviour, and the gold answer is restoring prose. Those tasks reward exactly
# the prose-mutation pathology the fixer is being filtered for, so we never plant
# them. Classification is syntactic (tokenize), never a substring heuristic.
_PROSE_TOKENS = {_tokenize.STRING, _tokenize.COMMENT}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(_tokenize, _name):
        _PROSE_TOKENS.add(getattr(_tokenize, _name))


# ------------------------------------------------------------------ shared prompt templates


FILE_MENU_Q = """TASK: {task}
Candidate files (pre-ranked by relevance):
{menu}
Which file contains the buggy function? Reply with only: {{"pick": "A"}}"""

FUNC_MENU_Q = """TASK: {task}
File {path} functions:
{menu}
Which function has the bug? Reply with only: {{"pick": "A"}}"""

LINE_MENU_Q = """TASK: {task}
Function `{fn}`:
{body}
Which line is wrong? Reply with only: {{"pick": "A"}}"""

FIX_Q = """TASK: {task}
The wrong line is:
{line}

Write the corrected version of this ONE line. Output only the line.{feedback}"""
