"""REAL-REPO BASELINE v2 — g1 cascade core + fairer oracle + region-context fix.

Changes vs v1 (same seeded 20 tasks for clean before/after):
  LOCATE (the 55%-of-loss stage, g1 design):
  - snake_case/camelCase SPLITTING in tokenization (v1's fatal flaw: the token
    'learner_concept_gap' never matched task words 'learner concept gap')
  - BM25-weighted lexical scoring over identifiers + docstrings + path words
  - import-graph 1-hop proximity expansion (union, recall-first per g1)
  - 1B binary relevance atoms filter 16 -> menu 8 (reject only confident NOs)
  ORACLE:
  - behavioral equivalence, not textual identity: mirrored comparisons
    (a<b == b>a) and commutative and/or operand order normalize equal
  FIX:
  - the fix turn shows a +/-3 line region for context (v5 lesson), still emits
    ONE corrected line
Checkpoint: results/real_repo_baseline_v2.jsonl
"""
import ast, json, math, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from swarm.client2 import ask as _ask, degenerate
from probes.real_repo_baseline import REPO, MENU_K

URL = "http://127.0.0.1:8098/v1/chat/completions"


POC = pathlib.Path(__file__).resolve().parents[1]
OUT = POC / "results" / "real_repo_baseline_v2.jsonl"

# ------------------------------------------------------------------ tokenization
def words(s):
    """Split identifiers into real words: snake_case, camelCase, path segments."""
    out = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        parts = re.split(r"_+", tok)
        for p in parts:
            for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]{2,}", p):
                if len(w) >= 3:
                    out.add(w.lower())
    return out


def build_index(all_files):
    """Per-file word bags (identifiers weighted by kind) + df counts + import graph."""
    idx, df, imports = {}, {}, {}
    for path, src in all_files.items():
        bag = {}
        def add(ws, w8):
            for w in ws:
                bag[w] = max(bag.get(w, 0), w8)
        add(words(path), 3.0)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            idx[path] = bag
            continue
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                add(words(node.name), 2.5)
                doc = ast.get_docstring(node) or ""
                add(words(doc[:300]), 1.5)
            elif isinstance(node, ast.Name):
                add(words(node.id), 1.0)
            elif isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        doc = ast.get_docstring(tree) or ""
        add(words(doc[:300]), 2.0)
        idx[path] = bag
        imports[path] = mods
        for w in bag:
            df[w] = df.get(w, 0) + 1
    n = len(all_files)
    # import adjacency by local module name
    mod_of = {p: pathlib.Path(p).stem for p in all_files}
    adj = {p: set() for p in all_files}
    for p, mods in imports.items():
        for q, m in mod_of.items():
            if m in mods and q != p:
                adj[p].add(q)
                adj[q].add(p)
    return idx, df, n, adj


def bm25_rank(task_text, idx, df, n, k1=1.5):
    q = words(task_text)
    scored = []
    for path, bag in idx.items():
        s = 0.0
        for w in q:
            if w in bag:
                s += math.log(1 + (n - df.get(w, 1) + 0.5) / (df.get(w, 1) + 0.5)) * \
                     (bag[w] * (k1 + 1)) / (bag[w] + k1)
        scored.append((s, path))
    scored.sort(reverse=True)
    return scored


REL_Q = """TASK: {task}
Candidate file: {path}
Its top identifiers: {idents}
Could this file plausibly contain the buggy function? Reply with only:
{{"answer": "yes"}} or {{"answer": "no"}} or {{"answer": "unsure"}}"""


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
{menu}
Which line is wrong? Reply with only: {{"pick": "A"}}"""

FIX_Q = """TASK: {task}
Region of the function (the wrong line is marked >>>):
{region}

Write the corrected version of the marked line ONLY. Output only that one line.{feedback}"""
