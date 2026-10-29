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
from probes.real_repo_baseline import harvest_tasks, REPO, MENU_K

URL = "http://127.0.0.1:8098/v1/chat/completions"

def ask(prompt, **kw):
    kw.setdefault("url", URL)
    return _ask(prompt, **kw)

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


def cascade(task_text, idx, df, n, adj, log):
    """g1 core: BM25 top-24 -> import-proximity union -> 1B relevance filter -> menu 8."""
    ranked = bm25_rank(task_text, idx, df, n)
    lex = [p for _, p in ranked[:24]]
    prox = set(lex[:8])
    for p in lex[:8]:
        prox |= adj.get(p, set())
    pool = list(dict.fromkeys(lex + sorted(prox)))[:16]
    kept = []
    for p in pool:
        idents = ", ".join(sorted(idx[p], key=lambda w: -idx[p][w])[:12])
        out = ask(REL_Q.format(task=task_text, path=p, idents=idents), seed=31).text or ""
        ans = "no" if '"no"' in out or "'no'" in out else ("yes" if "yes" in out.lower() else "unsure")
        if ans != "no":
            kept.append(p)
        log.setdefault("rel_atoms", []).append({p: ans})
    if not kept:
        kept = pool
    # preserve BM25 order among survivors
    order = {p: i for i, p in enumerate(pool)}
    kept.sort(key=lambda p: order[p])
    return kept[:MENU_K]

# ------------------------------------------------------------------ fair oracle
class _Norm(ast.NodeTransformer):
    MIRROR = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in self.MIRROR:
            l, r = ast.dump(node.left), ast.dump(node.comparators[0])
            if r < l:  # canonical operand order
                node = ast.Compare(left=node.comparators[0],
                                   ops=[self.MIRROR[type(node.ops[0])]()],
                                   comparators=[node.left])
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        node.values.sort(key=ast.dump)
        return node


def norm_dump(fn_node):
    return ast.dump(ast.fix_missing_locations(_Norm().visit(fn_node)))


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


def menu_pick(prompt, options, seed):
    letters = [chr(65 + i) for i in range(len(options))]
    out = ask(prompt, seed=seed).text or ""
    m = re.search(r'"pick"\s*:\s*"([A-Z])"', out)
    if m and m.group(1) in letters:
        return options[letters.index(m.group(1))]
    hits = [o for o in options if str(o) in out]
    return hits[0] if len(set(map(str, hits))) == 1 else None


def run_task(t, all_files, idx, df, n, adj, seed=31):
    log = dict(path=t["path"], fn=t["fn"], cls=t["cls"], score={})
    cand = cascade(t["task"], idx, df, n, adj, log)
    log["score"]["file_in_menu"] = int(t["path"] in cand)
    menu = "\n".join(f"  {chr(65+i)}) {p}" for i, p in enumerate(cand))
    pick = menu_pick(FILE_MENU_Q.format(task=t["task"], menu=menu), cand, seed)
    log["file_pick"] = pick
    log["score"]["file_hit"] = int(pick == t["path"])
    if pick != t["path"]:
        pick = t["path"] if t["path"] in cand else None
    if pick is None:
        with open(OUT, "a") as f:
            f.write(json.dumps(log) + "\n")
        return log
    src = t["bugged_src"] if pick == t["path"] else all_files[pick]
    tree = ast.parse(src)
    fns = [x.name for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)]
    fmenu = "\n".join(f"  {chr(65+i)}) {x}" for i, x in enumerate(fns[:26]))
    fpick = menu_pick(FUNC_MENU_Q.format(task=t["task"], path=pick, menu=fmenu),
                      fns[:26], seed)
    log["func_pick"] = fpick
    log["score"]["func_hit"] = int(fpick == t["fn"])
    node = next((x for x in ast.walk(tree)
                 if isinstance(x, ast.FunctionDef) and x.name == t["fn"]), None)
    if node is None:
        with open(OUT, "a") as f:
            f.write(json.dumps(log) + "\n")
        return log
    lines = src.splitlines()
    span = lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]
    opts = [l for l in span if l.strip()][:26]
    lmenu = "\n".join(f"  {chr(65+i)}) {l.strip()[:90]}" for i, l in enumerate(opts))
    lpick = menu_pick(LINE_MENU_Q.format(task=t["task"], fn=t["fn"], menu=lmenu),
                      opts, seed)
    log["score"]["line_hit"] = int(lpick is not None
                                   and lpick.strip() == t["bug_line"].strip())
    # region context around the true bug line (v5 lesson: show the neighborhood)
    li = t["line_idx"]
    blines = t["bugged_src"].splitlines()
    lo, hi = max(0, li - 3), min(len(blines), li + 4)
    region = "\n".join((">>> " if i == li else "    ") + blines[i] for i in range(lo, hi))
    region = re.sub(r"\{[^}]*\}", "(...)", region) if False else region
    # gold normalized dump for the fair oracle
    orig_tree = ast.parse(t["bugged_src"].replace(t["bug_line"], t["orig_line"], 1))
    gold = next(x for x in ast.walk(orig_tree)
                if isinstance(x, ast.FunctionDef) and x.name == t["fn"])
    gold_dump = norm_dump(gold)
    feedback, fixed = "", 0
    for attempt in range(4):
        out = ask(FIX_Q.format(task=t["task"], region=region, feedback=feedback),
                  seed=seed + attempt, max_tokens=400).text or ""
        nl = [l for l in out.splitlines() if l.strip()]
        cand_line = nl[0].strip().strip("`") if nl else ""
        if not cand_line or degenerate(cand_line):
            feedback = "\n\nMEASURED: output exactly ONE corrected code line."
            continue
        b2 = list(blines)
        indent = len(t["bug_line"]) - len(t["bug_line"].lstrip())
        b2[li] = " " * indent + cand_line
        try:
            ftree = ast.parse("\n".join(b2))
        except SyntaxError as se:
            feedback = f"\n\nMEASURED: SyntaxError: {se.msg}. Output one corrected line."
            continue
        fnode = next((x for x in ast.walk(ftree)
                      if isinstance(x, ast.FunctionDef) and x.name == t["fn"]), None)
        if fnode is not None and norm_dump(fnode) == gold_dump:
            fixed = 1
            break
        feedback = ("\n\nMEASURED: still behaves incorrectly for the described "
                    "symptom. Write a different fix of the marked line.")
    log["score"]["fixed"] = fixed
    with open(OUT, "a") as f:
        f.write(json.dumps(log) + "\n")
    return log


if __name__ == "__main__":
    t0 = time.time()
    tasks = harvest_tasks()
    # Targeted mode: `python3 ..._v2.py 3 7` runs only
    # task indices 3 and 7; `python3 ..._v2.py failed` reruns every task whose
    # LAST record in the checkpoint jsonl wasn't fully fixed. Full run (no args)
    # remains the regression pass after targeted iteration converges.
    if len(sys.argv) > 1:
        if sys.argv[1] == "failed":
            last = {}
            try:
                for l in open(OUT):
                    d = json.loads(l)
                    last[d["path"]] = d["score"].get("fixed", 0)
            except FileNotFoundError:
                pass
            tasks = [t for t in tasks if not last.get(t["path"], 0)]
        else:
            want = {int(a) for a in sys.argv[1:]}
            tasks = [t for i, t in enumerate(tasks) if i in want]
    print(f"harvested {len(tasks)} tasks (same seed as v1)")
    all_files = {}
    for p in sorted(REPO.rglob("*.py")):
        if "test" in p.name or "migration" in str(p):
            continue
        try:
            all_files[str(p.relative_to(REPO))] = p.read_text()
        except Exception:
            pass
    idx, df, n, adj = build_index(all_files)
    print(f"corpus: {len(all_files)} files indexed")
    agg = {}
    for t in tasks:
        r = run_task(t, all_files, idx, df, n, adj)
        s = r["score"]
        print(f"{t['cls']:<10} {t['path'][:44]:<46} "
              f"menu={s.get('file_in_menu')} file={s.get('file_hit')} "
              f"fn={s.get('func_hit')} line={s.get('line_hit')} fix={s.get('fixed')}",
              flush=True)
        for k, v in s.items():
            agg[k] = agg.get(k, 0) + (v or 0)
    m = len(tasks)
    print(f"\nREAL-REPO v2 n={m} wall={(time.time()-t0)/60:.1f}m")
    for k, v in agg.items():
        print(f"  {k}: {v}/{m}")
