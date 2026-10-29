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

def ask(prompt, **kw):
    kw.setdefault("url", URL)
    return _ask(prompt, **kw)

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


def prose_spans(src):
    """Map 1-based line number -> list of (start_col, end_col) prose column spans.

    end_col is exclusive; a span covering a whole continuation line of a triple
    quoted string is recorded as (0, inf). Unparseable sources fail CLOSED (every
    line treated as prose) so a tokenizer failure can never smuggle a fake task in.
    """
    spans = {}
    try:
        toks = list(_tokenize.generate_tokens(io.StringIO(src).readline))
    except (_tokenize.TokenError, SyntaxError, IndentationError):
        return None  # fail closed — caller skips the file
    for tok in toks:
        if tok.type not in _PROSE_TOKENS:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            lo = scol if row == srow else 0
            hi = ecol if row == erow else float("inf")
            spans.setdefault(row, []).append((lo, hi))
    return spans


def mutation_col(orig, mutated):
    """First column at which the injector changed the line (-1 if identical)."""
    n = min(len(orig), len(mutated))
    for i in range(n):
        if orig[i] != mutated[i]:
            return i
    return n if len(orig) != len(mutated) else -1


def is_prose_mutation(spans, lineno, orig, mutated):
    """True when the injector's edit lands inside a string/docstring or comment."""
    col = mutation_col(orig, mutated)
    if col < 0:
        return True  # a no-op "bug" is never a real task
    return any(lo <= col < hi for lo, hi in spans.get(lineno, ()))


def harvest_tasks(seed=7):
    rng = random.Random(seed)
    files = sorted(p for p in REPO.rglob("*.py")
                   if "test" not in p.name and "migration" not in str(p))
    rng.shuffle(files)
    tasks = []
    for p in files:
        if len(tasks) >= N_TASKS:
            break
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        pspans = prose_spans(src)
        if pspans is None:
            continue  # fail closed: cannot certify code-vs-prose for this file
        fns = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and 5 <= (getattr(n, "end_lineno", 0) - n.lineno) <= 30]
        rng.shuffle(fns)
        for fn in fns:
            lines = src.splitlines()
            span = list(range(fn.body[0].lineno - 1, getattr(fn, "end_lineno", fn.lineno)))
            rng.shuffle(span)
            done = False
            for li in span:
                for inj in INJECTORS:
                    r = inj(lines[li])
                    if r:
                        bugged, cls = r
                        # Executable-code gate: skip docstring/comment/string edits
                        # and move on to the next eligible line in this function
                        # (then the next function, then the next file) so the task
                        # COUNT stays stable and the harvest stays deterministic.
                        if is_prose_mutation(pspans, li + 1, lines[li], bugged):
                            continue
                        blines = list(lines)
                        blines[li] = bugged
                        try:
                            ast.parse("\n".join(blines))
                        except SyntaxError:
                            continue
                        purpose = (ast.get_docstring(fn) or fn.name.replace("_", " "))
                        purpose = re.sub(r"\{[^}]*\}", "", purpose).strip().split("\n")[0][:120]
                        task = (f"A function responsible for '{purpose}' {SYMPTOMS[cls]}. "
                                f"Find it and fix the bug.")
                        tasks.append(dict(
                            path=str(p.relative_to(REPO)), fn=fn.name, cls=cls,
                            line_idx=li, task=task,
                            orig_line=lines[li], bug_line=bugged,
                            bugged_src="\n".join(blines) + "\n",
                            orig_fn_dump=ast.dump(fn)))
                        done = True
                        break
                if done:
                    break
            if done:
                break
    return tasks

# ------------------------------------------------------------------ C0 narrowing
def tokenize(s):
    return set(w.lower() for w in re.findall(r"[a-zA-Z_]{3,}", s))

def narrow(task_text, all_files):
    q = tokenize(task_text)
    scored = []
    for path, src in all_files.items():
        toks = tokenize(path) | tokenize(src[:4000])
        scored.append((len(q & toks), path))
    scored.sort(reverse=True)
    return [p for _, p in scored[:MENU_K]]

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


def menu_pick(prompt, options, seed):
    letters = [chr(65 + i) for i in range(len(options))]
    out = ask(prompt, seed=seed).text or ""
    m = re.search(r'"pick"\s*:\s*"([A-Z])"', out)
    if m and m.group(1) in letters:
        return options[letters.index(m.group(1))]
    hits = [o for o in options if str(o) in out]
    return hits[0] if len(set(map(str, hits))) == 1 else None


def run_task(t, all_files, seed=31):
    log = dict(path=t["path"], fn=t["fn"], cls=t["cls"], score={})
    cand = narrow(t["task"], all_files)
    log["score"]["file_in_menu"] = int(t["path"] in cand)
    menu = "\n".join(f"  {chr(65+i)}) {p}" for i, p in enumerate(cand))
    pick = menu_pick(FILE_MENU_Q.format(task=t["task"], menu=menu), cand, seed)
    log["file_pick"] = pick
    log["score"]["file_hit"] = int(pick == t["path"])
    if pick != t["path"]:
        pick = t["path"] if t["path"] in cand else None  # partial credit continues
    if pick is None:
        with open(OUT, "a") as f:
            f.write(json.dumps(log) + "\n")
        return log
    src = t["bugged_src"] if pick == t["path"] else all_files[pick]
    tree = ast.parse(src)
    fns = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    fmenu = "\n".join(f"  {chr(65+i)}) {n}" for i, n in enumerate(fns[:26]))
    fpick = menu_pick(FUNC_MENU_Q.format(task=t["task"], path=pick, menu=fmenu),
                      fns[:26], seed)
    log["func_pick"] = fpick
    log["score"]["func_hit"] = int(fpick == t["fn"])
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == t["fn"]), None)
    if node is None:
        with open(OUT, "a") as f:
            f.write(json.dumps(log) + "\n")
        return log
    lines = src.splitlines()
    span = lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]
    opts = [l for l in span if l.strip()][:26]
    lmenu = "\n".join(f"  {chr(65+i)}) {l.strip()[:90]}" for i, l in enumerate(opts))
    body = "\n".join(span)
    lpick = menu_pick(LINE_MENU_Q.format(task=t["task"], fn=t["fn"], body=lmenu),
                      opts, seed)
    log["score"]["line_hit"] = int(lpick is not None
                                   and lpick.strip() == t["bug_line"].strip())
    target = lpick if lpick is not None else t["bug_line"]
    feedback, fixed = "", 0
    for attempt in range(4):
        out = ask(FIX_Q.format(task=t["task"], line=target.strip(), feedback=feedback),
                  seed=seed + attempt, max_tokens=400).text or ""
        nl = [l for l in out.splitlines() if l.strip()]
        cand_line = nl[0].strip().strip("`") if nl else ""
        if not cand_line or degenerate(cand_line):
            feedback = "\n\nMEASURED: output exactly ONE corrected code line."
            continue
        blines = t["bugged_src"].splitlines()
        indent = len(t["bug_line"]) - len(t["bug_line"].lstrip())
        blines[t["line_idx"]] = " " * indent + cand_line
        try:
            ftree = ast.parse("\n".join(blines))
        except SyntaxError as se:
            feedback = f"\n\nMEASURED: SyntaxError: {se.msg}. Output one corrected line."
            continue
        fnode = next((n for n in ast.walk(ftree)
                      if isinstance(n, ast.FunctionDef) and n.name == t["fn"]), None)
        if fnode is not None and ast.dump(fnode) == t["orig_fn_dump"]:
            fixed = 1
            break
        feedback = ("\n\nMEASURED: that line still leaves the function behaving "
                    "incorrectly for the described symptom. Write a different fix.")
    log["score"]["fixed"] = fixed
    with open(OUT, "a") as f:
        f.write(json.dumps(log) + "\n")
    return log


if __name__ == "__main__":
    t0 = time.time()
    tasks = harvest_tasks()
    print(f"harvested {len(tasks)} tasks from {REPO}")
    all_files = {}
    for p in sorted(REPO.rglob("*.py")):
        if "test" in p.name or "migration" in str(p):
            continue
        try:
            all_files[str(p.relative_to(REPO))] = p.read_text()
        except Exception:
            pass
    print(f"corpus: {len(all_files)} files")
    agg = {}
    for t in tasks:
        r = run_task(t, all_files)
        s = r["score"]
        print(f"{t['cls']:<10} {t['path'][:44]:<46} "
              f"menu={s.get('file_in_menu')} file={s.get('file_hit')} "
              f"fn={s.get('func_hit')} line={s.get('line_hit')} fix={s.get('fixed')}",
              flush=True)
        for k, v in s.items():
            agg[k] = agg.get(k, 0) + (v or 0)
    n = len(tasks)
    print(f"\nREAL-REPO BASELINE n={n} wall={(time.time()-t0)/60:.1f}m")
    for k, v in agg.items():
        print(f"  {k}: {v}/{n}")
