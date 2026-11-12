"""Maintainer-only v3 baseline corpus driver."""
import ast, json, pathlib, re, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src/stateframe/engine"))
from probes.real_repo_baseline import REPO, MENU_K
from probes.real_repo_baseline_v2 import build_index, bm25_rank, words, FUNC_MENU_Q
from probes.real_repo_baseline_v3 import (OUT, FILE_MENU_EV_Q, MUT_MENU_Q,
    infer_class, enumerate_mutations)
from real_repo_baseline import harvest_tasks
from real_repo_baseline_v2 import cascade, menu_pick, norm_dump
def run_task(t, all_files, idx, df, n, adj, seed=31):
    log = dict(path=t["path"], fn=t["fn"], cls=t["cls"], score={})
    cand = cascade(t["task"], idx, df, n, adj, log)
    log["score"]["file_in_menu"] = int(t["path"] in cand)
    q = words(t["task"])
    cards = []
    for p in cand:
        matched = sorted(q & set(idx[p]), key=lambda w: -idx[p][w])[:4]
        cards.append(f"(matched: {', '.join(matched) if matched else '-'})")
    menu = "\n".join(f"  {chr(65+i)}) {p}  {c}" for i, (p, c) in enumerate(zip(cand, cards)))
    pick = menu_pick(FILE_MENU_EV_Q.format(task=t["task"], menu=menu), cand, seed)
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
    # ---- MUTATION-MENU FIX (C0 class -> enumerate -> one menu atom) ----
    cls = infer_class(t["task"])
    log["inferred_cls"] = cls
    blines = t["bugged_src"].splitlines()
    span = blines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]
    muts = enumerate_mutations(span, node.lineno - 1, cls)[:12]
    log["n_mutations"] = len(muts)
    gold_src = t["bugged_src"].replace(t["bug_line"], t["orig_line"], 1)
    gold = next(x for x in ast.walk(ast.parse(gold_src))
                if isinstance(x, ast.FunctionDef) and x.name == t["fn"])
    gold_dump = norm_dump(gold)
    fixed = 0
    if muts:
        mmenu = "\n".join(f"  {chr(65+i)}) line {li+1}: {old.strip()[:60]}  ->  {new.strip()[:60]}"
                          for i, (li, old, new) in enumerate(muts))
        mpick = menu_pick(MUT_MENU_Q.format(task=t["task"], fn=t["fn"], menu=mmenu),
                          muts, seed)
        log["score"]["mut_in_menu"] = int(any(
            li == t["line_idx"] and new.strip() == t["orig_line"].strip()
            for li, _, new in muts))
        if mpick is not None:
            li, _, new = mpick
            b2 = list(blines)
            b2[li] = new
            try:
                ftree = ast.parse("\n".join(b2))
                fnode = next((x for x in ast.walk(ftree)
                              if isinstance(x, ast.FunctionDef) and x.name == t["fn"]),
                             None)
                if fnode is not None and norm_dump(fnode) == gold_dump:
                    fixed = 1
            except SyntaxError:
                pass
    log["score"]["fixed"] = fixed
    with open(OUT, "a") as f:
        f.write(json.dumps(log) + "\n")
    return log


if __name__ == "__main__":
    t0 = time.time()
    tasks = harvest_tasks()
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
    print(f"tasks: {len(tasks)}")
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
        print(f"{t['cls']:<10} {t['path'][:42]:<44} "
              f"menu={s.get('file_in_menu')} file={s.get('file_hit')} "
              f"fn={s.get('func_hit')} mut={s.get('mut_in_menu')} fix={s.get('fixed')}",
              flush=True)
        for k, v in s.items():
            agg[k] = agg.get(k, 0) + (v or 0)
    m = len(tasks)
    print(f"\nREAL-REPO v3 n={m} wall={(time.time()-t0)/60:.1f}m")
    for k, v in agg.items():
        print(f"  {k}: {v}/{m}")
