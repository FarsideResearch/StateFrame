"""Maintainer-only function-index corpus comparison driver."""
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src/stateframe/engine"))
import probes.real_repo_baseline as _baseline
from real_repo_baseline import harvest_tasks
from func_index import (CLASS_PATTERNS, REPO_DEFAULT, load_repo_files, build_func_index,
    function_pool, file_menu_from_functions)
from probes.real_repo_baseline_v2 import build_index, bm25_rank
# The verbatim shadow_compare body imports this name from the old module path.
_baseline.harvest_tasks = harvest_tasks
POOL_N = 32
MENU_N = 8
def class_pattern_rank_files(cls, all_files, top=24):
    """Offline copy of fixer.class_pattern_rank (windowed-count UNION whole-file
    density). Reproduced here — NOT imported — because fixer.py must not be touched
    while a live gym run depends on it; it shares CLASS_PATTERNS so the construct
    definition cannot drift."""
    pat = CLASS_PATTERNS.get(cls)
    if not pat:
        return []
    scored = []
    for path, src in all_files.items():
        lines = src.splitlines()
        hits = len(re.findall(pat, src))
        if not hits:
            continue
        per_line = [len(re.findall(pat, l)) for l in lines]
        w = sum(per_line[:30])
        best_w = w
        for i in range(30, len(per_line)):
            w += per_line[i] - per_line[i - 30]
            best_w = max(best_w, w)
        scored.append((best_w, hits / max(1, len(lines)), hits, path))
    by_window = sorted(scored, key=lambda s: -s[0])
    by_density = sorted(scored, key=lambda s: -s[1])
    out = []
    for a, b in zip(by_window, by_density):
        for s in (a, b):
            if s[3] not in out:
                out.append(s[3])
    return out[:top]


def file_cascade_pool(task_text, cls, all_files, idx, df, n, adj,
                      bm25_top=24, prox_seed=8, pool_cap=32):
    """The current file-level pool, offline: BM25 top-K interleaved with
    class-pattern rank, unioned with import proximity, capped. The 1B relevance
    atoms are deliberately omitted — this is the MECHANICAL layer and the shadow
    comparison must contain no model decision."""
    ranked = bm25_rank(task_text, idx, df, n)
    lex = [p for _, p in ranked[:bm25_top]]
    seeds = lex[:prox_seed]
    prox = set(seeds)
    for p in seeds:
        prox |= adj.get(p, set())
    clsrank = class_pattern_rank_files(cls, all_files)
    merged = []
    for a, b in zip(lex + [None] * len(clsrank), clsrank + [None] * len(lex)):
        for x in (a, b):
            if x is not None and x not in merged:
                merged.append(x)
    return list(dict.fromkeys(merged + sorted(prox)))[:pool_cap]


def shadow_compare(out=None, repo=REPO_DEFAULT):
    """Paired file-level vs function-level measurement. Measurement only: it prints
    whatever the numbers are, including regressions."""
    from probes.real_repo_baseline import harvest_tasks

    tasks = harvest_tasks()
    all_files = load_repo_files(repo)
    idx, df, n, adj = build_index(all_files)
    fidx = build_func_index(all_files)

    rows = []
    for t in tasks:
        gold, gold_fn, cls = t["path"], t["fn"], t["cls"]
        pool_f = file_cascade_pool(t["task"], cls, all_files, idx, df, n, adj,
                                   pool_cap=POOL_N)
        menu_f = pool_f[:MENU_N]

        pool_recs, ranked = function_pool(t["task"], cls, fidx, pool_cap=POOL_N)
        # scoring-only view (no construct-union arm), kept so the table shows what
        # each source actually contributes rather than one blended number.
        top = ranked[:POOL_N]
        pool_fn_files = list(dict.fromkeys(r.path for _, r, _ in top))
        union_files = list(dict.fromkeys(r.path for r in pool_recs))
        menu_fn, _membership = file_menu_from_functions(ranked, k=MENU_N)
        menu_fn_paths = [p for p, _ in menu_fn]

        rows.append(dict(
            path=gold, fn=gold_fn, cls=cls,
            file_pool=gold in pool_f,
            func_pool=gold in pool_fn_files,
            union_pool=gold in union_files,
            union_fn_hit=any(r.path == gold and (r.name == gold_fn or r.qualname.endswith("." + gold_fn))
                             for r in pool_recs),
            func_hit=any(r.path == gold and (r.name == gold_fn or r.qualname.endswith("." + gold_fn))
                         for _, r, _ in top),
            file_menu=gold in menu_f,
            func_menu=gold in menu_fn_paths,
            func_rank=next((i for i, (_, r, _) in enumerate(ranked)
                            if r.path == gold and (r.name == gold_fn or r.qualname.endswith("." + gold_fn))),
                           None),
            file_rank=next((i for i, p in enumerate(pool_f) if p == gold), None),
        ))

    lines = []
    lines.append(f"corpus files={len(all_files)}  function records={fidx.n}  tasks={len(rows)}")
    lines.append("")
    lines.append("fPool/fMenu = file-level cascade (BM25 U class-pattern U import-prox)")
    lines.append("nPool/nMenu = function-level FUSED SCORE only; uPool/uHit = function")
    lines.append("              pool with the construct-only union arm (the file view's equivalent)")
    lines.append("")
    hdr = (f"{'#':<3}{'cls':<11}{'gold path':<52}{'gold fn':<26}"
           f"{'fPool':>6}{'nPool':>6}{'uPool':>6}{'fMenu':>6}{'nMenu':>6}{'fnHit':>6}{'uHit':>6}{'fRank':>7}{'nRank':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(rows):
        lines.append(
            f"{i:<3}{r['cls']:<11}{r['path'][-51:]:<52}{r['fn'][:25]:<26}"
            f"{'Y' if r['file_pool'] else '.':>6}{'Y' if r['func_pool'] else '.':>6}"
            f"{'Y' if r['union_pool'] else '.':>6}"
            f"{'Y' if r['file_menu'] else '.':>6}{'Y' if r['func_menu'] else '.':>6}"
            f"{'Y' if r['func_hit'] else '.':>6}{'Y' if r['union_fn_hit'] else '.':>6}"
            f"{str(r['file_rank']):>7}{str(r['func_rank']):>7}")
    N = len(rows) or 1
    def c(k):
        return sum(1 for r in rows if r[k])
    lines.append("")
    lines.append(f"file-level  pool recall@{POOL_N}: {c('file_pool')}/{N}"
                 f"   menu recall@{MENU_N}: {c('file_menu')}/{N}")
    lines.append(f"func-level  pool recall@{POOL_N}: {c('func_pool')}/{N}"
                 f"   menu recall@{MENU_N}: {c('func_menu')}/{N}"
                 f"   function-hit@{POOL_N}: {c('func_hit')}/{N}")
    lines.append(f"func+union  pool recall@{POOL_N}: {c('union_pool')}/{N}"
                 f"   function-hit@{POOL_N}: {c('union_fn_hit')}/{N}")
    lines.append(f"delta       pool (score-only) {c('func_pool') - c('file_pool'):+d}"
                 f"   pool (w/ union arm) {c('union_pool') - c('file_pool'):+d}"
                 f"   menu {c('func_menu') - c('file_menu'):+d}")
    lines.append("")
    lines.append("per-task regressions (file pool found it, function pool+union did not):")
    reg = [r for r in rows if r["file_pool"] and not r["union_pool"]]
    lines.append("  none" if not reg else
                 "\n".join(f"  {r['path']}::{r['fn']} ({r['cls']})" for r in reg))
    lines.append("per-task gains (function pool+union found it, file did not):")
    gain = [r for r in rows if r["union_pool"] and not r["file_pool"]]
    lines.append("  none" if not gain else
                 "\n".join(f"  {r['path']}::{r['fn']} ({r['cls']})" for r in gain))

    text = "\n".join(lines)
    print(text)
    if out:
        pathlib.Path(out).write_text(text + "\n")
    return rows, text


if __name__ == "__main__":
    shadow_compare(out=(sys.argv[1] if len(sys.argv) > 1 else None))
