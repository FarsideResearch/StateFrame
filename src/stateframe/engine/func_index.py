"""FUNCTION-GRANULARITY MECHANICAL LAYER.

The file-level locate stage (probes/real_repo_baseline_v2.py build_index/bm25_rank
plus fixer.class_pattern_rank) ranks FILES. Its known residue is density dilution:
a 500-line file with one dense 30-line function scores like a big diluted file, so
the dense function is invisible. the commit: make the FUNCTION the index unit,
keep file-level signals (path words, import names) fused into function scoring
rather than discarded, and derive the file menu from ranked functions so the menu
contract (one entry per file) is unchanged.

This module is build + measurement only. It performs NO model calls. `fixer.py` is
untouched; the orchestrator wires this in after the bench-only shadow comparison
says it should.

Layout:
  CLASS_PATTERNS the ONE construct-regex table (fixer.class_pattern_rank
                          currently inlines an identical 4-family copy; it should
                          import this one when it is next edited).
  build_func_index() one record per function (nested + async included) plus a
                          synthetic '<module>' record per file.
  rank_functions() fused score: function lexical + parent-file lexical +
                          absolute construct count + function-local density.
  file_menu_from_functions() menu contract + membership ledger w/ controlled
                          elimination codes.
  maintainer task driver pairs file-level and function-level measurements
                          over the real corpus and seeded synthetic tasks.
"""
from __future__ import annotations

from collections import deque
import ast
import logging
import math
import pathlib
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# words() is REUSED, never reimplemented: a divergent tokenizer would make the
# function view and the file view incomparable (and the shadow table meaningless).
from probes.real_repo_baseline_v2 import words, bm25_rank  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- construct table
# Same regex families as fixer.class_pattern_rank. Kept here so both views share
# ONE definition of "the implicated construct" — two copies drifting apart would
# silently change what density even measures.
CLASS_PATTERNS = {
    "cmp_flip": r" >= | <= | > | < ",
    "bool_flip": r" and | or ",
    "none_flip": r" is None| is not None",
    "arith_flip": r"\w [+\-] \w",
    # not present in the current injector set, listed for the family named in the
    # mechanical spec so a future injector does not have to re-derive it.
    "not_flip": r"\bnot\b",
}

# ---------------------------------------------------------------- fusion weights
# Documented, not tuned-on-the-test-set. Rationale per weight:
W_FUNC_LEX = 1.0      # the function's own identifiers/docstring: the primary signal.
W_PARENT_LEX = 0.35   # a_mechanical Q1: file signals (path words, imports, module
                      # docstring) FUSE into function scoring, never discarded. Held
                      # well below 1.0 so a hot file cannot carry a cold function.
W_ABS = 1.20          # a_mechanical Q3(d): absolute construct count, so ONE real
                      # occurrence inside a large function is not erased by density.
W_DENSITY = 2.50      # a_mechanical Q3: function-LOCAL density. The denominator is
                      # now the function, which is what removes cross-function
                      # dilution by construction.
DENSITY_REF = 0.25    # constructs-per-line at which the density term saturates.
                      # 1 construct per 4 lines is already a very dense block.

MENU_SECOND_STRONG_FRAC = 0.80  # a second function must score >= 80% of the file's
                                # best to count as independent corroboration.
MENU_SECOND_BONUS = 0.15        # small diminishing-return bonus, applied at most
                                # ONCE. Never a second menu slot.


# ---------------------------------------------------------------- records
@dataclass(slots=True)
class FuncRecord:
    func_id: str                 # path::qualname::lineno — span-scoped, stable
    path: str                    # under overloads/nesting/agent edits 
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    loc: int
    is_module: bool              # the '<module>' fallback record
    ident_bag: dict = field(default_factory=dict)
    doc_bag: dict = field(default_factory=dict)
    construct_counts: dict = field(default_factory=dict)
    parent_path_words: set = field(default_factory=set)
    parent_import_words: set = field(default_factory=set)

    @property
    def bag(self) -> dict:
        """Fused lexical bag: identifiers + docstring words (docstring weighted
        lower — it describes intent, identifiers name the code)."""
        out = dict(self.ident_bag)
        for w, v in self.doc_bag.items():
            out[w] = max(out.get(w, 0.0), v)
        return out


@dataclass
class FuncIndex:
    records: list                 # list[FuncRecord]
    df: dict                      # word -> number of function records containing it
    n: int                        # number of function records
    file_idx: dict                # v2 build_index() per-file bag (parent view)
    file_df: dict
    file_n: int
    adj: dict                     # import adjacency (kept for the file fallback)
    by_path: dict = field(default_factory=dict)


def _count_constructs(text: str) -> dict:
    return {cls: len(re.findall(pat, text)) for cls, pat in CLASS_PATTERNS.items()}


def _add(bag: dict, ws, weight: float) -> None:
    for w in ws:
        bag[w] = max(bag.get(w, 0.0), weight)


def _iter_functions(tree):
    """Every function in the module: nested and async included. Yields
    (node, qualname)."""
    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                yield child, q
                yield from walk(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, f"{prefix}{child.name}.")
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def _function_body_bag(node) -> dict:
    """Identifier bag for one function: its own name, parameters, and every Name /
    Attribute / called symbol inside it (nested functions included — they are also
    indexed separately, so a name appearing in both is deliberate)."""
    return _collect_tree_facts(node)[-1][node]


def _collect_tree_facts(tree):
    """Collect file facts and function bags in one breadth-first AST walk.

    Each node contributes to its enclosing function bags once, avoiding a full
    subtree walk per function. Queue order matches ast.walk, preserving bag and
    file-token insertion order.
    """
    file_bag: dict = {}
    import_words: set[str] = set()
    modules: set[str] = set()
    module_terms: list[tuple[set[str], float]] = []
    body_bags: dict = {}
    pending = deque([(tree, ())])
    while pending:
        node, owners = pending.popleft()
        child_owners = owners
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name_words = words(node.name)
            _add(file_bag, name_words, 2.5)
            _add(file_bag, words((ast.get_docstring(node) or "")[:300]), 1.5)
            for owner in owners:
                _add(body_bags[owner], name_words, 1.5)
            bag: dict = {}
            _add(bag, name_words, 3.0)
            args = node.args
            for arg in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs):
                _add(bag, words(arg.arg), 1.5)
            body_bags[node] = bag
            child_owners = owners + (node,)
        elif isinstance(node, ast.ClassDef):
            name_words = words(node.name)
            _add(file_bag, name_words, 2.5)
            _add(file_bag, words((ast.get_docstring(node) or "")[:300]), 1.5)
            module_terms.append((name_words, 2.5))
            for owner in owners:
                _add(body_bags[owner], name_words, 1.5)
        elif isinstance(node, ast.Name):
            name_words = words(node.id)
            _add(file_bag, name_words, 1.0)
            module_terms.append((name_words, 0.8))
            for owner in owners:
                _add(body_bags[owner], name_words, 1.0)
        elif isinstance(node, ast.Attribute):
            attr_words = words(node.attr)
            for owner in owners:
                _add(body_bags[owner], attr_words, 1.2)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_words.update(words(alias.name))
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_words.update(words(node.module))
            modules.add(node.module.split(".")[0])
        for child in ast.iter_child_nodes(node):
            pending.append((child, child_owners))
    return file_bag, import_words, modules, module_terms, body_bags


def _extract_file_facts(path: str, source: str) -> dict:
    """Build the file and function projections from one transient syntax tree."""
    path_words = words(path)
    lines = source.splitlines()
    file_bag: dict = {}
    _add(file_bag, path_words, 3.0)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"status": "syntax_error", "file_bag": file_bag,
                "imports": [], "records": []}

    tree_file_bag, import_words, modules, module_terms, body_bags = _collect_tree_facts(tree)
    for word, weight in tree_file_bag.items():
        file_bag[word] = max(file_bag.get(word, 0.0), weight)
    _add(file_bag, words((ast.get_docstring(tree) or "")[:300]), 2.0)

    records: list[FuncRecord] = []
    covered = set()
    for node, qualname in _iter_functions(tree):
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        body = "\n".join(lines[start - 1:end])
        doc_bag: dict = {}
        _add(doc_bag, words((ast.get_docstring(node) or "")[:400]), 1.5)
        records.append(FuncRecord(
            func_id=f"{path}::{qualname}::{start}", path=path,
            name=node.name, qualname=qualname, lineno=start,
            end_lineno=end, loc=max(1, end - start + 1), is_module=False,
            ident_bag=body_bags[node], doc_bag=doc_bag,
            construct_counts=_count_constructs(body),
            parent_path_words=path_words, parent_import_words=import_words,
        ))
        if node.col_offset == 0:
            covered.update(range(start, end + 1))

    mod_lines = [line for number, line in enumerate(lines, 1) if number not in covered]
    mod_bag: dict = {}
    _add(mod_bag, import_words, 1.0)
    for tokens, weight in module_terms:
        _add(mod_bag, tokens, weight)
    mod_doc: dict = {}
    _add(mod_doc, words((ast.get_docstring(tree) or "")[:400]), 2.0)
    records.append(FuncRecord(
        func_id=f"{path}::<module>::1", path=path, name="<module>",
        qualname="<module>", lineno=1, end_lineno=max(1, len(lines)),
        loc=max(1, len(mod_lines)), is_module=True,
        ident_bag=mod_bag, doc_bag=mod_doc,
        construct_counts=_count_constructs("\n".join(mod_lines)),
        parent_path_words=path_words, parent_import_words=import_words,
    ))
    return {"status": "ok", "file_bag": file_bag,
            "imports": sorted(modules), "records": records}


def build_func_index(all_files: dict) -> FuncIndex:
    """One record per function, plus one synthetic '<module>' record per file.

    The module record is the declared fallback for
    class-wide / top-level defects and for files whose functions carry none of the
    evidence. A file that fails to parse gets a module record built from its raw
    text so it is never silently dropped from the index.
    """
    file_idx: dict = {}
    file_df: dict = {}
    paths_by_module: dict[str, str | list[str]] = {}
    for path in all_files:
        module = pathlib.Path(path).stem
        existing = paths_by_module.get(module)
        if existing is None:
            paths_by_module[module] = path
        elif isinstance(existing, list):
            existing.append(path)
        else:
            paths_by_module[module] = [existing, path]
    adj = {path: set() for path in all_files}
    records: list = []
    for path, source in all_files.items():
        facts = _extract_file_facts(path, source)
        file_idx[path] = facts["file_bag"]
        if facts["status"] == "ok":
            for word in file_idx[path]:
                file_df[word] = file_df.get(word, 0) + 1
            records.extend(facts["records"])
            for module in facts["imports"]:
                neighbors = paths_by_module.get(module, ())
                if isinstance(neighbors, str):
                    neighbors = (neighbors,)
                for neighbor in neighbors:
                    if neighbor != path:
                        adj[path].add(neighbor)
                        adj[neighbor].add(path)
        else:
            lines = source.splitlines()
            records.append(FuncRecord(
                func_id=f"{path}::<module>::1", path=path, name="<module>",
                qualname="<module>", lineno=1, end_lineno=max(1, len(lines)),
                loc=max(1, len(lines)), is_module=True,
                ident_bag=dict(file_idx[path]),
                construct_counts=_count_constructs(source),
                parent_path_words=words(path),
            ))

    df: dict = {}
    by_path: dict = {}
    for r in records:
        for w in r.bag:
            df[w] = df.get(w, 0) + 1
        by_path.setdefault(r.path, []).append(r)

    return FuncIndex(records=records, df=df, n=len(records), file_idx=file_idx,
                     file_df=file_df, file_n=len(all_files), adj=adj, by_path=by_path)


# ---------------------------------------------------------------- scoring
def _bm25(qwords, bag, df, n, k1=1.5):
    s = 0.0
    for w in qwords:
        f = bag.get(w)
        if f:
            s += math.log(1 + (n - df.get(w, 1) + 0.5) / (df.get(w, 1) + 0.5)) * \
                 (f * (k1 + 1)) / (f + k1)
    return s


def score_function(qwords, cls, rec: FuncRecord, fidx: FuncIndex):
    """Fused score for one record. Returns (score, evidence dict).

    Three independent contributions, per a_mechanical Q3's hierarchy:
      1. lexical over the function's own bag (function granularity)
      2. lexical over the PARENT FILE bag (path words + imports + module doc), so
         file-level signals fuse in instead of being thrown away
      3. class-construct evidence as BOTH an absolute count and a function-local
         density — count so a lone real occurrence survives, density so a function
         that is nothing but the implicated construct outranks one that merely
         mentions it.
    """
    func_bag = rec.bag
    func_lex = _bm25(qwords, func_bag, fidx.df, fidx.n)
    parent_bag = fidx.file_idx.get(rec.path, {})
    parent_lex = _bm25(qwords, parent_bag, fidx.file_df, fidx.file_n)

    cnt = rec.construct_counts.get(cls, 0) if cls else 0
    density = cnt / max(1, rec.loc)
    abs_term = W_ABS * math.log1p(cnt)
    den_term = W_DENSITY * min(density / DENSITY_REF, 1.0)

    score = (W_FUNC_LEX * func_lex + W_PARENT_LEX * parent_lex
             + abs_term + den_term)
    matched = sorted(w for w in qwords if w in func_bag)
    ev = {
        "func_lex": round(func_lex, 3),
        "parent_lex": round(parent_lex, 3),
        "construct_count": cnt,
        "construct_density": round(density, 4),
        "abs_term": round(abs_term, 3),
        "density_term": round(den_term, 3),
        "matched_idents": matched[:12],
    }
    return score, ev


def rank_functions(task_text: str, cls, fidx: FuncIndex):
    """[(score, FuncRecord, evidence)] over EVERY record, best first."""
    qwords = words(task_text)
    out = []
    for rec in fidx.records:
        s, ev = score_function(qwords, cls, rec, fidx)
        out.append((s, rec, ev))
    out.sort(key=lambda t: (-t[0], t[1].func_id))
    return out


def construct_rank_functions(cls, fidx: FuncIndex):
    """LEXICAL-AGNOSTIC arm: rank records purely on class-construct evidence,
    interleaving the absolute-count view and the local-density view exactly as
    class_pattern_rank interleaves windowed count and whole-file density.

    This exists because the file-level cascade unions a construct-only source into
    its pool, and the file view's whole recall edge on vocabulary-disjoint tasks
    (a `__repr__`, a `_map_level` — no docstring, generic name, symptom text shares
    zero tokens) comes from that arm. Scoring alone cannot substitute: the fused
    construct terms are deliberately capped so they cannot outrank real lexical
    evidence, which means a lexically-cold function can never climb on them.
    Recall-first doctrine: union the sources, never serially filter.
    """
    if not cls or cls not in CLASS_PATTERNS:
        return []
    cand = [r for r in fidx.records if r.construct_counts.get(cls, 0) > 0]
    by_count = sorted(cand, key=lambda r: (-r.construct_counts[cls], r.func_id))
    by_density = sorted(
        cand, key=lambda r: (-r.construct_counts[cls] / max(1, r.loc), r.func_id))
    out = []
    seen = set()
    for a, b in zip(by_count, by_density):
        for r in (a, b):
            if r.func_id not in seen:
                seen.add(r.func_id)
                out.append(r)
    return out


def function_pool(task_text, cls, fidx: FuncIndex, pool_cap=32):
    """The function-level analogue of file_cascade_pool: fused-score rank UNION
    construct-only rank, interleaved so neither source starves the pool, capped.
    Returns (pool_records, full_ranked) — full_ranked drives the menu and ledger."""
    ranked = rank_functions(task_text, cls, fidx)
    lex = [r for _, r, _ in ranked]
    cons = construct_rank_functions(cls, fidx)
    merged, seen = [], set()
    for a, b in zip(lex + [None] * len(cons), cons + [None] * len(lex)):
        for r in (a, b):
            if r is not None and r.func_id not in seen:
                seen.add(r.func_id)
                merged.append(r)
    return merged[:pool_cap], ranked


# ---------------------------------------------------------------- symptom anchors
def symptom_phrases(task_text):
    """Quoted phrases carried by the symptom text, longest first.

    Synthetic and real tickets alike quote the thing they are about — a
    docstring line, a description, a rendered function name ('map level'). This
    is the SAME evidence class as fixer.py's snake_case `code_like` anchor, only
    for phrases the ticket spelled with spaces instead of underscores.
    """
    out = re.findall(r"'([^']{3,120})'", task_text or "")
    out += re.findall(r'"([^"]{3,120})"', task_text or "")
    return sorted({p.strip() for p in out if p.strip()}, key=lambda p: -len(p))


def name_phrase_anchors(task_text, fidx: FuncIndex):
    """Function records whose NAME-words exactly equal a quoted symptom phrase's
    words. Measured 2026-08-28 over the 21-task gym: fires on 4 tasks, correct on
    4/4, unique on 3 — zero false positives. Equality (not containment) is what
    keeps it that way: containment would make every `get_x` match 'x'.

    Returns [] when no phrase matches. A phrase matching MANY records (`'repr'`
    matches every `__repr__` in the corpus) is deliberately NOT collapsed here —
    the caller decides, and the uniqueness test is what makes a pin safe.
    """
    hits, seen = [], set()
    for phrase in symptom_phrases(task_text):
        pw = words(phrase)
        if not pw:
            continue
        for rec in fidx.records:
            if rec.is_module:
                continue
            if words(rec.name) == pw and rec.func_id not in seen:
                seen.add(rec.func_id)
                hits.append(rec)
    return hits


def rank_name_phrase_anchors(task_text, cls, fidx: FuncIndex, limit=None):
    """The name-phrase anchor records, best-evidence-first.

    `name_phrase_anchors` deliberately refuses to collapse an AMBIGUOUS match
    ("'repr'" matches every `__repr__` in the corpus), and the caller's pin is
    correctly refused when the match is not unique. That left the whole anchor
    SET unused — which is the one structural recall miss measured 2026-08-28
    (autopsy /tmp/pool_autopsy.txt): with the shipping config the gold file
    reaches the file menu on 20 of 21 gym tasks, and the sole miss is
    `models/learner_concept_gap.py::__repr__`, whose gold file sits at file rank
    40 of the fused ranking but at rank 0 of its own 30-record anchor set.

    An ambiguous anchor is still exact-name evidence: it names the FUNCTION the
    ticket quoted, it just names it many times. Ordering that set by the same
    fused score the ranker already trusts turns it into a short, high-precision
    recall lane. Returns [] when nothing matched.
    """
    recs = name_phrase_anchors(task_text, fidx)
    if not recs:
        return []
    qwords = words(task_text)

    # CLASS FEASIBILITY OUTRANKS LEXICAL SCORE (2026-08-29 final-four autopsy,
    # task #4). An ambiguous name anchor is a set of functions that all share a
    # NAME, so their lexical bags are near-identical and the fused score orders
    # them by noise. MEASURED on `models/learner_concept_gap.py::__repr__`: the
    # phrase "repr" matches 37 `__repr__` records, the gold sat at rank 16 of
    # 37 by fused score (the docstring claiming "rank 0" was stale), and
    # `anchor_recall_k=2` could therefore never reach it.
    #
    # But the symptom names a CLASS, and a function that contains no construct
    # that class can mutate cannot host that bug. Exactly ONE of the 37 records
    # contains an `is`/`is not` comparison for the `none_flip` symptom — the
    # gold. That is not a heuristic, it is a precondition: the fix has to have
    # something to flip.
    #
    # Fails silent by construction: no class, or no record feasible, and the
    # key is constant so the lexical order is returned exactly as before. It
    # only ever REORDERS — no record is dropped, so recall cannot fall.
    def _feasible(rec):
        try:
            return 1 if (cls and (rec.construct_counts or {}).get(cls, 0) > 0) else 0
        except Exception:
            return 0

    if any(_feasible(r) for r in recs):
        scored = [(_feasible(r), score_function(qwords, cls, r, fidx)[0], r)
                  for r in recs]
        scored.sort(key=lambda t: (-t[0], -t[1], t[2].func_id))
        out = [r for _f, _s, r in scored]
    else:
        scored = [(score_function(qwords, cls, r, fidx)[0], r) for r in recs]
        scored.sort(key=lambda t: (-t[0], t[1].func_id))
        out = [r for _, r in scored]
    return out if limit is None else out[:limit]


# ---------------------------------------------------------------- menu contract
def file_menu_from_functions(ranked, k=8, threshold=0.0, append_paths=None,
                             append_k=0):
    """Derive the FILE menu from ranked functions and emit the membership ledger.

    Contract: rank functions internally, emit ONE entry per file at its
    best function's rank. Five strong functions in one file therefore consume one
    slot; the 2nd..Nth are recorded in the ledger as `duplicate_file`, and a second
    INDEPENDENT strong function (>= MENU_SECOND_STRONG_FRAC of the file's best)
    adds MENU_SECOND_BONUS once — corroboration, never capacity.

    Returns (menu, membership) where
      menu = [(path, evidence)] length <= k, evidence carries the best
                   function's name/why and the adjusted score
      membership = [{func_id, path, name, rank, score, state, code,...}] for EVERY
                   considered function, with a controlled elimination code
                   (below_threshold | duplicate_file | menu_capacity).
    """
    entries: dict = {}
    membership: list = []
    for rank, (score, rec, ev) in enumerate(ranked):
        row = {
            "func_id": rec.func_id, "path": rec.path, "name": rec.qualname,
            "rank": rank, "score": round(score, 4), "evidence": ev,
            "state": "considered", "code": None,
        }
        membership.append(row)
        if score <= threshold:
            row["code"] = "below_threshold"
            continue
        row["state"] = "survived"
        e = entries.get(rec.path)
        if e is None:
            entries[rec.path] = {
                "path": rec.path, "best_score": score, "adj_score": score,
                "best_func": rec.qualname, "best_func_id": rec.func_id,
                "best_lineno": rec.lineno, "best_rank": rank,
                "why": ev, "bonus_applied": False, "n_functions": 1,
                "_row": row,
            }
        else:
            e["n_functions"] += 1
            row["code"] = "duplicate_file"
            row["dedup_group"] = e["best_func_id"]
            if (not e["bonus_applied"]
                    and score >= MENU_SECOND_STRONG_FRAC * e["best_score"]):
                e["adj_score"] = e["best_score"] * (1.0 + MENU_SECOND_BONUS)
                e["bonus_applied"] = True
                e["second_func"] = rec.qualname
                row["code"] = "duplicate_file_corroborating"

    ordered = sorted(entries.values(), key=lambda e: (-e["adj_score"], e["path"]))
    menu = []
    for i, e in enumerate(ordered):
        row = e.pop("_row")
        if i < k:
            row["state"] = "surfaced"
            menu.append((e["path"], e))
        else:
            row["code"] = "menu_capacity"
    # ---- APPEND-ONLY RECALL ARM -------------------------------------------
    # Standing rule: a recall improvement must NEVER reorder existing menu
    # entries. Extra candidates (the construct-only union arm, which the fused
    # score deliberately caps so a lexically-cold function can never climb on
    # construct evidence alone) are APPENDED after everything the ranking
    # produced. Measured 2026-08-28: appending 4 rescues 3 of the 4 gym menu
    # misses, where simply widening the ranked menu to k=16 rescued 1.
    if append_paths and append_k > 0:
        have = {p for p, _ in menu}
        for p in append_paths:
            if len(menu) >= k + append_k:
                break
            if p in have:
                continue
            have.add(p)
            best = next((row for row in membership if row["path"] == p), None)
            ev = {"path": p, "best_score": None, "adj_score": None,
                  "best_func": (best or {}).get("name"),
                  "why": (best or {}).get("evidence", {}),
                  "source": "recall_append"}
            menu.append((p, ev))
            if best is not None:
                best["state"] = "surfaced"
                best["code"] = "recall_append"
    return menu, membership


def rank_function_names(task_text, cls, fidx: FuncIndex, path, names):
    """`names` reordered best-evidence-first, ties keeping their original order.

    The function menu used to be presented in AST-walk order with only the single
    best function promoted. Measured 2026-08-28 over the 21-task gym: gold was
    option A in 4/21 under AST order and 19/21 under this score order (21/21 in
    the top three) — and the live 1B answers the function menu with 'A' almost
    every time, so the ORDER is the decision.
    """
    qwords = words(task_text)
    pos = {n: i for i, n in enumerate(names)}
    best, feasible = {}, {}
    for rec in fidx.by_path.get(path, []):
        if rec.is_module:
            continue
        short = rec.qualname.split(".")[-1]
        if short not in pos:
            continue
        s, _ = score_function(qwords, cls, rec, fidx)
        if s > best.get(short, float("-inf")):
            best[short] = s
        if cls and (rec.construct_counts or {}).get(cls, 0) > 0:
            feasible[short] = 1
    # CLASS FEASIBILITY LEADS THE FUNCTION MENU too (2026-08-29, the second half
    # of the task-#4 conversion). Once the insufficient-menu escape had put the
    # run in `models/learner_concept_gap.py`, the FUNCTION menu abstained: the
    # file's functions are lexically indistinguishable to the symptom's words,
    # so the ranker had nothing to say and the 1B answered nothing. The same
    # precondition that ordered the anchor set applies here — a function with no
    # construct of the symptom's class cannot host the bug — and the function
    # menu is decided by ORDER, so this is where it pays.
    #
    # Silent when no function is feasible (the key is constant and the lexical
    # order returns unchanged), and it only ever REORDERS: every name that came
    # in goes out, so no function becomes unreachable.
    if feasible:
        return sorted(names, key=lambda n: (-feasible.get(n, 0),
                                            -best.get(n, float("-inf")), pos[n]))
    return sorted(names, key=lambda n: (-best.get(n, float("-inf")), pos[n]))


def function_evidence_lines(task_text, cls, fidx: FuncIndex, path, names):
    """Per-option evidence for the FUNCTION menu: {fn_name: 'evidence string'}.

    Computed by CODE, never asked of the model (the function menu previously
    showed bare names, and the gym's function.pick failures were dominated by
    files whose gold function was already rank 0 by score — the model simply had
    nothing to read). Each line carries the three things that discriminate:
    which symptom words the function body/docstring actually contains, how many
    of the implicated construct it holds, and how long it is.
    """
    qwords = words(task_text)
    want = {n for n in names}
    out = {}
    for rec in fidx.by_path.get(path, []):
        short = rec.qualname.split(".")[-1]
        if short not in want or rec.is_module:
            continue
        _, ev = score_function(qwords, cls, rec, fidx)
        matched = ev["matched_idents"][:6]
        bits = [f"{len(matched)} symptom words" +
                (f": {', '.join(matched)}" if matched else "")]
        if cls:
            bits.append(f"{ev['construct_count']}x {cls}")
        bits.append(f"{rec.loc} lines")
        line = "; ".join(bits)
        prev = out.get(short)
        # a name defined twice (overload / nested): keep the richer evidence
        if prev is None or len(matched) > prev[1]:
            out[short] = (line, len(matched))
    return {k: v[0] for k, v in out.items()}


# ---------------------------------------------------------------- shared repo reader
REPO_DEFAULT = pathlib.Path("/nonexistent/source-repo/backend/app")
POOL_N = 32
MENU_N = 8


def load_repo_files(root=REPO_DEFAULT):
    """Same exclusions as fixer.load_repo (tests, migrations, dotted paths).
    Duplicated rather than imported so this module never pulls fixer.py into a
    process while a live run owns it.

    An unreadable file is still skipped (the index covers what can be read), but
    the skip is reported in one WARNING per call so a shrunken index is visible."""
    root = pathlib.Path(root)
    files = {}
    skipped = []
    for p in sorted(root.rglob("*.py")):
        if "test" in p.name or "migration" in str(p) or "/." in str(p):
            continue
        rel = str(p.relative_to(root))
        try:
            files[rel] = p.read_text()
        except Exception as error:
            skipped.append(f"{rel} ({type(error).__name__})")
    if skipped:
        logger.warning("load_repo_files skipped %d unreadable file(s): %s", len(skipped),
                       ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    return files
