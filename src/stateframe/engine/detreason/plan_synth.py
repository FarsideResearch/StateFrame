"""PLAN_SYNTH — bounded plan synthesis (planning v1, module 3).

Violations in, typed plans out, ZERO model calls. Implements the spec committed
in ``/tmp/solq/w16/a_plan_module3.md``, pinned to the shapes that ``chains.py``
actually produces (the spec defers to those wherever it is vague).

The thesis this module exists to test: for the seam families the deterministic
kernel can already SEE, the repair is not a reasoning problem, it is a lookup.
``chains.seam_violations`` tells us a consumer imports a name its producer no
longer defines; the producer's own AST tells us what that name became. If
exactly one candidate survives every uniqueness test, no model needs an opinion.
If two survive, the answer is a MENU, not a guess — the caller may later ask a
1B to pick an ``atom_id``, but this module never calls one.

Three public entry points:

  * ``compile_*(files, violations, ...)`` — one per violation family. Each
    returns a ``Compiled`` (a ``chains``-shaped plan dict), an ``Ambiguous``
    (a bounded menu of typed atoms), or ``None`` when the family does not apply.
  * ``simulate(plan, files)`` — the QUALITY GATE. Applies the plan's edits
    deterministically, re-parses every intermediate state, recomputes the
    violations, and passes only when every targeted baseline violation is gone
    and nothing new appeared. A plan that fails simulation is never returned as
    valid.
  * ``synthesize(files)`` — manifest -> violations -> compilers in declared
    order -> ``{plans, ambiguous, unhandled}``. A TOTAL, honest partition: every
    baseline violation lands in exactly one bucket, with a reason when it lands
    in ``unhandled``.

Fail-closed everywhere. ``None``/``unhandled`` means "not provable here", never
"good enough". The uniqueness tests are deliberately stricter than they need to
be for the harvested corpus, because a wrong plan that simulates green on a
narrow gate is worse than no plan at all.

Deliberate scope boundaries (stated, not hidden — see ``PUNTED``):
  * schema-key drift has no schema/validator index in this corpus to be unique
    against, so it is not compiled; it reports as ``unhandled`` with a reason.
  * ``arity_mismatch`` with too FEW positionals is never repaired: supplying a
    missing required argument means inventing a value.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import tokenize
from dataclasses import dataclass, field

# ------------------------------------------------------------------ constants
#: Menu size cap for an ambiguous decision (spec: at most eight primary atoms).
MENU_CAP = 8

#: Minimum normalized-name similarity for a rename candidate to be considered.
RENAME_SIMILARITY_FLOOR = 0.55

#: Violation kinds a rename repair is allowed to act on (mirrors
#: ``chains_exec.RENAME_EVIDENCE_KINDS`` — kept as its own constant so a change
#: on either side is a visible divergence rather than a silent one).
RENAME_EVIDENCE_KINDS = ("undefined_import", "undefined_name")

#: Extra violation kind this module derives itself. ``seam_violations`` records
#: only the COUNT of keyword arguments, never their names, so an unknown keyword
#: is invisible to it. We recompute it from the AST rather than parsing prose
#: out of a ``detail`` string.
UNKNOWN_KWARG = "unknown_kwarg"

#: Families explicitly NOT compiled, with the reason. Surfaced by
#: ``synthesize`` so a caller never has to guess whether a gap is a bug.
PUNTED = {
    "schema_key_drift":
        "no schema/validator/serializer index exists in this corpus, so an "
        "old->new key-path mapping cannot be proven unique; a guessed key "
        "rename is exactly the speculative edit this module refuses to make",
    "arity_mismatch_too_few":
        "supplying a missing REQUIRED positional argument means inventing a "
        "value; only surplus-argument removal is code-derivable",
    "unknown_attribute":
        "resolving an attribute endpoint to its renamed owner needs type "
        "inference the manifest does not carry",
    "parse_error":
        "a file that does not parse has no seam to repair mechanically",
}


# =====================================================================
# result types
# =====================================================================
@dataclass(frozen=True)
class Compiled:
    """A single-answer compilation: a ``chains``-shaped plan plus its evidence."""
    compiler: str
    plan: dict
    targets: tuple            # violation keys this plan claims to clear
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Ambiguous:
    """2+ surviving candidates. Carries the menu; asks no model."""
    compiler: str
    targets: tuple
    options: tuple            # typed atoms, deterministically ordered
    reason: str = "multiple candidates"

    def atom_ids(self) -> tuple:
        return tuple(o["atom_id"] for o in self.options)


def _vkey(v: dict) -> tuple:
    """The stable identity of a violation (same triple ``chains`` gates on)."""
    return (v["kind"], v["file"], v["name"])


# =====================================================================
# derived violations (keyword arguments)
# =====================================================================
def _module_to_path(module: str, files: dict):
    import chains as _chains
    return _chains.module_to_path(module, files)


def _function_defs(files: dict) -> dict:
    """{(path, name): ast.FunctionDef} for every module-level function."""
    out = {}
    for path in sorted(files):
        try:
            tree = ast.parse(files[path])
        except SyntaxError:
            continue
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault((path, stmt.name), stmt)
    return out


def _param_names(fn) -> tuple:
    a = fn.args
    return tuple(p.arg for p in
                 list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs))


def kwarg_violations(files: dict) -> list:
    """``unknown_kwarg`` records, derived from the AST.

    A call passes ``name=`` to a repo-resolvable function that has no such
    parameter and no ``**kwargs``. Fails OPEN on every unresolvable step: an
    unknown callee, a starred call, or a ``**kwargs`` sink is never a violation.
    """
    fns = _function_defs(files)
    # id() -> defining path, built once. A linear scan per call site turned this
    # into an O(calls x defs) walk on a real repo.
    home = {id(node): path for (path, _n), node in fns.items()}
    out = []
    for path in sorted(files):
        try:
            tree = ast.parse(files[path])
        except SyntaxError:
            continue
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                for a in node.names:
                    imports[a.asname or a.name] = (node.module or "", a.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if any(k.arg is None for k in node.keywords):
                continue                      # **kwargs at the CALL site
            fn = None
            if node.func.id in imports:
                mod, orig = imports[node.func.id]
                tgt = _module_to_path(mod, files)
                if tgt:
                    fn = fns.get((tgt, orig))
            if fn is None:
                fn = fns.get((path, node.func.id))
            if fn is None or fn.args.kwarg is not None:
                continue                      # unknown callee / **kwargs sink
            params = set(_param_names(fn))
            for kw in node.keywords:
                if kw.arg not in params:
                    out.append(dict(
                        kind=UNKNOWN_KWARG, file=path, lineno=kw.lineno,
                        name=kw.arg, callee=node.func.id,
                        callee_file=home.get(id(fn), ""),
                        detail=f"{node.func.id}() accepts "
                               f"{sorted(params)[:8]}, not {kw.arg!r}"))
    out.sort(key=lambda v: (v["kind"], v["file"], v["lineno"], v["name"]))
    return out


def all_violations(files: dict) -> list:
    """Baseline violations = ``chains.seam_violations`` + our derived kinds."""
    import chains as _chains
    manifest = _chains.seam_manifest(files)
    return list(_chains.seam_violations(manifest, files)) + kwarg_violations(files)


# =====================================================================
# shared candidate machinery
# =====================================================================
def _normalize(name: str) -> str:
    return name.replace("_", "").lower()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _keeps_stem(old: str, cand: str) -> bool:
    """Is ``cand`` ``old`` plus (or minus) a whole underscore-delimited tail?

    ``get_db`` -> ``get_db_normalized`` yes; ``get_db`` -> ``get_dbx`` no
    (``dbx`` is a different token, not a qualifier). Token-boundary aware so a
    coincidental substring is not mistaken for a preserved stem.
    """
    if old == cand:
        return False
    a, b = old.split("_"), cand.split("_")
    if len(a) == len(b):
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return long_[:len(short)] == short or long_[-len(short):] == short


def _module_defs(files: dict, path: str) -> list:
    """Module-level definitions of ``path``, deterministically ordered."""
    out = []
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return out
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(dict(name=stmt.name, kind="function", lineno=stmt.lineno))
        elif isinstance(stmt, ast.ClassDef):
            out.append(dict(name=stmt.name, kind="class", lineno=stmt.lineno))
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out.append(dict(name=t.id, kind="assign",
                                    lineno=stmt.lineno))
    out.sort(key=lambda d: (d["name"], d["lineno"]))
    return out


def _names_bound_in(files: dict, path: str) -> set:
    """Every name already meaningful in ``path`` (defs + imports + assignments).

    A rename target that collides with one of these is not unique, whatever the
    similarity score says.
    """
    names = set()
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


def _importing_module(files: dict, viol: dict):
    """(module_string, producer_path) for the import that raised ``viol``."""
    try:
        tree = ast.parse(files.get(viol["file"], ""))
    except SyntaxError:
        return None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.lineno == viol["lineno"]:
            for a in node.names:
                if a.name == viol["name"]:
                    mod = node.module or ""
                    return mod, _module_to_path(mod, files)
    return None, None


def _menu(compiler: str, targets, options: list, reason: str) -> Ambiguous:
    """Bounded, deterministic menu + the mandatory ``NONE_OF_THESE`` escape."""
    primary = options[:MENU_CAP - 1]
    overflow = len(options) - len(primary)
    atoms = []
    for i, opt in enumerate(primary):
        atoms.append(dict(opt, atom_id=f"{compiler}:{i}"))
    atoms.append(dict(atom_id="NONE_OF_THESE", op="abstain",
                      overflow=overflow))
    return Ambiguous(compiler=compiler, targets=tuple(targets),
                     options=tuple(atoms), reason=reason)


def _edit(eid: str, path: str, *, role="consumer", phase="migrate",
          symbol="", old_name="", new_name="", task_text="",
          allowed=None, synth_op="rename", **extra) -> dict:
    """One ``chains.Edit``-shaped dict.

    ``synth_op`` is additive metadata: ``chains.edits_from_plan`` reads only the
    legacy keys and ignores the rest, so a plan built here stays executable by
    ``run_chain`` unchanged while ``simulate`` still knows which transformer to
    apply.
    """
    e = dict(id=eid, file=path, role=role, phase=phase, symbol=symbol or old_name,
             old_name=old_name, new_name=new_name, task_text=task_text,
             allowed_files=list(allowed or (path,)), synth_op=synth_op)
    e.update(extra)
    return e


# =====================================================================
# COMPILER 1 — rename
# =====================================================================
def compile_rename(files: dict, violations: list, viol: dict):
    """``undefined_import``/``undefined_name`` -> a consumer-side rename.

    Unique only when the producer module defines EXACTLY ONE plausible
    replacement: similar name, compatible kind, not already bound in the
    consumer, and not itself already imported by the consumer.
    """
    if viol["kind"] not in RENAME_EVIDENCE_KINDS:
        return None
    old = viol["name"]
    module, producer = _importing_module(files, viol)
    if producer is None or producer not in files:
        return None

    consumer_names = _names_bound_in(files, viol["file"])
    already_imported = _consumer_imports(files, viol["file"], module)
    wanted_kinds = _consumer_uses_as(files, viol["file"], old)

    kept, similar = [], []
    for d in _module_defs(files, producer):
        if d["name"] == old:
            return None                       # the producer still defines it
        if d["name"] in consumer_names or d["name"] in already_imported:
            continue                          # collision: not a free target
        if wanted_kinds and d["kind"] not in wanted_kinds:
            continue
        row = dict(name=d["name"], kind=d["kind"], lineno=d["lineno"],
                   score=round(_similarity(old, d["name"]), 4))
        if _keeps_stem(old, d["name"]):
            kept.append(row)                  # tier 0: the old name survives
        elif row["score"] >= RENAME_SIMILARITY_FLOOR:
            similar.append(row)               # tier 1: merely alike
    # A rename that PRESERVES the original name (a suffix/prefix qualifier) is
    # categorically stronger evidence than one that merely scores well, so a
    # sole tier-0 candidate is not put to a vote against tier-1 siblings. With
    # no tier-0 candidate we fall back to similarity, under the same rule.
    cands = kept or similar
    if not cands:
        return None
    cands.sort(key=lambda c: (-c["score"], c["name"]))

    # UNIQUENESS: a single candidate, or a strictly-better one is NOT enough —
    # a tie or a near-tie is a menu. Two names that both look like the answer
    # are exactly the case a model is for.
    if len(cands) > 1 and cands[0]["score"] - cands[1]["score"] < 1e-9:
        return _menu("rename", [_vkey(viol)], [
            dict(op="rename", producer=producer, consumer=viol["file"],
                 old_name=old, new_name=c["name"], score=c["score"])
            for c in cands], "tie between rename candidates")
    if len(cands) > 1:
        return _menu("rename", [_vkey(viol)], [
            dict(op="rename", producer=producer, consumer=viol["file"],
                 old_name=old, new_name=c["name"], score=c["score"])
            for c in cands], "multiple rename candidates in the producer")

    new = cands[0]["name"]
    plan = dict(
        intent=f"repair the stale reference to {old!r} in {viol['file']}",
        task=f"{producer} renamed {old!r} to {new!r}; {viol['file']} still "
             f"imports and uses the old name",
        edits=[_edit("e1", viol["file"], role="consumer", phase="migrate",
                     symbol=old, old_name=old, new_name=new,
                     task_text=f"rename every reference to {old} into {new} "
                               f"(the definition now lives in {producer} under "
                               f"the new name)",
                     allowed=(viol["file"],), synth_op="rename")],
        allowed_files=[viol["file"]],
    )
    return Compiled("rename", plan, (_vkey(viol),),
                    evidence=dict(producer=producer, module=module,
                                  candidate=cands[0]))


def _consumer_imports(files: dict, path: str, module: str) -> set:
    out = set()
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == module:
            for a in node.names:
                out.add(a.name)
    return out


def _consumer_uses_as(files: dict, path: str, name: str) -> set:
    """How the consumer uses the stale name: ``{'function','class'}`` if called,
    empty (= no constraint) otherwise."""
    kinds = set()
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return kinds
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == name:
            kinds |= {"function", "class"}
    return kinds


# =====================================================================
# COMPILER 2 — moved symbol
# =====================================================================
def compile_moved_symbol(files: dict, violations: list, viol: dict):
    """``undefined_import`` where the symbol NAME still exists, elsewhere.

    Unique only when exactly one other repo file defines it at module level, and
    that file is not the one we already import from. The symbol name is
    preserved; only the import's module path is rewritten.
    """
    if viol["kind"] not in RENAME_EVIDENCE_KINDS:
        return None
    old = viol["name"]
    module, producer = _importing_module(files, viol)
    if module is None:
        return None

    homes = []
    for path in sorted(files):
        if path == viol["file"] or path == producer:
            continue
        for d in _module_defs(files, path):
            if d["name"] == old:
                homes.append(dict(path=path, kind=d["kind"],
                                  lineno=d["lineno"]))
                break
    if not homes:
        return None
    if len(homes) > 1:
        return _menu("moved_symbol", [_vkey(viol)], [
            dict(op="move_import", symbol=old, old_module=module,
                 new_module=_path_to_module(h["path"]), consumer=viol["file"],
                 span=viol["lineno"]) for h in homes],
            "the symbol is defined in several modules")

    new_module = _path_to_module(homes[0]["path"])
    plan = dict(
        intent=f"repoint the import of {old!r} in {viol['file']}",
        task=f"{old!r} moved out of {module!r} and now lives in "
             f"{new_module!r}; {viol['file']} still imports the old module",
        edits=[_edit("e1", viol["file"], role="consumer", phase="migrate",
                     symbol=old, old_name=module, new_name=new_module,
                     task_text=f"import {old} from {new_module} instead of "
                               f"{module}",
                     allowed=(viol["file"],), synth_op="move_import",
                     move_symbol=old, move_lineno=viol["lineno"])],
        allowed_files=[viol["file"]],
    )
    return Compiled("moved_symbol", plan, (_vkey(viol),),
                    evidence=dict(old_module=module, new_module=new_module,
                                  home=homes[0]))


def _path_to_module(path: str) -> str:
    stem = path[:-3] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


# =====================================================================
# COMPILER 3 — arity / deleted parameter
# =====================================================================
_SIDE_EFFECT_FREE = (ast.Constant, ast.Name, ast.Attribute)


def compile_arity(files: dict, violations: list, viol: dict):
    """``arity_mismatch`` -> drop exactly one surplus, side-effect-free argument.

    Unique only when: exactly one call on the line matches, it passes exactly
    one positional too many, the surplus argument is the LAST positional, it is
    side-effect-free, and the whole argument sits on one physical line.
    Too-FEW is never repaired (see ``PUNTED``).
    """
    if viol["kind"] != "arity_mismatch":
        return None
    path, name = viol["file"], viol["name"]
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return None

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == name and n.lineno == viol["lineno"]]
    if len(calls) != 1:
        return None                            # 0 or 2+: not a unique span
    call = calls[0]

    lo, hi = _resolve_arity(files, path, name)
    if hi is None:
        return None
    n = len(call.args)
    if n < lo:
        return None                            # PUNTED: never invent a value
    if n != hi + 1:
        return None                            # more than one surplus: not unique

    surplus = call.args[-1]
    if not isinstance(surplus, _SIDE_EFFECT_FREE):
        return None
    if surplus.lineno != surplus.end_lineno:
        return None
    prev_end = (call.args[-2].end_col_offset if len(call.args) > 1
                else call.func.end_col_offset + 1)
    prev_line = (call.args[-2].end_lineno if len(call.args) > 1 else call.lineno)
    if prev_line != surplus.lineno:
        return None
    if call.keywords:
        return None                            # keywords after the cut: unsafe

    plan = dict(
        intent=f"drop the surplus argument in the call to {name!r}",
        task=f"{name}() no longer accepts {n} positional arguments "
             f"(it accepts {lo}..{hi}); the last one was removed upstream",
        edits=[_edit("e1", path, role="consumer", phase="migrate",
                     symbol=name, old_name="", new_name="",
                     task_text=f"remove the surplus positional argument from "
                               f"the call to {name} on line {viol['lineno']}",
                     allowed=(path,), synth_op="drop_arg",
                     arg_lineno=surplus.lineno, cut_start=prev_end,
                     cut_end=surplus.end_col_offset)],
        allowed_files=[path],
    )
    return Compiled("arity", plan, (_vkey(viol),),
                    evidence=dict(accepts=[lo, hi], passed=n,
                                  dropped=ast.dump(surplus)[:80]))


def _resolve_arity(files: dict, path: str, name: str):
    """(min, max) positional arity of the callee ``name`` as seen from ``path``."""
    import chains as _chains
    fns = _function_defs(files)
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return None, None
    fn = fns.get((path, name))
    if fn is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                for a in node.names:
                    if (a.asname or a.name) == name:
                        tgt = _chains.module_to_path(node.module or "", files)
                        if tgt:
                            fn = fns.get((tgt, a.name))
    if fn is None:
        return None, None
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args)
    lo = max(0, len(pos) - len(a.defaults))
    hi = None if a.vararg is not None else len(pos)
    return lo, hi


# =====================================================================
# COMPILER 4 — wrong keyword
# =====================================================================
def compile_kwarg(files: dict, violations: list, viol: dict):
    """``unknown_kwarg`` -> rename the keyword LABEL to the one free parameter.

    Unique only when exactly one parameter of the callee is both unsupplied by
    this call and similar enough to the wrong key.
    """
    if viol["kind"] != UNKNOWN_KWARG:
        return None
    path, old = viol["file"], viol["name"]
    try:
        tree = ast.parse(files.get(path, ""))
    except SyntaxError:
        return None

    hits = [(n, kw) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == viol["callee"]
            for kw in n.keywords
            if kw.arg == old and kw.lineno == viol["lineno"]]
    if len(hits) != 1:
        return None
    call, kwnode = hits[0]

    fns = _function_defs(files)
    fn = fns.get((viol["callee_file"], viol["callee"]))
    if fn is None:
        return None
    supplied = {k.arg for k in call.keywords if k.arg}
    positional = set(_param_names(fn)[:len(call.args)])
    free = [p for p in _param_names(fn)
            if p not in supplied and p not in positional]

    cands = sorted(
        ((p, round(_similarity(old, p), 4)) for p in free),
        key=lambda t: (-t[1], t[0]))
    cands = [c for c in cands if c[1] >= RENAME_SIMILARITY_FLOOR]
    if not cands:
        return None
    if len(cands) > 1 and cands[0][1] - cands[1][1] < 1e-9:
        return _menu("kwarg", [_vkey(viol)], [
            dict(op="rename_kwarg", old_key=old, new_key=p,
                 call_span=[viol["lineno"], kwnode.col_offset], score=s)
            for p, s in cands], "tie between keyword candidates")
    if len(cands) > 1:
        return _menu("kwarg", [_vkey(viol)], [
            dict(op="rename_kwarg", old_key=old, new_key=p,
                 call_span=[viol["lineno"], kwnode.col_offset], score=s)
            for p, s in cands], "multiple compatible parameters")

    new = cands[0][0]
    plan = dict(
        intent=f"repair the keyword argument {old!r} in the call to "
               f"{viol['callee']!r}",
        task=f"{viol['callee']}() has no parameter {old!r}; the compatible "
             f"parameter is {new!r}",
        edits=[_edit("e1", path, role="consumer", phase="migrate",
                     symbol=viol["callee"], old_name=old, new_name=new,
                     task_text=f"rename the keyword argument {old} to {new} in "
                               f"the call on line {viol['lineno']}",
                     allowed=(path,), synth_op="rename_kwarg",
                     kw_lineno=kwnode.lineno, kw_col=kwnode.col_offset)],
        allowed_files=[path],
    )
    return Compiled("kwarg", plan, (_vkey(viol),),
                    evidence=dict(callee=viol["callee"], new_key=new,
                                  free_params=free))


# =====================================================================
# EDIT TRANSFORMERS  (deterministic, in-memory)
# =====================================================================
def _splice(src: str, lineno: int, col: int, old: str, new: str):
    """Replace exactly ``old`` at (lineno, col). None if the span misreads."""
    lines = src.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        return None
    line = lines[lineno - 1]
    if line[col:col + len(old)] != old:
        return None
    lines[lineno - 1] = line[:col] + new + line[col + len(old):]
    return "".join(lines)


def _rewrite_import_module(src: str, symbol: str, lineno: int,
                           old_module: str, new_module: str):
    """``from OLD import symbol`` -> ``from NEW import symbol`` at one line.

    Token-driven, not regex: the module path is rebuilt from the NAME/DOT tokens
    between ``from`` and ``import`` on that logical line, so a matching string
    literal or comment elsewhere is structurally out of reach.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.lineno == lineno
                 and (n.module or "") == old_module
                 and any(a.name == symbol for a in n.names)), None)
    if node is None:
        return None
    lines = src.splitlines(keepends=True)
    line = lines[lineno - 1]
    head = f"from {old_module} "
    idx = line.find(head)
    if idx < 0:
        return None
    if line.count(head) != 1:
        return None
    lines[lineno - 1] = line.replace(head, f"from {new_module} ", 1)
    return "".join(lines)


def _drop_span(src: str, lineno: int, start: int, end: int):
    """Delete ``[start, end)`` on one line (used for a surplus argument)."""
    lines = src.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        return None
    line = lines[lineno - 1]
    if end > len(line) or start >= end:
        return None
    lines[lineno - 1] = line[:start] + line[end:]
    return "".join(lines)


def apply_edit(files: dict, edit: dict, *, info: dict | None = None):
    """Apply ONE synthesized edit. Returns a new files dict, or None.

    The ``rename`` op delegates to ``chains_exec.mechanical_rename`` so the
    simulation exercises the SAME transformer ``run_chain`` will use — a plan
    that simulates green here is a plan the executor can actually land.
    """
    import chains as _chains
    import chains_exec

    def _bail(why: str):
        if info is not None:
            info["reason"] = why
        return None

    op = edit.get("synth_op", "rename")
    path = edit["file"]
    if path not in files:
        return _bail(f"{path!r} is not in the repo")

    if op == "rename":
        e = _chains.Edit(
            id=edit["id"], file=path, phase=edit.get("phase", "migrate"),
            role=edit.get("role", "consumer"), symbol=edit.get("symbol", ""),
            old_name=edit.get("old_name", ""), new_name=edit.get("new_name", ""),
            task_text=edit.get("task_text", ""),
            allowed_files=tuple(edit.get("allowed_files") or (path,)))
        live = _chains.seam_violations(_chains.seam_manifest(files), files)
        return chains_exec.mechanical_rename(files, e, live, info=info)

    if op == "move_import":
        updated = _rewrite_import_module(
            files[path], edit["move_symbol"], edit["move_lineno"],
            edit["old_name"], edit["new_name"])
    elif op == "drop_arg":
        updated = _drop_span(files[path], edit["arg_lineno"],
                             edit["cut_start"], edit["cut_end"])
    elif op == "rename_kwarg":
        updated = _splice(files[path], edit["kw_lineno"], edit["kw_col"],
                          edit["old_name"], edit["new_name"])
    else:
        return _bail(f"no transformer for synth_op {op!r}")

    if updated is None:
        return _bail(f"{op} transformer could not apply its span")
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        return _bail(f"{op} produced invalid syntax: {exc}")
    out = dict(files)
    out[path] = updated
    return out


# =====================================================================
# THE QUALITY GATE
# =====================================================================
def simulate(plan: dict, files: dict, *, targets=None,
             baseline=None) -> dict:
    """Apply ``plan`` in memory and prove it repairs what it claims.

    Passes only when EVERY targeted baseline violation is gone and NO new
    violation appeared. Every intermediate state must parse. Applying the plan a
    second time to its own output must change nothing (idempotence), which is
    what stops a transformer that "works" by drifting.

    Returns ``{ok, reason, ...}``; ``ok`` is never True without proof.
    """
    import chains as _chains

    baseline = list(baseline if baseline is not None else all_violations(files))
    base_keys = {_vkey(v) for v in baseline}
    targets = tuple(targets or ())
    unknown = [t for t in targets if t not in base_keys]
    if unknown:
        return dict(ok=False, reason=f"targets absent from baseline: {unknown}")

    edits = plan.get("edits") or []
    if not edits:
        return dict(ok=False, reason="plan declares no edits")
    # The plan-level closure is a CEILING, not a suggestion: a per-edit
    # allowed_files may narrow it and may never widen it. Letting an edit widen
    # its own scope is how a "bounded" plan quietly stops being bounded.
    allowed = set(plan.get("allowed_files") or [])
    if not allowed:
        for e in edits:
            allowed |= set(e.get("allowed_files") or [e["file"]])
    else:
        for e in edits:
            escape = set(e.get("allowed_files") or [e["file"]]) - allowed
            if escape:
                return dict(ok=False,
                            reason=f"edit {e['id']} declares files outside the "
                                   f"plan closure: {sorted(escape)}")

    cur = dict(files)
    applied = []
    for e in edits:
        why: dict = {}
        nxt = apply_edit(cur, e, info=why)
        if nxt is None:
            return dict(ok=False,
                        reason=f"edit {e['id']} did not apply: "
                               f"{why.get('reason', 'unknown')}",
                        applied=applied)
        touched = sorted(p for p in nxt if nxt[p] != cur.get(p))
        out_of_scope = [p for p in touched if p not in allowed]
        if out_of_scope:
            return dict(ok=False,
                        reason=f"edit {e['id']} touched undeclared files: "
                               f"{out_of_scope}")
        if not touched:
            return dict(ok=False, reason=f"edit {e['id']} changed nothing")
        for p in touched:                       # every intermediate state parses
            try:
                ast.parse(nxt[p])
            except SyntaxError as exc:
                return dict(ok=False,
                            reason=f"edit {e['id']} broke {p}: {exc}")
        applied.append(e["id"])
        cur = nxt

    after = all_violations(cur)
    after_keys = {_vkey(v) for v in after}
    residual = sorted(t for t in targets if t in after_keys)
    introduced = sorted(after_keys - base_keys)
    if residual:
        return dict(ok=False, reason=f"targeted violation(s) survive: {residual}",
                    applied=applied, final_files=cur)
    if introduced:
        return dict(ok=False,
                    reason=f"plan introduced {len(introduced)} violation(s): "
                           f"{introduced[:3]}",
                    applied=applied, final_files=cur)

    # idempotence: re-running the plan on its own output must be a no-op.
    replay = dict(cur)
    for e in edits:
        again = apply_edit(replay, e)
        if again is not None and again != replay:
            return dict(ok=False,
                        reason=f"edit {e['id']} is not idempotent",
                        applied=applied, final_files=cur)

    return dict(ok=True, reason="", applied=applied, final_files=cur,
                cleared=sorted(base_keys - after_keys),
                resolved_targets=list(targets),
                final_tree_hash=_chains.tree_hash(cur),
                remaining_violations=len(after))


# =====================================================================
# TOP LEVEL
# =====================================================================
#: Compilers, in the declared order the spec fixes.
COMPILERS = (
    ("rename", compile_rename),
    ("moved_symbol", compile_moved_symbol),
    ("arity", compile_arity),
    ("kwarg", compile_kwarg),
)


def synthesize(files: dict) -> dict:
    """violations -> ``{plans, ambiguous, unhandled}``. Total, honest, no model.

    Every baseline violation lands in exactly one bucket:

      * ``plans``     — a compiled plan that PASSED simulation.
      * ``ambiguous`` — 2+ surviving candidates, carrying the menu. A caller may
                        later ask a 1B to pick an ``atom_id``; we never do.
      * ``unhandled`` — with a reason: no compiler applied, two compilers
                        disagreed, or the plan failed its own quality gate.
    """
    baseline = all_violations(files)
    plans, ambiguous, unhandled = [], [], []

    # One DECISION per violation KEY. `seam_violations` reports a site per line,
    # so three stale imports of the same symbol in one file are three records of
    # one seam — and one repair. The gate keys on the same triple, so collapsing
    # here keeps the partition honest instead of triple-counting.
    seen, decisions = set(), []
    for v in baseline:
        if _vkey(v) in seen:
            continue
        seen.add(_vkey(v))
        decisions.append(v)

    for viol in decisions:
        results = []
        for name, fn in COMPILERS:
            try:
                r = fn(files, baseline, viol)
            except Exception as exc:            # noqa: BLE001 — a compiler bug
                unhandled.append(dict(violation=viol, reason=(
                    f"compiler {name} raised {type(exc).__name__}: {exc}")))
                results = []
                break
            if r is not None:
                results.append(r)

        if not results:
            if not any(u["violation"] is viol for u in unhandled):
                unhandled.append(dict(violation=viol, reason=_no_compiler_reason(viol)))
            continue

        compiled = [r for r in results if isinstance(r, Compiled)]
        menus = [r for r in results if isinstance(r, Ambiguous)]

        # Two compilers both claiming a single answer is itself an ambiguity —
        # the families overlapped, and picking the first would be a guess.
        if len(compiled) > 1:
            ambiguous.append(_menu(
                "cross_compiler", [_vkey(viol)],
                [dict(op=c.compiler, plan_intent=c.plan["intent"],
                      edits=len(c.plan["edits"])) for c in compiled],
                "several compilers each produced a single answer"))
            continue
        if not compiled:
            ambiguous.append(menus[0])
            continue

        c = compiled[0]
        sim = simulate(c.plan, files, targets=c.targets, baseline=baseline)
        if not sim["ok"]:
            unhandled.append(dict(violation=viol, reason=(
                f"{c.compiler} plan failed simulation: {sim['reason']}")))
            continue
        plans.append(dict(compiler=c.compiler, plan=c.plan,
                          targets=[list(t) for t in c.targets],
                          evidence=c.evidence, simulation=_thin(sim)))

    return dict(
        baseline_violations=baseline,
        plans=plans,
        ambiguous=[_menu_dict(a) for a in ambiguous],
        unhandled=unhandled,
        punted=dict(PUNTED),
        model_calls=0,
        counts=dict(violations=len(baseline), decisions=len(decisions),
                    plans=len(plans),
                    ambiguous=len(ambiguous), unhandled=len(unhandled)),
    )


def _thin(sim: dict) -> dict:
    return {k: v for k, v in sim.items() if k != "final_files"}


def _menu_dict(a: Ambiguous) -> dict:
    return dict(compiler=a.compiler, reason=a.reason,
                targets=[list(t) for t in a.targets],
                options=[dict(o) for o in a.options])


def _no_compiler_reason(viol: dict) -> str:
    kind = viol["kind"]
    if kind in PUNTED:
        return PUNTED[kind]
    if kind == "arity_mismatch":
        return ("no unique surplus-argument repair: " + PUNTED[
            "arity_mismatch_too_few"] + " (or the call span was not unique / "
            "the surplus argument was not side-effect-free)")
    if kind in RENAME_EVIDENCE_KINDS:
        return ("no unique replacement definition in the producer module and no "
                "single other module defines the symbol")
    if kind == UNKNOWN_KWARG:
        return "no compatible free parameter on the callee"
    return f"no compiler handles {kind!r}"


# =====================================================================
# CLI (inspection only — no model calls)
# =====================================================================
if __name__ == "__main__":            # pragma: no cover
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import chains as _c

    if len(sys.argv) < 2:
        print("usage: python3 -m detreason.plan_synth <repo_root>")
        raise SystemExit(1)
    fs = _c.load_repo_dict(sys.argv[1])
    res = synthesize(fs)
    print(json.dumps(dict(counts=res["counts"],
                          plans=[p["plan"]["intent"] for p in res["plans"]],
                          ambiguous=[a["reason"] for a in res["ambiguous"]]),
                     indent=2))
