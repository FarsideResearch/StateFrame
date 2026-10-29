"""CHAINS — multi-file coding chains v1.

The single-file `fixer.run` is proven. Real changes touch 2-3 files coherently
(definition + consumer). This module is the bounded, transactional graph walk
committed in sol_answers/m6_multifile_a.md (v1 spec) on top of the parent design
in sol_answers/g3_multifile_a.md (producer->consumer ordering, expand/migrate/
contract, salvage-not-abandon).

Three pieces, all deterministic except the per-edit executor:

  1. `seam_manifest(files)` — AST cross-reference extraction over a repo dict.
     For every module-level function/class/class-attribute it records the
     DEFINITION site plus every REFERENCE site across every file (imports,
     calls with arity, attribute reads, string-key subscripts). Line numbers are
     evidence only; the stable identity is (file, qualified symbol).

  2. `seam_violations(manifest)` — the code-owned consistency check. Three
     families that a coordinated rename breaks and a single-file edit cannot see:
       * undefined_import  — `from a import f` where a.py no longer defines f
       * arity_mismatch    — a call whose positional arity no signature accepts
       * unknown_attribute — `obj.attr` where obj's class is repo-resolvable and
                             does not expose attr
     Plus `plan_invariants(...)` for the task-declared old->new migration.

  3. `run_chain(...)` — the executor. task -> manifest -> ordered edit plan
     (producer first, phase-ordered expand/migrate/contract) -> per-edit
     `fixer.run` in a sandbox -> seam-consistency gate after EACH edit ->
     checkpoint/rollback -> chain certificate.

  Verdicts: CHAIN_COMPLETE / PARTIAL_SALVAGED / ABSTAIN. Salvage doctrine: a
  failed edit rolls back ONLY its own candidate overlay; every earlier edit that
  passed its gate is kept and reported.

The ONE model seam is the injected `fixer_run` callable — offline tests script
it and touch nothing else (same discipline as test_fixer_offline.py).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pathlib
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import chains_exec  # noqa: E402  (deterministic edit executor; imports chains lazily)

# ------------------------------------------------------------------ verdicts
CHAIN_COMPLETE = "CHAIN_COMPLETE"
PARTIAL_SALVAGED = "PARTIAL_SALVAGED"
CHAIN_ABSTAIN = "ABSTAIN"

# fixer verdicts the chain is willing to build on
ACCEPTED_FIXER_VERDICTS = ("FIX_PROPOSED",)

PHASE_ORDER = {"expand": 0, "migrate": 1, "contract": 2}

_PY_BUILTIN_ATTRS = {
    "append", "extend", "items", "keys", "values", "get", "pop", "update",
    "join", "split", "strip", "lower", "upper", "format", "replace", "sort",
    "add", "remove", "insert", "copy", "count", "index", "startswith",
    "endswith", "encode", "decode", "read", "write", "close", "setdefault",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def tree_hash(files: dict) -> str:
    """Content-addressed hash of a {path: source} repo snapshot."""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode())
        h.update(b"\0")
        h.update(files[p].encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


# =====================================================================
# 1. SEAM MANIFEST
# =====================================================================
def module_to_path(module: str, files: dict) -> str | None:
    """Resolve a dotted module name onto a key of the repo dict.

    Tolerates the package prefix being present or absent (`app.services.x` vs
    `services/x.py`) by stripping leading components until something matches —
    the AHA corpus is rooted at backend/app but imports itself as `app.*`.
    """
    if not module:
        return None
    parts = module.split(".")
    for start in range(len(parts)):
        stem = "/".join(parts[start:])
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if cand in files:
                return cand
    # last resort: unique basename match
    tail = parts[-1] + ".py"
    hits = [p for p in files if p.endswith("/" + tail) or p == tail]
    return hits[0] if len(hits) == 1 else None


def _arity(node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool):
    """(min_positional, max_positional or None for *args) as CALLERS see it."""
    a = node.args
    pos = list(a.posonlyargs) + list(a.args)
    if is_method and pos and pos[0].arg in ("self", "cls"):
        pos = pos[1:]
    ndef = len(a.defaults)
    lo = max(0, len(pos) - ndef)
    hi = None if a.vararg is not None else len(pos)
    return lo, hi


def _class_attributes(cls: ast.ClassDef) -> dict:
    """Attribute names a class exposes: class-level assignments/annotations,
    methods, and `self.x = ...` bound anywhere in its body."""
    attrs = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    attrs[t.id] = "classvar"
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            attrs[stmt.target.id] = "annotation"
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs[stmt.name] = "method"
    for node in ast.walk(cls):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self" and isinstance(node.ctx, ast.Store):
            attrs.setdefault(node.attr, "instance")
    return attrs


class _FileScan(ast.NodeVisitor):
    """One file's definitions + references. Deliberately shallow: module-level
    definitions and every reference site anywhere in the file."""

    def __init__(self, path: str):
        self.path = path
        self.definitions = []
        self.references = []
        self.imports = {}          # local name -> {"module":..., "orig":...}
        self.instances = {}        # local var name -> class name (from `V = C(...)`)
        self._binds = {}           # name -> {classes} — ambiguous names are dropped

    def _bind(self, name, cls):
        self._binds.setdefault(name, set()).add(cls)

    def finish(self):
        """A name bound to two different classes is not evidence of anything —
        keep only unambiguous bindings (fail-open, never a false violation)."""
        self.instances = {n: next(iter(c)) for n, c in self._binds.items()
                          if len(c) == 1}

    # -- definitions
    def visit_Module(self, node):
        for stmt in node.body:
            self._top_level(stmt)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # `v = ClassName(...)` ANYWHERE (function bodies included) binds v to a
        # class we may know — module-level-only tracking missed every local.
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._bind(t.id, node.value.func.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
            self._bind(node.target.id, node.annotation.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        if isinstance(node.annotation, ast.Name):
            self._bind(node.arg, node.annotation.id)
        self.generic_visit(node)

    def _top_level(self, stmt):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo, hi = _arity(stmt, is_method=False)
            self.definitions.append(dict(
                kind="function", name=stmt.name, qualname=stmt.name,
                file=self.path, lineno=stmt.lineno, arity_min=lo, arity_max=hi))
        elif isinstance(stmt, ast.ClassDef):
            self.definitions.append(dict(
                kind="class", name=stmt.name, qualname=stmt.name,
                file=self.path, lineno=stmt.lineno,
                # a non-Name base (module.Base, a generic subscript, a factory
                # call) is recorded as OPAQUE so the attribute check fails open
                bases=[b.id if isinstance(b, ast.Name) else "<opaque>"
                       for b in stmt.bases]))
            for attr, how in sorted(_class_attributes(stmt).items()):
                d = dict(kind="classattr", name=attr,
                         qualname=f"{stmt.name}.{attr}", owner=stmt.name,
                         how=how, file=self.path, lineno=stmt.lineno)
                if how == "method":
                    for s in stmt.body:
                        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                                and s.name == attr:
                            lo, hi = _arity(s, is_method=True)
                            d["arity_min"], d["arity_max"] = lo, hi
                            d["lineno"] = s.lineno
                self.definitions.append(d)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    self.definitions.append(dict(
                        kind="assign", name=t.id, qualname=t.id,
                        file=self.path, lineno=stmt.lineno))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            self.definitions.append(dict(
                kind="assign", name=stmt.target.id, qualname=stmt.target.id,
                file=self.path, lineno=stmt.lineno))

    # -- references
    def visit_ImportFrom(self, node):
        for a in node.names:
            local = a.asname or a.name
            self.imports[local] = {"module": node.module or "", "orig": a.name}
            self.references.append(dict(
                kind="import", name=a.name, local=local, module=node.module or "",
                level=node.level, file=self.path, lineno=node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            local = (a.asname or a.name).split(".")[0]
            self.imports[local] = {"module": a.name, "orig": None}
            self.references.append(dict(
                kind="import_module", name=a.name, local=local,
                module=a.name, level=0, file=self.path, lineno=node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        nkw = len(node.keywords)
        has_star = any(isinstance(a, ast.Starred) for a in node.args)
        if isinstance(f, ast.Name):
            self.references.append(dict(
                kind="call", name=f.id, file=self.path, lineno=node.lineno,
                nargs=len(node.args), nkw=nkw, starred=has_star))
        elif isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else None
            self.references.append(dict(
                kind="call_attr", name=f.attr, base=base, file=self.path,
                lineno=node.lineno, nargs=len(node.args), nkw=nkw,
                starred=has_star))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            self.references.append(dict(
                kind="attribute", name=node.attr, base=node.value.id,
                file=self.path, lineno=node.lineno))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        s = node.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            self.references.append(dict(
                kind="key", name=s.value,
                base=node.value.id if isinstance(node.value, ast.Name) else None,
                file=self.path, lineno=node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.references.append(dict(kind="name", name=node.id,
                                        file=self.path, lineno=node.lineno))
        self.generic_visit(node)


def seam_manifest(files: dict) -> dict:
    """AST cross-reference manifest over a {path: source} repo dict.

    Returns:
      base_tree_hash, files (parsed/unparsed), definitions, references,
      by_name index, classes {name: {attrs}}, seams {qualname: seam},
      manifest_hash.

    A SEAM is a definition plus every reference site that resolves to it. A seam
    is `cross_file` when at least one reference lives outside the defining file —
    those are exactly the ones a single-file fixer cannot keep consistent.
    """
    definitions, references, parse_errors = [], [], {}
    per_file = {}
    for path in sorted(files):
        src = files[path]
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            parse_errors[path] = f"{e.msg} (line {e.lineno})"
            continue
        scan = _FileScan(path)
        scan.visit(tree)
        scan.finish()
        per_file[path] = scan
        definitions.extend(scan.definitions)
        references.extend(scan.references)

    by_name = {}
    for d in definitions:
        by_name.setdefault(d["name"], []).append(d)
    classes = {d["name"]: {} for d in definitions if d["kind"] == "class"}
    for d in definitions:
        if d["kind"] == "classattr":
            classes.setdefault(d["owner"], {})[d["name"]] = d["how"]

    # --- seams: definition -> its reference sites
    seams = {}
    for d in definitions:
        sites = [r for r in references
                 if r["name"] == d["name"] and r["kind"] != "name"]
        # a bare Name load only counts for module-level defs (not class attrs)
        if d["kind"] in ("function", "class", "assign"):
            sites += [r for r in references
                      if r["kind"] == "name" and r["name"] == d["name"]]
        sites = [r for r in sites
                 if not (r["file"] == d["file"] and r["lineno"] == d["lineno"])]
        sid = f"{d['file']}::{d['qualname']}"
        seams[sid] = dict(
            id=sid, definition=d, references=sites,
            cross_file=any(r["file"] != d["file"] for r in sites),
            ref_files=sorted({r["file"] for r in sites}))

    manifest = dict(
        base_tree_hash=tree_hash(files),
        files=sorted(files),
        parse_errors=parse_errors,
        definitions=definitions,
        references=references,
        by_name={k: len(v) for k, v in by_name.items()},
        classes=classes,
        seams=seams,
        imports={p: s.imports for p, s in per_file.items()},
        instances={p: s.instances for p, s in per_file.items()},
    )
    manifest["manifest_hash"] = _sha(json.dumps(
        {k: manifest[k] for k in ("base_tree_hash", "files")},
        sort_keys=True) + str(sorted(seams)))
    return manifest


def cross_file_seams(manifest: dict) -> list:
    """The seams a chain exists to protect, deterministically ordered."""
    return sorted((s for s in manifest["seams"].values() if s["cross_file"]),
                  key=lambda s: s["id"])


# =====================================================================
# 2. SEAM VIOLATIONS  (the gate)
# =====================================================================
def seam_violations(manifest: dict, files: dict) -> list:
    """Intrinsic cross-file inconsistencies. Deterministic, no model, no exec.

    Every violation is {kind, file, lineno, name, detail} and is stable across
    runs so a gate can diff BEFORE vs AFTER an edit and charge only what the
    edit actually introduced.
    """
    out = []
    if manifest["parse_errors"]:
        for p, msg in sorted(manifest["parse_errors"].items()):
            out.append(dict(kind="parse_error", file=p, lineno=0, name=p,
                            detail=msg))

    # module-level definitions per file, for import resolution
    defs_in_file = {}
    for d in manifest["definitions"]:
        if d["kind"] in ("function", "class", "assign"):
            defs_in_file.setdefault(d["file"], set()).add(d["name"])
    # names a file also re-exports via its own imports
    for p, imps in manifest["imports"].items():
        defs_in_file.setdefault(p, set()).update(imps)

    fn_defs = {}
    for d in manifest["definitions"]:
        if d["kind"] == "function":
            fn_defs.setdefault((d["file"], d["name"]), d)

    # ---- (a) undefined_import: repo-internal import of a name that is gone
    for r in manifest["references"]:
        if r["kind"] != "import" or r.get("level"):
            continue
        target = module_to_path(r["module"], files)
        if target is None or target not in defs_in_file:
            continue                                   # external / unresolvable
        if r["name"] == "*":
            continue
        # `from pkg import submod` is a SUBMODULE import, not a name lookup in
        # pkg/__init__.py — measured as 53/93 of the AHA false positives.
        if module_to_path(f"{r['module']}.{r['name']}", files):
            continue
        if r["name"] not in defs_in_file[target]:
            out.append(dict(kind="undefined_import", file=r["file"],
                            lineno=r["lineno"], name=r["name"],
                            detail=f"{target} does not define {r['name']!r}"))

    # ---- (b) arity_mismatch: a call whose positional count no signature accepts
    for r in manifest["references"]:
        if r["kind"] != "call" or r.get("starred"):
            continue
        imps = manifest["imports"].get(r["file"], {})
        d = None
        if r["name"] in imps and imps[r["name"]].get("orig"):
            tgt = module_to_path(imps[r["name"]]["module"], files)
            if tgt:
                d = fn_defs.get((tgt, imps[r["name"]]["orig"]))
        if d is None:
            d = fn_defs.get((r["file"], r["name"]))
        if d is None:
            continue
        lo, hi = d["arity_min"], d["arity_max"]
        n = r["nargs"]
        # a call that passes arguments BY NAME cannot be judged on count alone
        # (we do not match keyword names to parameter names) — fail open on the
        # lower bound and only ever flag TOO MANY positionals.
        too_few = (r["nkw"] == 0 and n < lo)
        if too_few or (hi is not None and n > hi):
            out.append(dict(
                kind="arity_mismatch", file=r["file"], lineno=r["lineno"],
                name=r["name"],
                detail=f"call passes {n} positional (+{r['nkw']} kw); "
                       f"{d['file']}::{d['name']} accepts {lo}..{hi}"))

    # ---- (c) unknown_attribute: obj.attr where obj's class is repo-resolvable
    classes = manifest["classes"]
    for r in manifest["references"]:
        if r["kind"] not in ("attribute", "call_attr"):
            continue
        base = r.get("base")
        if not base:
            continue
        cls = None
        if base in classes:                       # ClassName.attr
            cls = base
        else:
            cls = manifest["instances"].get(r["file"], {}).get(base)
        if cls is None or cls not in classes:
            continue
        attrs = classes[cls]
        if r["name"] in attrs or r["name"] in _PY_BUILTIN_ATTRS \
                or r["name"].startswith("__"):
            continue
        # A class inheriting from something OUTSIDE the corpus (pydantic
        # BaseModel, SQLAlchemy Base, Enum...) has an unknown attribute surface,
        # so no absence is evidence — fail open. This was 37/93 of the AHA
        # false positives (`.model_validate` on a pydantic model).
        inherited, opaque, seen, queue = set(), False, {cls}, [cls]
        while queue:
            cur = queue.pop()
            cdef = next((d for d in manifest["definitions"]
                         if d["kind"] == "class" and d["name"] == cur), None)
            for b in (cdef or {}).get("bases", []):
                if b not in classes:
                    opaque = True          # ancestor outside the corpus
                elif b not in seen:
                    seen.add(b)
                    queue.append(b)
                    inherited |= set(classes[b])
        if opaque or r["name"] in inherited:
            continue
        out.append(dict(kind="unknown_attribute", file=r["file"],
                        lineno=r["lineno"], name=r["name"],
                        detail=f"class {cls} exposes "
                               f"{sorted(attrs)[:8]}, not {r['name']!r}"))

    out.sort(key=lambda v: (v["kind"], v["file"], v["lineno"], v["name"]))
    return out


def _violation_key(v):
    return (v["kind"], v["file"], v["name"])


def plan_invariants(files: dict, edits: list, completed_ids: set) -> list:
    """Task-declared migration invariants for the edits done SO FAR.

    Per m6: producer exposes new_name; consumers reference new_name; no stale
    old_name reference remains in a file whose edit is complete.
    """
    out = []
    for e in edits:
        if e.id not in completed_ids:
            continue
        src = files.get(e.file, "")
        if e.new_name and not re.search(rf"\b{re.escape(e.new_name)}\b", src):
            out.append(dict(kind="new_name_absent", file=e.file, lineno=0,
                            name=e.new_name,
                            detail=f"edit {e.id} completed but {e.file} never "
                                   f"mentions {e.new_name!r}"))
        if e.old_name and re.search(rf"\b{re.escape(e.old_name)}\b", src):
            out.append(dict(kind="stale_old_name", file=e.file, lineno=0,
                            name=e.old_name,
                            detail=f"edit {e.id} completed but {e.file} still "
                                   f"references {e.old_name!r}"))
    return out


def _covered_by_pending_edit(v, edits, completed_ids) -> str | None:
    """Is this violation the EXPECTED, declared mid-migration state?

    A producer rename legitimately leaves its consumers stale until their own
    edit runs — g3's expand/migrate/contract exists precisely because that
    window is unavoidable. A violation is DEFERRED (not charged) only when a
    still-pending declared edit owns both the file and the symbol; anything else
    is a real break, which is what makes the single-edit case still fail closed.
    """
    for e in edits:
        if e.id in completed_ids:
            continue
        if v["file"] != e.file:
            continue
        if v["name"] and v["name"] in (e.old_name, e.new_name, e.symbol):
            return e.id
    return None


def seam_gate(files: dict, edits: list, completed_ids: set,
              baseline_violations: list) -> dict:
    """The post-edit consistency gate. Fails on NEWLY-introduced violations or a
    broken plan invariant; pre-existing repo debt is never charged to the chain,
    and a violation a pending declared edit will repair is deferred, not charged.
    The FINAL gate has no pending edits, so nothing can hide behind a deferral."""
    manifest = seam_manifest(files)
    viols = seam_violations(manifest, files)
    base_keys = {_violation_key(v) for v in baseline_violations}
    fresh = [v for v in viols if _violation_key(v) not in base_keys]
    introduced, deferred = [], []
    for v in fresh:
        owner = _covered_by_pending_edit(v, edits, completed_ids)
        if owner:
            deferred.append(dict(v, deferred_to_edit=owner))
        else:
            introduced.append(v)
    inv = plan_invariants(files, edits, completed_ids)
    resolved = [v for v in baseline_violations
                if _violation_key(v) not in {_violation_key(x) for x in viols}]
    return dict(
        ok=not introduced and not inv,
        manifest_hash=manifest["manifest_hash"],
        tree_hash=manifest["base_tree_hash"],
        total_violations=len(viols),
        introduced=introduced,
        deferred=deferred,
        invariant_failures=inv,
        resolved=resolved,
    )


# =====================================================================
# 3. THE CHAIN EXECUTOR
# =====================================================================
@dataclass
class Edit:
    """One bounded, single-file action the fixer executes."""
    id: str
    file: str
    phase: str = "migrate"          # expand | migrate | contract
    role: str = "consumer"          # producer | consumer
    symbol: str = ""
    old_name: str = ""
    new_name: str = ""
    task_text: str = ""
    allowed_files: tuple = ()

    def sort_key(self):
        return (PHASE_ORDER.get(self.phase, 9),
                0 if self.role == "producer" else 1, self.file, self.symbol)


def order_edits(edits: list) -> list:
    """Producer-first topological order (g3): contract anchors/producers, then
    consumers, then cleanup. Ties broken deterministically."""
    return sorted(edits, key=lambda e: e.sort_key())


def edits_from_plan(plan: dict) -> list:
    """Build the ordered edit list from a task's declared plan dict."""
    out = []
    for i, e in enumerate(plan.get("edits", [])):
        out.append(Edit(
            id=e.get("id") or f"e{i+1}",
            file=e["file"],
            phase=e.get("phase", "migrate"),
            role=e.get("role", "consumer"),
            symbol=e.get("symbol", ""),
            old_name=e.get("old_name", ""),
            new_name=e.get("new_name", ""),
            task_text=e.get("task_text", ""),
            allowed_files=tuple(e.get("allowed_files") or (e["file"],)),
        ))
    return order_edits(out)


class _Sandbox:
    """A disposable on-disk copy of the repo dict. fixer.run wants a root."""

    def __init__(self, files: dict):
        self.files = dict(files)
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="chain-sbx-"))
        self.flush()

    def flush(self):
        for rel, src in self.files.items():
            p = self.dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src)

    def reload(self):
        for rel in list(self.files):
            p = self.dir / rel
            if p.exists():
                self.files[rel] = p.read_text()
        return self.files

    def restore(self, snapshot: dict):
        self.files = dict(snapshot)
        self.flush()

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def _invoke_fixer(fixer_run, root: str, task_text: str, cfg, target_file: str):
    """Call the executor with the plan-pinned target file.

    Signature tolerance is deliberate: the offline suites and the gym benches
    script their own fixer callables, and some accept neither ``cfg`` nor
    ``target_file``. Introspect ONCE (never a blind ``except TypeError``, which
    would swallow a real TypeError raised inside the fixer body).
    """
    try:
        params = inspect.signature(fixer_run).parameters
        accepts = set(params)
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in params.values())
    except (TypeError, ValueError):          # builtins / C callables
        accepts, var_kw = set(), False
    kw = {}
    if var_kw or "cfg" in accepts:
        kw["cfg"] = cfg
    if var_kw or "target_file" in accepts:
        kw["target_file"] = target_file
    if var_kw or "apply" in accepts:
        return fixer_run(root, task_text, apply=True, **kw)
    return fixer_run(root, task_text, True, *([cfg] if "cfg" in accepts else []))


def run_chain(files: dict, plan: dict, *, fixer_run=None, cfg=None,
              accepted_verdicts=ACCEPTED_FIXER_VERDICTS,
              max_edits=6, use_mechanical=True) -> dict:
    """Execute a multi-file chain and return the CHAIN CERTIFICATE.

    files          {relpath: source} — the starting repo state (not mutated)
    plan           {"intent":..., "edits":[{file, role, phase, old_name,
                   new_name, symbol, task_text, allowed_files}], "task": ...}
    fixer_run      callable(root:str, task_text:str, apply:bool, cfg) -> cert.
                   Defaults to fixer.run (imported lazily so offline tests that
                   pass a scripted callable never touch the model client).

    use_mechanical  try the DETERMINISTIC edit executor (chains_exec.
                   mechanical_rename) before spending a model call. Default ON:
                   a declared rename backed by a seam violation is code-
                   derivable, and the fixer's function menu cannot even
                   represent a module-level import line, so it would abstain.
                   Set False ONLY in tests that must exercise the model path.

    Per edit: checkpoint -> code executor (or fixer) -> scope check -> seam gate
    -> accept, else
    roll back ONLY this edit's overlay and stop. Everything already accepted is
    kept (salvage doctrine, g3 (d)).
    """
    t0 = time.time()
    if fixer_run is None:
        import fixer as _fixer
        fixer_run = _fixer.run

    edits = edits_from_plan(plan)
    base_manifest = seam_manifest(files)
    baseline_violations = seam_violations(base_manifest, files)

    cert = dict(
        intent=plan.get("intent", ""),
        task=plan.get("task", ""),
        base_tree_hash=base_manifest["base_tree_hash"],
        base_manifest_hash=base_manifest["manifest_hash"],
        plan_hash=_sha(json.dumps(plan, sort_keys=True, default=str)),
        allowed_files=sorted({f for e in edits for f in e.allowed_files}),
        baseline_violations=baseline_violations,
        cross_file_seams=[s["id"] for s in cross_file_seams(base_manifest)],
        edits=[], seam_checks=[], rollbacks=[],
        completed_edit_ids=[], verdict=CHAIN_ABSTAIN,
    )

    if not edits:
        cert["reason"] = "plan declares no edits"
        return cert
    if len(edits) > max_edits:
        cert["reason"] = f"plan declares {len(edits)} edits (max {max_edits})"
        return cert
    missing = [e.file for e in edits if e.file not in files]
    if missing:
        cert["reason"] = f"plan targets files absent from the repo: {missing}"
        return cert

    sbx = _Sandbox(files)
    completed: set = set()
    try:
        for e in edits:
            checkpoint = dict(sbx.files)
            cp_hash = tree_hash(checkpoint)
            row = dict(id=e.id, file=e.file, role=e.role, phase=e.phase,
                       symbol=e.symbol, old_name=e.old_name,
                       new_name=e.new_name, checkpoint_hash=cp_hash,
                       status="pending")

            # ---- per-edit executor
            # CODE FIRST. A declared rename whose evidence is a seam violation
            # is fully derivable from the AST — and the fixer's function menu
            # cannot even represent a module-level import line, so sending it
            # buys a guaranteed ABSTAIN at model price. mechanical_rename
            # fails closed: None means genuine ambiguity, and the model seam
            # below runs exactly as it always did.
            mech_info: dict = {}
            mech_files = None
            if use_mechanical:
                live_violations = seam_violations(seam_manifest(checkpoint),
                                                  checkpoint)
                mech_files = chains_exec.mechanical_rename(
                    checkpoint, e, live_violations, info=mech_info)
            if mech_files is not None:
                sbx.restore(mech_files)
                fcert = chains_exec.mechanical_cert(mech_info)
                row["executed_by"] = "code"
            else:
                row["executed_by"] = "model"
                row["mechanical_declined"] = mech_info.get("reason", "")
                # The plan already KNOWS the target file — pin it so the fixer
                # cannot re-locate to a different file (which then fails the
                # scope check and abstains a chain it could have completed).
                try:
                    fcert = _invoke_fixer(fixer_run, str(sbx.dir), e.task_text,
                                          cfg, e.file)
                except Exception as exc:                # noqa: BLE001
                    fcert = dict(verdict="ABSTAIN",
                                 reason=f"{type(exc).__name__}: {exc}")
            row["fixer_cert"] = {k: v for k, v in (fcert or {}).items()
                                 if k != "config"}

            after = dict(sbx.reload())
            touched = sorted(p for p in after if after[p] != checkpoint[p])
            row["files_touched"] = touched

            reason = None
            if row["executed_by"] == "code":
                pass          # no verdict to judge: the code IS the evidence
            elif (fcert or {}).get("verdict") not in accepted_verdicts:
                reason = (f"fixer verdict {(fcert or {}).get('verdict')!r}: "
                          f"{(fcert or {}).get('reason', '')}".strip())
            elif not touched:
                reason = "fixer reported a fix but wrote nothing"
            else:
                out_of_scope = [p for p in touched if p not in e.allowed_files]
                if out_of_scope:
                    reason = f"edit touched undeclared files: {out_of_scope}"

            if reason is None:
                gate = seam_gate(after, edits, completed | {e.id},
                                 baseline_violations)
                gate["edit_id"] = e.id
                cert["seam_checks"].append(gate)
                row["seam_gate"] = "pass" if gate["ok"] else "FAIL"
                if not gate["ok"]:
                    reason = ("seam gate: " + json.dumps(
                        {"introduced": gate["introduced"][:3],
                         "invariants": gate["invariant_failures"][:3]}))
            else:
                row["seam_gate"] = "not_reached"

            if reason is None:
                row["status"] = "accepted"
                row["post_tree_hash"] = tree_hash(after)
                completed.add(e.id)
                cert["edits"].append(row)
                continue

            # ---- failure: roll back ONLY this candidate overlay
            sbx.restore(checkpoint)
            row["status"] = "failed"
            row["reason"] = reason
            row["rolled_back_to"] = cp_hash
            cert["rollbacks"].append(
                dict(edit_id=e.id, restored_tree_hash=cp_hash, reason=reason))
            cert["edits"].append(row)
            cert["reason"] = f"edit {e.id} failed: {reason}"
            break

        final_files = dict(sbx.files)
    finally:
        sbx.close()

    cert["completed_edit_ids"] = sorted(completed)
    cert["final_tree_hash"] = tree_hash(final_files)
    cert["final_files"] = final_files
    final_manifest = seam_manifest(final_files)
    final_gate = seam_gate(final_files, edits, completed, baseline_violations)
    final_gate["edit_id"] = "final"
    cert["seam_checks"].append(final_gate)
    cert["final_manifest_hash"] = final_manifest["manifest_hash"]

    n_ok = len(completed)
    if n_ok == len(edits) and final_gate["ok"]:
        cert["verdict"] = CHAIN_COMPLETE
        cert.pop("reason", None)
    elif n_ok > 0:
        cert["verdict"] = PARTIAL_SALVAGED
        cert.setdefault("reason", "chain did not complete every edit")
        cert["salvaged"] = sorted(completed)
    else:
        cert["verdict"] = CHAIN_ABSTAIN
    cert["wall_s"] = round(time.time() - t0, 2)
    return cert


# =====================================================================
# 4. UNCERTAINTY ROWS (episodic ledger tie-in)
# =====================================================================
def chain_uncertainty_rows(cert: dict, *, decision_id: str = "") -> list:
    """Chain certificate -> detreason.uledger URows.

    A failed edit is an unknown_atom; a surviving seam violation is an
    unverified_claim; an unattempted suffix after a salvage is an
    unexplored_branch.
    """
    from detreason.uledger import (
        URow, ReductionAction, UNKNOWN_ATOM, UNVERIFIED_CLAIM,
        UNEXPLORED_BRANCH, DETERMINISTIC_CHECK, EVIDENCE_SPAN)

    base = decision_id or f"chain:{cert.get('plan_hash', 'plan')}"
    rows = []

    for e in cert.get("edits", []):
        if e.get("status") != "failed":
            continue
        rows.append(URow(
            id=f"{base}#edit:{e['id']}",
            utype=UNKNOWN_ATOM,
            claim=f"edit {e['id']} on {e['file']} failed: {e.get('reason','')}",
            severity=0.9, irreversibility=0.0, u=1.0,
            closure_criterion=EVIDENCE_SPAN,
            correlation_group=f"{base}/edit",
            owner="chains", source="chain_cert",
            reduction_actions=[
                ReductionAction("replan_suffix_from_checkpoint", "cheap", 0.6),
                ReductionAction("escalate_to_larger_model", "expensive", 0.9),
            ]))

    final = next((g for g in cert.get("seam_checks", [])
                  if g.get("edit_id") == "final"), None)
    if final and not final["ok"]:
        rows.append(URow(
            id=f"{base}#seam",
            utype=UNVERIFIED_CLAIM,
            claim=f"{len(final['introduced'])} introduced seam violation(s) and "
                  f"{len(final['invariant_failures'])} invariant failure(s) "
                  f"survive the chain",
            severity=0.85, irreversibility=0.4, u=0.8,
            closure_criterion=DETERMINISTIC_CHECK,
            correlation_group=f"{base}/seam",
            owner="chains", source="chain_cert",
            reduction_actions=[
                ReductionAction("rerun_seam_gate", "free", 0.5),
                ReductionAction("human_review_diff", "human", 1.0),
            ]))

    if cert.get("verdict") == PARTIAL_SALVAGED:
        planned = [e["id"] for e in cert.get("edits", [])]
        rows.append(URow(
            id=f"{base}#suffix",
            utype=UNEXPLORED_BRANCH,
            claim=f"chain salvaged {cert.get('completed_edit_ids')} of "
                  f"{planned}; the suffix was never attempted",
            severity=0.7, irreversibility=0.0, u=0.9,
            closure_criterion=DETERMINISTIC_CHECK,
            correlation_group=f"{base}/suffix",
            owner="chains", source="chain_cert",
            reduction_actions=[
                ReductionAction("replan_suffix_from_checkpoint", "cheap", 0.7),
            ]))
    return rows


# =====================================================================
# CLI (inspection only — no model calls)
# =====================================================================
def load_repo_dict(root: str, limit: int | None = None) -> dict:
    root = pathlib.Path(root)
    files = {}
    for p in sorted(root.rglob("*.py")):
        if "/." in str(p):
            continue
        try:
            files[str(p.relative_to(root))] = p.read_text()
        except Exception:
            pass
        if limit and len(files) >= limit:
            break
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 chains.py <repo_root>   # print the seam manifest")
        sys.exit(1)
    fs = load_repo_dict(sys.argv[1])
    m = seam_manifest(fs)
    v = seam_violations(m, fs)
    print(json.dumps({
        "files": len(fs), "definitions": len(m["definitions"]),
        "references": len(m["references"]),
        "cross_file_seams": len(cross_file_seams(m)),
        "violations": len(v),
        "violation_kinds": sorted({x["kind"] for x in v}),
    }, indent=2))
