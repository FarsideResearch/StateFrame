"""CHAINS_EXEC — the DETERMINISTIC EDIT EXECUTOR.

Measured defect this module exists to close (task ``twofile-01``):

    ``run_chain`` sent EVERY edit to the LLM fixer. But a rename edit whose
    evidence is an ``undefined_import`` seam violation is purely CODE-DERIVABLE
    — the old name, the new name and the file are all in the plan, and the
    violation itself proves the site. Worse, the fixer *cannot* do it: the edit
    site is a module/function-local ``import`` line that lives outside every
    function body, so the fixer's function menu can never represent it and it
    abstains. A model call was spent to produce a guaranteed ABSTAIN.

So: try code first, fall through to the model only on genuine ambiguity.

Discipline (why this is safe to run without a human or a model in the loop):

  * Sites come from TOKENIZE, not from a regex over the source. A NAME token is
    a NAME token — a mention inside a comment, a docstring or a string literal
    is a COMMENT / STRING / FSTRING_MIDDLE token and is never touched. A rename
    repair deliberately leaves prose alone.
  * Every tokenize site is CROSS-CERTIFIED against the AST: if the two
    disagree on how many reference sites exist, we abstain rather than guess.
  * Anything the AST cannot resolve unambiguously — a shadowing binding, a
    parameter of the same name, an attribute endpoint, a keyword-argument
    label, an ``import x as y`` alias, a collision with ``new_name`` — returns
    None and the chain falls through to the fixer exactly as before.
  * The result is RE-PARSED (syntactic validity) and then re-checked against
    ``seam_violations``: the targeted violation must be GONE and no new
    violation may appear. A rename that does not clear its own evidence is not
    a repair, so it is discarded.

Fail-closed everywhere: ``None`` means "the model's turn", never "good enough".
"""
from __future__ import annotations

import ast
import io
import tokenize

# violation families whose evidence a rename is allowed to act on
RENAME_EVIDENCE_KINDS = ("undefined_import", "undefined_name")


# ---------------------------------------------------------------- AST survey
class _NameSurvey(ast.NodeVisitor):
    """Every syntactic appearance of ``name``, classified, with its scope."""

    def __init__(self, name: str):
        self.name = name
        self.scope = ["<module>"]
        self.loads = []          # ast.Name nodes (any ctx)
        self.bind_scopes = set()  # scopes that BIND the name
        self.defs = []           # FunctionDef / AsyncFunctionDef / ClassDef
        self.import_from = []    # ast.alias under ImportFrom, asname is None
        self.blockers = []       # anything that makes the rename ambiguous
        self.total = 0           # every appearance, for the tokenize cross-check

    # -- scope plumbing -------------------------------------------------
    def _sid(self):
        return "/".join(self.scope)

    def _scoped(self, node):
        self.scope.append(f"{type(node).__name__}:{getattr(node, 'name', '')}"
                          f"@{node.lineno}")
        self.generic_visit(node)
        self.scope.pop()

    # -- bindings -------------------------------------------------------
    def _visit_def(self, node):
        if node.name == self.name:
            self.defs.append(node)
            self.bind_scopes.add(self._sid())
            self.total += 1
        self._scoped(node)

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def
    visit_ClassDef = _visit_def

    def visit_Lambda(self, node):
        self._scoped(node)

    def visit_arg(self, node):
        if node.arg == self.name:
            # a parameter of the same name shadows the symbol inside its body
            self.blockers.append(f"parameter {self.name!r} at line {node.lineno}")
            self.total += 1
        self.generic_visit(node)

    def visit_Global(self, node):
        if self.name in node.names:
            self.blockers.append(f"global declaration at line {node.lineno}")
            self.total += 1
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        if self.name in node.names:
            self.blockers.append(f"nonlocal declaration at line {node.lineno}")
            self.total += 1
        self.generic_visit(node)

    # -- imports --------------------------------------------------------
    def visit_ImportFrom(self, node):
        for a in node.names:
            if a.name == self.name:
                self.total += 1
                if a.asname is not None:
                    # `from m import old as x`: renaming the import does NOT
                    # rename the local binding. Two coupled edits, not one.
                    self.blockers.append(
                        f"aliased import at line {node.lineno}")
                else:
                    self.import_from.append(a)
                    self.bind_scopes.add(self._sid())
            elif a.asname == self.name:
                self.total += 1
                self.blockers.append(f"import alias target at line {node.lineno}")
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            head = a.name.split(".")[0]
            if a.asname == self.name or (a.asname is None and head == self.name):
                self.total += 1
                self.blockers.append(f"module import at line {node.lineno}")
        self.generic_visit(node)

    # -- references -----------------------------------------------------
    def visit_Name(self, node):
        if node.id == self.name:
            self.loads.append(node)
            self.total += 1
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                self.bind_scopes.add(self._sid())
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == self.name:
            # `obj.old_name` — resolving the value chain to the renamed symbol
            # needs type inference we do not have. Hand it to the fixer.
            self.blockers.append(f"attribute access at line {node.lineno}")
            self.total += 1
        self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == self.name:
            # a keyword-argument LABEL is a parameter name, not this symbol
            self.blockers.append("keyword-argument label")
            self.total += 1
        self.generic_visit(node)


def _token_sites(src: str, name: str) -> list | None:
    """(row, col) of every NAME token equal to ``name``.

    Tokenize — not a regex — so comments, docstrings and string literals are
    structurally out of reach. Returns None if the source will not tokenize.
    """
    sites = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME and tok.string == name:
                sites.append((tok.start[0], tok.start[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return sites


def _apply(src: str, sites: list, old: str, new: str) -> str | None:
    """Replace exactly the given (row, col) spans. None if any span misreads."""
    lines = src.splitlines(keepends=True)
    by_line: dict[int, list] = {}
    for row, col in sites:
        by_line.setdefault(row, []).append(col)
    for row, cols in by_line.items():
        if row < 1 or row > len(lines):
            return None
        line = lines[row - 1]
        for col in sorted(cols, reverse=True):          # right-to-left
            if line[col:col + len(old)] != old:
                return None
            line = line[:col] + new + line[col + len(old):]
        lines[row - 1] = line
    return "".join(lines)


def mechanical_rename(files: dict, edit, violations: list,
                      *, info: dict | None = None):
    """Deterministically rename ``edit.old_name`` -> ``edit.new_name`` in
    ``edit.file``. Returns the NEW files dict, or None to defer to the fixer.

    Applicable when the edit carries both names AND either
      (a) a matching undefined_import/undefined_name violation in ``edit.file``
          references ``old_name``, or
      (b) ``old_name`` resolves unambiguously via AST in ``edit.file``.

    Post-condition (enforced, not assumed): the updated tree re-parses, the
    targeted violation is gone, and no new seam violation appeared.
    """
    from chains import seam_manifest, seam_violations       # local: no cycle

    def _bail(why: str):
        if info is not None:
            info["reason"] = why
        return None

    old = getattr(edit, "old_name", "") or ""
    new = getattr(edit, "new_name", "") or ""
    path = getattr(edit, "file", "") or ""
    if not old or not new or old == new:
        return _bail("edit does not declare a distinct old_name/new_name")
    if not old.isidentifier() or not new.isidentifier():
        return _bail("names are not plain identifiers")
    if path not in files:
        return _bail(f"{path!r} is not in the repo")
    allowed = tuple(getattr(edit, "allowed_files", ()) or (path,))
    if path not in allowed:
        return _bail(f"{path!r} is outside the edit's allowed_files")

    src = files[path]
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return _bail(f"{path} does not parse: {exc}")

    survey = _NameSurvey(old)
    survey.visit(tree)
    if survey.blockers:
        return _bail("ambiguous: " + "; ".join(sorted(set(survey.blockers))[:3]))
    n_ast = len(survey.loads) + len(survey.defs) + len(survey.import_from)
    if n_ast == 0:
        return _bail(f"{old!r} has no reference site in {path}")
    if len(survey.bind_scopes) > 1:
        return _bail(f"{old!r} is bound in {len(survey.bind_scopes)} scopes "
                     "(shadowed)")

    # the destination name must not already mean something else here
    collide = _NameSurvey(new)
    collide.visit(tree)
    if collide.total:
        return _bail(f"{new!r} already appears in {path}")

    # evidence: (a) a violation that names this symbol, or (b) clean AST resolve
    target = next((v for v in (violations or [])
                   if v.get("kind") in RENAME_EVIDENCE_KINDS
                   and v.get("file") == path and v.get("name") == old), None)

    sites = _token_sites(src, old)
    if sites is None:
        return _bail(f"{path} does not tokenize")
    if len(sites) != n_ast:
        # tokenize and the AST disagree on how many real sites exist — one of
        # them is seeing something the other is not. Never guess.
        return _bail(f"token/AST site disagreement ({len(sites)} vs {n_ast})")

    updated = _apply(src, sites, old, new)
    if updated is None:
        return _bail("a replacement span did not read back as the old name")
    try:
        ast.parse(updated)                    # syntactic validity, proven
    except SyntaxError as exc:
        return _bail(f"rename produced invalid syntax: {exc}")

    new_files = dict(files)
    new_files[path] = updated

    # ---- POST-CONDITION -------------------------------------------------
    before_keys = {(v["kind"], v["file"], v["name"]) for v in (violations or [])}
    after = seam_violations(seam_manifest(new_files), new_files)
    after_keys = {(v["kind"], v["file"], v["name"]) for v in after}
    if target is not None:
        tkey = (target["kind"], target["file"], target["name"])
        if tkey in after_keys:
            return _bail("rename did not clear the targeted violation")
    introduced = sorted(after_keys - before_keys)
    if introduced:
        return _bail(f"rename introduced {len(introduced)} new violation(s): "
                     f"{introduced[:2]}")

    if info is not None:
        info.update(sites=len(sites), file=path, old_name=old, new_name=new,
                    violation_cleared=target is not None,
                    evidence="violation" if target is not None else "ast")
    return new_files


def mechanical_cert(info: dict) -> dict:
    """The fixer_cert-equivalent for an edit the code executed by itself."""
    return {
        "executor": "mechanical_rename",
        "sites": info.get("sites", 0),
        "violation_cleared": bool(info.get("violation_cleared")),
        "evidence": info.get("evidence", ""),
        "model_calls": 0,
        "stages": [{"edit_by": "code"}],
    }
