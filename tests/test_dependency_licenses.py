"""Shipped code may import only the standard library, StateFrame's own modules, or a
third-party package declared in THIRD_PARTY_LICENSES.md (permissive licenses only)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "stateframe"
ENGINE = PACKAGE / "engine"


# Import names whose distribution name differs, and module names StateFrame provides itself.
IMPORT_ALIASES = {"yaml": "pyyaml"}
# ts_grammars.install_as_language_pack() registers this module at run time from the per-language
# grammar packages, so it is StateFrame code, not a dependency.
RUNTIME_SHIMS = {"tree_sitter_language_pack"}
# Reviewed 2026-10-30, present in the 0.1.0 release: lab-only modules absent from release builds
# (their importers are guarded or unreachable in a release) and one optional fallback parser that
# StateFrame never installs. Nothing here is shipped or installed. Add to this set only after the
# same review; a new third-party dependency belongs in THIRD_PARTY_LICENSES.md instead.
REVIEWED_ABSENT = {"kernel_rejected_loop_breaker", "authored_check_adapters", "scripts",
                   "capability_study_worker", "checkpoint", "restore", "tree_sitter_languages"}


def _own_names() -> set[str]:
    # The engine runs with several of its own directories on sys.path, so any module or package
    # name defined anywhere in the shipped tree is StateFrame's own.
    names = {"stateframe"} | RUNTIME_SHIMS
    for path in PACKAGE.rglob("*"):
        if path.suffix == ".py":
            names.add(path.stem)
        elif path.is_dir() and not path.name.startswith((".", "__")):
            names.add(path.name)
    return names


def _declared_third_party() -> set[str]:
    text = "\n".join(p.read_text() for p in (ROOT / "THIRD_PARTY_LICENSES.md",
                                             PACKAGE / "THIRD_PARTY_LICENSES.md") if p.exists())
    # Distribution names appear as headings or table cells; import names use underscores.
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9_.-]+", text))
    return {w.lower().replace("-", "_").split(".")[0] for w in words}


def _foreign_imports() -> dict[str, set[str]]:
    own, foreign = _own_names(), {}
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top not in sys.stdlib_module_names and top not in own:
                    foreign.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return foreign


def test_every_third_party_import_is_declared_with_its_license():
    declared = _declared_third_party()
    undeclared = {name: sorted(files) for name, files in _foreign_imports().items()
                  if name not in REVIEWED_ABSENT
                  and IMPORT_ALIASES.get(name, name).lower() not in declared}
    assert not undeclared, f"undeclared third-party imports (add to THIRD_PARTY_LICENSES.md or remove): {undeclared}"


def test_guard_detects_a_planted_undeclared_import(tmp_path, monkeypatch):
    # Non-vacuous: a planted import of an unknown package must be reported.
    planted = PACKAGE / "_license_guard_probe.py"
    planted.write_text("import definitely_not_a_declared_package\n")
    try:
        assert "definitely_not_a_declared_package" in _foreign_imports()
    finally:
        planted.unlink()
