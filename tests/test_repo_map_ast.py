"""The tree-sitter repo map and source outline: the AST path is really taken when the audited grammars are
installed, and the text-scan / lexical fallback gives its documented output when they are not.

Each probe runs in a fresh interpreter because both modules pick their parser once, at import time:
``repo_map`` binds ``tree_sitter_language_pack.get_parser`` on import, and the engine only installs that
module (``ts_grammars.install_as_language_pack``) at child start. A fresh process per probe keeps the two
modes from leaking into each other or into the rest of the suite.

The AST tests need the repomap extra (``requirements-repomap.txt``: tree-sitter + per-language grammars) and
skip without it. The fallback tests force the fallback and run everywhere.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

AST_MODULES = ("tree_sitter", "tree_sitter_python", "tree_sitter_typescript", "tree_sitter_go")
_MISSING = [name for name in AST_MODULES if importlib.util.find_spec(name) is None]
needs_ast = pytest.mark.skipif(
    bool(_MISSING),
    reason="repo map AST needs the repomap extra (requirements-repomap.txt); missing: " + ", ".join(_MISSING))

FIXTURE = {
    "pkg/app.py": '''\
        import os


        class Greeter:
            def greet(self, name: str) -> str:
                return f"hi {name}"


        async def fetch(url, *, timeout=3) -> bytes:
            return b""


        def helper(a, b=2):
            return a + b
        ''',
    "web/client.ts": '''\
        export interface User { id: number }

        export function load(id: number): User {
          return { id };
        }

        class Store {
          get(key: string): string { return key; }
        }

        type Key = string;
        ''',
    "svc/main.go": '''\
        package main

        type Server struct {
        \tport int
        }

        func (s *Server) Start() error {
        \treturn nil
        }

        func main() {}
        ''',
}

# Runs in a fresh interpreter: argv = mode, src root, fixture root. "ast" installs the audited grammar
# adapter exactly as the engine child does; "fallback" blocks both grammar bundles first.
PROBE = textwrap.dedent('''\
    import json, sys
    mode, src, fixture = sys.argv[1:4]
    sys.path[:0] = [src, src + "/stateframe/engine"]
    if mode == "ast":
        from stateframe import ts_grammars
        installed = ts_grammars.install_as_language_pack()
    else:
        sys.modules["tree_sitter_language_pack"] = None
        sys.modules["tree_sitter_languages"] = None
        installed = False
    from mc_repo_intel_vendor import repo_map, ts_dispatch
    outlines = {}
    for rel in sorted(json.loads(sys.argv[4])):
        outline = ts_dispatch.outline_source(open(fixture + "/" + rel, encoding="utf-8").read(), rel)
        outlines[rel] = {
            "parser": outline["parser"], "language": outline["language"],
            "degraded": outline["degraded"], "completeness": outline["completeness"],
            "capability": outline["capability"],
            "functions": sorted([f["name"], f["kind"]] for f in outline["functions"]),
            "types": sorted([t["name"], t["kind"]] for t in outline["types"]),
        }
    print(json.dumps({"installed": installed, "available": repo_map._TREE_SITTER_AVAILABLE,
                      "map": repo_map._generate_repo_map(fixture), "outlines": outlines}, sort_keys=True))
    ''')


@pytest.fixture
def fixture_repo(tmp_path):
    for rel, text in FIXTURE.items():
        path = tmp_path / "repo" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path / "repo"


def _probe(mode, repo, files=FIXTURE):
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, mode, str(SRC), str(repo), json.dumps(sorted(files))],
        capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


AST_MAP = """\
## Repository Map (3 files, tree-sitter)
pkg/app.py:
  class Greeter:
    def greet(self, name: str)
  async def fetch(url, *, timeout=3)
  def helper(a, b=2)
svc/main.go:
  type Server struct
  func (s *Server) Start() error
  func main()
web/client.ts:
  export interface User { id: number }
  export function load(id: number): User {
  class Store {
  type Key = string;"""

FALLBACK_MAP = """\
## Repository Map (3 files, text-scan fallback)
pkg/app.py:
  class Greeter:
  def greet(self, name: str) -> str:
  async def fetch(url, *, timeout=3) -> bytes:
  def helper(a, b=2):
svc/main.go:
  func (s *Server) Start() error
  func main()
web/client.ts:
  export interface User { id: number }
  export function load(id: number): User {
  class Store {
  type Key = string;"""

FUNCTIONS = {
    "pkg/app.py": [["fetch", "function"], ["greet", "method"], ["helper", "function"]],
    "web/client.ts": [["get", "method"], ["load", "function"]],
    "svc/main.go": [["Start", "method"], ["main", "function"]],
}


@needs_ast
def test_ast_path_is_taken_for_python_typescript_and_go(fixture_repo):
    probe = _probe("ast", fixture_repo)
    assert probe["installed"] is True and probe["available"] is True
    for rel, outline in probe["outlines"].items():
        assert (outline["parser"], outline["degraded"], outline["completeness"], outline["capability"]) == (
            "tree-sitter", False, "KNOWN", "AVAILABLE"), rel
        assert outline["functions"] == FUNCTIONS[rel], rel
    assert probe["outlines"]["pkg/app.py"]["types"] == [["Greeter", "class"]]
    assert probe["outlines"]["web/client.ts"]["types"] == [["Key", "type"], ["Store", "class"], ["User", "interface"]]
    assert probe["outlines"]["svc/main.go"]["types"] == [["Server", "struct"]]


@needs_ast
def test_ast_repo_map_renders_parser_signatures(fixture_repo):
    # The AST rendering nests methods under their class and keeps only the parameter list.
    assert _probe("ast", fixture_repo)["map"] == AST_MAP


def test_fallback_outline_is_degraded_and_documented(fixture_repo):
    probe = _probe("fallback", fixture_repo)
    assert probe["installed"] is False and probe["available"] is False
    outlines = probe["outlines"]
    # Python falls back to the stdlib ast outline, complete and not degraded.
    assert (outlines["pkg/app.py"]["parser"], outlines["pkg/app.py"]["degraded"]) == ("ast", False)
    assert outlines["pkg/app.py"]["types"] == [["Greeter", "class"]]
    # Other languages get the lexical scan, flagged as degraded and incomplete.
    for rel in ("web/client.ts", "svc/main.go"):
        assert (outlines[rel]["parser"], outlines[rel]["degraded"], outlines[rel]["completeness"],
                outlines[rel]["capability"]) == ("lexical", True, "UNKNOWN", "DEGRADED"), rel
    for rel, outline in outlines.items():
        assert outline["functions"] == FUNCTIONS[rel], rel
    # What the lexical scan misses: the TypeScript type alias and the Go struct.
    assert outlines["web/client.ts"]["types"] == [["Store", "class"], ["User", "interface"]]
    assert outlines["svc/main.go"]["types"] == []


def test_fallback_repo_map_is_the_documented_text_scan(fixture_repo):
    # Python/TS scans keep whole lines; the Go lexical outline keeps functions but misses the struct.
    assert _probe("fallback", fixture_repo)["map"] == FALLBACK_MAP


@needs_ast
def test_ast_and_fallback_differ_only_where_expected(fixture_repo):
    ast_probe, fallback = _probe("ast", fixture_repo), _probe("fallback", fixture_repo)
    for rel in FIXTURE:
        assert ast_probe["outlines"][rel]["functions"] == fallback["outlines"][rel]["functions"], rel
    extra = {rel: sorted(map(tuple, ast_probe["outlines"][rel]["types"]))
             for rel in FIXTURE}
    missing = {rel: sorted(set(extra[rel]) - set(map(tuple, fallback["outlines"][rel]["types"])))
               for rel in FIXTURE}
    assert missing == {"pkg/app.py": [], "web/client.ts": [("Key", "type")], "svc/main.go": [("Server", "struct")]}
    assert ast_probe["map"] != fallback["map"]


@pytest.mark.parametrize("mode", ["ast", "fallback"])
def test_each_mode_is_deterministic(fixture_repo, mode):
    if mode == "ast" and _MISSING:
        pytest.skip("repo map AST needs the repomap extra (requirements-repomap.txt); missing: "
                    + ", ".join(_MISSING))
    assert _probe(mode, fixture_repo) == _probe(mode, fixture_repo)


@needs_ast
def test_repo_map_lists_go_files(fixture_repo):
    assert "svc/main.go:" in _probe("ast", fixture_repo)["map"]


@needs_ast
def test_ast_repo_map_lists_other_supported_languages(fixture_repo):
    sources = {
        "code/lib.rs": "pub struct Widget {}\nimpl Widget { pub fn run(&self) {} }\n",
        "java/Main.java": "class Main { void run() {} }\n",
        "ruby/user.rb": "class User\n  def name\n    \"x\"\n  end\nend\n",
        "php/user.php": "<?php\nclass User { function name() { return \"x\"; } }\n",
        "c/main.c": "int main(void) { return 0; }\n",
        "cpp/main.cpp": "class Widget { public: void run() {} };\n",
        "cs/Program.cs": "class Program { void Run() {} }\n",
    }
    for rel, source in sources.items():
        path = fixture_repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    repo_map = _probe("ast", fixture_repo)["map"]
    for expected in ("pub struct Widget", "class Main", "class User", "function name()",
                     "int main(void)", "void run()", "void Run()"):
        assert expected in repo_map
    assert repo_map.startswith("## Repository Map (10 files, tree-sitter)\n")


BROKEN_PARSER = textwrap.dedent('''\
    import json, sys
    src, fixture = sys.argv[1:3]
    sys.path[:0] = [src, src + "/stateframe/engine"]
    from stateframe import ts_grammars
    ts_grammars.install_as_language_pack()
    original = sys.modules["tree_sitter_language_pack"].get_parser
    break_language = sys.argv[3] if len(sys.argv) > 3 else None
    def broken(name):
        if break_language is None or name == break_language:
            raise RuntimeError("grammar ABI mismatch")
        return original(name)
    sys.modules["tree_sitter_language_pack"].get_parser = broken
    from mc_repo_intel_vendor import repo_map
    print(json.dumps(repo_map._generate_repo_map(fixture)))
    ''')


@needs_ast
def test_repo_map_header_does_not_claim_tree_sitter_when_every_parse_failed(fixture_repo):
    result = subprocess.run([sys.executable, "-I", "-c", BROKEN_PARSER, str(SRC), str(fixture_repo)],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    repo_map = json.loads(result.stdout)
    assert repo_map.splitlines()[1:] == FALLBACK_MAP.splitlines()[1:]  # the entries really are text scan
    assert "tree-sitter" not in repo_map.splitlines()[0]
    assert repo_map.splitlines()[0] == "## Repository Map (3 files, text-scan fallback, 3 parse failures)"
    assert result.stderr.count("parser failures during map generation") == 1


@needs_ast
def test_repo_map_header_counts_one_failed_parser(fixture_repo):
    result = subprocess.run([sys.executable, "-I", "-c", BROKEN_PARSER, str(SRC), str(fixture_repo), "python"],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    repo_map = json.loads(result.stdout)
    assert repo_map.splitlines()[0] == "## Repository Map (3 files, tree-sitter; 1 text-scan fallback, 1 parse failures)"
    assert "  async def fetch(url, *, timeout=3) -> bytes:" in repo_map
    assert "  type Server struct" in repo_map
    assert result.stderr.count("parser failures during map generation") == 1


@needs_ast
def test_ast_repo_map_keeps_async(fixture_repo):
    assert "  async def fetch(url, *, timeout=3)" in _probe("ast", fixture_repo)["map"].splitlines()


@needs_ast
def test_mixed_map_reports_actual_fallback_count(fixture_repo):
    (fixture_repo / "empty.kt").write_text("package demo\n", encoding="utf-8")
    repo_map = _probe("ast", fixture_repo)["map"]
    assert repo_map.splitlines()[0] == "## Repository Map (4 files, tree-sitter; 1 text-scan fallback)"
    assert "empty.kt:\n  (no signatures)" in repo_map


PURE_FIXTURE = {
    "lib/__init__.py": "import os\n",  # no signatures: left out, as before
    "lib/tiny.py": "x = 1\n",  # under 10 bytes: left out, as before
    "lib/core.py": "class Core:\n    def run(self, n):\n        return n\n",
    "ui/view.js": "export function render(node) {\n  return node;\n}\n",
}


@pytest.mark.parametrize("mode", ["ast", "fallback"])
def test_pure_python_and_js_repo_map_is_unchanged(tmp_path, mode):
    # Only the newly covered languages may change the map: a py/ts/js repo keeps its exact map, header
    # included, and files without signatures or under 10 bytes stay out of it.
    if mode == "ast" and _MISSING:
        pytest.skip("repo map AST needs the repomap extra (requirements-repomap.txt); missing: "
                    + ", ".join(_MISSING))
    repo = tmp_path / "pure"
    for rel, text in PURE_FIXTURE.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    expected = {
        "ast": "## Repository Map (2 files, tree-sitter)\nlib/core.py:\n  class Core:\n    def run(self, n)\n"
               "ui/view.js:\n  export function render(node) {",
        "fallback": "## Repository Map (2 files, text-scan fallback)\nlib/core.py:\n  class Core:\n"
                    "  def run(self, n):\nui/view.js:\n  export function render(node) {",
    }[mode]
    assert _probe(mode, repo, PURE_FIXTURE)["map"] == expected


RECEIPT = textwrap.dedent('''\
    import json, sys
    src, fixture = sys.argv[1:3]
    sys.path[:0] = [src, src + "/stateframe/engine"]
    from stateframe import ts_grammars
    ts_grammars.install_as_language_pack()
    import kernel_pilot, kernel_repo_intel
    events = []
    class Journal:
        def emit(self, kind, **fields):
            events.append({"kind": kind, **fields})
    class Runtime:
        journal = Journal()
    kernel_pilot.ACTIVE_RUNTIME.set(Runtime())
    kernel_repo_intel.intake_orientation(__import__("pathlib").Path(fixture))
    print(json.dumps([e for e in events if e["kind"] == "repo_map_modes"]))
    ''')


@needs_ast
def test_orientation_journals_the_modes_actually_used(fixture_repo):
    # Orientation reads HEAD for its cache key, as in an engine workspace (always a git repository).
    for command in (["init", "-q"], ["add", "-A"],
                    ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(fixture_repo), *command], check=True, capture_output=True)
    # A private map cache, so a map cached by an earlier run cannot answer for this build.
    env = {**os.environ, "STATEFRAME_HOME": str(fixture_repo.parent / "stateframe-home")}
    result = subprocess.run([sys.executable, "-I", "-c", RECEIPT, str(SRC), str(fixture_repo)],
                            capture_output=True, text=True, timeout=120, check=False, env=env)
    assert result.returncode == 0, result.stderr
    (event,) = json.loads(result.stdout)
    assert {key: event[key] for key in ("source", "files", "tree_sitter", "text_scan", "parse_failures",
                                         "failure_reasons")} == {
        "source": "generated", "files": 3, "tree_sitter": 3, "text_scan": 0, "parse_failures": 0,
        "failure_reasons": {}}
    assert event["producer"] == "kernel_repo_intel" and event["map_chars"] > 0


STALE_CACHE = textwrap.dedent('''\
    import hashlib, json, os, sys
    src, fixture = sys.argv[1:3]
    sys.path[:0] = [src, src + "/stateframe/engine"]
    sys.modules["tree_sitter_language_pack"] = None
    sys.modules["tree_sitter_languages"] = None
    from mc_repo_intel_vendor import repo_map
    commit = "0" * 40
    # The key an older build (before the repo-map/2 rules) used for this repo at this commit.
    old_key = hashlib.sha256(f":{commit}".encode()).hexdigest()[:20]
    (repo_map._get_cache_dir() / f"{old_key}.txt").write_text("STALE MAP FROM AN OLDER BUILD", encoding="utf-8")
    text, stats = repo_map.build_repo_map_with_stats(fixture, commit_sha=commit)
    again, again_stats = repo_map.build_repo_map_with_stats(fixture, commit_sha=commit)
    print(json.dumps({"text": text, "source": stats["source"], "again_source": again_stats["source"],
                      "same": again == text, "old_key": old_key, "new_key": repo_map._make_cache_key("", commit)}))
    ''')


def test_map_cached_by_an_older_build_is_not_served(fixture_repo):
    env = {**os.environ, "STATEFRAME_HOME": str(fixture_repo.parent / "stateframe-home")}
    result = subprocess.run([sys.executable, "-I", "-c", STALE_CACHE, str(SRC), str(fixture_repo)],
                            capture_output=True, text=True, timeout=120, check=False, env=env)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)
    assert probe["old_key"] != probe["new_key"]
    assert "STALE" not in probe["text"] and probe["text"] == FALLBACK_MAP
    assert (probe["source"], probe["again_source"], probe["same"]) == ("generated", "cache", True)
