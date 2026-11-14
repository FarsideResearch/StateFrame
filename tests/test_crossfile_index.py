"""Per-revision cross-file index: defs, imports, refs, dangling detection, incremental update."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import kernel_crossfile_index as xf  # noqa: E402
from kernel_crossfile_index import CrossFileIndex as CFI  # noqa: E402


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


# ---- audit fixtures (existence + where_used, bad/good) ------------------------------
EXIST_BAD = {"lib.py": "def compute_total(x):\n    return x\n",
             "app.py": "from lib import compute_total_v2\nimport helpers_missing\nprint(compute_total_v2(1))\n"}
EXIST_GOOD = {"lib.py": "def compute_total(x):\n    return x\ndef compute_total_v2(x):\n    return x\n",
              "helpers_missing.py": "X=1\n",
              "app.py": "from lib import compute_total_v2\nimport helpers_missing\nprint(compute_total_v2(1), helpers_missing.X)\n"}
WU_API = "def fetch_user(uid, include_deleted):\n    return {'id': uid}\n"
WU_BAD = {"pkg/__init__.py": "", "pkg/api.py": WU_API,
          "pkg/svc.py": "from pkg.api import fetch_user\n\ndef load(uid):\n    return fetch_user(uid)\n"}
WU_GOOD = {**WU_BAD, "pkg/svc.py": "from pkg.api import fetch_user\n\ndef load(uid):\n    return fetch_user(uid, False)\n"}


def test_switch_defaults_off_and_rejects_junk(monkeypatch):
    monkeypatch.delenv(xf.CROSSFILE_INDEX_ENV, raising=False)
    assert xf.enabled() is False
    monkeypatch.setenv(xf.CROSSFILE_INDEX_ENV, " ON ")
    assert xf.enabled() is True
    monkeypatch.setenv(xf.CROSSFILE_INDEX_ENV, "maybe")
    with pytest.raises(ValueError):
        xf.enabled()


def test_switch_is_registered_and_not_in_profile():
    import json
    engine = Path(xf.__file__).parent
    assert xf.CROSSFILE_INDEX_ENV in json.loads((engine / "switch-registry.json").read_text())["switches"]
    profile = engine / "profiles" / "r2hc3.json"
    if profile.exists():
        assert xf.CROSSFILE_INDEX_ENV not in profile.read_text()


def test_public_surface_contract(tmp_path):
    """Pins the shared API (names, fields, kinds). Changing it breaks other crews' code."""
    import dataclasses
    assert [f.name for f in dataclasses.fields(xf.Def)] == [
        "path", "qualname", "kind", "lineno", "end_lineno", "signature", "returns", "fields", "bases"]
    assert [f.name for f in dataclasses.fields(xf.Ref)] == [
        "path", "lineno", "col", "name", "kind", "receiver", "target", "scope"]
    assert [f.name for f in dataclasses.fields(xf.Signature)] == ["params", "varargs", "varkw"]
    assert [f.name for f in dataclasses.fields(xf.Param)] == ["name", "kind", "has_default"]
    assert [f.name for f in dataclasses.fields(xf.Resolution)] == ["kind", "target"]
    write(tmp_path, {"lib.py": "LIMIT = 1\ncount = 2\nclass K:\n    x = 1\ndef f(a, /, b=1, *c, d, **e) -> K: ...\n",
                     "app.py": "import lib\nfrom lib import f\nk = f(1, d=2)\nk.x\nk['key']\nlib.LIMIT\n"})
    idx = CFI.build(tmp_path, ["lib.py", "app.py"])
    assert isinstance(idx.revision, str) and len(idx.revision) == 64
    assert {d.kind for d in idx.defs()} == {"def", "class", "const", "var"}
    assert idx.defs("lib.py") == tuple(d for d in idx.defs() if d.path == "lib.py")
    f = next(d for d in idx.defs("lib.py") if d.qualname == "f")
    assert f.returns == "lib.py::K" and f.signature.varargs == "c" and f.signature.varkw == "e"
    assert [(p.name, p.kind, p.has_default) for p in f.signature.params] == [
        ("a", "posonly", False), ("b", "pos", True), ("d", "kwonly", False)]
    assert {r.kind for r in idx.refs("app.py")} == {"call", "attr", "key", "import", "name"}
    assert all(isinstance(r, xf.Ref) for r in idx.refs("app.py"))
    assert isinstance(idx.resolve("app.py", "f", 3), xf.Resolution)
    assert idx.import_target("app.py", "lib", 0, "f") == "lib.py::f"
    assert idx.import_target("app.py", "lib", 0, None) is None  # only "path::name" or None
    assert idx.binding("app.py", "<module>", "k") == "lib.py::K"
    assert [r.path for r in idx.users("lib.py::f")] == ["app.py", "app.py"]  # import + call
    assert isinstance(idx.update("app.py", b"x = 1\n"), CFI)


def test_defs_signatures_fields_and_lowercase_vars(tmp_path):
    write(tmp_path, {"m.py": (
        "from dataclasses import dataclass\nfrom pydantic import BaseModel\n"
        "limit = 3\nNAME: str = 'x'\na, b = 1, 2\n"
        "def f(p, /, q, r=1, *args, k, z=2, **kw): ...\n"
        "async def g(): ...\n"
        "class C:\n    attr = 1\n    def __init__(self, x, y=0):\n        self.x = x\n        self.later = y\n"
        "    def meth(self, a): ...\n"
        "@dataclass\nclass D:\n    name: str\n    size: int = 0\n"
        "class P(BaseModel):\n    title: str\n    tags: list = []\n"
        "if limit:\n    def cond(a): ...\n")})
    idx = CFI.build(tmp_path, ["m.py"])
    d = {x.qualname: x for x in idx.defs("m.py")}
    assert (d["limit"].kind, d["NAME"].kind, d["a"].kind) == ("var", "const", "var")
    assert d["C"].kind == "class" and d["C.meth"].kind == "def" and d["C.attr"].kind == "var"
    assert d["g"].kind == d["cond"].kind == "def"
    assert d["C"].fields == ("attr", "later", "x")            # class attr + self.x in methods
    assert d["D"].fields == ("name", "size")                  # dataclass
    assert d["P"].fields == ("tags", "title")                 # pydantic
    assert d["f"].end_lineno >= d["f"].lineno
    sig = d["f"].signature
    assert [(p.name, p.kind, p.has_default) for p in sig.params] == [
        ("p", "posonly", False), ("q", "pos", False), ("r", "pos", True), ("k", "kwonly", False), ("z", "kwonly", True)]
    assert (sig.varargs, sig.varkw) == ("args", "kw")
    assert [p.name for p in d["C"].signature.params] == ["y", "x"][::-1]  # __init__ minus self


def test_aliased_relative_and_reexport_resolution(tmp_path):
    write(tmp_path, {
        "pkg/__init__.py": "from .core import run as go\nfrom . import util\n",
        "pkg/core.py": "def run(a, b=1): ...\nlimit = 5\n",
        "pkg/util.py": "def helper(): ...\n",
        "pkg/sub/__init__.py": "",
        "pkg/sub/deep.py": "from ..core import run\nfrom .. import util\nrun(1)\nutil.helper()\n",
        "app.py": ("import pkg\nimport pkg.core as pc\nfrom pkg import go as g, util as u\n"
                   "from pkg.core import limit as lim\n"
                   "g(1)\npc.run(1, b=2)\nu.helper()\npkg.util.helper()\nprint(lim)\n"),
    })
    idx = CFI.build(tmp_path)
    r = lambda p, n, ln=99: idx.resolve(p, n, ln)  # noqa: E731
    assert r("app.py", "g") == xf.Resolution("import", "pkg/core.py::run")         # re-export + alias
    assert r("app.py", "pc.run").target == "pkg/core.py::run"
    assert r("app.py", "u.helper").target == "pkg/util.py::helper"
    assert r("app.py", "pkg.util.helper").target == "pkg/util.py::helper"
    assert r("app.py", "lim").target == "pkg/core.py::limit"
    assert r("pkg/sub/deep.py", "run").target == "pkg/core.py::run"
    assert r("pkg/sub/deep.py", "util.helper").target == "pkg/util.py::helper"
    assert r("app.py", "len").kind == "builtin" and r("app.py", "nope").kind == "unknown"
    assert idx.import_target("pkg/sub/deep.py", "core", 2, "run") == "pkg/core.py::run"
    assert idx.import_target("pkg/sub/deep.py", "", 2, "util") is None  # a module, not a def
    assert idx.import_target("app.py", "pkg", 0, "nothing") is None
    assert idx.dangling_refs() == []
    assert [(c.path, c.nargs, c.kwnames) for c in idx.call_sites("pkg/core.py::run")] == [
        ("app.py", 1, ()), ("app.py", 1, ("b",)), ("pkg/sub/deep.py", 1, ())]


def test_resolve_is_scope_aware_param_and_local_shadow_module_defs(tmp_path):
    write(tmp_path, {"lib.py": "def producer(): ...\n",
                     "m.py": ("from lib import producer\n"
                              "def helper(): ...\n"
                              "def uses(helper, n):\n"           # 3
                              "    producer = 1\n"               # 4
                              "    return helper(n) + producer\n"  # 5
                              "def plain():\n"                   # 6
                              "    return helper() + producer()\n")})  # 7
    idx = CFI.build(tmp_path)
    assert idx.resolve("m.py", "helper", 5) == xf.Resolution("param")
    assert idx.resolve("m.py", "producer", 5) == xf.Resolution("local")
    assert idx.resolve("m.py", "helper", 7) == xf.Resolution("module", "m.py::helper")
    assert idx.resolve("m.py", "producer", 7) == xf.Resolution("import", "lib.py::producer")
    # shadowed uses are not consumers; the real ones are
    assert [(r.path, r.lineno) for r in idx.users("m.py::helper")] == [("m.py", 7)]
    assert [(r.path, r.lineno) for r in idx.users("lib.py::producer") if r.kind != "import"] == [("m.py", 7)]


def test_binding_via_return_annotation_across_files_and_construction(tmp_path):
    write(tmp_path, {
        "models.py": "class Page:\n    has_more: bool = False\n    def next(self): ...\n",
        "api.py": "from models import Page\ndef fetch(n) -> 'Page | None': ...\ndef make() -> Page: ...\n",
        "use.py": ("from api import fetch, make\nfrom models import Page as P\n"
                   "def go(p: P, other):\n"                       # 3
                   "    page = fetch(1)\n"
                   "    made = P()\n"
                   "    typed: P = other\n"
                   "    plain = other\n"
                   "    return page.has_more, made.next(), p.has_more, typed.has_more, plain.x\n"),
    })
    idx = CFI.build(tmp_path)
    for name in ("page", "made", "typed", "p"):
        assert idx.binding("use.py", "go", name) == "models.py::Page", name
    assert idx.binding("use.py", "go", "plain") is None
    assert idx.binding("use.py", "go", "other") is None
    assert idx.binding("use.py", "go", "missing") is None
    refs = {(r.name, r.receiver, r.target) for r in idx.refs("use.py") if r.kind in ("attr", "call") and r.scope == "go"}
    assert ("page.has_more", "models.py::Page", "models.py::Page.has_more") in refs  # class field is a def
    assert ("made.next", "models.py::Page", "models.py::Page.next") in refs
    assert ("plain.x", None, None) in refs
    assert next(d for d in idx.defs("api.py") if d.qualname == "fetch").returns == "models.py::Page"


def test_self_binds_to_enclosing_class(tmp_path):
    write(tmp_path, {"c.py": "class A:\n    def m(self): ...\n    def n(self):\n        return self.m()\n"})
    idx = CFI.build(tmp_path)
    assert idx.binding("c.py", "A.n", "self") == "c.py::A"
    assert [r.target for r in idx.refs("c.py") if r.name == "self.m"] == ["c.py::A.m"]
    assert [r.scope for r in idx.users("c.py::A.m")] == ["A.n"]


def test_key_refs_and_call_shape(tmp_path):
    write(tmp_path, {"lib.py": "class R: ...\ndef f(*a, **k): ...\n",
                     "app.py": ("from lib import R, f\n"
                                "def go(x: R):\n    f(1, 2, x=3)\n    f(*[1], **{})\n"
                                "    return x['id'], x.get('name')\n")})
    idx = CFI.build(tmp_path)
    keys = [(r.name, r.receiver) for r in idx.refs("app.py") if r.kind == "key"]
    assert keys == [("id", "lib.py::R"), ("name", "lib.py::R")]
    sites = idx.call_sites("lib.py::f")
    assert [(c.lineno, c.nargs, c.kwnames, c.has_star) for c in sites] == [(3, 2, ("x",), False), (4, 0, (), True)]


def test_dangling_flags_missing_symbol_and_module_quiet_when_consistent(tmp_path):
    bad = write(tmp_path / "bad", EXIST_BAD)
    good = write(tmp_path / "good", EXIST_GOOD)
    got = CFI.build(bad).dangling_refs()
    assert ("app.py", 1, "lib.compute_total_v2", "name_not_defined") in got
    assert CFI.build(good).dangling_refs() == []


def test_dangling_bad_import_variants(tmp_path):
    write(tmp_path, {
        "pkg/__init__.py": "", "pkg/api.py": "def ok(): ...\n",
        "app.py": ("import pkg.nope\nfrom pkg import nothing\nfrom .rel import x\nimport pkg.api as a\n"
                   "import os, json\nimport requests\n"
                   "try:\n    import pkg.optional\nexcept ImportError:\n    pass\n"
                   "a.ok()\na.gone()\n"),
    })
    got = {(line, name, why) for _f, line, name, why in CFI.build(tmp_path).dangling_refs()}
    assert (1, "pkg.nope", "module_not_found") in got
    assert (2, "pkg.nothing", "name_not_defined") in got
    assert (3, ".rel.x", "module_not_found") in got
    assert (12, "a.gone", "attribute_not_defined") in got
    assert not any(line in (5, 6, 8) for line, _n, _w in got)  # stdlib, third party, optional import


def test_where_used_signature_change_surfaces_caller(tmp_path):
    bad = CFI.build(write(tmp_path / "bad", WU_BAD))
    good = CFI.build(write(tmp_path / "good", WU_GOOD))
    fn = "pkg/api.py::fetch_user"
    assert [p.name for p in bad.signature(fn).params] == ["uid", "include_deleted"]
    assert [(c.path, c.nargs) for c in bad.call_sites(fn)] == [("pkg/svc.py", 1)]   # 1 arg < 2 required
    assert [(c.path, c.nargs) for c in good.call_sites(fn)] == [("pkg/svc.py", 2)]
    assert bad.dangling_refs() == good.dangling_refs() == []


def test_update_is_immutable_and_reparses_only_changed_file(tmp_path):
    write(tmp_path, WU_GOOD)
    one = CFI.build(tmp_path)
    base = one.parses
    assert one.dangling_refs() == []
    same = one.update("pkg/api.py", (tmp_path / "pkg/api.py").read_bytes())
    assert same.parses == base and same.revision == one.revision        # unchanged bytes reuse cache
    two = one.update("pkg/api.py", b"def fetch_account(uid, include_deleted):\n    return 1\n")
    assert two.parses == base + 1 and two.revision != one.revision
    assert one.dangling_refs() == []                                    # old revision untouched
    assert two.dangling_refs(["pkg/svc.py"]) == [("pkg/svc.py", 1, "pkg.api.fetch_user", "name_not_defined")]
    three = two.update("pkg/svc.py", None)
    assert "pkg/svc.py" not in three.files and "pkg/svc.py" in two.files


def test_cache_keyed_by_content_hash_across_paths(tmp_path):
    write(tmp_path, {"a.py": "def f(): ...\n", "b.py": "def f(): ...\n"})
    idx = CFI.build(tmp_path)
    assert idx.parses == 1  # identical bytes parsed once
    assert [d.path for d in idx.defs()] == ["a.py", "b.py"]


def test_parse_errors_are_entries_never_raises(tmp_path):
    write(tmp_path, {"ok.py": "x = 1\n", "bad.py": "def (:\n", "user.py": "import bad\nbad.thing\nfrom bad import q\n"})
    (tmp_path / "bin.py").write_bytes(b"x = '\xff\xfe'\n")
    idx = CFI.build(tmp_path)
    assert idx.parse_errors() == (xf.ParseError("bad.py", "parse_error"), xf.ParseError("bin.py", "not_utf8"))
    assert idx.dangling_refs() == []          # contents of an unparseable module are unknown, not absent
    assert idx.refs("bad.py") == () and idx.defs("nope.py") == ()
    assert idx.resolve("missing.py", "x", 1) == xf.Resolution("unknown")
    assert idx.binding("missing.py", "<module>", "x") is None
    assert idx.update("ok.py", b"def (:").parse_errors()[0].path == "bad.py"


def test_bounded_oversize_defs_only_hard_cap_and_max_files(tmp_path):
    write(tmp_path, {"big.py": "from lib import f\ndef g(): ...\nf()\n" + "#" * 200,
                     "lib.py": "def f(): ...\n", "huge.py": "x = 1\n" + "#" * 500,
                     "a.py": "", "b.py": ""})
    idx = CFI.build(tmp_path, ["big.py", "lib.py"], max_file_bytes=100)
    assert idx.defs_only_files() == ("big.py",)
    assert [d.qualname for d in idx.defs("big.py")] == ["g"] and idx.refs("big.py")[0].kind == "import"
    capped = CFI.build(tmp_path, sorted(["a.py", "b.py", "lib.py"]), max_files=2)
    assert capped.parse_errors() == (xf.ParseError("lib.py", "max_files"),)
    assert capped.update("lib.py", b"def f(): ...\n").defs("lib.py")[0].qualname == "f"


def test_vendored_dirs_skipped(tmp_path):
    write(tmp_path, {"a.py": "x=1\n", "node_modules/b.py": "x=1\n", ".git/c.py": "x=1\n"})
    assert CFI.build(tmp_path).files == ("a.py",)


def test_star_import_and_module_getattr_stay_quiet(tmp_path):
    write(tmp_path, {"star.py": "from lib import *\n", "lib.py": "def real(): ...\n",
                     "dyn.py": "def __getattr__(name): ...\n",
                     "app.py": "import star\nimport dyn\nstar.real()\nstar.unknown()\ndyn.anything()\n"})
    idx = CFI.build(tmp_path)
    assert idx.dangling_refs() == []
    assert idx.resolve("app.py", "star.real", 3).target == "lib.py::real"


def test_import_cycles_terminate(tmp_path):
    write(tmp_path, {"a.py": "from b import x\n", "b.py": "from a import x\n", "c.py": "from a import x\n"})
    CFI.build(tmp_path).dangling_refs()  # must return, not recurse forever


def test_output_is_deterministic(tmp_path):
    write(tmp_path, {**EXIST_BAD, **{f"m{i}.py": f"from lib import nope{i}\n" for i in range(5)}})
    one, two = CFI.build(tmp_path), CFI.build(tmp_path)
    assert one.revision == two.revision and one.files == two.files
    assert one.dangling_refs() == two.dangling_refs() == sorted(one.dangling_refs())
    assert one.defs() == two.defs() and one.refs("app.py") == two.refs("app.py")


def test_receiver_is_path_class_and_attr_refs_are_loads_only(tmp_path):
    write(tmp_path, {"m.py": "class K:\n    v = 1\n",
                     "u.py": ("from m import K\n"
                              "def go():\n    k = K()\n    k.v = 2\n    del k.v\n    return k.v\n")})
    idx = CFI.build(tmp_path)
    attrs = [(r.lineno, r.name, r.receiver) for r in idx.refs("u.py") if r.kind == "attr"]
    assert attrs == [(6, "k.v", "m.py::K")]        # the store (4) and del (5) are not refs


def test_bases_resolved_to_qualnames(tmp_path):
    write(tmp_path, {"base.py": "class B: ...\nclass M: ...\n",
                     "c.py": "import base\nfrom base import B as Bee\nimport abc\n"
                             "class C(Bee, base.M, abc.ABC, Missing): ...\nclass Plain: ...\n"})
    d = {x.qualname: x for x in CFI.build(tmp_path).defs("c.py")}
    assert d["C"].bases == ("base.py::B", "base.py::M") and d["Plain"].bases == ()
    assert next(x for x in CFI.build(tmp_path).defs("base.py") if x.qualname == "B").bases == ()


def test_explicit_annotation_is_authoritative_in_binding(tmp_path):
    write(tmp_path, {"m.py": "class Page: ...\nclass Other: ...\ndef f() -> Other: ...\ndef g() -> Other: ...\n",
                     "u.py": ("from m import Page, f, g\n"
                              "def a():\n    page: Page = f()\n    page = g()\n    return page\n"
                              "def b():\n    page = g()\n    page: Page = f()\n    return page\n")})
    idx = CFI.build(tmp_path)
    assert idx.binding("u.py", "a", "page") == "m.py::Page"
    assert idx.binding("u.py", "b", "page") == "m.py::Page"


def test_from_sources_matches_build(tmp_path):
    files = {**WU_GOOD, "extra.py": "x = 1\n", "notes.txt": "ignored"}
    write(tmp_path, files)
    disk = CFI.build(tmp_path)
    mem = CFI.from_sources({**{k: v for k, v in files.items()}, "bytes.py": b"y = 2\n"})
    assert mem.defs("pkg/api.py") == disk.defs("pkg/api.py")
    assert mem.refs("pkg/svc.py") == disk.refs("pkg/svc.py")
    assert CFI.from_sources(files).revision == CFI.build(tmp_path, [p for p in files if p.endswith(".py")]).revision
    assert "bytes.py" in mem.files and "notes.txt" not in mem.files
    assert CFI.from_sources({"b.py": b"\xff\xfe", "a.py": "def ("}).parse_errors() == (
        xf.ParseError("a.py", "parse_error"), xf.ParseError("b.py", "not_utf8"))
