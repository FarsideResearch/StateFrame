"""Pins the slice of the CrossFileIndex surface that chains_recall (piece C) consumes.

The real index (crew-02) must satisfy the same assertions as the fake before it is
swapped in: point INDEX_FACTORY at it. Names, signatures and dataclass fields are
taken from crew-ops/plans/build_specs/crossfile_index_api.md.
"""
from __future__ import annotations

import dataclasses
import inspect

from _fake_crossfile_index import Def, Ref, index_class

FILES = {
    "pkg/__init__.py": "",
    "pkg/models.py": "class Page:\n    def __init__(self):\n        self.has_more = 1\n",
    "pkg/api.py": "from pkg.models import Page\n\ndef fetch() -> Page:\n    return Page()\n",
    "pkg/use.py": "from pkg.api import fetch\nfrom .models import Page\n\ndef go():\n    page = fetch()\n    return page.has_more\n",
}
INDEX_FACTORY = index_class()   # set SF_REAL_CROSSFILE_INDEX_DIR to run against the real index


def test_def_and_ref_fields():
    assert {"path", "qualname", "kind", "fields"} <= {f.name for f in dataclasses.fields(Def)}
    assert {"path", "lineno", "name", "kind", "receiver", "scope"} <= {f.name for f in dataclasses.fields(Ref)}


def test_method_names_and_signatures():
    for name, params in {"defs": ["path"], "refs": ["path"], "binding": ["path", "scope", "name"],
                         "import_target": ["path", "module", "level", "name"]}.items():
        sig = inspect.signature(getattr(INDEX_FACTORY, name))
        assert list(sig.parameters)[1:] == params, name
    assert defs_ok(INDEX_FACTORY.from_sources(FILES))


def defs_ok(index) -> bool:
    page = [d for d in index.defs() if d.kind == "class" and d.qualname == "Page"]
    assert len(page) == 1 and "has_more" in page[0].fields
    assert [d.qualname for d in index.defs("pkg/api.py")] == ["fetch"]
    assert index.defs("pkg/api.py")[0].returns == "pkg/models.py::Page"
    return True


def test_refs_receiver_and_binding():
    index = INDEX_FACTORY.from_sources(FILES)
    attr = [r for r in index.refs("pkg/use.py") if r.kind == "attr" and r.name.rsplit(".", 1)[-1] == "has_more"]
    assert len(attr) == 1 and attr[0].scope == "go"
    assert attr[0].receiver.split("::")[-1].split(".")[-1] == "Page"
    assert index.binding("pkg/use.py", "go", "page").endswith("Page")
    assert index.binding("pkg/use.py", "go", "nothing") is None


def test_import_target_contract():
    index = INDEX_FACTORY.from_sources(FILES)
    assert index.import_target("pkg/use.py", "models", 1, "Page") == "pkg/models.py::Page"
    assert index.import_target("pkg/use.py", "models", 1, "Gone") is None
    assert index.import_target("pkg/use.py", "nowhere", 1, "Page") is None


def test_deterministic_outputs():
    a, b = INDEX_FACTORY.from_sources(FILES), INDEX_FACTORY.from_sources(dict(reversed(list(FILES.items()))))
    assert a.revision == b.revision
    assert a.defs() == b.defs() and a.refs("pkg/use.py") == b.refs("pkg/use.py")


def test_annotation_outranks_later_unknown_reassignment():
    files = dict(FILES, **{"pkg/use.py": "from pkg.models import Page\n\ndef go(src):\n"
                           "    page: Page = src(0)\n    page = src(1)\n    return page.has_more\n"})
    assert INDEX_FACTORY.from_sources(files).binding("pkg/use.py", "go", "page").endswith("Page")
