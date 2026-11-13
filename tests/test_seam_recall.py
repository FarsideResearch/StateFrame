"""STATEFRAME_SEAM_RECALL: seam_violations recall through a CrossFileIndex (fake here)."""
from __future__ import annotations

import json

import pytest

import chains
import chains_recall
from _fake_crossfile_index import index_class

MODELS = ("class Page:\n    def __init__(self, items, has_more):\n        self.items = items\n"
          "        self.has_more = has_more\n\nclass Unrelated:\n    def name(self):\n        return 'x'\n")
API = "from pkg.models import Page\n\ndef fetch(off) -> Page:\n    return Page([], False)\n"
CLIENT = ("from pkg.api import fetch\n\ndef all_items():\n    out = []\n    page = fetch(0)\n"
          "    while True:\n        out += page.items\n        if not page.has_more:\n            break\n"
          "        page = fetch(len(out))\n    return out\n")
GOOD = {
    "pkg/__init__.py": "",
    "pkg/models.py": MODELS,
    "pkg/api.py": API,
    "pkg/client.py": CLIENT,
    "pkg/rel/__init__.py": "",
    "pkg/rel/a.py": "from .b import helper\n\ndef run():\n    return helper()\n",
    "pkg/rel/b.py": "def helper():\n    return 1\n",
}


def violations(files, *, index=True, baseline=None):
    manifest = chains.seam_manifest(files)
    idx = index_class().from_sources(files) if index else None
    return chains.seam_violations(manifest, files, index=idx, baseline=baseline)


def variant(**edits):
    files = dict(GOOD)
    for path, (old, new) in edits.items():
        key = path.replace("__", "/") + ".py"
        files[key] = files[key].replace(old, new)
    return files


@pytest.fixture
def recall_on(monkeypatch):
    monkeypatch.setenv(chains_recall.MODE_ENV, "on")


def kinds(vs):
    return [(v["kind"], v["file"], v["name"]) for v in vs]


def test_good_fixture_has_no_violations(recall_on):
    assert violations(GOOD) == []


def test_producer_only_rename_through_untyped_local(recall_on):
    files = variant(pkg__models=("self.has_more = has_more", "self.more = has_more"))
    assert violations(files, index=False) == []            # v0.5.9 misses it
    got = violations(files)
    assert ("unknown_attribute", "pkg/client.py", "has_more") in kinds(got)


def test_consumer_typo_through_untyped_local(recall_on):
    files = variant(pkg__client=("page.has_more", "page.has_mroe"))
    assert violations(files, index=False) == []
    assert ("unknown_attribute", "pkg/client.py", "has_mroe") in kinds(violations(files))


def test_annotated_consumer_reassigned_from_call(recall_on):
    """`page: Page = src(0)` then `page = src(n)`: v0.5.9 binds `page` to {Page, src},
    calls that ambiguous and drops it, so the renamed attribute is never checked."""
    client = ("from pkg.models import Page\n\ndef fetch_all(src):\n    out = []\n    page: Page = src(0)\n"
              "    while True:\n        out += page.items\n        if not page.has_more:\n            break\n"
              "        page = src(len(out))\n    return out\n")
    files = variant(pkg__models=("self.has_more = has_more", "self.more = has_more"))
    files["pkg/client.py"] = client
    assert violations(files, index=False) == []
    assert ("unknown_attribute", "pkg/client.py", "has_more") in kinds(violations(files))
    # the same consumer against an intact producer stays clean
    clean = dict(GOOD, **{"pkg/client.py": client})
    assert violations(clean) == []


def test_relative_import_of_missing_name(recall_on):
    files = variant(pkg__rel__b=("def helper", "def helper2"))
    assert violations(files, index=False) == []
    got = violations(files)
    assert ("undefined_import", "pkg/rel/a.py", "helper") in kinds(got)


def test_relative_submodule_and_reexport_not_flagged(recall_on):
    files = dict(GOOD)
    files["pkg/rel/c.py"] = "from . import b\nfrom .reexp import helper\n"
    files["pkg/rel/reexp.py"] = "from .b import helper\n"
    assert violations(files) == []


def test_unbound_receiver_is_info_only(recall_on):
    files = variant(pkg__models=("self.has_more = has_more", "self.more = has_more"),
                    pkg__client=("page = fetch(0)", "page = src(0)"))
    files["pkg/client.py"] = files["pkg/client.py"].replace("def all_items():", "def all_items(src):")
    base = chains.seam_manifest(GOOD)
    got = violations(files, baseline=base)
    info = [v for v in got if v["kind"] == "unverified_attribute"]
    assert info and all(v["severity"] == "info" and v["name"] == "has_more" for v in info)
    assert not [v for v in got if v["kind"] == "unknown_attribute"]    # never fail-closed
    # without a baseline only a near-miss of a seam field is reported
    assert not [v for v in violations(files) if v["kind"] == "unverified_attribute"]
    typo = variant(pkg__client=("page = fetch(0)", "page = src(0)"))
    typo["pkg/client.py"] = typo["pkg/client.py"].replace("def all_items():", "def all_items(src):") \
        .replace("page.has_more", "page.has_mroe")
    typo["pkg/other.py"] = "from pkg.models import Page\n\ndef f(p: Page):\n    return p.has_more\n"
    assert [v["name"] for v in violations(typo) if v["kind"] == "unverified_attribute"] == ["has_mroe"]


def test_unknown_base_fails_open(recall_on):
    files = variant(pkg__models=("class Page:", "class Page(Base):"))
    files["pkg/models.py"] = "from vendor import Base\n" + files["pkg/models.py"].replace(
        "self.has_more = has_more", "self.more = has_more")
    assert violations(files) == []


def test_output_deterministic_and_sorted(recall_on):
    files = variant(pkg__models=("self.has_more = has_more", "self.more = has_more"),
                    pkg__rel__b=("def helper", "def helper2"))
    a, b = violations(files), violations(files)
    assert a == b and json.dumps(a) == json.dumps(b)
    keys = [(v["kind"], v["file"], v["lineno"], v["name"]) for v in a]
    assert keys == sorted(keys)


def test_broken_index_fails_open(recall_on):
    class Broken:
        def defs(self, path=None):
            raise RuntimeError("boom")
    files = variant(pkg__models=("self.has_more = has_more", "self.more = has_more"))
    assert chains.seam_violations(chains.seam_manifest(files), files, index=Broken()) == []


# ---- switch off == v0.5.9 (golden outputs captured from the 2045fde6 function)
def _audit_cases():
    base = {
        "pkg/__init__.py": "", "pkg/models.py": MODELS,
        "pkg/client.py": "from pkg.models import Page\n\ndef fetch_all(src):\n    out=[]\n    page=src(0)\n"
                         "    while True:\n        out+=page.items\n        if not page.has_more: break\n"
                         "        page=src(len(out))\n    return out\n",
        "pkg/consumer.py": "from pkg.client import fetch_all\n\ndef total(src):\n    return len(fetch_all(src))\n",
    }
    cases = {"baseline": dict(base)}
    f = dict(base); f["pkg/models.py"] = MODELS.replace("has_more", "more"); cases["producer_rename"] = f
    f = dict(base); f["pkg/models.py"] = MODELS.replace("class Page", "class Pg"); cases["class_rename"] = f
    f = dict(base); f["pkg/client.py"] = base["pkg/client.py"].replace("page.has_more", "page.has_mroe")
    cases["consumer_typo"] = f
    f = dict(base); f["pkg/client.py"] = base["pkg/client.py"].replace("def fetch_all(src)", "def fetch_all(src, limit)")
    cases["arity"] = f
    f = dict(base); f["pkg/models.py"] = MODELS.replace("self.has_more = has_more", "self.more = has_more")
    f["pkg/client.py"] = "from pkg.models import Page\n\ndef fetch_all(page: Page):\n    return page.has_more\n"
    cases["typed_param"] = f
    f = dict(base); f["pkg/client.py"] = "from pkg.models import Page\n\ndef fetch_all(src):\n    p = Page([], False)\n    return p.has_more_x\n"
    cases["instance_bad_attr"] = f
    return cases


GOLDEN = {
    "baseline": [],
    "producer_rename": [],
    "class_rename": [
        {"kind": "undefined_import", "file": "pkg/client.py", "lineno": 1, "name": "Page", "detail": "pkg/models.py does not define 'Page'"}],
    "consumer_typo": [],
    "arity": [
        {"kind": "arity_mismatch", "file": "pkg/consumer.py", "lineno": 4, "name": "fetch_all", "detail": "call passes 1 positional (+0 kw); pkg/client.py::fetch_all accepts 2..2"}],
    "typed_param": [
        {"kind": "unknown_attribute", "file": "pkg/client.py", "lineno": 4, "name": "has_more", "detail": "class Page exposes ['__init__', 'items', 'more'], not 'has_more'"}],
    "instance_bad_attr": [
        {"kind": "unknown_attribute", "file": "pkg/client.py", "lineno": 5, "name": "has_more_x", "detail": "class Page exposes ['__init__', 'has_more', 'items'], not 'has_more_x'"}],
}


@pytest.mark.parametrize("name", sorted(_audit_cases()))
def test_switch_off_matches_v059(monkeypatch, name):
    files = _audit_cases()[name]
    manifest = chains.seam_manifest(files)
    monkeypatch.delenv(chains_recall.MODE_ENV, raising=False)
    plain = chains.seam_violations(manifest, files)
    with_index = chains.seam_violations(manifest, files, index=index_class().from_sources(files))
    assert with_index == plain == GOLDEN[name]
