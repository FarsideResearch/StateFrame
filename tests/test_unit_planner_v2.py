"""STATEFRAME_UNIT_PLAN_V2: index-ordered, multi-file, cycle-coalescing, gated unit plans."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import chains_recall
import unit_planner as v1
import unit_planner_v2 as v2
from _fake_crossfile_index import FakeCrossFileIndex, Ref, index_class


class RefIndex(FakeCrossFileIndex):
    """The agreed-surface fake plus resolved call/name refs and users() (edges by name)."""

    def _imports(self, path):
        out = {}
        for stmt in self._trees[path].body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
                cand = stmt.module.replace(".", "/") + ".py"
                for a in stmt.names:
                    if cand in self._trees:
                        out[a.asname or a.name] = f"{cand}::{a.name}"
        return out

    def refs(self, path):
        base = list(super().refs(path))
        tree = self._trees.get(path)
        if tree is None:
            return tuple(base)
        local = {d.qualname: f"{path}::{d.qualname}" for d in self.defs(path) if "." not in d.qualname}
        names = {**local, **self._imports(path)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names:
                base.append(Ref(path, n.lineno, n.col_offset, n.id, "name", None, names[n.id], "<module>"))
        return tuple(sorted(base, key=lambda r: (r.lineno, r.col, r.name)))

    def users(self, qualname):
        return tuple(r for p in sorted(self._trees) for r in self.refs(p) if r.target == qualname)


if index_class() is not FakeCrossFileIndex:      # real index already emits name refs and users()
    RefIndex = index_class()  # noqa: F811


MODELS = ("class Page:\n    def __init__(self, items, has_more):\n        self.items = items\n"
          "        self.has_more = has_more\n")
CLIENT = ("from pkg.models import Page\n\ndef fetch(off) -> Page:\n    return Page([], False)\n\n"
          "def fetch_all():\n    out = []\n    page = fetch(0)\n    while True:\n"
          "        out += page.items\n        if not page.has_more:\n            break\n"
          "        page = fetch(len(out))\n    return out\n")
CONSUMER = "from pkg.client import fetch_all\n\ndef consume():\n    return fetch_all()\n"
ORPHAN = "def unused():\n    return 1\n"
PKG = {"pkg/__init__.py": "", "pkg/models.py": MODELS, "pkg/client.py": CLIENT,
       "pkg/consumer.py": CONSUMER, "pkg/orphan.py": ORPHAN}
PKG_TESTS = '''
from pkg import client, consumer, models

def test_page():
    assert models.Page([1], False).has_more is False

def test_fetch_all():
    assert client.fetch_all() == []

def test_consume():
    assert consumer.consume() == []

def test_consume_again():
    assert consumer.consume() == []
'''
SPEC = "Implement class Page, def fetch_all and def consume across the pkg package."


def row(test_src=PKG_TESTS, msg=SPEC, **extra):
    return dict(msg=msg, test_src=test_src, sha="t-v2", **extra)


def plan_for(files=PKG, **kw):
    idx = RefIndex.from_sources(files)
    return v2.derive_unit_plan_v2(row(**kw), idx, sorted(files)), idx


def symbols(plan):
    return [u.symbol for u in plan.units]


# ------------------------------------------------------------ ordering / units
def test_module_style_imports_make_units_and_multi_file():
    plan, _ = plan_for()
    assert symbols(plan) == ["Page", "fetch", "fetch_all", "consume"]
    assert {f for u in plan.units for f in u.files} == {"pkg/models.py", "pkg/client.py", "pkg/consumer.py"}
    assert "unused" not in symbols(plan)                      # untouched orphan stays out


def test_v1_cannot_plan_module_style_imports():
    with pytest.raises(ValueError):
        v1.derive_unit_plan(row(fn="pkg/client.py"))


FROM_STYLE = '''
from pkg.models import Page
from pkg.consumer import consume

def test_a(): assert consume() == [] and Page([], False)
def test_b(): assert consume() == []
def test_c(): assert consume() == []
'''


def test_provider_before_consumer_even_when_consumer_has_more_tests():
    plan, _ = plan_for(test_src=FROM_STYLE)
    order = symbols(plan)
    assert order.index("Page") < order.index("fetch") < order.index("fetch_all") < order.index("consume")
    # v1: co-occurrence is symmetric, so the more-covered consumer wins the tie-break
    legacy = v1.derive_unit_plan(row(test_src=FROM_STYLE, fn="pkg/models.py",
                                     msg="Implement class Page and def consume."))
    assert [u.symbol for u in legacy.units] == ["consume", "Page"]
    consume = plan.units[order.index("consume")]
    assert consume.depends_on and plan.units[0].unit_id in plan.units[order.index("fetch")].depends_on


def test_shared_tests_attach_to_every_unit_they_touch():
    shared = ("from pkg.models import Page\nfrom pkg.client import fetch_all\n"
              "def test_both(): assert fetch_all() == [] and Page([], False)\n"
              "def test_only_page(): assert Page([], True)\n")
    plan, _ = plan_for(test_src=shared)
    by = {u.symbol: u for u in plan.units}
    assert by["Page"].gate_tests == ("test_both", "test_only_page")
    assert by["fetch_all"].gate_tests == ("test_both",)


CYC2 = {"a.py": "from b import B\n\nclass A:\n    def mk(self):\n        return B()\n",
        "b.py": "from a import A\n\nclass B:\n    def back(self):\n        return A()\n",
        "c.py": "from a import A\n\ndef use():\n    return A()\n"}
CYC3 = {"a.py": "from c import C\n\nclass A:\n    def x(self):\n        return C()\n",
        "b.py": "from a import A\n\nclass B:\n    def x(self):\n        return A()\n",
        "c.py": "from b import B\n\nclass C:\n    def x(self):\n        return B()\n",
        "d.py": "from a import A\n\ndef use():\n    return A()\n"}


@pytest.mark.parametrize("files,joint", [(CYC2, "A+B"), (CYC3, "A+B+C")])
def test_cycles_coalesce_into_one_joint_unit(files, joint):
    src = "from a import A\nfrom " + ("c import use" if "c.py" in files and "def use" in files["c.py"]
                                       else "d import use") + "\ndef test_u(): assert use()\n"
    plan, _ = plan_for(files, test_src=src, msg="Implement class A, class B and def use.")
    joints = [u for u in plan.units if u.kind == "joint"]
    assert len(joints) == 1
    assert joints[0].symbol == joint and len(joints[0].members) == len(joint.split("+"))
    assert plan.units[-1].symbol == "use"                      # consumer of the cycle comes after it
    assert joints[0].unit_id in plan.units[-1].depends_on


# ------------------------------------------------------------ digest / render / determinism
def test_plan_is_immutable_digest_pinned_and_renders_with_v1_functions():
    plan, _ = plan_for()
    with pytest.raises(Exception):
        plan.units = ()
    assert plan.plan_digest == v1._digest(plan._basis())
    assert plan.to_dict()["schema"] == "unit-plan/2" and plan.to_dict()["plan_digest"] == plan.plan_digest
    block = v1.render_plan_block(plan)
    assert plan.plan_digest in block and "[unit:Page:g0]" in block
    assert v1.render_current_unit_line(plan, 1).startswith("[planner] current unit 2/4: fetch")
    assert plan.plan_digest != plan_for(test_src=PKG_TESTS.replace("test_page", "test_page_x"))[0].plan_digest


_DET_SCRIPT = """
import sys, json
sys.path[:0] = [{engine!r}, {tests!r}]
import test_unit_planner_v2 as t
plan, _ = t.plan_for(t.CYC3, test_src="from a import A\\nfrom d import use\\ndef test_u(): assert use()\\n",
                     msg="Implement class A, class B and def use.")
print(json.dumps(plan.to_dict(), sort_keys=True))
"""


def test_deterministic_across_hash_seeds():
    import chains
    script = _DET_SCRIPT.format(engine=str(Path(chains.__file__).parent), tests=str(Path(__file__).parent))
    outs = []
    for seed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                              timeout=120, check=True)
        outs.append(done.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1] and json.loads(outs[0])["units"]


# ------------------------------------------------------------ gate
@pytest.fixture
def recall_on(monkeypatch):
    monkeypatch.setenv(chains_recall.MODE_ENV, "on")


ALL_PASS = {t: True for t in ("test_page", "test_fetch_all", "test_consume", "test_consume_again")}


def gate(files, unit="unit:fetch_all:g0", results=None):
    plan, idx = plan_for(PKG)                                  # plan from the intended design
    return v2.unit_gate(plan, unit, RefIndex.from_sources(files), ALL_PASS if results is None else results, files=files)


def test_gate_passes_on_good_package(recall_on):
    verdict = gate(PKG)
    assert verdict.passed and verdict.seam_violations == () and verdict.failing_tests == ()


def test_gate_fails_client_unit_when_producer_field_renamed(recall_on):
    bad = {**PKG, "pkg/models.py": MODELS.replace("self.has_more = has_more", "self.more = has_more")}
    verdict = gate(bad)
    assert not verdict.passed
    assert any(v["file"] == "pkg/client.py" and v["name"] == "has_more" for v in verdict.seam_violations)
    # a unit whose files are untouched by the violation is not blamed for it
    assert gate(bad, unit="unit:consume:g0").passed


def test_gate_fails_on_failing_or_missing_gate_tests(recall_on):
    assert gate(PKG, results={**ALL_PASS, "test_fetch_all": False}).failing_tests == ("test_fetch_all",)
    assert not gate(PKG, results={}).passed
    assert gate(PKG, results={"tests/test_x.py::test_fetch_all": True}).passed


def test_results_from_run_maps_pytest_exit_and_failures():
    plan, _ = plan_for()
    assert set(v2.results_from_run(plan, 0, "").values()) == {True}
    got = v2.results_from_run(plan, 1, "FAILED tests/test_p.py::test_consume - assert 1")
    assert got["test_consume"] is False and got["test_page"] is True
    assert v2.results_from_run(plan, 2, "") is None and v2.results_from_run(plan, None, "") is None


class _Bolt:
    def __init__(self, plan, exit_code):
        self.unit_plan, self.last_test_exit, self.events = plan, exit_code, []

    def _rec(self, *args):
        self.events.append(args)


def test_pointer_advances_only_on_passing_gate(tmp_path, monkeypatch, recall_on):
    monkeypatch.setattr(v2, "_index_class", lambda: RefIndex)
    for rel, text in PKG.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setenv(v2.MODE_ENV, "on")
    plan = v2.derive_with_index(row(), tmp_path)
    assert plan is not None and symbols(plan)[0] == "Page"
    bolt = _Bolt(plan, 0)
    assert v2.gate_and_advance(bolt, tmp_path, "").passed and bolt.unit_index == 1
    bolt.last_test_exit, bolt.unit_index = 1, 2                # fetch_all red: pointer holds
    assert not v2.gate_and_advance(bolt, tmp_path, "FAILED t.py::test_fetch_all").passed
    assert bolt.unit_index == 2
    (tmp_path / "pkg/models.py").write_text(MODELS.replace("self.has_more = has_more", "self.more = has_more"))
    bolt.last_test_exit, bolt.unit_index = 0, 2                # fetch_all: tests green, seam broken
    assert not v2.gate_and_advance(bolt, tmp_path, "").passed and bolt.unit_index == 2
    assert [e[1] for e in bolt.events] == ["unit_gate_pass", "unit_gate_fail", "unit_gate_fail"]


# ------------------------------------------------------------ switch / golden / fallback
V1_ROW = dict(
    fn="pager.py", sha="t1", path="pager.py",
    msg="Implement class Page (items, has_more), class Client (pages()), def consume_all(client), class Orphan.",
    test_src='''
from pager import Page, Client, consume_all, Orphan

def _mk():
    return Client([Page([1,2],True),Page([3],False)])

def test_page_fields():
    p = Page([1], False); assert p.has_more is False
def test_client_pages():
    c=_mk(); assert list(c.pages())
def test_consume_all():
    assert consume_all(_mk()) == [1,2,3]
def test_page_alone_again():
    assert Page([], False).items == []
''')


@pytest.mark.parametrize("joint,digest,text_sha", [
    (False, "sha256:61e7d39e826af02cbf7538fec1e78462dff90598790e229e7857814dd9e2a9e5",
     "8a0c367becbb11333fc2e739a86110d2c3927fa5fe583352f2bb46e2473e82ac"),
    (True, "sha256:9b69a57efe8eb515474414d23757c0383e9e2688aef83a1baca4050adc875847",
     "24dfb3237af7eb13e41df508d4b5d321838a31ac68f03fca5ae6167fa1b3b683"),
])
def test_switch_off_is_byte_identical_to_v059_planner(monkeypatch, joint, digest, text_sha):
    monkeypatch.delenv(v2.MODE_ENV, raising=False)
    assert not v2.enabled() and v2.derive_with_index(V1_ROW, Path(".")) is None
    plan = v1.derive_unit_plan(V1_ROW, joint_handoffs=joint)
    assert plan.plan_digest == digest
    text = "|".join((v1.render_plan_block(plan), v1.render_current_unit_line(plan, 0),
                     v1.render_trim_message(plan, 1)))
    assert hashlib.sha256(text.encode()).hexdigest() == text_sha


def test_switch_values_and_default():
    assert v2.mode({}) == "off" and v2.mode({v2.MODE_ENV: "on"}) == "on"
    assert v2.mode({v2.MODE_ENV: "bogus"}) == "off"


def test_falls_back_to_v1_when_index_unavailable_or_unplannable(tmp_path, monkeypatch):
    monkeypatch.setenv(v2.MODE_ENV, "on")
    for rel, text in PKG.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setattr(v2, "_index_class", lambda: None)      # real index not importable
    assert v2.derive_with_index(row(), tmp_path) is None
    monkeypatch.setattr(v2, "_index_class", lambda: RefIndex)
    assert v2.derive_with_index(row(test_src="def test_x(): pass\n"), tmp_path) is None
    assert v2.derive_with_index(row(), tmp_path) is not None
