"""Signature diff-check: planted inconsistent callers flagged, consistent ones quiet, switch, determinism.

Fixtures port the step-A proof (xf3_proof/RESULT.md): miniature repos plus the audit's where_used
probe A pair (fetch_user gains a required parameter; the BAD service still calls it with one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import kernel_crossfile_signature_check as sc  # noqa: E402
from kernel_crossfile_index import CrossFileIndex  # noqa: E402

LIB = {
    "lib/__init__.py": "",
    "lib/core.py": (
        "def fetch(uid, flag=False):\n    return uid\n\n"
        "def conn(host, *, port):\n    return host\n\n"
        "def plain(a, b):\n    return a\n\n"
        "def swallow(a, *args, **kw):\n    return a\n\n"
        "class Repo:\n"
        "    def __init__(self, name):\n        self.name = name\n"
        "    def get(self, key):\n        return key\n"
        "    @staticmethod\n    def mk(a, b):\n        return a\n"
    ),
    "app/__init__.py": "",
    "app/use.py": (
        "from lib.core import fetch, conn, plain, swallow, Repo\n\n"
        "def go(u):\n"
        "    r = Repo('x')\n"
        "    r.get('k')\n"
        "    Repo.mk(1, 2)\n"
        "    conn('h', port=1)\n"
        "    plain(1, 2)\n"
        "    swallow(1, 2, z=3)\n"
        "    return fetch(u, flag=True)\n"
    ),
}


def edit(src: dict[str, str], *edits: tuple[str, str, str]):
    """(before_index, after_index, changed_paths) after replacing text in `src`."""
    after = dict(src)
    for path, old, new in edits:
        assert old in after[path], (path, old)
        after[path] = after[path].replace(old, new, 1)
    changed = sorted({p for p, _, _ in edits})
    return CrossFileIndex.from_sources(src), CrossFileIndex.from_sources(after), changed


def check(src, *edits):
    return sc.check_signature_edit(*edit(src, *edits))


def reasons(findings):
    return [(f.path, f.line, f.reason) for f in findings]


def test_switch_defaults_off_and_rejects_junk(monkeypatch):
    monkeypatch.delenv(sc.CROSSFILE_SIGNATURE_CHECK_ENV, raising=False)
    assert sc.enabled() is False
    monkeypatch.setenv(sc.CROSSFILE_SIGNATURE_CHECK_ENV, "on")
    assert sc.enabled() is True
    monkeypatch.setenv(sc.CROSSFILE_SIGNATURE_CHECK_ENV, "off")
    assert sc.enabled() is False
    monkeypatch.setenv(sc.CROSSFILE_SIGNATURE_CHECK_ENV, "maybe")
    with pytest.raises(ValueError):
        sc.enabled()


# ---- BAD: each kind of break is reported at the caller
BAD = {
    "add_required": (("lib/core.py", "def plain(a, b):", "def plain(a, b, c):"), "missing required 'c'"),
    "remove_param_kw": (("lib/core.py", "def fetch(uid, flag=False):", "def fetch(uid):"), "unexpected keyword 'flag'"),
    "rename_kw": (("lib/core.py", "def fetch(uid, flag=False):", "def fetch(uid, verbose=False):"),
                  "unexpected keyword 'flag'"),
    "rename_kwonly": (("lib/core.py", "def conn(host, *, port):", "def conn(host, *, p):"),
                      "unexpected keyword 'port'"),
    "pos_to_kwonly": (("lib/core.py", "def plain(a, b):", "def plain(a, *, b):"),
                      "2 positional arguments for at most 1"),
    "remove_positional": (("lib/core.py", "def plain(a, b):", "def plain(a):"),
                          "2 positional arguments for at most 1"),
    "ctor_add_required": (("lib/core.py", "def __init__(self, name):", "def __init__(self, name, mode):"),
                          "missing required 'mode'"),
    "method_add_required": (("lib/core.py", "def get(self, key):", "def get(self, key, default):"),
                            "missing required 'default'"),
    "remove_kwargs": (("lib/core.py", "def swallow(a, *args, **kw):", "def swallow(a, *args):"),
                      "unexpected keyword 'z'"),
}


@pytest.mark.parametrize("kind", sorted(BAD))
def test_bad_edit_flags_the_stale_caller(kind):
    edit_, reason = BAD[kind]
    found = check(LIB, edit_)
    assert [f.path for f in found] == ["app/use.py"], found
    assert found[0].reason == reason
    assert found[0].callee.startswith("lib/core.py::")


def test_bad_remove_vararg_flags_extra_positional():
    found = check(LIB, ("lib/core.py", "def swallow(a, *args, **kw):", "def swallow(a, **kw):"))
    assert reasons(found) == [("app/use.py", 9, "2 positional arguments for at most 1")]


# ---- GOOD: consistent after the edit means silence (the auto-close)
GOOD = {
    "defaulted_add": [("lib/core.py", "def plain(a, b):", "def plain(a, b, c=None):")],
    "defaulted_kwonly_add": [("lib/core.py", "def conn(host, *, port):", "def conn(host, *, port, extra=1):")],
    "required_with_caller_updated": [("lib/core.py", "def plain(a, b):", "def plain(a, b, c):"),
                                     ("app/use.py", "plain(1, 2)", "plain(1, 2, 3)")],
    "remove_kw_with_caller_updated": [("lib/core.py", "def fetch(uid, flag=False):", "def fetch(uid):"),
                                      ("app/use.py", "fetch(u, flag=True)", "fetch(u)")],
    "rename_kw_with_caller_updated": [("lib/core.py", "def fetch(uid, flag=False):", "def fetch(uid, verbose=False):"),
                                      ("app/use.py", "fetch(u, flag=True)", "fetch(u, verbose=True)")],
    "ctor_with_caller_updated": [("lib/core.py", "def __init__(self, name):", "def __init__(self, name, mode):"),
                                 ("app/use.py", "Repo('x')", "Repo('x', 'ro')")],
    "body_only": [("lib/core.py", "def plain(a, b):\n    return a", "def plain(a, b):\n    return b")],
    "rename_positional_param": [("lib/core.py", "def plain(a, b):", "def plain(x, y):")],
    "staticmethod_add_required_updated": [("lib/core.py", "def mk(a, b):", "def mk(a, b, c):"),
                                          ("app/use.py", "Repo.mk(1, 2)", "Repo.mk(1, 2, 3)")],
}


@pytest.mark.parametrize("kind", sorted(GOOD))
def test_consistent_edit_is_quiet(kind):
    assert check(LIB, *GOOD[kind]) == []


def test_audit_probe_a_bad_and_good():
    api_old = "def fetch_user(uid):\n    return uid\n"
    api_new = "def fetch_user(uid, include_deleted):\n    return uid\n"
    head = "from pkg.api import fetch_user\n\ndef load(uid):\n    return fetch_user(%s)\n"
    base = {"pkg/__init__.py": "", "pkg/api.py": api_old, "pkg/svc.py": head % "uid"}
    bad = check(base, ("pkg/api.py", api_old, api_new))
    assert reasons(bad) == [("pkg/svc.py", 4, "missing required 'include_deleted'")]
    good = check(base, ("pkg/api.py", api_old, api_new), ("pkg/svc.py", head % "uid", head % "uid, False"))
    assert good == []


# ---- stay quiet when unsure
def test_star_call_sites_and_decorated_defs_are_not_judged():
    src = {
        "m.py": "def f(a, b):\n    return a\n\ndef d(a):\n    return a\n",
        "u.py": "from m import f, d\n\ndef go(t):\n    f(*t)\n    d(1)\n",
    }
    assert check(src, ("m.py", "def f(a, b):", "def f(a, b, c):")) == []
    deco = {"m.py": "import functools\n\n@functools.cache\ndef d(a):\n    return a\n",
            "u.py": "from m import d\n\ndef go():\n    d(1)\n"}
    assert check(deco, ("m.py", "def d(a):", "def d(a, b):")) == []


def test_preexisting_break_is_not_reported():
    src = {"m.py": "def f(a):\n    return a\n", "u.py": "from m import f\n\ndef go():\n    f(1, 2)\n"}
    assert check(src, ("m.py", "def f(a):", "def f(a, b=None):")) == []
    assert check(src, ("m.py", "def f(a):", "def f(a, b, c=0):")) == []


def test_unresolved_receiver_is_not_judged():
    src = {"m.py": "class C:\n    def get(self, k):\n        return k\n",
           "u.py": "def go(thing):\n    return thing.get('a')\n"}
    assert check(src, ("m.py", "def get(self, k):", "def get(self, k, d):")) == []


def test_no_before_index_or_no_python_change_reports_nothing():
    after = CrossFileIndex.from_sources(LIB)
    assert sc.check_signature_edit(None, after, ["lib/core.py"]) == []
    assert sc.check_signature_edit(after, after, []) == []
    assert sc.check_signature_edit(after, after, ["README.md"]) == []


def test_unchanged_files_are_not_scanned():
    before, after, _ = edit(LIB, ("lib/core.py", "def plain(a, b):", "def plain(a, b, c):"))
    assert sc.check_signature_edit(before, after, ["app/use.py"]) == []


# ---- shape of the result
def test_one_finding_per_site_sorted_and_deterministic():
    src = {
        "m.py": "def f(a):\n    return a\n",
        "b.py": "from m import f\n\ndef g():\n    f(1)\n    f(2)\n",
        "a.py": "from m import f\n\ndef h():\n    f(3)\n",
    }
    # Both edited paths declare f's change once; passing m.py twice must not duplicate.
    b, a, _ = edit(src, ("m.py", "def f(a):", "def f(a, z):"))
    out1 = sc.check_signature_edit(b, a, ["m.py", "m.py"])
    out2 = sc.check_signature_edit(b, a, ["m.py"])
    assert out1 == out2
    assert [(f.path, f.line) for f in out1] == [("a.py", 4), ("b.py", 4), ("b.py", 5)]
    assert len({(f.path, f.line, f.callee) for f in out1}) == len(out1)


def test_rebuilt_indexes_give_identical_findings():
    runs = [check(LIB, BAD["add_required"][0]) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] and runs[0]
