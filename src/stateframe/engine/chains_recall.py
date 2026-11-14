"""SEAM RECALL — extra seam_violations coverage from an optional CrossFileIndex.

`chains.seam_violations` binds a receiver to a class only for `v = Class(...)`
and simple annotations, and drops any name bound twice. That misses field seams
through untyped locals (`page = fetch(); page.has_more`), annotated names that
are also reassigned from a call (`page: Page = src(0)` then `page = src(n)`),
and relative imports. This module adds those checks from an index that is
scope-aware and follows one return-annotation hop.

Index surface consumed (the crossfile_index_api contract; pinned by
tests/test_crossfile_index_contract_c.py):
  index.defs(path=None) -> Def(path, qualname, kind, fields, [bases])
  index.refs(path)      -> Ref(path, lineno, name, kind, receiver, scope)
  index.binding(path, scope, name) -> type qualname | None
  index.import_target(path, module, level, name) -> "path::name" | None

Fail-open everywhere: an unknown receiver, an unknown base class or an index
that raises yields no violation. Dict-key seams are NOT checked: the index
exposes key refs but no resolvable key producer.
"""
from __future__ import annotations

import difflib
import os
from typing import Mapping

MODE_ENV = "STATEFRAME_SEAM_RECALL"
_VALID_MODES = ("off", "on")


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _class_name(qualname: str | None) -> str | None:
    """'pkg/models.py::Outer.Page' | 'pkg.models.Page' | 'Page' -> 'Page'."""
    if not qualname:
        return None
    return qualname.split("::")[-1].split(".")[-1] or None


def _relative_module_path(path: str, module: str, level: int, files: dict) -> str | None:
    """File a `from <dots><module> import ...` in `path` names, by path arithmetic."""
    parts = path.split("/")[:-1]
    if level - 1 > len(parts):
        return None
    base = parts[:len(parts) - (level - 1)]
    stem = "/".join(base + ([p for p in module.split(".") if p] if module else []))
    cands = (f"{stem}.py", f"{stem}/__init__.py") if stem else ("__init__.py",)
    return next((c for c in cands if c in files), None)


def _index_fields(index) -> dict:
    """class name -> (set of fields + method names, extra base names) from the index."""
    out: dict = {}
    defs = tuple(index.defs())
    for d in defs:
        if d.kind == "class":
            slot = out.setdefault(d.qualname.split(".")[-1], [set(), set()])
            slot[0].update(d.fields)
            slot[1].update(_class_name(b) for b in (getattr(d, "bases", None) or ()))
    for d in defs:
        if d.kind == "def" and "." in d.qualname:
            owner, _, meth = d.qualname.rpartition(".")
            out.setdefault(owner.split(".")[-1], [set(), set()])[0].add(meth)
    return out


def _exposed(cls: str, manifest: dict, fields: dict) -> tuple[set, bool]:
    """(attribute names `cls` exposes incl. repo ancestors, opaque-ancestor flag)."""
    classes = manifest["classes"]
    attrs, opaque, seen, queue = set(), False, {cls}, [cls]
    while queue:
        cur = queue.pop()
        attrs |= set(classes.get(cur, {})) | fields.get(cur, [set(), set()])[0]
        bases = set(fields.get(cur, [set(), set()])[1])
        for d in manifest["definitions"]:
            if d["kind"] == "class" and d["name"] == cur:
                bases |= {b for b in d.get("bases", [])}
        for b in bases:
            if b in classes or b in fields:
                if b not in seen:
                    seen.add(b)
                    queue.append(b)
            else:
                opaque = True
    return attrs, opaque


def _all_exposed(manifest: dict, fields: dict) -> set:
    names = {n for v in fields.values() for n in v[0]}
    for attrs in manifest["classes"].values():
        names |= set(attrs)
    return names


def _seam_class_attrs(manifest: dict) -> set:
    return {d["name"] for d in manifest["definitions"]
            if d["kind"] == "classattr"
            and manifest["seams"].get(f"{d['file']}::{d['qualname']}", {}).get("cross_file")}


def recall_violations(manifest: dict, files: dict, index, builtin_attrs,
                      baseline: dict | None = None) -> list:
    """Extra violations (unknown_attribute, undefined_import, unverified_attribute)."""
    out: list = []
    try:
        fields = _index_fields(index)
        out += _attribute_checks(manifest, files, index, fields, builtin_attrs, baseline)
        out += _relative_import_checks(manifest, files, index)
    except Exception:       # a broken index must never fail a gate closed
        return []
    return out


def _attribute_checks(manifest, files, index, fields, builtin_attrs, baseline) -> list:
    out = []
    classes = manifest["classes"]
    exposed_anywhere = _all_exposed(manifest, fields)
    seam_attrs = _seam_class_attrs(manifest)
    gone = set()
    if baseline:
        gone = {d["name"] for d in baseline.get("definitions", ()) if d["kind"] == "classattr"} \
            - exposed_anywhere
    manifest_bases = {(r["file"], r["lineno"], r["name"]): r.get("base")
                      for r in manifest["references"]
                      if r["kind"] in ("attribute", "call_attr")}
    for path in sorted(files):
        if path in manifest["parse_errors"]:
            continue
        for ref in sorted(index.refs(path), key=lambda r: (r.lineno, r.name)):
            if ref.kind != "attr":
                continue
            # real index: Ref.name is the dotted expression ("page.has_more"); the fake's was bare
            name = ref.name.rsplit(".", 1)[-1]
            if name in builtin_attrs or name.startswith("__"):
                continue
            receiver = ref.receiver
            if receiver is None:        # one hop through binding when the index split them
                base = manifest_bases.get((path, ref.lineno, name))
                if base:
                    receiver = index.binding(path, ref.scope, base)
            cls = _class_name(receiver)
            if cls is not None and cls in classes:
                attrs, opaque = _exposed(cls, manifest, fields)
                if opaque or name in attrs:
                    continue
                out.append(dict(kind="unknown_attribute", file=path, lineno=ref.lineno,
                                name=name, detail=f"class {cls} exposes "
                                f"{sorted(attrs)[:8]}, not {name!r}"))
            elif cls is None and name not in exposed_anywhere:
                near = difflib.get_close_matches(name, sorted(seam_attrs), n=1, cutoff=0.8)
                if name in gone or near:
                    why = "removed from its class" if name in gone else f"resembles seam field {near[0]!r}"
                    out.append(dict(kind="unverified_attribute", severity="info",
                                    file=path, lineno=ref.lineno, name=name,
                                    detail=f"receiver type unknown; {name!r} {why}"))
    return out


def _relative_import_checks(manifest, files, index) -> list:
    out = []
    defs_in_file: dict = {}
    for d in manifest["definitions"]:
        if d["kind"] in ("function", "class", "assign"):
            defs_in_file.setdefault(d["file"], set()).add(d["name"])
    for r in manifest["references"]:
        if r["kind"] != "import" or not r.get("level") or r["name"] == "*":
            continue
        target = _relative_module_path(r["file"], r["module"], r["level"], files)
        if target is None or target not in manifest["files"]:
            continue                                    # external / unresolvable
        if index.import_target(r["file"], r["module"], r["level"], r["name"]):
            continue
        sub = ".".join(p for p in (r["module"], r["name"]) if p)
        if _relative_module_path(r["file"], sub, r["level"], files):
            continue                                    # `from . import submodule`
        if r["name"] in defs_in_file.get(target, set()):
            continue                                    # index lacks it; chains saw it
        if any(i["orig"] == r["name"] for i in manifest["imports"].get(target, {}).values()):
            continue                                    # re-exported
        out.append(dict(kind="undefined_import", file=r["file"], lineno=r["lineno"],
                        name=r["name"], detail=f"{target} does not define {r['name']!r}"))
    return out
