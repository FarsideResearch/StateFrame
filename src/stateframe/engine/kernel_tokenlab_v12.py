"""Optional task-state optimizations with independent feature switches.

Each component operates on generic source or task evidence and records whether
it contributed to the current request.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

CHALLENGE_SPLIT_ENV = "STATEFRAME_TOKENLAB_CHALLENGE_SPLIT"
SIGNATURE_FACTS_ENV = "STATEFRAME_TOKENLAB_SIGNATURE_FACTS"
STAGED_CLASS_ENV = "STATEFRAME_TOKENLAB_STAGED_CLASS"
EVENT_SPLIT = "challenge_split_applied"
EVENT_SIGFACT = "signature_fact_rendered"
EVENT_STAGED = "ripple_staged_class"
PRODUCER = "kernel_tokenlab_v12"
UNBOUND_REASONS = frozenset({"UNBOUND_OBSERVABLE", "AMBIGUOUS_CONTRACT"})
NOT_ENTAILED_REASONS = frozenset({"NOT_ENTAILED"})
BINDABLE_REASONS = frozenset({"INVALID_BINDING", "UNBOUND", "UNBOUND_INVOCATION", "UNBOUND_ENTRYPOINT"})
_ARITY_RE = re.compile(r"-call-(?P<fn>[A-Za-z_]\w*)-arity-(?P<n>\d+)$")
# v15 wording: current host evidence is bounded; do not make a timeless certification claim.
QUIET_SENTENCE = "not certified by the current host checks; review the ticket and gather evidence if needed"
STAGED_RESULT_CLASS = "STAGED"
STAGED_DISPOSITION = "staged (valid; applies verbatim when re-issued)"
SIGFACT_HEADER = "[Signature facts]"


def _mode(env: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(env, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{env} must be off or on")
    return value


def challenge_split_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(CHALLENGE_SPLIT_ENV, environ) == "on"


def signature_facts_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(SIGNATURE_FACTS_ENV, environ) == "on"


def staged_class_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(STAGED_CLASS_ENV, environ) == "on"


def _emit(runtime: object, kind: str, **payload: object) -> bool:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return False
    try:
        emit(kind, producer=PRODUCER, dispatch=False, **payload)
    except Exception:  # noqa: BLE001 - telemetry never changes behaviour
        return False
    return True


# ------------------------------------------------------------------------------------------------ signatures
def function_signatures(source: str) -> dict[str, dict[str, object]]:
    """Return qualified definitions with enough data to model Python call binding.

    Methods are keyed by their class-qualified name; bare-name aggregation was the
    source of false facts when two classes exposed the same method name.
    """
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, TypeError, ValueError):
        return {}
    out: dict[str, dict[str, object]] = {}

    def visit(body: Sequence[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name))
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            n_required = len(positional) - len(node.args.defaults)
            decorators = {_decorator_name(item) for item in node.decorator_list}
            method_kind = None
            if prefix:
                method_kind = ("staticmethod" if "staticmethod" in decorators else
                               "classmethod" if "classmethod" in decorators else "method")
            qualified = ".".join((*prefix, node.name))
            out[qualified] = {
                "required": [a.arg for a in positional[:n_required]],
                "optional": [a.arg for a in positional[n_required:]],
                "posonly": [a.arg for a in node.args.posonlyargs],
                "kwonly_required": [a.arg for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is None],
                "kwonly_optional": [a.arg for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None],
                "varargs": node.args.vararg is not None,
                "varkw": node.args.kwarg is not None,
                "method_kind": method_kind,
            }

    visit(tree.body)
    return out


def _is_method(sig: Mapping[str, object]) -> bool:
    req = sig.get("required") or []
    return bool(req) and req[0] in {"self", "cls"}


def bind_arity(sig: Mapping[str, object], n_args: int) -> tuple[bool, str]:
    """Does an n-positional-argument call bind?  (self/cls excluded for methods.)"""
    required = [a for a in (sig.get("required") or [])]
    optional = list(sig.get("optional") or [])
    method_kind = sig.get("method_kind")
    if method_kind in {"method", "classmethod"} or (
            method_kind is None and not sig.get("qualified_witness") and _is_method(sig)):
        required = required[1:]
    kw_req = list(sig.get("kwonly_required") or [])
    lo, hi = len(required), len(required) + len(optional)
    varargs = bool(sig.get("varargs"))
    if n_args < lo:
        return False, (f"{n_args}-argument call does not bind: {len(required)} required positional "
                       f"parameter(s) ({', '.join(required)}); TypeError")
    if n_args > hi and not varargs:
        return False, f"{n_args}-argument call does not bind: at most {hi} positional parameter(s); TypeError"
    if kw_req:
        return False, f"{n_args}-argument call does not bind: keyword-only required {', '.join(kw_req)}; TypeError"
    return True, f"{n_args}-argument call binds (required: {', '.join(required) or 'none'}; optional: {', '.join(optional) or 'none'})"


def _product_files(root: object) -> list[Path]:
    try:
        base = Path(str(root)).resolve()
    except (OSError, TypeError, ValueError):
        return []
    files: list[Path] = []
    for path in sorted(base.rglob("*.py")):
        parts = {p.lower() for p in path.relative_to(base).parts}
        if parts & {"tests", "test", ".git", "__pycache__", ".venv", "venv", "node_modules"}:
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        files.append(path)
        if len(files) >= 400:
            break
    return files


def find_signature(root: object, fn: str) -> tuple[str, dict[str, object]] | None:
    """First product file (sorted) defining ``fn``; returns (relative path, signature)."""
    for path in _product_files(root):
        try:
            sigs = function_signatures(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if fn in sigs:
            try:
                rel = path.relative_to(Path(str(root)).resolve()).as_posix()
            except ValueError:
                rel = path.as_posix()
            return rel, sigs[fn]
    return None


def _witness_values(row: Mapping[str, object], keys: Sequence[str]) -> list[str]:
    values: list[str] = []
    sources = [row]
    witness = row.get("witness")
    if isinstance(witness, Mapping):
        sources.insert(0, witness)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return list(dict.fromkeys(values))


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _qualified_signature(source: str, callable_name: str) -> dict[str, object] | None:
    """Resolve exactly one qualified AST callable with explicit Python binding semantics."""
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, TypeError, ValueError):
        return None
    target_parts = tuple(part for part in callable_name.split(".") if part)
    if not target_parts:
        return None
    found: list[dict[str, object]] = []

    def signature(node: ast.FunctionDef | ast.AsyncFunctionDef, *, bound: bool,
                  method_kind: str | None) -> dict[str, object] | None:
        positional = [*node.args.posonlyargs, *node.args.args]
        n_required = len(positional) - len(node.args.defaults)
        required = [item.arg for item in positional[:n_required]]
        optional = [item.arg for item in positional[n_required:]]
        decorators = {_decorator_name(item) for item in node.decorator_list}
        if decorators and not decorators <= {"staticmethod", "classmethod"}:
            return None
        return {
            "required": required, "optional": optional,
            "kwonly_required": [item.arg for item, default in
                                zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None],
            "varargs": node.args.vararg is not None, "varkw": node.args.kwarg is not None,
            "method_kind": method_kind, "qualified_witness": True,
        }

    def visit(body: Sequence[ast.stmt], prefix: tuple[str, ...] = (),
              class_prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name), (*prefix, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = (*prefix, node.name)
                decorators = {_decorator_name(item) for item in node.decorator_list}
                method_kind = None
                bound = bool(class_prefix)
                if bound:
                    method_kind = ("staticmethod" if "staticmethod" in decorators else
                                   "classmethod" if "classmethod" in decorators else "method")
                    bound = method_kind != "staticmethod"
                if qual == target_parts:
                    resolved = signature(node, bound=bound, method_kind=method_kind)
                    if resolved is not None:
                        found.append(resolved)
                # Nested functions are not methods even when declared inside one.
                visit(node.body, qual, ())

    visit(tree.body)
    return found[0] if len(found) == 1 else None


def _qualified_witness_signature(root: object, row: Mapping[str, object], fn: str,
                                 n_args: int) -> tuple[str, dict[str, object]] | None:
    """Require one product file + one module-qualified callable from the witness."""
    paths = _witness_values(row, ("product_path", "target_file", "target_path", "source_path", "file", "path"))
    callables = _witness_values(row, ("qualified_callable", "qualified_entry_point", "entry_point",
                                      "callable", "qualified_name"))
    if len(paths) != 1 or len(callables) != 1:
        return None
    try:
        base = Path(str(root)).resolve()
        candidate = Path(paths[0])
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve()
        relative = candidate.relative_to(base)
    except (OSError, TypeError, ValueError):
        return None
    forbidden = {"tests", "test", ".git", "__pycache__", "venv", ".venv"}
    if (not candidate.is_file()
            or any(part.lower() in forbidden or "backup" in part.lower() for part in relative.parts)):
        return None
    module = ".".join(relative.with_suffix("").parts)
    if module.endswith(".__init__"):
        module = module[:-9]
    qualified = callables[0]
    if ":" in qualified:
        witness_module, callable_name = qualified.split(":", 1)
    else:
        parts = qualified.split(".")
        witness_module, callable_name = ".".join(parts[:-1]), parts[-1]
    if witness_module != module or not callable_name or callable_name.split(".")[-1] != fn:
        return None
    witness = row.get("witness")
    witness_arity = row.get("arity")
    if isinstance(witness, Mapping):
        witness_arity = witness.get("arity", witness_arity)
    if witness_arity is not None and str(witness_arity) != str(n_args):
        return None
    try:
        source = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    signature = _qualified_signature(source, callable_name)
    return (relative.as_posix(), signature) if signature is not None else None


# ------------------------------------------------------------------------------------------------ M1
def split_unverified(rows: Sequence[Mapping[str, object]], runtime: object,
                     reason_text) -> tuple[str, bool, dict[str, int]]:
    """(detail text, budget_applies, counts).  ``reason_text`` = the challenge module's own renderer."""
    unobservable: list[str] = []
    not_entailed: list[str] = []
    bound: list[str] = []
    other: list[str] = []
    root = getattr(getattr(runtime, "owner", None), "root", None)
    for row in rows:
        cid = str(row.get("check_id") or "")
        reason = str(row.get("reason") or "").upper()
        match = _ARITY_RE.search(cid)
        if reason in NOT_ENTAILED_REASONS:
            not_entailed.append(f"{cid} ({reason_text(reason)}): the ticket does not establish this expectation; "
                                "no requirement is inferred")
            continue
        if reason in UNBOUND_REASONS:
            unobservable.append(f"{cid} ({reason_text(reason)})")
            continue
        if match and reason in BINDABLE_REASONS and root is not None:
            found = _qualified_witness_signature(root, row, match.group("fn"), int(match.group("n")))
            if found is not None:
                rel, sig = found
                ok, verdict = bind_arity(sig, int(match.group("n")))
                bound.append(f"{cid}: HOST-MEASURED at the current revision, {rel}: `{match.group('fn')}` {verdict}"
                             + ("" if ok else " -- this is the named behaviour; fix it or show it is intended"))
                continue
        other.append(f"{cid} ({reason_text(reason)})")
    parts: list[str] = []
    if other:
        parts.append(f"[unverified checks] {'; '.join(other)}: these checks could not be verified. This is not a "
                     "demonstrated failure. Check the named behaviour yourself before submitting.")
    if bound:
        parts.append("[bound checks] " + " | ".join(bound))
    if unobservable:
        parts.append(f"[unobservable checks] {'; '.join(unobservable)}: {QUIET_SENTENCE}")
    if not_entailed:
        parts.append("[not entailed checks] " + "; ".join(not_entailed) + ".")
    counts = {"unobservable": len(unobservable), "not_entailed": len(not_entailed),
              "bound": len(bound), "other": len(other)}
    _emit(runtime, EVENT_SPLIT, **counts)
    return " ".join(parts), bool(other or bound), counts


# ------------------------------------------------------------------------------------------------ M2
def _call_shapes(source: str) -> list[dict[str, object]]:
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, TypeError, ValueError):
        return []
    result: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            target = node.func.id
            receiver_bound = False
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.AST = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            target = ".".join(reversed(parts))
            receiver_bound = bool(parts and parts[0] in {"self", "cls"})
        else:
            continue
        result.append({
            "callable": target,
            "positional": len(node.args),
            "keywords": tuple(item.arg for item in node.keywords if item.arg is not None),
            "star_args": any(isinstance(item, ast.Starred) for item in node.args),
            "star_kwargs": any(item.arg is None for item in node.keywords),
            "line": getattr(node, "lineno", 0),
            "receiver_bound": receiver_bound,
        })
    return result


def _normalise_call_shapes(call_shapes: Sequence[Mapping[str, object]] | None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in call_shapes or ():
        target = raw.get("qualified_callable") or raw.get("callable") or raw.get("name")
        if not isinstance(target, str) or not target.strip():
            continue
        positional = raw.get("positional", raw.get("n_args", raw.get("arity", raw.get("args", 0))))
        if isinstance(positional, Sequence) and not isinstance(positional, (str, bytes, bytearray)):
            positional = len(positional)
        try:
            positional = int(positional)
        except (TypeError, ValueError):
            continue
        keywords = raw.get("keywords", ())
        if isinstance(keywords, str):
            keywords = (keywords,)
        if not isinstance(keywords, Sequence):
            keywords = ()
        result.append({
            "callable": target.strip(), "positional": max(0, positional),
            "keywords": tuple(str(item) for item in keywords),
            "star_args": bool(raw.get("star_args")), "star_kwargs": bool(raw.get("star_kwargs")),
            "line": raw.get("line", 0), "receiver_bound": bool(raw.get("receiver_bound")),
        })
    return result


def _candidate_signature(old: Mapping[str, Mapping[str, object]], target: str) -> tuple[str, Mapping[str, object]] | None:
    if target in old:
        return target, old[target]
    suffix = "." + target.rsplit(".", 1)[-1]
    candidates = [(name, sig) for name, sig in old.items() if name.endswith(suffix)]
    return candidates[0] if len(candidates) == 1 else None


def _binds(sig: Mapping[str, object], shape: Mapping[str, object]) -> bool:
    required = list(sig.get("required") or [])
    optional = list(sig.get("optional") or [])
    posonly = set(sig.get("posonly") or ())
    method_kind = sig.get("method_kind")
    if shape.get("receiver_bound") and method_kind in {"method", "classmethod"} and required:
        required = required[1:]
        posonly.discard(str(sig.get("required", [""])[0]))
    positional_names = [*required, *optional]
    count = int(shape.get("positional", 0) or 0)
    if shape.get("star_args"):
        return True
    if count > len(positional_names) and not bool(sig.get("varargs")):
        return False
    supplied = set(positional_names[:count])
    keywords = {str(item) for item in shape.get("keywords", ()) or ()}
    if supplied.intersection(keywords):
        return False
    if any(name in posonly for name in keywords):
        return False
    known = set(positional_names) | set(sig.get("kwonly_required") or ()) | set(sig.get("kwonly_optional") or ())
    if not bool(sig.get("varkw")) and not keywords.issubset(known):
        return False
    missing_required = [name for name in required if name not in supplied and name not in keywords]
    if missing_required and (not shape.get("star_kwargs")
                             or any(name in posonly for name in missing_required)):
        return False
    missing_kwonly = [name for name in (sig.get("kwonly_required") or ()) if name not in keywords]
    if missing_kwonly and not shape.get("star_kwargs"):
        return False
    return True


def _shape_label(name: str, shape: Mapping[str, object]) -> str:
    args = ["?" for _ in range(int(shape.get("positional", 0) or 0))]
    args.extend(f"{key}=?" for key in shape.get("keywords", ()) or ())
    return f"{name}({', '.join(args)})"


def _binding_failure(old: Mapping[str, object], new: Mapping[str, object], shape: Mapping[str, object]) -> str:
    old_required = list(old.get("required") or ())
    new_required = list(new.get("required") or ())
    old_kw = set(old.get("kwonly_required") or ())
    new_kw = set(new.get("kwonly_required") or ())
    added_required = [item for item in new_required if item not in old_required]
    added_kw = sorted(new_kw - old_kw)
    if added_required:
        return f"new required parameter {', '.join(added_required)} (no default)"
    if added_kw:
        return f"new required keyword-only parameter {', '.join(added_kw)} (no default)"
    keywords = set(shape.get("keywords", ()) or ())
    removed_keywords = sorted(keywords - set(new_required) - set(new.get("optional") or ())
                              - set(new.get("kwonly_required") or ()) - set(new.get("kwonly_optional") or ()))
    if removed_keywords and not bool(new.get("varkw")):
        return f"keyword parameter {', '.join(removed_keywords)} was removed or renamed"
    return "the recorded call shape no longer binds"


def signature_facts(before: str, after: str, *,
                    call_shapes: Sequence[Mapping[str, object]] | None = None) -> list[str]:
    """Report only recorded old calls that fail Python binding on the new source.

    No call shape means no claim: a parameter-list diff alone is not a FACT.
    """
    old = function_signatures(before)
    new = function_signatures(after)
    shapes = _normalise_call_shapes(call_shapes) if call_shapes is not None else _call_shapes(before)
    facts: list[str] = []
    for shape in shapes:
        candidate = _candidate_signature(old, str(shape.get("callable", "")))
        if candidate is None:
            continue
        name, old_sig = candidate
        new_sig = new.get(name)
        if new_sig is None or not _binds(old_sig, shape) or _binds(new_sig, shape):
            continue
        facts.append(f"FACT: {_shape_label(name, shape)} no longer binds: "
                     f"{_binding_failure(old_sig, new_sig, shape)}.")
    return list(dict.fromkeys(facts))


def stash_signature_facts(runtime: object, path: object, before: str, after: str, *,
                          revision: object = None,
                          call_shapes: Sequence[Mapping[str, object]] | None = None) -> list[str]:
    facts = signature_facts(before, after, call_shapes=call_shapes)
    if not facts:
        return []
    current = getattr(runtime, "_tokenlab_signature_facts", None)
    entries = list(current) if isinstance(current, list) else []
    entry = {"path": str(path), "revision": str(revision or "sha256:" + hashlib.sha256(
        after.encode("utf-8", errors="replace")).hexdigest()), "facts": facts}
    if not any(item.get("path") == entry["path"] and item.get("revision") == entry["revision"]
               for item in entries if isinstance(item, Mapping)):
        entries.append(entry)
    setattr(runtime, "_tokenlab_signature_facts", entries)
    return facts


def signature_notice(runtime: object) -> str | None:
    stash = getattr(runtime, "_tokenlab_signature_facts", None)
    if not isinstance(stash, list) or not stash:
        return None
    lines = [f"{SIGFACT_HEADER} measured against recorded call shapes at each applied revision:"]
    pending: list[tuple[str, str]] = []
    for entry in stash:
        if not isinstance(entry, Mapping) or not entry.get("facts"):
            continue
        path = str(entry.get("path", ""))
        revision = str(entry.get("revision", "UNKNOWN"))
        lines.append(f"- {path} @ {revision}:")
        lines.extend(f"  {fact}" for fact in entry.get("facts", ()))
        pending.append((path, revision))
    if not pending:
        return None
    lines.append("Callers with the old shape (in this repo or in the ticket's call shapes) need the new parameter "
                 "made optional or the call updated.")
    setattr(runtime, "_tokenlab_signature_notice_pending", pending)
    return "\n".join(lines)


def mark_signature_facts_delivered(runtime: object) -> int:
    """Clear only facts included in a successfully composed provider request."""
    pending = getattr(runtime, "_tokenlab_signature_notice_pending", None)
    entries = getattr(runtime, "_tokenlab_signature_facts", None)
    if not isinstance(pending, list) or not isinstance(entries, list):
        return 0
    keys = {(str(path), str(revision)) for path, revision in pending}
    remaining = [entry for entry in entries
                 if not isinstance(entry, Mapping)
                 or (str(entry.get("path")), str(entry.get("revision"))) not in keys]
    delivered = len(entries) - len(remaining)
    setattr(runtime, "_tokenlab_signature_facts", remaining)
    setattr(runtime, "_tokenlab_signature_notice_pending", [])
    return delivered


# ------------------------------------------------------------------------------------------------ M3
def concrete_evidence_candidates(candidates: Sequence[object]) -> int:
    """Candidates carrying concrete evidence: an obligation, a typed vote with a known family, or a line target."""
    count = 0
    for cand in candidates:
        if bool(getattr(cand, "is_obligation", False)):
            count += 1
            continue
        target = str(getattr(cand, "target", "") or "")
        if ":" in target and target.rsplit(":", 1)[-1].isdigit():
            count += 1
            continue
        for ev in getattr(cand, "evidence", ()) or ():
            family = str(getattr(ev, "family", "") or "").lower()
            if family and family != "unknown" and (getattr(ev, "reason", "") or getattr(ev, "ast_id", "")):
                count += 1
                break
    return count


def staged_feedback(render: str) -> str:
    return (f"[ripple-preview] {STAGED_RESULT_CLASS} ({STAGED_DISPOSITION.split('(', 1)[1].rstrip(')')}) -- "
            "edit NOT applied yet; the file is unchanged. This is NOT an old_str mismatch: the edit is valid.\n"
            + render
            + "\nTo apply this exact edit, re-issue the identical call. "
              "To change it, issue a revised edit. This advice never blocks completion.")


__all__ = ["CHALLENGE_SPLIT_ENV", "EVENT_SIGFACT", "EVENT_SPLIT", "EVENT_STAGED", "PRODUCER", "QUIET_SENTENCE",
           "SIGFACT_HEADER", "SIGNATURE_FACTS_ENV", "STAGED_CLASS_ENV", "STAGED_DISPOSITION", "STAGED_RESULT_CLASS",
           "bind_arity", "challenge_split_enabled", "concrete_evidence_candidates", "find_signature",
           "function_signatures", "mark_signature_facts_delivered", "signature_facts", "signature_facts_enabled",
           "signature_notice", "split_unverified", "staged_class_enabled", "staged_feedback", "stash_signature_facts"]
