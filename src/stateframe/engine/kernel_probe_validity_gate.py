"""Gold-free validity checks for model-authored public probes.

The public-review organ deliberately treats a generated test as untrusted.  This
module adds a narrower question before that test can become evidence: is the
probe itself a bounded, host-readable observation?  It never looks at hidden
tests or candidate implementation files.  A failure of this check is an
``INCONCLUSIVE`` probe and is quarantined from the evidence stream.

The implementation is intentionally structural.  A model may bind literal
inputs to a public call and choose one of the small comparison relations below;
the host owns the predicate shape and the public-test contradiction check.  If
the assertion is too expressive, the host may salvage the legal construction
and no-exception operation for inspection, but it never promotes that run to a
product violation.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


MODE_ENV = "STATEFRAME_PROBE_VALIDITY_GATE"
VALID = "VALID"
INCONCLUSIVE = "INCONCLUSIVE"
SCHEMA = "stateframe-probe-validity/1"
PREDICATE_SCHEMA = "stateframe-probe-predicate/1"
UNVERIFIED_HEADER = "[UNVERIFIED public probe]"
INCONCLUSIVE_HEADER = "[INCONCLUSIVE public probe]"

_RELATIONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "is", "is_not", "truthy"})
_COMPARES = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Is: "is",
    ast.IsNot: "is_not",
}
_INVERT_RELATION = {
    "eq": "eq", "ne": "ne", "lt": "gt", "le": "ge", "gt": "lt", "ge": "le",
    "is": "is", "is_not": "is_not",
}
# Fix (post-review r44, finding 3, part B): `pytest.raises(<name>)` used to
# accept ANY identifier/dotted-path string -- including an unresolved,
# fictitious exception type -- as a "restricted typed predicate". A raises
# clause naming a real but never-raised builtin exception is itself a
# guaranteed AssertionError/probe failure, exactly like an unresolved name.
# Restrict to the actual builtin exception hierarchy so the predicate can
# only ever name something Python itself could raise; a bare, resolvable
# exception name here is a real, approved type -- never a model invention.
_APPROVED_EXCEPTION_NAMES = frozenset(
    name for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _jsonable(value: object) -> object:
    """Convert a literal into the strict JSON subset used by typed predicates."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError("non-finite literal")
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("literal mapping keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    raise ValueError(f"unsupported literal type: {type(value).__name__}")


def _literal(node: ast.AST) -> object:
    try:
        return _jsonable(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError):
        raise ValueError("expected a bounded JSON literal") from None


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _imports(tree: ast.Module) -> tuple[dict[str, str], dict[str, str], set[str], list[str]]:
    """Return from-imports, module aliases, pytest aliases, and import errors."""
    names: dict[str, str] = {}
    modules: dict[str, str] = {}
    pytest_names: set[str] = set()
    errors: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                alias = item.asname or item.name.split(".")[0]
                if item.name == "pytest":
                    pytest_names.add(alias)
                elif item.name == "app" or item.name.startswith("app."):
                    modules[alias] = item.name
                else:
                    errors.append("UNSUPPORTED_IMPORT")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not (module == "pytest" or module == "app" or module.startswith("app.")):
                errors.append("UNSUPPORTED_IMPORT")
                continue
            for item in node.names:
                if item.name == "*":
                    errors.append("UNSUPPORTED_IMPORT")
                    continue
                alias = item.asname or item.name
                if module == "pytest":
                    if item.name not in {"raises", "approx"}:
                        errors.append("UNSUPPORTED_IMPORT")
                    else:
                        pytest_names.add(alias)
                else:
                    names[alias] = f"{module}:{item.name}"
    return names, modules, pytest_names, errors


def _resolve_subject(node: ast.AST, names: Mapping[str, str], modules: Mapping[str, str]) -> str | None:
    dotted = _dotted(node)
    if not dotted:
        return None
    if isinstance(node, ast.Name) and dotted in names:
        return names[dotted]
    root, _, suffix = dotted.partition(".")
    if root in modules:
        module = modules[root]
        if suffix:
            return f"{module}:{suffix}"
        return None
    if dotted.startswith("app."):
        module, _, function = dotted.rpartition(".")
        return f"{module}:{function}" if module and function else None
    return None


@dataclass(frozen=True)
class CallSpec:
    subject: str
    args: tuple[object, ...] = ()
    keywords: tuple[tuple[str, object], ...] = ()
    expression: str = ""

    def key(self) -> tuple[object, ...]:
        return self.subject, self.args, self.keywords

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "args": list(self.args),
            "keywords": {key: value for key, value in self.keywords},
            "expression": self.expression,
        }


@dataclass(frozen=True)
class TypedPredicate:
    subject: str
    relation: str
    expected: object = None
    args: tuple[object, ...] = ()
    keywords: tuple[tuple[str, object], ...] = ()
    state_key: str = ""
    line: int = 0
    operation: str = "return"

    @property
    def expected_type(self) -> str:
        if self.operation == "raises":
            return "exception"
        return type(self.expected).__name__

    def call_key(self) -> tuple[object, ...]:
        return self.subject, self.args, self.keywords

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PREDICATE_SCHEMA,
            "subject": self.subject,
            "operation": self.operation,
            "relation": self.relation,
            "expected": self.expected,
            "expected_type": self.expected_type,
            "args": list(self.args),
            "keywords": {key: value for key, value in self.keywords},
            "state_key": self.state_key,
            "line": self.line,
        }


@dataclass(frozen=True)
class LegalPrefix:
    source: str | None
    complete: bool
    reasons: tuple[str, ...] = ()
    operation_count: int = 0
    removed_assertions: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-probe-legal-prefix/1",
            "complete": self.complete,
            "available": self.source is not None,
            "operation_count": self.operation_count,
            "removed_assertions": self.removed_assertions,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProbeValidity:
    status: str
    reason_codes: tuple[str, ...] = ()
    predicates: tuple[TypedPredicate, ...] = ()
    salvaged_source: str | None = None
    legal_prefix: LegalPrefix | None = None
    quarantined: bool = False
    evidence_allowed: bool = False
    contradictions: tuple[dict[str, object], ...] = ()
    validation_error: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == VALID

    @property
    def inconclusive(self) -> bool:
        return self.status == INCONCLUSIVE

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "quarantined": self.quarantined,
            "evidence_allowed": self.evidence_allowed,
            "typed_predicates": [predicate.to_dict() for predicate in self.predicates],
            "contradictions": [dict(item) for item in self.contradictions],
            "legal_prefix": self.legal_prefix.to_dict() if self.legal_prefix else None,
            "validation_error": self.validation_error,
        }


@dataclass
class _FunctionAnalysis:
    function: ast.FunctionDef
    safe_prefix: list[ast.stmt] = field(default_factory=list)
    operation_nodes: list[ast.Call] = field(default_factory=list)
    predicates: list[TypedPredicate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    removed_assertions: int = 0
    first_bad_index: int | None = None


@dataclass
class _Analysis:
    source: str
    tree: ast.Module | None
    imports: dict[str, str] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=dict)
    pytest_names: set[str] = field(default_factory=set)
    functions: list[_FunctionAnalysis] = field(default_factory=list)
    predicates: list[TypedPredicate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items if item))


def _call_spec(node: ast.AST, names: Mapping[str, str], modules: Mapping[str, str]) -> CallSpec | None:
    if not isinstance(node, ast.Call):
        return None
    subject = _resolve_subject(node.func, names, modules)
    if subject is None:
        return None
    try:
        args = tuple(_literal(item) for item in node.args)
        keywords = []
        for item in node.keywords:
            if item.arg is None:
                return None
            keywords.append((item.arg, _literal(item.value)))
    except ValueError:
        return None
    try:
        expression = ast.unparse(node)
    except AttributeError:
        expression = subject
    return CallSpec(subject, args, tuple(keywords), expression)


def _state_key(prefix: Sequence[ast.stmt]) -> str:
    module = ast.Module(body=list(prefix), type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def _predicate_from_assert(
    node: ast.Assert,
    bindings: Mapping[str, tuple[CallSpec, ast.Call]],
    names: Mapping[str, str],
    modules: Mapping[str, str],
    prefix: Sequence[ast.stmt],
) -> tuple[TypedPredicate | None, ast.Call | None]:
    test = node.test
    relation = "truthy"
    expected: object = True
    call_node: ast.Call | None = None
    call: CallSpec | None = None
    negated = False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        negated = True
        relation, expected = "truthy", False
        test = test.operand
    if negated and isinstance(test, ast.Compare):
        # Fix (post-review r44, finding 2): `not <comparison>` used to fall
        # through into the Compare branch below, which unconditionally
        # OVERWRITES relation/expected from the comparison itself -- silently
        # dropping the negation (`assert not target() == 1` was accepted as
        # if it were `assert target() == 1`, admitting a publicly
        # contradicted probe as evidence_allowed=True). Correctly
        # complementing every comparison operator is error-prone; reject
        # negated comparisons outright rather than risk it -- the caller
        # gets INCONCLUSIVE instead of a false VALID.
        return None, None
    if isinstance(test, ast.Call):
        call_node = test
        call = _call_spec(test, names, modules)
    elif isinstance(test, ast.Name) and test.id in bindings:
        call, call_node = bindings[test.id]
    elif isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left, right = test.left, test.comparators[0]
        operation = _COMPARES.get(type(test.ops[0]))
        if operation is None:
            return None, None
        if isinstance(left, ast.Name) and left.id in bindings:
            call, call_node = bindings[left.id]
            try:
                expected = _literal(right)
            except ValueError:
                return None, None
            relation = operation
        else:
            call = _call_spec(left, names, modules)
            if call is not None:
                call_node = left
                try:
                    expected = _literal(right)
                except ValueError:
                    return None, None
                relation = operation
            else:
                call = _call_spec(right, names, modules)
                if call is None:
                    return None, None
                call_node = right
                try:
                    expected = _literal(left)
                except ValueError:
                    return None, None
                relation = _INVERT_RELATION.get(operation, "")
    else:
        return None, None
    if call is None or relation not in _RELATIONS:
        return None, None
    return TypedPredicate(
        subject=call.subject, relation=relation, expected=expected,
        args=call.args, keywords=call.keywords, state_key=_state_key(prefix),
        line=getattr(node, "lineno", 0),
    ), call_node


def _raises_predicate(
    node: ast.With,
    names: Mapping[str, str],
    modules: Mapping[str, str],
    pytest_names: set[str],
    prefix: Sequence[ast.stmt],
) -> tuple[TypedPredicate | None, ast.Call | None]:
    if len(node.items) != 1:
        return None, None
    context = node.items[0].context_expr
    if not isinstance(context, ast.Call) or not isinstance(context.func, ast.Attribute):
        return None, None
    if context.func.attr != "raises" or not isinstance(context.func.value, ast.Name):
        return None, None
    if context.func.value.id not in pytest_names or len(context.args) != 1 or context.keywords:
        return None, None
    exception = _dotted(context.args[0])
    if not exception or not exception.isidentifier() and "." not in exception:
        return None, None
    # Fix (post-review r44, finding 3, part B): reject any exception name
    # that does not resolve to a real approved builtin exception type. A
    # fictitious identifier (`ImaginaryError`) is a guaranteed error, not
    # public evidence.
    if exception not in _APPROVED_EXCEPTION_NAMES:
        return None, None
    body_calls = [item.value for item in node.body if isinstance(item, ast.Expr)]
    if len(body_calls) != len(node.body) or len(body_calls) != 1:
        return None, None
    call_node = body_calls[0]
    call = _call_spec(call_node, names, modules)
    if call is None:
        return None, None
    return TypedPredicate(
        subject=call.subject, relation="raises", expected=exception,
        args=call.args, keywords=call.keywords, state_key=_state_key(prefix),
        line=getattr(node, "lineno", 0), operation="raises",
    ), call_node


def _safe_prefix_statement(
    node: ast.stmt,
    names: Mapping[str, str],
    modules: Mapping[str, str],
) -> tuple[bool, ast.Call | None, str | None, str | None]:
    """Return (supported, operation call, bound name, reason)."""
    if isinstance(node, ast.Pass):
        return True, None, None, None
    if isinstance(node, ast.Expr):
        call = _call_spec(node.value, names, modules)
        return (call is not None, node.value if call is not None else None, None,
                None if call is not None else "UNSUPPORTED_LEGAL_PREFIX")
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        call = _call_spec(node.value, names, modules)
        if call is not None:
            return True, node.value, node.targets[0].id, None
        try:
            _literal(node.value)
        except ValueError:
            return False, None, None, "UNSUPPORTED_LEGAL_PREFIX"
        return True, None, node.targets[0].id, None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        call = _call_spec(node.value, names, modules)
        if call is not None:
            return True, node.value, node.target.id, None
        try:
            _literal(node.value)
        except ValueError:
            return False, None, None, "UNSUPPORTED_LEGAL_PREFIX"
        return True, None, node.target.id, None
    return False, None, None, "UNSUPPORTED_LEGAL_PREFIX"


def _analyse(source: object) -> _Analysis:
    text = source if isinstance(source, str) else ""
    result = _Analysis(text, None)
    if not text.strip():
        result.reasons.append("EMPTY_SOURCE")
        return result
    try:
        tree = ast.parse(text)
    except SyntaxError:
        result.reasons.append("SYNTAX_ERROR")
        return result
    result.tree = tree
    names, modules, pytest_names, import_errors = _imports(tree)
    result.imports, result.modules, result.pytest_names = names, modules, pytest_names
    result.reasons.extend(import_errors)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        result.reasons.append("NO_TEST_FUNCTION")
    if any(not node.name.startswith("test_") for node in functions):
        result.reasons.append("NON_TEST_FUNCTION")
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body):
        result.reasons.append("UNSUPPORTED_TOP_LEVEL")
    for function in functions:
        # Fix (post-review r44, finding 4, part B): a fixture-configured or
        # decorated function's real state/inputs are unmodeled here (no
        # fixture value, parametrize expansion, or setup/teardown is
        # analyzed). Comparing its predicates against a fixture-free
        # candidate (or vice versa) as "same state" is unsound -- exclude
        # such functions from predicate extraction entirely, on both the
        # candidate and public sides (this function serves both via
        # extract_typed_predicates / assess_probe).
        has_params = bool(function.args.args or function.args.posonlyargs
                          or function.args.kwonlyargs or function.args.vararg
                          or function.args.kwarg)
        if has_params or function.decorator_list:
            result.reasons.append("UNSUPPORTED_FIXTURE_OR_DECORATOR")
            continue
        info = _FunctionAnalysis(function)
        bindings: dict[str, tuple[CallSpec, ast.Call]] = {}
        prefix: list[ast.stmt] = []
        saw_check = False
        for index, statement in enumerate(function.body):
            if info.first_bad_index is not None:
                # Legal-prefix salvage is a prefix, not a filter over the
                # whole function.  Once construction or an assertion is
                # unsupported, later statements may not smuggle effects into
                # the diagnostic run.
                if isinstance(statement, (ast.Assert, ast.With)):
                    info.removed_assertions += 1
                continue
            predicate = None
            call_node = None
            if isinstance(statement, ast.Assert):
                saw_check = True
                predicate, call_node = _predicate_from_assert(
                    statement, bindings, names, modules, prefix)
                if predicate is None:
                    info.reasons.append("UNSUPPORTED_TYPED_PREDICATE")
                    info.removed_assertions += 1
                    info.first_bad_index = index if info.first_bad_index is None else info.first_bad_index
                else:
                    info.predicates.append(predicate)
                    if call_node is not None:
                        info.operation_nodes.append(call_node)
                    # Fix (post-review r44, finding 4, part A): executing
                    # this assertion's call is itself a state transition for
                    # every LATER statement in the same function. Advance
                    # the prefix so a subsequent assert's state_key differs
                    # -- otherwise two successive stateful calls
                    # (`assert target()==1` then `assert target()==2`) share
                    # the SAME empty state_key and a candidate matching only
                    # the first call gets falsely vetoed against the second.
                    prefix.append(statement)
                continue
            if isinstance(statement, ast.With):
                saw_check = True
                predicate, call_node = _raises_predicate(
                    statement, names, modules, pytest_names, prefix)
                if predicate is None:
                    info.reasons.append("UNSUPPORTED_TYPED_PREDICATE")
                    info.removed_assertions += 1
                    info.first_bad_index = index if info.first_bad_index is None else info.first_bad_index
                else:
                    info.predicates.append(predicate)
                    if call_node is not None:
                        info.operation_nodes.append(call_node)
                    prefix.append(statement)  # advance state (finding 4, part A)
                continue
            supported, call_node, bound_name, reason = _safe_prefix_statement(
                statement, names, modules)
            if not supported:
                info.reasons.append(reason or "UNSUPPORTED_LEGAL_PREFIX")
                info.first_bad_index = index if info.first_bad_index is None else info.first_bad_index
                continue
            # Fix (post-review r44, finding 3, part A): an assignment that
            # shadows an imported callable/module/pytest alias (e.g.
            # `target = 2` where `target` is the imported function under
            # test) must be rejected outright. Otherwise a LATER `target()`
            # call in an assertion is still resolved through the import map
            # -- as if the shadowing reassignment never happened -- admitting
            # a probe that would actually raise TypeError/NameError at
            # runtime as VALID, guaranteed-error evidence.
            if bound_name is not None and (
                    bound_name in names or bound_name in modules
                    or bound_name in pytest_names):
                info.reasons.append("SHADOWED_IMPORT_BINDING")
                info.first_bad_index = index if info.first_bad_index is None else info.first_bad_index
                continue
            # Fix (post-review r44, finding 3, part A): invalidate any STALE
            # binding on every reassignment of the same name, even when the
            # new value is not itself a call (e.g. `x = target(); x = 2`
            # must forget that `x` was ever call-bound -- the old code never
            # popped it, so a later `assert x == 1` still matched the OLD
            # call through `bindings`).
            if bound_name is not None:
                bindings.pop(bound_name, None)
            prefix.append(statement)
            if call_node is not None:
                info.operation_nodes.append(call_node)
                call = _call_spec(call_node, names, modules)
                if call is not None and bound_name:
                    bindings[bound_name] = (call, call_node)
        info.safe_prefix = prefix
        if not saw_check:
            info.reasons.append("NO_ASSERTION")
        result.functions.append(info)
        result.predicates.extend(info.predicates)
        result.reasons.extend(info.reasons)
    # Nested control flow, helpers, dynamic names and unsupported attributes are
    # deliberately rejected by the validity tier even if the old proposal
    # validator would allow them.
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.While,
                             ast.For, ast.AsyncFor, ast.Try, ast.Global, ast.Nonlocal,
                             ast.Yield, ast.YieldFrom)):
            result.reasons.append("UNSUPPORTED_LEGAL_PREFIX")
        if isinstance(node, ast.Call) and _dotted(node.func) in {
                "eval", "exec", "compile", "open", "getattr", "setattr", "delattr",
                "globals", "locals", "vars", "input", "breakpoint"}:
            result.reasons.append("UNSUPPORTED_LEGAL_PREFIX")
    return result


def extract_typed_predicates(source: object) -> tuple[TypedPredicate, ...]:
    """Extract only host-readable predicates; unsupported pieces are omitted."""
    return tuple(_analyse(source).predicates)


def typed_predicates(source: object) -> tuple[TypedPredicate, ...]:
    return extract_typed_predicates(source)


def restricted_typed_predicates(source: object) -> tuple[TypedPredicate, ...]:
    return extract_typed_predicates(source)


def _safe_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return all(item.name == "pytest" or item.name == "app" or item.name.startswith("app.")
                   for item in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return (not node.level and (module == "pytest" or module == "app" or module.startswith("app."))
                and all(item.name != "*" for item in node.names))
    return False


def salvage_legal_prefix(source: object) -> LegalPrefix:
    """Build a no-exception-only probe from the supported prefix.

    Exact assertions and any unsupported suffix are intentionally removed.  The
    resulting source is useful for checking that construction/operation is
    runnable, but callers must retain the ``INCONCLUSIVE`` classification.
    """
    analysis = _analyse(source)
    if analysis.tree is None:
        return LegalPrefix(None, False, _unique(analysis.reasons))
    imports = [node for node in analysis.tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
               and _safe_import(node)]
    functions: list[ast.FunctionDef] = []
    operation_count = 0
    removed = 0
    for info in analysis.functions:
        body: list[ast.stmt] = []
        existing_calls: set[str] = set()
        for statement in info.safe_prefix:
            # A prefix may be followed by an invalid assertion.  Keep only
            # construction statements, never a prior exact assertion.
            if isinstance(statement, (ast.Assert, ast.With)):
                removed += 1
                continue
            body.append(statement)
            value = None
            if isinstance(statement, ast.Expr):
                value = statement.value
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
            if isinstance(value, ast.Call):
                try:
                    existing_calls.add(ast.unparse(value))
                except AttributeError:
                    pass
        operation_texts = set(existing_calls)
        for call_node in info.operation_nodes:
            try:
                text = ast.unparse(call_node)
            except AttributeError:
                text = ""
            if text and text not in operation_texts:
                body.append(ast.Expr(value=call_node))
                operation_texts.add(text)
        operation_count += len(operation_texts)
        if not body and not info.operation_nodes:
            continue
        body.append(ast.Assert(test=ast.Constant(value=True)))
        functions.append(ast.FunctionDef(
            name=info.function.name,
            args=info.function.args,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        ))
        removed += info.removed_assertions
    if not functions or operation_count == 0:
        return LegalPrefix(None, False, _unique((*analysis.reasons, "NO_SAFE_OPERATION")),
                           operation_count, removed)
    module = ast.Module(body=[*imports, *functions], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        rendered = ast.unparse(module) + "\n"
    except AttributeError:
        return LegalPrefix(None, False, _unique((*analysis.reasons, "UNPARSEABLE_PREFIX")),
                           operation_count, removed)
    return LegalPrefix(rendered, not analysis.reasons, _unique(analysis.reasons),
                       operation_count, removed)


def legal_prefix_salvage(source: object) -> LegalPrefix:
    return salvage_legal_prefix(source)


def _sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        result = []
        for item in value.values():
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Mapping):
                source = item.get("source", item.get("content", item.get("parent_src")))
                if isinstance(source, str):
                    result.append(source)
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Mapping):
                source = item.get("source", item.get("content", item.get("parent_src")))
                if isinstance(source, str):
                    result.append(source)
        return tuple(result)
    return ()


def _same_literal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _relation_holds(relation: str, left: object, right: object) -> bool | None:
    try:
        if relation in {"eq", "is"}:
            return _same_literal(left, right)
        if relation in {"ne", "is_not"}:
            return not _same_literal(left, right)
        if relation == "truthy":
            return bool(left) == bool(right)
        if relation == "lt":
            return left < right
        if relation == "le":
            return left <= right
        if relation == "gt":
            return left > right
        if relation == "ge":
            return left >= right
    except (TypeError, ValueError):
        return None
    return None


def _numeric_intersection(first: TypedPredicate, second: TypedPredicate) -> bool | None:
    if not all(isinstance(item.expected, (int, float)) and not isinstance(item.expected, bool)
               for item in (first, second)):
        return None
    lower: tuple[float, bool] | None = None
    upper: tuple[float, bool] | None = None
    exclusions: set[float] = set()
    equality: float | None = None
    for item in (first, second):
        value = float(item.expected)
        relation = item.relation
        if relation in {"eq", "is"}:
            if equality is not None and equality != value:
                return False
            equality = value
        elif relation in {"ne", "is_not"}:
            exclusions.add(value)
        elif relation in {"gt", "ge"}:
            candidate = (value, relation == "ge")
            if lower is None or candidate[0] > lower[0] or (candidate[0] == lower[0] and not candidate[1] and lower[1]):
                lower = candidate
        elif relation in {"lt", "le"}:
            candidate = (value, relation == "le")
            if upper is None or candidate[0] < upper[0] or (candidate[0] == upper[0] and not candidate[1] and upper[1]):
                upper = candidate
        else:
            return None
    if equality is not None:
        if lower is not None and (equality < lower[0] or
                                  (equality == lower[0] and not lower[1])):
            return False
        if upper is not None and (equality > upper[0] or
                                  (equality == upper[0] and not upper[1])):
            return False
        return equality not in exclusions
    if lower is not None and upper is not None:
        if lower[0] > upper[0] or (lower[0] == upper[0] and not (lower[1] and upper[1])):
            return False
        if lower[0] == upper[0] and lower[0] in exclusions:
            return False
    return True


def _contradicts(public: TypedPredicate, candidate: TypedPredicate) -> bool:
    if public.operation != candidate.operation or public.call_key() != candidate.call_key():
        return False
    if public.operation == "raises":
        return public.expected != candidate.expected
    if public.relation == "truthy" and candidate.relation == "truthy":
        return public.expected != candidate.expected
    if public.relation in {"eq", "is"}:
        result = _relation_holds(candidate.relation, public.expected, candidate.expected)
        return result is False
    if candidate.relation in {"eq", "is"}:
        result = _relation_holds(public.relation, candidate.expected, public.expected)
        return result is False
    intersection = _numeric_intersection(public, candidate)
    return intersection is False


def public_test_contradictions(
    predicates: Sequence[TypedPredicate], public_tests: object,
) -> tuple[dict[str, object], ...]:
    """Find contradictions only for an identical call and identical setup prefix."""
    public_predicates = [predicate for source in _sources(public_tests)
                         for predicate in extract_typed_predicates(source)]
    conflicts: list[dict[str, object]] = []
    for candidate in predicates:
        for public in public_predicates:
            if candidate.state_key != public.state_key:
                continue
            if _contradicts(public, candidate):
                conflicts.append({
                    "candidate": candidate.to_dict(),
                    "public": public.to_dict(),
                    "basis": "same_call_same_literal_inputs_same_setup",
                })
    return tuple(conflicts)


def cross_check_public_tests(
    predicates: Sequence[TypedPredicate], public_tests: object,
) -> tuple[dict[str, object], ...]:
    return public_test_contradictions(predicates, public_tests)


def _proposal_source(proposal: object) -> str:
    if isinstance(proposal, Mapping):
        source = proposal.get("test_source")
        return source if isinstance(source, str) else ""
    return ""


def assess_probe(
    proposal: object,
    *,
    public_tests: object = None,
    validation_error: object = None,
) -> ProbeValidity:
    """Classify one model-written probe without consulting gold or candidate code."""
    source = _proposal_source(proposal)
    analysis = _analyse(source)
    reasons = list(analysis.reasons)
    contradictions = public_test_contradictions(analysis.predicates, public_tests)
    if contradictions:
        reasons.append("PUBLIC_TEST_CONTRADICTION")
    validation_text = None
    if validation_error is not None:
        validation_text = str(validation_error)[:300]
        reasons.append("VALIDATION_REJECTED")
    reasons = list(_unique(reasons))
    if not reasons:
        return ProbeValidity(
            VALID, predicates=tuple(analysis.predicates), quarantined=False,
            evidence_allowed=True, contradictions=contradictions,
        )
    prefix = salvage_legal_prefix(source)
    return ProbeValidity(
        INCONCLUSIVE,
        reason_codes=tuple(reasons),
        predicates=tuple(analysis.predicates),
        salvaged_source=prefix.source,
        legal_prefix=prefix,
        quarantined=True,
        evidence_allowed=False,
        contradictions=contradictions,
        validation_error=validation_text,
    )


def classify_probe(*args: object, **kwargs: object) -> ProbeValidity:
    return assess_probe(*args, **kwargs)


def public_probe_header(inconclusive: bool = False) -> str:
    if enabled() and inconclusive:
        return INCONCLUSIVE_HEADER
    return UNVERIFIED_HEADER


def is_inconclusive_card(value: object) -> bool:
    text = str(value or "")
    return (text.startswith(INCONCLUSIVE_HEADER)
            or text.startswith("PROBE_INCONCLUSIVE")
            or '"status": "INCONCLUSIVE"' in text)


def public_probe_result_class(value: object) -> str:
    if enabled() and is_inconclusive_card(value):
        return "PUBLIC_PROBE_INCONCLUSIVE"
    return "PUBLIC_PROBE_EVIDENCE"


def validity_digest(value: ProbeValidity) -> str:
    payload = json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CallSpec", "INCONCLUSIVE", "INCONCLUSIVE_HEADER", "LegalPrefix", "MODE_ENV",
    "PREDICATE_SCHEMA", "UNVERIFIED_HEADER",
    "ProbeValidity", "SCHEMA", "TypedPredicate", "VALID", "assess_probe", "classify_probe",
    "cross_check_public_tests", "enabled", "extract_typed_predicates", "is_inconclusive_card",
    "legal_prefix_salvage", "mode", "public_probe_header", "public_probe_result_class",
    "public_test_contradictions",
    "restricted_typed_predicates", "salvage_legal_prefix", "typed_predicates", "validity_digest",
]
