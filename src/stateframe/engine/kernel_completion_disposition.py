"""Honest, compact completion disposition and host-only brief smoke checks."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MODE_ENV = "STATEFRAME_DISPOSITION"
_NAME = r"[A-Za-z_]\w*"
_CLASS_RE = re.compile(r"\bclass\s+(" + _NAME + r")")
_EXPLICIT_METHOD_RE = re.compile(r"\b(?:def|function|method)\s+(" + _NAME + r")")
_CALL_RE = re.compile(r"\b(" + _NAME + r")\s*\(")
_VERB_PAIR_RE = re.compile(
    r"\b(" + _NAME + r")\s+(?:and|or)\s+(" + _NAME
    + r")\s+(?:jobs?|tasks?|items?|records?|entries?|values?|objects?)\b", re.I)
_FIELD_RE = re.compile(r"\b(?:field|fields|attribute|attributes)\s+([^.;]+)", re.I)
_IDENTIFIER_RE = re.compile(r"\b(" + _NAME + r")\b")
_STOP_WORDS = {
    "and", "or", "the", "with", "returns", "return", "counts", "count", "value",
    "values", "jobs", "items", "records", "record", "objects", "object", "a", "an",
}


def disposition_mode() -> str:
    """Return the strictly validated completion-disposition mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on", "v2"}:
        raise ValueError(f"{MODE_ENV} must be off, on or v2")
    return value


def enabled() -> bool:
    """Return the legacy mode flag; v2 has separate state and wiring."""
    return disposition_mode() == "on"


def v2_enabled() -> bool:
    return disposition_mode() == "v2"


@dataclass(frozen=True)
class V2DoneDecision:
    """A deterministic stop decision, separate from correctness grading."""

    action: str
    accepted: bool = False
    terminal: str | None = None
    blocker: str | None = None
    detail: str = ""


@dataclass
class V2Disposition:
    """Small state machine for the v2 completion protocol.

    The candidate key is supplied by the host and changes when product or public-test
    bytes change. Evidence gaps and inconclusive review never become blockers here.
    """

    candidate: object = None
    refusal_blocker: str | None = None
    refusal_count: int = 0
    terminal: str | None = None
    host_reviews: dict[str, dict] = field(default_factory=dict)
    repeated_requests: dict[str, int] = field(default_factory=dict)

    def set_candidate(self, candidate: object) -> None:
        if candidate != self.candidate:
            self.candidate = candidate
            self.refusal_blocker = None
            self.refusal_count = 0
            self.terminal = None
            # Reuse is scoped to the unchanged candidate and test set.  The
            # runtime normally includes the candidate in its request key, but
            # keeping this invariant in the state machine prevents a changed
            # candidate from inheriting a prior circuit-breaker count.
            self.repeated_requests.clear()

    def reused_verification(self, request: str) -> tuple[int, str | None]:
        count = self.repeated_requests.get(request, 0) + 1
        self.repeated_requests[request] = count
        if count >= 3:
            self.terminal = "STALLED"
            return count, self.terminal
        return count, None

    def done(self, *, eligible: bool, blocker: str | None = None,
             detail: str = "", recoverable: bool = True) -> V2DoneDecision:
        if self.terminal is not None:
            return V2DoneDecision("terminal", terminal=self.terminal,
                                  blocker=self.refusal_blocker, detail=detail)
        if eligible and not blocker:
            return V2DoneDecision("accept", accepted=True, detail=detail)
        if not blocker:
            # A missing, stale, or inconclusive fact is not an admissible violation.
            return V2DoneDecision("accept", accepted=True, detail=detail)
        if self.refusal_count == 0 and not recoverable:
            # r44 review F2: with no repair tool on the offered menu a refusal would refuse
            # every offered tool; terminalize truthfully (not submitted) instead.
            self.refusal_blocker = blocker
            self.terminal = "INCOMPLETE"
            return V2DoneDecision("terminal", terminal=self.terminal,
                                  blocker=blocker, detail=detail)
        if self.refusal_count == 0:
            self.refusal_blocker = blocker
            self.refusal_count = 1
            return V2DoneDecision("refuse", blocker=blocker, detail=detail)
        if blocker == self.refusal_blocker:
            self.terminal = "INCOMPLETE"
            return V2DoneDecision("terminal", terminal=self.terminal,
                                  blocker=blocker, detail=detail)
        # v2 allows only one recoverable refusal for a candidate. A second
        # witnessed blocker is therefore terminal as well, without another challenge.
        self.terminal = "INCOMPLETE"
        return V2DoneDecision("terminal", terminal=self.terminal,
                              blocker=blocker, detail=detail)


def _brief_symbols(brief: str | None) -> set[str]:
    elements = brief_named_elements(brief)
    symbols = set()
    for owner, values in elements.items():
        owner_name = "__module__" if owner is None else owner
        symbols.add(owner_name)
        for kind in ("methods", "attributes"):
            symbols.update(f"{owner_name}.{name}" for name in values.get(kind, ()))
    return symbols


def _finding_symbol(finding: object) -> str | None:
    if isinstance(finding, Mapping):
        for key in ("symbol", "qualified_symbol", "name", "target"):
            value = finding.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        finding = _finding_text(finding)
    match = re.search(r"brief-named\s+(?:class|method|attribute)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
                      finding)
    return match.group(1) if match else None


def _finding_text(finding: object) -> str:
    """Return a stable, human-readable form for typed or legacy findings."""
    if not isinstance(finding, Mapping):
        return str(finding)
    symbol = _finding_symbol_from_mapping(finding)
    detail = next((finding.get(key) for key in (
        "detail", "message", "finding", "text", "observation")
                    if finding.get(key) is not None), None)
    if symbol and detail is not None:
        return f"HOST_OBSERVED: brief-named {symbol}: {detail}"
    try:
        return json.dumps(dict(finding), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(dict(finding))


def _finding_symbol_from_mapping(finding: Mapping[str, object]) -> str | None:
    for key in ("symbol", "qualified_symbol", "name", "target"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _explicit_bool(finding: Mapping[str, object], keys: Sequence[str]) -> bool | None:
    for key in keys:
        if key not in finding:
            continue
        value = finding[key]
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "on"}:
                return True
            if lowered in {"false", "no", "0", "off"}:
                return False
        return bool(value)
    return None


def _abstract_symbols(source: str | None) -> set[str]:
    """Find abstract bindings so an intentional abstract stub is non-blocking."""
    if not isinstance(source, str):
        return set()
    try:
        import ast
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return set()
    abstract_classes: set[str] = set()
    abstract_methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {getattr(base, "id", None) or getattr(base, "attr", None)
                     for base in node.bases}
            if bases & {"ABC", "ABCMeta"}:
                abstract_classes.add(node.name)
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = {getattr(item, "id", None) or getattr(item, "attr", None)
                              for item in member.decorator_list}
                if "abstractmethod" in decorators:
                    abstract_methods.add(f"{node.name}.{member.name}")
    return abstract_classes | abstract_methods


def admit_v2_findings(findings: object, brief: str | None, *, source: str | None = None) -> dict:
    """Apply admission rule to host observations.

    Only a brief-named, successfully invoked element that actually raised the
    contradiction is a known violation. Structural duplicates and setup failures
    remain concerns/inconclusive and cannot block completion.
    """
    items = _finding_items(findings)
    grounded = _brief_symbols(brief)
    abstract = _abstract_symbols(source)
    known: list[str] = []
    inconclusive: list[str] = []
    for raw_finding in items:
        finding = _finding_text(raw_finding)
        if re.search(r"\babstract(?:method| class| binding)?\b", finding, re.I):
            inconclusive.append(finding)
            continue
        symbol = _finding_symbol(raw_finding)
        if symbol is None:
            inconclusive.append(finding)
            continue
        owner = symbol.split(".", 1)[0]
        if symbol not in grounded or symbol in abstract or owner in abstract:
            inconclusive.append(finding)
            continue
        if isinstance(raw_finding, Mapping):
            # Typed host observations must expose all four admission facts:
            # the brief binding, a valid invocation, actual execution, and a
            # justified contradiction.  An explicit negative keeps the item
            # advisory even if its prose mentions a familiar exception.
            typed_checks = (
                _explicit_bool(raw_finding, ("brief_grounded", "grounded",
                                              "brief_binding")),
                _explicit_bool(raw_finding, ("valid_invocation", "invocation_valid",
                                              "call_valid")),
                _explicit_bool(raw_finding, ("executed", "actually_executed",
                                              "execution_observed")),
                _explicit_bool(raw_finding, ("contradiction", "contradicts_requirement",
                                              "interpretation_justified", "justified")),
            )
            if any(value is not True for value in typed_checks):
                inconclusive.append(finding)
                continue
            known.append(finding)
            continue
        # The smoke path emits this only after a real zero-argument invocation.
        # Missing bindings, required arguments, imports and setup exceptions do not.
        if "NotImplementedError" not in finding or not re.search(r"\braise[ds]?\b", finding):
            inconclusive.append(finding)
            continue
        known.append(finding)
    return {
        "known_violations": list(dict.fromkeys(known)),
        "inconclusive": list(dict.fromkeys(inconclusive)),
        "grounded_symbols": sorted(grounded),
    }


def v2_host_review(findings: object, brief: str | None, *, source: str | None = None) -> dict:
    admitted = admit_v2_findings(findings, brief, source=source)
    if admitted["known_violations"]:
        status = "KNOWN_VIOLATION"
    elif admitted["inconclusive"]:
        status = "INCONCLUSIVE"
    else:
        status = "CLEAN"
    return {"status": status, **admitted}


def _empty_element() -> dict[str, list[str]]:
    return {"methods": [], "attributes": []}


def brief_named_elements(brief: str | None) -> dict[str | None, dict[str, list[str]]]:
    """Extract conservative class/function/method/attribute names from the brief.

    This is deliberately a name extractor, not an implementation parser.  Explicit
    ``class``, ``def``, ``method``, call-shaped, and field/attribute names are the only
    forms admitted to the smoke check.
    """
    if not isinstance(brief, str) or not brief.strip():
        return {}
    result: dict[str | None, dict[str, list[str]]] = {}
    class_matches = list(_CLASS_RE.finditer(brief))
    for index, match in enumerate(class_matches):
        class_name = match.group(1)
        end = class_matches[index + 1].start() if index + 1 < len(class_matches) else len(brief)
        segment = brief[match.end():end]
        element = result.setdefault(class_name, _empty_element())
        methods = [item.group(1) for item in _EXPLICIT_METHOD_RE.finditer(segment)]
        methods.extend(item.group(1) for item in _CALL_RE.finditer(segment))
        methods.extend(name for item in _VERB_PAIR_RE.finditer(segment)
                       for name in item.groups())
        fields = _FIELD_RE.search(segment)
        if fields:
            for item in _IDENTIFIER_RE.findall(fields.group(1)):
                if item.lower() not in _STOP_WORDS:
                    element["attributes"].append(item)
        element["methods"] = list(dict.fromkeys(
            name for name in methods if name != class_name and name.lower() not in _STOP_WORDS))
        element["attributes"] = list(dict.fromkeys(element["attributes"]))

    module_element = result.setdefault(None, _empty_element())
    module_element["methods"] = list(dict.fromkeys(
        match.group(1) for match in _EXPLICIT_METHOD_RE.finditer(brief)
        if not any(match.start() >= class_match.start() for class_match in class_matches)))
    for field_match in _FIELD_RE.finditer(brief):
        if not any(class_match.start() <= field_match.start() <
                   (class_matches[index + 1].start() if index + 1 < len(class_matches)
                    else len(brief)) for index, class_match in enumerate(class_matches)):
            module_element["attributes"].extend(
                item for item in _IDENTIFIER_RE.findall(field_match.group(1))
                if item.lower() not in _STOP_WORDS)
    module_element["attributes"] = list(dict.fromkeys(module_element["attributes"]))
    if not module_element["methods"] and not module_element["attributes"]:
        result.pop(None, None)
    return result


def structural_concerns(source: str, path: str) -> list[str]:
    """Return the existing edit-guard structural observations for disposition use."""
    if not isinstance(source, str):
        return []
    from kernel_edit_guard import host_observed_findings

    findings = host_observed_findings(source, force=True)
    return [f"{line} (location={path})" for line in findings]


_SMOKE_PROGRAM = r'''
import contextlib
import importlib.util
import inspect
import io
import json
import pathlib
import sys

elements = json.loads(sys.argv[1])
module_path = pathlib.Path(sys.argv[2])
concerns = []
marker = object()

def concern(kind, symbol, detail):
    concerns.append(f"HOST_OBSERVED: brief-named {kind} {symbol} at {module_path}: {detail}")

spec = importlib.util.spec_from_file_location("__stateframe_disposition_target__", module_path)
module = importlib.util.module_from_spec(spec) if spec is not None else None
try:
    if spec is None or spec.loader is None or module is None:
        raise ImportError("module spec unavailable")
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
except NotImplementedError:
    concern("module", str(module_path), "import raised NotImplementedError")
    print(json.dumps(concerns))
    raise SystemExit(0)
except Exception as error:
    concern("module", str(module_path), f"import raised {type(error).__name__}")
    print(json.dumps(concerns))
    raise SystemExit(0)

for class_name, requested in elements.items():
    owner = module if class_name == "__module__" else getattr(module, class_name, marker)
    if owner is marker:
        concern("class", class_name, "missing from the live module binding")
        continue
    instance = None
    try:
        signature = inspect.signature(owner)
        required = [parameter for parameter in signature.parameters.values()
                    if parameter.kind in (parameter.POSITIONAL_ONLY,
                                          parameter.POSITIONAL_OR_KEYWORD,
                                          parameter.KEYWORD_ONLY)
                    and parameter.default is parameter.empty]
    except (TypeError, ValueError):
        required = [True]
    if not required and inspect.isclass(owner):
        try:
            instance = owner()
        except NotImplementedError:
            concern("class", class_name, "zero-argument construction raised NotImplementedError")
        except Exception:
            instance = None
    for attribute in requested.get("attributes", []):
        target = instance if instance is not None else owner
        try:
            present = hasattr(target, attribute)
        except Exception:
            present = False
        if not present:
            concern("attribute", f"{class_name}.{attribute}", "missing from the live binding")
    for method in requested.get("methods", []):
        target = instance if instance is not None else owner
        try:
            callable_method = getattr(target, method)
        except Exception:
            callable_method = marker
        if callable_method is marker or not callable(callable_method):
            concern("method", f"{class_name}.{method}", "missing from the live binding")
            continue
        try:
            signature = inspect.signature(callable_method)
            required = [parameter for parameter in signature.parameters.values()
                        if parameter.kind in (parameter.POSITIONAL_ONLY,
                                              parameter.POSITIONAL_OR_KEYWORD,
                                              parameter.KEYWORD_ONLY)
                        and parameter.default is parameter.empty]
        except (TypeError, ValueError):
            required = [True]
        if required:
            continue
        try:
            callable_method()
        except NotImplementedError:
            concern("method", f"{class_name}.{method}",
                    "zero-argument call raised NotImplementedError")
        except Exception:
            pass

print(json.dumps(concerns))
'''


def _embedded_smoke_program(serializable: Mapping[str, object], target: Path) -> str:
    return (_SMOKE_PROGRAM
            .replace("elements = json.loads(sys.argv[1])",
                     "elements = " + repr(dict(serializable)))
            .replace("module_path = pathlib.Path(sys.argv[2])",
                     "module_path = pathlib.Path(" + repr(str(target)) + ")"))


def _parse_smoke_output(output: str, target: Path) -> list[str]:
    lines = output.strip().splitlines()
    if lines:
        candidate = lines[-1]
        if candidate.startswith("HOST_OBSERVED: "):
            candidate = candidate.removeprefix("HOST_OBSERVED: ")
        try:
            concerns = json.loads(candidate)
            if isinstance(concerns, list):
                return [str(item) for item in concerns if str(item).strip()]
        except (TypeError, ValueError):
            pass
    return [f"HOST_OBSERVED: smoke execution at {target} returned no structured result"]


def smoke_named_elements(root: Path, target_path: str, brief: str | None,
                         *, executable: str | None = None, runner=None, row=None) -> list[str]:
    """Observe named brief elements against the live target binding in the workspace."""
    elements = brief_named_elements(brief)
    if not elements:
        return []
    serializable = {
        (class_name if class_name is not None else "__module__"): values
        for class_name, values in elements.items()
    }
    target = Path(target_path)
    if not target.is_absolute():
        target = root / target
    try:
        target = target.resolve()
        target.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return [f"HOST_OBSERVED: smoke execution refused outside workspace: {target}"]
    if callable(runner):
        output = runner(root, _embedded_smoke_program(serializable, target), row=row)
        if isinstance(output, str):
            return _parse_smoke_output(output, target)
        return [f"HOST_OBSERVED: smoke execution at {target} returned no text result"]
    from kernel_test_python import selected_test_python
    interpreter = executable if isinstance(executable, str) and executable else selected_test_python()
    try:
        from kernel_execution_policy import ExecutionPolicyError, run_sandboxed
        execution = run_sandboxed(
            [interpreter, "-I", "-B", "-c", _SMOKE_PROGRAM,
             json.dumps(serializable), str(target)],
            root,
            timeout=10,
            output_limit=64 * 1024,
            readonly_workspace=True,
            cpu_seconds=5,
            memory_bytes=256 * 1024 * 1024,
            file_bytes=1024 * 1024,
        )
    except (OSError, ExecutionPolicyError) as error:
        return [f"HOST_OBSERVED: smoke execution at {target} raised {type(error).__name__}"]
    if execution.timed_out:
        return [f"HOST_OBSERVED: smoke execution at {target} timed out"]
    output = execution.completed.stdout or ""
    if output.strip():
        parsed = _parse_smoke_output(output, target)
        if parsed and not parsed[0].endswith("returned no structured result"):
            return parsed
    if execution.completed.returncode != 0:
        return [f"HOST_OBSERVED: smoke execution at {target} exited {execution.completed.returncode}"]
    return []


def _as_items(value: object) -> list[str]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _finding_items(value: object) -> list[object]:
    """Keep structured host findings structured during admission."""
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "symbol", "qualified_symbol", "name", "target", "detail",
                "message", "finding", "text", "observation")):
            return [value]
        return [item for item in value.values() if item is not None]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if item is not None]
    return [] if value is None else [value]


def _render_items(value: object) -> str:
    items = _as_items(value)
    return "; ".join(items) if items else "none"


def render_disposition(frame: Mapping[str, object], *, status: str,
                       changed_after_green: str, unmet: str,
                       checked_revision: object = None,
                       current_revision: object = None) -> list[str]:
    """Render the compact, qualified completion disposition."""
    revision = current_revision if current_revision is not None else checked_revision
    revision_text = "UNKNOWN" if revision is None else str(revision)
    if status == "GREEN" and changed_after_green == "no" and unmet == "[]":
        required = "PASS_CURRENT"
    elif status == "RED":
        required = "FAIL_CURRENT"
    else:
        required = "MISSING_OR_STALE"

    known = _as_items(frame.get("known_violations", frame.get("known_violation")))
    if status == "RED":
        known.insert(0, f"public checks on the CURRENT revision are RED ({revision_text})")
    if unmet != "[]" and unmet != "UNKNOWN":
        known.append(f"unmet public checks: {unmet}")
    known = list(dict.fromkeys(known))

    from kernel_findings import brief_clause_coverage

    unassessed = [
        f"{item['id']} {item['coverage']}: {item['clause']}"
        for item in brief_clause_coverage(frame)
        if item["coverage"] != "yes"
    ]
    concerns = _as_items(frame.get("host_observed_findings"))
    concerns.extend(_as_items(frame.get("heuristic_concerns")))
    concerns = list(dict.fromkeys(concerns))
    unresolved_questions = _as_items(frame.get("unresolved_questions"))
    eligible = required == "PASS_CURRENT" and not known and not concerns
    lines = [
        "COMPLETION_DISPOSITION (host facts, not instructions):",
        f"required_checks: {required} (revision={revision_text})",
        f"known_violations: {_render_items(known)}",
        f"unassessed_requirements: {_render_items(unassessed)}",
        f"heuristic_concerns: {_render_items(concerns)}",
        f"submission_eligible: {'YES' if eligible else 'NO'}",
    ]
    if unresolved_questions:
        lines.insert(4, f"unresolved_questions: {_render_items(unresolved_questions)}")
    if eligible:
        lines.append("Further identical test runs are not required; done is permitted")
    return lines


def render_v2_disposition(frame: Mapping[str, object], *, status: str,
                          changed_after_green: str, unmet: str,
                          checked_revision: object = None,
                          current_revision: object = None) -> list[str]:
    """Render v2 facts, keeping inconclusive concerns out of the blocking field."""
    revision = current_revision if current_revision is not None else checked_revision
    revision_text = "UNKNOWN" if revision is None else str(revision)
    if status == "GREEN" and changed_after_green == "no" and unmet == "[]":
        required = "PASS_CURRENT"
    elif status == "RED":
        required = "FAIL_CURRENT"
    else:
        required = "MISSING_OR_STALE"
    known = _as_items(frame.get("known_violations", frame.get("known_violation")))
    if status == "RED":
        known.insert(0, f"public checks on the CURRENT revision are RED ({revision_text})")
    if unmet != "[]" and unmet != "UNKNOWN":
        known.append(f"unmet public checks: {unmet}")
    known = list(dict.fromkeys(known))
    from kernel_findings import brief_clause_coverage
    unassessed = [
        f"{item['id']} {item['coverage']}: {item['clause']}"
        for item in brief_clause_coverage(frame) if item["coverage"] != "yes"
    ]
    review = frame.get("host_review")
    review_status = (review.get("status") if isinstance(review, Mapping)
                     else "NOT_RUN")
    if isinstance(review, Mapping):
        known.extend(_as_items(review.get("known_violations")))
    known = list(dict.fromkeys(known))
    settled = required == "PASS_CURRENT"
    review_complete = review_status in {"CLEAN", "KNOWN_VIOLATION", "INCONCLUSIVE"}
    review_pending = settled and review_status == "NOT_RUN"
    if settled and review_status == "INCONCLUSIVE":
        # PASS_CURRENT settles the public-check question.  Smoke transport
        # gaps and clause coverage are not witnessed product defects.
        review_status = "SETTLED"
    elif review_pending:
        # A public green is not an eligible submission until the bounded
        # first-green host review has actually completed.  This is a pending
        # host action, not a product violation.
        review_status = "HOST_CHECK_PENDING"
    concerns = [] if settled else _as_items(frame.get("heuristic_concerns"))
    if isinstance(review, Mapping) and not settled:
        concerns.extend(_as_items(review.get("inconclusive")))
    concerns = list(dict.fromkeys(concerns))
    unresolved_questions = _as_items(frame.get("unresolved_questions"))
    if settled:
        unassessed = []
    eligible = settled and review_complete and not known
    submission = ("BLOCKED" if not settled or known else
                  "HOST_CHECK_PENDING" if not review_complete else "ELIGIBLE")
    terminal = str(frame.get("terminal", frame.get("disposition_terminal", "ACTIVE")))
    lines = [
        "COMPLETION_DISPOSITION (host facts, not instructions):",
        f"required_checks: {required} (revision={revision_text})",
        f"known_violations: {_render_items(known)}",
        f"unassessed_requirements: {_render_items(unassessed)}",
        f"heuristic_concerns: {_render_items(concerns)}",
        f"host_review: {review_status}",
        f"submission: {submission}",
        f"control: terminal={terminal}",
        f"submission_eligible: {'YES' if eligible else 'NO'}",
    ]
    if unresolved_questions:
        lines.insert(4, f"unresolved_questions: {_render_items(unresolved_questions)}")
    if eligible:
        lines.append("Further identical test runs are not required; done is permitted")
    return lines


__all__ = [
    "MODE_ENV", "V2Disposition", "V2DoneDecision", "admit_v2_findings",
    "brief_named_elements", "disposition_mode", "enabled", "render_disposition",
    "render_v2_disposition", "smoke_named_elements", "structural_concerns",
    "v2_enabled", "v2_host_review",
]
