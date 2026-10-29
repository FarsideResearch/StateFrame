"""Bounded characterization of an installed dependency's observable behavior.

The dependency arm is a guidance scaffold.  A model may ask for a narrow
characterization, but the child output is never an application contract or a
completion predicate.  The host binds the result to the package/version pair,
the loaded module, the exercised target, and the sandbox receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from kernel_probe import ProbeReceipt, ProbeRunner, ProbeStatus


MODE_ENV = "STATEFRAME_DEPENDENCY_FACT"
SCHEMA = "stateframe-dependency-observation/1"
MAX_ARGUMENT_BYTES = 4 * 1024
MAX_TEXT = 240
_SENTINEL = "__STATEFRAME_DEPENDENCY_FACT__"
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$")


class DependencyFactStatus(StrEnum):
    OBSERVED = "OBSERVED"
    RAISED = "RAISED"
    TIMED_OUT = "TIMED_OUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    NOT_RUN = "NOT_RUN"
    INVALID = "INVALID"
    RUNNER_ERROR = "RUNNER_ERROR"
    STALE = "STALE"
    POLICY_DENIED = "POLICY_DENIED"


def dependency_fact_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return dependency_fact_mode() == "on"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text(value: object, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _jsonable(value: object) -> object:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("arguments must be JSON-shaped")
    if len(encoded.encode("utf-8")) > MAX_ARGUMENT_BYTES:
        raise ValueError(f"arguments exceed {MAX_ARGUMENT_BYTES} bytes")
    return json.loads(encoded)


def _validate_request(package: object, module: object, target: object,
                      args: object, kwargs: object) -> tuple[str, str, str, list, dict]:
    package = str(package or "").strip()
    module = str(module or package).strip()
    target = str(target or "").strip()
    if not _PACKAGE_RE.fullmatch(package):
        raise ValueError("package must be an import/distribution name")
    if not _NAME_RE.fullmatch(module) or not _NAME_RE.fullmatch(target):
        raise ValueError("module and target must be dotted Python names")
    normalized_args = _jsonable([] if args is None else args)
    normalized_kwargs = _jsonable({} if kwargs is None else kwargs)
    if not isinstance(normalized_args, list):
        raise ValueError("args must be a JSON array")
    if not isinstance(normalized_kwargs, dict):
        raise ValueError("kwargs must be a JSON object")
    return package, module, target, normalized_args, normalized_kwargs


def _child_source(package: str, module: str, target: str, args: list,
                  kwargs: dict, call: bool) -> str:
    request = json.dumps({
        "package": package, "module": module, "target": target,
        "args": args, "kwargs": kwargs, "call": bool(call),
    }, ensure_ascii=False, separators=(",", ":"))
    return """import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import sys

_request = json.loads(%r)
_sentinel = %r

def _short(value, limit=240):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."

def _shape(value):
    def _shape_text(raw, limit=240):
        text = " ".join(str(raw or "").split())
        return text if len(text) <= limit else text[:limit - 3] + "..."

    result = {"type": type(value).__name__,
              "module": getattr(type(value), "__module__", "")}
    qualname = getattr(type(value), "__qualname__", "")
    if qualname:
        result["qualname"] = qualname
    if callable(value):
        try:
            import inspect
            result["signature"] = _shape_text(str(inspect.signature(value)), 180)
        except (TypeError, ValueError):
            result["signature"] = "UNAVAILABLE"
    if isinstance(value, dict):
        result["length"] = len(value)
        result["keys"] = [str(key)[:80] for key in list(value)[:12]]
    elif isinstance(value, (list, tuple, set, frozenset)):
        result["length"] = len(value)
        result["item_types"] = sorted({type(item).__name__ for item in list(value)[:12]})
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result["value"] = _shape_text(repr(value), 180)
    else:
        result["repr"] = _shape_text(repr(value), 180)
    return result

def _version(package):
    import importlib.metadata
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        # Standard-library modules have no distribution metadata.  Binding
        # them to the interpreter version is still more precise than memory.
        return "stdlib:" + ".".join(str(item) for item in sys.version_info[:3])
    except Exception as error:
        return "UNKNOWN:" + type(error).__name__

def _emit(payload):
    import json
    print("__STATEFRAME_DEPENDENCY_FACT__" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")),
          flush=True)

_package = _request["package"]
_module_name = _request["module"]
_target_name = _request["target"]
_version = _version(_package)
try:
    _module = importlib.import_module(_module_name)
    _value = _module
    for _part in _target_name.split("."):
        _value = getattr(_value, _part)
    _operation = "attribute"
    if _request["call"] and callable(_value):
        _value = _value(*_request["args"], **_request["kwargs"])
        _operation = "call"
    _module_file = getattr(_module, "__file__", None)
    _emit({
        "schema": %r, "record_type": "observation", "status": "OBSERVED",
        "package": _package, "version": _version, "module": _module_name,
        "target": _target_name, "operation": _operation, "shape": _shape(_value),
        "module_file": str(_module_file or ""), "observer_complete": True,
    })
except BaseException as _error:
    _emit({
        "schema": %r, "record_type": "observation", "status": "RAISED",
        "package": _package, "version": _version, "module": _module_name,
        "target": _target_name, "operation": "call" if _request["call"] else "attribute",
        "shape": {"status": "RAISED", "exception_type": type(_error).__name__,
                  "message": " ".join(str(_error).split())[:237] + ("..." if len(str(_error)) > 240 else "")},
        "module_file": str(locals().get("_module_file", "")),
        "observer_complete": True,
    })
""" % (request, _SENTINEL, SCHEMA, SCHEMA)


def _payload(output: str) -> Mapping[str, object] | None:
    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(_SENTINEL):
            continue
        try:
            value = json.loads(line[len(_SENTINEL):])
        except (TypeError, ValueError, UnicodeError):
            return None
        return value if isinstance(value, Mapping) else None
    return None


@dataclass(frozen=True, slots=True)
class DependencyFact:
    package: str
    version: str
    module: str
    target: str
    operation: str
    status: str
    observation: Mapping[str, object]
    revision: str
    receipt_id: str = ""
    module_file: str = ""
    authority: str = "HOST_OBSERVED"
    record_type: str = "observation"
    observer_complete: bool = True
    normative: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.package, self.version

    @property
    def fact_key(self) -> dict[str, str]:
        return {"package": self.package, "version": self.version}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "record_type": self.record_type,
            "package": self.package,
            "version": self.version,
            "fact_key": [self.package, self.version],
            "module": self.module,
            "target": self.target,
            "operation": self.operation,
            "status": self.status,
            "observation": dict(self.observation),
            "revision": self.revision,
            "receipt_id": self.receipt_id,
            "module_file": self.module_file,
            "authority": self.authority,
            "observer_complete": self.observer_complete,
            "normative": self.normative,
        }

    def to_host_fact(self):
        from kernel_host_facts import CURRENT, OBSERVATION, HostFact
        return HostFact(
            kind=OBSERVATION,
            predicate="dependency_behavior",
            subject=f"{self.package}@{self.version}:{self.module}.{self.target}",
            value=self.to_dict(),
            scope={"fact_key": [self.package, self.version], "module": self.module,
                   "target": self.target},
            evidence_ids=(self.receipt_id or "dependency:" + _digest(self.to_dict()),),
            dependency_fingerprint=_digest(self.fact_key),
            status=CURRENT,
            revision=self.revision,
        )


@dataclass(frozen=True, slots=True)
class DependencyFactResult:
    status: DependencyFactStatus
    fact: DependencyFact | None = None
    probe_receipt: ProbeReceipt | None = None
    experiment: Mapping[str, object] = None
    receipt: Mapping[str, object] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, DependencyFactStatus):
            object.__setattr__(self, "status", DependencyFactStatus(str(self.status)))
        object.__setattr__(self, "experiment", dict(self.experiment or {}))
        object.__setattr__(self, "receipt", dict(self.receipt or {}))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status.value,
            "reason": self.reason,
            "fact": self.fact.to_dict() if self.fact is not None else None,
            "experiment": dict(self.experiment),
            "receipt": dict(self.receipt),
            "probe_receipt": (self.probe_receipt.to_dict()
                              if self.probe_receipt is not None else None),
        }

    @property
    def host_fact(self):
        return self.fact.to_host_fact() if self.fact is not None else None


class DependencyFactRunner:
    """Run one dependency characterization through the existing safe probe lane."""

    def __init__(self, root: pathlib.Path, *, task_id: str = "", task_deadline: float | None = None,
                 finish_reserve_seconds: float = 0.0, executable: str | None = None,
                 probe_runner: ProbeRunner | None = None):
        self.root = pathlib.Path(root).resolve()
        self.probe_runner = probe_runner or ProbeRunner(
            self.root, task_id=task_id, task_deadline=task_deadline,
            finish_reserve_seconds=finish_reserve_seconds, executable=executable)

    def _not_run(self, status: DependencyFactStatus, reason: str) -> DependencyFactResult:
        return DependencyFactResult(status=status, reason=reason)

    def run(self, package: object, target: object, *, module: object = None,
            args: object = None, kwargs: object = None, call: bool = True,
            question: str | None = None) -> DependencyFactResult:
        if dependency_fact_mode() == "off":
            return self._not_run(DependencyFactStatus.NOT_RUN, "feature_off")
        try:
            package, module, target, args, kwargs = _validate_request(
                package, module, target, args, kwargs)
            code = _child_source(package, module, target, args, kwargs, call)
        except (TypeError, ValueError) as error:
            return self._not_run(DependencyFactStatus.INVALID, str(error))
        bounded_question = _text(question, MAX_TEXT) if question else None
        probe = self.probe_runner.run(code, bounded_question, feature_enabled=True)
        if probe.status is not ProbeStatus.COMPLETED:
            mapped = {
                ProbeStatus.NOT_RUN: DependencyFactStatus.NOT_RUN,
                ProbeStatus.STALE: DependencyFactStatus.STALE,
                ProbeStatus.POLICY_DENIED: DependencyFactStatus.POLICY_DENIED,
                ProbeStatus.RAISED: DependencyFactStatus.RAISED,
                ProbeStatus.TIMED_OUT: DependencyFactStatus.TIMED_OUT,
                ProbeStatus.RESOURCE_LIMIT: DependencyFactStatus.RESOURCE_LIMIT,
                ProbeStatus.RUNNER_ERROR: DependencyFactStatus.RUNNER_ERROR,
            }.get(probe.status, DependencyFactStatus.RUNNER_ERROR)
            return DependencyFactResult(status=mapped, probe_receipt=probe,
                                        reason=probe.reason)
        payload = _payload(probe.captured_output)
        if payload is None:
            return DependencyFactResult(
                status=DependencyFactStatus.RUNNER_ERROR, probe_receipt=probe,
                reason="missing_observation_payload")
        raw_status = str(payload.get("status", "RAISED")).upper()
        status = (DependencyFactStatus.OBSERVED if raw_status == "OBSERVED"
                  else DependencyFactStatus.RAISED)
        version = _text(payload.get("version") or "UNKNOWN", 120)
        observation = payload.get("shape")
        if not isinstance(observation, Mapping):
            observation = {"status": raw_status}
        observation = dict(observation)
        observation.setdefault("status", raw_status)
        fact = DependencyFact(
            package=_text(payload.get("package") or package, 120),
            version=version,
            module=_text(payload.get("module") or module, 160),
            target=_text(payload.get("target") or target, 160),
            operation=_text(payload.get("operation") or ("call" if call else "attribute"), 32),
            status=raw_status,
            observation=observation,
            revision=probe.revision,
            receipt_id=probe.receipt_id,
            module_file=_text(payload.get("module_file"), 240),
            observer_complete=bool(payload.get("observer_complete", True)),
        )
        experiment = {
            "schema": SCHEMA, "record_type": "experiment",
            "authority": "OBSERVATION_ONLY", "question": bounded_question or "",
            "docs_claim": None,
            "application_assumption": None,
            "authorized_compatibility_requirement": None,
            "request_digest": _digest({"package": package, "module": module,
                                        "target": target, "args": args, "kwargs": kwargs,
                                        "call": bool(call)}),
            "candidate_digest": probe.definition_id,
            "loaded_artifact": fact.module_file,
            "dependency_snapshot": {"package": fact.package, "version": fact.version},
            "correlation_id": probe.receipt_id,
            "policy_events": list(probe.policy_events),
            "observer_complete": fact.observer_complete,
        }
        receipt = {
            "schema": SCHEMA, "record_type": "receipt",
            "receipt_id": probe.receipt_id,
            "candidate_digest": probe.definition_id,
            "exercised_process": "sandboxed_python_child",
            "loaded_artifact": fact.module_file,
            "config_digest": _digest({"module": fact.module, "target": fact.target,
                                       "operation": fact.operation}),
            "dependency_snapshot": {"package": fact.package, "version": fact.version},
            "correlation_id": probe.receipt_id,
            "observer_complete": fact.observer_complete,
            "substitutions": [],
            "probe_status": probe.status.value,
            "policy_events": list(probe.policy_events),
        }
        return DependencyFactResult(status=status, fact=fact, probe_receipt=probe,
                                    experiment=experiment, receipt=receipt)


def run_dependency_fact(root: pathlib.Path, package: object, target: object, *,
                        module: object = None, args: object = None, kwargs: object = None,
                        call: bool = True, question: str | None = None, row: object = None,
                        runner: object = None) -> DependencyFactResult:
    """Run a characterization, preserving the feature-off no-child path."""
    if dependency_fact_mode() == "off":
        return DependencyFactResult(status=DependencyFactStatus.NOT_RUN,
                                    reason="feature_off")
    task_id = (row.get("task_id") or row.get("id") or "") if isinstance(row, Mapping) else ""
    deadline = None
    reserve = 0.0
    executable = None
    if isinstance(row, Mapping):
        for key in ("task_deadline", "deadline_monotonic", "attempt_deadline", "deadline"):
            if row.get(key) is not None:
                try:
                    deadline = float(row[key])
                except (TypeError, ValueError):
                    pass
                break
        try:
            reserve = max(0.0, float(row.get("finish_reserve_seconds", 0)))
        except (TypeError, ValueError):
            reserve = 0.0
        if isinstance(row.get("python"), str):
            executable = row["python"]
    if isinstance(runner, DependencyFactRunner):
        selected = runner
    else:
        selected = DependencyFactRunner(
            root, task_id=str(task_id), task_deadline=deadline,
            finish_reserve_seconds=reserve, executable=executable,
            probe_runner=runner if isinstance(runner, ProbeRunner) else None)
    return selected.run(package, target, module=module, args=args, kwargs=kwargs,
                        call=call, question=question)


def render_dependency_fact(result: DependencyFactResult) -> str:
    if result.fact is None:
        return f"dependency_fact {result.status.value} reason={result.reason or 'unknown'}"
    fact = result.fact
    return (f"dependency_fact {result.status.value} receipt={fact.receipt_id} "
            f"key=({fact.package},{fact.version}) target={fact.module}.{fact.target}\n"
            f"HOST_OBSERVED characterization_only: {json.dumps(dict(fact.observation), sort_keys=True)}")


__all__ = [
    "DependencyFact", "DependencyFactResult", "DependencyFactRunner",
    "DependencyFactStatus", "MODE_ENV", "SCHEMA", "dependency_fact_mode",
    "enabled", "render_dependency_fact", "run_dependency_fact",
]
