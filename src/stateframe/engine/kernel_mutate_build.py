"""Immutable mutation bundle admission through the isolated executor."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

if __package__:
    from .kernel_mutate_wire import build_stdin_doc, parse_stdout, validate_schema
else:
    from kernel_mutate_wire import build_stdin_doc, parse_stdout, validate_schema


class BuildError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        self.detail = _safe_detail(message or code)
        super().__init__(self.detail)


_PATH_SEGMENT = r"[^/\s\"'<>():,;]+(?: [^/\s\"'<>():,;]+)*"
_ABSOLUTE_PATH = re.compile(rf"(?<![A-Za-z0-9_])/(?:{_PATH_SEGMENT}/)*{_PATH_SEGMENT}")


def _safe_detail(value):
    """Keep model-facing diagnostics bounded and free of host file names."""
    def redact(match):
        candidate = match.group()
        if candidate.startswith(("/sf/", "/in/")):
            return candidate
        suffix = candidate[len(candidate.rstrip(",;:.")):]
        return "<host-path>" + suffix

    value = _ABSOLUTE_PATH.sub(redact, str(value))
    value = "".join(char.encode("unicode_escape").decode("ascii")
                    if unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") else char
                    for char in value)
    return value[:1197] + "..." if len(value) > 1200 else value


def _quoted_tail(value, cap):
    """Return a quoted suffix that fits a diagnostic character budget."""
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(value[-middle:], ensure_ascii=True)) <= cap:
            low = middle
        else:
            high = middle - 1
    return json.dumps(value[-low:] if low else "", ensure_ascii=True)


def _quoted_head(value, cap):
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(value[:middle], ensure_ascii=True)) <= cap:
            low = middle
        else:
            high = middle - 1
    return json.dumps(value[:low], ensure_ascii=True)


_NAME = re.compile(r"[a-z0-9_]{1,40}\.py\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MANIFEST = {"schema", "name", "revision", "parent", "entry", "summary", "kind",
             "input_schema", "output_schema", "requires", "smoke", "limits"}
_OPTIONAL_MANIFEST = {"formats"}
_FORMAT = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*\Z")
MAX_MUTATION_MEMORY_MB = 256
MAX_MUTATION_WALL_S = 120
MAX_MUTATION_CPU_S = 300
_SCHEMA_KEYS = {"type", "properties", "required", "items", "enum", "maxLength", "maxItems", "additionalProperties"}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise BuildError("JSON", str(error)) from error


def _schema(schema, location):
    data = _canonical(schema)
    if len(data) > 8192:
        raise BuildError("SCHEMA_SIZE", f"{location}: exceeds 8192 bytes")
    stack = [(schema, 0, location)]
    while stack:
        node, depth, path = stack.pop()
        if depth > 8:
            raise BuildError("SCHEMA", f"{path}: exceeds maximum depth 8")
        if not isinstance(node, dict):
            raise BuildError("SCHEMA", f"{path}: must be an object")
        unknown = node.keys() - _SCHEMA_KEYS
        if unknown:
            raise BuildError("SCHEMA", f"{path}: unsupported keyword {sorted(unknown)[0]}")
        if not isinstance(node.get("type"), str) or node["type"] not in _TYPES:
            raise BuildError("SCHEMA", f"{path}.type: unsupported or missing type")
        for key in ("maxLength", "maxItems"):
            if key in node and (type(node[key]) is not int or node[key] < 0):
                raise BuildError("SCHEMA", f"{path}.{key}: must be a nonnegative integer")
        if ("maxLength" in node and node["type"] != "string") or ("maxItems" in node and node["type"] != "array"):
            raise BuildError("SCHEMA", f"{path}: size keyword does not match type")
        if ("properties" in node or "required" in node or "additionalProperties" in node) and node["type"] != "object":
            raise BuildError("SCHEMA", f"{path}: object keyword does not match type")
        if "items" in node and node["type"] != "array":
            raise BuildError("SCHEMA", f"{path}.items: requires array type")
        if "additionalProperties" in node and node["additionalProperties"] is not False:
            raise BuildError("SCHEMA", f"{path}.additionalProperties: only false is supported")
        if "required" in node and (not isinstance(node["required"], list) or any(not isinstance(x, str) for x in node["required"])):
            raise BuildError("SCHEMA", f"{path}.required: must be a list of strings")
        if "enum" in node and not isinstance(node["enum"], list):
            raise BuildError("SCHEMA", f"{path}.enum: must be a list")
        if "properties" in node:
            if not isinstance(node["properties"], dict):
                raise BuildError("SCHEMA", f"{path}.properties: must be an object")
            stack.extend((item, depth + 1, f"{path}.properties.{key}")
                         for key, item in node["properties"].items())
        if "items" in node:
            stack.append((node["items"], depth + 1, f"{path}.items"))


def _check(files, manifest):
    if not isinstance(files, Mapping) or not 1 <= len(files) <= 8:
        raise BuildError("FILES", "files: needs 1 to 8 Python modules")
    if not isinstance(manifest, dict):
        raise BuildError("MANIFEST", "manifest: must be an object")
    if not _MANIFEST <= set(manifest) or set(manifest) - _MANIFEST - _OPTIONAL_MANIFEST:
        missing = sorted(_MANIFEST - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST - _OPTIONAL_MANIFEST, key=str)
        raise BuildError("MANIFEST", f"manifest: missing {missing}; unsupported {extra}")
    for name, data in files.items():
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise BuildError("FILE", f"files: name must match [a-z0-9_]{{1,40}}.py: {name!r}")
        if not isinstance(data, bytes):
            raise BuildError("FILE", f"files.{name}: source must be bytes")
        if name[:-3] in sys.stdlib_module_names or name.startswith("sf_mutate_"):
            raise BuildError("SHADOW", f"files.{name}: shadows a trusted module")
    if sum(map(len, files.values())) + len(_canonical(manifest)) > 64 << 10:
        raise BuildError("SIZE", "bundle: exceeds 64 KiB source and manifest cap")
    if manifest["schema"] != "stateframe-mutation-unit/1":
        raise BuildError("MANIFEST", "schema: needs stateframe-mutation-unit/1")
    if not isinstance(manifest["name"], str) or re.fullmatch(r"[a-z][a-z0-9_]{0,39}", manifest["name"]) is None:
        raise BuildError("MANIFEST", "name: must be a lowercase catalog slug")
    if type(manifest["revision"]) is not int or manifest["revision"] < 1:
        raise BuildError("REVISION", "revision: must be a positive integer")
    parent = manifest["parent"]
    if parent is not None and (not isinstance(parent, str) or _DIGEST.fullmatch(parent) is None):
        raise BuildError("PARENT", "parent: must be null or a sha256 source digest")
    if (manifest["revision"] == 1) != (parent is None):
        raise BuildError("PARENT", "parent: revision 1 needs null; later revisions need a digest")
    if not isinstance(manifest["entry"], str) or manifest["entry"] not in files:
        raise BuildError("MANIFEST", "entry: must name one supplied Python module")
    if (not isinstance(manifest["summary"], str) or not manifest["summary"]
            or len(manifest["summary"].encode("utf-8")) > 2048):
        raise BuildError("MANIFEST", "summary: needs 1 to 2048 UTF-8 bytes")
    if not isinstance(manifest["kind"], str) or not manifest["kind"]:
        raise BuildError("MANIFEST", "kind: needs a nonempty string")
    formats = manifest.get("formats", [])
    if (not isinstance(formats, list) or len(formats) > 32
            or any(not isinstance(item, str) or len(item) > 32 or _FORMAT.fullmatch(item) is None
                   for item in formats)
            or len(set(formats)) != len(formats)
            or (manifest["kind"] != "reader" and formats)):
        raise BuildError("FORMATS", "formats: needs unique lowercase reader extensions")
    for name in ("input_schema", "output_schema"):
        _schema(manifest[name], name)
    format_schema = manifest["input_schema"].get("properties", {}).get("format")
    if ("formats" in manifest and isinstance(format_schema, Mapping)
            and isinstance(format_schema.get("enum"), list)
            and (any(not isinstance(item, str) for item in format_schema["enum"])
                 or set(formats) != set(format_schema["enum"]))):
        raise BuildError("FORMATS", "formats: must match the input format enum")
    requires = manifest["requires"]
    if not isinstance(requires, dict) or set(requires) != {"network", "writes"} or any(type(value) is not bool for value in requires.values()):
        raise BuildError("REQUIRES", "requires: needs boolean network and writes")
    limits = manifest["limits"]
    if (not isinstance(limits, dict) or set(limits) != {"wall_s", "cpu_s", "mem_mb"}
            or any(type(value) not in (int, float) or value <= 0 for value in limits.values())
            or limits["wall_s"] > MAX_MUTATION_WALL_S
            or limits["cpu_s"] > MAX_MUTATION_CPU_S
            or limits["mem_mb"] > MAX_MUTATION_MEMORY_MB
            or any(not math.isfinite(value) for value in limits.values())):
        raise BuildError("LIMITS", f"limits: needs positive wall_s at most {MAX_MUTATION_WALL_S}, "
                         f"cpu_s at most {MAX_MUTATION_CPU_S}, and mem_mb at most {MAX_MUTATION_MEMORY_MB}")
    smoke = manifest["smoke"]
    if not isinstance(smoke, list) or not smoke:
        raise BuildError("SMOKE", "smoke: needs at least one input and expected result")
    for index, row in enumerate(smoke):
        if not isinstance(row, dict) or not isinstance(row.get("input"), dict) or not isinstance(row.get("expect"), dict):
            raise BuildError("SMOKE", f"smoke[{index}]: needs input (object) and expect (object)")
        try:
            validate_schema(row["input"], manifest["input_schema"])
        except ValueError as error:
            raise BuildError("SMOKE", f"smoke[{index}].input: does not match input_schema ({getattr(error, 'code', type(error).__name__)})") from error


def source_digest(files: Mapping[str, bytes], manifest: dict) -> str:
    _check(files, manifest)
    rows = [[name, len(data), hashlib.sha256(data).hexdigest()] for name, data in sorted(files.items())]
    return "sha256:" + hashlib.sha256(_canonical(manifest) + _canonical(rows)).hexdigest()


@dataclass(frozen=True)
class StagedUnit:
    source_digest: str
    dir: Path
    manifest: dict
    fixture_root: Path | None = None


@dataclass(frozen=True)
class ValidationReceipt:
    ok: bool
    result: dict
    launch: str


@dataclass(frozen=True)
class SmokeReceipt:
    ok: bool
    results: tuple[dict, ...]
    digest: str


@dataclass(frozen=True)
class Admission:
    admission_digest: str
    source_digest: str
    validation: ValidationReceipt
    smoke: SmokeReceipt


def _fsync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_unit(files: Mapping[str, bytes], manifest: dict, runtime_root, *, fixture_root=None) -> StagedUnit:
    digest = source_digest(files, manifest)
    parent = Path(runtime_root) / "units"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_dir = parent / digest.removeprefix("sha256:")
    expected = {**files, "unit.json": _canonical(manifest)}
    if target_dir.exists() or target_dir.is_symlink():
        _verify_staged(target_dir, expected)
        return StagedUnit(digest, target_dir, dict(manifest), Path(fixture_root) if fixture_root else None)
    directory = Path(tempfile.mkdtemp(prefix=".unit-", dir=parent))
    try:
        for name, data in expected.items():
            target = directory / name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
                os.fchmod(fd, 0o400)
            finally:
                os.close(fd)
        _fsync(directory)
        os.chmod(directory, 0o500)
        try:
            os.rename(directory, target_dir)
        except FileExistsError:
            _verify_staged(target_dir, expected)
        _fsync(parent)
        return StagedUnit(digest, target_dir, dict(manifest), Path(fixture_root) if fixture_root else None)
    except BaseException:
        raise
    finally:
        if directory.exists():
            os.chmod(directory, 0o700)
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()


def _verify_staged(directory, expected):
    if (directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o222
            or {path.name for path in directory.iterdir()} != set(expected)):
        raise BuildError("STAGE_CONFLICT")
    for name, data in expected.items():
        try:
            fd = os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as error:
            raise BuildError("STAGE_CONFLICT") from error
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222
                    or info.st_size != len(data)):
                raise BuildError("STAGE_CONFLICT")
            received = bytearray()
            while len(received) < info.st_size:
                block = os.read(fd, info.st_size - len(received))
                if not block:
                    raise BuildError("STAGE_CONFLICT")
                received.extend(block)
            if received != data:
                raise BuildError("STAGE_CONFLICT")
        finally:
            os.close(fd)


def _snapshot_files(files, *, cap=16):
    if files is None:
        return {}, []
    if not isinstance(files, Mapping) or len(files) > cap:
        raise BuildError("FIXTURE_INPUTS")
    admitted = {}
    descriptors = []
    total = 0
    for ident, path in files.items():
        if not isinstance(ident, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,39}", ident) is None:
            raise BuildError("FIXTURE_INPUTS")
        try:
            path = Path(path)
            info = path.lstat()
        except (OSError, TypeError, ValueError) as error:
            raise BuildError("FIXTURE_INPUTS") from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4 << 20:
            raise BuildError("FIXTURE_INPUTS")
        total += info.st_size
        if total > 16 << 20:
            raise BuildError("FIXTURE_INPUTS")
        admitted[ident] = path
        descriptors.append({"id": ident, "path": "/in/" + ident, "bytes": info.st_size, "name": path.name})
    return admitted, descriptors


def execution_failure_detail(outcome, role, call) -> str | None:
    """Bounded, untrusted diagnostic for a launch that did not complete, or None when it did."""
    if outcome.stdout_overflow:
        return f"{role}: output overflow"
    if outcome.timed_out:
        return f"{role}: timed out"
    if outcome.exit_code == 0:
        return None
    if outcome.exit_code in (3, 4):
        try:
            frame = parse_stdout(outcome.stdout, call=call)
        except ValueError:
            frame = None
        if frame is not None and frame.kind == "error":
            error = frame.error
            trace = "\n".join(error["trace"].splitlines()[-12:])
            stderr = frame.stderr[-600:]
            detail = (f"{role}: boot error frame (untrusted) "
                      f"type={json.dumps(error['type'])} "
                      f"message={_quoted_head(error['message'], 180)} "
                      f"trace={_quoted_tail(trace, 360)} "
                      f"stderr={_quoted_tail(stderr, 520)}")
            import_name = re.fullmatch(r"No module named (['\"])([^'\"]+)\1", error["message"])
            last_line = trace.rsplit("\n", 1)[-1]
            exception_type, separator, exception_message = last_line.partition(": ")
            is_import_error = (error["type"] in ("ModuleNotFoundError", "ImportError")
                               or (separator and exception_type in ("ModuleNotFoundError", "ImportError")
                                   and exception_message == error["message"]))
            if import_name is not None and is_import_error:
                if __package__:
                    from .kernel_mutate_exec import MutationPlatformError, _runtime_closure
                else:
                    from kernel_mutate_exec import MutationPlatformError, _runtime_closure
                try:
                    excluded = _runtime_closure().excluded
                except MutationPlatformError:
                    return detail
                components = set(import_name.group(2).split("."))
                for filename, missing in excluded:
                    module = filename.split(".", 1)[0]
                    if module in components:
                        prefix = (f"unavailable extension {module} on this host "
                                  f"(not mounted: missing {', '.join(missing)}); "
                                  "use a pure-Python alternative")
                        return _safe_detail(prefix + "; " + detail)
            return detail
    return f"{role}: crashed (exit {outcome.exit_code}); no valid boot error frame"


def _run(staged, executor, role, payload, *, call=None, inputs=None, responses=None, response_files=None,
         timeout_retries=0, wall_timeout_s=None):
    if __package__:
        from .kernel_mutate_exec import LaunchSpec, MutationLimits
    else:
        from kernel_mutate_exec import LaunchSpec, MutationLimits
    if type(timeout_retries) is not int or timeout_retries not in (0, 1):
        raise BuildError("TIMEOUT_RETRIES", "timeout retries must be zero or one")
    call = call or "mc" + uuid.uuid4().hex
    limits = staged.manifest["limits"]
    wall_s = float(limits["wall_s"])
    if wall_timeout_s is not None:
        if (type(wall_timeout_s) not in (int, float) or not 0 < wall_timeout_s <= 60
                or not math.isfinite(wall_timeout_s)):
            raise BuildError("LIMITS", "qualification wall timeout must be in (0, 60]")
        wall_s = max(wall_s, float(wall_timeout_s))
    input_files, input_descriptors = _snapshot_files(inputs)
    if response_files is None:
        response_files = {}
    if not isinstance(response_files, Mapping) or any(not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None
                                                     for key in response_files):
        raise BuildError("FIXTURE_RESPONSES")
    response_docs = [] if responses is None else responses
    if (not isinstance(response_docs, list) or any(not isinstance(item, dict)
            or not isinstance(item.get("body"), str) for item in response_docs)
            or {item["body"].removeprefix("/resp/") for item in response_docs} != set(response_files)):
        raise BuildError("FIXTURE_RESPONSES")
    total_response_bytes = 0
    for digest, path in response_files.items():
        try:
            path = Path(path)
            info = path.lstat()
        except (OSError, TypeError, ValueError) as error:
            raise BuildError("FIXTURE_RESPONSES") from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise BuildError("FIXTURE_RESPONSES")
        total_response_bytes += info.st_size
        if total_response_bytes > 8 << 20:
            raise BuildError("FIXTURE_RESPONSES")
        calculated = hashlib.sha256()
        measured = 0
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            while block := os.read(fd, 1 << 20):
                measured += len(block)
                if measured > 8 << 20:
                    raise BuildError("FIXTURE_RESPONSES")
                calculated.update(block)
        finally:
            os.close(fd)
        if measured != info.st_size or calculated.hexdigest() != digest or any(row.get("bytes") != info.st_size for row in response_docs
                                                 if row.get("body") == "/resp/" + digest):
            raise BuildError("FIXTURE_RESPONSES")
    for attempt in range(timeout_retries + 1):
        launch = uuid.uuid4().hex
        doc = build_stdin_doc(wire=1, call=call, launch=launch, phase=0, role=role,
                              input=payload, inputs=input_descriptors, responses=response_docs,
                              grants=staged.manifest["requires"], max_fetch=0)
        spec = LaunchSpec(
            role=role, call=call, launch=launch, unit_dir=staged.dir,
            inputs=input_files, responses={key: Path(path) for key, path in response_files.items()},
            stdin_doc=doc,
            limits=MutationLimits(wall_s=wall_s, cpu_s=limits["cpu_s"],
                                  mem_mb=int(limits["mem_mb"])),
        )
        outcome = executor.run(spec)
        if not getattr(outcome.cleanup, "ok", False):
            if __package__:
                from .kernel_mutate_exec import MutationContainmentError
            else:
                from kernel_mutate_exec import MutationContainmentError
            raise MutationContainmentError("mutation validation cleanup could not be verified")
        detail = execution_failure_detail(outcome, role, call)
        if detail is None:
            break
        timed_out = outcome.timed_out and not outcome.stdout_overflow
        if timed_out and attempt < timeout_retries:
            continue
        timeout_code = timed_out and (timeout_retries > 0 or wall_timeout_s is not None)
        raise BuildError("TIMEOUT" if timeout_code else "EXECUTION", detail)
    frame = parse_stdout(outcome.stdout, call=call)
    if frame.kind != "result":
        raise BuildError("PROTOCOL")
    return launch, frame.result


def validate(staged: StagedUnit, executor, *, call=None, timeout_retries=0, wall_timeout_s=None) -> ValidationReceipt:
    launch, result = _run(staged, executor, "validator", {}, call=call,
                          timeout_retries=timeout_retries, wall_timeout_s=wall_timeout_s)
    if type(result.get("ok")) is not bool or not isinstance(result.get("errors"), list) or not isinstance(result.get("hints"), list):
        raise BuildError("VALIDATOR", "validator: invalid result object")
    return ValidationReceipt(result["ok"], result, launch)


def smoke(staged: StagedUnit, executor, fixtures=None, *, call=None,
          timeout_retries=0, wall_timeout_s=None) -> SmokeReceipt:
    rows = staged.manifest["smoke"] if fixtures is None else fixtures
    results = []
    for index, row in enumerate(rows):
        inputs = row.get("inputs")
        responses = row.get("responses")
        response_files = row.get("response_files")
        if staged.fixture_root is not None:
            if not isinstance(inputs, list) or any(not isinstance(item, str) or
                   not re.fullmatch(r"fixtures/[a-zA-Z0-9_.-]{1,128}", item) for item in inputs):
                raise BuildError("FIXTURE_INPUTS")
            inputs = {f"in{index}": staged.fixture_root / item for index, item in enumerate(inputs)}
            response_files = {}
            normalized = []
            for item in responses or []:
                if (not isinstance(item, dict) or set(item) != {"url", "body"} or
                        not isinstance(item["url"], str) or not isinstance(item["body"], str)
                        or not re.fullmatch(r"fixtures/[a-zA-Z0-9_.-]{1,128}", item["body"])):
                    raise BuildError("FIXTURE_RESPONSES")
                path = staged.fixture_root / item["body"]
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        info = os.fstat(fd)
                        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                                or info.st_size > 8 << 20):
                            raise BuildError("FIXTURE_RESPONSES")
                        data = os.read(fd, (8 << 20) + 1)
                        if len(data) != info.st_size:
                            raise BuildError("FIXTURE_RESPONSES")
                    finally:
                        os.close(fd)
                except OSError as error:
                    raise BuildError("FIXTURE_RESPONSES") from error
                digest = hashlib.sha256(data).hexdigest()
                response_files[digest] = path
                if __package__:
                    from .kernel_mutate_wire import request_key
                else:
                    from kernel_mutate_wire import request_key
                normalized.append({"key": request_key(item["url"], "text/html"), "method": "GET",
                                   "url": item["url"], "accept": "text/html", "status": 200,
                                   "headers": {"content-type": "text/html"},
                                   "final_url": item["url"], "body": "/resp/" + digest,
                                   "bytes": len(data), "error": None})
            responses = normalized
        _, result = _run(
            staged, executor, "smoke", row["input"], call=call, inputs=inputs,
            responses=responses, response_files=response_files,
            timeout_retries=timeout_retries, wall_timeout_s=wall_timeout_s,
        )
        validate_schema(result, staged.manifest["output_schema"])
        expect = row["expect"]
        if not isinstance(expect, dict) or not _subset(result, expect):
            path = _first_difference(result, expect)
            expected_json = _canonical(expect).decode("utf-8")
            actual_json = _canonical(result).decode("utf-8")
            raise BuildError("SMOKE_MISMATCH", f"smoke[{index}] first difference: {path}; "
                             f"expected: {_json_excerpt(expected_json)}; "
                             f"actual: {_json_excerpt(actual_json)}")
        results.append(result)
    return SmokeReceipt(True, tuple(results), "sha256:" + hashlib.sha256(_canonical(results)).hexdigest())


def _subset(actual, expected):
    stack = [(actual, expected)]
    while stack:
        found, wanted = stack.pop()
        if isinstance(wanted, dict):
            if not isinstance(found, dict) or not wanted.keys() <= found.keys():
                return False
            stack.extend((found[key], value) for key, value in wanted.items())
        elif isinstance(wanted, list):
            if not isinstance(found, list) or len(found) != len(wanted):
                return False
            stack.extend(zip(found, wanted))
        elif found != wanted or (type(found) is bool) != (type(wanted) is bool):
            return False
    return True


def _json_excerpt(value):
    return value if len(value) <= 400 else value[:397] + "..."


def _first_difference(actual, expected, path="$"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return path
        for key in sorted(expected):
            child = f"{path}.{key}" if path != "$" else key
            if key not in actual:
                return child
            difference = _first_difference(actual[key], expected[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return path
        for index, (found, wanted) in enumerate(zip(actual, expected)):
            difference = _first_difference(found, wanted, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if actual != expected or (type(actual) is bool) != (type(expected) is bool):
        return path
    return None


def admission_digest(source, runtime_closure, sdk, policy_version, grants, smoke_receipt):
    parts = [source, runtime_closure, sdk, policy_version, grants, smoke_receipt]
    return "sha256:" + hashlib.sha256(_canonical(parts)).hexdigest()


def admit(staged, executor, *, runtime_closure_digest, sdk_digest, policy_version, grants_digest,
          call=None, fixtures=None, timeout_retries=0, wall_timeout_s=None):
    validation = validate(staged, executor, call=call,
                          timeout_retries=timeout_retries, wall_timeout_s=wall_timeout_s)
    if not validation.ok:
        errors = validation.result["errors"][:5]
        locations = []
        for error in errors:
            if not isinstance(error, dict):
                continue
            filename = error.get("file", "?")
            line = error.get("line", 0)
            col = error.get("col", 0)
            message = error.get("msg", "invalid source")
            locations.append(f"{filename}:{line}:{col}: {message}")
        raise BuildError("STATIC", "; ".join(locations) or "validator returned ok=false")
    receipt = smoke(staged, executor, fixtures=fixtures, call=call,
                    timeout_retries=timeout_retries, wall_timeout_s=wall_timeout_s)
    digest = admission_digest(staged.source_digest, runtime_closure_digest, sdk_digest,
                              policy_version, grants_digest, receipt.digest)
    return Admission(digest, staged.source_digest, validation, receipt)
