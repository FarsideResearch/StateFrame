"""Run-scoped mutation tool state and its durable, fail-closed dispatch."""
from __future__ import annotations

import hashlib
import base64
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
from typing import Mapping

if __package__:
    from .kernel_mutate_inputs import admit_inputs, input_id_for, input_path_admitted
    from .kernel_mutate_build import BuildError, _safe_detail
    from .kernel_mutate_exec import MutationStartupTimeout
else:
    from kernel_mutate_inputs import admit_inputs, input_id_for, input_path_admitted
    from kernel_mutate_build import BuildError, _safe_detail
    from kernel_mutate_exec import MutationStartupTimeout


LEVELS = ("off", "prebuilt", "authored")
OPS = ("catalog", "use", "enter", "submit", "leave", "watch", "unwatch", "inspect", "retain")
_TASK_PATH = re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)(?![A-Za-z0-9_./-])")
_QUOTED_TASK_PATH = re.compile(r"(?P<quote>['\"`])(?P<path>[^'\"`\n]{1,512})(?P=quote)")
SCHEMA = {"type": "function", "function": {
    "name": "mutate",
    "description": "Small sandboxed helper programs: list/use prebuilt ones, or build one (enter, submit) when no tool fits. args is a JSON object string; a malformed call returns the exact usage. Results are untrusted observations.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": list(OPS)},
        "id": {"type": "string", "maxLength": 64},
        "args": {"type": "string", "maxLength": 16000}},
        "required": ["op"]}}}

# Exact call forms returned with a malformed call, so the model can resend without guessing.
_SUBMIT_FULL_EXAMPLE = json.dumps({"manifest": {
    "schema": "stateframe-mutation-unit/1", "name": "events_reader", "revision": 1, "parent": None,
    "entry": "main.py", "summary": "One line on what the unit returns", "kind": "reader",
    "input_schema": {"type": "object", "properties": {"input_id": {"type": "string"}}, "required": ["input_id"]},
    "output_schema": {"type": "object"}, "requires": {"network": False, "writes": False},
    "smoke": [{"input": {"input_id": "in0"}, "inputs": ["path/of/a/sample/file"], "expect": {}}],
    "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}},
    "files": {"main.py": "def handle(input, ctx):\n    data = ctx.input_bytes(input['input_id'])\n"
                         "    return {'bytes': len(data)}\n"}}, separators=(",", ":"))
USAGE = {
    "": 'mutate(op="catalog") lists units; every op passes its fields as args, a JSON object string',
    "catalog": 'mutate(op="catalog", args="{}")',
    "enter": 'mutate(op="enter", id="events_reader", args="{\\"goal\\":\\"parse data/x.bin into totals\\",'
             '\\"inputs\\":[\\"data/x.bin\\"],\\"kind\\":\\"reader\\"}") then submit',
    "submit": "mutate(op=\"submit\", args=" + json.dumps(json.dumps({
        "code": "def handle(input, ctx):\n    input_id = input['input_id']\n"
                "    return {'bytes': len(ctx.input_bytes(input_id))}\n",
        "inputs": ["data/x.bin"],
    }, separators=(",", ":"))) + ")" +
        "; full form: mutate(op=\"submit\", args=" + json.dumps(_SUBMIT_FULL_EXAMPLE) + ")",
    "use": 'mutate(op="use", id="UNIT", args="{\\"input\\":{\\"input_id\\":\\"in0\\"},'
           '\\"inputs\\":[\\"data/x.bin\\"]}"): inputs[i] is mounted as in<i>',
    "watch": 'mutate(op="watch", id="UNIT", args="{\\"input\\":{\\"input_id\\":\\"in0\\"},'
             '\\"inputs\\":[\\"logs/app.log\\"],\\"every_turns\\":1,\\"max_ticks\\":5}")',
    "unwatch": 'mutate(op="unwatch", id="WATCH")',
    "retain": 'mutate(op="retain", id="UNIT")',
    "inspect": 'mutate(op="inspect")',
    "leave": 'mutate(op="leave")',
}

_active: MutationRuntime | None = None
_UNIT_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")


def schema_for_level(level: str) -> dict:
    if level not in LEVELS or level == "off":
        raise ValueError("mutation tool requires prebuilt or authored level")
    return json.loads(json.dumps(SCHEMA))


def install(runtime: MutationRuntime | None) -> None:
    global _active
    _active = runtime


def service() -> MutationRuntime | None:
    return _active


class MutationUnavailable(ValueError):
    """The host could not qualify the mutation sandbox; the run continues without mutations."""

    code = "MUTATION_PLATFORM_UNQUALIFIED"


def _refuse(code: str, reason: str) -> str:
    return f"REFUSED MUTATION_{code} {reason}"


def _admission_refusal(error: Exception) -> str:
    code = error.code if isinstance(error, BuildError) else "INVALID"
    detail = error.detail if isinstance(error, BuildError) else _safe_detail(str(error))
    return _refuse("ADMISSION", f"{code}: {detail}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 40:
        raise ValueError("JSON integer is too long")
    return int(value)


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _loads(value: str | bytes) -> object:
    return json.loads(value, object_pairs_hook=_object_pairs, parse_constant=_reject_constant,
                      parse_int=_bounded_int, parse_float=_finite_float)


def _task_input_paths(task: object) -> tuple[str, ...]:
    if not isinstance(task, str):
        return ()
    source = task[:8192]
    paths: list[str] = []
    quoted_spans: list[tuple[int, int]] = []
    for match in _QUOTED_TASK_PATH.finditer(source):
        quoted_spans.append(match.span())
        candidate = match.group("path").strip()
        if _looks_like_input_path(candidate) and candidate not in paths:
            paths.append(candidate)
    for match in _TASK_PATH.finditer(source):
        if any(start <= match.start() < end for start, end in quoted_spans):
            continue
        candidate = match.group(1).rstrip(".,;:!?")
        if _looks_like_input_path(candidate) and candidate not in paths:
            paths.append(candidate)
    return tuple(paths)


def _looks_like_input_path(value: str) -> bool:
    if not value or len(value.encode("utf-8", "ignore")) > 512:
        return False
    name = value.rsplit("/", 1)[-1]
    suffix = name.rpartition(".")[2]
    return bool(suffix and suffix.isascii() and suffix.isalnum())


def _proposal_input(manifest: Mapping[str, object], file_format: str) -> dict[str, object] | None:
    schema = manifest.get("input_schema")
    if not isinstance(schema, Mapping):
        return None
    required = schema.get("required")
    properties = schema.get("properties")
    if (not isinstance(required, list) or not required
            or not isinstance(properties, Mapping)):
        return None
    file_ids = [key for key in required if isinstance(key, str) and key.endswith("_id")]
    if len(file_ids) != 1:
        return None
    smoke = manifest.get("smoke")
    if not isinstance(smoke, list):
        return None
    sample = next((row["input"] for row in smoke
                   if isinstance(row, Mapping) and isinstance(row.get("input"), Mapping)
                   and isinstance(row.get("inputs"), list) and len(row["inputs"]) == 1
                   and row["input"].get(file_ids[0]) == "in0"), None)
    if not isinstance(sample, Mapping):
        return None
    result: dict[str, object] = {}
    for key in required:
        if not isinstance(key, str) or key not in properties:
            return None
        if key == file_ids[0]:
            result[key] = input_id_for(0)
        elif key == "format":
            result[key] = file_format
        elif key in sample:
            result[key] = sample[key]
        else:
            return None
    if __package__:
        from .kernel_mutate_wire import validate_schema
    else:
        from kernel_mutate_wire import validate_schema
    try:
        validate_schema(result, schema)
    except (TypeError, ValueError):
        return None
    return result


def _read_json_file(path: Path, limit: int) -> dict[str, object]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("catalog file is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            document = _loads(source.read(limit + 1))
    finally:
        os.close(fd)
    if not isinstance(document, dict):
        raise ValueError("catalog document must be an object")
    return document


def _remove_snapshot_root(root: Path) -> None:
    for path in root.iterdir():
        path.unlink()
    root.rmdir()


def _remove_smoke_snapshots(fixtures: list[dict]) -> None:
    roots = {path.parent for fixture in fixtures for path in fixture["inputs"].values()}
    for root in roots:
        _remove_snapshot_root(root)


class MutationBudgetError(ValueError):
    pass


class DigestMismatch(ValueError):
    pass


class MutationRuntime:
    """The model-visible state; physical launches remain delegated to the W1/W2 APIs."""

    def __init__(self, *, level: str, journal: object, run_dir: Path | None = None,
                 policy: object | None = None, executor_factory: object | None = None,
                 catalog_root: Path | None = None, replay_external_allowed: bool = False,
                 repo: Path | None = None, runtime_root: Path | None = None,
                 edit_scope: set[str] | None = None, protected_paths: set[str] | None = None,
                 effect_row: Mapping[str, object] | None = None,
                 ordinary_read_globs: tuple[str, ...] | None = None):
        if level not in LEVELS or level == "off":
            raise ValueError("mutations must be enabled before creating a runtime")
        self.level = level
        self.journal = journal
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.policy = policy
        self.executor_factory = executor_factory
        self.repo = Path(repo) if repo is not None else (self.run_dir / "workspace" if self.run_dir else None)
        self.runtime_root = Path(runtime_root) if runtime_root is not None else None
        self.edit_scope = frozenset(edit_scope) if edit_scope is not None else None
        self.protected_paths = frozenset(protected_paths or ())
        self.effect_row = effect_row
        self.ordinary_read_globs = ordinary_read_globs
        self._edits = None
        self._hooks = None
        self.hook_lines: list[str] = []
        self.catalog_root = (Path(catalog_root) if catalog_root is not None else
                             Path(__file__).resolve().parents[1] / "mutations_prebuilt")
        self.replay_external_allowed = replay_external_allowed
        self.executor = None
        self._reopen_runtime_root = False
        self._platform_error: str | None = None
        self.episode: dict[str, object] | None = None
        self.ready: dict[str, object] = {}
        self.watches: dict[str, object] = {}
        self.last = "none"
        if self.run_dir is not None and hasattr(journal, "events"):
            active_watches = set()
            for row in journal.events():
                if row.get("kind") == "watch_start":
                    active_watches.add(row.get("watch"))
                elif row.get("kind") == "watch_stop":
                    active_watches.discard(row.get("watch"))
            for watch_id in sorted(item for item in active_watches if isinstance(item, str)):
                journal.append("watch_stop", watch=watch_id, reason="resume")
        self.episodes_started = sum(1 for row in journal.events()
                                    if row.get("kind") == "episode_enter") if hasattr(journal, "events") else 0

    def _ensure_executor(self):
        if self.executor is None:
            if self._platform_error is not None:
                raise MutationUnavailable(self._platform_error)
            if self._reopen_runtime_root:
                # A timed-out hook retired and swept the previous executor. Reopen
                # only its own runtime root, and fail if another owner took the path.
                self.runtime_root.mkdir(mode=0o700)
            if callable(self.executor_factory):
                if __package__:
                    from .kernel_mutate_exec import MutationPlatformError
                else:
                    from kernel_mutate_exec import MutationPlatformError
                try:
                    self.executor = self.executor_factory()
                except MutationPlatformError as error:
                    # Qualify once per run: later calls refuse with the same reason, no new ldd/selftest.
                    self._platform_error = _safe_detail(str(error))[:600]
                    raise MutationUnavailable(self._platform_error) from error
            elif self.runtime_root is not None:
                if __package__:
                    from .kernel_mutate_exec import MutationExecutor, MutationPlatformError
                else:
                    from kernel_mutate_exec import MutationExecutor, MutationPlatformError
                try:
                    self.executor = MutationExecutor(self.runtime_root, network_envelope=(
                        "full" if getattr(self.policy, "network", "none") == "public-web" else "none"))
                except MutationPlatformError as error:
                    self._platform_error = _safe_detail(str(error))[:600]
                    raise MutationUnavailable(self._platform_error) from error
            else:
                raise ValueError("qualified mutation executor is unavailable")
            if self.runtime_root is None:
                self.runtime_root = self.executor.root
            self._reopen_runtime_root = False
        return self.executor

    def expire_episode(self) -> str | None:
        """Return control to the parent after the last authoring send is answered."""
        episode = self.episode
        if episode is None or int(episode.get("sends_left", 0)) > 0:
            return None
        return self.fail_episode("authoring sends exhausted")

    def fail_episode(self, reason: str) -> str:
        """Close an episode with one durable parent-visible failure receipt."""
        episode = self.episode
        if episode is None:
            return self.last
        receipt = f"mutation episode {episode['id']} FAILED {reason}"
        self.journal.append("episode_failed", episode=episode["id"], reason=reason)
        self.episode = None
        self.last = receipt
        return receipt

    def render_observation(self, *, unit: str, revision: str, result: dict,
                           source_digest: str = "sha256:" + "0" * 64,
                           input_value: object = None) -> str:
        content = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        input_digest = hashlib.sha256(json.dumps(input_value, sort_keys=True,
                                                default=str).encode()).hexdigest()[:12]
        return (f"[untrusted mutation observation {unit} {revision} {source_digest[:19]} "
                f"in:sha256:{input_digest}]\n{content}")

    def _prebuilt(self, name: str):
        if __package__:
            from .kernel_mutate_build import source_digest
        else:
            from kernel_mutate_build import source_digest

        if _UNIT_NAME.fullmatch(name) is None:
            raise ValueError("invalid unit name")
        catalog = _read_json_file(self.catalog_root / "manifest.json", 64 << 10)
        expected = catalog.get("units", {}).get(name)
        if catalog.get("schema") != "stateframe-mutation-prebuilt/1" or not isinstance(expected, str):
            raise ValueError("unknown prebuilt unit")
        directory = self.catalog_root / name
        manifest = _read_json_file(directory / "unit.json", 8 << 10)
        files = {}
        for path in directory.iterdir():
            if path.suffix == ".py":
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 64 << 10:
                        raise ValueError("invalid prebuilt module")
                    files[path.name] = os.read(fd, (64 << 10) + 1)
                finally:
                    os.close(fd)
        digest = source_digest(files, manifest)
        if digest != expected:
            raise DigestMismatch("prebuilt manifest digest does not match source")
        return manifest, files, directory

    def _requirements_fit(self, requires: object) -> bool:
        if not isinstance(requires, Mapping):
            return False
        network, writes = requires.get("network"), requires.get("writes")
        if type(network) is not bool or type(writes) is not bool:
            return False
        return ((not network or getattr(self.policy, "network", "none") == "public-web")
                and (not writes or bool(getattr(self.policy, "write_globs", ()))))

    def _record_prebuilt_proposal(self, proposal: Mapping[str, object]) -> bool:
        events = getattr(self.journal, "events", None)
        append = getattr(self.journal, "append", None)
        if not callable(events) or not callable(append):
            return False
        unit = proposal["unit"]
        trigger_file = proposal["trigger_file"]
        input_id = proposal["input_id"]
        try:
            if any(row.get("kind") == "prebuilt_proposed"
                   and row.get("unit") == unit
                   and row.get("trigger_file") == trigger_file
                   and row.get("input_id") == input_id
                   for row in events() if isinstance(row, Mapping)):
                return True
            append("prebuilt_proposed", unit=unit, trigger_file=trigger_file, input_id=input_id)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def propose_prebuilt(self, task: str) -> dict[str, object] | None:
        """Select one schema-callable reader only from an exact task-file extension match."""
        if self.level not in ("prebuilt", "authored") or self.repo is None:
            return None
        paths = _task_input_paths(task)
        if not paths:
            return None
        try:
            catalog = _read_json_file(self.catalog_root / "manifest.json", 64 << 10)
            units = catalog.get("units")
            if (catalog.get("schema") != "stateframe-mutation-prebuilt/1"
                    or not isinstance(units, dict)
                    or any(not isinstance(name, str) for name in units)):
                return None
            readers = []
            for name in sorted(units):
                manifest, _files, _fixture_root = self._prebuilt(name)
                if manifest.get("name") != name:
                    return None
                formats = manifest.get("formats", [])
                if manifest.get("kind") == "reader" and formats:
                    readers.append((manifest, formats))
        except (OSError, TypeError, ValueError):
            # A bad catalog or digest makes uniqueness impossible to establish.
            return None

        matches: list[tuple[str, Mapping[str, object], str, dict[str, object], str]] = []
        for path in paths:
            if not input_path_admitted(self.repo, path, self.policy, self.ordinary_read_globs):
                continue
            filename = path.rsplit("/", 1)[-1].casefold()
            path_matches = []
            for reader, formats in readers:
                matching_formats = [fmt for fmt in formats
                                    if filename.endswith("." + fmt.casefold())]
                if matching_formats:
                    path_matches.append((reader, max(matching_formats, key=len)))
            if len(path_matches) > 1:
                return None
            if not path_matches:
                continue
            manifest, file_format = path_matches[0]
            if not self._requirements_fit(manifest.get("requires")):
                continue
            input_value = _proposal_input(manifest, file_format)
            summary = manifest.get("summary")
            if not input_value or not isinstance(summary, str):
                continue
            description = next((line.strip() for line in summary.splitlines() if line.strip()), "")[:160]
            if description:
                matches.append((path, manifest, file_format, input_value, description))
        if len(matches) != 1:
            return None
        path, manifest, file_format, input_value, description = matches[0]
        args = {"input": input_value, "inputs": [path]}
        if len(json.dumps(args, ensure_ascii=True, separators=(",", ":")).encode("utf-8")) > 768:
            return None
        proposal = {"unit": manifest["name"], "summary": description,
                    "trigger_file": path, "input_id": input_id_for(0),
                    "format": file_format, "input_schema": manifest["input_schema"],
                    "args": args}
        return proposal if self._record_prebuilt_proposal(proposal) else None

    def _launch(self, spec):
        reservations = self._check_launch_budget(spec.role, spec.limits)
        token = spec.launch
        for name, amount in reservations:
            self.journal.append("BUDGET_RESERVE", name=name, amount=amount,
                                id=token + ":" + name)
        try:
            outcome = self._ensure_executor().run(spec)
        except MutationStartupTimeout:
            self.retire_hook_executor()
            raise
        meter = "broker" if spec.role == "broker" else "payload"
        for name, measured in ((meter + "_wall_s", outcome.wall_s),
                               (meter + "_cpu_s", outcome.cpu_s)):
            if measured is not None:
                self.journal.append("BUDGET_SETTLE", id=token + ":" + name,
                                    amount=max(0, measured))
        return outcome

    def _check_launch_budget(self, role, limits):
        if __package__:
            from .kernel_mutate_policy import Budget
        else:
            from kernel_mutate_policy import Budget

        state = Budget().fold(self.journal.events())
        reservations = (("invocations", 1), ("broker_fetches", 1),
                        ("broker_wall_s", limits.wall_s),
                        ("broker_cpu_s", limits.cpu_s)) if role == "broker" else (
            ("invocations", 1), ("payload_wall_s", limits.wall_s),
            ("payload_cpu_s", limits.cpu_s))
        if state.disabled or any(amount > state.remaining()[name] for name, amount in reservations):
            name = next((name for name, amount in reservations
                         if amount > state.remaining()[name]), reservations[0][0])
            self.journal.append("BUDGET_EXHAUSTED", name=name)
            raise MutationBudgetError(name)
        return reservations

    def _broker_fetch(self, request: dict, call_id: str):
        if __package__:
            from .kernel_mutate_broker import parse_broker
        else:
            from kernel_mutate_broker import parse_broker
        if __package__:
            from .kernel_mutate_exec import LaunchSpec, MutationLimits, MutationContainmentError
        else:
            from kernel_mutate_exec import LaunchSpec, MutationLimits, MutationContainmentError
        if __package__:
            from .kernel_mutate_policy import Budget
        else:
            from kernel_mutate_policy import Budget
        if __package__:
            from .kernel_mutate_wire import request_key
        else:
            from kernel_mutate_wire import request_key

        remaining = Budget().fold(self.journal.events()).remaining()["broker_bytes"]
        if remaining < 1:
            raise MutationBudgetError("broker_bytes")
        max_bytes = min(int(remaining), 1 << 20)
        launch = uuid.uuid4().hex
        key = request_key(request["url"], request["accept"])
        self.journal.append("BUDGET_RESERVE", name="broker_bytes", amount=max_bytes,
                            id=launch + ":broker_bytes")
        self.journal.append("BROKER_INTENT", call=call_id, key=key, method="GET",
                            url=request["url"], accept=request["accept"])
        broker_request = {"url": request["url"], "accept": request["accept"] or None,
                          "max_bytes": max_bytes, "timeout_s": 10, "domains": (
                              list(self.policy.domains) if getattr(self.policy, "domains", ()) else None)}
        spec = LaunchSpec("broker", call_id, launch, None, {}, {}, json.dumps({
            "role": "broker", "call": call_id, "launch": launch,
            "request": broker_request}, separators=(",", ":")).encode(),
            MutationLimits(wall_s=12, cpu_s=5, output_bytes=max_bytes + (8 << 10) + 64))
        outcome = self._launch(spec)
        if not outcome.cleanup.ok:
            raise MutationContainmentError("broker cleanup is unverified")
        if outcome.exit_code != 0 or outcome.timed_out or outcome.stdout_overflow:
            raise ValueError("broker did not complete cleanly")
        metadata, body = parse_broker(outcome.stdout, max_bytes=max_bytes)
        digest = self.journal.put_blob(body)
        self.journal.append("BUDGET_SETTLE", id=launch + ":broker_bytes", amount=len(body))
        status = metadata["status"] if metadata["status"] is not None else 0
        headers = metadata["headers"]
        final_url = metadata["final_url"] or request["url"]
        error = metadata["error"]
        self.journal.append("BROKER_COMPLETE", call=call_id, key=key, status=status,
                            headers=headers, final_url=final_url, body_digest=digest,
                            bytes=len(body), error=error, redirects=metadata["redirects"])
        response = {"key": key, "method": "GET", "url": request["url"],
                    "accept": request["accept"], "status": status, "headers": headers,
                    "final_url": final_url, "body": "/resp/" + digest,
                    "bytes": len(body), "error": error}
        return response, self.journal.evidence / digest

    def _use(self, name: str, args: dict[str, object], call_id: str, *, snapshots=None,
             record_evidence: bool = True) -> str:
        if __package__:
            from .kernel_mutate_build import stage_unit, admit
        else:
            from kernel_mutate_build import stage_unit, admit
        if __package__:
            from .kernel_mutate_exec import LaunchSpec, MutationLimits, MutationContainmentError
        else:
            from kernel_mutate_exec import LaunchSpec, MutationLimits, MutationContainmentError
        if __package__:
            from .kernel_mutate_wire import build_stdin_doc, parse_stdout, validate_schema
        else:
            from kernel_mutate_wire import build_stdin_doc, parse_stdout, validate_schema
        if __package__:
            from .kernel_mutate_qualify import _binding
        else:
            from kernel_mutate_qualify import _binding

        if self.repo is None:
            return _refuse("SCOPE", "workspace is unavailable")
        if name in self.ready:
            prepared = self.ready[name]
            manifest, files, fixture_root = prepared["manifest"], prepared["files"], prepared.get("fixture_root")
        else:
            manifest, files, fixture_root = self._prebuilt(name)
        requires = manifest["requires"]
        if not self._requirements_fit(requires):
            if requires["network"] and getattr(self.policy, "network", "none") != "public-web":
                return _refuse("GRANT", "network access is not granted")
            if requires["writes"] and not getattr(self.policy, "write_globs", ()):
                return _refuse("GRANT", "writes are not granted")
            return _refuse("GRANT", "unit requirements are not granted")
        if set(args) - {"input", "inputs"} or not isinstance(args.get("input"), dict):
            return _refuse("BAD_ARGUMENTS", "use requires input and optional inputs")
        validate_schema(args["input"], manifest["input_schema"])
        # An authored unit used without inputs reads the repo files it was smoke-tested on.
        default_inputs = [path for row in manifest.get("smoke", ()) if isinstance(row, dict)
                          for path in row.get("inputs", ()) if isinstance(path, str)
                          and not path.startswith("fixtures/")][:16]
        requests = args.get("inputs", default_inputs)
        if not isinstance(requests, list):
            return _refuse("BAD_ARGUMENTS", "inputs must be a list")
        limits = manifest["limits"]
        launch_limits = MutationLimits(limits["wall_s"], limits["cpu_s"], int(limits["mem_mb"]))
        self._check_launch_budget("payload", launch_limits)
        executor = self._ensure_executor()
        staged = stage_unit(files, manifest, executor.root, fixture_root=fixture_root)
        binding = _binding()
        grants = hashlib.sha256(json.dumps(self.policy.__dict__, sort_keys=True).encode()).hexdigest()
        fixtures = (self._admit_smoke_fixtures(manifest, executor.root)
                    if name in self.ready else None)
        try:
            admission = admit(staged, self, runtime_closure_digest=binding["closure_digest"],
                              sdk_digest=binding["sdk_digest"], policy_version="mutation/1",
                              grants_digest=grants, call=call_id, fixtures=fixtures)
        finally:
            if fixtures is not None:
                _remove_smoke_snapshots(fixtures)
        if record_evidence:
            self.journal.append("ADMISSION", call=call_id, admission_digest=admission.admission_digest,
                                source_digest=staged.source_digest)
        snapshot_root = None
        if snapshots is None:
            snapshot_root = executor.root / "in" / uuid.uuid4().hex
            snapshots = admit_inputs(self.repo, requests, self.policy,
                                     snapshot_root=snapshot_root,
                                     ordinary_read_globs=self.ordinary_read_globs)
        try:
            payload = dict(args["input"])
            if manifest["name"] == "sqlite_peek" and "in0" in snapshots.facts:
                payload["wal_present"] = snapshots.facts["in0"]["wal_present"]
            call = call_id
            responses = []
            response_files = {}
            for phase in range(4):
                launch = uuid.uuid4().hex
                doc = build_stdin_doc(wire=1, call=call, launch=launch, phase=phase,
                                      role="payload", input=payload,
                                      inputs=list(snapshots.descriptors), responses=responses,
                                      grants=requires, max_fetch=4 if requires["network"] else 0)
                spec = LaunchSpec("payload", call, launch, staged.dir, snapshots.files,
                                  response_files, doc, launch_limits)
                outcome = self._launch(spec)
                if not outcome.cleanup.ok:
                    raise MutationContainmentError("payload cleanup is unverified")
                if outcome.exit_code != 0 or outcome.timed_out or outcome.stdout_overflow:
                    if __package__:
                        from .kernel_mutate_build import execution_failure_detail
                    else:
                        from kernel_mutate_build import execution_failure_detail
                    return _refuse("EXECUTION", _safe_detail(
                        execution_failure_detail(outcome, "payload", call) or "payload did not complete"))
                frame = parse_stdout(outcome.stdout, call=call,
                                     output_schema=manifest["output_schema"], phase=phase,
                                     recorded_keys=[row["key"] for row in responses])
                if frame.kind != "need":
                    break
                if not requires["network"]:
                    return _refuse("GRANT", "unit requested an ungranted fetch")
                for request in frame.fetch:
                    response, path = self._broker_fetch(request, call_id)
                    responses.append(response)
                    response_files[path.name] = path
            else:
                return _refuse("PROTOCOL", "payload exceeded phase limit")
            if frame.kind != "result":
                return _refuse("PROTOCOL", "payload did not return a terminal result")
            if record_evidence:
                self.journal.append("TERMINAL", call=call_id, frame_b64=base64.b64encode(outcome.stdout).decode(),
                                    exit_code=0, stdout_overflow=False, timed_out=False,
                                    output_schema=manifest["output_schema"])
                self.journal.append("CLEANUP_VERIFIED", call=call_id)
            if frame.edits:
                if not requires["writes"] or not getattr(self.policy, "write_globs", ()):
                    return _refuse("GRANT", "edit proposals are not granted")
                try:
                    self.apply_edit_proposals(call_id, list(frame.edits), cleanup_ok=outcome.cleanup.ok)
                except (ValueError, OSError) as error:
                    return _refuse("EDIT", str(error)[:160])
            reply = json.dumps(frame.result, sort_keys=True, ensure_ascii=False)
            if len(reply.encode()) > 64 << 10:
                return _refuse("BUDGET", "observation output exceeds limit")
            return self.render_observation(unit=name, revision=f"r{manifest['revision']}",
                                           result=frame.result, source_digest=staged.source_digest,
                                           input_value=args)
        finally:
            if snapshot_root is not None:
                _remove_snapshot_root(snapshot_root)

    def _submit(self, args: dict[str, object], call_id: str) -> str:
        if __package__:
            from .kernel_mutate_build import stage_unit, admit
        else:
            from kernel_mutate_build import stage_unit, admit
        if __package__:
            from .kernel_mutate_policy import Budget
        else:
            from kernel_mutate_policy import Budget
        if __package__:
            from .kernel_mutate_qualify import _binding
        else:
            from kernel_mutate_qualify import _binding

        if self.episode is None or self.level != "authored":
            return _refuse("FRAME", "submit requires an authoring episode")
        if "code" in args and not ({"manifest", "files"} & args.keys()):
            if (set(args) - {"code", "summary", "input_schema", "output_schema", "expect", "inputs"}
                    or not isinstance(args["code"], str)
                    or ("summary" in args and not isinstance(args["summary"], str))
                    or ("input_schema" in args and not isinstance(args["input_schema"], dict))
                    or ("output_schema" in args and not isinstance(args["output_schema"], dict))
                    or ("expect" in args and not isinstance(args["expect"], dict))
                    or ("inputs" in args and (not isinstance(args["inputs"], list) or len(args["inputs"]) > 16
                                              or any(not isinstance(item, str) or not item or len(item) > 256
                                                     for item in args["inputs"])))):
                return _refuse("BAD_ARGUMENTS", "invalid short-form submit")
            episode = self.episode
            name = episode.get("unit_id") or "unit"
            previous = self.ready.get(name)
            revision = previous["manifest"]["revision"] + 1 if previous else 1
            # Smoke on the episode's admitted inputs, or on inputs named with this submit.
            inputs = list(args.get("inputs") or episode.get("input_ids", ()))
            needs = episode.get("needs", {})
            args = {"manifest": {
                "schema": "stateframe-mutation-unit/1", "name": name,
                "revision": revision, "parent": previous["source_digest"] if previous else None,
                "entry": "main.py", "summary": args.get("summary", episode["goal"]),
                "kind": episode.get("kind", "reader"),
                "input_schema": args.get("input_schema", {"type": "object"}),
                "output_schema": args.get("output_schema", {"type": "object"}),
                "requires": {"network": bool(needs.get("network", False)),
                             "writes": bool(needs.get("writes", False))},
                "smoke": [{"input": {"input_id": "in0"} if inputs else {},
                           "inputs": inputs, "expect": args.get("expect", {})}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}},
                "files": {"main.py": args["code"]}}
        if set(args) != {"manifest", "files"} or not isinstance(args["manifest"], dict) or not isinstance(args["files"], dict):
            return _refuse("BAD_ARGUMENTS", "submit requires code (short form) or manifest and files")
        manifest = args["manifest"]
        try:
            files = {name: source.encode("utf-8", "strict") for name, source in args["files"].items()
                     if isinstance(name, str) and isinstance(source, str)}
        except UnicodeError as error:
            return _refuse("BAD_ARGUMENTS", str(error)[:160])
        if len(files) != len(args["files"]):
            return _refuse("BAD_ARGUMENTS", "files must contain UTF-8 source strings")
        source_bytes = sum(map(len, files.values()))
        state = Budget().fold(self.journal.events())
        name = manifest.get("name")
        if not isinstance(name, str) or _UNIT_NAME.fullmatch(name) is None:
            return _refuse("BAD_ARGUMENTS", "invalid unit name")
        charges = {"authored_source_bytes": source_bytes, "revisions_total": 1,
                   "revisions_per_unit": 1}
        exhausted = next((key for key, amount in charges.items()
                          if amount > state.remaining(scope=name)[key]), None)
        if state.disabled or exhausted is not None:
            self.journal.append("BUDGET_EXHAUSTED", name=exhausted or "revisions_total",
                                **({"scope": name} if exhausted == "revisions_per_unit" else {}))
            return _refuse("BUDGET", "revision or source budget exhausted")
        previous = self.ready.get(name)
        if previous is None:
            if manifest.get("revision") != 1 or manifest.get("parent") is not None:
                return _refuse("PARENT", "first revision must have no parent")
        elif (manifest.get("parent") != previous["source_digest"] or
              manifest.get("revision") != previous["manifest"]["revision"] + 1):
            return _refuse("PARENT", "repair must name the current source digest")
        requires = manifest.get("requires")
        if (not isinstance(requires, dict) or set(requires) != {"network", "writes"}
                or any(type(value) is not bool for value in requires.values())):
            return self._with_repair(_admission_refusal(BuildError(
                "REQUIRES", "requires: needs boolean network and writes")))
        if requires["network"] and getattr(self.policy, "network", "none") != "public-web":
            return _refuse("GRANT", "network access is not granted")
        if requires["writes"] and not getattr(self.policy, "write_globs", ()):
            return _refuse("GRANT", "writes are not granted")
        executor = self._ensure_executor()
        fixtures = self._admit_smoke_fixtures(manifest, executor.root)
        self.journal.append("BUDGET_ADMISSION", id=call_id, scope=name, charges=charges)
        admitted = False
        try:
            staged = stage_unit(files, manifest, executor.root)
            binding = _binding()
            grants = hashlib.sha256(json.dumps(self.policy.__dict__, sort_keys=True).encode()).hexdigest()
            admission = admit(staged, self, runtime_closure_digest=binding["closure_digest"],
                              sdk_digest=binding["sdk_digest"], policy_version="mutation/1",
                              grants_digest=grants, call=call_id, fixtures=fixtures)
            admitted = True
        except BuildError as error:
            refusal = self._with_repair(_admission_refusal(error))
            self.episode["diagnostic"] = error.detail
            return refusal
        finally:
            if not admitted:
                _remove_smoke_snapshots(fixtures)
        source_blob = self.journal.put_blob(json.dumps({"manifest": manifest,
            "files": {key: base64.b64encode(value).decode() for key, value in files.items()}},
            sort_keys=True).encode())
        self.journal.append("ADMISSION", call=call_id, source_digest=staged.source_digest,
                            admission_digest=admission.admission_digest, blob=source_blob,
                            name=name, revision=manifest["revision"])
        self.ready[name] = {"manifest": manifest, "files": files,
                            "source_digest": staged.source_digest}
        episode_id = self.episode["id"]
        self.episode = None
        return f"mutation {name} r{manifest['revision']} READY smoke {len(admission.smoke.results)}/{len(manifest['smoke'])} [{episode_id}]"

    def _admit_smoke_fixtures(self, manifest: dict, runtime_root: Path) -> list[dict]:
        """Snapshot every authored smoke input with the same grants used by use."""
        rows = manifest.get("smoke")
        if not isinstance(rows, list):
            raise ValueError("smoke must be a list")
        requests = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("inputs", []), list):
                raise ValueError("smoke inputs must be path lists")
            if any(not isinstance(path, str) for path in row.get("inputs", [])):
                raise ValueError("smoke inputs must be path strings")
            requests.extend(row.get("inputs", []))
        if len(requests) > 16:
            raise ValueError("at most sixteen smoke input files are allowed")
        if requests and self.repo is None:
            raise ValueError("workspace is unavailable for smoke inputs")
        admitted = []
        roots = []
        total = 0
        try:
            for row in rows:
                paths = row.get("inputs", [])
                fixture = dict(row)
                if paths:
                    root = runtime_root / "in" / uuid.uuid4().hex
                    snapshot = admit_inputs(self.repo, paths, self.policy, snapshot_root=root,
                                            ordinary_read_globs=self.ordinary_read_globs)
                    roots.append(root)
                    total += sum(path.stat().st_size for path in snapshot.files.values())
                    if total > 16 << 20:
                        raise ValueError("smoke input set exceeds the size cap")
                    fixture["inputs"] = snapshot.files
                else:
                    fixture["inputs"] = {}
                admitted.append(fixture)
            return admitted
        except (OSError, ValueError):
            for root in roots:
                for path in root.iterdir():
                    path.unlink()
                root.rmdir()
            raise

    def run(self, spec):
        """Meter validator and smoke launches through the same durable reservation path."""
        return self._launch(spec)

    def _edit_manager(self):
        if self._edits is None:
            if __package__:
                from .kernel_mutate_edits import MutationEdits
            else:
                from kernel_mutate_edits import MutationEdits
            self._edits = MutationEdits(self)
        return self._edits

    def _hook_manager(self):
        if self._hooks is None:
            if __package__:
                from .kernel_mutate_hooks import MutationHooks
            else:
                from kernel_mutate_hooks import MutationHooks
            self._hooks = MutationHooks(self)
        return self._hooks

    def attach_hook(self, unit: dict | str, *, event: str) -> None:
        if self._platform_error is not None and event in ("post_tool", "session_end"):
            name = unit if isinstance(unit, str) else unit.get("name") if isinstance(unit, dict) else None
            if isinstance(name, str):
                self._hook_manager().refuse_platform(name, event, None, self._platform_error)
                return
        if isinstance(unit, str):
            prepared = self.ready.get(unit)
            if prepared is None:
                raise ValueError("mutation hook unit is not admitted")
            unit = {**prepared["manifest"],
                    "files": dict(prepared["files"])}
        elif self.run_dir is not None:
            raise ValueError("mutation hook requires a retained or admitted unit")
        self._hook_manager().attach(unit, event)

    def invoke_hook(self, *, event: str, facts: dict):
        return self._hook_manager().invoke(event, facts)

    def disable_hook(self, name: str, *, reason: str) -> None:
        self._hook_manager().disable(name, reason)

    def hook_failures(self, name: str) -> int:
        return self._hook_manager().failures(name)

    def register_powerup_executor(self, name: str, *, supported: bool) -> None:
        self._powerup_executor_supported = name == "mutation" and supported

    def promote_unit(self, name: str, destination: Path):
        from types import SimpleNamespace
        from stateframe.mutations_cli import _promote

        if getattr(self, "_powerup_executor_supported", False) is not True:
            raise RuntimeError("isolated-unit mutation executor registration is unavailable")
        path = _promote(name, Path(destination))
        attachment = json.loads((path / "units.json").read_text())[0]
        return SimpleNamespace(attachment=attachment, mode="off", revalidated=True, path=path)

    def file_digest(self, path: Path) -> str:
        if __package__:
            from .kernel_mutate_edits import file_digest
        else:
            from kernel_mutate_edits import file_digest
        return file_digest(path)

    def inject_failure(self, point: str) -> None:
        self._edit_manager().failure_point = point

    def stage_edit_proposals(self, call: str, proposals: list[dict]):
        return self._edit_manager().stage(call, proposals)

    def commit_edit_proposals(self, staged, *, cleanup_ok: bool):
        if __package__:
            from .kernel_effect_application import apply_effect
        else:
            from kernel_effect_application import apply_effect
        if __package__:
            from .kernel_mutate_policy import Budget
        else:
            from kernel_mutate_policy import Budget

        state = Budget().fold(self.journal.events())
        files = len(staged.files)
        size = sum(max(len(file["pre"] or b""), len(file["post"] or b"")) for file in staged.files)
        if state.disabled or files > state.remaining()["edit_files"] or size > state.remaining()["edit_bytes"]:
            raise MutationBudgetError("edit_files or edit_bytes")
        for name, amount in (("edit_files", files), ("edit_bytes", size)):
            self.journal.append("BUDGET_RESERVE", name=name, amount=amount, id=staged.call + ":" + name)
        snapshot = {"call": staged.call, "paths": [file["path"] for file in staged.files]}
        try:
            outcome = apply_effect(operation="mutation_edit", snapshot=snapshot,
                                   dispatch=lambda: self._edit_manager().commit(
                                       staged, cleanup_ok=cleanup_ok),
                                   effect_applied=lambda result: result.committed)
            if not outcome.executed:
                raise ValueError("mutation edit was not authorized by the effect boundary")
            return outcome.value
        except (ValueError, OSError):
            if self.edit_state(staged.call) != "PREPARED":
                for name in ("edit_files", "edit_bytes"):
                    self.journal.append("BUDGET_SETTLE", id=staged.call + ":" + name, amount=0)
            raise

    def apply_edit_proposals(self, call: str, proposals: list[dict], *, cleanup_ok: bool):
        return self.commit_edit_proposals(self.stage_edit_proposals(call, proposals), cleanup_ok=cleanup_ok)

    def reconcile_edits(self, call: str):
        return self._edit_manager().recover(call)

    def edit_state(self, call: str) -> str:
        kinds = [row["kind"] for row in self.journal.events() if row.get("call") == call]
        return ("COMMITTED" if "EFFECT_COMMITTED" in kinds else
                "INDETERMINATE" if "EFFECT_INDETERMINATE" in kinds else
                "REFUSED" if "EFFECT_REFUSED" in kinds else
                "PREPARED" if "EFFECT_PREPARED" in kinds else "NONE")

    def committed_edit_paths(self, call: str) -> tuple[str, ...]:
        rows = [row for row in self.journal.events()
                if row.get("kind") == "EFFECT_COMMITTED" and row.get("call") == call]
        if len(rows) != 1:
            return ()
        return tuple(str(file["path"]) for file in rows[0]["files"])


    def close(self) -> None:
        if self.executor is not None:
            root = self.executor.root
            report = self.executor.close()
            self.executor = None
            if not report.ok:
                raise RuntimeError("MUTATION_CONTAINMENT: mutation executor cleanup failed")
            if __package__:
                from .kernel_mutate_sweep import sweep
            else:
                from kernel_mutate_sweep import sweep

            swept = sweep(root.parent, run_locks=lambda name: name != root.name)
            if root.exists() or root.name in swept.unresolved:
                raise RuntimeError("MUTATION_CONTAINMENT: runtime root cleanup failed")

    def retire_hook_executor(self) -> None:
        """Drop a timed-out hook's open ledger and session before returning control."""
        if self.executor is None:
            return
        self.close()
        self._reopen_runtime_root = self.executor_factory is None and self.runtime_root is not None

    def tick(self, parent_turn_id: str, turn_index: int) -> list[str]:
        """Run due offline watches after a committed turn, before the next request."""
        if __package__:
            from .kernel_tool_tape import replay_mode_active, tape_for_run
        else:
            from kernel_tool_tape import replay_mode_active, tape_for_run

        if replay_mode_active():
            events = self.journal.events()
            completions = {row["invocation_id"]: row for row in events
                           if row.get("kind") == "WATCH_TICK_COMPLETE"}
            stops = {row.get("invocation_id") for row in events
                     if row.get("kind") == "watch_stop" and row.get("reason") == "max_ticks"}
            previous: dict[str, str] = {}
            coalesced: dict[str, int] = {}
            replayed = []
            for row in events:
                if row.get("kind") != "WATCH_TICK_INTENT":
                    continue
                completion = completions.get(row["invocation_id"])
                if completion is None:
                    continue
                watch_id = row["watch"]
                observation = completion["observation"]
                if row["parent_turn_id"] == parent_turn_id:
                    record = tape_for_run().lookup(row["invocation_id"])
                    if record is None or record.observation != observation:
                        raise ValueError("watch tape differs from completed journal evidence")
                    if previous.get(watch_id) != observation:
                        prefix = (f"coalesced {coalesced.get(watch_id, 0)} unchanged ticks; "
                                  if coalesced.get(watch_id, 0) else "")
                        replayed.append(f"mutation watch {watch_id}: {prefix}{observation}")
                if previous.get(watch_id) == observation:
                    coalesced[watch_id] = coalesced.get(watch_id, 0) + 1
                else:
                    previous[watch_id] = observation
                    coalesced[watch_id] = 0
                if (row["parent_turn_id"] == parent_turn_id
                        and row["invocation_id"] in stops and coalesced[watch_id]):
                    replayed.append(f"mutation watch {watch_id}: coalesced "
                                    f"{coalesced[watch_id]} unchanged ticks")
            return replayed
        if not isinstance(parent_turn_id, str) or not parent_turn_id or type(turn_index) is not int:
            raise ValueError("invalid committed turn identity")
        output = []
        for watch_id, watch in list(self.watches.items()):
            if parent_turn_id in watch["seen_parents"]:
                continue
            watch["seen_parents"].add(parent_turn_id)
            watch["boundaries"] += 1
            if watch["boundaries"] % watch["every_turns"] != 0:
                continue
            invocation = "w" + hashlib.sha256((watch_id + ":" + parent_turn_id).encode()).hexdigest()[:30]
            if any(row.get("kind") == "WATCH_TICK_INTENT" and row.get("invocation_id") == invocation
                   for row in self.journal.events()):
                continue
            request = watch["args"]
            facts = []
            previous_identities = dict(watch["identities"])
            for item in request["inputs"]:
                path = item if isinstance(item, str) else item["path"]
                try:
                    info = (self.repo / path).stat(follow_symlinks=False)
                    identity = (info.st_dev, info.st_ino, info.st_size)
                    previous = watch["identities"].get(path)
                    state = ("rotation" if previous and previous[:2] != identity[:2] else
                             "truncation" if previous and identity[2] < previous[2] else "present")
                    watch["identities"][path] = identity
                except FileNotFoundError:
                    identity, state = None, "deletion"
                    watch["identities"].pop(path, None)
                facts.append({"path": path, "state": state, "identity": identity})
            snapshots = None
            snapshot_error = None
            if all(row["state"] != "deletion" for row in facts):
                try:
                    executor = self._ensure_executor()
                    snapshots = admit_inputs(self.repo, request["inputs"], self.policy,
                                             snapshot_root=executor.root / "in" / uuid.uuid4().hex,
                                             ordinary_read_globs=self.ordinary_read_globs)
                except (OSError, ValueError) as error:
                    snapshot_error = type(error).__name__
            snapshot_identity = hashlib.sha256(json.dumps({
                "facts": facts, "snapshots": [hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in snapshots.files.values()] if snapshots else [],
                "error": snapshot_error},
                sort_keys=True).encode()).hexdigest()
            self.journal.append("WATCH_TICK_INTENT", invocation_id=invocation,
                                parent_turn_id=parent_turn_id, watch=watch_id,
                                snapshot_identity=snapshot_identity, facts=facts,
                                snapshot_error=snapshot_error)
            tape = tape_for_run() if self.run_dir is not None else None
            if tape is not None:
                tape.intent(call_id=invocation, tool="mutation_watch", args={"watch": watch_id},
                            replay_class="observation", capability_digest=snapshot_identity,
                            definition_digest=None, env_identity="mutation-runtime/1",
                            pre_manifest_digest=snapshot_identity)
            try:
                if snapshot_error is not None:
                    raise ValueError("input snapshot admission refused")
                if snapshots is None:
                    observation = json.dumps({"facts": facts}, sort_keys=True)
                else:
                    invocation_args = {"input": request["input"], "inputs": request["inputs"]}
                    if watch["unit"] == "log_watch" and facts:
                        source = request["inputs"][0]
                        source_path = source if isinstance(source, str) else source["path"]
                        def describe(identity):
                            if identity is None:
                                return None
                            tail = source.get("tail_bytes") if isinstance(source, dict) else None
                            return {"identity": f"{identity[0]}:{identity[1]}", "size": identity[2],
                                    "start": max(0, identity[2] - tail) if tail else 0}
                        payload = dict(request["input"])
                        payload["current"] = describe(watch["identities"].get(source_path))
                        previous = describe(previous_identities.get(source_path))
                        if previous is not None:
                            payload["previous"] = previous
                        invocation_args["input"] = payload
                    observation = self._use(watch["unit"], invocation_args, invocation, snapshots=snapshots,
                                            record_evidence=False)
                if observation.startswith("REFUSED "):
                    raise ValueError(observation)
            except (OSError, ValueError) as error:
                observation = _refuse("WATCH", str(error)[:160])
            watch["ticks"] += 1
            if observation == watch.get("last"):
                watch["coalesced"] += 1
            else:
                prefix = (f"coalesced {watch['coalesced']} unchanged ticks; "
                          if watch["coalesced"] else "")
                output.append(f"mutation watch {watch_id}: {prefix}{observation}")
                watch["last"] = observation
                watch["coalesced"] = 0
            self.journal.append("WATCH_TICK_COMPLETE", invocation_id=invocation,
                                parent_turn_id=parent_turn_id, observation=observation)
            if tape is not None:
                tape.complete(call_id=invocation, exit_code=0, observation=observation,
                              post_manifest_digest=snapshot_identity, env_identity="mutation-runtime/1")
            if watch["ticks"] >= watch["max_ticks"]:
                if watch["coalesced"]:
                    output.append(f"mutation watch {watch_id}: coalesced {watch['coalesced']} unchanged ticks")
                self.journal.append("watch_stop", watch=watch_id, invocation_id=invocation,
                                    reason="max_ticks")
                del self.watches[watch_id]
        return output

    def _args(self, raw: object) -> dict[str, object]:
        if raw is None:
            return {}
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16000:
            raise ValueError("args must be a JSON object string of at most 16000 bytes")
        value = _loads(raw)
        if not isinstance(value, dict):
            raise ValueError("args must encode an object")
        return value

    def dispatch(self, fields: Mapping[str, object], call_id: str, *,
                 remaining_turns: int = 99) -> str:
        """Record a tool call exactly once; a malformed call also gets that op's exact usage."""
        result = self._dispatch_call(fields, call_id, remaining_turns=remaining_turns)
        op = fields.get("op") if isinstance(fields, Mapping) else None
        if result.startswith("REFUSED MUTATION_FRAME") and self.episode is not None:
            result += "\nusage: " + USAGE["submit"].split("; full form:", 1)[0]
        elif result.startswith("REFUSED MUTATION_BAD_ARGUMENTS"):
            result += "\nusage: " + USAGE.get(op if isinstance(op, str) else "", USAGE[""])
        return result

    def _with_repair(self, result: str) -> str:
        episode = self.episode
        if (episode is None or not result.startswith("REFUSED ")
                or "\nrepair:" in result):
            return result
        name = episode.get("unit_id") or "unit"
        ready = self.ready.get(name)
        revision = ready["manifest"]["revision"] + 1 if ready else 1
        left = max(0, int(episode.get("sends_left", 0)))
        return (result + "\nrepair: fix the unit and submit again "
                f"(it becomes revision {revision}); sends left {left}")

    def _dispatch_call(self, fields: Mapping[str, object], call_id: str, *,
                       remaining_turns: int = 99) -> str:
        """Record a tool call exactly once, or return its previously recorded observation."""
        if not isinstance(fields, Mapping) or fields.get("op") not in OPS:
            return _refuse("BAD_ARGUMENTS", "unknown operation")
        if any(key not in {"op", "id", "args"} for key in fields):
            return _refuse("BAD_ARGUMENTS", "unknown field")
        name = fields.get("id")
        if name is not None and (not isinstance(name, str) or len(name) > 64):
            return _refuse("BAD_ARGUMENTS", "id is too long")
        try:
            args = self._args(fields.get("args"))
        except (UnicodeError, ValueError, RecursionError) as error:
            return _refuse("BAD_ARGUMENTS", str(error)[:160])
        try:
            if self.run_dir is None:
                result = self._dispatch(str(fields["op"]), name, args, call_id, remaining_turns)
            else:
                result = self._taped(fields, call_id, lambda: self._dispatch(
                    str(fields["op"]), name, args, call_id, remaining_turns))
        except MutationUnavailable as error:
            result = _refuse("PLATFORM_UNQUALIFIED", str(error))
        except DigestMismatch as error:
            result = _refuse("DIGEST", str(error))
        except MutationBudgetError as error:
            result = _refuse("BUDGET", str(error))
        except MutationStartupTimeout as error:
            result = self._with_repair(_refuse("STARTUP_TIMEOUT", str(error)))
            self.journal.append("REFUSAL", call=call_id, observation=result)
        except (ValueError, OSError) as error:
            result = (_admission_refusal(error) if fields["op"] == "submit" and self.episode is not None
                      else _refuse("ADMISSION", _safe_detail(str(error))[:160]))
        result = self._with_repair(result)
        if not result.startswith("REFUSED ") and not result.startswith("[untrusted mutation observation "):
            content_digest = hashlib.sha256(result.encode()).hexdigest()[:12]
            input_digest = hashlib.sha256(json.dumps(dict(fields), sort_keys=True).encode()).hexdigest()[:12]
            result = (f"[untrusted mutation observation {str(name or 'none')[:64]} - "
                      f"sha256:{content_digest} in:sha256:{input_digest}]\n{result}")
        self.last = result.splitlines()[0][:100]
        return result

    def _taped(self, fields: Mapping[str, object], call_id: str, operation) -> str:
        if __package__:
            from .kernel_tool_tape import (replay_mode_active, tape_for_run, replay_decision,
                                            verify_intent, ToolReplayRefused)
        else:
            from kernel_tool_tape import (replay_mode_active, tape_for_run, replay_decision,
                                          verify_intent, ToolReplayRefused)

        tape = tape_for_run()
        policy = self.policy
        network = getattr(policy, "network", "none")
        replay_class = ("external" if network != "none" and fields["op"] in {"use", "submit"}
                        else "observation")
        capability = json.dumps({"level": self.level, "network": network,
                                 "writes": list(getattr(policy, "write_globs", ()))},
                                sort_keys=True, separators=(",", ":"))
        intent = {"tool": "mutate", "args": dict(fields), "replay_class": replay_class,
                  "capability_digest": hashlib.sha256(capability.encode()).hexdigest(),
                  "definition_digest": None, "env_identity": "mutation-runtime/1",
                  "pre_manifest_digest": "mutation-runtime/1"}
        if replay_mode_active():
            record = tape.lookup(call_id)
            decision = replay_decision(record, replay_external_allowed=self.replay_external_allowed)
            reason = decision.refuse_reason or verify_intent(record, **intent)
            if reason:
                raise ToolReplayRefused(reason)
            if not record.observation.startswith("REFUSED "):
                self._replay_state(fields, call_id)
            return record.observation
        tape.intent(call_id=call_id, **intent)
        try:
            result = operation()
        except MutationUnavailable as error:
            result = _refuse("PLATFORM_UNQUALIFIED", str(error))
        except DigestMismatch as error:
            result = _refuse("DIGEST", str(error))
        except MutationBudgetError as error:
            result = _refuse("BUDGET", str(error))
        except MutationStartupTimeout as error:
            result = _refuse("STARTUP_TIMEOUT", str(error))
        except (ValueError, OSError) as error:
            result = (_admission_refusal(error) if fields["op"] == "submit" and self.episode is not None
                      else _refuse("ADMISSION", _safe_detail(str(error))[:160]))
        result = self._with_repair(result)
        if result.startswith("REFUSED "):
            self.journal.append("REFUSAL", call=call_id, observation=result)
        tape.complete(call_id=call_id, exit_code=None, observation=result,
                      post_manifest_digest="mutation-runtime/1", env_identity="mutation-runtime/1")
        return result

    def _replay_state(self, fields: Mapping[str, object], call_id: str) -> None:
        op = fields.get("op")
        if op not in {"enter", "leave", "unwatch", "submit"}:
            return
        if __package__:
            from .kernel_tool_tape import ToolReplayRefused
        else:
            from kernel_tool_tape import ToolReplayRefused

        kind = {"enter": "episode_enter", "leave": "episode_leave",
                "unwatch": "watch_stop", "submit": "ADMISSION"}[op]
        rows = [row for row in self.journal.events()
                if row.get("kind") == kind and row.get("call") == call_id]
        if len(rows) != 1:
            raise ToolReplayRefused(f"mutation {kind} evidence is missing or ambiguous for {call_id}")
        row = rows[0]
        if op == "enter":
            if self.episode is not None and self.episode.get("id") != row.get("episode"):
                raise ToolReplayRefused("mutation episode replay order differs from the journal")
            self.episode = {"id": row["episode"], "goal": row["goal"], "sends_left": 4,
                            "unit_id": row.get("unit_id"),
                            "input_ids": list(row.get("inputs", ())),
                            "kind": row.get("unit_kind", "reader"),
                            "needs": dict(row.get("needs", {})),
                            "draft": {}, "diagnostic": "none"}
        elif op == "leave":
            if self.episode is None or self.episode.get("id") != row.get("episode"):
                raise ToolReplayRefused("mutation leave replay order differs from the journal")
            self.episode = None
        elif op == "submit":
            if __package__:
                from .kernel_mutate_build import source_digest
            else:
                from kernel_mutate_build import source_digest
            if __package__:
                from .kernel_mutate_wire import strict_loads
            else:
                from kernel_mutate_wire import strict_loads

            blob = row.get("blob")
            if self.episode is None or not isinstance(blob, str):
                raise ToolReplayRefused("mutation admission source evidence is missing")
            try:
                bundle = strict_loads(self.journal.get_blob(blob))
                manifest = bundle["manifest"]
                files = {name: base64.b64decode(source, validate=True)
                         for name, source in bundle["files"].items()}
                if source_digest(files, manifest) != row.get("source_digest") or manifest["name"] != row.get("name"):
                    raise ValueError("admission source digest differs")
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ToolReplayRefused("mutation admission source evidence is invalid") from error
            self.ready[manifest["name"]] = {"manifest": manifest, "files": files,
                                             "source_digest": row["source_digest"]}
            self.episode = None
        else:
            self.watches.pop(str(row.get("watch", "")), None)

    def _dispatch(self, op: str, name: object, args: dict[str, object], call_id: str,
                  remaining_turns: int) -> str:
        if self.episode is not None and op not in {"submit", "leave", "inspect", "catalog"}:
            return _refuse("FRAME", "only submit, leave, inspect or catalog in an authoring episode")
        if op in {"use", "watch", "retain"} and name is None:
            # Models often put the unit id inside args, or omit it with a single unit ready.
            if isinstance(args.get("id"), str):
                name = args.pop("id")
            elif op in {"use", "watch"} and len(self.ready) == 1:
                name = next(iter(self.ready))
        if op == "catalog":
            return self._catalog(args)
        if op == "enter":
            if self.level != "authored":
                return _refuse("LEVEL", "authoring requires the authored level")
            if self.episode is not None:
                return _refuse("FRAME", "an authoring episode is already open")
            if name is not None and _UNIT_NAME.fullmatch(name) is None:
                return _refuse("BAD_ARGUMENTS", "id must be a unit name")
            if remaining_turns < 2:
                return _refuse("BUDGET", "two nonterminal turns are required")
            if self.episodes_started >= 3:
                return _refuse("BUDGET", "authoring episode budget exhausted")
            if name is None and isinstance(args.get("id"), str):
                name = args.pop("id")
                if _UNIT_NAME.fullmatch(name) is None:
                    return _refuse("BAD_ARGUMENTS", "id must be a unit name")
            goal = args.get("goal")
            if not isinstance(goal, str) or not goal.strip() or len(goal) > 512:
                return _refuse("BAD_ARGUMENTS", "goal must be a short nonempty string")
            unknown = sorted(key for key in args if key not in {"goal", "inputs", "kind", "needs"})
            if unknown:
                return _refuse("BAD_ARGUMENTS", "unknown enter field " + ", ".join(unknown)[:80])
            inputs = args.get("inputs", [])
            if (not isinstance(inputs, list) or len(inputs) > 16
                    or any(not isinstance(item, str) or not item or len(item) > 256
                           for item in inputs)):
                return _refuse("BAD_ARGUMENTS", "inputs must be up to 16 path strings")
            kind = args.get("kind", "reader")
            if not isinstance(kind, str) or not kind or len(kind) > 40:
                return _refuse("BAD_ARGUMENTS", "kind must be a short string")
            needs = args.get("needs", {})
            if (not isinstance(needs, dict) or any(key not in {"network", "writes"} for key in needs)
                    or any(type(value) is not bool for value in needs.values())):
                return _refuse("BAD_ARGUMENTS", "needs must contain boolean network and writes flags")
            if needs.get("network") and getattr(self.policy, "network", "none") != "public-web":
                return _refuse("GRANT", "network access is not granted")
            if needs.get("writes") and not getattr(self.policy, "write_globs", ()):
                return _refuse("GRANT", "writes are not granted")
            episode_id = f"e{self.episodes_started + 1}"
            self.journal.append("episode_enter", episode=episode_id, goal=goal.strip(),
                                inputs=list(inputs), unit_kind=kind, needs=dict(needs),
                                unit_id=name, call=call_id)
            self.episode = {"id": episode_id, "goal": goal.strip(), "sends_left": 4,
                            "unit_id": name,
                            "input_ids": list(inputs), "kind": kind, "needs": dict(needs),
                            "draft": {}, "diagnostic": "none"}
            self.episodes_started += 1
            admitted = (", ".join(f"{path} as in{index}" for index, path in enumerate(inputs))
                        if inputs else "none; pass inputs to submit to smoke-test on real files")
            return (f"mutation episode {episode_id} OPEN; submit a unit or leave\n"
                    f"inputs: {admitted}\n"
                    "unit contract: main.py defines handle(input, ctx) -> dict; "
                    "episode input i is ctx.input_bytes(input['input_id']) with ids in0, in1, ...\n"
                    "submit: " + USAGE["submit"].split("; full form:", 1)[0])
        if op == "leave":
            if self.episode is None:
                return _refuse("FRAME", "no authoring episode is open")
            episode_id = str(self.episode["id"])
            self.journal.append("episode_leave", episode=episode_id, call=call_id)
            self.episode = None
            return f"mutation episode {episode_id} LEFT"
        if op == "inspect":
            if self.episode is not None:
                return json.dumps({"episode": self.episode["id"], "goal": self.episode["goal"],
                                   "sends_left": self.episode["sends_left"]}, sort_keys=True)
            if isinstance(name, str):
                if name in self.ready:
                    prepared = self.ready[name]
                    manifest = prepared["manifest"]
                    digest = prepared["source_digest"]
                else:
                    if __package__:
                        from .kernel_mutate_build import source_digest
                    else:
                        from kernel_mutate_build import source_digest

                    manifest, files, _directory = self._prebuilt(name)
                    digest = source_digest(files, manifest)
                return json.dumps({"name": manifest["name"],
                                   "revision": manifest["revision"],
                                   "summary": manifest["summary"],
                                   "kind": manifest["kind"],
                                   "requires": manifest["requires"],
                                   "source_digest": digest}, sort_keys=True)
            return self._catalog(args)
        if op == "unwatch":
            if not isinstance(name, str) or name not in self.watches:
                return _refuse("UNKNOWN", "watch id is not active")
            self.journal.append("watch_stop", watch=name, call=call_id)
            del self.watches[name]
            return f"mutation watch {name} STOPPED"
        if op == "submit":
            return self._submit(args, call_id)
        if op == "use":
            if not isinstance(name, str):
                return _refuse("BAD_ARGUMENTS", "unit id is required")
            return self._use(name, args, call_id)
        if op == "retain":
            if not isinstance(name, str) or _UNIT_NAME.fullmatch(name) is None:
                return _refuse("BAD_ARGUMENTS", "unit id is required")
            if set(args) - {"scope"}:
                return _refuse("BAD_ARGUMENTS", "unknown retention field")
            if args.get("scope", "run") != "run":
                return _refuse("GRANT", "model retention is limited to this run")
            if name in self.ready:
                digest = self.ready[name]["source_digest"]
            else:
                if __package__:
                    from .kernel_mutate_build import source_digest
                else:
                    from kernel_mutate_build import source_digest

                manifest, files, _directory = self._prebuilt(name)
                digest = source_digest(files, manifest)
            self.journal.append("retention", call=call_id, name=name,
                                source_digest=digest, scope="run")
            return f"mutation {name} RETAINED in run audit {digest[:19]}"
        if op == "watch":
            if not isinstance(name, str):
                return _refuse("BAD_ARGUMENTS", "unit id is required")
            if set(args) != {"input", "inputs", "every_turns", "max_ticks"}:
                return _refuse("BAD_ARGUMENTS", "watch requires input, inputs, every_turns and max_ticks")
            if (not isinstance(args["input"], dict) or not isinstance(args["inputs"], list)
                    or type(args["every_turns"]) is not int or not 1 <= args["every_turns"] <= 30
                    or type(args["max_ticks"]) is not int or not 1 <= args["max_ticks"] <= 30):
                return _refuse("BAD_ARGUMENTS", "invalid watch schedule")
            manifest = self.ready[name]["manifest"] if name in self.ready else self._prebuilt(name)[0]
            if (manifest["requires"]["network"] or manifest["requires"]["writes"]
                    or getattr(self.policy, "network", "none") != "none"
                    or getattr(self.policy, "write_globs", ())):
                return _refuse("GRANT", "watches require offline read-only authority")
            watch_id = f"w{len(self.watches) + 1}-{uuid.uuid4().hex[:8]}"
            self.journal.append("watch_start", watch=watch_id, call=call_id, unit=name,
                                args=args, every_turns=args["every_turns"], max_ticks=args["max_ticks"])
            self.watches[watch_id] = {"unit": name, "args": args, "every_turns": args["every_turns"],
                                      "max_ticks": args["max_ticks"], "ticks": 0, "identities": {},
                                      "last": None, "coalesced": 0,
                                      "boundaries": 0, "seen_parents": set()}
            return f"mutation watch {watch_id} WATCHING {name}"
        return _refuse("BAD_ARGUMENTS", "unknown operation")

    def _catalog(self, args: dict[str, object]) -> str:
        offset = args.get("offset", 0)
        if type(offset) is not int or offset < 0:
            return _refuse("BAD_ARGUMENTS", "catalog offset must be a nonnegative integer")
        manifest_path = self.catalog_root / "manifest.json"
        try:
            manifest = _read_json_file(manifest_path, 64 << 10)
        except FileNotFoundError:
            return "mutation catalog: no prebuilt units installed"
        except (OSError, ValueError, UnicodeError, RecursionError) as error:
            return _refuse("CATALOG", f"prebuilt catalog is invalid: {type(error).__name__}")
        units = manifest.get("units")
        if (manifest.get("schema") != "stateframe-mutation-prebuilt/1"
                or not isinstance(units, dict)):
            return _refuse("CATALOG", "prebuilt catalog has an invalid schema")
        names = sorted(units)
        if any(not isinstance(name, str) or _UNIT_NAME.fullmatch(name) is None
               or not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
               for name, digest in units.items()):
            return _refuse("CATALOG", "prebuilt catalog has an invalid entry")
        rows = []
        for name in names[offset:offset + 8]:
            try:
                unit = _read_json_file(self.catalog_root / name / "unit.json", 8 << 10)
            except (OSError, ValueError, UnicodeError, RecursionError):
                return _refuse("CATALOG", f"unit {name} manifest is invalid")
            if unit.get("name") != name or not isinstance(unit.get("summary"), str):
                return _refuse("CATALOG", f"unit {name} manifest does not match catalog")
            rows.append(f"{name} {str(unit['summary'])[:120]} {units[name][:19]}")
        following = f" next_offset={offset + 8}" if offset + 8 < len(names) else ""
        return "mutation catalog:\n" + ("\n".join(rows) or "none") + following
