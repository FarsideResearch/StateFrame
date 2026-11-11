"""Human-facing commands for inspecting and managing isolated mutation units."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

from . import profiles
if __package__:
    from .engine.kernel_safe_paths import safe_expanduser, safe_home
else:
    from engine.kernel_safe_paths import safe_expanduser, safe_home

_UNIT_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class _Refusal(Exception):
    """A requested operation is unavailable or does not meet its preconditions."""


def _engine_module(name: str) -> object:
    """Import an optional mutation engine module only for the command that needs it."""
    if __package__:
        return importlib.import_module(f".engine.{name}", package=__package__)
    return importlib.import_module(name)


def _platform_unsupported() -> str:
    """Load the platform refusal only when a mutation management command needs it."""
    if __package__:
        from .mutate_host import PLATFORM_UNSUPPORTED
    else:
        from mutate_host import PLATFORM_UNSUPPORTED
    return PLATFORM_UNSUPPORTED


def _powerup_consumer() -> object:
    """Require a consumer that positively advertises and validates mutation units."""
    module_name = f"{__package__}.powerups" if __package__ else "powerups"
    if module_name not in sys.modules and not (Path(__file__).resolve().parent / "powerups.py").is_file():
        raise _Refusal("isolated-unit mutation executor is not registered")
    try:
        consumer = (importlib.import_module(".powerups", package=__package__)
                    if __package__ else importlib.import_module("powerups"))
    except ImportError as error:
        raise _Refusal("isolated-unit mutation executor is unavailable") from error
    supports = getattr(consumer, "supports_isolated_unit_executor", None)
    validate = getattr(consumer, "validate_isolated_unit_attachment", None)
    try:
        registered = callable(supports) and supports("mutation") is True
    except Exception as error:
        raise _Refusal("isolated-unit mutation executor registration failed") from error
    if (not registered or not callable(validate)
            or not callable(getattr(consumer, "install", None))):
        raise _Refusal("isolated-unit mutation executor registration or validator is unavailable")
    return consumer


def _home() -> Path:
    configured_home = os.environ.get("STATEFRAME_HOME")
    if configured_home is not None:
        return safe_expanduser(Path(configured_home))
    user_home = safe_home()
    if user_home is None:
        raise _Refusal("cannot determine home directory; set HOME or STATEFRAME_HOME")
    return user_home / ".stateframe"


def _store_root(home: Path) -> Path:
    return home / "mutations"


def _read_json(path: Path, *, limit: int) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{path.name} must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f"{path.name} exceeds {limit} bytes")
    finally:
        os.close(descriptor)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_bundle_member(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 64 << 10:
            raise ValueError("invalid mutation bundle member")
        data = os.read(fd, (64 << 10) + 1)
        if len(data) != info.st_size:
            raise ValueError("mutation bundle member changed during read")
        return data
    finally:
        os.close(fd)


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in mutation catalog: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number in mutation catalog: {value}")


def _prebuilt_catalog() -> tuple[Path, dict[str, object]]:
    root = Path(__file__).resolve().parent / "mutations_prebuilt"
    manifest = _read_json(root / "manifest.json", limit=128 << 10)
    units = manifest.get("units")
    if manifest.get("schema") != "stateframe-mutation-prebuilt/1" or not isinstance(units, dict):
        raise ValueError("prebuilt mutation manifest has an unsupported shape")
    if any(not isinstance(name, str) or not _UNIT_NAME.fullmatch(name)
           or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
           for name, digest in units.items()):
        raise ValueError("prebuilt mutation manifest contains an invalid unit entry")
    return root, units


def _prebuilt_units() -> list[dict[str, object]]:
    root, units = _prebuilt_catalog()
    rows: list[dict[str, object]] = []
    for name in sorted(units):
        path = root / name / "unit.json"
        data = _read_json(path, limit=16 << 10)
        rows.append({
            "name": name,
            "revision": data.get("revision"),
            "digest": units[name],
            "summary": data.get("summary", ""),
            "kind": data.get("kind", ""),
            "origin": "prebuilt",
        })
    return rows


def _prebuilt_unit(name: str) -> dict[str, object] | None:
    root, units = _prebuilt_catalog()
    digest = units.get(name)
    if digest is None:
        return None
    data = _read_json(root / name / "unit.json", limit=16 << 10)
    return {
        "name": name,
        "revision": data.get("revision"),
        "digest": digest,
        "summary": data.get("summary", ""),
        "kind": data.get("kind", ""),
        "origin": "prebuilt",
        "input_schema": data.get("input_schema"),
        "output_schema": data.get("output_schema"),
        "requires": data.get("requires", {}),
    }


def _as_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        fields = getattr(value, "__dataclass_fields__")
        return {name: _as_json_value(getattr(value, name)) for name in fields}
    if value is None or type(value) in (str, int, float, bool):
        return value
    return str(value)


def _json(value: object) -> None:
    print(json.dumps(_as_json_value(value), sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False))


def _text_result(status: str, unit: str, *, scope: str | None = None,
                 result: object = None) -> None:
    print(f"status: {status}")
    print(f"unit: {unit}")
    if scope is not None:
        print(f"scope: {scope}")
    print("result: " + json.dumps(_as_json_value(result), sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _text_record(record: Mapping[str, object]) -> None:
    for key in ("name", "revision", "digest", "summary", "kind", "origin", "scope"):
        if key in record:
            value = record[key]
            print(f"{key}: {value}")


def _text_rows(rows: list[object]) -> None:
    if not rows:
        print("units: none")
        return
    for row in rows:
        if isinstance(row, Mapping):
            _text_record(row)
            print()
        else:
            print(f"unit: {row}")


def register(commands: argparse._SubParsersAction) -> None:
    """Register the ``stateframe mutations`` command group."""
    root = commands.add_parser("mutations", help="inspect and manage sandboxed mutation units")
    actions = root.add_subparsers(dest="mutations_command", required=True)

    listing = actions.add_parser("list", help="list prebuilt, run, or retained units")
    selector = listing.add_mutually_exclusive_group()
    selector.add_argument("--run", type=Path, help="list units recorded by this run")
    selector.add_argument("--scope", choices=("user", "project"), help="list retained units")
    listing.add_argument("--json", action="store_true", help="print JSON")

    show = actions.add_parser("show", help="show a unit's contract and provenance")
    show.add_argument("unit")
    show.add_argument("--json", action="store_true", help="print JSON")

    enable = actions.add_parser("enable-level", help="show how to set the mutation level")
    enable.add_argument("level", choices=("off", "prebuilt", "authored"))
    enable.add_argument("--profile", help="validate a profile without modifying it")

    selftest = actions.add_parser("selftest", help="qualify the local mutation sandbox")
    selftest.add_argument("--json", action="store_true", help="print JSON")

    doctor = actions.add_parser("doctor", help="show sandbox qualification and host facts")
    doctor.add_argument("--json", action="store_true", help="print JSON")

    retain = actions.add_parser("retain", help="retain a unit from a run")
    retain.add_argument("run", type=Path)
    retain.add_argument("unit")
    retain.add_argument("--scope", choices=("project", "user"), required=True)

    remove = actions.add_parser("remove", help="remove a retained unit reference")
    remove.add_argument("unit")
    remove.add_argument("--scope", choices=("project", "user"), required=True)

    promote = actions.add_parser("promote", help="promote a unit into a disabled powerup")
    promote.add_argument("unit")
    promote.add_argument("--out", type=Path, help="directory for the exported powerup")

    prune = actions.add_parser("prune", help="sweep abandoned runtimes and collect stored units")
    prune.add_argument("--dry-run", action="store_true", help="report planned cleanup without applying it")
    prune.add_argument("--json", action="store_true", help="print JSON")


def _profile_guidance(level: str, profile: str | None) -> int:
    if profile is not None:
        try:
            profiles.load(profile)
        except (OSError, ValueError) as error:
            raise _Refusal(str(error)) from error
        print(f"profile: {profile} (unchanged; profiles have no persistent user override writer)")
    print(f"environment: export STATEFRAME_MUTATIONS={level}")
    run_line = f"stateframe run --mutations {level}"
    chat_line = f"stateframe chat --mutations {level}"
    if profile is not None:
        run_line += " --profile " + shlex.quote(profile)
        chat_line += " --profile " + shlex.quote(profile)
    print(f"run: {run_line}")
    print(f"chat: {chat_line}")
    return 0


def _selftest_report(home: Path) -> dict[str, object]:
    module = _engine_module("kernel_mutate_qualify")
    function = getattr(module, "selftest", None)
    if not callable(function):
        raise RuntimeError("kernel_mutate_qualify does not provide selftest()")
    report = function(home)
    if not isinstance(report, Mapping):
        raise ValueError("mutation selftest returned a non-object report")
    return dict(report)


def _doctor_facts() -> dict[str, object]:
    linux = sys.platform.startswith("linux")
    darwin = sys.platform == "darwin"
    bwrap = shutil.which("bwrap") if linux else None
    seccomp = (Path("/proc/sys/kernel/seccomp/actions_avail").is_file() if linux else None)
    cgroup = (Path("/sys/fs/cgroup/cgroup.controllers").is_file() if linux else None)
    if linux:
        sandbox = "bwrap available" if bwrap else "bwrap unavailable"
    elif darwin:
        sandbox = "mutation execution unavailable in this release"
    else:
        sandbox = "unsupported platform"
    if linux:
        bwrap_fact = bwrap or "unavailable"
    else:
        bwrap_fact = "not_applicable"
    return {
        "platform": platform.platform(),
        "seccomp": "available" if seccomp else "unavailable" if linux else "not_applicable",
        "cgroup": "v2 available" if cgroup else "unavailable" if linux else "not_applicable",
        "bwrap": bwrap_fact,
        "sandbox": sandbox,
    }


def _run_lock_probe(home: Path, candidate: Path) -> bool:
    """Return True when a runtime may still belong to a live or unidentified run."""
    matches: list[Path] = []
    run_roots = {home / "runs"}
    user_home = safe_home()
    if user_home is not None:
        run_roots.add(user_home / ".stateframe" / "runs")
    for runs in run_roots:
        try:
            entries = sorted(runs.iterdir())
        except FileNotFoundError:
            continue
        except OSError:
            return True
        for run_dir in entries:
            try:
                info = run_dir.lstat()
            except OSError:
                return True
            if not stat.S_ISDIR(info.st_mode):
                continue
            try:
                spec_path = run_dir / "spec.json"
                spec = _read_json(spec_path, limit=1 << 20)
            except FileNotFoundError:
                continue
            except (OSError, UnicodeError, ValueError):
                return True
            if isinstance(spec, dict) and spec.get("run_identity_seed") == candidate.name:
                matches.append(run_dir)

    if len(matches) != 1:
        return True
    lock_path = matches[0] / ".run.lock"
    try:
        lock_info = lock_path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not stat.S_ISREG(lock_info.st_mode):
        return True
    descriptor = None
    try:
        descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        lock_info = os.fstat(descriptor)
        if not stat.S_ISREG(lock_info.st_mode):
            return True
        with os.fdopen(descriptor, "r+", closefd=False) as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return False


def _sweep(sweeper: object, root: Path, home: Path, *, dry_run: bool) -> tuple[object, list[str]]:
    function = getattr(sweeper, "sweep", None)
    if not callable(function):
        raise RuntimeError("kernel_mutate_sweep has no sweep function")
    if dry_run and "dry_run" not in inspect.signature(function).parameters:
        raise _Refusal("runtime sweeper does not support dry-run; no cleanup was started")

    locked: list[str] = []

    def run_locks(candidate: object) -> bool:
        path = Path(candidate)
        state = _run_lock_probe(home, path)
        if state:
            locked.append(str(path))
        return state

    kwargs: dict[str, object] = {"run_locks": run_locks}
    if dry_run:
        kwargs["dry_run"] = True
    return function(root, **kwargs), locked


def _call(module: object, name: str, *args: object, **kwargs: object) -> object:
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"kernel_mutate_store does not provide {name}()")
    return function(*args, **kwargs)


def _promote(name: str, destination: Path) -> Path:
    consumer = _powerup_consumer()
    if _UNIT_NAME.fullmatch(name) is None:
        raise _Refusal("invalid mutation unit name")
    prebuilt_root, prebuilt = _prebuilt_catalog()
    if name in prebuilt:
        digest = prebuilt[name]
        source_dir = prebuilt_root / name
        bundle = {path.name: _read_bundle_member(path) for path in source_dir.iterdir()
                  if path.name == "unit.json" or path.suffix == ".py"}
        origin = "prebuilt"
    else:
        store_module = _engine_module("kernel_mutate_store")
        record = _call(store_module, "show_unit", _store_root(_home()), name)
        if not isinstance(record, Mapping):
            raise _Refusal(f"unknown mutation unit: {name}")
        digest = record["digest"]
        store = getattr(store_module, "MutationStore")(_home())
        bundle = store.get(digest)
        origin = "retained"
    builder = _engine_module("kernel_mutate_build")
    manifest = _read_json(source_dir / "unit.json", limit=8192) if origin == "prebuilt" else json.loads(bundle["unit.json"])
    if builder.source_digest({key: value for key, value in bundle.items() if key != "unit.json"}, manifest) != digest:
        raise _Refusal("unit source digest differs from catalog")
    requires = manifest["requires"]
    ceiling = {"network": "public-web" if requires["network"] else "none",
               "writes": [], "domains": []}
    encoded = json.dumps({**ceiling, "read_globs": []}, sort_keys=True,
                         separators=(",", ":")).encode()
    attachment = {"schema": "stateframe-isolated-unit-attachment/1",
                  "unit_digest": digest, "executor": "mutation",
                  "capability_ceiling_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
                  "capability_ceiling": ceiling,
                  "replay_policy": "external" if requires["network"] else "observation",
                  "binding": {"kind": "command", "name": name}}
    try:
        valid_attachment = consumer.validate_isolated_unit_attachment(attachment) is True
    except Exception as error:
        raise _Refusal("mutation attachment validation failed") from error
    if not valid_attachment:
        raise _Refusal("mutation attachment was refused by the installed consumer")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    unit_dir = destination / "unit"
    unit_dir.mkdir(mode=0o700)
    for filename, value in bundle.items():
        (unit_dir / filename).write_bytes(value)
    (destination / "powerup.toml").write_text(
        f'name = "{name}"\nversion = "0.1.0"\nmode = "off"\n', encoding="utf-8")
    (destination / "units.json").write_text(json.dumps([attachment], sort_keys=True) + "\n", encoding="utf-8")
    (destination / "SKILL.md").write_text(
        f"# {name}\n\n{manifest['summary']}\n\nRuns as an isolated StateFrame mutation unit.\n",
        encoding="utf-8")
    (destination / "install.json").write_text(json.dumps({"schema": "stateframe-mutation-promotion/1",
        "unit_digest": digest, "origin": origin}, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _command(args: argparse.Namespace) -> int:
    action = args.mutations_command
    if action == "doctor" and sys.platform != "linux":
        platform_limit = _platform_unsupported()
        report = {"ok": False,
                  "reason": platform_limit,
                  "checks": [],
                  "platform": platform.platform()}
        result = {"ok": False, "selftest": report, "facts": _doctor_facts(),
                  "platform_limit": platform_limit}
        if args.json:
            _json(result)
        else:
            print("ok: false")
            for key, value in result["facts"].items():
                print(f"{key}: {value}")
            print(f"mutation execution: unavailable ({platform_limit})")
        return 0

    home = _home()
    store_root = _store_root(home)

    if action == "list":
        if args.run is None and args.scope is None:
            rows = _prebuilt_units()
        else:
            store = _engine_module("kernel_mutate_store")
            rows = _call(store, "list_units", store_root, scope=args.scope,
                         run=safe_expanduser(args.run).resolve() if args.run else None)
            if not isinstance(rows, list):
                raise ValueError("mutation store list_units() must return a list")
        if args.json:
            result = {"units": rows}
            if sys.platform != "linux":
                result["platform_limit"] = _platform_unsupported()
            _json(result)
        else:
            _text_rows(rows)
            if sys.platform != "linux":
                print(f"platform limit: {_platform_unsupported()}")
        return 0

    if action == "show":
        item = _prebuilt_unit(args.unit)
        if item is None:
            store = _engine_module("kernel_mutate_store")
            item = _call(store, "show_unit", store_root, args.unit)
        if not isinstance(item, Mapping):
            raise _Refusal(f"unknown mutation unit: {args.unit}")
        if args.json:
            result = dict(item)
            if sys.platform != "linux":
                result["platform_limit"] = _platform_unsupported()
            _json(result)
        else:
            _text_record(item)
            if sys.platform != "linux":
                print(f"platform limit: {_platform_unsupported()}")
            for key in ("input_schema", "output_schema", "requires"):
                if key in item:
                    print(f"{key}: {json.dumps(_as_json_value(item[key]), sort_keys=True)}")
        return 0

    if action == "enable-level":
        return _profile_guidance(args.level, args.profile)

    if action == "selftest":
        report = _selftest_report(home)
        if args.json:
            _json(report)
        else:
            print(f"ok: {str(bool(report.get('ok'))).lower()}")
            for key in ("platform", "reason"):
                if key in report:
                    print(f"{key}: {report[key]}")
            checks = report.get("checks", [])
            if isinstance(checks, list):
                for row in checks:
                    if isinstance(row, Mapping):
                        state = "ok" if row.get("ok") else "failed"
                        detail = f" {row['detail']}" if row.get("detail") else ""
                        print(f"check.{row.get('name', 'unknown')}: {state}{detail}")
        return 0 if report.get("ok") is True else 1

    if action == "doctor":
        report = _selftest_report(home)
        result = {"ok": report.get("ok") is True, "selftest": report,
                  "facts": _doctor_facts()}
        if args.json:
            _json(result)
        else:
            print(f"ok: {str(result['ok']).lower()}")
            for key, value in result["facts"].items():
                print(f"{key}: {value}")
            print(f"selftest checks: {len(report.get('checks', []))}")
        return 0 if result["ok"] else 1

    if action == "retain":
        store = _engine_module("kernel_mutate_store")
        result = _call(store, "retain_from_run", store_root,
                       safe_expanduser(args.run).resolve(), args.unit, args.scope)
        _text_result("retained", args.unit, scope=args.scope, result=result)
        return 0

    if action == "remove":
        store = _engine_module("kernel_mutate_store")
        result = _call(store, "remove_unit", store_root, args.unit, args.scope)
        _text_result("removed", args.unit, scope=args.scope, result=result)
        return 0

    if action == "promote":
        destination = args.out or Path.cwd() / (args.unit + "-mutation-powerup")
        consumer = _powerup_consumer()
        exported = _promote(args.unit, destination)
        consumer.install(exported)
        return 0

    if action == "prune":
        sweeper = _engine_module("kernel_mutate_sweep")
        store = _engine_module("kernel_mutate_store")
        sweep_report, locked_roots = _sweep(
            sweeper, home / "rt" / "mutations", home, dry_run=args.dry_run)
        gc_report = _call(store, "gc", store_root, dry_run=args.dry_run)
        result = {"dry_run": bool(args.dry_run), "locked_roots": locked_roots,
                  "sweep": sweep_report, "gc": gc_report}
        if args.json:
            _json(result)
        else:
            print(f"dry_run: {str(bool(args.dry_run)).lower()}")
            print(f"locked_roots: {json.dumps(locked_roots)}")
            print(f"sweep: {_as_json_value(sweep_report)}")
            print(f"gc: {_as_json_value(gc_report)}")
        unresolved = getattr(sweep_report, "unresolved", None)
        if unresolved is None and isinstance(sweep_report, Mapping):
            unresolved = sweep_report.get("unresolved", [])
        return 1 if unresolved else 0

    raise _Refusal(f"unknown mutations command: {action}")


def dispatch(args: argparse.Namespace) -> int:
    """Run a parsed mutations command with stable refusal and failure exit codes."""
    try:
        if sys.platform != "linux" and args.mutations_command not in {"list", "show", "doctor"}:
            raise _Refusal(_platform_unsupported())
        return _command(args)
    except _Refusal as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
