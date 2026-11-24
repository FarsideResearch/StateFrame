"""Opt-in CLI surface for local OpenTelemetry export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .. import __version__ as VERSION
from . import DETAIL_FLAGS


_SETTINGS = ("config", "traces", "protocol", "compression", "timeout_ms", "sample", "sample_ratio", *DETAIL_FLAGS)


class _TelemetryParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        _diagnostic("unsupported_setting")
        self.exit(2)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version"}:
        print(f"stateframe-otel {VERSION}")
        return 0
    command = args[0]
    if command == "export":
        return _export(args[1:])
    if command == "status":
        return _status(args[1:])
    if command == "purge":
        return _purge(args[1:])
    _diagnostic("unsupported_setting")
    return 2


def _export(args: list[str]) -> int:
    parser = _TelemetryParser(prog="stateframe-otel export", allow_abbrev=False)
    parser.add_argument("run", type=Path)
    parser.add_argument("--stopped", action="store_true")
    parser.add_argument("--trust-source", action="store_true")
    _add_settings(parser)
    try:
        parsed = parser.parse_args(args)
    except SystemExit as error:
        return int(error.code)
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() != "true" and _explain_source_mode(
        parsed.run, operation="export",
    ):
        return 2
    values = _settings(parsed)
    from .companion import export
    return export(parsed.run, settings=tuple(values.items()), stopped=parsed.stopped,
                  trust_source=parsed.trust_source)


def _status(args: list[str]) -> int:
    parser = _TelemetryParser(prog="stateframe-otel status", allow_abbrev=False)
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", action="store_true")
    _add_detail_settings(parser)
    parser.add_argument("--config", type=Path)
    try:
        parsed = parser.parse_args(args)
    except SystemExit as error:
        return int(error.code)
    if _explain_source_mode(parsed.run, operation="status"):
        return 2
    from .companion import status
    code, report = status(parsed.run)
    from .levels import describe
    from .config import Settings, resolve
    settings = Settings(config=parsed.config, **{name: getattr(parsed, name) for name in DETAIL_FLAGS})
    config = resolve(settings, os.environ)
    report["detail"] = describe(config.levels) if config.traces is not None else ()
    if "invalid_config" in config.diagnostics:
        code = 2
    if parsed.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(_status_text(report))
        print(_detail_text(report["detail"]))
    return code


def _explain_source_mode(run: Path, *, operation: str) -> bool:
    from .authority import source_mode_problem

    problem = source_mode_problem(Path(os.path.abspath(run)))
    if problem is None:
        return False
    _directory, mode = problem
    if mode == "700":
        reason = "run directory permission refused"
    else:
        reason = "source parent permission refused"
    prefix = f"{operation} " if operation in {"export", "status"} else ""
    print(f"stateframe-otel: {prefix}{reason}", file=sys.stderr)
    return True


def _purge(args: list[str]) -> int:
    parser = _TelemetryParser(prog="stateframe-otel purge", allow_abbrev=False)
    parser.add_argument("--yes", action="store_true", required=True)
    try:
        parser.parse_args(args)
    except SystemExit as error:
        return int(error.code)
    from .companion import purge
    return purge()


def _add_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--traces", choices=("otlp", "none"))
    parser.add_argument("--protocol", choices=("http/protobuf", "http/json"))
    parser.add_argument("--compression", choices=("none", "gzip"))
    parser.add_argument("--max-export-time-milliseconds", "--timeout-ms", dest="timeout_ms", type=int)
    parser.add_argument("--trace-sampler", "--sample", dest="sample",
                        choices=("always_on", "always_off", "traceidratio"))
    parser.add_argument("--trace-sampling-ratio", "--sample-ratio", dest="sample_ratio")
    _add_detail_settings(parser)


def _add_detail_settings(parser: argparse.ArgumentParser) -> None:
    for name in DETAIL_FLAGS:
        parser.add_argument("--" + name.replace("_", "-"), type=_boolean_value, metavar="{true,false}")


def _boolean_value(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def _detail_text(fields) -> str:
    return "Detail fields that can be exported: " + (", ".join(fields) if fields else "none")


def _settings(parsed: argparse.Namespace) -> dict[str, object]:
    return {name: getattr(parsed, name) for name in _SETTINGS if getattr(parsed, name) is not None}


def _status_text(report: dict[str, object]) -> str:
    trace = report.get("trace_id", "unavailable")
    runtime = report.get("runtime", "unknown")
    reason = f" reason={report['reason']}" if report.get("reason") in {
        "source_untrusted", "state_unavailable",
    } else ""
    traces = report.get("signals", {}).get("traces", {})
    counts = "".join(f" {key}={traces[key]}" for key in ("acked", "queue_items", "partial_items") if key in traces)
    return f"Telemetry status trace={trace} runtime={runtime}{counts}{reason}"


def _print_help() -> None:
    print("usage: stateframe-otel {export,status,purge} ...")


def _diagnostic(code: str) -> None:
    if code not in {"unsupported_setting", "invalid_config", "isolation_unverified"}:
        code = "invalid_config"
    print(f"stateframe-otel: {code}", file=sys.stderr)
