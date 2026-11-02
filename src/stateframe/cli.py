"""Command line: ``stateframe run`` and ``stateframe inspect``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, profiles
from .endpoint import CHAT_SUFFIX, Endpoint, EndpointError
from .inspect_run import InspectError, overview, show_send
from .resume import ResumeRefused
from .runner import RunError, RunOptions, resume_run, run
from . import routes
from .task_row import TaskRowError
from .workcopy import WorkCopyError
from .engine.kernel_tool_settings import ToolSettings, ToolSettingsError
from . import instructions as project_instructions
from .skills import install as install_skill, list_installed, remove as remove_skill
from datetime import datetime, timezone


def _task_text(value: str) -> str:
    """``--task`` is literal text, or ``@path`` to read it from a file."""
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"task file not found: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _key_value(value: str) -> tuple[str, str]:
    key, sep, rest = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return key, rest


def _turns(value: str) -> int:
    number = int(value)
    if number < 0 or number > 500:  # same range run's prepare() enforces; resume used to accept any
        raise argparse.ArgumentTypeError("must be 0 (no limit) or between 1 and 500")
    return number


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _add_task_arguments(command: argparse.ArgumentParser, *, for_run: bool = False) -> None:
    """Keep the run contract and chat task options defined in one place."""
    command.add_argument("--repo", required=True, type=Path, help="repository to work on")
    if for_run:
        command.add_argument("--task", required=True, type=_task_text,
                             help='what to fix: text, or @file to read it from a file')
    # Endpoint and model may come from STATEFRAME_ENDPOINT / STATEFRAME_MODEL so a container
    # (or a shell profile) can set them once; an explicit flag always wins.
    command.add_argument("--endpoint", default=os.environ.get("STATEFRAME_ENDPOINT"),
                         required="STATEFRAME_ENDPOINT" not in os.environ,
                         help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1 "
                              "(default: $STATEFRAME_ENDPOINT)")
    command.add_argument("--model", default=os.environ.get("STATEFRAME_MODEL"),
                         required="STATEFRAME_MODEL" not in os.environ,
                         help="model name the server serves (default: $STATEFRAME_MODEL)")
    command.add_argument("--api-key-env", metavar="VAR",
                         help="environment variable holding the API key (never the key itself)")
    command.add_argument("--max-turns", type=_turns, default=None,
                         help="model turn limit; 0 = no limit (default: the profile's step limit)")
    command.add_argument("--test-cmd",
                         help="command that decides pass/fail before, during and after "
                               "(default: python -m pytest -q <checks>)")
    command.add_argument("--check-cmd", help="explicit non-test check command for a repository without tests")
    command.add_argument("--runner-kind", choices=("auto", "pytest", "generic"), default="auto",
                         help="check runner kind (default: detect pytest from --test-cmd)")
    command.add_argument("--protected-test-glob", action="append", default=[], metavar="GLOB",
                          help="also protect existing generic-runner test files matching this relative glob")
    command.add_argument("--protect-tests", choices=("task", "all", "off"),
                         help="which existing tests to lock (default: profile setting)")
    command.add_argument("--check", action="append", default=[], metavar="PATH",
                         help="pytest file the engine runs as its public check, or an additional "
                              "protected file for a generic command (repeatable)")
    command.add_argument("--test-env", action="append", default=[], type=_key_value,
                         metavar="KEY=VALUE",
                         help="environment for your tests (engine checks and --test-cmd), e.g. "
                              "test-only settings; repeatable; never put real secrets here")
    command.add_argument("--pytest-plugins", choices=["installed"],
                         help="let pytest load installed plugins (off by default in the sandbox)")
    command.add_argument("--edit", action="append", default=[], metavar="PATH",
                         help="another file the fix may change (repeatable); with --edit or in a "
                              "repo over 200 source files only --target and --edit files are "
                              "editable, the rest of the repo is read-only context")
    command.add_argument("--hooks", action="append", default=[], metavar="FILE",
                         help="trusted hook configuration file (repeatable)")
    command.add_argument("--no-project-hooks", action="store_true",
                         help="skip the repository's trusted .stateframe/hooks.json file")
    command.add_argument("--target", metavar="PATH",
                         help="starting-file hint (default: first module the checks import)")
    if for_run:
        command.add_argument("--run-dir", type=Path,
                             help="where to keep the run (default: ~/.stateframe/runs/<repo>-<time>)")
    if for_run:
        command.add_argument("--in-place", action="store_true",
                             help="apply the saved candidate after a completed submission and passing checks")
    command.add_argument("--profile", default=profiles.DEFAULT,
                         help=f"switch profile (default {profiles.DEFAULT}; or a profile file)")
    command.add_argument("--test-python", metavar="PATH",
                         help="interpreter for Python checks: the one with your project's "
                              "dependencies and pytest when using pytest (default: the repo's .venv/, venv/ or env/, "
                              "else python3 on PATH)")
    command.add_argument("--python", default=None,
                         help="interpreter for the engine AND your tests (older option; prefer "
                              "--test-python; default: StateFrame's own for the engine)")
    command.add_argument("--keep-checkpoints", action="store_true",
                         help="keep the run's checkpoints after it finishes (default: removed once finished)")
    command.add_argument("--slot", type=int, help="llama.cpp server slot to pin (optional)")
    command.add_argument("--route", action="append", default=[], metavar="ROLE.FIELD=VALUE",
                         help="override one model route field (repeatable)")
    command.add_argument("--routes", type=Path, help="TOML or JSON route configuration file")
    command.add_argument("--timeout", type=float, default=None,
                         help="optional wall-clock limit in seconds (default: no limit; stop with Ctrl-C)")
    command.add_argument("--shell", choices=("off", "read-only", "standard", "unrestricted"),
                         default="standard")
    command.add_argument("--shell-ask", action=argparse.BooleanOptionalAction,
                         default=not for_run)
    command.add_argument("--shell-network", choices=("none", "loopback", "full"), default="loopback")
    for flag in ("shell-allow", "shell-deny", "shell-read", "web-allow", "tools", "mcp"):
        command.add_argument("--" + flag, action="append", default=[])
    command.add_argument("--skill", action="append", default=[], metavar="NAME_OR_PATH",
                         help="load a selected instruction bundle (repeatable)")
    command.add_argument("--trust-skill-tools", action="store_true",
                         help="offer selected skills' command definitions")
    project = command.add_mutually_exclusive_group()
    project.add_argument("--instructions", nargs="?", const="auto", default=None, metavar="FILE",
                         help="read project instructions: FILE, or with no value the repo's AGENTS.md, "
                              "else CLAUDE.md (default: off)")
    project.add_argument("--no-instructions", action="store_true",
                         help="do not read project instructions (the default)")
    command.add_argument("--shell-sandbox", choices=("on", "off"), default="on")
    for flag in ("run-env", "git-tool", "web"):
        command.add_argument("--" + flag, action=argparse.BooleanOptionalAction, default=None)
    if for_run:
        command.add_argument("--quiet", action="store_true", help="no live turn log")
    if for_run:
        command.add_argument("--json", action="store_true", help="print the summary as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stateframe",
        description="Run the StateFrame coding agent on a local repository.")
    parser.add_argument("--version", action="version", version=f"stateframe {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="fix a task in a repository (works on a safe copy)")
    _add_task_arguments(run_cmd, for_run=True)

    resume_cmd = commands.add_parser(
        "resume", help="continue an interrupted run where it stopped (same engine, same settings)")
    resume_cmd.add_argument("run_dir", type=Path)
    resume_cmd.add_argument("--endpoint", help="model server base URL, if it moved (default: the run's)")
    resume_cmd.add_argument("--model", default=None,
                            help="continue under this model name, when the endpoint serves the SAME model under "
                                 "another name (default: the run's model name)")
    resume_cmd.add_argument("--api-key-env", metavar="VAR",
                            help="environment variable holding the API key (default: the run's)")
    resume_cmd.add_argument("--max-turns", type=_turns, default=None,
                            help="raise the step limit (0 = no limit; default: the run's)")
    resume_cmd.add_argument("--timeout", type=float, default=None,
                            help="optional wall-clock limit in seconds for this resume")
    resume_cmd.add_argument("--replay-external", action="store_true",
                            help="explicitly repeat external tool operations and verify their results")
    resume_cmd.add_argument("--keep-checkpoints", action="store_true",
                            help="keep the run's checkpoints after it finishes (default: removed once finished)")
    resume_cmd.add_argument("--quiet", action="store_true", help="no live turn log")
    resume_cmd.add_argument("--json", action="store_true", help="print the summary as JSON")
    resume_cmd.add_argument("--route", action="append", default=[], metavar="ROLE.FIELD=VALUE",
                            help="relocate a role endpoint or rename its model")
    resume_cmd.add_argument("--routes", type=Path, help="TOML or JSON route configuration file")

    config_cmd = commands.add_parser("config", help="create a commented route configuration")
    config_actions = config_cmd.add_subparsers(dest="config_command", required=True)
    config_init = config_actions.add_parser("init", help="write an inactive routes.toml example")
    config_init.add_argument("--path", type=Path, default=Path.home() / ".stateframe" / "routes.toml")

    inspect_cmd = commands.add_parser("inspect", help="show the frames and request bytes of a run")
    inspect_cmd.add_argument("run_dir", type=Path)
    inspect_cmd.add_argument("--send", type=_positive, help="one provider send (1-based)")
    view = inspect_cmd.add_mutually_exclusive_group()
    view.add_argument("--frame", action="store_true", help="the typed StateFrame, pretty JSON")
    view.add_argument("--bytes", action="store_true", help="the exact request bytes sent")
    view.add_argument("--messages", action="store_true", help="every message of the request")

    chat_cmd = commands.add_parser("chat", help="interactive session (preview): tasks and follow-ups")
    _add_task_arguments(chat_cmd)
    chat_cmd.add_argument("--session-dir", type=Path,
                          help="where to keep the session (default: ~/.stateframe/sessions/<repo>-<time>)")
    skill_cmd = commands.add_parser("skill", help="manage installed instruction bundles")
    skill_actions = skill_cmd.add_subparsers(dest="skill_action", required=True)
    skill_install = skill_actions.add_parser("install", help="validate and install a bundle")
    skill_install.add_argument("path", type=Path)
    skill_install.add_argument("--force", action="store_true")
    skill_actions.add_parser("list", help="list installed bundles")
    skill_remove = skill_actions.add_parser("remove", help="remove an installed bundle")
    skill_remove.add_argument("name")

    hooks_cmd = commands.add_parser("hooks", help="manage explicitly trusted project hooks")
    hooks_subcommands = hooks_cmd.add_subparsers(dest="hooks_command", required=True)
    trust_cmd = hooks_subcommands.add_parser("trust", help="trust or withdraw a repository hook file")
    trust_cmd.add_argument("--repo", required=True, type=Path, help="repository containing .stateframe/hooks.json")
    trust_cmd.add_argument("--remove", action="store_true", help="withdraw trust for this repository hook file")
    return parser


def _run_options(args: argparse.Namespace, endpoint: Endpoint) -> RunOptions:
    if args.shell_sandbox == "off" and args.shell != "unrestricted":
        raise ToolSettingsError("--shell-sandbox off requires --shell unrestricted")
    host_authorization = None
    if args.shell_sandbox == "off":
        print("WARNING: SHELL SANDBOX OFF; COMMANDS CAN ACCESS THE HOST", file=sys.stderr)
        host_authorization = {"granted_at": datetime.now(timezone.utc).isoformat(),
                              "flags": ["--shell unrestricted", "--shell-sandbox off"]}
    settings = ToolSettings(shell=args.shell, shell_ask=args.shell_ask,
                            shell_network=args.shell_network, shell_allow=tuple(args.shell_allow),
                            shell_deny=tuple(args.shell_deny), shell_read=tuple(args.shell_read),
                            host_authorization=host_authorization,
                            run_env=args.run_env if args.run_env is not None else args.shell != "off",
                            git_tool=args.git_tool if args.git_tool is not None else True,
                            web=args.web if args.web is not None else False,
                            web_allow=tuple(args.web_allow),
                            tools_files=tuple({"path": str(Path(path).expanduser().resolve()), "digest": "pending"}
                                              for path in args.tools),
                            mcp_files=tuple({"path": str(Path(path).expanduser().resolve()), "digest": "pending"}
                                            for path in args.mcp))
    return RunOptions(repo=args.repo, task=getattr(args, "task", ""), endpoint=endpoint,
                       max_turns=args.max_turns, test_cmd=args.test_cmd, check_cmd=args.check_cmd,
                       checks=args.check,
                      runner_kind=args.runner_kind,
                       protected_test_globs=args.protected_test_glob,
                       protect_tests=args.protect_tests,
                      target=args.target, edit_paths=args.edit,
                      test_env=dict(args.test_env), pytest_plugins=args.pytest_plugins,
                      run_dir=getattr(args, "run_dir", None),
                      in_place=getattr(args, "in_place", False),
                      profile=args.profile, python=args.python or sys.executable,
                      python_explicit=args.python is not None, test_python=args.test_python,
                      slot=args.slot, keep_checkpoints=args.keep_checkpoints,
                      timeout_s=args.timeout,
                      quiet=getattr(args, "quiet", False) or getattr(args, "json", False),
                      tool_settings=settings, skills=tuple(args.skill),
                      trust_skill_tools=args.trust_skill_tools,
                      instructions=("off" if args.no_instructions else
                                    "auto" if args.instructions == "auto" else
                                    str(Path(args.instructions).expanduser().resolve()) if args.instructions
                                    else project_instructions.DEFAULT_MODE),
                      hooks_files=tuple(Path(path).expanduser().resolve() for path in args.hooks),
                      no_project_hooks=args.no_project_hooks,
                      hooks_repo=args.repo.resolve(),
                      route_cli=args.route, routes_file=args.routes)


def _print_summary(summary: dict) -> None:
    diff = Path(summary["run_dir"], "final.diff").read_text()
    print()
    if summary.get("runtime_outcome") == "hook_containment" and summary.get("applied_in_place"):
        title = "=== final diff (applied before HOOK_CONTAINMENT) ==="
    else:
        title = "=== final diff ===" if summary.get("succeeded") else "=== partial diff (not promoted) ==="
    print(title)
    print(diff if diff else "(no changes)")
    before, after = summary["tests_before"], summary["tests_after"]
    print("=== tests ===" if summary.get("check_source") == "tests" else "=== checks ===")
    print(f"before: {'PASS' if before['passed'] else 'FAIL' if before['passed'] is False else 'NOT RUN'}  after: "
          f"{summary.get('check_outcome', 'not_run').upper()}  ({after['command']})")
    print("=== run summary ===")
    cached = summary.get("cached_tokens")
    share = summary.get("cached_share")
    result_line = summary.get("verification")
    if result_line and summary.get("verification_detail"):
        result_line += f" ({summary['verification_detail']})"
    if result_line and summary.get("test_files_changed"):
        changed = summary["test_files_changed"]
        test_paths = ", ".join(item["path"] for item in changed)
        result_line += f"; {len(changed)} test file{'s' if len(changed) != 1 else ''} changed: {test_paths}"
    if result_line and summary.get("test_lock_note"):
        result_line += f"; {summary['test_lock_note']}"
    rows = [
        ("outcome", f"runtime={summary['runtime_outcome']} checks={summary['check_outcome']} "
         f"artifact={summary['artifact']}"),
        *([("result", result_line)] if result_line else []),
        *([("test evidence", json.dumps(summary["wrong_test_evidence"], sort_keys=True))]
          if summary.get("wrong_test_evidence") else []),
        ("engine result", summary.get("engine_result")),
        *([("engine error", summary["engine_error_reason"])] if summary.get("engine_error_reason") else []),
        *([("hook containment", summary["hook_containment"])] if summary.get("hook_containment") else []),
        ("model requests", summary.get("requests")),
        ("provider sends", summary.get("provider_sends")),
        ("request bytes sent", summary.get("request_bytes_sent")),
        ("largest request", f"{summary.get('max_request_bytes')} bytes"),
        ("prompt tokens sent", summary.get("prompt_tokens_sent")),
        ("cached prompt tokens", "not reported by server" if cached is None
         else f"{cached} ({share:.1%})" if share is not None else cached),
        ("completion tokens", summary.get("completion_tokens")),
        ("wall time", f"{summary.get('wall_seconds')} s"),
        ("files changed", ", ".join(summary.get("files_changed") or []) or "none"),
        *([("product files changed", ", ".join(summary.get("product_files_changed") or []) or "none"),
           ("test files changed", f"{len(summary['test_files_changed'])}: " +
            (", ".join(item["path"] for item in summary["test_files_changed"]) or "none"))]
          if "test_files_changed" in summary else []),
        ("run directory", summary["run_dir"]),
    ]
    resumed = summary.get("resume")
    if isinstance(resumed, dict):
        how = ("faithful replay" if resumed.get("faithful") else
               f"replay went live early at recorded exchange {resumed.get('went_live_at_recorded_exchange')}"
               if resumed.get("went_live_at_recorded_exchange") else "replay incomplete")
        rows.append(("resume", f"attempt {resumed.get('attempt')}: replayed {resumed.get('replayed')}/"
                               f"{resumed.get('recorded_exchanges')} recorded exchanges ({how})"))
    if "applied_in_place" in summary:
        rows.append(("applied to repo", ", ".join(summary["applied_in_place"]) or
                     summary.get("in_place_skipped", "nothing")))
    for label, value in rows:
        print(f"{label:>22}: {value}")
    costs = summary.get("route_costs")
    if isinstance(costs, dict):
        print("=== model routes ===")
        print("role          sends discarded   prompt   cached completion estimated cost")
        for role, row in [*costs["roles"].items(), ("total", costs["total"])]:
            cost = row["estimated_cost"]
            display = f"${cost:.6f}" if cost is not None else "unknown"
            counts = {key: "unknown" if row[key] is None else row[key]
                      for key in ("escalated", "prompt_tokens", "cached_tokens", "completion_tokens")}
            print(f"{role:<13} {row['requests']:>5} {counts['escalated']:>9} "
                  f"{counts['prompt_tokens']:>8} {counts['cached_tokens']:>8} "
                  f"{counts['completion_tokens']:>10} {display:>14}")
        if costs.get("unattributed_sends"):
            print(f"{costs['unattributed_sends']} send(s) missing route records; totals unknown")
        if costs.get("all_at_edit_rates") is not None:
            print(f"all at edit rates: ${costs['all_at_edit_rates']:.6f}; "
                  f"estimated savings: ${costs['estimated_savings']:.6f}")
    if summary.get("succeeded"):
        print(result_line or
              "checks passed on the saved candidate; correctness not independently established")
    terminal = summary.get("engine_terminal") or {}
    labels = terminal.get("labels") if isinstance(terminal.get("labels"), dict) else {}
    facts = terminal.get("done_facts") if isinstance(terminal.get("done_facts"), dict) else {}
    print(f"engine evidence: certification={labels.get('certification', 'unknown')}; "
          f"coverage={labels.get('coverage', 'unknown')}; readiness={labels.get('readiness', 'unknown')}; "
          f"unresolved requirements={facts.get('unresolved_gating_items', 'unknown')}")
    if summary.get("apply_error"):
        print(f"apply error: {summary['apply_error']}")


def main(argv: list[str] | None = None) -> int:
    # Live progress must reach a pipe or log file as it happens, not only when the run ends.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # a replaced stdout (tests, embedding) keeps its own buffering
        pass
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            if args.send is None:
                print(overview(args.run_dir.resolve()))
                return 0
            what = "frame" if args.frame else "bytes" if args.bytes else \
                "messages" if args.messages else "summary"
            out = show_send(args.run_dir.resolve(), args.send, what=what)
            if isinstance(out, bytes):
                sys.stdout.buffer.write(out)
                sys.stdout.flush()
            else:
                print(out)
            return 0
        if args.command == "config":
            path = args.path.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                output.write(routes.example())
            print(f"wrote {path}")
            return 0
        if args.command == "hooks":
            from .hook_host import trust_project_hooks
            trust_project_hooks(args.repo, remove=args.remove)
            return 0
        if args.command == "resume":
            endpoint = None
            if args.endpoint or args.api_key_env:
                spec = json.loads((args.run_dir / "spec.json").read_text())
                endpoint = Endpoint.from_args(
                    args.endpoint or spec["chat_url"].removesuffix(CHAT_SUFFIX), args.model or spec["model"],
                    args.api_key_env or spec.get("api_key_env"))
            summary = resume_run(args.run_dir, endpoint=endpoint, max_turns=args.max_turns, model=args.model,
                                 timeout_s=args.timeout, quiet=args.quiet or args.json,
                                 route_cli=args.route, routes_file=args.routes,
                                 keep_checkpoints=args.keep_checkpoints, replay_external=args.replay_external,
                                 printer=(lambda *_: None) if args.json else print)
            if args.json:
                print(json.dumps(summary, indent=1))
            else:
                _print_summary(summary)
            return summary["exit_code"]
        if args.command == "skill":
            if args.skill_action == "install":
                skill = install_skill(args.path, force=args.force)
                print(f"installed {skill.name} ({skill.skill_md_sha256[:12]})")
            elif args.skill_action == "list":
                for skill in list_installed():
                    print(f"{skill.name}\t{skill.description}\ttools={'yes' if skill.tools_json_sha256 else 'no'}\t{skill.skill_md_sha256}")
            else:
                remove_skill(args.name)
                print(f"removed {args.name}")
            return 0
        endpoint = Endpoint.from_args(args.endpoint, args.model, args.api_key_env)
        if args.command == "chat":
            from . import chat
            base = _run_options(args, endpoint)
            session = chat.Session(repo=args.repo.resolve(), base=base,
                                   home=(args.session_dir or chat.default_home(args.repo)).resolve())
            return chat.repl(session, print_summary=lambda p: _print_summary(json.loads(p.read_text())))
        options = _run_options(args, endpoint)
        summary = run(options, printer=(lambda *_: None) if args.json else print)
        if args.json:
            print(json.dumps(summary, indent=1))
        else:
            _print_summary(summary)
        return summary["exit_code"]
    except ResumeRefused as error:
        print(f"stateframe: cannot resume: {error}", file=sys.stderr)
        return 2
    except (EndpointError, RunError, TaskRowError, WorkCopyError, InspectError, ValueError, OSError) as error:
        print(f"stateframe: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stateframe: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
