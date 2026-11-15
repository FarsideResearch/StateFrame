"""Process-wide capabilities for one compiled parallel subframe."""
from __future__ import annotations

import re
import secrets
import shlex
import hashlib
import json
import os
import signal
import shutil
import sys
import tempfile
import fnmatch
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

if __package__:
    from .kernel_parallel_records import CheckReceipt, Refusal, UnitSpec
else:
    from kernel_parallel_records import CheckReceipt, Refusal, UnitSpec


UNIT_BLOCKED_TOOL = {
    "type": "function",
    "function": {
        "name": "subframe_blocked",
        "description": "End this subframe when its declared contract cannot be completed.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason_code": {"type": "string", "enum": [
                    "NEEDS_UNOWNED_FILE", "INTERFACE_WRONG", "CONTRACT_IMPOSSIBLE",
                    "CHECK_WRONG", "OTHER",
                ]},
                "detail": {"type": "string", "maxLength": 300},
            },
            "required": ["reason_code"],
            "additionalProperties": False,
        },
    },
}

_BASE_TOOLS = frozenset({"read", "grep", "str_replace", "insert_before",
                         "file_create", "done", "subframe_blocked"})
_MUTATING_TOOLS = frozenset({"str_replace", "insert_before", "file_create"})
_PYTEST_FRAME = re.compile(r'^\s*(?:File\s+)?(?:"?([^"\s:]+\.py)"?):(\d+)(?::|\s|$)')
_PYTHON_TRACE_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)')
_PYTEST_FILE = re.compile(r'^([^\s:][^:]*\.py[a-z]?):(\d+):')
# Fail-closed projection pieces: only structure survives, never free text.
_PYTEST_EXIT = re.compile(r'^pytest exit (\d+)\b')
_PYTEST_SUMMARY_PART = re.compile(
    r'\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)\b')
_PYTEST_DURATION = re.compile(r'\bin \d+(?:\.\d+)?s\b')
_PYTEST_SHORT_SUMMARY = re.compile(r'^(FAILED|ERROR) (\S+)((?: - ).*)?$')
_CAPTURED_HEADER = re.compile(r'^-+ Captured (stdout|stderr|log) (setup|call|teardown) -+$')
_NODE_PATH = re.compile(r'^[A-Za-z0-9_./\-]+$')
_ERROR_LINE = re.compile(r'^E(?:\s+(.*))?$')
_EXCEPTION_NAME = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?::|$)')
_IDENTIFIER = re.compile(r'^[A-Za-z_<][A-Za-z0-9_.<>]*$')
_WITHHELD_FRAME_PREFIX = "[frame in withheld file "
_MAX_NAME = 80
_PYTEST_SUMMARY = re.compile(r'\b\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed)\b')
_PYTEST_COUNT = re.compile(r'\b(\d+)\s+(passed|failed|errors?)\b')
_CHECK_EXIT = re.compile(r'^(?:pytest|check) exit (\d+)\b', re.MULTILINE)


def _scrub_error_group(bodies: Sequence[str], mentions_hidden: Callable[[str], bool]) -> str:
    """Collapse a run of pytest `E` lines to the exception type name only."""
    if any('File "' in body or mentions_hidden(body) for body in bodies):
        return "E   [exception detail withheld]"
    first = bodies[0].strip() if bodies else ""
    if first.startswith("assert ") or first == "assert":
        return "E   assert [detail withheld]"
    name = _EXCEPTION_NAME.match(first)
    if name and len(name.group(1)) <= _MAX_NAME:
        return f"E   {name.group(1)}: [message withheld]"
    return "E   [exception detail withheld]"


def _scrub_frame(line: str, path: str, number: str) -> str:
    """Rebuild a visible frame line from its path, line number and identifier-shaped tail."""
    stripped = line.strip()
    if stripped.startswith('File "'):
        tail = stripped.split(f'", line {number}', 1)[1] if f'", line {number}' in stripped else ""
        name = tail[len(", in "):].strip() if tail.startswith(", in ") else ""
        if name and len(name) <= _MAX_NAME and _IDENTIFIER.match(name):
            return f'  File "{path}", line {number}, in {name}'
        return f'  File "{path}", line {number}'
    tail = stripped.split(f"{number}:", 1)[1].strip() if f"{number}:" in stripped else ""
    if tail.startswith("in "):
        tail = tail[3:].strip()
        if tail and len(tail) <= _MAX_NAME and _IDENTIFIER.match(tail):
            return f"{path}:{number}: in {tail}"
    elif tail and len(tail) <= _MAX_NAME and _EXCEPTION_NAME.match(tail) and ":" not in tail:
        return f"{path}:{number}: {tail}"
    return f"{path}:{number}:"


def _scrub_node_id(node: str) -> str:
    """Keep a pytest node id's path and identifier segments; parametrize ids are free text."""
    base, _, params = node.partition("[")
    segments = base.split("::")
    if (not _NODE_PATH.match(segments[0])
            or any(not seg or len(seg) > _MAX_NAME or not _IDENTIFIER.match(seg)
                   for seg in segments[1:])):
        return "[node id withheld]"
    return base + ("[...]" if params or node.endswith("[") else "")


def _scrub_structured_line(line: str, mentions_hidden: Callable[[str], bool]) -> str | None:
    """Rebuild the pytest lines that carry structure; None for anything else."""
    exit_match = _PYTEST_EXIT.match(line)
    if exit_match:
        return f"pytest exit {exit_match.group(1)}"
    summary = _PYTEST_SHORT_SUMMARY.match(line)
    if summary:
        verdict, node, message = summary.groups()
        if mentions_hidden(node):
            return f"{verdict} [node id withheld]"
        return f"{verdict} {_scrub_node_id(node)}" + (" - [message withheld]" if message else "")
    parts = _PYTEST_SUMMARY_PART.findall(line)
    if parts:
        duration = _PYTEST_DURATION.search(line)
        text = ", ".join(parts) + (f" {duration.group(0)}" if duration else "")
        return f"{'=' * 27} {text} {'=' * 27}" if line.startswith("=") else text
    return None


class UnitCheckPermitError(RuntimeError):
    """A subframe check was refused because its shared permit is unavailable."""


class UnitCheckCleanupError(RuntimeError):
    """A subframe check could not confirm descendant termination."""


class UnitCheckContainmentError(UnitCheckCleanupError):
    """A detached check descendant remained live after bounded cleanup."""


def new_check_token() -> str:
    return secrets.token_hex(16)


def _matching_check_processes(token: str) -> tuple[int, ...]:
    """Find macOS check descendants after setsid and orphan reparenting."""
    marker = f"STATEFRAME_CHECK_TOKEN={token}"
    if sys.platform != "darwin":
        raise UnitCheckCleanupError("check descendant inspection is unavailable on this platform")
    try:
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                ["ps", "-Eww", "-axo", "pid=,uid=,command="], stdout=output,
                stderr=subprocess.DEVNULL, timeout=5, check=True)
            if output.tell() > 16 * 1024 * 1024:
                raise UnitCheckCleanupError("check descendant process listing exceeds 16 MiB")
            output.seek(0)
            listing = output.read().decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise UnitCheckCleanupError(f"cannot inspect check descendants: {error}") from error
    matches: list[int] = []
    for line in listing.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdecimal() and parts[1].isdecimal():
            if int(parts[1]) == os.getuid() and marker in parts[2].split():
                matches.append(int(parts[0]))
    return tuple(sorted(matches))


def _reap_check_token(token: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise UnitCheckCleanupError("check descendant token is invalid")
    if sys.platform == "linux":
        # Without a returned sandbox PID there is no namespace-init exit to
        # verify. An interrupted launch cannot publish a passing check.
        raise UnitCheckCleanupError("check PID namespace cleanup cannot be verified")
    # A detached macOS process has no launch-pinned signal authority here.
    # Keep the disposable copy until a short-lived descendant has exited;
    # the receipt remains invalid even after it disappears.
    deadline = time.monotonic() + 10.0
    clear = 0
    detached: tuple[int, ...] = ()
    while time.monotonic() < deadline:
        matches = _matching_check_processes(token)
        if matches:
            detached = matches
            clear = 0
            time.sleep(0.05)
            continue
        clear += 1
        if clear == 2:
            if detached:
                raise UnitCheckCleanupError(
                    f"check containment unverified: detached descendants {detached} exited after the check")
            return
        time.sleep(0.05)
    raise UnitCheckContainmentError(
        f"check containment unverified: detached descendants {detached} remained after the check")


def _terminate_check_token(token: str) -> str | None:
    """Stop macOS escapees by their kernel PID generation before leader reaping."""
    if sys.platform != "darwin":
        return
    if __package__:
        from .kernel_exec_tracker import ProcessTable, signal_mac_verified_process
    else:
        from kernel_exec_tracker import ProcessTable, signal_mac_verified_process
    table = ProcessTable()
    found = _matching_check_processes(token)
    for pid in found:
        try:
            before = table.identity(pid)
            if before is None or before.zombie:
                continue
            if pid not in _matching_check_processes(token):
                continue
            current = table.identity(pid)
            if current is None or current.start != before.start:
                raise UnitCheckCleanupError(
                    f"check containment unverified: detached descendant {pid} changed identity")
            signal_mac_verified_process(current, signal.SIGKILL)
        except (OSError, RuntimeError, ValueError) as error:
            raise UnitCheckCleanupError(
                f"check containment unverified: cannot terminate detached descendant {pid}: {error}") from error
    _reap_check_token(token)
    if found:
        return f"check result invalid: detached descendants {found} terminated after the check"
    return None


def _group_is_only_zombies(pgid: int) -> bool:
    """A zombie has exited even when an unrelated PID 1 never reaps it."""
    if sys.platform == "linux":
        if __package__:
            from .kernel_exec_tracker import ProcessTable
        else:
            from kernel_exec_tracker import ProcessTable
        try:
            members = [row for row in ProcessTable().snapshot().values() if row.pgid == pgid]
        except (OSError, RuntimeError, ValueError) as error:
            raise UnitCheckCleanupError(f"cannot inspect check process group: {error}") from error
        return bool(members) and all(row.zombie for row in members)
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["ps", "-axo", "pgid=,stat="], capture_output=True,
                                    text=True, timeout=5, check=True)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise UnitCheckCleanupError(f"cannot inspect check process group: {error}") from error
        if len(result.stdout) > 16 * 1024 * 1024:
            raise UnitCheckCleanupError("check process listing exceeds 16 MiB")
        states = [parts[1] for line in result.stdout.splitlines()
                  if len(parts := line.split()) == 2 and parts[0].isdecimal()
                  and int(parts[0]) == pgid]
        return bool(states) and all(state.startswith("Z") for state in states)
    return False


def reap_check_group(pid: int | None, *, token: str | None = None) -> None:
    """Confirm the Linux namespace init exited or reap the macOS session."""
    if not isinstance(pid, int) or pid <= 0:
        if token is not None:
            _reap_check_token(token)
        raise UnitCheckCleanupError("check process group identity is unavailable")
    if token is not None and not re.fullmatch(r"[0-9a-f]{32}", token):
        raise UnitCheckCleanupError("check descendant token is invalid")
    # The sandbox supervisor already performed launch-authorized cleanup before
    # returning its result. A PID alone cannot authorize a second group signal.
    if __package__:
        from .kernel_exec_tracker import ProcessTable
    else:
        from kernel_exec_tracker import ProcessTable
    deadline = time.monotonic() + 2.0
    group_gone = False
    while time.monotonic() < deadline:
        try:
            live = [row for row in ProcessTable().snapshot().values()
                    if row.pgid == pid and not row.zombie]
        except (OSError, RuntimeError, ValueError) as error:
            raise UnitCheckCleanupError(f"cannot inspect check process group: {error}") from error
        if not live:
            group_gone = True
            break
        time.sleep(0.02)
    # Subframe checks on Linux run as PID 1 in a private namespace. Once the
    # sandbox process is reaped and its group is gone, the namespace init has
    # exited; the kernel has killed every member, including detached children.
    scan_error: UnitCheckCleanupError | None = None
    if token is not None and sys.platform != "linux":
        try:
            _reap_check_token(token)
        except UnitCheckCleanupError as error:
            scan_error = error
    if not group_gone:
        raise UnitCheckCleanupError("check containment unverified: process group remained after cleanup")
    if scan_error is not None:
        raise scan_error


def _refusal(reason: str, detail: str, path: str = "") -> Refusal:
    return Refusal(reason, detail[:300], (path,) if path else (), (), ())


def _path(value: str) -> str:
    normalized = PurePosixPath(value).as_posix()
    if not value or value.startswith("/") or normalized in {"", "."} or ".." in PurePosixPath(value).parts:
        return ""
    return normalized.removeprefix("./")


@dataclass(frozen=True)
class UnitPolicy:
    spec: UnitSpec

    @property
    def allowed_tools(self) -> frozenset[str]:
        if self.spec.contract.check.command_kind == "none" or not self.spec.contract.check.selectors:
            return _BASE_TOOLS
        return _BASE_TOOLS | {"run_tests"}

    @property
    def mutating_tools(self) -> frozenset[str]:
        return _MUTATING_TOOLS

    def digest(self) -> str:
        return self.spec.policy_digest

    def check_binding_command(self) -> tuple[str, str]:
        """Bind a frozen check without the disposable clone's command path."""
        return ("subframe-frozen-check", self.spec.contract.check.digest())

    def matches_check_resolver(
            self, arguments: Mapping[str, object], receipts: Sequence[Mapping[str, object]],
    ) -> bool:
        """Accept a catalog resolver only for the executed frozen recipe."""
        identity = arguments.get("check_identity")
        if not isinstance(identity, str) or not identity:
            return False
        if any(arguments.get(key) != identity
               for key in ("check_id", "selector", "target")):
            return False
        if __package__:
            from .kernel_catalog import normalize_receipt
        else:
            from kernel_catalog import normalize_receipt
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            selectors = receipt.get("selectors")
            if (not isinstance(selectors, (list, tuple))
                    or tuple(selectors) != self.spec.contract.check.selectors):
                continue
            if normalize_receipt(receipt).get("check_identity") == identity:
                return True
        return False

    def first_request_fits(self, tokens: int, *, owned_source_complete: bool) -> bool:
        return owned_source_complete and tokens <= self.spec.limits.unit_request_tokens

    def authorize_tool(self, name: str) -> Refusal | None:
        if name in self.allowed_tools:
            return None
        return _refusal("SCOPE_VIOLATION", f"tool {name!r} is not in this subframe's capability set")

    def authorize_check_target(self, arguments: Mapping[str, object]) -> Refusal | None:
        """Refuse a run_tests target that names anything but this subframe's frozen check.

        A subframe always executes its frozen recipe, so a different target would otherwise
        run the frozen check while the model believes it ran something else.  Catalog
        resolver arguments are validated separately (``matches_check_resolver``).
        """
        check = self.spec.contract.check
        target = arguments.get("target")
        if (check.command_kind != "pytest" or not check.selectors
                or not isinstance(target, str) or not target.strip()
                or any(arguments.get(key) for key in ("unknown_id", "check_id", "check_identity"))):
            return None
        wanted = target.strip()
        # Accept the selector itself, its file, or its bare node name (the tool's documented forms).
        allowed = set(check.selectors)
        for selector in check.selectors:
            allowed.add(selector.split("::", 1)[0])
            allowed.add(selector.rsplit("::", 1)[-1])
        if wanted in allowed:
            return None
        return _refusal("SCOPE_VIOLATION",
                        f"run_tests target {wanted!r} is outside this subframe's frozen check; "
                        f"run {', '.join(check.selectors)} or call run_tests without a target")

    def authorize_mutation(self, op: str, path: str, *, created_by_unit: bool) -> Refusal | None:
        key = _path(path)
        if not key:
            return _refusal("SCOPE_VIOLATION", "invalid subframe mutation path", path)
        if op == "edit" and (key in self.spec.contract.owns or
                             (key in self.spec.contract.creates and created_by_unit)):
            return None
        if op == "create" and key in self.spec.contract.creates and not created_by_unit:
            return None
        return _refusal("SCOPE_VIOLATION", f"{op} is not authorized for {key}", key)

    def visible(self, path: str) -> bool:
        return _path(path) in self.spec.visible_paths

    def visible_scope(self, path: str) -> bool:
        if path in {"", "."}:
            return True
        key = _path(path)
        return bool(key and (self.visible(key) or any(
            item.startswith(key + "/") for item in self.spec.visible_paths)))

    def directory_page(self, root: Path, directory: Path, display: str, *,
                       cursor: object = None, page_size: object = None) -> object:
        """Page only direct children leading to visible subframe files."""
        if __package__:
            from .kernel_navigation_pages import (
                DEFAULT_DIRECTORY_PAGE_SIZE, _bounded_int, _page_rows, _page_text,
                NavigationPage,
            )
        else:
            from kernel_navigation_pages import (
                DEFAULT_DIRECTORY_PAGE_SIZE, _bounded_int, _page_rows, _page_text,
                NavigationPage,
            )
        size = _bounded_int(page_size, DEFAULT_DIRECTORY_PAGE_SIZE)
        children = []
        for child in directory.iterdir():
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                continue
            if child.is_file() and self.visible(relative):
                children.append(child.name)
            elif child.is_dir() and self.visible_scope(relative):
                children.append(child.name + "/")
        rows = [{"name": name} for name in sorted(set(children))]
        shown, metadata = _page_rows(rows, cursor=cursor, size=size)
        metadata = {**metadata, "path": display, "page_size": size}
        rendered = [str(row["name"]) for row in shown]
        text = (f"{display} is a DIRECTORY. Contents:\n"
                + _page_text("directory", metadata, rendered))
        return NavigationPage(text, metadata)

    def project_check_output(self, raw: str, runner: str, *, clone_root: Path | None = None) -> str:
        """Project diagnostics while retaining path-qualified pytest evidence.

        The check runs on a full clone of the snapshot, so any free text in pytest's
        output (``E`` message bodies, captured streams, parametrize ids, short-summary
        messages, inline prints) can carry the contents of a hidden or withheld file.
        Attribution of that text is unknowable, so the projection fails closed: it keeps
        only structure (exit code, counts, node ids, exception type names and the
        locations of visible frames) and replaces everything else with a marker.
        """
        prefix = str(Path(clone_root)) + os.sep if clone_root is not None else None
        lines = [line.replace(prefix, "") if prefix else line for line in raw.splitlines()]
        if runner != "pytest":
            if not self.spec.hidden_paths:
                return "best-effort projection\n" + "\n".join(
                    line for line in lines[-40:] if not self._mentions_hidden(line))
            return self._project_command_closed(lines)
        if not self.spec.hidden_paths and not any(self._withheld_frame(line) for line in lines):
            return self._project_pytest_open(lines)
        projected: list[str] = []
        captured = False
        # pytest prints an exception's `E` lines BEFORE the `path:line: ExcType` location of
        # the frame that raised it.  Hold them until that frame is known: a withheld frame
        # drops the group, a visible frame (or the end of the report) emits its scrubbed form.
        held_errors: list[str] = []
        held_tainted = False

        def flush(emit: bool) -> None:
            nonlocal held_tainted
            if held_errors and emit and not held_tainted:
                projected.append(_scrub_error_group(held_errors, self._mentions_hidden))
            held_errors.clear()
            held_tainted = False

        for line in lines:
            header = _CAPTURED_HEADER.match(line)
            if header:
                flush(True)
                projected.append(f"[captured {header.group(1)} {header.group(2)} withheld]")
                captured = True
                continue
            if captured and (line.startswith("----------------------------- ")
                             or line.startswith("=========================== ")
                             or _PYTEST_SHORT_SUMMARY.match(line) or _PYTEST_EXIT.match(line)):
                captured = False
            if captured:
                continue
            error = _ERROR_LINE.match(line)
            if error:
                if not held_errors:
                    # Classic traceback form prints the frame first; `E` lines that
                    # directly follow a withheld frame belong to it.
                    held_tainted = bool(projected) and projected[-1].startswith(_WITHHELD_FRAME_PREFIX)
                held_errors.append(error.group(1) or "")
                continue
            match = (_PYTEST_FRAME.match(line) or _PYTEST_FILE.match(line)
                     or _PYTHON_TRACE_FRAME.match(line))
            if match:
                path, number = match.groups()
                if not self.visible(path):
                    flush(False)
                    projected.append(f"{_WITHHELD_FRAME_PREFIX}{path}:{number}]")
                else:
                    flush(True)
                    projected.append(_scrub_frame(line, path, number))
                continue
            structured = _scrub_structured_line(line, self._mentions_hidden)
            if structured is not None:
                flush(True)
                projected.append(structured)
            # Any other line (source text, separators, chained-exception notes, inline
            # prints) is free text and is dropped without flushing the held group.
        flush(True)
        return "\n".join(projected)

    def _withheld_frame(self, line: str) -> bool:
        match = (_PYTEST_FRAME.match(line) or _PYTEST_FILE.match(line)
                 or _PYTHON_TRACE_FRAME.match(line))
        return bool(match) and not self.visible(match.group(1))

    def _project_command_closed(self, lines: Sequence[str]) -> str:
        """A command check with withheld files present: exit status and counts only."""
        exit_match = next((_CHECK_EXIT.match(line) for line in lines if _CHECK_EXIT.match(line)), None)
        projected = [f"check exit {exit_match.group(1) if exit_match else 'unknown'}"]
        for line in lines:
            parts = _PYTEST_SUMMARY_PART.findall(line)
            if parts:
                projected.append(", ".join(parts))
                break
        projected.append("[command output withheld]")
        return "\n".join(projected)

    def _project_pytest_open(self, lines: Sequence[str]) -> str:
        """Pytest output from a workspace with no withheld files and only visible frames.

        Nothing withheld can be present, so assertion and exception detail is kept; only
        the hidden-path name filter and withheld-frame handling remain for defence in depth.
        """
        projected: list[str] = []
        captured = False
        section: list[str] = []
        withheld_seen = False
        # pytest prints an exception's `E` lines BEFORE the `path:line: ExcType` location of
        # the frame that raised it.  Hold them until that frame is known: when it is withheld
        # source, the message is withheld too.
        held_errors: list[str] = []
        for line in lines:
            if line.startswith("----------------------------- Captured "):
                if section and not any(self._mentions_hidden(item) for item in section):
                    projected.extend(section)
                section = [line]
                captured = True
                continue
            if captured and (line.startswith("----------------------------- ")
                             or line.startswith("=========================== ")):
                if not any(self._mentions_hidden(item) for item in section):
                    projected.extend(section)
                section = []
                captured = False
            if captured:
                section.append(line)
                continue
            match = (_PYTEST_FRAME.match(line) or _PYTEST_FILE.match(line)
                     or _PYTHON_TRACE_FRAME.match(line))
            if match:
                path, number = match.groups()
                if not self.visible(path):
                    withheld_seen = True
                    held_errors.clear()
                    projected.append(f"[frame in withheld file {path}:{number}]")
                    continue
                projected.extend(held_errors)
                held_errors.clear()
            elif line.strip() and not (line.startswith("E ") or line.startswith("E   ")):
                projected.extend(held_errors)
                held_errors.clear()
            if line.startswith("E ") or line.startswith("E   "):
                if (projected and not projected[-1].startswith("[frame in withheld file ")
                        and not self._mentions_hidden(line)):
                    held_errors.append(line)
            elif line.startswith("[frame in withheld file "):
                projected.append(line)
            elif line.startswith("selector=") and not self._mentions_hidden(line):
                projected.append(line)
            elif (line.startswith("pytest exit ") or _PYTEST_SUMMARY.search(line)
                  or match or line.startswith("FAILED ") or line.startswith("ERROR ")):
                if line.startswith(("FAILED ", "ERROR ")) and withheld_seen and " - " in line:
                    # The short-summary message is the exception text raised in withheld
                    # source; keep the node id, drop the message.
                    line = line.split(" - ", 1)[0] + " - [message from withheld source]"
                if not self._mentions_hidden(line):
                    projected.append(line)
        projected.extend(held_errors)
        if section and not any(self._mentions_hidden(item) for item in section):
            projected.extend(section)
        return "\n".join(projected)

    def project_selector_status(self, selector: str, observation: str) -> str:
        """Keep one compact, explicit verdict for each frozen check selector."""
        match = _CHECK_EXIT.search(observation)
        exit_code = match.group(1) if match else "unknown"
        found = _PYTEST_COUNT.findall(observation)
        counts: dict[str, int | str] = ({"passed": 0, "failed": 0, "errors": 0}
                                         if found else {"passed": "?", "failed": "?", "errors": "?"})
        for count, label in found:
            counts["errors" if label.startswith("error") else label] = int(count)
        name = " ".join(selector.split())[:200]
        return (f"selector={name} exit={exit_code} passed={counts['passed']} "
                f"failed={counts['failed']} errors={counts['errors']}")

    def _mentions_hidden(self, line: str) -> bool:
        return any(path in line for path in self.spec.hidden_paths)

    def search_visible(self, root: Path, pattern: str, *, scope: str = "",
                       mode: str = "regex", file_glob: str = "",
                       cursor: str = "", page_size: int = 60) -> "UnitSearchPage":
        """Search only declared source; counts and cursors derive from that set."""
        if mode not in {"regex", "literal", "path"}:
            raise ValueError("subframe grep mode must be regex, literal, or path")
        if not 1 <= page_size <= 240:
            raise ValueError("subframe grep page_size must be between 1 and 240")
        expression = re.compile(pattern) if mode == "regex" else None
        scope_key = _path(scope) if scope not in {"", "."} else ""
        if scope not in {"", "."} and not scope_key:
            raise ValueError("invalid subframe grep scope")
        paths = [path for path in self.spec.visible_paths
                 if (not scope_key or path == scope_key or path.startswith(scope_key + "/"))
                 and (not file_glob or fnmatch.fnmatch(path, file_glob))]
        digest = hashlib.sha256()
        digest.update(json.dumps([pattern, mode, scope_key, file_glob, paths],
                                 separators=(",", ":")).encode("utf-8"))
        rows: list[str] = []
        total_bytes = 0
        for path in paths:
            candidate = Path(root) / path
            if not candidate.is_file() or candidate.is_symlink():
                continue
            data = candidate.read_bytes()
            total_bytes += len(data)
            if total_bytes > 16 << 20:
                raise ValueError("subframe grep visible source exceeds 16 MiB")
            digest.update(path.encode("utf-8"))
            digest.update(hashlib.sha256(data).digest())
            if mode == "path":
                if pattern in path:
                    rows.append(path)
                continue
            try:
                source = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(source.splitlines(), 1):
                match = pattern in line if mode == "literal" else bool(expression.search(line))
                if match:
                    rows.append(f"{path}:{number}:{line[:500]}")
        identity = digest.hexdigest()
        offset = 0
        if cursor:
            pieces = cursor.split(":")
            if (len(pieces) != 3 or pieces[0] != "subframe" or pieces[1] != identity
                    or not pieces[2].isdigit()):
                raise ValueError("subframe grep cursor is stale or invalid")
            offset = int(pieces[2])
            if offset > len(rows):
                raise ValueError("subframe grep cursor is out of range")
        page = rows[offset:offset + page_size]
        next_cursor = (f"subframe:{identity}:{offset + len(page)}"
                       if offset + len(page) < len(rows) else None)
        metadata = {"total": len(rows), "shown": len(page), "scan_complete": True,
                    "result_set_digest": identity, "next_cursor": next_cursor}
        text = "\n".join(page) or "(no matches in visible source)"
        if next_cursor:
            text += f"\nnext_cursor={next_cursor}"
        return UnitSearchPage(text, metadata)


@dataclass(frozen=True)
class UnitSearchPage:
    text: str
    metadata: dict[str, object]


_installed: UnitPolicy | None = None


def install(policy: UnitPolicy) -> None:
    global _installed
    if _installed is not None:
        raise RuntimeError("subframe policy is already installed")
    if not isinstance(policy, UnitPolicy):
        raise TypeError("subframe policy installation requires UnitPolicy")
    _installed = policy


def active() -> UnitPolicy | None:
    return _installed


def _manifest_digest(manifest: object, paths: set[str] | None = None) -> str:
    rows = {
        path: (entry.kind, entry.mode, entry.digest, entry.size)
        for path, entry in manifest.items()
        if paths is None or path in paths
    }
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_environment_digest() -> str:
    """Bind a frozen check to the interpreter and complete inherited environment."""
    payload = {
        "executable": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "os_name": os.name,
        "environment": dict(sorted(os.environ.items())),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _check_ignored(path: str) -> bool:
    return any(part in {"__pycache__", ".pytest_cache", ".ruff_cache",
                        ".mypy_cache", ".test-tmp", ".git", ".venv", "venv"}
               for part in PurePosixPath(path).parts)


def ready_to_auto_finish(
        policy: UnitPolicy, candidate: Path, receipt: CheckReceipt | None,
        original_sources: Mapping[str, str], *, pending_calls: bool = False,
        pending_mutations: bool = False, open_refusal: bool = False,
        decline: Callable[[str], None] | None = None) -> bool:
    """Admit host submission only for a current, complete, checked subframe edit set."""
    def refused(reason: str) -> bool:
        if receipt is not None and receipt.status == "PASS" and decline is not None:
            decline(reason)
        return False

    recipe = policy.spec.contract.check
    if receipt is None or receipt.status != "PASS":
        return False
    if (receipt.invalid_reason is not None or recipe.command_kind == "none"
            or not recipe.selectors or receipt.recipe_digest != recipe.digest()
            or receipt.policy_digest != policy.digest()):
        return refused("receipt_not_armed")
    if pending_calls:
        return refused("pending_calls")
    if pending_mutations:
        return refused("pending_mutations")
    if open_refusal:
        return refused("open_refusal")
    candidate = Path(candidate)
    changed = False
    for relative in policy.spec.contract.owns:
        path = candidate / relative
        if path.is_symlink() or not path.is_file() or relative not in original_sources:
            return refused("owned_path_mismatch")
        if path.read_text(encoding="utf-8") != original_sources[relative]:
            changed = True
    for relative in policy.spec.contract.creates:
        path = candidate / relative
        if path.is_symlink() or not path.is_file():
            return refused("created_path_mismatch")
        changed = True
    if not changed:
        return refused("no_change")
    if __package__:
        from .kernel_effects import scan
    else:
        from kernel_effects import scan
    if _manifest_digest(scan(candidate, exclude=_check_ignored)) != receipt.product_manifest_digest:
        return refused("manifest_mismatch")
    return True


def host_final_check_fields(receipt: object) -> dict[str, str]:
    """Receipt fields that settle a frozen check the host itself judged INVALID or NOT_RUN.

    A subframe's run_tests always executes the frozen recipe, so a non-verdict outcome is a
    host-final judgement (the check wrote files, lost its permit, never ran) that a re-run
    cannot change.  The canonical check row types it UNKNOWN, which the catalog treats as an
    unsettled check whose only resolver (re-run by identity) a subframe cannot dispatch, so
    the next request failed the final-surface proof and the unit stopped.  Carrying the host
    status on the test receipt settles it without touching non-subframe classification.
    """
    status = getattr(receipt, "status", None)
    return {"status": status} if status in {"INVALID", "NOT_RUN"} else {}


def current_receipt(policy: UnitPolicy, candidate: Path,
                    receipt: CheckReceipt | None) -> CheckReceipt | None:
    """Return ``receipt`` when it was produced for exactly this candidate and recipe.

    A completed (PASS or FAIL) receipt is current only while the frozen recipe, policy,
    product files, harness files, and environment digests all still match; INVALID
    receipts are never current because their cause may be transient.
    """
    if (receipt is None or receipt.status not in {"PASS", "FAIL"}
            or receipt.invalid_reason is not None or receipt.wrote_paths
            or receipt.recipe_digest != policy.spec.contract.check.digest()
            or receipt.policy_digest != policy.digest()):
        return None
    if __package__:
        from .kernel_effects import scan
    else:
        from kernel_effects import scan
    manifest = scan(Path(candidate), exclude=_check_ignored)
    if (receipt.product_manifest_digest == _manifest_digest(manifest)
            and receipt.harness_manifest_digest == _manifest_digest(
                manifest, set(policy.spec.contract.check.source_paths))
            and receipt.environment_digest == _check_environment_digest()):
        return receipt
    return None


def run_frozen_check(
    policy: UnitPolicy,
    candidate: Path,
    *,
    execute: Callable[[Path, str], str],
    evidence_dir: Path,
    previous_receipt: CheckReceipt | None = None,
) -> CheckReceipt:
    """Run the frozen selectors on a disposable copy and reject every write."""
    recipe = policy.spec.contract.check
    if recipe.command_kind == "none" or not recipe.selectors:
        return CheckReceipt(recipe.digest(), "", "", "", policy.digest(), "NOT_RUN",
                            None, (), "", "")
    if __package__:
        from .kernel_check_permits import active as active_permits, acquire
        from .kernel_effects import EffectError, scan
    else:
        from kernel_check_permits import active as active_permits, acquire
        from kernel_effects import EffectError, scan

    candidate = Path(candidate)
    raw = ""
    observations: list[str] = []
    wrote_paths: tuple[str, ...] = ()
    invalid_reason: str | None = None
    status = "INVALID"
    product_digest = harness_digest = ""
    product_modes: dict[str, int] = {}
    permit = active_permits()
    context = acquire(*permit, timeout=300.0) if permit is not None else None
    clone: Path | None = None
    before = None
    try:
        if (previous_receipt is not None and previous_receipt.status == "PASS"
                and previous_receipt.invalid_reason is None
                and not previous_receipt.wrote_paths
                and previous_receipt.recipe_digest == recipe.digest()
                and previous_receipt.policy_digest == policy.digest()):
            before = scan(candidate, exclude=_check_ignored)
            product_digest = _manifest_digest(before)
            harness_digest = _manifest_digest(before, set(recipe.source_paths))
            environment_digest = _check_environment_digest()
            if (previous_receipt.product_manifest_digest == product_digest
                    and previous_receipt.harness_manifest_digest == harness_digest
                    and previous_receipt.environment_digest == environment_digest):
                return previous_receipt
        if context is None:
            raise UnitCheckPermitError("shared subframe check permit is not installed")
        with context, tempfile.TemporaryDirectory(prefix="stateframe-unit-check-") as temp:
            clone = Path(temp) / "candidate"
            if before is None:
                before = scan(candidate, exclude=_check_ignored)
                product_digest = _manifest_digest(before)
                harness_digest = _manifest_digest(before, set(recipe.source_paths))
            product_modes = {path: entry.mode for path, entry in before.items()}
            cloned = False
            if sys.platform == "darwin":
                clone_attempt = subprocess.run(
                    ["cp", "-cR", str(candidate), str(clone)],
                    capture_output=True, text=True, timeout=120, check=False)
                cloned = clone_attempt.returncode == 0
            elif sys.platform == "linux":
                clone_attempt = subprocess.run(
                    ["cp", "-a", "--reflink=always", str(candidate), str(clone)],
                    capture_output=True, text=True, timeout=120, check=False)
                cloned = clone_attempt.returncode == 0
            if not cloned:
                clone = Path(temp) / "candidate-copy"
                shutil.copytree(candidate, clone, symlinks=True)
            copied = scan(clone, exclude=_check_ignored)
            if copied != before:
                invalid_reason = "CHECK_WROTE_FILES"
                raw = "check clone differs from candidate before execution"
            else:
                observations = [execute(clone, selector) for selector in recipe.selectors]
                raw = "\n\n".join(f"[{selector}]\n{observation}"
                                   for selector, observation in zip(recipe.selectors, observations))
                after = scan(clone, exclude=_check_ignored)
                wrote_paths = tuple(sorted(path for path in set(before) | set(after)
                                         if before.get(path) != after.get(path)))
                if wrote_paths:
                    invalid_reason = "CHECK_WROTE_FILES"
                else:
                    prefix = "pytest" if recipe.command_kind == "pytest" else "check"
                    status = ("PASS" if all(re.match(rf"^{prefix} exit 0\b", value)
                                            for value in observations) else "FAIL")
    except UnitCheckPermitError as error:
        invalid_reason = "RESOURCES_UNAVAILABLE"
        raw = str(error)
    except TimeoutError as error:
        invalid_reason = "RESOURCES_UNAVAILABLE"
        raw = f"shared subframe check permit unavailable: {error}"
    except UnitCheckContainmentError:
        raise
    except UnitCheckCleanupError as error:
        invalid_reason = "CHECK_CLEANUP_FAILED"
        raw = f"check cleanup failed: {error}"
    except (EffectError, OSError, ValueError,
            subprocess.TimeoutExpired) as error:
        invalid_reason = "CHECK_WROTE_FILES"
        raw = f"check isolation failed: {type(error).__name__}: {error}"
    projected = policy.project_check_output(
        raw, "pytest" if recipe.command_kind == "pytest" else "other", clone_root=clone)
    if recipe.command_kind == "pytest" and recipe.selectors:
        rows = [policy.project_selector_status(
            selector, observations[index] if index < len(observations) else "")
            for index, selector in enumerate(recipe.selectors)]
        projected = "\n".join([*rows, projected])
    if invalid_reason is not None:
        status = "INVALID"
        projected = (f"INVALID {invalid_reason}: {', '.join(wrote_paths) or 'check refused'}\n"
                     + projected)
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    evidence_path = evidence_dir / f"{evidence_hash}.txt"
    if evidence_path.exists():
        if evidence_path.read_text(encoding="utf-8") != raw:
            raise RuntimeError("check evidence digest collision")
    else:
        with evidence_path.open("x", encoding="utf-8") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    environment_digest = _check_environment_digest()
    return CheckReceipt(recipe.digest(), product_digest, harness_digest,
                        environment_digest, policy.digest(), status, invalid_reason,
                        wrote_paths, str(evidence_path), projected,
                        product_manifest_modes=product_modes)


def run_check_command(clone: Path, selector: str) -> str:
    """Execute a frozen command selector in the disposable check copy."""
    try:
        command = shlex.split(selector)
    except ValueError as error:
        return f"check exit 4 — invalid command selector: {error}"
    if not command:
        return "check exit 4 — empty command selector"
    if __package__:
        from .kernel_execution_policy import ExecutionPolicyError, run_sandboxed
    else:
        from kernel_execution_policy import ExecutionPolicyError, run_sandboxed
    token = new_check_token()
    try:
        result = run_sandboxed(command, clone, timeout=300, cpu_seconds=300,
                               output_limit=256 * 1024,
                               internal_env={"STATEFRAME_CHECK_TOKEN": token},
                               pre_reap_cleanup=lambda: _terminate_check_token(token))
    except (ExecutionPolicyError, OSError) as error:
        _reap_check_token(token)
        return f"check exit 4 — {type(error).__name__}: {error}"
    except UnitCheckContainmentError:
        raise
    except UnitCheckCleanupError:
        raise
    except BaseException:
        _reap_check_token(token)
        raise
    if result.cleanup_finding is not None:
        raise UnitCheckCleanupError(result.cleanup_finding)
    reap_check_group(result.pid, token=token)
    if result.timed_out:
        return f"check exit 124 — timed out\n{result.completed.stdout}{result.completed.stderr}"
    return (f"check exit {result.completed.returncode}\n"
            f"{result.completed.stdout}{result.completed.stderr}")
