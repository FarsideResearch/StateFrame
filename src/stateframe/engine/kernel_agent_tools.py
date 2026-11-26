"""Per-run tool registration and adapters for jobs, Git, web and custom commands."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import kernel_git_commit
from kernel_agent_jobs import AgentJobs
from kernel_agent_schemas import (
    BUILTIN_SCHEMAS,
    JOB_NAMES,
    OBSERVATION_NAMES,
    SchemaMeteringTimeout,
    builtin_schema_snapshot,
    prepare_schema_measurement,
    schema,
)
from kernel_exec_session import ExecSessionError
from kernel_shell_tool import ShellCleanupFailed, ShellReply, workspace_identity
from kernel_tool_tape import (
    ToolReplayRefused,
    replay_decision,
    replay_mode_active,
    tape_for_run,
    verify_intent,
)

_SECTION_OPENERS = "section_openers"
_OPENER_PATH_LIMIT = 512
_OPENER_INPUT_LIMIT = 128 * 1024
_OPENER_OUTPUT_LIMIT = 4096
_OPENER_BYTE_RESERVE = 8192
_OPENER_TOKEN_RESERVE = 2048
_OPENER_SCHEMA = schema(
    _SECTION_OPENERS,
    "Review a finished Markdown memo before completion. Supply its workspace-relative path. "
    "Returns section-opening sentences in document order, with coverage, "
    "line ranges and a source digest. Use only intact JSON and use native read for edit anchors.",
    {"path": {"type": "string", "maxLength": _OPENER_PATH_LIMIT}}, ["path"],
)
_ATX_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*)|$)")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_INDENTED_FENCE = re.compile(r"( *)(`{3,}|~{3,})(.*)")
_LIST_FENCE = re.compile(r" {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+(`{3,}|~{3,})(.*)")
_UNSUPPORTED_BLOCK = re.compile(r"^(?: {4}|\t| {0,3}(?:[>|<]|\[|[-+*]\s|\d{1,9}[.)]\s))")
_SETEXT = re.compile(r" {0,3}(?:=+|-+)[ \t]*$")
_THEMATIC_BREAK = re.compile(r" {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_TABLE_CELL_DELIMITER = re.compile(r":?-{3,}:?")
_SENTENCE_END = re.compile(r"""(?<![.!?])([.!?]+)["'”’)]*(?=\s|$)""")
_ABBREVIATION = re.compile(
    r"(?i)(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|cf|Inc|Ltd|No|Fig|Dept|(?:[A-Z]\.)+[A-Z])",
)
_MONTH_ABBREVIATION = re.compile(r"(?i)(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
_MONTH_ACRONYM_CONTINUATION = re.compile(
    r"[A-Z]{2,}(?:-[A-Za-z0-9]+)*(?:['’]s)?\s+(?P<noun>[a-z]+)\b",
)
_MONTH_NOUN_DETERMINER = re.compile(r"(?i)\b(?:the|a|an)$")
_MONTH_TIME_PREPOSITION = re.compile(r"(?i)\b(?:in|during|since|until|through|by|before|after)$")
_MONTH_DATE_COMPLEMENT = re.compile(
    r"(?i)(?:(?:the|a|an|our|your|their|its)\s+)?"
    r"(?:(?!to\s)[a-z]+(?:-[a-z]+)*\s+)?(?:deadline|cutoff|date|month)\s+(?:is|was|are|were)",
)
_MONTH_MODAL_CLAUSE = re.compile(r"(?:can|could|may|might|must|shall|should|will|would) [a-z]+\b")
_APPLICATION_TIME_MODIFIER = re.compile(
    r"""(?:filed|received|submitted) (?:after|before|by) (?:a|an|the) [a-z]+(?=[.!?]["'”’)]*(?: |$))""",
)
_MONTH_CLAUSE_PREDICATES = frozenset({
    "am", "are", "became", "become", "began", "begin", "begins", "can", "could",
    "did", "do", "does", "fell", "go", "goes", "grew", "grow", "has", "have", "had",
    "is", "led", "made", "may", "might", "must", "put", "ran", "rose", "saw", "said",
    "set", "should", "took", "was", "were", "went", "will", "would",
})


class _SectionOpenerTool:
    """Native observation using the per-run custom schema and hook admission contract."""

    writes = "none"

    @staticmethod
    def build_argv(args):
        from kernel_custom_tools import CustomToolError

        if not isinstance(args.get("path"), str) or set(args) != {"path"}:
            raise CustomToolError("expected one Markdown path")
        return ()

    @staticmethod
    def verify_executable():
        # The released engine implements the tool directly.
        return None


def _memo_bytes(root: Path, path: str) -> bytes:
    """Read only the named file, including when a path component is replaced."""
    from kernel_git_view import excluded_path

    parts = path.split("/")
    if (any(part in {"", ".", ".."} for part in parts)
            or any(char in path for char in "\\:")
            or not path.isprintable() or excluded_path(path)):
        raise ValueError("expected an admitted workspace-relative Markdown path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("expected a regular file with one link")
            if before.st_size > _OPENER_INPUT_LIMIT:
                raise ValueError("memo exceeds the input byte limit")
            raw = stream.read(_OPENER_INPUT_LIMIT + 1)
            after = os.fstat(fd)
            if (len(raw) != before.st_size or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns or after.st_nlink != 1):
                raise ValueError("memo changed during the read")
            return raw
    finally:
        os.close(parent)


def _opening_sentence(paragraph: list[str], start: int) -> dict:
    if not paragraph:
        return {"status": "empty"}
    prose = "\n".join(paragraph)
    for boundary in _SENTENCE_END.finditer(prose):
        quote = prose[:boundary.end()]
        words = prose[:boundary.start()].rsplit(maxsplit=1)
        last_word = words[-1].lstrip("\"'“‘(") if words else ""
        if boundary[1] == "." and _MONTH_ABBREVIATION.fullmatch(last_word):
            following = " ".join(prose[boundary.end():].split())
            if following and (following[0].islower() or following[0].isdigit()):
                continue
            acronym_continuation = _MONTH_ACRONYM_CONTINUATION.match(following)
            if acronym_continuation:
                before_month = prose[:boundary.start() - len(last_word)].rstrip()
                if _MONTH_NOUN_DETERMINER.search(before_month):
                    continue
                tail = following[acronym_continuation.end():].lstrip()
                temporal_month = bool(_MONTH_TIME_PREPOSITION.search(before_month))
                # An application is the object of submission. A simple time
                # qualifier can complete that noun phrase as a reduced relative.
                application_modifier = (
                    acronym_continuation["noun"] in {"application", "applications"}
                    and _APPLICATION_TIME_MODIFIER.match(tail) is not None
                )
                if application_modifier and not temporal_month:
                    continue
                predicate_match = re.match(r"[A-Za-z]+", tail)
                predicate = predicate_match[0].casefold() if predicate_match else ""
                temporal_boundary = (temporal_month and not application_modifier
                                     and (predicate.endswith("ed") or predicate in _MONTH_CLAUSE_PREDICATES))
                # Match the whole calendar subject, including a compound like
                # "application deadline", so "to date" cannot supply it.
                copular_boundary = (_MONTH_DATE_COMPLEMENT.fullmatch(before_month) is not None
                                    and _MONTH_MODAL_CLAUSE.match(tail) is not None)
                if not (temporal_boundary or copular_boundary):
                    # A possible predicate alone cannot distinguish a new
                    # sentence from a modifier in the opening claim.
                    return {"status": "ambiguous"}
        # Other abbreviations, ellipses and inline markup stay ambiguous in
        # this conservative subset rather than being guessed through.
        if (boundary[1] in {".", "!", "?"}
                and (boundary[1] != "." or not _ABBREVIATION.fullmatch(last_word))
                and not any(char in quote for char in "`[]<>\\")
                and any(char.isalpha() for char in quote)):
            return {"status": "quoted", "sentence": " ".join(quote.split()),
                    "source_lines": [start, start + quote.count("\n")]}
        return {"status": "ambiguous"}
    return {"status": "ambiguous"}


def _markdown_table_cells(line: str) -> list[str]:
    """Split the simple pipe or two-space table rows supported by the review parser."""
    stripped = line.strip()
    if "|" in stripped:
        cells = stripped.strip("|").split("|")
    elif re.search(r"[ \t]{2,}", stripped):
        cells = re.split(r"[ \t]{2,}", stripped)
    else:
        return []
    return [cell.strip() for cell in cells]


def _table_delimiter(line: str, header: list[str]) -> list[str] | None:
    cells = _markdown_table_cells(line)
    if (len(cells) < 2 or len(cells) != len(header)
            or not all(_TABLE_CELL_DELIMITER.fullmatch(cell) for cell in cells)):
        return None
    return cells


def _memo_sections(source: str) -> tuple[list[dict], bool]:
    """Top-level ATX headings and each heading's first plain-prose paragraph."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()
    rows = []
    fence = None
    fence_indent = None
    row = None
    paragraph: list[str] = []
    start = 0
    paragraph_done = False
    unsupported = False
    previous_prose = False
    table_columns = 0
    for number, line in enumerate(lines, 1):
        fence_match = _FENCE.fullmatch(line)
        list_fence_match = _LIST_FENCE.fullmatch(line)
        if fence is not None:
            closing_marker = fence_match[1] if fence_match else None
            closing_info = fence_match[2] if fence_match else None
            if fence_match is None and fence_indent is not None:
                indented_match = _INDENTED_FENCE.fullmatch(line)
                if indented_match:
                    indent = len(indented_match[1])
                    if fence_indent <= indent <= fence_indent + 3:
                        closing_marker = indented_match[2]
                        closing_info = indented_match[3]
            if (closing_marker and closing_marker[0] == fence[0]
                    and len(closing_marker) >= len(fence) and not closing_info.strip()):
                fence = None
                fence_indent = None
            continue
        list_fence_indent = None
        if list_fence_match:
            fence_match = list_fence_match
            list_fence_indent = list_fence_match.start(1)
            unsupported = True
        if fence_match and (fence_match[1][0] == "~" or "`" not in fence_match[2]):
            fence = fence_match[1]
            fence_indent = list_fence_indent
            paragraph_done |= bool(paragraph)
            previous_prose = False
            continue
        heading = _ATX_HEADING.fullmatch(line)
        if table_columns:
            cells = _markdown_table_cells(line)
            if heading or not line.strip():
                table_columns = 0
                paragraph_done = False
                previous_prose = False
                if not heading:
                    continue
            elif len(cells) == table_columns:
                previous_prose = False
                continue
            else:
                table_columns = 0
                paragraph_done = False
        if heading:
            if row is not None:
                row["section_end_line"] = number - 1
                row.update(_opening_sentence(paragraph, start))
            title = (heading[2] or "").strip(" \t")
            before_hashes = title.rstrip("#")
            if not before_hashes or before_hashes.endswith((" ", "\t")):
                title = before_hashes.rstrip(" \t")
            row = {"heading": title, "level": len(heading[1]), "heading_line": number}
            rows.append(row)
            paragraph = []
            paragraph_done = False
            previous_prose = False
            continue
        if row is not None and paragraph:
            table_header = _markdown_table_cells(paragraph[-1])
            if _table_delimiter(line, table_header) is not None:
                # A Markdown table is not an opener. Ignore its rows, then look
                # for the first prose paragraph that follows it.
                unsupported = True
                paragraph.pop()
                paragraph_done = True
                previous_prose = False
                table_columns = len(table_header)
                continue
        if (previous_prose and _SETEXT.fullmatch(line)) or line.lstrip().startswith("<"):
            unsupported = True
        if _THEMATIC_BREAK.fullmatch(line):
            paragraph_done |= bool(paragraph)
            previous_prose = False
            continue
        previous_prose = bool(line.strip()) and not _UNSUPPORTED_BLOCK.match(line)
        if not line.strip():
            paragraph_done |= bool(paragraph)
        elif row is not None and not paragraph_done:
            if not paragraph:
                start = number
            paragraph.append(line.strip() if previous_prose else "<unsupported>")
            if not previous_prose:
                paragraph_done = True
        elif row is None:
            unsupported = True
    if row is not None:
        row["section_end_line"] = len(lines)
        row.update(_opening_sentence(paragraph, start))
    return rows, unsupported or fence is not None


def _section_opener_view(root: Path, path: object, limit: int) -> str:
    if limit <= 0:
        return "REFUSED MEMO_VIEW_CAPACITY: use native read to review the memo"
    if not isinstance(path, str) or not 1 <= len(path) <= _OPENER_PATH_LIMIT:
        return "REFUSED BAD_ARGUMENT: supply one workspace-relative Markdown path"
    try:
        raw = _memo_bytes(root, path)
        source = raw.decode("utf-8-sig")
    except UnicodeError:
        return "REFUSED MEMO_SOURCE: expected UTF-8 Markdown"
    except ValueError as error:
        return f"REFUSED MEMO_SOURCE: {error}"
    except OSError:
        return "REFUSED MEMO_SOURCE: unable to open the workspace Markdown file safely"
    if any(not char.isprintable() and char not in "\t\r\n" for char in source):
        return "REFUSED MEMO_SOURCE: unsupported control characters in Markdown"

    rows, unsupported = _memo_sections(source)
    coverage = {"complete": False, "sections": len(rows), "shown": 0, "omitted": len(rows),
                "empty": sum(row["status"] == "empty" for row in rows),
                "ambiguous": sum(row["status"] == "ambiguous" for row in rows),
                "unsupported_structure": unsupported}
    incomplete = "Incomplete view. Use native read to resolve missing evidence before identifying the first transition."
    complete_review = (
        "Compare adjacent openers in document order for changes in population, criterion or "
        "conclusion. Use native read on the cited sections and relevant premises, starting with "
        "the earliest suspect transition. Verify whether its conclusion really lacks support "
        "or the outline omitted the premise. Repair only the first verified gap with a native "
        "edit and reread the changed passage."
    )
    eligible = bool(rows) and not (unsupported or coverage["empty"] or coverage["ambiguous"])
    document = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                "coverage": coverage, "sections": [],
                "review": (complete_review if eligible and len(rows) > 1 else
                           "At least two sections are needed to review a transition."
                           if eligible else incomplete)}

    def encode(value):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    for index, row in enumerate(rows, 1):
        document["sections"].append(row)
        coverage.update(shown=index, omitted=len(rows) - index)
        if len(encode(document)) + 1 > limit:
            document["sections"].pop()
            coverage.update(shown=index - 1, omitted=len(rows) - index + 1)
            break
    complete = eligible and not coverage["omitted"]
    coverage["complete"] = complete
    if not complete:
        document["review"] = incomplete
    result = encode(document)
    if len(result) + 1 <= limit:
        # The execution fitter can withhold this body as an opaque observation.
        # A clipped body is invalid JSON and carries no coverage claim.
        return "\n" + result
    # Never deliver a clipped digest, row, or review instruction.
    return "REFUSED MEMO_VIEW_CAPACITY: use native read to review the memo"


def _section_opener_limit(spec) -> int:
    """Use spare request capacity without asking the engine to evict evidence."""
    if not spec.get("chat_url"):
        return _OPENER_OUTPUT_LIMIT
    from kernel_pilot import ACTIVE_RUNTIME

    runtime = ACTIVE_RUNTIME.get()
    capture = getattr(runtime, "_active_capture", None)
    qualified = getattr(capture, "qualified_request", None)
    measured = (qualified.measurement if qualified is not None
                else getattr(capture, "_request_measurement", None))
    if measured is None or measured.prompt_tokens is None:
        return 0
    byte_room = measured.max_request_bytes - measured.request_bytes - _OPENER_BYTE_RESERVE
    token_ceiling = min(measured.max_prompt_tokens, runtime.frame_ceiling)
    token_room = token_ceiling - measured.prompt_tokens - _OPENER_TOKEN_RESERVE
    # The view is ASCII JSON embedded inside another JSON string.
    return max(0, min(_OPENER_OUTPUT_LIMIT, byte_room // 2, token_room // 2))


def _tokenizer_auth(spec):
    key_name = spec.get("api_key_env")
    key = os.environ.get(key_name, "") if isinstance(key_name, str) else ""
    endpoint = spec.get("chat_url")
    return key or None, endpoint if isinstance(endpoint, str) else None


def _replayed_schema_budget(spec):
    resume = spec.get("resume")
    if not isinstance(resume, dict) or resume.get("restart_live"):
        return None
    tape = resume.get("tape")
    first = tape[0] if tape else resume.get("pending")
    checkpoint = resume.get("checkpoint")
    if first is None and (not isinstance(checkpoint, dict) or checkpoint.get("status") == "skipped"):
        return None

    from kernel_journal_store import journal_paths, open_journal
    from kernel_measurement import digest, resolve_journal_blob_refs

    # A resumed attempt can carry forward sends from an earlier archive.
    # Use the journal that accompanied the first request being rebuilt.
    request = first.get("request") if first is not None else None
    attempt = Path(request).parent.parent if request else Path(resume["previous_attempt"])
    paths = journal_paths(attempt / "worker" / "measurement-runs")
    if len(paths) != 1:
        raise ToolReplayRefused("Cannot locate the startup journal for the recorded request")
    try:
        with open_journal(paths[0]) as stream:
            started = json.loads(stream.readline())
        manifest = resolve_journal_blob_refs(started["manifest"], paths[0], {})
        if (started.get("kind") != "run_started" or not isinstance(manifest, dict)
                or started.get("config_hash") != digest(manifest)):
            raise ValueError("invalid run start")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ToolReplayRefused("Invalid startup journal for the recorded request") from error
    budget = manifest.get("startup_schema_budget")
    if budget is None:
        raise ToolReplayRefused("Recorded startup schema counts are missing")
    return budget


class AgentTools:
    def __init__(self, shell, custom, mcp_servers=(), *, resume_active=False,
                 defer_schema_budget=False):
        self._section_openers = os.environ.get("STATEFRAME_R200_F060_READ_ONLY_VIEW", "off") == "on"
        if self._section_openers and any(tool.name == _SECTION_OPENERS for tool in custom.tools):
            from kernel_custom_tools import CustomToolError

            raise CustomToolError(f"custom tools duplicate built-in capabilities: {_SECTION_OPENERS}")
        self._pilot_schema_budget = shell.spec.get("send1_wave4") is True and not resume_active
        registration = None
        if self._pilot_schema_budget:
            try:
                from kernel_agent_registration import registration_data
            except ModuleNotFoundError as error:
                if error.name != "kernel_agent_registration":
                    raise
            else:
                registration = registration_data(shell, custom, mcp_servers, resume_active=resume_active)
        if registration is None and custom.tools:
            from kernel_completion_owner import TOOL_SPECS
            from kernel_custom_tools import CustomToolError
            from kernel_lean import _KNOWN_TOOL_NAMES

            collisions = sorted(tool.name for tool in custom.tools
                                if tool.name in TOOL_SPECS or tool.name in _KNOWN_TOOL_NAMES)
            if collisions:
                raise CustomToolError("custom tools duplicate built-in capabilities: " + ", ".join(collisions))
        self.shell = shell
        self.custom = {tool.name: tool for tool in custom.tools}
        self.mcp_servers = tuple(mcp_servers)
        self.resume_active = resume_active
        if registration is None:
            self.custom_schemas = copy.deepcopy(custom.schemas())
            for schema in self.custom_schemas:
                schema["function"]["parameters"]["properties"].setdefault("why", {"type": "string"})
        else:
            self.custom_schemas = registration.custom_schemas
        wall = os.environ.get("STATEFRAME_RUN_WALL_SECONDS")
        self.deadline = time.monotonic() + float(wall) if wall else float("inf")
        self.jobs = AgentJobs(shell, self.deadline)
        self.git = None
        source = shell.spec.get("repo")
        if shell.settings.git_tool and source:
            from kernel_git_view import GitView, excluded_path

            metadata = shell.run_dir / "git-view-metadata.json"
            if metadata.exists():
                pinned = json.loads(metadata.read_text())
                directory = Path(pinned["git_dir"]).resolve(strict=True)
                if not directory.is_relative_to(shell.run_dir.resolve()):
                    raise ValueError("pinned Git view is outside this run")
                if (directory / "HEAD").read_text().strip() != pinned["head_oid"]:
                    raise ValueError("pinned Git HEAD differs from its run record")
                self.git = GitView(git_dir=directory, workspace=shell.run_dir / "workspace",
                    baseline=shell.run_dir / "parent", head_oid=pinned["head_oid"],
                    object_dirs=tuple(Path(p) for p in pinned["object_dirs"]), secret_filter=excluded_path)
            else:
                directory = Path(tempfile.mkdtemp(prefix="git-attempt-", dir=shell.run_dir))
                self.git = GitView.build(Path(source), directory, secret_filter=excluded_path)
                if self.git is not None:
                    with metadata.open("x") as record:
                        json.dump({"head_oid": self.git.head_oid, "git_dir": str(self.git.git_dir),
                                   "object_dirs": [str(p) for p in self.git.object_dirs]}, record, sort_keys=True)
        # STATEFRAME_GIT_COMMIT: commit and branch tools, absent from every request when off.
        self.commit_names = frozenset(kernel_git_commit.TOOL_NAMES) if kernel_git_commit.enabled() else frozenset()
        self.names = OBSERVATION_NAMES | self.commit_names | self.custom.keys()
        self._builtin_schema_snapshot = (
            registration.builtin_schema_snapshot if registration is not None
            else builtin_schema_snapshot() if self._pilot_schema_budget else None
        )
        if self._section_openers:
            base_schemas = self._builtin_schema_snapshot or builtin_schema_snapshot()
            self._builtin_schema_snapshot = (*base_schemas, copy.deepcopy(_OPENER_SCHEMA))
            if not self.mcp_servers:
                self._register_section_openers()
        self._pending_schema_budget = None
        self.startup_schema_budget = None
        self._defer_schema_budget = defer_schema_budget and bool(shell.spec.get("chat_url"))
        if self._defer_schema_budget:
            self.startup_schema_budget = _replayed_schema_budget(shell.spec)
            if self.startup_schema_budget is not None:
                self._write_schema_budget(self.startup_schema_budget["counts"])
            elif not self.mcp_servers:
                self._start_schema_budget()
        elif not self.mcp_servers:
            self._write_schema_budget(self._prepare_schema_budget()())

    def _prepare_schema_budget(self, *, deadline=None):
        auth_key, endpoint = _tokenizer_auth(self.shell.spec)
        custom_schemas = self.custom_schemas
        if self._section_openers:
            custom_schemas = [s for s in custom_schemas if s["function"]["name"] != _SECTION_OPENERS]
        return prepare_schema_measurement(custom_schemas, endpoint=endpoint,
                                          auth_key=auth_key, generation_endpoint=endpoint,
                                          deadline=deadline,
                                          builtin_schemas=self._builtin_schema_snapshot,
                                          recorded=self.shell.spec if self._pilot_schema_budget else None,
                                          resume_active=self.resume_active)

    def _start_schema_budget(self):
        from concurrent.futures import Future
        from threading import Thread

        from kernel_request_diet import TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS

        deadline = time.monotonic() + TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS
        measure_schema_budget = self._prepare_schema_budget(deadline=deadline)
        pending = Future()
        self._pending_schema_budget = (deadline, measure_schema_budget, pending)

        def measure():
            if not pending.set_running_or_notify_cancel():
                return
            try:
                counts = measure_schema_budget()
            except BaseException as error:  # noqa: BLE001 - forward worker failures to the caller
                pending.set_result((time.monotonic(), None, error))
            else:
                pending.set_result((time.monotonic(), counts, None))

        # Workspace preparation can proceed while the tokenizer checks the schemas.
        # An abandoned transport (including DNS) cannot hold up process exit.
        Thread(target=measure, name="tool-schema-budget", daemon=True).start()

    def _write_schema_budget(self, budget):
        self.schema_budget = budget
        with tempfile.NamedTemporaryFile(mode="w", prefix="tool-schema-budget-", suffix=".json",
                                         dir=self.shell.run_dir, delete=False) as report:
            json.dump(budget, report, sort_keys=True)
            self.schema_budget_path = Path(report.name)

    def _finish_schema_budget(self):
        waiting = getattr(self, "_pending_schema_budget", None)
        if waiting is None:
            return
        deadline, measure, pending = waiting
        reason = "completed"
        try:
            result = pending.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError:
            result = None
        if result is None or result[0] >= deadline or isinstance(result[2], SchemaMeteringTimeout):
            pending.cancel()
            reason = "deadline"
        else:
            _, counts, error = result
            if error is not None:
                from concurrent.futures import CancelledError

                from kernel_custom_tools import CustomToolError

                if isinstance(error, (CustomToolError, CancelledError)) or not isinstance(error, Exception):
                    raise error
                reason = "measurement_error"
        if reason != "completed":
            # Schema tokenization is advisory startup work. Preserve a usable
            # byte estimate when the tokenizer is slow or unavailable.
            counts = measure(estimate=True)
        self._write_schema_budget(counts)
        self.startup_schema_budget = {"reason": reason, "counts": counts}
        self._pending_schema_budget = None

    def bind(self):
        if self.mcp_servers:
            from kernel_custom_tools import CustomToolError
            from kernel_mcp_tools import discover

            if self.resume_active and not self.shell.settings.replay_external:
                external = [server.name for server in self.mcp_servers if server.replay == "external"]
                if external:
                    raise ToolReplayRefused(
                        "MCP discovery requires --replay-external for servers: " + ", ".join(external))
            auth_key, generation_endpoint = _tokenizer_auth(self.shell.spec)
            found = discover(self.mcp_servers, self.shell.root, self.shell.run_dir,
                             shell_network=self.shell.settings.shell_network,
                             host_authorization=self.shell.settings.host_authorization is not None,
                             custom_schemas=self.custom_schemas, endpoint=self.shell.spec.get("chat_url"),
                             auth_key=auth_key, generation_endpoint=generation_endpoint,
                             budget_admission=self.shell.spec.get("mcp_schema_admission"))
            if (found.digest != self.shell.spec.get("mcp_tools_digest")
                    or [t.name for t in found.tools] != self.shell.spec.get("mcp_tools")
                    or list(found.omitted) != self.shell.spec.get("mcp_omitted")):
                raise CustomToolError("MCP tool list or schema changed since host validation")
            if self._section_openers and any(tool.name == _SECTION_OPENERS for tool in found.tools):
                raise CustomToolError(f"MCP tools duplicate built-in capabilities: {_SECTION_OPENERS}")
            self.custom.update({tool.name: tool for tool in found.tools})
            self.custom_schemas.extend(copy.deepcopy(found.schemas()))
            if self._section_openers:
                self._register_section_openers()
            self.names = OBSERVATION_NAMES | self.commit_names | self.custom.keys()
            if self.startup_schema_budget is None:
                if self._defer_schema_budget:
                    self._start_schema_budget()
                else:
                    self._write_schema_budget(self._prepare_schema_budget()())
        self._finish_schema_budget()
        if self.git is not None:
            self.git.workspace = self.shell.root
            self.git.baseline = Path(self.shell.spec.get("parent_snapshot", self.shell.root))

    def offered(self, name):
        if name in JOB_NAMES:
            return bool(self.jobs.jobs)
        if name == "git":
            return self.git is not None and self.shell.settings.git_tool
        if name == "skill_load":
            from kernel_skills import service as skills_service
            return skills_service() is not None
        if name == "web_fetch":
            return self.shell.settings.web
        if name in kernel_git_commit.TOOL_NAMES:
            return name in self.commit_names
        return name in self.custom

    def _register_section_openers(self):
        self.custom[_SECTION_OPENERS] = _SectionOpenerTool()
        self.custom_schemas.append(copy.deepcopy(_OPENER_SCHEMA))
        self.names = self.names | {_SECTION_OPENERS}

    def schemas(self):
        self._finish_schema_budget()
        commit_schemas = kernel_git_commit.SCHEMAS if self.commit_names else ()
        return [copy.deepcopy(s) for s in (*BUILTIN_SCHEMAS, *commit_schemas, *self.custom_schemas)
                if self.offered(s["function"]["name"])]

    def _commit(self, name, args, call_id):
        log = kernel_git_commit.CommitLog(self.shell.run_dir)
        if name == "git_branch":
            return kernel_git_commit.branch(log, call_id, args.get("name"))
        from kernel_git_view import excluded_path

        manifest = self.shell._scan()
        store = getattr(self.shell, "store", None)
        if store is None:
            from kernel_effects import ContentStore
            store = ContentStore(self.shell.run_dir)
        return kernel_git_commit.commit(log, store, manifest, self.shell.root, call_id,
                                        args.get("message"), excluded_path)

    def observation(self, name, args, call_id, operation, *, definition=None, capability=None):
        before = workspace_identity(self.shell._scan())
        identity = self.shell.identity()
        authority = {"settings": self.shell.settings.as_dict(), "purpose": name}
        authority["settings"].pop("replay_external", None)
        if capability is None and name != "web_fetch":
            capability = replace(self.shell.capability, purpose="job", workspace_writable=False, write_roots=())
        if capability is not None:
            authority["capability"] = capability.as_dict()
        encoded = json.dumps(authority, sort_keys=True)
        replacements = [(str(self.shell.root), "<workspace>"), (str(self.shell.run_dir), "<run>")]
        if self.git is not None:
            replacements.insert(0, (str(self.git.git_dir), "<git-view>"))
        for path, label in replacements:
            encoded = encoded.replace(path, label)
        intent = dict(tool=name, args=args, replay_class="observation",
            capability_digest=hashlib.sha256(encoded.encode()).hexdigest(),
            definition_digest=definition, env_identity=identity, pre_manifest_digest=before)
        tape = tape_for_run()
        if replay_mode_active():
            record = tape.lookup(call_id)
            reason = replay_decision(record, replay_external_allowed=False).refuse_reason
            if not reason:
                reason = verify_intent(record, **intent)
            if reason:
                raise ToolReplayRefused(reason)
            return record.observation
        tape.intent(call_id=call_id, **intent)
        try:
            result = operation()
        except ExecSessionError as error:
            if "requires " in str(error) and "no unsandboxed fallback" in str(error):
                result = f"REFUSED SANDBOX_UNAVAILABLE: {error}"
            else:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {error}") from error
        tape.complete(call_id=call_id, exit_code=None, observation=result,
                      post_manifest_digest=before, env_identity=identity)
        return result

    def background(self, args, call_id, admission):
        try:
            return self._background(args, call_id, admission)
        except ShellCleanupFailed as error:
            return ShellReply(str(error), fatal=True)

    def _background(self, args, call_id, admission):
        def launch():
            if admission.requires_confirmation:
                from kernel_shell_tool import request_confirmation
                if not request_confirmation(self.shell.run_dir, call_id, args["command"]):
                    return "REFUSED ASK_DECLINED: user declined shell command"
            return self.jobs.launch(args)
        timeout = args.get("timeout_seconds", 600)
        if type(timeout) is not int or not 1 <= timeout <= 600:
            return ShellReply("REFUSED DENIED_PROGRAM: timeout_seconds must be an integer from 1 to 600")
        capability = self.jobs.capability(f"j{len(self.jobs.jobs)+1}", timeout)
        observation = self.observation("shell", args, call_id, launch, capability=capability)
        if replay_mode_active():
            self.jobs.replay_launch(args, observation)
        return ShellReply(observation)

    def snapshot(self):
        def observe(sequence, build):
            value = self.observation("job_snapshot", {"sequence": sequence}, f"job-snapshot-{sequence}",
                                     lambda: json.dumps(build(), sort_keys=True))
            snapshot = json.loads(value)
            if replay_mode_active():
                for row in snapshot["jobs"]:
                    job = self.jobs.jobs.get(row["id"])
                    if job is not None and job.handle is None and row["state"] != "running":
                        job.state, job.exit_code = row["state"], row["exit_code"]
            return snapshot
        return self.jobs.snapshot(observe)

    def stop_jobs(self, call_id):
        if not self.jobs.jobs:
            return ""
        result = self.observation("job_stop_all", {}, call_id + ":stop-jobs", self.jobs.stop_all)
        if replay_mode_active():
            for job in self.jobs.jobs.values():
                if job.handle is None and job.state == "lost on restart":
                    job.state = "stopped"
        return result

    def execute(self, name, args, call_id):
        try:
            return self._execute(name, args, call_id)
        except ShellCleanupFailed as error:
            return ShellReply(str(error), fatal=True)

    def _execute(self, name, args, call_id):
        if self._section_openers and name == _SECTION_OPENERS:
            capability = replace(self.shell.capability, purpose="job", workspace_writable=False,
                                 network="none", write_roots=())
            result = self.observation(
                name, args, call_id,
                lambda: _section_opener_view(self.shell.root, args.get("path"),
                                             _section_opener_limit(self.shell.spec)),
                definition="section-openers/1", capability=capability,
            )
            return ShellReply(result)
        if name == "skill_load":
            from kernel_skills import service as skills_service
            skills = skills_service()
            if skills is None:
                return ShellReply("REFUSED SKILL_UNKNOWN: no skills configured")
            def operation():
                return skills.load(args.get("name"))
            observation = self.observation(name, args, call_id, operation,
                                           definition=self.shell.spec.get("skills_digest"))
            if replay_mode_active() and operation() != observation:
                raise ToolReplayRefused("skill load observation diverged on replay")
            return ShellReply(observation)
        if name in self.custom:
            from kernel_effects import EffectError
            from kernel_run_env import RunEnvError
            from kernel_tool_tape import ToolTapeError
            from kernel_shell_tool import ShellExecutionFailed
            try:
                return self.shell._execute(args, call_id, custom=self.custom[name])
            except (EffectError, RunEnvError, ToolTapeError, OSError) as error:
                raise ShellExecutionFailed(f"{type(error).__name__}: {error}") from error
        if name in kernel_git_commit.TOOL_NAMES:
            if not self.offered(name):
                return ShellReply(f"REFUSED NOT_OFFERED: {name} needs {kernel_git_commit.MODE_ENV}=on")
            params = {key: value for key, value in args.items() if key != "why"}
            observation = self.observation(name, args, call_id, lambda: self._commit(name, params, call_id))
            if (replay_mode_active() and not observation.startswith("REFUSED")
                    and kernel_git_commit.CommitLog(self.shell.run_dir).by_call(call_id) is None):
                raise ToolReplayRefused(f"{name} record for {call_id} is missing from the commit log")
            return ShellReply(observation)
        if name == "git" and not self.offered(name):
            return ShellReply("REFUSED NOT_A_GIT_REPO: Git is disabled or the source is not a Git repository")
        if name == "web_fetch" and not self.offered(name):
            return ShellReply("REFUSED WEB_HTTP_ERROR: web_fetch is disabled; enable --web")
        def run():
            if name in JOB_NAMES:
                return self.jobs.execute(name, args)
            if name == "git":
                return self._git(args)
            if name == "web_fetch":
                return self._web(args)
            raise ValueError(f"unregistered host tool: {name}")
        observation = self.observation(name, args, call_id, run,
                          definition=self.git.head_oid if name == "git" and self.git else None,
                          capability=self._git_capability() if name == "git" else None)
        if replay_mode_active() and name == "job_stop" and args.get("job") in self.jobs.jobs:
            self.jobs.jobs[args["job"]].state = "stopped"
        return ShellReply(observation)

    def _git_capability(self):
        roots = (*self.shell.capability.read_roots, *self.git.object_dirs, self.git.git_dir)
        return replace(self.shell.capability, purpose="git", workspace_writable=False,
                       network="none", write_roots=(), read_roots=tuple(set(roots)),
                       env={**self.shell.capability.env, **self.git.env()})

    def _git(self, args):
        from kernel_git_view import handle_git_call

        params = {k: v for k, v in args.items() if k != "why"}
        action = params.get("action")
        if action == "show" and isinstance(params.get("rev"), str) and ":" in params["rev"]:
            if "path" in params:
                return "REFUSED BAD_ARGUMENT: supply path once"
            params["rev"], params["path"] = params["rev"].split(":", 1)
        if action == "diff" and "path" in params:
            params["paths"] = [params.pop("path")]
        def runner(argv, env, roots):
            cap = self._git_capability()
            result = self.shell.session.run(argv, cap, cwd=self.shell.root)
            if not result.cleanup.ok:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {result.cleanup.detail}")
            return result.exit_code, result.output
        text, meta = handle_git_call(self.git, runner, params)
        if meta["ok"]:
            from kernel_shell_policy import normalize_output
            return normalize_output(text.encode()) or "No changes."
        detail = meta.get("detail", text)
        if "excluded" in detail:
            code = "EXCLUDED_PATH"
        elif "revision" in detail or "unknown revision" in detail or "ambiguous argument" in detail or "invalid object name" in detail:
            code = "BAD_REVISION"
        elif meta.get("refusal") == "object_unavailable":
            code = "MISSING_OBJECT"
        else:
            code = "BAD_ARGUMENT"
        return f"REFUSED {code}: {detail}"

    def _web(self, args):
        size = args.get("max_chars", 12000)
        if not isinstance(args.get("url"), str) or type(size) is not int or not 1 <= size <= 12000:
            return "REFUSED WEB_HTTP_ERROR: url must be text and max_chars an integer from 1 to 12000"
        from kernel_web_fetch import WebPolicy, fetch

        policy = WebPolicy(tuple(self.shell.settings.web_allow), max_chars=size,
                           deadline=max(0.001, min(20, self.deadline-time.monotonic())))
        result = fetch(args["url"], policy)
        if result.ok and result.status is not None and 200 <= result.status < 300:
            return result.text
        reason = result.refusal or f"HTTP status {result.status}"
        lower = reason.lower()
        if any(word in lower for word in ("private", "non-public", "non-global", "address")):
            code = "WEB_REFUSED_ADDRESS"
        elif "scheme" in lower or "http or https" in lower:
            code = "WEB_REFUSED_SCHEME"
        elif lower.startswith("port ") or "invalid url: port" in lower:
            code = "WEB_REFUSED_PORT"
        elif "domain" in lower or "web allow list" in lower:
            code = "WEB_DOMAIN_NOT_ALLOWED"
        elif any(word in lower for word in ("limit", "too large", "exceeds")):
            code = "WEB_TOO_LARGE"
        elif any(word in lower for word in ("content type", "charset", "encoding")):
            code = "WEB_CONTENT_TYPE"
        elif "timeout" in lower or "timed out" in lower or "deadline" in lower:
            code = "WEB_TIMEOUT"
        else:
            code = "WEB_HTTP_ERROR"
        return f"REFUSED {code}: {reason}"


_active: AgentTools | None = None


def configure(shell, custom, mcp_servers=(), *, resume_active=False,
              defer_schema_budget=False):
    global _active
    candidate = AgentTools(shell, custom, mcp_servers, resume_active=resume_active,
                           defer_schema_budget=defer_schema_budget)
    # Publish only after all definitions and the source view validate. Dynamic
    # dispatch and schema consumers resolve this same per-run object.
    _active = candidate


def service():
    return _active


def is_registered(name):
    return _active is not None and name in _active.names


def project_menu(tools):
    if _active is None:
        return [tool for tool in tools if tool["function"]["name"] not in OBSERVATION_NAMES]
    selected = [tool for tool in tools if tool["function"]["name"] not in _active.names]
    return [*selected, *_active.schemas()]
