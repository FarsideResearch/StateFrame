"""Small shell admission grammar; the execution capability is the boundary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
import shlex
from typing import Mapping

from kernel_tool_settings import ToolSettings


INSPECTION_PROGRAMS = frozenset("""ls cat head tail wc du df stat file tree pwd uname
whoami id date env printenv which ps uptime grep rg find sort uniq cut tr awk sed
diff cmp jq column comm paste basename dirname realpath sha256sum shasum md5sum
base64 xxd hexdump""".split())
DENIED_PROGRAMS = frozenset({"sudo", "doas", "su", "ssh", "scp", "git"})
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "\n"})
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")


class ShellPolicyRefused(ValueError):
    def __init__(self, code: str, reason: str):
        self.code = code
        super().__init__(reason)

    def reply(self) -> str:
        return f"REFUSED {self.code}: {self}"


@dataclass(frozen=True)
class ShellAdmission:
    programs: tuple[str, ...]
    requires_confirmation: bool
    workspace_writable: bool
    network: str


def _segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    try:
        for token in lexer:
            if token in _SEPARATORS or token == "\n":
                if segments[-1]:
                    segments.append([])
            else:
                segments[-1].append(token)
    except ValueError as error:
        raise ShellPolicyRefused("DENIED_PROGRAM", f"invalid shell quoting: {error}") from error
    return [segment for segment in segments if segment]


def _readonly(command: str, segments: list[list[str]]) -> None:
    def refuse(reason: str) -> None:
        raise ShellPolicyRefused("READ_ONLY_POLICY", reason)

    # Reject expansion/control syntax instead of pretending to parse POSIX shell.
    # Quoted occurrences are conservatively refused too.
    if any(char in command for char in "$`(){}") or "\\\n" in command:
        refuse("expansion, grouping and continuation are not inspection syntax")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    tokens = list(lexer)
    for index, token in enumerate(tokens):
        if token in {"|", "&&", ";", "\n"}:
            continue
        if token and all(char in ";&|<>()\n" for char in token):
            if token == ">" and index + 1 < len(tokens) and tokens[index + 1] == "/dev/null":
                continue
            refuse("only inspection pipelines, &&, ; and redirection to /dev/null are allowed")
    for segment in segments:
        program = segment[0]
        name = PurePosixPath(program).name
        if name not in INSPECTION_PROGRAMS or ("/" in program and
                program not in {f"/bin/{name}", f"/usr/bin/{name}"}):
            refuse(f"{program!r} is not an inspection program")
        args = segment[1:]
        if name == "find" and any(arg in {"-exec", "-execdir", "-ok", "-okdir", "-delete",
                                          "-fprint", "-fprint0", "-fprintf", "-fls"} for arg in args):
            refuse("find execution and file-writing actions are not allowed")
        if name == "env" and any(arg not in {"-0", "--null", "-", "--"} for arg in args):
            refuse("env may print the environment, not execute or assign a command")
        if name == "sort" and any(arg.startswith(("--output", "-o", "--compress-program")) for arg in args):
            refuse("sort output files and subprocess options are not allowed")
        if name == "awk" and any(re.search(r"\bsystem\b|\bgetline\b|[|>@]", arg)
                                 or arg.startswith(("-f", "--file", "-E", "--exec")) for arg in args):
            refuse("awk execution and file I/O are not allowed")
        if name == "sed":
            # Inspection address ranges and plain substitutions only. Script files
            # and sed's e/r/w commands can invoke programs or bypass the grammar.
            scripts = [arg for arg in args if arg not in {"-n", "-E", "-e", "--"}]
            if not scripts or not re.fullmatch(r"(?:\d+(?:,\d+)?[pdq]|s/[^/\n]*/[^/\n]*/[gp]*)",
                                               scripts[0]):
                refuse("sed requires an inspection address or substitution without execution flags")
            if any(arg.startswith(("-i", "--in-place", "-f", "--file")) for arg in args):
                refuse("sed in-place and script-file options are not allowed")
            script_index = args.index(scripts[0])
            if any(arg.startswith("-") and arg != "-" for arg in args[script_index + 1:]):
                refuse("sed additional scripts and options after the script are not allowed")
        if name == "rg" and any(arg.startswith(("--pre", "--hostname-bin")) for arg in args):
            refuse("rg subprocess options are not allowed")
        if name == "xxd" and ("-r" in args or "-revert" in args):
            refuse("xxd reverse mode is not an inspection")


def admit_shell(command: object, settings: ToolSettings) -> ShellAdmission:
    """Validate policy and return authority requirements, without executing text."""
    if settings.shell == "off":
        raise ShellPolicyRefused("SHELL_OFF", "shell is disabled by --shell off")
    if not isinstance(command, str) or not command.strip() or "\0" in command or len(command) > 16384:
        code = "READ_ONLY_POLICY" if settings.shell == "read-only" else "DENIED_PROGRAM"
        raise ShellPolicyRefused(code, "command must be nonempty text, at most 16384 characters, without NUL")
    try:
        segments = _segments(command)
    except ShellPolicyRefused as error:
        if settings.shell == "read-only":
            raise ShellPolicyRefused("READ_ONLY_POLICY", str(error)) from error
        raise
    if not segments:
        raise ShellPolicyRefused("DENIED_PROGRAM", "command contains no program")
    programs = tuple(PurePosixPath(segment[0]).name for segment in segments)
    network = "none" if settings.shell == "read-only" else settings.shell_network
    for segment in segments:
        names = {PurePosixPath(word).name for word in segment}
        install = ((bool(names & {"pip", "pip3", "npm", "yarn", "pnpm"}) and
                    bool(names & {"install", "add", "ci"})) or
                   ("uv" in names and bool(names & {"sync", "add", "pip"})))
        if network == "none" and install:
            raise ShellPolicyRefused("NETWORK_OFF", "package installation requires --shell-network full")
    if settings.shell == "read-only":
        _readonly(command, segments)
    elif settings.shell == "standard":
        denied = DENIED_PROGRAMS | set(settings.shell_deny)
        if network == "none":
            denied |= {"curl", "wget", "nc", "netcat"}
        for program in programs:
            if program in denied:
                raise ShellPolicyRefused("DENIED_PROGRAM", f"program {program!r} is denied")
    allowed = INSPECTION_PROGRAMS | set(settings.shell_allow)
    return ShellAdmission(programs, settings.shell_ask and any(p not in allowed for p in programs),
                          settings.shell != "read-only", network)


def normalize_output(raw: bytes, *, replacements: Mapping[str, str] | None = None,
                     total_bytes: int | None = None) -> str:
    """Bound opaque observations without inventing source-read or test evidence."""
    total = len(raw) if total_bytes is None else total_bytes
    if b"\0" in raw:
        return f"[binary output, {total} bytes]"
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"[binary output, {total} bytes]"
    text = _ANSI.sub("", decoded)
    text = "\n".join(line.split("\r")[-1] for line in text.replace("\r\n", "\n").split("\n"))
    for source, replacement in sorted((replacements or {}).items(), key=lambda pair: -len(pair[0])):
        if source:
            text = text.replace(source, replacement)
    encoded = text.encode("utf-8")
    omitted = max(0, total - len(raw)) + max(0, len(encoded) - 12288)
    if not omitted:
        return text
    split = min(6144, (len(encoded) + 1) // 2)
    return (encoded[:split].decode("utf-8", errors="ignore") +
            f"\n[... {omitted} bytes omitted ...]\n" +
            encoded[max(split, len(encoded) - 6144):].decode("utf-8", errors="ignore"))
