"""Host-side validation and management of explicitly selected skill bundles."""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NAME = re.compile(r"^[a-z][a-z0-9-]{1,40}$")
MAX_BODY_BYTES = 32 * 1024
MAX_DIRECTORY_BYTES = 1 << 20
MAX_SKILLS = 16


class SkillError(ValueError):
    """A bundle is missing, malformed, unsafe, or changed."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: str | None
    path: Path
    body: str
    skill_md_sha256: str
    tools_json_sha256: str | None
    tool_names: tuple[str, ...]

    def record(self, *, trusted: bool) -> dict[str, object]:
        return {"name": self.name, "path": str(self.path),
                "skill_md_sha256": self.skill_md_sha256,
                "tools_json_sha256": self.tools_json_sha256,
                "tools_trusted": trusted}


def user_skills_dir() -> Path:
    home = Path(os.environ.get("STATEFRAME_HOME") or Path.home() / ".stateframe").expanduser()
    return home / "skills"


def _canonical_skill_root(path: Path) -> Path:
    """Resolve platform aliases once, before the root is recorded in run settings."""
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise SkillError(f"skill directory missing or linked: {path}: {error}") from error


def _read_skill_file(path: Path, root: Path, *, maximum: int, missing_ok: bool) -> bytes | None:
    """Walk the recorded canonical root from /, then read only below its descriptor."""
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SkillError(f"{path}: expected a file inside the skill directory") from error
    if not root.is_absolute() or ".." in root.parts or not relative.parts or ".." in relative.parts:
        raise SkillError(f"{path}: expected a file inside the skill directory")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    directories: list[int] = []
    try:
        directories.append(os.open("/", directory_flags))
        for component in (*root.parts[1:], *relative.parts[:-1]):
            directories.append(os.open(component, directory_flags, dir_fd=directories[-1]))
        try:
            fd = os.open(relative.parts[-1], file_flags, dir_fd=directories[-1])
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SkillError(f"{path}: expected a regular file inside the skill directory")
            chunks = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == 0:
                raise SkillError(f"{path}: file exceeds {maximum} bytes")
            return b"".join(chunks)
        finally:
            os.close(fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SkillError(f"{path}: symlinks are not allowed") from error
        if error.errno == errno.ENOTDIR:
            raise SkillError(f"{path}: expected a file inside the skill directory (linked directory)") from error
        raise SkillError(f"{path}: cannot read: {error}") from error
    finally:
        for fd in reversed(directories):
            os.close(fd)


def read_skill_file(path: Path, root: Path, *, maximum: int) -> bytes:
    """Read a bounded regular file beneath a previously canonicalized skill root."""
    data = _read_skill_file(path, root, maximum=maximum, missing_ok=False)
    if data is None:
        raise SkillError(f"{path}: expected a regular file inside the skill directory")
    return data


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SkillError(f"tools.json: duplicate JSON key {key}")
        result[key] = value
    return result


def _tool_names(raw: bytes) -> tuple[str, ...]:
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object,
                              parse_constant=_invalid_json_constant)
    except (UnicodeError, ValueError) as error:
        raise SkillError(f"tools.json: invalid JSON: {error}") from error
    if (not isinstance(document, dict) or document.get("schema") != "stateframe-tools/1"
            or not isinstance(document.get("tools"), list)):
        raise SkillError("tools.json: expected stateframe-tools/1 with a tools list")
    names = []
    for item in document["tools"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise SkillError("tools.json: each tool needs a name")
        names.append(item["name"])
    if len(set(names)) != len(names):
        raise SkillError("tools.json: duplicate tool name")
    return tuple(names)


def _invalid_json_constant(value: str) -> object:
    raise SkillError(f"tools.json: non-finite JSON number {value}")


def parse_skill(path: Path) -> Skill:
    directory = Path(path).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise SkillError(f"skill directory missing or linked: {directory}")
    return _parse_skill_at_root(_canonical_skill_root(directory))


def _parse_skill_at_root(directory: Path) -> Skill:
    raw = read_skill_file(directory / "SKILL.md", directory, maximum=MAX_BODY_BYTES + 4096)
    try:
        source = raw.decode("utf-8")
    except UnicodeError as error:
        raise SkillError(f"{directory / 'SKILL.md'}: expected UTF-8") from error
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillError(f"{directory / 'SKILL.md'}: frontmatter must start with ---")
    fields: dict[str, str] = {}
    end = None
    for index, line in enumerate(lines[1:], 1):
        if line.rstrip("\r\n") == "---":
            end = index
            break
        if "\t" in line or ":" not in line:
            raise SkillError(f"{directory / 'SKILL.md'}: malformed frontmatter line {index + 1}")
        key, value = line.rstrip("\r\n").split(":", 1)
        if not key or key.strip() != key or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key):
            raise SkillError(f"{directory / 'SKILL.md'}: malformed frontmatter key on line {index + 1}")
        if key in fields:
            raise SkillError(f"{directory / 'SKILL.md'}: duplicate frontmatter key {key}")
        fields[key] = value.strip()
    if end is None:
        raise SkillError(f"{directory / 'SKILL.md'}: missing closing ---")
    name = fields.get("name", "")
    description = fields.get("description", "")
    if not NAME.fullmatch(name) or name != directory.name:
        raise SkillError(f"{directory / 'SKILL.md'}: name must match directory and ^[a-z][a-z0-9-]{{1,40}}$")
    if not 1 <= len(description) <= 300 or "\n" in description or "\r" in description:
        raise SkillError(f"{directory / 'SKILL.md'}: description must be one line of 1 to 300 characters")
    body = "".join(lines[end + 1:])
    if not body.strip() or len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise SkillError(f"{directory / 'SKILL.md'}: body must be nonempty and at most 32 KiB")
    tools_file = directory / "tools.json"
    tools_raw = _read_skill_file(tools_file, directory, maximum=MAX_DIRECTORY_BYTES, missing_ok=True)
    return Skill(name, description, fields.get("version"), directory, body,
                 hashlib.sha256(raw).hexdigest(),
                 hashlib.sha256(tools_raw).hexdigest() if tools_raw is not None else None,
                 _tool_names(tools_raw) if tools_raw is not None else ())


def resolve_skills(values: Sequence[str], *, trusted: bool) -> tuple[Skill, ...]:
    if len(values) > MAX_SKILLS:
        raise SkillError(f"at most {MAX_SKILLS} skills may be selected")
    selected = []
    names = set()
    for value in values:
        path = Path(value).expanduser()
        if not path.is_dir() and os.sep not in value and (os.altsep is None or os.altsep not in value):
            if not NAME.fullmatch(value):
                raise SkillError(f"invalid skill name: {value}")
            path = user_skills_dir() / value
        skill = parse_skill(path)
        if skill.name in names:
            raise SkillError(f"duplicate selected skill: {skill.name}")
        names.add(skill.name)
        selected.append(skill)
    return tuple(selected)


def skills_digest(records: Sequence[dict[str, object]]) -> str:
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def catalog_lines(skills: Sequence[Skill], records: Sequence[dict[str, object]]) -> tuple[str, ...]:
    return tuple(
        skill.name + ": " + skill.description +
        ((" (commands: " + ", ".join(skill.tool_names) + ")") if record["tools_trusted"] and skill.tool_names
         else " (commands not trusted)" if skill.tools_json_sha256 else "")
        for skill, record in zip(skills, records)
    )


def measured_skills_digest(records: Sequence[dict[str, object]], measurements: dict[str, object]) -> str:
    encoded = json.dumps({"skills": records, "measurements": measurements}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_measurements(measurements: object, names: Sequence[str]) -> dict[str, object]:
    def valid(value: object) -> bool:
        return (isinstance(value, dict) and set(value) == {"tokens", "method"}
                and type(value["tokens"]) is int and value["tokens"] >= 0
                and isinstance(value["method"], str) and value["method"] in
                {"tokenize", "calibrated_bytes", "verified_byte_bound"})
    if (not isinstance(measurements, dict) or set(measurements) != {"catalog", "bodies"}
            or not valid(measurements["catalog"]) or not isinstance(measurements["bodies"], dict)
            or set(measurements["bodies"]) != set(names)
            or any(not valid(item) for item in measurements["bodies"].values())):
        raise SkillError("recorded skill measurements have invalid fields")
    return measurements


def verify_records(records: Sequence[dict[str, object]], digest: str,
                   measurements: dict[str, object] | None = None) -> tuple[Skill, ...]:
    if (not isinstance(records, list) or len(records) > MAX_SKILLS
            or any(not isinstance(record, dict)
                   or set(record) != {"name", "path", "skill_md_sha256", "tools_json_sha256", "tools_trusted"}
                   or not isinstance(record["name"], str) or not NAME.fullmatch(record["name"])
                   or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
                   or not isinstance(record["skill_md_sha256"], str)
                   or re.fullmatch(r"[0-9a-f]{64}", record["skill_md_sha256"]) is None
                   or (record["tools_json_sha256"] is not None
                       and (not isinstance(record["tools_json_sha256"], str)
                            or re.fullmatch(r"[0-9a-f]{64}", record["tools_json_sha256"]) is None))
                   or type(record["tools_trusted"]) is not bool
                   for record in records)):
        raise SkillError("recorded skills have invalid fields")
    if len({record["name"] for record in records}) != len(records):
        raise SkillError("recorded skills have duplicate names")
    if measurements is not None:
        validate_measurements(measurements, [str(record["name"]) for record in records])
    if (measured_skills_digest(records, measurements) if measurements is not None
            else skills_digest(records)) != digest:
        raise SkillError("skill settings digest differs from the run spec")
    loaded = []
    for record in records:
        name = record.get("name")
        path = Path(str(record.get("path", "")))
        if not path.is_dir():
            raise SkillError(f"skill {name} missing")
        try:
            skill = _parse_skill_at_root(path)
        except SkillError as error:
            raise SkillError(f"skill {name} changed since the run started (SKILL.md digest or format: {error})") from error
        if skill.name != name or skill.skill_md_sha256 != record.get("skill_md_sha256"):
            raise SkillError(f"skill {name} changed since the run started (SKILL.md digest mismatch)")
        if skill.tools_json_sha256 != record.get("tools_json_sha256"):
            raise SkillError(f"skill {name} changed since the run started (tools.json digest mismatch)")
        loaded.append(skill)
    return tuple(loaded)


def _check_tree(directory: Path) -> None:
    total = 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        for entry in [*(Path(root) / name for name in dirs), *(Path(root) / name for name in files)]:
            mode = entry.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise SkillError(f"{entry}: symlinks and special files are not allowed")
            if stat.S_ISREG(mode):
                total += entry.stat().st_size
                if total > MAX_DIRECTORY_BYTES:
                    raise SkillError(f"{directory}: directory exceeds 1 MiB")


def install(path: Path, *, force: bool = False) -> Skill:
    skill = parse_skill(path)
    # Installation has no endpoint tokenizer. Use the request meter's conservative
    # initial byte calibration; a run measures again against its actual tokenizer.
    wire = json.dumps({"messages": [{"role": "user", "content": skill.body}], "tools": []}).encode()
    count = math.ceil(len(wire) / 2.46)
    if count > 8000:
        raise SkillError(f"SKILL_TOO_LARGE: skill {skill.name} body has {count} calibrated_bytes tokens; maximum 8000")
    _check_tree(skill.path)
    home = user_skills_dir()
    home.mkdir(parents=True, exist_ok=True)
    destination = home / skill.name
    if destination.is_symlink():
        raise SkillError(f"{destination}: linked installation destination is not allowed")
    if destination.exists() and not force:
        raise SkillError(f"skill {skill.name} is already installed; use --force to replace it")
    temporary = Path(tempfile.mkdtemp(prefix=".skill-install-", dir=home))
    try:
        def copy_file(source: str, target: str) -> str:
            source_path = Path(source)
            limit = (MAX_BODY_BYTES + 4096 if source_path == skill.path / "SKILL.md"
                     else MAX_DIRECTORY_BYTES)
            Path(target).write_bytes(read_skill_file(source_path, skill.path, maximum=limit))
            shutil.copystat(source, target, follow_symlinks=False)
            return target

        shutil.copytree(skill.path, temporary / skill.name, symlinks=True, copy_function=copy_file)
        _check_tree(temporary / skill.name)
        copied = parse_skill(temporary / skill.name)
        if (copied.skill_md_sha256 != skill.skill_md_sha256
                or copied.tools_json_sha256 != skill.tools_json_sha256):
            raise SkillError(f"skill {skill.name} changed during installation")
        if destination.exists():
            shutil.rmtree(destination)
        (temporary / skill.name).replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return parse_skill(destination)


def list_installed() -> tuple[Skill, ...]:
    home = user_skills_dir()
    if not home.exists():
        return ()
    return tuple(parse_skill(path) for path in sorted(home.iterdir()) if path.is_dir())


def remove(name: str) -> None:
    if not NAME.fullmatch(name):
        raise SkillError(f"invalid skill name: {name}")
    directory = user_skills_dir() / name
    if not directory.is_dir() or directory.is_symlink():
        raise SkillError(f"skill {name} missing")
    parse_skill(directory)
    shutil.rmtree(directory)
